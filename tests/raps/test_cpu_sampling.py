import json
import random
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exploration.raps import cpu_sampling as sampling


def result(stream, count, seed=10):
    ticks = [dict(step=i + 1, episode_id=1, skill_id=i + 1, reward=i)
             for i in range(count)]
    skills = [dict(skill_id=i + 1, episode_id=1, start_step=i, end_step=i + 1)
              for i in range(count)]
    return dict(stream_id=stream, stream_seed=seed, ticks=ticks, skills=skills, timings={"reset_s": 0.5})


def test_merge_canonical_order_and_offsets():
    ticks, skills, steps, episodes, skill_id, timings = sampling._merge(
        [result(1, 2, 12), result(0, 3, 11)], {0: 3, 1: 2}, 10, 4, 8)
    assert [r["step"] for r in ticks] == [11, 12, 13, 14, 15]
    assert [r["stream_id"] for r in ticks] == [0, 0, 0, 1, 1]
    assert [r["episode_id"] for r in ticks] == [5, 5, 5, 6, 6]
    assert [r["skill_id"] for r in ticks] == [9, 10, 11, 12, 13]
    assert [(r["start_step"], r["end_step"]) for r in skills] == [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]
    assert (steps, episodes, skill_id, timings) == (15, 6, 13, {"reset_s": 1.0})
    with pytest.raises(RuntimeError, match="expected 3"):
        sampling._merge([result(0, 2)], {0: 3}, 0, 0, 0)
    with pytest.raises(RuntimeError, match="every requested stream"):
        sampling._merge([result(0, 1)], {0: 1, 1: 1}, 0, 0, 0)


class FakePool:
    jobs = []
    fail = False
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def submit(self, fn, job):
        self.jobs.append(job)
        random.random(); np.random.rand(); torch.rand(1)
        f = Future()
        if self.fail and job["stream_id"] == 1:
            f.set_exception(ValueError("shard error"))
        else:
            f.set_result(result(job["stream_id"], job["count"], job["stream_seed"]))
        return f


def collector(tmp_path, workers=1):
    backend = SimpleNamespace(state_dict=lambda: {}, _move_tensors=lambda x, device: x)
    actor = SimpleNamespace(backend=backend, settings={}, state_dict=lambda: {})
    return sampling.CpuRoundCollector(dict(seed=5, cpu_sampling=dict(workers=workers, streams=2)), actor, tmp_path)


def test_round_jobs_rng_logs_and_fresh_stage(tmp_path, monkeypatch):
    FakePool.jobs = []
    monkeypatch.setattr(sampling, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(sampling, "as_completed", lambda fs: reversed(list(fs)))
    c = collector(tmp_path)
    c.total_steps, c.episode_id, c.skill_id = 7, 2, 3
    random.seed(15); np.random.seed(15); torch.manual_seed(15)
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    ticks, skills = c.collect(5, "random", seed=20, stage="round_0")
    assert [j["count"] for j in FakePool.jobs] == [3, 2]
    assert [j["stream_seed"] for j in FakePool.jobs] == [sampling._stream_seed(5, 20, i) for i in range(2)]
    assert random.getstate() == rng[0]
    assert np.array_equal(np.random.get_state()[1], rng[1][1])
    assert torch.equal(torch.get_rng_state(), rng[2])
    assert (c.total_steps, c.episode_id, c.skill_id) == (12, 4, 8)
    assert [json.loads(line) for line in (tmp_path / "transitions.jsonl").read_text().splitlines()] == ticks
    assert [json.loads(line) for line in (tmp_path / "skills.jsonl").read_text().splitlines()] == skills
    with pytest.raises(FileExistsError):
        c.collect(5, "random", seed=20, stage="round_0")
    c2 = collector(tmp_path / "other", workers=2)
    c2.collect(1, "random", seed=20, stage="round_0")
    assert FakePool.jobs[-1]["stream_seed"] == FakePool.jobs[0]["stream_seed"]
    assert FakePool.jobs[-1]["count"] == 1


def test_failure_no_canonical_output_or_counter_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(sampling, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(FakePool, "fail", True)
    c = collector(tmp_path)
    with pytest.raises(RuntimeError, match="stream 1 failed"):
        c.collect(2, "curiosity", seed=3, stage="failed")
    assert c.total_steps == 0 and not (tmp_path / "transitions.jsonl").exists()


def test_contract_validation():
    assert sampling.sampling_contract({}) is None
    assert sampling.sampling_contract({"cpu_sampling": {"workers": 1, "streams": 2}})["protocol"] == sampling.PROTOCOL
    for setting in ({"workers": 0, "streams": 2}, {"workers": True, "streams": 2},
                    {"workers": 1, "streams": 1}, {"workers": 1, "streams": 2, "other": 1}):
        with pytest.raises(ValueError):
            sampling.sampling_contract({"cpu_sampling": setting})


def test_resume_keeps_sampling_protocol_and_round_boundary(tmp_path):
    from omegaconf import OmegaConf
    from exploration.raps import raps_gnn_actor as raps

    config = OmegaConf.create(dict(seed=5, raps_actor={}, cpu_sampling=dict(workers=2, streams=2),
        bootstrap_steps=2, steps_per_round=2, rounds=1, evaluation_steps=0))
    directory = tmp_path / 'curiosity'
    bootstrap = tmp_path / 'bootstrap'
    directory.mkdir(); bootstrap.mkdir()
    ticks, skills = [], []
    for index, (folder, stage) in enumerate(((bootstrap, 'bootstrap'), (directory, 'round_1'))):
        state, action = [0.] * 50, [0.] * 31
        local_ticks = [dict(step=2*index+i+1, stage=stage, episode_id=index+1,
                            skill_id=index+1, state=state, next_state=state, action=action,
                            terminated=False, truncated=False) for i in range(2)]
        skill = dict(stage=stage, episode_id=index+1, skill_id=index+1,
                     start_step=2*index, end_step=2*index+2, duration_low_steps=2,
                     trainable=True, state=state, next_state=state, action=action,
                     terminated=False, truncated=False)
        (folder / 'transitions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in local_ticks))
        (folder / 'skills.jsonl').write_text(json.dumps(skill)+'\n')
        ticks.extend(local_ticks); skills.append(skill)
    path = directory / 'round_001.pt'
    torch.save(dict(schema=raps.SCHEMA, config=OmegaConf.to_container(config), real_steps=4), path)
    np.savez(path.with_suffix('.replay.npz'), **raps.skill_arrays(skills))
    config.rounds = 2
    restored_ticks, restored_skills, next_round = raps.resume_history(path, config)
    assert restored_ticks == ticks and restored_skills == skills and next_round == 2
    config.cpu_sampling.workers = 1
    with pytest.raises(ValueError, match='only rounds and evaluation_steps'):
        raps.resume_history(path, config)


def test_merge_preserves_zero_step_failure_and_budget_truncation():
    shard = dict(stream_id=0, stream_seed=10, timings={},
        ticks=[dict(step=1, episode_id=1, skill_id=2), dict(step=2, episode_id=1, skill_id=3)],
        skills=[dict(skill_id=1, episode_id=1, start_step=0, end_step=0,
                     duration_low_steps=0, trainable=False, status='path_collision'),
                dict(skill_id=2, episode_id=1, start_step=0, end_step=1,
                     duration_low_steps=1, trainable=True, status='tracking_blocked'),
                dict(skill_id=3, episode_id=1, start_step=1, end_step=2,
                     duration_low_steps=1, trainable=False, status='budget_truncated')])
    ticks, skills, steps, episodes, skill_id, _ = sampling._merge([shard], {0: 2}, 5, 3, 7)
    assert [row['step'] for row in ticks] == [6, 7]
    assert [row['skill_id'] for row in ticks] == [9, 10]
    assert [(row['start_step'], row['end_step']) for row in skills] == [(5, 5), (5, 6), (6, 7)]
    assert [row['duration_low_steps'] for row in skills] == [0, 1, 1]
    assert [row['trainable'] for row in skills] == [False, True, False]
    assert [row['status'] for row in skills] == ['path_collision', 'tracking_blocked', 'budget_truncated']
    assert (steps, episodes, skill_id) == (7, 4, 10)
