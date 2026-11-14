from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from domain.robocasa.primitives import RAPS_SPLIT_ACTION_SCHEMA
from exploration.jepa.replay import VisualReplay, canonical
from exploration.jepa.loop import Learner, read_checkpoint, run_loop
from exploration.jepa.model import StochasticJEPA
from exploration.jepa.planner import MixedCEM


def tiny():
    return dict(image_size=16, patch_size=8, embed_dim=12, encoder_depth=1, encoder_heads=3,
                encoder_mlp_dim=24, predictor_depth=1, predictor_heads=2, predictor_dim_head=6,
                predictor_mlp_dim=24, projector_dim=24, dropout=0., flow_steps=2, sigreg_projections=8)


def frame(value):
    return np.full((16, 16, 3), value % 256, np.uint8)


def skill(episode, index, completed=True):
    action = np.r_[1., np.zeros(12)].astype(np.float32)
    return dict(episode_id=episode, skill_id=index, action=action, requested_action=action,
                duration_low_steps=1, completed=completed, status="failed" if completed else "budget_truncated",
                terminated=False, truncated=False)


def test_replay_masks_failure_retention_and_no_privileged_fields(tmp_path):
    replay = VisualReplay(tmp_path, validation_every=2)
    replay.reset(1, frame(0))
    for i in range(4):
        replay.append({**skill(1, i + 1), "state": [999], "contact": True}, frame(i + 1))
    windows = replay.windows("train")
    assert len(windows) == 2
    batch = replay.batch(windows)
    assert set(batch) == {"pixels", "history_actions", "future_actions", "history_mask", "future_mask"}
    assert batch["history_mask"].tolist() == [[False, False, True], [False, True, True]]
    assert batch["pixels"][0, :, 0, 0, 0].tolist() == [0, 0, 0, 1, 2, 3]
    assert batch["history_actions"][0].count_nonzero() == 0
    assert all(row["status"] == "failed" for row in replay.rows)
    assert "state" not in replay.rows[0]
    assert np.array_equal(replay.read_frame(replay.rows[0]["start"]), frame(0))
    assert len(replay.windows("validation")) == 0


def test_replay_never_crosses_reset_incomplete_or_terminal(tmp_path):
    replay = VisualReplay(tmp_path, validation_every=2)
    replay.reset(1, frame(0))
    replay.append(skill(1, 1), frame(1))
    replay.append(skill(1, 2, False), frame(2))
    replay.append(skill(1, 3), frame(3))
    assert not replay.windows("train")
    replay.reset(2, frame(20))
    for i in range(3):
        replay.append(skill(2, 4 + i), frame(21 + i))
    assert len(replay.windows("validation")) == 1
    assert not replay.windows("train")
    replay.rows[-2]["terminated"] = True
    assert not replay.windows("validation")


def test_frame_checksum_and_action_contract(tmp_path):
    replay = VisualReplay(tmp_path, validation_every=2)
    replay.reset(1, frame(0)); replay.append(skill(1, 1), frame(1))
    snapshot = replay.snapshot()
    VisualReplay.verify_snapshot(snapshot)
    np.save(replay.rows[0]["start"]["path"], frame(9), allow_pickle=False)
    with pytest.raises(ValueError, match="checksum"):
        VisualReplay.verify_snapshot(snapshot)
    with pytest.raises(ValueError, match="bounds"):
        canonical(np.full(13, 2.))
    action = np.ones(13)
    action[:7] = 0; action[6] = 1
    actual = canonical(action)
    used = RAPS_SPLIT_ACTION_SCHEMA.primitive_name_to_action_idx[RAPS_SPLIT_ACTION_SCHEMA.primitive_names[6]]
    assert np.count_nonzero(actual[7:]) == len(used)


class FakeCollector:
    """Exercises production replay, model, CEM, fitting and checkpoint orchestration."""
    def __init__(self, replay, planner):
        self.replay, self.planner = replay, planner
        self.episode_id = self.skill_id = self.total_steps = 0
        self.schema = RAPS_SPLIT_ACTION_SCHEMA
        self.actor = SimpleNamespace(audits=[])

    def collect(self, count, policy, *, seed, stage):
        ticks, skills = [], []
        for i in range(count):
            if i % 4 == 0:
                self.episode_id += 1
                self.replay.reset(self.episode_id, frame(seed + i))
            self.skill_id += 1
            row = skill(self.episode_id, self.skill_id)
            if policy == "curiosity":
                action, audit = self.planner.plan(self.replay.context())
                row.update(action=action, requested_action=action)
                self.actor.audits.append(audit)
            self.replay.append(row, frame(seed + i + 1))
            skills.append(row)
            ticks.append({})
        self.total_steps += count
        return ticks, skills


def setup_loop(output):
    model = StochasticJEPA(tiny())
    planner = MixedCEM(model, dict(candidates=4, iterations=1, elites=2, num_futures=2, chunk_size=2))
    config = dict(seed=7, epochs=1, batch_size=4, bootstrap_steps=8, steps_per_round=8,
                  rounds=2, total_controls=24, model=model.config, planner=planner.config,
                  collection_policy="curiosity")
    replay = VisualReplay(output, validation_every=2)
    return config, Learner(model, config), planner, replay, FakeCollector(replay, planner)


def test_production_loop_two_rounds_and_exact_boundary_resume(monkeypatch, tmp_path):
    import exploration.raps.raps_gnn_actor as diagnostics
    monkeypatch.setattr(diagnostics, "summarize", lambda *args: {"diagnostic_only": True})
    torch.set_num_threads(1)
    torch.manual_seed(10)
    output = tmp_path / "full"
    config, learner, planner, replay, collector = setup_loop(output)
    before = deepcopy(learner.model.state_dict())
    result = run_loop(config, output, learner, planner, replay, collector)
    assert result["real_controls"] == 24 and len(result["history"]) == 3
    assert result["updates"] > 0 and result["replay_transitions"] == 24
    assert all(row["training"]["validation"]["windows"] >= 2 for row in result["history"])
    assert result["history"][1]["planning"]
    assert any(not torch.equal(value, learner.model.state_dict()[key]) for key, value in before.items())
    saved = read_checkpoint(output / "round_000.pt", config)
    new = tmp_path / "resumed"
    _, resumed_learner, resumed_planner, _, _ = setup_loop(tmp_path / "unused")
    resumed_replay = VisualReplay(new, validation_every=2, rows=saved["replay"]["rows"])
    resumed_collector = FakeCollector(resumed_replay, resumed_planner)
    actual = run_loop(config, new, resumed_learner, resumed_planner, resumed_replay, resumed_collector, resumed=saved)
    assert actual["real_controls"] == 24
    assert len(actual["history"]) == 3
    assert len((new / "training.jsonl").read_text().splitlines()) == 3
    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, resumed_learner.model.state_dict()[key], rtol=0, atol=0)
    assert planner.state_dict() == resumed_planner.state_dict()
    torch.testing.assert_close(learner.optimizer.state_dict(), resumed_learner.optimizer.state_dict(),
                               rtol=0, atol=0)
    expected_boundary = read_checkpoint(output / "round_001.pt", config)
    actual_boundary = read_checkpoint(new / "round_001.pt", config)
    for name in ("updates", "controls", "next_round", "episode_id", "skill_id"):
        assert expected_boundary[name] == actual_boundary[name]
    assert expected_boundary["rng"]["python"] == actual_boundary["rng"]["python"]
    np.testing.assert_equal(expected_boundary["rng"]["numpy"], actual_boundary["rng"]["numpy"])
    for name in ("torch", "cuda"):
        torch.testing.assert_close(expected_boundary["rng"][name], actual_boundary["rng"][name], rtol=0, atol=0)
    assert len(actual_boundary["replay"]["rows"]) == len(expected_boundary["replay"]["rows"])
    for expected, actual_row in zip(expected_boundary["replay"]["rows"], actual_boundary["replay"]["rows"]):
        assert {key: value for key, value in expected.items() if key not in ("start", "end")} == {
            key: value for key, value in actual_row.items() if key not in ("start", "end")}
        assert expected["start"]["sha256"] == actual_row["start"]["sha256"]
        assert expected["end"]["sha256"] == actual_row["end"]["sha256"]
    with pytest.raises(ValueError, match="config"):
        read_checkpoint(output / "round_000.pt", {**config, "seed": 8})


def test_insufficient_windows_fails_clearly(tmp_path):
    config, learner, _, replay, _ = setup_loop(tmp_path)
    with pytest.raises(ValueError, match="Insufficient visual training windows"):
        learner.fit(replay)


def test_cli_routes_branch_and_requires_training_budget(monkeypatch, tmp_path):
    from exploration.cee_us.cee_us_cabinet import main, ROOT
    from exploration.jepa import cabinet
    seen = []
    monkeypatch.setattr(cabinet, "run", lambda args, config: seen.append((args, config)) or 0)
    args = ["train", "--config", str(ROOT / "configs/epic_jepa_cabinet_split7.yaml"), "--device", "cpu"]
    assert main(args) == 0
    captured, config = seen[0]
    with pytest.raises(ValueError, match="explicit --total-controls"):
        cabinet.resolve(config, captured)
    captured.stage = "smoke"
    resolved = cabinet.resolve(config, captured)
    assert resolved["total_controls"] == 2000
    assert resolved["objective"] == "predictive_variance"


def test_visual_policy_ignores_diagnostic_state(tmp_path):
    from exploration.jepa.collector import VisualPolicy
    replay = VisualReplay(tmp_path)
    replay.reset(1, frame(0))
    received = []

    class Planner:
        def plan(self, context):
            received.append(context)
            return skill(1, 1)["action"], {"score": 1.}

    class ForbiddenState:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Privileged state was read")

    policy = VisualPolicy(replay, Planner(), dict(skill_set="split7"))
    policy.action(ForbiddenState())
    assert set(received[0]) == {"pixels", "history_actions", "history_mask"}
