"""Protocol and failure checks for opt-in RAPS runtime optimizations."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.cee_us.cee_us_cabinet import rng_state, restore_rng
from exploration.raps import raps_gnn_actor as raps


def assert_tree(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree(left, right)
    else:
        assert actual == expected


def fixture(device="cpu", skill_set="split7", optimized=False):
    torch.set_num_threads(1)
    dimension = 13 if skill_set == "split7" else 12
    backend = CEEUSBackend(device=device, action_dim=dimension, agent_dim=33,
                           hidden_dim=8, horizon=3, candidates=8, iterations=2,
                           elites=4, batch_size=4, runtime_optimizations=optimized)
    actor = raps.RAPSImaginationActor(backend, dict(
        skill_set=skill_set, latched_gripper=True, hidden_size=8,
        actor_layers=1, critic_layers=1, imagination_horizon=3,
        imagination_batch_size=4, updates_per_round=2, target_update_period=1))
    generator = np.random.default_rng(44)
    states = generator.normal(size=(9, 67)).astype(np.float32)
    next_states = states.copy()
    next_states[:, :57] += .01
    actions = raps.canonical_action(torch.tensor(generator.uniform(-1, 1, (9, dimension)),
                                                dtype=torch.float32), actor.schema).numpy()
    data = dict(states=states, next_states=next_states, actions=actions,
                terminated=np.zeros(9, dtype=bool))
    return backend, actor, data


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("override", [None, False, True])
def test_runtime_cli_override(monkeypatch, tmp_path, configured, override):
    from exploration.cee_us import cee_us_cabinet as cli
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(dict(agent="raps_gnn_actor", runtime_optimizations=configured)), path)
    captured = []
    monkeypatch.setattr(raps, "run", lambda args, config: captured.append(config.runtime_optimizations) or 0)
    arguments = ["train", "--config", str(path)]
    if override is not None:
        arguments += ["--runtime-optimizations" if override else "--no-runtime-optimizations"]
    assert cli.main(arguments) == 0
    assert captured == [configured if override is None else override]


@pytest.mark.parametrize("skill_set", ["compact6", "split7"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actor_updates_and_rng_match_reference(device, skill_set):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    backend, actor, data = fixture(device, skill_set)
    backend.train(data, 1)
    initial_actor = deepcopy(actor.state_dict())
    initial_model = deepcopy(backend.state_dict())
    torch.manual_seed(91)
    np.random.seed(91)
    initial_rng = rng_state()
    expected = actor.train(data)
    expected_rng, expected_actor = rng_state(), deepcopy(actor.state_dict())
    actor.load_state_dict(initial_actor)
    backend.runtime_optimizations = True
    restore_rng(initial_rng)
    actual = actor.train(data)
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert_tree(actor.state_dict(), expected_actor)
    assert_tree(rng_state(), expected_rng)
    assert_tree(backend.state_dict(), initial_model)


@pytest.mark.parametrize("skill_set", ["compact6", "split7"])
@pytest.mark.parametrize("execution", ["reference", "cuda_graph"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_full_fit_actor_and_graph_rounds_match(skill_set, execution):
    reference, actor, data = fixture("cuda", skill_set)
    optimized, fast_actor, _ = fixture("cuda", skill_set, optimized=True)
    if execution == "cuda_graph":
        from exploration.cee_us.cee_us_cuda_graph import CudaGraphPrediction
        for model in (reference, optimized):
            model.prediction_accelerator = CudaGraphPrediction(model.model)
            model.model.predict = model.prediction_accelerator.predict
    optimized.load_state_dict(reference.state_dict())
    fast_actor.load_state_dict(deepcopy(actor.state_dict()))
    for round_index in range(2):
        np.random.seed(100 + round_index)
        torch.manual_seed(100 + round_index)
        initial_rng = rng_state()
        expected_fit = reference.train(data, 1)
        expected_actor = actor.train(data)
        expected_rng = rng_state()
        restore_rng(initial_rng)
        actual_fit = optimized.train(data, 1)
        actual_actor = fast_actor.train(data)
        assert actual_fit == pytest.approx(expected_fit, rel=1e-5, abs=1e-6)
        assert actual_actor == pytest.approx(expected_actor, rel=1e-5, abs=1e-6)
        assert_tree(optimized.state_dict(), reference.state_dict())
        assert_tree(fast_actor.state_dict(), actor.state_dict())
        assert_tree(rng_state(), expected_rng)
        if optimized.prediction_accelerator is not None:
            assert optimized.prediction_accelerator._graphs


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("invalid", ["loss", "gradient"])
def test_nonfinite_stops_before_optimizer_update(optimized, invalid):
    backend, actor, _ = fixture(optimized=optimized)
    before = deepcopy(actor.actor.state_dict())
    parameters = list(actor.actor.parameters())
    loss = sum(p.square().sum() for p in parameters)
    if invalid == "loss":
        loss = loss * float("nan")
    else:
        parameters[-1].register_hook(lambda gradient: gradient * float("nan"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        actor._step(actor.actor, actor.actor_optimizer, loss)
    assert_tree(actor.actor.state_dict(), before)
    assert not actor.actor_optimizer.state


def checkpoint_fixture():
    backend, actor, data = fixture()
    backend.train(data, 1)
    actor.train(data)
    rows = [dict(state=data["states"][i], next_state=data["next_states"][i],
                 action=data["actions"][i], episode_id=1, terminated=False, truncated=False,
                 trainable=True, duration_low_steps=2) for i in range(len(data["states"]))]
    ticks = [{**row, "action": np.zeros(4, np.float32)} for row in rows]
    encoder = SimpleNamespace(contract=lambda: {"test": True})
    return backend, actor, encoder, ticks, rows


def test_background_checkpoint_owns_snapshot_and_preserves_rng(monkeypatch, tmp_path):
    backend, actor, encoder, ticks, rows = checkpoint_fixture()
    initial_rng = rng_state()
    reference = tmp_path / "reference.pt"
    raps.save(reference, backend, actor, encoder, OmegaConf.create({}), ticks, rows)
    assert_tree(rng_state(), initial_rng)
    target = tmp_path / "background.pt"
    entered, release = threading.Event(), threading.Event()
    original = raps._write_checkpoint

    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(raps, "_write_checkpoint", blocked)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = raps.save(target, backend, actor, encoder, OmegaConf.create({}), ticks, rows,
                           executor=executor)
        try:
            assert entered.wait(10)
            assert not target.exists()
            with torch.no_grad():
                for p in actor.actor.parameters():
                    p.add_(1)
                for p in backend.model.model.parameters():
                    p.add_(1)
            rows[0]["state"][:] = 10
            assert_tree(rng_state(), initial_rng)
        finally:
            release.set()
        result = future.result(timeout=10)
    assert result["checkpoint_write_s"] > 0
    assert_tree(torch.load(target, weights_only=False), torch.load(reference, weights_only=False))
    for suffix in (".replay.npz", ".low_level.npz"):
        with np.load(target.with_suffix(suffix)) as actual, np.load(reference.with_suffix(suffix)) as expected:
            assert_tree(dict(actual), dict(expected))


@pytest.mark.parametrize("fail_at", ["write", "publish"])
def test_background_checkpoint_failure_never_publishes_marker(monkeypatch, tmp_path, fail_at):
    backend, actor, encoder, ticks, rows = checkpoint_fixture()
    target = tmp_path / "failed.pt"
    if fail_at == "write":
        monkeypatch.setattr(raps.np, "savez_compressed", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    else:
        from pathlib import Path
        original = Path.replace
        def replace(source, destination):
            if destination == target.with_suffix(".low_level.npz"):
                raise OSError("publish failed")
            return original(source, destination)
        monkeypatch.setattr(Path, "replace", replace)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = raps.save(target, backend, actor, encoder, OmegaConf.create({}), ticks, rows,
                           executor=executor)
        with pytest.raises(OSError, match="disk full|publish failed"):
            future.result(timeout=10)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("fail_writer", [False, True])
def test_run_drains_writer_before_completion_and_propagates_errors(monkeypatch, tmp_path, fail_writer):
    from exploration.cee_us import cee_us_state
    backend, actor, encoder, ticks, rows = checkpoint_fixture()
    backend.runtime_optimizations = True
    config = OmegaConf.load(raps.ROOT / "configs/raps_gnn_actor_cabinet_split7.yaml")
    config.device = "cpu"
    config.runtime_optimizations = True
    config.bootstrap_steps = config.steps_per_round = config.rounds = 2
    config.evaluation_steps = 0
    config.raps_actor = OmegaConf.create(actor.settings)
    output = tmp_path / "run"
    closed = []
    env = SimpleNamespace(reset=lambda **kwargs: None, close=lambda: closed.append(True))
    monkeypatch.setattr(raps, "CEEUSBackend", lambda **kwargs: backend)
    monkeypatch.setattr(raps, "RAPSImaginationActor", lambda *args: actor)
    monkeypatch.setattr(raps, "make_env", lambda **kwargs: (env, SimpleNamespace(max_ee_steps_per_episode=800)))
    monkeypatch.setattr(cee_us_state, "CabinetSkillStateEncoder", lambda *args, **kwargs: encoder)

    def prepare(path, *args):
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps(dict(adaptations=[])))

    class Collector:
        def __init__(self, env, encoder, actor, directory):
            self.directory = directory
            directory.mkdir(exist_ok=True)
            self.episode_id = self.skill_id = self.total_steps = 0

        def collect(self, count, policy, **kwargs):
            self.episode_id += 1
            self.skill_id += count
            self.total_steps += count
            return deepcopy(ticks[:count]), deepcopy(rows[:count])

    monkeypatch.setattr(raps, "prepare_output", prepare)
    monkeypatch.setattr(raps, "SkillCollector", Collector)
    monkeypatch.setattr(raps, "fit", lambda *args, **kwargs: dict(completed_skills=2, actor=dict(actor_loss=0)))
    monkeypatch.setattr(raps, "summarize", lambda *args: {})
    original = raps._write_checkpoint
    written = []

    def write(path, *args):
        if fail_writer:
            raise OSError("background disk full")
        result = original(path, *args)
        written.append(path)
        return result

    monkeypatch.setattr(raps, "_write_checkpoint", write)
    arguments = SimpleNamespace(stage="train", output=output, device=None, checkpoint=None)
    if fail_writer:
        with pytest.raises(OSError, match="background disk full"):
            raps.run(arguments, config)
        assert not (output / "result.json").exists()
        assert "background disk full" in (output / "failure.json").read_text()
    else:
        assert raps.run(arguments, config) == 0
        assert len(written) == 3
        assert all(path.is_file() for path in written)
        assert json.loads((output / "result.json").read_text())["status"] == "completed"
        events = [json.loads(line) for line in (output / "timings.jsonl").read_text().splitlines()]
        assert sum(event["stage"] == "checkpoint" for event in events) == 3
    assert closed == [True]
