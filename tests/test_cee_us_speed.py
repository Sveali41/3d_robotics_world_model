"""Speed CLI tests use saved arrays and stub planners; no physics or CUDA work."""
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from exploration import cee_us_speed as speed


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.pt"
    np.savez(path.with_suffix(".replay.npz"), states=np.arange(20 * 66, dtype=np.float32).reshape(20, 66))
    torch.save({"schema": "cabinet-cee-us-state-v1", "cee_us_commit": speed.REFERENCE_COMMIT,
                "backend": {"weights": "fixed"}, "rng": speed.rng_state(),
                "state_contract": {"layout": 66}, "replay_path": "model.replay.npz"}, path)
    return path


@pytest.fixture
def stubs(monkeypatch):
    backends = []

    class Backend:
        def __init__(self, **kwargs):
            self.kwargs, self.resets, self.actions = kwargs, [], []
            backends.append(self)

        def load_state_dict(self, payload, *, restore_planner):
            assert payload == {"weights": "fixed"}
            assert restore_planner is False

        def reset_planner(self, state):
            print("upstream reset banner")
            self.resets.append(state.copy())

        def action(self, state):
            self.actions.append(state.copy())
            return np.zeros(4), {"planning_seconds": .01, "candidate_rollout_seconds": .008}

    def prepare(output, config, env_config):
        output.mkdir()

    monkeypatch.setattr(speed, "make_backend", Backend)
    monkeypatch.setattr(speed, "prepare_output", prepare)
    return backends


def test_latency_summary():
    assert speed.latency_summary([1, 2, 3])["mean"] == 2
    assert speed.latency_summary([1, 2, 3])["p95"] == pytest.approx(2.9)
    with pytest.raises(ValueError):
        speed.latency_summary([])


def test_benchmark_saved_states_and_cpu_modes_no_physics(checkpoint, tmp_path, monkeypatch, stubs, capsys):
    monkeypatch.setattr(speed, "make_env", lambda: pytest.fail("benchmark must not construct physics"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    payload = speed.load_checkpoint(checkpoint)
    payload["rng"]["cuda"] = [torch.zeros(4, dtype=torch.uint8)]
    torch.save(payload, checkpoint)
    output = tmp_path / "bench"
    assert speed.main(["benchmark", "--checkpoint", str(checkpoint), "--output", str(output),
                       "--device", "cpu", "--states", "12", "--warmup", "2"]) == 0
    import json
    result = json.loads((output / "benchmark.json").read_text())
    assert result["real_steps"] == result["training_updates"] == 0
    assert result["state_indices"] == np.linspace(0, 19, 12, dtype=int).tolist()
    assert len(result["checkpoint_sha256"]) == 64
    assert [backend.kwargs["candidates"] for backend in stubs] == [128, 128]
    assert [entry["threads"] for entry in result["modes"]] == [1, 4]
    for backend, entry in zip(stubs, result["modes"]):
        assert len(backend.resets) == len(backend.actions) == 14
        assert len(entry["full_action_seconds"]["samples"]) == 12
        assert len(entry["warmup_including_capture_seconds"]) == 2
        assert entry["first_decision_seconds"] == entry["warmup_including_capture_seconds"][0]
        assert entry["candidate_rollout_seconds"]["mean"] == pytest.approx(.008)
        assert "Host dispatch" in entry["candidate_rollout_timing"]
        np.testing.assert_array_equal(backend.actions[2], np.arange(66, dtype=np.float32))
    assert "upstream reset banner" not in capsys.readouterr().out
    assert len((output / "benchmark.csv").read_text().splitlines()) == 3


def test_cli_missing_checkpoint_existing_output_and_explicit_cuda(checkpoint, tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        speed.main(["benchmark", "--checkpoint", str(tmp_path / "missing"), "--output", str(tmp_path / "new")])
    with pytest.raises(FileExistsError):
        speed.main(["benchmark", "--checkpoint", str(checkpoint), "--output", str(tmp_path)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        speed.main(["benchmark", "--checkpoint", str(checkpoint), "--output", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()


def test_compare_exact_budget_fresh_environments_and_identical_rng(checkpoint, tmp_path, monkeypatch, stubs):
    environments, calls = [], []

    class Env:
        def __init__(self):
            self.seeds, self.closed = [], False
            environments.append(self)

        def reset(self, *, seed):
            self.seeds.append(seed)

        def close(self):
            self.closed = True

    class Collector:
        def __init__(self, env, backend, encoder, directory):
            self.env, self.backend = env, backend
            self.total_steps = 0
            directory.mkdir()

        def collect(self, count, policy, *, seed, stage):
            self.env.reset(seed=seed)
            calls.append((count, policy, seed, np.random.rand(), torch.rand(1).item()))
            self.total_steps = count
            return [{"planner": {"planning_seconds": .01}} for _ in range(count)]

    monkeypatch.setattr(speed, "make_env", lambda: (Env(), OmegaConf.create({})))
    monkeypatch.setattr(speed, "CabinetStateEncoder", lambda env: SimpleNamespace(contract=lambda: {"layout": 66}))
    monkeypatch.setattr(speed, "Collector", Collector)
    monkeypatch.setattr(speed, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(speed, "interaction_summary", lambda rows: {
        "low_level_steps": len(rows), "recoverable_coverage_auc": 1.,
        "recoverable_motion_primitives_per_1000_steps": 2.,
        "per_object": {name: {"recoverable_motion_primitives": 1}
                       for name in ("cabinet_item_16", "cabinet_item_17")}})
    output = tmp_path / "compare"
    speed.main(["compare", "--checkpoint", str(checkpoint), "--output", str(output), "--steps", "5"])
    import json
    result = json.loads((output / "comparison.json").read_text())
    assert result["total_real_steps"] == 10
    assert result["training_updates"] == 0
    assert len(environments) == len(stubs) == 2
    assert all(env.closed and env.seeds == [0, 0] for env in environments)
    assert calls[0] == calls[1]
    assert [backend.kwargs for backend in stubs] == [
        {"device": "cuda", "seed": 0, "execution": "cuda_graph", "candidates": value}
        for value in (128, 64)]
    assert result["screen"]["both_objects_moved_in_recovery_range"]
    assert result["screen"]["status"] == "not_passed"  # Equal metrics are not improvement.
    assert result["quality_not_degraded"] is True
    assert len((output / "comparison.csv").read_text().splitlines()) == 3


def test_compare_rejects_state_contract_before_collection(checkpoint, tmp_path, monkeypatch):
    closed = []
    env = SimpleNamespace(reset=lambda **kwargs: None, close=lambda: closed.append(True))
    monkeypatch.setattr(speed, "make_env", lambda: (env, OmegaConf.create({})))
    monkeypatch.setattr(speed, "CabinetStateEncoder", lambda env: SimpleNamespace(contract=lambda: {"layout": 99}))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    args = SimpleNamespace(checkpoint=checkpoint, output=tmp_path / "out")
    with pytest.raises(ValueError, match="state contract"):
        speed.compare(args, speed.load_checkpoint(checkpoint), OmegaConf.create({}))
    assert closed == [True]
    assert not args.output.exists()


def test_benchmark_all_modes_and_failure_report(checkpoint, tmp_path, monkeypatch, stubs):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(speed, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 456)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "mock CUDA")
    output = tmp_path / "all_modes"
    speed.main(["benchmark", "--checkpoint", str(checkpoint), "--output", str(output),
                "--states", "2", "--warmup", "1"])
    assert [(b.kwargs["device"], b.kwargs["execution"], b.kwargs["candidates"]) for b in stubs] == [
        ("cpu", "reference", 128), ("cpu", "reference", 128), ("cuda", "reference", 128),
        ("cuda", "cuda_graph", 128), ("cuda", "cuda_graph", 64)]

    def fail(**kwargs):
        raise RuntimeError("capture broke")

    monkeypatch.setattr(speed, "make_backend", fail)
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="capture broke"):
        speed.main(["benchmark", "--checkpoint", str(checkpoint), "--output", str(failed),
                    "--states", "2", "--warmup", "1"])
    assert "capture broke" in (failed / "failure.json").read_text()


def test_checkpoint_backend_optional_contracts(checkpoint, tmp_path, stubs):
    payload = speed.load_checkpoint(checkpoint)
    payload["backend"].update(object_heuristic={"weight": 1}, action_projection={"lower": [0]})
    # Construction options propagate even when a lightweight loader handles the payload.
    assert speed.checkpoint_backend_options(payload) == {
        "object_heuristic": {"weight": 1}, "action_projection": {"lower": [0]}}
    assert speed.checkpoint_backend_options({"backend": {}}) == {}


@pytest.fixture
def profile_stubs(checkpoint, monkeypatch, stubs):
    from exploration import cee_us_cabinet as cabinet
    states = np.arange(20 * 66, dtype=np.float32).reshape(20, 66)
    np.savez(checkpoint.with_suffix(".replay.npz"), states=states, actions=np.zeros((20, 4), np.float32),
             next_states=states + 1, episode_ids=np.ones(20, np.int64),
             terminated=np.zeros(20, bool), truncated=np.zeros(20, bool))
    backends, training, saves = [], [], []

    class Backend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.value = torch.tensor([0.])
            self.optimizer = torch.optim.Adam([self.value.requires_grad_()], lr=.01)
            backends.append(self)

        def load_state_dict(self, payload, *, restore_planner):
            assert not restore_planner
            assert payload == {"weights": "fixed"}

        def train(self, arrays, epochs):
            assert epochs == 1
            assert len(arrays["states"]) == 20
            random_value = np.random.rand() + torch.rand(1).item()
            training.append((self.kwargs["runtime_optimizations"], random_value, self.optimizer))
            self.optimizer.zero_grad()
            loss = (self.value - random_value).square().sum()
            loss.backward()
            self.optimizer.step()
            self.profile_timings.update(normalization_s=.01, model_training_s=.02)
            return {"loss": loss.item(), "updates": 1}

        def state_dict(self):
            return {"model": {"weight": self.value.detach().clone()},
                    "normalizers": {"mean": torch.tensor([1.])},
                    "optimizer": self.optimizer.state_dict()}

        def reset_planner(self, state):
            pass

        def action(self, state):
            return np.full(4, self.value.item()), {}

    def save(path, backend, encoder, config, rows, state, stage, *, arrays, timings):
        assert path != checkpoint
        assert path.parent.name == "scratch"
        assert len(rows) == 20 and stage == "profile"
        assert (arrays is not None) == backend.kwargs["runtime_optimizations"]
        timings.update(array_build_s=.001, replay_save_s=.002, checkpoint_save_s=.003)
        saves.append(path)

    monkeypatch.setattr(speed, "make_backend", Backend)
    monkeypatch.setattr(cabinet, "save_checkpoint", save)
    return backends, training, saves


def test_profile_isolated_training_parity_and_no_default_physics(checkpoint, tmp_path, monkeypatch, profile_stubs):
    import json
    monkeypatch.setattr(speed, "make_env", lambda **kwargs: pytest.fail("profile default must not construct physics"))
    source = checkpoint.read_bytes()
    output = tmp_path / "profile"
    assert speed.main(["profile", "--checkpoint", str(checkpoint), "--output", str(output),
                       "--device", "cpu"]) == 0
    result = json.loads((output / "profile.json").read_text())
    backends, training, saves = profile_stubs
    assert [value[0] for value in training] == [False, True, True, False, False, True]
    assert len({id(value[2]) for value in training}) == 6
    assert len({value[1] for value in training}) == 1
    assert all(b.kwargs["execution"] == "reference" for b in backends)
    assert result["passed"] and all(check["passed"] for check in result["parity"])
    assert result["real_steps"] == 0 and result["physical"] is None
    assert result["training_updates"] == 6
    assert len(saves) == 6
    assert all(len(row["steady_decision_s"]["samples"]) == 12 for row in result["modes"])
    assert all(len(row["warmup_including_capture_s"]) == 2 for row in result["modes"])
    assert len((output / "profile.csv").read_text().splitlines()) == 7
    assert checkpoint.read_bytes() == source


def test_profile_reports_numerical_mismatch(checkpoint, tmp_path, monkeypatch, profile_stubs):
    import json
    factory = speed.make_backend

    def mismatched(**kwargs):
        backend = factory(**kwargs)
        original = backend.state_dict
        def state():
            result = original()
            if kwargs["runtime_optimizations"]:
                result["model"]["weight"] += 1
            return result
        backend.state_dict = state
        return backend
    monkeypatch.setattr(speed, "make_backend", mismatched)
    output = tmp_path / "mismatch"
    assert speed.main(["profile", "--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu"]) == 1
    result = json.loads((output / "profile.json").read_text())
    assert not result["passed"]
    assert all("root.model.weight" in check["mismatches"] for check in result["parity"])


def test_profile_rejects_unbounded_physics(checkpoint, tmp_path):
    with pytest.raises(SystemExit):
        speed.main(["profile", "--checkpoint", str(checkpoint), "--output", str(tmp_path / "out"),
                    "--physical-steps", "65"])


def test_profile_physical_exact_budget_contract_and_paired_rng(checkpoint, tmp_path, monkeypatch, profile_stubs):
    environments, calls, constructed = [], [], []
    class Env:
        def reset(self, **kwargs):
            pass
        def close(self):
            self.closed = True
    def make_env(**kwargs):
        assert kwargs == {"project_goals_to_region": True}
        env = Env()
        environments.append(env)
        return env, OmegaConf.create({})
    class Collector:
        def __init__(self, env, backend, encoder, directory, **kwargs):
            assert kwargs == {"runtime_optimizations": backend.kwargs["runtime_optimizations"], "profile": True}
            constructed.append(backend)
            self.total_steps = 0
            self.timings = {"simulation_s": .01, "json_write_s": .001}
        def collect(self, count, policy, *, seed, stage):
            calls.append((count, policy, seed, np.random.rand(), torch.rand(1).item()))
            self.total_steps = count
            return [{"state": np.zeros(66), "next_state": np.ones(66),
                     "action": np.zeros(4), "requested_action": np.zeros(4)} for _ in range(count)]
    monkeypatch.setattr(speed, "make_env", make_env)
    monkeypatch.setattr(speed, "Collector", Collector)
    monkeypatch.setattr(speed, "CabinetStateEncoder", lambda env: SimpleNamespace(contract=lambda: {"layout": 66}))
    payload = speed.load_checkpoint(checkpoint)
    payload["config"] = {"project_goals_to_region": True}
    args = SimpleNamespace(device="cpu", physical_steps=3, output=tmp_path)
    result = speed.profile_physical(args, payload, payload["rng"])
    assert result["passed"] and result["model_frozen"] and result["training_updates"] == 0
    assert calls[0] == calls[1]
    assert calls[0][:3] == (3, "curiosity", 0)
    assert all(env.closed for env in environments)
    assert [backend.kwargs["runtime_optimizations"] for backend in constructed] == [False, True]


def test_profile_recommendation_requires_parity_and_round_improvement():
    rows = [{"mode": mode, "training_s": value, "array_build_s": value,
             "save_wall_s": value, "profiled_round_s": 2 * value}
            for mode, value in (("original", 1.), ("optimized", .9)) for _ in range(3)]
    summary = speed.profile_summary(rows, True)
    assert summary["recommendation"] == "enable_runtime_optimizations"
    assert summary["modes"]["original"]["training_s"] == {"median": 1., "p95": 1.}
    assert summary["ratios"]["profiled_round_s"]["median_improvement_fraction"] == pytest.approx(.1)
    assert speed.profile_summary(rows, False)["recommendation"] == "retain_original"
    for row in rows:
        row["profiled_round_s"] = 1.
    assert speed.profile_summary(rows, True)["recommendation"] == "retain_original"
