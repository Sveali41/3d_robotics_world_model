"""Paired runtime checks for the current four-dimensional grasp-only preset."""
from copy import deepcopy
import random

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.cee_us.cee_us_cabinet import rng_state, restore_rng
from exploration.cee_us.official.mbrl.torch_helpers import TorchTrainingIterator
from exploration.raps import raps_gnn_actor as raps
from tests.raps import test_runtime_optimizations as runtime


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


def grasp_fixture(device="cpu", optimized=False, execution="reference"):
    torch.set_num_threads(1)
    config = OmegaConf.load(raps.ROOT / "configs/raps_gnn_actor_cabinet_grasp_only.yaml")
    settings = OmegaConf.to_container(config.raps_actor, resolve=True)
    settings.update(hidden_size=8, actor_layers=1, critic_layers=1,
                    imagination_horizon=2, imagination_batch_size=4,
                    updates_per_round=2, target_update_period=1)
    backend = CEEUSBackend(device=device, agent_dim=33, action_dim=4, hidden_dim=8,
                           horizon=3, candidates=8, iterations=2, elites=4, batch_size=4,
                           object_heuristic=OmegaConf.to_container(config.object_heuristic),
                           runtime_optimizations=optimized, execution=execution)
    actor = raps.RAPSImaginationActor(backend, settings)
    generator = np.random.default_rng(44)
    states = generator.normal(size=(9, 67)).astype(np.float32)
    next_states = states.copy()
    next_states[:, :backend.dynamic_dim] += .01
    actions = raps.canonical_action(torch.tensor(generator.uniform(-1, 1, (9, 4)),
                                                dtype=torch.float32), actor.schema).numpy()
    data = dict(states=states, next_states=next_states, actions=actions,
                terminated=np.zeros(9, dtype=bool))
    return backend, actor, data


def assert_exact(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_exact(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("device,execution", [("cpu", "reference"),
    pytest.param("cuda", "reference", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable")),
    pytest.param("cuda", "cuda_graph", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_grasp_consecutive_fits_actor_sampling_and_rng_match(device, execution, monkeypatch):
    reference, actor, data = grasp_fixture(device, execution=execution)
    optimized, fast_actor, _ = grasp_fixture(device, optimized=True, execution=execution)
    optimized.load_state_dict(reference.state_dict())
    fast_actor.load_state_dict(deepcopy(actor.state_dict()))
    assert actor.schema.action_dim == 4 and actor.schema.primitive_names == ("grasp",)
    trace = {}
    original_indices = TorchTrainingIterator._device_indices
    original_randint = torch.randint

    def indices(self, values):
        trace["ensemble_indices"].append(np.asarray(values).copy())
        return original_indices(self, values)

    def randint(*args, **kwargs):
        result = original_randint(*args, **kwargs)
        trace["actor_indices"].append(result.detach().cpu().clone())
        return result

    monkeypatch.setattr(TorchTrainingIterator, "_device_indices", indices)
    monkeypatch.setattr(torch, "randint", randint)
    for backend in (reference, optimized):
        original_loss = backend.model.loss
        def loss(batch, original_loss=original_loss):
            trace["batches"].append({key: value.detach().cpu().numpy().copy()
                                    if torch.is_tensor(value) else value.copy()
                                    for key, value in batch.items()})
            return original_loss(batch)
        monkeypatch.setattr(backend.model, "loss", loss)
    for round_index, count in enumerate((9, 6)):
        replay = {key: value[:count] for key, value in data.items()}
        random.seed(100 + round_index)
        np.random.seed(100 + round_index)
        torch.manual_seed(100 + round_index)
        initial_rng = rng_state()
        results, traces, rngs = [], [], []
        for backend, policy in ((reference, actor), (optimized, fast_actor)):
            restore_rng(initial_rng)
            trace = dict(ensemble_indices=[], actor_indices=[], batches=[])
            graph = backend.prediction_accelerator
            if graph is not None and round_index:
                assert graph._graphs
            fit_metrics = backend.train(replay, 2)
            if graph is not None:
                assert not graph._graphs
            actor_metrics = policy.train(replay)
            if graph is not None:
                assert graph._graphs
            results.append((fit_metrics, actor_metrics))
            traces.append(trace)
            rngs.append(rng_state())
        assert results[1][0] == pytest.approx(results[0][0], rel=1e-5, abs=1e-6)
        assert results[1][1] == pytest.approx(results[0][1], rel=1e-5, abs=1e-6)
        assert results[0][0]["updates"] == 2 * ((count + 3) // 4)
        assert [batch["observations"].shape[-2] for batch in traces[0]["batches"]] == (
            [4, 4, 1] * 2 if count == 9 else [4, 2] * 2)
        assert len(traces[0]["ensemble_indices"]) == 2
        assert len(traces[0]["actor_indices"]) == 2
        assert_exact(traces[1], traces[0])
        assert_exact(rngs[1], rngs[0])
        runtime.assert_tree(optimized.state_dict(), reference.state_dict())
        runtime.assert_tree(fast_actor.state_dict(), actor.state_dict())
        assert all(parameter.dtype == torch.float32 for parameter in optimized.model.model.parameters())
        assert np.all(replay["actions"][:, 0] == 1.)
    if optimized.prediction_accelerator is not None:
        optimized.load_state_dict(optimized.state_dict())
        assert not optimized.prediction_accelerator._graphs
        fast_actor.imagine(torch.as_tensor(data["states"][:4], device=device))
        assert optimized.prediction_accelerator._graphs


@pytest.mark.parametrize("device", DEVICES)
def test_grasp_background_snapshot_and_replay_match_sync(device, monkeypatch, tmp_path):
    # Reuse the shared blocked-writer test with the actual 4D actor and trained Adam state.
    monkeypatch.setattr(runtime, "fixture", lambda: grasp_fixture(device))
    original = raps._write_checkpoint
    original_load = torch.load

    def load_cpu(*args, **kwargs):
        return original_load(*args, map_location="cpu", **kwargs)

    monkeypatch.setattr(torch, "load", load_cpu)

    def assert_cpu(path, payload, *args):
        def visit(value):
            if torch.is_tensor(value):
                assert value.device.type == "cpu"
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
        if path.name == "background.pt":
            visit(payload)
        assert payload["action_contract"]["action_dim"] == 4
        return original(path, payload, *args)

    monkeypatch.setattr(raps, "_write_checkpoint", assert_cpu)
    runtime.test_background_checkpoint_owns_snapshot_and_preserves_rng(monkeypatch, tmp_path)


@pytest.mark.parametrize("fail_at", ["write", "publish"])
def test_grasp_background_failure_propagates_without_checkpoint(fail_at, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "fixture", grasp_fixture)
    runtime.test_background_checkpoint_failure_never_publishes_marker(monkeypatch, tmp_path, fail_at)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("invalid", ["loss", "gradient"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_grasp_nonfinite_actor_stops_before_step(optimized, invalid, bad):
    _, actor, _ = grasp_fixture(optimized=optimized)
    before = deepcopy(actor.actor.state_dict())
    parameters = list(actor.actor.parameters())
    loss = sum(parameter.square().sum() for parameter in parameters)
    if invalid == "loss":
        loss = loss * bad
    else:
        parameters[-1].register_hook(lambda gradient: gradient * bad)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        actor._step(actor.actor, actor.actor_optimizer, loss)
    runtime.assert_tree(actor.actor.state_dict(), before)
    assert not actor.actor_optimizer.state
