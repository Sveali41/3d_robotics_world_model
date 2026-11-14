"""Paired reference/device replay tests, including real CUDA when available."""
import gc
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.cee_us.official.mbrl.torch_helpers import TorchTrainingIterator


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


def assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_tree_equal(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('mode', ['bootstrap', 'permutation', 'ordered', 'iterations'])
def test_device_indices_batches_and_rng_match_reference(device, mode, monkeypatch):
    data = np.arange(22, dtype=np.float32).reshape(11, 2)
    uploaded_indices = []
    original = TorchTrainingIterator._device_indices

    def record_indices(self, indices):
        result = original(self, indices)
        if self.runtime_optimizations:
            uploaded_indices.append(result.detach().cpu().numpy().copy())
        return result

    monkeypatch.setattr(TorchTrainingIterator, '_device_indices', record_indices)

    def run(optimized):
        iterator = TorchTrainingIterator(
            {'observations': torch.tensor(data, device=device) if optimized else data},
            ensemble=True, ensemble_size=3, shuffle=mode != 'ordered',
            runtime_optimizations=optimized)
        np.random.seed(871)
        if mode == 'bootstrap':
            batches = iterator.get_epoch_iterator(4, 3)
        elif mode == 'iterations':
            batches = iterator.get_basic_iterator(4, 5)
        else:
            batches = iterator.get_epoch_iterator_non_bootstrapped(4, 3)
        values = [batch['observations'] for batch in batches()]
        return [v.cpu().numpy() if optimized else v for v in values], np.random.random(8)

    expected, expected_rng = run(False)
    actual, actual_rng = run(True)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    np.testing.assert_array_equal(actual_rng, expected_rng)
    assert len(uploaded_indices) == (1 if mode == 'iterations' else 3)
    if mode != 'iterations':
        assert [v.shape[2] for v in actual] == [4, 4, 3] * 3


def replay(size, seed):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(size, 66)).astype(np.float32)
    next_states = states.copy()
    next_states[:, :56] += rng.normal(size=(size, 56)).astype(np.float32) * .1
    return dict(states=states, actions=rng.uniform(-1, 1, (size, 4)).astype(np.float32),
                next_states=next_states)


@pytest.mark.parametrize('device', DEVICES)
def test_repeated_fits_match_and_release_device_data(device, monkeypatch):
    options = dict(hidden_dim=8, ensemble_size=3, batch_size=4, attach_icem=False, device=device)
    reference = CEEUSBackend(**options)
    optimized = CEEUSBackend(**options, runtime_optimizations=True)
    assert not reference.model.runtime_optimizations
    assert optimized.model.runtime_optimizations
    data_refs = []
    shapes = []
    original_init = TorchTrainingIterator.__init__

    def record_init(self, data_dict, **kwargs):
        if kwargs.get('runtime_optimizations'):
            shapes.append({key: value.shape for key, value in data_dict.items()})
            data_refs.extend(weakref.ref(value) for value in data_dict.values())
            assert all(value.device.type == device for value in data_dict.values())
        original_init(self, data_dict, **kwargs)

    monkeypatch.setattr(TorchTrainingIterator, '__init__', record_init)
    for fit, size in enumerate((11, 6)):
        data = replay(size, fit + 100)
        logs = []
        results = []
        rng_states = []
        torch_states = []
        for backend in (reference, optimized):
            logged = []
            backend.model.logger = SimpleNamespace(log=lambda value, key: logged.append((key, value)))
            backend.profile_timings = {}
            np.random.seed(500 + fit)
            torch.manual_seed(600 + fit)
            results.append(backend.train(data, epochs=3))
            logs.append(logged)
            rng_states.append(np.random.random(8))
            torch_states.append(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state())
        assert_tree_equal(optimized.state_dict(), reference.state_dict())
        assert results[1] == pytest.approx(results[0], rel=1e-5, abs=1e-6)
        assert [key for key, _ in logs[1]] == [key for key, _ in logs[0]]
        assert len(logs[1]) == 4  # Existing first-batch boundary plus each epoch end.
        np.testing.assert_allclose([v for _, v in logs[1]], [v for _, v in logs[0]], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(*rng_states)
        assert torch.equal(*torch_states)
        assert optimized.profile_timings['data_upload_s'] >= 0
        assert optimized.profile_timings['model_training_s'] >= 0
        assert optimized.profile_timings['normalization_s'] >= 0
        if device == 'cuda':
            assert optimized.profile_timings['data_upload_gpu_s'] >= 0
        assert 'data_upload_s' not in reference.profile_timings
        assert optimized.model.profile_timings is None
        gc.collect()
        assert all(ref() is None for ref in data_refs)
    assert [entry['observations'][0] for entry in shapes] == [11, 6]
    assert all(set(entry) == {'observations', 'actions', 'next_observations'} for entry in shapes)


def test_optimized_oom_is_actionable_and_never_falls_back(monkeypatch):
    backend = CEEUSBackend(hidden_dim=8, batch_size=4, attach_icem=False, runtime_optimizations=True)
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise torch.cuda.OutOfMemoryError('injected allocation failure')

    monkeypatch.setattr(backend.model, 'loss', fail)
    with pytest.raises(RuntimeError, match='turn off runtime_optimizations'):
        backend.train(replay(6, 4), epochs=1)
    assert calls == [1]
    assert backend.model.profile_timings is None


def test_optimized_nonfinite_gradients_still_fail():
    backend = CEEUSBackend(hidden_dim=8, attach_icem=False, runtime_optimizations=True)
    parameter = next(backend.model.model.parameters())
    parameter.grad = torch.full_like(parameter, float('nan'))
    with pytest.raises(FloatingPointError, match='nonfinite gradients'):
        backend.model.optimizer.step()
    assert backend.updates == 0
