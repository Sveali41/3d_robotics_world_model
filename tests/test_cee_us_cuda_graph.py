"""CUDA graph equivalence checks; CPU environments skip GPU execution."""
import numpy as np
import pytest
import torch
import weakref

from exploration.cee_us_backend import CEEUSBackend
from exploration.cee_us_cuda_graph import CudaGraphPrediction
from third_party.cee_us.mbrl.controllers.abstract_controller import OpenLoopPolicy


def test_cpu_is_rejected():
    backend = CEEUSBackend(hidden_dim=8, horizon=3, candidates=8, elites=4)
    with pytest.raises(ValueError, match='CUDA device'):
        CudaGraphPrediction(backend.model)
    assert not backend.model.model.cache_batch_indices


@pytest.fixture
def backend():
    if not torch.cuda.is_available():
        pytest.skip('CUDA device unavailable')
    return CEEUSBackend(device='cuda', hidden_dim=16, horizon=3, candidates=8,
                        iterations=3, elites=4, batch_size=4)


def inputs(count):
    return torch.randn(5, count, 66, device='cuda'), torch.randn(5, count, 4, device='cuda')


def test_predictions_shapes_and_output_independence(backend):
    graph = CudaGraphPrediction(backend.model)
    graph.profile_timings = {}
    for count in (1, 8, 128, 1):
        obs, actions = inputs(count)
        expected = graph.original_predict(observations=obs, states=None, actions=actions)[0]
        with torch.inference_mode():
            actual, state, reward = graph.predict(observations=obs, states=None, actions=actions)
        assert state is None and reward is None
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual[..., 56:], obs[..., 56:], rtol=0, atol=0)
        saved = actual.clone()
        again = graph.predict(observations=obs + 1, states=None, actions=actions)[0]
        assert actual.data_ptr() != again.data_ptr()
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    assert len(graph._graphs) == 3
    assert len(graph.profile_timings['capture_wall_seconds']) == 3
    assert all(seconds > 0 for seconds in graph.profile_timings['capture_wall_seconds'])


def test_captured_edge_storage_survives_batch_changes(backend):
    graph = CudaGraphPrediction(backend.model)
    obs, actions = inputs(16)
    expected = graph.predict(observations=obs, states=None, actions=actions)[0]
    edges = weakref.ref(backend.model.model.edge_list)
    other_obs, other_actions = inputs(8)
    graph.predict(observations=other_obs, states=None, actions=other_actions)
    # Graph replay uses raw device addresses; the graph object alone does not
    # own this upstream index tensor after another batch replaces it.
    assert edges() is not None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    actual = graph.predict(observations=obs, states=None, actions=actions)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    graph.invalidate()
    assert edges() is None


def test_repeated_full_planner_matches_reference(backend):
    backend = CEEUSBackend(device='cuda')
    state = np.linspace(-.1, .1, 66, dtype=np.float32)
    backend.reset_planner(state)
    snapshot = backend.state_dict()
    np.random.seed(17)
    torch.manual_seed(17)
    expected = [backend.action(state) for _ in range(3)]
    backend.load_state_dict(snapshot)
    graph = CudaGraphPrediction(backend.model)
    backend.model.predict = graph.predict
    np.random.seed(17)
    torch.manual_seed(17)
    actual = [backend.action(state) for _ in range(3)]
    # Elite reuse expands cost buffers, but GNN inference stays at 128 samples.
    assert len(graph._graphs) == 1
    for (action, metrics), (ref_action, ref_metrics) in zip(actual, expected):
        np.testing.assert_allclose(action, ref_action, rtol=1e-5, atol=1e-6)
        assert metrics['curiosity_score'] == pytest.approx(ref_metrics['curiosity_score'], rel=1e-5)


def test_region_projection_cuda_capture_matches_reference(backend):
    projection = dict(root_to_cabinet_rotation=np.eye(3).tolist(),
                      root_to_cabinet_translation=[0., 0., 0.],
                      lower=[.325, -.13, .7], upper=[.625, .15, .96],
                      pos_lower=[-.1]*3, pos_upper=[.1]*3)
    projected = CEEUSBackend(device='cuda', hidden_dim=16, execution='cuda_graph',
                            action_projection=projection)
    for count in (16, 8, 16):
        obs, actions = inputs(count)
        expected = projected._predict_with_region(observations=obs, states=None, actions=actions)[0]
        actual = projected.model.predict(observations=obs, states=None, actions=actions)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rollout_and_fixed_candidate_scores(backend):
    rng = np.random.default_rng(11)
    states = rng.normal(size=(8, 66)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(8, 30, 4)).astype(np.float32)
    expected = backend.rollout(states, actions)
    graph = CudaGraphPrediction(backend.model)
    backend.model.predict = graph.predict
    backend.model.num_simulated_trajectories, backend.model.horizon = 8, 30
    backend.model.preallocate_memory()
    paths, _ = backend.model.predict_n_steps(
        start_observations=torch.tensor(states, device='cuda'), start_states=None,
        horizon=30, policy=OpenLoopPolicy(torch.tensor(actions, device='cuda')))
    actual = paths['next_observations'].permute(1, 0, 2, 3).cpu().numpy().copy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    expected_scores = -torch.as_tensor(expected).std(0).sum((-1, -2))
    actual_scores = -torch.as_tensor(actual).std(0).sum((-1, -2))
    torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-5, atol=1e-6)
    assert actual_scores.argmin() == expected_scores.argmin()
    # The official rollout buffer must retain every old observation.
    obs = torch.tensor(states, device='cuda')[None].expand(5, -1, -1)
    np.testing.assert_allclose(backend.model.all_observations_[0].cpu(), obs.cpu(), rtol=0, atol=0)
    np.testing.assert_allclose(backend.model.all_observations_[1:].cpu(), actual[:, :, :-1].transpose(2, 0, 1, 3), rtol=0, atol=0)


def test_fixed_sampler_action_and_curiosity(backend, monkeypatch):
    state = np.linspace(-1, 1, 66, dtype=np.float32)
    samples = torch.linspace(-1, 1, 8*3*4, device='cuda').reshape(8, 3, 4)
    def sampler(obs, num_traj, time_slice=None):
        selected = samples[:num_traj].clone()
        return selected if time_slice is None else selected[:, time_slice[0]:time_slice[1]]
    monkeypatch.setattr(backend.planner, 'sample_action_sequences', sampler)
    backend.reset_planner(state)
    snapshot = backend.state_dict()
    expected, expected_metrics = backend.action(state)
    backend.load_state_dict(snapshot)
    graph = CudaGraphPrediction(backend.model)
    backend.model.predict = graph.predict
    actual, actual_metrics = backend.action(state)
    np.testing.assert_array_equal(actual, expected)
    assert actual_metrics['curiosity_score'] == pytest.approx(expected_metrics['curiosity_score'], rel=1e-5, abs=1e-6)


def test_invalidate_normalizer_load_and_training(backend):
    backend = CEEUSBackend(device='cuda', execution='cuda_graph', hidden_dim=16,
                          horizon=3, candidates=8, iterations=3, elites=4, batch_size=4)
    graph = backend.prediction_accelerator
    obs, actions = inputs(8)
    before = graph.predict(observations=obs, states=None, actions=actions)[0]
    snapshot = backend.state_dict()
    norm = backend.model.output_agent_normalizer
    norm.update(torch.randn(20, 32, device='cuda') + 4)
    graph.invalidate()
    assert not graph._graphs
    expected = graph.original_predict(observations=obs, states=None, actions=actions)[0]
    after = graph.predict(observations=obs, states=None, actions=actions)[0]
    torch.testing.assert_close(after, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(after, before)
    backend.load_state_dict(snapshot)
    assert not graph._graphs
    torch.testing.assert_close(graph.predict(observations=obs, states=None, actions=actions)[0], before, rtol=1e-5, atol=1e-6)
    train_obs, train_actions = inputs(4)
    with torch.inference_mode():
        graph.predict(observations=train_obs, states=None, actions=train_actions)
    rng = np.random.default_rng(2)
    states = rng.normal(size=(8, 66)).astype(np.float32)
    next_states = states.copy()
    next_states[:, :56] += .1
    metrics = backend.train(dict(states=states, actions=np.zeros((8, 4), np.float32), next_states=next_states), 1)
    assert not graph._graphs
    assert metrics['updates'] == 2 and metrics['finitegrad']
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in backend.model.model.parameters())
    expected = graph.original_predict(observations=obs, states=None, actions=actions)[0]
    torch.testing.assert_close(graph.predict(observations=obs, states=None, actions=actions)[0], expected, rtol=1e-5, atol=1e-6)
