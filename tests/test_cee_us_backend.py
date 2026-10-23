"""Focused CPU checks of the fixed upstream model and iCEM adapter."""
import json

import numpy as np
import pytest
import torch

from exploration.cee_us_backend import CEEUSBackend
from third_party.cee_us.mbrl.controllers.mpc_torch_curiosity import TorchCuriosityMpcICem
from third_party.cee_us.mbrl.rolloutbuffer import SimpleRolloutBuffer


@pytest.fixture
def backend():
    return CEEUSBackend(hidden_dim=8, horizon=3, candidates=8,
                        iterations=3, elites=4, batch_size=4)


def replay():
    rng = np.random.default_rng(11)
    states = rng.normal(size=(8, 66)).astype(np.float32)
    next_states = states.copy()
    next_states[:, :56] += rng.normal(size=(8, 56)).astype(np.float32) * .1
    return dict(states=states, actions=rng.uniform(-1, 1, (8, 4)).astype(np.float32),
                next_states=next_states)


def test_forward_loss_training_and_static_fields(backend):
    data = replay()
    metrics = backend.train(data, epochs=1)
    assert metrics['updates'] == 2 and metrics['finitegrad']
    json.dumps(metrics, allow_nan=False)
    obs = torch.tensor(data['states'])[None].expand(5, -1, -1)
    actions = torch.tensor(data['actions'])[None].expand(5, -1, -1)
    expected = backend.model.predict(observations=obs, actions=actions, states=None)[0]
    predicted = backend.predict(data['states'], data['actions'])
    np.testing.assert_array_equal(predicted, expected.numpy())
    np.testing.assert_array_equal(predicted[:, :, 56:], np.broadcast_to(data['states'][None, :, 56:], (5, 8, 10)))
    batch = dict(observations=obs, actions=actions,
                 next_observations=torch.tensor(data['next_states'])[None].expand(5, -1, -1))
    inputs = backend.model._process_batch(batch)
    outputs = backend.model.model(*inputs[:4])
    errors = torch.cat(((outputs[0]-inputs[4]), (outputs[1]-inputs[5]).flatten(2)), dim=-1)
    torch.testing.assert_close(backend.model.loss(batch), errors.square().mean((1, 2)).sum())
    backend.model.optimizer.zero_grad()
    # One member's outputs must not differentiate any other member's parameters.
    out = backend.model.model(*inputs[:4])
    (out[0][2].sum() + out[1][2].sum()).backward()
    for parameter in backend.model.model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad[[0, 1, 3, 4]]) == 0


def test_object_permutation_equivariance(backend):
    data = replay()
    backend.train(data, 1)
    permutation = np.r_[np.arange(32), np.arange(44, 56), np.arange(32, 44),
                        np.arange(61, 66), np.arange(56, 61)]
    original = backend.predict(data['states'], data['actions'])
    permuted = backend.predict(data['states'][:, permutation], data['actions'])
    np.testing.assert_allclose(permuted, original[:, :, permutation], rtol=2e-5, atol=2e-6)


def test_execution_validation_and_loading_weights_without_planner(backend):
    with pytest.raises(ValueError, match='requires a CUDA device'):
        CEEUSBackend(device='cpu', execution='cuda_graph')
    with pytest.raises(ValueError, match='execution must be'):
        CEEUSBackend(execution='invalid')
    data = replay()
    backend.train(data, 1)
    backend.reset_planner(data['states'][0])
    backend.random_action(data['states'][0])
    snapshot = backend.state_dict()
    assert snapshot['random_sequence'] is not None
    expected = backend.predict(data['states'], data['actions'])
    backend.load_state_dict(snapshot, restore_planner=False)
    assert backend.planner.elite_samples is None and backend.random_sequence is None
    assert not backend.planner.was_reset
    np.testing.assert_array_equal(backend.predict(data['states'], data['actions']), expected)


def test_original_curiosity_and_distribution_update(backend):
    p = backend.planner
    states = torch.randn(8, 5, 3, 66)
    states[..., 56:] = 2.
    actions = torch.rand(8, 1, 3, 4).expand(-1, 5, -1, -1)
    paths = SimpleRolloutBuffer(next_observations=states, actions=actions, observations=states)
    out = torch.empty(8, 5)
    p.rollout_seconds = 0.
    p.trajectory_cost_fn(p.cost_fn, paths, out)
    expected = -states.std(dim=1, correction=1).sum((-1, -2))
    torch.testing.assert_close(out, expected[:, None].expand(-1, 5))
    before_mean, before_std = p.mean_.clone(), p.std_.clone()
    costs = torch.arange(8).flip(0).float()
    elite_actions = actions[torch.argsort(costs)[:4]]
    expected_mean = .1 * before_mean + .9 * elite_actions.mean((0, 1))
    expected_std = .1 * before_std + .9 * elite_actions.std((0, 1), correction=1)
    p.update_distributions(paths, costs)
    torch.testing.assert_close(p.mean_, expected_mean)
    torch.testing.assert_close(p.std_, expected_std)


def test_action_choice_matches_official_path_fixed_samples(backend, monkeypatch):
    state = replay()['states'][0]
    backend.reset_planner(state)
    samples = torch.linspace(-1, 1, 8*3*4).reshape(8, 3, 4)
    def fixed_sampler(obs, num_traj, time_slice=None):
        selected = samples[:num_traj].clone()
        return selected if time_slice is None else selected[:, time_slice[0]:time_slice[1]]
    monkeypatch.setattr(backend.planner, 'sample_action_sequences', fixed_sampler)
    snapshot = backend.state_dict()
    backend.planner.rollout_seconds = 0.
    official = TorchCuriosityMpcICem.get_action(backend.planner, state, state=None)
    backend.load_state_dict(snapshot)
    action, diagnostics = backend.action(state)
    np.testing.assert_array_equal(action, official)
    np.testing.assert_array_equal(action, backend.planner.elite_samples['actions'][0, 0, 0].numpy())
    assert action.dtype == np.float32 and np.all(np.abs(action) <= 1)
    assert sum(diagnostics['score_decomposition'].values()) == pytest.approx(diagnostics['curiosity_score'], rel=1e-6)
    json.dumps(diagnostics, allow_nan=False)


def test_rollout_member_history_and_checkpoint(backend):
    data = replay()
    backend.train(data, 1)
    state = data['states'][0]
    backend.reset_planner(state)
    backend.action(state)
    backend.random_action(state)
    snapshot = backend.state_dict()
    rng = torch.get_rng_state()
    planned, _ = backend.action(state)
    random = backend.random_action(state)
    backend.load_state_dict(snapshot)
    torch.set_rng_state(rng)
    np.testing.assert_array_equal(backend.action(state)[0], planned)
    np.testing.assert_array_equal(backend.random_action(state), random)
    actions = np.zeros((2, 3, 4), dtype=np.float32)
    rolled = backend.rollout(data['states'][:2], actions)
    obs = torch.tensor(data['states'][:2])[None].expand(5, -1, -1)
    expected = []
    for step in range(3):
        obs = backend.model.predict(observations=obs, states=None,
            actions=torch.tensor(actions[:, step])[None].expand(5, -1, -1))[0]
        expected.append(obs.numpy().copy())
    np.testing.assert_array_equal(rolled, np.stack(expected, axis=2))
    # The retained random sequence is shifted; a fresh sample supplies only its tail.
    previous = backend.random_sequence.clone()
    action = backend.random_action(state)
    np.testing.assert_array_equal(action, previous[0].numpy())
    torch.testing.assert_close(backend.random_sequence[:-1], previous[1:])
    backend.reset_planner(state)
    assert backend.random_sequence is None and backend.planner.elite_samples is None


def test_nonfinite_input_fails(backend):
    state = np.zeros(66, np.float32)
    state[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        backend.action(state)
    data = replay()
    data['next_states'][0, -1] += 1
    with pytest.raises(ValueError, match='static'):
        backend.train(data, 1)


def test_disk_checkpoint_preserves_optimizer_and_normalizers(backend, tmp_path):
    data = replay()
    backend.train(data, 1)
    backend.reset_planner(data['states'][0])
    backend.action(data['states'][0])
    backend.random_action(data['states'][0])
    path = tmp_path / 'backend.pt'
    torch.save(backend.state_dict(), path)
    snapshot = torch.load(path, map_location='cpu', weights_only=False)
    numpy_rng = np.random.get_state()
    first_metrics = backend.train(data, 1)
    expected = backend.predict(data['states'], data['actions'])
    backend.load_state_dict(snapshot)
    np.random.set_state(numpy_rng)
    assert backend.train(data, 1) == first_metrics
    np.testing.assert_array_equal(backend.predict(data['states'], data['actions']), expected)
    assert backend.planner.mean_.device == backend.device
    assert backend.planner.elite_samples['actions'].device == backend.device
    for name in backend.normalizer_names:
        assert getattr(backend.model, name).mean.device == backend.device


@pytest.fixture
def projection():
    return dict(root_to_cabinet_rotation=[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                root_to_cabinet_translation=[.25, -.1, .3],
                lower=[-.1, -.2, -.3], upper=[.1, .2, .3],
                pos_lower=[-.2, -.3, -.4], pos_upper=[.2, .3, .4])


def manual_projection(observations, actions, projection):
    rotation = np.array(projection['root_to_cabinet_rotation'])
    translation = np.array(projection['root_to_cabinet_translation'])
    low, high = np.array(projection['pos_lower']), np.array(projection['pos_upper'])
    position = observations[..., 18:21]
    delta = low + (actions[..., :3] + 1) / 2 * (high - low)
    local = np.clip((position + delta) @ rotation.T + translation,
                    projection['lower'], projection['upper'])
    target = (local - translation) @ np.linalg.inv(rotation).T
    xyz = np.clip(2 * (target - position - low) / (high - low) - 1, -1, 1)
    return np.concatenate((xyz, np.broadcast_to(actions[..., 3:], (*xyz.shape[:-1], 1))), axis=-1).astype(np.float32)


def test_projection_disabled_is_original_prediction(backend):
    data = replay()
    obs, actions = torch.tensor(data['states']), torch.tensor(data['actions'])
    assert backend.project_actions(obs, actions) is actions
    assert backend.model.predict == backend._original_predict
    expected = backend._original_predict(observations=obs[None].expand(5, -1, -1),
                                         actions=actions[None].expand(5, -1, -1), states=None)[0]
    np.testing.assert_array_equal(backend.predict(data['states'], data['actions']), expected.numpy())


def test_projection_rotated_frame_batches_and_repeated_projection(projection):
    projected = CEEUSBackend(hidden_dim=8, action_projection=projection)
    obs = torch.zeros(5, 2, 66)
    # These different cabinet-local positions are inside the box.
    local = np.array([[[.08 * (member - 2) / 2, .15 * (batch * 2 - 1), .1]
                       for batch in range(2)] for member in range(5)])
    rotation = np.array(projection['root_to_cabinet_rotation'])
    obs[..., 18:21] = torch.tensor((local - projection['root_to_cabinet_translation']) @ rotation)
    actions = torch.tensor([[[1., -1., 1., -.75], [-1., 1., -1., .25]]]).expand(5, -1, -1)
    actual = projected.project_actions(obs, actions)
    expected = manual_projection(obs.numpy(), actions.numpy(), projection)
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-7)
    torch.testing.assert_close(actual[..., 3], actions[..., 3], rtol=0, atol=0)
    assert not torch.equal(actual[0, :, :3], actual[-1, :, :3])
    torch.testing.assert_close(projected.project_actions(obs, actual), actual, atol=3e-7, rtol=1e-6)
    target = obs[..., 18:21] + torch.tensor(projection['pos_lower']) + (actual[..., :3] + 1) / 2 * (
        torch.tensor(projection['pos_upper']) - torch.tensor(projection['pos_lower']))
    local_target = target @ torch.tensor(projection['root_to_cabinet_rotation']).T + torch.tensor(projection['root_to_cabinet_translation'])
    assert torch.all(local_target <= torch.tensor(projection['upper']) + 1e-7)
    assert torch.all(local_target >= torch.tensor(projection['lower']) - 1e-7)


def test_projection_checkpoint_compatibility(backend, projection):
    legacy = backend.state_dict()
    assert legacy.pop('action_projection') is None
    backend.load_state_dict(legacy)
    projected = CEEUSBackend(hidden_dim=8, action_projection=projection)
    saved = projected.state_dict()
    assert saved['action_projection'] == projection
    projected.load_state_dict(saved)
    with pytest.raises(ValueError, match='action_projection'):
        projected.load_state_dict(legacy)
    with pytest.raises(ValueError, match='action_projection'):
        backend.load_state_dict(saved, restore_planner=False)
    saved['action_projection']['upper'][0] += .1
    with pytest.raises(ValueError, match='action_projection'):
        projected.load_state_dict(saved)


def test_projected_prediction_uses_each_predicted_position(projection):
    backend = CEEUSBackend(hidden_dim=8, horizon=3, candidates=8,
                          iterations=3, elites=4, action_projection=projection)
    data = replay()
    actions = np.ones((2, 3, 4), dtype=np.float32)
    actions[..., 3] = -.25
    obs = torch.tensor(data['states'][:2])[None].expand(5, -1, -1)
    for step in range(3):
        projected_actions = manual_projection(obs.numpy(), actions[:, step], projection)
        expected = backend._original_predict(observations=obs, states=None,
                                             actions=torch.tensor(projected_actions))[0]
        obs = backend.model.predict(observations=obs, states=None,
                                    actions=torch.tensor(actions[:, step])[None].expand(5, -1, -1))[0]
        torch.testing.assert_close(obs, expected, rtol=2e-5, atol=2e-6)


def test_offline_projected_rollout_keeps_executed_actions(projection):
    backend = CEEUSBackend(hidden_dim=8, horizon=3, candidates=8,
                          iterations=3, elites=4, action_projection=projection)
    data = replay()
    actions = np.ones((2, 3, 4), dtype=np.float32)
    predictor = backend.model.predict
    rolled = backend.rollout(data['states'][:2], actions)
    assert backend.model.predict == predictor
    obs = torch.tensor(data['states'][:2])[None].expand(5, -1, -1)
    expected = []
    for step in range(3):
        obs = backend._original_predict(observations=obs, states=None,
                                        actions=torch.tensor(actions[:, step])[None].expand(5, -1, -1))[0]
        expected.append(obs.numpy().copy())
    np.testing.assert_array_equal(rolled, np.stack(expected, axis=2))
