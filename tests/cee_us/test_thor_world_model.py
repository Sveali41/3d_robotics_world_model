import copy
import numpy as np
import torch

from exploration.cee_us.thor_state import ACTIONS, CAPABILITIES, DYNAMIC_FLAGS, ThorState, execute
from exploration.cee_us.thor_cem import ThorCEM
from exploration.cee_us.cee_us_backend import CEEUSBackend


def metadata():
    def item(name, x, **cap):
        obj = {k: False for k in (*CAPABILITIES, *DYNAMIC_FLAGS)}
        obj.update(objectId=name, objectType='Cup', position={'x': x, 'y': 0, 'z': 0},
                   rotation={'x': 0, 'y': 0, 'z': 0}, openness=0., fillLiquid=None,
                   temperature='RoomTemp', parentReceptacles=None, mass=1.,
                   axisAlignedBoundingBox={'size': {'x': 1, 'y': 1, 'z': 1}}, visible=True)
        obj.update(cap)
        return obj
    return {'agent': {'position': {'x': 0, 'y': 0, 'z': 0}, 'rotation': {'y': 0},
                      'cameraHorizon': 0, 'isStanding': True}, 'lastActionSuccess': True,
            'objects': [item('z', 1, pickupable=True, receptacle=True),
                        item('a', 1, pickupable=True, receptacle=True)]}


def test_target_order_capability_and_held():
    m = metadata(); encoder = ThorState(m); state = encoder.encode(m)
    assert encoder.ids == ('a', 'z')
    assert int(encoder.target_indices(state, ACTIONS.index('Pickup'))) == 0
    m['objects'].reverse()
    np.testing.assert_array_equal(state, encoder.encode(m))
    m['objects'][0]['isPickedUp'] = True
    state = encoder.encode(m)
    assert int(encoder.target_indices(state, ACTIONS.index('Put'))) == 1
    for obj in m['objects']:
        obj['visible'] = False
    state = encoder.encode(m)
    next_state, action, success, target, reason, called = execute(None, encoder, state, ACTIONS.index('Pickup'))
    assert not success and target is None and not called and reason == 'no_visible_capable_target'
    assert next_state[8] == 0 and np.array_equal(next_state[9:], state[9:])
    assert action[len(ACTIONS):].sum() == 0


def test_put_calls_simulator_even_empty_hand():
    m = metadata(); encoder = ThorState(m); state = encoder.encode(m)
    class Env:
        def step(self, action, **kwargs):
            assert action == 'PutObject'
            assert kwargs == {'objectId': 'a', 'forceAction': False, 'placeStationary': True}
            result = copy.deepcopy(m); result.update(lastActionSuccess=False, errorMessage='empty hand')
            return type('Event', (), {'metadata': result})()
    result = execute(Env(), encoder, state, ACTIONS.index('Put'))
    assert not result[2] and result[3] == 'a' and result[5]


def test_cem_initial_score_and_restore():
    m = metadata(); encoder = ThorState(m); state = encoder.encode(m)
    backend = CEEUSBackend(agent_dim=encoder.agent_dim, object_dyn_dim=encoder.object_dyn_dim,
                           object_stat_dim=encoder.object_stat_dim, num_objects=encoder.n,
                           object_names=encoder.ids, action_dim=len(ACTIONS) + encoder.n,
                           hidden_dim=8, batch_size=2, attach_icem=False)
    actions = np.stack([encoder.action_encoding(torch.from_numpy(state), i)[0].numpy() for i in (0, 1)])
    replay = {'states': np.stack([state, state]), 'actions': actions, 'next_states': np.stack([state, state])}
    backend.train(replay, 1)
    planner = ThorCEM(backend, encoder, horizon=2, candidates=3, iterations=1, elites=2)
    scores = planner.score(state, [[0, 1], [8, 9]])
    assert scores.shape == (2,) and np.isfinite(scores).all()
    assert backend.model.num_simulated_trajectories is None
    action, score = planner.plan(state)
    assert 0 <= action < len(ACTIONS) and np.isfinite(score)


def test_targets_resolve_independently_per_predicted_member():
    m = metadata(); encoder = ThorState(m); state = encoder.encode(m)
    states = torch.from_numpy(np.stack([state, state]))
    states[0, encoder.agent_dim + 9] = 0
    states[1, encoder.agent_dim + encoder.object_dyn_dim + 9] = 0
    encoded, targets = encoder.action_encoding(states, ACTIONS.index('Pickup'))
    assert targets.tolist() == [1, 0]
    assert encoded[:, len(ACTIONS):].tolist() == [[0, 1], [1, 0]]


def test_cem_returns_best_scored_sequence_across_iterations():
    encoder = ThorState(metadata())
    planner = ThorCEM(None, encoder, horizon=2, candidates=20, iterations=3, elites=4, seed=4)
    planner.score = lambda _state, sequences: np.asarray(sequences[:, 0], dtype=float)
    action, score = planner.plan(encoder.encode(metadata()))
    assert action == int(score) == len(ACTIONS) - 1


def _small_backend(encoder):
    backend = CEEUSBackend(agent_dim=encoder.agent_dim, object_dyn_dim=encoder.object_dyn_dim,
                           object_stat_dim=encoder.object_stat_dim, num_objects=encoder.n,
                           object_names=encoder.ids, action_dim=len(ACTIONS) + encoder.n,
                           hidden_dim=8, batch_size=2, attach_icem=False)
    state = encoder.encode(metadata())
    obs = np.stack([state, state])
    actions = np.stack([encoder.action_encoding(torch.from_numpy(state), i)[0].numpy() for i in (0, 8)])
    nxt = obs.copy(); nxt[1, 0] += .25
    backend.train({'states': obs, 'actions': actions, 'next_states': nxt}, 1)
    return backend, state


def test_cem_score_matches_manual_member_rollout_for_single_and_chunked_candidates():
    encoder = ThorState(metadata()); backend, state = _small_backend(encoder)
    sequences = np.array([[8, 9, 1], [0, 8, 4], [9, 1, 8]])
    with torch.no_grad():
        obs = torch.from_numpy(state)[None, None, :].expand(backend.ensemble_size, len(sequences), -1)
        predictions = []
        for t in range(sequences.shape[1]):
            action, _ = encoder.action_encoding(obs, torch.from_numpy(sequences[:, t]))
            obs = backend.model.predict(observations=obs, states=None, actions=action)[0]
            predictions.append(obs)
        expected = torch.stack(predictions, dim=2).std(dim=0, unbiased=True).sum(dim=(-1, -2)).numpy()
    for chunk in (1, 2):
        planner = ThorCEM(backend, encoder, horizon=3, candidates=3, iterations=1,
                          elites=2, chunk_size=chunk)
        np.testing.assert_allclose(planner.score(state, sequences), expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(planner.score(state, sequences[:1]), expected[:1], rtol=1e-5, atol=1e-6)


def test_cem_reuses_rollout_buffers_and_handles_remainder(monkeypatch):
    encoder = ThorState(metadata()); backend, state = _small_backend(encoder)
    sequences = np.array([[8, 9, 1], [0, 8, 4], [9, 1, 8], [2, 3, 0], [4, 5, 0]])
    expected = ThorCEM(backend, encoder, chunk_size=1).score(state, sequences)
    allocations = []
    allocate = backend.model.preallocate_memory
    def record():
        allocations.append((backend.model.num_simulated_trajectories, backend.model.horizon))
        allocate()
    monkeypatch.setattr(backend.model, 'preallocate_memory', record)
    actual = ThorCEM(backend, encoder, chunk_size=2).score(state, sequences)
    assert allocations == [(2, 3), (1, 3)]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert backend.model.num_simulated_trajectories is None
    assert backend.model.horizon is None
    assert not backend.model.memory_is_preallocated


def test_thor_state_original_gnn_mse_reduction():
    encoder = ThorState(metadata()); backend, state = _small_backend(encoder)
    obs = torch.from_numpy(np.stack([state, state]))[None].expand(backend.ensemble_size, -1, -1)
    nxt = obs.clone(); nxt[:, 1, 0] += .25
    action = torch.stack([encoder.action_encoding(obs[:, i], i)[0] for i in range(2)], dim=1)
    batch = {'observations': obs, 'next_observations': nxt, 'actions': action}
    model = backend.model
    inputs = model._process_batch(batch)
    agent_pred, obj_pred = model.model.forward(*inputs[:4])
    predicted = torch.cat((agent_pred, obj_pred.flatten(start_dim=2)), dim=-1)
    expected = torch.cat((inputs[4], inputs[5].flatten(start_dim=2)), dim=-1)
    direct = ((predicted - expected) ** 2).mean(dim=(1, 2)).sum()
    torch.testing.assert_close(model.loss(batch), direct)


def test_cem_probability_updates_use_elite_frequencies():
    encoder = ThorState(metadata())
    planner = ThorCEM(None, encoder, horizon=2, candidates=4, iterations=2, elites=2)
    class FixedSampler:
        def __init__(self):
            self.probabilities = []
        def choice(self, count, size, p):
            self.probabilities.append(p.copy())
            return np.array([[0, 1, 1, 2], [1, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]][len(self.probabilities) - 1])
    sampler = FixedSampler(); planner.rng = sampler
    planner.score = lambda _state, _sequences: np.arange(4.)
    planner.plan(encoder.encode(metadata()))
    expected0 = .1 * planner.prior.copy(); expected0[[1, 2]] += .45
    expected1 = .1 * planner.prior.copy(); expected1[1] += .9
    np.testing.assert_allclose(sampler.probabilities[2], expected0)
    np.testing.assert_allclose(sampler.probabilities[3], expected1)


def test_encoder_rejects_changed_identity_missing_required_field_and_unknown_categories():
    import pytest
    original = metadata(); encoder = ThorState(original)
    changed = copy.deepcopy(original); changed['objects'].pop()
    with pytest.raises(ValueError, match='object ID set changed'):
        encoder.encode(changed)
    changed = copy.deepcopy(original); changed['objects'][0]['objectId'] = 'new'
    with pytest.raises(ValueError, match='object ID set changed'):
        encoder.encode(changed)
    changed = copy.deepcopy(original); del changed['objects'][0]['isMoving']
    with pytest.raises(KeyError, match='isMoving'):
        encoder.encode(changed)
    changed = copy.deepcopy(original); changed['objects'][0]['temperature'] = 'unknown'
    with pytest.raises(ValueError, match='temperature'):
        encoder.encode(changed)
