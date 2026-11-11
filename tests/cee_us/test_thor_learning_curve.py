from types import SimpleNamespace
import math

import numpy as np
import torch

from exploration.cee_us.cee_us_thor import _metrics, make_backend
from exploration.cee_us.thor_learning_curve import evaluate_one_step
from exploration.cee_us.thor_state import ACTIONS, ThorState
from test_thor_world_model import metadata


def test_one_step_evaluation_streams_batches_and_aggregates_metrics():
    initial = metadata()
    encoder = ThorState(initial)
    args = SimpleNamespace(hidden_dim=8, device='cpu', seed=0)
    backend = make_backend(encoder, args)
    state = encoder.encode(initial)
    states = np.stack([state] * 6)
    types = np.asarray([0, 1, 8, 9, 10, 11])
    actions = encoder.action_encoding(torch.from_numpy(states), torch.from_numpy(types))[0].numpy()
    truth = states.copy()
    truth[2, 0] += .3
    truth[3, encoder.agent_dim + 11] = 1
    successes = np.asarray([True, False, True, False, True, False])
    replay = {'states': states, 'next_states': truth, 'actions': actions, 'types': types,
              'successes': successes, 'targets': np.asarray(['', '', 'a', 'a', '', '']),
              'unity_calls': np.ones(6, dtype=bool)}
    backend.train({'states': states, 'actions': actions, 'next_states': truth}, 1)

    norms = {name: getattr(backend.model, name).mean.cpu().numpy().copy()
             for name in backend.normalizer_names}
    original_predict = backend.predict
    batch_sizes = []

    def tracked_predict(batch_states, batch_actions):
        batch_sizes.append(len(batch_states))
        return original_predict(batch_states, batch_actions)

    backend.predict = tracked_predict
    report = evaluate_one_step(backend, encoder, replay, batch_size=2)
    predictions = np.concatenate([original_predict(states[i:i + 1], actions[i:i + 1])
                                  for i in range(len(states))], axis=1)
    full_metrics = _metrics(encoder, predictions.mean(axis=0), truth, states)

    assert max(batch_sizes) <= 2
    assert len(batch_sizes) == math.ceil(len(states) / 2)
    assert report['conditional_one_step']['count'] == 6
    for key, value in full_metrics.items():
        if key == 'count':
            continue
        if value is None:
            assert report['conditional_one_step'][key] is None
        else:
            np.testing.assert_allclose(report['conditional_one_step'][key], value, rtol=1e-6, atol=1e-7)
    assert report['groups']['success']['count'] == 3
    assert report['groups']['failure']['count'] == 3
    assert report['groups']['navigation']['count'] == 2
    assert report['groups']['interaction']['count'] == 4
    assert report['groups']['changed']['count'] == 2
    assert report['groups']['unchanged']['count'] == 4
    assert report['groups']['changed_interaction']['count'] == 2
    assert report['per_action'][ACTIONS[8]]['count'] == 1
    assert report['conditional_one_step']['changed_coordinates_mae'] is not None
    assert len(report['examples']) == 6
    assert all(report['examples'][i]['action'] == ACTIONS[types[i]] for i in range(6))
    for name, expected in norms.items():
        np.testing.assert_array_equal(getattr(backend.model, name).mean.cpu().numpy(), expected)


def test_streaming_metrics_match_full_row_metrics_for_groups_and_actions():
    encoder = ThorState(metadata())
    count = 137
    state = encoder.encode(metadata())
    states = np.repeat(state[None], count, axis=0)
    sample_ids = np.arange(count)
    states[:, 0] = sample_ids / 1000
    types = sample_ids % len(ACTIONS)
    actions = encoder.action_encoding(torch.from_numpy(states), torch.from_numpy(types))[0].numpy()
    truth = states.copy()
    truth[sample_ids % 4 == 0, 1] += .2
    successes = sample_ids % 3 == 0
    truth[:, 8] = successes
    parent_start = encoder.dynamic_dim - encoder.n
    truth[:, parent_start] = sample_ids % 3 == 0
    truth[:, parent_start + 1] = sample_ids % 5 == 0
    held_index = encoder.agent_dim + 11
    truth[:, held_index] = sample_ids % 7 == 0

    class Backend:
        def __init__(self):
            self.calls = []
            self.means = []

        def predict(self, batch_states, _actions):
            self.calls.append(len(batch_states))
            mean = batch_states.copy()
            ids = np.rint(batch_states[:, 0] * 1000).astype(int)
            mean[:, 0] += (ids % 7) * .002
            mean[:, 8] = (ids % 3) / 2
            mean[:, parent_start] = ids % 4 == 0
            mean[:, parent_start + 1] = ids % 6 == 0
            mean[:, held_index] = ids % 5 == 0
            self.means.append(mean.copy())
            return np.broadcast_to(mean[None], (3, *mean.shape)).copy()

    backend = Backend()
    data = {'states': states, 'next_states': truth, 'actions': actions, 'types': types,
            'successes': successes, 'targets': np.asarray(['target'] * count),
            'unity_calls': np.ones(count, dtype=bool)}
    report = evaluate_one_step(backend, encoder, data, batch_size=17)

    means = np.concatenate(backend.means)
    assert len(backend.calls) == math.ceil(count / 17)

    def compare_metrics(actual, mask):
        expected = _metrics(encoder, means[mask], truth[mask], states[mask])
        assert actual['count'] == int(mask.sum())
        for name, value in expected.items():
            if name == 'count':
                continue
            if value is None:
                assert actual[name] is None
            else:
                np.testing.assert_allclose(actual[name], value, rtol=1e-6, atol=1e-7)

    compare_metrics(report['conditional_one_step'], np.ones(count, dtype=bool))
    group_masks = {
        'success': successes,
        'failure': ~successes,
        'navigation': types < 8,
        'interaction': types >= 8,
        'changed': np.any(np.abs(truth[:, np.r_[np.arange(8), np.arange(encoder.agent_dim, encoder.dynamic_dim)]]
                                  - states[:, np.r_[np.arange(8), np.arange(encoder.agent_dim, encoder.dynamic_dim)]]) > 1e-6, axis=1),
    }
    group_masks['unchanged'] = ~group_masks['changed']
    group_masks['changed_interaction'] = group_masks['changed'] & (types >= 8)
    for name, mask in group_masks.items():
        if mask.any():
            compare_metrics(report['groups'][name], mask)
    for action_index, action in enumerate(ACTIONS):
        mask = types == action_index
        if mask.any():
            compare_metrics(report['per_action'][action], mask)
    assert len(report['examples']) == 16
    np.testing.assert_allclose(report['examples'][0]['predicted_success_value'], means[0, 8])


def test_no_changed_coordinates_has_null_changed_mae():
    encoder = ThorState(metadata())
    state = encoder.encode(metadata())
    states = np.stack([state, state])
    types = np.asarray([0, 8])
    actions = encoder.action_encoding(torch.from_numpy(states), torch.from_numpy(types))[0].numpy()

    class Backend:
        def predict(self, batch_states, _actions):
            return np.broadcast_to(batch_states[None], (2, *batch_states.shape)).copy()

    report = evaluate_one_step(Backend(), encoder, {
        'states': states, 'next_states': states.copy(), 'actions': actions,
        'types': types, 'successes': np.ones(2, dtype=bool),
        'targets': np.asarray(['', 'a']), 'unity_calls': np.ones(2, dtype=bool),
    })
    metrics = report['conditional_one_step']
    assert metrics['changed_coordinates_mae'] is None
    assert metrics['zero_change_changed_coordinates_mae'] is None
    assert report['groups']['changed']['count'] == 0
