import numpy as np
import pytest
import torch

from exploration.cee_us_backend import CEEUSBackend


def single_backend():
    return CEEUSBackend(agent_dim=33, num_objects=1, action_dim=12,
                        object_names=('cabinet_item_box',),
                        object_heuristic={'object_weights': {'cabinet_item_box': 1.}},
                        hidden_dim=8, batch_size=4, horizon=3, candidates=8, elites=4)


@pytest.mark.parametrize('batch_size', [1, 3, 8])
def test_single_node_prediction_training_and_static_attributes(batch_size):
    backend = single_backend()
    rng = np.random.default_rng(17)
    states = rng.normal(size=(batch_size, 50)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(batch_size, 12)).astype(np.float32)
    next_states = states.copy()
    next_states[:, :45] += rng.normal(size=(batch_size, 45)).astype(np.float32) * .01
    before = backend.predict(states, actions)
    assert before.shape == (5, batch_size, 50) and np.isfinite(before).all()
    assert np.isfinite(before.std(axis=0, ddof=1)).all()
    np.testing.assert_array_equal(before[:, :, 45:], np.broadcast_to(states[None, :, 45:], (5, batch_size, 5)))
    if batch_size == 1:
        states, actions, next_states = (np.repeat(x, 2, axis=0) for x in (states, actions, next_states))
    metrics = backend.train(dict(states=states, actions=actions, next_states=next_states), epochs=1)
    assert metrics['finitegrad'] and metrics['updates'] > 0
    assert np.isfinite(backend.predict(states, actions)).all()


def test_single_object_heuristic_uses_actual_slot():
    backend = single_backend()
    before = np.zeros(50, np.float32)
    before[33] = 1.
    after = before.copy()
    after[18] = .1
    result = backend.observed_object_heuristic(before, after)
    assert result['approach_progress_score'] == pytest.approx(50.)
    assert result['per_object_progress_m'] == pytest.approx({'cabinet_item_box': .1})
    with pytest.raises(ValueError, match='object_weights'):
        CEEUSBackend(num_objects=1, object_names=('cabinet_item_box',),
                     object_heuristic={'object_weights': {'cabinet_item_16': 1.}})


def test_two_node_prediction_unchanged_by_single_node_path():
    backend = CEEUSBackend(agent_dim=33, action_dim=12, hidden_dim=8)
    graph = backend.model.model
    rng = np.random.default_rng(3)
    states = rng.normal(size=(4, 67)).astype(np.float32)
    actions = rng.normal(size=(4, 12)).astype(np.float32)
    expected = backend.predict(states, actions)
    graph(torch.zeros(5, 4, 33), torch.zeros(5, 4, 1, 12),
          torch.zeros(5, 4, 1, 5), torch.zeros(5, 4, 12))
    np.testing.assert_array_equal(backend.predict(states, actions), expected)
