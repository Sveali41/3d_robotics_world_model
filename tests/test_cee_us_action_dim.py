"""Focused checks for configurable CEE-US action dimensions."""
import numpy as np
import pytest

from exploration.cee_us_backend import CEEUSBackend


def make_backend(action_dim=20):
    return CEEUSBackend(
        hidden_dim=8, horizon=2, candidates=4, iterations=1, elites=2,
        batch_size=2, action_dim=action_dim,
    )


def test_twenty_dimensional_actions_train_predict_and_checkpoint():
    backend = make_backend()
    rng = np.random.default_rng(11)
    states = rng.normal(size=(4, 66)).astype(np.float32)
    next_states = states.copy()
    next_states[:, :56] += rng.normal(size=(4, 56)).astype(np.float32) * .1
    actions = rng.uniform(-1, 1, size=(4, 20)).astype(np.float32)
    replay = dict(states=states, actions=actions, next_states=next_states)

    assert backend.action_dim == 20
    assert backend.env.action_space.shape == (20,)
    assert backend.train(replay, epochs=1)['updates'] > 0
    predicted = backend.predict(states, actions)
    assert predicted.shape == (5, 4, 66)
    np.testing.assert_array_equal(
        predicted[:, :, 56:], np.broadcast_to(states[None, :, 56:], (5, 4, 10))
    )

    checkpoint = backend.state_dict()
    assert checkpoint['action_dim'] == 20
    backend.load_state_dict(checkpoint)
    with pytest.raises(ValueError, match='action_dim'):
        make_backend(action_dim=4).load_state_dict(checkpoint)


def test_old_four_dimensional_checkpoint_without_action_dim_loads():
    backend = make_backend(action_dim=4)
    legacy_checkpoint = backend.state_dict()
    legacy_checkpoint.pop('action_dim')
    backend.load_state_dict(legacy_checkpoint)


def test_action_projection_requires_four_dimensional_actions():
    projection = dict(
        root_to_cabinet_rotation=np.eye(3), root_to_cabinet_translation=np.zeros(3),
        lower=-np.ones(3), upper=np.ones(3), pos_lower=-np.ones(3), pos_upper=np.ones(3),
    )
    with pytest.raises(ValueError, match='action_dim=4'):
        CEEUSBackend(action_dim=20, action_projection=projection)
