"""Protected replay integration with the existing GNN optimizer and actor input."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.cee_us.cee_us_cabinet import rng_state, restore_rng
from exploration.raps.raps_gnn_actor import fit, skill_arrays, _protected_replay_config


def setup_fit(mode='protected'):
    torch.set_num_threads(1)
    config = OmegaConf.load('configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml')
    if mode == 'protected':
        config.wm_replay = dict(updates=3, movement_threshold_m=.015, z_bin_m=.005, joint_bin_degrees=5)
        del config['wm_training']
    else:
        config.wm_replay.updates = 3
    agent_dim = 33 if mode == 'protected' else 31
    backend = CEEUSBackend(agent_dim=agent_dim, num_objects=1, object_names=('cup',),
                           action_dim=31, hidden_dim=8, batch_size=8, attach_icem=False,
                           **dict(config.get('wm_training') or {}))
    states = np.random.default_rng(13).normal(size=(8, backend.state_dim)).astype(np.float32)
    actions = np.zeros((8, 31), np.float32)
    actions[:, 12] = 1
    next_states = states.copy()
    next_states[[0, 4], agent_dim] += .02
    skills = [dict(state=states[i], action=actions[i], next_state=next_states[i],
                   trainable=True, completed=True, object_contact=i in (1, 5),
                   stage='bootstrap' if i < 4 else 'round_1', duration_low_steps=10,
                   episode_id=i + 1, terminated=False, truncated=False) for i in range(8)]
    observed = []
    actor = SimpleNamespace(train=lambda arrays, updates=None: (
        observed.append(arrays) or dict(updates=0)))
    return backend, actor, config, skills, observed


@pytest.mark.parametrize('mode', ['protected', 'bounded_dynamic'])
def test_fit_fixed_budget_keeps_all_actor_states_and_reproduces_resume(tmp_path, mode):
    backend, actor, config, skills, observed = setup_fit(mode)
    state = deepcopy(backend.state_dict())
    rng = rng_state()
    result = fit(backend, actor, skills, config, tmp_path, 'round_1')
    assert result['model']['updates'] == 3
    np.testing.assert_array_equal(observed[-1]['states'], skill_arrays(skills)['states'])
    assert len(skills) == 8
    prediction = backend.predict(observed[-1]['states'], observed[-1]['actions'])
    backend.load_state_dict(state)
    restore_rng(rng)
    repeated = fit(backend, actor, skills, config, tmp_path, 'round_1')
    assert result['model'] == repeated['model']
    np.testing.assert_array_equal(backend.predict(observed[-1]['states'], observed[-1]['actions']), prediction)


def test_old_records_require_explicit_contact_evidence(tmp_path):
    backend, actor, config, skills, _ = setup_fit()
    skills[0].pop('object_contact')
    with pytest.raises(ValueError, match='intermediate contact summaries'):
        fit(backend, actor, skills, config, tmp_path, 'round_1')
    assert backend.updates == 0


def test_legacy_fit_remains_available(tmp_path):
    backend, actor, config, skills, _ = setup_fit()
    del config['wm_replay']
    config.epochs = 2
    for row in skills:
        row.pop('object_contact')
    result = fit(backend, actor, skills, config, tmp_path, 'round_1')
    assert result['model']['updates'] == 2
    assert 'replay' not in result['model']


@pytest.mark.parametrize('key,value', [('updates', True), ('updates', 0),
                                     ('movement_threshold_m', float('nan')),
                                     ('joint_bin_degrees', -1)])
def test_replay_configuration_rejects_invalid_parameters(key, value):
    config = OmegaConf.load('configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml')
    config.wm_replay[key] = value
    with pytest.raises(ValueError, match='wm_replay'):
        _protected_replay_config(config)


def test_replay_configuration_rejects_other_action_units():
    config = OmegaConf.load('configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml')
    config.raps_actor.grasp_z_bounds_m = [.1, .2]
    with pytest.raises(ValueError, match='action contract'):
        _protected_replay_config(config)
