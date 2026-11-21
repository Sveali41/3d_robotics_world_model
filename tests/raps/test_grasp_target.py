"""Opt-in grasp target reward and bounded height retain dual pick/place."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from domain.robocasa.pick_place import PICK_PLACE_ACTION_SCHEMA
from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.raps.raps_gnn_actor import (
    DEFAULTS, RAPSImaginationActor, SkillCollector, action_contract,
    action_contract_matches, canonical_action, skill_schema,
)


def settings(**overrides):
    return {**DEFAULTS, 'pick_place_primitives': True, 'grasp_z_bounds_m': [.692, .772],
            'grasp_target_heuristic': True, 'motion_execution': 'planner', 'latched_gripper': True,
            'grasp_stop_on_contact': False, 'hidden_size': 8, 'actor_layers': 1,
            'critic_layers': 1, 'imagination_horizon': 2, 'imagination_batch_size': 4,
            'updates_per_round': 1, **overrides}


def backend():
    return CEEUSBackend(agent_dim=33, action_dim=5, hidden_dim=8, batch_size=4, object_heuristic={})


def test_schema_bounds_and_default_contract_identity():
    configured = settings()
    schema = skill_schema(configured)
    assert schema.primitive_names == ('grasp', 'place') and schema.action_dim == 5
    assert schema.specs[1] is PICK_PLACE_ACTION_SCHEMA.spec_by_name['place']
    assert schema.specs[0].parameters[:2] == PICK_PLACE_ACTION_SCHEMA.specs[0].parameters[:2]
    for value, expected in [(-1., .692), (1., .772)]:
        action = np.array([1., 0., 0., 0., value])
        assert schema.decode(action).parameters['z'] == pytest.approx(expected)
    action = canonical_action(torch.tensor([0., 1., .2, -.4, .9]), schema)
    torch.testing.assert_close(action, torch.tensor([0., 1., .2, -.4, 0.]))
    contract = action_contract(configured)
    np.testing.assert_allclose(contract['pick_place']['tcp_bounds_root_m']['grasp'][2], [.692, .772])
    original = settings(); original.pop('grasp_z_bounds_m'); original.pop('grasp_target_heuristic')
    assert skill_schema(original) is PICK_PLACE_ACTION_SCHEMA
    assert action_contract({**original, 'grasp_target_heuristic': False}) == action_contract(original)
    assert not action_contract_matches(action_contract(original), configured)
    assert not action_contract_matches(contract, original)
    for key in ('grasp_z_bounds_m', 'grasp_target_heuristic'):
        assert key not in DEFAULTS
    assert 'grasp_only' not in contract


@pytest.mark.parametrize('bounds', [[.69, .77], [.70, .88], [.77, .70], [.7, .7], [.7], [.7, np.nan]])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError, match='ordered pair'):
        skill_schema(settings(grasp_z_bounds_m=bounds))


def test_settings_require_pick_place():
    model = backend()
    with pytest.raises(ValueError, match='requires pick_place'):
        skill_schema(settings(pick_place_primitives=False))
    configured = settings(pick_place_primitives=False); configured.pop('grasp_z_bounds_m')
    with pytest.raises(ValueError, match='requires pick_place'):
        RAPSImaginationActor(model, configured)
    with pytest.raises(ValueError, match='must be boolean'):
        RAPSImaginationActor(model, settings(grasp_target_heuristic=1))


def target_inputs(actor):
    state = torch.zeros(3, 67)
    state[:, 33:36] = torch.tensor([.4, 0., .75])
    state[:, 45:48] = torch.tensor([.6, 0., .75])
    physical = torch.tensor([[.4, 0., .75], [.5, 0., .75], [.6, 0., .75]])
    parameters = actor.schema.specs[0].parameters
    center = torch.tensor([p.center for p in parameters])
    scale = torch.tensor([p.scale for p in parameters])
    action = torch.cat((torch.tensor([[1., 0.]]).expand(3, -1), (physical - center) / scale), -1)
    return state, action


def test_target_proximity_nearest_eligibility_scaling_and_rng():
    model = backend(); actor = RAPSImaginationActor(model, settings())
    state, action = target_inputs(actor)
    rng_before = torch.random.get_rng_state().clone(); numpy_before = np.random.get_state()
    score = actor.grasp_target_heuristic(state, action)
    torch.testing.assert_close(score, torch.tensor([50., 50. * np.exp(-1), 50.], dtype=torch.float32))
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and numpy_before[2:] == numpy_after[2:]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    model.object_heuristic['object_weights'] = {'cabinet_item_16': 9., 'cabinet_item_17': 0.}
    eligible = actor.grasp_target_heuristic(state, action)
    assert eligible[0] > eligible[1] > eligible[2]
    torch.testing.assert_close(eligible[0], torch.tensor(50.))
    model.object_heuristic['distance_scale_m'] = .2
    torch.testing.assert_close(actor.grasp_target_heuristic(state, action)[1], torch.tensor(50. * np.exp(-.5), dtype=torch.float32))
    model.object_heuristic = None
    torch.testing.assert_close(actor.grasp_target_heuristic(state, action), torch.zeros(3))


def test_imagination_grasp_endpoint_invariance_and_place_curiosity_only(monkeypatch):
    model = backend(); actor = RAPSImaginationActor(model, settings(imagination_horizon=1))
    starts, fixed = target_inputs(actor)
    fixed[1, :2] = torch.tensor([0., 1.])
    monkeypatch.setattr(actor.actor, 'forward', lambda features: SimpleNamespace(rsample=lambda: fixed))
    def predict_endpoint(value):
        def predict(**kwargs):
            predicted = kwargs['observations'].clone(); predicted[..., 18:21] = value
            predicted[..., 0] += torch.arange(predicted.shape[0])[:, None] * .001
            return predicted, None
        return predict
    monkeypatch.setattr(model.model, 'predict', predict_endpoint(0.))
    first = actor.imagine(starts)['heuristics']
    monkeypatch.setattr(model.model, 'predict', predict_endpoint(100.))
    trajectory = actor.imagine(starts)
    second = trajectory['heuristics']
    torch.testing.assert_close(first[0, [0, 2]], second[0, [0, 2]])
    next_state = starts.clone(); next_state[:, 18:21] = 100.
    legacy, _ = model._object_heuristic_scores(starts, next_state)
    torch.testing.assert_close(second[0, 1], torch.tensor(0.))
    assert legacy[1] != 0
    intrinsic = trajectory['components'][0, 1].sum()
    assert intrinsic > 0
    torch.testing.assert_close(trajectory['rewards'][0, 1, 0], intrinsic)
    setting = model.object_heuristic
    model.object_heuristic = None
    torch.testing.assert_close(actor.imagine(starts)['heuristics'], torch.zeros(1, 3))
    model.object_heuristic = setting
    actor.settings['grasp_target_heuristic'] = False
    torch.testing.assert_close(actor.imagine(starts)['heuristics'][0], legacy)
    contract = action_contract(settings())
    assert contract['grasp_target_heuristic']['version'] == 2
    old_contract = deepcopy(contract)
    old_contract['grasp_target_heuristic']['version'] = 1
    old_contract['grasp_target_heuristic']['selected_place'] = 'legacy_endpoint_TCP_progress_heuristic'
    assert not action_contract_matches(old_contract, settings())


def test_real_five_dimensional_model_actor_update_and_restore(tmp_path):
    torch.set_num_threads(1)
    model = backend(); actor = RAPSImaginationActor(model, settings())
    rng = np.random.default_rng(19)
    states = rng.normal(size=(8, 67)).astype(np.float32)
    actions = canonical_action(torch.tensor(rng.uniform(-1, 1, (8, 5)), dtype=torch.float32), actor.schema).numpy()
    next_states = states.copy(); next_states[:, :model.dynamic_dim] += .001
    data = dict(states=states, actions=actions, next_states=next_states, terminated=np.zeros(8, bool))
    model.train(data, 1)
    metrics = actor.train(data)
    assert metrics['updates'] == 1 and np.isfinite(list(metrics.values())).all()
    sampled = np.stack([actor.action(states[0]) for _ in range(64)])
    assert sampled.shape == (64, 5) and set(sampled[:, :2].argmax(-1)) == {0, 1}
    imagined = actor.imagine(torch.tensor(states))
    assert imagined['actions'].shape == (2, 8, 5)
    assert set(imagined['actions'][..., :2].argmax(-1).flatten().tolist()) == {0, 1}
    collector = SkillCollector.__new__(SkillCollector); collector.schema = actor.schema
    random = np.stack([collector._propose_action('random') for _ in range(64)])
    assert random.shape == (64, 5) and set(random[:, :2].argmax(-1)) == {0, 1}
    saved = deepcopy(actor.state_dict()); restored = RAPSImaginationActor(model, settings())
    restored.load_state_dict(saved)
    for key in ('actor', 'critic', 'target_critic', 'actor_optimizer', 'critic_optimizer'):
        torch.testing.assert_close(restored.state_dict()[key], saved[key])
    from exploration.raps.raps_gnn_actor import SCHEMA, load
    from exploration.cee_us.cee_us_cabinet import rng_state
    encoder = SimpleNamespace(contract=lambda: {'test_state_dim': 67})
    payload = dict(schema=SCHEMA, backend=deepcopy(model.state_dict()), actor=saved,
                   state_contract=encoder.contract(), action_contract=action_contract(actor.settings), rng=rng_state())
    checkpoint = tmp_path / 'target.pt'; torch.save(payload, checkpoint)
    assert load(checkpoint, model, restored, encoder)['action_contract']['action_dim'] == 5
    for changed in (settings(grasp_target_heuristic=False), {k:v for k,v in settings().items() if k != 'grasp_z_bounds_m'}):
        payload['action_contract'] = action_contract(changed); torch.save(payload, checkpoint)
        with pytest.raises(ValueError, match='contract differs'):
            load(checkpoint, model, restored, encoder)
        with pytest.raises(ValueError, match='settings differ'):
            actor.load_state_dict({**saved, 'settings': changed})


def test_target_config_preserves_preflight_settings():
    original = OmegaConf.load('configs/raps_gnn_actor_cabinet_pick_place_preflight.yaml')
    target = OmegaConf.load('configs/raps_gnn_actor_cabinet_pick_place_target.yaml')
    assert target.raps_actor.pop('grasp_target_heuristic') is True
    assert list(target.raps_actor.pop('grasp_z_bounds_m')) == [.692, .772]
    assert target == original
