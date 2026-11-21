"""Shared discrete XY action contract and actual reference actor integration."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from domain.robocasa.cabinet_grid import GRID_XY_ROOT_M
from domain.robocasa.primitives import PrimitiveAction
from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.raps.raps_gnn_actor import (
    DEFAULTS, RAPSImaginationActor, SkillCollector, SCHEMA, action_contract,
    action_contract_matches, canonical_action, load, skill_schema, summarize,
)
from exploration.cee_us.cee_us_cabinet import rng_state
from tests.raps.test_grasp_target import settings as target_settings


def settings(**overrides):
    return target_settings(**{'pick_place_grid': True, **overrides})


@pytest.mark.parametrize('grasp_only,dimension', [(False, 25), (True, 13)])
def test_all_grid_cells_roundtrip_and_z_boundaries(grasp_only, dimension):
    schema = skill_schema(settings(grasp_only=grasp_only))
    assert schema.action_dim == dimension
    assert schema.num_parameters == schema.continuous_action_dim == 1
    assert schema.primitive_names[:12] == tuple(f'grasp_{i:02d}' for i in range(12))
    for kind in schema.logical_primitive_names:
        for index, (x, y) in enumerate(GRID_XY_ROOT_M):
            for z in (.692, .772):
                parameters = dict(x=x, y=y, **({'z': z} if kind == 'grasp' else {}))
                encoded = schema.encode(PrimitiveAction(kind, parameters))
                assert encoded.argmax() == index + (12 if kind == 'place' else 0)
                decoded = schema.decode(encoded)
                assert decoded.primitive_type == kind
                assert decoded.parameters == pytest.approx(parameters)
                np.testing.assert_allclose(schema.encode(decoded), encoded)
                if kind == 'place':
                    encoded[-1] = .9
                    assert schema.decode(encoded).parameters == parameters
                    assert canonical_action(torch.tensor(encoded), schema)[-1] == 0
    if not grasp_only:
        assert schema.primitive_names[12:] == tuple(f'place_{i:02d}' for i in range(12))
        assert schema.primitive_name_to_action_idx['place_11'] == []
    assert schema.primitive_name_to_action_idx['grasp_11'] == [0]


@pytest.mark.parametrize('kind,parameters', [
    ('grasp', dict(x=.4, y=-.06, z=.73)),
    ('grasp', dict(x=.395, y=-.06, z=.69)),
    ('grasp', dict(x=.395, y=-.06, z=.78)),
    ('grasp', dict(x=.395, y=-.06)),
    ('place', dict(x=.395, y=-.06, z=.73)),
    ('place', dict(x=np.nan, y=-.06)),
])
def test_invalid_physical_arguments(kind, parameters):
    with pytest.raises(ValueError):
        skill_schema(settings()).encode(PrimitiveAction(kind, parameters))


def test_opt_in_flags_and_strict_contracts():
    assert 'pick_place_grid' not in DEFAULTS
    for bad in (1, 'true', None):
        with pytest.raises(ValueError, match='must be boolean'):
            skill_schema(settings(pick_place_grid=bad))
    with pytest.raises(ValueError, match='requires pick_place'):
        skill_schema(settings(pick_place_primitives=False))
    configured = settings()
    contract = action_contract(configured)
    grid = contract['pick_place_grid']
    assert grid['version'] == 1 and grid['action_dim'] == 25
    assert grid['xy_root_m'] == [list(xy) for xy in GRID_XY_ROOT_M]
    assert grid['grasp_z_root_m'] == pytest.approx([.692, .772])
    assert contract['pick_place']['action_dim'] == 25
    assert contract['pick_place']['tcp_bounds_root_m']['grasp'][2] == pytest.approx([.692, .772])
    old = target_settings()
    assert not action_contract_matches(contract, old)
    assert not action_contract_matches(action_contract(old), configured)
    changed = deepcopy(contract); changed['pick_place_grid']['xy_root_m'][0][0] += .001
    assert not action_contract_matches(changed, configured)
    changed = deepcopy(contract); changed['pick_place_grid']['version'] = 0
    assert not action_contract_matches(changed, configured)
    default_z = settings(); default_z.pop('grasp_z_bounds_m')
    assert action_contract(default_z)['pick_place_grid']['grasp_z_root_m'] == pytest.approx([.692, .872])


def test_selected_grid_target_and_place_reward(monkeypatch):
    model = CEEUSBackend(agent_dim=33, action_dim=25, hidden_dim=8, batch_size=4, object_heuristic={})
    actor = RAPSImaginationActor(model, settings(imagination_horizon=1))
    states = torch.zeros(3, 67)
    states[:, 33:36] = torch.tensor([*GRID_XY_ROOT_M[4], .732])
    states[:, 45:48] = torch.tensor([*GRID_XY_ROOT_M[5], .732])
    actions = torch.zeros(3, 25); actions[0, 4] = actions[1, 5] = actions[2, 16] = 1
    torch.testing.assert_close(actor.grasp_target_heuristic(states, actions), torch.full((3,), 50.))
    actions[1, -1] = 1
    assert actor.grasp_target_heuristic(states, actions)[1] == pytest.approx(50 * np.exp(-.04/.1))
    off_cell = actions.clone(); off_cell[0, 4] = 0; off_cell[0, 0] = 1
    assert actor.grasp_target_heuristic(states, off_cell)[0] == pytest.approx(50 * np.exp(-.12/.1))
    monkeypatch.setattr(actor.actor, 'forward', lambda features: SimpleNamespace(rsample=lambda: actions))
    imagined = actor.imagine(states)
    assert imagined['heuristics'][0, 2] == 0
    assert imagined['heuristics'][0, 0] == 50
    from exploration.raps import raps_gnn_actor as module
    monkeypatch.setattr(module, 'interaction_summary', lambda ticks: {})
    report = summarize([], [dict(primitive_type='grasp', trainable=True, completed=True),
                               dict(primitive_type='place', trainable=False, completed=False)], actor.schema)
    assert report['per_skill']['grasp']['selected'] == 1
    assert report['per_skill']['place']['selected'] == 1


@pytest.mark.parametrize('grasp_only,orientations,dimension,grid_layout', [
    (False, False, 25, 'legacy_3x4'), (True, False, 13, 'legacy_3x4'),
    (False, True, 73, 'legacy_3x4'), (True, True, 61, 'legacy_3x4'),
    (False, True, 73, 'compact_3x4'), (True, True, 61, 'compact_3x4'),
    (True, True, 31, 'trapezoid_4_3_2_1')])
def test_actual_grid_model_actor_update_restore(tmp_path, grasp_only, orientations, dimension, grid_layout):
    torch.set_num_threads(1)
    doorless = grid_layout == 'trapezoid_4_3_2_1'
    model = CEEUSBackend(agent_dim=31 if doorless else 33, action_dim=dimension,
        num_objects=1 if doorless else 2, object_names=('cabinet_item_16',) if doorless else None,
        hidden_dim=8, batch_size=4,
        object_heuristic={'object_weights': {'cabinet_item_16': 1.}} if doorless else {})
    configured = settings(grasp_only=grasp_only, grasp_orientation_candidates=orientations,
                          grid_layout=grid_layout)
    actor = RAPSImaginationActor(model, configured)
    rng = np.random.default_rng(19)
    states = rng.normal(size=(8, model.state_dim)).astype(np.float32)
    actions = canonical_action(torch.tensor(rng.uniform(-1, 1, (8, dimension)), dtype=torch.float32), actor.schema).numpy()
    next_states = states.copy(); next_states[:, :model.dynamic_dim] += .001
    data = dict(states=states, actions=actions, next_states=next_states, terminated=np.zeros(8, bool))
    model.train(data, 1)
    metrics = actor.train(data)
    assert metrics['updates'] == 1 and np.isfinite(list(metrics.values())).all()
    sampled = np.stack([actor.action(states[0]) for _ in range(64)])
    assert sampled.shape == (64, dimension)
    assert actor.imagine(torch.tensor(states))['actions'].shape == (2, 8, dimension)
    collector = SkillCollector.__new__(SkillCollector); collector.schema = actor.schema
    random = np.stack([collector._propose_action('random') for _ in range(128)])
    assert random.shape == (128, dimension)
    assert all(actor.schema.decode(action).primitive_type in actor.schema.logical_primitive_names for action in random)
    saved = deepcopy(actor.state_dict()); restored = RAPSImaginationActor(model, configured)
    restored.load_state_dict(saved)
    for key in ('actor', 'critic', 'target_critic', 'actor_optimizer', 'critic_optimizer'):
        torch.testing.assert_close(restored.state_dict()[key], saved[key])
    encoder = SimpleNamespace(contract=lambda: {'test_state_dim': model.state_dim})
    payload = dict(schema=SCHEMA, backend=deepcopy(model.state_dict()), actor=saved,
                   state_contract=encoder.contract(), action_contract=action_contract(actor.settings), rng=rng_state())
    checkpoint = tmp_path / 'grid.pt'; torch.save(payload, checkpoint)
    assert load(checkpoint, model, restored, encoder)['action_contract']['action_dim'] == dimension
    changed = {**actor.settings, **({'grasp_orientation_candidates': False} if orientations
                                  else {'pick_place_grid': False})}
    payload['action_contract'] = action_contract(changed); torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='contract differs'):
        load(checkpoint, model, restored, encoder)
    with pytest.raises(ValueError, match='settings differ'):
        restored.load_state_dict({**saved, 'settings': changed})


@pytest.mark.parametrize('grasp_only,dimension', [(False, 73), (True, 61)])
def test_orientation_grid_roundtrips_and_place_inactive_z(grasp_only, dimension):
    schema = skill_schema(settings(grasp_only=grasp_only, grasp_orientation_candidates=True))
    assert schema.action_dim == dimension and schema.num_grasp_branches == 60
    for cell, (x, y) in enumerate(GRID_XY_ROOT_M):
        for orientation_id in range(5):
            branch = cell * 5 + orientation_id
            assert schema.branch_cell_indices[branch] == cell
            assert schema.branch_orientation_indices[branch] == orientation_id
            for z in (.692, .772):
                action = PrimitiveAction('grasp', dict(x=x, y=y, z=z, orientation_id=orientation_id))
                encoded = schema.encode(action)
                assert encoded[:schema.num_primitives].argmax() == branch
                assert schema.decode(encoded).parameters == pytest.approx(action.parameters)
                np.testing.assert_allclose(schema.encode(schema.decode(encoded)), encoded)
            if not grasp_only:
                encoded = schema.encode(PrimitiveAction('place', dict(x=x, y=y)))
                assert encoded.argmax() == 60 + cell
                encoded[-1] = .9
                assert schema.decode(encoded).parameters == dict(x=x, y=y)
                assert canonical_action(torch.tensor(encoded), schema)[-1] == 0
    for orientation_id in (-1, 5, 1.5, True, None, float('nan'), '1'):
        with pytest.raises(ValueError):
            schema.encode(PrimitiveAction('grasp', dict(x=.395, y=-.06, z=.732,
                                                       orientation_id=orientation_id)))
    for parameters in (dict(x=.395, y=-.06, z=.732),
                       dict(x=.395, y=-.06, z=.69, orientation_id=0),
                       dict(x=.395, y=-.06, z=.78, orientation_id=0)):
        with pytest.raises(ValueError):
            schema.encode(PrimitiveAction('grasp', parameters))


def test_orientation_opt_in_and_contract_rejection():
    assert 'grasp_orientation_candidates' not in DEFAULTS
    configured = settings(grasp_orientation_candidates=True)
    for bad in (1, 'true', None):
        with pytest.raises(ValueError, match='must be boolean'):
            skill_schema({**configured, 'grasp_orientation_candidates': bad})
    for overrides in (dict(pick_place_grid=False), dict(pick_place_primitives=False),
                      dict(motion_execution='ee_delta')):
        with pytest.raises(ValueError, match='grasp_orientation_candidates requires'):
            skill_schema({**configured, **overrides})
    contract = action_contract(configured)
    assert contract['grasp_orientation_candidates']['yaw_pitch_degrees'] == [
        [0, 0], [-15, 0], [15, 0], [-45, 30], [45, 30]]
    assert not action_contract_matches(contract, settings())
    assert not action_contract_matches(action_contract(settings()), configured)
    changed = deepcopy(contract)
    changed['grasp_orientation_candidates']['yaw_pitch_degrees'][1][0] = -30
    assert not action_contract_matches(changed, configured)


def test_position_reward_across_all_orientation_branches(monkeypatch):
    model = CEEUSBackend(agent_dim=33, action_dim=73, hidden_dim=8, batch_size=4, object_heuristic={})
    actor = RAPSImaginationActor(model, settings(grasp_orientation_candidates=True, imagination_horizon=1))
    states = torch.zeros(61, 67)
    states[:, 33:36] = torch.tensor([*GRID_XY_ROOT_M[4], .732])
    states[:, 45:48] = torch.tensor([*GRID_XY_ROOT_M[5], .732])
    actions = torch.zeros(61, 73)
    actions[:60, :60] = torch.eye(60)
    actions[60, 64] = 1
    reward = actor.grasp_target_heuristic(states, actions)[:60].reshape(12, 5)
    torch.testing.assert_close(reward, reward[:, :1].expand(12, 5))
    assert reward[4, 0] == reward[5, 0] == 50
    monkeypatch.setattr(actor.actor, 'forward', lambda features: SimpleNamespace(rsample=lambda: actions))
    imagined = actor.imagine(states)
    torch.testing.assert_close(imagined['heuristics'][0, :60], reward.flatten())
    assert imagined['heuristics'][0, 60] == 0


@pytest.mark.parametrize('grasp_only,dimension', [(True, 61), (False, 73)])
def test_compact_grid_all_branches_and_contracts(grasp_only, dimension):
    from domain.robocasa.cabinet_grid import COMPACT_GRID_XY_ROOT_M
    compact = settings(grid_layout='compact_3x4', grasp_only=grasp_only,
                       grasp_orientation_candidates=True)
    schema = skill_schema(compact)
    assert schema.action_dim == dimension and schema.num_grasp_branches == 60
    assert schema.grid_xy_root_m == tuple((x, y) for x in (.375, .405, .435)
                                         for y in (0., .08, .16, .24))
    assert len(set(schema.grid_xy_root_m)) == 12
    for cell, (x, y) in enumerate(COMPACT_GRID_XY_ROOT_M):
        for orientation in range(5):
            for z in (.692, .772):
                action = PrimitiveAction('grasp', dict(x=x, y=y, z=z, orientation_id=orientation))
                vector = schema.encode(action)
                assert vector.argmax() == cell * 5 + orientation
                assert schema.decode(vector).parameters == pytest.approx(action.parameters)
        if not grasp_only:
            action = PrimitiveAction('place', dict(x=x, y=y))
            vector = schema.encode(action)
            assert vector.argmax() == 60 + cell
            assert schema.decode(vector).parameters == action.parameters
    with pytest.raises(ValueError):
        schema.encode(PrimitiveAction('grasp', dict(x=.375, y=0., z=.69, orientation_id=0)))
    legacy = settings(grasp_only=grasp_only, grasp_orientation_candidates=True)
    assert action_contract(legacy) == action_contract({**legacy, 'grid_layout': 'legacy_3x4'})
    assert not action_contract_matches(action_contract(legacy), compact)
    assert not action_contract_matches(action_contract(compact), legacy)
    grid = action_contract(compact)['pick_place_grid']
    assert grid['layout'] == 'compact_3x4'
    assert grid['place_xy_reference'] == 'collision_AABB_center_at_reset_orientation'


@pytest.mark.parametrize('grasp_only,dimension', [(True, 31), (False, 41)])
def test_trapezoid_restricts_grasp_to_horizontal_ids(grasp_only, dimension):
    configured = settings(grid_layout='trapezoid_4_3_2_1', grasp_only=grasp_only,
                          grasp_orientation_candidates=True)
    schema = skill_schema(configured)
    assert schema.action_dim == dimension
    assert schema.num_grasp_branches == 30
    assert schema.grasp_orientation_ids == (0, 1, 2)
    for cell, (x, y) in enumerate(schema.grid_xy_root_m):
        for orientation_id in (0, 1, 2):
            action = PrimitiveAction('grasp', dict(x=x, y=y, z=.732, orientation_id=orientation_id))
            vector = schema.encode(action)
            assert vector.argmax() == cell * 3 + orientation_id
            assert schema.decode(vector).parameters == pytest.approx(action.parameters)
        for orientation_id in (3, 4):
            with pytest.raises(ValueError, match='orientation_id must be one of'):
                schema.encode(PrimitiveAction('grasp', dict(x=x, y=y, z=.732,
                                                          orientation_id=orientation_id)))
        if not grasp_only:
            action = PrimitiveAction('place', dict(x=x, y=y))
            vector = schema.encode(action)
            assert vector.argmax() == 30 + cell
            assert schema.decode(vector).parameters == action.parameters
    contract = action_contract(configured)
    orientation_contract = contract['grasp_orientation_candidates']
    assert orientation_contract['orientation_ids'] == [0, 1, 2]
    assert orientation_contract['yaw_pitch_degrees'] == [[0, 0], [-15, 0], [15, 0]]
    assert action_contract_matches(contract, configured)
    old = deepcopy(contract)
    old['grasp_orientation_candidates']['yaw_pitch_degrees'] += [[-45, 30], [45, 30]]
    assert not action_contract_matches(old, configured)


def test_trapezoid_default_keeps_existing_learned_z_range():
    configured = settings(grid_layout='trapezoid_4_3_2_1', grasp_only=True,
                          grasp_orientation_candidates=True)
    configured.pop('grasp_z_bounds_m', None)
    schema = skill_schema(configured)
    assert schema.action_dim == 31
    x, y = schema.grid_xy_root_m[9]
    for z, normalized in ((.692, -1.), (.732, 0.), (.772, 1.)):
        action = PrimitiveAction('grasp', dict(x=x, y=y, z=z, orientation_id=0))
        vector = schema.encode(action)
        assert vector[-1] == pytest.approx(normalized)
        assert schema.decode(vector).parameters == pytest.approx(action.parameters)
    for z in (.691, .773, .812):
        with pytest.raises(ValueError, match='Position z'):
            schema.encode(PrimitiveAction('grasp', dict(x=x, y=y, z=z, orientation_id=0)))
    contract = action_contract(configured)
    assert contract['pick_place_grid']['grasp_z_root_m'] == pytest.approx([.692, .772])
    assert contract['pick_place']['tcp_bounds_root_m']['grasp'][2] == pytest.approx([.692, .772])


def test_compact_layout_validation_and_config_wiring(monkeypatch):
    from pathlib import Path
    from omegaconf import OmegaConf
    from exploration.cee_us.cee_us_cabinet import make_env
    import main

    with pytest.raises(ValueError, match='Unknown cabinet grid layout'):
        skill_schema(settings(grid_layout='bad'))
    with pytest.raises(ValueError, match='requires pick_place_grid'):
        skill_schema(settings(pick_place_grid=False, grid_layout='compact_3x4'))
    sentinel = object()
    monkeypatch.setattr(main, 'build_env', lambda cfg, raw_env=False: sentinel)
    env, cfg = make_env(cabinet_grid=True, cabinet_grid_layout='compact_3x4')
    assert env is sentinel and cfg.cabinet_grid_layout == 'compact_3x4'
    base = Path('configs')
    for name in ('grasp_grid_compact', 'grasp_grid_compact_fast', 'pick_place_grid_compact'):
        cfg = OmegaConf.load(base / f'raps_gnn_actor_cabinet_{name}.yaml')
        assert cfg.raps_actor.pick_place_grid
        assert cfg.raps_actor.grid_layout == 'compact_3x4'
        assert skill_schema(cfg.raps_actor).action_dim == (73 if name.startswith('pick_place') else 61)


@pytest.mark.parametrize('arm_name', ['curiosity_heuristic', 'curiosity_only', 'heuristic_only', 'random'])
def test_cup_trapezoid_four_arm_wm_actor_update_and_restore(tmp_path, arm_name):
    from pathlib import Path
    from omegaconf import OmegaConf
    from scripts.launch_raps_comparison import arm_configs

    torch.set_num_threads(1)
    base = OmegaConf.to_container(OmegaConf.load(
        Path('configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml')), resolve=True)
    config = arm_configs(base, include_heuristic_only=True)[arm_name]
    assert config['cabinet_item_layout'] == 'single_upper_second_row'
    configured = {**config['raps_actor'], 'hidden_size': 8, 'actor_layers': 1, 'critic_layers': 1,
                  'imagination_horizon': 2, 'imagination_batch_size': 4, 'updates_per_round': 1}
    schema = skill_schema(configured)
    assert schema.action_dim == 31 and schema.grasp_orientation_ids == (0, 1, 2)
    model = CEEUSBackend(agent_dim=33, num_objects=1, object_names=('cabinet_item_16',),
                        action_dim=31, hidden_dim=8, batch_size=4,
                        object_heuristic=config['object_heuristic'])
    actor = RAPSImaginationActor(model, configured)
    assert model.state_dim == 50 and model.dynamic_dim == 45
    rng = np.random.default_rng(19)
    states = rng.normal(size=(8, 50)).astype(np.float32)
    states[:, 45:] = [1., 0., .068, .097, .14]
    actions = canonical_action(torch.tensor(rng.uniform(-1, 1, (8, 31)), dtype=torch.float32), schema).numpy()
    next_states = states.copy()
    next_states[:, :model.dynamic_dim] += .001
    data = dict(states=states, actions=actions, next_states=next_states, terminated=np.zeros(8, bool))
    before = [parameter.detach().clone() for parameter in model.model.model.parameters()]
    model.train(data, 1)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.model.model.parameters()))
    metrics = actor.train(data)
    assert metrics['updates'] == 1 and np.isfinite(list(metrics.values())).all()
    assert actor.imagine(torch.tensor(states))['actions'].shape == (2, 8, 31)
    collector = SkillCollector.__new__(SkillCollector)
    collector.schema = schema
    for policy in ('random', 'curiosity'):
        collector.actor = actor
        collector.state = states[0]
        for _ in range(8):
            vector = collector._propose_action(policy)
            decoded = schema.decode(vector)
            assert decoded.parameters['orientation_id'] in (0, 1, 2)
            assert .692 <= decoded.parameters['z'] <= .772
    encoder = SimpleNamespace(contract=lambda: {'dimension': 50, 'object_names': ['cabinet_item_16']})
    payload = dict(schema=SCHEMA, backend=deepcopy(model.state_dict()), actor=deepcopy(actor.state_dict()),
                   state_contract=encoder.contract(), action_contract=action_contract(actor.settings), rng=rng_state())
    path = tmp_path / 'cup.pt'
    torch.save(payload, path)
    restored = RAPSImaginationActor(model, configured)
    load(path, model, restored, encoder)
    torch.testing.assert_close(restored.state_dict()['actor'], actor.state_dict()['actor'])
    incompatible = SimpleNamespace(contract=lambda: {'dimension': 67, 'object_names': ['cabinet_item_16', 'cabinet_item_17']})
    with pytest.raises(ValueError, match='state/action/controller contract differs'):
        load(path, model, restored, incompatible)


def test_trapezoid_physical_validation_includes_front_cells():
    schema = skill_schema(settings(grid_layout='trapezoid_4_3_2_1', grasp_only=True,
                                   grasp_orientation_candidates=True))
    physical = schema.physical_action_schema()
    for x, y in schema.grid_xy_root_m:
        for z in (.692, .772):
            physical.encode(PrimitiveAction('grasp', dict(x=x, y=y, z=z)))
    with pytest.raises(ValueError, match='Position x'):
        physical.encode(PrimitiveAction('grasp', dict(x=.30, y=0., z=.732)))


@pytest.mark.skipif(__import__('os').environ.get('RUN_NATIVE_CABINET') != '1', reason='native robotics runtime required')
def test_native_cup_trapezoid_collector_to_learning(tmp_path):
    from omegaconf import OmegaConf
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from exploration.raps.raps_gnn_actor import skill_arrays, save

    torch.set_num_threads(1)
    config = OmegaConf.load('configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml')
    # This test explicitly exercises the original single-environment collector.
    config.pop('cpu_sampling', None)
    config.device = 'cpu'
    config.raps_actor.update(dict(hidden_size=8, actor_layers=1, critic_layers=1,
        imagination_horizon=2, imagination_batch_size=4, updates_per_round=1))
    env, _ = make_env(episode_steps=config.episode_steps, cabinet_item_layout=config.cabinet_item_layout,
        cabinet_grid=True, cabinet_grid_layout=config.raps_actor.grid_layout,
        cabinet_door=config.cabinet_door, centered_initial_tcp=config.centered_initial_tcp,
        action_mode='joint_position', low_level_episode_steps=config.episode_steps)
    try:
        env.reset(seed=0)
        encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
        model = CEEUSBackend(agent_dim=33, num_objects=1, object_names=encoder.object_names,
            action_dim=31, hidden_dim=8, batch_size=4,
            object_heuristic=OmegaConf.to_container(config.object_heuristic))
        actor = RAPSImaginationActor(model, OmegaConf.to_container(config.raps_actor))
        collector = SkillCollector(env, encoder, actor, tmp_path / 'native', collection_optimizations=True)
        assert actor.settings['grasp_preflight'] is False
        def unexpected_preflight(*args, **kwargs):
            raise AssertionError('Current cup protocol must not preflight or resample actions')
        collector.execution.preflight_grasp = unexpected_preflight
        # Script only this bounded interface check; formal training samples the actor/random policy.
        target = actor.schema.encode(PrimitiveAction('grasp', dict(x=.410, y=0., z=.732, orientation_id=2)))
        collector._propose_action = lambda policy: target.copy()
        finished = []
        ticks, skills = collector.collect(800, 'random', seed=0, stage='interface_smoke',
            on_skill_end=finished.append, stop_when=lambda: len(finished) >= 2)
        assert len(ticks) <= 800 and len(skills) == 2
        assert all(row['trainable'] for row in skills)
        assert not (collector.directory / 'preflight.jsonl').exists()
        data = skill_arrays(skills)
        assert data['states'].shape == (2, 50) and data['actions'].shape == (2, 31)
        model.train(data, 1)
        metrics = actor.train(data)
        assert np.isfinite(list(metrics.values())).all()
        path = tmp_path / 'native.pt'
        save(path, model, actor, encoder, config, ticks, skills)
        load(path, model, actor, encoder)
        print('NATIVE_CUP_SMOKE', {'controls': len(ticks), 'statuses': [row['status'] for row in skills],
                                  'state_dim': 50, 'action_dim': 31, 'wm_and_actor_updated': True})
    finally:
        env.close()


def test_cup_trapezoid_does_not_reuse_previous_episode_contacts():
    from domain.robocasa.primitives import PrimitiveContext
    from domain.robocasa.motion_planning import CabinetMotionPlanner
    context = PrimitiveContext.__new__(PrimitiveContext)
    context.env = SimpleNamespace(cabinet_item_layout='single_upper_second_row',
        cabinet_grid_layout='trapezoid_4_3_2_1', _contact_report_current=False)
    context.agent = SimpleNamespace(is_grasping=lambda actor: torch.tensor([True]))
    cup = SimpleNamespace(name='cup')
    context.objects = {'cup': SimpleNamespace(actor=cup)}
    context.env.agent = context.agent
    context.env.cabinet_items = [cup]
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.env = context.env
    assert context.finger_contact_force('cup') == (0., 0.)
    assert not context.is_grasping('cup')
    assert planner._grasp_contacts() == {'cup': False}
    context.env._contact_report_current = True
    assert context.is_grasping('cup')
    assert planner._grasp_contacts() == {'cup': True}
    context.env._contact_report_current = False
    context.env._grasp_binding = SimpleNamespace(bound_object_id='cup')
    assert context.is_grasping('cup')
    del context.env._grasp_binding
    del context.env._contact_report_current
    assert context.is_grasping('cup')  # Preserve historical protocol behavior.
