import numpy as np
import pytest
from types import SimpleNamespace

from exploration.cee_us.cee_us_state import OBJECT_NAMES
from exploration.diagnostics.collect_primitive_combinations import (
    CombinationProvider, RECIPES, commands_for, generate_sampling_plan,
    sampling_bounds, select_pilot,
)
from exploration.raps.raps_gnn_actor import DEFAULTS


def provider(commands):
    return CombinationProvider({**DEFAULTS, 'skill_set':'split7'}, commands, OBJECT_NAMES[0])


def test_recipes_have_distinct_real_primitive_sequences_and_explicit_closing_actions():
    goal = [.45, -.06, .74]
    sequences = []
    for recipe in RECIPES:
        commands = commands_for(recipe, goal)
        names = [c['primitive'] for c in commands]
        sequences.append(tuple(names))
        assert names[0] == 'open_gripper' and names[-1] == 'lift'
        assert names.count('close_gripper') == 4
        assert not any('quaternion' in c or 'orientation' in c for c in commands)
    assert len(set(sequences)) == len(RECIPES)
    refine = commands_for('align_approach_refine', goal)
    np.testing.assert_allclose(refine[2]['target_root'], [.43, -.06, .74])
    np.testing.assert_allclose(refine[3]['target_root'], goal)
    earlier = commands_for('approach_align_close', goal)
    assert [c['primitive'] for c in earlier[:4]] == ['open_gripper', 'approach_x', 'align_yz', 'approach_x']
    np.testing.assert_allclose(earlier[3]['target_root'], goal)
    assert earlier[4]['primitive'] == 'close_gripper'
    for recipe in RECIPES:
        commands = commands_for(recipe, goal)
        positioning = [c for c in commands if 'target_root' in c]
        np.testing.assert_allclose(positioning[-1]['target_root'], goal)
        assert all(c['primitive'] == 'close_gripper' for c in commands[-5:-1])


def _geometry():
    geometry = {}
    for scene, shift in (('original', np.array([0., 0., 0.])),
                         ('shifted', np.array([.03, -.02, 0.]))):
        geometry[scene] = {}
        for index, name in enumerate(('cabinet_item_16', 'cabinet_item_17')):
            lo = np.array([.35 + index * .1, -.16, .68]) + shift
            hi = lo + np.array([.12, .12, .12])
            pose = [*((lo + hi) / 2).tolist(), 1., 0., 0., 0.]
            geometry[scene][name] = dict(
                collision_bounds_root_m=np.stack((lo, hi)), pose_in_root=pose)
    return geometry


def test_sampling_bounds_pad_xy_inset_z_and_reject_empty_or_nonfinite_ranges():
    bounds = sampling_bounds(np.array([[-.1, -.2, .4], [.1, .2, .8]]))
    np.testing.assert_allclose(bounds, [[-.12, -.22, .41], [.12, .22, .79]])
    with pytest.raises(ValueError):
        sampling_bounds([[0., 0., .5], [1., 1., .5]])
    with pytest.raises(ValueError):
        sampling_bounds([[0., 0., 0.], [1., np.inf, 1.]])


def test_sampling_plan_is_deterministic_bounded_and_paired_across_recipes_and_scenes():
    geometry = _geometry()
    plan = generate_sampling_plan(geometry)
    assert len(plan) == 160
    assert plan == generate_sampling_plan(geometry)
    assert plan != generate_sampling_plan(geometry, seed=4100001)
    assert all(set(('scene', 'object', 'recipe', 'cell', 'cell_fraction',
                    'normalized_position', 'position_root', 'seed', 'trial_id',
                    'pair_id', 'intent')) <= set(row) for row in plan)

    by_key = {(r['scene'], r['object'], r['recipe'], tuple(r['cell'])): r for r in plan}
    for row in plan:
        lo, hi = sampling_bounds(geometry[row['scene']][row['object']]['collision_bounds_root_m'])
        position = np.asarray(row['position_root'])
        assert np.all(position >= lo) and np.all(position <= hi)
        normalized = (position-lo)/(hi-lo)
        assert np.floor(normalized*2).astype(int).tolist() == row['cell']
    for name in ('cabinet_item_16', 'cabinet_item_17'):
        for recipe in RECIPES:
            for cell in sorted({tuple(r['cell']) for r in plan}):
                original = by_key['original', name, recipe, cell]
                shifted = by_key['shifted', name, recipe, cell]
                np.testing.assert_allclose(original['cell_fraction'], shifted['cell_fraction'])
                np.testing.assert_allclose(original['normalized_position'], shifted['normalized_position'])
                np.testing.assert_allclose(
                    np.asarray(shifted['position_root']) - original['position_root'],
                    [.03, -.02, 0.])
                for paired_recipe in RECIPES:
                    paired = by_key['original', name, paired_recipe, cell]
                    np.testing.assert_allclose(original['cell_fraction'], paired['cell_fraction'])
                    np.testing.assert_allclose(original['normalized_position'], paired['normalized_position'])


def test_pilot_selects_low_and_high_cells_for_each_scene_object_pair():
    plan = generate_sampling_plan(_geometry())
    pilot = select_pilot(plan)
    assert len(pilot) == 8
    for scene in ('original', 'shifted'):
        for name in ('cabinet_item_16', 'cabinet_item_17'):
            selected = [r for r in pilot if r['scene'] == scene and r['object'] == name]
            assert len(selected) == 2
            assert {r['recipe'] for r in selected} == {'align_approach', 'approach_align_refine'}
            cells = sorted({tuple(r['cell']) for r in plan})
            assert selected[0]['cell'] == list(cells[0])
            assert selected[1]['cell'] == list(cells[-1])


def test_parameters_use_feedback_and_reject_out_of_range_targets_instead_of_clipping():
    state = np.zeros(67)
    state[18:21] = [.35, 0., .72]
    p = provider([dict(primitive='align_yz', target_root=[.9, -.06, .74])])
    action = p.schema.decode(p.action(state))
    assert action.parameters['y_offset'] == pytest.approx(-.06)
    assert action.parameters['z_offset'] == pytest.approx(.02)
    p = provider([dict(primitive='approach_x', target_root=[.8, 0., .72])])
    with pytest.raises(ValueError, match='exceeds'):
        p.action(state)
    p = provider([dict(primitive='approach_x', target_root=[.2, 0., .72])])
    with pytest.raises(ValueError, match='behind'):
        p.action(state)
    p = provider([dict(primitive='drop', target_root=[.35, 0., .70])])
    action = p.schema.decode(p.action(state))
    assert action.parameters['z_dist'] == pytest.approx(.02)
    state[20] = .71
    p = provider([dict(primitive='drop', target_root=[.35, 0., .70])])
    action = p.schema.decode(p.action(state))
    assert action.parameters['z_dist'] == pytest.approx(.01)
    p = provider([dict(primitive='drop', target_root=[.35, 0., .74])])
    with pytest.raises(ValueError, match='wrong vertical direction'):
        p.action(state)


def test_failure_ends_attempt_before_closure_and_lift():
    p = provider(commands_for('align_approach', [.45, 0., .74]))
    state = np.zeros(67)
    p.action(state)
    p.on_skill_end(dict(completed=True,status='no_collision_free_ik',primitive_type='align_yz',skill_id=2))
    assert p.outcome['reason'] == 'primitive_failed'
    with pytest.raises(RuntimeError, match='ended'):
        p.action(state)


def test_newly_confirmed_payload_keeps_initial_shelf_support_intent_during_closure():
    calls = []
    p = provider([dict(primitive='close_gripper', parameters={})])
    p.audit = SimpleNamespace(raw=SimpleNamespace(motion_planner=SimpleNamespace(
        attachment_state=lambda: dict(name=OBJECT_NAMES[0]))),
        set_stage=lambda *args,**kwargs: calls.append(kwargs))
    p.action(np.zeros(67))
    assert p.audit.payload == OBJECT_NAMES[0]
    assert calls == [dict(allowed_items=OBJECT_NAMES, support=True)]


def test_final_closing_gate_blocks_empty_lift_and_lift_requires_actual_held_rise():
    state = np.zeros(67)
    state[35] = .74
    commands = [dict(primitive='close_gripper', parameters={}) for _ in range(4)]
    commands.append(dict(primitive='lift', parameters=dict(z_dist=.04)))
    skill = dict(completed=True,status='success',primitive_type='close_gripper',skill_id=1)
    p = provider(commands)
    for i in range(4):
        p.action(state)
        p.on_skill_end(skill)
        assert (p.outcome is None) == (i < 3)
    assert p.outcome['reason'] == 'unconfirmed_grasp'
    p = provider(commands)
    p.grasp_streak = 4
    for _ in range(4):
        p.action(state)
        p.on_skill_end(skill)
    p.action(state)
    assert p.lift_start_z == .74
    target = OBJECT_NAMES[0]
    row = dict(parameters_physical=dict(gripper=-1.), after=dict(
        contact_snapshot=dict(objects={target:dict(grasped=True)}),
        objects={target:dict(position_root=[0,0,.76])}))
    p.on_step(row)
    p.on_skill_end(dict(skill,primitive_type='lift'))
    assert p.outcome['status'] == 'grasp_and_lift_verified'
    assert p.max_held_lift_m == pytest.approx(.02)
