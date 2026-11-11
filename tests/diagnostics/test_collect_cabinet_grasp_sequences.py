import argparse
import json
from copy import deepcopy

import numpy as np
import pytest

from exploration.cee_us.cee_us_state import OBJECT_NAMES
from exploration.diagnostics import collect_cabinet_grasp_sequences as c
from exploration.raps.raps_gnn_actor import DEFAULTS


def state():
    s = np.zeros(67, dtype=np.float32)
    s[18:21] = [.35, 0., .72]
    s[33:36] = [.455, -.06, .744]
    s[45:48] = [.455, .06, .712]
    return s


def provider(template):
    case = next(x for x in c.cases() if x['base_case_id'] == 27 and x['template'] == template)
    x,y,z = case['offsets']
    args = argparse.Namespace(x_offset=x, cup_y_offset=y, bowl_y_offset=y,
                              cup_z_offset=z, bowl_z_offset=z, lift_distance=case['lift_distance'])
    return c.SequenceProvider(dict(DEFAULTS, skill_set='compact6', latched_gripper=True,
                                   grasp_stop_on_contact=False), case, args)


def act(p, s):
    return p.schema.decode(p.action(s))


def tick(p, s, held=True):
    objects = {name: dict(position_root=s[33+12*i:36+12*i].tolist()) for i,name in enumerate(OBJECT_NAMES)}
    snapshot = dict(objects=objects, ee_position=s[18:21].tolist(), failed_object_ids=[])
    row = dict(step=len(p.trace)+1, skill_id=len(p.commands), action=[0,0,0,-1],
               contact_events=[dict(objects={name:dict(grasped=held if name == p.target else False)
                                            for name in OBJECT_NAMES})],
               before=deepcopy(snapshot), after=deepcopy(snapshot), terminated=False, truncated=False)
    p.on_step(row)


def complete_at_goal(p, s):
    cmd = p.commands[-1]
    if 'requested_goal_root' in cmd:
        s[18:21] = cmd['requested_goal_root']
    p.on_end(dict(skill_id=len(p.commands), next_state=s.tolist()))


def test_grid_and_pairing():
    all_cases = c.cases()
    assert len(all_cases) == 972
    assert [a['case_id'] for a in all_cases] == list(range(972))
    for base_id in range(108):
        group = all_cases[9*base_id:9*(base_id+1)]
        assert [x['template'] for x in group] == list(c.TEMPLATES)
        assert {x['seed'] for x in group} == {1000000+base_id}
        assert {x['base_case_id'] for x in group} == {base_id}
    smoke = c.selected_cases('smoke', 1000000)
    assert len(smoke) == 8
    assert {x['base_case_id'] for x in smoke} == {27, 81}
    assert {x['template'] for x in smoke} == {'yzx', 'post_grasp', 'mid_lift', 'pre_carry'}


@pytest.mark.parametrize('template', c.TEMPLATES[:6])
def test_all_axis_orders_freeze_goal(template):
    p, s = provider(template), state()
    assert act(p,s).primitive_type == 'open_gripper'
    origin = s[18:21].astype(float).copy()
    desired = s[33:36].astype(float) + [-.005, 0, .006]
    for index, axis in enumerate(template):
        decoded = act(p,s)
        assert decoded.primitive_type == 'move_delta_ee_pose'
        expected = origin.copy()
        for applied in template[:index+1]: expected['xyz'.index(applied)] = desired['xyz'.index(applied)]
        np.testing.assert_allclose(p.commands[-1]['requested_goal_root'], expected, atol=1e-7)
        complete_at_goal(p, s)
        s[33:36] += .01  # Later object movement must not alter the goal.
    assert act(p,s).primitive_type == 'close_gripper'
    assert p.phase == 'grasp'
    np.testing.assert_allclose(p.initial['grasp_goal_root'], desired, atol=1e-7)
    assert all(cmd['goal_within_primitive_tolerance'] for cmd in p.commands[1:4])


@pytest.mark.parametrize('template', c.TEMPLATES[6:])
def test_insertion_order_and_amplitude(template):
    p, s = provider(template), state()
    act(p,s)
    assert act(p,s).primitive_type == 'approach'
    p.streak = 4
    act(p,s)
    p.streak = 4
    if template == 'mid_lift':
        assert p.phase == 'lift_half'
        assert p.commands[-1]['parameters']['z_dist'] == .04
        complete_at_goal(p,s)
        act(p,s)
    elif template == 'pre_carry':
        assert p.phase == 'lift'
        p.max_rise_held = p.max_event_rise = .02
        complete_at_goal(p,s)
        act(p,s)
    assert p.phase == 'insert_out'
    anchor = s[18:21].copy()
    np.testing.assert_allclose(p.commands[-1]['parameters']['dy'], .03, atol=1e-7)
    complete_at_goal(p,s)
    act(p,s)
    assert p.phase == 'insert_return'
    np.testing.assert_allclose(p.commands[-1]['requested_goal_root'], anchor, atol=1e-7)
    complete_at_goal(p,s)
    act(p,s)
    assert p.phase == {'post_grasp':'lift', 'mid_lift':'lift_remaining', 'pre_carry':'carry'}[template]
    if template == 'mid_lift':
        assert p.commands[-1]['parameters']['z_dist'] == .04
        assert not p.lift_valid
        complete_at_goal(p,s)
        assert 'full_lift_goal_z_error_m' in p.commands[-1]
        p.max_rise_held = p.max_event_rise = .02
        act(p,s)
        assert p.phase == 'carry'
        assert p.lift_valid


@pytest.mark.parametrize('phase', ['insert_out', 'insert_return', 'lift_half', 'lift_remaining'])
def test_grasp_loss_never_regrasped(phase):
    p, s = provider('mid_lift'), state()
    act(p,s); act(p,s); p.streak = 4; act(p,s)
    if phase != 'lift_half':
        p.streak = 4; act(p,s)
    if phase in ('insert_return', 'lift_remaining'):
        p.streak = 4; act(p,s)
    if phase == 'lift_remaining':
        p.streak = 4; act(p,s)
    assert p.phase == phase
    tick(p,s,held=False)
    for _ in range(5): tick(p,s,held=True)  # Recovery cannot erase an observed loss.
    assert act(p,s).primitive_type == 'open_gripper'
    assert p.phase == 'park'
    assert p.failure == ('lost_grasp_during_insertion' if phase.startswith('insert') else 'lost_grasp_during_lift')


def test_export_and_resume_use_distinct_attempts(tmp_path, monkeypatch):
    plan = c.selected_cases('smoke', 1000000)[:2]
    out = tmp_path/'output'; out.mkdir()
    manifest = dict(cases=plan, source_sha256={})
    (out/'manifest.json').write_text(json.dumps(manifest))
    def add(case, attempt, completed):
        directory = out/'cases'/f"case_{case['case_id']:03d}"/f'attempt_{attempt:06d}'
        directory.mkdir(parents=True)
        row = dict(episode_id=1, completed=completed, duration_low_steps=4,
                   state=[0]*67, next_state=[1]*67, action=[0]*12, terminated=False, truncated=False)
        (directory/'skills.jsonl').write_text(json.dumps(row)+'\n')
    add(plan[0], 1, True)
    case_dir = out/'cases'/f"case_{plan[0]['case_id']:03d}"
    (case_dir/'summary.json').write_text(json.dumps(dict(case=plan[0], success=False, report=dict(failure='failed'))))
    add(plan[1], 1, True)
    monkeypatch.setattr(c, 'checkpoint_contract', lambda _: ({}, None, dict(DEFAULTS, skill_set='compact6')))
    monkeypatch.setattr(c, 'identity', lambda *args: manifest)
    called = []
    def run(output, case, payload, settings):
        called.append(case['case_id']); add(case, 2, True)
        path = output/'cases'/f"case_{case['case_id']:03d}"/'summary.json'
        path.write_text(json.dumps(dict(case=case, success=True, report=dict(failure=None))))
    monkeypatch.setattr(c, 'run_case', run)
    c.main(['--checkpoint', 'unused', '--output', str(out), '--stage', 'smoke', '--resume'])
    assert called == [plan[1]['case_id']]
    with np.load(out/'evaluation.npz') as data:
        assert len(set(data['episode_ids'])) == 3
        assert data['actions'].shape == (3, 12)
    monkeypatch.setattr(c, 'identity', lambda *args: {})
    with pytest.raises(ValueError, match='Resume manifest'):
        c.main(['--checkpoint', 'unused', '--output', str(out), '--stage', 'smoke', '--resume'])


def split_provider(template):
    case = next(x for x in c.cases() if x['base_case_id'] == 27 and x['template'] == template)
    x,y,z = case['offsets']
    args = argparse.Namespace(x_offset=x, cup_y_offset=y, bowl_y_offset=y,
                              cup_z_offset=z, bowl_z_offset=z, lift_distance=case['lift_distance'])
    return c.SplitSequenceProvider(dict(DEFAULTS, skill_set='split7', latched_gripper=True,
                                       grasp_stop_on_contact=False), case, args)


@pytest.mark.parametrize('template', c.TEMPLATES[:6])
def test_split_axis_waypoints_use_explicit_primitives(template):
    p, s = split_provider(template), state()
    assert len(p.action(s)) == 13
    desired = s[33:36].astype(float)+[-.005, 0, .006]
    for axis in template:
        decoded = act(p,s)
        assert decoded.primitive_type == ('approach_x' if axis == 'x' else 'align_yz')
        assert len(p.schema.encode(decoded)) == 13
        assert p.commands[-1]['requested_goal_root']['xyz'.index(axis)] == pytest.approx(desired['xyz'.index(axis)])
        complete_at_goal(p,s)
    assert act(p,s).primitive_type == 'close_gripper'
    assert p.phase == 'grasp'
    assert all(command['primitive'] != 'approach' for command in p.commands)


@pytest.mark.parametrize('template', c.TEMPLATES[6:])
def test_split_insert_templates_use_align_approach_close(template):
    p, s = split_provider(template), state()
    assert act(p,s).primitive_type == 'open_gripper'
    assert act(p,s).primitive_type == 'align_yz'
    assert p.phase == 'split_align'
    goal = p.initial['grasp_goal_root'].copy()
    complete_at_goal(p,s)
    s[33:36] += .01
    assert act(p,s).primitive_type == 'approach_x'
    assert p.phase == 'split_approach'
    assert p.commands[-1]['requested_goal_root'][0] == pytest.approx(goal[0])
    complete_at_goal(p,s)
    assert act(p,s).primitive_type == 'close_gripper'
    assert p.phase == 'grasp'
    p.streak = 4
    act(p,s)
    assert p.phase == ('insert_out' if template == 'post_grasp' else 'lift_half' if template == 'mid_lift' else 'lift')


def test_split_nonpositive_approach_fails_and_opens():
    p, s = split_provider('post_grasp'), state()
    act(p,s); act(p,s)
    s[18] = p.initial['grasp_goal_root'][0] + .01
    assert act(p,s).primitive_type == 'open_gripper'
    assert p.phase == 'park' and p.failure == 'approach_x_nonpositive_or_out_of_range'


def test_split_align_tolerance_and_goal_errors():
    p, s = split_provider('post_grasp'), state()
    act(p,s); act(p,s)
    cmd = p.commands[-1]
    s[18:21] = np.asarray(cmd['requested_goal_root'])+[0,.014,0]
    p.on_end(dict(skill_id=2, next_state=s.tolist()))
    assert cmd['goal_within_primitive_tolerance']
    assert cmd['goal_error_m'] == pytest.approx(.014, abs=1e-7)


def test_split_plan_exposes_contract_and_no_sim(tmp_path, monkeypatch, capsys):
    def forbidden(*a, **kw): raise AssertionError('simulator not allowed')
    monkeypatch.setattr(c, 'checkpoint_contract', forbidden)
    output = tmp_path/'plan'
    c.main(['--checkpoint','unused','--output',str(output),'--skill-set','split7'])
    plan = json.loads((output/'plan.json').read_text())
    assert plan['skill_set'] == 'split7' and plan['action_dim'] == 13
    assert 'align_yz' in plan['primitive_names'] and 'approach' not in plan['primitive_names']
    capsys.readouterr()


def test_split_identity_source_and_effective_contract(tmp_path):
    from omegaconf import OmegaConf
    from exploration.raps.raps_gnn_actor import action_contract
    settings = dict(DEFAULTS, skill_set='compact6', latched_gripper=True, grasp_stop_on_contact=False)
    checkpoint = tmp_path/'checkpoint'; checkpoint.write_bytes(b'checkpoint')
    config = OmegaConf.create(dict(sim_freq=100, control_freq=20, raps_actor=settings))
    payload = dict(state_contract={'dimension':67}, action_contract=action_contract(settings))
    args = argparse.Namespace(checkpoint=checkpoint, stage='smoke', seed=1000000, skill_set='split7')
    manifest = c.identity(args, payload, config, settings)
    assert manifest['source_checkpoint_action_contract']['action_dim'] == 12
    assert manifest['action_contract']['action_dim'] == 13
    assert manifest['source_checkpoint_config']['raps_actor']['skill_set'] == 'compact6'
    assert manifest['config']['raps_actor']['skill_set'] == 'split7'
    assert manifest['checkpoint_weights_used'] is False
    args.skill_set = 'compact6'
    assert c.identity(args, payload, config, settings) != manifest
