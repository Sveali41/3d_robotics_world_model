from copy import deepcopy

import numpy as np
import pytest

from exploration.diagnostics.calibrate_wm_validation import CalibrationProvider, CASES, case_jobs, compare_raw, instrument_ik, parser
from exploration.cee_us.cee_us_state import OBJECT_NAMES
from exploration.raps.raps_gnn_actor import DEFAULTS


def arguments(*extra):
    return parser().parse_args(['--checkpoint', 'dummy.pt', '--output', 'dummy', '--mode', 'serial', *extra])


def state(shift=0.):
    s = np.zeros(67, dtype=np.float32)
    s[18:21] = [.35, 0., .72]
    s[33:36] = [.455+shift, -.06, .744]
    s[45:48] = [.455+shift, .06, .712]
    return s


def provider(case='original_cup'):
    return CalibrationProvider(dict(DEFAULTS, skill_set='compact6', latched_gripper=True,
                                    grasp_stop_on_contact=False), case, arguments())


def command(p, s):
    return p.schema.decode(p.action(s))


def ticks(p, s, n=4, held=True, closed=True, rise=0., dx=0.):
    index = OBJECT_NAMES.index(p.target)
    objects = {name: dict(position_root=s[33+12*i:36+12*i].tolist(),
                         position_world=s[33+12*i:36+12*i].tolist()) for i,name in enumerate(OBJECT_NAMES)}
    before = dict(objects=deepcopy(objects), ee_position=s[18:21].tolist(), failed_object_ids=[])
    for _ in range(n):
        after = deepcopy(before)
        after['objects'][p.target]['position_root'][2] += rise
        after['objects'][p.target]['position_root'][0] += dx
        row = dict(step=len(p.trace)+1, skill_id=len(p.commands), action=[0,0,0,-1 if closed else 1],
                   contact_events=[dict(objects={name:dict(grasped=held if name==p.target else False)
                                                for name in OBJECT_NAMES})],
                   before=before, after=after, terminated=False, truncated=False)
        p.on_step(row)
    s[33+12*index:36+12*index] = after['objects'][p.target]['position_root']


@pytest.mark.parametrize('case', CASES)
def test_direct_reset_grasp_goals_and_settings(case):
    p, s = provider(case), state(.03 if case.startswith('shifted') else 0.)
    saved = deepcopy(p.settings)
    assert command(p,s).primitive_type == 'open_gripper'
    grasp = command(p,s)
    assert grasp.primitive_type == 'approach'
    bowl = case.endswith('bowl')
    target = s[45:48] if bowl else s[33:36]
    expected = target + np.array([-.005, -.011 if bowl else 0., .028 if bowl else .006])-s[18:21]
    np.testing.assert_allclose(list(grasp.parameters.values()), expected, atol=1e-7)
    assert p.settings == saved
    assert [c['primitive'] for c in p.commands] == ['open_gripper','approach']


def test_grasp_failure_allows_one_explicit_closure_and_denies_lift():
    p,s = provider(),state()
    command(p,s); command(p,s)
    ticks(p,s,held=False)
    assert command(p,s).primitive_type == 'close_gripper'
    ticks(p,s,held=False)
    assert command(p,s).primitive_type == 'open_gripper'
    assert p.failure == 'grasp_not_confirmed'
    assert not any(c['primitive']=='lift' for c in p.commands)


def test_only_current_closed_four_tick_grasp_gates_lift():
    p,s = provider(),state()
    command(p,s); command(p,s)
    ticks(p,s,n=4,closed=False)
    assert p.streak == 0
    assert command(p,s).primitive_type == 'close_gripper'
    ticks(p,s,n=3)
    ticks(p,s,n=1,held=False)
    assert p.streak == 0
    assert command(p,s).primitive_type == 'open_gripper'


@pytest.mark.parametrize('rise,lost,expected', [(0.,False,'lift_not_achieved'),(.02,True,'lost_grasp_during_lift')])
def test_lift_command_without_actual_held_motion_denies_carry(rise,lost,expected):
    p,s = provider(),state()
    command(p,s); command(p,s); ticks(p,s)
    assert command(p,s).primitive_type == 'lift'
    ticks(p,s,n=4,rise=rise,held=not lost)
    assert command(p,s).primitive_type == 'open_gripper'
    assert p.failure == expected
    assert not p.lift_valid
    assert not any(c['phase']=='carry' for c in p.commands)


def test_actual_path_requires_carry_motion_drop_and_release():
    p,s = provider(),state()
    command(p,s); command(p,s); ticks(p,s)
    assert command(p,s).primitive_type == 'lift'
    ticks(p,s,rise=.02)
    carry = command(p,s)
    assert p.lift_valid
    assert carry.primitive_type == 'move_delta_ee_pose'
    assert carry.parameters['dx'] == pytest.approx(-.03)
    ticks(p,s,dx=-.025)
    assert command(p,s).primitive_type == 'drop'
    ticks(p,s)
    assert command(p,s).primitive_type == 'open_gripper'
    ticks(p,s,held=False,closed=False)
    assert p.released
    assert command(p,s).parameters['dx'] == pytest.approx(-.05)
    ticks(p,s,held=False,closed=False)
    command(p,s)
    assert p.report([])['calibration_success']


def test_out_of_range_goal_fails_without_clipping():
    p,s = provider(),state()
    command(p,s)
    s[33] = .2
    with pytest.raises(ValueError, match='strictly positive'):
        command(p,s)


def test_case_seeds_and_output_isolation_equal_across_modes():
    serial,parallel = arguments(),arguments('--mode','parallel')
    a,b = case_jobs(serial),case_jobs(parallel)
    assert [(j['case'],j['seed'],j['output']) for j in a] == [(j['case'],j['seed'],j['output']) for j in b]
    assert len({j['output'] for j in a}) == 4
    one = case_jobs(arguments('--case','shifted_bowl'))
    assert one[0]['seed'] == a[3]['seed']


def test_raw_comparison_reports_float32_difference(tmp_path):
    import json
    a,b = tmp_path/'a',tmp_path/'b'
    a.mkdir(); b.mkdir()
    row = dict(state=[0.],action=[1.],next_state=[2.])
    for filename in ('transitions.jsonl','skills.jsonl'):
        (a/filename).write_text(json.dumps(row)+'\n')
        (b/filename).write_text(json.dumps(dict(row,next_state=[3.]))+'\n')
    comparison = compare_raw(a,b)
    assert comparison['transitions.jsonl']['fields']['state']['exact']
    assert comparison['transitions.jsonl']['fields']['next_state'] == dict(exact=False,max_abs_diff=1.)


def test_optional_ik_observer_preserves_inputs_and_exact_return():
    from types import SimpleNamespace
    output = object()
    seen = []

    def solve(*args, **kwargs):
        seen.append((args, kwargs))
        return kwargs['result']

    kinematics = SimpleNamespace(compute_ik=solve)
    calls = []
    original = instrument_ik(kinematics, calls)
    assert kinematics.compute_ik('pose', result=output) is output
    assert kinematics.compute_ik('pose', result=None) is None
    assert calls == [False, True]
    assert seen == [(('pose',), dict(result=output)), (('pose',), dict(result=None))]
    kinematics.compute_ik = original
    assert not arguments().diagnose_contacts
