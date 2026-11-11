import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exploration.diagnostics import collect_cabinet_grasps as c


def test_plan_order_and_smoke():
    cases = c.cases()
    assert len(cases) == 108
    assert cases[0] == dict(case_id=0, object='cup', delta=[-.01, -.015, -.015],
                            lift_distance=.04, offsets=[-.015, -.015, -.009], seed=1000000)
    assert cases[-1]['object'] == 'bowl'
    assert cases[-1]['seed'] == 1000107
    assert [x['case_id'] for x in c.selected_cases('smoke', 1000000)] == [27, 81]
    assert c.selected_cases('smoke', 1000000)[1]['offsets'] == [-.005, -.011, .028]
    assert {v for x in cases for v in [x['lift_distance']]} == {.04, .08}


def test_stop_at_retreat_cleanup_or_episode_end():
    calls = []
    provider = SimpleNamespace(phase='retreat', failure=None, action=lambda state: calls.append(state))
    collector = SimpleNamespace(state='state')
    stop = c.stop_callback(provider, collector)
    skill = dict(completed=False, terminated=False, truncated=False)
    stop(skill)
    skill['completed'] = True
    with pytest.raises(c.Finished): stop(skill)
    assert calls == ['state']
    provider.phase, provider.failure = 'park', 'grasp_not_confirmed'
    with pytest.raises(c.Finished): stop(skill)
    provider.phase = 'grasp'
    skill.update(completed=False, terminated=True)
    with pytest.raises(c.Finished): stop(skill)


def row(completed, duration, episode=1):
    return dict(state=[0]*67, next_state=[1]*67, action=[0]*12, episode_id=episode,
                completed=completed, duration_low_steps=duration, terminated=False, truncated=False)


def write_attempt(root, case, attempt, rows):
    path = root/'cases'/f'case_{case:03d}'/f'attempt_{attempt:06d}'
    path.mkdir(parents=True)
    (path/'skills.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_export_partial_attempts_and_zero_rows(tmp_path):
    assert c.export(tmp_path) == 0
    with np.load(tmp_path/'evaluation.npz') as data:
        assert data['states'].shape == (0, 67)
        assert data['actions'].shape == (0, 12)
    write_attempt(tmp_path, 0, 1, [row(True, 3), row(False, 4), row(True, 0)])
    write_attempt(tmp_path, 0, 2, [row(True, 2)])
    write_attempt(tmp_path, 1, 1, [row(True, 1)])
    assert c.export(tmp_path) == 3
    with np.load(tmp_path/'evaluation.npz') as data:
        assert data['episode_ids'].tolist() == [1, 2, 1000001]
        assert data['duration_low_steps'].tolist() == [3, 2, 1]
        assert data['states'].shape == (3, 67)
    write_attempt(tmp_path, 2, 1, [row(True, 1, episode=2)])
    with pytest.raises(ValueError, match='Unexpected episode'): c.export(tmp_path)


def test_results_counts_closed_failures(tmp_path):
    plan = c.selected_cases('smoke', 1000000)
    for case, success in zip(plan, [True, False]):
        directory = tmp_path/'cases'/f"case_{case['case_id']:03d}"
        directory.mkdir(parents=True)
        (directory/'summary.json').write_text(json.dumps(dict(case=case, success=success)))
    result = c.results(tmp_path, plan)
    assert result['total']['closed'] == 2
    assert result['total']['success_rate'] == .5
    assert result['total']['failure_reasons'] == {'unknown': 1}
    assert result['per_object']['bowl']['success_rate'] == 0
    assert result['per_point'][0]['closed'] == 1


def test_resume_skips_closed_and_restarts_open(tmp_path, monkeypatch):
    output = tmp_path/'out'
    output.mkdir()
    plan = c.selected_cases('smoke', 1000000)
    manifest = dict(cases=plan, source_sha256={})
    (output/'manifest.json').write_text(json.dumps(manifest))
    closed = output/'cases'/f"case_{plan[0]['case_id']:03d}"
    closed.mkdir(parents=True)
    (closed/'summary.json').write_text(json.dumps(dict(case=plan[0], success=False)))
    write_attempt(output, plan[1]['case_id'], 1, [row(True, 5)])
    monkeypatch.setattr(c, 'checkpoint_contract', lambda p: ({}, None, {}))
    monkeypatch.setattr(c, 'identity', lambda *a: manifest)
    called = []
    def run(out, case, payload, settings):
        called.append(case['case_id'])
        write_attempt(out, case['case_id'], 2, [row(True, 3)])
        (out/'cases'/f"case_{case['case_id']:03d}"/'summary.json').write_text(json.dumps(dict(case=case, success=True)))
    monkeypatch.setattr(c, 'run_case', run)
    c.main(['--checkpoint', 'unused', '--output', str(output), '--stage', 'smoke', '--resume'])
    assert called == [81]
    with np.load(output/'evaluation.npz') as data:
        assert data['episode_ids'].tolist() == [81000001, 81000002]
    monkeypatch.setattr(c, 'identity', lambda *a: {'cases': []})
    with pytest.raises(ValueError, match='Resume manifest'): c.main(['--checkpoint', 'unused', '--output', str(output), '--stage', 'smoke', '--resume'])


def test_truncated_jsonl_tail_is_reported_and_internal_corruption_fails(tmp_path):
    path = tmp_path/'skills.jsonl'
    path.write_text(json.dumps(row(True, 2))+'\n{"partial":')
    omitted = []
    assert len(c.read_rows(path, omitted)) == 1
    assert omitted == [str(path)]
    path.write_text('{bad}\n'+json.dumps(row(True, 2))+'\n')
    with pytest.raises(ValueError, match='Corrupt JSONL'): c.read_rows(path)


def test_all_grid_cases_have_unique_sequential_ids_and_seeds():
    grid = c.cases()
    assert len({(case['object'], tuple(case['delta']), case['lift_distance']) for case in grid}) == 108
    assert [case['case_id'] for case in grid] == list(range(108))
    assert [case['seed'] for case in grid] == list(range(1000000, 1000108))


def test_all_grid_cases_encode_valid_calibration_approach():
    from exploration.diagnostics.calibrate_wm_validation import CalibrationProvider
    from exploration.raps.raps_gnn_actor import DEFAULTS

    settings = dict(DEFAULTS, skill_set='compact6', latched_gripper=True, grasp_stop_on_contact=False)
    state = np.zeros(67, dtype=np.float32)
    state[18:21] = [.35, 0., .72]
    state[33:36] = [.455, -.06, .744]
    state[45:48] = [.455, .06, .712]
    for case in c.cases():
        nominal = np.array([-.005, 0., .006] if case['object'] == 'cup' else [-.005, -.011, .028])
        offsets = nominal + case['delta']
        np.testing.assert_allclose(case['offsets'], offsets, atol=1e-12)
        x, y, z = case['offsets']
        args = argparse.Namespace(x_offset=x, cup_y_offset=y, bowl_y_offset=y,
                                  cup_z_offset=z, bowl_z_offset=z, lift_distance=case['lift_distance'])
        provider = CalibrationProvider(settings, 'original_'+case['object'], args)
        assert provider.schema.decode(provider.action(state)).primitive_type == 'open_gripper'
        encoded = provider.action(state)
        assert np.isfinite(encoded).all()
        approach = provider.schema.decode(encoded)
        assert approach.primitive_type == 'approach'
        target = state[33:36] if case['object'] == 'cup' else state[45:48]
        expected = target + offsets - state[18:21]
        np.testing.assert_allclose(list(approach.parameters.values()), expected, atol=1e-7)
        for param in provider.schema.spec_by_name['approach'].parameters:
            assert abs(approach.parameters[param.name]) <= param.scale


def test_plan_stage_never_constructs_environment(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('Plan stage must not access simulator or checkpoint')
    monkeypatch.setattr(c, 'make_env', forbidden)
    monkeypatch.setattr(c, 'checkpoint_contract', forbidden)
    output = tmp_path/'plan'
    assert c.main(['--checkpoint', 'unused', '--output', str(output)]) == 0
    plan = json.loads((output/'plan.json').read_text())
    assert plan['cases_count'] == 108
    assert plan['max_control_ticks'] == 108 * 800
    assert len(json.loads(capsys.readouterr().out)['cases']) == 108


def test_export_thirteen_dimensional_endpoints(tmp_path):
    assert c.export(tmp_path, action_dim=13) == 0
    with np.load(tmp_path/'evaluation.npz') as data:
        assert data['actions'].shape == (0, 13)
    data = row(True, 2); data['action'] = [0]*13
    write_attempt(tmp_path, 0, 1, [data, row(False, 5)])
    assert c.export(tmp_path, action_dim=13) == 1
    with np.load(tmp_path/'evaluation.npz') as saved:
        assert saved['actions'].shape == (1, 13)
    with pytest.raises(ValueError, match='Invalid endpoint dimensions'):
        c.export(tmp_path)
