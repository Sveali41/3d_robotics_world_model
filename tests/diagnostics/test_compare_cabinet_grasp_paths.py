import json
from types import SimpleNamespace

import numpy as np
import pytest

from exploration.diagnostics import compare_cabinet_grasp_paths as c


def test_trajectories_include_initial_positions_and_actual_frame():
    initial = np.zeros(67)
    initial[18:21] = [1, 2, 3]
    initial[33:36] = [4, 5, 6]
    initial[45:48] = [7, 8, 9]
    final = initial.copy()
    final[18:21] += [3, 4, 0]
    final[33:36] += [0, 0, 2]
    paths = c.trajectories([dict(next_state=final.tolist())], initial.tolist())
    assert paths['tcp'].tolist() == [[1, 2, 3], [4, 6, 3]]
    assert paths['cup'][-1].tolist() == [4, 5, 8]
    assert paths['bowl'][0].tolist() == [7, 8, 9]
    assert c.path_length(paths['tcp']) == 5
    assert c.path_length(c.trajectories([], initial)['tcp']) == 0


def test_rejections_with_zero_motion_are_retained():
    state = np.zeros(67)
    state[18:21] = [1, 2, 3]
    skills = [dict(state=state, duration_low_steps=0,
                   motion_planning=[dict(status='no_collision_free_ik', stage='align', target_position_root=[4, 5, 6])])]
    rejected = c.rejected_targets(skills)
    assert rejected[0]['target'] == [4, 5, 6]
    assert rejected[0]['start'].tolist() == [1, 2, 3]
    assert rejected[0]['status'] == 'no_collision_free_ik'


def test_monitor_uses_joint_gripper_without_mutation():
    row = dict(action=[1, 2, 3, 4, 5, 6, 7, -1], state=[0])
    assert c.monitor_row(row)['action'] == [0, 0, 0, -1]
    assert row['action'][3] == 4
    with pytest.raises(ValueError): c.monitor_row(dict(action=[0]*4))


def test_initial_mismatch_and_source_hash_failure(tmp_path, monkeypatch):
    assert c.verify_initial(np.zeros(67), np.zeros(67)) == 0
    with pytest.raises(ValueError, match='Initial state mismatch'):
        c.verify_initial(np.zeros(67), np.ones(67))
    source = tmp_path/'source.py'
    source.write_text('current')
    checkpoint = tmp_path/'checkpoint'
    checkpoint.write_text('checkpoint')
    manifest = dict(stage='smoke', split='validation_only', case_budget=800, physics_hz=100,
                    control_hz=20, object_spacing_m=.12, cases=[dict(case_id=27), dict(case_id=81)],
                    checkpoint=str(checkpoint), checkpoint_sha256=c.sha256(checkpoint),
                    source_sha256={'source.py': 'wrong'})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='source hash mismatch'):
        c.validate_baseline(tmp_path)
    checkpoint.write_text('changed')
    with pytest.raises(ValueError, match='Checkpoint hash'):
        c.validate_baseline(tmp_path)


def test_plotted_stationary_rejection(tmp_path):
    paths = dict(tcp=np.zeros((1, 3)), cup=np.zeros((1, 3)), bowl=np.zeros((1, 3)))
    entry = dict(case=dict(object='cup'), baseline=paths, planner=paths,
                 planner_result=dict(calibration=dict(calibration_success=False, failure='grasp_not_confirmed'),
                                     skill_statuses=[dict(status='no_collision_free_ik')]),
                 rejected_targets=[dict(start=[0, 0, 0], target=[.2, .1, .7], status='no_collision_free_ik')])
    c.draw(tmp_path, [entry, dict(entry, case=dict(object='bowl'))])
    assert (tmp_path/'paths.png').stat().st_size > 0
    assert (tmp_path/'paths.svg').stat().st_size > 0
