from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from exploration.diagnostics.probe_grasp_region import (
    ORIENTATION_RADIUS_DEG, POSITION_RADIUS_M, confirmed_grasp, execution_candidates, local_goals,
)


def test_continuous_local_samples_are_reproducible_bounded_and_keep_height_controls():
    anchor = dict(name='anchor', position_root=[.45, -.06, .74],
                  quaternion_wxyz=[0., 2**-.5, 0., 2**-.5])
    original = deepcopy(anchor)
    goals = local_goals(anchor, 123)
    assert goals == local_goals(anchor, 123)
    assert goals[9:] != local_goals(anchor, 124)[9:]
    assert anchor == original
    assert len(goals) == len({g['name'] for g in goals}) == 25
    np.testing.assert_allclose(goals[4]['position_root'], anchor['position_root'])
    np.testing.assert_allclose([g['position_delta_root_m'][2] for g in goals[:9]], np.linspace(-.02, .02, 9))
    held = Rotation.from_quat(np.roll(anchor['quaternion_wxyz'], -1))
    for goal in goals:
        assert np.all(np.abs(goal['position_delta_root_m']) <= POSITION_RADIUS_M)
        assert np.all(np.abs(goal['root_xyz_euler_delta_deg']) <= ORIENTATION_RADIUS_DEG)
        expected = Rotation.from_euler('xyz', goal['root_xyz_euler_delta_deg'], degrees=True)*held
        actual = Rotation.from_quat(np.roll(goal['quaternion_wxyz'], -1))
        np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=1e-12)
    # Latin hypercube strata cover each parameter rather than repeating one trajectory.
    values = np.array([g['position_delta_root_m']+g['root_xyz_euler_delta_deg'] for g in goals[9:]])
    normalized = (values/np.r_[POSITION_RADIUS_M, ORIENTATION_RADIUS_DEG]+1)/2
    for column in normalized.T:
        assert len(np.unique(np.floor(column*16))) == 16


def test_physical_selection_excludes_rejected_paths_and_deduplicates_extremes():
    rows = [dict(name=n, position_root=[0, 0, z], path_status=s) for n,z,s in [
        ('lowest_rejected', -.1, 'no_collision_free_ik'), ('height_04', 0, 'success'),
        ('low', -.01, 'success'), ('high', .01, 'success'), ('endpoint_only', .02, 'planning_failed')]]
    assert [r['name'] for r in execution_candidates(rows)] == ['height_04', 'low', 'high']
    assert execution_candidates(rows[:1]) == []
    assert [r['name'] for r in execution_candidates(rows[1:2])] == ['height_04']


def test_grasp_gate_requires_intended_attachment_and_final_four_closed_held_ticks():
    target = 'cabinet_item_16'
    rows = [dict(parameters_physical=dict(gripper=-1.),
                 after=dict(contact_snapshot=dict(objects={target:dict(grasped=True)}))) for _ in range(4)]
    attachment = dict(name=target)
    assert confirmed_grasp(rows, target, attachment)
    assert not confirmed_grasp(rows[:3], target, attachment)
    assert not confirmed_grasp(rows, target, None)
    assert not confirmed_grasp(rows, target, dict(name='cabinet_item_17'))
    rows[-1]['parameters_physical']['gripper'] = 1.
    assert not confirmed_grasp(rows, target, attachment)
    rows[-1]['parameters_physical']['gripper'] = -1.
    rows[-1]['after']['contact_snapshot']['objects'][target]['grasped'] = False
    assert not confirmed_grasp(rows, target, attachment)
