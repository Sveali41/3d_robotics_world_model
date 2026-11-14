from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from domain.robocasa.motion_planning import MotionPlanningError
from exploration.diagnostics.diagnose_grasp_targets import candidate_goals, check_goal, distal_pad_boxes, shape_bounds


def test_candidates_keep_baselines_and_distinguish_root_from_tool_rotations():
    position = np.array([.45, -.06, .74])
    quaternion = np.array([0., 2**-.5, 0., 2**-.5])
    jitter = np.array([.002, -.003, .004])
    goals = candidate_goals(position, quaternion, .81, jitter)
    assert len(goals) == len({g['name'] for g in goals}) == 17
    np.testing.assert_allclose(goals[0]['position_root'], position+jitter)
    np.testing.assert_allclose(goals[1]['position_root'], position+jitter+[-.08, 0, 0])
    held = Rotation.from_quat(np.roll(quaternion, -1))
    for goal in goals:
        rotation = Rotation.from_quat(np.roll(goal['quaternion_wxyz'], -1))
        if goal['orientation'] == 'root_y_plus30':
            expected = Rotation.from_euler('y', 30, degrees=True).apply(held.apply([0, 0, 1]))
            np.testing.assert_allclose(rotation.apply([0, 0, 1]), expected, atol=1e-12)
        if goal['orientation'].startswith('tool_z'):
            np.testing.assert_allclose(rotation.apply([0, 0, 1]), held.apply([0, 0, 1]), atol=1e-12)
        if goal['name'].startswith('upper_front'):
            assert goal['position_root'][2] == pytest.approx(.80)


@pytest.mark.parametrize('status,accepted', [('success', 1), ('path_collision', 1), ('no_collision_free_ik', 0)])
def test_endpoint_acceptance_is_separate_from_path_and_grasp_success(status, accepted):
    goal = candidate_goals(np.array([.45, 0, .74]), [1, 0, 0, 0], .81, np.zeros(3))[0]
    original = deepcopy(goal)
    diagnostics = dict(counts=dict(accepted=accepted))
    qpos = np.zeros((1, 9))

    def plan(position, quaternion, **kwargs):
        assert position == goal['position_root'] and quaternion == goal['quaternion_wxyz']
        assert kwargs['seed'] == 5 and kwargs['planning_time_s'] == .2
        if status != 'success':
            raise MotionPlanningError(status, dict(ik_diagnostics=diagnostics))
        return SimpleNamespace(metadata=dict(ik_diagnostics=diagnostics), position=np.zeros((2, 9)))

    raw = SimpleNamespace(motion_planner=SimpleNamespace(snapshot_scene=lambda: {'unchanged': True}),
                          agent=SimpleNamespace(robot=SimpleNamespace(get_qpos=lambda: qpos)), plan_to_pose=plan)
    row = check_goal(raw, goal, 5, .2)
    assert row['endpoint_ik_verified'] == bool(accepted)
    assert row['path_status'] == status
    assert row['grasp_success_verified'] is False
    assert goal == original


def test_diagnostic_detects_unexpected_physical_state_mutation():
    qpos = np.zeros((1, 9))
    def plan(*args, **kwargs):
        qpos[0, 0] = .01
        raise MotionPlanningError('no_collision_free_ik', {})
    raw = SimpleNamespace(motion_planner=SimpleNamespace(snapshot_scene=lambda: {}),
        agent=SimpleNamespace(robot=SimpleNamespace(get_qpos=lambda: qpos)), plan_to_pose=plan)
    goal = candidate_goals(np.zeros(3), [1, 0, 0, 0], .1, np.zeros(3))[0]
    with pytest.raises(RuntimeError, match='changed physical state'):
        check_goal(raw, goal, 0, .2)


def test_distal_pad_is_selected_from_all_four_collision_boxes_per_finger():
    boxes = [dict(link=link, pose_in_tcp=[0., sign*.04758, z, 1., 0., 0., 0.])
        for link, sign in [('panda_leftfinger', 1.), ('panda_rightfinger', -1.)]
        for z in [-.034, -.0428, -.01665, .00025]]
    pads = distal_pad_boxes(boxes)
    assert len(pads) == 2
    np.testing.assert_allclose([p['pose_in_tcp'][2] for p in pads], [.00025, .00025])
    np.testing.assert_allclose(np.mean([p['pose_in_tcp'][:3] for p in pads], axis=0), [0., 0., .00025])


def test_native_cylinder_support_keeps_height_along_corrected_z_axis():
    # The planner has converted PhysX's local X cylinder to an upright FCL Z cylinder.
    shape = SimpleNamespace(radius=.034, half_length=.070)
    pose = SimpleNamespace(p=np.array([.45, -.06, .74]), q=np.array([1., 0., 0., 0.]))
    bounds = shape_bounds(shape, pose, 'Cylinder')
    np.testing.assert_allclose(bounds[1]-bounds[0], [.068, .068, .14], atol=1e-7)
    np.testing.assert_allclose(bounds.mean(axis=0), pose.p, atol=1e-7)
