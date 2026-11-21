"""Bounded IK accuracy, with the historical relative-step solver as reference."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from domain.robocasa.motion_planning import CabinetMotionPlanner, _rotation


def baseline_bounded_pose_ik(self, goal, start, limits):
    """Frozen baseline from the captured pre-optimization source."""
    from scipy.optimize import least_squares

    target_rotation = _rotation(goal)
    indices = self.controlled
    def residual(values):
        q = start.copy()
        q[indices] = values
        self.kin.pmodel.compute_forward_kinematics(q)
        actual = self.kin.pmodel.get_link_pose(self.kin.end_link_idx)
        return np.r_[np.asarray(actual.p) - goal.p,
                     (target_rotation.inv() * _rotation(actual)).as_rotvec()]

    # Native FK is float32; default finite differences are below its resolution.
    fit = least_squares(residual, start[indices],
        bounds=(limits[indices, 0], limits[indices, 1]), diff_step=1e-4,
        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
    q = start.copy()
    q[indices] = fit.x
    error = residual(fit.x)
    success = np.linalg.norm(error[:3]) <= .001 and np.linalg.norm(error[3:]) <= np.radians(.1)
    return q, success, dict(optimizer_status=int(fit.status), evaluations=int(fit.nfev))


@pytest.mark.parametrize('initial', [0., 1e-7, .01])
def test_absolute_step_solves_quantized_fk_near_zero(initial):
    class QuantizedFK:
        def compute_forward_kinematics(self, q):
            self.position = np.array([.25 + q[0], q[1], 0.], dtype=np.float32)

        def get_link_pose(self, index):
            return SimpleNamespace(p=self.position, q=np.array([1., 0., 0., 0.]))

    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.controlled = np.array([0, 1])
    planner.kin = SimpleNamespace(pmodel=QuantizedFK(), end_link_idx=0)
    start, limits = np.array([initial, .01]), np.array([[-1., 1.], [-1., 1.]])
    goal = SimpleNamespace(p=np.array([.255, 0., 0.]), q=np.array([1., 0., 0., 0.]))
    if initial == 1e-7:
        assert not baseline_bounded_pose_ik(planner, goal, start, limits)[1]
    q, success, detail = planner._bounded_pose_ik(goal, start, limits)
    assert success and detail['position_error_m'] <= .001
    assert detail['orientation_error_deg'] <= .1
    assert q[0] == pytest.approx(.005, abs=1e-5)
    assert q[1] == pytest.approx(0., abs=1e-5)
    np.testing.assert_array_equal(start, [initial, .01])

    unreachable = SimpleNamespace(p=np.array([2., 0., 0.]), q=goal.q)
    q, success, detail = planner._bounded_pose_ik(unreachable, start, limits)
    assert not success and detail['position_error_m'] > .001
    assert limits[0, 0] <= q[0] <= limits[0, 1]


@pytest.mark.skipif(os.environ.get('RUN_NATIVE_CABINET') != '1', reason='native robotics runtime required')
def test_bounded_ik_accuracy_and_repeatability_without_physics_steps(monkeypatch):
    import sapien
    from exploration.cee_us.cee_us_cabinet import make_env
    from domain.robocasa.motion_planning import _array

    env, _ = make_env(episode_steps=2200, low_level_episode_steps=2200,
        centered_initial_tcp=True, cabinet_door=True, cabinet_item_layout='single_upper_second_row',
        cabinet_grid=True, cabinet_grid_layout='trapezoid_4_3_2_1', action_mode='joint_position')
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        planner = raw.motion_planner
        original_qpos = _array(planner.robot.get_qpos()).reshape(9).copy()
        start = original_qpos[planner.pindices].astype(float)
        limits = _array(planner.robot.get_qlimits()).reshape(9, 2)[planner.pindices]
        model = planner.kin.pmodel
        idx = planner.kin.end_link_idx

        def fk(q):
            model.compute_forward_kinematics(q)
            pose = model.get_link_pose(idx)
            return sapien.Pose(np.asarray(pose.p).copy(), np.asarray(pose.q).copy())

        goals = []
        for delta in (0., .005, -.005):
            q = start.copy()
            q[planner.controlled[0]] += delta
            assert np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1])
            goals.append(fk(q))
        goals.append(sapien.Pose(np.asarray(goals[0].p) + [1., 0., 0.], goals[0].q))
        evidence = []
        for index, goal in enumerate(goals):
            model.compute_forward_kinematics(start)
            reference = baseline_bounded_pose_ik(planner, goal, start, limits)
            reference_pose = fk(reference[0])
            model.compute_forward_kinematics(start)
            actual = planner._bounded_pose_ik(goal, start, limits)
            actual_pose = fk(actual[0])
            assert actual[1] == (index < 3)
            assert np.all(actual[0] >= limits[:, 0]) and np.all(actual[0] <= limits[:, 1])
            if actual[1]:
                assert np.linalg.norm(actual_pose.p - goal.p) <= .001
                assert (_rotation(goal).inv() * _rotation(actual_pose)).magnitude() <= np.radians(.1)
            model.compute_forward_kinematics(start)
            repeated = planner._bounded_pose_ik(goal, start, limits)
            np.testing.assert_array_equal(repeated[0], actual[0])
            assert repeated[1:] == actual[1:]
            assert np.array_equal(_array(planner.robot.get_qpos()).reshape(9), original_qpos)
            evidence.append(dict(reference_success=bool(reference[1]),
                                 reference_position_error_m=float(np.linalg.norm(reference_pose.p-goal.p)),
                                 success=bool(actual[1]), **actual[2]))
        print('IK_ACCURACY', evidence)
        # Captured Random homing waypoint: unscaled trust region stalls even
        # with a correct absolute Jacobian; LSMR preserves the target and bounds.
        import scipy.optimize
        failure_start = np.array([
            -.24102889537153846, -1.567930106050419, -.022840841262608814,
            -2.7206184561598343, -.05203341022517316, 2.7168408227333534,
            .8060651497241236, .039551299065351486, .039551299065351486])
        failure_goal = sapien.Pose(
            [.1993752121925354, -.06000081077218056, .734210729598999],
            [.09360712021589279, .7035342454910278, -.0913836807012558, .6985172629356384])
        solve = scipy.optimize.least_squares
        def dense_exact(*args, **kwargs):
            kwargs.pop('tr_solver', None)
            return solve(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(scipy.optimize, 'least_squares', dense_exact)
            _, accepted, detail = planner._bounded_pose_ik(failure_goal, failure_start, limits)
        assert not accepted and detail['evaluations'] == 200
        assert detail['position_error_m'] > .001
        for delta in (0., -.0005, .0005):
            goal = sapien.Pose(np.asarray(failure_goal.p) + [delta, 0., 0.], failure_goal.q)
            q, accepted, detail = planner._bounded_pose_ik(goal, failure_start, limits)
            assert accepted and detail['position_error_m'] <= .001
            assert detail['orientation_error_deg'] <= .1 and detail['evaluations'] <= 200
            assert np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1])
        np.testing.assert_array_equal(_array(planner.robot.get_qpos()).reshape(9), original_qpos)
    finally:
        if 'model' in locals():
            model.compute_forward_kinematics(start)
        env.close()
