"""Native-free checks of the collision planner's safety boundaries."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from domain.robocasa.motion_planning import (
    CabinetMotionPlanner, MotionPlanningError, _groups_allow, _interpolate,
    _native_modules, _plane_min_distance, _resample_to_control_dt, _validate_goal, _angle,
)


def pose(p=(0., 0., 0.), q=(1., 0., 0., 0.)):
    return SimpleNamespace(p=np.asarray(p), q=np.asarray(q))


@pytest.mark.parametrize("position,quaternion", [
    ([0., 1.], [1., 0., 0., 0.]), ([0., 0., np.inf], [1., 0., 0., 0.]),
    ([0., 0., 0.], [0., 0., 0., 0.]), ([0., 0., 0.], [np.nan, 0., 0., 0.]),
    ([0., 0., 0.], [[1., 0., 0., 0.]]),
])
def test_invalid_goal_fails_before_native_import(position, quaternion):
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    with pytest.raises(MotionPlanningError) as error:
        planner.plan_to_pose(position, quaternion)
    assert error.value.status == "invalid_goal"


def test_goal_normalizes_quaternion_without_projecting_position():
    position, quaternion = _validate_goal([100., -200., 300.], [1e300, 0., 0., 1e300])
    np.testing.assert_array_equal(position, [100., -200., 300.])
    np.testing.assert_allclose(quaternion, [2**-.5, 0., 0., 2**-.5])


def test_angle_normalization_avoids_float32_false_fk_mismatch():
    q = np.array([.90535587, .44637457, -.53695321, .58111811], dtype=np.float32)
    assert _angle(q, q) < 1e-5
    assert _angle(q, -q) < 1e-5


def test_numpy2_guard_precedes_native_import(monkeypatch):
    monkeypatch.setattr(np, "__version__", "2.2.6")
    with pytest.raises(RuntimeError, match="NumPy < 2"):
        _native_modules()


def test_physx_groups_respect_masks_and_shared_exclusion():
    assert _groups_allow([1, 1, 0, 0], [1, 1, 0, 0])
    assert not _groups_allow([1, 0, 0, 0], [2, 0, 0, 0])
    assert not _groups_allow([1, 1, 4, 9], [1, 1, 4, 9])
    assert _groups_allow([1, 1, 4, 9], [1, 1, 4, 10])


def test_interpolation_checks_both_endpoints_and_bounds_every_joint():
    source = np.array([[0., 0.], [.025, -.011], [.025, -.011]])
    points = np.array(list(_interpolate(source)))
    np.testing.assert_array_equal(points[0], source[0])
    np.testing.assert_array_equal(points[-1], source[-1])
    assert np.max(np.abs(np.diff(points, axis=0))) <= .01 + 1e-12
    assert any(np.array_equal(point, source[1]) for point in points)


def test_resample_preserves_speed_and_endpoint_with_partial_final_tick():
    source_time = np.array([0., .052417, .104834, .131])
    velocity = np.array([.5, -.25])
    positions = np.array([1., -1.]) + source_time[:, None] * velocity
    scheduled, times = _resample_to_control_dt(positions, source_time, .05)
    np.testing.assert_allclose(times, [0., .05, .1, .15])
    np.testing.assert_array_equal(scheduled[0], positions[0])
    np.testing.assert_array_equal(scheduled[-1], positions[-1])
    np.testing.assert_allclose(scheduled, positions[0] + np.minimum(times, source_time[-1])[:, None] * velocity)
    assert np.all(np.abs(np.diff(scheduled, axis=0)) <= np.abs(velocity) * .05 + 1e-12)
    assert source_time[-1] <= times[-1] < source_time[-1] + .05


def test_plane_support_uses_rotation_scaled_convex_and_sphere_radius():
    plane = pose(p=(0., 0., .5))
    assert _plane_min_distance(SimpleNamespace(radius=.2), pose(p=(0., 0., .8)), plane, "Sphere") == pytest.approx(.1)
    box = SimpleNamespace(half_size=np.array([.3, .1, .2]))
    rotated = pose(p=(0., 0., .75), q=np.roll(Rotation.from_euler("y", 90, degrees=True).as_quat(), 1))
    assert _plane_min_distance(box, rotated, plane, "Box") == pytest.approx(-.05)
    convex = SimpleNamespace(vertices=np.array([[0., 0., -1.], [0., 0., 1.]]), scale=np.array([1., 1., .3]))
    assert _plane_min_distance(convex, pose(p=(0., 0., .75)), plane, "ConvexMesh") == pytest.approx(-.05)


def test_cylinder_axis_conversion_matches_physx_local_x():
    class PhysxCollisionShapeCylinder:
        half_length = .3
        radius = .1

    class Pose:
        def __init__(self, p=(0., 0., 0.), q=(1., 0., 0., 0.)):
            self.p, self.q = np.asarray(p), np.asarray(q)

        def __mul__(self, other):
            a = Rotation.from_quat(np.roll(self.q, -1))
            b = Rotation.from_quat(np.roll(other.q, -1))
            return Pose(self.p + a.apply(other.p), np.roll((a*b).as_quat(), 1))

    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.sapien = SimpleNamespace(Pose=Pose)
    shape = PhysxCollisionShapeCylinder()
    converted = planner._shape_pose(shape, Pose(p=(0., 0., .2)))
    axis = Rotation.from_quat(np.roll(converted.q, -1)).apply([0., 0., 1.])
    np.testing.assert_allclose(axis, [1., 0., 0.], atol=1e-12)
    assert _plane_min_distance(shape, converted, pose(), "Cylinder") == pytest.approx(.1)


def test_scene_snapshot_reflects_live_object_and_cabinet_poses():
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    item = SimpleNamespace(name="cup", pose=pose(p=(1., 2., 3.)))
    door = SimpleNamespace(name="door", pose=pose())
    planner.root_link = SimpleNamespace(pose=pose())
    planner.env = SimpleNamespace(cabinet=SimpleNamespace(get_links=lambda: [door]),
                                  cabinet_items=[item], wall=SimpleNamespace(pose=pose()),
                                  ground=SimpleNamespace(pose=pose()))
    first = planner.snapshot_scene()
    item.pose.p[0] = 4
    door.pose.q = np.array([0., 1., 0., 0.])
    second = planner.snapshot_scene()
    assert first["objects"]["cup"][0] == 1
    assert second["objects"]["cup"][0] == 4
    assert second["cabinet"]["door"][3:] == [0., 1., 0., 0.]


@pytest.mark.parametrize("native_grasp,forces", [(True, (0., 0.)), (False, (.05, .05))])
def test_unconfirmed_grasp_is_rejected(native_grasp, forces):
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    agent = SimpleNamespace(is_grasping=lambda _: [native_grasp], finger1_link="left", finger2_link="right")
    scene = SimpleNamespace(get_pairwise_contact_forces=lambda link, _: [[0., 0., forces[link == "right"]]])
    planner.env = SimpleNamespace(agent=agent, cabinet_items=[SimpleNamespace(name="cup")], scene=scene)
    planner.attachment_state = lambda: None
    with pytest.raises(MotionPlanningError) as error:
        planner._planning_attachment()
    assert error.value.status == "unconfirmed_grasp"


def test_execution_start_requires_current_scene_and_joint_start():
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.check_attachment_feedback = lambda _: None
    qpos = np.zeros((1, 9))
    planner.robot = SimpleNamespace(get_qpos=lambda: qpos)
    signature = {"objects": {"cup": [0., 0., 0.]}}
    planner.snapshot_scene = lambda: signature
    plan = SimpleNamespace(initial_qpos=np.zeros(9), scene_signature={"objects": {"cup": [0., 0., 0.]}})
    planner.assert_plan_start(plan)
    qpos[0, 0] = .001
    with pytest.raises(MotionPlanningError, match="stale_plan"):
        planner.assert_plan_start(plan)
    qpos[0, 0] = 0.
    signature["objects"]["cup"][0] = .001
    with pytest.raises(MotionPlanningError, match="stale_plan"):
        planner.assert_plan_start(plan)


def test_self_audit_excludes_only_direct_parent_child_and_disabled_groups():
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.kin = SimpleNamespace(pmodel=SimpleNamespace(compute_forward_kinematics=lambda _: None,
                                                       get_link_pose=lambda _: None))
    class Identity:
        def __mul__(self, other):
            return self
    planner.root = Identity()
    planner.link_indices = {name: i for i, name in enumerate(("base", "arm", "tip"))}
    planner.robot_shapes = [dict(link=name, shape=None, geom=name, local=None, groups=[1, 1, 0, 0])
                            for name in planner.link_indices]
    planner._shape_pose = lambda shape, pose: pose
    planner._object = lambda geometry, pose: geometry
    planner.adjacent_pairs = {frozenset(("base", "arm")), frozenset(("arm", "tip"))}
    planner.obstacles = []
    planner.fcl = SimpleNamespace(CollisionRequest=lambda *args: None,
        collide=lambda *args: SimpleNamespace(is_collision=lambda: True))
    assert planner._check(np.zeros(9)) == [{"first": "base", "second": "tip", "kind": "self"}]
    planner.robot_shapes[-1]["groups"] = [2, 2, 0, 0]
    assert planner._check(np.zeros(9)) == []


@pytest.mark.parametrize("message", ["Fail to parameterize path", "Unrelated planner failure"])
def test_plan_api_classifies_only_known_topp_failure(monkeypatch, message):
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.env = SimpleNamespace(cabinet_items=[], control_freq=20)
    planner.robot = SimpleNamespace(get_qpos=lambda: np.zeros((1, 9)),
        get_qlimits=lambda: np.tile([-3., 3.], (1, 9, 1)))
    planner.pindices = np.arange(9)
    planner.controlled = np.arange(7)
    planner.root_link = SimpleNamespace(pose=pose())
    planner.links = []
    planner.kin = SimpleNamespace(end_link_idx=0, qmask=np.ones(9), pmodel=SimpleNamespace(
        compute_forward_kinematics=lambda _: None, get_link_pose=lambda _: pose(),
        compute_inverse_kinematics=lambda *args, **kwargs: (np.zeros(9), True, None)))
    planner._planning_attachment = lambda: None
    planner._pose = lambda value: value
    planner.snapshot_scene = lambda: {}
    planner._scene_geometry = lambda: None
    planner._check = lambda _: []
    planner.obstacles = []
    planner.support_contacts = {}
    failure = RuntimeError(message)
    calls = []
    def plan_qpos_to_qpos(**kwargs):
        calls.append(kwargs)
        raise failure
    planner._make_planner = lambda: SimpleNamespace(set_base_pose=lambda _: None,
        plan_qpos_to_qpos=plan_qpos_to_qpos)
    monkeypatch.setattr('domain.robocasa.motion_planning._native_modules',
        lambda: (SimpleNamespace(), SimpleNamespace(), SimpleNamespace(Pose=pose)))
    with pytest.raises(RuntimeError) as error:
        planner.plan_to_pose([0., 0., 0.], [1., 0., 0., 0.])
    assert len(calls) == 1
    if message == "Fail to parameterize path":
        assert type(error.value) is MotionPlanningError
        assert error.value.status == "time_parameterization_failed"
        assert error.value.__cause__ is failure
        assert "no motion was executed" in error.value.detail
    else:
        assert error.value is failure


def _diagnostic_planner(monkeypatch, inverse_kinematics, *, collision=None, path=None):
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.env = SimpleNamespace(cabinet_items=[], control_freq=20)
    planner.robot = SimpleNamespace(get_qpos=lambda: np.zeros((1, 9)),
        get_qlimits=lambda: np.tile([-3., 3.], (1, 9, 1)))
    planner.pindices = np.arange(9)
    planner.controlled = np.arange(7)
    planner.root_link = SimpleNamespace(pose=pose())
    planner.links = []
    planner.kin = SimpleNamespace(end_link_idx=0, qmask=np.ones(9), pmodel=SimpleNamespace(
        compute_forward_kinematics=lambda _: None,
        get_link_pose=lambda _: pose(), compute_inverse_kinematics=inverse_kinematics))
    planner._planning_attachment = lambda: None
    planner._pose = lambda value: value
    planner.snapshot_scene = lambda: {}
    planner._scene_geometry = lambda: None
    planner._check = collision or (lambda _: [])
    planner.obstacles = []
    planner.support_contacts = {}
    if path is not None:
        planner._make_planner = lambda: SimpleNamespace(
            move_group_joint_indices=np.arange(9), set_base_pose=lambda _: None,
            plan_qpos_to_qpos=path)
    monkeypatch.setattr('domain.robocasa.motion_planning._native_modules',
        lambda: (SimpleNamespace(__version__="test"), SimpleNamespace(), SimpleNamespace(Pose=pose)))
    return planner


def test_no_collision_free_ik_reports_each_seed_rejection_as_json_safe(monkeypatch):
    ik_calls = []

    def inverse_kinematics(*args, **kwargs):
        index = len(ik_calls)
        ik_calls.append((args, kwargs))
        q = np.zeros(9)
        if index == 0:
            return q, False, None
        if index == 1:
            q[0] = np.nan
        elif index == 2:
            q[7] = 4.
        elif index == 3:
            q[0] = 1.
        else:
            q[0] = 2.
        return q, True, None

    contacts = [dict(first="panda_link3", second="cabinet/door/0", kind="environment")]
    planner = _diagnostic_planner(monkeypatch, inverse_kinematics,
        collision=lambda q: contacts if q[0] == 2. else [])
    fk_state = {"q": np.zeros(9)}
    planner.kin.pmodel.compute_forward_kinematics = lambda q: fk_state.update(q=np.asarray(q).copy())
    planner.kin.pmodel.get_link_pose = lambda _: pose(p=(.01, 0., 0.) if fk_state["q"][0] == 1. else (0., 0., 0.))
    with pytest.raises(MotionPlanningError) as error:
        planner.plan_to_pose([0., 0., 0.], [1., 0., 0., 0.], seed=9)

    assert error.value.status == "no_collision_free_ik"
    assert error.value.detail["message"] == "No valid collision-free solution in 16 fixed-orientation IK starts"
    diagnostics = error.value.detail["ik_diagnostics"]
    assert diagnostics["candidate_count"] == len(diagnostics["candidates"]) == len(ik_calls) == 16
    assert diagnostics["counts"] == dict(solver_failure=1, nonfinite_solution=1,
        joint_limit_violation=1, fk_mismatch=1, collision=12, accepted=0)
    assert [candidate["reason"] for candidate in diagnostics["candidates"][:5]] == [
        "solver_failure", "nonfinite_solution", "joint_limit_violation", "fk_mismatch", "collision"]
    assert diagnostics["candidates"][0]["initial_seed_source"] == "current_qpos"
    assert diagnostics["candidates"][1]["initial_seed_source"] == "random_uniform"
    assert diagnostics["candidates"][1]["solution_qpos_rad"][0] is None
    assert diagnostics["candidates"][1]["nonfinite_solution_indices"] == [0]
    assert diagnostics["candidates"][2]["joint_limit_violations"][0]["joint_index"] == 7
    assert diagnostics["candidates"][4]["collision_contacts"] == contacts
    json.dumps(error.value.detail, allow_nan=False)

    rng = np.random.default_rng(9)
    expected_starts = [np.zeros(9)]
    for _ in range(15):
        start = np.zeros(9)
        start[:7] = rng.uniform(-3., 3., size=7)
        expected_starts.append(start)
    for index, ((args, kwargs), expected) in enumerate(zip(ik_calls, expected_starts)):
        assert len(args) == 2
        assert args[0] == 0
        np.testing.assert_array_equal(kwargs["initial_qpos"], expected)
        np.testing.assert_array_equal(kwargs["active_qmask"], np.ones(9))
        assert kwargs["max_iterations"] == 1000
        assert diagnostics["candidates"][index]["initial_seed_qpos_rad"] == expected.tolist()


def test_successful_plan_keeps_nearest_accepted_ik_and_records_diagnostics(monkeypatch):
    ik_calls, plan_calls = [], []

    def inverse_kinematics(*args, **kwargs):
        index = len(ik_calls)
        ik_calls.append((args, kwargs))
        q = np.zeros(9)
        q[0] = 2. if index == 0 else .2 if index == 1 else .5
        return q, True, None

    def plan_qpos_to_qpos(**kwargs):
        plan_calls.append(kwargs)
        target = kwargs["goal_qposes"][0]
        return dict(status="Success", position=np.stack([np.zeros(9), target]),
                    time=np.array([0., .1]))

    planner = _diagnostic_planner(monkeypatch, inverse_kinematics, path=plan_qpos_to_qpos)
    planner.mplib = SimpleNamespace(__version__="test")
    result = planner.plan_to_pose([0., 0., 0.], [1., 0., 0., 0.], seed=9)

    assert len(ik_calls) == 16
    np.testing.assert_allclose(plan_calls[0]["goal_qposes"][0], [ .2, 0., 0., 0., 0., 0., 0., 0., 0.])
    np.testing.assert_allclose(result.position[0], np.zeros(9))
    np.testing.assert_allclose(result.position[-1], plan_calls[0]["goal_qposes"][0])
    diagnostics = result.metadata["ik_diagnostics"]
    assert diagnostics["counts"]["accepted"] == 16
    assert diagnostics["candidates"][0]["solution_qpos_rad"][0] == 2.
    assert all(item["reason"] == "accepted" for item in diagnostics["candidates"])
    json.dumps(result.metadata, allow_nan=False)


@pytest.mark.parametrize("target_x,expected", [(.6, True), (3., False)])
def test_bounded_pose_ik_uses_redundancy_and_rejects_unreachable_goal(target_x, expected):
    planner = CabinetMotionPlanner.__new__(CabinetMotionPlanner)
    planner.controlled = np.arange(7)
    model = SimpleNamespace()
    model.compute_forward_kinematics = lambda q: setattr(model, 'q', np.asarray(q).copy())
    def fk(_):
        q = model.q
        return SimpleNamespace(p=np.asarray([q[0]+q[6], q[1], q[2]], dtype=np.float32),
            q=np.roll(Rotation.from_rotvec(q[3:6]).as_quat(), 1))
    model.get_link_pose = fk
    planner.kin = SimpleNamespace(pmodel=model, end_link_idx=0)
    start = np.array([0., 0., 0., 0., 0., 0., .39, .03, .03])
    limits = np.tile([-1., 1.], (9, 1)); limits[6] = [0., .4]
    goal = SimpleNamespace(p=np.array([target_x, 0., 0.]), q=np.array([1., 0., 0., 0.]))
    q, success, detail = planner._bounded_pose_ik(goal, start, limits)
    assert bool(success) == expected
    assert np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1])
    np.testing.assert_array_equal(q[7:], start[7:])
    assert detail['evaluations'] <= 200
    if expected:
        assert q[0] > .19
        np.testing.assert_allclose(q[0]+q[6], target_x, atol=.001)


@pytest.mark.parametrize('samples', [0, 1])
def test_short_bounded_cartesian_timing_keeps_endpoint_and_physical_duration(monkeypatch, samples):
    planner = _diagnostic_planner(monkeypatch, lambda *a, **kw: pytest.fail('Reference IK was used'))
    model = planner.kin.pmodel
    model.compute_forward_kinematics = lambda q: setattr(model, 'q', np.asarray(q).copy())
    model.get_link_pose = lambda _: pose(p=model.q[:3])
    def bounded(goal, start, limits):
        q = start.copy(); q[:3] = goal.p
        return q, True, {}
    planner._bounded_pose_ik = bounded
    planner._make_planner = lambda: SimpleNamespace(
        move_group_joint_indices=np.arange(9), set_base_pose=lambda _: None,
        TOPP=lambda waypoints, dt: (np.zeros(samples), np.zeros((samples, 9)), None, None, .02))
    result = planner.plan_to_pose([.0005, 0., 0.], [1., 0., 0., 0.],
                                  method='bounded_ik', maintain_orientation=True)
    np.testing.assert_allclose(result.time, [0., .05])
    assert result.position[-1, 0] == .0005
    assert result.metadata['source_duration_s'] == .02
    assert result.metadata['scheduled_duration_s'] == .05


def test_opt_in_rrt_densifies_before_topp_and_retains_exact_collision_gate(monkeypatch):
    calls, timed = [], []
    goal = np.zeros(9); goal[0] = .2
    planner = _diagnostic_planner(monkeypatch, lambda *a, **kw: (goal.copy(), True, None))
    def raw_plan(start, goals, **kwargs):
        calls.append(kwargs)
        return "Exact solution", np.stack([start, goals[0]])
    def timing(points, dt):
        timed.append(points.copy())
        return np.array([0., .1]), points[[0, -1]], None, None, .1
    planner._make_planner = lambda: SimpleNamespace(
        robot=SimpleNamespace(set_qpos=lambda *_: None),
        planning_world=SimpleNamespace(set_use_point_cloud=lambda _: None,
                                       set_use_attach=lambda _: None),
        planner=SimpleNamespace(plan=raw_plan), move_group_joint_indices=np.arange(9),
        set_base_pose=lambda _: None, TOPP=timing)
    result = planner.plan_to_pose([0., 0., 0.], [1., 0., 0., 0.], densify_rrt_path=True)
    assert np.max(np.abs(np.diff(timed[0], axis=0))) <= .02500001
    np.testing.assert_array_equal(timed[0][[0, -1]], np.stack([np.zeros(9), goal]))
    assert calls[0] == dict(range=.1, time=3., fixed_joints=set(),
        planner_name="RRTConnect", no_simplification=False, verbose=False)
    assert result.metadata["rrt_pre_TOPP_max_joint_step_rad"] == .025
    planner._check = lambda q: ([dict(kind="synthetic_timed_collision")]
                                if .09 < q[0] < .11 else [])
    with pytest.raises(MotionPlanningError, match="path_collision"):
        planner.plan_to_pose([0., 0., 0.], [1., 0., 0., 0.], densify_rrt_path=True)
