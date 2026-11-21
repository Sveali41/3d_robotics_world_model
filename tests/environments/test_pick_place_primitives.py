from types import SimpleNamespace

import numpy as np
import pytest

from domain.robocasa.pick_place import (
    INTERMEDIATE, PREPLACE, OUTSIDE_X, PICK_PLACE_ACTION_SCHEMA, ACTION_BOUNDS, TCP_BOUNDS, PLACE_CLEARANCE,
    PickPlaceSkillExecution, lower_shelf_supported, pick_place_contract,
)
from domain.robocasa.primitives import PrimitiveAction, RAPS_ACTION_SCHEMA, RAPS_COMPACT_ACTION_SCHEMA, RAPS_SPLIT_ACTION_SCHEMA


def context():
    actor = SimpleNamespace(_objs=[object()], linear_velocity=np.zeros(3))
    item = SimpleNamespace(actor=actor, position_root=np.array([.45, 0., .73]),
                           position_world=np.array([.45, 0., .73]))
    ctx = SimpleNamespace(ee_position=np.array([.18, 0., .84], dtype=np.float32),
                          gripper_open=1., objects={"generic": item}, held=False)
    ctx.is_grasping = lambda key: ctx.held
    ctx.finger_contact_force = lambda key: (.1, .1) if ctx.held else (0., 0.)
    frame_entity = object()
    frame = SimpleNamespace(name="frame", _objs=[SimpleNamespace(entity=frame_entity)])
    pose = SimpleNamespace(to_transformation_matrix=lambda: np.eye(4))
    ctx.arm_controller = SimpleNamespace(root_link=SimpleNamespace(pose=pose),
        ee_pose_at_base=SimpleNamespace(q=np.array([[1., 0., 0., 0.]])))
    ctx.support = False
    ctx.env = SimpleNamespace(cabinet=SimpleNamespace(get_links=lambda: [frame]),
        scene=SimpleNamespace(get_pairwise_contact_forces=lambda *_: np.array([[0., 0., .1 if ctx.support else 0.]]),
            get_contacts=lambda: [SimpleNamespace(bodies=[SimpleNamespace(entity=actor._objs[0]),
                SimpleNamespace(entity=frame_entity)], points=[SimpleNamespace(position=np.array([.45, 0., .330]))])] if ctx.support else []))
    def attachment():
        return dict(name="generic", tcp_to_object_pose=np.r_[
            ctx.objects["generic"].position_root-ctx.ee_position, [1.,0.,0.,0.]].tolist()) if ctx.held else None
    def bounds(reference, orientation):
        center = np.asarray(reference["tcp_to_object_pose"][:3])
        return np.array([center - [.02,.02,.05], center + [.02,.02,.05]])
    ctx.env.motion_planner = SimpleNamespace(attachment_state=attachment, placement_bounds=bounds)
    return ctx


def action(name, xyz=None):
    xyz = xyz or ((.45, 0., .75) if name == "grasp" else (.45, 0., .43))
    axes = "xyz" if name == "grasp" else "xy"
    return PrimitiveAction(name, dict(zip(axes, xyz)))


def tick(execution, ctx, *, acquire=True, support=True):
    low = execution.next_action(ctx)
    if low is None:
        return None
    previous = ctx.ee_position.copy()
    ctx.ee_position += low[:3] * .10
    if execution.primitive.phase == "close" and acquire:
        ctx.held = True
    if ctx.held:
        ctx.objects["generic"].position_root += ctx.ee_position - previous
    if execution.primitive.phase in {"descend", "settle"} and support and ctx.ee_position[2] <= .40:
        ctx.support = True
    if execution.primitive.phase == "release":
        ctx.held = False
    execution.record_action(low)
    return low


def run(execution, ctx, **kwargs):
    rows = []
    for _ in range(501):
        low = tick(execution, ctx, **kwargs)
        if low is None:
            return rows
        rows.append((execution.primitive.phase, low.copy(), ctx.ee_position.copy()))
    pytest.fail("macro exceeded its bounded control count")


@pytest.mark.parametrize("name", ["grasp", "place"])
def test_shared_bank_maps_grasp_xyz_and_place_xy_boundaries(name):
    assert PICK_PLACE_ACTION_SCHEMA.action_dim == 5
    assert PICK_PLACE_ACTION_SCHEMA.primitive_name_to_action_idx[name] == ([0, 1, 2] if name == "grasp" else [0, 1])
    for endpoint, normalized in [(0, -1.), (1, 1.)]:
        xyz = [bounds[endpoint] for bounds in ACTION_BOUNDS[name]]
        encoded = PICK_PLACE_ACTION_SCHEMA.encode(action(name, xyz))
        np.testing.assert_allclose(encoded[2:2 + len(xyz)], normalized)
        if name == "place":
            assert encoded[-1] == 0.
        decoded = PICK_PLACE_ACTION_SCHEMA.decode(encoded)
        np.testing.assert_allclose(list(decoded.parameters.values()), xyz)
        clipped = PICK_PLACE_ACTION_SCHEMA.decode(np.r_[encoded[:2], [normalized * 2] * 3])
        np.testing.assert_allclose(list(clipped.parameters.values()), xyz)


@pytest.mark.parametrize("value", [np.nan, np.inf, -.1, 1.])
@pytest.mark.parametrize("name", ["grasp", "place"])
def test_encode_rejects_invalid_physical_tcp(name, value):
    selected = action(name)
    selected.parameters["x"] = value
    with pytest.raises(ValueError, match="finite"):
        PICK_PLACE_ACTION_SCHEMA.encode(selected)


def test_sampling_and_contract_do_not_depend_on_objects():
    import json
    for name in ACTION_BOUNDS:
        for _ in range(20):
            values = PICK_PLACE_ACTION_SCHEMA.sample_parameters(name, np.random.default_rng(2))
            PICK_PLACE_ACTION_SCHEMA.encode(PrimitiveAction(name, values))
    contract = pick_place_contract()
    json.dumps(contract, allow_nan=False)
    contract["tcp_bounds_root_m"]["grasp"][0][0] = -100
    assert pick_place_contract()["tcp_bounds_root_m"]["grasp"][0][0] == .370


def test_place_rejects_obsolete_z_argument():
    with pytest.raises(ValueError, match="exactly"):
        PICK_PLACE_ACTION_SCHEMA.encode(PrimitiveAction("place", dict(x=.45, y=0., z=.43)))


def test_place_compensates_actual_grasp_offset_and_automatic_height():
    ctx = context()
    ctx.held = True
    ctx.ee_position[:] = [.44, .02, .79]
    ctx.objects["generic"].position_root[:] = [.46, -.01, .73]
    executor = PickPlaceSkillExecution()
    executor.gripper_command = -1.
    executor.start(action("place", [.50, .10]), ctx)
    np.testing.assert_allclose(executor.place_touch_goal, [.48, .13, .44], atol=1e-7)
    np.testing.assert_allclose(executor.goal, [.48, .13, .44 + PLACE_CLEARANCE], atol=1e-7)
    reference = executor._place_reference
    ctx.ee_position[2] += .1
    ctx.objects["generic"].position_root[2] += .1
    executor.start(action("place", [.50, .10]), ctx)
    assert executor._place_reference is reference
    np.testing.assert_allclose(executor.place_touch_goal, [.48, .13, .44], atol=1e-7)
    executor.reset()
    assert executor._place_reference is None


def test_place_rejects_oversize_footprint_and_keeps_gripper_closed():
    ctx = context()
    ctx.held = True
    ctx.env.motion_planner.placement_bounds = lambda *_: np.array([[-.10,-.02,-.05],[.10,.02,.05]])
    executor = PickPlaceSkillExecution()
    executor.gripper_command = -1.
    executor.start(action("place", [.370, 0.]), ctx)
    rows = run(executor, ctx)
    assert executor.primitive.status == "invalid_goal"
    assert "footprint" in executor.placement_failure
    assert len(rows) == 1 and rows[0][1][-1] < 0


def test_grasp_then_place_phase_order_closed_transport_and_safe_home_route():
    ctx = context()
    executor = PickPlaceSkillExecution()
    executor.start(action("grasp"), ctx)
    rows = run(executor, ctx)
    assert executor.primitive.status == "success"
    phases = list(dict.fromkeys(row[0] for row in rows))
    assert phases == ["open", "align", "approach", "close", "lift", "withdraw", "intermediate"]
    assert all(row[1][3] == -1. for row in rows if row[0] in {"lift", "withdraw", "intermediate"})
    np.testing.assert_allclose(ctx.ee_position, INTERMEDIATE, atol=.012)
    home = executor.home.copy()
    executor.start(action("place"), ctx)
    rows = run(executor, ctx)
    assert executor.primitive.status == "success"
    assert executor.support_stopped_object_ids == ["generic"]
    phases = list(dict.fromkeys(row[0] for row in rows))
    assert phases == ["preplace", "approach", "descend", "settle", "release", "clear", "withdraw", "home_raise", "home_align"]
    assert all(row[1][3] == -1. for row in rows if row[0] in {"preplace", "approach", "descend", "settle"})
    assert all(row[1][3] == 1. for row in rows if row[0] in {"clear", "withdraw", "home_raise", "home_align", "home"})
    np.testing.assert_allclose(executor.primitive.requested_goals["preplace"], PREPLACE)
    assert pick_place_contract()["preplace_root_m"] == list(PREPLACE)
    assert pick_place_contract()["version"] == 6
    for phase, _, position in rows:
        if phase in {"align", "home_align"}:
            assert position[0] <= OUTSIDE_X + .012
    np.testing.assert_allclose(ctx.ee_position, home, atol=.012)


def test_loss_during_retained_transport_fails_before_return():
    ctx = context()
    executor = PickPlaceSkillExecution()
    executor.start(action("grasp"), ctx)
    while executor.primitive.phase != "withdraw":
        assert tick(executor, ctx) is not None
    ctx.held = False
    assert executor.next_action(ctx) is None
    assert executor.primitive.status == "lost_grasp"


def test_unsupported_place_never_opens():
    ctx = context()
    ctx.held = True
    executor = PickPlaceSkillExecution()
    executor.gripper_command = -1.
    executor.start(action("place"), ctx)
    rows = run(executor, ctx, support=False)
    assert rows and all(row[1][3] == -1. for row in rows)
    assert executor.primitive.status == "no_support"


def test_missing_grasp_and_already_holding_emit_one_preserving_tick():
    ctx = context()
    executor = PickPlaceSkillExecution()
    executor.gripper_command = -1.
    executor.start(action("place"), ctx)
    rows = run(executor, ctx)
    assert len(rows) == 1 and rows[0][1][3] == -1.
    assert executor.primitive.status == "missing_grasp"
    ctx.held = True
    executor.start(action("grasp"), ctx)
    rows = run(executor, ctx)
    assert len(rows) == 1 and rows[0][1][3] == -1.
    assert executor.primitive.status == "already_holding"


def test_grasp_requires_real_hold_and_four_per_control_checks():
    ctx = context()
    executor = PickPlaceSkillExecution()
    executor.start(action("grasp"), ctx)
    rows = run(executor, ctx, acquire=False)
    assert rows
    assert executor.primitive.status == "missing_grasp"
    assert "lift" not in [row[0] for row in rows]


def test_support_gate_requires_lower_frame_geometry_upward_force_and_low_speed():
    ctx = context()
    ctx.support = True
    assert lower_shelf_supported(ctx, ["generic"]) == ["generic"]
    ctx.objects["generic"].actor.linear_velocity[0] = .02
    assert not lower_shelf_supported(ctx, ["generic"])
    assert lower_shelf_supported(ctx, ["generic"], require_settled=False) == ["generic"]
    ctx.objects["generic"].actor.linear_velocity[:] = 0
    contacts = ctx.env.scene.get_contacts()
    contacts[0].points[0].position[2] = .672
    ctx.env.scene.get_contacts = lambda: contacts
    assert not lower_shelf_supported(ctx, ["generic"])
    contacts[0].points[0].position[2] = .330
    ctx.env.scene.get_pairwise_contact_forces = lambda *_: np.array([0., 0., -.1])
    assert not lower_shelf_supported(ctx, ["generic"])


def test_baseline_layouts_unchanged():
    assert (RAPS_ACTION_SCHEMA.action_dim, RAPS_COMPACT_ACTION_SCHEMA.action_dim,
            RAPS_SPLIT_ACTION_SCHEMA.action_dim) == (20, 12, 13)
    assert "grasp" not in RAPS_ACTION_SCHEMA.primitive_names
    assert "place" not in RAPS_SPLIT_ACTION_SCHEMA.primitive_names


def test_place_retracts_before_cross_shelf_yz_motion():
    ctx = context()
    ctx.ee_position[:] = [.45, -.06, .84]
    ctx.held = True
    executor = PickPlaceSkillExecution()
    executor.gripper_command = -1.
    executor.start(action("place"), ctx)
    initial_yz = ctx.ee_position[1:].copy()
    retreat_count = 0
    while executor.primitive.phase == "retreat":
        low = tick(executor, ctx)
        assert low is not None
        if executor.primitive.phase == "retreat":
            retreat_count += 1
            np.testing.assert_array_equal(ctx.ee_position[1:], initial_yz)
        else:
            assert ctx.ee_position[0] <= OUTSIDE_X + .012
    assert retreat_count > 0


def planned_execution(monkeypatch):
    from domain.robocasa import motion_planning
    from domain.robocasa.pick_place import PlannedPickPlaceSkillExecution

    ctx = context()
    qpos = np.r_[ctx.ee_position, np.zeros(6)]
    planned_goals = []
    model = SimpleNamespace()
    model.compute_forward_kinematics = lambda q: setattr(model, "q", np.asarray(q))
    model.get_link_pose = lambda _: SimpleNamespace(p=model.q[:3], q=np.array([1., 0., 0., 0.]))
    planner = SimpleNamespace(kin=SimpleNamespace(pmodel=model, end_link_idx=0),
                              pindices=np.arange(9), attachment_state=lambda: (dict(name="generic",
                                  tcp_to_object_pose=np.r_[ctx.objects["generic"].position_root-ctx.ee_position,
                                                          [1.,0.,0.,0.]].tolist()) if ctx.held else None),
                              support_descent_step=lambda *_args, **_kwargs: .005,
                              placement_bounds=ctx.env.motion_planner.placement_bounds)
    planner.gripper_command = -1.
    planner._grasp_contacts = lambda: {"generic": ctx.held}
    planner.check_attachment_feedback = lambda plan: motion_planning.CabinetMotionPlanner.check_attachment_feedback(planner, plan)
    ctx.env.set_grasp_retention = lambda mode: None
    ctx.env.motion_planner = planner
    ctx.env.agent = SimpleNamespace(robot=SimpleNamespace(get_qpos=lambda: qpos.copy()))

    def plan(goal, quaternion, **kwargs):
        planned_goals.append((np.asarray(goal).copy(), kwargs))
        target = qpos.copy()
        target[:3] = goal
        return SimpleNamespace(position=np.stack([qpos, target]), metadata={},
                               target_position_root=np.asarray(goal).copy())

    ctx.env.plan_to_pose = plan

    class Cursor:
        def __init__(self, raw, plan, *, gripper_command, max_steps):
            self.plan = plan
            self.gripper = gripper_command
            self.status = "running"
            self.records = []
            self.endpoint_position_error_m = self.endpoint_angle_error_deg = 0.

        def next_action(self):
            return np.r_[self.plan.position[-1, :7], self.gripper].astype(np.float32)

        def record_step(self, **kwargs):
            previous = ctx.ee_position.copy()
            qpos[:] = self.plan.position[-1]
            ctx.ee_position[:] = qpos[:3]
            if ctx.held:
                ctx.objects["generic"].position_root += ctx.ee_position - previous
            self.status = "success"
            row = {"tcp_quaternion_wxyz": [1., 0., 0., 0.]}
            self.records.append(row)
            return row

        def stop(self, status):
            self.status = status

    monkeypatch.setattr(motion_planning, "MotionPlanExecution", Cursor)
    monkeypatch.setattr(motion_planning, "cabinet_contact_forces", lambda *_: {})
    executor = PlannedPickPlaceSkillExecution(orientation_target=lambda: [1., 0., 0., 0.])
    executor._feedback_orientation = lambda: np.array([1., 0., 0., 0.])
    return executor, ctx, planned_goals


def run_planned(executor, ctx, *, support):
    rows = []
    for _ in range(2001):
        low = executor.next_action(ctx)
        if low is None:
            return rows
        assert low.shape == (8,)
        phase = executor.primitive.phase
        if phase == "close":
            ctx.held = True
        if phase == "release":
            ctx.held = False
        executor.record_action(low)
        if phase == "descend" and support and ctx.ee_position[2] <= .40:
            ctx.support = True
        rows.append((phase, low.copy()))
    pytest.fail("planned macro exceeded bounded control count")


def test_planned_macro_reuses_grasp_support_and_release_gates(monkeypatch):
    executor, ctx, plans = planned_execution(monkeypatch)
    executor.start(action("grasp"), ctx)
    rows = run_planned(executor, ctx, support=False)
    assert executor.primitive.status == "success"
    assert len([r for r in rows if r[0] == "close"]) == 30
    assert executor._grasp_frames["generic"] >= 4
    assert all(kwargs["maintain_orientation"] for _, kwargs in plans)
    executor.start(action("place"), ctx)
    rows = run_planned(executor, ctx, support=True)
    assert executor.primitive.status == "success"
    assert executor.support_stopped_object_ids == ["generic"]
    assert any(kwargs.get("densify_rrt_path") and not kwargs["maintain_orientation"]
               for _, kwargs in plans)
    assert all(row[1][-1] == -1. for row in rows if row[0] in {"preplace", "approach", "descend", "settle"})
    assert len([r for r in rows if r[0] == "release"]) == 9
    descent_plans = [(goal, kwargs) for goal, kwargs in plans if "support_surface_root_z" in kwargs]
    assert descent_plans and all(kwargs["support_surface_root_z"] == .330 for _, kwargs in descent_plans)
    assert all(kwargs["method"] == "bounded_ik" for _, kwargs in descent_plans)
    assert any(kwargs.get("method") == "bounded_ik" and "support_surface_root_z" not in kwargs for _, kwargs in plans)
    assert all(abs(a[0][2] - b[0][2]) <= .005001 for a, b in zip(descent_plans, descent_plans[1:]))


def test_planned_unsupported_place_never_opens(monkeypatch):
    executor, ctx, _ = planned_execution(monkeypatch)
    ctx.held = True
    executor.gripper_command = -1.
    executor.start(action("place"), ctx)
    rows = run_planned(executor, ctx, support=False)
    assert rows and all(row[1][-1] == -1. for row in rows)
    assert executor.primitive.status == "no_support"


@pytest.mark.parametrize("stage", ["clear", "withdraw", "home_raise", "home_align", "home"])
def test_released_cup_is_an_obstacle_during_egress(monkeypatch, stage):
    executor, ctx, plans = planned_execution(monkeypatch)
    ctx.held = True
    ctx.ee_position[:] = [.45, .15, .40]
    executor.gripper_command = -1.
    executor.start(action("place"), ctx)
    # The support gate releases its retention guard before opening.
    ctx.held = False
    executor._loss_guard = False
    executor._phase(stage, ctx)
    if stage == "clear":
        np.testing.assert_allclose(executor._anchor, [.39, .15, .40], atol=1e-7)
    goal = executor._anchor.copy()
    if np.linalg.norm(goal-ctx.ee_position) < .012:
        goal[0] -= .03
    executor._motion_action(ctx, goal, 1.)
    assert plans[-1][1]["allowed_contact_objects"] == ()
    if stage == "clear":
        assert plans[-1][1]["method"] == "bounded_ik"


def test_planned_first_plan_failure_emits_one_closed_hold(monkeypatch):
    from domain.robocasa.motion_planning import MotionPlanningError

    executor, ctx, _ = planned_execution(monkeypatch)
    ctx.held = True
    ctx.ee_position[0] = .45
    executor.gripper_command = -1.
    ctx.env.plan_to_pose = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        MotionPlanningError("no_collision_free_ik", "synthetic failure"))
    executor.start(action("place"), ctx)
    rows = run_planned(executor, ctx, support=False)
    assert len(rows) == 1 and rows[0][1][-1] == -1.
    assert executor.primitive.status == "no_collision_free_ik"


def test_planned_path_audit_rejects_intermediate_wrist_rotation(monkeypatch):
    from domain.robocasa.motion_planning import MotionPlanningError

    executor, ctx, _ = planned_execution(monkeypatch)
    executor.start(action("grasp"), ctx)
    model = ctx.env.motion_planner.kin.pmodel
    model.get_link_pose = lambda _: SimpleNamespace(p=model.q[:3], q=np.array([0., 1., 0., 0.]))
    plan = SimpleNamespace(position=np.zeros((2, 9)), metadata={})
    with pytest.raises(MotionPlanningError, match="orientation_path_violation"):
        executor._audit_path(plan)


def test_planned_reached_waypoint_skips_zero_length_plan_only_with_reset_orientation(monkeypatch):
    executor, ctx, plans = planned_execution(monkeypatch)
    executor.start(action("grasp"), ctx)
    low, reached = executor._motion_action(ctx, ctx.ee_position.copy(), 1.)
    assert low is None and reached
    assert not plans and not executor.motion_attempts
    executor._feedback_orientation = lambda: np.array([0., 1., 0., 0.])
    low, reached = executor._motion_action(ctx, ctx.ee_position.copy(), 1.)
    assert low.shape == (8,) and not reached
    assert len(plans) == 1


def test_planned_half_millimeter_support_step_is_not_skipped(monkeypatch):
    executor, ctx, plans = planned_execution(monkeypatch)
    ctx.held = True
    ctx.ee_position[:] = [.45, 0., .40]
    executor.gripper_command = -1.
    executor.start(action("place"), ctx)
    executor._phase("descend", ctx)
    ctx.env.motion_planner.support_descent_step = lambda *_args, **_kwargs: .0005
    low = executor._descent_action(ctx, ctx.ee_position.copy())
    assert low.shape == (8,)
    assert len(plans) == 1
    assert ctx.ee_position[2] - plans[0][0][2] == pytest.approx(.0005, abs=1e-7)


def test_retention_loss_is_detected_after_control_without_waiting_for_next_action():
    ctx=context()
    executor=PickPlaceSkillExecution()
    ctx.held=True
    executor.gripper_command=-1.
    executor.start(action("place"),ctx)
    low=executor.next_action(ctx)
    assert low is not None and low[-1]<0
    executor.record_action(low)
    ctx.held=False
    executor.check_retention(ctx)
    assert executor.primitive.status=="lost_grasp"
    assert executor.gripper_command<0


def test_release_command_is_rejected_without_settled_support():
    ctx=context()
    ctx.held=True
    executor=PickPlaceSkillExecution()
    executor.gripper_command=-1.
    executor.start(action("place"),ctx)
    executor._phase("release",ctx)
    assert executor.next_action(ctx) is None
    assert executor.primitive.status=="unsafe_release"
    assert executor.gripper_command<0


def test_retention_reference_survives_replanning_and_place_start(monkeypatch):
    executor,ctx,_=planned_execution(monkeypatch)
    executor.start(action("grasp"),ctx)
    run_planned(executor,ctx,support=False)
    anchor=executor._retention_attachment
    assert anchor is not None
    executor.start(action("place"),ctx)
    ctx.objects["generic"].position_root[0]+=.0015
    executor.check_retention(ctx)
    assert executor._pending_failure is None
    assert executor._retention_attachment is anchor
    ctx.objects["generic"].position_root[0]+=.001
    executor.check_retention(ctx)
    assert executor._pending_failure=="attachment_slip"
    low=executor.next_action(ctx)
    assert low[-1]<0
    executor.record_action(low)
    assert executor.next_action(ctx) is None
    assert executor.primitive.status=="attachment_slip"


def test_open_approach_stops_at_bilateral_contact_before_closing():
    ctx = context()
    executor = PickPlaceSkillExecution()
    executor.start(action("grasp"), ctx)
    while executor.primitive.phase != "approach":
        assert tick(executor, ctx, acquire=False) is not None
    contact_position = ctx.ee_position.copy()
    ctx.held = True
    low = executor.next_action(ctx)
    assert executor.primitive.phase == "close" and low[-1] == -1.
    np.testing.assert_array_equal(executor.primitive.executed_goals["approach"], contact_position)
    assert not executor._loss_guard


def test_exterior_preplace_rotation_restores_orientation_before_entry(monkeypatch):
    from domain.robocasa.motion_planning import MotionPlanningError
    executor, ctx, _ = planned_execution(monkeypatch)
    executor.start(action("place"), ctx)
    executor._phase("preplace", ctx)
    model = ctx.env.motion_planner.kin.pmodel
    model.get_link_pose = lambda _: SimpleNamespace(
        p=model.q[:3], q=np.array([1., 0., 0., 0.]) if model.q[3] == 0 else np.array([0., 1., 0., 0.]))
    points = np.zeros((3, 9))
    points[:, 0] = [.18, .20, .28]
    points[1, 3] = .1
    executor._audit_path(SimpleNamespace(position=points, metadata={}))
    points[1, 0] = .30
    with pytest.raises(MotionPlanningError, match="orientation_path_violation"):
        executor._audit_path(SimpleNamespace(position=points, metadata={}))
    points[1, 0] = .20
    points[-1, 3] = .1
    with pytest.raises(MotionPlanningError, match="orientation_path_violation"):
        executor._audit_path(SimpleNamespace(position=points, metadata={}))


def test_bound_support_uses_observed_substep_speed_and_friction_keeps_native_speed():
    ctx=context();ctx.support=True
    ctx.objects["generic"].actor.linear_velocity=np.array([.23,0.,0.])
    assert not lower_shelf_supported(ctx,["generic"])
    feedback={"max_substep_pose_speed_m_s":.001,"pose_speed_samples":5}
    ctx.env._grasp_binding=SimpleNamespace(bound_object_id="generic",feedback=lambda:feedback)
    assert lower_shelf_supported(ctx,["generic"]) == ["generic"]
    feedback["max_substep_pose_speed_m_s"] = .02
    assert not lower_shelf_supported(ctx,["generic"])
    feedback["max_substep_pose_speed_m_s"] = None
    feedback["pose_speed_samples"] = 0
    assert not lower_shelf_supported(ctx,["generic"])


def test_bound_hold_slip_is_reported_as_macro_failure(monkeypatch):
    from domain.robocasa.motion_planning import MotionPlanningError
    executor, ctx, _ = planned_execution(monkeypatch)
    executor.start(action("grasp"),ctx)
    def slipped():
        raise MotionPlanningError("grasp_binding_slip", {"position_error_m": .003})
    ctx.env._grasp_binding=SimpleNamespace(bound_object_id="generic",feedback=slipped)
    executor.record_action(np.r_[np.zeros(7), -1.].astype(np.float32))
    assert executor.primitive.status == "grasp_binding_slip"
    assert executor.last_motion_tick["grasp_retention_failure"]["detail"]["position_error_m"] == .003
