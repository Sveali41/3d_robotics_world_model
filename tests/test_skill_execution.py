from types import SimpleNamespace

import numpy as np
import pytest
import torch

from domain.robocasa.primitives import PrimitiveAction
from domain.robocasa.skill_execution import RAPSSkillExecution


def context(held=True, force=None):
    item = SimpleNamespace(actor=object(), position_root=np.zeros(3),
                           position_world=np.zeros(3))
    force = np.array([0., 0., 0.1]) if force is None else force
    return SimpleNamespace(
        ee_position=np.zeros(3), gripper_open=0.3,
        arm_controller=SimpleNamespace(ee_pose_at_base=SimpleNamespace(
            q=np.array([1., 0., 0., 0.]))),
        objects={"item": item}, is_grasping=lambda _: held,
        finger_contact_force=lambda _: (0., 0.),
        env=SimpleNamespace(ground=object(), scene=SimpleNamespace(
            get_pairwise_contact_forces=lambda actor, ground: force)),
    )


def start_motion(execution, ctx, primitive="lift"):
    execution.start(PrimitiveAction(primitive, {"z_dist": .2}), ctx)


@pytest.mark.parametrize("gripper, expected", [("close_gripper", -1.),
                                               ("open_gripper", 1.)])
def test_recorded_gripper_survives_motion(gripper, expected):
    execution = RAPSSkillExecution(latched_gripper=True)
    ctx = context()
    execution.start(PrimitiveAction(gripper, {}), ctx)
    requested = execution.next_action(ctx)
    assert requested[3] == expected
    assert execution.gripper_command == 1.
    execution.record_action(requested)
    start_motion(execution, ctx)
    assert execution.next_action(ctx)[3] == expected
    execution.reset()
    assert execution.gripper_command == 1.
    assert execution.primitive is None
    assert execution.support_stopped_object_ids == []


@pytest.mark.parametrize("primitive", ["open_gripper", "close_gripper"])
@pytest.mark.parametrize("tensor", [False, True])
def test_controlled_gripper_captures_normalized_actual_orientation(primitive, tensor):
    execution = controlled_execution()
    ctx = context()
    quaternion = np.array([[0., 2., 0., 0.]]) if tensor else np.array([0., 2., 0., 0.])
    ctx.arm_controller.ee_pose_at_base.q = torch.tensor(quaternion) if tensor else quaternion
    execution.start(PrimitiveAction(primitive, {}), ctx)
    np.testing.assert_array_equal(execution.orientation_hold, [0., 1., 0., 0.])
    ctx.arm_controller.ee_pose_at_base.q = np.array([0., 0., 0., 1.])
    execution.next_action(ctx)
    np.testing.assert_array_equal(execution.orientation_hold, [0., 1., 0., 0.])
    execution.start(PrimitiveAction(primitive, {}), ctx)
    np.testing.assert_array_equal(execution.orientation_hold, [0., 0., 0., 1.])
    execution.reset()
    assert execution.orientation_hold is None


def test_settling_captures_current_orientation_at_first_support():
    execution = controlled_execution()
    ctx = context(force=[0., 0., 0.])
    start_motion(execution, ctx, "drop")
    assert execution.orientation_hold is None
    execution.observe(ctx)
    assert execution.orientation_hold is None
    ctx.arm_controller.ee_pose_at_base.q = torch.tensor([[0., 0., 3., 0.]])
    ctx.env.scene.get_pairwise_contact_forces = lambda *_: [0., 0., .1]
    execution.observe(ctx)
    np.testing.assert_array_equal(execution.orientation_hold, [0., 0., 1., 0.])
    ctx.arm_controller.ee_pose_at_base.q = np.array([1., 0., 0., 0.])
    actual_tick(execution, ctx)
    execution.observe(ctx)
    np.testing.assert_array_equal(execution.orientation_hold, [0., 0., 1., 0.])


@pytest.mark.parametrize("primitive,dz,z,holds", [
    ("lift", .1, .1, True), ("lift", .1, .15, False),
    ("move_delta_ee_pose", .1, .1, True),
    ("move_delta_ee_pose", .1, .2, False),
    ("move_delta_ee_pose", -.1, .1, False),
    ("move_delta_ee_pose", 0., .1, False),
])
def test_controlled_low_upward_retreat_captures_actual_orientation(primitive, dz, z, holds):
    execution = controlled_execution()
    ctx = context()
    ctx.ee_position[2] = z
    ctx.arm_controller.ee_pose_at_base.q = np.array([0., 0., 0., 2.])
    parameters = {"z_dist": dz} if primitive == "lift" else {"dx": 0., "dy": 0., "dz": dz}
    execution.start(PrimitiveAction(primitive, parameters), ctx)
    if holds:
        np.testing.assert_array_equal(execution.orientation_hold, [0., 0., 0., 1.])
    else:
        assert execution.orientation_hold is None
    start_motion(execution, ctx, "drop")
    assert execution.orientation_hold is None


@pytest.mark.parametrize("quaternion", [[0., 0., 0., 0.], [np.nan, 0., 0., 0.],
                                       [1., 0., 0.], [[1., 0., 0., 0.]] * 2])
def test_controlled_orientation_rejects_invalid_quaternion(quaternion):
    execution = controlled_execution()
    ctx = context()
    ctx.arm_controller.ee_pose_at_base.q = np.array(quaternion)
    with pytest.raises(ValueError, match="Orientation hold"):
        execution.start(PrimitiveAction("open_gripper", {}), ctx)


def test_legacy_does_not_require_or_override_orientation():
    execution = RAPSSkillExecution()
    ctx = context()
    del ctx.arm_controller
    execution.start(PrimitiveAction("open_gripper", {}), ctx)
    assert execution.orientation_hold is None


def test_controlled_drop_slows_from_vertical_distance_despite_lateral_error():
    execution = controlled_execution()
    ctx = context()
    start_motion(execution, ctx, "drop")
    ctx.ee_position = np.array([.1, 0., -.19])
    assert np.linalg.norm(execution.next_action(ctx)[:3]) == pytest.approx(.03125)
    ctx.ee_position[2] = -.21
    assert np.linalg.norm(execution.next_action(ctx)[:3]) == pytest.approx(.025)


def test_latch_records_actual_action_only():
    execution = RAPSSkillExecution(latched_gripper=True)
    ctx = context()
    execution.start(PrimitiveAction("close_gripper", {}), ctx)
    assert execution.next_action(ctx)[3] == -1.
    execution.record_action([0., 0., 0., .25])
    start_motion(execution, ctx)
    assert execution.next_action(ctx)[3] == .25


@pytest.mark.parametrize("action", [np.zeros(3), np.zeros((1, 4)),
                                   [0., 0., 0., np.nan]])
def test_invalid_recorded_action(action):
    with pytest.raises(ValueError, match="finite shape"):
        RAPSSkillExecution().record_action(action)


def test_default_uses_measured_fraction():
    execution = RAPSSkillExecution()
    ctx = context()
    start_motion(execution, ctx)
    assert execution.next_action(ctx)[3] == pytest.approx(-.4)
    assert not hasattr(ctx, "gripper_command")


@pytest.mark.parametrize("stop_on_contact", [True, False])
def test_grasp_contact_stop_flag_preserves_feedback_and_goal_closure(stop_on_contact):
    execution = RAPSSkillExecution(grasp_stop_on_contact=stop_on_contact)
    ctx = context()
    ctx.finger_contact_force = lambda _: (.1, .1)
    execution.start(PrimitiveAction("top_grasp", dict(x_forward=.1, y_offset=0., z_offset=0.)), ctx)
    execution.primitive.stage = "approach"
    snapshot = execution.observe(ctx)
    assert snapshot["objects"]["item"]["left_force"] == .1
    assert execution.primitive.stage == ("close" if stop_on_contact else "approach")
    if not stop_on_contact:
        ctx.ee_position = execution.primitive.grasp_goal.copy()
        execution.next_action(ctx)
        assert execution.primitive.stage == "close"
    assert execution.next_action(ctx)[3] == -1.


@pytest.mark.parametrize("primitive, expected", [("lift", -1.),
                                                 ("open_gripper", 1.)])
def test_legacy_hold_closed(primitive, expected):
    execution = RAPSSkillExecution(hold_closed_during_motion=True)
    ctx = context()
    execution.start(PrimitiveAction(primitive, {"z_dist": .2}), ctx)
    assert execution.next_action(ctx)[3] == expected


def test_latch_takes_precedence_over_legacy_hold_closed():
    execution = RAPSSkillExecution(latched_gripper=True,
                                   hold_closed_during_motion=True)
    ctx = context()
    start_motion(execution, ctx)
    assert execution.next_action(ctx)[3] == 1.


@pytest.mark.parametrize("tensor", [False, True])
def test_drop_stops_on_current_support_even_after_grasp_lost(tensor):
    execution = RAPSSkillExecution(stop_drop_on_support=True)
    ctx = context(force=np.array([[0., 0., 0.]]))
    start_motion(execution, ctx, "drop")
    assert execution.next_action(ctx) is not None
    execution.observe(ctx)
    assert not execution.primitive.done
    ctx.is_grasping = lambda _: False
    force = [[0., 0., .051]]
    force = torch.tensor(force) if tensor else np.array(force)
    calls = []

    def contact(actor, ground):
        calls.append((actor, ground))
        return force

    ctx.env.scene.get_pairwise_contact_forces = contact
    execution.observe(ctx)
    assert calls == [(ctx.objects["item"].actor, ctx.env.ground)]
    assert execution.primitive.status == "supported"
    assert execution.support_stopped_object_ids == ["item"]
    assert execution.next_action(ctx) is None


@pytest.mark.parametrize("force", [[0., 0., .05], [0., 0., -.2]])
def test_drop_requires_upward_force_above_threshold(force):
    execution = RAPSSkillExecution(stop_drop_on_support=True)
    ctx = context(force=force)
    start_motion(execution, ctx, "drop")
    execution.observe(ctx)
    assert not execution.primitive.done


def test_drop_does_not_stop_for_unheld_objects():
    execution = RAPSSkillExecution(stop_drop_on_support=True)
    ctx = context(held=False)
    start_motion(execution, ctx, "drop")
    ctx.is_grasping = lambda _: True
    execution.observe(ctx)
    assert not execution.primitive.done
    assert execution.support_stopped_object_ids == []


def test_drop_missing_ground_fails_clearly():
    execution = RAPSSkillExecution(stop_drop_on_support=True)
    ctx = context()
    del ctx.env.ground
    start_motion(execution, ctx, "drop")
    with pytest.raises(RuntimeError, match="env.ground"):
        execution.observe(ctx)


@pytest.mark.parametrize("force", [[0., 0., np.nan], [0., 0., np.inf],
                                  [1., 2.], [], None])
def test_invalid_support_force_fails_clearly(force):
    execution = RAPSSkillExecution(stop_drop_on_support=True)
    ctx = context()
    ctx.env.scene.get_pairwise_contact_forces = lambda *_: force
    start_motion(execution, ctx, "drop")
    with pytest.raises(ValueError, match="ground contact force"):
        execution.observe(ctx)


def test_default_drop_does_not_query_support():
    execution = RAPSSkillExecution()
    ctx = context()
    del ctx.env
    start_motion(execution, ctx, "drop")
    execution.observe(ctx)
    assert not execution.primitive.done


def controlled_execution():
    return RAPSSkillExecution(latched_gripper=True, stop_drop_on_support=True,
                              controlled_placement=True)


def actual_tick(execution, ctx):
    action = execution.next_action(ctx)
    assert action is not None
    execution.record_action(action)


@pytest.mark.parametrize("latched,support", [(False, False), (False, True),
                                            (True, False)])
def test_controlled_placement_requires_flags(latched, support):
    with pytest.raises(ValueError, match="requires latched_gripper"):
        RAPSSkillExecution(latched_gripper=latched, stop_drop_on_support=support,
                           controlled_placement=True)


def test_controlled_drop_slows_near_goal_and_uses_smaller_tolerance():
    ctx = context()
    execution = controlled_execution()
    start_motion(execution, ctx, "drop")
    assert np.linalg.norm(execution.next_action(ctx)[:3]) == pytest.approx(.25)
    ctx.ee_position = np.array([0., 0., -.16])
    assert np.linalg.norm(execution.next_action(ctx)[:3]) == pytest.approx(.125)
    ctx.ee_position[2] = -.19
    assert execution.next_action(ctx) is not None
    legacy = RAPSSkillExecution()
    ctx.ee_position = np.zeros(3)
    start_motion(legacy, ctx, "drop")
    ctx.ee_position = np.array([0., 0., -.16])
    assert np.linalg.norm(legacy.next_action(ctx)[:3]) == pytest.approx(.25)
    ctx.ee_position[2] = -.19
    assert legacy.next_action(ctx) is None
    assert legacy.primitive.status == "success"


def test_controlled_contact_requires_four_stable_observations():
    execution = controlled_execution()
    ctx = context()
    execution.record_action([0., 0., 0., -1.])
    start_motion(execution, ctx, "drop")
    execution.observe(ctx)
    assert execution.primitive.stage == "settle_support"
    assert execution.primitive.status == "running"
    assert execution.settle_count == 0
    assert execution.support_stopped_object_ids == []
    execution.observe(ctx)
    assert execution.settle_count == 0  # No actual elapsed control tick.
    for count in range(1, 5):
        actual_tick(execution, ctx)
        execution.observe(ctx)
        assert execution.settle_count == count
        assert execution.primitive.status == ("supported" if count == 4 else "running")
    assert execution.support_stopped_object_ids == ["item"]
    assert execution.next_action(ctx) is None


@pytest.mark.parametrize("disturbance", ["support", "anchor", "speed"])
def test_settle_disturbance_resets_stability_and_keeps_anchor(disturbance):
    execution = controlled_execution()
    ctx = context()
    start_motion(execution, ctx, "drop")
    execution.observe(ctx)
    anchor = execution.settle_goal.copy()
    actual_tick(execution, ctx)
    execution.observe(ctx)
    assert execution.settle_count == 1
    if disturbance == "support":
        ctx.env.scene.get_pairwise_contact_forces = lambda *_: [0., 0., 0.]
    elif disturbance == "anchor":
        ctx.ee_position = np.array([.004, 0., 0.])
    else:
        ctx.ee_position = np.array([.001, 0., 0.])
    actual_tick(execution, ctx)
    execution.observe(ctx)
    assert execution.settle_count == 0
    np.testing.assert_array_equal(execution.settle_goal, anchor)
    ctx.ee_position = np.array([.001, 0., 0.])
    correction = execution.next_action(ctx)
    assert correction[0] == pytest.approx(-.01)


def test_settling_uses_actual_elapsed_ticks_for_speed():
    execution = controlled_execution()
    ctx = context()
    start_motion(execution, ctx, "drop")
    execution.observe(ctx)
    for _ in range(2):
        actual_tick(execution, ctx)
    ctx.ee_position = np.array([.0009, 0., 0.])
    execution.observe(ctx)
    assert execution.settle_count == 1  # .009 m/s over two .05 s ticks.


def test_controlled_settle_times_out_after_twenty_actual_ticks():
    execution = controlled_execution()
    ctx = context()
    start_motion(execution, ctx, "drop")
    execution.observe(ctx)
    ctx.env.scene.get_pairwise_contact_forces = lambda *_: [0., 0., 0.]
    for _ in range(20):
        actual_tick(execution, ctx)
        execution.observe(ctx)
    assert execution.primitive.status == "support_unstable"
    assert execution.support_stopped_object_ids == []
    assert execution.next_action(ctx) is None


def test_settle_primitive_timeout_is_unstable():
    execution = controlled_execution()
    ctx = context()
    start_motion(execution, ctx, "drop")
    execution.observe(ctx)
    execution.primitive.steps = execution.primitive.max_steps
    assert execution.next_action(ctx) is None
    assert execution.primitive.status == "support_unstable"


@pytest.mark.parametrize("primitive,grip", [("open_gripper", 1.),
                                           ("close_gripper", -1.)])
def test_controlled_gripper_holds_fixed_anchor_without_deadband(primitive, grip):
    execution = controlled_execution()
    ctx = context()
    execution.start(PrimitiveAction(primitive, {}), ctx)
    ctx.ee_position = np.array([.0001, 0., 0.])
    action = execution.next_action(ctx)
    assert action[0] == pytest.approx(-.001)
    assert action[3] == grip
    ctx.ee_position = np.array([.1, 0., 0.])
    action = execution.next_action(ctx)
    assert action[0] == pytest.approx(-.25)
    assert action[3] == grip
    for _ in range(7):
        assert execution.next_action(ctx) is not None
    assert execution.next_action(ctx) is None
    assert execution.primitive.steps == 10


def test_controlled_reset_clears_per_skill_state_but_start_preserves_latch():
    execution = controlled_execution()
    ctx = context()
    execution.record_action([0., 0., 0., -1.])
    start_motion(execution, ctx, "drop")
    assert execution.executed_steps == 0
    assert execution.gripper_command == -1.
    execution.observe(ctx)
    actual_tick(execution, ctx)
    execution.observe(ctx)
    assert execution.settle_count == 1
    execution.start(PrimitiveAction("open_gripper", {}), ctx)
    assert execution.settle_goal is None
    assert execution.settle_count == execution.executed_steps == 0
    assert execution.gripper_command == -1.
    execution.reset()
    assert execution.primitive is None
    assert execution._hold_goal is None
    assert execution.settle_goal is None
    assert execution.settle_count == execution.executed_steps == 0
    assert execution.gripper_command == 1.
