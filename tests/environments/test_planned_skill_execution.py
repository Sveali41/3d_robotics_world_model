from types import SimpleNamespace

import numpy as np
import pytest
import torch

from domain.robocasa.motion_planning import MotionPlan, MotionPlanningError
from domain.robocasa.primitives import PrimitiveAction
from domain.robocasa.skill_execution import PlannedSkillExecution


class Identity:
    def inv(self):
        return self

    def __mul__(self, other):
        return other


def fixture():
    qpos = torch.tensor([[.35, -.06, .72, 0., 0., 0., 0., .04, .04]])
    pose = SimpleNamespace(p=qpos[0, :3].clone(), q=torch.tensor([1., 0., 0., 0.]))
    robot = SimpleNamespace(get_qpos=lambda: qpos,
                            get_qlimits=lambda: torch.tensor([[[-3., 3.]] * 7 + [[0., .04]] * 2]),
                            pose=Identity(),
                            get_links=lambda: [SimpleNamespace(name="arm")])
    raw = SimpleNamespace(agent=SimpleNamespace(robot=robot, tcp=SimpleNamespace(pose=pose),
        _controller_configs={"pd_joint_pos": {"gripper": SimpleNamespace(lower=-.01, upper=.04)}}),
        cabinet=SimpleNamespace(get_links=lambda: [SimpleNamespace(name="frame")]),
        scene=SimpleNamespace(get_pairwise_contact_forces=lambda *_: torch.zeros(1, 3)))
    raw.motion_planner = SimpleNamespace(assert_plan_start=lambda _: None,
        check_attachment_feedback=lambda _: None, attachment_state=lambda: None)
    calls, emitted = [], []

    def plan_to_pose(position, quaternion, **kwargs):
        initial = qpos.numpy().reshape(9).copy()
        goal = initial.copy()
        goal[:3] = position
        bend = (initial + goal) / 2
        bend[3] = .2
        plan = MotionPlan(np.stack([initial, bend, goal]), np.array([0., .05, .10]),
            np.asarray(position).copy(), np.asarray(quaternion).copy(), initial,
            np.array([0., 0., 0., 1., 0., 0., 0.]), {}, {})
        calls.append(dict(position=np.asarray(position).copy(), quaternion=np.asarray(quaternion).copy(), **kwargs))
        return plan

    raw.plan_to_pose = plan_to_pose
    ctx = SimpleNamespace(env=raw, ee_position=qpos.numpy().reshape(9)[:3].copy(),
        gripper_open=1., objects={"cup": SimpleNamespace(position_root=np.zeros(3), position_world=np.zeros(3))},
        finger_contact_force=lambda _: (0., 0.), is_grasping=lambda _: False)
    execution = PlannedSkillExecution(orientation_target=lambda: np.array([1., 0., 0., 0.]),
        latched_gripper=True, grasp_stop_on_contact=False)

    def step(action):
        emitted.append(action.copy())
        qpos[0, :7] = torch.as_tensor(action[:7])
        pose.p = qpos[0, :3].clone()
        ctx.ee_position = qpos.numpy().reshape(9)[:3].copy()
        execution.record_action(action)
        execution.observe(ctx)

    return raw, ctx, execution, calls, emitted, step


@pytest.mark.parametrize("name,parameters,offset", [
    ("move_delta_ee_pose", dict(dx=.04, dy=-.02, dz=.03), [.04, -.02, .03]),
    ("align_yz", dict(y_offset=.02, z_offset=.03), [0., .02, .03]),
    ("approach_x", dict(x_forward=-.04), [.04, 0., 0.]),
    ("lift", dict(z_dist=.03), [0., 0., .03]),
    ("drop", dict(z_dist=.03), [0., 0., -.03]),
    ("move_left", dict(y_dist=.03), [0., -.03, 0.]),
    ("move_right", dict(y_dist=.03), [0., .03, 0.]),
    ("move_forward", dict(x_dist=.03), [.03, 0., 0.]),
    ("move_backward", dict(x_dist=.03), [-.03, 0., 0.]),
    ("approach", dict(x_forward=.04, y_offset=.02, z_offset=.03), [.04, .02, .03]),
])
def test_all_motion_primitives_use_checked_joint_paths_without_changing_targets(name, parameters, offset):
    raw, ctx, execution, calls, emitted, step = fixture()
    initial = ctx.ee_position.copy()
    execution.start(PrimitiveAction(name, parameters), ctx)
    for _ in range(100):
        action = execution.next_action(ctx)
        if action is None:
            break
        assert action.shape == (8,)
        step(action)
    assert execution.primitive.done
    assert execution.primitive.status == ("no_stable_grasp" if name == "approach" else "success")
    np.testing.assert_allclose(calls[-1]["position"], initial + offset)
    np.testing.assert_allclose(ctx.ee_position, initial + offset)
    assert len(calls) == (2 if name == "approach" else 1)
    assert all(call["allowed_contact_objects"] == ("cup",) for call in calls)
    assert all(np.array_equal(call["quaternion"], [1., 0., 0., 0.]) for call in calls)
    assert any(action[3] == pytest.approx(.2) for action in emitted)
    if name != "approach":
        assert all(action[-1] == 1. for action in emitted)
    else:
        assert [call["seed"] for call in calls] == [0, 1]
        assert any(action[-1] == -1. for action in emitted)


@pytest.mark.parametrize("name,gripper", [("open_gripper", 1.), ("close_gripper", -1.)])
def test_gripper_primitives_hold_joints_and_remain_separate(name, gripper):
    raw, ctx, execution, calls, emitted, step = fixture()
    initial = raw.agent.robot.get_qpos().numpy().reshape(9)[:7].copy()
    execution.start(PrimitiveAction(name, {}), ctx)
    for _ in range(12):
        action = execution.next_action(ctx)
        if action is None:
            break
        step(action)
    assert not calls
    assert len(emitted) == 9
    assert execution.primitive.status == "success"
    for action in emitted:
        np.testing.assert_array_equal(action[:7], initial)
        assert action[-1] == gripper


def test_planning_failure_records_exact_target_and_never_issues_a_fallback_action():
    raw, ctx, execution, calls, emitted, step = fixture()
    def fail(*args, **kwargs):
        raise MotionPlanningError("no_collision_free_ik", "blocked endpoint")
    raw.plan_to_pose = fail
    execution.start(PrimitiveAction("approach_x", dict(x_forward=.04)), ctx)
    assert execution.next_action(ctx) is None
    assert execution.primitive.status == "no_collision_free_ik"
    assert execution.executed_steps == 0 and not emitted
    assert execution.motion_attempts[0]["detail"] == "blocked endpoint"


def test_payload_loss_preserves_real_tick_without_continuing():
    raw, ctx, execution, calls, emitted, step = fixture()
    def lost(_):
        raise MotionPlanningError("grasp_lost", "payload fell")
    raw.motion_planner.check_attachment_feedback = lost
    execution.start(PrimitiveAction("lift", dict(z_dist=.03)), ctx)
    step(execution.next_action(ctx))
    assert execution.primitive.status == "grasp_lost"
    assert execution.next_action(ctx) is None
    assert len(emitted) == 1
    assert execution.last_motion_tick["attachment_error"] == "payload fell"


def test_contact_stop_cancels_planned_approach_without_closing():
    raw, ctx, execution, calls, emitted, step = fixture()
    execution.grasp_stop_on_contact = True
    execution.start(PrimitiveAction("approach_x", dict(x_forward=.04)), ctx)
    step(execution.next_action(ctx))
    ctx.finger_contact_force = lambda _: (.1, .1)
    execution.observe(ctx)
    assert execution.next_action(ctx) is None
    assert execution.gripper_command == 1.
    assert execution.motion_attempts[-1]["status"] == "contact_stop"


def test_joint_record_has_native_units_and_rejects_ee_action_shapes():
    from exploration.trajectory_recorder import PrimitiveRecord
    record = PrimitiveRecord(1, 1, "joint_position", np.zeros(8),
        schema_version="cabinet-joint-position-v1", dense_trajectory=False,
        status="success", termination_reason="success")
    record.append_compact_transition(np.zeros(8), contact_event={"objects": {}})
    record.validate()
    assert record.duration_low_steps == 1
    with pytest.raises(ValueError, match="shape"):
        record.append_compact_transition(np.zeros(4))
    legacy = PrimitiveRecord(1, 1, "ee_delta", np.zeros(4),
        schema_version="cabinet-ee-delta-v1", dense_trajectory=False)
    with pytest.raises(ValueError, match="shape"):
        legacy.append_compact_transition(np.zeros(8))


def test_all_native_steps_respect_skill_budget_including_stationary_gripper():
    raw, ctx, execution, calls, emitted, step = fixture()
    execution.max_control_steps = 1
    execution.start(PrimitiveAction("close_gripper", {}), ctx)
    step(execution.next_action(ctx))
    assert execution.next_action(ctx) is None
    assert execution.primitive.status == "budget_exhausted"
    assert len(emitted) == 1


def test_native_episode_end_is_passed_to_checked_tracker():
    raw, ctx, execution, calls, emitted, step = fixture()
    execution.start(PrimitiveAction("approach_x", dict(x_forward=.04)), ctx)
    action = execution.next_action(ctx)
    execution.record_action(action, truncated=True)
    assert execution.next_action(ctx) is None
    assert execution.primitive.status == "truncated"
    assert execution.last_motion_tick["truncated"]


def test_planned_and_ee_checkpoint_contracts_are_distinct():
    from exploration.raps.raps_gnn_actor import DEFAULTS, action_contract, action_contract_matches
    settings = dict(DEFAULTS, skill_set="split7", latched_gripper=True, motion_execution="planner")
    planned = action_contract(settings)
    ee_settings = dict(settings, motion_execution="ee_delta")
    assert action_contract_matches(planned, settings)
    assert not action_contract_matches(planned, ee_settings)
    assert not action_contract_matches(action_contract(ee_settings), settings)
    assert planned["motion_execution"]["low_level_action"] == "seven_joint_targets_rad_plus_normalized_gripper"


@pytest.mark.parametrize("name,parameters,budget", [
    ("approach_x", dict(x_forward=.04), 18),
    ("close_gripper", {}, 9),
])
def test_completion_on_last_allowed_native_tick_is_not_mislabeled_as_budget_failure(name, parameters, budget):
    raw, ctx, execution, calls, emitted, step = fixture()
    execution.max_control_steps = budget
    execution.start(PrimitiveAction(name, parameters), ctx)
    for _ in range(budget):
        step(execution.next_action(ctx))
    assert execution.next_action(ctx) is None
    assert execution.primitive.status == "success"
    assert len(emitted) == budget
