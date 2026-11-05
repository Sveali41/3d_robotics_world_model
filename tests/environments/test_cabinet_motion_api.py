"""Execution lifecycle checks without native planning or physical simulation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from domain.robocasa.motion_planning import MotionPlan, MotionPlanExecution, MotionPlanningError
from domain.wall_cabinet_env import WallCabinetEnv


class IdentityRoot:
    def inv(self):
        return self

    def __mul__(self, pose):
        return pose


def execution_fixture():
    initial = np.r_[np.zeros(7), .04, .04]
    positions = np.stack([initial, initial + np.r_[np.full(7, .01), 0, 0]])
    goal = np.array([.4, 0, .7])
    plan = MotionPlan(positions, np.array([0., .05]), goal, np.array([1., 0, 0, 0]),
                      initial, np.array([0, 0, 0, 1, 0, 0, 0]), {}, {})
    resets, actions = [], []
    agent = SimpleNamespace(control_mode="pd_ee_delta_pose")
    agent.controller = SimpleNamespace(reset=lambda: resets.append(agent.control_mode))
    agent.set_control_mode = lambda mode: setattr(agent, "control_mode", mode)
    agent._controller_configs = {"pd_joint_pos": {"gripper": SimpleNamespace(lower=-.01, upper=.04)}}
    qpos = torch.tensor(initial[None, :], dtype=torch.float32)
    qlimits = torch.tensor([[[-3., 3.]] * 7 + [[0., .04], [0., .04]]])
    agent.robot = SimpleNamespace(pose=IdentityRoot(), get_qpos=lambda: qpos,
                                  get_qlimits=lambda: qlimits,
                                  get_links=lambda: [SimpleNamespace(name="arm")])
    agent.tcp = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor(goal), q=torch.tensor([1., 0, 0, 0])))
    planner = SimpleNamespace(assert_plan_start=lambda plan: None, check_attachment_feedback=lambda _: None)
    raw = SimpleNamespace(agent=agent, motion_planner=planner,
                          cabinet=SimpleNamespace(get_links=lambda: [SimpleNamespace(name="frame")]))
    raw.scene = SimpleNamespace(get_pairwise_contact_forces=lambda *_: torch.zeros((1, 3)))

    def step(action):
        actions.append(action.copy())
        qpos[0, :7] = torch.from_numpy(action[0, :7])
        return {"state": qpos.clone()}, 0., np.array([False]), np.array([False]), {}

    raw.step = step
    return raw, plan, actions, resets


def test_joint_execution_records_actual_controls_and_restores_ee_controller():
    raw, plan, actions, resets = execution_fixture()
    result = WallCabinetEnv.execute_motion_plan(raw, plan)
    assert result["status"] == "success"
    assert result["schema"] == "cabinet-joint-plan-execution-v1"
    assert result["control_steps"] == len(actions) == 17
    assert resets == ["pd_joint_pos", "pd_ee_delta_pose"]
    assert raw.agent.control_mode == "pd_ee_delta_pose"
    assert result["ticks"][-1]["stage"] == "settle"
    for row, action in zip(result["ticks"], actions):
        assert action.shape == (1, 8)
        np.testing.assert_array_equal(row["controller_action"], action[0])
        np.testing.assert_array_equal(row["qpos"][:7], action[0, :7])
        assert row["gripper_command"] == pytest.approx(1.)


def test_macro_budget_counts_all_control_ticks_including_settle():
    raw, plan, actions, resets = execution_fixture()
    result = WallCabinetEnv.execute_motion_plan(raw, plan, max_steps=3)
    assert result["status"] == "budget_exhausted"
    assert len(actions) == result["control_steps"] == 3
    assert result["ticks"][-1]["stage"] == "settle"
    assert resets[-1] == "pd_ee_delta_pose"


def test_callback_error_still_restores_original_controller():
    raw, plan, actions, resets = execution_fixture()

    def fail(row):
        raise RuntimeError("trace sink failed")

    with pytest.raises(RuntimeError, match="trace sink failed"):
        WallCabinetEnv.execute_motion_plan(raw, plan, on_step=fail)
    assert len(actions) == 1
    assert resets[-1] == raw.agent.control_mode == "pd_ee_delta_pose"


def test_stale_plan_and_unplanned_gripper_change_do_not_step():
    raw, plan, actions, resets = execution_fixture()
    with pytest.raises(MotionPlanningError, match="gripper_plan_mismatch"):
        WallCabinetEnv.execute_motion_plan(raw, plan, gripper_command=-1.)

    def stale(plan):
        raise MotionPlanningError("stale_plan", "scene moved")

    raw.motion_planner.assert_plan_start = stale
    with pytest.raises(MotionPlanningError, match="stale_plan"):
        WallCabinetEnv.execute_motion_plan(raw, plan)
    assert not actions and not resets


def test_tracking_error_is_reported_instead_of_success():
    raw, plan, _, _ = execution_fixture()
    raw.agent.tcp.pose.p[0] += .01
    result = WallCabinetEnv.execute_motion_plan(raw, plan)
    assert result["status"] == "tracking_error"
    assert result["endpoint_position_error_m"] == pytest.approx(.01)


@pytest.mark.parametrize("stop", ["force", "attached", "terminated"])
def test_feedback_stops_and_returns_executed_prefix(stop):
    raw, plan, actions, resets = execution_fixture()
    expected_steps = 1
    if stop == "force":
        raw.scene.get_pairwise_contact_forces = lambda *_: torch.tensor([[21., 0, 0]])
        expected_status, expected_steps = "tracking_blocked", 3
    elif stop == "attached":
        def attached(plan):
            raise MotionPlanningError("unexpected_attachment", "cup")
        raw.motion_planner.check_attachment_feedback = attached
        expected_status = "unexpected_attachment"
    else:
        step = raw.step
        def terminate(action):
            obs, reward, _, truncated, info = step(action)
            return obs, reward, np.array([True]), truncated, info
        raw.step = terminate
        expected_status = "terminated"
    result = WallCabinetEnv.execute_motion_plan(raw, plan)
    assert result["status"] == expected_status
    assert result["control_steps"] == len(actions) == expected_steps
    assert resets[-1] == "pd_ee_delta_pose"


@pytest.mark.parametrize("lose_grasp", [False, True])
def test_carry_keeps_closing_pressure_and_stops_on_grasp_loss(lose_grasp):
    raw, plan, actions, resets = execution_fixture()
    plan.position[:, 7:] = .02
    plan.initial_qpos[7:] = .02
    attachment = dict(name="cup", gripper_command=-1., tcp_to_object_pose=[0, 0, 0, 1, 0, 0, 0])
    plan.metadata["attachment"] = attachment
    restored_commands = []
    raw.device = "cpu"
    raw.cabinet_items = [SimpleNamespace(name="cup")]
    raw.agent.controller.controllers = {"gripper": SimpleNamespace(
        set_action=lambda action: restored_commands.append(action.clone()))}

    def feedback(plan):
        if lose_grasp and len(actions) >= 2:
            raise MotionPlanningError("grasp_lost", "cup")
        return attachment

    raw.motion_planner.check_attachment_feedback = feedback
    result = WallCabinetEnv.execute_motion_plan(raw, plan)
    assert result["status"] == ("grasp_lost" if lose_grasp else "success")
    assert len(actions) == (2 if lose_grasp else 17)
    assert all(action[0, -1] == -1 for action in actions)
    assert result["ticks"][0]["attachment"]["name"] == "cup"
    assert restored_commands[-1].item() == -1.
    assert raw.agent.control_mode == resets[-1] == "pd_ee_delta_pose"


def test_cursor_reads_targets_without_stepping_or_advancing():
    raw, plan, actions, resets = execution_fixture()
    execution = MotionPlanExecution(raw, plan, max_steps=2)
    first = execution.next_action()
    assert first.shape == (8,) and first.dtype == np.float32
    np.testing.assert_array_equal(execution.next_action(), first)
    assert execution.status == "running" and execution.records == []
    assert not actions and not resets
    raw.step(first[None, :])
    row = execution.record_step()
    assert row["step"] == 1
    np.testing.assert_array_equal(execution.next_action()[:7], plan.position[1, :7].astype(np.float32))
    assert len(actions) == 1 and not resets
    raw.step(execution.next_action()[None, :])
    execution.record_step()
    assert execution.status == "budget_exhausted"
    assert execution.next_action() is None
    assert execution.result()["control_steps"] == 2


def test_cursor_rejects_stale_start_without_controller_or_physics_changes():
    raw, plan, actions, resets = execution_fixture()
    def stale(plan):
        raise MotionPlanningError("stale_plan", "scene moved")
    raw.motion_planner.assert_plan_start = stale
    with pytest.raises(MotionPlanningError, match="stale_plan"):
        MotionPlanExecution(raw, plan)
    assert not actions and not resets


def test_cursor_external_stop_does_not_step():
    raw, plan, actions, resets = execution_fixture()
    execution = MotionPlanExecution(raw, plan)
    execution.stop("contact_stop")
    assert execution.next_action() is None
    assert execution.result()["status"] == "contact_stop"
    assert not actions and not resets


@pytest.mark.parametrize("finger_lower", [0., .005])
def test_empty_closed_gripper_accepts_attainable_target_and_retains_pressure(finger_lower):
    raw, plan, actions, resets = execution_fixture()
    raw.agent.robot.get_qlimits()[0, 7:, 0] = finger_lower
    plan.position[:, 7:] = finger_lower
    plan.initial_qpos[7:] = finger_lower
    execution = MotionPlanExecution(raw, plan, gripper_command=-1.)
    assert execution.gripper_command == -1.
    assert execution.next_action()[-1] == -1.
    assert not actions and not resets


@pytest.mark.parametrize("gripper_command", [-1., 1.])
def test_unheld_fingers_partway_to_close_or_open_target_still_reject(gripper_command):
    raw, plan, actions, resets = execution_fixture()
    plan.position[:, 7:] = .02
    plan.initial_qpos[7:] = .02
    with pytest.raises(MotionPlanningError, match="gripper_plan_mismatch"):
        MotionPlanExecution(raw, plan, gripper_command=gripper_command)
    assert not actions and not resets
