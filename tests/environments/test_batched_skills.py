"""Synthetic scalar-reference checks without simulator or GPU requirements."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from domain.robocasa.batched_skills import (
    BatchedObservation, BatchedSkillExecution, STATUS_NAMES, STAGE_NAMES,
)
from domain.robocasa.primitives import PrimitiveAction, RAPS_SPLIT_ACTION_SCHEMA
from domain.robocasa.skill_execution import RAPSSkillExecution


def observation(n=1, device="cpu"):
    return BatchedObservation(
        torch.tensor([[.1, .2, .1]], device=device).repeat(n, 1),
        torch.tensor([[1., 0, 0, 0]], device=device).repeat(n, 1),
        torch.zeros(n, 2, 3, device=device), torch.zeros(n, 2, 3, device=device),
        torch.full((n,), .7, device=device), torch.zeros(n, 2, 2, device=device),
        torch.zeros(n, 2, dtype=torch.bool, device=device),
        torch.zeros(n, 2, device=device),
    )


def context(obs, slot=0):
    objects = {str(i): SimpleNamespace(
        actor=i, position_root=obs.object_positions_root[slot, i].numpy().copy(),
        position_world=obs.object_positions_world[slot, i].numpy().copy()) for i in range(2)}
    def forces(actor, ground):
        return np.array([0., 0., float(obs.support_forces[slot, actor])])
    return SimpleNamespace(
        ee_position=obs.ee_position[slot].numpy().copy(),
        gripper_open=float(obs.gripper_open[slot]), objects=objects,
        arm_controller=SimpleNamespace(ee_pose_at_base=SimpleNamespace(q=obs.ee_quaternion[slot:slot + 1])),
        finger_contact_force=lambda key: obs.finger_forces[slot, int(key)].numpy(),
        is_grasping=lambda key: bool(obs.grasped[slot, int(key)]),
        env=SimpleNamespace(ground=object(), scene=SimpleNamespace(get_pairwise_contact_forces=forces)),
    )


ACTIONS = [
    PrimitiveAction("move_delta_ee_pose", dict(dx=.08, dy=-.03, dz=.04)),
    PrimitiveAction("align_yz", dict(y_offset=.08, z_offset=.04)),
    PrimitiveAction("approach_x", dict(x_forward=-.12)),
    PrimitiveAction("lift", dict(z_dist=-.08)),
    PrimitiveAction("drop", dict(z_dist=-.08)),
    PrimitiveAction("open_gripper", {}),
    PrimitiveAction("close_gripper", {}),
]


@pytest.mark.parametrize("action", ACTIONS, ids=lambda a: a.primitive_type)
@pytest.mark.parametrize("settings", [{}, {"latched_gripper": True}, {
    "latched_gripper": True, "stop_drop_on_support": True, "controlled_placement": True},
    {"hold_closed_during_motion": True}])
def test_scalar_reference_actions_and_tick_order(action, settings):
    obs = observation()
    execution = BatchedSkillExecution(1, "cpu", settings)
    scalar = RAPSSkillExecution(**settings)
    scalar.start(action, context(obs))
    execution.start([0], RAPS_SPLIT_ACTION_SCHEMA.encode(action)[None], obs)
    np.testing.assert_allclose(execution.goal[0].numpy(), getattr(scalar.primitive, "goal", obs.ee_position[0]))
    assert bool(execution.orientation_hold_mask[0]) == (scalar.orientation_hold is not None)
    for _ in range(160):
        expected = scalar.next_action(context(obs))
        actual = execution.next_action(obs)
        assert bool(execution.action_mask[0]) == (expected is not None)
        assert int(execution.steps[0]) == scalar.primitive.steps
        assert STATUS_NAMES[int(execution.status[0])] == scalar.primitive.status
        if expected is None:
            break
        np.testing.assert_allclose(actual[0].numpy(), expected, atol=1e-7)
        scalar.record_action(expected)
        execution.record_action(actual)
        obs.ee_position += actual[:, :3] * .05
        if scalar.executed_steps % 2 == 0:
            scalar.observe(context(obs))
        execution.observe(obs)
    else:
        pytest.fail("Reference execution failed to finish")
    assert bool(execution.complete[0])


def test_mixed_slots_restart_latches_and_local_feedback():
    obs = observation(3)
    execution = BatchedSkillExecution(3, "cpu", {"latched_gripper": True})
    actions = np.stack([RAPS_SPLIT_ACTION_SCHEMA.encode(a) for a in (ACTIONS[6], ACTIONS[2])])
    execution.start([0, 2], actions, obs)
    obs.grasped[:] = True
    obs.finger_forces[:] = .05
    for step in range(1, 10):
        low = execution.next_action(obs)
        execution.record_action(low)
        execution.observe(obs)
        assert execution.stable_grasp_frames[0, 0] == step // 2
        assert execution.executed_steps[1] == 0
        if step >= 2:
            assert STATUS_NAMES[int(execution.status[2])] == "contact_stop"
    execution.next_action(obs)
    assert execution.complete.tolist() == [True, False, True]
    execution.start([0], RAPS_SPLIT_ACTION_SCHEMA.encode(ACTIONS[0])[None], obs)
    assert execution.next_action(obs)[0, 3] == -1
    assert execution.stable_grasp_frames[0, 0] == 0
    assert execution.gripper_command[2] == 1
    execution.reset([0])
    assert execution.gripper_command[0] == 1
    assert not execution.active[0]


@pytest.mark.parametrize("action", ACTIONS[:5], ids=lambda a: a.primitive_type)
def test_timeout_matches_scalar(action):
    obs = observation()
    scalar = RAPSSkillExecution()
    scalar.start(action, context(obs))
    execution = BatchedSkillExecution(1, "cpu")
    execution.start([0], RAPS_SPLIT_ACTION_SCHEMA.encode(action)[None], obs)
    for _ in range(scalar.primitive.max_steps + 1):
        expected = scalar.next_action(context(obs))
        execution.next_action(obs)
        assert bool(execution.action_mask[0]) == (expected is not None)
    assert STATUS_NAMES[int(execution.status[0])] == "timeout"
    assert execution.steps[0] == scalar.primitive.steps


@pytest.mark.parametrize("controlled", [False, True])
def test_support_and_four_stable_feedback_samples_match_scalar(controlled):
    obs = observation()
    obs.grasped[0, 0] = True
    settings = dict(latched_gripper=True, stop_drop_on_support=True, controlled_placement=controlled)
    scalar = RAPSSkillExecution(**settings)
    scalar.start(ACTIONS[4], context(obs))
    execution = BatchedSkillExecution(1, "cpu", settings)
    execution.start([0], RAPS_SPLIT_ACTION_SCHEMA.encode(ACTIONS[4])[None], obs)
    obs.support_forces[0, 0] = .051
    for _ in range(12):
        expected = scalar.next_action(context(obs))
        actual = execution.next_action(obs)
        assert bool(execution.action_mask[0]) == (expected is not None)
        if expected is None:
            break
        np.testing.assert_allclose(actual[0].numpy(), expected, atol=1e-7)
        scalar.record_action(expected)
        execution.record_action(actual)
        if scalar.executed_steps % 2 == 0:
            scalar.observe(context(obs))
        execution.observe(obs)
        assert STATUS_NAMES[int(execution.status[0])] == scalar.primitive.status
        assert STAGE_NAMES[int(execution.stage[0])] == scalar.primitive.stage
        assert execution.settle_count[0] == scalar.settle_count
    assert STATUS_NAMES[int(execution.status[0])] == "supported"
    assert execution.executed_steps[0] == (10 if controlled else 2)
    assert execution.support_stopped_objects.tolist() == [[True, False]]


@pytest.mark.parametrize("failure", ["no_support", "anchor", "speed"])
def test_unstable_support_deadline_and_resets(failure):
    obs = observation()
    obs.grasped[0, 0] = True
    execution = BatchedSkillExecution(1, "cpu", dict(
        latched_gripper=True, stop_drop_on_support=True, controlled_placement=True))
    execution.start([0], RAPS_SPLIT_ACTION_SCHEMA.encode(ACTIONS[4])[None], obs)
    obs.support_forces[0, 0] = .051
    for _ in range(22):
        low = execution.next_action(obs)
        execution.record_action(low)
        if execution.executed_steps[0] > 2:
            if failure == "no_support":
                obs.support_forces.zero_()
            elif failure == "anchor":
                obs.ee_position[0, 0] = .104
            else:
                obs.ee_position[0, 0] = .1 + (.002 if execution.executed_steps[0] % 4 == 0 else 0)
        execution.observe(obs)
    assert STATUS_NAMES[int(execution.status[0])] == "support_unstable"
    assert execution.executed_steps[0] == 22


def test_zero_duration_ticks_and_contact_threshold_boundaries():
    obs = observation(2)
    execution = BatchedSkillExecution(2, "cpu")
    zero = PrimitiveAction("move_delta_ee_pose", dict(dx=0., dy=0., dz=0.))
    execution.start([0, 1], np.stack([RAPS_SPLIT_ACTION_SCHEMA.encode(zero),
                                   RAPS_SPLIT_ACTION_SCHEMA.encode(ACTIONS[2])]), obs)
    execution.next_action(obs)
    assert execution.steps.tolist() == [1, 1]
    assert execution.action_mask.tolist() == [False, True]
    for _ in range(2):
        low = execution.next_action(obs)
        execution.record_action(low)
        obs.finger_forces[1, 0] = torch.tensor([.05, .049999])
        execution.observe(obs)
    assert execution.active[1]
    obs.finger_forces[1, 0] = .05
    for _ in range(2):
        execution.record_action(execution.next_action(obs))
        execution.observe(obs)
    assert STATUS_NAMES[int(execution.status[1])] == "contact_stop"


def test_invalid_settings_and_quaternion_fail_clearly():
    with pytest.raises(ValueError, match="requires"):
        BatchedSkillExecution(1, "cpu", dict(controlled_placement=True))
    obs = observation()
    obs.ee_quaternion.zero_()
    execution = BatchedSkillExecution(1, "cpu", dict(
        controlled_placement=True, latched_gripper=True, stop_drop_on_support=True))
    with pytest.raises(ValueError, match="quaternions"):
        execution.start([0], RAPS_SPLIT_ACTION_SCHEMA.encode(ACTIONS[6])[None], obs)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_orientation_hold_device_and_duplicate_feedback(device):
    obs = observation(2, device)
    obs.ee_quaternion[:] = torch.tensor([0., 0., 0., 2.], device=device)
    obs.grasped[:] = True
    execution = BatchedSkillExecution(2, device, dict(
        controlled_placement=True, latched_gripper=True, stop_drop_on_support=True))
    encoded = torch.as_tensor(np.stack([RAPS_SPLIT_ACTION_SCHEMA.encode(ACTIONS[6]),
                                       RAPS_SPLIT_ACTION_SCHEMA.encode(ACTIONS[3])]), device=device)
    execution.start(torch.arange(2, device=device), encoded, obs)
    torch.testing.assert_close(execution.orientation_hold, obs.ee_quaternion / 2)
    obs.ee_position[:, 0] += .1
    low = execution.next_action(obs)
    assert low.device.type == device
    assert torch.linalg.vector_norm(low[0, :3]) <= .250001
    for _ in range(2):
        execution.record_action(execution.next_action(obs))
        execution.observe(obs)
    previous = execution.stable_grasp_frames.clone()
    execution.observe(obs)
    torch.testing.assert_close(execution.stable_grasp_frames, previous)
    assert previous[:, 0].tolist() == [1, 1]


@pytest.mark.parametrize("action", ACTIONS, ids=lambda a: a.primitive_type)
def test_schema_parameter_scales_and_bank_indices(action):
    obs = observation()
    encoded = RAPS_SPLIT_ACTION_SCHEMA.encode(action)
    encoded[RAPS_SPLIT_ACTION_SCHEMA.num_primitives:] = [-1.4, -.7, .3, -.8, -1.2, .6]
    scalar = RAPSSkillExecution()
    scalar.start(RAPS_SPLIT_ACTION_SCHEMA.decode(encoded), context(obs))
    execution = BatchedSkillExecution(1, "cpu")
    execution.start([0], encoded[None], obs)
    np.testing.assert_allclose(execution.goal[0].numpy(),
                               getattr(scalar.primitive, "goal", obs.ee_position[0]), atol=1e-7)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_forced_terminal_odd_feedback_exactly_once(device):
    obs = observation(2, device)
    obs.grasped[:] = True
    execution = BatchedSkillExecution(2, device)
    encoded = torch.as_tensor(np.stack([RAPS_SPLIT_ACTION_SCHEMA.encode(ACTIONS[6])] * 2), device=device)
    execution.start(torch.arange(2, device=device), encoded, obs)
    force = torch.tensor([True, False], device=device)
    assert not execution.observe(obs, force_mask=force).any()
    execution.record_action(execution.next_action(obs))
    assert not execution.observe(obs).any()
    assert execution.observe(obs, force_mask=force).tolist() == [True, False]
    assert not execution.observe(obs, force_mask=force).any()
    assert execution.stable_grasp_frames[:, 0].tolist() == [1, 0]
    execution.record_action(execution.next_action(obs))
    assert execution.observe(obs, force_mask=force).tolist() == [True, True]
    assert not execution.observe(obs, force_mask=force).any()
    assert execution.stable_grasp_frames[:, 0].tolist() == [2, 1]


def test_masked_zero_duration_draining_preserves_peer_tick_and_action_mask():
    obs = observation(2)
    execution = BatchedSkillExecution(2, "cpu")
    zero = PrimitiveAction("move_delta_ee_pose", dict(dx=0., dy=0., dz=0.))
    encoded = RAPS_SPLIT_ACTION_SCHEMA.encode(zero)
    execution.start([0, 1], np.stack([encoded, RAPS_SPLIT_ACTION_SCHEMA.encode(ACTIONS[6])]), obs)
    initial_low = execution.next_action(obs)
    assert execution.steps.tolist() == [1, 1]
    assert execution.action_mask.tolist() == [False, True]
    selected = torch.tensor([True, False])
    for _ in range(3):
        execution.start([0], encoded[None], obs)
        low = execution.next_action(obs, mask=selected)
        assert execution.steps.tolist() == [1, 1]
        assert execution.executed_steps.tolist() == [0, 0]
        assert execution.action_mask.tolist() == [False, True]
        torch.testing.assert_close(low[1], initial_low[1])
    execution.record_action(low)
    assert execution.executed_steps.tolist() == [0, 1]
