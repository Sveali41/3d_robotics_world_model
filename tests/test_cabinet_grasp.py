from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from domain.robocasa.primitives import (
    LiftPrimitive,
    PrimitiveAction,
    RAPS_ACTION_SCHEMA,
    TopGraspPrimitive,
)
from exploration.contact_reward import (
    ContactExplorationRewardConfig,
    ContactExplorationRewardWrapper,
)
from exploration.ppo_adapter import RAPSActorCritic
from exploration.robot_flow_bridge import NominalRAPSRobotFlowBridge
from exploration.scripted_grasp import SCRIPTED_GRASP_SEQUENCES


class ContactContext:
    def __init__(self):
        self.ee_position = np.array([0.35, -0.06, 0.72], dtype=np.float32)
        self.gripper_open = 1.0
        self.objects = {
            name: SimpleNamespace(position_root=np.array(position, dtype=np.float32))
            for name, position in (
                ("cabinet_item_16", [0.45, -0.06, 0.75]),
                ("cabinet_item_17", [0.45, 0.06, 0.71]),
            )
        }
        self.forces = {name: (0.0, 0.0) for name in self.objects}
        self.grasping = set()

    def finger_contact_force(self, name):
        return self.forces[name]

    def is_grasping(self, name):
        return name in self.grasping

    def contact_snapshot(self):
        return {
            "objects": {
                name: {
                    "left_force": left,
                    "right_force": right,
                    "grasped": self.is_grasping(name),
                }
                for name, (left, right) in self.forces.items()
            }
        }


def test_grasp_alignment_parameters_receive_policy_gradients():
    policy = RAPSActorCritic(4, 20, 8, 0.6)
    for index, name in enumerate(RAPS_ACTION_SCHEMA.primitive_names):
        expected = set(RAPS_ACTION_SCHEMA.primitive_name_to_action_idx[name])
        actual = set(torch.where(policy.parameter_mask[index] != 0)[0].tolist())
        assert actual == expected

    observation = torch.zeros(1, 4)
    action = torch.from_numpy(
        PrimitiveAction(
            "top_grasp", {"x_forward": 0.1, "y_offset": 0.04, "z_offset": 0.03}
        ).to_vector()
    ).unsqueeze(0)
    log_probability, _ = policy._log_prob_entropy(observation, action)
    log_probability.sum().backward()
    gradient = policy.actor[-1].bias.grad
    assert torch.all(gradient[13:16] != 0)
    assert gradient[16] == 0


def test_grasp_requires_simultaneous_contact_on_the_same_object():
    context = ContactContext()
    primitive = TopGraspPrimitive(PrimitiveAction(
        "top_grasp", {"x_forward": 0.1, "y_offset": 0.0, "z_offset": 0.0}
    ))
    primitive.reset(context)
    primitive.stage = "approach"
    context.forces["cabinet_item_16"] = (0.1, 0.0)
    primitive.observe(context)
    context.forces["cabinet_item_16"] = (0.0, 0.1)
    primitive.observe(context)
    assert primitive.stage == "approach"
    context.forces["cabinet_item_16"] = (0.1, 0.0)
    context.forces["cabinet_item_17"] = (0.0, 0.1)
    primitive.observe(context)
    assert primitive.stage == "approach"
    context.forces["cabinet_item_16"] = (0.1, 0.1)
    primitive.observe(context)
    assert primitive.stage == "close"


def test_lift_requires_stable_grasp_at_the_end():
    context = ContactContext()
    primitive = LiftPrimitive(PrimitiveAction("lift", {"z_dist": 0.1}))
    primitive.reset(context)
    context.grasping.add("cabinet_item_16")
    context.objects["cabinet_item_16"].position_root[2] += 0.02
    for _ in range(3):
        primitive.observe(context)
    assert primitive.result(context)["lifted_object_ids"] == []
    primitive.observe(context)
    assert primitive.result(context)["lifted_object_ids"] == ["cabinet_item_16"]
    context.grasping.clear()
    result = primitive.result(context)
    assert result["held_object_ids"] == []
    assert result["lifted_object_ids"] == []
    assert result["grasped_object_ids"] == ["cabinet_item_16"]


def test_grasp_does_not_report_success_after_losing_the_object():
    context = ContactContext()
    primitive = TopGraspPrimitive(PrimitiveAction(
        "top_grasp", {"x_forward": 0.1, "y_offset": 0.0, "z_offset": 0.0}
    ))
    primitive.reset(context)
    context.grasping.add("cabinet_item_16")
    for _ in range(4):
        primitive.observe(context)
    context.grasping.clear()
    primitive.stage = "close"
    primitive.close_steps = 29
    assert primitive.next_action(context) is None
    assert primitive.status == "no_stable_grasp"


def test_scripted_actions_and_nominal_grasp_match_the_current_schema():
    for sequence in SCRIPTED_GRASP_SEQUENCES.values():
        for name, parameters in sequence:
            action = PrimitiveAction(name, parameters).to_vector()
            assert RAPS_ACTION_SCHEMA.decode(action).primitive_type == name
    action = PrimitiveAction(
        "top_grasp", {"x_forward": -0.1, "y_offset": 0.04, "z_offset": -0.03}
    ).to_vector()
    waypoints = NominalRAPSRobotFlowBridge._waypoints(action)
    np.testing.assert_allclose(waypoints, [
        [0.0, 0.0, 0.0], [0.0, 0.04, -0.03], [0.1, 0.04, -0.03]
    ], atol=1e-7)


class RecordedOutcomeEnv(gym.Env):
    def __init__(self, outcomes):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (20,), dtype=np.float32)
        self.context = ContactContext()
        self.cabinet_items = [SimpleNamespace(name=name) for name in self.context.objects]
        self.outcomes = iter(outcomes)

    def _context(self):
        return self.context

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        record = SimpleNamespace(
            outcome_metrics=next(self.outcomes), contact_events=[], duration_low_steps=1
        )
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {"primitive_record": record}


def test_contact_grasps_do_not_complete_the_task():
    env = ContactExplorationRewardWrapper(RecordedOutcomeEnv([
        {"grasped_object_ids": ["cabinet_item_16", "cabinet_item_17"]}
    ]))
    env.reset()
    obs, _, terminated, _, info = env.step(None)
    report = info["exploration_reward"]
    assert report["raw"]["all_objects_grasped"]
    assert not report["exploration_complete"]
    assert not terminated
    np.testing.assert_array_equal(obs[-2:], [0, 0])


@pytest.mark.parametrize("terminate", [True, False])
def test_lift_reward_is_once_per_object_and_completion_requires_both(terminate):
    mug = {"lifted_object_ids": ["cabinet_item_16"]}
    bowl = {"lifted_object_ids": ["cabinet_item_17"]}
    env = ContactExplorationRewardWrapper(
        RecordedOutcomeEnv([mug, mug, bowl]),
        ContactExplorationRewardConfig(terminate_when_all_lifted=terminate),
    )
    env.reset()
    obs, _, terminated, _, info = env.step(None)
    assert info["exploration_reward"]["terms"]["lifted_object"] == 3.0
    assert not terminated
    np.testing.assert_array_equal(obs[-2:], [1, 0])
    _, _, _, _, info = env.step(None)
    assert info["exploration_reward"]["terms"]["lifted_object"] == 0.0
    assert info["exploration_reward"]["raw"]["grasp_alignment"] == 0.0
    obs, _, terminated, _, info = env.step(None)
    assert info["exploration_reward"]["exploration_complete"]
    assert terminated == terminate
    np.testing.assert_array_equal(obs[-2:], [1, 1])
    obs, _ = env.reset()
    assert env.ever_lifted_object_ids == set()
    np.testing.assert_array_equal(obs[-2:], [0, 0])


def test_pushing_a_previously_grasped_object_does_not_earn_held_motion_reward():
    env = ContactExplorationRewardWrapper(RecordedOutcomeEnv([
        {"grasped_object_ids": ["cabinet_item_16"]},
        {"moved_object_ids": ["cabinet_item_16"]},
        {"moved_object_ids": ["cabinet_item_16"], "held_object_ids": ["cabinet_item_16"]},
    ]))
    env.reset()
    env.step(None)
    _, _, _, _, info = env.step(None)
    assert info["exploration_reward"]["terms"]["moved_object"] == 0.0
    _, _, _, _, info = env.step(None)
    assert info["exploration_reward"]["terms"]["moved_object"] == 3.0
