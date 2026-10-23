from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from domain.robocasa.primitives import PrimitiveContext, PrimitiveAction, PRIMITIVE_CLASSES
from domain.robocasa.primitive_wrapper import RAPSEnvWrapper
from exploration.contact_reward import ContactExplorationRewardConfig, _grasp_alignment
from exploration.ppo_adapter import RAPSActorCritic, Rollout, _ppo_update


class MatrixPose:
    def __init__(self, matrix):
        self.matrix = matrix

    def to_transformation_matrix(self):
        return self.matrix


def projection_context(enabled=True):
    context = PrimitiveContext.__new__(PrimitiveContext)
    # Distinct rotations and translations expose accidental world/root clamping.
    root = np.array([[0., -1., 0., 2.], [1., 0., 0., 3.],
                     [0., 0., 1., 4.], [0., 0., 0., 1.]])
    cabinet = np.array([[-1., 0., 0., 1.], [0., -1., 0., 2.],
                        [0., 0., 1., 3.], [0., 0., 0., 1.]])
    context.arm_controller = SimpleNamespace(root_link=SimpleNamespace(pose=MatrixPose(root[None])))
    context.env = SimpleNamespace(cabinet=SimpleNamespace(pose=MatrixPose(cabinet[None])),
                                 exploration_region_bounds_local=np.array([[-.1]*3, [.1]*3]))
    context.project_goals_to_region = enabled
    context.ee_position = np.zeros(3, dtype=np.float32)
    context.gripper_open = 1.
    context.objects = {}
    return context, root, cabinet


def test_projection_uses_actual_cabinet_and_controller_root_transforms():
    context, root, cabinet = projection_context()
    goal = np.array([.3, .2, .4], dtype=np.float32)
    executed = context.project_goal(goal)
    local_requested = (np.linalg.inv(cabinet) @ root @ np.append(goal, 1))[:3]
    local_executed = (np.linalg.inv(cabinet) @ root @ np.append(executed, 1))[:3]
    np.testing.assert_allclose(local_executed, np.clip(local_requested, -.1, .1), atol=1e-6)
    context.project_goals_to_region = False
    np.testing.assert_array_equal(context.project_goal(goal), goal)


@pytest.mark.parametrize('name,parameters', [
    ('move_delta_ee_pose', dict(dx=.15, dy=.1, dz=.1)),
    ('lift', dict(z_dist=.16)), ('drop', dict(z_dist=.16)),
    ('move_left', dict(y_dist=.15)), ('move_right', dict(y_dist=.15)),
    ('move_forward', dict(x_dist=.15)), ('move_backward', dict(x_dist=.15)),
    ('top_grasp', dict(x_forward=.3, y_offset=.2, z_offset=.1)),
    ('open_gripper', {}), ('close_gripper', {}),
])
def test_all_translation_goals_are_projected_and_p0_is_unchanged(name, parameters):
    context, _, _ = projection_context(False)
    primitive = PRIMITIVE_CLASSES[name](PrimitiveAction(name, parameters))
    primitive.reset(context)
    requested = {key: value[:] for key, value in primitive.requested_goals.items()}
    assert primitive.result(context)['changed_goal_count'] == 0
    assert primitive.executed_goals == requested
    context.project_goals_to_region = True
    primitive.reset(context)
    expected_count = 2 if name == 'top_grasp' else (0 if 'gripper' in name else 1)
    assert len(primitive.executed_goals) == expected_count
    assert primitive.result(context)['changed_goal_count'] == expected_count
    for key, goal in requested.items():
        np.testing.assert_allclose(getattr(primitive, key), context.project_goal(goal))


class CounterEnv(gym.Env):
    action_space = gym.spaces.Box(-1., 1., (4,), dtype=np.float32)
    observation_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)

    def reset(self, **kwargs):
        self.steps = 0
        return np.zeros(1), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(1), 0., False, False, {}


@pytest.mark.parametrize('budget,expected', [(100, [4, 8]), (6, [4, 6])])
def test_compact_contact_cadence_and_budget_final_step(monkeypatch, budget, expected):
    env = CounterEnv()
    wrapper = RAPSEnvWrapper(env, record_dense_trajectory=False,
                             contact_check_interval=4, max_low_level_steps_per_episode=budget)
    checked = []
    state = SimpleNamespace(position_root=np.zeros(3), position_world=np.zeros(3))

    def force(_):
        checked.append(env.steps)
        return 0., 0.

    context = SimpleNamespace(ee_position=np.zeros(3), gripper_open=1.,
                              objects={'item': state}, finger_contact_force=force,
                              is_grasping=lambda _: False)
    monkeypatch.setattr(wrapper, '_context', lambda: context)
    wrapper.reset()
    _, _, _, truncated, info = wrapper.step(PrimitiveAction('close_gripper', {}).to_vector())
    assert checked == expected
    assert info['duration_low_steps'] == min(budget, 9)
    assert truncated == (budget == 6)
    assert info['primitive_record'].outcome_metrics['initial_object_positions_world'] == {'item': [0., 0., 0.]}


def test_default_alignment_offset_preserved_and_zero_offset_changes_target():
    context = SimpleNamespace(ee_position=np.array([.455, 0., 0.]),
                              objects={'item': SimpleNamespace(position_root=np.array([.5, 0., 0.]))})
    assert ContactExplorationRewardConfig().grasp_tcp_offset == .045
    assert _grasp_alignment(context) == _grasp_alignment(context, None, .045)
    assert _grasp_alignment(context) == pytest.approx(1.)
    assert _grasp_alignment(context, None, 0.) == pytest.approx(np.exp(-12*.045))


def test_ppo_metrics_are_finite_and_bad_rollout_fails_clearly():
    policy = RAPSActorCritic(4, 20, 8, .6)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    rollout = Rollout()
    observation = torch.zeros(2, 4)
    with torch.no_grad():
        action = policy.deterministic_action(observation)
        log_prob, _, value = policy.evaluate(observation, action)
    rollout.append_batch(observation, action, log_prob, value, value,
                         np.array([0., 1.]), np.array([False]*2), np.array([True]*2))
    config = OmegaConf.create(dict(gamma=.99, gae_lambda=.95, num_mini_batch=1,
                                  ppo_epochs=1, clip_param=.2, value_loss_coef=.5,
                                  entropy_coef=.01, max_grad_norm=.5))
    metrics = _ppo_update(policy, optimizer, rollout, config, torch.device('cpu'))
    assert {'approx_kl', 'clip_fraction', 'explained_variance'} <= metrics.keys()
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0 <= metrics['clip_fraction'] <= 1
    rollout.rewards[0][0] = np.nan
    with pytest.raises(ValueError, match='PPO rollout.*NaN'):
        _ppo_update(policy, optimizer, rollout, config, torch.device('cpu'))
