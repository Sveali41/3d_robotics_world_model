from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from domain.robocasa.primitives import update_cumulative_lifts
from domain.robocasa.primitive_wrapper import RAPSEnvWrapper
from exploration.ppo_adapter import RAPSActorCritic, Rollout, _ppo_update, policy_checkpoint_path


class ControllerEnv(gym.Env):
    action_space = gym.spaces.Box(-1., 1., (7,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

    def reset(self, **kwargs):
        self.actions = []
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action.copy())
        return np.array(action[:3]), 99., False, False, {}


def set_arm_rotation(arm, euler_xyz):
    xyzw = Rotation.from_euler('XYZ', euler_xyz).as_quat()
    arm.ee_pose_at_base.q = torch.tensor([xyzw[3], *xyzw[:3]], dtype=torch.float32)


def make_wrapper(monkeypatch, dense=False, limit=10, primitive_limit=40, ee_limit=100):
    base = ControllerEnv()
    arm = SimpleNamespace(
        config=SimpleNamespace(
            rot_lower=-.1, rot_upper=.1, normalize_action=True, use_delta=True,
            use_target=False,
            frame='root_translation:root_aligned_body_rotation',
        ),
        ee_pose_at_base=SimpleNamespace(q=torch.tensor([1., 0., 0., 0.])),
    )
    base.agent = SimpleNamespace(
        controller=SimpleNamespace(controllers={'arm': arm})
    )
    base.arm = arm
    wrapper = RAPSEnvWrapper(base, robot='panda', action_mode='ee_delta',
                             record_dense_trajectory=dense,
                             max_primitives_per_episode=primitive_limit,
                             max_ee_steps_per_episode=ee_limit,
                             max_low_level_steps_per_episode=limit)
    state = SimpleNamespace(position_root=np.zeros(3), position_world=np.zeros(3))
    context = SimpleNamespace(ee_position=np.zeros(3), gripper_open=1.,
                              objects={} if dense else {'item': state},
                              finger_contact_force=lambda _: (1., 1.),
                              is_grasping=lambda _: True)
    monkeypatch.setattr(wrapper, '_context', lambda: context)
    wrapper.reset()
    return base, wrapper


@pytest.mark.parametrize('dense', [False, True])
def test_direct_command_one_tick_mapping_record_and_budget(monkeypatch, dense):
    base, wrapper = make_wrapper(monkeypatch, dense, limit=2)
    action = np.array([.2, -.3, .4, -1.], dtype=np.float32)
    _, reward, _, truncated, info = wrapper.step(action)
    assert reward == 0 and not truncated
    assert len(base.actions) == 1
    np.testing.assert_array_equal(base.actions[0], np.array([.2, -.3, .4, 0., 0., 0., -1.], dtype=np.float32))
    record = info['primitive_record']
    assert record.schema_version == 'cabinet-ee-delta-v1'
    np.testing.assert_array_equal(record.raps_action_normalized, action)
    assert record.duration_low_steps == 1
    assert not info['primitive_timeout']
    record.validate()
    with pytest.raises(ValueError, match='not RAPS'):
        record.phase_view()
    assert wrapper.step(action)[3]
    with pytest.raises(RuntimeError, match='reset'):
        wrapper.step(action)


def test_grasp_persistence_validation_and_episode_reset(monkeypatch):
    _, wrapper = make_wrapper(monkeypatch)
    for _ in range(3):
        info = wrapper.step(np.zeros(4))[4]
        assert not info['primitive_record'].outcome_metrics['held_object_ids']
    info = wrapper.step(np.zeros(4))[4]
    assert info['primitive_record'].outcome_metrics['held_object_ids'] == ['item']
    wrapper.reset()
    assert not wrapper.step(np.zeros(4))[4]['primitive_record'].outcome_metrics['held_object_ids']
    for action in (np.zeros(20), np.array([np.nan, 0, 0, 0])):
        with pytest.raises(ValueError):
            wrapper.step(action)


def test_cumulative_lift_detects_small_steps_over_continuous_grasp():
    references = {}
    lifted = []
    for frame in range(1, 5):
        before_z = (frame - 1) * .003
        after_z = frame * .003
        lifted, displacements = update_cumulative_lifts(
            references, {'item': [0., 0., before_z]},
            {'item': [0., 0., after_z]}, {'item': frame}, ['item'])
        assert ('item' in lifted) is (frame == 4)
    assert references == {'item': 0.}
    assert displacements['item'] == pytest.approx(.012)


def test_cumulative_lift_requires_four_stable_grasp_frames():
    references = {}
    for frame in range(1, 4):
        lifted, _ = update_cumulative_lifts(
            references, {'item': [0., 0., 0.]}, {'item': [0., 0., .05]},
            {'item': frame}, [])
        assert lifted == []


def test_cumulative_lift_clears_anchor_on_loss_and_reanchors_on_regrasp():
    references = {}
    for frame in range(1, 5):
        before_z = (frame - 1) * .003
        lifted, _ = update_cumulative_lifts(
            references, {'item': [0., 0., before_z]},
            {'item': [0., 0., frame * .003]}, {'item': frame}, ['item'])
    assert lifted == ['item']

    lifted, _ = update_cumulative_lifts(
        references, {'item': [0., 0., .012]}, {'item': [0., 0., .012]},
        {'item': 0}, [])
    assert lifted == []
    assert references == {}

    for frame, after_z in enumerate((.013, .015, .018, .021), start=1):
        before_z = after_z - .003
        lifted, displacements = update_cumulative_lifts(
            references, {'item': [0., 0., before_z]},
            {'item': [0., 0., after_z]}, {'item': frame}, ['item'])
        assert ('item' in lifted) is (frame == 4)
    assert references['item'] == pytest.approx(.010)
    assert displacements['item'] == pytest.approx(.011)


def test_cumulative_lift_tracks_objects_independently_and_episode_reset():
    references = {}
    lifted, displacements = update_cumulative_lifts(
        references,
        {'a': [0., 0., 0.], 'b': [0., 0., .2]},
        {'a': [0., 0., .012], 'b': [0., 0., .203]},
        {'a': 4, 'b': 4}, ['a', 'b'])
    assert lifted == ['a']
    assert displacements == pytest.approx({'a': .012, 'b': .003})
    assert references == pytest.approx({'a': 0., 'b': .2})

    # A reset starts a fresh episode with an independent reference map.
    references = {}
    lifted, displacements = update_cumulative_lifts(
        references, {'a': [0., 0., .012]}, {'a': [0., 0., .015]},
        {'a': 4}, ['a'])
    assert lifted == []
    assert displacements['a'] == pytest.approx(.003)


def test_direct_wrapper_uses_cumulative_lift_metric(monkeypatch):
    base, wrapper = make_wrapper(monkeypatch)
    context = wrapper._context()
    original_step = base.step

    def step_and_raise_object(action):
        result = original_step(action)
        context.objects['item'].position_root[2] += .003
        return result

    monkeypatch.setattr(base, 'step', step_and_raise_object)
    outcomes = [wrapper.step(np.zeros(4))[4]['primitive_record'] for _ in range(4)]
    assert [record.outcome_metrics['lifted_object_ids'] for record in outcomes] == [
        [], [], [], ['item']]
    assert outcomes[-1].metadata['lift_metric'] == 'continuous_grasp_height_v2'
    assert outcomes[-1].outcome_metrics['lift_reference_heights_root'] == {'item': 0.}
    assert outcomes[-1].outcome_metrics['object_lift_displacements']['item'] == pytest.approx(.012)

    wrapper.reset()
    assert wrapper._direct_lift_reference_heights == {}


def test_wrist_hold_targets_reset_pose_and_maps_rotation_correction(monkeypatch):
    base, wrapper = make_wrapper(monkeypatch)
    target_euler = np.array([.04, -.02, .03])
    set_arm_rotation(base.arm, target_euler)
    wrapper.reset()

    drift = np.array([.02, -.03, .01])
    set_arm_rotation(base.arm, drift)
    action = np.array([.2, -.3, .4, -1.], dtype=np.float32)
    wrapper.step(action)

    # The controller's normalized rotation action is the intrinsic XYZ
    # correction divided by Panda's -0.1 rotation lower bound.
    correction = Rotation.from_euler('XYZ', target_euler) * Rotation.from_euler('XYZ', drift).inv()
    expected_rot_action = correction.as_euler('XYZ') / -.1
    np.testing.assert_allclose(base.actions[-1],
                               [action[0], action[1], action[2], *expected_rot_action, action[3]],
                               atol=2e-5)
    np.testing.assert_allclose(
        (Rotation.from_euler('XYZ', base.actions[-1][3:6] * -.1) *
         Rotation.from_euler('XYZ', drift)).as_matrix(),
        Rotation.from_euler('XYZ', target_euler).as_matrix(), atol=2e-5,
    )

    # A new reset captures a new reference instead of retaining the old one.
    reset_target = np.array([-.03, .01, .02])
    set_arm_rotation(base.arm, reset_target)
    wrapper.reset()
    later_drift = np.array([.01, .02, -.02])
    set_arm_rotation(base.arm, later_drift)
    wrapper.step(action)
    reset_correction = Rotation.from_euler('XYZ', reset_target) * Rotation.from_euler('XYZ', later_drift).inv()
    np.testing.assert_allclose(base.actions[-1][3:6], reset_correction.as_euler('XYZ') / -.1,
                               atol=2e-5)


def test_wrist_hold_clips_rotation_norm_and_ee_episode_budget_is_independent(monkeypatch):
    base, wrapper = make_wrapper(monkeypatch, limit=10, primitive_limit=2, ee_limit=4)
    action = np.array([.2, -.3, .4, -1.], dtype=np.float32)
    set_arm_rotation(base.arm, [0., 0., 1.])
    wrapper.step(action)
    rot_action = base.actions[-1][3:6]
    assert np.linalg.norm(rot_action) == pytest.approx(1.)
    np.testing.assert_array_equal(base.actions[-1][[0, 1, 2, 6]], action)

    # max_primitives_per_episode does not limit the direct ee_delta mode.
    for _ in range(2):
        _, _, terminated, truncated, _ = wrapper.step(action)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = wrapper.step(action)
    assert not terminated and truncated


def test_ee_delta_low_level_budget_still_truncates_first(monkeypatch):
    _, wrapper = make_wrapper(monkeypatch, limit=2, primitive_limit=40, ee_limit=100)
    action = np.zeros(4, dtype=np.float32)
    assert not wrapper.step(action)[3]
    assert wrapper.step(action)[3]


def test_continuous_policy_probability_ratio_update_and_checkpoint_isolation():
    policy = RAPSActorCritic(3, 4, 8, .6)
    observations = torch.zeros(8, 3)
    with torch.no_grad():
        actions, log_probs, values = policy.act(observations)
        new_log_probs, entropy, _ = policy.evaluate(observations, actions)
    assert actions.shape == (8, 4)
    assert (actions.abs() < 1).all()
    torch.testing.assert_close(new_log_probs, log_probs)
    assert torch.isfinite(entropy).all()
    rollout = Rollout()
    rollout.append_batch(observations, actions, log_probs, values, values,
                         np.arange(8), np.zeros(8, dtype=bool), np.ones(8, dtype=bool))
    config = OmegaConf.create(dict(gamma=.8, gae_lambda=.95, num_mini_batch=2,
                                  ppo_epochs=1, clip_param=.2, value_loss_coef=.5,
                                  entropy_coef=.01, max_grad_norm=.5))
    metrics = _ppo_update(policy, torch.optim.Adam(policy.parameters()), rollout,
                          config, torch.device('cpu'))
    assert all(np.isfinite(value) for value in metrics.values())
    assert policy.deterministic_action(observations).shape == (8, 4)
    cfg = OmegaConf.create({'ppo': {'checkpoint_dir': '/tmp/check', 'checkpoint_name': 'policy.pt'}})
    assert str(policy_checkpoint_path(cfg)) == '/tmp/check/policy.pt'
    cfg.action_mode = 'ee_delta'
    assert str(policy_checkpoint_path(cfg)) == '/tmp/check/ee_delta/policy.pt'
    cfg.exploration_mode = 'D'
    assert str(policy_checkpoint_path(cfg)) == '/tmp/check/ee_delta/D/policy.pt'


def test_direct_projection_preserves_policy_action_and_clamps_target(monkeypatch):
    base, wrapper = make_wrapper(monkeypatch)
    wrapper.project_goals_to_region = True
    context = wrapper._context()
    context.arm_controller = SimpleNamespace(config=SimpleNamespace(
        normalize_action=True, use_delta=True, use_target=False,
        frame="root_translation:root_aligned_body_rotation", pos_lower=-.1, pos_upper=.1))
    context.project_goal = lambda target: np.clip(target, -.02, .02)
    action = np.array([1., -.5, .1, -1.], dtype=np.float32)
    record = wrapper.step(action)[4]['primitive_record']
    np.testing.assert_allclose(base.actions[0], [.2, -.2, .1, 0, 0, 0, -1], atol=1e-6)
    np.testing.assert_array_equal(record.raps_action_normalized, action)
    np.testing.assert_allclose(record.outcome_metrics['requested_goals']['target'], [.1, -.05, .01], atol=1e-6)
    np.testing.assert_allclose(record.outcome_metrics['executed_goals']['target'], [.02, -.02, .01], atol=1e-6)
    assert record.outcome_metrics['changed_goal_count'] == 1
    assert record.metadata['project_goals_to_region']
    np.testing.assert_allclose(record.metadata['controller_target'], [.02, -.02, .01], atol=1e-6)
    context.arm_controller.config.use_target = True
    with pytest.raises(ValueError, match='current-pose'):
        wrapper.step(action)


def test_d_reward_matches_diagnostic_r1():
    from dataclasses import asdict
    from exploration.contact_reward import ContactExplorationRewardConfig, interaction_reward_config
    from exploration.diagnose_cabinet import reward_configs
    cfg = OmegaConf.create({'reward': asdict(ContactExplorationRewardConfig())})
    reward = interaction_reward_config(ContactExplorationRewardConfig())
    assert reward == reward_configs(cfg)['R1']
    assert reward.grasp_tcp_offset == reward.grasped_object == reward.lifted_object == 0
    assert reward.moved_object_requires_grasp is False
    assert reward.moved_object == 3
