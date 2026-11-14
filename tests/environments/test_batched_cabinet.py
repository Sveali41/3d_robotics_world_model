"""Numerical and opt-in native GPU checks for cabinet batching."""
import os
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from domain.robocasa.batched_cabinet import euler_xyz, quaternion_matrix


def test_euler_matches_scipy_including_gimbal_lock():
    rng = np.random.default_rng(4)
    angles = np.concatenate((rng.uniform(-np.pi, np.pi, (100, 3)),
        [[.2, np.pi/2, -.6], [-.4, -np.pi/2, .8], [np.pi, 0, 0]]))
    matrices = Rotation.from_euler('xyz', angles).as_matrix()
    with pytest.warns(UserWarning, match='Gimbal lock'):
        expected = Rotation.from_matrix(matrices).as_euler('xyz')
    actual = euler_xyz(torch.as_tensor(matrices)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)


def test_quaternion_matrix_normalization():
    q = np.array([[.3, .1, -.4, .7], [0, 1., 0, 0]])
    actual = quaternion_matrix(torch.as_tensor(q)).numpy()
    expected = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.skipif(os.environ.get('RUN_GPU_CABINET') != '1', reason='opt-in native GPU simulation')
@pytest.mark.parametrize('num_envs', [2, 4])
def test_native_gpu_sensor_partial_reset_and_contacts(num_envs):
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from domain.robocasa.batched_cabinet import BatchedCabinetSensor, reset_slots, verify_sparse_joint_writes
    env, _ = make_env(sim_backend='gpu', num_envs=num_envs, raw_env=True, centered_initial_tcp=True)
    raw = env.unwrapped
    try:
        try:
            verify_sparse_joint_writes(raw)
        except RuntimeError as error:
            pytest.skip(str(error))
        sensor = BatchedCabinetSensor(raw)
        command = torch.ones(num_envs, device=raw.device)
        initial, obs, inside, failed = sensor.read(command)
        assert initial.shape == (num_envs, 67)
        assert obs.finger_forces.shape == (num_envs, 2, 2)
        assert obs.support_forces.shape == (num_envs, 2)
        torch.testing.assert_close(initial[0], initial[1], rtol=0, atol=1e-6)
        # Move/close only slot 0, establishing non-identical robot/contact state.
        action = torch.zeros((num_envs, 7), device=raw.device)
        action[:, -1] = 1
        action[0, 2], action[0, -1] = -.1, -1
        for _ in range(4):
            raw.step(action)
        before_state, *_ = sensor.read(command)
        q = raw.agent.robot.qpos.clone()
        v = raw.agent.robot.qvel.clone()
        poses = [a.pose.raw_pose.clone() for a in raw.cabinet_items]
        velocities = [a.linear_velocity.clone() for a in raw.cabinet_items]
        targets = {k: c._target_qpos.clone() for k, c in raw.agent.controller.controllers.items()}
        elapsed = raw._elapsed_steps.clone()
        peer_rng = raw._batched_episode_rng[0]
        peer_seed = raw._episode_seed[0]
        reset_slots(raw, [1], [1000081])
        assert torch.equal(q[0], raw.agent.robot.qpos[0])
        assert torch.equal(v[0], raw.agent.robot.qvel[0])
        assert raw._elapsed_steps[0] == elapsed[0] and raw._elapsed_steps[1] == 0
        for actor, pose, velocity in zip(raw.cabinet_items, poses, velocities):
            torch.testing.assert_close(actor.pose.raw_pose[0], pose[0], rtol=0, atol=1e-7)
            assert torch.equal(actor.linear_velocity[0], velocity[0])
        for k, controller in raw.agent.controller.controllers.items():
            assert torch.equal(controller._target_qpos[0], targets[k][0])
        after, _, _, _ = sensor.read(command)
        assert torch.equal(after[0], before_state[0])
        torch.testing.assert_close(after[1], initial[1], rtol=0, atol=1e-6)
        assert raw._episode_seed[1] == 1000081
        assert raw._episode_seed[0] == peer_seed
        assert raw._batched_episode_rng[0] is peer_rng
        # Exercise high indices and noncontiguous groups after moving peers.
        if num_envs == 4:
            for ids in ([3], [0, 2], [1], [2, 3]):
                for _ in range(2):
                    raw.step(action)
                before, *_ = sensor.read(command)
                keep = torch.ones(num_envs, dtype=torch.bool, device=raw.device)
                keep[ids] = False
                reset_slots(raw, ids, [1000100+i for i in ids])
                after, *_ = sensor.read(command)
                assert torch.equal(after[keep], before[keep])
                torch.testing.assert_close(after[ids], initial[ids], rtol=0, atol=1e-6)
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_GPU_CABINET') != '1', reason='opt-in native GPU simulation')
def test_native_single_gpu_state_matches_scalar_encoder():
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from domain.robocasa.batched_cabinet import BatchedCabinetSensor
    # N=1 can also use the scalar encoder on the very same GPU state.
    env, _ = make_env(sim_backend='gpu', num_envs=1, raw_env=True, centered_initial_tcp=True)
    try:
        sensor = BatchedCabinetSensor(env.unwrapped)
        actual, *_ = sensor.read(torch.ones(1, device=env.unwrapped.device))
        encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
        np.testing.assert_allclose(actual.cpu().numpy()[0], encoder.encode(), atol=1e-6, rtol=0)
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_GPU_CABINET') != '1', reason='opt-in native GPU simulation')
def test_native_sparse_preflight_diagnostic_and_probe_restoration():
    from exploration.cee_us.cee_us_cabinet import make_env
    from domain.robocasa.batched_cabinet import verify_sparse_joint_writes
    env, _ = make_env(sim_backend='gpu', num_envs=2, raw_env=True, centered_initial_tcp=True)
    raw = env.unwrapped
    try:
        px = raw.scene.px
        qpos = px.cuda_articulation_qpos.torch().clone()
        qvel = px.cuda_articulation_qvel.torch().clone()
        elapsed = raw._elapsed_steps.clone()
        try:
            verify_sparse_joint_writes(raw)
        except RuntimeError as error:
            assert 'patches/sapien-3.0.3-sparse-joint-indices.patch' in str(error)
            assert not getattr(raw, '_sparse_joint_writes_verified', False)
        else:
            assert raw._sparse_joint_writes_verified
        assert torch.equal(px.cuda_articulation_qpos.torch(), qpos)
        assert torch.equal(px.cuda_articulation_qvel.torch(), qvel)
        assert torch.equal(raw._elapsed_steps, elapsed)
    finally:
        env.close()
