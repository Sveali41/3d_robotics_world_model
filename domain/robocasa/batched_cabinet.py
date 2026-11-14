"""Tensor observations and subset resets for the GPU cabinet collector."""
from __future__ import annotations

import torch


SPARSE_RESET_DIAGNOSTIC = (
    "SAPIEN sparse articulation or rigid-body writes are incorrect in this runtime. "
    "GPU collection with live-slot refills is disabled to prevent cross-environment state changes. "
    "Build the same-version native correction in patches/sapien-3.0.3-sparse-joint-indices.patch "
    "and follow docs/sapien_gpu_reset.md; do not bypass this preflight or mix the failed calibration data into validation."
)

from domain.robocasa.batched_skills import BatchedObservation
from exploration.cee_us.cee_us_state import OBJECT_NAMES


def quaternion_matrix(q):
    from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
    q = q.to(torch.float64)
    return quaternion_to_matrix(q / torch.linalg.vector_norm(q, dim=-1, keepdim=True))


def euler_xyz(matrix):
    """SciPy's extrinsic xyz convention, including its zero-third-angle lock."""
    y = torch.asin((-matrix[..., 2, 0]).clamp(-1, 1))
    x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    lock = torch.abs(torch.cos(y)) <= 1e-7
    x = torch.where(lock, torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1]), x)
    z = torch.where(lock, torch.zeros_like(z), z)
    return torch.stack((x, y, z), -1)


def verify_sparse_joint_writes(env):
    """Check real sparse joint/body writes before any data-collection controls.

    SAPIEN 3.0.3 accepts index buffers but incorrectly uses only their length.
    Test a nonzero articulation ID and restore all state before returning.
    This probe changes no physics clock and runs only before assigning cases.
    """
    import sapien
    px = env.scene.px
    buffers = [(getattr(px, 'cuda_articulation_'+name).torch(), name)
               for name in ('qpos', 'qvel')]
    originals = [value.clone() for value, _ in buffers]
    target = buffers[0][0].shape[0]-1
    if target < 1:
        return
    indices = torch.tensor([target], dtype=torch.int32, device=env.device)
    index_buffer = sapien.CudaArray(indices)
    valid = True
    try:
        for (value, name), original in zip(buffers, originals):
            value.copy_(original)
            expected = original[target, 0] + .125
            value[target, 0] = expected
            torch.cuda.synchronize(env.device)
            getattr(px, 'gpu_apply_articulation_'+name)(index_buffer)
            getattr(px, 'gpu_fetch_articulation_'+name)()
            torch.cuda.synchronize(env.device)
            valid &= bool(torch.equal(value[target, :1], expected.reshape(1)))
    finally:
        for (value, name), original in zip(buffers, originals):
            value.copy_(original)
            torch.cuda.synchronize(env.device)
            getattr(px, 'gpu_apply_articulation_'+name)()
        px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()
        torch.cuda.synchronize(env.device)
    if not valid:
        raise RuntimeError(SPARSE_RESET_DIAGNOSTIC)
    # The original sparse body gather packs data but retains global srcIndex.
    # Use a high nonzero body ID so an articulation-only patch is rejected too.
    bodies = px.cuda_rigid_body_data.torch()
    original = bodies.clone()
    target = int(env.cabinet_items[-1]._body_data_index[-1])
    indices = torch.tensor([target], dtype=torch.int32, device=env.device)
    try:
        expected = original[target, :3] + torch.tensor([.03125, .0625, .125], device=env.device)
        bodies[target, :3] = expected
        torch.cuda.synchronize(env.device)
        px.gpu_apply_rigid_dynamic_data(sapien.CudaArray(indices))
        px.gpu_fetch_rigid_dynamic_data()
        torch.cuda.synchronize(env.device)
        valid = bool(torch.allclose(bodies[target, :3], expected, rtol=0, atol=1e-6))
    finally:
        bodies.copy_(original)
        torch.cuda.synchronize(env.device)
        px.gpu_apply_rigid_dynamic_data()
        env.scene._gpu_fetch_all()
        torch.cuda.synchronize(env.device)
    if not valid:
        raise RuntimeError(SPARSE_RESET_DIAGNOSTIC)
    env._sparse_joint_writes_verified = True


def _indexed_reset_apply(scene, slot_mask):
    """Submit only reset articulations/bodies, requiring corrected native APIs."""
    import sapien
    articulation_ids = torch.cat([a._data_index[slot_mask[a._scene_idxs]]
                                  for a in scene.articulations.values()]).unique().int().contiguous()
    actor_parts = [a._body_data_index[slot_mask[a._scene_idxs]]
                   for a in scene.actors.values() if a.px_body_type != 'static']
    actor_ids = torch.cat(actor_parts).unique().int().contiguous() if actor_parts else None
    assert not scene._needs_fetch, 'Fetch GPU state before applying a subset reset'
    torch.cuda.synchronize(slot_mask.device)
    # Retain the tensors and their CudaArray views until all native writes finish.
    if actor_ids is not None and actor_ids.numel():
        scene.px.gpu_apply_rigid_dynamic_data(sapien.CudaArray(actor_ids))
    if articulation_ids.numel():
        indices = sapien.CudaArray(articulation_ids)
        for name in ('qpos', 'qvel', 'qf', 'root_pose', 'root_velocity',
                     'target_position', 'target_velocity'):
            getattr(scene.px, 'gpu_apply_articulation_'+name)(indices)
    torch.cuda.synchronize(slot_mask.device)
    scene._needs_fetch = True


def reset_slots(env, ids, seeds):
    """Reset only selected slots, preserving pinned-version controller buffers.

    The pinned BaseEnv resets controller buffers with an all-true mask after
    applying physical state. Preserve the unaffected rows explicitly.
    """
    ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
    if env.num_envs > 1 and not getattr(env, '_sparse_joint_writes_verified', False):
        raise RuntimeError(SPARSE_RESET_DIAGNOSTIC)
    if len(ids) != len(seeds):
        raise ValueError("Each reset slot needs its explicit case seed")
    keep = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    keep[ids] = False
    saved = []
    for ctrl in env.agent.controller.controllers.values():
        if getattr(ctrl.config, 'interpolate', False):
            raise ValueError("Subset reset requires the reference non-interpolated controllers")
        attrs = {}
        for name in ('_start_qpos', '_target_qpos', '_target_pose'):
            value = getattr(ctrl, name, None)
            tensor = value.raw_pose if hasattr(value, 'raw_pose') else value
            if isinstance(tensor, torch.Tensor) and tensor.shape[0] == env.num_envs:
                attrs[name] = tensor[keep].clone()
        saved.append((ctrl, attrs))
    # This pinned version interprets seed lists as the entire batch even for
    # subset resets. Supply slot-indexed seeds and preserve peers' RNG streams.
    slot_ids = ids.cpu().tolist()
    batch_seeds = env._episode_seed.tolist()
    for slot, seed in zip(slot_ids, seeds):
        batch_seeds[slot] = int(seed)
    previous_rngs = [(name, getattr(env, name)) for name in
                     ('_batched_main_rng', '_batched_episode_rng')]
    scene = env.scene
    original_apply = scene._gpu_apply_all
    scene._gpu_apply_all = lambda: _indexed_reset_apply(scene, ~keep)
    try:
        env.reset(seed=batch_seeds, options={'env_idx': ids})
    finally:
        scene._gpu_apply_all = original_apply
    peers = keep.nonzero().flatten().cpu().tolist()
    for name, previous in previous_rngs:
        current = getattr(env, name)
        for slot in peers:
            current[slot] = previous[slot]
    env._main_rng = env._batched_main_rng[0]
    env._episode_rng = env._batched_episode_rng[0]
    for ctrl, attrs in saved:
        for name, value in attrs.items():
            target = getattr(ctrl, name)
            target = target.raw_pose if hasattr(target, 'raw_pose') else target
            target[keep] = value


class BatchedCabinetSensor:
    """Keep all control observations on device with the historical 67D layout."""

    def __init__(self, env):
        self.env = env
        if env.control_freq != 20 or env.sim_freq != 100:
            raise ValueError("Batched split7 requires the reference 100/20 Hz timing")
        self.arm = env.agent.controller.controllers['arm']
        actors = {a.name: a for a in env.cabinet_items}
        self.objects = [actors[name] for name in OBJECT_NAMES]
        self.static = []
        self.collision_metadata = {}
        for index, actor in enumerate(self.objects):
            mesh = actor.get_first_collision_mesh(to_world_frame=False)
            if mesh is None:
                raise ValueError(f'{actor.name} has no collision mesh')
            extents = torch.tensor(mesh.extents.copy(), device=env.device, dtype=torch.float64)
            if extents.shape != (3,) or not torch.isfinite(extents).all() or (extents <= 0).any():
                raise ValueError(f'Invalid collision extents for {actor.name}')
            self.static.append(torch.cat((torch.eye(2, device=env.device)[index], extents)))
            self.collision_metadata[actor.name] = dict(source='get_first_collision_mesh(to_world_frame=False)',
                mesh_type=type(mesh).__name__, extents_m=extents.cpu().tolist())
        self.static = torch.cat(self.static).float().expand(env.num_envs, -1)
        self.reset_orientation = self.arm.ee_pose_at_base.q.clone()

    def read(self, gripper_command):
        env, arm = self.env, self.arm
        root = arm.root_link.pose
        rp, rr = root.p.double(), quaternion_matrix(root.q)
        inv = rr.transpose(-1, -2)
        def local(p):
            return (inv @ (p.double() - rp).unsqueeze(-1)).squeeze(-1)
        def vector(v):
            return (inv @ v.double().unsqueeze(-1)).squeeze(-1)
        tcp = env.agent.tcp
        objects_root = torch.stack([local(a.pose.p) for a in self.objects], 1)
        objects_world = torch.stack([a.pose.p for a in self.objects], 1)
        robot = env.agent.robot
        state = [robot.qpos.double(), robot.qvel.double()*.05, local(tcp.pose.p),
                 euler_xyz(inv @ quaternion_matrix(tcp.pose.q)), vector(tcp.linear_velocity)*.05,
                 vector(tcp.angular_velocity)*.05, env.door_joint.qpos.double().reshape(env.num_envs, 1),
                 env.door_joint.qvel.double().reshape(env.num_envs, 1)*.05,
                 gripper_command.double().reshape(-1, 1)]
        for i, actor in enumerate(self.objects):
            state += [objects_root[:, i], euler_xyz(inv @ quaternion_matrix(actor.pose.q)),
                      vector(actor.linear_velocity)*.05, vector(actor.angular_velocity)*.05]
        encoded = torch.cat(state+[self.static], -1).float()
        left = getattr(env.agent, 'finger1pad_link', None) or env.agent.finger1_link
        right = getattr(env.agent, 'finger2pad_link', None) or env.agent.finger2_link
        forces = torch.stack([torch.stack([torch.linalg.vector_norm(
            env.scene.get_pairwise_contact_forces(finger, actor), dim=-1)
            for finger in (left, right)], -1) for actor in self.objects], 1)
        grasped = torch.stack([env.agent.is_grasping(actor) for actor in self.objects], 1)
        grasped |= (forces >= .05).all(-1)
        support = torch.stack([env.scene.get_pairwise_contact_forces(actor, env.ground)[:, 2]
                               for actor in self.objects], 1)
        obs = BatchedObservation(ee_position=local(tcp.pose.p).float(),
            ee_quaternion=arm.ee_pose_at_base.q.clone(), object_positions_root=objects_root.float(),
            object_positions_world=objects_world, gripper_open=(robot.qpos[:, 7:9].sum(-1)/.08).clamp(0, 1),
            finger_forces=forces, grasped=grasped, support_forces=support)
        cabinet = env.cabinet.pose
        ci = quaternion_matrix(cabinet.q).transpose(-1, -2)
        ee_local = (ci @ (tcp.pose.p.double()-cabinet.p.double()).unsqueeze(-1)).squeeze(-1)
        object_local = torch.einsum('nij,nkj->nki', ci, objects_world.double()-cabinet.p.double()[:, None])
        lower, upper = torch.as_tensor(env.exploration_region_bounds_local, device=env.device)
        inside = ((ee_local >= lower) & (ee_local <= upper)).all(-1)
        lower, upper = torch.as_tensor(env.OBJECT_RECOVERY_BOUNDS_LOCAL, device=env.device)
        failed = ((object_local < lower) | (object_local > upper)).any(-1)
        return encoded, obs, inside, failed

    def controller_action(self, low, execution):
        from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles
        target = torch.where(execution.orientation_hold_mask[:, None],
                             execution.orientation_hold, self.reset_orientation)
        error = quaternion_matrix(target) @ quaternion_matrix(self.arm.ee_pose_at_base.q).transpose(-1, -2)
        rotation = matrix_to_euler_angles(error, 'XYZ') / float(self.arm.config.rot_lower)
        rotation /= torch.linalg.vector_norm(rotation, dim=-1, keepdim=True).clamp_min(1)
        return torch.cat((low[:, :3], rotation.float(), low[:, 3:4]), -1)
