"""Static Panda pose planning against the live cabinet scene.

MPLib supplies RRTConnect and timing; independent actual-shape checks reject
collisions at joint intervals of at most .01 rad. This finite sampling is not a
continuous collision certificate. Grasped items are tracked as rigid payloads
for planning only; simulation contacts remain physical.
"""
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from pathlib import Path
import importlib
from os.path import relpath
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _profile_call(owner, key, function, *args, **kwargs):
    profile = getattr(getattr(owner, "env", None), "_collection_profile", None)
    if not isinstance(profile, dict):
        return function(*args, **kwargs)
    started = perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        profile[key] = profile.get(key, 0.) + perf_counter() - started


def _profile_method(key):
    def decorate(function):
        @wraps(function)
        def measured(self, *args, **kwargs):
            return _profile_call(self, key, function, self, *args, **kwargs)
        return measured
    return decorate


class MotionPlanningError(RuntimeError):
    def __init__(self, status, detail):
        self.status = status
        self.detail = detail
        super().__init__(f"{status}: {detail}")

    def __reduce__(self):
        return type(self), (self.status, self.detail)


@dataclass
class MotionPlan:
    position: np.ndarray
    time: np.ndarray
    target_position_root: np.ndarray
    target_quaternion_wxyz: np.ndarray
    initial_qpos: np.ndarray
    initial_root_pose: np.ndarray
    scene_signature: dict
    metadata: dict


def cabinet_contact_forces(env, attachment=None):
    """Read robot and optional payload contact forces against cabinet links."""
    import torch

    forces = {}
    for link in env.agent.robot.get_links():
        for cabinet in env.cabinet.get_links():
            force = float(torch.linalg.norm(
                env.scene.get_pairwise_contact_forces(link, cabinet)).item())
            if force > 0:
                forces[f"{link.name}|cabinet/{cabinet.name}"] = force
    if attachment:
        actor = next(actor for actor in env.cabinet_items if actor.name == attachment["name"])
        for cabinet in env.cabinet.get_links():
            force = float(torch.linalg.norm(env.scene.get_pairwise_contact_forces(actor, cabinet)).item())
            if force > 0:
                forces[f"{actor.name}|cabinet/{cabinet.name}"] = force
    return forces


class MotionPlanExecution:
    """Checked joint-plan cursor; the caller owns control modes and physics steps."""

    def __init__(self, env, plan, *, gripper_command=None, max_steps=400):
        if not isinstance(plan, MotionPlan):
            raise TypeError("execute_motion_plan requires a MotionPlan from plan_to_pose")
        gripper_config = env.agent._controller_configs["pd_joint_pos"]["gripper"]
        lower, upper = float(gripper_config.lower), float(gripper_config.upper)
        attachment = plan.metadata.get("attachment")
        if gripper_command is None:
            gripper_command = attachment["gripper_command"] if attachment else float(np.clip(
                2 * (np.mean(plan.initial_qpos[7:]) - lower) / (upper - lower) - 1, -1, 1))
        if not np.isfinite(gripper_command) or not -1 <= gripper_command <= 1:
            raise ValueError("gripper_command must be finite and in [-1,1]")
        finger_target = lower + (gripper_command + 1) * (upper - lower) / 2
        if attachment and (gripper_command >= 0 or not np.isclose(gripper_command, attachment["gripper_command"], atol=1e-6)):
            raise MotionPlanningError("gripper_plan_mismatch", "Keep the planned closing command while carrying")
        if not attachment:
            finger_limits = _array(env.agent.robot.get_qlimits()).reshape(-1, 2)[7:]
            # Closing pressure can target below the physical finger joint limits.
            attainable_target = np.clip(finger_target, finger_limits[:, 0], finger_limits[:, 1])
            if np.max(np.abs(plan.position[:, 7:] - attainable_target)) > .001:
                raise MotionPlanningError("gripper_plan_mismatch",
                                          "Plan assumes fixed finger width; open/close separately, then replan")
        if isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps < 1:
            raise ValueError("max_steps must be a positive control-tick budget")
        self.planner = env.motion_planner
        self.planner.assert_plan_start(plan)
        self.env, self.plan = env, plan
        self.gripper_command = gripper_command
        self.max_steps = max_steps
        self.attachment = attachment
        self.records = []
        self.status = "running"
        self.peak_force = 0.0
        self.consecutive_force_ticks = 0
        self.terminated = self.truncated = False
        self.endpoint_position_error_m = self.endpoint_angle_error_deg = None

    def next_action(self):
        """Return the current native joint target without advancing the cursor."""
        if self.status != "running":
            return None
        target = self.plan.position[min(len(self.records), len(self.plan.position) - 1)]
        return np.asarray([*target[:7], self.gripper_command], dtype=np.float32)

    def record_step(self, *, terminated=False, truncated=False):
        """Check feedback after one caller-owned control step and record its action."""
        action = self.next_action()
        if action is None:
            raise RuntimeError("Cannot record a step after motion-plan execution has finished")
        env, plan = self.env, self.plan
        index = len(self.records)
        target = plan.position[min(index, len(plan.position) - 1)]
        qpos = env.agent.robot.get_qpos().detach().cpu().numpy().reshape(-1)
        tcp = env.agent.robot.pose.inv() * env.agent.tcp.pose
        position = tcp.p.detach().cpu().numpy().reshape(3)
        quaternion = tcp.q.detach().cpu().numpy().reshape(4)
        if not np.isfinite(np.r_[qpos, position, quaternion]).all():
            raise MotionPlanningError("execution_error", "Nonfinite robot feedback")
        attachment_feedback = attachment_error = None
        try:
            attachment_feedback = self.planner.check_attachment_feedback(plan)
        except MotionPlanningError as exc:
            self.status, attachment_error = exc.status, exc.detail
        forces = cabinet_contact_forces(env, self.attachment)
        current_peak = max(forces.values(), default=0.0)
        self.peak_force = max(self.peak_force, current_peak)
        self.consecutive_force_ticks = self.consecutive_force_ticks + 1 if current_peak > 20 else 0
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        row = dict(step=index + 1, control_mode="pd_joint_pos",
                   joint_target_rad=target[:7].tolist(),
                   controller_action=action.tolist(), gripper_command=float(self.gripper_command),
                   qpos=qpos.tolist(), tcp_position_root=position.tolist(),
                   tcp_quaternion_wxyz=quaternion.tolist(),
                   attachment=attachment_feedback, attachment_error=attachment_error,
                   stage="trajectory" if index < len(plan.position) else "settle",
                   cabinet_forces_N=forces, terminated=self.terminated, truncated=self.truncated)
        self.records.append(row)
        self.endpoint_position_error_m = float(np.linalg.norm(
            np.asarray(row["tcp_position_root"]) - plan.target_position_root))
        self.endpoint_angle_error_deg = _angle(quaternion, plan.target_quaternion_wxyz)
        if self.terminated or self.truncated:
            self.status = "terminated" if self.terminated else "truncated"
        elif self.consecutive_force_ticks >= 3:
            self.status = "tracking_blocked"
        elif self.status == "running":
            if len(self.records) == len(plan.position) + 15:
                self.status = "tracking_error" if (self.endpoint_position_error_m > .002
                                                   or self.endpoint_angle_error_deg > .5) else "success"
            elif len(self.records) >= self.max_steps:
                self.status = "budget_exhausted"
        return row

    def stop(self, status):
        """Stop for a caller-owned reason without advancing physics."""
        self.status = status

    def result(self, *, observation=None, info=None, controller_restored=None):
        return dict(schema="cabinet-joint-plan-execution-v1", status=self.status,
                    control_steps=len(self.records), ticks=self.records, observation=observation,
                    info=info, terminated=self.terminated, truncated=self.truncated,
                    endpoint_position_error_m=self.endpoint_position_error_m,
                    endpoint_angle_error_deg=self.endpoint_angle_error_deg,
                    peak_cabinet_force_N=self.peak_force,
                    controller_restored=controller_restored, planning=self.plan.metadata)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose_vector(pose):
    return np.r_[_array(pose.p).reshape(3), _array(pose.q).reshape(4)]


def _angle(q, r):
    q, r = np.asarray(q, dtype=float), np.asarray(r, dtype=float)
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(
        q / np.linalg.norm(q), r / np.linalg.norm(r))), 0., 1.))))


def _finite_list(values):
    """Convert numeric diagnostics to JSON-safe values without NaN or infinity."""
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values).reshape(-1)]


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _validate_goal(position, quaternion):
    try:
        position = np.asarray(position, dtype=float)
        quaternion = np.asarray(quaternion, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MotionPlanningError("invalid_goal", "Pose must contain numeric arrays") from exc
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion))):
        raise MotionPlanningError("invalid_goal", "Expected finite position (3,) and wxyz quaternion (4,)")
    scale = np.max(np.abs(quaternion))
    if scale == 0:
        raise MotionPlanningError("invalid_goal", "Quaternion must be nonzero")
    quaternion = quaternion / scale
    return position.copy(), quaternion / np.linalg.norm(quaternion)


def _lower_support_contact_allowed(contacts, world_to_root, surface_z):
    """Allow only shallow downward payload contacts on an explicit shelf top."""
    if not contacts:
        return False
    for contact in contacts:
        normal = world_to_root[:3, :3] @ np.asarray(contact.normal)
        position = world_to_root @ np.r_[np.asarray(contact.pos), 1.]
        depth = float(contact.penetration_depth)
        if (not np.isfinite(np.r_[normal, position, depth]).all() or normal[2] > -.99
                or abs(position[2] - surface_z) > .002 or not 0 <= depth <= .001 + 1e-9):
            return False
    return True


def _native_modules():
    # MPLib 0.1.1 can abort the interpreter under NumPy 2; check before import.
    if int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError("MPLib 0.1.1 requires NumPy < 2; run planning in an isolated NumPy 1.26 environment")
    try:
        mplib = importlib.import_module("mplib")
        fcl = importlib.import_module("mplib.pymp.collision_detection.fcl")
        sapien = importlib.import_module("sapien")
    except ImportError as exc:
        raise RuntimeError("Pose planning requires MPLib 0.1.1 and SAPIEN in the robotics environment") from exc
    if mplib.__version__ != "0.1.1":
        raise RuntimeError(f"Pose planning requires MPLib 0.1.1; found {mplib.__version__}")
    return mplib, fcl, sapien


def _groups_allow(a, b):
    return bool(((a[0] & b[1]) or (b[0] & a[1]))
                and not ((a[2] & b[2]) and a[3] == b[3]))


def _interpolate(positions, max_step=.01):
    positions = np.asarray(positions, dtype=float)
    if len(positions) == 0:
        return
    yield positions[0]
    for start, end in zip(positions[:-1], positions[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / max_step)))
        for k in range(1, count + 1):
            yield start + (end - start) * k / count


def _resample_to_control_dt(positions, source_time, dt):
    """Schedule TOPP output on real control ticks, holding the final pose."""
    times = np.arange(int(np.ceil(source_time[-1] / dt)) + 1) * dt
    scheduled = np.column_stack([
        np.interp(times, source_time, positions[:, joint])
        for joint in range(positions.shape[1])
    ])
    return scheduled, times


def _rotation(pose):
    return Rotation.from_quat(np.roll(pose.q, -1))


def _plane_min_distance(shape, pose, plane_pose, kind):
    """Exact support along a plane normal, avoiding native Convex–Plane."""
    normal = _rotation(plane_pose).apply([0., 0., 1.])
    local_normal = _rotation(pose).inv().apply(normal)
    if kind == "ConvexMesh":
        support = np.min((np.asarray(shape.vertices) * shape.scale) @ local_normal)
    elif kind == "Box":
        support = -np.dot(np.abs(local_normal), shape.half_size)
    elif kind == "Sphere":
        support = -float(shape.radius)
    elif kind == "Cylinder":
        # pose already converts the PhysX local X axis to the FCL Z axis.
        support = -(shape.half_length * abs(local_normal[2])
                    + shape.radius * np.linalg.norm(local_normal[:2]))
    else:
        raise MotionPlanningError("unsupported", f"Plane support for {kind}")
    return float(np.dot(np.asarray(pose.p) - plane_pose.p, normal) + support)


class CabinetMotionPlanner:
    """CPU, single-environment planner; never steps or sets physical state."""

    def __init__(self, raw_env):
        self.env = raw_env
        self.robot = raw_env.agent.robot
        controllers = getattr(raw_env.agent.controller, "controllers", {})
        arm = controllers.get("arm")
        if arm is None or not hasattr(arm, "kinematics"):
            raise MotionPlanningError("unsupported", "Initialize planner while pd_ee controller kinematics are available")
        if getattr(raw_env, "num_envs", 1) != 1 or _array(self.robot.get_qpos()).size != 9:
            raise MotionPlanningError("unsupported", "Planning supports a single CPU Panda with nine joints")
        if bool(getattr(raw_env, "gpu_sim_enabled", False)):
            raise MotionPlanningError("unsupported", "Planning supports CPU simulation only")
        self.kin = arm.kinematics
        self.root_link = arm.root_link
        if self.root_link.name != "panda_link0":
            raise MotionPlanningError("unsupported", "Expected controller root panda_link0")
        self.links = [link for link in self.robot.get_links()
                      if link.name in self.kin._kinematic_chain_link_names]
        self.link_indices = {link.name: i for i, link in enumerate(self.links)}
        if self.link_indices.get("panda_hand_tcp") != self.kin.end_link_idx:
            raise MotionPlanningError("unsupported", "Expected TCP panda_hand_tcp in the controller FK model")
        self.adjacent_pairs = {frozenset((link.name, link.joint.parent_link.name))
                               for link in self.links if link.joint.parent_link is not None}
        self.pindices = _array(self.kin.pmodel_active_joint_indices).astype(int)
        self.controlled = _array(self.kin.pmodel_controlled_joint_indices).astype(int)
        self.mplib = self.fcl = self.sapien = None
        self._geometry_cache = {}
        self.reset_grasp()

    def reset_grasp(self):
        """Clear payload tracking when the environment starts a new episode."""
        self.grasp_frames = {actor.name: 0 for actor in self.env.cabinet_items}
        self.attached_name = None
        self.gripper_command = 1.

    def _pose(self, pose):
        return self.sapien.Pose(_array(pose.p).reshape(3), _array(pose.q).reshape(4))

    def snapshot_scene(self):
        """JSON-compatible current scene poses for execution start checks."""
        return dict(root_pose=_pose_vector(self.root_link.pose).tolist(),
                    cabinet={link.name: _pose_vector(link.pose).tolist()
                             for link in self.env.cabinet.get_links()},
                    objects={actor.name: _pose_vector(actor.pose).tolist()
                             for actor in self.env.cabinet_items},
                    wall=_pose_vector(self.env.wall.pose).tolist(),
                    ground=_pose_vector(self.env.ground.pose).tolist())

    def assert_plan_start(self, plan):
        """Require unchanged scene poses and initial joint positions (1e-4 rad)."""
        self.check_attachment_feedback(plan)
        qpos = _array(self.robot.get_qpos()).reshape(-1)
        if (qpos.shape != (9,) or not np.all(np.isfinite(qpos))
                or np.max(np.abs(qpos - plan.initial_qpos)) > 1e-4
                or self.snapshot_scene() != plan.scene_signature):
            raise MotionPlanningError("stale_plan", "Scene poses changed or robot differs from plan start by more than 1e-4 rad; replan")

    def _grasp_contacts(self):
        agent = self.env.agent
        result = {}
        binding = getattr(self.env, "_grasp_binding", None)
        for actor in self.env.cabinet_items:
            if binding is not None and binding.bound_object_id == actor.name:
                result[actor.name] = True
                continue
            if not getattr(self.env, "_contact_report_current", True):
                result[actor.name] = False
                continue
            grasped = bool(_array(agent.is_grasping(actor)).reshape(-1)[0])
            if not grasped:
                left = getattr(agent, "finger1pad_link", None) or agent.finger1_link
                right = getattr(agent, "finger2pad_link", None) or agent.finger2_link
                forces = [float(np.linalg.norm(_array(self.env.scene.get_pairwise_contact_forces(
                    link, actor)).reshape(-1, 3)[0])) for link in (left, right)]
                grasped = min(forces) >= .05
            result[actor.name] = grasped
        return result

    def observe_grasp(self):
        """Called once after a real CPU control tick, including old EE actions."""
        gripper = self.env.agent.controller.controllers["gripper"]
        lower, upper = float(gripper.config.lower), float(gripper.config.upper)
        target = float(_array(gripper._target_qpos).reshape(-1)[0])
        self.gripper_command = float(np.clip(2 * (target-lower)/(upper-lower)-1, -1, 1))
        contacts = self._grasp_contacts()
        for name, grasped in contacts.items():
            self.grasp_frames[name] = self.grasp_frames.get(name, 0)+1 if grasped and self.gripper_command < 0 else 0
        # Keep an opening gripper attached until physical contact is lost.
        if self.attached_name is not None and not contacts[self.attached_name]:
            self.attached_name = None
        stable = [name for name, ticks in self.grasp_frames.items() if ticks >= 4]
        if len(stable) == 1:
            self.attached_name = stable[0]

    def attachment_state(self):
        """Return the measured TCP-relative pose, never a physics constraint."""
        stable = [name for name, ticks in self.grasp_frames.items() if ticks >= 4]
        if len(stable) > 1:
            raise MotionPlanningError("multiple_grasped_objects", "Only one payload per Panda gripper is supported")
        if self.attached_name is None:
            return None
        actor = next(actor for actor in self.env.cabinet_items if actor.name == self.attached_name)
        relative = self.env.agent.tcp.pose.inv() * actor.pose
        return dict(name=actor.name, tcp_to_object_pose=_pose_vector(relative).tolist(),
                    gripper_command=float(self.gripper_command))

    def check_attachment_feedback(self, plan):
        expected = plan.metadata.get("attachment")
        actual = self.attachment_state()
        contacts = self._grasp_contacts()
        if expected is None:
            permitted_open_contact = (plan.metadata.get("allow_open_grasp_contact", False)
                                      and self.gripper_command >= 0)
            if actual is not None or (any(contacts.values()) and not permitted_open_contact):
                raise MotionPlanningError("unexpected_attachment", "Grasp/contact changed; confirm the grasp and replan")
            return None
        if actual is None or actual["name"] != expected["name"] or not contacts[expected["name"]]:
            raise MotionPlanningError("grasp_lost", "The planned payload is no longer held")
        if self.gripper_command >= 0:
            raise MotionPlanningError("release_in_progress", "Finish opening and releasing before planning another motion")
        actual_pose, expected_pose = np.asarray(actual["tcp_to_object_pose"]), np.asarray(expected["tcp_to_object_pose"])
        position_error = float(np.linalg.norm(actual_pose[:3]-expected_pose[:3]))
        angle_error = _angle(actual_pose[3:], expected_pose[3:])
        if position_error > .002 or angle_error > 2.:
            raise MotionPlanningError("attachment_slip", dict(position_error_m=position_error, angle_error_deg=angle_error))
        return dict(**actual, position_error_m=position_error, angle_error_deg=angle_error)

    def _planning_attachment(self):
        attachment = self.attachment_state()
        contacts = self._grasp_contacts()
        if attachment is not None and self.gripper_command >= 0:
            raise MotionPlanningError("release_in_progress", "Finish releasing before planning")
        if any(grasped and (attachment is None or name != attachment["name"]) for name, grasped in contacts.items()):
            raise MotionPlanningError("unconfirmed_grasp", "Initialize env.motion_planner before closing; require four consecutive closed-grasp control ticks")
        if attachment is not None and not contacts[attachment["name"]]:
            raise MotionPlanningError("grasp_lost", "Payload contact was lost; take a feedback tick before replanning")
        return attachment

    @staticmethod
    def _kind(shape):
        return type(shape).__name__.removeprefix("PhysxCollisionShape")

    @_profile_method("planner_geometry_s")
    def _geometry(self, shape, margin=0.):
        if not getattr(getattr(self, "env", None), "collection_optimizations", False):
            return self._build_geometry(shape, margin)
        kind = self._kind(shape)
        def exact_array(value):
            array = np.asarray(value)
            return array.dtype.str, array.shape, array.tobytes()
        if kind == "Box":
            signature = (kind, exact_array(shape.half_size))
        elif kind == "Sphere":
            signature = (kind, exact_array(shape.radius))
        elif kind == "Cylinder":
            signature = (kind, exact_array(shape.radius), exact_array(shape.half_length))
        elif kind == "ConvexMesh":
            signature = (kind, exact_array(shape.vertices), exact_array(shape.scale),
                         exact_array(shape.triangles))
        else:
            signature = (kind,)
        key = (id(shape), margin)
        cache = self._geometry_cache
        entry = cache.get(key)
        hit = entry is not None and entry[0] is shape and entry[1] == signature
        profile = getattr(self.env, "_collection_profile", None)
        if isinstance(profile, dict):
            counter = "geometry_cache_hits" if hit else "geometry_cache_misses"
            profile[counter] = profile.get(counter, 0) + 1
        if hit:
            return entry[2]
        geometry = self._build_geometry(shape, margin)
        # Retain the shape itself so its identity cannot be reused while cached.
        cache[key] = (shape, signature, geometry)
        return geometry

    def _build_geometry(self, shape, margin=0.):
        kind = self._kind(shape)
        if kind == "Box":
            return self.fcl.Box(*(2 * np.asarray(shape.half_size) + 2 * margin))
        if kind == "Sphere":
            return self.fcl.Sphere(float(shape.radius + margin))
        if kind == "Cylinder":
            return self.fcl.Cylinder(float(shape.radius + margin), float(2 * shape.half_length + 2 * margin))
        if kind == "ConvexMesh":
            return self.fcl.Convex(np.asarray(shape.vertices * shape.scale, dtype=np.float64),
                                   np.asarray(shape.triangles, dtype=np.int32), True)
        if kind == "Plane":
            return None
        raise MotionPlanningError("unsupported", f"Unsupported actual collision shape: {type(shape).__name__}")

    def _shape_pose(self, shape, pose):
        if self._kind(shape) in ("Cylinder", "Plane"):
            return pose * self.sapien.Pose(q=[2**-.5, 0, 2**-.5, 0])
        return pose

    def _object(self, geometry, pose):
        return self.fcl.CollisionObject(geometry, np.asarray(pose.p, dtype=float), np.asarray(pose.q, dtype=float))

    def _scene_geometry(self):
        self.robot_shapes = []
        for link in self.robot.get_links():
            for shape in link._objs[0].get_collision_shapes():
                if link.name not in self.link_indices:
                    raise MotionPlanningError("unsupported", f"Collision link absent from FK model: {link.name}")
                self.robot_shapes.append(dict(link=link.name, shape=shape, geom=self._geometry(shape),
                                              local=shape.local_pose, groups=shape.get_collision_groups()))
        self.obstacles = []
        actors = [("cabinet/" + link.name, link, link._objs[0]) for link in self.env.cabinet.get_links()]
        actors += [(actor.name, actor, actor._bodies[0]) for actor in
                   [self.env.wall, self.env.ground, *self.env.cabinet_items]]
        for name, actor, body in actors:
            if self.attachment is not None and name == self.attachment["name"]:
                continue
            for i, shape in enumerate(body.get_collision_shapes()):
                self.obstacles.append(dict(name=f"{name}/{i}", actor=name, shape=shape,
                    geom=self._geometry(shape), groups=shape.get_collision_groups(),
                    pose=self._shape_pose(shape, self._pose(actor.pose) * shape.local_pose)))
        self.payload_shapes, self.support_contacts = [], {}
        self.lower_support_obstacles = set()
        surface_z = getattr(self, "support_surface_root_z", None)
        if surface_z is not None:
            for obstacle in self.obstacles:
                if obstacle["actor"] != "cabinet/frame" or self._kind(obstacle["shape"]) != "Box":
                    continue
                relative = self.root.inv() * obstacle["pose"]
                rotation = _rotation(relative).as_matrix()
                top_z = relative.p[2] + float(obstacle["shape"].half_size[2])
                if np.allclose(np.abs(rotation[:, 2]), [0., 0., 1.], atol=1e-6) and abs(top_z - surface_z) < 1e-4:
                    self.lower_support_obstacles.add(obstacle["name"])
                    self.support_contacts[obstacle["name"]] = .001
            if len(self.lower_support_obstacles) != 1:
                raise MotionPlanningError("invalid_support_surface", "Expected exactly one horizontal frame box at the requested shelf top")
        if self.attachment is not None:
            actor = next(actor for actor in self.env.cabinet_items if actor.name == self.attachment["name"])
            relative = self.attachment["tcp_to_object_pose"]
            self.tcp_to_payload = self.sapien.Pose(relative[:3], relative[3:])
            for i, shape in enumerate(actor._bodies[0].get_collision_shapes()):
                self.payload_shapes.append(dict(name=f"{actor.name}/{i}", shape=shape,
                    geom=self._geometry(shape), groups=shape.get_collision_groups(), local=shape.local_pose))
                obj = self._object(self.payload_shapes[-1]["geom"],
                                   self._shape_pose(shape, self._pose(actor.pose)*shape.local_pose))
                # A resting object may touch its support. Omit only a shallow
                # horizontal support from RRT; the exact audit retains it.
                for obstacle in self.obstacles:
                    if self._kind(obstacle["shape"]) != "Box" or not obstacle["actor"].startswith("cabinet/"):
                        continue
                    result = self.fcl.collide(obj, self._object(obstacle["geom"], obstacle["pose"]),
                                              self.fcl.CollisionRequest(8, True))
                    contacts = result.get_contacts()
                    if (contacts and max(c.penetration_depth for c in contacts) <= .001
                            and all(c.normal[2] <= -.99 for c in contacts)):
                        self.support_contacts[obstacle["name"]] = max(
                            self.support_contacts.get(obstacle["name"], 1e-6),
                            max(c.penetration_depth for c in contacts)+1e-6)
            if not self.payload_shapes:
                raise MotionPlanningError("unsupported", "Payload has no collision shapes")

    def placement_bounds(self, attachment, tcp_quaternion):
        """Payload collision AABB relative to a zero-position TCP in root axes."""
        actor = next(actor for actor in self.env.cabinet_items if actor.name == attachment["name"])
        relative = attachment["tcp_to_object_pose"]
        payload = self.sapien.Pose(q=tcp_quaternion) * self.sapien.Pose(relative[:3], relative[3:])
        lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
        for shape in actor._bodies[0].get_collision_shapes():
            pose = self._shape_pose(shape, payload * shape.local_pose)
            for axis, vector in enumerate(np.eye(3)):
                # Plane normals are +/- root axes; reuse exact shape support.
                rotation = Rotation.align_vectors([vector], [[0., 0., 1.]])[0]
                plane = self.sapien.Pose(q=np.roll(rotation.as_quat(), 1))
                lower[axis] = min(lower[axis], _plane_min_distance(shape, pose, plane, self._kind(shape)))
                rotation = Rotation.align_vectors([-vector], [[0., 0., 1.]])[0]
                plane = self.sapien.Pose(q=np.roll(rotation.as_quat(), 1))
                upper[axis] = max(upper[axis], -_plane_min_distance(shape, pose, plane, self._kind(shape)))
        bounds = np.array([lower, upper])
        if not np.isfinite(bounds).all() or np.any(lower > upper):
            raise MotionPlanningError("invalid_goal", "Placement requires finite payload collision bounds")
        return bounds

    def support_descent_step(self, surface_z, *, max_step=.005):
        """Shorten only the last support-seeking step using live collision geometry."""
        if not np.isfinite(surface_z) or not np.isfinite(max_step) or max_step <= 0:
            raise MotionPlanningError("invalid_support_surface", "Finite support height and positive step required")
        self.attachment = self._planning_attachment()
        if self.attachment is None:
            raise MotionPlanningError("missing_grasp", "Support descent requires a confirmed physical payload")
        self.support_surface_root_z = float(surface_z)
        self._scene_geometry()
        actor = next(actor for actor in self.env.cabinet_items if actor.name == self.attachment["name"])
        gap = np.inf
        for item in self.payload_shapes:
            obj = self._object(item["geom"], self._shape_pose(item["shape"],
                               self._pose(actor.pose) * item["local"]))
            for obstacle in self.obstacles:
                if obstacle["name"] in self.lower_support_obstacles:
                    distance = self.fcl.distance(obj, self._object(obstacle["geom"], obstacle["pose"]),
                                                self.fcl.DistanceRequest(enable_signed_distance=True))
                    gap = min(gap, float(distance.min_distance))
        if not np.isfinite(gap):
            raise MotionPlanningError("invalid_support_surface", "No finite payload-to-shelf distance")
        return min(float(max_step), max(.0005, gap + .0005))

    def _make_planner(self):
        from mani_skill.agents.robots.panda.panda import Panda
        from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
        urdf = Path(str(PandaWristCam.urdf_path))
        kwargs = dict(move_group="panda_hand_tcp",
            user_joint_names=Panda.arm_joint_names+["panda_finger_joint1", "panda_finger_joint2"],
            joint_vel_limits=np.full(7, .5), joint_acc_limits=np.ones(7))
        if self.attachment is None:
            return self.mplib.Planner(urdf=str(urdf), srdf=str(urdf.with_suffix(".srdf")), **kwargs)
        model = ET.parse(urdf)
        for element in model.iter("mesh"):
            filename = element.attrib["filename"]
            if not Path(filename).is_absolute():
                element.set("filename", str((urdf.parent/filename).resolve()))
        # MPLib 0.1.1's tool API missed positive-control collisions. A fixed
        # collision link instead uses the normal, verified robot-world checker.
        link = ET.SubElement(model.getroot(), "link", name="cabinet_payload")
        joint = ET.SubElement(model.getroot(), "joint", name="cabinet_payload_fixed", type="fixed")
        ET.SubElement(joint, "parent", link="panda_hand_tcp")
        ET.SubElement(joint, "child", link="cabinet_payload")
        ET.SubElement(joint, "origin", xyz=" ".join(map(str, self.tcp_to_payload.p)),
                      rpy=" ".join(map(str, _rotation(self.tcp_to_payload).as_euler("xyz"))))
        srdf = ET.parse(urdf.with_suffix(".srdf"))
        for finger in ("panda_leftfinger", "panda_rightfinger"):
            ET.SubElement(srdf.getroot(), "disable_collisions", link1="cabinet_payload", link2=finger,
                          reason="AllowedGraspContact")
        with TemporaryDirectory(prefix="cabinet_payload_") as directory:
            model_path, srdf_path = Path(directory)/"panda.urdf", Path(directory)/"panda.srdf"
            # This MPLib loader prefixes the URDF directory even for absolute
            # mesh paths. Relative paths preserve the original asset lookup.
            for element in model.iter("mesh"):
                element.set("filename", relpath(element.attrib["filename"], directory))
            for i, item in enumerate(self.payload_shapes):
                shape, kind = item["shape"], self._kind(item["shape"])
                local = self._shape_pose(shape, item["local"])
                collision = ET.SubElement(link, "collision")
                ET.SubElement(collision, "origin", xyz=" ".join(map(str, local.p)),
                    rpy=" ".join(map(str, _rotation(local).as_euler("xyz"))))
                geometry = ET.SubElement(collision, "geometry")
                if kind == "Box":
                    ET.SubElement(geometry, "box", size=" ".join(map(str, 2*np.asarray(shape.half_size))))
                elif kind == "Cylinder":
                    ET.SubElement(geometry, "cylinder", radius=str(shape.radius), length=str(2*shape.half_length))
                elif kind == "Sphere":
                    ET.SubElement(geometry, "sphere", radius=str(shape.radius))
                elif kind == "ConvexMesh":
                    import trimesh
                    mesh = trimesh.Trimesh(vertices=np.asarray(shape.vertices)*shape.scale,
                                           faces=np.asarray(shape.triangles), process=False)
                    filename = f"payload_{i}.stl"
                    mesh.export(Path(directory)/filename)
                    # MPLib's convex loader looks for a precomputed suffix.
                    mesh.export(Path(directory)/(filename+".convex.stl"))
                    ET.SubElement(geometry, "mesh", filename=filename)
                else:
                    raise MotionPlanningError("unsupported", f"Payload URDF shape {kind}")
            model.write(model_path)
            srdf.write(srdf_path)
            return self.mplib.Planner(urdf=str(model_path), srdf=str(srdf_path), **kwargs)

    def _check(self, q):
        self.kin.pmodel.compute_forward_kinematics(q)
        robot, contacts = [], []
        for item in self.robot_shapes:
            pose = self._shape_pose(item["shape"], self.root * self.kin.pmodel.get_link_pose(
                self.link_indices[item["link"]]) * item["local"])
            robot.append((item, self._object(item["geom"], pose), pose))
        request = self.fcl.CollisionRequest(8, True)
        for item, obj, pose in robot:
            for obstacle in self.obstacles:
                if not _groups_allow(item["groups"], obstacle["groups"]):
                    continue
                if self._kind(obstacle["shape"]) == "Plane":
                    collision = _plane_min_distance(item["shape"], pose, obstacle["pose"],
                                                    self._kind(item["shape"])) < -1e-6
                else:
                    collision = self.fcl.collide(obj, self._object(obstacle["geom"], obstacle["pose"]),
                                                 request).is_collision()
                permitted = (item["link"] == "panda_link0" and obstacle["actor"] == self.env.ground.name) or (
                    item["link"] in ("panda_leftfinger", "panda_rightfinger")
                    and obstacle["actor"] in self.allowed_contacts)
                if collision and not permitted:
                    contacts.append(dict(first=item["link"], second=obstacle["name"], kind="environment"))
        for i, (a, obj_a, _) in enumerate(robot):
            for b, obj_b, _ in robot[i+1:]:
                if (a["link"] == b["link"] or frozenset((a["link"], b["link"])) in self.adjacent_pairs
                        or not _groups_allow(a["groups"], b["groups"])):
                    continue
                if self.fcl.collide(obj_a, obj_b, request).is_collision():
                    contacts.append(dict(first=a["link"], second=b["link"], kind="self"))
        if getattr(self, "payload_shapes", ()):
            payload_pose = self.root*self.kin.pmodel.get_link_pose(self.kin.end_link_idx)*self.tcp_to_payload
            for item in self.payload_shapes:
                pose = self._shape_pose(item["shape"], payload_pose*item["local"])
                obj = self._object(item["geom"], pose)
                for obstacle in self.obstacles:
                    if not _groups_allow(item["groups"], obstacle["groups"]):
                        continue
                    if self._kind(obstacle["shape"]) == "Plane":
                        collision = _plane_min_distance(item["shape"], pose, obstacle["pose"], self._kind(item["shape"])) < -1e-6
                    else:
                        result = self.fcl.collide(obj, self._object(obstacle["geom"], obstacle["pose"]), request)
                        collision = result.is_collision()
                        if collision and obstacle["name"] in self.support_contacts:
                            cs = result.get_contacts()
                            if obstacle["name"] in self.lower_support_obstacles:
                                matrix = np.eye(4)
                                matrix[:3, :3] = _rotation(self.root).as_matrix()
                                matrix[:3, 3] = self.root.p
                                collision = not _lower_support_contact_allowed(cs, np.linalg.inv(matrix),
                                                                              self.support_surface_root_z)
                            else:
                                collision = not cs or max(c.penetration_depth for c in cs) > self.support_contacts[obstacle["name"]]
                    if collision:
                        contacts.append(dict(first=item["name"], second=obstacle["name"], kind="payload_environment"))
                for link, robot_obj, _ in robot:
                    if (link["link"] in ("panda_leftfinger", "panda_rightfinger")
                            or not _groups_allow(item["groups"], link["groups"])):
                        continue
                    if self.fcl.collide(obj, robot_obj, request).is_collision():
                        contacts.append(dict(first=item["name"], second=link["link"], kind="payload_self"))
        return contacts

    def _fixed_orientation_path(self, positions, quaternion, limits):
        """Project a candidate's TCP positions with the upstream IK, then audit it."""
        projected = [np.asarray(positions[0]).copy()]
        points = list(_interpolate(positions, max_step=.025))
        for index, point in enumerate(points[1:], 1):
            if index == len(points) - 1:
                projected.append(point.copy())
                break
            self.kin.pmodel.compute_forward_kinematics(point[self.pindices])
            position = np.asarray(self.kin.pmodel.get_link_pose(self.kin.end_link_idx).p).copy()
            seed = point[self.pindices].copy()
            q, success, _ = _profile_call(self, "planner_ik_s", self.kin.pmodel.compute_inverse_kinematics,
                self.kin.end_link_idx, self.sapien.Pose(position, quaternion), initial_qpos=seed,
                active_qmask=_array(self.kin.qmask), max_iterations=1000)
            q = np.asarray(q).copy()
            if not success or not np.isfinite(q).all():
                raise MotionPlanningError("orientation_projection_failed", "Fixed-orientation IK failed along candidate path")
            for joint in self.controlled:
                if self.pindices[joint] >= 7:
                    continue
                lo = int(np.ceil((limits[joint, 0] - q[joint]) / (2*np.pi)))
                hi = int(np.floor((limits[joint, 1] - q[joint]) / (2*np.pi)))
                if lo <= hi:
                    q[joint] += 2*np.pi*int(np.clip(np.rint((seed[joint]-q[joint])/(2*np.pi)), lo, hi))
            if np.any(q < limits[:, 0]-1e-6) or np.any(q > limits[:, 1]+1e-6):
                raise MotionPlanningError("orientation_projection_failed", "Projected path exceeds joint limits")
            self.kin.pmodel.compute_forward_kinematics(q)
            actual = self.kin.pmodel.get_link_pose(self.kin.end_link_idx)
            if np.linalg.norm(actual.p-position) > .001 or _angle(actual.q, quaternion) > .1:
                raise MotionPlanningError("orientation_projection_failed", "Projected IK failed FK verification")
            full = point.copy()
            full[self.pindices] = q
            if np.max(np.abs(full-projected[-1])) > 1e-8:
                projected.append(full)
        return np.asarray(projected)

    @_profile_method("planner_ik_s")
    def _bounded_pose_ik(self, goal, start, limits):
        """Solve a nearby pose with explicit bounds on all controlled joints."""
        from scipy.optimize import least_squares

        target_rotation_inverse = _rotation(goal).inv()
        target_position = np.asarray(goal.p)
        indices = self.controlled
        def residual(values):
            q = start.copy()
            q[indices] = values
            self.kin.pmodel.compute_forward_kinematics(q)
            actual = self.kin.pmodel.get_link_pose(self.kin.end_link_idx)
            return np.concatenate((np.asarray(actual.p) - target_position,
                                   (target_rotation_inverse * _rotation(actual)).as_rotvec()))

        def jacobian(values):
            # SciPy's diff_step is relative: near-zero joints produce steps
            # below native float32 FK resolution and corrupt the derivatives.
            jac = np.empty((6, len(indices)))
            for column, joint in enumerate(indices):
                lower, upper = values.copy(), values.copy()
                lower[column] = max(values[column] - 1e-4, limits[joint, 0])
                upper[column] = min(values[column] + 1e-4, limits[joint, 1])
                jac[:, column] = ((residual(upper) - residual(lower)) /
                                  (upper[column] - lower[column]))
            return jac

        # Dense trust-region steps stalled on a captured 7-joint/6-residual pose.
        # LSMR solves that same bounded objective without raising the work cap.
        fit = least_squares(residual, start[indices],
            bounds=(limits[indices, 0], limits[indices, 1]), jac=jacobian,
            tr_solver="lsmr",
            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
        q = start.copy()
        q[indices] = fit.x
        error = residual(fit.x)
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))
        success = position_error <= .001 and orientation_error <= np.radians(.1)
        return q, success, dict(optimizer_status=int(fit.status), evaluations=int(fit.nfev),
                               position_error_m=position_error,
                               orientation_error_deg=float(np.degrees(orientation_error)))

    @_profile_method("planner_total_s")
    def plan_to_pose(self, position_root3, quaternion_wxyz4, *, planning_time_s=3.0,
                     seed=0, allowed_contact_objects=(), support_surface_root_z=None,
                     maintain_orientation=False, method="rrt_connect", densify_rrt_path=False,
                     start_qpos=None, target_qpos=None):
        position, quaternion = _validate_goal(position_root3, quaternion_wxyz4)
        if method not in {"rrt_connect", "bounded_ik"}:
            raise MotionPlanningError("invalid_goal", "Unknown pose planning method")
        if support_surface_root_z is not None and not np.isfinite(support_surface_root_z):
            raise MotionPlanningError("invalid_support_surface", "Support height must be finite")
        self.support_surface_root_z = support_surface_root_z
        if not isinstance(maintain_orientation, bool):
            raise MotionPlanningError("invalid_goal", "maintain_orientation must be boolean")
        if not np.isfinite(planning_time_s) or planning_time_s <= 0:
            raise MotionPlanningError("invalid_goal", "planning_time_s must be positive and finite")
        self.allowed_contacts = tuple(allowed_contact_objects)
        names = {actor.name for actor in self.env.cabinet_items}
        if any(name not in names for name in self.allowed_contacts):
            raise MotionPlanningError("invalid_goal", "allowed_contact_objects must name current cabinet items")
        self.attachment = self._planning_attachment()
        self.mplib, self.fcl, self.sapien = _native_modules()
        q0full = _array(self.robot.get_qpos()).reshape(9).astype(float)
        q0 = q0full[self.pindices]
        limits = _array(self.robot.get_qlimits()).reshape(9, 2)[self.pindices]
        self.root = self._pose(self.root_link.pose)
        signature = self.snapshot_scene()
        self.kin.pmodel.compute_forward_kinematics(q0)
        for link in self.links:
            expected = self.root * self.kin.pmodel.get_link_pose(self.link_indices[link.name])
            actual = self._pose(link.pose)
            position_error = float(np.linalg.norm(expected.p-actual.p))
            angle_error = _angle(expected.q, actual.q)
            if position_error > 1e-5 or angle_error > .01:
                raise MotionPlanningError("unsupported", f"FK mapping disagrees with live link {link.name}: {position_error} m, {angle_error} deg")
        # Dry-run route checks can chain plans without moving the physical robot.
        # The FK mapping check above must always use the measured live state.
        if start_qpos is not None:
            q0full = np.asarray(start_qpos, dtype=float).copy()
            full_limits = _array(self.robot.get_qlimits()).reshape(9, 2)
            if (q0full.shape != (9,) or not np.isfinite(q0full).all()
                    or np.any(q0full < full_limits[:, 0] - 1e-6)
                    or np.any(q0full > full_limits[:, 1] + 1e-6)):
                raise MotionPlanningError("invalid_start", "Virtual start requires nine finite in-limit joint positions")
            q0 = q0full[self.pindices]
        self._scene_geometry()
        start_contacts = self._check(q0)
        if start_contacts:
            raise MotionPlanningError("path_collision", {"start_contacts": start_contacts})
        rng = np.random.default_rng(seed)
        starts = [q0.copy()] if method == "rrt_connect" else []
        for _ in range(15 if method == "rrt_connect" else 0):
            start = q0.copy()
            start[self.controlled] = rng.uniform(limits[self.controlled, 0], limits[self.controlled, 1])
            starts.append(start)
        candidates = []
        ik_diagnostics = dict(candidate_count=len(starts),
            counts=dict(solver_failure=0, nonfinite_solution=0, joint_limit_violation=0,
                        fk_mismatch=0, collision=0, accepted=0),
            candidates=[])
        goal = self.sapien.Pose(position, quaternion)
        if target_qpos is not None:
            if method != "rrt_connect" or maintain_orientation:
                raise MotionPlanningError("invalid_goal", "Joint homing requires unconstrained RRTConnect")
            target = np.asarray(target_qpos, dtype=float)
            if target.shape != (9,) or not np.isfinite(target).all():
                raise MotionPlanningError("invalid_goal", "Joint target requires nine finite positions")
            q = target[self.pindices]
            if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
                raise MotionPlanningError("invalid_goal", "Joint target exceeds limits")
            if np.max(np.abs(target[7:] - q0full[7:])) > .001:
                raise MotionPlanningError("invalid_goal", "Open the gripper before joint homing")
            self.kin.pmodel.compute_forward_kinematics(q)
            fk = self.kin.pmodel.get_link_pose(self.kin.end_link_idx)
            if np.linalg.norm(fk.p-position) > .001 or _angle(fk.q, quaternion) > .1:
                raise MotionPlanningError("invalid_goal", "Joint target disagrees with requested home TCP pose")
            contacts = self._check(q)
            if contacts:
                raise MotionPlanningError("path_collision", {"target_contacts": contacts})
            candidates = [q.copy()]
            starts = []
            ik_diagnostics["candidate_count"] = 0
        for candidate_index, start in enumerate(starts):
            q, success, _ = _profile_call(self, "planner_ik_s", self.kin.pmodel.compute_inverse_kinematics, self.kin.end_link_idx, goal,
                initial_qpos=start, active_qmask=_array(self.kin.qmask), max_iterations=1000)
            q = np.asarray(q).copy()
            diagnostic = dict(candidate_index=candidate_index,
                initial_seed_source="current_qpos" if candidate_index == 0 else "random_uniform",
                random_draw_index=None if candidate_index == 0 else candidate_index - 1,
                initial_seed_qpos_rad=_finite_list(start))
            if not success:
                diagnostic.update(status="rejected", reason="solver_failure")
                ik_diagnostics["counts"]["solver_failure"] += 1
                ik_diagnostics["candidates"].append(diagnostic)
                continue
            if not np.all(np.isfinite(q)):
                diagnostic.update(status="rejected", reason="nonfinite_solution",
                    solution_qpos_rad=_finite_list(q),
                    nonfinite_solution_indices=np.flatnonzero(~np.isfinite(q)).astype(int).tolist())
                ik_diagnostics["counts"]["nonfinite_solution"] += 1
                ik_diagnostics["candidates"].append(diagnostic)
                continue
            for j in self.controlled:
                # Only the seven Panda revolute joints have 2pi equivalents.
                if self.pindices[j] >= 7:
                    continue
                lo = int(np.ceil((limits[j, 0] - q[j]) / (2*np.pi)))
                hi = int(np.floor((limits[j, 1] - q[j]) / (2*np.pi)))
                if lo <= hi:
                    k = int(np.clip(np.rint((q0[j]-q[j])/(2*np.pi)), lo, hi))
                    q[j] += 2*np.pi*k
            violations = np.flatnonzero((q < limits[:, 0]-1e-6) | (q > limits[:, 1]+1e-6))
            if len(violations):
                diagnostic.update(status="rejected", reason="joint_limit_violation",
                    solution_qpos_rad=_finite_list(q),
                    joint_limit_violations=[dict(joint_index=int(j), value_rad=float(q[j]),
                        lower_rad=float(limits[j, 0]), upper_rad=float(limits[j, 1]))
                        for j in violations])
                ik_diagnostics["counts"]["joint_limit_violation"] += 1
                ik_diagnostics["candidates"].append(diagnostic)
                continue
            self.kin.pmodel.compute_forward_kinematics(q)
            fk = self.kin.pmodel.get_link_pose(self.kin.end_link_idx)
            fk_position_error = float(np.linalg.norm(fk.p-position))
            diagnostic.update(solution_qpos_rad=_finite_list(q),
                fk_position_error_m=_finite_float(fk_position_error))
            if fk_position_error <= .001:
                fk_angle_error = _angle(fk.q, quaternion)
                diagnostic["fk_orientation_error_deg"] = _finite_float(fk_angle_error)
                if fk_angle_error <= .1:
                    contacts = self._check(q)
                    if contacts:
                        diagnostic.update(status="rejected", reason="collision", collision_contacts=contacts)
                        ik_diagnostics["counts"]["collision"] += 1
                        ik_diagnostics["candidates"].append(diagnostic)
                        continue
                    candidates.append(q)
                    diagnostic.update(status="accepted", reason="accepted")
                    ik_diagnostics["counts"]["accepted"] += 1
                else:
                    diagnostic.update(status="rejected", reason="fk_mismatch",
                        fk_mismatch_axes=["orientation"])
                    ik_diagnostics["counts"]["fk_mismatch"] += 1
            else:
                diagnostic.update(status="rejected", reason="fk_mismatch",
                    fk_mismatch_axes=["position"])
                ik_diagnostics["counts"]["fk_mismatch"] += 1
            ik_diagnostics["candidates"].append(diagnostic)
        candidates.sort(key=lambda q: np.linalg.norm((q-q0)[self.controlled]))
        if not candidates and method == "rrt_connect":
            raise MotionPlanningError("no_collision_free_ik", {
                "message": "No valid collision-free solution in 16 fixed-orientation IK starts",
                "ik_diagnostics": ik_diagnostics})
        planner = self._make_planner()
        # Path-search timings include their nested TOPP work; do not sum them.
        if isinstance(getattr(self.env, "_collection_profile", None), dict):
            topp = planner.TOPP
            planner.TOPP = lambda *args, **kwargs: _profile_call(
                self, "planner_topp_s", topp, *args, **kwargs)
        planner.set_base_pose(_pose_vector(self.root))
        inflated, uninflated, omitted = [], [], []
        for obstacle in self.obstacles:
            kind = self._kind(obstacle["shape"])
            if kind == "Plane" or obstacle["actor"] in self.allowed_contacts or obstacle["name"] in self.support_contacts:
                omitted.append(obstacle["name"])
                continue
            if method == "bounded_ik":
                # Bounded Cartesian waypoints use the actual-shape audit below.
                continue
            changed = kind in ("Box", "Cylinder", "Sphere")
            geometry = self._geometry(obstacle["shape"], .001 if changed else 0.)
            planner.set_normal_object(obstacle["name"], self._object(geometry, obstacle["pose"]))
            (inflated if changed else uninflated).append(obstacle["name"])
        dt = 1 / self.env.control_freq
        attempts = []
        for q in ([None] if method == "bounded_ik" else candidates[:3]):
            qfull = q0full.copy()
            if q is not None:
                qfull[self.pindices] = q
            try:
                if method == "bounded_ik":
                    self.kin.pmodel.compute_forward_kinematics(q0)
                    initial_position = np.asarray(self.kin.pmodel.get_link_pose(self.kin.end_link_idx).p).copy()
                    count = max(1, int(np.ceil(np.linalg.norm(position-initial_position)/.005)))
                    waypoints = [q0full.copy()]
                    seed_q = q0.copy()
                    for point in np.linspace(initial_position, position, count+1)[1:]:
                        candidate, success, detail = self._bounded_pose_ik(
                            self.sapien.Pose(point, quaternion), seed_q, limits)
                        contacts = self._check(candidate) if success else []
                        if not success or contacts:
                            raise MotionPlanningError("cartesian_path_failed", dict(
                                target_position_root=point.tolist(), contacts=contacts,
                                pose_ik_succeeded=bool(success), **detail))
                        ik_diagnostics["candidates"].append(dict(
                            status="accepted", solution_qpos_rad=_finite_list(candidate), **detail))
                        full = q0full.copy()
                        full[self.pindices] = candidate
                        waypoints.append(full)
                        seed_q = candidate
                    ik_diagnostics["candidate_count"] = count
                    ik_diagnostics["counts"]["accepted"] = count
                    qfull = waypoints[-1]
                    times, pos, vel, acc, duration = planner.TOPP(
                        np.asarray(waypoints)[:, planner.move_group_joint_indices], dt)
                    # MPLib samples int(duration/dt) points: a short segment can
                    # omit its endpoint or even have no samples. Preserve its
                    # actual TOPP duration before control-grid resampling.
                    if duration > 0 and len(times) < 2:
                        times = np.array([0., duration])
                        pos = np.stack([q0full, qfull])[:, planner.move_group_joint_indices]
                    path = dict(status="Success", time=times, position=pos)
                elif densify_rrt_path:
                    # TOPP's cubic spline can cut corners of sparse RRT paths.
                    # Preserve the same upstream RRT settings, adding waypoints
                    # before its existing timing; exact postchecks remain below.
                    planner.robot.set_qpos(q0full, True)
                    planner.planning_world.set_use_point_cloud(False)
                    planner.planning_world.set_use_attach(False)
                    status, raw_path = _profile_call(self, "planner_path_search_s", planner.planner.plan,
                        q0full[planner.move_group_joint_indices],
                        [qfull[planner.move_group_joint_indices]], range=.1,
                        time=float(planning_time_s), fixed_joints=set(),
                        planner_name="RRTConnect", no_simplification=False, verbose=False)
                    if status == "Exact solution":
                        dense = np.asarray(list(_interpolate(raw_path, max_step=.025)))
                        times, pos, _, _, duration = planner.TOPP(dense, dt)
                        path = dict(status="Success", time=times, position=pos)
                    else:
                        path = dict(status="RRT Failed. "+status)
                else:
                    path = _profile_call(self, "planner_path_search_s", planner.plan_qpos_to_qpos, goal_qposes=[qfull], current_qpos=q0full.copy(),
                        time_step=dt, fix_joint_limits=False, planning_time=float(planning_time_s))
            except RuntimeError as exc:
                if str(exc) != "Fail to parameterize path":
                    raise
                raise MotionPlanningError("time_parameterization_failed",
                    "MPLib TOPP could not time-parameterize the planned path; no motion was executed") from exc
            attempt = dict(planner_status=path["status"])
            attempts.append(attempt)
            if path["status"] != "Success":
                continue
            positions = np.tile(q0full, (len(path["position"]), 1))
            positions[:, planner.move_group_joint_indices] = path["position"]
            times = np.asarray(path["time"]).reshape(-1)
            if (not len(positions) or not np.all(np.isfinite(positions))
                    or times.shape != (len(positions),) or not np.all(np.isfinite(times))
                    or times[0] != 0. or np.any(np.diff(times) <= 0)
                    or np.max(np.abs(positions[0] - q0full)) > 1e-4
                    or np.max(np.abs(positions[-1] - qfull)) > 1e-4):
                attempt["postcheck_status"] = "invalid_path"
                continue
            if maintain_orientation and method == "rrt_connect" and len(positions) > 1:
                try:
                    projected = self._fixed_orientation_path(positions, quaternion, limits)
                    times, joint_positions, _, _, _ = planner.TOPP(projected[:, planner.move_group_joint_indices], dt)
                    positions = np.tile(q0full, (len(joint_positions), 1))
                    positions[:, planner.move_group_joint_indices] = joint_positions
                    times = np.asarray(times).reshape(-1)
                except MotionPlanningError as exc:
                    attempt.update(postcheck_status=exc.status, detail=exc.detail)
                    continue
                except RuntimeError as exc:
                    attempt.update(postcheck_status="orientation_time_parameterization_failed", detail=str(exc))
                    continue
            if maintain_orientation and (not len(positions) or not np.isfinite(positions).all()
                    or times.shape != (len(positions),) or not np.isfinite(times).all()
                    or times[0] != 0. or np.any(np.diff(times) <= 0)
                    or np.max(np.abs(positions[0]-q0full)) > 1e-4
                    or np.max(np.abs(positions[-1]-qfull)) > 1e-4):
                attempt.update(postcheck_status="invalid_projected_path")
                continue
            source_duration = float(times[-1])
            positions, times = _resample_to_control_dt(positions, times, dt)
            checked = 0
            for point in _interpolate(np.vstack([q0full, positions, qfull])):
                checked += 1
                qp = point[self.pindices]
                contacts = self._check(qp)
                if maintain_orientation:
                    angle = _angle(self.kin.pmodel.get_link_pose(self.kin.end_link_idx).q, quaternion)
                    if angle > .5:
                        attempt.update(postcheck_status="orientation_path_violation", observed_angle_deg=angle)
                        break
                if np.any(qp < limits[:, 0]-1e-6) or np.any(qp > limits[:, 1]+1e-6) or contacts:
                    attempt.update(postcheck_status="path_collision", contacts=contacts)
                    break
            else:
                metadata = dict(status="Success", planner="MPLib", version=self.mplib.__version__,
                    algorithm="RRTConnect" if method == "rrt_connect" else "bounded_Cartesian_IK_TOPP",
                    rrt_pre_TOPP_max_joint_step_rad=.025 if densify_rrt_path else None,
                    requested_seed=int(seed), ik_seed_applied=method == "rrt_connect" and target_qpos is None,
                    planner_seed_applied=False, planner_seed_limitation="MPLib 0.1.1 exposes no seed setter",
                    obstacle_margin_m=.001 if method == "rrt_connect" else 0., inflated_obstacles=inflated, uninflated_obstacles=uninflated,
                    uninflated_reason="Actual convex geometry retained; no approximate inflation",
                    omitted_from_planner=omitted, allowed_contact_objects=list(self.allowed_contacts),
                    allowed_contact_links=["panda_leftfinger", "panda_rightfinger"], base_ground_contact_allowed=True,
                    attached_objects_supported=True, attachment=self.attachment,
                    attachment_planning_geometry="Fixed TCP collision link with actual compound shapes; exact compound postcheck" if self.attachment else None,
                    support_contacts=dict(self.support_contacts), maximum_initial_support_penetration_m=.001,
                    support_surface_root_z=support_surface_root_z,
                    maintain_orientation=maintain_orientation,
                    fixed_orientation_path_method=("RRT_TCP_positions_projected_by_SAPIEN_IK_then_TOPP"
                        if method == "rrt_connect" else "bounded_Cartesian_IK_TOPP") if maintain_orientation else None,
                    attachment_position_tolerance_m=.002, attachment_angle_tolerance_deg=2.,
                    postcheck_points=checked, max_postcheck_joint_step_rad=.01,
                    collision_limitation="Finite samples do not certify continuous collision freedom",
                    planning_time_s_per_candidate=float(planning_time_s), time_step_s=dt,
                    source_duration_s=source_duration, scheduled_duration_s=float(times[-1]),
                    resampled_to_control_dt=True,
                    joint_velocity_limit_rad_s=.5, joint_acceleration_limit_rad_s2=1., attempts=attempts,
                    ik_diagnostics=ik_diagnostics)
                return MotionPlan(positions, times, position, quaternion,
                                  q0full, _pose_vector(self.root), signature, metadata)
        status = "path_collision" if any("postcheck_status" in a for a in attempts) else "planning_failed"
        if maintain_orientation:
            status = next((a["postcheck_status"] for a in attempts if "postcheck_status" in a), "planning_failed")
        raise MotionPlanningError(status, {"attempts": attempts, "ik_diagnostics": ik_diagnostics})
