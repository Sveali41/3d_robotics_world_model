"""State encoder for the CEE-US cabinet task."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


AGENT_DIM = 32
OBJECT_DYN_DIM = 12
OBJECT_STAT_DIM = 5
OBJECT_NAMES = ("cabinet_item_16", "cabinet_item_17")
OBJECT_TYPES = ("mug", "bowl")


def _numpy(value, name: str, size: int) -> np.ndarray:
    """Convert ManiSkill's batched Torch values or fixture arrays to a vector."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} values; got shape {array.shape}")
    array = array.reshape(size)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _pose(pose, name: str):
    position = _numpy(pose.p, f"{name}.position", 3)
    quaternion = _numpy(pose.q, f"{name}.quaternion", 4)
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
        raise ValueError(f"{name}.quaternion has zero norm")
    # ManiSkill/SAPIEN poses use wxyz quaternion order.
    rotation = Rotation.from_quat(np.roll(quaternion / norm, -1)).as_matrix()
    return position, rotation


def _euler_xyz(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rotation).as_euler("xyz")


class CabinetStateEncoder:
    """Encode the CEE-US cabinet state in the Panda controller root frame."""

    def __init__(self, env, dt: float = 0.05):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        self.dt = float(dt)
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        control_freq = getattr(self.env, "control_freq", None)
        if control_freq is None or float(control_freq) != 20.0:
            raise ValueError(
                f"CabinetStateEncoder requires control_freq=20 Hz; got {control_freq!r}"
            )
        if not np.isclose(self.dt, 1.0 / float(control_freq), atol=1e-12, rtol=0):
            raise ValueError("dt must match the 20 Hz controller interval (0.05 seconds)")
        self.controller = self.env.agent.controller.controllers["arm"]
        self.root_link = self.controller.root_link
        self._root_position, self._root_rotation = _pose(
            self.root_link.pose, "robot root pose"
        )

        actors = {getattr(actor, "name", None): actor
                  for actor in getattr(self.env, "cabinet_items", [])}
        object_names = OBJECT_NAMES
        object_types = OBJECT_TYPES
        missing = [name for name in object_names if name not in actors]
        if missing:
            raise ValueError(f"Missing required cabinet actors: {missing}")
        item_types = getattr(self.env, "cabinet_item_types", [])
        type_by_name = {
            getattr(actor, "name", None): item_type
            for actor, item_type in zip(self.env.cabinet_items, item_types)
        }
        if tuple(type_by_name.get(name) for name in object_names) != object_types:
            raise ValueError(
                f"Expected cabinet object types {object_types!r}; "
                f"got {tuple(type_by_name.get(name) for name in object_names)!r}"
            )

        self.object_names = object_names
        self.object_types = object_types
        self.objects = tuple((name, actors[name]) for name in object_names)
        self.collision_extents = {}
        self.collision_metadata = {}
        for name, actor in self.objects:
            mesh = actor.get_first_collision_mesh(to_world_frame=False)
            if mesh is None:
                raise ValueError(f"{name} has no collision mesh")
            extents = _numpy(mesh.extents, f"{name} collision mesh extents", 3)
            if np.any(extents <= 0):
                raise ValueError(f"{name} collision mesh extents must be positive")
            self.collision_extents[name] = extents.astype(np.float32)
            self.collision_metadata[name] = {
                "source": "get_first_collision_mesh(to_world_frame=False)",
                "mesh_type": type(mesh).__name__,
                "extents_m": extents.tolist(),
            }

    def _root_transform(self):
        position, rotation = _pose(self.root_link.pose, "robot root pose")
        if not np.allclose(position, self._root_position, atol=1e-5, rtol=0) or not np.allclose(
            rotation, self._root_rotation, atol=1e-5, rtol=0
        ):
            raise RuntimeError("Robot controller root moved after encoder initialization")
        return position, rotation

    def encode(self) -> np.ndarray:
        root_position, root_rotation = self._root_transform()
        world_to_root = root_rotation.T

        robot = self.env.agent.robot
        qpos = _numpy(robot.qpos, "robot qpos", 9)
        qvel = _numpy(robot.qvel, "robot qvel", 9)

        tcp_position_world, tcp_rotation_world = _pose(
            self.env.agent.tcp.pose, "TCP pose"
        )
        tcp_position = world_to_root @ (tcp_position_world - root_position)
        tcp_rotation = world_to_root @ tcp_rotation_world
        tcp_linear_velocity = world_to_root @ _numpy(
            self.env.agent.tcp.linear_velocity, "TCP linear velocity", 3
        )
        tcp_angular_velocity = world_to_root @ _numpy(
            self.env.agent.tcp.angular_velocity, "TCP angular velocity", 3
        )

        door_angle = _numpy(self.env.door_joint.qpos, "door qpos", 1)
        door_qvel = _numpy(self.env.door_joint.qvel, "door qvel", 1)
        state = [
            *qpos,
            *(qvel * self.dt),
            *tcp_position,
            *_euler_xyz(tcp_rotation),
            *(tcp_linear_velocity * self.dt),
            *(tcp_angular_velocity * self.dt),
            *door_angle,
            *(door_qvel * self.dt),
        ]

        for name, actor in self.objects:
            position_world, rotation_world = _pose(actor.pose, f"{name} pose")
            position = world_to_root @ (position_world - root_position)
            rotation = world_to_root @ rotation_world
            linear_velocity = world_to_root @ _numpy(
                actor.linear_velocity, f"{name} linear velocity", 3
            )
            angular_velocity = world_to_root @ _numpy(
                actor.angular_velocity, f"{name} angular velocity", 3
            )
            state.extend((
                *position,
                *_euler_xyz(rotation),
                *(linear_velocity * self.dt),
                *(angular_velocity * self.dt),
            ))

        for object_type, (name, _) in zip(self.object_types, self.objects):
            one_hot = np.zeros(2, dtype=np.float32)
            if object_type in OBJECT_TYPES:
                one_hot[OBJECT_TYPES.index(object_type)] = 1.0
            state.extend((*one_hot, *self.collision_extents[name]))

        result = np.asarray(state, dtype=np.float32)
        if result.shape != (AGENT_DIM + len(self.objects) * (OBJECT_DYN_DIM + OBJECT_STAT_DIM),):
            raise ValueError(f"Encoded state has unexpected shape {result.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError("Encoded state contains non-finite values")
        return result

    def action_projection_contract(self) -> dict:
        """Fixed geometry used for identical real and imagined goal clipping."""
        cabinet_position, cabinet_rotation = _pose(self.env.cabinet.pose, "cabinet")
        rotation = cabinet_rotation.T @ self._root_rotation
        translation = cabinet_rotation.T @ (self._root_position - cabinet_position)
        lower, upper = self.env.exploration_region_bounds_local
        cfg = self.controller.config
        return {
            "root_to_cabinet_rotation": rotation.tolist(),
            "root_to_cabinet_translation": translation.tolist(),
            "lower": np.asarray(lower).tolist(), "upper": np.asarray(upper).tolist(),
            "pos_lower": np.broadcast_to(cfg.pos_lower, (3,)).tolist(),
            "pos_upper": np.broadcast_to(cfg.pos_upper, (3,)).tolist(),
        }

    def contract(self) -> dict:
        """Return JSON-serializable dimensions, order, units, and scene metadata."""
        fields = [
            {"name": "robot_qpos", "shape": [9], "units": "arm rad; gripper m"},
            {"name": "robot_qvel_dt", "shape": [9], "units": "arm rad/s and gripper m/s, scaled by dt"},
            {"name": "tcp_root_position", "shape": [3], "units": "m"},
            {"name": "tcp_root_euler_xyz", "shape": [3], "units": "rad"},
            {"name": "tcp_root_linear_velocity_dt", "shape": [3], "units": "m per control step"},
            {"name": "tcp_root_angular_velocity_dt", "shape": [3], "units": "rad per control step"},
            {"name": "door_angle", "shape": [1], "units": "rad"},
            {"name": "door_qvel_dt", "shape": [1], "units": "rad per control step"},
        ]
        for name in self.object_names:
            fields.append({"name": f"{name}.root_position", "shape": [3], "units": "m"})
            fields.append({"name": f"{name}.root_euler_xyz", "shape": [3], "units": "rad"})
            fields.append({"name": f"{name}.root_linear_velocity_dt", "shape": [3], "units": "m per control step"})
            fields.append({"name": f"{name}.root_angular_velocity_dt", "shape": [3], "units": "rad per control step"})
        for name in self.object_names:
            fields.append({"name": f"{name}.type_one_hot_mug_bowl", "shape": [2], "units": "binary"})
            fields.append({"name": f"{name}.collision_mesh_extents", "shape": [3], "units": "m"})
        position, rotation = self._root_position, self._root_rotation
        return {
            "dtype": "float32",
            "dimension": AGENT_DIM + len(self.objects) * (OBJECT_DYN_DIM + OBJECT_STAT_DIM),
            "dt_seconds": self.dt,
            "control_freq_hz": 20,
            "layout": ["agent", "object_dynamics_sorted_by_name", "object_static_sorted_by_name"],
            "agent_dim": AGENT_DIM,
            "object_dyn_dim": OBJECT_DYN_DIM,
            "object_stat_dim": OBJECT_STAT_DIM,
            "object_names": list(self.object_names),
            "object_types": list(self.object_types),
            "fields": fields,
            "root_transform_world": {
                "position_m": position.tolist(),
                "quaternion_wxyz": np.roll(Rotation.from_matrix(rotation).as_quat(), 1).tolist(),
            },
            "collision_meshes": self.collision_metadata,
        }


class CabinetSkillStateEncoder(CabinetStateEncoder):
    """Expose the persistent skill-controller command as an observed state."""

    def __init__(self, env, *, latched_gripper=False):
        super().__init__(env)
        self.latched_gripper = bool(latched_gripper)
        self.gripper_command = 1.0

    def encode(self):
        state = super().encode()
        if not self.latched_gripper:
            return state
        if not np.isfinite(self.gripper_command) or not -1 <= self.gripper_command <= 1:
            raise ValueError("Skill gripper command must be finite and normalized")
        return np.insert(state, AGENT_DIM, self.gripper_command).astype(np.float32)

    def contract(self):
        contract = super().contract()
        if self.latched_gripper:
            contract["agent_dim"] += 1
            contract["dimension"] += 1
            contract["fields"].insert(8, dict(name="last_gripper_command", shape=[1],
                                              units="normalized controller command"))
            contract["controller_state"] = "latched_gripper_v1"
        return contract
