"""Official-RAPS-compatible primitives over fixed-base robot controllers.

The policy-facing action contains only a primitive one-hot vector and a bank of
normalized continuous arguments.  Primitive execution depends on robot state
and those arguments; simulator object IDs and poses are used only for outcome
logging. Every primitive emits a robot-independent
``[dx, dy, dz, gripper]`` command; the environment wrapper maps that command
to the selected robot controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

MOVED_OBJECT_THRESHOLD = 0.015


def update_cumulative_lifts(reference_heights, before_positions, after_positions,
                            stable_frames, held_ids):
    """Track lift from the start of each uninterrupted grasp, not one EE tick."""
    for object_id in list(reference_heights):
        if stable_frames.get(object_id, 0) <= 0 or object_id not in after_positions:
            del reference_heights[object_id]
    for object_id, frames in stable_frames.items():
        if frames > 0 and object_id in before_positions and object_id in after_positions:
            reference_heights.setdefault(object_id, float(before_positions[object_id][2]))
    displacements = {
        object_id: float(after_positions[object_id][2]) - height
        for object_id, height in reference_heights.items()
    }
    lifted = sorted(object_id for object_id in held_ids
                    if stable_frames.get(object_id, 0) >= 4
                    and displacements.get(object_id, 0.0) >= 0.01)
    return lifted, displacements


SUPPORTED_PRIMITIVE_ROBOTS = ("fetch", "panda")
_ROBOT_ALIASES = {
    "fetch": "fetch",
    "panda": "panda",
    "panda_wristcam": "panda",
}
_GRIPPER_JOINT_NAMES = {
    "fetch": ("l_gripper_finger_joint", "r_gripper_finger_joint"),
    "panda": ("panda_finger_joint1", "panda_finger_joint2"),
}
_GRIPPER_OPEN_SUM = {
    "fetch": 0.10,
    "panda": 0.08,
}


def normalize_primitive_robot(robot: str) -> str:
    """Return the canonical robot name used by the primitive adapter."""
    name = str(robot).lower()
    try:
        return _ROBOT_ALIASES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported primitive robot {robot!r}; "
            f"expected one of {SUPPORTED_PRIMITIVE_ROBOTS}"
        ) from exc


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class PrimitiveParameter:
    name: str
    scale: float

    def normalize(self, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError(f"Parameter {self.name!r} must be finite, got {value}")
        return float(np.clip(value / self.scale, -1.0, 1.0))

    def denormalize(self, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError(
                f"Normalized parameter {self.name!r} must be finite, got {value}"
            )
        return float(np.clip(value, -1.0, 1.0) * self.scale)


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    parameters: tuple[PrimitiveParameter, ...]


# This is the action layout used by the official RAPS RoboSuite kitchen
# wrapper: 10 primitive logits/one-hot entries followed by one shared 10D
# argument bank.  Arguments are normalized to [-1, 1]; the scales only adapt
# the official contract to RoboCasa's metric EE controller.
RAPS_PRIMITIVE_SPECS = (
    PrimitiveSpec(
        "move_delta_ee_pose",
        (
            PrimitiveParameter("dx", 0.15),
            PrimitiveParameter("dy", 0.15),
            PrimitiveParameter("dz", 0.12),
        ),
    ),
    # Cabinet task: horizontal side-grasp. Approach is always along robot-base
    # +x. The policy learns y_offset and z_offset to aim at the target object
    # (directly readable from tcp_to_mug / tcp_to_bowl in the observation).
    PrimitiveSpec(
        "top_grasp",
        (
            PrimitiveParameter("x_forward", 0.35),  # push depth along +x (covers full drift range)
            PrimitiveParameter("y_offset", 0.20),   # lateral offset to target
            PrimitiveParameter("z_offset", 0.12),   # vertical offset to target
        ),
    ),
    PrimitiveSpec("lift", (PrimitiveParameter("z_dist", 0.16),)),
    PrimitiveSpec("drop", (PrimitiveParameter("z_dist", 0.16),)),
    PrimitiveSpec("move_left", (PrimitiveParameter("y_dist", 0.15),)),
    PrimitiveSpec("move_right", (PrimitiveParameter("y_dist", 0.15),)),
    PrimitiveSpec("move_forward", (PrimitiveParameter("x_dist", 0.15),)),
    PrimitiveSpec("move_backward", (PrimitiveParameter("x_dist", 0.15),)),
    PrimitiveSpec("open_gripper", ()),
    PrimitiveSpec("close_gripper", ()),
)


class RAPSActionSchema:
    """Fixed hybrid action layout: primitive one-hot + argument bank."""

    def __init__(self, specs=RAPS_PRIMITIVE_SPECS):
        self.specs = tuple(specs)
        self.spec_by_name = {spec.name: spec for spec in self.specs}
        if len(self.spec_by_name) != len(self.specs):
            raise ValueError("Primitive names must be unique")
        self.primitive_names = tuple(spec.name for spec in self.specs)
        self.num_primitives = len(self.specs)
        official_indices = {
            "move_delta_ee_pose": [0, 1, 2], "top_grasp": [3, 4, 5],
            "lift": [4], "drop": [5], "move_left": [6], "move_right": [7],
            "move_forward": [8], "move_backward": [9],
            "open_gripper": [], "close_gripper": [],
        }
        self.primitive_name_to_action_idx = official_indices
        self.max_arg_len = 10
        self.parameter_slices = {
            name: (indices[0], indices[-1] + 1) if indices else (0, 0)
            for name, indices in official_indices.items()
        }
        self.num_parameters = self.max_arg_len
        self.action_dim = self.num_primitives + self.num_parameters
        self.low = np.concatenate(
            [np.zeros(self.num_primitives), -np.ones(self.num_parameters)]
        ).astype(np.float32)
        self.high = np.ones(self.action_dim, dtype=np.float32)

    @property
    def continuous_action_dim(self) -> int:
        return self.max_arg_len

    def encode(self, action: "PrimitiveAction") -> np.ndarray:
        if action.primitive_type not in self.spec_by_name:
            raise KeyError(
                f"Unknown primitive {action.primitive_type!r}; "
                f"available={self.primitive_names}"
            )
        vector = np.zeros(self.action_dim, dtype=np.float32)
        primitive_index = self.primitive_names.index(action.primitive_type)
        vector[primitive_index] = 1.0
        spec = self.spec_by_name[action.primitive_type]
        indices = self.primitive_name_to_action_idx[action.primitive_type]
        for index, parameter in enumerate(spec.parameters):
            if parameter.name not in action.parameters:
                raise ValueError(
                    f"{action.primitive_type} requires parameter {parameter.name!r}"
                )
            vector[self.num_primitives + indices[index]] = parameter.normalize(
                action.parameters[parameter.name]
            )
        return vector

    def decode(self, vector) -> "PrimitiveAction":
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.action_dim,):
            raise ValueError(
                f"Expected RAPS action shape {(self.action_dim,)}, got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("RAPS action contains NaN or infinity")
        primitive_index = int(np.argmax(vector[: self.num_primitives]))
        spec = self.specs[primitive_index]
        indices = self.primitive_name_to_action_idx[spec.name]
        parameters = {
            parameter.name: parameter.denormalize(
                vector[self.num_primitives + indices[index]]
            )
            for index, parameter in enumerate(spec.parameters)
        }
        return PrimitiveAction(spec.name, parameters)

    def sample_parameters(self, primitive_type: str, rng) -> dict[str, float]:
        spec = self.spec_by_name[primitive_type]
        return {
            parameter.name: float(rng.uniform(-parameter.scale, parameter.scale))
            for parameter in spec.parameters
        }


RAPS_ACTION_SCHEMA = RAPSActionSchema()


@dataclass(frozen=True)
class PrimitiveAction:
    primitive_type: str
    parameters: dict[str, float]

    def to_vector(self) -> np.ndarray:
        return RAPS_ACTION_SCHEMA.encode(self)

    @classmethod
    def from_vector(cls, vector) -> "PrimitiveAction":
        return RAPS_ACTION_SCHEMA.decode(vector)


@dataclass
class ObjectState:
    object_id: str
    scene_id: int
    actor: Any
    position_root: np.ndarray
    position_world: np.ndarray


class PrimitiveContext:
    """Robot state for control and optional dense-trajectory recording."""

    def __init__(self, env, *, robot: str = "fetch", record_state: bool = True,
                 project_goals_to_region: bool = False):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.project_goals_to_region = bool(project_goals_to_region)
        self.robot = normalize_primitive_robot(robot)
        self.agent = self.env.agent
        self.arm_controller = self.agent.controller.controllers["arm"]
        self.ee_position = _numpy(self.arm_controller.ee_pose_at_base.p[0]).astype(
            np.float32
        )
        self.gripper_open = self._gripper_open_fraction()
        if record_state:
            # Recorder-facing aliases used by the WDS conversion code.
            ee_pose = self.arm_controller.ee_pose_at_base
            ee_quaternion = _numpy(ee_pose.q[0]).astype(np.float32)
            self.ee_pose = np.concatenate(
                [self.ee_position, ee_quaternion], axis=0
            ).astype(np.float32)
            self.qpos = _numpy(self.agent.robot.get_qpos()[0]).astype(np.float32)
            self.qvel = _numpy(self.agent.robot.get_qvel()[0]).astype(np.float32)
            robot_pose = self.agent.robot.pose
            base_position = _numpy(robot_pose.p)
            base_quaternion = _numpy(robot_pose.q)
            if base_position.ndim == 2:
                base_position = base_position[0]
            if base_quaternion.ndim == 2:
                base_quaternion = base_quaternion[0]
            self.base_pose = np.concatenate(
                [base_position, base_quaternion], axis=0
            ).astype(np.float32)
            self.body_qpos = self._body_qpos()
        self.objects = self._read_objects()

    def project_goal(self, goal: np.ndarray) -> np.ndarray:
        """Clamp a controller-root goal in the actual cabinet-local volume."""
        goal = np.asarray(goal, dtype=np.float32)
        if not self.project_goals_to_region:
            return goal.copy()
        if not hasattr(self.env, "cabinet") or not hasattr(
            self.env, "exploration_region_bounds_local"
        ):
            raise RuntimeError(
                "Goal projection requires cabinet.pose and exploration_region_bounds_local"
            )
        root_world = _numpy(
            self.arm_controller.root_link.pose.to_transformation_matrix()
        ).reshape(-1, 4, 4)[0]
        cabinet_world = _numpy(
            self.env.cabinet.pose.to_transformation_matrix()
        ).reshape(-1, 4, 4)[0]
        root_to_cabinet = np.linalg.inv(cabinet_world) @ root_world
        local = (root_to_cabinet @ np.append(goal, 1.0))[:3]
        lower, upper = self.env.exploration_region_bounds_local
        clamped = np.clip(local, lower, upper)
        if np.array_equal(local, clamped):
            return goal.copy()
        return (np.linalg.inv(root_to_cabinet) @ np.append(clamped, 1.0))[:3].astype(np.float32)

    def _body_qpos(self) -> np.ndarray:
        controllers = getattr(self.agent.controller, "controllers", {})
        body_controller = controllers.get("body")
        if body_controller is None:
            return np.empty(0, dtype=np.float32)
        active_names = [
            joint.name for joint in self.agent.robot.get_active_joints()
        ]
        qpos = _numpy(self.agent.robot.get_qpos()[0])
        indices = [
            active_names.index(joint.name)
            for joint in body_controller.joints
        ]
        return np.asarray(qpos[indices], dtype=np.float32)

    def _gripper_open_fraction(self) -> float:
        qpos = _numpy(self.agent.robot.get_qpos()[0])
        active_names = [joint.name for joint in self.agent.robot.get_active_joints()]
        joint_names = _GRIPPER_JOINT_NAMES[self.robot]
        try:
            indices = [active_names.index(name) for name in joint_names]
        except ValueError as exc:
            raise RuntimeError(
                f"robot={self.robot!r} expects gripper joints {joint_names}, "
                f"but the environment exposes {active_names}"
            ) from exc
        return float(
            np.clip(qpos[indices].sum() / _GRIPPER_OPEN_SUM[self.robot], 0.0, 1.0)
        )

    def _read_objects(self) -> dict[str, ObjectState]:
        objects = {}
        root_inv = self.arm_controller.root_link.pose.inv()
        if hasattr(self.env, "object_actors"):
            actors = (
                (object_id, actor_data["actor"])
                for object_id, actor_data in self.env.object_actors[0].items()
            )
        elif hasattr(self.env, "cabinet_items"):
            actors = (
                (getattr(actor, "name", f"cabinet_item_{index:02d}"), actor)
                for index, actor in enumerate(self.env.cabinet_items)
            )
        else:
            actors = ()

        for object_id, actor in actors:
            pose_root = root_inv * actor.pose
            objects[object_id] = ObjectState(
                object_id=object_id,
                scene_id=int(actor.per_scene_id[0]),
                actor=actor,
                position_root=_numpy(pose_root.p[0]).astype(np.float32),
                position_world=_numpy(actor.pose.p[0]).astype(np.float32),
            )
        return objects

    def finger_contact_force(self, object_id: str) -> tuple[float, float]:
        actor = self.objects[object_id].actor
        # PandaWristCam declares the pad-link attributes but leaves them as
        # ``None`` because its URDF only exposes the regular finger links.
        # ``getattr(..., fallback)`` is insufficient in that case.
        left_link = getattr(self.agent, "finger1pad_link", None) or self.agent.finger1_link
        right_link = getattr(self.agent, "finger2pad_link", None) or self.agent.finger2_link
        left = self.env.scene.get_pairwise_contact_forces(left_link, actor)
        right = self.env.scene.get_pairwise_contact_forces(right_link, actor)
        return (
            float(torch.linalg.norm(left, dim=-1)[0].item()),
            float(torch.linalg.norm(right, dim=-1)[0].item()),
        )

    def is_grasping(self, object_id: str) -> bool:
        actor = self.objects[object_id].actor
        result = self.agent.is_grasping(actor)
        if bool(result[0].item()):
            return True
        left, right = self.finger_contact_force(object_id)
        # RoboCasa's Panda finger links report small but valid contact forces
        # for this lightweight cabinet prop. The lift check below is the
        # physical success criterion; this threshold only detects a closed
        # two-finger contact opportunity.
        return left >= 0.05 and right >= 0.05

    def contact_snapshot(self) -> dict[str, Any]:
        """Return post-transition contact labels for dense recording only."""
        objects = {}
        for object_id in self.objects:
            left, right = self.finger_contact_force(object_id)
            objects[object_id] = {
                "left_force": left,
                "right_force": right,
                "grasped": self.is_grasping(object_id),
            }
        return {"objects": objects}


class Primitive:
    name = "primitive"

    def __init__(self, action: PrimitiveAction):
        if action.primitive_type != self.name:
            raise ValueError(
                f"{type(self).__name__} cannot execute {action.primitive_type!r}"
            )
        self.action = action
        self.parameters = dict(action.parameters)
        self.steps = 0
        self.max_steps = 100
        self.status = "running"
        self.stage = "start"
        self.initial_object_positions: dict[str, np.ndarray] = {}
        self.max_finger_contact_force: dict[str, list[float]] = {}
        self.grasped_object_ids: set[str] = set()
        self.stable_grasp_frames: dict[str, int] = {}
        self.lifted_object_ids: set[str] = set()

    @property
    def done(self) -> bool:
        return self.status != "running"

    def reset(self, context: PrimitiveContext):
        self.steps = 0
        self.status = "running"
        self.requested_goals = {}
        self.executed_goals = {}
        self.initial_object_positions = {
            object_id: state.position_root.copy()
            for object_id, state in context.objects.items()
        }
        self.max_finger_contact_force = {
            object_id: [0.0, 0.0] for object_id in context.objects
        }
        self.grasped_object_ids = set()
        self.stable_grasp_frames = {object_id: 0 for object_id in context.objects}
        self.lifted_object_ids = set()
        self.initial_object_positions_world = {
            object_id: state.position_world.copy()
            for object_id, state in context.objects.items()
            if hasattr(state, "position_world")
        }

    def _set_goal(self, context, name: str):
        requested = np.asarray(getattr(self, name), dtype=np.float32).copy()
        project = getattr(context, "project_goal", None)
        executed = project(requested) if project is not None else requested.copy()
        setattr(self, name, executed)
        self.requested_goals[name] = requested.tolist()
        self.executed_goals[name] = executed.tolist()

    def observe(self, context: PrimitiveContext) -> dict[str, Any]:
        # Object state is never used to choose or control an action.  It is read
        # here solely to label the interaction outcome after execution.
        snapshot = {"objects": {}}
        for object_id in context.objects:
            left, right = context.finger_contact_force(object_id)
            force = self.max_finger_contact_force.setdefault(object_id, [0.0, 0.0])
            force[0] = max(force[0], left)
            force[1] = max(force[1], right)
            grasped = context.is_grasping(object_id)
            if grasped:
                self.stable_grasp_frames[object_id] = self.stable_grasp_frames.get(object_id, 0) + 1
                # A transient impulse while pushing is not a grasp. Require
                # several consecutive contact observations before accepting it.
                if self.stable_grasp_frames[object_id] >= 4:
                    self.grasped_object_ids.add(object_id)
            else:
                self.stable_grasp_frames[object_id] = 0
            snapshot["objects"][object_id] = {
                "left_force": left,
                "right_force": right,
                "grasped": grasped,
            }
        return snapshot

    def _tick(self) -> bool:
        self.steps += 1
        if self.steps > self.max_steps:
            self.status = "timeout"
            return False
        return True

    def _move_to(
        self,
        context: PrimitiveContext,
        goal: np.ndarray,
        gripper: float,
        tolerance: float = 0.012,
        speed: float = 0.25,
    ) -> tuple[np.ndarray, bool]:
        error = np.asarray(goal, dtype=np.float32) - context.ee_position
        distance = float(np.linalg.norm(error))
        if distance <= tolerance:
            return np.array([0.0, 0.0, 0.0, gripper], dtype=np.float32), True
        normalized = error / 0.10
        norm = float(np.linalg.norm(normalized))
        if norm > speed:
            normalized *= speed / norm
        action = np.array([*normalized, gripper], dtype=np.float32)
        return np.clip(action, -1.0, 1.0), False

    @staticmethod
    def _hold_gripper(context: PrimitiveContext) -> float:
        return float(np.clip(2.0 * context.gripper_open - 1.0, -1.0, 1.0))

    def next_action(self, context: PrimitiveContext) -> np.ndarray | None:
        raise NotImplementedError

    def result(self, context: PrimitiveContext) -> dict[str, Any]:
        translations = {}
        for object_id, initial in self.initial_object_positions.items():
            if object_id in context.objects:
                translations[object_id] = float(
                    np.linalg.norm(context.objects[object_id].position_root - initial)
                )
        contacted = [
            object_id
            for object_id, force in self.max_finger_contact_force.items()
            if max(force) > 1e-4
        ]
        held = {
            object_id
            for object_id in context.objects
            if self.stable_grasp_frames.get(object_id, 0) >= 4
            and context.is_grasping(object_id)
        }
        self.lifted_object_ids = set()
        for object_id, initial in self.initial_object_positions.items():
            state = context.objects.get(object_id)
            if state is None or object_id not in held:
                continue
            if float(state.position_root[2] - initial[2]) >= 0.01:
                self.lifted_object_ids.add(object_id)
        return {
            "primitive_type": self.name,
            "parameters": self.parameters,
            "requested_goals": self.requested_goals,
            "executed_goals": self.executed_goals,
            "changed_goal_count": sum(
                self.requested_goals[name] != goal
                for name, goal in self.executed_goals.items()
            ),
            "initial_object_positions_root": {
                key: value.tolist() for key, value in self.initial_object_positions.items()
            },
            "final_object_positions_root": {
                key: state.position_root.tolist() for key, state in context.objects.items()
            },
            "initial_object_positions_world": {
                key: value.tolist() for key, value in self.initial_object_positions_world.items()
            },
            "final_object_positions_world": {
                key: state.position_world.tolist() for key, state in context.objects.items()
                if hasattr(state, "position_world")
            },
            "high_level_action": self.action.to_vector().tolist(),
            "final_tcp_position": context.ee_position.tolist(),
            "duration_steps": self.steps,
            "status": self.status,
            "final_stage": self.stage,
            "object_translations": translations,
            "moved_object_ids": [
                object_id
                for object_id, translation in translations.items()
                if translation >= MOVED_OBJECT_THRESHOLD
            ],
            "contacted_object_ids": contacted,
            "max_finger_contact_force": {
                object_id: list(self.max_finger_contact_force[object_id])
                for object_id in contacted
            },
            "grasped_object_ids": sorted(self.grasped_object_ids),
            "held_object_ids": sorted(held),
            "lifted_object_ids": sorted(self.lifted_object_ids),
            "stable_grasp_frames": dict(self.stable_grasp_frames),
        }


class MoveDeltaPrimitive(Primitive):
    name = "move_delta_ee_pose"

    def reset(self, context):
        super().reset(context)
        self.goal = context.ee_position + np.array(
            [self.parameters["dx"], self.parameters["dy"], self.parameters["dz"]],
            dtype=np.float32,
        )
        self._set_goal(context, "goal")
        self.max_steps = 100
        self.stage = "move"

    def next_action(self, context):
        if not self._tick():
            return None
        action, reached = self._move_to(context, self.goal, self._hold_gripper(context))
        if reached:
            self.status = "success"
            return None
        return action


class LiftPrimitive(Primitive):
    name = "lift"

    def reset(self, context):
        super().reset(context)
        # Directional distance parameters are magnitudes.  The normalized
        # action bank remains symmetric for official-RAPS compatibility, so
        # negative samples represent the same requested magnitude.
        self.goal = context.ee_position + np.array(
            [0.0, 0.0, abs(self.parameters["z_dist"])], dtype=np.float32
        )
        self._set_goal(context, "goal")
        self.max_steps = 60
        self.stage = "lift"

    def next_action(self, context):
        if not self._tick():
            return None
        action, reached = self._move_to(
            context, self.goal, self._hold_gripper(context)
        )
        if reached:
            self.status = "success"
            return None
        return action


class DropPrimitive(Primitive):
    name = "drop"

    def reset(self, context):
        super().reset(context)
        # See LiftPrimitive: z_dist is interpreted as a magnitude.
        self.goal = context.ee_position + np.array(
            [0.0, 0.0, -abs(self.parameters["z_dist"])], dtype=np.float32
        )
        self._set_goal(context, "goal")
        self.max_steps = 60
        self.stage = "drop"

    def next_action(self, context):
        if not self._tick():
            return None
        action, reached = self._move_to(
            context, self.goal, self._hold_gripper(context)
        )
        if reached:
            self.status = "success"
            return None
        return action


class GripperPrimitive(Primitive):
    command_sign = 1.0

    def reset(self, context):
        super().reset(context)
        self.amount = 1.0
        self.max_steps = 12
        self.stage = self.name

    def next_action(self, context):
        if not self._tick():
            return None
        if self.steps >= 10:
            self.status = "success"
            return None
        return np.array(
            [0.0, 0.0, 0.0, self.command_sign * self.amount], dtype=np.float32
        )


class OpenGripperPrimitive(GripperPrimitive):
    name = "open_gripper"
    command_sign = 1.0


class CloseGripperPrimitive(GripperPrimitive):
    name = "close_gripper"
    command_sign = -1.0


class TopGraspPrimitive(Primitive):
    name = "top_grasp"

    def reset(self, context):
        super().reset(context)
        # RAPS-style goal-directed grasp for horizontal cabinet approach.
        # The approach direction is always robot-base +x (fixed by cabinet
        # geometry).  PPO learns y_offset and z_offset from the observation
        # vectors tcp_to_mug / tcp_to_bowl to aim at the desired object.
        #
        # Stage 1 "align"  : move EE to (ee_x, ee_y + y_offset, ee_z + z_offset)
        # Stage 2 "approach": advance +x by x_forward until contact
        # Stage 3 "close"  : close gripper 30 steps
        ee = context.ee_position
        y_offset = float(self.parameters["y_offset"])
        z_offset = float(self.parameters["z_offset"])
        x_forward = abs(float(self.parameters["x_forward"]))

        self.align_goal = np.array(
            [ee[0], ee[1] + y_offset, ee[2] + z_offset], dtype=np.float32
        )
        self.grasp_goal = np.array(
            [ee[0] + x_forward, ee[1] + y_offset, ee[2] + z_offset],
            dtype=np.float32,
        )
        self._set_goal(context, "align_goal")
        self._set_goal(context, "grasp_goal")
        self.max_steps = 150
        self.close_steps = 0
        self.stage = "align"

    def observe(self, context):
        snapshot = super().observe(context)
        if self.stage == "approach":
            # Stop as soon as both fingertips have a light contact. Waiting
            # for both sides avoids closing on a one-sided collision while
            # preventing the high-force push seen with the old threshold.
            for state in snapshot["objects"].values():
                if float(state["left_force"]) >= 0.05 and float(state["right_force"]) >= 0.05:
                    self.stage = "close"
                    break
        return snapshot

    def next_action(self, context):
        if not self._tick():
            return None
        if self.stage == "align":
            action, reached = self._move_to(
                context, self.align_goal, 1.0, tolerance=0.015, speed=0.30
            )
            if reached:
                self.stage = "approach"
            return action
        if self.stage == "approach":
            action, reached = self._move_to(
                context, self.grasp_goal, 1.0, tolerance=0.012, speed=0.08
            )
            if reached:
                self.stage = "close"
            return action
        # stage == "close"
        self.close_steps += 1
        if self.close_steps >= 30:
            held = any(
                frames >= 4 and context.is_grasping(object_id)
                for object_id, frames in self.stable_grasp_frames.items()
            )
            self.status = "success" if held else "no_stable_grasp"
            return None
        return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)


class DirectionalMovePrimitive(Primitive):
    direction = np.zeros(3, dtype=np.float32)
    parameter_name = "distance"

    def reset(self, context):
        super().reset(context)
        # Direction is selected by the primitive type; the scalar parameter is
        # a magnitude even though the shared normalized bank spans [-1, 1].
        self.goal = context.ee_position + self.direction * abs(self.parameters[self.parameter_name])
        self._set_goal(context, "goal")
        self.max_steps = 70
        self.stage = "move"

    def next_action(self, context):
        if not self._tick():
            return None
        action, reached = self._move_to(context, self.goal, self._hold_gripper(context))
        if reached:
            self.status = "success"
            return None
        return action


class MoveLeftPrimitive(DirectionalMovePrimitive):
    name = "move_left"
    direction = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    parameter_name = "y_dist"


class MoveRightPrimitive(DirectionalMovePrimitive):
    name = "move_right"
    direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    parameter_name = "y_dist"


class MoveForwardPrimitive(DirectionalMovePrimitive):
    name = "move_forward"
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    parameter_name = "x_dist"


class MoveBackwardPrimitive(DirectionalMovePrimitive):
    name = "move_backward"
    direction = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    parameter_name = "x_dist"


PRIMITIVE_CLASSES = {
    "move_delta_ee_pose": MoveDeltaPrimitive,
    "lift": LiftPrimitive,
    "drop": DropPrimitive,
    "open_gripper": OpenGripperPrimitive,
    "close_gripper": CloseGripperPrimitive,
    "top_grasp": TopGraspPrimitive,
    "move_left": MoveLeftPrimitive,
    "move_right": MoveRightPrimitive,
    "move_forward": MoveForwardPrimitive,
    "move_backward": MoveBackwardPrimitive,
}


class UniformPrimitiveExplorer:
    """Weighted RAPS sampler with no object-dependent action generation."""

    def __init__(self, seed=0, weights=None, verbose=True):
        self.rng = np.random.default_rng(seed)
        self.verbose = bool(verbose)
        self.weights = weights or {
            "move_delta_ee_pose": 0.15,
            "top_grasp": 0.10,
            "lift": 0.12,
            "drop": 0.08,
            "move_left": 0.10,
            "move_right": 0.10,
            "move_forward": 0.10,
            "move_backward": 0.10,
            "open_gripper": 0.05,
            "close_gripper": 0.10,
        }
        unknown = set(self.weights) - set(PRIMITIVE_CLASSES)
        if unknown:
            raise ValueError(
                f"Unknown primitive weights {sorted(unknown)}; "
                f"available={list(PRIMITIVE_CLASSES)}"
            )
        if not any(float(weight) > 0 for weight in self.weights.values()):
            raise ValueError("At least one primitive weight must be positive")
        self.current: Primitive | None = None
        self.completed: list[dict[str, Any]] = []
        self.primitive_index = 0

    def reset(self):
        self.current = None
        self.completed.clear()
        self.primitive_index = 0

    def _sample_action(self) -> PrimitiveAction:
        names = [name for name, weight in self.weights.items() if weight > 0]
        probabilities = np.asarray([self.weights[name] for name in names], dtype=float)
        probabilities /= probabilities.sum()
        primitive_type = str(self.rng.choice(names, p=probabilities))
        parameters = RAPS_ACTION_SCHEMA.sample_parameters(primitive_type, self.rng)
        return PrimitiveAction(primitive_type, parameters)

    def sample_high_level_action(self) -> np.ndarray:
        """Sample one RAPS action without starting a primitive rollout."""
        return self._sample_action().to_vector()

    def sample_candidates(self, count: int) -> np.ndarray:
        """Sample a batch of high-level actions for active selection."""
        if int(count) < 1:
            raise ValueError("count must be positive")
        return np.stack(
            [self.sample_high_level_action() for _ in range(int(count))], axis=0
        ).astype(np.float32)

    def _start_next(self, context: PrimitiveContext):
        action = self._sample_action()
        primitive_cls = PRIMITIVE_CLASSES[action.primitive_type]
        self.current = primitive_cls(action)
        self.current.reset(context)
        self.primitive_index += 1
        if self.verbose:
            print(
                f"[Primitive {self.primitive_index}] start "
                f"{action.primitive_type} params={action.parameters}"
            )

    def next_action(self, env) -> np.ndarray:
        for _ in range(3):
            context = PrimitiveContext(env)
            if self.current is None:
                self._start_next(context)
            self.current.observe(context)
            action = self.current.next_action(context)
            if action is not None:
                return action
            result = self.current.result(context)
            self.completed.append(result)
            if self.verbose:
                print(f"[Primitive {self.primitive_index}] done {result}")
            self.current = None
        return np.zeros(4, dtype=np.float32)
