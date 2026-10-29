"""Shared state for executing RAPS primitives against real observations."""

import numpy as np

from domain.robocasa.primitives import PRIMITIVE_CLASSES, PrimitiveAction


class RAPSSkillExecution:
    def __init__(self, latched_gripper=False, stop_drop_on_support=False,
                 hold_closed_during_motion=False, grasp_stop_on_contact=True,
                 controlled_placement=False):
        self.latched_gripper = bool(latched_gripper)
        self.stop_drop_on_support = bool(stop_drop_on_support)
        self.hold_closed_during_motion = bool(hold_closed_during_motion)
        self.grasp_stop_on_contact = bool(grasp_stop_on_contact)
        self.controlled_placement = bool(controlled_placement)
        if self.controlled_placement and not (
                self.latched_gripper and self.stop_drop_on_support):
            raise ValueError("controlled_placement requires latched_gripper "
                             "and stop_drop_on_support")
        self.reset()

    def reset(self):
        self.gripper_command = 1.0
        self.primitive = None
        self.support_stopped_object_ids = []
        self._drop_object_ids = []
        self.executed_steps = 0
        self.settle_count = 0
        self.settle_goal = None
        self._hold_goal = None
        self._settle_start_step = None
        self._previous_observed_position = None
        self._previous_observed_step = None
        self.orientation_hold = None

    def start(self, action: PrimitiveAction, context):
        self.primitive = PRIMITIVE_CLASSES[action.primitive_type](action)
        if action.primitive_type == "top_grasp":
            self.primitive.stop_on_contact = self.grasp_stop_on_contact
        if action.primitive_type == "drop":
            self.primitive.controlled_placement = self.controlled_placement
        self.primitive.reset(context)
        self.support_stopped_object_ids = []
        self._drop_object_ids = []
        self.executed_steps = 0
        self.settle_count = 0
        self.settle_goal = None
        self._hold_goal = None
        self._settle_start_step = None
        self._previous_observed_position = None
        self._previous_observed_step = None
        self.orientation_hold = None
        if self.controlled_placement and action.primitive_type in {
                "open_gripper", "close_gripper"}:
            self._hold_goal = np.asarray(context.ee_position, dtype=np.float32).copy()
            self.orientation_hold = self._current_orientation(context)
        if (self.controlled_placement and context.ee_position[2] < .15
                and (action.primitive_type == "lift"
                     or (action.primitive_type == "move_delta_ee_pose"
                         and action.parameters["dz"] > 0))):
            self.orientation_hold = self._current_orientation(context)
        if self.stop_drop_on_support and action.primitive_type == "drop":
            self._drop_object_ids = [
                object_id for object_id in context.objects
                if context.is_grasping(object_id)
            ]

    def next_action(self, context):
        if self.primitive is None or self.primitive.done:
            return None
        if self.latched_gripper:
            context.gripper_command = self.gripper_command
        if self.controlled_placement and self.primitive.stage == "settle_support":
            if (self.executed_steps - self._settle_start_step >= 20
                    or not self.primitive._tick()):
                self.primitive.status = "support_unstable"
                return None
            return self._hold_action(context, self.settle_goal, self.gripper_command)
        action = self.primitive.next_action(context)
        if action is not None and self._hold_goal is not None:
            action = self._hold_action(context, self._hold_goal, action[3])
        if (action is not None and not self.latched_gripper
                and self.hold_closed_during_motion
                and self.primitive.action.primitive_type not in
                {"top_grasp", "open_gripper", "close_gripper"}):
            action[3] = -1.0
        return action

    @staticmethod
    def _current_orientation(context):
        quaternion = context.arm_controller.ee_pose_at_base.q
        if hasattr(quaternion, "detach"):
            quaternion = quaternion.detach().cpu().numpy()
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if quaternion.shape == (1, 4):
            quaternion = quaternion[0]
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("Orientation hold requires a finite wxyz quaternion "
                             "with shape (4,) or (1, 4)")
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Orientation hold requires a nonzero finite quaternion norm")
        return (quaternion / norm).astype(np.float32)

    @staticmethod
    def _hold_action(context, goal, gripper):
        motion = (goal - context.ee_position) / .10
        norm = float(np.linalg.norm(motion))
        if norm > .25:
            motion = motion * (.25 / norm)
        return np.array([*motion, gripper], dtype=np.float32)

    def record_action(self, actual_action):
        action = np.asarray(actual_action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Executed RAPS action must be a finite shape-(4,) array")
        if self.latched_gripper:
            self.gripper_command = float(action[3])
        self.executed_steps += 1

    def observe(self, context):
        if self.primitive is None:
            raise RuntimeError("Start a RAPS primitive before observing execution")
        snapshot = self.primitive.observe(context)
        if not (self.stop_drop_on_support
                and self.primitive.action.primitive_type == "drop"):
            return snapshot
        ground = getattr(context.env, "ground", None)
        if ground is None:
            raise RuntimeError("Drop support stopping requires context.env.ground")
        supported = []
        for object_id in self._drop_object_ids:
            force = context.env.scene.get_pairwise_contact_forces(
                context.objects[object_id].actor, ground
            )
            if hasattr(force, "detach"):
                force = force.detach().cpu().numpy()
            try:
                force = np.asarray(force, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ground contact force for {object_id!r}") from exc
            if (force.ndim < 1 or force.shape[-1] != 3 or force.size == 0
                    or not np.isfinite(force).all()):
                raise ValueError(f"Invalid ground contact force for {object_id!r}: "
                                 "expected finite (..., 3) vectors")
            if np.any(force[..., 2] > 0.05):
                supported.append(object_id)
        if self.controlled_placement:
            if self.primitive.stage == "settle_support":
                if self.primitive.done:
                    return snapshot
                elapsed = self.executed_steps - self._previous_observed_step
                if elapsed > 0:
                    position = np.asarray(context.ee_position, dtype=np.float32)
                    speed = float(np.linalg.norm(
                        position - self._previous_observed_position)) / (elapsed * .05)
                    anchor_error = float(np.linalg.norm(position - self.settle_goal))
                    stable = bool(supported) and anchor_error <= .003 and speed <= .01
                    self.settle_count = self.settle_count + 1 if stable else 0
                    self._previous_observed_position = position.copy()
                    self._previous_observed_step = self.executed_steps
                    if self.settle_count >= 4:
                        self.primitive.status = "supported"
                        self.support_stopped_object_ids = sorted(supported)
                if (not self.primitive.done
                        and self.executed_steps - self._settle_start_step >= 20):
                    self.primitive.status = "support_unstable"
            elif supported:
                self.orientation_hold = self._current_orientation(context)
                self.primitive.stage = "settle_support"
                self.primitive.status = "running"
                self.settle_goal = np.asarray(context.ee_position, dtype=np.float32).copy()
                self._settle_start_step = self.executed_steps
                self._previous_observed_step = self.executed_steps
                self._previous_observed_position = self.settle_goal.copy()
                self.settle_count = 0
        elif supported:
            self.primitive.status = "supported"
            self.support_stopped_object_ids = sorted(supported)
        return snapshot
