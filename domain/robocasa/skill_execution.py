"""Shared state for executing RAPS primitives against real observations."""

import numpy as np

from domain.robocasa.primitives import PRIMITIVE_CLASSES, PrimitiveAction, OpenGripperPrimitive


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
        if action.primitive_type in {"approach", "approach_x"}:
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
                         and action.parameters["dz"] > 0)
                     or (action.primitive_type == "align_yz"
                         and action.parameters["z_offset"] > 0))):
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
                {"approach", "align_yz", "approach_x", "open_gripper", "close_gripper"}):
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


class PlannedSkillExecution(RAPSSkillExecution):
    """Execute primitive goals through the shared checked native joint planner."""

    def __init__(self, *, orientation_target, planning_seed=0, max_control_steps=400, **kwargs):
        if not kwargs.get("latched_gripper", False):
            raise ValueError("Planned primitives require latched_gripper: true")
        if isinstance(max_control_steps, bool) or int(max_control_steps) != max_control_steps or max_control_steps < 1:
            raise ValueError("motion_plan_max_steps must be a positive integer")
        self.orientation_target = orientation_target
        self.planning_seed = int(planning_seed)
        self.max_control_steps = int(max_control_steps)
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.plan_counter = 0
        self.motion_cursor = None
        self.motion_attempts = []
        self._planned_stage = None
        self._hold_qpos = None
        self._active_motion_step = False
        self._force_ticks = 0
        self.last_motion_tick = {}
        self._opening_primitive = None

    def start(self, action, context):
        super().start(action, context)
        self.motion_cursor = None
        self.motion_attempts = []
        self._planned_stage = None
        self._hold_qpos = None
        self._active_motion_step = False
        self._force_ticks = 0
        self.last_motion_tick = {}
        self.raw = context.env
        # Initialize while the retained EE kinematics are available, before closing.
        self.raw.motion_planner
        self._opening_primitive = None
        if action.primitive_type == "approach":
            # The legacy composite opens during alignment. Checked joint plans
            # require fixed finger geometry, so finish that opening first.
            self._opening_primitive = OpenGripperPrimitive(PrimitiveAction("open_gripper", {}))
            self._opening_primitive.reset(context)
        if action.primitive_type not in {"open_gripper", "close_gripper"}:
            self.primitive.max_steps = self.max_control_steps

    def finish_motion(self, status):
        if self.motion_cursor is not None and self.motion_cursor.status == "running":
            self.motion_cursor.stop(status)
            self.motion_attempts[-1]["status"] = status

    def _hold_joints(self, context):
        self.finish_motion("contact_stop")
        self._active_motion_step = False
        if self._hold_qpos is None:
            self._hold_qpos = self.raw.agent.robot.get_qpos().detach().cpu().numpy().reshape(9)[:7].copy()
        low = super().next_action(context)
        if low is None:
            return None
        return np.r_[self._hold_qpos, low[3]].astype(np.float32)

    def next_action(self, context):
        from domain.robocasa.motion_planning import MotionPlanExecution, MotionPlanningError

        if self.primitive is None or self.primitive.done:
            self.finish_motion(self.primitive.status if self.primitive is not None else "interrupted")
            return None
        if self.executed_steps >= self.max_control_steps:
            if (self.motion_cursor is not None and self.motion_cursor.status == "success"
                    and self.primitive.name != "approach"):
                self.primitive.status = "success"
            elif (self.primitive.name in {"open_gripper", "close_gripper"}
                  or self.primitive.stage == "close") and self.primitive.next_action(context) is None:
                return None
            else:
                self.finish_motion("budget_exhausted")
                self.primitive.status = "budget_exhausted"
            return None
        context.gripper_command = self.gripper_command
        if self._opening_primitive is not None:
            low = self._opening_primitive.next_action(context)
            if low is not None:
                if self._hold_qpos is None:
                    self._hold_qpos = self.raw.agent.robot.get_qpos().detach().cpu().numpy().reshape(9)[:7].copy()
                self._active_motion_step = False
                self.primitive.steps += 1
                return np.r_[self._hold_qpos, low[3]].astype(np.float32)
            self._opening_primitive = None
            self._hold_qpos = None
        if (self.primitive.name in {"open_gripper", "close_gripper"}
                or self.primitive.stage in {"close", "settle_support"}):
            return self._hold_joints(context)
        if self.motion_cursor is not None and self.motion_cursor.status != "running":
            if self.motion_cursor.status != "success":
                self.primitive.status = self.motion_cursor.status
                return None
            if self.primitive.name == "approach":
                self.primitive.stage = "approach" if self._planned_stage == "align" else "close"
                self.motion_cursor = None
                return self.next_action(context)
            self.primitive.status = "success"
            return None
        if self.motion_cursor is None:
            if self.primitive.name == "approach":
                goal = self.primitive.align_goal if self.primitive.stage == "align" else self.primitive.grasp_goal
            else:
                goal = self.primitive.goal
            quaternion = self.orientation_hold
            if quaternion is None:
                quaternion = self.orientation_target()
            self._planned_stage = self.primitive.stage
            attempt = dict(stage=self._planned_stage, target_position_root=np.asarray(goal).tolist(),
                           target_quaternion_wxyz=np.asarray(quaternion).tolist(),
                           seed=self.planning_seed + self.plan_counter, control_steps=0)
            self.plan_counter += 1
            self.motion_attempts.append(attempt)
            try:
                plan = self.raw.plan_to_pose(goal, quaternion, seed=attempt["seed"],
                    allowed_contact_objects=tuple(context.objects))
                self.motion_cursor = MotionPlanExecution(self.raw, plan,
                    gripper_command=self.gripper_command,
                    max_steps=self.max_control_steps - self.executed_steps)
                attempt.update(status="running", planning=plan.metadata)
            except MotionPlanningError as exc:
                attempt.update(status=exc.status, detail=exc.detail)
                self.primitive.status = exc.status
                return None
        self._active_motion_step = True
        self.primitive.steps += 1
        return self.motion_cursor.next_action()

    def record_action(self, actual_action, *, terminated=False, truncated=False):
        from domain.robocasa.motion_planning import cabinet_contact_forces

        action = np.asarray(actual_action, dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Planned primitives require actual 8D joint-position controls")
        self.gripper_command = float(action[-1])
        self.executed_steps += 1
        if self._active_motion_step:
            expected = self.motion_cursor.next_action()
            if not np.array_equal(action, expected):
                raise RuntimeError("Executed joint target differs from checked plan")
            self.last_motion_tick = self.motion_cursor.record_step(terminated=terminated, truncated=truncated)
            self.motion_attempts[-1].update(status=self.motion_cursor.status,
                control_steps=len(self.motion_cursor.records),
                endpoint_position_error_m=self.motion_cursor.endpoint_position_error_m,
                endpoint_angle_error_deg=self.motion_cursor.endpoint_angle_error_deg)
            if self.motion_cursor.status not in {"running", "success"}:
                self.primitive.status = self.motion_cursor.status
        else:
            forces = cabinet_contact_forces(self.raw, self.raw.motion_planner.attachment_state())
            peak = max(forces.values(), default=0.)
            self._force_ticks = self._force_ticks + 1 if peak > 20 else 0
            self.last_motion_tick = dict(stage=self.primitive.stage, control_mode="pd_joint_pos",
                                         cabinet_forces_N=forces)
            if terminated or truncated:
                self.primitive.status = "terminated" if terminated else "truncated"
            elif self._force_ticks >= 3:
                self.primitive.status = "tracking_blocked"

    def observe(self, context):
        if self.primitive.done:
            return self.primitive.observe(context)
        return super().observe(context)
