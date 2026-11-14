"""Opt-in TCP grasp and object-target placement for the fixed cabinet scene."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from domain.robocasa.primitives import (
    Primitive, PrimitiveAction, PrimitiveParameter, PrimitiveSpec, RAPSActionSchema,
)
from domain.robocasa.skill_execution import RAPSSkillExecution

CABINET_ORIGIN = (.525, .12, .66)
XY_MARGIN = .08
Z_MARGIN = .02
UPPER_SUPPORT_Z = .672
LOWER_SUPPORT_Z = .330
OUTSIDE_X = .18
CABINET_MOUTH_X = .29
INTERMEDIATE = (.18, 0., .84)
PREPLACE = (.28, .15, .56)
TCP_BOUNDS = {
    "grasp": ((.370, .671), (-.070, .310), (.692, .872)),
}
PLACE_XY_BOUNDS = ((.370, .671), (-.070, .310))
ACTION_BOUNDS = {**TCP_BOUNDS, "place": PLACE_XY_BOUNDS}
PLACE_CLEARANCE = .04
PLACE_EDGE_CLEARANCE = .02
PHASE_BUDGETS = {
    "open": 9, "retreat": 85, "align": 100, "approach": 120, "preplace": 100, "clear": 100, "home_raise": 100,
    "close": 30, "confirm": 12, "lift": 45, "withdraw": 85,
    "intermediate": 85, "descend": 110, "settle": 20,
    "release": 9, "home_align": 100, "home": 85, "invalid": 1,
}
PICK_PLACE_MACRO_CONTRACT = {
    "grasp_retention": {
        "closed_until": "four_consecutive_settled_lower_shelf_support_controls",
        "feedback": "after_every_real_control_including_final_budget_tick",
        "attachment_reference": "confirmed_grasp_until_supported_release; not_reset_by_replanning",
        "maximum_relative_position_slip_m": .002, "maximum_relative_angle_slip_deg": 2.,
        "on_loss_or_slip": "fail_current_macro_stop_motion_keep_closed",
        "physical_no_drop_guarantee": False, "teleport_or_physics_attachment": False,
    },
    "version": 5, "coordinates": "controller_root_meters; grasp_tcp_xyz; place_object_collision_aabb_center_xy",
    "orientation": "fixed_grasp_place; exterior_preplace_transfer_may_rotate",
    "exterior_rrt_pre_TOPP_max_joint_step_rad": .025,
    "orientation_rotation_region": {"stage": "preplace", "tcp_root_x_less_than_m": CABINET_MOUTH_X}, "action_dim": 5,
    "cabinet_origin_root_m": list(CABINET_ORIGIN),
    "cabinet_inside_local_xy_m": [[-.235, .226], [-.27, .27]],
    "common_xy_margin_m": XY_MARGIN, "grasp_z_endpoint_margin_m": Z_MARGIN,
    "grasp_nominal_support_relative_z_band_m": [0., .22],
    "tcp_bounds_root_m": {name: [list(axis) for axis in bounds]
                          for name, bounds in TCP_BOUNDS.items()},
    "place_object_xy_bounds_root_m": [list(axis) for axis in PLACE_XY_BOUNDS],
    "place_reference": "collision_geometry_root_axis_aligned_bounding_box_center_at_fixed_place_orientation",
    "place_geometry": "measured_confirmed_tcp_to_object_pose; preserve_across_grasp_and_place",
    "place_entry_clearance_m": PLACE_CLEARANCE,
    "place_full_footprint_edge_clearance_m": PLACE_EDGE_CLEARANCE,
    "upper_support_root_z_m": UPPER_SUPPORT_Z,
    "lower_support_root_z_m": LOWER_SUPPORT_Z,
    "cabinet_mouth_root_x_m": CABINET_MOUTH_X,
    "outside_root_x_m": OUTSIDE_X, "intermediate_root_m": list(INTERMEDIATE),
    "preplace_root_m": list(PREPLACE),
    "home": "first_start_tcp_after_episode_reset",
    "place_z_semantics": "automatic_from_payload_bottom_and_lower_shelf; release_requires_settled_lower_shelf_support",
    "phase_sequences": {
        "grasp": ["open", "retreat", "align", "approach", "close", "confirm", "lift", "withdraw", "intermediate"],
        "place": ["retreat", "preplace", "approach", "descend", "settle", "release", "clear", "withdraw", "home_raise", "home_align", "home"],
    },
    "failure_statuses": ["unsafe_release", "already_holding", "missing_grasp", "lost_grasp", "insufficient_lift", "no_support", "release_failed", "timeout",
        "planning_failed", "path_collision", "no_collision_free_ik", "time_parameterization_failed",
        "cartesian_path_failed", "orientation_projection_failed", "orientation_path_violation", "orientation_tracking_error",
        "descent_path_violation", "tracking_blocked", "tracking_error", "budget_exhausted",
        "attachment_slip", "grasp_lost", "unexpected_attachment", "unconfirmed_grasp",
        "multiple_grasped_objects", "release_in_progress", "gripper_plan_mismatch", "stale_plan",
        "invalid_support_surface", "invalid_goal", "execution_error", "unsupported", "terminated", "truncated"],
    "release_clearance_m": .04,
    "lift_m": .08, "lift_root_z_cap_m": .90, "minimum_object_rise_m": .01,
    "motion_tolerance_m": .012, "approach_speed_normalized": .08,
    "transport_speed_normalized": .25, "descent_speed_normalized": .025,
    "ee_delta_speed_usage": "internal geometry executor only; unused by planned execution, which uses joint TOPP timing",
    "descent_floor_root_z_m": .330, "support_geometry_z_tolerance_m": .015,
    "support_min_vertical_force_N": .05, "support_max_object_speed_m_s": .01,
    "grasp_confirmation_controls": 4, "support_settle_controls": 4,
    "max_controls": 2000, "phase_control_budgets": {
        name: (1000 if name == "descend" else budget if name in {"open", "close", "confirm", "settle", "release", "invalid"} else 400)
        for name, budget in PHASE_BUDGETS.items()},
    "motion_execution": "planner", "control_mode": "pd_joint_pos",
    "grasp_motion_planner": "approach_lift: bounded_Cartesian_IK_TOPP; max_waypoint.005m",
    "grasp_approach_contact": "stop_on_bilateral_finger_contact_then_close_and_confirm",
    "descent_planner": "bounded_pose_IK_then_TOPP; translation_m_rotation_rad; diff_step.0001; max_nfev200",
    "descent_cartesian_step_m": .005, "descent_cartesian_corridor_m": .002,
    "descent_waypoint_tolerance_m": .0001,
    "path_orientation_tolerance_deg": .5, "fixed_orientation_ik_starts": 16,
    "planned_segment_budget": "motion_plan_max_steps; default400",
    "descent_gap_safety_overlap_m": .0005,
    "support_collision_allowance": "payload_only_matching_lower_shelf_top; max_penetration.001m",
}


def pick_place_contract(grasp_retention="friction"):
    if grasp_retention not in {"friction", "constraint"}:
        raise ValueError("grasp_retention must be friction or constraint")
    contract = deepcopy(PICK_PLACE_MACRO_CONTRACT)
    if grasp_retention == "constraint":
        retention = contract["grasp_retention"]
        retention.pop("teleport_or_physics_attachment")
        retention.update(mode="constraint", physics_attachment=True, teleport=False,
                         backend="SAPIEN_PhysxDrive_linear_locks_angular_slerp",
                         angular_drive=dict(stiffness=1e6, damping=2e3, mode="acceleration"),
                         support_speed="max_physics_substep_pose_displacement_per_dt_over_last_control; native_speed_also_recorded",
                         support_finger_target="hold_measured_width_on_first_support; logical_command_stays_closed",
                         binding_trigger="four_consecutive_physical_closed_grasp_controls",
                         release_trigger="opening_command_before_physics_or_reset_or_close",
                         grasp_feedback_after_binding="active_constraint_and_measured_relative_pose")
        contract["failure_statuses"] += ["grasp_binding_conflict", "grasp_binding_lost",
                                         "grasp_binding_slip", "grasp_binding_error"]
    return contract


@dataclass(frozen=True)
class BoundedPositionParameter(PrimitiveParameter):
    center: float

    def normalize(self, value):
        if not np.isfinite(value) or not self.center - self.scale <= value <= self.center + self.scale:
            raise ValueError(f"Position {self.name} must be finite and in "
                             f"[{self.center - self.scale}, {self.center + self.scale}]")
        return float((value - self.center) / self.scale)

    def denormalize(self, value):
        if not np.isfinite(value):
            raise ValueError(f"Normalized position {self.name} must be finite")
        return float(self.center + np.clip(value, -1., 1.) * self.scale)


class PickPlaceActionSchema(RAPSActionSchema):
    def encode(self, action):
        expected = {p.name for p in self.spec_by_name[action.primitive_type].parameters}
        if set(action.parameters) != expected:
            raise ValueError(f"{action.primitive_type} requires exactly {sorted(expected)}")
        return super().encode(action)

    def sample_parameters(self, primitive_type, rng):
        return {p.name: p.denormalize(float(rng.uniform(-1., 1.)))
                for p in self.spec_by_name[primitive_type].parameters}


PICK_PLACE_ACTION_SCHEMA = PickPlaceActionSchema(tuple(
    PrimitiveSpec(name, tuple(BoundedPositionParameter(axis, (hi - lo) / 2, (hi + lo) / 2)
                              for axis, (lo, hi) in zip("xyz", bounds)))
    for name, bounds in ACTION_BOUNDS.items()
))


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def lower_shelf_supported(context, object_ids, *, require_settled=True):
    """Require frame force and actual contact geometry at the lower shelf."""
    env = context.env
    if not hasattr(env, "cabinet") or not hasattr(env.scene, "get_contacts"):
        raise RuntimeError("Pick/place support requires cabinet frame and scene.get_contacts()")
    frames = [link for link in env.cabinet.get_links() if link.name == "frame"]
    if len(frames) != 1:
        raise RuntimeError("Pick/place support requires exactly one cabinet link named 'frame'")
    frame = frames[0]
    if not hasattr(frame, "_objs"):
        raise RuntimeError("Pick/place support requires native cabinet frame entities")
    root_world = _numpy(context.arm_controller.root_link.pose.to_transformation_matrix()).reshape(-1, 4, 4)[0]
    world_to_root = np.linalg.inv(root_world)
    frame_entity = frame._objs[0].entity
    contacts = env.scene.get_contacts()
    supported = []
    for object_id in object_ids:
        if object_id not in context.objects:
            continue
        actor = context.objects[object_id].actor
        force = _numpy(env.scene.get_pairwise_contact_forces(actor, frame))
        if force.size == 0 or force.shape[-1] != 3 or not np.isfinite(force).all():
            raise ValueError("Pick/place support requires finite (...,3) contact forces")
        speed = _numpy(actor.linear_velocity)
        if not np.isfinite(speed).all():
            raise ValueError("Pick/place support requires finite object linear velocity")
        stable_speed = float(np.linalg.norm(speed))
        binding = getattr(env, "_grasp_binding", None)
        if binding is not None and binding.bound_object_id == object_id:
            feedback = binding.feedback()
            stable_speed = feedback["max_substep_pose_speed_m_s"]
            if stable_speed is None or not feedback["pose_speed_samples"]:
                continue
        if not np.any(force[..., 2] > .05) or (require_settled and stable_speed > .01):
            continue
        entity = actor._objs[0]
        geometry_supported = False
        for contact in contacts:
            entities = [body.entity for body in contact.bodies]
            if not ((entities[0] == entity and entities[1] == frame_entity)
                    or (entities[1] == entity and entities[0] == frame_entity)):
                continue
            for point in contact.points:
                position = world_to_root @ np.r_[_numpy(point.position), 1.]
                if abs(position[2] - LOWER_SUPPORT_Z) <= .015:
                    geometry_supported = True
        if geometry_supported:
            supported.append(object_id)
    return sorted(supported)


class PickPlacePrimitive(Primitive):
    def __init__(self, action):
        self.name = action.primitive_type
        super().__init__(action)
        self.max_steps = 500


class PickPlaceSkillExecution(RAPSSkillExecution):
    """Bounded state machine with measured payload geometry for placement."""
    def __init__(self, **kwargs):
        kwargs["latched_gripper"] = True
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.home = None
        self._held_ids = set()
        self.max_controls = 500
        self.phase_budgets = PHASE_BUDGETS
        self._place_reference = None

    def start(self, action: PrimitiveAction, context):
        PICK_PLACE_ACTION_SCHEMA.encode(action)  # Reject invalid physical coordinates.
        if self.home is None:
            self.home = np.asarray(context.ee_position, dtype=np.float32).copy()
        self.primitive = PickPlacePrimitive(action)
        self.primitive.reset(context)
        self.primitive.max_steps = self.max_controls
        self.executed_steps = 0
        self.support_stopped_object_ids = []
        self.orientation_hold = None  # Wrapper retains the common reset orientation.
        self.goal = None
        self.place_touch_goal = None
        self._sampled_step = 0
        self._grasp_frames = {}
        self._held_ids = {key for key in context.objects if context.is_grasping(key)}
        self._loss_guard = False
        self._settle_count = 0
        self._rejection = None
        self.placement_failure = None
        if action.primitive_type == "grasp":
            self.goal = np.array([action.parameters[axis] for axis in "xyz"], dtype=np.float32)
            if self._held_ids:
                self._rejection = "already_holding"
                self._phase("invalid", context)
            else:
                self._place_reference = None
                self._phase("open", context)
        elif not self._held_ids or self.gripper_command >= 0:
            self._rejection = "missing_grasp"
            self._phase("invalid", context)
        else:
            self._loss_guard = True
            try:
                if self._place_reference is None:
                    self._capture_place_reference(context)
                if self._held_ids != {self._place_reference["name"]}:
                    raise ValueError("Placement reference does not match the held object")
                lower, upper = np.asarray(self._place_reference["bounds_tcp_offset_m"])
                if (lower.shape != (3,) or upper.shape != (3,) or
                        not np.isfinite([lower, upper]).all() or np.any(lower > upper)):
                    raise ValueError("Placement requires finite ordered collision bounds")
                offset = np.r_[(lower[:2] + upper[:2]) / 2, lower[2]]
                self.place_touch_goal = np.r_[[action.parameters[axis] for axis in "xy"], LOWER_SUPPORT_Z] - offset
                for axis, (lo, hi) in enumerate(PLACE_XY_BOUNDS):
                    if (self.place_touch_goal[axis] + lower[axis] < lo - XY_MARGIN + PLACE_EDGE_CLEARANCE or
                            self.place_touch_goal[axis] + upper[axis] > hi + XY_MARGIN - PLACE_EDGE_CLEARANCE):
                        raise ValueError("Placed payload footprint must stay at least 2 cm inside cabinet edges")
                self.goal = self.place_touch_goal.copy()
                self.goal[2] += PLACE_CLEARANCE
                self.primitive.requested_goals["object_target_xy"] = [action.parameters[axis] for axis in "xy"]
                self.primitive.requested_goals["predicted_touch_tcp"] = self.place_touch_goal.tolist()
            except (ValueError, RuntimeError) as exc:
                self._rejection = getattr(exc, "status", "invalid_goal")
                self.placement_failure = str(exc)
                self._phase("invalid", context)
                return
            self._phase("retreat", context)

    def _capture_place_reference(self, context):
        planner = context.env.motion_planner
        attachment = planner.attachment_state()
        if attachment is None or self._held_ids != {attachment["name"]}:
            raise ValueError("Placement requires one confirmed payload and its measured TCP-relative pose")
        orientation = (self._orientation if hasattr(self, "_orientation") else
                       _numpy(context.arm_controller.ee_pose_at_base.q).reshape(-1, 4)[0])
        bounds = planner.placement_bounds(attachment, orientation)
        self._place_reference = dict(name=attachment["name"],
            tcp_to_object_pose=deepcopy(attachment["tcp_to_object_pose"]),
            bounds_tcp_offset_m=np.asarray(bounds).tolist())

    def _phase(self, stage, context):
        self.primitive.stage = stage
        self.primitive.phase = stage
        self._phase_start = self.executed_steps
        self._anchor = np.asarray(context.ee_position, dtype=np.float32).copy()
        if stage == "settle":
            binding = getattr(context.env, "_grasp_binding", None)
            if binding is not None and binding.bound_object_id is not None:
                binding.hold_fingers_at_current_width()
        if stage in {"retreat", "withdraw"}:
            self._anchor[0] = min(OUTSIDE_X, self._anchor[0])
            if stage == "withdraw" and self.primitive.name == "place":
                self._anchor = np.array(PREPLACE, dtype=np.float32)
        elif stage == "align":
            self._anchor = np.array([OUTSIDE_X, *self.goal[1:]], dtype=np.float32)
        elif stage == "clear":
            self._anchor[2] = max(float(self.goal[2]), float(self._anchor[2]) + .04)
        elif stage == "preplace":
            self._anchor = np.array(PREPLACE, dtype=np.float32)
        elif stage == "approach":
            self._anchor = self.goal.copy()
        elif stage == "lift":
            self._anchor[2] = min(.90, self._anchor[2] + .08)
        elif stage == "intermediate":
            self._anchor = np.array(INTERMEDIATE, dtype=np.float32)
        elif stage == "home_raise":
            self._anchor[0] = OUTSIDE_X
            self._anchor[2] = self.home[2]
        elif stage == "home_align":
            self._anchor = self.home.copy()
            self._anchor[0] = min(OUTSIDE_X, self.home[0])
        elif stage == "home":
            self._anchor = self.home.copy()
        self.primitive.requested_goals[stage] = self._anchor.tolist()
        self.primitive.executed_goals[stage] = self._anchor.tolist()

    def _fail(self, status):
        self.primitive.status = status

    def _sample(self, context):
        if self._sampled_step == self.executed_steps:
            return
        self._sampled_step = self.executed_steps
        raw = {key for key in context.objects if context.is_grasping(key)}
        for key in context.objects:
            self._grasp_frames[key] = (self._grasp_frames.get(key, 0) + 1
                                       if key in raw and self.gripper_command < 0 else 0)
        if self._loss_guard and not self._held_ids.issubset(raw):
            self._fail("lost_grasp")

    def check_retention(self, context):
        if self.primitive is None or self.primitive.done:
            return
        self._sample(context)
        if self._loss_guard and self.gripper_command >= 0:
            self._fail("unsafe_release")

    def next_action(self, context):
        if self.primitive is None or self.primitive.done:
            return None
        self._sample(context)
        if self.primitive.done:
            return None
        if self.executed_steps >= self.max_controls:
            self._fail("timeout")
            return None
        for _ in range(20):
            stage = self.primitive.stage
            elapsed = self.executed_steps - self._phase_start
            grasp = self.primitive.name == "grasp"
            command = -1. if (grasp and stage not in {"open", "retreat", "align", "approach"}) or (not grasp and stage not in {"release", "clear", "withdraw", "home_raise", "home_align", "home"}) else 1.
            if self._loss_guard and command >= 0:
                self._fail("unsafe_release")
                return None
            if stage == "approach" and grasp and any(context.is_grasping(key) for key in context.objects):
                # Stop at real bilateral finger contact rather than pushing through it.
                self.primitive.executed_goals["approach"] = np.asarray(context.ee_position).tolist()
                self._phase("close", context)
                continue
            if stage == "invalid":
                if elapsed:
                    self._fail(self._rejection)
                    return None
                command = -1. if self._rejection == "already_holding" else self.gripper_command
                return self._invalid_action(context, command)
            if stage in {"open", "close", "release"}:
                if elapsed < self.phase_budgets[stage]:
                    return self._hold_action(context, self._anchor, command)
                if stage == "release":
                    if any(context.is_grasping(key) for key in context.objects):
                        self._fail("release_failed")
                        return None
                    self._phase("clear", context)
                else:
                    self._phase("retreat" if stage == "open" else "confirm", context)
                continue
            if elapsed >= self.phase_budgets[stage]:
                self._fail("missing_grasp" if stage == "confirm" else
                           "no_support" if stage in {"descend", "settle"} else "timeout")
                return None
            if stage == "confirm":
                held = {key for key, frames in self._grasp_frames.items() if frames >= 4}
                if held:
                    self._held_ids = held
                    try:
                        self._capture_place_reference(context)
                    except (ValueError, RuntimeError) as exc:
                        self._fail(getattr(exc, "status", "invalid_goal"))
                        self.placement_failure = str(exc)
                        return None
                    self._rise_reference = {key: float(context.objects[key].position_root[2]) for key in held}
                    self._loss_guard = True
                    self._phase("lift", context)
                    continue
                return self._hold_action(context, self._anchor, -1.)
            if stage in {"descend", "settle"}:
                if stage == "descend" and lower_shelf_supported(context, self._held_ids, require_settled=False):
                    self._phase("settle", context)
                    continue
                supported = lower_shelf_supported(context, self._held_ids)
                if supported:
                    if stage == "descend":
                        self._phase("settle", context)
                    self._settle_count += 1
                    if self._settle_count >= 4:
                        self.support_stopped_object_ids = supported
                        self._loss_guard = False
                        self._phase("release", context)
                        continue
                    return self._hold_action(context, self._anchor, -1.)
                self._settle_count = 0
                if stage == "settle":
                    return self._hold_action(context, self._anchor, -1.)
                if context.ee_position[2] <= LOWER_SUPPORT_Z:
                    self._fail("no_support")
                    return None
                target = self._anchor.copy()
                target[2] = LOWER_SUPPORT_Z
                return self._descent_action(context, target)
            action, reached = self._motion_action(context, self._anchor, command,
                tolerance=.012,
                speed=.08 if stage == "approach" else .25)
            if not reached:
                return action
            if stage == "lift":
                if not all(float(context.objects[key].position_root[2]) - height >= .01
                           for key, height in self._rise_reference.items()):
                    self._fail("insufficient_lift")
                    return None
                following = "withdraw"
            else:
                following = {"retreat": "align" if grasp else "preplace",
                             "preplace": "approach", "align": "approach", "clear": "withdraw",
                             "approach": "close" if grasp else "descend",
                             "withdraw": "intermediate" if grasp else "home_raise",
                             "home_raise": "home_align",
                             "home_align": "home"}.get(stage)
            if following is None:
                self.primitive.status = "success"
                return None
            self._phase(following, context)
        raise RuntimeError("Pick/place phase transition loop did not emit a control")

    def _invalid_action(self, context, command):
        return np.array([0., 0., 0., command], dtype=np.float32)

    def _motion_action(self, context, goal, command, **kwargs):
        return self.primitive._move_to(context, goal, command, **kwargs)

    def _descent_action(self, context, goal):
        return self._motion_action(context, goal, -1., tolerance=.001, speed=.025)[0]

    def record_action(self, actual_action):
        super().record_action(actual_action)
        self.primitive.steps = self.executed_steps

    def observe(self, context):
        # Logging does not advance the per-control safety counters.
        result = self.primitive.observe(context)
        result["place_reference"] = deepcopy(self._place_reference)
        result["placement_failure"] = self.placement_failure
        return result


class PlannedPickPlaceSkillExecution(PickPlaceSkillExecution):
    """Use checked joint plans for the shared macro's motion phases."""

    def __init__(self, *, orientation_target, planning_seed=0, max_control_steps=400,
                 grasp_retention="friction", **kwargs):
        if (isinstance(max_control_steps, bool) or int(max_control_steps) != max_control_steps
                or max_control_steps < 1):
            raise ValueError("motion_plan_max_steps must be a positive integer")
        if grasp_retention not in {"friction", "constraint"}:
            raise ValueError("grasp_retention must be friction or constraint")
        self.grasp_retention = grasp_retention
        self.orientation_target = orientation_target
        self.planning_seed = int(planning_seed)
        self.max_control_steps = int(max_control_steps)
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.max_controls = 2000
        self.phase_budgets = dict(PICK_PLACE_MACRO_CONTRACT["phase_control_budgets"])
        self.plan_counter = 0
        self.motion_cursor = None
        self.motion_attempts = []
        self.last_motion_tick = {}
        self._hold_qpos = None
        self._active_motion_step = False
        self._pending_failure = None
        self._force_ticks = 0
        self._retention_attachment = None

    def start(self, action, context):
        self.motion_cursor = None
        self.motion_attempts = []
        self.last_motion_tick = {}
        self._hold_qpos = None
        self._active_motion_step = False
        self._pending_failure = None
        self._force_ticks = 0
        self.raw = context.env
        self.raw.set_grasp_retention(self.grasp_retention)
        self.raw.motion_planner  # Initialize contact tracking before closing.
        self._orientation = _numpy(self.orientation_target()).reshape(4)
        if not np.isfinite(self._orientation).all() or np.linalg.norm(self._orientation) <= 0:
            raise ValueError("Planned pick/place requires a finite nonzero reset quaternion")
        self._orientation = self._orientation / np.linalg.norm(self._orientation)
        super().start(action, context)
        if action.primitive_type == "grasp" and not self._held_ids:
            self._retention_attachment = None

    def check_retention(self, context):
        from domain.robocasa.motion_planning import MotionPlanningError

        super().check_retention(context)
        if not self._loss_guard or self.primitive.done or self._pending_failure is not None:
            return
        try:
            if self.grasp_retention == "constraint":
                binding = self.raw._grasp_binding
                feedback = binding.feedback()
                if feedback is None or feedback["name"] not in self._held_ids:
                    raise MotionPlanningError("grasp_binding_lost", "Confirmed carry requires its physical grasp constraint")
            if self._retention_attachment is None:
                self._retention_attachment = deepcopy(self.raw.motion_planner.attachment_state())
                if self._retention_attachment is None:
                    raise MotionPlanningError("unconfirmed_grasp", "Retained transport requires a confirmed physical payload")
            self.raw.motion_planner.check_attachment_feedback(
                SimpleNamespace(metadata={"attachment": self._retention_attachment}))
        except MotionPlanningError as exc:
            self._fail(exc.status)
            self.last_motion_tick["grasp_retention_failure"] = dict(status=exc.status, detail=exc.detail)

    def finish_motion(self, status):
        if self.motion_cursor is not None and self.motion_cursor.status == "running":
            self.motion_cursor.stop(status)
            self.motion_attempts[-1]["status"] = status

    def _phase(self, stage, context):
        self.finish_motion("support_stop" if stage == "settle" else "phase_complete")
        self.motion_cursor = None
        self._hold_qpos = None
        super()._phase(stage, context)

    def _fail(self, status):
        self.finish_motion(status)
        if self.executed_steps == 0:
            self._pending_failure = status
        else:
            super()._fail(status)

    def next_action(self, context):
        self._active_motion_step = False
        if self._pending_failure is not None:
            if self.executed_steps:
                self.primitive.status = self._pending_failure
                return None
            return self._hold_action(context, context.ee_position, self.gripper_command)
        return super().next_action(context)

    def _hold_action(self, context, goal, gripper):
        if self._hold_qpos is None:
            self._hold_qpos = _numpy(self.raw.agent.robot.get_qpos()).reshape(9)[:7].copy()
        return np.asarray([*self._hold_qpos, gripper], dtype=np.float32)

    def _invalid_action(self, context, command):
        return self._hold_action(context, context.ee_position, command)

    def _rotation_allowed(self, position):
        return self.primitive.stage == "preplace" and float(position[0]) < CABINET_MOUTH_X

    def _audit_path(self, plan, *, descent_start=None):
        from domain.robocasa.motion_planning import MotionPlanningError, _angle, _interpolate

        planner = self.raw.motion_planner
        maximum_angle = 0.
        for point in _interpolate(np.asarray(plan.position)):
            planner.kin.pmodel.compute_forward_kinematics(point[planner.pindices])
            pose = planner.kin.pmodel.get_link_pose(planner.kin.end_link_idx)
            position = _numpy(pose.p).reshape(3)
            angle = _angle(pose.q, self._orientation)
            maximum_angle = max(maximum_angle, angle)
            if angle > .5 and not self._rotation_allowed(position):
                raise MotionPlanningError("orientation_path_violation", {
                    "maximum_allowed_deg": .5, "observed_deg": angle})
            if descent_start is not None:
                if (np.linalg.norm(position[:2] - descent_start[:2]) > .002
                        or position[2] > descent_start[2] + .002
                        or position[2] < plan.target_position_root[2] - .002):
                    raise MotionPlanningError("descent_path_violation", {
                        "position_root_m": position.tolist(),
                        "start_root_m": descent_start.tolist(),
                        "goal_root_m": plan.target_position_root.tolist()})
        if angle > .5:
            raise MotionPlanningError("orientation_path_violation", {
                "endpoint_requires_fixed_orientation": True, "observed_deg": angle})
        plan.metadata["fixed_orientation_path_audit"] = {
            "maximum_angle_deg": maximum_angle, "tolerance_deg": .5,
            "max_joint_step_rad": .01, "exterior_rotation_stage": "preplace"}
        if descent_start is not None:
            plan.metadata["descent_cartesian_corridor_m"] = .002

    def _motion_action(self, context, goal, command, *, tolerance=.012, speed=.25,
                       support_surface_root_z=None):
        from domain.robocasa.motion_planning import MotionPlanExecution, MotionPlanningError, _angle

        if (self.motion_cursor is None
                and np.linalg.norm(np.asarray(goal) - context.ee_position) <= tolerance
                and _angle(self._feedback_orientation(), self._orientation) <= .5):
            return None, True
        self.check_retention(context)
        if self._pending_failure is not None:
            return self._hold_action(context, context.ee_position, command), False
        if self.primitive.done:
            return None, False
        if self.motion_cursor is not None and self.motion_cursor.status != "running":
            if self.motion_cursor.status != "success":
                self._fail(self.motion_cursor.status)
                return None, False
            return None, True
        if self.motion_cursor is None:
            attempt = dict(stage=self.primitive.stage, target_position_root=np.asarray(goal).tolist(),
                           target_quaternion_wxyz=self._orientation.tolist(),
                           seed=self.planning_seed + self.plan_counter, control_steps=0)
            self.plan_counter += 1
            self.motion_attempts.append(attempt)
            try:
                kwargs = dict(seed=attempt["seed"], allowed_contact_objects=tuple(context.objects),
                              maintain_orientation=self.primitive.stage != "preplace")
                if self.primitive.stage == "preplace":
                    kwargs["densify_rrt_path"] = True
                if support_surface_root_z is not None:
                    kwargs["support_surface_root_z"] = support_surface_root_z
                    kwargs["method"] = "bounded_ik"
                elif (self.primitive.stage == "clear"
                      or self.primitive.name == "grasp" and self.primitive.stage in {"approach", "lift"}):
                    kwargs["method"] = "bounded_ik"
                plan = self.raw.plan_to_pose(goal, self._orientation, **kwargs)
                if self.primitive.name == "grasp" and self.primitive.stage == "approach":
                    plan.metadata["allow_open_grasp_contact"] = True
                self._audit_path(plan, descent_start=(np.asarray(context.ee_position).copy()
                                 if support_surface_root_z is not None else None))
                budget = min(self.max_control_steps, self.max_controls - self.executed_steps,
                             self.phase_budgets[self.primitive.stage]
                             - (self.executed_steps - self._phase_start))
                self.motion_cursor = MotionPlanExecution(self.raw, plan, gripper_command=command,
                                                         max_steps=budget)
                attempt.update(status="running", planning=plan.metadata)
            except MotionPlanningError as exc:
                attempt.update(status=exc.status, detail=exc.detail)
                self._fail(exc.status)
                if self._pending_failure is not None:
                    return self._hold_action(context, context.ee_position, command), False
                return None, False
        self._active_motion_step = True
        return self.motion_cursor.next_action(), False

    def _descent_action(self, context, goal):
        from domain.robocasa.motion_planning import MotionPlanningError

        if self.motion_cursor is not None and self.motion_cursor.status == "success":
            self.motion_cursor = None
        target = np.asarray(context.ee_position, dtype=np.float32).copy()
        if self.motion_cursor is None:
            try:
                step = self.raw.motion_planner.support_descent_step(LOWER_SUPPORT_Z, max_step=.005)
            except MotionPlanningError as exc:
                self._fail(exc.status)
                self.last_motion_tick = dict(stage="descend", status=exc.status, detail=exc.detail)
                return None
            target[2] = max(LOWER_SUPPORT_Z, float(target[2]) - step)
        action, reached = self._motion_action(context, target, -1., tolerance=.0001,
                                               support_surface_root_z=LOWER_SUPPORT_Z)
        if reached:
            self._fail("no_support")
        return action

    def _feedback_orientation(self):
        tcp = self.raw.agent.robot.pose.inv() * self.raw.agent.tcp.pose
        return _numpy(tcp.q).reshape(4)

    def record_action(self, actual_action, *, terminated=False, truncated=False):
        from domain.robocasa.motion_planning import MotionPlanningError, _angle, cabinet_contact_forces

        action = np.asarray(actual_action, dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Planned pick/place requires finite 8D joint-position controls")
        self.gripper_command = float(action[-1])
        self.executed_steps += 1
        self.primitive.steps = self.executed_steps
        if self._active_motion_step:
            expected = self.motion_cursor.next_action()
            if not np.array_equal(action, expected):
                raise RuntimeError("Executed joint target differs from checked pick/place plan")
            try:
                self.last_motion_tick = self.motion_cursor.record_step(terminated=terminated, truncated=truncated)
            except MotionPlanningError as exc:
                self._fail(exc.status)
                self.motion_attempts[-1].update(status=exc.status, detail=exc.detail)
                return
            self.motion_attempts[-1].update(status=self.motion_cursor.status,
                control_steps=len(self.motion_cursor.records),
                endpoint_position_error_m=self.motion_cursor.endpoint_position_error_m,
                endpoint_angle_error_deg=self.motion_cursor.endpoint_angle_error_deg)
            if (_angle(self.last_motion_tick["tcp_quaternion_wxyz"], self._orientation) > .5
                    and not self._rotation_allowed(self.last_motion_tick["tcp_position_root"])):
                self._fail("orientation_tracking_error")
            elif self.motion_cursor.status not in {"running", "success"}:
                self._fail(self.motion_cursor.status)
        else:
            forces = cabinet_contact_forces(self.raw, self.raw.motion_planner.attachment_state())
            peak = max(forces.values(), default=0.)
            self._force_ticks = self._force_ticks + 1 if peak > 20 else 0
            quaternion = self._feedback_orientation()
            self.last_motion_tick = dict(stage=self.primitive.stage, control_mode="pd_joint_pos",
                                         cabinet_forces_N=forces,
                                         tcp_quaternion_wxyz=quaternion.tolist())
            binding = getattr(self.raw, "_grasp_binding", None)
            if binding is not None and binding.bound_object_id is not None:
                try:
                    self.last_motion_tick["binding"] = binding.feedback()
                except MotionPlanningError as exc:
                    self._fail(exc.status)
                    self.last_motion_tick["grasp_retention_failure"] = dict(status=exc.status, detail=exc.detail)
                    return
            if (_angle(quaternion, self._orientation) > .5
                    and not self._rotation_allowed(_numpy(
                        (self.raw.agent.robot.pose.inv() * self.raw.agent.tcp.pose).p).reshape(3))):
                self._fail("orientation_tracking_error")
            elif terminated or truncated:
                self._fail("terminated" if terminated else "truncated")
            elif self._force_ticks >= 3:
                self._fail("tracking_blocked")
