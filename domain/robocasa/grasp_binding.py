"""Explicit physical grasp binding after measured closed-gripper confirmation."""
import numpy as np

from domain.robocasa.motion_planning import MotionPlanningError, _angle, _array, _pose_vector


class GraspBinding:
    """Own one native positional lock and angular drive without moving either physical body."""

    def __init__(self, env):
        if getattr(env, "num_envs", 1) != 1 or bool(getattr(env, "gpu_sim_enabled", False)):
            raise MotionPlanningError("unsupported", "Grasp binding requires single-environment CPU simulation")
        self.env = env
        self._bound_object_id = None
        self._drive = None
        self._actor = None
        self._child_entity = None
        self._native_scene = None
        self._tcp_to_object = None
        self._finger_hold_target = None
        self._substep_position = None
        self._control_max_pose_speed = None
        self._control_pose_samples = 0

    @property
    def bound_object_id(self):
        return self._bound_object_id

    @staticmethod
    def _pose(pose):
        import sapien

        return sapien.Pose(_array(pose.p).reshape(3), _array(pose.q).reshape(4))

    def bind_confirmed(self, object_id):
        if self._bound_object_id is not None:
            if object_id != self._bound_object_id:
                raise MotionPlanningError("grasp_binding_conflict", "Release the current bound object before binding another")
            return self.feedback()
        planner = self.env.motion_planner
        attachment = planner.attachment_state()
        if (attachment is None or attachment["name"] != object_id
                or attachment["gripper_command"] >= 0 or planner.gripper_command >= 0
                or planner.grasp_frames.get(object_id, 0) < 4
                or not planner._grasp_contacts().get(object_id, False)):
            raise MotionPlanningError("unconfirmed_grasp", "Binding requires four consecutive closed-gripper physical grasp controls for the selected object")
        actors = [actor for actor in self.env.cabinet_items if actor.name == object_id]
        parents = [link for link in self.env.agent.robot.get_links() if link.name == "panda_hand"]
        if len(actors) != 1 or len(parents) != 1 or len(self.env.scene.sub_scenes) != 1:
            raise MotionPlanningError("unsupported", "Grasp binding requires one named cabinet object, one panda_hand link, and one native scene")
        actor, parent = actors[0], parents[0]
        if len(actor._bodies) != 1 or len(parent._objs) != 1:
            raise MotionPlanningError("unsupported", "Grasp binding requires one native object body and one native hand body")
        import sapien

        child_body = actor._bodies[0]
        object_pose = self._pose(actor.pose)
        parent_anchor = self._pose(parent.pose).inv() * object_pose
        tcp_to_object = _pose_vector(self._pose(self.env.agent.tcp.pose).inv() * object_pose)
        scene = self.env.scene.sub_scenes[0]
        drive = scene.create_drive(parent._objs[0], parent_anchor, child_body, sapien.Pose())
        try:
            drive.set_limit_x(0., 0.)
            drive.set_limit_y(0., 0.)
            drive.set_limit_z(0., 0.)
            # SAPIEN 3.0.3 cone setters leave swing axes FREE. Explicit angular
            # drive is required; zero cone angles do not lock those axes.
            drive.set_drive_property_slerp(1e6, 2e3, mode="acceleration")
            drive.set_drive_target(sapien.Pose())
        except Exception:
            child_body.entity.remove_component(drive)
            raise
        self._drive, self._actor = drive, actor
        self._child_entity, self._native_scene = child_body.entity, scene
        self._tcp_to_object = tcp_to_object
        self._bound_object_id = object_id
        return self.feedback()

    def begin_control(self):
        self._control_max_pose_speed = None
        self._control_pose_samples = 0
        self._substep_position = (_array(self._actor.pose.p).reshape(3).copy()
                                 if self._bound_object_id is not None else None)

    def observe_substep(self, dt):
        if self._bound_object_id is None:
            return
        if not np.isfinite(dt) or dt <= 0:
            raise MotionPlanningError("grasp_binding_error", "Invalid physics substep duration")
        position = _array(self._actor.pose.p).reshape(3).copy()
        if self._substep_position is not None:
            speed = float(np.linalg.norm(position-self._substep_position)/dt)
            if not np.isfinite(speed):
                raise MotionPlanningError("grasp_binding_error", "Invalid bound-object physical pose speed")
            self._control_max_pose_speed = max(self._control_max_pose_speed or 0., speed)
            self._control_pose_samples += 1
        self._substep_position = position

    def hold_fingers_at_current_width(self):
        """Remove closing pressure at support while the physical binding stays active."""
        if self._bound_object_id is None:
            raise MotionPlanningError("missing_grasp", "Finger hold requires an active grasp binding")
        self._finger_hold_target = self.env.agent.controller.controllers["gripper"].qpos.clone()

    def apply_finger_hold(self):
        if self._finger_hold_target is not None and self._bound_object_id is not None:
            # Preserve the logical closing target used for opening/release checks.
            self.env.agent.controller.controllers["gripper"].set_drive_targets(self._finger_hold_target)

    def release(self):
        if self._drive is not None and self._drive.entity is not None:
            self._drive.entity.remove_component(self._drive)
        self._drive = self._actor = self._child_entity = self._native_scene = None
        self._tcp_to_object = None
        self._finger_hold_target = None
        self._substep_position = None
        self._control_max_pose_speed = None
        self._control_pose_samples = 0
        self._bound_object_id = None

    def reset(self):
        self.release()

    def feedback(self):
        if self._bound_object_id is None:
            return None
        if (not self._drive.is_enabled or self._drive.entity != self._child_entity
                or self._drive not in self._child_entity.components
                or self._child_entity.scene != self._native_scene
                or not any(actor is self._actor for actor in self.env.cabinet_items)
                or self._actor._bodies[0].entity != self._child_entity):
            raise MotionPlanningError("grasp_binding_lost", "The grasp drive or its bound object was disabled, removed, or replaced")
        relative = _pose_vector(self._pose(self.env.agent.tcp.pose).inv() * self._pose(self._actor.pose))
        if not np.isfinite(relative).all():
            raise MotionPlanningError("grasp_binding_error", "Bound object feedback contains nonfinite pose values")
        position_error = float(np.linalg.norm(relative[:3] - self._tcp_to_object[:3]))
        angle_error = _angle(relative[3:], self._tcp_to_object[3:])
        if position_error > .002 or angle_error > 2.:
            raise MotionPlanningError("grasp_binding_slip", {
                "position_error_m": position_error, "angle_error_deg": angle_error,
                "message": "Physical grasp binding exceeded its measured TCP-relative pose tolerance"})
        return dict(name=self._bound_object_id, tcp_to_object_pose=relative.tolist(),
                    position_error_m=position_error, angle_error_deg=angle_error,
                    physical_binding=True,
                    native_speed_m_s=float(np.linalg.norm(_array(self._actor.linear_velocity))),
                    max_substep_pose_speed_m_s=self._control_max_pose_speed,
                    pose_speed_samples=self._control_pose_samples)
