"""Device-resident split7 execution; scalar primitives remain the reference."""

from dataclasses import dataclass

import torch

from domain.robocasa.primitives import RAPS_SPLIT_ACTION_SCHEMA

PRIMITIVE_NAMES = RAPS_SPLIT_ACTION_SCHEMA.primitive_names


@dataclass
class BatchedObservation:
    ee_position: torch.Tensor
    ee_quaternion: torch.Tensor
    object_positions_root: torch.Tensor
    object_positions_world: torch.Tensor
    gripper_open: torch.Tensor
    finger_forces: torch.Tensor
    grasped: torch.Tensor
    support_forces: torch.Tensor


STATUS_NAMES = ("idle", "running", "success", "timeout", "contact_stop",
                "supported", "support_unstable", "interrupted")
STAGE_NAMES = ("start", "move", "align", "approach", "lift", "drop",
               "open_gripper", "close_gripper", "settle_support")


class BatchedSkillExecution:
    """Independent slot state, with feedback every two local executed steps.

    ``next_action`` returns zero motion and the latched gripper for idle rows;
    ``action_mask`` identifies rows which emitted a real scalar-reference action.
    Call ``record_action`` after stepping, then ``observe`` with post-step sensors.
    Object-axis order must stay fixed throughout each skill. ``grasped`` includes
    the scalar context's two-finger force fallback.
    """

    def __init__(self, n, device, settings=None):
        self.n, self.device = n, torch.device(device)
        settings = {} if settings is None else settings
        def setting(name, default):
            return bool(settings.get(name, default) if isinstance(settings, dict)
                        else getattr(settings, name, default))
        self.latched_gripper = setting("latched_gripper", False)
        self.stop_drop_on_support = setting("stop_drop_on_support", False)
        self.hold_closed_during_motion = setting("hold_closed_during_motion", False)
        self.grasp_stop_on_contact = setting("grasp_stop_on_contact", True)
        self.controlled_placement = setting("controlled_placement", False)
        if self.controlled_placement and not (self.latched_gripper and self.stop_drop_on_support):
            raise ValueError("controlled_placement requires latched_gripper and stop_drop_on_support")
        if setting("project_goals_to_region", False):
            raise ValueError("Batched split7 does not support goal projection")
        schema = RAPS_SPLIT_ACTION_SCHEMA
        self._delta_scales = torch.zeros(
            schema.num_primitives, schema.num_parameters, 3, device=self.device)
        axes = {
            "move_delta_ee_pose": {"dx": (0, 1), "dy": (1, 1), "dz": (2, 1)},
            "align_yz": {"y_offset": (1, 1), "z_offset": (2, 1)},
            "approach_x": {"x_forward": (0, 1)},
            "lift": {"z_dist": (2, 1)}, "drop": {"z_dist": (2, -1)},
        }
        for primitive_index, spec in enumerate(schema.specs):
            for argument_index, parameter in zip(
                    schema.primitive_name_to_action_idx[spec.name], spec.parameters):
                axis, sign = axes[spec.name][parameter.name]
                self._delta_scales[primitive_index, argument_index, axis] = sign * parameter.scale
        self._magnitude_parameters = torch.tensor([
            name in {"approach_x", "lift", "drop"} for name in schema.primitive_names
        ], device=self.device)
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros((n, *shape), device=self.device, dtype=dtype)
        self.status = zeros(dtype=torch.long)
        self.stage = zeros(dtype=torch.long)
        self.primitive_type = zeros(dtype=torch.long)
        self.steps = zeros(dtype=torch.long)
        self.executed_steps = zeros(dtype=torch.long)
        self.max_steps = zeros(dtype=torch.long)
        self.actions = zeros(schema.action_dim)
        self.goal = zeros(3)
        self.gripper_command = torch.ones(n, device=self.device)
        self.action_mask = zeros(dtype=torch.bool)
        self.orientation_hold = zeros(4)
        self.orientation_hold_mask = zeros(dtype=torch.bool)
        self.hold_goal = zeros(3)
        self.hold_mask = zeros(dtype=torch.bool)
        self.settle_goal = zeros(3)
        self.settle_count = zeros(dtype=torch.long)
        self.settle_start_step = zeros(dtype=torch.long)
        self.previous_observed_step = zeros(dtype=torch.long)
        self.previous_observed_position = zeros(3)
        self.last_feedback_step = zeros(dtype=torch.long)
        self.drop_objects = zeros(2, dtype=torch.bool)
        self.support_stopped_objects = zeros(2, dtype=torch.bool)
        self.stable_grasp_frames = zeros(2, dtype=torch.long)
        self.grasped_objects = zeros(2, dtype=torch.bool)
        self.max_finger_contact_force = zeros(2, 2)
        self.initial_object_positions_root = zeros(2, 3)
        self.initial_object_positions_world = zeros(2, 3)

    @property
    def active(self):
        return self.status == 1

    @property
    def complete(self):
        return (self.status != 0) & ~self.active

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.status[ids] = 0
        self.gripper_command[ids] = 1
        self.action_mask[ids] = False
        self.orientation_hold_mask[ids] = False

    def _orientation(self, obs, mask):
        q = obs.ee_quaternion[mask]
        norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        if not torch.isfinite(q).all() or not (norm > 0).all():
            raise ValueError("Orientation hold requires finite nonzero wxyz quaternions")
        self.orientation_hold[mask] = q / norm
        self.orientation_hold_mask[mask] = True

    def start(self, ids, actions, observation):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        schema = RAPS_SPLIT_ACTION_SCHEMA
        if actions.shape != (ids.numel(), schema.action_dim) or not torch.isfinite(actions).all():
            raise ValueError("Expected finite split7 actions with shape [K,13]")
        if ids.unique().numel() != ids.numel():
            raise ValueError("Skill slot IDs must be unique")
        obs = observation
        kind = actions[:, :schema.num_primitives].argmax(-1)
        args = actions[:, schema.num_primitives:].clamp(-1, 1)
        args = torch.where(self._magnitude_parameters[kind, None], args.abs(), args)
        delta = torch.bmm(args[:, None, :], self._delta_scales[kind]).squeeze(1)
        self.actions[ids] = actions
        self.primitive_type[ids] = kind
        self.goal[ids] = obs.ee_position[ids] + delta
        self.status[ids] = 1
        self.stage[ids] = kind + 1
        self.max_steps[ids] = torch.tensor([100, 100, 150, 60, 60, 12, 12], device=self.device)[kind]
        for value in (self.steps, self.executed_steps, self.settle_count,
                      self.settle_start_step, self.previous_observed_step,
                      self.last_feedback_step, self.stable_grasp_frames,
                      self.max_finger_contact_force):
            value[ids] = 0
        for value in (self.orientation_hold_mask, self.hold_mask, self.drop_objects,
                      self.support_stopped_objects, self.grasped_objects, self.action_mask):
            value[ids] = False
        self.initial_object_positions_root[ids] = obs.object_positions_root[ids]
        self.initial_object_positions_world[ids] = obs.object_positions_world[ids]
        if self.controlled_placement:
            self.hold_mask[ids] = kind >= 5
            self.hold_goal[ids] = obs.ee_position[ids]
            orientation = ((kind >= 5) | ((obs.ee_position[ids, 2] < .15)
                & ((kind == 3) | (((kind == 0) | (kind == 1)) & (delta[:, 2] > 0)))))
            mask = torch.zeros(self.n, device=self.device, dtype=torch.bool)
            mask[ids] = orientation
            self._orientation(obs, mask)
        if self.stop_drop_on_support:
            self.drop_objects[ids] = obs.grasped[ids] & (kind == 4)[:, None]

    @staticmethod
    def _motion(error, speed):
        motion = error / .10
        norm = torch.linalg.vector_norm(motion, dim=-1)
        return motion * (speed / norm.clamp_min(1e-30)).clamp_max(1)[:, None]

    def next_action(self, obs, mask=None):
        """Advance selected rows; preserve unselected rows' action masks."""
        selected = torch.ones_like(self.active) if mask is None else mask
        running = self.active & selected
        settling = self.active & (self.stage == 8)
        # The support deadline short-circuits _tick in the scalar executor.
        deadline = running & settling & (self.executed_steps - self.settle_start_step >= 20)
        self.status[deadline] = 6
        ticking = running & ~deadline
        self.steps[ticking] += 1
        timeout = ticking & (self.steps > self.max_steps)
        self.status[timeout & ~settling] = 3
        self.status[timeout & settling] = 6
        running = self.active & selected
        kind = self.primitive_type
        error = self.goal - obs.ee_position
        distance = torch.linalg.vector_norm(error, dim=-1)
        tolerance = torch.full((self.n,), .012, device=self.device)
        speed = torch.full_like(tolerance, .25)
        tolerance[kind == 1] = .015
        speed[kind == 1] = .30
        speed[kind == 2] = .08
        if self.controlled_placement:
            drop = kind == 4
            tolerance[drop] = .002
            speed[drop] = (.25 * (obs.ee_position[drop, 2] - self.goal[drop, 2]).clamp_min(0) / .08).clamp(.025, .25)
        finished = running & ~settling & (((kind < 5) & (distance <= tolerance))
                                         | ((kind >= 5) & (self.steps >= 10)))
        self.status[finished] = 2
        self.action_mask[selected] = self.active[selected]
        grip = self.gripper_command.clone() if self.latched_gripper else (2 * obs.gripper_open - 1).clamp(-1, 1)
        grip[kind == 5] = 1
        grip[kind == 6] = -1
        if not self.latched_gripper and self.hold_closed_during_motion:
            grip[(kind == 0) | (kind == 3) | (kind == 4)] = -1
        low = torch.cat((self._motion(error, speed).clamp(-1, 1), grip[:, None]), -1)
        low[kind >= 5, :3] = 0
        hold = self.hold_mask & self.active
        low[hold, :3] = self._motion(self.hold_goal - obs.ee_position, torch.full_like(speed, .25))[hold]
        low[settling, :3] = self._motion(self.settle_goal - obs.ee_position, torch.full_like(speed, .25))[settling]
        low[settling, 3] = self.gripper_command[settling]
        low[~self.action_mask, :3] = 0
        low[~self.action_mask, 3] = self.gripper_command[~self.action_mask]
        return low

    def record_action(self, actual_action, mask=None):
        mask = self.action_mask if mask is None else mask & self.action_mask
        if actual_action.shape != (self.n, 4) or not torch.isfinite(actual_action[mask]).all():
            raise ValueError("Executed actions must be finite with shape [N,4]")
        if self.latched_gripper:
            self.gripper_command[mask] = actual_action[mask, 3]
        self.executed_steps[mask] += 1

    def observe(self, obs, mask=None, force_mask=None):
        """Read each new even local step, or a forced final odd step, once."""
        cadence = self.executed_steps.remainder(2) == 0
        if force_mask is not None:
            cadence |= force_mask
        feedback = ((self.status != 0) & (self.executed_steps > self.last_feedback_step)
                    & cadence)
        if mask is not None:
            feedback &= mask
        self.last_feedback_step[feedback] = self.executed_steps[feedback]
        self.max_finger_contact_force[feedback] = torch.maximum(
            self.max_finger_contact_force[feedback], obs.finger_forces[feedback])
        frames = torch.where(obs.grasped, self.stable_grasp_frames + 1, 0)
        self.stable_grasp_frames[feedback] = frames[feedback]
        self.grasped_objects[feedback] |= frames[feedback] >= 4
        if self.grasp_stop_on_contact:
            contact = (obs.finger_forces >= .05).all(-1).any(-1)
            self.status[feedback & self.active & (self.primitive_type == 2) & contact] = 4
        if not self.stop_drop_on_support:
            return feedback
        supported_objects = self.drop_objects & (obs.support_forces > .05)
        supported = supported_objects.any(-1)
        drop = feedback & (self.primitive_type == 4)
        if not self.controlled_placement:
            done = drop & supported
            self.status[done] = 5
            self.support_stopped_objects[done] = supported_objects[done]
            return feedback
        settling = drop & (self.stage == 8) & self.active
        elapsed = self.executed_steps - self.previous_observed_step
        speed = torch.linalg.vector_norm(obs.ee_position - self.previous_observed_position, dim=-1) / (elapsed.clamp_min(1) * .05)
        anchor_error = torch.linalg.vector_norm(obs.ee_position - self.settle_goal, dim=-1)
        stable = supported & (anchor_error <= .003) & (speed <= .01)
        update = settling & (elapsed > 0)
        self.settle_count[update] = torch.where(stable[update], self.settle_count[update] + 1, 0)
        self.previous_observed_position[update] = obs.ee_position[update]
        self.previous_observed_step[update] = self.executed_steps[update]
        done = update & (self.settle_count >= 4)
        self.status[done] = 5
        self.support_stopped_objects[done] = supported_objects[done]
        self.status[settling & self.active & (self.executed_steps - self.settle_start_step >= 20)] = 6
        enter = drop & (self.stage != 8) & supported
        self.stage[enter] = 8
        self.status[enter] = 1
        self.settle_goal[enter] = obs.ee_position[enter]
        self.previous_observed_position[enter] = obs.ee_position[enter]
        self.settle_start_step[enter] = self.executed_steps[enter]
        self.previous_observed_step[enter] = self.executed_steps[enter]
        self.settle_count[enter] = 0
        self._orientation(obs, enter)
        return feedback
