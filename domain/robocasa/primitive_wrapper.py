"""Robot-aware adapter for executing one RAPS primitive as one environment step.

Why this wrapper exists
-----------------------
RAPS does not expose the simulator's low-level controller to the policy.  The
policy emits one hybrid high-level action, for example
``top_grasp(x_forward=0.10, y_offset=0.0, z_offset=0.0)``.
RoboCasa, however, must receive many low-level
robot-independent ``[dx, dy, dz, gripper]`` actions to execute that behavior.
``RAPSEnvWrapper`` bridges those interfaces and adapts the command to either
the fixed-base Fetch controller or Panda's EE-delta-pose controller:

``20D RAPS action -> decode primitive -> run its state machine -> H low-level steps``

The wrapper returns only the final observation to the caller, while storing the
dense ``H+1`` observation trajectory in ``PrimitiveRecord``.  That record is
the input for the T=11 PointWorld/WDS conversion.  This class does not select
candidates or run PointWorld; it only executes and records the selected action.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - compatibility for older RAPS setups
    import gym
    from gym import spaces

from .primitives import (
    PRIMITIVE_CLASSES,
    Primitive,
    PrimitiveAction,
    PrimitiveContext,
    RAPS_ACTION_SCHEMA,
    normalize_primitive_robot,
    update_cumulative_lifts,
)
from exploration.trajectory_recorder import PrimitiveRecord


class _EEDeltaAction(PrimitiveAction):
    def to_vector(self):
        return np.array([self.parameters[key] for key in ("dx", "dy", "dz", "gripper")],
                        dtype=np.float32)


class _EEDeltaStep(Primitive):
    """One controller tick; reuse outcome observation without a motion state machine."""
    name = "ee_delta"

    def __init__(self, action):
        super().__init__(action)
        self.max_steps = 1
        self.stage = "control"

    def next_action(self, context):
        if self.done:
            return None
        self.steps = 1
        self.status = "success"
        return self.action.to_vector()


class RAPSEnvWrapper(gym.Wrapper):
    """Expose one parameterized primitive as one reward-free environment step.

    Input:
        A normalized 20D vector consisting of a 10D primitive selector and a
        shared 10D argument bank.

    Output:
        The final observation after the primitive terminates, reward ``0.0``
        for reward-free exploration, Gymnasium termination flags, and an
        ``info['primitive_record']`` containing every low-level transition.

    Internally, the selected :class:`Primitive` runs as a state machine.  It
    may take many simulator steps before this wrapper returns.
    """

    def __init__(
        self,
        env,
        *,
        max_primitives_per_episode: int = 40,
        max_low_level_steps_per_episode: int = 1000,
        max_ee_steps_per_episode: int = 100,
        debug_base_reward: bool = False,
        observation_transform=None,
        robot: str = "fetch",
        render_during_primitive: bool = False,
        record_dense_trajectory: bool = True,
        contact_check_interval: int = 1,
        project_goals_to_region: bool = False,
        action_mode: str = "primitives",
    ):
        if int(max_primitives_per_episode) < 1:
            raise ValueError("max_primitives_per_episode must be positive")
        if int(max_low_level_steps_per_episode) < 1:
            raise ValueError("max_low_level_steps_per_episode must be positive")
        if int(max_ee_steps_per_episode) < 1:
            raise ValueError("max_ee_steps_per_episode must be positive")
        if int(contact_check_interval) < 1:
            raise ValueError("contact_check_interval must be positive")
        if action_mode not in {"primitives", "ee_delta"}:
            raise ValueError("action_mode must be primitives or ee_delta")
        super().__init__(env)
        self.action_mode = action_mode
        self._direct_grasp_frames = {}
        self._direct_lift_reference_heights = {}
        self.action_space = spaces.Box(
            low=RAPS_ACTION_SCHEMA.low,
            high=RAPS_ACTION_SCHEMA.high,
            dtype=np.float32,
        )
        if self.action_mode == "ee_delta":
            self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.max_primitives_per_episode = int(max_primitives_per_episode)
        self.max_low_level_steps_per_episode = int(max_low_level_steps_per_episode)
        self.max_ee_steps_per_episode = int(max_ee_steps_per_episode)
        self._ee_orientation_target = None
        self._ee_orientation_override = None
        self.debug_base_reward = bool(debug_base_reward)
        self.observation_transform = observation_transform
        self.robot = normalize_primitive_robot(robot)
        self.render_during_primitive = bool(render_during_primitive)
        self.record_dense_trajectory = bool(record_dense_trajectory)
        self.contact_check_interval = int(contact_check_interval)
        self.project_goals_to_region = bool(project_goals_to_region)
        self.episode_id = 0
        self.primitive_id = 0
        self.low_level_steps = 0
        self.last_primitive_record: PrimitiveRecord | None = None
        self.records: list[PrimitiveRecord] = []
        self._needs_reset = True
        self._validate_robot_action_space()
        self._configure_low_level_time_limit()

    @property
    def controller_action_dim(self) -> int:
        return 4 if self.robot == "fetch" else 7

    def _validate_robot_action_space(self) -> None:
        shape = getattr(getattr(self.env, "action_space", None), "shape", None)
        if shape is None or len(shape) != 1:
            raise ValueError(
                "RAPSEnvWrapper requires a one-dimensional low-level action space"
            )
        actual_dim = int(shape[0])
        if actual_dim != self.controller_action_dim:
            controller = (
                "pd_ee_delta_pos with fixed Fetch base/body"
                if self.robot == "fetch"
                else "pd_ee_delta_pose"
            )
            raise ValueError(
                f"robot={self.robot!r} requires a {self.controller_action_dim}D "
                f"{controller} action, but the environment exposes {actual_dim}D"
            )

    def _controller_action(self, primitive_action: np.ndarray) -> np.ndarray:
        """Map canonical [dxyz, gripper] to the selected robot controller."""
        action = np.asarray(primitive_action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError(f"primitive emitted invalid low-level action: {action}")
        if self.robot == "fetch":
            return action.copy()
        rotation = np.zeros(3, dtype=np.float32)
        if self.action_mode == "ee_delta":
            if self._ee_orientation_target is None:
                raise RuntimeError("ee_delta orientation target requires reset()")
            arm = self.env.unwrapped.agent.controller.controllers["arm"]
            current = arm.ee_pose_at_base.q.detach().cpu().numpy().reshape(4)
            current = Rotation.from_quat(np.roll(current, -1))
            # ManiSkill composes intrinsic XYZ Euler deltas on the left in
            # root coordinates, and scales normalized rotation by rot_lower.
            target = self._ee_orientation_override
            if target is None:
                target = self._ee_orientation_target
            correction = (target * current.inv()).as_euler("XYZ")
            rotation = correction / float(arm.config.rot_lower)
            rotation /= max(1.0, float(np.linalg.norm(rotation)))
        return np.array(
            [*action[:3], *rotation, action[3]],
            dtype=np.float32,
        )

    def set_ee_orientation_override(self, quaternion_wxyz):
        """A skill may hold its observed reachable wrist pose; None restores reset."""
        if self.robot != "panda" or self.action_mode != "ee_delta":
            raise ValueError("Skill orientation hold requires Panda ee_delta control")
        if quaternion_wxyz is None:
            self._ee_orientation_override = None
            return
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) <= 1e-12:
            raise ValueError("Orientation override requires a finite nonzero wxyz quaternion")
        self._ee_orientation_override = Rotation.from_quat(np.roll(quaternion / np.linalg.norm(quaternion), -1))

    def _context(self):
        return PrimitiveContext(
            self.env,
            robot=self.robot,
            record_state=self.record_dense_trajectory,
            project_goals_to_region=self.project_goals_to_region,
        )

    def _configure_low_level_time_limit(self) -> None:
        """Align any outer low-level TimeLimit with the high-level budget.

        ManiSkill registers RoboCasa with a 200-step TimeLimitWrapper.  One
        high-level RAPS episode may legitimately use more than 200 simulator
        steps, so leaving that wrapper unchanged silently truncates collection
        before ``max_low_level_steps_per_episode`` is reached.
        """
        current = self.env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "_max_episode_steps"):
                current._max_episode_steps = self.max_low_level_steps_per_episode
            current = getattr(current, "env", None)

    def _transform_observation(self, observation):
        if self.observation_transform is None:
            return observation
        return self.observation_transform(observation)

    def _record_metadata(self) -> dict[str, Any]:
        base = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        agent = getattr(base, "agent", None)
        robot = getattr(agent, "robot", None)
        joint_names = (
            [joint.name for joint in robot.get_active_joints()]
            if robot is not None
            else []
        )
        return {
            "robot_type": getattr(agent, "uid", type(agent).__name__.lower()),
            "primitive_robot": self.robot,
            "robot_urdf": str(getattr(agent, "urdf_path", "")),
            "active_joint_names": joint_names,
            "camera_names": list(getattr(base, "_sensors", {}).keys()),
            "layout_and_style_ids": getattr(base, "layout_and_style_ids", None),
            "public_action_dim": int(self.env.action_space.shape[-1]),
            "full_controller_action_dim": int(
                getattr(base, "_full_action_space", self.env.action_space).shape[-1]
            ),
        }

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._ee_orientation_override = None
        if self.action_mode == "ee_delta" and self.robot == "panda":
            arm = self.env.unwrapped.agent.controller.controllers["arm"]
            cfg = arm.config
            if (not cfg.normalize_action or not cfg.use_delta or cfg.use_target
                    or cfg.frame != "root_translation:root_aligned_body_rotation"
                    or not np.isscalar(cfg.rot_lower) or not np.isscalar(cfg.rot_upper)
                    or not np.isfinite(cfg.rot_lower) or cfg.rot_lower >= 0
                    or cfg.rot_upper != -cfg.rot_lower):
                raise ValueError("ee_delta wrist hold requires normalized current-pose root delta control with symmetric scalar rotation bounds")
            quaternion = arm.ee_pose_at_base.q.detach().cpu().numpy().reshape(4)
            self._ee_orientation_target = Rotation.from_quat(np.roll(quaternion, -1))
        if isinstance(result, tuple):
            result = (self._transform_observation(result[0]), *result[1:])
            self._last_observation = result[0]
        else:
            result = self._transform_observation(result)
            self._last_observation = result
        self._direct_grasp_frames = {}
        self._direct_lift_reference_heights = {}
        self.episode_id += 1
        self.primitive_id = 0
        self.low_level_steps = 0
        self.last_primitive_record = None
        self.records = []
        self._needs_reset = False
        return result

    @staticmethod
    def _step_result(result):
        if len(result) == 5:
            return result
        observation, reward, done, info = result
        return observation, reward, bool(done), False, info

    def step(self, raps_action):
        """Execute exactly one decoded RAPS primitive to completion.

        The primitive controller, rather than the PointWorld model, produces
        the low-level actions.  PointWorld only consumes the resulting record
        later during dataset construction or model training.
        """
        if self._needs_reset:
            raise RuntimeError("RAPSEnvWrapper.step() called before reset or after episode end")

        action = np.asarray(raps_action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"{self.action_mode} action must have shape {self.action_space.shape}, got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError("RAPS action contains NaN or infinity")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.action_mode == "ee_delta":
            primitive_action = _EEDeltaAction("ee_delta", dict(zip(
                ("dx", "dy", "dz", "gripper"), map(float, action))))
            primitive = _EEDeltaStep(primitive_action)
        else:
            primitive_action = RAPS_ACTION_SCHEMA.decode(action)
            primitive_cls = PRIMITIVE_CLASSES[primitive_action.primitive_type]
            primitive = primitive_cls(primitive_action)

        observation = getattr(self, "_last_observation", None)
        if observation is None:
            observation = self.env.get_observation() if hasattr(self.env, "get_observation") else None
        context = self._context()
        primitive.reset(context)
        if self.action_mode == "ee_delta":
            primitive.stable_grasp_frames.update(self._direct_grasp_frames)
            if self.project_goals_to_region:
                controller_config = context.arm_controller.config
                if (not controller_config.normalize_action or not controller_config.use_delta
                        or controller_config.use_target
                        or not controller_config.frame.startswith("root_translation")):
                    raise ValueError("ee_delta projection requires normalized, current-pose root delta control")
                lower = np.broadcast_to(controller_config.pos_lower, (3,)).astype(np.float32)
                upper = np.broadcast_to(controller_config.pos_upper, (3,)).astype(np.float32)
                if np.any(upper <= lower):
                    raise ValueError("controller translation bounds must have positive width")
                delta = lower + (action[:3] + 1.0) * 0.5 * (upper - lower)
                requested = context.ee_position + delta
                projected = context.project_goal(requested)
                normalized = 2.0 * (projected - context.ee_position - lower) / (upper - lower) - 1.0
                executed = np.clip(normalized, -1.0, 1.0)
                primitive.parameters.update(zip(("dx", "dy", "dz"), map(float, executed)))
                primitive.action.parameters.update(primitive.parameters)
                primitive.requested_goals["target"] = requested.tolist()
                primitive.executed_goals["target"] = projected.tolist()
                primitive.controller_target = (context.ee_position + lower
                    + (executed + 1.0) * 0.5 * (upper - lower)).tolist()
        record = PrimitiveRecord(
            episode_id=self.episode_id,
            primitive_id=self.primitive_id,
            primitive_type=primitive_action.primitive_type,
            raps_action_normalized=action.copy(),
            parameters_physical=dict(primitive_action.parameters),
            metadata=(
                self._record_metadata()
                if self.record_dense_trajectory
                else {"dense_trajectory": False}
            ),
            dense_trajectory=self.record_dense_trajectory,
        )
        if self.action_mode == "ee_delta":
            record.schema_version = "cabinet-ee-delta-v1"
            record.metadata.update(action_mode="ee_delta", action_units="normalized_controller",
                                   project_goals_to_region=self.project_goals_to_region,
                                   wrist_orientation_hold=("skill_actual_pose" if self._ee_orientation_override is not None
                                                           else "episode_reset") if self.robot == "panda" else "native_controller")
            if self.project_goals_to_region:
                record.metadata["controller_target"] = primitive.controller_target
        if self.record_dense_trajectory:
            record.append_observation(observation, context)

        terminated = truncated = False
        base_rewards = []
        for _ in range(primitive.max_steps + 1):
            if self.low_level_steps >= self.max_low_level_steps_per_episode:
                break
            low_action = primitive.next_action(context)
            if low_action is None:
                break
            low_action = np.asarray(low_action, dtype=np.float32)
            controller_action = self._controller_action(low_action)
            result = self._step_result(self.env.step(controller_action))
            observation, base_reward, terminated, truncated, info = result
            if self.render_during_primitive:
                self.env.render()
            observation = self._transform_observation(observation)
            if self.debug_base_reward:
                base_rewards.append(float(np.asarray(base_reward).mean()))
            post_context = self._context()
            base_env = (
                self.env.unwrapped
                if hasattr(self.env, "unwrapped")
                else self.env
            )
            should_check_contact = (
                self.record_dense_trajectory
                or primitive.steps % self.contact_check_interval == 0
                or terminated
                or truncated
                or primitive.steps >= primitive.max_steps
                or self.low_level_steps + 1 >= self.max_low_level_steps_per_episode
            )
            contact_event = (
                primitive.observe(post_context)
                if should_check_contact
                else {"objects": {}}
            )
            if self.record_dense_trajectory:
                record.append_transition(
                    low_action,
                    full_controller_action=getattr(
                        base_env, "last_full_controller_action", controller_action
                    ),
                    contact_event=contact_event,
                )
                record.append_observation(observation, post_context)
            else:
                record.append_compact_transition(
                    low_action, full_controller_action=controller_action,
                    contact_event=contact_event
                )
            self.low_level_steps += 1
            self._last_observation = observation
            context = post_context
            if terminated or truncated or self.low_level_steps >= self.max_low_level_steps_per_episode:
                break

        budget_limit = self.low_level_steps >= self.max_low_level_steps_per_episode
        if not primitive.done:
            if budget_limit:
                primitive.status = "budget_exhausted"
            elif terminated:
                primitive.status = "terminated"
            elif truncated:
                primitive.status = "truncated"
            else:
                primitive.status = "timeout"

        self.primitive_id += 1
        episode_limit = (self.low_level_steps >= self.max_ee_steps_per_episode
                         if self.action_mode == "ee_delta"
                         else self.primitive_id >= self.max_primitives_per_episode)
        if not terminated and (episode_limit or budget_limit):
            truncated = True

        record.status = primitive.status
        record.termination_reason = primitive.status
        record.final_stage = primitive.stage
        record.terminated = bool(terminated)
        record.truncated = bool(truncated)
        record.outcome_metrics = primitive.result(context)
        if self.action_mode == "ee_delta":
            self._direct_grasp_frames = dict(primitive.stable_grasp_frames)
            lifted, displacements = update_cumulative_lifts(
                self._direct_lift_reference_heights,
                record.outcome_metrics["initial_object_positions_root"],
                record.outcome_metrics["final_object_positions_root"],
                self._direct_grasp_frames, record.outcome_metrics["held_object_ids"],
            )
            record.outcome_metrics["lifted_object_ids"] = lifted
            record.outcome_metrics["lift_reference_heights_root"] = dict(self._direct_lift_reference_heights)
            record.outcome_metrics["object_lift_displacements"] = displacements
            record.metadata["lift_metric"] = "continuous_grasp_height_v2"
        record.validate()
        self.records.append(record)
        self.last_primitive_record = record
        self._needs_reset = bool(terminated or truncated)
        info = {
            "primitive_done": bool(primitive.done),
            "primitive_completed": record.status == "success",
            "primitive_timeout": record.status == "timeout",
            "primitive_interrupted": record.status in {
                "budget_exhausted", "terminated", "truncated"
            },
            "primitive_id": record.primitive_id,
            "primitive_type": record.primitive_type,
            "duration_low_steps": record.duration_low_steps,
            "episode_limit_reached": bool(episode_limit),
            "low_level_budget_reached": bool(budget_limit),
            "primitive_record": record,
        }
        if self.debug_base_reward:
            info["base_env_reward_debug"] = float(np.sum(base_rewards))
        return observation, 0.0, bool(terminated), bool(truncated), info
