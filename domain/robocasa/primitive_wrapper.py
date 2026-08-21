"""Robot-aware adapter for executing one RAPS primitive as one environment step.

Why this wrapper exists
-----------------------
RAPS does not expose the simulator's low-level controller to the policy.  The
policy emits one hybrid high-level action, for example
``top_grasp(z_down=0.20)``.  RoboCasa, however, must receive many low-level
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

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - compatibility for older RAPS setups
    import gym
    from gym import spaces

from .primitives import (
    PRIMITIVE_CLASSES,
    PrimitiveContext,
    RAPS_ACTION_SCHEMA,
    normalize_primitive_robot,
)
from exploration.trajectory_recorder import PrimitiveRecord


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
        debug_base_reward: bool = False,
        observation_transform=None,
        robot: str = "fetch",
        render_during_primitive: bool = False,
        record_dense_trajectory: bool = True,
        contact_check_interval: int = 1,
    ):
        if int(max_primitives_per_episode) < 1:
            raise ValueError("max_primitives_per_episode must be positive")
        if int(max_low_level_steps_per_episode) < 1:
            raise ValueError("max_low_level_steps_per_episode must be positive")
        if int(contact_check_interval) < 1:
            raise ValueError("contact_check_interval must be positive")
        super().__init__(env)
        self.action_space = spaces.Box(
            low=RAPS_ACTION_SCHEMA.low,
            high=RAPS_ACTION_SCHEMA.high,
            dtype=np.float32,
        )
        self.max_primitives_per_episode = int(max_primitives_per_episode)
        self.max_low_level_steps_per_episode = int(max_low_level_steps_per_episode)
        self.debug_base_reward = bool(debug_base_reward)
        self.observation_transform = observation_transform
        self.robot = normalize_primitive_robot(robot)
        self.render_during_primitive = bool(render_during_primitive)
        self.record_dense_trajectory = bool(record_dense_trajectory)
        self.contact_check_interval = int(contact_check_interval)
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
        # Panda pd_ee_delta_pose: [dxyz, rotation-vector, gripper]. Keeping the
        # rotation delta at zero preserves the wrist pose during a primitive.
        return np.array(
            [action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]],
            dtype=np.float32,
        )

    def _context(self):
        return PrimitiveContext(
            self.env,
            robot=self.robot,
            record_state=self.record_dense_trajectory,
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
        if isinstance(result, tuple):
            result = (self._transform_observation(result[0]), *result[1:])
            self._last_observation = result[0]
        else:
            result = self._transform_observation(result)
            self._last_observation = result
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
        if action.shape != (RAPS_ACTION_SCHEMA.action_dim,):
            raise ValueError(
                f"RAPS action must have shape {(RAPS_ACTION_SCHEMA.action_dim,)}, got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError("RAPS action contains NaN or infinity")
        action = np.clip(action, RAPS_ACTION_SCHEMA.low, RAPS_ACTION_SCHEMA.high)
        primitive_action = RAPS_ACTION_SCHEMA.decode(action)
        primitive_cls = PRIMITIVE_CLASSES[primitive_action.primitive_type]
        primitive = primitive_cls(primitive_action)

        observation = getattr(self, "_last_observation", None)
        if observation is None:
            observation = self.env.get_observation() if hasattr(self.env, "get_observation") else None
        context = self._context()
        primitive.reset(context)
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
                    low_action, contact_event=contact_event
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
        episode_limit = self.primitive_id >= self.max_primitives_per_episode
        if not terminated and (episode_limit or budget_limit):
            truncated = True

        record.status = primitive.status
        record.termination_reason = primitive.status
        record.final_stage = primitive.stage
        record.terminated = bool(terminated)
        record.truncated = bool(truncated)
        record.outcome_metrics = primitive.result(context)
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
