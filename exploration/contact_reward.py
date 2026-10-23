"""Object-agnostic contact reward for learning a RAPS exploration policy.

The wrapped environment must already expose one completed RAPS primitive as
one Gymnasium step (normally :class:`RAPSEnvWrapper`).  This wrapper converts
the primitive diagnostics into a scalar reward without selecting a target
object or changing how the primitive is executed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from gymnasium import spaces

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - compatibility with older RAPS setups
    import gym


@dataclass(frozen=True)
class ContactExplorationRewardConfig:
    """Weights for a reward maximized by a standard RL algorithm."""

    approach_progress: float = 0.5
    contact_started: float = 2.0
    new_object_contact: float = 5.0
    meaningful_contact: float = 0.2
    grasp_alignment: float = 1.0
    grasp_tcp_offset: float = 0.045
    grasped_object: float = 2.0
    lifted_object: float = 3.0
    moved_object: float = 3.0
    moved_object_requires_grasp: bool = True
    primitive_duration: float = 0.005
    primitive_timeout: float = 0.5
    ee_outside: float = 1.0
    contact_force_threshold: float = 1e-4
    terminate_when_all_lifted: bool = True
    terminate_on_object_failure: bool = True


def interaction_reward_config(config: ContactExplorationRewardConfig):
    """R1 reward used by diagnostic conditions B/D and the D training preset."""
    return replace(config, grasp_tcp_offset=0.0, grasped_object=0.0,
                   lifted_object=0.0, moved_object_requires_grasp=False)


def _contact_ids(snapshot: dict[str, Any], threshold: float) -> set[str]:
    contacted = set()
    for object_id, state in snapshot.get("objects", {}).items():
        if max(float(state.get("left_force", 0.0)), float(state.get("right_force", 0.0))) > threshold:
            contacted.add(str(object_id))
    return contacted


def _meaningful_contact_ids(
    snapshots: list[dict[str, Any]], threshold: float
) -> set[str]:
    """Return objects grasped or touched by both fingertips in one frame."""

    meaningful = set()
    for snapshot in snapshots:
        for object_id, state in snapshot.get("objects", {}).items():
            left = float(state.get("left_force", 0.0))
            right = float(state.get("right_force", 0.0))
            if bool(state.get("grasped", False)) or (
                left > threshold and right > threshold
            ):
                meaningful.add(str(object_id))
    return meaningful


def _nearest_object_distance(
    context: Any, exclude_ids: set[str] | None = None
) -> float | None:
    """Distance to the nearest object not already in *exclude_ids*.

    If all objects are excluded (e.g. both items already lifted) falls back
    to the global nearest so the shaping signal stays non-zero.
    """
    objects = getattr(context, "objects", {})
    if not objects:
        return None
    ee = np.asarray(context.ee_position, dtype=np.float32)
    candidates = {
        oid: state for oid, state in objects.items()
        if exclude_ids is None or oid not in exclude_ids
    }
    # Fall back to all objects when every target has been handled.
    if not candidates:
        candidates = objects
    return min(
        float(np.linalg.norm(ee - np.asarray(state.position_root, dtype=np.float32)))
        for state in candidates.values()
    )


def _grasp_alignment(
    context: Any, exclude_ids: set[str] | None = None, grasp_tcp_offset: float = 0.045
) -> float:
    """Score proximity to the nearest **unfinished** object's side-grasp pose.

    Objects in *exclude_ids* are skipped so that, after the first item is
    lifted, the gradient points toward the remaining target.
    """
    objects = getattr(context, "objects", {})
    if not objects:
        return 0.0
    ee = np.asarray(context.ee_position, dtype=np.float32)
    candidates = {
        oid: state for oid, state in objects.items()
        if exclude_ids is None or oid not in exclude_ids
    }
    if not candidates:
        candidates = objects
    distances = []
    for state in candidates.values():
        target = np.asarray(state.position_root, dtype=np.float32).copy()
        # With the horizontal reset orientation, the finger pads extend about
        # 4.5 cm along robot-base +x from the TCP.
        target[0] -= grasp_tcp_offset
        distances.append(float(np.linalg.norm(ee - target)))
    return float(np.exp(-12.0 * min(distances)))


class ContactExplorationRewardWrapper(gym.Wrapper):
    """Attach an object-agnostic contact reward to high-level RAPS steps.

    Positive terms reward approach and new contact.  Duration, timeout, and an
    end-effector position outside the environment's exploration region are
    penalties.  The complete calculation is returned in
    ``info['exploration_reward']`` for auditing and policy debugging.
    """

    def __init__(
        self,
        env,
        config: ContactExplorationRewardConfig | None = None,
    ):
        super().__init__(env)
        self.config = config or ContactExplorationRewardConfig()
        self.ever_contacted_object_ids: set[str] = set()
        self.ever_grasped_object_ids: set[str] = set()
        self.ever_lifted_object_ids: set[str] = set()
        self.ever_moved_object_ids: set[str] = set()
        self.ever_rewarded_moved_object_ids: set[str] = set()

        # Ordered list of cabinet item names for a stable ever-lifted vector.
        raw_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        cabinet_items = getattr(raw_env, "cabinet_items", [])
        self._tracked_object_ids: list[str] = [
            getattr(item, "name", f"cabinet_item_{i:02d}")
            for i, item in enumerate(cabinet_items)
        ]
        # Extend the observation space to include the ever-lifted status bits.
        n = len(self._tracked_object_ids)
        if n > 0:
            base_obs = self.env.observation_space
            if isinstance(base_obs, spaces.Box) and base_obs.shape:
                low = np.concatenate(
                    [base_obs.low.flatten(), np.zeros(n, dtype=np.float32)]
                )
                high = np.concatenate(
                    [base_obs.high.flatten(), np.ones(n, dtype=np.float32)]
                )
                self.observation_space = spaces.Box(
                    low=low, high=high, dtype=np.float32
                )

    def _append_lifted_bits(self, observation) -> np.ndarray:
        """Append a binary ever-lifted vector to a flat observation array."""
        n = len(self._tracked_object_ids)
        if n == 0:
            return observation
        if isinstance(observation, np.ndarray):
            obs_flat = observation.flatten().astype(np.float32)
        else:
            import torch
            obs_flat = np.asarray(
                observation.detach().cpu() if isinstance(observation, torch.Tensor)
                else observation,
                dtype=np.float32,
            ).flatten()
        lifted_bits = np.array(
            [1.0 if oid in self.ever_lifted_object_ids else 0.0
             for oid in self._tracked_object_ids],
            dtype=np.float32,
        )
        return np.concatenate([obs_flat, lifted_bits])

    def _context(self):
        context_factory = getattr(self.env, "_context", None)
        if context_factory is None:
            raise TypeError(
                "ContactExplorationRewardWrapper requires an environment "
                "with RAPSEnvWrapper._context()"
            )
        return context_factory()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.ever_contacted_object_ids = _contact_ids(
            self._context().contact_snapshot(),
            self.config.contact_force_threshold,
        )
        self.ever_grasped_object_ids = set()
        self.ever_lifted_object_ids = set()
        self.ever_moved_object_ids = set()
        self.ever_rewarded_moved_object_ids = set()
        # Inject lifted bits (all zero at episode start) into the observation.
        if isinstance(result, tuple):
            obs, info = result
            return self._append_lifted_bits(obs), info
        return self._append_lifted_bits(result)

    def _ee_outside(self) -> bool:
        raw_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        contains = getattr(raw_env, "is_ee_inside_exploration_region", None)
        return bool(not contains()) if contains is not None else False

    def step(self, action):
        before_context = self._context()
        # A contact grasp is unfinished until the object has been lifted.
        already_lifted = set(self.ever_lifted_object_ids)
        distance_before = _nearest_object_distance(before_context, already_lifted)
        alignment_before = _grasp_alignment(
            before_context, already_lifted, self.config.grasp_tcp_offset
        )
        contact_before = _contact_ids(
            before_context.contact_snapshot(), self.config.contact_force_threshold
        )

        observation, _, terminated, truncated, info = self.env.step(action)
        record = info.get("primitive_record")
        if record is None:
            raise RuntimeError("RAPS step did not provide info['primitive_record']")

        after_context = self._context()
        distance_after = _nearest_object_distance(after_context, already_lifted)
        alignment_after = _grasp_alignment(
            after_context, already_lifted, self.config.grasp_tcp_offset
        )
        if distance_before is None or distance_after is None:
            approach_progress = 0.0
        else:
            # Signed progress: moving closer is positive and moving away is
            # negative. Clipping prevents one pathological transition from
            # overwhelming discrete contact events.
            approach_progress = float(
                np.clip(distance_before - distance_after, -0.25, 0.25)
            )
        alignment_progress = float(
            np.clip(alignment_after - alignment_before, -1.0, 1.0)
        )

        snapshots = list(record.contact_events)
        contacted_this_primitive = set().union(
            *(
                _contact_ids(snapshot, self.config.contact_force_threshold)
                for snapshot in snapshots
            )
        ) if snapshots else set()
        new_contact_ids = contacted_this_primitive - self.ever_contacted_object_ids
        contact_started = bool(contacted_this_primitive - contact_before)
        meaningful_ids = _meaningful_contact_ids(
            snapshots, self.config.contact_force_threshold
        )
        self.ever_contacted_object_ids.update(contacted_this_primitive)

        outcome_metrics = record.outcome_metrics or {}
        grasped_ids = set(map(str, outcome_metrics.get("grasped_object_ids", [])))
        held_ids = set(map(str, outcome_metrics.get("held_object_ids", [])))
        lifted_ids = set(map(str, outcome_metrics.get("lifted_object_ids", [])))
        moved_ids = set(map(str, outcome_metrics.get("moved_object_ids", [])))
        new_grasped_ids = grasped_ids - self.ever_grasped_object_ids
        new_lifted_ids = lifted_ids - self.ever_lifted_object_ids
        new_moved_ids = moved_ids - self.ever_moved_object_ids
        self.ever_grasped_object_ids.update(grasped_ids)
        self.ever_lifted_object_ids.update(lifted_ids)
        self.ever_moved_object_ids.update(moved_ids)
        grasp_eligible_moved_ids = moved_ids & held_ids
        newly_rewardable_moved_ids = (
            moved_ids - self.ever_rewarded_moved_object_ids
        )
        rewarded_moved_ids = (
            newly_rewardable_moved_ids & grasp_eligible_moved_ids
            if self.config.moved_object_requires_grasp
            else newly_rewardable_moved_ids
        )
        self.ever_rewarded_moved_object_ids.update(rewarded_moved_ids)
        object_ids = set(self._context().objects)
        all_grasped = bool(object_ids) and object_ids <= self.ever_grasped_object_ids
        all_lifted = bool(object_ids) and object_ids <= self.ever_lifted_object_ids
        raw_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        failed_ids = sorted(
            map(str, getattr(raw_env, "failed_object_ids", lambda: [])())
        )

        duration = int(record.duration_low_steps)
        timeout = bool(info.get("primitive_timeout", False))
        ee_outside = self._ee_outside()
        terms = {
            "approach_progress": self.config.approach_progress * approach_progress,
            "grasp_alignment": self.config.grasp_alignment * alignment_progress,
            "contact_started": self.config.contact_started * float(contact_started),
            "new_object_contact": self.config.new_object_contact * len(new_contact_ids),
            "meaningful_contact": self.config.meaningful_contact * len(meaningful_ids),
            "grasped_object": self.config.grasped_object * len(new_grasped_ids),
            "lifted_object": self.config.lifted_object * len(new_lifted_ids),
            "moved_object": self.config.moved_object * len(rewarded_moved_ids),
            "primitive_duration": -self.config.primitive_duration * duration,
            "primitive_timeout": -self.config.primitive_timeout * float(timeout),
            "ee_outside": -self.config.ee_outside * float(ee_outside),
        }
        reward = float(sum(terms.values()))
        observation = self._append_lifted_bits(observation)
        info["exploration_reward"] = {
            "total": reward,
            "raw": {
                "approach_progress": approach_progress,
                "grasp_alignment": alignment_progress,
                "contact_started": contact_started,
                "new_object_contact_count": len(new_contact_ids),
                "meaningful_contact_count": len(meaningful_ids),
                "grasped_object_count": len(grasped_ids),
                "new_grasped_object_count": len(new_grasped_ids),
                "lifted_object_count": len(lifted_ids),
                "new_lifted_object_count": len(new_lifted_ids),
                "moved_object_count": len(moved_ids),
                "rewarded_moved_object_count": len(rewarded_moved_ids),
                "primitive_duration": duration,
                "primitive_timeout": timeout,
                "ee_outside": ee_outside,
                "all_objects_grasped": all_grasped,
                "all_objects_lifted": all_lifted,
                "failed_object_count": len(failed_ids),
            },
            "terms": terms,
            "contacted_object_ids": sorted(contacted_this_primitive),
            "new_object_ids": sorted(new_contact_ids),
            "ever_contacted_object_ids": sorted(self.ever_contacted_object_ids),
            "grasped_object_ids": sorted(grasped_ids),
            "held_object_ids": sorted(held_ids),
            "lifted_object_ids": sorted(lifted_ids),
            "moved_object_ids": sorted(moved_ids),
            "new_grasped_object_ids": sorted(new_grasped_ids),
            "new_lifted_object_ids": sorted(new_lifted_ids),
            "new_moved_object_ids": sorted(new_moved_ids),
            "rewarded_moved_object_ids": sorted(rewarded_moved_ids),
            "ever_grasped_object_ids": sorted(self.ever_grasped_object_ids),
            "ever_lifted_object_ids": sorted(self.ever_lifted_object_ids),
            "ever_moved_object_ids": sorted(self.ever_moved_object_ids),
            "ever_rewarded_moved_object_ids": sorted(
                self.ever_rewarded_moved_object_ids
            ),
            "exploration_complete": all_lifted,
            "failed_object_ids": failed_ids,
            "weights": asdict(self.config),
        }
        if all_lifted and self.config.terminate_when_all_lifted:
            terminated = True
        if failed_ids and self.config.terminate_on_object_failure:
            terminated = True
        return observation, reward, terminated, truncated, info
