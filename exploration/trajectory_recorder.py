"""Dense, reward-free trajectory records for RAPS primitive execution.

The original RAPS wrapper exposes one high-level transition while the
primitive itself performs many simulator steps.  PointWorld needs those
intermediate states, so this module keeps both views: the complete low-level
trajectory and a phase-resampled T=11 view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import copy
import numpy as np

RAPS_SAMPLE_SCHEMA_VERSION = "robocasa-raps-v1"


def _copy(value: Any) -> Any:
    """Copy numpy/tensor observations without imposing an observation format."""
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return value.detach().cpu().clone()
    except AttributeError:
        return copy.deepcopy(value)


@dataclass
class PrimitiveRecord:
    episode_id: int
    primitive_id: int
    primitive_type: str
    raps_action_normalized: np.ndarray
    parameters_physical: dict[str, float] = field(default_factory=dict)
    observations: list[Any] = field(default_factory=list)
    low_actions: list[np.ndarray] = field(default_factory=list)
    full_controller_actions: list[np.ndarray] = field(default_factory=list)
    contact_events: list[dict[str, Any]] = field(default_factory=list)
    qpos: list[Any] = field(default_factory=list)
    qvel: list[Any] = field(default_factory=list)
    ee_pose: list[Any] = field(default_factory=list)
    gripper_open: list[float] = field(default_factory=list)
    base_pose: list[Any] = field(default_factory=list)
    body_qpos: list[Any] = field(default_factory=list)
    object_poses: list[dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    status: str = "running"
    termination_reason: str = "running"
    final_stage: str = "start"
    outcome_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RAPS_SAMPLE_SCHEMA_VERSION
    dense_trajectory: bool = True

    @property
    def duration_low_steps(self) -> int:
        return len(self.low_actions)

    def append_observation(self, observation: Any, context: Any = None) -> None:
        self.observations.append(_copy(observation))
        if context is not None:
            self.qpos.append(_copy(getattr(context, "qpos", np.empty(0))))
            self.qvel.append(_copy(getattr(context, "qvel", np.empty(0))))
            self.ee_pose.append(_copy(getattr(context, "ee_pose", np.empty(0))))
            self.gripper_open.append(
                float(getattr(context, "gripper_open", np.nan))
            )
            self.base_pose.append(
                _copy(getattr(context, "base_pose", np.empty(0)))
            )
            self.body_qpos.append(
                _copy(getattr(context, "body_qpos", np.empty(0)))
            )
            self.object_poses.append({
                object_id: {
                    "scene_id": int(state.scene_id),
                    "pose": _copy(state.actor.pose.to_transformation_matrix()[0]),
                }
                for object_id, state in getattr(context, "objects", {}).items()
            })
        self._validate_prefix_lengths()

    def append_transition(
        self,
        low_action: np.ndarray,
        *,
        full_controller_action: Any = None,
        contact_event: dict[str, Any] | None = None,
    ) -> None:
        action = np.asarray(low_action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError(f"low-level RAPS action must be finite shape (4,), got {action.shape}")
        self.low_actions.append(action.copy())
        if full_controller_action is None:
            full_controller_action = action
        full_action = np.asarray(full_controller_action, dtype=np.float32)
        if full_action.ndim != 1 or not np.isfinite(full_action).all():
            raise ValueError(
                "full controller action must be a finite one-dimensional array, "
                f"got shape {full_action.shape}"
            )
        self.full_controller_actions.append(full_action.copy())
        self.contact_events.append(copy.deepcopy(contact_event or {}))
        self._validate_prefix_lengths()

    def append_compact_transition(
        self,
        low_action: np.ndarray,
        *,
        full_controller_action: Any = None,
        contact_event: dict[str, Any] | None = None,
    ) -> None:
        """Store PPO reward diagnostics without copying a full simulator state."""
        if self.dense_trajectory:
            raise RuntimeError("compact transitions require dense_trajectory=False")
        action = np.asarray(low_action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError(
                f"low-level RAPS action must be finite shape (4,), got {action.shape}"
            )
        full_action = np.asarray(action if full_controller_action is None else full_controller_action,
                                 dtype=np.float32)
        if full_action.ndim != 1 or not np.isfinite(full_action).all():
            raise ValueError("full controller action must be a finite one-dimensional array")
        self.low_actions.append(action.copy())
        self.full_controller_actions.append(full_action.copy())
        self.contact_events.append(copy.deepcopy(contact_event or {}))

    def _validate_prefix_lengths(self) -> None:
        """Check alignment after every append without requiring finalization."""
        transition_lengths = {
            len(self.low_actions),
            len(self.full_controller_actions),
            len(self.contact_events),
        }
        if len(transition_lengths) != 1:
            raise RuntimeError("dense transition fields became misaligned")
        observation_lengths = {
            len(self.observations),
            len(self.qpos),
            len(self.qvel),
            len(self.ee_pose),
            len(self.gripper_open),
            len(self.base_pose),
            len(self.body_qpos),
            len(self.object_poses),
        }
        if len(observation_lengths) != 1:
            raise RuntimeError("dense observation fields became misaligned")
        H = len(self.low_actions)
        observation_count = len(self.observations)
        if observation_count not in {H, H + 1}:
            raise RuntimeError(
                "dense prefix must have H or H+1 observations while appending, "
                f"got H={H}, observations={observation_count}"
            )

    def validate(self, *, finalized: bool = True) -> None:
        """Raise when a primitive record violates its storage contract."""
        action = np.asarray(self.raps_action_normalized)
        expected_shape = (4,) if self.schema_version == "cabinet-ee-delta-v1" else (20,)
        if action.shape != expected_shape or not np.isfinite(action).all():
            raise RuntimeError(
                f"record action must be finite shape {expected_shape}, "
                f"got {action.shape}"
            )
        H = self.duration_low_steps
        if not self.dense_trajectory:
            if len(self.contact_events) != H:
                raise RuntimeError(
                    f"contact_events must contain H={H} entries, got {len(self.contact_events)}"
                )
            if finalized and (
                self.status == "running" or self.termination_reason == "running"
            ):
                raise RuntimeError("finalized primitive record still has running status")
            return
        transition_fields = {
            "full_controller_actions": self.full_controller_actions,
            "contact_events": self.contact_events,
        }
        observation_fields = {
            "observations": self.observations,
            "qpos": self.qpos,
            "qvel": self.qvel,
            "ee_pose": self.ee_pose,
            "gripper_open": self.gripper_open,
            "base_pose": self.base_pose,
            "body_qpos": self.body_qpos,
            "object_poses": self.object_poses,
        }
        for name, values in transition_fields.items():
            if len(values) != H:
                raise RuntimeError(f"{name} must contain H={H} entries, got {len(values)}")
        for name, values in observation_fields.items():
            if len(values) != H + 1:
                raise RuntimeError(
                    f"{name} must contain H+1={H + 1} entries, got {len(values)}"
                )
        if finalized and (self.status == "running" or self.termination_reason == "running"):
            raise RuntimeError("finalized primitive record still has running status")

    def phase_view(self, horizon: int = 11) -> dict[str, Any]:
        if self.schema_version == "cabinet-ee-delta-v1":
            raise ValueError("ee_delta records are not RAPS PointWorld samples; use raw transitions")
        if horizon < 2:
            raise ValueError("horizon must be at least 2")
        if not self.dense_trajectory:
            raise RuntimeError(
                "phase_view requires a dense trajectory; create RAPSEnvWrapper "
                "with record_dense_trajectory=True"
            )
        self.validate(finalized=False)
        H = self.duration_low_steps
        if H < horizon - 1:
            # Preserve every real low-level frame as a prefix, then repeat the
            # final state as explicit right-padding.
            indices = np.concatenate(
                [
                    np.arange(H + 1, dtype=np.int64),
                    np.full(horizon - H - 1, H, dtype=np.int64),
                ]
            )
            valid = np.arange(horizon) <= H
        else:
            indices = np.rint(np.linspace(0, H, horizon)).astype(np.int64)
            valid = np.ones(horizon, dtype=bool)

        transition_valid = (
            valid[:-1] & valid[1:] & (indices[1:] > indices[:-1])
        )
        low_actions = np.zeros((horizon - 1, 4), dtype=np.float32)
        if H:
            low_indices = np.minimum(indices[:-1], H - 1)
            low_actions[transition_valid] = np.asarray(self.low_actions)[
                low_indices[transition_valid]
            ]
        primitive_phase = (
            indices.astype(np.float32) / float(H)
            if H
            else np.zeros(horizon, dtype=np.float32)
        )
        return {
            "schema_version": self.schema_version,
            "raps_action_normalized": self.raps_action_normalized.copy(),
            "primitive_type": self.primitive_type,
            "episode_id": np.int64(self.episode_id),
            "primitive_id": np.int64(self.primitive_id),
            "parameters_physical": copy.deepcopy(self.parameters_physical),
            "primitive_duration": np.int32(H),
            "sample_indices": indices,
            "primitive_phase": primitive_phase,
            "temporal_valid_mask": valid,
            "low_action_valid_mask": transition_valid,
            "observations": [_copy(self.observations[int(i)]) for i in indices],
            "object_poses": [copy.deepcopy(self.object_poses[int(i)]) for i in indices],
            "low_actions": low_actions,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "outcome_metrics": copy.deepcopy(self.outcome_metrics),
            "metadata": copy.deepcopy(self.metadata),
        }
