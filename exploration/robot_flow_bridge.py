"""Simulator-free nominal robot-flow prediction for RAPS candidates.

This first bridge intentionally predicts only the geometric motion induced by
the primitive parameters.  It does not query future simulator states.  The
learned qpos/FK bridge can replace it later without changing the loop API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from domain.robocasa.primitives import RAPS_ACTION_SCHEMA


@dataclass
class NominalBridgeConfig:
    horizon: int = 11
    workspace_low: tuple[float, float, float] = (-1.0, -1.0, -0.3)
    workspace_high: tuple[float, float, float] = (1.0, 1.0, 1.0)


class NominalRAPSRobotFlowBridge:
    """Predict robot point trajectories using only current robot points + action."""

    def __init__(self, robot_points_fn, config: NominalBridgeConfig | None = None):
        self.robot_points_fn = robot_points_fn
        self.config = config or NominalBridgeConfig()

    @staticmethod
    def _waypoints(action: np.ndarray) -> list[np.ndarray]:
        primitive = RAPS_ACTION_SCHEMA.decode(action)
        p = primitive.parameters
        zero = np.zeros(3, dtype=np.float32)
        if primitive.primitive_type == "move_delta_ee_pose":
            return [zero, np.array([p["dx"], p["dy"], p["dz"]], dtype=np.float32)]
        if primitive.primitive_type == "lift":
            return [zero, np.array([0.0, 0.0, abs(p["z_dist"])], dtype=np.float32)]
        if primitive.primitive_type == "drop":
            return [zero, np.array([0.0, 0.0, -abs(p["z_dist"])], dtype=np.float32)]
        if primitive.primitive_type == "move_left":
            return [zero, np.array([0.0, -abs(p["y_dist"]), 0.0], dtype=np.float32)]
        if primitive.primitive_type == "move_right":
            return [zero, np.array([0.0, abs(p["y_dist"]), 0.0], dtype=np.float32)]
        if primitive.primitive_type == "move_forward":
            return [zero, np.array([abs(p["x_dist"]), 0.0, 0.0], dtype=np.float32)]
        if primitive.primitive_type == "move_backward":
            return [zero, np.array([-abs(p["x_dist"]), 0.0, 0.0], dtype=np.float32)]
        if primitive.primitive_type == "top_grasp":
            target = np.array([0.0, 0.0, -abs(p["z_down"])], dtype=np.float32)
            return [zero, target]
        return [zero, zero]

    def predict(self, observation, actions) -> tuple[torch.Tensor, torch.Tensor]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != RAPS_ACTION_SCHEMA.action_dim:
            raise ValueError(f"actions must have shape [K,{RAPS_ACTION_SCHEMA.action_dim}]")
        points = np.asarray(self.robot_points_fn(observation), dtype=np.float32)
        if points.ndim == 3:
            points = points[0]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("robot_points_fn must return [Nr,3] robot points")
        if not np.isfinite(points).all():
            raise ValueError("current robot points contain NaN or infinity")

        K, T = len(actions), int(self.config.horizon)
        flows = np.empty((K, T, len(points), 3), dtype=np.float32)
        valid = np.ones(K, dtype=bool)
        low = np.asarray(self.config.workspace_low, dtype=np.float32)
        high = np.asarray(self.config.workspace_high, dtype=np.float32)
        for k, action in enumerate(actions):
            waypoints = self._waypoints(action)
            phase = np.linspace(0.0, 1.0, T)
            segment = np.minimum((phase * (len(waypoints) - 1)).astype(int), len(waypoints) - 2)
            local = phase * (len(waypoints) - 1) - segment
            delta = np.stack(waypoints, axis=0)
            offsets = delta[segment] * (1.0 - local[:, None]) + delta[segment + 1] * local[:, None]
            flows[k] = points[None] + offsets[:, None, :]
            valid[k] = bool(np.isfinite(flows[k]).all() and ((flows[k] >= low) & (flows[k] <= high)).all())
        return torch.from_numpy(flows), torch.from_numpy(valid)

    def __call__(self, observation, actions):
        return self.predict(observation, actions)[0]
