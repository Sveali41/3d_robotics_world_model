"""Interaction and prediction evaluation for the state-based CEE-US diagnostic."""

import numpy as np
from scipy.spatial.transform import Rotation

from exploration.diagnostic_metrics import summarize_records


def interaction_summary(rows):
    result = summarize_records(rows)
    result["requested_action_mean"] = (np.mean([r["requested_action"] for r in rows], axis=0).tolist()
                                       if rows else [])
    diagnostics = [r["planner"] for r in rows if r.get("planner")]
    if diagnostics:
        result["planner"] = {
            "decisions": len(diagnostics),
            "planning_seconds_mean": float(np.mean([d["planning_seconds"] for d in diagnostics])),
            "candidate_rollout_seconds_mean": float(np.mean([d["candidate_rollout_seconds"] for d in diagnostics])),
            "curiosity_score_mean": float(np.mean([d["curiosity_score"] for d in diagnostics])),
            "score_decomposition_mean": {
                key: float(np.mean([d["score_decomposition"][key] for d in diagnostics]))
                for key in ("joint", "tcp", "door", "objects")},
        }
        heuristics = [d["object_heuristic_score"] for d in diagnostics if "object_heuristic_score" in d]
        if heuristics:
            result["planner"]["object_heuristic_score_mean"] = {
                key: float(np.mean([h[key] for h in heuristics]))
                for key in ("approach_progress_score", "combined_score")}
    observed = [r["object_heuristic_observed"] for r in rows if "object_heuristic_observed" in r]
    if observed:
        result["observed_object_heuristic_score_total"] = float(sum(
            h["approach_progress_score"] for h in observed))
    return result


def screen(random_summary, curiosity_summary):
    if random_summary["low_level_steps"] != curiosity_summary["low_level_steps"]:
        return {"status": "incomparable", "reason": "Real step budgets differ"}
    both_moved = all(curiosity_summary["per_object"].get(name, {}).get(
        "recoverable_motion_primitives", 0) > 0 for name in ("cabinet_item_16", "cabinet_item_17"))
    efficiency = "recoverable_motion_primitives_per_1000_steps"
    coverage = "recoverable_coverage_auc"
    passed = (both_moved and curiosity_summary[efficiency] > random_summary[efficiency]
              and curiosity_summary[coverage] > random_summary[coverage])
    return {"status": "screening_pass" if passed else "not_passed",
            "both_objects_moved_in_recovery_range": both_moved,
            "motion_efficiency_delta": curiosity_summary[efficiency] - random_summary[efficiency],
            "coverage_auc_delta": curiosity_summary[coverage] - random_summary[coverage],
            "limitation": "Single seed, fixed layout, privileged state; no grasp or visual-generalization claim"}


def _errors(prediction, target, agent_dim=32):
    """Physical errors; velocity channels are intentionally not mixed into pose errors."""
    positions = [slice(agent_dim, agent_dim + 3), slice(agent_dim + 12, agent_dim + 15)]
    rotations = [slice(agent_dim + 3, agent_dim + 6), slice(agent_dim + 15, agent_dim + 18)]
    position = np.stack([np.linalg.norm(prediction[..., s] - target[..., s], axis=-1)
                         for s in positions], axis=-1)
    angular = []
    for s in rotations:
        p = Rotation.from_euler("xyz", prediction[..., s].reshape(-1, 3))
        t = Rotation.from_euler("xyz", target[..., s].reshape(-1, 3))
        angular.append((p.inv() * t).magnitude().reshape(prediction.shape[:-1]))
    return position, np.stack(angular, axis=-1)


def prediction_metrics(backend, replay, horizons=(1, 10, 30), batch_size=16):
    """Evaluate on supplied held-out transitions, never across episode/reset boundaries."""
    states, next_states = replay["states"], replay["next_states"]
    actions, episodes = replay["actions"], replay["episode_ids"]
    agent_dim = getattr(getattr(backend, "env", None), "agent_dim", 32)
    physical_indices = np.r_[np.arange(32), np.arange(agent_dim, agent_dim + 24)]
    result = {}
    for horizon in horizons:
        starts = [i for i in range(len(states) - horizon + 1)
                  if np.all(episodes[i:i + horizon] == episodes[i])]
        if not starts:
            result[str(horizon)] = {"windows": 0, "status": "insufficient_contiguous_data"}
            continue
        errors, baseline_errors, angles, baseline_angles, moved = [], [], [], [], []
        disagreements, actual_errors = [], []
        for offset in range(0, len(starts), batch_size):
            indices = starts[offset:offset + batch_size]
            initial = states[indices]
            sequence = np.stack([actions[i:i + horizon] for i in indices])
            predictions = np.asarray(backend.rollout(initial, sequence))
            expected = (5, len(indices), horizon, initial.shape[-1])
            if predictions.shape != expected or not np.isfinite(predictions).all():
                raise ValueError(f"Invalid held-out rollout: {predictions.shape}, expected {expected}")
            target = np.stack([next_states[i:i + horizon] for i in indices])
            mean = predictions.mean(axis=0)
            baseline = np.broadcast_to(initial[:, None], target.shape)
            ep, ea = _errors(mean[:, -1], target[:, -1], agent_dim)
            bp, ba = _errors(baseline[:, -1], target[:, -1], agent_dim)
            errors.extend(ep); angles.extend(ea)
            baseline_errors.extend(bp); baseline_angles.extend(ba)
            # Same 1.5 cm threshold as the interaction metrics, over this window.
            moved.extend(bp >= 0.015)
            if horizon == 1:
                disagreements.extend(predictions.std(axis=0, ddof=1)[..., physical_indices].sum(axis=(-1, -2)))
                actual_errors.extend(np.sqrt(np.mean((mean[..., physical_indices] - target[..., physical_indices]) ** 2,
                                                     axis=(-1, -2))))
        errors, baseline_errors = np.asarray(errors), np.asarray(baseline_errors)
        angles, baseline_angles, moved = np.asarray(angles), np.asarray(baseline_angles), np.asarray(moved)
        entry = {"windows": len(starts), "object_position_mae_m": errors.mean(axis=0).tolist(),
                 "zero_delta_object_position_mae_m": baseline_errors.mean(axis=0).tolist(),
                 "object_rotation_mae_rad": angles.mean(axis=0).tolist(),
                 "zero_delta_object_rotation_mae_rad": baseline_angles.mean(axis=0).tolist(),
                 "moved_object_windows": moved.sum(axis=0).tolist(),
                 "moving_position_mae_m": [float(errors[:, j][moved[:, j]].mean())
                                           if moved[:, j].any() else None for j in range(2)],
                 "moving_zero_delta_position_mae_m": [float(baseline_errors[:, j][moved[:, j]].mean())
                                                      if moved[:, j].any() else None for j in range(2)]}
        if horizon == 1:
            valid = np.std(disagreements) > 0 and np.std(actual_errors) > 0
            entry["disagreement_error_pearson"] = (float(np.corrcoef(disagreements, actual_errors)[0, 1])
                                                     if valid else None)
        result[str(horizon)] = entry
    result["interaction_dynamics_status"] = (
        "moving_samples_available" if any(any(v.get("moved_object_windows", []))
                                           for v in result.values() if isinstance(v, dict))
        else "cannot_validate_interaction_dynamics_without_moving_samples")
    return result
