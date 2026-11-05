"""Offline cabinet exploration metrics; failed transitions are never discarded."""

from collections import Counter

import numpy as np

from domain.robocasa.primitives import MOVED_OBJECT_THRESHOLD


def motion_bucket(object_id, primitive, displacement):
    displacement = np.asarray(displacement, dtype=np.float64)
    distance = float(np.linalg.norm(displacement))
    if distance < MOVED_OBJECT_THRESHOLD:
        return None
    axis = int(np.argmax(np.abs(displacement)))
    direction = "xyz"[axis] + ("+" if displacement[axis] >= 0 else "-")
    magnitude = "15-30mm" if distance < 0.03 else "30-60mm" if distance < 0.06 else "60mm+"
    return (object_id, primitive, direction, magnitude)


def summarize_records(records):
    """Integrate coverage at real step boundaries, discovering buckets at step end."""
    records = list(records)
    steps = contacts = motions = recoverable_motions = timeouts = failures = outside = 0
    lifted = grasped = bilateral = 0
    raw_buckets, recoverable_buckets = set(), set()
    raw_area = recoverable_area = 0.0
    visits = repeated = 0
    primitives = Counter()
    object_metrics = {}
    parameter_samples = {}
    reward_components = Counter()
    episode_returns = Counter()
    curve = [{"low_level_steps": 0, "raw_coverage": 0, "recoverable_coverage": 0}]
    first_contact = first_motion = None
    episodes, finished_episodes, completed_episodes = set(), set(), set()
    for row in records:
        duration = int(row["duration_low_steps"])
        raw_area += len(raw_buckets) * duration
        recoverable_area += len(recoverable_buckets) * duration
        steps += duration
        primitive = row["primitive_type"]
        primitives[primitive] += 1
        for name, value in row["parameters_physical"].items():
            parameter_samples.setdefault(primitive, {}).setdefault(name, []).append(float(value))
        episode = row["episode_id"]
        episodes.add(episode)
        if (row["terminated"] or row["truncated"]) and not row["run_budget_exhausted"]:
            finished_episodes.add(episode)
        report = row["reward"]
        episode_returns[str(episode)] += report["total"]
        reward_components.update(report["terms"])
        outcomes = row["outcome_metrics"]
        failed_ids = set(row["after"]["failed_object_ids"])
        contacted_ids = set(report["contacted_object_ids"])
        contacts += bool(contacted_ids)
        bilateral += report["raw"]["meaningful_contact_count"] > 0
        grasped += bool(outcomes["grasped_object_ids"])
        lifted += bool(outcomes["lifted_object_ids"])
        timeouts += row["status"] == "timeout"
        failures += bool(failed_ids)
        outside += not row["after"]["inside_region"]
        if report["exploration_complete"]:
            completed_episodes.add(episode)
        if contacted_ids and first_contact is None:
            threshold = report["weights"]["contact_force_threshold"]
            detected = next((i + 1 for i, event in enumerate(row["contact_events"])
                             if any(max(obj["left_force"], obj["right_force"]) > threshold
                                    for obj in event.get("objects", {}).values())), duration)
            first_contact = steps - duration + detected
        moved_ids, recoverable_ids = set(), set()
        for object_id, before in row["before"]["objects"].items():
            stats = object_metrics.setdefault(object_id, {
                "contact_primitives": 0, "motion_primitives": 0,
                "recoverable_motion_primitives": 0, "lift_primitives": 0,
            })
            stats["contact_primitives"] += object_id in contacted_ids
            stats["lift_primitives"] += object_id in outcomes["lifted_object_ids"]
            after = row["after"]["objects"].get(object_id)
            if after is None:
                continue
            displacement = np.asarray(after["position_world"]) - before["position_world"]
            bucket = motion_bucket(object_id, primitive, displacement)
            if bucket is None:
                continue
            moved_ids.add(object_id)
            stats["motion_primitives"] += 1
            raw_buckets.add(bucket)
            if object_id not in failed_ids:
                recoverable_ids.add(object_id)
                stats["recoverable_motion_primitives"] += 1
                visits += 1
                repeated += bucket in recoverable_buckets
                recoverable_buckets.add(bucket)
        motions += bool(moved_ids)
        recoverable_motions += bool(recoverable_ids)
        if moved_ids and first_motion is None:
            first_motion = steps
        curve.append({
            "low_level_steps": steps, "raw_coverage": len(raw_buckets),
            "recoverable_coverage": len(recoverable_buckets),
        })
    count = len(records)
    rate = lambda number: number / count if count else 0.0
    per_thousand = lambda number: 1000.0 * number / steps if steps else 0.0
    completed_finished = len(completed_episodes & finished_episodes)
    return {
        "low_level_steps": steps, "primitives": count,
        "contact_primitives_per_1000_steps": per_thousand(contacts),
        "motion_primitives_per_1000_steps": per_thousand(motions),
        "recoverable_motion_primitives_per_1000_steps": per_thousand(recoverable_motions),
        "contact_primitive_rate": rate(contacts), "bilateral_primitive_rate": rate(bilateral),
        "grasp_primitive_rate": rate(grasped), "lift_primitive_rate": rate(lifted),
        "timeout_primitive_rate": rate(timeouts), "failed_state_primitive_rate": rate(failures),
        "endpoint_outside_primitive_rate": rate(outside),
        "first_contact_detected_step": first_contact,
        "first_motion_upper_bound_steps": first_motion,
        "raw_coverage": len(raw_buckets), "recoverable_coverage": len(recoverable_buckets),
        "raw_coverage_auc": raw_area / steps if steps else 0.0,
        "recoverable_coverage_auc": recoverable_area / steps if steps else 0.0,
        "recoverable_bucket_repeat_fraction": repeated / visits if visits else 0.0,
        "episodes_observed": len(episodes), "episodes_finished": len(finished_episodes),
        "episodes_censored": len(episodes - finished_episodes),
        "completed_episodes": completed_finished,
        "episode_completion_rate": completed_finished / len(finished_episodes) if finished_episodes else 0.0,
        "primitive_counts": dict(primitives), "per_object": object_metrics,
        "total_reward": sum(episode_returns.values()),
        "reward_per_1000_steps": per_thousand(sum(episode_returns.values())),
        "episode_returns": dict(episode_returns), "reward_component_totals": dict(reward_components),
        "parameters_by_primitive": {
            primitive: {name: {"mean": float(np.mean(values)), "std": float(np.std(values)),
                               "min": min(values), "max": max(values), "count": len(values)}
                        for name, values in parameters.items()}
            for primitive, parameters in parameter_samples.items()
        },
        "coverage_curve": curve,
    }


def compare_arms(summaries):
    """Single-seed screening, without rewarding failures or ranking by return."""
    baseline = summaries.get("A")
    if baseline is None:
        return {"status": "incomplete", "candidates": [], "reason": "A evaluation is missing"}
    if len({summary["low_level_steps"] for summary in summaries.values()}) != 1:
        return {"status": "incomparable", "candidates": [], "reason": "Evaluation step budgets differ"}
    candidates = []
    expected_objects = {"cabinet_item_16", "cabinet_item_17"}
    for arm, summary in summaries.items():
        both_moved = expected_objects and all(
            summary["per_object"].get(name, {}).get("recoverable_motion_primitives", 0) > 0
            for name in expected_objects
        )
        if arm != "A" and both_moved and (
            summary["recoverable_coverage_auc"] > baseline["recoverable_coverage_auc"]
            and summary["recoverable_motion_primitives_per_1000_steps"]
            > baseline["recoverable_motion_primitives_per_1000_steps"]
        ):
            candidates.append(arm)
    contrasts = {}
    for left, right, label in (
        ("B", "A", "reward_P0"), ("D", "C", "reward_P1"),
        ("C", "A", "projection_R0"), ("D", "B", "projection_R1"),
    ):
        if left in summaries and right in summaries:
            contrasts[label] = {
                key: summaries[left][key] - summaries[right][key]
                for key in ("recoverable_coverage_auc", "recoverable_motion_primitives_per_1000_steps")
            }
    return {
        "status": "screening_pass" if candidates else "not_passed",
        "candidates": sorted(candidates), "contrasts": contrasts,
        "interpretation": "interaction_suspected" if candidates == ["D"] else "inspect_factor_contrasts",
        "limitation": "Single seed and fixed layout; no significance or world-model error claim.",
    }
