"""Export and plot measured end-effector paths from primitive collection logs.

This module only reads collector output. It does not construct an environment or
run inference. Positions are robot-root-frame metres as recorded in the logs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECTIONS = ((0, 1), (0, 2), (1, 2))
AXES = "XYZ"
OUTCOMES = (
    "verified_lift",
    "unconfirmed_grasp",
    "planning_rejected",
    "budget_truncated",
    "other_failure",
)
COLORS = {
    "verified_lift": "tab:green",
    "unconfirmed_grasp": "tab:orange",
    "planning_rejected": "tab:red",
    "budget_truncated": "tab:purple",
    "other_failure": "0.45",
}
RESET_OBJECT_COLORS = {"cup": "tab:blue", "bowl": "tab:purple"}
KNOWN_OBJECT_IDS = {"cabinet_item_16": "cup", "cabinet_item_17": "bowl"}


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path}: {exc}") from exc


def _jsonl(path: Path):
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    result = []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        result.append(row)
    return result


def _position(value, context):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected XYZ position in {context}")
    position = np.asarray(value, dtype=np.float64)
    if not np.isfinite(position).all():
        raise ValueError(f"Non-finite XYZ position in {context}")
    return position


def _nested_position(record, side, context):
    try:
        return _position(record[side]["ee_position"], f"{context} {side}.ee_position")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing {side}.ee_position in {context}") from exc


def _goal(summary, context):
    value = summary.get("goal_root")
    if value is None:
        return [None, None, None]
    return _position(value, f"{context} goal_root").tolist()


def _raw_outcome(summary):
    outcome = summary.get("outcome")
    if isinstance(outcome, dict):
        return outcome
    if isinstance(outcome, str):
        return {"status": outcome}
    if outcome is None:
        return {}
    raise ValueError("summary outcome must be an object, string, or null")


def _outcome_category(summary, outcome):
    status = str(outcome.get("status", "")).lower()
    reason = str(outcome.get("reason", "")).lower()
    primitive_status = str(outcome.get("primitive_status", "")).lower()
    if status in {"verified_lift", "lift_verified", "grasp_and_lift_verified"} or outcome.get("verified_lift") is True:
        return "verified_lift"
    if status == "unconfirmed_grasp" or reason == "unconfirmed_grasp":
        return "unconfirmed_grasp"
    if (status in {"budget_exhausted", "budget_truncated", "truncated"}
            or reason in {"budget_exhausted", "budget_truncated"}
            or primitive_status in {"budget_exhausted", "budget_truncated"}):
        return "budget_truncated"
    planning_rejections = {
        "no_collision_free_ik", "no_valid_ik", "no_ik_solution", "ik_rejected",
        "planner_rejected", "planning_rejected", "planner_failed", "planning_failed",
        "no_collision_free_path", "no_path",
    }
    if primitive_status in planning_rejections or status in planning_rejections:
        return "planning_rejected"
    return "other_failure"


def _object_kind(summary, attempt_path):
    value = str(summary.get("object", "")).lower()
    if "cup" in value:
        return "cup"
    if "bowl" in value:
        return "bowl"
    if value == "cabinet_item_16":
        return "cup"
    if value == "cabinet_item_17":
        return "bowl"
    for part in reversed(attempt_path.parts):
        part = part.lower()
        if part in {"cup", "cups"} or "cup" in part:
            return "cup"
        if part in {"bowl", "bowls"} or "bowl" in part:
            return "bowl"
    return str(summary.get("object", "unknown"))


def _geometry_bounds(value, context):
    bounds = np.asarray(value, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"Expected finite [minimum, maximum] XYZ collision bounds in {context}")
    if np.any(bounds[1] <= bounds[0]):
        raise ValueError(f"Invalid collision AABB extents in {context}")
    return bounds


def read_scene_geometry(source: Path):
    """Aggregate only saved robot-root collision bounds from source manifests."""
    source = source.resolve()
    manifests = sorted(source.rglob("manifest.json"))
    scenes = {}
    with_geometry = []
    without_geometry = []
    for path in manifests:
        rel_path = path.relative_to(source).as_posix()
        manifest = _json(path)
        if not isinstance(manifest, dict):
            raise ValueError(f"Expected manifest object in {rel_path}")
        raw_geometry = manifest.get("geometry")
        if raw_geometry is None:
            without_geometry.append(rel_path)
            continue
        if not isinstance(raw_geometry, dict):
            raise ValueError(f"Expected manifest.geometry object in {rel_path}")
        with_geometry.append(rel_path)
        for scene, objects in raw_geometry.items():
            if not isinstance(objects, dict):
                raise ValueError(f"Expected geometry object map for {scene} in {rel_path}")
            scene_data = scenes.setdefault(str(scene), {})
            for object_id, record in objects.items():
                context = f"{rel_path} geometry.{scene}.{object_id}.collision_bounds_root_m"
                if not isinstance(record, dict) or "collision_bounds_root_m" not in record:
                    raise ValueError(f"Missing collision_bounds_root_m in {context}")
                bounds = _geometry_bounds(record["collision_bounds_root_m"], context)
                if object_id in scene_data:
                    prior = np.asarray(scene_data[object_id]["collision_bounds_root_m"], dtype=np.float64)
                    if not np.allclose(prior, bounds, rtol=0., atol=1e-9):
                        prior_sources = ", ".join(scene_data[object_id]["manifest_sources"])
                        raise ValueError(
                            f"Inconsistent collision bounds for {scene}/{object_id}: "
                            f"{prior_sources} and {rel_path}"
                        )
                    scene_data[object_id]["manifest_sources"].append(rel_path)
                else:
                    scene_data[object_id] = {
                        "object": KNOWN_OBJECT_IDS.get(str(object_id), str(object_id)),
                        "collision_bounds_root_m": bounds.tolist(),
                        "manifest_sources": [rel_path],
                    }

    return {
        "available": bool(scenes),
        "position_frame": "robot root",
        "position_units": "metres",
        "source_manifests": with_geometry,
        "missing_geometry_manifests": without_geometry,
        "scenes": scenes,
    }


def _row_groups(attempts):
    object_order = {"cup": 0, "bowl": 1}
    return sorted({(str(attempt["scene"]), str(attempt["object"])) for attempt in attempts},
                  key=lambda pair: (pair[0], object_order.get(pair[1], 2), pair[1]))


def _geometry_for_scene(geometry, scene):
    return geometry.get("scenes", {}).get(str(scene), {})


def _draw_geometry_2d(ax, geometry, scene, x, y):
    from matplotlib.patches import Rectangle

    objects = _geometry_for_scene(geometry, scene)
    for object_id, record in objects.items():
        bounds = np.asarray(record["collision_bounds_root_m"], dtype=np.float64)
        kind = record["object"]
        color = RESET_OBJECT_COLORS.get(kind, "tab:gray")
        lower, upper = bounds
        rect = Rectangle((lower[x], lower[y]), upper[x] - lower[x], upper[y] - lower[y],
                         facecolor=color, edgecolor=color, linewidth=1.2, alpha=.14,
                         zorder=0)
        ax.add_patch(rect)
        center = (lower + upper) / 2
        label = f"{kind.title()} (reset)"
        ax.plot(center[x], center[y], marker="+", markersize=5, color=color,
                linestyle="None", zorder=1)
        ax.annotate(label, (center[x], center[y]), xytext=(3, 3),
                    textcoords="offset points", color=color, fontsize=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .7, "pad": .4},
                    zorder=3)
    return len(objects)


def _draw_geometry_3d(ax, geometry, scene):
    objects = _geometry_for_scene(geometry, scene)
    for _, record in objects.items():
        lower, upper = np.asarray(record["collision_bounds_root_m"], dtype=np.float64)
        color = RESET_OBJECT_COLORS.get(record["object"], "tab:gray")
        corners = np.asarray([[upper[0] if mask & 1 else lower[0],
                               upper[1] if mask & 2 else lower[1],
                               upper[2] if mask & 4 else lower[2]] for mask in range(8)])
        for first in range(8):
            for axis in range(3):
                second = first ^ (1 << axis)
                if first < second:
                    edge = corners[[first, second]]
                    ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=color,
                            alpha=.55, linewidth=1.)
        center = (lower + upper) / 2
        ax.text(*center, f"{record['object'].title()} (reset)", color=color, fontsize=7)
    return len(objects)


def _geometry_note(geometry):
    if not geometry.get("available"):
        return "Reset object geometry not saved in source manifests"
    return "Boxes show reset collision bounds; objects may move during attempts. Not silhouettes or final bounds."


def read_attempts(source: Path):
    """Read every trial summary and its raw trajectory, keyed by relative path."""
    source = source.resolve()
    summaries = sorted(path for path in source.rglob("summary.json")
                       if path.parent.name.startswith("trial_"))
    if not summaries:
        raise ValueError(f"No trial_*/summary.json files found under {source}")
    attempts = []
    for summary_path in summaries:
        rel_path = summary_path.parent.relative_to(source).as_posix()
        summary = _json(summary_path)
        if not isinstance(summary, dict):
            # Collection summaries are lists and are not trial records.
            continue
        transitions_path = summary_path.parent / "transitions.jsonl"
        skills_path = summary_path.parent / "skills.jsonl"
        if not transitions_path.is_file() or not skills_path.is_file():
            raise ValueError(f"Missing raw transitions.jsonl or skills.jsonl for {rel_path}")
        transitions = _jsonl(transitions_path)
        skills = _jsonl(skills_path)
        controls = summary.get("controls")
        if not isinstance(controls, int) or controls < 0:
            raise ValueError(f"Invalid summary controls for {rel_path}: {controls!r}")
        if len(transitions) != controls:
            raise ValueError(
                f"Tick count mismatch for {rel_path}: {len(transitions)} rows, summary says {controls}"
            )

        points = []
        endpoint_source = "transitions"
        if transitions:
            for index, row in enumerate(transitions):
                context = f"{rel_path} transition {index}"
                before = _nested_position(row, "before", context)
                after = _nested_position(row, "after", context)
                if index == 0:
                    points.append(before)
                points.append(after)
        else:
            fallback = None
            if skills:
                try:
                    fallback = _nested_position(skills[0], "before", f"{rel_path} first skill")
                except ValueError:
                    pass
            if fallback is None:
                rejected_path = summary_path.parent / "rejected_command.json"
                if rejected_path.is_file():
                    rejected = _json(rejected_path)
                    if not isinstance(rejected, dict):
                        raise ValueError(f"Expected an object in {rejected_path}")
                    if rejected.get("tcp_before") is not None:
                        fallback = _position(rejected["tcp_before"], f"{rejected_path} tcp_before")
            if fallback is not None:
                points = [fallback]
                endpoint_source = "zero_tick_fallback"
            else:
                endpoint_source = "missing"

        outcome = _raw_outcome(summary)
        goal = _goal(summary, rel_path)
        start = points[0].tolist() if points else [None, None, None]
        end = points[-1].tolist() if points else [None, None, None]
        raw_status = str(outcome.get("status", ""))
        raw_reason = str(outcome.get("reason", ""))
        raw_primitive_status = str(outcome.get("primitive_status", ""))
        attempts.append({
            "source_path": rel_path,
            "trial_id": summary.get("trial_id"),
            "object": _object_kind(summary, summary_path.parent.relative_to(source)),
            "object_id": str(summary.get("object", "unknown")),
            "scene": summary.get("scene", ""),
            "recipe": summary.get("recipe", ""),
            "controls": controls,
            "start": start,
            "end": end,
            "goal": goal,
            "endpoint_source": endpoint_source,
            "outcome_category": _outcome_category(summary, outcome),
            "outcome_status": raw_status,
            "outcome_reason": raw_reason,
            "primitive_status": raw_primitive_status,
            "outcome_raw": json.dumps(outcome, sort_keys=True, separators=(",", ":")),
            "trajectory": np.asarray(points, dtype=np.float64).reshape((-1, 3)),
        })
    if not attempts:
        raise ValueError(f"No trial records found under {source}")
    return attempts


def export_data(attempts, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    columns = ["source_path", "trial_id", "object", "object_id", "scene", "recipe", "controls",
               "start_x", "start_y", "start_z", "end_x", "end_y", "end_z",
               "goal_x", "goal_y", "goal_z", "endpoint_source", "outcome_category",
               "outcome_status", "outcome_reason", "primitive_status", "outcome_raw"]
    with (output / "ee_endpoints.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for attempt in attempts:
            coordinate_columns = {f"{prefix}_{axis}" for prefix in ("start", "end", "goal")
                                  for axis in AXES.lower()}
            row = {key: attempt.get(key, "") for key in columns if key not in coordinate_columns}
            for prefix, key in (("start", "start"), ("end", "end"), ("goal", "goal")):
                for axis, value in zip(AXES.lower(), attempt[key]):
                    row[f"{prefix}_{axis}"] = "" if value is None else value
            writer.writerow(row)

    offsets = np.zeros(len(attempts) + 1, dtype=np.int64)
    for index, attempt in enumerate(attempts):
        offsets[index + 1] = offsets[index] + len(attempt["trajectory"])
    positions = (np.concatenate([a["trajectory"] for a in attempts], axis=0)
                 if offsets[-1] else np.empty((0, 3), dtype=np.float64))
    np.savez_compressed(
        output / "ee_trajectories.npz",
        positions=positions,
        offsets=offsets,
        source_paths=np.asarray([a["source_path"] for a in attempts], dtype=str),
        trial_ids=np.asarray([str(a["trial_id"]) for a in attempts], dtype=str),
        object=np.asarray([a["object"] for a in attempts], dtype=str),
        object_ids=np.asarray([a["object_id"] for a in attempts], dtype=str),
    )
    missing = sum(a["endpoint_source"] == "missing" for a in attempts)
    endpoint_counts = {}
    for obj in sorted({a["object"] for a in attempts}):
        counts = Counter(tuple(a["end"]) for a in attempts
                         if a["object"] == obj and all(v is not None for v in a["end"]))
        endpoint_counts[obj] = {
            "actual_endpoints": sum(counts.values()),
            "stacked_duplicates": sum(count - 1 for count in counts.values()),
            "exact_xyz_counts": {
                json.dumps(list(position), separators=(",", ":")): count
                for position, count in sorted(counts.items())
            },
        }
    summary = {
        "attempts": len(attempts),
        "actual_endpoints_available": len(attempts) - missing,
        "actual_endpoints_missing": missing,
        "outcomes": dict(Counter(a["outcome_category"] for a in attempts)),
        "outcomes_by_object": {
            obj: dict(Counter(a["outcome_category"] for a in attempts if a["object"] == obj))
            for obj in sorted({a["object"] for a in attempts})
        },
        "endpoint_counts_by_object": endpoint_counts,
        "endpoint_source_counts": dict(Counter(a["endpoint_source"] for a in attempts)),
        "goal_marker_count": sum(all(v is not None for v in a["goal"]) for a in attempts),
        "goal_markers_are_per_attempt": True,
        "goal_marker_duplicates_are_retained": True,
        "position_frame": "robot root",
        "position_units": "metres",
    }
    (output / "ee_collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _geometry_legend_handles(geometry, plt):
    from matplotlib.patches import Patch

    kinds = {record["object"] for objects in geometry.get("scenes", {}).values()
             for record in objects.values()}
    return [Patch(facecolor=RESET_OBJECT_COLORS.get(kind, "tab:gray"), edgecolor=RESET_OBJECT_COLORS.get(kind, "tab:gray"),
                  alpha=.3, label=f"{kind.title()} (reset AABB)")
            for kind in sorted(kinds)]


def plot_endpoints(attempts, output: Path, geometry=None):
    plt = _pyplot()
    geometry = geometry or {"scenes": {}, "available": False}
    row_groups = _row_groups(attempts)
    fig, axes = plt.subplots(len(row_groups), 3, figsize=(15, max(7, 4.5 * len(row_groups))),
                             squeeze=False, constrained_layout=True)
    present_categories = [category for category in OUTCOMES
                         if any(a["outcome_category"] == category
                                and all(v is not None for v in a["end"]) for a in attempts)]
    handles = [plt.Line2D([], [], marker="o", linestyle="None", color=COLORS[category],
                          label=f"{category} (n={sum(a['outcome_category'] == category and all(v is not None for v in a['end']) for a in attempts)})")
               for category in present_categories]
    if any(all(v is not None for v in a["goal"]) for a in attempts):
        handles.append(plt.Line2D([], [], marker="o", linestyle="None", markerfacecolor="none",
                                  markeredgecolor="0.45", label="requested goals"))
    handles.extend(_geometry_legend_handles(geometry, plt))
    for row, (scene, obj) in enumerate(row_groups):
        selected = [a for a in attempts if str(a["scene"]) == scene and a["object"] == obj]
        for col, (x, y) in enumerate(PROJECTIONS):
            ax = axes[row, col]
            _draw_geometry_2d(ax, geometry, scene, x, y)
            for category in OUTCOMES:
                points = [a["end"] for a in selected if a["outcome_category"] == category
                          and all(v is not None for v in a["end"])]
                if points:
                    coords = np.asarray(points)
                    ax.scatter(coords[:, x], coords[:, y], s=18, color=COLORS[category], alpha=.75)
            goals = [a["goal"] for a in selected if all(v is not None for v in a["goal"])]
            if goals:
                coords = np.asarray(goals)
                ax.scatter(coords[:, x], coords[:, y], s=48, facecolors="none",
                           edgecolors="0.45", linewidths=.8, alpha=.55)
            ax.set(title=f"{scene} / {obj}: {AXES[x]}–{AXES[y]}", xlabel=f"Root {AXES[x]} (m)",
                   ylabel=f"Root {AXES[y]} (m)")
            ax.grid(alpha=.2)
            ax.set_aspect("equal", adjustable="datalim")
            if row == 0 and col == 0 and handles:
                ax.legend(handles=handles, fontsize=7)
    fig.suptitle("Measured final end-effector positions and requested goals\n" + _geometry_note(geometry))
    for extension in ("png", "svg"):
        fig.savefig(output / f"ee_endpoints.{extension}", dpi=180)
    plt.close(fig)


def plot_trajectories(attempts, output: Path, geometry=None):
    plt = _pyplot()
    geometry = geometry or {"scenes": {}, "available": False}
    row_groups = _row_groups(attempts)
    fig, axes = plt.subplots(len(row_groups), 3, figsize=(15, max(7, 4.5 * len(row_groups))),
                             squeeze=False, constrained_layout=True)
    for row, (scene, obj) in enumerate(row_groups):
        selected = [a for a in attempts if str(a["scene"]) == scene and a["object"] == obj]
        for col, (x, y) in enumerate(PROJECTIONS):
            ax = axes[row, col]
            _draw_geometry_2d(ax, geometry, scene, x, y)
            for attempt in selected:
                trajectory = attempt["trajectory"]
                if len(trajectory) >= 2:
                    ax.plot(trajectory[:, x], trajectory[:, y], color=COLORS[attempt["outcome_category"]],
                            alpha=.15, linewidth=.55)
            ax.set(title=f"{scene} / {obj}: {AXES[x]}–{AXES[y]}", xlabel=f"Root {AXES[x]} (m)",
                   ylabel=f"Root {AXES[y]} (m)")
            ax.grid(alpha=.2)
            ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Measured end-effector trajectories (one line per attempt; no subsampling)\n" + _geometry_note(geometry))
    for extension in ("png", "svg"):
        fig.savefig(output / f"ee_trajectories.{extension}", dpi=180)
    plt.close(fig)


def plot_endpoints_3d(attempts, output: Path, geometry=None):
    plt = _pyplot()
    geometry = geometry or {"scenes": {}, "available": False}
    scenes = sorted({str(a["scene"]) for a in attempts})
    fig = plt.figure(figsize=(10, 7 * len(scenes)), constrained_layout=True)
    for index, scene in enumerate(scenes, 1):
        ax = fig.add_subplot(1, len(scenes), index, projection="3d")
        selected = [a for a in attempts if str(a["scene"]) == scene]
        _draw_geometry_3d(ax, geometry, scene)
        for category in OUTCOMES:
            points = [a["end"] for a in selected if a["outcome_category"] == category
                      and all(v is not None for v in a["end"])]
            if points:
                coords = np.asarray(points)
                ax.scatter(*coords.T, s=20, color=COLORS[category], alpha=.75,
                           label=f"{category} (n={len(points)})")
        goals = [a["goal"] for a in selected if all(v is not None for v in a["goal"])]
        if goals:
            coords = np.asarray(goals)
            ax.scatter(*coords.T, s=45, facecolors="none", edgecolors="0.45",
                       alpha=.55, label=f"requested goals (n={len(goals)})")
        ax.set(xlabel="Root X (m)", ylabel="Root Y (m)", zlabel="Root Z (m)", title=scene)
        spans = [np.ptp(limits) for limits in
                 (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())]
        ax.set_box_aspect(np.maximum(spans, 1e-3))
        if selected:
            ax.legend(fontsize=7)
    fig.suptitle("Measured final end-effector positions and requested goals\n" + _geometry_note(geometry))
    for extension in ("png", "svg"):
        fig.savefig(output / f"ee_endpoints_3d.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(source: Path, output: Path):
    attempts = read_attempts(source)
    geometry = read_scene_geometry(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ee_scene_geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    summary = export_data(attempts, output)
    plot_endpoints(attempts, output, geometry)
    plot_trajectories(attempts, output, geometry)
    plot_endpoints_3d(attempts, output, geometry)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Root containing collector trial directories")
    parser.add_argument("--output", type=Path, help="Output directory (default: SOURCE/ee_visualization)")
    args = parser.parse_args(argv)
    output = args.output or args.source / "ee_visualization"
    summary = run(args.source, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
