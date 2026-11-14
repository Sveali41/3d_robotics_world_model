import csv
import json

import numpy as np

from exploration.diagnostics import plot_primitive_collection as module


def _write_attempt(directory, *, trial_id, object_id, controls, ticks=(), skills=(), outcome=None,
                   goal=(0.8, 0.8, 0.8), rejected=None):
    directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "scene": "original",
        "object": object_id,
        "recipe": "test_recipe",
        "trial_id": trial_id,
        "goal_root": list(goal) if goal is not None else None,
        "outcome": outcome or {"status": "failed", "reason": "test_failure"},
        "controls": controls,
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "transitions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ticks)
    )
    (directory / "skills.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in skills)
    )
    if rejected is not None:
        (directory / "rejected_command.json").write_text(json.dumps(rejected))


def _tick(step, before, after):
    return {
        "step": step,
        "before": {"ee_position": list(before)},
        "after": {"ee_position": list(after)},
    }


def test_reads_actual_path_and_preserves_nested_duplicate_trial_ids(tmp_path):
    source = tmp_path / "source"
    cup = source / "cup" / "trial_000001"
    bowl = source / "bowl" / "trial_000001"
    _write_attempt(cup, trial_id=1, object_id="cup_0", controls=2,
                   ticks=[_tick(1, (1, 2, 3), (2, 3, 4)),
                          _tick(2, (2, 3, 4), (3, 4, 5))],
                   outcome={"status": "failed", "reason": "primitive_failed",
                            "primitive_status": "no_collision_free_ik"},
                   goal=(9, 9, 9))
    _write_attempt(bowl, trial_id=1, object_id="bowl_0", controls=1,
                   ticks=[_tick(1, (-1, -2, -3), (-2, -3, -4))],
                   outcome={"status": "unconfirmed_grasp"}, goal=(-9, -9, -9))

    attempts = module.read_attempts(source)
    assert [a["source_path"] for a in attempts] == [
        "bowl/trial_000001", "cup/trial_000001"
    ]
    by_object = {a["object"]: a for a in attempts}
    assert by_object["cup"]["start"] == [1, 2, 3]
    assert by_object["cup"]["end"] == [3, 4, 5]
    assert by_object["cup"]["end"] != by_object["cup"]["goal"]
    assert by_object["cup"]["trajectory"].tolist() == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    assert by_object["cup"]["outcome_category"] == "planning_rejected"
    assert by_object["cup"]["object_id"] == "cup_0"
    assert by_object["bowl"]["outcome_category"] == "unconfirmed_grasp"

    output = tmp_path / "export"
    module.export_data(attempts, output)
    with np.load(output / "ee_trajectories.npz") as data:
        assert data["source_paths"].tolist() == ["bowl/trial_000001", "cup/trial_000001"]
        assert data["trial_ids"].tolist() == ["1", "1"]
        assert data["offsets"].tolist() == [0, 2, 5]
        assert data["positions"].tolist() == [
            [-1, -2, -3], [-2, -3, -4], [1, 2, 3], [2, 3, 4], [3, 4, 5]
        ]
    with (output / "ee_endpoints.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["end_x"] == "3.0"
    assert rows[1]["goal_x"] == "9.0"
    assert rows[1]["outcome_category"] == "planning_rejected"


def test_zero_tick_fallback_and_missing_endpoint_are_explicit(tmp_path):
    source = tmp_path / "source"
    skill = {"before": {"ee_position": [1.25, 2.5, 3.75]}}
    _write_attempt(source / "cup" / "trial_000000", trial_id=0, object_id="cup_0",
                   controls=0, skills=[skill], goal=(4, 5, 6),
                   outcome={"status": "failed", "reason": "planning_rejected",
                            "primitive_status": "no_collision_free_ik"})
    _write_attempt(source / "bowl" / "trial_000000", trial_id=0, object_id="bowl_0",
                   controls=0, goal=(7, 8, 9), outcome={"status": "budget_truncated"})

    attempts = module.read_attempts(source)
    by_object = {a["object"]: a for a in attempts}
    assert by_object["cup"]["endpoint_source"] == "zero_tick_fallback"
    assert by_object["cup"]["start"] == by_object["cup"]["end"] == [1.25, 2.5, 3.75]
    assert by_object["cup"]["trajectory"].tolist() == [[1.25, 2.5, 3.75]]
    assert by_object["bowl"]["endpoint_source"] == "missing"
    assert by_object["bowl"]["start"] == by_object["bowl"]["end"] == [None, None, None]
    assert by_object["bowl"]["outcome_category"] == "budget_truncated"

    summary = module.export_data(attempts, tmp_path / "export")
    assert summary["actual_endpoints_missing"] == 1
    assert summary["endpoint_source_counts"] == {"zero_tick_fallback": 1, "missing": 1}
    with np.load(tmp_path / "export" / "ee_trajectories.npz") as data:
        assert data["offsets"].tolist() == [0, 0, 1]
        assert data["positions"].tolist() == [[1.25, 2.5, 3.75]]


def test_zero_tick_rejected_command_tcp_fallback(tmp_path):
    source = tmp_path / "source"
    _write_attempt(source / "bowl" / "trial_000004", trial_id=4, object_id="bowl_4",
                   controls=0, rejected={"tcp_before": [0.1, 0.2, 0.3]},
                   outcome={"status": "failed", "reason": "invalid_parameters"})
    attempt, = module.read_attempts(source)
    assert attempt["endpoint_source"] == "zero_tick_fallback"
    assert attempt["start"] == attempt["end"] == [0.1, 0.2, 0.3]


def test_real_collector_outcome_and_object_ids_map_to_display_categories(tmp_path):
    source = tmp_path / "source"
    root = source / "original"
    _write_attempt(root / "trial_000000", trial_id=0, object_id="cabinet_item_17",
                   controls=1, ticks=[_tick(1, (0, 0, 0), (1, 1, 1))],
                   outcome={"status": "grasp_and_lift_verified"}, goal=(1, 1, 1))
    _write_attempt(root / "trial_000001", trial_id=1, object_id="cabinet_item_16",
                   controls=0, outcome={"status": "failed", "reason": "primitive_failed",
                                       "primitive_status": "budget_truncated"})
    _write_attempt(root / "trial_000002", trial_id=2, object_id="cabinet_item_16",
                   controls=0, outcome={"status": "failed", "reason": "physical_collision",
                                       "primitive_status": "physical_collision"})

    attempts = module.read_attempts(source)
    by_id = {a["trial_id"]: a for a in attempts}
    assert by_id[0]["object"] == "bowl"
    assert by_id[0]["outcome_category"] == "verified_lift"
    assert by_id[1]["object"] == "cup"
    assert by_id[1]["outcome_category"] == "budget_truncated"
    assert by_id[2]["outcome_category"] == "other_failure"
    summary = module.export_data(attempts, tmp_path / "export")
    assert summary["outcomes"] == {
        "verified_lift": 1, "budget_truncated": 1, "other_failure": 1
    }
    assert summary["endpoint_counts_by_object"]["bowl"]["actual_endpoints"] == 1


def test_corrupt_ticks_or_nonfinite_positions_fail_clearly(tmp_path):
    source = tmp_path / "source"
    attempt = source / "cup" / "trial_000000"
    _write_attempt(attempt, trial_id=0, object_id="cup_0", controls=1,
                   ticks=[_tick(1, (0, 0, 0), (1, 1, 1))])
    (attempt / "summary.json").write_text(json.dumps({"controls": 2}))
    try:
        module.read_attempts(source)
    except ValueError as exc:
        assert "Tick count mismatch" in str(exc)
    else:
        raise AssertionError("tick count mismatch was accepted")

    _write_attempt(attempt, trial_id=0, object_id="cup_0", controls=1,
                   ticks=[_tick(1, (0, 0, 0), (float("nan"), 1, 1))])
    try:
        module.read_attempts(source)
    except ValueError as exc:
        assert "Non-finite XYZ position" in str(exc)
    else:
        raise AssertionError("non-finite position was accepted")


def _write_manifest(path, geometry=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {} if geometry is None else {"geometry": geometry}
    path.write_text(json.dumps(manifest))


def _box(lo, hi):
    return {"collision_bounds_root_m": [list(lo), list(hi)]}


def test_manifest_geometry_is_deduplicated_and_projected_with_reset_labels(tmp_path):
    source = tmp_path / "source"
    cup_bounds = [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]]
    bowl_bounds = [[-0.1, 0.05, 0.25], [0.15, 0.35, 0.5]]
    _write_manifest(source / "cup" / "manifest.json", {
        "original": {"cabinet_item_16": _box(*cup_bounds)},
    })
    _write_manifest(source / "bowl" / "manifest.json", {
        "original": {"cabinet_item_16": _box(*cup_bounds),
                     "cabinet_item_17": _box(*bowl_bounds)},
    })

    geometry = module.read_scene_geometry(source)
    assert geometry["available"] is True
    assert geometry["scenes"]["original"]["cabinet_item_16"]["object"] == "cup"
    assert geometry["scenes"]["original"]["cabinet_item_17"]["object"] == "bowl"
    assert len(geometry["scenes"]["original"]["cabinet_item_16"]["manifest_sources"]) == 2
    assert geometry["scenes"]["original"]["cabinet_item_16"]["collision_bounds_root_m"] == cup_bounds

    plt = module._pyplot()
    for x, y in module.PROJECTIONS:
        fig, ax = plt.subplots()
        assert module._draw_geometry_2d(ax, geometry, "original", x, y) == 2
        assert len(ax.patches) == 2
        assert {text.get_text() for text in ax.texts} == {"Cup (reset)", "Bowl (reset)"}
        expected = [cup_bounds, bowl_bounds]
        for patch, bounds in zip(ax.patches, expected):
            lo, hi = np.asarray(bounds)
            assert patch.get_x() == lo[x]
            assert patch.get_y() == lo[y]
            assert patch.get_width() == hi[x] - lo[x]
            assert patch.get_height() == hi[y] - lo[y]
        ax.relim()
        ax.autoscale_view()
        assert ax.get_xlim()[0] <= min(cup_bounds[0][x], bowl_bounds[0][x])
        assert ax.get_xlim()[1] >= max(cup_bounds[1][x], bowl_bounds[1][x])
        assert ax.get_ylim()[0] <= min(cup_bounds[0][y], bowl_bounds[0][y])
        assert ax.get_ylim()[1] >= max(cup_bounds[1][y], bowl_bounds[1][y])
        plt.close(fig)


def test_manifest_geometry_rejects_inconsistent_or_invalid_bounds(tmp_path):
    source = tmp_path / "source"
    _write_manifest(source / "cup" / "manifest.json", {
        "original": {"cabinet_item_16": _box([0, 0, 0], [1, 1, 1])},
    })
    _write_manifest(source / "bowl" / "manifest.json", {
        "original": {"cabinet_item_16": _box([0, 0, 0], [1, 1.01, 1])},
    })
    try:
        module.read_scene_geometry(source)
    except ValueError as exc:
        assert "Inconsistent collision bounds" in str(exc)
    else:
        raise AssertionError("materially inconsistent manifest bounds were accepted")

    _write_manifest(source / "cup" / "manifest.json", {
        "original": {"cabinet_item_16": _box([0, 0, 0], [1, float("inf"), 1])},
    })
    _write_manifest(source / "bowl" / "manifest.json", {})
    try:
        module.read_scene_geometry(source)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite manifest bounds were accepted")


def test_geometry_missing_is_reported_and_scene_rows_remain_separate(tmp_path):
    source = tmp_path / "source"
    _write_manifest(source / "manifest.json")
    geometry = module.read_scene_geometry(source)
    assert geometry["available"] is False
    assert geometry["missing_geometry_manifests"] == ["manifest.json"]
    attempts = [
        {"scene": "original", "object": "cup"},
        {"scene": "shifted", "object": "cup"},
        {"scene": "original", "object": "bowl"},
    ]
    assert module._row_groups(attempts) == [
        ("original", "cup"), ("original", "bowl"), ("shifted", "cup")
    ]
    assert "not saved" in module._geometry_note(geometry)
