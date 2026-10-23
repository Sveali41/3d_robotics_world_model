import copy
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

from exploration import diagnose_cabinet as diagnostic
from exploration.contact_reward import ContactExplorationRewardConfig
from exploration.diagnostic_metrics import compare_arms, motion_bucket, summarize_records


def state(x=0.0, failed=False):
    return {
        "ee_position": [0.35, -0.06, 0.72],
        "objects": {
            "cabinet_item_16": {"position_root": [0.455 + x, 0.0, 0.7], "position_world": [0.455 + x, 0.0, 0.7]},
            "cabinet_item_17": {"position_root": [0.455, 0.12, 0.7], "position_world": [0.455, 0.12, 0.7]},
        },
        "contact_snapshot": {"objects": {}}, "inside_region": True,
        "failed_object_ids": ["cabinet_item_16"] if failed else [],
    }


def transition(before=None, after=None, episode=1, held=False):
    return {
        "episode_id": episode, "primitive_id": 0, "primitive_type": "move_forward",
        "duration_low_steps": 10, "status": "success", "final_stage": "move",
        "terminated": False, "truncated": False, "run_budget_exhausted": False,
        "before": before or state(), "after": after or state(0.02),
        "action": [0.0] * 20, "parameters_physical": {"x_dist": 0.02},
        "contact_events": [{"objects": {"cabinet_item_16": {
            "left_force": 0.1, "right_force": 0.0, "grasped": False,
        }}}] + [{"objects": {}}] * 9,
        "outcome_metrics": {
            "grasped_object_ids": [], "held_object_ids": ["cabinet_item_16"] if held else [],
            "lifted_object_ids": [], "moved_object_ids": ["cabinet_item_16"],
        },
    }


def with_rewards(rows):
    cfg = OmegaConf.create({"reward": asdict(ContactExplorationRewardConfig())})
    diagnostic.replay_rewards(rows, cfg)
    for row in rows:
        row["reward"] = row["counterfactual_rewards"]["R0"]
    return rows


def test_offline_reward_profiles_have_independent_motion_history_and_reset():
    rows = with_rewards([
        transition(), transition(state(0.02), state(0.04), held=True),
        transition(episode=2),
    ])
    assert [row["counterfactual_rewards"]["R0"]["terms"]["moved_object"] for row in rows] == [0, 3, 0]
    assert [row["counterfactual_rewards"]["R1"]["terms"]["moved_object"] for row in rows] == [3, 0, 3]
    assert rows[1]["counterfactual_cumulative_returns"]["R1"] == pytest.approx(sum(
        row["counterfactual_rewards"]["R1"]["total"] for row in rows[:2]))
    assert rows[2]["counterfactual_cumulative_returns"]["R1"] == rows[2]["counterfactual_rewards"]["R1"]["total"]
    assert rows[0]["counterfactual_rewards"]["R0"]["raw"]["grasp_alignment"] != rows[0]["counterfactual_rewards"]["offset_only"]["raw"]["grasp_alignment"]


@pytest.mark.parametrize("distance,expected", [
    (0.0149, None), (0.015, "15-30mm"), (0.03, "30-60mm"), (0.06, "60mm+")
])
def test_motion_bucket_boundaries(distance, expected):
    bucket = motion_bucket("object", "lift", [0, 0, -distance])
    if expected is None:
        assert bucket is None
    else:
        assert bucket == ("object", "lift", "z-", expected)


def test_coverage_uses_real_steps_and_keeps_failed_transitions():
    rows = with_rewards([
        transition(), transition(state(0.02), state(0.04)),
        transition(state(0.04), state(0.07, failed=True)),
    ])
    summary = summarize_records(rows)
    assert summary["low_level_steps"] == 30
    assert summary["primitives"] == 3
    assert summary["raw_coverage"] == 2
    assert summary["recoverable_coverage"] == 1
    assert summary["recoverable_coverage_auc"] == pytest.approx(20 / 30)
    assert summary["recoverable_bucket_repeat_fraction"] == 0.5
    assert summary["motion_primitives_per_1000_steps"] == 100
    assert summary["recoverable_motion_primitives_per_1000_steps"] == pytest.approx(2000 / 30)
    assert summary["episodes_censored"] == 1
    assert summary["episodes_finished"] == 0


def test_episode_completion_count_is_not_counted_per_primitive():
    rows = with_rewards([transition(), transition(state(0.02), state(0.04))])
    for row in rows:
        row["reward"]["exploration_complete"] = True
    rows[-1]["truncated"] = True
    summary = summarize_records(rows)
    assert summary["completed_episodes"] == 1
    assert summary["episode_completion_rate"] == 1
    rows[-1]["run_budget_exhausted"] = True
    assert summarize_records(rows)["episodes_finished"] == 0


def screening_summary(coverage=1.0, motion=1.0, objects=True):
    return {
        "low_level_steps": 5000, "recoverable_coverage_auc": coverage,
        "recoverable_motion_primitives_per_1000_steps": motion,
        "per_object": {name: {"recoverable_motion_primitives": int(objects)} for name in diagnostic.TARGETS},
    }


def test_screening_requires_two_objects_and_both_metrics():
    summaries = {"A": screening_summary(), "B": screening_summary(2, 0.5),
                 "C": screening_summary(2, 2, objects=False), "D": screening_summary(2, 2)}
    result = compare_arms(summaries)
    assert result["candidates"] == ["D"]
    assert result["interpretation"] == "interaction_suspected"
    summaries["D"]["low_level_steps"] = 4000
    assert compare_arms(summaries)["status"] == "incomparable"


def test_report_rejects_metrics_without_completed_evaluation(tmp_path, capsys):
    for arm in diagnostic.ARMS:
        directory = tmp_path / "evaluate" / arm
        directory.mkdir(parents=True)
        diagnostic.write_json(directory / "metrics.json", screening_summary(2, 2))
    result = diagnostic.run_report(tmp_path)
    assert result["status"] == "incomplete"
    assert result["candidates"] == []


class BudgetEnv:
    def __init__(self, fail=False):
        self.env = SimpleNamespace(max_low_level_steps_per_episode=1000, low_level_steps=998)
        self.fail = fail

    def step(self, action):
        if self.fail:
            raise RuntimeError("simulation failure")
        duration = min(7, self.env.max_low_level_steps_per_episode - self.env.low_level_steps)
        self.env.low_level_steps += duration
        record = SimpleNamespace(
            duration_low_steps=duration, episode_id=1, primitive_id=1, primitive_type="move_forward",
            parameters_physical={"x_dist": 0.1}, status="budget_exhausted", final_stage="move",
            outcome_metrics={}, contact_events=[],
        )
        return np.zeros(1), 0.0, False, True, {"primitive_record": record, "exploration_reward": {}}


def test_run_cap_does_not_enlarge_episode_cap_and_is_restored(monkeypatch):
    monkeypatch.setattr(diagnostic, "snapshot", lambda env: state())
    env = BudgetEnv()
    *_, row = diagnostic.bounded_step(env, np.zeros(20), 3)
    assert row["duration_low_steps"] == 2
    assert not row["run_budget_exhausted"]
    assert env.env.max_low_level_steps_per_episode == 1000
    env.env.low_level_steps = 0
    *_, row = diagnostic.bounded_step(env, np.zeros(20), 3)
    assert row["duration_low_steps"] == 3
    assert row["run_budget_exhausted"]
    assert env.env.max_low_level_steps_per_episode == 1000


def test_run_cap_restored_after_simulation_error(monkeypatch):
    monkeypatch.setattr(diagnostic, "snapshot", lambda env: state())
    env = BudgetEnv(fail=True)
    with pytest.raises(RuntimeError, match="simulation failure"):
        diagnostic.bounded_step(env, np.zeros(20), 1)
    assert env.env.max_low_level_steps_per_episode == 1000


def test_audit_cases_are_local_and_do_not_modify_scripted_baselines():
    original = copy.deepcopy(diagnostic.SCRIPTED_GRASP_SEQUENCES)
    cases = diagnostic.audit_cases()
    assert len(cases) == 20
    assert [case[1] for case in cases[:2]] == ["center", "center"]
    for target in diagnostic.TARGETS:
        assert sum(t == target and (label == "center" or label.endswith("0.005")) for t, label, _ in cases) == 7
    assert original == diagnostic.SCRIPTED_GRASP_SEQUENCES


def test_gae_positive_control_checks_truncation_and_worker_boundaries():
    assert diagnostic.gae_check()


def test_stage_gate_rejects_absent_or_failed_audit(tmp_path):
    with pytest.raises(ValueError, match="has not passed"):
        diagnostic.require_pass(tmp_path, "audit")
    (tmp_path / "audit").mkdir()
    diagnostic.write_json(tmp_path / "audit/result.json", {"passed": False})
    with pytest.raises(ValueError, match="has not passed"):
        diagnostic.require_pass(tmp_path, "audit")


def test_prepare_output_rejects_changed_config(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "source_evidence", lambda output: {
        "fingerprint": "same", "commit": "test", "diff": "", "untracked_sources": {},
    })
    cfg = OmegaConf.create({"seed": 0})
    diagnostic.prepare_output(tmp_path, cfg)
    diagnostic.prepare_output(tmp_path, cfg)
    cfg.seed = 1
    with pytest.raises(ValueError, match="configuration changed"):
        diagnostic.prepare_output(tmp_path, cfg)


def test_offline_rewards_are_written_without_reexecuting_physics(tmp_path):
    cfg = OmegaConf.create({"reward": asdict(ContactExplorationRewardConfig())})
    rows = with_rewards([transition()])
    summary = diagnostic.write_records(tmp_path, rows, cfg, "R0")
    assert summary["low_level_steps"] == 10
    saved = json.loads((tmp_path / "primitives.jsonl").read_text())
    assert set(saved["counterfactual_rewards"]) == {"R0", "offset_only", "R1"}
