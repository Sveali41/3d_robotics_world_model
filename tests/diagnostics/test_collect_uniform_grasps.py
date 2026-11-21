import numpy as np
import pytest

from exploration.diagnostics.collect_uniform_grasps import replay_from_outcomes, uniform_cases


@pytest.fixture
def settings():
    from exploration.raps.raps_gnn_actor import DEFAULTS
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml")
    return {**DEFAULTS, **OmegaConf.to_container(config.raps_actor, resolve=True)}


def test_default_plan_is_full_canonical_uniform_150_unique_cases(settings):
    cases = uniform_cases(settings)
    assert len(cases) == 150
    assert [(c["cell_id"], c["orientation_id"], c["z_m"]) for c in cases[:6]] == [
        (0, 0, .692), (0, 0, .712), (0, 0, .732), (0, 0, .752), (0, 0, .772), (0, 1, .692)]
    assert {c["cell_id"] for c in cases} == set(range(10))
    assert {c["orientation_id"] for c in cases} == {0, 1, 2}
    assert len({(c["cell_id"], c["orientation_id"], c["z_m"]) for c in cases}) == 150
    assert len({c["seed"] for c in cases}) == 150
    assert all(len(c["action"]) == 31 for c in cases)
    assert [c["seed"] for c in cases] == [c["seed"] for c in uniform_cases(settings)]


def test_finer_z_grid_preserves_bounds_and_has_new_cases(settings):
    cases = uniform_cases(settings, z_levels=7)
    levels = sorted({c["z_m"] for c in cases})
    assert len(cases) == 210
    assert np.allclose(levels, np.linspace(.692, .772, 7))
    assert not any(c["orientation_id"] in (3, 4) for c in cases)


@pytest.mark.parametrize("levels", [0, 1, -2, 2.5, True])
def test_invalid_z_level_count_rejected(settings, levels):
    with pytest.raises(ValueError, match="z_levels"):
        uniform_cases(settings, z_levels=levels)


def test_replay_export_keeps_only_completed_nonzero_and_isolates_cases():
    def outcome(case_id, completed, duration, status="success"):
        skill = dict(state=np.zeros(50).tolist(), action=np.zeros(31).tolist(),
                     next_state=np.ones(50).tolist(), terminated=False, truncated=False,
                     duration_low_steps=duration)
        return dict(case_id=case_id, cell_id=4, orientation_id=2, z_m=.732,
                    completed=completed, duration_low_steps=duration,
                    status=status, wm_eligible=completed and duration > 0, skill=skill)

    result = replay_from_outcomes([outcome(72, True, 8), outcome(73, False, 0, "budget_truncated"),
                                   outcome(74, True, 0, "failed"), outcome(75, True, 4, "failed")])
    assert result["states"].shape == (2, 50)
    assert result["actions"].shape == (2, 31)
    assert result["next_states"].shape == (2, 50)
    assert result["duration_low_steps"].tolist() == [8, 4]
    assert result["case_ids"].tolist() == [72, 75]
    assert result["episode_ids"].tolist() == [73, 76]
    assert result["cell_ids"].tolist() == [4, 4]


def test_empty_replay_has_real_empty_model_shapes():
    result = replay_from_outcomes([dict(case_id=1, wm_eligible=False)],
                                 state_contract={"dimension": 50})
    assert result["states"].shape == (0, 50)
    assert result["actions"].shape == (0, 31)
    assert result["next_states"].shape == (0, 50)
    assert result["terminated"].shape == result["truncated"].shape == (0,)


def test_empty_doorless_replay_uses_supplied_state_contract():
    result = replay_from_outcomes([dict(case_id=1, wm_eligible=False)],
                                 state_contract={"dimension": 48})
    assert result["states"].shape == result["next_states"].shape == (0, 48)


@pytest.mark.parametrize("status,eligible,expected", [
    ("success", True, True), ("lift_failed", True, False),
    ("budget_truncated", False, False), ("success", False, False)])
def test_success_only_filters_export_without_discarding_failure_evidence(monkeypatch, status, eligible, expected):
    from exploration.diagnostics import collect_uniform_grasps as collector
    raw = dict(status=status, wm_eligible=eligible, skill={"status": status})
    monkeypatch.setattr(collector, "run_case", lambda job: dict(raw))
    default = collector._case_worker({})
    selected = collector._case_worker({"success_only": True})
    assert default == raw
    assert selected["wm_eligible"] is expected
    assert selected["macro_endpoint_eligible"] is eligible
    assert selected["skill"] == raw["skill"]


def test_success_only_plan_is_cell4_five_mm_grid(tmp_path):
    import json
    from exploration.diagnostics.collect_uniform_grasps import main
    output = tmp_path / "success_plan"
    assert main(["plan", "--output", str(output), "--success-only", "--z-levels", "17",
                 "--case-ids", *map(str, range(204, 255))]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    cases = [json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()]
    assert manifest["success_only"] is True
    assert manifest["dataset_role"] == "heldout_success_conditioned_reset"
    assert manifest["cases_planned"] == 51
    assert {c["cell_id"] for c in cases} == {4}
    assert {c["orientation_id"] for c in cases} == {0, 1, 2}
    assert np.allclose(manifest["z_values_m"], np.linspace(.692, .772, 17))


def test_plan_accepts_current_single_step_config_without_weakening_profile(tmp_path):
    import json
    from omegaconf import OmegaConf
    from exploration.diagnostics.collect_uniform_grasps import main
    from exploration.diagnostics.profile_cup_grasp import validate_config
    config = OmegaConf.load('configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml')
    assert config.raps_actor.imagination_horizon == 1
    historical = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    historical.cabinet_door = True
    with pytest.raises(ValueError, match='imagination_horizon'):
        validate_config(historical)
    output = tmp_path / 'plan'
    assert main(['plan', '--output', str(output)]) == 0
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['cases_planned'] == 150
    assert manifest['config']['raps_actor']['imagination_horizon'] == 1
    config.cabinet_door = True
    with pytest.raises(ValueError, match='cabinet_door'):
        validate_config(config, learning=False)
