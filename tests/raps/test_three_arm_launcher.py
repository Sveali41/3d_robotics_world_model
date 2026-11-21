"""Matched configuration and immutable source-copy checks without container launches."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from omegaconf import OmegaConf
import pytest

from scripts.launch_raps_comparison import arm_configs, snapshot_sources


def test_three_arms_only_change_exploration_and_keep_control_budget():
    root = Path(__file__).resolve().parents[2]
    base = OmegaConf.to_container(OmegaConf.load(root / "configs/raps_gnn_actor_cabinet_split7.yaml"))
    original = deepcopy(base)
    arms = arm_configs(base)
    assert tuple(arms) == ("curiosity_heuristic", "curiosity_only", "random")
    assert base == original
    for name, config in arms.items():
        restored = deepcopy(config)
        assert restored.pop("collection_policy") == ("random" if name == "random" else "curiosity")
        restored["object_heuristic"] = base["object_heuristic"]
        assert restored == base
        assert config["bootstrap_steps"] + config["rounds"] * config["steps_per_round"] + config["evaluation_steps"] == 400000
        assert config["raps_actor"]["skill_set"] == "split7"
        assert not config["runtime_optimizations"]
    assert arms["curiosity_heuristic"]["object_heuristic"] == base["object_heuristic"]
    assert arms["curiosity_only"]["object_heuristic"] is None
    assert arms["random"]["object_heuristic"] is None
    assert arms["curiosity_heuristic"]["raps_actor"].get("disagreement_weight", 1.0) == 1.0
    arms["random"]["raps_actor"]["hidden_size"] = 32
    assert arms["curiosity_only"]["raps_actor"]["hidden_size"] == 128


def test_snapshot_owns_current_tracked_and_untracked_sources(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "tracked.py").write_text("original = True\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)
    (root / "tracked.py").write_text("original = False\n")
    (root / "new.yaml").write_text("seed: 0\n")
    target = tmp_path / "snapshot"
    snapshot_sources(root, target)
    assert (target / "tracked.py").read_text() == "original = False\n"
    assert (target / "new.yaml").read_text() == "seed: 0\n"
    assert (target / ".git/index").is_file()
    (root / "tracked.py").write_text("changed = True\n")
    assert (target / "tracked.py").read_text() == "original = False\n"
    hashes = json.loads((tmp_path / "source_sha256.json").read_text())
    assert set(hashes) == {"tracked.py", "new.yaml"}


def test_reject_non_split_or_missing_heuristic():
    with pytest.raises(ValueError, match="split7"):
        arm_configs(dict(agent="icem"))
    with pytest.raises(ValueError, match="direct execution"):
        arm_configs(dict(agent="raps_gnn_actor", raps_actor={"skill_set": "split7",
                                                                "motion_execution": "planner"}))
    with pytest.raises(ValueError, match="object heuristic"):
        arm_configs(dict(agent="raps_gnn_actor", raps_actor={"skill_set": "split7"},
                         object_heuristic=None))


@pytest.mark.parametrize("preset", [
    "raps_gnn_actor_cabinet_pick_place.yaml",
    "raps_gnn_actor_cabinet_pick_place_target.yaml",
    "raps_gnn_actor_cabinet_grasp_only.yaml",
    "raps_gnn_actor_cabinet_grasp_grid.yaml",
    "raps_gnn_actor_cabinet_grasp_grid_fast.yaml",
    "raps_gnn_actor_cabinet_pick_place_grid.yaml",
])
def test_pick_place_preset_and_three_arm_parity(preset):
    root = Path(__file__).resolve().parents[2]
    legacy = OmegaConf.to_container(OmegaConf.load(root / "configs/raps_gnn_actor_cabinet_split7.yaml"))
    base = OmegaConf.to_container(OmegaConf.load(root / "configs" / preset))
    assert (base["bootstrap_steps"], base["rounds"], base["steps_per_round"],
            base["evaluation_steps"], base["episode_steps"]) == (2000, 398, 2000, 2000, 2200)
    assert base["bootstrap_steps"] + base["rounds"] * base["steps_per_round"] + base["evaluation_steps"] == 800000
    assert (base["seed"], base["device"], base["epochs"]) == (
        legacy["seed"], legacy["device"], legacy["epochs"])
    for key in ("hidden_size", "actor_layers", "critic_layers", "init_std", "imagination_horizon",
                "imagination_batch_size", "updates_per_round", "discount", "lambda_return",
                "actor_lr", "critic_lr", "adam_eps", "entropy_scale", "gradient_clip",
                "target_update_period"):
        assert base["raps_actor"][key] == legacy["raps_actor"][key]
    actor = base["raps_actor"]
    if actor.get("pick_place_grid", False):
        from exploration.raps.raps_gnn_actor import skill_schema
        assert actor["grasp_orientation_candidates"]
        assert skill_schema(actor).action_dim == (61 if actor.get("grasp_only", False) else 73)
    assert base["action_mode"] == "joint_position"
    assert actor["pick_place_primitives"] and actor["motion_execution"] == "planner"
    assert actor["grasp_retention"] == "constraint" and actor["latched_gripper"]
    assert not any(actor[key] for key in ("hold_closed_during_motion", "stop_drop_on_support",
                                          "grasp_stop_on_contact", "controlled_placement"))

    arms = arm_configs(base)
    assert tuple(arms) == ("curiosity_heuristic", "curiosity_only", "random")
    for name, config in arms.items():
        restored = deepcopy(config)
        assert restored.pop("collection_policy") == ("random" if name == "random" else "curiosity")
        restored["object_heuristic"] = base["object_heuristic"]
        assert restored == base
        assert config["bootstrap_steps"] + config["rounds"] * config["steps_per_round"] + config["evaluation_steps"] == 800000
        assert config["raps_actor"]["pick_place_primitives"]
    assert arms["curiosity_only"]["object_heuristic"] is None
    assert arms["random"]["object_heuristic"] is None


def test_optional_heuristic_only_arm_preserves_base_and_other_settings():
    root = Path(__file__).resolve().parents[2]
    base = OmegaConf.to_container(OmegaConf.load(root / "configs/raps_gnn_actor_cabinet_pick_place.yaml"))
    original = deepcopy(base)
    arms = arm_configs(base, include_heuristic_only=True)

    assert tuple(arms) == ("curiosity_heuristic", "curiosity_only", "heuristic_only", "random")
    assert base == original
    for name, config in arms.items():
        restored = deepcopy(config)
        restored.pop("collection_policy")
        if config["object_heuristic"] is None:
            restored["object_heuristic"] = base["object_heuristic"]
        if name == "heuristic_only":
            assert config["collection_policy"] == "curiosity"
            assert config["object_heuristic"] == base["object_heuristic"]
            assert config["raps_actor"]["disagreement_weight"] == 0.0
            restored["raps_actor"].pop("disagreement_weight")
        else:
            assert config["raps_actor"].get("disagreement_weight", 1.0) == 1.0
        assert restored == base

    assert arms["curiosity_heuristic"]["object_heuristic"] == base["object_heuristic"]
    assert arms["curiosity_only"]["object_heuristic"] is None
    assert arms["random"]["collection_policy"] == "random"
    assert arms["random"]["object_heuristic"] is None


def test_current_cup_trapezoid_four_arm_contract_and_budget():
    from exploration.raps.raps_gnn_actor import skill_schema, action_contract
    root = Path(__file__).resolve().parents[2]
    base = OmegaConf.to_container(OmegaConf.load(
        root / 'configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml'), resolve=True)
    arms = arm_configs(base, include_heuristic_only=True)
    assert tuple(arms) == ('curiosity_heuristic', 'curiosity_only', 'heuristic_only', 'random')
    expected = action_contract(base['raps_actor'])
    for name, config in arms.items():
        assert config['cabinet_item_layout'] == 'single_upper_second_row'
        assert config['cpu_sampling'] == {'workers': 2, 'streams': 2}
        assert config['wm_replay'] == base['wm_replay']
        assert config['wm_replay']['updates'] == 100
        assert config['wm_replay']['mode'] == 'bounded_dynamic'
        assert config['wm_training'] == {'learning_rate': .0001, 'position_loss_scale_m': .01}
        assert config['raps_actor']['reconfigure_on_reset'] is True
        assert config['cabinet_door'] is False and config['centered_initial_tcp'] is True
        assert skill_schema(config['raps_actor']).action_dim == 31
        assert config['raps_actor']['grasp_preflight'] is False
        assert 'grasp_preflight' not in action_contract(config['raps_actor'])
        contract = action_contract(config['raps_actor'])
        contract.pop('disagreement_weight', None)
        assert contract == expected
        assert config['bootstrap_steps'] + config['rounds'] * config['steps_per_round'] + config['evaluation_steps'] == 400000
        assert config['raps_actor']['imagination_horizon'] == 1
        assert config['raps_actor']['discount'] == 0.0
        assert config['collection_policy'] == ('random' if name == 'random' else 'curiosity')
        if name in ('curiosity_heuristic', 'heuristic_only'):
            assert config['object_heuristic']['object_weights'] == {'cabinet_item_16': 1.0}
        else:
            assert config['object_heuristic'] is None
    assert arms['heuristic_only']['raps_actor']['disagreement_weight'] == 0.
