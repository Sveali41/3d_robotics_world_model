"""Budgeted state-GNN/iCEM exploration in the existing ManiSkill cabinet.

Run ``python -m exploration.cee_us_cabinet --help``. Explicit ``compare`` or
``train`` starts training; ``smoke`` is capped at 300 real controller steps.
"""

import argparse
import copy
import csv
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from exploration.cee_us_metrics import interaction_summary, prediction_metrics, screen
from exploration.diagnose_cabinet import jsonable, snapshot, source_evidence, write_json

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "b0beaed1e9c3fe03d99aa300faae063dc208941e"
ACTION_SEMANTICS = "panda-ee-delta-reset-orientation-v2"


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def replay_arrays(rows):
    if not rows:
        raise ValueError("Cannot train or evaluate from empty replay")
    return {"states": np.asarray([r["state"] for r in rows], dtype=np.float32),
            "actions": np.asarray([r["action"] for r in rows], dtype=np.float32),
            "next_states": np.asarray([r["next_state"] for r in rows], dtype=np.float32),
            "episode_ids": np.asarray([r["episode_id"] for r in rows], dtype=np.int64),
            "terminated": np.asarray([r["terminated"] for r in rows], dtype=bool),
            "truncated": np.asarray([r["truncated"] for r in rows], dtype=bool)}


def allocate_replay(capacity):
    if capacity < 1:
        raise ValueError('Replay capacity must be positive')
    return {key: np.empty((capacity, *shape), dtype=dtype) for key, shape, dtype in (
        ('states', (66,), np.float32), ('actions', (4,), np.float32),
        ('next_states', (66,), np.float32), ('episode_ids', (), np.int64),
        ('terminated', (), bool), ('truncated', (), bool))}


def append_replay(storage, start, rows):
    end = start + len(rows)
    if start < 0 or end > len(storage['states']):
        raise ValueError('Replay append exceeds allocated capacity')
    if rows:
        for key, value in replay_arrays(rows).items():
            storage[key][start:end] = value
    return end


def replay_view(storage, count):
    if count < 1 or count > len(storage['states']):
        raise ValueError('Replay view must contain valid initialized transitions')
    return {key: value[:count] for key, value in storage.items()}


def make_env(*, project_goals_to_region=False, episode_steps=None, centered_initial_tcp=False,
             object_shift_xy=(0.0, 0.0)):
    from main import build_env
    cfg = OmegaConf.load(ROOT / "configs/exploration.yaml")
    cfg.action_mode = "ee_delta"
    cfg.exploration_mode = "baseline"
    cfg.project_goals_to_region = bool(project_goals_to_region)
    cfg.centered_initial_tcp = centered_initial_tcp
    if any(object_shift_xy):
        cfg.object_shift_xy = list(object_shift_xy)
    cfg.headless = True
    cfg.render_mode = None
    cfg.sim_backend = "cpu"
    cfg.sim.sim_freq, cfg.sim.control_freq = 100, 20
    cfg.ppo.record_dense_trajectory = False
    if episode_steps is not None:
        if isinstance(episode_steps, bool) or int(episode_steps) != episode_steps or episode_steps < 1:
            raise ValueError("episode_steps must be a positive integer")
        cfg.max_ee_steps_per_episode = int(episode_steps)
    return build_env(cfg), cfg


def prepare_output(output, config, env_config):
    output.mkdir(parents=True, exist_ok=False)
    evidence = source_evidence(output)
    versions = {}
    for name in ("torch", "numpy", "gymnasium", "scipy", "sapien", "mani_skill"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed as a distribution"
    write_json(output / "manifest.json", {
        "config": OmegaConf.to_container(config, resolve=True),
        "environment": OmegaConf.to_container(env_config, resolve=True),
        "commit": evidence["commit"], "source_fingerprint": evidence["fingerprint"],
        "cee_us_commit": REFERENCE_COMMIT, "dependencies": versions,
        "action_semantics": ACTION_SEMANTICS,
        "lift_metric": "continuous_grasp_height_v2",
        "privileged_state": True,
        "reward_used_for_planning": config.get("object_heuristic") is not None,
        "environment_reward_used_for_planning": False,
        "object_heuristic": OmegaConf.to_container(config.object_heuristic, resolve=True)
            if config.get("object_heuristic") is not None else None,
        "layout_randomized": False, "prediction_horizon_seconds": 1.5,
        "project_goals_to_region": bool(env_config.get("project_goals_to_region", False)),
        "runtime_optimizations": bool(config.get('runtime_optimizations', False)),
        "log_flush_interval_steps": 32 if config.get('runtime_optimizations', False) else 1,
        "log_durability": 'Normal close and handled exceptions flush all records; abrupt termination may lose buffered records.',
        "adaptations": ["Panda and cup/bowl replace Fetch and blocks", "20 Hz control",
                        "Robot joints and moving door included in global state",
                        "Object type and collision dimensions are static features",
                        "Shared random bootstrap differs from original zero-bootstrap CEE-US"],
    })
    (output / "working_tree.patch").write_text(evidence["diff"])
    write_json(output / "untracked_sources.json", evidence["untracked_sources"])
    OmegaConf.save(config, output / "config.yaml")


def save_checkpoint(path, backend, encoder, config, rows, state, stage, *, arrays=None, timings=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    replay_path = path.with_suffix(".replay.npz")
    began = time.perf_counter()
    arrays = replay_arrays(rows) if arrays is None else arrays
    if len(arrays['states']) != len(rows):
        raise ValueError('Checkpoint replay length differs from recorded history')
    if timings is not None:
        timings['array_build_s'] = time.perf_counter() - began
    began = time.perf_counter()
    np.savez_compressed(replay_path, **arrays)
    if timings is not None:
        timings['replay_save_s'] = time.perf_counter() - began
    began = time.perf_counter()
    payload = {"schema": "cabinet-cee-us-state-v1", "cee_us_commit": REFERENCE_COMMIT,
               "action_semantics": ACTION_SEMANTICS,
               "lift_metric": "continuous_grasp_height_v2",
               "backend": copy.deepcopy(backend.state_dict()), "rng": rng_state(),
               "state_contract": encoder.contract(), "last_state": state.copy(),
               "config": OmegaConf.to_container(config, resolve=True),
               "replay_path": replay_path.name, "replay_steps": len(rows), "stage": stage,
               "environment_state_restored": False}
    torch.save(payload, path)
    if timings is not None:
        timings['checkpoint_save_s'] = time.perf_counter() - began
    return payload


def read_checkpoint(path, backend):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "cabinet-cee-us-state-v1" or payload.get("cee_us_commit") != REFERENCE_COMMIT:
        raise ValueError("Not a compatible fixed-version CEE-US cabinet checkpoint")
    backend.load_state_dict(payload["backend"])
    restore_rng(payload["rng"])
    return payload


class Collector:
    """One real controller tick per transition, with no imagined physics calls."""

    def __init__(self, env, backend, encoder, directory, *, runtime_optimizations=False, profile=False):
        self.env, self.backend, self.encoder = env, backend, encoder
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.episode_id = 0
        self.total_steps = 0
        self.state = None
        self.runtime_optimizations = bool(runtime_optimizations)
        self.profile = bool(profile)
        self.timings = {"state_encoding_s": 0.0, "collection_s": 0.0}
        if self.profile:
            self.timings.update(dict.fromkeys(('simulation_s', 'snapshot_s', 'record_construction_s',
                                              'json_serialization_s', 'json_write_s', 'reset_s'), 0.))

    def reset(self, seed):
        began = time.perf_counter() if self.profile else 0.
        self.env.reset(seed=int(seed))
        self.episode_id += 1
        self.state = self.encoder.encode()
        self.backend.reset_planner(self.state)
        if self.profile:
            self.timings['reset_s'] += time.perf_counter() - began

    def collect(self, count, policy, *, seed, stage):
        if count < 1 or policy not in {"random", "curiosity"}:
            raise ValueError("Positive step budget and random/curiosity policy required")
        self.reset(seed)
        rows = []
        start = time.perf_counter()
        with (self.directory / "transitions.jsonl").open("a", encoding="utf-8") as handle:
            for step in range(count):
                began = time.perf_counter() if self.profile else 0.
                before = snapshot(self.env)
                if self.profile:
                    self.timings['snapshot_s'] += time.perf_counter() - began
                current = self.state.copy()
                if policy == "curiosity":
                    requested, diagnostics = self.backend.action(current)
                else:
                    requested, diagnostics = self.backend.random_action(current), {}
                requested = np.asarray(requested, dtype=np.float32)
                if requested.shape != (4,) or not np.isfinite(requested).all():
                    raise ValueError("Planner emitted a non-finite or non-4D action")
                began = time.perf_counter() if self.profile else 0.
                _, _, terminated, truncated, info = self.env.step(requested)
                if self.profile:
                    self.timings['simulation_s'] += time.perf_counter() - began
                record = info["primitive_record"]
                self.total_steps += record.duration_low_steps
                if record.duration_low_steps != 1:
                    raise RuntimeError("CEE-US requires exactly one real control step per action")
                actual = np.asarray(record.low_actions[0], dtype=np.float32)
                t0 = time.perf_counter()
                self.state = self.encoder.encode()
                self.timings["state_encoding_s"] += time.perf_counter() - t0
                began = time.perf_counter() if self.profile else 0.
                after = snapshot(self.env)
                if self.profile:
                    self.timings['snapshot_s'] += time.perf_counter() - began
                    began = time.perf_counter()
                row = jsonable({
                    "step": self.total_steps, "stage": stage, "policy": policy,
                    "episode_id": self.episode_id, "state": current, "next_state": self.state,
                    "requested_action": requested, "action": actual,
                    "full_controller_action": record.full_controller_actions[0],
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "run_budget_exhausted": step == count - 1,
                    "primitive_type": "ee_delta", "duration_low_steps": 1,
                    "parameters_physical": dict(zip(("dx", "dy", "dz", "gripper"), actual.tolist())),
                    "action_units": "normalized_controller", "status": record.status,
                    "contact_events": record.contact_events, "outcome_metrics": record.outcome_metrics,
                    "before": before, "after": after, "reward": info["exploration_reward"],
                    "planner": diagnostics,
                })
                if getattr(self.backend, "object_heuristic", None) is not None:
                    row["object_heuristic_observed"] = self.backend.observed_object_heuristic(current, self.state)
                if self.profile:
                    self.timings['record_construction_s'] += time.perf_counter() - began
                    began = time.perf_counter()
                serialized = json.dumps(row, allow_nan=False) + "\n"
                if self.profile:
                    self.timings['json_serialization_s'] += time.perf_counter() - began
                    began = time.perf_counter()
                handle.write(serialized)
                if (not self.runtime_optimizations or (step + 1) % 32 == 0
                        or terminated or truncated or step + 1 == count):
                    handle.flush()
                if self.profile:
                    self.timings['json_write_s'] += time.perf_counter() - began
                rows.append(row)
                if (terminated or truncated) and step + 1 < count:
                    self.reset(seed + self.episode_id)
        self.timings["collection_s"] += time.perf_counter() - start
        return rows


def train_round(backend, rows, epochs, directory, label, *, arrays=None):
    t0 = time.perf_counter()
    metrics = backend.train(replay_arrays(rows) if arrays is None else arrays, epochs=epochs)
    result = {"label": label, "replay_steps": len(rows), "epochs": epochs,
              "seconds": time.perf_counter() - t0, "metrics": metrics}
    with (directory / "training.jsonl").open("a") as handle:
        handle.write(json.dumps(jsonable(result), allow_nan=False) + "\n")
    return result


def checkpoint_check(path, backend, encoder, config, rows, state):
    payload = save_checkpoint(path, backend, encoder, config, rows, state, "smoke")
    action = np.zeros((1, 4), dtype=np.float32)
    expected = backend.predict(state[None], action)
    numpy_expected, torch_expected = np.random.random(4), torch.rand(4)
    read_checkpoint(path, backend)
    actual = backend.predict(state[None], action)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.random.random(4), numpy_expected)
    torch.testing.assert_close(torch.rand(4), torch_expected, rtol=0, atol=0)
    restore_rng(payload["rng"])
    return {"prediction_equal": True, "rng_equal": True,
            "simulation_resume": "New episodes reset physics; no exact simulator resume claimed"}


def smoke(env, backend, encoder, config, output):
    collector = Collector(env, backend, encoder, output / "smoke",
                          runtime_optimizations=bool(config.get('runtime_optimizations', False)))
    rows = collector.collect(256, "random", seed=config.seed, stage="bootstrap")
    fits = [train_round(backend, rows, 2, collector.directory, "bootstrap")]
    rows.extend(collector.collect(8, "curiosity", seed=config.seed + 1000, stage="round_1"))
    fits.append(train_round(backend, rows, 1, collector.directory, "round_1"))
    rows.extend(collector.collect(8, "curiosity", seed=config.seed + 2000, stage="round_2"))
    if collector.total_steps > 300:
        raise AssertionError("Smoke exceeded its 300-step physical budget")
    checks = checkpoint_check(output / "smoke/checkpoint.pt", backend, encoder,
                              config, rows, collector.state)
    result = {"status": "wiring_pass", "real_steps": collector.total_steps,
              "checkpoint": checks, "training": fits, "timings": collector.timings,
              "interactions": interaction_summary(rows),
              "effectiveness": "not_evaluated_by_smoke",
              "interaction_dynamics": "not_validated_on_independent_moving_samples"}
    decision_times = [r["planner"].get("planning_seconds") for r in rows if r["planner"]]
    decision_times = [float(v) for v in decision_times if v is not None]
    if decision_times:
        result["planning_seconds_mean"] = float(np.mean(decision_times))
        result["planning_seconds_p95"] = float(np.quantile(decision_times, .95))
        result["estimated_4000_curiosity_decisions_seconds"] = float(np.mean(decision_times) * 4000)
    if torch.cuda.is_available() and config.device == "cuda":
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_json(output / "smoke/result.json", result)
    print(json.dumps({key: jsonable(value) for key, value in result.items()
                      if key not in {"interactions", "training"}}, indent=2, allow_nan=False))


def compare(env, backend, encoder, config, output):
    optimized = bool(config.get('runtime_optimizations', False))
    capacity = int(config.bootstrap_steps) + int(config.rounds) * int(config.steps_per_round)
    bootstrap = Collector(env, backend, encoder, output / "bootstrap", runtime_optimizations=optimized)
    common = bootstrap.collect(int(config.bootstrap_steps), "random", seed=config.seed, stage="bootstrap")
    cache = allocate_replay(capacity) if optimized else None
    options = {}
    if optimized:
        append_replay(cache, 0, common)
        options['arrays'] = replay_view(cache, len(common))
    train_round(backend, common, int(config.epochs), bootstrap.directory, "bootstrap", **options)
    initial = save_checkpoint(output / "bootstrap/checkpoint.pt", backend, encoder,
                              config, common, bootstrap.state, "bootstrap", **options)
    checkpoints, evaluations = {}, {}
    total = bootstrap.total_steps
    for policy in ("random", "curiosity"):
        backend.load_state_dict(copy.deepcopy(initial["backend"]))
        restore_rng(initial["rng"])
        collector = Collector(env, backend, encoder, output / policy, runtime_optimizations=optimized)
        rows = list(common)
        if optimized:
            # Each arm owns storage; bootstrap is its only shared data.
            cache = allocate_replay(capacity)
            append_replay(cache, 0, common)
        # Shared bootstrap and each arm use distinct episode namespaces in replay.
        collector.episode_id = max(r["episode_id"] for r in common)
        for index in range(int(config.rounds)):
            latest = collector.collect(int(config.steps_per_round), policy,
                                       seed=config.seed + 10000 * (index + 1), stage=f"round_{index + 1}")
            rows.extend(latest)
            if optimized:
                append_replay(cache, len(rows) - len(latest), latest)
                options['arrays'] = replay_view(cache, len(rows))
            fit = train_round(backend, rows, int(config.epochs), collector.directory, f"round_{index + 1}", **options)
            path = collector.directory / f"round_{index + 1:03d}.pt"
            save_checkpoint(path, backend, encoder, config, rows, collector.state, f"round_{index + 1}", **options)
            write_json(path.with_suffix(".json"), {"interactions": interaction_summary(latest), "training": fit})
        checkpoints[policy] = path
        evaluation = collector.collect(int(config.evaluation_steps), policy,
                                       seed=config.seed + 100000, stage="evaluation")
        evaluations[policy] = evaluation
        write_json(collector.directory / "evaluation.json", interaction_summary(evaluation))
        total += collector.total_steps
    # Both models see the same independent RANDOM evaluation trajectories for prediction metrics.
    holdout = replay_arrays(evaluations["random"])
    for policy, path in checkpoints.items():
        read_checkpoint(path, backend)
        write_json(output / policy / "prediction.json", prediction_metrics(backend, holdout))
    expected = int(config.bootstrap_steps) + 2 * (
        int(config.rounds) * int(config.steps_per_round) + int(config.evaluation_steps))
    if total != expected:
        raise AssertionError(f"Real-step accounting mismatch: {total} != {expected}")
    write_json(output / "budget.json", {"real_steps": total, "expected": expected})
    report(output)


def train(env, backend, encoder, config, output):
    """Fresh single-arm curiosity run; evaluation never enters training replay."""
    optimized = bool(config.get('runtime_optimizations', False))
    collector = Collector(env, backend, encoder, output / "curiosity", runtime_optimizations=optimized)
    rows = collector.collect(int(config.bootstrap_steps), "random", seed=config.seed, stage="bootstrap")
    options = {}
    if optimized:
        cache = allocate_replay(int(config.bootstrap_steps) + int(config.rounds) * int(config.steps_per_round))
        append_replay(cache, 0, rows)
        options['arrays'] = replay_view(cache, len(rows))
    train_round(backend, rows, int(config.epochs), collector.directory, "bootstrap", **options)
    save_checkpoint(collector.directory / "bootstrap.pt", backend, encoder, config,
                    rows, collector.state, "bootstrap", **options)
    for index in range(1, int(config.rounds) + 1):
        write_json(output / "progress.json", {"stage": "collecting", "round": index,
                   "real_steps": collector.total_steps, "training_transitions": len(rows)})
        latest = collector.collect(int(config.steps_per_round), "curiosity",
                                   seed=int(config.seed) + 10000 * index, stage=f"round_{index}")
        rows.extend(latest)
        if optimized:
            append_replay(cache, len(rows) - len(latest), latest)
            options['arrays'] = replay_view(cache, len(rows))
        write_json(output / "progress.json", {"stage": "training", "round": index,
                   "real_steps": collector.total_steps, "training_transitions": len(rows)})
        fit = train_round(backend, rows, int(config.epochs), collector.directory, f"round_{index}", **options)
        path = collector.directory / f"round_{index:03d}.pt"
        save_checkpoint(path, backend, encoder, config, rows, collector.state, f"round_{index}", **options)
        summary = interaction_summary(latest)
        write_json(path.with_suffix(".json"), {"interactions": summary, "training": fit})
        print(json.dumps({"round": index, "training_transitions": len(rows),
              "contacts": summary["contact_primitives_per_1000_steps"],
              "motion": summary["recoverable_motion_primitives_per_1000_steps"],
              "grasp_rate": summary["grasp_primitive_rate"], "train_seconds": fit["seconds"]}), flush=True)
    expected_training = int(config.bootstrap_steps) + int(config.rounds) * int(config.steps_per_round)
    if len(rows) != expected_training or collector.total_steps != expected_training:
        raise AssertionError("Single-arm training real-step accounting mismatch")
    write_json(output / "progress.json", {"stage": "evaluation", "real_steps": collector.total_steps,
               "training_transitions": len(rows)})
    evaluation = collector.collect(int(config.evaluation_steps), "curiosity",
                                   seed=int(config.seed) + 100000, stage="evaluation")
    write_json(collector.directory / "evaluation.json", interaction_summary(evaluation))
    write_json(collector.directory / "prediction.json", prediction_metrics(backend, replay_arrays(evaluation)))
    if collector.total_steps != expected_training + int(config.evaluation_steps):
        raise AssertionError("Single-arm evaluation real-step accounting mismatch")
    write_json(output / "budget.json", {"training_transitions": len(rows),
               "evaluation_steps": len(evaluation), "real_steps": collector.total_steps})
    write_json(output / "result.json", {"status": "completed", "training_transitions": len(rows),
               "real_steps": collector.total_steps, "interactions": interaction_summary(evaluation),
               "prediction_holdout_policy": "on_policy_curiosity", "matched_random_baseline": False})
    write_json(output / "progress.json", {"stage": "completed", "real_steps": collector.total_steps,
               "training_transitions": len(rows)})


def evaluate(env, backend, encoder, config, output, checkpoint, policy):
    payload = read_checkpoint(checkpoint, backend)
    if payload.get("action_semantics") != ACTION_SEMANTICS:
        raise ValueError("Checkpoint uses legacy wrist control; retain it for offline analysis and train a new model for reset-orientation control")
    # State dimensions, units, robot root, and collision geometry must agree.
    if encoder.contract() != payload["state_contract"]:
        raise ValueError("Checkpoint state contract differs from the current environment")
    collector = Collector(env, backend, encoder, output / "evaluation",
                          runtime_optimizations=bool(config.get('runtime_optimizations', False)))
    rows = collector.collect(int(config.evaluation_steps), policy,
                             seed=config.seed + 100000, stage="evaluation")
    write_json(collector.directory / "result.json", {
        "interactions": interaction_summary(rows), "policy": policy,
        "predictive_evaluation_policy": "random" if policy == "random" else "on_policy_curiosity",
        "prediction": prediction_metrics(backend, replay_arrays(rows)),
        "timings": collector.timings, "model_updated": False})


def report(output):
    summaries, predictions = {}, {}
    for policy in ("random", "curiosity"):
        path = output / policy / "evaluation.json"
        if path.exists():
            summaries[policy] = json.loads(path.read_text())
        path = output / policy / "prediction.json"
        if path.exists():
            predictions[policy] = json.loads(path.read_text())
    verdict = (screen(summaries["random"], summaries["curiosity"])
               if len(summaries) == 2 else {"status": "incomplete", "reason": "Both evaluations required"})
    write_json(output / "report.json", {"screening": verdict, "interactions": summaries,
                                        "prediction_on_shared_random_holdout": predictions})
    keys = ("low_level_steps", "contact_primitives_per_1000_steps", "motion_primitives_per_1000_steps",
            "recoverable_motion_primitives_per_1000_steps", "recoverable_coverage_auc",
            "failed_state_primitive_rate", "endpoint_outside_primitive_rate")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("policy", *keys))
        writer.writeheader()
        for policy, value in summaries.items():
            writer.writerow({"policy": policy, **{key: value[key] for key in keys}})
    print(json.dumps(verdict, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "compare", "train", "evaluate", "report"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cee_us_cabinet_120k_total_heuristic.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--object-shift-xy", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"),
                        help="Frozen evaluation only: move both objects in cabinet XY (meters); +X is deeper")
    parser.add_argument("--evaluation-steps", type=int, help="Frozen evaluation control-step budget")
    parser.add_argument("--policy", choices=("random", "curiosity"), default="curiosity")
    parser.add_argument("--agent", choices=("icem", "raps_gnn_actor"),
                        help="Exploration implementation; omitted keeps the configured agent, default iCEM")
    args = parser.parse_args(argv)
    if (any(args.object_shift_xy) or args.evaluation_steps is not None) and args.stage != "evaluate":
        parser.error("Position shift and evaluation budget overrides require evaluate")
    if not np.isfinite(args.object_shift_xy).all():
        parser.error("Position shifts must be finite")
    if args.evaluation_steps is not None and args.evaluation_steps < 1:
        parser.error("evaluation-steps must be positive")
    selected_agent = args.agent or OmegaConf.load(args.config).get("agent", "icem")
    if selected_agent == "raps_gnn_actor":
        from exploration.raps_gnn_actor import run
        return run(args, OmegaConf.load(args.config))
    if selected_agent != "icem":
        parser.error(f"Unknown exploration agent: {selected_agent}")
    if args.stage == "report":
        if args.output is None or not args.output.is_dir():
            parser.error("report requires an existing --output directory")
        report(args.output)
        return 0
    if args.stage == "evaluate" and args.checkpoint is None:
        parser.error("evaluate requires --checkpoint")
    config = OmegaConf.load(args.config)
    if args.evaluation_steps is not None:
        config.evaluation_steps = args.evaluation_steps
    if args.agent is not None:
        config.agent = args.agent
    if args.device:
        config.device = args.device
    for key in ("bootstrap_steps", "rounds", "steps_per_round", "evaluation_steps", "epochs"):
        if int(config[key]) < 1:
            parser.error(f"{key} must be positive")
    if (config.action_mode != "ee_delta" or config.exploration_mode != "baseline"
            or config.sim_freq != 100 or config.control_freq != 20):
        parser.error("This diagnostic requires ee_delta, no D projection, and 100/20 Hz")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for a CPU diagnostic")
    candidates = int(config.get("planner_candidates", 128))
    if candidates < 20:
        parser.error("planner_candidates must be >=20 to retain ten elites")
    threads = int(config.get("cpu_threads", 1))
    if threads < 1:
        parser.error("cpu_threads must be positive")
    execution = str(config.get("execution", "reference"))
    if execution not in {"reference", "cuda_graph"}:
        parser.error("execution must be reference or cuda_graph")
    if execution == "cuda_graph" and config.device != "cuda":
        parser.error("cuda_graph execution requires device=cuda")
    torch.set_num_threads(threads)
    random.seed(int(config.seed)); np.random.seed(int(config.seed)); torch.manual_seed(int(config.seed))
    from exploration.cee_us_backend import CEEUSBackend
    from exploration.cee_us_state import CabinetStateEncoder
    output = args.output or ROOT / "logs" / time.strftime("cee_us_%Y%m%dT%H%M%S")
    if output.exists():
        raise FileExistsError(f"Use a new output directory; refusing to overwrite {output}")
    env = backend = None
    prepared = False
    try:
        env, env_config = make_env(project_goals_to_region=bool(config.get("project_goals_to_region", False)),
                                   object_shift_xy=args.object_shift_xy)
        env.reset(seed=int(config.seed))
        encoder = CabinetStateEncoder(env)
        prepare_output(output, config, env_config)
        prepared = True
        write_json(output / "state_contract.json", encoder.contract())
        backend = CEEUSBackend(device=str(config.device), seed=int(config.seed),
                               candidates=candidates, execution=execution,
                               action_projection=(encoder.action_projection_contract()
                                   if env_config.project_goals_to_region else None),
                               object_heuristic=(OmegaConf.to_container(config.object_heuristic, resolve=True)
                                   if config.get("object_heuristic") is not None else None),
                               runtime_optimizations=bool(config.get('runtime_optimizations', False)))
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if args.stage == "smoke":
            smoke(env, backend, encoder, config, output)
        elif args.stage == "compare":
            compare(env, backend, encoder, config, output)
        elif args.stage == "train":
            train(env, backend, encoder, config, output)
        else:
            evaluate(env, backend, encoder, config, output, args.checkpoint, args.policy)
    except BaseException as exc:
        if prepared:
            write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                "status": "stopped_without_random_fallback"})
        raise
    finally:
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
