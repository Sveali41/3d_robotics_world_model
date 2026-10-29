"""Frozen-model planner latency benchmark and explicit two-arm physical screening.

Benchmark performs no physics or training. Timing is not evidence of exploration
quality; CPU/CUDA random streams are not a numerical parity test. Profile fits
isolated checkpoint copies for three paired one-epoch repeats and defaults to
zero physical steps.
"""

import argparse
import contextlib
import csv
import gc
import hashlib
import io
import platform
import resource
from types import SimpleNamespace
import time
import warnings
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from exploration.cee_us_cabinet import (
    Collector, REFERENCE_COMMIT, ROOT, make_env, prepare_output, restore_rng, rng_state,
)
from exploration.cee_us_metrics import interaction_summary, screen
from exploration.cee_us_state import CabinetStateEncoder
from exploration.diagnose_cabinet import write_json


def make_backend(**kwargs):
    from exploration.cee_us_backend import CEEUSBackend
    return CEEUSBackend(**kwargs)


def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "cabinet-cee-us-state-v1"
            or payload.get("cee_us_commit") != REFERENCE_COMMIT):
        raise ValueError("Not a compatible fixed-version CEE-US cabinet checkpoint")
    return payload


def checkpoint_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"checkpoint": str(path.resolve()), "checkpoint_sha256": digest.hexdigest()}


def latency_summary(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Latency samples must be nonempty and finite")
    return {"mean": float(values.mean()), "p95": float(np.quantile(values, .95)),
            "samples": values.tolist()}


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_states(checkpoint, payload, count):
    replay_path = checkpoint.parent / payload["replay_path"]
    with np.load(replay_path, allow_pickle=False) as replay:
        states = replay["states"]
        if states.ndim != 2 or len(states) < count or not np.isfinite(states).all():
            raise ValueError(f"Replay must contain at least {count} finite saved states")
        indices = np.linspace(0, len(states) - 1, count, dtype=int)
        return states[indices].copy(), indices.tolist(), str(replay_path.resolve())


def checkpoint_backend_options(payload):
    """Older checkpoints omit these optional construction contracts."""
    return {key: payload["backend"][key] for key in ("object_heuristic", "action_projection")
            if key in payload["backend"]}


def benchmark(args, payload, config):
    states, indices, replay_path = selected_states(args.checkpoint, payload, args.states)
    prepare_output(args.output, config, OmegaConf.create({"physics": "not_created"}))
    modes = [("cpu", 1, "reference", 128), ("cpu", 4, "reference", 128)]
    if args.device != "cpu":
        modes += [("cuda", 1, "reference", 128), ("cuda", 1, "cuda_graph", 128),
                  ("cuda", 1, "cuda_graph", 64)]
    results, csv_rows = [], []
    for device, threads, execution, candidates in modes:
        torch.set_num_threads(threads)
        backend = make_backend(device=device, seed=0, execution=execution, candidates=candidates,
                               **checkpoint_backend_options(payload))
        try:
            backend.load_state_dict(payload["backend"], restore_planner=False)
            restore_rng({**payload["rng"], "cuda": []} if device == "cpu" else payload["rng"])
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            warmup, elapsed, rollout, planner = [], [], [], []
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                for index in range(args.warmup + len(states)):
                    state = states[index % len(states)] if index < args.warmup else states[index - args.warmup]
                    # Upstream beginning_of_rollout prints a reset banner on every call.
                    with contextlib.redirect_stdout(io.StringIO()):
                        backend.reset_planner(state)
                    synchronize(device)
                    start = time.perf_counter()
                    _, diagnostics = backend.action(state)
                    synchronize(device)
                    seconds = time.perf_counter() - start
                    if index < args.warmup:
                        warmup.append(seconds)
                    else:
                        elapsed.append(seconds)
                        rollout.append(float(diagnostics["candidate_rollout_seconds"]))
                        planner.append(float(diagnostics["planning_seconds"]))
            entry = {"device": device, "threads": threads, "execution": execution,
                     "candidates": candidates, "decisions": len(states),
                     "warmup_including_capture_seconds": warmup,
                     "first_decision_seconds": warmup[0],
                     "capture_timing": "Included in warmup; not isolated from planner work",
                     "full_action_seconds": latency_summary(elapsed),
                     "planner_reported_seconds": latency_summary(planner),
                     "candidate_rollout_seconds": latency_summary(rollout),
                     "candidate_rollout_timing": "Host dispatch time; GPU completion is included only in full_action_seconds",
                     "warnings": [str(item.message) for item in recorded],
                     "peak_gpu_allocated_bytes_torch_only": (torch.cuda.max_memory_allocated() if device == "cuda" else None),
                     "peak_gpu_reserved_bytes_torch_only": (torch.cuda.max_memory_reserved() if device == "cuda" else None)}
            results.append(entry)
            csv_rows.append({key: entry[key] for key in ("device", "threads", "execution", "candidates", "decisions")}
                            | {"full_action_mean_s": entry["full_action_seconds"]["mean"],
                               "full_action_p95_s": entry["full_action_seconds"]["p95"],
                               "candidate_rollout_host_mean_s": entry["candidate_rollout_seconds"]["mean"],
                               "warmup_including_capture_s": sum(warmup),
                               "peak_gpu_allocated_bytes_torch_only": entry["peak_gpu_allocated_bytes_torch_only"],
                               "peak_gpu_reserved_bytes_torch_only": entry["peak_gpu_reserved_bytes_torch_only"]})
        finally:
            del backend
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    result = {**checkpoint_info(args.checkpoint), "state_indices": indices, "replay_path": replay_path,
              "real_steps": 0, "training_updates": 0, "modes": results,
              "machine": {"platform": platform.platform(), "processor": platform.processor(),
                          "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                          "cuda_device": torch.cuda.get_device_name() if args.device != "cpu" else None},
              "limitations": "Timing does not prove exploration quality. CPU/CUDA RNG differs; this is not numerical parity."}
    write_json(args.output / "benchmark.json", result)
    write_csv(args.output / "benchmark.csv", csv_rows)
    return result


def compare(args, payload, config):
    initial_rng = rng_state()
    results, csv_rows = {}, []
    prepared = False
    for candidates in (128, 64):
        env = backend = collector = None
        label = f"optimized{candidates}"
        try:
            restore_rng(initial_rng)
            env, env_config = make_env()
            env.reset(seed=0)
            encoder = CabinetStateEncoder(env)
            if encoder.contract() != payload["state_contract"]:
                raise ValueError("Checkpoint state contract does not match the current cabinet encoder")
            if not prepared:
                prepare_output(args.output, config, env_config)
                write_json(args.output / "state_contract.json", encoder.contract())
                prepared = True
            backend = make_backend(device="cuda", seed=0, execution="cuda_graph", candidates=candidates,
                                   **checkpoint_backend_options(payload))
            backend.load_state_dict(payload["backend"], restore_planner=False)
            # Model construction seeds Torch; restore identical initial streams after loading.
            restore_rng(initial_rng)
            collector = Collector(env, backend, encoder, args.output / label)
            synchronize("cuda")
            start = time.perf_counter()
            rows = collector.collect(args.steps, "curiosity", seed=0, stage=label)
            synchronize("cuda")
            seconds = time.perf_counter() - start
            if collector.total_steps != args.steps:
                raise RuntimeError(f"{label} exceeded or missed its exact real-step budget")
            interactions = interaction_summary(rows)
            planning = [float(row["planner"]["planning_seconds"]) for row in rows]
            entry = {"candidates": candidates, "execution": "cuda_graph", "seed": 0,
                     "real_steps": collector.total_steps, "training_updates": 0,
                     "model_frozen": True, "weights_saved": False,
                     **checkpoint_info(args.checkpoint), "wall_seconds": seconds,
                     "planning_seconds": latency_summary(planning), "interactions": interactions}
            results[label] = entry
            write_json(args.output / label / "summary.json", entry)
            csv_rows.append({"arm": label, "candidates": candidates, "real_steps": collector.total_steps,
                             "training_updates": 0, "wall_seconds": seconds,
                             "planning_seconds_mean": entry["planning_seconds"]["mean"],
                             **{key: interactions[key] for key in (
                                 "recoverable_coverage_auc", "recoverable_motion_primitives_per_1000_steps")}})
        finally:
            if env is not None:
                env.close()
            del collector, backend
            gc.collect()
            torch.cuda.empty_cache()
    verdict = screen(results["optimized128"]["interactions"], results["optimized64"]["interactions"])
    verdict["comparison"] = "optimized64 relative to optimized128; single seed screening only"
    quality_not_degraded = (verdict["both_objects_moved_in_recovery_range"] and all(
        results["optimized64"]["interactions"][key] >= results["optimized128"]["interactions"][key]
        for key in ("recoverable_coverage_auc", "recoverable_motion_primitives_per_1000_steps")))
    result = {"arms": results, "total_real_steps": 2 * args.steps, "training_updates": 0,
              "quality_not_degraded": quality_not_degraded,
              "quality_limitation": "Descriptive single-seed screen; no statistical noninferiority claim",
              "screen": verdict, "selection": "No automatic planner selection"}
    write_json(args.output / "comparison.json", result)
    write_csv(args.output / "comparison.csv", csv_rows)
    return result


def parity(left, right, path="root"):
    """Compare complete nested model/normalizer/optimizer snapshots."""
    failures = []
    if torch.is_tensor(left) or isinstance(left, np.ndarray):
        a, b = torch.as_tensor(left), torch.as_tensor(right)
        if a.shape != b.shape or not torch.allclose(a, b, rtol=1e-6, atol=1e-7):
            failures.append(path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            return [path + ".keys"]
        for key in left:
            failures.extend(parity(left[key], right[key], f"{path}.{key}"))
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return [path + ".length"]
        for index, (a, b) in enumerate(zip(left, right)):
            failures.extend(parity(a, b, f"{path}.{index}"))
    elif isinstance(left, float):
        if not np.isclose(left, right, rtol=1e-6, atol=1e-7):
            failures.append(path)
    elif left != right:
        failures.append(path)
    return failures


def timed(device, operation):
    synchronize(device)
    start = time.perf_counter()
    events = None
    if device == "cuda":
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        events[0].record()
    value = operation()
    if events:
        events[1].record()
    synchronize(device)
    return value, time.perf_counter() - start, (events[0].elapsed_time(events[1]) / 1000 if events else None)


def profile_summary(results, passed):
    summaries = {}
    for mode in ("original", "optimized"):
        samples = [row for row in results if row["mode"] == mode]
        summaries[mode] = {}
        for key in ("training_s", "array_build_s", "save_wall_s", "profiled_round_s"):
            values = np.asarray([row[key] for row in samples])
            summaries[mode][key] = {"median": float(np.median(values)),
                                    "p95": float(np.quantile(values, .95))}
    ratios = {}
    for key in summaries["original"]:
        original = summaries["original"][key]["median"]
        optimized = summaries["optimized"][key]["median"]
        ratios[key] = {"original_over_optimized": original / optimized if optimized else None,
                       "median_improvement_fraction": 1 - optimized / original if original else 0.}
    improvement = ratios["profiled_round_s"]["median_improvement_fraction"]
    return {"modes": summaries, "ratios": ratios,
            "recommendation": "enable_runtime_optimizations" if passed and improvement >= .05 else "retain_original",
            "recommendation_basis": "Parity passes and median training plus checkpoint/replay save wall time improves at least 5%",
            "recommendation_parity_passed": passed}


def profile(args, payload, config):
    from exploration import cee_us_cabinet as cabinet
    states, indices, replay_path = selected_states(args.checkpoint, payload, 12)
    with np.load(replay_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in
                  ("states", "actions", "next_states", "episode_ids", "terminated", "truncated")}
    count = len(arrays["states"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("Replay arrays have inconsistent transition counts")
    rows = [{"state": arrays["states"][i], "action": arrays["actions"][i],
             "next_state": arrays["next_states"][i], "episode_id": arrays["episode_ids"][i],
             "terminated": arrays["terminated"][i], "truncated": arrays["truncated"][i]}
            for i in range(count)]
    prepare_output(args.output, config, OmegaConf.create({"physics": "optional_physical_smoke_only"}))
    initial_rng = {**payload["rng"], "cuda": []} if args.device == "cpu" else payload["rng"]
    cache_start = time.perf_counter()
    storage = cabinet.allocate_replay(count)
    cached_count = cabinet.append_replay(storage, 0, rows)
    cache_initialization_s = time.perf_counter() - cache_start
    results, csv_rows, checks = [], [], []
    for repeat in range(3):
        snapshots = {}
        for optimized in ((False, True) if repeat % 2 == 0 else (True, False)):
            label = "optimized" if optimized else "original"
            execution = "cuda_graph" if args.device == "cuda" else "reference"
            backend = make_backend(device=args.device, seed=0, execution=execution, candidates=128,
                                   runtime_optimizations=optimized, **checkpoint_backend_options(payload))
            try:
                backend.load_state_dict(payload["backend"], restore_planner=False)
                restore_rng(initial_rng)
                backend.profile_timings = {}
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                metrics, training_s, training_cuda_s = timed(args.device, lambda: backend.train(arrays, epochs=1))
                if int(metrics["updates"]) <= 0:
                    raise RuntimeError("Profile requires actual isolated optimizer updates on full replay")
                saved_state = backend.state_dict()
                snapshots[label] = {key: saved_state[key] for key in ("model", "normalizers", "optimizer")}
                snapshots[label]["loss"] = float(metrics["loss"])
                # Capture training state before planner warmup mutates planner state.
                restore_rng(initial_rng)
                accelerator = getattr(backend, "prediction_accelerator", None)
                if accelerator is not None:
                    accelerator.profile_timings = {}
                warmup, decisions, planner_actions = [], [], []
                with contextlib.redirect_stdout(io.StringIO()):
                    for index in range(14):
                        state = states[index % 12] if index < 2 else states[index - 2]
                        backend.reset_planner(state)
                        (action, _), elapsed, _ = timed(args.device, lambda: backend.action(state))
                        (warmup if index < 2 else decisions).append(elapsed)
                        if index >= 2:
                            planner_actions.append(action)
                snapshots[label]["planner_actions"] = np.asarray(planner_actions)
                def build_arrays():
                    if optimized:
                        return cabinet.replay_view(storage, cached_count)
                    return cabinet.replay_arrays(rows)
                built, array_s, _ = timed("cpu", build_arrays)
                save_timings = {}
                scratch = args.output / "scratch" / f"{repeat}_{label}.pt"
                encoder = SimpleNamespace(contract=lambda: payload["state_contract"])
                _, save_s, _ = timed(args.device, lambda: cabinet.save_checkpoint(
                    scratch, backend, encoder, OmegaConf.merge(config, {"runtime_optimizations": optimized}),
                    rows, arrays["next_states"][-1], "profile",
                    arrays=built if optimized else None, timings=save_timings))
                entry = {"repeat": repeat, "mode": label, "device": args.device, "execution": execution,
                         "epochs": 1, "replay_steps": count, "optimizer_updates": int(metrics["updates"]),
                         "loss": float(metrics["loss"]), "training_s": training_s,
                         "training_cuda_event_s": training_cuda_s, "training_parts_s": dict(backend.profile_timings),
                         "array_build_s": array_s, "save_wall_s": save_s, "save_parts_s": save_timings,
                         "profiled_round_s": training_s + save_s,
                         "warmup_including_capture_s": warmup,
                         "graph_capture": dict(accelerator.profile_timings) if accelerator is not None else {},
                         "warmup_excluding_capture_s": max(0., sum(warmup) - (
                             sum(accelerator.profile_timings.get("capture_wall_seconds", [])) if accelerator is not None else 0.)),
                         "capture_timing": "Capture measured by accelerator; also included in warmup wall time",
                         "steady_decision_s": latency_summary(decisions),
                         "host_rss_high_water_bytes_process_lifetime": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                         "peak_gpu_allocated_bytes_torch_only": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                         "peak_gpu_reserved_bytes_torch_only": torch.cuda.max_memory_reserved() if args.device == "cuda" else None}
                results.append(entry)
                csv_rows.append({key: entry[key] for key in ("repeat", "mode", "device", "execution", "optimizer_updates", "loss", "training_s", "array_build_s", "save_wall_s", "host_rss_high_water_bytes_process_lifetime", "peak_gpu_allocated_bytes_torch_only")}
                                | {"steady_decision_mean_s": entry["steady_decision_s"]["mean"],
                                   "warmup_including_capture_s": sum(warmup)})
            finally:
                accelerator = None
                del backend
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
        failures = parity(snapshots["original"], snapshots["optimized"])
        checks.append({"repeat": repeat, "passed": not failures, "mismatches": failures})
    physical = profile_physical(args, payload, initial_rng) if args.physical_steps else None
    result = {**checkpoint_info(args.checkpoint), "replay_path": replay_path, "state_indices": indices,
              "modes": results, "parity": checks, "rtol": 1e-6, "atol": 1e-7,
              "optimized_replay_cache_initialization_s": cache_initialization_s,
              "passed": all(check["passed"] for check in checks) and (physical is None or physical["passed"]),
              "physical": physical, "real_steps": 2 * args.physical_steps,
              "training_updates": sum(row["optimizer_updates"] for row in results),
              "training_scope": "Isolated checkpoint copies; one full-replay epoch per mode per repeat",
              "source_manifest": "manifest.json", "source_diff": "working_tree.patch",
              "memory_scope": "Torch GPU allocator only; host RSS is process-lifetime high-water, not per-mode allocation",
              "timing_scope": "Synchronized full boundaries; CUDA events enclose training GPU stream work"}
    result["summary"] = profile_summary(results, result["passed"])
    write_json(args.output / "profile.json", result)
    write_csv(args.output / "profile.csv", csv_rows)
    return result


def profile_physical(args, payload, initial_rng):
    arms = {}
    for optimized in (False, True):
        label = "optimized" if optimized else "original"
        env = backend = collector = None
        try:
            restore_rng(initial_rng)
            projected = payload.get("config", {}).get("project_goals_to_region", False)
            env, _ = make_env(project_goals_to_region=bool(projected))
            env.reset(seed=0)
            encoder = CabinetStateEncoder(env)
            if encoder.contract() != payload["state_contract"]:
                raise ValueError("Checkpoint state contract does not match physical smoke encoder")
            backend = make_backend(device=args.device, seed=0, candidates=128,
                                   execution="cuda_graph" if args.device == "cuda" else "reference",
                                   runtime_optimizations=optimized, **checkpoint_backend_options(payload))
            backend.load_state_dict(payload["backend"], restore_planner=False)
            restore_rng(initial_rng)
            collector = Collector(env, backend, encoder, args.output / f"physical_{label}",
                                  runtime_optimizations=optimized, profile=True)
            rows, elapsed, _ = timed(args.device, lambda: collector.collect(
                args.physical_steps, "curiosity", seed=0, stage=f"profile_{label}"))
            if collector.total_steps != args.physical_steps:
                raise RuntimeError("Physical profile missed exact real-step budget")
            arms[label] = {"states": np.asarray([row["state"] for row in rows]),
                           "actions": np.asarray([row["action"] for row in rows]),
                           "next_states": np.asarray([row["next_state"] for row in rows]),
                           "requested_actions": np.asarray([row["requested_action"] for row in rows]),
                           "wall_s": elapsed, "timings": collector.timings}
        finally:
            if env is not None:
                env.close()
            del collector, backend
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    failures = parity({k: arms["original"][k] for k in ("states", "actions", "next_states", "requested_actions")},
                      {k: arms["optimized"][k] for k in ("states", "actions", "next_states", "requested_actions")})
    return {"passed": not failures, "mismatches": failures, "steps_per_mode": args.physical_steps,
            "model_frozen": True, "training_updates": 0,
            "arms": {key: {"wall_s": value["wall_s"], "timings": value["timings"]} for key, value in arms.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark", "compare", "profile"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda",
                        help="cpu explicitly limits benchmark to its two CPU reference cases")
    parser.add_argument("--physical-steps", type=int, default=0,
                        help="Optional profile physical steps per mode, capped at 64")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--states", type=int, default=12, help="Benchmark saved states; profile always uses 12")
    parser.add_argument("--warmup", type=int, default=2, help="Benchmark warmup calls; profile always uses 2")
    args = parser.parse_args(argv)
    if not 0 <= args.physical_steps <= 64:
        parser.error("physical-steps must be between 0 and 64")
    if args.stage != "profile" and args.physical_steps:
        parser.error("physical-steps applies only to profile")
    if args.steps < 1 or args.states < 1 or args.warmup < 1:
        parser.error("steps, states and warmup must be positive")
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory; refusing to overwrite {args.output}")
    payload = load_checkpoint(args.checkpoint)
    if args.stage == "compare" and args.device != "cuda":
        parser.error("compare requires CUDA cuda_graph execution")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; benchmark can use --device cpu explicitly")
    config = OmegaConf.load(ROOT / "configs/cee_us_cabinet_120k_total_heuristic.yaml")
    config = OmegaConf.merge(config, payload.get("config", {}), {"stage": args.stage, "seed": 0, "device": args.device,
                            "checkpoint": str(args.checkpoint.resolve()),
                            "training_updates": "measured_isolated_updates" if args.stage == "profile" else 0,
                            "profile_epochs": 1, "profile_repeats": 3, "physical_steps_per_mode": args.physical_steps,
                            "states": args.states, "warmup": args.warmup,
                            "steps_per_arm": args.steps, "arms": [
                                {"execution": "cuda_graph", "candidates": value} for value in (128, 64)]})
    if args.stage == "profile":
        config.states, config.warmup = 12, 2
        config.steps_per_arm = args.physical_steps
        config.arms = [{"runtime_optimizations": value, "candidates": 128,
                        "execution": "cuda_graph" if args.device == "cuda" else "reference"}
                       for value in (False, True)]
    torch.set_num_threads(1)
    # Both physical arms use the same fixed seed, independent of caller RNG state.
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    try:
        result = {"benchmark": benchmark, "compare": compare, "profile": profile}[args.stage](args, payload, config)
        if args.stage == "profile" and not result["passed"]:
            return 1
    except BaseException as error:
        if args.output.is_dir():
            write_json(args.output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                                      "status": "stopped_without_fallback"})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
