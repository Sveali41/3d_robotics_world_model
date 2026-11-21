"""Plan or collect a deterministic uniform sweep of current cup grasp actions."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from domain.robocasa.primitives import PrimitiveAction
from exploration.diagnostics.profile_cup_grasp import CONFIG, ROOT, SOURCES, validate_config

PROTOCOL = "current_cup_uniform_grasp_reset_v1"
DEFAULT_SEED = 2_000_000
MAX_CONTROLS_PER_CASE = 400


def uniform_cases(settings, z_levels=5, seed=DEFAULT_SEED, *, limit=None):
    """Build canonical cell-major, pose-major, Z-major cases from the production schema."""
    if isinstance(z_levels, bool) or int(z_levels) != z_levels or z_levels < 2:
        raise ValueError("z_levels must be an integer >= 2")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if limit is not None and (isinstance(limit, bool) or int(limit) != limit or limit < 1):
        raise ValueError("limit must be a positive integer")
    from exploration.raps.raps_gnn_actor import skill_schema

    schema = skill_schema(settings)
    xy = np.asarray(schema.grid_xy_root_m, dtype=float)
    orientations = tuple(schema.grasp_orientation_ids)
    z_spec = schema.grasp_z
    z_values = np.linspace(z_spec.center - z_spec.scale, z_spec.center + z_spec.scale, int(z_levels))
    cases = []
    for cell_id, (x, y) in enumerate(xy):
        for pose_id in orientations:
            for z_m in z_values:
                case_id = len(cases)
                action = PrimitiveAction("grasp", dict(x=float(x), y=float(y), z=float(z_m),
                                                          orientation_id=int(pose_id)))
                encoded = schema.encode(action)
                decoded = schema.decode(encoded)
                if decoded.primitive_type != "grasp" or not np.allclose(
                        [decoded.parameters[k] for k in ("x", "y", "z")], [x, y, z_m], atol=1e-7):
                    raise ValueError(f"Production action schema failed round-trip for case {case_id}")
                case_seed = int(np.random.SeedSequence([int(seed), case_id]).generate_state(1)[0] % (2**31))
                cases.append(dict(case_id=case_id, cell_id=cell_id, orientation_id=int(pose_id),
                                  x_m=float(x), y_m=float(y), z_m=float(z_m), seed=case_seed,
                                  action=encoded.tolist()))
    return cases if limit is None else cases[:int(limit)]


def empty_replay(state_dim):
    return dict(states=np.empty((0, state_dim), np.float32), actions=np.empty((0, 31), np.float32),
                next_states=np.empty((0, state_dim), np.float32), terminated=np.empty(0, bool),
                truncated=np.empty(0, bool), episode_ids=np.empty(0, np.int64),
                duration_low_steps=np.empty(0, np.int64), case_ids=np.empty(0, np.int64),
                cell_ids=np.empty(0, np.int64), orientation_ids=np.empty(0, np.int64),
                z_m=np.empty(0, np.float32))


def replay_from_outcomes(outcomes, *, state_contract=None):
    """Export completed nonzero endpoints; honor supplied episodes or isolate fresh cases."""
    eligible = [r for r in outcomes if r.get("wm_eligible") and r.get("skill")]
    contracts = [state_contract] if state_contract is not None else []
    contracts.extend(r["state_contract"] for r in outcomes if r.get("state_contract") is not None)
    dimensions = {int(contract["dimension"]) for contract in contracts}
    dimensions.update(len(r["skill"][field]) for r in eligible
                      for field in ("state", "next_state"))
    if len(dimensions) != 1:
        raise ValueError(f"Replay state dimension must be unambiguous; found {sorted(dimensions)}")
    state_dim = dimensions.pop()
    replay = empty_replay(state_dim)
    if not eligible:
        return replay
    replay.update(
        states=np.asarray([r["skill"]["state"] for r in eligible], np.float32),
        actions=np.asarray([r["skill"]["action"] for r in eligible], np.float32),
        next_states=np.asarray([r["skill"]["next_state"] for r in eligible], np.float32),
        terminated=np.asarray([r["skill"]["terminated"] for r in eligible], bool),
        truncated=np.asarray([r["skill"]["truncated"] for r in eligible], bool),
        episode_ids=np.asarray([r.get("episode_id", r["case_id"] + 1) for r in eligible], np.int64),
        duration_low_steps=np.asarray([r["skill"]["duration_low_steps"] for r in eligible], np.int64),
        case_ids=np.asarray([r["case_id"] for r in eligible], np.int64),
        cell_ids=np.asarray([r["cell_id"] for r in eligible], np.int64),
        orientation_ids=np.asarray([r["orientation_id"] for r in eligible], np.int64),
        z_m=np.asarray([r["z_m"] for r in eligible], np.float32))
    return replay


def run_case(job):
    """Run one fresh CPU scene and retain the collector's raw tick/skill logs."""
    import torch
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from exploration.raps.raps_gnn_actor import DEFAULTS, SkillCollector, action_contract

    torch.set_num_threads(1)
    settings = {**DEFAULTS, **job["settings"]}
    env, _ = make_env(episode_steps=job["episode_steps"],
                      centered_initial_tcp=job["centered_initial_tcp"], cabinet_door=job["cabinet_door"],
                      cabinet_item_layout=job["cabinet_item_layout"], cabinet_grid=job["cabinet_grid"],
                      cabinet_grid_layout=settings["grid_layout"], action_mode="joint_position",
                      low_level_episode_steps=job["episode_steps"])
    directory = Path(job["directory"])
    controls = 0
    try:
        encoder = CabinetSkillStateEncoder(env, latched_gripper=settings["latched_gripper"])
        case_seed = int(job["case"]["seed"])
        random.seed(case_seed)
        np.random.seed(case_seed)
        torch.manual_seed(case_seed)
        collector = SkillCollector(env, encoder, SimpleNamespace(settings=settings), directory,
                                   collection_optimizations=bool(job["collection_optimizations"]))
        fixed_action = np.asarray(job["case"]["action"], dtype=np.float32)
        collector._propose_action = lambda policy: fixed_action.copy()
        finished = []
        ticks, skills = collector.collect(MAX_CONTROLS_PER_CASE, "random", seed=job["case"]["seed"],
            stage="uniform_grasp_case", on_skill_end=finished.append, stop_when=lambda: bool(finished))
        controls = len(ticks)
        if len(skills) != 1 or len(ticks) > MAX_CONTROLS_PER_CASE:
            raise RuntimeError("collector violated the one-macro/400-control case bound")
        skill = skills[0]
        return dict(case_id=job["case"]["case_id"], cell_id=job["case"]["cell_id"],
                    orientation_id=job["case"]["orientation_id"], z_m=job["case"]["z_m"],
                    seed=job["case"]["seed"], controls=len(ticks), completed=bool(skill["completed"]),
                    status=skill["status"], duration_low_steps=int(skill["duration_low_steps"]),
                    wm_eligible=bool(skill["completed"] and skill["duration_low_steps"] > 0),
                    state_contract=encoder.contract(), action_contract=action_contract(settings),
                    skill=skill)
    except Exception as exc:
        if "collector" in locals():
            controls = collector.total_steps
        case = job["case"]
        return dict(case_id=case["case_id"], cell_id=case["cell_id"],
                    orientation_id=case["orientation_id"], z_m=case["z_m"], seed=case["seed"],
                    controls=controls, completed=False, status="error", wm_eligible=False,
                    error=repr(exc))
    finally:
        env.close()


def _case_worker(job):
    try:
        outcome = run_case(job)
    except Exception as exc:
        case = job["case"]
        outcome = dict(case_id=case["case_id"], cell_id=case["cell_id"],
                       orientation_id=case["orientation_id"], z_m=case["z_m"],
                       seed=case["seed"], controls=0, completed=False, status="error",
                       wm_eligible=False, error=repr(exc))
    if outcome["status"] == "error":
        path = Path(job["directory"]) / "failure.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(outcome, indent=2, allow_nan=False) + "\n")
    if job.get("success_only", False):
        outcome["macro_endpoint_eligible"] = outcome["wm_eligible"]
        outcome["wm_eligible"] = bool(outcome["wm_eligible"] and outcome["status"] == "success")
    return outcome


def _write_jsonl(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "collect"), nargs="?", default="plan")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--z-levels", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, help="schedule only the first N canonical cases")
    parser.add_argument("--case-ids", type=int, nargs="+", help="schedule selected canonical case IDs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--success-only", action="store_true",
                        help="export successful complete macros only; retain every raw attempt")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 2:
        parser.error("--workers must be between 1 and 2")
    if args.limit is not None and args.case_ids is not None:
        parser.error("--limit and --case-ids are mutually exclusive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    config = OmegaConf.load(args.config)
    validate_config(config, learning=False)
    from exploration.raps.raps_gnn_actor import DEFAULTS, action_contract, skill_schema
    settings = {**DEFAULTS, **OmegaConf.to_container(config.raps_actor, resolve=True)}
    all_cases = uniform_cases(settings, args.z_levels, args.seed)
    if args.case_ids is not None:
        wanted = set(args.case_ids)
        known = {case["case_id"] for case in all_cases}
        if len(wanted) != len(args.case_ids) or not wanted <= known:
            parser.error("--case-ids must be unique IDs in the canonical sweep")
        cases = [case for case in all_cases if case["case_id"] in wanted]
    else:
        cases = all_cases if args.limit is None else all_cases[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(args.output / "cases.jsonl", cases)
    schema = skill_schema(settings)
    xy = np.asarray(schema.grid_xy_root_m)
    manifest = dict(protocol=PROTOCOL, mode=args.mode, status="planned",
        dataset_role="heldout_success_conditioned_reset" if args.success_only else "heldout_uniform_reset",
        success_only=args.success_only,
        replay_selection="completed positive-duration success only" if args.success_only else "all completed positive-duration macros",
        config=OmegaConf.to_container(config, resolve=True), action_contract=action_contract(settings),
        fixed_cup_cell_id=4,
        grid_xy_root_m=xy.tolist(), orientation_ids=[0, 1, 2], z_values_m=sorted({c["z_m"] for c in cases}),
        cases_planned=len(cases), full_sweep=len(cases) == len(all_cases),
        subset_limit=args.limit, selected_case_ids=args.case_ids, workers=args.workers,
        max_controls_per_case=MAX_CONTROLS_PER_CASE,
        max_possible_controls=len(cases) * MAX_CONTROLS_PER_CASE, seed=args.seed,
        action_schema_details=dict(z_center_m=float(schema.grasp_z.center),
            z_scale_m=float(schema.grasp_z.scale), schema_normalization="production encode/decode"),
        policy="fixed encoded uniform grasp proposal; no learned policy, fitting, checkpoint, rejection, or resampling",
        preprocessing="Each case starts in a fresh reset CPU scene; independent deterministic case seed. This supports one-macro endpoint prediction, not chained multi-horizon inference.",
        files_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                      for p in (*SOURCES, "exploration/diagnostics/collect_uniform_grasps.py")
                      if (ROOT / p).is_file()})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if args.mode == "plan":
        print(json.dumps(dict(status="planned", cases=len(cases), maximum_controls=manifest["max_possible_controls"])))
        return 0

    # Build one job per case; spawn workers only when explicitly requested.
    jobs = [dict(case=case, settings=settings, success_only=args.success_only,
                 collection_optimizations=bool(config.collection_optimizations),
                 episode_steps=int(config.episode_steps), centered_initial_tcp=bool(config.centered_initial_tcp),
                 cabinet_door=bool(config.cabinet_door), cabinet_item_layout=config.cabinet_item_layout,
                 cabinet_grid=bool(settings["pick_place_grid"]),
                 directory=str(args.output / "cases" / f"case_{case['case_id']:04d}")) for case in cases]
    if args.workers == 1:
        outcomes = []
        for job in jobs:
            outcome = _case_worker(job)
            outcomes.append(outcome)
            _write_jsonl(args.output / "attempts.jsonl", outcomes)
            if outcome["status"] == "error":
                break
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
            outcomes = list(pool.map(_case_worker, jobs))
        _write_jsonl(args.output / "attempts.jsonl", outcomes)
    from exploration.cee_us.cee_us_state import cabinet_agent_dim
    layout = config.get("cabinet_item_layout", "two_upper_second_row")
    state_dim = cabinet_agent_dim(config.get("cabinet_door", False),
                                  latched_gripper=settings["latched_gripper"]) + 17 * (
                                      1 if layout == "single_upper_second_row" else 2)
    replay = replay_from_outcomes(outcomes, state_contract={"dimension": state_dim})
    np.savez_compressed(args.output / "wm_replay.npz", **replay)
    statuses = {}
    for outcome in outcomes:
        statuses[outcome["status"]] = statuses.get(outcome["status"], 0) + 1
    state_contracts = [o["state_contract"] for o in outcomes if "state_contract" in o]
    action_contracts = [o["action_contract"] for o in outcomes if "action_contract" in o]
    manifest.update(status="failed" if any(o["status"] == "error" for o in outcomes) else "collected",
        cases_attempted=len(outcomes), cases_eligible=len(replay["states"]), controls=sum(o["controls"] for o in outcomes),
        state_contract=state_contracts[0] if state_contracts else None,
        action_contract=action_contracts[0] if action_contracts else manifest["action_contract"],
        outcome_status_counts=statuses, coverage_by_cell_pose_z=[dict(case_id=o["case_id"], cell_id=o["cell_id"],
            orientation_id=o["orientation_id"], z_m=o["z_m"], status=o["status"],
            wm_eligible=o.get("wm_eligible", False)) for o in outcomes])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if manifest["status"] == "failed":
        failed = next(o for o in outcomes if o["status"] == "error")
        raise RuntimeError(f"Uniform grasp case {failed['case_id']} failed; inspect its failure.json and raw case logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
