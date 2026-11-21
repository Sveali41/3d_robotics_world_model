"""Launch matched split7 exploration arms in three persistent Apptainer instances."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def arm_configs(base, *, include_heuristic_only=False):
    actor = base.get("raps_actor", {})
    if base.get("agent") != "raps_gnn_actor" or actor.get("skill_set") != "split7":
        raise ValueError("Comparison requires the split7 RAPS configuration")
    macro_mode = actor.get("pick_place_primitives", False)
    if macro_mode:
        if (actor.get("motion_execution") != "planner" or
                base.get("action_mode") != "joint_position"):
            raise ValueError("Pick/place comparison requires planner execution and joint_position actions")
    elif actor.get("motion_execution", "ee_delta") != "ee_delta":
        raise ValueError("Legacy comparison requires split7 direct execution")
    if base.get("object_heuristic") is None:
        raise ValueError("Base configuration must define the object heuristic")
    arms = {}
    names = (("curiosity_heuristic", "curiosity_only", "heuristic_only", "random")
             if include_heuristic_only else
             ("curiosity_heuristic", "curiosity_only", "random"))
    for name in names:
        config = deepcopy(base)
        config["collection_policy"] = "random" if name == "random" else "curiosity"
        if name in ("curiosity_only", "random"):
            config["object_heuristic"] = None
        if name == "heuristic_only":
            config.setdefault("raps_actor", {})["disagreement_weight"] = 0.0
        arms[name] = config
    return arms


def snapshot_sources(root, target):
    target.mkdir()
    # Preserve current source and its Git evidence, without creating any commits.
    shutil.copytree(root / ".git", target / ".git", ignore=shutil.ignore_patterns("*.lock"))
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root).split(b"\0")
    for raw in set(tracked + untracked):
        if not raw:
            continue
        relative = Path(raw.decode())
        source, destination = root / relative, target / relative
        if source.is_dir():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    hashes = {}
    for path in sorted(target.rglob("*")):
        relative = path.relative_to(target)
        if path.is_file() and ".git" not in relative.parts:
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (target.parent / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=Path("/data/data/siyao/3d_world_model.sif"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/raps_gnn_actor_cabinet_split7.yaml")
    parser.add_argument("--numpy-shim", type=Path, default=Path("/tmp/wm_gated_numpy126"))
    parser.add_argument("--include-heuristic-only", action="store_true",
                        help="Include a curiosity-policy arm with the object heuristic and zero disagreement weight")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must be a new directory; refusing overwrite or duplicate launches")
    if not args.image.is_file() or not (args.numpy_shim / "numpy/__init__.py").is_file():
        parser.error("Existing SIF image and verified executable NumPy1.26 shim are required")
    command = ["apptainer", "exec", str(args.image), "env",
               f"PYTHONPATH={args.numpy_shim.resolve()}:{ROOT}", "python", "-c",
               "import json,sys; from omegaconf import OmegaConf; "
               "from exploration.raps.raps_gnn_actor import skill_schema; "
               "config=OmegaConf.to_container(OmegaConf.load(sys.argv[1]),resolve=True); "
               "print(json.dumps(dict(config=config, action_dim=skill_schema(config['raps_actor']).action_dim)))",
               str(args.config.resolve())]
    resolved = json.loads(subprocess.check_output(command, text=True))
    base = resolved["config"]
    arms = arm_configs(base, include_heuristic_only=args.include_heuristic_only)
    output.mkdir(parents=True)
    source = output / "source_snapshot"
    snapshot_sources(ROOT, source)
    runtime = Path(tempfile.mkdtemp(prefix="raps-three-arm-", dir="/tmp"))
    numpy_shim = runtime / "numpy126"
    shutil.copytree(args.numpy_shim, numpy_shim, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    started_at = datetime.now(ZoneInfo("Europe/London"))
    tag = started_at.strftime("%Y%m%d_%H%M%S")
    budget = dict(bootstrap_steps=base["bootstrap_steps"],
                  online_training_steps=base["rounds"] * base["steps_per_round"],
                  evaluation_steps=base["evaluation_steps"])
    budget["total_control_steps_per_arm"] = sum(budget.values())
    manifest = dict(started_at=started_at.isoformat(), image=str(args.image.resolve()),
                    source_snapshot=str(source), runtime_directory=str(runtime),
                    seed=base["seed"], budget=budget, arms={},
                    comparison=("Collection policy and object heuristic vary across arms; the optional heuristic-only arm "
                                "also sets RAPS curiosity disagreement weight to zero; matched collection rules"),
                    cpu_sampling=base.get("cpu_sampling"),
                    action_interface=dict(
                        pick_place_primitives=base["raps_actor"].get("pick_place_primitives", False),
                        grasp_only=base["raps_actor"].get("grasp_only", False),
                        pick_place_grid=base["raps_actor"].get("pick_place_grid", False),
                        grasp_orientation_candidates=base["raps_actor"].get("grasp_orientation_candidates", False),
                        policy_action_dimensions=resolved["action_dim"],
                        grasp_retention=(base["raps_actor"].get("grasp_retention", "friction")
                                         if base["raps_actor"].get("pick_place_primitives", False) else None)),
                    training_budget=("Fixed WM updates and actor updates; bounded per-record dynamic replay"
                                     if base.get("wm_replay", {}).get("mode") == "bounded_dynamic" else
                                     "Fixed WM updates and actor updates; protected new/history replay"
                                     if base.get("wm_replay") is not None else
                                     "Same epochs and actor updates; WM minibatch counts depend on completed skills"),
                    wm_replay=base.get("wm_replay"),
                    wm_training=base.get("wm_training"),
                    limitations="Single seed; shared GPU contention makes wall-time comparisons uncontrolled")
    manifest_path = output / "comparison.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, config in arms.items():
        directory = output / name
        directory.mkdir()
        config_path = directory / "config.yaml"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        instance = "raps_" + name + "_" + tag
        private = runtime / name
        private.mkdir()
        (private / "mpl").mkdir()
        env_path = directory / "environment.txt"
        environment = dict(PYTHONPATH=f"{numpy_shim}:{source}", PYTHONUNBUFFERED="1",
                           PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=str(base["seed"]),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           NUMEXPR_NUM_THREADS="1", GIT_OPTIONAL_LOCKS="0",
                           MPLCONFIGDIR=str(private / "mpl"), TMPDIR=str(private))
        env_path.write_text("".join(key + "=" + value + "\n" for key, value in environment.items()))
        start = ["apptainer", "instance", "start", "--nv", "--bind", f"{source}:{source}:ro",
                 "--bind", f"{numpy_shim}:{numpy_shim}:ro", "--env-file", str(env_path),
                 str(args.image.resolve()), instance]
        with (directory / "instance.log").open("w") as handle:
            subprocess.run(start, check=True, stdout=handle, stderr=subprocess.STDOUT)
        train = ["apptainer", "exec", "--pwd", str(source), "--env-file", str(env_path),
                 "instance://" + instance, "python", "-u", "-m", "exploration.cee_us.cee_us_cabinet",
                 "train", "--config", str(config_path), "--policy", config["collection_policy"],
                 "--output", str(directory / "run")]
        wrapper = directory / "run.sh"
        wrapper.write_text("#!/bin/bash\n" + shlex.join(train) + "\n" +
                           "run_status=$?\n" +
                           'printf "%s\\n" "$run_status" > ' + shlex.quote(str(directory / "exit_code")) + "\n" +
                           shlex.join(["apptainer", "instance", "stop", instance]) + "\n" +
                           'exit "$run_status"\n')
        with (directory / "stdout.log").open("w") as handle:
            process = subprocess.Popen(["/bin/bash", str(wrapper)], stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        manifest["arms"][name] = dict(instance=instance, pid=process.pid, command=train,
                                     config=str(config_path), output=str(directory / "run"))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(dict(arm=name, instance=instance, pid=process.pid)), flush=True)
    print(str(manifest_path), flush=True)


if __name__ == "__main__":
    main()
