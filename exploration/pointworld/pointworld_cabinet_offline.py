"""Paired, episode-disjoint cabinet dynamics data; no online-policy replacement.

PointWorld retains its upstream future-robot-flow conditioning. This is an
offline prediction comparison, not evidence that it can replace the GNN planner.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PW = ROOT / "exploration/pointworld/official"
BASELINE = ROOT / "configs/raps_gnn_actor_cabinet_400k_episode40s_centered.yaml"
DOMAIN = "robocasa_raps"  # Existing direct-flow decoder, not the physical domain.
SCHEMA = "cabinet-paired-pointworld-v1"


def write_json(path, value):
    from exploration.diagnostics.diagnose_cabinet import jsonable
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_split(episode_id, test_every):
    if episode_id < 1 or test_every < 2:
        raise ValueError("episode IDs start at 1; test_every must be >=2")
    return "test" if episode_id % test_every == 0 else "train"


def samples(directory, split):
    """Read our fixed paired payloads without resampling or skipping errors."""
    for shard in sorted(Path(directory).joinpath(split).glob("*.tar")):
        with tarfile.open(shard) as archive:
            current, sample = None, {}
            for member in archive:
                if not member.isfile():
                    continue
                key, field = member.name.split(".", 1)
                if current is not None and key != current:
                    yield current, sample
                    sample = {}
                current = key
                sample[field] = archive.extractfile(member).read()
            if sample:
                yield current, sample


def array(payload, key):
    return np.load(io.BytesIO(payload[key + ".npy"]), allow_pickle=False)


def gnn_arrays(directory, split):
    rows = list(samples(directory, split))
    if not rows:
        raise ValueError(f"No complete paired skills in {split}; collect more whole episodes")
    result = {name: np.stack([array(p, field) for _, p in rows]) for name, field in (
        ("states", "gnn_state"), ("actions", "skill_action"),
        ("next_states", "gnn_next_state"))}
    return [key for key, _ in rows], result


class PairedRecorder:
    """Attach RGB-D recording to the maintained SkillCollector execution loop."""
    def __init__(self, env, builder, output, test_every):
        self.env, self.builder = env, builder
        self.output, self.test_every = Path(output), test_every
        self.record = None
        self.counts = {"train": 0, "test": 0, "incomplete": 0}

    def observe(self):
        import torch
        from torch.utils._pytree import tree_map
        raw = self.env.unwrapped
        # Re-rendering observes the same post-control state; never steps physics.
        obs = raw.get_obs()
        obs["sensor_data"] = {"base_camera": obs["sensor_data"]["base_camera"]}
        obs["sensor_param"] = {"base_camera": obs["sensor_param"]["base_camera"]}
        obs = tree_map(lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x, obs)
        self.record.append_observation(obs, self.env.env._context())
        door = raw.door_link
        self.record.object_poses[-1]["cabinet_door"] = {
            "scene_id": int(door.per_scene_id[0]),
            "pose": door.pose.to_transformation_matrix()[0].detach().cpu().clone(),
        }

    def start(self, episode_id, skill_id, decoded, action, initial):
        from exploration.trajectory_recorder import PrimitiveRecord
        self.initial, self.action = initial.copy(), action.copy()
        self.record = PrimitiveRecord(
            episode_id, skill_id, decoded.primitive_type, decoded.to_vector(),
            parameters_physical=decoded.parameters,
            metadata={"paired_schema": SCHEMA, "physical_dt_seconds": .05,
                      "phase_resampling": "11 frames across one variable-duration skill"})
        self.observe()

    def step(self, row):
        self.record.append_transition(
            row["action"], full_controller_action=row["full_controller_action"],
            contact_event={"contact_events": row["contact_events"],
                           "outcome_metrics": row["outcome_metrics"]})
        self.observe()

    def end(self, row):
        from common.utils import numpy_bytes, json_bytes
        record = self.record
        record.status = record.termination_reason = row["status"]
        record.terminated, record.truncated = row["terminated"], row["truncated"]
        record.outcome_metrics = row["after"]
        split = episode_split(record.episode_id, self.test_every)
        key = f"ep{record.episode_id:06d}_skill{record.primitive_id:08d}"
        # Incomplete skills remain archived but are excluded from BOTH learners.
        destination = split if row["trainable"] else "incomplete"
        parent = self.output / destination
        parent.mkdir(exist_ok=True)
        path = parent / f"{destination}-{key}.tar"
        payload = self.builder(record, horizon=11) if row["duration_low_steps"] else {}
        payload.update({"gnn_state.npy": numpy_bytes(self.initial),
                        "gnn_next_state.npy": numpy_bytes(row["next_state"]),
                        "skill_action.npy": numpy_bytes(self.action),
                        "paired_skill.json": json_bytes(row)})
        if record.duration_low_steps:
            phase = record.phase_view(11)
            for field in ("rgb", "position", "segmentation"):
                payload[f"paired_camera_{field}.npy"] = numpy_bytes(np.stack([
                    obs["sensor_data"]["base_camera"][field][0].numpy()
                    for obs in phase["observations"]]))
        with tarfile.open(path, "w") as archive:
            for field, blob in sorted(payload.items()):
                item = tarfile.TarInfo(f"{key}.{field}")
                item.size = len(blob)
                archive.addfile(item, io.BytesIO(blob))
        self.counts[destination] += 1
        with (self.output / "manifest.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(key=key, episode_id=record.episode_id,
                split=split, destination=destination, trainable=row["trainable"],
                steps=record.duration_low_steps, path=str(path.relative_to(self.output)),
                sha256=sha256(path)), allow_nan=False) + "\n")
        self.record = None


def collect(args):
    import torch
    from omegaconf import OmegaConf
    from exploration.cee_us.cee_us_cabinet import make_env, prepare_output
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from exploration.raps.raps_gnn_actor import DEFAULTS, RAPSImaginationActor, SkillCollector
    from exploration.cee_us.cee_us_backend import CEEUSBackend
    from common.utils import RoboCasaPointWorldSampleBuilder
    from exploration.pointworld.official.integration.pointworld_wrapper import PointWorldWrapper
    if args.steps < 1 or args.test_every < 2:
        raise ValueError("steps must be positive; test-every must be >=2")
    config = OmegaConf.load(BASELINE)
    payload = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != "cabinet-raps-gnn-actor-v1":
            raise ValueError("Collection requires a compact6 RAPS actor checkpoint")
        config = OmegaConf.create(payload["config"])
    settings = {**DEFAULTS, **OmegaConf.to_container(config.raps_actor, resolve=True)}
    if settings["skill_set"] != "compact6" or not settings["latched_gripper"]:
        raise ValueError("This paired protocol requires compact6 with the 67D latched-gripper state")
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    env, env_config = make_env(episode_steps=int(config.episode_steps),
        centered_initial_tcp=bool(config.centered_initial_tcp),
        obs_mode="rgb+position+segmentation",
        sensor_configs={"base_camera": {"width": 320, "height": 180}})
    try:
        encoder = CabinetSkillStateEncoder(env, latched_gripper=settings["latched_gripper"])
        if payload is None:
            actor = SimpleNamespace(settings=settings)
        else:
            # Only the frozen actor chooses data; neither old dynamics nor replay train a learner.
            backend = CEEUSBackend(agent_dim=33, action_dim=12, device=args.device)
            actor = RAPSImaginationActor(backend, settings)
            if payload["state_contract"] != encoder.contract():
                raise ValueError("Source state contract differs from paired collection")
            from exploration.raps.raps_gnn_actor import action_contract
            if payload["action_contract"] != action_contract(settings):
                raise ValueError("Source action/execution contract differs")
            actor.actor.load_state_dict(payload["actor"]["actor"])
            normalizers = backend._move_tensors(payload["backend"]["normalizers"], backend.device)
            for name in ("input_normalizer_agent", "input_normalizer_obj_dyn", "input_normalizer_obj_stat"):
                getattr(backend.model, name).load_state_dict(normalizers[name])
        wrapper = PointWorldWrapper(None, data_domain=DOMAIN, device=args.device,
            robot_type="droid_panda", camera_names=("base_camera",),
            robot_urdf_path=str(env.unwrapped.agent.urdf_path),
            robot_joint_names=[j.name for j in env.unwrapped.agent.robot.get_active_joints()],
            filter_scene_workspace=False)
        prepare_output(args.output, config, env_config)
        manifest_path = args.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(agent="paired_offline_data_collection", reward_used_for_planning=False,
                        prediction_horizon_seconds=None, privileged_state=True,
                        phase_frames=11, dynamics_training_started=False)
        write_json(manifest_path, manifest)
        write_json(args.output / "protocol.json", dict(schema=SCHEMA, seed=args.seed,
            real_step_budget=args.steps, test_every=args.test_every,
            policy="frozen_actor" if payload else "uniform_compact6",
            source_checkpoint=None if payload is None else str(args.checkpoint.resolve()),
            source_sha256=None if payload is None else sha256(args.checkpoint),
            collection_actor_uses_source_input_normalizers=payload is not None,
            dynamics_initialization="both_from_scratch", dino="pretrained_frozen",
            domain_decoder=DOMAIN, physical_domain="ManiSkill Panda wall cabinet",
            pointcloud_workspace_filter=False, control_region_constraint=False,
            split_unit="whole_episode", phase_frames=11,
            conditioning_difference="PointWorld: actual future robot flows; GNN: skill action",
            pointworld_revision=subprocess.check_output(["git", "-C", str(PW), "rev-parse", "HEAD"], text=True).strip()))
        (args.output / "pointworld.diff").write_text(subprocess.check_output(
            ["git", "-C", str(PW), "diff"], text=True))
        builder = RoboCasaPointWorldSampleBuilder(wrapper, env)
        recorder = PairedRecorder(env, builder, args.output, args.test_every)
        collector = SkillCollector(env, encoder, actor, args.output)
        try:
            ticks, skills = collector.collect(args.steps, "curiosity" if payload else "random",
                seed=args.seed, stage="paired_offline", on_step=recorder.step,
                on_skill_start=recorder.start, on_skill_end=recorder.end)
        except BaseException as error:
            # Preserve the partly collected raw trajectory for diagnosis; never train on it.
            if recorder.record is not None:
                torch.save(recorder.record, args.output / "interrupted_record.pt")
            write_json(args.output / "failure.json", {"error": repr(error),
                       "executed_steps": collector.total_steps})
            raise
        write_json(args.output / "collection.json", dict(real_steps=len(ticks),
                   skills=len(skills), counts=recorder.counts, state_contract=encoder.contract()))
        print(json.dumps(audit(args.output)))
    finally:
        env.close()


def audit(directory):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    sets = {split: {r["episode_id"] for r in rows if r["trainable"] and r["split"] == split}
            for split in ("train", "test")}
    if sets["train"] & sets["test"]:
        raise ValueError("Episode leakage in paired data")
    seen = set()
    by_key = {r["key"]: r for r in rows if r["trainable"]}
    if len(by_key) != sum(bool(r["trainable"]) for r in rows):
        raise ValueError("Duplicate trainable keys in manifest")
    stats = {}
    for split in ("train", "test"):
        count = moved = 0
        per_object = [0, 0]
        for key, p in samples(directory, split):
            if key in seen:
                raise ValueError(f"Duplicate paired sample {key}")
            seen.add(key)
            if key not in by_key or by_key[key]["split"] != split or int(array(p, "episode_id")) != by_key[key]["episode_id"]:
                raise ValueError(f"Paired episode/split manifest differs: {key}")
            s, n = array(p, "gnn_state"), array(p, "gnn_next_state")
            if s.shape != (67,) or n.shape != (67,) or array(p, "skill_action").shape != (12,):
                raise ValueError(f"Unexpected GNN contract: {key}")
            if not np.array_equal(s[57:], n[57:]):
                raise ValueError(f"Object static attributes changed across skill: {key}")
            if array(p, "camera_base_camera_initial_rgb").shape != (180, 320, 3):
                raise ValueError(f"PointWorld image contract differs: {key}")
            for field in ("gnn_state", "gnn_next_state", "skill_action", "scene_flows", "robot_flows"):
                if not np.isfinite(array(p, field)).all():
                    raise FloatingPointError(f"Nonfinite {field}: {key}")
            active = [np.linalg.norm(n[i:i+3] - s[i:i+3]) >= .015 for i in (33, 45)]
            per_object = [a + int(b) for a, b in zip(per_object, active)]
            moved += int(any(active)); count += 1
        stats[split] = dict(samples=count, episodes=len(sets[split]),
                            moving_skills=moved, per_object_moving_skills=per_object)
    expected = {r["key"] for r in rows if r["trainable"]}
    if seen != expected:
        raise ValueError("Paired manifest and archives differ")
    for row in rows:
        if sha256(directory / row["path"]) != row["sha256"]:
            raise ValueError(f"Archive hash differs: {row['path']}")
    result = dict(schema=SCHEMA, splits=stats, episode_disjoint=True,
        ready_for_training=all(stats[s]["samples"] >= 2 for s in stats),
        interaction_dynamics_verified=False,
        note="Motion counts are coverage evidence, not proof of grasp/lift prediction quality")
    write_json(directory / "audit.json", result)
    return result


def train_gnn(args):
    import torch
    from exploration.cee_us.cee_us_backend import CEEUSBackend
    result = audit(args.data)
    if not result["ready_for_training"]:
        raise ValueError("Need at least two paired skills in each episode-disjoint split")
    if args.output.exists():
        raise FileExistsError(args.output)
    keys, replay = gnn_arrays(args.data, "train")
    torch.set_num_threads(1)
    backend = CEEUSBackend(agent_dim=33, action_dim=12, seed=args.seed, device=args.device)
    args.output.mkdir(parents=True)
    metrics = backend.train(replay, args.epochs)
    torch.save(dict(schema=SCHEMA, backend=backend.state_dict(), train_keys=keys,
                    seed=args.seed, epochs=args.epochs, data=str(args.data.resolve()),
                    manifest_sha256=sha256(args.data / "manifest.jsonl")), args.output / "gnn.pt")
    write_json(args.output / "training.json", metrics)


def rigid_object_points(state, target, points, object_index):
    """Express the GNN's endpoint pose prediction on the recorded scene points."""
    from scipy.spatial.transform import Rotation
    offset = 33 + 12 * object_index
    initial_rotation = Rotation.from_euler("xyz", state[offset+3:offset+6]).as_matrix()
    predicted_rotation = Rotation.from_euler("xyz", target[offset+3:offset+6]).as_matrix()
    return ((points - state[offset:offset+3]) @ initial_rotation) @ predicted_rotation.T + target[offset:offset+3]


def pointworld_prediction(model, raw, args, key, *, return_diagnostics=False):
    """Use the official decoder, camera sampler, transforms, and model forward."""
    import torch
    from scipy.spatial import cKDTree
    from dataset_components.decoders import decode_data
    from dataset_components.cameras import sample_cameras
    from dataset_components.pipeline import apply_release_pipeline_to_sample
    from dataset_components.collate import custom_collate_fn
    decoded = decode_data({**raw, "__key__": key}, DOMAIN)
    decoded = sample_cameras(decoded, min_num_cameras=1, max_num_cameras=1,
                             deterministic=True, seed=0)
    batch_sample = apply_release_pipeline_to_sample(decoded, DOMAIN, "test", args)
    batch = custom_collate_fn([batch_sample], args)
    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(batch, training=False)
        predicted = outputs["scene_flows"][0].cpu().numpy()
    shift = batch_sample["__shift_amount__"].cpu().numpy()
    ground_truth = batch_sample["gt_scene_flows"].cpu().numpy() - shift
    predicted = predicted - shift
    original = array(raw, "scene_flows")
    distances, indices = cKDTree(original[0]).query(ground_truth[0])
    # Grid sampling selects recorded points; never silently compare different identities.
    if np.any(distances > 1e-5) or not np.allclose(
            original[:, indices], ground_truth, rtol=1e-5, atol=1e-5):
        raise ValueError(f"Official transforms changed point correspondence for {key}")
    if not np.isfinite(predicted).all():
        raise FloatingPointError(f"Nonfinite PointWorld output: {key}")
    if return_diagnostics:
        diagnostics = None
        if "disagreement" in outputs:
            diagnostics = dict(std=outputs["disagreement"][0].cpu().numpy(),
                score=float(outputs["disagreement_score"][0]),
                per_step=outputs["disagreement_per_step"][0].cpu().tolist(),
                points=int(outputs["scored_points"][0]))
        return predicted, indices, diagnostics
    return predicted, indices


def evaluate(args):
    import torch
    from exploration.cee_us.cee_us_backend import CEEUSBackend
    audit(args.data)
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.gnn_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA or checkpoint["manifest_sha256"] != sha256(args.data / "manifest.jsonl"):
        raise ValueError("GNN checkpoint must belong to this paired dataset")
    backend = CEEUSBackend(agent_dim=33, action_dim=12, seed=0, device=args.device)
    backend.load_state_dict(checkpoint["backend"])
    model = pw_args = None
    ensemble_path = getattr(args, "pointworld_ensemble", None)
    if ensemble_path:
        from exploration.pointworld.pointworld_ensemble import PointWorldEnsemble
        model = PointWorldEnsemble.from_manifest(ensemble_path, device=args.device)
        if model.manifest["data_manifest_sha256"] != sha256(args.data / "manifest.jsonl"):
            raise ValueError("PointWorld ensemble must belong to this paired dataset")
        pw_args = model.args
        if pw_args.domains != [DOMAIN] or getattr(pw_args, "raps_action_dim", 0):
            raise ValueError("Requires the native robot-flow-conditioned paired cabinet ensemble")
    elif args.pointworld_checkpoint:
        sys.path.insert(0, str(PW))
        from pointworld.base import BaseModel
        pw_checkpoint = torch.load(args.pointworld_checkpoint, map_location="cpu", weights_only=False)
        pw_args = pw_checkpoint["args"]
        if pw_args.domains != [DOMAIN] or Path(pw_args.data_dirs[0]).resolve() != args.data.resolve():
            raise ValueError("PointWorld checkpoint must be trained on this paired dataset/domain")
        if getattr(pw_args, "raps_action_dim", 0):
            raise ValueError("This baseline evaluates native robot-flow-conditioned PointWorld (raps_action_dim=0)")
        pw_args.device, pw_args.distributed, pw_args.disable_compile = args.device, False, True
        model = BaseModel(pw_args, {"scene_features_dim": 31, "robot_features_dim": 16})
        model.load_state_dict(pw_checkpoint["model"])
        model.to(args.device).eval()
    args.output.mkdir(parents=True)
    rows, ensemble_scores = [], []
    for key, raw in samples(args.data, "test"):
        initial, target, action = (array(raw, field) for field in ("gnn_state", "gnn_next_state", "skill_action"))
        with torch.no_grad():
            prediction = backend.predict(initial[None], action[None]).mean(axis=0)[0]
        truth = array(raw, "scene_flows")
        if model is not None:
            pw_prediction, indices, diagnostics = pointworld_prediction(
                model, raw, pw_args, key, return_diagnostics=True)
            if diagnostics is not None:
                ensemble_scores.append(dict(key=key, score=diagnostics["score"],
                    per_step=diagnostics["per_step"], initial_valid_scene_points=diagnostics["points"]))
        else:
            pw_prediction, indices, diagnostics = None, np.arange(truth.shape[1]), None
        truth = truth[:, indices]
        segmentation = array(raw, "scene_segmentation").reshape(-1)[indices]
        poses = json.loads(raw["dense_object_poses.json"])[0]
        for object_index, name in enumerate(("cabinet_item_16", "cabinet_item_17")):
            mask = segmentation == poses[name]["scene_id"]
            row = dict(key=key, object=name, points=int(mask.sum()))
            if not mask.any():
                row["status"] = "no_initial_visible_points"
            else:
                points = truth[0, mask]
                goal = truth[-1, mask]
                predicted = rigid_object_points(initial, prediction, points, object_index)
                offset = 33 + object_index * 12
                row.update(status="evaluated",
                    moving=bool(np.linalg.norm(target[offset:offset+3]-initial[offset:offset+3]) >= .015),
                    gnn_epe_m=float(np.linalg.norm(predicted-goal, axis=-1).mean()),
                    persistence_epe_m=float(np.linalg.norm(points-goal, axis=-1).mean()))
                if pw_prediction is not None:
                    row["pointworld_epe_m"] = float(np.linalg.norm(pw_prediction[-1, mask]-goal, axis=-1).mean())
                if diagnostics is not None:
                    row["pointworld_endpoint_disagreement_m"] = float(diagnostics["std"][-1, mask].sum(-1).mean())
            rows.append(row)
    summary = {}
    for name in ("cabinet_item_16", "cabinet_item_17"):
        summary[name] = {}
        for category in ("all", "moving", "static"):
            selected = [r for r in rows if r["object"] == name and r["status"] == "evaluated"
                        and (category == "all" or r["moving"] == (category == "moving"))]
            summary[name][category] = dict(samples=len(selected), **{
                metric: float(np.mean([r[metric] for r in selected])) if selected else None
                for metric in ("gnn_epe_m", "persistence_epe_m", *(["pointworld_epe_m"] if model else []))})
    write_json(args.output / "per_sample.json", rows)
    write_json(args.output / "evaluation.json", dict(metrics=summary,
        metric="endpoint visible-object point EPE in robot-root metres; equal weight per skill",
        paired_point_subset=model is not None, split="test (never trained)",
        manifest_sha256=sha256(args.data / "manifest.jsonl"),
        gnn_sha256=sha256(args.gnn_checkpoint),
        pointworld_sha256=sha256(args.pointworld_checkpoint) if args.pointworld_checkpoint else None,
        pointworld_ensemble_sha256=sha256(ensemble_path) if ensemble_path else None,
        ensemble_scores=ensemble_scores,
        disagreement_definition="Sample std (correction=1) across independent members; sum XYZ, initial valid points and phase frames; unnormalized; scene only, excludes robot flow",
        limitation="PointWorld sees actual future robot flows; GNN sees only a skill command. Offline comparison, not an online replacement test."))


def pointworld(args):
    """Launch official scripts, keeping normalization restricted to training data."""
    import os
    import yaml
    size = getattr(args, "ensemble_size", 1)
    if size < 1:
        raise ValueError("ensemble-size must be >=1; use 1 for the single-model baseline")
    result = audit(args.data)
    if not result["ready_for_training"]:
        raise ValueError("Paired train/test splits need >=2 complete skills each")
    small_shards = [str(p) for split in ("train", "test")
                    for p in (args.data / split).glob("*.tar") if p.stat().st_size < 1048576]
    if small_shards:
        raise ValueError(f"Official loader would skip shards <1 MiB; repack before training: {small_shards[:3]}")
    weights = PW / "third_party/dinov3/checkpoints"
    if not list(weights.glob("dinov3_vitl16_pretrain*.pth")):
        raise FileNotFoundError(f"Pretrained DINOv3 ViT-L16 weights required in {weights}; no auto-download")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config = dict(seed=args.seed, domains=DOMAIN, data_dirs=str(args.data.resolve()),
        train_splits="train", deterministic_train="true", device=args.device,
        batch_size=args.batch_size, num_workers=0, eval_num_workers=0,
        num_epochs=args.epochs, norm_stats_path=str(args.output.resolve() / "stats"),
        robot_urdf_path=None, log_dir=str(args.output.resolve()), disable_compile="true",
        train_min_num_cameras=1, train_max_num_cameras=1,
        eval_min_num_cameras=1, eval_max_num_cameras=1,
        raps_action_dim=0, model_path=None)
    config_path = args.output.resolve() / "pointworld.yaml"
    config_path.write_text(yaml.safe_dump(config))
    write_json(args.output / "provenance.json", dict(dynamics_initialization="from_scratch",
        dino_weights=[dict(path=str(p), sha256=sha256(p)) for p in weights.glob("dinov3_vitl16_pretrain*.pth")],
        manifest_sha256=sha256(args.data / "manifest.jsonl"),
        conditioning="actual future robot point trajectories (privileged offline input)"))
    commands = [
        [args.python, str(PW / "generate_norm_stats.py"), "--config", str(config_path),
         "--output", str(Path(config["norm_stats_path"]) / "norm_stats.json")]]
    manifest = None
    if size == 1:
        commands.append([args.python, str(PW / "train.py"), "--config", str(config_path)])
    else:
        from exploration.pointworld.pointworld_ensemble import SCHEMA as ENSEMBLE_SCHEMA
        manifest = dict(schema=ENSEMBLE_SCHEMA, status="prepared",
            data_manifest_sha256=sha256(args.data / "manifest.jsonl"),
            initialization="independent dynamics; common frozen pretrained DINO",
            training="sequential official trainers; full same train split; no bootstrap resampling",
            inference="one shared DINO computation; independent learned visual projections and dynamics",
            members=[])
        for index in range(size):
            name = f"member_{index:03d}"
            member_config = {**config, "seed": args.seed + index,
                             "log_dir": str(args.output.resolve() / name), "exp_name": name}
            path = args.output.resolve() / f"{name}.yaml"
            path.write_text(yaml.safe_dump(member_config))
            manifest["members"].append(dict(seed=args.seed + index, config=path.name, status="pending"))
            commands.append([args.python, str(PW / "train.py"), "--config", str(path)])
        write_json(args.output / "ensemble.json", manifest)
    # The reference trainer explicitly selects online W&B unless it sees disabled.
    environment = {**os.environ, "WANDB_MODE": "disabled"}
    try:
        for index, command in enumerate(commands):
            with (args.output / f"stage_{index}.log").open("w") as log:
                subprocess.run(command, cwd=PW, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            if manifest is not None and index > 0:
                paths = list((args.output / f"member_{index-1:03d}").rglob("model-last.pt"))
                if len(paths) != 1:
                    raise RuntimeError(f"Expected exactly one final checkpoint for member {index-1}, found {paths}")
                manifest["members"][index-1].update(status="complete",
                    checkpoint=str(paths[0].relative_to(args.output)), sha256=sha256(paths[0]))
                manifest["status"] = "complete" if index == size else "training"
                write_json(args.output / "ensemble.json", manifest)
    except BaseException as error:
        if manifest is not None:
            manifest.update(status="failed", error=repr(error))
            write_json(args.output / "ensemble.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("collect", "audit", "train-gnn", "train-pointworld", "evaluate"):
        p = sub.add_parser(stage)
        if stage != "collect":
            p.add_argument("--data", type=Path, required=True)
        if stage != "audit":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--device", choices=("cpu", "cuda"), default="cpu" if stage == "collect" else "cuda")
        if stage == "collect":
            p.add_argument("--steps", type=int, required=True)
            p.add_argument("--test-every", type=int, default=5)
            p.add_argument("--checkpoint", type=Path)
        elif stage.startswith("train"):
            p.add_argument("--epochs", type=int, required=True)
            if stage == "train-pointworld":
                p.add_argument("--python", default=sys.executable)
                p.add_argument("--batch-size", type=int, default=1)
                p.add_argument("--ensemble-size", type=int, default=1,
                               help="1 preserves single-model training; use 3 for an independent ensemble")
        elif stage == "evaluate":
            p.add_argument("--gnn-checkpoint", type=Path, required=True)
            group = p.add_mutually_exclusive_group()
            group.add_argument("--pointworld-checkpoint", type=Path)
            group.add_argument("--pointworld-ensemble", type=Path)
    args = parser.parse_args(argv)
    if hasattr(args, "epochs") and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.stage == "collect":
        collect(args)
    elif args.stage == "audit":
        print(json.dumps(audit(args.data)))
    elif args.stage == "train-gnn":
        train_gnn(args)
    elif args.stage == "evaluate":
        evaluate(args)
    else:
        pointworld(args)


if __name__ == "__main__":
    main()
