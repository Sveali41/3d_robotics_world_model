"""Fixed independent CPU collection streams with a round barrier."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch

PROTOCOL = "cpu-multistream-fresh-scene-v1"


def sampling_contract(config):
    setting = config.get("cpu_sampling")
    if setting is None:
        return None
    if OmegaConf.is_config(setting):
        setting = OmegaConf.to_container(setting, resolve=True)
    if not isinstance(setting, dict) or set(setting) != {"workers", "streams"}:
        raise ValueError("cpu_sampling requires exactly workers and streams")
    for name in ("workers", "streams"):
        value = setting[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"cpu_sampling.{name} must be a positive integer")
    if setting["streams"] != 2:
        raise ValueError("cpu_sampling.streams must be 2 for this protocol")
    return dict(protocol=PROTOCOL, workers=setting["workers"], streams=setting["streams"],
                rng_scheme="SeedSequence([config_seed, collection_seed, stream_id]) modulo 2**31")


def _stream_seed(base_seed, collection_seed, stream_id):
    return int(np.random.SeedSequence([int(base_seed), int(collection_seed), int(stream_id)])
               .generate_state(1, dtype=np.uint32)[0] % (2**31))


def _initialize_worker():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _run_shard(job):
    from exploration.cee_us.cee_us_backend import CEEUSBackend
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder, cabinet_agent_dim
    from exploration.raps.raps_gnn_actor import RAPSImaginationActor, SkillCollector, skill_schema

    started = time.perf_counter()
    config, settings = job["config"], job["config"]["raps_actor"]
    layout = config.get("cabinet_item_layout", "two_upper_second_row")
    if layout not in {"single_upper_second_row", "two_upper_second_row"}:
        raise ValueError(f"Unsupported cabinet_item_layout={layout!r}")
    names = (("cabinet_item_16",) if layout == "single_upper_second_row"
             else ("cabinet_item_16", "cabinet_item_17"))
    backend = CEEUSBackend(device="cpu", seed=int(config["seed"]),
        action_dim=skill_schema(settings).action_dim,
        agent_dim=cabinet_agent_dim(config.get("cabinet_door", False),
                                    latched_gripper=settings["latched_gripper"]), num_objects=len(names),
        object_names=names, execution="reference", object_heuristic=config.get("object_heuristic"),
        **dict(config.get("wm_training") or {}),
        runtime_optimizations=bool(config.get("runtime_optimizations", False)))
    actor = RAPSImaginationActor(backend, settings)
    backend.load_state_dict(job["backend"])
    actor.load_state_dict(job["actor"])
    env = None
    try:
        env, _ = make_env(episode_steps=config.get("episode_steps", 100),
            centered_initial_tcp=config.get("centered_initial_tcp", False),
            cabinet_door=config.get("cabinet_door", False), cabinet_item_layout=layout,
            cabinet_grid=settings.get("pick_place_grid", False),
            cabinet_grid_layout=settings.get("grid_layout", "legacy_3x4"),
            object_shift_xy=job["object_shift_xy"],
            action_mode="joint_position" if settings["motion_execution"] == "planner" else "ee_delta",
            low_level_episode_steps=(int(config.get("episode_steps", 100))
                                     if settings["pick_place_primitives"] else None))
        env.reset(seed=int(config["seed"]))
        encoder = CabinetSkillStateEncoder(env, latched_gripper=settings["latched_gripper"])
        if encoder.contract()["dimension"] != backend.state_dim or tuple(encoder.object_names) != backend.object_names:
            raise ValueError("Cabinet state encoder and world model object/state dimensions differ")
        collector = SkillCollector(env, encoder, actor, job["directory"],
            collection_optimizations=bool(config.get("collection_optimizations", False)))
        startup = time.perf_counter() - started
        seed = job["stream_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        ticks, skills = collector.collect(job["count"], job["policy"], seed=seed, stage=job["stage"])
        return dict(stream_id=job["stream_id"], stream_seed=seed, ticks=ticks, skills=skills,
                    timings={"startup_s": startup, **collector.profile_timings})
    finally:
        if env is not None:
            env.close()


def _merge(results, quotas, total_steps, episode_id, skill_id):
    ticks, skills, timings = [], [], {}
    if sorted(r["stream_id"] for r in results) != sorted(quotas):
        raise RuntimeError("Collection did not return every requested stream exactly once")
    for result in sorted(results, key=lambda r: r["stream_id"]):
        stream = result["stream_id"]
        shard_ticks, shard_skills = result["ticks"], result["skills"]
        if len(shard_ticks) != quotas[stream]:
            raise RuntimeError(f"Stream {stream} returned {len(shard_ticks)} controls; expected {quotas[stream]}")
        episode_map, skill_map = {}, {}
        for row in shard_skills:
            for field, mapping in (("episode_id", episode_map), ("skill_id", skill_map)):
                if row[field] not in mapping:
                    mapping[row[field]] = (episode_id if field == "episode_id" else skill_id) + len(mapping) + 1
        for row in shard_ticks:
            if row["episode_id"] not in episode_map or row["skill_id"] not in skill_map:
                raise RuntimeError(f"Stream {stream} tick refers to an unknown episode or skill")
        provenance = dict(stream_id=stream, stream_seed=result["stream_seed"])
        for row in shard_ticks:
            ticks.append({**row, **provenance, "step": total_steps + row["step"],
                          "episode_id": episode_map[row["episode_id"]], "skill_id": skill_map[row["skill_id"]]})
        for row in shard_skills:
            skills.append({**row, **provenance, "start_step": total_steps + row["start_step"],
                           "end_step": total_steps + row["end_step"],
                           "episode_id": episode_map[row["episode_id"]], "skill_id": skill_map[row["skill_id"]]})
        total_steps += len(shard_ticks)
        episode_id += len(episode_map)
        skill_id += len(skill_map)
        for name, seconds in result["timings"].items():
            timings[name] = timings.get(name, 0.) + seconds
    return ticks, skills, total_steps, episode_id, skill_id, timings


class CpuRoundCollector:
    def __init__(self, config, actor, directory, *, object_shift_xy=(0, 0), workers=None):
        contract = sampling_contract(config)
        if contract is None:
            raise ValueError("CpuRoundCollector requires cpu_sampling configuration")
        self.workers = contract["workers"] if workers is None else workers
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        self.streams = contract["streams"]
        self.config = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config
        self.config = dict(self.config, raps_actor=dict(actor.settings))
        self.actor = actor
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.object_shift_xy = tuple(object_shift_xy)
        self.total_steps = self.episode_id = self.skill_id = 0
        self.profile_timings = {}

    def collect(self, count, policy, *, seed, stage):
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            return self._collect(count, policy, seed=seed, stage=stage)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)

    def _collect(self, count, policy, *, seed, stage):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1 or policy not in {"random", "curiosity", "task"}:
            raise ValueError("Positive integer real-step budget and random/curiosity/task policy required")
        if not isinstance(stage, str) or not stage or stage in {".", ".."} or Path(stage).name != stage:
            raise ValueError("stage must be a single nonempty path component")
        stage_dir = self.directory / "streams" / stage
        stage_dir.mkdir(parents=True, exist_ok=False)
        backend = self.actor.backend
        state = backend._move_tensors(dict(backend=backend.state_dict(), actor=self.actor.state_dict()), "cpu")
        base, remainder = divmod(count, self.streams)
        quotas = {stream: base + (stream < remainder) for stream in range(self.streams) if base + (stream < remainder)}
        jobs = [dict(config=self.config, **state, stream_id=stream,
                     stream_seed=_stream_seed(self.config["seed"], seed, stream), count=quota,
                     policy=policy, stage=stage, directory=str(stage_dir / f"stream_{stream}"),
                     object_shift_xy=self.object_shift_xy) for stream, quota in quotas.items()]
        started = time.perf_counter()
        results = []
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize_worker) as pool:
            futures = {pool.submit(_run_shard, job): job["stream_id"] for job in jobs}
            for future in as_completed(futures):
                stream = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"CPU collection stage {stage!r} stream {stream} failed: {exc}") from exc
        merged = _merge(results, quotas, self.total_steps, self.episode_id, self.skill_id)
        ticks, skills, total_steps, episode_id, skill_id, timings = merged
        for filename, rows in (("transitions.jsonl", ticks), ("skills.jsonl", skills)):
            with (self.directory / filename).open("a") as handle:
                for row in rows:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.total_steps, self.episode_id, self.skill_id = total_steps, episode_id, skill_id
        self.profile_timings = {"worker_sum_s": timings, "round_wall_s": time.perf_counter() - started}
        return ticks, skills
