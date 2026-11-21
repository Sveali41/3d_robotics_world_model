"""Profile one bounded current-protocol cup grasp and a fixed two-row offline fit.

The offline input is a genuine pre-existing replay; the single newly collected
endpoint is never duplicated or used as a substitute for training data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np

from exploration.diagnostics.benchmark_grasp_optimizations import file_digest, resources

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml'
SOURCES = (
    'configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml', 'configs/exploration.yaml',
    'exploration/raps/raps_gnn_actor.py', 'exploration/cee_us/cee_us_backend.py',
    'exploration/cee_us/cee_us_cabinet.py', 'exploration/cee_us/cee_us_state.py',
    'domain/robocasa/motion_planning.py', 'domain/robocasa/primitives.py',
    'domain/robocasa/pick_place.py', 'domain/robocasa/grid_pick_place.py',
    'domain/wall_cabinet_env.py', 'main.py', 'exploration/diagnostics/profile_cup_grasp.py',
)


def validate_replay(replay):
    required = {'states': (2, 50), 'next_states': (2, 50), 'actions': (2, 31),
                'terminated': (2,), 'truncated': (2,), 'episode_ids': (2,)}
    for key, shape in required.items():
        if key not in replay or replay[key].shape != shape:
            raise ValueError(f'Expected genuine two-row replay {key} shape {shape}')
    if not np.isfinite(replay['states']).all() or not np.isfinite(replay['actions']).all() or not np.isfinite(replay['next_states']).all():
        raise ValueError('Replay contains nonfinite values')
    if np.asarray(replay['terminated'], dtype=bool).all():
        raise ValueError('Replay has no nonterminal actor starting states')


def validate_config(config, *, learning=True):
    expected = {'seed': 0, 'epochs': 25, 'sim_freq': 100, 'control_freq': 20,
                'action_mode': 'joint_position', 'cabinet_item_layout': 'single_upper_second_row',
                'cabinet_door': True, 'centered_initial_tcp': True,
                'collection_optimizations': True, 'runtime_optimizations': True,
                'execution': 'reference', 'cpu_threads': 1}
    actor_expected = {'grid_layout': 'trapezoid_4_3_2_1', 'grasp_preflight': False,
                      'updates_per_round': 100, 'imagination_horizon': 5,
                      'imagination_batch_size': 64, 'hidden_size': 128,
                      'grasp_only': True, 'pick_place_grid': True,
                      'grasp_orientation_candidates': True, 'latched_gripper': True}
    if not learning:
        expected['cabinet_door'] = False
        expected.pop('epochs')
        for key in ('updates_per_round', 'imagination_horizon', 'imagination_batch_size', 'hidden_size'):
            actor_expected.pop(key)
    for values, wanted in ((config, expected), (config.raps_actor, actor_expected)):
        for key, value in wanted.items():
            if values.get(key) != value:
                raise ValueError(f'Current profiling protocol requires {key}={value!r}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    parser.add_argument('--replay', type=Path, required=True, help='genuine two-endpoint native replay NPZ')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--label', required=True)
    args = parser.parse_args(argv)
    from omegaconf import OmegaConf
    import torch
    from domain.robocasa.primitives import PrimitiveAction
    from exploration.cee_us.cee_us_backend import CEEUSBackend
    from exploration.cee_us.cee_us_cabinet import make_env, rng_state
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from exploration.raps.raps_gnn_actor import RAPSImaginationActor, SkillCollector, _timed, jsonable

    config = OmegaConf.load(CONFIG)
    validate_config(config)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    with np.load(args.replay, allow_pickle=False) as source:
        replay = {key: source[key] for key in source.files}
    validate_replay(replay)
    args.output.mkdir(parents=True, exist_ok=False)
    for source_path in SOURCES:
        destination = args.output / 'sources' / source_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source_path, destination)
    torch.set_num_threads(int(config.cpu_threads))
    random.seed(int(config.seed)); np.random.seed(int(config.seed)); torch.manual_seed(int(config.seed))
    started = time.perf_counter()
    backend = CEEUSBackend(device=args.device, seed=int(config.seed), action_dim=31,
        agent_dim=33, num_objects=1, object_names=('cabinet_item_16',),
        execution=config.execution, object_heuristic=OmegaConf.to_container(config.object_heuristic, resolve=True),
        runtime_optimizations=config.runtime_optimizations)
    actor = RAPSImaginationActor(backend, OmegaConf.to_container(config.raps_actor, resolve=True))
    env, env_config = make_env(episode_steps=config.episode_steps,
        centered_initial_tcp=config.centered_initial_tcp, cabinet_door=config.cabinet_door,
        cabinet_item_layout=config.cabinet_item_layout, cabinet_grid=True,
        cabinet_grid_layout=config.raps_actor.grid_layout, action_mode=config.action_mode,
        low_level_episode_steps=config.episode_steps)
    try:
        encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
        collector = SkillCollector(env, encoder, actor, args.output / 'collection',
                                   collection_optimizations=config.collection_optimizations)
        if backend.device.type == 'cuda':
            torch.cuda.synchronize(backend.device)
        initialization_s = time.perf_counter() - started
        if actor.schema.action_dim != 31 or actor.settings['grasp_preflight']:
            raise ValueError('Unexpected current action contract')
        target = actor.schema.encode(PrimitiveAction('grasp', dict(x=.410, y=0., z=.732, orientation_id=2)))
        collector._propose_action = lambda policy: target.copy()
        finished = []
        scene = env.unwrapped.scene
        scene_step = scene.step
        physics = dict(seconds=0., calls=0)
        def timed_scene_step():
            started_step = time.perf_counter()
            try:
                return scene_step()
            finally:
                physics['seconds'] += time.perf_counter() - started_step
                physics['calls'] += 1
        scene.step = timed_scene_step
        try:
            (ticks, skills), collection_s = _timed(backend, lambda: collector.collect(
                400, 'random', seed=int(config.seed), stage='profile_scripted_grasp',
                on_skill_end=finished.append, stop_when=lambda: bool(finished)))
        finally:
            scene.step = scene_step
        if len(skills) != 1 or len(ticks) > 400:
            raise RuntimeError('Collection violated the one-macro, 400-control bound')
        save_started = time.perf_counter()
        torch.save(dict(backend=backend.state_dict(), actor=actor.state_dict(), rng=rng_state()), args.output / 'start_state.pt')
        start_save_s = time.perf_counter() - save_started
        backend.profile_timings = {}
        model_metrics, model_s = _timed(backend, lambda: backend.train(replay, epochs=int(config.epochs)))
        model_components = dict(backend.profile_timings)
        actor_metrics, actor_s = _timed(backend, lambda: actor.train(replay))
        _, final_save_s = _timed(backend, lambda: torch.save(
            dict(backend=backend.state_dict(), actor=actor.state_dict(), rng=rng_state()), args.output / 'final_state.pt'))
        report = dict(label=args.label, device=args.device, replay=str(args.replay.resolve()),
            replay_sha256=file_digest(args.replay), source_sha256={p: file_digest(ROOT / p) for p in SOURCES},
            config=OmegaConf.to_container(config, resolve=True), env_config=OmegaConf.to_container(env_config, resolve=True),
            resources=resources(), times_s=dict(initialization=initialization_s, collection=collection_s,
                start_state_save=start_save_s, world_model_fit=model_s, actor_critic=actor_s,
                final_state_save_completion=final_save_s),
            collection=dict(controls=len(ticks), skills=skills, profile_timings=collector.profile_timings,
                physics_step=physics,
                timing_note='Components overlap and are not additive. physics_step is scene.px.step only, nested in composite env_step_s.'),
            training=dict(replay_rows=2, epochs=25, actor_updates=100, model=model_metrics,
                          model_profile_timings=model_components, actor=actor_metrics),
            state_files_note='State snapshots are offline comparison proxies, not production checkpoints; no fabricated skills.')
        (args.output / 'report.json').write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + '\n')
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
