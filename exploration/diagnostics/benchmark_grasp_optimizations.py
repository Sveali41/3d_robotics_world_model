"""Bounded pipeline proxy for opt-in grasp collection/runtime engineering changes.

Run against a frozen grasp-only RAPS checkpoint and its sibling replay files.
This measures random collection followed by fixed offline fitting, actor updates
and a completed checkpoint write. It does not measure production grasp throughput.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import numpy as np


PROFILES = ('reference', 'geometry', 'reads', 'runtime', 'combined')


def validate_budget(repeats, ticks):
    if (isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 3
            or isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1
            or repeats * ticks > 300):
        raise ValueError('Require 1..3 blocks, positive integer ticks, and blocks*ticks <= 300 per profile')
    return repeats * ticks


def profile_order(block):
    if block == 0:
        return PROFILES
    if block == 1:
        return PROFILES[::-1]
    return ('geometry', 'reads', 'reference', 'runtime', 'combined')


def compare_tree(left, right, path='root', *, exact=False):
    """Return leaf mismatches, using the approved floating point tolerance."""
    if hasattr(left, 'detach'):
        left = left.detach().cpu().numpy()
    if hasattr(right, 'detach'):
        right = right.detach().cpu().numpy()
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return [path + ': keys']
        return [item for key in left for item in compare_tree(left[key], right[key], f'{path}.{key}', exact=exact or key == 'rng')]
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [path + ': length']
        return [item for i, (a, b) in enumerate(zip(left, right))
                for item in compare_tree(a, b, f'{path}[{i}]', exact=exact)]
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        if a.shape != b.shape:
            return [path + ': shape']
        equal = (np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=False)
                 if not exact and (a.dtype.kind in 'fc' or b.dtype.kind in 'fc') else np.array_equal(a, b))
        return [] if equal else [path]
    if not exact and (isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating))):
        return [] if np.isclose(left, right, rtol=1e-5, atol=1e-6) else [path]
    return [] if left == right else [path]


def resources():
    result = dict(load_average=os.getloadavg(), cpu_count=os.cpu_count(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)), pid=os.getpid())
    for key, fields in (('gpu', 'index,name,memory.total,memory.used,utilization.gpu'),
                        ('gpu_processes', 'pid,process_name,used_memory')):
        command = ['nvidia-smi', '--query-' + ('gpu=' if key == 'gpu' else 'compute-apps=') + fields,
                   '--format=csv,noheader,nounits']
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
            result[key] = completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            result[key] = str(error)
    return result


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def replay_rows(arrays):
    return [dict(state=arrays['states'][i], action=arrays['actions'][i],
                 next_state=arrays['next_states'][i], episode_id=int(arrays['episode_ids'][i]),
                 terminated=bool(arrays['terminated'][i]), truncated=bool(arrays['truncated'][i]),
                 trainable=True, duration_low_steps=int(arrays.get('duration_low_steps',
                     np.ones(len(arrays['states']), dtype=int))[i]))
            for i in range(len(arrays['states']))]


def geometry_equivalence(raw):
    """Audit actual native shapes in two virtual obstacle frames without physics."""
    from domain.robocasa.motion_planning import _array, _native_modules, _pose_vector
    planner = raw.motion_planner
    planner.mplib, planner.fcl, planner.sapien = _native_modules()
    planner.root = planner._pose(planner.root_link.pose)
    planner.attachment, planner.allowed_contacts = None, ()
    planner.support_surface_root_z = None
    q0 = _array(planner.robot.get_qpos()).reshape(9).copy()
    limits = _array(planner.robot.get_qlimits()).reshape(9, 2)[planner.pindices]
    candidates = [q0[planner.pindices].copy()]
    candidates.append(np.clip(candidates[0] + np.r_[np.full(7, .005), 0., 0.], limits[:, 0], limits[:, 1]))
    original_env = planner.env
    before = dict(qpos=q0.copy(), scene=planner.snapshot_scene())
    frames = []
    try:
        for shift in (0., .01):
            items = [SimpleNamespace(name=item.name, _bodies=item._bodies,
                        pose=planner.sapien.Pose(_array(item.pose.p).reshape(3) + [shift, 0., 0.],
                                                 _array(item.pose.q).reshape(4)))
                     for item in raw.cabinet_items]
            proxy = SimpleNamespace(cabinet=raw.cabinet, wall=raw.wall, ground=raw.ground,
                                    cabinet_items=items, _collection_profile={})
            planner.env = proxy
            outputs = []
            for optimized in (False, True, True):
                proxy.collection_optimizations = optimized
                planner._scene_geometry()
                outputs.append(dict(
                    obstacles=[(x['name'], list(x['groups']), _pose_vector(x['pose']).tolist())
                               for x in planner.obstacles],
                    robot=[(x['link'], list(x['groups']), _pose_vector(x['local']).tolist())
                           for x in planner.robot_shapes],
                    collisions=[planner._check(q) for q in candidates]))
            if outputs[0] != outputs[1] or outputs[1] != outputs[2]:
                raise AssertionError('Native cached/uncached geometry audit differs')
            frames.append(outputs[0])
        if frames[0]['obstacles'] == frames[1]['obstacles']:
            raise AssertionError('Virtual obstacle pose change was not refreshed')
    finally:
        planner.env = original_env
        planner.kin.pmodel.compute_forward_kinematics(q0[planner.pindices])
    if not np.array_equal(q0, _array(planner.robot.get_qpos()).reshape(9)) or before['scene'] != planner.snapshot_scene():
        raise AssertionError('Zero-control geometry audit changed physical state')
    return dict(status='equivalent', physical_ticks=0, candidate_count=len(candidates),
                frames='current scene and +.01m virtual item X offset; physical poses unchanged',
                frame_results=frames)


def run(args):
    import torch
    from omegaconf import OmegaConf
    from exploration.raps import raps_gnn_actor as raps
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from exploration.diagnostics.diagnose_cabinet import jsonable, write_json

    validate_budget(args.repeats, args.ticks)
    checkpoint = args.checkpoint.resolve()
    source_paths = [checkpoint, checkpoint.with_suffix('.replay.npz'), checkpoint.with_suffix('.low_level.npz')]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f'Frozen checkpoint bundle is incomplete: {path}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = OmegaConf.create(payload['config'])
    settings = payload['actor']['settings']
    if (payload.get('schema') != raps.SCHEMA or not settings.get('grasp_only', False)
            or payload['action_contract']['action_dim'] != 4 or payload['backend'].get('action_dim') != 4
            or raps.skill_schema(settings).action_dim != 4):
        raise ValueError('Require a genuine grasp-only 4D RAPS/GNN checkpoint')
    if int(config.epochs) != 25 or int(settings['updates_per_round']) != 100:
        raise ValueError('Saved preset must retain 25 WM epochs and 100 actor updates')
    if config.execution != 'reference' or int(config.sim_freq) != 100 or int(config.control_freq) != 20:
        raise ValueError('Require unchanged reference prediction and 100/20 Hz simulation')
    if int(settings['hidden_size']) != 128:
        raise ValueError('Require saved actor hidden_size=128')
    torch.set_num_threads(int(config.get('cpu_threads', 1)))
    device = str(config.device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Saved CUDA checkpoint requires CUDA; no device fallback')
    with np.load(source_paths[1], allow_pickle=False) as source:
        count = min(len(source['states']), 251)
        fixed_replay = {key: value[:count].copy() for key, value in source.items()}
    with np.load(source_paths[2], allow_pickle=False) as source:
        fixed_low = {key: value[:min(len(source['states']), 251)].copy() for key, value in source.items()}
    if count < 2 or fixed_replay['actions'].shape[1] != 4:
        raise ValueError('Require at least two valid grasp-only endpoint replay rows')
    fixed_skills, fixed_ticks = replay_rows(fixed_replay), replay_rows(fixed_low)
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = args.output / 'source'
    frozen.mkdir()
    import shutil
    for path in source_paths:
        shutil.copy2(path, frozen / path.name)
    checkpoint = frozen / checkpoint.name
    manifest = dict(
        description='Bounded random-collection + fixed offline training/checkpoint pipeline proxy',
        production_grasp_throughput=False, physical_ticks_per_profile=args.repeats * args.ticks,
        physical_warmup_ticks=0, blocks=args.repeats, ticks_per_block=args.ticks,
        fixed_replay_rows=count, fixed_replay_selection='first min(source rows,251); identical every fit',
        epochs=25, actor_updates=100, wm_batch_size=125, wm_hidden_dim=128,
        source=[dict(path=str(path), sha256=file_digest(path)) for path in source_paths],
        config=OmegaConf.to_container(config, resolve=True), settings=settings,
        cabinet_door=config.get('cabinet_door', False),
        source_checkpoint_real_steps=payload.get('real_steps'),
        source_checkpoint_backend_updates=payload['backend'].get('updates'),
        source_checkpoint_actor_updates=payload['actor'].get('updates'),
        source_checkpoint_benchmark_proxy=payload.get('benchmark_proxy'),
        source_checkpoint_benchmark_fixture=payload.get('benchmark_fixture'),
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'scipy', 'sapien', 'mplib')},
        initial_resources=resources(),
        limitations=['Native MPLib RRT is unseeded; live trajectories are not compared bitwise',
                     'Shared CPU/GPU resources are observed, not isolated',
                     'Checkpoint writes are drained immediately: no save/training overlap is measured',
                     'Fixed offline replay decouples fit work from collection; this is not full grasp throughput',
                     'Planner total/path/TOPP and collector stage timings overlap; never sum child timings'])
    write_json(args.output / 'manifest.json', manifest)
    env = None
    records = []
    comparisons = []
    states = {}
    physical_ticks = {name: 0 for name in PROFILES}
    collector, active_name, completed_profile_ticks = None, None, 0
    try:
        env, _ = raps.make_env(episode_steps=int(config.episode_steps),
            cabinet_door=config.get('cabinet_door', False),
            centered_initial_tcp=bool(config.centered_initial_tcp), action_mode='joint_position',
            low_level_episode_steps=int(config.episode_steps))
        env.reset(seed=int(config.seed))
        encoder = CabinetSkillStateEncoder(env, latched_gripper=bool(settings['latched_gripper']))
        raw = env.unwrapped
        geometry = geometry_equivalence(raw)
        write_json(args.output / 'geometry_equivalence.json', geometry)
        for optimized in (False, True):
            backend = raps.CEEUSBackend(device=device, seed=int(config.seed), action_dim=4,
                agent_dim=32 + int(settings['latched_gripper']), execution='reference',
                object_heuristic=payload['backend']['object_heuristic'], runtime_optimizations=optimized)
            actor = raps.RAPSImaginationActor(backend, settings)
            raps.load(checkpoint, backend, actor, encoder)
            # Offline warmup never steps physics; each measured profile reloads the complete checkpoint.
            backend.train(fixed_replay, epochs=25)
            actor.train(fixed_replay, updates=100)
            states[optimized] = backend, actor
        for block in range(args.repeats):
            paired = {}
            for name in profile_order(block):
                runtime = name in ('runtime', 'combined')
                reads = name in ('reads', 'combined')
                geometry_enabled = name in ('geometry', 'combined')
                backend, actor = states[runtime]
                raps.load(checkpoint, backend, actor, encoder)
                directory = args.output / f'block_{block + 1}' / name
                class ProfileCollector(raps.SkillCollector):
                    def reset(self, seed):
                        super().reset(seed)
                        self.env.unwrapped.collection_optimizations = geometry_enabled
                collector = ProfileCollector(env, encoder, actor, directory, collection_optimizations=reads)
                active_name, completed_profile_ticks = name, physical_ticks[name]
                backend.profile_timings = {}
                def synchronize():
                    if backend.device.type == 'cuda':
                        torch.cuda.synchronize(backend.device)
                synchronize()
                if backend.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(backend.device)
                resource_before = resources()
                total_started = time.perf_counter()
                collect_started = time.perf_counter()
                def count_tick(row):
                    physical_ticks[name] += 1
                    if physical_ticks[name] > args.repeats * args.ticks:
                        raise AssertionError('Physical control budget exceeded')
                ticks, skills = collector.collect(args.ticks, 'random', seed=int(config.seed),
                    stage='benchmark_proxy', on_step=count_tick)
                synchronize()
                collection_s = time.perf_counter() - collect_started
                if len(ticks) != args.ticks:
                    raise AssertionError('Collector did not execute the exact bounded control budget')
                # Collection is intentionally separate from the fixed paired offline experiment.
                raps.restore_rng(payload['rng'])
                fit_started = time.perf_counter()
                wm_report = backend.train(fixed_replay, epochs=25)
                synchronize()
                wm_s = time.perf_counter() - fit_started
                actor_started = time.perf_counter()
                actor_report = actor.train(fixed_replay, updates=100)
                synchronize()
                actor_s = time.perf_counter() - actor_started
                saved_config = deepcopy(config)
                saved_config.runtime_optimizations = runtime
                saved_config.collection_optimizations = reads
                save_started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    completion = raps.save(directory / 'proxy.pt', backend, actor, encoder, saved_config,
                        fixed_ticks, fixed_skills, executor=executor if runtime else None)
                    checkpoint_timings = completion.result() if runtime else completion
                synchronize()
                save_completion_s = time.perf_counter() - save_started
                total_s = time.perf_counter() - total_started
                peak = (torch.cuda.max_memory_allocated(backend.device) if backend.device.type == 'cuda' else None)
                saved = torch.load(directory / 'proxy.pt', map_location='cpu', weights_only=False)
                paired[name] = dict(backend=saved['backend'], actor=saved['actor'], rng=saved['rng'],
                                    wm_report=wm_report, actor_report=actor_report)
                record = dict(block=block + 1, profile=name, status='complete', physical_ticks=len(ticks),
                    collection_s=collection_s, wm_s=wm_s, actor_s=actor_s,
                    checkpoint_completion_s=save_completion_s, total_s=total_s,
                    proxy_ticks_per_s=len(ticks) / total_s, gpu_peak_allocated_bytes=peak,
                    collector_timings=collector.profile_timings, backend_timings=backend.profile_timings,
                    checkpoint_timings=checkpoint_timings, wm_report=wm_report, actor_report=actor_report,
                    outcomes=raps.summarize(ticks, skills, actor.schema),
                    macro_statuses=[dict(status=x['status'], duration=x['duration_low_steps'],
                                         phase=x.get('macro_phase')) for x in skills],
                    resources_before=resource_before, resources_after=resources())
                records.append(record)
                write_json(directory / 'measurement.json', record)
                with (args.output / 'measurements.jsonl').open('a') as handle:
                    handle.write(json.dumps(jsonable(record), allow_nan=False) + '\n')
            for name in PROFILES:
                mismatch = compare_tree(paired['reference'], paired[name])
                comparison = dict(block=block + 1, profile=name, rtol=1e-5, atol=1e-6,
                                  equivalent=not mismatch, mismatches=mismatch)
                comparisons.append(comparison)
                write_json(args.output / 'equivalence.json', comparisons)
                if mismatch:
                    raise AssertionError(f'Paired training equivalence failed: {comparison}')
        summary = {}
        for name in PROFILES:
            rows = [row for row in records if row['profile'] == name]
            values = [row['total_s'] for row in rows]
            ratios = [next(row['total_s'] for row in records if row['block'] == value['block']
                           and row['profile'] == 'reference') / value['total_s'] for value in rows]
            summary[name] = dict(total_s_median=statistics.median(values), total_s_min=min(values),
                                 total_s_max=max(values), paired_speed_ratio_median=statistics.median(ratios),
                                 physical_ticks=sum(row['physical_ticks'] for row in rows))
        result = dict(status='complete', verdict='unconfirmed_speedup', profiles=summary,
                      training_equivalent=all(item['equivalent'] for item in comparisons),
                      reason='Bounded proxy, shared resources and unseeded native RRT do not establish a production speed target',
                      target_10_percent_achieved=False, final_resources=resources())
        write_json(args.output / 'summary.json', result)
        return result
    except Exception as error:
        if collector is not None and active_name is not None:
            physical_ticks[active_name] = completed_profile_ticks + collector.total_steps
        write_json(args.output / 'failure.json', dict(error=repr(error), completed_records=len(records),
            physical_ticks_by_profile=physical_ticks,
            verdict='unconfirmed_speedup'))
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--ticks', type=int, default=32)
    args = parser.parse_args()
    validate_budget(args.repeats, args.ticks)
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
