"""Compare bounded independent CPU grasp jobs with one and two workers.

This diagnostic measures frozen scripted collection only. Each job starts from
its own reset seed; it does not model one continuing production environment or
training updates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import atexit
import json
import multiprocessing
from multiprocessing.util import Finalize
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

from exploration.diagnostics.benchmark_grasp_optimizations import compare_tree, file_digest, resources
from exploration.diagnostics.profile_cup_grasp import CONFIG, ROOT, SOURCES, validate_config


JOB_SEEDS = ((0, 0), (1, 1))
CONTROL_LIMIT = 400
TRACE_FIELDS = ('action', 'full_controller_action', 'state', 'next_state', 'before', 'after',
                'macro_phase', 'macro_status', 'parameters_physical')
_WORKER = None


def canonical_job_results(results):
    """Return results in job order, independent of worker completion order."""
    ids = [int(result['job_id']) for result in results]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate job_id in collection results')
    return sorted(results, key=lambda result: int(result['job_id']))


def compare_job_transitions(left, right):
    """Compare exact recorded physical/controller fields for one fixed job."""
    if len(left) != len(right):
        return [f'transitions.length: {len(left)} != {len(right)}']
    mismatches = []
    for index, (a, b) in enumerate(zip(left, right)):
        selected_a = {key: a.get(key) for key in TRACE_FIELDS}
        selected_b = {key: b.get(key) for key in TRACE_FIELDS}
        mismatches.extend(compare_tree(selected_a, selected_b, f'transitions[{index}]', exact=True))
    return mismatches


def read_transitions(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _close_worker():
    global _WORKER
    if _WORKER is not None:
        try:
            _WORKER['env'].close()
        finally:
            _WORKER = None


def _initialize_worker(ready_queue):
    global _WORKER
    started = time.perf_counter()
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    from omegaconf import OmegaConf
    import torch
    from exploration.cee_us.cee_us_backend import CEEUSBackend
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from exploration.raps.raps_gnn_actor import RAPSImaginationActor

    config = OmegaConf.load(CONFIG)
    validate_config(config)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    backend = CEEUSBackend(device='cpu', seed=0, action_dim=31, agent_dim=33,
        num_objects=1, object_names=('cabinet_item_16',), execution=config.execution,
        object_heuristic=OmegaConf.to_container(config.object_heuristic, resolve=True),
        runtime_optimizations=config.runtime_optimizations)
    actor = RAPSImaginationActor(backend, OmegaConf.to_container(config.raps_actor, resolve=True))
    env, env_config = make_env(episode_steps=config.episode_steps,
        centered_initial_tcp=config.centered_initial_tcp, cabinet_door=config.cabinet_door,
        cabinet_item_layout=config.cabinet_item_layout, cabinet_grid=True,
        cabinet_grid_layout=config.raps_actor.grid_layout, action_mode=config.action_mode,
        low_level_episode_steps=config.episode_steps)
    encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
    if actor.schema.action_dim != 31 or actor.settings['grasp_preflight']:
        env.close()
        raise ValueError('Unexpected current action contract')
    _WORKER = dict(config=config, backend=backend, actor=actor, env=env, encoder=encoder,
                   env_config=OmegaConf.to_container(env_config, resolve=True))
    atexit.register(_close_worker)
    Finalize(None, _close_worker, exitpriority=0)
    if ready_queue is not None:
        ready_queue.put(dict(pid=os.getpid(), initialization_s=time.perf_counter() - started))


def _worker_ready():
    if _WORKER is None:
        raise RuntimeError('CPU worker initialization did not complete')
    return os.getpid()


def _run_job(job_id, seed, directory):
    from domain.robocasa.primitives import PrimitiveAction
    from exploration.raps.raps_gnn_actor import SkillCollector

    worker = _WORKER
    if worker is None:
        raise RuntimeError('CPU worker was not initialized')
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    collector = SkillCollector(worker['env'], worker['encoder'], worker['actor'], directory,
                               collection_optimizations=worker['config'].collection_optimizations)
    target = worker['actor'].schema.encode(
        PrimitiveAction('grasp', dict(x=.410, y=0., z=.732, orientation_id=2)))
    collector._propose_action = lambda policy: target.copy()
    finished = []
    started = time.perf_counter()
    try:
        ticks, skills = collector.collect(CONTROL_LIMIT, 'random', seed=seed,
            stage='cpu_parallel_probe_scripted_grasp', on_skill_end=finished.append,
            stop_when=lambda: bool(finished))
        status = skills[0].get('status', 'unknown') if skills else 'no_macro'
        result = dict(job_id=job_id, seed=seed, status=status, controls=len(ticks),
            macro_attempts=len(skills), success=bool(skills and skills[0].get('status') == 'success'),
            collection_s=time.perf_counter() - started,
            profile_timings=dict(collector.profile_timings),
            transitions=str(directory / 'transitions.jsonl'), skills=str(directory / 'skills.jsonl'),
            environment=worker['env_config'])
        (directory / 'job.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
        return result
    except BaseException as error:
        result = dict(job_id=job_id, seed=seed, status='error', controls=collector.total_steps,
            macro_attempts=len(finished), collection_s=time.perf_counter() - started,
            error=f'{type(error).__name__}: {error}',
            transitions=str(directory / 'transitions.jsonl'))
        (directory / 'job.json').write_text(json.dumps(result, indent=2) + '\n')
        return result


def _run_fresh_job(job_id, seed, directory):
    """Use the same empty scene history for each job in both execution modes."""
    _close_worker()
    started = time.perf_counter()
    _initialize_worker(None)
    initialization_s = time.perf_counter() - started
    try:
        result = _run_job(job_id, seed, directory)
        result['fresh_scene_initialization_s'] = initialization_s
        return result
    finally:
        _close_worker()


def run_mode(output, mode, workers, *, matched_history=False):
    context = multiprocessing.get_context('spawn')
    ready_queue = context.Queue()
    startup_started = time.perf_counter()
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
        initializer=_initialize_worker, initargs=(ready_queue,))
    ready = []
    try:
        ready_futures = [executor.submit(_worker_ready) for _ in range(workers)]
        for future in ready_futures:
            future.result(timeout=240)
        for _ in range(workers):
            ready.append(ready_queue.get(timeout=240))
        if len({item['pid'] for item in ready}) != workers:
            raise RuntimeError('Executor did not initialize the requested number of persistent workers')
        startup_s = time.perf_counter() - startup_started
        jobs = []
        collection_started = time.perf_counter()
        job_function = _run_fresh_job if matched_history else _run_job
        futures = {executor.submit(job_function, job_id, seed,
                    output / mode / f'job_{job_id:02d}' / 'collection'): (job_id, seed)
                   for job_id, seed in JOB_SEEDS}
        for future in as_completed(futures):
            job_id, seed = futures[future]
            try:
                jobs.append(future.result())
            except BaseException as error:
                jobs.append(dict(job_id=job_id, seed=seed, status='worker_error', controls=0,
                                 error=f'{type(error).__name__}: {error}'))
        wall_collection_s = time.perf_counter() - collection_started
        return dict(mode=mode, workers=workers, startup_s=startup_s,
                    startup_plus_collection_wall_s=startup_s + wall_collection_s,
                    wall_collection_s=wall_collection_s, workers_ready=ready,
                    jobs=canonical_job_results(jobs))
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        ready_queue.close()


def compare_modes(output, serial, parallel):
    serial_jobs = {job['job_id']: job for job in serial['jobs']}
    parallel_jobs = {job['job_id']: job for job in parallel['jobs']}
    comparisons = []
    for job_id, seed in JOB_SEEDS:
        left, right = serial_jobs[job_id], parallel_jobs[job_id]
        entry = dict(job_id=job_id, seed=seed, matches=False, mismatches=[])
        if left.get('status') == 'error' or right.get('status') in ('error', 'worker_error'):
            entry['error'] = 'One or both collection jobs failed; see job results.'
        elif left.get('status') == 'worker_error':
            entry['error'] = 'One or both collection jobs failed; see job results.'
        else:
            if (left.get('status') != right.get('status')
                    or left.get('controls') != right.get('controls')
                    or left.get('macro_attempts') != right.get('macro_attempts')):
                entry['mismatches'].append('job status/control/macro counts differ')
            try:
                a = read_transitions(left['transitions'])
                b = read_transitions(right['transitions'])
                entry['mismatches'].extend(compare_job_transitions(a, b))
                entry['matches'] = not entry['mismatches']
            except (OSError, KeyError, ValueError) as error:
                entry['error'] = f'{type(error).__name__}: {error}'
        comparisons.append(entry)
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new output root; must not exist')
    parser.add_argument('--matched-history', action='store_true',
                        help='create a fresh scene for every job in both execution modes')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('serial', 'parallel'):
        (output / name).mkdir()
    serial = run_mode(output, 'serial', workers=1, matched_history=args.matched_history)
    parallel = run_mode(output, 'parallel', workers=2, matched_history=args.matched_history)
    comparisons = compare_modes(output, serial, parallel)
    source_files = tuple(dict.fromkeys((*SOURCES, 'exploration/diagnostics/profile_cpu_sampling.py')))
    report = dict(protocol=dict(jobs=[dict(job_id=i, seed=s) for i, s in JOB_SEEDS],
            controls_per_macro_limit=CONTROL_LIMIT, macro_attempts_per_mode=2,
            total_macro_attempts_limit=4, target=dict(cell=4, orientation_id=2, tcp_z_m=.732),
            physical_fields_compared=TRACE_FIELDS, comparison='exact via compare_tree',
            formal_updates=0, device='cpu', threads_per_worker=1,
            environment_history=('fresh_scene_per_job' if args.matched_history else 'persistent_scene_per_worker')),
        modes=dict(serial=serial, parallel=parallel),
        worker_collection_seconds_sum=dict(
            serial=sum(job.get('collection_s', 0.) for job in serial['jobs']),
            parallel=sum(job.get('collection_s', 0.) for job in parallel['jobs'])),
        speed_interpretation_eligible=all(
            job.get('success') and job.get('controls', 0) > 0
            for mode in (serial, parallel) for job in mode['jobs']),
        comparisons=comparisons,
        resources=resources(), runtime=dict(host=platform.node(), platform=platform.platform(),
            python=sys.version, cpu_count=os.cpu_count()),
        source_sha256={name: file_digest(ROOT / name) for name in source_files},
        limitations=['Frozen scripted actions measure independent reset streams, not production one-environment state continuation.',
                     'Only physical/controller transition fields are compared; no full training equivalence is claimed.',
                     'Shared host load is observed, not isolated; timings are diagnostic and do not assert speedup.'])
    (output / 'report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    return 0 if all(item['matches'] for item in comparisons) else 1


if __name__ == '__main__':
    raise SystemExit(main())
