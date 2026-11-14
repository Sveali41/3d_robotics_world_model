"""Bounded, isolated full-recording scalar/GPU collection measurements."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from exploration.diagnostics import collect_cabinet_grasp_sequences as collection
from exploration.diagnostics.calibrate_wm_validation import checkpoint_contract
from exploration.diagnostics.diagnose_cabinet import write_json


def benchmark_cases(seed):
    """One fixed balanced pool; no timed configuration chooses easier cases."""
    all_cases = collection.cases(seed)
    ids = np.linspace(0, 485, 64, dtype=int)
    return sorted((all_cases[int(i)+shift] for i in ids for shift in (0, 486)),
                  key=lambda case: case['case_id'])


def recommend(measurements, memory_total_mib):
    eligible = [m for m in measurements if m.get('status') == 'complete'
                and m.get('collector') == 'gpu_batch'
                and 0 < m.get('active_control_ticks', 0)
                and m.get('peak_gpu_memory_mib') is not None
                and m['peak_gpu_memory_mib'] <= .8*memory_total_mib]
    if not eligible:
        return None
    fastest = max(m['end_to_end_ticks_per_second'] for m in eligible)
    return min((m for m in eligible if m['end_to_end_ticks_per_second'] >= .95*fastest),
               key=lambda m: m['num_envs'])['num_envs']


def scalar_measurement(output, manifest, payload, settings, max_steps, max_seconds):
    """Bound the actual scalar collector through its existing on-step callback."""
    from exploration.raps.raps_gnn_actor import SkillCollector
    original = SkillCollector.collect
    original_make = collection.make_env
    ticks = 0
    init_seconds = 0.
    collection_start = None
    start = time.perf_counter()

    def make(*args, **kwargs):
        nonlocal init_seconds
        began = time.perf_counter()
        value = original_make(*args, **kwargs)
        init_seconds += time.perf_counter()-began
        return value

    def collect(self, count, policy, **kwargs):
        nonlocal ticks, collection_start
        if collection_start is None:
            collection_start = time.perf_counter()
        callback = kwargs.get('on_step')
        def step(row):
            nonlocal ticks
            ticks += 1
            if callback is not None:
                callback(row)
            if ticks >= max_steps or time.perf_counter()-collection_start >= max_seconds:
                raise collection.Finished()
        kwargs['on_step'] = step
        return original(self, min(count, max_steps-ticks), policy, **kwargs)

    collection.make_env = make
    SkillCollector.collect = collect
    try:
        for case in manifest['cases']:
            if ticks >= max_steps or (collection_start is not None and
                    time.perf_counter()-collection_start >= max_seconds):
                break
            collection.run_case(output, case, payload, settings)
    finally:
        collection.make_env = original_make
        SkillCollector.collect = original
    loop_end = time.perf_counter()
    collection.export(output, action_dim=13)
    result = collection.results(output, manifest['cases'])
    write_json(output/'result.json', result)
    finished = time.perf_counter()
    endpoints = sum(1 for p in (output/'cases').glob('*/attempt_*/skills.jsonl')
                    for row in collection.read_rows(p) if row['completed'] and row['duration_low_steps'] > 0)
    return dict(active_control_ticks=ticks, initialization_seconds=init_seconds,
                collection_seconds=loop_end-(collection_start or loop_end),
                total_seconds=finished-start, export_seconds=finished-loop_end,
                completed_skills=endpoints, closed_cases=result['total']['closed'])


def worker(args):
    torch.set_num_threads(1)
    payload, config, settings = checkpoint_contract(args.checkpoint)
    settings = dict(settings, skill_set='split7')
    args.stage, args.skill_set, args.export_every_cases = 'collect', 'split7', 64
    args.resume = False
    manifest = collection.identity(args, payload, config, settings)
    manifest.update(split='performance_calibration_only', automatic_training=False,
                    cases=benchmark_cases(args.seed), max_active_control_ticks=args.max_steps,
                    max_collection_seconds=args.max_seconds)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in manifest['source_sha256']:
        destination = args.output/'sources'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((collection.ROOT/name).read_bytes())
    write_json(args.output/'manifest.json', manifest)
    if args.collector == 'scalar':
        metrics = scalar_measurement(args.output, manifest, payload, settings, args.max_steps, args.max_seconds)
    else:
        from exploration.diagnostics.collect_gpu_split7 import run_gpu
        metrics = run_gpu(args.output, manifest, payload, settings, args.num_envs, 64,
                          max_control_steps=args.max_steps, max_seconds=args.max_seconds)
    metrics.update(collector=args.collector, num_envs=args.num_envs,
                   status='complete', calibration_only=True)
    write_json(args.output/'benchmark_metrics.json', metrics)
    return 0


def gpu_memory():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    rows = [tuple(map(int, row.split(','))) for row in result.strip().splitlines()]
    if len(rows) != 1:
        raise RuntimeError('This benchmark requires one explicitly visible GPU')
    return rows[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--max-steps', type=int, default=4096)
    parser.add_argument('--max-seconds', type=float, default=60.)
    parser.add_argument('--num-envs', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--collector', choices=('scalar', 'gpu_batch'), default='scalar', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 < args.max_steps <= 4096 or not 0 < args.max_seconds <= 60:
        parser.error('Benchmark limits must be within 4096 active controls and 60 seconds')
    if any(n < 1 or n > 128 for n in args.num_envs):
        parser.error('Benchmark environments must be between 1 and 128')
    if args.worker:
        if len(args.num_envs) != 1:
            parser.error('Worker needs exactly one environment count')
        args.num_envs = args.num_envs[0]
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    total_memory, _ = gpu_memory()
    write_json(args.output/'case_pool.json', benchmark_cases(args.seed))
    measurements = []
    for collector, n in [('scalar', 1)] + [('gpu_batch', n) for n in args.num_envs]:
        destination = args.output/f'{collector}_{n}'
        command = [sys.executable, '-m', __name__ if __name__ != '__main__' else
                   'exploration.diagnostics.benchmark_split7_collection', '--worker',
                   '--checkpoint', str(args.checkpoint.resolve()), '--output', str(destination.resolve()),
                   '--collector', collector, '--num-envs', str(n), '--seed', str(args.seed),
                   '--max-steps', str(args.max_steps), '--max-seconds', str(args.max_seconds)]
        patch_manifest = os.environ.get('SAPIEN_SPARSE_PATCH_MANIFEST')
        if collector == 'gpu_batch' and patch_manifest:
            command = [sys.executable, str(collection.ROOT/'scripts/run_patched_sapien.py'),
                       '--runtime', str(Path(patch_manifest).parent), '--', *command[1:]]
        peak = gpu_memory()[1]
        with (args.output/f'{collector}_{n}.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                peak = max(peak, gpu_memory()[1])
                time.sleep(.5)
        path = destination/'benchmark_metrics.json'
        if process.returncode == 0 and path.exists():
            measurement = json.loads(path.read_text())
            elapsed = measurement['total_seconds']
            measurement['end_to_end_ticks_per_second'] = measurement['active_control_ticks']/elapsed
            measurement['end_to_end_skills_per_second'] = measurement['completed_skills']/elapsed
            measurement['end_to_end_cases_per_second'] = measurement['closed_cases']/elapsed
        else:
            measurement = dict(collector=collector, num_envs=n, status='failed',
                               returncode=process.returncode, log=str(log.name))
        measurement['peak_gpu_memory_mib'] = peak
        measurements.append(measurement)
        selected = recommend(measurements, total_memory)
        report = dict(gpu_total_memory_mib=total_memory, measurements=measurements,
                      recommended_num_envs=selected, calibration_only=True)
        scalar = next((m for m in measurements if m['collector'] == 'scalar' and m['status'] == 'complete'), None)
        winner = next((m for m in measurements if m['collector'] == 'gpu_batch' and m['num_envs'] == selected), None)
        if scalar and winner:
            report['measured_speedup'] = winner['end_to_end_ticks_per_second']/scalar['end_to_end_ticks_per_second']
            report['fivefold_target_met'] = report['measured_speedup'] >= 5
        write_json(args.output/'report.json', report)
        print(json.dumps(measurement), flush=True)
    return 0 if recommend(measurements, total_memory) is not None else 2


if __name__ == '__main__':
    raise SystemExit(main())
