"""Check THOR GPU prediction parity, planning throughput and peak device memory.

Uses an existing checkpoint's training replay; it never starts Unity or fits on
validation data. The optional tiny fit is discarded and does not save weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from exploration.cee_us.cee_us_thor import (
    file_hash, read_checkpoint, restore_checkpoint, source_hashes, write_json,
)
from exploration.cee_us.thor_cem import ThorCEM
from exploration.cee_us.thor_state import ThorState


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunks', type=int, nargs='+', default=[1, 4, 8, 12])
    parser.add_argument('--eval-batches', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--fit-smoke', action='store_true')
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this process. Check nvidia-smi and GPU device access; no CPU fallback is used.')
    if min(args.chunks + args.eval_batches + [args.repeats]) < 1:
        parser.error('Chunk sizes, evaluation batches and repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    payload = read_checkpoint(args.checkpoint)
    encoder = ThorState(payload['initial_metadata'])
    replay = payload['replay']
    settings = SimpleNamespace(hidden_dim=payload['hidden_dim'], device='cpu', seed=payload['seed'])
    cpu, _, _ = restore_checkpoint(payload, encoder, settings, restore_rng=False)
    state = replay['states'][0]
    sequences = np.random.default_rng(0).integers(0, 14, size=(12, 3))
    planner_args = dict(horizon=3, candidates=12, iterations=2, elites=4)
    expected = cpu.predict(replay['states'][:4], replay['actions'][:4])
    expected_scores = ThorCEM(cpu, encoder, **planner_args).score(state, sequences)
    cpu_times, cpu_plans = [], []
    for seed in range(args.repeats):
        planner = ThorCEM(cpu, encoder, seed=seed, **planner_args)
        start = perf_counter(); result = planner.plan(state)
        cpu_times.append(perf_counter() - start); cpu_plans.append(result)
    del cpu

    settings.device = 'cuda'
    gpu, _, _ = restore_checkpoint(payload, encoder, settings, restore_rng=False)
    actual = gpu.predict(replay['states'][:4], replay['actions'][:4])
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    report = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  cuda=torch.version.cuda, checkpoint_sha256=file_hash(args.checkpoint),
                  sources=source_hashes(), script_sha256=file_hash(__file__),
                  prediction_max_abs_difference=float(np.abs(actual - expected).max()),
                  cpu_plan_seconds=cpu_times, cpu_plans=cpu_plans,
                  planning=planner_args, planning_results=[], prediction_batches=[],
                  precision='FP32; existing PyTorch TF32 settings unchanged',
                  allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tolerances=dict(prediction_rtol=1e-4, prediction_atol=1e-5,
                                  score_rtol=1e-4, score_atol=1e-2))
    for chunk in args.chunks:
        planner = ThorCEM(gpu, encoder, chunk_size=chunk, **planner_args)
        scores = planner.score(state, sequences)
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-4, atol=1e-2)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times, plans = [], []
        for seed in range(args.repeats):
            planner = ThorCEM(gpu, encoder, chunk_size=chunk, seed=seed, **planner_args)
            torch.cuda.synchronize(); start = perf_counter(); result = planner.plan(state)
            torch.cuda.synchronize(); times.append(perf_counter() - start); plans.append(result)
        row = dict(chunk_size=chunk, seconds=times,
                   median_seconds=float(np.median(times)), plans=plans,
                   actions_match_cpu=all(a[0] == b[0] for a, b in zip(plans, cpu_plans)),
                   score_max_abs_difference=float(np.abs(scores - expected_scores).max()),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report['planning_results'].append(row)
        write_json(args.output / 'benchmark.json', report)
        print(json.dumps(row), flush=True)
    for size in args.eval_batches:
        if size > len(replay['states']):
            raise ValueError('Evaluation batch exceeds available checkpoint training samples')
        gpu.predict(replay['states'][:size], replay['actions'][:size])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        for _ in range(args.repeats):
            gpu.predict(replay['states'][:size], replay['actions'][:size])
        torch.cuda.synchronize()
        row = dict(batch_size=size, seconds_per_sample=(perf_counter() - start) / (size * args.repeats),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report['prediction_batches'].append(row)
        print(json.dumps(row), flush=True)
    if args.fit_smoke:
        count = min(8, len(replay['states']))
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = perf_counter()
        fit = gpu.train({key: value[:count] for key, value in replay.items()}, 1)
        torch.cuda.synchronize()
        report['discarded_training_smoke'] = dict(samples=count, epochs=1, fit=fit,
            seconds=perf_counter() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
    report['cpu_median_plan_seconds'] = float(np.median(cpu_times))
    write_json(args.output / 'benchmark.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
