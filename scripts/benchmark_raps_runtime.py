"""Bounded paired offline runtime ablations; never runs physics or edits its source checkpoint.

Run from the repository root with PYTHONPATH=. in the established RAPS runtime.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch

from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.cee_us.official.mbrl.torch_helpers import TorchTrainingIterator
from exploration.raps.raps_gnn_actor import (
    RAPSImaginationActor, fit, load, restore_rng, rng_state, save,
)


MODES = {
    'baseline': (False, False, False),
    'device_replay_only': (True, False, False),
    'sync_only': (False, True, False),
    'background_save_only': (False, False, True),
    'combined': (True, True, True),
}


def assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_equal(a, e)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    else:
        assert actual == expected


def rows(arrays, *, skills=False):
    result = []
    for i in range(len(arrays['states'])):
        row = {key: arrays[source][i] for key, source in (
            ('state', 'states'), ('action', 'actions'), ('next_state', 'next_states'),
            ('episode_id', 'episode_ids'), ('terminated', 'terminated'), ('truncated', 'truncated'))}
        if skills:
            row.update(trainable=True, duration_low_steps=int(arrays['duration_low_steps'][i]))
        result.append(row)
    return result


def read_replay(path, count=None):
    with np.load(path) as saved:
        if count is not None and count > len(saved['states']):
            raise ValueError(f'--samples exceeds {path} replay size')
        return {key: saved[key][:count].copy() for key in saved.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--low-level-replay', type=Path,
                        help='Actual control replay used only for checkpoint serialization; required if companion is absent')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--updates', type=int, default=5)
    parser.add_argument('--execution', choices=('reference', 'cuda_graph'), default='reference')
    args = parser.parse_args()
    if min(args.repeats, args.epochs, args.updates) < 1 or args.samples < 2:
        parser.error('repeats/epochs/updates must be positive; samples must be at least two')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; select --device cpu explicitly')
    if args.execution == 'cuda_graph' and args.device != 'cuda':
        parser.error('cuda_graph requires CUDA')
    if args.output.exists():
        parser.error('--output must be a new directory; refusing overwrite')
    def gpu_load():
        if args.device != 'cuda':
            return None
        return {
            'processes': subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                '--format=csv,noheader'], text=True).strip().splitlines(),
            'load': subprocess.check_output([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader'], text=True).strip().splitlines(),
        }

    gpu_load_before = gpu_load()
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    replay = read_replay(args.checkpoint.with_suffix('.replay.npz'), args.samples)
    low_source = args.low_level_replay or args.checkpoint.with_suffix('.low_level.npz')
    if not low_source.exists():
        parser.error('Companion low-level NPZ missing; supply --low-level-replay with actual recorded control data')
    low_replay = {key: value[:args.samples] for key, value in read_replay(low_source).items()}
    skills, ticks = rows(replay, skills=True), rows(low_replay)
    contract = payload['state_contract']
    encoder = SimpleNamespace(contract=lambda: deepcopy(contract))
    config = OmegaConf.create(deepcopy(payload['config']))
    config.epochs = args.epochs
    config.raps_actor.updates_per_round = args.updates
    config.device = args.device
    config.execution = args.execution
    torch.set_num_threads(int(config.get('cpu_threads', 1)))
    first_weight = payload['backend']['model']['edge_mlp.layers.0.0.weight']
    backend = CEEUSBackend(
        device=args.device, execution=args.execution,
        agent_dim=contract['agent_dim'], object_dyn_dim=contract['object_dyn_dim'],
        object_stat_dim=contract['object_stat_dim'], num_objects=len(contract['object_names']),
        object_names=contract['object_names'], action_dim=payload['action_contract']['action_dim'],
        ensemble_size=first_weight.shape[0], hidden_dim=first_weight.shape[-1],
        object_heuristic=payload['backend']['object_heuristic'])
    actor = RAPSImaginationActor(backend, payload['actor']['settings'])
    load(args.checkpoint, backend, actor, encoder)
    actor.settings['updates_per_round'] = args.updates
    initial_backend = deepcopy(backend.state_dict())
    initial_actor = deepcopy(actor.state_dict())
    args.output.mkdir(parents=True, exist_ok=False)

    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize(backend.device)

    def timed(operation):
        sync()
        began = time.perf_counter()
        value = operation()
        sync()
        return value, time.perf_counter() - began

    def reset(mode, seed):
        device_replay, scalar_sync, _ = MODES[mode]
        backend.runtime_optimizations = scalar_sync
        backend.model.runtime_optimizations = device_replay
        backend.load_state_dict(deepcopy(initial_backend))
        actor.load_state_dict(deepcopy(initial_actor))
        # Restore identical NumPy/Torch/Python states for every paired trial.
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    cold = {}
    # Warm forward/backward/optimizer and any graph recapture after fitting in every mode.
    for mode in MODES:
        reset(mode, 101)
        _, cold[mode] = timed(lambda: (backend.train(replay, epochs=args.epochs),
                                      actor.train(replay, updates=2)))
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    results = {mode: [] for mode in MODES}
    iterator_results = {'reference': [], 'device': []}
    actor_results = {'reference': [], 'sync': []}
    parity_checks = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        for repeat in range(args.repeats):
            order = list(MODES) if repeat % 2 == 0 else list(reversed(MODES))
            snapshots = {}
            for mode in order:
                reset(mode, 200 + repeat)
                directory = args.output / f'{repeat}_{mode}'
                directory.mkdir()
                sync()
                began = time.perf_counter()
                fitted = fit(backend, actor, skills, config, directory, 'offline')
                saved, submit_s = timed(lambda: save(
                    directory / 'snapshot.pt', backend, actor, encoder, config, ticks, skills,
                    executor=executor if MODES[mode][2] else None))
                # Same short real actor work in every mode, allowing asynchronous I/O overlap.
                extra, extra_s = timed(lambda: actor.train(replay, updates=1))
                drain_started = time.perf_counter()
                save_timings = saved.result() if MODES[mode][2] else saved
                drain_s = time.perf_counter() - drain_started
                sync()
                total_s = time.perf_counter() - began
                saved_payload = torch.load(directory / 'snapshot.pt', map_location='cpu', weights_only=False)
                for suffix, arrays in (('.replay.npz', replay), ('.low_level.npz', low_replay)):
                    with np.load((directory / 'snapshot.pt').with_suffix(suffix)) as actual:
                        assert set(actual.files) == set(arrays)
                        for key in arrays:
                            np.testing.assert_array_equal(actual[key], arrays[key])
                snapshots[mode] = dict(backend=deepcopy(backend.state_dict()),
                                       actor=deepcopy(actor.state_dict()), rng=deepcopy(rng_state()),
                                       saved=saved_payload, model_metrics=fitted['model'],
                                       actor_metrics=fitted['actor'], extra_metrics=extra)
                results[mode].append(dict(
                    **fitted['timings'], **{k: v for k, v in save_timings.items() if k != 'path'},
                    save_submit_s=submit_s, overlap_actor_s=extra_s, drain_s=drain_s,
                    total_including_drain_s=total_s,
                    reference_minibatch_upload_s=None,
                    wm_optimizer_steps=fitted['model']['updates'], actor_updates=args.updates + 1))
            for mode in MODES:
                assert_equal(snapshots[mode], snapshots['baseline'])
                parity_checks += 1
            # Isolate scalar-sync actor work with the unchanged checkpoint WM.
            for mode in (('reference', 'sync') if repeat % 2 == 0 else ('sync', 'reference')):
                reset('sync_only' if mode == 'sync' else 'baseline', 300 + repeat)
                metrics, seconds = timed(lambda: actor.train(replay, updates=args.updates))
                actor_results[mode].append(dict(seconds=seconds, metrics=metrics))
            assert_equal(actor_results['sync'][-1]['metrics'], actor_results['reference'][-1]['metrics'])
            # Isolate upload + NumPy sampling + gathers + transfers, with identical batches/RNG.
            batches = {}
            for mode in (('reference', 'device') if repeat % 2 == 0 else ('device', 'reference')):
                np.random.seed(400 + repeat)

                def iterate():
                    data = {key: replay[key] for key in ('states', 'actions', 'next_states')}
                    if mode == 'device':
                        data = {key: torch.as_tensor(value, device=args.device) for key, value in data.items()}
                    iterator = TorchTrainingIterator(data, ensemble=True, ensemble_size=backend.ensemble_size,
                                                     runtime_optimizations=mode == 'device')
                    return [{key: torch.as_tensor(value, device=args.device) for key, value in batch.items()}
                            for batch in iterator.get_epoch_iterator_non_bootstrapped(
                                backend.model.batch_size, args.epochs)()]

                batches[mode], seconds = timed(iterate)
                iterator_results[mode].append(seconds)
                batches[mode + '_rng'] = np.random.get_state()
            assert_equal(batches['device'], batches['reference'])
            assert_equal(batches['device_rng'], batches['reference_rng'])
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    medians = {mode: {key: statistics.median(row[key] for row in values)
                      for key in values[0] if values[0][key] is not None}
               for mode, values in results.items()}
    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
        source_checkpoint_unchanged=True, torch=torch.__version__, device=args.device,
        gpu_load_before=gpu_load_before, gpu_load_after=gpu_load(),
        device_name=torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU',
        execution=args.execution, samples=args.samples, epochs=args.epochs, updates=args.updates,
        low_level_replay_source=str(low_source), low_level_save_samples=len(low_replay['states']),
        low_level_replay_sha256=hashlib.sha256(low_source.read_bytes()).hexdigest(),
        mode_switches={name: dict(device_replay_and_wm_loss_batching=flags[0],
                                 actor_and_backend_scalar_sync=flags[1], background_save=flags[2])
                       for name, flags in MODES.items()},
        repeats=args.repeats, cold_fit_and_two_actor_updates_s=cold, results=results, medians=medians,
        actor_sync_only=actor_results, iterator_upload_gather_s=iterator_results,
        combined_speed_ratio=medians['baseline']['total_including_drain_s'] / medians['combined']['total_including_drain_s'],
        actor_sync_speed_ratio=statistics.median(r['seconds'] for r in actor_results['reference']) /
                              statistics.median(r['seconds'] for r in actor_results['sync']),
        iterator_speed_ratio=statistics.median(iterator_results['reference']) / statistics.median(iterator_results['device']),
        parity_checks=parity_checks, parity_rtol=1e-5, parity_atol=1e-6,
        exact_saved_replay_verified=True, real_control_steps=0,
        measured_wm_steps_per_mode=args.repeats * args.epochs * int(np.ceil(args.samples / backend.model.batch_size)),
        measured_actor_updates_per_mode=args.repeats * (args.updates + 1),
        warmup_wm_steps_per_mode=args.epochs * int(np.ceil(args.samples / backend.model.batch_size)),
        warmup_actor_updates_per_mode=2,
        isolated_actor_updates_per_sync_mode=args.repeats * args.updates,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == 'cuda' else None,
        limitations=[
            'Offline capped real compact6 replay; no simulation, collection, or full-pipeline speed claim.',
            'device_replay_only also batches WM scalar-loss logging; this existing model branch is coupled.',
            'Reference minibatch upload is included in WM fit; isolated iterator timing includes upload/gather/indexing.',
            'Same one extra actor update follows every save; async total includes drain and checkpoint write.',
            'Warm trials restore deepcopied model/actor/optimizer/normalizer/RNG state; graph fitting invalidates captures.',
            'Low-level save-only replay has explicit independent provenance; it is never used to fit WM or actor.',
            'Capped low-level replay and skill replay are serialized independently; no historical rollout reconstructed.',
            'Peak CUDA allocation includes parity snapshots, retained microbenchmark batches, and live model.',
            'Other GPU processes may vary during timing; before/after process and utilization metadata are recorded.',
            'Timing variability is retained in individual trial rows; no trained split7 performance claim.'])
    (args.output / 'offline.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in (
        'combined_speed_ratio', 'actor_sync_speed_ratio', 'iterator_speed_ratio', 'parity_checks')}, indent=2))


if __name__ == '__main__':
    main()
