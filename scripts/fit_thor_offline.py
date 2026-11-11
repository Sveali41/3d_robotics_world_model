"""Fit an existing THOR dataset as an explicitly contaminated diagnostic.

This is not held-out validation. No Unity instance or exploration is started.
Diagnostic checkpoints have a distinct schema and cannot resume online training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch

from exploration.cee_us.cee_us_thor import (
    file_hash, load_dataset, make_backend, make_manifest, source_hashes, write_json,
)
from exploration.cee_us.official.mbrl.rolloutbuffer import Rollout, RolloutBuffer
from exploration.cee_us.thor_learning_curve import evaluate_one_step

PURPOSE = 'same-data fitting diagnostic; validation data intentionally used for training; not held-out accuracy'


def assess(backend, encoder, data, batch_size):
    report = evaluate_one_step(backend, encoder, data, batch_size=batch_size)
    scores = []
    for start in range(0, len(data['states']), batch_size):
        stop = start + batch_size
        scores.append(backend.predict(data['states'][start:stop], data['actions'][start:stop]).mean(0)[:, 8])
    probability = np.concatenate(scores).astype(np.float64)
    truth = data['next_states'][:, 8] >= .5
    if np.any(probability < 0) or np.any(probability > 1):
        raise ValueError('This diagnostic requires probability outputs from --prediction-mode mixed')
    bounded = np.clip(probability, 1e-7, 1 - 1e-7)
    previous = data['states'][:, 8] >= .5
    predicted = probability >= .5
    report['success_probability'] = {
        'brier': float(np.mean((probability - truth) ** 2)),
        'bce': float(-np.mean(truth * np.log(bounded) + (~truth) * np.log(1 - bounded))),
        'matches_previous_count': int(np.sum(previous == predicted)),
        'previous_outcome_accuracy': float(np.mean(previous == truth)),
        'balanced_accuracy': float(np.mean([np.mean(predicted[truth] == truth[truth]),
                                           np.mean(predicted[~truth] == truth[~truth])])) if truth.any() and (~truth).any() else None,
        'confusion_actual_rows_predicted_columns_success_first': [
            [int(np.sum(truth & predicted)), int(np.sum(truth & ~predicted))],
            [int(np.sum(~truth & predicted)), int(np.sum(~truth & ~predicted))]],
    }
    report['purpose'] = PURPOSE
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--monitor-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--prediction-mode', choices=('mixed', 'mixed-relations'), default='mixed')
    parser.add_argument('--gns', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--gns-radius', type=float, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.epochs, args.hidden_dim, args.eval_batch_size) < 1:
        parser.error('Epochs, hidden dimension and batch size must be positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this process')
    from exploration.cee_us.thor_gns import graph_config
    graph = graph_config(args)
    args.eval_data = args.data
    torch.set_num_threads(1)
    encoder, _, training = load_dataset(args.data)
    monitor_encoder, _, monitor = load_dataset(args.monitor_data)
    if monitor_encoder.contract() != encoder.contract():
        raise ValueError('Training and monitoring scene contracts differ')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = make_manifest(args)
    manifest.update(purpose=PURPOSE, online_interactions=0,
                    training_rows=len(training['states']), monitoring_rows=len(monitor['states']),
                    monitoring_sha256=file_hash(args.monitor_data), script_sha256=file_hash(__file__))
    write_json(args.output / 'manifest.json', manifest)
    for name in (*manifest['sources'], __file__):
        source = Path(name)
        relative = source.relative_to(Path.cwd()) if source.is_absolute() else source
        target = args.output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    backend = make_backend(encoder, args)
    normalization_data = RolloutBuffer(rollouts=[Rollout.from_dict(
        observations=training['states'], actions=training['actions'],
        next_observations=training['next_states'], rewards=np.zeros(len(training['states']), dtype=np.float32))])
    backend.model.update_normalizer(normalization_data)
    del normalization_data
    history = []
    for epoch in range(args.epochs + 1):
        fit = None
        if epoch:
            started = perf_counter()
            fit = backend.train(training, 1)
            if backend.device.type == 'cuda':
                torch.cuda.synchronize()
            fit['seconds'] = perf_counter() - started
            payload = dict(schema='thor-offline-fit-v1', purpose=PURPOSE,
                           backend=backend.state_dict(), contract=encoder.contract(),
                           hidden_dim=args.hidden_dim, prediction_mode=args.prediction_mode, seed=args.seed,
                           graph=graph, epochs_completed=epoch, training_rows=len(training['states']),
                           dataset_sha256=manifest['evaluation_sha256'], sources=source_hashes())
            temporary = args.output / 'checkpoint.tmp'
            torch.save(payload, temporary)
            temporary.replace(args.output / 'checkpoint.pt')
        started = perf_counter()
        evaluation = assess(backend, encoder, monitor, args.eval_batch_size)
        history.append(dict(epoch=epoch, fit=fit, evaluation_seconds=perf_counter()-started,
                            evaluation=evaluation))
        write_json(args.output / 'history.json', history)
        print(json.dumps(dict(epoch=epoch, fit=fit,
                              metrics=evaluation['conditional_one_step'],
                              success_probability=evaluation['success_probability'])), flush=True)
    full = assess(backend, encoder, training, args.eval_batch_size)
    full['training_dataset_sha256'] = manifest['evaluation_sha256']
    full['epochs_completed'] = args.epochs
    write_json(args.output / 'full_fitting_report.json', full)
    assert file_hash(args.data) == manifest['evaluation_sha256']
    write_json(args.output / 'completed.json', dict(epochs=args.epochs, online_interactions=0,
               training_rows=len(training['states']), purpose=PURPOSE))
    print('Completed offline fitting diagnostic', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
