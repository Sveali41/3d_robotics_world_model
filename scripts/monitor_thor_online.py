"""Evaluate saved online checkpoints with the same terms as offline fitting.

This separate reader leaves the running trainer and its source hashes unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from exploration.cee_us.cee_us_thor import (
    file_hash, load_dataset, read_checkpoint, restore_checkpoint, write_json,
)
from scripts.fit_thor_offline import assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--eval-data', type=Path, required=True)
    parser.add_argument('--full-data', type=Path, required=True)
    parser.add_argument('--trainer-pid', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--variant', choices=('random', 'curiosity'),
                        help='Checkpoint/history prefix; inferred from the run manifest when omitted')
    parser.add_argument('--compare-history', type=Path, action='append', default=[],
                        help='Overlay another standardized online history on the same evaluation data')
    parser.add_argument('--reference-history', type=Path, help='Offline fitted checkpoint level as a horizontal reference')
    args = parser.parse_args()
    torch.set_num_threads(1)
    encoder, _, data = load_dataset(args.eval_data)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    variant = args.variant or manifest.get('arguments', {}).get('exploration', 'curiosity')
    if variant not in ('random', 'curiosity'):
        raise ValueError(f'Unsupported online exploration variant: {variant}')
    if file_hash(args.eval_data) != manifest['evaluation_sha256']:
        raise ValueError('Monitoring data hash differs from training manifest')
    output = args.run / 'standardized_learning_curve.json'
    history = json.loads(output.read_text()) if output.exists() else []
    done = {r['training_attempts'] for r in history}
    reference = None
    if args.reference_history:
        reference_rows = json.loads(args.reference_history.read_text())
        reference = {'history': str(args.reference_history.resolve()),
                     'sha256': file_hash(args.reference_history), 'epoch': reference_rows[-1]['epoch']}
    comparisons = [{'history': str(path.resolve()), 'sha256': file_hash(path)}
                   for path in args.compare_history]
    write_json(args.run / 'standardized_evaluation_manifest.json', {
        'evaluation_sha256': file_hash(args.eval_data),
        'probability_metrics_source': 'scripts.fit_thor_offline.assess',
        'probability_metrics_sha256': file_hash('scripts/fit_thor_offline.py'),
        'monitor_sha256': file_hash(__file__),
        'plot_sha256': file_hash('scripts/plot_thor_offline_fit.py'),
        'training_changed': False, 'offline_reference': reference,
        'comparison_histories': comparisons,
        'evaluation_role': 'Excluded from this run fitting; reused exploratory benchmark, not untouched test data',
    })
    os.environ['MPLCONFIGDIR'] = '/tmp/mpl-thor-standardized-online'
    os.environ['MPLBACKEND'] = 'Agg'
    while True:
        source = args.run / f'{variant}_learning_curve.json'
        try:
            rows = json.loads(source.read_text())
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        for row in rows:
            if row['training_attempts'] in done:
                continue
            name = 'bootstrap.pt' if row['round'] == 0 else f"{variant}_round_{row['round']:04d}.pt"
            payload = read_checkpoint(args.run / name)
            if payload['attempts'] != row['training_attempts']:
                raise ValueError('Checkpoint attempts differ from history entry')
            settings = SimpleNamespace(hidden_dim=payload['hidden_dim'], device=args.device, seed=payload['seed'])
            backend, _, _ = restore_checkpoint(payload, encoder, settings, restore_rng=False)
            report = assess(backend, encoder, data, 4)
            report['purpose'] = 'Fixed evaluation data excluded from this online run fitting; not same-data fitting'
            # Existing shared metrics must agree; only probability diagnostics are added.
            for key in ('dynamic_mae', 'changed_coordinates_mae', 'success_accuracy', 'parent_relation_f1'):
                expected = row['evaluation']['conditional_one_step'][key]
                actual = report['conditional_one_step'][key]
                if not np.isclose(expected, actual, rtol=1e-5, atol=1e-6):
                    raise ValueError(f'Checkpoint re-evaluation differs for {key}: {actual} vs {expected}')
            history.append({**row, 'evaluation': report, 'checkpoint': name})
            history.sort(key=lambda r: r['training_attempts'])
            write_json(output, history)
            done.add(row['training_attempts'])
            plot = ['python3', 'scripts/plot_thor_offline_fit.py', '--online',
                    '--history', str(output), '--output', str(args.run / 'learning_curve.png')]
            for comparison in args.compare_history:
                plot.extend(['--compare-history', str(comparison)])
            subprocess.run(plot, check=True)
            print(json.dumps({'attempts': row['training_attempts'], 'success_bce': report['success_probability']['bce']}), flush=True)
            del backend, payload
        proc = Path(f'/proc/{args.trainer_pid}/cmdline')
        try:
            active = str(args.run.resolve()).encode() in proc.read_bytes().replace(b'\0', b' ')
        except FileNotFoundError:
            active = False
        if not active:
            latest = json.loads(source.read_text())
            if any(r['training_attempts'] not in done for r in latest):
                continue
            # A final complete training report distinguishes successful completion from interruption.
            if (args.run / 'report.json').exists():
                full = args.run / 'full_evaluation_cuda.json'
                if not full.exists():
                    subprocess.run([str(Path('.runtime/thor-ceeus/bin/python').resolve()), '-s', '-m',
                                    'artifacts.thor_learning_curve.evaluate_full', '--checkpoint', str(args.run / f'{variant}.pt'),
                                    '--eval-data', str(args.full_data), '--device', args.device, '--batch-size', '4',
                                    '--output', str(full)], check=True)
                write_json(args.run / 'monitor_completed.json', {'training_completed': True, 'full_evaluation': True})
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
