"""Render an explicitly in-sample fitting curve using the system plotting runtime."""
import argparse
import json
from pathlib import Path


def plot_history(history, path, *, label='Current run', comparisons=(), online=False, reference=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, grid = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = grid.ravel()
    fields = [('conditional_one_step', 'success_accuracy', 'Action-success accuracy'),
              ('success_probability', 'bce', 'Action-success BCE'),
              ('conditional_one_step', 'changed_coordinates_mae', 'Changed-coordinate MAE'),
              ('conditional_one_step', 'parent_relation_f1', 'Parent-relation F1')]
    for name, records in [*comparisons, (label, history)]:
        epoch = [r['training_attempts' if online else 'epoch'] for r in records]
        line_color = ({'Random actions': 'tab:green', 'Curiosity': 'tab:blue'}.get(name)
                      if online else None)
        for ax, (group, field, title) in zip(axes, fields):
            ax.plot(epoch, [r['evaluation'][group][field] for r in records], marker='o',
                    label=name, color=line_color)
            ax.set(title=title, xlabel='Online training attempts' if online else 'Epochs over the fixed training dataset')
    if reference is not None:
        name, record = reference
        for ax, (group, field, _) in zip(axes, fields):
            ax.axhline(record['evaluation'][group][field], color='tab:orange', linestyle='--', label=name)
    if not online:
        axes[0].axhline(history[0]['evaluation']['success_majority_baseline'], color='gray', linestyle='--', label='Majority')
        axes[2].axhline(history[0]['evaluation']['conditional_one_step']['zero_change_changed_coordinates_mae'], color='gray', linestyle='--', label='Zero change')
    for ax in (axes[0], axes[3]):
        ax.set_ylim(0, 1)
    for ax in axes:
        ax.legend()
    fig.suptitle('Online exploration — fixed evaluation data' if online else 'Same-data fitting diagnostic — not held-out validation')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_label(manifest):
    mode = manifest['prediction_mode']
    heads = 'Three heads + constrained/weighted relations' if mode == 'mixed-relations' else 'Two heads'
    graph = 'GNS' if manifest.get('graph', {}).get('mode') == 'gns' else 'full graph'
    return f'{heads}, {graph}'


def online_label(manifest):
    variant = manifest.get('arguments', {}).get('exploration', 'curiosity')
    return 'Random actions' if variant == 'random' else 'Curiosity'


def online_evaluation_sha(history_path):
    metadata_path = history_path.parent / 'standardized_evaluation_manifest.json'
    if metadata_path.exists():
        return json.loads(metadata_path.read_text()).get('evaluation_sha256')
    manifest_path = history_path.parent / 'manifest.json'
    if manifest_path.exists():
        return json.loads(manifest_path.read_text()).get('evaluation_sha256')
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--online', action='store_true', help='Use online interaction counts and evaluation labeling')
    parser.add_argument('--compare-history', type=Path, action='append', default=[],
                        help='Overlay another run; training and monitor dataset hashes must match')
    args = parser.parse_args()
    comparisons = []
    reference = None
    if args.online:
        metadata = json.loads((args.history.parent / 'standardized_evaluation_manifest.json').read_text())
        saved = metadata.get('offline_reference')
        if saved:
            import hashlib
            path = Path(saved['history'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != saved['sha256']:
                parser.error('Offline reference history changed; update the explicit reference selection')
            manifest = json.loads((path.parent / 'manifest.json').read_text())
            if manifest.get('monitoring_sha256') != metadata['evaluation_sha256']:
                parser.error('Offline reference uses different evaluation samples')
            records = json.loads(path.read_text())
            record = next(r for r in records if r['epoch'] == saved['epoch'])
            reference = (f"Offline fit, epoch {saved['epoch']} (in-sample)", record)
    label = 'Current run'
    if args.compare_history:
        manifest = json.loads((args.history.parent / 'manifest.json').read_text())
        label = online_label(manifest) if args.online else run_label(manifest)
        for path in args.compare_history:
            other = json.loads((path.parent / 'manifest.json').read_text())
            if args.online:
                if not online_evaluation_sha(args.history) or online_evaluation_sha(args.history) != online_evaluation_sha(path):
                    parser.error(f'Cannot overlay {path}: evaluation_sha256 differs or is missing')
                comparison_label = online_label(other)
            else:
                for key in ('evaluation_sha256', 'monitoring_sha256'):
                    if not manifest.get(key) or manifest[key] != other.get(key):
                        parser.error(f'Cannot overlay {path}: {key} differs or is missing')
                comparison_label = run_label(other)
            comparisons.append((comparison_label, json.loads(path.read_text())))
    elif args.online:
        manifest = json.loads((args.history.parent / 'manifest.json').read_text())
        label = online_label(manifest)
    plot_history(json.loads(args.history.read_text()), args.output, label=label, comparisons=comparisons, online=args.online, reference=reference)


if __name__ == '__main__':
    main()
