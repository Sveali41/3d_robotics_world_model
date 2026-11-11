"""Streaming one-step evaluation and learning-curve plotting for THOR CEE-US."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class _MetricAccumulator:
    def __init__(self):
        self.count = 0
        self.sums = {}
        self.parent_tp = 0
        self.parent_positive_count = 0
        self.changed_error = 0.0
        self.changed_count = 0
        self.zero_change_error = 0.0

    def add(self, encoder, mean, truth, initial, metrics):
        count = len(mean)
        if not count:
            return
        self.count += count
        for name, value in metrics.items():
            if name not in ('count', 'parent_relation_f1'):
                self.sums[name] = self.sums.get(name, 0.0) + float(value) * count

        pred_obj = mean[:, encoder.agent_dim:encoder.dynamic_dim].reshape(
            count, encoder.n, encoder.object_dyn_dim)
        true_obj = truth[:, encoder.agent_dim:encoder.dynamic_dim].reshape(pred_obj.shape)
        pred_parent = pred_obj[..., -encoder.n:] >= .5
        true_parent = true_obj[..., -encoder.n:] >= .5
        self.parent_tp += int((pred_parent & true_parent).sum())
        self.parent_positive_count += int(pred_parent.sum() + true_parent.sum())

        physical = np.r_[np.arange(8), np.arange(encoder.agent_dim, encoder.dynamic_dim)]
        changed = np.abs(truth[:, physical] - initial[:, physical]) > 1e-6
        self.changed_error += float(np.abs(mean[:, physical] - truth[:, physical])[changed].sum())
        self.zero_change_error += float(np.abs(initial[:, physical] - truth[:, physical])[changed].sum())
        self.changed_count += int(changed.sum())

    def result(self):
        if not self.count:
            return {'count': 0, 'status': 'no_samples'}
        report = {'count': self.count,
                  **{name: value / self.count for name, value in self.sums.items()},
                  'parent_relation_f1': (2 * self.parent_tp / self.parent_positive_count
                                          if self.parent_positive_count else None),
                  'changed_coordinates_mae': (self.changed_error / self.changed_count
                                              if self.changed_count else None),
                  'zero_change_changed_coordinates_mae': (self.zero_change_error / self.changed_count
                                                         if self.changed_count else None)}
        return report


def evaluate_one_step(backend, encoder, data, *, batch_size=4):
    """Evaluate recorded transitions in batches without fitting or changing normalizers.

    Coordinate MAEs use raw encoded state units, which mix metres, radians-as-
    sine/cosine components, and binary fields. The physical-change mask excludes
    the last-action-success bookkeeping coordinate at index 8.
    """
    from exploration.cee_us.cee_us_thor import _metrics, interaction_summary
    from exploration.cee_us.thor_state import ACTIONS

    states, truth, actions = data['states'], data['next_states'], data['actions']
    if not len(states) or any(len(data[key]) != len(states)
                              for key in ('next_states', 'actions', 'types', 'successes')):
        raise ValueError('Evaluation data must contain aligned nonempty transitions')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')

    groups = {
        'success': np.asarray(data['successes'], dtype=bool),
        'failure': ~np.asarray(data['successes'], dtype=bool),
        'navigation': np.asarray(data['types']) < 8,
        'interaction': np.asarray(data['types']) >= 8,
    }
    physical = np.r_[np.arange(8), np.arange(encoder.agent_dim, encoder.dynamic_dim)]
    changed = np.any(np.abs(truth[:, physical] - states[:, physical]) > 1e-6, axis=1)
    groups.update(changed=changed, unchanged=~changed,
                  changed_interaction=changed & (np.asarray(data['types']) >= 8))
    accumulators = {name: _MetricAccumulator() for name in ('conditional_one_step', *groups)}
    per_action = {name: _MetricAccumulator() for name in ACTIONS}
    example_predictions = []
    action_types = np.asarray(data['types'])

    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        batch_states, batch_truth = states[start:stop], truth[start:stop]
        predictions = backend.predict(batch_states, actions[start:stop])
        mean = predictions.mean(axis=0)
        for index in range(start, min(stop, 16)):
            example_predictions.append(mean[index - start].copy())

        metrics = _metrics(encoder, mean, batch_truth, batch_states)
        accumulators['conditional_one_step'].add(encoder, mean, batch_truth, batch_states, metrics)

        batch_types = action_types[start:stop]
        for name, mask in groups.items():
            selected = mask[start:stop]
            if selected.any():
                selected_metrics = _metrics(encoder, mean[selected], batch_truth[selected], batch_states[selected])
                accumulators[name].add(encoder, mean[selected], batch_truth[selected],
                                       batch_states[selected], selected_metrics)
        for action_index, name in enumerate(ACTIONS):
            selected = batch_types == action_index
            if selected.any():
                selected_metrics = _metrics(encoder, mean[selected], batch_truth[selected], batch_states[selected])
                per_action[name].add(encoder, mean[selected], batch_truth[selected],
                                     batch_states[selected], selected_metrics)
        if stop % 1000 == 0 or stop == len(states):
            print(f'One-step evaluation: {stop}/{len(states)} transitions', flush=True)

    success_count = int(np.asarray(data['successes'], dtype=bool).sum())
    count = len(states)
    result = {
        'conditional_one_step': accumulators['conditional_one_step'].result(),
        'groups': {name: accumulators[name].result() for name in groups},
        'per_action': {name: accumulator.result() for name, accumulator in per_action.items()},
        'success_majority_baseline': max(success_count, count - success_count) / count,
    }
    result['interaction_validation'] = (
        'changed_interactions_present' if result['groups']['changed_interaction']['count']
        else 'cannot_validate_without_changed_interactions')
    result['coverage'] = interaction_summary(data)
    examples = []
    for index in range(min(count, 16)):
        predicted = example_predictions[index]
        actual, initial = truth[index], states[index]
        examples.append({
            'action': ACTIONS[int(data['types'][index])], 'target': str(data['targets'][index]),
            'actual_success': bool(data['successes'][index]),
            'predicted_success_value': float(predicted[8]),
            'actual_agent_position': actual[:3].tolist(),
            'predicted_agent_position': predicted[:3].tolist(),
            'actual_held': [encoder.ids[j] for j in range(encoder.n)
                            if actual[encoder.agent_dim + j * encoder.object_dyn_dim + 11] >= .5],
            'predicted_held': [encoder.ids[j] for j in range(encoder.n)
                               if predicted[encoder.agent_dim + j * encoder.object_dyn_dim + 11] >= .5],
        })
    result['examples'] = examples
    return result


evaluate_learning_curve = evaluate_one_step


def render_learning_curve(history, output):
    """Write a four-panel PNG from history entries containing evaluation reports."""
    import matplotlib.pyplot as plt

    rows = [row for row in history if row.get('evaluation')]
    if not rows:
        raise ValueError('History must contain at least one evaluation report')
    x = [row['training_attempts'] for row in rows]
    reports = [row['evaluation'] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    def series(path, metric):
        values = []
        for report in reports:
            current = report
            for key in path:
                current = current.get(key, {}) if isinstance(current, dict) else {}
            value = current.get(metric) if isinstance(current, dict) else None
            values.append(np.nan if value is None else value)
        return values

    ax = axes[0, 0]
    ax.plot(x, series(('conditional_one_step',), 'dynamic_mae'), marker='o', label='Model')
    ax.plot(x, series(('conditional_one_step',), 'zero_change_dynamic_mae'), marker='o', label='Zero-change')
    ax.set(title='Dynamic-state MAE', xlabel='Training attempts', ylabel='MAE (mixed state units)')
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, series(('conditional_one_step',), 'changed_coordinates_mae'), marker='o', label='Model')
    ax.plot(x, series(('conditional_one_step',), 'zero_change_changed_coordinates_mae'), marker='o', label='Zero-change')
    ax.set(title='Changed-coordinate MAE', xlabel='Training attempts', ylabel='MAE (mixed state units)')
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, series(('conditional_one_step',), 'success_accuracy'), marker='o', label='Model accuracy')
    baseline = [report.get('success_majority_baseline', np.nan) for report in reports]
    if any(np.isfinite(value) for value in baseline):
        ax.plot(x, baseline, marker='o', label='Majority baseline')
    ax.set(title='Action-success prediction', xlabel='Training attempts', ylabel='Accuracy', ylim=(0, 1))
    ax.legend()

    ax = axes[1, 1]
    ax.plot(x, series(('groups', 'success'), 'dynamic_mae'), marker='o', label='Successful actions')
    ax.plot(x, series(('groups', 'failure'), 'dynamic_mae'), marker='o', label='Failed actions')
    ax.set(title='Dynamic-state MAE by outcome', xlabel='Training attempts', ylabel='MAE (mixed state units)')
    ax.legend()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', required=True, help='JSON file containing learning-curve history')
    parser.add_argument('--output', required=True, help='Output PNG path')
    args = parser.parse_args(argv)
    render_learning_curve(json.loads(Path(args.history).read_text()), args.output)


if __name__ == '__main__':
    main()
