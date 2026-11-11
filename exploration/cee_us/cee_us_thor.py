"""FloorPlan1 CEE-US with categorical CEM and nearest-visible-target execution."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from domain.thor.env import PROJECT_ROOT, THOR_COMMIT, ThorEnv
from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.cee_us.thor_cem import ThorCEM
from exploration.cee_us.thor_gns import ThorGNSGraph, graph_config, restore_graph_config
from exploration.cee_us.thor_state import ACTIONS, CAPABILITIES, DYNAMIC_FLAGS, ThorState, execute

SCHEMA = 'thor-ceeus-nearest-v1'
CEEUS_COMMIT = 'b0beaed1e9c3fe03d99aa300faae063dc208941e'
PRIOR = np.asarray([.5 / 8] * 8 + [.5 / 6] * 6)
REPLAY_KEYS = ('states', 'actions', 'next_states', 'types', 'successes', 'targets',
               'unity_calls', 'episodes', 'errors')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def make_backend(encoder, args):
    prediction_mode = getattr(args, 'prediction_mode', None) or 'mse'
    model_options = {}
    if prediction_mode == 'mixed':
        from exploration.cee_us.thor_mixed_model import ThorMixedForwardModel
        model_options['forward_model_class'] = ThorMixedForwardModel
    elif prediction_mode == 'mixed-relations':
        from exploration.cee_us.thor_relation_model import ThorRelationForwardModel
        receptacle_indices = tuple(np.flatnonzero(
            encoder._capabilities[:, CAPABILITIES.index('receptacle')]).tolist())
        model_options['forward_model_class'] = partial(
            ThorRelationForwardModel, receptacle_indices=receptacle_indices)
    elif prediction_mode != 'mse':
        raise ValueError(f'Unsupported prediction mode: {prediction_mode}')
    backend = CEEUSBackend(agent_dim=encoder.agent_dim, object_dyn_dim=encoder.object_dyn_dim,
                        object_stat_dim=encoder.object_stat_dim, num_objects=encoder.n,
                        object_names=encoder.ids, action_dim=len(ACTIONS) + encoder.n,
                        ensemble_size=5, hidden_dim=args.hidden_dim, batch_size=4,
                        **model_options,
                        device=args.device, seed=args.seed, attach_icem=False)
    graph = graph_config(args)
    if graph['mode'] == 'gns':
        backend.model.model = ThorGNSGraph(backend.model.model, backend.model, graph['radius'])
    return backend


def collect(env, encoder, steps, rng, *, planner=None, episode_length=100):
    """Each collection starts at an episode boundary; failures remain transitions."""
    collection_started = time.monotonic()
    execution_seconds = 0.0
    rows = {key: [] for key in REPLAY_KEYS}
    obs = encoder.encode(env.reset().metadata)
    scores, seconds = [], []
    for index in range(steps):
        if index and index % episode_length == 0:
            obs = encoder.encode(env.reset().metadata)
        started = time.monotonic()
        if planner is None:
            action, score = int(rng.choice(len(ACTIONS), p=PRIOR)), None
        else:
            action, score = planner.plan(obs)
        seconds.append(time.monotonic() - started)
        scores.append(score)
        started = time.monotonic()
        successor, encoded, success, target, error, called = execute(env, encoder, obs, action)
        execution_seconds += time.monotonic() - started
        values = (obs, encoded, successor, action, success, target or '', called,
                  index // episode_length, error or '')
        for key, value in zip(REPLAY_KEYS, values):
            rows[key].append(value)
        obs = successor
    result = {key: np.asarray(values) for key, values in rows.items()}
    print(json.dumps({'attempts': steps, 'unity_calls': int(result['unity_calls'].sum()),
                      'successes': int(result['successes'].sum()),
                      'planning_seconds': float(sum(seconds)),
                      'curiosity_scores': scores}), flush=True)
    return result, {'planning_seconds': seconds, 'curiosity_scores': scores,
                    'collection_seconds': time.monotonic() - collection_started,
                    'execution_and_logging_seconds': execution_seconds}


def append_replay(a, b):
    if a is None:
        return b
    b = dict(b)
    b['episodes'] = b['episodes'] + int(a['episodes'].max()) + 1
    return {key: np.concatenate((a[key], b[key])) for key in REPLAY_KEYS}


def save_dataset(path, encoder, initial_metadata, data):
    np.savez_compressed(path, **data, schema=SCHEMA,
                        contract=json.dumps(encoder.contract()),
                        initial_metadata=json.dumps(initial_metadata))


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['schema']) != SCHEMA:
            raise ValueError('Unsupported evaluation dataset schema')
        initial = json.loads(str(archive['initial_metadata']))
        encoder = ThorState(initial)
        if encoder.contract() != json.loads(str(archive['contract'])):
            raise ValueError('Evaluation dataset state/action contract is inconsistent')
        data = {key: archive[key] for key in REPLAY_KEYS}
    if len(data['states']) < 1 or any(len(data[k]) != len(data['states']) for k in REPLAY_KEYS):
        raise ValueError('Evaluation data must contain aligned nonempty transitions')
    return encoder, initial, data


def check_scene_contract(reference, observed):
    """Reuse canonical initial AABB sizes; THOR initial physics can rotate objects."""
    expected, current = reference.contract(), observed.contract()
    expected_static = np.asarray(expected.pop('static')).reshape(reference.n, reference.object_stat_dim)
    current_static = np.asarray(current.pop('static')).reshape(observed.n, observed.object_stat_dim)
    if expected != current or not np.array_equal(expected_static[:, :-3], current_static[:, :-3]):
        raise ValueError('FloorPlan1 identities, types, capabilities or mass differ from the frozen scene contract')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name('thor_state.py'),
             Path(__file__).with_name('thor_learning_curve.py'),
             Path(__file__).with_name('thor_cem.py'), Path(__file__).with_name('thor_validation.py'),
             Path(__file__).with_name('cee_us_backend.py'),
             Path(__file__).with_name('thor_mixed_model.py'),
             Path(__file__).with_name('thor_relation_model.py'),
             Path(__file__).with_name('thor_gns.py'), PROJECT_ROOT / 'domain/thor/env.py']
    return {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in paths}


def save_checkpoint(path, backend, encoder, initial, replay, rng, args, *, planner=None,
                    rounds_completed=0):
    torch.save({'schema': SCHEMA, 'backend': backend.state_dict(),
                'contract': encoder.contract(), 'initial_metadata': initial, 'replay': replay,
                'rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'cem_rng': None if planner is None else planner.rng.bit_generator.state,
                'hidden_dim': args.hidden_dim, 'seed': args.seed,
                'exploration': getattr(args, 'exploration', None) or 'curiosity',
                'prediction_mode': getattr(args, 'prediction_mode', None) or 'mse',
                'graph': graph_config(args),
                'rounds_completed': rounds_completed, 'attempts': len(replay['states']),
                'planning': {k: getattr(args, k) for k in
                             ('horizon', 'candidates', 'iterations', 'elites', 'cem_chunk_size')},
                'episode_length': args.episode_length, 'sources': source_hashes(),
                'boundary': 'next collection resets the scene; no mid-episode Unity restore'}, path)


def read_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['schema'] != SCHEMA:
        raise ValueError('Unsupported THOR checkpoint schema')
    return payload


def restore_checkpoint(payload, encoder, args, *, restore_rng=True):
    if payload['contract'] != encoder.contract():
        raise ValueError('Checkpoint state/action contract differs from this scene or dataset')
    if payload['hidden_dim'] != args.hidden_dim:
        raise ValueError('Checkpoint hidden dimension differs; pass matching --hidden-dim')
    saved_mode = payload.get('prediction_mode', 'mse')
    requested_mode = getattr(args, 'prediction_mode', None)
    if requested_mode is not None and requested_mode != saved_mode:
        raise ValueError(f'Checkpoint prediction mode is {saved_mode!r}, but requested {requested_mode!r}')
    backend_args = deepcopy(args)
    backend_args.prediction_mode = saved_mode
    restore_graph_config(payload, backend_args)
    backend = make_backend(encoder, backend_args)
    backend.load_state_dict(payload['backend'], restore_planner=False)
    rng = np.random.default_rng()
    rng.bit_generator.state = payload['rng']
    if restore_rng:
        torch.set_rng_state(payload['torch_rng'])
        random.setstate(payload['python_rng'])
        np.random.set_state(payload['numpy_rng'])
        if payload['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return backend, deepcopy(payload['replay']), rng


def _metrics(encoder, mean, truth, initial):
    error, baseline = np.abs(mean - truth), np.abs(initial - truth)
    obj = mean[..., encoder.agent_dim:encoder.dynamic_dim].reshape(-1, encoder.n, encoder.object_dyn_dim)
    target = truth[..., encoder.agent_dim:encoder.dynamic_dim].reshape(obj.shape)
    previous = initial[..., encoder.agent_dim:encoder.dynamic_dim].reshape(obj.shape)
    def angle_error(a, b):
        aa = np.arctan2(a[..., 0], a[..., 1])
        bb = np.arctan2(b[..., 0], b[..., 1])
        return float(np.abs(np.arctan2(np.sin(aa - bb), np.cos(aa - bb))).mean())
    result = {'count': len(mean), 'dynamic_mae': float(error[..., :encoder.dynamic_dim].mean()),
              'zero_change_dynamic_mae': float(baseline[..., :encoder.dynamic_dim].mean()),
              'agent_position_mae_m': float(error[..., :3].mean()),
              'zero_change_agent_position_mae_m': float(baseline[..., :3].mean()),
              'object_position_mae_m': float(np.abs(obj[..., :3] - target[..., :3]).mean()),
              'zero_change_object_position_mae_m': float(np.abs(previous[..., :3] - target[..., :3]).mean()),
              'agent_yaw_mae_rad': angle_error(mean[..., 3:5], truth[..., 3:5]),
              'agent_horizon_mae_rad': angle_error(mean[..., 5:7], truth[..., 5:7]),
              'object_rotation_mae_rad': angle_error(obj[..., 3:9].reshape(-1, 3, 2), target[..., 3:9].reshape(-1, 3, 2)),
              'success_accuracy': float(((mean[..., 8] >= .5) == (truth[..., 8] >= .5)).mean()),
              'openness_mae': float(np.abs(obj[..., 21] - target[..., 21]).mean()),
              'parent_relation_accuracy': float(((obj[..., -encoder.n:] >= .5) == (target[..., -encoder.n:] >= .5)).mean())}
    pred_parent, true_parent = obj[..., -encoder.n:] >= .5, target[..., -encoder.n:] >= .5
    tp = int((pred_parent & true_parent).sum())
    denom = int(pred_parent.sum() + true_parent.sum())
    result['parent_relation_f1'] = 2 * tp / denom if denom else None
    for name in ('isPickedUp', 'isOpen', 'isToggled', 'visible'):
        column = 9 + DYNAMIC_FLAGS.index(name)
        result[name + '_accuracy'] = float(((obj[..., column] >= .5) == (target[..., column] >= .5)).mean())
    return result


def evaluate(backend, encoder, data):
    """Offline only: condition on recorded targets, then roll out action types."""
    states, truth, actions = data['states'], data['next_states'], data['actions']
    predictions = np.concatenate([backend.predict(states[i:i+1], actions[i:i+1])
                                  for i in range(len(states))], axis=1)
    mean = predictions.mean(axis=0)
    report = {'conditional_one_step': _metrics(encoder, mean, truth, states)}
    # State-change strata exclude only the bookkeeping last-action-success field.
    physical = np.r_[np.arange(8), np.arange(encoder.agent_dim, encoder.dynamic_dim)]
    changed = np.any(np.abs(truth[:, physical] - states[:, physical]) > 1e-6, axis=1)
    groups = {}
    for label, mask in [('navigation', data['types'] < 8), ('interaction', data['types'] >= 8),
                        ('success', data['successes']), ('failure', ~data['successes']),
                        ('changed', changed), ('unchanged', ~changed),
                        ('no_target', (data['types'] >= 8) & (data['targets'] == '')),
                        ('changed_interaction', changed & (data['types'] >= 8))]:
        groups[label] = _metrics(encoder, mean[mask], truth[mask], states[mask]) if mask.any() else {'count': 0, 'status': 'no_samples'}
    report['groups'] = groups
    disagreement = predictions.std(axis=0, ddof=1).sum(axis=1)
    actual_error = np.abs(mean[:, :encoder.dynamic_dim] - truth[:, :encoder.dynamic_dim]).sum(axis=1)
    report['disagreement_error_correlation'] = (float(np.corrcoef(disagreement, actual_error)[0, 1])
        if len(states) > 1 and disagreement.std() > 0 and actual_error.std() > 0 else None)
    report['coverage'] = interaction_summary(data)
    report['interaction_validation'] = ('changed_interactions_present' if groups['changed_interaction']['count']
                                        else 'cannot_validate_without_changed_interactions')
    report['rollouts'] = {}
    for horizon in (1, 3, 5):
        starts = [i for i in range(len(states) - horizon + 1)
                  if np.all(data['episodes'][i:i+horizon] == data['episodes'][i])]
        if not starts:
            report['rollouts'][str(horizon)] = {'count': 0, 'status': 'insufficient_contiguous_data'}
            continue
        means, expected, initial, target_correct = [], [], [], []
        for index in starts:
            obs = torch.as_tensor(states[index:index+1], device=backend.device).unsqueeze(0).expand(backend.ensemble_size, -1, -1)
            for step in range(horizon):
                action_type = int(data['types'][index+step])
                encoded, resolved = encoder.action_encoding(obs, action_type)
                recorded = data['targets'][index+step]
                expected_target = encoder.ids.index(str(recorded)) if recorded else -1
                if action_type >= 8:
                    target_correct.extend((resolved.cpu().numpy().reshape(-1) == expected_target).tolist())
                obs = backend.model.predict(observations=obs, states=None, actions=encoded)[0]
            means.append(obs.mean(dim=0)[0].cpu().numpy())
            expected.append(truth[index+horizon-1]); initial.append(states[index])
        value = _metrics(encoder, np.asarray(means), np.asarray(expected), np.asarray(initial))
        value['target_resolution_accuracy_per_member'] = float(np.mean(target_correct)) if target_correct else None
        report['rollouts'][str(horizon)] = value
    # State prediction has no image decoder; provide actual vs predicted state values.
    report['examples'] = [{'action': ACTIONS[int(data['types'][i])], 'target': str(data['targets'][i]),
        'actual_success': bool(data['successes'][i]), 'predicted_success_value': float(mean[i, 8]),
        'actual_agent_position': truth[i, :3].tolist(), 'predicted_agent_position': mean[i, :3].tolist(),
        'actual_held': [encoder.ids[j] for j in range(encoder.n) if truth[i, encoder.agent_dim+j*encoder.object_dyn_dim+11] >= .5],
        'predicted_held': [encoder.ids[j] for j in range(encoder.n) if mean[i, encoder.agent_dim+j*encoder.object_dyn_dim+11] >= .5]}
        for i in range(min(len(states), 16))]
    return report


def interaction_summary(data):
    types, successes = data['types'], data['successes']
    targets = data['targets']
    return {'attempts': len(types), 'unity_calls': int(data['unity_calls'].sum()),
            'local_failures': int((~data['unity_calls']).sum()),
            'successes': int(successes.sum()),
            'no_target_failures': int(((types >= 8) & (targets == '')).sum()),
            'unique_targets': len(set(str(x) for x in targets if x)),
            'actions': {name: {'attempts': int((types == i).sum()),
                               'successes': int(successes[types == i].sum())}
                        for i, name in enumerate(ACTIONS)}}


def render_comparison(path, report):
    """Render a state comparison table; the GNN has no predicted RGB decoder."""
    from PIL import Image, ImageDraw, ImageFont
    rows = report['examples']
    picture = Image.new('RGB', (1300, 100 + 45 * len(rows)), '#111827')
    draw = ImageDraw.Draw(picture)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    draw.text((15, 12), 'FloorPlan1: actual vs predicted state (one-step, recorded target)', font=font, fill='white')
    draw.text((15, 48), 'Action / resolved target', font=font, fill='#93c5fd')
    draw.text((680, 48), 'Actual success', font=font, fill='#93c5fd')
    draw.text((850, 48), 'Predicted value', font=font, fill='#93c5fd')
    draw.text((1050, 48), 'Held: actual / predicted', font=font, fill='#93c5fd')
    for i, row in enumerate(rows):
        y = 85 + 45 * i
        draw.text((15, y), f"{row['action']}  {row['target']}", font=font, fill='white')
        draw.text((700, y), str(row['actual_success']), font=font, fill='white')
        draw.text((880, y), f"{row['predicted_success_value']:.3f}", font=font, fill='white')
        draw.text((1100, y), f"{len(row['actual_held'])} / {len(row['predicted_held'])}", font=font, fill='white')
    picture.save(path)


def make_manifest(args):
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'ai2thor', 'gymnasium', 'forwardable', 'Pillow')}
    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
    prediction_mode = getattr(args, 'prediction_mode', None) or 'mse'
    objective = ('numeric MSE on normalized next-state deltas' if prediction_mode == 'mse' else
                 'regression=(mean agent continuous MSE + mean object continuous MSE)/2; classification=mean of six mean losses (standing BCE, success BCE, object flags BCE, parent BCE, liquid CE4, temperature CE3); total=(regression+classification).sum over ensemble members')
    if prediction_mode == 'mixed-relations':
        objective += '; separate relation head predicts next-state multi-label receptacle relations, excluding self targets; relation positive weights are computed from training data with zero-class weight 1'
    decoding = ('numeric next-state prediction' if prediction_mode == 'mse' else
                'continuous fields are denormalized deltas added to current state; binary and categorical fields are absolute next-state logits decoded to probabilities with sigmoid/softmax')
    return {'schema': SCHEMA, 'created': datetime.now(ZoneInfo('Europe/London')).isoformat(),
            'scene': 'FloorPlan1', 'thor_commit': THOR_COMMIT, 'cee_us_commit': CEEUS_COMMIT,
            'git_head': git_head, 'sources': source_hashes(), 'versions': versions,
            'prediction_mode': prediction_mode, 'prediction_objective': objective,
            'prediction_decoding': decoding, 'graph': graph_config(args),
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'sampling_prior': PRIOR.tolist(), 'forceAction': False, 'placeStationary': True,
            'local_failures_count_as_attempts': True,
            'physics_timing': 'one native action; no hidden settling or retries',
            'static_geometry': 'canonical initial AABB dimensions from frozen scene contract; current AABB is not a static feature',
            'restore': 'episode boundaries only; every collection resets the scene',
            'adaptations': ['categorical CEM replaces continuous iCEM',
                            'nearest visible capable target, no task preference',
                            f'fixed FloorPlan1 instance metadata state, {objective}',
                            'shared random bootstrap differs from original zero-bootstrap'],
            'evaluation_sha256': file_hash(args.eval_data) if args.eval_data else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke', 'collect-eval', 'train', 'compare', 'evaluate', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exploration', choices=('curiosity', 'random'), default=None,
                        help='Train policy: defaults to curiosity for new runs, or checkpoint policy when restoring')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--eval-data', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--bootstrap-steps', type=int)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--steps-per-round', type=int)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--episode-length', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--prediction-mode', choices=('mse', 'mixed', 'mixed-relations'), default=None,
                        help='Forward prediction objective; defaults to MSE for new runs and the checkpoint mode when restoring')
    parser.add_argument('--gns', action=argparse.BooleanOptionalAction, default=None,
                        help='GNS-inspired dynamic sparse graph; default full graph, or infer from checkpoint')
    parser.add_argument('--gns-radius', type=float, default=None,
                        help='Estimated bounding-box gap threshold in metres (new GNS runs: 0.5)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--candidates', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--elites', type=int, default=8)
    parser.add_argument('--cem-chunk-size', type=int, default=1,
                        help='Number of CEM candidate sequences scored together')
    parser.add_argument('--eval-batch-size', type=int, default=4,
                        help='Number of frozen transitions predicted together during periodic evaluation')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='Evaluate all frozen transitions after bootstrap and every N rounds; 0 retains final full rollout evaluation')
    parser.add_argument('--metadata-only', action='store_true', help='Run Unity without rendered images')
    args = parser.parse_args(argv)
    if args.exploration is not None and args.stage not in ('train', 'evaluate', 'report'):
        parser.error('--exploration is supported by train/evaluate/report; compare runs both policies')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    if args.eval_every < 0:
        parser.error('--eval-every must be nonnegative')
    if min(args.epochs, args.episode_length, args.hidden_dim, args.horizon,
           args.candidates, args.iterations, args.elites, args.cem_chunk_size,
           args.eval_batch_size) < 1 or args.elites > args.candidates:
        parser.error('Model, training, episode and CEM sizes must be positive; elites <= candidates')
    if args.stage == 'collect-eval' and (args.steps is None or args.steps < 1):
        parser.error('--steps must be positive (random navigation/interaction attempts in addition to curated episodes)')
    if args.stage in ('train', 'compare'):
        if args.eval_data is None:
            parser.error('--eval-data is required; freeze an independent dataset with collect-eval first')
        if any(x is None or x < 1 for x in (args.rounds, args.steps_per_round)):
            parser.error('--rounds and --steps-per-round must be positive')
        if args.checkpoint is None and (args.bootstrap_steps is None or args.bootstrap_steps < 2):
            parser.error('--bootstrap-steps must be at least two for a new training run')
    if args.stage in ('evaluate', 'report') and (args.checkpoint is None or args.eval_data is None):
        parser.error('--checkpoint and --eval-data are required')
    if args.checkpoint is not None:
        checkpoint_metadata = read_checkpoint(args.checkpoint)
        saved_policy = checkpoint_metadata.get('exploration', 'curiosity')
        if args.exploration is not None and args.exploration != saved_policy:
            raise ValueError('Checkpoint exploration policy differs from --exploration')
        args.exploration = saved_policy
        saved_mode = checkpoint_metadata.get('prediction_mode', 'mse')
        if args.prediction_mode is not None and args.prediction_mode != saved_mode:
            raise ValueError(f'Checkpoint prediction mode is {saved_mode!r}, but requested {args.prediction_mode!r}')
        args.prediction_mode = saved_mode
        restore_graph_config(checkpoint_metadata, args)
    elif args.prediction_mode is None:
        args.prediction_mode = 'mse'
    if args.exploration is None:
        args.exploration = 'curiosity'
    graph_config(args)
    if args.stage == 'smoke':
        args.bootstrap_steps, args.rounds, args.steps_per_round = 8, 1, 3
        args.hidden_dim, args.horizon, args.candidates, args.iterations, args.elites = 16, 2, 4, 1, 2
        args.episode_length = 100
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    write_json(args.output / 'manifest.json', make_manifest(args))
    eval_data = None
    if args.eval_data:
        eval_encoder, eval_initial, eval_data = load_dataset(args.eval_data)
    if args.stage in ('evaluate', 'report'):
        # Evaluation uses stored metadata; it does not start Unity or update normalizers.
        payload = read_checkpoint(args.checkpoint)
        backend, _, _ = restore_checkpoint(payload, eval_encoder, args, restore_rng=False)
        report = evaluate(backend, eval_encoder, eval_data)
        write_json(args.output / 'report.json', report)
        render_comparison(args.output / 'state_comparison.png', report)
        print(json.dumps(report['coverage']), flush=True)
        return 0
    from exploration.cee_us.thor_validation import collect_validation
    with ThorEnv(scene='FloorPlan1', seed=args.seed, output_dir=args.output / 'environment',
                 metadata_only=args.metadata_only) as env:
        initial = deepcopy(env.controller.last_event.metadata)
        encoder = ThorState(initial)
        if eval_data is not None:
            check_scene_contract(eval_encoder, encoder)
            encoder, initial = eval_encoder, eval_initial
        write_json(args.output / 'contract.json', encoder.contract())
        if args.stage in ('collect-eval', 'smoke'):
            eval_data = collect_validation(env, encoder, args.seed + 10000,
                                           args.steps if args.stage == 'collect-eval' else 5,
                                           args.output / 'validation')
            save_dataset(args.output / 'evaluation.npz', encoder, initial, eval_data)
            manifest = json.loads((args.output / 'manifest.json').read_text())
            manifest['evaluation_sha256'] = file_hash(args.output / 'evaluation.npz')
            write_json(args.output / 'manifest.json', manifest)
            if args.stage == 'collect-eval':
                write_json(args.output / 'coverage.json', interaction_summary(eval_data))
                return 0
        rng = np.random.default_rng(args.seed)
        resumed = None
        if args.checkpoint:
            resumed = read_checkpoint(args.checkpoint)
            if resumed['sources'] != source_hashes():
                raise ValueError('Resume source fingerprint differs; use original source for an exact continuation')
            if resumed['seed'] != args.seed or resumed['episode_length'] != args.episode_length:
                raise ValueError('Resume seed or episode length differs')
            if resumed['planning'] != {k: getattr(args, k) for k in
                                        ('horizon', 'candidates', 'iterations', 'elites', 'cem_chunk_size')}:
                raise ValueError('Resume CEM settings differ')
            backend, replay, rng = restore_checkpoint(resumed, encoder, args)
        else:
            replay, _ = collect(env, encoder, args.bootstrap_steps, rng, episode_length=args.episode_length)
            backend = make_backend(encoder, args)
            started = time.monotonic()
            fit = backend.train(replay, args.epochs)
            if backend.device.type == 'cuda':
                torch.cuda.synchronize(backend.device)
            fit['seconds'] = time.monotonic() - started
            write_json(args.output / 'bootstrap_fit.json', fit)
            print(json.dumps(fit), flush=True)
        save_checkpoint(args.output / 'bootstrap.pt', backend, encoder, initial, replay, rng, args,
                        rounds_completed=resumed['rounds_completed'] if resumed else 0)
        initial_checkpoint = read_checkpoint(args.output / 'bootstrap.pt')
        variants = ('random', 'curiosity') if args.stage == 'compare' else (args.exploration,)
        summary = {}
        for variant in variants:
            args.exploration = variant
            backend, replay, rng = restore_checkpoint(initial_checkpoint, encoder, args)
            planner = None if variant == 'random' else ThorCEM(backend, encoder, horizon=args.horizon,
                        candidates=args.candidates, iterations=args.iterations, elites=args.elites,
                        chunk_size=args.cem_chunk_size, seed=args.seed)
            if planner is not None and resumed and resumed['cem_rng'] is not None:
                planner.rng.bit_generator.state = resumed['cem_rng']
            start_round = resumed['rounds_completed'] if resumed else 0
            diagnostics = []
            history = []
            def assess(round_number):
                from exploration.cee_us.thor_learning_curve import evaluate_one_step
                started = time.monotonic()
                assessment = evaluate_one_step(backend, encoder, eval_data,
                                               batch_size=args.eval_batch_size)
                history.append({'round': round_number, 'training_attempts': len(replay['states']),
                                'evaluation_seconds': time.monotonic() - started,
                                'evaluation': assessment})
                write_json(args.output / f'{variant}_learning_curve.json', history)
                print(json.dumps({'variant': variant, 'round': round_number,
                                  'training_attempts': len(replay['states']),
                                  'evaluation_seconds': history[-1]['evaluation_seconds'],
                                  'validation': assessment['conditional_one_step']}), flush=True)
                return assessment
            if args.eval_every:
                assess(start_round)
            for index in range(args.rounds):
                round_started = time.monotonic()
                batch, timing = collect(env, encoder, args.steps_per_round, rng, planner=planner,
                                        episode_length=args.episode_length)
                replay = append_replay(replay, batch)
                started = time.monotonic()
                fit = backend.train(replay, args.epochs)
                if backend.device.type == 'cuda':
                    torch.cuda.synchronize(backend.device)
                fit['seconds'] = time.monotonic() - started
                diagnostics.append({'round': start_round + index + 1, 'fit': fit,
                                    'coverage': interaction_summary(batch), **timing})
                started = time.monotonic()
                save_checkpoint(args.output / f'{variant}.pt', backend, encoder, initial, replay, rng, args,
                                planner=planner, rounds_completed=start_round+index+1)
                if args.eval_every and ((index + 1) % args.eval_every == 0 or index + 1 == args.rounds):
                    save_checkpoint(args.output / f'{variant}_round_{start_round+index+1:04d}.pt',
                                    backend, encoder, initial, replay, rng, args,
                                    planner=planner, rounds_completed=start_round+index+1)
                diagnostics[-1]['checkpoint_seconds'] = time.monotonic() - started
                write_json(args.output / f'{variant}_rounds.json', diagnostics)
                if args.eval_every and ((index + 1) % args.eval_every == 0 or index + 1 == args.rounds):
                    assess(start_round + index + 1)
                diagnostics[-1]['round_seconds'] = time.monotonic() - round_started
                write_json(args.output / f'{variant}_rounds.json', diagnostics)
            report = history[-1]['evaluation'] if args.eval_every else evaluate(backend, encoder, eval_data)
            report['training_coverage'] = interaction_summary(replay)
            summary[variant] = report
            write_json(args.output / f'{variant}_evaluation.json', report)
            render_comparison(args.output / f'{variant}_state_comparison.png', report)
            if args.stage == 'smoke':
                if report['groups']['changed_interaction']['count'] < 1:
                    raise RuntimeError('Smoke lacks state-changing evaluation interactions')
                actual = backend.predict(eval_data['states'][:1], eval_data['actions'][:1])
                loaded, _, _ = restore_checkpoint(read_checkpoint(args.output / 'curiosity.pt'), encoder, args)
                np.testing.assert_array_equal(actual, loaded.predict(eval_data['states'][:1], eval_data['actions'][:1]))
                write_json(args.output / 'smoke_checks.json', {'finite_training': True,
                           'checkpoint_prediction_identical': True, 'evaluation': report['coverage'],
                           'official_model': type(backend.model).__name__,
                           'planning': 'categorical CEM over action types; member-specific nearest targets'})
        write_json(args.output / 'report.json', summary)
        write_json(args.output / 'simulator_counts.json', {'resets': env.resets,
                   'recorded_unity_action_calls': env.decision,
                   'reset_setup': 'each reset includes seed and map-camera setup queries',
                   'validation_extra_setup': str(args.output / 'validation/setup.json') if args.stage == 'smoke' else None})
        print(f'Completed {args.stage}; results: {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
