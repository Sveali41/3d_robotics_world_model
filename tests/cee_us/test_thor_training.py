"""Checkpoint continuation and evaluation protocol tests on a small object graph."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exploration.cee_us.cee_us_thor import (
    append_replay, check_scene_contract, collect, evaluate, load_dataset, make_backend, read_checkpoint,
    restore_checkpoint, save_checkpoint, save_dataset,
)
from exploration.cee_us.thor_cem import ThorCEM
from exploration.cee_us.thor_state import ACTIONS, ThorState
from test_thor_world_model import metadata


def settings():
    return SimpleNamespace(hidden_dim=8, device='cpu', seed=0, episode_length=3,
                           horizon=2, candidates=4, iterations=1, elites=2,
                           cem_chunk_size=1, eval_batch_size=4)


def replay(encoder, initial):
    obs = encoder.encode(initial)
    state = np.stack([obs] * 6)
    types = np.asarray([0, 1, 8, 9, 10, 11])
    actions = encoder.action_encoding(torch.from_numpy(state), torch.from_numpy(types))[0].numpy()
    return dict(states=state, actions=actions, next_states=state.copy(), types=types,
                successes=np.ones(6, bool), targets=np.asarray(['', '', 'a', 'a', '', '']),
                unity_calls=np.ones(6, bool), episodes=np.zeros(6, int), errors=np.asarray(['']*6))


def test_checkpoint_reproduces_model_and_cem_rng(tmp_path):
    initial = metadata(); encoder = ThorState(initial); args = settings()
    data = replay(encoder, initial); backend = make_backend(encoder, args)
    backend.train(data, 1)
    planner = ThorCEM(backend, encoder, horizon=2, candidates=4, iterations=1, elites=2)
    planner.plan(data['states'][0])
    rng = np.random.default_rng(2)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, backend, encoder, initial, data, rng, args, planner=planner, rounds_completed=3)
    expected = backend.predict(data['states'][:1], data['actions'][:1])
    expected_plan = planner.plan(data['states'][0])
    expected_random = rng.integers(10000)
    restored = read_checkpoint(path)
    loaded, rows, loaded_rng = restore_checkpoint(restored, encoder, args)
    other = ThorCEM(loaded, encoder, horizon=2, candidates=4, iterations=1, elites=2)
    other.rng.bit_generator.state = restored['cem_rng']
    np.testing.assert_array_equal(loaded.predict(data['states'][:1], data['actions'][:1]), expected)
    assert other.plan(data['states'][0]) == expected_plan
    assert loaded_rng.integers(10000) == expected_random
    assert restored['rounds_completed'] == 3 and restored['attempts'] == 6
    assert restored['planning']['cem_chunk_size'] == 1
    assert len(append_replay(rows, data)['states']) == 12
    assert append_replay(rows, data)['episodes'][-1] == 1
    bad = ThorState(metadata()); bad.static[0] += 1
    with pytest.raises(ValueError, match='contract'):
        restore_checkpoint(restored, bad, args)


def test_checkpoint_restores_saved_mixed_prediction_mode_without_mutating_args(tmp_path):
    from exploration.cee_us.thor_mixed_model import ThorMixedForwardModel
    initial = metadata(); encoder = ThorState(initial); args = settings()
    args.prediction_mode = 'mixed'
    backend = make_backend(encoder, args)
    data = replay(encoder, initial)
    path = tmp_path / 'mixed.pt'
    save_checkpoint(path, backend, encoder, initial, data, np.random.default_rng(0), args)

    restore_args = settings()
    loaded, _, _ = restore_checkpoint(read_checkpoint(path), encoder, restore_args)
    assert isinstance(loaded.model, ThorMixedForwardModel)
    assert not hasattr(restore_args, 'prediction_mode')


def test_checkpoint_restores_mixed_relations_and_rejects_mode_mismatch(tmp_path):
    from exploration.cee_us.thor_relation_model import ThorRelationForwardModel
    initial = metadata(); encoder = ThorState(initial); args = settings()
    args.prediction_mode = 'mixed-relations'
    backend = make_backend(encoder, args)
    data = replay(encoder, initial)
    path = tmp_path / 'mixed-relations.pt'
    save_checkpoint(path, backend, encoder, initial, data, np.random.default_rng(0), args)

    restore_args = settings()
    loaded, _, _ = restore_checkpoint(read_checkpoint(path), encoder, restore_args)
    assert isinstance(loaded.model, ThorRelationForwardModel)
    assert not hasattr(restore_args, 'prediction_mode')
    mismatch = settings(); mismatch.prediction_mode = 'mixed'
    with pytest.raises(ValueError, match='prediction mode'):
        restore_checkpoint(read_checkpoint(path), encoder, mismatch)


def test_checkpoint_without_prediction_mode_uses_mse(tmp_path):
    initial = metadata(); encoder = ThorState(initial); args = settings()
    backend = make_backend(encoder, args)
    data = replay(encoder, initial)
    path = tmp_path / 'legacy.pt'
    save_checkpoint(path, backend, encoder, initial, data, np.random.default_rng(0), args)
    payload = read_checkpoint(path)
    payload.pop('prediction_mode')

    loaded, _, _ = restore_checkpoint(payload, encoder, settings())
    from exploration.cee_us.official.mbrl.models.gnn_ensemble import GNNForwardEnsembleModel
    assert isinstance(loaded.model, GNNForwardEnsembleModel)


def test_checkpoint_rejects_explicit_prediction_mode_mismatch(tmp_path):
    initial = metadata(); encoder = ThorState(initial); args = settings()
    args.prediction_mode = 'mixed'
    backend = make_backend(encoder, args)
    data = replay(encoder, initial)
    path = tmp_path / 'mixed.pt'
    save_checkpoint(path, backend, encoder, initial, data, np.random.default_rng(0), args)
    mismatch = settings(); mismatch.prediction_mode = 'mse'
    with pytest.raises(ValueError, match='prediction mode'):
        restore_checkpoint(read_checkpoint(path), encoder, mismatch)


def test_evaluation_is_offline_frozen_and_stratified(tmp_path):
    initial = metadata(); encoder = ThorState(initial); args = settings()
    data = replay(encoder, initial); backend = make_backend(encoder, args)
    backend.train(data, 1)
    path = tmp_path / 'evaluation.npz'
    save_dataset(path, encoder, initial, data)
    loaded_encoder, _, loaded_data = load_dataset(path)
    norms = {name: norm.copy() for name, norm in [(n, getattr(backend.model, n).mean.cpu().numpy())
                                                for n in backend.normalizer_names]}
    report = evaluate(backend, loaded_encoder, loaded_data)
    assert [report['rollouts'][str(h)]['count'] for h in (1, 3, 5)] == [6, 4, 2]
    assert report['groups']['changed_interaction']['count'] == 0
    assert report['interaction_validation'] == 'cannot_validate_without_changed_interactions'
    for name, expected in norms.items():
        np.testing.assert_array_equal(getattr(backend.model, name).mean.cpu().numpy(), expected)


def test_no_target_attempt_stays_contiguous_in_collector():
    initial = metadata()
    for obj in initial['objects']:
        obj['visible'] = False
    encoder = ThorState(initial)
    class Env:
        def reset(self):
            return type('Event', (), {'metadata': initial})()
        def step(self, *args, **kwargs):
            raise AssertionError('No target must not call Unity')
    class Rng:
        def choice(self, *args, **kwargs):
            return ACTIONS.index('Pickup')
    data, _ = collect(Env(), encoder, 3, Rng())
    assert not data['unity_calls'].any()
    np.testing.assert_array_equal(data['states'][1:], data['next_states'][:-1])
    assert data['states'][1, 8] == 0


def test_canonical_static_sizes_survive_initial_physics_variation():
    initial = metadata()
    canonical = ThorState(initial)
    initial['objects'][0]['axisAlignedBoundingBox']['size']['y'] = .8
    check_scene_contract(canonical, ThorState(initial))
    initial['objects'][0]['mass'] = 2.
    with pytest.raises(ValueError, match='mass'):
        check_scene_contract(canonical, ThorState(initial))


@pytest.mark.parametrize('exploration', ['curiosity', 'random'])
def test_periodic_loop_keeps_validation_out_of_replay(tmp_path, monkeypatch, exploration):
    import json
    from exploration.cee_us import cee_us_thor as runner
    initial = metadata(); encoder = ThorState(initial)
    training = replay(encoder, initial)
    validation = replay(encoder, initial)
    # A distinct recorded agent position makes accidental validation ingestion visible.
    validation['states'][:, 0] = 99
    validation['next_states'][:, 0] = 99
    frozen = tmp_path / 'evaluation.npz'
    save_dataset(frozen, encoder, initial, validation)
    class Env:
        resets = 0
        decision = 0
        def __init__(self, **kwargs):
            assert kwargs['metadata_only'] is True
            self.controller = type('Controller', (), {
                'last_event': type('Event', (), {'metadata': initial})()})()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(runner, 'ThorEnv', Env)
    collected_planners = []
    def collect(*args, **kwargs):
        collected_planners.append(kwargs.get('planner'))
        return training, {}
    monkeypatch.setattr(runner, 'collect', collect)
    monkeypatch.setattr(runner, 'render_comparison', lambda *a: None)
    planner_options = {}
    class Planner:
        def __init__(self, *args, **kwargs):
            assert exploration == 'curiosity', 'Random baseline must not instantiate CEM'
            planner_options.update(kwargs)
            self.rng = np.random.default_rng(kwargs['seed'])
    monkeypatch.setattr(runner, 'ThorCEM', Planner)
    from exploration.cee_us import thor_learning_curve
    evaluation_batches = []
    def evaluate_one_step(*args, batch_size):
        evaluation_batches.append(batch_size)
        return {'conditional_one_step': {'count': len(args[2]['states'])}}
    monkeypatch.setattr(thor_learning_curve, 'evaluate_one_step', evaluate_one_step)
    output = tmp_path / 'run'
    runner.main(['train', '--exploration', exploration, '--output', str(output), '--eval-data', str(frozen),
                 '--bootstrap-steps', '6', '--rounds', '2', '--steps-per-round', '6',
                 '--hidden-dim', '8', '--eval-every', '1', '--metadata-only',
                 '--cem-chunk-size', '2', '--eval-batch-size', '3'])
    history = json.loads((output / f'{exploration}_learning_curve.json').read_text())
    assert [point['training_attempts'] for point in history] == [6, 12, 18]
    assert [point['round'] for point in history] == [0, 1, 2]
    assert all(point['evaluation']['conditional_one_step']['count'] == 6 for point in history)
    for round_number in (1, 2):
        saved = read_checkpoint(output / f'{exploration}_round_{round_number:04d}.pt')
        assert not (saved['replay']['states'][:, 0] == 99).any()
        assert saved['planning']['cem_chunk_size'] == 2
        assert saved['exploration'] == exploration
        assert next(iter(saved['backend']['model'].values())).shape[0] == 5
    if exploration == 'curiosity':
        assert planner_options['chunk_size'] == 2
        assert collected_planners[0] is None and all(p is not None for p in collected_planners[1:])
    else:
        assert all(p is None for p in collected_planners)
        assert not (output / 'curiosity.pt').exists()
    assert evaluation_batches == [3, 3, 3]


@pytest.mark.parametrize(('option', 'value'), [('--cem-chunk-size', '0'), ('--eval-batch-size', '0')])
def test_runtime_batch_controls_must_be_positive(tmp_path, option, value):
    from exploration.cee_us import cee_us_thor as runner
    with pytest.raises(SystemExit):
        runner.main(['train', '--output', str(tmp_path / 'run'), option, value])
