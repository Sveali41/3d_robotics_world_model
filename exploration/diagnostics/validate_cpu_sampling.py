"""Bounded native acceptance of production CPU multi-stream collection.

Runs 1,616 controls and a single smoke fit. The scripted target is diagnostic-only;
random and curiosity probes use unmodified production policies.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

from exploration.diagnostics.benchmark_grasp_optimizations import compare_tree, file_digest
from exploration.diagnostics.profile_cup_grasp import CONFIG, ROOT, SOURCES, validate_config


def _scripted_shard(job):
    # Spawn imports this module afresh, so this is the unpatched production function.
    from domain.robocasa.primitives import PrimitiveAction
    from exploration.raps import cpu_sampling
    from exploration.raps.raps_gnn_actor import SkillCollector
    original_shard = cpu_sampling._run_shard
    original_propose = SkillCollector._propose_action
    def propose(self, policy):
        return self.schema.encode(PrimitiveAction('grasp', dict(x=.410, y=0., z=.732, orientation_id=2)))
    SkillCollector._propose_action = propose
    try:
        return original_shard(job)
    finally:
        SkillCollector._propose_action = original_propose


@contextmanager
def scripted_collection():
    from exploration.raps import cpu_sampling
    original = cpu_sampling._run_shard
    cpu_sampling._run_shard = _scripted_shard
    try:
        yield
    finally:
        cpu_sampling._run_shard = original


def _comparable(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _comparable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_comparable(v) for v in value)
    return value


def require_equal(left, right, label):
    differences = compare_tree(_comparable(left), _comparable(right), label, exact=True)
    if differences:
        raise AssertionError(f'{label} differs: {differences[:20]} ({len(differences)} differences)')


def metrics(ticks, skills):
    statuses = Counter(row['status'] for row in skills)
    completed = Counter(row['status'] for row in skills if row['completed'])
    rejections = 0
    for skill in skills:
        rejections += sum('collision' in str(attempt.get('status', '')).lower()
                          for attempt in skill.get('motion_planning', []))
    return dict(controls=len(ticks), skill_status=dict(statuses), completed_status=dict(completed),
                successes=completed.get('success', 0),
                completed_failures=sum(count for status, count in completed.items() if status != 'success'),
                budget_truncated=statuses.get('budget_truncated', 0), planning_collision_rejections=rejections,
                trainable=sum(bool(row['trainable']) for row in skills))


def _finite_rows(ticks, skills):
    for row in ticks + skills:
        for key in ('state', 'next_state', 'action'):
            if not np.isfinite(np.asarray(row[key], dtype=float)).all():
                raise AssertionError(f'Nonfinite {key} in collection')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    import torch
    from omegaconf import OmegaConf
    from exploration.cee_us.cee_us_backend import CEEUSBackend
    from exploration.cee_us.cee_us_cabinet import make_env, rng_state, restore_rng
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from exploration.raps.cpu_sampling import CpuRoundCollector, sampling_contract
    from exploration.raps.raps_gnn_actor import (DEFAULTS, RAPSImaginationActor, fit, load, save, skill_arrays)
    from scripts.launch_raps_comparison import arm_configs

    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  controls_requested=1616, seed=0, collection_seed=0,
                  scripted_target=dict(cell=4, x=.410, y=0., z=.732, orientation_id=2),
                  python=sys.version, platform=platform.platform(), numpy=np.__version__, torch=torch.__version__)
    started = time.perf_counter()
    env = None
    try:
        config = OmegaConf.load(CONFIG)
        validate_config(config)
        config.device = 'cpu'
        config.raps_actor = OmegaConf.merge(DEFAULTS, config.raps_actor)
        config.cpu_sampling = dict(workers=2, streams=2)
        report['config'] = OmegaConf.to_container(config, resolve=True)
        source_paths = (*SOURCES, 'exploration/raps/cpu_sampling.py',
                        'exploration/diagnostics/validate_cpu_sampling.py', 'scripts/launch_raps_comparison.py')
        report['source_sha256'] = {path: file_digest(ROOT / path) for path in source_paths}
        arms = arm_configs(report['config'], include_heuristic_only=True)
        if len(arms) != 4 or any(arm['cpu_sampling'] != {'workers': 2, 'streams': 2} for arm in arms.values()):
            raise AssertionError('Four-arm sampling settings differ')
        report['arm_sampling'] = {name: arm['cpu_sampling'] for name, arm in arms.items()}
        torch.set_num_threads(1)
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        backend = CEEUSBackend(device='cpu', seed=0, action_dim=31, agent_dim=33,
            num_objects=1, object_names=('cabinet_item_16',), execution='reference',
            object_heuristic=OmegaConf.to_container(config.object_heuristic, resolve=True),
            runtime_optimizations=config.runtime_optimizations)
        actor = RAPSImaginationActor(backend, OmegaConf.to_container(config.raps_actor, resolve=True))
        env, _ = make_env(episode_steps=config.episode_steps,
            centered_initial_tcp=config.centered_initial_tcp, cabinet_door=config.cabinet_door,
            cabinet_item_layout=config.cabinet_item_layout, cabinet_grid=True,
            cabinet_grid_layout=config.raps_actor.grid_layout, action_mode=config.action_mode,
            low_level_episode_steps=config.episode_steps)
        env.reset(seed=0)
        encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
        snapshot = backend._move_tensors(dict(backend=backend.state_dict(), actor=actor.state_dict(), rng=rng_state()), 'cpu')
        results = {}
        report['modes'] = {}
        for workers in (1, 2):
            backend.load_state_dict(snapshot['backend']); actor.load_state_dict(snapshot['actor']); restore_rng(snapshot['rng'])
            collector = CpuRoundCollector(config, actor, output / f'workers_{workers}', workers=workers)
            with scripted_collection():
                ticks, skills = collector.collect(800, 'random', seed=0, stage='scripted')
            require_equal(snapshot, dict(backend=backend.state_dict(), actor=actor.state_dict(), rng=rng_state()),
                          f'parent state after workers={workers}')
            if len(ticks) != 800 or Counter(row['stream_id'] for row in ticks) != {0: 400, 1: 400}:
                raise AssertionError('Scripted collection did not produce 400 controls per stream')
            _finite_rows(ticks, skills)
            results[workers] = (ticks, skills)
            report['modes'][str(workers)] = dict(**metrics(ticks, skills), timings=collector.profile_timings)
        # Motion planner diagnostics include wall-clock durations; recorded physical
        # state, controller commands, actions, rewards and IDs are compared exactly.
        for index, label, excluded in ((0, 'canonical physical ticks', {'motion_execution'}),
                                        (1, 'canonical skills', {'motion_planning'})):
            selected = [[{key: value for key, value in row.items() if key not in excluded}
                         for row in results[workers][index]] for workers in (1, 2)]
            require_equal(selected[0], selected[1], label)
        require_equal(report['modes']['1'] | {'timings': None}, report['modes']['2'] | {'timings': None}, 'task metrics')
        report['serial_parallel_exact'] = True
        ticks, skills = results[1]
        if sum(row['trainable'] and row['status'] == 'success' for row in skills) < 2:
            raise AssertionError('Need two genuine successful completed endpoints for fit')
        events = []
        original_model_train, original_actor_train = backend.train, actor.train
        def model_train(*args, **kwargs):
            events.append('world_model'); return original_model_train(*args, **kwargs)
        def actor_train(*args, **kwargs):
            events.append('actor'); return original_actor_train(*args, **kwargs)
        backend.train, actor.train = model_train, actor_train
        try:
            fit_result = fit(backend, actor, skills, config, output, 'acceptance', smoke=True)
        finally:
            backend.train, actor.train = original_model_train, original_actor_train
        if events != ['world_model', 'actor']:
            raise AssertionError(f'Unexpected training order: {events}')
        report['fit'] = dict(order=events, world_model_epochs=1, actor_updates=2, result=fit_result)
        replay = skill_arrays(skills)
        expected_prediction = backend.predict(replay['states'][:1], replay['actions'][:1])
        saved_rng = rng_state()
        expected_action = actor.action(replay['states'][0])
        restore_rng(saved_rng)
        checkpoint = output / 'checkpoint.pt'
        save(checkpoint, backend, actor, encoder, config, ticks, skills)
        try:
            load(checkpoint, backend, actor, encoder, sampling=None)
        except ValueError as error:
            if 'sampling contract' not in str(error):
                raise
        else:
            raise AssertionError('Incompatible sampling checkpoint was accepted')
        load(checkpoint, backend, actor, encoder, sampling=sampling_contract(config))
        np.testing.assert_array_equal(expected_prediction, backend.predict(replay['states'][:1], replay['actions'][:1]))
        np.testing.assert_array_equal(expected_action, actor.action(replay['states'][0]))
        report['checkpoint'] = dict(path=str(checkpoint), restore_prediction_action_exact=True,
                                    incompatible_sampling_rejected=True)
        report['policy_probes'] = {}
        for policy in ('random', 'curiosity'):
            collector = CpuRoundCollector(config, actor, output / f'policy_{policy}')
            probe_ticks, probe_skills = collector.collect(8, policy, seed=0, stage=policy)
            if len(probe_ticks) != 8 or Counter(row['stream_id'] for row in probe_ticks) != {0: 4, 1: 4} or not probe_skills:
                raise AssertionError(f'{policy} probe did not produce its real controls and skill logs')
            _finite_rows(probe_ticks, probe_skills)
            report['policy_probes'][policy] = metrics(probe_ticks, probe_skills)
        report['controls_observed'] = sum(mode['controls'] for mode in report['modes'].values()) + sum(
            probe['controls'] for probe in report['policy_probes'].values())
        report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if env is not None:
            env.close()
        report['elapsed_s'] = time.perf_counter() - started
        (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
