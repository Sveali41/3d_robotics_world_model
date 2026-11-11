"""Collect deterministic cabinet grasp validation cases without training."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
from omegaconf import OmegaConf
import torch

from exploration.cee_us.cee_us_cabinet import make_env, replay_arrays
from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
from exploration.diagnostics.calibrate_wm_validation import CalibrationProvider, checkpoint_contract
from exploration.diagnostics.diagnose_cabinet import write_json
from exploration.raps.raps_gnn_actor import SkillCollector

ROOT = Path(__file__).resolve().parents[2]
BUDGET = 800
# Execution sources, rather than unrelated working-tree edits, define resume identity.
SOURCES = ('exploration/diagnostics/collect_cabinet_grasps.py',
           'exploration/diagnostics/calibrate_wm_validation.py',
           'exploration/raps/raps_gnn_actor.py', 'exploration/cee_us/cee_us_cabinet.py',
           'exploration/cee_us/cee_us_state.py', 'domain/robocasa/primitives.py',
           'domain/robocasa/skill_execution.py', 'domain/robocasa/primitive_wrapper.py',
           'exploration/contact_reward.py', 'domain/wall_cabinet_env.py',
           'configs/exploration.yaml', 'main.py')


def cases(seed=1000000):
    result = []
    for i, (obj, dx, dy, dz, lift) in enumerate(itertools.product(
            ('cup', 'bowl'), (-.01, 0., .01), (-.015, 0., .015),
            (-.015, 0., .015), (.04, .08))):
        nominal = (-.005, 0., .006) if obj == 'cup' else (-.005, -.011, .028)
        result.append(dict(case_id=i, object=obj, delta=[dx, dy, dz], lift_distance=lift,
                           offsets=[nominal[0]+dx, nominal[1]+dy, nominal[2]+dz], seed=seed+i))
    return result


def selected_cases(stage, seed):
    all_cases = cases(seed)
    return [c for c in all_cases if c['delta'] == [0., 0., 0.] and c['lift_distance'] == .08] if stage == 'smoke' else all_cases


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(args, payload, config, settings):
    if (config.sim_freq != 100 or config.control_freq != 20 or
            settings.get('motion_execution', 'ee_delta') != 'ee_delta' or
            payload['action_contract'].get('action_dim') != 12):
        raise ValueError('Requires 100/20 Hz, ee_delta and 12D checkpoint action contract')
    return dict(version=1, split='validation_only', automatic_training=False,
                stage=args.stage, seed=args.seed, case_budget=BUDGET, physics_hz=100,
                control_hz=20, object_spacing_m=.12, checkpoint=str(args.checkpoint.resolve()),
                checkpoint_sha256=sha256(args.checkpoint), config=OmegaConf.to_container(config, resolve=True),
                settings=settings, state_contract=payload['state_contract'], action_contract=payload['action_contract'],
                source_sha256={name: sha256(ROOT/name) for name in SOURCES}, cases=selected_cases(args.stage, args.seed))


class Finished(Exception):
    pass


def stop_callback(provider, collector):
    def end(skill):
        if skill['terminated'] or skill['truncated']:
            raise Finished()
        if skill['completed'] and provider.phase == 'retreat':
            # report() recognizes park as successful retreat; do not execute park.
            provider.action(collector.state)
            raise Finished()
        if skill['completed'] and provider.phase == 'park' and provider.failure:
            raise Finished()
    return end


def read_rows(path, omitted=None):
    if not path.exists():
        return []
    data = path.read_bytes().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(data):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index == len(data)-1 and not line.endswith(b'\n'):
                if omitted is not None:
                    omitted.append(str(path))
                break
            raise ValueError(f'Corrupt JSONL row {index+1}: {path}') from None
    return rows


def export(output, omitted=None, action_dim=12):
    rows = []
    for case_dir in sorted((output/'cases').glob('case_*')):
        case_id = int(case_dir.name.split('_')[1])
        for attempt_dir in sorted(case_dir.glob('attempt_*')):
            attempt = int(attempt_dir.name.split('_')[1])
            if attempt >= 1000000:
                raise ValueError('Too many attempts for unique episode IDs')
            episode_id = case_id * 1000000 + attempt
            for row in read_rows(attempt_dir/'skills.jsonl', omitted):
                if row['episode_id'] != 1:
                    raise ValueError('Unexpected episode in attempt')
                if row['completed'] and row['duration_low_steps'] > 0:
                    if len(row['state']) != 67 or len(row['next_state']) != 67 or len(row['action']) != action_dim:
                        raise ValueError('Invalid endpoint dimensions')
                    rows.append(dict(row, episode_id=episode_id))
    if rows:
        arrays = replay_arrays(rows)
        arrays['duration_low_steps'] = np.asarray([r['duration_low_steps'] for r in rows], dtype=np.int64)
    else:
        arrays = dict(states=np.empty((0, 67), np.float32), actions=np.empty((0, action_dim), np.float32),
                      next_states=np.empty((0, 67), np.float32), episode_ids=np.empty(0, np.int64),
                      terminated=np.empty(0, bool), truncated=np.empty(0, bool), duration_low_steps=np.empty(0, np.int64))
    np.savez_compressed(output/'evaluation.npz', **arrays)
    return len(rows)


def results(output, planned):
    summaries = []
    for case in planned:
        path = output/'cases'/f"case_{case['case_id']:03d}"/'summary.json'
        if path.exists():
            summary = json.loads(path.read_text())
            if summary['case'] != case:
                raise ValueError(f'Case summary mismatch: {path}')
            summaries.append(summary)
    def counts(items):
        successes = sum(bool(s['success']) for s in items)
        failures = {}
        for summary in items:
            if not summary['success']:
                reason = summary.get('report', {}).get('failure') or summary.get('status', 'unknown')
                failures[reason] = failures.get(reason, 0) + 1
        return dict(closed=len(items), successes=successes, success_rate=successes/len(items) if items else None,
                    failure_reasons=failures)
    objects = {obj: counts([s for s in summaries if s['case']['object'] == obj]) for obj in ('cup', 'bowl')}
    points = []
    for case in planned:
        key = (case['object'], case['delta'])
        if any((p['object'], p['delta']) == key for p in points):
            continue
        matching = [s for s in summaries if (s['case']['object'], s['case']['delta']) == key]
        points.append(dict(object=case['object'], delta=case['delta'], **counts(matching)))
    return dict(split='validation_only', total=counts(summaries), per_object=objects, per_point=points, cases=summaries)


def run_case(output, case, payload, settings):
    case_dir = output/'cases'/f"case_{case['case_id']:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.split('_')[1]) for p in case_dir.glob('attempt_*') if p.is_dir()]
    attempt = max(existing, default=0)+1
    directory = case_dir/f'attempt_{attempt:06d}'
    directory.mkdir(exist_ok=False)
    random.seed(case['seed']); np.random.seed(case['seed']); torch.manual_seed(case['seed'])
    env = None
    try:
        env, env_config = make_env(episode_steps=BUDGET, centered_initial_tcp=True,
                                   project_goals_to_region=False, object_shift_xy=(0., 0.))
        env.reset(seed=case['seed'])
        encoder = CabinetSkillStateEncoder(env, latched_gripper=settings['latched_gripper'])
        if encoder.contract() != payload['state_contract']:
            raise ValueError('Encoder state contract differs from checkpoint')
        x, y, z = case['offsets']
        args = argparse.Namespace(x_offset=x, cup_y_offset=y, bowl_y_offset=y,
                                  cup_z_offset=z, bowl_z_offset=z, lift_distance=case['lift_distance'])
        provider = CalibrationProvider(settings, 'original_'+case['object'], args)
        collector = SkillCollector(env, encoder, provider, directory)
        write_json(directory/'attempt.json', dict(case=case, attempt=attempt,
                   environment=OmegaConf.to_container(env_config, resolve=True)))
        try:
            collector.collect(BUDGET, 'curiosity', seed=case['seed'], stage='validation_only',
                              on_step=provider.on_step, on_skill_end=stop_callback(provider, collector))
        except Finished:
            pass
        skills = read_rows(directory/'skills.jsonl')
        report = provider.report(skills)
        success = report['calibration_success']
        status = 'success' if success else ('failure' if provider.failure else 'timeout')
        summary = dict(case=case, attempt=attempt, status=status, success=success,
                       control_ticks=collector.total_steps, report=report)
        write_json(directory/'phase_trace.json', provider.trace)
        write_json(directory/'summary.json', summary)
        # Write closure last: an interrupted attempt is never mistaken for a closed case.
        closure = case_dir/'summary.tmp'
        write_json(closure, summary)
        os.replace(closure, case_dir/'summary.json')
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('plan', 'smoke', 'collect'), default='plan')
    parser.add_argument('--seed', type=int, default=1000000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= 2**32 - 1 - 107:
        raise ValueError('Seed must leave all 108 case seeds in the NumPy uint32 range')
    if args.stage == 'plan':
        if args.resume:
            raise ValueError('Plan cannot be resumed')
        plan = dict(split='validation_only', cases=cases(args.seed), cases_count=108,
                    max_control_ticks_per_case=BUDGET, max_control_ticks=108*BUDGET,
                    physics_hz=100, control_hz=20)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        write_json(args.output/'plan.json', plan)
        print(json.dumps(plan))
        return 0
    payload, config, settings = checkpoint_contract(args.checkpoint)
    manifest = identity(args, payload, config, settings)
    if args.resume:
        if json.loads((args.output/'manifest.json').read_text()) != manifest:
            raise ValueError('Resume manifest differs from requested checkpoint, configuration, source or cases')
        for name, digest in manifest['source_sha256'].items():
            if sha256(args.output/'sources'/name) != digest:
                raise ValueError(f'Resume source snapshot mismatch: {name}')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        for name in SOURCES:
            destination = args.output/'sources'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, destination)
        write_json(args.output/'manifest.json', manifest)
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    for case in manifest['cases']:
        if not (args.output/'cases'/f"case_{case['case_id']:03d}"/'summary.json').exists():
            run_case(args.output, case, payload, settings)
        omitted = []
        export(args.output, omitted)
        result = results(args.output, manifest['cases'])
        result['incomplete_jsonl_tail_omitted'] = omitted
        write_json(args.output/'result.json', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
