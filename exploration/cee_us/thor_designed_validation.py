"""Collect bounded scripted FloorPlan1 validation with the frozen THOR contract."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import torch
from PIL import Image

from domain.thor.env import PROJECT_ROOT, THOR_COMMIT, ThorEnv
from exploration.cee_us.cee_us_thor import (CEEUS_COMMIT, REPLAY_KEYS, SCHEMA,
    check_scene_contract, file_hash, load_dataset, save_dataset, source_hashes)
from exploration.cee_us.thor_state import ACTIONS, ThorState, execute


def plan_cases(metadata):
    """Enumerate every capable instance and interleave independent case families."""
    objects = sorted(metadata['objects'], key=lambda obj: obj['objectId'])
    groups = []
    for family, capability, sequence in (
        ('openable', 'openable', ('Open', 'Open', 'Close', 'Close', 'Open', 'Close')),
        ('toggleable', 'toggleable', ('ToggleOn', 'ToggleOn', 'ToggleOff', 'ToggleOff', 'ToggleOn', 'ToggleOff')),
        ('pickupable', 'pickupable', ('Pickup', 'Pickup', 'Put', 'Put', 'Pickup', 'Put')),
    ):
        groups.append([dict(family=family, target=o['objectId'], actions=sequence)
                       for o in objects if o[capability]])
    sources = {kind: [o['objectId'] for o in objects if o['objectType'] == kind and o['pickupable']]
               for kind in ('Apple', 'Mug', 'Plate')}
    receptacles = []
    for i, obj in enumerate(o for o in objects if o['receptacle']):
        kind = ('Apple', 'Mug', 'Plate')[i % 3]
        source = next((x for x in sources[kind] if x != obj['objectId']), None)
        receptacles.append(dict(family='receptacle', target=obj['objectId'], source=source,
                                source_type=kind, actions=('Put', 'Put', 'Pickup', 'Put', 'Put', 'Pickup')))
    groups.append(receptacles)
    nav = ('MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Crouch', 'Stand')
    groups.append([dict(family='navigation', pose_index=i, actions=nav) for i in range(8)])
    groups.append([dict(family='no_target', actions=('Pickup', 'Put', 'Open', 'Close', 'ToggleOn', 'ToggleOff'))])
    groups.append([dict(family='default', actions=('MoveAhead', 'RotateLeft', 'LookDown', 'Open', 'Pickup', 'Put', 'Close', 'ToggleOn', 'ToggleOff'))])
    cases = []
    for i in range(max(map(len, groups))):
        for group in groups:
            if i < len(group):
                case = dict(group[i], case_id=len(cases))
                cases.append(case)
    return cases


def _selected(encoder, state, action):
    _, index = encoder.action_encoding(torch.from_numpy(state), ACTIONS.index(action))
    index = int(index.item())
    return encoder.ids[index] if index >= 0 else None


class Setup:
    def __init__(self, env, output):
        self.env, self.path = env, output / 'setup.jsonl'
        self.calls = 0
        self.resets = 0
        self.cache = {}

    def record(self, row):
        with self.path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    def step(self, action, **kwargs):
        event = self.env.controller.step(action=action, **kwargs)
        self.calls += 1
        self.record(dict(action=action, parameters=kwargs, success=event.metadata['lastActionSuccess'],
                         error=event.metadata['errorMessage']))
        return event

    def reset(self):
        event = self.env.reset()
        self.resets += 1
        self.record({'action': 'reset', 'reset_number': self.resets,
                     'note': 'SetRandomSeed; map camera setup only when rendering is enabled'})
        return event

    def poses(self, target):
        if target not in self.cache:
            event = self.step('GetInteractablePoses', objectId=target, rotations=[0, 90, 180, 270],
                              horizons=[0, 30, 60], standings=[True])
            if not event.metadata['lastActionSuccess']:
                self.cache[target] = ([], 'pose_query_failed: ' + event.metadata['errorMessage'])
            else:
                self.cache[target] = (event.metadata['actionReturn'] or [], None)
        return self.cache[target]

    def place(self, encoder, target, action, max_poses):
        poses, error = self.poses(target)
        if error:
            return None, error
        for pose in poses[:max_poses]:
            event = self.step('TeleportFull', **pose)
            if event.metadata['lastActionSuccess']:
                state = encoder.encode(event.metadata)
                if _selected(encoder, state, action) == target:
                    return pose, None
        return None, f'no_matching_nearest_target_in_first_{min(len(poses), max_poses)}_poses'


def collect(env, encoder, cases, output, max_attempts, max_poses, seed, case_limit=None, save_frames=False):
    setup = Setup(env, output)
    data = {key: [] for key in REPLAY_KEYS}
    labels, results = [], []
    rng = np.random.default_rng(seed)
    reachable = None
    for index, case in enumerate(cases):
        if case_limit is not None and index >= case_limit:
            break
        if len(data['types']) + len(case['actions']) > max_attempts:
            break
        event = setup.reset()
        check_scene_contract(encoder, ThorState(event.metadata))
        reason = None
        poses = {}
        family = case['family']
        if family in ('openable', 'toggleable', 'pickupable', 'receptacle'):
            if family == 'receptacle':
                setup.poses(case['target'])  # Query the default layout before preparing the hand.
                source = case['source']
                if source is None:
                    reason = 'no_distinct_source_of_requested_type'
                else:
                    poses['source'], reason = setup.place(encoder, source, 'Pickup', max_poses)
                    if reason is None:
                        state = encoder.encode(env.controller.last_event.metadata)
                        state, _, success, target, error, called = execute(setup, encoder, state, ACTIONS.index('Pickup'))
                        setup.record(dict(action='prepared_pickup_result', target=target, success=success,
                                          error=error, unity_call=called))
                        if target != source or not success:
                            reason = 'source_pickup_failed: ' + str(error)
            if reason is None:
                poses['target'], reason = setup.place(encoder, case['target'], case['actions'][0], max_poses)
        elif family == 'navigation':
            if reachable is None:
                response = setup.step('GetReachablePositions')
                if not response.metadata['lastActionSuccess']:
                    raise RuntimeError('GetReachablePositions: ' + response.metadata['errorMessage'])
                reachable = response.metadata['actionReturn']
                if not reachable:
                    raise RuntimeError('GetReachablePositions returned no positions')
                reachable = [reachable[i] for i in rng.permutation(len(reachable))]
            i = case['pose_index']
            pos = reachable[i % len(reachable)]
            pose = dict(x=pos['x'], y=pos['y'], z=pos['z'], rotation=(i % 4) * 90,
                        horizon=(i % 3) * 30, standing=(i % 2 == 0))
            event = setup.step('TeleportFull', **pose)
            poses['start'] = pose
            if not event.metadata['lastActionSuccess']:
                reason = 'navigation_teleport_failed: ' + event.metadata['errorMessage']
        elif family == 'no_target':
            agent = event.metadata['agent']
            pos = agent['position']
            pose = dict(x=pos['x'], y=pos['y'], z=pos['z'], rotation=agent['rotation']['y'], horizon=-30, standing=True)
            event = setup.step('TeleportFull', **pose)
            poses['start'] = pose
            if not event.metadata['lastActionSuccess']:
                reason = 'look_up_teleport_failed: ' + event.metadata['errorMessage']
        if reason:
            results.append(dict(case=case, status='skipped', reason=reason, setup_poses=poses))
            print(json.dumps({'case_id': case['case_id'], 'family': family, 'status': 'skipped', 'reason': reason, 'attempts': len(data['types'])}), flush=True)
            continue
        episode = sum(r['status'] == 'visited' for r in results)
        event = env.controller.last_event
        state = encoder.encode(event.metadata)
        if save_frames and event.frame is not None:
            Image.fromarray(event.frame).save(output / f'episode_{episode}_ego.png')
        if save_frames and event.third_party_camera_frames:
            Image.fromarray(event.third_party_camera_frames[-1]).save(output / f'episode_{episode}_topdown.png')
        results.append(dict(case=case, status='visited', episode=episode, setup_poses=poses))
        for action in case['actions']:
            action_index = ACTIONS.index(action)
            successor, encoded, success, target, error, called = execute(env, encoder, state, action_index)
            values = (state, encoded, successor, action_index, success, target or '', called, episode, error or '')
            for key, value in zip(REPLAY_KEYS, values):
                data[key].append(value)
            label = dict(episode=episode, case_id=case['case_id'], family=family,
                         requested_target=case.get('target'), source=case.get('source'), action=action,
                         actual_target=target, success=bool(success), error=error or '', unity_call=bool(called),
                         state_changed=bool(np.any(state[:8] != successor[:8]) or
                                            np.any(state[9:encoder.dynamic_dim] != successor[9:encoder.dynamic_dim])))
            labels.append(label)
            state = successor
        print(json.dumps({'case_id': case['case_id'], 'family': family, 'status': 'visited', 'attempts': len(data['types'])}), flush=True)
    pending = cases[len(results):]
    dtypes = dict(states=np.float32, actions=np.float32, next_states=np.float32, types=np.int64,
                  successes=bool, targets=str, unity_calls=bool, episodes=np.int64, errors=str)
    data = {key: np.asarray(values, dtype=dtypes[key]) for key, values in data.items()}
    with (output / 'rows.jsonl').open('w') as stream:
        for row in labels:
            stream.write(json.dumps(row) + '\n')
    dimensions = ('family', 'action', 'requested_target', 'actual_target', 'success', 'error', 'state_changed')
    coverage = {key: dict(Counter(str(row[key]) for row in labels)) for key in dimensions}
    action_target = {}
    for row in labels:
        key = json.dumps([row['action'], row['actual_target']])
        counts = action_target.setdefault(key, {'attempts': 0, 'successes': 0, 'state_changed': 0})
        counts['attempts'] += 1
        counts['successes'] += int(row['success'])
        counts['state_changed'] += int(row['state_changed'])
    coverage.update(action_actual_target_counts=action_target, requested_cases=cases, visited_cases=[r for r in results if r['status'] == 'visited'],
                    skipped_cases=[r for r in results if r['status'] == 'skipped'], pending_cases=pending,
                    attempts=len(labels), transition_unity_calls=int(data['unity_calls'].sum()),
                    setup_direct_unity_calls=setup.calls, setup_resets=setup.resets,
                    reset_internal_step_calls=(3 if getattr(env, "metadata_only", False) is False else 1) * setup.resets,
                    controller_reset_calls=setup.resets,
                    environment_initialization_resets=1, environment_initialization_reset_internal_step_calls=(1 if getattr(env, "metadata_only", False) else 3),
                    requested_actual_pairs=dict(Counter(json.dumps([r['requested_target'], r['actual_target']]) for r in labels)),
                    receptacle_source_pairs=dict(Counter(json.dumps([r['source'], r['requested_target']]) for r in labels if r['family'] == 'receptacle')))
    return data, coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-data', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-attempts', type=int, default=600)
    parser.add_argument('--max-poses', type=int, default=6)
    parser.add_argument('--case-limit', type=int)
    parser.add_argument('--save-frames', action='store_true', help='Enable rendered ego/map frames; default is metadata-only headless collection')
    args = parser.parse_args(argv)
    if args.max_attempts <= 0 or args.max_poses <= 0 or (args.case_limit is not None and args.case_limit <= 0):
        parser.error('budgets and case limit must be positive')
    encoder, initial, _ = load_dataset(args.reference_data)
    cases = plan_cases(initial)
    args.output.mkdir(parents=True, exist_ok=False)
    with ThorEnv(scene='FloorPlan1', seed=args.seed, output_dir=args.output / 'environment', metadata_only=not args.save_frames) as env:
        check_scene_contract(encoder, ThorState(env.controller.last_event.metadata))
        data, coverage = collect(env, encoder, cases, args.output, args.max_attempts, args.max_poses, args.seed, args.case_limit, args.save_frames)
    if not len(data['types']):
        raise RuntimeError('No cases were retained; inspect setup.jsonl for placement failures')
    save_dataset(args.output / 'evaluation.npz', encoder, initial, data)
    (args.output / 'coverage.json').write_text(json.dumps(coverage, indent=2))
    sources = source_hashes()
    sources[str(Path(__file__).relative_to(PROJECT_ROOT))] = file_hash(__file__)
    manifest = dict(schema=SCHEMA, created=datetime.now(ZoneInfo('Europe/London')).isoformat(),
        scene='FloorPlan1', thor_commit=THOR_COMMIT, cee_us_commit=CEEUS_COMMIT,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip(),
        versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'ai2thor', 'gymnasium', 'forwardable', 'Pillow')},
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        reference_sha256=file_hash(args.reference_data), dataset_sha256=file_hash(args.output / 'evaluation.npz'),
        sources=sources, save_frames=args.save_frames, metadata_only=not args.save_frames, learning=False, cross_scene=False, forceAction=False, placeStationary=True,
        target_selection='nearest visible capable object; no task preference',
        setup='episode placement and prepared held object only; no setup between retained transitions')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'attempts': coverage['attempts'], 'visited_cases': len(coverage['visited_cases']),
                      'skipped_cases': len(coverage['skipped_cases']), 'pending_cases': len(coverage['pending_cases']),
                      'dataset': str(args.output / 'evaluation.npz')}), flush=True)


if __name__ == '__main__':
    main()
