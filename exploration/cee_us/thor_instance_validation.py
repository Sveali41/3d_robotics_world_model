"""Collect bounded instance-level FloorPlan1 validation without rendering."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np

from domain.thor.env import PROJECT_ROOT, THOR_COMMIT, ThorEnv
from exploration.cee_us.cee_us_thor import (CEEUS_COMMIT, REPLAY_KEYS, SCHEMA,
    check_scene_contract, file_hash, load_dataset, save_dataset, source_hashes)
from exploration.cee_us.thor_coverage import InstanceCoverage
from exploration.cee_us.thor_designed_validation import Setup as BaseSetup, _selected
from exploration.cee_us.thor_state import ACTIONS, ThorState, execute


def ordered_poses(poses, position):
    """Visit different agent locations before repeating camera/posture variants."""
    groups = {}
    for pose in sorted(poses, key=lambda q: (q['x']-position['x'])**2 + (q['z']-position['z'])**2):
        groups.setdefault((pose['x'], pose['z']), []).append(pose)
    return [group[i] for i in range(max(map(len, groups.values()), default=0))
            for group in groups.values() if i < len(group)]


class Setup(BaseSetup):
    def reset(self):
        self.cache.clear()
        return super().reset()

    def step(self, action, **kwargs):
        event = super().step(action, **kwargs)
        if action in ('OpenObject', 'CloseObject', 'PickupObject', 'PutObject'):
            self.cache.clear()
        return event

    def poses(self, target):
        if target not in self.cache:
            event = self.step('GetInteractablePoses', objectId=target,
                              rotations=[0, 90, 180, 270], horizons=[-30, 0, 30, 60],
                              standings=[True, False])
            self.cache[target] = ((event.metadata['actionReturn'] or []), None) if event.metadata['lastActionSuccess'] else ([], 'pose_query_failed: ' + event.metadata['errorMessage'])
        return self.cache[target]

    def place(self, encoder, target, action, max_poses, offset=0, poses=None):
        poses, error = self.poses(target) if poses is None else (poses, None)
        if error:
            return None, error
        obj = next(o for o in self.env.controller.last_event.metadata['objects'] if o['objectId'] == target)
        p = obj['position']
        poses = ordered_poses(poses, p)[:max_poses]
        if poses:
            offset %= len(poses)
            poses = poses[offset:] + poses[:offset]
        for pose in poses:
            event = self.step('TeleportFull', **pose)
            if event.metadata['lastActionSuccess'] and _selected(encoder, encoder.encode(event.metadata), action) == target:
                return pose, None
        return None, 'no_matching_nearest_target_in_searched_poses'

    def parents(self, encoder, target, max_poses, offset):
        objects = {o['objectId']: o for o in self.env.controller.last_event.metadata['objects']}
        ordered, seen = [], set()
        def visit(item):
            if item in seen or item not in objects:
                return
            seen.add(item)
            for parent in objects[item].get('parentReceptacles') or []:
                visit(parent)
            if item != target and objects[item]['openable'] and not objects[item]['isOpen']:
                ordered.append(item)
        visit(target)
        for parent in ordered:
            pose, reason = self.place(encoder, parent, 'Open', max_poses, offset)
            if reason:
                poses, query_error = self.poses(parent)
                if query_error:
                    return 'parent_' + parent + ': ' + query_error
                obj = objects[parent]['position']
                poses = sorted(poses, key=lambda q: (q['x']-obj['x'])**2 + (q['z']-obj['z'])**2)[:max_poses]
                event = None
                for pose in poses:
                    teleported = self.step('TeleportFull', **pose)
                    if teleported.metadata['lastActionSuccess']:
                        event = self.step('OpenObject', objectId=parent, forceAction=False, openness=1.0)
                        if event.metadata['lastActionSuccess']:
                            break
                if event is None or not event.metadata['lastActionSuccess']:
                    return 'parent_' + parent + ': direct_open_failed'
                continue
            event = self.step('OpenObject', objectId=parent, forceAction=False, openness=1.0)
            if not event.metadata['lastActionSuccess']:
                return 'parent_open_failed: ' + event.metadata['errorMessage']
        return None


def sources_for(objects, target):
    preferred = {'CoffeeMachine': ('Mug',), 'Toaster': ('Bread',),
                 'StoveBurner': ('Pot', 'Pan', 'Kettle'),
                 'Cup': ('Egg', 'Tomato', 'Potato'), 'Bowl': ('Egg', 'Tomato', 'Potato')}
    order = preferred.get(target['objectType'], ()) + ('Apple',)
    return sorted((o for o in objects if o['pickupable'] and o['objectId'] != target['objectId']),
                  key=lambda o: (order.index(o['objectType']) if o['objectType'] in order else len(order), o['objectId']))


def navigation_poses(reachable, action, blocked):
    grid = {(round(p['x'] / .25), round(p['z'] / .25)) for p in reachable}
    for p in reachable:
        x, z = round(p['x'] / .25), round(p['z'] / .25)
        for yaw, dx, dz in ((0, 0, 1), (90, 1, 0), (180, 0, -1), (270, -1, 0)):
            sign = -1 if action == 'MoveBack' else 1
            if (((x + sign*dx, z + sign*dz) not in grid) == blocked):
                yield dict(x=p['x'], y=p['y'], z=p['z'], rotation=yaw, horizon=0, standing=True)


def load_placements(path):
    """Read observed successful placement contexts for later native revalidation."""
    placements = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            pose = row.get('setup_pose')
            source, target = row.get('source'), row.get('actual_target')
            if (row.get('action') == 'Put' and row.get('success') is True
                    and source and target and isinstance(pose, dict)
                    and all(k in pose for k in ('x', 'y', 'z', 'rotation', 'horizon', 'standing'))):
                placements[(source, target)] = pose
    return placements


def validate_targets(initial, targets):
    if targets is not None:
        known = {o['objectId'] for o in initial['objects']} | set(ACTIONS[:8])
        unknown = set(targets) - known
        if unknown:
            raise ValueError('Unknown targets: ' + ', '.join(sorted(unknown)))


def collect(env, encoder, initial, output, min_count, max_attempts, max_poses, seed, case_limit=None, failure_controls=False, reuse_placements=None, targets=None):
    setup = Setup(env, output)
    ledger = InstanceCoverage(encoder, initial, min_count)
    data = {k: [] for k in REPLAY_KEYS}
    visited, skipped, contexts = [], [], {}
    controls = {}
    rng = np.random.default_rng(seed)
    objects = sorted(initial['objects'], key=lambda o: o['objectId'])
    cases = [(family, o['objectId']) for family, cap in [('openable', 'openable'), ('toggleable', 'toggleable'), ('pickupable', 'pickupable'), ('receptacle', 'receptacle')] for o in objects if o[cap]]
    cases += [('navigation', a) for a in ('MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Crouch', 'Stand')]
    validate_targets(initial, targets)
    if targets is not None:
        cases = [case for case in cases if case[1] in targets]
    if case_limit is not None:
        cases = cases[:case_limit]
    reachable = None
    successful_poses = dict(reuse_placements or {})
    successful_sources = {target: source for source, target in successful_poses}
    episode = -1
    last_marker = None
    last_state = None
    current_case = None
    current_pose = None
    source = None

    def report():
        result = ledger.report()
        result.update(visited_cases=visited, skipped_cases=skipped, attempts=len(data['types']),
                      transition_unity_calls=sum(data['unity_calls']), setup_direct_unity_calls=setup.calls,
                      setup_resets=setup.resets, distinct_starting_contexts={k: len(v) for k, v in contexts.items()},
                      budgets=dict(max_attempts=max_attempts, max_poses=max_poses, case_limit=case_limit, targets=targets),
                      failure_controls=controls)
        (output / 'coverage.json').write_text(json.dumps(result, indent=2))
        return result

    def missing(action, target):
        return any(r['action'] == action and r['object_id'] == target and not r['covered'] for r in ledger.report()['requirements'])

    def attempt(action, control=None):
        nonlocal episode, last_marker, last_state
        if len(data['types']) >= max_attempts:
            return False, None, 'attempt_budget_exhausted'
        marker = (setup.calls, setup.resets)
        if marker != last_marker:
            episode += 1
            state = encoder.encode(env.controller.last_event.metadata)
        else:
            state = last_state
        successor, encoded, success, target, error, called = execute(env, encoder, state, ACTIONS.index(action))
        ledger.observe(state, successor, action, target, success)
        setup.cache.clear()
        for k, v in zip(REPLAY_KEYS, (state, encoded, successor, ACTIONS.index(action), success, target or '', called, episode, error or '')):
            data[k].append(v)
        row = dict(episode=episode, case=current_case, source=source, requested_target=current_case[1],
                   actual_target=target, action=action, success=bool(success), error=error or '',
                   unity_call=bool(called), setup_pose=current_pose, setup_reset=setup.resets,
                   control=control)
        if control is not None:
            counts = controls.setdefault(control, {'attempts': 0, 'successes': 0, 'failures': 0, 'requested_objects': {}})
            counts['attempts'] += 1
            counts['successes' if success else 'failures'] += 1
            counts['requested_objects'][current_case[1]] = counts['requested_objects'].get(current_case[1], 0) + 1
        with (output / 'rows.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        key = json.dumps([action, target])
        contexts.setdefault(key, set()).add(json.dumps(current_pose, sort_keys=True))
        last_marker, last_state = marker, successor
        return success, target, error or ''

    try:
        for case_index, (family, target) in enumerate(cases):
            current_case = (family, target)
            obj = next((o for o in objects if o['objectId'] == target), None)
            actions = {'openable': ('Open', 'Close'), 'toggleable': ('ToggleOn', 'ToggleOff'),
                       'pickupable': ('Pickup', 'Release'), 'receptacle': ('Put',), 'navigation': (target,)}[family]
            diagnostics = []
            control_labels = {'openable': ('already_open', 'already_closed'),
                              'toggleable': ('already_on', 'already_off'),
                              'pickupable': ('hand_occupied',),
                              'receptacle': ('empty_hand',), 'navigation': ()}[family]
            # Bounded retries preserve missing requirements when no legal execution is found.
            before_case = len(data['types'])
            trial_limit = min_count * 3 + (sum(o['pickupable'] for o in objects) if family == 'receptacle' else 0)
            for repetition in range(trial_limit):
                effect_missing = any(missing(a, None if family == 'navigation' else target) for a in actions)
                control_missing = failure_controls and any(
                    controls.get(label, {}).get('requested_objects', {}).get(target, 0) == 0
                    for label in control_labels)
                if len(data['types']) >= max_attempts or not (effect_missing or control_missing):
                    break
                setup.reset()
                check_scene_contract(encoder, ThorState(env.controller.last_event.metadata))
                source, current_pose = None, None
                if family == 'navigation':
                    if reachable is None:
                        event = setup.step('GetReachablePositions')
                        if not event.metadata['lastActionSuccess']:
                            raise RuntimeError('GetReachablePositions: ' + event.metadata['errorMessage'])
                        reachable = list(event.metadata['actionReturn'] or [])
                        rng.shuffle(reachable)
                    if target in ('MoveAhead', 'MoveBack'):
                        requirements = [r for r in ledger.report()['requirements'] if r['action'] == target and not r['covered']]
                        blocked = bool(requirements and 'failure' in requirements[0]['effect'])
                        poses = list(navigation_poses(reachable, target, blocked))
                    else:
                        poses = [dict(x=p['x'], y=p['y'], z=p['z'], rotation=0, horizon=0, standing=target != 'Stand') for p in reachable]
                    if not poses:
                        diagnostics.append('no_navigation_pose')
                        break
                    current_pose = poses[repetition % len(poses)]
                    event = setup.step('TeleportFull', **current_pose)
                    if event.metadata['lastActionSuccess']:
                        attempt(target)
                    else:
                        diagnostics.append('navigation_teleport_failed: ' + event.metadata['errorMessage'])
                    continue
                # Opening a parent can move its door in front of an already accessible object.
                initial_pickup_pose = None
                if family == 'pickupable':
                    initial_pickup_pose, _ = setup.place(encoder, target, 'Pickup', max_poses, repetition)
                reason = None if initial_pickup_pose is not None else setup.parents(encoder, target, max_poses, repetition)
                if reason:
                    diagnostics.append(reason)
                    break
                countertop_poses = {}
                if family == 'pickupable':
                    for countertop in (o for o in objects if o['objectType'] == 'CounterTop' and o['receptacle']):
                        countertop_poses[countertop['objectId']], _ = setup.poses(countertop['objectId'])
                if family in ('openable', 'toggleable', 'pickupable'):
                    current = next(o for o in env.controller.last_event.metadata['objects'] if o['objectId'] == target)
                    first = actions[0]
                    if family == 'openable' and current['isOpen']:
                        first = 'Close'
                    if family == 'toggleable' and current['isToggled']:
                        first = 'ToggleOff'
                    second = {'Open': 'Close', 'Close': 'Open', 'ToggleOn': 'ToggleOff', 'ToggleOff': 'ToggleOn'}.get(first)
                    current_pose, reason = setup.place(encoder, target, first, max_poses, repetition)
                    if reason:
                        diagnostics.append(reason)
                        break
                    success, actual, error = attempt(first)
                    if failure_controls and success and actual == target:
                        if family == 'pickupable':
                            attempt('Pickup', 'hand_occupied')
                        else:
                            labels = {'Open': 'already_open', 'Close': 'already_closed',
                                      'ToggleOn': 'already_on', 'ToggleOff': 'already_off'}
                            state = encoder.encode(env.controller.last_event.metadata)
                            if _selected(encoder, state, first) != target:
                                current_pose, reason = setup.place(encoder, target, first, max_poses, repetition)
                            else:
                                reason = None
                            if reason:
                                diagnostics.append('control_' + labels[first] + ': ' + reason)
                            else:
                                attempt(first, labels[first])
                    if family != 'pickupable' and success and actual == target:
                        inverse_success, inverse_target, _ = attempt(second)
                        if inverse_target != target and missing(second, target):
                            current_pose, reason = setup.place(encoder, target, second, max_poses, repetition)
                            if reason:
                                diagnostics.append(reason)
                            else:
                                inverse_success, inverse_target, _ = attempt(second)
                        if failure_controls and inverse_success and inverse_target == target:
                            state = encoder.encode(env.controller.last_event.metadata)
                            if _selected(encoder, state, second) != target:
                                current_pose, reason = setup.place(encoder, target, second, max_poses, repetition)
                            else:
                                reason = None
                            if reason:
                                diagnostics.append('control_' + labels[second] + ': ' + reason)
                            else:
                                attempt(second, labels[second])
                    if family != 'pickupable' or not success or actual != target:
                        continue
                    source = target
                    receptacles = [o for o in objects if o['objectType'] == 'CounterTop' and o['receptacle']]
                else:
                    receptacles = [obj]
                for receptor in receptacles:
                    receptor_id = receptor['objectId']
                    if family == 'receptacle':
                        current = next(o for o in env.controller.last_event.metadata['objects'] if o['objectId'] == receptor_id)
                        if current['openable'] and not current['isOpen']:
                            pose, reason = setup.place(encoder, receptor_id, 'Open', max_poses, repetition)
                            if reason:
                                diagnostics.append(reason)
                                continue
                            event = setup.step('OpenObject', objectId=receptor_id, forceAction=False, openness=1.0)
                            if not event.metadata['lastActionSuccess']:
                                diagnostics.append('receptacle_open_failed: ' + event.metadata['errorMessage'])
                                continue
                        cached_poses, _ = setup.poses(receptor_id)
                        if failure_controls:
                            current_pose, reason = setup.place(encoder, receptor_id, 'Put', max_poses, repetition, cached_poses)
                            if reason:
                                diagnostics.append('control_empty_hand: ' + reason)
                            else:
                                attempt('Put', 'empty_hand')
                        picked = False
                        candidates = sources_for(objects, receptor)
                        if candidates:
                            shift = repetition % len(candidates)
                            candidates = candidates[shift:] + candidates[:shift]
                        if target in successful_sources:
                            candidates.sort(key=lambda o: o['objectId'] != successful_sources[target])
                        for candidate in candidates:
                            source = candidate['objectId']
                            reason = setup.parents(encoder, source, max_poses, repetition)
                            if reason:
                                continue
                            cached_poses, _ = setup.poses(receptor_id)
                            _, reason = setup.place(encoder, source, 'Pickup', max_poses, repetition)
                            if reason:
                                continue
                            state = encoder.encode(env.controller.last_event.metadata)
                            _, _, ok, actual, error, called = execute(setup, encoder, state, ACTIONS.index('Pickup'))
                            setup.record(dict(action='prepared_pickup_result', target=actual, success=ok, error=error, unity_call=called))
                            if ok and actual == source:
                                picked = True
                                break
                        if not picked:
                            diagnostics.append('no_source_pickup')
                            continue
                    else:
                        cached_poses = countertop_poses.get(receptor_id, [])
                    current_pose, reason = setup.place(encoder, receptor_id, 'Put', max_poses, repetition, cached_poses)
                    if reason:
                        diagnostics.append(reason)
                        continue
                    success, actual, error = attempt('Put')
                    if success and actual == receptor_id:
                        successful_sources[receptor_id] = source
                        successful_poses[(source, receptor_id)] = current_pose
                        break
                    diagnostics.append('put_failed: ' + error)
                    known_pose = successful_poses.get((source, receptor_id))
                    held = any(o['objectId'] == source and o['isPickedUp']
                               for o in env.controller.last_event.metadata['objects'])
                    if known_pose is not None and held and len(data['types']) < max_attempts:
                        current_pose, reason = setup.place(encoder, receptor_id, 'Put', max_poses,
                                                           poses=[known_pose])
                        if reason:
                            diagnostics.append('known_pose: ' + reason)
                        else:
                            success, actual, error = attempt('Put')
                            if success and actual == receptor_id:
                                successful_sources[receptor_id] = source
                                successful_poses[(source, receptor_id)] = current_pose
                                break
                            diagnostics.append('known_pose_put_failed: ' + error)
                    if family == 'receptacle':
                        break
            record = dict(case=current_case, diagnostics=diagnostics, attempts=len(data['types']))
            (visited if len(data['types']) > before_case else skipped).append(record)
            report()
            print(json.dumps(dict(case_index=case_index, case=current_case, attempts=len(data['types']), diagnostics=diagnostics[-1:])), flush=True)
            if len(data['types']) >= max_attempts:
                break
    finally:
        coverage = report()
        if data['types']:
            dtypes = dict(states=np.float32, actions=np.float32, next_states=np.float32, types=np.int64,
                          successes=bool, targets=str, unity_calls=bool, episodes=np.int64, errors=str)
            arrays = {}
            for k in REPLAY_KEYS:
                arrays[k] = np.asarray(data.pop(k), dtype=dtypes[k])
            save_dataset(output / 'evaluation.npz', encoder, initial, arrays)
    return coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-count', type=int, default=3)
    parser.add_argument('--max-attempts', type=int, default=5000)
    parser.add_argument('--max-poses', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--case-limit', type=int)
    parser.add_argument('--failure-controls', action='store_true')
    parser.add_argument('--reuse-placements-from', type=Path)
    parser.add_argument('--targets', nargs='+')
    args = parser.parse_args(argv)
    if min(args.min_count, args.max_attempts, args.max_poses) < 1 or (args.case_limit is not None and args.case_limit < 1):
        parser.error('counts and budgets must be positive')
    encoder, initial, _ = load_dataset(args.reference_data)
    try:
        validate_targets(initial, args.targets)
    except ValueError as error:
        parser.error(str(error))
    placements = load_placements(args.reuse_placements_from) if args.reuse_placements_from else {}
    args.output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    for path in (Path(__file__), Path(__file__).with_name('thor_coverage.py'), Path(__file__).with_name('thor_designed_validation.py')):
        sources[str(path.relative_to(PROJECT_ROOT))] = file_hash(path)
    manifest = dict(schema=SCHEMA, created=datetime.now(ZoneInfo('Europe/London')).isoformat(), scene='FloorPlan1',
                    thor_commit=THOR_COMMIT, cee_us_commit=CEEUS_COMMIT,
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip(),
                    versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'ai2thor', 'gymnasium', 'forwardable', 'Pillow')},
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    reference_sha256=file_hash(args.reference_data),
                    reuse_placements_sha256=file_hash(args.reuse_placements_from) if args.reuse_placements_from else None,
                    sources=sources, contract=encoder.contract(),
                    protocol='metadata-only; native nearest-visible execution; setup logged separately; new episode after setup; forceAction=False; placeStationary=True; optional native failure controls')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    try:
        with ThorEnv(scene='FloorPlan1', seed=args.seed, output_dir=args.output / 'environment', metadata_only=True) as env:
            check_scene_contract(encoder, ThorState(env.controller.last_event.metadata))
            collect(env, encoder, initial, args.output, args.min_count, args.max_attempts, args.max_poses, args.seed, args.case_limit, failure_controls=args.failure_controls,
                    reuse_placements=placements, targets=args.targets)
    finally:
        dataset = args.output / 'evaluation.npz'
        if dataset.exists():
            manifest['dataset_sha256'] = file_hash(dataset)
            (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
