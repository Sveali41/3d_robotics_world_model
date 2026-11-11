"""Collect bounded, reproducible FloorPlan1 interaction validation episodes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from domain.thor.env import snapshot
from exploration.cee_us.thor_state import ACTIONS, execute


def collect_validation(env, encoder, seed, random_steps, output_dir):
    """Collect actual transitions, allowing only explicit initial-pose setup per episode.

    Pose queries, reset and teleport are setup, never included as transitions. Every
    attempted interaction is recorded even if the chosen nearest object differs from
    the object whose pose was queried. Coverage is reported rather than assumed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {k: [] for k in ('states', 'actions', 'next_states', 'types', 'successes',
                             'targets', 'unity_calls', 'episodes', 'errors')}
    rng = np.random.default_rng(seed)
    episode = 0
    setup_calls = 0
    details = []
    prior = np.asarray([.5 / 8] * 8 + [.5 / 6] * 6)

    def attempt(state, action):
        index = ACTIONS.index(action) if isinstance(action, str) else int(action)
        successor, encoded, success, target, error, called = execute(env, encoder, state, index)
        for key, value in zip(data, (state, encoded, successor, index, success,
                                     target or '', called, episode, error or '')):
            data[key].append(value)
        return successor, success, target

    # A normal episode provides navigation, natural interaction failures and a baseline.
    state = encoder.encode(env.reset().metadata)
    for _ in range(random_steps):
        state, _, _ = attempt(state, rng.choice(len(ACTIONS), p=prior))
    episode += 1

    # A fixed trajectory from the default FloorPlan1 start exercises navigation and
    # known open, pickup, close and toggle interactions, plus a failed Put.
    event = env.reset()
    state = encoder.encode(event.metadata)
    before = snapshot(event)
    if event.frame is not None:
        Image.fromarray(event.frame).save(output_dir / f'episode_{episode}_ego.png')
    if event.third_party_camera_frames:
        Image.fromarray(event.third_party_camera_frames[-1]).save(output_dir / f'episode_{episode}_topdown.png')
    for action in ('MoveAhead', 'RotateLeft', 'LookDown', 'Open', 'Pickup', 'Put',
                   'Close', 'ToggleOn', 'ToggleOff'):
        state, _, _ = attempt(state, action)
    details.append({'episode': episode, 'kind': 'default_pose_fixed_sequence',
                    'before': before, 'after': snapshot(env.controller.last_event)})
    episode += 1

    # At most three objects and eight poses per capability; stop when a first action
    # succeeds. Secondary paired actions exercise the resulting state in the same episode.
    groups = (('openable', 'Open', 'Close'), ('pickupable', 'Pickup', 'Put'),
              ('toggleable', 'ToggleOn', 'ToggleOff'))
    for capability, first, second in groups:
        if any(a == ACTIONS.index(first) and ok for a, ok in zip(data['types'], data['successes'])):
            continue
        initial = env.reset().metadata
        choices = sorted((o for o in initial['objects'] if o[capability]),
                         key=lambda o: o['objectId'])[:3]
        found = False
        for obj in choices:
            env.reset()
            response = env.controller.step(action='GetInteractablePoses', objectId=obj['objectId'],
                                           rotations=[0, 90, 180, 270], horizons=[0, 30],
                                           standings=[True])
            setup_calls += 1
            if not response.metadata['lastActionSuccess']:
                details.append({'query_object': obj['objectId'], 'error': response.metadata['errorMessage']})
                continue
            poses = response.metadata['actionReturn'] or []
            for pose in poses[:8]:
                env.reset()
                event = env.controller.step(action='TeleportFull', **pose)
                setup_calls += 1
                if not event.metadata['lastActionSuccess']:
                    continue
                before = snapshot(event)
                state = encoder.encode(event.metadata)
                if event.frame is not None:
                    Image.fromarray(event.frame).save(output_dir / f'episode_{episode}_ego.png')
                if event.third_party_camera_frames:
                    Image.fromarray(event.third_party_camera_frames[-1]).save(output_dir / f'episode_{episode}_topdown.png')
                state, success, target = attempt(state, first)
                state, second_success, second_target = attempt(state, second)
                details.append({'episode': episode, 'query_object': obj['objectId'], 'pose': pose,
                                'first': first, 'first_target': target, 'first_success': success,
                                'second': second, 'second_target': second_target,
                                'second_success': second_success, 'before': before,
                                'after': snapshot(env.controller.last_event)})
                episode += 1
                if success:
                    found = True
                    break
            if found:
                break
    # Search a small number of bowl/countertop starts while carrying the Apple.
    # Navigation, pickup and teleport are explicitly logged setup, never dataset
    # transitions; the subsequent Put is a real nearest-target action attempt.
    if not any(a == ACTIONS.index('Put') and ok for a, ok in zip(data['types'], data['successes'])):
        initial = env.reset().metadata
        receptacles = sorted((o for o in initial['objects']
                              if o['receptacle'] and o['objectType'] in ('Bowl', 'CounterTop')),
                             key=lambda o: (o['objectType'] != 'Bowl', o['objectId']))[:3]
        class SetupEnv:
            def step(self, action, **kwargs):
                return env.controller.step(action=action, **kwargs)
        setup_env = SetupEnv()
        put_success = False
        for receptacle in receptacles:
            env.reset()
            response = env.controller.step(action='GetInteractablePoses', objectId=receptacle['objectId'],
                                           rotations=[0, 90, 180, 270], horizons=[0, 30],
                                           standings=[True])
            setup_calls += 1
            if not response.metadata['lastActionSuccess']:
                details.append({'put_pose_query': receptacle['objectId'], 'error': response.metadata['errorMessage']})
                continue
            for pose in (response.metadata['actionReturn'] or [])[:8]:
                state = encoder.encode(env.reset().metadata)
                preparation = []
                ready = True
                for action in ('MoveAhead', 'RotateLeft', 'LookDown', 'Open', 'Pickup'):
                    state, _, success, target, error, called = execute(setup_env, encoder, state, ACTIONS.index(action))
                    setup_calls += int(called)
                    preparation.append({'action': action, 'target': target, 'success': success,
                                        'error': error, 'unity_call': called})
                    if not success:
                        ready = False
                        break
                if not ready:
                    details.append({'put_pose_query': receptacle['objectId'], 'pose': pose,
                                    'preparation': preparation, 'error': 'hand preparation failed'})
                    break
                event = env.controller.step(action='TeleportFull', **pose)
                setup_calls += 1
                if not event.metadata['lastActionSuccess']:
                    details.append({'put_pose_query': receptacle['objectId'], 'pose': pose,
                                    'preparation': preparation, 'error': event.metadata['errorMessage']})
                    continue
                state = encoder.encode(event.metadata)
                before = snapshot(event)
                if event.frame is not None:
                    Image.fromarray(event.frame).save(output_dir / f'episode_{episode}_ego.png')
                if event.third_party_camera_frames:
                    Image.fromarray(event.third_party_camera_frames[-1]).save(output_dir / f'episode_{episode}_topdown.png')
                state, success, target = attempt(state, 'Put')
                details.append({'episode': episode, 'kind': 'held_apple_put',
                                'put_pose_query': receptacle['objectId'], 'pose': pose,
                                'preparation': preparation, 'target': target, 'success': success,
                                'before': before, 'after': snapshot(env.controller.last_event)})
                episode += 1
                if success:
                    put_success = True
                    break
            if put_success:
                break
    (output_dir / 'setup.json').write_text(json.dumps({'setup_calls': setup_calls, 'episodes': details}, indent=2))
    dtypes = {'states': np.float32, 'actions': np.float32, 'next_states': np.float32,
              'types': np.int64, 'successes': bool, 'targets': str, 'unity_calls': bool,
              'episodes': np.int64, 'errors': str}
    return {key: np.asarray(values, dtype=dtypes[key]) for key, values in data.items()}
