from types import SimpleNamespace

from exploration.cee_us.thor_instance_validation import Setup, navigation_poses, sources_for


def test_source_priority():
    objects = [dict(objectId=k, objectType=k, pickupable=True) for k in ('Apple', 'Mug', 'Bread', 'Pot')]
    assert sources_for(objects, dict(objectId='machine', objectType='CoffeeMachine'))[0]['objectType'] == 'Mug'
    assert sources_for(objects, dict(objectId='toaster', objectType='Toaster'))[0]['objectType'] == 'Bread'
    assert sources_for(objects, dict(objectId='stove', objectType='StoveBurner'))[0]['objectType'] == 'Pot'


def test_navigation_grid_both_directions():
    positions = [dict(x=0, y=0, z=0), dict(x=0, y=0, z=.25)]
    free = list(navigation_poses(positions, 'MoveAhead', False))
    blocked = list(navigation_poses(positions, 'MoveBack', True))
    assert any(p['rotation'] == 0 and p['z'] == 0 for p in free)
    assert any(p['rotation'] == 0 and p['z'] == 0 for p in blocked)


def test_pose_options_and_cache_invalidation(tmp_path):
    calls = []
    class Controller:
        def step(self, action, **kwargs):
            calls.append((action, kwargs))
            return SimpleNamespace(metadata={'lastActionSuccess': True, 'errorMessage': '', 'actionReturn': []})
    class Env:
        controller = Controller()
        def reset(self):
            return None
    setup = Setup(Env(), tmp_path)
    setup.poses('x')
    assert calls[0][1]['horizons'] == [-30, 0, 30, 60]
    assert calls[0][1]['standings'] == [True, False]
    setup.step('OpenObject', objectId='x', forceAction=False)
    setup.poses('x')
    assert len(calls) == 3
    setup.reset()
    setup.poses('x')
    assert len(calls) == 4


def test_parent_order(tmp_path):
    objects = [dict(objectId='inside', parentReceptacles=['inner'], openable=False),
               dict(objectId='inner', parentReceptacles=['outer'], openable=True, isOpen=False),
               dict(objectId='outer', parentReceptacles=None, openable=True, isOpen=False)]
    class Controller:
        last_event = SimpleNamespace(metadata={'objects': objects})
        def step(self, action, **kwargs):
            return SimpleNamespace(metadata={'lastActionSuccess': True, 'errorMessage': ''})
    setup = Setup(SimpleNamespace(controller=Controller()), tmp_path)
    opened = []
    setup.place = lambda encoder, target, action, max_poses, offset: ({}, None)
    setup.step = lambda action, **kwargs: (opened.append(kwargs['objectId']) or SimpleNamespace(metadata={'lastActionSuccess': True, 'errorMessage': ''}))
    assert setup.parents(None, 'inside', 100, 0) is None
    assert opened == ['outer', 'inner']


def test_pose_repetitions_visit_distinct_locations_first():
    from exploration.cee_us.thor_instance_validation import ordered_poses
    poses = [dict(x=x, z=0, horizon=h) for x in (0, 1) for h in (0, 30, 60)]
    ordered = ordered_poses(poses, dict(x=0, z=0))
    assert [p['x'] for p in ordered[:2]] == [0, 1]
    assert len(ordered) == len(poses)


def test_attempt_cap_keeps_real_continuity_and_reports_incomplete(tmp_path):
    import runpy
    from pathlib import Path
    import numpy as np
    from exploration.cee_us.thor_instance_validation import collect
    from exploration.cee_us.thor_state import ThorState
    fixtures = runpy.run_path(str(Path(__file__).with_name('test_thor_designed_validation.py')))
    metadata = fixtures['metadata']()
    for obj in metadata['objects']:
        obj['openable'] = False
    metadata['objects'][0]['openable'] = True
    metadata['objects'][0]['isOpen'] = False

    class Env(fixtures['FakeEnv']):
        def step(self, action, **kwargs):
            super().step(action, **kwargs)
            if action in ('OpenObject', 'CloseObject'):
                obj = next(o for o in self.m['objects'] if o['objectId'] == kwargs['objectId'])
                obj['isOpen'] = action == 'OpenObject'
                self.last_event = fixtures['Event'](self.m)
            return self.last_event

    result = collect(Env(metadata), ThorState(metadata), metadata, tmp_path,
                     min_count=2, max_attempts=2, max_poses=10, seed=0, case_limit=1)
    assert result['attempts'] == 2 and not result['complete']
    with np.load(tmp_path / 'evaluation.npz', allow_pickle=False) as data:
        assert data['episodes'].tolist() == [0, 0]
        np.testing.assert_array_equal(data['next_states'][0], data['states'][1])


def test_native_idempotent_controls_and_default(tmp_path):
    import json
    import runpy
    from pathlib import Path
    import numpy as np
    from exploration.cee_us.thor_instance_validation import collect
    from exploration.cee_us.thor_state import ThorState
    fixtures = runpy.run_path(str(Path(__file__).with_name('test_thor_designed_validation.py')))
    metadata = fixtures['metadata']()
    for obj in metadata['objects']:
        obj['openable'] = False
    metadata['objects'][0].update(openable=True, isOpen=False)

    class Env(fixtures['FakeEnv']):
        def step(self, action, **kwargs):
            before = self.m['objects'][0]['isOpen']
            super().step(action, **kwargs)
            if action in ('OpenObject', 'CloseObject'):
                desired = action == 'OpenObject'
                self.m['lastActionSuccess'] = before != desired
                self.m['errorMessage'] = '' if before != desired else 'already in state'
                self.m['objects'][0]['isOpen'] = desired
                self.last_event = fixtures['Event'](self.m)
            return self.last_event

    for enabled in (False, True):
        output = tmp_path / str(enabled)
        output.mkdir()
        env = Env(metadata)
        result = collect(env, ThorState(metadata), metadata, output, 1, 10, 10, 0,
                         case_limit=1, failure_controls=enabled)
        rows = [json.loads(x) for x in (output / 'rows.jsonl').read_text().splitlines()]
        assert [r['action'] for r in rows] == (['Open', 'Open', 'Close', 'Close'] if enabled else ['Open', 'Close'])
        assert len([a for a, _ in env.calls if a in ('OpenObject', 'CloseObject')]) == len(rows)
        assert not result['complete']
        if enabled:
            assert [r['control'] for r in rows] == [None, 'already_open', None, 'already_closed']
            assert [r['success'] for r in rows] == [True, False, True, False]
            assert result['failure_controls']['already_open']['failures'] == 1
            assert result['failure_controls']['already_closed']['successes'] == 0
            requirements = {r['action']: r for r in result['requirements'] if r['object_id'] == rows[0]['actual_target']}
            assert requirements['Open']['count'] == requirements['Close']['count'] == 1
        else:
            assert result['failure_controls'] == {}
        with np.load(output / 'evaluation.npz', allow_pickle=False) as data:
            assert len(set(data['episodes'])) == 1
            np.testing.assert_array_equal(data['states'][1:], data['next_states'][:-1])


def test_hand_occupied_and_empty_hand_are_native_attempts(tmp_path):
    import json
    import runpy
    from pathlib import Path
    from exploration.cee_us.thor_instance_validation import collect
    from exploration.cee_us.thor_state import ThorState
    fixtures = runpy.run_path(str(Path(__file__).with_name('test_thor_designed_validation.py')))
    metadata = fixtures['metadata']()
    for obj in metadata['objects']:
        obj.update(openable=False, toggleable=False, pickupable=obj['objectId'] == 'a',
                   receptacle=obj['objectId'] == 'z', isPickedUp=False)
    class Env(fixtures['FakeEnv']):
        def step(self, action, **kwargs):
            occupied = any(o['isPickedUp'] for o in self.m['objects'])
            super().step(action, **kwargs)
            if action == 'PickupObject' and occupied:
                self.m.update(lastActionSuccess=False, errorMessage='hand occupied')
                self.last_event = fixtures['Event'](self.m)
            return self.last_event
    env = Env(metadata)
    report = collect(env, ThorState(metadata), metadata, tmp_path, 1, 20, 10, 0,
                     case_limit=2, failure_controls=True)
    rows = [json.loads(x) for x in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    controls = [r for r in rows if r['control']]
    assert {r['control'] for r in controls} == {'hand_occupied', 'empty_hand'}
    assert all(not r['success'] and r['unity_call'] for r in controls)
    assert any(r['action'] == 'Put' and r['source'] == 'a' for r in rows)
    assert report['failure_controls']['empty_hand']['failures'] >= 1
    assert report['failure_controls']['hand_occupied']['failures'] >= 1


def test_unexpected_control_success_reported_without_effect_credit(tmp_path):
    import json
    import runpy
    from pathlib import Path
    from exploration.cee_us.thor_instance_validation import collect
    from exploration.cee_us.thor_state import ThorState
    fixtures = runpy.run_path(str(Path(__file__).with_name('test_thor_designed_validation.py')))
    metadata = fixtures['metadata']()
    for obj in metadata['objects']:
        obj.update(openable=False, isOpen=False)
    metadata['objects'][0]['openable'] = True
    class Env(fixtures['FakeEnv']):
        def step(self, action, **kwargs):
            super().step(action, **kwargs)
            if action == 'OpenObject':
                self.m['objects'][0]['isOpen'] = True
                self.last_event = fixtures['Event'](self.m)
            return self.last_event
    report = collect(Env(metadata), ThorState(metadata), metadata, tmp_path, 2, 2, 10, 0,
                     case_limit=1, failure_controls=True)
    rows = [json.loads(x) for x in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    assert rows[1]['control'] == 'already_open' and rows[1]['success']
    assert report['failure_controls']['already_open']['successes'] == 1
    req = next(r for r in report['requirements'] if r['action'] == 'Open' and r['object_id'] == rows[0]['actual_target'])
    assert req['count'] == 1 and not req['covered']


def test_control_target_drift_starts_new_episode(tmp_path, monkeypatch):
    import json
    import runpy
    from pathlib import Path
    import numpy as np
    from exploration.cee_us import thor_instance_validation as module
    from exploration.cee_us.thor_state import ThorState
    fixtures = runpy.run_path(str(Path(__file__).with_name('test_thor_designed_validation.py')))
    metadata = fixtures['metadata']()
    for obj in metadata['objects']:
        obj.update(openable=False, isOpen=False)
    metadata['objects'][0]['openable'] = True
    class Env(fixtures['FakeEnv']):
        def step(self, action, **kwargs):
            super().step(action, **kwargs)
            if action in ('OpenObject', 'CloseObject'):
                desired = action == 'OpenObject'
                obj = self.m['objects'][0]
                self.m['lastActionSuccess'] = obj['isOpen'] != desired
                obj['isOpen'] = desired
                self.last_event = fixtures['Event'](self.m)
            return self.last_event
    env = Env(metadata)
    selected = module._selected
    drift = [True]
    def selection(encoder, state, action):
        if action == 'Open' and env.m['objects'][0]['isOpen'] and drift[0]:
            drift[0] = False
            return 'another-object'
        return selected(encoder, state, action)
    monkeypatch.setattr(module, '_selected', selection)
    module.collect(env, ThorState(metadata), metadata, tmp_path, 1, 10, 10, 0,
                   case_limit=1, failure_controls=True)
    rows = [json.loads(x) for x in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    assert rows[1]['control'] == 'already_open'
    assert rows[1]['episode'] != rows[0]['episode']
    with np.load(tmp_path / 'evaluation.npz', allow_pickle=False) as data:
        for i in range(len(data['episodes']) - 1):
            if data['episodes'][i] == data['episodes'][i + 1]:
                np.testing.assert_array_equal(data['next_states'][i], data['states'][i + 1])


def test_covered_receptacle_still_attempts_empty_hand_control(tmp_path, monkeypatch):
    import json
    import runpy
    from pathlib import Path
    from exploration.cee_us import thor_instance_validation as module
    from exploration.cee_us.thor_state import ThorState
    fixtures = runpy.run_path(str(Path(__file__).with_name('test_thor_designed_validation.py')))
    metadata = fixtures['metadata']()
    for obj in metadata['objects']:
        obj.update(openable=False, toggleable=False, pickupable=obj['objectId'] == 'a',
                   receptacle=obj['objectId'] == 'z', isPickedUp=False)

    class CoveredLedger:
        def __init__(self, *args):
            pass
        def observe(self, *args):
            pass
        def report(self):
            return {'requirements': [], 'total': 0, 'covered': 0, 'complete': True, 'missing': []}

    monkeypatch.setattr(module, 'InstanceCoverage', CoveredLedger)
    report = module.collect(fixtures['FakeEnv'](metadata), ThorState(metadata), metadata,
                            tmp_path, 1, 20, 10, 0, case_limit=2, failure_controls=True)
    rows = [json.loads(x) for x in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    assert any(r['requested_target'] == 'z' and r['control'] == 'empty_hand' for r in rows)
    assert report['failure_controls']['empty_hand']['requested_objects']['z'] == 1


def test_seed_placements_and_target_filter(tmp_path):
    import json
    import pytest
    import runpy
    from pathlib import Path
    from exploration.cee_us.thor_instance_validation import collect, load_placements
    from exploration.cee_us.thor_state import ThorState
    pose = dict(x=0, y=0, z=0, rotation=0, horizon=0, standing=True)
    rows = [dict(action='Put', success=True, source='a', actual_target='z', setup_pose=pose),
            dict(action='Put', success=False, source='a', actual_target='z', setup_pose={}),
            dict(action='Pickup', success=True, source='a', actual_target='z', setup_pose=pose)]
    path = tmp_path / 'input.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert load_placements(path) == {('a', 'z'): pose}
    fixtures = runpy.run_path(str(Path(__file__).with_name('test_thor_designed_validation.py')))
    metadata = fixtures['metadata']()
    with pytest.raises(ValueError, match='Unknown targets: absent'):
        collect(fixtures['FakeEnv'](metadata), ThorState(metadata), metadata, tmp_path,
                1, 5, 2, 0, targets=['absent'])
    output = tmp_path / 'filtered'
    output.mkdir()
    result = collect(fixtures['FakeEnv'](metadata), ThorState(metadata), metadata, output,
                     1, 5, 2, 0, targets=['z'])
    assert not result['complete']
    assert all(r['case'][1] == 'z' for r in result['visited_cases'] + result['skipped_cases'])
    assert len(result['requirements']) > 1


def test_known_pose_fallback_revalidates_held_source_and_splits_episode(tmp_path):
    import json
    import runpy
    from pathlib import Path
    import numpy as np
    from exploration.cee_us.thor_instance_validation import collect
    from exploration.cee_us.thor_state import ThorState
    fixtures = runpy.run_path(str(Path(__file__).with_name('test_thor_designed_validation.py')))
    metadata = fixtures['metadata']()
    for obj in metadata['objects']:
        obj.update(openable=False, toggleable=False, pickupable=obj['objectId'] == 'a',
                   receptacle=obj['objectId'] == 'z', isPickedUp=False)
    pose = dict(x=0, y=0, z=0, rotation=0, horizon=0, standing=True)
    class Env(fixtures['FakeEnv']):
        lose_held = False
        def step(self, action, **kwargs):
            event = super().step(action, **kwargs)
            if action == 'PutObject' and self.lose_held:
                next(o for o in self.m['objects'] if o['objectId'] == 'a')['isPickedUp'] = False
                self.last_event = fixtures['Event'](self.m)
                event = self.last_event
            return event
    for lose in (False, True):
        output = tmp_path / str(lose)
        output.mkdir()
        env = Env(metadata)
        env.lose_held = lose
        collect(env, ThorState(metadata), metadata, output, 1, 2, 10, 0,
                reuse_placements={('a', 'z'): pose}, targets=['z'])
        rows = [json.loads(x) for x in (output / 'rows.jsonl').read_text().splitlines()]
        puts = [row for row in rows if row['action'] == 'Put']
        assert len(puts) == 2
        assert (puts[0]['setup_reset'] != puts[1]['setup_reset']) == lose
        if not lose:
            assert puts[0]['episode'] != puts[1]['episode']
            assert sum(action == 'TeleportFull' for action, _ in env.calls) >= 3
            with np.load(output / 'evaluation.npz', allow_pickle=False) as data:
                assert data['episodes'].tolist() == [0, 1]
