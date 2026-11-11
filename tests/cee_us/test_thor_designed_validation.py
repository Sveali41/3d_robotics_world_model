import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from exploration.cee_us import thor_designed_validation as module
from exploration.cee_us.thor_state import ThorState
from exploration.cee_us.cee_us_thor import check_scene_contract

metadata = runpy.run_path(str(Path(__file__).with_name('test_thor_world_model.py')))['metadata']


def test_plan_exhausts_and_interleaves_capabilities():
    m = metadata()
    m['objects'][0].update(objectType='Apple', openable=True, toggleable=True)
    cases = module.plan_cases(m)
    for family, cap in [('pickupable', 'pickupable'), ('openable', 'openable'),
                        ('toggleable', 'toggleable'), ('receptacle', 'receptacle')]:
        assert {c['target'] for c in cases if c['family'] == family} == {o['objectId'] for o in m['objects'] if o[cap]}
    assert {c['family'] for c in cases[:6]} == {'openable', 'toggleable', 'pickupable', 'receptacle', 'navigation', 'no_target'}
    assert all(c['source'] != c['target'] for c in cases if c['family'] == 'receptacle')


class Event:
    def __init__(self, m):
        self.metadata = copy.deepcopy(m)
        self.frame = None
        self.third_party_camera_frames = []


class FakeEnv:
    def __init__(self, m):
        self.initial = copy.deepcopy(m)
        self.controller = self
        self.calls = []
        self.reset()

    def reset(self):
        self.m = copy.deepcopy(self.initial)
        self.m['errorMessage'] = ''
        self.last_event = Event(self.m)
        return self.last_event

    def step(self, action, **kwargs):
        self.calls.append((action, kwargs))
        self.m.update(lastActionSuccess=True, errorMessage='')
        if action == 'GetInteractablePoses':
            self.m['actionReturn'] = [dict(x=0, y=0, z=0, rotation=0, horizon=0, standing=True)]
        elif action == 'PickupObject':
            next(o for o in self.m['objects'] if o['objectId'] == kwargs['objectId'])['isPickedUp'] = True
        elif action == 'PutObject':
            self.m.update(lastActionSuccess=False, errorMessage='simulator failure')
        self.last_event = Event(self.m)
        return self.last_event


def test_put_preparation_attempt_budget_and_contiguous_transitions(tmp_path):
    m = metadata()
    m['objects'][1]['objectType'] = 'Apple'
    m['objects'][0]['position']['x'] = 2
    encoder = ThorState(m)
    env = FakeEnv(m)
    cases = [dict(case_id=0, family='receptacle', target='z', source='a', actions=('Put', 'Put', 'Pickup', 'Put')),
             dict(case_id=1, family='default', actions=('MoveAhead',) * 4)]
    data, coverage = module.collect(env, encoder, cases, tmp_path, 5, 2, 0)
    assert len(data['types']) == 4 and coverage['pending_cases'] == cases[1:]
    assert coverage['setup_resets'] == 1 and coverage['setup_direct_unity_calls'] == 5
    assert len(coverage['visited_cases']) == 1
    assert data['targets'][0] == 'z' and not data['successes'][0]
    assert data['errors'][0] == 'simulator failure'
    np.testing.assert_array_equal(data['states'][1:], data['next_states'][:-1])
    assert [x[0] for x in env.calls] == ['GetInteractablePoses', 'GetInteractablePoses', 'TeleportFull', 'PickupObject', 'TeleportFull', 'PutObject', 'PutObject', 'PickupObject', 'PutObject']
    put = next(kwargs for action, kwargs in env.calls if action == 'PutObject')
    assert put == {'objectId': 'z', 'forceAction': False, 'placeStationary': True}
    assert len((tmp_path / 'rows.jsonl').read_text().splitlines()) == 4


def test_no_target_failure_is_retained_and_case_limit_is_pending(tmp_path):
    m = metadata()
    for o in m['objects']:
        o['visible'] = False
    encoder = ThorState(m)
    env = FakeEnv(m)
    cases = [dict(case_id=0, family='no_target', actions=('Pickup', 'Put')),
             dict(case_id=1, family='default', actions=('MoveAhead',))]
    data, coverage = module.collect(env, encoder, cases, tmp_path, 20, 1, 0, case_limit=1)
    assert not data['unity_calls'].any()
    assert list(data['errors']) == ['no_visible_capable_target'] * 2
    assert len(coverage['pending_cases']) == 1
    assert [x[0] for x in env.calls] == ['TeleportFull']


def test_scene_contract_only_ignores_aabb():
    m = metadata(); encoder = ThorState(m)
    changed = copy.deepcopy(m)
    changed['objects'][0]['axisAlignedBoundingBox']['size']['x'] = 4
    check_scene_contract(encoder, ThorState(changed))
    changed['objects'][0]['mass'] = 3
    with pytest.raises(ValueError, match='contract'):
        check_scene_contract(encoder, ThorState(changed))
