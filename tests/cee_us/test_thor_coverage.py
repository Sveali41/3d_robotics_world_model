import copy

from exploration.cee_us.thor_state import CAPABILITIES, DYNAMIC_FLAGS, ThorState
from exploration.cee_us.thor_coverage import InstanceCoverage


def metadata():
    def item(name, x, **cap):
        obj = {key: False for key in (*CAPABILITIES, *DYNAMIC_FLAGS)}
        obj.update(objectId=name, objectType='Cup', position={'x': x, 'y': 0, 'z': 0},
                   rotation={'x': 0, 'y': 0, 'z': 0}, openness=0., fillLiquid=None,
                   temperature='RoomTemp', parentReceptacles=None, mass=1.,
                   axisAlignedBoundingBox={'size': {'x': 1, 'y': 1, 'z': 1}}, visible=True)
        obj.update(cap)
        return obj
    return {'agent': {'position': {'x': 0, 'y': 0, 'z': 0}, 'rotation': {'y': 0},
                      'cameraHorizon': 0, 'isStanding': True}, 'lastActionSuccess': True,
            'objects': [item('cup', 1, pickupable=True),
                        item('cabinet', 2, pickupable=True, receptacle=True,
                             openable=True, toggleable=True)]}


def transition(encoder, original, mutate):
    changed = copy.deepcopy(original)
    mutate(changed)
    return encoder.encode(original), encoder.encode(changed)


def by_id(coverage, requirement_id):
    return next(item for item in coverage.report()['requirements']
                if item['id'] == requirement_id)


def test_interactions_require_matching_target_and_intended_effect():
    initial = metadata(); encoder = ThorState(initial); coverage = InstanceCoverage(encoder, initial)
    before, after = transition(encoder, initial,
                               lambda m: m['objects'][0].update(isOpen=True))
    coverage.observe(before, after, 'Open', 'cabinet', True)
    assert by_id(coverage, 'Open:cabinet')['count'] == 0
    coverage.observe(before, after, 'Open', 'cup', True)
    assert by_id(coverage, 'Open:cabinet')['count'] == 0
    assert by_id(coverage, 'Open:cabinet')['attempts'] == 1

    relevant_before, relevant_after = transition(
        encoder, initial, lambda m: m['objects'][1].update(isOpen=True))
    coverage.observe(relevant_before, relevant_after, 'Open', 'cabinet', True)
    assert by_id(coverage, 'Open:cabinet')['count'] == 1


def test_repeated_idempotent_actions_are_attempts_without_effects():
    initial = metadata(); encoder = ThorState(initial); coverage = InstanceCoverage(encoder, initial)
    state = encoder.encode(initial)
    for _ in range(3):
        coverage.observe(state, state, 'Open', 'cabinet', True)
    item = by_id(coverage, 'Open:cabinet')
    assert item['attempts'] == 3 and item['count'] == 0 and not item['covered']


def test_put_tracks_released_source_separately_from_receptacle_target():
    initial = metadata()
    initial['objects'][0]['isPickedUp'] = True
    encoder = ThorState(initial); coverage = InstanceCoverage(encoder, initial)

    def put_in_cabinet(m):
        m['objects'][0]['isPickedUp'] = False
        m['objects'][0]['parentReceptacles'] = ['cabinet']

    before, after = transition(encoder, initial, put_in_cabinet)
    coverage.observe(before, after, 'Put', 'cabinet', True)
    release = by_id(coverage, 'Release:cup')
    placed = by_id(coverage, 'Put:cabinet')
    assert release['count'] == 1 and release['attempts'] == 1
    assert placed['count'] == 1 and placed['attempts'] == 1

    coverage.observe(before, after, 'Put', 'cup', True)
    assert by_id(coverage, 'Put:cabinet')['attempts'] == 1


def test_failed_interactions_never_cover_even_when_state_changes():
    initial = metadata(); encoder = ThorState(initial); coverage = InstanceCoverage(encoder, initial)
    before, after = transition(encoder, initial,
                               lambda m: m['objects'][1].update(isToggled=True))
    coverage.observe(before, after, 'ToggleOn', 'cabinet', False)
    item = by_id(coverage, 'ToggleOn:cabinet')
    assert item['attempts'] == 1 and item['count'] == 0 and not item['covered']


def test_navigation_requires_action_specific_success_or_blocked_effect():
    initial = metadata(); encoder = ThorState(initial); coverage = InstanceCoverage(encoder, initial)
    state = encoder.encode(initial)
    blocked_id = 'navigation:MoveAhead:failure with no position movement <= 1e-4 m'
    moved_id = 'navigation:MoveAhead:success with position movement > 1e-4 m'

    coverage.observe(state, state, 'MoveAhead', None, False)
    assert by_id(coverage, blocked_id)['count'] == 1
    coverage.observe(state, state, 'MoveAhead', None, True)
    assert by_id(coverage, moved_id)['count'] == 0

    moved = state.copy(); moved[0] += 0.25
    coverage.observe(state, moved, 'MoveAhead', None, True)
    assert by_id(coverage, moved_id)['count'] == 1
    coverage.observe(state, moved, 'MoveAhead', None, False)
    assert by_id(coverage, moved_id)['count'] == 1
    assert by_id(coverage, blocked_id)['count'] == 1
