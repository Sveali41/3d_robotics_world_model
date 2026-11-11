import json
from types import SimpleNamespace

import numpy as np
import pytest

from exploration.diagnostics import teleop_wm_validation as m
from domain.robocasa.primitives import RAPS_COMPACT_ACTION_SCHEMA as SCHEMA


def segment(name, delta=(0, 0, 0), before=1, after=1, complete=True):
    start = np.zeros(67); start[32] = before
    end = start.copy(); end[18:21] = delta; end[32] = after
    return dict(name=name, start_state=start.tolist(), end_state=end.tolist(),
                start_step=0, end_step=3, complete=complete)


def test_extract_observed_parameters_and_vectors():
    segments = [segment('move_delta_ee_pose', (.02, -.03, .01)),
                segment('approach', (.04, .01, -.01), after=-1),
                segment('lift', (.001, 0, .04), -1, -1),
                segment('drop', (0, 0, -.03), -1, -1),
                segment('open_gripper', before=-1), segment('close_gripper', after=-1)]
    plan = m.extract_plan(segments, SCHEMA)
    for row in plan:
        decoded = SCHEMA.decode(row['action'])
        assert decoded.primitive_type == row['name']
        assert decoded.parameters == pytest.approx(row['parameters'])
    assert plan[1]['parameters']['x_forward'] == pytest.approx(.04)
    assert plan[3]['parameters']['z_dist'] == pytest.approx(.03)


@pytest.mark.parametrize('item, message', [
    (segment('lift', (.02, 0, .04)), 'vertical'),
    (segment('lift', (0, 0, -.04)), 'direction'),
    (segment('drop', (0, 0, -.04), 1, -1), 'preserved'),
    (segment('approach', (-.01, 0, 0), after=-1), 'positive'),
    (segment('approach', (.01, 0, 0)), 'closed'),
    (segment('open_gripper', (.02, 0, 0)), 'stationary'),
    (segment('close_gripper'), 'expected gripper'),
    (segment('move_delta_ee_pose', (9, 0, 0)), 'outside scale'),
    (segment('lift', complete=False), 'incomplete'),
])
def test_extract_rejects_unrepresentable_segments(item, message):
    with pytest.raises(ValueError, match=message):
        m.extract_plan([item], SCHEMA)


def test_extract_rejects_transient_gripper_change():
    s = segment('lift', (0, 0, .04))
    rows = [dict(step=i, next_state=segment('lift', after=(-1 if i == 2 else 1))['end_state'])
            for i in (1, 2, 3)]
    with pytest.raises(ValueError, match='throughout'):
        m.extract_plan([s], SCHEMA, transitions=rows)


def test_initial_physics_comparison_checks_tree_and_tolerance():
    m.verify_physics({'robot': [[1., 2.]]}, {'robot': [[1., 2.0000005]]})
    with pytest.raises(ValueError, match='physics state'):
        m.verify_physics({'robot': [[1., 2.]]}, {'robot': [[1., 2.001]]})
    with pytest.raises(ValueError, match='physics state'):
        m.verify_physics({'robot': [1.]}, {'object': [1.]})


def test_replay_stops_at_termination_and_initial_mismatch(tmp_path, monkeypatch):
    class Env:
        unwrapped = SimpleNamespace(get_state_dict=lambda: {'x': [0.]})
        def reset(self, **kw): pass
        def close(self): pass
    class Encoder:
        def __init__(self, *a, **kw): pass
        def contract(self): return {'dimension': 67}
    class Collector:
        def __init__(self, env, encoder, actor, directory):
            self.actor, self.directory, self.total_steps = actor, directory, 0
        def collect(self, count, policy, *, seed, stage, on_skill_start, on_skill_end):
            action = self.actor.action(None)
            on_skill_start(1, 1, None, action, np.zeros(67))
            self.total_steps = 1
            row = dict(state=[0]*67, next_state=[0]*67, action=action.tolist(), episode_id=1,
                       terminated=True, truncated=False, completed=False, duration_low_steps=1)
            (self.directory/'skills.jsonl').write_text(json.dumps(row)+'\n')
            (self.directory/'transitions.jsonl').write_text('{}\n')
            on_skill_end(row)
            raise AssertionError('termination must stop replay')
    r = dict(make_env=lambda **kw: (Env(), {}), OmegaConf=SimpleNamespace(to_container=lambda x, **kw: x),
             CabinetSkillStateEncoder=Encoder, SkillCollector=Collector,
             jsonable=lambda x: x,
             write_json=lambda p, x: p.write_text(json.dumps(x)), replay_arrays=lambda rows: {})
    spacing = []
    monkeypatch.setattr(m, 'build_env', lambda config, runtime, **kwargs: (
        spacing.append(kwargs['object_spacing_increase']) or Env(), {}))
    args = SimpleNamespace(output=tmp_path, max_steps=10)
    (tmp_path/'plan.json').write_text(json.dumps({'plan': [{'action': [1]*12}, {'action': [1]*12}]}))
    manifest = dict(checkpoint='x', checkpoint_sha256='hash', seed=1, initial_state=[0]*67,
                    initial_physics_state={'x': [0.]}, object_spacing_increase_m=.02)
    payload = dict(config={}, state_contract={'dimension': 67}, action_contract={})
    m.replay(args, manifest, payload, {}, r)
    assert spacing == [.02]
    result = json.loads((tmp_path/'replay/manifest.json').read_text())
    assert result['status'] == 'incomplete' and result['recorded_skills'] == 1
    assert not (tmp_path/'replay/evaluation.npz').exists()
    mismatch = tmp_path/'other'; mismatch.mkdir()
    (mismatch/'plan.json').write_text((tmp_path/'plan.json').read_text())
    args.output = mismatch
    manifest['initial_state'][0] = 1
    manifest.pop('object_spacing_increase_m')
    with pytest.raises(ValueError, match='initial encoded state'):
        m.replay(args, manifest, payload, {}, r)
    assert spacing == [.02, 0.0]
    assert not (mismatch/'replay/skills.jsonl').exists()
    assert not (mismatch/'replay/evaluation.npz').exists()


def test_completed_failure_is_saved_as_evaluation_endpoint(tmp_path):
    class Env:
        unwrapped = SimpleNamespace(get_state_dict=lambda: {'x': [0.]})
        def reset(self, **kw): pass
        def close(self): pass
    class Encoder:
        def __init__(self, *a, **kw): pass
        def contract(self): return {}
    class Collector:
        def __init__(self, env, encoder, actor, directory):
            self.actor, self.directory, self.total_steps = actor, directory, 2
        def collect(self, count, policy, *, seed, stage, on_skill_start, on_skill_end):
            action = self.actor.action(None)
            on_skill_start(1, 1, None, action, np.zeros(67))
            row = dict(state=[0]*67, next_state=[1]*67, action=action.tolist(), episode_id=1,
                       terminated=False, truncated=False, completed=True, duration_low_steps=2,
                       status='no_stable_grasp')
            (self.directory/'skills.jsonl').write_text(json.dumps(row)+'\n')
            on_skill_end(row)
            raise AssertionError('last skill must stop replay')
    from exploration.cee_us.cee_us_cabinet import replay_arrays
    r = dict(make_env=lambda **kw: (Env(), {}), OmegaConf=SimpleNamespace(to_container=lambda x, **kw: x),
             CabinetSkillStateEncoder=Encoder, SkillCollector=Collector, jsonable=lambda x: x,
             write_json=lambda p, x: p.write_text(json.dumps(x)), replay_arrays=replay_arrays)
    (tmp_path/'plan.json').write_text(json.dumps({'plan': [{'action': [1]*12}]}))
    manifest = dict(checkpoint='x', checkpoint_sha256='hash', seed=1, initial_state=[0]*67,
                    initial_physics_state={'x': [0.]})
    m.replay(SimpleNamespace(output=tmp_path, max_steps=10), manifest,
             dict(config={}, state_contract={}, action_contract={}), {}, r)
    with np.load(tmp_path/'replay/evaluation.npz') as data:
        assert data['states'].shape == (1, 67)
        assert data['next_states'][0, 0] == 1
        assert data['duration_low_steps'].tolist() == [2]
    assert json.loads((tmp_path/'replay/skills.jsonl').read_text())['status'] == 'no_stable_grasp'


def test_mouse_delta_action_converts_world_to_root_and_limits_norm():
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    action = m.mouse_delta_action([0, .01, 0], [0, 0, 0], rotation, [.1]*3, .2, -1)
    assert action == pytest.approx([.1, 0, 0, -1])
    action = m.mouse_delta_action([1, 1, 0], [0, 0, 0], np.eye(3), [.1]*3, .2, 1)
    assert np.linalg.norm(action[:3]) == pytest.approx(.2)
    assert action[:3] == pytest.approx([.2 / np.sqrt(2), .2 / np.sqrt(2), 0])
    with pytest.raises(ValueError, match='Invalid mouse'):
        m.mouse_delta_action([np.nan, 0, 0], [0, 0, 0], np.eye(3), [.1]*3, .2, 1)


@pytest.mark.parametrize('max_steps', [10, None])
def test_mouse_record_uses_shared_segment_and_tick_path(tmp_path, monkeypatch, max_steps):
    monkeypatch.setattr(m, 'MANUAL_OBJECT_SPACING_INCREASE_M', 0.0)
    class Env:
        def __init__(self):
            self.unwrapped = self
            self.agent = SimpleNamespace(tcp=SimpleNamespace(pose=SimpleNamespace(p=np.zeros(3))))
            self.actions = []
        def reset(self, **kwargs): pass
        def get_state_dict(self): return {'x': [0.]}
        def close(self): pass
        def step(self, action):
            self.actions.append(action.copy())
            rec = SimpleNamespace(duration_low_steps=1, low_actions=[action], full_controller_actions=[action],
                                  status='ok', contact_events=[], outcome_metrics={})
            return None, None, False, False, {'primitive_record': rec}
    env = Env()
    class Encoder:
        def __init__(self, *args, **kwargs):
            self.controller = SimpleNamespace(config=SimpleNamespace(pos_lower=-.1, pos_upper=.1))
            self._root_rotation = np.eye(3)
            self.gripper_command = 1.
        def contract(self): return {}
        def encode(self):
            state = np.zeros(67); state[32] = self.gripper_command
            return state
    class UI:
        target = np.array([.01, 0, 0])
        active = None
        following = False
        step = 0
        def __init__(self): self.commands = iter(['begin move_delta_ee_pose', 'mouse', 'c', 'end', 'quit'])
        def read(self): return next(self.commands)
        def reset_target(self): pass
    monkeypatch.setattr(m, 'mouse_ui', lambda *args: UI())
    monkeypatch.setattr(m, 'render_viewer', lambda raw: None)
    jsonable = lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    def convert(x):
        return json.loads(json.dumps(x, default=jsonable))
    r = dict(make_env=lambda **kw: (env, {}), OmegaConf=SimpleNamespace(to_container=lambda x, **kw: x),
             CabinetSkillStateEncoder=Encoder, jsonable=convert,
             write_json=lambda p, x: p.write_text(json.dumps(convert(x))))
    args = SimpleNamespace(output=tmp_path/'record', checkpoint=tmp_path/'checkpoint', seed=1,
                           max_steps=max_steps, speed=.2, headless=False)
    m.record(args, dict(config={}, state_contract={}, action_contract={}), {}, 'hash', r)
    manifest = json.loads((args.output/'demo.json').read_text())
    rows = [json.loads(line) for line in (args.output/'demo_transitions.jsonl').read_text().splitlines()]
    assert manifest['status'] == 'complete' and manifest['input_mode'] == 'mouse'
    assert manifest['segments'][0]['complete'] and manifest['control_ticks'] == 2
    assert [row['command'] for row in rows] == ['mouse', 'c']
    assert rows[0]['target_world'] == pytest.approx([.01, 0, 0])
    assert env.actions[0] == pytest.approx([.1, 0, 0, 1])
    assert env.actions[1] == pytest.approx([0, 0, 0, -1])


def test_manual_env_removes_all_nested_episode_limits():
    raw = SimpleNamespace()
    time_limit = SimpleNamespace(env=raw, _max_episode_steps=1000)
    wrapper = SimpleNamespace(env=time_limit, max_ee_steps_per_episode=800,
                              max_low_level_steps_per_episode=1000)
    r = dict(make_env=lambda **kw: (SimpleNamespace(env=wrapper), {}),
             OmegaConf=SimpleNamespace(to_container=lambda x, **kw: dict(x)))
    _, config = m.build_env({}, r)
    assert wrapper.max_ee_steps_per_episode == float('inf')
    assert wrapper.max_low_level_steps_per_episode == float('inf')
    assert time_limit._max_episode_steps == float('inf')
    assert config['max_ee_steps_per_episode'] is None
    assert config['max_low_level_steps_per_episode'] is None


def test_manual_object_spacing_is_optional_copied_and_non_cumulative():
    cup = np.array([.45, -.06, .75], dtype=np.float32)
    bowl = np.array([.45, .06, .71], dtype=np.float32)
    other = np.array([.2, .3, .4], dtype=np.float32)
    raw = SimpleNamespace(
        cabinet_items=[SimpleNamespace(name=name) for name in
                       ('cabinet_item_16', 'cabinet_item_17', 'cabinet_item_03')],
        _item_spawn_positions=[cup, bowl, other])
    env = SimpleNamespace(unwrapped=raw)
    r = dict(make_env=lambda **kw: (env, {}),
             OmegaConf=SimpleNamespace(to_container=lambda x, **kw: dict(x)))

    _, baseline = m.build_env({}, r)
    assert raw._item_spawn_positions[0] is cup
    assert raw._item_spawn_positions[1] is bowl
    assert baseline['object_spacing_increase_m'] == 0.0

    _, environment = m.build_env({}, r, object_spacing_increase=.02)
    assert raw._item_spawn_positions[0] is not cup
    assert raw._item_spawn_positions[1] is not bowl
    assert raw._item_spawn_positions[0] == pytest.approx([.45, -.07, .75])
    assert raw._item_spawn_positions[1] == pytest.approx([.45, .07, .71])
    assert raw._item_spawn_positions[2] == pytest.approx(other)
    assert cup == pytest.approx([.45, -.06, .75])
    assert bowl == pytest.approx([.45, .06, .71])
    assert environment['object_spacing_increase_m'] == .02


@pytest.mark.parametrize('value', [-.01, np.nan, np.inf])
def test_manual_object_spacing_rejects_invalid_increase(value):
    r = dict(make_env=lambda **kw: (SimpleNamespace(), {}),
             OmegaConf=SimpleNamespace(to_container=lambda x, **kw: dict(x)))
    with pytest.raises(ValueError, match='finite and nonnegative'):
        m.build_env({}, r, object_spacing_increase=value)


def test_joint_target_action_validates_all_seven_targets():
    limits = np.tile([-2., 2.], (7, 1))
    targets = np.linspace(-1.5, 1.5, 7)
    assert m.joint_target_action(targets, limits, -1) == pytest.approx([*targets, -1])
    for i in range(7):
        for value in (-2.1, 2.1, np.nan, np.inf):
            bad = targets.copy(); bad[i] = value
            with pytest.raises(ValueError, match='Joint targets'):
                m.joint_target_action(bad, limits, 1)
    with pytest.raises(ValueError, match='Joint targets'):
        m.joint_target_action(targets[:6], limits, 1)


def test_joint_record_keeps_targets_for_gripper_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'MANUAL_OBJECT_SPACING_INCREASE_M', 0.0)
    targets = np.linspace(-.6, .6, 7)
    limits = np.tile([-2., 2.], (7, 1))
    names = [f'panda_joint{i}' for i in range(7)]
    class Env:
        unwrapped = None
        def __init__(self): self.unwrapped = self; self.actions = []
        def reset(self, **kwargs): pass
        def get_state_dict(self): return {}
        def close(self): pass
        def step(self, action):
            self.actions.append(action.copy())
            rec = SimpleNamespace(duration_low_steps=1, low_actions=[action], full_controller_actions=[action],
                                  status='ok', contact_events=[], outcome_metrics={})
            return None, None, False, False, {'primitive_record': rec}
    env = Env()
    class Encoder:
        def __init__(self, *a, **kw): self.gripper_command = 1.
        def contract(self): return {}
        def encode(self):
            state = np.zeros(67); state[32] = self.gripper_command
            return state
    class UI:
        joint_targets = targets
        active = None; following = False; step = 0
        def __init__(self): self.commands = iter(['begin joint_position', 'mouse', 'c', 'o', 'end', 'quit'])
        def read(self): return next(self.commands)
        def reset_target(self): pass
    monkeypatch.setattr(m, 'joint_details', lambda raw: (names, limits))
    monkeypatch.setattr(m, 'mouse_ui', lambda *a, **kw: UI())
    monkeypatch.setattr(m, 'render_viewer', lambda raw: None)
    def convert(x): return json.loads(json.dumps(x, default=lambda y: y.tolist()))
    modes = []
    def make_env(**kw): modes.append(kw['action_mode']); return env, {}
    r = dict(make_env=make_env, OmegaConf=SimpleNamespace(to_container=lambda x, **kw: x),
             CabinetSkillStateEncoder=Encoder, jsonable=convert,
             write_json=lambda p, x: p.write_text(json.dumps(convert(x))))
    args = SimpleNamespace(output=tmp_path/'record', checkpoint=tmp_path/'checkpoint', seed=1,
                           max_steps=None, speed=.2, headless=False, control_mode='joints')
    m.record(args, dict(config={}, state_contract={}, action_contract={'compact': 6}), {}, 'hash', r)
    manifest = json.loads((args.output/'demo.json').read_text())
    rows = [json.loads(line) for line in (args.output/'demo_transitions.jsonl').read_text().splitlines()]
    assert modes == ['joint_position']
    assert manifest['source_action_contract'] == {'compact': 6}
    assert manifest['action_contract']['joint_names'] == names
    assert manifest['action_contract']['dimension'] == 8
    assert manifest['segments'][0]['name'] == 'joint_position'
    assert [a[-1] for a in env.actions] == [1, -1, 1]
    for action, row in zip(env.actions, rows):
        assert action[:7] == pytest.approx(targets)
        assert row['joint_targets_rad'] == pytest.approx(targets)
        assert row['action_mode'] == 'joint_position'


@pytest.mark.parametrize('stage', ['extract', 'replay'])
def test_joint_manifest_rejected_before_output_or_checkpoint(tmp_path, monkeypatch, stage):
    (tmp_path/'demo.json').write_text(json.dumps({'control_mode': 'joints'}))
    monkeypatch.setattr(m, 'runtime', lambda: {})
    with pytest.raises(ValueError, match='cannot be extracted or replayed'):
        m.main(['--stage', stage, '--output', str(tmp_path)])
    assert not (tmp_path/'plan.json').exists()
    assert not (tmp_path/'replay').exists()
    if stage == 'replay':
        with pytest.raises(ValueError, match='cannot be extracted or replayed'):
            m.replay(SimpleNamespace(output=tmp_path), {'control_mode': 'joints'}, {}, {}, {})


def test_world_pose_to_root_preserves_rotation_and_rejects_invalid():
    from scipy.spatial.transform import Rotation
    root = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    target = np.eye(4)
    target[:3, :3] = root @ Rotation.from_euler('x', 90, degrees=True).as_matrix()
    target[:3, 3] = [1, 3, 3]
    position, quaternion = m.world_pose_to_root(target, [1, 2, 3], root)
    assert position == pytest.approx([1, 0, 0], abs=1e-7)
    assert quaternion == pytest.approx([2**-.5, 2**-.5, 0, 0], abs=1e-7)
    target[0, 0] = np.nan
    with pytest.raises(ValueError, match='rigid'):
        m.world_pose_to_root(target, [1, 2, 3], root)


@pytest.mark.parametrize('ending', ['complete', 'terminated', 'budget_truncated', 'viewer_closed'])
def test_pose_requests_use_real_ticks_and_save_failures(tmp_path, monkeypatch, ending):
    monkeypatch.setattr(m, 'MANUAL_OBJECT_SPACING_INCREASE_M', 0.0)
    import domain.robocasa.motion_planning as planning
    names = [f'joint{i}' for i in range(7)]
    limits = np.tile([-2., 2.], (7, 1))
    targets = np.linspace(-.6, .6, 7)
    class Env:
        def __init__(self): self.unwrapped = self; self.actions = []; self.attempts = 0
        def reset(self, **kw): pass
        def close(self): pass
        def get_state_dict(self): return {}
        def plan_to_pose(self, *a, **kw):
            self.attempts += 1
            if self.attempts == 1: raise planning.MotionPlanningError('path_collision', 'blocked')
            return SimpleNamespace(metadata={'status': 'Success'}, position=np.zeros((2, 9)), time=np.array([0., .05]))
        def step(self, action):
            self.actions.append(action.copy())
            rec = SimpleNamespace(duration_low_steps=1, low_actions=[action], full_controller_actions=[action],
                                  status='ok', contact_events=[], outcome_metrics={})
            return None, None, ending == 'terminated', False, {'primitive_record': rec}
    env = Env()
    class Execution:
        def __init__(self, *a, **kw): self.status = 'running'; self.records = []
        def next_action(self): return np.array([*targets, 1.], dtype=np.float32)
        def record_step(self, **kw):
            self.records.append({'feedback': len(self.records) + 1})
            if len(self.records) == 2: self.status = 'success'
            return self.records[-1]
        def stop(self, status): self.status = status
        def result(self): return {'status': self.status, 'ticks': self.records}
    class Encoder:
        def __init__(self, *a, **kw):
            self.gripper_command = 1.; self._root_rotation = np.eye(3); self._root_position = np.zeros(3)
        def contract(self): return {}
        def encode(self):
            state = np.zeros(67); state[32] = self.gripper_command; return state
    class UI:
        target_matrix = np.eye(4); contacts = ['none', 'cup', 'bowl']; contact_selected = 0
        active = None; following = False; step = 0; busy = False; status = ''
        def __init__(self):
            self.commands = iter(['begin pose_control', 'plan', 'plan', 'mouse', 'mouse', 'c'] +
                                 ['mouse'] * 30 + ['end', 'quit'])
        def read(self):
            command = next(self.commands)
            if ending == 'viewer_closed' and command == 'mouse' and len(env.actions) == 1:
                raise EOFError()
            return command
        def reset_target(self): pass
    monkeypatch.setattr(m, 'validate_pose_runtime', lambda: None)
    monkeypatch.setattr(m, 'joint_details', lambda raw: (names, limits))
    monkeypatch.setattr(m, 'joint_angles', lambda raw: targets.copy())
    monkeypatch.setattr(m, 'mouse_ui', lambda *a, **kw: UI())
    monkeypatch.setattr(m, 'render_viewer', lambda raw: None)
    monkeypatch.setattr(planning, 'MotionPlanExecution', Execution)
    def convert(x): return json.loads(json.dumps(x, default=lambda y: y.tolist()))
    r = dict(make_env=lambda **kw: (env, {}), OmegaConf=SimpleNamespace(to_container=lambda x, **kw: x),
             CabinetSkillStateEncoder=Encoder, jsonable=convert,
             write_json=lambda p, x: p.write_text(json.dumps(convert(x))))
    args = SimpleNamespace(output=tmp_path/'record', checkpoint=tmp_path/'checkpoint', seed=1,
                           max_steps=1 if ending == 'budget_truncated' else None, speed=.2, headless=False, control_mode='pose')
    m.record(args, dict(config={}, state_contract={}, action_contract={}), {}, 'hash', r)
    requests = [json.loads(line) for line in (args.output/'pose_requests.jsonl').read_text().splitlines()]
    rows = [json.loads(line) for line in (args.output/'demo_transitions.jsonl').read_text().splitlines()]
    if ending != 'complete':
        assert [x['status'] for x in requests] == ['path_collision', ending]
        assert requests[1]['physical_ticks'] == 1 and len(rows) == 1
        assert requests[1]['execution_result']['status'] == ending
        return
    assert [x['status'] for x in requests] == ['path_collision', 'success', 'success']
    assert [x['physical_ticks'] for x in requests] == [0, 2, 30]
    assert len(rows) == len(env.actions) == 32 and rows[0]['request_id'] == 2
    assert rows[0]['planner_feedback'] == {'feedback': 1}
    assert all(a[:7] == pytest.approx(targets) and a[-1] == -1 for a in env.actions[2:])
    assert (args.output/'pose_plan_0002.npz').exists()
    with pytest.raises(ValueError, match='cannot be extracted or replayed'):
        m.reject_joint_manifest({'control_mode': 'pose'})
