"""Grasp-only policy and successful macro reset boundaries."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.raps.raps_gnn_actor import (
    DEFAULTS, RAPSImaginationActor, SkillCollector, action_contract,
    action_contract_matches, canonical_action, skill_arrays, skill_schema,
)


def settings(**overrides):
    return {**DEFAULTS, 'pick_place_primitives': True, 'grasp_only': True,
            'grasp_z_bounds_m': [.692, .772], 'grasp_target_heuristic': True,
            'motion_execution': 'planner', 'latched_gripper': True,
            'grasp_stop_on_contact': False, 'hidden_size': 8, 'actor_layers': 1,
            'critic_layers': 1, 'imagination_horizon': 2, 'imagination_batch_size': 4,
            'updates_per_round': 1, **overrides}


@pytest.mark.parametrize("agent_dim", [31, 33])
def test_intrinsic_excludes_command_and_counts_objects_once(agent_dim):
    backend = CEEUSBackend(agent_dim=agent_dim, action_dim=4, hidden_dim=8,
                           num_objects=1, object_names=('cabinet_item_16',))
    actor = RAPSImaginationActor(backend, settings(imagination_horizon=1))
    physical = agent_dim - 1

    def predict(*, observations, actions, states):
        predicted = observations.clone()
        offsets = torch.arange(5, dtype=predicted.dtype)[:, None]
        predicted[..., physical] += offsets * 1000  # Controller memory is not novelty.
        predicted[..., agent_dim] += offsets
        if physical == 32:
            predicted[..., 30] += offsets * 2
        return predicted, None

    backend.model.predict = predict
    trajectory = actor.imagine(torch.zeros(2, backend.state_dim))
    sigma = torch.arange(5, dtype=torch.float32).std(unbiased=True)
    expected = sigma * (3 if physical == 32 else 1)
    torch.testing.assert_close(trajectory['rewards'], expected.expand(1, 2, 1))
    torch.testing.assert_close(trajectory['components'][..., 2],
                               (sigma * (2 if physical == 32 else 0)).expand(1, 2))
    torch.testing.assert_close(trajectory['components'].sum(-1), trajectory['rewards'][..., 0])


def test_schema_contract_and_config():
    schema = skill_schema(settings())
    assert schema.primitive_names == ('grasp',) and schema.action_dim == 4
    for value, expected in [(-1., [.370, -.070, .692]), (1., [.671, .310, .772])]:
        decoded = schema.decode(np.array([1., value, value, value]))
        np.testing.assert_allclose(list(decoded.parameters.values()), expected)
    assert 'grasp_only' not in DEFAULTS
    dual = settings(); dual.pop('grasp_only')
    assert action_contract({**dual, 'grasp_only': False}) == action_contract(dual)
    contract = action_contract(settings())
    assert contract['action_dim'] == contract['pick_place']['action_dim'] == 4
    np.testing.assert_allclose(contract['pick_place']['tcp_bounds_root_m']['grasp'][2], [.692, .772])
    assert not action_contract_matches(action_contract(dual), settings())
    assert not action_contract_matches(contract, dual)
    base = OmegaConf.load('configs/raps_gnn_actor_cabinet_pick_place_target.yaml')
    grasp = OmegaConf.load('configs/raps_gnn_actor_cabinet_grasp_only.yaml')
    assert grasp.raps_actor.pop('grasp_only') is True
    assert grasp.pop('cabinet_door') is False
    assert grasp == base


@pytest.mark.parametrize("agent_dim", [31, 33])
def test_real_four_dimensional_actor_model_and_restore(tmp_path, agent_dim):
    torch.set_num_threads(1)
    backend = CEEUSBackend(agent_dim=agent_dim, action_dim=4, hidden_dim=8, batch_size=4, object_heuristic={})
    actor = RAPSImaginationActor(backend, settings())
    rng = np.random.default_rng(19)
    states = rng.normal(size=(8, backend.state_dim)).astype(np.float32)
    actions = canonical_action(torch.tensor(rng.uniform(-1, 1, (8, 4)), dtype=torch.float32), actor.schema).numpy()
    next_states = states.copy(); next_states[:, :backend.dynamic_dim] += .001
    data = dict(states=states, actions=actions, next_states=next_states, terminated=np.zeros(8, bool))
    backend.train(data, 1)
    metrics = actor.train(data)
    assert metrics['updates'] == 1 and np.isfinite(list(metrics.values())).all()
    sampled = actor.action(states[0])
    assert sampled.shape == (4,) and sampled[0] == 1.
    imagined = actor.imagine(torch.tensor(states[:4]))
    assert imagined['actions'].shape == (2, 4, 4)
    assert torch.all(imagined['actions'][..., 0] == 1.)
    assert torch.isfinite(imagined['rewards']).all()
    collector = SkillCollector.__new__(SkillCollector); collector.schema = actor.schema
    for _ in range(32):
        action = collector._propose_action('random')
        assert action.shape == (4,) and action[0] == 1.
        assert actor.schema.decode(action).primitive_type == 'grasp'
    saved = deepcopy(actor.state_dict()); restored = RAPSImaginationActor(backend, settings())
    restored.load_state_dict(saved)
    for key in ('actor', 'critic', 'target_critic', 'actor_optimizer', 'critic_optimizer'):
        torch.testing.assert_close(restored.state_dict()[key], saved[key])
    from exploration.raps.raps_gnn_actor import SCHEMA, load
    from exploration.cee_us.cee_us_cabinet import rng_state
    encoder = SimpleNamespace(contract=lambda: {'test_state_dim': backend.state_dim})
    payload = dict(schema=SCHEMA, backend=deepcopy(backend.state_dict()), actor=saved,
                   state_contract=encoder.contract(), action_contract=action_contract(actor.settings), rng=rng_state())
    checkpoint = tmp_path / 'grasp.pt'; torch.save(payload, checkpoint)
    assert load(checkpoint, backend, restored, encoder)['action_contract']['action_dim'] == 4
    payload['action_contract'] = action_contract(settings(grasp_only=False)); torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match='contract differs'):
        load(checkpoint, backend, restored, encoder)
    with pytest.raises(ValueError, match='settings differ'):
        actor.load_state_dict({**saved, 'settings': settings(grasp_only=False)})
    with pytest.raises(ValueError, match='fresh 4D'):
        RAPSImaginationActor(CEEUSBackend(agent_dim=33, action_dim=5, hidden_dim=8), settings())
    with pytest.raises(ValueError, match='requires pick_place'):
        RAPSImaginationActor(backend, settings(pick_place_primitives=False))
    for invalid in (1, 1.0, 'true'):
        with pytest.raises(ValueError, match='must be boolean'):
            RAPSImaginationActor(backend, settings(grasp_only=invalid))


def fake_collector(monkeypatch, tmp_path, outcomes):
    import exploration.raps.raps_gnn_actor as module
    class Execution:
        def __init__(self, **kwargs): self.reset()
        def reset(self):
            self.gripper_command = -1.; self.support_stopped_object_ids = []
            self.motion_attempts = []; self.last_motion_tick = {}
        def start(self, action, context):
            assert action.primitive_type == 'grasp'
            self.primitive = SimpleNamespace(status='running', phase='intermediate', done=False)
        def next_action(self, context): return np.r_[np.zeros(7), -1.].astype(np.float32)
        def record_action(self, *args, **kwargs): pass
        def check_retention(self, context):
            self.primitive.status = outcomes.pop(0); self.primitive.done = True
        def observe(self, context): pass
    class Env:
        action_mode = 'joint_position'
        def __init__(self):
            self.env = self; self.steps = 0; self.seeds = []; self.value = 0.; self.reset_options = []
        def reset(self, seed, options=None):
            self.reset_options.append(options)
            self.seeds.append(seed); self.value = -100. * len(self.seeds)
        def _context(self): return None
        def step(self, low):
            self.steps += 1; self.value += 1.
            record = SimpleNamespace(duration_low_steps=1, low_actions=[low], full_controller_actions=[low],
                                     status='success', contact_events=[], outcome_metrics={})
            return None, 0., False, False, {'primitive_record': record, 'exploration_reward': {}}
    monkeypatch.setattr(module, 'PlannedPickPlaceSkillExecution', Execution)
    # These lifecycle fixtures model an already homed robot. Homing has separate tests.
    monkeypatch.setattr(SkillCollector, '_capture_home', lambda self: None)
    monkeypatch.setattr(SkillCollector, '_home_actions', lambda self: iter(()))
    monkeypatch.setattr(module, 'snapshot', lambda env: {})
    env = Env(); encoder = SimpleNamespace(encode=lambda: np.full(67, env.value, np.float32))
    actor = SimpleNamespace(settings=settings(), action=lambda state: np.array([1., 0., 0., 0.], np.float32))
    return SkillCollector(env, encoder, actor, tmp_path)


def test_reconfigure_only_at_episode_resets_and_rebind_before_encoding(monkeypatch, tmp_path):
    collector = fake_collector(monkeypatch, tmp_path, ['unconfirmed_grasp', 'success', 'success'])
    collector.actor.settings['reconfigure_on_reset'] = True
    rebound = []
    collector.encoder.rebind_after_reset = lambda: rebound.append(len(collector.env.seeds))
    encode = collector.encoder.encode
    def checked_encode():
        assert rebound[-1] == len(collector.env.seeds)
        return encode()
    collector.encoder.encode = checked_encode
    _, skills = collector.collect(3, 'random', seed=5, stage='test')
    assert collector.env.seeds == [5, 6]
    assert collector.env.reset_options == [{'reconfigure': True}, {'reconfigure': True}]
    assert rebound == [1, 2]
    assert [row['episode_id'] for row in skills] == [1, 1, 2]
    np.testing.assert_array_equal(skills[0]['next_state'], skills[1]['state'])
    assert not action_contract_matches(action_contract(settings()), collector.actor.settings)
    assert action_contract(collector.actor.settings)['physics_reset'] == 'fresh_scene_on_episode_v1'


def test_success_reset_keeps_endpoints_callback_order_and_budget(monkeypatch, tmp_path):
    collector = fake_collector(monkeypatch, tmp_path, ['success', 'success', 'success'])
    callbacks = []
    ticks, skills = collector.collect(3, 'curiosity', seed=9, stage='test',
                                     on_skill_end=lambda skill: callbacks.append((collector.env.value, len(collector.env.seeds), skill['reset_after_skill'])))
    assert collector.env.steps == collector.total_steps == 3
    assert collector.env.seeds == [9, 10, 11]
    assert callbacks == [(-99., 1, 'grasp_success'), (-199., 2, 'grasp_success'), (-299., 3, 'grasp_success')]
    assert all(not row['terminated'] and not row['truncated'] for row in ticks + skills)
    data = skill_arrays(skills)
    np.testing.assert_array_equal(data['states'][:, 0], [-100., -200., -300.])
    np.testing.assert_array_equal(data['next_states'][:, 0], [-99., -199., -299.])
    np.testing.assert_array_equal(data['episode_ids'], [1, 2, 3])
    assert len(ticks) == len(skills) == 3


def test_failed_grasp_continues_same_scene(monkeypatch, tmp_path):
    collector = fake_collector(monkeypatch, tmp_path, ['unconfirmed_grasp', 'lost_grasp'])
    ticks, skills = collector.collect(2, 'random', seed=4, stage='test')
    assert collector.env.seeds == [4] and collector.env.steps == 2
    assert [row['episode_id'] for row in skills] == [1, 1]
    assert all('reset_after_skill' not in row for row in skills)
    np.testing.assert_array_equal(skills[0]['next_state'], skills[1]['state'])


def test_home_controls_count_but_only_grasp_enters_replay(monkeypatch, tmp_path):
    collector = fake_collector(monkeypatch, tmp_path, ['unconfirmed_grasp', 'lost_grasp'])
    def home_actions():
        if collector.total_steps:
            yield np.r_[np.zeros(7), 1.].astype(np.float32)
    collector._home_actions = home_actions
    proposals, callbacks = [], []
    def propose(policy):
        proposals.append(collector.state.copy())
        return np.array([1., 0., 0., 0.], np.float32)
    collector._propose_action = propose
    ticks, skills = collector.collect(3, 'random', seed=4, stage='test',
                                     on_skill_end=callbacks.append)
    assert collector.env.seeds == [4]  # Homing must not reset the cup.
    assert len(ticks) == collector.total_steps == 3
    assert [s['primitive_type'] for s in skills] == ['grasp', 'return_default', 'grasp']
    assert len(callbacks) == 2 and not skills[1]['trainable']
    assert proposals[1][0] == skills[1]['next_state'][0] == skills[2]['state'][0] == -98.
    assert skills[0]['next_state'][0] == -99.  # Preserve pre-home grasp endpoint.
    data = skill_arrays(skills)
    assert len(data['states']) == 2
    np.testing.assert_array_equal(data['duration_low_steps'], [1, 1])


def test_budget_ending_during_home_never_proposes_next_grasp(monkeypatch, tmp_path):
    collector = fake_collector(monkeypatch, tmp_path, ['unconfirmed_grasp'])
    def home_actions():
        if collector.total_steps:
            yield np.r_[np.zeros(7), 1.].astype(np.float32)
            raise AssertionError('Homing must not advance past budget')
    collector._home_actions = home_actions
    callbacks = []
    ticks, skills = collector.collect(2, 'random', seed=4, stage='test', on_skill_end=callbacks.append)
    assert len(ticks) == 2 and len(callbacks) == 1
    assert skills[-1]['primitive_type'] == 'return_default'
    assert not skills[-1]['completed'] and not skills[-1]['trainable']


def test_home_failure_prevents_grasp(monkeypatch, tmp_path):
    collector = fake_collector(monkeypatch, tmp_path, [])
    from domain.robocasa.motion_planning import MotionPlanningError
    def fail():
        raise MotionPlanningError('path_collision', 'injected home collision')
    collector._home_actions = fail
    collector._propose_action = lambda policy: pytest.fail('Grasp proposed before homing')
    with pytest.raises(MotionPlanningError, match='injected home collision'):
        collector.collect(2, 'random', seed=4, stage='test')
    assert collector.total_steps == 0


def test_preparation_contract_rejects_old_checkpoints():
    current = action_contract(settings())
    old = deepcopy(current)
    del old['grasp_only']['preparation']
    assert not action_contract_matches(old, settings())


def test_home_skips_submillimetre_exterior_retreat(monkeypatch):
    from domain.robocasa import motion_planning
    home = np.r_[np.zeros(7), .04, .04]
    q = home.copy(); q[:7] += .01
    velocity = np.ones(9)*.1
    calls = []
    robot = SimpleNamespace(get_qpos=lambda: q, get_qvel=lambda: velocity)
    class Root:
        def inv(self): return self
        def __mul__(self, other): return other
    robot.pose = Root()
    def plan(position, quaternion, **kwargs):
        calls.append(kwargs)
        return None
    raw = SimpleNamespace(agent=SimpleNamespace(robot=robot,
        tcp=SimpleNamespace(pose=SimpleNamespace(
            p=np.array([.1800005, 0., .7]), q=np.array([1.,0.,0.,0.])))),
        plan_to_pose=plan)
    class Cursor:
        def __init__(self, *args, **kwargs): self.status = 'running'
        def next_action(self): return np.r_[home[:7], 1.]
        def record_step(self):
            q[:] = home; velocity[:] = 0.; self.status = 'success'
    monkeypatch.setattr(motion_planning, 'MotionPlanExecution', Cursor)
    c = SkillCollector.__new__(SkillCollector)
    c.env = SimpleNamespace(unwrapped=raw)
    c.actor = SimpleNamespace(settings={})
    c.execution = SimpleNamespace(gripper_command=-1.)
    c._home_qpos = home
    c._home_position = np.array([.15, 0., .7])
    c._home_quaternion = np.array([1., 0., 0., 0.])
    c._context = lambda: SimpleNamespace(objects={})
    assert len(list(c._home_actions())) == 10
    assert len(calls) == 1 and 'target_qpos' in calls[0]


def test_macro_contact_summary_keeps_intermediate_contact(monkeypatch, tmp_path):
    collector = fake_collector(monkeypatch, tmp_path, ['unconfirmed_grasp', 'lost_grasp'])
    step = collector.env.step
    def contact_step(low):
        result = step(low)
        if collector.env.steps == 1:
            result[-1]['primitive_record'].contact_events = [dict(objects={
                'cup': dict(grasped=False, left_force=.01, right_force=0.)})]
        return result
    collector.env.step = contact_step
    _, skills = collector.collect(2, 'random', seed=4, stage='test')
    assert [row['object_contact'] for row in skills] == [True, False]
    assert all(row['before'] == row['after'] == {} for row in skills)


def test_stop_callback_prevents_success_reset(monkeypatch, tmp_path):
    collector = fake_collector(monkeypatch, tmp_path, ['success'])
    ticks, skills = collector.collect(3, 'curiosity', seed=5, stage='test', stop_when=lambda: True)
    assert len(ticks) == 1 and collector.env.seeds == [5]
    assert skills[0]['next_state'][0] == -99.


@pytest.mark.parametrize('status', ['path_collision', 'already_holding'])
def test_held_failure_resets_before_next_attempt(monkeypatch, tmp_path, status):
    collector = fake_collector(monkeypatch, tmp_path, [status, 'success'])
    collector._snapshot = lambda: {
        'contact_snapshot': {'objects': {'cup': {'grasped': collector.env.value == -99.}}}}
    callbacks = []
    ticks, skills = collector.collect(
        2, 'curiosity', seed=9, stage='test',
        on_skill_end=lambda row: callbacks.append((len(collector.env.seeds), row['status'])))
    assert collector.env.seeds == [9, 10]
    assert collector.total_steps == len(ticks) == 2
    assert callbacks == [(1, status), (2, 'success')]
    assert skills[0]['status'] == status and skills[0]['trainable']
    assert skills[0]['reset_after_skill'] == 'grasp_failure_still_holding'
    assert skills[0]['after']['contact_snapshot']['objects']['cup']['grasped']
    assert [r['episode_id'] for r in skills] == [1, 2]
    assert not any(r['terminated'] or r['truncated'] for r in skills)
    data = skill_arrays(skills)
    np.testing.assert_array_equal(data['states'][:, 0], [-100., -200.])
    np.testing.assert_array_equal(data['next_states'][:, 0], [-99., -199.])


def test_held_failure_at_budget_boundary_keeps_endpoint(monkeypatch, tmp_path):
    collector = fake_collector(monkeypatch, tmp_path, ['path_collision'])
    collector._snapshot = lambda: {'contact_snapshot': {'objects': {'cup': {'grasped': True}}}}
    ticks, skills = collector.collect(1, 'random', seed=3, stage='test')
    assert collector.env.seeds == [3] and len(ticks) == 1
    assert skills[0]['reset_after_skill'] == 'grasp_failure_still_holding'
    assert skills[0]['next_state'][0] == -99.


def test_old_grasp_reset_contract_is_rejected():
    contract = action_contract(settings())
    assert contract['grasp_only']['version'] == 2
    old = deepcopy(contract)
    old['grasp_only'].update(version=1, failure='continue_current_scene')
    assert not action_contract_matches(old, settings())


def test_single_step_training_has_no_future_bootstrap(monkeypatch):
    torch.set_num_threads(1)
    backend = CEEUSBackend(agent_dim=33, action_dim=4, hidden_dim=8, batch_size=4, object_heuristic={})
    actor = RAPSImaginationActor(backend, settings(imagination_horizon=1, discount=0.0))
    rng = np.random.default_rng(19)
    states = rng.normal(size=(8, 67)).astype(np.float32)
    actions = canonical_action(torch.tensor(rng.uniform(-1, 1, (8, 4)), dtype=torch.float32), actor.schema).numpy()
    data = dict(states=states, actions=actions, next_states=states.copy(), terminated=np.zeros(8, bool))
    backend.train(data, 1)
    original_predict, original_return = backend.model.predict, actor.lambda_return
    calls, targets = [], []
    def predict(**kwargs):
        calls.append(1)
        return original_predict(**kwargs)
    def checked_return(rewards, values, discount, bootstrap, **kwargs):
        assert rewards.shape[0] == 1 and torch.count_nonzero(discount) == 0
        result = original_return(rewards, values, discount, bootstrap, **kwargs)
        torch.testing.assert_close(result, rewards, rtol=0, atol=0)
        changed = original_return(rewards, values + 1000, discount, bootstrap - 1000, **kwargs)
        torch.testing.assert_close(changed, result, rtol=0, atol=0)
        targets.append(result)
        return result
    monkeypatch.setattr(backend.model, 'predict', predict)
    monkeypatch.setattr(actor, 'lambda_return', checked_return)
    metrics = actor.train(data)
    assert len(calls) == len(targets) == metrics['updates'] == 1
    assert np.isfinite(list(metrics.values())).all()
