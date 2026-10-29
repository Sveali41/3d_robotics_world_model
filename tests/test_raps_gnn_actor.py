from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exploration.cee_us_backend import CEEUSBackend
from exploration.raps_gnn_actor import (
    RAPSImaginationActor, SkillCollector, canonical_action, load, save, skill_arrays,
)


def backend():
    return CEEUSBackend(action_dim=20, hidden_dim=8, horizon=3, candidates=8,
                        iterations=2, elites=4, batch_size=4)


def replay():
    rng = np.random.default_rng(3)
    states = rng.normal(size=(8, 66)).astype(np.float32)
    next_states = states.copy()
    next_states[:, :56] += rng.normal(size=(8, 56)).astype(np.float32) * .01
    actions = torch.tensor(rng.uniform(-1, 1, (8, 20)), dtype=torch.float32)
    actions = canonical_action(actions).numpy()
    return dict(states=states, actions=actions, next_states=next_states,
                terminated=np.zeros(8, bool))


def actor(model):
    return RAPSImaginationActor(model, dict(hidden_size=8, actor_layers=1, critic_layers=1,
                                          imagination_horizon=3, imagination_batch_size=4,
                                          updates_per_round=2, target_update_period=1))


def test_canonical_actions_only_consume_selected_arguments():
    raw = torch.ones(3, 20)
    raw[:, :10] = 0
    raw[0, 1] = raw[1, 2] = raw[2, 9] = 1
    result = canonical_action(raw)
    assert torch.equal(result[:, :10], raw[:, :10])
    assert torch.nonzero(result[0, 10:]).flatten().tolist() == [3, 4, 5]
    assert torch.nonzero(result[1, 10:]).flatten().tolist() == [4]
    assert torch.count_nonzero(result[2, 10:]) == 0


def test_imagination_updates_actor_and_critic_without_updating_gnn():
    torch.set_num_threads(1)
    model, data = backend(), replay()
    model.train(data, epochs=1)
    policy = actor(model)
    policy.settings["entropy_scale"] = 0.0
    old_model = deepcopy(model.model.model.state_dict())
    old_actor = deepcopy(policy.actor.state_dict())
    old_critic = deepcopy(policy.critic.state_dict())
    trajectory = policy.imagine(torch.tensor(data["states"][:4]))
    assert trajectory["states"].shape == (4, 4, 66)
    assert trajectory["actions"].shape == (3, 4, 20)
    assert not trajectory["rewards"].requires_grad
    torch.testing.assert_close(trajectory["states"][1:, :, 56:], trajectory["states"][:1, :, 56:].expand(3, -1, -1))
    result = policy.train(data)
    assert result["total_updates"] == 2
    assert all(np.isfinite(value) for value in result.values())
    for head in ("discrete_actor.", "continuous_actor."):
        assert any(not torch.equal(old_actor[k], v) for k, v in policy.actor.state_dict().items() if k.startswith(head))
    assert any(not torch.equal(old_critic[k], v) for k, v in policy.critic.state_dict().items())
    for key, value in model.model.model.state_dict().items():
        torch.testing.assert_close(value, old_model[key], rtol=0, atol=0)


def test_disagreement_uses_shared_state_sample_std_and_no_env():
    model, data = backend(), replay()
    policy = actor(model)
    def predict(*, observations, actions, states):
        assert torch.equal(observations[0], observations[1])
        out = observations.clone()
        out[..., 0] += torch.arange(5)[:, None]
        return out, None, None
    model.model.predict = predict
    trajectory = policy.imagine(torch.tensor(data["states"][:2]))
    expected = torch.arange(5, dtype=torch.float32).std(unbiased=True)
    torch.testing.assert_close(trajectory["rewards"], expected.expand(3, 2, 1))


@pytest.mark.parametrize('weight', [0., 1.])
def test_disagreement_flag_preserves_heuristic_and_default_reward(weight):
    model = backend()
    model.object_heuristic = model._validate_object_heuristic(
        {'object_weights': {'cabinet_item_16': 1.}})
    policy = actor(model)
    policy.settings['disagreement_weight'] = weight
    def predict(*, observations, actions, states):
        out = observations.clone()
        out[..., 0] += torch.arange(5)[:, None]
        out[..., 18] += .01
        return out, None, None
    model.model.predict = predict
    starts = torch.zeros(2, 66)
    starts[:, 32] = 1.  # Cup one meter ahead of TCP.
    trajectory = policy.imagine(starts)
    intrinsic = torch.arange(5, dtype=torch.float32).std(unbiased=True)
    torch.testing.assert_close(trajectory['heuristics'], torch.full((3, 2), 5.), atol=1e-4, rtol=1e-5)
    expected = trajectory['heuristics'] + weight * intrinsic
    torch.testing.assert_close(trajectory['rewards'].squeeze(-1), expected)


def test_resume_history_reuses_saved_prefix_and_rejects_protocol_change(tmp_path):
    import json
    from omegaconf import OmegaConf
    from exploration.raps_gnn_actor import resume_history, SCHEMA
    config = OmegaConf.create(dict(seed=0, bootstrap_steps=1, steps_per_round=1,
                                    rounds=1, evaluation_steps=0, raps_actor={}))
    rows = []
    for step, stage in [(1, 'bootstrap'), (2, 'round_1')]:
        row = dict(step=step, skill_id=step, end_step=step, stage=stage,
            state=[0.] * 66, action=[0.] * 20, next_state=[.01] * 66,
            episode_id=step, terminated=False, truncated=False,
            trainable=True, duration_low_steps=1)
        rows.append(row)
        directory = tmp_path / ('bootstrap' if step == 1 else 'curiosity')
        directory.mkdir()
        for name in ['transitions.jsonl', 'skills.jsonl']:
            (directory / name).write_text(json.dumps(row) + '\n')
    path = tmp_path / 'curiosity/round_001.pt'
    torch.save(dict(schema=SCHEMA, config=OmegaConf.to_container(config), real_steps=2), path)
    np.savez_compressed(path.with_suffix('.replay.npz'), **skill_arrays(rows))
    config.rounds = 3
    config.evaluation_steps = 1
    ticks, skills, next_round = resume_history(path, config)
    assert ticks == skills == rows and next_round == 2
    config.seed = 1
    with pytest.raises(ValueError, match='only rounds and evaluation_steps'):
        resume_history(path, config)


def test_nonfinite_prediction_stops_before_actor_update():
    model, data = backend(), replay()
    policy = actor(model)
    old = deepcopy(policy.actor.state_dict())
    model.model.predict = lambda **kw: (torch.full_like(kw["observations"], float("nan")), None, None)
    with pytest.raises(FloatingPointError, match="imagination"):
        policy.train(data)
    assert policy.updates == 0
    for key, value in policy.actor.state_dict().items():
        assert torch.equal(value, old[key])


def test_latched_state_training_and_command_uncertainty_not_rewarded():
    model = CEEUSBackend(agent_dim=33, action_dim=20, hidden_dim=8, horizon=3,
                        candidates=8, iterations=2, elites=4, batch_size=4,
                        object_heuristic={"object_weights": {"cabinet_item_16": 1.}})
    data = replay()
    data["states"] = np.insert(data["states"], 32, 1.0, axis=1)
    data["next_states"] = np.insert(data["next_states"], 32, -1.0, axis=1)
    model.train(data, 1)
    policy = RAPSImaginationActor(model, dict(latched_gripper=True, hidden_size=8,
        actor_layers=1, critic_layers=1, imagination_horizon=2,
        imagination_batch_size=4, updates_per_round=1))
    assert policy.features(torch.tensor(data["states"])).shape == (8, 67)
    result = policy.train(data)
    assert np.isfinite(result["actor_loss"])
    def predict(*, observations, actions, states):
        out = observations.clone()
        out[..., 32] = torch.arange(5)[:, None]
        return out, None, None
    model.model.predict = predict
    imagined = policy.imagine(torch.tensor(data["states"][:2]))
    torch.testing.assert_close(imagined["rewards"], torch.zeros(2, 2, 1))
    # Object offsets moved by one; the physical approach signal is unchanged.
    before = np.zeros(67, np.float32); before[33] = 1
    after = before.copy(); after[18] = .1
    assert model.observed_object_heuristic(before, after)["approach_progress_score"] == pytest.approx(50., abs=1e-4)


def test_prediction_metrics_use_shifted_object_offsets_with_latch():
    from exploration.cee_us_metrics import prediction_metrics
    states = np.zeros((1, 67), np.float32)
    next_states = states.copy()
    next_states[0, 45] = .02  # Bowl x, not its orientation or command memory.
    def rollout(states, actions):
        result = np.broadcast_to(states[None, :, None], (5, 1, 1, 67)).copy()
        result[..., 32] = np.arange(5)[:, None, None]
        return result
    model = SimpleNamespace(env=SimpleNamespace(agent_dim=33), rollout=rollout)
    data = dict(states=states, next_states=next_states, actions=np.zeros((1, 20)), episode_ids=np.zeros(1))
    result = prediction_metrics(model, data, horizons=(1,))
    assert result["1"]["object_position_mae_m"] == pytest.approx([0., .02])
    assert result["1"]["moved_object_windows"] == [0, 1]


class Primitive:
    def __init__(self, action):
        self.action = action
        self.status = "running"
    @property
    def done(self):
        return self.status != "running"
    def reset(self, context):
        self.steps = 0
    def next_action(self, context):
        if self.steps == 3:
            self.status = "success"
            return None
        self.steps += 1
        return np.array([0., 0., .1, -1.], dtype=np.float32)
    def observe(self, context):
        pass


class Env:
    def __init__(self):
        self.env = self
    def _context(self):
        return None
    def reset(self, seed):
        self.tick = 0
    def step(self, action):
        self.tick += 1
        record = SimpleNamespace(duration_low_steps=1, low_actions=[action.copy()],
                                 full_controller_actions=[action.copy()], status="success",
                                 contact_events=[], outcome_metrics={})
        return None, 0, False, self.tick == 5, {
            "primitive_record": record, "exploration_reward": {"total": 0}}


@pytest.mark.parametrize("skill_set", ["legacy10", "compact6"])
def test_collector_exact_budget_episode_boundaries_and_partial_exclusion(monkeypatch, tmp_path, skill_set):
    import exploration.raps_gnn_actor as module
    env = Env()
    monkeypatch.setattr(module, "snapshot", lambda env: {})
    monkeypatch.setattr("domain.robocasa.skill_execution.PRIMITIVE_CLASSES", dict.fromkeys(module.RAPS_ACTION_SCHEMA.primitive_names, Primitive))
    collector = SkillCollector(env, SimpleNamespace(encode=lambda: np.full(66, env.tick, np.float32)),
                               SimpleNamespace(settings={"hold_closed_during_motion": False, "skill_set": skill_set}), tmp_path)
    ticks, skills = collector.collect(13, "random", seed=0, stage="test")
    assert len(ticks) == collector.total_steps == 13
    assert [s["duration_low_steps"] for s in skills] == [3, 2, 3, 2, 3]
    # Last skill reaches its endpoint at the budget boundary, but has not
    # reported completion: conservatively exclude it rather than invent a label.
    assert [s["trainable"] for s in skills] == [True, False, True, False, False]
    assert all(len(s["action"]) == module.skill_schema({"skill_set": skill_set}).action_dim for s in skills)
    arrays = skill_arrays(skills)
    assert arrays["episode_ids"].tolist() == [1, 2]
    assert arrays["duration_low_steps"].tolist() == [3, 3]
    assert len((tmp_path / "transitions.jsonl").read_text().splitlines()) == 13
    assert ticks[-1]["run_budget_exhausted"]
    np.testing.assert_array_equal(ticks[5]["state"], np.zeros(66))


def test_checkpoint_restores_model_actor_optimizers_and_rng(tmp_path):
    from omegaconf import OmegaConf
    model, data = backend(), replay()
    model.train(data, 1)
    policy = actor(model)
    policy.train(data)
    rows = [dict(state=data["states"][i], next_state=data["next_states"][i], action=data["actions"][i],
                 episode_id=1, terminated=False, truncated=False, trainable=True, duration_low_steps=3) for i in range(8)]
    low_rows = [{**r, "action": np.zeros(4)} for r in rows]
    encoder = SimpleNamespace(contract=lambda: {"test": True})
    path = tmp_path / "actor.pt"
    save(path, model, policy, encoder, OmegaConf.create({}), low_rows, rows)
    expected = policy.action(data["states"][0])
    policy.train(data)
    load(path, model, policy, encoder)
    assert policy.updates == 2
    np.testing.assert_array_equal(policy.action(data["states"][0]), expected)
    with pytest.raises(ValueError, match="incompatible"):
        torch.save({"schema": "cabinet-cee-us-state-v1"}, tmp_path / "old.pt")
        load(tmp_path / "old.pt", model, policy, encoder)


def test_compact_skill_space_sampling_imagination_and_updates():
    from exploration.raps_gnn_actor import skill_schema, action_contract, summarize
    from domain.robocasa.primitives import PrimitiveAction
    schema = skill_schema(dict(skill_set="compact6"))
    assert schema.primitive_names == ("move_delta_ee_pose", "top_grasp", "lift", "drop",
                                      "open_gripper", "close_gripper")
    assert schema.action_dim == 12
    for spec in schema.specs:
        params = {p.name: p.scale * .25 for p in spec.parameters}
        decoded = schema.decode(schema.encode(PrimitiveAction(spec.name, params)))
        assert decoded.primitive_type == spec.name
        for name, value in params.items():
            assert decoded.parameters[name] == pytest.approx(value)
    torch.set_num_threads(1)
    model = CEEUSBackend(action_dim=12, hidden_dim=8, horizon=3, candidates=8,
                         iterations=2, elites=4, batch_size=4)
    data = replay()
    raw = torch.ones(8, 12)
    raw[:, :6] = 0
    raw[torch.arange(8), torch.arange(8) % 6] = 1
    data["actions"] = canonical_action(raw, schema).numpy()
    assert not data["actions"][4:, 6:][0:2].any()  # Open/close consume no arguments.
    model.train(data, epochs=1)
    policy = RAPSImaginationActor(model, dict(skill_set="compact6", hidden_size=8,
        actor_layers=1, critic_layers=1, imagination_horizon=3,
        imagination_batch_size=4, updates_per_round=2))
    assert policy.action(data["states"][0]).shape == (12,)
    assert policy.imagine(torch.tensor(data["states"][:4]))["actions"].shape == (3, 4, 12)
    assert policy.train(data)["total_updates"] == 2
    assert set(summarize([], [], schema)["per_skill"]) == set(schema.primitive_names)
    assert action_contract(policy.settings)["action_dim"] == 12
    from exploration.raps_gnn_actor import DEFAULTS
    assert action_contract(DEFAULTS)["action_dim"] == 20
    with pytest.raises(ValueError, match="fresh 20D"):
        RAPSImaginationActor(model)
    with pytest.raises(ValueError, match="Unknown raps_actor.skill_set"):
        skill_schema(dict(skill_set="invalid"))
