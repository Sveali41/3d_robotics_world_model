from types import SimpleNamespace

import numpy as np
import pytest

from exploration.cee_us_cabinet import Collector, compare, replay_arrays, rng_state
from exploration.cee_us_metrics import prediction_metrics, screen


class Model:
    def __init__(self):
        self.resets = 0

    def reset_planner(self, state):
        self.resets += 1

    def random_action(self, state):
        return np.array([2., 0., 0., -1.], dtype=np.float32)

    def action(self, state):
        return self.random_action(state), {"curiosity_score": 2.}

    def rollout(self, states, actions):
        return np.broadcast_to(states[None, :, None], (5, len(states), actions.shape[1], 66)).copy()


class Env:
    def reset(self, seed):
        self.tick = 0
        return np.zeros(1), {}

    def step(self, action):
        self.tick += 1
        record = SimpleNamespace(duration_low_steps=1, low_actions=[np.clip(action, -1, 1)],
                                 full_controller_actions=[np.r_[np.clip(action[:3], -1, 1),
                                                               [.2, .3, .4], action[3]]],
                                 status="success", contact_events=[], outcome_metrics={})
        return None, 0., False, self.tick == 2, {
            "primitive_record": record, "exploration_reward": {"total": 0}}

    @property
    def env(self):
        return self

    def _controller_action(self, action):
        raise AssertionError("Collector must use the recorded command, not recompute it")


def test_collector_actual_actions_exact_budget_and_episode_boundaries(monkeypatch, tmp_path):
    env, backend = Env(), Model()
    monkeypatch.setattr("exploration.cee_us_cabinet.snapshot", lambda env: {})
    collector = Collector(env, backend, SimpleNamespace(encode=lambda: np.full(66, env.tick)), tmp_path)
    rows = collector.collect(5, "random", seed=0, stage="test")
    assert collector.total_steps == 5
    assert backend.resets == 3
    assert [r["episode_id"] for r in rows] == [1, 1, 2, 2, 3]
    assert rows[0]["requested_action"][0] == 2
    assert rows[0]["action"][0] == 1
    np.testing.assert_allclose(rows[0]["full_controller_action"], [1, 0, 0, .2, .3, .4, -1])
    assert rows[2]["state"][0] == 0
    assert rows[1]["next_state"][0] == 2
    assert rows[-1]["run_budget_exhausted"]
    replay = replay_arrays(rows)
    np.testing.assert_array_equal(replay["actions"][:, 0], np.ones(5))
    assert len((tmp_path / "transitions.jsonl").read_text().splitlines()) == 5


def test_prediction_windows_do_not_cross_resets_and_identify_missing_motion():
    replay = dict(states=np.zeros((12, 66)), next_states=np.zeros((12, 66)),
                  actions=np.zeros((12, 4)), episode_ids=np.repeat([1, 2], 6))
    result = prediction_metrics(Model(), replay)
    assert result["1"]["windows"] == 12
    assert result["10"]["windows"] == 0
    assert "without_moving_samples" in result["interaction_dynamics_status"]
    assert result["1"]["object_position_mae_m"] == [0., 0.]


def test_physical_evaluation_rejects_legacy_controller_checkpoint(monkeypatch, tmp_path):
    import exploration.cee_us_cabinet as runner
    monkeypatch.setattr(runner, 'read_checkpoint', lambda *args: {})
    with pytest.raises(ValueError, match='legacy wrist control'):
        runner.evaluate(None, None, None, None, tmp_path, 'legacy.pt', 'curiosity')


def test_motion_prediction_errors_and_screen_require_both_objects():
    replay = dict(states=np.zeros((1, 66)), next_states=np.zeros((1, 66)),
                  actions=np.zeros((1, 4)), episode_ids=np.zeros(1))
    replay["next_states"][0, 32] = .02
    result = prediction_metrics(Model(), replay, horizons=(1,))
    assert result["1"]["moving_position_mae_m"][0] == pytest.approx(.02)
    assert result["1"]["moving_position_mae_m"][1] is None
    baseline = {"low_level_steps": 1000, "per_object": {},
                "recoverable_motion_primitives_per_1000_steps": 1., "recoverable_coverage_auc": 1.}
    candidate = dict(baseline, recoverable_motion_primitives_per_1000_steps=2., recoverable_coverage_auc=2.)
    assert screen(baseline, candidate)["status"] == "not_passed"
    candidate["per_object"] = {name: {"recoverable_motion_primitives": 1}
                               for name in ("cabinet_item_16", "cabinet_item_17")}
    assert screen(baseline, candidate)["status"] == "screening_pass"


@pytest.mark.parametrize('optimized', [False, True])
def test_compare_budget_independent_replay_and_shared_holdout(monkeypatch, tmp_path, optimized):
    from omegaconf import OmegaConf
    import exploration.cee_us_cabinet as runner

    class Trainable(Model):
        weight = 0

        def state_dict(self):
            return {"weight": self.weight}

        def load_state_dict(self, state):
            self.weight = state["weight"]

    env, model = Env(), Trainable()
    encoder = SimpleNamespace(encode=lambda: np.full(66, env.tick), contract=lambda: {})
    monkeypatch.setattr(runner, "snapshot", lambda env: {})
    training, saved, holdouts = [], {}, []

    def train(backend, rows, epochs, directory, label, *, arrays=None):
        if optimized:
            for key, value in replay_arrays(rows).items():
                np.testing.assert_array_equal(arrays[key], value)
        training.append((directory.name, label, backend.weight, [r["policy"] for r in rows]))
        backend.weight += 1

    def save(path, backend, encoder, config, rows, state, stage, *, arrays=None):
        if optimized:
            for key, value in replay_arrays(rows).items():
                np.testing.assert_array_equal(arrays[key], value)
        payload = {"backend": backend.state_dict(), "rng": rng_state()}
        saved[path] = payload
        return payload

    def load(path, backend):
        backend.load_state_dict(saved[path]["backend"])

    def predictions(backend, replay):
        holdouts.append(replay["states"].copy())
        return {"windows": len(replay["states"])}

    monkeypatch.setattr(runner, "train_round", train)
    monkeypatch.setattr(runner, "save_checkpoint", save)
    monkeypatch.setattr(runner, "read_checkpoint", load)
    monkeypatch.setattr(runner, "interaction_summary", lambda rows: {"count": len(rows)})
    monkeypatch.setattr(runner, "prediction_metrics", predictions)
    monkeypatch.setattr(runner, "report", lambda output: None)
    cfg = OmegaConf.create(dict(seed=0, bootstrap_steps=4, rounds=2,
                                steps_per_round=3, evaluation_steps=4, epochs=1,
                                runtime_optimizations=optimized))
    compare(env, model, encoder, cfg, tmp_path)
    import json
    assert json.loads((tmp_path / "budget.json").read_text())["real_steps"] == 24
    assert training[1][2] == training[3][2] == 1  # Identical learned initialization.
    assert all(policy == "random" for policy in training[2][3])
    assert training[4][3].count("random") == 4  # Curiosity replay only shares bootstrap.
    assert training[4][3].count("curiosity") == 6
    np.testing.assert_array_equal(holdouts[0], holdouts[1])


@pytest.mark.parametrize('optimized', [False, True])
def test_single_arm_training_budget_and_no_evaluation_leakage(monkeypatch, tmp_path, optimized):
    import json
    from omegaconf import OmegaConf
    import exploration.cee_us_cabinet as runner
    env, model = Env(), Model()
    encoder = SimpleNamespace(encode=lambda: np.full(66, env.tick))
    training, saved, holdouts = [], [], []
    monkeypatch.setattr(runner, 'snapshot', lambda env: {})
    def fit(backend, rows, epochs, directory, label, *, arrays=None):
        if optimized:
            for key, value in replay_arrays(rows).items():
                np.testing.assert_array_equal(arrays[key], value)
        training.append((label, len(rows), [r['stage'] for r in rows]))
        return {'seconds': 0.}
    def save(path, backend, encoder, config, rows, state, stage, *, arrays=None):
        if optimized:
            for key, value in replay_arrays(rows).items():
                np.testing.assert_array_equal(arrays[key], value)
        saved.append((stage, len(rows)))
    def predictions(backend, replay):
        holdouts.append(len(replay['states']))
        return {}
    monkeypatch.setattr(runner, 'train_round', fit)
    monkeypatch.setattr(runner, 'save_checkpoint', save)
    monkeypatch.setattr(runner, 'prediction_metrics', predictions)
    monkeypatch.setattr(runner, 'interaction_summary', lambda rows: {
        'contact_primitives_per_1000_steps': 0.,
        'recoverable_motion_primitives_per_1000_steps': 0., 'grasp_primitive_rate': 0.})
    cfg = OmegaConf.create(dict(seed=0, bootstrap_steps=4, rounds=2,
                               steps_per_round=3, evaluation_steps=4, epochs=1,
                               runtime_optimizations=optimized))
    runner.train(env, model, encoder, cfg, tmp_path)
    assert [(label, n) for label, n, _ in training] == [('bootstrap', 4), ('round_1', 7), ('round_2', 10)]
    assert all('evaluation' not in stages for _, _, stages in training)
    assert saved == [('bootstrap', 4), ('round_1', 7), ('round_2', 10)]
    assert holdouts == [4]
    assert json.loads((tmp_path/'budget.json').read_text()) == {
        'training_transitions': 10, 'evaluation_steps': 4, 'real_steps': 14}


def test_collector_logs_heuristic_from_executed_transition(monkeypatch, tmp_path):
    class Guided(Model):
        object_heuristic = {'approach_weight': 50.}
        def observed_object_heuristic(self, state, next_state):
            return {'approach_progress_score': float(next_state[0] - state[0])}
    env = Env()
    monkeypatch.setattr('exploration.cee_us_cabinet.snapshot', lambda env: {})
    collector = Collector(env, Guided(), SimpleNamespace(encode=lambda: np.full(66, env.tick)), tmp_path)
    rows = collector.collect(3, 'curiosity', seed=0, stage='test')
    assert [r['object_heuristic_observed']['approach_progress_score'] for r in rows] == [1., 1., 1.]
    assert all(r['reward'] == {'total': 0} for r in rows)


def test_heuristic_configuration_matches_completed_baseline():
    from omegaconf import OmegaConf
    import exploration.cee_us_cabinet as runner
    baseline = OmegaConf.to_container(OmegaConf.load(runner.ROOT/'configs/cee_us_cabinet_60k.yaml'))
    guided = OmegaConf.to_container(OmegaConf.load(runner.ROOT/'configs/cee_us_cabinet_60k_heuristic.yaml'))
    heuristic = guided.pop('object_heuristic')
    assert guided == baseline
    assert heuristic['approach_weight'] == 50.
    assert heuristic['object_weights'] == {'cabinet_item_16': 1., 'cabinet_item_17': 1.}
