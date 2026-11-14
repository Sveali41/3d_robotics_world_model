from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from exploration.jepa import cabinet


def saved_config():
    return dict(
        bootstrap_steps=4000,
        rounds=10,
        steps_per_round=1000,
        total_controls=14000,
        seed=0,
        device="cpu",
        collection_policy="curiosity",
        cpu_threads=1,
        validation_every=5,
        model=dict(history_size=3, horizon=3),
        planner={},
    )


def test_resume_config_extends_aligned_training_budget_without_mutation():
    original = saved_config()
    before = deepcopy(original)

    resolved = cabinet.resume_config(original, stage="train", total_controls=400000)

    assert resolved["total_controls"] == 400000
    assert resolved["rounds"] == 396
    assert {key: value for key, value in resolved.items() if key != "rounds" and key != "total_controls"} == {
        key: value for key, value in before.items() if key != "rounds" and key != "total_controls"
    }
    assert original == before
    assert resolved is not original


def test_resume_config_omitted_or_unchanged_budget_preserves_saved_config():
    original = saved_config()
    assert cabinet.resume_config(original, stage="train", total_controls=None) == original
    assert cabinet.resume_config(original, stage="train", total_controls=14000) == original


@pytest.mark.parametrize("target", [True, False, 14000.0, 14001.5, 0, -1])
def test_resume_config_rejects_invalid_budget(target):
    with pytest.raises(ValueError, match="positive integer"):
        cabinet.resume_config(saved_config(), stage="train", total_controls=target)


def test_resume_config_rejects_decreases_misalignment_and_nontraining_extensions():
    original = saved_config()
    with pytest.raises(ValueError, match="cannot decrease"):
        cabinet.resume_config(original, stage="train", total_controls=13000)
    with pytest.raises(ValueError, match="whole steps_per_round"):
        cabinet.resume_config(original, stage="train", total_controls=400500)
    for stage in ("evaluate", "smoke"):
        with pytest.raises(ValueError, match="Only --stage train"):
            cabinet.resume_config(original, stage=stage, total_controls=15000)


def test_run_validates_saved_config_then_uses_extended_config(monkeypatch, tmp_path):
    original = saved_config()
    payload = {"config": original, "replay": {"rows": []}}
    observed = {}

    monkeypatch.setattr(cabinet.torch, "load", lambda *args, **kwargs: payload)

    def read_checkpoint(path, config):
        observed["checkpoint_config"] = deepcopy(config)
        assert config == original
        return payload

    monkeypatch.setattr(cabinet, "read_checkpoint", read_checkpoint)

    class Model:
        def __init__(self, config, device):
            self.config = config

    class Planner:
        def __init__(self, model, config, seed):
            self.config = config

    class Replay:
        def __init__(self, *args, **kwargs):
            pass

    class Collector:
        class Env:
            @staticmethod
            def close():
                pass

        env = Env()

    monkeypatch.setattr("exploration.jepa.model.StochasticJEPA", Model)
    monkeypatch.setattr("exploration.jepa.planner.MixedCEM", Planner)
    monkeypatch.setattr("exploration.jepa.collector.make_collector", lambda *args: Collector())
    monkeypatch.setattr(cabinet, "VisualReplay", Replay)
    monkeypatch.setattr(cabinet, "Learner", lambda model, config: object())
    monkeypatch.setattr(cabinet, "run_loop", lambda config, *args, **kwargs: observed.update(run_config=config) or {
        "status": "complete", "real_controls": 14000, "updates": 0, "replay_transitions": 0
    })
    written = {}
    monkeypatch.setattr(cabinet, "write_json", lambda path, value: written.update({Path(path).name: value}))
    monkeypatch.setattr(cabinet.importlib.metadata, "version", lambda name: "test")

    args = SimpleNamespace(
        stage="train", checkpoint=tmp_path / "round_009.pt", device=None,
        policy="curiosity", total_controls=400000, output=tmp_path / "resumed",
        execution=None, runtime_optimizations=False, evaluation_steps=None, object_shift_xy=[],
    )
    assert cabinet.run(args, {}) == 0
    assert observed["checkpoint_config"] == original
    assert observed["run_config"]["total_controls"] == 400000
    assert observed["run_config"]["rounds"] == 396
    assert original["total_controls"] == 14000 and original["rounds"] == 10
    assert written["manifest.json"]["budget_extension"] == dict(
        original_total_controls=14000, new_total_controls=400000, rounds=396
    )
