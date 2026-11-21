"""Collection caching preserves recorded data and live safety decisions."""
import json
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from domain.robocasa.primitives import PrimitiveContext
from exploration.cee_us.cee_us_cabinet import rng_state, restore_rng
from exploration.raps import raps_gnn_actor as raps
from tests.raps.test_grasp_only import settings
from tests.raps.test_grasp_runtime_equivalence import assert_exact


def collector_fixture(monkeypatch, directory, optimized, fail_step=False):
    monkeypatch.setattr(raps.SkillCollector, '_capture_home', lambda self: None)
    monkeypatch.setattr(raps.SkillCollector, '_home_actions', lambda self: iter(()))
    events = []

    class Execution:
        def __init__(self, **kwargs):
            self.outcomes = iter(("failed_motion", "success", "lost_grasp"))
            self.reset()

        def reset(self):
            self.gripper_command = -1.
            self.support_stopped_object_ids = []
            self.motion_attempts = []
            self.last_motion_tick = {}

        def start(self, action, context):
            events.append(("start", float(context.ee_position[0])))
            self.age = 0
            self.primitive = SimpleNamespace(status="running", phase="intermediate", done=False)

        def next_action(self, context):
            events.append(("next", float(context.ee_position[0])))
            return np.r_[np.zeros(7), -1.].astype(np.float32)

        def record_action(self, *args, **kwargs):
            events.append(("record", env.value))
            self.age += 1

        def check_retention(self, context):
            events.append(("retention", float(context.ee_position[0])))
            context.is_grasping("item")
            if self.age == 2:
                self.primitive.status = next(self.outcomes)
                self.primitive.done = True

        def observe(self, context):
            events.append(("observe", float(context.ee_position[0])))
            context.contact_snapshot()

    class Env:
        action_mode = "joint_position"

        def __init__(self):
            self.env = self.unwrapped = self
            self.value = 0.
            self.seeds = []
            self.contexts = []
            self.force_calls = 0
            self.agent = SimpleNamespace(finger1pad_link=None, finger2pad_link=None,
                finger1_link="left", finger2_link="right", is_grasping=self.grasping)
            self.scene = SimpleNamespace(get_pairwise_contact_forces=self.forces)

        def grasping(self, actor):
            events.append(("grasping", self.value))
            return torch.tensor([False])

        def forces(self, link, actor):
            self.force_calls += 1
            return torch.tensor([[.01, 0., 0.]])

        def _context(self):
            context = PrimitiveContext.__new__(PrimitiveContext)
            context.env, context.agent = self, self.agent
            context.ee_position = np.full(3, self.value, np.float32)
            context.objects = {"item": SimpleNamespace(actor="item",
                position_root=np.full(3, self.value + 1, np.float32),
                position_world=np.full(3, self.value + 2, np.float32))}
            self.contexts.append(context)
            return context

        def is_ee_inside_exploration_region(self):
            events.append(("inside", self.value))
            return True

        def failed_object_ids(self):
            events.append(("failed", self.value))
            return []

        def reset(self, seed):
            self.seeds.append(seed)
            self.value = -100. * len(self.seeds)

        def step(self, low):
            self.value += 1.
            events.append(("step", self.value))
            if fail_step:
                raise RuntimeError("injected motion failure")
            record = SimpleNamespace(duration_low_steps=1, low_actions=[low.copy()],
                full_controller_actions=[low.copy()], status="success", contact_events=[], outcome_metrics={})
            return None, 0., False, False, {"primitive_record": record, "exploration_reward": {}}

    monkeypatch.setattr(raps, "PlannedPickPlaceSkillExecution", Execution)
    env = Env()
    encoder = SimpleNamespace(encode=lambda: np.full(67, env.value, np.float32))
    actor = SimpleNamespace(settings=settings(grasp_preflight=False))
    return raps.SkillCollector(env, encoder, actor, directory,
                              collection_optimizations=optimized), events


def test_collection_records_rng_and_safety_order_match(monkeypatch, tmp_path):
    initial_rng = rng_state()
    runs = []
    for optimized in (False, True):
        directory = tmp_path / str(optimized)
        directory.mkdir()
        collector, events = collector_fixture(monkeypatch, directory, optimized)
        restore_rng(initial_rng)
        ticks, skills = collector.collect(6, "random", seed=9, stage="test")
        logs = {name: (directory / name).read_text() for name in ("transitions.jsonl", "skills.jsonl")}
        assert [json.loads(line) for line in logs["transitions.jsonl"].splitlines()] == ticks
        assert [json.loads(line) for line in logs["skills.jsonl"].splitlines()] == skills
        assert collector.env.seeds == [9, 10]
        assert [row["status"] for row in skills] == ["failed_motion", "success", "lost_grasp"]
        assert skills[1]["reset_after_skill"] == "grasp_success"
        assert [row["next_state"][0] for row in skills] == [-98., -96., -198.]
        assert skills[2]["before"]["ee_position"] == [-200.] * 3
        control_events = [event for event in events if event[0] in
                          {"start", "next", "step", "record", "retention", "observe"}]
        expected = []
        for start in (-100., -98., -200.):
            expected.append(("start", start))
            for age in (1, 2):
                expected.extend((("next", start + age - 1), ("step", start + age),
                                 ("record", start + age), ("retention", start + age)))
                if age == 2:
                    expected.append(("observe", start + age))
        assert control_events == expected
        assert len({id(context.ee_position) for context in collector.env.contexts}) == len(collector.env.contexts)
        runs.append((ticks, skills, logs, events, rng_state(), len(collector.env.contexts), collector.env.force_calls))
    assert_exact(runs[1][:5], runs[0][:5])
    assert runs[1][5] < runs[0][5]
    assert runs[1][6] < runs[0][6]


@pytest.mark.parametrize("optimized", [False, True])
def test_failed_step_invalidates_context(monkeypatch, tmp_path, optimized):
    collector, _ = collector_fixture(monkeypatch, tmp_path, optimized, fail_step=True)
    with pytest.raises(RuntimeError, match="injected motion failure"):
        collector.collect(1, "random", seed=9, stage="test")
    assert collector._context_cache is None
    assert collector._context().ee_position[0] == -99.
    assert collector.env.contexts[0].ee_position[0] == -100.


def test_force_cache_keeps_binding_and_agent_checks_live():
    calls = []
    grasped = [False]
    binding = SimpleNamespace(bound_object_id=None)
    def force(link, actor):
        calls.append(link)
        return torch.tensor([[.01, 0., 0.]])
    def is_grasping(actor):
        calls.append("agent")
        return torch.tensor(grasped)
    context = PrimitiveContext.__new__(PrimitiveContext)
    context.env = SimpleNamespace(_grasp_binding=binding,
        scene=SimpleNamespace(get_pairwise_contact_forces=force))
    context.agent = SimpleNamespace(finger1pad_link=None, finger2pad_link=None,
        finger1_link="left", finger2_link="right", is_grasping=is_grasping)
    context.objects = {"item": SimpleNamespace(actor="item")}
    context._contact_force_cache = {}
    assert context.finger_contact_force("item") == pytest.approx((.01, .01))
    assert context.finger_contact_force("item") == pytest.approx((.01, .01))
    assert not context.is_grasping("item")
    grasped[0] = True
    assert context.is_grasping("item")
    grasped[0] = False
    assert not context.is_grasping("item")
    binding.bound_object_id = "item"
    assert context.is_grasping("item")
    binding.bound_object_id = None
    assert not context.is_grasping("item")
    assert calls == ["left", "right", "agent", "agent", "agent", "agent"]


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("override", [None, False, True])
@pytest.mark.parametrize("runtime_flag", [False, True])
def test_collection_cli_override_is_independent(monkeypatch, tmp_path, configured, override, runtime_flag):
    from exploration.cee_us import cee_us_cabinet as cli
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(dict(agent="raps_gnn_actor", collection_optimizations=configured,
                                         runtime_optimizations=runtime_flag)), path)
    captured = []
    monkeypatch.setattr(raps, "run", lambda args, config: captured.append(config) or 0)
    arguments = ["train", "--config", str(path)]
    if override is not None:
        arguments.append("--collection-optimizations" if override else "--no-collection-optimizations")
    assert cli.main(arguments) == 0
    assert captured[0].collection_optimizations == (configured if override is None else override)
    assert captured[0].runtime_optimizations == runtime_flag


@pytest.mark.parametrize("flag", ["--collection-optimizations", "--no-collection-optimizations"])
def test_collection_cli_rejects_wrong_agent(tmp_path, flag):
    from exploration.cee_us import cee_us_cabinet as cli
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(dict(agent="icem")), path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--config", str(path), flag])
    assert exc.value.code == 2


@pytest.mark.parametrize("saved", [None, False, True])
def test_collection_resume_mismatch_rejected_before_replay(tmp_path, saved):
    config = OmegaConf.create(dict(raps_actor={}, rounds=2))
    if saved is not None:
        config.collection_optimizations = saved
    directory = tmp_path / "curiosity"
    directory.mkdir()
    path = directory / "round_001.pt"
    torch.save(dict(schema=raps.SCHEMA, config=OmegaConf.to_container(config)), path)
    config.collection_optimizations = not bool(saved)
    with pytest.raises(ValueError, match="only rounds and evaluation_steps"):
        raps.resume_history(path, config)
    assert not path.with_suffix(".replay.npz").exists()


def test_fast_preset_changes_only_engineering_flags():
    base = OmegaConf.load(raps.ROOT / "configs/raps_gnn_actor_cabinet_grasp_only.yaml")
    fast = OmegaConf.load(raps.ROOT / "configs/raps_gnn_actor_cabinet_grasp_only_fast.yaml")
    assert fast.pop("collection_optimizations") is True
    assert fast.runtime_optimizations is True
    fast.runtime_optimizations = base.runtime_optimizations
    assert fast == base
