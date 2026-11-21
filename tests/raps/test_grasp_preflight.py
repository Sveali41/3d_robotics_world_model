"""Dry-run grasp planning and policy-level candidate filtering."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from domain.robocasa.motion_planning import MotionPlanningError
from domain.robocasa.pick_place import PlannedPickPlaceSkillExecution
from domain.robocasa.primitives import PrimitiveAction
from exploration.raps.raps_gnn_actor import (
    DEFAULTS, SkillCollector, action_contract, skill_schema,
)


def _grasp_vector(schema=None):
    schema = schema or skill_schema({"pick_place_primitives": True})
    return schema.encode(PrimitiveAction("grasp", {"x": .5, "y": .12, "z": .78}))


def _place_vector(schema):
    return schema.encode(PrimitiveAction("place", {"x": .5, "y": .12}))


class _FakePlannerModel:
    def __init__(self):
        self.fk_inputs = []
        self.end_link_idx = 4

    def compute_forward_kinematics(self, qpos):
        self.fk_inputs.append(np.asarray(qpos).copy())

    def get_link_pose(self, _):
        # Keep every route stage away from the already-at-target shortcut.
        return SimpleNamespace(p=np.array([1.0, 1.0, 1.0]), q=np.array([1., 0., 0., 0.]))


def _fake_preflight_context(plan_to_pose):
    live = np.arange(9, dtype=float) / 10
    model = _FakePlannerModel()
    planner = SimpleNamespace(kin=SimpleNamespace(pmodel=model, end_link_idx=model.end_link_idx),
                              pindices=np.arange(7))

    class Robot:
        def get_qpos(self):
            return live.copy()

        def set_qpos(self, _):
            raise AssertionError("preflight must not mutate the live robot")

    class Raw:
        motion_planner = planner
        agent = SimpleNamespace(
            robot=Robot(),
            _controller_configs={"pd_joint_pos": {"gripper": SimpleNamespace(upper=.8)}},
        )

        def plan_to_pose(self, *args, **kwargs):
            return plan_to_pose(*args, **kwargs)

        def step(self, *_):
            raise AssertionError("preflight must not advance physics")

    return SimpleNamespace(env=Raw(), ee_position=np.array([.4, .15, .82]),
                           objects={"item": object()}), live, model


def _execution():
    execution = PlannedPickPlaceSkillExecution.__new__(PlannedPickPlaceSkillExecution)
    execution.orientation_target = lambda: np.array([1., 0., 0., 0.])
    execution.planning_seed = 17
    execution.plan_counter = 3
    return execution


def test_grasp_preflight_chains_virtual_open_hand_plans_and_restores_fk():
    calls = []

    def plan_to_pose(target, orientation, **kwargs):
        calls.append((np.asarray(target).copy(), np.asarray(orientation).copy(), kwargs))
        start = np.asarray(kwargs["start_qpos"])
        end = start + len(calls) / 100
        return SimpleNamespace(position=np.stack((start, end)))

    context, live, model = _fake_preflight_context(plan_to_pose)
    result = _execution().preflight_grasp(
        skill_schema({"pick_place_primitives": True}).decode(_grasp_vector()), context)

    assert result["accepted"] is True
    assert [stage["stage"] for stage in result["stages"]] == ["retreat", "align", "approach"]
    assert [call[2]["method"] for call in calls] == ["rrt_connect", "rrt_connect", "bounded_ik"]
    assert all(call[2]["maintain_orientation"] is True for call in calls)
    assert all(call[2]["allowed_contact_objects"] == ("item",) for call in calls)
    first_start = calls[0][2]["start_qpos"]
    np.testing.assert_array_equal(first_start[:7], live[:7])
    np.testing.assert_array_equal(first_start[7:], [.8, .8])
    np.testing.assert_allclose(calls[1][2]["start_qpos"], first_start + .01)
    np.testing.assert_allclose(calls[2][2]["start_qpos"], first_start + .03)
    assert [call[2]["seed"] for call in calls] == [20, 21, 22]
    np.testing.assert_array_equal(model.fk_inputs[-1], live[:7])
    assert len(model.fk_inputs) == 4


def test_grasp_preflight_reports_failed_stage_and_still_restores_fk():
    calls = []

    def plan_to_pose(target, orientation, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise MotionPlanningError("path_collision", "blocked route")
        start = np.asarray(kwargs["start_qpos"])
        return SimpleNamespace(position=np.stack((start, start + .01)))

    context, live, model = _fake_preflight_context(plan_to_pose)
    result = _execution().preflight_grasp(
        skill_schema({"pick_place_primitives": True}).decode(_grasp_vector()), context)

    assert result == {
        "accepted": False, "stage": "align", "status": "path_collision",
        "detail": "blocked route", "stages": [{"stage": "retreat", "status": "planned", "controls": 2}],
    }
    np.testing.assert_array_equal(model.fk_inputs[-1], live[:7])


def _collector_for_candidates(tmp_path, proposals, results, *, held=False):
    settings = {**DEFAULTS, "pick_place_primitives": True, "motion_execution": "planner",
                "latched_gripper": True, "grasp_stop_on_contact": False,
                "grasp_preflight": True, "grasp_preflight_max_candidates": 4}
    schema = skill_schema(settings)

    class Context:
        objects = {"item": object()}
        ee_position = np.array([.4, .15, .82])

        def is_grasping(self, _):
            return held

    class Env:
        action_mode = "joint_position"

        def __init__(self):
            self.env = self
            self.steps = 0

        def _context(self):
            return Context()

        def reset(self, **_):
            pass

        def step(self, *_):
            self.steps += 1
            raise AssertionError("rejected preflight candidates must not reach env.step")

    class Execution:
        gripper_command = -1.

        def __init__(self):
            self.preflight_actions = []
            self.started = []

        def reset(self):
            pass

        def preflight_grasp(self, action, context):
            self.preflight_actions.append(action)
            return results.pop(0)

        def start(self, action, context):
            self.started.append(action)

    class Actor:
        def __init__(self):
            self.settings = settings
            self.calls = []
            self.proposed_actions = []

        def action(self, state):
            self.calls.append(state)
            action = proposals.pop(0)
            self.proposed_actions.append(action.copy())
            return action

    collector = SkillCollector.__new__(SkillCollector)
    collector.collection_optimizations = False
    collector.profile_timings = {}
    collector._context_cache = None
    collector.env = Env()
    collector.actor = Actor()
    collector.schema = schema
    collector.directory = tmp_path
    collector.execution = Execution()
    collector.encoder = SimpleNamespace(gripper_command=-1., encode=lambda: np.zeros(67, np.float32))
    collector.state = np.zeros(67, dtype=np.float32)
    collector.episode_id = 2
    collector.skill_id = 4
    collector.total_steps = 9
    collector.planned_motion = True
    return collector


def test_collector_resamples_policy_and_audits_place_precondition(tmp_path):
    schema = skill_schema({"pick_place_primitives": True})
    proposals = [_place_vector(schema), _grasp_vector(schema), _grasp_vector(schema)]
    collector = _collector_for_candidates(
        tmp_path, proposals, [dict(accepted=False, status="path_collision"),
                              dict(accepted=True, status="approach_route_feasible")])

    requested, actual, decoded = collector._select_preflight_action("task", "validation")

    assert decoded.primitive_type == "grasp"
    assert len(collector.actor.calls) == 3
    assert [a.primitive_type for a in collector.execution.preflight_actions] == ["grasp", "grasp"]
    assert collector.execution.started == []
    rows = [json.loads(line) for line in (tmp_path / "preflight.jsonl").read_text().splitlines()]
    assert [row["result"]["status"] for row in rows] == ["missing_grasp", "path_collision",
                                                            "approach_route_feasible"]
    np.testing.assert_array_equal(collector.schema.encode(decoded), actual)
    np.testing.assert_array_equal(requested, collector.actor.proposed_actions[-1])


def test_collector_exhaustion_does_not_start_or_step_and_has_no_fallback(tmp_path):
    schema = skill_schema({"pick_place_primitives": True})
    grasp = _grasp_vector(schema)
    collector = _collector_for_candidates(tmp_path, [grasp.copy(), grasp.copy()],
                                          [dict(accepted=False, status="path_collision"),
                                           dict(accepted=False, status="path_collision")])
    collector.actor.settings["grasp_preflight_max_candidates"] = 2

    with pytest.raises(RuntimeError, match="no random fallback"):
        collector.collect(1, "curiosity", seed=0, stage="train")

    assert len(collector.actor.calls) == 2
    assert collector.execution.started == []
    assert collector.env.steps == 0
    rows = [json.loads(line) for line in (tmp_path / "preflight.jsonl").read_text().splitlines()]
    assert [row["candidate"] for row in rows] == [1, 2]
    assert {row["policy"] for row in rows} == {"curiosity"}


def test_grasp_preflight_contract_is_opt_in_and_records_its_limits():
    macro = {**DEFAULTS, "pick_place_primitives": True, "motion_execution": "planner",
             "latched_gripper": True, "grasp_stop_on_contact": False}
    disabled = action_contract(macro)
    assert "grasp_preflight" not in disabled
    enabled = action_contract({**macro, "grasp_preflight": True,
                               "grasp_preflight_max_candidates": 3})
    assert enabled["grasp_preflight"] == {
        "version": 1,
        "route": "open_hand_retreat_align_approach_in_current_scene",
        "proposal": "same_policy_rejection_sampling_no_random_fallback",
        "max_candidates": 3,
        "invalid_preconditions": "reject_before_physics",
        "rejected_transitions": "audit_only",
        "imagination": "unchanged_unfiltered",
        "checked_plans": "discard_and_replan_before_execution",
    }
    assert "grasp_preflight" not in DEFAULTS


@pytest.mark.parametrize("settings,match", [
    ({"grasp_preflight": "yes"}, "must be boolean"),
    ({"grasp_preflight_max_candidates": True}, "positive integer"),
    ({"grasp_preflight_max_candidates": 1.5}, "positive integer"),
    ({"grasp_preflight_max_candidates": 0}, "positive integer"),
    ({"grasp_preflight": True}, "requires planned pick_place_primitives"),
])
def test_actor_rejects_invalid_preflight_settings(settings, match):
    from exploration.raps.raps_gnn_actor import RAPSImaginationActor

    backend = SimpleNamespace(env=SimpleNamespace(agent_dim=32), action_dim=10)
    with pytest.raises(ValueError, match=match):
        RAPSImaginationActor(backend, settings)
