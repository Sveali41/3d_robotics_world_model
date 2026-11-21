"""Discrete grasp rotations and matching exterior-alignment execution checks."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from domain.robocasa.grasp_orientations import (
    GRASP_ORIENTATION_YAW_PITCH_DEG, grasp_orientation_quaternion,
    validate_grasp_orientation_id,
)
from domain.robocasa.motion_planning import MotionPlanningError
from domain.robocasa.pick_place import PlannedPickPlaceSkillExecution
from domain.robocasa.primitives import PrimitiveAction
from tests.environments.test_pick_place_primitives import planned_execution, action
from tests.raps.test_grasp_preflight import _fake_preflight_context, _execution


@pytest.mark.parametrize("index", range(5))
def test_candidate_rotates_about_root_axes_in_declared_order(index):
    reset = Rotation.from_euler("xyz", [27., -11., 83.], degrees=True)
    yaw, pitch = GRASP_ORIENTATION_YAW_PITCH_DEG[index]
    a, b = np.deg2rad([yaw, pitch])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    actual = grasp_orientation_quaternion(np.roll(reset.as_quat(), 1), index)
    np.testing.assert_allclose(Rotation.from_quat(np.roll(actual, -1)).as_matrix(),
                               rz @ ry @ reset.as_matrix(), atol=1e-14)


@pytest.mark.parametrize("value", [-1, 5, .5, np.nan, np.inf, True, "1", None, [1]])
def test_orientation_id_is_strict(value):
    with pytest.raises(ValueError, match="orientation_id"):
        validate_grasp_orientation_id(value)


def test_grasp_preserves_action_target_and_rotates_only_during_exterior_alignment(monkeypatch):
    executor, ctx, _ = planned_execution(monkeypatch)
    executor.grasp_orientation_candidates = True
    current = np.roll(Rotation.from_euler("z", 20, degrees=True).as_quat(), 1)
    executor._feedback_orientation = lambda: current.copy()
    selected = PrimitiveAction("grasp", dict(x=.395, y=-.06, z=.732, orientation_id=3))
    executor.start(selected, ctx)
    assert executor.primitive.action == selected
    np.testing.assert_allclose(executor.goal, [.395, -.06, .732])
    np.testing.assert_array_equal(executor._orientation, current)
    executor._phase("retreat", ctx)
    np.testing.assert_array_equal(executor._orientation, current)
    executor._phase("align", ctx)
    expected = grasp_orientation_quaternion([1, 0, 0, 0], 3)
    np.testing.assert_allclose(executor._orientation, expected)
    assert executor._rotation_allowed([.289, 0, 0])
    assert not executor._rotation_allowed([.29, 0, 0])
    for stage in ("approach", "close", "confirm", "lift", "withdraw", "intermediate"):
        executor._phase(stage, ctx)
        np.testing.assert_allclose(executor._orientation, expected)
        assert not executor._rotation_allowed([.18, 0, 0])
    with pytest.raises(ValueError, match="orientation_id"):
        executor.start(action("grasp"), ctx)


def test_place_recomputes_payload_bounds_at_reset_orientation(monkeypatch):
    executor, ctx, _ = planned_execution(monkeypatch)
    executor.grasp_orientation_candidates = True
    current = grasp_orientation_quaternion([1., 0., 0., 0.], 4)
    executor._feedback_orientation = lambda: current.copy()
    ctx.held = True
    executor.gripper_command = -1.
    confirmed_pose = [.01, .02, -.01, 1., 0., 0., 0.]
    executor._place_reference = dict(name="generic", tcp_to_object_pose=confirmed_pose,
                                     bounds_tcp_offset_m=[[0, 0, 0], [0, 0, 0]])
    calls = []
    original = ctx.env.motion_planner.placement_bounds
    ctx.env.motion_planner.placement_bounds = lambda attachment, q: (
        calls.append(np.asarray(q).copy()) or original(attachment, q))
    executor.start(action("place"), ctx)
    assert executor.primitive.stage == "retreat"
    assert executor._place_reference["name"] == "generic"
    assert executor._place_reference["tcp_to_object_pose"] == confirmed_pose
    np.testing.assert_allclose(calls, [[1, 0, 0, 0]])
    np.testing.assert_allclose(executor._orientation, current)
    executor._phase("preplace", ctx)
    np.testing.assert_allclose(executor._orientation, [1, 0, 0, 0])


def test_rejected_grasp_while_holding_preserves_measured_orientation(monkeypatch):
    executor, ctx, _ = planned_execution(monkeypatch)
    executor.grasp_orientation_candidates = True
    current = grasp_orientation_quaternion([1, 0, 0, 0], 4)
    executor._feedback_orientation = lambda: current.copy()
    ctx.held = True
    executor.gripper_command = -1.
    executor.start(PrimitiveAction("grasp", dict(x=.395, y=-.06, z=.732, orientation_id=1)), ctx)
    assert executor.primitive.stage == "invalid" and executor._rejection == "already_holding"
    np.testing.assert_allclose(executor._orientation, current)
    control = executor.next_action(ctx)
    assert control[-1] == -1.
    executor.record_action(control)
    assert executor.primitive.status == "running"
    assert executor.next_action(ctx) is None
    assert executor.primitive.status == "already_holding"


def test_execution_plans_selected_pose_and_does_not_switch_on_failure(monkeypatch):
    executor, ctx, _ = planned_execution(monkeypatch)
    executor.grasp_orientation_candidates = True
    executor.start(PrimitiveAction("grasp", dict(x=.395, y=-.06, z=.732, orientation_id=3)), ctx)
    selected = grasp_orientation_quaternion([1, 0, 0, 0], 3)
    model = ctx.env.motion_planner.kin.pmodel
    model.get_link_pose = lambda _: SimpleNamespace(p=model.q[:3], q=selected.copy())
    calls = []
    original = ctx.env.plan_to_pose

    def plan(goal, q, **kwargs):
        calls.append((np.asarray(goal).copy(), np.asarray(q).copy(), kwargs))
        return original(goal, q, **kwargs)

    ctx.env.plan_to_pose = plan
    executor._phase("align", ctx)
    control, reached = executor._motion_action(ctx, executor._anchor, 1.)
    assert control.shape == (8,) and not reached
    np.testing.assert_allclose(calls[-1][1], selected)
    assert calls[-1][2]["maintain_orientation"] is False
    assert calls[-1][2]["densify_rrt_path"] is True
    executor._phase("approach", ctx)
    control, reached = executor._motion_action(ctx, executor._anchor, 1.)
    np.testing.assert_allclose(calls[-1][0], [.395, -.06, .732])
    np.testing.assert_allclose(calls[-1][1], selected)
    assert calls[-1][2]["maintain_orientation"] is True
    assert calls[-1][2]["method"] == "bounded_ik"
    assert executor.motion_cursor.plan.metadata["allow_open_grasp_contact"] is True
    executor._phase("approach", ctx)
    failed_calls = []

    def reject(goal, q, **kwargs):
        failed_calls.append(q.copy())
        raise MotionPlanningError("no_collision_free_ik", "blocked selected orientation")

    ctx.env.plan_to_pose = reject
    executor._motion_action(ctx, executor._anchor, 1.)
    assert len(failed_calls) == 1
    np.testing.assert_allclose(failed_calls[0], selected)
    assert executor._pending_failure == "no_collision_free_ik"


@pytest.mark.parametrize("index", range(5))
def test_preflight_uses_selected_pose_after_current_orientation_retreat_and_restores_fk(index):
    calls = []
    context, live, model = _fake_preflight_context(lambda *_args, **_kwargs: None)
    model.compute_forward_kinematics = lambda q: setattr(model, "q", np.asarray(q).copy())
    model.get_link_pose = lambda _: SimpleNamespace(p=model.q[:3], q=model.q[3:7]/np.linalg.norm(model.q[3:7]))

    def plan_to_pose(target, orientation, **kwargs):
        calls.append((np.asarray(target).copy(), orientation.copy(), kwargs))
        start = kwargs["start_qpos"].copy()
        end = start.copy()
        end[:3], end[3:7] = target, orientation
        return SimpleNamespace(position=np.stack([start, end]), metadata={})

    context.env.plan_to_pose = plan_to_pose
    executor = _execution()
    executor.grasp_orientation_candidates = True
    candidate = PrimitiveAction("grasp", dict(x=.395, y=-.06, z=.732, orientation_id=index))
    result = executor.preflight_grasp(candidate, context)
    assert result["accepted"]
    assert len(calls) == 3
    np.testing.assert_allclose(calls[0][1], live[3:7]/np.linalg.norm(live[3:7]))
    selected = grasp_orientation_quaternion([1, 0, 0, 0], index)
    np.testing.assert_allclose(calls[1][1], selected)
    np.testing.assert_allclose(calls[2][1], selected)
    np.testing.assert_allclose(calls[2][0], [.395, -.06, .732])
    assert [c[2]["maintain_orientation"] for c in calls] == [True, False, True]
    assert calls[1][2]["densify_rrt_path"] is True
    assert calls[2][2]["method"] == "bounded_ik"
    np.testing.assert_array_equal(model.q, live[:7])


def test_alignment_audit_rejects_rotation_inside_and_wrong_endpoint(monkeypatch):
    executor, ctx, _ = planned_execution(monkeypatch)
    executor.grasp_orientation_candidates = True
    executor.start(PrimitiveAction("grasp", dict(x=.395, y=-.06, z=.732, orientation_id=0)), ctx)
    executor._phase("align", ctx)
    model = ctx.env.motion_planner.kin.pmodel
    model.get_link_pose = lambda _: SimpleNamespace(p=model.q[:3],
        q=np.array([1, 0, 0, 0]) if model.q[3] == 0 else np.array([0, 1, 0, 0]))
    points = np.zeros((3, 9)); points[:, 0] = [.18, .20, .18]; points[1, 3] = .1
    executor._audit_path(SimpleNamespace(position=points, metadata={}))
    points[1, 0] = .30
    with pytest.raises(MotionPlanningError, match="orientation_path_violation"):
        executor._audit_path(SimpleNamespace(position=points, metadata={}))
    points[1, 0] = .20; points[-1, 3] = .1
    with pytest.raises(MotionPlanningError, match="orientation_path_violation"):
        executor._audit_path(SimpleNamespace(position=points, metadata={}))


def test_legacy_executor_rejects_orientation_argument(monkeypatch):
    executor, ctx, _ = planned_execution(monkeypatch)
    with pytest.raises(ValueError, match="requires exactly"):
        executor.start(PrimitiveAction("grasp", dict(x=.395, y=-.06, z=.732, orientation_id=0)), ctx)
