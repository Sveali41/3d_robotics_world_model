"""Bounded native five-pose planning smoke and one stationary open control."""
import os

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("RUN_NATIVE_CABINET") != "1", reason="native robotics runtime required")
def test_native_selected_orientations_preflight_without_physics_and_begin_open():
    from domain.robocasa.grasp_orientations import grasp_orientation_quaternion
    from domain.robocasa.motion_planning import _array, _angle
    from domain.robocasa.pick_place import PlannedPickPlaceSkillExecution
    from domain.robocasa.primitives import PrimitiveAction
    from exploration.cee_us.cee_us_cabinet import make_env

    env, _ = make_env(cabinet_grid=True, cabinet_door=False, centered_initial_tcp=True,
                      action_mode="joint_position")
    try:
        env.reset(seed=0)
        wrapper = env.env
        ctx = wrapper._context()
        raw = ctx.env
        reference = np.roll(wrapper._ee_orientation_target.as_quat(), 1)
        executor = PlannedPickPlaceSkillExecution(orientation_target=lambda: reference,
            grasp_orientation_candidates=True, grasp_retention="constraint")
        qpos, qvel = _array(raw.agent.robot.get_qpos()).copy(), _array(raw.agent.robot.get_qvel()).copy()
        items = [_array(item.pose.raw_pose).copy() for item in raw.cabinet_items]
        original_plan = raw.plan_to_pose
        accepted = []
        for index in range(5):
            selected = grasp_orientation_quaternion(reference, index)
            calls = []

            def checked_plan(target, quaternion, **kwargs):
                calls.append((np.asarray(target).copy(), np.asarray(quaternion).copy(), kwargs))
                if len(calls) > 1:
                    assert _angle(quaternion, selected) < 1e-5
                return original_plan(target, quaternion, **kwargs)

            raw.plan_to_pose = checked_plan
            action = PrimitiveAction("grasp", dict(x=.395, y=.06, z=.732, orientation_id=index))
            report = executor.preflight_grasp(action, wrapper._context())
            assert calls
            accepted.append(report["accepted"])
            np.testing.assert_array_equal(_array(raw.agent.robot.get_qpos()), qpos)
            np.testing.assert_array_equal(_array(raw.agent.robot.get_qvel()), qvel)
            for item, initial in zip(raw.cabinet_items, items):
                np.testing.assert_array_equal(_array(item.pose.raw_pose), initial)
            print(f"orientation {index}: {report['status']}, stage={report.get('stage', 'accepted')}")
        raw.plan_to_pose = original_plan
        assert accepted[0], "Previously reachable front-cell reset pose failed native preflight"
        # A nonzero selected angle must not trigger tracking failure while opening.
        executor.start(PrimitiveAction("grasp", dict(x=.395, y=.06, z=.732, orientation_id=1)),
                       wrapper._context())
        low = executor.next_action(wrapper._context())
        assert low.shape == (8,) and low[-1] == 1.
        _, _, terminated, truncated, info = env.step(low)
        executor.record_action(info["primitive_record"].low_actions[0],
                              terminated=terminated, truncated=truncated)
        assert executor.primitive.status == "running" and executor.primitive.stage == "open"
        assert executor.executed_steps == 1
    finally:
        env.close()
