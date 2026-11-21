"""Opt-in native scene checks; two stationary controls, no training."""
import os

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("RUN_NATIVE_CABINET") != "1", reason="native robotics runtime required")
@pytest.mark.parametrize("cabinet_door", [False, True])
def test_cabinet_door_geometry_state_and_reset(cabinet_door):
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder

    env, config = make_env(cabinet_door=cabinet_door, centered_initial_tcp=True)
    raw = env.unwrapped
    try:
        env.reset(seed=0)
        assert config.cabinet_door == cabinet_door
        assert (raw.door_link is not None) == cabinet_door
        assert (raw.door_joint is not None) == cabinet_door
        assert set(raw.cabinet.links_map) == ({"frame", "door"} if cabinet_door else {"frame"})
        frame = raw.cabinet.links_map["frame"]
        assert len(frame._objs[0].get_collision_shapes()) == 8
        encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
        state = encoder.encode()
        assert state.shape == (67,) and np.isfinite(state).all()
        assert ("cabinet_door" not in encoder.contract()) == cabinet_door
        if not cabinet_door:
            np.testing.assert_array_equal(state[30:32], [0., 0.])
            assert "handle_position" not in raw.evaluate()
        else:
            assert "handle_position" in raw.evaluate()
        from domain.robocasa.motion_planning import _native_modules
        planner = raw.motion_planner
        planner.mplib, planner.fcl, planner.sapien = _native_modules()
        planner.root = planner._pose(planner.root_link.pose)
        planner.attachment = None
        planner._scene_geometry()
        names = [obstacle["name"] for obstacle in planner.obstacles]
        assert any(name.startswith("cabinet/door/") for name in names) == cabinet_door
        assert sum(name.startswith("cabinet/frame/") for name in names) == 8
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[-1] = 1.
        env.step(action)
        env.reset(seed=0)
        assert np.isfinite(encoder.encode()).all()
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get("RUN_NATIVE_CABINET") != "1", reason="native robotics runtime required")
def test_default_cup_scene_has_no_door_or_planner_door_obstacles():
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from domain.robocasa.motion_planning import _native_modules

    env, config = make_env(centered_initial_tcp=True, cabinet_grid=True,
                           cabinet_grid_layout="trapezoid_4_3_2_1",
                           cabinet_item_layout="single_upper_second_row")
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        assert config.cabinet_door is False
        assert raw.door_link is None and raw.door_joint is None
        assert set(raw.cabinet.links_map) == {"frame"}
        encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
        state = encoder.encode()
        assert state.shape == (50,)
        np.testing.assert_array_equal(state[30:32], [0., 0.])
        assert encoder.contract()["cabinet_door"] is False
        planner = raw.motion_planner
        planner.mplib, planner.fcl, planner.sapien = _native_modules()
        planner.root = planner._pose(planner.root_link.pose)
        planner.attachment = None
        planner._scene_geometry()
        assert not any(o["name"].startswith("cabinet/door/") for o in planner.obstacles)
        assert sum(o["name"].startswith("cabinet/frame/") for o in planner.obstacles) == 8
    finally:
        env.close()
