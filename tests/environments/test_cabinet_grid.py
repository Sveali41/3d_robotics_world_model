"""Native fixed-grid placement checks without a training run."""
import os

import numpy as np
import pytest

from domain.robocasa.cabinet_grid import GRID_INITIAL_CELLS, GRID_XY_ROOT_M


@pytest.mark.skipif(os.environ.get("RUN_NATIVE_CABINET") != "1", reason="native robotics runtime required")
def test_native_grid_spawns_collision_centers_and_preserves_reset():
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from domain.robocasa.cabinet_grid import local_collision_bounds

    env, config = make_env(cabinet_grid=True, cabinet_door=False, centered_initial_tcp=True)
    raw = env.unwrapped
    try:
        env.reset(seed=0)
        assert config.cabinet_grid and not raw.cabinet_door
        assert len(raw.cabinet_items) == 2 and raw.door_joint is None
        boxes = []
        initial_poses = []
        for item in raw.cabinet_items:
            pose = raw.agent.robot.pose.inv() * item.pose
            position = pose.p.detach().cpu().numpy().reshape(3)
            bounds = local_collision_bounds(item)
            center = position + bounds.mean(axis=0)
            np.testing.assert_allclose(center[:2], GRID_XY_ROOT_M[GRID_INITIAL_CELLS[item.name]], atol=1e-6)
            assert position[2] + bounds[0, 2] >= .672
            # Every candidate cell fits this object's initial XY footprint, with 2cm edge clearance.
            offset = bounds - bounds.mean(axis=0)
            for xy in GRID_XY_ROOT_M:
                assert np.all(np.asarray(xy) + offset[0, :2] >= [.310, -.130])
                assert np.all(np.asarray(xy) + offset[1, :2] <= [.731, .370])
            boxes.append(position + bounds)
            initial_poses.append(item.pose.raw_pose.clone())
        assert np.any((boxes[0][1] <= boxes[1][0]) | (boxes[1][1] <= boxes[0][0]))
        encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
        contract = encoder.contract()
        assert encoder.encode().shape == (67,)
        assert len(contract["cabinet_grid"]["xy_root_m"]) == 12
        assert contract["cabinet_grid"]["initial_cells"] == GRID_INITIAL_CELLS
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[-1] = 1.
        env.step(action)
        env.reset(seed=123)
        for item, initial in zip(raw.cabinet_items, initial_poses):
            np.testing.assert_allclose(item.pose.raw_pose.cpu(), initial.cpu(), atol=1e-6)
    finally:
        env.close()


def test_grid_layout_resolver():
    from domain.robocasa.cabinet_grid import (
        COMPACT_GRID_XY_ROOT_M, COMPACT_GRID_INITIAL_CELLS,
        TRAPEZOID_GRID_XY_ROOT_M, TRAPEZOID_GRID_INITIAL_CELLS, resolve_grid_layout,
    )
    assert resolve_grid_layout() == (GRID_XY_ROOT_M, GRID_INITIAL_CELLS)
    assert resolve_grid_layout('compact_3x4') == (COMPACT_GRID_XY_ROOT_M, COMPACT_GRID_INITIAL_CELLS)
    assert COMPACT_GRID_INITIAL_CELLS == {'cabinet_item_16': 4, 'cabinet_item_17': 7}
    assert resolve_grid_layout('trapezoid_4_3_2_1') == (TRAPEZOID_GRID_XY_ROOT_M, TRAPEZOID_GRID_INITIAL_CELLS)
    assert TRAPEZOID_GRID_XY_ROOT_M == ((.335, -.06), (.335, .06), (.335, .18), (.335, .30),
                                       (.410, 0.), (.410, .12), (.410, .24),
                                       (.485, .06), (.485, .18), (.560, .12))
    assert TRAPEZOID_GRID_INITIAL_CELLS == {'cabinet_item_16': 4, 'cabinet_item_17': 6}
    with pytest.raises(ValueError, match='Unknown cabinet grid layout'):
        resolve_grid_layout('unknown')


@pytest.mark.skipif(os.environ.get('RUN_NATIVE_CABINET') != '1', reason='native robotics runtime required')
def test_native_compact_grid_actor_origins_and_contract():
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from domain.robocasa.cabinet_grid import (
        COMPACT_GRID_XY_ROOT_M, COMPACT_GRID_INITIAL_CELLS, local_collision_bounds,
    )

    env, config = make_env(cabinet_grid=True, cabinet_grid_layout='compact_3x4',
                           cabinet_door=False, centered_initial_tcp=True)
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        assert config.cabinet_grid_layout == raw.cabinet_grid_layout == 'compact_3x4'
        boxes = []
        for item in raw.cabinet_items:
            position = (raw.agent.robot.pose.inv() * item.pose).p.detach().cpu().numpy().reshape(3)
            bounds = local_collision_bounds(item)
            np.testing.assert_allclose(position[:2], COMPACT_GRID_XY_ROOT_M[COMPACT_GRID_INITIAL_CELLS[item.name]], atol=1e-6)
            assert position[2] + bounds[0, 2] >= .672
            for xy in COMPACT_GRID_XY_ROOT_M:
                assert np.all(np.asarray(xy) + bounds[0, :2] >= [.310, -.130])
                assert np.all(np.asarray(xy) + bounds[1, :2] <= [.731, .370])
            boxes.append(position + bounds)
        assert np.any((boxes[0][1] <= boxes[1][0]) | (boxes[1][1] <= boxes[0][0]))
        contract = CabinetSkillStateEncoder(env, latched_gripper=True).contract()['cabinet_grid']
        assert contract['layout'] == 'compact_3x4'
        assert contract['initial_cells'] == COMPACT_GRID_INITIAL_CELLS
        assert contract['initial_xy_reference'] == 'native_actor_origin; fixed_per_object_on_reset'
    finally:
        env.close()


def test_main_config_defaults_to_trapezoid():
    from pathlib import Path
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / 'configs/exploration.yaml')
    assert cfg.cabinet_grid is True
    assert cfg.cabinet_grid_layout == 'trapezoid_4_3_2_1'
    assert cfg.cabinet_item_layout == 'single_upper_second_row'


@pytest.mark.skipif(os.environ.get('RUN_NATIVE_CABINET') != '1', reason='native robotics runtime required')
def test_native_main_default_trapezoid_cup_only_scene():
    from pathlib import Path
    from omegaconf import OmegaConf
    from main import build_env
    from domain.robocasa.cabinet_grid import local_collision_bounds

    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / 'configs/exploration.yaml')
    env = build_env(cfg, raw_env=True)
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        assert raw.cabinet_item_layout == 'single_upper_second_row'
        assert [item.name for item in raw.cabinet_items] == ['cabinet_item_16']
        assert raw.cabinet_grid_layout == 'trapezoid_4_3_2_1'
        cup = raw.cabinet_items[0]
        position = (raw.agent.robot.pose.inv() * cup.pose).p.cpu().numpy().reshape(3)
        np.testing.assert_allclose(position[:2], [.410, 0.], atol=1e-6)
        assert position[2] + local_collision_bounds(cup)[0, 2] == pytest.approx(.674, abs=1e-6)
        assert not bool(raw.agent.is_grasping(cup).item())
        initial = cup.pose.raw_pose.clone()
        env.reset(seed=123)
        np.testing.assert_allclose(cup.pose.raw_pose.cpu(), initial.cpu(), atol=1e-6)
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_NATIVE_CABINET') != '1', reason='native robotics runtime required')
def test_native_make_env_single_cup_encoder_contract():
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetStateEncoder, CabinetSkillStateEncoder

    env, config = make_env(cabinet_item_layout='single_upper_second_row', cabinet_grid=True,
                           cabinet_grid_layout='trapezoid_4_3_2_1')
    try:
        env.reset(seed=0)
        assert config.cabinet_item_layout == env.unwrapped.cabinet_item_layout == 'single_upper_second_row'
        assert [item.name for item in env.unwrapped.cabinet_items] == ['cabinet_item_16']
        encoder = CabinetStateEncoder(env)
        contract = encoder.contract()
        assert encoder.encode().shape == (49,) and contract['dimension'] == 49
        assert contract['object_names'] == ['cabinet_item_16']
        assert contract['object_types'] == ['mug']
        assert contract['cabinet_grid']['initial_cells'] == {'cabinet_item_16': 4}
        assert len(contract['cabinet_grid']['xy_root_m']) == 10
        latched = CabinetSkillStateEncoder(env, latched_gripper=True)
        assert latched.encode().shape == (50,) and latched.contract()['dimension'] == 50
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get('RUN_NATIVE_CABINET') != '1', reason='native robotics runtime required')
def test_native_main_default_trapezoid_two_item_scene():
    from pathlib import Path
    from omegaconf import OmegaConf
    from main import build_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
    from domain.robocasa.cabinet_grid import (
        TRAPEZOID_GRID_XY_ROOT_M, TRAPEZOID_GRID_INITIAL_CELLS, local_collision_bounds,
    )

    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / 'configs/exploration.yaml')
    cfg.cabinet_item_layout = 'two_upper_second_row'
    env = build_env(cfg, raw_env=True)
    geometry = {}
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        assert raw.cabinet_grid is True and raw.cabinet_grid_layout == 'trapezoid_4_3_2_1'
        assert len(raw.cabinet_items) == 2
        grid_contract = CabinetSkillStateEncoder(env, latched_gripper=True).contract()['cabinet_grid']
        assert grid_contract['rows'] == 4 and grid_contract['row_sizes'] == [4, 3, 2, 1]
        assert grid_contract['xy_root_m'] == [list(xy) for xy in TRAPEZOID_GRID_XY_ROOT_M]
        assert grid_contract['initial_cells'] == TRAPEZOID_GRID_INITIAL_CELLS
        boxes = []
        initial_poses = []
        for item in raw.cabinet_items:
            position = (raw.agent.robot.pose.inv() * item.pose).p.detach().cpu().numpy().reshape(3)
            bounds = local_collision_bounds(item)
            xy = np.asarray(TRAPEZOID_GRID_XY_ROOT_M[TRAPEZOID_GRID_INITIAL_CELLS[item.name]])
            np.testing.assert_allclose(position[:2], xy, atol=1e-6)
            cell_center = xy + [0., .0145]
            cell_lower = cell_center - [.0375, .06]
            cell_upper = cell_center + [.0375, .06]
            assert np.all(position[:2] + bounds[0, :2] >= cell_lower - 1e-6)
            assert np.all(position[:2] + bounds[1, :2] <= cell_upper + 1e-6)
            assert np.all(position[:2] + bounds[0, :2] >= [.29, -.15])
            assert np.all(position[:2] + bounds[1, :2] <= [.751, .39])
            assert position[2] + bounds[0, 2] >= .672
            assert not bool(raw.agent.is_grasping(item).item())
            boxes.append(position + bounds)
            initial_poses.append(item.pose.raw_pose.clone())
            geometry[item.name] = (bounds, float(item._bodies[0].mass),
                                   len(item._bodies[0].get_collision_shapes()))
        assert np.any((boxes[0][1] <= boxes[1][0]) | (boxes[1][1] <= boxes[0][0]))
        env.reset(seed=123)
        for item, initial in zip(raw.cabinet_items, initial_poses):
            np.testing.assert_allclose(item.pose.raw_pose.cpu(), initial.cpu(), atol=1e-6)
    finally:
        env.close()

    cfg.cabinet_grid_layout = 'legacy_3x4'
    env = build_env(cfg, raw_env=True)
    try:
        env.reset(seed=0)
        for item in env.unwrapped.cabinet_items:
            bounds, mass, shape_count = geometry[item.name]
            np.testing.assert_allclose(local_collision_bounds(item), bounds, atol=1e-8)
            assert float(item._bodies[0].mass) == pytest.approx(mass)
            assert len(item._bodies[0].get_collision_shapes()) == shape_count
    finally:
        env.close()
