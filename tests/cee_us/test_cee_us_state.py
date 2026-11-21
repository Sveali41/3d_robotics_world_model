from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from exploration.cee_us.cee_us_state import (
    AGENT_DIM,
    OBJECT_DYN_DIM,
    OBJECT_NAMES,
    OBJECT_STAT_DIM,
    CabinetStateEncoder,
    CabinetSkillStateEncoder,
)


def _pose(position, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    # SAPIEN/ManiSkill quaternion order is wxyz.
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return SimpleNamespace(p=np.asarray(position), q=np.roll(xyzw, 1))


def _actor(name, position, linear, angular, extents, rotation=None):
    return SimpleNamespace(
        name=name,
        pose=_pose(position, rotation),
        linear_velocity=np.asarray(linear),
        angular_velocity=np.asarray(angular),
        get_first_collision_mesh=lambda to_world_frame: SimpleNamespace(
            extents=np.asarray(extents)
        ),
    )


def _env():
    root_rotation = Rotation.from_euler("z", np.pi / 2).as_matrix()
    root_position = np.array([2.0, -1.0, 0.5])
    root_pose = _pose(root_position, root_rotation)

    def world(local):
        return root_position + root_rotation @ np.asarray(local)

    mug = _actor(
        "cabinet_item_16", world([1, 2, 3]), [2, 4, 6], [1, 3, 5],
        [0.1, 0.2, 0.3], root_rotation @ Rotation.from_euler("y", 0.2).as_matrix(),
    )
    bowl = _actor(
        "cabinet_item_17", world([-1, 0, 1]), [0, 2, 0], [0, 0, 4],
        [0.4, 0.5, 0.6], root_rotation,
    )
    tcp = SimpleNamespace(
        pose=_pose(world([0.5, 0.25, -0.5]), root_rotation),
        linear_velocity=root_rotation @ np.array([2.0, 0.0, 0.0]),
        angular_velocity=root_rotation @ np.array([0.0, 4.0, 0.0]),
    )
    controller = SimpleNamespace(root_link=SimpleNamespace(pose=root_pose))
    robot = SimpleNamespace(qpos=np.arange(9), qvel=np.arange(10, 19))
    return SimpleNamespace(
        control_freq=20,
        agent=SimpleNamespace(
            controller=SimpleNamespace(controllers={"arm": controller}),
            robot=robot,
            tcp=tcp,
        ),
        cabinet_items=[bowl, mug],  # Deliberately unsorted source order.
        cabinet_item_types=["bowl", "mug"],
        door_joint=SimpleNamespace(qpos=np.array([0.7]), qvel=np.array([2.0])),
    ), mug, bowl, controller


def test_state_layout_root_transform_and_dt_scaling():
    env, mug, bowl, _ = _env()
    state = CabinetStateEncoder(env, dt=0.05).encode()
    assert state.dtype == np.float32
    assert state.shape == (66,)
    np.testing.assert_allclose(state[:9], np.arange(9))
    np.testing.assert_allclose(state[9:18], np.arange(10, 19) * 0.05)
    np.testing.assert_allclose(state[18:21], [0.5, 0.25, -0.5])
    np.testing.assert_allclose(state[21:24], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(state[24:27], [0.1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(state[27:30], [0, 0.2, 0], atol=1e-6)
    np.testing.assert_allclose(state[30:32], [0.7, 0.1])
    # Dynamics are sorted by actor name, and world vectors are rotated into root.
    np.testing.assert_allclose(state[32:35], [1, 2, 3])
    np.testing.assert_allclose(state[38:41], [0.2, -0.1, 0.3], atol=1e-6)
    np.testing.assert_allclose(state[41:44], [0.15, -0.05, 0.25], atol=1e-6)
    np.testing.assert_allclose(state[44:47], [-1, 0, 1], atol=1e-6)
    np.testing.assert_allclose(state[56:66], [1, 0, 0.1, 0.2, 0.3, 0, 1, 0.4, 0.5, 0.6])
    assert mug.name == OBJECT_NAMES[0] and bowl.name == OBJECT_NAMES[1]


def test_skill_command_is_explicit_agent_state_and_legacy_layout_is_unchanged():
    env, *_ = _env()
    original = CabinetStateEncoder(env).encode()
    legacy = CabinetSkillStateEncoder(env)
    np.testing.assert_array_equal(original, legacy.encode())
    encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
    encoder.gripper_command = -1.0
    state = encoder.encode()
    assert state.shape == (67,) and state[32] == -1
    np.testing.assert_array_equal(np.delete(state, 32), original)
    contract = encoder.contract()
    assert contract["agent_dim"] == 33 and contract["dimension"] == 67
    assert contract["fields"][8]["name"] == "last_gripper_command"
    encoder.gripper_command = float("nan")
    with pytest.raises(ValueError, match="gripper command"):
        encoder.encode()


def test_single_cup_state_and_grid_contract():
    env, mug, _, _ = _env()
    env.cabinet_item_layout = "single_upper_second_row"
    env.cabinet_items = [mug]
    env.cabinet_item_types = ["mug"]
    env.cabinet_grid = True
    env.cabinet_grid_layout = "trapezoid_4_3_2_1"
    encoder = CabinetStateEncoder(env)
    state = encoder.encode()
    assert state.shape == (49,)
    np.testing.assert_allclose(state[44:], [1, 0, .1, .2, .3])
    contract = encoder.contract()
    assert contract["dimension"] == 49
    assert contract["object_names"] == ["cabinet_item_16"]
    assert contract["object_types"] == ["mug"]
    assert contract["cabinet_grid"]["initial_cells"] == {"cabinet_item_16": 4}
    assert not any("cabinet_item_17" in field["name"] for field in contract["fields"])
    latched = CabinetSkillStateEncoder(env, latched_gripper=True)
    assert latched.encode().shape == (50,)
    assert latched.contract()["dimension"] == 50


def test_projection_contract_transforms_root_to_cabinet():
    env, _, _, controller = _env()
    rotation = Rotation.from_euler('z', -.3).as_matrix()
    position = np.array([1., 2., 3.])
    env.cabinet = SimpleNamespace(pose=_pose(position, rotation))
    env.exploration_region_bounds_local = np.array([[-.2, -.25, .04], [.1, .03, .30]])
    controller.config = SimpleNamespace(pos_lower=-.1, pos_upper=.1)
    encoder = CabinetStateEncoder(env)
    contract = encoder.action_projection_contract()
    point = np.array([.1, .2, .3])
    local = np.asarray(contract['root_to_cabinet_rotation']) @ point + contract['root_to_cabinet_translation']
    expected = rotation.T @ (encoder._root_rotation @ point + encoder._root_position - position)
    np.testing.assert_allclose(local, expected)
    np.testing.assert_allclose(contract['lower'], env.exploration_region_bounds_local[0])
    assert contract['pos_upper'] == [.1, .1, .1]


def test_static_features_do_not_depend_on_object_orientation():
    env, mug, _, _ = _env()
    encoder = CabinetStateEncoder(env)
    before = encoder.encode()[56:]
    mug.pose = _pose(mug.pose.p, Rotation.from_euler("xyz", [0.7, -0.4, 1.2]).as_matrix())
    after = encoder.encode()[56:]
    np.testing.assert_array_equal(after, before)


def test_contract_records_order_dimensions_and_collision_metadata():
    env, _, _, _ = _env()
    contract = CabinetStateEncoder(env).contract()
    assert contract["dimension"] == 66
    assert contract["agent_dim"] == AGENT_DIM
    assert contract["object_dyn_dim"] == OBJECT_DYN_DIM
    assert contract["object_stat_dim"] == OBJECT_STAT_DIM
    assert contract["object_names"] == list(OBJECT_NAMES)
    assert contract["dt_seconds"] == 0.05
    assert contract["collision_meshes"]["cabinet_item_16"]["extents_m"] == [0.1, 0.2, 0.3]
    assert [field["name"] for field in contract["fields"][:3]] == [
        "robot_qpos", "robot_qvel_dt", "tcp_root_position"
    ]


def test_invalid_control_frequency_and_nonfinite_values_fail_clearly():
    env, _, _, _ = _env()
    env.control_freq = 10
    with pytest.raises(ValueError, match="control_freq=20"):
        CabinetStateEncoder(env)

    env.control_freq = 20
    with pytest.raises(ValueError, match="dt must match"):
        CabinetStateEncoder(env, dt=.1)
    encoder = CabinetStateEncoder(env)
    env.agent.robot.qpos = np.full(9, np.nan)
    with pytest.raises(ValueError, match="robot qpos.*non-finite"):
        encoder.encode()


def test_scene_rebind_refreshes_handles_in_place_and_preserves_command():
    env, *_ = _env()
    env.door_joint = None
    encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
    encoder.gripper_command = -1.
    old_controller = encoder.controller
    rebuilt, mug, _, controller = _env()
    env.agent = rebuilt.agent
    env.cabinet_items = rebuilt.cabinet_items
    identity = id(encoder)
    encoder.rebind_after_reset()
    assert id(encoder) == identity
    assert encoder.controller is controller and encoder.controller is not old_controller
    assert encoder.objects[0][1] is mug
    assert encoder.encode()[30] == -1.
    assert encoder.contract()['agent_dim'] == 31


@pytest.mark.parametrize("latched", [False, True])
@pytest.mark.parametrize("single_cup", [False, True])
def test_doorless_state_omits_door_fields_and_preserves_remaining_state(latched, single_cup):
    env, *_ = _env()
    if single_cup:
        env.cabinet_item_layout = "single_upper_second_row"
    legacy = CabinetSkillStateEncoder(env, latched_gripper=latched)
    before, contract = legacy.encode(), legacy.contract()
    env.door_joint = None
    doorless = CabinetSkillStateEncoder(env, latched_gripper=latched)
    after = doorless.encode()
    assert len(after) == len(before) - 2
    np.testing.assert_array_equal(after, np.delete(before, [30, 31]))
    assert doorless.contract()["agent_dim"] == 30 + int(latched)
    assert doorless.contract()["dimension"] == len(after)
    assert sum(np.prod(field["shape"]) for field in doorless.contract()["fields"]) == len(after)
    assert not any(field["name"].startswith("door_") for field in doorless.contract()["fields"])
    if latched:
        assert after[30] == 1.
        assert doorless.contract()["fields"][6]["name"] == "last_gripper_command"
    assert doorless.contract()["cabinet_door"] is False
    assert doorless.contract() != contract
    assert "cabinet_door" not in contract
