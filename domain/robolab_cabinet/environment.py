"""RoboLab cabinet variants; import only after Isaac Lab's AppLauncher."""
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation

from robolab.core.environments.base import TerminationsCfg
from robolab.core.environments.config import (
    generate_scene_env_cfg,
    generate_task_env_cfg,
    register_generated_env,
)
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task
from robolab.robots.droid import BinaryJointPositionZeroToOneActionCfg
from robolab.robots.franka import FrankaCfg
from robolab.robots.franka_definitions import FrankaJointPositionActionCfg
from robolab.variations.lighting import SphereLightCfg

from .scene import export_scene


def register_cabinet(config: dict, variant: str, output_dir: Path):
    """Register a reward-free exploration scene with exact subset geometry."""
    physics_hz, control_hz = config["sim_freq"], config["control_freq"]
    if physics_hz <= 0 or control_hz <= 0 or physics_hz % control_hz:
        raise ValueError("sim_freq must be a positive integer multiple of control_freq")
    scene_path = output_dir / "cabinet.usda"
    spec = export_scene(scene_path, item_names=config["variants"][variant]["items"],
                        include_physics_scene=False)
    names = [item["name"] for item in spec["items"]]
    env_name = f"Cabinet{variant.title()}"

    @configclass
    class CabinetRobotCfg(FrankaCfg):
        # The cabinet is floor-mounted; suppress the stock frozen table fixture
        # before Lab 3 traverses nested configuration values.
        table_fixture = None

    CabinetRobotCfg.table_fixture = None

    @configclass
    class CabinetActions(FrankaJointPositionActionCfg):
        arm_action = FrankaJointPositionActionCfg().arm_action.replace(
            scale=1.0, use_default_offset=False, preserve_order=True)
        gripper_action = BinaryJointPositionZeroToOneActionCfg(
            asset_name="robot", joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_joint.*": 0.04},
            close_command_expr={"panda_finger_joint.*": 0.0})

    camera = config["camera"]
    eye, lookat = np.asarray(camera["eye"]), np.asarray(camera["lookat"])
    forward = (lookat - eye) / np.linalg.norm(lookat - eye)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # OpenGL camera axes: +X right, +Y up, -Z forward; authored quaternions are WXYZ.
    xyzw = Rotation.from_matrix(np.column_stack([right, up, -forward])).as_quat()
    rotation = tuple(xyzw[[3, 0, 1, 2]])
    aperture = 20.955
    focal_length = aperture / (2 * np.tan(camera["fov"] / 2))

    @configclass
    class CabinetCameraCfg:
        cabinet_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/cabinet_camera",
            width=camera["width"], height=camera["height"],
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(focal_length), horizontal_aperture=aperture,
                clipping_range=(0.01, 100.)),
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=rotation, convention="opengl"))

    @configclass
    class Proprioception(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    @configclass
    class Observations:
        proprio_obs = Proprioception()
        image_obs = generate_image_obs_from_cameras(CabinetCameraCfg)()

    task = type(env_name, (Task,), {
        "scene": import_scene(str(scene_path), [*names, "door"]),
        "contact_object_list": names,
        "terminations": TerminationsCfg,
        "instruction": f"Explore the {variant} cabinet; no task success criterion is defined.",
        "episode_length_s": config["episode_steps"] / control_hz,
        "attributes": ["cabinet", "exploration", variant],
        "subtasks": None,
    })
    scene_cfg = generate_scene_env_cfg(task, CabinetRobotCfg, CabinetCameraCfg, SphereLightCfg)
    grippers = {
        "left_finger": "{ENV_REGEX_NS}/robot/panda_leftfinger",
        "right_finger": "{ENV_REGEX_NS}/robot/panda_rightfinger",
        "gripper": ["left_finger", "right_finger"],
    }
    base_env_cfg = generate_task_env_cfg(
        task, scene_cfg, Observations(), CabinetActions(), grippers,
        dt=1 / physics_hz, render_interval=physics_hz // control_hz,
        decimation=physics_hz // control_hz, seed=config["seed"],
        eye=tuple(eye), lookat=tuple(lookat),
        ee_recorder_bodies=CabinetRobotCfg.ee_recorder_bodies)

    @configclass
    class CabinetEnvCfg(base_env_cfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.robot.spawn.usd_path = config["robot_usd"]
            self.scene.robot.init_state.pos = tuple(spec["robot_base_position"])
            self.scene.robot.init_state.joint_pos = {
                **{f"panda_joint{i + 1}": q for i, q in enumerate(spec["pregrasp_qpos"][:7])},
                "panda_finger_joint.*": 0.04,
            }

    register_generated_env(CabinetEnvCfg, env_name)
    return env_name, spec
