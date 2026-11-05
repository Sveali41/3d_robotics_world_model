import io
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.utils import RoboCasaPointWorldSampleBuilder


@pytest.fixture
def pointworld_wrapper_class(monkeypatch):
    module_names_before = set(sys.modules)
    original_sys_path = list(sys.path)
    observations_module = SimpleNamespace(
        sensor_data_to_pointcloud=lambda *args, **kwargs: None
    )
    for module_name in (
        "mani_skill",
        "mani_skill.envs",
        "mani_skill.envs.utils",
        "mani_skill.envs.utils.observations",
    ):
        monkeypatch.setitem(sys.modules, module_name, SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "mani_skill.envs.utils.observations.observations",
        observations_module,
    )
    wrapper_module = importlib.import_module(
        "exploration.pointworld.official.integration.pointworld_wrapper"
    )
    yield wrapper_module.PointWorldWrapper

    for module_name in set(sys.modules) - module_names_before:
        if module_name == "exploration.pointworld.official.integration" or module_name.startswith(
            "exploration.pointworld.official.integration."
        ) or module_name == "integration" or module_name.startswith("integration."):
            sys.modules.pop(module_name, None)
    sys.path[:] = original_sys_path


def _blob_array(payload, name):
    return np.load(io.BytesIO(payload[name]))


class _Robot:
    root_link = SimpleNamespace(pose=None)

    def __init__(self):
        class _Pose:
            def to_transformation_matrix(self):
                return torch.eye(4).unsqueeze(0)

            def inv(self):
                return self

        self.root_link.pose = _Pose()

    def get_links(self):
        return [SimpleNamespace(per_scene_id=[7])]

    def get_active_joints(self):
        return [SimpleNamespace(name="joint")]


class _Env:
    def __init__(self):
        self.agent = SimpleNamespace(robot=_Robot())
        self._sensors = {name: object() for name in ("cam_a", "cam_b")}


class _Record:
    def __init__(self, observations):
        self.observations = observations
        self.low_actions = np.zeros((1, 1))
        self.full_controller_actions = np.zeros((1, 1))
        self.qpos = np.zeros((1, 1))
        self.qvel = np.zeros((1, 1))
        self.ee_pose = np.zeros((1, 7))
        self.gripper_open = np.zeros((1, 1))
        self.base_pose = np.zeros((1, 7))
        self.body_qpos = np.zeros((1, 1))
        self.contact_events = []
        self.object_poses = []

    def phase_view(self, horizon):
        return {
            "observations": self.observations,
            "object_poses": None,
            "schema_version": "test",
            "raps_action_normalized": np.zeros((1,)),
            "primitive_type": "test",
            "episode_id": 0,
            "primitive_id": 0,
            "parameters_physical": {},
            "primitive_duration": 1,
            "sample_indices": np.arange(1),
            "primitive_phase": np.zeros(1),
            "temporal_valid_mask": np.ones(1, dtype=bool),
            "low_action_valid_mask": np.ones(1, dtype=bool),
            "low_actions": np.zeros((1, 1)),
            "status": "ok",
            "termination_reason": "done",
            "terminated": True,
            "truncated": False,
            "metadata": {},
            "outcome_metrics": {},
        }

    def validate(self):
        pass


class _RobotAdapter:
    @staticmethod
    def gripper_open(observations):
        return torch.zeros((len(observations), 1))


class _Wrapper:
    robot_adapter = _RobotAdapter()

    def __init__(self):
        self.calls = []

    def process_observations(
        self, observations, link_ids, sensors, **kwargs
    ):
        observation = observations[0]
        camera_name = next(iter(observation["sensor_data"]))
        pointcloud = observation.get("pointcloud")
        if pointcloud is None:
            marker_value = observation["sensor_data"][camera_name]["rgb"].reshape(1, 1, 1)
            pointcloud = {"rgb": marker_value}
        self.calls.append((
            camera_name,
            list(sensors),
            list(observation["sensor_data"]),
            list(observation["sensor_param"]),
            pointcloud["rgb"].clone(),
        ))
        marker = float(pointcloud["rgb"][0, 0, 0])
        camera_fields = {
            "cam0_initial_rgb": torch.full((1, 1, 1, 3), marker),
            "cam0_initial_depth": torch.full((1, 1, 1), marker),
            "cam0_intrinsic": torch.full((1, 3, 3), marker),
            "cam0_extrinsic": torch.full((1, 4, 4), marker),
        }
        result = {
            "scene_flows": torch.full((1, 1, 1, 3), marker),
            "scene_features": torch.full((1, 1, 1, 9), marker / 255),
            "scene_exists": torch.ones((1, 1, 1), dtype=torch.bool),
            "scene_visibility": torch.ones((1, 1, 1), dtype=torch.bool),
            "scene_depth_valid_mask": torch.ones((1, 1, 1), dtype=torch.bool),
            "scene_pose_valid": torch.ones((1, 1, 1), dtype=torch.bool),
            "scene_dynamic_mask": torch.zeros((1, 1), dtype=torch.bool),
            "scene_actor_local": torch.zeros((1, 1, 3)),
            "scene_actor_local_valid": torch.zeros((1, 1), dtype=torch.bool),
            "scene_segmentation": torch.zeros((1, 1), dtype=torch.int32),
            "robot_flows": torch.zeros((1, 1, 1, 3)),
            "robot_features": torch.zeros((1, 1, 1, 9)),
        }
        result.update(camera_fields)
        return result, {}


def _observation(cam_a, cam_b):
    return {
        "sensor_data": {
            "cam_a": {"rgb": torch.full((1, 1), cam_a)},
            "cam_b": {"rgb": torch.full((1, 1), cam_b)},
        },
        "sensor_param": {"cam_a": {"p": cam_a}, "cam_b": {"p": cam_b}},
        "pointcloud": {"rgb": torch.tensor([[[cam_a], [cam_b]]])},
        "pointcloud_cameras": {
            "cam_a": {"rgb": torch.tensor([[[cam_a]]])},
            "cam_b": {"rgb": torch.tensor([[[cam_b]]])},
        },
    }


def test_builder_processes_camera_isolated_pointclouds_and_writes_matching_fields():
    wrapper = _Wrapper()
    builder = RoboCasaPointWorldSampleBuilder(wrapper, _Env())
    payload = builder(_Record([_observation(11, 22)]), horizon=1)

    assert [call[0] for call in wrapper.calls] == ["cam_a", "cam_b"]
    assert [call[1] for call in wrapper.calls] == [["cam_a"], ["cam_b"]]
    assert [call[2] for call in wrapper.calls] == [["cam_a"], ["cam_b"]]
    assert [call[3] for call in wrapper.calls] == [["cam_a"], ["cam_b"]]
    assert [int(call[4][0, 0, 0]) for call in wrapper.calls] == [11, 22]
    for camera_name, marker in (("cam_a", 11), ("cam_b", 22)):
        prefix = f"camera_{camera_name}"
        for suffix in (
            "scene_flows.npy",
            "initial_rgb.npy",
            "initial_depth.npy",
            "intrinsic.npy",
            "extrinsic.npy",
        ):
            assert f"{prefix}_{suffix}" in payload
        assert _blob_array(payload, f"{prefix}_scene_flows.npy")[0, 0, 0] == marker
        assert _blob_array(payload, f"{prefix}_initial_rgb.npy").flat[0] == marker
    assert "scene_flows.npy" in payload


def test_builder_derives_cameras_from_sensor_data_when_camera_pointclouds_absent():
    wrapper = _Wrapper()
    builder = RoboCasaPointWorldSampleBuilder(wrapper, _Env())
    observation = _observation(11, 22)
    observation.pop("pointcloud_cameras")
    payload = builder(_Record([observation]), horizon=1)

    assert [call[0] for call in wrapper.calls] == ["cam_a", "cam_b"]
    assert [call[1] for call in wrapper.calls] == [["cam_a"], ["cam_b"]]
    assert [call[2] for call in wrapper.calls] == [["cam_a"], ["cam_b"]]
    assert [int(call[4][0, 0, 0]) for call in wrapper.calls] == [11, 22]
    # Sensor-only paths let PointWorld derive the selected camera's cloud from
    # the selected sensor block and do not pass the merged cloud through.
    assert "camera_cam_a_initial_rgb.npy" in payload
    assert "camera_cam_b_initial_rgb.npy" in payload


def test_wrapper_model_free_preprocessing_initialization(
    monkeypatch, pointworld_wrapper_class
):
    class _Sampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def presample(self, count, seed):
            self.count = count

    monkeypatch.setitem(
        sys.modules, "robot_sampler", SimpleNamespace(RobotSampler=_Sampler)
    )
    wrapper = pointworld_wrapper_class(
        model_path=None,
        device="cpu",
        robot_type="fetch",
        data_domain="robocasa",
        filter_scene_workspace=False,
    )

    assert wrapper.args.domains == ["robocasa"]
    assert wrapper.model is None
    assert wrapper.filter_scene_workspace is False
    with pytest.raises(RuntimeError, match="preprocessing only"):
        wrapper.predict([], [], {})


def test_wrapper_model_free_requires_domain(monkeypatch, pointworld_wrapper_class):
    monkeypatch.setitem(
        sys.modules,
        "robot_sampler",
        SimpleNamespace(RobotSampler=lambda **kwargs: SimpleNamespace(presample=lambda *a, **k: None)),
    )
    with pytest.raises(ValueError, match="data_domain is required"):
        pointworld_wrapper_class(model_path=None, device="cpu")
