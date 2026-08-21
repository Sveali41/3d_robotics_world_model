"""Small shared functions used by collection, training, and exploration."""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import torch


def select_pointcloud_cameras(observation: dict, env, camera_names: list[str]) -> dict:
    """Keep selected camera blocks from a merged ManiSkill point cloud."""
    if not camera_names:
        return observation
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    sensors = getattr(raw_env, "_sensors", {})
    available = list(sensors.keys())
    unknown = [name for name in camera_names if name not in available]
    if unknown:
        raise ValueError(f"Unknown camera_names={unknown}; available cameras={available}")
    pointcloud = observation.get("pointcloud")
    if not pointcloud:
        raise KeyError("camera selection requires observation['pointcloud']")
    first = next(iter(pointcloud.values()))
    total_points = int(first.shape[1])
    selected_ranges = []
    camera_ranges = {}
    offset = 0
    for name, sensor in sensors.items():
        count = int(sensor.config.width) * int(sensor.config.height)
        if name in camera_names:
            selected_ranges.append((offset, offset + count))
            camera_ranges[name] = (offset, offset + count)
        offset += count
    if offset != total_points:
        raise ValueError(f"Point-cloud size {total_points} does not match sensor pixels {offset}")
    indices = np.concatenate(
        [np.arange(start, end, dtype=np.int64) for start, end in selected_ranges]
    )
    filtered = dict(observation)
    filtered["pointcloud"] = {}
    filtered["pointcloud_cameras"] = {}
    for key, value in pointcloud.items():
        if isinstance(value, torch.Tensor):
            index = torch.as_tensor(indices, device=value.device)
            filtered["pointcloud"][key] = value.index_select(1, index)
        else:
            filtered["pointcloud"][key] = np.take(value, indices, axis=1)
        for name, (start, end) in camera_ranges.items():
            camera_indices = np.arange(start, end, dtype=np.int64)
            if isinstance(value, torch.Tensor):
                camera_index = torch.as_tensor(camera_indices, device=value.device)
                camera_value = value.index_select(1, camera_index)
            else:
                camera_value = np.take(value, camera_indices, axis=1)
            filtered["pointcloud_cameras"].setdefault(name, {})[key] = camera_value
    return filtered


def numpy_bytes(value: Any) -> bytes:
    """Serialize an array-like value to a WebDataset-compatible ``.npy`` blob."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value))
    return buffer.getvalue()


def text_bytes(value: Any) -> bytes:
    return str(value).encode("utf-8")


def _json_default(value: Any):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def json_bytes(value: Any) -> bytes:
    """Serialize nested simulator metadata without losing array values."""
    return json.dumps(value, default=_json_default, sort_keys=True).encode("utf-8")


def tensor_numpy(value: Any) -> np.ndarray:
    """Convert a PointWorld tensor to CPU numpy without a batch dimension."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return np.asarray(value)


def primitive_record_payload(record, phase: dict[str, Any]) -> dict[str, bytes]:
    """Serialize the common high-level view and dense primitive sidecar."""
    record.validate()
    return {
        "sample_schema_version.txt": text_bytes(phase["schema_version"]),
        "raps_action.npy": numpy_bytes(phase["raps_action_normalized"]),
        "primitive_type.txt": text_bytes(phase["primitive_type"]),
        "episode_id.npy": numpy_bytes(phase["episode_id"]),
        "primitive_id.npy": numpy_bytes(phase["primitive_id"]),
        "parameters_physical.json": json_bytes(phase["parameters_physical"]),
        "primitive_duration.npy": numpy_bytes(phase["primitive_duration"]),
        "sample_indices.npy": numpy_bytes(phase["sample_indices"]),
        "primitive_phase.npy": numpy_bytes(phase["primitive_phase"]),
        "temporal_valid_mask.npy": numpy_bytes(phase["temporal_valid_mask"]),
        "low_action_valid_mask.npy": numpy_bytes(
            phase["low_action_valid_mask"]
        ),
        "low_actions.npy": numpy_bytes(phase["low_actions"]),
        "dense_low_actions.npy": numpy_bytes(record.low_actions),
        "dense_full_controller_actions.npy": numpy_bytes(
            record.full_controller_actions
        ),
        "dense_qpos.npy": numpy_bytes(record.qpos),
        "dense_qvel.npy": numpy_bytes(record.qvel),
        "dense_ee_pose.npy": numpy_bytes(record.ee_pose),
        "dense_gripper_open.npy": numpy_bytes(record.gripper_open),
        "dense_base_pose.npy": numpy_bytes(record.base_pose),
        "dense_body_qpos.npy": numpy_bytes(record.body_qpos),
        "dense_contact_events.json": json_bytes(record.contact_events),
        "dense_object_poses.json": json_bytes(record.object_poses),
        "status.txt": text_bytes(phase["status"]),
        "termination_reason.txt": text_bytes(phase["termination_reason"]),
        "terminated.npy": numpy_bytes(np.bool_(phase["terminated"])),
        "truncated.npy": numpy_bytes(np.bool_(phase["truncated"])),
        "record_metadata.json": json_bytes(phase["metadata"]),
        "outcome_metrics.json": json_bytes(phase["outcome_metrics"]),
    }


class RoboCasaPointWorldSampleBuilder:
    """Convert one recorded RoboCasa primitive into PointWorld WDS fields.

    The existing PointWorldWrapper owns RGB-D, segmentation, voxel, FK,
    feature, and mask conversion.  This adapter only supplies the primitive's
    T=11 raw observations and serializes the resulting tensors.
    """

    def __init__(self, pointworld_wrapper, env):
        self.pointworld_wrapper = pointworld_wrapper
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.robot_link_ids = [
            int(link.per_scene_id[0]) for link in self.env.agent.robot.get_links()
        ]
        self.robot_joint_names = [
            joint.name for joint in self.env.agent.robot.get_active_joints()
        ]

    def _base_transforms(self):
        root_link = getattr(self.env.agent.robot, "root_link", None)
        root = root_link if root_link is not None else self.env.agent.robot.root
        root_pose = root.pose
        base_pose = root_pose.to_transformation_matrix()[0].detach().clone()
        base_inv = root_pose.inv().to_transformation_matrix()[0].detach().clone()
        return base_pose, base_inv

    def __call__(self, record, horizon: int = 11) -> dict[str, bytes]:
        phase = record.phase_view(horizon=horizon)
        base_pose, base_inv = self._base_transforms()
        observations = phase["observations"]
        camera_names = list(observations[0].get("pointcloud_cameras", {}).keys())
        if not camera_names:
            camera_names = ["merged"]

        camera_data = {}
        for camera_name in camera_names:
            camera_observations = []
            for observation in observations:
                camera_observation = dict(observation)
                if camera_name != "merged":
                    camera_observation["pointcloud"] = observation["pointcloud_cameras"][camera_name]
                camera_observation.pop("pointcloud_cameras", None)
                camera_observations.append(camera_observation)
            camera_data[camera_name], _ = self.pointworld_wrapper.process_observations(
                camera_observations,
                self.robot_link_ids,
                self.env._sensors,
                base_pose_mat=base_pose,
                base_pose_inv_mat=base_inv,
                object_pose_seq=phase.get("object_poses"),
                visibility_camera_name=(
                    camera_name if camera_name != "merged" else None
                ),
            )

        # Keep one merged fallback for direct consumers, while also retaining
        # per-camera fields so PointWorld's official sample_cameras() can
        # choose and concatenate cameras later in the training pipeline.
        first_data = next(iter(camera_data.values()))
        merged = dict(first_data)
        if len(camera_data) > 1:
            merged["scene_flows"] = torch.cat(
                [data["scene_flows"] for data in camera_data.values()], dim=2
            )
            merged["scene_features"] = torch.cat(
                [data["scene_features"] for data in camera_data.values()], dim=2
            )
            merged["scene_exists"] = torch.cat(
                [data["scene_exists"] for data in camera_data.values()], dim=2
            )
            for field in (
                "scene_visibility",
                "scene_depth_valid_mask",
                "scene_pose_valid",
            ):
                merged[field] = torch.cat(
                    [data[field] for data in camera_data.values()], dim=2
                )
            for field in (
                "scene_dynamic_mask",
                "scene_actor_local",
                "scene_actor_local_valid",
                "scene_segmentation",
            ):
                merged[field] = torch.cat(
                    [data[field] for data in camera_data.values()], dim=1
                )
        data_dict = merged
        # Store PointWorld's canonical raw fields.  Feature vectors such as
        # velocity, acceleration, gripper state, and dist2robot are assembled
        # later by the official dataset_components.pipeline.gather_features.
        scene_features = tensor_numpy(data_dict["scene_features"])
        robot_features = tensor_numpy(data_dict["robot_features"])
        scene_exists = tensor_numpy(data_dict["scene_exists"]).astype(bool)
        scene_visibility = tensor_numpy(
            data_dict["scene_visibility"]
        ).astype(bool)
        scene_depth_valid = tensor_numpy(
            data_dict["scene_depth_valid_mask"]
        ).astype(bool)
        gripper_open = tensor_numpy(
            self.pointworld_wrapper.robot_adapter.gripper_open(phase["observations"])
        ).reshape(-1, 1)
        payload = primitive_record_payload(record, phase)
        payload.update({
            "scene_flows.npy": numpy_bytes(tensor_numpy(data_dict["scene_flows"])),
            "robot_flows.npy": numpy_bytes(tensor_numpy(data_dict["robot_flows"])),
            "right_gripper_open.npy": numpy_bytes(gripper_open),
        })
        # PointWorld's training transforms expect the canonical color/normal
        # and visibility fields as well as coordinates/features.  The wrapper
        # stores colors and normals inside the feature vectors, so recover the
        # canonical views here without recomputing the point cloud.
        payload.update({
            "scene_colors.npy": numpy_bytes(
                np.clip(scene_features[..., 3:6] * 255.0, 0, 255).astype(np.uint8)
            ),
            "scene_normals.npy": numpy_bytes(scene_features[..., 6:9].astype(np.float32)),
            "scene_exists.npy": numpy_bytes(scene_exists),
            "scene_visibility.npy": numpy_bytes(scene_visibility),
            "scene_depth_valid_mask.npy": numpy_bytes(scene_depth_valid),
            "scene_pose_valid.npy": numpy_bytes(
                tensor_numpy(data_dict["scene_pose_valid"]).astype(bool)
            ),
            "scene_dynamic_mask.npy": numpy_bytes(
                tensor_numpy(data_dict["scene_dynamic_mask"]).astype(bool)
            ),
            "scene_actor_local.npy": numpy_bytes(
                tensor_numpy(data_dict["scene_actor_local"]).astype(np.float32)
            ),
            "scene_actor_local_valid.npy": numpy_bytes(
                tensor_numpy(data_dict["scene_actor_local_valid"]).astype(bool)
            ),
            "scene_segmentation.npy": numpy_bytes(
                tensor_numpy(data_dict["scene_segmentation"]).astype(np.int32)
            ),
            "robot_colors.npy": numpy_bytes(
                np.clip(robot_features[..., 3:6] * 255.0, 0, 255).astype(np.uint8)
            ),
            "robot_normals.npy": numpy_bytes(robot_features[..., 6:9].astype(np.float32)),
        })
        for camera_name, camera_result in camera_data.items():
            if camera_name == "merged":
                continue
            camera_features = tensor_numpy(camera_result["scene_features"])
            camera_exists = tensor_numpy(camera_result["scene_exists"]).astype(bool)
            camera_visibility = tensor_numpy(
                camera_result["scene_visibility"]
            ).astype(bool)
            camera_depth_valid = tensor_numpy(
                camera_result["scene_depth_valid_mask"]
            ).astype(bool)
            prefix_name = camera_name.replace("fetch_", "")
            prefix = f"camera_{prefix_name}"
            payload.update({
                f"{prefix}_scene_flows.npy": numpy_bytes(
                    tensor_numpy(camera_result["scene_flows"])
                ),
                f"{prefix}_scene_colors.npy": numpy_bytes(
                    np.clip(camera_features[..., 3:6] * 255.0, 0, 255).astype(np.uint8)
                ),
                f"{prefix}_scene_normals.npy": numpy_bytes(
                    np.clip(camera_features[..., 6:9] * 127.0, -127, 127).astype(np.int8)
                ),
                f"{prefix}_scene_exists.npy": numpy_bytes(camera_exists),
                f"{prefix}_scene_visibility.npy": numpy_bytes(camera_visibility),
                f"{prefix}_scene_depth_valid_mask.npy": numpy_bytes(
                    camera_depth_valid
                ),
                f"{prefix}_scene_pose_valid.npy": numpy_bytes(
                    tensor_numpy(camera_result["scene_pose_valid"]).astype(bool)
                ),
                f"{prefix}_scene_segmentation.npy": numpy_bytes(
                    tensor_numpy(camera_result["scene_segmentation"]).astype(np.int32)
                ),
            })
        for key, value in data_dict.items():
            if key.endswith(("_initial_rgb", "_initial_depth", "_intrinsic", "_extrinsic")):
                payload[f"{key}.npy"] = numpy_bytes(tensor_numpy(value))
        return payload
