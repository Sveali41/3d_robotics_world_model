import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from common.utils import numpy_bytes
from exploration.pointworld.pointworld_cabinet_offline import (
    audit, episode_split, gnn_arrays, rigid_object_points, sha256,
)


def dataset(tmp_path):
    manifest = []
    for split, episode in (("train", 1), ("test", 5)):
        (tmp_path / split).mkdir()
        for index in range(2):
            key = f"ep{episode}_skill{index}"
            state = np.zeros(67, np.float32)
            target = state.copy()
            target[33] = .02 if split == "test" else .04
            payload = {
                "episode_id.npy": numpy_bytes(episode),
                "gnn_state.npy": numpy_bytes(state),
                "gnn_next_state.npy": numpy_bytes(target),
                "skill_action.npy": numpy_bytes(np.zeros(12, np.float32)),
                "scene_flows.npy": numpy_bytes(np.zeros((11, 3, 3), np.float32)),
                "robot_flows.npy": numpy_bytes(np.zeros((11, 5, 3), np.float32)),
                "camera_base_camera_initial_rgb.npy": numpy_bytes(np.zeros((180, 320, 3), np.uint8)),
            }
            path = tmp_path / split / f"{split}-{key}.tar"
            with tarfile.open(path, "w") as archive:
                for field, blob in payload.items():
                    item = tarfile.TarInfo(f"{key}.{field}")
                    item.size = len(blob)
                    archive.addfile(item, io.BytesIO(blob))
            manifest.append(dict(key=key, episode_id=episode, split=split,
                                 trainable=True, path=str(path.relative_to(tmp_path)),
                                 sha256=sha256(path)))
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    return manifest


def test_episode_split_and_train_test_isolation(tmp_path):
    dataset(tmp_path)
    result = audit(tmp_path)
    assert result["ready_for_training"]
    assert result["episode_disjoint"]
    assert result["splits"]["test"]["per_object_moving_skills"] == [2, 0]
    train_keys, train = gnn_arrays(tmp_path, "train")
    test_keys, test = gnn_arrays(tmp_path, "test")
    assert not set(train_keys) & set(test_keys)
    np.testing.assert_array_equal(train["next_states"][:, 33], np.full(2, .04, np.float32))
    np.testing.assert_array_equal(test["next_states"][:, 33], np.full(2, .02, np.float32))
    assert {episode_split(i, 5) for i in range(1, 11)} == {"train", "test"}
    with pytest.raises(ValueError):
        episode_split(0, 5)


def test_leakage_is_rejected(tmp_path):
    manifest = dataset(tmp_path)
    manifest[-1]["episode_id"] = 1
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    with pytest.raises(ValueError, match="leakage"):
        audit(tmp_path)


def test_archive_integrity_and_missing_pair_are_rejected(tmp_path):
    manifest = dataset(tmp_path)
    archive = tmp_path / manifest[0]["path"]
    with archive.open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="hash"):
        audit(tmp_path)
    archive.unlink()
    with pytest.raises(ValueError, match="manifest"):
        audit(tmp_path)


def test_rigid_object_pose_to_common_points():
    state = np.zeros(67)
    state[33:36] = [1, 2, 3]
    target = state.copy()
    target[33:36] += [0, 0, .1]
    target[38] = np.pi / 2  # XYZ Euler z rotation in radians.
    initial_points = np.array([[2, 2, 3], [1, 3, 3]])
    expected = np.array([[1, 3, 3.1], [0, 2, 3.1]])
    np.testing.assert_allclose(rigid_object_points(state, target, initial_points, 0), expected, atol=1e-10)
    np.testing.assert_allclose(rigid_object_points(state, state, initial_points, 0), initial_points)


def test_visual_reward_observation_preserves_sensors_and_lift_bits():
    from exploration.contact_reward import ContactExplorationRewardWrapper
    wrapper = SimpleNamespace(_tracked_object_ids=["cup", "bowl"], ever_lifted_object_ids={"bowl"})
    observation = {"sensor_data": {"base_camera": {"rgb": np.ones((1, 2, 2, 3))}}}
    result = ContactExplorationRewardWrapper._append_lifted_bits(wrapper, observation)
    assert result["sensor_data"] is observation["sensor_data"]
    np.testing.assert_array_equal(result["ever_lifted"], [0, 1])
    assert "ever_lifted" not in observation


def test_gnn_launcher_only_trains_on_training_episodes(tmp_path, monkeypatch):
    import torch
    import exploration.cee_us.cee_us_backend as backend_module
    from exploration.pointworld.pointworld_cabinet_offline import train_gnn
    data = tmp_path / "data"
    data.mkdir()
    dataset(data)
    captured = {}

    class Backend:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def train(self, replay, epochs):
            captured["replay"], captured["epochs"] = replay, epochs
            return {"loss": 1.0}

        def state_dict(self):
            return {"fake_model_for_launcher_test": True}

    monkeypatch.setattr(backend_module, "CEEUSBackend", Backend)
    args = SimpleNamespace(data=data, output=tmp_path / "model", seed=0, device="cpu", epochs=1)
    train_gnn(args)
    assert captured["init"] == dict(agent_dim=33, action_dim=12, seed=0, device="cpu")
    np.testing.assert_array_equal(captured["replay"]["next_states"][:, 33], np.full(2, .04, np.float32))
    payload = torch.load(args.output / "gnn.pt", weights_only=False)
    assert payload["train_keys"] == ["ep1_skill0", "ep1_skill1"]


def test_pointworld_launcher_scratch_and_train_only_normalization(tmp_path, monkeypatch):
    import yaml
    import exploration.pointworld.pointworld_cabinet_offline as module
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.jsonl").write_text("test manifest")
    pointworld = tmp_path / "PointWorld"
    weights = pointworld / "third_party/dinov3/checkpoints"
    weights.mkdir(parents=True)
    (weights / "dinov3_vitl16_pretrain_test.pth").write_bytes(b"test stub, not model weights")
    monkeypatch.setattr(module, "PW", pointworld)
    monkeypatch.setattr(module, "audit", lambda _: {"ready_for_training": True})
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: commands.append(command))
    args = SimpleNamespace(data=data, output=tmp_path / "model", seed=0, device="cpu",
                           epochs=1, batch_size=1, python="python")
    module.pointworld(args)
    config = yaml.safe_load((args.output / "pointworld.yaml").read_text())
    assert config["train_splits"] == "train"
    assert config["model_path"] is None
    assert config["raps_action_dim"] == 0
    assert config["norm_stats_path"] == str(args.output / "stats")
    assert commands[0][-1] == str(args.output / "stats/norm_stats.json")
    assert commands[1][1] == str(pointworld / "train.py")
