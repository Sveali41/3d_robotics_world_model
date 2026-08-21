"""Bootstrap collection for RAPS primitives in RoboCasa.

This module connects the execution wrapper to the dataset boundary:

    random 20D RAPS action
        -> RAPSEnvWrapper executes one primitive
        -> PrimitiveRecord stores H low-level transitions
        -> phase_view(T=11) creates a fixed-length training sample
        -> WebDataset writer stores the sample

The collector intentionally does not know how RoboCasa observations are
projected into PointWorld scene/robot fields.  That conversion is injected as
``sample_builder`` so the same collector can be used with RGB-D point clouds,
cached point clouds, or a later PointWorld-specific adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import pickle
from typing import Any, Callable

import numpy as np

from domain.robocasa.primitive_wrapper import RAPSEnvWrapper
from domain.robocasa.primitives import UniformPrimitiveExplorer
from common import primitive_record_payload, text_bytes


def _pickle_bytes(value: Any) -> bytes:
    # Pickle is used only for the raw observation archive.  PointWorld fields
    # should be emitted as .npy entries by a production sample_builder.
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


@dataclass(frozen=True)
class RoboCasaRAPSCollectionConfig:
    robot: str = "fetch"
    num_primitives: int = 100
    horizon: int = 11
    test_split_ratio: float = 0.1
    seed: int = 0
    max_primitives_per_episode: int = 40
    max_low_level_steps_per_episode: int = 1000
    save_gif: bool = False
    gif_dir: str | Path = "logs/collection"
    gif_fps: int = 5
    gif_max_points: int = 12000
    gif_point_size: float = 2.0


def select_test_primitive_indices(
    num_primitives: int, test_split_ratio: float, seed: int
) -> set[int]:
    """Return a deterministic primitive-level test split."""
    test_target = max(
        0, int(round(int(num_primitives) * float(test_split_ratio)))
    )
    if not test_target:
        return set()
    return set(
        np.random.default_rng(int(seed)).choice(
            int(num_primitives), size=test_target, replace=False
        ).tolist()
    )


def _as_numpy(value: Any) -> np.ndarray:
    """Convert a numpy/torch observation field to a CPU numpy array."""
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        pass
    return np.asarray(value)


def _save_pointcloud_gif(
    record,
    path: Path,
    *,
    fps: int,
    max_points: int,
    point_size: float,
) -> None:
    """Save the recorded T=11 point-cloud observations as a diagnostic GIF.

    RoboCasa's point-cloud observation is stored as ``xyzw``, ``rgb`` and
    ``segmentation`` tensors.  This function is intentionally diagnostic only:
    it visualizes the scene in 3D and does not alter the training sample.
    Imports are lazy so normal collection does not require matplotlib/imageio
    when GIF output is disabled.
    """
    try:
        import imageio.v2 as imageio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "GIF collection requires matplotlib and imageio. "
            "Install them in the active environment or set save_gif: false."
        ) from exc

    frames_data = []
    for observation in record.phase_view(horizon=11)["observations"]:
        pointcloud = observation.get("pointcloud", {}) if isinstance(observation, dict) else {}
        if "xyzw" not in pointcloud:
            continue
        xyz = _as_numpy(pointcloud["xyzw"])
        rgb = _as_numpy(pointcloud.get("rgb", np.ones((*xyz.shape[:-1], 3), dtype=np.float32)))
        if xyz.ndim == 3:
            xyz = xyz[0]
        if rgb.ndim == 3:
            rgb = rgb[0]
        xyz = xyz[..., :3]
        valid = np.isfinite(xyz).all(axis=-1)
        if xyz.shape[-1] >= 3:
            valid &= np.linalg.norm(xyz, axis=-1) > 1e-6
        xyz = xyz[valid]
        rgb = rgb[valid]
        if len(xyz) == 0:
            frames_data.append((xyz, rgb))
            continue
        if len(xyz) > max_points:
            indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
            xyz, rgb = xyz[indices], rgb[indices]
        rgb = rgb.astype(np.float32)
        if rgb.size and rgb.max() > 1.5:
            rgb /= 255.0
        frames_data.append((xyz, np.clip(rgb, 0.0, 1.0)))

    if not frames_data:
        raise RuntimeError("No point-cloud frames were found for GIF generation.")

    nonempty = [points for points, _ in frames_data if len(points)]
    all_points = np.concatenate(nonempty, axis=0) if nonempty else np.zeros((1, 3))
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float((upper - lower).max()) / 2.0, 0.05) * 1.1
    frames = []
    for frame_index, (xyz, rgb) in enumerate(frames_data):
        figure = plt.figure(figsize=(6, 6), dpi=120)
        axis = figure.add_subplot(111, projection="3d")
        if len(xyz):
            axis.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=rgb,
                s=max(float(point_size), 0.1),
                alpha=0.85,
                depthshade=False,
            )
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(f"{record.primitive_type} | frame {frame_index + 1}/{len(frames_data)}")
        figure.tight_layout()
        figure.canvas.draw()
        frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(figure)

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1.0 / max(int(fps), 1), loop=0)


def default_sample_builder(record, horizon: int = 11) -> dict[str, bytes]:
    """Build a transport-safe WDS sample from one ``PrimitiveRecord``.

    This default builder preserves the complete raw observation sequence and
    all RAPS metadata.  A PointWorld adapter can add fields such as
    ``scene_flows.npy`` and ``robot_flows.npy`` without changing collection.
    """

    phase = record.phase_view(horizon=horizon)
    payload = primitive_record_payload(record, phase)
    payload.update({
        "object_poses.pkl": _pickle_bytes(phase["object_poses"]),
        "observations.pkl": _pickle_bytes(phase["observations"]),
    })
    return payload


class RoboCasaRAPSBootstrapCollector:
    """Collect random RAPS primitives and write one WDS sample per primitive."""

    def __init__(
        self,
        env,
        *,
        train_tar: str | Path,
        test_tar: str | Path | None = None,
        config: RoboCasaRAPSCollectionConfig | None = None,
        sample_builder: Callable[[Any, int], dict[str, bytes]] | None = None,
        primitive_weights: dict[str, float] | None = None,
        observation_transform=None,
    ):
        self.config = config or RoboCasaRAPSCollectionConfig()
        if self.config.horizon < 2:
            raise ValueError("horizon must be at least 2")
        self.env = env if isinstance(env, RAPSEnvWrapper) else RAPSEnvWrapper(
            env,
            robot=self.config.robot,
            max_primitives_per_episode=self.config.max_primitives_per_episode,
            max_low_level_steps_per_episode=self.config.max_low_level_steps_per_episode,
            observation_transform=observation_transform,
        )
        self.train_tar = Path(train_tar)
        self.test_tar = Path(test_tar) if test_tar is not None else None
        self.sample_builder = sample_builder or default_sample_builder
        self.gif_dir = Path(self.config.gif_dir)
        self.explorer = UniformPrimitiveExplorer(
            seed=self.config.seed,
            weights=primitive_weights,
            verbose=False,
        )

    @staticmethod
    def _import_webdataset():
        try:
            import webdataset as wds
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "RoboCasa WDS collection requires webdataset; install the project dependencies."
            ) from exc
        return wds

    def collect(self) -> dict[str, int]:
        """Execute random primitives and write train/test shards."""

        if self.test_tar is None:
            self.test_tar = self.train_tar.with_name(self.train_tar.stem + "_test.tar")
        self.train_tar.parent.mkdir(parents=True, exist_ok=True)
        self.test_tar.parent.mkdir(parents=True, exist_ok=True)

        wds = self._import_webdataset()
        train_sink = wds.TarWriter(str(self.train_tar))
        test_sink = wds.TarWriter(str(self.test_tar))
        train_count = test_count = 0
        test_indices = select_test_primitive_indices(
            self.config.num_primitives,
            self.config.test_split_ratio,
            self.config.seed,
        )

        try:
            observation = self.env.reset(seed=self.config.seed)
            observation = observation[0] if isinstance(observation, tuple) else observation
            for primitive_index in range(self.config.num_primitives):
                # The random explorer samples a complete high-level action;
                # the wrapper, not this collector, expands it into low-level
                # controller steps.
                action = self.explorer.sample_high_level_action()
                observation, _, terminated, truncated, info = self.env.step(action)
                record = info.get("primitive_record")
                if record is None:
                    raise RuntimeError("RAPSEnvWrapper did not return primitive_record")

                payload = self.sample_builder(record, self.config.horizon)
                split = "test" if primitive_index in test_indices else "train"
                count = test_count if split == "test" else train_count
                if self.config.save_gif:
                    gif_path = self.gif_dir / (
                        f"episode_{record.episode_id:04d}_primitive_{record.primitive_id:04d}_"
                        f"{record.primitive_type}_{split}.gif"
                    )
                    _save_pointcloud_gif(
                        record,
                        gif_path,
                        fps=self.config.gif_fps,
                        max_points=self.config.gif_max_points,
                        point_size=self.config.gif_point_size,
                    )
                sample = {"__key__": f"primitive_{count:06d}", "split.txt": text_bytes(split)}
                sample.update(payload)
                (test_sink if split == "test" else train_sink).write(sample)
                if split == "test":
                    test_count += 1
                else:
                    train_count += 1

                if terminated or truncated:
                    observation = self.env.reset()
                    observation = observation[0] if isinstance(observation, tuple) else observation
        finally:
            train_sink.close()
            test_sink.close()
            if hasattr(self.env, "close"):
                self.env.close()

        metadata = {
            "sample_schema_version": "robocasa-raps-v1",
            "train_samples": train_count,
            "test_samples": test_count,
            "horizon": self.config.horizon,
            "action_dim": 20,
            "seed": self.config.seed,
        }
        with self.train_tar.parent.joinpath("raps_bootstrap_metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)
        return {"train": train_count, "test": test_count}
