"""Render the deterministic horizontal cabinet grasp as a GIF."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import DictConfig

from domain.robocasa.primitives import PrimitiveAction
from .scripted_grasp import SCRIPTED_MUG_SEQUENCE


def _frame(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4:
        image = image[0]
    image = image[..., :3].astype(np.uint8, copy=False)
    if image.ndim != 3 or image.shape[-1] != 3 or image.max() == image.min():
        raise RuntimeError(f"Invalid or blank render frame: shape={image.shape}")
    return image


def run_scripted_gif(
    cfg: DictConfig, env_factory: Callable[[DictConfig], object]
) -> int:
    env = env_factory(cfg)
    output_path = Path(str(cfg.visualize.gif_path))
    frames = []
    try:
        env.reset(seed=int(cfg.seed))
        frames.append(_frame(env.render()))
        for index, (name, parameters) in enumerate(SCRIPTED_MUG_SEQUENCE, start=1):
            action = PrimitiveAction(name, parameters).to_vector()
            _, _, terminated, truncated, info = env.step(action)
            frames.append(_frame(env.render()))
            record = info.get("primitive_record")
            report = info.get("exploration_reward", {})
            print(
                f"gif primitive={index:02d} type={name} "
                f"status={getattr(record, 'status', 'unknown')} "
                f"lifted={report.get('lifted_object_ids', [])}"
            )
            if terminated or truncated:
                break
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(
            output_path,
            np.stack(frames),
            plugin="pillow",
            duration=1000 / float(cfg.visualize.fps),
            loop=0,
        )
        print(f"Saved scripted GIF: {output_path} ({len(frames)} frames)")
        return 0
    finally:
        env.close()
