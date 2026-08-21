"""Render a deterministic hybrid PPO rollout to an animated GIF."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import DictConfig

from .ppo_adapter import RAPSActorCritic, _observation_tensor


def _rgb_frame(frame) -> np.ndarray:
    """Convert ManiSkill's batched GPU image to one RGB image."""
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    image = np.asarray(frame)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"expected an RGB render frame, got shape={image.shape}")
    image = image[..., :3].astype(np.uint8, copy=False)
    if image.max() == image.min():
        raise RuntimeError("renderer returned a blank frame")
    return image


def run_policy_gif(cfg: DictConfig, env_factory: Callable[[DictConfig], object]) -> int:
    """Save one deterministic policy episode as a GIF."""
    if str(cfg.render_mode) != "rgb_array":
        raise ValueError("policy=visualize requires render_mode=rgb_array")

    env = env_factory(cfg)
    ppo_cfg = cfg.ppo
    output_path = Path(str(cfg.visualize.gif_path))
    checkpoint_path = Path(str(ppo_cfg.checkpoint_dir)) / str(ppo_cfg.checkpoint_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames: list[np.ndarray] = []
    try:
        torch.manual_seed(int(cfg.seed))
        observation, _ = env.reset(seed=int(cfg.seed))
        observation_tensor = _observation_tensor(observation, device)
        actor_critic = RAPSActorCritic(
            int(observation_tensor.shape[-1]),
            int(env.action_space.shape[0]),
            int(ppo_cfg.hidden_size),
            float(ppo_cfg.action_std),
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint["model"]
        log_std = model_state.get("log_std")
        if log_std is None or tuple(log_std.shape) != (10,):
            raise ValueError(
                f"{checkpoint_path} uses the legacy Gaussian policy and cannot "
                "be rendered by the hybrid RAPS policy"
            )
        actor_critic.load_state_dict(model_state)
        actor_critic.eval()

        total_rollouts = int(getattr(cfg.visualize, "num_rollouts", 1))
        for rollout_index in range(total_rollouts):
            if rollout_index > 0:
                observation, _ = env.reset(seed=int(cfg.seed) + rollout_index)
                observation_tensor = _observation_tensor(observation, device)
            frames.append(_rgb_frame(env.render()))
            grasped = set()
            lifted = set()
            for primitive_index in range(1, int(cfg.visualize.max_primitives) + 1):
                if bool(cfg.visualize.deterministic):
                    action = actor_critic.deterministic_action(
                        observation_tensor.unsqueeze(0)
                    )[0].cpu().numpy()
                else:
                    with torch.no_grad():
                        action = actor_critic.act(observation_tensor.unsqueeze(0))[0][0]
                    action = action.cpu().numpy()
                observation, _, terminated, truncated, info = env.step(action)
                observation_tensor = _observation_tensor(observation, device)
                frames.append(_rgb_frame(env.render()))
                report = info.get("exploration_reward", {})
                grasped.update(report.get("ever_grasped_object_ids", []))
                lifted.update(report.get("lifted_object_ids", []))
                print(
                    f"gif rollout={rollout_index + 1:02d} primitive={primitive_index:03d} "
                    f"type={info.get('primitive_type')} "
                    f"status={info.get('primitive_record').status if info.get('primitive_record') else 'unknown'} "
                    f"grasped={report.get('ever_grasped_object_ids', [])} "
                    f"lifted={report.get('lifted_object_ids', [])}"
                )
                if terminated or truncated:
                    break
            print(
                f"gif rollout={rollout_index + 1:02d} "
                f"contact_grasped={sorted(grasped)} "
                f"real_lifted={sorted(lifted)}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        changed_frames = sum(
            not np.array_equal(frame, frames[index - 1])
            for index, frame in enumerate(frames[1:], start=1)
        )
        iio.imwrite(
            output_path,
            np.stack(frames),
            plugin="pillow",
            duration=1000 / float(cfg.visualize.fps),
            loop=0,
        )
        print(
            f"Saved policy GIF: {output_path} ({len(frames)} source frames, "
            f"{changed_frames} changed)"
        )
        return 0
    finally:
        env.close()
