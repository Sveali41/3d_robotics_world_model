"""Run random RAPS contact exploration in the two-item wall cabinet."""

from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig


def build_env(cfg: DictConfig):
    import gymnasium as gym

    # Importing the class registers WallCabinetTwoItems-v1 with Gymnasium.
    try:
        from domain.wall_cabinet_env import WallCabinetTwoItemsEnv  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name in {"mani_skill", "sapien"}:
            raise RuntimeError(
                "Cabinet exploration requires the rlPractice Conda environment. "
                "Run: conda run -n rlPractice python main.py"
            ) from exc
        raise
    from domain.robocasa.primitive_wrapper import RAPSEnvWrapper
    from exploration.contact_reward import (
        ContactExplorationRewardConfig,
        ContactExplorationRewardWrapper,
    )
    from omegaconf import OmegaConf

    action_mode = str(getattr(cfg, "action_mode", "primitives"))
    if action_mode not in {"primitives", "ee_delta"}:
        raise ValueError("action_mode must be primitives or ee_delta")

    exploration_mode = str(getattr(cfg, "exploration_mode", "baseline"))
    if exploration_mode not in {"baseline", "D"}:
        raise ValueError("exploration_mode must be baseline or D")

    render_mode = cfg.render_mode or (None if cfg.headless else "human")
    base_env = gym.make(
        "WallCabinetTwoItems-v1",
        obs_mode="state",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose",
        render_mode=render_mode,
        sim_backend=cfg.sim_backend,
        sim_config={
            "sim_freq": int(cfg.sim.sim_freq),
            "control_freq": int(cfg.sim.control_freq),
        },
        visualize_exploration_region=not cfg.headless,
        viewer_camera_configs={"shader_pack": cfg.shader},
        centered_initial_tcp=getattr(cfg, "centered_initial_tcp", False),
    )
    raps_env = RAPSEnvWrapper(
        base_env,
        robot="panda",
        action_mode=action_mode,
        project_goals_to_region=(exploration_mode == "D"
                                or bool(getattr(cfg, "project_goals_to_region", False))),
        max_primitives_per_episode=cfg.max_primitives_per_episode,
        max_low_level_steps_per_episode=cfg.max_low_level_steps_per_episode,
        max_ee_steps_per_episode=int(getattr(cfg, "max_ee_steps_per_episode", 100)),
        render_during_primitive=not cfg.headless,
        record_dense_trajectory=cfg.ppo.record_dense_trajectory,
        contact_check_interval=cfg.ppo.contact_check_interval,
    )
    reward_cfg = ContactExplorationRewardConfig(
        **OmegaConf.to_container(cfg.reward, resolve=True)
    )
    if exploration_mode == "D":
        from exploration.contact_reward import interaction_reward_config
        reward_cfg = interaction_reward_config(reward_cfg)
    return ContactExplorationRewardWrapper(raps_env, reward_cfg)


@hydra.main(config_path="configs", config_name="exploration", version_base="1.3")
def main(cfg: DictConfig) -> int:
    if cfg.primitives < 1:
        raise ValueError("primitives must be positive")
    if cfg.max_primitives_per_episode < 1:
        raise ValueError("max_primitives_per_episode must be positive")
    if cfg.max_low_level_steps_per_episode < 1:
        raise ValueError("max_low_level_steps_per_episode must be positive")

    if cfg.policy == "ppo":
        from exploration.ppo_adapter import run_ppo

        return run_ppo(cfg, build_env)

    if cfg.policy == "evaluate":
        from exploration.ppo_adapter import run_evaluation

        return run_evaluation(cfg, build_env)

    if cfg.policy in {"scripted", "scripted_gif"} and str(cfg.action_mode) != "primitives":
        raise ValueError("scripted grasp sequences require action_mode=primitives")

    if cfg.policy == "scripted":
        from exploration.scripted_grasp import run_scripted_grasp

        status = run_scripted_grasp(cfg, build_env)
        if status:
            raise RuntimeError("scripted grasp acceptance failed")
        return 0

    if cfg.policy == "visualize":
        from exploration.policy_gif import run_policy_gif

        return run_policy_gif(cfg, build_env)

    if cfg.policy == "scripted_gif":
        from exploration.scripted_gif import run_scripted_gif

        return run_scripted_gif(cfg, build_env)

    if cfg.policy != "random":
        raise ValueError(
            f"Unknown policy={cfg.policy!r}; expected 'ppo', 'evaluate', "
            "'scripted', 'scripted_gif', 'visualize', or 'random'"
        )

    env = build_env(cfg)
    from domain.robocasa.primitives import UniformPrimitiveExplorer

    explorer = UniformPrimitiveExplorer(seed=cfg.seed, verbose=False)
    env.action_space.seed(int(cfg.seed))
    viewer = None
    total_reward = 0.0

    try:
        env.reset(seed=cfg.seed)
        if not cfg.headless:
            viewer = env.render()
            if viewer is not None:
                viewer.paused = cfg.pause

        for primitive_index in range(1, cfg.primitives + 1):
            if viewer is not None and viewer.closed:
                break

            action = (env.action_space.sample() if str(cfg.action_mode) == "ee_delta"
                      else explorer.sample_high_level_action())
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(np.asarray(reward).mean())
            report = info["exploration_reward"]
            print(
                f"primitive={primitive_index:04d} "
                f"type={info['primitive_type']} "
                f"reward={report['total']:+.3f} "
                f"new={report['new_object_ids']} "
                f"contacts={report['contacted_object_ids']}"
            )

            if terminated or truncated:
                env.reset(seed=cfg.seed + primitive_index)

        print(f"Exploration complete. Total reward: {total_reward:+.3f}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
