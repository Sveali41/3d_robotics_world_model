"""Visual callbacks around the existing scalar split7 collector."""

from pathlib import Path

import numpy as np

from exploration.raps.raps_gnn_actor import DEFAULTS, SkillCollector


def camera_rgb(env):
    rgb = env.unwrapped.get_obs()["sensor_data"]["base_camera"]["rgb"]
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    return rgb[0] if rgb.ndim == 4 else rgb


class VisualPolicy:
    def __init__(self, replay, planner, settings):
        self.settings = {**DEFAULTS, **settings}
        self.replay, self.planner = replay, planner
        self.audits = []

    def action(self, _diagnostic_state):
        # SkillCollector's state argument is deliberately excluded from planning.
        action, audit = self.planner.plan(self.replay.context())
        self.audits.append(audit)
        return action


class VisualSkillCollector(SkillCollector):
    def __init__(self, env, encoder, policy, directory, replay):
        super().__init__(env, encoder, policy, directory)
        self.visual_replay = replay

    def reset(self, seed):
        super().reset(seed)
        self.visual_replay.reset(self.episode_id, camera_rgb(self.env))

    def collect(self, count, policy, *, seed, stage):
        return super().collect(count, policy, seed=seed, stage=stage,
                               on_skill_end=lambda skill: self.visual_replay.append(skill, camera_rgb(self.env)))


def make_collector(config, output, replay, planner):
    from exploration.cee_us.cee_us_cabinet import make_env
    from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder

    env, _ = make_env(episode_steps=config["episode_steps"], centered_initial_tcp=True,
                      action_mode="ee_delta", obs_mode="rgb",
                      sensor_configs={"base_camera": {"width": 320, "height": 180}},
                      sim_backend="cpu", num_envs=1)
    try:
        env.reset(seed=config["seed"])
        encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
        policy = VisualPolicy(replay, planner, config["raps_actor"])
        collector = VisualSkillCollector(env, encoder, policy, Path(output) / "diagnostics", replay)
        return collector
    except BaseException:
        env.close()
        raise
