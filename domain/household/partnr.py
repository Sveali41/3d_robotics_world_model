"""Use upstream PARTNR oracle skills without an LLM or a learned controller."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


PARTNR_REVISION = "ddfff19f4b6c098a31edea4d19e7b75db72433c2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARTNR_ROOT = PROJECT_ROOT / "third_party" / "partnr-planner"
SKILLS = ("Navigate", "Pick", "Place", "Open", "Close")


def skill_argument(skill, target, destination=None, relation="on"):
    if skill not in SKILLS:
        raise ValueError(f"Unknown skill {skill!r}; choose from {SKILLS}")
    if not target or "," in target:
        raise ValueError("Use an entity name from 'entities' as target")
    if skill == "Place":
        if not destination or "," in destination:
            raise ValueError("Place requires an object and destination entity")
        if relation not in {"on", "within"}:
            raise ValueError("PARTNR placement relation must be 'on' or 'within'")
        return f"{target},{relation},{destination},none,none"
    if destination is not None:
        raise ValueError(f"{skill} takes one target")
    return target


def check_installation(root, robot="spot"):
    """Check local prerequisites without loading Habitat or downloading assets."""
    root = Path(root).resolve()
    required = [
        "habitat_llm/conf/examples/skill_runner_default_config.yaml",
        "third_party/habitat-lab/habitat-lab/habitat/__init__.py",
        "data/datasets/partnr_episodes/v0_0/ci.json.gz",
        "data/hssd-partnr-ci/metadata",
        "data/hssd-partnr-ci/hssd-hab-partnr.scene_dataset_config.json",
        "data/objects_ovmm/train_val/ai2thorhab/configs/objects/pillow_9.object_config.json",
        "data/objects_ovmm/train_val/google_scanned/configs/CREATIVE_BLOCKS_35_MM.object_config.json",
        ("data/robots/hab_fetch/robots/hab_fetch.urdf" if robot == "fetch" else
         "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"),
        "data/humanoids/humanoid_data/female_0/female_0.urdf",
        "data/humanoids/humanoid_data/female_0/female_0_motion_data_smplx.pkl",
    ]
    missing = [str(root / name) for name in required if not (root / name).exists()]
    # Discoverability alone does not establish binary/driver compatibility.
    from importlib.util import find_spec

    modules = [name for name in ("habitat_sim", "habitat", "habitat_baselines", "habitat_llm")
               if find_spec(name) is None]
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    return {"partnr_root": str(root), "expected_revision": PARTNR_REVISION,
            "actual_revision": revision or None,
            "missing_paths": missing, "missing_modules": modules,
            "ready_for_runtime_attempt": not missing and not modules and revision == PARTNR_REVISION}


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    try:
        items = iter(value)
    except TypeError as error:
        raise TypeError(f"Cannot serialize household state value {type(value).__name__}") from error
    return [_json_value(item) for item in items]


def graph_snapshot(graph):
    """Keep original directed relation labels; omit nonserializable region handles."""
    nodes, edges = [], []
    for node, neighbors in sorted(graph.graph.items(), key=lambda pair: pair[0].name):
        properties = {key: value for key, value in node.properties.items() if key != "region"}
        held = properties.get("last_held_object")
        if held is not None:
            properties["last_held_object"] = {"name": held.name, "sim_handle": held.sim_handle}
        nodes.append({"name": node.name, "kind": type(node).__name__,
                      "sim_handle": node.sim_handle,
                      "properties": _json_value(properties)})
        for other, relation in sorted(neighbors.items(), key=lambda pair: pair[0].name):
            if relation is not None:
                edges.append({"source": node.name, "target": other.name, "relation": relation})
    return {"nodes": nodes, "edges": edges}


class HouseholdEnv:
    """One full CI household; only robot 0 is commanded, the human remains idle.

    PARTNR uses paths relative to its checkout. This environment temporarily owns
    the process working directory; close it before opening a second environment.
    States are privileged simulator/graph snapshots, not visual observations.
    """

    def __init__(self, partnr_root=DEFAULT_PARTNR_ROOT, output_dir=None,
                 episode_index=0, seed=0, save_video=False, max_skill_steps=1500,
                 frame_callback=None, robot="spot"):
        if robot not in {"spot", "fetch"}:
            raise ValueError("robot must be spot or fetch")
        self.robot = robot
        if max_skill_steps < 1 or episode_index < 0:
            raise ValueError("max_skill_steps must be positive and episode_index nonnegative")
        self.root = Path(partnr_root).resolve()
        if output_dir is None:
            output_dir = PROJECT_ROOT / "outputs" / "household" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_dir = Path(output_dir).resolve()
        self.max_skill_steps = max_skill_steps
        self.save_video = save_video
        self.frame_callback = frame_callback
        self.interface = None
        self._old_cwd = Path.cwd()
        self._done = False
        self._decision = 0
        self._episode_index = episode_index
        report = check_installation(self.root, robot)
        if not report["ready_for_runtime_attempt"]:
            raise RuntimeError("PARTNR prerequisites are missing. Run 'python main.py household --check' "
                               "and resolve the missing prerequisites listed below.\n" + json.dumps(report, indent=2))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any((self.output_dir / name).exists() for name in ("run.json", "transitions.jsonl")):
            raise FileExistsError(f"Choose a new output directory; a household run already exists at {self.output_dir}")
        os.chdir(self.root)
        try:
            from habitat_llm.agent import Agent
            from habitat_llm.agent.env import (
                EnvironmentInterface, register_actions, register_measures, register_sensors,
            )
            from habitat_llm.agent.env.dataset import CollaborationDatasetV0
            from habitat_llm.utils.core import (
                HabitatBaselinesConfigPlugin, HabitatConfigPlugin, get_config,
                register_hydra_plugin, setup_config,
            )

            overrides = [
                "device=cpu",
                "habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/ci.json.gz",
                "habitat.dataset.scenes_dir=data/hssd-partnr-ci",
                "+habitat.dataset.metadata.metadata_folder=data/hssd-partnr-ci/metadata",
                "habitat.environment.iterator_options.shuffle=False",
                "world_model.partial_obs=False",
                "evaluation.save_video=False", "trajectory.save=False",
                f"paths.results_dir={json.dumps(str(self.output_dir))}",
                f"evaluation.output_dir={json.dumps(str(self.output_dir))}",
            ]
            for field in ("video_dir", "tensorboard_dir", "eval_ckpt_path_dir",
                          "checkpoint_folder", "log_file"):
                overrides.append(f"habitat_baselines.{field}={json.dumps(str(self.output_dir / field))}")
            register_hydra_plugin(HabitatConfigPlugin)
            register_hydra_plugin(HabitatBaselinesConfigPlugin)
            config = get_config("examples/skill_runner_default_config.yaml", overrides)
            if robot == "fetch":
                from omegaconf import open_dict

                with open_dict(config):
                    robot_config = config.habitat.simulator.agents.agent_0
                    robot_config.articulated_agent_type = "FetchRobot"
                    robot_config.articulated_agent_urdf = "data/robots/hab_fetch/robots/hab_fetch.urdf"
                    config.evaluation.agents.agent_0.config.tools.motor_skills.oracle_nav.skill_config.skill._target_ = "domain.household.fetch.FetchOracleNavSkill"
            config = setup_config(config, seed)
            register_sensors(config)
            register_actions(config)
            register_measures(config)
            dataset = CollaborationDatasetV0(config.habitat.dataset)
            if episode_index >= len(dataset.episodes):
                raise ValueError(f"episode_index={episode_index}; dataset has {len(dataset.episodes)} episodes")
            self.interface = EnvironmentInterface(config, dataset=dataset)
            if robot == "fetch":
                # Use the configured 0.3 m circular navigation proxy for this wheeled base.
                self.interface.sim.agents_mgr[0].articulated_agent.params.navmesh_offsets = [[0.0, 0.0]]
            # Fully observed, single commanded agent: call the upstream skill API
            # directly, without importing PARTNR's LLM planner package.
            self.agent = Agent(0, config.evaluation.agents.agent_0.config, self.interface)
            self.reset()
            (self.output_dir / "run.json").write_text(json.dumps({
                "partnr_revision": PARTNR_REVISION, "seed": seed,
                "episode_index": episode_index, "skills": SKILLS,
                "robot": robot,
                "world_graph_robot_class": "SpotRobot (upstream generic agent_0 role)",
                "controller": "PARTNR oracle skills; kinematic simulator",
                "human": "present but not commanded", "privileged_state": True,
            }, indent=2))
        except BaseException:
            self.close()
            raise

    def reset(self):
        self.interface.env.habitat_env.episode_iterator.set_next_episode_by_index(self._episode_index)
        self.interface.reset_environment()
        self.agent.reset()
        self._done = False
        state = self.state()
        (self.output_dir / "initial_state.json").write_text(json.dumps(state, indent=2))
        return state

    def state(self):
        sim = self.interface.sim
        state = graph_snapshot(self.interface.full_world_graph)
        state.update({"episode_id": str(sim.ep_info.episode_id), "scene_id": sim.ep_info.scene_id,
                      "coordinates": "Habitat world frame, metres, Y up",
                      "simulator_objects": [], "agents": []})
        for manager in (sim.get_rigid_object_manager(), sim.get_articulated_object_manager()):
            for handle in sorted(manager.get_object_handles()):
                obj = manager.get_object_by_handle(handle)
                item = {"sim_handle": handle,
                        "transform_columns": [list(column) for column in obj.transformation]}
                if hasattr(obj, "joint_positions"):
                    item["joint_positions"] = _json_value(obj.joint_positions)
                state["simulator_objects"].append(item)
        for uid in (0, 1):
            agent = sim.agents_mgr[uid].articulated_agent
            state["agents"].append({"uid": uid, "base_position": list(agent.base_pos),
                                    "robot_class": type(agent).__name__,
                                    "base_rotation": float(agent.base_rot),
                                    "holding_object_id": sim.agents_mgr[uid].grasp_mgr.snap_idx})
        return state

    def execute_skill(self, skill, target, destination=None, relation="on"):
        argument = skill_argument(skill, target, destination, relation)
        if self._done:
            raise RuntimeError("Episode ended; call reset() before another skill")
        graph = self.interface.full_world_graph
        graph.get_node_from_name(target)
        if destination is not None:
            graph.get_node_from_name(destination)
        before = self.state()
        actions = {0: (skill, argument, None)}
        observations = self.interface.get_observations()
        response, steps, terminated = "", 0, False
        started = time.monotonic()
        video = None
        if self.save_video:
            from habitat_llm.examples.example_utils import DebugVideoUtil
            video = DebugVideoUtil(self.interface, str(self.output_dir))
        # Same dispatch and simulator-step order as upstream example_utils.execute_skill,
        # with an explicit timeout and episode boundary instead of an assertion.
        try:
            while steps < self.max_skill_steps:
                low_action, response = self.agent.process_high_level_action(
                    skill, argument, self.interface.filter_obs_space(observations, 0)
                )
                response = response.rstrip("\n")
                if low_action is not None:
                    obs, _, terminated, _ = self.interface.step({0: low_action})
                    steps += 1
                    observations = self.interface.parse_observations(obs)
                    self.interface.batch = observations
                    if self.frame_callback is not None:
                        self.frame_callback(observations, actions)
                    if video is not None:
                        video._store_for_video(observations, actions)
                elif not response:
                    raise RuntimeError("PARTNR returned neither a control action nor a skill response")
                if response or terminated:
                    break
        except Exception:
            self._done = True
            raise  # Infrastructure errors are not failed-skill training labels.
        finally:
            self.agent.reset()
        timed_out = not response and not terminated
        self._done = bool(terminated)
        transition = {"decision": self._decision, "skill": skill, "target": target,
                      "destination": destination, "relation": relation if skill == "Place" else None,
                      "response": response, "success": response == "Successful execution!",
                      "timed_out": timed_out, "episode_done": bool(terminated),
                      "control_steps": steps, "wall_seconds": time.monotonic() - started,
                      "state_before": before, "state_after": self.state()}
        with (self.output_dir / "transitions.jsonl").open("a") as handle:
            handle.write(json.dumps(transition) + "\n")
        if video is not None and video.frames:
            video._make_video(play=False, postfix=f"{self._decision:05d}_{skill}")
        self._decision += 1
        return transition

    def save_topdown(self):
        from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer

        visualizer = DebugVisualizer(self.interface.sim, str(self.output_dir))
        try:
            visualizer.create_dbv_agent(resolution=(1000, 1000))
            observation = visualizer.peek("stage")
            return observation.save(output_path=str(self.output_dir), prefix="household_topdown")
        finally:
            visualizer.remove_dbv_agent()

    def close(self):
        try:
            if self.interface is not None:
                self.interface.env.close()
                self.interface = None
        finally:
            os.chdir(self._old_cwd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
