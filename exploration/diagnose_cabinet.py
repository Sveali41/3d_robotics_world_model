"""Staged, budgeted diagnostics for the fixed two-object cabinet.

Run ``python -m exploration.diagnose_cabinet --help``. Training is never
started by audit, sample, evaluate, or report.
"""

import argparse
import copy
import csv
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
import torch

from domain.robocasa.primitives import PrimitiveAction, RAPS_ACTION_SCHEMA
from exploration.contact_reward import ContactExplorationRewardConfig, ContactExplorationRewardWrapper
from exploration.diagnostic_metrics import compare_arms, summarize_records
from exploration.ppo_adapter import RAPSActorCritic, Rollout, _observation_tensor, _ppo_update
from exploration.scripted_grasp import SCRIPTED_GRASP_SEQUENCES

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}
TARGETS = ("cabinet_item_16", "cabinet_item_17")
LIMITS = {"audit": 5000, "sample": 5000, "train": 50000, "evaluate": 5000}


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def source_evidence(output=None):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT).decode()
    patch = git("diff", "--binary", "--submodule=diff", "HEAD")
    sources = {}
    for name in git("ls-files", "--others", "--exclude-standard").splitlines():
        path = ROOT / name
        if output is not None and path.resolve().is_relative_to(output.resolve()):
            continue
        if path.suffix in {".py", ".yaml", ".md"} and path.is_file():
            sources[name] = path.read_text()
    evidence = {"commit": git("rev-parse", "HEAD").strip(), "diff": patch, "untracked_sources": sources}
    encoded = json.dumps(evidence, sort_keys=True).encode()
    evidence["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return evidence


def prepare_output(output, cfg):
    output.mkdir(parents=True, exist_ok=True)
    evidence = source_evidence(output)
    config = OmegaConf.to_container(cfg, resolve=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_fingerprint"] != evidence["fingerprint"] or manifest["config"] != config:
            raise ValueError("Source or configuration changed since this run. Use a new --output directory.")
        return
    versions = {}
    for name in ("torch", "numpy", "gymnasium", "omegaconf", "sapien", "mani_skill"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed as a distribution"
    write_json(manifest_path, {
        "source_fingerprint": evidence["fingerprint"], "commit": evidence["commit"],
        "config": config, "python": platform.python_version(), "dependencies": versions,
        "seed": 0, "layout_randomized": False, "bowl_diameter_m": 0.072,
        "arms": ARMS, "limits": LIMITS, "heuristic_baseline_only": True,
    })
    (output / "working_tree.patch").write_text(evidence["diff"])
    write_json(output / "untracked_sources.json", evidence["untracked_sources"])
    OmegaConf.save(cfg, output / "config.yaml")


def reward_configs(cfg):
    baseline = ContactExplorationRewardConfig(**OmegaConf.to_container(cfg.reward, resolve=True))
    corrected = replace(baseline, grasp_tcp_offset=0.0)
    from exploration.contact_reward import interaction_reward_config
    interaction = interaction_reward_config(baseline)
    return {"R0": baseline, "offset_only": corrected, "R1": interaction}


def make_env(cfg, arm):
    from main import build_env
    wrapped = build_env(cfg)
    raps = wrapped.env
    raps.project_goals_to_region = ARMS[arm][0]
    if ARMS[arm][1]:
        return ContactExplorationRewardWrapper(raps, reward_configs(cfg)["R1"])
    return wrapped


def snapshot(env):
    context = env.env._context()
    raw = env.unwrapped
    return jsonable({
        "ee_position": context.ee_position,
        "objects": {name: {"position_root": obj.position_root, "position_world": obj.position_world}
                    for name, obj in context.objects.items()},
        "contact_snapshot": context.contact_snapshot(),
        "inside_region": raw.is_ee_inside_exploration_region(),
        "failed_object_ids": list(raw.failed_object_ids()),
    })


class ReplayEnv(gym.Env):
    """Replay reward inputs only, without simulating or selecting another action."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (20,), dtype=np.float32)
        self.cabinet_items = []
        self.current = None
        self.next_row = None

    def _context(self):
        return SimpleNamespace(
            ee_position=np.asarray(self.current["ee_position"], dtype=np.float32),
            objects={name: SimpleNamespace(position_root=np.asarray(obj["position_root"], dtype=np.float32))
                     for name, obj in self.current["objects"].items()},
            contact_snapshot=lambda: self.current["contact_snapshot"],
        )

    def is_ee_inside_exploration_region(self):
        return self.current["inside_region"]

    def failed_object_ids(self):
        return self.current["failed_object_ids"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        row = self.next_row
        self.current = row["after"]
        record = SimpleNamespace(
            duration_low_steps=row["duration_low_steps"], contact_events=row["contact_events"],
            outcome_metrics=row["outcome_metrics"],
        )
        info = {"primitive_record": record, "primitive_timeout": row["status"] == "timeout"}
        return np.zeros(1, dtype=np.float32), 0.0, row["terminated"], row["truncated"], info


def replay_rewards(records, cfg):
    wrappers = {name: ContactExplorationRewardWrapper(ReplayEnv(), config)
                for name, config in reward_configs(cfg).items()}
    episode = None
    cumulative = {}
    for row in records:
        new_episode = row["episode_id"] != episode
        if new_episode:
            cumulative = {name: 0.0 for name in wrappers}
        result = {}
        for name, wrapper in wrappers.items():
            wrapper.env.current = row["before"]
            if new_episode:
                wrapper.reset(seed=0)
            wrapper.env.next_row = row
            *_, info = wrapper.step(None)
            result[name] = info["exploration_reward"]
            cumulative[name] += result[name]["total"]
        row["counterfactual_rewards"] = result
        row["counterfactual_cumulative_returns"] = dict(cumulative)
        episode = row["episode_id"]
    return records


def bounded_step(env, action, remaining):
    """Cap the existing episode limit, including a primitive's final real step."""
    if remaining <= 0:
        raise ValueError("No remaining low-level step budget")
    raps = env.env
    original_limit = raps.max_low_level_steps_per_episode
    raps.max_low_level_steps_per_episode = min(original_limit, raps.low_level_steps + remaining)
    try:
        before = snapshot(env)
        observation, reward, terminated, truncated, info = env.step(action)
        record = info["primitive_record"]
        if record.duration_low_steps > remaining:
            raise AssertionError("Low-level budget exceeded")
        row = jsonable({
            "episode_id": record.episode_id, "primitive_id": record.primitive_id,
            "primitive_type": record.primitive_type, "action": action,
            "parameters_physical": record.parameters_physical,
            "duration_low_steps": record.duration_low_steps, "status": record.status,
            "final_stage": record.final_stage, "terminated": bool(terminated), "truncated": bool(truncated),
            "run_budget_exhausted": record.duration_low_steps == remaining,
            "before": before, "after": snapshot(env), "outcome_metrics": record.outcome_metrics,
            "contact_events": record.contact_events, "reward": info["exploration_reward"],
        })
        return observation, float(reward), bool(terminated), bool(truncated), row
    finally:
        raps.max_low_level_steps_per_episode = original_limit


def geometry_audit(env):
    """Use the loaded Panda URDF and actual link poses, not an assumed TCP offset."""
    agent = env.unwrapped.agent
    tree = ET.parse(agent.urdf_path)
    centers = []
    for name in ("panda_leftfinger", "panda_rightfinger"):
        link = tree.find(f"./link[@name='{name}']")
        pads = []
        for collision in link.findall("collision"):
            box = collision.find("geometry/box")
            if box is not None and np.allclose(np.fromstring(box.attrib["size"], sep=" "), [0.0175, 0.0152, 0.0185]):
                pads.append(np.fromstring(collision.find("origin").attrib["xyz"], sep=" "))
        if len(pads) != 1:
            raise ValueError(f"Cannot uniquely identify pad collision on {name} in {agent.urdf_path}")
        pose = agent.robot.links_map[name].pose.to_transformation_matrix()[0].detach().cpu().numpy()
        centers.append((pose @ np.append(pads[0], 1.0))[:3])
    tcp = agent.tcp.pose.p[0].detach().cpu().numpy()
    displacement = np.mean(centers, axis=0) - tcp
    return {
        "urdf": str(agent.urdf_path), "tcp_world": tcp.tolist(),
        "mean_pad_center_minus_tcp_world_m": displacement.tolist(),
        "pad_tcp_distance_m": float(np.linalg.norm(displacement)),
        "zero_offset_supported": bool(np.linalg.norm(displacement) < 0.005),
        "note": "Pad-center geometry check; does not imply object-center alignment is an optimal grasp pose.",
    }


def policy_hash(policy):
    digest = hashlib.sha256()
    for name, tensor in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def gae_check():
    rollout = Rollout()
    values = ([0.2, 0.3], [0.4, 0.5])
    for i in range(2):
        rollout.append_batch(
            torch.zeros(2, 1), torch.zeros(2, 20), torch.zeros(2), torch.tensor(values[i]),
            torch.tensor(([0.4, 0.5], [0.6, 0.7])[i]), np.array(([1.0, 2.0], [3.0, 4.0])[i]),
            np.array(([False, True], [True, False])[i]), np.array(([True, False], [False, False])[i]),
        )
    actual = rollout.tensors(torch.device("cpu"), 0.8, 0.95)["advantages"]
    expected = torch.tensor([1.12, 1.7, 2.6, 4.06])
    return bool(torch.allclose(actual, expected, atol=1e-6))


def ppo_positive_control(cfg, observation_dim):
    torch.manual_seed(0)
    np.random.seed(0)
    policy = RAPSActorCritic(observation_dim, 20, int(cfg.ppo.hidden_size), float(cfg.ppo.action_std))
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(cfg.ppo.learning_rate))
    device = torch.device("cpu")
    observation = torch.zeros(256, observation_dim)
    target_index = RAPS_ACTION_SCHEMA.primitive_names.index("top_grasp")
    slots = RAPS_ACTION_SCHEMA.primitive_name_to_action_idx["top_grasp"]
    mask_ok = all(
        set(torch.where(policy.parameter_mask[i] != 0)[0].tolist())
        == set(RAPS_ACTION_SCHEMA.primitive_name_to_action_idx[name])
        for i, name in enumerate(RAPS_ACTION_SCHEMA.primitive_names)
    )
    target = torch.tensor([0.3, 0.2, 0.1])
    history = []
    max_ratio_error = 0.0
    for update in range(50):
        with torch.no_grad():
            action, log_prob, value = policy.act(observation)
            repeated, _ = policy._log_prob_entropy(observation, action)
            ratio_error = float(((repeated - log_prob).exp() - 1.0).abs().max())
            max_ratio_error = max(max_ratio_error, ratio_error)
            selected = action[:, :10].argmax(-1) == target_index
            accuracy = 1.0 - ((action[:, 10:][:, slots] - target) ** 2).mean(-1)
            reward = (selected * accuracy).numpy()
        rollout = Rollout()
        rollout.append_batch(observation, action, log_prob, value, torch.zeros(256),
                             reward, np.ones(256, dtype=bool), np.zeros(256, dtype=bool))
        metrics = _ppo_update(policy, optimizer, rollout, cfg.ppo, device)
        with torch.no_grad():
            categorical, continuous = policy.distribution(observation[:1])
            probability = float(categorical.probs[0, target_index])
            mean = torch.tanh(continuous.base_dist.loc[0, slots])
            error = (mean - target).abs().tolist()
        history.append({"update": update + 1, "target_probability": probability,
                        "parameter_absolute_errors": error, **metrics})
    passed = probability > 0.8 and max(error) <= 0.15 and mask_ok and gae_check() and max_ratio_error < 1e-5
    return {"passed": passed, "mask_correct": mask_ok, "gae_boundaries_correct": gae_check(),
            "max_preupdate_ratio_error": max_ratio_error, "history": history,
            "target_normalized_parameters": target.tolist(), "synthetic_transitions": 50 * 256}


def write_records(directory, records, cfg, expected_reward):
    replay_rewards(records, cfg)
    for row in records:
        replay = row["counterfactual_rewards"][expected_reward]["total"]
        if not np.isclose(replay, row["reward"]["total"], rtol=1e-6, atol=1e-6):
            raise AssertionError("Offline reward replay differs from the executed reward")
        row["episode_return"] = row["counterfactual_cumulative_returns"][expected_reward]
    with (directory / "primitives.jsonl").open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = summarize_records(records)
    write_json(directory / "metrics.json", summary)
    with (directory / "metrics.csv").open("x", newline="") as stream:
        scalar = {key: value for key, value in summary.items() if not isinstance(value, (dict, list))}
        writer = csv.DictWriter(stream, fieldnames=list(scalar))
        writer.writeheader()
        writer.writerow(scalar)
    with (directory / "coverage.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["coverage_curve"][0]))
        writer.writeheader()
        writer.writerows(summary["coverage_curve"])
    return summary


def audit_cases():
    cases = []
    for target, original in SCRIPTED_GRASP_SEQUENCES.items():
        for parameter, delta in [(None, 0.0)] + [
            (name, delta) for name in ("x_forward", "y_offset", "z_offset") for delta in (-0.005, 0.005)
        ]:
            sequence = copy.deepcopy(original)
            for name, parameters in sequence:
                if name == "top_grasp" and parameter is not None:
                    parameters[parameter] += delta
            label = "center" if parameter is None else f"{parameter}_{delta:+.3f}"
            cases.append((target, label, sequence))
        missed = copy.deepcopy(original)
        for name, parameters in missed:
            if name == "top_grasp":
                parameters["y_offset"] += 0.03
        cases.append((target, "offset_30mm", missed))
        pushing = []
        for name, parameters in original:
            if name == "top_grasp":
                pushing.extend([
                    ("close_gripper", {}),
                    ("move_delta_ee_pose", {"dx": 0.0, "dy": parameters["y_offset"], "dz": parameters["z_offset"]}),
                    ("move_forward", {"x_dist": parameters["x_forward"]}),
                ])
            elif name != "lift":
                pushing.append((name, copy.deepcopy(parameters)))
        cases.append((target, "closed_approach", pushing))
        cases.append((target, "no_lift", [item for item in copy.deepcopy(original) if item[0] != "lift"]))
    # Verify both central controls before spending budget on perturbations.
    return sorted(cases, key=lambda item: item[1] != "center")


def run_audit(cfg, directory):
    records, attempts = [], []
    used = 0
    env = make_env(cfg, "A")
    try:
        observation, _ = env.reset(seed=0)
        observation_dim = int(_observation_tensor(observation, torch.device("cpu")).numel())
        geometry = geometry_audit(env)
        for target, label, sequence in audit_cases():
            env.reset(seed=0)
            start = len(records)
            for name, parameters in sequence:
                if used >= LIMITS["audit"]:
                    break
                _, _, terminated, truncated, row = bounded_step(
                    env, PrimitiveAction(name, parameters).to_vector(), LIMITS["audit"] - used)
                row.update({"target": target, "case": label})
                records.append(row)
                used += row["duration_low_steps"]
                if terminated or truncated:
                    break
            attempt = records[start:]
            lifted = any(target in row["outcome_metrics"]["lifted_object_ids"] for row in attempt)
            completed = len(attempt) == len(sequence) and not any(
                row["terminated"] or row["truncated"] for row in attempt
            )
            attempts.append({"target": target, "case": label, "lifted": lifted,
                             "sequence_completed": completed,
                             "low_level_steps": sum(row["duration_low_steps"] for row in attempt)})
            print(f"audit target={target} case={label} lifted={lifted} steps={used}", flush=True)
            if label == "center" and not lifted:
                break
            if used >= LIMITS["audit"]:
                break
    finally:
        env.close()
    write_records(directory, records, cfg, "R0")
    for attempt in attempts:
        matching = [row for row in records if row["target"] == attempt["target"] and row["case"] == attempt["case"]]
        attempt["counterfactual_returns"] = matching[-1]["counterfactual_cumulative_returns"] if matching else {}
    positive = ppo_positive_control(cfg, observation_dim)
    write_json(directory / "ppo_positive_control.json", positive)
    write_json(directory / "geometry.json", geometry)
    write_json(directory / "attempts.json", attempts)
    warnings = []
    for previous, current in zip(records, records[1:]):
        if previous["episode_id"] != current["episode_id"]:
            continue
        raw = current["reward"]["raw"]
        if (raw["new_object_contact_count"] or raw["moved_object_count"]) and current["reward"]["total"] < previous["reward"]["total"]:
            warnings.append({"episode_id": current["episode_id"], "primitive_id": current["primitive_id"],
                             "previous_total": previous["reward"]["total"], "current_total": current["reward"]["total"],
                             "note": "Inspect reward components and duration; this alone is not a reward bug."})
    centers_pass = all(any(a["target"] == target and a["case"] == "center" and a["lifted"] for a in attempts) for target in TARGETS)
    complete = len(attempts) == 20 and all(a["sequence_completed"] for a in attempts)
    result = {"passed": centers_pass and geometry["zero_offset_supported"] and positive["passed"] and complete,
              "centers_passed": centers_pass, "all_cases_completed": complete,
              "geometry_passed": geometry["zero_offset_supported"], "ppo_positive_control_passed": positive["passed"],
              "low_level_steps": used, "reward_order_warnings": warnings}
    write_json(directory / "result.json", result)
    return result


def require_pass(output, stage):
    path = output / stage / "result.json"
    if not path.exists() or not json.loads(path.read_text()).get("passed"):
        raise ValueError(f"{stage} has not passed in {output}; inspect its result before proceeding")


def run_policy(cfg, directory, arm, stage, budget, checkpoint=None, deterministic=False):
    torch.manual_seed(0)
    np.random.seed(0)
    env = make_env(cfg, arm)
    records, updates = [], []
    used = episode = zero_steps = 0
    next_progress = 1000
    device = torch.device("cpu")
    write_json(directory / "effective_config.json", {
        "base_config": OmegaConf.to_container(cfg, resolve=True),
        "project_goals_to_region": ARMS[arm][0],
        "reward": asdict(reward_configs(cfg)["R1" if ARMS[arm][1] else "R0"]),
        "low_level_step_budget": budget, "policy_sampling": "deterministic" if deterministic else "stochastic", "seed": 0,
    })
    try:
        observation, _ = env.reset(seed=0)
        obs = _observation_tensor(observation, device).unsqueeze(0)
        torch.manual_seed(0)
        policy = RAPSActorCritic(obs.shape[-1], 20, int(cfg.ppo.hidden_size), float(cfg.ppo.action_std))
        initial_hash = policy_hash(policy)
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            if saved["arm"] != arm or saved["source_fingerprint"] != json.loads((directory.parents[1] / "manifest.json").read_text())["source_fingerprint"]:
                raise ValueError("Checkpoint arm or source fingerprint does not match")
            policy.load_state_dict(saved["model"])
        optimizer = torch.optim.Adam(policy.parameters(), lr=float(cfg.ppo.learning_rate))
        rollout = Rollout()
        while used < budget:
            with torch.no_grad():
                if deterministic:
                    action = policy.deterministic_action(obs)
                    log_prob, _ = policy._log_prob_entropy(obs, action)
                    value = policy.value(obs)
                else:
                    action, log_prob, value = policy.act(obs)
            observation, reward, terminated, truncated, row = bounded_step(env, action[0].numpy(), budget - used)
            records.append(row)
            duration = row["duration_low_steps"]
            used += duration
            if used >= next_progress:
                print(f"{stage} arm={arm} steps={used}/{budget} primitives={len(records)}", flush=True)
                next_progress = (used // 1000 + 1) * 1000
            zero_steps = zero_steps + 1 if duration == 0 else 0
            if zero_steps >= int(cfg.max_primitives_per_episode):
                raise RuntimeError("An entire episode of consecutive zero-step actions made no physics progress")
            next_obs = _observation_tensor(observation, device).unsqueeze(0)
            if stage == "train":
                with torch.no_grad():
                    next_value = policy.value(next_obs)
                rollout.append_batch(obs, action, log_prob, value, next_value,
                                     np.array([reward]), np.array([terminated]), np.array([truncated]))
                if len(rollout) >= 256 or used == budget:
                    if len(rollout) >= 2:
                        metrics = _ppo_update(policy, optimizer, rollout, cfg.ppo, device)
                        with torch.no_grad():
                            categorical, continuous = policy.distribution(torch.cat(rollout.observations))
                        updates.append({
                            "update": len(updates) + 1, "low_level_steps": used, "transitions": len(rollout),
                            "primitive_probabilities": categorical.probs.mean(0).tolist(),
                            "parameter_means": torch.tanh(continuous.base_dist.loc).mean(0).tolist(),
                            "parameter_base_std": continuous.base_dist.scale.mean(0).tolist(), **metrics,
                        })
                        print(f"train arm={arm} update={len(updates)} steps={used}/{budget} kl={metrics['approx_kl']:.5f}", flush=True)
                    rollout = Rollout()
            if (terminated or truncated) and used < budget:
                episode += 1
                observation, _ = env.reset(seed=episode)
                next_obs = _observation_tensor(observation, device).unsqueeze(0)
            obs = next_obs
        manifest = json.loads((directory.parents[1] / "manifest.json").read_text())
        if stage == "train":
            torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "arm": arm,
                        "low_level_steps": used, "updates": len(updates), "initial_policy_hash": initial_hash,
                        "source_fingerprint": manifest["source_fingerprint"]}, directory / "policy.pt")
        summary = write_records(directory, records, cfg, "R1" if ARMS[arm][1] else "R0")
        write_json(directory / "updates.json", updates)
        write_json(directory / "result.json", {"passed": used == budget, "arm": arm, "stage": stage,
                                              "budget": budget, "initial_policy_hash": initial_hash,
                                              "low_level_steps": used})
        return summary
    finally:
        env.close()


def run_report(output):
    summaries = {}
    problems = []
    for arm in ARMS:
        path = output / "evaluate" / arm / "metrics.json"
        result_path = path.with_name("result.json")
        if not path.exists() or not result_path.exists():
            problems.append(f"{arm}: missing completed evaluation")
            continue
        status = json.loads(result_path.read_text())
        summary = json.loads(path.read_text())
        if (not status.get("passed") or status.get("budget") != LIMITS["evaluate"]
                or status.get("low_level_steps") != LIMITS["evaluate"]
                or summary["low_level_steps"] != LIMITS["evaluate"]
                or set(summary["per_object"]) != set(TARGETS)):
            problems.append(f"{arm}: evaluation budget or target set is invalid")
            continue
        summaries[arm] = summary
    result = compare_arms(summaries)
    if set(summaries) != set(ARMS):
        result.update(status="incomplete", candidates=[], reason="All four complete evaluations are required", problems=problems)
    result["arms"] = {arm: {key: value for key, value in summary.items() if key != "coverage_curve"}
                      for arm, summary in summaries.items()}
    write_json(output / "report.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "sample", "train", "evaluate", "report"), required=True)
    parser.add_argument("--arm", choices=("all", *ARMS), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/cabinet_diagnostics_seed0")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exploration.yaml")
    parser.add_argument("--deterministic", action="store_true", help="Auxiliary evaluation only; excluded from screening")
    args = parser.parse_args(argv)
    if args.deterministic and args.stage != "evaluate":
        parser.error("--deterministic is only supported by --stage evaluate")
    torch.set_num_threads(1)
    cfg = OmegaConf.load(args.config)
    cfg.seed = 0
    cfg.headless = True
    cfg.render_mode = None
    cfg.ppo.num_envs = 1
    cfg.ppo.rollout_steps = 256
    cfg.ppo.record_dense_trajectory = False
    cfg.ppo.contact_check_interval = 2
    prepare_output(args.output, cfg)
    if args.stage == "report":
        run_report(args.output)
        return 0
    if args.stage != "audit":
        require_pass(args.output, "audit")
    if args.stage in {"train", "evaluate"}:
        require_pass(args.output, "sample")
    directory = args.output / ("evaluate_deterministic" if args.deterministic else args.stage)
    if args.stage == "audit":
        directory.mkdir(exist_ok=False)
        result = run_audit(cfg, directory)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["passed"] else 1
    arms = ("A", "C") if args.stage == "sample" else tuple(ARMS)
    if args.arm != "all":
        if args.arm not in arms:
            raise ValueError("Sample compares P0=A and P1=C; use --arm A, C, or all")
        arms = (args.arm,)
    checkpoints = {}
    for arm in arms:
        if (directory / arm).exists():
            raise FileExistsError(f"{directory / arm} already exists; use a new output or another arm")
        checkpoint = None
        if args.stage == "evaluate":
            result_path = args.output / "train" / arm / "result.json"
            if not result_path.exists() or not json.loads(result_path.read_text()).get("passed"):
                raise ValueError(f"Training arm {arm} has not completed its fixed budget")
            checkpoint = result_path.parent / "policy.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
        checkpoints[arm] = checkpoint
    for arm in arms:
        destination = directory / arm
        destination.mkdir(parents=True, exist_ok=False)
        run_policy(cfg, destination, arm, args.stage, LIMITS[args.stage], checkpoints[arm], args.deterministic)
    if args.stage == "sample":
        passed = all((directory / arm / "result.json").exists()
                     and json.loads((directory / arm / "result.json").read_text())["passed"] for arm in ("A", "C"))
        if passed:
            hashes = {json.loads((directory / arm / "result.json").read_text())["initial_policy_hash"] for arm in ("A", "C")}
            passed = len(hashes) == 1
        write_json(directory / "result.json", {"passed": passed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
