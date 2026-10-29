"""State-based RAPS/GNN imagination actor; an explicit alternative to iCEM.

Reuses the vendored RAPS conditional actor, distributions, MLP and lambda-return.
This is a research adaptation, not the original image/RSSM Plan2Explore trainer.
Each model transition is one completed skill. All physical ticks are also kept.
"""

from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch

from domain.robocasa.primitives import RAPS_ACTION_SCHEMA, RAPS_COMPACT_ACTION_SCHEMA
from domain.robocasa.skill_execution import RAPSSkillExecution
from exploration.cee_us_backend import CEEUSBackend
from exploration.cee_us_cabinet import (
    ROOT, make_env, prepare_output, replay_arrays, restore_rng, rng_state,
)
from exploration.cee_us_metrics import interaction_summary, prediction_metrics, screen
from exploration.diagnose_cabinet import jsonable, snapshot, write_json


SCHEMA = "cabinet-raps-gnn-actor-v1"
RAPS_REFERENCE_COMMIT = "1c56b0fe367cc32d3ee548f407ac3a9af9e3ce2e"
DEFAULTS = dict(
    skill_set="legacy10", hidden_size=128, actor_layers=2, critic_layers=3, init_std=1.0,
    imagination_horizon=5, imagination_batch_size=64, updates_per_round=100,
    discount=0.8, lambda_return=0.95, actor_lr=8e-5, critic_lr=8e-5,
    adam_eps=1e-5, entropy_scale=1e-4, gradient_clip=100.0,
    target_update_period=100, hold_closed_during_motion=False,
    latched_gripper=False, stop_drop_on_support=False,
    grasp_stop_on_contact=True,
    controlled_placement=False,
    disagreement_weight=1.0,
)


def reference_components(device):
    """Import the checked-in reference, never an unrelated installed rlkit."""
    source = ROOT / "raps_official/rlkit"
    if not (source / "rlkit/torch/model_based/dreamer/actor_models.py").is_file():
        raise FileNotFoundError(f"Official RAPS source is required at {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import rlkit
    if Path(rlkit.__file__).resolve().parent != source / "rlkit":
        raise ImportError("Another rlkit is already imported; use a fresh process for RAPS")
    from rlkit.torch import pytorch_util as ptu
    from rlkit.torch.model_based.dreamer.actor_models import ConditionalActorModel, TanhBijector
    from rlkit.torch.model_based.dreamer.mlp import Mlp
    from rlkit.torch.model_based.dreamer.utils import lambda_return
    # PyTorch 2 requires transform metadata omitted by the torch-1.7 source.
    # Keep the reference tanh, inverse clamp and Jacobian calculations intact.
    TanhBijector.domain = torch.distributions.constraints.real
    TanhBijector.codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    ptu.set_gpu_mode(torch.device(device).type == "cuda", torch.device(device).index or 0)
    return ConditionalActorModel, Mlp, lambda_return


def skill_schema(settings):
    name = settings.get("skill_set", "legacy10")
    if name == "legacy10":
        return RAPS_ACTION_SCHEMA
    if name == "compact6":
        return RAPS_COMPACT_ACTION_SCHEMA
    raise ValueError(f"Unknown raps_actor.skill_set: {name!r}; use legacy10 or compact6")


def action_contract(settings):
    schema = skill_schema(settings)
    contract = dict(
        action_dim=schema.action_dim, primitive_names=list(schema.primitive_names),
        parameter_indices=schema.primitive_name_to_action_idx,
        parameters={spec.name: {p.name: p.scale for p in spec.parameters} for spec in schema.specs},
        model_transition="completed_skill_endpoint", velocity_dt_seconds=0.05,
        controller="existing_cabinet_side_grasp_on_ee_delta_reset_orientation",
        hold_closed_during_motion=settings["hold_closed_during_motion"],
        inactive_model_arguments="zero", policy_gradient="REINFORCE_all_reference_arguments",
        disagreement="sample_std_dynamic_state_sum_at_shared_state_action",
        discount="fixed_per_skill_no_learned_termination", state_propagation="ensemble_mean",
    )
    if settings["latched_gripper"] or settings["stop_drop_on_support"]:
        contract.update(latched_gripper=settings["latched_gripper"],
                        stop_drop_on_support=settings["stop_drop_on_support"],
                        ground_support_threshold_N=0.05,
                        controller_state_in_observation=settings["latched_gripper"],
                        command_memory_excluded_from_curiosity=True)
    if not settings["grasp_stop_on_contact"]:
        contract["grasp_stop_on_contact"] = False
    if settings["controlled_placement"]:
        contract["controlled_placement"] = dict(
            version=1, slowdown_distance_m=0.08, min_normalized_speed=0.025,
            drop_goal_tolerance_m=0.002, hold_position_tolerance_m=0.003,
            stable_speed_m_s=0.01, stable_contact_checks=4,
            settling_budget_steps=20, gripper_position_hold=True,
            support_and_gripper_orientation="actual_skill_pose",
            upward_retreat_actual_orientation_below_m=0.15)
    if settings.get("disagreement_weight", 1.0) != 1.0:
        contract["disagreement_weight"] = settings["disagreement_weight"]
    return contract


def canonical_action(actions, schema=RAPS_ACTION_SCHEMA):
    """The endpoint model sees only arguments that the real controller consumes."""
    if actions.shape[-1] != schema.action_dim:
        raise ValueError(f"RAPS actions must have {schema.action_dim} entries")
    mask = actions.new_zeros((schema.num_primitives, schema.num_parameters))
    for index, name in enumerate(schema.primitive_names):
        mask[index, schema.primitive_name_to_action_idx[name]] = 1
    indices = actions[..., :schema.num_primitives].argmax(-1)
    onehot = torch.nn.functional.one_hot(indices, schema.num_primitives).to(actions)
    return torch.cat((onehot, actions[..., schema.num_primitives:] * mask[indices]), -1)


class RAPSImaginationActor:
    def __init__(self, backend, settings=None):
        self.backend = backend
        self.settings = {**DEFAULTS, **(settings or {})}
        unknown = set(self.settings) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown raps_actor settings: {sorted(unknown)}")
        for name in ("hidden_size", "actor_layers", "critic_layers", "imagination_horizon",
                     "imagination_batch_size", "updates_per_round", "target_update_period"):
            if int(self.settings[name]) != self.settings[name] or self.settings[name] < 1:
                raise ValueError(f"raps_actor.{name} must be a positive integer")
        for name in ("actor_lr", "critic_lr", "adam_eps", "gradient_clip", "init_std"):
            if not np.isfinite(self.settings[name]) or self.settings[name] <= 0:
                raise ValueError(f"raps_actor.{name} must be finite and positive")
        for name in ("discount", "lambda_return"):
            if not np.isfinite(self.settings[name]) or not 0 <= self.settings[name] <= 1:
                raise ValueError(f"raps_actor.{name} must be in [0, 1]")
        if not np.isfinite(self.settings["entropy_scale"]) or self.settings["entropy_scale"] < 0:
            raise ValueError("raps_actor.entropy_scale must be finite and nonnegative")
        if not np.isfinite(self.settings["disagreement_weight"]) or self.settings["disagreement_weight"] < 0:
            raise ValueError("raps_actor.disagreement_weight must be finite and nonnegative")
        for name in ("hold_closed_during_motion", "latched_gripper", "stop_drop_on_support", "grasp_stop_on_contact", "controlled_placement"):
            if not isinstance(self.settings[name], bool):
                raise ValueError(f"{name} must be boolean")
        if self.settings["latched_gripper"] and self.settings["hold_closed_during_motion"]:
            raise ValueError("Select latched_gripper or legacy hold_closed_during_motion, not both")
        if self.settings["controlled_placement"] and not (
                self.settings["latched_gripper"] and self.settings["stop_drop_on_support"]):
            raise ValueError("controlled_placement requires latched_gripper and stop_drop_on_support")
        if backend.env.agent_dim != 32 + int(self.settings["latched_gripper"]):
            raise ValueError("GNN agent state must include the enabled gripper-command memory")
        self.schema = skill_schema(self.settings)
        if backend.action_dim != self.schema.action_dim:
            raise ValueError(f"Imagination actor requires a fresh {self.schema.action_dim}D skill GNN")
        Actor, Mlp, self.lambda_return = reference_components(backend.device)
        env = SimpleNamespace(num_primitives=self.schema.num_primitives,
                              primitive_name_to_action_idx=self.schema.primitive_name_to_action_idx)
        self.actor = Actor(
            hidden_size=self.settings["hidden_size"], obs_dim=backend.state_dim, env=env,
            num_layers=self.settings["actor_layers"], discrete_action_dim=env.num_primitives,
            continuous_action_dim=self.schema.num_parameters,
            init_std=self.settings["init_std"], min_std=0.1,
        ).to(backend.device)
        self.critic = Mlp([self.settings["hidden_size"]] * self.settings["critic_layers"],
                          input_size=backend.state_dim, output_size=1).to(backend.device)
        # deepcopy of the reference ScriptModule can produce non-leaf tensors
        # under modern PyTorch. Construct a separate module and copy weights.
        self.target_critic = Mlp([self.settings["hidden_size"]] * self.settings["critic_layers"],
                                 input_size=backend.state_dim, output_size=1).to(backend.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.settings["actor_lr"],
                                                eps=self.settings["adam_eps"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.settings["critic_lr"],
                                                 eps=self.settings["adam_eps"])
        self.updates = 0

    def features(self, states):
        model = self.backend.model
        agent, dyn, static = model._obs_preprocessing(states.unsqueeze(0))
        agent = model.input_normalizer_agent.normalize(agent.reshape(-1, model.agent_dim))
        dyn = model.input_normalizer_obj_dyn.normalize(dyn.reshape(-1, 12)).reshape(-1, 12 * self.backend.env.nObj)
        static = model.input_normalizer_obj_stat.normalize(static.reshape(-1, 5)).reshape(-1, 5 * self.backend.env.nObj)
        result = torch.cat((agent, dyn, static), -1).float()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite normalized actor state")
        return result

    @torch.no_grad()
    def action(self, state):
        state = self.backend._array(state, (self.backend.state_dim,), "actor state")
        features = self.features(torch.as_tensor(state, device=self.backend.device)[None])
        action = self.actor(features).rsample()[0]
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite RAPS actor action")
        return action.cpu().numpy()

    @torch.no_grad()
    def imagine(self, starts):
        """No env access or model gradients: REINFORCE uses detached imagined data."""
        states, actions, rewards, components, heuristics = [], [], [], [], []
        state = starts
        for _ in range(self.settings["imagination_horizon"]):
            states.append(state)
            raw = self.actor(self.features(state)).rsample()
            action = canonical_action(raw, self.schema)
            obs = state[None].expand(self.backend.ensemble_size, -1, -1)
            predicted = self.backend.model.predict(
                observations=obs, actions=action[None].expand(self.backend.ensemble_size, -1, -1),
                states=None)[0]
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("Nonfinite skill-model imagination; no random fallback")
            # Query all members at the SAME state/action: accumulated trajectory
            # divergence is not substituted for one-step epistemic disagreement.
            std = predicted[..., :self.backend.dynamic_dim].std(dim=0, unbiased=True)
            next_state = predicted.mean(0)
            object_start = self.backend.env.agent_dim
            # The extra command is controller memory, not physical novelty.
            intrinsic = std[:, :32].sum(-1) + std[:, object_start:].sum(-1)
            heuristic = torch.zeros_like(intrinsic)
            if self.backend.object_heuristic is not None:
                heuristic, _ = self.backend._object_heuristic_scores(state, next_state)
            actions.append(raw)
            reward = self.imagination_reward(state, action, next_state, starts,
                                             intrinsic, heuristic)
            if reward.shape != intrinsic.shape or not torch.isfinite(reward).all():
                raise FloatingPointError("Invalid imagination reward")
            rewards.append(reward.unsqueeze(-1))
            components.append(torch.stack((std[:, :18].sum(-1), std[:, 18:30].sum(-1),
                                           std[:, 30:32].sum(-1), std[:, object_start:].sum(-1)), -1))
            heuristics.append(heuristic)
            state = next_state
        states.append(state)
        return dict(states=torch.stack(states), actions=torch.stack(actions),
                    rewards=torch.stack(rewards), components=torch.stack(components),
                    heuristics=torch.stack(heuristics))

    def imagination_reward(self, state, action, next_state, starts, intrinsic, heuristic):
        return self.settings["disagreement_weight"] * intrinsic + heuristic

    def _step(self, module, optimizer, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite imagination actor/critic loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(module.parameters())
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("Nonfinite imagination gradient; optimizer update stopped")
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.settings["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        return float(norm)

    def train(self, replay, updates=None, *, start_groups=None):
        eligible = np.flatnonzero(~np.asarray(replay["terminated"], dtype=bool))
        if not len(eligible):
            raise ValueError("No nonterminal real replay states for imagination")
        source = torch.as_tensor(np.asarray(replay["states"])[eligible], device=self.backend.device)
        groups = None
        if start_groups is not None:
            labels = np.asarray(start_groups)
            if labels.shape != (len(replay["states"]),) or not np.isin(labels, [0, 1]).all():
                raise ValueError("start_groups must identify old=0/new=1 for every replay state")
            groups = [torch.as_tensor(np.flatnonzero(labels[eligible] == group), device=source.device)
                      for group in (0, 1)]
            if any(len(group) == 0 for group in groups) or self.settings["imagination_batch_size"] % 2:
                raise ValueError("Balanced imagination requires both nonterminal groups and an even batch size")
        metrics = []
        count = self.settings["updates_per_round"] if updates is None else int(updates)
        if count < 1:
            raise ValueError("Actor updates must be positive")
        for _ in range(count):
            if groups is None:
                indices = torch.randint(len(source), (self.settings["imagination_batch_size"],), device=source.device)
            else:
                half = self.settings["imagination_batch_size"] // 2
                indices = torch.cat([group[torch.randint(len(group), (half,), device=source.device)]
                                     for group in groups])
            trajectory = self.imagine(source[indices])
            state = trajectory["states"]
            horizon, batch = trajectory["actions"].shape[:2]
            features = self.features(state.reshape(-1, self.backend.state_dim)).detach()
            with torch.no_grad():
                values = self.target_critic(features).reshape(horizon + 1, batch, 1)
                discount = torch.full_like(trajectory["rewards"], self.settings["discount"])
                returns = self.lambda_return(trajectory["rewards"], values[:-1], discount,
                                             values[-1], lambda_=self.settings["lambda_return"])
                baseline = self.critic(features).reshape(horizon + 1, batch, 1)[:-1]
                advantages = (returns - baseline).reshape(-1)
                weights = torch.cumprod(torch.cat((torch.ones_like(discount[:1]), discount[:-1]), 0), 0)
            actor_features = features.reshape(horizon + 1, batch, -1)[:-1].reshape(horizon * batch, -1)
            distribution = self.actor(actor_features)
            log_prob = distribution.log_prob(trajectory["actions"].reshape(horizon * batch, -1).detach())
            entropy = distribution.entropy()
            actor_loss = ((-log_prob * advantages - self.settings["entropy_scale"] * entropy)
                          * weights.reshape(-1)).mean()
            actor_norm = self._step(self.actor, self.actor_optimizer, actor_loss)
            value = self.critic(actor_features).reshape(horizon, batch, 1)
            # Unit-variance Normal log likelihood, as in the reference value head.
            critic_loss = (-torch.distributions.Normal(value, 1).log_prob(returns.detach()) * weights).mean()
            critic_norm = self._step(self.critic, self.critic_optimizer, critic_loss)
            self.updates += 1
            if self.updates % self.settings["target_update_period"] == 0:
                self.target_critic.load_state_dict(self.critic.state_dict())
            metrics.append(dict(actor_loss=float(actor_loss.detach()), critic_loss=float(critic_loss.detach()),
                                actor_gradient_norm=actor_norm, critic_gradient_norm=critic_norm,
                                entropy=float(entropy.detach().mean()), imagined_reward=float(trajectory["rewards"].mean()),
                                disagreement=float(trajectory["components"].sum(-1).mean()),
                                heuristic=float(trajectory["heuristics"].mean()),
                                joint_disagreement=float(trajectory["components"][..., 0].mean()),
                                tcp_disagreement=float(trajectory["components"][..., 1].mean()),
                                door_disagreement=float(trajectory["components"][..., 2].mean()),
                                object_disagreement=float(trajectory["components"][..., 3].mean())))
        return {**{key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]},
                "updates": count, "total_updates": self.updates}

    def state_dict(self):
        return dict(settings=self.settings, updates=self.updates,
                    actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    target_critic=self.target_critic.state_dict(),
                    actor_optimizer=self.actor_optimizer.state_dict(),
                    critic_optimizer=self.critic_optimizer.state_dict())

    def load_state_dict(self, payload):
        if {**DEFAULTS, **payload["settings"]} != self.settings:
            raise ValueError("Checkpoint actor settings differ; use its saved config")
        for name in ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        self.updates = payload["updates"]


class SkillCollector:
    """Run existing primitives THROUGH the unchanged 4D execution path."""
    def __init__(self, env, encoder, actor, directory):
        self.env, self.encoder, self.actor = env, encoder, actor
        self.schema = skill_schema(actor.settings)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.total_steps = self.episode_id = self.skill_id = 0
        self.state = None
        self.execution = RAPSSkillExecution(
            latched_gripper=actor.settings.get("latched_gripper", False),
            stop_drop_on_support=actor.settings.get("stop_drop_on_support", False),
            grasp_stop_on_contact=actor.settings.get("grasp_stop_on_contact", True),
            controlled_placement=actor.settings.get("controlled_placement", False),
            hold_closed_during_motion=actor.settings["hold_closed_during_motion"])

    def reset(self, seed):
        self.env.reset(seed=int(seed))
        self.episode_id += 1
        self.execution.reset()
        self.encoder.gripper_command = self.execution.gripper_command
        self.state = self.encoder.encode()

    def collect(self, count, policy, *, seed, stage, on_step=None):
        if count < 1 or policy not in {"random", "curiosity", "task"}:
            raise ValueError("Positive real-step budget and random/curiosity/task policy required")
        self.reset(seed)
        ticks, skills = [], []
        empty_attempts = 0
        with (self.directory / "transitions.jsonl").open("a") as low_log, \
                (self.directory / "skills.jsonl").open("a") as skill_log:
            while len(ticks) < count:
                self.skill_id += 1
                if policy == "random":
                    index = np.random.randint(self.schema.num_primitives)
                    requested = np.r_[np.eye(self.schema.num_primitives)[index],
                                      np.random.uniform(-1, 1, self.schema.num_parameters)].astype(np.float32)
                else:
                    requested = self.actor.action(self.state)
                actual = canonical_action(torch.as_tensor(requested), self.schema).numpy()
                decoded = self.schema.decode(actual)
                context = self.env.env._context()
                self.execution.start(decoded, context)
                primitive = self.execution.primitive
                initial, before = self.state.copy(), snapshot(self.env)
                start = len(ticks)
                terminated = truncated = False
                complete = False
                while len(ticks) < count:
                    context = self.env.env._context()
                    low = self.execution.next_action(context)
                    if low is None:
                        complete = True
                        break
                    current, low_before = self.state.copy(), snapshot(self.env)
                    if self.actor.settings.get("controlled_placement", False):
                        self.env.env.set_ee_orientation_override(self.execution.orientation_hold)
                    _, _, terminated, truncated, info = self.env.step(low)
                    record = info["primitive_record"]
                    if record.duration_low_steps != 1:
                        raise RuntimeError("Skill collector requires the one-tick ee_delta execution path")
                    self.total_steps += 1
                    self.execution.record_action(record.low_actions[0])
                    self.encoder.gripper_command = self.execution.gripper_command
                    self.state = self.encoder.encode()
                    if not np.isfinite(self.state).all():
                        raise FloatingPointError("Nonfinite real skill state")
                    # Match compact primitive execution's every-two-step contact feedback.
                    if (len(ticks) - start + 1) % 2 == 0 or terminated or truncated:
                        self.execution.observe(self.env.env._context())
                    row = jsonable(dict(
                        step=self.total_steps, stage=stage, policy=policy, episode_id=self.episode_id,
                        skill_id=self.skill_id, skill_type=decoded.primitive_type,
                        skill_action_requested=requested, skill_action=actual,
                        state=current, next_state=self.state, requested_action=low,
                        action=record.low_actions[0], full_controller_action=record.full_controller_actions[0],
                        terminated=bool(terminated), truncated=bool(truncated),
                        run_budget_exhausted=len(ticks) + 1 == count,
                        primitive_type="ee_delta", duration_low_steps=1,
                        parameters_physical=dict(zip(("dx", "dy", "dz", "gripper"), record.low_actions[0])),
                        action_units="normalized_controller", status=record.status,
                        contact_events=record.contact_events, outcome_metrics=record.outcome_metrics,
                        before=low_before, after=snapshot(self.env), reward=info["exploration_reward"], planner={},
                    ))
                    low_log.write(json.dumps(row, allow_nan=False) + "\n")
                    ticks.append(row)
                    if on_step is not None:
                        on_step(row)
                    if len(ticks) % 32 == 0 or terminated or truncated:
                        low_log.flush()
                    if terminated or truncated:
                        break
                duration = len(ticks) - start
                if duration == 0:
                    empty_attempts += 1
                    if empty_attempts >= 100:
                        raise RuntimeError("100 consecutive zero-duration skills; check actor/primitive parameters")
                else:
                    empty_attempts = 0
                skill = jsonable(dict(
                    skill_id=self.skill_id, stage=stage, policy=policy, episode_id=self.episode_id,
                    state=initial, next_state=self.state, requested_action=requested, action=actual,
                    primitive_type=decoded.primitive_type, parameters_physical=decoded.parameters,
                    duration_low_steps=duration, terminated=bool(terminated), truncated=bool(truncated),
                    completed=complete, trainable=complete and duration > 0,
                    status=primitive.status if complete else (
                        "terminated" if terminated else "episode_truncated" if truncated else "budget_truncated"),
                    support_stopped_object_ids=self.execution.support_stopped_object_ids,
                    start_step=self.total_steps - duration, end_step=self.total_steps,
                    before=before, after=snapshot(self.env),
                ))
                skill_log.write(json.dumps(skill, allow_nan=False) + "\n")
                skill_log.flush()
                skills.append(skill)
                if (terminated or truncated) and len(ticks) < count:
                    self.reset(seed + self.episode_id)
        return ticks, skills


def skill_arrays(skills):
    valid = [row for row in skills if row["trainable"]]
    if len(valid) < 2:
        raise ValueError("Need at least two completed nonzero-duration skills; inspect skills.jsonl for truncations")
    arrays = replay_arrays(valid)
    arrays["duration_low_steps"] = np.asarray([row["duration_low_steps"] for row in valid], dtype=np.int64)
    return arrays


def save(path, backend, actor, encoder, config, ticks, skills):
    path = Path(path)
    arrays = skill_arrays(skills)
    np.savez_compressed(path.with_suffix(".replay.npz"), **arrays)
    # Full low-level data remain usable for subsequent control-step WMs.
    np.savez_compressed(path.with_suffix(".low_level.npz"), **replay_arrays(ticks))
    torch.save(dict(schema=SCHEMA, backend=backend.state_dict(), actor=actor.state_dict(),
                    rng=rng_state(), state_contract=encoder.contract(),
                    action_contract=action_contract(actor.settings),
                    config=OmegaConf.to_container(config, resolve=True),
                    real_steps=len(ticks), completed_skills=len(arrays["states"]),
                    environment_state_restored=False), path)


def load(path, backend, actor, encoder):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Requires a RAPS/GNN actor checkpoint; 4D CEE-US checkpoints are incompatible")
    if payload["state_contract"] != encoder.contract() or payload["action_contract"] != action_contract(actor.settings):
        raise ValueError("Checkpoint state/action/controller contract differs")
    backend.load_state_dict(payload["backend"])
    actor.load_state_dict(payload["actor"])
    restore_rng(payload["rng"])
    return payload


def resume_history(path, config):
    """Resume only at a saved round boundary, where collection resets physics."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or path.parent.name != "curiosity":
        raise ValueError("train --checkpoint requires a curiosity RAPS/GNN round checkpoint")
    old = deepcopy(payload["config"])
    new = OmegaConf.to_container(config, resolve=True)
    for value in (old, new):
        value.pop("rounds", None)
        value.pop("evaluation_steps", None)
        value["raps_actor"] = {**DEFAULTS, **value["raps_actor"]}
    if old != new:
        raise ValueError("Resume may change only rounds and evaluation_steps; retain saved settings")
    steps = payload["real_steps"]
    collected = steps - int(config.bootstrap_steps)
    if collected < 0 or collected % int(config.steps_per_round):
        raise ValueError("Checkpoint is not at the configured collection-round boundary")
    next_round = collected // int(config.steps_per_round) + 1
    if next_round > int(config.rounds):
        raise ValueError("Resume requires a larger total rounds budget")
    ticks, skills = [], []
    source = path.parent.parent
    for directory in (source / "bootstrap", source / "curiosity"):
        with (directory / "transitions.jsonl").open() as handle:
            ticks.extend(row for line in handle if (row := json.loads(line))["stage"] != "evaluation"
                         and row["step"] <= steps)
        with (directory / "skills.jsonl").open() as handle:
            skills.extend(row for line in handle if (row := json.loads(line))["stage"] != "evaluation"
                          and row["end_step"] <= steps)
    if len(ticks) != steps or ticks[-1]["step"] != steps:
        raise ValueError("Source JSONL does not contain the complete saved training prefix")
    with np.load(path.with_suffix(".replay.npz")) as saved:
        for key, value in skill_arrays(skills).items():
            np.testing.assert_array_equal(value, saved[key], err_msg=f"Resume replay mismatch: {key}")
    return ticks, skills, next_round


def fit(backend, actor, skills, config, directory, label, *, smoke=False):
    arrays = skill_arrays(skills)
    started = time.perf_counter()
    model_metrics = backend.train(arrays, epochs=1 if smoke else int(config.epochs))
    actor_metrics = actor.train(arrays, updates=2 if smoke else None)
    result = dict(label=label, real_steps=sum(row["duration_low_steps"] for row in skills),
                  completed_skills=len(arrays["states"]), model=model_metrics, actor=actor_metrics,
                  seconds=time.perf_counter() - started)
    with (directory / "training.jsonl").open("a") as handle:
        handle.write(json.dumps(result, allow_nan=False) + "\n")
    return result


def summarize(ticks, skills, schema=RAPS_ACTION_SCHEMA):
    return dict(interactions=interaction_summary(ticks), skill_decisions=len(skills),
                completed_skills=sum(row["trainable"] for row in skills),
                incomplete_skills=sum(not row["completed"] for row in skills),
                per_skill={name: dict(selected=sum(row["primitive_type"] == name for row in skills),
                                     completed=sum(row["primitive_type"] == name and row["trainable"] for row in skills))
                           for name in schema.primitive_names})


def run(args, config):
    config.agent = "raps_gnn_actor"
    config.raps_actor = OmegaConf.merge(DEFAULTS, config.get("raps_actor", {}))
    if args.device:
        config.device = args.device
    if getattr(args, "evaluation_steps", None) is not None:
        config.evaluation_steps = args.evaluation_steps
    output = args.output or ROOT / "logs" / time.strftime("raps_gnn_actor_%Y%m%dT%H%M%S")
    if args.stage == "report":
        if not output.is_dir():
            raise ValueError("report requires an existing --output")
        reports = {path.parent.name: json.loads(path.read_text()) for path in output.glob("*/evaluation.json")}
        verdict = (screen(reports["random"]["interactions"], reports["curiosity"]["interactions"])
                   if {"random", "curiosity"} <= reports.keys() else {"status": "no_matched_random_comparison"})
        write_json(output / "report.json", dict(evaluations=reports, screening=verdict))
        print(json.dumps(verdict, ensure_ascii=False))
        return 0
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if config.get("execution", "reference") != "reference":
        raise ValueError("RAPS actor uses reference GNN execution; set execution: reference")
    if config.get("project_goals_to_region", False):
        raise ValueError("Region projection is not supported by the skill endpoint model")
    if (config.action_mode != "ee_delta" or config.exploration_mode != "baseline"
            or config.sim_freq != 100 or config.control_freq != 20):
        raise ValueError("RAPS actor requires the existing baseline 100/20 Hz ee_delta execution path")
    for key in ("bootstrap_steps", "rounds", "steps_per_round", "epochs", "cpu_threads"):
        if int(config.get(key, 1)) < 1:
            raise ValueError(f"{key} must be positive")
    if int(config.evaluation_steps) < 0:
        raise ValueError("evaluation_steps must be nonnegative")
    if args.stage == "evaluate" and int(config.evaluation_steps) == 0:
        raise ValueError("evaluate requires positive evaluation_steps")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    if args.stage == "evaluate" and args.checkpoint is None:
        raise ValueError("evaluate requires --checkpoint")
    torch.set_num_threads(int(config.get("cpu_threads", 1)))
    random.seed(int(config.seed)); np.random.seed(int(config.seed)); torch.manual_seed(int(config.seed))
    backend = CEEUSBackend(device=str(config.device), seed=int(config.seed),
                           action_dim=skill_schema(config.raps_actor).action_dim,
                           agent_dim=32 + int(config.raps_actor.latched_gripper),
                           object_heuristic=OmegaConf.to_container(config.object_heuristic, resolve=True)
                           if config.get("object_heuristic") is not None else None,
                           runtime_optimizations=bool(config.get("runtime_optimizations", False)))
    actor = RAPSImaginationActor(backend, OmegaConf.to_container(config.raps_actor, resolve=True))
    from exploration.cee_us_state import CabinetSkillStateEncoder
    env = None
    prepared = False
    try:
        env, env_config = make_env(
            episode_steps=config.get("episode_steps", 100),
            centered_initial_tcp=config.get("centered_initial_tcp", False),
            object_shift_xy=getattr(args, "object_shift_xy", (0.0, 0.0)),
        )
        env.reset(seed=int(config.seed))
        encoder = CabinetSkillStateEncoder(env, latched_gripper=actor.settings["latched_gripper"])
        prepare_output(output, config, env_config)
        prepared = True
        write_json(output / "action_contract.json", action_contract(actor.settings))
        write_json(output / "state_contract.json", encoder.contract())
        manifest = json.loads((output / "manifest.json").read_text())
        manifest.update(agent=config.agent, checkpoint_schema=SCHEMA, raps_reference_commit=RAPS_REFERENCE_COMMIT,
                        episode_steps=env_config.max_ee_steps_per_episode,
                        prediction_horizon_seconds=None, imagination_horizon_skills=actor.settings["imagination_horizon"],
                        log_flush_interval_steps=32,
                        adaptations=manifest["adaptations"] + [
                            "RAPS reference conditional actor/MLP/lambda-return; state GNN replaces RSSM",
                            "GNN endpoint sample std summed over dynamic dimensions, not latent P2E std mean",
                            "Fixed per-skill discount; no learned continuation or duration model",
                            "Shared-state ensemble disagreement, ensemble-mean imagined state propagation",
                            "Inactive model arguments zeroed; reference actor logprob/entropy retains all arguments",
                            "Skill-controller latch and ground-support stop are explicit flag branches",
                            "Enabled command latch is observed as agent state, excluded from curiosity",
                            "Incomplete skills excluded from endpoint replay; all real ticks preserved",
                        ])
        write_json(output / "manifest.json", manifest)
        if args.stage == "evaluate":
            load(args.checkpoint, backend, actor, encoder)
            collector = SkillCollector(env, encoder, actor, output / "evaluation")
            ticks, skills = collector.collect(int(config.evaluation_steps), args.policy,
                                              seed=int(config.seed) + 100000, stage="evaluation")
            result = summarize(ticks, skills, actor.schema)
            valid = [row for row in skills if row["trainable"]]
            result["prediction"] = (prediction_metrics(backend, skill_arrays(valid), horizons=(1, 3, 5))
                                    if len(valid) >= 2 else {"status": "insufficient_completed_skills"})
            result.update(model_updated=False, actor_updated=False, normalizers_updated=False,
                          source_checkpoint=str(args.checkpoint.resolve()),
                          object_shift_xy_m=list(getattr(args, "object_shift_xy", (0.0, 0.0))))
            write_json(collector.directory / "evaluation.json", result)
            return 0
        smoke = args.stage == "smoke"
        resuming = args.stage == "train" and args.checkpoint is not None
        start_round = 1
        if resuming:
            ticks, skills, start_round = resume_history(args.checkpoint, config)
            load(args.checkpoint, backend, actor, encoder)
            initial_episode = ticks[-1]["episode_id"]
            initial_skill = max(row["skill_id"] for row in skills)
            # Keep independent complete JSONL logs; do not modify the source run.
            for name, rows in (("transitions.jsonl", ticks), ("skills.jsonl", skills)):
                for directory in ("bootstrap", "curiosity"):
                    target = output / directory
                    target.mkdir(exist_ok=True)
                    with (target / name).open("w") as handle:
                        for row in rows:
                            if (row["stage"] == "bootstrap") == (directory == "bootstrap"):
                                handle.write(json.dumps(row, allow_nan=False) + "\n")
            shutil.copy2(args.checkpoint.parent / "training.jsonl", output / "curiosity/training.jsonl")
            manifest.update(resume_checkpoint=str(args.checkpoint.resolve()),
                            imported_training_steps=len(ticks), next_round=start_round,
                            simulation_resume="Physics resets at the next collection round, as in baseline")
            write_json(output / "manifest.json", manifest)
        else:
            bootstrap = SkillCollector(env, encoder, actor, output / "bootstrap")
            ticks, skills = bootstrap.collect(192 if smoke else int(config.bootstrap_steps), "random",
                                              seed=int(config.seed), stage="bootstrap")
            fit(backend, actor, skills, config, bootstrap.directory, "bootstrap", smoke=smoke)
            save(bootstrap.directory / "checkpoint.pt", backend, actor, encoder, config, ticks, skills)
            initial_episode, initial_skill = bootstrap.episode_id, bootstrap.skill_id
        initial_model, initial_actor, initial_rng = deepcopy(backend.state_dict()), deepcopy(actor.state_dict()), rng_state()
        policies = ("random", "curiosity") if args.stage == "compare" else ("curiosity",)
        for policy in policies:
            backend.load_state_dict(deepcopy(initial_model))
            actor.load_state_dict(deepcopy(initial_actor))
            restore_rng(initial_rng)
            history, endpoints = list(ticks), list(skills)
            collector = SkillCollector(env, encoder, actor, output / policy)
            collector.episode_id, collector.skill_id = initial_episode, initial_skill
            collector.total_steps = len(ticks)
            rounds = 1 if smoke else int(config.rounds)
            for index in range(start_round, rounds + 1):
                write_json(output / "progress.json", dict(stage="collecting", policy=policy, round=index,
                                                          arm_training_steps=len(history)))
                latest, macros = collector.collect(32 if smoke else int(config.steps_per_round), policy,
                                                     seed=int(config.seed) + 10000 * index, stage=f"round_{index}")
                history.extend(latest); endpoints.extend(macros)
                write_json(output / "progress.json", dict(stage="training", policy=policy, round=index,
                                                          arm_training_steps=len(history)))
                training = fit(backend, actor, endpoints, config, collector.directory, f"round_{index}", smoke=smoke)
                path = collector.directory / f"round_{index:03d}.pt"
                save(path, backend, actor, encoder, config, history, endpoints)
                write_json(path.with_suffix(".json"), dict(training=training, collection=summarize(latest, macros, actor.schema)))
                print(json.dumps(dict(policy=policy, round=index, real_training_steps=len(history),
                                      completed_skills=training["completed_skills"], actor_loss=training["actor"]["actor_loss"])), flush=True)
            if smoke:
                replay = skill_arrays(endpoints)
                expected = backend.predict(replay["states"][:1], replay["actions"][:1])
                random_before = rng_state()
                expected_action = actor.action(replay["states"][0])
                # Checkpoint stores RNG before the sampled action above.
                restore_rng(random_before)
                save(path, backend, actor, encoder, config, history, endpoints)
                load(path, backend, actor, encoder)
                np.testing.assert_array_equal(expected, backend.predict(replay["states"][:1], replay["actions"][:1]))
                np.testing.assert_array_equal(expected_action, actor.action(replay["states"][0]))
            if not smoke and int(config.evaluation_steps) == 0:
                write_json(collector.directory / "evaluation.json", dict(status="not_run", real_steps=0))
                continue
            evaluation, eval_skills = collector.collect(32 if smoke else int(config.evaluation_steps), "curiosity" if policy == "curiosity" else "random",
                                                         seed=int(config.seed) + 100000, stage="evaluation")
            result = summarize(evaluation, eval_skills, actor.schema)
            # Short smoke/evaluation may contain no complete skill; don't claim
            # that fitting static endpoints validates interaction dynamics.
            valid = [row for row in eval_skills if row["trainable"]]
            if len(valid) >= 2:
                result["prediction"] = prediction_metrics(backend, skill_arrays(valid), horizons=(1, 3, 5))
            else:
                result["prediction"] = {"status": "insufficient_completed_evaluation_skills"}
            write_json(collector.directory / "evaluation.json", result)
        real_steps = len(ticks) + len(policies) * ((rounds - start_round + 1) * (32 if smoke else int(config.steps_per_round))
                                                  + (32 if smoke else int(config.evaluation_steps)))
        if smoke and real_steps > 300:
            raise AssertionError("Physical smoke budget exceeded")
        write_json(output / "result.json", dict(status="completed", real_steps=real_steps,
                                               imported_training_steps=len(ticks) if resuming else 0,
                                               new_real_steps=real_steps - (len(ticks) if resuming else 0),
                                               smoke=smoke, checkpoint_restore_checked=smoke,
                                               matched_random_baseline=args.stage == "compare"))
        write_json(output / "progress.json", dict(stage="completed", real_steps=real_steps))
    except BaseException as exc:
        if prepared:
            write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc),
                                                     status="stopped_without_random_fallback"))
        raise
    finally:
        if env is not None:
            env.close()
    return 0
