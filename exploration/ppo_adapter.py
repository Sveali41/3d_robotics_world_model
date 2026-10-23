"""PPO training adaptor for 20D RAPS or 4D Cartesian cabinet actions.

The rollout and clipped PPO update follow the official RAPS PPO setup, while
the environment side uses the current Gymnasium/ManiSkill wrappers directly.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path
import traceback
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal, TanhTransform, TransformedDistribution
from omegaconf import DictConfig, OmegaConf


TRAINING_METRIC_FIELDS = (
    "update",
    "total_primitives",
    "transitions",
    "mean_reward",
    "contact_rate",
    "new_contact_rate",
    "meaningful_contact_rate",
    "new_grasp_rate",
    "moved_object_rate",
    "timeout_rate",
    "ee_outside_rate",
    "completion_rate",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
)


def _as_bool(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().bool().any().item())
    return bool(np.asarray(value).any())


def _as_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    return float(np.asarray(value, dtype=np.float32).mean())


def _observation_tensor(observation, device: torch.device) -> torch.Tensor:
    if isinstance(observation, torch.Tensor):
        tensor = observation.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(observation, device=device, dtype=torch.float32)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError(
            "PPO adaptor expects one flat state observation, "
            f"got shape={tuple(tensor.shape)}"
        )
    return tensor


def _observation_numpy(observation) -> np.ndarray:
    """Convert a worker observation to a small CPU payload for IPC."""
    if isinstance(observation, torch.Tensor):
        observation = observation.detach().cpu().numpy()
    array = np.asarray(observation, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(
            "PPO worker expects one flat state observation, "
            f"got shape={array.shape}"
        )
    return np.ascontiguousarray(array)


def _environment_worker(
    remote,
    env_factory: Callable[[DictConfig], object],
    config_payload: dict,
    worker_index: int,
    base_seed: int,
) -> None:
    """Run one complete cabinet environment outside the PPO process."""
    env = None
    try:
        # Prevent several workers from multiplying PyTorch's default CPU
        # thread pool for the small contact-force queries.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        worker_cfg = OmegaConf.create(config_payload)
        env = env_factory(worker_cfg)
        observation, _ = env.reset(seed=int(base_seed) + worker_index)
        remote.send(
            (
                "ready",
                _observation_numpy(observation),
                np.asarray(env.action_space.low, dtype=np.float32),
                np.asarray(env.action_space.high, dtype=np.float32),
            )
        )

        while True:
            command, payload = remote.recv()
            if command == "step":
                observation, reward, terminated, truncated, info = env.step(payload)
                report = info.get("exploration_reward", {})
                summary = {
                    "primitive_type": info.get("primitive_type", "unknown"),
                    "terms": dict(report.get("terms", {})),
                    "raw": dict(report.get("raw", {})),
                    "contacted_object_ids": list(
                        report.get("contacted_object_ids", [])
                    ),
                    "new_object_ids": list(report.get("new_object_ids", [])),
                    "exploration_complete": bool(
                        report.get("exploration_complete", False)
                    ),
                    "ever_grasped_object_ids": list(
                        report.get("ever_grasped_object_ids", [])
                    ),
                    "ever_moved_object_ids": list(
                        report.get("ever_moved_object_ids", [])
                    ),
                }
                remote.send(
                    (
                        "step",
                        _observation_numpy(observation),
                        _as_scalar(reward),
                        _as_bool(terminated),
                        _as_bool(truncated),
                        summary,
                    )
                )
            elif command == "reset":
                observation, _ = env.reset(seed=int(payload))
                remote.send(("reset", _observation_numpy(observation)))
            elif command == "close":
                break
            else:
                raise ValueError(f"unknown environment worker command: {command!r}")
    except BaseException:
        try:
            remote.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        remote.close()


class ParallelCabinetEnvs:
    """Spawn independent CPU cabinet environments and batch their transitions."""

    def __init__(
        self,
        env_factory: Callable[[DictConfig], object],
        cfg: DictConfig,
        num_envs: int,
    ):
        if int(num_envs) < 1:
            raise ValueError("ppo.num_envs must be positive")
        self.num_envs = int(num_envs)
        context = mp.get_context("spawn")
        config_payload = OmegaConf.to_container(cfg, resolve=True)
        self.remotes = []
        self.processes = []
        for worker_index in range(self.num_envs):
            parent_remote, child_remote = context.Pipe()
            process = context.Process(
                target=_environment_worker,
                args=(
                    child_remote,
                    env_factory,
                    config_payload,
                    worker_index,
                    int(cfg.seed),
                ),
                daemon=True,
            )
            process.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(process)

        try:
            ready = [self._receive(remote) for remote in self.remotes]
        except BaseException:
            self.close()
            raise
        if any(message[0] != "ready" for message in ready):
            self.close()
            raise RuntimeError(f"failed to start cabinet workers: {ready}")
        self.observations = np.stack([message[1] for message in ready])
        self.action_low = ready[0][2]
        self.action_high = ready[0][3]
        for message in ready[1:]:
            if not np.array_equal(message[2], self.action_low) or not np.array_equal(
                message[3], self.action_high
            ):
                self.close()
                raise RuntimeError("cabinet workers expose different action spaces")

    @staticmethod
    def _receive(remote):
        message = remote.recv()
        if message[0] == "error":
            raise RuntimeError(f"cabinet worker failed:\n{message[1]}")
        return message

    def step(self, actions: np.ndarray) -> list[tuple]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.action_low.shape[0]):
            raise ValueError(
                f"expected batched actions with shape "
                f"{(self.num_envs, self.action_low.shape[0])}, got {actions.shape}"
            )
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", np.ascontiguousarray(action)))
        return [self._receive(remote) for remote in self.remotes]

    def reset(self, worker_index: int, seed: int) -> np.ndarray:
        self.remotes[worker_index].send(("reset", int(seed)))
        message = self._receive(self.remotes[worker_index])
        if message[0] != "reset":
            raise RuntimeError(f"unexpected reset response: {message[0]!r}")
        return message[1]

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for remote in self.remotes:
            remote.close()


class RAPSActorCritic(nn.Module):
    """PPO actor with a categorical primitive and bounded parameters.

    With action_dim=4, all entries are continuous Cartesian/gripper commands.
    Otherwise the action follows the official 20D RAPS layout. The first ten
    entries are emitted as a one-hot categorical sample; the final ten entries
    are tanh-normal parameters. Only the parameter slots used by the sampled
    primitive contribute to the continuous log probability.
    """

    PARAMETER_MASK = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int,
        action_std: float,
    ):
        super().__init__()
        if action_std <= 0:
            raise ValueError("ppo.action_std must be positive")
        if action_dim not in (4, 20):
            raise ValueError("PPO requires a 20D primitive or 4D ee_delta action")
        self.num_primitives = 10 if action_dim == 20 else 0
        self.num_parameters = 10 if action_dim == 20 else 4
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.num_primitives + self.num_parameters),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(
            torch.full((self.num_parameters,), float(np.log(action_std)), dtype=torch.float32)
        )
        if self.num_primitives:
            self.register_buffer("parameter_mask", self.PARAMETER_MASK)
        self._initialize()

    def _initialize(self) -> None:
        for network, final_gain in ((self.actor, 0.01), (self.critic, 1.0)):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    gain = final_gain if layer is network[-1] else np.sqrt(2.0)
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)

    def distribution(self, observation: torch.Tensor):
        output = self.actor(observation)
        logits = output[..., : self.num_primitives]
        mean = output[..., self.num_primitives :]
        categorical = Categorical(logits=logits) if self.num_primitives else None
        continuous = TransformedDistribution(
            Normal(mean, self.log_std.exp().expand_as(mean)),
            TanhTransform(cache_size=1),
        )
        return categorical, continuous

    def _log_prob_entropy(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        categorical, continuous = self.distribution(observation)
        if categorical is None:
            return (continuous.log_prob(action).sum(dim=-1),
                    continuous.base_dist.entropy().sum(dim=-1))
        primitive_indices = action[..., : self.num_primitives].argmax(dim=-1)
        parameter_mask = self.parameter_mask[primitive_indices].to(action.device)
        selector_log_prob = categorical.log_prob(primitive_indices)
        parameter_log_prob = continuous.log_prob(action[..., self.num_primitives:])
        log_prob = selector_log_prob + (parameter_log_prob * parameter_mask).sum(dim=-1)
        # TransformedDistribution entropy is not implemented for tanh-normal;
        # the untransformed entropy is a stable approximation for PPO's bonus.
        parameter_entropy = Normal(
            continuous.base_dist.loc, continuous.base_dist.scale
        ).entropy()
        entropy = categorical.entropy() + (parameter_entropy * parameter_mask).sum(dim=-1)
        return log_prob, entropy

    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        categorical, continuous = self.distribution(observation)
        if categorical is None:
            action = continuous.sample()
            log_prob, _ = self._log_prob_entropy(observation, action)
            return action, log_prob, self.value(observation)
        primitive_indices = categorical.sample()
        primitive_one_hot = torch.nn.functional.one_hot(
            primitive_indices, self.num_primitives
        ).to(dtype=observation.dtype)
        parameters = continuous.sample()
        action = torch.cat([primitive_one_hot, parameters], dim=-1)
        log_prob, _ = self._log_prob_entropy(observation, action)
        value = self.critic(observation).squeeze(-1)
        return action, log_prob, value

    @torch.no_grad()
    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the modal primitive and tanh-normal mean parameters."""
        categorical, continuous = self.distribution(observation)
        if categorical is None:
            return torch.tanh(continuous.base_dist.loc)
        primitive_indices = categorical.probs.argmax(dim=-1)
        primitive_one_hot = torch.nn.functional.one_hot(
            primitive_indices, self.num_primitives
        ).to(dtype=observation.dtype)
        parameters = torch.tanh(continuous.base_dist.loc)
        return torch.cat([primitive_one_hot, parameters], dim=-1)

    def evaluate(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prob, entropy = self._log_prob_entropy(observations, actions)
        value = self.critic(observations).squeeze(-1)
        return log_prob, entropy, value

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)


class Rollout:
    """A batched rollout with independent GAE traces for each worker."""

    def __init__(self):
        self.observations: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.next_values: list[torch.Tensor] = []
        self.rewards: list[np.ndarray] = []
        self.terminated: list[np.ndarray] = []
        self.truncated: list[np.ndarray] = []

    def append_batch(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        self.observations.append(observation.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.next_values.append(next_value.detach())
        self.rewards.append(np.asarray(reward, dtype=np.float32))
        self.terminated.append(np.asarray(terminated, dtype=np.bool_))
        self.truncated.append(np.asarray(truncated, dtype=np.bool_))

    def tensors(
        self, device: torch.device, gamma: float, gae_lambda: float
    ) -> dict[str, torch.Tensor]:
        # Keep the time/env axes while computing GAE, then flatten them for
        # PPO minibatches. This prevents advantages leaking between workers.
        rewards = torch.as_tensor(np.stack(self.rewards), device=device, dtype=torch.float32)
        values = torch.stack(self.values).to(device)
        next_values = torch.stack(self.next_values).to(device)
        terminated = torch.as_tensor(
            np.stack(self.terminated), device=device, dtype=torch.float32
        )
        truncated = torch.as_tensor(
            np.stack(self.truncated), device=device, dtype=torch.float32
        )
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((rewards.shape[1],), device=device)

        for step in reversed(range(len(self.rewards))):
            bootstrap_mask = 1.0 - terminated[step]
            continuation_mask = 1.0 - torch.maximum(
                terminated[step], truncated[step]
            )
            delta = rewards[step] + gamma * bootstrap_mask * next_values[step] - values[step]
            gae = delta + gamma * gae_lambda * continuation_mask * gae
            advantages[step] = gae

        return {
            "observations": torch.stack(self.observations).to(device).flatten(0, 1),
            "actions": torch.stack(self.actions).to(device).flatten(0, 1),
            "old_log_probs": torch.stack(self.log_probs).to(device).flatten(0, 1),
            "old_values": values.flatten(0, 1),
            "advantages": advantages.flatten(0, 1),
            "returns": (advantages + values).flatten(0, 1),
        }

    def __len__(self) -> int:
        return sum(reward.size for reward in self.rewards)


def _ppo_update(
    actor_critic: RAPSActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    tensors = rollout.tensors(device, float(cfg.gamma), float(cfg.gae_lambda))
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"PPO rollout {name} contains NaN or infinity; inspect the collected transition")
    advantages = tensors["advantages"]
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    if not torch.isfinite(advantages).all():
        raise ValueError("PPO normalized advantages are non-finite; collect at least two finite transitions")
    batch_size = len(rollout)
    num_mini_batch = min(int(cfg.num_mini_batch), batch_size)
    mini_batch_size = max(1, batch_size // num_mini_batch)
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
               "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
    update_count = 0

    for _ in range(int(cfg.ppo_epochs)):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, mini_batch_size):
            batch_indices = indices[start : start + mini_batch_size]
            new_log_probs, entropy, values = actor_critic.evaluate(
                tensors["observations"][batch_indices],
                tensors["actions"][batch_indices],
            )
            for name, tensor in (("log probabilities", new_log_probs),
                                 ("entropy", entropy), ("values", values)):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"PPO policy {name} contains NaN or infinity; inspect policy parameters and observations")
            log_ratio = new_log_probs - tensors["old_log_probs"][batch_indices]
            ratio = log_ratio.exp()
            surrogate_1 = ratio * advantages[batch_indices]
            surrogate_2 = ratio.clamp(
                1.0 - float(cfg.clip_param), 1.0 + float(cfg.clip_param)
            ) * advantages[batch_indices]
            policy_loss = -torch.minimum(surrogate_1, surrogate_2).mean()

            value_delta = values - tensors["old_values"][batch_indices]
            value_clipped = tensors["old_values"][batch_indices] + value_delta.clamp(
                -float(cfg.clip_param), float(cfg.clip_param)
            )
            value_loss = 0.5 * torch.maximum(
                (values - tensors["returns"][batch_indices]).square(),
                (value_clipped - tensors["returns"][batch_indices]).square(),
            ).mean()
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + float(cfg.value_loss_coef) * value_loss
                - float(cfg.entropy_coef) * entropy_mean
            )
            if not torch.isfinite(loss):
                raise ValueError("PPO loss is non-finite; inspect log-probability ratios, returns, and learning rate")
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > float(cfg.clip_param)).float().mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), float(cfg.max_grad_norm), error_if_nonfinite=True
            )
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in actor_critic.parameters()):
                raise ValueError("PPO optimizer produced non-finite parameters; inspect optimizer state and learning rate")

            metrics["loss"] += float(loss.detach().item())
            metrics["policy_loss"] += float(policy_loss.detach().item())
            metrics["value_loss"] += float(value_loss.detach().item())
            metrics["entropy"] += float(entropy_mean.detach().item())
            metrics["approx_kl"] += float(approx_kl.item())
            metrics["clip_fraction"] += float(clip_fraction.item())
            update_count += 1

    if update_count == 0:
        raise ValueError("PPO update requires at least one epoch and a nonempty rollout")
    metrics = {key: value / update_count for key, value in metrics.items()}
    return_variance = tensors["returns"].var(unbiased=False)
    metrics["explained_variance"] = (
        float((1.0 - (tensors["returns"] - tensors["old_values"]).var(unbiased=False)
               / return_variance).item())
        if return_variance > 0 else 0.0
    )
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("PPO diagnostics are non-finite; inspect return and value magnitudes")
    return metrics


def policy_checkpoint_path(cfg: DictConfig) -> Path:
    directory = Path(str(cfg.ppo.checkpoint_dir))
    if str(getattr(cfg, "action_mode", "primitives")) == "ee_delta":
        directory = directory / "ee_delta"
    if str(getattr(cfg, "exploration_mode", "baseline")) == "D":
        directory = directory / "D"
    return directory / str(cfg.ppo.checkpoint_name)


def _save_checkpoint(
    path: Path,
    actor_critic: RAPSActorCritic,
    optimizer: torch.optim.Optimizer,
    update_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": actor_critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update_index,
            "action_mode": "primitives" if actor_critic.num_primitives else "ee_delta",
        },
        path,
    )


def run_ppo(cfg: DictConfig, env_factory: Callable[[DictConfig], object]) -> int:
    """Train PPO with one high-level RAPS primitive per transition.

    ``ppo.rollout_steps`` is measured independently in every worker.  The
    policy is evaluated on a batch of worker observations, while each worker
    keeps its own simulator and episode-level reward state.
    """

    ppo_cfg = cfg.ppo
    if int(ppo_cfg.rollout_steps) < 2:
        raise ValueError("ppo.rollout_steps must be at least 2")
    if int(ppo_cfg.num_updates) < 1:
        raise ValueError("ppo.num_updates must be positive")
    num_envs = int(ppo_cfg.num_envs)
    if num_envs < 1:
        raise ValueError("ppo.num_envs must be positive")
    if not bool(cfg.headless) and num_envs > 1:
        raise ValueError("parallel PPO workers require headless=true")

    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = ParallelCabinetEnvs(env_factory, cfg, num_envs)
    checkpoint_path = policy_checkpoint_path(cfg)
    metrics_path = checkpoint_path.parent / str(ppo_cfg.metrics_filename)
    total_primitives = 0
    episode_index = 0
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    metrics_file = None

    try:
        observation_tensors = torch.stack(
            [_observation_tensor(observation, device) for observation in workers.observations]
        )
        observation_dim = int(observation_tensors.shape[-1])
        action_dim = int(workers.action_low.shape[0])
        actor_critic = RAPSActorCritic(
            observation_dim,
            action_dim,
            int(ppo_cfg.hidden_size),
            float(ppo_cfg.action_std),
        ).to(device)
        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=float(ppo_cfg.learning_rate))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w", newline="", encoding="ascii")
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=TRAINING_METRIC_FIELDS)
        metrics_writer.writeheader()
        metrics_file.flush()

        for update_index in range(1, int(ppo_cfg.num_updates) + 1):
            rollout = Rollout()
            reward_terms: dict[str, float] = {}
            completions = 0
            event_counts = {
                "contact": 0,
                "new_contact": 0,
                "meaningful_contact": 0,
                "new_grasp": 0,
                "moved_object": 0,
                "timeout": 0,
                "ee_outside": 0,
            }

            for _ in range(int(ppo_cfg.rollout_steps)):
                with torch.no_grad():
                    action, log_prob, value = actor_critic.act(observation_tensors)

                raw_actions = action.detach().cpu().numpy()
                env_actions = np.clip(
                    raw_actions, workers.action_low, workers.action_high
                ).astype(np.float32)
                results = workers.step(env_actions)
                next_observations = np.stack([result[1] for result in results])
                next_observation_tensors = torch.stack(
                    [
                        _observation_tensor(observation, device)
                        for observation in next_observations
                    ]
                )
                with torch.no_grad():
                    next_values = actor_critic.value(next_observation_tensors)

                for worker_index, result in enumerate(results):
                    (
                        _message_type,
                        _next_observation,
                        reward_value,
                        terminated,
                        truncated,
                        report,
                    ) = result
                    total_primitives += 1
                    episode_returns[worker_index] += float(reward_value)

                    for name, term in report["terms"].items():
                        reward_terms[name] = reward_terms.get(name, 0.0) + float(term)
                    completions += int(report["exploration_complete"])
                    raw = report["raw"]
                    event_counts["contact"] += int(
                        bool(report["contacted_object_ids"])
                    )
                    event_counts["new_contact"] += int(bool(report["new_object_ids"]))
                    event_counts["meaningful_contact"] += int(
                        int(raw.get("meaningful_contact_count", 0)) > 0
                    )
                    event_counts["new_grasp"] += int(
                        int(raw.get("new_grasped_object_count", 0)) > 0
                    )
                    event_counts["moved_object"] += int(
                        int(raw.get("moved_object_count", 0)) > 0
                    )
                    event_counts["timeout"] += int(
                        bool(raw.get("primitive_timeout", False))
                    )
                    event_counts["ee_outside"] += int(
                        bool(raw.get("ee_outside", False))
                    )

                    if terminated or truncated:
                        episode_index += 1
                        print(
                            f"episode={episode_index:04d} worker={worker_index:02d} "
                            f"primitives={total_primitives:06d} "
                            f"return={episode_returns[worker_index]:+.3f} "
                            f"complete={report['exploration_complete']} "
                            f"grasped={report['ever_grasped_object_ids']} "
                            f"moved={report['ever_moved_object_ids']}"
                        )
                        next_observations[worker_index] = workers.reset(
                            worker_index, int(cfg.seed) + episode_index + num_envs
                        )
                        episode_returns[worker_index] = 0.0

                rollout.append_batch(
                    observation_tensors,
                    action,
                    log_prob,
                    value,
                    next_values,
                    np.asarray([result[2] for result in results], dtype=np.float32),
                    np.asarray([result[3] for result in results], dtype=np.bool_),
                    np.asarray([result[4] for result in results], dtype=np.bool_),
                )

                observation_tensors = torch.stack(
                    [
                        _observation_tensor(observation, device)
                        for observation in next_observations
                    ]
                )

            if len(rollout) < 2:
                break
            metrics = _ppo_update(actor_critic, optimizer, rollout, ppo_cfg, device)
            mean_reward = float(np.mean(rollout.rewards))
            mean_terms = {
                name: value / len(rollout) for name, value in sorted(reward_terms.items())
            }
            transitions = len(rollout)
            metric_row = {
                "update": update_index,
                "total_primitives": total_primitives,
                "transitions": transitions,
                "mean_reward": mean_reward,
                "contact_rate": event_counts["contact"] / transitions,
                "new_contact_rate": event_counts["new_contact"] / transitions,
                "meaningful_contact_rate": event_counts["meaningful_contact"] / transitions,
                "new_grasp_rate": event_counts["new_grasp"] / transitions,
                "moved_object_rate": event_counts["moved_object"] / transitions,
                "timeout_rate": event_counts["timeout"] / transitions,
                "ee_outside_rate": event_counts["ee_outside"] / transitions,
                "completion_rate": completions / transitions,
                **metrics,
            }
            metrics_writer.writerow(metric_row)
            metrics_file.flush()
            print(
                f"update={update_index:04d} envs={num_envs:02d} "
                f"primitives={total_primitives:06d} "
                f"reward={mean_reward:+.3f} loss={metrics['loss']:+.4f} "
                f"policy_loss={metrics['policy_loss']:+.4f} "
                f"value_loss={metrics['value_loss']:+.4f} "
                f"entropy={metrics['entropy']:+.4f} "
                f"contact_rate={metric_row['contact_rate']:.3f} "
                f"grasp_rate={metric_row['new_grasp_rate']:.3f} "
                f"completions={completions} "
                f"terms={mean_terms}"
            )
            if update_index % int(ppo_cfg.save_interval) == 0:
                _save_checkpoint(checkpoint_path, actor_critic, optimizer, update_index)

        _save_checkpoint(checkpoint_path, actor_critic, optimizer, update_index)
        print(f"PPO checkpoint saved to {checkpoint_path}")
        return 0
    finally:
        workers.close()
        if metrics_file is not None:
            metrics_file.close()


def run_evaluation(cfg: DictConfig, env_factory: Callable[[DictConfig], object]) -> int:
    """Evaluate a saved hybrid policy with deterministic primitive selection."""
    ppo_cfg = cfg.ppo
    env = env_factory(cfg)
    checkpoint_path = policy_checkpoint_path(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = None
    episode_reports = []
    try:
        observation, _ = env.reset(seed=int(cfg.seed))
        observation_tensor = _observation_tensor(observation, device)
        actor_critic = RAPSActorCritic(
            int(observation_tensor.shape[-1]),
            int(env.action_space.shape[0]),
            int(ppo_cfg.hidden_size),
            float(ppo_cfg.action_std),
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected_parameters = actor_critic.num_parameters
        if tuple(checkpoint["model"]["log_std"].shape) != (expected_parameters,):
            raise ValueError(f"{checkpoint_path} is incompatible with action_mode={cfg.action_mode}")
        actor_critic.load_state_dict(checkpoint["model"])
        actor_critic.eval()

        for episode in range(int(getattr(ppo_cfg, "eval_episodes", 20))):
            if episode > 0:
                observation, _ = env.reset(seed=int(cfg.seed) + episode)
                observation_tensor = _observation_tensor(observation, device)
            ever_grasped = set()
            lifted = set()
            contact = timeout = pushed = outside = False
            for _ in range(int(cfg.max_primitives_per_episode)):
                action = actor_critic.deterministic_action(
                    observation_tensor.unsqueeze(0)
                )[0].cpu().numpy()
                observation, _, terminated, truncated, info = env.step(action)
                observation_tensor = _observation_tensor(observation, device)
                report = info.get("exploration_reward", {})
                raw = report.get("raw", {})
                ever_grasped.update(report.get("ever_grasped_object_ids", []))
                lifted.update(report.get("lifted_object_ids", []))
                contact |= bool(report.get("contacted_object_ids"))
                timeout |= bool(raw.get("primitive_timeout", False))
                pushed |= int(raw.get("moved_object_count", 0)) > 0
                outside |= bool(raw.get("ee_outside", False))
                if terminated or truncated:
                    break
            episode_reports.append(
                {
                    "grasped": sorted(ever_grasped),
                    "lifted": sorted(lifted),
                    "contact": contact,
                    "timeout": timeout,
                    "pushed": pushed,
                    "ee_outside": outside,
                }
            )

        object_ids = {"cabinet_item_16", "cabinet_item_17"}
        # Contact is a diagnostic; physical success requires the object to
        # move upward during a subsequent lift primitive.
        at_least_one = sum(bool(set(r["lifted"]) & object_ids) for r in episode_reports)
        all_grasped = sum(object_ids <= set(r["lifted"]) for r in episode_reports)
        mug = sum("cabinet_item_16" in r["lifted"] for r in episode_reports)
        bowl = sum("cabinet_item_17" in r["lifted"] for r in episode_reports)
        count = max(1, len(episode_reports))
        print(
            "Evaluation "
            f"episodes={len(episode_reports)} "
            f"mug={mug / count:.3f} bowl={bowl / count:.3f} "
            f"at_least_one={at_least_one / count:.3f} "
            f"all_objects={all_grasped / count:.3f} "
            f"contact={sum(r['contact'] for r in episode_reports) / count:.3f} "
            f"timeout={sum(r['timeout'] for r in episode_reports) / count:.3f} "
            f"push={sum(r['pushed'] for r in episode_reports) / count:.3f} "
            f"ee_outside={sum(r['ee_outside'] for r in episode_reports) / count:.3f}"
        )
        for index, report in enumerate(episode_reports):
            print(
                f"episode={index:04d} contact_grasped={report['grasped']} "
                f"lifted={report['lifted']}"
            )
        return 0
    finally:
        env.close()
