"""Round-boundary learning and persistence for RGB-only stochastic JEPA."""

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from exploration.cee_us.cee_us_cabinet import rng_state, restore_rng
from .replay import VisualReplay


SCHEMA = "cabinet-stochastic-jepa-cem-v1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Learner:
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
        self.updates = 0

    def assess(self, replay, split="validation"):
        windows = replay.windows(split)
        if not windows:
            return dict(status="insufficient_windows", windows=0)
        self.model.eval()
        metrics, latents, errors, variances = [], [], [], []
        device = next(self.model.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        # Held-out diagnostics must not consume the training/planning RNG stream.
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(self.config["seed"] + 2000000)
            for start in range(0, len(windows), self.config["batch_size"]):
                batch = replay.batch(windows[start:start + self.config["batch_size"]])
                loss, _ = self.model.loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite held-out JEPA loss")
                metrics.append((float(loss), len(batch["pixels"])))
                encoded = self.model.encode_history(batch["pixels"][:, :3], batch["history_mask"])
                latents.append(encoded[:, -1].detach().cpu())
                noise = torch.randn(len(encoded), self.config["planner"]["num_futures"],
                                    3, encoded.shape[-1], device=device)
                predicted = self.model.sample_futures(encoded, batch["history_actions"], batch["history_mask"],
                                                      batch["future_actions"], noise)
                target = self.model.encode_history(batch["pixels"][:, 3:], batch["future_mask"])
                errors.append(((predicted.mean(1) - target).square().mean().item(), len(encoded)))
                variances.append((predicted.var(1, unbiased=True).mean().item(), len(encoded)))
        encoded = torch.cat(latents)
        return dict(status="evaluated", windows=len(windows),
                    loss=sum(value * size for value, size in metrics) / len(windows),
                    predictive_latent_mse=sum(value * size for value, size in errors) / len(windows),
                    sampled_future_variance=sum(value * size for value, size in variances) / len(windows),
                    latent_variance=float(encoded.var(0, unbiased=False).mean()),
                    latent_mean_norm=float(encoded.mean(0).norm()))

    def fit(self, replay):
        windows = replay.windows("train")
        if len(windows) < 2:
            raise ValueError("Insufficient visual training windows: need >=2 consecutive three-skill windows. "
                             "Increase the explicit bootstrap budget or episode length; state-only data cannot substitute RGB.")
        started = time.perf_counter()
        losses, gradient_norms = [], []
        self.model.train()
        for _ in range(self.config["epochs"]):
            order = torch.randperm(len(windows)).tolist()
            for offset in range(0, len(order), self.config["batch_size"]):
                selected = order[offset:offset + self.config["batch_size"]]
                if len(selected) < 2:
                    continue  # The upstream projector uses BatchNorm during training.
                batch = replay.batch([windows[index] for index in selected])
                self.optimizer.zero_grad(set_to_none=True)
                loss, _ = self.model.loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite stochastic JEPA training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
                self.optimizer.step()
                self.updates += 1
                losses.append(float(loss.detach()))
                gradient_norms.append(float(norm))
        self.model.eval()
        return dict(loss=float(np.mean(losses)), gradient_norm=float(np.mean(gradient_norms)),
                    updates=self.updates, train_windows=len(windows),
                    fit_seconds=time.perf_counter() - started, validation=self.assess(replay))


def checkpoint(path, learner, planner, replay, collector, config, next_round, controls, history):
    payload = dict(schema=SCHEMA, config=deepcopy(config), model=learner.model.state_dict(),
                   optimizer=learner.optimizer.state_dict(), updates=learner.updates,
                   planner=planner.state_dict(), replay=replay.snapshot(), rng=rng_state(),
                   episode_id=collector.episode_id, skill_id=collector.skill_id,
                   next_round=next_round, controls=controls, history=deepcopy(history),
                   action_contract=dict(skill_set="split7", action_dim=13),
                   environment_state_restored=False)
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path, config):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or payload.get("config") != config:
        raise ValueError("Checkpoint schema/config differs; use its saved JEPA configuration and round budget")
    if payload.get("action_contract") != dict(skill_set="split7", action_dim=13):
        raise ValueError("Checkpoint split7 contract differs")
    VisualReplay.verify_snapshot(payload["replay"])
    return payload


def run_loop(config, output, learner, planner, replay, collector, *, resumed=None):
    """Production orchestration; collector injection also permits bounded fake-env tests."""
    output = Path(output)
    controls, history, first_round = 0, [], -1
    if resumed is not None:
        learner.model.load_state_dict(resumed["model"])
        learner.optimizer.load_state_dict(resumed["optimizer"])
        learner.updates = resumed["updates"]
        planner.load_state_dict(resumed["planner"])
        collector.episode_id, collector.skill_id = resumed["episode_id"], resumed["skill_id"]
        collector.total_steps = resumed["controls"]
        controls, history, first_round = resumed["controls"], resumed["history"], resumed["next_round"]
        restore_rng(resumed["rng"])
        with (output / "training.jsonl").open("w") as stream:
            for row in history:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    for round_id in range(first_round, config["rounds"]):
        bootstrap = round_id == -1
        count = config["bootstrap_steps"] if bootstrap else config["steps_per_round"]
        if controls + count > config["total_controls"]:
            raise ValueError("Declared control budget exhausted before requested round")
        stage = "bootstrap" if bootstrap else f"round_{round_id:03d}"
        learner.model.eval()
        started = time.perf_counter()
        ticks, skills = collector.collect(count, "random" if bootstrap else config["collection_policy"],
                                          seed=config["seed"] + 10000 * (round_id + 2), stage=stage)
        if len(ticks) > count:
            raise RuntimeError("Collector exceeded physical control budget")
        controls += len(ticks)
        collected = dict(seconds=time.perf_counter() - started, controls=len(ticks), skills=len(skills),
                         completed=sum(bool(row["completed"]) for row in skills))
        # Privileged interaction summaries are diagnostics only, never planner rewards.
        if ticks:
            from exploration.raps.raps_gnn_actor import summarize
            collected["diagnostics"] = summarize(ticks, skills, collector.schema)
        training = learner.fit(replay)
        if training["validation"]["status"] != "evaluated":
            raise ValueError("No held-out three-skill RGB windows. Increase bootstrap/episode budget "
                             "so validation episodes contain consecutive completed primitives.")
        row = dict(stage=stage, round=round_id, real_controls=controls,
                   collection=collected, training=training)
        audits = getattr(collector.actor, "audits", [])
        if audits:
            row["planning"] = audits.copy()
            audits.clear()
        history.append(row)
        with (output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(output / "progress.json", dict(state="running", **row))
        checkpoint(output / f"{stage}.pt", learner, planner, replay, collector,
                   config, round_id + 1, controls, history)
    result = dict(status="complete", real_controls=controls, updates=learner.updates,
                  replay_transitions=len(replay.rows), history=history,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                  interpretation="Integration evidence; no exploration advantage established")
    write_json(output / "result.json", result)
    write_json(output / "progress.json", dict(state="complete", real_controls=controls, updates=learner.updates))
    return result
