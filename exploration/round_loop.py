"""Round-based Uniform-RAPS / PW-RAPS collection loop.

The loop deliberately depends on small injected callables instead of importing
ManiSkill or PointWorld internals.  This makes the orchestration testable with
fake environments and keeps domain-specific scene decoding in adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import json
import numpy as np
import torch

from domain.robocasa.primitives import UniformPrimitiveExplorer
from .pointworld_raps import PointWorldDisagreement, PointWorldEnsemble, RAPSSelector


@dataclass
class RoundConfig:
    candidates_per_decision: int = 32
    low_level_budget: int = 1000
    max_primitives: int = 100
    bootstrap: bool = False


@dataclass
class CandidateAudit:
    round_id: int
    episode_id: int
    decision_id: int
    actions: np.ndarray
    valid_mask: np.ndarray
    scores: np.ndarray
    selected_index: int
    selection_random: bool
    probabilities: np.ndarray
    primitive_type: str | None = None
    duration_low_steps: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "episode_id": self.episode_id,
            "decision_id": self.decision_id,
            "actions": self.actions.tolist(),
            "valid_mask": self.valid_mask.tolist(),
            "scores": self.scores.tolist(),
            "selected_index": self.selected_index,
            "selection_random": self.selection_random,
            "probabilities": self.probabilities.tolist(),
            "primitive_type": self.primitive_type,
            "duration_low_steps": self.duration_low_steps,
        }


@dataclass
class RoundResult:
    round_id: int
    mode: str
    low_level_steps: int = 0
    decisions: int = 0
    records: list[Any] = field(default_factory=list)
    audits: list[CandidateAudit] = field(default_factory=list)


class RoundController:
    """Execute one frozen-model collection round.

    Required injected functions:

    ``scene_points_fn(observation)`` -> ``[Ns,3]`` tensor/array
    ``robot_flow_fn(observation, actions)`` -> ``[K,T,Nr,3]`` tensor
    ``batch_fn(observation, actions, robot_flows)`` -> PointWorld batch

    The last function may also add explicit ``raps_action`` conditioning.
    ``record_fn(record, audit)`` is called after every real primitive and is
    where the WDS writer/replay index should be attached.
    """

    def __init__(
        self,
        env,
        *,
        candidate_generator: UniformPrimitiveExplorer,
        scene_points_fn: Callable[[Any], Any],
        robot_flow_fn: Callable[[Any, np.ndarray], torch.Tensor],
        batch_fn: Callable[[Any, np.ndarray, torch.Tensor], dict[str, Any]],
        ensemble: PointWorldEnsemble | Any,
        scorer: PointWorldDisagreement,
        selector: RAPSSelector,
        record_fn: Callable[[Any, CandidateAudit | None], None] | None = None,
        audit_dir: str | Path | None = None,
    ):
        self.env = env
        self.candidate_generator = candidate_generator
        self.scene_points_fn = scene_points_fn
        self.robot_flow_fn = robot_flow_fn
        self.batch_fn = batch_fn
        self.ensemble = ensemble
        self.scorer = scorer
        self.selector = selector
        self.record_fn = record_fn
        self.audit_dir = Path(audit_dir) if audit_dir is not None else None

    @staticmethod
    def _reset(env):
        result = env.reset()
        return result[0] if isinstance(result, tuple) else result

    @staticmethod
    def _as_tensor(value) -> torch.Tensor:
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)

    def _write_audits(self, result: RoundResult) -> None:
        if self.audit_dir is None:
            return
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        path = self.audit_dir / f"round_{result.round_id:03d}_candidates.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for audit in result.audits:
                handle.write(json.dumps(audit.to_json()) + "\n")

    def run(self, round_id: int, config: RoundConfig | None = None) -> RoundResult:
        config = config or RoundConfig()
        result = RoundResult(
            round_id=int(round_id),
            mode="bootstrap" if config.bootstrap else "active",
        )
        self.candidate_generator.reset()
        observation = self._reset(self.env)
        episode_id = 0

        while result.low_level_steps < config.low_level_budget and result.decisions < config.max_primitives:
            if config.bootstrap:
                actions = self.candidate_generator.sample_candidates(config.candidates_per_decision)
                selected_index = int(
                    torch.randint(
                        len(actions), (1,), generator=self.selector.generator
                    ).item()
                )
                scores = torch.zeros(len(actions), dtype=torch.float32)
                probabilities = torch.full_like(scores, 1.0 / len(actions))
                selection_random = True
            else:
                actions = self.candidate_generator.sample_candidates(config.candidates_per_decision)
                scene_points = self._as_tensor(self.scene_points_fn(observation))
                robot_flows = self.robot_flow_fn(observation, actions)
                robot_flows = self._as_tensor(robot_flows)
                batch = self.batch_fn(observation, actions, robot_flows)
                predictions = self.ensemble.predict(batch)
                score_data = self.scorer.score(
                    predictions,
                    scene_points,
                    robot_flows,
                    valid_mask=batch.get("temporal_valid_mask"),
                )
                scores = score_data["score"]
                valid_mask = batch.get("candidate_valid_mask")
                if valid_mask is None:
                    valid_mask = torch.ones(len(actions), dtype=torch.bool, device=scores.device)
                choice = self.selector.select(scores, valid_mask)
                selected_index = int(choice["index"].item())
                probabilities = choice["probabilities"]
                selection_random = bool(choice["random"])

            valid_numpy = np.ones(len(actions), dtype=bool)
            audit = CandidateAudit(
                round_id=int(round_id),
                episode_id=episode_id,
                decision_id=result.decisions,
                actions=np.asarray(actions, dtype=np.float32),
                valid_mask=valid_numpy,
                scores=scores.detach().cpu().numpy(),
                selected_index=selected_index,
                selection_random=selection_random,
                probabilities=probabilities.detach().cpu().numpy(),
            )
            step_result = self.env.step(actions[selected_index])
            observation, _, terminated, truncated, info = step_result
            record = info.get("primitive_record") if isinstance(info, dict) else None
            if record is not None:
                audit.primitive_type = getattr(record, "primitive_type", None)
                audit.duration_low_steps = int(getattr(record, "duration_low_steps", 0))
                result.low_level_steps += audit.duration_low_steps
                result.records.append(record)
                if self.record_fn is not None:
                    self.record_fn(record, audit)
            result.audits.append(audit)
            result.decisions += 1
            if isinstance(observation, tuple):
                observation = observation[0]
            if terminated or truncated:
                episode_id += 1
                if result.low_level_steps < config.low_level_budget:
                    observation = self._reset(self.env)

        self._write_audits(result)
        return result
