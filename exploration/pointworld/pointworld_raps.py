"""PointWorld-facing Plan2Explore acquisition utilities.

This is the point-cloud analogue of RAPS' one-step ensemble acquisition:
candidate actions are proposed before simulator execution, all ensemble
members see the same candidate robot flow, and only epistemic disagreement is
used by the selector.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class AcquisitionConfig:
    sigma: float = 0.05
    epsilon: float = 0.20
    temperature: float = 0.10
    top_quantile: float = 0.10


class PointWorldDisagreement:
    def __init__(self, config: AcquisitionConfig | None = None):
        self.config = config or AcquisitionConfig()

    @torch.no_grad()
    def score(self, predictions: torch.Tensor, scene_points: torch.Tensor,
              robot_flows: torch.Tensor, valid_mask: torch.Tensor | None = None):
        """Return one epistemic score per candidate.

        Args:
            predictions: ``[E,K,T,Ns,3]`` predicted scene positions or flows.
            scene_points: ``[K,Ns,3]`` or ``[Ns,3]`` initial scene points.
            robot_flows: ``[K,T,Nr,3]`` predicted robot point trajectory.
            valid_mask: optional ``[K,T,Ns]`` mask.
        """
        if predictions.ndim != 5 or predictions.shape[-1] != 3:
            raise ValueError("predictions must have shape [E,K,T,Ns,3]")
        E, K, T, Ns, _ = predictions.shape
        if robot_flows.shape[:2] != (K, T):
            raise ValueError("robot_flows must have shape [K,T,Nr,3]")
        if scene_points.ndim == 2:
            scene_points = scene_points.unsqueeze(0).expand(K, -1, -1)
        if scene_points.shape != (K, Ns, 3):
            raise ValueError("scene_points must have shape [K,Ns,3] or [Ns,3]")
        disagreement = predictions.var(dim=0, unbiased=False).sum(dim=-1)  # K,T,Ns
        distances = torch.cdist(scene_points, robot_flows.reshape(K, T * robot_flows.shape[2], 3))
        neighborhood = torch.exp(-(distances.min(dim=-1).values ** 2) / (2 * self.config.sigma ** 2))
        weights = neighborhood.unsqueeze(1).expand(K, T, Ns)
        if valid_mask is not None:
            weights = weights * valid_mask.to(weights.dtype)
        denom = weights.sum(dim=(1, 2)).clamp_min(1e-8)
        score = (weights * disagreement).sum(dim=(1, 2)) / denom
        flat = disagreement.flatten(1)
        q = max(1, int(flat.shape[1] * self.config.top_quantile))
        top_q = flat.topk(q, dim=1).values.mean(dim=1)
        return {"score": score, "top_quantile_score": top_q, "point_disagreement": disagreement}


class PointWorldEnsemble:
    """Small batched adapter around the repository's PointWorld BaseModels."""

    def __init__(self, models):
        if not models:
            raise ValueError("PointWorldEnsemble needs at least one model")
        self.models = list(models)

    @torch.no_grad()
    def predict(self, candidate_batch, encoded_scene_features=None):
        outputs = []
        for index, model in enumerate(self.models):
            encoded = None
            if encoded_scene_features is not None:
                encoded = encoded_scene_features[index] if isinstance(encoded_scene_features, (list, tuple)) else encoded_scene_features
            result = model(candidate_batch, training=False, encoded_scene_feat0=encoded)
            if "scene_relative" in result:
                outputs.append(result["scene_relative"])
            elif "pred" in result:
                outputs.append(result["pred"])
            else:
                raise KeyError("PointWorld model output must contain scene_relative or pred")
        return torch.stack(outputs, dim=0)


class RAPSSelector:
    def __init__(self, config: AcquisitionConfig | None = None, seed: int = 0):
        self.config = config or AcquisitionConfig()
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    def select(self, scores: torch.Tensor, valid_mask: torch.Tensor | None = None):
        scores = scores.detach().float()
        valid = torch.ones_like(scores, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
        if not valid.any():
            raise ValueError("no valid RAPS candidates")
        masked = scores.masked_fill(~valid, -torch.inf)
        probabilities = torch.softmax(masked / max(self.config.temperature, 1e-6), dim=0)
        random_index = torch.where(valid)[0][torch.randint(valid.sum(), (1,), generator=self.generator)]
        greedy_index = torch.argmax(masked)
        explore = torch.rand((), generator=self.generator).item() < self.config.epsilon
        selected = random_index if explore else greedy_index
        return {"index": selected, "probabilities": probabilities, "random": explore}
