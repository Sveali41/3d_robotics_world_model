"""Separate, candidate-masked multi-label parent head for the THOR mixed model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from exploration.cee_us.thor_mixed_model import ThorMixedForwardModel
from exploration.cee_us.official.mbrl.models.torch_parallel_ensembles.utils import MLPParallelEnsembleLayer


class _RelationReadout(nn.Module):
    """Share the existing trunk, with regression, attribute and relation heads."""
    def __init__(self, original, receptacle_indices, num_objects):
        super().__init__()
        previous = original.classification
        device = previous.weight.device
        candidates = torch.tensor(receptacle_indices, dtype=torch.long, device=device)
        if (not len(candidates) or len(set(receptacle_indices)) != len(candidates)
                or (candidates < 0).any() or (candidates >= num_objects).any()):
            raise ValueError('Relation head requires unique receptacle indices inside the scene contract')
        self.trunk = original.trunk
        self.regression = original.regression
        self.classification = MLPParallelEnsembleLayer(previous.ensemble_size, previous.in_features, 19).to(device)
        self.relations = MLPParallelEnsembleLayer(previous.ensemble_size, previous.in_features, len(candidates)).to(device)
        self.register_buffer('continuous', original.continuous.clone())
        self.register_buffer('discrete', original.discrete[original.discrete < 29].clone())
        self.register_buffer('receptacles', candidates)
        valid = torch.zeros((num_objects, num_objects), dtype=torch.bool, device=device)
        valid[:, candidates] = True
        valid.fill_diagonal_(False)
        self.register_buffer('valid_parents', valid)
        self.register_buffer('parent_pos_weight', torch.tensor(1., device=device))
        self.register_buffer('parent_positive_count', torch.tensor(-1, dtype=torch.long, device=device))
        self.register_buffer('parent_negative_count', torch.tensor(-1, dtype=torch.long, device=device))
        self.output_dim = original.output_dim
        with torch.no_grad():
            for head, indices in ((self.classification, self.discrete), (self.relations, 29 + candidates)):
                columns = (indices[:, None] == original.discrete[None, :]).long().argmax(-1)
                head.weight.copy_(previous.weight[..., columns])
                head.bias.copy_(previous.bias[..., columns])

    def forward(self, inputs):
        hidden = self.trunk(inputs)
        output = hidden.new_zeros((*hidden.shape[:-1], self.output_dim))
        output[..., self.continuous] = self.regression(hidden)
        output[..., self.discrete] = self.classification(hidden)
        output[..., 29 + self.receptacles] = self.relations(hidden)
        return output


class ThorRelationForwardModel(ThorMixedForwardModel):
    """Keep legacy state shape; invalid parent probabilities decode exactly to zero.

    One global positive weight is N_negative/N_positive on valid training labels,
    refreshed only before fitting. If either class is absent, weight is one.
    No monitor labels or per-minibatch counts are used. Parent loss still has
    the original one-of-six classification group weight.
    """
    def __init__(self, *, receptacle_indices, **kwargs):
        super().__init__(**kwargs)
        self.model.node_mlp = _RelationReadout(self.model.node_mlp, receptacle_indices, self.env.nObj)
        self._init_for_training()

    @torch.no_grad()
    def update_normalizer(self, batch):
        # Compute counts on the source device; do not copy the full dataset to GPU.
        next_obs = torch.as_tensor(batch['next_observations'])
        n, d = self.env.nObj, self.object_dyn_dim
        parents = next_obs[..., 9:9+n*d].reshape(-1, n, d)[..., 29:]
        head = self.model.node_mlp
        valid = head.valid_parents.to(parents.device)
        if ((parents >= .5) & ~valid).any():
            raise ValueError('Training parent labels include a non-receptacle or self-parent relation; check scene contract')
        positives = (parents[..., valid] >= .5).sum().item()
        negatives = parents[..., valid].numel() - positives
        weight = negatives / positives if positives and negatives else 1.
        super().update_normalizer(batch)
        head.parent_positive_count.fill_(positives)
        head.parent_negative_count.fill_(negatives)
        head.parent_pos_weight.fill_(weight)

    def parent_loss(self, prediction, target):
        head = self.model.node_mlp
        valid = head.valid_parents
        if (head.parent_positive_count < 0).item():
            raise ValueError('Fit relation weights from training data via update_normalizer before evaluating loss')
        if ((target >= .5) & ~valid).any():
            raise ValueError('Parent labels include a non-receptacle or self-parent relation')
        if not valid.any():
            return prediction.flatten(1).sum(-1) * 0
        loss = F.binary_cross_entropy_with_logits(prediction[..., valid], target[..., valid],
                                                  pos_weight=head.parent_pos_weight, reduction='none')
        return loss.flatten(1).mean(-1)

    def parent_probabilities(self, prediction):
        return prediction.sigmoid() * self.model.node_mlp.valid_parents
