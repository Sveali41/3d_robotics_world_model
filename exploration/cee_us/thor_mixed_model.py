"""THOR research variant: continuous deltas and next-state classification.

Keeps the upstream message passing and training iterator. Regression averages
agent/object losses equally; classification averages six losses equally:
standing, action outcome, object flags, parent relations, liquid, temperature.
The total is regression + classification, summed over ensemble members.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from exploration.cee_us.official.mbrl.models.gnn_ensemble import GNNForwardEnsembleModel
from exploration.cee_us.official.mbrl.models.torch_parallel_ensembles.utils import MLPParallelEnsembleLayer


class _SplitReadout(nn.Module):
    """Retain two shared hidden layers, with separate linear output heads."""

    def __init__(self, original, continuous):
        super().__init__()
        output = original.layers[-1][0]
        continuous = tuple(continuous)
        discrete = tuple(i for i in range(output.out_features) if i not in continuous)
        self.trunk = original.layers[:-1]
        self.regression = MLPParallelEnsembleLayer(output.ensemble_size, output.in_features, len(continuous))
        self.classification = MLPParallelEnsembleLayer(output.ensemble_size, output.in_features, len(discrete))
        self.register_buffer('continuous', torch.tensor(continuous, dtype=torch.long, device=output.weight.device))
        self.register_buffer('discrete', torch.tensor(discrete, dtype=torch.long, device=output.weight.device))
        self.output_dim = output.out_features
        self.to(output.weight.device)
        with torch.no_grad():
            for head, indices in ((self.regression, self.continuous), (self.classification, self.discrete)):
                head.weight.copy_(output.weight[..., indices])
                head.bias.copy_(output.bias[..., indices])

    def forward(self, inputs):
        hidden = self.trunk(inputs)
        output = hidden.new_empty((*hidden.shape[:-1], self.output_dim))
        output[..., self.continuous] = self.regression(hidden)
        output[..., self.discrete] = self.classification(hidden)
        return output


class ThorMixedForwardModel(GNNForwardEnsembleModel):
    """Uses unchanged encoded states; predictions contain class probabilities.

Binary and categorical fields are next-state predictions, not delta targets.
Only continuous deltas are denormalized and added to the current state.
Soft probabilities feed the existing rollout and nearest-target thresholds;
categorical CEM retains sum-of-ensemble-standard-deviations on decoded states.
"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if (self.agent_dim != 9 or self.object_dyn_dim != 29 + self.env.nObj
                or not self.agent_as_global_node or not self.target_is_delta):
            raise ValueError('Mixed heads require the THOR agent9/object(29+N) delta-state contract')
        self.model.node_mlp = _SplitReadout(self.model.node_mlp, (*range(9), 21))
        self.model.global_mlp = _SplitReadout(self.model.global_mlp, range(7))
        # The optimizer must own the new heads, not the replaced output layers.
        self._init_for_training()

    def loss_terms(self, batch):
        agent, objects, static, actions, target_agent, target_objects = self._process_batch(batch)
        pred_agent, pred_objects = self.model(agent, objects, static, actions)
        truth = self._prep_batch(batch)[2]
        truth_agent, truth_objects, _ = self._obs_preprocessing(truth)

        def mse(pred, target):
            return (pred - target).square().flatten(1).mean(-1)

        def binary(pred, target):
            return F.binary_cross_entropy_with_logits(pred, target, reduction='none').flatten(1).mean(-1)

        def categorical(pred, target):
            return -(target * F.log_softmax(pred, dim=-1)).sum(-1).flatten(1).mean(-1)

        indices = self.model.node_mlp.continuous
        terms = {
            'agent_continuous': mse(pred_agent[..., :7], target_agent[..., :7]),
            'object_continuous': mse(pred_objects[..., indices], target_objects[..., indices]),
            'standing': binary(pred_agent[..., 7:8], truth_agent[..., 7:8]),
            'success': binary(pred_agent[..., 8:9], truth_agent[..., 8:9]),
            'flags': binary(pred_objects[..., 9:21], truth_objects[..., 9:21]),
            'parents': self.parent_loss(pred_objects[..., 29:], truth_objects[..., 29:]),
            'liquid': categorical(pred_objects[..., 22:26], truth_objects[..., 22:26]),
            'temperature': categorical(pred_objects[..., 26:29], truth_objects[..., 26:29]),
        }
        terms['regression'] = (terms['agent_continuous'] + terms['object_continuous']) / 2
        terms['classification'] = sum(terms[k] for k in
            ('standing', 'success', 'flags', 'parents', 'liquid', 'temperature')) / 6
        return terms

    def parent_loss(self, prediction, target):
        return F.binary_cross_entropy_with_logits(prediction, target, reduction='none').flatten(1).mean(-1)

    def parent_probabilities(self, prediction):
        return prediction.sigmoid()

    def loss(self, batch, target=None):
        if target is not None:
            raise ValueError('Mixed dynamics targets must come from next_observations')
        terms = self.loss_terms(batch)
        return (terms['regression'] + terms['classification']).sum()

    @torch.no_grad()
    def do_evaluate_eval(self, eval_set):
        terms = self.loss_terms(eval_set)
        per_model = terms['regression'] + terms['classification']
        return {f'model{i}': value.item() for i, value in enumerate(per_model)}, per_model.sum()

    @torch.no_grad()
    def predict(self, *, observations, states, actions):
        agent, objects, static, action, current = self._get_model_input_for_predict(observations, actions)
        raw_agent, raw_objects = self.model(agent, objects, static, action)
        result = current.clone()
        result[..., :7] = current[..., :7] + self.output_agent_normalizer.denormalize(raw_agent)[..., :7]
        result[..., 7:9] = raw_agent[..., 7:9].sigmoid()
        current_objects = current[..., self.agent_dim:self.agent_dim + objects.shape[2] * self.object_dyn_dim].reshape_as(raw_objects)
        next_objects = raw_objects.clone()
        indices = self.model.node_mlp.continuous
        delta = self.output_obj_normalizer.denormalize(raw_objects.reshape(-1, self.object_dyn_dim)).reshape_as(raw_objects)
        next_objects[..., indices] = current_objects[..., indices] + delta[..., indices]
        next_objects[..., 9:21] = raw_objects[..., 9:21].sigmoid()
        next_objects[..., 22:26] = raw_objects[..., 22:26].softmax(-1)
        next_objects[..., 26:29] = raw_objects[..., 26:29].softmax(-1)
        next_objects[..., 29:] = self.parent_probabilities(raw_objects[..., 29:])
        result[..., self.agent_dim:self.agent_dim + objects.shape[2] * self.object_dyn_dim] = next_objects.flatten(-2)
        return self.env.obs_postproc(result), None, None
