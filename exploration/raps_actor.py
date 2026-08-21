"""RAPS-style hybrid actor, independent of the reference rlkit package.

The distribution mirrors the reference actor's contract: a categorical
primitive choice and a conditional tanh-normal over the shared argument bank.
It is intentionally observation-backend agnostic; an encoded PointWorld scene
feature can be passed directly as ``observation``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal


class RAPSActor(nn.Module):
    def __init__(self, observation_dim: int, schema, hidden_size: int = 256, num_layers: int = 4,
                 min_std: float = 0.1, init_std: float = 0.0):
        super().__init__()
        self.schema = schema
        layers = []
        dim = int(observation_dim)
        for _ in range(int(num_layers)):
            layers.extend([nn.Linear(dim, hidden_size), nn.ELU()])
            dim = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.primitive_head = nn.Linear(dim, schema.num_primitives)
        self.continuous_head = nn.Linear(dim + schema.num_primitives, 2 * schema.max_arg_len)
        self.min_std = float(min_std)
        self.init_std = float(init_std)

    def _stats(self, observation, primitive_one_hot):
        h = self.trunk(observation)
        primitive_logits = self.primitive_head(h)
        conditioned = torch.cat([h, primitive_one_hot], dim=-1)
        mean, raw_std = self.continuous_head(conditioned).chunk(2, dim=-1)
        mean = 5.0 * torch.tanh(mean / 5.0)
        std = torch.nn.functional.softplus(raw_std + self.init_std) + self.min_std
        return primitive_logits, mean, std

    def distribution(self, observation):
        h = self.trunk(observation)
        logits = self.primitive_head(h)
        primitive = Categorical(logits=logits)
        # A uniform one-hot is used for the continuous proposal; callers that
        # need exact conditional log-probs should use sample_with_log_prob.
        one_hot = torch.nn.functional.one_hot(logits.argmax(-1), self.schema.num_primitives).to(h.dtype)
        _, mean, std = self._stats_from_hidden(h, one_hot)
        return primitive, Independent(Normal(torch.tanh(mean), std), 1)

    def _stats_from_hidden(self, h, primitive_one_hot):
        logits = self.primitive_head(h)
        mean, raw_std = self.continuous_head(torch.cat([h, primitive_one_hot], -1)).chunk(2, -1)
        mean = 5.0 * torch.tanh(mean / 5.0)
        std = torch.nn.functional.softplus(raw_std + self.init_std) + self.min_std
        return logits, mean, std

    def sample(self, observation, deterministic: bool = False):
        h = self.trunk(observation)
        logits = self.primitive_head(h)
        primitive_dist = Categorical(logits=logits)
        primitive_idx = logits.argmax(-1) if deterministic else primitive_dist.sample()
        primitive_one_hot = torch.nn.functional.one_hot(primitive_idx, self.schema.num_primitives).to(h.dtype)
        _, mean, std = self._stats_from_hidden(h, primitive_one_hot)
        continuous = torch.tanh(mean) if deterministic else torch.tanh(Normal(mean, std).rsample())
        action = torch.cat([primitive_one_hot, continuous], dim=-1)
        return action, {"primitive_index": primitive_idx, "primitive_logits": logits}
