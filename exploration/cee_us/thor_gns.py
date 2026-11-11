"""GNS-inspired THOR neighborhoods; not a reproduction of the particle simulator."""
from __future__ import annotations

import math
import torch
from torch import nn


def graph_config(args):
    enabled = getattr(args, 'gns', None) or False
    radius = getattr(args, 'gns_radius', None)
    radius = .5 if radius is None else radius
    if not math.isfinite(radius) or radius < 0:
        raise ValueError('--gns-radius must be finite and nonnegative (metres)')
    return {'mode': 'gns' if enabled else 'full', 'radius': radius if enabled else None}


def restore_graph_config(payload, args):
    saved = payload.get('graph', {'mode': 'full', 'radius': None})
    requested = getattr(args, 'gns', None)
    radius = getattr(args, 'gns_radius', None)
    if requested is not None and requested != (saved['mode'] == 'gns'):
        raise ValueError('Checkpoint graph mode differs from --gns/--no-gns')
    if radius is not None and radius != saved['radius']:
        raise ValueError('Checkpoint graph radius differs from --gns-radius')
    args.gns = saved['mode'] == 'gns'
    args.gns_radius = saved['radius']


class ThorGNSGraph(nn.Module):
    """Reuse existing MLP parameters, packing only selected edges for evaluation.

    Each ensemble member has its own adjacency. Shorter edge lists are padded
    to the largest member list, then excluded from both sums and mean counts.
    Geometry uses raw metres and canonical sizes, not true current colliders.
    """
    def __init__(self, model, forward_model, radius):
        super().__init__()
        if not model.ignore_global_v_node or model.num_message_passing != 1:
            raise ValueError('THOR GNS requires one message pass and global agent context')
        self.edge_mlp = model.edge_mlp
        self.node_mlp = model.node_mlp
        self.global_mlp = model.global_mlp
        self.radius = radius
        self.hidden_dim = model.hidden_dim
        self.aggr_fn = model.aggr_fn
        self.normalizers = {name: getattr(forward_model, name) for name in
                            ('input_normalizer_obj_dyn', 'input_normalizer_obj_stat', 'action_normalizer')}
        self.normalize = forward_model.use_input_normalization

    @torch.no_grad()
    def adjacency(self, objects, static, actions):
        if self.normalize:
            objects = self.normalizers['input_normalizer_obj_dyn'].denormalize(objects)
            static = self.normalizers['input_normalizer_obj_stat'].denormalize(static)
            actions = self.normalizers['action_normalizer'].denormalize(actions)
        n = objects.shape[-2]
        positions, sizes = objects[..., :3], static[..., -3:]
        gap = ((positions.unsqueeze(-2) - positions.unsqueeze(-3)).abs()
               - (sizes.unsqueeze(-2) + sizes.unsqueeze(-3)) / 2).clamp_min(0)
        adjacent = gap.square().sum(-1) <= self.radius ** 2
        parents = objects[..., 29:29+n] >= .5
        adjacent |= parents | parents.transpose(-1, -2)
        held = objects[..., 11] >= .5
        target = (actions[..., 14:] >= .5) & (actions[..., 9:10] >= .5)
        put = held.unsqueeze(-1) & target.unsqueeze(-2)
        adjacent |= put | put.transpose(-1, -2)
        adjacent &= ~torch.eye(n, dtype=torch.bool, device=objects.device)
        return adjacent

    def forward(self, agent, objects, static, actions):
        e, b, n, _ = objects.shape
        adjacent = self.adjacency(objects, static, actions)
        # Pack independently: do not union member graphs or evaluate all N*(N-1) messages.
        lists = [adjacent[i].nonzero() for i in range(e)]
        length = max(x.shape[0] for x in lists)
        indices = torch.zeros((e, length, 3), dtype=torch.long, device=objects.device)
        valid = torch.zeros((e, length), dtype=torch.bool, device=objects.device)
        for member, edges in enumerate(lists):
            indices[member, :len(edges)] = edges
            valid[member, :len(edges)] = True
        members = torch.arange(e, device=objects.device)[:, None]
        sample, row, col = indices.unbind(-1)
        nodes = torch.cat([objects, static], -1)
        shared = torch.cat([agent, actions], -1)
        if length:
            messages = self.edge_mlp(torch.cat([nodes[members, sample, row],
                                                nodes[members, sample, col], shared[members, sample]], -1))
            messages = messages * valid[..., None]
        else:
            messages = objects.new_zeros((e, 0, self.hidden_dim))

        def aggregate(ids, count):
            out = objects.new_zeros((e, count, self.hidden_dim))
            out.scatter_add_(1, ids[..., None].expand(-1, -1, self.hidden_dim), messages)
            if self.aggr_fn == 'mean':
                degree = objects.new_zeros((e, count, 1))
                degree.scatter_add_(1, ids[..., None], valid[..., None].to(objects.dtype))
                out = out / degree.clamp_min(1)
            return out

        local = aggregate(sample * n + row, b*n).reshape(e, b, n, -1)
        global_edges = aggregate(sample, b)
        next_objects = self.node_mlp(torch.cat([nodes, shared.unsqueeze(-2).expand(-1, -1, n, -1), local], -1).reshape(e, b*n, -1))
        next_agent = self.global_mlp(torch.cat([shared, global_edges], -1))
        return next_agent, next_objects.reshape(e, b, n, -1)
