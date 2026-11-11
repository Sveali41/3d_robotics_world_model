"""Dynamic graph semantics, dense equivalence, sparse work and restore contracts."""
import copy

import numpy as np
import pytest
import torch

from exploration.cee_us.cee_us_thor import make_backend, restore_checkpoint, save_checkpoint, read_checkpoint
from exploration.cee_us.thor_cem import ThorCEM
from exploration.cee_us.thor_gns import ThorGNSGraph
from exploration.cee_us.thor_state import ThorState
from tests.cee_us.test_thor_training import settings, replay
from tests.cee_us.test_thor_world_model import metadata


def setup(mode='mixed', radius=.5, count=3):
    torch.set_num_threads(1)
    initial = metadata()
    template = initial['objects'][0]
    initial['objects'] = [dict(copy.deepcopy(template), objectId=str(i),
                               position={'x': float(i*10), 'y': 0., 'z': 0.}) for i in range(count)]
    encoder = ThorState(initial)
    args = settings()
    args.prediction_mode, args.gns, args.gns_radius = mode, True, radius
    backend = make_backend(encoder, args)
    return backend, encoder, replay(encoder, initial), args, initial


def raw_inputs(backend, encoder, data):
    obs = torch.tensor(data['states'][:2])[None].expand(5, -1, -1).clone()
    action = torch.tensor(data['actions'][:2])[None].expand(5, -1, -1).clone()
    return backend.model._get_model_input_for_predict(obs, action)[:4]


@pytest.mark.parametrize('mode', ['mse', 'mixed'])
def test_full_radius_matches_original_outputs_and_gradients(mode):
    sparse, encoder, data, args, _ = setup(mode, 100.)
    dense_args = copy.deepcopy(args); dense_args.gns = False
    dense = make_backend(encoder, dense_args)
    dense.model.model.load_state_dict(sparse.model.model.state_dict())
    inputs = raw_inputs(sparse, encoder, data)
    expected = dense.model.model(*inputs)
    actual = sparse.model.model(*inputs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    sum(x.square().sum() for x in actual).backward()
    sum(x.square().sum() for x in expected).backward()
    for a, b in zip(sparse.model.model.parameters(), dense.model.model.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)


def test_dynamic_graph_raw_units_parents_put_and_member_independence():
    backend, encoder, data, _, _ = setup()
    agent, objects, static, actions = raw_inputs(backend, encoder, data)
    # Build raw synthetic observations, then normalize with nontrivial statistics.
    objects = torch.zeros_like(objects); static = torch.zeros_like(static)
    objects[..., 0] = torch.tensor([0., 10., 20.])
    static[..., -3:] = 1
    objects[0, 0, 1, 0] = 1.4  # gap 0.4m, only member0/sample0
    objects[1, 0, 0, 31] = 1  # parent of object0 is object2
    objects[2, 0, 0, 11] = 1  # held object0
    actions.zero_(); actions[2, 0, 9] = 1; actions[2, 0, 16] = 1
    norms = backend.model.model.normalizers
    for norm in norms.values():
        norm.mean.fill_(3); norm.std.fill_(2)
    normalized = [norms[name].normalize(value) for name, value in
                  [('input_normalizer_obj_dyn', objects), ('input_normalizer_obj_stat', static), ('action_normalizer', actions)]]
    adjacency = backend.model.model.adjacency(*normalized)
    assert adjacency[0, 0, 0, 1] and adjacency[0, 0, 1, 0]
    assert adjacency[1, 0, 0, 2] and adjacency[1, 0, 2, 0]
    assert adjacency[2, 0, 0, 2] and adjacency[2, 0, 2, 0]
    assert not adjacency[3:].any() and not adjacency.diagonal(dim1=-2, dim2=-1).any()
    objects[0, 0, 1, 0] = 10
    normalized[0] = norms['input_normalizer_obj_dyn'].normalize(objects)
    assert not backend.model.model.adjacency(*normalized)[0].any()


def test_empty_graph_and_packing_skip_dense_edge_mlp_work():
    backend, encoder, data, _, _ = setup()
    agent, objects, static, actions = raw_inputs(backend, encoder, data)
    graph = backend.model.model
    calls = []
    hook = graph.edge_mlp.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape))
    # All objects far apart: isolated nodes and global state must remain finite.
    result = graph(agent, objects, static, actions)
    assert calls == [] and all(torch.isfinite(x).all() for x in result)
    # Only member0 has two directed parent edges; padded edges must not affect others.
    objects = objects.clone(); objects[0, 0, 0, 30] = 1
    sparse_result = graph(agent, objects, static, actions)
    assert calls[-1][1] == 2 < 2*encoder.n*(encoder.n-1)
    for before, after in zip(result, sparse_result):
        torch.testing.assert_close(before[1:], after[1:])
    hook.remove()


def test_training_checkpoint_graph_inference_and_cem(tmp_path):
    backend, encoder, data, args, initial = setup()
    fit = backend.train(data, 1)
    assert fit['finitegrad'] and np.isfinite(fit['loss'])
    owned = {id(p) for group in backend.model.optimizer.param_groups for p in group['params']}
    assert owned == {id(p) for p in backend.model.model.parameters()}
    path = tmp_path/'gns.pt'
    save_checkpoint(path, backend, encoder, initial, data, np.random.default_rng(0), args)
    payload = read_checkpoint(path)
    assert payload['graph'] == {'mode': 'gns', 'radius': .5}
    restored, _, _ = restore_checkpoint(payload, encoder, settings())
    assert isinstance(restored.model.model, ThorGNSGraph)
    np.testing.assert_array_equal(backend.predict(data['states'][:2], data['actions'][:2]),
                                  restored.predict(data['states'][:2], data['actions'][:2]))
    bad = copy.deepcopy(args); bad.gns_radius = .25
    with pytest.raises(ValueError, match='radius'):
        restore_checkpoint(payload, encoder, bad)
    bad.gns = False; bad.gns_radius = None
    with pytest.raises(ValueError, match='graph mode'):
        restore_checkpoint(payload, encoder, bad)
    planner = ThorCEM(restored, encoder, horizon=3, candidates=3, iterations=1, elites=2, chunk_size=2)
    sequences = np.array([[0, 8, 9], [1, 10, 11], [12, 13, 0]])
    scores = planner.score(data['states'][0], sequences)
    obs = torch.tensor(data['states'][0])[None, None].expand(5, 3, -1)
    manual = torch.zeros(3)
    for step in range(3):
        actions, _ = encoder.action_encoding(obs, torch.tensor(sequences[:, step]))
        obs = restored.model.predict(observations=obs, states=None, actions=actions)[0]
        manual += obs.std(dim=0, unbiased=True).sum(-1)
    np.testing.assert_allclose(scores, manual.numpy(), rtol=1e-5, atol=1e-5)
