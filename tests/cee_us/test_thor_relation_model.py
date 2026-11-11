"""Relation head candidates, balanced gradients, training-only counts and persistence."""
import copy
import math

import numpy as np
import pytest
import torch

from exploration.cee_us.cee_us_thor import make_backend
from exploration.cee_us.official.mbrl.rolloutbuffer import Rollout, RolloutBuffer
from exploration.cee_us.thor_cem import ThorCEM
from exploration.cee_us.thor_state import ThorState
from tests.cee_us.test_thor_training import settings, replay
from tests.cee_us.test_thor_world_model import metadata


def setup(gns=False):
    torch.set_num_threads(1)
    initial = metadata()
    template = initial['objects'][0]
    initial['objects'] = [dict(copy.deepcopy(template), objectId=name, receptacle=name == 'b') for name in ('a', 'b', 'c')]
    encoder = ThorState(initial)
    args = settings(); args.prediction_mode = 'mixed-relations'; args.gns = gns
    backend = make_backend(encoder, args)
    data = replay(encoder, initial)
    # 2 positives and 10 negatives over the 2 valid slots in each of 6 records.
    data['next_states'][:2, 9+29+1] = 1
    return backend, encoder, data


def normalize(backend, data):
    backend.model.update_normalizer(RolloutBuffer(rollouts=[Rollout.from_dict(
        observations=data['states'], actions=data['actions'], next_observations=data['next_states'],
        rewards=np.zeros(len(data['states']), dtype=np.float32))]))


def test_training_only_weight_and_masked_bce_gradient():
    backend, encoder, data = setup()
    normalize(backend, data)
    model = backend.model
    head = model.model.node_mlp
    assert head.relations.out_features == 1
    assert head.classification.out_features == 19
    assert head.parent_positive_count.item() == 2
    assert head.parent_negative_count.item() == 10
    assert head.parent_pos_weight.item() == 5
    assert head.valid_parents.tolist() == [[False, True, False], [False]*3, [False, True, False]]
    logits = torch.zeros((5, 1, 3, 3), requires_grad=True)
    truth = torch.zeros_like(logits); truth[..., 0, 1] = 1
    loss = model.parent_loss(logits, truth)
    torch.testing.assert_close(loss, torch.full((5,), 3*math.log(2)))
    loss.sum().backward()
    assert (logits.grad[..., 0, 1] < 0).all()
    assert (logits.grad[..., 2, 1] > 0).all()
    assert not logits.grad[..., ~head.valid_parents].any()
    torch.testing.assert_close(-logits.grad[..., 0, 1], logits.grad[..., 2, 1]*5)
    # Computing a monitoring loss must never recompute its imbalance weight.
    model.parent_loss(torch.zeros_like(logits), torch.zeros_like(truth))
    assert head.parent_pos_weight.item() == 5
    bad = truth.clone(); bad[..., 0, 0] = 1
    with pytest.raises(ValueError, match='non-receptacle or self-parent'):
        model.parent_loss(logits, bad)


def test_absent_classes_and_illegal_training_labels():
    backend, _, data = setup()
    empty = copy.deepcopy(data); empty['next_states'] = empty['states'].copy()
    normalize(backend, empty)
    assert backend.model.model.node_mlp.parent_pos_weight.item() == 1
    positive = copy.deepcopy(data)
    for source in (0, 2):
        positive['next_states'][:, 9+source*backend.model.object_dyn_dim+30] = 1
    normalize(backend, positive)
    assert backend.model.model.node_mlp.parent_negative_count.item() == 0
    assert backend.model.model.node_mlp.parent_pos_weight.item() == 1
    bad = copy.deepcopy(data); bad['next_states'][0, 9+29] = 1
    with pytest.raises(ValueError, match='non-receptacle or self-parent'):
        normalize(backend, bad)


@pytest.mark.parametrize('gns', [False, True])
def test_decode_checkpoint_fitting_and_member_rollouts(gns):
    backend, encoder, data = setup(gns)
    normalize(backend, data)
    with torch.no_grad():
        for parameter in backend.model.model.parameters():
            parameter.zero_()
    prediction = backend.predict(data['states'][:2], data['actions'][:2])
    parent = prediction[..., 9:encoder.dynamic_dim].reshape(5,2,3,encoder.object_dyn_dim)[...,29:]
    mask = backend.model.model.node_mlp.valid_parents.cpu().numpy()
    assert (parent[...,~mask] == 0).all()
    np.testing.assert_allclose(parent[...,mask], .5)
    fit = backend.train(data, 1)
    assert fit['finitegrad'] and np.isfinite(fit['loss'])
    head = backend.model.model.node_mlp
    # Head must receive gradients, and optimizer must own all three readouts.
    assert head.relations.weight.grad is not None
    owned = {id(p) for group in backend.model.optimizer.param_groups for p in group['params']}
    assert owned == {id(p) for p in backend.model.model.parameters()}
    other, _, _ = setup(gns)
    other.load_state_dict(backend.state_dict(), restore_planner=False)
    assert other.model.model.node_mlp.parent_pos_weight.item() == 5
    np.testing.assert_array_equal(backend.predict(data['states'][:2],data['actions'][:2]),
                                  other.predict(data['states'][:2],data['actions'][:2]))
    planner = ThorCEM(other, encoder, horizon=3, candidates=3, iterations=1, elites=2, chunk_size=2)
    sequences = np.array([[0,8,9],[1,10,11],[12,13,0]])
    scores = planner.score(data['states'][0], sequences)
    obs = torch.tensor(data['states'][0])[None,None].expand(5,3,-1)
    manual = torch.zeros(3)
    for step in range(3):
        action, _ = encoder.action_encoding(obs, torch.tensor(sequences[:,step]))
        obs = other.model.predict(observations=obs,states=None,actions=action)[0]
        manual += obs.std(dim=0,unbiased=True).sum(-1)
    np.testing.assert_allclose(scores,manual.numpy(),rtol=1e-5,atol=1e-5)
