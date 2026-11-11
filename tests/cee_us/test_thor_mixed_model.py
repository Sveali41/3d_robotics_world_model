"""Mixed THOR losses, decoding, optimizer ownership and rollout semantics."""
import math

import numpy as np
import torch

from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.cee_us.thor_cem import ThorCEM
from exploration.cee_us.thor_mixed_model import ThorMixedForwardModel
from exploration.cee_us.thor_state import ThorState
from tests.cee_us.test_thor_training import replay
from tests.cee_us.test_thor_world_model import metadata


def setup_model():
    torch.set_num_threads(1)
    initial = metadata()
    encoder = ThorState(initial)
    backend = CEEUSBackend(agent_dim=9, object_dyn_dim=encoder.object_dyn_dim,
                          object_stat_dim=encoder.object_stat_dim, num_objects=encoder.n,
                          object_names=encoder.ids, action_dim=14 + encoder.n,
                          hidden_dim=8, batch_size=2, attach_icem=False,
                          forward_model_class=ThorMixedForwardModel)
    return backend, encoder, replay(encoder, initial)


def batch(data):
    return {name: np.repeat(data[key][None], 5, axis=0) for name, key in
            [('observations', 'states'), ('next_observations', 'next_states'), ('actions', 'actions')]}


def zero_model(backend):
    with torch.no_grad():
        for param in backend.model.model.parameters():
            param.zero_()


def test_losses_use_next_labels_and_keep_success_independent_of_object_count():
    backend, encoder, data = setup_model()
    zero_model(backend)
    # On a 1->0 transition, the BCE target must be 0, not delta -1.
    data['next_states'][:, 8] = 0
    with torch.no_grad():
        backend.model.model.global_mlp.classification.bias[:, 1] = 2
    terms = backend.model.loss_terms(batch(data))
    torch.testing.assert_close(terms['success'], torch.full((5,), math.log1p(math.exp(2))))
    torch.testing.assert_close(terms['flags'], torch.full((5,), math.log(2)))
    torch.testing.assert_close(terms['liquid'], torch.full((5,), math.log(4)))
    torch.testing.assert_close(terms['temperature'], torch.full((5,), math.log(3)))
    expected = sum(terms[k] for k in ('standing', 'success', 'flags', 'parents', 'liquid', 'temperature')) / 6
    torch.testing.assert_close(terms['classification'], expected)
    expected_total = (terms['regression'] + expected).sum()
    torch.testing.assert_close(backend.model.loss(batch(data)), expected_total)
    by_member, eval_loss = backend.model.do_evaluate_eval(batch(data))
    assert len(by_member) == 5
    torch.testing.assert_close(eval_loss, expected_total)
    backend.model.loss(batch(data)).backward()
    # Positive logit on an actual failure must be pushed downward.
    assert (backend.model.model.global_mlp.classification.bias.grad[:, 1] > 0).all()


def test_predictions_decode_absolute_classes_and_normalized_continuous_deltas():
    backend, encoder, data = setup_model()
    zero_model(backend)
    with torch.no_grad():
        for name in ('output_agent_normalizer', 'output_obj_normalizer'):
            norm = getattr(backend.model, name)
            norm.mean.fill_(2)
            norm.std.fill_(3)
        backend.model.model.global_mlp.classification.bias[:, 1] = -8
    # Previous success differs, but the absolute classification is the same.
    data['states'][0, 8] = 0
    pred = backend.predict(data['states'][:2], data['actions'][:2])
    assert (pred[..., 8] < .001).all()
    np.testing.assert_allclose(pred[0, :, 8], pred[1, :, 8])
    np.testing.assert_allclose(pred[..., :7], np.broadcast_to(data['states'][None, :2, :7] + 2, pred[..., :7].shape))
    np.testing.assert_array_equal(pred[..., encoder.dynamic_dim:], np.broadcast_to(data['states'][None, :2, encoder.dynamic_dim:], pred[..., encoder.dynamic_dim:].shape))
    obj = pred[..., 9:encoder.dynamic_dim].reshape(5, 2, encoder.n, encoder.object_dyn_dim)
    np.testing.assert_allclose(obj[..., 9:21], .5)
    np.testing.assert_allclose(obj[..., 29:], .5)
    np.testing.assert_allclose(obj[..., 22:26].sum(-1), 1)
    np.testing.assert_allclose(obj[..., 26:29].sum(-1), 1)


def test_fit_updates_new_heads_checkpoint_and_multistep_cem():
    backend, encoder, data = setup_model()
    zero_model(backend)
    data['next_states'][:, 8] = 0
    before = backend.model.model.global_mlp.classification.bias.detach().clone()
    optimizer_ids = {id(p) for group in backend.model.optimizer.param_groups for p in group['params']}
    assert optimizer_ids == {id(p) for p in backend.model.model.parameters()}
    result = backend.train(data, 2)
    assert result['finitegrad'] and np.isfinite(result['loss'])
    assert (backend.model.model.global_mlp.classification.bias[:, 1] < before[:, 1]).all()
    saved = backend.state_dict()
    other, _, _ = setup_model()
    other.load_state_dict(saved, restore_planner=False)
    np.testing.assert_array_equal(backend.predict(data['states'][:2], data['actions'][:2]),
                                  other.predict(data['states'][:2], data['actions'][:2]))
    planner = ThorCEM(backend, encoder, horizon=3, candidates=3, iterations=1, elites=2, chunk_size=2)
    sequences = np.asarray([[0, 8, 9], [1, 10, 11], [12, 13, 0]])
    scores = planner.score(data['states'][0], sequences)
    assert scores.shape == (3,) and np.isfinite(scores).all()
    # Compare the planner objective to manually decoded member-wise rollouts.
    obs = torch.tensor(data['states'][0])[None, None].expand(5, 3, -1)
    manual = torch.zeros(3)
    for step in range(3):
        actions, _ = encoder.action_encoding(obs, torch.tensor(sequences[:, step]))
        obs = backend.model.predict(observations=obs, states=None, actions=actions)[0]
        manual += obs.std(dim=0, unbiased=True).sum(-1)
    np.testing.assert_allclose(scores, manual.numpy(), rtol=1e-5, atol=1e-5)
