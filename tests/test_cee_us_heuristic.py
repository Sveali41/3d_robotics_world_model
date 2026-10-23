"""Object approach scoring without physics or model fitting."""
import json

import numpy as np
import pytest
import torch

from exploration.cee_us_backend import CEEUSBackend
from third_party.cee_us.mbrl.controllers.mpc_torch_curiosity import TorchCuriosityMpcICem
from third_party.cee_us.mbrl.rolloutbuffer import SimpleRolloutBuffer


def backend(setting=None, **kwargs):
    return CEEUSBackend(hidden_dim=8, horizon=3, candidates=8, iterations=1,
                        elites=4, object_heuristic=setting, **kwargs)


def state(tcp=0.):
    result = np.zeros(66, dtype=np.float32)
    result[18] = tcp
    result[32] = 1.
    result[44] = -1.
    return result


def costs(b, positions):
    # Include an extra candidate, as upstream appends reused elites.
    before = torch.tensor(state())[None, None, None].expand(9, 5, 3, -1).clone()
    after = before.clone()
    after[..., 18] = torch.tensor(positions)[:, None, None]
    paths = SimpleRolloutBuffer(observations=before, next_observations=after,
                               actions=torch.zeros(9, 5, 3, 4))
    out = torch.empty(9, 5)
    b.planner.trajectory_cost_fn(b.planner.cost_fn, paths, out)
    return out, paths


def test_approach_beats_retreat_and_diagnostics_follow_combined_winner():
    b = backend({'object_weights': {'cabinet_item_16': 1.}})
    out, _ = costs(b, [-.1, .1, 0., 0., 0., 0., 0., 0., 0.])
    assert out.mean(-1).argmin() == 1
    assert out[1, 0] == pytest.approx(-150.)
    assert out[0, 0] == pytest.approx(150.)
    assert float(b.planner.selected_approach_progress_score) == pytest.approx(150.)
    assert torch.count_nonzero(b.planner.selected_std) == 0


def test_stationary_roundtrip_target_weights_and_scaling():
    b = backend({})
    assert b.observed_object_heuristic(state(), state())['approach_progress_score'] == 0.
    # Equal targets on opposite sides cancel; selecting either target reverses ranking.
    assert b.observed_object_heuristic(state(), state(.1))['approach_progress_score'] == pytest.approx(0., abs=1e-4)
    for name, sign in [('cabinet_item_16', 1), ('cabinet_item_17', -1)]:
        b = backend({'approach_weight': 2., 'distance_scale_m': .2,
                     'object_weights': {name: 3.}})
        forward = b.observed_object_heuristic(state(), state(.1))
        backward = b.observed_object_heuristic(state(.1), state())
        assert forward['approach_progress_score'] == pytest.approx(sign)
        assert forward['approach_progress_score'] + backward['approach_progress_score'] == 0.
        assert forward['per_object_progress_m'][name] == pytest.approx(sign * .1)
        json.dumps(forward, allow_nan=False)


def test_disabled_and_zero_weight_keep_original_costs_and_rng():
    for setting in (None, {'approach_weight': 0.}):
        b = backend(setting)
        before = torch.randn(9, 5, 3, 66)
        after = torch.randn_like(before)
        paths = SimpleRolloutBuffer(observations=before, next_observations=after,
                                   actions=torch.zeros(9, 5, 3, 4))
        actual, original = torch.empty(9, 5), torch.empty(9, 5)
        rng = torch.get_rng_state().clone()
        b.planner.trajectory_cost_fn(b.planner.cost_fn, paths, actual)
        TorchCuriosityMpcICem.trajectory_cost_fn(b.planner, b.planner.cost_fn, paths, original)
        assert torch.equal(actual, original)
        assert torch.equal(rng, torch.get_rng_state())


def test_selected_std_uses_combined_winner():
    b = backend({'object_weights': {'cabinet_item_16': 1.}})
    _, paths = costs(b, [-.1, .1, 0., 0., 0., 0., 0., 0., 0.])
    paths['next_observations'][0, :, :, 0] = torch.arange(5)[:, None] * .01
    out = torch.empty(9, 5)
    b.planner.trajectory_cost_fn(b.planner.cost_fn, paths, out)
    assert out.mean(-1).argmin() == 1
    assert b.planner.stds_of_means_[0].sum() > 0
    assert b.planner.selected_std.sum() == 0


def test_checkpoint_configuration_and_legacy():
    b = backend({})
    payload = b.state_dict()
    assert payload['object_heuristic'] == b.object_heuristic
    backend({}).load_state_dict(payload)
    with pytest.raises(ValueError, match='object_heuristic'):
        backend(None).load_state_dict(payload)
    with pytest.raises(ValueError, match='object_heuristic'):
        backend({'approach_weight': 1.}).load_state_dict(payload)
    legacy = backend(None).state_dict()
    del legacy['object_heuristic']
    backend(None).load_state_dict(legacy)
    with pytest.raises(ValueError, match='object_heuristic'):
        b.load_state_dict(legacy)


@pytest.mark.parametrize('setting', [
    {'approach_weight': -1}, {'approach_weight': float('nan')},
    {'distance_scale_m': 0}, {'distance_scale_m': float('inf')},
    {'object_weights': {}}, {'object_weights': {'other': 1}},
    {'object_weights': {'cabinet_item_16': -1}},
    {'object_weights': {'cabinet_item_16': 0}},
    {'object_weights': {'cabinet_item_16': float('inf')}},
    {'unknown': 1},
])
def test_invalid_settings(setting):
    with pytest.raises(ValueError, match='object_heuristic'):
        backend(setting)


def test_fixed_layout_required():
    with pytest.raises(ValueError, match='requires agent_dim=32'):
        backend({}, agent_dim=31)


def test_action_diagnostics_and_zero_weight_parity():
    baseline = backend(None)
    zero = backend({'approach_weight': 0.})
    for b in (baseline, zero):
        b.reset_planner(state())
    torch.manual_seed(19)
    expected, original = baseline.action(state())
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    actual, diagnostics = zero.action(state())
    np.testing.assert_array_equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert diagnostics['curiosity_score'] == original['curiosity_score']
    assert diagnostics['score_decomposition'] == original['score_decomposition']
    assert 'object_heuristic_score' not in original
    assert diagnostics['object_heuristic_score'] == {
        'approach_progress_score': 0., 'combined_score': diagnostics['curiosity_score']}
    json.dumps(diagnostics, allow_nan=False)
