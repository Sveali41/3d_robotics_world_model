import json

import numpy as np
import pytest

from exploration.raps.protected_replay import bounded_dynamic_replay_indices, protected_replay_indices


def replay_data(count=120):
    # Seven arm joints followed by two three-dimensional object dynamics blocks.
    states = np.zeros((count, 13), dtype=np.float64)
    actions = np.zeros((count, 4), dtype=np.float64)
    actions[:, 0] = 1.0
    next_states = states.copy()
    new = np.zeros(count, dtype=bool)
    contact = np.zeros(count, dtype=bool)
    return states, actions, next_states, new, contact


def sample(states, actions, next_states, new, contact, *, updates=1, batch_size=40):
    return protected_replay_indices(states, actions, next_states,
        agent_dim=7, object_dyn_dim=3, num_objects=2, new_mask=new,
        contact_mask=contact, updates=updates, batch_size=batch_size)


@pytest.mark.parametrize('moving_count, expected_weight', [(0, 1), (10, 4), (40, 1.5), (80, 1), (100, 1)])
def test_bounded_dynamic_record_weights_and_rng(moving_count, expected_weight):
    states, _, next_states, _, _ = replay_data(100)
    next_states[:moving_count, 10] = .015
    def draw():
        return bounded_dynamic_replay_indices(states, next_states, agent_dim=7, object_dyn_dim=3,
                                              num_objects=2, updates=20, batch_size=50)
    np.random.seed(42)
    indices, stats = draw()
    np.random.seed(42)
    repeated, repeated_stats = draw()
    np.testing.assert_array_equal(indices, repeated)
    assert stats == repeated_stats and len(indices) == 1000
    assert stats['moving_records'] == moving_count and stats['moving_weight'] == expected_weight
    assert stats['sampled_moving'] == (indices < moving_count).sum()
    probability = moving_count * expected_weight / (100 - moving_count + moving_count * expected_weight)
    assert abs(stats['sampled_moving'] / 1000 - probability) < .06
    json.dumps(stats, allow_nan=False)


def test_exact_sample_count_rng_restore_and_even_source_stratum_quotas():
    states, actions, next_states, new, contact = replay_data()
    new[:40] = True
    # Historical rows 40:80 dynamic, 80:100 contact, 100:120 static.
    next_states[40:80, 7,] = 0.02
    contact[80:100] = True
    rng = np.random.get_state()
    np.random.seed(23)
    indices, stats = sample(states, actions, next_states, new, contact, updates=2)
    first_rng = np.random.get_state()
    np.random.seed(23)
    repeated, repeated_stats = sample(states, actions, next_states, new, contact, updates=2)
    assert np.array_equal(indices, repeated)
    assert stats == repeated_stats
    assert np.array_equal(first_rng[1], np.random.get_state()[1])
    np.random.set_state(rng)

    assert indices.shape == (80,)
    assert np.issubdtype(indices.dtype, np.integer)
    assert np.all((indices >= 0) & (indices < len(states)))
    assert stats["source_counts"] == {"new": 40, "history": 40}
    assert stats["historical_stratum_counts"] == {"dynamic": 20, "contact": 10, "static": 10}
    assert stats["num_samples"] == 2 * 40
    json.dumps(stats, allow_nan=False)


def test_absent_stratum_weights_redistribute_proportionally():
    states, actions, next_states, new, contact = replay_data(80)
    next_states[:48, 10] = 0.02  # movement of the second object is dynamic.
    contact[48:] = True
    indices, stats = sample(states, actions, next_states, new, contact, batch_size=40)
    assert len(indices) == 40
    assert stats["source_counts"] == {"new": 0, "history": 40}
    # Renormalized 50:25 dynamic/contact weights, largest-remainder rounding.
    assert stats["historical_stratum_counts"] == {"dynamic": 27, "contact": 13, "static": 0}


def test_bootstrap_without_history_and_history_without_new_data():
    states, actions, next_states, new, contact = replay_data(60)
    new[:] = True
    indices, stats = sample(states, actions, next_states, new, contact, updates=3, batch_size=7)
    assert len(indices) == 21 and stats["source_counts"] == {"new": 21, "history": 0}
    new[:] = False
    indices, stats = sample(states, actions, next_states, new, contact, updates=3, batch_size=7)
    assert len(indices) == 21 and stats["source_counts"] == {"new": 0, "history": 21}


def test_diversity_bins_prevent_large_duplicate_bin_from_dominating():
    states, actions, next_states, new, contact = replay_data(1000)
    new[:] = True
    states[900:, 0] = np.pi / 36  # exactly one 5-degree joint bin above the others.
    np.random.seed(91)
    indices, stats = sample(states, actions, next_states, new, contact, batch_size=2000)
    selected_rare_bin = np.count_nonzero(states[indices, 0] != 0)
    assert stats["occupied_bins"]["new"] == 2
    assert 900 < selected_rare_bin < 1100


@pytest.mark.parametrize("case, message", [
    ("rank", "rank-2"),
    ("mismatch", "matching record counts"),
    ("short_state", "do not contain"),
    ("mask", "new_mask must be a boolean vector"),
    ("nonfinite", "states must contain only finite"),
    ("action", "at least one categorical"),
])
def test_invalid_replay_inputs_fail_clearly(case, message):
    states, actions, next_states, new, contact = replay_data(4)
    if case == "rank":
        states = states[0]
    elif case == "mismatch":
        next_states = next_states[:3]
    elif case == "short_state":
        states, next_states = states[:, :8], next_states[:, :8]
    elif case == "mask":
        new = np.zeros(4, dtype=np.int8)
    elif case == "nonfinite":
        states[0, 0] = np.nan
    elif case == "action":
        actions = actions[:, -1:]
    with pytest.raises(ValueError, match=message):
        sample(states, actions, next_states, new, contact, batch_size=2)


def test_dynamic_threshold_includes_exact_boundary_and_overrides_contact():
    states, actions, next_states, new, contact = replay_data(12)
    next_states[:4, 10] = 0.015  # second object's xyz displacement meets threshold exactly
    contact[:4] = True
    indices, stats = sample(states, actions, next_states, new, contact, updates=2, batch_size=8)
    assert stats["historical_stratum_counts"] == {"dynamic": 10, "contact": 0, "static": 6}
    assert len(indices) == 16
