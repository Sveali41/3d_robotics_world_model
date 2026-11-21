"""Budget and reporting rules for the bounded grasp optimization proxy."""
import numpy as np
import pytest

from exploration.diagnostics.benchmark_grasp_optimizations import (
    PROFILES, compare_tree, profile_order, replay_rows, validate_budget,
)


@pytest.mark.parametrize('repeats,ticks', [(0, 32), (4, 32), (3, 101), (1, 301),
                                         (3, 0), (True, 32), (3, True), (3., 32)])
def test_rejects_budget_that_is_not_positive_bounded_integer(repeats, ticks):
    with pytest.raises(ValueError, match='300 per profile'):
        validate_budget(repeats, ticks)


def test_default_and_boundary_include_all_physical_ticks_without_warmup():
    assert validate_budget(3, 32) == 96
    assert validate_budget(3, 100) == 300
    assert validate_budget(1, 300) == 300


def test_profile_orders_are_interleaved_and_reference_moves_to_middle():
    assert profile_order(0) == PROFILES
    assert profile_order(1) == PROFILES[::-1]
    assert profile_order(2)[2] == 'reference'
    assert all(set(profile_order(i)) == set(PROFILES) for i in range(3))


def test_tree_comparison_checks_optimizer_reports_rng_and_tolerance():
    before = dict(model=np.array([1.], np.float32),
                  optimizer={'state': {0: {'step': 100, 'exp_avg': np.zeros(2)}}},
                  reports={'loss': .25}, rng=('MT19937', np.array([1, 2], np.uint32), 4))
    after = dict(model=np.array([1. + 1e-6], np.float32),
                 optimizer={'state': {0: {'step': 100, 'exp_avg': np.zeros(2)}}},
                 reports={'loss': .25 + 1e-7}, rng=('MT19937', np.array([1, 2], np.uint32), 4))
    assert compare_tree(before, after) == []
    assert compare_tree({'rng': np.array([1.])}, {'rng': np.array([1. + 1e-7])}) == ['root.rng']
    assert compare_tree({'rng': 1.}, {'rng': 1. + 1e-7}) == ['root.rng']
    after['rng'] = ('MT19937', np.array([1, 3], np.uint32), 4)
    assert compare_tree(before, after) == ['root.rng[1]']
    after['optimizer']['state'][0]['step'] = 101
    assert 'root.optimizer.state.0.step' in compare_tree(before, after)


def test_proxy_save_rows_preserve_fixed_replay_fields():
    arrays = dict(states=np.zeros((2, 3)), next_states=np.ones((2, 3)),
                  actions=np.zeros((2, 4)), episode_ids=np.array([1, 2]),
                  terminated=np.array([False, True]), truncated=np.array([True, False]),
                  duration_low_steps=np.array([5, 6]))
    rows = replay_rows(arrays)
    assert len(rows) == 2
    assert rows[0]['trainable'] is True
    assert rows[1]['duration_low_steps'] == 6
    assert rows[1]['terminated'] is True
    np.testing.assert_array_equal(rows[0]['action'], arrays['actions'][0])
