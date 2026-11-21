import numpy as np
import pytest
import torch

from exploration.diagnostics.validate_cpu_sampling import metrics, require_equal, scripted_collection
from exploration.raps import cpu_sampling


def test_exact_comparison_accepts_tensor_snapshots_and_rejects_changes():
    snapshot = {'x': torch.tensor([1., 2.]), 'rng': (np.array([1, 2]),)}
    require_equal(snapshot, {'x': torch.tensor([1., 2.]), 'rng': (np.array([1, 2]),)}, 'snapshot')
    with pytest.raises(AssertionError, match='snapshot differs'):
        require_equal(snapshot, {'x': torch.tensor([1., 3.]), 'rng': (np.array([1, 2]),)}, 'snapshot')


def test_script_patch_restored_on_error():
    original = cpu_sampling._run_shard
    with pytest.raises(RuntimeError):
        with scripted_collection():
            assert cpu_sampling._run_shard is not original
            raise RuntimeError('failure')
    assert cpu_sampling._run_shard is original


def test_status_metrics_do_not_count_contact_as_collision():
    skills = [dict(status='success', completed=True, trainable=True,
                   motion_planning=[dict(status='success', fingertip_contact=True)]),
              dict(status='budget_truncated', completed=False, trainable=False,
                   motion_planning=[dict(status='collision_rejected')])]
    result = metrics([{}, {}], skills)
    assert result['controls'] == 2
    assert result['successes'] == result['budget_truncated'] == result['planning_collision_rejections'] == 1
    assert result['completed_failures'] == 0
