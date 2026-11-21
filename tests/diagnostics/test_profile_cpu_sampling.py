import pytest

from exploration.diagnostics.profile_cpu_sampling import (
    canonical_job_results,
    compare_job_transitions,
)


def test_job_results_are_canonicalized_and_duplicate_jobs_rejected():
    assert [item['job_id'] for item in canonical_job_results(
        [{'job_id': 1}, {'job_id': 0}])] == [0, 1]
    with pytest.raises(ValueError, match='Duplicate job_id'):
        canonical_job_results([{'job_id': 0}, {'job_id': 0}])


def test_transition_comparison_checks_exact_physical_values_and_lengths():
    left = [dict(action=[0.1], full_controller_action=[0.2], state=[1.], next_state=[2.],
                 before={'pose': [3.]}, after={'pose': [4.]}, macro_phase='approach',
                 macro_status='running', parameters_physical={'x': 0.4})]
    right = [dict(left[0])]
    assert compare_job_transitions(left, right) == []
    right[0]['parameters_physical'] = {'x': 0.40000000001}
    assert compare_job_transitions(left, right) == ['transitions[0].parameters_physical.x']
    assert compare_job_transitions(left, []) == ['transitions.length: 1 != 0']
