import pytest
from exploration.diagnostics.benchmark_split7_collection import benchmark_cases, recommend
from exploration.diagnostics import collect_cabinet_grasp_sequences as collection


def test_fixed_pool_balanced_unique_and_seeded():
    cases = benchmark_cases(1000000)
    assert len(cases) == len({c['case_id'] for c in cases}) == 128
    assert [c['case_id'] for c in cases] == sorted(c['case_id'] for c in cases)
    assert {c['template'] for c in cases} == set(collection.TEMPLATES)
    assert sum(c['object'] == 'cup' for c in cases) == 64
    assert all(c['seed'] == 1000000+c['base_case_id'] for c in cases)
    assert cases == benchmark_cases(1000000)


def measurement(n, speed, memory=1000, status='complete'):
    return dict(collector='gpu_batch', num_envs=n, end_to_end_ticks_per_second=speed,
                peak_gpu_memory_mib=memory, active_control_ticks=4096, status=status)


def test_select_near_fastest_smaller_with_memory_headroom():
    values = [measurement(16, 94), measurement(32, 96), measurement(64, 100),
              measurement(128, 200, memory=9000)]
    assert recommend(values, 10000) == 32
    assert recommend([measurement(32, 100, status='failed')], 10000) is None
    assert recommend([measurement(32, 100, memory=None)], 10000) is None


@pytest.mark.parametrize('options', [
    ['--collector', 'gpu_batch'],
    ['--collector', 'gpu_batch', '--skill-set', 'split7'],
    ['--collector', 'gpu_batch', '--skill-set', 'split7', '--num-envs', '0'],
    ['--num-envs', '2'], ['--export-every-cases', '0'],
])
def test_invalid_collection_options_fail_before_checkpoint_or_directory(tmp_path, options):
    output = tmp_path/'data'
    with pytest.raises(SystemExit) as error:
        collection.main(['--checkpoint', '/missing.pt', '--output', str(output), *options])
    assert error.value.code == 2
    assert not output.exists()


def test_gpu_plan_has_original_case_protocol(tmp_path):
    output = tmp_path/'plan'
    assert collection.main(['--checkpoint', '/unused.pt', '--output', str(output),
                            '--stage', 'plan', '--skill-set', 'split7',
                            '--collector', 'gpu_batch', '--num-envs', '32']) == 0
    import json
    plan = json.loads((output/'plan.json').read_text())
    assert plan['cases'] == collection.cases()
    assert plan['num_envs'] == 32 and plan['action_dim'] == 13
    assert plan['max_control_ticks_per_case'] == 800


def test_cached_summary_report_matches_disk_without_rescanning(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    cases = collection.selected_cases('smoke', 1000000)[:2]
    summaries = [dict(case=c, success=i == 0, status='success' if i == 0 else 'failure',
                      report={'failure': None if i == 0 else 'grasp_not_confirmed'})
                 for i, c in enumerate(cases)]
    for summary in summaries:
        directory = tmp_path/'cases'/f"case_{summary['case']['case_id']:03d}"
        directory.mkdir(parents=True)
        (directory/'summary.json').write_text(json.dumps(summary))
    expected = collection.results(tmp_path, cases)
    def unexpected_read(*args, **kwargs):
        raise AssertionError('Cached report must not rescan summary files')
    monkeypatch.setattr(Path, 'read_text', unexpected_read)
    assert collection.results(tmp_path, cases, summaries=summaries) == expected
