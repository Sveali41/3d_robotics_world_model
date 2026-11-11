import json

import numpy as np
import pytest

from exploration.diagnostics import visualize_cabinet_grasp_sequences as v


def fixture_case(root, case_id=0, obj='cup'):
    case = dict(case_id=case_id, object=obj, template='xyz')
    directory = root/'cases'/f'case_{case_id:03d}'
    attempt = directory/'attempt_000001'
    attempt.mkdir(parents=True)
    initial = np.zeros(67)
    initial[18:21] = [1,2,3]
    initial[33:36] = [4,5,6]
    initial[45:48] = [7,8,9]
    following = initial.copy(); following[18:21] = [4,6,3]; following[33] += 2; following[45] += 3
    (attempt/'skills.jsonl').write_text(json.dumps(dict(state=initial.tolist()))+'\n')
    (attempt/'transitions.jsonl').write_text(json.dumps(dict(state=initial.tolist(), next_state=following.tolist(), episode_id=1, step=1))+'\n')
    trace = [dict(step=1, phase='lift', target_grasped=True, stable_grasp_ticks=4, closed_command=True)]
    (attempt/'phase_trace.json').write_text(json.dumps(trace))
    report = dict(ever_confirmed_grasp=True, max_stable_grasp_ticks=6, held_lift_verified=True,
                  max_root_rise_while_confirmed_held_m=.03, max_continuous_grasp_rise_m=.02,
                  carry_displacement_m=.01, released_after_drop=False, failure='lost_grasp_during_carry',
                  commands=[dict(phase='lift', primitive='lift', status='timeout'),
                            dict(phase='park', primitive='open_gripper', status='requested')])
    summary = dict(case=case, attempt=1, control_ticks=1, success=False, report=report)
    (directory/'summary.json').write_text(json.dumps(summary))
    return case, directory/'summary.json'


def test_indices_initial_and_full_path_lengths(tmp_path):
    case, path = fixture_case(tmp_path)
    record = v.read_case(case, path)
    assert record['outcome'] == 'grasp_then_failed'
    assert record['tcp'].tolist() == [[1,2,3],[4,6,3]]
    assert record['target'].tolist() == [[4,5,6],[6,5,6]]
    assert record['tcp_path_length_m'] == 5
    assert record['target_path_length_m'] == 2
    case2, path2 = fixture_case(tmp_path, 1, 'bowl')
    assert v.read_case(case2, path2)['target'].tolist() == [[7,8,9],[10,8,9]]


def test_downsample_keeps_endpoints_and_original_lengths():
    points = np.column_stack((np.arange(500), np.zeros(500), np.zeros(500)))
    record = dict(tcp=points, target=points, initial_cup=points[0], initial_bowl=points[0], tcp_path_length_m=499.)
    display = v.display_case(record)
    assert len(display['tcp']) == 128
    assert display['tcp'][0] == points[0].tolist()
    assert display['tcp'][-1] == points[-1].tolist()
    assert display['tcp_path_length_m'] == 499.


def test_snapshot_freezes_closed_list_excludes_unfinished(tmp_path):
    case, path = fixture_case(tmp_path)
    other = dict(case_id=1, object='bowl', template='xyz')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(protocol='paired_axis_order_and_grasp_insertion_v1', cases=[case,other])))
    manifest, frozen = v.freeze_closed(tmp_path)
    fixture_case(tmp_path, 1, 'bowl')
    assert len(frozen) == 1 and frozen[0][1] == path
    assert len(manifest['cases']) == 2
    assert v.read_case(*frozen[0])['case_id'] == 0


def test_discontinuous_episode_is_rejected(tmp_path):
    case, path = fixture_case(tmp_path)
    transition = path.parent/'attempt_000001'/'transitions.jsonl'
    row = json.loads(transition.read_text()); row['episode_id'] = 2
    transition.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='discontinuous'): v.read_case(case, path)


def test_inline_bundle_and_html_escaping(tmp_path):
    html_source = tmp_path/'source.html'
    html_source.write_text('<script>other</script><script>/** plotly.js v2.12.1 License */\nconst valid=true;</script>')
    bundle = v.plotly_bundle(html_source)
    assert 'License' in bundle
    assert '</script' not in v.json_script({'name':'</script><script>alert(1)</script>'}).lower()
    record = dict(case_id=0, object='cup', template='xyz', outcome='no_grasp',
                  tcp=np.zeros((1,3)), target=np.zeros((1,3)), initial_cup=np.zeros(3), initial_bowl=np.zeros(3))
    output = tmp_path/'view.html'
    v.write_html(output, bundle, [record], dict(captured_at_london='2026', unfinished=1, density=dict(cell_size_m=.02)))
    content = output.read_text()
    assert 'plotly.js v2.12.1' in content
    assert 'Plotly.react' in content and 'const DATA=' in content
    assert 'https://cdn' not in content


def test_generate_files_no_simulation(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    case, _ = fixture_case(source)
    (source/'manifest.json').write_text(json.dumps(dict(protocol='paired_axis_order_and_grasp_insertion_v1', cases=[case])))
    bundle = tmp_path/'bundle.html'; bundle.write_text('<script>/** plotly.js v2.12.1 */</script>')
    out = tmp_path/'out'
    assert v.main(['--input',str(source),'--output',str(out),'--plotly-source',str(bundle)]) == 0
    meta = json.loads((out/'snapshot.json').read_text())
    assert meta['closed'] == 1 and meta['unfinished'] == 0 and meta['raw_ticks'] == 1
    assert (out/'trajectories.html').stat().st_size > 0
    assert (out/'overview.png').stat().st_size > 0


def test_density_floor_unique_visits_and_case_counts():
    first = dict(tcp=np.array([[-.001, 0., 0.], [-.02, 0., 0.], [-.001, 0., 0.], [.02, 0., 0.]]),
                 target=np.zeros((1,3)))
    second = dict(tcp=np.array([[-.001, 0., 0.], [-.001, 0., 0.]]), target=np.zeros((1,3)))
    for record in (first, second): v.add_voxels(record, .02)
    assert first['tcp_voxels'] == [[-1, 0, 0], [1, 0, 0]]
    counts = v.voxel_visits([first, second], 'tcp')
    assert counts[(-1, 0, 0)] == 2
    assert counts[(1, 0, 0)] == 1


def test_density_uses_full_resolution_not_display_samples():
    points = np.zeros((300,3)); points[1,0] = .1
    record = dict(tcp=points, target=points, initial_cup=points[0], initial_bowl=points[0])
    v.add_voxels(record, .02)
    display = v.display_case(record)
    assert [5,0,0] in display['tcp_voxels']
    assert [.1,0.,0.] not in display['tcp']


@pytest.mark.parametrize('size', ['0', '-.01', 'nan', 'inf'])
def test_density_cli_rejects_invalid_size(tmp_path, size):
    with pytest.raises(ValueError, match='finite and positive'):
        v.main(['--input',str(tmp_path), '--output',str(tmp_path/'out'), '--voxel-size',size])


def test_density_and_paths_controls_and_metadata(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    case, _ = fixture_case(source)
    (source/'manifest.json').write_text(json.dumps(dict(protocol='paired_axis_order_and_grasp_insertion_v1', cases=[case])))
    bundle = tmp_path/'bundle.html'; bundle.write_text('<script>/** plotly.js v2.12.1 */</script>')
    out = tmp_path/'out'
    v.main(['--input',str(source), '--output',str(out), '--plotly-source',str(bundle)])
    html = (out/'trajectories.html').read_text()
    assert 'id="view"' in html and 'id="density_kind"' in html
    assert 'value="density"' in html and 'value="paths"' in html
    assert 'gaps between samples not filled' in html
    meta = json.loads((out/'snapshot.json').read_text())
    assert meta['density']['origin_m'] == [0,0,0]
    assert meta['density']['cell_size_m'] == .02
    assert meta['density']['max_cases_per_cell']['tcp'] == 1
    assert (out/'density.png').stat().st_size > 0


def test_metrics_and_trace_feedback_preserve_grasp_then_failure(tmp_path):
    case, path = fixture_case(tmp_path)
    record = v.read_case(case, path)
    assert record['metrics'] == dict(ever_confirmed_grasp=True, max_stable_grasp_ticks=6,
        held_lift_verified=True, max_root_rise_while_confirmed_held_m=.03,
        max_continuous_grasp_rise_m=.02, carry_displacement_m=.01, released_after_drop=False,
        failure='lost_grasp_during_carry', calibration_success=False,
        primitive_statuses=[dict(phase='lift', primitive='lift', status='timeout')])
    assert record['step'].tolist() == [0, 1]
    assert record['phase'].tolist() == ['reset', 'lift']
    assert record['confirmed_grasp'].tolist() == [False, True]
    assert record['target_height_relative_initial_m'].tolist() == [0., 0.]


def test_trace_mismatch_and_missing_fail_clearly(tmp_path):
    case, path = fixture_case(tmp_path)
    trace = path.parent/'attempt_000001'/'phase_trace.json'
    trace.write_text('[]')
    with pytest.raises(ValueError, match='trace step count'): v.read_case(case, path)
    trace.unlink()
    with pytest.raises(ValueError, match='Missing phase trace'): v.read_case(case, path)


def test_downsampling_aligns_all_feedback_arrays():
    n = 300
    pts = np.column_stack((np.arange(n), np.zeros(n), np.zeros(n)))
    record = dict(tcp=pts, target=pts, step=np.arange(n), phase=np.asarray([str(i) for i in range(n)]),
                  target_grasped=np.arange(n)%2==0, stable_grasp_ticks=np.arange(n),
                  closed_command=np.arange(n)%2==0, confirmed_grasp=np.arange(n)%2==0,
                  target_height_relative_initial_m=np.arange(n), initial_cup=pts[0], initial_bowl=pts[0])
    display = v.display_case(record)
    assert len(display['step']) == 128
    for i, step in enumerate(display['step']):
        assert display['tcp'][i][0] == step
        assert display['phase'][i] == str(step)
        assert display['confirmed_grasp'][i] == (step%2==0)
        assert display['stable_grasp_ticks'][i] == step
    assert display['step'][0] == 0 and display['step'][-1] == n-1


def test_feedback_ui_has_step_detail_and_no_threshold_reinterpretation(tmp_path):
    html = v.HTML
    for identifier in ('step', 'step_label', 'grasp_counts', 'case_metrics', 'case_motion', 'case_primitives'):
        assert f'id="{identifier}"' in html
    assert 'target height change' in html and 'reset: no grasp feedback' in html
    assert 'm.ever_confirmed_grasp' in html and 'm.held_lift_verified' in html
    assert 'm.primitive_statuses' in html and 'm.failure' in html
    assert "join(String.fromCharCode(10))" in html
    assert "textContent='Primitive sequence / status\n'" not in html


def test_representative_robot_movies_are_embedded_offline(tmp_path):
    (tmp_path/'case_002.mp4').write_bytes(b'example-movie')
    (tmp_path/'replay_manifest.json').write_text(json.dumps({'cases': [dict(
        case_id=2, object='cup', template='yxz', success=True, failure=None,
        video='case_002.mp4'), dict(case_id=542, object='bowl', template='yxz',
        success=True, failure=None, video='case_002.mp4')]}))
    panel = v.replay_panel(tmp_path)
    assert 'Watch what the robot actually did' in panel
    assert 'data:video/mp4;base64,ZXhhbXBsZS1tb3ZpZQ==' in panel
    assert 'controls playsinline' in panel and 'replay_speed' in panel
    assert 'Complete pipeline' in panel
    assert '<h2>Cup</h2>' in panel and '<h2>Bowl</h2>' in panel
    assert panel.count('<video ') == 2
    assert v.replay_panel(None) == ''


def test_html_preserves_every_recorded_path_point(tmp_path):
    import re
    points = np.column_stack((np.arange(501)/1000, np.zeros(501), np.ones(501)))
    record = dict(tcp=points, target=points.copy(), step=np.arange(501),
                  phase=np.array(['reset']+['lift']*500))
    path = tmp_path/'paths.html'
    v.write_html(path, '/** plotly.js v2.12.1 */', [record],
                 dict(captured_at_london='2026', unfinished=0, density=dict(cell_size_m=.02)))
    data = json.loads(re.search(r'const DATA=(.*?); const COLORS=', path.read_text()).group(1))
    assert data['cases'][0]['step'] == list(range(501))
    assert data['cases'][0]['tcp'] == points.tolist()
    assert len(data['cases'][0]['phase']) == 501
