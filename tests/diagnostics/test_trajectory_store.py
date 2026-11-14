import json
import math

import numpy as np
import pytest

from exploration.diagnostics.trajectory_store import TrajectoryWriter, read_transitions, transition_source


def attempt(root, case_id=3, attempt_id=1):
    path = root/'cases'/f'case_{case_id:03d}'/f'attempt_{attempt_id:06d}'
    path.mkdir(parents=True, exist_ok=True)
    (path/'attempt.json').write_text(json.dumps(dict(case=dict(case_id=case_id), attempt=attempt_id)))
    return path


def row(step, case_id=3, attempt_id=1):
    return dict(case_id=case_id, attempt_id=attempt_id, step=step, episode_id=1,
                state=[float(i)/7 for i in range(67)], next_state=[float(i)/3 for i in range(67)],
                action=[-0.0, 1., 0., .3], full_controller_action=[0.]*7,
                before=dict(grasped=False, contacts=[]),
                after=dict(grasped=True, contacts=[dict(name='cup☕', force=[i, float(i), None])
                                                 for i in range(step % 3)]),
                outcome=dict(reward=.1, done=False, timeout=None),
                mixed=[True, 1, 1., None, '', [], {}, [1, [2.]]], big=2**80)


def assert_same(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_same(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_same(a, b)
    else:
        assert actual == expected
        if isinstance(actual, float):
            assert math.copysign(1., actual) == math.copysign(1., expected)


def test_roundtrip_batched_ragged_and_chunk_boundaries(tmp_path):
    first, second = attempt(tmp_path), attempt(tmp_path, 4)
    expected = [row(i) for i in range(1, 6)]
    others = [row(i, case_id=4) for i in range(1, 6)]
    with TrajectoryWriter(tmp_path, chunk_size=2, queue_size=1) as writer:
        for a, b in zip(expected, others):
            writer.append([a, b])
        writer.flush()
        assert_same(read_transitions(first), expected)
        assert_same(read_transitions(second), others)
    index = json.loads((tmp_path/'trajectory_chunks'/'index.json').read_text())
    assert [c['rows'] for c in index['chunks']] == [4, 4, 2]
    assert [c['batches'] for c in index['chunks']] == [2, 2, 1]
    for path in (tmp_path/'trajectory_chunks').rglob('*.npy'):
        assert np.load(path, allow_pickle=False).dtype.kind in 'biuf'
    assert not list(tmp_path.rglob('transitions.jsonl'))


def test_empty_and_resume_ignores_unpublished_chunks(tmp_path):
    path = attempt(tmp_path)
    with TrajectoryWriter(tmp_path, chunk_size=2) as writer:
        writer.flush()
        assert read_transitions(path) == []
        writer.append([row(1)])
    directory = tmp_path/'trajectory_chunks'
    before = (directory/'chunk_00000000'/'schema.json').read_bytes()
    (directory/'chunk_00000001').mkdir()  # Renamed chunk interrupted before index publication.
    (directory/'chunk_00000002.partial').mkdir()
    (directory/'index.tmp').write_text('interrupted')
    assert_same(read_transitions(path), [row(1)])
    with TrajectoryWriter(tmp_path, chunk_size=1) as writer:
        writer.append([row(2)])
    assert_same(read_transitions(path), [row(1), row(2)])
    assert (directory/'chunk_00000003').exists()
    assert (directory/'chunk_00000000'/'schema.json').read_bytes() == before


def test_append_owns_batch_and_preserves_missing_fields(tmp_path):
    path = attempt(tmp_path)
    values = [row(1), dict(case_id=3, attempt_id=1, optional=None), dict(case_id=3, attempt_id=1)]
    expected = json.loads(json.dumps(values))
    with TrajectoryWriter(tmp_path, chunk_size=3) as writer:
        writer.append(values)
        values[0]['state'][0] = 999.
    assert_same(read_transitions(path), expected)


def test_historical_and_local_store_detection(tmp_path):
    path = attempt(tmp_path)
    values = [row(1)]
    historical = path/'transitions.jsonl'
    historical.write_text(json.dumps(values[0])+'\n\n')
    assert_same(read_transitions(path), values)
    assert transition_source(path) == historical
    historical.unlink()
    with TrajectoryWriter(path) as writer:
        writer.append(values)
    assert_same(read_transitions(path), values)


def test_writer_error_surfaces_and_close_cleans_up(tmp_path):
    writer = TrajectoryWriter(tmp_path, chunk_size=1, queue_size=1)
    writer.append([dict(case_id=3, attempt_id=1, unsupported=object())])
    with pytest.raises(RuntimeError, match='JSON types'):
        writer.flush()
    with pytest.raises(RuntimeError, match='Trajectory writer failed'):
        writer.close()
    assert not writer._process.is_alive()
    assert json.loads((tmp_path/'trajectory_chunks'/'index.json').read_text())['chunks'] == []


def test_dead_writer_cannot_hang_full_queue(tmp_path):
    writer = TrajectoryWriter(tmp_path, queue_size=1)
    writer._process.terminate()
    writer._process.join()
    with pytest.raises(RuntimeError, match='Writer exited'):
        writer.append([row(1)])
    with pytest.raises(RuntimeError):
        writer.close()
    assert not writer._process.is_alive()


def test_invalid_route_surfaces_at_flush(tmp_path):
    writer = TrajectoryWriter(tmp_path)
    writer.append([dict(step=1)])
    with pytest.raises(RuntimeError, match='routing metadata'):
        writer.flush()
    with pytest.raises(RuntimeError):
        writer.close()


def test_missing_committed_chunk_fails_clearly(tmp_path):
    path = attempt(tmp_path)
    with TrajectoryWriter(tmp_path) as writer:
        writer.append([row(1)])
    (tmp_path/'trajectory_chunks'/'chunk_00000000'/'schema.json').unlink()
    with pytest.raises(FileNotFoundError):
        read_transitions(path)


def test_corrupt_index_reports_initialization_failure(tmp_path):
    directory = tmp_path/'trajectory_chunks'
    directory.mkdir()
    (directory/'index.json').write_text('{interrupted')
    with pytest.raises(RuntimeError, match='Trajectory writer failed'):
        TrajectoryWriter(tmp_path)


def test_bounded_queue_blocks_until_writer_consumes(tmp_path):
    import os
    import signal
    import threading

    writer = TrajectoryWriter(tmp_path, queue_size=1)
    completed = threading.Event()
    failure = []
    os.kill(writer._process.pid, signal.SIGSTOP)
    os.waitpid(writer._process.pid, os.WUNTRACED)
    try:
        writer.append([row(1)])

        def append():
            try:
                writer.append([row(2)])
            except BaseException as error:
                failure.append(error)
            finally:
                completed.set()

        thread = threading.Thread(target=append)
        thread.start()
        assert not completed.wait(.2)
    finally:
        os.kill(writer._process.pid, signal.SIGCONT)
    assert completed.wait(10)
    thread.join()
    assert not failure
    writer.close()
    assert len(read_transitions(tmp_path)) == 2


def test_gpu_replay_fails_before_building_cpu_environment(tmp_path, monkeypatch):
    from exploration.diagnostics import render_cabinet_grasp_replays as replay

    monkeypatch.setattr(replay, 'read_case', lambda *args: (
        dict(seed=1), {}, dict(environment=dict(sim_backend='gpu')), [], []))
    monkeypatch.setattr(replay, 'build_env', lambda *args: pytest.fail('Backend must be checked before building environment'))
    with pytest.raises(ValueError, match='GPU trajectory replay is unsupported'):
        replay.render_case(tmp_path, tmp_path, {}, 3)


@pytest.mark.parametrize('values, dense', [
    ([[1, 2], [3, 4]], True),
    ([[False, True], [True, False]], True),
    ([[[.1, -0.0]], [[.3, 1.]]], True),
    ([[1, 1.], [True, 2]], False),
    ([[1.], [2., 3.]], False),
    ([[2**80], [1]], False),
    ([[], []], True),
])
def test_numeric_fast_path_preserves_leaf_types_and_shape(tmp_path, values, dense):
    from exploration.diagnostics.trajectory_store import _Columns, _decode

    columns = _Columns(tmp_path)
    node = columns.encode(values)
    assert ('dense' in node) == dense
    columns.publish()
    assert_same(_decode(tmp_path, node), values)
    assert len(list(tmp_path.glob('*.npy'))) == 1
    assert np.load(tmp_path/'columns.npy', allow_pickle=False).dtype == np.uint8


def test_representative_complete_rows_pool_small_fields(tmp_path):
    from exploration.diagnostics.trajectory_store import _Columns, _decode

    rows = []
    for step in range(1, 12):
        value = row(step)
        objects = {name: dict(position_root=[.1, .2, .3], position_world=[.5, .6, .7])
                   for name in ('cabinet_item_16', 'cabinet_item_17')}
        contacts = dict(objects={name: dict(left_force=.1, right_force=.2, grasped=True) for name in objects})
        for key in ('before', 'after'):
            value[key] = dict(ee_position=[.1, .2, .3], objects=objects, contact_snapshot=contacts,
                              inside_region=True, failed_object_ids=[])
        value['contact_events'] = [contacts]
        value['outcome_metrics'] = dict(initial_object_positions_root={name: obj['position_root'] for name, obj in objects.items()},
            final_object_positions_root={name: obj['position_root'] for name, obj in objects.items()},
            held_object_ids=list(objects), stable_grasp_frames={name: step for name in objects},
            lifted_object_ids=list(objects) if step >= 4 else [], object_lift_displacements={name: .02 for name in objects})
        rows.append(value)
    columns = _Columns(tmp_path)
    node = columns.encode(rows)
    columns.publish()
    assert_same(_decode(tmp_path, node), rows)
    # Prior layout created one durable file per logical column (675 on native rows).
    assert len(list(tmp_path.glob('*.npy'))) == 1 < columns.number
    assert (tmp_path/'column_index.json').is_file()


def test_prior_numeric_chunk_schema_remains_readable(tmp_path):
    from exploration.diagnostics.trajectory_store import _decode

    def save(name, values, dtype):
        np.save(tmp_path/name, np.asarray(values, dtype=dtype), allow_pickle=False)
        return name

    # Historical root dictionary -> ragged list -> scalar float field schema.
    floats = dict(tags=save('types.npy', [3, 3, 3], np.uint8),
                  groups={'3': dict(data=save('values.npy', [.1, -0., .3], np.float64))})
    lists = dict(tags=save('lists.npy', [5, 5], np.uint8), groups={
        '5': dict(offsets=save('offsets.npy', [0, 2, 3], np.int64), items=floats)})
    field = dict(positions=save('positions.npy', [0, 1], np.int64), values=lists)
    node = dict(tags=save('root.npy', [6, 6], np.uint8), groups={'6': dict(fields={'state': field})})
    assert_same(_decode(tmp_path, node), [dict(state=[.1, -0.]), dict(state=[.3])])
