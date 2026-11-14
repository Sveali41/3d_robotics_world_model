"""Build offline full-control trajectories from stopped runs; no physics/inference."""
from pathlib import Path
from collections import Counter
import base64
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exploration.diagnostics.visualize_cabinet_grasp_sequences import plotly_bundle, json_script
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'logs/raps_pick_place_four_baselines_20261113_1626'
ARMS = ['curiosity_heuristic', 'curiosity_only', 'heuristic_only', 'random']
COLUMNS = [18, 19, 20, 33, 34, 35, 45, 46, 47]


def reverse_lines(path):
    """Read only the interrupted tail, rather than multi-GB transition histories."""
    with path.open('rb') as stream:
        offset = stream.seek(0, 2)
        pending = b''
        while offset:
            size = min(offset, 1024 * 1024)
            offset -= size
            stream.seek(offset)
            lines = (stream.read(size) + pending).split(b'\n')
            pending = lines[0]
            for line in reversed(lines[1:]):
                if line.strip():
                    yield line
        if pending.strip():
            yield pending


def encode(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode('ascii')


def build_arm(arm):
    run = SOURCE / arm / 'run'
    online = run / ('random' if arm == 'random' else 'curiosity')
    checkpoint = max(online.glob('round_*.low_level.npz'))
    with np.load(checkpoint) as archive:
        # NPZ contains the cumulative bootstrap + online real control history.
        states = archive['states']
        after = archive['next_states']
        episodes = archive['episode_ids'].astype('<i4')
        count = len(states)
        assert after.shape == states.shape == (count, 67)
        positions = np.concatenate([states[:1, COLUMNS], after[:, COLUMNS]])
    tail = []
    ignored_partial_tail_lines = 0
    for line in reverse_lines(online / 'transitions.jsonl'):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if tail:
                raise
            ignored_partial_tail_lines += 1
            continue
        if row['step'] <= count:
            break
        tail.append(row)
    tail.reverse()
    assert [r['step'] for r in tail] == list(range(count + 1, count + len(tail) + 1))
    if tail:
        positions = np.concatenate([positions, np.asarray([r['next_state'] for r in tail], dtype=np.float32)[:, COLUMNS]])
        episodes = np.r_[episodes, np.asarray([r['episode_id'] for r in tail], dtype='<i4')]
    n = len(episodes)
    assert positions.shape == (n + 1, 9) and np.isfinite(positions).all()
    skills = []
    statuses = Counter()
    success = Counter()
    for path in [run / 'bootstrap/skills.jsonl', online / 'skills.jsonl']:
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    # SIGTERM can interrupt the final appended record.
                    assert not stream.read().strip(), 'Malformed interior skill record'
                    continue
                start, end = row['start_step'], row['end_step']
                assert 0 <= start <= end <= n
                assert end - start == row['duration_low_steps']
                skills.append([start, end, row['primitive_type'], row['status'], row['episode_id'], row['skill_id']])
                statuses[row['status']] += 1
                if row['status'] == 'success':
                    success[row['primitive_type']] += 1
    assert all(a[1] == b[0] for a, b in zip(skills, skills[1:])), 'Skill ranges must be contiguous'
    assert skills[0][0] == 0
    last_end = skills[-1][1]
    # An interrupted macro has ticks but no completed skill record. Preserve it.
    if last_end < n:
        assert tail and tail[-1]['skill_id'] > skills[-1][5]
        r = tail[-1]
        skills.append([last_end, n, r['skill_type'], 'interrupted_unfinished', r['episode_id'], r['skill_id']])
        statuses['interrupted_unfinished'] += 1
    assert skills[-1][1] == n
    summary = dict(actual_controls=n, saved_controls=count, recovered_tail_controls=len(tail),
                   recorded_macro_ranges=len(skills), statuses=dict(statuses), success=dict(success),
                   episodes=int(len(np.unique(episodes))), source_npz=str(checkpoint.relative_to(ROOT)),
                   source_npz_bytes=checkpoint.stat().st_size,
                   transition_jsonl_bytes=(online/'transitions.jsonl').stat().st_size,
                   ignored_partial_tail_lines=ignored_partial_tail_lines,
                   includes_bootstrap=True, includes_interrupted_tail=True,
                   coordinate_frame='robot root', position_units='meters',
                   position_columns=['tcp_x','tcp_y','tcp_z','cup_x','cup_y','cup_z','bowl_x','bowl_y','bowl_z'])
    payload = dict(n=n, positions=encode(positions, '<f4'), episodes=encode(episodes, '<i4'), skills=skills, summary=summary)
    (OUT / (arm + '.js')).write_text('window.TRAJECTORIES=window.TRAJECTORIES||{};window.TRAJECTORIES[' + json.dumps(arm) + ']=' + json_script(payload) + ';\n')
    # Round-trip validation includes every plotted coordinate and macro range.
    decoded = np.frombuffer(base64.b64decode(payload['positions']), dtype='<f4').reshape(n+1,9)
    assert np.array_equal(decoded, positions)
    print(arm, json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    summary = {arm: build_arm(arm) for arm in ARMS}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    template = (OUT / 'template.html').read_text()
    bundle = plotly_bundle(ROOT / 'artifacts/raps_coverage_950k_20261113/coverage.html')
    (OUT / 'trajectories.html').write_text(template.replace('__BUNDLE__', bundle))
    print('Built offline viewer; all controls retained, no simulation or training.', flush=True)
