import json
from types import SimpleNamespace

import numpy as np
import pytest

from exploration.diagnostics import render_cabinet_grasp_replays as r


def test_verify_state_strict_mismatch():
    expected = np.zeros(67, np.float32)
    assert r.verify_state(expected, expected.tolist(), 'initial') == 0
    actual = expected.copy(); actual[18] = 1e-7
    with pytest.raises(ValueError, match='step 1 after: replay state mismatch'):
        r.verify_state(actual, expected, 'step 1 after')


def test_labels_use_recorded_feedback_and_height():
    case = dict(case_id=2, object='cup', template='yxz')
    summary = dict(success=False, status='failure', report=dict(failure='lost_grasp_during_lift'))
    initial = np.zeros(67); after = initial.copy(); after[35] = .03
    trace = dict(phase='lift', target_grasped=True, stable_grasp_ticks=3)
    text = ' '.join(r.label_lines(case, summary, 10, trace, after, initial))
    assert 'not confirmed' in text and '+3.00 cm' in text and '0.50 s' in text
    assert 'lost_grasp_during_lift' in text
    trace['stable_grasp_ticks'] = 4
    assert 'Grasp: confirmed' in ' '.join(r.label_lines(case, summary, 10, trace, after, initial))
    assert 'reset (no grasp feedback)' in ' '.join(r.label_lines(case, summary, 0, None, initial, initial))


def test_invalid_episode_row_rejected(tmp_path):
    case = dict(case_id=2)
    directory = tmp_path/'cases'/'case_002'
    attempt = directory/'attempt_000001'; attempt.mkdir(parents=True)
    (directory/'summary.json').write_text(json.dumps(dict(case=case, attempt=1, control_ticks=1)))
    (attempt/'attempt.json').write_text(json.dumps(dict(case=case, attempt=1)))
    (attempt/'transitions.jsonl').write_text(json.dumps(dict(step=1, episode_id=2, full_controller_action=[0]*7))+'\n')
    (attempt/'phase_trace.json').write_text(json.dumps([dict(step=1)]))
    with pytest.raises(ValueError, match='Invalid recorded step'): r.read_case(tmp_path, 2)


def test_replay_mismatch_does_not_publish_video(tmp_path, monkeypatch):
    case = dict(case_id=2, object='cup', template='yxz', seed=1)
    summary = dict(success=False, status='failure', report=dict(failure='failed'), attempt=1)
    config = dict(action_mode='ee_delta', headless=True, sim_backend='cpu', sim=dict(sim_freq=100, control_freq=20),
                  render_mode=None, centered_initial_tcp=True)
    initial = np.zeros(67, np.float32)
    expected = initial.copy(); expected[18] = .1
    rows = [dict(state=initial.tolist(), next_state=expected.tolist(), full_controller_action=[0]*7)]
    trace = [dict(step=1, phase='lift', target_grasped=False, stable_grasp_ticks=0)]
    monkeypatch.setattr(r, 'read_case', lambda *args: (case, summary, dict(environment=config), rows, trace))
    class FakeEnv:
        unwrapped = None
        def __init__(self): self.unwrapped=self
        def reset(self, seed): pass
        def step(self, action): pass
        def render_rgb_array(self): return np.tile(np.arange(100, dtype=np.uint8)[None,:,None], (80,1,3))
        def close(self): pass
    class FakeEncoder:
        def __init__(self, env, latched_gripper): self.gripper_command=1
        def contract(self): return {'dimension':67}
        def encode(self): return initial.copy()
    class FakeWriter:
        def __init__(self, path): self.path=path; self.path.touch()
        def append_data(self, data): pass
        def close(self): pass
    monkeypatch.setattr(r, 'build_env', lambda config: FakeEnv())
    monkeypatch.setattr(r, 'CabinetSkillStateEncoder', FakeEncoder)
    monkeypatch.setattr(r.imageio, 'get_writer', lambda path, **kwargs: FakeWriter(__import__('pathlib').Path(path)))
    with pytest.raises(ValueError, match='step 1 after'):
        r.render_case(tmp_path, tmp_path, {'state_contract':{'dimension':67}}, 2)
    assert not (tmp_path/'case_002.mp4').exists()
    assert not (tmp_path/'.case_002.partial.mp4').exists()
