from types import SimpleNamespace
import copy
import json

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from domain.robocasa.batched_skills import BatchedObservation
from exploration.diagnostics import collect_gpu_split7 as gpu
from exploration.diagnostics.collect_cabinet_grasp_sequences import selected_cases
from exploration.diagnostics.trajectory_store import read_transitions


def snapshot(z=0., held=False):
    return dict(ee_position=[0., 0., .2], inside_region=True, failed_object_ids=[],
        objects={name: dict(position_root=[0., 0., z], position_world=[0., 0., z]) for name in gpu.OBJECT_NAMES},
        contact_snapshot=dict(objects={name: dict(left_force=.1 if held else 0., right_force=.1 if held else 0.,
            grasped=held) for name in gpu.OBJECT_NAMES}))


def test_tick_labels_preserve_scalar_cumulative_grasp_and_lift():
    slot = dict(grasp_frames={}, lift_reference={})
    before = snapshot(held=True)
    for tick in range(1, 6):
        after = snapshot(z=tick*.004, held=True)
        contact, outcome = gpu._tick_outcome(slot, [0., 0., .1, -1.], before, after)
        assert outcome['stable_grasp_frames'][gpu.OBJECT_NAMES[0]] == tick
        assert outcome['duration_steps'] == 1 and outcome['final_stage'] == 'control'
        assert contact == after['contact_snapshot']
        assert outcome['held_object_ids'] == (list(gpu.OBJECT_NAMES) if tick >= 4 else [])
        assert outcome['lifted_object_ids'] == (list(gpu.OBJECT_NAMES) if tick >= 4 else [])
        before = after
    _, outcome = gpu._tick_outcome(slot, [0., 0., 0., 1.], before, snapshot(z=.02))
    assert outcome['lift_reference_heights_root'] == {}
    assert outcome['held_object_ids'] == []


def test_contract_tolerates_only_approved_backend_numeric_noise():
    expected = dict(dimension=67, dtype='float32', root_transform_world=dict(position_m=[0., 0., 0.],
        quaternion_wxyz=[1., 0., 0., 0.]), collision_meshes={'cup': dict(extents_m=[.1, .1, .1], source='mesh')})
    actual = copy.deepcopy(expected)
    actual['root_transform_world']['position_m'][2] = 7e-6
    actual['collision_meshes']['cup']['extents_m'][0] += 1e-8
    gpu._check_contract(actual, expected)
    bad_root = copy.deepcopy(actual)
    bad_root['root_transform_world']['position_m'][2] = 2e-5
    with pytest.raises(ValueError, match='root_transform_world.position_m'):
        gpu._check_contract(bad_root, expected)
    actual['dimension'] = 66
    with pytest.raises(ValueError, match='contract differs'):
        gpu._check_contract(actual, expected)
    actual = copy.deepcopy(expected)
    actual['collision_meshes']['cup']['extents_m'][0] += .001
    with pytest.raises(ValueError, match='extents differ'):
        gpu._check_contract(actual, expected)


@pytest.fixture
def fake_gpu(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('Fake scheduler exercises real batched CUDA tensor execution')
    class Raw:
        device = torch.device('cuda')
        unwrapped = None
        def __init__(self, n):
            self.num_envs = n
            self.unwrapped = self
            self.positions = torch.zeros((n, 3), device=self.device)
            self.positions[:, 2] = .2
            self.resets = []
            self.closed = False
        def step(self, full):
            self.positions += full[:, :3]*.1
            flags = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            return None, None, flags, flags, {}
        def close(self): self.closed = True
    class Sensor:
        def __init__(self, raw):
            self.raw = raw
            self.reset_orientation = torch.zeros((raw.num_envs, 4), device=raw.device)
            self.arm = SimpleNamespace(ee_pose_at_base=SimpleNamespace(q=self.reset_orientation.clone()))
        def read(self, grip):
            raw, n = self.raw, self.raw.num_envs
            state = torch.zeros((n, 67), device=raw.device)
            state[:, 18:21] = raw.positions
            state[:, 32] = grip
            objects = torch.zeros((n, 2, 3), device=raw.device)
            obs = BatchedObservation(raw.positions.clone(), torch.tensor([1.,0.,0.,0.], device=raw.device).expand(n, -1),
                objects, objects, torch.ones(n, device=raw.device), torch.zeros((n, 2, 2), device=raw.device),
                torch.zeros((n, 2), dtype=torch.bool, device=raw.device), torch.zeros((n, 2), device=raw.device))
            return state, obs, torch.ones(n, dtype=torch.bool, device=raw.device), torch.zeros((n, 2), dtype=torch.bool, device=raw.device)
        def controller_action(self, low, execution):
            return torch.cat((low[:, :3], torch.zeros((self.raw.num_envs, 3), device=self.raw.device), low[:, 3:]), -1)
    class Provider:
        def __init__(self, settings, case, args):
            self.settings, self.case, self.trace, self.phase, self.failure = settings, case, [], 'retreat', None
            self.called = False
        def action(self, state):
            if self.called:
                self.phase = 'park'
            self.called = True
            schema = gpu.skill_schema(self.settings)
            if self.case['case_id'] == 0:
                return schema.encode(gpu._EEDeltaAction('align_yz', dict(y_offset=0., z_offset=0.)))
            return schema.encode(gpu._EEDeltaAction('move_delta_ee_pose', dict(dx=.05, dy=0., dz=0.)))
        def on_step(self, row): self.trace.append(dict(step=row['step']))
        def on_end(self, row): pass
        def report(self, skills): return dict(calibration_success=self.phase == 'park', failure=None)
    raws = []
    def make_env(**kwargs):
        raw = Raw(kwargs['num_envs']); raws.append(raw)
        return raw, OmegaConf.create(dict(reward={}, sim_backend='gpu'))
    def reset(raw, ids, seeds):
        raw.positions[ids] = torch.tensor([0.,0.,.2], device=raw.device)
        raw.resets.append((list(ids), list(seeds)))
    monkeypatch.setattr(gpu, 'make_env', make_env)
    monkeypatch.setattr(gpu, 'BatchedCabinetSensor', Sensor)
    monkeypatch.setattr(gpu, 'SplitSequenceProvider', Provider)
    monkeypatch.setattr(gpu, 'reset_slots', reset)
    monkeypatch.setattr(gpu, 'verify_sparse_joint_writes', lambda raw: None)
    monkeypatch.setattr(gpu, '_contract', lambda *args: dict(dimension=67))
    return raws


def cases():
    result = selected_cases('collect', 1000000)[:3]
    return dict(cases=result)


SETTINGS = dict(skill_set='split7', latched_gripper=True, hold_closed_during_motion=False)


def test_independent_zero_duration_refill_preserves_peer_ticks(tmp_path, fake_gpu):
    metrics = gpu.run_gpu(tmp_path, cases(), None, SETTINGS, 2, export_every_cases=2)
    assert metrics['closed_cases'] == 3
    assert metrics['actual_control_ticks'] == 4
    assert metrics['completed_skills'] == 2  # Excludes the completed zero-tick rejection.
    assert metrics['batch_control_steps'] == 2
    assert not metrics['halted_by_benchmark_limit']
    assert len(fake_gpu[0].resets) == 4  # Initial pair, then only newly free slot, twice each.
    assert fake_gpu[0].resets[-1][0] == [0]
    assert fake_gpu[0].closed
    for case_id, ticks in [(0, 0), (1, 2), (2, 2)]:
        path = tmp_path/'cases'/f'case_{case_id:03d}'
        summary = json.loads((path/'summary.json').read_text())
        assert summary['control_ticks'] == ticks
        rows = read_transitions(path/'attempt_000001')
        assert len(rows) == ticks
        assert [r['step'] for r in rows] == list(range(1, ticks+1))
        skills = [json.loads(line) for line in (path/'attempt_000001'/'skills.jsonl').read_text().splitlines()]
        assert skills[-1]['completed'] and skills[-1]['duration_low_steps'] == ticks
    with np.load(tmp_path/'evaluation.npz') as data:
        assert data['actions'].shape == (2, 13)


def test_benchmark_does_not_overshoot_or_publish_active_case(tmp_path, fake_gpu):
    metrics = gpu.run_gpu(tmp_path, cases(), None, SETTINGS, 2, max_control_steps=3)
    assert metrics['actual_control_ticks'] == 2
    assert metrics['halted_by_benchmark_limit']
    assert metrics['closed_cases'] == 1
    for case_id in (1, 2):
        path = tmp_path/'cases'/f'case_{case_id:03d}'
        assert not (path/'summary.json').exists()
        rows = read_transitions(path/'attempt_000001')
        assert len(rows) == 1
        skill = json.loads((path/'attempt_000001'/'skills.jsonl').read_text().splitlines()[-1])
        assert not skill['completed'] and skill['status'] == 'budget_truncated'
    # Resume gives active cases a fresh attempt and preserves the interrupted raw data.
    gpu.run_gpu(tmp_path, cases(), None, SETTINGS, 2)
    assert (tmp_path/'cases'/'case_001'/'attempt_000002').exists()
    assert len(read_transitions(tmp_path/'cases'/'case_001'/'attempt_000001')) == 1


def test_final_case_budget_cannot_mark_skill_completed(tmp_path, fake_gpu, monkeypatch):
    monkeypatch.setattr(gpu, 'BUDGET', 1)
    manifest = cases()
    manifest['cases'] = manifest['cases'][1:]
    metrics = gpu.run_gpu(tmp_path, manifest, None, SETTINGS, 2)
    assert metrics['active_control_ticks'] == 2 and metrics['completed_skills'] == 0
    for case in manifest['cases']:
        path = tmp_path/'cases'/f"case_{case['case_id']:03d}"/'attempt_000001'
        row = read_transitions(path)[0]
        skill = json.loads((path/'skills.jsonl').read_text().splitlines()[0])
        assert row['truncated'] and row['run_budget_exhausted']
        assert not skill['completed'] and not skill['trainable']
        assert skill['truncated'] and skill['status'] == 'episode_truncated'


def test_native_preflight_failure_prevents_case_assignment(tmp_path, monkeypatch, fake_gpu):
    def reject(raw):
        raise RuntimeError('SAPIEN sparse articulation writes are incorrect')
    monkeypatch.setattr(gpu, 'verify_sparse_joint_writes', reject)
    with pytest.raises(RuntimeError, match='sparse articulation'):
        gpu.run_gpu(tmp_path, cases(), None, SETTINGS, 2)
    assert not (tmp_path/'cases').exists()
    assert not (tmp_path/'trajectory_chunks').exists()
    assert json.loads((tmp_path/'runtime_preflight.json').read_text())['sparse_joint_writes'] is False
    assert fake_gpu[0].closed


def test_refill_stops_if_native_reset_changes_a_live_peer(tmp_path, monkeypatch, fake_gpu):
    original = gpu.reset_slots
    def bad_reset(raw, ids, seeds):
        original(raw, ids, seeds)
        peers = [i for i in range(raw.num_envs) if i not in ids]
        if peers:
            raw.positions[peers[0], 0] += .01
    monkeypatch.setattr(gpu, 'reset_slots', bad_reset)
    with pytest.raises(RuntimeError, match='Subset reset changed live slot'):
        gpu.run_gpu(tmp_path, cases(), None, SETTINGS, 2)
    assert fake_gpu[0].closed
