import json
from copy import deepcopy

import numpy as np
import pytest

from exploration.diagnostics import collect_primitive_combinations as m


def skill(i, **changes):
    return dict(dict(completed=True, duration_low_steps=2, skill_id=i, episode_id=1,
        start_step=2*(i-1), end_step=2*i, terminated=False, truncated=False,
        state=np.full(67, i).tolist(), action=np.zeros(13).tolist(),
        next_state=np.full(67, i+1).tolist(), status='success'), **changes)


def trial(skills, trial_id=0):
    rows = []
    for s in m.completed_endpoints(skills):
        for target in m.OBJECT_NAMES:
            rows.append(dict(raw_path='raw/skills.jsonl', trial_id=trial_id,
                skill_id=s['skill_id'], object=target, primary_event='static',
                endpoint_hash=s['endpoint_hash'], object_hash=s['endpoint_hash']+':'+target))
    return dict(plan=dict(trial_id=trial_id, scene='original', object=m.OBJECT_NAMES[0]),
        raw_path='raw/skills.jsonl', skills=skills, annotations=rows)


def test_endpoints_retain_executed_failures_and_reject_invalid_contract():
    rows = [skill(1, status='physical_collision'), skill(2, duration_low_steps=0),
            skill(3, completed=False), skill(4)]
    endpoints = m.completed_endpoints(rows)
    assert [s['raw_reference'] for s in endpoints] == [0, 3]
    assert endpoints[0]['status'] == 'physical_collision'
    for field, value in [('state', [0]*66), ('action', [0]*12), ('next_state', [np.nan]*67)]:
        with pytest.raises(ValueError):
            m.completed_endpoints([skill(1, **{field:value})])


def test_indices_preserve_gaps_reset_boundaries_and_deduplicate_full_states():
    skills = [skill(i) for i in range(1, 7)]
    index = m.build_indices([trial(skills), trial(deepcopy(skills), 1)], 20, 7)
    assert len(index['balanced']['samples']) == 12  # Two object rows per endpoint.
    assert len(index['full_state']) == 6
    assert index['balanced']['duplicate_object_endpoints'] == 12
    assert index['windows']['5'][0]['raw_references'] == [0, 1, 2, 3, 4]
    assert len(index['windows']['5']) == 2
    assert not index['windows_complete']  # The other cases have no windows.
    skills[2]['completed'] = False
    index = m.build_indices([trial(skills)], 20, 7)
    assert index['windows']['5'] == []
    assert [w['raw_references'] for w in index['windows']['3']] == [[3, 4, 5]]
    skills[4]['episode_id'] = 2
    assert m.build_indices([trial(skills)], 20, 7)['windows']['3'] == []


@pytest.mark.parametrize('source', ['training_hashes', 'calibration_hashes'])
def test_overlap_excludes_whole_attempt_without_hiding_raw_labels(source):
    t = trial([skill(i) for i in range(1, 7)])
    digest = m.completed_endpoints(t['skills'])[2]['endpoint_hash']
    index = m.build_indices([t], 20, 7, **{source:{digest}})
    assert not index['balanced']['samples'] and not index['full_state']
    assert all(not w for w in index['windows'].values())
    assert len(index['annotations']) == 12
    assert index['excluded_attempts'][0][source.replace('_hashes', '_overlap')]


def test_calibration_provenance_and_independent_sampling_seed(tmp_path):
    contract = {'dimension':13}
    (tmp_path/'manifest.json').write_text(json.dumps(dict(action_contract=contract, sampling_seed=4)))
    (tmp_path/'skills.jsonl').write_text(json.dumps(skill(1))+'\n')
    with pytest.raises(ValueError, match='independent'):
        m.calibration_exclusions([tmp_path], contract, 4)
    hashes, evidence = m.calibration_exclusions([tmp_path], contract, 5)
    assert hashes == {m.completed_endpoints([skill(1)])[0]['endpoint_hash']}
    assert evidence[0]['files'][0]['sha256']
    with pytest.raises(ValueError, match='contract'):
        m.calibration_exclusions([tmp_path], {'dimension':12}, 5)


def test_cli_budgets_and_separate_default_seed():
    def args(*extra):
        return m.parse_args(['--output', 'unused', *extra])
    assert args().max_steps == 0
    assert args('--stage', 'pilot').max_steps == 3200
    assert args('--stage', 'pilot').seed != args('--stage', 'collect', '--max-steps', '1').seed
    with pytest.raises(ValueError, match='explicit'):
        args('--stage', 'collect')
    with pytest.raises(ValueError, match='3200'):
        args('--stage', 'pilot', '--max-steps', '3201')


@pytest.mark.parametrize('stage', ['plan', 'pilot'])
def test_main_plan_executes_nothing_and_pilot_shares_global_budget(tmp_path, monkeypatch, stage):
    class Env:
        unwrapped = None
        cabinet_items = [type('Object', (), dict(name=n))() for n in m.OBJECT_NAMES]
        def __init__(self):
            self.unwrapped = self
        def reset(self, **kwargs):
            pass
        def close(self):
            pass
    class Encoder:
        def __init__(self, *args, **kwargs):
            pass
        def contract(self):
            return {'dimension':67}
    calls = []
    def collect(env, encoder, settings, plan, directory, budget, args):
        calls.append(budget)
        return dict(plan=plan, raw_path=str(directory/'skills.jsonl'), skills=[], annotations=[],
            summary=dict(scene=plan['scene'], object=plan['object'], recipe=plan['recipe'],
                trial_id=plan['trial_id'], outcome={'status':'failed'}, controls=budget,
                max_held_lift_m=0, physics_substeps=5*budget, forbidden_contact_substeps=0))
    monkeypatch.setattr(m, '_native_modules', lambda:None)
    monkeypatch.setattr(m, 'source_evidence', lambda *a:{})
    monkeypatch.setattr(m, 'make_env', lambda **kwargs:(Env(), None))
    monkeypatch.setattr(m, 'CabinetSkillStateEncoder', Encoder)
    monkeypatch.setattr(m, 'object_audit', lambda *a:dict(
        collision_bounds_root_m=[[.4, 0, .6], [.5, .1, .7]], pose_in_root=[.45, .05, .65]))
    monkeypatch.setattr(m, 'collect_trial', collect)
    out = tmp_path/'output'
    assert m.main(['--stage', stage, '--output', str(out), '--max-steps', '5',
                   '--trial-controls', '3']) == 0
    assert len(json.loads((out/'sampling_plan.json').read_text())) == 160
    manifest = json.loads((out/'manifest.json').read_text())
    assert not manifest['validation_ready']
    if stage == 'plan':
        assert calls == [] and not (out/'summary.json').exists()
    else:
        assert calls == [3, 2]
        summary = json.loads((out/'collection_summary.json').read_text())
        assert summary['physical_control_ticks'] == 5
        assert not summary['readiness_checks']['training_independence']
        assert not summary['readiness_checks']['calibration_independence']
