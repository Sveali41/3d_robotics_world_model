from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from exploration.diagnostics.collect_wm_validation import (
    CLASSES, OBJECT_NAMES, SCENES, ScriptedProvider, annotate, endpoint_hash,
    generate_plans, local_to_root, parser, region_info, select_balanced,
    validate_args, windows_index,
)
from exploration.raps.raps_gnn_actor import DEFAULTS


def args():
    return parser().parse_args(['--checkpoint','dummy.pt','--output','dummy'])


def region():
    return dict(root_to_cabinet_rotation=[[0,-1,0],[1,0,0],[0,0,1]],
                root_to_cabinet_translation=[1,2,3],lower=[0,0,0],upper=[2,4,6])


def fixture(rise=0., force=0., moving_tcp=True, other_contact=False):
    objects={n:dict(position_root=[.5,0,.1],position_world=[.5,0,.1]) for n in OBJECT_NAMES}
    before=dict(ee_position=[0,0,0],objects=deepcopy(objects))
    ticks=[]
    for i in range(6):
        after=deepcopy(before)
        after['objects'][OBJECT_NAMES[0]]['position_root'][2] += rise*i/5
        after['objects'][OBJECT_NAMES[0]]['position_world'][2] += rise*i/5
        after.update(ee_position=[.02 if moving_tcp else 0,0,0],failed_object_ids=[OBJECT_NAMES[1]])
        contact={n:dict(left_force=force if j==0 else (1. if other_contact else 0.),right_force=0.,grasped=True) for j,n in enumerate(OBJECT_NAMES)}
        ticks.append(dict(stage='validation_only',episode_id=1,skill_id=1,step=i+1,
                          action=[0,0,0,-1],after=after,contact_events=[dict(objects=contact)],
                          reward=dict(weights=dict(contact_force_threshold=.05)),terminated=False,truncated=False))
    skill=dict(completed=True,trainable=True,duration_low_steps=6,skill_id=1,episode_id=1,start_step=0,end_step=6,
               state=np.zeros(67).tolist(),action=np.zeros(12).tolist(),next_state=np.ones(67).tolist(),
               before=before,after=ticks[-1]['after'],status='no_stable_grasp',primitive_type='lift')
    plan=dict(trial_id=0,object=OBJECT_NAMES[0],intent='lift')
    return skill,ticks,plan


def test_actual_lift_and_commanded_lift_failure():
    s,t,p=fixture(rise=.02,force=.1)
    rows=annotate(s,t,p,'raw',region(),args())
    assert rows[0]['primary_event']=='lift' and rows[0]['grasp']
    assert rows[0]['event_booleans']['contact'] and not rows[0]['event_booleans']['static']
    assert rows[0]['failed_object_ids']==[OBJECT_NAMES[1]]
    s,t,p=fixture(rise=0,force=.1)
    row=annotate(s,t,p,'raw',region(),args())[0]
    assert row['primary_event']=='contact_only' and not row['event_booleans']['lift']
    assert row['intended_event']=='lift' and row['status']=='no_stable_grasp'


def test_force_boundary_other_object_and_displacement():
    s,t,p=fixture(force=.05)
    assert annotate(s,t,p,'raw',region(),args())[0]['primary_event']=='free_motion'
    s,t,p=fixture(force=.05,other_contact=True)
    assert annotate(s,t,p,'raw',region(),args())[0]['primary_event']=='static'
    s,t,p=fixture(rise=.02,force=0)
    for tick in t:
        tick['action'][3]=1
    assert annotate(s,t,p,'raw',region(),args())[0]['primary_event']=='displacement'
    s['completed']=False
    assert annotate(s,t,p,'raw',region(),args())==[]


def test_transform_bin_edges_outside():
    r=region()
    for local,cell in [([0,0,0],[0,0,0]),([2,4,6],[1,1,1]),([2.01,0,0],'outside')]:
        root=local_to_root(local,r)
        out=region_info(root,{},r,[2,2,2],.1)
        assert np.allclose(out['local_position'],local)
        assert out['spatial_bin']==cell


def test_reservoir_deterministic_no_duplicate_or_overlap_inflation():
    rows=[]
    for i in range(30):
        rows.append(dict(scene='original',object=OBJECT_NAMES[0],primary_event='static',object_hash=str(i),training_overlap=False))
    rows.extend([deepcopy(rows[0]),dict(rows[1],object_hash='excluded',training_overlap=True)])
    a=select_balanced(rows,3,5)
    assert a==select_balanced(rows,3,5)
    assert len(a['samples'])==3 and a['duplicate_object_endpoints']==1
    assert a['training_overlap_object_endpoints']==1
    assert len(a['strata'])==len(SCENES)*len(OBJECT_NAMES)*len(CLASSES)
    assert a['strata'][4]['observed_unique']==30
    assert a['status']=='quota_unmet'


def test_windows_never_bridge_reset_incomplete_overlap_or_physical_gap():
    skills=[dict(episode_id=1,skill_id=i,start_step=(i-1)*2,end_step=i*2,terminated=False,truncated=False,raw_reference=i) for i in range(1,7)]
    assert len(windows_index(skills)['3'])==4
    # Removing an incomplete or overlapping endpoint leaves a raw skill-id gap.
    assert windows_index(skills[:2]+skills[3:])['5']==[]
    assert windows_index(skills[:2]+skills[3:])['3']==[[4,5,6]]
    skills[2]['episode_id']=2
    skills[4]['start_step']+=1
    assert windows_index(skills)['3']==[]


def test_shared_plan_seed_parameter_ranges_and_hash():
    a=args()
    plans=generate_plans(a,20)
    assert plans==generate_plans(a,20)
    assert len({(p['object'],p['intent']) for p in plans[:10]})==10
    assert all(.02<=p['lift']<=.1 and np.linalg.norm(p['neighborhood_offset'])<=a.radius for p in plans)
    assert endpoint_hash(np.zeros(67,dtype=np.float64),np.zeros(12),np.zeros(67))==endpoint_hash(np.zeros(67,dtype=np.float32),np.zeros(12),np.zeros(67))
    provider=ScriptedProvider(dict(DEFAULTS,skill_set='compact6'),plans[0],region(),a)
    with pytest.raises(ValueError,match='outside primitive'):
        provider.encode('lift',z_dist=.17)


def test_budget_validation():
    a=args();a.stage='collect'
    with pytest.raises(ValueError,match='explicit'):
        validate_args(a)
    a=args();a.stage='smoke';validate_args(a)
    assert a.max_steps==200
    for steps in (201,202,0,1):
        a.max_steps=steps
        with pytest.raises(ValueError):validate_args(a)


def test_global_budget_shared_trial_plans_without_physics(tmp_path,monkeypatch):
    import exploration.diagnostics.collect_wm_validation as m
    settings=dict(DEFAULTS,skill_set='compact6',latched_gripper=True)
    checkpoint=tmp_path/'input.pt';checkpoint.write_bytes(b'fake')
    contract=dict(dimension=67)
    payload=dict(schema=m.SCHEMA,state_contract=contract,action_contract=m.action_contract(settings),
                 config=dict(raps_actor=settings,episode_steps=800,centered_initial_tcp=True,execution='reference'))
    monkeypatch.setattr(m.torch,'load',lambda *a,**k:payload)
    monkeypatch.setattr(m,'source_evidence',lambda *a:dict(commit='test'))
    monkeypatch.setattr(m.importlib.metadata,'version',lambda n:'test')
    calls=[]
    class Env:
        def reset(self,**kw):pass
        def close(self):pass
    class Encoder:
        def __init__(self,*a,**k):pass
        def contract(self):return contract
        def action_projection_contract(self):return region()
    class Collector:
        def __init__(self,env,encoder,provider,directory):
            self.plan=provider.plan;self.directory=directory
        def collect(self,count,policy,**kw):
            calls.append((count,deepcopy(self.plan)))
            self.directory.mkdir()
            return [None]*count,[]
    monkeypatch.setattr(m,'make_env',lambda **kw:(Env(),m.OmegaConf.create({})))
    monkeypatch.setattr(m,'CabinetSkillStateEncoder',Encoder)
    monkeypatch.setattr(m,'SkillCollector',Collector)
    output=tmp_path/'out'
    assert m.main(['--stage','smoke','--checkpoint',str(checkpoint),'--output',str(output),'--max-steps','10','--trial-steps','3'])==0
    assert [c[0] for c in calls]==[3,2,3,2]
    assert calls[:2]==calls[2:]
    import json
    assert json.loads((output/'manifest.json').read_text())['physical_control_ticks']==10


def test_training_exclusion_union_and_missing_referenced_source(tmp_path):
    from exploration.diagnostics.collect_wm_validation import replay_exclusions
    a=args();a.checkpoint=tmp_path/'model.pt'
    state=np.zeros((1,67),np.float32);action=np.zeros((1,12),np.float32)
    target=np.ones((1,67),np.float32)
    np.savez(a.checkpoint.with_suffix('.replay.npz'),states=state,actions=action,next_states=target)
    second=tmp_path/'second.npz'
    np.savez(second,states=state+1,actions=action,next_states=target)
    a.exclude_replay=[second]
    hashes,evidence=replay_exclusions(a,dict(config={}))
    assert len(hashes)==2 and len(evidence)==2
    assert endpoint_hash(state[0],action[0],target[0]) in hashes
    with pytest.raises(FileNotFoundError,match='Required training replay missing'):
        replay_exclusions(a,dict(config=dict(replay_experiment=dict(source=str(tmp_path/'missing.pt')))))


def test_spatial_audit_counts_physical_skills_once_and_unvisited_bins():
    from exploration.diagnostics.collect_wm_validation import spatial_audit
    endpoint=dict(spatial_bin=[0,0,0],object_neighborhoods={n:True for n in OBJECT_NAMES})
    row=dict(scene='original',raw_path='trial/skills.jsonl',skill_id=1,region=dict(before=endpoint,after=dict(endpoint,spatial_bin='outside')))
    audit=spatial_audit([row,deepcopy(row)],[],[2,2,2])
    assert audit['original']['physical_skills']==1
    assert audit['original']['bins']['[0, 0, 0]']['start_endpoints']==1
    assert audit['original']['bins']['outside']['end_endpoints']==1
    assert len(audit['original']['unvisited_bins'])==7
    assert len(audit['shifted']['unvisited_bins'])==8
    assert audit['original']['object_neighborhoods'][OBJECT_NAMES[0]]['end_endpoints']==1


def test_reached_free_motion_goal_dwells_without_resampling():
    a=args();plan=generate_plans(a,10)[0]
    plan.update(intent='free_motion',cell=[0,0,0],cell_fraction=[.5,.5,.5])
    provider=ScriptedProvider(dict(DEFAULTS,skill_set='compact6'),plan,region(),a)
    state=np.zeros(67)
    provider.action(state)  # initial open
    goal=local_to_root(np.asarray(region()['lower'])+(np.array(plan['cell'])+plan['cell_fraction'])/a.grid*(np.array(region()['upper'])-region()['lower']),region())
    state[18:21]=goal+[.00934565,-.00407632,.00240499]
    action=provider.schema.decode(provider.action(state))
    assert action.primitive_type=='open_gripper'
    assert np.array_equal(provider.goal,goal)
    assert provider.schema.decode(provider.action(state)).primitive_type=='open_gripper'
    assert np.array_equal(provider.goal,goal)


def test_resume_partial_budget_skips_completed_and_breaks_attempt_windows(tmp_path,monkeypatch):
    import json
    import exploration.diagnostics.collect_wm_validation as m
    settings=dict(DEFAULTS,skill_set='compact6',latched_gripper=True)
    checkpoint=tmp_path/'input.pt';checkpoint.write_bytes(b'fake')
    contract=dict(dimension=67)
    payload=dict(schema=m.SCHEMA,state_contract=contract,action_contract=m.action_contract(settings),
                 config=dict(raps_actor=settings,episode_steps=800,centered_initial_tcp=True,execution='reference'))
    monkeypatch.setattr(m.torch,'load',lambda *a,**k:payload)
    monkeypatch.setattr(m,'source_evidence',lambda *a:dict(commit='original'))
    monkeypatch.setattr(m.importlib.metadata,'version',lambda n:'test')
    class Env:
        def reset(self,**kw):pass
        def close(self):pass
    class Encoder:
        def __init__(self,*a,**k):pass
        def contract(self):return contract
        def action_projection_contract(self):return region()
    calls=[]
    fail=[True]
    class Collector:
        def __init__(self,env,encoder,provider,directory):
            self.plan=provider.plan;self.directory=directory
        def collect(self,count,policy,**kw):
            calls.append((self.directory,self.plan['trial_id'],self.plan['seed'],count))
            self.directory.mkdir(parents=True)
            interrupted=fail[0] and self.plan['trial_id']==6
            actual=19 if interrupted else count
            skill,ticks,_=fixture()
            skill.update(terminated=False,truncated=False)
            ticks += [dict(step=i+1,skill_id=0) for i in range(6,actual)]
            skills=[skill]
            if interrupted:
                skills += [dict(skill,skill_id=i+2,completed=True,duration_low_steps=0) for i in range(100)]
            for name,rows in [('transitions.jsonl',ticks),('skills.jsonl',skills)]:
                (self.directory/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            if interrupted:
                fail[0]=False
                raise RuntimeError('100 consecutive zero-duration skills')
            return ticks,skills
    monkeypatch.setattr(m,'make_env',lambda **kw:(Env(),m.OmegaConf.create({})))
    monkeypatch.setattr(m,'CabinetSkillStateEncoder',Encoder)
    monkeypatch.setattr(m,'SkillCollector',Collector)
    output=tmp_path/'out'
    command=['--stage','collect','--checkpoint',str(checkpoint),'--output',str(output),'--max-steps','32000','--quota','5']
    with pytest.raises(RuntimeError,match='zero-duration'):
        m.main(command)
    interrupted_file=output/'original/trial_000006/skills.jsonl'
    preserved=interrupted_file.read_bytes()
    assert sum(x[3] for x in calls[:6])+19==4819
    calls.clear()
    with pytest.raises(ValueError,match='sampling plan differs'):
        m.main(command+['--resume','--seed','1000001'])
    assert not calls
    monkeypatch.setattr(m,'source_evidence',lambda *a:dict(commit='repaired'))
    assert m.main(command+['--resume'])==2
    assert interrupted_file.read_bytes()==preserved
    assert calls[0][1:]==(6,1000006,781)
    assert 'attempt_000001' in str(calls[0][0])
    assert sum(x[3] for x in calls)==27181
    assert not any(x[1]<6 for x in calls if 'original' in x[0].parts)
    old_plans={x[1]:x[2] for x in calls if 'original' in x[0].parts}
    new_plans={x[1]:x[2] for x in calls if 'shifted' in x[0].parts}
    assert all(new_plans[i]==seed for i,seed in old_plans.items())
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['physical_control_ticks']==32000
    assert manifest['source_evidence']['commit']=='original'
    recovery=manifest['recoveries'][0]
    assert recovery['source_evidence']['commit']=='repaired'
    assert recovery['saved_control_ticks']==dict(original=4819,shifted=0)
    assert recovery['partial_trials'][0]['saved_ticks']==19
    with np.load(output/'original/skills.npz') as data:
        assert len(data['episode_ids'])==len(set(data['episode_ids']))
    assert json.loads((output/'original/windows_index.json').read_text())['3']==[]
    with pytest.raises(ValueError,match='finished'):
        m.main(command+['--resume'])


def test_split_provider_separates_alignment_approach_and_closure():
    a = args()
    plan = dict(object=OBJECT_NAMES[0], intent='lift', jitter=[0., 0., 0.], lift=.04,
                neighborhood_offset=[0., .01, 0.])
    provider = ScriptedProvider(dict(DEFAULTS, skill_set='split7', motion_execution='planner'), plan, region(), a)
    state = np.zeros(67)
    state[18:21] = [.35, 0., .72]
    state[33:36] = [.45, .02, .74]
    assert provider.schema.decode(provider.action(state)).primitive_type == 'open_gripper'
    provider.phase = 'interact'
    aligned = provider.schema.decode(provider.action(state))
    assert aligned.primitive_type == 'align_yz'
    np.testing.assert_allclose(list(aligned.parameters.values()), [.02, .02], atol=1e-7)
    state[19:21] = [.02, .74]
    approach = provider.schema.decode(provider.action(state))
    assert approach.primitive_type == 'approach_x'
    assert approach.parameters['x_forward'] == pytest.approx(.10)
    assert provider.schema.decode(provider.action(state)).primitive_type == 'close_gripper'
    assert provider.schema.decode(provider.action(state)).primitive_type == 'lift'


def test_split_config_plans_without_loading_checkpoint_or_advancing_physics(tmp_path, monkeypatch):
    import json
    import exploration.diagnostics.collect_wm_validation as m
    config = m.OmegaConf.load('configs/raps_gnn_actor_cabinet_split7_planner.yaml')
    source = tmp_path/'config.yaml'
    m.OmegaConf.save(config, source)
    monkeypatch.setattr(m, 'source_evidence', lambda *a: dict(commit='test'))
    def forbidden(*a, **kw):
        raise AssertionError('Plan stage must not load a checkpoint or create physics')
    monkeypatch.setattr(m.torch, 'load', forbidden)
    monkeypatch.setattr(m, 'make_env', forbidden)
    out = tmp_path/'out'
    assert m.main(['--config', str(source), '--output', str(out), '--stage', 'plan']) == 0
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['checkpoint'] is None
    assert manifest['state_contract'] is None
    assert manifest['action_contract']['action_dim'] == 13
    assert manifest['config']['action_mode'] == 'joint_position'
    assert manifest['automatic_training'] is False


def test_planner_free_goal_has_bounded_attempts_and_keeps_original_target():
    a = args()
    plan = generate_plans(a, 1)[0]
    plan.update(intent='free_motion', cell=[0, 0, 0], cell_fraction=[.5, .5, .5])
    provider = ScriptedProvider(dict(DEFAULTS, skill_set='split7', motion_execution='planner'), plan, region(), a)
    state = np.zeros(67)
    provider.action(state)
    for _ in range(12):
        assert provider.schema.decode(provider.action(state)).primitive_type == 'move_delta_ee_pose'
    goal = provider.goal.copy()
    assert provider.schema.decode(provider.action(state)).primitive_type == 'open_gripper'
    np.testing.assert_array_equal(provider.goal, goal)
