from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from exploration.cee_us_backend import CEEUSBackend
from exploration.raps_gnn_actor import DEFAULTS, SCHEMA, RAPSImaginationActor, action_contract
from exploration.raps_replay_compare import (
    initialize, balanced_arrays, update, evaluate, events, frozen_digest, seed_all,
)


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    settings={**DEFAULTS,'skill_set':'compact6','latched_gripper':True,
              'hidden_size':8,'actor_layers':1,'critic_layers':1,
              'imagination_batch_size':4,'imagination_horizon':3,'updates_per_round':2}
    config=OmegaConf.create(dict(seed=0,device='cpu',raps_actor=settings,object_heuristic={
        'approach_weight':50.,'distance_scale_m':.1,'object_weights':{'cabinet_item_16':1.,'cabinet_item_17':1.}}))
    # initialize() uses the original 128-wide GNN, as do all checkpoints in the experiment.
    backend=CEEUSBackend(agent_dim=33,action_dim=12,object_heuristic=OmegaConf.to_container(config.object_heuristic))
    actor=RAPSImaginationActor(backend,settings)
    rng=np.random.default_rng(9)
    states=rng.normal(size=(8,67)).astype(np.float32)*.01
    states[:,57:]=np.array([1,0,.14,.097,.092,0,1,.076,.072,.072])
    targets=states.copy();targets[:,:57]+=.001
    actions=np.zeros((8,12),np.float32);actions[:,4]=1
    replay=dict(states=states,next_states=targets,actions=actions,terminated=np.zeros(8,bool))
    backend.train(replay,epochs=1);actor.train(replay,updates=1)
    contract={'dimension':67}
    payload=dict(schema=SCHEMA,state_contract=contract,action_contract=action_contract(settings),
                 backend=deepcopy(backend.state_dict()),actor=deepcopy(actor.state_dict()))
    return config,payload,SimpleNamespace(contract=lambda:contract),replay


def compare_nested(left,right):
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:compare_nested(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):compare_nested(a,b)
    elif torch.is_tensor(left):torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
    else:assert left==right


def test_only_policy_learner_initialization_differs(fixture):
    config,payload,encoder,_=fixture
    b1,a1=initialize(config,payload,encoder,True)
    b2,a2=initialize(config,payload,encoder,False)
    compare_nested(b1.state_dict(),b2.state_dict())
    compare_nested(a1.state_dict(),payload['actor'])
    assert a2.updates==0 and not a2.actor_optimizer.state and not a2.critic_optimizer.state
    assert any(not torch.equal(a1.actor.state_dict()[k],v) for k,v in a2.actor.state_dict().items())
    assert any(not torch.equal(a1.critic.state_dict()[k],v) for k,v in a2.critic.state_dict().items())
    # Mutable optimizer tensors must not alias the source or the other arm.
    first=next(iter(b1.model.optimizer.state.values()))['exp_avg']
    other=next(iter(b2.model.optimizer.state.values()))['exp_avg']
    assert first.data_ptr()!=other.data_ptr()
    first.add_(1)
    assert not torch.equal(first,other)


def test_balanced_sampling_counts_replacement_and_no_source_mutation(fixture):
    *_,old=fixture
    new=deepcopy(old);new['states'][:,0]=9.
    untouched=deepcopy(old)
    wm,starts,groups,oi,ni=balanced_arrays(old,new,np.random.default_rng(0))
    assert len(oi)==len(ni)==250
    assert len(wm['states'])==500
    assert np.all(wm['states'][250:,0]==9.)
    assert np.array_equal(groups,np.r_[np.zeros(8),np.ones(8)])
    compare_nested(old,untouched)
    with pytest.raises(ValueError):balanced_arrays(old,{k:v[:0] for k,v in new.items()},np.random.default_rng(0))


def test_actor_samples_half_old_half_new_and_all_three_learn(fixture,tmp_path):
    config,payload,encoder,old=fixture
    backend,actor=initialize(config,payload,encoder,True)
    new=deepcopy(old);new['states'][:,0]+=.2;new['next_states'][:,0]+=.2
    calls=[];original=actor.imagine
    def imagine(starts):
        calls.append(starts[:,0].detach().numpy().copy())
        assert int((starts[:,0]>.1).sum())==2
        return original(starts)
    actor.imagine=imagine
    skills=[dict(state=s.tolist(),next_state=y.tolist(),action=a.tolist(),episode_id=1,
                 terminated=False,truncated=False,duration_low_steps=1,trainable=True)
            for s,y,a in zip(new['states'],new['next_states'],new['actions'])]
    before_model=deepcopy(backend.model.model.state_dict());before_actor=deepcopy(actor.actor.state_dict());before_critic=deepcopy(actor.critic.state_dict())
    result=update(backend,actor,old,skills,tmp_path,'test',np.random.default_rng(0),smoke=True)
    assert len(calls)==2 and result['model']['updates']==4
    for previous,module in [(before_model,backend.model.model),(before_actor,actor.actor),(before_critic,actor.critic)]:
        assert any(not torch.equal(previous[k],v) for k,v in module.state_dict().items())
    for name in backend.normalizer_names:
        norm=getattr(backend.model,name)
        assert torch.isfinite(norm.mean).all() and torch.isfinite(norm.std).all()
    with pytest.raises(ValueError,match='both nonterminal groups'):
        actor.train(dict(states=old['states'],terminated=np.zeros(8,bool)),start_groups=np.zeros(8))


def row(step,episode=1,held=True,z=.7):
    names=('cabinet_item_16','cabinet_item_17')
    return dict(step=step,stage='round_1',episode_id=episode,action=[0,0,0,-1],
        after={'failed_object_ids':[],'objects':{n:{'position_root':[.45,0,z]} for n in names}},
        contact_events=[{'objects':{n:{'left_force':.01,'right_force':0.,'grasped':held} for n in names}}],
        reward={'weights':{'contact_force_threshold':.0001}})


def test_events_are_deduplicated_in_time_and_reset_at_episode_boundaries():
    rows=[row(i,z=.7+.004*(i-1)) for i in range(1,7)]
    rows+=[row(i,episode=2) for i in (7,8,9)]
    result=events(rows,[])
    assert result['contact_time_percent']==100.
    assert result['grasp_events']==2 and result['lift_events']==2
    assert result['per_object']['cabinet_item_16']['first_grasp_step']==4


def test_frozen_evaluation_restores_rng_shift_and_does_not_update(fixture,monkeypatch,tmp_path):
    import exploration.raps_replay_compare as module
    config,payload,encoder,_=fixture
    backend,actor=initialize(config,payload,encoder,True)
    env=SimpleNamespace(unwrapped=SimpleNamespace(object_shift_xy=np.array([.03,0.])))
    calls=[]
    class Collector:
        def __init__(self,*a):pass
        def collect(self,count,*a,**kwargs):
            calls.append(env.unwrapped.object_shift_xy.copy())
            np.random.rand();torch.rand(1)
            rows=[row(i,held=False) for i in range(1,count+1)]
            for r in rows:kwargs['on_step'](r)
            return rows,[]
    monkeypatch.setattr(module,'SkillCollector',Collector)
    before=module.rng_state();digest=frozen_digest(backend,actor);ticks=[]
    result=evaluate(env,encoder,backend,actor,tmp_path,4,(0.,0.),ticks.append)
    assert result['frozen_state_verified'] and len(ticks)==4
    assert frozen_digest(backend,actor)==digest
    np.testing.assert_array_equal(env.unwrapped.object_shift_xy,[.03,0.])
    np.testing.assert_array_equal(calls[0],[0,0])
    compare_nested(module.rng_state(),before)


def test_fixed_wm_test_excludes_either_arms_overlap_and_missing_motion():
    from exploration.raps_replay_compare import fixed_wm_test
    states=np.zeros((3,67),np.float32);states[:,0]=[1,2,3]
    target=states.copy();target[:,33]+=.02
    actions=np.zeros((3,12),np.float32)
    model=SimpleNamespace(predict=lambda s,a:np.repeat(np.stack([target[int(x[0])-1] for x in s])[None],5,axis=0))
    data=dict(states=states,actions=actions,next_states=target)
    train=[{k:v[:1] for k,v in data.items()},{k:v[1:2] for k,v in data.items()}]
    result=fixed_wm_test(model,data,train)
    assert result['segments']==1 and result['excluded_overlap']==2
    assert result['moving_error_cm']==0. and result['moving_endpoints']==1
    assert result['no_motion_error_cm']==pytest.approx(2.)
    static=dict(states=states,actions=actions,next_states=states.copy())
    result=fixed_wm_test(model,static,[])
    assert result['moving_endpoints']==0 and result['moving_error_cm'] is None


def test_shared_bootstrap_checks_source_contract_and_is_independently_loaded(fixture,tmp_path):
    from exploration.raps_replay_compare import read_bootstrap
    from exploration.cee_us_cabinet import rng_state
    config,payload,encoder,replay=fixture
    _,actor=initialize(config,payload,encoder,False)
    skills=[dict(state=s.tolist(),next_state=y.tolist(),action=a.tolist(),episode_id=1,
                 terminated=False,truncated=False,duration_low_steps=1,trainable=True)
            for s,y,a in zip(replay['states'],replay['next_states'],replay['actions'])]
    package=dict(schema='cabinet-shared-bootstrap-v1',source_sha256='source',source_replay_sha256='replay',
                 state_contract=encoder.contract(),action_contract=action_contract(actor.settings),smoke=True,
                 device='cpu',object_shift_xy_m=[.03,0.],ticks=[{'step':i+1} for i in range(160)],skills=skills,rng=rng_state())
    path=tmp_path/'bootstrap.pt';torch.save(package,path)
    a=read_bootstrap(path,'source','replay',encoder,actor,True,'cpu')
    b=read_bootstrap(path,'source','replay',encoder,actor,True,'cpu')
    compare_nested(a,b)
    a[0][0]['step']=999
    assert b[0][0]['step']==1
    with pytest.raises(ValueError,match='source_sha256'):
        read_bootstrap(path,'other','replay',encoder,actor,True,'cpu')
    with pytest.raises(ValueError,match='smoke'):
        read_bootstrap(path,'source','replay',encoder,actor,False,'cpu')


def test_independent_cli_requires_shared_bootstrap_and_selects_one_arm(monkeypatch,tmp_path):
    from exploration import raps_replay_compare as module
    seen=[]
    monkeypatch.setattr(module,'run',lambda args:seen.append(args) or 0)
    for arm in ('continue','fresh'):
        with pytest.raises(SystemExit) as exc:module.main([arm,'--output',str(tmp_path/arm)])
        assert exc.value.code==2
        assert module.main([arm,'--output',str(tmp_path/arm),'--bootstrap','shared/bootstrap.pt'])==0
    assert [a.stage for a in seen]==['continue','fresh']
    assert seen[0].output!=seen[1].output
    assert seen[0].bootstrap==seen[1].bootstrap
