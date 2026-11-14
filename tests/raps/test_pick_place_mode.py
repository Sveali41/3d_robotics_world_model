"""Opt-in macro CLI, model contract and actor integration."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from exploration.cee_us.cee_us_backend import CEEUSBackend
from exploration.raps.raps_gnn_actor import (
    DEFAULTS, RAPSImaginationActor, action_contract, action_contract_matches,
    canonical_action, skill_schema,
)

@pytest.mark.parametrize('configured,flag,expected', [
    (False,[],False),(True,[],True),(False,['--pick-place-primitives'],True),
    (True,['--no-pick-place-primitives'],False),
])
def test_cli_precedence(tmp_path, monkeypatch, configured, flag, expected):
    from exploration.cee_us.cee_us_cabinet import main
    import exploration.raps.raps_gnn_actor as module
    config=tmp_path/'config.yaml'
    OmegaConf.save(OmegaConf.create({'agent':'raps_gnn_actor','raps_actor':{
        'pick_place_primitives':configured}}), config)
    seen=[]
    monkeypatch.setattr(module,'run',lambda args,cfg:seen.append(cfg.raps_actor.pick_place_primitives) or 0)
    assert main(['train','--config',str(config)]+flag)==0
    assert seen==[expected]


def test_default_contract_and_mode_separation():
    old={**DEFAULTS,'skill_set':'split7','latched_gripper':True}
    contract=action_contract(old)
    assert skill_schema(old).action_dim==13
    assert 'pick_place' not in contract
    macro={**old,'pick_place_primitives':True,'grasp_stop_on_contact':False}
    assert skill_schema(macro).action_dim==5
    assert skill_schema(macro).primitive_name_to_action_idx['place']==[0,1]
    assert set(action_contract(macro)['parameters']['place'])=={'x','y'}
    assert not action_contract_matches(contract,macro)
    assert not action_contract_matches(action_contract(macro),old)
    metadata=action_contract(macro)['pick_place']
    assert metadata['tcp_bounds_root_m']['grasp'][2]==[.692,.872]
    for version in (1, 2, 3, 4):
        previous=deepcopy(action_contract(macro))
        previous['pick_place']['version']=version
        assert not action_contract_matches(previous,macro)


def test_place_inactive_height_is_zeroed_for_world_model():
    schema = skill_schema({**DEFAULTS, 'pick_place_primitives': True})
    actions = torch.tensor([[0., 1., .2, -.4, .9], [1., 0., .2, -.4, .9]])
    canonical = canonical_action(actions, schema)
    torch.testing.assert_close(canonical[0], torch.tensor([0., 1., .2, -.4, 0.]))
    torch.testing.assert_close(canonical[1], actions[1])


def test_five_dimensional_model_actor_update_and_restore(tmp_path):
    torch.set_num_threads(1)
    backend=CEEUSBackend(agent_dim=33,action_dim=5,hidden_dim=8,batch_size=4)
    settings=dict(pick_place_primitives=True,motion_execution="planner",latched_gripper=True,grasp_stop_on_contact=False,
                  hidden_size=8,actor_layers=1,critic_layers=1,imagination_horizon=2,
                  imagination_batch_size=4,updates_per_round=1)
    actor=RAPSImaginationActor(backend,settings)
    rng=np.random.default_rng(19)
    states=rng.normal(size=(8,67)).astype(np.float32)
    actions=canonical_action(torch.tensor(rng.uniform(-1,1,(8,5)),dtype=torch.float32),actor.schema).numpy()
    next_states=states.copy()
    next_states[:,:backend.dynamic_dim]+=.001
    data=dict(states=states,actions=actions,next_states=next_states,terminated=np.zeros(8,bool))
    backend.train(data,1)
    metrics=actor.train(data)
    assert metrics['updates']==1
    assert np.isfinite(list(metrics.values())).all()
    saved=deepcopy(actor.state_dict())
    restored=RAPSImaginationActor(backend,settings)
    restored.load_state_dict(saved)
    assert restored.settings == saved["settings"]
    assert restored.updates == saved["updates"]
    for key in ("actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer"):
        torch.testing.assert_close(restored.state_dict()[key], saved[key])
    from types import SimpleNamespace
    from exploration.raps.raps_gnn_actor import SCHEMA, load
    from exploration.cee_us.cee_us_cabinet import rng_state
    encoder=SimpleNamespace(contract=lambda: {'test_state_dim':67})
    payload=dict(schema=SCHEMA, backend=deepcopy(backend.state_dict()), actor=saved,
                 state_contract=encoder.contract(), action_contract=action_contract(actor.settings),
                 rng=rng_state())
    checkpoint=tmp_path/'macro.pt'
    torch.save(payload,checkpoint)
    assert load(checkpoint,backend,restored,encoder)['action_contract']['action_dim']==5
    payload['action_contract']=action_contract({**DEFAULTS,'skill_set':'split7','latched_gripper':True})
    torch.save(payload,checkpoint)
    with pytest.raises(ValueError,match='contract differs'):
        load(checkpoint,backend,restored,encoder)
    with pytest.raises(ValueError,match='settings differ'):
        actor.load_state_dict({**saved,'settings':{**saved['settings'],'pick_place_primitives':False}})


def test_cli_rejects_macro_flag_for_other_agents(tmp_path):
    from exploration.cee_us.cee_us_cabinet import main
    config=tmp_path/'config.yaml'
    OmegaConf.save(OmegaConf.create({'agent':'icem'}), config)
    with pytest.raises(SystemExit):
        main(['train','--config',str(config),'--pick-place-primitives'])


def test_final_budget_tick_preserves_retention_failure(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import exploration.raps.raps_gnn_actor as module
    class Execution:
        def __init__(self,**kwargs): self.reset()
        def reset(self):
            self.gripper_command=-1.;self.support_stopped_object_ids=[]
            self.motion_attempts=[];self.last_motion_tick={}
        def start(self,*args):
            self.primitive=SimpleNamespace(status="running",phase="lift",done=False)
        def next_action(self,*args): return np.r_[np.zeros(7),-1.].astype(np.float32)
        def record_action(self,*args,**kwargs): pass
        def check_retention(self,*args):
            self.primitive.status="lost_grasp";self.primitive.done=True
        def observe(self,*args): pass
        def finish_motion(self,*args): pass
    class Env:
        action_mode="joint_position"
        def __init__(self): self.env=self
        def reset(self,**kwargs): pass
        def _context(self): return None
        def step(self,low):
            record=SimpleNamespace(duration_low_steps=1,low_actions=[low],full_controller_actions=[low],
                                   status="success",contact_events=[],outcome_metrics={})
            return None,0.,False,False,{'primitive_record':record,'exploration_reward':{}}
    monkeypatch.setattr(module,'PlannedPickPlaceSkillExecution',Execution)
    monkeypatch.setattr(module,'snapshot',lambda env:{})
    settings={**DEFAULTS,'pick_place_primitives':True,'motion_execution':'planner','latched_gripper':True}
    actor=SimpleNamespace(settings=settings,action=lambda state:np.array([1.,0.,0.,0.,0.],np.float32))
    collector=module.SkillCollector(Env(),SimpleNamespace(encode=lambda:np.zeros(67,np.float32)),actor,tmp_path)
    ticks,skills=collector.collect(1,'curiosity',seed=0,stage='test')
    assert ticks[0]['macro_status']=='lost_grasp'
    assert skills[0]['status']=='lost_grasp'
    assert skills[0]['completed'] and skills[0]['trainable']
    assert skills[0]['duration_low_steps']==1

@pytest.mark.parametrize('configured,flag,expected',[
    ('friction',[], 'friction'),('constraint',[],'constraint'),
    ('friction',['--grasp-retention','constraint'],'constraint'),
    ('constraint',['--grasp-retention','friction'],'friction'),
])
def test_retention_cli_precedence(tmp_path,monkeypatch,configured,flag,expected):
    from exploration.cee_us.cee_us_cabinet import main
    import exploration.raps.raps_gnn_actor as module
    path=tmp_path/'mode.yaml'
    OmegaConf.save(OmegaConf.create({'agent':'raps_gnn_actor','raps_actor':{
        'pick_place_primitives':True,'grasp_retention':configured}}),path)
    observed=[]
    monkeypatch.setattr(module,'run',lambda args,cfg:observed.append(cfg.raps_actor.grasp_retention) or 0)
    assert main(['train','--config',str(path)]+flag)==0
    assert observed==[expected]


def test_retention_modes_have_distinct_contracts_and_resume_settings():
    settings={**DEFAULTS,'pick_place_primitives':True,'motion_execution':'planner',
              'latched_gripper':True,'grasp_stop_on_contact':False,
              'hidden_size':8,'actor_layers':1,'critic_layers':1}
    constraint={**settings,'grasp_retention':'constraint'}
    friction_contract=action_contract(settings)
    constraint_contract=action_contract(constraint)
    assert not action_contract_matches(friction_contract,constraint)
    assert not action_contract_matches(constraint_contract,settings)
    assert constraint_contract['pick_place']['grasp_retention']['physics_attachment']
    model=CEEUSBackend(agent_dim=33,action_dim=5,hidden_dim=8,batch_size=4)
    policy=RAPSImaginationActor(model,constraint)
    with pytest.raises(ValueError,match='settings differ'):
        policy.load_state_dict({'settings':settings})
    with pytest.raises(ValueError,match='requires pick_place'):
        RAPSImaginationActor(model,{**constraint,'pick_place_primitives':False})


def test_macro_episode_budget_reaches_native_time_limit_configuration(monkeypatch):
    import exploration.cee_us.cee_us_cabinet as runner
    source=OmegaConf.create({"sim":{},"ppo":{},"max_low_level_steps_per_episode":1000})
    monkeypatch.setattr(runner.OmegaConf,"load",lambda _:source.copy())
    captured=[]
    monkeypatch.setattr("main.build_env",lambda cfg,**kw:captured.append((cfg,kw)) or object())
    runner.make_env(episode_steps=800)
    runner.make_env(action_mode="joint_position",episode_steps=2200,low_level_episode_steps=2200)
    assert captured[0][0].max_low_level_steps_per_episode == 1000
    assert captured[0][0].max_ee_steps_per_episode == 800
    assert captured[1][0].max_low_level_steps_per_episode == 2200
    assert captured[1][0].max_ee_steps_per_episode == 2200
    for value in (0,True,1.5):
        with pytest.raises(ValueError,match="low_level_episode_steps"):
            runner.make_env(low_level_episode_steps=value)
