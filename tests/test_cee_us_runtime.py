"""Runtime optimizations must not change sampling, training, or recorded data."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from exploration.cee_us_backend import CEEUSBackend
from exploration import cee_us_cabinet as runner


def data_rows(count):
    rng = np.random.default_rng(17)
    rows = []
    for i in range(count):
        state = rng.normal(size=66).astype(np.float32)
        nxt = state.copy()
        nxt[:56] += rng.normal(size=56).astype(np.float32)*.01
        rows.append(dict(state=state, next_state=nxt, action=rng.uniform(-1,1,4).astype(np.float32),
                         episode_id=i//4, terminated=False, truncated=i%4==3))
    return rows


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: assert_tree_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_tree_equal(x,y)
    else:
        assert a==b


def test_cached_replay_append_views_capacity_and_arm_isolation():
    rows=data_rows(13)
    a,b=runner.allocate_replay(13),runner.allocate_replay(13)
    for cache in [a,b]: runner.append_replay(cache,0,rows[:4])
    count=runner.append_replay(a,4,rows[4:10])
    assert count==10
    runner.append_replay(a,count,rows[10:])
    for key,value in runner.replay_arrays(rows).items():
        np.testing.assert_array_equal(value,runner.replay_view(a,13)[key])
        assert value.dtype==a[key].dtype
        assert np.shares_memory(a[key],runner.replay_view(a,13)[key])
    for key,value in runner.replay_arrays(rows[:4]).items():
        np.testing.assert_array_equal(value,runner.replay_view(b,4)[key])
    assert not np.shares_memory(a['states'],b['states'])
    with pytest.raises(ValueError,match='capacity'): runner.append_replay(a,13,rows[:1])
    with pytest.raises(ValueError): runner.replay_view(a,14)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_training_batches_gradients_normalizers_and_adam_match(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    replay=runner.replay_arrays(data_rows(13))
    traces=[];snapshots=[];fits=[]
    for optimized in [False,True]:
        backend=CEEUSBackend(device=device,seed=0,hidden_dim=8,batch_size=5,
                             runtime_optimizations=optimized)
        trace=dict(batches=[],gradients=[])
        original=backend.model.loss
        def loss(batch):
            trace['batches'].append({k:np.asarray(v).copy() for k,v in batch.items()})
            return original(batch)
        backend.model.loss=loss
        backend.model.optimizer.register_step_pre_hook(lambda *args:trace['gradients'].append(
            [p.grad.detach().cpu().clone() for p in backend.model.model.parameters()]))
        np.random.seed(9);torch.manual_seed(9)
        backend.profile_timings={}
        fits.append(backend.train(replay,epochs=2))
        assert backend.profile_timings['normalization_s']>=0
        assert backend.profile_timings['model_training_s']>0
        traces.append(trace);snapshots.append(copy.deepcopy(backend.state_dict()))
    assert fits[0]==fits[1]
    assert fits[0]['updates']==6
    assert_tree_equal(traces[0],traces[1])
    for key in ['model','optimizer','normalizers']:
        assert_tree_equal(snapshots[0][key],snapshots[1][key])


@pytest.mark.parametrize('optimized',[False,True])
@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_nonfinite_gradients_stop_before_parameter_update(optimized,bad,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    backend=CEEUSBackend(hidden_dim=8,runtime_optimizations=optimized,device=device)
    parameters=list(backend.model.model.parameters())
    before=[p.clone() for p in parameters]
    for p in parameters: p.grad=torch.zeros_like(p)
    parameters[-1].grad.reshape(-1)[0]=bad
    with pytest.raises(FloatingPointError,match='nonfinite gradients'):
        backend.model.optimizer.step()
    assert backend.updates==0
    assert not backend.model.optimizer.state
    for a,b in zip(parameters,before): assert torch.equal(a,b)


def test_cached_checkpoint_roundtrip_keeps_schema_and_replay(tmp_path):
    backend=CEEUSBackend(hidden_dim=8)
    rows=data_rows(7)
    cache=runner.allocate_replay(10);runner.append_replay(cache,0,rows)
    enc=SimpleNamespace(contract=lambda:{'test':True})
    cfg=OmegaConf.create(dict(runtime_optimizations=True))
    timings={}
    path=tmp_path/'model.pt'
    payload=runner.save_checkpoint(path,backend,enc,cfg,rows,rows[-1]['next_state'],'test',
                                  arrays=runner.replay_view(cache,7),timings=timings)
    assert set(timings)=={'array_build_s','replay_save_s','checkpoint_save_s'}
    with np.load(path.with_suffix('.replay.npz')) as actual:
        for k,v in runner.replay_arrays(rows).items(): np.testing.assert_array_equal(actual[k],v)
    loaded=runner.read_checkpoint(path,backend)
    assert payload['schema']==loaded['schema']=='cabinet-cee-us-state-v1'
    assert_tree_equal(payload['backend'],loaded['backend'])


@pytest.mark.parametrize('fail_at',[None,35])
def test_batched_logs_flush_episode_tail_and_handled_exception(tmp_path,monkeypatch,fail_at):
    from tests.test_cee_us_cabinet import Env,Model
    env=Env()
    def env_step(action):
        if fail_at is not None and env.tick+1==fail_at: raise RuntimeError('handled failure')
        result=Env.step(env,action)
        return (*result[:3],env.tick==34,result[4])
    env.step=env_step
    monkeypatch.setattr(runner,'snapshot',lambda env:{})
    original_open=Path.open
    flushes=[]
    class Handle:
        def __init__(self,handle): self.handle=handle;self.lines=0
        def __enter__(self): return self
        def write(self,value): self.lines+=1;return self.handle.write(value)
        def flush(self): flushes.append(self.lines);return self.handle.flush()
        def __exit__(self,*args): return self.handle.__exit__(*args)
    def opening(path,*args,**kwargs):
        handle=original_open(path,*args,**kwargs)
        return Handle(handle) if path.name=='transitions.jsonl' and args and args[0]=='a' else handle
    monkeypatch.setattr(Path,'open',opening)
    # Keep the diagnostic episode counter monotonic for the injected failure.
    backend=Model()
    collector=runner.Collector(env,backend,SimpleNamespace(encode=lambda:np.full(66,env.tick)),tmp_path,
                               runtime_optimizations=True,profile=True)
    if fail_at:
        # A non-terminating version of the fixture step preserves its actual record.
        def failing_step(action):
            if env.tick+1==fail_at: raise RuntimeError('handled failure')
            result=Env.step(env,action)
            return (*result[:3],False,result[4])
        env.step=failing_step
        with pytest.raises(RuntimeError,match='handled failure'):
            collector.collect(37,'random',seed=0,stage='test')
        expected=34
    else:
        collector.collect(37,'random',seed=0,stage='test')
        assert flushes==[32,34,37]
        expected=37
    rows=[json.loads(x) for x in (tmp_path/'transitions.jsonl').read_text().splitlines()]
    assert len(rows)==expected
    assert collector.timings['json_serialization_s']>0
