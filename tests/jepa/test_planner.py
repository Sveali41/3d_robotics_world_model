import json

import numpy as np
import pytest
import torch

from exploration.jepa.planner import MixedCEM


class ToyModel(torch.nn.Module):
    def __init__(self, target=0, identical=False, bad=False):
        super().__init__()
        self.config=dict(horizon=3,embed_dim=2)
        self.device='cpu'
        self.target=target; self.identical=identical; self.bad=bad
        self.calls=[]; self.encodes=0

    def encode_history(self,pixels,mask):
        self.encodes+=1
        return torch.zeros(1,3,2)

    def sample_futures(self,h,past,mask,actions,noise):
        self.calls.append((actions.clone(),noise.clone()))
        if self.bad:
            return noise * float('nan')
        if self.identical:
            return noise * 0 + actions[:,:,0][:,None,:,None]
        scale=actions[:,:,self.target][:,None,:,None]
        return noise * scale


def history():
    return dict(pixels=np.zeros((1,3,3,8,8),dtype=np.uint8),
                history_actions=np.zeros((1,2,13),dtype=np.float32),
                history_mask=np.ones((1,3),dtype=bool))


def config():
    return dict(candidates=64,iterations=2,elites=8,num_futures=4,chunk_size=8)


def test_model_dependent_optimum_mask_and_fixed_noise():
    actions=[]
    for target in (0,5):
        model=ToyModel(target)
        planner=MixedCEM(model,config(),seed=5)
        action,stats=planner.plan(history())
        actions.append(action)
        assert action[:7].argmax()==target
        assert model.encodes==1
        assert model.training
        json.dumps(stats)
        for candidates,noise in model.calls:
            ids=candidates[:,:,:7].argmax(-1).numpy()
            args=candidates[:,:,7:].numpy()
            assert (args[~planner.active[ids]]==0).all()
            assert (np.abs(args)<=1).all()
            torch.testing.assert_close(noise,model.calls[0][1])
    assert not np.array_equal(*actions)


def test_identical_outputs_zero_variance_resume():
    model=ToyModel(identical=True)
    planner=MixedCEM(model,config(),seed=1)
    state=planner.state_dict()
    a,stats=planner.plan(history())
    assert stats['score']==0
    planner.load_state_dict(state)
    again,stats2=planner.plan(history())
    np.testing.assert_array_equal(a,again)
    assert stats.pop('elapsed_s')>=0
    assert stats2.pop('elapsed_s')>=0
    assert stats==stats2


def test_nonfinite_fails_and_restores_mode():
    model=ToyModel(bad=True)
    with pytest.raises(ValueError,match='NaN'):
        MixedCEM(model,config()).plan(history())
    assert model.training


def test_unbiased_variance_definition():
    model=ToyModel()
    planner=MixedCEM(model,dict(candidates=8,iterations=1,elites=2,num_futures=4,chunk_size=8),seed=2)
    _,stats=planner.plan(history())
    a,n=model.calls[0]
    z=n*a[:,:,0][:,None,:,None]
    expected=(z.var(1,unbiased=True).mean(-1)*torch.tensor([1.,.8,.64])).sum(-1).max().item()
    assert stats['score']==pytest.approx(expected)
