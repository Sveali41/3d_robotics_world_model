import pytest
import torch

from exploration.jepa.model import StochasticJEPA


def tiny_config():
    return dict(image_size=16,patch_size=8,embed_dim=12,encoder_depth=1,
                encoder_heads=3,encoder_mlp_dim=24,predictor_depth=1,
                predictor_heads=2,predictor_dim_head=6,predictor_mlp_dim=24,
                projector_dim=24,dropout=0.,flow_steps=2,sigreg_projections=8)


@pytest.fixture
def model():
    torch.manual_seed(7)
    return StochasticJEPA(tiny_config()).eval()


def batch(b=2):
    return dict(pixels=torch.randint(0,256,(b,6,3,16,16),dtype=torch.uint8),
                history_mask=torch.ones(b,3,dtype=torch.bool),
                future_mask=torch.ones(b,3,dtype=torch.bool),
                history_actions=torch.zeros(b,2,13),future_actions=torch.zeros(b,3,13))


def test_joint_loss_backward(model):
    model.train()
    loss,metrics=model.loss(batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.embeddings.patch_embeddings.projection.weight.grad.abs().sum()>0
    assert model.action_encoder.patch_embed.weight.grad is not None
    assert set(metrics)=={'loss','flow_loss','sigreg_loss'}


def test_fixed_noise_distinct_draws_and_causality(model):
    # Nonzero gates ensure the causal check is not vacuous at initialization.
    for block in model.predictor.layers:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight,std=.1)
    b=batch(1)
    with torch.no_grad():
        h=model.encode_history(b['pixels'][:,:3],b['history_mask'])
        noise=torch.randn(1,4,3,12)
        call=lambda a,n: model.sample_futures(h,b['history_actions'],b['history_mask'],a,n)
        z=call(b['future_actions'],noise)
        assert torch.equal(z,call(b['future_actions'],noise))
        assert not torch.equal(z[:,0],z[:,1])
        changed=b['future_actions'].clone(); changed[:,2]=1
        after=call(changed,noise)
        torch.testing.assert_close(z[:,:,:2],after[:,:,:2],rtol=0,atol=0)
        assert not torch.equal(z[:,:,2],after[:,:,2])


def test_padding_ignored(model):
    b=batch(1); b['history_mask'][:,:2]=False
    a=b['pixels'][:,:3].clone(); z=model.encode_history(a,b['history_mask'])
    a[:,:2]=255-a[:,:2]
    torch.testing.assert_close(z,model.encode_history(a,b['history_mask']))
    assert z[:,:2].count_nonzero()==0


def test_nonfinite_and_incomplete_windows(model):
    b=batch(); b['future_actions'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='NaN'):
        model.loss(b)
    b=batch(); b['future_mask'][0,-1]=False
    with pytest.raises(ValueError,match='complete'):
        model.loss(b)
    with pytest.raises(ValueError,match='uint8'):
        model.encode_history(b['pixels'][:,:3].float(),b['history_mask'])


def test_target_receives_gradient(model):
    b=batch()
    captured=[]
    original=model._encode
    def encode(*args):
        z=original(*args); z.retain_grad(); captured.append(z); return z
    model._encode=encode
    model.loss(b)[0].backward()
    assert captured[0].grad[:,3:].abs().sum()>0


def test_training_velocity_has_no_future_action_leakage(model):
    model.train()
    for block in model.predictor.layers:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight,std=.1)
    history=torch.randn(2,3,12)
    past=torch.randn(2,2,13)
    actions=torch.randn(2,3,13)
    x=torch.randn(2,3,12)
    mask=torch.ones(2,3,dtype=torch.bool)
    time=torch.full((2,),.5)
    before=model._velocity(history,past,mask,actions,x,time)
    actions[:,2]+=10
    after=model._velocity(history,past,mask,actions,x,time)
    torch.testing.assert_close(before[:,:2],after[:,:2],rtol=0,atol=0)
    assert not torch.equal(before[:,2],after[:,2])


def test_planning_does_not_mutate_model(model):
    from exploration.jepa.planner import MixedCEM
    b=batch(1)
    history=dict(pixels=b['pixels'][:,:3],history_actions=b['history_actions'],history_mask=b['history_mask'])
    before={name:tensor.clone() for name,tensor in model.state_dict().items()}
    action,stats=MixedCEM(model,dict(candidates=4,iterations=1,elites=2,num_futures=2,chunk_size=2)).plan(history)
    assert action.shape==(13,)
    assert stats['selected_primitive']
    assert len(stats['best_sequence'])==3
    assert not model.training
    for name,tensor in model.state_dict().items():
        assert torch.equal(tensor,before[name])
    assert all(p.grad is None for p in model.parameters())
