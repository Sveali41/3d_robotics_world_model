"""RGB JEPA with standard conditional flow matching and upstream LeWM blocks."""
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .official.module import Embedder, MLP, SIGReg, Transformer, ConditionalBlock


DEFAULTS = dict(history_size=3, horizon=3, embed_dim=192, image_size=224,
                patch_size=14, encoder_depth=12, encoder_heads=3,
                encoder_mlp_dim=768, predictor_depth=6, predictor_heads=16,
                predictor_dim_head=64, predictor_mlp_dim=2048, projector_dim=2048,
                dropout=0.1, flow_steps=10, sigreg_knots=17,
                sigreg_projections=1024, sigreg_weight=0.09)


def finite(x, name):
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinity")


class StochasticJEPA(nn.Module):
    def __init__(self, config=None, device='cpu'):
        super().__init__()
        config = dict(config or {})
        unknown = set(config) - DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unknown model options: {sorted(unknown)}")
        self.config = {**DEFAULTS, **config}
        c = self.config
        for key, default in DEFAULTS.items():
            value = c[key]
            if isinstance(default, int) and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{key} must be a positive integer")
        if not 0 <= c['dropout'] < 1 or not 0 <= c['sigreg_weight']:
            raise ValueError("Invalid dropout or SIGReg weight")
        if c['sigreg_knots'] < 2 or c['image_size'] % c['patch_size'] or c['embed_dim'] % c['encoder_heads']:
            raise ValueError("Invalid encoder dimensions or SIGReg knots")
        try:
            from transformers import ViTConfig, ViTModel
        except ImportError as exc:
            raise ImportError("JEPA requires isolated requirements/jepa.txt dependencies") from exc
        d = c['embed_dim']
        self.encoder = ViTModel(ViTConfig(hidden_size=d, num_hidden_layers=c['encoder_depth'],
            num_attention_heads=c['encoder_heads'], intermediate_size=c['encoder_mlp_dim'],
            image_size=c['image_size'], patch_size=c['patch_size']),
            add_pooling_layer=False, use_mask_token=False)
        self.projector = MLP(d, c['projector_dim'], d, norm_fn=partial(nn.BatchNorm1d))
        self.action_encoder = Embedder(input_dim=13, emb_dim=d)
        self.time_encoder = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.position = nn.Parameter(torch.randn(1, c['history_size'] + c['horizon'], d))
        self.predictor = Transformer(d, d, d, c['predictor_depth'], c['predictor_heads'],
            c['predictor_dim_head'], c['predictor_mlp_dim'], c['dropout'], ConditionalBlock)
        self.pred_proj = MLP(d, c['projector_dim'], d, norm_fn=partial(nn.BatchNorm1d))
        self.sigreg = SIGReg(c['sigreg_knots'], c['sigreg_projections'])
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _encode(self, pixels, mask):
        pixels = torch.as_tensor(pixels, device=self.device)
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if pixels.ndim != 5 or pixels.shape[2] != 3 or mask.shape != pixels.shape[:2]:
            raise ValueError("pixels must be [B,T,3,h,w], mask [B,T]")
        if pixels.dtype != torch.uint8:
            raise ValueError("JEPA pixels must be uint8 RGB")
        if not mask.any():
            raise ValueError("At least one valid image is required")
        x = pixels[mask].float() / 255
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        x = F.interpolate((x - mean) / std, (self.config['image_size'],) * 2,
                          mode='bilinear', align_corners=False, antialias=True)
        z = self.projector(self.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])
        out = z.new_zeros(*pixels.shape[:2], self.config['embed_dim'])
        out[mask] = z
        finite(out, 'encoded images')
        return out

    def encode_history(self, pixels, history_mask):
        if pixels.shape[1] != self.config['history_size']:
            raise ValueError("History length differs from model history_size")
        if not torch.as_tensor(history_mask)[:, -1].all():
            raise ValueError("Latest history observation must be valid")
        return self._encode(pixels, history_mask)

    def _velocity(self, history, history_actions, mask, actions, x, time):
        b, h, d = history.shape
        k = actions.shape[1]
        if (h, k, d) != (self.config['history_size'], self.config['horizon'], self.config['embed_dim']):
            raise ValueError("History/future latent dimensions differ from configuration")
        if history_actions.shape != (b, h-1, 13) or actions.shape != (b, k, 13) or mask.shape != (b,h):
            raise ValueError("Invalid action or history mask shape")
        history = history * mask[..., None]
        ha_mask = mask[:, :-1] & mask[:, 1:]
        past_actions = history_actions * ha_mask[..., None]
        action = torch.cat([past_actions, actions.new_zeros(b,1,13), actions], dim=1)
        cond = self.action_encoder(action)
        cond = cond + self.time_encoder(time.reshape(b,1,1)).expand(-1,h+k,-1)
        tokens = torch.cat([history, x], dim=1) + self.position
        velocity = self.predictor(tokens, cond)[:, h:]
        return torch.stack([self.pred_proj(velocity[:, i]) for i in range(k)], dim=1)

    def sample_futures(self, history_latents, history_actions, history_mask, candidate_actions, noise):
        tensors = [torch.as_tensor(v, device=self.device) for v in
                   (history_latents, history_actions, history_mask, candidate_actions, noise)]
        history, past, mask, actions, noise = tensors
        mask = mask.bool()
        b, n, k, d = noise.shape
        if noise.shape != (history.shape[0], n, self.config['horizon'], self.config['embed_dim']):
            raise ValueError("noise must be [B,N,K,D]")
        for name, value in [('history',history),('history actions',past),('actions',actions),('noise',noise)]:
            finite(value, name)
        expand = lambda v: v[:,None].expand(-1,n,*v.shape[1:]).reshape(b*n,*v.shape[1:])
        x = noise.reshape(b*n,k,d).clone()
        for step in range(self.config['flow_steps']):
            t = x.new_full((b*n,), step / self.config['flow_steps'])
            x = x + self._velocity(expand(history),expand(past),expand(mask),expand(actions),x,t) / self.config['flow_steps']
            finite(x, 'sampled futures')
        return x.reshape(b,n,k,d)

    def loss(self, batch):
        c = self.config
        pixels = torch.as_tensor(batch['pixels'], device=self.device)
        hm = torch.as_tensor(batch['history_mask'], device=self.device, dtype=torch.bool)
        fm = torch.as_tensor(batch['future_mask'], device=self.device, dtype=torch.bool)
        if fm.shape != (pixels.shape[0],c['horizon']) or not fm.all():
            raise ValueError("Training requires complete future windows")
        z = self._encode(pixels, torch.cat([hm,fm],dim=1))
        history, target = z[:, :c['history_size']], z[:, c['history_size']:]
        noise = torch.randn_like(target)
        time = torch.rand(target.shape[0], device=self.device)
        xt = (1-time[:,None,None])*noise + time[:,None,None]*target
        past = torch.as_tensor(batch['history_actions'],device=self.device,dtype=z.dtype)
        actions = torch.as_tensor(batch['future_actions'],device=self.device,dtype=z.dtype)
        finite(past, 'history actions'); finite(actions, 'future actions')
        predicted = self._velocity(history,past,hm,actions,xt,time)
        fm_loss = (predicted-(target-noise)).square().mean()
        # Preserve LeWM joint gradients, including the target encoder (no EMA/stop-gradient).
        # Apply the unchanged statistic to valid observations only, never padded images.
        valid = torch.cat([hm,fm],dim=1)
        if valid.all():
            reg = self.sigreg(z.transpose(0, 1))
        else:
            reg = torch.stack([self.sigreg(z[valid[:, i], i].unsqueeze(0))
                               for i in range(z.shape[1]) if valid[:, i].any()]).mean()
        total = fm_loss + c['sigreg_weight'] * reg
        finite(total, 'JEPA loss')
        return total, {'loss':total.detach(), 'flow_loss':fm_loss.detach(), 'sigreg_loss':reg.detach()}
