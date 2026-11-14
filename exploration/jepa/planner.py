"""Mixed categorical/continuous CEM maximizing stochastic latent variance."""
import copy
import time

import numpy as np
import torch

from domain.robocasa.primitives import RAPS_SPLIT_ACTION_SCHEMA as SCHEMA
from .model import finite

DEFAULTS = dict(horizon=3, candidates=64, iterations=3, elites=8,
                num_futures=10, chunk_size=8, gamma=0.8,
                pseudocount=1.0, blend=0.5, std_floor=0.1, initial_std=0.5)


class MixedCEM:
    def __init__(self, model, config=None, seed=0):
        config = dict(config or {})
        if set(config) - DEFAULTS.keys():
            raise ValueError(f"Unknown planner options: {sorted(set(config)-DEFAULTS.keys())}")
        self.config = {**DEFAULTS, **config}
        c = self.config
        for key in ('horizon','candidates','iterations','elites','num_futures','chunk_size'):
            if isinstance(c[key],bool) or not isinstance(c[key],int) or c[key]<1:
                raise ValueError(f"{key} must be a positive integer")
        if c['num_futures']<2 or c['elites']>c['candidates'] or c['horizon'] != model.config['horizon']:
            raise ValueError("Invalid CEM sample counts or horizon")
        if (not 0 <= c['gamma'] <= 1 or not 0 < c['blend'] <= 1 or
            not 0 < c['std_floor'] <= c['initial_std'] or not c['pseudocount']>0):
            raise ValueError("Invalid CEM distribution settings")
        for key,value in c.items():
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite planner option {key}")
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.active = np.zeros((SCHEMA.num_primitives, SCHEMA.num_parameters),dtype=bool)
        for i,name in enumerate(SCHEMA.primitive_names):
            self.active[i, SCHEMA.primitive_name_to_action_idx[name]] = True

    def state_dict(self):
        return {'rng_state':copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state['rng_state'])

    def _sample(self, probabilities, means, stds):
        c = self.config
        ids = np.stack([self.rng.choice(SCHEMA.num_primitives, c['candidates'],p=p) for p in probabilities],axis=1)
        positions = np.arange(c['horizon'])[None,:]
        args = self.rng.normal(means[positions,ids], stds[positions,ids]).clip(-1,1)
        args *= self.active[ids]
        return np.concatenate([np.eye(SCHEMA.num_primitives)[ids],args],axis=-1).astype(np.float32),ids

    @torch.no_grad()
    def plan(self, history):
        started = time.perf_counter()
        c = self.config
        device = self.model.device
        pixels = torch.as_tensor(history['pixels'],device=device)
        mask = torch.as_tensor(history['history_mask'],device=device,dtype=torch.bool)
        past = torch.as_tensor(history['history_actions'],device=device,dtype=torch.float32)
        if pixels.shape[0] != 1:
            raise ValueError("CEM planning expects one history")
        finite(past,'history actions')
        was_training = self.model.training
        self.model.eval()
        try:
            latent = self.model.encode_history(pixels,mask)
            finite(latent,'history latent')
            h, d = c['horizon'],self.model.config['embed_dim']
            noise = torch.as_tensor(self.rng.standard_normal((1,c['num_futures'],h,d)),device=device,dtype=latent.dtype)
            probabilities = np.full((h,SCHEMA.num_primitives),1/SCHEMA.num_primitives)
            means = np.zeros((h,SCHEMA.num_primitives,SCHEMA.num_parameters))
            stds = np.full_like(means,c['initial_std'])
            best_score, best = -np.inf,None
            for _ in range(c['iterations']):
                actions, ids = self._sample(probabilities,means,stds)
                scores=[]
                for start in range(0,c['candidates'],c['chunk_size']):
                    a=torch.as_tensor(actions[start:start+c['chunk_size']],device=device)
                    count=a.shape[0]
                    z=self.model.sample_futures(latent.expand(count,-1,-1),past.expand(count,-1,-1),
                        mask.expand(count,-1),a,noise.expand(count,-1,-1,-1))
                    finite(z,'sampled futures')
                    discount=z.new_tensor([c['gamma']**i for i in range(h)])
                    score=(z.var(dim=1,unbiased=True).mean(dim=-1)*discount).sum(dim=-1)
                    finite(score,'CEM score')
                    scores.extend(score.cpu().tolist())
                scores=np.asarray(scores)
                index=int(scores.argmax())
                if scores[index]>best_score:
                    best_score=float(scores[index]); best=actions[index].copy()
                elite=np.argsort(scores)[-c['elites']:]
                blend=c['blend']
                for pos in range(h):
                    counts=np.bincount(ids[elite,pos],minlength=SCHEMA.num_primitives)+c['pseudocount']
                    probabilities[pos]=(1-blend)*probabilities[pos]+blend*counts/counts.sum()
                    for skill in range(SCHEMA.num_primitives):
                        selected=elite[ids[elite,pos]==skill]
                        if len(selected):
                            args=actions[selected,pos,SCHEMA.num_primitives:]
                            means[pos,skill]=(1-blend)*means[pos,skill]+blend*args.mean(axis=0)
                            stds[pos,skill]=np.maximum(c['std_floor'],(1-blend)*stds[pos,skill]+blend*args.std(axis=0))
            return best[0],dict(score=best_score,candidates_evaluated=c['iterations']*c['candidates'],
                elapsed_s=time.perf_counter()-started, best_sequence=best.tolist(),
                selected_primitive=SCHEMA.primitive_names[int(best[0,:SCHEMA.num_primitives].argmax())])
        finally:
            self.model.train(was_training)
