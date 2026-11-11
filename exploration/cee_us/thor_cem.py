"""Categorical CEM using the upstream GNN rollout and disagreement objective."""
import numpy as np
import torch

from exploration.cee_us.thor_state import ACTIONS


class _Policy:
    def __init__(self, encoder, sequences):
        self.encoder, self.sequences, self.index = encoder, sequences, 0

    def get_action(self, observations, state=None, mode=None):
        actions = self.sequences[:, self.index]
        self.index += 1
        encoded, _ = self.encoder.action_encoding(observations, actions)
        # Upstream rollout_generator expects the batch dimension omitted when p=1.
        return encoded[:, 0] if observations.shape[1] == 1 else encoded


class ThorCEM:
    def __init__(self, backend, encoder, *, horizon=5, candidates=32, iterations=3,
                 elites=8, chunk_size=1, seed=0):
        if min(horizon, candidates, iterations, elites, chunk_size) < 1 or elites > candidates:
            raise ValueError('Invalid CEM parameters')
        self.backend, self.encoder = backend, encoder
        self.horizon, self.candidates, self.iterations = horizon, candidates, iterations
        self.elites, self.chunk_size = elites, chunk_size
        self.rng = np.random.default_rng(seed)
        self.prior = np.asarray([.5 / 8] * 8 + [.5 / 6] * 6)

    def score(self, state, sequences):
        sequences = np.asarray(sequences, dtype=np.int64)
        model = self.backend.model
        old_count, old_horizon = model.num_simulated_trajectories, model.horizon
        scores = []
        try:
            for start in range(0, len(sequences), self.chunk_size):
                batch = sequences[start:start + self.chunk_size]
                if (not model.memory_is_preallocated or
                        model.num_simulated_trajectories != len(batch) or
                        model.horizon != batch.shape[1]):
                    model.num_simulated_trajectories, model.horizon = len(batch), batch.shape[1]
                    model.preallocate_memory()
                obs = torch.as_tensor(state, device=self.backend.device).expand(len(batch), -1)
                policy = _Policy(self.encoder, torch.as_tensor(batch, device=self.backend.device))
                rollout, _ = model.predict_n_steps(start_observations=obs, start_states=None,
                                                   horizon=batch.shape[1], policy=policy)
                predictions = rollout.as_array('next_observations')
                value = predictions.std(dim=1, unbiased=True).sum(dim=(-1, -2))
                if not torch.isfinite(value).all():
                    raise FloatingPointError('Nonfinite CEE-US disagreement')
                scores.extend(value.cpu().numpy().tolist())
        finally:
            model.num_simulated_trajectories, model.horizon = old_count, old_horizon
            if old_count is not None and old_horizon is not None:
                model.preallocate_memory()
            else:
                model.memory_is_preallocated = False
        return np.asarray(scores)

    def plan(self, state):
        probs = np.tile(self.prior, (self.horizon, 1))
        best_score, best_sequence = -np.inf, None
        for _ in range(self.iterations):
            samples = np.stack([self.rng.choice(len(ACTIONS), self.candidates, p=probs[t])
                                for t in range(self.horizon)], axis=1)
            scores = self.score(state, samples)
            best = int(np.argmax(scores))
            if scores[best] > best_score:
                best_score, best_sequence = float(scores[best]), samples[best].copy()
            elite_indices = np.argsort(-scores, kind='stable')[:self.elites]
            elite = samples[elite_indices]
            frequencies = np.stack([np.bincount(elite[:, t], minlength=len(ACTIONS)) / self.elites
                                    for t in range(self.horizon)])
            probs = .1 * probs + .9 * frequencies
        return int(best_sequence[0]), best_score
