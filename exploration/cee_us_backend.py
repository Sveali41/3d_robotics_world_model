"""Goal-free cabinet adapter for the fixed, vendored CEE-US implementation."""
from copy import deepcopy
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from third_party.cee_us.mbrl import torch_helpers
from third_party.cee_us.mbrl.controllers.abstract_controller import OpenLoopPolicy
from third_party.cee_us.mbrl.controllers.mpc_torch_curiosity import TorchCuriosityMpcICem
from third_party.cee_us.mbrl.models.gnn_ensemble import GNNForwardEnsembleModel
from third_party.cee_us.mbrl.rolloutbuffer import Rollout, RolloutBuffer, SimpleRolloutBuffer


class _MeasuredPlanner(TorchCuriosityMpcICem):
    def simulate_trajectories(self, **kwargs):
        started = perf_counter()
        result = super().simulate_trajectories(**kwargs)
        self.rollout_seconds += perf_counter() - started
        return result

    def trajectory_cost_fn(self, cost_fn, rollout_buffer, out):
        result = super().trajectory_cost_fn(cost_fn, rollout_buffer, out)
        heuristic = None
        if self.object_heuristic_fn is not None:
            heuristic, _ = self.object_heuristic_fn(
                rollout_buffer.as_array('observations'),
                rollout_buffer.as_array('next_observations'))
            heuristic = heuristic.sum(dim=-1)
            out.sub_(heuristic)
        if not torch.isfinite(out).all():
            raise FloatingPointError("CEE-US produced nonfinite trajectory costs")
        best = out.mean(-1).argmin()
        self.selected_std = self.stds_of_means_[best].clone()
        if heuristic is not None:
            self.selected_approach_progress_score = heuristic[best].mean().clone()
        return result


class CEEUSBackend:
    normalizer_names = (
        'input_normalizer_agent', 'input_normalizer_obj_dyn',
        'input_normalizer_obj_stat', 'action_normalizer',
        'output_agent_normalizer', 'output_obj_normalizer',
    )

    def __init__(self, agent_dim=32, object_dyn_dim=12, object_stat_dim=5,
                 num_objects=2, device='cpu', seed=0, *, ensemble_size=5,
                 hidden_dim=128, horizon=30, candidates=128, iterations=3,
                 elites=10, batch_size=125, execution='reference', action_projection=None,
                 object_heuristic=None, runtime_optimizations=False, action_dim=4,
                 object_names=None):
        if isinstance(action_dim, bool) or not isinstance(action_dim, (int, np.integer)) or action_dim < 1:
            raise ValueError('action_dim must be a positive integer')
        if action_projection is not None and action_dim != 4:
            raise ValueError('action_projection requires action_dim=4')
        if execution not in {'reference', 'cuda_graph'}:
            raise ValueError('execution must be reference or cuda_graph')
        if execution == 'cuda_graph' and torch.device(device).type != 'cuda':
            raise ValueError('cuda_graph execution requires a CUDA device')
        if object_names is None:
            object_names = ('cabinet_item_16', 'cabinet_item_17') if num_objects == 2 else ()
        if object_names and (len(object_names) != num_objects or
                             len(set(object_names)) != num_objects or
                             not all(isinstance(name, str) and name for name in object_names)):
            raise ValueError('object_names must contain one unique nonempty name per object')
        self.object_names = tuple(object_names)
        self.object_heuristic = self._validate_object_heuristic(object_heuristic, self.object_names)
        if self.object_heuristic is not None and (agent_dim not in (32, 33) or object_dyn_dim != 12 or len(self.object_names) != num_objects):
            raise ValueError('object_heuristic requires agent_dim=32 or 33, object_dyn_dim=12, and object_names for all objects')
        torch_helpers.device = torch.device(device)
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.runtime_optimizations = bool(runtime_optimizations)
        self.profile_timings = None
        self.action_projection = deepcopy(action_projection)
        if action_projection is not None:
            self._projection = {}
            for name, shape in (
                ('root_to_cabinet_rotation', (3, 3)),
                ('root_to_cabinet_translation', (3,)), ('lower', (3,)),
                ('upper', (3,)), ('pos_lower', (3,)), ('pos_upper', (3,)),
            ):
                value = self._array(action_projection[name], shape, name)
                self._projection[name] = torch.as_tensor(value, device=self.device)
            if not torch.all(self._projection['lower'] <= self._projection['upper']):
                raise ValueError('action_projection lower must be <= upper')
            if not torch.all(self._projection['pos_lower'] < self._projection['pos_upper']):
                raise ValueError('action_projection pos_lower must be < pos_upper')
            self._cabinet_to_root_rotation = torch.linalg.inv(self._projection['root_to_cabinet_rotation'])
        self.dynamic_dim = agent_dim + num_objects * object_dyn_dim
        self.state_dim = self.dynamic_dim + num_objects * object_stat_dim
        self.ensemble_size = ensemble_size
        self.env = SimpleNamespace(
            agent_dim=agent_dim, object_dyn_dim=object_dyn_dim,
            object_stat_dim=object_stat_dim, nObj=num_objects,
            action_space=spaces.Box(-1., 1., (self.action_dim,), dtype=np.float32),
            observation_space=spaces.Box(-np.inf, np.inf, (self.state_dim,), dtype=np.float32),
            obs_preproc=lambda x: x, obs_postproc=lambda x: x,
            cost_fn=lambda obs, actions, next_obs: torch.zeros_like(next_obs[..., 0]),
        )
        self.model = GNNForwardEnsembleModel(
            env=self.env, normalize_w_running_stats=False,
            model_params=dict(n=ensemble_size, hidden_dim=hidden_dim, num_layers=2,
                              num_message_passing=1, layer_norm=True),
            train_params=dict(epochs=1, batch_size=batch_size, learning_rate=1e-5,
                              weight_decay=.001, bootstrapped=False),
        )
        self.planner = _MeasuredPlanner(
            env=self.env, forward_model=self.model, horizon=horizon,
            num_simulated_trajectories=candidates, cost_along_trajectory='sum',
            use_async_action=False, logging=False,
            action_sampler_params=dict(alpha=.1, elites_size=elites,
                opt_iterations=iterations, init_std=.8, use_mean_actions=True,
                keep_previous_elites=True, shift_elites_over_time=True,
                finetune_first_action=False, fraction_elites_reused=.3,
                colored_noise=True, noise_beta=3.5, relative_init=True,
                execute_best_elite=True, use_ensemble_cost_std=False),
        )
        self.planner.object_heuristic_fn = (
            self._object_heuristic_scores if self.object_heuristic is not None else None)
        self.planner.elite_samples = None
        self.random_sequence = None
        self.updates = 0
        self._fit_updates = 0
        self.model.optimizer.register_step_pre_hook(self._check_gradients)
        self.execution = execution
        self._original_predict = self.model.predict
        if self.action_projection is not None:
            self.model.predict = self._predict_with_region
        self.prediction_accelerator = None
        if execution == 'cuda_graph':
            from exploration.cee_us_cuda_graph import CudaGraphPrediction
            self.prediction_accelerator = CudaGraphPrediction(self.model)
            self.model.predict = self.prediction_accelerator.predict

    @staticmethod
    def _validate_object_heuristic(setting, object_names=('cabinet_item_16', 'cabinet_item_17')):
        if setting is None:
            return None
        if not isinstance(setting, dict) or set(setting) - {'approach_weight', 'distance_scale_m', 'object_weights'}:
            raise ValueError('object_heuristic must be a dict with approach_weight, distance_scale_m, object_weights')
        result = dict(approach_weight=50.0, distance_scale_m=0.1,
                      object_weights={'cabinet_item_16': 1.0, 'cabinet_item_17': 1.0})
        result.update(deepcopy(setting))
        for name in ('approach_weight', 'distance_scale_m'):
            try:
                result[name] = float(result[name])
            except (TypeError, ValueError):
                raise ValueError(f'object_heuristic {name} must be finite') from None
            if not np.isfinite(result[name]) or result[name] < 0 or (name == 'distance_scale_m' and result[name] == 0):
                raise ValueError(f'object_heuristic {name} must be finite and ' +
                                 ('positive' if name == 'distance_scale_m' else 'nonnegative'))
        weights = result['object_weights']
        if not isinstance(weights, dict) or not weights or set(weights) - set(object_names):
            raise ValueError('object_heuristic object_weights must target configured object_names')
        try:
            weights = {name: float(value) for name, value in weights.items()}
        except (TypeError, ValueError):
            raise ValueError('object_heuristic object_weights must be finite and nonnegative') from None
        if not all(np.isfinite(value) and value >= 0 for value in weights.values()) or not np.isfinite(sum(weights.values())) or sum(weights.values()) <= 0:
            raise ValueError('object_heuristic object_weights must be finite, nonnegative, and have positive sum')
        result['object_weights'] = weights
        return result

    def _object_heuristic_scores(self, observations, next_observations):
        setting = self.object_heuristic
        progress = {}
        for index, name in enumerate(self.object_names):
            start = self.env.agent_dim + index * self.env.object_dyn_dim
            if name in setting['object_weights']:
                before = observations[..., 18:21] - observations[..., start:start + 3]
                after = next_observations[..., 18:21] - next_observations[..., start:start + 3]
                progress[name] = torch.linalg.vector_norm(before, dim=-1) - torch.linalg.vector_norm(after, dim=-1)
        weights = setting['object_weights']
        weighted = sum(progress[name] * weight for name, weight in weights.items()) / sum(weights.values())
        return weighted * (setting['approach_weight'] / setting['distance_scale_m']), progress

    def observed_object_heuristic(self, state, next_state):
        if self.object_heuristic is None:
            return {}
        before = torch.as_tensor(self._array(state, (self.state_dim,), 'state'), device=self.device)
        after = torch.as_tensor(self._array(next_state, (self.state_dim,), 'next_state'), device=self.device)
        score, progress = self._object_heuristic_scores(before, after)
        return dict(approach_progress_score=float(score),
                    per_object_progress_m={name: float(value) for name, value in progress.items()})

    def project_actions(self, observations, actions):
        if self.action_projection is None:
            return actions
        p = {name: value.to(actions) for name, value in self._projection.items()}
        position = observations[..., 18:21]
        delta = p['pos_lower'] + (actions[..., :3] + 1) * .5 * (p['pos_upper'] - p['pos_lower'])
        local_target = (position + delta) @ p['root_to_cabinet_rotation'].T + p['root_to_cabinet_translation']
        local_target = torch.maximum(torch.minimum(local_target, p['upper']), p['lower'])
        target = (local_target - p['root_to_cabinet_translation']) @ self._cabinet_to_root_rotation.to(actions).T
        xyz = (2 * (target - position - p['pos_lower']) / (p['pos_upper'] - p['pos_lower']) - 1).clamp(-1, 1)
        gripper = actions[..., 3:].expand(*xyz.shape[:-1], 1)
        return torch.cat((xyz, gripper), dim=-1)

    def _predict_with_region(self, *, observations, states, actions):
        return self._original_predict(observations=observations, states=states,
                                      actions=self.project_actions(observations, actions))

    def _check_gradients(self, optimizer, args, kwargs):
        if self.runtime_optimizations:
            gradients = [p.grad for p in self.model.model.parameters() if p.grad is not None]
            if gradients and not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
                invalid = [name for name, p in self.model.model.named_parameters()
                           if p.grad is not None and not torch.isfinite(p.grad).all()]
                raise FloatingPointError(f'CEE-US training produced nonfinite gradients: {invalid}')
        else:
            for parameter in self.model.model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('CEE-US training produced nonfinite gradients')
        self.updates += 1
        self._fit_updates += 1

    def _array(self, value, shape, name):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite with shape {shape}; got {value.shape}')
        return value

    def predict(self, states, actions):
        states = np.asarray(states, dtype=np.float32)
        states = self._array(states, (len(states), self.state_dim), 'states')
        actions = self._array(actions, (len(states), self.action_dim), 'actions')
        obs = torch.as_tensor(states, device=self.device)[None].expand(self.ensemble_size, -1, -1)
        act = torch.as_tensor(actions, device=self.device)[None].expand(self.ensemble_size, -1, -1)
        result = self.model.predict(observations=obs, states=None, actions=act)[0].cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError('CEE-US prediction is nonfinite')
        return result

    def rollout(self, states, actions):
        states = np.asarray(states, dtype=np.float32)
        states = self._array(states, (len(states), self.state_dim), 'states')
        actions = np.asarray(actions, dtype=np.float32)
        actions = self._array(actions, (len(states), actions.shape[1], self.action_dim), 'actions')
        old_count, old_horizon = self.model.num_simulated_trajectories, self.model.horizon
        old_predict = self.model.predict
        try:
            # Offline rollouts evaluate actions already executed by the environment.
            if self.action_projection is not None:
                self.model.predict = self._original_predict
            self.model.num_simulated_trajectories, self.model.horizon = actions.shape[:2]
            self.model.preallocate_memory()
            result, _ = self.model.predict_n_steps(
                start_observations=torch.as_tensor(states, device=self.device),
                start_states=None, horizon=actions.shape[1],
                policy=OpenLoopPolicy(torch.as_tensor(actions, device=self.device)),
            )
            result = result['next_observations'].permute(1, 0, 2, 3).cpu().numpy().copy()
            if not np.isfinite(result).all():
                raise FloatingPointError('CEE-US rollout is nonfinite')
            return result
        finally:
            self.model.predict = old_predict
            self.model.num_simulated_trajectories, self.model.horizon = old_count, old_horizon
            self.model.preallocate_memory()

    def train(self, replay, epochs):
        # Captures reference normalizer storage; recomputing statistics replaces it.
        if self.prediction_accelerator is not None:
            self.prediction_accelerator.invalidate()
        states = np.asarray(replay['states'], dtype=np.float32)
        states = self._array(states, (len(states), self.state_dim), 'states')
        if len(states) < 2 or epochs < 1:
            raise ValueError('CEE-US fitting requires at least two transitions and epochs >= 1')
        batch = dict(observations=states,
                     actions=self._array(replay['actions'], (len(states), self.action_dim), 'actions'),
                     next_observations=self._array(replay['next_states'], states.shape, 'next_states'),
                     rewards=np.zeros(len(states), dtype=np.float32))
        if not np.array_equal(states[:, self.dynamic_dim:], batch['next_observations'][:, self.dynamic_dim:]):
            raise ValueError('Object static fields must remain unchanged across transitions')
        buffer = RolloutBuffer(rollouts=[Rollout.from_dict(**batch)])
        def timed(label, operation):
            if self.profile_timings is None:
                return operation()
            if self.device.type == 'cuda':
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(self.device)
                start.record()
            began = perf_counter()
            result = operation()
            if self.device.type == 'cuda':
                end.record()
                end.synchronize()
                self.profile_timings[label.replace('_s', '_gpu_s')] = start.elapsed_time(end) / 1000
            self.profile_timings[label] = perf_counter() - began
            return result

        timed('normalization_s', lambda: self.model.update_normalizer(buffer))
        for name in self.normalizer_names:
            norm = getattr(self.model, name)
            if not torch.isfinite(norm.mean).all() or not torch.isfinite(norm.std).all():
                raise FloatingPointError(f'Nonfinite CEE-US normalizer: {name}')
        self.model.epochs = int(epochs)
        self._fit_updates = 0
        try:
            metrics = timed('model_training_s', lambda: self.model.train(buffer, maybe_update_normalizer=False))
        finally:
            if self.prediction_accelerator is not None:
                self.prediction_accelerator.invalidate()
        loss = float(metrics['epoch_train_loss'])
        if not np.isfinite(loss):
            raise FloatingPointError('CEE-US training loss is nonfinite')
        return dict(loss=loss, updates=self._fit_updates, total_updates=self.updates, finitegrad=True)

    def reset_planner(self, state):
        state = self._array(state, (self.state_dim,), 'state')
        self.planner.beginning_of_rollout(observation=state, mode='train')
        self.planner.last_action = None
        self.random_sequence = None

    def action(self, state):
        state = self._array(state, (self.state_dim,), 'state')
        started = perf_counter()
        self.planner.rollout_seconds = 0.
        action = self.planner.get_action(state, state=None)
        action = self._array(action, (self.action_dim,), 'action').copy()
        self.planner.last_action = torch.as_tensor(action, device=self.device).clone()
        std = self.planner.selected_std
        components = dict(joint=float(std[:, :18].sum()), tcp=float(std[:, 18:30].sum()),
                          door=float(std[:, 30:32].sum()), objects=float(std[:, 32:self.dynamic_dim].sum()))
        diagnostics = dict(planning_seconds=perf_counter()-started,
                            candidate_rollout_seconds=self.planner.rollout_seconds,
                            curiosity_score=float(std.sum()), score_decomposition=components)
        if self.object_heuristic is not None:
            score = float(self.planner.selected_approach_progress_score)
            diagnostics['object_heuristic_score'] = dict(
                approach_progress_score=score, combined_score=diagnostics['curiosity_score'] + score)
        return action, diagnostics

    def random_action(self, state):
        state = self._array(state, (self.state_dim,), 'state')
        # Sample under the original initial distribution, independent of fitted iCEM means.
        saved_mean, saved_std = self.planner.mean_.clone(), self.planner.std_.clone()
        try:
            self.planner.reset_mean(self.planner.mean_, True)
            self.planner.reset_std(self.planner.std_, True)
            if self.random_sequence is None:
                self.random_sequence = self.planner.sample_action_sequences(state, 1)[0]
            action = self.random_sequence[0].clone()
            tail = self.planner.sample_action_sequences(state, 1, time_slice=(-1, None))[0]
            self.random_sequence = torch.cat((self.random_sequence[1:], tail), dim=0)
            return self._array(action.cpu().numpy(), (self.action_dim,), 'random action').copy()
        finally:
            self.planner.mean_.copy_(saved_mean)
            self.planner.std_.copy_(saved_std)

    @staticmethod
    def _move_tensors(value, device):
        if torch.is_tensor(value):
            return value.detach().to(device).clone()
        if isinstance(value, dict):
            return {k: CEEUSBackend._move_tensors(v, device) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(CEEUSBackend._move_tensors(v, device) for v in value)
        return deepcopy(value)

    def state_dict(self):
        payload = dict(
            model=self.model.model.state_dict(), optimizer=self.model.optimizer.state_dict(),
            action_dim=self.action_dim,
            normalizers={n: getattr(self.model, n).state_dict() for n in self.normalizer_names},
            planner={n: getattr(self.planner, n, None) for n in
                     ('mean_', 'std_', 'last_action', 'was_reset', 'forward_model_state')},
            random_sequence=self.random_sequence, updates=self.updates,
            action_projection=self.action_projection, object_heuristic=self.object_heuristic,
            elite_samples=None if self.planner.elite_samples is None else self.planner.elite_samples.buffer,
        )
        return self._move_tensors(payload, 'cpu')

    def load_state_dict(self, payload, *, restore_planner=True):
        if payload.get('action_dim', 4) != self.action_dim:
            raise ValueError('Checkpoint action_dim does not match backend action_dim')
        if payload.get('object_heuristic') != self.object_heuristic:
            raise ValueError('Checkpoint object_heuristic does not match backend object_heuristic')
        if payload.get('action_projection') != self.action_projection:
            raise ValueError('Checkpoint action_projection does not match backend action_projection')
        if self.prediction_accelerator is not None:
            self.prediction_accelerator.invalidate()
        payload = self._move_tensors(payload, self.device)
        self.model.model.load_state_dict(payload['model'])
        self.model.optimizer.load_state_dict(payload['optimizer'])
        for name, state in payload['normalizers'].items():
            getattr(self.model, name).load_state_dict(state)
        if restore_planner:
            for name, state in payload['planner'].items():
                setattr(self.planner, name, state)
            elite = payload['elite_samples']
            self.planner.elite_samples = None if elite is None else SimpleRolloutBuffer(rollouts=elite)
            self.random_sequence = payload['random_sequence']
        else:
            # Changing candidate count is a new planning experiment, not a resume.
            self.planner.reset_mean(self.planner.mean_, self.planner.relative_init)
            self.planner.reset_std(self.planner.std_, self.planner.relative_init)
            self.planner.elite_samples = None
            self.planner.last_action = None
            self.planner.forward_model_state = None
            self.planner.was_reset = False
            self.random_sequence = None
        self.updates = payload['updates']
