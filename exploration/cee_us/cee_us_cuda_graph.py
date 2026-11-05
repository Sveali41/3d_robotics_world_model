"""Opt-in CUDA graph replay of the unchanged CEE-US prediction function."""

import torch
from time import perf_counter


class CudaGraphPrediction:
    """Capture each input shape; invalidate after fitting or loading model state."""

    def __init__(self, model):
        self.device = torch.device(model.device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('CEE-US cuda_graph execution requires an available CUDA device')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.original_predict = model.predict
        self.model = model.model
        model.model.cache_batch_indices = True
        self._graphs = {}
        self.profile_timings = None

    def invalidate(self):
        """Discard captures that retain old normalizer tensors or model storage."""
        self._graphs.clear()

    @torch.no_grad()
    def predict(self, *, observations, states, actions):
        # The upstream GNN is stateless and ignores the planner's states argument.
        for name, value in (('observations', observations), ('actions', actions)):
            if not torch.is_tensor(value) or not value.is_cuda:
                raise ValueError(f'CEE-US cuda_graph {name} must be a CUDA tensor')
            if value.device != self.device:
                raise ValueError(f'CEE-US cuda_graph {name} must be on {self.device}')
        key = tuple((tuple(x.shape), x.dtype, x.device) for x in (observations, actions))
        if key not in self._graphs:
            if self.profile_timings is not None:
                torch.cuda.synchronize(self.device)
                began = perf_counter()
            self._graphs[key] = self._capture(observations, actions)
            if self.profile_timings is not None:
                torch.cuda.synchronize(self.device)
                self.profile_timings.setdefault('capture_wall_seconds', []).append(perf_counter() - began)
        graph, static_obs, static_actions, outputs, edge_list = self._graphs[key]
        static_obs.copy_(observations)
        static_actions.copy_(actions)
        graph.replay()
        # Rollout generation retains old observations after the next prediction.
        return tuple(value.clone() if torch.is_tensor(value) else value for value in outputs)

    def _capture(self, observations, actions):
        # These buffers and cached graph indices may also be used outside inference_mode.
        with torch.inference_mode(False), torch.no_grad(), torch.cuda.device(observations.device):
            static_obs = observations.clone()
            static_actions = actions.clone()
            current = torch.cuda.current_stream()
            warmup = torch.cuda.Stream()
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    self.original_predict(observations=static_obs, states=None, actions=static_actions)
            current.wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    outputs = self.original_predict(observations=static_obs, states=None, actions=static_actions)
            except RuntimeError as error:
                raise RuntimeError('CEE-US CUDA graph capture failed; use execution=reference or resolve the CUDA error') from error
        # Upstream replaces this tensor when batch size changes. CUDA graphs
        # retain its address, so each capture must keep its index storage alive.
        edge_list = self.model.edge_list
        return graph, static_obs, static_actions, outputs, edge_list
