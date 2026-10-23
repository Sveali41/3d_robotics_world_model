# Fixed CEE-US source

Source: https://github.com/martius-lab/cee-us
Commit: `b0beaed1e9c3fe03d99aa300faae063dc208941e`.
License: upstream MIT, reproduced in LICENSE.

This is the source dependency closure for `GNNForwardEnsembleModel`,
`GraphNeuralNetworkEnsemble`, `Normalizer_v2`, `TorchCuriosityMpcICem`, and
its original colored-noise sampler. The cabinet adapter calls the original
training loop and MPC optimization loop. No robotics simulator is included.
`upstream_manifest.json` records original SHA-256 hashes; `compatibility.patch`
records every changed source file relative to that commit. Empty package
initializers and `allogger.py` are local import/logging scaffolding.

Compatibility changes:

- Namespace imports under `third_party.cee_us.mbrl` to avoid global `mbrl` collisions.
- Use Gymnasium's equivalent Box/Discrete/Env interfaces; no environment registry,
  old Gym robotics, ground-truth simulation, or visualization is imported. The
  MPC wrapper rejects non-Torch forward models. Simulator hooks are unsupported.
- Retain only `Decorator` in helpers and replace optional output logging with a
  no-op logger. Metrics are returned by the cabinet adapter.
- Defer scikit-learn import until the unused dataset-splitting method is called.
  It is not a backend dependency.
- Copy static object fields directly from the input after prediction to avoid
  roundoff from the upstream normalize/denormalize roundtrip. Static dynamics
  and their ensemble disagreement remain exactly zero.
- Handle a zero-length candidate batch before colored-noise FFT. Small test
  fixtures may reuse zero elites; MKL rejects an empty FFT. Production defaults
  reuse three elites and never take this branch.
- Optionally cache node/edge batch indices for CUDA graph capture. This cache
  is enabled only by `exploration.cee_us_cuda_graph.CudaGraphPrediction`; the
  reference path keeps the upstream index construction and prediction math.
  Captures replay the original prediction function and are discarded after
  fitting or loading, so updated normalizer tensors are captured afresh.

Original initialization, MLP/LayerNorm math, parameter independence, loss reduction,
Adam epsilon (1e-4), shuffled non-bootstrap ensemble minibatches, normalization
statistics (sample std), raw next-state curiosity (sample std), elite smoothing,
colored-noise generation, and planning/action-selection order are unchanged.
The upstream planner reuses out buffers across candidate counts; current PyTorch
emits resize deprecation warnings when retained elites enlarge those counts.

Runtime dependencies: numpy, torch, gymnasium, forwardable (0.4.1 tested).
Run focused checks from the repository root:

```
/home/siyao/Apps/anaconda3/envs/rlPractice/bin/python -m pytest -q tests/test_cee_us_backend.py
```
