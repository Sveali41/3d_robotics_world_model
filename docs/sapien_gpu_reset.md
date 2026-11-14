# Isolated SAPIEN 3.0.3 sparse GPU reset correction

The GPU collector needs subset resets that preserve every active peer. The original SIF remains unchanged at `/data/data/siyao/3d_world_model.sif`; the correction is an explicitly selected shared-library overlay in `.runtime/sapien_sparse_patch`. CPU reference runs continue to use the original runtime.

## Source and defects

The installed version is SAPIEN 3.0.3, matching [tag commit 731622eac5b140b320076c8a1b6eb4b553c3ccd4](https://github.com/haosulab/SAPIEN/tree/731622eac5b140b320076c8a1b6eb4b553c3ccd4). The pinned `src/physx/physx_system.cpp` SHA256 is `8173ce5960fe79b3657eed60c79a13834a786b409b1eaf56264169be7d65d27b`. The installed `physx_system.h` matches upstream exactly, SHA256 `95a0653c9dd45702287a59aa9bec902a74b26d812e929340a552308ca8dd5df7`.

The [source patch](../patches/sapien-3.0.3-sparse-joint-indices.patch) corrects six sparse overloads:

- Joint position, velocity, force and position/velocity targets validate the caller's indices but submit the full internal index pointer with only the caller's count. They now pass `indices.ptr`.
- Sparse rigid-body conversion packs data at consecutive positions but retains global `srcIndex` values in `PxGpuActorPair`. PhysX documents that `applyActorData` reads those source indices. The correction retains the sparse actor pairs and regenerates the data scratch buffer in global order before submitting only the selected actors. It reuses the existing compiled CUDA converters, with their original quaternion, offset, velocity and stream conventions. The full conversion prepares scratch data; it does not apply state to unselected bodies.

The second defect appeared after the five joint fixes passed: resetting the second environment moved its cup to the bowl's position. Correcting rigid-body source indexing restores the intended initial state. This is distinct from the original live-peer joint perturbations.

## Build and launch

The wheel already contains matching SAPIEN, PhysX and graphics headers/libraries. Compiling the six C++ methods against that ABI avoids rebuilding CUDA kernels, PhysX or the renderer. Only the missing Eigen 3.4.0 headers were downloaded (approximately 2.6 MB, archive SHA256 `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72`). No dependency upgrade or CUDA/Vulkan SDK installation was necessary.

From the repository root:

```bash
apptainer exec /data/data/siyao/3d_world_model.sif \
  python scripts/build_sapien_sparse_patch.py
```

This verifies pinned source/header/archive hashes, applies the repository patch without fuzz, extracts exactly six methods and compiles with the container's GCC 11.4 toolchain using C++20, CUDA-enabled class declarations and the original `libsapien.so`. The manifest records the upstream commit, input hashes, original binaries, generated source, helper binary and compiler command. Optional `--source` and `--eigen` arguments accept the same hash-verified cached inputs.

The data mount disallows executable mappings. The launcher verifies the original ABI binaries and helper hash, copies the helper into an executable `/tmp` cache identified by its full SHA256, prepares the installed OIDN SONAME paths and preserves the existing container CUDA paths:

```bash
apptainer exec --nv /data/data/siyao/3d_world_model.sif env \
  PYTHONPATH="/tmp/wm_gated_numpy126:$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_patched_sapien.py --runtime .runtime/sapien_sparse_patch -- \
  -m exploration.diagnostics.collect_cabinet_grasp_sequences \
  --checkpoint logs/raps_compact6_train_400k_episode40s_centered_20261026T173840/curiosity/round_398.pt \
  --output logs/cabinet_split7_gpu_smoke --stage smoke \
  --skill-set split7 --collector gpu_batch --num-envs 2
```

The NumPy 1.26.4 shim is separate from the SIF and must exist as described in README. The launcher imports Torch before globally loading the overlay, and loads the overlay before SAPIEN. Using `LD_PRELOAD` for this entire native dependency tree caused `torch.load` to fail; an import-only smoke did not detect that incompatibility. Torch-first loading passes checkpoint deserialization and real sparse writes. Module/script execution preserves arguments and multiprocessing module identity. GPU benchmark subprocesses explicitly reenter this launcher. Existing user `LD_PRELOAD` values are preserved.

The collection manifest records the original extension and `libsapien` hashes plus the complete overlay build manifest and its hash. An incompatible original library or altered helper fails clearly. Runtime preflight exercises actual sparse joint and rigid-body writes before case assignment; the collector also rejects any reset that changes live-peer sensor records without a control tick. Do not bypass these checks.

## Validation and limits

Original GPU4 reset probes and source are retained in [reset_debug](../logs/gpu_split7_validation_20261111/reset_debug/). All 16 original probes changed peer state67 (maximum absolute difference0.000118839); indexed joint submission still changed peers 8/16. Forward kinematics alone preserved peers exactly 16/16. The old incomplete correction and rejected performance samples remain historical evidence, not validated datasets.

The six-method correction passed targeted native tests with GPU 2 and GPU 4: complete 67D peer state equality, robot qpos/qvel, object pose/velocity, controller targets, elapsed steps, peer RNG preservation and selected-slot initial-state restoration. Repeated high-index and noncontiguous reset groups `[3]`, `[0,2]`, `[1]`, `[2,3]` preserve the other moving environments exactly. State encoding also matches the original single-environment encoder. These checks do not assert CPU/GPU trajectory equivalence or identical results across batch sizes.

Final collection, compatibility and throughput acceptance are recorded in `docs/LOG.md` and the separate patched-run report. Formal 972-case collection and training require a separate launch; the correction does not tune grasp success, thresholds or physical timing.
