# A5000 split7 collection performance acceptance

Measured on 2026-11-11 using the original SIF plus the isolated SAPIEN 3.0.3 correction. Full lossless trajectories and disk writes stayed enabled. Each GPU configuration recorded 4096 active controls; the unchanged scalar baseline stopped at 60 collection seconds with 1352 controls. All use the same saved 128-case pool in case-ID order, fixed seeds and unchanged 800-control case budgets, 100/20 Hz and 67D/13D contracts. Total time includes environment initialization, writer drain and NPZ export, excluding process/Python startup. Initialization is separately measured.

| Collector | Environments | Total seconds | Controls/s | Device peak MiB | Valid endpoints | Closed cases |
|---|---:|---:|---:|---:|---:|---:|
| scalar | 1 | 65.685 | 20.58 | not retained | 57 | 9 |
| gpu_batch | 16 | 33.230 | 123.26 | 3005 | 255 | 32 |
| gpu_batch | 32 | 23.780 | 172.24 | 3005 | 220 | 17 |
| gpu_batch | 64 | 15.711 | 260.71 | 3069 | 269 | 0 |
| gpu_batch | 128 | 13.901 | 294.65 | 3069 | 233 | 0 |

**Recommendation: 128 environments** among the four measured sizes. Observed end-to-end control throughput was 294.65/s versus 20.58/s scalar (**14.32x**), exceeding the 5x engineering target in this bounded measurement. Collection-only throughput was 477.05/s. Peak whole-device memory 3069 MiB was 12.5% of 24564 MiB, below the 80% ceiling. The 64-environment rate was more than 5% slower, so the smaller-size tie rule does not apply.

These are shared-machine observations, not isolated or repeated performance estimates. Another terminal started three authorized RAPS jobs at 16:04; they shared GPU/CPU during these measurements. Memory counts the whole device, including those jobs. The earlier valid CPU baseline was reused because its collection behavior did not change. Prefix lengths differ physically, and 64/128 stop before any case finishes; zero completed-case throughput there does not estimate sustained case throughput. This is the fastest tested configuration under the stated bounds, not proof of a global maximum or a full 972-case duration.

## Correctness and compatibility

- Six actual native tests passed with GPU 2/4, including noncontiguous subset resets, exact full 67D peer state, robot joint state, object motion, controller targets, clocks/RNG and selected-slot restoration. Joint and rigid-body sparse capability are checked at startup.
- Patched GPU 2 eight-case smoke recorded 729 controls and 67 valid endpoints. Both slots refilled, with 6 replacement assignments. All before/after states were exactly continuous; every exported endpoint matched its raw segment. Four cases completed successfully; unchanged thresholds and CPU/GPU physics differences remain. GPU smoke successes are not evidence of a grasp improvement.
- All 16384 benchmark controls passed the same exact full-record/endpoint audit, including 32 refills at GPU 16 and 17 at GPU 32, with zero discontinuities. Every numeric pool loads with pickle disabled. Interrupted prefixes retain their own attempts and incomplete skills cannot become endpoints. Endpoint counts in the table exclude completed zero-duration skills; the original raw timing files retain their earlier counters for traceability. The final collector counter now follows the export rule.
- The existing viewer read all 729 smoke controls. Historical JSONL and earlier numeric chunks remain covered by the recorded compatibility tests. GPU video replay still fails explicitly rather than presenting CPU physics as exact reproduction.
- Identical-source smoke resume preserved all 65 case/chunk files byte-for-byte, created no new attempts and executed zero controls. Interrupted fresh-attempt recovery, chunk boundaries, writer failure/backpressure and episode uniqueness passed focused tests in the implementation log.

The source snapshots accompanying measured runs remain frozen. After measurement, only metadata generation was extended to record the actual ManiSkill commit/patch hash, and the zero-duration endpoint metric was corrected; no physical behavior changed. The recorded runtime provenance supplement identifies ManiSkill e406239d2b1c554e4b99010bc9b40ad2bbd91f05 and its existing SIF patches. Exact resume deliberately rejects changed source snapshots; do not resume a measured artifact with the final metadata-only source revision.

## Timing and remaining costs

At GPU 128, initialization was 5.20s and total measured runtime 13.90s. Physics/controller processing took 2.19s; sensor reading 0.58s; row construction 1.09s; writer enqueue/copy/backpressure 1.69s; final drain 1.74s; reset 0.10s; host transfer 0.012s; export 0.067s. These measured stages identify initialization and host/writer work as substantial remaining costs. Enqueue/drain includes copying, queue waits and process completion, so it cannot all be attributed to disk throughput. More environments were outside the approved calibration grid. No further optimization or full run was started.

## Reusable command

Build once as described in [sapien_gpu_reset.md](sapien_gpu_reset.md). The following is ready for a separately started formal collection; it was **not executed** by this task:

```bash
apptainer exec --nv /data/data/siyao/3d_world_model.sif env \
  PYTHONPATH="/tmp/wm_gated_numpy126:$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_patched_sapien.py --runtime .runtime/sapien_sparse_patch -- \
  -m exploration.diagnostics.collect_cabinet_grasp_sequences \
  --checkpoint logs/raps_compact6_train_400k_episode40s_centered_20261026T173840/curiosity/round_398.pt \
  --output logs/cabinet_split7_gpu_collection --stage collect \
  --skill-set split7 --collector gpu_batch --num-envs 128 --export-every-cases 64
```

Repeat the identical command with `--resume` after interruption. CPU reference uses the original SIF without the patch launcher and `--collector scalar`. Source/settings/native-build changes require a new output directory.

Evidence: [audited report](../artifacts/gpu_split7_patched_20261111/report.json), [step/endpoint audit](../artifacts/gpu_split7_patched_20261111/benchmark_audit.json), [smoke audit](../artifacts/gpu_split7_patched_20261111/smoke_audit.json), [resume audit](../artifacts/gpu_split7_patched_20261111/resume_audit.json), [viewer](../artifacts/gpu_split7_patched_20261111/viewer/trajectories.html), and [original worker metrics/logs](../logs/gpu_split7_patched_20261111/benchmark/). The saved artifact-local benchmark/audit scripts reproduce the bounded recheck without changing historical data.
