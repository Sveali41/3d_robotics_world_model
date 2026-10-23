# Cabinet state-GNN curiosity diagnostic

This experiment reuses the current ManiSkill two-object cabinet and normalized
4D `ee_delta` actions. It uses privileged simulator states, not images. It does
not train PPO or PointWorld. Existing experiments and their checkpoints are
unchanged. The current cabinet, Panda reset, bowl size, contact thresholds,
and 100 Hz physics / 20 Hz control remain in effect.

### Corrected direct control protocol

Panda `ee_delta` now holds the wrist orientation measured at episode reset.
The adapter sends corrective rotation commands using ManiSkill's intrinsic
XYZ convention and existing rotation limits. This replaces zero rotation
deltas, which let measured orientation errors accumulate into later targets.
The four policy outputs and native +/-0.1 m translation bounds are unchanged.
Both compact and dense records store the actual seven-dimensional command.

`max_ee_steps_per_episode: 100` independently limits direct control to five
seconds at 20 Hz; the primitive-count limit only applies to primitive mode.
The low-level step budget and environment termination still apply.

New checkpoints carry `panda-ee-delta-reset-orientation-v2` action semantics.
Legacy models and replay describe different closed-loop dynamics: retain them
as research baselines/offline data, and start new training for this protocol.
Physical `evaluate` rejects checkpoints without the matching action tag;
offline checkpoint loading remains available. This control fix is not evidence
of successful grasping.

Direct-control lift statistics use `continuous_grasp_height_v2`: the reference
height is the pre-step object height at the first instantaneous grasp of an
uninterrupted grasp segment. A lift is recorded while held for at least four
consecutive checks and at least 1 cm above that reference. Each sub-centimetre
control step accumulates; releasing the object clears its reference, and each
episode reset clears all references. Ordinary per-step movement/coverage
metrics retain their original displacement definition. This fixes the older
per-tick lift test, which missed gradual lifts. Existing running processes keep
the code they imported; historical results can be recomputed from logged
positions, held IDs and stable-grasp counts without modifying model data.

## Reference and intentional adaptations

The dependency in `third_party/cee_us` is pinned to martius-lab/cee-us commit
`b0beaed1e9c3fe03d99aa300faae063dc208941e`. Its provenance file records the
selected files and compatibility edits. The source MIT license is retained.
The five-member GNN, trainer, normalization, colored-noise sampler, and iCEM
are reused, rather than replaced with approximations. Dependency compatibility
does not require installing MuJoCo or the original Gym robotics environments.
Install the small additional Python dependency with:

```bash
conda run -n rlPractice python -m pip install forwardable==0.4.1
```

The model has width 128, two MLP layers, one message-passing iteration,
LayerNorm and delta targets. It trains on the whole accumulated replay with
the upstream Adam/normalizer/loss behavior: learning rate 1e-5, decay .001,
batch 125, no bootstrap resampling. Formal collection uses 25 epochs per round.
The upstream optimizer epsilon and independent member shuffles are retained.

iCEM uses 30 steps, 128 candidates, three iterations, ten elites, alpha .1,
initial std .8, noise beta 3.5, elite reuse .3, mean candidate, shifted elites,
and the best elite's first action. At 20 Hz this is 1.5 seconds. Curiosity is
the upstream **sample standard deviation**, summed over raw predicted state
coordinates and time; it is not variance, confidence, or model prediction error.
Existing reward is logged but never used to choose an action. Position
projection is selected independently with `project_goals_to_region`; the
reference and `cee_us_cabinet_cuda_graph.yaml` configurations leave it disabled.
When explicitly enabled, both real execution and imagined GNN
rollouts apply the same projection. This does not enable D's reward changes.
Outside-region trajectories are retained and counted.

The existing cabinet-local bounds are `[-.20, -.25, .04]` to
`[.10, .03, .30]` metres (30 x 28 x 26 cm). Projection constrains TCP position
targets; collision avoidance, IK reachability, and containment of the physical
arm or objects are not guaranteed. Requested/executed targets and projection
trigger counts remain in each transition's outcome metrics. Checkpoints record
the projection geometry and reject loading into a different projection setup.

Panda replaces Fetch; mug and bowl replace cubes. Robot joints and the moving
door are observed and predicted. Object type and local collision-mesh extent
are static features. Fixed cabinet geometry is implicit in this one scene.
No cabinet-layout or visual generalization is claimed. The common random
bootstrap below is a diagnostic adaptation; the reference CEE-US configuration
starts without one.

## Commands

### Optional object approach heuristic

`configs/cee_us_cabinet_60k_heuristic.yaml` matches the completed 60k baseline
except for this opt-in planning score:

`score = original_curiosity + 50 * mean_object(distance_before - distance_after) / 0.1 m`

Distances are from TCP to each selected object's center in the fixed robot-root
frame, computed per ensemble member and time step. The per-step terms are
summed over the planning horizon and averaged over members for selection. Cup
and bowl initially have equal weights; a subset or unequal named weights can
be configured. Stationary proximity earns zero; approaching earns a positive
score and retreating a negative score. Round trips cancel. This initial weight
is an experimental setting, not a validated optimum or a grasp reward.

Only approach progress is enabled. The GNN does not predict contact forces;
real contact, stable grasp, and lifting remain simulator measurements, with no
contact proxy added. The existing environment reward remains diagnostic and
is not used for planning. Region projection stays disabled.

Per-step logs include `planner.object_heuristic_score` for predicted approach
and combined scores, and `object_heuristic_observed` for progress measured from
the true executed transition. Summaries keep the original curiosity score
separate. Checkpoints preserve the heuristic settings and reject mismatches.
The baseline configuration omits `object_heuristic` and retains its prior cost
and sampling behavior. No world-model architecture or training loss changes.

Small wiring validation (272 real steps) and explicit full training:

```bash
python -m exploration.cee_us_cabinet smoke \
  --config configs/cee_us_cabinet_60k_heuristic.yaml --output logs/cee_us_heuristic_smoke
python -m exploration.cee_us_cabinet train \
  --config configs/cee_us_cabinet_60k_heuristic.yaml --output logs/cee_us_heuristic_60k
```

The smoke test establishes wiring only. Assess interaction gains against the
saved baseline using matched budgets and per-object motion/coverage, with
contact/grasp/lift as separate measurements; high heuristic scores alone do
not establish exploration or grasp success.

Fresh curiosity-only run with 60,000 training transitions (1,000 random
bootstrap + 59 x 1,000 curiosity), followed by 1,000 frozen evaluation steps:

```bash
python -m exploration.cee_us_cabinet train \
  --config configs/cee_us_cabinet_60k.yaml --output logs/cee_us_curiosity_60k
```

This uses corrected reset-orientation control, 100-step episodes, and no region
projection. It does not load legacy data or provide a matched random baseline.
`progress.json` records collection/training/evaluation stages; each completed
round writes a checkpoint and interaction summary. Predictions are evaluated
on the held-out on-policy evaluation trajectories, without model updates.

Run from the project root. CUDA must be visible to the process; an unavailable
CUDA device fails explicitly. Use `--device cpu` only when intentionally running
a CPU diagnostic.

```bash
# 256 random + 8 curious + update + 8 curious = 272 real control steps.
# Fits for 2 epochs, then 1 epoch; checks disk checkpoint and RNG recovery.
conda run -n rlPractice python -m exploration.cee_us_cabinet smoke

# Explicit formal experiment: 9000 real control steps total.
conda run -n rlPractice python -m exploration.cee_us_cabinet compare

# Frozen checkpoint evaluation, no training. Use --policy random for a
# predictor holdout collected independently of curiosity selection.
conda run -n rlPractice python -m exploration.cee_us_cabinet evaluate \
  --checkpoint logs/<run>/curiosity/round_003.pt --policy curiosity

# Recompute CSV/JSON screening without simulation or training.
conda run -n rlPractice python -m exploration.cee_us_cabinet report \
  --output logs/<compare-run>
```

Outputs default to a new timestamped `logs/cee_us_*` directory. `--output` sets
an explicit **new** directory, except for `report`, which reads an existing run.
The independent configuration is `configs/cee_us_cabinet.yaml`; `--config`
can select an explicit configuration for an intentional protocol variant.
No command automatically expands budgets or starts additional seeds.

## State, replay, and checkpoints

The 66-value state is `[global32, mug_dynamic12, bowl_dynamic12,
mug_static5, bowl_static5]`. Global state contains qpos9, qvel9, TCP position3,
XYZ Euler3, TCP linear/angular velocity3+3, door angle1 and velocity1.
Each dynamic object has position3, XYZ Euler3 and linear/angular velocity3+3.
Static features are mug/bowl one-hot2 and actual collision extents3. Poses and
vectors are in the fixed controller-root frame; velocity fields are multiplied
by the 0.05-second control interval once. Euler angles follow the XYZ convention
recorded in the state contract; branch discontinuities remain a possible
modeling limitation. No contact or future object state is supplied to planning.

Each real tick logs requested and executed actions, before/after state, episode
boundary, contacts, grasp/lift labels, failure, reward components and planner
diagnostics. Only actual executed actions train the model. Failed and no-contact
transitions stay in replay. Training never invents reset transitions. Prediction
evaluation never makes a multi-step window cross episode boundaries.

Checkpoints include model, optimizer, normalizers, iCEM distribution/elite
history, random-policy sequence, RNG state, state contract and replay index;
the neighboring `.replay.npz` stores transitions. They do not restore simulator
physics state. Evaluation starts a fresh episode and resets planner history.
Checkpoints are trusted local pickle files; do not load untrusted checkpoints.

## Formal comparison and interpretation

One shared 1000-step random bootstrap is fitted, then the complete learned
initial state is copied to both arms. Each arm collects three 1000-step rounds,
training on its own accumulated replay after every round. The random arm takes
the first sequence from the upstream colored-noise sampler and shifts it,
without model-based selection. Both arms finally collect 1000 evaluation steps
with frozen models. Physical accounting is `1000 + 2*(3*1000 + 1000) = 9000`.
The random arm's independent evaluation trajectories form the shared predictor
holdout for both final models. Evaluation data never enters training.

`report.json` and `summary.csv` compare per-1000-step contact/motion rates,
per-object interaction, displacement direction/magnitude coverage, repetitions,
outside endpoints and failures. Coverage uses the existing 1.5 cm per-transition
threshold; it can miss slow cumulative motion, so retain raw trajectories for
inspection. Recovery-range coverage prevents falling objects from counting as
the sole improvement. Curiosity logs separate joint, TCP, door and object
contributions to expose optimization of robot prediction errors alone.

Predictor evaluation reports endpoint pose errors at 1/10/30 steps, a zero-delta
baseline, moved-object errors separately, and one-step disagreement/error
correlation. Missing moving samples are reported explicitly, not treated as
successful interaction modeling. Horizons are member-consistent autoregressive
rollouts, not teacher-forced forecasts or ensemble-mean rollouts.

The screening passes only when curious evaluation exceeds random evaluation in
both recovery-range motion efficiency and coverage area, with motion for both
objects. Single-seed/fixed-layout screening establishes neither reliable grasping
nor continual learning or generalization. Smoke success establishes wiring only.

## Execution acceleration

The original configuration remains `configs/cee_us_cabinet.yaml` with 128
candidates and `execution: reference`. Two explicit presets are available:

| Configuration | Execution | Candidates | Meaning |
| --- | --- | --- | --- |
| `configs/cee_us_cabinet_cuda_graph.yaml` | CUDA Graph | 128 | Execution optimization; original search budget |
| `configs/cee_us_cabinet_fast.yaml` | CUDA Graph | 64 | Smaller search budget; exploration quality must be checked |

CUDA Graph replays the original prediction operations, with cached graph indices
and cloned outputs so autoregressive history cannot alias later predictions.
The model, loss, normalizers, ensemble size, 30-step horizon, three iCEM
iterations, feedback frequency, actions, physics, and 25 training epochs stay
the same. The captures are discarded automatically before/after training and
when loading weights, because normalization tensors can change storage. Captures
are not serialized in checkpoints. CUDA Graph requires CUDA and fails explicitly
if unavailable or capture fails; it never silently switches implementations.
First use after an update captures the graph and costs more than steady replay.

Benchmark on saved real states without physics or training:

```bash
conda run -n rlPractice python -m exploration.cee_us_speed benchmark \
  --checkpoint logs/<run>/curiosity/round_003.pt \
  --output logs/cee_us_speed_benchmark
```

This compares CPU with 1/4 threads, reference CUDA with 128 candidates, and
CUDA Graph with 128/64 candidates, on the same weights and 12 saved states.
It saves mean/P95 decision latency, rollout latency, inclusive capture/warmup
time, Torch-only memory peaks, checkpoint hash, warnings, JSON and CSV. GPU
timings synchronize the device. An explicit `--device cpu` benchmarks only CPU
cases. Different CPU/CUDA random streams make this a timing comparison, not
a numerical equivalence test; separate fixed-candidate CUDA tests check that.
The upstream rollout sub-timer measures host dispatch time, without waiting for
GPU completion. Use `full_action_seconds` for CUDA speed comparisons; the
sub-timer is not GPU kernel duration.

To check whether 64 candidates retains interaction quality, explicitly run:

```bash
conda run -n rlPractice python -m exploration.cee_us_speed compare \
  --checkpoint logs/<run>/curiosity/round_003.pt \
  --output logs/cee_us_speed_quality
```

This evaluates the same frozen model with optimized128 and optimized64, 1000
real steps each, matched initial seeds and no training. `comparison.json` and
`comparison.csv` record timing, interaction coverage, per-object movement and
failures; per-step logs are retained. `quality_not_degraded` requires both motion
efficiency and recovery coverage area to be at least the 128-candidate values,
with motion for both objects. This is a single-seed screen, not statistical
noninferiority. No preset is selected automatically from reward or speed alone.
Checkpoint weights and normalizers are reused; planner history is reset when
switching candidate counts, rather than presented as an exact resume.

To use acceleration in a later collection/training experiment:

```bash
# Original planning budget, faster execution. This explicitly starts 9000 steps.
conda run -n rlPractice python -m exploration.cee_us_cabinet compare \
  --config configs/cee_us_cabinet_cuda_graph.yaml

# Smaller planning budget; use after the quality screen above.
conda run -n rlPractice python -m exploration.cee_us_cabinet compare \
  --config configs/cee_us_cabinet_fast.yaml
```

### Runtime optimizations and paired profiling

Set `runtime_optimizations: true` in an experiment YAML to combine finite-gradient
checks into one host synchronization per optimizer step, incrementally build the
six replay arrays, and flush transition logs every 32 steps. Missing or false
keeps the original path. This is independent of `execution: cuda_graph` and does
not alter the training schedule, model, sampling, loss, planner or episode limit.
Episode boundaries, normal close and handled exceptions flush logs; abrupt
process termination can lose the last buffered records. Checkpoints retain the
existing schema, full replay, compression and save frequency.

```bash
conda run -n rlPractice python -m exploration.cee_us_speed profile \
  --checkpoint logs/<run>/curiosity/round_059.pt \
  --output logs/cee_us_runtime_profile --physical-steps 64
```

This performs three paired one-epoch fits on isolated checkpoint/optimizer
copies, alternates mode order, compares complete model/normalizer/Adam states,
and times 12 planning states after two warmups. Both CUDA paths use the existing
prediction CUDA Graph acceleration. It never modifies the source checkpoint or
its replay. The optional physical check is capped at 64 steps per mode (128
total); omit `--physical-steps` for zero physical steps. Numerical disagreement
produces a failed report and nonzero exit status.

`profile.json` and `profile.csv` record synchronized timings, graph capture,
training subparts, array construction, full replay/checkpoint saves and memory.
Host RSS is the process-lifetime high-water mark, not isolated per-mode memory;
Torch GPU peaks exclude simulator allocations. The physical `simulation_s`
timer includes the full wrapped `env.step()`, including contact/reward logging.
Microbenchmarks measure code throughput, not grasp learning or full-run speedup.
