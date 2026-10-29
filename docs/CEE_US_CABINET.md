# Cabinet exploration baselines

Only three maintained experiment configurations remain. Historical run snapshots,
logs, checkpoints, replay and figures remain in `logs/`; they are not Git inputs.

| Baseline | Configuration | Policy / model | Budget |
|---|---|---|---|
| Direct EE | `configs/cee_us_cabinet_120k_total_heuristic.yaml` | 4D EE + original iCEM + GNN disagreement and approach heuristic | 119k training + 1k evaluation |
| Primitive, disagreement + heuristic | `configs/raps_gnn_actor_cabinet_400k_episode40s_centered.yaml` | compact6 + GNN + imagination actor/critic | 399k training + 1k evaluation |
| Primitive, heuristic only | `configs/raps_gnn_actor_cabinet_400k_heuristic_only.yaml` | same primitive pipeline, disagreement reward weight 0 | 399k training + 1k evaluation |

The recorded EE run currently contains 89k collected training steps; the config
specifies its original 120k target, not a claim that all target steps completed.
Primitive arms use 40 s episodes and centered initial TCP. EE uses the existing
5 s default. Compare matching real control-step budgets and disclose these
reset/episode differences; this is not a controlled action-space-only ablation.

## Commands

```bash
# Short acceptance run; existing entry caps smoke at 300 real control steps.
python -m exploration.cee_us_cabinet smoke --config configs/cee_us_cabinet_120k_total_heuristic.yaml --output logs/ee_smoke

# Explicit training, never triggered automatically by configuration cleanup.
python -m exploration.cee_us_cabinet train --config configs/cee_us_cabinet_120k_total_heuristic.yaml --output logs/ee_new
python -m exploration.cee_us_cabinet train --config configs/raps_gnn_actor_cabinet_400k_episode40s_centered.yaml --output logs/primitive_combined_new
python -m exploration.cee_us_cabinet train --config configs/raps_gnn_actor_cabinet_400k_heuristic_only.yaml --output logs/primitive_heuristic_new
```

The default `--config` selects the EE baseline. Pass a config explicitly for
primitive experiments. `--device cpu` is an explicit diagnostic option; CUDA
failures never silently fall back to another policy or device.

For primitive continuation, pass `train --checkpoint PATH` and a new output
folder with the matching config. The existing strict loader restores network,
optimizer, normalization and RNG states, checks state/action/controller
contracts, and checks the saved replay against its JSONL prefix. Only total
rounds/evaluation budget may change. Physics resets at each collection round;
mid-episode simulation continuation is not claimed.

## Maintained code and dependencies

`exploration/cee_us_cabinet.py` is the common entry and EE collector;
`exploration/raps_gnn_actor.py` provides the primitive actor branch.
`cee_us_backend.py`, `cee_us_state.py`, `cee_us_metrics.py` and the pinned vendored
CEE-US code provide model fitting, state encoding and diagnostics. Official RAPS
actor/distribution/lambda-return code and `domain/robocasa/skill_execution.py`
remain dependencies. Do not remove them as unused scripts.

`configs/exploration.yaml` supplies the common environment/reward settings.
`main.py` builds the environment and wrappers. `diagnose_cabinet.py` and
`diagnostic_metrics.py` provide shared snapshots, source evidence and metrics;
these modules remain imported by the baselines. Speed/CUDA Graph tools and their
regression tests remain to validate the EE execution path. Other project
PointWorld/data/scene workflows are separate from this exploration cleanup and
are preserved.

Six skills are move, side grasp (legacy name `top_grasp`), lift, lower (`drop`),
open gripper and close gripper. There is no open/close-door skill. Low-level
execution is 100 Hz simulation, 20 Hz control. The primitive model predicts one
complete skill endpoint; all actual control ticks are also saved. Incomplete
skills do not become complete-skill model samples.

The primitive GNN state is 67D: 33 robot/global/controller-memory, 24 object
dynamics and 10 static object attributes. The original EE state is 66D without
command memory. Velocities are multiplied by 0.05 s; positions use the fixed
robot root frame. Five independent GNN members predict dynamic deltas and keep
static attributes unchanged. Primitive imagination uses a shared-state sample
standard deviation, excludes command memory, and propagates the ensemble mean.
The original EE planner instead propagates each model member's own trajectory.

Training uses the full real history and reference normalization/optimizer
protocol. Approach heuristic is 50 times distance progress divided by 0.1 m,
with equal cup/bowl weights. Environment task reward is logged, not used by the
exploration actor or iCEM. Heuristic-only still trains a GNN and actor/critic;
its intrinsic disagreement contribution is zero, although raw disagreement is
logged. No region projection is enabled in the three baselines.

## Results and interpretation

Keep contact **time fraction**, confirmed grasp events and lift events distinct.
Confirmed grasp requires a closed command and real grasp detector for four
continuous control ticks. A lift event requires at least 1 cm rise within that
bout. Missing or failed trajectories remain in logs; contact does not establish
stable manipulation.

Fixed diagnostic prediction error uses the independent script trajectories
retained under `logs/ee_primitive_wm_validation_20261028` and its source files.
Its 106 skill segments include 66 moving object endpoints; it is not a balanced
general-purpose benchmark. EE rolls out recorded low-level controls, while the
primitive models predict those same physical skill endpoints. Training loss is
normalized on an evolving replay and is not a fixed-test accuracy metric.

Existing coverage, interaction and WM plots remain under
`logs/ee_primitive_heuristic_comparison_20261028`,
`logs/heuristic_latest_comparison_20261028` and
`logs/wm_loss_curves_20261029`. Historical config snapshots inside retained runs
are preserved even when their source config filenames have been retired.

Downstream task actor training, enumerated multi-skill planning and single-box
transfer entrypoints/configs/tests have been removed from the maintained source
set. Reintroducing them requires an explicit new experiment protocol.

## Frozen position generalization (same two objects)

The existing `evaluate` entry accepts `--object-shift-xy DX DY` in meters.
Both actors move by the same cabinet-local horizontal offset at every reset;
+X is deeper into this fixed, unrotated cabinet. Height, collision geometry,
robot reset and state/action dimensions do not change. The default is zero.
This override is restricted to evaluation; it does not relax checkpoint contract
checks or create another training configuration. `--evaluation-steps N` sets an
explicit physical budget for this frozen test.

```bash
python -m exploration.cee_us_cabinet evaluate \
  --config configs/raps_gnn_actor_cabinet_400k_episode40s_centered.yaml \
  --checkpoint logs/raps_compact6_train_400k_episode40s_centered_20261026T173840/curiosity/round_398.pt \
  --object-shift-xy 0.03 0 --evaluation-steps 800 --output logs/position_shift_test
```

Compare with `--object-shift-xy 0 0` using the same checkpoint and budget.
The stochastic actor starts from the checkpoint RNG state, and both conditions
use evaluation seed 100000. No parameters, optimizers or normalization statistics
are updated. Predictions use complete real skills only, at 1/3/5 skill horizons.
These on-policy diagnostic trajectories differ between conditions: compare
errors to each condition's own no-motion baseline, not as a common fixed test
set. If no moving samples occur, explicitly report that interaction dynamics
cannot be assessed. One episode per position is a screening, not robust
position generalization evidence or causal attribution to actor vs WM.

## Old/new replay actor–critic continuation ablation

`python -m exploration.raps_replay_compare` adds an explicit experiment, reusing
the source checkpoint configuration rather than adding baseline YAML variants.
The environment contains the same cup and bowl shifted +3 cm into the cabinet.
The default source is the final 399k disagreement+heuristic `round_398.pt`.
Its paired `.replay.npz` contains 5,061 complete primitive transitions; the
399k real-tick history also includes incomplete skills not eligible for this
endpoint replay.

```bash
# Targeted acceptance only: exactly 200 real control steps across both arms.
python -m exploration.raps_replay_compare smoke --output logs/replay_ablation_smoke
# Explicit formal experiment; smoke never starts this automatically.
python -m exploration.raps_replay_compare compare --output logs/replay_ablation_60k
# Regenerate common held-out WM tests and derived interaction curves (no physics).
python -m exploration.raps_replay_compare report --output logs/replay_ablation_60k
```

Both arms load identical old WM weights, normalizers, Adam state and update
counter. `continue` also loads the full actor/critic/target-critic learner,
including optimizer states and update count; `fresh` initializes all policy
learner state from seed 0. Thus this ablates the whole policy learner, not only
actor weights. All three components keep learning. The original strict baseline
resume path is unchanged; this experiment uses its own initialization path.

One shared 1k random bootstrap is collected at the new positions, then each arm
collects 59×1k new ticks. Old replay is immutable; new data are kept independently
per arm. No evaluation data enter either training buffer. Each round samples 250
old and 250 new complete skills (with replacement only when the pool is smaller
than 250), updates normalizers on this 500-row dataset and runs the original WM
training for 25 epochs with batch 125: exactly 100 optimizer steps. Sample indices
are saved. Actor/critic perform 100 updates, each with exactly 32 old and 32 new
nonterminal imagination starting states drawn from their full respective pools.
The existing 5-skill imagination/lambda-return/REINFORCE implementation is reused;
all imagined disagreement and approach rewards are recomputed using the current
WM. Target critic synchronization retains the original update-count schedule.
This is a new balanced-replay protocol, not the historical full-replay baseline.

Frozen stochastic evaluation at 0/10k/30k/60k uses 1k ticks each at original AND
shifted positions, with seed 100000. Evaluation checks a digest of all saved
network/optimizer/normalizer state before/after and restores training RNG. The
0-data test precedes bootstrap fitting. Formal physical budget: shared 1k +
2×59k training + 2 arms×4 milestones×2 scenes×1k evaluation = 135k ticks.
The smoke uses 160 shared bootstrap ticks, two 8-tick evaluations per arm and
4 new ticks per arm, with shortened updates, exactly 200 ticks; it does not
validate grasp dynamics or transfer performance.

Per-round checkpoints preserve each arm's new replay/low-level history,
model/actor/critic/optimizer/RNG state and a separate NumPy sampler state. The
source old replay is referenced by absolute path and SHA256 rather than copied
into every checkpoint. This runner currently refuses existing output paths and
does not offer automatic interrupted-run continuation; the saved states provide
an explicit restoration basis. Do not use ordinary baseline `train --checkpoint`
as a replacement for this experiment's balanced-replay update protocol.

`learning.jsonl`, `learning_curves.csv/png`, evaluation JSONL and `report.json`
record contact time fraction, first/confirmed grasp and lift events, complete-
skill displacement coverage and failures. Four consecutive closed+real-grasp
ticks confirm a grasp; lifts require >=1 cm within that bout. Coverage preserves
all movement, separately counting recoverable skills only if no failure occurs
within the skill. Missing first events are null. `wm_validation.json` evaluates
both arms on the same old fixed diagnostic cohort and the union of both arms'
initial frozen new-position complete skills. Exact training overlaps from either
arm are excluded for both. The new cohort costs no extra physical steps but is
not a balanced grasp/lift benchmark; absent moving samples cannot demonstrate
interaction prediction quality. Evaluation reports new adaptation and old
retention separately; a single seed does not establish generality, and without a
no-old-replay arm this experiment cannot isolate the effect of replay itself.
