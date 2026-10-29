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
