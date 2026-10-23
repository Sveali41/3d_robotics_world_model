# Cabinet exploration diagnostics

These experiments screen a **heuristic data-collection baseline** in the fixed
two-item Panda cabinet. The bowl diameter is 72 mm. They do not train PointWorld,
change dataset splits, or establish that better interaction coverage improves
world-model prediction error. Scene resets do not randomize object layouts;
this is a single-seed investigation, not a generalization benchmark.

Activate the existing environment and run commands from the repository root:

```bash
conda activate rlPractice
python -m exploration.diagnose_cabinet --help
```

## Stages

Use the **same output directory** for all stages. Each stage/arm refuses to
overwrite existing output. Source/configuration changes require a fresh output
directory so that results from different implementations cannot be mixed.
The runner saves a config snapshot, dependency versions, Git commit, working-tree
patch, and untracked source contents. All stages use compact recording and a
two-step contact-check interval; logging never enables dense execution.

### 1. Audit, with no physical policy training

```bash
python -m exploration.diagnose_cabinet --stage audit --output logs/cabinet_diagnostics_seed0
```

The two existing central grasp sequences must both lift their target. Each
grasp parameter is then perturbed separately by +/-5 mm: seven attempts per
object, including the center. Each object also gets a +30 mm lateral-offset
control, a closed-gripper approach, and a grasp-without-lift control. Labels
describe intended controls; metrics always describe the actual observed outcome.
The combined physical budget is at most **5000 real low-level steps**.

Rewards R0, offset-only, and R1 are recomputed through the production reward
wrapper on the same recorded inputs, with separate episode histories. No
counterfactual physics is executed. Per-step reward components and cumulative
episode returns are saved for all three reward profiles, with per-case totals
in `attempts.json`. Geometry is checked against the loaded
URDF's pad collision centers and the actual TCP pose; zero offset is accepted
only when their distance is below 5 mm. This does not assert that the object
center is an optimal grasp position.

The audit also performs the specified synthetic one-step PPO control, using
the production actor and update function: seed 0, 50 updates, 256 samples each.
Reward is zero for other primitives; for `top_grasp` it is
`1 - mean((normalized_parameters - [0.3, 0.2, 0.1])**2)`.
The target selector probability must exceed 0.8 and each parameter mean error
must be at most 0.15. Ratio consistency, masks and GAE boundaries must also pass.
This synthetic control uses no simulator steps and does not prove physical
exploration can be learned.

Inspect `audit/result.json`, `geometry.json`, `attempts.json`,
`ppo_positive_control.json`, and `primitives.jsonl`. A failed central control,
geometry check, incomplete audit, or failed PPO control blocks subsequent
stages. The runner never expands budgets or fixes thresholds automatically.

### 2. Frozen-policy sampling

```bash
python -m exploration.diagnose_cabinet --stage sample --output logs/cabinet_diagnostics_seed0
```

The same randomly initialized, frozen actor samples actions in A/P0 and C/P1,
each for exactly **5000 low-level steps**. Compare effective interaction and
coverage, not only the frequency of endpoints outside the cabinet region.
Completing both sampling runs is required before physical training.

### 3. Explicit short PPO training

This command is the only stage that trains policies in the simulator:

```bash
python -m exploration.diagnose_cabinet --stage train --output logs/cabinet_diagnostics_seed0
```

| Arm | Primitive | Reward |
| --- | --- | --- |
| A | P0: current goals | R0: current reward |
| B | P0 | R1: interaction reward |
| C | P1: cabinet-region goal projection | R0 |
| D | P1 | R1 |

P1 transforms every translational goal into the actual cabinet frame, clips it
to the existing exploration bounds, and transforms it back. Both grasp waypoints
are projected. Open/close commands and normalized action scales are unchanged.
This is a separate execution variant, **not collision avoidance or an IK check**.
Requested/executed goals and the number of changed goals are logged.

R1 sets `grasped_object=0`, `lifted_object=0`,
`moved_object_requires_grasp=false`, and `grasp_tcp_offset=0`. All remaining
weights, observations, contact definitions and termination settings stay the same.
No novelty bonus is introduced. P0/R0 remains the default production behavior.

Each arm starts from identical seed-0 weights and uses one environment with
256 transitions per rollout. The other PPO hyperparameters remain unchanged.
Each arm has an exact **50000 low-level-step** budget. The last primitive can
be truncated by the remaining budget; its transition is bootstrapped as a time
limit. A final rollout with fewer than two transitions is logged but not used
for an optimizer update. No action or real step is silently dropped from metrics.

Checkpoints are separate under `train/A` through `train/D`; the normal training
checkpoint directory is never used. `effective_config.json` records the actual
reward, projection flag and budget. `updates.json` includes KL, clip fraction,
explained variance, selector probabilities and parameter distribution summaries.
Use `--arm A` (or B/C/D) to run a single arm. There is no implicit resume or retry.

### 4. Evaluation and report

```bash
python -m exploration.diagnose_cabinet --stage evaluate --output logs/cabinet_diagnostics_seed0
python -m exploration.diagnose_cabinet --stage report --output logs/cabinet_diagnostics_seed0
```

Each trained policy is frozen and sampled **stochastically** for 5000 real steps.
All four complete evaluations, with equal budgets and both target IDs present,
are required for screening. An optional deterministic evaluation is separate:

```bash
python -m exploration.diagnose_cabinet --stage evaluate --deterministic --output logs/cabinet_diagnostics_seed0
```

Its output goes to `evaluate_deterministic/` and does not enter the report.

## Reading the metrics

Every primitive, including air motions, failures and falling objects, is kept
in `primitives.jsonl`. Each run also writes `metrics.json`, scalar `metrics.csv`,
and a coverage curve in `coverage.csv`.

- Efficiency uses contact/motion **primitive counts per 1000 actual low-level
  steps**, not per update. The first contact reports the detected contact sample's
  step; motion onset is only bounded by the primitive's end, since full object
  trajectories are not recorded.
- Coverage buckets are object x primitive x signed principal world-frame displacement axis
  x distance range: [15,30), [30,60), and [60,infinity) mm. Ties choose x before
  y before z. Coverage is integrated at actual step boundaries; a new bucket
  becomes available only at the end of its primitive.
- Both raw and recoverable coverage are reported. Recoverable motion excludes
  an object's event when that object ends outside the existing recovery bounds;
  its raw record is still retained. Repeat fraction uses recoverable bucket visits.
- Endpoint-outside and timeout rates use primitive counts as their denominator.
  Endpoint-outside is **not time spent outside**. Episode completion is counted
  once per finished episode, excluding the last budget-censored episode.
- `parameters_by_primitive` reports physical requested-parameter distributions;
  projected targets are recorded separately in outcomes.

B-A and D-C isolate the reward factor; C-A and D-B isolate the projection factor.
If only D qualifies, inspect the interaction between changes. An arm qualifies
only if recoverable coverage AUC **and** recoverable motion efficiency exceed A,
and both objects have recoverable motion. The report never chooses a policy by
return alone. If no arm qualifies, it reports `not_passed` rather than claiming
the exploration setup is effective. Qualification is a single-seed screening
result, not statistical significance or a world-model usefulness result.

## Focused validation

```bash
python -m pytest -q tests/test_cabinet_grasp.py tests/test_diagnostic_core.py tests/test_diagnostic_runner.py
```
