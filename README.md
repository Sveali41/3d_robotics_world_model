# MAC → 3D Robotics

This workspace extends **Model-driven Adversarial Curriculum (MAC)** to **3D robotics**, focusing on robot–object interaction, dynamics/world models, and model-driven exploration and curriculum construction.

This is an evolving research codebase. The sections below describe existing modules and entry points. The mapping of the complete MAC algorithm to 3D tasks, adversarial objectives, and curriculum updates must follow the original method and subsequent experimental design. Existing exploration baselines are not a complete MAC implementation.

## Research goals and current implementation

The goal is an experimental workflow connecting 3D interaction, data collection, model learning, curriculum updates, and evaluation. Current components include:

| Area | Code | Role and limitations |
| --- | --- | --- |
| Cabinet manipulation | `domain/wall_cabinet_env.py`, `domain/robocasa/` | ManiSkill cabinet scenes, robot control, and action primitives |
| Direct-action exploration | `exploration/cee_us/cee_us_cabinet.py`, `exploration/cee_us/cee_us_backend.py` | State GNN and iCEM exploration |
| Collision-aware skill execution | `domain/robocasa/motion_planning.py`, `domain/robocasa/skill_execution.py` | Shared pose planning and checked trajectory execution; split7 is a separate action contract, and planning does not guarantee grasp success |
| Skill-level exploration | `exploration/raps/raps_gnn_actor.py` | RAPS/GNN imagination actor–critic as a separate research variant |
| Point-cloud dynamics comparison | `exploration/pointworld/pointworld_cabinet_offline.py`, `exploration/pointworld/official/` | Paired interaction data and offline GNN / PointWorld training and evaluation; offline comparisons alone do not establish suitability as a replacement for the online planning model |
| PointWorld exploration components | `exploration/pointworld/pointworld_raps.py`, `exploration/pointworld/robot_flow_bridge.py`, `exploration/pointworld/round_loop.py` | Candidate actions, robot flow, ensemble disagreement, and round-based collection; integration and validation depend on the experiment |
| RGB stochastic JEPA exploration | `exploration/jepa/` | Split7 collection, flow-matching visual learning, sampled-future variance and mixed CEM; an experimental pure-curiosity branch |
| Household skills | `domain/household/` | PARTNR oracle skill interaction and a Habitat-Sim room viewer |
| THOR environments | `domain/thor/` | iTHOR / ProcTHOR scene interaction |

## Code map

```text
main.py             Hydra entry for cabinet tasks; room / household / thor dispatch
configs/            Environment, control, exploration, and experiment configurations
requirements/       Dependency lists for individual runtimes
domain/             Scenes, environment wrappers, robot control, and household skills
exploration/        Collection, rewards, policies, model integration, and evaluation
├── cee_us/         GNN, iCEM, state encoding, and performance tools; upstream in official/
├── raps/           Skill-level GNN actor–critic and replay comparisons; upstream in official/
├── pointworld/     Point-cloud models, robot flow, offline comparison, and collection;
│                   upstream submodule in official/
├── diagnostics/    Diagnostic experiments and metrics
└── visualization/  Policy visualization and scripted grasp demonstrations
common/             Shared utilities
third_party/        Other upstream code and runtime dependencies
scripts/            Runtime setup and supporting scripts
patches/            Local upstream modifications and base-version records
tests/              cee_us/, raps/, pointworld/, diagnostics/, environments/ groups
data/               Datasets and scene assets
logs/               Experiment logs and associated artifacts
outputs/            Environment interaction outputs
artifacts/          Generated artifacts
docs/AGENTS.md      Agent working rules
docs/LOG.md         Dated development history and Git commit mappings
```

`domain/ManiSkill/` contains the local ManiSkill upstream checkout. Keep upstream implementations, research baselines, and project adaptations distinguishable for reproduction.

The `exploration/` root retains shared rewards, the PPO interface, collection, action distributions, and trajectory recording. `PrimitiveRecord` retains its module path so existing serialized trajectories can still locate its class.

Local VS Code settings hide caches, `.runtime/`, `logs/`, `outputs/`, `artifacts/`, `wandb/`, and `data/`. Files remain in place; disable entries in `files.exclude` to show them. These settings are in the Git-ignored `.vscode/settings.json` and are not distributed with commits.

## Main entry points

Run commands from the project root. Environments have different dependency and asset requirements; there is currently no single installation manifest covering every entry point.

### Cabinet environments and exploration

Cabinet experiments use the existing `rlPractice` Conda environment. The base configuration is `configs/exploration.yaml`. Its default is `policy=ppo`, so running `python main.py` directly enters training.

The following commands only inspect configuration or command help:

```bash
conda run -n rlPractice python main.py --cfg job
conda run -n rlPractice python -m exploration.cee_us.cee_us_cabinet --help
conda run -n rlPractice python -m exploration.pointworld.pointworld_cabinet_offline --help
```

Main exploration configurations:

| Configuration | Purpose |
| --- | --- |
| `configs/exploration.yaml` | Base cabinet environment, action mode, rewards, and PPO |
| `configs/cee_us_cabinet_120k_total_heuristic.yaml` | Direct EE actions with GNN and iCEM |
| `configs/raps_gnn_actor_cabinet_split7.yaml` | Split7 skill-level curiosity exploration with direct EE execution |
| `configs/raps_gnn_actor_cabinet_400k_episode40s_centered.yaml` | Retained compact6 skill-level GNN actor–critic baseline |
| `configs/raps_gnn_actor_cabinet_400k_heuristic_only.yaml` | Skill-level heuristic-only comparison |
| `configs/raps_gnn_actor_cabinet_split7_planner.yaml` | Split7 primitives with collision-aware joint-trajectory execution |

`exploration.cee_us.cee_us_cabinet` provides `smoke / compare / train / evaluate / report`. The offline comparison entry provides `collect / audit / train-gnn / train-pointworld / evaluate`. Consult each entry's `--help` for parameters. Configured budgets do not indicate completed experiments.

The separate `epic_jepa_cem` agent uses RGB and primitive action history only;
it does not feed the existing diagnostic robot/object states into planning.
Use `configs/epic_jepa_cabinet_split7.yaml`; `smoke` caps collection at 2,000
controls, and new `train` runs require an explicit `--total-controls` budget.
See [JEPA setup, algorithm, commands and validation](docs/jepa_curiosity.md).

The validation-only collector and planner diagnostics use the reorganized diagnostic package:

```bash
python -m exploration.diagnostics.collect_wm_validation --help
python -m exploration.diagnostics.calibrate_wm_validation --help
python -m exploration.diagnostics.validate_cabinet_planner --help
python -m exploration.diagnostics.diagnose_grasp_targets --help
python -m exploration.diagnostics.probe_grasp_region --help
python -m exploration.diagnostics.collect_primitive_combinations --help
python -m exploration.diagnostics.plot_primitive_collection --help
```

Existing datasets and checkpoint action contracts remain unchanged. New split7 data has a 13D skill action and cannot directly evaluate an older 12D world model. Diagnostic source fingerprints still require matching code versions.

The collector uses the `feedback_gated_v1` protocol: a failed primitive ends the current attempt, and grasp sequences require the intended object to be held with a closed gripper for the final four consecutive control ticks before proceeding. Raw failed skills remain in `skills.jsonl`; `trial_outcome.json` records why the attempt ended. Budgets are upper bounds and unused steps are not filled with extra actions. Resumption skips attempts with saved outcomes and rejects datasets from the old protocol. These gates apply to the diagnostic script; the general `lift` primitive still permits empty-gripper motion. Planner records include per-start IK rejection reasons, joint-limit violations, FK errors, and collision pairs.

`diagnose_grasp_targets` audits the live TCP/finger geometry and checks 17 explicit position/orientation requests for each cup/bowl and original/shifted case. Run it in the MPLib/NumPy 1.26 robotics runtime with `--sampling-plan <saved sampling_plan.json> --output <new directory>`. It performs nine stationary open-gripper setup ticks per case; candidate trajectories are planned but not executed. Endpoint IK feasibility, path status, and unverified physical grasp success remain separate. Object bounds reuse the planner's native-shape support and cylinder-axis conversion; the installed ManiSkill collision-mesh helper's axis-swapped cylinder bounds are recorded for comparison rather than used to generate goals. Existing WM state geometry features are unchanged.

`probe_grasp_region --source <completed endpoint diagnostic directory> --output <new directory>` samples each selected anchor at nine heights and sixteen continuous Latin-hypercube position/orientation offsets. It records every IK/path rejection, then physically tests at most three path-accepted candidates per case (anchor and height extremes), capped at 300 control ticks each. Approach must succeed before closure; four final consecutive closed-grasp ticks on the intended object plus its measured attachment are required before a planned 4 cm lift. Independent physics-substep contact auditing stops forbidden contacts. Raw requests, control feedback, payload planning, event metrics, and PNG/SVG projections are retained. These finite local probes do not certify a continuous feasible region and are not an unbiased WM validation dataset. No production primitive, planner goal, or WM feature is changed.

`collect_primitive_combinations` uses protocol `object_relative_split7_v1`: sample desired closure positions from each object's corrected native collision AABB (X/Y padded 2 cm, Z inset 1 cm), invert five fixed primitive recipes, and execute through the real `SkillCollector` and `PlannedSkillExecution`. The existing 67D state, 13D action, planner, reset orientation, collision checks, and grasp checks are preserved. One random position per 2x2x2 cell is shared across recipes; original and shifted layouts share normalized positions and recipe IDs. These are attempted positions, not known feasible grasps. Parameters are recomputed from measured EE feedback before each primitive; out-of-range or backward approach requests are recorded and rejected without clipping.

```bash
# Default: save 160 candidate attempts; initialize geometry but execute no primitives.
python -m exploration.diagnostics.collect_primitive_combinations --stage plan --output logs/object_plan
# Two attempts per object/layout: YZ-first and X-first, at most 400 controls each / 3200 total.
python -m exploration.diagnostics.collect_primitive_combinations --stage pilot --output logs/object_pilot
# Formal collection requires an explicit global budget and separate sampling seed.
# Replace the budget and replay paths with the approved budget and ALL compared models' training replays.
python -m exploration.diagnostics.collect_primitive_combinations --stage collect \
  --output logs/object_validation --seed 5100000 --max-steps <approved-control-budget> \
  --calibration-data logs/object_pilot --exclude-replay <training-replay.npz>
```

Five recipes cover YZ-first approach, approach/refinement, elevated approach/descent, X-first alignment/refinement, and X-first closure without refinement. All now target the same nominal closure position: the last recipe compensates its internal reference by +2 cm X, replacing the historical reference-relative early-close convention. `--recipes`, `--scene`, and `--object` restrict attempts; `--grid`, `--seed`, `--trial-controls`, `--max-steps`, and `--quota` control the declared plan. Sampling seeds default to 4100000 for plan/pilot and 5100000 for collect. Every attempt uses four real closing primitives, then requires four final consecutive intended-object held ticks before lift. Failed approach ends the attempt. Shallow initial shelf support remains allowed during closure/initial lift; forbidden contact retains the current tick and ends the attempt. No hidden control steps, target adjustment, or orientation search is added.

Outputs include `sampling_plan.json`, geometry/contracts/seeds in `manifest.json`, scheduled trial IDs, raw transitions/skills, rejection reasons, physics contacts, and completed nonzero endpoints (including executed failures with status, duration, and raw references). Zero-duration planning rejections remain raw records without becoming dynamic transitions. `windows_index.json` contains deduplicated h=1/3/5 continuous skill sequences built within individual attempts; balanced object rows are never spliced into rollouts. `balanced_index.json` selects by layout/object/observed event (lift, displacement, contact-only, free motion, static), defaulting to 20 **object-level skill indices** per stratum. This quota is neither a control-step count nor a count of independent trajectories. `full_state_index.json` deduplicates the selected object rows by endpoint; `coverage_report.json` reports candidate cells, recipe coverage, event deficits, duplicates, and window counts. Candidate-cell coverage does not imply successful physical visitation or grasp coverage.

Plan/pilot artifacts are calibration data and cannot be marked validation-ready. Formal collection never increases the supplied budget to fill deficits. Supply every compared model's 67D/13D replay with repeated `--exclude-replay` and calibration directories with repeated `--calibration-data`; source hashes are saved. A supplied calibration dataset with the same sampling seed is rejected. An exact state/action/next-state overlap with either source excludes the entire attempt from evaluation indices while retaining its raw data and labels. These checks detect exact duplicates, not broader statistical dependence. Without both source checks, all event quotas, h=1/3/5 windows in every object/layout case, and completion of scheduled attempts, readiness stays false and deficits remain explicit. An incomplete formal collection returns exit code 2. A successful pilot verifies the execution/recording pipeline only; it does not establish sufficient grasp coverage or an unbiased validation distribution. No model is trained.

For an original-layout dataset with 500 attempts per object, use `--stage collect --scene original --grid 5 5 4 --max-steps 400000`: 100 spatial cells per object times five recipes gives 1000 attempts, each capped at 400 controls. The sampling bounds and primitive semantics remain unchanged; the finer grid increases coverage. Raw failed attempts and partially executed budget-truncated skills remain available. This budget is an upper bound and does not guarantee successful grasps or a validation-ready dataset.

`plot_primitive_collection --source <collection directory>` reads all saved attempts, including nested per-object collection directories, and exports actual EE paths and endpoints in robot-root meters. It generates endpoint and full-trajectory projections for both objects, with actual outcomes distinguished from requested closure targets. Labeled object references use the saved corrected native collision bounds at reset; they show initial occupied bounds, not exact object silhouettes or final object positions after interaction. Both objects appear as references, and different layouts are shown separately. The source transitions retain every recorded control step; plotting does not replace actual endpoints with targets, filter out failed attempts, concatenate paths across resets, or jitter repeated positions. Identical endpoints can overlap on the plot, so use the accompanying tabular/array exports for counts. The same command can visualize a partial dataset while a longer collection continues, then regenerate the complete figures afterward.

For new skill-level curiosity runs, use the direct-execution split7 configuration:

```bash
.runtime/thor-ceeus/bin/python -m exploration.cee_us.cee_us_cabinet train \
  --config configs/raps_gnn_actor_cabinet_split7.yaml \
  --output logs/raps_split7_curiosity
```

This selects the existing RAPS imagination actor with the CEE-US GNN ensemble, using completed primitive endpoints. It preserves the compact6 baseline's ensemble-disagreement reward, object approach heuristic, training budgets and 100/20 Hz timing. The action is now 7 primitive choices plus 6 shared parameters: `move_delta_ee_pose`, `align_yz`, `approach_x`, `lift`, `drop`, `open_gripper`, `close_gripper`. The actor selects their order; `approach_x` does not automatically close the gripper. Execution uses the existing direct `ee_delta` controller and latched gripper. Start a fresh actor and WM without `--checkpoint`; compact6 checkpoints and the 12D `logs/cabinet_grasp_sequences_972` dataset are incompatible. This configuration does not migrate those artifacts. The split7 planner configuration remains a separate variant, and the 4D iCEM entry retains its existing configuration.

The persistent [grasp-retention and primitive contract](docs/pick_place_primitives.md) requires closed-gripper transport, per-control grasp/slip checks and supported release. Physical retention is not guaranteed by a closing command alone.

The experimental `--pick-place-primitives` flag replaces the selected RAPS skill set with two macros and a shared three-slot argument bank (5D action): grasp uses TCP XYZ and place uses object-target XY, with its inactive Z slot zeroed. It defaults to false; `--no-pick-place-primitives` overrides a YAML opt-in. YAML uses `raps_actor.pick_place_primitives: true`. The resolved run enables gripper-command memory and disables the legacy contact/drop/placement hooks, because the macros own those gates. This mode selects the existing joint planner (`motion_execution: planner`, `action_mode: joint_position`), requires a fresh 5D WM/actor and a new output directory. Existing split7 replay/checkpoints are incompatible; strict resume checks remain enabled. The original skill set remains selected when the flag is disabled. To bind a physically confirmed grasp until opening, add `--grasp-retention constraint` (YAML `raps_actor.grasp_retention: constraint`). This CPU-physics mode locks the object's current pose relative to the hand with a native positional constraint and angular drive, then removes it on opening, reset or close. `--grasp-retention friction` preserves contact/friction holding and is the default. Settings, action contracts and recorded rows distinguish these modes; strict resume rejects switching them. Under the previous version 4 TCP-XYZ placement contract, cup and bowl each passed one complete planned grasp/place/release/home example plus 20 postcheck controls; those fixed-layout examples do not validate version 5 object-target placement.

`grasp(x,y,z)` opens, approaches the bounded TCP position with the common reset orientation, closes, confirms continuous grasp, lifts 8 cm and carries to the shared intermediate TCP `[0.18, 0, 0.84]`. `place(x,y)` targets the object collision-AABB center projected onto the lower shelf. The measured confirmed TCP-relative payload geometry compensates XY grasp offset and computes the shelf-touch TCP height. Place preserves closure, retreats through shared preplace TCP `[0.28, 0.15, 0.56]`, approaches 4 cm above the predicted shelf-touch TCP, descends until settled shelf support, opens, clears the object and returns through preplace to the episode's initial TCP. Cabinet-exterior preplace transport may rotate when TCP X<0.29 m, restoring the common fixed orientation at the preplace endpoint; grasp, placement and cabinet-interior motion retain fixed orientation. The interface and route are macro contract version 5; earlier macro checkpoints are incompatible. Return poses are TCP poses, not exact joint configurations. Placement uses actual held-object collision geometry, with no object-type offsets, automatic object selection or teleportation. Missing/lost grasp, unsupported release and timeouts are recorded as failures; failed completed macros remain valid WM endpoints, while budget-truncated macros are excluded.

All coordinates are **robot controller-root meters**, not world coordinates:

| Variable | Grasp (upper) | Place (lower) |
| --- | --- | --- |
| X | 0.370–0.671 | 0.370–0.671 |
| Y | -0.070–0.310 | -0.070–0.310 |
| Z | 0.692–0.872 | Automatic from grasp geometry and shelf height |

Grasp retains the nominal support +0–22 cm Z band and a 2 cm endpoint margin (+2–20 cm). XY retains an 8 cm margin from cabinet edges. Place additionally rejects goals whose full object footprint would approach within 2 cm of the nominal cabinet edges, keeping the hand closed. Grasp bounds refer to TCP coordinates; place bounds refer to the object collision-AABB center at the fixed placement orientation. They do not guarantee robot reachability or collision-free paths. Place height is computed automatically, and explicit place Z arguments are rejected; release still requires positive cabinet-frame force at lower-shelf contact geometry and low object speed for four consecutive controls. Constraint mode uses the peak pose-displacement speed across the latest control’s physics substeps, recording native speed alongside it; friction mode retains native speed. On first shelf contact, constraint mode holds measured finger drive positions while retaining logical closure and binding to remove continued squeeze pressure. Targets outside the physical bounds are rejected by encoding; normalized policy values are mapped into the bounds. Full bounds, routes and safety gates are included in the checkpoint action contract. Except for the explicitly permitted exterior preplace rotation, sampled paths and per-control feedback must remain within 0.5 degrees of the reset orientation. Fixed-orientation RRT transfers project TCP positions using existing SAPIEN IK before TOPP. Exterior preplace transfers densify RRT joint waypoints to at most 0.025 rad before TOPP to reduce spline corner-cutting. Both paths retain limit, collision and endpoint-orientation audits. Failed projections stop explicitly. Finite path/control samples are not a continuous-motion certificate. Each macro is bounded at 2000 controls, each planned segment at `motion_plan_max_steps` (default 400), and support seeking at 1000 controls. Descent uses at most 5 mm per planned segment and shortens the final step using collision-shape clearance; only shallow (at most 1 mm) payload contact with the matching lower shelf top is allowed. Other collision checks remain active.

This is an opt-in experimental controller. See `artifacts/complete_pick_place_20261113/index.html` for historical version 4 complete cup/bowl videos, measured postchecks and retained failure diagnostics; version 5 physical placement remains unverified, and synthetic state-machine tests alone do not establish successful lower-shelf placement. No existing experiment is migrated or restarted automatically. Collection still resets at each collection call and retains the configured physical budget: the existing 800-control episode and 1000-control collection chunks can truncate longer macros. Choose a new experiment budget explicitly before formal training; the flag does not silently extend those budgets.

The fresh pick/place preset is [configs/raps_gnn_actor_cabinet_pick_place.yaml](configs/raps_gnn_actor_cabinet_pick_place.yaml). It connects the same CEE-US GNN world model, RAPS imagination actor/critic, random bootstrap, online fitting, evaluation, checkpoints and strict resume to the 5D macro interface. Physical planning/constraint retention uses CPU simulation; WM and actor/critic use CUDA. The existing prediction/runtime flags remain independent.

This explicitly uses 2000 bootstrap controls, 398 rounds of 2000 controls, 2000 evaluation controls and 2200-control episodes: **800000 controls per arm, including evaluation**. Both the primitive wrapper and native TimeLimit receive the episode budget. The legacy split7 preset is unchanged. Every collection call resets physics; a macro can still be truncated near the remaining chunk boundary and then is excluded from WM endpoints while all executed controls count.

The existing three-container launcher accepts the new preset:

```bash
/opt/anaconda3/bin/python scripts/launch_raps_comparison.py \
  --config configs/raps_gnn_actor_cabinet_pick_place.yaml \
  --numpy-shim /tmp/raps-three-arm-ls76ex80/numpy126 \
  --output logs/raps_pick_place_comparison
```

This command **starts the full experiment**; integration validation does not execute it. It creates curiosity+heuristic, curiosity-only and random arms with identical resolved settings except collection policy/object heuristic. Their actual completed-macro sample counts and WM minibatch counts may differ. The original approach-progress heuristic is retained on macro endpoints, without adding a task/grasp/place bonus; its usefulness under the new action semantics is not established. Native RRT is unseeded, so independent-container bootstraps are not guaranteed identical even with matched environment/IK seeds. Shared GPU use prevents controlled wall-time comparisons.

Add `--include-heuristic-only` to launch four arms: curiosity+heuristic, curiosity-only, heuristic-only and random. The heuristic-only arm retains the learned RAPS actor, world model and existing heuristic, but sets `raps_actor.disagreement_weight: 0.0`. All arms retain the base training/control budgets; the pick/place preset totals 800000 controls per arm (3200000 for four arms, including evaluation). Each arm owns its output/checkpoints and uses the launcher's frozen source snapshot. The three-arm default is preserved.

Start fresh: old 13D split7/4D iCEM checkpoints, replays and validation actions cannot be reused as 5D macro data. New macro checkpoints can resume with matching per-arm saved config and a new output directory; only total rounds/evaluation budget may change. The separate 4D iCEM implementation remains unchanged. See [the primitive contract](docs/pick_place_primitives.md) and bounded integration evidence in artifacts/pick_place_pipeline_20261113/.

RAPS supports `--execution reference` (the existing default) and `--execution cuda_graph` with `--device cuda`. Both use the same FP32 GNN, ensemble, imagination batch/horizon, rewards and optimizer updates. CUDA Graph mode reuses the existing CEE-US prediction capture for each input shape; fitting or loading the world model invalidates captures. First use after invalidation includes capture overhead. This accelerates imagined GNN predictions; simulation, world-model fitting and checkpoint compression retain their existing paths. CUDA unavailability or capture errors stop the run explicitly.

```bash
# Run in the existing environment with PyTorch/CUDA and the project dependencies.
python -m exploration.cee_us.cee_us_cabinet train \
  --config configs/raps_gnn_actor_cabinet_split7.yaml \
  --device cuda --execution cuda_graph --output logs/raps_split7_cuda_graph
# Use --execution reference with a separate output directory for the baseline.
```

The flag overrides YAML `execution` and the resolved value is saved in the run configuration. Training resume retains the saved execution setting under the existing strict configuration check; frozen evaluation can select either execution mode for a compatible checkpoint. Neither mode changes the primitive controller or enables GPU physics.

RAPS also accepts `--runtime-optimizations` and `--no-runtime-optimizations`, independently of `--execution`. These override YAML `runtime_optimizations`; the default remains false. Opt-in fitting keeps replay arrays on the training device for the duration of each fit and uploads the original NumPy-sampled ensemble indices once per epoch. It retains batch sizes, tail batches, sampling/RNG order and optimizer updates. Actor/critic gradient checks are aggregated before their optimizer steps, and detached diagnostic values are transferred in batches. Nonfinite checks remain enabled. Device-memory exhaustion stops with an instruction to disable runtime optimizations.

With runtime optimizations enabled, one background CPU thread compresses and saves independent CPU checkpoint snapshots while the next collection/training runs. Only one save may be outstanding; reading a newly saved checkpoint, successful completion and shutdown wait for it, and write failures propagate to the main run. Existing checkpoint/NPZ formats and save frequency are retained; `.pt` is published after both replay files. The collection/WM/actor update sequence remains synchronous, with no old-policy collection or GPU-physics migration.

```bash
python -m exploration.cee_us.cee_us_cabinet train \
  --config configs/raps_gnn_actor_cabinet_split7.yaml \
  --device cuda --execution cuda_graph --runtime-optimizations \
  --output logs/raps_split7_optimized
```

Use `--no-runtime-optimizations` with a separate output directory for the baseline. The resolved runtime setting is saved and must match on training resume. Collection and checkpoint timings are written to `timings.jsonl`; each `training.jsonl` entry includes replay preparation, normalizer/WM and actor/critic stage timings. Device-resident fitting additionally reports bulk-upload time; reference minibatch transfers remain included in WM time. `result.json` records wall time after all saves complete. Concurrent save/work durations overlap and must not be summed to infer wall time.

`PYTHONPATH=. python scripts/benchmark_raps_runtime.py --checkpoint <checkpoint.pt> --output <new-directory>` performs bounded offline comparisons using the checkpoint's real skill replay. Defaults are 512 samples, one WM epoch, five actor updates and three warmed alternating-order repeats. It measures replay/WM, scalar-sync and background-save components plus the combined mode, verifies state/RNG/replay parity and includes save drain in wall time. If the checkpoint's companion `.low_level.npz` is absent, supply `--low-level-replay <actual-control-replay.npz>`; that source is recorded and used only for serialization, never fitting. No physics is run and original checkpoints are not modified.

The A5000 bounded check in `artifacts/raps_runtime_20261111/` found no overall speedup: isolated replay transfer/gather was about 2.06x faster, while combined offline median was 0.363s versus 0.344s reference. Both 256-control split7 smokes reproduced identical recorded trajectories/replays and matching model/optimizer/RNG state, but wall times were 20.90s optimized versus 20.11s reference. These small runs do not establish long-training performance; external GPU activity and trial variability were observed. Runtime optimizations remain opt-in and the 10% target is unproven.

For three simultaneous matched exploration arms, run `/opt/anaconda3/bin/python scripts/launch_raps_comparison.py --output logs/raps_split7_three_arm_<unique-name>` on the Apptainer host. This explicitly starts three independent named instances of the same existing SIF, with a shared read-only snapshot of the actual working source/Git evidence and private process RNG/logs/checkpoints. It derives `curiosity_heuristic`, `curiosity_only` and `random` configurations from the same split7 YAML, changing only `object_heuristic` and `collection_policy`. The current budget is 1000 random bootstrap controls, 398 rounds of1000 online controls, then1000 frozen evaluation controls:400000 real20Hz controls per arm, not400000 skill decisions. All use seed0, the same reset-seed schedule, direct CPU physics/GPU networks, WM25epochs and actor100updates per round, reference execution and runtime optimizations disabled. Effective skill counts and resulting WM minibatch-update counts can differ because exploration changes executed skill durations and completion. The random arm still fits the same WM/actor/critic, but never uses the actor for collection or evaluation. The existing `train --policy random|curiosity` option now selects the actual training collection policy; default remains curiosity and `compare` keeps its original two-arm behavior. Random checkpoints resume from their `random/` prefix with the saved policy enforced.

Each launch writes `comparison.json`, derived arm configs, `source_sha256.json`, exact launch commands, `stdout.log`, process IDs and per-arm `exit_code`; the corresponding instance stops when its training process exits. A new output directory is required and automatic retries are not performed. The verified NumPy1.26 shim is copied to an executable private `/tmp` runtime directory; no dependency downloads or container upgrades occur. The single shared GPU can contend, so this is an exploration-quality comparison with equal interaction/configuration budgets, not a controlled wall-time benchmark or multi-seed result. Final WM prediction metrics use each policy's own evaluation trajectories; absolute cross-policy accuracy claims additionally require a common independent evaluation dataset.

### GPU batched split7 collection

The original SAPIEN 3.0.3 has confirmed sparse joint/body indexing defects. Use the isolated, same-version native correction described in [docs/sapien_gpu_reset.md](docs/sapien_gpu_reset.md). The original SIF and installed libraries remain unchanged. The collector checks actual sparse joint/body writes at startup and exact live-peer sensor continuity on each refill; an incompatible runtime stops clearly. CPU scalar collection retains the original reference runtime.

`collect_cabinet_grasp_sequences` accepts `--collector gpu_batch --skill-set split7 --num-envs N`. The default `scalar` collector remains the CPU reference. GPU collection keeps one scene batch alive, advances each slot's primitive independently, and refills completed slots with the next case ID. Case seeds, nine templates, 800-control per-case budget, 100/20 Hz timing and 67D/13D contracts are retained. It runs direct EE control without a motion planner, training, image capture or goal projection. `--num-envs` is required; unsupported settings fail explicitly.

Use the existing pinned runtime. On this machine `/data/data/siyao/3d_world_model.sif` contains ManiSkill `e406239d2b1c554e4b99010bc9b40ad2bbd91f05` and Torch 2.3.1/CUDA. Its default NumPy 2 installation requires a separate historical NumPy 1.26.4 directory; install once without replacing the container packages:

```bash
apptainer exec /data/data/siyao/3d_world_model.sif python -m pip install \
  --no-deps --target /tmp/wm_gated_numpy126 numpy==1.26.4
apptainer exec /data/data/siyao/3d_world_model.sif \
  python scripts/build_sapien_sparse_patch.py
apptainer exec --nv /data/data/siyao/3d_world_model.sif env \
  PYTHONPATH="/tmp/wm_gated_numpy126:$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_patched_sapien.py --runtime .runtime/sapien_sparse_patch -- \
  -m exploration.diagnostics.collect_cabinet_grasp_sequences \
  --checkpoint logs/raps_compact6_train_400k_episode40s_centered_20261026T173840/curiosity/round_398.pt \
  --output logs/cabinet_split7_gpu_smoke --stage smoke \
  --skill-set split7 --collector gpu_batch --num-envs 2
```

Use `--stage plan` to inspect the original 972 cases without simulation. For a separately authorized full run, change to `--stage collect`, select a measured environment count, and use a new output directory. The compact6 checkpoint supplies configuration/state provenance only; its 12D weights are not converted or used. Source snapshots, backend, dependency versions, environment count, actual runtime state contract, original native binaries, overlay build manifest and per-attempt slot assignments are recorded. GPU and CPU trajectories are different numerical protocols; seeds identify requested cases but do not guarantee identical trajectories across backends or batch sizes.

GPU steps are stored losslessly in `trajectory_chunks/`: each atomic chunk has one uncompressed numeric `.npy` pool plus a typed schema and routing index. The default chunk spans at most 256 batch control steps. Full before/after states, requested/executed actions, seven controller values, contacts, rewards, phase and case/attempt/skill identity remain available. A bounded single writer process overlaps writes with simulation and applies backpressure; it does not discard rows. `skills.jsonl`, attempt metadata, phase traces and summaries stay readable JSON. Case completion is published only after its trajectory chunks are durable. Periodic reports use cached summaries; `evaluation.npz` is rebuilt every 64 closed cases by default (`--export-every-cases`) and at normal shutdown.

Repeat the identical command with `--resume` to skip closed cases. Interrupted attempts retain committed chunks and start a new attempt from reset; trajectories never join across attempts. Resume requires identical sources, configuration, backend and batch size. Historical JSONL and numeric chunks share the `read_transitions` interface and the existing visualization CLI. The viewer still requires an existing offline Plotly bundle through `--plotly-source`. GPU action-video replay is explicitly unsupported by the CPU replay renderer; it stops rather than publishing an unverified movie.

The audited A5000 calibration recommends **128 environments** among the tested sizes: **294.65 active controls/s including initialization and final writes, 14.32x the retained CPU baseline**, with whole-device peak 3069 MiB of 24564 MiB. All 16384 benchmark controls and 49 refills passed exact continuity/endpoint audits. These bounded samples shared the GPU with other training jobs; they do not predict full-dataset or grasp-success throughput. See the [performance report and ready-to-use collection command](docs/gpu_split7_performance.md). No formal 972-case collection was launched.

A bounded calibration compares the scalar collector and GPU batches of 16/32/64/128, keeping complete recording enabled:

```bash
python -m exploration.diagnostics.benchmark_split7_collection \
  --checkpoint <source-checkpoint.pt> --output logs/split7_collection_benchmark
```

Run it through `scripts/run_patched_sapien.py -- -m` inside the same GPU-visible pinned runtime shown above. GPU workers reenter the isolated launcher. Each configuration stops at 4096 active controls or 60 seconds of collection; initialization, final drain and export are reported separately. All configurations use the same fixed 128-case pool in case-ID order. Reports include control/endpoint/case rates and sampled whole-device memory, retaining failures. The recommendation selects the fastest measured end-to-end control rate below 80% GPU memory; within 5% of fastest it prefers fewer environments. These short, different-length prefixes measure control throughput, not complete-dataset runtime, successful-grasp throughput or independent dynamics coverage. Calibration data is marked separately from validation data.

### Automatic multi-point cabinet grasp collection

The selected collection and WM-validation workflow uses the original compact6 primitives with `motion_execution: ee_delta`, without the motion planner. The planner comparison below is retained as a separate diagnostic and is not part of collection or training.

`exploration.diagnostics.collect_cabinet_grasps` collects validation-only compact6 primitive endpoints through the existing physical executor, without training a model. It uses the original 12 cm cup/bowl spacing and fixed wrist orientation. Its 108 cases combine two objects, 27 grasp offsets around the calibrated object-relative targets, and 4/8 cm lift distances. Root-frame offset increments are X = -1/0/+1 cm and Y/Z = -1.5/0/+1.5 cm. This is a local grasp-neighborhood evaluation, not exhaustive environment coverage.

```bash
.runtime/thor-ceeus/bin/python -m exploration.diagnostics.collect_cabinet_grasps \
  --checkpoint logs/raps_compact6_train_400k_episode40s_centered_20261026T173840/curiosity/round_001.pt \
  --output logs/cabinet_grasps_smoke --stage smoke
```

Use `--stage plan` with a new output directory to inspect the full sampling plan without simulation. Use `--stage collect` with another new output directory to run all 108 cases headlessly in one process. The default seed is 1000000; each case uses that seed plus its fixed case ID. Each case resets the scene and executes open, approach/close, grasp confirmation, lift, carry, drop, release and retreat; it ends after the sequence or failure cleanup, with at most 800 control ticks at 20 Hz (100 Hz physics). The smoke stage runs only each object's center grasp with an 8 cm lift.

Raw actions, partial trajectories, zero-duration attempts and unsuccessful outcomes remain on disk. `evaluation.npz` includes completed nonzero-duration skill endpoints from successful, failed and interrupted attempts, with 67D states, 12D actions, unique episode IDs and actual durations; incomplete skills are excluded. Reports retain all finished cases in success-rate denominators. `--resume` preserves old attempts, skips finished cases and restarts interrupted cases from reset without joining their trajectories. Resume requires matching execution sources, checkpoint and sampling configuration. No data is automatically added to training replay, and collection does not certify collision-free grasping.

Case success preserves the calibration script's measured grasp/lift/carry/release criteria. It does not require every constituent primitive to report `success`: for example, a drop may time out before the subsequent release completes. Inspect the per-primitive statuses in each case report alongside its aggregate outcome.

`exploration.diagnostics.collect_cabinet_grasp_sequences` provides a separate sequence-variation protocol using compact6 by default and the direct executor:

- Six permutations of X/Y/Z approach waypoints followed by `close_gripper`. The root-frame grasp target is frozen after opening; each waypoint updates target coordinates in the selected order. Actual reachability and arrival errors are recorded rather than assumed.
- Three templates retain the original `approach` and insert a +Y 3 cm move and a return to the anchored TCP position after grasping, midway through lifting, or before carrying. The midway template uses two equal requested lift halves. Existing motion tolerances remain unchanged, so physical return and final lift positions are approximate. Loss of grasp during inserted motions is retained as a failure.

These are nine templates, not a Cartesian product of every approach order and insertion stage. The full plan has 972 cases (108 base cases times nine templates); each template shares its base case's initial seed. Outputs are separate from the original 108-case dataset. The smoke stage runs eight center cases: two objects times one approach-order template and the three insertion templates. It does not automatically launch full collection or training.

```bash
.runtime/thor-ceeus/bin/python -m exploration.diagnostics.collect_cabinet_grasp_sequences \
  --checkpoint logs/raps_compact6_train_400k_episode40s_centered_20261026T173840/curiosity/round_001.pt \
  --output logs/cabinet_grasp_sequences_smoke --stage smoke
```

Use `--stage plan` to inspect the sampling plan, `--stage collect` with a new output directory for all cases, or the matching command plus `--resume` after interruption. Reports include template-specific outcomes and measured adjustments. Data still contains completed physical primitive endpoints; repeated initial conditions and shared sequence prefixes are correlated observations, not independent samples. Keep dataset episode IDs separate when comparing or combining runs.

For the split7 action set, add `--skill-set split7` and use a new output directory, such as `logs/cabinet_grasp_sequences_split7_smoke`. Its seven primitives are `move_delta_ee_pose`, `align_yz`, `approach_x`, `lift`, `drop`, `open_gripper`, and `close_gripper`. The six axis-order templates use `align_yz` for Y/Z and `approach_x` for positive X, followed by explicit closure. The three insertion templates replace composite `approach` with separate YZ alignment, X approach and closure; subsequent carry and insertion motions retain the original movement primitive. This uses direct EE control, without a planner, and preserves each primitive's existing tolerance. Splitting changes physical execution: alignment can stop within its 1.5 cm tolerance, and the subsequent X primitive preserves achieved Y/Z rather than continuing to correct toward the original 3D grasp target. Success parameters from composite approach therefore need not transfer unchanged.

Split7 exports **13D actions** with the same **67D state**. The supplied compact6 checkpoint supplies only state/configuration provenance; its model and policy weights are not used or converted. The manifest records both its source 12D contract and the effective split7 contract. Old datasets remain compact6 and cannot be relabeled or evaluated with an incompatible action contract. Resume requires the same skill set and source snapshot; historical runs retain their saved sources. `--stage plan --skill-set split7` lists all 972 cases without simulation; `--stage collect --skill-set split7` starts a separate full collection only when explicitly requested.

Create an offline interactive 3D snapshot while collection continues:

```bash
.runtime/thor-ceeus/bin/python -m exploration.diagnostics.visualize_cabinet_grasp_sequences \
  --input logs/cabinet_grasp_sequences_972 \
  --output artifacts/cabinet_grasp_sequences_3d
```

The viewer uses only closed cases from the captured snapshot. Open its generated HTML in a browser to rotate/zoom, select an object, template, outcome or individual case, and compare actual TCP and target-object paths. Colors distinguish complete pipelines, confirmed grasps followed by pipeline failure, and unconfirmed grasps. The static overview shows labelled representatives; the HTML includes all captured cases. Interactive paths retain every recorded 20 Hz controller step, including reset; path lengths use these full trajectories, in robot-root metres. No trajectories are joined across cases or attempts. This is a snapshot, not a live feed; regenerate into a new output directory to include later cases. The exporter reuses the inline Plotly.js bundle from the existing coverage-network HTML by default (`--plotly-source` selects another HTML with an inline bundle), embedding it for offline use without installing packages or loading a CDN.

New snapshots open on one complete case in **Paths** mode; select **Density** to inspect the aggregate distribution. Density uses a root-frame grid anchored at `(0, 0, 0)`, with `--voxel-size 0.02` metres by default. Each cell counts distinct cases whose recorded TCP or target-object positions entered it, with each case counted at most once per cell. Counts use all recorded positions, including the initial position, before any display downsampling; gaps between samples are not filled. Filters recompute counts for the selected cases. Color uses a labelled logarithmic scale, with raw case counts available on hover; `density.png` shows TCP and target-object distributions side by side. Common initial states can form hotspots. The spatial extent shows where recorded positions occurred, while color shows repetition within that extent; this statistic does not establish whole-environment or dynamics coverage.

Grasp/lift counters use saved calibration feedback rather than movement alone. Select one case to inspect confirmed grasp, held-lift verification, measured rise/carry distance, release, failure reason and actual primitive statuses (including timeouts in otherwise successful pipelines). In **Paths**, a single case shows its complete TCP route colored by recorded action phase, dotted target-object route, direction arrows and start/end labels. A cabinet wireframe and translucent shelf provide fixed spatial context (schematic frame only, without door or robot collision geometry). The step slider and Play/Pause animate current TCP/object markers while leaving the entire route visible. Each video has a **Show complete 3D trajectory** button to select its case directly. The plot retains original 20 Hz step/time labels; these display changes do not change the underlying success criteria, metrics or raw dataset. Initial reset points have no grasp feedback.

For an intuitive view of what happened, render a few representative robot videos, then embed them above the distribution plot:

```bash
.runtime/thor-ceeus/bin/python -m exploration.diagnostics.render_cabinet_grasp_replays \
  --input logs/cabinet_grasp_sequences_972 --output artifacts/cabinet_robot_replays
.runtime/thor-ceeus/bin/python -m exploration.diagnostics.visualize_cabinet_grasp_sequences \
  --input logs/cabinet_grasp_sequences_972 --output artifacts/cabinet_with_robot_playback \
  --replays artifacts/cabinet_robot_replays
```

The default six videos cover both cup and bowl, grouped by object: cup cases 2/0/272 and bowl cases 542/486/630 show a complete pipeline, unconfirmed grasp, and lost grasp during lift respectively. Rendering needs the simulator's GPU/Vulkan runtime. This diagnostic replays recorded full controller actions, checking every 67D state exactly against the original logs; mismatched replays fail instead of publishing a misleading video. Labels show recorded phases, grasp confirmation and target height change. Videos run at the original 20 Hz, with an extra one-second final hold; use native play/seek/fullscreen controls and the playback-speed selector. Only the selected examples are rendered. Their movies are embedded in the HTML for offline viewing; the underlying dataset, primitives and evaluation metrics stay unchanged.

To compare the two center-case trajectories with the existing collision-aware executor, use the separate diagnostic:

```bash
.runtime/thor-ceeus/bin/python -m exploration.diagnostics.compare_cabinet_grasp_paths \
  --baseline artifacts/cabinet_grasps_smoke_20261110 \
  --output artifacts/cabinet_grasp_path_comparison
```

It reuses the baseline checkpoint, seeds and grasp offsets, verifies the initial states, and executes the planner variant in a separate headless scene. PNG/SVG plots show actual root-frame TCP and object paths in 3D, top and side views. Rejected targets are labelled separately from executed paths. The diagnostic preserves raw planner results and does not export planner data as an old-WM validation set. A TCP curve does not describe the whole arm's swept volume or certify collision freedom.

### Teleoperation-derived primitive validation

`exploration.diagnostics.teleop_wm_validation` records a mouse-controlled EE demonstration, extracts parameters from human-labelled skill segments, then resets the scene and executes those parameters through the existing skill collector. It supports the compact6, latched-gripper, EE-delta checkpoint contract; it does not train a model or use demonstration endpoints as primitive prediction labels.

Run these commands from an interactive terminal in the `rlPractice` environment:

```bash
python -m exploration.diagnostics.teleop_wm_validation --stage record \
  --checkpoint logs/raps_compact6_train_400k_episode40s_centered_20261026T173840/curiosity/round_001.pt \
  --output logs/teleop_mouse_validation
python -m exploration.diagnostics.teleop_wm_validation --stage extract \
  --output logs/teleop_mouse_validation
python -m exploration.diagnostics.teleop_wm_validation --stage replay \
  --output logs/teleop_mouse_validation
```

The record stage opens a scene viewer with a teleoperation panel. Select a skill and click Begin, then drag the target gizmo at the EE using the mouse. The existing EE-delta controller moves the arm toward that target; the robot is never teleported. Run/Pause controls whether an active segment advances physics at 20 Hz. Use Open/Close gripper, End to finish the segment, and Finish/save after closing the last segment. Available labels are `move_delta_ee_pose`, `approach`, `lift`, `drop`, `open_gripper`, and `close_gripper`. The `approach` primitive includes closing the gripper. Only target translation is used; wrist orientation remains fixed. Paused viewing does not advance physics.

New manual recordings place the cup and bowl 1 cm farther apart on each side: their initial lateral center spacing is 14 cm instead of 12 cm. Only this diagnostic's spawn positions change; object geometry, shelf height, robot pose and training scenes remain unchanged. The extra 2 cm is recorded as `object_spacing_increase_m` in the demonstration manifest. Primitive replay uses that recorded layout; older demonstrations without the field retain their original 12 cm layout. Restart the collector to create the new scene; existing recordings are preserved.

Manual recording has no default step cap and ends when you click Finish/save. This diagnostic removes the inherited 800-step EE episode limit and the nested simulator time limits, including during primitive replay; training episode limits remain unchanged. Replay ends after the extracted plan. Supply optional `--max-steps N` to explicitly cap a record/replay run. Actual termination or simulator errors still stop execution. Single-primitive parameter bounds still apply: end movement segments before exceeding X/Y +/-0.15m or Z +/-0.12m, then begin another segment. Longer demonstrations are not a guarantee of broader dynamics coverage.

For scripted headless checks, `--headless` retains terminal commands: `begin NAME`, `w/s` for root-frame +/-X, `a/d` for +/-Y, `r/f` for +/-Z (optional tick count, such as `w 5`), `o/c` for the gripper, `end`, and `quit`. Existing output directories are never overwritten; choose a new directory for each demonstration.

For independent arm-joint adjustment, add `--control-mode joints` to the record command and choose a new output directory. Its mouse panel provides seven named joint-angle sliders in radians, bounded by the robot's physical joint limits, plus gripper and Begin/End/Run/Pause controls. Joint targets execute through the existing joint-position controller and physical simulation; this mode does not hold the wrist orientation or teleport the arm. Physical collisions can prevent reaching a requested angle. This mouse-only diagnostic records a distinct 8D action (seven absolute joint targets plus normalized gripper), alongside its source checkpoint contract. It cannot be passed to compact6 extract/replay; use the default `--control-mode ee` for the existing primitive validation workflow. Direct joint control alone does not establish an obstacle-free path into the cabinet.

For mouse-controlled position and orientation with whole-arm path planning, use the explicit pose mode in the existing NumPy 1.26 runtime (MPLib 0.1.1 cannot run with NumPy 2):

```bash
.runtime/thor-ceeus/bin/python -m exploration.diagnostics.teleop_wm_validation --stage record \
  --control-mode pose \
  --checkpoint logs/raps_compact6_train_400k_episode40s_centered_20261026T173840/curiosity/round_001.pt \
  --output logs/teleop_pose_validation
```

In the pose panel, click Begin segment at the top and confirm the visible recording state changes to `pose_control`, drag the gizmo in Translate or Rotate mode, then click Plan & Execute. Target editing does not advance physics. The existing planner checks the complete arm against the live scene and then executes its joint trajectory; it does not teleport the robot or fall back to direct EE motion when planning fails. For intentional grasp contact, explicitly select the target object in the finger-contact selector; this permits only the existing planner's finger/object contact exception, not arm/cabinet contact. Gripper buttons hold the measured arm configuration for 30 control ticks. Pause/resume, cancel, and end controls retain partial executions and their status. A failed planning request is recorded with zero physical transitions, rather than inventing a failed simulator action.

Pose requests, planner outcomes and planned trajectories are saved separately from dense actual 8D joint-control transitions, with target frames, source checkpoint contract and execution feedback. These are a new pose/macro diagnostic action definition and cannot enter old compact6 extract/replay. Finite planner collision checks and execution feedback do not guarantee continuous collision freedom or grasp success; an unreachable or obstructed target is an expected rejection. The per-request planner/execution budgets remain finite even though the manual session has no default total step cap.

Extraction rejects unsupported directions, gripper changes, incomplete segments, and out-of-range parameters instead of clipping them. Replay checks the seeded initial state and contracts before stepping. Actual primitive execution may differ from the demonstration; its completed nonzero-duration endpoints, including failed outcomes, populate `replay/evaluation.npz`. Dense replay logs retain incomplete attempts as well. This is a human-designed diagnostic dataset, not exhaustive dynamics coverage; keep it separate from training replay and split complete episodes if collecting additional data for training.

### Household environments

```bash
python main.py room --help
python main.py household --help
python main.py household --check
python main.py thor --help
```

- `room`: Habitat-Sim small-room viewer; prepare its runtime with `scripts/setup_room_viewer.sh`.
- `household`: PARTNR skills `Navigate / Pick / Place / Open / Close`; requires upstream code, a Habitat runtime, and scene assets. Use `--check` to inspect prerequisites.
- `thor`: iTHOR / ProcTHOR entry; dependencies are listed in `requirements/thor.txt`.

Help and prerequisite checks do not establish that full simulation works. Setup scripts may download assets and change runtimes; run them as needed.

### FloorPlan1 CEE-US world model

`exploration.cee_us.cee_us_thor` reuses the pinned official state-GNN ensemble, normalizers, delta/MSE training and member-wise rollout. Categorical CEM replaces continuous iCEM. It searches 8 navigation/posture and 6 interaction types; a fixed executor chooses the nearest visible object with the required capability. Put chooses a receptacle excluding the held object and retains `placeStationary=True`. No automatic navigation, forced interaction, task-target preference or object randomization is used.

Prepare the isolated combined runtime without changing the existing THOR or training environments:

```bash
bash scripts/setup_thor_cee_us.sh
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.cee_us_thor --help
```

The setup reuses PyTorch from `rlPractice`; set `CEEUS_PYTHON` to its Python executable on another machine. It installs only iTHOR runtime dependencies, not ProcTHOR. The existing pinned Unity build must already be available (`scripts/setup_thor.sh unity` downloads it explicitly if needed). The standard smoke/training collector uses local graphics unless `--metadata-only` is passed; the designed validation collector below defaults to metadata-only headless Unity. Offline evaluation does not start Unity. CPU is the default; unavailable CUDA requests fail explicitly.

GPU training, CEM planning and periodic validation use the same FP32 GNN with `--device cuda`. The local RTX 2000 Ada Laptop GPU (8GB) is accessible from the host, but this session's default sandbox hides NVIDIA device nodes. Run GPU commands from a GPU-visible terminal or an explicitly approved host execution context; missing sandbox device access does not imply a broken driver. No driver/package replacement was needed.

For the hidden-dimension-32 FloorPlan1 pilot, measured candidate chunks of 4 were about as fast as chunks of 8/12 and used less memory. Keep the training minibatch at its original 4 and use `--cem-chunk-size 4 --eval-batch-size 4`; larger batches did not improve per-sample prediction throughput in this bounded test. CEM reuses rollout storage while chunk dimensions match. These controls preserve candidate count, iterations, horizon, ensemble size, loss and optimizer; batched/device arithmetic can differ slightly in floating point. Existing CPU defaults remain available. Chunk size is recorded in checkpoint planning settings; exact resume still checks sources and settings and must not be used to silently migrate an older experiment.

Run a new bounded GPU learning-curve pilot:

```bash
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.cee_us_thor train \
  --device cuda --metadata-only --cem-chunk-size 4 --eval-batch-size 4 \
  --eval-data outputs/thor_cee_us/learning_curve_subset_20261109/evaluation.npz \
  --bootstrap-steps 64 --rounds 3 --steps-per-round 32 --epochs 1 \
  --hidden-dim 32 --horizon 3 --candidates 12 --iterations 2 --elites 4 \
  --eval-every 1 --output outputs/thor_cee_us/gpu_learning_curve
```

Reproduce the inference/memory check using an existing checkpoint:

```bash
.runtime/thor-ceeus/bin/python -s -m scripts.benchmark_thor_gpu \
  --checkpoint outputs/thor_cee_us/learning_curve_subset_pilot_20261109/curiosity.pt \
  --output outputs/thor_cee_us/gpu_benchmark --fit-smoke
.runtime/thor-ceeus/bin/python -s -m artifacts.thor_learning_curve.evaluate_full \
  --checkpoint outputs/thor_cee_us/gpu_learning_curve/curiosity.pt \
  --eval-data outputs/thor_cee_us/instance_validation_frozen_20261109/evaluation.npz \
  --device cuda --batch-size 4 \
  --output outputs/thor_cee_us/gpu_learning_curve/full_evaluation.json
```

This tests CPU/CUDA predictions, three paired CEM seeds at chunks 1/4/8/12, prediction batches 4/8/16, and an optional discarded eight-training-transition fit. It never starts Unity or trains on frozen validation. The measured chunk-4 planning median was 0.272s with 0.802GiB peak allocated device memory; a separate 24-action CUDA loop averaged 0.353s per CEM decision. Full4979 CUDA validation completed in 28.6s, including overlap with another small GPU check, and its headline metrics agreed with the saved CPU report (rtol 1e-4, atol 1e-6). These are small-model measurements, not guaranteed speedups or memory limits for larger configurations. Reports are in `outputs/thor_cee_us/gpu_benchmark_20261109/` and `gpu_loop_smoke_20261109/`.

The earlier 9.3s subset validation overlapped a separate full GPU evaluation; it is not an isolated-validation baseline. Standalone 512-transition measurements are around 2–3s with this hidden32 model. Example displays reuse the first 16 predictions from the metric pass, avoiding 16 extra forward calls without changing samples, groups or metrics. A 5000-attempt experiment still requires thousands of planning decisions and repeated fitting of cumulative replay; planning-only estimates do not predict total runtime, especially when increasing training epochs.

Run the bounded smoke test (use a new output directory for every command):

```bash
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.cee_us_thor smoke \
  --output outputs/thor_cee_us/smoke
```

This freezes separate validation episodes, collects eight random bootstrap attempts, trains a small 5-member GNN, executes three CEM decisions, checks checkpoint prediction equality and evaluates horizons 1/3/5. Validation includes a fixed default-pose interaction sequence and bounded explicit initial-pose setup to seek missing interaction coverage. Any setup actions are separate from recorded evaluation transitions. Inspect actual coverage; unsupported success cases are never fabricated. The tiny smoke model is a wiring check, not evidence of learned general dynamics.

For a small explicit random/curiosity comparison, first freeze evaluation data, then run both variants under the same attempted-action budget:

```bash
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.cee_us_thor collect-eval \
  --steps 20 --output outputs/thor_cee_us/eval
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.cee_us_thor compare \
  --eval-data outputs/thor_cee_us/eval/evaluation.npz \
  --bootstrap-steps 16 --rounds 1 --steps-per-round 10 --epochs 1 \
  --hidden-dim 16 --horizon 2 --candidates 4 --elites 2 --iterations 1 \
  --output outputs/thor_cee_us/compare
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.cee_us_thor evaluate \
  --checkpoint outputs/thor_cee_us/compare/curiosity.pt --hidden-dim 16 \
  --eval-data outputs/thor_cee_us/eval/evaluation.npz \
  --output outputs/thor_cee_us/evaluation
```

`train` defaults to curiosity; `train --exploration random` uses the existing random collector without constructing CEM, retaining the same five-member WM and training/evaluation settings. Random action types use the same navigation/interaction50/50 prior as bootstrap, not uniform sampling over14 actions; target selection remains nearest-visible-capable. Checkpoints record the exploration policy and explicit restore mismatches fail. `compare` runs both from the shared random bootstrap. These stages require explicit budgets and a frozen evaluation dataset. `--checkpoint` resumes at collection/episode boundaries; use matching seed, hidden dimension, CEM settings and unchanged source. `--rounds` specifies additional rounds. Mid-episode Unity restoration is not supported. `report` performs the same offline evaluation as `evaluate`.

Outputs include `manifest.json` (versions, source hashes, protocol, evaluation hash), `contract.json`, independent `evaluation.npz`, model/replay/RNG checkpoints, per-round diagnostics, stratified prediction reports, actual validation ego/topdown frames and a PNG table of actual versus predicted state. The GNN does not synthesize predicted RGB. No-target attempts are local failures without a Unity call and still count against the exploration budget; a 50/50 navigation/interaction prior is shared by random exploration and CEM initialization.

The optional `--prediction-mode mixed` is a separate THOR research variant; the default `mse` retains the upstream numeric-delta objective. Mixed mode keeps the five-member GNN, edge messages and mean aggregation. Each object/global readout shares two hidden layers and branches into continuous regression and classification linear heads. Agent position/yaw/horizon (indices 0–6) and object position/rotation/openness use normalized delta MSE. Standing, action outcome, the twelve object flags and multi-hot parent relations use absolute next-state BCE-with-logits. Liquid (four classes) and temperature (three classes) use separate cross-entropies; their fields remain represented even when an action does not change them. Static fields are copied exactly. No labels are inferred from changes or from action types.

Mixed regression is the equal average of the agent and object continuous mean losses. Mixed classification is the equal average of six mean losses: standing, action success, object flags, parent relations, liquid and temperature. The objective is `regression + classification`, summed over the five ensemble members. These are explicit initial research weights, not weights tuned on validation data. The two loss families and per-task averaging intentionally change training relative to the MSE baseline; in particular success no longer receives only one coordinate's weight out of all dynamic outputs.

The separate `--prediction-mode mixed-relations` variant gives each object readout three linear heads after the existing shared trunk: continuous regression, attribute classification (flags/liquid/temperature), and multi-label parent relations. The global agent readout is unchanged. The relation head outputs only actual receptacle instances from the frozen scene capabilities. Its BCE excludes self-parent candidates; decoding writes zero for all invalid parent pairs and preserves the existing full state layout. It retains multiple simultaneous parents, rather than a single-parent softmax. Current FloorPlan1 has43 receptacle candidates and3268 valid directed relation slots, versus5929 unfiltered slots; other scenes use their own contract.

Before each fit, the new variant computes one global relation `pos_weight = N_negative / N_positive` over valid next-state labels in the training buffer. Either class absent gives weight1; an empty valid mask gives zero relation loss. This weight stays fixed during minibatches and evaluation; monitoring data do not update it. Invalid positive labels fail explicitly. Counts and weight are registered model buffers saved in checkpoints. Relation BCE remains one of the original six equally weighted classification terms; other losses, thresholds, normalization and action semantics are unchanged. Weighted sigmoid scores need not be calibrated probabilities, and positive weighting changes their decision tradeoff; evaluate relation F1, not the mostly-negative accuracy alone. The old `mixed` two-head/unweighted baseline and original `mse` remain available. Choose the new mode explicitly for a fresh experiment; old mixed checkpoints do not convert into new-head checkpoints. Both online and offline fitting accept, for example, `--prediction-mode mixed-relations --gns --gns-radius 0.5`.

For an already-running online experiment, `scripts.monitor_thor_online` reads the saved bootstrap/round checkpoints and uses the exact `scripts.fit_thor_offline.assess` diagnostics to supplement the recorded metrics with action-success BCE/Brier/balanced accuracy. It verifies the dataset hash and agreement with existing MAE/accuracy/F1 values, preserves the original history, and writes `standardized_learning_curve.json`. It uses the same four-panel plotter with `--online`: action-success accuracy, action-success BCE, changed-coordinate MAE, and parent-relation F1. Online x values are training attempts; offline x values are epochs. This observer performs no fitting and changes no trainer/source fingerprints. Re-evaluated online reports explicitly describe their evaluation role rather than inheriting the offline in-sample label. The original standalone relation plot is superseded by the relation panel in the unified learning curve.

The online monitor optionally accepts `--reference-history PATH` to pin a completed offline epoch as a horizontal reference in each of the same four panels. Its path, exact history SHA256 and selected epoch are stored in `standardized_evaluation_manifest.json`; the plotter checks the offline monitoring-data hash matches the online evaluation hash and fails if the selected history changes. References are explicitly labeled as in-sample fitting, not a lower bound or a matched training-budget comparison. The current run uses the three-head GNS offline epoch18, its last completed checkpoint. The reference metadata can be added while the observer is running; subsequent plot renders read it without restarting training.

The offline fitting plotter also accepts repeated `--compare-history PATH` arguments to overlay previous runs, with labels read from their manifests. It requires identical training and monitoring dataset hashes and plots action-success accuracy/BCE, changed-coordinate MAE and parent-relation F1. Each line ends at its actual last recorded epoch; stopped baseline runs are not extrapolated. These remain in-sample comparisons. The mixed-relations experiment changes head layout, legal candidates and relation loss together, so a difference cannot isolate which component caused it.

The optional `--gns` enables a GNS-inspired dynamic sparse graph for either `mse` or `mixed`; omitting it retains the full graph. `--gns-radius 0.5` sets the estimated box-gap threshold in metres (default 0.5 for new GNS runs). At every forward call, including each member-wise imagined rollout step, edges are rebuilt from the current input: spatial neighbors, bidirectional parent relations, and bidirectional held-object/Put-target relations. No future labels enter construction. Geometry uses object positions and canonical initial AABB sizes, so it approximates proximity rather than THOR collision detection; it does not model swept volumes, opening geometry or guarantee all possible contacts. Binary relations use the existing 0.5 threshold on raw states or predicted probabilities. All nodes and global agent/action context remain. Local and global means divide by retained-edge counts, with zero aggregation for isolated nodes. This changes the graph and thus the prediction function; accuracy must be compared experimentally.

The implementation in `exploration/cee_us/thor_gns.py` reuses the existing MLP parameters and optimizer without modifying upstream sources. Selected edges are packed per ensemble member and padded to the largest member edge count; only that packed list passes through the edge MLP. Adjacency search still uses quadratic pairwise geometry, but discarded pairs do not run the message MLP. Sparse graph construction and packing introduce overhead, so speedups are not guaranteed. Online and offline entry points accept the same flags. Manifests/checkpoints record graph mode/radius; online restore infers saved settings and rejects explicit mismatches. Legacy checkpoints without graph metadata remain full graph. Switching graphs is a new experiment, not exact resume of a baseline. The method borrows radius-based rebuilding from [GNS (ICML 2020)](https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf), with THOR-specific relation additions; it is not a reproduction of the particle simulator.

For an explicitly in-sample fitting diagnostic, `scripts/fit_thor_offline.py` can train a fresh mixed model directly on the recorded evaluation dataset. This is intentional training on validation data, not held-out evaluation or evidence of exploration performance. It starts no Unity instance, preserves source data, fits normalizers on the training file only, records every epoch, and saves a distinct `thor-offline-fit-v1` checkpoint that the online resume command rejects. This diagnostic does not train on online curiosity replay. The fixed512 monitor is a subset of the full4979 training records, so its curve measures fit, not generalization. Five epochs are an early checkpoint, not a convergence criterion.

```bash
.runtime/thor-ceeus/bin/python -s -m scripts.fit_thor_offline \
  --data outputs/thor_cee_us/instance_validation_frozen_20261109/evaluation.npz \
  --monitor-data outputs/thor_cee_us/learning_curve_subset_20261109/evaluation.npz \
  --epochs 20 --hidden-dim 128 --device cuda --seed 0 \
  --output outputs/thor_cee_us/gpu_offline_validation_fit_20261109
MPLCONFIGDIR=/tmp/mpl-thor-offline python3 scripts/plot_thor_offline_fit.py \
  --history outputs/thor_cee_us/gpu_offline_validation_fit_20261109/history.json \
  --output outputs/thor_cee_us/gpu_offline_validation_fit_20261109/fitting_curve.png
```

Plotting uses the existing system matplotlib environment; the training runtime lacks its `cycler` dependency. The trainer writes `history.json`, latest model/optimizer checkpoint, provenance/source snapshots and, on completion, a full training-data fitting report. Success probability BCE/Brier, balanced accuracy and confusion counts complement threshold accuracy. No claim that exploration is the only bottleneck follows from a single converged fit; failure to fit within a finite budget also does not prove the architecture cannot represent the transitions.

Inference adds continuous deltas to the current state and decodes classification logits with sigmoid/softmax directly into next-state probabilities, without adding old binary values. The existing target resolver thresholds probabilities at 0.5 during imagined rollouts. CEM still sums ensemble standard deviations of decoded states over the planning horizon; it now includes probability disagreement, so curiosity scores and numerical training losses are not directly comparable between prediction modes. No new entropy bonus, task mask, hard category projection, learning rate or automatic loss balancing is introduced. Checkpoints record the prediction mode; restoration infers it when omitted and rejects an explicitly different mode. Old checkpoints without this field are MSE. Switching modes requires fresh training, and exact resume still requires the checkpoint's original source version.

A bounded mixed-mode GPU integration check (20 online attempts, not a learning-quality experiment):

```bash
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.cee_us_thor train \
  --prediction-mode mixed --device cuda --metadata-only --hidden-dim 128 \
  --eval-data outputs/thor_cee_us/learning_curve_subset_20261109/evaluation.npz \
  --bootstrap-steps 16 --rounds 1 --steps-per-round 4 --epochs 1 \
  --horizon 2 --candidates 4 --iterations 1 --elites 2 --cem-chunk-size 2 \
  --eval-every 1 --eval-batch-size 4 --seed 0 \
  --output outputs/thor_cee_us/gpu_mixed_smoke_20261109
```

State features use all default FloorPlan1 metadata objects, fixed instance identities and parent-receptacle multi-hot relations. Training runs adopt the frozen dataset's initial scene schema, including canonical AABB sizes, because initial physics can change observed AABB sizes across Unity launches. Object IDs, types, capabilities and mass must still match; evaluation transition targets are never used to fit normalizers. Boolean states are regressed with the original MSE, not calibrated probabilities; a 0.5 threshold is used for evaluation and future target resolution without clipping rollout states. This first version does not establish cross-scene generalization. The all-object fully connected graph costs substantially more than the two-object cabinet; normal model defaults are wider than smoke settings and should be budgeted explicitly.

### Designed FloorPlan1 validation collection

Collect independent scripted trajectories using the frozen scene contract:

```bash
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.thor_designed_validation \
  --reference-data outputs/thor_cee_us/smoke_2/evaluation.npz \
  --max-attempts 600 --max-poses 6 \
  --output outputs/thor_cee_us/designed_validation
```

The reference supplies canonical object identities/static metadata only. Evaluation transitions are not used for training or normalization. Cases enumerate openable, toggleable, pickupable and receptacle instances, plus navigation, no-target and default-pose controls. Initial placement and held-object preparation are logged separately in `setup.jsonl`; retained episodes contain no setup actions. Each episode ends after its fixed sequence (6 interaction/control steps, 8 navigation steps or 9 default mixed steps), including failed attempts. Cases that cannot be prepared within the pose budget are skipped explicitly. The total attempt cap never truncates an episode.

This collector defaults to headless metadata-only Unity, without RGB, segmentation, map cameras or saved images. Use `--save-frames` to enable rendering and initial episode frames for debugging. The state/action representation and nearest-target executor are unchanged. `rows.jsonl`, `coverage.json` and `manifest.json` record actual targets, failures, skipped cases and provenance; `evaluation.npz` can be passed directly to `compare` or `evaluate` via `--eval-data`. A generic `state_changed` flag includes visibility changes and floating-point physics motion; it is not proof of the intended interaction effect.

The first frozen collection is `outputs/thor_cee_us/designed_validation_20261109/evaluation.npz`: 355 attempts across 56 episodes, 203 successes and 152 failures. All 114 planned cases were considered; 58 were skipped during setup. It includes rendered debug frames because it completed before metadata-only collection was enabled. This is a bounded coverage baseline, not uniform coverage of all dynamics, object placements or source/receptacle combinations. No model training was performed for collection.

### Instance-level dynamics validation

The earlier 114-case collector is retained as a bounded scripted baseline. For instance-level completion accounting, use:

```bash
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.thor_instance_validation \
  --reference-data outputs/thor_cee_us/smoke_2/evaluation.npz \
  --min-count 10 --max-attempts 5000 --max-poses 1024 \
  --output outputs/thor_cee_us/instance_validation
```

`thor_coverage.InstanceCoverage` defines 175 explicit FloorPlan1 requirements: both open/close directions for 22 instances, both toggle directions for 9, Pickup and Release via Put for 30, an observed new parent relationship from a successful Put into each of 43 receptacles, and 10 navigation/posture effects (including failed, stationary MoveAhead and MoveBack). Coverage requires actual before/after fields and actual resolved targets; repeated idempotent actions do not count. Setup does not contribute to coverage. `--min-count` specifies the required effect count per item, not the number of attempted actions. The global attempt cap is a safety bound and cannot guarantee completion.

The metadata-only collector searches standing/crouched poses and horizons -30/0/30/60, opens necessary parent/target containers using native actions, checks inverse-action target changes, and tries distinct source objects for receptacles. Successful source choices are reused, with different starting locations tried before repeated camera variants. Initial agent placement and all preparation are logged separately. Consecutive retained actions share an episode only when no setup intervenes. Objects are not teleported, forceAction remains false, and the nearest-target execution rule is unchanged.

Inspect the running `coverage.json`: `complete` is true only when every fixed requirement reaches its count; `missing` is never removed merely because a case fails. All attempts are retained in `evaluation.npz`, with labels in `rows.jsonl` and setup diagnostics separately. Source/version/reference/dataset hashes appear in `manifest.json`. Each output requires a new directory; prior data are preserved. This finite checklist measures instance-level effects, not all possible room layouts or source/receptacle combinations. The first expanded one-observation audit (`instance_audit_20261109`) observed 144/175 requirements using 293 attempts; it remains incomplete.

Failure-control collection uses the same native executor and separately labels deliberate invalid-precondition attempts:

```bash
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.thor_instance_validation \
  --reference-data outputs/thor_cee_us/smoke_2/evaluation.npz \
  --min-count 1 --max-attempts 1200 --failure-controls \
  --output outputs/thor_cee_us/instance_failure_controls
```

Controls include already-open/closed, already-on/off, occupied-hand Pickup and empty-hand Put. Labels describe the intended test condition; actual returned success/failure and errors remain authoritative. These control attempts are retained even when the positive effect has already been observed elsewhere. Failure observations contribute to outcome coverage and world-model validation; they do not satisfy a different successful-effect requirement. A claim of exhaustive failure coverage needs an explicit feasible precondition checklist, not an assumed success/failure pair for every possible action.

An offline interactive example is `artifacts/thor_dynamic_coverage_graph/coverage_graph.html`, with auditable `graph_data.json` and artifact-local `render_graph.py`. The graph includes all metadata instances and current actions, separates unattempted/success-only/failure-only/both outcomes and local no-target failures, and supports object filtering and detailed error counts. It is an object-action outcome graph, not the complete continuous state-transition graph. Snapshots of running JSONL inputs include only complete records and record the exact bytes read; regenerate after collection to update it. Preparation failures are not world-model transitions and are excluded.

The pooled frozen benchmark is `outputs/thor_cee_us/instance_validation_frozen_20261109/evaluation.npz` (4979 attempts: 2146 successes and 2833 failures). It combines the original independent 355-transition baseline, 3281-transition main collection, 1200 deliberate-control transitions and 143 placement-supplement transitions. Episode IDs are shifted to keep sources separate; setup and diagnostic pilots/audits are excluded. Pass this NPZ directly as `--eval-data` for evaluation. All four source datasets remain intact, with hashes in the pooled manifest.

For a bounded learning-curve pilot, freeze a 512-transition proportional action/outcome-stratified subset, use the same subset after bootstrap and each online round, then evaluate all 4979 transitions once at the end:

```bash
python3 artifacts/thor_learning_curve/prepare_subset.py \
  --source outputs/thor_cee_us/instance_validation_frozen_20261109/evaluation.npz \
  --output outputs/thor_cee_us/learning_curve_subset --size 512 --seed 10001
.runtime/thor-ceeus/bin/python -s -m exploration.cee_us.cee_us_thor train \
  --eval-data outputs/thor_cee_us/learning_curve_subset/evaluation.npz \
  --bootstrap-steps 64 --rounds 3 --steps-per-round 32 --epochs 1 \
  --hidden-dim 32 --horizon 3 --candidates 12 --iterations 2 --elites 4 \
  --metadata-only --eval-every 1 --output outputs/thor_cee_us/learning_curve_pilot
MPLCONFIGDIR=/tmp/mpl-thor-learning python3 -m exploration.cee_us.thor_learning_curve \
  --history outputs/thor_cee_us/learning_curve_pilot/curiosity_learning_curve.json \
  --output outputs/thor_cee_us/learning_curve_pilot/learning_curve.png
.runtime/thor-ceeus/bin/python -s -m artifacts.thor_learning_curve.evaluate_full \
  --checkpoint outputs/thor_cee_us/learning_curve_pilot/curiosity.pt \
  --eval-data outputs/thor_cee_us/instance_validation_frozen_20261109/evaluation.npz \
  --output outputs/thor_cee_us/learning_curve_pilot/full_evaluation.json
```

The subset manifest preserves exact source indices, random seed, source/subset hashes and stratum counts. Every nonempty action/outcome stratum receives at least one sample; other allocations approximate source proportions. Gaps in sampled episodes create new episode boundaries. Periodic reports stream conditional one-step predictions in small batches and retain action/outcome strata, a zero-change baseline and changed-coordinate errors. Changed-coordinate MAE mixes the original feature units and is a diagnostic, not a physical-unit accuracy score. This mode omits the more expensive free-running rollout evaluation; the existing `evaluate` command still reports horizons 1/3/5. Evaluation transitions never enter the training replay or normalizer fitting. Each evaluated round saves a checkpoint and the JSON curve is refreshed after every assessment. The x-axis counts online training attempts, including failures; it is not the size of the validation set. Keep the final full-data report separate from the subset curve because their distributions differ slightly. This small single-seed model/planner budget checks the loop and does not establish a curiosity advantage over random exploration.

The final positive-effect/navigation checklist reaches 161/175 items at least ten times (163 observed at least once); it is explicitly incomplete. The object-action graph has 134/135 applicable object-action relations attempted, with SaltShaker Pickup unobserved. These are different denominators: failed actions cover observed outcomes but do not count as successful state transitions. The paired-state graph classifies 2061 successful attempts with relevant effects and 85 successes with no relevant effect; all 2833 failures are preserved. Observed repeated Open/Close controls return success without changing the target flag in this pinned runtime, whereas repeated toggles, occupied-hand Pickup and empty-hand Put yield failures. Do not assume an idempotent native action must fail.

Use `--reuse-placements-from previous/rows.jsonl --targets EXACT_OBJECT_ID ...` to supplement selected instances with native revalidation of previously successful source/pose combinations. It first tries a varied pose and, if Put fails while the same source is still held, can retry the known pose in a separately logged setup/new episode. Selection never removes missing items from the ledger. Initial-placement changes for inaccessible/colliding targets remain a separate experimental decision; no such object rearrangement was performed here.

Refresh the graph from the matching frozen rows and report:

```bash
MPLCONFIGDIR=/tmp/mpl-thor-coverage-graph python3 artifacts/thor_dynamic_coverage_graph/render_graph.py \
  --rows outputs/thor_cee_us/instance_validation_frozen_20261109/rows.jsonl \
  --coverage-report outputs/thor_cee_us/instance_validation_frozen_20261109/coverage.json --png
```

## Using the reorganized directories

Exploration modules now live in subpackages. For example, `exploration.cee_us_cabinet` became `exploration.cee_us.cee_us_cabinet`, and `exploration.diagnose_cabinet` became `exploration.diagnostics.diagnose_cabinet`. Update external scripts using old module commands. Calls to `main.py` are unchanged.

Dependency lists are in `requirements/thor.txt` and `requirements/partnr_oracle.txt`; setup scripts use the new paths. Standalone datasets and experiment outputs remain in place. CEE-US source is in `exploration/cee_us/official/`; RAPS and PointWorld submodules are in `exploration/raps/official/` and `exploration/pointworld/official/`. Model files inside those submodules moved with them, and references were updated. Existing source-fingerprint checks remain active: reorganized source is not the same version as an old run. Use a new output directory when diagnostics require it rather than bypassing checks.

Choose tests according to the changed modules, in an environment with their dependencies installed:

```bash
conda run -n rlPractice python -m pytest -q tests/cee_us/test_cee_us_cabinet.py
conda run -n rlPractice python -m pytest -q tests/pointworld/test_pointworld_cabinet_offline.py
```

## Experiments and maintenance

RAPS, PointWorld, and ManiSkill retain their independent Git histories without forks. The main repository saves their modifications relative to pinned commits in `patches/*.patch`; `patches/upstream.json` records paths, base commits, and patch hashes. CEE-US source is managed directly in the main repository.

After changing those upstream checkouts, run `python scripts/snapshot_upstream_patches.py`. Before committing, run `python scripts/snapshot_upstream_patches.py --check` and include updated patches and the manifest. Snapshots include staged and unstaged tracked modifications; stage intended new files within the submodule first and ignore runtime artifacts. The script does not commit, push, or watch files in the background.

On another machine, initialize upstream code with `git submodule update --init --recursive`, verify versions against `patches/upstream.json`, then apply the corresponding patches. From the project root:

```bash
git -C exploration/pointworld/official apply --check ../../../patches/pointworld.patch
git -C exploration/pointworld/official apply ../../../patches/pointworld.patch
git -C domain/ManiSkill apply --check ../../patches/maniskill.patch
git -C domain/ManiSkill apply ../../patches/maniskill.patch
```

Apply only to worktrees that do not already contain these changes. Skip empty patches. If `--check` fails, inspect versions and existing modifications rather than forcing an overwrite. Patches preserve modifications, but only committing and pushing the main repository creates a remote backup.

- Algorithm definitions, coordinate systems, units, control frequencies, action semantics, random seeds, data splits, and metrics are experimental protocol. Record the rationale for changes and disclose differences in comparisons.
- Distinguish direct control steps from skill decisions. Different skill durations mean equal decision counts do not imply equal environment interaction budgets.
- Prefer tests relevant to the change. Large training runs, simulations, and downloads require explicit authorization.
- Update this README when goals, module responsibilities, or main entry points change. Record individual development tasks in [docs/LOG.md](docs/LOG.md).
- Agents start with root [AGENTS.md](AGENTS.md), then follow [docs/AGENTS.md](docs/AGENTS.md). Commit only when requested, and subsequently record the mapping between log entries and actual Git commits.
- Write project-owned files in English, including documentation, comments, plans, and logs. Preserve upstream source and generated patches when changing their language would interfere with reproduction.
