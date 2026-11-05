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
| `configs/raps_gnn_actor_cabinet_400k_episode40s_centered.yaml` | Skill-level GNN actor–critic baseline |
| `configs/raps_gnn_actor_cabinet_400k_heuristic_only.yaml` | Skill-level heuristic-only comparison |
| `configs/raps_gnn_actor_cabinet_split7_planner.yaml` | Split7 primitives with collision-aware joint-trajectory execution |

`exploration.cee_us.cee_us_cabinet` provides `smoke / compare / train / evaluate / report`. The offline comparison entry provides `collect / audit / train-gnn / train-pointworld / evaluate`. Consult each entry's `--help` for parameters. Configured budgets do not indicate completed experiments.

The validation-only collector and planner diagnostics use the reorganized diagnostic package:

```bash
python -m exploration.diagnostics.collect_wm_validation --help
python -m exploration.diagnostics.calibrate_wm_validation --help
python -m exploration.diagnostics.validate_cabinet_planner --help
```

Existing datasets and checkpoint action contracts remain unchanged. New split7 data has a 13D skill action and cannot directly evaluate an older 12D world model. Diagnostic source fingerprints still require matching code versions.

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
