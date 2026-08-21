# Project layout

The supported entry point is `python main.py <command> ...`.

```text
PointWorld/
├── pointworld/              # model, losses, metrics, checkpoint contract
├── dataset_components/      # WDS decoding and training pipeline
├── training/                # trainer and checkpoint persistence
├── evaluation/              # evaluation and visualization
└── integration/             # adapters used by external environments

domain/
├── ManiSkill/               # vendor ManiSkill checkout used by the kitchen env
└── robocasa/                # RoboCasa wrapper, low-level controller, RAPS primitives

exploration/                 # RAPS actor, trajectory archive, acquisition and selector
common/                      # small shared serialization and PointWorld adapters
bin/                         # legacy experiment/collection implementations and archived domains
main.py                      # single project CLI

logs/
├── outputs/                 # Hydra/collection outputs
├── pointworld_train/        # PointWorld training logs and checkpoints
└── curriculum_random_rollouts/

data/replay_manifests/       # replay selection metadata, kept with datasets
```

The old experiment and collection implementations live in `bin/`; they should
be invoked through `main.py`. The root is intentionally kept free of runnable
experiment scripts.

Examples:

```bash
python main.py robocasa --help
python main.py collect --config-name collect
python main.py train --config configs/pointworld.yaml
python main.py eval --model_path /path/to/model-last.pt
```
