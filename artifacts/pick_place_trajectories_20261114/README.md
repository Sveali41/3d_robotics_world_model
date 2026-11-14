# Stopped pick/place experiments: full-control trajectory viewer

Open `trajectories.html` in a browser. Keep the four sibling `.js` data files beside it; all scripts and Plotly are local, with no CDN or server required. In remote VS Code, download this entire directory or serve the project through a forwarded HTTP port. A VS Code source editor does not execute the interactive page.

Four independent runs are included, with unequal current physical-control budgets. This viewer is descriptive, not a matched-budget policy evaluation. It contains all recorded bootstrap and online control positions, including the stopped run's tail beyond the latest saved checkpoint. Source: `logs/raps_pick_place_four_baselines_20261113_1626`. No evaluation run or new simulation was launched.

- Choose an arm and episode to inspect a whole episode.
- Filter primitive or status to find macro records, then select a table row to inspect that macro's full control trajectory.
- Success means the recorded macro status is `success`, not merely a bilateral-contact snapshot. Random has three successful grasp macros and no successful place macro.
- Overview rendering uses a stated stride for responsiveness. Individual selected trajectories use every recorded control point. Episode reset discontinuities must not be interpreted as robot motion.
- TCP, cup and bowl positions use the robot root frame; plots display centimeters. Relative Z is displacement from the displayed segment's start, not automatically a confirmed lift event. The plots show measured positions, not a mesh rendering, video or a world-model prediction.
- Failures and budget-truncated macros remain visible. `interrupted_unfinished` identifies physical ticks for a macro whose final record was absent at interruption.

`summary.json` records exact counts and source paths. `.js` payload coordinates preserve source float32 values via base64, rather than decimal rounding. The builder verifies coordinate round trips, all macro time ranges, coverage of the complete control history, and sequential recovered tail steps.

Rebuild with the existing simulation Python environment (imports reuse the repository's Plotly extraction helper):

```bash
apptainer exec --nv /data/data/siyao/3d_world_model.sif env PYTHONPATH=/tmp/raps-three-arm-ls76ex80/numpy126:/data/data/siyao/3d_robotics_world_model OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python artifacts/pick_place_trajectories_20261114/build.py
```

This command only reads saved data and writes visualization artifacts; it does not run physics, train, or resume an experiment.
