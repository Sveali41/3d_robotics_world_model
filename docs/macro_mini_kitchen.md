# Macro kitchen and extracted mini environments

The runnable example is [`bin/macro_mini_kitchen.py`](../bin/macro_mini_kitchen.py).
It registers three ManiSkill environments:

| Environment | Contents |
| --- | --- |
| `FullKitchen-v1` | cabinet `2005`, microwave `1001`, table, sink, and wall |
| `MiniCabinet-v1` | only cabinet `2005` and the ground |
| `MiniMicrowave-v1` | only microwave `1001` and the ground |

Run from the repository root:

```bash
conda run --no-capture-output -n rlPractice \
  python bin/macro_mini_kitchen.py --env full
conda run --no-capture-output -n rlPractice \
  python bin/macro_mini_kitchen.py --env cabinet
conda run --no-capture-output -n rlPractice \
  python bin/macro_mini_kitchen.py --env microwave
```

The file uses `KitchenAssetFactory` as the single source of truth. The macro
and mini scenes therefore construct the same geometry; the mini scenes merely
select one asset ID. The current assets are procedural boxes so the example is
immediately runnable. To use RoboCasa or URDF assets, replace the body of
`KitchenAssetFactory.build` while keeping its `(scene, asset_id, origin,
name)` interface.

The example is scene-only (`robot_uids="none"`, `obs_mode="none"`) so it can
be viewed without introducing a robot into the mini scene. For RL, pass a
robot UID and implement the task action/reward in a subclass, or use these
classes as the scene-loading base for an existing ManiSkill manipulation task.
With multiple environments, use the default `sim_backend="auto"` on a
working CUDA/PhysX installation; ManiSkill requires multiprocessing for CPU
vectorization.
