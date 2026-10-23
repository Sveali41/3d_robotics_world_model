# Switching the cabinet action interface

`action_mode=primitives` (default) preserves the 20D RAPS interface.
`action_mode=ee_delta` exposes a continuous 4D action:

```
[dx, dy, dz, gripper] in [-1, 1]
```

Each direct action executes exactly one environment control step (20 Hz by
 default). XYZ are normalized controller commands, not absolute positions or
values in meters. The existing Panda `pd_ee_delta_pose` controller maps them to
up to +/-0.1 m per axis in its root translation frame. Rotation commands are
zero, preserving wrist orientation. Gripper -1 closes and +1 opens. IK and PD
control remain inside ManiSkill; there is no waypoint state machine or action
repetition. Reaching a requested delta is not guaranteed by the controller.

Run a random interface check:

```bash
conda run -n rlPractice python main.py policy=random action_mode=ee_delta primitives=20
```

Train or evaluate explicitly:

```bash
conda run -n rlPractice python main.py policy=ppo action_mode=ee_delta max_primitives_per_episode=1000
conda run -n rlPractice python main.py policy=evaluate action_mode=ee_delta max_primitives_per_episode=1000
```

For the original interface, use `action_mode=primitives`. Random sampling, PPO,
evaluation and policy GIF rendering support both interfaces. Scripted grasp
sequences require primitives. The existing A/B/C/D diagnostic protocol remains
a primitive experiment, not a direct-action comparison.

The configuration keys `primitives`, `max_primitives_per_episode`, and
`visualize.max_primitives` count policy actions in either mode. In direct mode,
one such action is one low-level step. The unchanged default episode cap of 60
therefore means 3 seconds at 20 Hz; set the cap explicitly for an experiment.
The low-level episode budget still applies independently.

The existing PPO update is reused with a 4D tanh-Normal actor and no categorical
selector or parameter mask. Direct-mode checkpoints and training metrics go in
`<ppo.checkpoint_dir>/ee_delta/`, separate from primitive checkpoints. Policies
trained with one interface cannot be loaded into the other.

Reward weights, contact thresholds, observations and PPO hyperparameters are
not retuned by this switch. Movement (1.5 cm) and lift (1 cm) labels retain their
per-transition displacement definitions; in direct mode this is a single
control step. Stable-grasp counts persist over consecutive direct actions and
reset at episode reset. Consequently reward frequency, grasp-check cadence and
effective discount horizon differ from primitive mode. Equal update counts or
unchanged reward weights are not a controlled comparison; use actual low-level
steps and specify the temporal protocol before comparing exploration results.
No online WM or CEE-US planner is enabled by this flag.

Direct trajectory records contain the real 4D action with schema version
`cabinet-ee-delta-v1` and normalized-controller units in metadata. The legacy
record field name `raps_action_normalized` is retained in memory, but these
records cannot be converted with the RAPS `phase_view`/PointWorld sample path.
This change does not alter the formal world-model dataset or its split.

## D interaction preset with direct actions

```bash
conda run -n rlPractice python main.py policy=ppo action_mode=ee_delta exploration_mode=D
```

D uses the same R1 reward definition as the earlier diagnostic: grasp/lift
weights and extra TCP offset are zero; movement does not require a grasp.
Other weights and thresholds are retained. This encourages interaction, not
specifically grasping. Checkpoints are isolated in `ee_delta/D/`.

For direct actions, the proposed target is current TCP position plus the
controller-scaled XYZ delta. It is transformed into cabinet coordinates,
clamped to the existing exploration region, and transformed back, using the
same projection as primitive P1. It is converted to a normalized controller
command; gripper and rotation are unchanged. If the robot starts outside the
region, native per-step delta bounds can prevent reaching the projected target
in one tick. The record stores raw and projected targets, projection count,
and the actual bounded controller target. Projection guarantees neither actual
endpoint containment nor collision-free/IK-feasible execution. This is a direct
action adaptation of D, not an identical reproduction of the primitive D arm.
The per-action movement threshold and short direct-mode PPO horizon caveats
above still apply. The preset does not automatically retune them.
