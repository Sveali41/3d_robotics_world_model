# Bounded grasp/place primitive contract

This is the persistent behavior contract for the default-off RAPS `--pick-place-primitives` mode. Its executable, checkpointed specification is `PICK_PLACE_MACRO_CONTRACT` in `domain/robocasa/pick_place.py`. Legacy10, compact6 and split7 retain their own controllers.

## Grasp retention rule

**After confirming a grasp, keep the gripper closed and retain the object until placement confirms stable lower-shelf support.** Returning to the intermediate pose does not release the object. Selecting another grasp while holding must not open the gripper.

- Confirm physical grasp for four consecutive controls with a closed command before lifting. Closing alone does not establish grasp success.
- Check grasp and TCP-relative payload slip after every real control, including the final tick of a collection budget. At 20 Hz this is once per 50 ms, rather than every 100 Hz physics substep.
- Preserve the confirmed TCP-relative object pose across transport segments, replanning and the subsequent place macro. Replanning must not erase accumulated slip.
- More than 2 mm relative position slip, more than 2 degrees relative rotation, loss of grasp or an unexpected opening command fails the macro. Stop its motion, preserve closure and record the failure. A failed carry is not successful placement.
- Open only after four consecutive controls confirm upward cabinet-frame support above 0.05 N, contact geometry at the lower shelf and object speed at most 0.01 m/s. Unsupported placement stays closed. Confirm physical grasp contact clears before returning empty.
- Retention is selected explicitly with `--grasp-retention friction|constraint` (YAML `raps_actor.grasp_retention`). The default is `friction`; explicit CLI values override YAML. `constraint` requires `--pick-place-primitives` and single-environment CPU physics.

In `friction` mode, the planner's payload attachment only represents measured pose and collision geometry. Geometry, contact forces, friction and tracking determine retention; detection cannot undo a drop.

In `constraint` mode, four consecutive closed-gripper physical grasp controls trigger a native SAPIEN `PhysxDriveComponent` between `panda_hand` and the object. The three linear degrees of freedom are locked, and an explicit SLERP angular drive (stiffness 1e6, damping 2e3, acceleration mode) holds the measured relative rotation, without teleporting either body or changing friction. Binding persists through lift, transport and the intermediate pose. An opening command removes the drive **before the next physics step**; reset and environment close also remove it. On first valid lower-shelf contact, retain binding and logical closure but hold the measured native finger drive positions each physics substep. This removes continued squeeze pressure while seeking stable support. As explicitly authorized, constraint mode checks the maximum measured object displacement per physics-substep duration over the latest control; it records that speed and native linear speed together. Missing substep samples cannot confirm support. The 0.01 m/s, upward-force, geometry and four-control thresholds are preserved. Friction mode continues using native linear velocity. The place macro still requires stable support before commanding opening. This abstracts away subsequent frictional holding difficulty, as explicitly requested; it does not make initially missed grasps succeed.

After binding, grasp feedback means held by the active constraint even if finger contact forces disappear. Tick/skill records include `grasp_retention`. Drive integrity and measured relative pose remain checked; errors above 2 mm/2 degrees fail explicitly. This is numerical constraint enforcement, not an absolute guarantee under arbitrary solver failures. Resolved settings and action contracts distinguish the modes, and strict resume rejects switching between them. Keep their datasets and experimental comparisons separate. Neither mode teleports objects.

## Position interface

The action retains a grasp/place one-hot plus a shared three-slot argument bank (5D). `grasp(x,y,z)` specifies a TCP position; `place(x,y)` specifies the final object's collision-geometry bounding-box center projected onto the lower shelf. Coordinates are absolute controller-root meters. Place uses only the first two argument slots; the third is inactive and canonicalized to zero. Explicit place Z arguments are rejected.

| Parameter | X | Y | Z |
| --- | --- | --- | --- |
| Upper grasp TCP | 0.370–0.671 | -0.070–0.310 | 0.692–0.872 |
| Lower placed-object center | 0.370–0.671 | -0.070–0.310 | Automatic |

XY retains an 8 cm center margin inside the cabinet. Before place motion, the complete predicted object footprint must additionally remain at least 2 cm inside the cabinet's nominal XY edges; oversized or edge-overhanging targets fail `invalid_goal`, preserving closure. These are geometric bounds, not guarantees of robot reachability or collision-free transport. Grasp Z retains its nominal support +0–22 cm band with a 2 cm endpoint margin (+2–20 cm).

After four closed physical grasp controls, record the actual TCP-relative object pose and compute its collision AABB at the fixed placement orientation using all supported collision shapes. The reference is preserved through carrying and the following place macro. The center is the root-axis-aligned collision AABB center, not the asset origin or center of mass. A place started with an already confirmed payload captures this reference before moving; reset or a new empty-hand grasp clears it.

For collision bounds `lower, upper` measured relative to the TCP in root axes, the predicted shelf-touch TCP is `[place_x, place_y, lower_shelf_z] - [center_x, center_y, lower_z]`. This compensates lateral grasp offset and actual TCP height above the object bottom. The descent-entry TCP is 4 cm above that prediction. Calculated height guides motion; four stable supported controls still determine release. A requested final location is not a guarantee against physical drift after opening. This geometry-informed executor uses simulator payload geometry; no object type-specific offsets or automatic object selection are introduced.

Grasp opens, approaches until the requested position or bilateral physical finger contact, closes, confirms, lifts 8 cm and carries to intermediate TCP `[0.18, 0, 0.84]`. Place retreats, moves to the common lower-entry preplace TCP `[0.28, 0.15, 0.56]`, approaches its automatically computed descent-entry TCP, seeks support, releases and returns to the episode's initial TCP. After release, the empty hand rises to at least its descent-entry height and 4 cm above its release TCP, withdraws through preplace, then returns home. Return poses are TCP poses, not exact joint configurations. Requested object XY, predicted touch TCP and placement geometry/failures are recorded alongside ordinary phase goals and feedback.

## Planner and experiment compatibility

The flag selects `motion_execution: planner`, `action_mode: joint_position`, gripper-command memory and macro-specific safety gates. CLI on/off overrides YAML `raps_actor.pick_place_primitives`. A fresh 5D actor/world model is required; split7 checkpoints/replay are incompatible. Strict saved-config and action-contract checks remain active. The current controller is macro contract version 5; earlier macro contracts are rejected rather than silently changing resumed behavior.

Grasp and placement use the shared episode-reset orientation. Long transfers retain RRTConnect. As explicitly authorized, the preplace transfer may rotate only while TCP X<0.29 m, with the fixed orientation restored at its endpoint before lower-shelf entry. Planning and measured control feedback reject rotation inside the cabinet. This exterior transfer skips fixed-orientation projection; all collision and payload checks remain active. Other transfers retain projection. The exterior transfer densifies raw RRT waypoints to at most0.025rad before upstream cubic-spline TOPP, reducing observed spline corner-cutting without removing the exact postcheck or changing RRT settings. Grasp approach/lift, empty-hand release clearance and support descent use Cartesian waypoints at most 5 mm apart, solved with existing SciPy bounded least squares and retimed with MPLib TOPP. The pose residual is translation in meters plus rotation-vector radians; controlled-joint bounds are explicit, finite differences use relative step1e-4 for float32 FK, and each solve is capped at200evaluations. FK acceptance remains1mm/.1degree; existing exact-shape joint/angle/path checks and the2mm descent corridor remain active. The bounded method uses actual geometry checks; RRT's optional1mm planning inflation applies only to RRT. Failed bounded paths fail explicitly without switching controllers.

During an explicitly marked open-gripper grasp approach, unconfirmed bilateral finger contact is expected and stops approach before closure. An unexpected confirmed payload still fails. Four consecutive closed-gripper physical controls remain necessary before binding or lifting.

For fixed-orientation transfers, candidate RRT TCP positions are projected using existing SAPIEN IK and retimed with existing TOPP, then checked again for collisions, joint limits and at most 0.5 degree orientation error. Actual orientation is checked after controls. Projection or tracking failure stops explicitly. Finite samples do not certify continuous-motion safety. Other controller variants keep their existing behavior.

Each macro is capped at 2000 controls, each planned segment at `motion_plan_max_steps` (default 400), and support seeking at 1000 controls. Descent segments are at most 5 mm, shortened near support using live collision-shape clearance. Only shallow payload contact with the specifically matched shelf top may penetrate by at most 1 mm, with the correct direction and height. Robot/cabinet, side-wall, ceiling and ground collision checks remain active.

Collection retains configured physical budgets and resets between calls. Existing 800-control episodes and 1000-control chunks can truncate longer macros; new training budgets need an explicit experiment decision. Completed failed macros remain WM endpoints; budget-truncated macros are excluded. Phase, status, planner attempts and failure details are logged.

## Physical readiness

The version 4 focused motion/attachment/pick-place/binding/RAPS tests passed 107 cases. **Under that previous TCP-XYZ placement contract, both cup and bowl completed actual planned grasp/carry/lower-place/release/home examples**, with 20 additional open-gripper home controls confirming shelf support, removed binding and stable object position. Cup used381grasp+978place+20postcheckcontrols; bowl334+964+20. Maximum postcheck home error was approximately0.10mm. Videos, source transitions, summaries and inspected contact sheets are in artifacts/complete_pick_place_20261113/index.html. These are two selected successful examples in one fixed layout after development failures, not an unbiased success-rate estimate or certification of the entire XYZ bounds. MPLib RRT remains unseeded upstream. Earlier failed probes are preserved. These recordings do not validate the version 5 object-XY targets or its new computed entry route. No formal training was launched for the version 5 change.

The separate native binding probe in `artifacts/grasp_binding_20261113/` used 309 actual CPU control ticks across cup and bowl, including a reset check. Both objects bound after real grasp confirmation, remained held during lift/retraction/holding, and recovered free falling after opening removed the drive. Maximum measured relative position error was 0.022 mm. This probe reused direct-EE calibration to isolate binding; it does **not** establish complete planned grasp/place route acceptance or general retention success rates.


## Lower-entry preplace evidence

`artifacts/preplace_20261113/` records the bounded position/route investigation. With a real production bowl grasp, `[0.24, 0.15, 0.60]` completed both fixed-orientation joint-planned motion from the upper intermediate and subsequent motion to descent-entry `[0.45, 0.15, 0.53]`, keeping the physical binding and all collision checks. This is a usable candidate for the tested layout, not a globally optimal pose or proof for all place parameters/objects. Several lower or laterally shifted outside candidates failed limits, payload/frame clearance or orientation projection. Complete placement/release/home must be assessed separately from these successful transport segments.


With the version 2 production macro, the bowl completed preplace (61 controls), lower entry (140 controls), then descended to TCP Z approximately 0.460 m. The next 5 mm step had no accepted IK among 16 starts: the current branch exceeded joint 7's upper limit (2.9069 rad versus 2.8973 rad), while other valid-limit alternatives hit the door. The place macro stopped closed without releasing. This demonstrates improved entry, not successful final support/release/home. A preplace **joint posture** compatible with the remaining descent is still required; XYZ feasibility alone is insufficient. See `artifacts/preplace_20261113/production/` for the actual 818-control run. Cup transport was not validated in that version2 run; the current version4 examples supersede that readiness status.


### Correction to the initial angular binding implementation

Inspection of SAPIEN3.0.3's pinned `src/physx/joint_component.cpp` showed that cone-limit setters do not change the swing-axis motion state from FREE. The earlier zero-cone call therefore did not establish a hard angular lock; low-load short probes did not expose this. Version3 uses explicit angular drive rather than claiming six hard locks. Position and angular deviations remain measured and rejected at2mm/2degrees. No safety threshold was relaxed.


## Full exploration pipeline

The fresh configs/raps_gnn_actor_cabinet_pick_place.yaml preset selects the existing RAPS actor with the CEE-US state GNN WM. The schema is 5D throughout random bootstrap, learned imagination, completed-macro replay, fitting, evaluation and checkpoint restoration. Three-container comparison uses scripts/launch_raps_comparison.py with that config; only policy/object heuristic differs across curiosity+heuristic, curiosity-only and random.

The approved new preset uses2000bootstrap/2000per-round/2000evaluation controls and2200-control episodes, preserving398rounds and existing network/training hyperparameters. Total800000controls per arm includes evaluation. Both low-level and EE/native-wrapper limits are2200; legacy configurations keep their existing limits. Collection still resets at each call; longer budgets reduce but do not eliminate partial macro censoring. A completed failed macro remains a valid endpoint; an unfinished one does not.

CPU physical planning/binding and CUDA WM/actor training remain sequential under the existing protocol. Neither random action selection nor imagined rewards add primitive masks, automatic alternation or task bonuses. The original approach-progress heuristic is unchanged and now evaluates macro endpoints; its effect on learning successful place actions requires a new experiment. Random also fits a WM/actor under the same training settings while using random actions for collection.

Old split7/iCEM models and validation datasets have incompatible actions and cannot migrate. Fresh macro round checkpoints retain strict saved-config/action-contract checks and round-boundary resume with physical reset. Native RRT has no seed setter, so independent seeded containers need not share identical bootstrap transitions. Current evidence is bounded wiring/update/resume validation, not a formal baseline training result.

Current version5 integration evidence is in artifacts/pick_place_pipeline_20261113/summary.json: all three arms pass256-control smoke plus32-control train-resume checks,288actualcontrols each, on the same macro-contract hash. CUDA WM/actor updates, exact prediction/actor RNG restoration,5D replay/inactive-place-Z handling, partial-macro exclusion and strict resume are checked. A zero-control native audit verifies the2200TimeLimit. These short checks do not establish a complete physical version5 cycle or exploration performance; the full comparison has not been launched.

## Optional grasp preflight before physical collection

`configs/raps_gnn_actor_cabinet_pick_place_preflight.yaml` enables
`raps_actor.grasp_preflight: true` and `grasp_preflight_max_candidates: 32`.
The original preset and action contracts remain unchanged when this option is absent.
This is a new collection protocol and requires a fresh experiment; strict resume
must not mix filtered and unfiltered settings.

For each proposal, collection checks grasp/place holding preconditions. An empty-hand
grasp then dry-runs the open-finger retreat, exterior alignment and Cartesian approach
against the current scene, with the existing fixed orientation, joint limits and
collision checks. Virtual joint starts chain the plans without moving or stepping
the physical robot. Plans are discarded; actual execution replans after opening and
retains all existing contact, tracking and retention checks. This checks reachability
through approach, not grasp contact quality, closure, loaded lift or successful place.
A solver rejection is not a proof of physical unreachability.

Rejected proposals are resampled from the same current actor or random policy, up to
32 candidates per selected macro. There is no random fallback for a learned policy.
Exhaustion raises a diagnostic error rather than consuming repeated no-op controls.
Every checked candidate, accepted or rejected, is recorded in `preflight.jsonl`, with
its physical-step position, stage, planner reason and check duration. Rejections do
not consume real controls and do not enter endpoint replay or WM training. Planning
cost remains part of collection wall time. Empty-hand place is rejected before physics;
place route feasibility is still checked during normal execution.

The actor's imagined rewards and continuous proposal distribution remain unchanged.
Collection therefore samples the proposal distribution conditioned on passing this
filter. This is an execution guard, not an actor feasibility model or a guarantee that
the actor will escape collapse; a collapsed actor may exhaust all 32 proposals. Apply
the same filter settings to all comparison arms, report acceptance rates and wall time,
and do not compare filtered runs as if they used the old sampling protocol.

## Grasp target proximity and narrower height

The fresh preset `configs/raps_gnn_actor_cabinet_pick_place_target.yaml` retains
both grasp and place, the original 5D action layout, and the existing planned
controller. A successful grasp keeps holding its payload for a subsequent place;
there is no additional reset after grasp success. Existing collection-call and
physical episode resets remain unchanged. No stopped experiment is restarted.

The preset sets `raps_actor.grasp_z_bounds_m: [0.692, 0.772]`. This is a common
root-frame TCP target range for both objects: 2–10 cm above the upper support at
Z=0.672 m, narrowed from the original 2–20 cm. It changes the grasp Z variable's
normalization in random collection, actor imagination and WM actions together.
X/Y bounds, place variables, lift height, withdrawal and intermediate poses remain
unchanged. Only ordered finite subsets of the original grasp Z range are accepted.
The schema and saved action contract record the actual bounds; old replay or
checkpoints must not be reinterpreted using the narrower normalization.

The preset also enables `raps_actor.grasp_target_heuristic: true`. When the object
heuristic is enabled, grasp actions replace the old start/end TCP-distance progress
reward with

```text
goal = physical root-frame XYZ decoded from the grasp action
d = min_object ||goal - current_object_center||_2
heuristic_grasp = approach_weight * exp(-d / distance_scale_m)
reward = disagreement_weight * intrinsic_disagreement + heuristic
```

The preset retains `approach_weight: 50.0` and `distance_scale_m: 0.1` (meters).
Objects with a positive configured `object_weights` value are eligible; for this
grasp score those values select objects rather than scaling or averaging scores.
Both current objects are eligible. Taking the nearest eligible object avoids
rewarding an empty midpoint merely for being between two objects. This is center
proximity, not distance to the object's surface or certification of a valid grasp.
**Place has zero heuristic in this mode; its curiosity term remains active.**
With `grasp_target_heuristic` absent or false, existing configs retain the original
endpoint-progress heuristic for both actions. Target-reward contract version 2
rejects checkpoints from version 1, which still gave place endpoint-progress reward.

In imagination, object centers come from the current imagined state and grasp XYZ
comes directly from the sampled action. The grasp score therefore does not depend
on that transition's predicted endpoint or on returning to the exterior intermediate
pose. `training.jsonl`'s actor `heuristic` metric reports the active score. Endpoint
WM targets still contain completed physical macro endpoints; intermediate waypoints
are not substituted into training targets. Existing configs without these opt-ins
retain the original bounds and reward definitions.

Set `object_heuristic: null` to disable both heuristic terms, as the comparison
launcher does for curiosity-only and random. Heuristic-only retains the selected
heuristic with `disagreement_weight: 0.0`. All arms share the same dual-action,
height and preflight settings. Strict saved-config/action-contract checks require a
fresh run for this reward/normalization change. Preflight still checks unloaded
reachability through approach, not successful grasp/lift/place. An unfiltered imagined
policy can still exhaust the preflight proposal limit. No native rollout or training
has yet established improved success or coverage for this preset.

## Optional grasp-only exploration

`configs/raps_gnn_actor_cabinet_grasp_only.yaml` adds `raps_actor.grasp_only: true`
to the target-reward preset. Both this preset and its fast variant also set
`cabinet_door: false`: the door panel, handle, hinge and their collision/visual
geometry are not built. The cabinet frame, shelves, robot and object poses remain
unchanged. Historical presets and absent settings retain `cabinet_door: true`.
The original door angle/velocity state slots are reserved as zeros to retain the
67D latched state layout; the state contract explicitly marks the absent door,
so old door-equipped checkpoints cannot be loaded into this scene. Saved run and
environment configurations record the option. A fresh run is required; removing
the door does not resolve frame/shelf collisions or certify grasp coverage.

It retains the same upper-shelf grasp controller, shared
Z range 0.692–0.772 m and target-proximity heuristic, but removes place from random
collection, the actor, imagination and WM action schemas. The action is 4D: one
constant grasp indicator and three normalized XYZ variables. Existing dual-action
presets retain their 5D action schema and lifecycle. A fresh actor/WM/replay and output
directory are required; strict saved-config/action contracts reject mixing modes.

After a complete successful grasp, including lift/withdrawal/exterior intermediate
return, collection records the measured endpoint and then resets the scene before
the next attempt. It does not reset merely on first finger contact. Completed failed
macros reset before the next attempt if the endpoint contact snapshot still reports
an object grasped; failures without a held object continue in the current scene.
`reset_after_skill: grasp_success` or `grasp_failure_still_holding` records the reason.
Failure status, measured endpoints and trainability are retained; resets generate no
control ticks or WM transitions, and environment termination flags retain their
original meaning. All budget, episode-limit and normal collection-call resets still
apply. Grasp-only action contract version 2 records this lifecycle change and rejects
old checkpoints. Existing frozen training runs retain the old protocol; the new
protocol requires a fresh run and must not be silently mixed with old results.

The existing comparison launcher accepts this preset for the same three arms or
four with `--include-heuristic-only`. All arms share the grasp-only action/reset/Z/
preflight protocol and training budgets; only their exploration policy/reward settings
differ. Native CPU planning/physics and CUDA WM/actor remain sequential. The preset
does not enable runtime optimizations or change the full-history WM training budget.


## Shared discrete 3 by 4 shelf grid

The new `configs/raps_gnn_actor_cabinet_pick_place_grid.yaml` enables both grasp
and place with `raps_actor.pick_place_grid: true` in a doorless scene. The separate
`raps_gnn_actor_cabinet_grasp_grid.yaml` and `_fast.yaml` retain grasp-only success
resets. Older presets retain their continuous XY actions and original placements.
These are fresh-run configurations, not settings to apply to existing checkpoints.

`domain/robocasa/cabinet_grid.py` defines one authoritative row-major XY table in
robot-root meters: X rows `[0.395, 0.515, 0.635]` run from front to back, and Y columns
`[-0.060, 0.060, 0.180, 0.300]` increase along root Y. Cell IDs are zero-based
`4 * row + column`. Both shelves use these same 12 XY coordinates, with upper
support Z 0.672 m and lower support Z 0.330 m. These are candidate locations,
not a certified reachable set.

The cup initially occupies cell 4 (row 2, column 1, XY `[0.515, -0.060]`), and the
bowl occupies cell 5 (row 2, column 2, XY `[0.515, 0.060]`). Each reset restores
these fixed locations without random assignment or jitter. Initial object XY means
the native collision-AABB center; exact shape support accounts for the cup handle's
nonzero center offset. Original spawn Z, object orientation, geometry and physics
remain unchanged. Nonzero `object_shift_xy` is rejected for grid scenes so the
candidate and initial-position tables cannot silently disagree.

All three grid presets now enable `raps_actor.grasp_orientation_candidates: true`.
The policy selects `grasp(cell_id, z, orientation_id)`: the TCP XY remains the exact
cell center, and changing orientation does not change XYZ. The five zero-based
orientation IDs are:

| ID | Root yaw (degrees) | Root pitch (degrees) |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | -15 | 0 |
| 2 | +15 | 0 |
| 3 | -45 | +30 |
| 4 | +45 | +30 |

`domain/robocasa/grasp_orientations.py` defines the authoritative candidates.
Rotations are `Rz(root yaw) @ Ry(root pitch) @ R_reset`, using the measured
episode-reset TCP orientation and wxyz quaternions. These are proposed grasp
orientations, not certified solutions for every cell, height or object.

The existing reference conditional actor selects a joint categorical branch for
cell and orientation: 60 grasp branches ordered by cell, then orientation ID.
Place appends 12 cell branches with no orientation argument. One shared continuous
normalized argument controls grasp TCP Z (0.692–0.772 m in these presets); its
unused place slot is zero for the WM. The action vector is therefore 61D for
grasp-only or 73D for grasp/place. This reuses the existing conditional actor and
REINFORCE objective without adding an actor head. Physical grasp decoding adds
`orientation_id` to the unchanged XY/Z coordinates; place remains `place(x,y)`.

With the option absent or false, the earlier grid contract remains 13D/25D with
`grasp_00` through `grasp_11` followed by the optional place branches. The option
requires grid pick/place and planner motion execution. Changing it requires fresh
models, replay and output directories; strict contracts reject old checkpoints.

With candidates enabled, each macro retreats while holding its measured starting
orientation. Grasp alignment may rotate only outside root X<0.29 m, restoring the
selected orientation at the alignment endpoint before entry. Entry, closing,
lift and withdrawal retain that orientation. Preflight uses the same selected ID,
exterior-rotation audit and fixed-orientation entry as execution; execution still
replans because opening/settling can change the actual start. Planning failure
does not substitute a different orientation. The existing proposal filter may
request a new complete action from the policy, and records rejected actions.

For place, exterior preplace transfer restores the reset orientation; measured
payload bounds are recomputed for that placement orientation. Place still computes
TCP Z from payload geometry and lower-shelf support. Its XY is the object
collision-AABB center, whereas grasp XY is a TCP target. Joint limits, actual-shape
collision checks, contact/retention gates, timing and planning budgets are retained.

Random collection, learned proposals, imagination rewards and WM replay use this
same grid/orientation encoding, so the WM receives the selected orientation through
the categorical action. Grasp target proximity uses the actual selected cell and
decoded Z; it remains independent of orientation. Place heuristic remains zero in
target-reward mode. Strict action and state
contracts record the grid, dimensions, initial cells and coordinate definitions and
reject continuous-XY checkpoints even when dimensions happen to coincide. Existing
preflight and physical collision/contact/lift/support gates still apply. The full
12-cell grasp/lift/place coverage of this historical layout has not been established; discretization alone
cannot remove IK, frame collision, occupied-cell or contact-quality failures.

## Compact reachable grasp samples

The separate `raps_gnn_actor_cabinet_grasp_grid_compact.yaml`,
`raps_gnn_actor_cabinet_grasp_grid_compact_fast.yaml` and
`raps_gnn_actor_cabinet_pick_place_grid_compact.yaml` select
`raps_actor.grid_layout: compact_3x4`. The historical grid remains the default
for grid-action presets that omit this setting.
Compact row-major centers use X `[.375, .405, .435]` and Y `[0, .08, .16, .24]`
in robot-root meters. These are twelve independent single-cup sample positions;
the 3 cm row spacing does not accommodate twelve simultaneous non-overlapping cups.
The native cup starts at cell4 and bowl at cell7. Compact initial XY locates the
native actor origin (the upright cup body axis), unlike the legacy collision-AABB
center. Spawn Z, rotations, geometry and timing are unchanged.

Grasp still learns cell, one of the same five orientation IDs and continuous Z.
Native single-cup trials at Z .732 m have demonstrated complete grasp, lift,
withdrawal and intermediate return at every compact sample: ID0 at cell3, ID2
at the other eleven. This is one successful attempt per location, with the
existing constraint activated after actual confirmed contact; it does not establish
repeatability, friction-only retention, simultaneous occupancy, or every ID/Z pair.
The actor/WM schema remains 61D grasp-only or 73D grasp/place. Compact and legacy
coordinates/spawn references are recorded in distinct strict contracts. Use fresh
models, replay and output directories; no training is launched by these diagnostics.

Macro contract v6 retracts open fingers 6 cm along negative root X at fixed Z
after supported release, before the exterior withdrawal. Post-release motion
treats the cup as an obstacle for the fingers as well as other robot links.
The previous vertical clearance could disturb or tip the released handled cup;
the unexpected-attachment guard remains active. The change does not relax contact,
support, collision or retention gates. Full lower-shelf coverage is a separate
verification: upper-shelf graspability does not establish low-height loaded reach.
The default production-v6 route has completed full grasp, supported release,
outward clear, withdrawal and home at lower collision-AABB XY `[.435,.24]`
in 995 physical controls. A nearby `[.405,.18]` trial supported and released the
cup but failed clear IK; `[.42,.20]` failed approach orientation projection.
Neither is a successful complete placement. Thus the twelve upper grasp samples
are validated individually, while twelve lower destinations and arbitrary
source-to-destination combinations remain open. No training was run.
See [native coverage evidence and rerun protocol](../artifacts/cabinet_coverage_20261114/README.md).

## Default miniature cabinet placement

The base `configs/exploration.yaml` scene selects `cabinet_grid: true` and
`cabinet_grid_layout: trapezoid_4_3_2_1`. Its ten upper-shelf locations follow
rows of 4, 3, 2 and 1; the authoritative coordinates and initial assignments are
`TRAPEZOID_GRID_XY_ROOT_M` and `TRAPEZOID_GRID_INITIAL_CELLS` in
[cabinet_grid.py](../domain/robocasa/cabinet_grid.py). The original handled cup
(`cabinet_item_16`) starts at cell 4, root body XY `[.410, 0]` m. The default
`cabinet_item_layout: single_upper_second_row` omits the bowl physically.
Setting `cabinet_item_layout: two_upper_second_row` restores the original bowl
(`cabinet_item_17`) at cell 6, `[.410, .240]` m. Reset preserves these positions.

Each cell has a 75 mm depth and 120 mm lateral allocation; its lateral bounds
are centered 14.5 mm toward the cup handle from the body-origin target. Native
geometry checks confirm that both selected objects fit their cells and the shelf
without overlapping. The unchanged bowl diameter is 72 mm, leaving just 1.5 mm
per side along cell depth. This verifies static fit, not robust placement motion,
all-cell bowl graspability, or lower-shelf placement.

Legacy and compact resolvers and experiment presets are preserved. The default
scene change does not change the PPO/primitive action schema, motion planner,
robot reset or door configuration, and does not start training. The independent
ten-point grasp diagnostics used some different routes and resets; they are not
yet a single integrated ten-cell learning preset. The CEE-US/RAPS `make_env`
helper retains the two-object layout by default for historical experiments.
The explicit `cabinet_item_layout: single_upper_second_row` training option now
uses a real one-cup state and compatible model dimensions; see the current
fixed-cell training protocol below.

The current trapezoid action schema restricts orientation candidates to IDs
`[0, 1, 2]`: root yaw/pitch `(0,0)`, `(-15,0)`, `(15,0)` degrees relative to
reset. IDs3/4 are rejected by encoding and cannot be sampled or decoded from
its categorical branches. Ten cells produce 30 grasp branches plus one shared
continuous Z (31D grasp-only; 41D including ten place branches). The action
contract records the active IDs and rotations and rejects former five-pose
trapezoid contracts. Historical legacy/compact schemas and the generic five-pose
diagnostic rotation definitions are retained for reproducibility. This action
reduction does not establish physical feasibility of every cell or approach.

The user-selected learning protocol retains the earlier grasp TCP Z range
`[0.692, 0.772]` m in the robot-root frame (2–10 cm above the upper support).
The trapezoid schema uses this range by default even without an explicit
`grasp_z_bounds_m`; existing grid-learning presets already specify the same range.
Only Z is continuous, normalized as `(Z - 0.732) / 0.040`; XY is the selected
cell center and orientation is one of IDs0/1/2. The action contract records the
effective Z range. Do not count the diagnostic T9 successes at Z0.812 as coverage
of this restricted protocol: a full grasp at T9 within the retained range is
still unverified. This decision does not widen Z to include that diagnostic.

### Current fixed-cell cup-only training interface

Use `configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml` for the current protocol:
cup-only, fixedcell4 bodyXY[.410,0],10cell targets, poseIDs0/1/2, TCPZ[.692,.772].
The physical door remains present/open. The preset uses the historical centered
robot reset because the miniature default uncentered reset has an initial
panda_hand/cabinet-sideboard planning collision. This explicit reset correction
does not change cup position, geometry, timing, contact or collision tolerances.

The state encoder and GNN use one object:33agent/controller features +12cup
dynamic +5static =50D. Actor, random sampler, WM replay and imagination all use
the same31D canonical action,30one-hot(cell,pose)choices plus normalizedZ. Physical
coordinate validation uses the grid's actual XY envelope (including frontX.335)
and narrowZ instead of rejecting front targets under the historical Xmin.370;
the learned schema still permits only exactcellXY and currentposeIDs. Planning
and all physical collision checks remain active, with boundedIK approach as
before. Constraint-after-confirmation retention is preserved. The current user-approved
budget is400000controls per arm; historical presets retain their budgets. Deep RRT
diagnostics remain separate; no all-cell reachability claim.

The current preset disables `raps_actor.grasp_preflight` for all four arms at the
user's request. Each proposed action is passed directly to execution without
feasibility rejection sampling; execution still plans each motion and retains
collision, joint-limit and grasp-confirmation checks. Completed failed macros
with real control steps remain WM endpoints. Zero-control failures remain audited
but excluded from WM fitting, and the existing100consecutive-zero-duration guard
still stops an unproductive loop. No artificial control steps or failed-state
labels are inserted. This differs from the earlier preflight-enabled protocol
and requires a fresh run/checkpoint contract.

The new cup-only trapezoid protocol also invalidates CPU contact reports during
reset until the first real physics substep. PhysX retains the previous episode's
contact report after pose-only reset; interpreting it as current caused a false
already-holding/unconfirmed-grasp rejection immediately after the first successful
grasp. Context and planner queries share the validity flag, while explicit active
binding remains authoritative. No extra simulation steps or relaxed contact
thresholds are used. Historical two-object protocols retain their prior behavior.

The existing launcher selects all four comparison arms with the same preset:

```bash
python scripts/launch_raps_comparison.py \
  --config configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml \
  --include-heuristic-only \
  --numpy-shim /tmp/raps-three-arm-ls76ex80/numpy126 \
  --output logs/raps_cup_trapezoid_<new-name>
```

This command launches formal training. The current approved budget is2000bootstrap
+198rounds*2000training+2000evaluation=400000controls per arm (1600000total for
fourarms including evaluation), with two sampling processes per arm. Only
exploration reward/policy differs. Use fresh output/checkpoints/replay; the new
state/action contracts reject historical two-object/five-pose artifacts. Fixed
cell4 training supplies no reset-distribution evidence for occupied cups in the
otherninecells.

### Bounded profiling without changing the learning protocol

`python -m exploration.diagnostics.profile_cup_grasp --output <new-directory>
--replay <genuine-two-row-native-replay.npz> --device cuda --label <name>` profiles
one scripted fixedcell4 grasp (400-control cap), then 25 WM epochs and 100
actor/critic updates with the current full-size model. It saves nested collection
and learning timers, control traces and offline state bundles for exact comparison.
The two-row replay is a bounded pipeline proxy, not full-replay training timing.
See the [2026-11-15 profiling report](../artifacts/cup_profile_20261115/report.md)
for the native runtime command, measurements and exact-equivalence evidence.
The retained optimizations move fixed goal calculations outside bounded-IK
residual evaluations and directly concatenate the two residual vectors; they
preserve arithmetic, the solver and all execution checks. A tested
actor grid-cache candidate was removed after measuring a slowdown.

A historical bounded CPU multiprocessing probe is available with
`python -m exploration.diagnostics.profile_cpu_sampling --output <new-directory>`.
It compares two independent scripted jobs in one versus two persistent spawned
CPU workers, without training updates or production collector wiring. The
[2026-11-15 probe report](../artifacts/cpu_sampling_20261115/probe/report.md)
records approximately1.97x collection throughput, but one job failed strict
trajectory equality. Single-env failed-attempt continuation and independent
worker histories differ; keeping update order alone does not preserve training
data/parameters. That scene-history-confounded result is superseded by the
matched-history validation below; the original evidence remains available.

### Fixed CPU sampling streams

The current cup-only trapezoid preset selects `cpu_sampling: {workers: 2, streams: 2}`.
All four comparison arms inherit these exact settings. Configurations without
this key retain the original single-environment collector. The new protocol is
`cpu-multistream-fresh-scene-v1`; it deliberately changes the sampling distribution
from the old continuing single-env stream. Old serial cup checkpoints cannot be
resumed into it. Use fresh run/replay/checkpoints; training resume retains all
configuration except the existing rounds/evaluation extensions.

Each collection call snapshots the learner's model, normalizers and actor. Two
logical streams each create a fresh CPU scene and use a seed from
`SeedSequence([config_seed, collection_seed, stream_id]) modulo 2**31`. RNG is
reseeded after model/scene initialization, and the parent learner RNG is preserved.
Within each stream, failed attempts continue from their actual state and existing
success/termination resets remain. All action, Z, physics, contact, IK, collision
and retention rules are unchanged. Proposals run on CPU; learner device is unchanged.

The exact total real-control budget is split by quotient/remainder in stream-ID
order (the default2000controls gives1000perstream). Each stream ends at its quota;
a macro cut by that boundary remains budget-truncated and is excluded from WM
endpoints as before. There is no padding, resampling preflight or per-worker fit.
After all streams finish, results merge by stream ID and local order. Episode,
skill and step identifiers are globally reindexed to keep replay windows from
crossing streams. Raw shard logs are retained under `streams/<stage>/stream_<id>`
alongside canonical merged JSONLs. Only then does the existing WM fit run, followed
by actor/critic updates and checkpoint publication. Worker failures stop the round
without a partial canonical merge or a serial fallback.

The manifest/checkpoint stores the sampling contract. A round-boundary resume
starts fresh stream scenes, so it does not require restoring hidden PhysX solver
state. Keep `streams: 2` when testing `workers: 1` versus `workers: 2`; otherwise
the scientific sampling protocol changes with worker count. Current resume
validation conservatively also keeps worker count fixed. Two processes per arm
means eight sampling workers if all four baseline arms run concurrently; shared
CPU/GPU contention can reduce observed speedup.

[Native acceptance report](../artifacts/cpu_sampling_20261115/integration/report.md)
records exact matched-history trajectories, task outcomes, genuine replay fitting,
checkpoint restore and limited real actor/random probes. Reproduce only the
bounded interface check with `python -m exploration.diagnostics.validate_cpu_sampling
--output <new-directory>` in the documented native runtime. This does not launch
formal baseline training or establish learned-policy success rates.

### Uniform grasp target enumeration for held-out WM data

`exploration/diagnostics/collect_uniform_grasps.py` enumerates the current ten
trapezoid target cells, poseIDs0/1/2 and evenly spaced TCP Z values spanning the
schema's existing bounds. Five Z levels produce150cases: .692,.712,.732,.752,.772m.
The cup stays at its current fixedcell4 reset position. Each case has a fresh
scene and deterministic independent seed; only the grasp target varies.

The default `plan` mode writes the full case list/configuration without physics.
Explicit `collect` runs one attempt per selected case, retaining zero-step
rejections, completed failures, successes and budget truncations. There is no
preflight, rejection resampling or fitting. Each attempt has a400-real-control
cap. Full-grid coverage is150attempts with a60000-control upper bound; initial
bounded smoke cases do not establish that full coverage has been collected.

All attempts remain audited; only completed positive-duration macro endpoints
enter `wm_replay.npz`, including completed failures. Report eligibility per grid
combination because the usable subset may be unbalanced. Keep this dataset
independent of training replay and use the same frozen dataset for every WM.
Do not interpret single-control-step records as completed-macro WM targets, or
claim state-space coverage from these fixed-reset cases. World-model evaluation
must handle the current one-cup state schema and moving-versus-static baselines.

Prediction evaluation in `exploration/cee_us/cee_us_metrics.py` uses the actual
object count from the replay state layout (agent fields +12 dynamic +5 static
fields per object), checked against the backend object count. Position, Euler
rotation, moving-object baselines and disagreement/error indices cover the
actual objects; controller memory and static geometry remain excluded from
physical errors. Both current50D one-cup and historical two-object states are
supported. Fresh independent reset cases support only horizon1 evaluation;
longer windows require consecutive endpoints within one episode. This metric
fix does not compute an accuracy result or change training.


### Current cup single-step imagination protocol

The current cup trapezoid preset uses `imagination_horizon: 1` and `discount: 0.0`.
One step is a completed grasp macro, not one control tick. Actor/critic targets
contain only the current intrinsic-plus-heuristic reward; future critic values do
not bootstrap across a pre-reset endpoint. The physical WM continues predicting
measured macro endpoints before reset. Held failures reset under grasp-only
contract version2. Reward definitions, 31D actions, 50D states, narrow TCP Z,
cell4 reset, two CPU streams, learning update counts and400000-control budget per
arm remain unchanged. All four arms inherit the same imagination settings.
This is an immediate-reward research variant, not learned multi-step continuation;
it does not itself penalize infeasible grasp targets. Historical presets and frozen
old runs retain their previous horizon/discount. Use fresh models/replay/output;
strict resume checks reject changing these settings on an old checkpoint.
