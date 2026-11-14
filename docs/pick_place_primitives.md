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
