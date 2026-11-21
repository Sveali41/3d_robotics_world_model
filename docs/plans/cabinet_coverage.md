# Twelve-cell grasp and loaded-place coverage

Goal: establish actual production-compatible grasp(cell, orientation ID, Z) access
at twelve upper-shelf cup locations, preferably full carry and supported placement
at twelve lower-shelf cells. Preserve prior grid/orientation experiments as evidence.

Known baseline: only front-row cells have confirmed lift evidence at Z .732 m.
Cell8 failed all five IDs at .692/.712/.732/.752/.772 m; every entry plan rejected
before physical approach. No active physical audit process at task start.

Decisions: keep native cup/cabinet geometry, root coordinates,100/20Hz timing,
contact confirmation, constraint-after-contact retention, exact collision checks,
joint limits and original planner budgets. Do not claim friction-only retention or
multi-object simultaneous coverage. Investigate planner route restrictions before
shrinking the grid. Alternative route experiments must be explicitly identified;
do not silently reinterpret historical action/replay contracts.

1. Reuse baseline evidence; bounded read-only scout maps loaded-place API.
2. At failed rear cell8, compare original bounded approach against the existing
   fixed-orientation RRT planner, first at upper sampled Z and current IDs.
3. Based on measured failures, test selected new orientations or a smaller usable
   grid; preflight first, then fresh-scene physical single-cup attempts.
4. Integrate only validated behavior with strict contracts and retained historical
   protocol, validate real grasp/lift/withdraw and loaded-place feasibility.
5. Record exact coverage, remaining gaps, source/configuration/protocol and limits.

Initial diagnostic batch: five cell8 current-ID RRT preflights at Z .772 m,
zero physics controls. Further batches follow inspected evidence. Do not relaunch
completed trials, resume the paused historical audit, or start training.

Progress / active batch:
- Existing cell8 RRT/current IDs at .772:0/5 endpoint solutions; six steeper
  yaw/pitch candidates at .772 and seven moderate candidates at .752 also reject.
- Compact single-cup candidate X(.375,.405,.435),Y(0,.08,.16,.24) is being tested
  at Z .732,ID2 through the FULL grasp macro. This is twelve distinct sampled
  centers, not twelve simultaneously occupied non-overlapping cup slots.
- Native job tool session19686 runs compact_batch.py for cells0–10; preserve
  finalized evidence and inspect interrupted folders before any continuation.
  Cell11 already succeeds in compact_rear_right_physical including fullwithdraw.
- Cell11 lower sameXY trial supported/released but fails unexpected_attachment
  during egress. Binding released correctly; cup tilts during vertical clear.
  Debugger traced permissive post-release finger/cup contact planning. Next
  explicit diagnostic: replace vertical clear with .06m outward fixed-Z clear,
  and prohibit cup contacts during post-release motion. Preserve existing guard.
- Integration, pending full coverage: preserve historical grid as default; add
  explicit compact layout/config and distinguish contracts. Existing body-center
  diagnostics must not be misrepresented as historical AABB-center spawn tests.

Later progress (2026-11-14 Europe/London):
- Upper compact FULL grasp coverage12/12 achieved: cell3ID0, allothersID2,
  Z .732m; evidence compact_cell_*/result.json plus compact_rear_right_physical
  forcell11. Original cell3ID2preflight rejection remains a failure.
- Integrated opt-in compact_3x4 and three *_grid_compact presets. Historical grid
  default remains unchanged. Compact initialcup4/bowl7 use nativeactor/bodyXY;
  action/state contracts differ from legacy. Source integration agent completed.
- Production macrov6 now uses -.06m fixedZ post-release clear and STRICT cup
  obstacle treatment during egress. Explicit prior diagnostic completes fullplace
  atcorner11. Focused macro/orientation tests65passed; gridtests22passed/2native
  skipped; tiny actual compact61D/73D WM/actorupdate/restore2passed; launcher/state
  tests17passed; compact native scene/contract1passed11.74s.
- final_batch.py sameXY lower tests: cells0–3 grasp succeeds but placeapproach
  planning fails. Cell4 interrupted when orchestration stop propagated, preserved
  resultunknown. Do NOT overwrite these directories or resume this obsoletebatch.
- loaded_bounded_routes job toolsession80468 is the current native diagnostic:
  realconfirmed cup fromupper[.435,.24],ID2,Z.732; then30virtual loaded routes
  fromlower_candidates.json using boundedCartesian approach and existingpreplace.
  No physicalplace duringqueries, exactstate/binding invariants asserted.
  physical.py CLI now records --probe-place-grid,--place-approach-method and
  --preplace-y explicitdiagnostic variants. ProductionstillusesRRTplaceapproach.
- README/docs main entrypoints updated forcompactupper protocol; artifactreport
  generator exists but --require12mustNOTrununtilfullplacedestinations finalized.

Artifacts: artifacts/cabinet_coverage_20261114/. Native runtime: existing SIF --nv,
NumPy1.26 shim /tmp/raps-three-arm-ls76ex80/numpy126, one CPU thread. User requested
iterative diagnosis and in-scope changes until coverage is achieved; no hardware.

Additional lower diagnostics (all previous processes completed):
- loaded_bounded_routes: 30 virtual candidates,2 approaches,0 descents.
- loaded_bounded_outer_preplace: preplaceY .30,19 approaches,6 descents,
  still zero physical place controls; all state/binding invariants passed.
- high_grasp_loaded_routes: source[.375,.16],Z .772 real grasp success338;
  virtual preplaceY .24 yields1 approach/1 descent of30, no place physics.
- outer_preplace_real_place: source11 fullgrasp382, preplaceY .30 rejected
  orientation_path_violation (RRT entered X>=.29 while rotating);1 failure hold.
- far_exterior_real_place: preplace[.18,.30,.56] physically succeeds, lower
  [.405,.12] bounded approach fails localIK at[.2621,.2282,.5118];462controls.
- cartesian_preplace_real_place: straight preplace[.28,.30,.56] fails exact
  payload/upper-frame5 collision near[.2531,.2189,.6356],383controls. No guard
  changes accepted. All results completed, with source snapshots.
- Current held_orientation_loaded_routes (toolsession21487) tests explicit
  measured held wrist instead of reset wrist, preplace[.18,.24,.56], bounded
  preplace/approach,30 original candidates after real source11 grasp. Different
  orientation is DIAGNOSTIC ONLY, no production protocol change yet. Inspect
  result/process before editing hash-guarded sources or restarting.
- Provisional report generated using --require-upper12; lower remains unverified.

Final current state: all listed native jobs completed. production_v6_corner_pair default route full grasp/place/home succeeds995controls. Nearby lower_region_a supports/releases but fails clear IK793controls; lower_region_b fails approach projection466controls. Upper12goal achieved; preferred lower12 remains unfinished. No native job remains active. See latest docs/LOG.md compact-coverage-results entry and artifact report for authoritative evidence and continuation decisions.

2026-11-15: user explicitly requests deeper grasp locations after rejecting tiny
front-strip spatial coverage. Existing compact grid remains unchanged while
checking deeper options, not claiming its12densecenters cover the cabinet.
- deeper_center_475_high_bounded at[.475,.16],Z.772: all5IDs preflightfail.
- deeper_center_495_high_rrt at[.495,.16],Z.772: ID2accepted;ID0endpointvalid
  butorientationprojectionfails;IDs1,3,4noacceptedendpoint. All0physics.
- Active toolsession54286: physical.py --name deeper_center_495_high_physical
  --cell deeper_center_495 --xy .495 .16 --z .772 --orientation 2
  --approach-method rrt_connect. Preserve hashes/inspect before edits/restart.
- Concurrent deeper_center_555_high_rrt no-physics probe ID0/2centerY.16,Z.772.
- Read-only deeper-route evidence review delegated to place_release_diagnosis.
  Do not add newposes or weaken collisions while currentID2route may suffice.


2026-11-15 12:52:28 Europe/London — Deeper investigation completed for this increment.
All process statements above are superseded: no native job remains active.
See docs/LOG.md entry 20261115-125228-deeper-grasp-diagnosis for protocol/results/rerun command.
Strong deeper sample [.495,.16],TCPZ.772,ID2,RRT: full friction-only
grasp376controls,no binding,+.096018m rise (deeper_center_495_high_friction).
It is20.5cm inside mouth,+6cm vs compact rear. Existing12grid unchanged.
Do not count deeper_center_555_pitch10_physical constraint success as stable
grasp: matched friction-only loses grasp; native fallback distinction documented.
Rear .455 endpoints atZ.752 fail full friction macro;Z.732 bounded IDs0/1/2
all fail shelf-collision route. Deeper12/lower12 remain unfinished.
New comparison: artifacts/cabinet_coverage_20261114/deeper-cell.png/svg,
builder plot_deeper_cell.py; no production RRT flag integrated yet.

2026-11-15 13:18:42 — Depth-only outward2/4/6cm checks for original cell10/11: all40RRT preflight queries reject at declared heights; zero physics, no active jobs or production changes. See LOG entry 20261115-131842-depth-only-forward-check and artifactREADME depth-only section. Preserve these reports; no validated replacement rear coordinates selected.

2026-11-15 13:38:22 — User changed current exploratory target to4+3+2+1full-cupcells(10), supersedingdenseV1candidate. V2static75x120mmcellsand10fullfrictionexistencegraspsachieved;seeLOG 20261115-133822-trapezoid-full-cup-ten,artifacts/cabinet_coverage_20261114/trapezoid_v2_selected.json. No nativejobsactive. Formal10celllearningconfig/unifiedresetnotintegrated;fronttwoneedexplicitexteriorresetandfrontrowdiagnosticXmin.33,otherscentered;productiongridsunchanged. Preferredlowerplacementstillunfinished. Interactive3Dviewerrequestedandbeingbuilt;no newphysicsneeded.

2026-11-15 13:40:04 — Interactive3Dtrapezoid.html complete andlocalChrometested;LOG 20261115-134004-trapezoid-interactive-viewer. HTTPservertoolsession75358 serves8766;no nativephysicsjobsactive. Trainingintegration/unifiedreset/lowerplacementremainunverified perpreviousentry.

2026-11-15 13:58:30 — User-approved trapezoid_4_3_2_1 now is default reset placement in configs/exploration.yaml (not action schema). Native cup4(.410,0), bowl6(.410,.24), unchanged objects/robot/door; static fit/reset/nooverlap and encoder contract tested (8passed). Actual reset minienv.html + minienv-scene.json exported with0controls. See LOG 20261115-135830-default-trapezoid-mini-scene. This supersedes previous "no production placement integration" status; unified ten-cell grasp routes/reset/action config and lower placement remain unverified. No native jobs active.

2026-11-15 14:10:34 — Default miniature now cup-only at trapezoidcell4; historical training helper explicitly preserves two objects. Native grid/four-arm launcher tests16passed. New cup-only ten-cell four-baseline learning is not ready: see LOG 20261115-141034-cup-only-baseline-readiness for state/action/bounds/Z/routes/reset/retention/door/distribution gaps and reproduction. No training or native diagnostic jobs active. Prior exported two-object images remain historical.

2026-11-15 14:22:03 — User retains learning TCPZ[.692,.772]m and current poseIDs0/1/2 for10-celltrapezoid. Default schema and action metadata now match; focused gridtests26passed. DiagnosticT9successZ.812 is outside authorizedlearningrange; T9fullgrasp atretainedZ remains unverified. Do not widenZ silently. Prior cup-only training integration gaps remain.

2026-11-15 14:38:11 — Supersedes prior cup-only interface-not-integrated status for fixedcell4: new configs/raps_gnn_actor_cabinet_grasp_trapezoid.yaml aligns50Dsinglecupstate/31Dactions, fourarms, exactgridphysicalvalidation andresetcontactfreshness. Userselectedfixedcell4. Centeredbaselinearmreset avoidsminiaturedefaultsideboardoverlap; doorretained. Tests126passed/1skipped plusnative2success644controls, WM/actorupdate/checkpointrestore1passed. See LOG 20261115-143811-cup-four-baseline-interface and docs/pick_place_primitives.md currentfixedcellsection. Noformaltraining/nativejobactive. Fulltenoccupiedcells/deeprestrictedZ/lowerplacement remainunverified.
