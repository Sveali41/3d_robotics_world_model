# Project Change Log

Records the goal, key changes, validation, and handoff information for each task. Append entries in chronological order at the end of the file. See `AGENTS.md` for maintenance rules. This file is for development handoffs and is separate from program runtime logs in directories such as `logs/`.

## Entry template

```markdown
## YYYY-MM-DD HH:mm:ss Europe/London — Brief task title

- Log ID: YYYYMMDD-HHmmss-task
- Git status: Uncommitted (after a successful commit, append the log ID and full commit hash association separately)
- Executor: agent or tool name
- Goal and reason: Why the change was made and what it is intended to solve.
- Key changes: Files involved, behavior changes, and the basis for important decisions.
- Validation: Commands actually run and their results; if none were run, state why.
- Unverified and follow-up: Limitations, blockers, or next steps; if none, write “None”.
```

## 2026-11-05 Europe/London — Establish agent work rules and change log

- Executor: Codex
- Goal and reason: Help agents in different terminals and future maintainers understand the intent, contents, and validation status of each change, and have agents record changes after editing.
- Key changes: Added root `AGENTS.md`, requiring agents to read history at task start, append a log after editing, and follow multi-terminal collaboration rules; added `LOG.md` as the shared handoff log and format template. Automatic recording is performed by agents that follow these rules.
- Validation: Checked the contents and cross-references of both files, patch whitespace formatting, and confirmed that the files were not ignored by Git. This task only added documentation; no code tests were run.
- Unverified and follow-up: Did not verify whether agents in other terminals or other tools read these rules. The workspace already had uncommitted changes during initialization; they were left as-is, and their implementation history was not retrospectively recorded.

## 2026-11-05 11:52:55 Europe/London — Documentation archiving and temporary plan cleanup rules

- Executor: Codex
- Goal and reason: Consolidate agent rules and the log in the existing `docs/` directory and automatically clean up temporary execution plans after tasks are complete.
- Key changes: Moved the rules into `docs/AGENTS.md` and the log into `docs/LOG.md`, preserving existing history; reduced root `AGENTS.md` to a project instruction entry point. Updated the required log path and added a rule to automatically delete a completed temporary plan after the task is finished, validated, and necessary information is archived.
- Validation: Used Python to check the entry point, document paths, preservation of the original log history, and removal of the old root log; checked patch whitespace formatting in all three documents. This task only modified documentation; no code tests were run.
- Unverified and follow-up: Existing `docs/ROBOCASA_RAPS_NEXT_EXECUTION_PLAN.md` and `docs/ROBOCASA_RAPS_POINTWORLD_PLAN.md` contain unfinished items and research design and were not deleted; there was no temporary plan that could be confirmed complete in its entirety. Whether other tools read project rules still depends on their configuration.

## 2026-11-05 11:55:08 Europe/London — Streamline docs as requested by the user

- Executor: Codex
- Goal and reason: The user explicitly requested that only the agent rules and change log be retained, and all other documents be deleted.
- Key changes: Deleted `ARCHITECTURE.md`, `CABINET_EXPLORATION_DIAGNOSTICS.md`, `CARTESIAN_EXPLORATION.md`, `CEE_US_CABINET.md`, `HOUSEHOLD_PARTNR.md`, `macro_mini_kitchen.md`, `POINTWORLD_CABINET_OFFLINE.md`, `ROBOCASA_RAPS_NEXT_EXECUTION_PLAN.md`, `ROBOCASA_RAPS_POINTWORLD_PLAN.md`, and `THOR_MINI_ENV.md` from `docs/`; retained `docs/AGENTS.md`, `docs/LOG.md`, and the root instruction entry point. Adjusted the missing-dependency message in `domain/household/partnr.py` to remove references to deleted documentation and continue pointing to the inspection command and missing-item report.
- Validation: Checked that only the two retained documents remained in `docs/`, that the instruction entry point worked, parsed the modified file with `ast.parse`, checked for references to deleted documents in project-owned code, and checked this patch. Only documentation was deleted and prompt text was adjusted; no simulation or training was run.
- Unverified and follow-up: In accordance with the explicit deletion request, old documents and research plans were not archived; run instructions and old plans are no longer kept in `docs/`.

## 2026-11-05 12:10:12 Europe/London — Associate the change log with a Git commit

- Log ID: 20261105-121012-git-log
- Git status: Uncommitted
- Executor: Codex
- Goal and reason: As requested by the user, record the corresponding Git commit hash when an actual commit is made, while continuing to require explicit authorization before committing.
- Key changes: Specified in `docs/AGENTS.md` a process for unique log IDs, a `Log-Entry` field in commit messages, and appending the actual full commit hash after a successful commit; updated the `docs/LOG.md` template. The association record created after a commit is saved in the next authorized commit, avoiding a commit hash referencing itself.
- Validation: Checked the rules and template fields, preservation of historical log entries, and document whitespace formatting; only Markdown was changed, no code tests were run, and no Git commit was created.
- Unverified and follow-up: The commit procedure has not yet been performed. A manual commit in an IDE or terminal does not automatically trigger an agent record; an agent must later verify and append the association.

## 2026-11-05 12:12:49 Europe/London — Add a project overview of extending MAC to 3D robotics

- Log ID: 20261105-121249-project-readme
- Git status: Uncommitted
- Executor: Codex
- Goal and reason: As explained by the user, position the workspace as an extension of Model-driven Adversarial Curriculum (MAC) to 3D robotics, and retain an overall code guide for maintainers and agents.
- Key changes: Added root `README.md` describing research goals, existing modules, directory structure, configuration, main entry points, and experiment conventions, distinguishing the existing baseline from a complete MAC extension; updated `docs/AGENTS.md` to require keeping the README in sync when project structure or entry points change. Continued the process of committing only when requested by the user.
- Validation: Compared the guide against CLI dispatch, argument parsing, and configuration in the current source; checked project paths in the README and patch whitespace formatting. Only Markdown was added or modified; no training, simulation, or dependency installation was started.
- Unverified and follow-up: The run commands in the README were not actually executed. The full MAC curriculum objective, adversarial mechanism, and 3D experiment protocol still require the original method and subsequent research design to determine.

## 2026-11-05 12:18:39 Europe/London — Organize code, tests, and dependency lists by responsibility

- Log ID: 20261105-121839-organize-workspace
- Git status: Uncommitted
- Executor: Codex
- Goal and reason: Reduce directory clutter caused by flat source layout and generated artifacts while preserving current research implementations and existing workspace changes.
- Key changes: Moved exploration modules into `exploration/cee_us/`, `raps/`, `pointworld/`, `diagnostics/`, and `visualization/`, and tests into corresponding groups and `tests/environments/`; moved 43 code, test, and dependency-list files in total, and updated imports, CLI documentation, and dependency installation script paths. Shared rewards, PPO, action distributions, collection, and trajectory recording remain in the exploration root; retained the serialization module path for `PrimitiveRecord`. Moved dependency lists from `configs/` to `requirements/`. Updated the README, maintenance rules, and `.gitignore`; the local `.vscode/settings.json` hides cache, runtime, data, and artifact directories (this setting is Git-ignored and is not distributed with commits). Data, weights, and upstream repositories were not moved.
- Validation: Compared each Python AST against the pre-organization workspace contents, confirming that only module references, cross-test references, and paths affected by the root-depth changes differed; ran `bash -n` on three installation scripts. In `rlPractice`, ran `python -m pytest -q tests/cee_us tests/raps tests/diagnostics tests/pointworld`: initially 141 passed, 12 skipped, and 2 failed due to old cross-test imports; after fixing references, reran only the failures with `python -m pytest -q tests/cee_us/test_cee_us_runtime.py -k batched_logs`, and both passed, for 143 total passed and 12 skipped. Collection across the project succeeded except for THOR room tests, collecting 258 tests; `--help` for the three migrated CLIs, `main.py --cfg job`, visualization module imports, and four project-root location checks passed.
- Unverified and follow-up: `rlPractice` lacks `shapely`, so `tests/environments/test_thor_rooms.py` could not be collected; no additional dependencies were installed. Other environment tests, full GPU regression, training, and physical simulation were not run. External old scripts need to use the new module paths from the README; source fingerprint checks on existing diagnostic outputs remain active. No Git commit was created.

## 2026-11-05 12:25:02 Europe/London — Move upstream RAPS and PointWorld into their corresponding modules

- Log ID: 20261105-122502-relocate-upstream
- Git status: Uncommitted; `git mv` has staged the two submodule migrations and `.gitmodules` path update; code and log changes are uncommitted.
- Executor: Codex
- Goal and reason: As requested by the user, move upstream code from the root directory into the corresponding RAPS and PointWorld directories.
- Key changes: Moved `raps_official/` to `exploration/raps/official/` and `PointWorld/` to `exploration/pointworld/official/`. Updated `.gitmodules`, the RAPS reference component loading path, PointWorld loading/collection paths, preprocessing test imports, the model path in `configs/main.yaml`, and the README. Preserved submodule names, URLs, commit versions, and existing modifications; did not create compatibility copies at the old paths.
- Validation: In `rlPractice`, ran `python -m pytest -q tests/raps/test_raps_gnn_actor.py -k 'imagination_updates or checkpoint_restores'`: 2 passed; ran `python -m pytest -q tests/pointworld`: 11 passed. Both submodule Git states were readable; the RAPS commit was `1c56b0fe367cc32d3ee548f407ac3a9af9e3ce2e`, and the PointWorld commit was `d8a46a80b5f4c9f0e0c6ee4e17450039f8e8a6a2`. Compared the complete PointWorld Git patch before and after moving it; all three original modifications were identical. Checked new paths, module syntax, and final differences.
- Unverified and follow-up: Training, model weight loading, and physical collection were not run; external scripts with hard-coded old absolute paths need updating. No Git commit was created.

## 2026-11-05 12:27:19 Europe/London — Move CEE-US upstream source into its corresponding module

- Log ID: 20261105-122719-relocate-cee-us
- Git status: Uncommitted
- Executor: Codex
- Goal and reason: As requested by the user, standardize the locations of upstream CEE-US, RAPS, and PointWorld code.
- Key changes: Moved `third_party/cee_us/` to `exploration/cee_us/official/` and updated import namespaces in upstream source, adapters, and tests; updated the namespace in `compatibility.patch` and the test path in the upstream README, preserving the version, LICENSE, and `upstream_manifest.json`. Updated the project README.
- Validation: In `rlPractice`, ran `python -m pytest -q tests/cee_us/test_cee_us_backend.py tests/cee_us/test_cee_us_heuristic.py tests/cee_us/test_cee_us_cuda_graph.py`: 31 passed and 7 skipped; compared Python ASTs file by file before and after the move, which were identical except for import namespaces, and confirmed the version manifest was unchanged.
- Unverified and follow-up: Training and physical simulation were not run. An additional check found that the pre-existing `compatibility.patch` fails to apply in reverse at one location in `gnn_modules.py`; the same failure was reproduced against the original content before the move, so this is a pre-existing gap in the reproduction record. This task only synchronized namespaces and did not change model mathematics or rebuild the patch. No Git commit was created.

## 2026-11-05 12:39:09 Europe/London — Track local upstream changes as patches in the main repository

- Log ID: 20261105-123909-track-upstream-patches
- Git status: Uncommitted; the new patches, version manifest, and snapshot script have been added to the main repository's staging area.
- Executor: Codex
- Goal and reason: The user does not want to create forks for now, but requires upstream modifications to be saved in the main project's Git history so they do not exist only in submodule workspaces.
- Key changes: Retained the submodule structure and added `scripts/snapshot_upstream_patches.py` to export all changes relative to HEAD and check consistency with `--check`; the main repository tracks `patches/pointworld.patch`, `maniskill.patch`, `raps.patch`, and `upstream.json`, totaling 21518 bytes. PointWorld has 3 modified files, ManiSkill has 5, and the RAPS patch is empty. The script checks that the base versions match the submodule pointers staged in the main repository and clearly requires untracked files to be staged or ignored first. Added the previously missing ManiSkill `.gitmodules` entry; updated the README restoration steps and the agent rules to refresh snapshots after edits and before commits. Did not expand and include PointWorld's approximately 106 MB of tracked files in the main repository.
- Validation: Successfully ran the export and `python scripts/snapshot_upstream_patches.py --check`; extracted the original files from the recorded upstream commits into a temporary directory, ran `git apply --check` on each patch, and applied them; all 8 modified files were restored byte for byte to match the current workspace. Checked the final differences for the new script, patches, and manifest.
- Unverified and follow-up: Runtime code was not changed; no training, simulation, or downloads were run; no commit or push was created. Snapshotting is not a background monitor, so future modifications must be snapshotted by an agent according to the rules; they become a remote backup only after the main repository is committed and pushed. Existing changes were not forcibly merged or overwritten.

## 2026-11-05 12:44:34 Europe/London — Use English throughout project-owned files

- Log ID: 20261105-124434-english-files
- Git status: Uncommitted
- Executor: Codex
- Goal and reason: The user requested English in all project files.
- Key changes: Translated `README.md`, root `AGENTS.md`, `docs/AGENTS.md`, existing `docs/LOG.md` entries, and Chinese comments in `configs/main.yaml` and `configs/kitchen_custom.yaml`. Preserved historical facts, dates, log IDs, commands, paths, results, and commit hashes. The user's explicit translation request authorized translating existing log text; future entries remain append-only. Updated agent instructions to require English for project-owned documentation, comments, messages, plans, and logs. Preserved upstream reference code and generated patches.
- Validation: Parsed both YAML files before and after translation and confirmed equal values. Compared historical technical references, dates, and commit hashes. Checked README links, added whitespace, and remaining Han characters in project-owned files; no Chinese text remained in that scope. No executable Python was changed, so code tests were not run.
- Unverified and follow-up: Upstream repositories, generated patches, runtime artifacts, and datasets were excluded from translation. No Git commit or push was performed.

## 2026-11-05 12:47:46 Europe/London — Prepare the workspace for the requested push

- Log ID: 20261105-124746-publish-workspace
- Git status: Prepared for the user-authorized commit and push
- Executor: Codex
- Goal and reason: The user requested pushing the current project to its configured remote repository.
- Key changes: Prepared the current workspace changes, including module/test organization, household and THOR interfaces, offline PointWorld comparison and preprocessing, English documentation, agent/log rules, and versioned upstream patches. Runtime data, weights, ignored local settings, and installed dependencies are excluded. The current remote main matches the local base commit `9c47ade4983edb3d2119b8aea967799f706e54d8`.
- Validation: `python scripts/snapshot_upstream_patches.py --check` passed. Final integration command `python -m pytest -q tests --ignore=tests/environments/test_thor_rooms.py` in `rlPractice` completed with 245 passed and 13 skipped. Reviewed the staged changes; no main-repository tracked file exceeds 5 MiB.
- Unverified and follow-up: THOR room tests remain excluded because `rlPractice` lacks shapely; no dependencies were installed. No large training or physical simulation was launched. The actual commit hash will be appended after commit creation; that mapping remains a local change until the next authorized commit under the established logging rule.


## 2026-11-05 13:37:44 Europe/London — Preserve local robotics changes while integrating origin/main

- Log ID: 20261105-133744-preserve-planner-git-integration
- Git status: Uncommitted; origin/main merge prepared with conflicts resolved, without a merge commit or push.
- Executor: Codex
- Goal and reason: The user requested preserving the latest local code and integrating the Git version. The local main contained RoboLab container commit `6b59383`; remote main `90bc2ce` reorganized modules, tests, and upstream source paths.
- Preservation: Saved all 43 modified or new paths, a binary diff, original Git status, file hashes, the original repository bundle, and the retained stash identity in `logs/code_merge_backup_20261105T133125Z`. All 43 paths remain present at their original or mapped canonical locations; `integrated_paths.json` records the mapping. Existing experiment outputs, checkpoints, and validation data remain untouched.
- Key changes: Combined the local split7/planner implementation and validation collector with the remote package organization. Resolved import conflicts in `exploration/raps/raps_gnn_actor.py` and `raps_replay_compare.py`; retained both joint-position execution and remote observation/sensor options in `exploration/cee_us/cee_us_cabinet.py`. Placed collection, calibration, and planner validation under `exploration/diagnostics/`, with corresponding grouped tests. Retained locally modified cabinet documentation and new planner/RoboLab documentation to honor the preservation request, updating invocation paths. Updated README entry points. Relocated the existing clean RAPS checkout to `exploration/raps/official/` without changing its commit or source; PointWorld and ManiSkill gitlink versions remain unchanged. No research algorithm, coordinate system, unit, seed, action contract, or dataset was intentionally changed by this integration.
- Validation: In `/data/data/siyao/3d_world_model.sif`, ran `python -m pytest -q tests/raps tests/diagnostics tests/environments/test_split_primitives.py tests/environments/test_planned_skill_execution.py tests/environments/test_motion_planning.py tests/environments/test_motion_attachment.py tests/environments/test_cabinet_motion_api.py tests/environments/test_skill_execution.py tests/environments/test_cabinet_grasp.py tests/environments/test_robolab_scene.py tests/environments/test_ee_delta.py tests/cee_us/test_cee_us_cabinet.py tests/cee_us/test_centered_initial_tcp_wiring.py`: 278 passed, 2 skipped. The three migrated diagnostic modules passed `--help`; `python main.py --cfg job` passed. Changed Python files parsed successfully and all preserved paths were accounted for. Validation output is saved beside the backup.
- Unverified and follow-up: No physical simulation, native MPLib planning run, data collection, formal training, dependency installation, commit, or push was performed. PointWorld and ManiSkill upstream worktrees are not initialized on this host; their recorded pointers and remote patch snapshots were retained rather than downloading dependencies. Existing diagnostic fingerprint guards remain enabled, so old output directories still require the original source version for exact resumption. Finalize the prepared merge only after explicit commit authorization.


## 2026-11-05 13:39:40 Europe/London — Remove the obsolete pre-integration code backup

- Log ID: 20261105-133940-remove-code-backup
- Git status: Uncommitted; the prepared merge remains uncommitted.
- Executor: Codex
- Goal and reason: The user explicitly requested deleting the original code backup to avoid unnecessary disk use after integration.
- Key changes: Deleted only `logs/code_merge_backup_20261105T133125Z` (677247 bytes of file content, approximately 704 KiB allocated) and the exact temporary preservation stash `56ae1d940499cbe5034b09b353aa44eb8bb7d4e0`. This supersedes the preceding entry's statement that those backup artifacts and stash remain available. Integrated source, experiment outputs, datasets, weights, and Git history were retained; no object pruning was performed.
- Validation: Confirmed the backup directory was removed; SHA-256 checks for all 43 integrated paths and the staged diff were identical before and after cleanup. No code behavior changed, so tests were not repeated. The preceding 278-pass integration result remains recorded above.
- Unverified and follow-up: No commit or push was performed; the prepared merge still awaits explicit commit authorization.


## 2026-11-05 13:42:20 Europe/London — Publish the integrated robotics code

- Log ID: 20261105-134220-push-integrated-robotics-code
- Git status: Uncommitted; prepared for the user-authorized merge commit and push.
- Executor: Codex
- Goal and reason: The user explicitly requested pushing the integrated code to the configured Git remote.
- Key changes: Finalized the pending merge of `origin/main` (`90bc2ce`) with the local planner, split7 primitive, validation collection, and RoboLab adaptations. Included the four documentation deletions already present in the working tree at the start of this task; these were existing workspace changes, not new code changes by this task. Retained `docs/AGENTS.md` and `docs/LOG.md`, existing experiment artifacts, and upstream reproduction patches. The earlier temporary code backup and stash remain deleted as requested.
- Validation: Fetched origin and confirmed no new remote divergence. Reused the recorded 278 passed / 2 skipped integration result and successful diagnostic help/config checks because executable code has not changed since that validation. `python3 scripts/snapshot_upstream_patches.py --check` could not complete because the PointWorld and ManiSkill worktrees are uninitialized on this host; independently verified that all three patch SHA-256 values, snapshot manifest bytes, and staged upstream gitlinks match `origin/main` exactly, and that the initialized RAPS checkout is clean. Checked no unresolved index entries, no tracked files over 5 MiB, and the integration diff whitespace against origin/main. No simulation, training, dependency download, or upstream modification was performed.
- Unverified and follow-up: Full native planning and simulation have not been rerun. The actual commit hash and push result will be recorded after the operation; the post-commit mapping stays uncommitted under project rules rather than triggering another automatic commit.
