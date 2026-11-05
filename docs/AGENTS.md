# Project agent working rules

These rules apply to the entire project alongside global instructions. They help agents in different terminals and future maintainers understand change rationale, current behavior, and validation status.

## Before starting work

- Read this file and recent entries in `docs/LOG.md` relative to the project root. Search relevant history by path or keyword; reading the entire log is unnecessary.
- Read the root `README.md` when first learning the project. Update it when project goals, module responsibilities, or main entry points change; keep individual change history in `docs/LOG.md`.
- Inspect `git status --short` to distinguish existing changes from your own. Do not overwrite, revert, or claim someone else's work.
- Use the log for handoffs; current code and actual validation results determine behavior. Explicitly record discrepancies.

## Implementation and validation

- Prefer the smallest complete, local change. Reuse existing code and dependencies; avoid unrelated refactoring.
- Follow the directory responsibilities in README: place exploration implementations and tests in their corresponding groups, dependency lists in `requirements/`, and temporary outputs in existing artifact directories. Do not accumulate modules or scratch scripts in the root.
- Preserve research baselines, upstream reference implementations, and dependencies needed for reproduction. Do not silently change algorithm definitions, coordinate systems, units, time steps, normalization, data splits, metrics, random seeds, or experiment protocols. Investigate ambiguity and ask the user when necessary.
- Run the smallest sufficient validation. Do not launch expensive training, large simulations, sweeps, large downloads, or hardware actions without explicit authorization.
- Write project-owned documentation, comments, docstrings, user-facing messages, plans, and log entries in English. Preserve identifiers, commands, paths, and technical meaning. Do not translate upstream reference code or generated patches merely for language consistency.

## Automatic change logging: task completion requirement

- After modifying functionality, code, configuration, dependencies, tests, scripts, or project documentation, proactively append to the root-relative `docs/LOG.md` before finishing, handing off, or committing. Do not wait for a reminder; complete the log in the same task.
- Write one entry per logical task describing the final changes, rather than each tool call or save. For long tasks, record completed work and pending items at handoffs; append later progress separately.
- Read-only explanations and reviews require no entry. Appending a log entry does not recursively require another entry.
- Write in English. Include date and timezone, author, goal and rationale, key changes and paths, actual validation commands and results, and unverified work or next steps. Explain scientifically meaningful behavior changes and their basis.
- Use actual `Europe/London` time and a known agent or tool name. Do not invent identity, results, or history. Explain why tests were not run; implementation is not validation.
- Append rather than overwriting or rewriting other authors' history. Append corrections to historical conclusions. Do not attribute existing work of unknown origin to the current task.
- For multiple terminals, reread the end of the log immediately before appending. Append directly instead of replacing the file from an old copy; preserve both sides of any conflict.
- Do not log secrets, credentials, sensitive data, or large raw outputs. Include concise summaries and relevant file or artifact paths.
- In the final response, summarize changes, validation, and the update to `docs/LOG.md`.

## Linking logs to Git commits

- Do not commit automatically. Commit only when explicitly requested, and include only authorized changes rather than unrelated work.
- After changing PointWorld, RAPS, or ManiSkill submodules, run `python scripts/snapshot_upstream_patches.py` to save modifications and upstream versions in the main repository's `patches/`. Before committing, run its `--check` mode and include the corresponding patches and manifest in the authorized commit. Never commit only a submodule pointer while omitting local modifications. If snapshots include someone else's work, check the authorized scope rather than including it incidentally. Do not create upstream commits solely to export snapshots.
- Give each new log entry a unique ID, such as `20261105-121012-git-log`, and mark its Git status as `Uncommitted`. Use distinct task suffixes for identical timestamps. Do not invent IDs for old entries.
- Before committing, add a separate `Log-Entry: <ID>` line to the commit message for each log entry actually represented in the staged changes. Commit the associated log as well. Do not mark excluded changes as committed.
- After a successful commit, obtain the full hash with `git rev-parse HEAD`, verify its correspondence to the logged changes, and append a mapping containing time, log ID, and full hash to `docs/LOG.md`. A read-only lookup is `git log --all --fixed-strings --grep='Log-Entry: <ID>' --format='%H %s'`.
- The mapping is generated after the commit and remains an uncommitted log change until the next authorized commit. Do not create an extra commit or amend automatically to store a hash, or claim that the mapping was included in the original commit.
- Manual terminal or IDE commits do not automatically trigger these steps. An agent may later inspect actual history and append verified mappings; do not guess historical mappings from the current HEAD.

## Temporary execution plans

- If a task needs a persistent temporary plan, place it in `docs/plans/`. Create a plan file only when warranted.
- After completing the task and applicable validation, first append key decisions, implementation results, validation evidence, and necessary handoff details to `docs/LOG.md`, then delete the completed temporary plan without asking again. Record its path in the log.
- Keep plans that are unfinished, blocked, or in use by another terminal, and record their status at handoff. Do not delete a file solely because its name contains "plan" or it is old.
- Research designs, algorithm definitions, experimental protocols, and reproduction evidence are long-lived documentation rather than temporary plans. Move information that still needs preservation into appropriate long-lived documentation and update references before deletion.
- Do not clean up tool-internal plans or files outside the project.

## Scope of automation

Agents execute these rules; this is not a file watcher. Agents that follow this file should maintain the log proactively. Manual edits and tools that do not read `AGENTS.md` do not automatically create log entries. Configure other tools to use this file as their project instruction entry point.
