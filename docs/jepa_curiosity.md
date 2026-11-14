# RGB-only stochastic JEPA curiosity

The `epic_jepa_cem` branch runs the current cabinet's split7 primitives through
the existing scalar CPU-physics executor. It alternates RGB collection, visual
world-model fitting and mixed categorical/continuous CEM planning. GNN,
actor-critic and PointWorld variants remain separate research baselines.

## Method and reference implementation

This is an adaptation inspired by [EpicWorldModel](https://arxiv.org/abs/2610.05996),
not an official reproduction. The authors' project page advertised "Code (soon)"
when this branch was implemented. The unchanged LeWM reference is pinned at
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` under `exploration/jepa/official/`;
its README documents source hashes, licenses, encoder construction and numerical
adaptations. The branch imports its actual SIGReg and transformer components.

The visual encoder is the reference randomly initialized Hugging Face ViT-Tiny,
patch14, 224x224, with a 192D projected CLS representation. Lossless uint8 RGB
is divided by255, ImageNet normalized and deterministically bilinearly resized
with antialiasing. There is no pretrained-weight download or implicit transfer
from another task. The original 192D latent space and upstream projector are
retained; this branch is not a pixel reconstruction model.

The standard conditional flow-matching objective transports Gaussian noise to
the next three visual skill-boundary embeddings. Both context and target encoder
paths receive gradients, as in the upstream LeWM training implementation. SIGReg
uses the unchanged 17-knot/1024-projection statistic with weight0.09. Complete
batches preserve its T,B,D estimator; padded episode-start histories exclude
invalid timestamps. Training uses AdamW, learning rate5e-5, weight decay1e-3 and
gradient clipping1.0. Singleton training batches are omitted because the
prediction projector uses BatchNorm. Model training does not use an EMA encoder.

The stochastic extension adds noise/time conditioning and causal future tokens.
Each future token is conditioned on its own canonical primitive action; causal
attention gives it access to earlier tokens. The prediction projector applies
BatchNorm separately at each future position to prevent later actions leaking
through prediction-head batch statistics. The visual encoder projector retains
the upstream joint context/target training statistics; inference freezes these
statistics. This is standard FM with Euler sampling, not MeanFlow/iMF.

## Observations, timing and replay

The model and planner receive only base-camera RGB, canonical primitive actions
and availability masks. No joints, TCP, controller latch, object poses, contacts,
skill outcome labels or actual future duration enter model conditioning. The
existing executor still uses its ordinary feedback and controller memory.
Privileged metrics remain in a separate diagnostics directory.

Every model transition is one completed, variable-duration primitive. Existing
100Hz physics,20Hz control, action definitions and termination logic remain in
the reference executor. Skill-time prediction is an explicit adaptation: three
future steps do not represent a fixed physical duration. Discount0.8 applies per
completed primitive, without a learned continuation or duration correction.

Replay stores lossless uint8 `.npy` RGB frames, SHA256 references and a JSONL
index. Each transition includes requested/canonical actions, episode/skill IDs,
boundary frames, completion, duration and termination metadata. Completed failed
interactions are usable training data. Zero-duration and interrupted skills are
retained but excluded from complete three-skill windows. History consists of
three boundary observations and two intervening primitive actions. Missing
episode-start history is masked; windows cannot cross discontinuities or resets.

Whole episodes are assigned deterministically: episode IDs divisible by
`validation_every` are held out. Normal default is5. Validation never enters
fitting. Embeddings are recomputed after encoder updates. Latent losses are
diagnostics in the current representation, not comparable physical errors across
models or a fixed scale across training rounds.

## Planning and online loop

Normal planning uses three skills,64 candidates,3 CEM iterations,8 elites,
10 future samples and10 Euler steps. Each decision reuses its observation
embedding and fixed Gaussian noise samples across all candidates and iterations.
Candidates are evaluated in chunks of8. The predictor is frozen in evaluation
mode throughout collection. Model inference never queries future simulator state.

For each candidate, compute unbiased sample variance across noise samples,
average over the192 latent dimensions, and sum future-step scores with powers
of0.8. This is predictive-distribution variance, not a parameter ensemble or a
calibrated epistemic uncertainty/information-gain estimator. It can reward
irreducible ambiguity; no exploration advantage is assumed. No goal-image,
object-approach or privileged reward is added, and variance is not clipped.

CEM keeps categorical skill probabilities and per-skill parameter Gaussians at
each horizon position. Initial probabilities are uniform, mean0 and std0.5.
Elite counts get a unit pseudocount. Distribution updates blend0.5 with previous
values, parameter std is floored at0.1, and inactive parameters are zeroed.
Only elites selecting a given skill update that skill's continuous distribution.
Execute the first primitive of the highest-scoring evaluated sequence, then
observe and replan. This is a mixed-action CEM adaptation, not unchanged iCEM.

Bootstrap uses random primitives. Subsequent rounds use `--policy curiosity`
by default; `--policy random` runs the same model-learning loop with random
collection as a separate baseline. Each collection call starts with a reset
seed `seed + 10000*(round_id+2)`, where bootstrap has round_id=-1. After every
round the model fits accumulated training replay and evaluates held-out windows.
Missing training or validation windows stop with an actionable error; there is
no state-data substitute or silent random-action fallback.

## Runtime and commands

Use the existing `3d_world_model.sif` with Torch2.3.1+cu121. Install only the pinned
JEPA additions into a separate executable target; do not upgrade the shared SIF:

```bash
apptainer exec /data/data/siyao/3d_world_model.sif python -m pip install \
  --target /tmp/jepa-runtime -r requirements/jepa.txt

apptainer exec --nv /data/data/siyao/3d_world_model.sif \
  env PYTHONPATH=.:/tmp/jepa-runtime OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m exploration.cee_us.cee_us_cabinet smoke \
  --agent epic_jepa_cem --config configs/epic_jepa_cabinet_split7.yaml \
  --output logs/epic_jepa_smoke
```

Smoke retains the full encoder/predictor and uses1,200 bootstrap controls plus
two400-control rounds, episode length200, validation every third episode,
one epoch/update, batch8,8 candidates,1 CEM iteration,2 elites,2 future samples
and2 Euler steps. These explicit smoke overrides are not a formal protocol.
They permit a real bounded pipeline test; a very small number of windows can
still fail the documented bootstrap requirement.

The normal configuration requests4,000 bootstrap controls and10 rounds of1,000.
Launching it requires an explicit budget, for example `train --total-controls
14000`, and a new output directory. The budget is an upper bound, not permission
for an automatic formal launch. Running `--device cpu` is explicit; unavailable
CUDA does not silently switch runtimes. Only reference execution is supported.

`evaluate --checkpoint <round.pt> --output <new-directory>` evaluates saved
held-out RGB replay without simulation or training. The checkpoint configuration
is authoritative for resume/evaluate. Use the saved `--policy` when evaluating a
random-policy checkpoint. Live shifted-state evaluation is not implemented.
`report --output <completed-run>` prints the saved result.

Resume with `train --checkpoint <round.pt> --output <new-directory>`. Checkpoints
store model/optimizer state, global and planner RNG states, replay frame hashes,
resolved settings, split assignments and counters. Resume restores a round
boundary, not mid-skill physics. It reuses immutable frame references from the
source run, so keep that run accessible. Replayed log history is retained in
the new output. Checkpoint schema/config/action mismatches and changed frame
bytes are rejected. Existing outputs are never overwritten.

An explicit larger `--total-controls` on `train --checkpoint` extends a saved
run's cumulative target. Only the total budget and derived round count change;
model, optimizer, random states, replay, policy and all training settings are
restored. The new target must be greater than the saved budget and equal
`bootstrap_steps + rounds * steps_per_round`. A decrease, partial-round target,
or budget extension during evaluate/smoke is rejected. The source checkpoint
is validated with its original configuration and is never rewritten; the new
manifest records `budget_extension`.

For example, extending the14,000-control pilot to400,000 uses:

```bash
python -m exploration.cee_us.cee_us_cabinet train \
  --agent epic_jepa_cem --config configs/epic_jepa_cabinet_split7.yaml \
  --policy curiosity --checkpoint <curiosity-run>/round_009.pt \
  --total-controls 400000 --output <new-output-directory>
```

Use the isolated container/runtime command above. The existing4,000 bootstrap
and1,000-control rounds imply396 total online rounds; the completed10 rounds
remain saved, so386 rounds/386,000 controls remain. The random arm uses its own
checkpoint and `--policy random` with the same target. No bootstrap is repeated.

Artifacts include `manifest.json`, `config.json`, `visual_replay.jsonl`, frames,
physical diagnostics, `training.jsonl`, round checkpoints, `progress.json`,
`result.json` and, on failure, `failure.json`. Planning audits record score,
selected sequence/skill and elapsed time. Training reports gradient norms and
held-out flow loss, predictive latent MSE, sampled variance and latent collapse
diagnostics. Peak GPU memory is Torch's allocation peak for this process, not
whole-device memory or a reservation guarantee.

## Validation

Focused tests cover visual/action alignment, episode masks, failed/interrupted
skills, privileged-state exclusion, source-frame integrity, gradients, fixed
noise reproducibility, causal action dependence, CEM action legality, model
immutability, synthetic full-loop learning and exact round-boundary resumption.

The native bounded smoke at `logs/epic_jepa_smoke_20261111` completed2,000 controls,
65 archived visual transitions and8 optimizer updates through bootstrap and two
curiosity rounds. This establishes native integration only. It does not establish
grasp success, generalization, superiority to random/GNN exploration, or calibrated
uncertainty. The paper's goal-conditioned exploration findings do not establish
the effectiveness of this new pure-curiosity online protocol.

The smoke archived54 completed and11 interrupted/zero-duration transitions over
10 episodes. Final held-out evaluation used7 windows: latent prediction MSE0.6964,
sampled future variance1.1263, and across-observation latent variance2.83e-6.
That small representation spread and the tiny training budget do not establish
useful representation learning. Peak process Torch allocation was2,214,916,096
bytes (about2.06GiB); mean reduced-budget planning latency was0.0407s and0.0376s
in the two rounds. These timings are not the normal64-candidate/10-sample budget
and were observed on a shared GPU.

Separate saved-replay evaluation reproduced all final held-out metrics within
rtol1e-6/atol1e-8. Completed-boundary native restoration preserved the full
history/2,000-control/8-update counters with zero new controls and no new frames.
An offline audit verified every referenced frame checksum, canonical action,
within-episode boundary continuity and the exact2,000-control duration sum.

### Intermediate-checkpoint resume audit

A separate new-process test resumed `round_000.pt` at1,600 controls/5 updates
into `logs/epic_jepa_resume_midround_20261111`. It executed only the remaining400
controls, added8 visual transitions and3 optimizer updates, and reached the
declared2,000-control/8-update endpoint. Bootstrap and the first online round
were not repeated. Prefix history, episode/skill counters, replay labels,
high-level primitive actions and final Python/NumPy/Torch/CUDA/planner RNG states
matched the uninterrupted run exactly. The strengthened synthetic production-loop
regression also compares model, optimizer and all RNG states exactly.

Native continuation is **functional round-boundary resume, not bitwise physical
replay**. The first starting diagnostic state and low action match, but the next
physical state differs. Ordinary ManiSkill resets reuse a native scene in the
uninterrupted run; restarting creates a fresh scene. No native scene snapshot is
saved. The specific internal source of numerical divergence has not been isolated,
so no controller/physics formula or reset protocol was changed to force agreement.
Differences can accumulate during subsequent physical steps and affect RGB data.
The final model state (including buffers) differs by at most9.66434e-5 in this
test; optimizer tensors differ by at most1.24556e-6. These observations are not
universal error bounds or a guarantee of identical future trajectories.

The audit script and machine-readable report are
`artifacts/jepa_resume_audit_20261111/audit.py` and `report.json`.
To resume an intermediate checkpoint in the existing isolated runtime:

```bash
apptainer exec --nv /data/data/siyao/3d_world_model.sif \
  env PYTHONPATH=.:/tmp/jepa-runtime OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m exploration.cee_us.cee_us_cabinet train \
  --agent epic_jepa_cem --config configs/epic_jepa_cabinet_split7.yaml \
  --checkpoint logs/epic_jepa_smoke_20261111/round_000.pt \
  --output logs/epic_jepa_resumed_new
```

Choose an unused output directory and retain the source replay frames. A crash
within a round resumes from its last completed checkpoint and recollects that
unfinished round; it does not continue from the exact interrupted skill/tick.
