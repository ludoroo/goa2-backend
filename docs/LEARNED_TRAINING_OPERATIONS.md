# Learned training and evaluation operations

**Publication approval:** the owner approved two logical commits and one combined
draft PR for the verified native run/evaluation checkpoint. Verification-time
references below to uncommitted/unpublished work describe the frozen test state;
commit/PR publication is now authorized. Merge and experiments are not.

**Status: native Gen1 run and paired-evaluation libraries locally verified;
pilot not authorized.** Existing generation, training, and arena CLI tools still
use decision-level joint data and historical search leaves. They are not native
Gen1 commands. Use the library contracts below and
[AI_LEARNING_CONTRACT.md](AI_LEARNING_CONTRACT.md); do not start a campaign without
an explicitly approved budget.

Historical findings and verdicts live in
[AI_EXPERIMENT_JOURNAL.md](AI_EXPERIMENT_JOURNAL.md) and the
[lineage ledger](LEARNED_MODEL_LINEAGE.md). The old stage-by-stage commands are
retired; Git history retains them. Old models are not bootstrap dependencies.

## Supported code and formats

| Responsibility | Retained implementation |
|---|---|
| Actual game execution and recording | `automata.harness.game_runner`, `trajectory`, `value_boundaries` |
| Generation and publication | `training.generation`; `scripts.generate_joint_bootstrap`, `run_self_play`, `run_parallel`, `reconcile_self_play` |
| Dataset access and learning | `training.dataset`, `indexed_dataset`, `splits`, `trainer`, `losses`, `metrics` |
| Replay and artifact registration | `training.replay_buffer`, `registry`, `io` |
| Evaluation | `evaluation.arena`, `arena_stats`, `protocol`, `provenance`, `promotion_gates`; `training.evaluate_artifact`; `scripts.run_learned_arena` |

The retained CLI/serving learned-model path uses decision observation v4, tensor
schema v2, joint model/runtime v2. These are format versions, **not** training
generations. Artifact integrity, scope, candidate alignment, and schema checks
remain mandatory. Observation-v3/tensor-v1 bridges and their command-line flags
have been removed; old artifacts are rejected rather than adapted.

The current CLI joint dataset is still schema v2. It is retained until the
native policy/value path is adopted end to end. Native records and whole-game
publication landed in #16, and candidate-free tensor preparation landed in #17.
The native model/runtime checkpoint adds `Gen1PolicyValueModel` and
`Gen1SharedEncoderRuntime` with separate policy and stable-value inference APIs.
Its artifact manifest schema 3, kind `GEN1_POLICY_STABLE_VALUE`, and runtime
compatibility 1 are distinct from the retained joint path; both loaders reject
the other format. The stable search adapter is an explicit library API, not a
new CLI or serving mode.

Merged #19 adds separate native indexing, training-batch, and loss APIs. Merged
#20 adds opt-in recorder completion provenance and persistent seed splits/replay.
Merged #21 implements replay-bound one-logical-batch training/export and concrete
single-game generation (5,316 full-suite and 366 focused tests at that checkpoint).
The subsequent bounded run driver and native paired-evaluation library checkpoints
are locally verified but uncommitted: latest verification passes **5,399 full-suite
and 508 focused tests**, **87.76%** branch-aware GoA2 coverage (80% gate), and
Ruff/Black/mypy/diff checks; independent final review found no remaining blockers.
Existing commands cannot train or run this Gen1 artifact format. No old artifact,
index, dataset, or model weights are converted. Checkpoint resume, CLI adoption,
executable iteration, and experiment authorization remain separate gates.

The unused curriculum, callback-only generation coordinator, and callback-only
policy-iteration wrapper have been removed. **There is no executable complete
learning loop yet.** Their removal does not remove the replay, registry,
checkpoint, split, or arena primitives needed to implement that loop.

## Native index and loss contract

`training.native_indexed_dataset` consumes an explicit canonical source receipt
and a physical source root. The ordered receipt names each per-game `.jsonl` or
`.jsonl.zst` file and pins its game ID, exact SHA-256/size, and total/head/boundary
counts. There is no implicit directory glob. `create_native_source_receipt`
constructs an inventory from explicit logical names; it is **not** evidence that
an arbitrary file came from legitimate terminal gameplay. Controlled recorder
completion receipts are a distinct type and admission requirement (see below);
the native single-game generator below issues these through the controlled recorder.

The source digest hashes canonical receipt bytes. The dataset digest hashes
canonical uncompressed record bytes in receipt order. Moving the physical source
root preserves both; changing compression changes the source digest, not the
semantic dataset digest. Reordering sources changes semantic dataset identity.
Game/seed/provenance is retained for split/replay admission; repeated world seeds
across distinct games are not separate train/validation entitlements.

The index stores homogeneous policy/value JSONL-zstd chunks, pinned to both
native tensor schemas. Opening/rebuilding validates the explicit receipt and
source files. Cache reuse is intentionally a full integrity check, not a cheap
handle open: it reconstructs source order, verifies semantic identity, and checks
schema collation. Budget for full source/chunk scans at open; a later optimized
cache must preserve those guarantees. Each completed game is the resumable staging
unit, and only a complete verified index is published. Native cache/staging
ownership is explicit: unrelated files/directories and any path overlapping a
source or receipt must never be replaced. Owned cache contents are disposable;
keep user files elsewhere. Loading validates chunk hashes, canonical rows, tags,
offsets, and record contracts before collation. JSON chunks are not
pickle or a tensor-cache migration. Memory is bounded by chunk buffers plus
per-game/per-chunk metadata, not accumulated game observations. Compressed input
and decoded output are bounded by their declared chunk sizes, with frame-size
checks before decompression; this is not an absolute process-memory quota for an
arbitrarily rewritten manifest.

Recovery is deliberately fail-closed, not lossless at every process-kill point.
A kill near publication can discard resumable staging work or leave an `.old`
backup; a self-consistently rewritten stage that disagrees with its sources can
require manual removal. Inspect and remove only verified, owned native cache/
staging data before retrying. Never remove source files/receipts, and never add
an ownership marker to unrelated data just to make cleanup proceed. Valid source
receipts/files remain the authority for rebuilding a disposable index.

`training.native_batches` collates policy records into `DecisionBatch` and value
records into `StableValueBatch`. Policy targets preserve exact candidate order
and visit probabilities; priors remain separate diagnostic metadata. Value
batches and metrics contain no candidate table or policy target. Missing heads
produce no batch; game-level head masks/counts identify which games contribute.

`training.native_losses` exposes `native_policy_loss` (CE and optional entropy)
and `native_stable_value_loss` (bounded-score probability BCE). For each head:

```text
row_weight = 1 / full_game_row_count_for_this_head
head_loss = sum(row_weight * row_loss) / head_normalizer
```

The default normalizer `1` gives an additive sum of game contributions. For a
mean over a complete logical batch, pass its number of contributing games for
that head to **every** chunk; never use the chunk's row count, current weight
sum, or number of represented games. Sum chunk contributions before the future
optimizer step. Policy-only games do not enter the value denominator and vice
versa. Use independent row masks when needed; inactive/empty heads yield a
differentiable zero without a fabricated observation from the other head.
Outputs, targets, and weights must use matching floating dtypes/devices; this
checkpoint uses float32 CPU batches and does not add mixed-precision training.
Cross-head coefficients, regularization (once per optimizer step), scheduling,
and optimizer behavior belong to the later trainer checkpoint.

## Native completion, splits, and replay

These are library APIs, not new commands or authorization to generate data.

### Controlled completion, not a directory scan

Opt into `NativeDatasetRecorder(..., completion_target=NativeCompletionTarget(...))`
to issue `normal-decisive-native-game-v1` sidecars. Raw/default recording is
unchanged and supplies no completion provenance. The controlled path requires
nonzero recorded rows, a normal decisive `game_over` callback with a canonical
RED/BLUE winner, complete spool validation against live recorder counts, and
successful no-clobber source publication. Its durable canonical sidecar binds
full game identity, exact bytes/hash, and policy/value/boundary counts. The
recorder and receipt contracts remain Torch-free. Controlled paths are anchored
at construction, so later working-directory changes cannot redirect publication.
Equal or nested source/sidecar destinations reject before filesystem changes.

`create_native_dataset_completion_receipt` combines an explicit ordered list of
sidecars after validating their sources; it never discovers or blesses raw files.
`create_native_source_receipt_from_completions` creates the separate inventory
needed by the native index. Replay still requires the completion set itself and
binds its digest; an inventory alone cannot enroll games.

Ordinary sidecar-publication failure rolls back this recorder's newly published
source, without replacing a competing sidecar. A process kill between source and
sidecar publication can leave an untrusted orphan. Future reconciliation must
identify and delete/rerun such orphans, never promote them by scanning. No cleanup
command is added here. Completion is controlled-pipeline evidence, **not** a
cryptographic signature proving honest execution against hostile callers.

### Immutable seed-only membership

`NativeSplitConfig` requires explicit, disjoint, nonnegative half-open purpose
ranges plus a fixed namespace, salt, and validation fraction. No phase-0 seed
registry is inherited. The `native-seed-split-v1` SHA256 threshold depends only on
the recipe, namespace, salt, and world seed—not generation, game ID, map,
composition, or arrival order. The comparison uses the float fraction's exact
integer ratio. Bootstrap/training ranges use the threshold; dedicated validation
ranges are always validation. Evaluation, arena, screen, and promotion seeds
cannot be enrolled. Repeated seeds retain the same assignment across generations
and reloads. Exact per-cohort/per-stratum quotas and advanced map/composition
holdouts are not supplied by this recipe.

The pure ledger is embedded in the replay catalog. Do not create a separately
mutable ledger file whose publication could drift from replay admission.

### Atomic replay admission and whole-game selection

`update_native_replay_catalog` validates under a sibling lock, then atomically
publishes one canonical manifest. The complete lock ownership marker is also
published atomically, so cooperating first writers cannot observe a partial
marker. Completion sources must match the native index
one-for-one and in order, including identities, exact bytes, counts, and semantic
dataset identity. One enrollment has one generation and one source-model digest.
Bootstrap requires both model digest and parent artifact to be `None`. Learned
enrollment requires the exact compatible Gen1 parent, both tensor schemas,
stable-outcome semantics, and the **complete current runtime scope**: all current
maps, game types, heroes, and exact adapter versions—not only the subset present
in one generation. Future native trainer/exporter wiring must satisfy this; the
retained trainer's observed-dataset scope must not be reused blindly. Legacy joint
artifacts are not compatible parents. Admission deliberately performs multiple
source scans plus index validation; budget for full I/O, not a cheap metadata open.

All generation seeds enter the ledger; only TRAIN game references enter replay.
Duplicate generations, game IDs, datasets, incompatible configuration, and
forbidden seed purposes fail without publishing a partial catalog. Optional
capacity evicts oldest retained whole-game references, preserving generation
history and the seed ledger. Capacity limits retained games, not accumulated
history or ledger size. Keep the catalog outside disposable index caches; its
portable references bind digests and logical names, not physical roots.

Controlled publication and catalog paths reject symlinks in ancestor directories
as well as final files. Use canonical physical directories (for example,
`/private/tmp/...` rather than macOS's `/tmp` alias). This deliberate strictness
is stronger than the raw recorder/index path policy. Topology is rechecked before
controlled publication, but hostile concurrent filesystem replacement after that
check is outside this contract; directory-descriptor-relative hardening is not
implemented.

`sample_native_replay` implements `uniform-train-games-v1`: deterministic uniform
sampling without replacement over complete game references, with separate policy
and value contributing-game counts. It loads no rows or tensors. A head with no
contributors must be skipped, not given a synthetic denominator. Optimizer
scheduling, strata such as latest/recent/hard, and the executable learning loop
remain future work. Physical binding and consumption-time revalidation are
provided by the native trainer below.

## Native trainer and single-game generator (merged in PR #21)

This checkpoint adds callable library paths, not an epoch runner or a complete
learning loop. Bounded optimizer and terminal-game tests are not authorization to
run an actual training, generation, or arena experiment.

### Replay-bound optimization and export

`native_gen1` centralizes complete-current-scope enumeration and exact Gen1 parent
validation for replay, training, and generation. Full scope tuples must use
canonical sorted order, deliberately stricter than the earlier replay-only set
comparison. Parent artifacts supply verified
CPU-float32 weights only. Legacy artifacts, incompatible schemas/scopes/digests,
and silently cast floating weights must reject. A fresh Adam optimizer is created
for both fresh bootstrap initialization and cross-generation parent initialization;
this is **not** optimizer resume. The first trainer requires `dropout=0.0`;
nonzero dropout, mixed precision, and non-CPU training are not supported. Parent
loading and fresh initialization preserve the caller's CPU Torch RNG state.
Manifest-only shape/dtype validation uses meta tensors rather than allocating
parameter storage. This is not an absolute quota for hostile metadata.

`bind_native_replay_sample` resolves a catalog/sample through explicit physical
source, inventory, completion-set, and index-cache paths. All four paths must be
absolute so binding authority cannot change with the working directory. Opening
bindings also rejects symlinked ancestors: use physical `/private/tmp` rather
than macOS's `/tmp` alias. This tightens native binding path acceptance; legacy
joint APIs are unchanged. Bindings must match exact
retained TRAIN references and all dataset/source/completion digests. Metadata-only
references and raw inventories do not bypass completion or split validation.
Binding and training consumption revalidate the physical data; budget for full
source/index scans, not a cheap hot-path handle lookup. Initial binding may build
or rebuild a disposable cache. Training consumption uses strict read-only opening
(`open_native_indexed_dataset(..., rebuild=False)`): invalid or missing caches,
including corrupt chunks in unselected games, reject rather than being repaired.
Use distinct cache directories for different chunk-size configurations; an
explicit rebuild can invalidate earlier handles.

`NativeTrainer.train_logical_batch` streams separate policy/value chunks under the
sample-wide contributing-game denominator for each head. Head coefficients are
explicit. L2 is the sum of trainable parameter squares, applied once per logical
batch when enabled, with no additional Adam weight decay. Disabled L2 does not
create zero gradients or Adam state for otherwise unused heads; its reported
regularization term is zero. Clipping and the optimizer step each
occur once after all chunk gradients are accumulated. An absent or disabled head
gets no synthetic row or denominator mass; a selection with no enabled contributor
rejects. Nonfinite losses/gradients fail before stepping. If the optimizer itself
fails after potentially changing state, the trainer must become unusable rather
than claim transactional rollback.

Export uses the full current runtime scope, not merely observed maps or heroes.
Provenance binds configuration, initialization/parent, successful optimizer-step
lineage, replay/sample/dataset/source/completion identities, and source revision.
Physical paths are not portable artifact identity. Stale, foreign, or failed-step
results cannot authorize export. Configuration and initialization identity are
pinned at trainer creation; changing them mid-lineage rejects instead of
mislabeling earlier updates. Chunk parity is tested to float32 tolerance, not
bit-exact reproducibility from exported provenance alone.

### Concrete actual-play generation

`generate_native_game` constructs one real stable-transition ISMCTS teacher per
side and runs one explicitly configured game through the existing harness. It
records only played root search targets and actual live stable boundaries. The
recording wrapper follows visit sampling so the target's selected action is the
one actually played; counterfactual leaves never become terminal-labeled rows.

Heuristic bootstrap uses heuristic prior/continuation/value. Exact Gen1 parents
provide learned root policy, sampled controlled continuations, and candidate-free
stable values, with explicit heuristic environment/foreign routing. Separate
SEARCH/ACTION/ENVIRONMENT streams are derived per side and world seed from a
pinned namespace. The generator-settings ID excludes generation ID and source
revision/dirty-tree hash; those remain separate game-identity fields. No Phase-0
experiment defaults are imported. Seed purpose must
match explicit split ranges; evaluation/arena/screen/promotion seeds reject before
play. Search is iteration-bounded with deterministic progression guards, no
cooperative decision deadline, and no incompatible request schedule.

Only normal decisive nonempty games receive controlled completion receipts.
Max-step/round caps and exceptions leave no newly certified training game and
must preserve competing/preexisting files. There is no wall-clock watchdog,
multiworker coordinator, reconciliation/resume command, automatic catalog
admission, or automatic next generation in this API. Those remain separate
implementation and authorization gates.

## Bounded native run driver (locally verified, uncommitted)

`native_run_contracts.create_native_run_manifest` pins the explicit configuration
and physical authorities; `native_run.run_native_one` executes it;
`native_run_contracts.load_completed_native_run` verifies completion and products.
There is no command-line entry point.

This library checkpoint connects the verified primitives into one finite run,
not an automatic learning loop. Its manifest pins an ordered game cohort, one
teacher/initialization pairing, immutable split configuration, explicit search
and game caps, index chunk size, replay capacity, a fixed optimizer-step count,
and one explicit sampling seed per step. There is no replacement generation:
a censored, empty, or exceptional game fails the run and leaves later games
unattempted. Parent weights always get a fresh optimizer.

The output root must be new, absolute, and reached without symlink components
(use physical `/private/tmp`, not macOS's `/tmp` alias). Controlled dataset
binding authorities follow the same physical-path rule. An immutable manifest and atomic
RUNNING/FAILED/SUCCEEDED snapshots describe progress; a digest-bound completion
marker is published last. Partial products remain available for diagnosis after
failure, without implying that the run succeeded or can be resumed. Only the
completed-run loader may treat a matching manifest/result/marker as completion.

Validation reports separate equal-game-weighted policy CE, policy entropy, and
stable-value BCE, before and after the fixed updates, over the same certified
validation set. Missing heads report `None` with zero contributing counts.
Validation must be read-only: no model/gradient/mode/RNG mutation, no cache repair,
and no optimization, adaptive budget, or best-model choice from validation data.
Its portable ledger records every explicitly supplied dataset authority, including
TRAIN-only datasets with no validation references. Bindings must match this exact
inventory; metric authority arrays describe that inventory, while game IDs and
head counts describe only validation games. The general validation API permits
unrelated previously enrolled seeds outside the explicit inventory. The run driver
additionally requires its validation IDs to equal the complete planned held-out
cohort. Model-shape checks use meta tensors instead of allocating duplicate CPU
weights; both numeric zero representations of disabled dropout are accepted.
Mutation detection still snapshots model parameters, gradients, and buffers:
chunk streaming bounds data memory, not this model-sized safety overhead. Current
Gen1 policy output is expected to be finite even at masked candidate positions.

**The guarantee is held out from current-run updates.** The run creates one local
catalog; it does not resume a previous catalog or verify an ancestor's complete
training exposure. Artifact-only parent loading cannot establish globally unseen
validation seeds. Results identify this scope explicitly and report inherited
parent training exposure as unknown. Preserving and verifying cross-run split
ancestry requires a separate protocol; resetting/changing split context is not
silently declared safe.

There is no CLI, wall-clock watchdog, optimizer checkpoint resume, multiworker
coordination, automatic next generation, arena evaluation, or experiment
authorization in this checkpoint. Final verification passes **5,359 full-suite
tests**, **420 focused tests**, **87.76%** GoA2 branch-aware coverage, and all
Ruff/Black/mypy/diff checks. Independent reviews and follow-ups found no remaining
blockers. All 746 source/test/dependency fingerprints remained unchanged; the
original parked checkout is preserved. These are local bounded fixtures, not
remote CI, experiments, or playing-strength evidence. Commit and publication
require separate approval.

## Native paired gameplay evaluation (locally verified, uncommitted)

This library-only slice evaluates frozen Gen1 artifacts against explicit
controls, without training, recording training rows, or admitting replay data.
Its public APIs are `training.native_paired_contracts.create_native_paired_evaluation_manifest`,
`training.native_paired.run_native_paired_evaluation`, and
`training.native_paired_contracts.load_completed_native_paired_evaluation`.
Native artifact evaluation belongs to the training-owned workflow, alongside
existing artifact evaluators; the generic evaluation package stays independent.
The architecture test and all previously verified files are unchanged.

Three comparisons are mandatory: full candidate-versus-heuristic search; the
same candidate policy/continuation with learned versus heuristic value leaves;
and candidate-versus-heuristic policy argmax with **zero search**. An optional
fourth compares full candidate search with an explicitly pinned Gen1 parent.
Search modes use identical fixed budgets and an explicit common heuristic
opponent model; that model is not a claim to simulate the played opponent exactly.
Nonbranchable requests, including simultaneous/UPGRADE decisions, use the common
heuristic environment in every arm; the learned comparison applies to supported
branchable decisions. Within each pair, board rosters, map, game type,
and world seed stay fixed while candidate and baseline swap sides. This lets both
policies play both lineups instead of confounding policy quality with a dedicated
candidate roster.

The native engine has no draw rule: normal completion requires a normalized
winning side. Censored games retain no winner or score; only pairs with two normal
decisive games contribute to a paired score. Remaining declared cases still run
once after censoring, without replacement or budget adaptation. Unexpected engine
or inference errors fail the evaluation rather than being hidden as censoring or
heuristic fallback. Declared evaluation seed ranges do not prove that an artifact's
ancestors never trained on those seeds. Results make this explicit with
`evaluation_scope="DECLARED_EVALUATION_SEEDS"`,
`artifact_training_exposure="UNKNOWN"`, and
`strength_claim="DESCRIPTIVE_COMPLETED_PAIRS_ONLY"`. A completed protocol can
contain only censored pairs and legitimately report no score. Matching streams
start with the same physical-side seeds in both legs, but divergent actions need
not consume subsequent random events identically. Seed tuples use canonical JSON
encoding, preserving namespace/fixture boundaries even with embedded separators.

`rounds` records the harness's absolute counter, not completed-round count. Games
start at round 1; a `max_rounds` censor normally reports `max_rounds + 1`.
The observation validator accepts that truthful counter and rejects larger values.
Effects are registered before canonical full-scope capture/revalidation, so a fresh
process and a process that has already played games agree. All artifact/runtime
compatibility preflight completes before claiming the output directory. Failure
messages are UTF-8-safe and stripped after truncation; valid FAILED evidence does
not replace the original exception.

Final local verification: **5,399 full-suite tests**, **508 focused tests**,
**87.76%** GoA2 branch-aware coverage, Ruff/Black/mypy/diff checks, and independent
follow-up review with no remaining blockers. All 751 frozen fingerprints and the
746-file prior checkpoint are preserved. Real pytest fixtures cover the trained
artifact handoff, censoring at step and round limits, and inference failure without
fallback. These are correctness fixtures, not performance or playing-strength
experiments. No evaluation experiment, CLI, resume, automatic promotion, or
iterative training campaign is authorized.

## Retained command entry points

Inspect the actual parsers rather than copying a historical experiment recipe:

```bash
PYTHONPATH=src uv run python -m automata.scripts.run_parallel --help
PYTHONPATH=src uv run python -m automata.scripts.generate_joint_bootstrap --help
PYTHONPATH=src uv run python -m automata.scripts.run_self_play --help
PYTHONPATH=src uv run python -m automata.scripts.reconcile_self_play --help
PYTHONPATH=src uv run python -m automata.training.trainer --help
PYTHONPATH=src uv run python -m automata.training.evaluate_artifact --help
PYTHONPATH=src uv run python -m automata.scripts.run_learned_arena --help
```

- `run_parallel joint-bootstrap` partitions a half-open seed range, resumes
  validated shard checkpoints, and merges complete games in stable order.
- `run_parallel self-play` coordinates four persistent workers and invokes the
  bounded-memory reconciler only when all workers succeed. `run_self_play` is
  the lower-level single-worker command.
- `run_learned_arena` pairs each seed with candidate-on-RED and candidate-on-BLUE.
  It verifies native artifacts against pinned digests and matchup scope. The
  supported per-side matrix cells are `L/H` and `L/L` (policy/value); keeping the
  same policy while changing the leaf evaluator supports controlled ablations.
- Search configuration is strict. Unknown keys, invalid types/enums, and a
  caller-supplied search seed are rejected. The search-config field
  `decision_timeout_seconds` must be `null`: serving's cooperative deadline can
  return partial visits and is not an offline teacher budget. Outer CLI/game
  watchdogs remain available and censor failed games rather than recover partial
  teacher evidence. Self-play/arena derive their own per-world/per-side streams.
  Use a shared random-stream namespace only for intentionally matched arms.

## Data integrity, seeds, and resuming

1. **Publish complete games only.** Decision timeout, whole-game timeout,
   progression failure, and incomplete games must not become winner-labeled
   training rows. Preserve provisional fragments for diagnosis, not as training
   examples. Returned visit counts must match the declared effective search
   budget; a partial result discards all earlier provisional rows from that game.
   A boundary observer exception aborts the run; discard its game spool rather
   than continuing a partially processed session.
2. **Respect seed ownership and game-level splits.** Current experiment seed
   ranges are in `training.experiments.phase0`. The trainer's
   `--dataset-seed-purpose` must match the source corpus. Never move validation
   or arena seeds into training/replay merely by changing a run name.
3. **Resume only the same work.** Configuration, source revision/dirty content,
   seed assignment, parent digest, and data/split identities must match.
   Changed code or settings require fresh output/checkpoint destinations. Hidden
   source-identity options support worker coordination, not pretending changed
   code is an old run.
4. **Preserve provenance.** Canonical sidecars/checkpoints bind source, resolved
   configuration, and artifact/dataset identities. `runs/` and explicit output
   roots are excluded from source hashes; untracked source changes are not.
   Never edit manifests or relabel old evidence to force compatibility.
5. **Keep output handling safe.** Publication is locked/atomic and rejects unsafe
   paths, corrupt fragments, and incompatible identities. Artifact deletion is
   a separate inventoried task; no cleanup command here deletes `runs/`.

## Indexed training

Datasets support JSONL and `.jsonl.zst`. Dataset identity is computed over the
validated uncompressed row bytes; source/index identity additionally binds the
source container bytes. Loaders must be exhausted so end-of-stream invariants
are checked. Do not hand-assemble externally supplied JSON rows; use the typed
writer and preserve duplicate/identity checks.

The disposable `<dataset>.index` cache is bound to source bytes, complete tensor
schema identity, chunk format, and **metric metadata version 2**. Stale caches
are rebuilt, not accepted as historical-compatible inputs. Per-game tensorizing
work is checkpointed in `.<index>.staging`; publication occurs only after every
required game is complete. A lock serializes builders for a destination.

`--dataset-index`, `--index-workers`, and `--decisions-per-chunk` control storage,
parallelism, and bounded working memory. A rebuild temporarily needs both old
and replacement cache space. Do not delete staging directories during active
work. Tensor chunks are loaded with `torch.load(..., weights_only=True)`.

Current training checkpoints resume model, optimizer, RNG, and progress for the
**same run** on the retained joint path. This is distinct from the native Gen1
library's verified parent-weight initialization with a fresh optimizer. No immutable artifact should be published
from a failed or interrupted training run.

## Reading evidence

- Compare candidate and parent on identical dataset/split memberships and keep
  metrics game-aware. Singletons and all-tied targets cannot establish policy
  ranking quality; report informative decision and game counts.
- Search overturn compares **actual root priors** with search targets. Missing
  priors produce unavailable evidence, not a visit-derived substitute.
  `q_variance` is within-action return variance, not between-action Q separation.
- Use typed decision roles. `SELECT_HEX` is `SPATIAL_SELECTION`; that category
  includes more than movement. Do not derive semantics from prompt text.
- Value in `[-1, 1]` maps to probability by `(value + 1) / 2`. Neutral probability
  has Brier score 0.25 and log loss `ln(2)`. Good prediction does not establish
  improved search control or gameplay strength.
- Report paired gameplay, completion reasons, operational failures, and cost.
  Timeouts are censored/reliability evidence, not ordinary draws or strategic
  losses. Arena results do not by themselves provide per-decision latency
  evidence. Promotion helpers remain utilities, not an automatic acceptance
  policy for the fresh lineage.

## Verification before fresh generation

The full suite plus Ruff, Black checks, and mypy pass for the current native
library snapshot. A tiny generate → train → native runtime load → paired gameplay
handoff is also covered by pytest, without establishing useful learning or strength.
The next experimental step is a separately approved, explicitly budgeted pilot
before scaling. Do not launch another historical-style generation or reuse the
retained CLI commands as native Gen1 entry points.
