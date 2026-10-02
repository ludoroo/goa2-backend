# Learned training and evaluation operations

**Status: cleanup before the fresh Gen1 pipeline.** Working generation, training,
and arena tools remain available, but they still use decision-level joint
training data and historical search leaves. Do not use them to declare a fresh
Gen1 run until [AI_LEARNING_CONTRACT.md](AI_LEARNING_CONTRACT.md) is implemented
end to end.

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

Merged #19 adds separate native indexing, training-batch, and loss APIs. The
current locally verified library checkpoint adds opt-in recorder completion
provenance and persistent seed splits/replay; final verification and review
results are recorded in the learning contract.
This is not trainer/generator adoption. Existing commands cannot train or run
this Gen1 artifact format. No old artifact, index, dataset, or model weights are
converted. Full generator adoption, trainer/optimizer/parent initialization, and
executable iteration remain gated.

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
actual generator adoption remains separate.

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
scheduling, strata such as latest/recent/hard, physical dataset resolution and
revalidation at future training consumption, and the executable learning loop
remain future work.

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
**same run**. This is not cross-generation parent initialization, which remains
part of the Gen1 implementation work. No immutable artifact should be published
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

Run the full suite plus Ruff, Black checks, and mypy. Once the new search/data/
model contract is integrated, run a tiny generate → train → native runtime load
→ paired gameplay diagnostic before scaling. Until then, use tests and CLI
`--help` smoke checks; do not launch another historical-style generation.
