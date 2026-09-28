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

There is one supported learned-model path: decision observation v4, tensor
schema v2, model/runtime v2. These are format versions, **not** training
generations. Native artifact integrity, scope, candidate alignment, and schema
checks remain mandatory. Observation-v3/tensor-v1 bridges and their command-line
flags have been removed; old artifacts are rejected rather than adapted.

The current joint dataset is still schema v2. It is retained until separate
policy-decision and actual-boundary value samples replace it. The new
candidate-free `StableValueObservation` exists, but is not yet a training input.

The unused curriculum, callback-only generation coordinator, and callback-only
policy-iteration wrapper have been removed. **There is no executable complete
learning loop yet.** Their removal does not remove the replay, registry,
checkpoint, split, or arena primitives needed to implement that loop.

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
