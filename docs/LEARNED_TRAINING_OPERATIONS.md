# Learned training and evaluation operations

## Vocabulary and boundary

**Learned** means model-informed search behavior. ISMCTS remains the search
algorithm. Its two independent extension points are policy prior and leaf value.
The canonical matrix is:

| Cell | policy prior | leaf value |
|---|---|---|
| H/H | Heuristic | Heuristic |
| H/L | Heuristic | Learned |
| L/H | Learned | Heuristic |
| L/L | Learned | Learned |

Use **shared encoder** for the concrete Torch model, artifact, and
`SharedEncoderRuntime`. Offline generation, datasets, replay, training,
curriculum, experiment declarations, registry, and policy iteration live in
`automata.training`. Arena evidence and promotion decisions live in
`automata.evaluation`. The shared offline headless game runner and trajectory
recorders live in neutral `automata.harness`. Product `automata.runtime` and
`automata.search` may not import any of these offline packages. Training and
evaluation may depend on the harness, product runtime/search contracts, and
concrete shared encoder. They do not import each other except that policy
iteration necessarily composes evaluation; evaluation never imports training.

Applications compose policy iteration through
`automata.training.policy_iteration.run_policy_iteration`. It is not a
standalone command because the registry, generation pipeline, and arena runner
are application-provided dependencies.

Policy-only is explicitly a non-ISMCTS baseline. Root-shallow and multiply
search are deferred: they do not yet add enough evidence to justify another
algorithm family or operational surface.

## Data contract

`JointDatasetRow` stores the exact `DecisionObservation`, ordered legal
candidates, selected engine value, policy target, terminal value target, and
complete source/game/search provenance. Files are canonical JSONL. Decisions
are immediately serialized to a temporary compressed disk spool; no game-sized
observation buffer is retained. The final dataset is streamed and atomically
published only after a normal terminal result. Decision and dataset identities
are SHA-256 content identities.

Policy loss is masked cross entropy over legal candidates. The bounded value
score in `[-1, 1]` is converted explicitly to win probability with
`p = (value + 1) / 2`; value loss is binary cross-entropy after clamping that
probability away from zero and one. Weighting is game-aware so long games
cannot silently dominate. Policy metrics include cross-entropy, top-1,
top-k recall, pairwise accuracy, entropy, search-overturn rate, and Q variance.
Value metrics include log-loss, Brier score, expected calibration error,
accuracy, and saturation rate. Metrics are reported overall and by relevant
candidate-family, hero, map, composition, and round buckets. Splits group by
game and can hold out source, map, mode, or composition; no game may cross
split boundaries.

`ReplayBuffer` validates schema, experiment scope, seed ownership, and parent
artifact compatibility before deterministic capacity sampling. It does not
rewrite source rows.

## Bootstrap and self-play

Run deterministic heuristic bootstrap with:

```bash
PYTHONPATH=src uv run python -m automata.scripts.generate_joint_bootstrap \
  --out runs/bootstrap.jsonl.zst --checkpoint runs/bootstrap.checkpoint.jsonl \
  --seed-start 10000 --seed-end 10100 \
  --target-source heuristic --target-recipe one-hot-exact-choice \
  --max-steps 10000 --timeout-seconds 300
```

Parallel orchestration partitions a half-open seed range into disjoint shards,
resumes each shard from an append-only checkpoint, and merges by stable
decision identity:

```bash
PYTHONPATH=src uv run python -m automata.scripts.run_parallel joint-bootstrap \
  --out runs/bootstrap.jsonl.zst --checkpoint runs/bootstrap.checkpoint.jsonl \
  --seed-start 10000 --seed-end 10100 --workers 4 -- \
  --target-source heuristic --target-recipe one-hot-exact-choice \
  --max-steps 10000 --timeout-seconds 300
```

The parent process reports aggregate progress from the shard checkpoints; it
never infers game completion from worker logs. A tqdm-backed bar shows native
throughput and ETA plus active workers and terminal outcome counts. Existing
checkpoint records appear as the initial completed count when resuming. Use
`--progress-interval SECONDS` before `--` to change the default two-second
refresh cadence, or `--no-progress` to disable the bar. Arguments after `--`
continue to be forwarded only to the generator workers.

Self-play uses a content-addressed champion artifact and the same game and row
contracts. Seeds belong to named, disjoint ranges. A retry must preserve world
seed, agent seed, configuration identity, source revision, dirty-tree digest,
and parent artifact digest. Never merge partial games.

## Training

Train the bootstrap dataset with:

```bash
PYTHONPATH=src uv run python -m automata.training.trainer \
  --dataset runs/bootstrap.jsonl.zst --split-manifest runs/split.json \
  --checkpoint runs/training.checkpoint.pt --run-manifest runs/training.json \
  --artifact runs/model --seed 20000
```

`automata.training.trainer.train_joint(JointTrainingConfig(...))` creates or validates a grouped
split manifest, trains both heads, checkpoints optimizer/model/RNG/progress,
and resumes only when the complete configuration and input identities match.
Success atomically exports a PR2-compatible model directory. Validate every
result with `SharedEncoderRuntime.from_artifact(...).evaluate(...)` before arena use.
Run manifests are canonical and immutable evidence; failed or interrupted runs
must not publish an artifact.

Training datasets use transparent Zstandard compression when their path ends in
`.jsonl.zst`; legacy `.jsonl` datasets remain readable and writable. Dataset
digests are SHA-256 over canonical, uncompressed JSONL rows, so compression
settings and container bytes do not change semantic dataset identity. Checkpoint
and audit JSONL files remain uncompressed. `iter_joint_dataset` strictly validates
rows and cross-row invariants incrementally; consumers must exhaust it because a
truncation or end-of-stream invariant can be reported after earlier rows were
yielded. The trainer intentionally still uses materialized, indexed datasets;
indexed streaming training is deferred.

## Evaluation and promotion

Arena schedules paired A-on-RED/A-on-BLUE games for every seed. Record wins,
draws, completion reason, rounds, steps, latency, timeout/error rates, and
artifact/protocol identities. Append observations to a locked checkpoint;
publish canonical evidence only when the declared schedule is complete.

Promotion gates are predeclared. They consume immutable paired evidence and
operational statistics, not ad-hoc reruns. A candidate must satisfy strength,
reliability, and latency gates. The registry stores candidates and champions
under content digests; champion updates are atomic pointers to immutable
objects. Rejections retain their evidence.

`practical_margin` has one canonical scale everywhere in arena configuration:
the candidate-minus-champion score advantage in `[0, 1)`. For paired outcomes,
candidate and champion shares sum to one, so the sequential test derives its
candidate-score threshold explicitly as `0.5 + practical_margin / 2`.
`ArenaConfig` rejects a sequential plan and promotion gate with different
margins rather than evaluating silently mismatched thresholds.

Curricula define ordered stages and advancement requirements. Policy iteration
is generate → replay selection → train → screen → promotion arena → gate →
atomic promote/reject. Every stage journals its input/output digest and can be
resumed without repeating a completed stage.

## Ablation and search boundaries

`AblationPlan` crosses H/H, H/L, L/H, L/L with immediate and bounded-
continuation leaves under both equal-iteration and equal-wall-clock budgets.
Use identical paired seeds. Policy marginal comparisons are H/H→L/H and
H/L→L/L; value marginal comparisons are H/H→H/L and L/H→L/L. Report paired
effect estimates and operational deltas, never only aggregate win rate.

`search_boundaries` records policy-only, immediate consequence, one opposing
response, and two response cycles as explicitly named experimental boundaries.
These are evaluation labels, not permission to call non-ISMCTS algorithms
ISMCTS.

## Required verification

Before retaining evidence: run `tests/automata`, mypy, ruff, black check, and
the full test suite. Also run a tiny generate → train → `SharedEncoderRuntime` load →
H/H,H/L,L/H,L/L construction → paired arena smoke. Archive configuration,
source identity, dataset/split/artifact digests, raw observations, summary, and
gate verdict together.
