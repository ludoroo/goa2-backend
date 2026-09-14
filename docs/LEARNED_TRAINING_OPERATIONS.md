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
split boundaries. `policy_source=HEURISTIC` identifies the exact heuristic
source. Diverse planning rows use
`UNIFORM_PLANNING_SOFT_HEURISTIC` to state that behavior sampled the surfaced
card uniformly while the training distribution came from softened heuristic
scores; their downstream input rows remain `HEURISTIC`. `ISMCTS_VISITS`
continues to identify visit-count targets.

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

### Diverse pilot generation

The locally generated, untracked pilot at `runs/diverse-pilot` used seeds
`[10000, 10040)`, four workers, and this fully explicit resolved configuration:

```bash
PYTHONPATH=src uv run python -m automata.scripts.run_parallel joint-bootstrap \
  --out runs/diverse-pilot/bootstrap.jsonl.zst \
  --checkpoint runs/diverse-pilot/bootstrap.checkpoint.jsonl \
  --seed-start 10000 --seed-end 10040 --workers 4 -- \
  --target-source heuristic --target-recipe softmax-heuristic-card \
  --pilot-mode diverse --planning-behavior uniform \
  --variant-schedule balanced \
  --card-target-temperature 4.0 --card-target-uniform-mass 0.2 \
  --max-steps 10000 --timeout-seconds 300
```

Do not overwrite that preserved artifact merely to reproduce it; select a new
output directory. Its source revision and dirty-tree digest remain part of the
identity, so identical flags on a different tree intentionally produce a new
dataset identity.

The pilot knobs mean:

- `--planning-behavior uniform` samples uniformly from legal hand cards and,
  only after a first card when legal, `FINISH`. It delegates all input and
  downstream decisions to `HeuristicAgent`.
- `--target-recipe softmax-heuristic-card` assigns planning-card targets from
  heuristic card scores. Temperature divides scores before softmax; larger
  values flatten the target. `--card-target-uniform-mass m` then mixes `m` of a
  uniform distribution into that softmax. Input targets remain the one-hot
  exact heuristic choice.
- `--variant-schedule balanced` cycles, by world seed, through QUICK and LONG
  games and the original/swapped Wasp+Xargatha versus Arien+Brogan sides. The
  assignment is anchored to the owned bootstrap seed range, so sharding does
  not change it.
- `--pilot-mode diverse` supplies `uniform`, `balanced`, and
  `softmax-heuristic-card` when those three options are omitted. Supplying them
  explicitly is preferred in operational commands. The temperature and mass
  defaults are `1.0` and `0.1`; the preserved pilot deliberately used `4.0`
  and `0.2`.

Every successful standalone dataset and every successful merged parallel
output gets canonical, atomically replaced provenance at
`<out>.provenance.json`. The sidecar records the half-open seed range, exact
resolved generator configuration, source identity, target behavior/recipe,
and their dataset-row identity digests. Output paths and worker count are
operational rather than generator identity. The default heuristic configuration
and its `generator_config_id` are unchanged; the sidecar is separate from the
JSONL row contract. Standalone shard workers also write sidecars beside their
shard outputs, while the parent writes the authoritative full-range sidecar
beside the merged output.

Source identity format v2 is bound to the exact `HEAD` revision and hashes a
canonical binary diff for tracked changes plus repository-relative paths and
exact bytes (or symlink targets) for every untracked, non-ignored source file.
The repository `runs/` root and each command's explicit output/checkpoint roots
are operational exclusions, so creating or retaining generated artifacts does
not change identity; untracked files elsewhere still count as source. Paths
outside the repository are rejected rather than silently omitted.

This content-sensitive digest intentionally migrates identities produced by the
old status-name-only implementation, even when the visible set of dirty paths
is unchanged. Resume is fail-closed: old checkpoints/fragments will not match a
new generator configuration and must not be relabeled or supplied through the
hidden source-identity overrides. Preserve old evidence in place and start the
v2 run with fresh output/checkpoint destinations (or deliberately archive and
remove the old resume state) before generation.

### Reading pilot headline metrics

Use only multi-candidate rows for headline policy metrics; singleton decisions
have no choice to learn and contribute automatic top-1 accuracy and zero cross
entropy. Keep candidate-family breakdowns because a strong aggregate can still
be dominated by trivial or non-discriminating families.

For a soft target distribution `q` and predicted distribution `p`, raw cross
entropy has the irreducible target-entropy floor `H(q)`. Report
`CE(q, p) - H(q)` (equivalently `KL(q || p)`, up to numerical error) when
comparing runs with different target sharpness. The manifest's policy
`entropy` is predictive entropy `H(p)`, not target entropy `H(q)`, so compute
the latter from the stored `policy_target` values.

Interpret the value head against chance explicitly. A constant 50% win
probability has log-loss `ln(2) ~= 0.693`, Brier score `0.25`, no saturation,
and 50% accuracy on balanced decisive outcomes; it can also have deceptively
small calibration error. Draws map to a 50% value target. Report outcome balance
and per-mode results rather than treating calibration alone as learned value
signal.

Self-play uses a content-addressed champion artifact and the same game and row
contracts. Seeds belong to named, disjoint ranges. A retry must preserve world
seed, agent seed, configuration identity, source revision, dirty-tree digest,
and parent artifact digest. Never merge partial games.

### Official 64-game L/H generation

Run the four workers for the official `[40000, 40064)` generation with this
exact invocation from the repository root. Constant temperature `0.5` is
intentional: the matched pilot retained useful late-game exploration and
finished faster than the tested round-decay schedule.

```bash
mkdir -p runs/self-play-depth8-balanced-adaptive-t05-v1-64/{output,checkpoints,logs}
for worker_id in 0 1 2 3; do
  PYTHONPATH=src uv run python -m automata.scripts.run_self_play \
    --artifact runs/diverse-pilot-v1-action-aware/model \
    --parent-model-digest 8da2e32f3d687bdb6c324f94153036937cbe8e1eb560a07afc65fc25afba0ec5 \
    --parent-generation 1 --observation-schema-version 3 \
    --generation-id self-play-depth8-balanced-adaptive-t05-v1-64 \
    --seed-start 40000 --seed-end 40064 --worker-id "${worker_id}" \
    --output-dir runs/self-play-depth8-balanced-adaptive-t05-v1-64/output \
    --checkpoint-dir runs/self-play-depth8-balanced-adaptive-t05-v1-64/checkpoints \
    --strategy-preset learned-policy-heuristic-value \
    --search-config '{"iterations":8,"uct_c":1.4,"cutoff_limit":1,"cutoff_unit":"DECISIONS","max_advance_transitions":1024,"max_forced_decisions":256,"leaf_mode":"IMMEDIATE","widening_c":2.0,"widening_alpha":0.5,"root_widening_c":1.0,"root_widening_alpha":0.5,"adaptive_hex_root_schedule_version":1,"use_prior":true,"puct_c":0.0,"root_puct_c":1.5}' \
    --variant-schedule balanced \
    --visit-temperature 0.5 --visit-temperature-schedule constant \
    --random-stream-namespace self-play-depth8-adaptive-t05-v1-64 \
    --decision-timeout-seconds 15 \
    --source-config '{"run":"official-64-game-lh-v1"}' \
    --max-steps 10000 --timeout-seconds 3600 --no-progress \
    >"runs/self-play-depth8-balanced-adaptive-t05-v1-64/logs/worker-${worker_id}.stdout" \
    2>"runs/self-play-depth8-balanced-adaptive-t05-v1-64/logs/worker-${worker_id}.stderr" &
done
wait
```

After all four workers have completed all 64 seeds, publish the training
aggregate with the bounded-memory reconciler (not the in-memory
`generation_pipeline` reconciliation path):

```bash
PYTHONPATH=src uv run python -m automata.scripts.reconcile_self_play \
  --output-dir runs/self-play-depth8-balanced-adaptive-t05-v1-64/output \
  --checkpoint-dir runs/self-play-depth8-balanced-adaptive-t05-v1-64/checkpoints \
  --seed-start 40000 --seed-end 40064 --workers 4 \
  --out runs/self-play-depth8-balanced-adaptive-t05-v1-64/generation.jsonl.zst
```

The default provenance destination is
`<out>.provenance.json`; use `--provenance PATH` only when an explicit sidecar
location is operationally required. The command takes a blocking advisory lock,
requires exact round-robin worker/seed coverage, verifies canonical checkpoints
and every checksummed fragment, and streams canonical rows in stable seed/game/
decision order. It atomically publishes the compressed aggregate and canonical
sidecar, including exact compressed and uncompressed digests plus every source
checkpoint and fragment identity. A matching publication is validated and
returned idempotently. Any mismatch, unsafe/symlinked path, incomplete game, or
corruption fails closed without altering worker fragments or overwriting an
existing artifact.

The tracked preset is the L/H matrix cell: it builds a fresh
`HeuristicAgent(agent_seed)` for environment/continuation play and passes the
loaded `--artifact` runtime to `build_learned_ismcts`. Its `--search-config`
input is strict: unknown fields, wrong JSON types, and invalid `LeafMode` or
`CutoffUnit` values fail before a worker starts. Do not provide `seed` in that
JSON; the per-world/per-side `agent_seed` is authoritative. The command records
the preset name, matrix cell, and every resolved `SearchConfig` value (including
the `agent_seed` seed source) in source and generator identity. A custom dotted
`--strategy-factory module:callable` remains supported for experiments, but it
is mutually exclusive with `--strategy-preset`, and one of them is required for
normal CLI use.

`run_self_play --variant-schedule fixed` is the default and preserves the Phase-0
QUICK Wasp+Xargatha versus Arien+Brogan setup for every seed. The opt-in
`--variant-schedule balanced` cycles through QUICK original sides, QUICK swapped
sides, LONG original sides, and LONG swapped sides. The cycle offset is derived
from the globally owned training seed-range start (20,000), not a worker's shard
or requested subrange, so all four workers and resumed/subrange runs assign the
same variant to a world seed. `source_config` records the schedule name, anchor,
and fully expanded ordered variants; these fields participate in generator,
worker, game, checkpoint, and fragment identity. A balanced worker also requires
the champion artifact to declare support for both QUICK and LONG plus all four
heroes before play starts; the existing schema, map, adapter, and integrity
compatibility checks remain mandatory.

`run_self_play --random-stream-namespace NAME` opts an experiment family into
comparable per-side agent RNG streams. `NAME` must contain a non-whitespace
character and be at most 128 characters. When set, the agent seed is the
versioned digest of only the namespace, world seed, and side; changing search
knobs such as visit temperature or an adaptive schedule therefore does not also
change the random stream. Worker count, shard assignment, resume boundaries,
and requested subranges likewise do not affect that seed. Use the same namespace
only for arms intended to share randomness, and a different namespace for an
independent experiment. The namespace is recorded in `source_config` and in the
generator identity, so generator, worker, game, checkpoint, and fragment
identities remain distinct and content-addressed even when experiment arms share
agent streams. Omitting the flag preserves the exact legacy generator-config-
bound seed derivation and identities; `None` is not added to `source_config` or
the identity payload.

`run_self_play --visit-temperature T` controls behavior-action sampling without
changing the stored policy target. `T=0` (the default) preserves ISMCTS's robust
child/argmax action. The default `--visit-temperature-schedule constant` uses
`T` for every round and is behaviorally identical to the prior constant path;
omitting the schedule flag and passing `constant` explicitly resolve to the
same identity. The opt-in
`--visit-temperature-schedule round-decay-v1` resolves temperature from the
public pre-decision `state.round`: rounds 1–4 use `T`, rounds 5–8 use `T / 2`,
and round 9 onward uses zero/robust child. For the planned schedule use
`--visit-temperature 0.5 --visit-temperature-schedule round-decay-v1`, yielding
0.5, 0.25, then 0.0 at those boundaries.

For an effective temperature above zero, self-play samples only positive-visit
root actions with weights `visits ** (1 / T)`, using the deterministic per-side
agent RNG stream. Resolving a schedule does not consume that stream, zero
sampling temperature does not consume it, and singleton roots remain forced.
The schedule name is recorded in resolved `source_config`
and participates in generator identity; generator identity flows into worker,
game, checkpoint, and fragment identities. The
recorded `policy_target` and `action_stats.improved_probability` stay the
untempered normalized visit distribution, while exactly the sampled action has
`action_stats.selected=true`. Arena and server agents use `ISMCTSStrategy`
directly and therefore remain argmax-based. A zero-visit root is valid only for
a singleton forced action; other zero-visit or misaligned results fail closed.

`run_self_play --decision-timeout-seconds N` optionally enables a self-play-only
hard deadline around each source strategy decision. It is disabled when omitted,
must be finite and positive when supplied, and participates in generator/game
identity. On main-thread POSIX workers the deadline composes with the whole-game
`--timeout-seconds` budget without resetting that budget. A decision timeout
discards the entire game's fragment, publishes no checkpoint, and moves to the
next assigned game. JSONL telemetry emits bounded `decision_started`,
`decision_completed`, and `decision_timeout` events with decision identity,
round/phase, public request type/id, legal-family counts, elapsed time, and
aggregate visit coverage when available; it never includes candidate values,
request prompts/options, or state snapshots. Progress events retain their
current round and step count. Arena and server decision behavior is unchanged.

## Training

Train the bootstrap dataset with:

```bash
PYTHONPATH=src uv run python -m automata.training.trainer \
  --dataset runs/bootstrap.jsonl.zst --split-manifest runs/split.json \
  --checkpoint runs/training.checkpoint.pt --run-manifest runs/training.json \
  --artifact runs/model --seed 20000
```

For a distinct regularization pilot on a mixed QUICK/LONG dataset, hold out LONG
while keeping all maps represented. Use separate split, checkpoint, manifest,
and artifact paths because these settings are part of run identity:

```bash
PYTHONPATH=src uv run python -m automata.training.trainer \
  --dataset runs/pilot-quick-long.jsonl.zst \
  --split-manifest runs/pilot-mode-split.json \
  --checkpoint runs/pilot-training.checkpoint.pt \
  --run-manifest runs/pilot-training.json --artifact runs/pilot-model \
  --seed 21000 --dropout 0.1 --entropy-weight 0.01 --l2-weight 0.0001 \
  --value-weight 1.0 --validation-fraction 0.2 --holdout-game-mode LONG
```

`--holdout-game-mode` is repeatable when more than one mode must be reserved.
Omitting all new tuning options preserves the prior trainer defaults.

`automata.training.trainer.train_joint(JointTrainingConfig(...))` creates or validates a grouped
split manifest, trains both heads, checkpoints optimizer/model/RNG/progress,
and resumes only when the complete configuration and input identities match.
Success atomically exports a PR2-compatible model directory. Validate every
result with `SharedEncoderRuntime.from_artifact(...).evaluate(...)` before arena use.
Run manifests are canonical and immutable evidence; failed or interrupted runs
must not publish an artifact.

Training datasets use transparent Zstandard compression when their path ends in
`.jsonl.zst`; legacy `.jsonl` datasets remain readable and writable. Dataset
digests are SHA-256 over the exact validated, uncompressed JSONL rows, so
compression settings and container bytes do not change dataset identity. JSON
internal whitespace and key ordering are accepted but remain part of that exact
identity; surrounding row whitespace and CRLF are rejected. The source SHA-256
and typed validation make a costly canonical round-trip unnecessary. These
training files are trusted generator output: external producers should use
`write_joint_dataset`, because duplicate JSON object keys otherwise follow
pydantic-core's last-key-wins parsing. Checkpoint and audit JSONL files remain uncompressed. `iter_joint_dataset` strictly validates
rows and cross-row invariants incrementally; consumers must exhaust it because a
truncation or end-of-stream invariant can be reported after earlier rows were
yielded.

The trainer builds a disposable per-game index at `<dataset>.index` on first use.
The index is atomically published, bound to the exact source bytes and complete
tensor schema identity (`id`, version, and digest), and retains compressed audit
fragments, safe pre-collated tensor chunks, and compact split, scope, provenance,
row-offset, metric, and digest metadata. Tensor schema v1 expands ACTION/OPTION
IDs into 64-column domain-separated signed character n-gram hashes; exact IDs
remain non-tensor batch metadata. Cached indexes without the complete matching
schema identity and digest are rejected and atomically rebuilt from source.
Incompatible index staging fragments are discarded and regenerated. Training
checkpoints and immutable model artifacts are also identity-bound: discard stale
ones and restart training/export with fresh destinations rather than editing
schema files, manifests, or checkpoint metadata to force compatibility.
Schema vectorization and tensor collation happen once during indexing; training
and metric evaluation load validated tensor-only files with
`torch.load(..., weights_only=True)`. `--dataset-index` selects another cache location,
`--index-workers` parallelizes independent per-game tensorization after the
strict source scan, and `--decisions-per-chunk` controls the memory/throughput
tradeoff. Each index worker may use roughly 0.75 GiB on the phase-0 dataset. The chunk size
is part of checkpoint identity and can change bit-exact floating-point results,
so resumed runs must keep it unchanged. Index paths and compression do not affect
dataset or split identity.

The source scan remains all-or-nothing. After that scan is complete, the builder
durably checkpoints each tensorized game in `.<index>.staging`. An interrupted
or failed retry validates the source, schema, fragments, checkpoint, and chunk
hashes, then schedules only unfinished games. The final index remains invisible
until atomic publication, and any previously published index remains untouched.

Budget space for compressed audit fragments and tensor chunks in addition to the
source dataset (about 6 GiB for the phase-0 bootstrap at chunk size 32).
Rebuilding an existing index temporarily requires space for both the old and
replacement indexes. Do not remove `.<index>.staging` while an index builder is
running; it contains resumable phase-two work and is discarded automatically
when stale or after successful publication. A persistent `.<index>.lock` file
serializes builders for the same destination; the file itself is harmless when
no process holds its advisory lock. The index destination must be a real
directory path rather than a symlink; select storage elsewhere with an explicit
`--dataset-index` path.

## Evaluation and promotion

Arena schedules paired A-on-RED/A-on-BLUE games for every seed. Record wins,
draws, completion reason, rounds, steps, latency, timeout/error rates, and
artifact/protocol identities. Append observations to a locked checkpoint;
publish canonical evidence only when the declared schedule is complete.

Use the tracked L/H candidate-versus-parent command for production arena
workers. All matchup inputs are required; seed ranges are half-open. The
command verifies both artifacts against their pinned model digests, current
runtime/schema versions, the map inferred from `--map-path`, game mode, hero
adapters, and the complete scheduled roster before it starts. Each case is
spawn-isolated under the whole-game timeout and writes no training rows. A
successful run prints one canonical summary JSON object containing
candidate/parent wins, draws, completion-reason counts, rounds, steps, and the
95% Wilson interval:

```bash
PYTHONPATH=src uv run python -m automata.scripts.run_learned_arena \
  --candidate-artifact runs/candidate-model --candidate-digest "$CANDIDATE_DIGEST" \
  --parent-artifact runs/parent-model --parent-digest "$PARENT_DIGEST" \
  --checkpoint runs/arena/stratum-0.jsonl \
  --map-path src/goa2/data/maps/forgotten_island.json --game-type QUICK \
  --red-heroes Wasp Xargatha --blue-heroes Arien Brogan \
  --seed-start 1020000 --seed-end 1020100 \
  --search-config '{"iterations":8,"cutoff_limit":1,"cutoff_unit":"DECISIONS","leaf_mode":"IMMEDIATE","max_advance_transitions":1024,"max_forced_decisions":256,"widening_c":2.0,"widening_alpha":0.5,"use_prior":true,"uct_c":1.4,"puct_c":0.0,"root_puct_c":1.5,"root_widening_c":1.0,"root_widening_alpha":0.5}' \
  --random-stream-namespace promotion-generation-12 \
  --max-steps 10000 --timeout-seconds 3600
```

For four parallel independent strata, launch four processes with disjoint
`--seed-start/--seed-end` ranges and distinct checkpoint paths. Use the same
random-stream namespace across all four and any candidate/parent swap intended
to reuse side-specific randomness. The search seed is derived only from that
namespace, world seed, and RED/BLUE side; artifact assignment is deliberately
excluded. Re-running an identical command resumes missing cases from its
checkpoint. Source identity is collected automatically from the current tree
with `runs/` excluded; changing source, artifacts, matchup, search settings,
limits, timeout, or namespace invalidates stale checkpoint rows.

The `--search-config` object uses the same strict, complete public L/H preset
parser as tracked self-play. Unknown keys, coercible-but-wrong JSON types,
invalid enums, and a caller-supplied `seed` fail before artifact loading.

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
