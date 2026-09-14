# AI search and learned runtime architecture

## Implementation status

- **PR1 complete:** classic runtime, ISMCTS contracts, fallback wrappers, and
  bounded server lifecycle.
- **PR2 implemented:** information-safe observations; architecture-neutral
  `automata.models.contracts` and shared-encoder tensor schemas;
  joint model, batching, artifact, `SharedEncoderRuntime`, and serving cache;
  `LearnedSearchPolicy` and `LearnedLeafEvaluator`; independent
  `policy_source`/`value_source` server composition.
- **PR3/PR4 preserved:** learned runtime contracts and the shared-encoder
  implementation remain independent from offline orchestration.
- **PR6 boundary:** `automata.training` owns generation, datasets, replay,
  training, curriculum, experiment declarations, registry, and policy iteration.
  `automata.evaluation` owns arenas, statistics, protocols, promotion gates,
  ablations, and matchup evidence. The neutral offline `automata.harness` owns
  the shared headless game runner and trajectory recorders.

## Scope

This plan defines the product runtime for classic bots and the stable seams a
later Learned component can implement. Public architecture is
**Learned-model-neutral** and uses **Learned**
(`L`), not a framework or model-family name. PR1 contains no training,
trajectory, model, observation-encoding, or ML framework dependency.

## Module boundaries

| Boundary | Contract/protocol home | Concrete implementation home |
|---|---|---|
| Agents | `automata.agents.contracts` | `random_agent`, `heuristic_agent`, `ismcts_agent` |
| Learned models | `automata.models.contracts.{observation,candidates,inference,artifacts,compatibility,serialization}` | `automata.models.shared_encoder` |
| Shared-encoder artifacts | neutral errors, scope, and runtime requirements in `automata.models.contracts.artifacts` | manifest and IO in `automata.models.shared_encoder.artifacts` |
| Tensor schema/vectorizer | — | `automata.models.shared_encoder.schema.feature_schema` |
| Observations | `automata.decision.DecisionDescriptor` | `automata.observation.graph.encoder`, `automata.observation.decision_encoder`, `automata.observation.projector` |
| Hero adapters | `automata.observation.hero_adapters.protocol` | `automata.observation.hero_adapters.registry` |
| Search components | `automata.search.contracts` | `automata.search.heuristic`, `fallback`, `learned`, `ismcts` |
| Offline harness | — | `automata.harness.game_runner`, `automata.harness.trajectory` |
| Offline training | `automata.training.contracts` | `automata.training` |
| Evaluation | `automata.evaluation.protocol` | `arena`, `arena_stats`, `promotion_gates`, `ablation`, `matchup` |

Artifact errors, scope, and runtime requirements are model contracts. The concrete
manifest, tensor/file inventory, and artifact export/loading are specific to the
shared-encoder implementation. Observation code consumes the neutral decision descriptor and
does not import the concrete ISMCTS implementation.

## Product closure in PR1

- `automata.agents`: the `Agent` protocol plus Random and Heuristic agents.
- `automata.runtime`: isolated clone, information-safe determinization, effect
  registration, and the decision driver.
- `automata.search`: classic single-perspective ISMCTS with strict
  `RootTarget` and canonical legal-root validation.
- `goa2.server`: persisted Random/Heuristic/ISMCTS bot specs, bounded compute,
  one coordinator per game, and the ordinary save/log/replay/clock/broadcast
  mutation paths.
- Permanent public card-reveal knowledge sufficient to sample hidden upgraded
  loadouts without inspecting an opponent's private cards.

In PR1, the package boundary deliberately excluded `automata.models`,
`automata.observation`, learned policy/value implementations, Torch, and other
ML dependencies. PR2 adds those runtime pieces while still excluding
offline evaluation, training, or harness code. Product runtime and search never
import `automata.training`, `automata.evaluation`, or `automata.harness`; the
offline packages depend inward on stable product and model contracts. Training
and evaluation may use the neutral harness. Their only direct cross-package
composition is policy iteration importing evaluation; evaluation does not
import training.

## Stable search context

`SearchContext` is immutable. It has:

- `root_viewer_id`: fixed for an entire search and used as the information-set
  viewer;
- `perspective_team`: fixed score perspective;
- `current_owner_id`: concrete hero owner of the decision currently being scored;
- `current_decision`: the exact surfaced decision, including descendant input
  requests that may no longer match `state.input_stack`.

Tree traversal carries both fields with `context.for_decision(...)`, producing a
new context. Team-scoped requests retain the last concrete eligible hero rather
than replacing the owner with a synthetic `team:*` identifier. Traversal never
changes the root viewer or score perspective. This prevents allied/opponent
decisions from accidentally changing what hidden information the search may
observe or which side a value favors.

## Policy contract

`SearchPolicy.score(context, state, legal_actions)` returns `PolicyScores`:

- actions exactly equal the supplied legal actions, in the same order;
- one finite score per action;
- explicit `LOGITS` or `PROBABILITIES` semantics;
- probabilities are non-negative and sum to one;
- model-neutral `PRIMARY` / `FALLBACK` source metadata.

The policy cannot add, remove, deduplicate, or reorder caller legality. A
Learned adapter builds its observation in the decision's canonical candidate
order, validates runtime output in that order, then realigns logits to the
caller's order before returning. Search may rank a copy for expansion but
retains caller order for result alignment and tie breaking.

Tensor feature schema v1 gives non-graph `ACTION` and `OPTION` candidates model
identity without introducing a closed vocabulary. Their exact typed IDs stay
Python-side for output alignment; tensors contain a deterministic, fixed-width,
L2-normalized signed character n-gram hash. ACTION and OPTION use distinct hash
namespaces, so the same text in the two domains cannot be silently conflated.
The schema artifact declares the hash algorithm/version, 64-column dimension,
n-gram range, Unicode unit, boundary/framing rules, digest/index/sign semantics,
and normalization; all declarations participate in the schema digest. Unknown
future IDs therefore vectorize without schema edits. Cached tensors and model
artifacts whose schema identity or digest differs fail validation rather than
being reinterpreted.

Treat all learned-runtime outputs as bound to that complete identity. A stale
index is discarded and rebuilt from its source dataset; incompatible staging
fragments are discarded and regenerated by the index builder. A stale training
checkpoint or immutable model artifact must be discarded and training/export
restarted at a fresh destination. Never relabel or hand-edit a schema, manifest,
checkpoint, or artifact to make it appear compatible.

## Leaf contract

`LeafEvaluator.evaluate(context, state)` returns a `LeafEvaluation` containing
a finite normalized value in `[-1, 1]`, positive for
`context.perspective_team`. The Pydantic model validates this contract at every
evaluator boundary.

`LeafMode` is explicit:

- `IMMEDIATE`: evaluate the state reached by expansion;
- `BOUNDED_CONTINUATION`: advance with `continuation_policy` to the configured
  round bound, then evaluate.

`environment_policy` and `continuation_policy` are separate. The environment
policy resolves decisions outside the controlled information set. The
continuation policy chooses controlled actions only during bounded leaf
continuation. They may be configured independently.

## Composition matrix

`H` means the classic Heuristic component and `L` means a Learned component.
The server exposes the following matrix without coupling policy and value. In
L/L, both adapters share one `SharedEncoderRuntime`:

| `policy_source` | `value_source` | Purpose |
|---|---|---|
| H | H | fully classic baseline |
| L | H | Learned expansion/ranking with heuristic leaf |
| H | L | heuristic policy with Learned leaf |
| L | L | shared-runtime Learned policy and value |

The environment remains H in this first product architecture so opponent
behavior is a fixed search assumption. PR1 used a fake L implementation in
contract tests; PR2 ships the Learned runtime adapters described above.

## Availability and fallback

Optional components declare only two recoverable failures:

- `ComponentUnavailableError`: component/artifact/runtime is unavailable;
- `ComponentInferenceError`: an otherwise available component cannot score the
  request.

`FallbackSearchPolicy` and `FallbackLeafEvaluator` catch exactly those errors.
Programming errors, invalid candidate alignment, non-finite values, and other
exceptions propagate. This prevents fallback from concealing correctness bugs.
A policy fallback marks its returned scores as `FALLBACK`; root-only learned
PUCT and widening overrides are then disabled for that decision, while the
fallback scores may still order expansion under the classic schedule.

Server-level bounded compute separately protects the whole expensive agent
call with a process-wide concurrency limit, queue timeout, decision timeout,
output revalidation, and a cached Heuristic fallback.

## Statistics

`RunningStatistics` is a model-independent online accumulator for count, mean,
and population variance. Search statistics remain in the classic package and
do not depend on observation or inference code.

## Information safety

Determinization clones the authoritative state and samples only facts unknown
to the fixed root viewer. The viewer's own commitment is retained. Opponent
facedown commitments and loadout hypotheses are sampled from legal candidates
consistent with permanent public reveals, public item aggregates, and public
card lifecycle. Revelation, public discard, direct hand reveal, and card-guess
resolution are the authoritative reveal hooks. Unsupported/nonstandard
loadouts fail closed rather than reading private card identity.

## Server lifecycle

`BotSpec` is persisted; live agents, tasks, fallback agents, and futures are
runtime-only. The coordinator:

1. clones state and result under the game lock;
2. computes outside locks;
3. revalidates ownership, request identity, and legality on live state;
4. applies one decision through `GameSession` under the established lock order;
5. reuses ordinary clock finalization, save, log, replay, and scoped broadcast;
6. schedules idempotently from create/restore, REST, WebSocket, and timeout
   completion paths.

ISMCTS exposes only bounded iterations and wall-clock timeout publicly. Search
constants, widening, and exploration remain internal. The opt-in learned root
schedule constants live in `automata.search.config` for future consumers, but
self-play and evaluation continue to use their existing configuration until
cross-game evidence is available.

`SearchResult.root_action_diagnostics` stays in caller legal order. Priors are
the normalized probabilities from the policy actually used and always drive
expansion ordering; they affect selection only when effective PUCT is positive.
Means and variances use search reward `[0, 1]`, after mapping leaf values from
`[-1, 1]`. Zero-visit actions report zero mean/variance as unvisited sentinels,
not neutral estimates.

### Adaptive broad-HEX root schedule

`SearchConfig.adaptive_hex_root_schedule_version=1` is an internal, versioned
opt-in; `None` preserves all classic defaults. It is deliberately absent from
client `SearchSettings`. Version 1 applies only when a root has more than eight
legal actions and every key is a canonical HEX key, apart from an optional
`SKIP`. With `K` legal actions it sets:

- coverage `M = min(K, 12, max(4, ceil(sqrt(K))))`;
- effective iterations `I = max(config.iterations, 2 * M)`.

The first `M` root visits force distinct actions in deterministic descending
prior order (stable caller order for ties, and caller order when no prior is
available). This also applies when the learned policy falls back: the fallback
prior controls ordering while the fallback's classic PUCT/widening behavior is
retained. After coverage, normal PUCT and progressive widening resume.

The schedule never truncates legality. Diagnostics and training targets retain
every caller candidate; unvisited candidates have zero visits and therefore
zero improved-policy mass. `SearchResult` reports `requested_iterations`,
`effective_iterations`, and `root_coverage_target` for self-play telemetry.

## Future PRs

PR2 provides the implementation-specific `SharedEncoderRuntime`, observation encoder,
artifact loading, and batching. They adapt through `LearnedSearchPolicy` and
`LearnedLeafEvaluator` and do not alter root identity, legality, score
semantics, fallback rules, or server lifecycle. Training remains out of scope.
