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
- `current_owner_id`: owner of the decision currently being scored.

Tree traversal changes owner with `context.for_owner(...)`, producing a new
context. It never changes the root viewer or score perspective. This prevents
allied/opponent decisions from accidentally changing what hidden information
the search may observe or which side a value favors.

## Policy contract

`SearchPolicy.score(context, state, legal_actions)` returns `PolicyScores`:

- actions exactly equal the supplied legal actions, in the same order;
- one finite score per action;
- explicit `LOGITS` or `PROBABILITIES` semantics;
- probabilities are non-negative and sum to one.

The policy cannot add, remove, deduplicate, or reorder legality. Classic
Heuristic policy and a future Learned policy therefore share one checked
boundary. The search may rank a copy for expansion but retains canonical legal
order for result alignment and tie breaking.

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
constants, widening, and exploration remain internal.

## Future PRs

PR2 provides the implementation-specific `SharedEncoderRuntime`, observation encoder,
artifact loading, and batching. They adapt through `LearnedSearchPolicy` and
`LearnedLeafEvaluator` and do not alter root identity, legality, score
semantics, fallback rules, or server lifecycle. Training remains out of scope.
