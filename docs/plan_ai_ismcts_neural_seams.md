# Classic ISMCTS and Learned-model-neutral seams

## Scope

This plan defines the product runtime for classic bots and the stable seams a
later Learned component can implement. Public architecture is
**Learned-model-neutral** and uses **Learned**
(`L`), not a framework or model-family name. PR1 contains no training,
trajectory, model, observation-encoding, or ML framework dependency.

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

The package boundary deliberately excludes `automata.nn`,
`automata.observation`, evaluation/training harnesses, learned policy/value
implementations, Torch, and other ML dependencies.

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

`LeafEvaluator.evaluate(context, state)` returns a finite normalized value in
`[-1, 1]`, positive for `context.perspective_team`. `ValueFnLeafEvaluator`
adapts the older `(state, team) -> value` callable without changing score
semantics.

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
PR1 guarantees the following useful matrix rather than coupling policy and
value into one runtime:

| Environment | Continuation policy | Leaf | Purpose |
|---|---|---|---|
| H | H | H | fully classic baseline |
| H | L | H | isolate Learned action ranking |
| H | H | L | isolate Learned evaluation |
| H | L | L | Learned continuation and evaluation |

The environment remains H in this first product architecture so opponent
behavior is a fixed search assumption. A fake L implementation is used in PR1
contract tests; no Learned implementation ships.

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

A future implementation may provide an implementation-specific
`NeuralRuntime`, observation encoder, artifact loading, batching, and training.
Those modules adapt to `SearchPolicy` and/or `LeafEvaluator`; they do not alter
root identity, legality, score semantics, fallback rules, or server lifecycle.
