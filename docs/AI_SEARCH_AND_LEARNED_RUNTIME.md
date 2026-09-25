# AI search and learned runtime architecture

> The runtime/model foundations below are retained. The experimental learning
> and leaf-evaluation contract is being reset; see
> [AI_LEARNING_CONTRACT.md](AI_LEARNING_CONTRACT.md) for the target and
> [AI_EXPERIMENT_JOURNAL.md](AI_EXPERIMENT_JOURNAL.md) for implementation status.
> Existing search/training behavior has not yet been switched to that contract.

## Implementation status

- **Classic foundation:** runtime, ISMCTS contracts, fallback wrappers, and
  bounded server lifecycle.
- **Learned foundation:** information-safe observations; architecture-neutral
  `automata.models.contracts` and shared-encoder tensor schemas;
  joint model, batching, artifact, `SharedEncoderRuntime`, and serving cache;
  `LearnedSearchPolicy` and `LearnedLeafEvaluator`; independent
  `policy_source`/`value_source` server composition.
- **Offline boundary:** `automata.training` owns generation, datasets, replay,
  training, experiment declarations, and the registry. The unused curriculum
  and callback-only pipeline/iteration wrappers have been retired; a complete
  executable Gen1 loop is not implemented yet.
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

## Product runtime boundary

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

Classic runtime remains usable without Torch. Learned composition adds model
and observation components without importing offline orchestration. Product
runtime and search never import `automata.training`, `automata.evaluation`, or
`automata.harness`; the offline packages depend inward on product and model
contracts. Training and evaluation may use the neutral harness. There is no
cross-package policy-iteration wrapper; evaluation does not import training.

## Stable search context

`SearchContext` is immutable. It has:

- `root_viewer_id`: fixed for an entire search and used as the information-set
  viewer;
- `perspective_team`: fixed score perspective;
- `current_owner_id`: concrete hero representing the current decision owner;
- `decision`: the live `DecisionDescriptor`, including the exact pending
  `InputRequest` or planning hero and second-card eligibility.

Tree traversal uses `context.for_decision(decision, owner_id=...)`, producing a
new context with the live decision and its owner. It never reconstructs pending
input from `state.input_stack`, which may already be empty at a session boundary.
Team-scoped descendants prefer the most recent eligible concrete decision owner,
using the root viewer only as a fallback when eligible. Root requests still use
the coordinator-selected owner. This deliberately preserves the retained search's
owner continuity; it does not widen the root viewer's access. Traversal never
changes the root viewer or score perspective.
This prevents allied/opponent decisions from accidentally changing what hidden
information the search may observe or which side a value favors.

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

Tensor feature schema v2 consumes decision-observation v4. It adds exactly one
shared decision-context row (`decision_kind`, typed input request type,
`can_skip`, and normalized semantic role) to the state trunk used by both policy
and value. It also gives non-graph `ACTION` and `OPTION` candidates model
identity without introducing a closed vocabulary. Their exact typed IDs stay
Python-side for output alignment; tensors contain a deterministic, fixed-width,
L2-normalized signed character n-gram hash. ACTION and OPTION use distinct hash
namespaces, so the same text in the two domains cannot be silently conflated.
The schema artifact declares the hash algorithm/version, 64-column dimension,
n-gram range, Unicode unit, boundary/framing rules, digest/index/sign semantics,
and normalization; all declarations participate in the schema digest. Unknown
future IDs therefore vectorize without schema edits. Request-type and semantic-role
categorical vocabularies are different: v2 freezes them as literal snapshots,
not comprehensions over the live enums, so adding an enum cannot silently mutate
a released digest. When either enum gains a model-relevant value, add and test
its typed classifier first; then create a new tensor-schema ID/version with a
new copied vocabulary snapshot, bump model/runtime and dataset compatibility as
needed, regenerate golden schema/output coverage, and retrain. Do not edit the
v2 snapshot in place. Cached tensors and model artifacts whose schema identity
or digest differs fail validation rather than being reinterpreted.

Treat all learned-runtime outputs as bound to that complete identity. Current
training accepts only joint-dataset v2 / decision-observation v4 / tensor-schema
v2 and emits model/runtime generation v2. A stale
index is discarded and rebuilt from its source dataset; incompatible staging
fragments are discarded and regenerated by the index builder. A stale training
checkpoint or immutable model artifact must be discarded and training/export
restarted at a fresh destination. Never relabel or hand-edit a schema, manifest,
checkpoint, or artifact to make it appear compatible.

Only the current native observation/tensor/model/runtime format is supported.
The observation-v3/tensor-v1 execution path, old digest-loading exceptions, and
self-play/arena bridge flags have been removed. Unsupported versions fail closed;
artifact integrity, scope, and candidate validation have not been relaxed.
Historical checkpoints remain evidence, not executable dependencies of Gen1.

## Leaf contract

`LeafEvaluator.evaluate(context, state)` returns a `LeafEvaluation` containing
a finite normalized value in `[-1, 1]`, positive for
`context.perspective_team`. The Pydantic model validates this contract at every
evaluator boundary.

`LeafMode` is explicit:

- `IMMEDIATE`: evaluate the state reached by expansion. An evaluator may also
  implement `ImmediateEdgeLeafEvaluator` to compare the parent action with its
  public immediate consequence. The heuristic recipe `public-consequence-v4`
  captures an immutable before-vector from the fixed root viewer/perspective,
  then adds independently normalized public consequence deltas only on newly
  expanded nonterminal edges; exact terminal rewards bypass it.
- `IMMEDIATE_ACTION`: at an `INPUT` root, capture the request owner, concrete
  resolution actor, and round before applying each root edge, then use
  `continuation_policy` only for follow-up inputs owned by that same request
  owner in the same resolution action and round. Foreign opponent reactions are
  resolved by `environment_policy`; team/ally-owned inputs stop the
  continuation. Team-scoped roots are anchored to the coordinator-selected
  concrete hero, so a later team-scoped prompt also stops. A same-owner
  `CHOOSE_RESPAWN_HEX` continuation is completed through hero placement and
  then stops before `ResolveCardStep`. Other action flows cut off before
  confirmation, another action choice, tie breaking, resolution-owner
  change, turn finalization, or planning. `CARD` planning roots retain normal
  `IMMEDIATE` expansion and tree descent rather than stopping at an unresolved
  `ResolveCardStep`. This resolves compound movement, targeting, and combat
  consequences without rolling into a later action.
- `STABLE_TURN`: experimental actor-bound `INPUT` horizon. Search separately
  captures the fixed root viewer/perspective and the enclosing
  `resolution_owner_id`, phase, round, and turn. The root edge and owned
  follow-up prompts use `continuation_policy` until that actor finalizes;
  foreign prompts use `environment_policy`. After finalization clears
  `resolution_owner_id`, every minion-return, lane-push, and tie decision uses
  the environment policy even when addressed to the controlled team. Search
  executes `ReturnMinionToZoneStep`, `CheckLanePushStep`, and
  `FindNextActorStep`, then stops before the selected next actor's independent
  `RespawnHeroStep`/`ResolveCardStep`. If no actor remains, it stops on the
  resulting planning/cleanup transition after required finishing work. At the
  final resolution turn it stops on entry to `CLEANUP`; it does not execute
  `EndPhaseStep`, so an outcome decided only by end-phase processing remains a
  nonterminal leaf. Any game-over outcome encountered before that boundary
  bypasses leaf evaluation. Actorless or interphase roots fall back to an
  immediate horizon rather than drifting across turns.
- `BOUNDED_CONTINUATION`: advance with `continuation_policy` to the configured
  round bound, then evaluate. Immediate-edge shaping is deliberately excluded.

The version-4 heuristic measures public life/gold/on-board minion material;
perspective-signed public hand sizes for every hero; public numeric stat
modifiers and markers with known team/sign; non-consumable attack setup for the
public current actor/card; and directional lane/battle-zone control. Hidden card
identity never enters the vector. Defense has no transient context component:
card expenditure and realized defeat/survival consequences provide its value.
Attack setup snapshots explicit source/target pairs and compares only pairs that
remain present after the edge, so killing a target or losing the attacker cannot
create geometry value. Its small range/reach terms reuse canonical action,
target, immunity, LOS, topology, and stat helpers without target-material or
straight-line preferences, and handle multi-piece actors through their public
pieces. Card-resource, modifier, and attack-setup deltas freeze when an edge
crosses a turn or round boundary, preventing refill/expiry artifacts. This
established v4 rule also applies to a final-actor `STABLE_TURN` edge: those
three components stay neutral across the transition while material and
objectives remain measurable. Unknown,
non-numeric, or relation-ambiguous effects remain neutral. Each component delta
is normalized separately, weighted, and clamped to the existing +/-5 edge-unit
envelope before the atanh/tanh base combination. `BOUNDED_CONTINUATION` sees a
longer realized horizon but deliberately receives no immediate-edge shaping.
Consumers must not compare those horizons as if they were identical value
recipes.

At HEX+`SKIP` roots, `IMMEDIATE_ACTION` and `STABLE_TURN` use the learned policy to order which
concrete hex is expanded first, guarantees one real comparison with `SKIP`, and
then uses UCB1 rather than PUCT for value-led selection. This prevents a
previously learned SKIP prior from overwhelming better observed Q. RESPawn/PASS
and action/HOLD retain learned PUCT after their guaranteed comparison. None of
these rules rewrites logits, priors, legality, visits, or stored diagnostics,
and harmful concrete choices can still lose to the no-op. At a boundary leaf,
search supplies a private synthetic one-option `CONFIRM` decision context for
value encoding; it never exposes a foreign reaction/cleanup request or sends
the internal `BOUNDARY` sentinel to a learned encoder. Stable-turn context keeps
`root_viewer_id` and perspective fixed but assigns `current_owner_id` to the
original enclosing actor (for example, a defender-viewed reaction root still
ends on the attacker's turn boundary). There is no action-label opportunity
proxy at complete or interrupted boundaries: equal measured ATTACK, SKILL,
MOVEMENT, and HOLD outcomes tie. The executable RESPAWN continuation remains a
narrow special case because classic immediate expansion stops before placement.

The heuristic recipe is global whenever heuristic value is active; it is not a
client-tunable switch. Provenance binds both `value_recipe` and `leaf_mode`. A
learned value is never shaped. If learned inference raises a declared
recoverable failure, its heuristic fallback uses the contextual recipe for the
already-expanded edge.

`environment_policy` and `continuation_policy` are separate. The environment
policy resolves decisions outside the controlled information set and remains
heuristic. The continuation policy chooses controlled actions only during leaf
continuation. Heuristic-policy configurations preserve the historical Agent
choices through `AgentContinuationPolicy` when those choices are canonically
legal; a stale or out-of-set custom-Agent result now fails closed rather than
being submitted to the engine. Learned-policy configurations use
`learned-argmax-v1`: the same root `SearchPolicy` scores canonical legal keys,
then `ArgmaxContinuationPolicy` chooses the first maximum in canonical order.
It never routes a learned continuation through `Agent.choose_input`; applying
the canonical key through the simulator also preserves synthetic `SKIP`.

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

The environment remains H in this first product architecture so opponent and
foreign behavior is a fixed search assumption. When policy is L, controlled
follow-up decisions use the same Learned runtime with stable argmax; when policy
is H, they remain heuristic. Self-play and arena provenance identify the
Learned continuation recipe as `continuation_policy=learned-argmax-v1`.

## Availability and fallback

Optional components declare only two recoverable failures:

- `ComponentUnavailableError`: component/artifact/runtime is unavailable;
- `ComponentInferenceError`: an otherwise available component cannot score the
  request.

`FallbackSearchPolicy` and `FallbackLeafEvaluator` catch exactly those errors.
In server composition, the same fallback-wrapped root policy is used for Learned
argmax continuation, so only those declared failures recover to heuristic
scores. Self-play and arena composition deliberately use the same bare Learned
policy for roots and continuation, so inference failures propagate in both
places. Programming errors, invalid output types, candidate reordering,
non-finite values, and other exceptions always propagate. This prevents fallback
from concealing correctness bugs. In server degraded mode, fallback heuristic
scores are argmaxed over the complete canonical set; this can select FINISH or
SKIP when every concrete score is negative, unlike direct `HeuristicAgent`
continuation.
A policy fallback marks its returned scores as `FALLBACK`; root-only learned
PUCT and widening overrides are then disabled for that decision, while the
fallback scores may still order expansion under the classic schedule.

Server-level bounded compute separately protects the whole expensive agent
call with a process-wide concurrency limit, queue timeout, decision timeout,
output revalidation, and a cached Heuristic fallback. Search also checks its
monotonic deadline cooperatively between iterations and engine advances and
caps the advances within a simulation. A cooperative deadline reserves a small
margin before the coordinator's hard timeout: if any iterations completed,
search returns the most-visited fully evaluated legal move rather than discarding
that work. With no completed visits, the normal heuristic fallback still applies;
advance-limit and contract failures are not suppressed. Simulation sessions skip rollback
snapshot serialization without changing rollback boundary or confirmation rules.

Artifact construction failures (missing optional dependencies, missing or
invalid artifacts, and unsafe resolved paths) are logged and replace the
unavailable learned components with heuristics before observation encoding.
The requested bot specification remains persisted for a future restart.

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
2. builds agents/runtimes in a worker thread outside locks, publishing the cache
   only if the game/session/configuration still matches, then resnapshots state
   and computes decisions outside locks;
3. revalidates ownership, request identity, and legality on live state;
4. applies one decision through `GameSession` under the established lock order;
5. reuses ordinary clock finalization, save, log, replay, and scoped broadcast;
6. schedules idempotently from create/restore, REST, WebSocket, and timeout
   completion paths.

ISMCTS exposes bounded iterations and wall-clock timeout, independent
policy/value sources, leaf mode, and continuation horizon. The server's default
leaf mode is `bounded_continuation`; `immediate` must be requested explicitly.
The lower-level `SearchConfig` default remains separate. Search constants,
widening, and exploration remain internal. Offline configurations must pin their
leaf mode. The shared offline preset rejects non-null search decision timeouts;
wall-clock partial-tree recovery is a serving safety feature, not reproducible
teacher evidence. Generation also rejects results whose visits do not complete
the declared effective budget. Outer censoring watchdogs remain separate.

Observation projection uses a shallow neutral-topology view rather than cloning
the entire game. Hex adjacency uses six-neighbor lookup instead of an all-pairs
scan. No map-ID-only cache is used: geometry and moving topology effects are
read from each snapshot.

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
available). Only when the adaptive schedule is active (and therefore has a
root coverage target), an optional broad HEX root using direct contextual
heuristic value compares the best-ranked concrete hex with `SKIP` before
remaining coverage. With `adaptive_hex_root_schedule_version=None`, broad
HEX+SKIP roots retain classic scheduling. Exact `RESPAWN`/`PASS` and
`CHOOSE_ACTION`/`HOLD` roots using direct contextual heuristic value likewise
receive two real root simulations when needed. A learned-primary
`FallbackLeafEvaluator` does not force this narrow comparison merely because
its recoverable fallback supports edge shaping; if learned evaluation fails,
the edge that classic scheduling did expand still receives the prepared
heuristic fallback. These comparisons do not rewrite policy scores, seed Q
values, add pseudo-visits, or remove legal actions. When the learned *policy*
falls back, its fallback prior controls ordering while classic PUCT/widening
behavior is retained. After coverage, normal PUCT and progressive widening
resume.

The schedule never truncates legality. Diagnostics and training targets retain
every caller candidate; unvisited candidates have zero visits and therefore
zero improved-policy mass. `SearchResult` reports `requested_iterations`,
`effective_iterations`, and `root_coverage_target` for self-play telemetry.

### Request-aware root schedule

`SearchConfig.request_schedule_version` is a separate, versioned opt-in.
`None` preserves the full legacy path exactly, including the independent
adaptive-HEX and contextual no-op rules above. Version 1 and its
`request-aware-v1` identity are frozen unchanged. The field participates in the
strict self-play/arena search-config identity. Version 1 resolves an immutable
root plan only after strict root and legal-set validation, using typed
`classify_decision` semantics rather than prompts or option labels:

- validated singletons use zero simulations;
- binary defense/passive reactions use
  `max(2, min(config.iterations, 4))`, `IMMEDIATE_ACTION`, and two-action
  coverage;
- movement, spatial-selection, and respawn-destination HEX roots use coverage
  `M = min(K, 12, max(4, ceil(sqrt(K))))`, effective iterations
  `max(config.iterations, 16, 2 * M)`, and `IMMEDIATE_ACTION`. The 16-iteration
  floor is intentional even for narrow HEX roots (`K <= 8`): v1 reserves a
  stable minimum value-comparison budget rather than applying only broad-root
  coverage;
- action choices use `max(config.iterations, 8)` and `IMMEDIATE_ACTION`, while
  retaining the existing contextual action/no-op comparison rather than adding
  a second coverage rule;
- other input roots keep the configured iteration budget and use
  `IMMEDIATE_ACTION`; planning-card roots retain both their configured budget
  and configured leaf mode.

Version 2 (`request-aware-v2`) preserves every v1 budget and coverage formula,
but selects `STABLE_TURN` only for actor-bound resolution `INPUT` roots with a
live enclosing `resolution_owner_id` and action semantics. Planning-card,
tie-breaker/actor-choice, upgrade/interphase, and actorless roots retain safe
configured/immediate behavior. Explicit `STABLE_TURN` with the legacy schedule
is available for isolated actor-bound experiments; ineligible explicit roots
fall back immediately. Config and telemetry naturally distinguish both the leaf
mode and schedule ID/version without a serialized-shape bump.

When either request-aware schedule is active it owns broad-HEX scheduling, so enabling the
legacy adaptive-HEX option too cannot apply the formula twice. Existing
contextual no-op coverage may still raise a smaller plan to two real visits.
For HEX+`SKIP`, the effective immediate/stable horizon intentionally enables the
value-led no-op comparison: `effective_root_puct_c` becomes `0`, so the prior
orders expansion/coverage but does not bias the later value comparison. Action
choices containing `HOLD` retain their configured root PUCT while the existing
contextual no-op rule guarantees both the best concrete action and `HOLD` are
visited.

Search uses `dataclasses.replace` to create the effective horizon config; it
never mutates the caller's `SearchConfig` or removes/reorders legal candidates.
`SearchResult` and bounded decision telemetry carry the schedule ID, effective
leaf mode, request type, semantic role, requested/effective iterations, and
coverage target.

## Fresh Gen1 integration

The shared-encoder runtime, observation encoder, artifact loading, and batching
remain reusable foundations. The leaf and dataset changes required for fresh
Gen1 are tracked in [AI_LEARNING_CONTRACT.md](AI_LEARNING_CONTRACT.md). Until
search/data/model integration is complete, the existing leaf modes above remain
operational but must not be represented as the new learning contract.
