# Gen1 learning contract

**Scope:** the fresh AI lineage replacing the historical #6/#7 experimental
pipeline. #3/#4 remain the runtime/model foundation; engine rules and the client
API remain intact. Historical AI artifact compatibility is not a requirement.

This is the target contract and current execution plan. Boundary recognition,
actual-play observation, candidate-free value encoding, and the first source
cleanup are implemented and published in the replacement draft stack. Opt-in
heuristic-valued search now uses the shared transition contract on the local
search-parity branch. Learned-value inference, dataset publication, model
batching/losses, and the learning loop must adopt it before any fresh Gen1
generation. Existing commands are not yet Gen1 commands.

**Status verified 2026-09-25:** #3 is merged at `7e75671`; #4 is merged at
`851f96a480dd0fcd48c21a95dec30c3536110b2f`. GitHub `main`, `origin/main`, and local
`main` agree. The merged foundation passes 4,046 tests plus Ruff/mypy; targeted
review found no blockers and reran 78 of those tests. Black flags two formatting-
only changes. The preserved reset source at `fc20bb9` passes 4,631 tests.
Foundation integration is **complete and published as drafts #8 → #9 → #10**,
based on `851f96a`. The reviewed combined result passes **4,821 tests** and source
Ruff/Black/mypy; draft #8 independently passes 4,090 tests. Superseded #6/#7 are
closed, not merged. Their source branches and artifacts remain intact; no history
was rewritten. This is a reviewable foundation, not a complete Gen1 pipeline.
Commit and integration details are maintained in
[AI_PR_RESET_HANDOFF.md](AI_PR_RESET_HANDOFF.md); historical findings stay in
[AI_EXPERIMENT_JOURNAL.md](AI_EXPERIMENT_JOURNAL.md).

## 1. Value is defined at stable completed transitions

There are two nonterminal value boundaries:

| Kind | Engine state | What has completed |
|---|---|---|
| `ACTOR_READY` | Resolution has selected an actor; execution is immediately before their unstarted `RESPAWN_HERO` or `RESOLVE_CARD`. | Planning/revelation and initiative selection, or the previous actor's full turn and finalization. |
| `PLANNING_READY` | New planning is open, with no partial card commitments or pending resolution/cleanup work. Automatic passes for empty-handed heroes are allowed. | The previous resolution phase and all required turn/round cleanup. |

A boundary includes `(kind, round, turn, actor_id)`. Recognition is based on typed
engine state, not prompts, phase changes alone, or a synthetic `CONFIRM` request.
The transition anchor records the starting phase/round/turn/resolution owner:

- **Planning root:** complete remaining planning, revelation, and actor selection;
  stop at first `ACTOR_READY`, not after a partial commitment. If everyone passes
  and no actor exists, accept the next later `PLANNING_READY` after cleanup.
- **Actor-bound root, including a defender's reaction:** complete the enclosing
  actor's turn; stop at a different actor's `ACTOR_READY` or later
  `PLANNING_READY`. An intermediate second card or respawn of the same actor does
  not complete the transition.
- **Actorless/tie/cleanup root:** complete pending work until the next genuine
  actor/planning boundary.
- **Last actor in turn 4:** entering `CLEANUP` is not enough. Process minion
  battle, removals, lane movement, upgrades, and round reset before evaluating.
- **Terminal:** use the exact outcome, not the learned value head. Mandatory
  abort, empty options, repeated states, and watchdog exhaustion are not terminal
  outcomes or valid value boundaries.

The same detector and anchor semantics govern hypothetical search and actual
play. Recording observes the real engine before steps execute and does **not**
add ticks, pause gameplay, choose actions, or alter RNG consumption.

## 2. Policy and value observations have different meanings

**Policy observation:** information-safe graph plus typed current decision and
ordered legal candidates. Policy can act at intermediate requests. Its target
is the search visit distribution over those exact candidates.

**Value observation:** information-safe graph plus explicit boundary kind. It has
no legal candidates, fake action, or policy request. Its graph identifies the
boundary actor, where one exists, separately from the private viewer. Actor and
owner context are explicitly empty at `PLANNING_READY`, even if a finishing
effect left advisory actor fields on the live engine state.

During each search transition, keep the root viewer hero and perspective team
fixed. Subsequent actors do not grant access to their private cards. Owned
continuation choices use the controlled policy; foreign choices use an explicit
information-safe environment policy. Unsupported simultaneous inputs require an
explicit fallback rather than silently being treated as foreign.

For actual play, collect distinct real decision owners since the previous
boundary. At the next accepted boundary, encode one observation for each of
those viewers using that hero's team perspective. Repeated decisions by the same
viewer do not duplicate a boundary sample. No viewer decisions means no sample.
Initial setup is not a completed-transition sample.

Changing only another hero's hidden cards must not change a viewer's encoded
value observation. Encoder inputs are validated against the actual boundary;
stale descriptors and non-boundary states are rejected.

## 3. Targets come from played games, not imagined outcomes

The new dataset has distinct sample kinds:

- **Policy:** actual decision observation plus aligned root visits and actual
  priors/return diagnostics. No value loss at this intermediate observation.
- **Value:** actual trajectory boundary observation plus that game's terminal
  outcome. No policy target at this candidate-free observation.

Value scale is `[-1, 1]`: win `+1`, loss `-1`, and a genuine terminal draw `0`,
from the fixed perspective team's viewpoint. Individual hero winners must be
resolved through their team, not treated as an unknown winning team. Probability
metrics use `(value + 1) / 2` explicitly.

A counterfactual search leaf is never labeled with the played game's winner.
Search may inspect it to choose an action; only the subsequent actual trajectory
can contribute terminal-supervised value examples.

Boundaries and decisions are provisionally spooled during play. Publish a game
only after normal `game_over`. Timeout, max steps/rounds, search watchdog failure,
inference failure, and exceptions discard the entire game spool. Count these
as operational failures/censored evidence, not strategic draws or losses.

## 4. Learning and evidence

- Bootstrap from heuristic-backed search, train fresh Gen1, then immediately
  test learned leaves with fixed-policy controls. Old models are not parents of
  this lineage and need not be loadable.
- Later training starts from declared parent weights with compatible replay.
  Optimizer resume and cross-generation weight initialization are distinct.
- Keep durable game/seed-level train/validation/arena isolation across the
  accumulated replay population. Normalize policy and value contributions
  separately per game; rows are not independent game outcomes.
- Preserve actual priors versus visit targets. Missing priors mean unavailable
  overturn evidence, never a visit-derived substitute. Within-action return
  variance is distinct from between-action Q separation. Use typed spatial
  roles rather than inventing movement categories from prompt wording.
- Record source/configuration/data/parent identities. Evaluate held-out boundary
  prediction, distilled-player gameplay, and operational cost. Starting visit
  budgets are tunable experimental settings, not schema invariants.

## Implementation sequence / generation gate

1. **Complete — foundation and first cleanup.** Shared boundary detection/anchors,
   planning stop hooks, behavior-neutral actual-play observation, candidate-free
   value encoding, truthful prior evidence, and removal of legacy model bridges
   and unused callback coordinators. The preserved work is in drafts #8/#9/#10;
   old #6/#7 are closed as superseded.
2. **Complete; drafts published — integrate the verified #4 foundation.** The explicit
   `SearchContext.decision` / `for_decision` API and main's engine/server/privacy
   fixes coexist with the retained native runtime and infrastructure. Review caught
   and corrected forced-pass leaf handling and incomplete offline search evidence.
   Engine `f26d9a6`, runtime/model `2af43f5`, and offline `718ed47` are based on
   `851f96a`; no historical chain was blindly replayed. Independent reviews and
   all combined tests/source checks pass. These are published draft checkpoints,
   not merged replacements or a completed Gen1 pipeline.
3. **Complete locally and reviewed — heuristic search parity.** Work is
   on `ai-gen1-search-parity`, based on publication checkpoint `3eef358`; published
   foundation drafts stay fixed. `STABLE_TRANSITION` uses shared boundaries for
   planning and INPUT roots, exact terminal orientation, owned/foreign routing,
   and fail-closed bounds. Search/live candidate-free encodings agree byte-for-byte
   for the same world/viewer/boundary. Current verification: 4,879 full-suite tests
   pass, including 35 new-mode tests, plus source Ruff/Black/mypy. Independent
   review and correction/gap-test follow-up found no blockers. Unsupported
   learned/fallback evaluators are rejected, even for singleton roots. Live-bot
   deadline recovery still cannot authorize incomplete teacher evidence or turn
   an interrupted transition into a stable value leaf.
4. **Pending — data and model.** Discriminated policy/value rows, atomic complete-
   game publication, bounded indexing, per-head masks/weights, candidate-free value
   batching and runtime. Align offline winner normalization too: the generic
   matchup evaluator currently counts hero-ID winners as draws, whereas joint
   training and the learned arena reject them. Neither behavior is the Gen1 team-
   outcome contract. Replace the retained joint-data path and resolve source/seed
   identity portability without relabeling old datasets or checkpoints.
5. **Pending — executable iteration, then fresh generation.** Bootstrap → train →
   paired evaluation → parent initialization/replay, with persistent split/seed
   isolation. Start only after the preceding steps and behavior/engine/server
   tests pass. The first fresh Gen1 run is a small diagnostic, not a large
   historical-style experiment.

### Current search slice: implementation choices

- **Implemented locally (`d66682e`):** exact search terminal scoring resolves hero-ID winners
  through authoritative team membership and rejects unknown non-null winners.
  Both terminal paths bypass leaf evaluators. Independent review found no blockers;
  verification after follow-up coverage: 4,844 full-suite tests pass (23 new cases),
  with source Ruff/Black/mypy clean.
- Opt-in `STABLE_TRANSITION` leaves `STABLE_TURN` unchanged. It uses the shared
  transition anchor/detector for planning, actor, reaction, tie, and cleanup roots.
  Historical request schedules 1/2 are rejected with this mode rather than
  silently downgrading its horizon; the unscheduled/adaptive-HEX path is supported.
- `StableValueContext` and `StableValueEvaluator.evaluate_stable_value` are the
  candidate-free seam, implemented by the heuristic evaluator using public
  material only. Incompatible learned/fallback evaluators fail before inference;
  no synthetic policy candidates or heuristic substitution are permitted.
- Owned continuation uses the controlled policy with fixed root viewer/team and
  persistent latest eligible owner. `UPGRADE_PHASE` has an explicit environment
  fallback until policy encoding supports simultaneous upgrades; other unknown
  simultaneous requests fail. Noncanonical selections are rejected rather than
  silently treated as planning finish or input skip; absent boundaries fail too.
- Keep the old synthetic-context path operational until the data/model slice
  supplies its replacement. This search slice alone does not open the generation
  gate or change client APIs, training schemas, dependencies, or old artifacts.

Artifact deletion is a separate inventoried task. No reset command may delete
`runs/` or mutate historical results as a side effect.
