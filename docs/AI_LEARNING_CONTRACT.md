# Gen1 learning contract

**Current status — native data landed:** #15 merged into `main` at `2f1bc92`,
with a tree identical to the verified landing `1406cac`. The reviewed AI stack
and newer upstream Swift fix are both present. The merged foundation passes
4,963 tests. The native rows/recorder checkpoint is now verified locally at
`9060250` on `ai-gen1-native-dataset`: **5,024 full-suite tests**, 67 focused
native/adapter tests, and all Ruff/Black/mypy checks pass. GoA2 branch-aware
coverage is 87.76%. Independent read-only follow-up review found no blockers;
the reviewer ran no tests. [#16](https://github.com/ludoroo/goa2-backend/pull/16)
merged into `main` at `471b6f8`, with a tree identical to verified `298c63d` and
a fresh 5,024-test baseline. Original parked files remain unchanged.

The schema/batching checkpoint landed in
[#17](https://github.com/ludoroo/goa2-backend/pull/17) at `f8469f4`, with the same
tree as published `e66249f` and source/tests matching verified `52f5085`:
**5,049 full-suite tests**, 141 focused model/schema/index tests, all
Ruff/Black/mypy checks, and 87.76% GoA2 branch coverage. Independent read-only
follow-up review found no blockers and ran no tests.

The current local branch, `ai-gen1-stable-value-runtime`, adds separate policy
and candidate-free stable-value model forwards, distinct Gen1 artifacts, CPU
runtime inference, and `LearnedStableValueEvaluator`. Model identity is
`goa2-gen1-policy-stable-value-v1`; manifest schema 3 declares
`GEN1_POLICY_STABLE_VALUE` and `stable-boundary-outcome-v1`. Legacy joint
artifacts are rejected, not relabelled or used to initialize this model.
Local verification: **5,091 full-suite tests**, **161 focused tests**, Ruff and
Black over `src tests`, mypy over `src`, and **87.76%** GoA2 branch-aware coverage
(80% gate). Independent read-only review found no blockers and ran no tests.
The owner approved committing, publishing, and merging this checkpoint directly
into `main`; delivery is tracked in the PR reset handoff.

This does **not** make Gen1 training ready: bounded indexing, separate losses,
and trainer/generator/iteration integration remain separate checkpoints.
Existing joint-data commands are not Gen1 commands; no generation is authorized.

**Cleaned-stack scope:** follow
[AI_STACK_CLEANUP.md](AI_STACK_CLEANUP.md) for the original PR decomposition. The separate
upstream-sync base owns the gameplay fixes; #8 retains only agreed AI seams and
bot/replay support. Deferred engine recovery and immunity ownership changes are
not included. Replay cleanup is folded into #8 and prior-sampled continuations
into #9/#10. Historical verification numbers below describe their original
checkpoints, not the rewritten heads. The original native-data files remain
parked outside that stack; their reviewed port is delivered separately in #16.
No generation/training gate has been lifted.

**Scope:** the fresh AI lineage replacing the historical #6/#7 experimental
pipeline. #3/#4 remain the runtime/model foundation; engine rules and the client
API remain intact. Historical AI artifact compatibility is not a requirement.

This is the target contract and current execution plan. The following foundation
checkpoint descriptions are historical; the cleaned versions are now on `main`
through #15. Boundary recognition, actual-play observation, candidate-free value
encoding, and the first source cleanup were published in the replacement draft stack. Opt-in
heuristic-valued search now uses the shared transition contract on
`ai-gen1-search-parity`. Offline outcomes are normalized on
`ai-gen1-outcome-normalization`;
4,936 full-suite tests and source checks pass, with independent review complete.
Search and outcome checkpoints are published as drafts #11 → #12 above #10.
Learned-value inference, Gen1 dataset publication, model batching/losses, and the
learning loop must adopt the transition contract before any fresh Gen1
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

Learned controlled continuations use `learned-prior-sampling-v1`: sample legal
follow-ups from the policy distribution, using stable softmax at temperature 1
for logits. A fresh, domain-separated RNG is bound per search from its configured
seed and advances across iterations; it is not shared between searches or with
root/tree, determinization, environment, or live-game randomness. Root PUCT and
actual-play visit sampling remain separate. Server, self-play, and arena use the
same continuation recipe. Historical `learned-argmax-v1` evidence retains its
original identity; sampling must not silently reuse its checkpoint/cache entries.

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
metrics use `(value + 1) / 2` explicitly. The current engine has no terminal draw
rule: `GAME_OVER` with no winner is malformed and fails closed, not a zero label.
Nullable dataset/evaluation outcomes retain abstract draw support; adding an actual
engine draw requires positive rule evidence rather than missing winner markers.

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
3. **Complete, reviewed, and published as draft #11 — heuristic search parity.**
   `ai-gen1-search-parity` is based on publication checkpoint `3eef358`; published
   foundation drafts stay fixed. `STABLE_TRANSITION` uses shared boundaries for
   planning and INPUT roots, exact terminal orientation, owned/foreign routing,
   and fail-closed bounds. Search/live candidate-free encodings agree byte-for-byte
   for the same world/viewer/boundary. Current verification: 4,879 full-suite tests
   pass, including 35 new-mode tests, plus source Ruff/Black/mypy. Independent
   review and correction/gap-test follow-up found no blockers. Unsupported
   learned/fallback evaluators are rejected, even for singleton roots. Live-bot
   deadline recovery still cannot authorize incomplete teacher evidence or turn
   an interrupted transition into a stable value leaf.
4. **Complete, reviewed, and published as draft #12 — offline outcomes; data/model next.**
   `ai-gen1-outcome-normalization` starts from reviewed search checkpoint `02b3cce`.
   Search and actual play share authoritative terminal-team resolution; raw
   individual-winner diagnostics survive even rejected outcomes. Censored games
   never become draws or strength evidence. Verification: **4,936 full-suite tests**
   and source Ruff/Black/mypy pass; dependencies are unchanged.
   Next add discriminated policy/value rows, atomic complete-game publication,
   bounded indexing, per-head masks/weights, and candidate-free value batching/
   runtime. Replace the retained joint-data path and resolve source/seed identity
   portability without relabeling old datasets or checkpoints.
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

### Offline outcome slice (implemented and independently reviewed)

The neutral runtime resolver maps raw team/hero winners using authoritative
finished-state rosters, shared by search and actual play. `RunResult` keeps raw
`winner` and requires explicit normalized `winner_side`. Raw trajectory recording
happens before normalization, so rejected unknown/missing winners remain diagnostic
evidence. Learning observers and the strict dataset recorder use `winner_side`.
Unknown/missing engine winners, contradictory fields, and nonterminal winners fail
closed; invalid recorder outcomes also discard provisional data immediately.

Generic matchup evaluation rejects nonterminal results because it lacks a
censoring model. Strict arena checkpoints retain censored observations only for
operational diagnostics: summaries exclude them from wins/draws, paired scoring
rejects them, and arena promotion stops without strength metrics. Fresh sequential
execution, fully cached replay, and partial-pair resume have regression coverage.
Cost averages still include non-timeout censored rows and are named
`average_non_timeout` in new learned-arena summary schema version 2.

Bootstrap receipts carry raw winner plus required normalized side; incompatible
old receipts are rejected, not inferred or migrated. Use fresh bootstrap checkpoint
paths: a pre-change row missing `winner_side` invalidates the entire checkpoint
file even if its configuration identity differs. Generation identities pin
`raw-winner+canonical-side-v1`. Self-play checks both live result/fragment and
resumed checkpoint/fragment winners; known contradictory new fragments are removed
so resume cannot recover them as valid orphans. Legitimate crash-orphan recovery
is unchanged. Independent review found no production blockers; its final identity-
test defect was corrected and the full suite rerun. No training-row schema expansion
or historical artifact migration is part of this fix. Native policy/value data
and model changes still follow it.

### Native dataset/recorder slice

The new API is separate from the retained joint-data commands:

- `training/native_dataset.py`: `NativeGameIdentity`, `NativeBoundaryProvenance`,
  tagged `PolicyDatasetRecord` / `ValueDatasetRecord`, `native_game_id`,
  `native_sample_id`, `iter_native_game_records`, and `publish_native_game`.
- `training/native_recorder.py`: `NativeDatasetRecorder(path, *, game=...)`, with
  `record_policy`, `record_boundary`, `record_outcome`, and idempotent `close`.
- `training/search_targets.py`: `search_policy_target_from_result` is the public
  root-evidence adapter. The existing joint generator delegates to it; this does
  not silently convert legacy commands into native generation.

Record schema version 1 is discriminated by `sample_kind`. Policy records contain
`DecisionObservation` schema 4 and `SearchPolicyTarget`; value records contain
`StableValueObservation` schema 1, boundary provenance, terminal winner and a
signed perspective-correct label. Existing observation/viewer contracts are not
changed. Actual priors remain distinct from visit probabilities; singleton roots
retain zero visits/returns and probability one, not invented search evidence.

Game IDs hash a namespaced canonical encoding of world seed, map/game type,
setup-ordered team compositions, generation/source revision/dirty-tree provenance,
source-model digest (nullable for heuristic bootstrap), and search/generator
configuration IDs. Sample IDs additionally bind kind and a global sample index.
Policy indexes count only decisions; boundary indexes count retained boundary
groups, with one value row per entitled viewer. Hero references in boundary
provenance are observation-local, not strings to compare across observations.
Grouping resolves them to public hero IDs, with one consistent roster per game.
Token sorting does not change setup-order identity: observation membership is
validated independently of token order. Policy viewers must be the unique SELF
hero and decision owner on the declared team.
Identity inputs are caller-supplied provenance; hashes do not certify execution.

The live outer strategy supplies actual searched policy roots. Pass the recorder
as the harness's **`boundary_observer`**, not its decision/trajectory `recorder`:
the harness already supplies distinct entitled viewers and one normalized outcome.
Search leaves do not call this sink. Enclose its lifetime in a context manager so
exceptions/interruption discard its provisional data even before an outcome.

During play, samples are immediately serialized to a private `.pending.zst`
spool, outside the final `*.jsonl` / `*.jsonl.zst` discovery patterns. Publication
checks frame integrity and sample/policy/boundary counts against live bookkeeping.
Only a normal decisive `game_over` publishes the complete game to `.jsonl` or
`.jsonl.zst`; timeout/censoring/failure/unfinished close discards both heads.
Nullable value labels preserve abstract draw support in the schema only—the
current recorder rejects missing engine winners rather than inventing draws.
Final publication validates records, fsyncs staged data, and uses an atomic
no-clobber link. Existing files and racing publishers are never overwritten.
Memory scales with one record/boundary group rather than accumulated game length.

Readers validate canonical JSON, schema/index/game consistency, and compressed
frame integrity. They must be consumed to exhaustion before accepting a source;
yielding an early row is not certification of the remaining file. An arbitrary
syntactically valid JSONL prefix is not proof of normal game completion: production
completion is established by the live recorder, not inferred from a filename.
Persisted source receipts/index verification belong to the later integration
checkpoint. No native CLI, dataset conversion, or training run is added here.

### Next data/model checkpoints (no generation yet)

1. **Native rows and publication — merged in #16, implementation `9060250`.** Discriminated
   policy/value records and a whole-game recorder over the existing `StableBoundaryObserver` and
   `encode_stable_value` seams. Policy rows retain exact candidates/root visits
   without a value target; value rows contain actual boundary observations and
   terminal labels without policy candidates. Coverage includes per-viewer
   deduplication, unsorted 2v2 rosters, perspective orientation, atomic publication,
   corruption/count mismatch, canonical target alignment, and whole-game discard.
2. **Candidate-free model/runtime — local integration checkpoint.** Schema/batching
   landed in #17: `GraphBatch`, `StableValueBatch`, `collate_stable_values`, and
   torch-free `StableValueTensorSchema` (`goa2-stable-value-tensor-v1`). The new
   model shares graph processing but separates decision/candidate policy inputs
   from stable boundary context and the value head. Runtime protocols expose
   `evaluate_policy` and `evaluate_stable_value`, not a joint Gen1 `evaluate`.
   Both tensor schemas and the stable-outcome semantics are pinned in a distinct
   artifact manifest. The native search adapter uses actual boundary encoding
   with a fixed viewer/perspective; exact terminal scoring still bypasses it.
   Legacy decision-v2 batching/cache and joint model/runtime remain operational.
   No serving/CLI adoption or decision-trained artifact reinterpretation occurs.
3. **Bounded indexing and separate losses.** Index tagged policy/value chunks with
   source/dataset identity checks. Normalize each head independently per game;
   policy metrics must ignore value rows and value metrics need no candidate
   metadata. Test interrupted indexing, digest/range validation, and weighting.
4. **Trainer and generation integration.** Wire native training and actual-play
   recording end to end, then retire the replaced joint path deliberately. The
   retained joint commands stay operational during these checkpoints, not as a
   permanent legacy-format bridge. Only after these checks and the separate
   executable-iteration gate may a small fresh Gen1 diagnostic be generated.

Artifact deletion is a separate inventoried task. No reset command may delete
`runs/` or mutate historical results as a side effect.
