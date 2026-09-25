# AI experiment journal

## Direction: reset the learning stack, retain the lessons

The goal is an AlphaZero-like learner: information-safe search produces policy
improvements, terminal outcomes supervise value, and repeated self-play updates
a network using accumulated experience.

The owner approved a fresh Gen1 lineage rather than preserving compatibility
with every experimental generation. Historical schemas, bridges, schedules,
models, and promotion protocols need not dictate the new design. This is a reset
of the experimental AI stack, not permission to break the game engine or its
client-facing API.

**Current status:** the first reset foundation and source cleanup are tested
and checkpointed in focused local commits (see final sections). #3/#4 have merged;
the final foundation at `851f96a` passes a fresh 4,046-test run. Integration is
complete locally on `ai-gen1-integration`; all 4,821 post-review tests and source
checks pass. Replacement drafts #8/#9/#10 are published; #6/#7 are closed as
superseded with their source branches preserved. Shared Gen1 search/data/model
implementation is next.
No new generation has started and no run artifacts were removed.
The historical code checkpoint before reset implementation is
`88b85d0c1b57c30a31be5fd00a0320236008924a`.

The older [lineage ledger](LEARNED_MODEL_LINEAGE.md) contains detailed artifact
identities and original verdicts. If historical artifacts are later deleted,
this journal preserves reported findings, not the ability to reproduce or
independently audit those runs. Deleting evidence does not turn a rejected model
into an accepted one.

## Historical experiments and lessons

| Experiment | Observed result | Lesson |
|---|---|---|
| Early learned-policy / heuristic-value baseline | Legacy Gen2 was the last arena-validated champion. | Hybrid search can produce a useful baseline. This does not establish learned-value search strength. |
| Complete-action leaves | Fixed important partial-action evaluation problems; one candidate still lost 16–48 to Gen2. | Mechanical correctness and lower training loss do not establish playing strength. |
| Typed decisions, native-v4 bootstrap and child | Gen3 established a native model; Gen4 failed policy gates and lost an exploratory arena 0–16 to Gen2. | Request context matters, but a schema change and successful training are not evidence of policy improvement. |
| Stable-turn leaves | Eight-game and larger generation runs completed without recorded search-progression failures. Gen5 still failed policy gates. | Completing actions and finalization is a useful evaluation boundary. Operational success alone does not prove strategic superiority. |
| 32-requested-visit probe | More candidate coverage and observed Q separation; eight games took about 83 minutes. Trajectories differed from the lower-budget run. | A stronger offline teacher is worth testing; this probe did not establish causal improvement or a stronger distilled player. |
| 256-game corpus / Gen6 | 202,182 decision rows; held-out equal-game Brier about 0.209 versus neutral 0.25. Policy gates failed. Value signal was strongest later in games. | More independent outcomes improved prediction, but decision rows are not independent games and prediction is not counterfactual control. |
| Learned-value runtime probes | Same-policy L/L-vs-L/H and L/H-vs-L/H both produced one-hour timeouts. Against Gen2, Gen6 L/H and L/L each lost one paired seed, with 11.5 average rounds. Total times were 10m34s and 17m52s. | Learned value was not necessary for the long-game behavior. The fixed-opponent probe observed about 69% more wall time, not an isolated measurement of neural inference cost or a reliable strength estimate. |

### Concrete implementation findings

- **Search-overturn reporting was defective.** Generation omitted actual root
  priors; indexed metrics substituted normalized visits for priors. Comparing
  visits to themselves cannot measure search overturn. This does not invalidate
  the separate policy-ranking or Brier computations.
- **The movement coverage requirement mismatched actual requests.** The recent
  corpora used `SELECT_HEX`, classified as `SPATIAL_SELECTION`, while a required
  `MOVEMENT_DESTINATION` bucket was empty. Spatial selection is broader than
  movement and should not be relabeled as movement without typed evidence.
- **Training did not warm-start successive models.** New runs initialized new
  weights; checkpoint resume only resumed the same run. This is not by itself
  proof of why candidates failed: training from scratch can still support policy
  iteration. Parent initialization is a sensible controlled improvement, not a
  demonstrated cure.
- **The operational loop was not closed.** Replay and iteration components
  existed, but experiments primarily trained on independent corpora, repeatedly
  returning to the old hybrid teacher. More scaffolding alone will not establish
  a policy-improvement mechanism.
- **Value training and leaf inference differed.** Training used actual
  pre-decision observations, while stable-turn leaves used a synthetic boundary
  context. The magnitude of this distribution shift has not been measured.
- **Stable leaves were selective, not universal.** Planning and actorless flows
  retained shorter horizons. Final-turn stable progression stopped on entry to
  cleanup, before end-phase processing. The new boundary contract must explicitly
  address these cases rather than calling all historical leaves stable.

## What the reset should keep

1. Deterministic engine, legal-action contracts, information-safe observations,
   and fixed viewer/perspective during search. GoA2 has hidden information; a
   textbook perfect-information AlphaZero search is not a drop-in replacement.
2. Stable completed-turn evaluation as the design direction. Define planning,
   reactions, cleanup, terminal detection, and watchdog behavior explicitly.
   A watchdog interruption must not silently become a normal stable value leaf.
3. Policy targets from search visits and value labels from terminal outcomes.
   Dedicated boundary value samples must come from the **actual played
   trajectory**, not unplayed counterfactual leaves labeled with that game's
   winner. Policy can still choose intermediate inputs without asking the value
   head to evaluate every intermediate state.
4. Honest diagnostics and held-out evaluation: actual priors versus visits,
   visited-action Q separation versus within-action variance, game-aware splits,
   playing strength, runtime cost, and censored timeout reporting.
5. A small executable generate/train/evaluate loop before large-scale runs.
   New experiments need traceable inputs, but not loaders for every discarded
   experimental format.

## Proposed clean Gen1 plan

This is a design direction, not a claim that the new backbone is implemented.

1. **Define one learning contract.** One observation/data format, one stable
   boundary contract with explicit phase handling, one search implementation,
   and one documented evaluation protocol. Remove obsolete AI compatibility
   code and contradictory runbook paths as part of implementation, retaining
   useful correctness tests.
2. **Bootstrap fresh data.** Start with a reproducible heuristic-backed search
   teacher; old Gen2/Gen6 weights and bridges are not required. Collect policy
   examples at decisions and terminal-labeled value examples at actual stable
   trajectory boundaries. A 32-requested-visit offline teacher and an
   eight-requested-visit player are proposed starting budgets, to be measured
   rather than treated as universal constants. Historical request scheduling
   used more than eight simulations for some spatial choices and fewer for
   forced/reaction decisions.
3. **Train fresh Gen1.** Validate learning and boundary prediction on held-out
   games, then evaluate actual play against the declared bootstrap baseline.
   Genesis is a new experimental starting point, not a retroactive promotion
   over the legacy champion.
4. **Test learned value immediately after bootstrap.** Hold policy weights and
   other search settings fixed while comparing heuristic and learned boundary
   evaluation. Use held-out boundary prediction plus paired gameplay and cost
   evidence. A fixed-opponent diagnostic and a bounded direct comparison serve
   different purposes. Do not require an arbitrary number of heuristic-only
   generations, or assume a 25/50/100% value blend is necessary.
5. **Repeat with accepted search.** Continue from parent weights with compatible
   replay and persistent train/evaluation isolation. Adopt learned leaves when
   the declared evidence supports them; otherwise run a bounded value-focused
   experiment rather than indefinitely scaling heuristic-only data. If a blend
   is tried, compare it explicitly: the heuristic's shaped score is not
   automatically a calibrated win probability.

### Provenance and cleanup

A fresh run under a fresh lineage avoids dependence on old commit identities;
it does not repair historical evidence. Keep recording the new code revision,
configuration, input data, and parent weights. Do not relabel old datasets or
reuse old checkpoints as if they came from the new recipe.

Retirement can remove legacy AI readers, bridges, unused leaf recipes, and old
operational paths. Artifact deletion is a separate explicit cleanup step with
an inventory, not a side effect of running training or changing a loader.

## Reset implementation: first foundation

The target contract now lives in
[AI_LEARNING_CONTRACT.md](AI_LEARNING_CONTRACT.md). The other thread owns merging
PRs #3/#4; this thread owns all #6/#7 rework and engine-fix preservation.

The first implementation slice adds:

- Shared `ACTOR_READY` / `PLANNING_READY` detection and transition anchors, plus
  internal planning stop hooks. Actual-play observation uses the hook without
  stopping execution or adding harness ticks. End-of-round cleanup and empty-hand
  automatic passes are covered. Everyone passing may lead directly to a later
  planning boundary without selecting an actor.
- Candidate-free, information-safe value observations with a fixed private
  viewer and the actual next actor's context. Real decision owners are
  deduplicated per completed trajectory transition; observations remain
  provisional until normal terminal completion.
- Actual root prior preservation, including unvisited candidates, and indexed
  metrics that no longer substitute visits for priors. Missing prior evidence
  stays missing. Obsolete metric caches are rejected/rebuilt; historical run
  artifacts have not been opened for rewriting or retroactively relabeled.

Baseline verification found 4,602 passing tests and one machine-path-dependent
hash assertion. Replacing the old Herdr map path reproduced that expected hash,
confirming the cause. The test now checks stable identities and seed/schema
separation rather than one developer's absolute checkout path.

Independent review found that end-of-turn finishing effects can leave a stale
actor flag in a planning observation. The value encoder now takes actor/owner
context explicitly from the boundary without mutating the live state. A real
finishing-effect regression proves canonical planning observations are
path-independent. Additional tests cover foreign-actor privacy, tie-choice
viewers, and the same hero acting in a later turn. Hook integration tests live
under `tests/automata/`, not the engine-only test tree.

Post-review verification: **4,653 tests passed**, full source Ruff and mypy,
Black/Ruff for changed Python files, and `git diff --check`. The existing
engine/server compatibility subset also passed (505 tests).

This is **not** a working Gen1 pipeline yet. Existing search still uses its
historical leaf behavior; the new value contract is not yet wired into dataset
publication, model batching/losses, or the training loop. No fresh data or models
have been generated and no legacy run artifacts have been removed.

## Cleanup before further implementation

The owner asked to tidy the #6/#7 code before extending the new backbone.
Feature work was paused. This pass removed:

- `training.curriculum`, `generation_pipeline`, and `policy_iteration`: no
  production/CLI callers existed. Their dedicated orchestration tests were
  retired; independent replay/registry/split/generation/arena tests remain.
- Observation-v3 / tensor-model-runtime-v1 execution support: legacy contract,
  adapter, old schema-digest exceptions, old model branch, bridge flags, and
  compatibility-only tests/fixture. Native observation-v4/tensor-v2/model-runtime-v2
  identities, shapes, validation, privacy, and artifact integrity remain intact.
- The accumulated historical stage recipes from the operations guide, replacing
  them with current tool discovery and safety/status documentation. Historical
  outcomes remain in this journal and the lineage ledger; Git retains old code
  and commands without requiring permanent legacy loaders.

Independent review found no removal blockers. A pre-existing self-play validation
weakness was also corrected: map/hero-adapter runtime requirements now come from
current engine constants and the adapter registry, rather than echoing the
artifact's declarations. Arena tests cover old-format rejection before weight
loading and continued binding of runtime identity into protocol identity.

Relative to the saved pre-tidy working tree, this pass removed a net **1,940
production Python lines** and **1,674 test lines**. The test reduction is
intentional retirement of obsolete paths, not removal of the retained
infrastructure's correctness tests. Post-review verification: **4,631 tests
passed**, source Ruff/Black/mypy and diff checks passed, and all seven documented
CLI `--help` entry points passed. `pyproject.toml`, `uv.lock`, and run artifacts
were not changed.

This is **local source cleanup**, not a published PR rewrite. #6/#7 have not been
restacked, pushed, closed, or merged here. The current native joint dataset,
trainer, and historical search modes remain operational until their Gen1
replacements exist. A new executable learning loop is still required.

Remaining identity consideration: current source configuration includes an
absolute map path, so relocation can change generator IDs and unnamespaced
agent seeds. The Gen1 identity design must resolve that intentionally; this
cleanup did not silently rewrite existing source/seed identities.

## Local checkpoint and foundation integration plan — 2026-09-25

The tested source has now been split into four focused local commits without
altering the working source or replaying old history:

- `311d643`: stable boundaries, observation hooks, candidate-free encoding/tests.
- `a913e38`: real search priors and cache/identity evidence tests.
- `5cefa61`: native-only model/runtime/CLI path and explicit old-format rejection.
- `a782dad`: unused orchestration removal, retained infrastructure tests intact.

The documentation/runbook checkpoint follows separately. These commits preserve
our work; they are not yet replacement PRs based on the newly merged foundation.
No push, remote PR mutation, or rebase has been performed by this thread.

At that checkpoint, GitHub confirmed #3 merged at `7e75671`; #4 remained open
at `b688058`, targeting `main`, with tracked fixes in the owner's checkout. Read-only review
identified decision-context/team-owner, forced-transition/deadline, topology,
and bot-lifecycle overlap. The details and preservation checklist are in
[AI_PR_RESET_HANDOFF.md](AI_PR_RESET_HANDOFF.md). Do not treat the other thread's
uncommitted patch as verified or overwrite it with this older source tree.

## Merged foundation verification — 2026-09-25

GitHub confirms #4 merged at `851f96a480dd0fcd48c21a95dec30c3536110b2f`, directly
on #3's `7e75671`. The final PR4 tree (`b5ba737`) and merged tree are identical;
GitHub `main`, `origin/main`, local `main`, and the primary checkout agree.
Published #6/#7 heads remain `5c844f3` / `a0f8181`, both open. #6 now targets
`main`, while #7 still targets `ai-learned-harness`.

Fresh checks on the merged foundation: **4,046 tests passed**, Ruff and mypy
passed. Black found only line-wrapping changes in `search/contracts.py` and
`server/bots.py`; they were not edited during verification. Tests ran with frozen,
no-sync dependencies. Existing untracked evaluation artifacts in the primary
checkout were not inspected or changed. This is a separate foundation result,
not a new test count for our 4,631-test reset branch or proof of integration.

Independent targeted review of `b688058..851f96a` found **no blockers** and reran
seven affected test files: **78 passed** (a subset of the full suite). It verified
fixed-viewer decision context, team ownership, deadline recovery, topology parity,
and detached bot construction/stale-result rejection. Non-blocking fallback and
coverage follow-ups are recorded in the handoff. Crucially, main's learned value
path still requires candidates: this is not the new candidate-free Gen1 runtime.
Deterministic offline search must also disable wall-clock decision timeouts and
must not silently substitute heuristic agents when evaluating learned artifacts.

Both squash merge commits preserve the private author but use a personal-email
committer. This is an unresolved identity caveat, not a runtime defect. Published
history was left intact; changing it requires a separate explicit decision.

## Integration started — 2026-09-25

Created local `ai-gen1-integration` directly on `851f96a` in the same checkout.
The old `ai-learned-self-play-next` branch remains at `fc20bb9`. An explicit
three-way assembly used original #4 tree `2dd1772f8519` as the source boundary;
this carries retained changes forward without replaying the rewritten history
or discarding the newly merged upstream fixes. The preliminary result identified
13 conflicting files, split between search/observation, engine/serving, and docs.
Mechanically merged files still require tests and semantic review.

Dependency files remain identical to `fc20bb9`: existing harness requirements
are carried forward, not changed during integration. The native-only cleanup,
boundary foundation, and real prior evidence remain in scope. New Gen1 search/
data/model functionality remains deferred until this integration passes.

Reconciliation now passes **4,806 combined tests**, source Ruff/Black/mypy
(282 source files), conflict-marker checks, and byte-identity checks for the
preserved dependency files. The suite includes the seven documented CLI help
checks. Engine, server, and Automata subsets passed separately too.

Failing-first regressions caught two integration issues: a following team-scoped
request must keep the latest eligible concrete decision owner without changing
the root private viewer; a game removed/replaced in the registry must not be
mutated after compute or receive a detached agent cache. Both are fixed and
covered. Main's explicit descriptor API, topology optimization, off-lock serving,
and deadline safety coexist with our current native format and retained search.
The candidate-free observation foundation remains separate; historical learned
leaves are still transitional, not a newly finished Gen1 runtime.

### Review corrections and completion

Search review found that a zero-hand forced pass could still reach a learned
leaf in immediate modes or at a bounded cutoff. The apparent regression test
was vacuous because determinization refilled the cleared hand. The corrected
loop advances forced transitions before evaluation across all modes, retaining
fingerprint, forced-count, deadline, and engine-advance guards. Strengthened tests
use a faithful clone and failed before the fix.

Serving's recovered partial search also required an explicit offline boundary.
The shared offline preset now rejects non-null cooperative search timeouts, and
generation rejects results that do not complete their declared effective visit
budget. A failing-first test proves an earlier valid provisional row is discarded
when a subsequent decision returns partial visits, even if a runner could report
normal game-over. Outer game-censoring watchdogs remain supported.

The reviewed team-owner precedence intentionally preserves the latest eligible
concrete owner for team-scoped descendants, not main's root-first preference.
The regression distinguishes Xargatha's ownership from Wasp's fixed private
viewer and team perspective. Registry ownership now has real missing/replaced
coverage before publication, apply, and idle advance; new-format automatic replay
has standalone reconstruction coverage without a companion save.

Independent engine/serving review found no blockers; a focused follow-up review
cleared the search correction and offline guards (83 targeted tests passed).
Final combined verification: **4,821 tests passed**, source Ruff/Black/mypy
(282 source files) and diff/dependency identity checks passed. A separately tested
engine + runtime/model snapshot passes **1,000 tests** before adding the offline
layer, supporting the focused commit boundaries.

Local integration commits, directly on `851f96a`:

- `f26d9a6`: engine progression, boundaries, registry safety, and automatic replay.
- `2af43f5`: current native runtime/model, explicit context, guarded search.
- `718ed47`: retained offline harness/training/replay/arena infrastructure.

At that integration checkpoint, no published PR had been rewritten, pushed, or
closed. The original source branch and run artifacts remain intact. Dependency
files are unchanged from `fc20bb9`;
their existing harness additions relative to main are carried forward. This is
foundation integration, not completion of the fresh learning loop.

## Draft publication and superseded PR closure — 2026-09-25

With the owner's approval to proceed autonomously, published a dependency-ordered
draft train without rewriting existing history:

- [#8](https://github.com/ludoroo/goa2-backend/pull/8): engine/replay foundation,
  `ai-gen1-engine-foundation` → `main`.
- [#9](https://github.com/ludoroo/goa2-backend/pull/9): native runtime/model/search,
  `ai-gen1-runtime-foundation` → `ai-gen1-engine-foundation`.
- [#10](https://github.com/ludoroo/goa2-backend/pull/10): offline infrastructure and
  reset plan, `ai-gen1-integration` → `ai-gen1-runtime-foundation`.

Before publication, the exact first-layer snapshot passed **4,090 tests** in
isolation. The second-layer isolation result remains 1,000 focused tests; the
full stack remains 4,821 tests. No CI success is inferred from creating the drafts.

Closed **#7, then #6**, as superseded after the replacement links existed. Both
received explicit preservation/scope comments. Verified their old remote heads
still exist at `a0f8181` and `5c844f3`; no branches or artifacts were deleted.
The new PRs remain drafts and were not merged. Their descriptions explicitly
separate the tested foundation from the still-missing Gen1 learning loop.

**Next action:** shared Gen1 search/live boundary parity and exact team-oriented
terminal handling, then candidate-free value runtime/data integration. Keep the
active checklist in [AI_LEARNING_CONTRACT.md](AI_LEARNING_CONTRACT.md) and evidence
in [AI_PR_RESET_HANDOFF.md](AI_PR_RESET_HANDOFF.md) current at each checkpoint.
