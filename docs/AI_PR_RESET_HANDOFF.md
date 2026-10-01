# AI PR reset: merge/rebase handoff

## Current checkpoint — native indexing and separate losses

Current branch: `ai-gen1-native-index-losses`, based on merged main `5fae662`.
Work and reviews use `/tmp/goa2-clean-stack.e0SwTB/repo`; do not create additional
agent workspaces or alter the original parked checkout. Fresh baseline: **5,091
tests passed**. Final local verification: **5,184 full-suite tests**, **249 focused
tests**, Ruff/Black over `src tests`, mypy over `src`, and **87.76%** GoA2
branch-aware coverage (80% gate). Independent loss and index reviews, including
hardening follow-up, found no remaining blockers. Reviewers also ran focused
checks; these are local results, not remote CI or playing-strength evidence.

The owner approved committing, publishing, and merging this checkpoint as one
direct PR into `main`: [#19](https://github.com/ludoroo/goa2-backend/pull/19),
`ai-gen1-native-index-losses` → `main`, source/test checkpoint `73d187a`.
Publication follow-ups are documentation only. The owner also approved continuing
the next integration checkpoint; no training/generation run is authorized.
Final logs and
source/test fingerprints are recorded outside the repository at
`/tmp/goa2-clean-stack.e0SwTB/index-losses-verification.json`.

Scope: explicit canonical source receipts, bounded homogeneous policy/value
index chunks, separate typed training batches/metrics, and independently weighted
head losses. Full-game head counts define row weights; chunk boundaries must not
change losses or gradients. Receipt construction inventories expected file bytes,
not proof of completed legitimate gameplay. No recorder/generator issuance,
split/replay/trainer integration, CLI, or experimental training is included.

Red/green fixes protect sources/receipts from cache overlap, reject unrelated
cache/staging paths through exact ownership markers, recompute dataset identity
on reuse, validate completed checkpoint ranges, reject duplicate normalizer game
IDs, and bound single-frame decompression before output allocation. Integration
coverage checks full-game per-head mass and identical model gradients across
chunk sizes. Cold cache open still performs full source/chunk validation; abnormal
publication interruption can discard resumable work or leave backups, without
authorizing deletion of source data. Recovery limits are in the operations guide.

### Landed native Gen1 model/runtime checkpoint

[#18](https://github.com/ludoroo/goa2-backend/pull/18) merged at `5fae662`.
Its tree equals published `d55e0e1`; source/tests/dependencies equal verified
`dc8742a`. No separate full test run was claimed for the merge operation; the
fresh baseline above is the new checkpoint's pre-change run.

The verified local slice adds separate policy/stable-value model forwards,
distinct Gen1 artifacts and CPU inference, and the native stable-search adapter.
Verification: **5,091 full-suite tests**, **161 focused tests**, Ruff/Black over
`src tests`, mypy over `src`, and **87.76%** GoA2 branch-aware coverage (80% gate).
Independent read-only review found no blockers and ran no tests. Model-only
review also found no blockers. These are local checks, not remote CI or strength
evidence. The owner approved committing, publishing, and merging this checkpoint
as one direct PR into `main`:
[#18](https://github.com/ludoroo/goa2-backend/pull/18),
`ai-gen1-stable-value-runtime` → `main`. Source/test checkpoint is `dc8742a`;
subsequent publication changes are documentation only. No training, generation,
or arena run is authorized by that delivery approval.

Artifact hardening rejects incompatible config before weight loading and preserves
a competing empty destination directory. Publication reserves the destination
exclusively and installs the manifest last; an interrupted publication fails
closed and can require manual cleanup of its incomplete directory before retry.
Logs and source/test fingerprints are recorded outside the repository at
`/tmp/goa2-clean-stack.e0SwTB/runtime-verification.json`. The original checkout's
HEAD, branch, and all nine parked file fingerprints remain unchanged.
Bounded native indexing and per-head losses are the current local checkpoint;
trainer/generator/iteration adoption follows separately. Existing joint commands
and artifacts retain their old identities; native inference is not a
training-readiness claim.

### Landed schema/batching and native-data foundations

PR #15 merged into `main` at `2f1bc92`; the tree equals verified `1406cac` exactly.
Both the cleaned AI stack and newer upstream Swift fix are present. A fresh
post-merge baseline passes 4,963 tests. The intermediate-branch landing gap from
#8–#13 is resolved; no foundation PR remains to merge. See
[AI_STACK_CLEANUP.md](AI_STACK_CLEANUP.md) for the historical landing evidence.

The native-data branch was `ai-gen1-native-dataset`, based on that merged main.
Production/test checkpoint `9060250` passes **5,024 full-suite tests**, **67
focused native/adapter tests**, Ruff/Black over `src tests`, mypy over `src`, and
the GoA2 branch-coverage gate (87.76%). Independent read-only follow-up review
found no blockers and ran no tests; the test results are parent-run local evidence.
[#16](https://github.com/ludoroo/goa2-backend/pull/16) merged into `main` at
`471b6f8`, whose tree equals verified `298c63d`; source/tests still match
`9060250`. A fresh post-merge baseline passes 5,024 tests.

The previous branch `ai-gen1-candidate-free-value` started from `471b6f8` and covered
only shared graph vectorization/batching plus distinctly identified candidate-free
value tensors. Source/test checkpoint `52f5085` passes **5,049 full-suite tests**,
**141 focused tests**, source/test Ruff and Black, source mypy, and the GoA2
branch-coverage gate (87.76%). Independent read-only follow-up review found no
blockers and ran no tests. [#17](https://github.com/ludoroo/goa2-backend/pull/17)
merged directly into `main` at `f8469f4`, tree-identical to published `e66249f`.
Source/tests match tested `52f5085`; the subsequent commits only record
verification/publication. A fresh baseline before the native-runtime slice passed
5,049 tests; this is not a separate full-suite claim for the merge operation itself.

A differential audit against immutable `471b6f8` confirmed legacy schema canonical
bytes, vectorized fields, and all decision-batch tensors for nine real decision
fixtures, plus the positional constructor and flattened layout. New-path fixes
reject finite Python numbers that overflow float32 and missing required references.
Model heads, new artifact/runtime capability, and learned stable search are the
current local checkpoint. Native indexes/losses and trainer/iteration adoption
remain separate. Do not turn decision-trained artifacts into stable-boundary
value models.

### Landed native-data checkpoint (historical detail)

Eight source/test files were ported from the parked native-data work, then
corrected during review; its old contract patch was not applied over current
documentation. The original checkout, branch/HEAD, and all nine dirty files
remain unchanged. Work/reviews use the same
shared workspace, `/tmp/goa2-clean-stack.e0SwTB/repo`.

This checkpoint covers policy/value records, strict one-game I/O, whole-game
recording, and the public root-target adapter only. Red/green regressions fixed
unsorted multi-hero rosters, truncated/incomplete spools, observation-local ref
comparison, and reordered candidate evidence. Policy viewer/owner identity is
validated, and provisional spools cannot match final-dataset filename patterns.
Candidate-free model/runtime, bounded indexing/per-head losses, and
trainer/iteration integration follow it.
No Gen1 generation, training, arena experiment, dependency change, historical
artifact conversion, engine change, or client API change is part of this slice.

## Cleaned-stack scope before landing (historical publication)

**Do not merge the old five drafts unchanged.** The stack is rebuilt according to
[AI_STACK_CLEANUP.md](AI_STACK_CLEANUP.md), which is the current source of truth
for scope and merge order. First land the separate fork upstream-sync base
(`ai-gen1-upstream-gameplay-base`), then #8 → #9 → #10 → #11 → #12 using merge
commits. Fork `main` is not changed by preparing or publishing this stack.

Only the agreed optional search hooks and exact-semantics immunity query remain
in the AI engine delta. Stale-actor pruning, broad phase abort-survival flags,
the drained-resolution engine guard, and immunity ownership normalization are
removed, not newly approved. Minion-return and duration fixes come from merged
upstream #46/#47, not duplicated AI patches. Replay cleanup is folded into #8;
sampled continuations into #9/#10. No final corrective follow-up is needed.

The old combined checkpoint `0631d50` and prior source history stay preserved.
Passing its tests did not approve its mixed engine scope. The chronology below
records historical heads and test counts; it is not a second current merge plan.
Historical instructions to preserve whole progression patches are superseded by
the explicit scope above. The dirty `ai-gen1-native-data` checkout stays parked
at `18916f4`, outside this delivery; generation and training remain gated.

## Historical recommendation — 2026-09-25

PRs **#3 and #4 are merged**. The verified integration base is
`851f96a480dd0fcd48c21a95dec30c3536110b2f`. PRs **#6 and #7 are closed as
superseded, not merged**. Their useful code and engine fixes are preserved in
draft replacements **#8 → #9 → #10**; their source branches remain intact.
Reviewed search and outcome follow-ups are now published as drafts **#11 → #12**
on top of that foundation. Review/merge order is **#8 → #9 → #10 → #11 → #12**.

The original recommendation was a scope/dependency assessment, not a merge
approval. Fresh verification of the merged foundation is recorded below. Neither
PR reports GitHub status checks, so the passing local test run is the evidence;
the integration and later search-slice checks below are separate evidence.

The experiment history and reset direction are in
[AI_EXPERIMENT_JOURNAL.md](AI_EXPERIMENT_JOURNAL.md). The active implementation
checklist is in [AI_LEARNING_CONTRACT.md](AI_LEARNING_CONTRACT.md).

## Published search/outcome follow-ups

| Draft | Branch → base | Reviewed implementation |
|---|---|---|
| [#11](https://github.com/ludoroo/goa2-backend/pull/11) | `ai-gen1-search-parity` → `ai-gen1-integration` | `d66682e` + `02b3cce`: terminal-team scoring and stable-transition heuristic search; 4,879 tests |
| [#12](https://github.com/ludoroo/goa2-backend/pull/12) | `ai-gen1-outcome-normalization` → `ai-gen1-search-parity` | `c78a4ca`: offline outcome normalization, censoring, and resume correctness; 4,936 tests |

Both branches were pushed with the owner's approval; both PRs are drafts and
cross-linked. Foundation #8/#9/#10 remain drafts at their prior heads. Nothing
was merged, marked ready, or force-pushed. Publication adds only documentation
on top of the reviewed outcome implementation. Source/test verification remains
at the checkpoints above; these local results are not a claim of GitHub CI success.
No generation, training, arena experiment, or artifact conversion was performed.

## Current offline-outcome checkpoint — 2026-09-25

Implemented and verified on `ai-gen1-outcome-normalization`, starting from reviewed
search checkpoint `02b3cce` (4,879 passing tests). Search history remains on
`ai-gen1-search-parity`; no worktree moved. Draft #8/#9/#10 states and heads were
reverified unchanged before this slice; this recovery did not change remote refs.

Raw engine winners remain diagnostic evidence, with a required explicit normalized
`winner_side` for learning/evaluation resolved from actual team rosters. Learning
callbacks use that named side; raw trajectory callbacks retain raw `winner` even
when normalization rejects it. Unknown or missing engine winners fail closed:
there is no engine draw rule that would justify labeling absent markers as draws.
Generic matchup rejects censored games; strict evaluation retains them only as
operational evidence, excluding them from draws, wins, and paired strength scores
and blocking promotion. Bootstrap receipts/config identities make the outcome
contract explicit; no historical receipt inference or migration occurs. Use fresh
bootstrap checkpoint paths: any legacy row missing `winner_side` rejects the file,
even if it has another configuration identity. Self-play receipts keep normalized
`winner` (unlike bootstrap's raw `winner`), cross-checked against fragment labels.

Recovered the interrupted session's 4,923-test run and independent review. Fixed
its missing-winner and raw-diagnostic findings, added sequential fresh/cache/
partial-pair censor regressions, avoided new type-only runtime import coupling,
and made invalid dataset outcomes clean their spool immediately. A raw-engine
integration test covers both perspectives, real decisions, normalized labels,
invalid/missing winners, and whole-game discard. Learned-arena cost averages are
now truthfully named `average_non_timeout` in summary schema version 2. Additional
red/green regressions exposed a contradictory newly published fragment surviving
into resume, and an unchecked cached receipt winner. Both now fail closed without
breaking valid crash-orphan recovery; rejected new fragments cannot be recovered.

Final verification: **4,936 full-suite tests pass**, source Ruff/Black/mypy pass
(283 files), and dependency files remain byte-identical to `fc20bb9`. Independent
review found no production blockers and verified the resume corrections. Its last
finding was a lazy-property identity test reading its baseline too late; corrected
that test and reran the full suite. No generation, training, arena experiment,
push, merge, artifact conversion, dependency, or client API change occurred.

Censoring remains explicit in observation `reason` fields; a separate redundant
arena halt marker is deferred. The exploratory `evaluation/search_boundaries.py`
scoring path is not connected to an engine runner and must be audited before reuse
as Gen1 evidence. Neither point weakens this slice's promotion gates.

**Next action:** implement native discriminated policy/value records and atomic
actual-boundary publication first; the ordered model/index/trainer checkpoints are
in `AI_LEARNING_CONTRACT.md`. Iteration and fresh-generation gates remain closed.

## Reviewed search-parity checkpoint — 2026-09-25

Published branch `ai-gen1-search-parity` (draft #11) starts from checkpoint `3eef358`.
The #8/#9/#10 foundation heads remain unchanged. Terminal checkpoint `d66682e`
fixes search rewards for individual hero winners using authoritative team membership;
unknown non-null winners now raise rather than counting as losses for both teams.
Both rollout and tree terminal paths bypass leaf evaluation.

Independent review found no blockers; follow-up tests cover losing tree backup,
missing perspective team, and rejection of piece IDs as hero winner identities.
Verification: **4,844 full-suite tests pass**, including 23 new terminal tests;
source Ruff/Black/mypy and dependency-identity checks pass.

The next slice is implemented locally and independently reviewed:
`STABLE_TRANSITION` uses shared live/search boundaries with a candidate-free
`StableValueContext`/`StableValueEvaluator` interface, currently public-material
heuristic value only. Every root visit completes the transition, including
planning and already-expanded roots. All session mutators carry the boundary
hook; cleaned planning is checked after calls. Owned continuation retains its
latest eligible owner and fixed viewer/team. Simultaneous upgrades use an explicit
environment fallback; unknown/unencodable requests fail closed.

Parent-written tests compare live/search canonical observations byte-for-byte
and exercise budgets, terminal bypass, owner continuity, and actual learned-runtime
rejection. Additional guard tests exposed permissive invalid planning/input keys
and an absent-boundary acceptance case; these now fail clearly instead of
silently finishing/skipping or valuing a non-boundary. Review probes became durable
end-to-end tests for real upgrades, owned/foreign and teammate routing, Emmitt's
second commit/retrieval, unsupported simultaneous input, and evaluator contracts;
the private-helper-only upgrade test was replaced. **4,879 full-suite tests pass**,
including 35 new-mode tests; source Ruff/Black/mypy and dependency checks pass.
Independent follow-up verified the corrections and all gap tests, with no blockers
(127 focused tests including historical modes; counts overlap the full suite).

Learned-value use remains unsupported until candidate-free model/runtime support
exists. Historical `STABLE_TURN`, serving defaults, schemas, and artifacts are
unchanged. Request schedules 1/2 cannot be combined with the new mode; no
historical schedule may silently change its horizon. No Gen1 generation was run.

**Gap at this search checkpoint, now addressed by the outcome slice above:** generic
matchup counted hero-ID winners as draws; retained joint training and the learned
arena rejected those IDs. The shared resolver now aligns these paths without
relabeling historical results or artifacts.

## Published replacement stack — 2026-09-25

| Draft | Branch → base | Scope |
|---|---|---|
| [#8](https://github.com/ludoroo/goa2-backend/pull/8) | `ai-gen1-engine-foundation` → `main` | Engine progression, boundary hooks, registry and replay safety |
| [#9](https://github.com/ludoroo/goa2-backend/pull/9) | `ai-gen1-runtime-foundation` → `ai-gen1-engine-foundation` | Native model/runtime, explicit context, guarded search |
| [#10](https://github.com/ludoroo/goa2-backend/pull/10) | `ai-gen1-integration` → `ai-gen1-runtime-foundation` | Offline foundations, evidence safeguards, reset documentation |

Review/merge in dependency order; none was merged or marked ready automatically.
#6/#7 were closed only after all three replacement links existed, with explanatory
comments on both old PRs. Remote `ai-learned-harness` remains at `5c844f3`, and
remote `ai-learned-self-play` remains at `a0f8181`. No branch or artifact was deleted.
The first draft's exact source snapshot additionally passes **4,090 tests** on its
own; runtime/model isolation and combined results are recorded below.

This is published foundation salvage, **not a completed Gen1 learning loop**.
Further Gen1 features follow these checkpoints rather than silently changing the
meaning of the historical experiments.

## Integration checkpoint — 2026-09-25

**Integration complete and published as drafts:** `ai-gen1-integration` is based directly on
`851f96a`; `ai-learned-self-play-next` preserves the reviewed source at `fc20bb9`.
No worktree was moved and no existing published history was rewritten. An explicit three-way
source boundary (`2dd1772f8519`, the original #4 tree) carried the retained delta
forward, rather than replaying the rewritten history. Thirteen conflicting files
and the mechanically merged engine/server changes were reconciled and reviewed.

| Integration commit | Scope |
|---|---|
| `f26d9a6` | Engine progression/boundaries, live-card rebinding, replay safety, and registry ownership |
| `2af43f5` | Native model/runtime, explicit decision context, guarded search, and strict offline preset |
| `718ed47` | Retained harness, streaming/indexing, training/replay/arena foundations and operations guide |

Final verification: **4,821 tests passed**, source Ruff/Black/mypy passed (282
source files), and the seven documented CLI help checks passed within the suite.
A separate snapshot of engine + runtime/model passes **1,000 tests** without the
offline layer, confirming that boundary of the commit series works independently.
The two main formatting findings are resolved locally. Dependency files remain
byte-identical to `fc20bb9`; existing harness additions relative to main were
retained, not newly changed dependencies. `runs/` remains untouched.

Engine/serving review found no blockers. Search review caught a forced-pass leaf
regression hidden by determinization refilling the test hand; the strengthened
tests now fail on the old integration and pass across all leaf modes/cutoff edges.
The correction reuses fingerprint, forced-decision, deadline, and advance guards.
A separate follow-up review cleared it and the new offline guards: non-null
cooperative search deadlines are rejected by the offline preset, and incomplete
visit budgets cannot publish any provisional game rows. The last focused review
passed 83 tests. These test counts overlap with the full suite.

Team-scoped descendant owner precedence is deliberate: retain the latest eligible
concrete decision owner, with the fixed root viewer as fallback. This preserves
`fc20bb9` semantics rather than main's root-first preference; a regression explicitly
distinguishes Xargatha as decision owner from Wasp as private viewer and asserts
that viewer/team remain fixed. Root-level ownership stays coordinator-selected.

The following records the verified input checkpoints, not a combined-tree verdict:

- GitHub reports #3 merged at `7e75671d18e4faff6312c98d62ad220054075971`.
- #4 is **merged** at `851f96a480dd0fcd48c21a95dec30c3536110b2f`, whose parent is
  the #3 merge. GitHub `main`, `origin/main`, local `main`, and the primary checkout
  match that SHA. Its tree exactly matches final PR4 head `b5ba737`.
- Fresh verification on merged `main`: **4,046 tests passed**, Ruff passed, mypy
  passed (237 source files). Black reports formatting-only failures in
  `src/automata/search/contracts.py` and `src/goa2/server/bots.py`; no files were
  reformatted during verification. Existing untracked `data/evaluations/` in the
  primary checkout was left untouched; tracked files stayed clean.
- At the foundation verification, #6/#7 were still open at `5c844f3` / `a0f8181`.
  They have since been closed as superseded; those source heads remain preserved.
  The merged #3/#4 remote branches were removed by the merge thread. The original
  verification itself did not push, mutate shared refs, or integrate changes.
- **Identity caveat:** both merge commits have the private `ludoroo` author but a
  personal-email committer. Published history was not rewritten here. Changing
  those existing identities would require a separate explicit history decision;
  local Git identity settings do not retroactively fix them.
- Our tested changes have been checkpointed on `ai-learned-self-play-next` without
  changing their source contents or rewriting the old history:

  | Commit | Focus |
  |---|---|
  | `311d643` | Stable value boundaries, actual-play observation, candidate-free encoding, hooks and behavior tests |
  | `a913e38` | Actual search prior evidence, stale-cache rejection, and portable identity assertions |
  | `5cefa61` | Native-only learned format/runtime/CLI validation and legacy-path retirement |
  | `a782dad` | Unused callback orchestration removal with independent infrastructure tests retained |

  The reset contract, journal, handoff, and consolidated operations guide form
  documentation checkpoint `afd9fbd`. These commits are local preservation/review
  units, **not yet the replacement #6/#7 PR series on the new foundation**.

### Landed #4 integration checklist

The final `b688058..851f96a` fix diff spans 19 files, including regression tests.
Independent targeted review found **no blockers**, and reran seven affected test
files: **78 passed**. These are a subset of the 4,046-test foundation suite, not
additional distinct tests. Preserve these invariants during integration:

1. **Decision context and team ownership.** #4 passes the actual descriptor rather
   than reconstructing it from an empty input stack, and resolves team requests to
   concrete heroes. Reconcile `SearchContext.decision` there with this branch's
   `current_decision` deliberately, keeping one canonical interface. Adopt the
   explicit descriptor and `for_decision(decision, owner_id=...)` API first. Keep
   root viewer/perspective separate from decision owner and resolution owner.
2. **Candidate-free value.** Main's learned leaf still requires legal candidates
   and advances forced passes to obtain them. Our new boundary-value runtime must
   have a separate candidate-free path, not fabricate candidates or inherit that
   advancement workaround. Preserve working transitional search until replacement
   behavior is implemented and tested.
3. **Forced transitions and deadlines.** Preserve bounded work and completed-visit
   recovery for live bots without weakening complete-game publication or treating
   censored transitions as stable training leaves. Wall-clock partial-tree return
   is nondeterministic: deterministic self-play must explicitly disable decision
   timeouts. Advance-limit/evaluator failures must remain fail-closed for teachers.
4. **Observation projection.** Carry forward six-neighbor graph construction and
   occupancy-free topology projection without losing fixed-viewer privacy or our
   canonical boundary actor/owner override. Review confirmed projection parity and
   state non-mutation on main; rerun those checks on the combined encoder.
5. **Bot lifecycle and fallbacks.** Preserve off-lock construction on detached
   snapshots, stale-result rejection, artifact-reference validation, and serving
   fallback observability. Offline evaluation must not silently become heuristic
   play on invalid artifacts. Pin leaf mode explicitly: the server default is now
   `bounded_continuation`, while `SearchConfig` has not changed its default.

Review follow-ups addressed locally: artifact fallback no longer swallows
programmer `ValueError`s; actual registry missing/replaced cases cover publication,
apply, and idle advancement; new-format automatic replay reconstructs with no live
save. Forced-pass advancement is intentionally evaluator-neutral, preserving main's
H/H behavior as well as learned leaves. Published `main` was not edited here.

**Next action:** implement shared Gen1 stable-transition search and exact terminal
orientation on this integrated base, following the learning contract. Existing
synthetic decision-leaf behavior remains transitional until that replacement is
implemented and tested; it is not the candidate-free Gen1 value path.

## Retained source cleanup inventory

The initial cleanup was checkpointed on `ai-learned-self-play-next`, then carried
into `ai-gen1-integration`; it does **not** rewrite the live #6/#7 PR branches.
Feature implementation was paused for this cleanup. That source checkpoint passed
**4,631 tests**, source Ruff/Black/mypy, and all seven
documented CLI help checks. Production code is a net 1,940 lines smaller than
the pre-tidy working tree. The retained changes are now published through the
replacement draft stack above; original source checkpoints remain preserved.

| Group | Action | Preserved behavior |
|---|---|---|
| Observation-v3 / tensor-model-runtime-v1 bridge | Remove adapter, old contracts/execution/digest exceptions, CLI bridge flags, and compatibility-only fixtures/tests | One native format (observation v4 / tensor v2 / model-runtime v2), candidate validation, artifact integrity/scope, modern round trips |
| `training.curriculum`, `generation_pipeline`, `policy_iteration` | Remove unused callback/declarative coordinators and their dedicated tests; no production/CLI callers | Actual generation/checkpointing, replay, registry, splits, I/O, training, arena and promotion helpers, independent correctness tests |
| Historical operations runbook | Replace accumulated stage recipes with a short current operations/status guide | Journal, original verdicts/lineage identities, current tool entry points, provenance/resume/seed safety rules |
| Existing search modes and joint dataset/trainer | Keep for now; rework against the Gen1 contract later | Working search, current generation/train/evaluate tools, watchdog and privacy tests |
| Engine/client correctness and new boundary foundation | Keep intact | All engine fixes, server API/default behavior, stable boundary detection/observation and prior-evidence correction |

Deleted experimental code remains accessible in Git history. No run artifacts
are deleted. The old format bridge is not retained just to replay old results.
Removing the callback wrappers makes the missing executable learning loop
explicit rather than representing test-only composition as an operational loop.

The integration base is the verified #4 merge, `851f96a`. Preserve all historical
source refs until replacement #6/#7 delivery and a separate cleanup decision; the
new local series does not authorize deleting archival branches or artifacts.

## Original PR stack at the first inspection (historical)

```text
main
  └─ #3 ai-classic-search
       └─ #4 ai-nn-rl
            └─ #6 ai-learned-harness
                 └─ #7 ai-learned-self-play (published portion)
```

| PR | Recommendation | Keep | Planned changes |
|---|---|---|---|
| [#3 Classic AI runtime](https://github.com/ludoroo/goa2-backend/pull/3) | KEEP; merge after normal checks | Determinization, fixed viewer/perspective, legality, policy/value seams, classic agents, bounded server execution, card knowledge and tests | Stable-boundary search will later extend/replace experimental leaf behavior. No wholesale runtime/server rewrite is planned. |
| [#4 Learned model/runtime](https://github.com/ludoroo/goa2-backend/pull/4) | KEEP; merge after #3 and normal checks | Information-safe graph observations, candidate alignment, shared encoder, policy/value heads, artifact integrity, learned adapters and serving | Update the one active observation/boundary contract; support boundary-value inputs without requiring fake policy decisions. Value-only inference can be added here later. Old serialized AI layouts need not remain supported. |
| [#6 Training/evaluation harness](https://github.com/ludoroo/goa2-backend/pull/6) | HOLD; rework before merging | Headless harness, streaming/indexing, safe I/O, game-level splitting, arena/statistical primitives, registry, replay and associated behavior tests | Separate policy-decision samples from actual stable-boundary value samples; masked losses; parent initialization; truthful metrics; real executable iteration composition and persistent holdout isolation. Re-establish dataset/checkpoint contracts rather than adding compatibility shims. |
| [#7 Learned-guided self-play](https://github.com/ludoroo/goa2-backend/pull/7) | REPLACE after salvage | Engine progression fixes, search watchdogs, candidate/action identity, bounded-memory publication, provenance, paired arenas and useful tests | Fresh bootstrap/self-play recipe using the final stable-boundary contract; clean budgets/exploration settings; remove accumulated legacy recipes and contradictory operations instructions. |

"Keep" does not mean those modules are frozen forever. It means their
architecture remains useful and the reset does not justify throwing them away.
Likewise, #6/#7 contain substantial reusable code; replacement is about a clear
learning contract and reviewable changes, not a ground-up rewrite of all I/O and
execution infrastructure.

## Original ownership for the two threads (historical, superseded)

### Fork cleanup / merge thread

1. Preserve the existing refs and this handoff before rewriting ancestry.
2. Review/test and land #3, then restack/review/test and land #4.
3. Leave #6 and #7 unmerged pending the replacement learning contract. Do not
   delete their source branches or the current working branch until salvage is
   complete. Close superseded PRs only once their replacement is accounted for.

Additional local engine changes in the old `e3b9c99` / new `9174a83` and old
`bb7205f` / new `7b6df70` commits include action-boundary support and basic-only
attack-immunity behavior. Inspect these separately; they are mixed with AI work
and are not blanket cherry-pick recommendations.

### Learning reset thread

The owner clarified that this thread owns **all #6/#7 work**, including engine
fix salvage. The cleanup thread should only tidy and merge #3/#4.

- The earlier plan was to preserve/review the engine progression patch from #7:
  original `1c8c4e2450f4341ea674561bb538c409c7c31891`, rewritten `1deb58985cff`.
  It touched `goa2/engine/phases.py`, `session.py`, `steps/combat.py`,
  `steps/phases.py`, and three engine test files. **Do not carry it wholesale:**
  the current cleanup retains only agreed AI seams and the separately merged
  upstream fixes; the unproven recovery/ownership behavior is parked.
- Define the single stable-boundary/observation/data contract, including planning,
  foreign reactions, cleanup, terminal handling, and watchdog interruption.
- Rework #6's data/loss/training/replay integration on that contract.
- Build the replacement for #7 using the useful existing generation/search code
  and later local stable-turn implementation/tests.
- Keep current-format validation and information safety. Remove legacy format
  bridges rather than bypassing validation to load old artifacts.
- Generate fresh Gen1 only after the new contract is implemented and tested.

No source-code reset, merge, rebase, PR closure, push, or artifact deletion was
performed while preparing the original handoff. Reset implementation has since
started in this checkout; see [AI_LEARNING_CONTRACT.md](AI_LEARNING_CONTRACT.md).

## Original published versus local history: do not confuse them

At the first inspection the published tips used the original author-history IDs.
The other thread has since restacked #3/#4; the table below records historical
source boundaries, not their current GitHub tips. All 32 feature commits then on
`ai-learned-self-play-next` were rewritten to the private identity; corresponding
trees and messages were verified equal.

| Boundary | Published/original ID | Equivalent on current rewritten branch |
|---|---|---|
| Base `main` | `95dc69e2d49d328bba5fea32aa238c4005f5f761` | Same |
| #3 tip | `f4b5401c6d23404e9dcd0b332506d5631b061180` | `8386a2e5e965` |
| #4 tip | `9d983411c5fcfa45cf42be0a5f4da3f3da50438f` | `2dd1772f8519` |
| #6 tip | `5c844f3a6e367c30156aedfef554a847e03de879` | `798d74defffd` |
| Published #7 tip | `a0f81815cca33e26fa7fae96b23e6b07207135fa` | `84c1004627aa` |
| Local experiment tip | `e3e3c66f5edfcfdf157ee24669fda1c4131e25c4` | `88b85d0c1b57c30a31be5fd00a0320236008924a` |

There are **22 additional local commits after published #7**. In particular:

- Typed decision context: old `e180b91`, rewritten `6a1dedd`.
- Learned owned continuation: old `cb2bf34`, rewritten `92b90fe`.
- Stable-turn leaves and tests: old `2ebce82`, rewritten `40213fb`.
- Later arena value-ablation support and experiment reports are local too.

These are sources for the reset, not independent patches guaranteed to apply on
#4. The stable-turn commit depends on intervening search/context changes. Extract
coherent functionality and tests, rather than cherry-picking that commit alone.

If #3/#4 are squash-merged or rebased, restack descendants using their exact
previous base boundaries. Do not treat the rewritten current branch as identical
Git ancestry to the published stack or blindly replay all 32 commits onto the
new base. For substantially rebuilt #6/#7, assemble intentional changes atop the
landed foundations rather than resolving obsolete experiments one by one.

At the original audit, the private author rewrite had not been applied to the
published branches. The merge thread owns final #3/#4 authorship verification;
local Git configuration alone does not rewrite existing commits.

## Files and artifacts to preserve during concurrent cleanup

- Working checkout: `/Users/lucasbarcelos/code/.goa2-backend-ai-learned-self-play-next`
- Working branch: `ai-gen1-outcome-normalization`, based on reviewed search
  checkpoint `02b3cce`; the published foundation remains `ai-gen1-integration`
  on merged #4 at `851f96a`.
- Preserved source branch: `ai-learned-self-play-next` at `fc20bb9`.
- Preserve the reviewed search/outcome checkpoints and their contract/journal/handoff
  documents; these follow-ups are published in drafts #11/#12, not merged.
- `runs/` is ignored and now lives in this checkout, not the removed Herdr one.
  Do not delete or clean this worktree as a side effect of restacking PRs.
- Historical artifact retirement is a separate cleanup decision; it is not
  necessary for merging #3/#4.
