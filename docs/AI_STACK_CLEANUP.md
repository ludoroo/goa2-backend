# Clean AI stack: merge scope

This is the scope boundary for the cleaned replacement stack. It supersedes the
old proposal to merge the five drafts unchanged and add a corrective follow-up.
The old combined checkpoint `0631d50` remains preserved for provenance, not as
the branch to merge.

## Merge order

1. **Upstream sync** — `ai-gen1-upstream-gameplay-base` → `main`: merged upstream
   [#47](https://github.com/PedroVIOliv/goa2-backend/pull/47) and
   [#46](https://github.com/PedroVIOliv/goa2-backend/pull/46), through `822e096`.
   Also fixes five pre-existing Black formatting failures with unchanged Python
   ASTs, so the existing CI workflow can pass. The later Swift fix `803bad1` is
   not included.
2. **[#8](https://github.com/ludoroo/goa2-backend/pull/8)** — AI engine seams and
   bot/replay lifecycle support, based on that sync branch.
3. **[#9](https://github.com/ludoroo/goa2-backend/pull/9)** — learned runtime and
   guarded search, including the reviewed per-search continuation sampler.
4. **[#10](https://github.com/ludoroo/goa2-backend/pull/10)** — retained offline
   learning infrastructure, with the matching sampled-continuation composition
   and provenance.
5. **[#11](https://github.com/ludoroo/goa2-backend/pull/11)** then
   **[#12](https://github.com/ludoroo/goa2-backend/pull/12)** — stable-transition
   search, then offline outcome normalization.

Use merge commits, not squash/rebase merges, to preserve the stacked ancestry.
Do not delete base branches until their dependent PRs have landed. No additional
replay/sampling follow-up is needed: those changes are folded into the owning
layers. The native-data working tree is not part of this stack.

## Why engine files still appear in #8

Only these engine seams are intended to differ from the upstream-fixed base:

| File | Permitted AI change | Default game behavior |
|---|---|---|
| `engine/handler.py` | Optional `stop_before_step` callback | Unchanged when omitted |
| `engine/session.py` | Thread that optional callback through planning/advance | Unchanged when omitted |
| `engine/filters_units.py` | Extract the attack-immunity query with an explicit actor | Exact upstream source-ID and attacker-exception matching; no ownership reinterpretation |

The pure immunity query is shared by the engine filter and AI evaluation so
search does not have to mutate the live actor to ask a question. Its duration,
basic/non-basic attack, and exception semantics remain upstream's semantics.
Minion return, phase advancement, abort handling, card effects, and effect timing
are not separately changed by the AI layer.

Server changes in #8 remain AI operational support: reject stale bot results for
removed/replaced games, consume late abandoned-future exceptions, record automatic
bot-input provenance, and reconstruct replay without consulting live save files.
Card decisions are rebound to the live hero's card in the AI runtime driver.
#9's only additional GoA2 source change is learned-bot construction in
`server/bot_factory.py`; #10–#12 add no GoA2 source changes.

## Deferred behavior is not in the merge stack

The following old #8 behavior is removed, not renamed as AI support:

- Pruning stale unresolved-hero queues in `resolve_next_action()`.
- Broad `survives_action_abort` flags on phase-control steps.
- Raising an engine-wide error for an actorless, drained resolution stack.
- Normalizing immunity subjects, hero/piece ownership, or attacker exceptions.

Synthetic tests whose only purpose was asserting those changes are removed.
Upstream gameplay regressions remain. AI-local bounded-search/progression checks
remain appropriate; removing a speculative engine guard is not permission for
search to hang or treat an interrupted simulation as a valid value boundary.
A future gameplay proposal needs a legitimate interaction and its own review.

## Evidence and parked work

Historical model/data identities are not relabelled. `learned-argmax-v1` remains
an explicit historical/baseline recipe; new learned continuation composition uses
`learned-prior-sampling-v1`. Replay cleanup requires recorded `automatic` provenance;
it does not guess how to repair unflagged old bot logs.

The dirty `ai-gen1-native-data` checkout remains untouched and uncommitted.
No generation, training, or arena experiment is authorized by this cleanup.
The dependency manifests retain the already-reviewed changes in their original
layers; this cleanup adds no dependency changes.

## Cleaned-stack verification — 2026-09-28

Every layer passed the repository's existing CI commands locally, including
Ruff and Black over **both `src` and `tests`**, mypy over `src`, and the full
pytest suite with GoA2 branch-aware coverage and the 80% coverage gate:

| Layer | Tested implementation | Full-suite tests | GoA2 coverage (branches enabled) |
|---|---|---:|---:|
| Upstream sync | `d5b0540` | 4,063 | 87.53% |
| #8 AI seams/replay | `22cef92` | 4,100 | 87.61% |
| #9 runtime/sampling | `b511ba8` | 4,313 | 87.62% |
| #10 offline infrastructure | `40af1c8` | 4,847 | 87.73% |
| #11 stable-transition search | `9c7ac3b` | 4,905 | 87.73% |
| #12 outcome normalization | `f8b80a4` | 4,962 | 87.76% |

A separate read-only review found no blockers in the source cleanup, the layer
partition, or the subsequent test/documentation correction. That reviewer ran
no tests; the counts above are parent-run full suites, not reviewer evidence.
Local passes do not imply a GitHub Actions run succeeded; check the PR's current
remote checks separately.

The first #10 coverage run exposed a timing-sensitive test: a real first search
could exceed its arbitrary one-second deadline before the intended second-search
timeout. The test now injects that second-decision expiry after a real search
plan, keeping the fragment discard, next-game continuation, and telemetry checks.
Dedicated tests still exercise real POSIX expiry and nested deadlines. Production
timeout behavior is unchanged; the full corrected suite passed.

The corrected replay `f42c78365da2` reconstructs 61/61 decisions with an empty save
directory and matches its full normalized saved state and rollback state. Replay
and save bytes were not changed. Parked checkout branch, HEAD, status, and all
nine dirty/untracked file fingerprints were rechecked unchanged.

The final `src/automata` tree and dependency manifests match the prior reviewed
combined checkpoint `0631d50`. The intended differences are the engine-scope
removal, associated engine-test cleanup/parity coverage, the deterministic test
fixture above, and documentation. No native-data implementation is included.
