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
