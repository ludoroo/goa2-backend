# Consensus Overrides — Design

**Date:** 2026-08-05
**Status:** Approved design, not yet planned

## Problem

Playing on a tabletop simulator, the players are the referee: if something goes
wrong, someone picks up the piece and puts it where it belongs. Playing against
this engine, the engine is the referee. When it refuses a legal move, resolves a
defeat that shouldn't have happened, or miscredits a lane push, the table has no
recourse. The session stops being fun and the players stop playing.

The existing recovery machinery does not close this gap:

- `GameSession.rollback()` restores the *current actor's* turn-start snapshot.
  It is limited to that one player, and replaying the turn re-enters the same
  buggy code path with the same result.
- Replay logs and bug reports capture the failure for later, but do nothing for
  the six people sitting around the table right now.

## Goal

Give the table a way to force the game into the state it should be in, by
agreement, and keep playing.

Explicit non-goal: reducing the engine's error rate. That is worthwhile and
separate. This design assumes errors will happen and makes them survivable.

## Overview

Three capabilities, all gated behind a consensus vote:

- **patch** — correct a wrong value (position, defeat, gold, life counters, …)
- **unstick** — escape a wedged control flow (skip the pending input, abort the
  action, end the turn, fix the actor)
- **rewind** — return the whole game to an earlier decision

## Architecture

### Overrides are replay records

An override never pokes at `GameState` out of band. It is a new decision type in
the replay log, applied through one code path shared by live play and
reconstruction — exactly the pattern `cheat_gold` already establishes.

```json
{"type":"ov_patch","r":3,"t":2,"hero":"hero_arien","op":"move_entity","args":{"entity":"minion_4","hex":{"q":1,"r":-2,"s":1}},"voters":["hero_arien","hero_wasp"]}
{"type":"ov_unstick","r":3,"t":2,"hero":"hero_arien","op":"abort_action","voters":[...]}
{"type":"ov_rewind","r":3,"t":2,"hero":"hero_arien","to":47,"voters":[...]}
```

A new module `engine/overrides.py` owns the op registry: each named op has a
Pydantic arg model and an `apply(state, args)`. `replay._apply_decision` gains
three branches that call into it. **Nothing else may mutate state for an
override.** That single-path rule is what keeps saves, replays, and rewinds in
agreement, and it is directly testable: round-trip a game through replay after
an override and assert state equality.

`voters` is recorded for auditability. It does not affect reconstruction.

### The log stays append-only

`ov_rewind` does **not** truncate the log. It is a record meaning *the cursor
moves back to N*. Reconstruction handles it by seeking backward —
`ReplayCursor.seek` already rebuilds from the seed on a backward move — and live
play keeps appending after it. The superseded segment stays in the file forever.

This matters because truncation would destroy the evidence of the bug that
caused the rewind, on the one artifact you would want when debugging it. It also
preserves the append-only property that `ReplayRecorder` is built around.

Cost: after a rewind, a decision index is a position in a log containing dead
segments, so "decision 47" and "the 47th thing that actually happened" diverge.
Bug reports are unaffected — `count_replay_decisions` counts lines, `seek`
applies lines in order, and `ov_rewind` is just another line. The replay-viewer
UI should render a rewind as a visible marker rather than hiding the dead
segment.

Bug reporting stays manual and player-initiated. No override files a report.

**Implementation subtlety.** `ov_rewind` is the one record that cannot be
handled inside `_apply_decision(session, decision)`, because it changes the
*cursor*, not the session — and the caller (`ReplayCursor.seek`'s
`while self.cursor < target` loop, and `replay_game`'s equivalent) owns the
cursor. Applying a rewind therefore has to be handled one level up, in the
driving loop: on encountering `ov_rewind`, rebuild from the seed and re-apply
records `0..N` (skipping any nested rewinds already accounted for), then
continue forward from the record after the rewind. The plan must settle this
control flow explicitly; a naive rewind-inside-apply either no-ops or loops
forever.

### Live rewind

Rewind cannot be *applied* to a session; it *replaces* one. To rewind to index N:

1. `build_session_from_setup` + apply the N-record prefix
2. swap `ManagedGame.session`
3. drop `last_result` and any in-flight request id
4. append the `ov_rewind` record, re-save, re-broadcast

Granularity is any decision index. The proposal payload carries the target
index; presenting a readable decision list is frontend work.

## Consensus protocol

**Location.** Pending proposals live on `ManagedGame`, not `GameState`. A
proposal is coordination, not game state — it does not belong in saves, views,
or the replay log. Only the outcome is recorded. Unresolved proposals die on
server restart; re-propose. `GameState` and the client contract are untouched by
the negotiation half of the feature.

**Threshold.** Strictly more than half of the players connected *at proposal
time*, snapshotted. The proposer auto-counts as yes. Snapshotting prevents a
flaky connection from silently flipping an outcome. In a 2-player game, majority
of 2 is 2 — both must agree. Spectators never vote. One open proposal at a time.

**Timeout.** Because the threshold is snapshotted, a mid-vote disconnect can make
it unreachable. Proposals expire after 120s (configurable). Expiry is a
rejection: an override nobody actively agreed to must not apply.

**Messages** (WebSocket, via the existing `msg_type` dispatch in `ws.py`):

| Inbound | Broadcast |
|---|---|
| `PROPOSE_OVERRIDE` | `OVERRIDE_PROPOSED` |
| `VOTE_OVERRIDE` | `OVERRIDE_UPDATED` (tally) |
| `CANCEL_OVERRIDE` (proposer only) | `OVERRIDE_RESOLVED` (applied / rejected / expired) |

On approval, under `ManagedGame.lock`: apply via the op registry → append the
replay record → `registry.save_game()` → `broadcast()`. Same ordering as every
other mutation path, inheriting auto-save and player-scoped views. These join
`MUTATION_MESSAGE_TYPES` only at the apply step; a vote alone mutates nothing.

**Turn clock.** The active timer pauses while a proposal is open and resumes on
resolution. This departs from the `rollback()` precedent of never refunding
elapsed time, deliberately: a 120s negotiation is not the active player's doing.

**REST parity.** Propose/vote are WebSocket-only. A REST client cannot
meaningfully join a 120-second live negotiation without polling; endpoints
nobody can use well are ceremony, not parity.

## Patching while an input request is pending

The main engine hazard. A step awaiting input is pushed *back onto the stack*
(`handler.py:76-84`) with `pending_request_id` set. Re-running `process_stack()`
pops the same step, calls `resolve()` again, and recomputes filters and options
against current state. Line 81-83 reuses the stored id — a branch that exists so
`persistence.py` can re-emit an identical request after restart. So "apply,
re-derive, re-broadcast" is a path the codebase already depends on.

Three cases:

**Stale option list — fixed by re-deriving.** A `SelectStep` offers enemies in
range 2: `minion_4, minion_7`. A patch moves `minion_7` five hexes away. Without
re-derivation every client still highlights an illegal target. Re-running
`process_stack` drops it.

**Stale answer accepted — not fixed by re-deriving.** A player's click is
already in flight when the patch lands. `submit_input` validates only that
`response.request_id == current_step.pending_request_id`, and the id was
deliberately preserved, so the stale answer is accepted and the player spends a
turn on a decision made against a board that has changed. Therefore: **bump the
request id after any patch** rather than reusing it. The existing mismatch check
then rejects the in-flight answer with an error clients already handle. This is
a deliberate departure from the persistence convention.

**Stale captured context — not patchable.** An earlier step wrote
`context["target_id"] = "minion_4"`; the current step only asks where to push
it. Patching `minion_4` off the board cannot be repaired by re-deriving the
current step, because the stale value lives upstream in `execution_context`.

The rule: **patches repair values, unstick repairs control flow, and when a
patch would invalidate mid-action context, the answer is unstick.**

**To verify during planning:** re-running `resolve()` requires idempotence, and
`steps/base.py:72` notes `resolve()` writes counters and flags back onto `self`.
A step incrementing on every call would double-count. Persistence already leans
on this globally, so it likely holds, but auditing steps that mutate `self` in
`resolve()` is real work in the plan, not a footnote.

## Primitive catalogue

A **closed whitelist**, not generic JSON-patch over `GameState`. Generic
patching would permit states no legal play could produce, and downstream
invariants — occupancy cache, card identity unification, multi-piece bindings —
assume reachable states.

### `ov_patch`

| Op | Args | Fixes |
|---|---|---|
| `move_entity` | entity_id, hex | Refused legal move, wrong distance |
| `remove_entity` | entity_id | A unit that should have been defeated wasn't |
| `place_entity` | entity_id, hex | Wrongly defeated unit, or a bad respawn |
| `set_life_counters` | team, value | Wrong LC loss on hero defeat or lane push |
| `set_gold` / `set_level` | hero_id, value | Miscredited bounty or upgrade |
| `add_marker` / `remove_marker` | marker_type, target | Wrong marker attribution |
| `add_effect` / `remove_effect` | effect ref | Spurious buff/debuff, or early expiry |
| `move_card` | hero_id, card_id, zone | Card stuck in the wrong zone |
| `set_wave_counter` | lane_id, value | Lane push scored wrong |
| `set_tie_breaker_team` | team | Wrong coin face (Ignatia) |

Note: this game has **no HP**. `Unit`/`Minion`/`Hero` carry no health field;
combat resolves attack against defense and a unit is either defeated or not.
"Calculated something wrong" means a wrong defeat outcome, movement reach, stat
total, gold/level award, or lane push — not a wrong number of hit points.

`set_gold` supersedes the existing `cheat_gold` record, which stays as a legacy
branch in `_apply_decision` so old replays keep loading.

### `set_life_counters` is special

Life counters are the win condition, so this op alone can end or un-end a game.

- Setting a team to 0 must re-run the endgame check, not leave a state where a
  team is dead but `winner` is unset.
- Raising LC above 0 on a finished game is the only patch that resurrects one.
  It must clear `winner`, `individual_winner_id`, and `victory_condition`, and
  move `phase` off `GAME_OVER` — `process_stack` returns immediately on
  `GAME_OVER` (`handler.py:52-53`), so without that the game stays frozen
  regardless of the counter.

This is the highest-value case in the feature: an engine bug that wrongly *ends*
a session is the one a table cannot house-rule around. It gets dedicated tests.

`starting_life_counters` is immutable under override — it is setup data and the
baseline the double-lane endgame coordinator compares against.

### `ov_unstick`

| Op | Effect |
|---|---|
| `skip_input` | Submit the existing `"SKIP"` sentinel to the pending request |
| `abort_action` | Route through the existing `_clear_after_abort` path |
| `end_turn` | Force `FinalizeHeroTurnStep` |
| `force_actor` | Set `current_actor_id` when turn order itself went wrong |

`abort_action` reuses `_clear_after_abort` rather than reimplementing it, so
defense/reaction sequences unwind correctly for free.

### Excluded

Direct `execution_stack` editing and arbitrary field paths. If the stack is
genuinely wedged, `abort_action` or a rewind is the answer; hand-editing a LIFO
stack of Pydantic steps is not something a table can reason about at 11pm.

## Atomicity and conventions

Each op applies to a deep copy, then runs `rebuild_occupancy_cache()` and
`state.validator`. If the result is invalid the whole override is rejected with
an error and nothing commits. No half-applied patches.

Every positional op goes through `state.get_position()` / `get_piece_ids()` /
placement helpers and never touches `entity_locations` directly. Otherwise
`move_entity` on `hero_razzle` silently does nothing — Razzle has no board
position of its own, only its pieces do.

## Testing

- **Replay parity** (the load-bearing test): apply each op live, reconstruct the
  game from its replay log, assert state equality.
- **Rewind determinism**: rewind, play forward differently, reconstruct, compare.
- **Consensus protocol**: threshold snapshotting, expiry-as-rejection,
  spectator rejection, one-proposal-at-a-time, proposer auto-yes.
- **Pending-input interaction**: stale option list re-derived; stale answer
  rejected via bumped request id.
- **Life counters**: game ends on drop to 0; game resumes on raise from 0,
  including `phase` leaving `GAME_OVER`.
- **Atomicity**: an op producing an invalid state commits nothing.
- **Multi-piece**: `move_entity` on `hero_razzle` behaves correctly.
- **Schema completeness**: every registered op appears in
  `GET /overrides/schema` with a valid arg schema.
- **History masking**: a player requesting the decision history never sees an
  opponent's facedown committed card id; spectators see the fully-masked form.

## Client discoverability

A client must be able to build any valid override request without hardcoding
the catalogue. `build_view()` already supplies most argument *values* — board,
units, life counters, effects, markers, scoped cards. Three gaps remain.

### 1. The op catalogue — `GET /overrides/schema` (static)

Auto-derived from the `engine/overrides.py` registry via `model_json_schema()`
on each op's Pydantic arg model. Returns op names, arg schemas, the family
(`patch` / `unstick`), and a human-readable label and description per op.

Auto-derivation is the point: a hand-written catalogue in the guide drifts the
first time an op is added. A test asserts every registered op appears in the
endpoint output.

Static and game-independent, so clients fetch it once and cache it.

### 2. Rewind history — `GET /games/{game_id}/overrides/history`

"Rewind to any decision index" is unusable unless the client can show what the
indices *mean*. The backend owns the replay format, so it renders the list:
decision index, round, turn, acting hero, kind, and a human label
("Arien committed Liquid Leap", "Wasp attacked minion_4"), plus a flag marking
records superseded by an earlier rewind so the viewer can grey them out.

**This endpoint leaks hidden information if built naively, and must not be.**
Replay `commit` records contain card ids — including cards committed facedown
that opponents are not entitled to see. A raw decision list handed to a player
is a direct violation of the facedown-card visibility rule. The endpoint is
therefore player-scoped like every other client-facing surface: card identity is
masked using the same `current_*` / identity-hidden guard the view uses, so a
label reads "Wasp committed a card" until that card is public. Spectators get
the fully-masked form. The omniscient variant already exists for the offline
replay debugger (`reveal_all`) and must never be reachable from this endpoint.

**Rewind depth is unrestricted.** Rewinding past a round boundary restores a
state in which cards are hidden again, though every player has already seen
them. This is deliberately not the engine's problem: the override passed a vote,
and a table that agrees to go back that far has accepted the consequence. The
engine does not second-guess an agreed decision, and no clamp or warning gate
applies.

Note the interaction with majority rule: a deep rewind can carry on a 3-2 vote,
over the objection of players who preferred the original outcome. That is a
property of majority consensus generally, not of rewind specifically.

### 3. Proposal payloads

`OVERRIDE_PROPOSED` carries everything needed to render a vote prompt without a
follow-up fetch: proposal id, proposer hero id, family, op, args, a
server-rendered human summary of the effect, the snapshotted eligible-voter
list, the threshold, the current tally, and an absolute expiry timestamp
(so clients render a countdown without clock-skew guesswork).

`OVERRIDE_UPDATED` carries proposal id and updated tally.
`OVERRIDE_RESOLVED` carries proposal id, outcome
(`applied` / `rejected` / `expired` / `cancelled`), the final tally, and on
failure a structured reason — machine-readable code plus message — so a client
can distinguish "validation rejected this patch" from "you were outvoted".

Spectators receive all three broadcasts (they can watch the negotiation) but
`VOTE_OVERRIDE` from a spectator is rejected, consistent with the existing
spectator guard in `ws.py:619`.

## Client contract impact

New WebSocket message types (`PROPOSE_OVERRIDE`, `VOTE_OVERRIDE`,
`CANCEL_OVERRIDE`, `OVERRIDE_PROPOSED`, `OVERRIDE_UPDATED`, `OVERRIDE_RESOLVED`)
and their payload shapes must be documented in
`docs/CLIENT_INTEGRATION_GUIDE.md`, along with the two new REST endpoints
(`GET /overrides/schema`, `GET /games/{game_id}/overrides/history`).

Response models for both endpoints are added to `server/models.py`. No changes
to `domain/input.py` or `build_view()` output — override state deliberately
stays off the view, since proposals are rare and adding them would inflate every
state broadcast in every game that never uses the feature.
