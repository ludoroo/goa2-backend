# Client Integration Guide

This guide covers everything a frontend developer needs to connect to the GoA2 backend API.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [REST API Reference](#rest-api-reference)
4. [WebSocket Protocol](#websocket-protocol)
5. [Game Flow](#game-flow)
6. [Understanding the Game View](#understanding-the-game-view)
7. [Handling Input Requests](#handling-input-requests)
8. [Events](#events)
9. [Persistence & Reconnection](#persistence--reconnection)
10. [Character Draft Lobby](#character-draft-lobby)
11. [Consensus Overrides](#consensus-overrides)
12. [Error Handling](#error-handling)

---

## Quick Start

### 1. Start the server

```bash
PYTHONPATH=src uv run uvicorn goa2.server.app:create_app --factory --reload
```

The server runs on `http://localhost:8000` by default.

### 2. Create a game

```bash
curl -X POST http://localhost:8000/games \
  -H "Content-Type: application/json" \
  -d '{
    "map_name": "forgotten_island",
    "red_heroes": ["arien"],
    "blue_heroes": ["knight"]
  }'
```

Response:

```json
{
  "game_id": "a1b2c3d4e5f6",
  "player_tokens": [
    {"hero_id": "hero_arien", "token": "abc123..."},
    {"hero_id": "hero_knight", "token": "def456..."}
  ],
  "spectator_token": "ghi789..."
}
```

Save these tokens — they are the only way to authenticate.

Games are untimed unless creation (or draft-lobby settings) includes an explicit
`time_control` object:

```json
{
  "planning_allowance_seconds": 60,
  "resolution_allowance_seconds": 45,
  "initiative_bonus_seconds": 15,
  "response_grant_seconds": 15,
  "initial_time_bank_seconds": 120,
  "time_bank_increment_seconds": 10,
  "max_time_bank_seconds": 240,
  "upgrade_allowance_seconds": 45,
  "automatic_turn_limit": 2
}
```

Time fields are integers from 0 through 86,400, `automatic_turn_limit` is an
integer from 0 through 100, and the maximum Time Bank must be at least its
initial value. `initiative_bonus_seconds` is a one-shot bonus for the first
primary Resolution actor in each shared turn. It is spendable only by that
actor's primary Resolution clock; it is not part of the Time Bank and cannot
be spent during Response prompts. The automatic limit counts consecutive
shared turns completed without an accepted human gameplay decision; `0`
disables inactivity suspension. A timed game stays in its public ready check until
every player readies through `POST /games/{game_id}/ready` with
`{"ready":true}`, or the WebSocket `SET_READY` message below. Game decisions
are rejected until then. Disconnecting does not pause a running clock.

### 3. Get the game view

```bash
curl http://localhost:8000/games/a1b2c3d4e5f6 \
  -H "Authorization: Bearer abc123..."
```

### 4. Connect via WebSocket

```
ws://localhost:8000/games/a1b2c3d4e5f6/ws?token=abc123...
```

On connection, the server immediately sends a `STATE_UPDATE` message with the current game view.

---

## Authentication

The server uses bearer tokens generated at game creation. There are no usernames, passwords, or sessions — tokens are the sole identity.

### Token types

| Type | Created per | Access level |
|------|-------------|-------------|
| Player token | Each hero in the game | Full: view own cards, commit cards, submit input |
| Spectator token | One per game | Read-only: view game state (no facedown cards visible) |

### REST authentication

Include the token in the `Authorization` header:

```
Authorization: Bearer <token>
```

All endpoints except `POST /games` and `GET /heroes` require authentication. The server validates that the token belongs to the game specified in the URL path.

### WebSocket authentication

Pass the token as a query parameter:

```
ws://host/games/{game_id}/ws?token=<token>
```

Invalid tokens are rejected with WebSocket close code `4001`. Tokens that don't match the game ID are rejected with close code `4003`.

---

## REST API Reference

### `GET /heroes`

List released hero IDs. No authentication required.

Pass the optional `include_playtest=true` query parameter to include registered
playtest heroes. Playtest heroes are excluded by default.

**Response:** `200 OK`

```json
["Arien", "Bain", "Brogan"]
```

With `include_playtest=true`, the response also includes playtest hero IDs such
as `"Cordelia"`.

### `GET /heroes/metadata`

List released heroes with pre-game selection metadata. No authentication required.

Pass the optional `include_playtest=true` query parameter to include registered
playtest heroes. Release status and difficulty are independent: for example,
Cordelia is a playtest hero with `difficulty_stars: 2`.

`difficulty_stars` describes how difficult the hero is to play and is only relevant before game creation.

**Response:** `200 OK`

```json
[
  { "id": "Arien", "difficulty_stars": 1 },
  { "id": "Bain", "difficulty_stars": 2 }
]
```

With `include_playtest=true`, the response also includes:

```json
{ "id": "Cordelia", "difficulty_stars": 2 }
```

### `POST /games`

Create a new game. No authentication required.

**Request body:**

```json
{
  "map_name": "forgotten_island",
  "red_heroes": ["arien"],
  "blue_heroes": ["knight"],
  "cheats_enabled": false,
  "game_type": "LONG"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `map_name` | string | `"forgotten_island"` | Map to use (must exist in `data/maps/`) |
| `red_heroes` | string[] | required | Hero IDs for the red team |
| `blue_heroes` | string[] | required | Hero IDs for the blue team |
| `cheats_enabled` | boolean | `false` | Enable cheats for this game (unlocks gold cheat API) |
| `game_type` | string | `"LONG"` | `"QUICK"` or `"LONG"`. Controls wave and life counter setup (see below) |
| `time_control` | object/null | `null` | Optional timed-match configuration shown in the quick-start example above |

**Game types:**

| Game Type | Players | Wave Counters | Life Counters |
|-----------|---------|---------------|---------------|
| `QUICK` | 4 | 3 | 4 |
| `QUICK` | 5 | 3 | 4 |
| `QUICK` | 6 | 3 | 5 |
| `LONG` | 4–5 | 5 | 6 |
| `LONG` | 6 | 5 | 8 |

Omitting `game_type` defaults to `"LONG"` (standard game).

**Response:** `201 Created`

```json
{
  "game_id": "a1b2c3d4e5f6",
  "player_tokens": [
    {"hero_id": "hero_arien", "token": "abc123..."},
    {"hero_id": "hero_knight", "token": "def456..."}
  ],
  "spectator_token": "ghi789..."
}
```

### `GET /games/{game_id}`

Get the current game view for the authenticated player.

**Response:** `200 OK`

```json
{
  "view": { ... },
  "input_request": null,
  "awaiting_input": ["hero_wasp"],
  "winner": "RED"
}
```

The `view` object contains the player-scoped game state (see [Understanding the Game View](#understanding-the-game-view)). The `input_request` is present only when the authenticated hero is allowed to answer it. Opponents and spectators receive `null`. Team-level requests are visible only to that team's heroes; simultaneous upgrade requests contain only the authenticated hero's entry.

The `awaiting_input` array names every hero the pending request is waiting on, and is sent to **all** recipients — including those whose `input_request` was withheld. Use it to render "waiting for X" indicators; use `input_request` to decide whether *you* may answer. See [Who is being waited on](#who-is-being-waited-on).

The `winner` key is only present when game has ended (`view.phase === "GAME_OVER"`). Its value is `"RED"` or `"BLUE"` for a team victory, or the winning hero ID (for example, `"hero_cutter"`) for an individual victory. Check for its presence with `response.get("winner")` rather than assuming it exists.

### `POST /games/{game_id}/ready`

Set the authenticated hero's readiness during a timed match's initial ready
check or an inactivity-resume ready check. Sending `false` removes that hero
from the ready set. The final required `true` starts the match clocks.

**Request body:**

```json
{
  "ready": true
}
```

**Response:** `200 OK` — returns `GameViewResponse` with the updated public
clock snapshot. The mutation also broadcasts a `STATE_UPDATE` to every
connected game WebSocket.

**Error conditions:**

- `400` — The match is untimed, or its clock is already running/finished.
- `403` — A spectator attempted to change readiness.

### `POST /games/{game_id}/cards`

Commit a card during the PLANNING phase.

**Request body:**

```json
{
  "card_id": "arien_tidal_wave_1"
}
```

**Response:** `200 OK` — returns `ActionResultResponse` (see below).

### `POST /games/{game_id}/uncommit`

Take your committed card back into hand during the PLANNING phase (the
board-game "take-back"), e.g. to commit a different card while waiting for the
other players. For a two-card hero (Emmitt's *Alternative Timelines*)
take-backs are LIFO: a committed second card returns first, then the first
commit; any planning-done signal is cleared.

Allowed only while the phase is still PLANNING — the instant the last player
commits, revelation runs and further uncommits fail.

**Request body:** empty

**Response:** `200 OK` — returns `ActionResultResponse`; the card is back in
`hand` and `current_turn_card` reverts. `400` if there is nothing to take back
(no commit yet, or the player passed). `403` for spectators. `409` if the
phase is no longer PLANNING (all cards locked in).

### `POST /games/{game_id}/pass`

Pass during the PLANNING phase when the hero has no cards in hand (the hero
will not play a card this round). A hero who can still play a card must commit
one instead.

**Request body:** empty

**Response:** `200 OK` — returns `ActionResultResponse`. `400` if the hero has
cards in hand or already completed planning.

### `POST /games/{game_id}/planning-done`

Only relevant for a hero whose active ultimate allows playing two cards per
turn (Emmitt's *Alternative Timelines*, level >= 8). Such a hero's planning
does **not** close after the first commit: they may `POST .../cards` a second
time, or call this endpoint to declare they are playing only one card this
turn. Planning also auto-closes if their hand is empty after the first commit.

If the hero played two cards, immediately after revelation the server issues a
mandatory `SELECT_CARD` input request (routed to that hero) to retrieve one of
the two revealed cards back to hand; the other resolves normally. Both cards
are visible to all players until the choice is made (see `extra_turn_card`).
During PLANNING after the second commit, `current_turn_card` contains the latest
commit and `extra_turn_card` contains the first buffered commit. A single commit
does not populate `extra_turn_card`. Facedown-card visibility rules still apply.

**Request body:** empty

**Response:** `200 OK` — returns `ActionResultResponse`. `400` if the hero has
not committed a card yet.

### `POST /games/{game_id}/input`

Submit a response to an input request (e.g., selecting a unit, choosing a hex).

**Request body:**

```json
{
  "request_id": "<input_request.request_id>",
  "selection": "hero_knight"
}
```

`request_id` is required and must exactly match the pending input request. A
missing, stale, or mismatched ID is rejected without applying the selection.
The `selection` value depends on the input request type — it may be a string
(unit ID), a hex dict (`{"q": 0, "r": 1, "s": -1}`), an integer, or a card ID.

**Response:** `200 OK` — returns `ActionResultResponse`.

### `POST /games/{game_id}/advance`

Advance the game state without submitting input. Used when the engine needs to continue processing (e.g., transitioning between phases).

**Request body:** empty

**Response:** `200 OK` — returns `ActionResultResponse`.

### `POST /games/{game_id}/rollback`

Rollback the current actor's resolution to the action choice step. Only the player the current `InputRequest` is addressed to (its `player_id`) can rollback, and only when `can_rollback` is `true`. This is normally the current actor; under Hanu's ultimate (action control) the action's inputs are remapped to the controller, so the controller — not the controlled hero — owns the rollback.

**Request body:** empty

**Response:** `200 OK` — returns `ActionResultResponse`. The `input_request` will be the action choice prompt again.

**Error conditions:**
- `400` — No active resolution or no rollback snapshot available
- `403` — Not the player the current input is addressed to, or spectator token used

**When rollback is or is not available:**
- **Foreign input (segment boundary, not a freeze):** As soon as any input request during the current resolution is addressed to a player other than the resolution owner (e.g., an opponent prompted for defense card selection), the actor's pre-foreign snapshot is dropped and `can_rollback` is `false` on the foreign prompt. This is a segment boundary, not a permanent freeze: a later owner *actionable* prompt in the same resolution may re-anchor a fresh snapshot and rollback to that post-foreign prompt is allowed. Rollback never restores past a foreign player's committed decision — at most the current post-foreign segment is undone. Under Hanu's ultimate (action control), input remapped back to the controller is *not* foreign — the controller still owns rollback for the controlled action.
- **`ConfirmResolutionStep` after a foreign segment:** The confirmation step at the end of resolution never *creates* a fresh snapshot on its own. If a valid actionable snapshot is still live (e.g., a post-foreign owner prompt re-anchored earlier in this resolution), confirm inherits it and keeps `can_rollback = true`, so rolling back returns to that actionable prompt. If a foreign segment boundary dropped the snapshot and no owner actionable prompt re-anchored before confirm runs, the confirmation step **auto-completes** — no `input_request` is sent to the client and the resolution ends silently. Clients therefore never see a bare "click Confirm" prompt whose only choice is confirmation with rollback disabled.
- **Hidden-info reveals (segment boundary, not freeze):** Hidden information being revealed (e.g. land mine reveals and card-color guess reveals) creates a segment boundary with the same semantics as foreign input. The pre-reveal snapshot is invalidated so the actor cannot re-guess with the newly-revealed information; the first owner *actionable* prompt after the reveal (e.g., the forced discard from a blast, or We're Not Done Yet's repeat-vs-coins choice) re-anchors at the post-reveal state. If the only owner prompt after the reveal is `ConfirmResolutionStep`, confirm auto-completes and no `input_request` is sent.
- **Timer timeout (permanent freeze):** When the acting player's turn timer expires, rollback is permanently frozen for the rest of that turn. `can_rollback` stays `false` on every prompt for the remainder of the resolution and `ConfirmResolutionStep` is auto-confirmed. This is the only case where rollback is fully disabled mid-resolution.

### `POST /games/{game_id}/cheats/gold`

Give gold to a hero (cheats must be enabled and game must be in PLANNING phase).

**Request body:**

```json
{
  "hero_id": "hero_arien",
  "amount": 5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `hero_id` | string | ID of the hero to give gold to |
| `amount` | integer | Amount of gold to give (must be positive) |

**Response:** `200 OK` — returns `ActionResultResponse` with a `GOLD_GAINED` event.

**Error conditions:**
- `403` — Cheats not enabled for this game, spectator token used, or not in PLANNING phase
- `404` — Hero not found
- `400` — Amount is not a positive integer

### `POST /games/{game_id}/bug-reports`

Submit a bug report for this game. The server links the report to the game's
replay log by recording the current decision index — the exact replay moment
the report was filed — so nothing about the game position is sent by (or
trusted from) the client.

**Auth:** any of the game's tokens (player or spectator).

**Request body:**

```json
{
  "title": "Arien Silver did not block skill",
  "description": "I played Sting and it let me hit a hero behind an obstacle."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Short summary, 1–120 characters after trimming (required) |
| `description` | string | Details, up to 4000 characters (optional, may be empty) |

**Response:** `201 Created`

```json
{
  "id": "br_1a2b3c4d",
  "game_id": "abc123",
  "title": "Arien Silver did not block skill",
  "description": "I played Sting and it let me hit a hero behind an obstacle.",
  "reporter_hero": "hero_wasp",
  "decision_index": 42,
  "round": 3,
  "turn": 2,
  "status": "open",
  "created_at": 1751470000.0,
  "resolved_at": null
}
```

`reporter_hero` is `null` when submitted with the spectator token.
`decision_index` is `null` in the unlikely case the replay log is missing.

**Error conditions:**
- `401` — Missing/invalid token
- `403` — Token belongs to a different game
- `404` — Game not found
- `422` — Title empty or too long, or description too long
- `429` — Report limit for this game reached (10)

### ActionResultResponse shape

All mutation endpoints return this shape:

```json
{
  "result_type": "INPUT_NEEDED",
  "current_phase": "RESOLUTION",
  "events": [
    {
      "event_type": "UNIT_MOVED",
      "actor_id": "hero_arien",
      "target_id": null,
      "from_hex": {"q": 0, "r": 0, "s": 0},
      "to_hex": {"q": 1, "r": -1, "s": 0},
      "metadata": {}
    }
  ],
  "input_request": {
    "type": "SELECT_HEX",
    "player_id": "hero_arien",
    "prompt": "Choose a hex to move to",
    "valid_hexes": [{"q": 1, "r": 0, "s": -1}]
  },
  "winner": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `result_type` | string | `INPUT_NEEDED`, `ACTION_COMPLETE`, `PHASE_CHANGED`, or `GAME_OVER` |
| `current_phase` | string | Current game phase (see [Game Flow](#game-flow)) |
| `events` | array | Recipient-scoped game events emitted during this action (see [Events](#events)) |
| `input_request` | object/null | Present only when the authenticated action performer may answer the pending request. It can be `null` even when `result_type` is `INPUT_NEEDED` if the action advanced to another player's decision |
| `awaiting_input` | array | Hero IDs the pending request is waiting on, unscoped; `[]` when none. Populated even when `input_request` is `null` — that pairing is exactly the "waiting on someone else" case (see [Who is being waited on](#who-is-being-waited-on)) |
| `winner` | string/null | `"RED"`/`"BLUE"` for a team victory or a hero ID for an individual victory when `result_type` is `GAME_OVER` |

---

## WebSocket Protocol

### Connection

```
ws://host/games/{game_id}/ws?token=<bearer_token>
```

On successful connection, the server sends an initial `STATE_UPDATE` message with the full game view.

A player token owns one live WebSocket; reconnecting with that token supersedes
the prior socket. The shared spectator token may be used by multiple concurrent
WebSockets, and every connected spectator receives broadcasts.

### Client-to-server messages

All messages are JSON with a `type` field:

#### `COMMIT_CARD`

```json
{
  "type": "COMMIT_CARD",
  "card_id": "arien_tidal_wave_1"
}
```

#### `UNCOMMIT_CARD`

Take the committed card back into hand during PLANNING (LIFO for a two-card
hero). See `POST /games/{game_id}/uncommit`. Reply: `ACTION_RESULT`; failure
(e.g. cards already locked in): `ERROR`.

```json
{
  "type": "UNCOMMIT_CARD"
}
```

#### `PASS_TURN`

```json
{
  "type": "PASS_TURN"
}
```

#### `FINISH_PLANNING`

Done-signal for a two-card-capable hero (Emmitt's ultimate) playing only one
card this turn. See `POST /games/{game_id}/planning-done`.

```json
{
  "type": "FINISH_PLANNING"
}
```

#### `SUBMIT_INPUT`

```json
{
  "type": "SUBMIT_INPUT",
  "request_id": "<input_request.request_id>",
  "selection": "hero_knight"
}
```

#### `GET_VIEW`

Request a fresh state update (available to both players and spectators):

```json
{
  "type": "GET_VIEW"
}
```

#### `PING`

Point at a board hex or a public card location on the tabletop. Pings are
ephemeral communication: they do not mutate or persist game state, advance a
clock, enter a replay, or produce a `STATE_UPDATE`. Only authenticated players
may send them; players and spectators receive them. The server rate-limits each
connection to one accepted ping every 450 ms.

```json
{
  "type": "PING",
  "target": {
    "kind": "HEX",
    "hex": {"q": 0, "r": -2, "s": 2}
  }
}
```

Card targets use their visible table location, never a card ID, so pointing at
a facedown committed card cannot expose its private identity:

```json
{
  "type": "PING",
  "target": {
    "kind": "CARD",
    "hero_id": "hero_arien",
    "zone": "PLAYED",
    "index": 1
  }
}
```

`zone` is one of `CURRENT`, `EXTRA`, `PLAYED`, `DISCARD`, `ULTIMATE`, or
`CAST`. `PLAYED`, `DISCARD`, and `CAST` require a zero-based `index`. The
server validates that the referenced hex or card location is currently on the
table and discards any client-supplied ping identity or card ID.

#### `SET_READY`

Timed matches only. Players may toggle readiness until the final player readies;
that final mutation starts every incomplete Planning clock together. Spectators
cannot ready.

```json
{
  "type": "SET_READY",
  "ready": true
}
```

The sender first receives a direct `READY_UPDATED` acknowledgement, followed
by the same `STATE_UPDATE` broadcast sent to every connected client:

```json
{
  "type": "READY_UPDATED",
  "hero_id": "hero_arien",
  "ready": true
}
```

#### `ROLLBACK`

Rollback the current actor's resolution to the action choice. Only the player the current input request is addressed to (its `player_id`) can send this, and only when `can_rollback` is `true`. Under Hanu's ultimate (action control) that is the controller, not the controlled hero.

```json
{
  "type": "ROLLBACK"
}
```

**Response:** `ACTION_RESULT` with the action choice input request.

#### `CHEATS_GOLD`

Give gold to a hero (cheats must be enabled and game must be in PLANNING phase):

```json
{
  "type": "CHEATS_GOLD",
  "hero_id": "hero_arien",
  "amount": 5
}
```

**Error responses:**
- `Cheats are not enabled for this game` — Cheats were not enabled at game creation
- `Expected phase PLANNING, but game is in RESOLUTION` — Gold cheat only works during PLANNING phase
- `Hero 'X' not found` — The specified hero_id does not exist
- `Amount must be a positive integer` — The amount must be > 0

### Server-to-client messages

#### `PING`

Broadcast immediately to every connected player and spectator after a valid
player ping. `hero_id` is derived from the authenticated sender token, and the
opaque `ping_id` lets clients animate overlapping/repeated pings independently.

```json
{
  "type": "PING",
  "ping_id": "3c64b4ccf0584ed2bd40df648c18a2fe",
  "hero_id": "hero_arien",
  "target": {
    "kind": "HEX",
    "hex": {"q": 0, "r": -2, "s": 2}
  }
}
```

#### `READY_UPDATED`

Direct acknowledgement of an accepted `SET_READY` message. It contains the
authenticated hero ID and the requested boolean value. A `STATE_UPDATE`
broadcast follows with the authoritative ready set and clock status.

#### `STATE_UPDATE`

Sent on connection, on `GET_VIEW` requests, and broadcast to all connected clients after any mutation:

```json
{
  "type": "STATE_UPDATE",
  "view": { ... },
  "input_request": { ... },
  "awaiting_input": ["hero_wasp"],
  "winner": "RED",
  "events": [ ... ]
}
```

The `input_request` key is present only when the receiving hero is allowed to answer the pending request. It is omitted for opponents and spectators. Team-level requests go only to that team, and simultaneous upgrade requests contain only the receiving hero's `players` entry. Check for its presence with `msg.get("input_request")` rather than assuming it exists.

The `awaiting_input` array is always present and identical for every recipient, naming the heroes the pending request is waiting on (`[]` when there is none). Unlike `input_request` it is not scoped, so observers can name the blocking player. See [Who is being waited on](#who-is-being-waited-on).

The `winner` key is only present when the game has ended (`view.phase === "GAME_OVER"`). Its value is `"RED"` or `"BLUE"` for a team victory, or the winning hero ID for an individual victory. Check for its presence with `msg.get("winner")` rather than assuming it exists.

The `events` key is only present on **broadcasts that follow a mutation**. It lets every connected client — including non-acting players and spectators — animate the action, not just the actor. Event metadata is projected independently for each recipient: hidden card IDs/names are `null` (or omitted from ID lists), and facedown mine placement reports `metadata.token_type: "mine"` to everyone except the token's owner. It is **absent** on the initial connection update and on `GET_VIEW` responses (there is nothing to animate), so treat it as optional with `msg.get("events", [])`. The view itself remains authoritative; events are for animation only.

#### `ACTION_RESULT`

Sent to the player who performed the action:

```json
{
  "type": "ACTION_RESULT",
  "result_type": "INPUT_NEEDED",
  "current_phase": "RESOLUTION",
  "events": [ ... ],
  "input_request": { ... },
  "awaiting_input": ["hero_wasp"],
  "winner": null
}
```

#### `ERROR`

```json
{
  "type": "ERROR",
  "detail": "Input expected from 'hero_knight', not 'hero_arien'"
}
```

### Broadcast behavior

After a mutation (`COMMIT_CARD`, `PASS_TURN`, `FINISH_PLANNING`, `SUBMIT_INPUT`):

1. The acting player receives an `ACTION_RESULT` message
2. **All** connected clients (including the acting player) receive a `STATE_UPDATE` broadcast with their player-scoped view, carrying a recipient-scoped projection of the action's `events`

This means the acting player gets both messages, with the same recipient-safe events on each — animate from one source only (the `STATE_UPDATE` broadcast is recommended, since every client receives it). Different players may receive different metadata for the same event when private information is involved. The authoritative state lives on `STATE_UPDATE.view`; events are animation hints.

`SET_READY` follows the same two-message pattern, using `READY_UPDATED` as its
direct response. A timeout discovered while processing a REST mutation is also
broadcast before that REST request completes, even when the late request is
rejected with `Decision already timed out`.

`PING` is the exception to the mutation pattern: it is itself broadcast to all
connections, including the sender, and has no direct acknowledgement or
following `STATE_UPDATE`.

### Spectator restrictions

Spectators can only send `GET_VIEW` messages. All other message types return an error:

```json
{"type": "ERROR", "detail": "Spectators can only GET_VIEW"}
```

---

## Game Flow

The game progresses through these phases:

```
PLANNING → REVELATION → RESOLUTION → CLEANUP → LEVEL_UP → PLANNING
                                                     ↓
                                                 GAME_OVER
```

### Phase descriptions

| Phase | Description | Client action |
|-------|-------------|---------------|
| `PLANNING` | Each player selects a card to commit (or passes). | Call `commit_card` or `pass_turn` for each hero. Once all heroes have committed/passed, the phase transitions automatically. |
| `REVELATION` | Cards are revealed (flipped faceup). | Call `advance` to progress. No player input needed. |
| `RESOLUTION` | Heroes act in initiative order. The engine pauses for input requests (selecting targets, movement hexes, etc.). | Respond to `input_request`s via `submit_input`. Call `advance` when `result_type` is `ACTION_COMPLETE` or `PHASE_CHANGED` to continue. |
| `CLEANUP` | Round-end bookkeeping (discard cards, reset effects). | Call `advance` to progress. |
| `LEVEL_UP` | Heroes upgrade cards if they've earned enough gold. May require input for upgrade choices. | Respond to any `input_request`s, then `advance`. |
| `GAME_OVER` | A victory condition has ended the game, such as depleted life counters or an individual hero victory. | Check the `winner` field. |

### Typical client loop

```
1. Check result_type from last response
2. If INPUT_NEEDED → render input_request options, wait for player choice, submit_input
3. If ACTION_COMPLETE → call advance to continue (REST only — see note)
4. If PHASE_CHANGED → update phase UI, call advance to continue
5. If GAME_OVER → show winner
```

**Note:** The `advance` action is only available via REST (`POST /games/{game_id}/advance`). There is no WebSocket equivalent. WebSocket-only clients can use `SUBMIT_INPUT` for input responses and `COMMIT_CARD`/`PASS_TURN` for planning, but must use REST for advance calls.

---

## Understanding the Game View

The `view` object returned by `GET /games/{game_id}` and WebSocket `STATE_UPDATE` messages has this structure:

```json
{
  "phase": "PLANNING",
  "round": 1,
  "turn": 1,
  "current_actor_id": null,
  "unresolved_hero_ids": ["hero_arien", "hero_knight"],
  "unresolved_cards": [
    { "hero_id": "hero_arien", "initiative": 7, "card": { ... } },
    { "hero_id": "hero_knight", "initiative": 5, "card": { ... } }
  ],
  "active_zone_id": "Mid",
  "battle_zones": { "lane_1": "Mid" },
  "wave_counters": { "lane_1": 5 },
  "cheats_enabled": false,
  "tie_breaker_team": "RED",
  "teams": {
    "RED": { ... },
    "BLUE": { ... }
  },
  "board": {
    "map": "forgotten_island",
    "tiles": { ... },
    "zones": { ... },
    "entity_locations": { ... }
  },
  "effects": [ ... ],
  "markers": { ... },
  "tokens": [ ... ],
  "board_entities": [ ... ],
  "hero_pieces": { ... },
  "card_guess": null,
  "card_reveal": null,
  "time_control": null,
  "clock": null
}
```

For a timed match, `time_control` is the immutable creation configuration and
`clock` is a public snapshot containing `status`, `server_now_ms`, the shared
`turn_key`, `ready_hero_ids`, active clock kind/targets, and every hero's
remaining Planning, Resolution, Initiative Bonus, Response, Upgrade, and Time
Bank milliseconds. The Initiative Bonus is granted when the first primary
Resolution request of a shared turn starts and is reset at the next shared
turn.
Clients should extrapolate running values from `server_now_ms`; the server does
not broadcast one-second ticks. Timeout outcomes arrive as `TIMER_EXPIRED`
events, while the authoritative state update contains the resulting automatic
decision.

After `automatic_turn_limit` consecutive fully automatic shared turns, the
clock status becomes `SUSPENDED_FOR_INACTIVITY` before the next Planning phase.
Deadline tasks stop and no more fallback choices are made. Every player must
ready again through the same REST endpoint or WebSocket message; the completed
ready check starts that pending shared turn with its normal fresh allowances.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `phase` | string | Current game phase |
| `round` | int | Current round number (starts at 1) |
| `turn` | int | Current turn within the round |
| `current_actor_id` | string/null | Hero currently acting during RESOLUTION |
| `unresolved_hero_ids` | string[] | Heroes that haven't acted yet this round |
| `unresolved_cards` | object[] | Cards in resolution order (highest initiative first). Each entry: `{hero_id, initiative, card}`. Only populated during RESOLUTION phase; empty array otherwise. Ties broken by `tie_breaker_team`. Recalculated dynamically — order may change between actions due to modifiers. |
| `active_zone_id` | string/null | Legacy single-lane field: the current Battle Zone when the game has exactly one lane, `null` otherwise. Prefer `battle_zones`. |
| `battle_zones` | object | Current Battle Zone per lane (`lane_id -> zone_id`). Single-lane maps have one entry (`"lane_1"`). |
| `wave_counters` | object | Remaining Wave counters per lane (`lane_id -> int`). |
| `cheats_enabled` | boolean | Whether cheats are enabled for this game |
| `tie_breaker_team` | string | Team that currently wins ties (`"RED"` or `"BLUE"`) |
| `tokens` | object[] | Tokens currently on the board (see [Tokens](#tokens)) |
| `board_entities` | object[] | Non-unit, non-token board entities currently known to the game (see [Board Entities](#board-entities)) |
| `hero_pieces` | object | Stable board pieces for multi-piece heroes (see [Hero Pieces](#hero-pieces)) |
| `card_guess` | object/null | Public physical state while a card-color guess is active: `{guesser_id, attempts}`. Each attempt has `attempt`, `victim_id`, and either `card: null` while facedown or the complete revealed card plus `guessed_color`, `actual_color`, and `correct`. This is the authoritative source for rendering the guess — it survives reconnects and reveals a card only once it is flipped. A completed reveal stays in the view through the following hero's turn, then clears, so clients do not need to latch it from events. |
| `card_reveal` | object/null | Public direct faceup reveal state: `{revealer_id, target_unit_id, owner_id, card, tier_value, discarded}`. Tier values are Gold/Silver/untiered = 0, I = 1, II = 2, III = 3, IV = 4. `card` is the complete public card face even when the retained card has returned to an otherwise-hidden hand. The state survives reconnects, stays through the following hero's turn, then clears. |
| `time_control` | object/null | Immutable time-control values, or null for an untimed match |
| `clock` | object/null | Public authoritative clock snapshot, or null for an untimed match |

### Team data

Each team contains:

```json
{
  "color": "RED",
  "life_counters": 4,
  "heroes": [ ... ],
  "minions": [ ... ]
}
```

### Hero data

```json
{
  "id": "hero_arien",
  "name": "Arien",
  "title": "Tideshaper",
  "team": "RED",
  "level": 1,
  "gold": 0,
  "items": [],
  "wish_cast_count": 0,
  "rune_slots": {},
  "hand": [ ... ],
  "deck": [ ... ],
  "spellbook": null,
  "cast_spells": [],
  "played_cards": [ ... ],
  "current_turn_card": null,
  "extra_turn_card": null,
  "can_commit_second_card": false,
  "discard_pile": [ ... ],
  "ultimate_card": null
}
```

For heroes that can commit two cards, `extra_turn_card` also exposes the first
buffered card during PLANNING after the second card is committed;
`current_turn_card` is the latest commit. Both remain facedown, so only the
owning hero receives their identities.

**`rune_slots`** (object, `{slot_index: rune_type}` with string keys `"1"`-`"4"`) —
Snorri's rune slots. Absent/empty slots are simply missing keys (default `{}`).
Rune type is one of `"axe"`, `"bird"`, `"anvil"`, `"horn"`. Persistent
(survives round-end card cleanup) and **public to all viewers**, including
opponents and spectators — always populated the same way regardless of
`for_hero_id`.

**`wish_cast_count`** (integer) — public, persistent, per-caster progress for
Gydion's *Wish*. The caster's team wins after their third Wish finishes
resolving its selected spell.

**`can_commit_second_card`** (bool) — relevant only to a hero whose active
ultimate lets them play two cards per turn (Emmitt's *Alternative Timelines*,
level >= 8). It is `true` while, during the PLANNING phase, this hero has
committed their first card and may still either `POST .../cards` a second time
or call `.../planning-done`. It flips back to `false` once the second card is
committed, planning-done is called, the hand empties, or planning ends. For
secrecy it is **only ever `true` in the requesting hero's own view** — it is
always `false` for opponents and for spectators, so use it to drive the local
player's "commit second card / done" UI, not opponents'.

**`spellbook`** (array, `{ "count": N }`, or null) — heroes with spell cards
show prepared spells as complete card objects in their owner's view and in
offline `reveal_all` views. Opponents and spectators receive only
`{ "count": N }`, so prepared identities never leak. Heroes without spells
return `null`.

**`cast_spells`** (array) — spells currently outside the spellbook. They are
faceup public information and contain complete card objects in every player
and spectator view. Heroes without spells return an empty array.

### Card data

Each card object in the view contains:

| Field | Type | Facedown | Description |
|-------|------|----------|-------------|
| `id` | string | hidden | Unique card ID |
| `name` | string | hidden | Card display name |
| `image_id` | string | hidden | Frontend image identifier (e.g. `"BlueIIA"`, `"Gold"`, `"Ultimate"`). Maps to the card image asset filename |
| `tier` | string | shown | `"I"`, `"II"`, `"III"`, `"IV"`, or `"UNTIERED"` |
| `color` | string\|null | shown | `"RED"`, `"BLUE"`, `"GREEN"`, `"GOLD"`, `"SILVER"`, `"PURPLE"`, or `null` |
| `primary_action` | string\|null | shown | Primary action type: `"ATTACK"`, `"SKILL"`, `"MOVEMENT"`, `"DEFENSE"`, `"DEFENSE_SKILL"`, or `null` |
| `primary_action_value` | int\|null | shown | Value for the primary action (has value for ATTACK/MOVEMENT; null or value for SKILL/DEFENSE/DEFENSE_SKILL depending on card) |
| `secondary_actions` | object | shown | Map of action types to values (e.g. `{"DEFENSE": 3, "MOVEMENT": 2}`). Always includes `HOLD` |
| `effect_id` | string\|null | shown | ID for looking up card effect logic |
| `effect_text` | string | shown | Human-readable card text |
| `initiative` | int | shown | Initiative value for turn ordering (0 when facedown) |
| `state` | string | shown | Current card state: `"HAND"`, `"DECK"`, `"DISCARD"`, `"UNRESOLVED"`, `"RESOLVED"`, `"ITEM"`, `"PASSIVE"`, or `"RETIRED"` |
| `is_facedown` | bool | shown | Whether this card is hidden from opponents |
| `is_ranged` | bool | hidden | Whether the card is ranged |
| `range_value` | int\|null | hidden | Max distance when ranged |
| `radius_value` | int\|null | hidden | Area of effect radius |
| `item` | string\|null | hidden | When visible, which stat the card boosts as an item: `"ATTACK"`, `"DEFENSE"`, `"MOVEMENT"`, `"INITIATIVE"`, `"RANGE"`, `"RADIUS"`. Always `null` in a masked facedown-card view |
| `is_active` | bool | shown | Whether the card's active effect is available (tapped/un-tapped) |
| `spell_rank` | int | hidden | Spellbook rank. Present only on complete spell-card views; prepared opponent spells are count-only and never expose it. |

Spell cards use the additional states `"SPELLBOOK"` (prepared and facedown)
and `"OUTSIDE_SPELLBOOK"` (spent and faceup).

**Important:** `played_cards` is a fixed-position array where:
- Turn 1 card → `played_cards[0]`
- Turn 2 card → `played_cards[1]`
- Turn 3 card → `played_cards[2]`
- etc.

When a card is removed (discarded, returned to hand, etc.), its position becomes `null` but subsequent cards fill their correct turn-based positions:

```json
"played_cards": [
  { "id": "card_1", ... },  // Turn 1 card (position 0)
  null,                        // Turn 2 card was removed
  { "id": "card_3", ... },  // Turn 3 card (position 2, not 1)
  { "id": "card_4", ... }   // Turn 4 card (position 3)
]
```

Positions reset to empty at the start of each round.

### Card visibility rules

The view is player-scoped — what you see depends on your token:

- **Your hero's cards:** Full details visible for all cards (hand, deck, played, current turn card, ultimate)
- **Other heroes' FACEUP cards:** Full details visible (id, name, tier, action, is_ranged, range_value, radius_value, etc.)
- **Other heroes' FACEDOWN cards:** Partial details - hides `id`, `name`, `image_id`, `is_ranged`, `range_value`, `radius_value`, and the printed `item` stat (`item` remains present but is `null`). Other card-face values use their masked `current_*` representation; `state`, `is_facedown`, and the public `is_active` orientation remain visible
- **Other heroes' hand:** Empty array `[]` (no cards visible at all in hand)
- **Deck of other heroes:** Shows `{"count": N}` instead of card details
- **Discard piles:** Always fully visible (public information) — except facedown cards, see below
- **FACEDOWN cards in `discard_pile` / `played_cards`:** Masked for **every** viewer, the owner included. A facedown card outside the hand has lost its type, color and actions per the rulebook, so it is hidden information for everyone. It renders in the same partial shape as an opponent's facedown card (no `id`/`name`/`image_id`; `is_facedown: true`) — draw a card back. Takahide's Bushido is the effect that produces them; they turn faceup again when they return to hand at end of round.

### Board structure

```json
{
  "map": "forgotten_island",
  "tiles": {
    "0_0_0": {
      "hex": {"q": 0, "r": 0, "s": 0},
      "zone_id": "zone_center",
      "is_terrain": false,
      "occupant_id": "hero_arien",
      "spawn_point": null
    }
  },
  "zones": {
    "zone_center": {
      "id": "zone_center",
      "neighbors": ["zone_north", "zone_south"],
      "spawn_points": [ ... ]
    }
  },
  "entity_locations": {
    "hero_arien": {"q": 0, "r": 0, "s": 0},
    "hero_knight": {"q": 2, "r": -1, "s": -1}
  }
}
```

Hex coordinates use the cube coordinate system: `q + r + s = 0`.

`map` is the identifier of the loaded map (the map file's stem, e.g.
`"forgotten_island"`, `"narrow_passages"`). It is an empty string for boards
built without a map file (ad-hoc/test boards).

`entity_locations` is the authoritative source for unit positions.

### Hero Pieces

Some heroes, currently Razzle, exist on the board as up to 4 identical pieces
with stable IDs like `hero_razzle_piece_1`, while remaining one player-level
hero. The view contains a top-level `hero_pieces` object:

```json
{
  "hero_razzle_piece_1": {
    "owner_hero_id": "hero_razzle",
    "team": "RED",
    "position": {"q": 0, "r": 0, "s": 0}
  },
  "hero_razzle_piece_2": {
    "owner_hero_id": "hero_razzle",
    "team": "RED",
    "position": null
  }
}
```

`position: null` means the piece is in supply. Render every on-board piece as
the owning hero; pieces of the same owner are visually interchangeable.
`SELECT_UNIT` options may contain piece IDs, and clients should submit that
piece ID. Defense prompts for an attacked piece still use the owner hero ID in
`player_id`, so route the prompt to the owner player's token/session. The owner
hero itself does not appear in `board.entity_locations`; derive board presence
from its pieces. Piece spawn/removal uses `UNIT_PLACED` and `UNIT_REMOVED`
events with `metadata.owner_hero_id`.

### Minion data

```json
{
  "id": "minion_1",
  "type": "MELEE",
  "team": "RED",
  "value": 2,
  "is_heavy": false
}
```

Minion types: `MELEE` (value 2), `RANGED` (value 2), `HEAVY` (value 4).

### Effects

Active area effects on the board:

```json
{
  "id": "effect_1",
  "type": "BUFF",
  "source_card_id": "arien_tidal_wave_1",
  "duration": "UNTIL_END_OF_ROUND",
  "is_active": true,
  "scope": {
    "shape": "SINGLE",
    "range": 0,
    "origin_id": null,
    "origin": {"q": 0, "r": 0, "s": 0},
    "affects": "ALLIES"
  },
  "stat_type": "ATTACK",
  "stat_value": 1,
  "split_axis": null,
  "split_value": 0,
  "named_color": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique effect ID |
| `type` | string | Effect type (e.g. `"los_blocker"`, `"placement_prevention"`, `"area_stat_modifier"`) |
| `source_card_id` | string/null | Card that created this effect. `null` for token-bound effects |
| `duration` | string | `"THIS_TURN"`, `"THIS_ROUND"`, `"PASSIVE"`, or `"NEXT_TURN"` |
| `is_active` | boolean | Whether the effect is currently active |
| `scope.shape` | string | `"point"`, `"radius"`, `"adjacent"`, `"line"`, `"zone"`, `"global"` |
| `scope.range` | int | Range for radius/line shapes |
| `scope.origin_id` | string/null | Entity ID the effect is anchored to (e.g. a token ID). Look up in `entity_locations` for position |
| `scope.origin` | hex/null | Fixed hex origin (overrides `origin_id` when present) |
| `scope.affects` | string | Who is affected (e.g. `"all_units"`, `"enemy_heroes"`) |
| `stat_type` | string/null | For stat modifier effects |
| `stat_value` | int | Modifier amount |
| `split_axis` | string/null | For `topology_split` / `topology_isolation` (NebKher's reality splits): `"q"`, `"r"`, or `"s"` — the cube coordinate defining the split line. `null` for all other effects |
| `split_value` | int | The value of that coordinate: the line is every hex where `hex[split_axis] === split_value`, fixed at cast time. Draw the lasting split visual there |
| `named_color` | string/null | Card color publicly announced when the effect was created (e.g. NebKher's Imbue Doubt family: `"BLUE"`, `"GOLD"`, `"GREEN"`, `"RED"`, `"SILVER"`). Public information — display it to all players while the effect is pending. `null` when no color was named |

**Token-bound effects:** When `source_card_id` is `null` and `scope.origin_id` points to a token, the effect's lifecycle is tied to the token — it persists as long as the token is on the board and is automatically removed when the token is removed.

### Tokens

Tokens are board objects (obstacles, traps, bombs, etc.) that are distinct from units. Only currently placed tokens appear in the `tokens` array and in `entity_locations`; reserve token supplies are server-private.

```json
{
  "id": "smoke_bomb_1",
  "name": "Smoke Bomb",
  "token_type": "smoke_bomb",
  "owner_id": "hero_min",
  "is_passable": false,
  "is_facedown": false,
  "hex": {"q": 1, "r": -1, "s": 0}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique token ID (also appears in `entity_locations` and tile `occupant_id`). Facedown mine IDs are opaque and must not be interpreted as a subtype or supply order |
| `name` | string | Display name |
| `token_type` | string | Token type: `"smoke_bomb"`, `"grenade"`, `"mine_blast"`, `"mine_dud"`, `"zombie"`, `"pyro"`, `"barrier"`, `"ice"`, `"totem"`, `"tree"`, `"rock"`, `"magma"`, `"glitch"`, `"illusion"`, `"familiar"`. For facedown tokens the viewer does not own, this is `"mine"` (true type hidden) |
| `owner_id` | string/null | Hero ID that owns/placed this token |
| `is_passable` | boolean | If `true`, units can move through this token but not land on it. Mine tokens are passable |
| `is_facedown` | boolean | If `true`, the token's actual type is hidden from everyone but its owner. Only the owning hero sees the real `token_type`; teammates, opponents, and spectators see `"mine"` |
| `hex` | hex | Current position on the board |

Tokens are obstacles — any tile with a token as `occupant_id` is impassable unless the token is **passable** (e.g. mines). Passable tokens can be traversed but not landed on. When an enemy hero moves through a passable mine token, the mine is triggered and removed. Blast mines (`mine_blast`) force the moved hero to discard a card; dud mines (`mine_dud`) have no effect. Some effects can make specific tokens unselectable by enemy actions, such as Tali's Venerated Totem.

When a token is removed from the board, any effects anchored to it (via `scope.origin_id`) are automatically removed.

### Board Entities

Some board objects are neither units nor tokens. They appear in `board_entities` and in `entity_locations`.

```json
{
  "id": "trinkets_turret",
  "name": "Turret",
  "entity_kind": "turret",
  "owner_id": "hero_trinkets",
  "is_obstacle": true,
  "hex": {"q": 1, "r": 0, "s": -1}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique board entity ID (also appears in `entity_locations` and tile `occupant_id`) |
| `name` | string | Display name |
| `entity_kind` | string | Entity kind, currently `"turret"` |
| `owner_id` | string/null | Hero ID that owns/placed this entity |
| `is_obstacle` | boolean | Whether this entity blocks occupancy/pathing |
| `hex` | hex/null | Current position on the board. `null` if the entity exists but is not placed |

The Trinkets Turret is a unique board entity. It is an obstacle, but it is not a unit and not a token.

#### Mine path choice

When an enemy hero's movement path passes through mines and multiple routes with different mine combinations exist, the player controlling the movement receives a `SELECT_OPTION` input request to choose which path to take. Each option includes:

```json
{
  "id": "0",
  "text": "Path through 1 mine(s)",
  "metadata": {
    "mine_count": 1,
    "mine_hexes": [{"q": 1, "r": -1, "s": 0}],
    "path": [
      {"q": 0, "r": 0, "s": 0},
      {"q": 1, "r": -1, "s": 0},
      {"q": 2, "r": -1, "s": -1}
    ]
  }
}
```

- `id`: numeric index of the path option (opaque — do not parse)
- `mine_count`: number of mines on this path
- `mine_hexes`: hex positions of the mines
- `path`: full sequence of hexes from start to destination

The path choice is made by the **current actor** (the player controlling the movement), not necessarily the hero being moved (relevant for forced movement effects).

### Markers

```json
{
  "STUN": {
    "target_id": "hero_knight",
    "value": 1,
    "source_id": "hero_arien"
  }
}
```

---

## Handling Input Requests

When the engine needs a hero's input, only that hero (or an authorized member for a team-level request) receives the `input_request` object. Other players and spectators receive `null` via REST and no `input_request` key via WebSocket. A simultaneous upgrade request is reduced to the receiving hero's own `players` entry.

### Who is being waited on

The scoping above hides the request body from everyone who may not answer it — including its *identity*, which observers need to render "waiting for X". The `awaiting_input` array carries that identity separately, unscoped, on every REST view response, `ACTION_RESULT`, and `STATE_UPDATE`:

| Request `player_id` | `awaiting_input` |
|---|---|
| A hero ID (`"hero_wasp"`) | `["hero_wasp"]` |
| `"team:RED"` | every hero on that team |
| `"simultaneous"` | every hero with an entry still pending |
| No pending request | `[]` |

It contains only hero IDs, never options or card identities, so it is safe to send to opponents and spectators.

Two consequences worth designing around:

- **`current_actor_id` is not a fallback for this.** During a defense reaction the current actor is still the *attacker* while the *defender* is being waited on. Read `awaiting_input`; fall back to `view.current_actor_id` only when it is empty.
- **A simultaneous request shrinks as players finish.** Heroes drop out of `awaiting_input` as their pending upgrades reach zero, and a hero owed several upgrades stays listed until all are spent. A player who has finished still receives the request with an empty `players` entry — that is the "done, waiting on others" state, and `awaiting_input` tells them who remains.

### Input request shape

```json
{
  "request_id": "9af46c637790493db8176f323f415b1c",
  "type": "SELECT_HEX",
  "player_id": "hero_arien",
  "prompt": "Choose a hex to move to",
  "valid_hexes": [
    {"q": 1, "r": 0, "s": -1},
    {"q": 0, "r": 1, "s": -1}
  ]
}
```

The `request_id` correlates this prompt with its response. Echo it unchanged in
every REST or WebSocket input response. The `type` field determines what kind
of input is needed and what options fields are present.

### Common input request types

| Type | Options field | Selection value | Description |
|------|--------------|-----------------|-------------|
| `SELECT_HEX` | `valid_hexes` | `{"q": 1, "r": 0, "s": -1}` | Choose a hex on the board |
| `SELECT_UNIT` | `valid_options` | `"hero_knight"` | Choose a unit by ID |
| `SELECT_CARD` | `valid_options` | `"card_id"` | Choose a card by ID. When the chooser may be unable to resolve an ID against their own view (a non-owner caster picking from a masked spellbook, e.g. a copy effect driving Gydion's cast), the request also carries `options` (list of `{id, text}`) with display names — render those as labelled choices for any ID that has no card in the view |
| `CHOOSE_ACTION` | `options` (list of `{id, text}`) | `"ATTACK"` | Choose from named actions |
| `SELECT_OPTION` | `options` (list of `{id, text}`) | `"option_id"` | Choose from generic options |
| `SELECT_CARD_OR_PASS` | `options` (list of `{id, text, ...}`) | `"card_id"` or `"PASS"` | Choose a defense card in reaction. Includes combat context fields and per-card metadata. Some cards make the defender block with a stat other than Defense — always render `defense_value` (see below) |
| `CHOOSE_ACTOR` | `player_ids` | `"hero_arien"` | Choose which hero acts next |
| `UPGRADE_PHASE` | `players` (special structure) | upgrade selection | Choose card upgrades |
| `CONFIRM_PASSIVE` | `options` (`["YES", "NO"]`) | `"YES"` or `"NO"` | Confirm a passive ability |

### How to respond

**Via REST:**

```bash
curl -X POST http://localhost:8000/games/{game_id}/input \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"request_id": "<input_request.request_id>", "selection": {"q": 1, "r": 0, "s": -1}}'
```

**Via WebSocket:**

```json
{
  "type": "SUBMIT_INPUT",
  "request_id": "<input_request.request_id>",
  "selection": {"q": 1, "r": 0, "s": -1}
}
```

The `selection` value must be one of the valid options provided in the input request. For hex selections, send the hex dict. For unit/card selections, send the ID string.

### Skippable inputs

If `can_skip` is `true` in the input request, the player skips by submitting the
literal string `"SKIP"` as the selection (**not** `null` — a null selection is
treated as an invalid choice and the step re-requests input):

```json
{"request_id": "<input_request.request_id>", "selection": "SKIP"}
```

For multi-select inputs, submit the literal string `"DONE"` to finish selecting
once the minimum number of selections has been made.

```json
{"request_id": "<input_request.request_id>", "selection": "DONE"}
```

### Controlled actions (Hanu — The Ultimate Trick)

When Hanu's ultimate is active and Hurry Up! targeted a hero, that hero's next
action is decided by Hanu's player. During such an action, input requests for
the acting hero carry:

- `player_id`: the **controller's** hero id (e.g. `hero_hanu`) — this player
  must answer, and only their token is accepted.
- `controlled_hero_id`: the hero whose action is being performed.

Everything else is unchanged: options are computed relative to the controlled
hero (the controller can only pick choices that hero could legally make), and
requests addressed to other players (defenders, team choices) are unaffected.
Clients should render prompts as "*controller* is controlling *hero*" when
`controlled_hero_id` is present. This field only appears during controlled
actions; the change is additive. An `EFFECT_CREATED` event with
`metadata.effect = "action_control"` announces the control when Hurry Up!
resolves.

### Rollback

If `can_rollback` is `true` in the input request, the client should show a rollback button. When clicked, send a `POST /games/{game_id}/rollback` request (REST) or a `{"type": "ROLLBACK"}` message (WebSocket). This restores the game state to the anchor snapshot — the current actor's most recent actionable prompt — so the player can choose a different action from that point forward.

Rollback is bounded by *segment boundaries*, not a permanent freeze. When any input during the current resolution is addressed to a player other than the resolution owner (e.g., an opponent prompted for defense card selection), the actor's pre-foreign snapshot is dropped and `can_rollback` is `false` on the foreign prompt. This is a segment boundary: a later owner *actionable* prompt in the same resolution may re-anchor a fresh snapshot, and rolling back from there returns to that post-foreign prompt — never past the foreign player's committed decision. In effect, rollback can only undo the current post-foreign owner segment, never an opponent's committed input.

The final `ConfirmResolutionStep` at the end of resolution never *creates* a snapshot on its own — a confirm alone cannot be rolled back to. If an earlier actionable owner snapshot in this resolution is still live, confirm inherits it and continues to advertise `can_rollback`, so rolling back from the confirm prompt returns to that prior actionable prompt. If a foreign segment boundary dropped the snapshot and no owner actionable prompt re-anchored before confirm runs, the confirm step **auto-completes** — no `input_request` is sent to the client and the resolution ends silently, so the actor never sees a bare "click Confirm" prompt whose only choice is confirmation with rollback disabled.

Hidden-info reveals (mine explosions/detonations, card-color guess reveals) are also segment boundaries with the same semantics as foreign input: the pre-reveal snapshot is invalidated so the actor cannot re-guess with the newly-revealed knowledge, but the first owner actionable prompt afterwards re-anchors. Rollback is permanently frozen (fully disabled) for the rest of the turn only when the acting player's turn timer expires. When rollback is frozen, the confirmation step at the end of resolution is auto-confirmed and any direct rollback request is rejected.

Under action control (Hanu's ultimate), a control remap does **not** count as foreign: because the controlled action's inputs are addressed to the controller (with `context.controlled_hero_id` preserving the original owner), `can_rollback` is offered to the controller, who owns the confirm/rollback for that action. The controlled hero cannot rollback it.

### Defense card context

When a `SELECT_CARD_OR_PASS` input request is sent for defense, it includes additional combat context fields so the client can display attack/defense information to the player:

```json
{
  "type": "SELECT_CARD_OR_PASS",
  "player_id": "hero_knight",
  "prompt": "Player hero_knight, select a Defense card. Attack: 3, Defense needed: 2 (minion mod: +1)",
  "options": [
    {
      "id": "knight_shield_wall_1",
      "text": "Shield Wall (Def: 3)",
      "defense_value": 3,
      "base_defense": 2
    },
    {"id": "PASS", "text": "PASS"}
  ],
  "attack_value": 3,
  "minion_modifier": 1,
  "defense_needed": 2
}
```

**Top-level combat context fields:**

| Field | Type | Description |
|-------|------|-------------|
| `attack_value` | int \| null | The incoming attack's damage value |
| `minion_modifier` | int | Defense bonus from adjacent friendly minions |
| `defense_needed` | int \| null | Minimum card defense value to block (`attack_value - minion_modifier`) |

**Per-card option fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Card ID (or `"PASS"`) |
| `text` | string | Display text for the option |
| `defense_value` | int | Total computed block value (base + items + modifiers). Only on card options, not `"PASS"`. |
| `base_defense` | int | Card's base block stat before modifiers. Only on card options, not `"PASS"`. |

The `"PASS"` option is always available — submit `"PASS"` if the player chooses not to defend.

#### Cards that change which stat defends

Some cards make the defender block with a stat other than Defense. Emmitt's
Temporal Punch / Temporal Slam / Temporal Judgment force the defending hero to
use the **Initiative** value of their card and items instead of the Defense
value.

The request shape does not change. `defense_value` and `base_defense` keep
their names but carry the initiative-derived numbers, and the option's `text`
reads `(Init: N)` rather than `(Def: N)`:

```json
{
  "id": "knight_shield_wall_1",
  "text": "Shield Wall (Init: 9)",
  "defense_value": 9,
  "base_defense": 9
}
```

**Always render `defense_value` — never recompute the block value from the
card's own Defense stat.** A client that displays its local copy of the card's
Defense will show the wrong number against these cards.

`defense_needed` still means what it always did (`attack_value -
minion_modifier`), and it stays comparable against `defense_value`. Minion
defense modifiers apply as normal — they are subtracted from the attack rather
than folded into the defender's stat, so this card does not change them.

Stat auras follow the stat actually being used: a Defense aura (Trinkets'
Barrier) does not raise `defense_value` under these cards, while an Initiative
aura (Tali's Ice) does change it.

Card-granted Defense bonuses follow the same rule. A defense card whose text
reads "+2 Defense if ..." (Dodger's Shield of Decay, Vampiric Shield, Aegis of
Doom; Razzle's Crowd Control) contributes nothing when the block is made with
Initiative. Such bonuses are never reflected in `defense_value` either way —
they are applied when combat resolves, not when the option is offered — so a
client cannot and should not try to predict the final block total from the
request.

---

## Events

Events describe what happened during a game action. They are meant for animation and logs — they don't change what's displayed (the view does that), but they tell you *how* it changed. Live REST and WebSocket events are recipient-scoped: metadata that would identify a card or facedown token hidden from the receiver is masked. Clients must tolerate nullable private metadata fields and must treat the view as authoritative. `GUESSED_CARD_REVEALED` and `CARD_REVEALED` are intentional exceptions: the named card's identity is public to every recipient because the effect explicitly flips it faceup.

### Event structure

```json
{
  "event_type": "UNIT_MOVED",
  "actor_id": "hero_arien",
  "target_id": null,
  "from_hex": {"q": 0, "r": 0, "s": 0},
  "to_hex": {"q": 1, "r": -1, "s": 0},
  "metadata": {}
}
```

### Event types

| Event Type | Description | Key fields |
|------------|-------------|------------|
| `UNIT_MOVED` | A unit walked to a new hex | `actor_id`, `from_hex`, `to_hex` |
| `TOKEN_MOVED` | A token moved to a new hex | `target_id`, `from_hex`, `to_hex` |
| `UNIT_PLACED` | A unit was placed on the board (spawn, summon) | `actor_id`, `to_hex` |
| `TOKEN_PLACED` | A token was placed on the board. `metadata.token_type` is the real type for visible tokens and for the owner of a facedown mine, and `"mine"` for everyone else (teammates, opponents, spectators) | `actor_id`, `target_id`, `to_hex`, `metadata.token_type` |
| `BOARD_ENTITY_PLACED` | A non-unit, non-token board entity was placed or repositioned | `actor_id`, `target_id`, `from_hex`, `to_hex`, `metadata.entity_kind` |
| `UNIT_PUSHED` | A unit was forcibly moved | `actor_id`, `from_hex`, `to_hex` |
| `TOKEN_PUSHED` | A token was forcibly moved | `actor_id`, `target_id`, `from_hex`, `to_hex` |
| `UNITS_SWAPPED` | Two units exchanged positions | `actor_id`, `target_id`, `from_hex`, `to_hex` |
| `COMBAT_RESOLVED` | An attack was resolved | `actor_id`, `target_id`, `metadata` (combat details) |
| `UNIT_DEFEATED` | A unit was defeated | `actor_id` (defeated unit) |
| `UNIT_REMOVED` | A unit was removed from the board | `actor_id` |
| `TOKEN_REMOVED` | A token was removed from the board | `target_id`, `from_hex` |
| `BOARD_ENTITY_REMOVED` | A non-unit, non-token board entity was removed from the board | `actor_id`, `target_id`, `from_hex`, `metadata.entity_kind` |
| `MINE_TRIGGERED` | A mine token was triggered by hero movement | `actor_id` (current actor), `target_id` (mine ID), `from_hex`, `metadata.token_type`, `metadata.is_blast` |
| `EFFECT_CREATED` | A new area effect was placed | `metadata` (effect details). For NebKher's reality splits (`topology_split` / `topology_isolation`) also `metadata.split_axis` (`"q"`/`"r"`/`"s"`) and `metadata.split_value` — the line of hexes where that cube coordinate equals the value; draw the split there |
| `HERO_LAUGHED` | A hero laughed diabolically as part of an action (NebKher) | `actor_id` |
| `RESOLVED_CARDS_SWAPPED` | Two resolved cards traded turn slots without canceling active effects (NebKher) | `actor_id`, `target_id` (card owner), `metadata.card_a_id`, `metadata.card_b_id` |
| `DECK_CARD_SWAPPED` | A card in play traded places with a card in its owner's deck (Takahide's gold cycle / Bushido). Card IDs/names are `null` for recipients who cannot see those cards. Takahide's ultimate also emits it for the silver card it retires to the deck, with `metadata.incoming_card_id: null` and `metadata.source: "ready_for_war"` | `actor_id` (card owner), `metadata.outgoing_card_id`, `metadata.incoming_card_id`, `metadata.incoming_card_state`, `metadata.incoming_is_facedown` |
| `CARD_SELECTED_FOR_GUESS` | The target placed one privately selected hand card facedown for a color guess. The event deliberately contains no card identity. | `actor_id` (guesser), `target_id` (hero choosing the card), `metadata.attempt` |
| `GUESSED_CARD_REVEALED` | The selected guess card was flipped faceup for everyone. Log/animation metadata only — render the card itself from `view.card_guess`, which stays correct when a wrong guess returns the card to its hidden hand. | `actor_id` (guesser), `target_id` (card owner), `metadata.attempt`, `metadata.card_id`, `metadata.card_name`, `metadata.card_color`, `metadata.guessed_color`, `metadata.guess_correct` |
| `CARD_REVEALED` | A selected hand card was directly revealed faceup to everyone. Render the persistent face from `view.card_reveal`; the event is the animation/log cue. | `actor_id` (revealer), `target_id` (selected hero unit), `metadata.owner_id`, `metadata.card_id`, `metadata.card_name`, `metadata.card_color`, `metadata.card_tier`, `metadata.tier_value` |
| `SPELL_CAST` | A prepared spell was spent and revealed before its action choice | `actor_id` (caster), `metadata.spell_id`, `metadata.owner_id`, `metadata.caster_id` |
| `SPELL_REMOVED_FROM_SPELLBOOK` | A prepared spell was revealed and removed without being cast | `actor_id` (caster), `metadata.spell_id`, `metadata.owner_id`, `metadata.caster_id` |
| `SPELLBOOK_PREPARED` | Outside spells returned facedown to their owner's spellbook | `actor_id`, `metadata.returned_spell_ids`, `metadata.spellbook_count` |
| `WISH_CAST_COUNT_CHANGED` | A caster advanced their personal Wish victory counter | `actor_id`, `metadata.count`, `metadata.required` |
| `MARKER_PLACED` | A marker was placed on a unit | `target_id`, `metadata` |
| `MARKER_REMOVED` | A marker was removed | `target_id`, `metadata` |
| `GOLD_GAINED` | A hero gained gold | `actor_id`, `metadata.amount` |
| `GOLD_LOST` | A hero lost coins without transferring them | `actor_id` (effect source), `target_id` (affected hero unit), `metadata.owner_id`, `metadata.amount` |
| `LIFE_COUNTER_CHANGED` | A team's life counter changed | `metadata.team`, `metadata.amount` |
| `TURN_ENDED` | A hero's turn ended | `actor_id` |
| `TIMER_EXPIRED` | A match clock expired and the server applied an automatic legal decision. Hidden card IDs remain recipient-scoped. | `actor_id`, `metadata.clock_kind`, `metadata.automatic_action`; optional `metadata.request_id`, `metadata.selection`, `metadata.card_id`, `metadata.team`, `metadata.eligible_hero_ids` |
| `TIE_BREAKER_FLIPPED` | The Tie Breaker coin flipped (after a cross-team tie winner's turn, or via Ignatia's ultimate) | `metadata.tie_breaker_team`, `metadata.coin_face` |
| `RUNES_PLACED` | A hero's rune slots changed (Snorri) | `actor_id`, `metadata.rune_slots` (the new arrangement, e.g. `{"1": "axe", "2": "bird", "3": "anvil", "4": "horn"}`) |
| `GAME_OVER` | The game ended | `metadata.winner`, `metadata.winner_type` (`TEAM` or `HERO`), `metadata.condition` |

### Using events for animation

Process events in order to build an animation sequence:

```
1. Receive ACTION_RESULT with events
2. For each event:
   - UNIT_MOVED → animate unit sliding from from_hex to to_hex
   - COMBAT_RESOLVED → show attack animation
   - UNIT_DEFEATED → show defeat animation
   - etc.
3. After animation, apply the STATE_UPDATE view as the final state
```

The events list may be empty if no observable state changes occurred (e.g., a phase transition with no actions).

---

## Persistence & Reconnection

### Auto-save behavior

The server automatically saves game state to disk after every mutation:
- Card commits, pass turns, input submissions, and advance calls all trigger a save
- Saves use atomic writes (temp file + rename) to prevent corruption
- Save directory defaults to `data/games/`, configurable via `GOA2_SAVE_DIR` environment variable

### Reconnection

Games survive server restarts. When the server starts, it restores all saved games from disk.

To reconnect after a server restart:
1. Use the same tokens you received at game creation
2. Call `GET /games/{game_id}` or connect via WebSocket — you'll get the current game state
3. If there's a pending `input_request`, continue responding to it normally

Tokens are not rotated on restart — the original tokens remain valid for the lifetime of the game.

### Limitations

- Games are stored in-memory with file-based persistence — there is no database
- If the save file is deleted while the server is running, the game continues in memory but won't survive a restart

---

## Character Draft Lobby

An alternative way to start a game. Instead of one player choosing every hero up front via
`POST /games`, a **host** opens a draft lobby and shares a link. Friends join, pick (or are
randomized into) teams, each team's **captain** runs a ban/pick draft, players then **claim**
which drafted hero they will play, and the backend automatically creates the game. From there
clients switch to the normal `/games/{id}` flow.

The draft engine is **pluggable** — `GET /drafts/modes` lists available modes. The default mode is
`sequential_ban_pick`; `simple_draft` is a no-ban snake-pick alternative.

> **Note:** Draft lobbies are **in-memory only** — they do not survive a server restart and are
> not persisted to disk. For live updates, connect the draft WebSocket (see
> [Draft WebSocket](#draft-websocket) below); polling `GET /drafts/{id}` also works as a fallback.

### Lifecycle

```
LOBBY  →  DRAFTING  →  CLAIMING  →  COMPLETE
```

- **LOBBY** — players join, choose teams (or host randomizes), host may reassign captains.
- **DRAFTING** — captains follow the resolved `sequence` of ban/pick actions.
- **CLAIMING** — each player claims one of their team's drafted heroes.
- **COMPLETE** — once everyone has claimed, the game is created; `game_id` and each player's
  `game_token` become available on their scoped view.

### Tokens

| Token | Source | Powers |
|-------|--------|--------|
| Host/player token (admin) | `POST /drafts` response (`player_token`, always player `p1`) | Full player rights **plus** host-only actions (start, randomize, set captain) |
| Player token | `POST /drafts/{id}/join` response | Join a team, draft (if captain), claim a hero |
| Spectator token | `POST /drafts` response (`spectator_token`) | Read-only `GET /drafts/{id}` |

Send as `Authorization: Bearer <token>`, the same scheme as the game API.

### Endpoints

| Method & path | Auth | Body | Purpose |
|---------------|------|------|---------|
| `GET /drafts/modes` | none | — | List available draft modes (`name`, `description`) |
| `GET /drafts/maps` | none | — | List available map names (`["forgotten_island", ...]`) for the lobby picker |
| `POST /drafts` | none | `CreateDraftRequest` | Create a lobby; host becomes player `p1` |
| `PATCH /drafts/{id}/settings` | host | `UpdateDraftSettingsRequest` | Change match settings while in LOBBY |
| `POST /drafts/{id}/join` | none | `{ "display_name": "Bob" }` | Add a player; returns their token |
| `GET /drafts/{id}` | player/spectator | — | Player-scoped draft view |
| `POST /drafts/{id}/team` | player | `{ "team": "RED" }` | Self-select team (LOBBY only) |
| `POST /drafts/{id}/randomize-teams` | host | — | Shuffle players evenly into teams (LOBBY only) |
| `POST /drafts/{id}/captain` | host | `{ "player_id": "p3" }` | Make that player their team's captain (LOBBY only) |
| `POST /drafts/{id}/start` | host | — | Validate & begin drafting |
| `POST /drafts/{id}/action` | acting captain | `{ "hero": "Arien" }` | Submit the current ban or pick |
| `POST /drafts/{id}/claim` | player | `{ "hero": "Arien" }` | Claim a drafted hero (CLAIMING) |

**Team sizes are not chosen up front.** Players join and self-select RED/BLUE freely (up to 6
players total); the per-team sizes are simply however many landed on each side, fixed at `start`.
All match settings (map, game type, draft mode, cheats) default at creation but are
**host-editable inside the lobby** via `PATCH /drafts/{id}/settings`.

`CreateDraftRequest` — only `host_name` is required; the rest seed the lobby defaults:

```json
{
  "host_name": "Alice",
  "map_name": "forgotten_island",
  "game_type": "LONG",
  "draft_mode": "sequential_ban_pick",
  "cheats_enabled": false,
  "max_hero_stars": 4,
  "time_control": null
}
```

`UpdateDraftSettingsRequest` (host-only, LOBBY only) — every field is optional; omitted fields
are left unchanged. Broadcasts a `STATE_UPDATE` like any other mutation:

```json
{
  "map_name": "forgotten_island",
  "game_type": "QUICK",
  "draft_mode": "sequential_ban_pick",
  "cheats_enabled": true,
  "max_hero_stars": 3,
  "time_control": {
    "planning_allowance_seconds": 60,
    "resolution_allowance_seconds": 45,
    "initiative_bonus_seconds": 15,
    "response_grant_seconds": 15,
    "initial_time_bank_seconds": 120,
    "time_bank_increment_seconds": 10,
    "max_time_bank_seconds": 240,
    "upgrade_allowance_seconds": 45,
    "automatic_turn_limit": 2
  }
}
```

Sending `"time_control": null` explicitly disables clocks; omitting it leaves
the current lobby setting unchanged.

At `start`, the number of **assigned** players (those on a team) must be a supported match size
(2, 4, 5, or 6) — otherwise `start` returns `400`. Each team must have at least one player and a
captain. Team sizes may be uneven (e.g. 2 vs 3 for a 5-player game).

Available draft modes:

| Mode | Order |
|------|-------|
| `sequential_ban_pick` | Ban pair before each pick pair; see the table below. |
| `simple_draft` | No bans. Team A picks 1, Team B picks 2, then teams alternate up to 2 picks until both rosters are full. |

`sequential_ban_pick` resolves draft rounds relative to `first_team` (Team A) and the other team
(Team B):

| Round | Ban order | Pick order |
|-------|-----------|------------|
| 1 | Team A, Team B | Team A, Team B |
| 2 | Team B, Team A | Team B, Team A |
| 3 | Team A, Team B | Team B, Team A |

Only the rounds needed to fill the larger team are emitted. In uneven drafts, both teams still
make the round's bans, but a team that already has enough picked heroes skips that round's pick.

### Draft view shape

Every mutating endpoint and `GET /drafts/{id}` return a `DraftViewResponse`:

```json
{
  "draft": {
    "draft_id": "ab12cd34ef56",
    "status": "DRAFTING",
    "map_name": "forgotten_island",
    "game_type": "LONG",
    "draft_mode": "sequential_ban_pick",
    "cheats": false,
    "red_size": 2,
    "blue_size": 2,
    "players": [
      {"id": "p1", "display_name": "Alice", "team": "RED",
       "is_host": true, "is_captain": true, "claimed_hero": null}
    ],
    "hero_pool": ["Arien", "Wasp", "..."],
    "sequence": [
      {"index": 0, "action": "BAN", "team": "RED"},
      {"index": 1, "action": "BAN", "team": "BLUE"},
      {"index": 2, "action": "PICK", "team": "RED"},
      {"index": 3, "action": "PICK", "team": "BLUE"},
      {"index": 4, "action": "BAN", "team": "BLUE"},
      {"index": 5, "action": "BAN", "team": "RED"},
      {"index": 6, "action": "PICK", "team": "BLUE"},
      {"index": 7, "action": "PICK", "team": "RED"}
    ],
    "current_index": 2,
    "bans": {"RED": ["Mortimer"], "BLUE": ["Widget"]},
    "picks": {"RED": [], "BLUE": []},
    "first_team": "RED",
    "game_id": null
  },
  "you": {"id": "p1", "display_name": "Alice", "team": "RED",
          "is_host": true, "is_captain": true, "claimed_hero": null},
  "game_id": null,
  "game_token": null
}
```

- `draft` is public (identical for all callers) — all bans/picks are visible to everyone.
- `red_size`/`blue_size` are `0` during `LOBBY` (sizes aren't decided yet); they are set from
  team membership when the draft starts. During `LOBBY`, derive the live counts from `players`.
- `cheats` reflects the current (host-editable) cheat setting; it flows into the created game.
- `you` is the caller's own player record (omitted/`null` for spectators).
- Once `status` is `COMPLETE`, `game_id` is set, and a player calling `GET /drafts/{id}` with
  their own token also receives their `game_token`. Use these to switch to the normal game flow:
  `GET /games/{game_id}` with `Authorization: Bearer <game_token>`.

To know whose turn it is during `DRAFTING`, read `sequence[current_index]`: the `team` field
tells you which team acts, and its captain (the player with `is_captain: true` on that team)
must submit the next `action`.

### Draft WebSocket

For live updates, connect:

```
ws://<host>/drafts/{draft_id}/ws?token=<bearer_token>
```

Any token works (host, player, or spectator). The channel is **read-only** — you still perform
every action through the REST endpoints. The shared spectator token may be used
by multiple concurrent WebSockets, and every connected spectator receives the
same public draft broadcasts. On connect, and after **every** REST mutation by
any participant (join, team change, randomize, captain change, start, ban/pick,
claim, and the final game creation), the server pushes a player-scoped
`STATE_UPDATE`:

```json
{
  "type": "STATE_UPDATE",
  "draft": { "...": "same shape as GET /drafts/{id} -> draft" },
  "you": { "id": "p1", "...": "your player record, or null for spectators" },
  "game_id": null,
  "game_token": null
}
```

`STATE_UPDATE` carries the same player-scoping as the REST view, so once the draft is
`COMPLETE` each player's socket receives their own `game_id` and `game_token`.

Inbound messages:

| Send | Effect |
|------|--------|
| `{"type": "GET_VIEW"}` | Server replies with a fresh `STATE_UPDATE` |
| anything else | Server replies `{"type": "ERROR", ...}` (the channel is read-only) |

Close codes: `4001` invalid token, `4003` token does not match draft, `4004` draft not found.

### Draft errors

In addition to the standard error shape (`{"detail": "..."}`), draft endpoints can return:

| Status | Cause |
|--------|-------|
| `403` | Non-host called a host-only action; wrong captain called `action`; spectator attempted a mutation |
| `404` | Unknown `draft_id` or `player_id` |
| `409` | Lobby full; wrong draft phase; hero already banned/picked; hero not claimable |
| `400` | Invalid team; unsupported player count; unknown map or draft mode |

---

## Consensus Overrides

When the engine gets something wrong — refuses a legal move, resolves a defeat
that should not have happened, wedges mid-action — the table can force the game
into the state it should be in, by majority vote. Three families of override
exist:

- **patch** — correct a wrong value (position, defeat, gold, life counters, …)
- **unstick** — escape a wedged control flow (skip the pending input, abort the
  action, force the turn to end, fix the actor)
- **rewind** — return the whole game to an earlier decision index

Overrides are recorded as replay decisions and survive reconstruction; the
negotiation itself (proposals, votes) is transient — it is **not** part of
`build_view()` output and does not survive a server restart (re-propose).

### Consensus rules

- Eligible voters are the players **connected at proposal time** (snapshotted).
  Spectators never vote. The proposer automatically counts as a yes.
- The threshold is a strict majority of the snapshot. In a 2-player game both
  players must agree.
- One open proposal at a time.
- Proposals expire after 120 seconds (server-configurable via
  `GOA2_OVERRIDE_TIMEOUT_SECONDS`); expiry is a **rejection**.
- The turn clock is paused while a proposal is open and resumes on resolution.
- Propose/vote/cancel are WebSocket-only; there are no REST equivalents.

### WebSocket messages

Client → server:

```json
{"type": "PROPOSE_OVERRIDE", "family": "patch", "op": "move_entity",
 "args": {"entity_id": "minion_4", "hex": {"q": 1, "r": -2, "s": 1}}}

{"type": "PROPOSE_OVERRIDE", "family": "rewind", "to": 47}

{"type": "VOTE_OVERRIDE", "proposal_id": "abc123", "approve": true}

{"type": "CANCEL_OVERRIDE", "proposal_id": "abc123"}
```

Only the proposer may cancel. A spectator sending any of these gets an
`ERROR` (spectators can only `GET_VIEW`).

Server → all connections (players and spectators):

```json
{"type": "OVERRIDE_PROPOSED",
 "proposal_id": "abc123",
 "proposer_hero_id": "hero_arien",
 "family": "patch",
 "op": "move_entity",
 "args": {"entity_id": "minion_4", "hex": {"q": 1, "r": -2, "s": 1}},
 "to": null,
 "summary": "Move minion_4 to {'q': 1, 'r': -2, 's': 1}",
 "eligible_voters": ["hero_arien", "hero_wasp"],
 "threshold": 2,
 "tally": {"yes": ["hero_arien"], "no": []},
 "expires_at": 1718900123.4}
```

`expires_at` is an absolute epoch timestamp so clients can render a countdown
without clock-skew guesswork. `summary` is a server-rendered human summary for
the vote prompt — no follow-up fetch needed.

```json
{"type": "OVERRIDE_UPDATED", "proposal_id": "abc123",
 "tally": {"yes": ["hero_arien"], "no": ["hero_wasp"]}}

{"type": "OVERRIDE_RESOLVED", "proposal_id": "abc123",
 "outcome": "applied",
 "tally": {"yes": ["hero_arien", "hero_wasp"], "no": []}}
```

`outcome` is one of `applied` / `rejected` / `expired` / `cancelled`. When an
*approved* override fails validation at apply time (e.g. the target hex became
occupied), the outcome is `rejected` **with** a structured `reason`:

```json
{"type": "OVERRIDE_RESOLVED", "proposal_id": "abc123",
 "outcome": "rejected",
 "tally": {"yes": ["hero_arien", "hero_wasp"], "no": []},
 "reason": {"code": "not_on_board", "message": "minion_999 is not on the board"}}
```

No `reason` field means the proposal was outvoted, cancelled, or expired.

On `applied`, every connection receives the `OVERRIDE_RESOLVED` message first,
immediately followed by a fresh `STATE_UPDATE`.

**After an applied patch, any in-flight `SUBMIT_INPUT` may be rejected** with a
request-id mismatch error: the pending input request is re-derived against the
patched board and gets a **new** request id. Read the fresh `input_request`
from the broadcast and re-render.

### `GET /overrides/schema`

Static, unauthenticated, game-independent — fetch once and cache. Returns the
full op catalogue with a JSON Schema per op, so clients never hardcode the op
list:

```json
{"ops": [
  {"name": "move_entity", "family": "patch", "label": "Move entity",
   "description": "Move a unit, hero piece, or token to a hex (fixes a refused legal move).",
   "args_schema": {"type": "object", "properties": {"entity_id": {"type": "string"},
                    "hex": {"$ref": "#/$defs/HexArg"}}, "required": ["entity_id", "hex"]}}
]}
```

Patch ops: `move_entity`, `remove_entity`, `place_entity`, `set_life_counters`,
`set_gold`, `set_level`, `add_marker`, `remove_marker`, `add_effect`,
`remove_effect`, `move_card`, `set_wave_counter`, `set_tie_breaker_team`.
Unstick ops: `skip_input`, `abort_action`, `end_turn`, `force_actor`.
(The endpoint output is authoritative; this list is illustrative.)

### `GET /games/{game_id}/overrides/history`

Bearer-authenticated (player or spectator token). Renders the game's decision
list so a rewind target index means something to the table:

```json
{"total": 3, "decisions": [
  {"index": 0, "type": "commit", "round": 1, "turn": 1,
   "hero_id": "hero_arien", "label": "hero_arien committed a card",
   "superseded": true},
  {"index": 1, "type": "ov_rewind", "round": 1, "turn": 1,
   "hero_id": "hero_arien", "label": "The table rewound the game to decision 0",
   "superseded": false},
  {"index": 2, "type": "pass", "round": 1, "turn": 1,
   "hero_id": "hero_arien", "label": "hero_arien passed", "superseded": false}
]}
```

- Card identity in labels is player-scoped with the same visibility rule as the
  view: an opponent's facedown commit reads "a card" until the card is public;
  spectators get the fully-masked form.
- `superseded: true` marks records behind a rewind (dead segments). Render
  `ov_rewind` rows as visible markers and grey out superseded rows rather than
  hiding them.
- `PROPOSE_OVERRIDE` with `family: "rewind"` takes `to` in this `index` space.
  Rewind depth is unrestricted by design — a table that votes to go back past a
  round boundary has accepted that already-seen cards become hidden again.

---

## Error Handling

### HTTP status codes

| Status | Meaning | Example |
|--------|---------|---------|
| `201` | Game created | `POST /games` success |
| `200` | Success | All other successful responses |
| `400` | Bad request | Invalid input value |
| `401` | Unauthorized | Missing or invalid bearer token |
| `403` | Forbidden | Spectator trying to mutate, token doesn't match game, not your turn |
| `404` | Not found | Game ID doesn't exist, map not found, card not in hand |
| `409` | Conflict | Wrong game phase for the operation |

### Error response shape

```json
{
  "detail": "Expected phase PLANNING, but game is in RESOLUTION"
}
```

### WebSocket errors

WebSocket errors are sent as messages (the connection stays open):

```json
{
  "type": "ERROR",
  "detail": "Input expected from 'hero_knight', not 'hero_arien'"
}
```

Connection-level errors close the WebSocket with a code:

| Code | Reason |
|------|--------|
| `4001` | Invalid token |
| `4003` | Token does not match game |
| `4004` | Game not found |

### Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `"Missing or invalid Authorization header"` | No `Bearer` prefix or missing header | Add `Authorization: Bearer <token>` header |
| `"Invalid token"` | Token doesn't match any game | Use a token from `POST /games` response |
| `"Token does not match this game"` | Token belongs to a different game | Check the game_id in the URL |
| `"Expected phase PLANNING, but game is in RESOLUTION"` | Called commit_card/pass during wrong phase | Check `current_phase` before acting |
| `"Input expected from 'X', not 'Y'"` | Wrong player submitting input | Only the `player_id` from `input_request` should submit |
| `"Spectators cannot commit cards"` | Spectator token used for a mutation | Use a player token instead |
| `"Card 'X' not in Y's hand"` | Invalid card_id for commit | Check the hero's `hand` in the view |
