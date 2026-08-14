# Guards of Atlantis II Backend

Deterministic Python backend for **Guards of Atlantis II**. The project contains the authoritative game engine, hero card data and effects, and a FastAPI server that exposes player-scoped REST and WebSocket APIs for clients.

The engine is built around a stack of serializable game steps instead of nested rule calls. A card, phase, reaction, or passive ability expands into small `GameStep` objects on `GameState.execution_stack`; the resolver can pause for player input, persist the full paused state, and resume from the same stack later.

## Project Goal

The goal is to make the backend the single source of truth for Guards of Atlantis II rules. Clients should not need to reimplement movement legality, targeting, combat math, reaction timing, card text, upgrades, hidden information, or phase progression. They should render a player-scoped view, submit explicit player choices, and animate the events emitted by the engine.

That pushes most complexity into the backend on purpose:

- Rules are tested once in Python instead of being duplicated across clients.
- A paused mid-card resolution is represented as data and can be persisted or resumed.
- Client integrations receive stable REST/WebSocket contracts instead of raw engine state.
- Hidden information is filtered server-side through player-scoped views.
- Card effects are authored from reusable steps and filters rather than ad hoc turn scripts.

## Design Philosophy

### Logic as Data

Game flow is represented as data on `GameState.execution_stack`. Composite actions produce smaller atomic steps; the resolver executes those steps until it either finishes, needs input, aborts an action, or reaches game over. This makes nested rules, reactions, optional choices, and defense windows manageable without deeply nested call stacks.

### Deterministic State Transitions

Given the same starting state and the same sequence of inputs, the engine should produce the same result. Randomness, input pauses, card decisions, and tie breakers are modeled explicitly so tests and clients can reason about the flow.

### Server-Owned Visibility

The backend owns hidden information rules. API clients receive views from `build_view()` rather than raw `GameState`, so facedown cards and player-specific information stay scoped to the correct token.

### Effects from Reusable Primitives

Hero card logic should be built from reusable `GameStep` and `FilterCondition` primitives. This keeps complex card text testable, serializable, and consistent with the rest of the engine.

## Current Scope

- Stack-based, deterministic rules engine for turn, phase, movement, combat, reactions, upgrades, effects, markers, topology, and validation.
- Hero definitions and card effect implementations under `src/goa2/data/heroes/` and `src/goa2/scripts/`.
- FastAPI server with game creation, bearer-token auth, player-scoped views, card commit/pass/input/advance endpoints, WebSocket updates, rollback, cheats, autosave persistence, and game logs.
- Client-facing contracts for input requests, events, views, response models, and WebSocket messages.
- Test suite covering domain models, engine behavior, card effects, persistence, API routes, auth, and WebSockets.

For deeper orientation, start with:

- [Codebase Map](docs/CODEBASE_MAP.md) for architecture and module responsibilities.
- [Client Integration Guide](docs/CLIENT_INTEGRATION_GUIDE.md) for REST, WebSocket, auth, views, inputs, and events.
- [Effect Author Reference](docs/EFFECT_AUTHOR_REFERENCE.md) for card effect steps, filters, and patterns.
- [Card Effects Guidelines](docs/card_effects_guidelines.md) for mandatory/optional behavior and effect authoring rules.

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

Install dependencies:

```bash
uv sync
```

Install local hooks:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Common Commands

Run all tests:

```bash
PYTHONPATH=src uv run pytest tests/ -q
```

Run tests with coverage:

```bash
PYTHONPATH=src uv run pytest --cov=goa2 tests/
```

Run server tests only:

```bash
PYTHONPATH=src uv run pytest tests/server/ -q
```

Run one test file or test:

```bash
PYTHONPATH=src uv run pytest tests/engine/test_steps.py
PYTHONPATH=src uv run pytest tests/engine/test_steps.py::test_function_name -v
```

Run quality checks:

```bash
uv run ruff check src/
uv run black src/
uv run mypy src/
```

Start the development API server:

```bash
PYTHONPATH=src uv run uvicorn goa2.server.app:create_app --factory --reload
```

Run the step-engine demo:

```bash
PYTHONPATH=src uv run python -m goa2.scripts.demo_step_engine
```

## API Quick Start

Start the server, then create a game:

```bash
curl -X POST http://localhost:8000/games \
  -H "Content-Type: application/json" \
  -d '{
    "map_name": "forgotten_island",
    "red_heroes": ["Arien"],
    "blue_heroes": ["Knight"]
  }'
```

The response contains one bearer token per hero plus a spectator token. Save them; tokens are the identity model for both REST and WebSocket access.

Fetch a player-scoped view:

```bash
curl http://localhost:8000/games/<game_id> \
  -H "Authorization: Bearer <player_token>"
```

Connect over WebSocket:

```text
ws://localhost:8000/games/<game_id>/ws?token=<player_token>
```

Full details are in [docs/CLIENT_INTEGRATION_GUIDE.md](docs/CLIENT_INTEGRATION_GUIDE.md).

## Architecture

### Step Engine

The core resolver lives in `src/goa2/engine/handler.py`.

```text
process_resolution_stack(state)
  pop GameStep from state.execution_stack
  resolve step against GameState + execution_context
  if input is required: pause and return InputRequest
  if new steps are returned: push them in reverse for LIFO execution
  repeat until the stack is empty, paused, aborted, or game over
```

Important step modules:

- `src/goa2/engine/steps/base.py` - `GameStep` and `StepResult`
- `src/goa2/engine/steps/selection.py` - `SelectStep`, multi-select, tie breakers
- `src/goa2/engine/steps/movement.py` - movement, push, place, swap, displacement
- `src/goa2/engine/steps/combat.py` - attacks, defense windows, damage, defeat
- `src/goa2/engine/steps/cards.py` - card lifecycle, economy, upgrades
- `src/goa2/engine/steps/effects.py` - active effects and passive triggers
- `src/goa2/engine/steps/markers.py` - marker/token behavior
- `src/goa2/engine/steps/phases.py` - end phase, level-up, round reset
- `src/goa2/engine/steps/utility.py` - conditionals and helper/control-flow steps

New step classes must also be added to `StepType` in `src/goa2/domain/models/enums.py` and the `AnyStep` union in `src/goa2/engine/step_types.py`, otherwise persisted games cannot deserialize them.

### Game State

`src/goa2/domain/state.py` is the central mutable world model. The most important fields are:

- `execution_stack` - LIFO queue of pending steps.
- `execution_context` - transient data shared between steps during a chain.
- `entity_locations` - authoritative positions for heroes, minions, tokens, and markers. Do not mutate board tile occupants directly.
- `active_effects` - temporary and passive effects.
- `teams`, `board`, `markers`, `current_actor_id`, phase and round fields.

`board.tiles[*].occupant_id` is derived/synchronized from `entity_locations`; it is not the source of truth.

### Effects and Filters

Card effects subclass `CardEffect`, register by effect id, and return steps from `build_steps()` or specialized hooks such as defense, on-block, or passive hooks.

```python
from goa2.engine.effects import CardEffect, register_effect
from goa2.engine.steps import AttackSequenceStep


@register_effect("my_card")
class MyCardEffect(CardEffect):
    def build_steps(self, state, hero, card, stats):
        return [
            AttackSequenceStep(
                damage=stats.primary_value,
                range_val=stats.range,
                is_ranged=True,
            )
        ]
```

Selections use composable filters from `src/goa2/engine/filters*.py`:

```python
SelectStep(
    target_type="UNIT",
    prompt="Choose an enemy within 2.",
    output_key="target_id",
    filters=[
        TeamFilter(relation="ENEMY"),
        RangeFilter(max_range=2),
    ],
)
```

New filters must be added to `FilterType` and the `AnyFilter` union in `src/goa2/engine/step_types.py`.

### Server Layer

`src/goa2/server/` wraps the engine for client use:

- `app.py` - FastAPI factory and registry setup.
- `routes_games.py` - REST game endpoints.
- `routes_heroes.py` - hero discovery and metadata.
- `ws.py` - WebSocket protocol and broadcasts.
- `auth.py` - bearer-token auth.
- `registry.py` - in-memory game registry, persistence, and logs.
- `models.py` - public response/request models.

Server code must send player-scoped views from `src/goa2/domain/views.py`; never expose raw `GameState` to clients. Mutating endpoints and WebSocket actions should save after mutation and broadcast updated views when appropriate.

## Project Layout

```text
src/goa2/
  domain/              Pydantic state, board, hex, input, events, views, models
  engine/              Resolver, steps, effects, filters, validation, topology, persistence
  data/heroes/         Hero card definitions and registry
  data/maps/           JSON map data
  scripts/             Hero effect implementations and demos
  server/              FastAPI REST/WebSocket API

tests/
  domain/              Model, state, view, token, and event tests
  engine/              Step, phase, rule, effect, persistence, and scenario tests
  engine/effects/      Character effect helper framework and effect cases
  server/              REST, auth, registry, persistence, and WebSocket tests

docs/
  CODEBASE_MAP.md
  CLIENT_INTEGRATION_GUIDE.md
  EFFECT_AUTHOR_REFERENCE.md
  card_effects_guidelines.md
```

## Development Rules That Matter

- Read the existing step/effect patterns before changing behavior.
- Use `InputRequest` from `src/goa2/domain/input.py`; do not return raw dicts from steps that need input.
- Emit `GameEvent`s for observable state changes.
- Keep client-facing contracts stable. If you change `server/models.py`, `domain/input.py`, `domain/events.py`, `domain/views.py`, REST endpoints, or WebSocket message shapes, update [docs/CLIENT_INTEGRATION_GUIDE.md](docs/CLIENT_INTEGRATION_GUIDE.md).
- Add new steps and filters to the discriminated unions in `src/goa2/engine/step_types.py`.
- Use `state.entity_locations` and state helpers for positions; do not directly edit board tile occupants.
- For new or touched character effects, prefer the helper framework in `tests/engine/effects/` and mark tests with `effect_contract` or `effect_flow`.
- For server changes, run `PYTHONPATH=src uv run pytest tests/server/ -q` before merging.

## Testing Notes

Focused character effect tests should usually use:

- `tests/engine/effects/builders.py`
- `tests/engine/effects/runner.py`
- `tests/engine/effects/assertions.py`

Use behavior assertions rather than checking only that setup wiring happened. For narrow effect contracts, prefer `@pytest.mark.effect_contract`; for longer integration flows, use `@pytest.mark.effect_flow`.

## Useful Environment Variables

- `GOA2_SAVE_DIR` - save-game directory for the server. Defaults to `data/games`.
- `GOA2_LOG_DIR` - game log directory. Defaults to `logs/games`.
- `GOA2_REPLAY_DIR` - minimal replay-log directory. Defaults to `data/replays`.
- `GOA2_REPLAY_TTL_DAYS` - how long replay logs are retained, in days. Defaults to `30`.

### Replay logs

Every server game writes a compact, append-only replay log
(`data/replays/<game_id>.jsonl`) capturing only the setup parameters (including
the RNG seed) and the ordered player decisions (card commits, passes, inputs).
Because gameplay is deterministic, this is enough to reconstruct a game exactly.
Each decision record also carries a wall-clock `ts` timestamp (epoch seconds)
for analytics. Replays are retained for 30 days — independent of the 24h
save-game cleanup — so a bug reported after a game ends can still be
reproduced. Reconstruct one with the CLI:

```bash
# Replay the whole game
PYTHONPATH=src uv run python -m goa2.scripts.replay data/replays/<game_id>.jsonl

# Stop at the start of round 3, turn 2 (the moment a bug was reported)
PYTHONPATH=src uv run python -m goa2.scripts.replay data/replays/<game_id>.jsonl --round 3 --turn 2

# Or by decision index, and dump a player-scoped view at the stop point
PYTHONPATH=src uv run python -m goa2.scripts.replay data/replays/<game_id>.jsonl --decision 47 --view hero_arien
```

## Status

The backend is under active development. Treat the code and tests as the authority for implemented rule behavior, and treat [docs/CLIENT_INTEGRATION_GUIDE.md](docs/CLIENT_INTEGRATION_GUIDE.md) as the authority for client contracts.
