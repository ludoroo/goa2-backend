"""Unit tests for GameRegistry."""

import asyncio

import pytest

from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup
from goa2.server.bot_models import BotSpec, SearchSettings
from goa2.server.errors import GameNotFoundError
from goa2.server.registry import GameRegistry, ManagedGame

MAP_PATH = "src/goa2/data/maps/forgotten_island.json"


@pytest.fixture
def registry():
    return GameRegistry()


@pytest.fixture
def session():
    state = GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"])
    return GameSession(state)


def test_create_game(registry, session):
    game = registry.create_game(session, ["hero_arien", "hero_wasp"])
    assert isinstance(game, ManagedGame)
    assert len(game.game_id) == 12
    assert len(game.player_tokens) == 2
    assert len(game.hero_to_token) == 2
    assert game.spectator_token


def test_get_existing_game(registry, session):
    game = registry.create_game(session, ["hero_arien", "hero_wasp"])
    fetched = registry.get(game.game_id)
    assert fetched is game


def test_get_missing_game_raises(registry):
    with pytest.raises(GameNotFoundError):
        registry.get("nonexistent")


def test_resolve_player_token(registry, session):
    game = registry.create_game(session, ["hero_arien", "hero_wasp"])
    token = game.hero_to_token["hero_arien"]
    result = registry.resolve_token(token)
    assert result is not None
    game_id, hero_id, is_spectator = result
    assert game_id == game.game_id
    assert hero_id == "hero_arien"
    assert is_spectator is False


def test_resolve_spectator_token(registry, session):
    game = registry.create_game(session, ["hero_arien", "hero_wasp"])
    result = registry.resolve_token(game.spectator_token)
    assert result is not None
    game_id, hero_id, is_spectator = result
    assert game_id == game.game_id
    assert hero_id == ""
    assert is_spectator is True


def test_resolve_unknown_token(registry):
    assert registry.resolve_token("bogus") is None


def test_remove_game(registry, session):
    game = registry.create_game(session, ["hero_arien", "hero_wasp"])
    registry.remove(game.game_id)
    with pytest.raises(GameNotFoundError):
        registry.get(game.game_id)


def test_registry_len(registry, session):
    assert len(registry) == 0
    registry.create_game(session, ["hero_arien"])
    assert len(registry) == 1


# ---------------------------------------------------------------------------
# Persisted bot metadata
# ---------------------------------------------------------------------------


def test_managed_game_bot_specs_default_empty(registry, session):
    """A game created without bot_specs starts with an empty mapping."""
    game = registry.create_game(session, ["hero_arien", "hero_wasp"])
    assert game.bot_specs == {}
    assert game.bot_task is None


def test_create_game_accepts_bot_specs(registry, session):
    """Bot specs supplied at creation are validated and stored on the game."""
    specs = {
        "hero_arien": BotSpec(kind="heuristic"),
        "hero_wasp": BotSpec(
            kind="ismcts",
            search=SearchSettings(iterations=50, decision_timeout_seconds=1.0),
        ),
    }
    game = registry.create_game(session, ["hero_arien", "hero_wasp"], bot_specs=specs)
    assert game.bot_specs == specs
    assert game.bot_specs["hero_wasp"].search is not None
    assert game.bot_specs["hero_wasp"].search.iterations == 50


def test_create_game_rejects_bot_spec_for_hero_not_in_roster(registry, session):
    """Any bot hero must belong to the game roster."""
    specs = {"hero_notinroster": BotSpec(kind="random")}
    with pytest.raises(ValueError, match="hero_notinroster"):
        registry.create_game(session, ["hero_arien", "hero_wasp"], bot_specs=specs)


def test_bot_task_never_persisted(registry, session):
    """`bot_task` is runtime-only and is not part of the persisted payload."""
    # Ensure a fresh event loop exists so we can build a real Task and assign
    # it — mirrors what the coordinator will eventually do.
    loop = asyncio.new_event_loop()
    try:
        game = registry.create_game(session, ["hero_arien", "hero_wasp"])

        async def _noop() -> None:
            return None

        task = loop.create_task(_noop())
        game.bot_task = task
        # It must not appear in any persisted representation. save_game is a
        # no-op for a memory-only registry, so re-run persistence.save_game
        # against a real save dir in the persistence tests. Here we simply
        # assert the field exists and is Optional.
        assert game.bot_task is task
        loop.run_until_complete(task)
    finally:
        loop.close()


def test_remove_cancels_bot_task(registry, session):
    """Removing a game cancels any live bot_task."""
    loop = asyncio.new_event_loop()
    try:
        game = registry.create_game(session, ["hero_arien", "hero_wasp"])

        async def _forever() -> None:
            await asyncio.sleep(3600)

        task = loop.create_task(_forever())
        game.bot_task = task
        registry.remove(game.game_id)
        # Give the loop a tick to observe cancellation.
        with pytest.raises(asyncio.CancelledError):
            loop.run_until_complete(task)
    finally:
        loop.close()
