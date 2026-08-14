"""Unstick override ops + patch/pending-input interaction."""

import pytest

from goa2.domain.input import InputResponse
from goa2.domain.models import GamePhase
from goa2.engine.overrides import OverrideRejectedError, apply_override_decision
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup

MAP_PATH = "src/goa2/data/maps/forgotten_island.json"


def _build_mid_resolution_session() -> GameSession:
    """A 1v1 game advanced into RESOLUTION with a pending input request."""
    state = GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"], False, "QUICK", seed=42)
    session = GameSession(state)
    result = None
    for hero_id in ("hero_arien", "hero_wasp"):
        hero = state.get_hero(hero_id)
        result = session.commit_card(hero_id, hero.hand[0])
    # Committing the final card fires revelation and processes the stack;
    # drive until an input request is pending.
    for _ in range(20):
        if result is not None and result.input_request is not None:
            break
        if state.phase == GamePhase.PLANNING:
            raise AssertionError("game unexpectedly back in PLANNING")
        result = session.advance(None)
    assert result is not None and result.input_request is not None
    session._last_request = result.input_request  # test-side stash
    return session


@pytest.fixture
def mid_resolution_session() -> GameSession:
    return _build_mid_resolution_session()


def test_patch_bumps_pending_request_id(mid_resolution_session):
    session = mid_resolution_session
    old_request = session._last_request
    result = apply_override_decision(session, "set_gold", {"hero_id": "hero_arien", "value": 9})
    assert result is not None and result.input_request is not None
    # Re-derived request has a NEW id: a stale in-flight answer must be rejected.
    assert result.input_request.id != old_request.id
    with pytest.raises(ValueError):
        session.advance(InputResponse(request_id=old_request.id, selection="anything"))


def test_skip_input_answers_pending_request(mid_resolution_session):
    session = mid_resolution_session
    result = apply_override_decision(session, "skip_input", {})
    # The wedged request was answered with SKIP; play moved on (a new request
    # or a completed action, but not the same request id).
    if result is not None and result.input_request is not None:
        assert result.input_request.id != session._last_request.id


def test_abort_action_unwinds_wedged_step(mid_resolution_session):
    session = mid_resolution_session
    apply_override_decision(session, "abort_action", {})
    assert session.state.phase in (
        GamePhase.RESOLUTION,
        GamePhase.PLANNING,
        GamePhase.CLEANUP,
        GamePhase.LEVEL_UP,
    )
    # The wedged step is gone from the stack.
    assert all(
        s.pending_request_id != session._last_request.id for s in session.state.execution_stack
    )


def test_end_turn_forces_turn_end(mid_resolution_session):
    session = mid_resolution_session
    actor = str(session.state.current_actor_id)
    apply_override_decision(session, "end_turn", {})
    hero = session.state.get_hero(actor)
    assert hero.current_turn_card is None  # finalized


def test_force_actor_sets_current_actor(mid_resolution_session):
    session = mid_resolution_session
    apply_override_decision(session, "force_actor", {"hero_id": "hero_wasp"})
    assert str(session.state.current_actor_id) == "hero_wasp"


def test_force_actor_unknown_hero_rejected(mid_resolution_session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(mid_resolution_session, "force_actor", {"hero_id": "hero_x"})


def test_skip_input_without_pending_request_rejected():
    state = GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"], False, "QUICK", seed=1)
    session = GameSession(state)  # PLANNING, nothing pending
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "skip_input", {})
