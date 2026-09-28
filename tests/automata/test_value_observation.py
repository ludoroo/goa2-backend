"""Candidate-free observations at real stable value boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from automata.models.contracts import (
    StableValueObservation,
    canonical_json_bytes,
    from_canonical_json,
)
from automata.observation.value_encoder import encode_stable_value
from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import (
    StableValueBoundaryKind,
    detect_stable_value_boundary,
)
from goa2.domain.models import CardState, GamePhase, TeamColor
from goa2.domain.models.effect import ActiveEffect, DurationType, EffectScope, EffectType, Shape
from goa2.domain.types import HeroID
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.phases import commit_card, resolve_next_action
from goa2.engine.setup import GameSetup
from goa2.engine.steps import FinalizeHeroTurnStep, LogMessageStep

MAP = str(Path("src/goa2/data/maps/forgotten_island.json"))


def _planning_ready_state():
    return GameSetup.create_game(
        MAP,
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=73,
    )


def _actor_ready_state():
    state = _planning_ready_state()
    actor = state.get_hero(HeroID("hero_wasp"))
    assert actor is not None and actor.hand
    card = actor.hand.pop()
    card.state = CardState.UNRESOLVED
    card.is_facedown = False
    actor.current_turn_card = card
    state.phase = GamePhase.RESOLUTION
    state.unresolved_hero_ids = [actor.id]

    resolve_next_action(state)

    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    assert boundary.kind is StableValueBoundaryKind.ACTOR_READY
    assert boundary.actor_id == "hero_wasp"
    return state, boundary


def _encode(state, boundary, *, viewer: str = "hero_wasp", team=TeamColor.RED):
    return encode_stable_value(
        state,
        boundary,
        viewer_hero_id=viewer,
        perspective_team=team,
    )


def _hero_token(observation: StableValueObservation, hero_id: str):
    return next(
        token
        for token in observation.state.tokens
        if token.kind == "HERO" and token.features.get("hero_id") == hero_id
    )


def test_encodes_real_actor_ready_boundary_with_fixed_viewer_and_actor_context() -> None:
    state, boundary = _actor_ready_state()

    observation = _encode(state, boundary)

    assert observation.schema_version == 1
    assert observation.boundary_kind == "ACTOR_READY"
    assert observation.state.viewer.private_hero_id == "hero_wasp"
    assert observation.state.viewer.perspective_team == "RED"
    actor = _hero_token(observation, "hero_wasp")
    assert actor.features["relation"] == "SELF"
    assert actor.features["is_current_actor"] is True
    assert actor.features["is_decision_owner"] is True


def test_encodes_real_clean_planning_boundary_without_policy_context() -> None:
    state = _planning_ready_state()
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    assert boundary.kind is StableValueBoundaryKind.PLANNING_READY

    observation = _encode(state, boundary)
    payload = observation.model_dump(mode="json")
    encoded = canonical_json_bytes(observation)

    assert observation.boundary_kind == "PLANNING_READY"
    assert set(payload) == {"schema_version", "state", "boundary_kind"}
    assert observation.state.candidate_ids == ()
    assert observation.state.features == {}
    assert b'"candidates"' not in encoded
    assert b'"input_request_type"' not in encoded
    assert b'"semantic_role"' not in encoded
    assert b"CONFIRM" not in encoded


def test_stable_value_observation_round_trips_canonically() -> None:
    state, boundary = _actor_ready_state()
    observation = _encode(state, boundary)

    encoded = canonical_json_bytes(observation)

    assert from_canonical_json(StableValueObservation, encoded) == observation
    assert canonical_json_bytes(from_canonical_json(StableValueObservation, encoded)) == encoded


def test_hidden_opponent_card_identity_does_not_change_value_observation() -> None:
    state = _planning_ready_state()
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    changed = state.model_copy(deep=True)
    opponent = changed.get_hero(HeroID("hero_arien"))
    assert opponent is not None and opponent.hand
    opponent.hand[0].id = "private_replacement_card"
    changed_boundary = detect_stable_value_boundary(changed)
    assert changed_boundary == boundary

    baseline = canonical_json_bytes(_encode(state, boundary))
    replaced = canonical_json_bytes(_encode(changed, changed_boundary))

    assert replaced == baseline


def test_foreign_actor_does_not_change_viewer_entitlement() -> None:
    state, boundary = _actor_ready_state()
    changed = state.model_copy(deep=True)
    actor = changed.get_hero(HeroID("hero_wasp"))
    assert actor is not None and actor.hand
    actor.hand[0].id = "private_actor_card"

    observation = _encode(state, boundary, viewer="hero_arien", team=TeamColor.BLUE)
    hidden_changed = _encode(changed, boundary, viewer="hero_arien", team=TeamColor.BLUE)

    assert canonical_json_bytes(observation) == canonical_json_bytes(hidden_changed)
    assert observation.state.viewer.private_hero_id == "hero_arien"
    assert observation.state.viewer.perspective_team == "BLUE"
    assert _hero_token(observation, "hero_arien").features["relation"] == "SELF"
    assert _hero_token(observation, "hero_wasp").features["is_current_actor"] is True
    assert _hero_token(observation, "hero_wasp").features["is_decision_owner"] is True
    assert canonical_json_bytes(observation) != canonical_json_bytes(_encode(state, boundary))


def test_planning_encoding_ignores_actor_left_by_a_finished_effect() -> None:
    # Raw finalization isolates a real engine transition rather than a hero's
    # particular card effect. The logging effect changes no public position.
    register_all_effects()
    plain, _ = _actor_ready_state()
    delayed = plain.model_copy(deep=True)
    delayed.active_effects.append(
        ActiveEffect(
            id="finished_actor_context",
            source_id="hero_arien",
            effect_type=EffectType.DELAYED_TRIGGER,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_TURN,
            created_at_turn=delayed.turn,
            created_at_round=delayed.round,
            finishing_steps=[LogMessageStep(message="finished")],
        )
    )
    for state in (plain, delayed):
        state.execution_stack.clear()
        push_steps(state, [FinalizeHeroTurnStep(hero_id="hero_wasp")])
        process_stack(state)
    assert delayed.current_actor_id == "hero_arien"
    boundary = detect_stable_value_boundary(delayed)
    assert boundary is not None and boundary.kind is StableValueBoundaryKind.PLANNING_READY
    assert detect_stable_value_boundary(plain) == boundary

    observation = _encode(delayed, boundary)

    assert canonical_json_bytes(observation) == canonical_json_bytes(_encode(plain, boundary))
    for hero_id in ("hero_wasp", "hero_arien"):
        features = _hero_token(observation, hero_id).features
        assert features["is_current_actor"] is False
        assert features["is_decision_owner"] is False
    # Encoding normalizes a private projected snapshot, never the live game.
    assert delayed.current_actor_id == "hero_arien"


@pytest.mark.parametrize("mutation", ["round", "turn", "actor", "kind"])
def test_rejects_bogus_or_stale_boundary(mutation: str) -> None:
    state, boundary = _actor_ready_state()
    replacements = {
        "round": {"round": boundary.round + 1},
        "turn": {"turn": boundary.turn + 1},
        "actor": {"actor_id": "hero_arien"},
        "kind": {"kind": StableValueBoundaryKind.PLANNING_READY},
    }
    stale = replace(boundary, **replacements[mutation])

    with pytest.raises(ValueError, match=r"boundary|match|stable"):
        _encode(state, stale)


def test_rejects_partial_planning_and_terminal_states() -> None:
    planning = _planning_ready_state()
    clean_boundary = detect_stable_value_boundary(planning)
    assert clean_boundary is not None
    wasp = planning.get_hero(HeroID("hero_wasp"))
    assert wasp is not None and wasp.hand
    commit_card(planning, wasp.id, wasp.hand[0])
    assert detect_stable_value_boundary(planning) is None

    with pytest.raises(ValueError, match=r"boundary|stable"):
        _encode(planning, clean_boundary)

    terminal = _planning_ready_state()
    terminal.phase = GamePhase.GAME_OVER
    terminal.winner = TeamColor.RED
    assert detect_stable_value_boundary(terminal) is None

    with pytest.raises(ValueError, match=r"boundary|stable"):
        _encode(terminal, clean_boundary)


def test_rejects_unknown_viewer_and_mismatched_perspective() -> None:
    state, boundary = _actor_ready_state()

    with pytest.raises(ValueError, match="viewer hero does not exist"):
        _encode(state, boundary, viewer="hero_missing")
    with pytest.raises(ValueError, match="does not belong"):
        _encode(state, boundary, team=TeamColor.BLUE)


@pytest.mark.parametrize("field", ["candidates", "input_request_type", "policy_target"])
def test_value_contract_rejects_policy_decision_fields(field: str) -> None:
    state, boundary = _actor_ready_state()
    payload = _encode(state, boundary).model_dump(mode="json")
    payload[field] = []

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StableValueObservation.model_validate(payload)
