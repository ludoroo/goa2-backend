import logging

from goa2.domain.board import Board
from goa2.domain.models import (
    ActionType,
    Card,
    CardColor,
    CardState,
    CardTier,
    GamePhase,
    Hero,
    Team,
    TeamColor,
)
from goa2.domain.models.effect import (
    ActiveEffect,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.state import GameState
from goa2.engine.handler import process_stack, push_steps
from goa2.engine.phases import resolve_next_action
from goa2.engine.session import GameSession, SessionResultType
from goa2.engine.steps import FindNextActorStep, LogMessageStep, ResolveTieBreakerStep


def _card(
    card_id: str,
    *,
    initiative: int = 0,
    state: CardState = CardState.UNRESOLVED,
    is_facedown: bool = False,
) -> Card:
    return Card(
        id=card_id,
        name=card_id,
        tier=CardTier.I,
        color=CardColor.RED,
        initiative=initiative,
        primary_action=ActionType.SKILL,
        primary_action_value=None,
        effect_id="test",
        effect_text="Test",
        state=state,
        is_facedown=is_facedown,
    )


def _state(*heroes: Hero) -> GameState:
    return GameState(
        board=Board(),
        teams={
            TeamColor.RED: Team(color=TeamColor.RED, heroes=list(heroes), minions=[]),
            TeamColor.BLUE: Team(color=TeamColor.BLUE, heroes=[], minions=[]),
        },
        phase=GamePhase.RESOLUTION,
    )


def test_resolve_next_action_prunes_stale_ids_but_preserves_valid_facedown_candidates(
    caplog,
):
    stale = Hero(id="stale", name="Stale", team=TeamColor.RED, deck=[])
    faceup = Hero(id="faceup", name="Faceup", team=TeamColor.RED, deck=[])
    facedown = Hero(id="facedown", name="Facedown", team=TeamColor.RED, deck=[])
    faceup.current_turn_card = _card("faceup_card")
    facedown.current_turn_card = _card("facedown_card", initiative=20, is_facedown=True)
    state = _state(stale, faceup, facedown)
    state.unresolved_hero_ids = ["missing", "stale", "faceup", "facedown"]

    with caplog.at_level(logging.WARNING, logger="goa2.engine.phases"):
        resolve_next_action(state)

    assert state.unresolved_hero_ids == ["faceup", "facedown"]
    assert isinstance(state.execution_stack[-1], ResolveTieBreakerStep)
    assert state.execution_stack[-1].tied_hero_ids == ["faceup", "facedown"]
    assert "missing" in caplog.text
    assert "stale" in caplog.text


def test_session_advance_does_not_stall_when_only_stale_unresolved_ids_remain():
    hero = Hero(
        id="stale",
        name="Stale",
        team=TeamColor.RED,
        deck=[],
        hand=[_card("next_turn_card", state=CardState.HAND)],
    )
    state = _state(hero)
    state.unresolved_hero_ids = ["stale", "missing"]
    push_steps(state, [FindNextActorStep()])
    session = GameSession(state)

    result = session.advance()

    assert result.result_type == SessionResultType.PHASE_CHANGED
    assert result.current_phase == GamePhase.PLANNING
    assert result.events == []
    assert state.turn == 2
    assert state.unresolved_hero_ids == []
    assert state.current_actor_id is None
    assert state.execution_stack == []


def test_stale_cleanup_uses_deferred_end_turn_when_effect_has_finishing_steps():
    hero = Hero(
        id="stale",
        name="Stale",
        team=TeamColor.RED,
        deck=[],
        hand=[_card("next_turn_card", state=CardState.HAND)],
    )
    state = _state(hero)
    state.unresolved_hero_ids = ["stale"]
    state.active_effects.append(
        ActiveEffect(
            id="finishing_effect",
            source_id="stale",
            effect_type=EffectType.DELAYED_TRIGGER,
            scope=EffectScope(shape=Shape.GLOBAL),
            duration=DurationType.THIS_TURN,
            created_at_turn=state.turn,
            created_at_round=state.round,
            finishing_steps=[LogMessageStep(message="finished")],
        )
    )

    resolve_next_action(state)

    assert state.phase == GamePhase.RESOLUTION
    assert state.turn == 1
    assert state.unresolved_hero_ids == []
    assert state.execution_stack

    process_stack(state)

    assert state.phase == GamePhase.PLANNING
    assert state.turn == 2
    assert state.execution_stack == []
