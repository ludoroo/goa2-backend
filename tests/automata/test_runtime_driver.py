from __future__ import annotations

import pytest

from automata.agents.contracts import PlanningDecision
from automata.runtime.driver import (
    BotDecision,
    DecisionKind,
    IllegalBotDecisionError,
    apply_decision,
)
from automata.runtime.effects import register_all_effects
from goa2.domain.types import HeroID
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup


def _session() -> GameSession:
    register_all_effects()
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=7,
    )
    return GameSession(state)


def test_apply_planning_commit_resolves_the_live_card_by_id() -> None:
    session = _session()
    hero_id = HeroID("hero_wasp")
    hero = session.state.get_hero(hero_id)
    assert hero is not None
    live_card = hero.hand[0]
    detached_card = live_card.model_copy(deep=True)
    assert detached_card is not live_card

    apply_decision(
        session,
        BotDecision(
            kind=DecisionKind.PLANNING,
            hero_id=hero_id,
            planning=PlanningDecision.commit(detached_card),
        ),
    )

    assert session.state.pending_inputs[hero_id] is live_card


def test_apply_planning_commit_rejects_a_card_id_absent_from_the_live_hand() -> None:
    session = _session()
    hero_id = HeroID("hero_wasp")
    hero = session.state.get_hero(hero_id)
    assert hero is not None
    missing_card = hero.hand[0].model_copy(update={"id": "missing-card"})

    with pytest.raises(IllegalBotDecisionError, match=r"missing-card.*not in live hand"):
        apply_decision(
            session,
            BotDecision(
                kind=DecisionKind.PLANNING,
                hero_id=hero_id,
                planning=PlanningDecision.commit(missing_card),
            ),
        )
