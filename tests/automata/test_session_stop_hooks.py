from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from automata.agents.contracts import PlanningDecision
from automata.runtime.driver import BotDecision, DecisionKind, apply_decision
from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import (
    StableValueBoundaryKind,
    capture_transition_anchor,
    detect_stable_value_boundary,
    should_stop_before_stable_boundary,
)
from goa2.domain.input import InputRequest, InputRequestType
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.phases import commit_card as commit_card_to_state
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup


@pytest.fixture(autouse=True)
def _effects() -> None:
    register_all_effects()


def _game(red: list[str] | None = None, blue: list[str] | None = None):
    return GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        red or ["Wasp"],
        blue or ["Arien"],
        game_type="QUICK",
        seed=73,
    )


def _assert_actor_ready(state) -> None:
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    assert boundary.kind is StableValueBoundaryKind.ACTOR_READY


def test_final_commit_forwards_stop_hook_into_automatic_resolution() -> None:
    state = _game()
    session = GameSession(state)
    red = state.teams[TeamColor.RED].heroes[0]
    blue = state.teams[TeamColor.BLUE].heroes[0]
    session.commit_card(HeroID(red.id), red.hand[0])
    anchor = capture_transition_anchor(state)

    session.commit_card(
        HeroID(blue.id),
        blue.hand[0],
        stop_before_step=lambda current, step: should_stop_before_stable_boundary(
            anchor, current, step
        ),
    )

    _assert_actor_ready(state)


def test_final_pass_forwards_stop_hook_into_automatic_resolution() -> None:
    state = _game()
    red = state.teams[TeamColor.RED].heroes[0]
    blue = state.teams[TeamColor.BLUE].heroes[0]
    red.hand.clear()
    commit_card_to_state(state, HeroID(blue.id), blue.hand[0])
    session = GameSession(state)
    anchor = capture_transition_anchor(state)

    session.pass_turn(
        HeroID(red.id),
        stop_before_step=lambda current, step: should_stop_before_stable_boundary(
            anchor, current, step
        ),
    )

    _assert_actor_ready(state)


def test_finish_planning_forwards_stop_hook_for_emmitt() -> None:
    state = _game(["Emmitt"], ["Wasp"])
    emmitt = state.teams[TeamColor.RED].heroes[0]
    opponent = state.teams[TeamColor.BLUE].heroes[0]
    emmitt.level = 8
    commit_card_to_state(state, HeroID(emmitt.id), emmitt.hand[0])
    commit_card_to_state(state, HeroID(opponent.id), opponent.hand[0])
    assert state.phase is GamePhase.PLANNING
    session = GameSession(state)
    anchor = capture_transition_anchor(state)

    session.finish_planning(
        HeroID(emmitt.id),
        stop_before_step=lambda current, step: should_stop_before_stable_boundary(
            anchor, current, step
        ),
    )

    _assert_actor_ready(state)


class _RecordingSession:
    def __init__(self, live_card: object | None = None) -> None:
        self.calls: list[tuple[str, object, object]] = []
        self.state = type(
            "RecordingState",
            (),
            {
                "get_hero": lambda _self, _hero_id: type(
                    "RecordingHero", (), {"hand": [live_card] if live_card is not None else []}
                )()
            },
        )()

    def commit_card(self, hero_id, card, **kwargs):
        self.calls.append(("commit", hero_id, kwargs))
        return "commit-result"

    def finish_planning(self, hero_id, **kwargs):
        self.calls.append(("finish", hero_id, kwargs))
        return "finish-result"

    def pass_turn(self, hero_id, **kwargs):
        self.calls.append(("pass", hero_id, kwargs))
        return "pass-result"

    def advance(self, response, **kwargs):
        self.calls.append(("input", response, kwargs))
        return "input-result"


@pytest.mark.parametrize(
    ("decision", "expected_name"),
    [
        (
            BotDecision(
                kind=DecisionKind.PLANNING,
                hero_id=HeroID("hero_a"),
                planning=PlanningDecision.commit(SimpleNamespace(id="card")),  # type: ignore[arg-type]
            ),
            "commit",
        ),
        (
            BotDecision(
                kind=DecisionKind.PLANNING,
                hero_id=HeroID("hero_a"),
                planning=PlanningDecision.finish(),
            ),
            "finish",
        ),
        (
            BotDecision(
                kind=DecisionKind.PLANNING,
                hero_id=HeroID("hero_a"),
                planning=PlanningDecision.pass_(),
            ),
            "pass",
        ),
        (
            BotDecision(
                kind=DecisionKind.INPUT,
                hero_id=HeroID("hero_a"),
                request=InputRequest(
                    request_type=InputRequestType.SELECT_OPTION,
                    player_id="hero_a",
                ),
                selection="choice",
            ),
            "input",
        ),
    ],
)
def test_apply_decision_forwards_non_none_stop_hook_on_every_branch(
    decision: BotDecision,
    expected_name: str,
) -> None:
    planned_card = decision.planning.card if decision.planning is not None else None
    session = _RecordingSession(planned_card)

    def hook(_state: Any, _step: Any) -> bool:
        return False

    apply_decision(session, decision, stop_before_step=hook)  # type: ignore[arg-type]

    assert session.calls[0][0] == expected_name
    assert session.calls[0][2] == {"stop_before_step": hook}


def test_apply_decision_does_not_forward_none_to_existing_substitutes() -> None:
    class LegacySession:
        def pass_turn(self, hero_id):
            return hero_id

    decision = BotDecision(
        kind=DecisionKind.PLANNING,
        hero_id=HeroID("hero_a"),
        planning=PlanningDecision.pass_(),
    )

    assert apply_decision(LegacySession(), decision) == HeroID("hero_a")  # type: ignore[arg-type]
