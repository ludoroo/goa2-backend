from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.search.ismcts.engine import validate_search_root
from automata.search.root import RootMismatchError, RootTarget, validate_root, validate_root_legal
from goa2.domain.input import InputRequest, InputRequestType
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.types import HeroID
from goa2.engine.handler import push_steps
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup
from goa2.engine.steps import ConfirmResolutionStep


@dataclass
class _Decision:
    kind: str
    hero: Any | None = None
    request: Any | None = None


@pytest.mark.parametrize(
    ("request_id", "player_id"),
    (("different-request", "hero_wasp"), ("surfaced-request", "hero_arien")),
)
def test_input_root_rejects_request_identity_mismatch(request_id: str, player_id: str) -> None:
    request = InputRequest(
        id="surfaced-request",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
    )

    with pytest.raises(ValueError, match="request"):
        RootTarget.input(
            request_id=request_id,
            player_id=player_id,
            request=request,
            owned_hero_ids=frozenset({"hero_wasp"}),
            decision_owner_hero_id="hero_wasp",
        )


def test_validate_root_legal_preserves_caller_order_and_rejects_drift() -> None:
    assert validate_root_legal(["b", "a"], ["a", "b"]) == ("b", "a")
    with pytest.raises(RootMismatchError, match=r"missing=.*extra="):
        validate_root_legal(["a", "a"], ["a", "b"])


def test_validate_search_root_preserves_a_live_confirm_resolution_prompt() -> None:
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=2,
    )
    state.phase = GamePhase.RESOLUTION
    state.current_actor_id = HeroID("hero_wasp")
    state.resolution_owner_id = HeroID("hero_wasp")
    push_steps(state, [ConfirmResolutionStep(hero_id="hero_wasp")])
    live_result = GameSession(state).advance()
    request = live_result.input_request
    assert request is not None

    target = RootTarget.input(
        request_id=request.id,
        player_id=request.player_id,
        owned_hero_ids=frozenset({"hero_wasp"}),
        decision_owner_hero_id="hero_wasp",
    )
    validated = validate_search_root(
        state,
        TeamColor.RED,
        target,
        ["CONFIRM"],
        HeuristicAgent(seed=1),
    )

    assert validated.decision.request is not None
    assert validated.decision.request.id == request.id


def test_validate_root_treats_none_as_a_valid_singleton() -> None:
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=2,
    )
    target = RootTarget.card(hero_id="hero_wasp", owned_hero_ids=frozenset({"hero_wasp"}))
    validated = validate_root(
        state,
        TeamColor.RED,
        target,
        _Decision("CARD", hero=SimpleNamespace(id="hero_wasp")),
        legal_candidates=[None],
        canonical_legal=[None],
    )
    assert validated.is_singleton
    assert validated.singleton_candidate is None
