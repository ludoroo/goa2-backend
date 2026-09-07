from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from automata.search.root import RootMismatchError, RootTarget, validate_root, validate_root_legal
from goa2.domain.models import TeamColor
from goa2.engine.setup import GameSetup


@dataclass
class _Decision:
    kind: str
    hero: Any | None = None
    request: Any | None = None


def test_validate_root_legal_preserves_caller_order_and_rejects_drift() -> None:
    assert validate_root_legal(["b", "a"], ["a", "b"]) == ("b", "a")
    with pytest.raises(RootMismatchError, match=r"missing=.*extra="):
        validate_root_legal(["a", "a"], ["a", "b"])


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
