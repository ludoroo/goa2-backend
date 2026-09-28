from __future__ import annotations

import pytest

from automata.harness.game_runner import RunResult
from automata.runtime.outcomes import resolve_terminal_winner_side
from goa2.domain.models import TeamColor
from goa2.engine.setup import GameSetup


def _state():
    return GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=73,
    )


@pytest.mark.parametrize(
    ("raw_winner", "expected"),
    [
        ("RED", "RED"),
        ("red", "RED"),
        ("BLUE", "BLUE"),
        ("blue", "BLUE"),
        ("hero_wasp", "RED"),
        ("hero_arien", "BLUE"),
    ],
)
def test_resolve_terminal_winner_side_uses_authoritative_rosters(
    raw_winner: str | None, expected: str | None
) -> None:
    assert resolve_terminal_winner_side(_state(), raw_winner) == expected


def test_resolve_terminal_winner_side_rejects_missing_winner() -> None:
    with pytest.raises(ValueError, match="missing a winner"):
        resolve_terminal_winner_side(_state(), None)


@pytest.mark.parametrize("raw_winner", ["GREEN", "hero_missing", "Wasp"])
def test_resolve_terminal_winner_side_rejects_unknown_values(raw_winner: str) -> None:
    with pytest.raises(ValueError, match="unknown terminal winner"):
        resolve_terminal_winner_side(_state(), raw_winner)


def test_resolve_terminal_winner_side_rejects_piece_ids() -> None:
    state = GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Razzle"],
        ["Arien"],
        game_type="QUICK",
        seed=73,
    )

    with pytest.raises(ValueError, match="unknown terminal winner"):
        resolve_terminal_winner_side(state, str(state.get_piece_ids("hero_razzle")[0]))


def test_resolve_terminal_winner_side_rejects_missing_team() -> None:
    state = _state()
    del state.teams[TeamColor.BLUE]

    with pytest.raises(ValueError, match="not present"):
        resolve_terminal_winner_side(state, "BLUE")


def test_resolve_terminal_winner_side_rejects_roster_inconsistency() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    state.teams[TeamColor.BLUE].heroes.append(hero)

    with pytest.raises(ValueError, match="roster"):
        resolve_terminal_winner_side(state, "hero_wasp")


@pytest.mark.parametrize(
    "result",
    [
        RunResult("hero_wasp", 1, 2, 3, "game_over", winner_side="RED"),
        RunResult(None, 1, 2, 3, "game_over", winner_side=None),
        RunResult(None, 1, 2, 3, "max_steps", winner_side=None),
    ],
)
def test_run_result_accepts_explicit_consistent_outcomes(result: RunResult) -> None:
    assert result.winner_side in {"RED", None}


def test_run_result_requires_explicit_winner_side() -> None:
    with pytest.raises(TypeError):
        RunResult("RED", 1, 2, 3, "game_over")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"winner": "RED", "reason": "game_over", "winner_side": "BLUE"},
        {"winner": "hero_wasp", "reason": "game_over", "winner_side": None},
        {"winner": None, "reason": "game_over", "winner_side": "RED"},
        {"winner": "RED", "reason": "max_steps", "winner_side": "RED"},
        {"winner": None, "reason": "max_steps", "winner_side": "GREEN"},
    ],
)
def test_run_result_rejects_inconsistent_outcomes(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RunResult(rounds=1, turns=2, steps=3, **kwargs)  # type: ignore[arg-type]
