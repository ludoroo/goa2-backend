"""Canonical terminal-outcome normalization at the engine boundary."""

from __future__ import annotations

from typing import Literal

from goa2.domain.models import TeamColor
from goa2.domain.state import GameState

WinnerSide = Literal["RED", "BLUE"]


def resolve_terminal_winner_side(state: GameState, raw_winner: str | None) -> WinnerSide:
    """Resolve an engine winner token to its authoritative team side.

    Team names are case-insensitive. Hero winners must be exact IDs in one
    authoritative team roster; display names, inferred ID prefixes, and
    multi-piece hero piece IDs are not accepted. The engine currently has no
    terminal draw rule: a missing winner is invalid, not evidence of a draw.
    """
    if raw_winner is None:
        raise ValueError("terminal outcome is missing a winner; no engine draw rule exists")

    normalized = raw_winner.upper()
    named_team = next(
        (team for team in TeamColor if normalized == team.value.upper()),
        None,
    )
    if named_team is not None:
        if named_team not in state.teams:
            raise ValueError(f"winning team {named_team.value!r} is not present in game state")
        return named_team.value

    matches = [
        (roster_team, hero)
        for roster_team, roster in state.teams.items()
        for hero in roster.heroes
        if str(hero.id) == raw_winner
    ]
    if not matches:
        raise ValueError(f"unknown terminal winner {raw_winner!r}")
    if len(matches) != 1:
        raise ValueError(f"terminal winner {raw_winner!r} has inconsistent team rosters")

    roster_team, hero = matches[0]
    if hero.team is None:
        raise ValueError(f"terminal winner {raw_winner!r} has no team")
    if hero.team != roster_team:
        raise ValueError(f"terminal winner {raw_winner!r} has inconsistent team roster membership")
    return roster_team.value


__all__ = ["WinnerSide", "resolve_terminal_winner_side"]
