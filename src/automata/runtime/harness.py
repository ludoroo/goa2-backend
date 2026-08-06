"""Headless self-play harness.

Drives a full game between agents through the engine's `GameSession`, with no
web server. Deterministic given a seed. This is both the smoke test for the
integration and the substrate the eval harness / MCTS will build on.

The per-turn decision logic (who acts next, how to answer an input request,
how to close Emmitt's two-card window) lives in :mod:`automata.runtime.driver`.
This module is now a thin loop over ``inspect_next_decision`` /
``apply_decision`` plus recording — sharing decision semantics with the live
server bot coordinator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goa2.domain.input import selection_value
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.state import GameState
from goa2.engine.session import GameSession, SessionResult, SessionResultType
from goa2.engine.setup import GameSetup

from ..agents.base import Agent, PlanningKind
from .driver import (
    BotDecision,
    DecisionKind,
    apply_decision,
    inspect_next_decision,
)
from .effects import register_all_effects
from .trajectory import NullRecorder, TrajectoryRecorder

DEFAULT_MAP = str(
    Path(__file__).resolve().parents[2] / "goa2" / "data" / "maps" / "forgotten_island.json"
)


@dataclass
class RunResult:
    winner: str | None
    rounds: int
    turns: int
    steps: int
    reason: str


def _team_of_request(state: GameState, player_id: str) -> str | None:
    """Team responsible for a decision addressed to ``player_id`` (best-effort)."""
    if player_id.startswith("team:"):
        return player_id.split(":", 1)[1]
    for team in state.teams.values():
        for hero in team.heroes:
            if hero.id == player_id and hero.team is not None:
                return hero.team.value
    return None


def _record_decision(
    rec: TrajectoryRecorder,
    state: GameState,
    decision: BotDecision,
) -> None:
    """Emit a trajectory entry for one applied :class:`BotDecision`.

    Recorded ``legal_keys`` for INPUT decisions use the canonical
    :func:`goa2.domain.input.selection_value` conversion — the same value the
    engine will accept as ``InputResponse.selection`` — so trajectories key
    consistently with the search-agent's raw-action space (see
    ``automata.search.ismcts._input_raw_map``).
    """
    if decision.kind is DecisionKind.PLANNING:
        plan = decision.planning
        assert plan is not None
        hero = state.get_hero(decision.hero_id)
        team = hero.team.value if hero and hero.team else ""
        legal = [c.id for c in hero.hand] if hero else []
        if plan.kind is PlanningKind.COMMIT:
            assert plan.card is not None
            chosen: Any = plan.card.id
        elif plan.kind is PlanningKind.FINISH:
            chosen = "FINISH"
        else:  # PASS
            chosen = None
        rec.record_decision(
            state=state,
            team=team,
            decision_kind="CARD",
            player_id=str(decision.hero_id),
            legal_keys=legal,
            chosen_key=chosen,
        )
        return

    # INPUT
    request = decision.request
    assert request is not None
    legal_keys: list[Any] = [selection_value(o) for o in request.options]
    # ``"SKIP"`` is a legal answer whenever the engine advertises it, even if
    # the request carries no options. Recording it in ``legal_keys`` keeps
    # trajectories self-consistent: a chosen ``"SKIP"`` must be a member of
    # the enumerated legal set for downstream policy learning to key on it.
    if request.can_skip and "SKIP" not in legal_keys:
        legal_keys.append("SKIP")
    rec.record_decision(
        state=state,
        team=(_team_of_request(state, request.player_id) or ""),
        decision_kind="INPUT",
        player_id=request.player_id,
        legal_keys=legal_keys,
        chosen_key=decision.selection,
    )


def _validate_agent_coverage(state: GameState, agents: Mapping[str, Agent]) -> None:
    """Fail fast when the harness is missing an agent for any hero.

    The headless harness is exclusively bot-vs-bot; every hero must be mapped
    or the driver will legitimately return ``None`` mid-game (a human's
    turn) and the harness would either livelock or terminate as a draw.
    Instead we raise ``ValueError`` up front with the missing hero list so
    the caller sees the mistake at setup time.
    """
    missing = [
        h.id
        for team in state.teams.values()
        for h in team.heroes
        if h.id not in agents
    ]
    if missing:
        raise ValueError(
            "run_game requires an agent for every hero; missing "
            f"{sorted(missing)!r}"
        )


def _winner_from(result: SessionResult | None, state: GameState) -> str | None:
    """Derive a winner string from the last :class:`SessionResult` / state.

    - When the last result was ``GAME_OVER``, its ``winner`` field is
      already set by :meth:`GameSession._build_result` from the engine's
      authoritative outcome markers.
    - Otherwise we read the same public state fields (``individual_winner_id``,
      ``winner``) or fall back to life-counter annihilation.

    Kept as a small local helper (rather than reaching into
    :meth:`GameSession._determine_winner`) so the harness depends only on
    public contract.
    """
    if result is not None and result.result_type is SessionResultType.GAME_OVER:
        return result.winner

    if state.individual_winner_id is not None:
        return str(state.individual_winner_id)
    if state.winner is not None:
        return state.winner.value

    red = state.teams.get(TeamColor.RED)
    blue = state.teams.get(TeamColor.BLUE)
    if red is not None and blue is not None:
        if red.life_counters <= 0:
            return "BLUE"
        if blue.life_counters <= 0:
            return "RED"
    return None


def run_game(
    red_heroes: list[str],
    blue_heroes: list[str],
    agents: Mapping[str, Agent],
    *,
    map_path: str = DEFAULT_MAP,
    game_type: str = "QUICK",
    seed: int = 0,
    max_steps: int = 20_000,
    recorder: TrajectoryRecorder | None = None,
) -> RunResult:
    """Play one game to completion; return the outcome.

    ``agents`` maps hero_id -> Agent. Every hero on the created state must be
    covered — a missing agent raises ``ValueError`` up front rather than
    livelocking mid-game. A single agent instance may control several
    heroes. The headless harness is bot-only; the server coordinator uses
    the same driver against mixed bot/human games.

    When ``recorder`` is given, a full state snapshot + decision context is
    emitted per decision, and the final outcome at game end (Seam 4 —
    self-play training data). Recording is off by default and perf-neutral
    when absent.
    """
    register_all_effects()
    rec: TrajectoryRecorder = recorder if recorder is not None else NullRecorder()
    state = GameSetup.create_game(
        map_path=map_path,
        red_heroes=red_heroes,
        blue_heroes=blue_heroes,
        game_type=game_type,
        seed=seed,
    )
    _validate_agent_coverage(state, agents)

    session = GameSession(state)

    steps = 0
    last_result: SessionResult | None = None

    while steps < max_steps:
        steps += 1

        if state.phase == GamePhase.GAME_OVER:
            winner = _winner_from(last_result, state)
            rec.record_outcome(winner=winner, rounds=state.round, reason="game_over")
            return RunResult(
                winner=winner,
                rounds=state.round,
                turns=state.turn,
                steps=steps,
                reason="game_over",
            )

        decision = inspect_next_decision(state, agents, last_result)

        if decision is None:
            # In a fully bot-covered game, ``None`` from the driver only
            # legitimately happens outside PLANNING when there is no pending
            # request — the engine just needs to be nudged with ``advance()``
            # to surface the next request / action. During PLANNING, ``None``
            # would mean every uncommitted hero is human, which cannot occur
            # under the coverage-checked bot-only harness — fail fast rather
            # than livelock.
            if state.phase == GamePhase.PLANNING:
                raise RuntimeError(
                    "run_game: no bot decision available during PLANNING but "
                    "the game requires progress; coverage was validated at "
                    "setup, so this indicates a driver contract bug"
                )
            last_result = session.advance()
            if last_result.result_type is SessionResultType.GAME_OVER:
                winner = _winner_from(last_result, state)
                rec.record_outcome(winner=winner, rounds=state.round, reason="game_over")
                return RunResult(
                    winner=winner,
                    rounds=state.round,
                    turns=state.turn,
                    steps=steps,
                    reason="game_over",
                )
            continue

        _record_decision(rec, state, decision)
        last_result = apply_decision(session, decision)

        if last_result.result_type is SessionResultType.GAME_OVER:
            winner = _winner_from(last_result, state)
            rec.record_outcome(winner=winner, rounds=state.round, reason="game_over")
            return RunResult(
                winner=winner,
                rounds=state.round,
                turns=state.turn,
                steps=steps,
                reason="game_over",
            )

    rec.record_outcome(winner=None, rounds=state.round, reason="max_steps")
    return RunResult(
        winner=None, rounds=state.round, turns=state.turn, steps=steps, reason="max_steps"
    )
