"""Smoke tests and behavioral coverage for the headless harness and the
server-neutral bot driver.

The two original tests (:func:`test_random_quick_game_completes` and
:func:`test_random_game_is_deterministic`) prove the full end-to-end loop —
that :func:`automata.runtime.run_game` can drive a real game to completion via
the engine's :class:`GameSession` and that a fixed seed reproduces the same
winner.

The rest of the file covers the driver contract
(``automata.runtime.driver``): who decides what, when the driver refuses to
answer, and how PLANNING / INPUT-request / UPGRADE_PHASE decisions are
shaped. These are unit-level tests that hit :func:`inspect_next_decision` /
:func:`apply_decision` directly rather than going through :func:`run_game`.

Contract highlights covered here:

- :func:`inspect_next_decision` takes ``GameState`` plus the last
  :class:`~goa2.engine.session.SessionResult` (or ``None`` at game start) and
  derives the pending :class:`InputRequest` internally.
- ``None`` from :func:`inspect_next_decision` is reserved for
  "no mapped owner / no current decision" — illegal bot output raises
  :class:`IllegalBotDecisionError` instead.
- Mixed human/bot ownership, team-addressed input, and UPGRADE_PHASE
  scoping are all exercised.
"""

from __future__ import annotations

from typing import Any

import pytest

import goa2.scripts.emmitt_effects  # noqa: F401  (registers alternative_timelines)
from automata.agents import PlanningKind
from automata.agents.base import PlanningDecision
from automata.agents.random_agent import RandomAgent
from automata.runtime.driver import (
    BotDecision,
    DecisionKind,
    IllegalBotDecisionError,
    apply_decision,
    inspect_next_decision,
)
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP, run_game
from goa2.domain.input import (
    InputOption,
    InputRequest,
    InputRequestType,
    create_input_request,
)
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.session import (
    GameSession,
    SessionResult,
    SessionResultType,
)
from goa2.engine.setup import GameSetup

# --------------------------------------------------------------------------- #
# Existing end-to-end smoke.
# --------------------------------------------------------------------------- #


def test_random_quick_game_completes() -> None:
    # Quick game, 2v2, recommended roster (Wasp, Xargatha / Arien, Brogan).
    red = ["Wasp", "Xargatha"]
    blue = ["Arien", "Brogan"]

    # One shared random agent (seeded) controls everyone; deterministic.
    agent = RandomAgent(seed=42)
    hero_ids = ["hero_wasp", "hero_xargatha", "hero_arien", "hero_brogan"]
    agents = {hid: agent for hid in hero_ids}

    result = run_game(red, blue, agents, seed=7)

    # The game should terminate with a real result, not hit the step cap.
    assert result.reason == "game_over", f"did not finish: {result}"
    assert result.winner in {"RED", "BLUE", "red", "blue"}, f"unexpected winner {result.winner!r}"
    assert result.rounds >= 1


def test_random_game_is_deterministic() -> None:
    red = ["Wasp", "Xargatha"]
    blue = ["Arien", "Brogan"]
    hero_ids = ["hero_wasp", "hero_xargatha", "hero_arien", "hero_brogan"]

    def play() -> str | None:
        agents = {hid: RandomAgent(seed=1) for hid in hero_ids}
        return run_game(red, blue, agents, seed=123).winner

    assert play() == play()


# --------------------------------------------------------------------------- #
# Fixtures & helpers.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module", autouse=True)
def _effects_registered() -> None:
    """The driver depends on ``hero_can_play_two_cards`` which reads the
    effect registry (Emmitt's ultimate flag). Register once per module."""
    register_all_effects()


def _new_game(red: list[str], blue: list[str], *, seed: int = 1) -> GameState:
    return GameSetup.create_game(
        map_path=DEFAULT_MAP,
        red_heroes=red,
        blue_heroes=blue,
        game_type="QUICK",
        seed=seed,
    )


def _input_needed_result(state: GameState, request: InputRequest) -> SessionResult:
    """Build a ``SessionResult`` in the ``INPUT_NEEDED`` shape.

    Test doubles construct these directly to exercise the driver without
    going through the engine, matching the shape :meth:`GameSession.advance`
    would return for a pending input.
    """
    return SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=request,
        current_phase=state.phase,
    )


def _non_input_result(state: GameState) -> SessionResult:
    """A neutral ``ACTION_COMPLETE`` result (no pending request)."""
    return SessionResult(
        result_type=SessionResultType.ACTION_COMPLETE,
        current_phase=state.phase,
    )


class _StubAgent:
    """Minimal :class:`Agent` implementation with controllable behavior.

    Configured with per-hero card selections and a global (or callable)
    input-selection value. Records every call to ``.calls`` so tests can
    assert the driver did not consult an agent it shouldn't have.
    """

    def __init__(
        self,
        *,
        card_for: dict[str, Card | None] | None = None,
        input_selection: Any = "SKIP",
    ) -> None:
        self._card_for = card_for or {}
        self._input_selection = input_selection
        self.calls: list[tuple[str, str]] = []  # (method, hero_or_request_id)
        # Populated by ``choose_input`` if invoked with the owned_hero_ids kwarg.
        self.last_owned: frozenset[str] | None = None

    def choose_card(self, state: GameState, hero: Hero) -> Card | None:
        self.calls.append(("choose_card", hero.id))
        if hero.id in self._card_for:
            return self._card_for[hero.id]
        return hero.hand[0] if hero.hand else None

    def choose_input(
        self,
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
    ) -> Any:
        self.calls.append(("choose_input", request.id))
        self.last_owned = owned_hero_ids
        sel = self._input_selection
        if callable(sel):
            return sel(state, request)
        return sel


# --------------------------------------------------------------------------- #
# Planning: mapping, human-owner, mixed teams.
# --------------------------------------------------------------------------- #


def test_planning_returns_none_when_no_bots_are_mapped() -> None:
    """No arbitrary fallback: an empty agent map produces no bot decision.

    ``last_result=None`` represents the very first tick of the game (before
    ``session.advance()`` has been called).
    """
    state = _new_game(["Wasp", "Xargatha"], ["Arien", "Brogan"])
    assert state.phase == GamePhase.PLANNING
    assert inspect_next_decision(state, agents={}, last_result=None) is None


def test_planning_returns_none_when_all_remaining_owners_are_human() -> None:
    """If the only heroes left to commit are unmapped (human), return None
    rather than pick a random mapped agent to fill in for them."""
    state = _new_game(["Wasp", "Xargatha"], ["Arien", "Brogan"])
    session = GameSession(state)
    bot = _StubAgent()
    # Wasp is the only bot; commit its first card manually then verify the
    # driver refuses to keep planning on behalf of the humans.
    wasp = state.teams[TeamColor.RED].heroes[0]
    session.commit_card(HeroID(wasp.id), wasp.hand[0])
    # Xargatha, Arien, Brogan are all "human" (unmapped).
    assert inspect_next_decision(state, agents={wasp.id: bot}, last_result=None) is None


def test_planning_picks_next_bot_hero_when_others_are_human() -> None:
    """Mixed teams: driver commits for the mapped bot even when a human
    teammate is uncommitted."""
    state = _new_game(["Wasp", "Xargatha"], ["Arien", "Brogan"])
    xarg = state.teams[TeamColor.RED].heroes[1]
    bot = _StubAgent(card_for={xarg.id: xarg.hand[0]})
    decision = inspect_next_decision(state, agents={xarg.id: bot}, last_result=None)
    assert decision is not None
    assert decision.kind is DecisionKind.PLANNING
    assert decision.hero_id == xarg.id
    assert decision.planning is not None
    assert decision.planning.kind is PlanningKind.COMMIT
    assert decision.planning.card is xarg.hand[0]


def test_apply_decision_commits_a_planning_card() -> None:
    state = _new_game(["Wasp"], ["Arien"])
    session = GameSession(state)
    wasp = state.teams[TeamColor.RED].heroes[0]
    card = wasp.hand[0]
    decision = BotDecision(
        kind=DecisionKind.PLANNING,
        hero_id=HeroID(wasp.id),
        planning=PlanningDecision.commit(card),
    )
    apply_decision(session, decision)
    assert state.pending_inputs.get(HeroID(wasp.id)) is card
    assert card not in wasp.hand


# --------------------------------------------------------------------------- #
# Emmitt: multi-card FINISH after a second commit is declined.
# --------------------------------------------------------------------------- #


def _emmitt_state() -> GameState:
    state = _new_game(["Emmitt"], ["Wasp"])
    emmitt = state.teams[TeamColor.RED].heroes[0]
    emmitt.level = 8  # Ultimate active — plays_two_cards path.
    return state


def test_emmitt_second_card_choose_none_maps_to_finish() -> None:
    """After Emmitt's first commit the driver re-asks the agent; ``None`` from
    ``choose_card`` produces a :class:`PlanningKind.FINISH`, not another
    COMMIT or a stray PASS."""
    state = _emmitt_state()
    session = GameSession(state)
    emmitt = state.teams[TeamColor.RED].heroes[0]
    first = emmitt.hand[0]
    session.commit_card(HeroID(emmitt.id), first)

    call_count = {"n": 0}

    def choose(state: GameState, hero: Hero) -> Card | None:
        call_count["n"] += 1
        return None  # "no second card"

    bot: Any = _StubAgent()
    bot.choose_card = choose  # type: ignore[assignment]

    decision = inspect_next_decision(
        state, agents={emmitt.id: bot}, last_result=None
    )
    assert decision is not None
    assert decision.kind is DecisionKind.PLANNING
    assert decision.hero_id == emmitt.id
    assert decision.planning is not None
    assert decision.planning.kind is PlanningKind.FINISH
    assert call_count["n"] == 1

    # Applying it closes Emmitt's planning and lets Wasp / phase progress.
    apply_decision(session, decision)
    assert HeroID(emmitt.id) in state.planning_done


def test_emmitt_second_card_choose_card_maps_to_second_commit() -> None:
    """If the agent picks a second card, driver produces COMMIT (not FINISH)."""
    state = _emmitt_state()
    session = GameSession(state)
    emmitt = state.teams[TeamColor.RED].heroes[0]
    first, second = emmitt.hand[0], emmitt.hand[1]
    session.commit_card(HeroID(emmitt.id), first)
    bot = _StubAgent(card_for={emmitt.id: second})
    decision = inspect_next_decision(
        state, agents={emmitt.id: bot}, last_result=None
    )
    assert decision is not None
    assert decision.planning is not None
    assert decision.planning.kind is PlanningKind.COMMIT
    assert decision.planning.card is second
    apply_decision(session, decision)
    assert HeroID(emmitt.id) in state.pending_second_cards


# --------------------------------------------------------------------------- #
# Input requests: SessionResult seam, team addressing, ownership.
# --------------------------------------------------------------------------- #


def _select_option_request(player_id: str, options: list[str]) -> InputRequest:
    return InputRequest(
        request_type=InputRequestType.SELECT_OPTION,
        player_id=player_id,
        options=[InputOption(id=o, text=o) for o in options],
    )


def test_input_request_seam_takes_input_needed_session_result() -> None:
    """Driver derives the pending request from a real ``SessionResult`` —
    callers never handle the ``pending_request`` seam themselves."""
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    bot = _StubAgent(input_selection="A")
    req = _select_option_request(wasp.id, ["A", "B"])
    result = _input_needed_result(state, req)
    decision = inspect_next_decision(state, {wasp.id: bot}, result)
    assert decision is not None
    assert decision.kind is DecisionKind.INPUT
    assert decision.selection == "A"
    assert decision.request is req


def test_non_input_session_result_falls_back_to_phase_inspection() -> None:
    """A non-``INPUT_NEEDED`` result carries no pending request — the driver
    should treat this like ``last_result=None`` and (during PLANNING) pick
    the next planning move."""
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    bot = _StubAgent(card_for={wasp.id: wasp.hand[0]})
    result = _non_input_result(state)
    decision = inspect_next_decision(state, {wasp.id: bot}, result)
    assert decision is not None
    assert decision.kind is DecisionKind.PLANNING
    assert decision.hero_id == wasp.id


def test_team_addressed_input_only_resolves_when_a_team_bot_is_mapped() -> None:
    """A ``team:XXX`` request routes to a bot only if a mapped bot sits on
    that team; otherwise ``None`` (no arbitrary fallback)."""
    state = _new_game(["Wasp"], ["Arien"])
    arien = state.teams[TeamColor.BLUE].heroes[0]
    bot = _StubAgent(input_selection="pick_a")
    req = _select_option_request("team:RED", ["pick_a", "pick_b"])
    result = _input_needed_result(state, req)
    assert inspect_next_decision(state, {arien.id: bot}, result) is None


def test_team_addressed_input_resolves_via_teammate_bot() -> None:
    """A ``team:RED`` request is answered by the first RED-side mapped bot,
    and the ``owned_hero_ids`` passed to the agent is the bot's own hero(es)
    on that team — never a still-uncommitted teammate."""
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    bot = _StubAgent(input_selection="pick_a")
    req = _select_option_request("team:RED", ["pick_a", "pick_b"])
    result = _input_needed_result(state, req)
    decision = inspect_next_decision(state, {wasp.id: bot}, result)
    assert decision is not None
    assert decision.kind is DecisionKind.INPUT
    assert decision.hero_id == wasp.id
    assert decision.request is req  # echoed for revalidation
    assert decision.selection == "pick_a"
    # Ownership coordination: driver hands the search-eligible responder set to
    # the agent, and it contains only bot-mapped heroes on the addressed team.
    assert bot.last_owned == frozenset({wasp.id})


def test_team_addressed_input_owned_hero_ids_excludes_human_teammate() -> None:
    """Two-hero team, one bot + one human on the same team: the bot's
    ``owned_hero_ids`` for a team-addressed request must be the bot only,
    never the human teammate."""
    state = _new_game(["Wasp", "Xargatha"], ["Arien", "Brogan"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    xarg = state.teams[TeamColor.RED].heroes[1]
    bot = _StubAgent(input_selection="X")
    req = _select_option_request("team:RED", ["X", "Y"])
    result = _input_needed_result(state, req)
    decision = inspect_next_decision(state, {wasp.id: bot}, result)
    assert decision is not None
    assert bot.last_owned == frozenset({wasp.id})
    assert xarg.id not in (bot.last_owned or frozenset())


def test_hero_addressed_input_returns_none_for_unmapped_owner() -> None:
    """A hero-id-addressed request whose owner has no bot mapping is left
    alone — the driver never falls back to a teammate."""
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    arien = state.teams[TeamColor.BLUE].heroes[0]
    bot = _StubAgent(input_selection="X")
    req = _select_option_request(wasp.id, ["X", "Y"])
    result = _input_needed_result(state, req)
    assert inspect_next_decision(state, {arien.id: bot}, result) is None


def test_hero_addressed_input_uses_owner_agent_with_singleton_owned_set() -> None:
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    arien = state.teams[TeamColor.BLUE].heroes[0]
    wasp_bot = _StubAgent(input_selection="A")
    arien_bot = _StubAgent(input_selection="B")
    req = _select_option_request(wasp.id, ["A", "B"])
    result = _input_needed_result(state, req)
    decision = inspect_next_decision(
        state, {wasp.id: wasp_bot, arien.id: arien_bot}, result
    )
    assert decision is not None
    assert decision.selection == "A"
    assert decision.hero_id == wasp.id
    assert wasp_bot.last_owned == frozenset({wasp.id})
    # Arien's agent must not have been consulted at all.
    assert not arien_bot.calls


# --------------------------------------------------------------------------- #
# UPGRADE_PHASE scoping.
# --------------------------------------------------------------------------- #


def _upgrade_request(players: dict[Any, dict[str, Any]]) -> InputRequest:
    return create_input_request(
        request_type=InputRequestType.UPGRADE_PHASE,
        player_id="simultaneous",
        prompt="Mandatory Upgrade Phase",
        players=players,
    )


def test_upgrade_phase_returns_none_when_only_humans_have_pending_upgrades() -> None:
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    arien = state.teams[TeamColor.BLUE].heroes[0]
    req = _upgrade_request(
        {arien.id: {"remaining": 1, "options": [{"pair": ["c1", "c2"]}]}}
    )
    bot = _StubAgent()
    result = _input_needed_result(state, req)
    assert inspect_next_decision(state, {wasp.id: bot}, result) is None


def test_upgrade_phase_scoped_to_bot_owned_pending_heroes() -> None:
    """The driver must never let a bot's agent see (and thus target) a
    human-owned pending upgrade."""
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    arien = state.teams[TeamColor.BLUE].heroes[0]
    players = {
        wasp.id: {"remaining": 1, "options": [{"pair": ["w1", "w2"]}]},
        arien.id: {"remaining": 1, "options": [{"pair": ["a1", "a2"]}]},
    }
    req = _upgrade_request(players)
    result = _input_needed_result(state, req)

    seen_players: dict[str, Any] = {}

    def choose(
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
    ) -> Any:
        seen_players.update(request.context.get("players") or {})
        return {"hero_id": wasp.id, "card_id": "w1"}

    bot: Any = _StubAgent()
    bot.choose_input = choose  # type: ignore[assignment]

    decision = inspect_next_decision(state, {wasp.id: bot}, result)
    assert decision is not None
    assert decision.kind is DecisionKind.INPUT
    # The scoped request handed to the agent must contain only Wasp — never
    # Arien — because Arien is unmapped (human).
    assert set(seen_players.keys()) == {wasp.id}
    assert decision.selection == {"hero_id": wasp.id, "card_id": "w1"}
    # The decision echoes the *original* request (unmodified) so the server
    # coordinator can revalidate it against its own state.
    assert decision.request is req


def test_upgrade_phase_passes_bot_owned_ids_when_agent_accepts_kwarg() -> None:
    """UPGRADE_PHASE decisions go through the same ``owned_hero_ids`` path
    as team-addressed input — the eligible set is the bot-owned pending
    hero(es), not the raw context players."""
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    arien = state.teams[TeamColor.BLUE].heroes[0]
    players = {
        wasp.id: {"remaining": 1, "options": [{"pair": ["w1", "w2"]}]},
        arien.id: {"remaining": 1, "options": [{"pair": ["a1", "a2"]}]},
    }
    req = _upgrade_request(players)
    result = _input_needed_result(state, req)
    bot = _StubAgent(input_selection={"hero_id": wasp.id, "card_id": "w1"})
    decision = inspect_next_decision(state, {wasp.id: bot}, result)
    assert decision is not None
    assert bot.last_owned == frozenset({wasp.id})


def test_upgrade_phase_skips_heroes_with_zero_remaining() -> None:
    """A bot hero with remaining=0 is not eligible even if listed."""
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    players = {wasp.id: {"remaining": 0, "options": [{"pair": ["w1", "w2"]}]}}
    req = _upgrade_request(players)
    result = _input_needed_result(state, req)
    bot = _StubAgent()
    assert inspect_next_decision(state, {wasp.id: bot}, result) is None


# --------------------------------------------------------------------------- #
# IllegalBotDecisionError paths.
# --------------------------------------------------------------------------- #


def test_planning_raises_illegal_decision_for_card_not_in_hand() -> None:
    """A bot ``choose_card`` returning a foreign :class:`Card` is a bug —
    driver surfaces :class:`IllegalBotDecisionError`, not a silent commit or
    a downgrade to pass."""
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    # Steal a card from Arien's hand — definitely not in Wasp's hand.
    other = state.teams[TeamColor.BLUE].heroes[0].hand[0]
    bot = _StubAgent(card_for={wasp.id: other})
    with pytest.raises(IllegalBotDecisionError) as excinfo:
        inspect_next_decision(state, {wasp.id: bot}, last_result=None)
    assert excinfo.value.hero_id == wasp.id
    assert "not in hand" in excinfo.value.reason


def test_planning_raises_when_bot_declines_with_nonempty_hand() -> None:
    """The engine disallows passing with a non-empty hand. If a bot returns
    ``None`` while it still has cards (and has no open Emmitt window), the
    driver raises :class:`IllegalBotDecisionError` — ``None`` is reserved
    for "no mapped owner / no current decision"."""
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    bot = _StubAgent(card_for={wasp.id: None})
    with pytest.raises(IllegalBotDecisionError) as excinfo:
        inspect_next_decision(state, {wasp.id: bot}, last_result=None)
    assert excinfo.value.hero_id == wasp.id
    assert "None" in excinfo.value.reason


def test_illegal_decision_error_is_valueerror_subclass() -> None:
    """Callers already catching :class:`ValueError` (from
    :func:`plan_from_card_choice`) keep working."""
    assert issubclass(IllegalBotDecisionError, ValueError)


def test_emmitt_finish_is_legal_none_and_does_not_raise() -> None:
    """Regression: the illegal-None guard must not fire for Emmitt's open
    two-card window. ``choose_card`` returning ``None`` after the first
    commit is the FINISH signal, not a bug."""
    state = _emmitt_state()
    session = GameSession(state)
    emmitt = state.teams[TeamColor.RED].heroes[0]
    session.commit_card(HeroID(emmitt.id), emmitt.hand[0])
    bot = _StubAgent(card_for={emmitt.id: None})
    decision = inspect_next_decision(state, {emmitt.id: bot}, last_result=None)
    assert decision is not None
    assert decision.planning is not None
    assert decision.planning.kind is PlanningKind.FINISH


# --------------------------------------------------------------------------- #
# BotDecision invariants (defensive — cheap to test, easy to regress).
# --------------------------------------------------------------------------- #


def test_bot_decision_planning_requires_a_plan() -> None:
    with pytest.raises(ValueError):
        BotDecision(kind=DecisionKind.PLANNING, hero_id=HeroID("hero_wasp"))


def test_bot_decision_input_requires_a_request() -> None:
    with pytest.raises(ValueError):
        BotDecision(
            kind=DecisionKind.INPUT, hero_id=HeroID("hero_wasp"), selection="X"
        )


def test_bot_decision_planning_rejects_stray_request_fields() -> None:
    with pytest.raises(ValueError):
        BotDecision(
            kind=DecisionKind.PLANNING,
            hero_id=HeroID("hero_wasp"),
            planning=PlanningDecision.pass_(),
            selection="oops",
        )


# --------------------------------------------------------------------------- #
# Non-bot phases: no pending request + non-planning phase.
# --------------------------------------------------------------------------- #


def test_returns_none_on_game_over() -> None:
    state = _new_game(["Wasp"], ["Arien"])
    state.phase = GamePhase.GAME_OVER
    bot = _StubAgent()
    assert inspect_next_decision(state, {"hero_wasp": bot}, last_result=None) is None


def test_returns_none_when_no_pending_request_outside_planning() -> None:
    state = _new_game(["Wasp"], ["Arien"])
    state.phase = GamePhase.RESOLUTION
    bot = _StubAgent()
    # Neutral ACTION_COMPLETE result → no pending → and RESOLUTION → nothing.
    assert (
        inspect_next_decision(state, {"hero_wasp": bot}, _non_input_result(state))
        is None
    )


# --------------------------------------------------------------------------- #
# End-to-end: harness uses the driver semantics.
# --------------------------------------------------------------------------- #


def test_run_game_completes_bot_vs_bot_smoke() -> None:
    """The refactored :func:`run_game` still finishes a game — this proves the
    driver's per-decision semantics compose into a full turn loop."""
    red = ["Wasp", "Xargatha"]
    blue = ["Arien", "Brogan"]
    agent = RandomAgent(seed=3)
    hero_ids = ["hero_wasp", "hero_xargatha", "hero_arien", "hero_brogan"]
    agents = {hid: agent for hid in hero_ids}
    result = run_game(red, blue, agents, seed=99)
    assert result.reason == "game_over"
    assert result.rounds >= 1


def test_run_game_rejects_missing_agent_coverage() -> None:
    """The bot-only harness must fail fast when any hero lacks an agent."""
    agent = RandomAgent(seed=0)
    # Only three of the four heroes are mapped.
    agents = {"hero_wasp": agent, "hero_xargatha": agent, "hero_arien": agent}
    with pytest.raises(ValueError, match="hero_brogan"):
        run_game(["Wasp", "Xargatha"], ["Arien", "Brogan"], agents, seed=1)


def test_run_game_completes_with_emmitt_present() -> None:
    """Emmitt at level 1 (default): still terminates. This checks the harness
    is not deadlocked by Emmitt's roster."""
    red = ["Emmitt"]
    blue = ["Wasp"]
    agent = RandomAgent(seed=11)
    agents = {"hero_emmitt": agent, "hero_wasp": agent}
    result = run_game(red, blue, agents, seed=17)
    assert result.reason == "game_over"


def test_driver_loop_completes_with_emmitt_level_8() -> None:
    """Emmitt at level 8: exercises the two-card FINISH branch inside a real
    game loop.

    We drive the game manually (mirroring :func:`run_game`'s inner loop)
    because we need to mutate ``emmitt.level`` after ``GameSetup``.
    Terminates → the FINISH path composes correctly.
    """
    state = _new_game(["Emmitt"], ["Wasp"])
    emmitt = state.teams[TeamColor.RED].heroes[0]
    emmitt.level = 8

    agent = RandomAgent(seed=5)
    agents = {emmitt.id: agent, "hero_wasp": agent}

    session = GameSession(state)
    last_result: SessionResult | None = None
    max_steps = 20_000
    for _ in range(max_steps):
        if state.phase == GamePhase.GAME_OVER:
            break
        decision = inspect_next_decision(state, agents, last_result)
        if decision is None:
            if state.phase == GamePhase.PLANNING:
                pytest.fail("driver returned None during PLANNING with all bots mapped")
            last_result = session.advance()
            if last_result.result_type is SessionResultType.GAME_OVER:
                break
            continue
        last_result = apply_decision(session, decision)
        if last_result.result_type is SessionResultType.GAME_OVER:
            break

    assert state.phase == GamePhase.GAME_OVER


# --------------------------------------------------------------------------- #
# Owned-hero-ids kwarg: uniform protocol contract.
# --------------------------------------------------------------------------- #


def test_driver_calls_real_agent_with_explicit_owned_hero_ids() -> None:
    """The Agent protocol requires ``owned_hero_ids``. The driver must pass
    it on every ``choose_input`` call — no signature probing, no legacy
    shim. We use a real :class:`RandomAgent` (a stock agent following the
    owned_hero_ids protocol) to prove the wiring is uniform end-to-end.
    """
    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    agent = RandomAgent(seed=0)

    seen: dict[str, Any] = {}
    original = agent.choose_input

    def spy(
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
    ) -> Any:
        seen["owned_hero_ids"] = owned_hero_ids
        return original(state, request, owned_hero_ids=owned_hero_ids)

    agent.choose_input = spy  # type: ignore[method-assign]

    req = _select_option_request(wasp.id, ["A", "B"])
    result = _input_needed_result(state, req)
    decision = inspect_next_decision(state, {wasp.id: agent}, result)
    assert decision is not None
    # Real agent's kwarg was populated by the driver — never omitted.
    assert seen["owned_hero_ids"] == frozenset({wasp.id})


def test_driver_raises_typeerror_if_agent_drops_owned_hero_ids_kwarg() -> None:
    """A concrete :class:`Agent` implementation that violates the protocol
    (does not accept ``owned_hero_ids``) must fail loudly, not silently.
    The driver no longer papers over protocol violations with a signature
    probe.
    """

    class _BrokenAgent:
        def choose_card(self, state: GameState, hero: Hero) -> Card | None:
            return hero.hand[0] if hero.hand else None

        def choose_input(self, state: GameState, request: InputRequest) -> Any:
            return "A"

    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    req = _select_option_request(wasp.id, ["A", "B"])
    result = _input_needed_result(state, req)
    broken: Any = _BrokenAgent()  # deliberately violates the Agent protocol
    with pytest.raises(TypeError):
        inspect_next_decision(state, {wasp.id: broken}, result)


def test_random_agent_bot_vs_bot_smoke_uses_owned_hero_ids_uniformly() -> None:
    """End-to-end: a full bot-vs-bot game using real :class:`RandomAgent`s
    completes. Because :func:`run_game` funnels every input decision
    through the driver, a full game exercises the uniform
    ``owned_hero_ids`` dispatch on every INPUT_NEEDED tick without any
    fallback path in play. If the protocol wiring were broken, the smoke
    would crash with :class:`TypeError` long before this assertion.
    """
    red = ["Wasp"]
    blue = ["Arien"]
    agent = RandomAgent(seed=101)
    agents = {"hero_wasp": agent, "hero_arien": agent}
    result = run_game(red, blue, agents, seed=13)
    assert result.reason == "game_over"


# --------------------------------------------------------------------------- #
# Trajectory recording: SKIP legality.
# --------------------------------------------------------------------------- #


def test_record_decision_includes_skip_in_legal_keys_when_can_skip() -> None:
    """Regression: when the engine advertises ``can_skip`` on an input
    request, the recorded ``legal_keys`` must include the ``"SKIP"``
    sentinel — otherwise a chosen SKIP would be recorded as an
    out-of-set answer and downstream policy learning would treat it as
    illegal.

    Tested directly against the harness's private ``_record_decision`` so
    the invariant does not depend on which cards ``RandomAgent`` happens
    to draw in a full game.
    """
    from automata.runtime.harness import _record_decision
    from automata.runtime.trajectory import InMemoryRecorder

    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]

    # A skippable request with two options; the bot chose to SKIP.
    request = InputRequest(
        request_type=InputRequestType.SELECT_OPTION,
        player_id=wasp.id,
        options=[InputOption(id="A", text="A"), InputOption(id="B", text="B")],
        can_skip=True,
    )
    decision = BotDecision(
        kind=DecisionKind.INPUT,
        hero_id=HeroID(wasp.id),
        request=request,
        selection="SKIP",
    )
    recorder = InMemoryRecorder()
    _record_decision(recorder, state, decision)

    assert len(recorder.decisions) == 1
    row = recorder.decisions[0]
    assert row["decision_kind"] == "INPUT"
    assert row["chosen_key"] == "SKIP"
    # The recorded legal set MUST contain SKIP so trajectory consumers can
    # verify the chosen value is a member of the enumerated legal keys.
    assert "SKIP" in row["legal_keys"], (
        f"chosen SKIP was not in legal_keys={row['legal_keys']!r}"
    )
    # Original option values are still present.
    assert "A" in row["legal_keys"]
    assert "B" in row["legal_keys"]


def test_record_decision_omits_skip_when_can_skip_is_false() -> None:
    """Complementary invariant: SKIP is only recorded as legal when the
    engine explicitly advertised ``can_skip=True``. We do not inject SKIP
    into trajectories where it wouldn't be a legal answer."""
    from automata.runtime.harness import _record_decision
    from automata.runtime.trajectory import InMemoryRecorder

    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    request = InputRequest(
        request_type=InputRequestType.SELECT_OPTION,
        player_id=wasp.id,
        options=[InputOption(id="A", text="A")],
        can_skip=False,
    )
    decision = BotDecision(
        kind=DecisionKind.INPUT,
        hero_id=HeroID(wasp.id),
        request=request,
        selection="A",
    )
    recorder = InMemoryRecorder()
    _record_decision(recorder, state, decision)
    assert "SKIP" not in recorder.decisions[0]["legal_keys"]


def test_record_decision_does_not_duplicate_skip_option() -> None:
    """Defensive: if an option's raw value happens to be ``"SKIP"`` the
    driver must not append a duplicate — legal_keys stays a set-like list."""
    from automata.runtime.harness import _record_decision
    from automata.runtime.trajectory import InMemoryRecorder

    state = _new_game(["Wasp"], ["Arien"])
    wasp = state.teams[TeamColor.RED].heroes[0]
    request = InputRequest(
        request_type=InputRequestType.SELECT_OPTION,
        player_id=wasp.id,
        options=[InputOption(id="SKIP", text="Skip")],
        can_skip=True,
    )
    decision = BotDecision(
        kind=DecisionKind.INPUT,
        hero_id=HeroID(wasp.id),
        request=request,
        selection="SKIP",
    )
    recorder = InMemoryRecorder()
    _record_decision(recorder, state, decision)
    legal = recorder.decisions[0]["legal_keys"]
    assert legal.count("SKIP") == 1


# --------------------------------------------------------------------------- #
# inspect_next_owner + driver-side selection validation.                      #
# --------------------------------------------------------------------------- #


def test_inspect_next_owner_agrees_with_inspect_next_decision_planning() -> None:
    """The owner helper must pick the same hero
    :func:`inspect_next_decision` would answer next during planning."""
    from automata.runtime.driver import inspect_next_owner

    state = _new_game(["Wasp"], ["Arien"])
    agents = {"hero_wasp": RandomAgent(0), "hero_arien": RandomAgent(0)}
    owner = inspect_next_owner(state, agents, None)
    decision = inspect_next_decision(state, agents, None)
    assert owner is not None
    assert decision is not None
    assert str(decision.hero_id) == owner


def test_inspect_next_owner_skips_committed_planners() -> None:
    """A hero already committed (in ``pending_inputs``) must not be
    considered as the next owner — the driver moves on to the next
    uncommitted mapped bot."""
    from automata.runtime.driver import inspect_next_owner

    state = _new_game(["Wasp"], ["Arien"])
    # Simulate hero_wasp already committed by adding to pending_inputs.
    # ``None`` is a legitimate placeholder here (the field is
    # ``Card | None`` and a committed-but-facedown hero can appear
    # with a ``None`` entry in some engine paths).
    state.pending_inputs[HeroID("hero_wasp")] = None
    agents = {"hero_wasp": RandomAgent(0), "hero_arien": RandomAgent(0)}
    owner = inspect_next_owner(state, agents, None)
    assert owner == "hero_arien"


def test_inspect_next_owner_returns_none_when_no_mapped_bot() -> None:
    """A game with no bot-mapped heroes must return ``None`` — no bot
    owes work, so the coordinator has no bounded path to take."""
    from automata.runtime.driver import inspect_next_owner

    state = _new_game(["Wasp"], ["Arien"])
    assert inspect_next_owner(state, {}, None) is None


def test_inspect_next_owner_input_uses_eligibility_order() -> None:
    """For an INPUT request the owner is the first eligible mapped hero,
    matching :func:`inspect_next_decision`'s ordering. We prove this
    with a team-scoped request and two bot-mapped heroes on RED."""
    from automata.runtime.driver import inspect_next_owner

    state = _new_game(["Wasp", "Xargatha"], ["Arien"])
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="team:RED",
        options=[InputOption(id="hero_arien", text="A")],
    )
    result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=req,
        current_phase=GamePhase.RESOLUTION,
    )
    # Only hero_xargatha is bot-mapped even though hero_wasp is first in
    # roster order — the owner helper must return hero_xargatha because
    # hero_wasp is a human.
    owner = inspect_next_owner(state, {"hero_xargatha": RandomAgent(0)}, result)
    assert owner == "hero_xargatha"


def test_inspect_next_owner_input_returns_none_for_human_scoped() -> None:
    """When the pending input is addressed to a hero the coordinator has
    no bot for, the owner must be ``None``."""
    from automata.runtime.driver import inspect_next_owner

    state = _new_game(["Wasp"], ["Arien"])
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_wasp",  # human
        options=[InputOption(id="hero_arien", text="A")],
    )
    result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=req,
        current_phase=GamePhase.RESOLUTION,
    )
    owner = inspect_next_owner(state, {"hero_arien": RandomAgent(0)}, result)
    assert owner is None


def test_legal_selection_values_for_request_covers_all_shapes() -> None:
    """The helper unwraps every option shape (id, hex, numeric, raw
    metadata) into the same JSON-comparable values the engine's response
    validator expects."""
    from automata.runtime.driver import legal_selection_values_for_request
    from goa2.domain.hex import Hex

    request = InputRequest(
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[
            InputOption(id="42", text="42"),  # numeric → int
            InputOption.from_value(Hex(q=1, r=-1, s=0)),  # hex → dict
            InputOption(id="hero_arien", text="Arien"),  # plain id
        ],
        can_skip=True,
    )
    legal = legal_selection_values_for_request(request)
    assert 42 in legal
    assert {"q": 1, "r": -1, "s": 0} in legal
    assert "hero_arien" in legal
    assert "SKIP" in legal


def test_driver_rejects_illegal_input_selection_planning_path_regressed() -> None:
    """Regression: the planning-path IllegalBotDecisionError semantics
    must still fire (e.g. out-of-hand card) alongside the new INPUT
    validation. Both raise the same exception type so the coordinator's
    fallback handler works identically."""
    state = _new_game(["Wasp"], ["Arien"])

    class _BadPlanner:
        def choose_card(self, state, hero):
            # Fabricate a Card that isn't in the hand.
            from copy import deepcopy
            fake = deepcopy(hero.hand[0])
            fake.id = "not-in-hand-id"
            return fake

        def choose_input(self, state, request, *, owned_hero_ids=None):
            return None

    with pytest.raises(IllegalBotDecisionError):
        inspect_next_decision(state, {"hero_wasp": _BadPlanner()}, None)


def test_driver_rejects_illegal_input_selection() -> None:
    """A bot's INPUT selection that is not among the request's legal raw
    values must raise :class:`IllegalBotDecisionError` at the driver
    boundary — no engine mutation ever runs on the illegal choice."""
    state = _new_game(["Wasp"], ["Arien"])

    class _BadInput:
        def choose_card(self, state, hero):
            return None

        def choose_input(self, state, request, *, owned_hero_ids=None):
            return "not-a-real-option-id"

    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_wasp",
        options=[InputOption(id="hero_arien", text="A")],
    )
    result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=req,
        current_phase=GamePhase.RESOLUTION,
    )
    with pytest.raises(IllegalBotDecisionError):
        inspect_next_decision(state, {"hero_wasp": _BadInput()}, result)


def test_driver_rejects_skip_when_can_skip_false() -> None:
    """``"SKIP"`` is only legal when ``request.can_skip`` is true."""
    state = _new_game(["Wasp"], ["Arien"])

    class _Skipper:
        def choose_card(self, state, hero):
            return None

        def choose_input(self, state, request, *, owned_hero_ids=None):
            return "SKIP"

    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_wasp",
        options=[InputOption(id="hero_arien", text="A")],
        can_skip=False,
    )
    result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=req,
        current_phase=GamePhase.RESOLUTION,
    )
    with pytest.raises(IllegalBotDecisionError):
        inspect_next_decision(state, {"hero_wasp": _Skipper()}, result)


def test_driver_accepts_legal_input_selection() -> None:
    """A legal INPUT selection round-trips through the driver into a
    :class:`BotDecision` without raising."""
    state = _new_game(["Wasp"], ["Arien"])

    class _GoodInput:
        def choose_card(self, state, hero):
            return None

        def choose_input(self, state, request, *, owned_hero_ids=None):
            return "hero_arien"

    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_wasp",
        options=[InputOption(id="hero_arien", text="A")],
    )
    result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=req,
        current_phase=GamePhase.RESOLUTION,
    )
    decision = inspect_next_decision(state, {"hero_wasp": _GoodInput()}, result)
    assert decision is not None
    assert decision.selection == "hero_arien"


def test_driver_upgrade_phase_selection_bypasses_option_validation() -> None:
    """UPGRADE_PHASE selections are ``{hero_id, card_id}`` dicts validated
    by the engine — not by ``request.options``. The driver must not
    reject them via option-based validation."""

    state = _new_game(["Wasp"], ["Arien"])
    upgrade_ctx = {
        "players": {
            "hero_wasp": {
                "remaining": 1,
                "options": [{"pair": ("basic_a", "basic_b")}],
            }
        }
    }
    req = InputRequest(
        request_type=InputRequestType.UPGRADE_PHASE,
        player_id="simultaneous",
        options=[],
        can_skip=False,
        context=upgrade_ctx,
    )
    result = SessionResult(
        result_type=SessionResultType.INPUT_NEEDED,
        input_request=req,
        current_phase=GamePhase.RESOLUTION,
    )

    class _UpgradeAgent:
        def choose_card(self, state, hero):
            return None

        def choose_input(self, state, request, *, owned_hero_ids=None):
            # A well-shaped UPGRADE selection.
            return {"hero_id": "hero_wasp", "card_id": "basic_a"}

    # Must NOT raise — even though ``{"hero_id": ..., "card_id": ...}``
    # is not a value that ``legal_selection_values_for_request`` would
    # produce from ``request.options``, UPGRADE_PHASE bypasses that
    # validator.
    decision = inspect_next_decision(state, {"hero_wasp": _UpgradeAgent()}, result)
    assert decision is not None
    assert decision.selection == {"hero_id": "hero_wasp", "card_id": "basic_a"}
