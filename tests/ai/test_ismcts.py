"""Tests for the ISMCTS search agent (cut B: fixed opponent model).

Kept deliberately cheap (tiny iteration budgets, capped games). We assert the
search is a well-formed, deterministic, legal-move agent that drives the engine
forward without stalling — not that it is strong (strength tuning is a separate,
much slower eval run).

This file also carries the ISMCTS-side conversion coverage: the
agent-facing ``choose_input`` must ultimately submit selections in the raw form
produced by ``goa2.domain.input.selection_value``. See ``test_heuristic.py`` for
the shared PlanningDecision / conversion coverage.
"""

from __future__ import annotations

from typing import Any

import pytest

from automata.agents.base import Agent
from automata.agents.heuristic_agent import HeuristicAgent
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP, run_game
from automata.search import ISMCTSAgent, SearchConfig
from automata.search.ismcts import (
    Decision,
    RootMismatchError,
    RootTarget,
    _input_raw_map,
    _Simulator,
    search,
)
from goa2.domain.hex import Hex
from goa2.domain.input import (
    InputOption,
    InputRequest,
    InputRequestType,
    selection_value,
)
from goa2.domain.models import TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.engine.setup import GameSetup

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]


def _tiny_cfg(seed: int = 0) -> SearchConfig:
    return SearchConfig(iterations=2, cutoff_rounds=1, seed=seed)


def _agents(agent: Agent, opp: Agent) -> dict[str, Agent]:
    return {
        "hero_wasp": agent,
        "hero_xargatha": agent,
        "hero_arien": opp,
        "hero_brogan": opp,
    }


def test_search_returns_legal_card() -> None:
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=2)
    agent = ISMCTSAgent(_tiny_cfg())
    hero = state.teams[next(iter(state.teams))].heroes[0]
    card = agent.choose_card(state, hero)
    if hero.hand:
        assert card is not None
        assert card.id in {c.id for c in hero.hand}
    else:
        assert card is None


def test_ismcts_progresses_without_stalling() -> None:
    register_all_effects()
    agents = _agents(ISMCTSAgent(_tiny_cfg()), HeuristicAgent(1))
    # Capped run: the fix for the UPGRADE_PHASE loop means rounds must advance.
    r = run_game(RED, BLUE, agents, seed=3, max_steps=300)
    assert r.rounds >= 3  # would be stuck at <=2 if the engine were looping


def test_ismcts_is_deterministic() -> None:
    register_all_effects()

    def run() -> tuple[str | None, int, int]:
        agents = _agents(ISMCTSAgent(_tiny_cfg(seed=7)), HeuristicAgent(1))
        r = run_game(RED, BLUE, agents, seed=5, max_steps=300)
        return (r.winner, r.rounds, r.steps)

    assert run() == run()


def test_ismcts_sustains_progress_over_long_horizon() -> None:
    register_all_effects()
    # One ISMCTS hero vs an otherwise-heuristic table. Over a long-ish capped
    # horizon the game must keep advancing (many rounds) and, if it ends, end
    # cleanly via game_over — proving no mid-game input loop. Natural
    # time-to-finish under weak search play is an eval concern, not a unit test.
    opp = HeuristicAgent(1)
    agents: dict[str, Agent] = {
        "hero_wasp": ISMCTSAgent(_tiny_cfg()),
        "hero_xargatha": opp,
        "hero_arien": opp,
        "hero_brogan": opp,
    }
    r = run_game(RED, BLUE, agents, seed=3, max_steps=1200)
    assert r.reason in ("game_over", "max_steps")
    assert r.rounds >= 8


# --------------------------------------------------------------------------- #
# ISMCTS routes resolution outputs through selection_value.
#
# Earlier iterations of these tests exercised
# ``ISMCTSAgent.choose_input`` on a synthetic request over a fresh planning
# state to verify the returned selection had the right type. That shape is no
# longer valid under the current strict-root contract — the search validates
# that its determinized simulator surfaces exactly the requested root, which
# a request that never entered the engine cannot satisfy. The invariant is
# still covered at the raw-map seam in
# ``test_input_raw_map_uses_domain_selection_value`` below (``_input_raw_map``
# is the same code path an agent uses to translate an option key back to a
# raw selection value).
# --------------------------------------------------------------------------- #


def _unit_request(unit_ids: list[str]) -> InputRequest:
    return InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="team:RED",
        options=[InputOption(id=uid, text=uid) for uid in unit_ids],
    )


def test_input_raw_map_uses_domain_selection_value() -> None:
    # The ISMCTS branchable-request path materializes legal actions through
    # `_input_raw_map`, whose raw values must be identical to those produced
    # by `goa2.domain.input.selection_value` — this is the invariant the
    # agent boundary preserves. We check across all three shapes (numeric,
    # hex, id) plus the SKIP sentinel.
    hex_dict = {"q": 3, "r": -3, "s": 0}
    req = InputRequest(
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_test",
        options=[
            InputOption(id="7", text="7"),  # numeric -> int
            InputOption.from_value(hex_dict),  # hex -> {q,r,s} dict
            InputOption(id="hero_arien", text="Arien"),  # plain id -> str
            InputOption(
                id="raw_hex", text="raw hex",
                metadata={"raw": Hex(q=1, r=-1, s=0)},  # raw Hex -> dict
            ),
        ],
        can_skip=True,
    )
    raw_map = _input_raw_map(req)
    # Every option's value round-trips to the same JSON-safe raw the domain
    # would emit — the search's node keys stay consistent with the engine
    # submission format.
    for opt in req.options:
        expected = selection_value(opt)
        assert expected in raw_map.values(), (
            f"{opt.id!r} mapped incorrectly: raw_map={raw_map!r}"
        )
    assert raw_map["SKIP"] == "SKIP"


# --------------------------------------------------------------------------- #
# Anchor search/simulator advancement to explicit requested hero.
#
# Cover the failure modes:
#   1. bot with an uncommitted human teammate must not have the teammate become
#      the search's root decision;
#   2. multiple bots on one team — each bot's search anchors only to its own
#      hero, not any uncommitted teammate bot's hero;
#   3. team-addressed input is only "ours" when the configured bot's own hero
#      is on that team;
#   4. unchanged single-team self-play still completes (integration guard).
#
# These probe `_Simulator` directly with a tracking default policy so we can
# assert *which* hero/request the simulator delegated vs stopped on — that is
# the precise anchoring contract, without depending on tree-visit ordering.
# --------------------------------------------------------------------------- #


class _TrackingPolicy:
    """Default policy stand-in that logs which heroes/requests it handled.

    Wraps a real ``HeuristicAgent`` so the engine still receives legal answers.
    Test assertions look at ``handled_card_hero_ids`` /
    ``handled_input_player_ids`` to check that the simulator delegated the
    right decisions to the default policy (i.e. did NOT anchor on them as
    "ours").
    """

    def __init__(self) -> None:
        self._inner = HeuristicAgent(0)
        self.handled_card_hero_ids: list[str] = []
        self.handled_input_player_ids: list[str] = []

    def choose_card(self, state: GameState, hero: Hero) -> Card | None:
        self.handled_card_hero_ids.append(hero.id)
        return self._inner.choose_card(state, hero)

    def choose_input(
        self,
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
    ) -> Any:
        self.handled_input_player_ids.append(request.player_id)
        return self._inner.choose_input(state, request)


def _fresh_planning_state() -> GameState:
    register_all_effects()
    return GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=2)


def _red_heroes(state: GameState) -> list[Hero]:
    return list(state.teams[TeamColor.RED].heroes)


def test_simulator_anchors_to_owned_hero_not_uncommitted_teammate() -> None:
    """Bot + human teammate: search root must be the bot's hero.

    In a fresh planning phase both RED heroes are uncommitted. If the simulator
    picked the "first uncommitted team hero" (the pre-Task-3 behavior) it could
    return the teammate as the root Decision, which is exactly what the plan
    forbids — the human teammate must not become the root. With the owned-hero
    anchor, the simulator delegates the teammate's card to the default policy
    and stops on the bot's own hero.

    We anchor to the *second* RED hero in iteration order so that a naive
    "first uncommitted teammate" pick would return the wrong hero — this makes
    the ordering assumption explicit and forces the delegation path.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    teammate, bot_hero = reds[0], reds[1]
    policy = _TrackingPolicy()

    sim = _Simulator(
        state,
        TeamColor.RED,
        policy,
        owned_hero_ids=frozenset({bot_hero.id}),
    )
    decision = sim.advance()

    assert decision.kind == "CARD"
    assert decision.hero is not None
    assert decision.hero.id == bot_hero.id
    # The teammate was played by the default policy, never returned as a root.
    assert teammate.id in policy.handled_card_hero_ids
    assert bot_hero.id not in policy.handled_card_hero_ids


def test_simulator_stops_only_on_configured_bots_hero_when_multiple_bots() -> None:
    """Multiple bots on one team: each bot's search anchors to its own hero.

    Two bots share team RED. Bot A's search must not treat bot B's hero as its
    root (they are separate ISMCTS instances that will each search from their
    own perspective).

    We check the ordering-sensitive direction (anchor to the second-in-iteration
    hero) to prove the simulator actually skipped past the earlier teammate via
    the default policy. The opposite direction (anchor to the first) is covered
    implicitly: both directions must stop on the owned hero.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_a_hero, bot_b_hero = reds[0], reds[1]

    # Bot A anchors to hero A only — stops immediately (A comes first).
    policy_a = _TrackingPolicy()
    sim_a = _Simulator(
        state.model_copy(deep=True),
        TeamColor.RED,
        policy_a,
        owned_hero_ids=frozenset({bot_a_hero.id}),
    )
    dec_a = sim_a.advance()
    assert dec_a.kind == "CARD" and dec_a.hero is not None
    assert dec_a.hero.id == bot_a_hero.id
    # Bot A did NOT branch on Bot B's decision (never surfaced as CARD root).
    assert bot_a_hero.id not in policy_a.handled_card_hero_ids

    # Bot B anchors to hero B only — must skip past hero A via the policy.
    policy_b = _TrackingPolicy()
    sim_b = _Simulator(
        state.model_copy(deep=True),
        TeamColor.RED,
        policy_b,
        owned_hero_ids=frozenset({bot_b_hero.id}),
    )
    dec_b = sim_b.advance()
    assert dec_b.kind == "CARD" and dec_b.hero is not None
    assert dec_b.hero.id == bot_b_hero.id
    assert bot_a_hero.id in policy_b.handled_card_hero_ids
    assert bot_b_hero.id not in policy_b.handled_card_hero_ids


def test_simulator_owned_hero_ids_can_cover_multiple_heroes() -> None:
    """All heroes owned (classic self-play): simulator still stops on some
    owned hero, no default-policy calls for those heroes at the root.

    Guards the "preserve all-AI self-play" invariant: when both RED heroes are
    owned, the simulator returns a CARD decision for one of them and does not
    delegate any owned hero to the default policy before stopping.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    owned = frozenset(h.id for h in reds)
    policy = _TrackingPolicy()

    sim = _Simulator(state, TeamColor.RED, policy, owned_hero_ids=owned)
    decision = sim.advance()

    assert decision.kind == "CARD"
    assert decision.hero is not None
    assert decision.hero.id in owned
    # No owned hero was silently played by the default policy before stopping.
    for hid in owned:
        assert hid not in policy.handled_card_hero_ids


def test_search_root_ignores_uncommitted_teammate() -> None:
    """ISMCTSAgent.choose_card asked for hero X must produce a card from X's
    hand — never from a still-uncommitted teammate's hand.

    The pre-Task-3 bug: if the simulator picks the "first uncommitted teammate"
    as the search root, ``root_legal`` (which is X's hand) will not match the
    tree's actual root actions and the returned card can silently be from the
    wrong hand. This test asserts the returned card belongs to the requested
    hero even when a teammate is uncommitted.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero, teammate = reds[0], reds[1]
    # Sanity: both are uncommitted at the fresh planning phase.
    assert bot_hero.id not in state.pending_inputs
    assert teammate.id not in state.pending_inputs

    agent = ISMCTSAgent(_tiny_cfg())
    card = agent.choose_card(state, bot_hero)
    if bot_hero.hand:
        assert card is not None
        assert card.id in {c.id for c in bot_hero.hand}
        # And explicitly NOT a card that only teammate holds.
        teammate_only = {c.id for c in teammate.hand} - {c.id for c in bot_hero.hand}
        assert card.id not in teammate_only


def test_choose_input_raises_when_team_addressed_and_bot_ineligible() -> None:
    """A team-addressed input where the bot is not on that team must fail
    closed at the public boundary — not silently delegate to the default
    policy or return an arbitrary selection.

    The contract: ``ISMCTSAgent.choose_input`` is a public boundary
    that must never answer for a team the configured bot does not control.
    Callers (the server bot coordinator) rely on this to catch mis-routed
    decisions early; the driver decides eligibility before delegation, so a
    wrong-team call is a caller bug and should be surfaced.
    """
    state = _fresh_planning_state()
    agent = ISMCTSAgent(_tiny_cfg())

    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="team:RED",
        options=[
            InputOption(id="hero_arien", text="Arien"),
            InputOption(id="hero_brogan", text="Brogan"),
        ],
    )
    with pytest.raises(ValueError):
        agent.choose_input(
            state, req, owned_hero_ids=frozenset({"hero_arien"})
        )


def test_choose_input_raises_when_hero_scoped_and_bot_not_owner() -> None:
    """Hero-scoped input where the addressed hero is NOT in owned_hero_ids
    must fail closed. Same rationale as team-addressed: the boundary refuses
    to answer for a hero it does not control.
    """
    state = _fresh_planning_state()
    agent = ISMCTSAgent(_tiny_cfg())
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_arien",  # addressed to Arien
        options=[InputOption(id="hero_wasp", text="Wasp")],
    )
    with pytest.raises(ValueError):
        # Bot owns Brogan, not the addressed hero.
        agent.choose_input(
            state, req, owned_hero_ids=frozenset({"hero_brogan"})
        )


def test_choose_input_raises_on_empty_ownership() -> None:
    """An empty ``owned_hero_ids`` is never a legal call: with no owned
    heroes there is nothing to search for. Must raise, not fall through.
    """
    state = _fresh_planning_state()
    agent = ISMCTSAgent(_tiny_cfg())
    req = _unit_request(["hero_arien", "hero_brogan"])
    with pytest.raises(ValueError):
        agent.choose_input(state, req, owned_hero_ids=frozenset())


def test_choose_input_accepts_team_request_when_bot_eligible() -> None:
    """When the configured bot IS on the addressed team, choose_input passes
    the pre-search eligibility check and reaches the search.

    We can't run a real search on a synthetically constructed request (the
    determinized simulator won't surface a request the engine never
    scheduled), so we patch ``search`` to a canned stub and assert the call
    reaches it — i.e. no ValueError was raised for eligibility.
    """
    import automata.search.agent as agent_mod
    from automata.search.ismcts import SearchResult
    from automata.search.node import Node

    called: dict[str, Any] = {}

    def _stub_search(*args: Any, **kwargs: Any) -> SearchResult:
        called["ok"] = True
        legal = args[3] if len(args) > 3 else []
        legal_list = list(legal)
        return SearchResult(root=Node(), best_key=legal_list[0] if legal_list else None)

    state = _fresh_planning_state()
    unit_ids = ["hero_wasp", "hero_xargatha"]
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="team:RED",
        options=[InputOption(id=uid, text=uid) for uid in unit_ids],
    )
    state.input_stack.append(req)
    agent = ISMCTSAgent(_tiny_cfg())
    real_search = agent_mod.search
    agent_mod.search = _stub_search  # type: ignore[assignment]
    try:
        result = agent.choose_input(
            state, req, owned_hero_ids=frozenset({"hero_wasp"})
        )
    finally:
        agent_mod.search = real_search  # type: ignore[assignment]

    assert called.get("ok") is True
    assert result in unit_ids


def test_choose_card_passes_hero_as_owner_anchor() -> None:
    """The agent's choose_card must pass its hero as the root anchor.

    Regression guard: without an explicit owner, ``_Simulator`` would fall
    back to its old "first uncommitted team hero" behavior. We verify by
    setting up a state where the requested hero is the *second* uncommitted
    RED hero (via iteration order) and confirming the returned card comes from
    that specific hero's hand.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    # Pick the second RED hero deliberately so team-iteration order can't
    # accidentally satisfy the assertion.
    target = reds[1]
    if not target.hand:
        return  # no ranking possible; still safe under new anchoring

    agent = ISMCTSAgent(_tiny_cfg())
    card = agent.choose_card(state, target)
    assert card is not None
    assert card.id in {c.id for c in target.hand}


def test_all_ai_self_play_still_completes() -> None:
    """Integration guard: existing bot-vs-bot self-play (both RED heroes
    controlled by an ISMCTS instance) must still progress and terminate cleanly
    even after root anchoring changes. Cheap capped run — strength unchecked.
    """
    register_all_effects()
    agents = _agents(ISMCTSAgent(_tiny_cfg()), HeuristicAgent(1))
    r = run_game(RED, BLUE, agents, seed=3, max_steps=300)
    assert r.rounds >= 3
    assert r.reason in ("game_over", "max_steps")


# --------------------------------------------------------------------------- #
# RootTarget / strict root validation.
#
# The contract requires an explicit typed root anchor threaded into search and
# strict validation at search entry: the simulator MUST surface exactly the
# requested root (matching kind, hero id for CARD, or exact InputRequest.id
# for INPUT). Mismatch/stale/missing surfaced root raises RootMismatchError;
# the search MUST NOT return an arbitrary zero-visit action.
# --------------------------------------------------------------------------- #


def _default_policy() -> HeuristicAgent:
    return HeuristicAgent(0)


def test_root_target_card_requires_hero_in_owned() -> None:
    """A CARD root target must name a specific hero that is also in the owned
    set — the two must be consistent. Rejecting inconsistent targets at
    construction stops the bug from leaking into the search."""
    with pytest.raises(ValueError):
        RootTarget.card(hero_id="hero_wasp", owned_hero_ids=frozenset({"hero_xargatha"}))


def test_root_target_rejects_empty_ownership() -> None:
    """Owned set must be non-empty regardless of kind."""
    with pytest.raises(ValueError):
        RootTarget.card(hero_id="hero_wasp", owned_hero_ids=frozenset())
    with pytest.raises(ValueError):
        RootTarget.input(
            request_id="r1", player_id="team:RED", owned_hero_ids=frozenset()
        )


def test_simulator_advance_to_root_matches_card_target() -> None:
    """``advance_to_root`` surfaces the exact target hero at planning."""
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[1]
    target = RootTarget.card(
        hero_id=bot_hero.id, owned_hero_ids=frozenset({bot_hero.id})
    )
    sim = _Simulator(
        state,
        TeamColor.RED,
        _default_policy(),
        owned_hero_ids=target.owned_hero_ids,
    )
    decision = sim.advance_to_root(target)
    assert decision.kind == "CARD"
    assert decision.hero is not None
    assert decision.hero.id == bot_hero.id


def test_simulator_advance_to_root_raises_on_kind_mismatch() -> None:
    """A CARD target on a state that would surface an INPUT (or vice versa)
    is a mismatch — the caller misidentified the current decision. Raise."""
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    # Fake INPUT target on a state whose next decision is CARD.
    fake_req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id=bot_hero.id,
        options=[InputOption(id="x", text="x")],
    )
    target = RootTarget.input(
        request_id=fake_req.id,
        player_id=bot_hero.id,
        owned_hero_ids=frozenset({bot_hero.id}),
    )
    sim = _Simulator(
        state,
        TeamColor.RED,
        _default_policy(),
        owned_hero_ids=target.owned_hero_ids,
    )
    with pytest.raises(RootMismatchError):
        sim.advance_to_root(target)


def test_simulator_advance_to_root_raises_on_stale_request_id() -> None:
    """An INPUT target whose ``request_id`` doesn't match the surfaced request
    is stale — the state has moved on since the caller captured the request.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    stale_target = RootTarget.input(
        request_id="stale-not-real",
        player_id=bot_hero.id,
        owned_hero_ids=frozenset({bot_hero.id}),
    )
    sim = _Simulator(
        state,
        TeamColor.RED,
        _default_policy(),
        owned_hero_ids=stale_target.owned_hero_ids,
    )
    with pytest.raises(RootMismatchError):
        sim.advance_to_root(stale_target)


def test_search_raises_on_root_mismatch_never_returns_zero_visit_action() -> None:
    """``search`` must never fall back to an arbitrary root_legal[0] when the
    simulator can't surface the requested root. It must raise so the caller
    knows the decision is invalid."""
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    # Fabricate a CARD target for a hero that isn't in the game.
    bogus = RootTarget.card(
        hero_id="hero_ghost", owned_hero_ids=frozenset({"hero_ghost"})
    )
    with pytest.raises(RootMismatchError):
        search(
            state,
            TeamColor.RED,
            "CARD",
            [c.id for c in bot_hero.hand],
            _default_policy(),
            _tiny_cfg(),
            root_target=bogus,
        )


def test_search_raises_on_empty_root_legal() -> None:
    """Search MUST reject an empty ``root_legal`` — there's nothing to pick."""
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    target = RootTarget.card(
        hero_id=bot_hero.id, owned_hero_ids=frozenset({bot_hero.id})
    )
    with pytest.raises(ValueError):
        search(
            state,
            TeamColor.RED,
            "CARD",
            [],  # empty legal set
            _default_policy(),
            _tiny_cfg(),
            root_target=target,
        )


def test_search_uses_root_decision_kind_and_raises_on_kind_mismatch() -> None:
    """``root_decision_kind`` is authoritative. If it disagrees with the
    root target, search fails closed — it does not silently defer to the
    target's kind."""
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    target = RootTarget.card(
        hero_id=bot_hero.id, owned_hero_ids=frozenset({bot_hero.id})
    )
    with pytest.raises(ValueError):
        search(
            state,
            TeamColor.RED,
            "INPUT",  # disagrees with target.kind == "CARD"
            [c.id for c in bot_hero.hand],
            _default_policy(),
            _tiny_cfg(),
            root_target=target,
        )


def test_choose_input_raises_when_request_is_stale() -> None:
    """A request that no longer matches the state's active pending input is
    stale — the caller captured it, some other player acted, and now the
    engine is elsewhere. The public boundary refuses stale decisions.

    We construct a stale scenario by giving the state a real pending input on
    the stack and passing a *different* request id — this is the concrete
    signal that a coordinator would detect.
    """
    state = _fresh_planning_state()
    # Simulate an active pending input on the state's stack (this is what the
    # engine records when it pauses for input). The caller then hands us a
    # different request id — that's the stale case.
    active = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_wasp",
        options=[InputOption(id="hero_arien", text="Arien")],
    )
    state.input_stack.append(active)

    other = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_wasp",
        options=[InputOption(id="hero_arien", text="Arien")],
    )
    # Sanity: distinct ids.
    assert other.id != active.id
    agent = ISMCTSAgent(_tiny_cfg())
    with pytest.raises(ValueError):
        agent.choose_input(
            state, other, owned_hero_ids=frozenset({"hero_wasp"})
        )


def test_choose_input_exact_request_id_is_threaded_into_root_target() -> None:
    """The request the caller hands to ``choose_input`` must be the exact
    request used as the search root (``RootTarget.request_id`` matches).

    We assert observationally: replace ``search`` with a capturing stub that
    records the ``root_target`` it received and returns a canned result. This
    isolates the AGENT-side threading from whether the underlying search can
    actually surface the synthetic request from a determinized clone (it
    can't, because the engine never scheduled it).
    """
    import automata.search.agent as agent_mod
    from automata.search.ismcts import SearchResult
    from automata.search.node import Node

    captured: dict[str, Any] = {}

    def _capturing_search(*args: Any, **kwargs: Any) -> SearchResult:
        captured["root_target"] = kwargs.get("root_target")
        captured["root_decision_kind"] = args[2] if len(args) > 2 else None
        captured["root_legal"] = args[3] if len(args) > 3 else None
        # Return a canned SearchResult picking the first legal key so the
        # agent can complete its call — we only care about what was passed
        # into search, not what came out.
        legal = list(captured["root_legal"] or [])
        return SearchResult(root=Node(), best_key=legal[0] if legal else None)

    state = _fresh_planning_state()
    # Line up state.input_stack with the request so the freshness check passes.
    unit_ids = ["hero_wasp", "hero_xargatha"]
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="team:RED",
        options=[InputOption(id=uid, text=uid) for uid in unit_ids],
    )
    state.input_stack.append(req)
    agent = ISMCTSAgent(_tiny_cfg())
    real_search = agent_mod.search
    agent_mod.search = _capturing_search  # type: ignore[assignment]
    try:
        result = agent.choose_input(
            state, req, owned_hero_ids=frozenset({"hero_wasp"})
        )
    finally:
        agent_mod.search = real_search  # type: ignore[assignment]

    root_target = captured["root_target"]
    assert root_target is not None
    assert root_target.kind == "INPUT"
    assert root_target.request_id == req.id
    assert root_target.player_id == req.player_id
    assert root_target.owned_hero_ids == frozenset({"hero_wasp"})
    # And ``root_decision_kind`` was explicitly threaded, not left at some
    # default — the review requires the kind be used.
    assert captured["root_decision_kind"] == "INPUT"
    # The agent's public result routes through _input_raw_map for the picked
    # key, so it must resolve to one of the legal raw values.
    assert result in unit_ids


def test_agent_protocol_accepts_owned_hero_ids_kwarg() -> None:
    """Random/HeuristicAgent must accept the ``owned_hero_ids`` kwarg (and
    ignore it) so the runtime driver can pass it uniformly to every agent.
    """
    from automata.agents.random_agent import RandomAgent

    state = _fresh_planning_state()
    req = _unit_request(["hero_wasp", "hero_xargatha"])

    rnd = RandomAgent(0)
    heur = HeuristicAgent(0)
    # Both must accept the kwarg without crashing and return a legal value.
    r1 = rnd.choose_input(state, req, owned_hero_ids=frozenset({"hero_wasp"}))
    r2 = heur.choose_input(state, req, owned_hero_ids=frozenset({"hero_wasp"}))
    assert r1 in {"hero_wasp", "hero_xargatha"}
    assert r2 in {"hero_wasp", "hero_xargatha"}
    # And the default (None / omitted) still works — existing callsites.
    assert rnd.choose_input(state, req) in {"hero_wasp", "hero_xargatha"}
    assert heur.choose_input(state, req) in {"hero_wasp", "hero_xargatha"}


# --------------------------------------------------------------------------- #
# Singleton root validation, tightened choose_input ordering, RootTarget
# player_id check, and agent-level conversion coverage via search stubbing.
# --------------------------------------------------------------------------- #


def test_root_target_matches_rejects_wrong_player_id() -> None:
    """``RootTarget.matches`` for INPUT must validate both request_id AND
    player_id. A decision whose request has the target's request_id but a
    different player_id is NOT a match — the addressing scope must agree
    (hero-scoped vs team-scoped), else routing has drifted."""
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="team:RED",
        options=[InputOption(id="x", text="x")],
    )
    target = RootTarget.input(
        request_id=req.id,
        player_id="hero_wasp",  # target expects hero-scoped
        owned_hero_ids=frozenset({"hero_wasp"}),
    )
    decision = Decision("INPUT", request=req)  # request is team-scoped
    assert not target.matches(decision)

    # And it accepts a matching player_id.
    matched_target = RootTarget.input(
        request_id=req.id,
        player_id="team:RED",
        owned_hero_ids=frozenset({"hero_wasp"}),
    )
    assert matched_target.matches(decision)


def test_search_validates_root_target_before_singleton_early_return_card() -> None:
    """A CARD singleton root_legal (hero has one card) must still validate the
    root target against the state. A bogus hero anchor MUST raise even when
    only one legal action exists — the pre-review fix skipped validation on
    the singleton path, letting stale/mismatched calls silently return a
    ``best_key``. Uses a clone-based ``advance_to_root`` — no full search cost.
    """
    state = _fresh_planning_state()
    # Bogus target: hero doesn't exist in state.
    target = RootTarget.card(
        hero_id="hero_ghost", owned_hero_ids=frozenset({"hero_ghost"})
    )
    with pytest.raises(RootMismatchError):
        search(
            state,
            TeamColor.RED,
            "CARD",
            ["only_one_card_id"],  # singleton
            HeuristicAgent(0),
            _tiny_cfg(),
            root_target=target,
        )


def test_search_validates_root_target_before_singleton_early_return_input() -> None:
    """A singleton INPUT root_legal must also validate — stale request_id on a
    singleton must raise, not silently return the one legal key."""
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    # Bogus request id: the state has no such active request.
    target = RootTarget.input(
        request_id="stale-not-in-state",
        player_id=bot_hero.id,
        owned_hero_ids=frozenset({bot_hero.id}),
    )
    with pytest.raises(RootMismatchError):
        search(
            state,
            TeamColor.RED,
            "INPUT",
            ["SKIP"],  # singleton
            HeuristicAgent(0),
            _tiny_cfg(),
            root_target=target,
        )


def test_search_singleton_early_return_after_validation_passes() -> None:
    """When the root target is valid AND ``root_legal`` matches the canonical
    surfaced legal set, the singleton early return still fires — no wasted
    iterations. We construct a real singleton by trimming the bot hero's hand
    to a single card, then pass that card's id as ``root_legal``.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    if len(bot_hero.hand) < 1:
        return  # nothing to trim; setup precondition
    # Trim to one card so the surfaced decision's canonical legal_keys is
    # exactly [only_card.id] — matches ``root_legal`` singleton.
    only_card = bot_hero.hand[0]
    bot_hero.hand[:] = [only_card]
    target = RootTarget.card(
        hero_id=bot_hero.id, owned_hero_ids=frozenset({bot_hero.id})
    )
    result = search(
        state,
        TeamColor.RED,
        "CARD",
        [only_card.id],
        HeuristicAgent(0),
        _tiny_cfg(),
        root_target=target,
    )
    assert result.best_key == only_card.id
    # No iterations ran — the root node has no explored children.
    assert not result.root.children


# --- Caller root_legal vs canonical legal_keys --- #


def test_search_singleton_root_legal_disagrees_with_canonical_raises() -> None:
    """A singleton caller ``root_legal`` that names a key which is NOT in the
    surfaced decision's canonical ``legal_keys`` must fail closed. The old
    behavior returned the fabricated key unchanged; the reviewed contract
    forbids that — a caller's mistaken legal set would silently corrupt the
    action applied downstream.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    if not bot_hero.hand:
        return
    only_card = bot_hero.hand[0]
    bot_hero.hand[:] = [only_card]
    target = RootTarget.card(
        hero_id=bot_hero.id, owned_hero_ids=frozenset({bot_hero.id})
    )
    with pytest.raises((RootMismatchError, ValueError)):
        search(
            state,
            TeamColor.RED,
            "CARD",
            ["fabricated_not_in_hand"],  # NOT the real singleton key
            HeuristicAgent(0),
            _tiny_cfg(),
            root_target=target,
        )


def test_search_multi_key_root_legal_missing_canonical_key_raises() -> None:
    """A multi-key caller ``root_legal`` that omits a key present in the
    canonical ``legal_keys(decision)`` must fail closed. Pruning legal
    actions silently at the search boundary could hide better moves from
    the tree; the reviewed contract requires an exact set match."""
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    if len(bot_hero.hand) < 2:
        return
    hand_ids = [c.id for c in bot_hero.hand]
    target = RootTarget.card(
        hero_id=bot_hero.id, owned_hero_ids=frozenset({bot_hero.id})
    )
    # Drop one legal key from the caller's set.
    truncated = hand_ids[:-1]
    with pytest.raises((RootMismatchError, ValueError)):
        search(
            state,
            TeamColor.RED,
            "CARD",
            truncated,
            HeuristicAgent(0),
            _tiny_cfg(),
            root_target=target,
        )


def test_search_multi_key_root_legal_extra_key_raises() -> None:
    """A multi-key caller ``root_legal`` that contains a key NOT in the
    canonical set must fail closed. Extra keys would waste iterations on
    illegal branches and could return a ``best_key`` the engine will reject.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    if len(bot_hero.hand) < 2:
        return
    hand_ids = [c.id for c in bot_hero.hand]
    target = RootTarget.card(
        hero_id=bot_hero.id, owned_hero_ids=frozenset({bot_hero.id})
    )
    with pytest.raises((RootMismatchError, ValueError)):
        search(
            state,
            TeamColor.RED,
            "CARD",
            [*hand_ids, "fabricated_extra"],
            HeuristicAgent(0),
            _tiny_cfg(),
            root_target=target,
        )


def test_search_multi_key_root_legal_duplicate_key_raises() -> None:
    """Multiplicity matters. A caller ``root_legal`` with a duplicated key
    is a bug (each legal action must appear once); fail closed rather than
    silently biasing progressive widening / ranking toward the duplicate.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    if len(bot_hero.hand) < 2:
        return
    hand_ids = [c.id for c in bot_hero.hand]
    target = RootTarget.card(
        hero_id=bot_hero.id, owned_hero_ids=frozenset({bot_hero.id})
    )
    with pytest.raises((RootMismatchError, ValueError)):
        search(
            state,
            TeamColor.RED,
            "CARD",
            [*hand_ids, hand_ids[0]],  # one key appears twice
            HeuristicAgent(0),
            _tiny_cfg(),
            root_target=target,
        )


def test_search_multi_key_root_legal_reordered_preserves_caller_order() -> None:
    """Same set, different order is ALLOWED. The caller's order must be
    preserved for downstream policy tie-breaking (progressive widening reveals
    in the order given by the policy prior, but where two children are tied
    the caller's order provides a deterministic secondary key).

    We assert by running a real (tiny) search and verifying the returned
    ``best_key`` is a member of the caller's set — proving the reordered
    input was accepted, not rejected.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    if len(bot_hero.hand) < 2:
        return
    hand_ids = [c.id for c in bot_hero.hand]
    # Reverse the canonical order.
    reordered = list(reversed(hand_ids))
    assert reordered != hand_ids  # reordering is real
    target = RootTarget.card(
        hero_id=bot_hero.id, owned_hero_ids=frozenset({bot_hero.id})
    )
    result = search(
        state,
        TeamColor.RED,
        "CARD",
        reordered,
        HeuristicAgent(0),
        _tiny_cfg(),
        root_target=target,
    )
    assert result.best_key in reordered


def test_search_singleton_input_root_legal_disagrees_raises() -> None:
    """The singleton mismatch case for INPUT roots: search called with a
    singleton ``["SKIP"]`` on an active request whose canonical legal set is
    ``["hero_arien"]`` (no SKIP allowed) must raise.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id=bot_hero.id,
        options=[InputOption(id="hero_arien", text="Arien")],
        can_skip=False,  # SKIP is NOT a legal key
    )
    state.input_stack.append(req)
    target = RootTarget.input(
        request_id=req.id,
        player_id=bot_hero.id,
        owned_hero_ids=frozenset({bot_hero.id}),
    )
    with pytest.raises((RootMismatchError, ValueError)):
        search(
            state,
            TeamColor.RED,
            "INPUT",
            ["SKIP"],  # NOT a legal key for this request
            HeuristicAgent(0),
            _tiny_cfg(),
            root_target=target,
        )


def test_choose_input_checks_freshness_before_non_branchable_fallback() -> None:
    """A stale request that happens to be non-branchable (e.g. a fake
    ``UPGRADE_PHASE`` with a stale id) must raise, not silently delegate to
    the default policy. The freshness check runs BEFORE the fallback so a
    coordinator can't hand the bot a stale broadcast request and get a
    quietly-computed answer.
    """
    state = _fresh_planning_state()
    # State has an active pending input.
    active = InputRequest(
        request_type=InputRequestType.UPGRADE_PHASE,
        player_id="simultaneous",
    )
    state.input_stack.append(active)

    # Caller hands us a DIFFERENT non-branchable request id.
    stale = InputRequest(
        request_type=InputRequestType.UPGRADE_PHASE,
        player_id="simultaneous",
    )
    assert stale.id != active.id
    agent = ISMCTSAgent(_tiny_cfg())
    with pytest.raises(ValueError):
        agent.choose_input(
            state, stale, owned_hero_ids=frozenset({"hero_wasp"})
        )


def test_choose_input_non_branchable_fallback_only_for_simultaneous() -> None:
    """The default-policy fallback is narrowed to intentional global request
    shapes (``player_id == 'simultaneous'``). A non-branchable request
    *addressed* to a hero (which would be an engine bug or a malformed call)
    must NOT silently delegate — raise instead so the bug surfaces.
    """
    state = _fresh_planning_state()
    reds = _red_heroes(state)
    bot_hero = reds[0]
    # Non-branchable (no options) but hero-scoped — not a global broadcast.
    weird = InputRequest(
        request_type=InputRequestType.SELECT_OPTION,
        player_id=bot_hero.id,
        options=[],
        can_skip=False,
    )
    agent = ISMCTSAgent(_tiny_cfg())
    with pytest.raises(ValueError):
        agent.choose_input(
            state, weird, owned_hero_ids=frozenset({bot_hero.id})
        )


def test_choose_input_non_branchable_simultaneous_still_fallback() -> None:
    """Genuine simultaneous/global requests (e.g. UPGRADE_PHASE) are the
    intended fallback path — they must still delegate to the default policy
    after ownership passes. Otherwise the harness would break on upgrade
    phase turns."""
    state = _fresh_planning_state()
    request = InputRequest(
        request_type=InputRequestType.UPGRADE_PHASE,
        player_id="simultaneous",
    )
    sentinel = object()

    class _StubPolicy:
        def choose_card(self, state: GameState, hero: Hero) -> Card | None:
            return None

        def choose_input(
            self,
            state: GameState,
            request: InputRequest,
            *,
            owned_hero_ids: frozenset[str] | None = None,
        ) -> Any:
            return sentinel

    agent = ISMCTSAgent(_tiny_cfg(), default_policy=_StubPolicy())
    result = agent.choose_input(
        state, request, owned_hero_ids=frozenset({"hero_wasp"})
    )
    assert result is sentinel


# --- Agent-boundary conversion via search stubbing ------------------------- #


def _stub_ismcts_search_return_key(
    monkeypatch: pytest.MonkeyPatch, chosen_key: Any
) -> None:
    """Stub the ``search()`` used by ISMCTSAgent to return ``chosen_key`` as
    ``best_key`` without exercising the determinized simulator.

    This lets synthetic-request tests (a fresh planning state) still exercise
    the agent's post-search conversion — the raw_map
    key → raw value translation via ``selection_value`` — without needing a
    real engine state that actually has the request pending.
    """
    import automata.search.agent as agent_mod
    from automata.search.ismcts import SearchResult
    from automata.search.node import Node

    def _stub(*args: Any, **kwargs: Any) -> SearchResult:
        return SearchResult(root=Node(), best_key=chosen_key)

    monkeypatch.setattr(agent_mod, "search", _stub)


def _prepare_input_state(
    state: GameState, req: InputRequest
) -> None:
    """Line up ``state.input_stack`` with ``req`` so the freshness check in
    ``ISMCTSAgent.choose_input`` passes. This is the concrete state shape a
    real coordinator would present."""
    state.input_stack.append(req)


def test_ismcts_choose_input_returns_int_for_select_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When search picks a NUMBER option's key, choose_input returns an int
    (numeric selections are ints, not strings)."""
    state = _fresh_planning_state()
    req = InputRequest(
        request_type=InputRequestType.SELECT_NUMBER,
        player_id="team:RED",
        options=[InputOption(id=str(v), text=str(v)) for v in (1, 2, 3)],
    )
    _prepare_input_state(state, req)
    # Numeric selections are converted to ints by ``selection_value``, and
    # ``action_key`` on an int is the int itself — so the raw_map key is 2.
    _stub_ismcts_search_return_key(monkeypatch, 2)
    agent = ISMCTSAgent(_tiny_cfg())
    result = agent.choose_input(
        state, req, owned_hero_ids=frozenset({"hero_wasp"})
    )
    assert isinstance(result, int)
    assert result == 2


def test_ismcts_choose_input_returns_hex_dict_for_select_hex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When search picks a HEX option's key, choose_input returns a
    ``{q, r, s}`` dict (hex selections are always dicts)."""
    hexes = [{"q": 0, "r": 0, "s": 0}, {"q": 1, "r": -1, "s": 0}]
    req = InputRequest(
        request_type=InputRequestType.SELECT_HEX,
        player_id="team:RED",
        options=[InputOption.from_value(h) for h in hexes],
    )
    state = _fresh_planning_state()
    _prepare_input_state(state, req)
    # Search picks the first hex option (its action_key is based on the raw).
    from automata.search.node import action_key

    picked_hex = hexes[0]
    picked_key = action_key(picked_hex)
    _stub_ismcts_search_return_key(monkeypatch, picked_key)
    agent = ISMCTSAgent(_tiny_cfg())
    result = agent.choose_input(
        state, req, owned_hero_ids=frozenset({"hero_wasp"})
    )
    assert isinstance(result, dict)
    assert result == picked_hex


def test_ismcts_choose_input_returns_string_id_for_select_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When search picks a UNIT option's key, choose_input returns the raw
    unit id string (ids are strings, not ints)."""
    unit_ids = ["hero_arien", "hero_brogan"]
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="team:RED",
        options=[InputOption(id=uid, text=uid) for uid in unit_ids],
    )
    state = _fresh_planning_state()
    _prepare_input_state(state, req)
    _stub_ismcts_search_return_key(monkeypatch, "hero_brogan")
    agent = ISMCTSAgent(_tiny_cfg())
    result = agent.choose_input(
        state, req, owned_hero_ids=frozenset({"hero_wasp"})
    )
    assert isinstance(result, str)
    assert result == "hero_brogan"


def test_ismcts_choose_input_returns_skip_when_search_picks_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``can_skip=True`` and search picks the ``SKIP`` raw-map key,
    choose_input returns the literal string ``"SKIP"`` (the skip sentinel is
    the string, never null).
    """
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="team:RED",
        options=[InputOption(id="hero_arien", text="Arien")],
        can_skip=True,
    )
    state = _fresh_planning_state()
    _prepare_input_state(state, req)
    _stub_ismcts_search_return_key(monkeypatch, "SKIP")
    agent = ISMCTSAgent(_tiny_cfg())
    result = agent.choose_input(
        state, req, owned_hero_ids=frozenset({"hero_wasp"})
    )
    assert result == "SKIP"


# --------------------------------------------------------------------------- #
# Bounded ISMCTS for live use.
#
# The runtime bounds (semaphore, queue timeout, search timeout, fallback)
# live in ``goa2.server.bots`` and are exercised in tests/server/test_server_bots.py.
# The AI-side pieces we cover here are:
#
#   1. Production bounds on SearchSettings (upper/lower validation).
#   2. Deterministic behavior of ``ISMCTSAgent`` under a fixed seed + iteration
#      budget (the property the coordinator relies on for reproducible bots).
#   3. The public SearchSettings ↔ SearchConfig mapping used by agent_for_spec.
# --------------------------------------------------------------------------- #


class TestSearchSettingsValidation:
    """Validation limits on the public/internal SearchSettings model.

    The coordinator enforces these bounds at the request boundary; a
    restored save cannot bypass them either because the same model is used
    for persistence.
    """

    def test_iterations_rejects_zero(self) -> None:
        from goa2.server.bot_models import SearchSettings

        with pytest.raises(ValueError):
            SearchSettings(iterations=0)

    def test_iterations_rejects_negative(self) -> None:
        from goa2.server.bot_models import SearchSettings

        with pytest.raises(ValueError):
            SearchSettings(iterations=-1)

    def test_iterations_rejects_over_max(self) -> None:
        from automata.search.config import PROD_MAX_ITERATIONS
        from goa2.server.bot_models import SearchSettings

        with pytest.raises(ValueError):
            SearchSettings(iterations=PROD_MAX_ITERATIONS + 1)

    def test_iterations_accepts_min_and_max(self) -> None:
        from automata.search.config import (
            PROD_MAX_ITERATIONS,
            PROD_MIN_ITERATIONS,
        )
        from goa2.server.bot_models import SearchSettings

        # Both boundaries valid.
        SearchSettings(iterations=PROD_MIN_ITERATIONS)
        SearchSettings(iterations=PROD_MAX_ITERATIONS)

    def test_decision_timeout_rejects_over_max(self) -> None:
        from automata.search.config import PROD_MAX_DECISION_TIMEOUT_SECONDS
        from goa2.server.bot_models import SearchSettings

        with pytest.raises(ValueError):
            SearchSettings(
                decision_timeout_seconds=PROD_MAX_DECISION_TIMEOUT_SECONDS + 1.0
            )

    def test_decision_timeout_rejects_under_min(self) -> None:
        from automata.search.config import PROD_MIN_DECISION_TIMEOUT_SECONDS
        from goa2.server.bot_models import SearchSettings

        with pytest.raises(ValueError):
            SearchSettings(
                decision_timeout_seconds=PROD_MIN_DECISION_TIMEOUT_SECONDS / 2.0
            )

    def test_defaults_are_within_bounds(self) -> None:
        from automata.search.config import (
            PROD_DEFAULT_DECISION_TIMEOUT_SECONDS,
            PROD_DEFAULT_ITERATIONS,
            PROD_MAX_DECISION_TIMEOUT_SECONDS,
            PROD_MAX_ITERATIONS,
            PROD_MIN_DECISION_TIMEOUT_SECONDS,
            PROD_MIN_ITERATIONS,
        )
        from goa2.server.bot_models import SearchSettings

        s = SearchSettings()
        assert PROD_MIN_ITERATIONS <= s.iterations <= PROD_MAX_ITERATIONS
        assert (
            PROD_MIN_DECISION_TIMEOUT_SECONDS
            <= s.decision_timeout_seconds
            <= PROD_MAX_DECISION_TIMEOUT_SECONDS
        )
        # Defaults exactly match the module-level constants (kept in sync).
        assert s.iterations == PROD_DEFAULT_ITERATIONS
        assert s.decision_timeout_seconds == PROD_DEFAULT_DECISION_TIMEOUT_SECONDS


def test_agent_for_spec_ismcts_returns_bounded_agent() -> None:
    """:func:`agent_for_spec` for ``kind='ismcts'`` returns a real bounded
    :class:`ISMCTSAgent`. The
    agent's :class:`SearchConfig` must inherit the ``iterations`` from the
    supplied :class:`SearchSettings`."""
    from automata.search.agent import ISMCTSAgent as ISMCTSAgentType
    from goa2.server.bot_models import BotSpec, SearchSettings
    from goa2.server.bots import agent_for_spec

    spec = BotSpec(
        kind="ismcts",
        search=SearchSettings(iterations=42, decision_timeout_seconds=1.5),
    )
    agent = agent_for_spec(spec, seed=7)
    assert isinstance(agent, ISMCTSAgentType)
    # The agent must be seeded off the provided seed and configured with
    # the requested iteration budget.
    assert agent._cfg.iterations == 42
    assert agent._cfg.seed == 7


def test_agent_for_spec_ismcts_defaults_when_no_search_settings() -> None:
    """A BotSpec(kind='ismcts') with no explicit ``search`` still yields a
    bounded agent — the coordinator applies the SearchSettings defaults."""
    from automata.search.agent import ISMCTSAgent as ISMCTSAgentType
    from automata.search.config import PROD_DEFAULT_ITERATIONS
    from goa2.server.bot_models import BotSpec
    from goa2.server.bots import agent_for_spec

    agent = agent_for_spec(BotSpec(kind="ismcts"), seed=3)
    assert isinstance(agent, ISMCTSAgentType)
    assert agent._cfg.iterations == PROD_DEFAULT_ITERATIONS


def test_ismcts_agent_deterministic_under_fixed_seed_and_budget() -> None:
    """Fixed seed + fixed iteration budget → identical decisions on
    identical states. This is the property the coordinator's reproducibility
    guarantees depend on (a fallback triggered on one restart must not
    trigger on the next if the search itself is deterministic)."""
    register_all_effects()
    state = GameSetup.create_game(DEFAULT_MAP, RED, BLUE, game_type="QUICK", seed=11)
    hero = state.teams[next(iter(state.teams))].heroes[0]
    if not hero.hand:
        return

    def _pick() -> str | None:
        a = ISMCTSAgent(SearchConfig(iterations=4, cutoff_rounds=1, seed=99))
        card = a.choose_card(state, hero)
        return None if card is None else card.id

    assert _pick() == _pick()
