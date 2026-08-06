"""Tests for the heuristic agent and state evaluation.

Note: we deliberately do NOT assert the heuristic beats random. Empirically the
greedy one-ply static heuristic is ~random-strength in aggregate (it wins some
games decisively and loses others) — one-ply scoring without lookahead is not
reliably better in this imperfect-information game. Its value here is (a) a
non-trivial rollout policy and (b) exercising `evaluate_state`, which MCTS needs.

This file also carries the agent-contract coverage — selection-value
conversions and the typed :class:`PlanningDecision` model — because the
heuristic agent is the primary consumer of the shared conversion path.
ISMCTS-specific coverage lives in ``test_ismcts.py``.
"""

from __future__ import annotations

import pytest

from automata.agents import PlanningDecision, PlanningKind, plan_from_card_choice
from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.random_agent import RandomAgent
from automata.evaluation.features import evaluate_state
from automata.evaluation.matchup import evaluate
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from goa2.domain.hex import Hex
from goa2.domain.input import (
    InputOption,
    InputRequest,
    InputRequestType,
    selection_value,
)
from goa2.domain.models import TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.enums import ActionType, CardColor, CardTier
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.engine.setup import GameSetup


def test_evaluate_state_symmetry_and_terminal() -> None:
    register_all_effects()
    st = GameSetup.create_game(
        DEFAULT_MAP, ["Wasp", "Xargatha"], ["Arien", "Brogan"], game_type="QUICK", seed=1
    )
    # Fresh symmetric game: neither side favoured.
    assert evaluate_state(st, TeamColor.RED) == -evaluate_state(st, TeamColor.BLUE)
    assert abs(evaluate_state(st, TeamColor.RED)) < 1.0

    # Draining an enemy's life should strictly favour us.
    st.teams[TeamColor.BLUE].life_counters -= 2
    assert evaluate_state(st, TeamColor.RED) > 0
    assert evaluate_state(st, TeamColor.BLUE) < 0


def test_heuristic_game_is_deterministic() -> None:
    red = ["Wasp", "Xargatha"]
    blue = ["Arien", "Brogan"]

    def run() -> tuple[int, int]:
        r = evaluate(
            lambda s: HeuristicAgent(s),
            lambda s: RandomAgent(s),
            red_heroes=red,
            blue_heroes=blue,
            games=4,
            base_seed=0,
        )
        return (r.a_wins, r.b_wins)

    assert run() == run()


def test_heuristic_completes_games() -> None:
    r = evaluate(
        lambda s: HeuristicAgent(s),
        lambda s: RandomAgent(s),
        red_heroes=["Wasp", "Xargatha"],
        blue_heroes=["Arien", "Brogan"],
        games=4,
        base_seed=5,
    )
    # Every game resolves to a winner (no step-cap draws on this map).
    assert r.draws == 0
    assert r.a_wins + r.b_wins == 4


# --------------------------------------------------------------------------- #
# selection_value conversions through HeuristicAgent.choose_input.
#
# The agent must submit whatever `selection_value` says the raw form is — ints
# for numeric ids, hex dicts for hex options (both the metadata["hex"] and
# metadata["raw"]=Hex paths), plain ids as strings, and "SKIP" when nothing is
# actionable. Testing through `choose_input` exercises the same code path a
# real game runs, not a private helper.
# --------------------------------------------------------------------------- #


class _StateStub:
    """Tiny GameState stand-in — the heuristic paths hit here don't touch it.

    `choose_input` only reads `state` for scorers that inspect the board; the
    request types below either don't call those scorers or the scorer paths
    tolerate a missing field. If the heuristic ever grows a dependency on
    concrete state for one of these requests, we swap this for
    ``GameSetup.create_game(...)`` at the cost of ~1s per test.
    """


def _stub_state() -> GameState:
    # Cast at the call site keeps the test compact; the heuristic never
    # dereferences `state` for the request types exercised here.
    return _StateStub()  # type: ignore[return-value]


def _select_number(values: list[int]) -> InputRequest:
    return InputRequest(
        request_type=InputRequestType.SELECT_NUMBER,
        player_id="hero_test",
        options=[InputOption(id=str(v), text=str(v)) for v in values],
    )


def _select_hex(hexes: list[dict[str, int]]) -> InputRequest:
    # Uses InputOption.from_value which stashes both metadata["hex"] and
    # metadata["raw"]=<dict>. selection_value prefers "hex" — this is the
    # dominant hex path in production.
    return InputRequest(
        request_type=InputRequestType.SELECT_HEX,
        player_id="hero_test",
        options=[InputOption.from_value(h) for h in hexes],
    )


def _select_unit(unit_ids: list[str]) -> InputRequest:
    return InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_test",
        options=[InputOption(id=uid, text=uid) for uid in unit_ids],
    )


def test_heuristic_choose_input_returns_int_for_select_number() -> None:
    agent = HeuristicAgent(seed=0)
    # Heuristic picks the largest number (SELECT_NUMBER scorer). The point of
    # this test is the *type*: it must be int, not the numeric string.
    result = agent.choose_input(_stub_state(), _select_number([1, 5, 2]))
    assert isinstance(result, int)
    assert result == 5


def test_heuristic_choose_input_returns_hex_dict_for_select_hex() -> None:
    # Use a stub state, but the hex scorer *does* inspect the board. We swap
    # to a real state for this one case so the score can be computed without
    # exception. The invariant asserted is on the returned *shape*, not which
    # hex wins.
    register_all_effects()
    st = GameSetup.create_game(
        DEFAULT_MAP, ["Wasp", "Xargatha"], ["Arien", "Brogan"], game_type="QUICK", seed=1
    )
    agent = HeuristicAgent(seed=0)
    hexes = [{"q": 0, "r": 0, "s": 0}, {"q": 1, "r": -1, "s": 0}]
    result = agent.choose_input(st, _select_hex(hexes))
    assert isinstance(result, dict)
    assert result in hexes
    # And the domain-authoritative conversion agrees: for a hex option the
    # metadata["hex"] entry is the raw form.
    assert result == selection_value(
        next(o for o in _select_hex(hexes).options if o.metadata["hex"] == result)
    )


def test_heuristic_choose_input_returns_string_id_for_select_unit() -> None:
    # SELECT_UNIT scorer needs board context, so use a real state again.
    register_all_effects()
    st = GameSetup.create_game(
        DEFAULT_MAP, ["Wasp", "Xargatha"], ["Arien", "Brogan"], game_type="QUICK", seed=1
    )
    agent = HeuristicAgent(seed=0)
    unit_ids = ["hero_arien", "hero_brogan"]
    result = agent.choose_input(st, _select_unit(unit_ids))
    assert isinstance(result, str)
    assert result in unit_ids


def test_heuristic_choose_input_returns_skip_when_empty_options_and_can_skip() -> None:
    agent = HeuristicAgent(seed=0)
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_test",
        options=[],
        can_skip=True,
    )
    assert agent.choose_input(_stub_state(), req) == "SKIP"


# --------------------------------------------------------------------------- #
# RandomAgent.choose_input routes selections through selection_value.
#
# RandomAgent is the sole reference policy for the eval matrix and one of
# the production bot kinds, so its output shape must match the engine's
# submission format exactly — not just Heuristic's. Determinism comes from
# either a single-option request (only one legal pick) or a seeded RNG on a
# multi-option request; the constructor seeds `random.Random` so the same
# seed always makes the same choice.
# --------------------------------------------------------------------------- #


def test_random_agent_choose_input_returns_int_for_select_number() -> None:
    # Single-option SELECT_NUMBER: the pick is forced, and the test is
    # asserting the *type* of the returned value (int, not str). The RNG
    # branch that would skip is masked out by ``can_skip=False`` (the default).
    agent = RandomAgent(seed=0)
    req = _select_number([7])
    result = agent.choose_input(_stub_state(), req)
    assert result == 7
    assert isinstance(result, int)


def test_random_agent_choose_input_returns_hex_dict_for_select_hex() -> None:
    # Single-option hex request: `InputOption.from_value(dict)` populates
    # metadata["hex"], and `selection_value` returns that dict verbatim.
    agent = RandomAgent(seed=0)
    hex_dict = {"q": 2, "r": -1, "s": -1}
    req = _select_hex([hex_dict])
    result = agent.choose_input(_stub_state(), req)
    assert result == hex_dict
    assert isinstance(result, dict)


def test_random_agent_choose_input_returns_string_id_for_select_unit() -> None:
    agent = RandomAgent(seed=0)
    req = _select_unit(["hero_arien"])
    result = agent.choose_input(_stub_state(), req)
    assert result == "hero_arien"
    assert isinstance(result, str)


def test_random_agent_choose_input_returns_skip_when_empty_options_and_can_skip() -> None:
    # No options + can_skip → the sole legal move is to skip.
    agent = RandomAgent(seed=0)
    req = InputRequest(
        request_type=InputRequestType.SELECT_UNIT,
        player_id="hero_test",
        options=[],
        can_skip=True,
    )
    assert agent.choose_input(_stub_state(), req) == "SKIP"


def test_random_agent_choose_input_multi_option_selection_type_is_stable() -> None:
    # Multi-option deterministic case: given a fixed seed, the same choice is
    # made every run, and — regardless of which option wins — the returned
    # value must be an int (never the numeric string id). Also asserts the
    # picked value matches `selection_value` for one of the options.
    agent = RandomAgent(seed=123)
    req = _select_number([1, 2, 3])
    result = agent.choose_input(_stub_state(), req)
    assert isinstance(result, int)
    assert result in {1, 2, 3}
    # Deterministic re-run under the same seed picks the same option.
    assert RandomAgent(seed=123).choose_input(_stub_state(), _select_number([1, 2, 3])) == result
    # And matches the domain conversion for one of the options.
    assert result in {selection_value(o) for o in req.options}


# --------------------------------------------------------------------------- #
# metadata["raw"] conversion coverage.
#
# `selection_value` has three branches:
#   1. metadata["hex"]  -> return the hex dict verbatim
#   2. metadata["raw"]  -> return the raw value; if the raw is a Hex object,
#                          convert to a JSON-safe {q,r,s} dict on the way out
#   3. bare option id   -> int if numeric, else the string id
#
# Tests above cover (1) and (3). Below covers (2) — the "raw" fallback — for
# both a Hex object and a plain raw value, since a real game routes hex
# selections through `InputOption.from_value(hex_obj)` and non-hex non-dict
# raw values through the generic bucket.
# --------------------------------------------------------------------------- #


def test_selection_value_from_hex_object_returns_dict() -> None:
    # `InputOption.from_value(Hex)` stashes the object under metadata["raw"]
    # AND its dict form under metadata["hex"]. selection_value should return
    # the dict either way — hitting the "hex" branch here.
    h = Hex(q=1, r=-2, s=1)
    opt = InputOption.from_value(h)
    result = selection_value(opt)
    assert result == {"q": 1, "r": -2, "s": 1}


def test_selection_value_from_raw_hex_object_without_hex_metadata() -> None:
    # Explicit "raw" branch: no metadata["hex"] present, raw is a Hex object.
    # `selection_value` must still produce a JSON-safe dict.
    h = Hex(q=2, r=0, s=-2)
    opt = InputOption(id="custom", text="custom", metadata={"raw": h})
    assert selection_value(opt) == {"q": 2, "r": 0, "s": -2}


def test_selection_value_from_raw_plain_value_returns_raw() -> None:
    # Non-hex raw values (e.g. a payload dict) pass through unchanged.
    raw_payload = {"hero_id": "hero_wasp", "card_id": "wasp_sting"}
    opt = InputOption(id="wasp_sting_pair", text="Wasp Sting", metadata={"raw": raw_payload})
    assert selection_value(opt) is raw_payload


# --------------------------------------------------------------------------- #
# PlanningDecision — COMMIT / FINISH / PASS are distinct and enforced.
# --------------------------------------------------------------------------- #


def _dummy_card(cid: str = "test_card") -> Card:
    return Card(
        id=cid,
        name=cid,
        tier=CardTier.I,
        color=CardColor.RED,
        initiative=5,
        primary_action=ActionType.ATTACK,
        primary_action_value=2,
        secondary_actions={},
        effect_id="",
        effect_text="",
    )


def _dummy_hero(hero_id: str = "hero_test", hand: list[Card] | None = None) -> Hero:
    return Hero(id=hero_id, name=hero_id, deck=[], hand=hand or [])


def test_planning_decision_commit_carries_card() -> None:
    card = _dummy_card()
    dec = PlanningDecision.commit(card)
    assert dec.kind is PlanningKind.COMMIT
    assert dec.card is card


def test_planning_decision_finish_and_pass_carry_no_card() -> None:
    finish = PlanningDecision.finish()
    passd = PlanningDecision.pass_()
    assert finish.kind is PlanningKind.FINISH
    assert finish.card is None
    assert passd.kind is PlanningKind.PASS
    assert passd.card is None


def test_planning_decisions_of_different_kinds_are_unequal() -> None:
    card = _dummy_card()
    commit = PlanningDecision.commit(card)
    finish = PlanningDecision.finish()
    passd = PlanningDecision.pass_()
    assert commit != finish
    assert commit != passd
    assert finish != passd


def test_planning_decision_commit_requires_card() -> None:
    with pytest.raises(ValueError):
        PlanningDecision(kind=PlanningKind.COMMIT, card=None)


def test_planning_decision_finish_rejects_card() -> None:
    with pytest.raises(ValueError):
        PlanningDecision(kind=PlanningKind.FINISH, card=_dummy_card())


def test_planning_decision_pass_rejects_card() -> None:
    with pytest.raises(ValueError):
        PlanningDecision(kind=PlanningKind.PASS, card=_dummy_card())


def test_planning_decision_rejects_unknown_string_kind() -> None:
    # Guards against a naive deserializer or a typo. `PlanningKind` is a
    # closed set of runtime kinds; anything else must raise.
    with pytest.raises(ValueError):
        PlanningDecision(kind="RESIGN")  # type: ignore[arg-type]


def test_planning_decision_accepts_string_alias_and_coerces_to_enum() -> None:
    # Historical string values are still accepted for construction (str-backed
    # Enum), and they coerce to the canonical enum member so downstream code
    # can rely on `is PlanningKind.X` identity comparisons.
    dec = PlanningDecision(kind="COMMIT", card=_dummy_card())  # type: ignore[arg-type]
    assert dec.kind is PlanningKind.COMMIT


# --------------------------------------------------------------------------- #
# plan_from_card_choice — hand membership is required for COMMIT.
# --------------------------------------------------------------------------- #


def test_plan_from_card_choice_none_maps_to_pass() -> None:
    hero = _dummy_hero(hand=[_dummy_card()])
    dec = plan_from_card_choice(hero, None)
    assert dec.kind is PlanningKind.PASS


def test_plan_from_card_choice_card_in_hand_maps_to_commit() -> None:
    card = _dummy_card()
    hero = _dummy_hero(hand=[card])
    dec = plan_from_card_choice(hero, card)
    assert dec.kind is PlanningKind.COMMIT
    assert dec.card is card


def test_plan_from_card_choice_empty_hand_with_card_raises() -> None:
    # A bug in an agent's choose_card must not be silently downgraded to PASS.
    hero = _dummy_hero(hand=[])
    with pytest.raises(ValueError, match="not in hero"):
        plan_from_card_choice(hero, _dummy_card())


def test_plan_from_card_choice_card_not_in_hand_raises() -> None:
    # Distinct object identity: a card the agent invented (or one from a
    # different hero's hand) is rejected explicitly.
    in_hand = _dummy_card("legal_card")
    foreign = _dummy_card("foreign_card")
    hero = _dummy_hero(hand=[in_hand])
    with pytest.raises(ValueError, match="not in hero"):
        plan_from_card_choice(hero, foreign)
