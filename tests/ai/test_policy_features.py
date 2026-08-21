"""Behavioral contract for learned-policy candidate features."""

from __future__ import annotations

import importlib
import math
from types import ModuleType

import pytest

from automata.evaluation.features import state_features
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from automata.search import POLICY_FEATURE_SCHEMA_ID, policy_candidate_features
from automata.search.ismcts import Decision, legal_keys
from automata.search.node import action_key
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import TeamColor
from goa2.domain.models.enums import CardState
from goa2.domain.types import HeroID
from goa2.engine.phases import commit_card
from goa2.engine.setup import GameSetup


def _policy_module() -> ModuleType:
    # Import at test time so this RED contract still collects before the new
    # production module exists.
    return importlib.import_module("automata.search.policy_features")


def _state(*, seed: int = 7):
    register_all_effects()
    return GameSetup.create_game(
        DEFAULT_MAP,
        ["Wasp", "Xargatha"],
        ["Arien", "Brogan"],
        game_type="QUICK",
        seed=seed,
    )


def _features(state, decision: Decision, legal):
    return policy_candidate_features(state, decision, legal)


def _feature_with_suffix(values: dict[str, float], suffix: str) -> float:
    matches = [value for name, value in values.items() if name.endswith(suffix)]
    assert len(matches) == 1, f"expected one feature ending in {suffix!r}"
    return matches[0]


def test_card_and_finish_candidates_have_finite_sparse_features() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    decision = Decision("CARD", hero=hero, can_finish_planning=True)
    legal = legal_keys(decision)

    result = _features(state, decision, legal)

    assert set(result) == set(legal)
    assert None in result
    assert len(hero.hand) >= 2
    assert result[hero.hand[0].id] != result[hero.hand[1].id]
    for candidate in result.values():
        assert candidate
        assert all(isinstance(name, str) and name for name in candidate)
        assert all(type(value) is float and math.isfinite(value) for value in candidate.values())


def test_common_features_use_versioned_rich_acting_side_perspective() -> None:
    module = _policy_module()
    assert isinstance(module.POLICY_FEATURE_SCHEMA_ID, str)
    assert module.POLICY_FEATURE_SCHEMA_ID
    assert "v" in module.POLICY_FEATURE_SCHEMA_ID.lower()
    assert POLICY_FEATURE_SCHEMA_ID == module.POLICY_FEATURE_SCHEMA_ID

    state = _state()
    state.teams[TeamColor.RED].life_counters -= 2
    state.round = 4
    request = InputRequest(
        id="perspective",
        request_type=InputRequestType.SELECT_NUMBER,
        player_id="hero_wasp",
        options=[InputOption.from_value(1)],
    )
    values = _features(state, Decision("INPUT", request=request), [1])[1]
    expected = state_features(state, TeamColor.RED, "rich-v1")

    for name in ("own_life", "enemy_life", "round_number", "own_hand_cards", "enemy_hand_cards"):
        assert _feature_with_suffix(values, name) == float(expected[name])


def test_input_candidates_cover_public_action_unit_number_hex_and_skip_data() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    free_hexes = [hex_ for hex_ in state.board.tiles if not state.board.get_tile(hex_).is_occupied][:2]
    requests = [
        InputRequest(
            id="actions",
            request_type=InputRequestType.CHOOSE_ACTION,
            player_id=f"team:{TeamColor.RED.value}",
            options=[
                InputOption(id="advance", text="Advance", metadata={"action_type": "MOVE", "value": 2}),
                InputOption(id="strike", text="Strike", metadata={"action_type": "ATTACK", "value": 3}),
            ],
        ),
        InputRequest(
            id="units",
            request_type=InputRequestType.SELECT_UNIT,
            player_id=str(hero.id),
            options=[InputOption.from_value(h.id) for h in state.teams[TeamColor.RED].heroes],
        ),
        InputRequest(
            id="numbers",
            request_type=InputRequestType.SELECT_NUMBER,
            player_id=str(hero.id),
            options=[InputOption.from_value(1), InputOption.from_value(3)],
        ),
        InputRequest(
            id="hexes",
            request_type=InputRequestType.SELECT_HEX,
            player_id=str(hero.id),
            options=[InputOption.from_value(hex_) for hex_ in free_hexes],
        ),
        InputRequest(
            id="optional",
            request_type=InputRequestType.SELECT_OPTION,
            player_id=str(hero.id),
            options=[InputOption.from_value("hold")],
            can_skip=True,
        ),
    ]

    for request in requests:
        decision = Decision("INPUT", request=request)
        legal = legal_keys(decision)
        result = _features(state, decision, legal)
        assert set(result) == set(legal)
        assert all(result[key] for key in legal)
        assert all(
            type(value) is float and math.isfinite(value)
            for candidate in result.values()
            for value in candidate.values()
        )
        if len(legal) > 1:
            assert len({tuple(sorted(result[key].items())) for key in legal}) == len(legal)

    hex_keys = legal_keys(Decision("INPUT", request=requests[3]))
    assert hex_keys == [action_key({"q": h.q, "r": h.r, "s": h.s}) for h in free_hexes]
    assert "SKIP" in legal_keys(Decision("INPUT", request=requests[4]))


def test_features_are_deterministic_per_key_independent_of_legal_order() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    decision = Decision("CARD", hero=hero, can_finish_planning=True)
    legal = legal_keys(decision)

    forward = _features(state, decision, legal)
    reverse = _features(state, decision, list(reversed(legal)))

    assert forward == reverse


def test_enemy_hidden_card_identity_swap_does_not_change_features() -> None:
    first = _state(seed=11)
    second = _state(seed=11)
    for state, index in ((first, 0), (second, 1)):
        enemy = state.teams[TeamColor.BLUE].heroes[0]
        for card in enemy.hand:
            card.is_facedown = True
        commit_card(state, HeroID(enemy.id), enemy.hand[index])

    first_hero = first.teams[TeamColor.RED].heroes[0]
    second_hero = second.teams[TeamColor.RED].heroes[0]
    legal = [card.id for card in first_hero.hand]

    assert [card.id for card in first_hero.hand] == [card.id for card in second_hero.hand]
    first_commit = first.pending_inputs[HeroID("hero_arien")]
    second_commit = second.pending_inputs[HeroID("hero_arien")]
    assert first_commit is not None and second_commit is not None
    assert first_commit.id != second_commit.id
    assert _features(first, Decision("CARD", hero=first_hero), legal) == _features(
        second, Decision("CARD", hero=second_hero), legal
    )


def _assert_no_private_card_face(values: dict[str, float]) -> None:
    private_prefixes = (
        "card.tier.",
        "card.color.",
        "card.primary_action.",
        "card.secondary_action.",
        "card.effect.",
    )
    private_names = {
        "card.initiative",
        "card.primary_action_value",
        "card.range",
        "card.radius",
        "card.is_ranged",
    }
    assert not any(name.startswith(private_prefixes) for name in values)
    assert not private_names.intersection(values)
    assert not any(name.startswith("card.public_identity.") for name in values)


def test_faceup_enemy_hand_card_option_remains_hidden_from_hero_viewer() -> None:
    state = _state()
    viewer = state.teams[TeamColor.RED].heroes[0]
    enemy_card = state.teams[TeamColor.BLUE].heroes[0].hand[0]
    enemy_card.is_facedown = False
    assert enemy_card.state == CardState.HAND
    request = InputRequest(
        id="enemy-hand",
        request_type=InputRequestType.SELECT_CARD,
        player_id=str(viewer.id),
        options=[InputOption.from_value(enemy_card.id)],
    )

    values = _features(state, Decision("INPUT", request=request), [enemy_card.id])[
        enemy_card.id
    ]

    assert values["card.hidden"] == 1.0
    _assert_no_private_card_face(values)


def test_teammate_facedown_card_option_remains_hidden_from_hero_viewer() -> None:
    state = _state()
    viewer, teammate = state.teams[TeamColor.RED].heroes
    teammate_card = teammate.hand[0]
    teammate_card.is_facedown = True
    request = InputRequest(
        id="teammate-hand",
        request_type=InputRequestType.SELECT_CARD,
        player_id=str(viewer.id),
        options=[InputOption.from_value(teammate_card.id)],
    )

    values = _features(state, Decision("INPUT", request=request), [teammate_card.id])[
        teammate_card.id
    ]

    assert values["card.hidden"] == 1.0
    _assert_no_private_card_face(values)


def test_team_scoped_input_has_no_private_hand_card_viewer() -> None:
    state = _state()
    card = state.teams[TeamColor.RED].heroes[0].hand[0]
    card.is_facedown = False
    request = InputRequest(
        id="team-hand",
        request_type=InputRequestType.SELECT_CARD,
        player_id=f"team:{TeamColor.RED.value}",
        options=[InputOption.from_value(card.id)],
    )

    values = _features(state, Decision("INPUT", request=request), [card.id])[card.id]

    assert values["card.hidden"] == 1.0
    _assert_no_private_card_face(values)


def test_faceup_enemy_resolved_card_is_public_but_not_viewer_owned() -> None:
    state = _state()
    viewer = state.teams[TeamColor.RED].heroes[0]
    card = state.teams[TeamColor.BLUE].heroes[0].hand[0]
    card.state = CardState.RESOLVED
    card.is_facedown = False
    request = InputRequest(
        id="public-card",
        request_type=InputRequestType.SELECT_CARD,
        player_id=str(viewer.id),
        options=[InputOption.from_value(card.id)],
    )

    values = _features(state, Decision("INPUT", request=request), [card.id])[card.id]

    assert values[f"card.public_identity.{card.id}"] == 1.0
    assert "card.viewer_owned" not in values


def test_card_candidates_include_viewers_public_identity() -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    card = hero.hand[0]

    values = _features(state, Decision("CARD", hero=hero), [card.id])[card.id]

    assert values[f"card.public_identity.{card.id}"] == 1.0
    assert values["card.viewer_owned"] == 1.0


@pytest.mark.parametrize("kind", ["CARD", "INPUT"])
def test_unreconciled_legal_key_fails_closed(kind: str) -> None:
    state = _state()
    hero = state.teams[TeamColor.RED].heroes[0]
    if kind == "CARD":
        decision = Decision("CARD", hero=hero)
    else:
        request = InputRequest(
            id="known-options",
            request_type=InputRequestType.SELECT_OPTION,
            player_id=str(hero.id),
            options=[InputOption.from_value("known")],
        )
        decision = Decision("INPUT", request=request)

    with pytest.raises(ValueError, match=r"legal|candidate|option|reconcil"):
        _features(state, decision, ["not-a-legal-candidate"])


def test_unresolved_acting_perspective_fails_closed_but_sentinels_are_supported() -> None:
    state = _state()
    unknown_request = InputRequest(
        id="unknown-player",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_not_in_this_game",
        options=[InputOption.from_value("known")],
    )
    with pytest.raises(ValueError, match=r"perspective|player|team|hero"):
        _features(state, Decision("INPUT", request=unknown_request), ["known"])

    hero = state.teams[TeamColor.RED].heroes[0]
    finish = _features(state, Decision("CARD", hero=hero), [None])
    skip_request = InputRequest(
        id="skip-only",
        request_type=InputRequestType.SELECT_OPTION,
        player_id=str(hero.id),
        can_skip=True,
    )
    skip = _features(state, Decision("INPUT", request=skip_request), ["SKIP"])
    assert set(finish) == {None}
    assert set(skip) == {"SKIP"}
