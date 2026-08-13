"""Behavioral contract for versioned, richer learned-value features."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any

import pytest

from automata.evaluation import features
from automata.evaluation.learned_value import LearnedValue
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP
from goa2.domain.models import CardState, TeamColor
from goa2.domain.types import BoardEntityID
from goa2.engine.hero_pieces import piece_id
from goa2.engine.setup import GameSetup

BASE_NAMES = (
    "life_diff",
    "push_diff",
    "minion_diff",
    "level_diff",
    "alive_diff",
    "gold_diff",
)
RICH_NAMES = (
    "own_life",
    "enemy_life",
    "own_push",
    "enemy_push",
    "own_battle_minions",
    "enemy_battle_minions",
    "own_level_total",
    "enemy_level_total",
    "own_alive_heroes",
    "enemy_alive_heroes",
    "own_gold_total",
    "enemy_gold_total",
    "round_number",
    "wave_remaining_mean",
    "own_hand_cards",
    "enemy_hand_cards",
    "own_discard_cards",
    "enemy_discard_cards",
    "own_played_cards",
    "enemy_played_cards",
    "own_battle_heroes",
    "enemy_battle_heroes",
    "own_hero_progress_mean",
    "enemy_hero_progress_mean",
)


def _state(red: list[str] | None = None, blue: list[str] | None = None):
    register_all_effects()
    return GameSetup.create_game(
        DEFAULT_MAP,
        red or ["Wasp", "Xargatha"],
        blue or ["Arien", "Brogan"],
        game_type="QUICK",
        seed=7,
    )


def _gbm(names: tuple[str, ...], schema: str | None = None) -> dict[str, Any]:
    artifact = {
        "model_version": "gbm-v1",
        "schema_version": 1,
        "red_roster": ["Wasp", "Xargatha"],
        "blue_roster": ["Arien", "Brogan"],
        "feature_names": list(names),
        "base_raw_score": 0.0,
        "learning_rate": 1.0,
        "trees": [
            {
                "root": 0,
                "nodes": [
                    {"feature": 0, "threshold": 0, "left": 1, "right": 2},
                    {"value": -1},
                    {"value": 1},
                ],
            }
        ],
    }
    if schema is not None:
        artifact["feature_schema"] = schema
    return artifact


def test_schema_registry_preserves_base_and_defines_exact_rich_schema() -> None:
    assert tuple(features.FEATURE_SCHEMAS) == ("base-v1", "rich-v1")
    assert isinstance(features.FEATURE_SCHEMAS["base-v1"], features.FeatureSchema)
    assert features.FEATURE_SCHEMAS["base-v1"].feature_names == BASE_NAMES
    assert features.FEATURE_SCHEMAS["rich-v1"].feature_names == RICH_NAMES
    assert features.FEATURE_NAMES == BASE_NAMES

    state = _state()
    assert features.feature_vector(state, TeamColor.RED) == features.feature_vector(
        state, TeamColor.RED, "base-v1"
    )
    with pytest.raises(ValueError, match=r"unknown|schema"):
        features.state_features(state, TeamColor.RED, "missing-v1")


def test_rich_features_are_finite_side_aware_and_use_deck_card_states() -> None:
    state = _state()
    red_heroes = state.teams[TeamColor.RED].heroes
    selected_card_ids: set[str] = set()
    for hero in red_heroes:
        for card in hero.deck:
            card.state = CardState.DECK
        for card, card_state in zip(
            hero.deck[:3],
            (CardState.HAND, CardState.DISCARD, CardState.RESOLVED),
            strict=True,
        ):
            card.state = card_state
            selected_card_ids.add(card.id)
    assert len({hero.id for hero in red_heroes}) == 2
    assert len(selected_card_ids) == 6
    state.round = 4
    state.wave_counters = {lane: i + 2 for i, lane in enumerate(state.wave_counters)}

    red = features.state_features(state, TeamColor.RED, "rich-v1")
    blue = features.state_features(state, TeamColor.BLUE, "rich-v1")
    assert tuple(red) == RICH_NAMES
    assert len(features.feature_vector(state, TeamColor.RED, "rich-v1")) == 24
    assert all(math.isfinite(value) for value in red.values())
    for name in RICH_NAMES:
        if name.startswith("own_"):
            assert red[name] == blue["enemy_" + name.removeprefix("own_")]
        elif name.startswith("enemy_"):
            assert red[name] == blue["own_" + name.removeprefix("enemy_")]
    assert red["round_number"] == blue["round_number"] == 4
    assert red["wave_remaining_mean"] == blue["wave_remaining_mean"]
    assert red["own_hand_cards"] == 2
    assert red["own_discard_cards"] == 2
    assert red["own_played_cards"] == 2
    assert 0.0 <= red["own_hero_progress_mean"] <= 1.0


def test_multipiece_hero_counts_owner_once_in_alive_and_battle_features() -> None:
    state = _state(["Razzle"], ["Wasp"])
    zone = state.board.zones[next(iter(state.battle_zones.values()))]
    free = [h for h in zone.hexes if not state.board.get_tile(h).is_occupied]
    pieces = [piece_id("hero_razzle", 1), piece_id("hero_razzle", 2)]
    state.place_entity(BoardEntityID(pieces[0]), free[0])
    state.place_entity(BoardEntityID(pieces[1]), free[1])

    rich = features.state_features(state, TeamColor.RED, "rich-v1")
    assert len(state.get_positions("hero_razzle")) == 2
    assert rich["own_alive_heroes"] == 1
    assert rich["own_battle_heroes"] == 1


def test_learned_value_schema_selection_validation_and_legacy_digest() -> None:
    legacy = _gbm(BASE_NAMES)
    canonical = json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    assert LearnedValue(legacy).digest == hashlib.sha256(canonical).hexdigest()
    assert LearnedValue({**legacy, "feature_schema": "base-v1"})(
        _state(), TeamColor.RED
    ) == LearnedValue(legacy)(_state(), TeamColor.RED)

    rich = _gbm(RICH_NAMES, "rich-v1")
    state = _state()
    first = features.feature_vector(state, TeamColor.RED, "rich-v1")[0]
    expected = math.tanh((1 if first >= 0 else -1) / 2)
    assert LearnedValue(rich)(state, TeamColor.RED) == pytest.approx(expected)
    wrong_names = deepcopy(rich)
    wrong_names["feature_names"] = list(BASE_NAMES)
    with pytest.raises(ValueError):
        LearnedValue(wrong_names)
    bad_index = deepcopy(rich)
    bad_index["trees"][0]["nodes"][0]["feature"] = len(RICH_NAMES)
    with pytest.raises(ValueError):
        LearnedValue(bad_index)
