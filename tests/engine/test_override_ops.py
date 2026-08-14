"""Override op registry: patch ops, atomicity, multi-piece conventions."""

import pytest

from goa2.domain.hex import Hex
from goa2.engine.overrides import (
    OVERRIDE_OPS,
    OverrideRejectedError,
    apply_override_decision,
    summarize_op,
)
from goa2.engine.session import GameSession
from goa2.engine.setup import GameSetup

MAP_PATH = "src/goa2/data/maps/forgotten_island.json"


@pytest.fixture
def session() -> GameSession:
    state = GameSetup.create_game(MAP_PATH, ["Arien"], ["Wasp"], False, "QUICK", seed=42)
    return GameSession(state)


def _hex_dict(h: Hex) -> dict:
    return {"q": h.q, "r": h.r, "s": h.s}


def _free_adjacent(state, entity_id: str) -> Hex:
    pos = state.get_position(entity_id)
    for n in pos.neighbors():
        tile = state.board.tiles.get(n)
        if tile is not None and tile.occupant_id is None:
            return n
    raise AssertionError("no free adjacent hex")


def test_registry_contains_all_patch_and_unstick_ops():
    expected = {
        "move_entity",
        "remove_entity",
        "place_entity",
        "set_life_counters",
        "set_gold",
        "set_level",
        "add_marker",
        "remove_marker",
        "add_effect",
        "remove_effect",
        "move_card",
        "set_wave_counter",
        "set_tie_breaker_team",
        "skip_input",
        "abort_action",
        "end_turn",
        "force_actor",
    }
    assert expected <= set(OVERRIDE_OPS)
    for op in OVERRIDE_OPS.values():
        assert op.family in ("patch", "unstick")
        assert op.label and op.description


def test_unknown_op_rejected(session):
    with pytest.raises(OverrideRejectedError) as exc:
        apply_override_decision(session, "teleport_everything", {})
    assert exc.value.code == "unknown_op"


def test_move_entity_moves_a_hero(session):
    state = session.state
    target = _free_adjacent(state, "hero_arien")
    apply_override_decision(
        session, "move_entity", {"entity_id": "hero_arien", "hex": _hex_dict(target)}
    )
    assert session.state.get_position("hero_arien") == target
    # occupancy cache rebuilt
    assert str(session.state.board.tiles[target].occupant_id) == "hero_arien"


def test_move_entity_to_occupied_hex_rejected_and_commits_nothing(session):
    state = session.state
    arien_pos = state.get_position("hero_arien")
    wasp_pos = state.get_position("hero_wasp")
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(
            session, "move_entity", {"entity_id": "hero_arien", "hex": _hex_dict(wasp_pos)}
        )
    assert session.state.get_position("hero_arien") == arien_pos
    assert session.state.get_position("hero_wasp") == wasp_pos


def test_move_entity_off_map_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(
            session,
            "move_entity",
            {"entity_id": "hero_arien", "hex": {"q": 99, "r": -99, "s": 0}},
        )


def test_remove_then_place_entity_round_trips(session):
    state = session.state
    pos = state.get_position("hero_wasp")
    apply_override_decision(session, "remove_entity", {"entity_id": "hero_wasp"})
    assert session.state.get_position("hero_wasp") is None
    apply_override_decision(
        session, "place_entity", {"entity_id": "hero_wasp", "hex": _hex_dict(pos)}
    )
    assert session.state.get_position("hero_wasp") == pos


def test_place_unknown_entity_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(
            session, "place_entity", {"entity_id": "minion_999", "hex": {"q": 0, "r": 0, "s": 0}}
        )


def test_move_entity_multi_piece_hero_requires_piece_id():
    state = GameSetup.create_game(MAP_PATH, ["Razzle"], ["Wasp"], False, "QUICK", seed=7)
    session = GameSession(state)
    pieces = state.get_piece_ids("hero_razzle")
    assert pieces, "expected Razzle pieces on the board"
    if len(pieces) > 1:
        with pytest.raises(OverrideRejectedError) as exc:
            apply_override_decision(
                session,
                "move_entity",
                {"entity_id": "hero_razzle", "hex": {"q": 0, "r": 0, "s": 0}},
            )
        assert "piece" in exc.value.message.lower()
    # Moving an explicit piece works
    piece = pieces[0]
    target = _free_adjacent(state, piece)
    apply_override_decision(session, "move_entity", {"entity_id": piece, "hex": _hex_dict(target)})
    assert session.state.get_position(piece) == target


def test_summarize_op_is_human_readable():
    text = summarize_op("move_entity", {"entity_id": "minion_4", "hex": {"q": 1, "r": -2, "s": 1}})
    assert "minion_4" in text


# ---------------------------------------------------------------------------
# Resource / counter patch ops (Task 2)
# ---------------------------------------------------------------------------

from goa2.domain.models import GamePhase, TeamColor  # noqa: E402


def test_set_gold_and_level(session):
    apply_override_decision(session, "set_gold", {"hero_id": "hero_arien", "value": 7})
    assert session.state.get_hero("hero_arien").gold == 7
    apply_override_decision(session, "set_level", {"hero_id": "hero_arien", "value": 3})
    assert session.state.get_hero("hero_arien").level == 3


def test_set_gold_unknown_hero_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "set_gold", {"hero_id": "hero_nobody", "value": 7})


def test_set_wave_counter(session):
    lane_id = next(iter(session.state.wave_counters))
    apply_override_decision(session, "set_wave_counter", {"lane_id": lane_id, "value": 3})
    assert session.state.wave_counters[lane_id] == 3


def test_set_wave_counter_unknown_lane_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "set_wave_counter", {"lane_id": "lane_zz", "value": 3})


def test_set_tie_breaker_team(session):
    apply_override_decision(session, "set_tie_breaker_team", {"team": "BLUE"})
    assert session.state.tie_breaker_team == TeamColor.BLUE


def test_set_life_counters_to_zero_ends_game(session):
    apply_override_decision(session, "set_life_counters", {"team": "BLUE", "value": 0})
    state = session.state
    assert state.phase == GamePhase.GAME_OVER
    assert state.teams[TeamColor.BLUE].life_counters == 0
    assert state.winner == TeamColor.RED
    assert state.victory_condition is not None


def test_set_life_counters_resurrects_finished_game(session):
    apply_override_decision(session, "set_life_counters", {"team": "BLUE", "value": 0})
    assert session.state.phase == GamePhase.GAME_OVER
    # Raise back above 0: the ONLY patch that un-ends a game.
    apply_override_decision(session, "set_life_counters", {"team": "BLUE", "value": 2})
    state = session.state
    assert state.phase != GamePhase.GAME_OVER
    assert state.winner is None
    assert state.individual_winner_id is None
    assert state.victory_condition is None
    assert state.teams[TeamColor.BLUE].life_counters == 2


def test_set_life_counters_negative_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "set_life_counters", {"team": "BLUE", "value": -1})


def test_starting_life_counters_untouched(session):
    before = session.state.teams[TeamColor.BLUE].starting_life_counters
    apply_override_decision(session, "set_life_counters", {"team": "BLUE", "value": 1})
    assert session.state.teams[TeamColor.BLUE].starting_life_counters == before


# ---------------------------------------------------------------------------
# Card / marker / effect patch ops (Task 3)
# ---------------------------------------------------------------------------

from goa2.domain.models.effect import (  # noqa: E402
    ActiveEffect,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.models.marker import MarkerType  # noqa: E402


def test_move_card_hand_to_discard_and_back(session):
    hero = session.state.get_hero("hero_arien")
    card = hero.hand[0]
    apply_override_decision(
        session,
        "move_card",
        {"hero_id": "hero_arien", "card_id": card.id, "zone": "discard"},
    )
    hero = session.state.get_hero("hero_arien")
    assert card.id in [c.id for c in hero.discard_pile]
    assert card.id not in [c.id for c in hero.hand]

    apply_override_decision(
        session,
        "move_card",
        {"hero_id": "hero_arien", "card_id": card.id, "zone": "hand"},
    )
    hero = session.state.get_hero("hero_arien")
    assert card.id in [c.id for c in hero.hand]
    assert card.id not in [c.id for c in hero.discard_pile]


def test_move_card_unknown_card_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(
            session,
            "move_card",
            {"hero_id": "hero_arien", "card_id": "not_a_card", "zone": "hand"},
        )


def test_add_and_remove_marker(session):
    apply_override_decision(
        session,
        "add_marker",
        {
            "marker_type": "venom",
            "target_id": "hero_wasp",
            "value": -1,
            "source_id": "hero_arien",
        },
    )
    markers = session.state.get_markers_on_hero("hero_wasp")
    assert any(m.type == MarkerType.VENOM for m in markers)

    apply_override_decision(session, "remove_marker", {"marker_type": "venom"})
    assert not session.state.get_markers_on_hero("hero_wasp")


def test_remove_effect(session):
    session.state.add_effect(
        ActiveEffect(
            id="fx_test_1",
            source_id="hero_arien",
            effect_type=EffectType.AREA_STAT_MODIFIER,
            scope=EffectScope(shape=Shape.GLOBAL, origin_id="hero_arien"),
            duration=DurationType.THIS_ROUND,
            created_at_turn=1,
            created_at_round=1,
        )
    )
    apply_override_decision(session, "remove_effect", {"effect_id": "fx_test_1"})
    assert all(e.id != "fx_test_1" for e in session.state.active_effects)


def test_remove_effect_unknown_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "remove_effect", {"effect_id": "fx_none"})


def test_add_effect_from_payload(session):
    payload = {
        "id": "fx_test_2",
        "source_id": "hero_arien",
        "effect_type": EffectType.AREA_STAT_MODIFIER.value,
        "scope": {"shape": Shape.GLOBAL.value, "origin_id": "hero_arien"},
        "duration": DurationType.THIS_ROUND.value,
        "created_at_turn": 1,
        "created_at_round": 1,
    }
    apply_override_decision(session, "add_effect", {"effect": payload})
    assert any(e.id == "fx_test_2" for e in session.state.active_effects)


def test_add_effect_invalid_payload_rejected(session):
    with pytest.raises(OverrideRejectedError):
        apply_override_decision(session, "add_effect", {"effect": {"id": "x"}})
