from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from automata.decision import DecisionDescriptor
from automata.runtime.clone import clone_state
from automata.runtime.effects import register_all_effects
from automata.search.contracts import SearchContext
from automata.search.heuristic import HeuristicLeafEvaluator
from automata.search.public_consequence import (
    PublicConsequenceSnapshot,
    PublicConsequenceVector,
)
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.models import ActionType, StatType, TeamColor
from goa2.domain.models.effect import (
    ActiveEffect,
    AffectsFilter,
    DurationType,
    EffectScope,
    EffectType,
    Shape,
)
from goa2.domain.models.marker import MarkerType
from goa2.domain.types import HeroID
from goa2.engine.map_logic import zones_between
from goa2.engine.setup import GameSetup


def _state():
    register_all_effects()
    return GameSetup.create_game(
        "src/goa2/data/maps/forgotten_island.json",
        ["Wasp", "Xargatha"],
        ["Arien"],
        game_type="QUICK",
        seed=7,
    )


def _snapshot(state) -> PublicConsequenceSnapshot:
    return PublicConsequenceSnapshot.capture(state, "hero_wasp", TeamColor.RED)


def test_public_consequence_snapshot_is_immutable_and_own_card_spend_costs_value() -> None:
    state = _state()
    snapshot = _snapshot(state)
    spent = clone_state(state)
    wasp = spent.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    wasp.discard_card(wasp.hand[0])

    delta = snapshot.normalized_delta(spent)

    assert isinstance(snapshot.before, PublicConsequenceVector)
    assert delta.card_resources < 0
    assert snapshot.edge_units(spent) < 0
    with pytest.raises(FrozenInstanceError):
        snapshot.before.material = 99  # type: ignore[misc]


def test_card_spend_signs_are_perspective_directional_without_block_context() -> None:
    state = _state()
    snapshot = _snapshot(state)

    own_spend = clone_state(state)
    own = own_spend.get_hero(HeroID("hero_wasp"))
    assert own is not None
    own.discard_card(own.hand[0])
    own_spend.execution_context["block_succeeded"] = True

    enemy_spend = clone_state(state)
    enemy = enemy_spend.get_hero(HeroID("hero_arien"))
    assert enemy is not None
    enemy.discard_card(enemy.hand[0])
    enemy_spend.execution_context["block_succeeded"] = True

    assert snapshot.normalized_delta(own_spend).card_resources < 0
    assert snapshot.edge_units(own_spend) < 0
    assert snapshot.normalized_delta(enemy_spend).card_resources > 0
    assert snapshot.edge_units(enemy_spend) > 0


def test_worthwhile_defense_beats_spending_the_card_and_still_being_defeated() -> None:
    state = _state()
    snapshot = _snapshot(state)
    blocked = clone_state(state)
    defender = blocked.get_hero(HeroID("hero_wasp"))
    assert defender is not None
    defender.discard_card(defender.hand[0])
    blocked.execution_context["block_succeeded"] = True

    defeated = clone_state(blocked)
    defeated.execution_context["block_succeeded"] = False
    defeated.teams[TeamColor.RED].life_counters -= 1
    defeated.remove_entity("hero_wasp")

    assert snapshot.edge_units(blocked) < 0
    assert snapshot.edge_units(blocked) > snapshot.edge_units(defeated)


def test_public_modifier_and_marker_signs_follow_affected_team() -> None:
    state = _state()
    snapshot = _snapshot(state)
    red_buff = clone_state(state)
    red_buff.add_effect(
        ActiveEffect(
            id="red-attack-buff",
            source_id="hero_wasp",
            effect_type=EffectType.AREA_STAT_MODIFIER,
            scope=EffectScope(shape=Shape.GLOBAL, affects=AffectsFilter.FRIENDLY_HEROES),
            stat_type=StatType.ATTACK,
            stat_value=1,
            duration=DurationType.PASSIVE,
            created_at_turn=red_buff.turn,
            created_at_round=red_buff.round,
        )
    )
    blue_buff = clone_state(state)
    blue_buff.add_effect(
        ActiveEffect(
            id="blue-defense-buff",
            source_id="hero_arien",
            effect_type=EffectType.AREA_STAT_MODIFIER,
            scope=EffectScope(
                shape=Shape.GLOBAL,
                affects=AffectsFilter.SELF_AND_FRIENDLY_HEROES,
            ),
            stat_type=StatType.DEFENSE,
            stat_value=1,
            duration=DurationType.THIS_ROUND,
            created_at_turn=blue_buff.turn,
            created_at_round=blue_buff.round,
            is_active=True,
        )
    )
    enemy_debuff = clone_state(state)
    enemy_debuff.place_marker(MarkerType.VENOM, "hero_arien", -1, "hero_wasp")
    non_stat_marker = clone_state(state)
    non_stat_marker.place_marker(MarkerType.BOUNTY, "hero_arien", 7, "hero_wasp")

    assert snapshot.normalized_delta(red_buff).modifiers > 0
    assert snapshot.normalized_delta(blue_buff).modifiers < 0
    assert snapshot.normalized_delta(enemy_debuff).modifiers > 0
    assert snapshot.normalized_delta(non_stat_marker).modifiers == 0


def test_attack_geometry_values_movement_into_and_out_of_reach() -> None:
    state = _state()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    card = next(card for card in wasp.hand if card.primary_action is ActionType.ATTACK)
    wasp.hand.remove(card)
    wasp.current_turn_card = card
    card.is_facedown = False
    state.current_actor_id = wasp.id
    actor_hex = state.get_position("hero_wasp")
    assert actor_hex is not None

    target = state.teams[TeamColor.BLUE].minions[0]
    for enemy in (
        *state.teams[TeamColor.BLUE].heroes,
        *state.teams[TeamColor.BLUE].minions,
    ):
        state.remove_entity(str(enemy.id))
    line = [
        hex_
        for hex_ in state.board.tiles
        if actor_hex.distance(hex_) in {2, 3}
        and actor_hex.is_straight_line(hex_)
        and not state.board.get_tile(hex_).is_obstacle
    ]
    far_hex = min(line, key=lambda hex_: actor_hex.distance(hex_))
    state.place_entity(str(target.id), far_hex)
    snapshot = _snapshot(state)

    moved_in = clone_state(state)
    moved_in.move_unit(wasp.id, min(actor_hex.neighbors(), key=lambda h: h.distance(far_hex)))
    moved_out = clone_state(state)
    farther = max(
        (hex_ for hex_ in moved_out.board.tiles if not moved_out.board.get_tile(hex_).is_obstacle),
        key=lambda hex_: hex_.distance(far_hex),
    )
    moved_out.move_unit(wasp.id, farther)

    assert snapshot.normalized_delta(moved_in).attack_geometry > 0
    assert snapshot.normalized_delta(moved_out).attack_geometry < 0


def test_attack_geometry_has_no_straight_line_preference_but_tracks_public_los() -> None:
    state = _state()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    base = next(card for card in wasp.hand if card.primary_action is ActionType.ATTACK)
    card = base.model_copy(update={"is_ranged": True, "range_value": 3})
    wasp.hand.remove(base)
    wasp.current_turn_card = card
    card.is_facedown = False
    state.current_actor_id = wasp.id
    actor_hex = state.get_position("hero_wasp")
    assert actor_hex is not None

    target = next(minion for minion in state.teams[TeamColor.BLUE].minions if not minion.is_heavy)
    for enemy in (
        *state.teams[TeamColor.BLUE].heroes,
        *state.teams[TeamColor.BLUE].minions,
    ):
        state.remove_entity(str(enemy.id))
    target_hex = next(
        hex_
        for hex_ in state.board.tiles
        if actor_hex.distance(hex_) == 2
        and actor_hex.is_straight_line(hex_)
        and not state.board.get_tile(hex_).is_obstacle
    )
    state.place_entity(str(target.id), target_hex)
    snapshot = _snapshot(state)

    off_line = clone_state(state)
    off_line_hex = next(
        hex_
        for hex_ in off_line.board.tiles
        if hex_.distance(target_hex) == 2
        and not hex_.is_straight_line(target_hex)
        and not off_line.board.get_tile(hex_).is_obstacle
    )
    off_line.move_unit(wasp.id, off_line_hex)

    blocked = clone_state(state)
    midpoint = next(
        hex_
        for hex_ in blocked.board.tiles
        if hex_.is_on_segment(actor_hex, target_hex, exclusive=True)
    )
    blocked.add_effect(
        ActiveEffect(
            id="public-los-blocker",
            source_id="hero_arien",
            effect_type=EffectType.LOS_BLOCKER,
            scope=EffectScope(shape=Shape.POINT, origin_hex=midpoint),
            duration=DurationType.THIS_ROUND,
            created_at_turn=blocked.turn,
            created_at_round=blocked.round,
            is_active=True,
        )
    )

    assert snapshot.normalized_delta(off_line).attack_geometry == pytest.approx(0.0)
    assert snapshot.normalized_delta(blocked).attack_geometry < 0


def test_realized_minion_kill_outvalues_merely_moving_adjacent() -> None:
    state = _state()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    card = next(card for card in wasp.hand if card.primary_action is ActionType.ATTACK)
    wasp.hand.remove(card)
    wasp.current_turn_card = card
    card.is_facedown = False
    state.current_actor_id = wasp.id
    actor_hex = state.get_position("hero_wasp")
    assert actor_hex is not None

    target = next(
        minion
        for minion in state.teams[TeamColor.BLUE].minions
        if minion.value == 2 and not minion.is_heavy
    )
    for enemy in (
        *state.teams[TeamColor.BLUE].heroes,
        *state.teams[TeamColor.BLUE].minions,
    ):
        state.remove_entity(str(enemy.id))
    target_hex = next(
        hex_
        for hex_ in state.board.tiles
        if actor_hex.distance(hex_) == 2 and not state.board.get_tile(hex_).is_obstacle
    )
    state.place_entity(str(target.id), target_hex)

    request = InputRequest(
        id="active-attack",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value("resolve")],
    )
    decision = DecisionDescriptor("INPUT", request=request)
    context = SearchContext("hero_wasp", TeamColor.RED, "hero_wasp", decision)
    evaluator = HeuristicLeafEvaluator()
    prepared = evaluator.prepare_immediate_edge(context, state, decision, "resolve")

    moved = clone_state(state)
    adjacent = min(
        (hex_ for hex_ in actor_hex.neighbors() if hex_ in moved.board.tiles),
        key=lambda hex_: hex_.distance(target_hex),
    )
    moved.move_unit(wasp.id, adjacent)
    killed = clone_state(state)
    killed.remove_entity(str(target.id))

    assert (
        evaluator.evaluate_immediate_edge(context, killed, prepared).value
        > evaluator.evaluate_immediate_edge(context, moved, prepared).value
    )


def test_attack_geometry_has_no_no_target_floor() -> None:
    state = _state()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    card = next(card for card in wasp.hand if card.primary_action is ActionType.ATTACK)
    wasp.hand.remove(card)
    wasp.current_turn_card = card
    card.is_facedown = False
    state.current_actor_id = wasp.id
    for enemy in (
        *state.teams[TeamColor.BLUE].heroes,
        *state.teams[TeamColor.BLUE].minions,
    ):
        state.remove_entity(str(enemy.id))

    snapshot = _snapshot(state)

    assert snapshot.before.attack_geometry == 0
    assert snapshot.attack_spec is not None
    assert snapshot.attack_spec.pairs == ()


def test_consumed_attack_pairs_make_death_or_kill_geometry_neutral() -> None:
    state = _state()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    card = next(card for card in wasp.hand if card.primary_action is ActionType.ATTACK)
    wasp.hand.remove(card)
    wasp.current_turn_card = card
    card.is_facedown = False
    state.current_actor_id = wasp.id
    snapshot = _snapshot(state)

    target_id = snapshot.attack_spec.pairs[0].target_id  # type: ignore[union-attr]
    killed = clone_state(state)
    killed.remove_entity(target_id)
    died = clone_state(state)
    died.remove_entity("hero_wasp")

    assert snapshot.normalized_delta(killed).attack_geometry == 0
    assert snapshot.normalized_delta(died).attack_geometry == 0


def test_attack_geometry_is_zero_sum_when_enemy_escapes_the_active_actor() -> None:
    state = _state()
    wasp = state.get_hero(HeroID("hero_wasp"))
    arien = state.get_hero(HeroID("hero_arien"))
    assert wasp is not None and arien is not None
    card = next(card for card in wasp.hand if card.primary_action is ActionType.ATTACK)
    wasp.hand.remove(card)
    wasp.current_turn_card = card
    card.is_facedown = False
    state.current_actor_id = wasp.id
    actor_hex = state.get_position("hero_wasp")
    assert actor_hex is not None

    for enemy in state.teams[TeamColor.BLUE].minions:
        state.remove_entity(str(enemy.id))
    adjacent = next(
        hex_
        for hex_ in actor_hex.neighbors()
        if hex_ in state.board.tiles and not state.board.get_tile(hex_).is_obstacle
    )
    state.move_unit(arien.id, adjacent)
    red_snapshot = PublicConsequenceSnapshot.capture(state, "hero_wasp", TeamColor.RED)
    blue_snapshot = PublicConsequenceSnapshot.capture(state, "hero_arien", TeamColor.BLUE)

    escaped = clone_state(state)
    occupied = set(escaped.entity_locations.values())
    far_hex = max(
        (
            hex_
            for hex_ in escaped.board.tiles
            if not escaped.board.get_tile(hex_).is_obstacle and hex_ not in occupied
        ),
        key=lambda hex_: actor_hex.distance(hex_),
    )
    escaped.move_unit(arien.id, far_hex)

    red_delta = red_snapshot.normalized_delta(escaped).attack_geometry
    blue_delta = blue_snapshot.normalized_delta(escaped).attack_geometry
    assert red_delta < 0 < blue_delta
    assert red_delta == pytest.approx(-blue_delta)


def test_attack_geometry_uses_spec_actor_when_current_actor_changes() -> None:
    state = _state()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    base = next(card for card in wasp.hand if card.primary_action is ActionType.ATTACK)
    card = base.model_copy(update={"is_ranged": True, "range_value": 3})
    wasp.hand.remove(base)
    wasp.current_turn_card = card
    card.is_facedown = False
    state.current_actor_id = wasp.id
    actor_hex = state.get_position("hero_wasp")
    assert actor_hex is not None

    target = state.get_hero(HeroID("hero_arien"))
    assert target is not None
    for enemy in (
        *state.teams[TeamColor.BLUE].heroes,
        *state.teams[TeamColor.BLUE].minions,
    ):
        state.remove_entity(str(enemy.id))
    target_hex = next(
        hex_
        for hex_ in state.board.tiles
        if actor_hex.distance(hex_) == 2 and not state.board.get_tile(hex_).is_obstacle
    )
    state.place_entity(str(target.id), target_hex)
    state.add_effect(
        ActiveEffect(
            id="target-attack-immunity",
            source_id=str(target.id),
            effect_type=EffectType.ATTACK_IMMUNITY,
            scope=EffectScope(shape=Shape.POINT, origin_id=str(target.id)),
            except_attacker_ids=["hero_wasp"],
            duration=DurationType.THIS_ROUND,
            created_at_turn=state.turn,
            created_at_round=state.round,
            is_active=True,
        )
    )
    snapshot = _snapshot(state)
    switched = clone_state(state)
    switched.current_actor_id = HeroID("hero_xargatha")

    assert snapshot.normalized_delta(switched).attack_geometry == 0


@pytest.mark.parametrize(("field", "increment"), [("turn", 1), ("round", 1)])
def test_transient_components_freeze_across_turn_or_round_boundary(
    field: str, increment: int
) -> None:
    state = _state()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    card = next(card for card in wasp.hand if card.primary_action is ActionType.ATTACK)
    wasp.hand.remove(card)
    wasp.current_turn_card = card
    card.is_facedown = False
    state.current_actor_id = wasp.id
    state.add_effect(
        ActiveEffect(
            id="expiring-buff",
            source_id="hero_wasp",
            effect_type=EffectType.AREA_STAT_MODIFIER,
            scope=EffectScope(shape=Shape.GLOBAL, affects=AffectsFilter.FRIENDLY_HEROES),
            stat_type=StatType.ATTACK,
            stat_value=2,
            duration=DurationType.THIS_TURN,
            created_at_turn=state.turn,
            created_at_round=state.round,
            is_active=True,
        )
    )
    snapshot = _snapshot(state)
    crossed = clone_state(state)
    setattr(crossed, field, getattr(crossed, field) + increment)
    crossed_wasp = crossed.get_hero(HeroID("hero_wasp"))
    assert crossed_wasp is not None
    crossed_wasp.discard_card(crossed_wasp.hand[0])
    actor_hex = crossed.get_position("hero_wasp")
    assert actor_hex is not None
    destination = next(hex_ for hex_ in actor_hex.neighbors() if hex_ in crossed.board.tiles)
    crossed.move_unit(HeroID("hero_wasp"), destination)

    delta = snapshot.normalized_delta(crossed)
    assert delta.card_resources == 0
    assert delta.modifiers == 0
    assert delta.attack_geometry == 0


def test_objective_progress_and_battle_zone_control_are_directional() -> None:
    state = _state()
    wasp = state.get_hero(HeroID("hero_wasp"))
    assert wasp is not None
    start = state.get_position(wasp.id)
    assert start is not None
    lane = state.lane_of_zone(state.board.get_zone_for_hex(start) or "")
    assert lane is not None
    zones = state.board.lanes[lane]
    start_index = zones.index(state.board.get_zone_for_hex(start) or "")
    forward_zone = state.board.zones[zones[start_index + 2]]
    destination = min(
        forward_zone.hexes,
        key=lambda hex_: (start.distance(hex_), hex_.q, hex_.r, hex_.s),
    )
    assert zones_between(state, TeamColor.RED, lane, zones[start_index + 2]) > zones_between(
        state, TeamColor.RED, lane, zones[start_index]
    )
    snapshot = _snapshot(state)
    advanced = clone_state(state)
    advanced.move_unit(wasp.id, destination)

    battle_zone = state.board.zones[state.battle_zones[lane]]
    battle_hex = next(
        hex_ for hex_ in battle_zone.hexes if not advanced.board.get_tile(hex_).is_obstacle
    )
    controlled = clone_state(advanced)
    controlled.move_unit(HeroID("hero_xargatha"), battle_hex)

    assert snapshot.normalized_delta(advanced).objectives > 0
    assert snapshot.edge_units(controlled) > snapshot.edge_units(advanced)


def test_multipiece_battle_zone_occupancy_is_averaged() -> None:
    state = GameSetup.create_game(
        "src/goa2/data/maps/across_the_river.json",
        ["Razzle"],
        ["Arien"],
        game_type="QUICK",
        seed=31,
    )
    pieces = [
        str(entity_id)
        for entity_id, entity in state.misc_entities.items()
        if getattr(entity, "owner_hero_id", None) == "hero_razzle"
    ][:2]
    for entity_id in list(state.entity_locations):
        state.remove_entity(entity_id)
    lane_id = sorted(state.battle_zones)[0]
    battle_zone = state.board.zones[state.battle_zones[lane_id]]
    battle_hex = sorted(battle_zone.hexes, key=lambda h: (h.q, h.r, h.s))[0]
    outside_zone_id = next(
        zone_id for zone_id in state.board.lanes[lane_id] if zone_id != state.battle_zones[lane_id]
    )
    outside_hex = sorted(
        state.board.zones[outside_zone_id].hexes,
        key=lambda h: (h.q, h.r, h.s),
    )[0]
    state.place_entity(pieces[0], battle_hex)
    state.place_entity(pieces[1], outside_hex)

    both = PublicConsequenceSnapshot.capture(state, "hero_razzle", TeamColor.RED).before.objectives
    battle_only = clone_state(state)
    battle_only.remove_entity(pieces[1])
    outside_only = clone_state(state)
    outside_only.remove_entity(pieces[0])

    assert both == pytest.approx(
        (
            PublicConsequenceSnapshot.capture(
                battle_only, "hero_razzle", TeamColor.RED
            ).before.objectives
            + PublicConsequenceSnapshot.capture(
                outside_only, "hero_razzle", TeamColor.RED
            ).before.objectives
        )
        / 2.0
    )


def test_hidden_enemy_card_identity_is_invariant_but_public_count_changes_value() -> None:
    state = _state()
    baseline = _snapshot(state)
    identity_variant = clone_state(state)
    enemy = identity_variant.get_hero(HeroID("hero_arien"))
    assert enemy is not None
    enemy.hand.reverse()

    assert _snapshot(identity_variant).before == baseline.before

    count_variant = clone_state(state)
    enemy = count_variant.get_hero(HeroID("hero_arien"))
    assert enemy is not None
    enemy.hand.pop()

    assert baseline.normalized_delta(count_variant).card_resources > 0
