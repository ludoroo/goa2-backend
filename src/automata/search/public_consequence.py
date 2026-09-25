"""Privacy-safe measurement of public consequences across one search edge."""

from __future__ import annotations

from dataclasses import dataclass

from goa2.domain.models import ActionType, Card, CardColor, Minion, StatType, TeamColor, Unit
from goa2.domain.models.effect import EffectType
from goa2.domain.state import GameState
from goa2.domain.types import HeroID, UnitID
from goa2.engine.filters_units import is_attack_immune_to_actor
from goa2.engine.map_logic import zones_between
from goa2.engine.rules import can_perform_action_on_card, is_immune_to_actor
from goa2.engine.stats import (
    _is_effect_active as is_effect_active,
)
from goa2.engine.stats import (
    compute_card_stats,
    is_unit_in_effect_scope,
)
from goa2.engine.topology import get_topology_service

_KNOWN_BENEFICIAL_STATS = frozenset(
    {StatType.ATTACK, StatType.DEFENSE, StatType.MOVEMENT, StatType.RANGE, StatType.RADIUS}
)


def _opponent(team: TeamColor) -> TeamColor:
    return TeamColor.BLUE if team == TeamColor.RED else TeamColor.RED


def _sign(team: TeamColor | None, perspective: TeamColor) -> float:
    if team is None:
        return 0.0
    return 1.0 if team == perspective else -1.0


def _clamp(value: float, limit: float = 1.0) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True, slots=True)
class AttackGeometryPair:
    """One public source/target pair and its setup value before the edge."""

    source_id: str
    target_id: str
    before_value: float


@dataclass(frozen=True, slots=True)
class AttackGeometrySpec:
    """Public attack identity and the non-consumable pairs present before an edge."""

    actor_id: str
    card_id: str
    attack_is_basic: bool
    perspective_sign: float
    pairs: tuple[AttackGeometryPair, ...]


@dataclass(frozen=True, slots=True)
class PublicConsequenceVector:
    """Immutable raw component scores from one fixed viewer and perspective."""

    material: float = 0.0
    card_resources: float = 0.0
    modifiers: float = 0.0
    attack_geometry: float = 0.0
    objectives: float = 0.0

    def normalized_change_from(self, before: PublicConsequenceVector) -> PublicConsequenceVector:
        """Normalize every component independently before weighting."""
        return PublicConsequenceVector(
            material=_clamp(self.material - before.material),
            card_resources=_clamp(self.card_resources - before.card_resources),
            modifiers=_clamp((self.modifiers - before.modifiers) / 2.0),
            attack_geometry=_clamp((self.attack_geometry - before.attack_geometry) / 2.0),
            objectives=_clamp((self.objectives - before.objectives) / 2.0),
        )


@dataclass(frozen=True, slots=True)
class PublicConsequenceSnapshot:
    """Before-state token for one edge, pinned to the root information set."""

    root_viewer_id: str
    perspective_team: TeamColor
    turn: int
    round: int
    before: PublicConsequenceVector
    attack_spec: AttackGeometrySpec | None

    @classmethod
    def capture(
        cls,
        state: GameState,
        root_viewer_id: str,
        perspective_team: TeamColor,
    ) -> PublicConsequenceSnapshot:
        viewer = state.get_hero(HeroID(root_viewer_id))
        if viewer is None:
            raise ValueError(f"root viewer does not exist: {root_viewer_id!r}")
        if viewer.team != perspective_team:
            raise ValueError("root viewer does not belong to the consequence perspective")
        attack_spec = _public_attack_spec(state, root_viewer_id, perspective_team)
        return cls(
            root_viewer_id=root_viewer_id,
            perspective_team=perspective_team,
            turn=state.turn,
            round=state.round,
            before=_capture_vector(
                state,
                perspective_team,
                attack_geometry=_attack_geometry_baseline(attack_spec),
            ),
            attack_spec=attack_spec,
        )

    def after(self, state: GameState) -> PublicConsequenceVector:
        crossed_boundary = state.turn != self.turn or state.round != self.round
        return _capture_vector(
            state,
            self.perspective_team,
            card_resources=(
                self.before.card_resources
                if crossed_boundary
                else _public_card_resources(state, self.perspective_team)
            ),
            modifiers=(
                self.before.modifiers
                if crossed_boundary
                else _public_numeric_modifiers(state, self.perspective_team)
            ),
            attack_geometry=(
                self.before.attack_geometry
                if crossed_boundary
                else self.before.attack_geometry + _attack_geometry_delta(state, self.attack_spec)
            ),
        )

    def normalized_delta(self, state: GameState) -> PublicConsequenceVector:
        return self.after(state).normalized_change_from(self.before)

    def edge_units(self, state: GameState) -> float:
        """Combine normalized components in the established +/-5 edge envelope."""
        delta = self.normalized_delta(state)
        value = (
            1.5 * delta.material
            + 0.5 * delta.card_resources
            + 0.75 * delta.modifiers
            + 0.75 * delta.attack_geometry
            + 1.0 * delta.objectives
        )
        return _clamp(value, 5.0)


def _capture_vector(
    state: GameState,
    perspective: TeamColor,
    *,
    card_resources: float | None = None,
    modifiers: float | None = None,
    attack_geometry: float = 0.0,
) -> PublicConsequenceVector:
    return PublicConsequenceVector(
        material=public_material(state, perspective),
        card_resources=(
            _public_card_resources(state, perspective) if card_resources is None else card_resources
        ),
        modifiers=(
            _public_numeric_modifiers(state, perspective) if modifiers is None else modifiers
        ),
        attack_geometry=attack_geometry,
        objectives=_objective_control(state, perspective),
    )


def public_material(state: GameState, perspective: TeamColor) -> float:
    """Existing public life, gold, and on-board minion material recipe."""
    ours = state.teams[perspective]
    enemy = state.teams[_opponent(perspective)]
    score = float(ours.life_counters - enemy.life_counters)
    score += 0.1 * (sum(hero.gold for hero in ours.heroes) - sum(h.gold for h in enemy.heroes))
    score += 0.25 * (
        sum(m.value for m in ours.minions if state.has_board_presence(str(m.id)))
        - sum(m.value for m in enemy.minions if state.has_board_presence(str(m.id)))
    )
    return score


def _public_card_resources(state: GameState, perspective: TeamColor) -> float:
    """Orient every public hand size without inspecting any hidden card identity."""
    return float(
        sum(
            _sign(team_color, perspective) * len(hero.hand)
            for team_color, team in state.teams.items()
            for hero in team.heroes
        )
    )


def _public_numeric_modifiers(state: GameState, perspective: TeamColor) -> float:
    """Measure only public numeric effects whose beneficiary and sign are known."""
    score = 0.0
    units = [
        unit
        for team in state.teams.values()
        for unit in (*team.heroes, *team.minions)
        if state.has_board_presence(str(unit.id))
    ]
    for effect in state.active_effects:
        known_area_modifier = (
            effect.effect_type == EffectType.AREA_STAT_MODIFIER
            and effect.stat_type in _KNOWN_BENEFICIAL_STATS
        )
        known_basic_modifier = effect.effect_type == EffectType.BASIC_ACTION_STAT_BONUS
        if (
            not is_effect_active(effect, state)
            or not (known_area_modifier or known_basic_modifier)
            or effect.stat_value == 0
        ):
            continue
        for unit in units:
            if known_basic_modifier and state.get_hero(HeroID(str(unit.id))) is None:
                continue
            candidate_ids = state.get_piece_ids(str(unit.id)) or [str(unit.id)]
            if any(
                is_unit_in_effect_scope(effect, candidate, state) for candidate in candidate_ids
            ):
                score += _sign(unit.team, perspective) * effect.stat_value

    for marker in state.markers.values():
        if marker.target_id is None:
            continue
        target = state.get_hero(HeroID(marker.target_id))
        effects = [
            value for stat, value in marker.get_stat_effects() if stat in _KNOWN_BENEFICIAL_STATS
        ]
        if target is not None and effects:
            # One marker is one public consequence, even when it modifies several stats.
            score += _sign(target.team, perspective) * sum(effects) / len(effects)
    return score


def _public_attack_spec(
    state: GameState,
    root_viewer_id: str,
    perspective: TeamColor,
) -> AttackGeometrySpec | None:
    actor_id = str(state.current_actor_id) if state.current_actor_id else None
    if actor_id is None:
        return None
    actor = state.get_hero(HeroID(actor_id))
    if actor is None or actor.team is None:
        return None
    card = state.get_performing_card(str(actor.id)) or actor.current_turn_card
    if card is None or (str(actor.id) != root_viewer_id and card.is_facedown):
        return None
    actions = {card.current_primary_action, *card.current_secondary_actions}
    if ActionType.ATTACK not in actions:
        return None

    is_basic = card.current_color in (CardColor.GOLD, CardColor.SILVER)
    attack_range = compute_card_stats(state, UnitID(str(actor.id)), card).range
    action_allowed = can_perform_action_on_card(state, str(actor.id), ActionType.ATTACK, card)
    source_ids = state.get_piece_ids(str(actor.id)) or [str(actor.id)]
    enemy = _opponent(actor.team)
    targets = [*state.teams[enemy].heroes, *state.teams[enemy].minions]
    pairs: list[AttackGeometryPair] = []
    for source_id in source_ids:
        for target in targets:
            for target_id in state.get_piece_ids(str(target.id)) or [str(target.id)]:
                value = _attack_pair_value(
                    state,
                    attack_is_basic=is_basic,
                    attack_range=attack_range,
                    action_allowed=action_allowed,
                    source_id=source_id,
                    target_id=target_id,
                )
                if value is not None:
                    pairs.append(AttackGeometryPair(source_id, target_id, value))
    return AttackGeometrySpec(
        str(actor.id),
        str(card.id),
        is_basic,
        _sign(actor.team, perspective),
        tuple(pairs),
    )


def _find_actor_card(state: GameState, spec: AttackGeometrySpec) -> tuple[Unit | None, Card | None]:
    actor = state.get_hero(HeroID(spec.actor_id))
    if actor is None:
        return None, None
    return actor, state.get_card_by_id(spec.card_id)


def _attack_pair_value(
    state: GameState,
    *,
    attack_is_basic: bool,
    attack_range: int,
    action_allowed: bool,
    source_id: str,
    target_id: str,
) -> float | None:
    source = state.get_unit(UnitID(source_id))
    target = state.get_unit(UnitID(target_id))
    source_hex = state.get_position(source_id)
    target_hex = state.get_position(target_id)
    if source is None or target is None or source_hex is None or target_hex is None:
        return None

    distance = get_topology_service().distance(
        source_hex,
        target_hex,
        state,
        unit_ids=[source_id, target_id],
    )
    range_setup = _clamp(float(attack_range - distance), 2.0) * 0.125
    reachable = bool(
        distance <= attack_range
        and action_allowed
        and state.validator.can_be_targeted(state, source_id, target_id).allowed
        and not is_immune_to_actor(target, state, actor_id=source_id)
        and not is_attack_immune_to_actor(
            target_id,
            state,
            actor_id=source_id,
            attack_is_basic=attack_is_basic,
        )
    )
    return range_setup + (0.25 if reachable else 0.0)


def _attack_geometry_baseline(spec: AttackGeometrySpec | None) -> float:
    if spec is None or not spec.pairs:
        return 0.0
    return spec.perspective_sign * (sum(pair.before_value for pair in spec.pairs) / len(spec.pairs))


def _attack_geometry_delta(state: GameState, spec: AttackGeometrySpec | None) -> float:
    if spec is None:
        return 0.0
    _actor, card = _find_actor_card(state, spec)
    if card is None:
        return 0.0
    attack_range = compute_card_stats(state, UnitID(spec.actor_id), card).range
    action_allowed = can_perform_action_on_card(state, spec.actor_id, ActionType.ATTACK, card)
    changes: list[float] = []
    for pair in spec.pairs:
        after_value = _attack_pair_value(
            state,
            attack_is_basic=spec.attack_is_basic,
            attack_range=attack_range,
            action_allowed=action_allowed,
            source_id=pair.source_id,
            target_id=pair.target_id,
        )
        if after_value is not None:
            changes.append(after_value - pair.before_value)
    return spec.perspective_sign * sum(changes) / len(changes) if changes else 0.0


def _objective_control(state: GameState, perspective: TeamColor) -> float:
    """Directional lane progress plus occupancy of each current battle zone."""
    score = 0.0
    for team_color, team in state.teams.items():
        relation = _sign(team_color, perspective)
        for unit in (*team.heroes, *team.minions):
            positions = state.get_positions(str(unit.id))
            progress_values: list[float] = []
            for position in positions:
                zone_id = state.board.get_zone_for_hex(position)
                lane_id = state.lane_of_zone(zone_id) if zone_id is not None else None
                if lane_id is None or zone_id is None:
                    continue
                lane_length = max(1, len(state.board.lanes[lane_id]) - 1)
                progress_values.append(
                    zones_between(state, team_color, lane_id, zone_id) / lane_length
                )
            if progress_values:
                weight = 0.5 if isinstance(unit, Minion) else 1.0
                score += relation * weight * sum(progress_values) / len(progress_values)

        for zone_id in state.battle_zones.values():
            zone = state.board.zones.get(zone_id)
            if zone is None:
                continue
            for minion in team.minions:
                minion_position = state.get_position(str(minion.id))
                if minion_position in zone.hexes:
                    score += relation * 0.5
            for hero in team.heroes:
                positions = state.get_positions(str(hero.id))
                if positions:
                    occupancy = sum(position in zone.hexes for position in positions) / len(
                        positions
                    )
                    score += relation * 0.25 * occupancy
    return score


__all__ = [
    "AttackGeometryPair",
    "AttackGeometrySpec",
    "PublicConsequenceSnapshot",
    "PublicConsequenceVector",
    "public_material",
]
