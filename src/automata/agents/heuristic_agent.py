"""Greedy heuristic agent.

Picks locally-best decisions from the engine's enumerated legal options using
fast static scoring.

Priorities, roughly:
- Play impactful cards (attack when a target is reachable, else advance).
- Attack enemy heroes > enemy minions (esp. in the battle zone).
- Move toward the enemy throne / the fight (push the objective).
- Defend rather than die; take the biggest number when asked.
"""

from __future__ import annotations

import math
from typing import Any

import goa2.engine.stats as stats
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequest, selection_value
from goa2.domain.models import ActionType, StatType, TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.effect import EffectType
from goa2.domain.models.unit import Hero, is_hero_unit
from goa2.domain.state import GameState
from goa2.domain.types import HeroID, UnitID
from goa2.engine.filters_hex import FastTravelDestinationFilter
from goa2.engine.map_logic import zones_between
from goa2.engine.rules import find_reachable_hexes, validate_target
from goa2.engine.stats import calculate_minion_defense_modifier, compute_card_stats
from goa2.engine.topology import get_topology_service

# CHOOSE_ACTION priority (higher = preferred as the played action).
_ACTION_PRIORITY = {
    ActionType.ATTACK: 5,
    ActionType.SKILL: 4,
    ActionType.MOVEMENT: 3,
    # When the engine offers fast travel it has already proved there is a safe
    # destination. Prefer that free zone reposition over ordinary movement,
    # while still yielding to an actionable attack or skill.
    ActionType.FAST_TRAVEL: 3.5,
    ActionType.DEFENSE: 2,
    ActionType.CLEAR: 1,
    ActionType.HOLD: 0,
}

_POSITIONAL_ACTION_BASELINE = 3.5
_POSITIONAL_GAIN_SCALE = 10.0
_POSITIONAL_ACTION_CEILING = math.nextafter(
    float(_ACTION_PRIORITY[ActionType.SKILL]), -math.inf
)


def _is_reversed_initiative(state: GameState) -> bool:
    return any(
        effect.effect_type == EffectType.REVERSED_INITIATIVE
        and effect.is_active
        and state.round == effect.created_at_round
        and state.turn == effect.created_at_turn + 1
        for effect in state.active_effects
    )


class HeuristicAgent:
    def __init__(self, seed: int = 0) -> None:
        import random

        self._rng = random.Random(seed)

    # --- helpers -----------------------------------------------------------
    def _enemy_positions(self, state: GameState, team: TeamColor) -> list[Any]:
        enemy = TeamColor.BLUE if team == TeamColor.RED else TeamColor.RED
        out = []
        for unit in [*state.teams[enemy].heroes, *state.teams[enemy].minions]:
            out.extend(state.get_positions(str(unit.id)))
        return out

    def _unit_team(self, state: GameState, uid: str) -> TeamColor | None:
        owner_id = state.hero_owner_id(uid)
        for color, team in state.teams.items():
            if any(str(u.id) == owner_id for u in [*team.heroes, *team.minions]):
                return color
        return None

    def _zone_of(self, state: GameState, loc: Any) -> str | None:
        for zid, zone in state.board.zones.items():
            if loc in zone.hexes:
                return zid
        return None

    # --- planning ----------------------------------------------------------
    def choose_card(self, state: GameState, hero: Hero) -> Card | None:
        if not hero.hand:
            return None
        initiative_direction = -1 if _is_reversed_initiative(state) else 1

        # Highest score; break ties by effective acting order, then rng.
        best = max(
            hero.hand,
            key=lambda card: (
                self.score_card(state, hero, card),
                initiative_direction
                * stats.get_computed_stat(
                    state,
                    hero.id,
                    StatType.INITIATIVE,
                    card.get_base_stat_value(StatType.INITIATIVE),
                    performing_card=card,
                ),
                self._rng.random(),
            ),
        )
        return best

    def score_card(self, state: GameState, hero: Hero, card: Card) -> float:
        """Static desirability of committing ``card`` for ``hero`` (higher = better).

        Public so the search layer can reuse it as an expansion prior. Pure and
        side-effect free; does not consult the RNG (callers break ties).
        """
        team = hero.team or TeamColor.RED
        positions = state.get_positions(str(hero.id))
        enemies = self._enemy_positions(state, team)
        topology = get_topology_service()
        nearest = min(
            (topology.distance(pos, enemy, state) for pos in positions for enemy in enemies),
            default=float("inf"),
        )

        pa = card.primary_action
        stats = compute_card_stats(state, UnitID(str(hero.id)), card)
        if pa == ActionType.ATTACK:
            reachable = self._has_attack_target(state, hero, stats.range)
            return 10 + stats.primary_value + (5 if reachable else -3)
        if pa == ActionType.SKILL:
            return 6
        if pa == ActionType.MOVEMENT:
            return 5 + (3 if nearest > 2 else 0)
        if pa == ActionType.DEFENSE:
            return 4
        return 3

    def _has_attack_target(self, state: GameState, hero: Hero, range_val: int) -> bool:
        """Return whether the canonical target validator finds a public enemy."""
        team = hero.team or TeamColor.RED
        source_ids = state.get_piece_ids(str(hero.id)) or [str(hero.id)]
        sources = [state.get_unit(UnitID(uid)) for uid in source_ids]
        for other_color, other_team in state.teams.items():
            if other_color == team:
                continue
            target_ids = [
                piece_id
                for unit in [*other_team.heroes, *other_team.minions]
                for piece_id in (state.get_piece_ids(str(unit.id)) or [str(unit.id)])
            ]
            for source in sources:
                if source is None:
                    continue
                for target_id in target_ids:
                    target = state.get_unit(UnitID(target_id))
                    if target is not None and validate_target(
                        source, target, ActionType.ATTACK, state, range_val
                    ):
                        return True
        return False

    # --- resolution --------------------------------------------------------
    def choose_input(
        self,
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
        decision_owner_hero_id: str | None = None,
    ) -> Any:
        # ``owned_hero_ids`` is accepted for Agent-protocol compatibility with
        # the runtime driver (see ``automata.agents.base.Agent``); the
        # heuristic policy doesn't need it, but the driver passes it
        # uniformly so search-backed agents can enforce ownership.
        rt = request.request_type.value
        opts = list(request.options)

        if rt == "UPGRADE_PHASE":
            return self._choose_upgrade(request)

        if not opts:
            return "SKIP" if request.can_skip else None

        best = max(opts, key=lambda o: self.score_option(state, request, o))
        return selection_value(best)

    def score_option(self, state: GameState, request: InputRequest, option: Any) -> float:
        """Static desirability of an input ``option`` (higher = better).

        Public so the search layer can reuse it as an expansion prior. Mirrors
        the per-request-type ranking used by :meth:`choose_input`. Pure and
        side-effect free.
        """
        rt = request.request_type.value

        if rt == "CHOOSE_ACTION":
            return self._action_score(state, request, option)

        if rt in ("SELECT_UNIT", "SELECT_ENEMY", "SELECT_UNIT_OR_TOKEN"):
            return self._unit_score(state, option, self._acting_team(state, request))

        if rt in ("SELECT_HEX", "MOVEMENT_HEX", "FAST_TRAVEL_DESTINATION", "CHOOSE_RESPAWN_HEX"):
            return self._hex_score(state, option, self._acting_team(state, request))

        if rt == "SELECT_NUMBER":
            # More (push/move/repeat) is usually better.
            return float(_as_int(selection_value(option)))

        if rt in ("DEFENSE_CARD", "SELECT_CARD_OR_PASS"):
            # Prefer to defend (survive) rather than skip into defeat.
            return float(_as_int(option.metadata.get("defense_value", 0)))

        # Default: no preference between concrete options.
        return 0.0

    # --- scoring -----------------------------------------------------------
    def _action_priority(self, option: Any) -> float:
        return _ACTION_PRIORITY.get(option.metadata.get("type"), 0)

    def _action_score(self, state: GameState, request: InputRequest, option: Any) -> float:
        fallback = float(self._action_priority(option))
        action = option.metadata.get("type")
        if action not in (ActionType.MOVEMENT, ActionType.FAST_TRAVEL, ActionType.ATTACK):
            return fallback

        try:
            actor_id = str(state.current_actor_id or request.player_id)
            board_actor_id = state.resolve_board_actor(actor_id)
            current = state.get_position(board_actor_id)
            actor = state.get_unit(UnitID(board_actor_id))
            hero = state.get_hero(HeroID(actor_id))
        except (AttributeError, KeyError, TypeError, ValueError):
            return fallback
        if (
            current is None
            or actor is None
            or hero is None
            or hero.is_multi_piece
            or hero.current_turn_card is None
        ):
            return fallback

        if action == ActionType.ATTACK:
            return self._basic_attack_score(state, option, actor_id, board_actor_id, fallback)

        team = actor.team
        if team is None:
            return fallback
        current_score = self._position_score(state, current, team)
        if current_score is None:
            return fallback

        if action == ActionType.MOVEMENT:
            movement = _strict_nonnegative_int(option.metadata.get("value"))
            if movement is None:
                return fallback
            try:
                destinations = find_reachable_hexes(
                    board=state.board,
                    start=current,
                    max_steps=movement,
                    state=state,
                    actor_id=board_actor_id,
                    topology_unit_ids=[board_actor_id],
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                return fallback
        else:
            destination_filter = FastTravelDestinationFilter(unit_id=board_actor_id)
            context = state.execution_context
            try:
                destinations = [
                    candidate
                    for candidate in state.board.tiles
                    if destination_filter.apply(candidate, state, context)
                ]
            except (AttributeError, KeyError, TypeError, ValueError):
                return fallback

        scores = [self._position_score(state, destination, team) for destination in destinations]
        legal_scores = [score for score in scores if score is not None]
        if not legal_scores:
            return fallback
        gain = max(legal_scores) - current_score
        positional_score = _POSITIONAL_ACTION_BASELINE + gain / _POSITIONAL_GAIN_SCALE
        return min(positional_score, _POSITIONAL_ACTION_CEILING)

    def _basic_attack_score(
        self,
        state: GameState,
        option: Any,
        actor_id: str,
        board_actor_id: str,
        fallback: float,
    ) -> float:
        hero = state.get_hero(HeroID(actor_id))
        card = hero.current_turn_card if hero else None
        attack_value = _strict_nonnegative_int(option.metadata.get("value"))
        if card is None or attack_value is None:
            return fallback
        is_primary_attack = card.current_primary_action == ActionType.ATTACK
        if is_primary_attack and (card.current_effect_id or card.effect_id):
            return fallback
        if not is_primary_attack and ActionType.ATTACK not in card.current_secondary_actions:
            return fallback

        source = state.get_unit(UnitID(board_actor_id))
        if source is None or source.team is None:
            return fallback
        try:
            range_value = compute_card_stats(state, UnitID(actor_id), card).range
        except (AttributeError, KeyError, TypeError, ValueError):
            return fallback

        target_scores: list[float] = []
        for color, enemy_team in state.teams.items():
            if color == source.team:
                continue
            for unit in [*enemy_team.heroes, *enemy_team.minions]:
                for target_id in state.get_piece_ids(str(unit.id)) or [str(unit.id)]:
                    target = state.get_unit(UnitID(target_id))
                    if target is not None and validate_target(
                        source, target, ActionType.ATTACK, state, range_value
                    ):
                        target_scores.append(self._unit_id_score(state, target_id, source.team))

        if not target_scores:
            return float(_ACTION_PRIORITY[ActionType.HOLD])
        return fallback + attack_value / 10.0 + max(target_scores) / 10.0

    def _acting_team(self, state: GameState, request: InputRequest) -> TeamColor:
        for uid in (request.player_id, state.current_actor_id):
            if not uid:
                continue
            hero = state.get_hero(HeroID(str(uid)))
            if hero is not None and hero.team is not None:
                return hero.team
        return TeamColor.RED

    def _unit_score(self, state: GameState, option: Any, team: TeamColor) -> float:
        return self._unit_id_score(state, option.id, team)

    def _unit_id_score(self, state: GameState, uid: str, team: TeamColor) -> float:
        ut = self._unit_team(state, uid)
        if ut is None:
            return 0.0
        enemy = ut != team
        if not enemy:
            return -5.0
        unit = state.get_unit(UnitID(uid))
        hero_target = unit is not None and is_hero_unit(unit)
        base = 10.0 if hero_target else 5.0
        positions = state.get_positions(uid)
        in_battle = any(
            loc in state.board.zones[z].hexes
            for loc in positions
            for z in state.battle_zones.values()
        )
        score = base + (2.0 if in_battle else 0.0)
        if hero_target:
            owner_id = state.hero_owner_id(uid)
            if owner_id in state.unresolved_hero_ids:
                score += 2.0
            score -= float(calculate_minion_defense_modifier(state, UnitID(uid)))
        return score

    def _hex_score(self, state: GameState, option: Any, team: TeamColor) -> float:
        hexd = option.metadata.get("hex")
        if hexd is None:
            return 0.0
        try:
            destination = Hex.model_validate(hexd)
        except (TypeError, ValueError):
            return 0.0
        return self._position_score(state, destination, team) or 0.0

    def _position_score(
        self, state: GameState, destination: Hex, team: TeamColor
    ) -> float | None:
        zid = self._zone_of(state, destination)
        if zid is None:
            return None
        # Coarse push signal: how many zones this hex is toward the enemy throne.
        # Zone-granular (0..~3 on a 5-zone lane) — the *strategic* signal.
        lane_id = next(iter(state.battle_zones), None)
        toward_enemy = zones_between(state, team, lane_id, zid) if lane_id else 0

        # Intra-zone placement gradient: prefer landing closer to the nearest
        # enemy so the agent closes distance to fight instead of stalling. This
        # matters for BOTH ordinary movement and fast travel — fast travel picks
        # a better zone (and may free up movement steps), but *placement within*
        # that zone is still critical, so we never discard this term. Raw cube
        # distance ignores terrain, so it stays a sub-unit tie-breaker: the
        # zone-push term (x10) dominates and it never trades a better zone for a
        # few hexes. Adjacent -> ~1.0 pull; distances of ten or more -> no pull.
        enemies = self._enemy_positions(state, team)
        approach = 0.0
        if enemies:
            topology = get_topology_service()
            nearest = min(topology.distance(destination, e, state) for e in enemies)
            approach = max(0.0, 1.0 - nearest / 10.0)

        return 10.0 * float(toward_enemy) + approach

    def _choose_upgrade(self, request: InputRequest) -> Any:
        players = request.context.get("players", {})
        for hid, info in players.items():
            if info.get("remaining", 0) > 0 and info.get("options"):
                # Prefer a group containing an attack card; else first group.
                groups = info["options"]
                group = next(
                    (
                        g
                        for g in groups
                        if any(
                            d.get("primary_action") == ActionType.ATTACK
                            for d in g.get("card_details", [])
                        )
                    ),
                    groups[0],
                )
                pair = group.get("pair") or [d["id"] for d in group.get("card_details", [])]
                return {"hero_id": hid, "card_id": pair[0]}
        return None


def _as_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _strict_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
