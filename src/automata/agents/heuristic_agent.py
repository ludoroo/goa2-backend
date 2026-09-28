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
from dataclasses import dataclass, field
from typing import Any

import goa2.engine.stats as stats
from goa2.data.heroes.registry import HeroRegistry
from goa2.domain.card_knowledge import (
    CardKnowledgeStatus,
    PublicCardKnowledge,
    build_public_card_knowledge,
)
from goa2.domain.hex import Hex
from goa2.domain.input import InputRequest, selection_value
from goa2.domain.models import ActionType, CardTier, StatType, TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero, is_hero_unit
from goa2.domain.state import GameState
from goa2.domain.types import HeroID, UnitID
from goa2.engine.filters_hex import FastTravelDestinationFilter
from goa2.engine.map_logic import zones_between
from goa2.engine.phases import initiative_is_reversed
from goa2.engine.rules import (
    can_perform_action_on_card,
    find_reachable_hexes,
    get_safe_zones_for_fast_travel,
    validate_target,
)
from goa2.engine.stats import calculate_minion_defense_modifier, compute_card_stats
from goa2.engine.topology import get_topology_service

from .contracts import PlanningDecision

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
_POSITIONAL_ACTION_CEILING = math.nextafter(float(_ACTION_PRIORITY[ActionType.SKILL]), -math.inf)
_DODGE_RISK_MIN_PENALTY = 2.0
_DODGE_RISK_MAX_PENALTY = 4.0


@dataclass
class _DodgeScoreContext:
    """Public, per-score facts shared by every legal attack target."""

    knowledge: PublicCardKnowledge | None = None
    definitions: dict[str, Hero | None] = field(default_factory=dict)
    loadouts: dict[str, list[list[Card]] | None] = field(default_factory=dict)


def _is_reversed_initiative(state: GameState) -> bool:
    """Backward-compatible local name for the engine's canonical predicate."""
    return initiative_is_reversed(state)


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
    def choose_planning(self, state: GameState, hero: Hero) -> PlanningDecision:
        if not hero.hand:
            return PlanningDecision.pass_()
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
        return PlanningDecision.commit(best)

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

        hero_id = str(hero.id)
        card_stats = compute_card_stats(state, UnitID(hero_id), card)
        actions: list[tuple[ActionType, int, bool]] = []

        primary = card.primary_action
        if primary == ActionType.DEFENSE_SKILL:
            primary = ActionType.SKILL
        if primary is not None and primary != ActionType.DEFENSE:
            actions.append((primary, card_stats.primary_value, True))

        for action, base_value in card.secondary_actions.items():
            if action == ActionType.DEFENSE:
                continue
            value = base_value
            if action in (ActionType.ATTACK, ActionType.MOVEMENT):
                stat_type = StatType.ATTACK if action == ActionType.ATTACK else StatType.MOVEMENT
                value = stats.get_computed_stat(
                    state,
                    hero.id,
                    stat_type,
                    base_value,
                    performing_card=card,
                )
            actions.append((action, value, False))

        scored_actions = []
        legal_attack_targets: list[tuple[str, str]] | None = None
        dodge_context = _DodgeScoreContext()
        for action, value, is_primary in actions:
            if action == ActionType.ATTACK:
                if not can_perform_action_on_card(state, hero_id, action, card):
                    continue
                if legal_attack_targets is None:
                    legal_attack_targets = self._legal_attack_targets(
                        state, hero, card, card_stats.range
                    )
                if not legal_attack_targets:
                    continue
            elif not self._is_card_action_available(state, hero, card, action, card_stats.range):
                continue
            score = self._card_action_score(action, value, nearest)
            if action == ActionType.ATTACK:
                score -= self._attack_dodge_risk_penalty(
                    state,
                    hero,
                    card,
                    card_stats.range,
                    legal_targets=legal_attack_targets,
                    context=dodge_context,
                )
            scored_actions.append((score, is_primary))
        if not scored_actions:
            return 0.0

        best_score = max(score for score, _ in scored_actions)
        if any(is_primary for score, is_primary in scored_actions if score == best_score):
            return best_score
        # Primary card text generally resolves only for primary actions. Encode
        # that preference solely as a tie-break, not as a material heuristic
        # penalty that could hide a genuinely stronger secondary action.
        return math.nextafter(best_score, -math.inf)

    def _card_action_score(self, action: ActionType, value: int, nearest_enemy: float) -> float:
        if action == ActionType.ATTACK:
            return 15.0 + value
        if action == ActionType.SKILL:
            return 6.0
        if action == ActionType.MOVEMENT:
            return 5.0 + (3.0 if nearest_enemy > 2 else 0.0)
        return float(_ACTION_PRIORITY.get(action, 0))

    def _is_card_action_available(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        action: ActionType,
        range_val: int,
    ) -> bool:
        """Mirror proactive action availability, with attack target feasibility."""
        hero_id = str(hero.id)
        if not can_perform_action_on_card(state, hero_id, action, card):
            return False
        if action == ActionType.ATTACK:
            return self._has_attack_target(state, hero, card, range_val)
        if action != ActionType.FAST_TRAVEL:
            return True

        team = hero.team
        if team is None:
            return False
        zone_ids = {
            zone_id
            for position in state.get_positions(hero_id)
            if (zone_id := state.board.get_zone_for_hex(position)) is not None
        }
        return any(get_safe_zones_for_fast_travel(state, team, zone_id) for zone_id in zone_ids)

    def _has_attack_target(self, state: GameState, hero: Hero, card: Card, range_val: int) -> bool:
        """Return whether the canonical target validator finds a public enemy."""
        return bool(self._legal_attack_targets(state, hero, card, range_val))

    def _legal_attack_targets(
        self, state: GameState, hero: Hero, card: Card, range_val: int
    ) -> list[tuple[str, str]]:
        """Return legal ``(attacking piece, target piece)`` pairs from public board state."""
        team = hero.team or TeamColor.RED
        source_ids = state.get_piece_ids(str(hero.id)) or [str(hero.id)]
        target_ids = [
            piece_id
            for other_color, other_team in state.teams.items()
            if other_color != team
            for unit in [*other_team.heroes, *other_team.minions]
            for piece_id in (state.get_piece_ids(str(unit.id)) or [str(unit.id)])
        ]
        legal: list[tuple[str, str]] = []
        for source_id in source_ids:
            if not can_perform_action_on_card(state, source_id, ActionType.ATTACK, card):
                continue
            source = state.get_unit(UnitID(source_id))
            if source is None:
                continue
            for target_id in target_ids:
                target = state.get_unit(UnitID(target_id))
                if target is not None and validate_target(
                    source, target, ActionType.ATTACK, state, range_val
                ):
                    legal.append((source_id, target_id))
        return legal

    def _attack_dodge_risk_penalty(
        self,
        state: GameState,
        hero: Hero,
        card: Card,
        range_val: int,
        *,
        legal_targets: list[tuple[str, str]] | None = None,
        context: _DodgeScoreContext | None = None,
    ) -> float:
        """Estimate public initiative/movement risk for the safest legal target.

        The estimate deliberately reconstructs possible enemy cards from the
        static registry and public card knowledge. It never reads an enemy hand,
        facedown commitment, pending input, or private deck state.
        """
        targets = (
            legal_targets
            if legal_targets is not None
            else self._legal_attack_targets(state, hero, card, range_val)
        )
        context = context or _DodgeScoreContext()
        safest = _DODGE_RISK_MAX_PENALTY
        found_target = False
        for source_id, target_id in targets:
            found_target = True
            penalty = self._target_dodge_risk_penalty(
                state,
                hero,
                card,
                source_id,
                target_id,
                range_val,
                context=context,
            )
            if penalty == 0:
                return 0.0
            safest = min(safest, penalty)
        return safest if found_target else 0.0

    def _target_dodge_risk_penalty(
        self,
        state: GameState,
        attacker: Hero,
        attack_card: Card,
        source_id: str,
        target_id: str,
        attack_range: int,
        *,
        context: _DodgeScoreContext | None = None,
    ) -> float:
        target = state.get_unit(UnitID(target_id))
        if target is None or not is_hero_unit(target):
            return 0.0

        target_owner_id = state.hero_owner_id(target_id)
        target_hero = state.get_hero(HeroID(target_owner_id))
        if target_hero is None:
            return 0.0

        loadouts = self._public_target_loadouts(
            state,
            viewer_id=str(attacker.id),
            target=target_hero,
            context=context,
        )
        if loadouts is None:
            # Static knowledge should exist for production heroes. If a custom
            # or nonstandard definition cannot be reconstructed, retain a small
            # uncertainty adjustment rather than consulting private lifecycle.
            return _DODGE_RISK_MIN_PENALTY

        attack_initiative = stats.get_computed_stat(
            state,
            attacker.id,
            StatType.INITIATIVE,
            attack_card.get_base_stat_value(StatType.INITIATIVE),
            performing_card=attack_card,
        )
        largest_risky_set = 0
        risk_by_card_id: dict[str, bool] = {}
        for loadout in loadouts:
            risky_count = 0
            for candidate in loadout:
                candidate_id = str(candidate.id)
                if candidate_id not in risk_by_card_id:
                    risk_by_card_id[candidate_id] = self._card_can_dodge_before_attack(
                        state,
                        attacker,
                        attack_initiative,
                        candidate,
                        source_id,
                        target_id,
                        target_hero,
                        attack_range,
                    )
                risky_count += risk_by_card_id[candidate_id]
            largest_risky_set = max(largest_risky_set, risky_count)
            if largest_risky_set >= 3:
                return _DODGE_RISK_MAX_PENALTY

        if largest_risky_set == 0:
            return 0.0
        return min(
            _DODGE_RISK_MAX_PENALTY,
            _DODGE_RISK_MIN_PENALTY + largest_risky_set - 1,
        )

    def _public_target_loadouts(
        self,
        state: GameState,
        viewer_id: str,
        target: Hero,
        *,
        context: _DodgeScoreContext | None = None,
    ) -> list[list[Card]] | None:
        """Reconstruct plausible remaining loadouts without private card zones.

        Definition cards are a per-call registry snapshot, normalized once and
        then treated as read-only. This avoids another deep copy for every
        candidate and target.
        """
        context = context or _DodgeScoreContext()
        target_id = str(target.id)
        if target_id in context.loadouts:
            return context.loadouts[target_id]
        if target.name not in context.definitions:
            definition_snapshot = HeroRegistry.get(target.name)
            if definition_snapshot is not None:
                for candidate in definition_snapshot.deck:
                    candidate.is_facedown = False
            context.definitions[target.name] = definition_snapshot
        definition = context.definitions[target.name]
        if definition is None:
            context.loadouts[target_id] = None
            return None
        cards_by_id = {str(card.id): card for card in definition.deck}
        if context.knowledge is None:
            context.knowledge = build_public_card_knowledge(state, viewer_id)
        knowledge = context.knowledge.heroes.get(target_id)
        if knowledge is None:
            context.loadouts[target_id] = None
            return None

        if knowledge.status == CardKnowledgeStatus.INFERRED:
            hypotheses = knowledge.loadout_hypotheses
        elif (
            knowledge.status == CardKnowledgeStatus.EXACT
            and knowledge.active_upgraded_card_ids is not None
        ):
            from goa2.domain.card_knowledge import LoadoutHypothesis

            hypotheses = (LoadoutHypothesis(active_card_ids=knowledge.active_upgraded_card_ids),)
        else:
            # UNAVAILABLE/INCONSISTENT means public facts cannot support a
            # plausible loadout. Let the caller apply bounded minimum
            # uncertainty instead of treating every definition card as active.
            context.loadouts[target_id] = None
            return None

        unavailable = self._publicly_unavailable_card_ids(target)
        loadouts: list[list[Card]] = []
        for hypothesis in hypotheses:
            active_ids = set(knowledge.starting_card_ids)
            for upgraded_id in hypothesis.active_card_ids:
                upgraded = cards_by_id.get(str(upgraded_id))
                if upgraded is None:
                    continue
                active_ids = {
                    card_id
                    for card_id in active_ids
                    if not (
                        (starting := cards_by_id.get(card_id)) is not None
                        and starting.tier == CardTier.I
                        and starting.color == upgraded.color
                    )
                }
                active_ids.add(str(upgraded_id))
            loadouts.append(
                [
                    cards_by_id[card_id]
                    for card_id in active_ids - unavailable
                    if card_id in cards_by_id
                ]
            )
        context.loadouts[target_id] = loadouts
        return loadouts

    @staticmethod
    def _publicly_unavailable_card_ids(hero: Hero) -> set[str]:
        """Use only faceup table facts to exclude cards already spent this round."""
        visible_cards = [
            *hero.played_cards,
            *hero.discard_pile,
            hero.current_turn_card,
            hero.extra_turn_card,
        ]
        return {str(card.id) for card in visible_cards if card is not None and not card.is_facedown}

    def _card_can_dodge_before_attack(
        self,
        state: GameState,
        attacker: Hero,
        attack_initiative: int,
        candidate: Card,
        source_id: str,
        target_id: str,
        target_hero: Hero,
        attack_range: int,
    ) -> bool:
        """Model escape through generic MOVEMENT actions only.

        Special movement or placement granted by individual card text is not
        interpreted here; doing so would require effect-specific simulation.
        """
        movement_base = None
        if candidate.primary_action == ActionType.MOVEMENT:
            movement_base = candidate.primary_action_value or 0
        elif ActionType.MOVEMENT in candidate.secondary_actions:
            movement_base = candidate.secondary_actions[ActionType.MOVEMENT]
        if movement_base is None:
            return False

        candidate_initiative = stats.get_computed_stat(
            state,
            target_hero.id,
            StatType.INITIATIVE,
            candidate.get_base_stat_value(StatType.INITIATIVE),
            performing_card=candidate,
        )
        if not self._initiative_resolves_before(
            state, target_hero, candidate_initiative, attacker, attack_initiative
        ):
            return False
        if not can_perform_action_on_card(state, target_id, ActionType.MOVEMENT, candidate):
            return False

        movement = stats.get_computed_stat(
            state,
            UnitID(target_id),
            StatType.MOVEMENT,
            movement_base,
            performing_card=candidate,
        )
        while (
            movement > 0
            and not state.validator.can_move(
                state,
                target_id,
                movement,
                context={"card": candidate},
                is_movement_action=True,
            ).allowed
        ):
            movement -= 1
        if movement <= 0:
            return False

        start = state.get_position(target_id)
        source_position = state.get_position(source_id)
        if start is None or source_position is None:
            return False
        destinations = find_reachable_hexes(
            state.board,
            start,
            movement,
            state=state,
            actor_id=target_id,
        )
        topology = get_topology_service()
        return any(
            topology.distance(
                source_position,
                destination,
                state,
                unit_ids=[source_id, target_id],
            )
            > attack_range
            for destination in destinations
        )

    @staticmethod
    def _initiative_resolves_before(
        state: GameState,
        first_hero: Hero,
        first_initiative: int,
        second_hero: Hero,
        second_initiative: int,
    ) -> bool:
        if first_initiative != second_initiative:
            if initiative_is_reversed(state):
                return first_initiative < second_initiative
            return first_initiative > second_initiative
        return (
            first_hero.team == state.tie_breaker_team and second_hero.team != state.tie_breaker_team
        )

    # --- resolution --------------------------------------------------------
    def choose_input(
        self,
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
        decision_owner_hero_id: str | None = None,
    ) -> Any:
        # The driver passes ownership uniformly; heuristic selection does not use it.
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
                destinations = list(
                    find_reachable_hexes(
                        board=state.board,
                        start=current,
                        max_steps=movement,
                        state=state,
                        actor_id=board_actor_id,
                        topology_unit_ids=[board_actor_id],
                    )
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

    def _position_score(self, state: GameState, destination: Hex, team: TeamColor) -> float | None:
        zid = self._zone_of(state, destination)
        if zid is None:
            return None
        # Coarse push signal: how many zones this hex is toward the enemy throne.
        # Zone-granular (0..~3 on a 5-zone lane) — the *strategic* signal.
        lane_id = state.lane_of_zone(zid)
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
