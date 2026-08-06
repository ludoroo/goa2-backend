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

from typing import Any

from goa2.domain.input import InputRequest, selection_value
from goa2.domain.models import ActionType, TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.map_logic import zones_between

from .base import HexLike, hex_distance

# CHOOSE_ACTION priority (higher = preferred as the played action).
_ACTION_PRIORITY = {
    ActionType.ATTACK: 5,
    ActionType.SKILL: 4,
    ActionType.MOVEMENT: 3,
    ActionType.FAST_TRAVEL: 2,
    ActionType.DEFENSE: 2,
    ActionType.CLEAR: 1,
    ActionType.HOLD: 0,
}


class HeuristicAgent:
    def __init__(self, seed: int = 0) -> None:
        import random

        self._rng = random.Random(seed)

    # --- helpers -----------------------------------------------------------
    def _enemy_positions(self, state: GameState, team: TeamColor) -> list[Any]:
        enemy = TeamColor.BLUE if team == TeamColor.RED else TeamColor.RED
        out = []
        for unit in [*state.teams[enemy].heroes, *state.teams[enemy].minions]:
            loc = state.unit_locations.get(unit.id)
            if loc is not None:
                out.append(loc)
        return out

    def _unit_team(self, state: GameState, uid: str) -> TeamColor | None:
        for color, team in state.teams.items():
            if any(u.id == uid for u in [*team.heroes, *team.minions]):
                return color
        return None

    def _is_hero(self, state: GameState, uid: str) -> bool:
        return any(h.id == uid for t in state.teams.values() for h in t.heroes)

    def _zone_of(self, state: GameState, loc: Any) -> str | None:
        for zid, zone in state.board.zones.items():
            if loc in zone.hexes:
                return zid
        return None

    # --- planning ----------------------------------------------------------
    def choose_card(self, state: GameState, hero: Hero) -> Card | None:
        if not hero.hand:
            return None
        # Highest score; break ties by initiative (act earlier), then rng.
        best = max(
            hero.hand,
            key=lambda c: (self.score_card(state, hero, c), c.initiative, self._rng.random()),
        )
        return best

    def score_card(self, state: GameState, hero: Hero, card: Card) -> float:
        """Static desirability of committing ``card`` for ``hero`` (higher = better).

        Public so the search layer can reuse it as an expansion prior. Pure and
        side-effect free; does not consult the RNG (callers break ties).
        """
        team = hero.team or TeamColor.RED
        pos = state.unit_locations.get(hero.id)
        enemies = self._enemy_positions(state, team)
        nearest = min((hex_distance(pos, e) for e in enemies), default=99) if pos else 99

        pa = card.primary_action
        val = card.primary_action_value or 0
        if pa == ActionType.ATTACK:
            reach = card.range_value or 1
            reachable = nearest <= reach
            return 10 + val + (5 if reachable else -3)
        if pa == ActionType.SKILL:
            return 6
        if pa == ActionType.MOVEMENT:
            return 5 + (3 if nearest > 2 else 0)
        if pa == ActionType.DEFENSE:
            return 4
        return 3

    # --- resolution --------------------------------------------------------
    def choose_input(
        self,
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
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
            return float(self._action_priority(option))

        if rt in ("SELECT_UNIT", "SELECT_ENEMY", "SELECT_UNIT_OR_TOKEN"):
            return self._unit_score(state, option, self._acting_team(state, request))

        if rt in ("SELECT_HEX", "MOVEMENT_HEX", "FAST_TRAVEL_DESTINATION", "CHOOSE_RESPAWN_HEX"):
            return self._hex_score(state, option, self._acting_team(state, request))

        if rt == "SELECT_NUMBER":
            # More (push/move/repeat) is usually better.
            return float(_as_int(selection_value(option)))

        if rt in ("DEFENSE_CARD", "SELECT_CARD_OR_PASS"):
            # Prefer to defend (survive) rather than skip into defeat.
            return float(_as_int(option.metadata.get("defense", 0)))

        # Default: no preference between concrete options.
        return 0.0


    # --- scoring -----------------------------------------------------------
    def _action_priority(self, option: Any) -> int:
        return _ACTION_PRIORITY.get(option.metadata.get("type"), 0)

    def _acting_team(self, state: GameState, request: InputRequest) -> TeamColor:
        for uid in (request.player_id, state.current_actor_id):
            if not uid:
                continue
            hero = state.get_hero(HeroID(str(uid)))
            if hero is not None and hero.team is not None:
                return hero.team
        return TeamColor.RED

    def _unit_score(self, state: GameState, option: Any, team: TeamColor) -> float:
        uid = option.id
        ut = self._unit_team(state, uid)
        if ut is None:
            return 0.0
        enemy = ut != team
        if not enemy:
            return -5.0
        base = 10.0 if self._is_hero(state, uid) else 5.0
        loc = state.unit_locations.get(uid)
        in_battle = loc is not None and any(
            loc in state.board.zones[z].hexes for z in state.battle_zones.values()
        )
        return base + (2.0 if in_battle else 0.0)

    def _hex_score(self, state: GameState, option: Any, team: TeamColor) -> float:
        hexd = option.metadata.get("hex")
        if hexd is None:
            return 0.0
        zid = self._zone_of(state, HexLike(hexd))
        if zid is None:
            return 0.0
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
        # few hexes. Distance ~10 -> ~0 pull; adjacent -> ~1.0 pull.
        enemies = self._enemy_positions(state, team)
        approach = 0.0
        if enemies:
            nearest = min(hex_distance(hexd, e) for e in enemies)
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
