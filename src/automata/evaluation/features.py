"""State evaluation features for heuristic (and later, learned) agents.

`state_features(state, team)` returns a named feature vector (differentials
from ``team``'s perspective); `evaluate_state(state, team)` is the hand-weighted
dot product of that vector — positive = good for ``team``. Splitting the two
lets the same features feed a *learned* value/policy later (Rungs 2-3) without
recomputing anything, while keeping today's behavior byte-identical.

It reuses the engine's own lane/push helpers so "winning the push" matches the
rules exactly.

Signals (rulebook win conditions first):
- Life-counter differential  — a team loses at 0 (hero-kill race).
- Push progress differential — zones each team has pushed the battle toward the
  enemy throne (`endgame_totals`); the other win condition.
- Battle-zone minion control — who is winning the current minion battle.
- Tempo — hero levels, gold, and how many heroes are alive on the board.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from goa2.domain.models import CardState, TeamColor
from goa2.domain.state import GameState
from goa2.engine.map_logic import endgame_totals

# Terminal sentinel; dwarfs positional signal so wins/losses dominate.
WIN_SCORE = 1_000_000.0

# Feature weights (hand-tuned; life and push are the win conditions). Keyed by
# feature name so weights and features stay aligned as the set grows.
FEATURE_WEIGHTS: dict[str, float] = {
    "life_diff": 100.0,
    "push_diff": 60.0,
    "minion_diff": 8.0,
    "level_diff": 5.0,
    "alive_diff": 15.0,
    "gold_diff": 1.0,
}

# Stable ordering for callers that want a plain vector (e.g. learned models).
FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_WEIGHTS.keys())

RICH_FEATURE_NAMES: tuple[str, ...] = (
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


@dataclass(frozen=True)
class FeatureSchema:
    """A stable feature-name order paired with its extractor."""

    feature_names: tuple[str, ...]
    extractor: Callable[[GameState, TeamColor], dict[str, float]]


def _enemy(team: TeamColor) -> TeamColor:
    return TeamColor.BLUE if team == TeamColor.RED else TeamColor.RED


def _battle_zone_minion_counts(state: GameState) -> dict[TeamColor, int]:
    """Minions each team currently has standing in a battle zone."""
    counts = {TeamColor.RED: 0, TeamColor.BLUE: 0}
    battle_hexes = set()
    for zone_id in state.battle_zones.values():
        zone = state.board.zones.get(zone_id)
        if zone:
            battle_hexes.update(zone.hexes)
    for color, team in state.teams.items():
        for minion in team.minions:
            loc = state.unit_locations.get(minion.id)
            if loc is not None and loc in battle_hexes:
                counts[color] = counts.get(color, 0) + 1
    return counts


def _heroes_alive(state: GameState, team: TeamColor) -> int:
    return sum(1 for h in state.teams[team].heroes if state.has_board_presence(h.id))


def _base_features(state: GameState, team: TeamColor) -> dict[str, float]:
    """Named feature differentials for ``state`` from ``team``'s perspective.

    Non-terminal only — terminal states are handled by ``evaluate_state`` (and,
    for learning, by the recorded game outcome). Every value is an *own minus
    enemy* differential so the sign already encodes "good for us".
    """
    enemy = _enemy(team)

    my_life = state.teams[team].life_counters
    en_life = state.teams[enemy].life_counters

    push = endgame_totals(state)  # {team: zones between own throne and battle zones}
    minions = _battle_zone_minion_counts(state)

    my_level = sum(h.level for h in state.teams[team].heroes)
    en_level = sum(h.level for h in state.teams[enemy].heroes)
    my_gold = sum(h.gold for h in state.teams[team].heroes)
    en_gold = sum(h.gold for h in state.teams[enemy].heroes)

    return {
        "life_diff": float(my_life - en_life),
        "push_diff": float(push.get(team, 0) - push.get(enemy, 0)),
        "minion_diff": float(minions.get(team, 0) - minions.get(enemy, 0)),
        "level_diff": float(my_level - en_level),
        "alive_diff": float(_heroes_alive(state, team) - _heroes_alive(state, enemy)),
        "gold_diff": float(my_gold - en_gold),
    }


def _card_counts(state: GameState, team: TeamColor) -> tuple[int, int, int]:
    heroes = state.teams[team].heroes
    counts = {CardState.HAND: 0, CardState.DISCARD: 0, CardState.RESOLVED: 0}
    played = 0
    for hero in heroes:
        for card in hero.deck:
            if card.state in counts:
                counts[card.state] += 1
            if card.state in (CardState.UNRESOLVED, CardState.RESOLVED):
                played += 1
    return counts[CardState.HAND], counts[CardState.DISCARD], played


def _hero_board_features(state: GameState, team: TeamColor) -> tuple[int, float]:
    battle_zones = state.battle_zone_ids()
    in_battle = 0
    progresses: list[float] = []
    for hero in state.teams[team].heroes:
        positions = state.get_positions(str(hero.id))
        if any(state.board.get_zone_for_hex(pos) in battle_zones for pos in positions):
            in_battle += 1

        piece_progress: list[float] = []
        for position in positions:
            zone_id = state.board.get_zone_for_hex(position)
            if zone_id is None:
                continue
            lane_id = state.lane_of_zone(zone_id)
            lane = state.board.lanes.get(lane_id or "", [])
            if zone_id not in lane or len(lane) < 2:
                continue
            progress = lane.index(zone_id) / (len(lane) - 1)
            if team == TeamColor.BLUE:
                progress = 1.0 - progress
            piece_progress.append(min(1.0, max(0.0, progress)))
        progresses.append(sum(piece_progress) / len(piece_progress) if piece_progress else 0.0)
    return in_battle, (sum(progresses) / len(progresses) if progresses else 0.0)


def _rich_features(state: GameState, team: TeamColor) -> dict[str, float]:
    enemy = _enemy(team)
    push = endgame_totals(state)
    minions = _battle_zone_minion_counts(state)
    own_cards = _card_counts(state, team)
    enemy_cards = _card_counts(state, enemy)
    own_battle, own_progress = _hero_board_features(state, team)
    enemy_battle, enemy_progress = _hero_board_features(state, enemy)
    own = state.teams[team]
    opposing = state.teams[enemy]
    waves = list(state.wave_counters.values())
    return {
        "own_life": float(own.life_counters),
        "enemy_life": float(opposing.life_counters),
        "own_push": float(push.get(team, 0)),
        "enemy_push": float(push.get(enemy, 0)),
        "own_battle_minions": float(minions.get(team, 0)),
        "enemy_battle_minions": float(minions.get(enemy, 0)),
        "own_level_total": float(sum(hero.level for hero in own.heroes)),
        "enemy_level_total": float(sum(hero.level for hero in opposing.heroes)),
        "own_alive_heroes": float(_heroes_alive(state, team)),
        "enemy_alive_heroes": float(_heroes_alive(state, enemy)),
        "own_gold_total": float(sum(hero.gold for hero in own.heroes)),
        "enemy_gold_total": float(sum(hero.gold for hero in opposing.heroes)),
        "round_number": float(state.round),
        "wave_remaining_mean": float(sum(waves) / len(waves) if waves else 0.0),
        "own_hand_cards": float(own_cards[0]),
        "enemy_hand_cards": float(enemy_cards[0]),
        "own_discard_cards": float(own_cards[1]),
        "enemy_discard_cards": float(enemy_cards[1]),
        "own_played_cards": float(own_cards[2]),
        "enemy_played_cards": float(enemy_cards[2]),
        "own_battle_heroes": float(own_battle),
        "enemy_battle_heroes": float(enemy_battle),
        "own_hero_progress_mean": own_progress,
        "enemy_hero_progress_mean": enemy_progress,
    }


FEATURE_SCHEMAS: dict[str, FeatureSchema] = {
    "base-v1": FeatureSchema(FEATURE_NAMES, _base_features),
    "rich-v1": FeatureSchema(RICH_FEATURE_NAMES, _rich_features),
}


def _schema(schema_id: str) -> FeatureSchema:
    try:
        return FEATURE_SCHEMAS[schema_id]
    except KeyError as exc:
        raise ValueError(f"unknown feature schema: {schema_id!r}") from exc


def state_features(
    state: GameState, team: TeamColor, schema_id: str = "base-v1"
) -> dict[str, float]:
    """Named features for the selected versioned schema."""
    return _schema(schema_id).extractor(state, team)


def feature_vector(state: GameState, team: TeamColor, schema_id: str = "base-v1") -> list[float]:
    """``state_features`` as a vector in the selected schema's stable order."""
    schema = _schema(schema_id)
    feats = schema.extractor(state, team)
    return [feats[name] for name in schema.feature_names]


def evaluate_state(state: GameState, team: TeamColor) -> float:
    """Score ``state`` from ``team``'s perspective (higher = better).

    Terminal states dominate; otherwise the hand-weighted dot product of
    ``state_features``.
    """
    if state.winner is not None:
        winner = state.winner.upper() if isinstance(state.winner, str) else state.winner
        won = winner == team.value or winner == team.name
        return WIN_SCORE if won else -WIN_SCORE

    feats = state_features(state, team)
    return sum(FEATURE_WEIGHTS[name] * feats[name] for name in FEATURE_WEIGHTS)
