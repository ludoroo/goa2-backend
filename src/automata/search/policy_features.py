"""Public-information features for legal learned-policy candidates.

The extractor is pure: it does not mutate the state or consume randomness.
Only explicitly public fields are read.  In particular, input metadata is
allowlisted rather than traversed, so an engine prompt cannot accidentally
leak an opponent's hidden card identity into training data.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from goa2.domain.input import InputOption, InputRequest, selection_value
from goa2.domain.models import CardState, TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero, HeroPiece, Unit
from goa2.domain.state import GameState
from goa2.domain.types import HeroID, UnitID

from .ismcts import Decision
from .node import Key, action_key

POLICY_FEATURE_SCHEMA_ID = "policy-candidates-v1"

_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_SAFE_CATEGORICAL_METADATA = ("action_type", "type")
_SAFE_NUMERIC_METADATA = (
    "value",
    "number",
    "min",
    "max",
    "minimum",
    "maximum",
    "defense_value",
)


def _token(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    return _TOKEN_RE.sub("_", str(value).lower()).strip("_") or "empty"


def _indicator(features: dict[str, float], name: str, value: object) -> None:
    features[f"{name}.{_token(value)}"] = 1.0


def _number(features: dict[str, float], name: str, value: object) -> None:
    # bool is categorical rather than an accidental integer feature.
    if isinstance(value, bool):
        features[name] = 1.0 if value else 0.0
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        features[name] = float(value)


def _hero_team(state: GameState, hero_id: str) -> TeamColor | None:
    hero = state.get_hero(HeroID(hero_id))
    return hero.team if hero is not None else None


@dataclass(frozen=True, slots=True)
class _Perspective:
    team: TeamColor
    viewer_hero_id: str | None


def _perspective(state: GameState, decision: Decision) -> _Perspective:
    if decision.kind == "CARD" and decision.hero is not None:
        team = decision.hero.team
        if team is not None and team in state.teams:
            return _Perspective(team, str(decision.hero.id))

    if decision.kind == "INPUT" and decision.request is not None:
        player_id = decision.request.player_id
        if player_id.startswith("team:"):
            team_name = player_id.partition(":")[2]
            team = next(
                (color for color in state.teams if team_name in (color.name, color.value)),
                None,
            )
            if team is not None:
                return _Perspective(team, None)
            raise ValueError(f"Cannot resolve policy perspective for team player {player_id!r}")
        if player_id:
            team = _hero_team(state, player_id)
            if team is not None:
                return _Perspective(team, player_id)
            # An explicit but unknown player must fail closed rather than be
            # silently attributed to whichever actor happens to be current.
            if player_id != "system":
                raise ValueError(f"Cannot resolve policy perspective for player {player_id!r}")

    if state.current_actor_id is not None:
        team = _hero_team(state, str(state.current_actor_id))
        if team is not None:
            return _Perspective(team, str(state.current_actor_id))
    raise ValueError("Cannot resolve acting team perspective for policy features")


def _common_features(state: GameState, team: TeamColor) -> dict[str, float]:
    # Keep the evaluation feature stack out of search-only import paths.
    from automata.evaluation.features import state_features

    return {
        f"state.rich-v1.{name}": float(value)
        for name, value in state_features(state, team, "rich-v1").items()
    }


def _card_features(card: Card) -> dict[str, float]:
    values: dict[str, float] = {"candidate.card": 1.0}
    _indicator(values, "card.tier", card.tier)
    _indicator(values, "card.color", card.color or "none")
    _indicator(values, "card.primary_action", card.primary_action or "none")
    for action, amount in card.secondary_actions.items():
        _number(values, f"card.secondary_action.{_token(action)}", amount)
    _number(values, "card.initiative", card.initiative)
    _number(values, "card.primary_action_value", card.primary_action_value)
    _number(values, "card.range", card.range_value)
    _number(values, "card.radius", card.radius_value)
    _number(values, "card.is_ranged", card.is_ranged)
    _indicator(values, "card.item", card.item or "none")
    # The acting player can see the cards in their own hand; effect identity is
    # therefore public in CARD decisions (unlike an enemy facedown commit).
    _indicator(values, "card.effect", card.effect_id)
    return values


def _card_owner(state: GameState, card: Card) -> Hero | None:
    for team in state.teams.values():
        for hero in team.heroes:
            if any(candidate is card for candidate in hero.hand):
                return hero
            if any(candidate is card for candidate in hero.deck):
                return hero
            if hero.ultimate_card is card or any(candidate is card for candidate in hero.spells):
                return hero
    return None


def _public_card_features(
    state: GameState, card: Card, perspective: _Perspective
) -> dict[str, float]:
    owner = _card_owner(state, card)
    is_viewers = owner is not None and str(owner.id) == perspective.viewer_hero_id
    is_public = (
        owner is not None
        and not card.is_facedown
        and card.state in {CardState.UNRESOLVED, CardState.RESOLVED, CardState.DISCARD}
    )
    if not is_viewers and not is_public:
        # Team membership does not confer hand visibility. This also fails safe
        # for team-scoped requests, which intentionally have no individual viewer.
        return {"candidate.card": 1.0, "card.hidden": 1.0}
    values = _card_features(card)
    _indicator(values, "card.public_identity", card.id)
    if is_viewers:
        values["card.viewer_owned"] = 1.0
    return values


def _unit_features(state: GameState, unit: Unit, perspective: _Perspective) -> dict[str, float]:
    values: dict[str, float] = {"candidate.unit": 1.0}
    _indicator(values, "unit.kind", unit.__class__.__name__)
    _indicator(values, "unit.public_identity", unit.id)
    _indicator(values, "unit.team", unit.team or "none")
    if unit.team is not None:
        _indicator(values, "unit.relation", "own" if unit.team == perspective.team else "enemy")
    if isinstance(unit, Hero):
        _number(values, "unit.level", unit.level)
        _number(values, "unit.gold", unit.gold)
    elif isinstance(unit, HeroPiece):
        _indicator(values, "unit.owner", unit.owner_hero_id)
    position = state.get_position(str(unit.id))
    if position is not None:
        _number(values, "unit.hex.q", position.q)
        _number(values, "unit.hex.r", position.r)
        _number(values, "unit.hex.s", position.s)
    return values


def _input_candidate_features(
    state: GameState,
    request: InputRequest,
    option: InputOption | None,
    key: Key,
    perspective: _Perspective,
) -> dict[str, float]:
    values: dict[str, float] = {}
    _indicator(values, "input.request_type", request.request_type)
    _indicator(values, "candidate.key_kind", type(key).__name__)
    if option is None:
        values["candidate.skip"] = 1.0
        return values

    raw = selection_value(option)
    if isinstance(raw, dict) and {"q", "r"} <= raw.keys():
        values["candidate.hex"] = 1.0
        for coordinate in ("q", "r", "s"):
            _number(values, f"hex.{coordinate}", raw.get(coordinate))
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        values["candidate.number"] = 1.0
        _number(values, "number.value", raw)
    else:
        values["candidate.option"] = 1.0

    card = state.get_card_by_id(str(raw)) if isinstance(raw, str) else None
    if card is not None:
        values.update(_public_card_features(state, card, perspective))
    else:
        unit = state.get_unit(UnitID(str(raw))) if isinstance(raw, str) else None
        if unit is not None:
            values.update(_unit_features(state, unit, perspective))
        elif isinstance(raw, str) and "CARD" not in request.request_type.value:
            # Option/action IDs are part of the public prompt. Card-like prompts
            # are excluded because an ID may itself reveal a hidden enemy card.
            _indicator(values, "option.public_id", raw)

    # Metadata is deliberately allowlisted; never iterate arbitrary metadata.
    for name in _SAFE_CATEGORICAL_METADATA:
        if name in option.metadata:
            _indicator(values, f"option.metadata.{name}", option.metadata[name])
    for name in _SAFE_NUMERIC_METADATA:
        if name in option.metadata:
            _number(values, f"option.metadata.{name}", option.metadata[name])
    return values


def policy_candidate_features(
    state: GameState, decision: Decision, legal: Sequence[Key]
) -> dict[Key, dict[str, float]]:
    """Return deterministic sparse features for every legal candidate.

    ``legal`` controls only outer mapping insertion order. Candidate mappings
    are built independently and fail closed when they cannot be reconciled with
    the live CARD hand or INPUT options.
    """
    perspective = _perspective(state, decision)
    common = _common_features(state, perspective.team)
    result: dict[Key, dict[str, float]] = {}

    if decision.kind == "CARD" and decision.hero is not None:
        cards = {card.id: card for card in decision.hero.hand}
        for key in legal:
            if key is None:
                candidate = {"candidate.finish": 1.0}
            elif isinstance(key, str) and key in cards:
                candidate = _public_card_features(state, cards[key], perspective)
            else:
                raise ValueError(f"Legal CARD candidate {key!r} cannot be reconciled")
            result[key] = common | candidate
        return result

    if decision.kind == "INPUT" and decision.request is not None:
        options: dict[Key, InputOption | None] = {}
        # Deliberately stricter than ``_input_raw_map``: duplicate normalized
        # keys are ambiguous for training rows, so fail closed rather than use
        # that search helper's last-option-wins behavior.
        for option in decision.request.options:
            key = action_key(selection_value(option))
            if key in options:
                raise ValueError(f"Duplicate INPUT option key {key!r} is ambiguous")
            options[key] = option
        if decision.request.can_skip:
            if "SKIP" in options:
                raise ValueError("Duplicate INPUT option key 'SKIP' is ambiguous")
            options["SKIP"] = None
        for key in legal:
            if key not in options:
                raise ValueError(f"Legal INPUT key {key!r} cannot be reconciled to an option")
            candidate = _input_candidate_features(
                state, decision.request, options[key], key, perspective
            )
            result[key] = common | candidate
        return result

    raise ValueError(f"Unsupported policy decision kind {decision.kind!r}")


__all__ = ["POLICY_FEATURE_SCHEMA_ID", "policy_candidate_features"]
