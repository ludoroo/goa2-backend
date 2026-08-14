"""Recipient-scoped serialization for server messages.

``build_view`` protects the durable game-state snapshot.  Input requests and
events travel beside that snapshot, so they need the same visibility boundary:
only an authorized responder receives a pending request, and event metadata
must not identify cards or facedown tokens hidden from that recipient.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from goa2.domain.events import GameEventType
from goa2.domain.input import InputRequest
from goa2.domain.models import Card, TeamColor, Token
from goa2.domain.state import GameState
from goa2.domain.types import BoardEntityID, HeroID


def _can_answer_input(state: GameState, expected: str, for_hero_id: str | None) -> bool:
    if for_hero_id is None:
        return False
    if expected in (for_hero_id, "simultaneous"):
        return True
    if not expected.startswith("team:"):
        return False
    hero = state.get_hero(HeroID(for_hero_id))
    return bool(hero and hero.team and hero.team.value == expected[5:])


def input_request_for_viewer(
    request: InputRequest | None,
    state: GameState,
    for_hero_id: str | None,
) -> dict[str, Any] | None:
    """Serialize a pending request only for a player allowed to answer it.

    Simultaneous upgrade requests carry a map of every pending player's card
    options internally.  Each client receives only its own entry; players with
    no pending upgrade and spectators receive no request at all.
    """
    if request is None or not _can_answer_input(state, request.player_id, for_hero_id):
        return None

    payload = request.to_dict()
    if request.player_id != "simultaneous":
        return payload

    players = payload.get("players")
    if not isinstance(players, dict):
        return payload
    if for_hero_id not in players:
        return None
    payload["players"] = {for_hero_id: players[for_hero_id]}
    return payload


def awaiting_input_hero_ids(request: InputRequest | None, state: GameState) -> list[str]:
    """Return every hero a pending request is still waiting on.

    Who is being waited on is public even where the request body is not, so
    this travels beside the scoped payload from ``input_request_for_viewer``:
    observers can name the blocking player without seeing their options. A
    simultaneous request drops a hero from ``players`` as soon as they have
    nothing left to answer, so the list shrinks as players finish.
    """
    if request is None:
        return []

    expected = request.player_id
    if expected == "simultaneous":
        players = request.context.get("players")
        return sorted(str(hero_id) for hero_id in players) if isinstance(players, dict) else []
    if expected.startswith("team:"):
        team = state.teams.get(TeamColor(expected[5:]))
        return [str(hero.id) for hero in team.heroes] if team else []
    return [expected]


def _card_location(state: GameState, card_id: str) -> tuple[str, str, Card] | None:
    """Return ``(owner_id, public-zone name, card object)`` for a card ID.

    Container checks intentionally precede the master ``hero.deck`` list.  The
    deck is the normal-card catalogue and also contains cards currently in hand
    or play, so membership in it alone does not describe visibility.
    """
    for team in state.teams.values():
        for hero in team.heroes:
            owner_id = str(hero.id)
            for card in hero.hand:
                if card.id == card_id:
                    return owner_id, "hand", card
            if hero.current_turn_card and hero.current_turn_card.id == card_id:
                return owner_id, "current_turn", hero.current_turn_card
            if hero.extra_turn_card and hero.extra_turn_card.id == card_id:
                return owner_id, "extra_turn", hero.extra_turn_card
            for played_card in hero.played_cards:
                if played_card is not None and played_card.id == card_id:
                    return owner_id, "played", played_card
            for card in hero.discard_pile:
                if card.id == card_id:
                    return owner_id, "discard", card
            if hero.ultimate_card and hero.ultimate_card.id == card_id:
                return owner_id, "ultimate", hero.ultimate_card
            for spell in hero.spellbook:
                if spell.id == card_id:
                    return owner_id, "spellbook", spell
            for spell in hero.cast_spells:
                if spell.id == card_id:
                    return owner_id, "cast_spell", spell
            for card in hero.deck:
                if card.id == card_id:
                    return owner_id, "deck", card
    return None


def _card_visible_to(state: GameState, card_id: str, for_hero_id: str | None) -> bool | None:
    located = _card_location(state, card_id)
    if located is None:
        return None
    owner_id, location, card = located

    if location in {"hand", "deck", "spellbook"}:
        return for_hero_id == owner_id
    if location in {"played", "discard"}:
        # Facedown cards outside the hand are anonymous even to their owner.
        return not card.is_facedown
    if location == "cast_spell":
        return not card.is_facedown
    # Current/extra turn cards and ultimates follow the normal owner exception.
    return for_hero_id == owner_id or not card.is_facedown


def _hidden_card_names(state: GameState, for_hero_id: str | None) -> set[str]:
    names: set[str] = set()
    seen: set[str] = set()
    for team in state.teams.values():
        for hero in team.heroes:
            cards: list[Card | None] = [
                *hero.hand,
                *hero.deck,
                *hero.played_cards,
                *hero.discard_pile,
                *hero.spellbook,
                *hero.cast_spells,
                *hero.spells,
            ]
            cards.extend(
                card for card in (hero.current_turn_card, hero.extra_turn_card) if card is not None
            )
            if hero.ultimate_card:
                cards.append(hero.ultimate_card)
            for card in cards:
                if card is None:
                    continue
                if card.id in seen:
                    continue
                seen.add(card.id)
                if _card_visible_to(state, card.id, for_hero_id) is False:
                    names.add(card.name)
    return names


def _sanitize_card_values(
    value: Any,
    *,
    state: GameState,
    for_hero_id: str | None,
    hidden_names: set[str],
    key: str = "",
) -> Any:
    if isinstance(value, dict):
        return {
            child_key: _sanitize_card_values(
                child_value,
                state=state,
                for_hero_id=for_hero_id,
                hidden_names=hidden_names,
                key=f"{key}.{child_key}" if key else str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        sanitized = [
            _sanitize_card_values(
                item,
                state=state,
                for_hero_id=for_hero_id,
                hidden_names=hidden_names,
                key=key,
            )
            for item in value
        ]
        # Lists of card/spell IDs must not preserve hidden placeholders/counts.
        if "card" in key or "spell" in key:
            return [item for item in sanitized if item is not None]
        return sanitized
    if not isinstance(value, str):
        return value

    visible = _card_visible_to(state, value, for_hero_id)
    if visible is False:
        return None
    if ("card" in key or "spell" in key) and value in hidden_names:
        return None
    return value


def _token_identity_visible(state: GameState, token: Token, for_hero_id: str | None) -> bool:
    if not token.is_facedown:
        return True
    return bool(for_hero_id is not None and token.owner_id == for_hero_id)


def events_for_viewer(
    events: list[dict[str, Any]],
    state: GameState,
    for_hero_id: str | None,
) -> list[dict[str, Any]]:
    """Return event dictionaries with recipient-private metadata removed."""
    hidden_names = _hidden_card_names(state, for_hero_id)
    projected: list[dict[str, Any]] = []

    for source in events:
        event = deepcopy(source)
        event_type = str(event.get("event_type", ""))

        if event_type == GameEventType.GUESSED_CARD_REVEALED.value:
            # A guess reveal is a deliberate public flip of one hand card, so
            # its identity is public here even though a wrong guess leaves the
            # card in an otherwise-private hand. This exemption is scoped to an
            # event type that only the guess step emits — do not widen it to a
            # generic reveal, which would leak scoped reveals to every viewer.
            projected.append(event)
            continue

        if event_type == GameEventType.CARD_REVEALED.value:
            # An explicit public reveal is another narrow exception to normal
            # hand masking. Only RevealHandCardStep emits this event; unrelated
            # hand cards and ordinary metadata remain recipient-scoped.
            projected.append(event)
            continue

        event["metadata"] = _sanitize_card_values(
            event.get("metadata", {}),
            state=state,
            for_hero_id=for_hero_id,
            hidden_names=hidden_names,
        )

        target_id = event.get("target_id")
        token = (
            state.misc_entities.get(BoardEntityID(str(target_id)))
            if target_id is not None
            else None
        )
        # Placement announces that a generic facedown mine exists, not whether
        # it is a blast or dud. MINE_TRIGGERED deliberately remains a reveal.
        if (
            event_type == GameEventType.TOKEN_PLACED.value
            and isinstance(token, Token)
            and not _token_identity_visible(state, token, for_hero_id)
        ):
            event["metadata"]["token_type"] = "mine"

        projected.append(event)

    return projected
