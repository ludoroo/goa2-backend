"""Encode trusted engine decisions as learned-model observations."""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Sequence
from typing import Any

from automata.decision import DecisionDescriptor
from automata.models.contracts import (
    ActionCandidateID,
    CandidateID,
    CardCandidateID,
    DecisionObservation,
    EncodedCandidate,
    EntityCandidateID,
    FinishCandidateID,
    HexCandidateID,
    LearnedObservation,
    NumberCandidateID,
    OptionCandidateID,
    SkipCandidateID,
    UnitCandidateID,
    Viewer,
)
from automata.search.contracts import SearchContext
from goa2.domain.input import InputOption, InputRequestType, selection_value
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

from .graph.encoder import encode_snapshot
from .projector import project_snapshot

_CARD = frozenset(
    {InputRequestType.DEFENSE_CARD, InputRequestType.UPGRADE_CHOICE, InputRequestType.SELECT_CARD}
)
_UNIT = frozenset(
    {
        InputRequestType.TIE_BREAKER,
        InputRequestType.SELECT_ALLY,
        InputRequestType.SELECT_ENEMY,
        InputRequestType.SELECT_UNIT,
        InputRequestType.CHOOSE_ACTOR,
    }
)
_HEX = frozenset(
    {
        InputRequestType.MOVEMENT_HEX,
        InputRequestType.FAST_TRAVEL_DESTINATION,
        InputRequestType.SELECT_HEX,
        InputRequestType.CHOOSE_RESPAWN_HEX,
    }
)
_ACTION = frozenset(
    {
        InputRequestType.ACTION_CHOICE,
        InputRequestType.CHOOSE_ACTION,
        InputRequestType.CHOOSE_RESPAWN,
    }
)
_OPTION = frozenset({InputRequestType.SELECT_OPTION, InputRequestType.CONFIRM_PASSIVE})
_UNSUPPORTED = frozenset({InputRequestType.NONE, InputRequestType.UPGRADE_PHASE})


def _stable(value: object) -> str:
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("legal candidate is not finite JSON-compatible data") from exc


def _hex(option: InputOption) -> dict[str, int]:
    prefix = "hex_"
    if not option.id.startswith(prefix):
        raise ValueError("hex candidate option has no coordinate identity")
    try:
        q, r, s = (int(part) for part in option.id[len(prefix) :].split("_"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid hex candidate coordinates") from exc
    if q + r + s != 0:
        raise ValueError("invalid hex candidate coordinates")
    return {"q": q, "r": r, "s": s}


def _number(option: InputOption) -> int | float:
    try:
        value: int | float = int(option.id)
    except ValueError:
        try:
            value = float(option.id)
        except ValueError as exc:
            raise ValueError("number candidate is not numeric") from exc
    if not math.isfinite(float(value)):
        raise ValueError("number candidate must be finite")
    return value


def _search_key(request_type: InputRequestType, option: InputOption) -> Hashable:
    if request_type in _HEX:
        coordinates = _hex(option)
        return ("hex", coordinates["q"], coordinates["r"], coordinates["s"])
    if request_type == InputRequestType.SELECT_NUMBER:
        # Existing search keys preserve integral values but use the option ID for fractions.
        value = _number(option)
        return value if isinstance(value, int) else option.id
    # Trap: search keys normalize integer-looking IDs, while encoded OPTION and
    # ACTION identities and submitted selections intentionally remain strings.
    if option.id.lstrip("-").isdigit():
        return int(option.id)
    return option.id


def _refs(
    state: LearnedObservation,
) -> tuple[dict[str, str], dict[str, str], dict[tuple[int, int, int], str], dict[str, str]]:
    cards: dict[str, str] = {}
    units: dict[str, str] = {}
    tiles: dict[tuple[int, int, int], str] = {}
    entities: dict[str, str] = {}
    for token in state.tokens:
        if token.kind == "CARD" and isinstance(token.features.get("card_id"), str):
            cards[str(token.features["card_id"])] = token.local_ref
        elif token.kind == "UNIT" and isinstance(token.features.get("entity_id"), str):
            units[str(token.features["entity_id"])] = token.local_ref
        elif token.kind == "HERO" and isinstance(token.features.get("hero_id"), str):
            units.setdefault(str(token.features["hero_id"]), token.local_ref)
        elif token.kind == "TILE":
            coordinates = (
                token.features.get("q"),
                token.features.get("r"),
                token.features.get("s"),
            )
            if not all(
                isinstance(value, int) and not isinstance(value, bool) for value in coordinates
            ):
                raise ValueError("tile graph token has invalid coordinates")
            q, r, s = coordinates
            assert isinstance(q, int) and isinstance(r, int) and isinstance(s, int)
            tiles[(q, r, s)] = token.local_ref
        elif token.kind in {"TOKEN", "ENTITY"} and isinstance(token.features.get("entity_id"), str):
            entities[str(token.features["entity_id"])] = token.local_ref
    return cards, units, tiles, entities


def _input_candidate(
    request_type: InputRequestType,
    option: InputOption,
    refs: tuple[dict[str, str], dict[str, str], dict[tuple[int, int, int], str], dict[str, str]],
    *,
    card_fallback_ref: str,
    card_fallback_ids: frozenset[str],
) -> EncodedCandidate:
    cards, units, tiles, entities = refs
    selection: Any = selection_value(option)
    target: str | None = None
    identity: CandidateID
    if request_type in _CARD or request_type == InputRequestType.SELECT_CARD_OR_PASS:
        # The trusted legal-option contract may identify a card-like action that
        # visibility intentionally omits from the graph (for example the explicit
        # PASS option in a defense-card request). Keep its exact action identity,
        # but attach it only to the decision owner rather than inventing a card
        # token or copying option metadata into the observation.
        target = cards.get(option.id)
        if target is None:
            if option.id not in card_fallback_ids:
                raise ValueError("card candidate is hidden or has no visible graph reference")
            target = card_fallback_ref
        identity = CardCandidateID(schema_version=1, card_id=option.id)
    elif request_type in _UNIT:
        target = units.get(option.id)
        if target is None:
            raise ValueError("unit candidate has no graph reference")
        identity = UnitCandidateID(schema_version=1, unit_id=option.id)
    elif request_type == InputRequestType.SELECT_UNIT_OR_TOKEN:
        target = units.get(option.id)
        if target is not None:
            identity = UnitCandidateID(schema_version=1, unit_id=option.id)
        else:
            target = entities.get(option.id)
            if target is None:
                raise ValueError("unit or entity candidate has no graph reference")
            identity = EntityCandidateID(schema_version=1, entity_ref=target)
    elif request_type in _HEX:
        selection = _hex(option)
        key = (selection["q"], selection["r"], selection["s"])
        target = tiles.get(key)
        if target is None:
            raise ValueError("hex candidate has no tile graph reference")
        identity = HexCandidateID(schema_version=1, **selection)
    elif request_type == InputRequestType.SELECT_NUMBER:
        selection = _number(option)
        identity = NumberCandidateID(schema_version=1, value=selection)
    elif request_type in _ACTION:
        identity = ActionCandidateID(schema_version=1, action_id=option.id)
    elif request_type in _OPTION:
        identity = OptionCandidateID(schema_version=1, option_id=option.id)
    else:
        raise ValueError(f"unsupported input request candidate policy: {request_type.value}")
    return EncodedCandidate(
        schema_version=1,
        candidate_id=identity,
        selection=selection,
        target_ref=target,
        features={},
    )


def encode_decision(
    state: GameState,
    decision: DecisionDescriptor,
    legal_keys: Sequence[Any],
    *,
    decision_owner_hero_id: str,
    perspective_team: str,
    context: SearchContext | None = None,
) -> DecisionObservation:
    """Project and encode one exact ordered legal decision without hidden metadata."""
    if context is not None:
        decision_owner_hero_id = context.root_viewer_id
        perspective_team = context.perspective_team.value
    if not decision_owner_hero_id:
        raise ValueError("decision owner hero ID is required")
    viewer = Viewer(
        schema_version=2,
        private_hero_id=decision_owner_hero_id,
        perspective_team=perspective_team,
    )
    current_owner_id = context.current_owner_id if context is not None else decision_owner_hero_id
    graph = encode_snapshot(
        project_snapshot(
            state,
            viewer,
            current_owner_id=current_owner_id,
        )
    )
    refs = _refs(graph)
    card_fallback_ref = next(
        (
            token.local_ref
            for token in graph.tokens
            if token.kind == "HERO" and token.features.get("hero_id") == current_owner_id
        ),
        None,
    )
    if card_fallback_ref is None:
        raise ValueError(f"decision owner has no hero graph reference: {current_owner_id!r}")

    if len({_stable(key) for key in legal_keys}) != len(legal_keys):
        raise ValueError("duplicate legal candidates")
    candidates: list[EncodedCandidate] = []
    expected_keys: list[Any]
    if decision.kind == "CARD":
        if decision.hero is None:
            raise ValueError("CARD decision requires an owner")
        hidden_hand_ref = next(
            (
                token.local_ref
                for token in graph.tokens
                if token.kind == "CARD"
                and token.features.get("owner_ref") == f"hero:{decision.hero.id}"
                and token.features.get("area") == "hand"
            ),
            f"hero:{decision.hero.id}",
        )
        expected_keys = [card.id for card in decision.hero.hand]
        if decision.can_finish_planning:
            expected_keys.append(None)
        for key in expected_keys:
            if key is None:
                candidates.append(
                    EncodedCandidate(
                        schema_version=1,
                        candidate_id=FinishCandidateID(schema_version=1),
                        selection=None,
                    )
                )
                continue
            target = refs[0].get(str(key))
            if target is None:
                target = hidden_hand_ref
            candidates.append(
                EncodedCandidate(
                    schema_version=1,
                    candidate_id=CardCandidateID(schema_version=1, card_id=str(key)),
                    selection=str(key),
                    target_ref=target,
                )
            )
    elif decision.kind == "INPUT":
        request = decision.request
        if request is None:
            raise ValueError("INPUT decision requires a request")
        if request.request_type in _UNSUPPORTED:
            raise ValueError(
                f"unsupported non-branchable input request: {request.request_type.value}"
            )
        card_fallback_ids = frozenset(
            option.id
            for option in request.options
            if state.get_card_for_hero(current_owner_id, option.id) is not None
            or (
                request.request_type == InputRequestType.SELECT_CARD_OR_PASS and option.id == "PASS"
            )
        )
        expected_keys = [_search_key(request.request_type, option) for option in request.options]
        candidates = [
            _input_candidate(
                request.request_type,
                option,
                refs,
                card_fallback_ref=card_fallback_ref,
                card_fallback_ids=card_fallback_ids,
            )
            for option in request.options
        ]
        if request.can_skip:
            expected_keys.append("SKIP")
            candidates.append(
                EncodedCandidate(
                    schema_version=1,
                    candidate_id=SkipCandidateID(schema_version=1),
                    selection="SKIP",
                )
            )
    else:
        raise ValueError(f"unsupported decision kind: {decision.kind!r}")

    if [_stable(key) for key in legal_keys] != [_stable(key) for key in expected_keys]:
        raise ValueError("legal candidates do not match request options")
    return DecisionObservation(
        schema_version=3,
        state=graph,
        decision_kind=decision.kind,
        candidates=tuple(candidates),
    )


def encode_search_context(
    context: SearchContext,
    state: GameState,
    legal_keys: Sequence[Any],
) -> DecisionObservation:
    """Encode the current decision while preserving the root information viewer."""
    if state.input_stack:
        decision = DecisionDescriptor("INPUT", request=state.input_stack[-1])
    else:
        hero = state.get_hero(HeroID(context.current_owner_id))
        if hero is None:
            raise ValueError(f"current decision owner does not exist: {context.current_owner_id!r}")
        from goa2.engine.phases import planning_open_for_second_card

        decision = DecisionDescriptor(
            "CARD",
            hero=hero,
            can_finish_planning=planning_open_for_second_card(state, hero.id),
        )
    return encode_decision(
        state,
        decision,
        legal_keys,
        decision_owner_hero_id=context.root_viewer_id,
        perspective_team=context.perspective_team.value,
        context=context,
    )


def legal_keys_for_decision(decision: DecisionDescriptor) -> list[Any]:
    """Return the canonical ordered search keys for an encodable decision."""
    if decision.kind == "CARD":
        if decision.hero is None:
            raise ValueError("CARD decision requires a hero")
        keys: list[Any] = [card.id for card in decision.hero.hand]
        if decision.can_finish_planning:
            keys.append(None)
        return keys
    if decision.kind == "INPUT":
        if decision.request is None:
            raise ValueError("INPUT decision requires a request")
        keys = [
            _search_key(decision.request.request_type, option)
            for option in decision.request.options
        ]
        if decision.request.can_skip:
            keys.append("SKIP")
        return keys
    raise ValueError(f"unsupported decision kind: {decision.kind!r}")


__all__ = ["encode_decision", "encode_search_context", "legal_keys_for_decision"]
