"""Consensus-override REST endpoints (schema catalogue + decision history)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from goa2.domain.state import GameState
from goa2.engine.overrides import OVERRIDE_OPS, summarize_op
from goa2.server.auth import PlayerDep, RegistryDep
from goa2.server.errors import GameNotFoundError
from goa2.server.models import (
    OverrideHistoryEntry,
    OverrideHistoryResponse,
    OverrideOpSchema,
    OverrideSchemaResponse,
)
from goa2.server.replay import effective_indices, load_replay
from goa2.server.visibility import _card_location, _card_visible_to

router = APIRouter(tags=["overrides"])


@router.get("/overrides/schema", response_model=OverrideSchemaResponse)
async def get_override_schema() -> OverrideSchemaResponse:
    """The op catalogue, auto-derived from the registry.

    Static and game-independent (like /heroes): clients fetch once and cache.
    A hand-written catalogue would drift the first time an op is added.
    """
    return OverrideSchemaResponse(
        ops=[
            OverrideOpSchema(
                name=op.name,
                family=op.family,
                label=op.label,
                description=op.description,
                args_schema=op.args_model.model_json_schema(),
            )
            for op in sorted(OVERRIDE_OPS.values(), key=lambda o: (o.family, o.name))
        ]
    )


# ---------------------------------------------------------------------------
# Player-scoped decision history
# ---------------------------------------------------------------------------


def _card_label(state: GameState, card_id: str, for_hero_id: str | None) -> str:
    """Card name if the viewer is entitled to it NOW, else an anonymous form.

    Identity is masked with the same visibility rule the view uses; a card
    committed facedown reads "a card" until it is public. The omniscient
    replay-debugger view (reveal_all) is never reachable from here.
    """
    if _card_visible_to(state, card_id, for_hero_id) is False:
        return "a card"
    located = _card_location(state, card_id)
    return located[2].name if located else card_id


def _decision_label(d: dict[str, Any], state: GameState, for_hero_id: str | None) -> str:
    kind = d.get("type", "?")
    hero = d.get("hero", "?")
    if kind == "commit":
        return f"{hero} committed {_card_label(state, d.get('card', ''), for_hero_id)}"
    if kind == "pass":
        return f"{hero} passed"
    if kind == "uncommit":
        return f"{hero} took back a committed card"
    if kind == "finish_planning":
        return f"{hero} finished planning"
    if kind == "input":
        sel = d.get("sel")
        if isinstance(sel, str) and _card_visible_to(state, sel, for_hero_id) is False:
            sel = "a hidden card"
        return f"{hero} chose {sel!r}"
    if kind == "rollback":
        return f"{hero} rolled back their action"
    if kind == "cheat_gold":
        return f"{hero} gained {d.get('amount')} gold (cheat)"
    if kind == "timer_timeout":
        return f"Automatic decision for {hero} (timer expired)"
    if kind in ("ov_patch", "ov_unstick"):
        try:
            return f"Override: {summarize_op(d.get('op', ''), d.get('args', {}))}"
        except Exception:
            return f"Override: {d.get('op')}"
    if kind == "ov_rewind":
        return f"The table rewound the game to decision {d.get('to')}"
    return str(kind)


@router.get(
    "/games/{game_id}/overrides/history",
    response_model=OverrideHistoryResponse,
)
async def get_override_history(
    game_id: str, player: PlayerDep, registry: RegistryDep
) -> OverrideHistoryResponse:
    """Player-scoped decision list so a rewind target index means something.

    Card identity is masked with the view's visibility rule: spectators get
    the fully-masked form; a player never sees an opponent's facedown commit.
    """
    try:
        game = registry.get(game_id)
    except GameNotFoundError:
        raise HTTPException(status_code=404, detail="Game not found") from None
    recorder = game.replay_recorder
    if recorder is None or not recorder.path.is_file():
        return OverrideHistoryResponse(total=0, decisions=[])

    _, decisions = load_replay(str(recorder.path))
    live = set(effective_indices(decisions))
    state = game.session.state
    viewer = player.hero_id if not player.is_spectator else None

    entries = [
        OverrideHistoryEntry(
            index=i,
            type=str(d.get("type", "?")),
            round=d.get("r"),
            turn=d.get("t"),
            hero_id=d.get("hero"),
            label=_decision_label(d, state, viewer),
            superseded=(i not in live and d.get("type") != "ov_rewind"),
        )
        for i, d in enumerate(decisions)
    ]
    return OverrideHistoryResponse(total=len(decisions), decisions=entries)
