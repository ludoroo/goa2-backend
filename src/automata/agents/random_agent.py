"""A uniformly-random baseline agent (seeded for reproducibility).

The weakest sensible opponent and the reference every stronger agent must beat
in the eval harness. Only ever chooses among engine-provided legal options.
"""

from __future__ import annotations

import random
from typing import Any

from goa2.domain.input import InputRequest, selection_value
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState


class RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def choose_card(self, state: GameState, hero: Hero) -> Card | None:  # pyright: ignore[reportUnusedParameter]
        if not hero.hand:
            return None
        return self._rng.choice(list(hero.hand))

    def choose_input(
        self,
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
    ) -> Any:  # pyright: ignore[reportUnusedParameter]
        # ``owned_hero_ids`` is accepted for Agent-protocol compatibility with
        # the runtime driver (see ``automata.agents.base.Agent``); a
        # random policy doesn't use it, but the driver passes it uniformly.
        # UPGRADE_PHASE is a simultaneous, legacy-shaped request: pick one
        # (hero, upgrade card) among those still owing an upgrade. The engine
        # applies one per advance() and re-requests until pending_upgrades empty.
        if request.request_type.value == "UPGRADE_PHASE":
            players = request.context.get("players", {})
            candidates = [
                (hid, info)
                for hid, info in players.items()
                if info.get("remaining", 0) > 0 and info.get("options")
            ]
            if not candidates:
                return None
            hid, info = self._rng.choice(candidates)
            # Each option is a colour group with a `pair` of two card ids; pick
            # one card id from a chosen group (that card goes to hand, its pair
            # partner becomes the item).
            group = self._rng.choice(list(info["options"]))
            pair = group.get("pair") or [d["id"] for d in group.get("card_details", [])]
            card_id = self._rng.choice(list(pair))
            return {"hero_id": hid, "card_id": card_id}

        options = list(request.options)
        # If there are options, usually pick one; occasionally skip when allowed.
        if options and not (request.can_skip and self._rng.random() < 0.1):
            return selection_value(self._rng.choice(options))
        if request.can_skip:
            return "SKIP"
        # No options and cannot skip: return None and let the engine handle it.
        return None
