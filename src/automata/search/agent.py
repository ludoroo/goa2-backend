"""ISMCTS agent: same `Agent` protocol as random/heuristic, backed by search.

Each decision the harness asks for becomes the *root* of a fresh search. The
opponent (and, during rollouts, we ourselves) are played by a `HeuristicAgent`
default policy — this is the "opponent-as-environment" first cut (B).

The search is anchored by an explicit :class:`~automata.search.ismcts.RootTarget`:

- :meth:`ISMCTSAgent.choose_card` anchors to the specific hero it is asked for.
- :meth:`ISMCTSAgent.choose_input` requires a non-empty ``owned_hero_ids`` set
  and refuses (raises :class:`ValueError`) when the bot is not an eligible
  responder for the request or when the request is stale. There is no
  team-wide fallback and the public boundary never silently delegates to the
  default policy — callers (the server bot coordinator, the runtime driver,
  tests) must pass an explicit set.
"""

from __future__ import annotations

from typing import Any

from goa2.domain.input import InputRequest
from goa2.domain.models import TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

from ..agents.base import Agent
from ..agents.heuristic_agent import HeuristicAgent
from ..evaluation.value import HeuristicValue, ValueFn
from .config import SearchConfig
from .ismcts import (
    RootTarget,
    _branchable,
    _input_raw_map,
    _team_of_player,
    search,
)
from .prior import HeuristicPrior

# Engine convention: `player_id="simultaneous"` marks a global broadcast
# request (UPGRADE_PHASE and other legacy-shaped simultaneous requests). See
# ``src/goa2/engine/steps/cards.py`` and ``src/goa2/server/errors.py``.
# The agent's public boundary uses this to narrow the non-branchable fallback
# so it can't quietly answer stray non-branchable requests addressed to a
# specific hero or team.
_SIMULTANEOUS_PLAYER_ID = "simultaneous"


class ISMCTSAgent:
    """Information-Set MCTS decision-maker (cut B: fixed opponent model)."""

    def __init__(
        self,
        config: SearchConfig | None = None,
        *,
        default_policy: Agent | None = None,
        value_fn: ValueFn | None = None,
    ) -> None:
        self._cfg = config or SearchConfig()
        # Default policy drives opponents and rollouts. Seeded off the search
        # seed for reproducibility.
        self._policy: Agent = default_policy or HeuristicAgent(self._cfg.seed)
        # Leaf value estimate at the rollout cutoff. Swappable for a learned
        # value model (Rung 2) without touching the search loop.
        self._value: ValueFn = value_fn or HeuristicValue()
        # Expansion prior reuses the heuristic scorers so widening surfaces
        # promising moves first. Only used when the policy exposes the scorers.
        self._prior = (
            HeuristicPrior(self._policy)
            if self._cfg.use_prior and isinstance(self._policy, HeuristicAgent)
            else None
        )

    # -- planning ----------------------------------------------------------- #
    def choose_card(
        self,
        state: GameState,
        hero: Hero,
        *,
        owned_hero_ids: frozenset[str] | None = None,
    ) -> Card | None:
        """Pick a card for ``hero``.

        The search is anchored to this hero; ``owned_hero_ids`` may name a
        broader set (e.g. all bot-controlled teammates) so subsequent tree
        decisions can also be MAX nodes for those heroes, but it MUST include
        ``hero.id``. Omitted → defaults to a single-hero anchor ``{hero.id}``.
        """
        if not hero.hand:
            return None
        our_team = hero.team or TeamColor.RED
        legal = [c.id for c in hero.hand]

        if owned_hero_ids is None:
            owned: frozenset[str] = frozenset({hero.id})
        else:
            if not owned_hero_ids:
                raise ValueError("owned_hero_ids must be non-empty")
            if hero.id not in owned_hero_ids:
                raise ValueError(
                    f"hero {hero.id!r} must be in owned_hero_ids "
                    f"{sorted(owned_hero_ids)!r}"
                )
            owned = owned_hero_ids

        target = RootTarget.card(hero_id=hero.id, owned_hero_ids=owned)
        result = search(
            state,
            our_team,
            "CARD",
            legal,
            self._policy,
            self._cfg,
            self._prior,
            self._value,
            root_target=target,
        )
        if result.best_key is None:
            return None
        return next((c for c in hero.hand if c.id == result.best_key), None)

    # -- resolution --------------------------------------------------------- #
    def choose_input(
        self,
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
    ) -> Any:
        """Answer an input ``request`` on behalf of the configured bot.

        Contract (fail-closed at the public boundary). All checks run in order
        **before** any fallback to the default policy — so a stale, mis-routed
        or ineligible request can never be quietly answered by the fallback:

        1. ``owned_hero_ids`` is **required** non-empty (``ValueError`` else).
        2. Hero-scoped requests: addressed hero must be in ``owned_hero_ids``.
        3. Team-scoped requests (``"team:RED"``): at least one owned hero must
           be on the addressed team.
        4. Freshness: if ``state.input_stack`` has an active request, its id
           must match ``request.id``.
        5. Non-branchable requests are delegated to the default policy **only**
           when they are the intentional global/simultaneous shape
           (``player_id == "simultaneous"``). A non-branchable request
           addressed to a specific hero or team is a caller/engine bug and
           raises ``ValueError``.
        6. Otherwise: run search anchored to a :class:`RootTarget` naming this
           exact request.
        """
        if owned_hero_ids is None or not owned_hero_ids:
            raise ValueError(
                "ISMCTSAgent.choose_input requires a non-empty owned_hero_ids "
                "set — the caller must name the bot's owned heroes explicitly"
            )

        pid = request.player_id

        # 2 / 3. Eligibility — hero/team-scoped requests. Also validates
        # ``simultaneous`` and any other non-hero/non-team player_id: those
        # skip the per-scope check but must still pass ownership (rule 1) so
        # the boundary cannot be called without an anchor.
        if pid.startswith("team:"):
            addressed_team_opt = _team_of_player(state, pid)
            if addressed_team_opt is None:
                raise ValueError(
                    f"request.player_id {pid!r} does not resolve to a known team"
                )
            addressed_team: TeamColor = addressed_team_opt
            eligible = any(
                (
                    (h := state.get_hero(HeroID(hid))) is not None
                    and h.team == addressed_team
                )
                for hid in owned_hero_ids
            )
            if not eligible:
                raise ValueError(
                    f"bot with owned heroes {sorted(owned_hero_ids)!r} is not "
                    f"an eligible responder for team-scoped request "
                    f"addressed to {pid!r}"
                )
        elif pid == _SIMULTANEOUS_PLAYER_ID:
            # Global broadcast — no per-hero/team eligibility to check. We
            # still need an ``addressed_team`` for search's value perspective,
            # but this path only feeds non-branchable requests to the default
            # policy (see rule 5), so the fallback below defaults it if
            # somehow the request is branchable (defensive: we still search
            # from RED's perspective as a stable arbitrary choice).
            addressed_team = TeamColor.RED
        else:
            resolved = _team_of_player(state, pid)
            addressed_team = resolved if resolved is not None else TeamColor.RED
            if pid not in owned_hero_ids:
                raise ValueError(
                    f"bot with owned heroes {sorted(owned_hero_ids)!r} does "
                    f"not control hero {pid!r} addressed by this request"
                )

        # 4. Staleness — check BEFORE the non-branchable fallback so a stale
        # UPGRADE_PHASE-like request can't sneak through as "just a simultaneous
        # global answer".
        if state.input_stack:
            active = state.input_stack[-1]
            if active.id != request.id:
                raise ValueError(
                    f"request {request.id!r} is stale — state's active "
                    f"pending input is {active.id!r}"
                )

        # 5. Non-branchable requests: only the intentional global/simultaneous
        # shape delegates to the default policy. A non-branchable request
        # addressed to a hero or team is a bug (empty options with no way to
        # respond); raise rather than answer arbitrarily.
        if not _branchable(request):
            if pid == _SIMULTANEOUS_PLAYER_ID:
                return self._policy.choose_input(state, request)
            raise ValueError(
                f"non-branchable request addressed to {pid!r} has no options "
                "and is not a simultaneous/global broadcast — refusing to "
                "answer (this indicates an engine or caller bug)"
            )

        raw_map = _input_raw_map(request)
        legal = list(raw_map.keys())
        if not legal:
            return "SKIP" if request.can_skip else None

        target = RootTarget.input(
            request_id=request.id,
            player_id=pid,
            owned_hero_ids=owned_hero_ids,
        )
        result = search(
            state,
            addressed_team,
            "INPUT",
            legal,
            self._policy,
            self._cfg,
            self._prior,
            self._value,
            root_target=target,
        )
        # `search` never returns a zero-visit key for a non-empty legal set;
        # the >1-key path is the only branch that returns None-best, and by
        # construction ``result.best_key`` here is always a member of ``legal``.
        assert result.best_key is not None
        return raw_map[result.best_key]
