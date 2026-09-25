"""Public Agent-protocol adapter backed by an ISMCTS search strategy.

Each decision the runtime driver asks for becomes the *root* of a fresh search. The
opponent (and, during rollouts, we ourselves) are played by a `HeuristicAgent`
default policy — this is an "opponent-as-environment" first cut.

The search is anchored by an explicit :class:`~automata.search.ismcts.RootTarget`:

- :meth:`ISMCTSAgent.choose_planning` anchors to the specific hero it is asked for.
- :meth:`ISMCTSAgent.choose_input` requires a non-empty ``owned_hero_ids`` set
  and refuses (raises :class:`ValueError`) when the bot is not an eligible
  responder for the request or when the request is stale. There is no
  team-wide fallback and the public boundary never silently delegates to the
  default policy — callers (the server bot coordinator, the runtime driver,
  tests) must pass an explicit set.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from automata.search.config import PROD_DEFAULT_DECISION_TIMEOUT_SECONDS, SearchConfig
from automata.search.contracts import LeafEvaluator, SearchPolicy
from automata.search.heuristic import HeuristicLeafEvaluator, HeuristicPrior
from automata.search.ismcts.engine import (
    CutoffObserver,
    RootTarget,
    _branchable,
    _input_raw_map,
    _team_of_player,
    search,
)
from automata.search.ismcts.strategy import ISMCTSStrategy, SearchStrategy, StrategyResult
from automata.search.node import Key
from goa2.domain.input import InputRequest
from goa2.domain.models import TeamColor
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.phases import planning_open_for_second_card

from .capabilities import BoundedComputeCapability, heuristic_fallback
from .contracts import Agent, PlanningDecision
from .heuristic_agent import HeuristicAgent

# Engine convention: `player_id="simultaneous"` marks a global broadcast
# request (UPGRADE_PHASE and other simultaneous requests). See
# ``src/goa2/engine/steps/cards.py`` and ``src/goa2/server/errors.py``.
# The agent's public boundary uses this to narrow the non-branchable fallback
# so it can't quietly answer stray non-branchable requests addressed to a
# specific hero or team.
_SIMULTANEOUS_PLAYER_ID = "simultaneous"


class ISMCTSAgent:
    """Information-Set MCTS decision-maker with an injected opponent model."""

    def __init__(
        self,
        config: SearchConfig | None = None,
        *,
        environment_policy: Agent | None = None,
        continuation_policy: Agent | None = None,
        leaf_evaluator: LeafEvaluator | None = None,
        cutoff_observer: CutoffObserver | None = None,
        prior: SearchPolicy | None = None,
        strategy: SearchStrategy | None = None,
    ) -> None:
        self._cfg = config or SearchConfig()
        if prior is not None and not self._cfg.use_prior:
            raise ValueError("an explicit prior cannot be used when use_prior=False")
        self._policy: Agent = environment_policy or HeuristicAgent(self._cfg.seed)
        self._continuation_policy = continuation_policy or self._policy
        # Leaf value estimate at the rollout cutoff. Swappable for a learned
        # value model (Rung 2) without touching the search loop.
        self._leaf_evaluator: LeafEvaluator = leaf_evaluator or HeuristicLeafEvaluator()
        self._cutoff_observer = cutoff_observer
        # Expansion prior reuses the heuristic scorers so widening surfaces
        # promising moves first. Only used when the policy exposes the scorers.
        self._prior: SearchPolicy | None = prior
        if self._prior is None and self._cfg.use_prior and isinstance(self._policy, HeuristicAgent):
            self._prior = HeuristicPrior(self._policy)
        self._strategy = strategy or ISMCTSStrategy(
            environment_policy=self._policy,
            config=self._cfg,
            prior=self._prior,
            leaf_evaluator=self._leaf_evaluator,
            continuation_policy=self._continuation_policy,
            cutoff_observer=self._cutoff_observer,
            search_runner=lambda *args, **kwargs: search(*args, **kwargs),
        )

    def _select(
        self,
        state: GameState,
        team: TeamColor,
        target: RootTarget,
        legal: Sequence[Key],
    ) -> StrategyResult[Key]:
        result = self._strategy.select(state, team, target, legal)
        if result.candidates != tuple(legal):
            raise ValueError("search strategy changed or reordered the legal candidates")
        return result

    # -- planning ----------------------------------------------------------- #
    def choose_planning(
        self,
        state: GameState,
        hero: Hero,
        *,
        owned_hero_ids: frozenset[str] | None = None,
    ) -> PlanningDecision:
        """Pick a card for ``hero``.

        The search is anchored to this hero; ``owned_hero_ids`` may name a
        broader set (e.g. all bot-controlled teammates) so subsequent tree
        decisions can also be MAX nodes for those heroes, but it MUST include
        ``hero.id``. Omitted → defaults to a single-hero anchor ``{hero.id}``.
        """
        if not hero.hand:
            return PlanningDecision.pass_()
        our_team = hero.team or TeamColor.RED
        legal: list[str | None] = [c.id for c in hero.hand]
        if planning_open_for_second_card(state, HeroID(hero.id)):
            legal.append(None)

        if owned_hero_ids is None:
            owned: frozenset[str] = frozenset({hero.id})
        else:
            if not owned_hero_ids:
                raise ValueError("owned_hero_ids must be non-empty")
            if hero.id not in owned_hero_ids:
                raise ValueError(
                    f"hero {hero.id!r} must be in owned_hero_ids " f"{sorted(owned_hero_ids)!r}"
                )
            owned = owned_hero_ids

        target = RootTarget.card(hero_id=hero.id, owned_hero_ids=owned)
        strategy_result = self._select(state, our_team, target, legal)
        if strategy_result.selected_candidate is None:
            return PlanningDecision.finish()
        card = next(c for c in hero.hand if c.id == strategy_result.selected_candidate)
        return PlanningDecision.commit(card)

    # -- resolution --------------------------------------------------------- #
    def choose_input(
        self,
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
        decision_owner_hero_id: str | None = None,
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
        if decision_owner_hero_id is None:
            raise ValueError("ISMCTSAgent.choose_input requires decision_owner_hero_id")
        if decision_owner_hero_id not in owned_hero_ids:
            raise ValueError("decision owner must be in owned_hero_ids")
        owner = state.get_hero(HeroID(decision_owner_hero_id))
        if owner is None or owner.team is None:
            raise ValueError(f"unknown decision owner {decision_owner_hero_id!r}")

        pid = request.player_id

        # 2 / 3. Eligibility — hero/team-scoped requests. Also validates
        # ``simultaneous`` and any other non-hero/non-team player_id: those
        # skip the per-scope check but must still pass ownership (rule 1) so
        # the boundary cannot be called without an anchor.
        if pid.startswith("team:"):
            addressed_team_opt = _team_of_player(state, pid)
            if addressed_team_opt is None:
                raise ValueError(f"request.player_id {pid!r} does not resolve to a known team")
            addressed_team: TeamColor = addressed_team_opt
            eligible = any(
                ((h := state.get_hero(HeroID(hid))) is not None and h.team == addressed_team)
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

        if pid != _SIMULTANEOUS_PLAYER_ID and owner.team != addressed_team:
            raise ValueError("decision owner is not on the request's addressed team")

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
            decision_owner_hero_id=decision_owner_hero_id,
        )
        strategy_result = self._select(state, owner.team, target, legal)
        selected_key = strategy_result.selected_candidate
        assert selected_key is not None
        return raw_map[selected_key]

    bounded_compute = BoundedComputeCapability(
        PROD_DEFAULT_DECISION_TIMEOUT_SECONDS, heuristic_fallback
    )
