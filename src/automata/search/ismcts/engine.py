"""Single-perspective Information-Set Monte Carlo Tree Search engine.

Design:
- **Determinization** fixes one plausible hidden world per iteration by
  resampling enemy face-down commits (`runtime.determinize`).
- **Opponents are the environment.** Every enemy decision — planning commits and
  resolution inputs — is played by the injected environment policy, which
  defaults to `HeuristicAgent` in the public agent. Opponent choices are never
  branched on, so every tree node is one of the controlled side's MAX nodes.
- **Multi-step actions.** Card, hex, target, and later controlled decisions are
  successive nodes on one path. A terminal or leaf reward is back-propagated
  through every visited node, crediting the complete decision sequence.
- **Depth cutoff.** A rollout can stop after a configured number of round
  advances or controlled decisions, then consults the injected
  :class:`~automata.search.contracts.LeafEvaluator` (already normalized to
  ``[-1, 1]``) and maps it into ``[0, 1]`` reward exactly once via
  ``(v + 1) / 2``. Terminal wins/losses bypass the evaluator and use
  :func:`terminal_reward` (1.0 / 0.0 / 0.5).

The engine is driven through a throwaway `GameSession` over a single clone that
is *mutated in place* as the iteration descends and rolls out — one clone per
iteration, no per-edge cloning.

Root anchoring: the search is anchored by an explicit
:class:`RootTarget` that names the decision kind (``CARD`` / ``INPUT``), its
owner, and — for input roots — the exact request id and addressed player.
Validation rejects stale, mismatched, and terminal roots with
:class:`RootMismatchError`, so callers never receive an arbitrary action.
"""

from __future__ import annotations

import contextvars
import hashlib
import logging
import math
import random
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar, cast

from automata.agents.contracts import Agent, PlanningKind
from automata.decision import ActionBoundaryKind, DecisionDescriptor, DecisionSemanticRole
from automata.runtime.clone import clone_state
from automata.runtime.determinize import determinize
from goa2.domain.input import (
    InputOption,
    InputRequest,
    InputRequestType,
    InputResponse,
    selection_value,
)
from goa2.domain.models import GamePhase, StepType, TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.phases import planning_open_for_second_card
from goa2.engine.session import GameSession, SessionResultType

from ..config import SearchConfig
from ..contextual_noop import ContextualNoopKind, contextual_noop_shape
from ..continuation import AgentContinuationPolicy, as_continuation_policy
from ..contracts import (
    ContinuationPolicy,
    CutoffUnit,
    LeafEvaluator,
    LeafMode,
    PolicyScores,
    PolicyScoreSource,
    ScoreSemantics,
    SearchContext,
    SearchPolicy,
    score_policy,
    supports_contextual_root_coverage,
    supports_immediate_edge,
)
from ..heuristic import HeuristicLeafEvaluator
from ..node import Key, Node, action_key
from ..root import RootMismatchError as RootMismatchError
from ..root import RootTarget, ValidatedRoot, validate_root, validate_root_decision
from ..scheduling import RootSearchPlan, _publish_root_search_plan, resolve_root_search_plan

# --------------------------------------------------------------------------- #
# Decision representation: what the engine is asking *us* for right now.
# --------------------------------------------------------------------------- #


_P = ParamSpec("_P")
_R = TypeVar("_R")
_IN_HYPOTHETICAL_SEARCH: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "automata_in_hypothetical_search", default=False
)
_NO_IMMEDIATE_EDGE = object()


class _HypotheticalEngineInfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            _IN_HYPOTHETICAL_SEARCH.get()
            and record.levelno == logging.INFO
            and record.name.startswith("goa2.engine")
        )


_HYPOTHETICAL_ENGINE_INFO_FILTER = _HypotheticalEngineInfoFilter()


@contextmanager
def _suppress_hypothetical_engine_info() -> Iterator[None]:
    """Suppress engine INFO records emitted while this context drives a clone.

    The filter is installed permanently on existing handlers and activated by
    a context variable. That keeps concurrent real-game logs in other threads
    visible, while warnings and errors from the hypothetical engine still pass.
    """
    handlers = set(logging.getLogger().handlers)
    for candidate in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(candidate, logging.Logger):
            handlers.update(candidate.handlers)
    for handler in handlers:
        if _HYPOTHETICAL_ENGINE_INFO_FILTER not in handler.filters:
            handler.addFilter(_HYPOTHETICAL_ENGINE_INFO_FILTER)

    token = _IN_HYPOTHETICAL_SEARCH.set(True)
    try:
        yield
    finally:
        _IN_HYPOTHETICAL_SEARCH.reset(token)


def _without_hypothetical_engine_info(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(func)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with _suppress_hypothetical_engine_info():
            return func(*args, **kwargs)

    return wrapped


class CutoffObserver(Protocol):
    """Diagnostic callback invoked after validating a nonterminal leaf value.

    Exceptions raised by the observer propagate and abort the search.
    """

    def __call__(self, state: GameState, team: TeamColor, active_value: float) -> object: ...


class SearchBudgetExceeded(RuntimeError):
    """Search stopped cooperatively after exhausting an internal safety bound."""


class SearchDeadlineExceeded(SearchBudgetExceeded):
    """Search exceeded its internal monotonic decision deadline."""


class SearchAdvanceLimitExceeded(SearchBudgetExceeded):
    """One simulation exceeded its deterministic engine-advance limit."""


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise SearchDeadlineExceeded("ISMCTS decision deadline exceeded")


def _enemy(team: TeamColor) -> TeamColor:
    return TeamColor.BLUE if team == TeamColor.RED else TeamColor.RED


def _team_of_player(state: GameState, player_id: str) -> TeamColor | None:
    """Team responsible for a decision addressed to `player_id`.

    `player_id` is a hero id or a team delegate like "team:RED".
    """
    if player_id.startswith("team:"):
        name = player_id.split(":", 1)[1]
        for color in state.teams:
            if color.value == name or color.name == name:
                return color
        return None
    hero = state.get_hero(HeroID(player_id))
    return hero.team if hero is not None else None


def _find_card(hero: Hero, card_id: Any) -> Card | None:
    return next((c for c in hero.hand if c.id == card_id), None)


def _input_raw_map(request: InputRequest) -> dict[Key, Any]:
    """Map each legal action key at this request back to its raw selection."""
    raw: dict[Key, Any] = {}
    for opt in request.options:
        value = selection_value(opt)
        raw[action_key(value)] = value
    if request.can_skip:
        raw["SKIP"] = "SKIP"
    return raw


def _branchable(request: InputRequest) -> bool:
    """Can the search meaningfully branch on this request?

    Requests whose choices are not carried in `options` are not searchable.
    Searching an empty-option request would otherwise loop the engine forever.
    """
    return bool(request.options)


def _decision_owner_id(
    state: GameState, decision: DecisionDescriptor, context: SearchContext
) -> str:
    """Resolve a concrete eligible hero without changing the fixed root viewer."""
    if decision.hero is not None:
        return str(decision.hero.id)
    if decision.request is None:
        return context.current_owner_id

    player_id = decision.request.player_id
    if not player_id.startswith("team:"):
        return player_id

    addressed_team = _team_of_player(state, player_id)
    if addressed_team is None:
        raise ValueError(f"unknown team-scoped request owner {player_id!r}")
    for candidate_id in (context.current_owner_id, context.root_viewer_id):
        candidate = state.get_hero(HeroID(candidate_id))
        if candidate is not None and candidate.team == addressed_team:
            return candidate_id
    raise ValueError(f"no eligible decision owner for team-scoped request {player_id!r}")


def legal_keys(decision: DecisionDescriptor) -> list[Key]:
    """Legal action keys at one of *our* decisions ([] means forced / no branch)."""
    if decision.kind == "CARD":
        hero = decision.hero
        assert hero is not None
        keys: list[Key] = [c.id for c in hero.hand]
        if decision.can_finish_planning:
            keys.append(None)
        return keys  # empty hand -> forced pass, no branch
    if decision.kind == "INPUT":
        assert decision.request is not None
        return list(_input_raw_map(decision.request).keys())
    return []


@dataclass(frozen=True, slots=True)
class SearchProgressionDiagnostics:
    """Compact engine position and transition counters at a search stall."""

    phase: str
    round: int
    actor: str | None
    pending_request: str | None
    stack_depth: int
    top_step: str | None
    transition_counts: tuple[tuple[str, int], ...]


class SearchProgressionError(RuntimeError):
    """Hypothetical progression failed to make bounded deterministic progress."""

    def __init__(self, reason: str, diagnostics: SearchProgressionDiagnostics) -> None:
        self.reason = reason
        self.diagnostics = diagnostics
        self.root_search_plan: RootSearchPlan | None = None
        self.phase = diagnostics.phase
        self.round = diagnostics.round
        self.actor = diagnostics.actor
        self.pending_request = diagnostics.pending_request
        self.stack_depth = diagnostics.stack_depth
        self.top_step = diagnostics.top_step
        self.transition_counts = dict(diagnostics.transition_counts)
        counts = ",".join(f"{name}={value}" for name, value in diagnostics.transition_counts)
        super().__init__(
            f"ISMCTS hypothetical progression failed: {reason} "
            f"(phase={self.phase}, round={self.round}, actor={self.actor}, "
            f"request={self.pending_request}, stack={self.stack_depth}:{self.top_step}, "
            f"transitions={counts})"
        )

    def attach_root_search_plan(self, plan: RootSearchPlan) -> None:
        """Preserve the validated plan that was active when progression failed."""
        if self.root_search_plan is None:
            self.root_search_plan = plan


def _state_fingerprint(state: GameState) -> bytes:
    """Stable progression fingerprint without the immutable/derived board."""
    payload = state.model_dump_json(exclude={"board"}).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).digest()


def _decision_fingerprint(state: GameState, decision: DecisionDescriptor) -> tuple[object, ...]:
    request = decision.request
    return (
        _state_fingerprint(state),
        decision.kind,
        str(decision.hero.id) if decision.hero is not None else None,
        request.id if request is not None else None,
        request.player_id if request is not None else None,
        tuple(legal_keys(decision)),
    )


class _Simulator:
    """Drives one determinized clone forward, auto-playing the opponent.

    Stops (returns a `Decision`) only on decisions owned by the configured bot
    or game over. Every other decision — including uncommitted teammates that
    the bot does *not* own and any opponent decision — is resolved
    via the default policy so it never becomes a search root.

    ``owned_hero_ids`` is the anchor. A planning decision is "ours" only if the
    uncommitted hero's id is in this set — a still-uncommitted teammate (bot
    or human) that is NOT owned is played by the default policy instead of
    surfacing as a Decision. A resolution input request is "ours" when its
    ``player_id`` addresses an owned hero, or, for team-scoped requests, when
    at least one owned hero is on the addressed team (the "bot eligible"
    branch — the caller decides eligibility before invoking search).

    ``our_team`` is retained for value estimation (terminal winner mapping,
    ``evaluate_state`` perspective).
    """

    def __init__(
        self,
        state: GameState,
        our_team: TeamColor,
        environment_policy: Agent,
        *,
        owned_hero_ids: frozenset[str],
        cfg: SearchConfig | None = None,
        deadline: float | None = None,
        max_advance_steps: int | None = None,
    ) -> None:
        self.state = state
        # Disposable simulations preserve control flow but never serialize rollback snapshots.
        self.session = GameSession(state, create_rollback_snapshots=False)
        self.our_team = our_team
        self.environment_policy = environment_policy
        self.owned_hero_ids = owned_hero_ids
        self.cfg = cfg or SearchConfig()
        self.deadline = deadline
        self.max_advance_steps = (
            self.cfg.max_advance_steps if max_advance_steps is None else max_advance_steps
        )
        if self.max_advance_steps < 1:
            raise ValueError("max_advance_steps must be positive")
        self.advance_steps = 0
        self._advance_calls = 0
        self._advance_transitions = 0
        self._session_advances = 0
        self._environment_planning = 0
        self._environment_inputs = 0
        self._continuation_inputs = 0
        self._current_advance_transitions = 0

    def _progression_diagnostics(
        self,
        decision: DecisionDescriptor | None = None,
        *,
        forced_decisions: int = 0,
    ) -> SearchProgressionDiagnostics:
        request = decision.request if decision is not None else None
        if request is None and self.state.input_stack:
            request = self.state.input_stack[-1]
        top = self.state.execution_stack[-1] if self.state.execution_stack else None
        top_type = getattr(top, "type", None)
        top_step = (
            str(getattr(top_type, "value", top_type))
            if top_type is not None
            else type(top).__name__ if top is not None else None
        )
        actor = str(self.state.current_actor_id) if self.state.current_actor_id else None
        if actor is None and decision is not None:
            if decision.hero is not None:
                actor = str(decision.hero.id)
            elif request is not None:
                actor = request.player_id
        return SearchProgressionDiagnostics(
            phase=self.state.phase.value,
            round=self.state.round,
            actor=actor,
            pending_request=request.id if request is not None else None,
            stack_depth=len(self.state.execution_stack),
            top_step=top_step,
            transition_counts=(
                ("advance_calls", self._advance_calls),
                ("advance_transitions", self._advance_transitions),
                ("current_advance_transitions", self._current_advance_transitions),
                ("session_advances", self._session_advances),
                ("environment_planning", self._environment_planning),
                ("environment_inputs", self._environment_inputs),
                ("continuation_inputs", self._continuation_inputs),
                ("forced_decisions", forced_decisions),
            ),
        )

    def progression_error(
        self,
        reason: str,
        decision: DecisionDescriptor | None = None,
        *,
        forced_decisions: int = 0,
    ) -> SearchProgressionError:
        return SearchProgressionError(
            reason,
            self._progression_diagnostics(decision, forced_decisions=forced_decisions),
        )

    def _check_advance_budget(self) -> None:
        _check_deadline(self.deadline)
        if self.advance_steps >= self.max_advance_steps:
            raise SearchAdvanceLimitExceeded(
                f"ISMCTS simulator advance-step limit exceeded ({self.max_advance_steps})"
            )
        self.advance_steps += 1

    def _record_transition(self, kind: str, decision: DecisionDescriptor | None = None) -> None:
        self._current_advance_transitions += 1
        self._advance_transitions += 1
        if kind == "session":
            self._session_advances += 1
        elif kind == "planning":
            self._environment_planning += 1
        elif kind == "input":
            self._environment_inputs += 1
        elif kind == "continuation":
            self._continuation_inputs += 1
        else:
            raise ValueError(f"unknown transition kind: {kind!r}")
        if self._current_advance_transitions > self.cfg.max_advance_transitions:
            raise self.progression_error(
                f"advance transition limit exceeded ({self.cfg.max_advance_transitions})",
                decision,
            )

    # -- opponent-as-environment advance ----------------------------------- #
    def _next_uncommitted(self) -> Hero | None:
        for team in self.state.teams.values():
            for hero in team.heroes:
                hid = HeroID(hero.id)
                if planning_open_for_second_card(self.state, hid):
                    return hero
                if hid not in self.state.pending_inputs:
                    return hero
        return None

    def _is_owned_hero(self, hero: Hero) -> bool:
        return hero.id in self.owned_hero_ids

    def _is_owned_request(self, request: InputRequest) -> bool:
        """Is `request` addressed to one of our owned heroes?

        Hero-scoped ``player_id`` matches by exact id. Team-scoped requests
        (``"team:RED"``) match when any owned hero is on that team — the
        driver/coordinator delegates the responder identity to the search only
        after it has already confirmed the bot is an eligible responder for
        that team, so this check just filters out cross-team addressing.
        """
        pid = request.player_id
        if pid in self.owned_hero_ids:
            return True
        if pid.startswith("team:"):
            addressed = _team_of_player(self.state, pid)
            if addressed is None:
                return False
            for hid in self.owned_hero_ids:
                hero = self.state.get_hero(HeroID(hid))
                if hero is not None and hero.team == addressed:
                    return True
        return False

    def advance(
        self,
        pending: InputResponse | None = None,
        *,
        action_boundary: _ActionBoundary | None = None,
    ) -> DecisionDescriptor:
        """Advance until our decision, game end, or an optional action boundary."""
        self._advance_calls += 1
        self._current_advance_transitions = 0
        resp = pending
        seen_planning: set[tuple[object, ...]] = set()
        seen_environment_inputs: set[tuple[object, ...]] = set()
        while True:
            self._check_advance_budget()
            if action_boundary is not None and resp is None:
                boundary_kind = _action_boundary_kind(self.state, action_boundary)
                if boundary_kind is not None:
                    return DecisionDescriptor("BOUNDARY", action_boundary_kind=boundary_kind)
            if self.state.phase == GamePhase.PLANNING:
                hero = self._next_uncommitted()
                if hero is not None:
                    second_card_window = planning_open_for_second_card(self.state, HeroID(hero.id))
                    if self._is_owned_hero(hero):
                        return DecisionDescriptor(
                            "CARD", hero=hero, can_finish_planning=second_card_window
                        )
                    # Non-owned commit (teammate or opponent) = hidden sample
                    # via the default policy so it never becomes a root.
                    planning_fingerprint = (
                        _state_fingerprint(self.state),
                        str(hero.id),
                        second_card_window,
                    )
                    self._record_transition("planning")
                    if planning_fingerprint in seen_planning:
                        raise self.progression_error(
                            "repeated environment planning decision",
                            DecisionDescriptor(
                                "CARD",
                                hero=hero,
                                can_finish_planning=second_card_window,
                            ),
                        )
                    seen_planning.add(planning_fingerprint)
                    planning = self.environment_policy.choose_planning(self.state, hero)
                    if planning.kind is PlanningKind.FINISH:
                        self.session.finish_planning(HeroID(hero.id))
                    elif planning.kind is PlanningKind.PASS:
                        self.session.pass_turn(HeroID(hero.id))
                    else:
                        assert planning.card is not None
                        self.session.commit_card(HeroID(hero.id), planning.card)
                    continue
                # All committed: fall through to advance the resolution stack.

            before = _state_fingerprint(self.state)
            self._record_transition("session")
            if action_boundary is None:
                result = self.session.advance(resp)
            else:
                result = self.session.advance(
                    resp,
                    stop_before_step=_is_active_action_boundary_step,
                )
            resp = None

            if result.result_type == SessionResultType.GAME_OVER:
                return DecisionDescriptor("OVER", winner=result.winner)
            if action_boundary is not None:
                boundary_kind = _action_boundary_kind(self.state, action_boundary)
                if boundary_kind is not None:
                    return DecisionDescriptor("BOUNDARY", action_boundary_kind=boundary_kind)
            if result.result_type == SessionResultType.INPUT_NEEDED:
                request = result.input_request
                assert request is not None
                if action_boundary is not None and (
                    request.request_type is InputRequestType.TIE_BREAKER
                    or (
                        request.player_id != action_boundary.owner_id
                        and self._is_owned_request(request)
                    )
                ):
                    return DecisionDescriptor(
                        "BOUNDARY",
                        action_boundary_kind=ActionBoundaryKind.INTERRUPTED,
                    )
                if self._is_owned_request(request) and _branchable(request):
                    return DecisionDescriptor("INPUT", request=request)
                # Non-owned input, or a non-branchable request (e.g. UPGRADE_PHASE):
                # resolve with the default policy and keep advancing.
                decision = DecisionDescriptor("INPUT", request=request)
                input_fingerprint = _decision_fingerprint(self.state, decision)
                self._record_transition("input", decision)
                if input_fingerprint in seen_environment_inputs:
                    raise self.progression_error("repeated environment input decision", decision)
                seen_environment_inputs.add(input_fingerprint)
                selection = self.environment_policy.choose_input(self.state, request)
                resp = InputResponse(request_id=request.id, selection=selection)
                continue
            if (
                result.result_type == SessionResultType.ACTION_COMPLETE
                and before == _state_fingerprint(self.state)
            ):
                raise self.progression_error("unchanged ACTION_COMPLETE")
            # ACTION_COMPLETE / PHASE_CHANGED: keep advancing.

    # -- root anchoring ---------------------------------------------------- #
    def advance_to_root(self, target: RootTarget) -> DecisionDescriptor:
        """Advance to *exactly* the target root decision, or raise.

        Delegates to :meth:`advance`, then validates that the surfaced
        :class:`Decision` matches the target and that the target references
        entities in the advanced state. On any mismatch
        (terminal, wrong hero, stale request id, wrong kind) raises
        :class:`RootMismatchError`. Never returns a non-matching Decision.
        """
        decision = self.advance()
        if decision.is_terminal:
            raise RootMismatchError(
                f"expected root {target.kind} but game is already over "
                f"(winner={decision.winner!r})"
            )
        validate_root_decision(self.state, self.our_team, target, decision)
        return decision

    # -- applying *our* action --------------------------------------------- #
    def apply_ours(
        self,
        decision: DecisionDescriptor,
        key: Key | None,
        *,
        action_boundary: _ActionBoundary | None = None,
    ) -> DecisionDescriptor:
        """Apply our action, optionally stopping before its resolution boundary."""
        if decision.kind == "CARD":
            hero = decision.hero
            assert hero is not None
            card = _find_card(hero, key) if key is not None else None
            if planning_open_for_second_card(self.state, HeroID(hero.id)) and card is None:
                self.session.finish_planning(HeroID(hero.id))
            elif card is None or not hero.hand:
                self.session.pass_turn(HeroID(hero.id))
            else:
                self.session.commit_card(HeroID(hero.id), card)
            return self.advance(action_boundary=action_boundary)

        # INPUT
        request = decision.request
        assert request is not None
        if key is None:
            selection = "SKIP" if request.can_skip else None
        else:
            selection = _input_raw_map(request).get(key, key)
        return self.advance(
            InputResponse(request_id=request.id, selection=selection),
            action_boundary=action_boundary,
        )

    def apply_stable_turn_root(
        self,
        decision: DecisionDescriptor,
        key: Key | None,
        *,
        boundary: _StableTurnBoundary,
        continuation_policy: ContinuationPolicy,
        context: SearchContext,
    ) -> DecisionDescriptor:
        """Apply an INPUT root and drive only its enclosing actor to stability."""
        request = decision.request
        if decision.kind != "INPUT" or request is None:
            raise self.progression_error("stable-turn boundary requires an INPUT root", decision)
        selection = (
            "SKIP"
            if key is None and request.can_skip
            else None if key is None else _input_raw_map(request).get(key, key)
        )
        return self._advance_stable_turn(
            InputResponse(request_id=request.id, selection=selection),
            boundary=boundary,
            continuation_policy=continuation_policy,
            context=context,
        )

    def _advance_stable_turn(
        self,
        pending: InputResponse,
        *,
        boundary: _StableTurnBoundary,
        continuation_policy: ContinuationPolicy,
        context: SearchContext,
    ) -> DecisionDescriptor:
        """Stable-turn state machine with explicit post-finalization routing.

        Before the enclosing actor finalizes, owned follow-up prompts use the
        continuation policy and foreign prompts use the environment. Once
        ``resolution_owner_id`` is cleared, every cleanup/tie prompt is an
        environment decision, including same-team requests.
        """
        self._advance_calls += 1
        self._current_advance_transitions = 0
        response: InputResponse | None = pending
        decisions = 0
        seen_inputs: set[tuple[object, ...]] = set()

        while True:
            self._check_advance_budget()
            if _stable_turn_boundary_reached(self.state, boundary):
                return DecisionDescriptor(
                    "BOUNDARY", action_boundary_kind=ActionBoundaryKind.COMPLETE
                )

            before = _state_fingerprint(self.state)
            self._record_transition("session")
            result = self.session.advance(
                response,
                stop_before_step=lambda state, _step: _stable_turn_boundary_reached(
                    state, boundary
                ),
            )
            response = None

            if result.result_type is SessionResultType.GAME_OVER:
                return DecisionDescriptor("OVER", winner=result.winner)
            if _stable_turn_boundary_reached(self.state, boundary):
                return DecisionDescriptor(
                    "BOUNDARY", action_boundary_kind=ActionBoundaryKind.COMPLETE
                )
            if result.result_type is SessionResultType.INPUT_NEEDED:
                request = result.input_request
                assert request is not None
                decision = DecisionDescriptor("INPUT", request=request)
                fingerprint = _decision_fingerprint(self.state, decision)
                if fingerprint in seen_inputs:
                    raise self.progression_error(
                        "repeated stable-turn input decision",
                        decision,
                        forced_decisions=decisions,
                    )
                if decisions >= self.cfg.max_forced_decisions:
                    raise self.progression_error(
                        f"stable-turn decision limit exceeded ({self.cfg.max_forced_decisions})",
                        decision,
                        forced_decisions=decisions,
                    )
                seen_inputs.add(fingerprint)
                decisions += 1

                actor_not_finalized = (
                    self.state.resolution_owner_id is not None
                    and str(self.state.resolution_owner_id) == boundary.actor_id
                )
                if actor_not_finalized and self._is_owned_request(request) and _branchable(request):
                    self._record_transition("continuation", decision)
                    follow_up_context = context.for_decision(
                        decision,
                        owner_id=_decision_owner_id(self.state, decision, context),
                    )
                    key = continuation_policy.choose(
                        follow_up_context,
                        self.state,
                        decision,
                        tuple(legal_keys(decision)),
                    )
                    selection = _input_raw_map(request).get(key, key)
                    if key is None and request.can_skip:
                        selection = "SKIP"
                else:
                    self._record_transition("input", decision)
                    selection = self.environment_policy.choose_input(self.state, request)
                response = InputResponse(request_id=request.id, selection=selection)
                continue
            if (
                result.result_type is SessionResultType.ACTION_COMPLETE
                and before == _state_fingerprint(self.state)
            ):
                raise self.progression_error("unchanged stable-turn ACTION_COMPLETE")


# --------------------------------------------------------------------------- #
# Value estimation
# --------------------------------------------------------------------------- #


def terminal_reward(winner: str | None, our_team: TeamColor) -> float:
    """Reward for a terminal game outcome, from ``our_team``'s perspective.

    Public helper — the terminal branch of :func:`_rollout` and any offline
    training / evaluation code MUST go through this so wins/draws/losses stay
    on the same 1.0 / 0.5 / 0.0 scale as the mapped value estimate.
    """
    if winner is None:
        return 0.5  # draw / undecided
    return 1.0 if winner.upper() == our_team.value.upper() else 0.0


def _value_to_reward(value: float) -> float:
    """Map a validated normalized leaf value into search reward space."""
    return 0.5 * (value + 1.0)


def _normalize_weights(
    weights: dict[Key, float] | None, legal: list[Key]
) -> dict[Key, float] | None:
    """Turn raw prior scores into a probability distribution over ``legal``.

    Heuristic scores are unbounded and can be negative, so we softmax them
    (shifted by the max for numerical stability) into P(a) that sums to 1 over
    the legal keys. Missing keys get the minimum score. Returns ``None`` if
    there is nothing usable, so the caller falls back to plain UCB1.
    """
    if not weights:
        return None
    vals = [weights[k] for k in legal if k in weights]
    if not vals:
        return None
    lo = min(vals)
    hi = max(weights.get(k, lo) for k in legal)
    exps = {k: math.exp(weights.get(k, lo) - hi) for k in legal}
    total = sum(exps.values())
    if total <= 0.0:
        return None
    return {k: v / total for k, v in exps.items()}


_ACTION_BOUNDARY_STEPS = frozenset(
    {
        StepType.CONFIRM_RESOLUTION,
        StepType.RESOLVE_CARD,
        StepType.PERFORM_CARD_ACTION,
        StepType.FINALIZE_HERO_TURN,
        StepType.RESOLVE_TIE_BREAKER,
    }
)


@dataclass(frozen=True, slots=True)
class _ActionBoundary:
    owner_id: str
    actor_id: str
    start_round: int


@dataclass(frozen=True, slots=True)
class _StableTurnBoundary:
    """Identity of one enclosing resolution turn, independent of root viewer."""

    actor_id: str
    start_phase: GamePhase
    start_round: int
    start_turn: int


def _resolution_actor_id(state: GameState) -> str | None:
    actor = state.resolution_owner_id or state.current_actor_id
    return str(actor) if actor is not None else None


def _is_active_action_boundary_step(state: GameState, step: Any) -> bool:
    return bool(
        step.type in _ACTION_BOUNDARY_STEPS
        and step.pending_input is None
        and not step.should_skip(state.execution_context)
    )


def _action_boundary_kind(state: GameState, boundary: _ActionBoundary) -> ActionBoundaryKind | None:
    if state.phase is not GamePhase.RESOLUTION or state.round != boundary.start_round:
        return ActionBoundaryKind.COMPLETE
    if _resolution_actor_id(state) != boundary.actor_id:
        return ActionBoundaryKind.COMPLETE
    if state.execution_stack:
        top = state.execution_stack[-1]
        if _is_active_action_boundary_step(state, top):
            return (
                ActionBoundaryKind.INTERRUPTED
                if top.type is StepType.RESOLVE_TIE_BREAKER
                else ActionBoundaryKind.COMPLETE
            )
    return None


def _stable_turn_boundary_reached(state: GameState, boundary: _StableTurnBoundary) -> bool:
    if state.phase is GamePhase.GAME_OVER:
        return False
    if (
        state.phase is not boundary.start_phase
        or state.round != boundary.start_round
        or state.turn != boundary.start_turn
    ):
        return True
    owner = str(state.resolution_owner_id) if state.resolution_owner_id is not None else None
    if owner is None or owner == boundary.actor_id or not state.execution_stack:
        return False
    return state.execution_stack[-1].type in {StepType.RESPAWN_HERO, StepType.RESOLVE_CARD}


def _uses_immediate_edge(cfg: SearchConfig) -> bool:
    return cfg.leaf_mode in {
        LeafMode.IMMEDIATE,
        LeafMode.IMMEDIATE_ACTION,
        LeafMode.STABLE_TURN,
    }


def _root_stable_turn_boundary(
    cfg: SearchConfig,
    root_target: RootTarget,
    *,
    is_root: bool,
    state: GameState,
) -> _StableTurnBoundary | None:
    """Capture an enclosing actor only for an actor-bound resolution INPUT."""
    if (
        cfg.leaf_mode is not LeafMode.STABLE_TURN
        or not is_root
        or root_target.kind != "INPUT"
        or state.phase is not GamePhase.RESOLUTION
        or state.resolution_owner_id is None
    ):
        return None
    return _StableTurnBoundary(
        actor_id=str(state.resolution_owner_id),
        start_phase=state.phase,
        start_round=state.round,
        start_turn=state.turn,
    )


def _root_action_boundary(
    cfg: SearchConfig,
    root_target: RootTarget,
    *,
    is_root: bool,
    context: SearchContext,
    state: GameState,
) -> _ActionBoundary | None:
    """Capture the request owner, action actor, and round for an INPUT root edge."""
    if cfg.leaf_mode is LeafMode.IMMEDIATE_ACTION and is_root and root_target.kind == "INPUT":
        actor_id = _resolution_actor_id(state) or context.current_owner_id
        return _ActionBoundary(context.current_owner_id, actor_id, state.round)
    return None


def _apply_ours(
    sim: _Simulator,
    decision: DecisionDescriptor,
    key: Key | None,
    action_boundary: _ActionBoundary | None,
) -> DecisionDescriptor:
    """Preserve the legacy simulator seam when no action boundary is active."""
    if action_boundary is None:
        return sim.apply_ours(decision, key)
    return sim.apply_ours(decision, key, action_boundary=action_boundary)


def _apply_root_edge(
    sim: _Simulator,
    decision: DecisionDescriptor,
    key: Key | None,
    cfg: SearchConfig,
    root_target: RootTarget,
    *,
    is_root: bool,
    context: SearchContext,
    continuation_policy: ContinuationPolicy,
) -> tuple[DecisionDescriptor, _ActionBoundary | None, _StableTurnBoundary | None]:
    stable_boundary = _root_stable_turn_boundary(
        cfg,
        root_target,
        is_root=is_root,
        state=sim.state,
    )
    if stable_boundary is not None:
        return (
            sim.apply_stable_turn_root(
                decision,
                key,
                boundary=stable_boundary,
                continuation_policy=continuation_policy,
                context=context,
            ),
            None,
            stable_boundary,
        )
    action_boundary = _root_action_boundary(
        cfg,
        root_target,
        is_root=is_root,
        context=context,
        state=sim.state,
    )
    return _apply_ours(sim, decision, key, action_boundary), action_boundary, None


def _effective_root_puct_c(
    cfg: SearchConfig,
    *,
    root_schedule_enabled: bool,
    value_led_noop_comparison: bool,
) -> float:
    """Resolve the root selection constant once for runtime and diagnostics."""
    if (
        cfg.leaf_mode in {LeafMode.IMMEDIATE_ACTION, LeafMode.STABLE_TURN}
        and value_led_noop_comparison
    ):
        return 0.0
    if root_schedule_enabled and cfg.root_puct_c is not None:
        return cfg.root_puct_c
    return cfg.puct_c


def _boundary_leaf_decision(owner_id: str, start_round: int) -> DecisionDescriptor:
    """Provide a privacy-safe, always-encodable context for a boundary value."""
    return DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id=f"immediate-action-boundary:{owner_id}:{start_round}",
            request_type=InputRequestType.CHOOSE_ACTION,
            player_id=owner_id,
            prompt="Immediate action boundary.",
            options=[InputOption.from_value("CONFIRM")],
        ),
    )


def _stable_turn_leaf_decision(boundary: _StableTurnBoundary) -> DecisionDescriptor:
    """Synthesize an encodable leaf without exposing a pending foreign request."""
    return DecisionDescriptor(
        "INPUT",
        request=InputRequest(
            id=(
                f"stable-turn-boundary:{boundary.actor_id}:"
                f"{boundary.start_round}:{boundary.start_turn}"
            ),
            request_type=InputRequestType.CHOOSE_ACTION,
            player_id=boundary.actor_id,
            prompt="Stable turn boundary.",
            options=[InputOption.from_value("CONFIRM")],
        ),
    )


def _same_action_input(
    sim: _Simulator,
    decision: DecisionDescriptor,
    *,
    boundary: _ActionBoundary,
) -> bool:
    if decision.kind != "INPUT" or decision.is_terminal:
        return False
    request = decision.request
    if request is None or request.player_id != boundary.owner_id:
        return False
    if request.request_type is InputRequestType.TIE_BREAKER:
        return False
    if _action_boundary_kind(sim.state, boundary) is not None:
        return False
    if not sim.state.execution_stack:
        return False
    return sim.state.execution_stack[-1].type not in _ACTION_BOUNDARY_STEPS


def _rollout(
    sim: _Simulator,
    decision: DecisionDescriptor,
    cfg: SearchConfig,
    continuation_policy: ContinuationPolicy | Agent,
    leaf_evaluator: LeafEvaluator,
    context: SearchContext,
    *,
    immediate_edge: object = _NO_IMMEDIATE_EDGE,
    action_boundary: _ActionBoundary | None = None,
    stable_turn_boundary: _StableTurnBoundary | None = None,
    cutoff_observer: CutoffObserver | None = None,
) -> float:
    """Continuation-policy playout until the configured cutoff."""
    continuation = as_continuation_policy(continuation_policy)
    start_round = sim.state.round
    decisions = 0
    forced_decisions = 0
    seen_forced_decisions: set[tuple[object, ...]] = set()

    def within_cutoff() -> bool:
        if cfg.cutoff_unit is CutoffUnit.ROUNDS:
            return (sim.state.round - start_round) < cfg.cutoff_limit
        return decisions < cfg.cutoff_limit

    if cfg.leaf_mode is LeafMode.IMMEDIATE_ACTION and action_boundary is not None:
        while _same_action_input(sim, decision, boundary=action_boundary):
            if decisions >= cfg.max_forced_decisions:
                raise sim.progression_error(
                    f"same-action decision limit exceeded ({cfg.max_forced_decisions})",
                    decision,
                    forced_decisions=decisions,
                )
            fingerprint = _decision_fingerprint(sim.state, decision)
            if fingerprint in seen_forced_decisions:
                raise sim.progression_error(
                    "repeated same-action decision",
                    decision,
                    forced_decisions=decisions,
                )
            seen_forced_decisions.add(fingerprint)
            context = context.for_decision(
                decision,
                owner_id=_decision_owner_id(sim.state, decision, context),
            )
            legal = tuple(legal_keys(decision))
            key = continuation.choose(context, sim.state, decision, legal)
            decision = sim.apply_ours(decision, key, action_boundary=action_boundary)
            decisions += 1

    while (
        cfg.leaf_mode is LeafMode.BOUNDED_CONTINUATION
        and not decision.is_terminal
        and within_cutoff()
    ):
        context = context.for_decision(
            decision,
            owner_id=_decision_owner_id(sim.state, decision, context),
        )
        if not legal_keys(decision):
            fingerprint = _decision_fingerprint(sim.state, decision)
            if fingerprint in seen_forced_decisions:
                raise sim.progression_error(
                    "repeated forced decision",
                    decision,
                    forced_decisions=forced_decisions,
                )
            if forced_decisions >= cfg.max_forced_decisions:
                raise sim.progression_error(
                    f"forced decision limit exceeded ({cfg.max_forced_decisions})",
                    decision,
                    forced_decisions=forced_decisions,
                )
            seen_forced_decisions.add(fingerprint)
            forced_decisions += 1
            decision = sim.apply_ours(decision, None)
            decisions += 1
            continue
        legal = tuple(legal_keys(decision))
        key = continuation.choose(context, sim.state, decision, legal)
        decision = sim.apply_ours(decision, key)
        decisions += 1

    # A zero-legal CARD pass is an engine transition, not a model decision.
    # Advance through it even when the selected leaf mode does not run bounded
    # continuation (or its cutoff is already exhausted), so no evaluator is
    # asked to value a pre-transition forced leaf. In particular, learned
    # decision observations are never asked to fabricate planning candidates.
    while not decision.is_terminal and decision.kind != "BOUNDARY" and not legal_keys(decision):
        fingerprint = _decision_fingerprint(sim.state, decision)
        if fingerprint in seen_forced_decisions:
            raise sim.progression_error(
                "repeated forced decision",
                decision,
                forced_decisions=forced_decisions,
            )
        if forced_decisions >= cfg.max_forced_decisions:
            raise sim.progression_error(
                f"forced decision limit exceeded ({cfg.max_forced_decisions})",
                decision,
                forced_decisions=forced_decisions,
            )
        seen_forced_decisions.add(fingerprint)
        forced_decisions += 1
        context = context.for_decision(
            decision,
            owner_id=_decision_owner_id(sim.state, decision, context),
        )
        decision = _apply_ours(sim, decision, None, action_boundary)

    if decision.is_terminal:
        return terminal_reward(decision.winner, sim.our_team)
    if decision.kind == "BOUNDARY":
        boundary_kind = decision.action_boundary_kind
        if boundary_kind is None:
            raise sim.progression_error("boundary missing its cutoff kind")
        if stable_turn_boundary is not None:
            leaf_context = context.for_decision(
                _stable_turn_leaf_decision(stable_turn_boundary),
                owner_id=stable_turn_boundary.actor_id,
            ).for_action_boundary(boundary_kind)
        elif action_boundary is not None:
            leaf_context = context.for_decision(
                _boundary_leaf_decision(action_boundary.owner_id, action_boundary.start_round),
                owner_id=action_boundary.owner_id,
            ).for_action_boundary(boundary_kind)
        else:
            raise sim.progression_error("boundary missing its captured identity")
    else:
        leaf_context = context.for_decision(
            decision,
            owner_id=_decision_owner_id(sim.state, decision, context),
        )
    active_value = (
        leaf_evaluator.evaluate_immediate_edge(leaf_context, sim.state, immediate_edge).value
        if _uses_immediate_edge(cfg)
        and immediate_edge is not _NO_IMMEDIATE_EDGE
        and supports_immediate_edge(leaf_evaluator)
        else leaf_evaluator.evaluate(leaf_context, sim.state).value
    )
    reward = _value_to_reward(active_value)
    if cutoff_observer is not None:
        cutoff_observer(sim.state, sim.our_team, active_value)
    return reward


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def _simulate(
    root: Node,
    root_state: GameState,
    root_target: RootTarget,
    root_legal: tuple[Key, ...],
    root_policy: PolicyScores | None,
    root_priors: dict[Key, float] | None,
    root_coverage_target: int | None,
    root_coverage_noop: Key | None,
    value_led_noop_comparison: bool,
    our_team: TeamColor,
    environment_policy: Agent,
    cfg: SearchConfig,
    rng: random.Random,
    continuation_policy: ContinuationPolicy,
    leaf_evaluator: LeafEvaluator,
    prior: SearchPolicy | None = None,
    *,
    cutoff_observer: CutoffObserver | None = None,
    deadline: float | None = None,
) -> None:
    """One ISMCTS iteration on a fresh determinized clone (mutated in place).

    The first ``Decision`` surfaced by the simulator MUST match ``root_target``
    exactly, else :class:`RootMismatchError` is raised — the search never
    silently descends from a mismatched root.
    """
    world = determinize(root_state, root_target.decision_owner_hero_id, rng)
    simulator_kwargs: dict[str, Any] = {
        "owned_hero_ids": root_target.owned_hero_ids,
        "cfg": cfg,
    }
    if deadline is not None:
        simulator_kwargs["deadline"] = deadline
    sim = _Simulator(world, our_team, environment_policy, **simulator_kwargs)
    # Strict root validation: surface EXACTLY the requested root or raise.
    decision = sim.advance_to_root(root_target)
    context = SearchContext(
        root_viewer_id=root_target.decision_owner_hero_id,
        perspective_team=our_team,
        current_owner_id=root_target.decision_owner_hero_id,
        decision=decision,
    )

    node = root
    path = [root]
    value: float | None = None
    forced_decisions = 0
    seen_forced_decisions: set[tuple[object, ...]] = set()

    while not decision.is_terminal:
        is_root = node is root
        context = context.for_decision(
            decision,
            owner_id=_decision_owner_id(sim.state, decision, context),
        )
        legal = list(root_legal) if is_root else legal_keys(decision)
        if not legal:
            # Forced move (empty hand / no options): no branch, just advance.
            fingerprint = _decision_fingerprint(sim.state, decision)
            if fingerprint in seen_forced_decisions:
                raise sim.progression_error(
                    "repeated forced decision",
                    decision,
                    forced_decisions=forced_decisions,
                )
            if forced_decisions >= cfg.max_forced_decisions:
                raise sim.progression_error(
                    f"forced decision limit exceeded ({cfg.max_forced_decisions})",
                    decision,
                    forced_decisions=forced_decisions,
                )
            seen_forced_decisions.add(fingerprint)
            forced_decisions += 1
            decision = sim.apply_ours(decision, None)
            continue

        # The exact root policy is evaluated, aligned, and normalized once by
        # ``search``. Descendant policies remain tied to each determinized
        # state and exact descendant decision.
        current_context = context
        pol = (
            root_policy
            if is_root
            else (
                score_policy(prior, current_context, sim.state, legal)
                if prior is not None
                else None
            )
        )
        weights = dict(zip(legal, pol.scores, strict=True)) if pol is not None else None
        use_root_schedule = is_root and (
            root_policy is None or root_policy.source is not PolicyScoreSource.FALLBACK
        )
        widen_c = (
            cfg.root_widening_c
            if use_root_schedule and cfg.root_widening_c is not None
            else cfg.widening_c
        )
        widen_alpha = (
            cfg.root_widening_alpha
            if use_root_schedule and cfg.root_widening_alpha is not None
            else cfg.widening_alpha
        )
        order = (
            sorted(legal, key=lambda key: weights[key], reverse=True)
            if weights is not None
            else None
        )

        # Eligible roots deterministically compare the best-ranked concrete
        # action with their contextual no-op using real simulations. Adaptive
        # broad-HEX roots then cover the remaining best-prior actions to M.
        # Priors, legality, and visit counts are not modified by this ordering.
        if is_root and root_coverage_target is not None:
            visited = sum(child.visits > 0 for child in node.children.values())
            if visited < root_coverage_target:
                ranked = order if order is not None else legal
                if root_coverage_noop is not None:
                    concrete = [key for key in ranked if key != root_coverage_noop]
                    coverage_order = [concrete[0], root_coverage_noop, *concrete[1:]]
                else:
                    coverage_order = ranked
                key = next(
                    key
                    for key in coverage_order
                    if key not in node.children or node.children[key].visits == 0
                )
                if key not in node.children:
                    node.expand(legal, rng, coverage_order)
                child = node.children[key]
                node = child
                path.append(child)
                immediate_edge = (
                    leaf_evaluator.prepare_immediate_edge(current_context, sim.state, decision, key)
                    if _uses_immediate_edge(cfg) and supports_immediate_edge(leaf_evaluator)
                    else None
                )
                decision, action_boundary, stable_turn_boundary = _apply_root_edge(
                    sim,
                    decision,
                    key,
                    cfg,
                    root_target,
                    is_root=is_root,
                    context=current_context,
                    continuation_policy=continuation_policy,
                )
                value = _rollout(
                    sim,
                    decision,
                    cfg,
                    continuation_policy,
                    leaf_evaluator,
                    current_context,
                    immediate_edge=immediate_edge,
                    action_boundary=action_boundary,
                    stable_turn_boundary=stable_turn_boundary,
                    cutoff_observer=cutoff_observer,
                )
                break

        if node.should_expand(legal, widen_c, widen_alpha):
            key = node.expand(legal, rng, order)
            child = node.children[key]
            node = child
            path.append(child)
            immediate_edge = (
                leaf_evaluator.prepare_immediate_edge(current_context, sim.state, decision, key)
                if _uses_immediate_edge(cfg) and supports_immediate_edge(leaf_evaluator)
                else None
            )
            decision, action_boundary, stable_turn_boundary = _apply_root_edge(
                sim,
                decision,
                key,
                cfg,
                root_target,
                is_root=is_root,
                context=current_context,
                continuation_policy=continuation_policy,
            )
            value = _rollout(
                sim,
                decision,
                cfg,
                continuation_policy,
                leaf_evaluator,
                current_context,
                immediate_edge=immediate_edge,
                action_boundary=action_boundary,
                stable_turn_boundary=stable_turn_boundary,
                cutoff_observer=cutoff_observer,
            )  # evaluate freshly expanded leaf
            break

        priors = root_priors if is_root else None
        if not is_root and pol is not None:
            priors = (
                weights
                if pol.semantics is ScoreSemantics.PROBABILITIES
                else _normalize_weights(weights, legal)
            )
        puct_c = (
            _effective_root_puct_c(
                cfg,
                root_schedule_enabled=use_root_schedule,
                value_led_noop_comparison=value_led_noop_comparison,
            )
            if is_root
            else cfg.puct_c
        )
        key = node.select(legal, cfg.uct_c, rng, priors, puct_c)
        child = node.children[key]
        node = child
        path.append(child)
        if (
            is_root
            and root_target.kind == "INPUT"
            and cfg.leaf_mode in {LeafMode.IMMEDIATE_ACTION, LeafMode.STABLE_TURN}
        ):
            immediate_edge = (
                leaf_evaluator.prepare_immediate_edge(current_context, sim.state, decision, key)
                if supports_immediate_edge(leaf_evaluator)
                else None
            )
            decision, action_boundary, stable_turn_boundary = _apply_root_edge(
                sim,
                decision,
                key,
                cfg,
                root_target,
                is_root=True,
                context=current_context,
                continuation_policy=continuation_policy,
            )
            value = _rollout(
                sim,
                decision,
                cfg,
                continuation_policy,
                leaf_evaluator,
                current_context,
                immediate_edge=immediate_edge,
                action_boundary=action_boundary,
                stable_turn_boundary=stable_turn_boundary,
                cutoff_observer=cutoff_observer,
            )
            break
        decision = sim.apply_ours(decision, key)

    if value is None:
        value = terminal_reward(decision.winner, our_team)

    for n in path:
        n.update(value)


@dataclass(frozen=True, slots=True)
class RootActionDiagnostic:
    """Root statistics aligned to one caller-supplied legal action.

    ``mean_value`` and ``value_variance`` are in the search reward space
    ``[0, 1]`` (not the leaf evaluator's ``[-1, 1]`` space). An action with
    zero visits reports zero for both fields as an explicit unvisited sentinel,
    not as an estimated neutral value. ``prior_probability`` is the normalized
    probability returned by the policy actually used at the root (including a
    fallback policy); it is ``None`` when no policy was scored. The prior always
    orders expansion and participates in selection only when the effective
    root PUCT constant is positive.
    """

    action: Key
    prior_probability: float | None
    visits: int
    mean_value: float
    value_variance: float


@dataclass
class SearchResult:
    root: Node
    best_key: Key | None  # None => no real choice (forced move)
    root_action_diagnostics: tuple[RootActionDiagnostic, ...] = ()
    requested_iterations: int | None = None
    effective_iterations: int | None = None
    root_coverage_target: int | None = None
    effective_root_puct_c: float | None = None
    schedule_id: str | None = None
    effective_leaf_mode: LeafMode | None = None
    request_type: str | None = None
    semantic_role: DecisionSemanticRole | None = None


def _root_action_diagnostics(
    root: Node,
    legal: Sequence[Key],
    priors: dict[Key, float] | None,
) -> tuple[RootActionDiagnostic, ...]:
    return tuple(
        RootActionDiagnostic(
            action=key,
            prior_probability=priors.get(key, 0.0) if priors is not None else None,
            visits=child.visits if child is not None else 0,
            mean_value=child.q if child is not None else 0.0,
            value_variance=child.value_variance if child is not None else 0.0,
        )
        for key in legal
        for child in (root.children.get(key),)
    )


def _validated_search_root_and_state(
    state: GameState,
    perspective_team: TeamColor,
    root_target: RootTarget,
    legal_candidates: Sequence[Key],
    environment_policy: Agent,
    cfg: SearchConfig | None = None,
    *,
    deadline: float | None = None,
    max_advance_steps: int | None = None,
) -> tuple[ValidatedRoot, GameState]:
    validation_clone = clone_state(state)
    simulator_kwargs: dict[str, Any] = {
        "owned_hero_ids": root_target.owned_hero_ids,
        "cfg": cfg,
    }
    if deadline is not None:
        simulator_kwargs["deadline"] = deadline
    if max_advance_steps is not None:
        simulator_kwargs["max_advance_steps"] = max_advance_steps
    validation_sim = _Simulator(
        validation_clone,
        perspective_team,
        environment_policy,
        **simulator_kwargs,
    )
    surfaced = validation_sim.advance()
    validated = validate_root(
        validation_clone,
        perspective_team,
        root_target,
        surfaced,
        legal_candidates=legal_candidates,
        canonical_legal=legal_keys(surfaced),
    )
    return validated, validation_clone


def validate_search_root(
    state: GameState,
    perspective_team: TeamColor,
    root_target: RootTarget,
    legal_candidates: Sequence[Key],
    environment_policy: Agent,
    *,
    cfg: SearchConfig | None = None,
    deadline: float | None = None,
    max_advance_steps: int | None = None,
) -> ValidatedRoot:
    """Surface and validate one cloned classic-search root."""
    validated, _ = _validated_search_root_and_state(
        state,
        perspective_team,
        root_target,
        legal_candidates,
        environment_policy,
        cfg,
        deadline=deadline,
        max_advance_steps=max_advance_steps,
    )
    return validated


@_without_hypothetical_engine_info
def search(
    state: GameState,
    our_team: TeamColor,
    root_legal: Sequence[Key],
    environment_policy: Agent,
    cfg: SearchConfig,
    prior: SearchPolicy | None = None,
    *,
    root_target: RootTarget,
    cutoff_observer: CutoffObserver | None = None,
    continuation_policy: ContinuationPolicy | Agent | None = None,
    leaf_evaluator: LeafEvaluator | None = None,
) -> SearchResult:
    """Run ISMCTS anchored to ``root_target`` and return its robust child.

    The root is defined explicitly by ``root_target`` (see :class:`RootTarget`):
    kind, owned hero(s), and — for input roots — the exact ``InputRequest.id``
    plus addressed ``player_id``.

    Fail-closed semantics (never returns a zero-visit best_key, never returns
    a singleton on a stale/mismatched target):

    - ``root_legal`` empty → :class:`ValueError`.
    - Simulator can't surface the target root against a cloned state →
      :class:`RootMismatchError`.
    - ``root_legal`` disagrees with the canonical ``legal_keys(decision)`` of
      the surfaced root (different elements, or different multiplicities like
      duplicates) → :class:`RootMismatchError`. Same set in a different order
      is allowed and the caller's order is preserved for tie-breaking.

    Both the singleton and multi-key paths run root-target and legal-set
    validation against a single cloned state before returning / ranking —
    no duplicate clones. The multi-key path then runs its determinized
    iterations as usual; each iteration re-clones via ``determinize`` and
    re-validates the root anchor on its own clone (rollouts stay safe under
    hidden-info resampling), but does NOT redo the legal-set diff since
    that's a property of the caller's arguments, not the determinization.
    """
    # Leave a cooperative grace margin for the final engine/inference call and
    # result delivery before a live coordinator's hard timeout. Deterministic
    # offline configurations explicitly leave this as ``None``.
    deadline = (
        time.monotonic() + cfg.decision_timeout_seconds * 0.9
        if cfg.decision_timeout_seconds is not None
        else None
    )
    # One shared validation clone for BOTH paths: surface the root, compare
    # ``root_legal`` against the canonical legal set, and score the exact
    # surfaced root once. Iterations still build independent determinized
    # worlds inside ``_simulate``.
    validated, validated_root_state = _validated_search_root_and_state(
        state,
        our_team,
        root_target,
        root_legal,
        environment_policy,
        cfg,
        deadline=deadline,
    )
    _check_deadline(deadline)
    root_legal = validated.legal_candidates
    root_decision = cast(DecisionDescriptor, validated.decision)
    root_plan = resolve_root_search_plan(
        validated_root_state,
        root_decision,
        root_legal,
        cfg,
    )
    # Publish only the plan derived from the validated clone and canonical
    # legal candidates. Timeout telemetry must never reconstruct this from the
    # unvalidated caller state.
    _publish_root_search_plan(root_plan)

    root = Node()
    if len(root_legal) == 1:
        # Singleton root: legal set already validated above; no policy/model
        # evaluation or branching, preserving the fast forced-choice path.
        diagnostics = _root_action_diagnostics(root, root_legal, None)
        return SearchResult(
            root,
            root_legal[0],
            diagnostics,
            schedule_id=root_plan.schedule_id,
            effective_leaf_mode=root_plan.effective_leaf_mode,
            request_type=root_plan.request_type,
            semantic_role=root_plan.semantic_role,
            requested_iterations=root_plan.requested_iterations,
            effective_iterations=root_plan.effective_iterations,
            root_coverage_target=root_plan.root_coverage_target,
        )

    root_policy: PolicyScores | None = None
    root_priors: dict[Key, float] | None = None
    if prior is not None:
        root_context = SearchContext(
            root_viewer_id=root_target.decision_owner_hero_id,
            perspective_team=our_team,
            current_owner_id=root_target.decision_owner_hero_id,
            decision=root_decision,
        )
        root_policy = score_policy(prior, root_context, validated_root_state, root_legal)
        root_weights = dict(zip(root_legal, root_policy.scores, strict=True))
        root_priors = (
            root_weights
            if root_policy.semantics is ScoreSemantics.PROBABILITIES
            else _normalize_weights(root_weights, list(root_legal))
        )

    evaluator = leaf_evaluator or HeuristicLeafEvaluator()
    continuation = (
        AgentContinuationPolicy(environment_policy)
        if continuation_policy is None
        else as_continuation_policy(continuation_policy)
    )
    effective_iterations = root_plan.effective_iterations
    root_coverage_target = root_plan.root_coverage_target
    effective_cfg = replace(
        cfg,
        iterations=effective_iterations,
        leaf_mode=root_plan.effective_leaf_mode,
    )
    noop_shape = (
        contextual_noop_shape(root_decision, root_legal)
        if _uses_immediate_edge(effective_cfg) and supports_contextual_root_coverage(evaluator)
        else None
    )
    root_coverage_noop = (
        noop_shape.noop_key
        if noop_shape is not None
        and (
            noop_shape.kind in {ContextualNoopKind.RESPAWN_PASS, ContextualNoopKind.ACTION_HOLD}
            or (
                noop_shape.kind is ContextualNoopKind.NARROW_HEX_SKIP
                and effective_cfg.leaf_mode in {LeafMode.IMMEDIATE_ACTION, LeafMode.STABLE_TURN}
            )
            or (
                noop_shape.kind is ContextualNoopKind.BROAD_HEX_SKIP
                and root_coverage_target is not None
            )
        )
        else None
    )
    if root_coverage_noop is not None:
        effective_iterations = max(effective_iterations, 2)
        root_coverage_target = max(root_coverage_target or 0, 2)
        root_plan = replace(
            root_plan,
            effective_iterations=effective_iterations,
            root_coverage_target=root_coverage_target,
        )
        effective_cfg = replace(effective_cfg, iterations=effective_iterations)
        _publish_root_search_plan(root_plan)
    value_led_noop_comparison = bool(
        noop_shape is not None
        and noop_shape.kind
        in {ContextualNoopKind.NARROW_HEX_SKIP, ContextualNoopKind.BROAD_HEX_SKIP}
        and root_coverage_noop is not None
    )
    root_schedule_enabled = (
        root_policy is None or root_policy.source is not PolicyScoreSource.FALLBACK
    )
    effective_root_puct_c = _effective_root_puct_c(
        effective_cfg,
        root_schedule_enabled=root_schedule_enabled,
        value_led_noop_comparison=value_led_noop_comparison,
    )
    rng = random.Random(cfg.seed)
    try:
        for _ in range(effective_iterations):
            _check_deadline(deadline)
            try:
                _simulate(
                    root,
                    state,
                    root_target,
                    root_legal,
                    root_policy,
                    root_priors,
                    root_coverage_target,
                    root_coverage_noop,
                    value_led_noop_comparison,
                    our_team,
                    environment_policy,
                    effective_cfg,
                    rng,
                    continuation,
                    evaluator,
                    prior,
                    cutoff_observer=cutoff_observer,
                    deadline=deadline,
                )
            except SearchProgressionError as exc:
                exc.attach_root_search_plan(root_plan)
                raise
    except SearchDeadlineExceeded:
        # Back-propagation occurs only after a complete evaluation. Recover
        # useful completed visits, but never return a zero-visit guess.
        if root.visits == 0:
            raise

    # Robust child: most-visited legal root action (ties -> highest Q).
    # ``max`` iterates in the caller's order, so ties break toward earlier
    # entries — the reason we preserve caller order rather than sorting.
    def rank(key: Key) -> tuple[int, float]:
        child = root.children.get(key)
        return (child.visits, child.q) if child else (0, 0.0)

    best = max(root_legal, key=rank)
    diagnostics = _root_action_diagnostics(root, root_legal, root_priors)
    return SearchResult(
        root,
        best,
        diagnostics,
        schedule_id=root_plan.schedule_id,
        effective_leaf_mode=root_plan.effective_leaf_mode,
        request_type=root_plan.request_type,
        semantic_role=root_plan.semantic_role,
        requested_iterations=root_plan.requested_iterations,
        effective_iterations=root_plan.effective_iterations,
        root_coverage_target=root_plan.root_coverage_target,
        effective_root_puct_c=effective_root_puct_c,
    )


__all__ = [
    "CutoffObserver",
    "RootActionDiagnostic",
    "RootMismatchError",
    "RootTarget",
    "SearchAdvanceLimitExceeded",
    "SearchBudgetExceeded",
    "SearchDeadlineExceeded",
    "SearchProgressionDiagnostics",
    "SearchProgressionError",
    "SearchResult",
    "legal_keys",
    "search",
    "validate_search_root",
]
