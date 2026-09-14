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
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar, cast

from automata.agents.contracts import Agent, PlanningKind
from automata.decision import DecisionDescriptor
from automata.runtime.clone import clone_state
from automata.runtime.determinize import determinize
from goa2.domain.input import InputRequest, InputResponse, selection_value
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.phases import planning_open_for_second_card
from goa2.engine.session import GameSession, SessionResultType

from ..config import SearchConfig
from ..contracts import (
    CutoffUnit,
    LeafEvaluator,
    LeafMode,
    PolicyScores,
    PolicyScoreSource,
    ScoreSemantics,
    SearchContext,
    SearchPolicy,
    score_policy,
)
from ..heuristic import HeuristicLeafEvaluator
from ..node import Key, Node, action_key
from ..root import RootMismatchError as RootMismatchError
from ..root import RootTarget, ValidatedRoot, validate_root, validate_root_decision

# --------------------------------------------------------------------------- #
# Decision representation: what the engine is asking *us* for right now.
# --------------------------------------------------------------------------- #


_P = ParamSpec("_P")
_R = TypeVar("_R")
_IN_HYPOTHETICAL_SEARCH: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "automata_in_hypothetical_search", default=False
)


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


def _decision_owner_id(decision: DecisionDescriptor, fallback: str) -> str:
    """Return a concrete hero owner for observation/value context.

    Team-scoped requests identify who may answer, not a hero entity. Keep the
    previous concrete owner chosen by the coordinator/search root for those
    requests so learned observations can mark a real decision-owning hero.
    """
    if decision.hero is not None:
        return str(decision.hero.id)
    if decision.request is not None and not decision.request.player_id.startswith("team:"):
        return decision.request.player_id
    return fallback


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
    ) -> None:
        self.state = state
        self.session = GameSession(state)
        self.our_team = our_team
        self.environment_policy = environment_policy
        self.owned_hero_ids = owned_hero_ids
        self.cfg = cfg or SearchConfig()
        self._advance_calls = 0
        self._advance_transitions = 0
        self._session_advances = 0
        self._environment_planning = 0
        self._environment_inputs = 0
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

    def _record_transition(self, kind: str, decision: DecisionDescriptor | None = None) -> None:
        self._current_advance_transitions += 1
        self._advance_transitions += 1
        if kind == "session":
            self._session_advances += 1
        elif kind == "planning":
            self._environment_planning += 1
        else:
            self._environment_inputs += 1
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

    def advance(self, pending: InputResponse | None = None) -> DecisionDescriptor:
        """Advance until the engine needs one of *our* decisions, or ends."""
        self._advance_calls += 1
        self._current_advance_transitions = 0
        resp = pending
        seen_planning: set[tuple[object, ...]] = set()
        seen_environment_inputs: set[tuple[object, ...]] = set()
        while True:
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
            result = self.session.advance(resp)
            resp = None

            if result.result_type == SessionResultType.GAME_OVER:
                return DecisionDescriptor("OVER", winner=result.winner)
            if result.result_type == SessionResultType.INPUT_NEEDED:
                request = result.input_request
                assert request is not None
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
    def apply_ours(self, decision: DecisionDescriptor, key: Key | None) -> DecisionDescriptor:
        """Apply our chosen action (key=None means the forced/no-branch move)."""
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
            return self.advance()

        # INPUT
        request = decision.request
        assert request is not None
        if key is None:
            selection = "SKIP" if request.can_skip else None
        else:
            selection = _input_raw_map(request).get(key, key)
        return self.advance(InputResponse(request_id=request.id, selection=selection))


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


def _rollout(
    sim: _Simulator,
    decision: DecisionDescriptor,
    cfg: SearchConfig,
    continuation_policy: Agent,
    leaf_evaluator: LeafEvaluator,
    context: SearchContext,
    *,
    cutoff_observer: CutoffObserver | None = None,
) -> float:
    """Continuation-policy playout until the configured cutoff."""
    start_round = sim.state.round
    decisions = 0
    forced_decisions = 0
    seen_forced_decisions: set[tuple[object, ...]] = set()

    def within_cutoff() -> bool:
        if cfg.cutoff_unit is CutoffUnit.ROUNDS:
            return (sim.state.round - start_round) < cfg.cutoff_limit
        return decisions < cfg.cutoff_limit

    while (
        cfg.leaf_mode is LeafMode.BOUNDED_CONTINUATION
        and not decision.is_terminal
        and within_cutoff()
    ):
        context = context.for_decision(
            decision,
            owner_id=_decision_owner_id(decision, context.current_owner_id),
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
        if decision.kind == "CARD":
            hero = decision.hero
            assert hero is not None
            planning = continuation_policy.choose_planning(sim.state, hero)
            decision = sim.apply_ours(
                decision,
                (
                    planning.card.id
                    if planning.kind is PlanningKind.COMMIT and planning.card
                    else None
                ),
            )
        else:
            request = decision.request
            assert request is not None
            selection = continuation_policy.choose_input(sim.state, request)
            decision = sim.advance(InputResponse(request_id=request.id, selection=selection))
        decisions += 1
    if decision.is_terminal:
        return terminal_reward(decision.winner, sim.our_team)
    leaf_context = context.for_decision(
        decision,
        owner_id=_decision_owner_id(decision, context.current_owner_id),
    )
    active_value = leaf_evaluator.evaluate(leaf_context, sim.state).value
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
    our_team: TeamColor,
    environment_policy: Agent,
    cfg: SearchConfig,
    rng: random.Random,
    continuation_policy: Agent,
    leaf_evaluator: LeafEvaluator,
    prior: SearchPolicy | None = None,
    *,
    cutoff_observer: CutoffObserver | None = None,
) -> None:
    """One ISMCTS iteration on a fresh determinized clone (mutated in place).

    The first ``Decision`` surfaced by the simulator MUST match ``root_target``
    exactly, else :class:`RootMismatchError` is raised — the search never
    silently descends from a mismatched root.
    """
    world = determinize(root_state, root_target.decision_owner_hero_id, rng)
    sim = _Simulator(
        world,
        our_team,
        environment_policy,
        owned_hero_ids=root_target.owned_hero_ids,
        cfg=cfg,
    )
    # Strict root validation: surface EXACTLY the requested root or raise.
    decision = sim.advance_to_root(root_target)
    context = SearchContext(
        root_viewer_id=root_target.decision_owner_hero_id,
        perspective_team=our_team,
        current_owner_id=root_target.decision_owner_hero_id,
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
            owner_id=_decision_owner_id(decision, context.current_owner_id),
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

        # Adaptive broad-HEX roots deterministically cover the best-prior
        # actions before returning to the configured PUCT/widening policy. A
        # stable sort preserves caller order for equal priors; with no prior,
        # caller order itself is the deterministic expansion order.
        if is_root and root_coverage_target is not None:
            visited = sum(child.visits > 0 for child in node.children.values())
            if visited < root_coverage_target:
                coverage_order = order if order is not None else legal
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
                decision = sim.apply_ours(decision, key)
                value = _rollout(
                    sim,
                    decision,
                    cfg,
                    continuation_policy,
                    leaf_evaluator,
                    current_context,
                    cutoff_observer=cutoff_observer,
                )
                break

        if node.should_expand(legal, widen_c, widen_alpha):
            key = node.expand(legal, rng, order)
            child = node.children[key]
            node = child
            path.append(child)
            decision = sim.apply_ours(decision, key)
            value = _rollout(
                sim,
                decision,
                cfg,
                continuation_policy,
                leaf_evaluator,
                current_context,
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
            cfg.root_puct_c if use_root_schedule and cfg.root_puct_c is not None else cfg.puct_c
        )
        key = node.select(legal, cfg.uct_c, rng, priors, puct_c)
        child = node.children[key]
        node = child
        path.append(child)
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


def _adaptive_hex_root_schedule(cfg: SearchConfig, legal: Sequence[Key]) -> tuple[int, int | None]:
    """Return the effective iteration budget and optional root coverage target.

    Version 1 applies only when every legal action is a canonical HEX key,
    except for an optional ``SKIP`` sentinel, and the root has more than eight
    actions. It retains all actions; the target controls visitation only.
    """
    count = len(legal)
    is_hex_or_skip = all(
        key == "SKIP" or (isinstance(key, tuple) and len(key) == 4 and key[0] == "hex")
        for key in legal
    )
    has_hex = any(isinstance(key, tuple) and len(key) == 4 and key[0] == "hex" for key in legal)
    if (
        cfg.adaptive_hex_root_schedule_version != 1
        or count <= 8
        or not is_hex_or_skip
        or not has_hex
    ):
        return cfg.iterations, None

    coverage_target = min(count, 12, max(4, math.ceil(math.sqrt(count))))
    return max(cfg.iterations, 2 * coverage_target), coverage_target


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
) -> tuple[ValidatedRoot, GameState]:
    validation_clone = clone_state(state)
    validation_sim = _Simulator(
        validation_clone,
        perspective_team,
        environment_policy,
        owned_hero_ids=root_target.owned_hero_ids,
        cfg=cfg,
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
) -> ValidatedRoot:
    """Surface and validate one cloned classic-search root."""
    validated, _ = _validated_search_root_and_state(
        state, perspective_team, root_target, legal_candidates, environment_policy, cfg
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
    continuation_policy: Agent | None = None,
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
    # One shared validation clone for BOTH paths: surface the root, compare
    # ``root_legal`` against the canonical legal set, and score the exact
    # surfaced root once. Iterations still build independent determinized
    # worlds inside ``_simulate``.
    validated, validated_root_state = _validated_search_root_and_state(
        state, our_team, root_target, root_legal, environment_policy, cfg
    )
    root_legal = validated.legal_candidates

    root = Node()
    if len(root_legal) == 1:
        # Singleton root: legal set already validated above; no policy/model
        # evaluation or branching, preserving the fast forced-choice path.
        diagnostics = _root_action_diagnostics(root, root_legal, None)
        return SearchResult(
            root,
            root_legal[0],
            diagnostics,
            requested_iterations=cfg.iterations,
            effective_iterations=0,
        )

    root_policy: PolicyScores | None = None
    root_priors: dict[Key, float] | None = None
    if prior is not None:
        root_decision = cast(DecisionDescriptor, validated.decision)
        root_context = SearchContext(
            root_viewer_id=root_target.decision_owner_hero_id,
            perspective_team=our_team,
            current_owner_id=root_target.decision_owner_hero_id,
            current_decision=root_decision,
        )
        root_policy = score_policy(prior, root_context, validated_root_state, root_legal)
        root_weights = dict(zip(root_legal, root_policy.scores, strict=True))
        root_priors = (
            root_weights
            if root_policy.semantics is ScoreSemantics.PROBABILITIES
            else _normalize_weights(root_weights, list(root_legal))
        )

    evaluator = leaf_evaluator or HeuristicLeafEvaluator()
    continuation = continuation_policy or environment_policy
    effective_iterations, root_coverage_target = _adaptive_hex_root_schedule(cfg, root_legal)
    rng = random.Random(cfg.seed)
    for _ in range(effective_iterations):
        _simulate(
            root,
            state,
            root_target,
            root_legal,
            root_policy,
            root_priors,
            root_coverage_target,
            our_team,
            environment_policy,
            cfg,
            rng,
            continuation,
            evaluator,
            prior,
            cutoff_observer=cutoff_observer,
        )

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
        requested_iterations=cfg.iterations,
        effective_iterations=effective_iterations,
        root_coverage_target=root_coverage_target,
    )


__all__ = [
    "CutoffObserver",
    "RootActionDiagnostic",
    "RootMismatchError",
    "RootTarget",
    "SearchProgressionDiagnostics",
    "SearchProgressionError",
    "SearchResult",
    "legal_keys",
    "search",
    "validate_search_root",
]
