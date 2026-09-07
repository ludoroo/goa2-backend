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

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

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
    if decision.hero is not None:
        return str(decision.hero.id)
    if decision.request is not None:
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
    ) -> None:
        self.state = state
        self.session = GameSession(state)
        self.our_team = our_team
        self.environment_policy = environment_policy
        self.owned_hero_ids = owned_hero_ids

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
        resp = pending
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
                selection = self.environment_policy.choose_input(self.state, request)
                resp = InputResponse(request_id=request.id, selection=selection)
                continue
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

    def within_cutoff() -> bool:
        if cfg.cutoff_unit is CutoffUnit.ROUNDS:
            return (sim.state.round - start_round) < cfg.cutoff_limit
        return decisions < cfg.cutoff_limit

    while (
        cfg.leaf_mode is LeafMode.BOUNDED_CONTINUATION
        and not decision.is_terminal
        and within_cutoff()
    ):
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
    leaf_context = context.for_owner(_decision_owner_id(decision, context.current_owner_id))
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
    sim = _Simulator(world, our_team, environment_policy, owned_hero_ids=root_target.owned_hero_ids)
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

    while not decision.is_terminal:
        legal = legal_keys(decision)
        if not legal:
            # Forced move (empty hand / no options): no branch, just advance.
            decision = sim.apply_ours(decision, None)
            continue

        # One policy call per node visit: its ordering drives expansion, its
        # (normalized) weights drive PUCT selection.
        owner_id = _decision_owner_id(decision, context.current_owner_id)
        current_context = context.for_owner(owner_id)
        pol = score_policy(prior, current_context, sim.state, legal) if prior is not None else None
        weights = dict(zip(legal, pol.scores, strict=True)) if pol is not None else None

        if node.should_expand(legal, cfg.widening_c, cfg.widening_alpha):
            order = (
                sorted(legal, key=lambda key: weights[key], reverse=True)
                if weights is not None
                else None
            )
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

        priors = None
        if pol is not None:
            priors = (
                weights
                if pol.semantics is ScoreSemantics.PROBABILITIES
                else _normalize_weights(weights, legal)
            )
        key = node.select(legal, cfg.uct_c, rng, priors, cfg.puct_c)
        child = node.children[key]
        node = child
        path.append(child)
        decision = sim.apply_ours(decision, key)

    if value is None:
        value = terminal_reward(decision.winner, our_team)

    for n in path:
        n.update(value)


@dataclass
class SearchResult:
    root: Node
    best_key: Key | None  # None => no real choice (forced move)


def validate_search_root(
    state: GameState,
    perspective_team: TeamColor,
    root_target: RootTarget,
    legal_candidates: Sequence[Key],
    environment_policy: Agent,
) -> ValidatedRoot:
    """Surface and validate one cloned classic-search root."""
    validation_clone = clone_state(state)
    validation_sim = _Simulator(
        validation_clone,
        perspective_team,
        environment_policy,
        owned_hero_ids=root_target.owned_hero_ids,
    )
    surfaced = validation_sim.advance()
    return validate_root(
        validation_clone,
        perspective_team,
        root_target,
        surfaced,
        legal_candidates=legal_candidates,
        canonical_legal=legal_keys(surfaced),
    )


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
    # ``root_legal`` against the canonical legal set. The clone is discarded
    # after — the multi-key path builds fresh determinized worlds per
    # iteration inside ``_simulate``.
    validated = validate_search_root(state, our_team, root_target, root_legal, environment_policy)
    root_legal = validated.legal_candidates

    root = Node()
    if len(root_legal) == 1:
        # Singleton root: legal set already validated above; no branching.
        return SearchResult(root, root_legal[0])

    evaluator = leaf_evaluator or HeuristicLeafEvaluator()
    continuation = continuation_policy or environment_policy
    rng = random.Random(cfg.seed)
    for _ in range(cfg.iterations):
        _simulate(
            root,
            state,
            root_target,
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
    return SearchResult(root, best)


__all__ = [
    "CutoffObserver",
    "RootMismatchError",
    "RootTarget",
    "SearchResult",
    "legal_keys",
    "search",
    "validate_search_root",
]
