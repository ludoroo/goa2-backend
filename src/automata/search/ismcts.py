"""Single-perspective ISMCTS driver (cut B).

Design (see the AI layer sketch):
- **Determinization** fixes one plausible hidden world per iteration by
  resampling enemy face-down commits (`runtime.determinize`).
- **Opponents are the environment.** Every enemy decision — planning commits and
  resolution inputs — is played by a fixed *default policy* (a HeuristicAgent),
  never branched on. So every tree node is one of *our* decisions: a MAX node.
- **Depth cutoff.** A rollout stops once the round counter advances
  `cfg.cutoff_rounds` (≈ one wave), then substitutes `evaluate_state` squashed
  into [0, 1]. Terminal wins/losses map to 1.0 / 0.0 and dominate.

The engine is driven through a throwaway `GameSession` over a single clone that
is *mutated in place* as the iteration descends and rolls out — one clone per
iteration, no per-edge cloning.

Root anchoring: the search is anchored by an explicit
:class:`RootTarget` that names the decision kind (``CARD`` / ``INPUT``), the
owned hero(s), and — for input roots — the exact ``InputRequest.id``. Every
iteration validates that the simulator surfaces *exactly* that root before
descent; a mismatch (wrong hero, wrong request id, game already over, no
surfaced root) raises :class:`RootMismatchError` so callers never silently
receive an arbitrary zero-visit action.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from goa2.domain.input import InputRequest, InputResponse, selection_value
from goa2.domain.models import GamePhase, TeamColor
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.session import GameSession, SessionResultType

from ..agents.base import Agent
from ..evaluation.value import HeuristicValue, ValueFn
from ..runtime.clone import clone_state
from ..runtime.determinize import determinize
from .config import SearchConfig
from .node import Key, Node, action_key

# --------------------------------------------------------------------------- #
# Decision representation: what the engine is asking *us* for right now.
# --------------------------------------------------------------------------- #


@dataclass
class Decision:
    kind: str  # "CARD" | "INPUT" | "OVER"
    hero: Hero | None = None
    request: InputRequest | None = None
    winner: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.kind == "OVER"


# --------------------------------------------------------------------------- #
# Root anchoring: the caller names the exact decision the
# search must anchor on. The simulator must surface *this* decision, or raise.
# --------------------------------------------------------------------------- #


RootKind = Literal["CARD", "INPUT"]


class RootMismatchError(ValueError):
    """The simulator did not surface the exact requested root decision.

    Raised when ``_Simulator.advance_to_root(target)`` finds that the state's
    next actionable decision is not the one the caller anchored to — e.g. the
    wrong hero is up (root mismatch), a different / stale ``InputRequest.id``
    is active (stale), or the game is already terminal (no surfaced root).

    A subclass of :class:`ValueError` so ordinary callers can use
    ``except ValueError`` without special-casing this type, while more
    sophisticated coordinators (the server bot coordinator) can catch this
    precise class for telemetry / fallback decisions.
    """


@dataclass(frozen=True)
class RootTarget:
    """Explicit anchor for a search: the exact decision the tree's root is on.

    A ``RootTarget`` names four things:

    - ``kind`` — ``"CARD"`` or ``"INPUT"``.
    - ``owned_hero_ids`` — the heroes the bot controls; used to route
      non-root decisions (deeper in the search tree, or during rollouts) to
      either "our" MAX nodes or the default policy.
    - ``hero_id`` — set for ``kind="CARD"``. The single hero whose planning
      slot this search resolves. Must be in ``owned_hero_ids``.
    - ``request_id`` / ``player_id`` — set for ``kind="INPUT"``. The exact
      ``InputRequest.id`` at the root and the request's addressed
      ``player_id`` (used to detect team-vs-hero routing errors).

    Construct via the class methods :meth:`card` and :meth:`input` — the
    ``__post_init__`` validation rejects inconsistent combinations (empty
    ownership, CARD target whose hero is not owned, INPUT target missing a
    request id) so search code can assume a well-formed target.
    """

    kind: RootKind
    owned_hero_ids: frozenset[str]
    hero_id: str | None = None
    request_id: str | None = None
    player_id: str | None = None

    def __post_init__(self) -> None:
        if not self.owned_hero_ids:
            raise ValueError("RootTarget requires a non-empty owned_hero_ids set")
        if self.kind == "CARD":
            if self.hero_id is None:
                raise ValueError("CARD RootTarget requires hero_id")
            if self.hero_id not in self.owned_hero_ids:
                raise ValueError(
                    f"CARD RootTarget hero_id {self.hero_id!r} is not in "
                    f"owned_hero_ids {sorted(self.owned_hero_ids)!r}"
                )
            if self.request_id is not None or self.player_id is not None:
                raise ValueError("CARD RootTarget must not carry request_id/player_id")
        elif self.kind == "INPUT":
            if self.request_id is None or self.player_id is None:
                raise ValueError(
                    "INPUT RootTarget requires request_id and player_id"
                )
            if self.hero_id is not None:
                raise ValueError("INPUT RootTarget must not carry hero_id")
        else:  # pragma: no cover - Literal enforcement
            raise ValueError(f"Unknown RootTarget kind: {self.kind!r}")

    @classmethod
    def card(cls, *, hero_id: str, owned_hero_ids: frozenset[str]) -> RootTarget:
        return cls(kind="CARD", owned_hero_ids=owned_hero_ids, hero_id=hero_id)

    @classmethod
    def input(
        cls,
        *,
        request_id: str,
        player_id: str,
        owned_hero_ids: frozenset[str],
    ) -> RootTarget:
        return cls(
            kind="INPUT",
            owned_hero_ids=owned_hero_ids,
            request_id=request_id,
            player_id=player_id,
        )

    def matches(self, decision: Decision) -> bool:
        """Does ``decision`` match this target exactly?

        For INPUT roots we compare BOTH ``request_id`` and ``player_id``: a
        request with the same id but different addressing scope (hero-scoped
        vs team-scoped, or a different team) is a routing mismatch and must
        not be treated as the requested root.
        """
        if decision.kind != self.kind:
            return False
        if self.kind == "CARD":
            return decision.hero is not None and decision.hero.id == self.hero_id
        # INPUT — require both id and player_id to match.
        return (
            decision.request is not None
            and decision.request.id == self.request_id
            and decision.request.player_id == self.player_id
        )


class _PolicyResultLike(Protocol):
    """Structural view of a policy result: a best-first key ordering plus
    optional per-key prior weights (for PUCT selection)."""

    @property
    def order(self) -> list[Key]: ...

    @property
    def weights(self) -> dict[Key, float] | None: ...


class PolicyLike(Protocol):
    """A policy ranks a decision's legal keys (see ``search.prior.Policy``).

    Declared structurally here to avoid a circular import with ``prior`` (which
    imports ``Decision`` from this module). The search consumes only ``.order``.
    """

    def __call__(
        self, state: GameState, decision: Decision, legal: list[Key]
    ) -> _PolicyResultLike: ...


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

    Requests whose choices are not carried in `options` — notably the
    simultaneous, legacy-shaped `UPGRADE_PHASE` (choices live in
    `context["players"]`) — are not searchable. We defer them to the default
    policy, which knows their bespoke response shape. Searching (or forcing) an
    empty-option request would otherwise loop the engine forever.
    """
    return bool(request.options)


def legal_keys(decision: Decision) -> list[Key]:
    """Legal action keys at one of *our* decisions ([] means forced / no branch)."""
    if decision.kind == "CARD":
        hero = decision.hero
        assert hero is not None
        return [c.id for c in hero.hand]  # empty hand -> forced pass, no branch
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
        default_policy: Agent,
        *,
        owned_hero_ids: frozenset[str],
    ) -> None:
        self.state = state
        self.session = GameSession(state)
        self.our_team = our_team
        self.default_policy = default_policy
        self.owned_hero_ids = owned_hero_ids

    # -- opponent-as-environment advance ----------------------------------- #
    def _next_uncommitted(self) -> Hero | None:
        for team in self.state.teams.values():
            for hero in team.heroes:
                if hero.id not in self.state.pending_inputs:
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

    def advance(self, pending: InputResponse | None = None) -> Decision:
        """Advance until the engine needs one of *our* decisions, or ends."""
        resp = pending
        while True:
            if self.state.phase == GamePhase.PLANNING:
                hero = self._next_uncommitted()
                if hero is not None:
                    if self._is_owned_hero(hero):
                        return Decision("CARD", hero=hero)
                    # Non-owned commit (teammate or opponent) = hidden sample
                    # via the default policy so it never becomes a root.
                    card = self.default_policy.choose_card(self.state, hero)
                    if card is None or not hero.hand:
                        self.session.pass_turn(HeroID(hero.id))
                    else:
                        self.session.commit_card(HeroID(hero.id), card)
                    continue
                # All committed: fall through to advance the resolution stack.

            result = self.session.advance(resp)
            resp = None

            if result.result_type == SessionResultType.GAME_OVER:
                return Decision("OVER", winner=result.winner)
            if result.result_type == SessionResultType.INPUT_NEEDED:
                request = result.input_request
                assert request is not None
                if self._is_owned_request(request) and _branchable(request):
                    return Decision("INPUT", request=request)
                # Non-owned input, or a non-branchable request (e.g. UPGRADE_PHASE):
                # resolve with the default policy and keep advancing.
                selection = self.default_policy.choose_input(self.state, request)
                resp = InputResponse(request_id=request.id, selection=selection)
                continue
            # ACTION_COMPLETE / PHASE_CHANGED: keep advancing.

    # -- root anchoring ---------------------------------------------------- #
    def advance_to_root(self, target: RootTarget) -> Decision:
        """Advance to *exactly* the target root decision, or raise.

        Pre-validates that the target references entities that exist in the
        current state (a CARD target's hero id, an INPUT target's addressed
        hero for hero-scoped requests). This catches "bogus hero" callers up
        front so we can't spin the session forward looking for a decision
        that will never surface.

        After pre-validation, delegates to :meth:`advance` and asserts the
        surfaced :class:`Decision` matches the target. On any mismatch
        (terminal, wrong hero, stale request id, wrong kind) raises
        :class:`RootMismatchError`. Never returns a non-matching Decision.
        """
        if target.kind == "CARD":
            assert target.hero_id is not None
            if self.state.get_hero(HeroID(target.hero_id)) is None:
                raise RootMismatchError(
                    f"CARD RootTarget references hero_id {target.hero_id!r} "
                    "that does not exist in the current state"
                )
        else:  # INPUT
            assert target.player_id is not None
            pid = target.player_id
            if not pid.startswith("team:") and self.state.get_hero(HeroID(pid)) is None:
                raise RootMismatchError(
                    f"INPUT RootTarget references player_id {pid!r} that "
                    "does not exist in the current state"
                )

        decision = self.advance()
        if decision.is_terminal:
            raise RootMismatchError(
                f"expected root {target.kind} but game is already over "
                f"(winner={decision.winner!r})"
            )
        if not target.matches(decision):
            surfaced_id = (
                decision.hero.id if decision.hero is not None else None
            ) or (decision.request.id if decision.request is not None else None)
            surfaced_pid = (
                decision.request.player_id
                if decision.request is not None
                else None
            )
            raise RootMismatchError(
                f"simulator surfaced {decision.kind}("
                f"id={surfaced_id!r}, player_id={surfaced_pid!r}) but "
                f"root target was {target.kind}("
                f"hero_id={target.hero_id!r}, request_id={target.request_id!r}, "
                f"player_id={target.player_id!r})"
            )
        return decision

    # -- applying *our* action --------------------------------------------- #
    def apply_ours(self, decision: Decision, key: Key | None) -> Decision:
        """Apply our chosen action (key=None means the forced/no-branch move)."""
        if decision.kind == "CARD":
            hero = decision.hero
            assert hero is not None
            card = _find_card(hero, key) if key is not None else None
            if card is None or not hero.hand:
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


def _terminal_value(winner: str | None, our_team: TeamColor) -> float:
    if winner is None:
        return 0.5  # draw / undecided
    return 1.0 if winner.upper() == our_team.value.upper() else 0.0


def _squash(score: float, scale: float) -> float:
    """Map an unbounded evaluate_state score into (0, 1)."""
    return 0.5 * (1.0 + math.tanh(score / scale))


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
    sim: _Simulator, decision: Decision, cfg: SearchConfig, value_fn: ValueFn
) -> float:
    """Default-policy playout from `decision` until the round-count cutoff."""
    start_round = sim.state.round
    while not decision.is_terminal and (sim.state.round - start_round) < cfg.cutoff_rounds:
        if decision.kind == "CARD":
            hero = decision.hero
            assert hero is not None
            card = sim.default_policy.choose_card(sim.state, hero)
            decision = sim.apply_ours(decision, card.id if card is not None else None)
        else:
            request = decision.request
            assert request is not None
            selection = sim.default_policy.choose_input(sim.state, request)
            decision = sim.advance(InputResponse(request_id=request.id, selection=selection))
    if decision.is_terminal:
        return _terminal_value(decision.winner, sim.our_team)
    return _squash(value_fn(sim.state, sim.our_team), cfg.value_scale)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def _validate_root_legal(
    caller_legal: Sequence[Key], canonical: list[Key]
) -> None:
    """Check that ``caller_legal`` and ``canonical`` describe the same multiset.

    Same set, different order is ALLOWED — the caller's order is preserved
    downstream for policy tie-breaking (progressive widening consumes the
    prior's order; the caller's ordering becomes the deterministic secondary
    key when the prior is None or ties). Different elements, different
    multiplicities (e.g. duplicates), or size mismatch is fail-closed:
    raises :class:`RootMismatchError` (a :class:`ValueError` subclass) with a
    diff of the caller's set vs the canonical set.
    """
    from collections import Counter

    caller_counts = Counter(caller_legal)
    canonical_counts = Counter(canonical)
    if caller_counts == canonical_counts:
        return
    missing = canonical_counts - caller_counts
    extra = caller_counts - canonical_counts
    raise RootMismatchError(
        "caller root_legal disagrees with the surfaced decision's canonical "
        f"legal_keys: missing={sorted(missing.elements(), key=repr)!r}, "
        f"extra={sorted(extra.elements(), key=repr)!r} "
        f"(caller={list(caller_legal)!r}, canonical={canonical!r})"
    )


def _simulate(
    root: Node,
    root_state: GameState,
    root_target: RootTarget,
    our_team: TeamColor,
    default_policy: Agent,
    cfg: SearchConfig,
    rng: random.Random,
    value_fn: ValueFn,
    prior: PolicyLike | None = None,
) -> None:
    """One ISMCTS iteration on a fresh determinized clone (mutated in place).

    The first ``Decision`` surfaced by the simulator MUST match ``root_target``
    exactly, else :class:`RootMismatchError` is raised — the search never
    silently descends from a mismatched root.
    """
    world = determinize(root_state, our_team, rng)
    sim = _Simulator(
        world, our_team, default_policy, owned_hero_ids=root_target.owned_hero_ids
    )
    # Strict root validation: surface EXACTLY the requested root or raise.
    decision = sim.advance_to_root(root_target)

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
        pol = prior(sim.state, decision, legal) if prior is not None else None

        if node.should_expand(legal, cfg.widening_c, cfg.widening_alpha):
            order = pol.order if pol is not None else None
            key = node.expand(legal, rng, order)
            child = node.children[key]
            node = child
            path.append(child)
            decision = sim.apply_ours(decision, key)
            value = _rollout(sim, decision, cfg, value_fn)  # evaluate freshly expanded leaf
            break

        priors = _normalize_weights(pol.weights, legal) if pol is not None else None
        key = node.select(legal, cfg.uct_c, rng, priors, cfg.puct_c)
        child = node.children[key]
        node = child
        path.append(child)
        decision = sim.apply_ours(decision, key)

    if value is None:
        value = _terminal_value(decision.winner, our_team)

    for n in path:
        n.update(value)


@dataclass
class SearchResult:
    root: Node
    best_key: Key | None  # None => no real choice (forced move)


def search(
    state: GameState,
    our_team: TeamColor,
    root_decision_kind: RootKind,
    root_legal: Sequence[Key],
    default_policy: Agent,
    cfg: SearchConfig,
    prior: PolicyLike | None = None,
    value_fn: ValueFn | None = None,
    *,
    root_target: RootTarget,
) -> SearchResult:
    """Run ISMCTS anchored to ``root_target`` and return its robust child.

    The root is defined explicitly by ``root_target`` (see :class:`RootTarget`):
    kind, owned hero(s), and — for input roots — the exact ``InputRequest.id``
    plus addressed ``player_id``. ``root_decision_kind`` must agree with
    ``root_target.kind`` — it is a caller-side redundancy check, not a
    fallback.

    Fail-closed semantics (never returns a zero-visit best_key, never returns
    a singleton on a stale/mismatched target):

    - ``root_legal`` empty → :class:`ValueError`.
    - ``root_decision_kind`` disagrees with ``root_target.kind`` → :class:`ValueError`.
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
    if not root_legal:
        raise ValueError("search requires a non-empty root_legal set")
    if root_decision_kind != root_target.kind:
        raise ValueError(
            f"root_decision_kind={root_decision_kind!r} disagrees with "
            f"root_target.kind={root_target.kind!r}"
        )

    # One shared validation clone for BOTH paths: surface the root, compare
    # ``root_legal`` against the canonical legal set. The clone is discarded
    # after — the multi-key path builds fresh determinized worlds per
    # iteration inside ``_simulate``.
    validation_clone = clone_state(state)
    validation_sim = _Simulator(
        validation_clone,
        our_team,
        default_policy,
        owned_hero_ids=root_target.owned_hero_ids,
    )
    surfaced = validation_sim.advance_to_root(root_target)  # raises on mismatch
    canonical = legal_keys(surfaced)
    _validate_root_legal(root_legal, canonical)  # raises on element/count mismatch

    root = Node()
    if len(root_legal) == 1:
        # Singleton root: legal set already validated above; no branching.
        return SearchResult(root, root_legal[0])

    value = value_fn if value_fn is not None else HeuristicValue()
    rng = random.Random(cfg.seed)
    for _ in range(cfg.iterations):
        _simulate(
            root,
            state,
            root_target,
            our_team,
            default_policy,
            cfg,
            rng,
            value,
            prior,
        )

    # Robust child: most-visited legal root action (ties -> highest Q).
    # ``max`` iterates in the caller's order, so ties break toward earlier
    # entries — the reason we preserve caller order rather than sorting.
    def rank(key: Key) -> tuple[int, float]:
        child = root.children.get(key)
        return (child.visits, child.q) if child else (0, 0.0)

    best = max(root_legal, key=rank)
    return SearchResult(root, best)
