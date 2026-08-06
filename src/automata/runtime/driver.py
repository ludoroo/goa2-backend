"""Server-neutral, one-decision-at-a-time bot driver.

Extracts the decision-ownership and application logic that used to live in the
headless :func:`~automata.runtime.harness.run_game` loop into a reusable pair of
pure(-ish) functions:

- :func:`inspect_next_decision` looks at a :class:`GameState` and the most
  recent :class:`~goa2.engine.session.SessionResult` and asks the mapped agents
  what a bot would do next. It **never mutates state**, **never falls back** to
  an arbitrary agent, and returns ``None`` whenever the next legal decision
  belongs to a human or there is nothing pending.
- :func:`apply_decision` takes the returned :class:`BotDecision` and applies it
  to a :class:`GameSession`, returning the fresh :class:`SessionResult`.

Splitting inspect from apply is what lets the server compute a bot's move
outside its locks and revalidate the world before applying — the server-side
bot coordinator uses exactly this seam. The headless harness's ``run_game``
uses both back-to-back so the two paths share semantics.

Design notes:

- **Typed state/result seam.** Callers pass the last ``SessionResult`` (or
  ``None`` at game start) and the current ``GameState``; the driver derives
  the pending :class:`~goa2.domain.input.InputRequest` internally. Callers
  never assemble the "which request is pending" seam themselves.
- **No arbitrary fallback.** If a decision is addressed to a hero/team with
  no mapped bot, we return ``None``. The old harness's ``_agent_for`` would
  fall through to any agent; that is unsafe once humans and bots share a
  game.
- **Illegal bot output raises.** A bot's ``choose_card`` returning ``None``
  while the hero still holds cards (and is not signalling FINISH via
  Emmitt's two-card window) is a bug in the agent — the driver raises
  :class:`IllegalBotDecisionError` so callers (the server bot coordinator)
  can log/quarantine that bot rather than silently downgrading to an illegal
  pass. Reserved ``None`` returns from :func:`inspect_next_decision`
  now mean *only* "no mapped owner / no current decision".
- **Emmitt's second commit.** After a two-card-capable bot commits its first
  card, the driver re-consults ``choose_card``. ``None`` there means "we're
  done planning this turn" and produces a :class:`PlanningKind.FINISH`; a
  legal card produces a second :class:`PlanningKind.COMMIT`.
- **UPGRADE_PHASE scoping.** UPGRADE_PHASE is a simultaneous request with
  choices in ``request.context['players']``. The driver copies the request
  down to only *bot-owned* heroes that still have pending upgrades before
  handing it to the responsible bot's agent, so we can never accidentally
  apply an upgrade to a human's hero.
- **Team-addressed input.** ``request.player_id`` may be ``"team:RED"`` /
  ``"simultaneous"``. Ownership resolves via team membership; the first
  bot-owned eligible responder (deterministic hero order) drives the
  answer. The driver always passes an explicit ``owned_hero_ids``
  keyword to ``Agent.choose_input`` (Agent protocol); a mixed
  human/bot team never causes the search to anchor to a still-uncommitted
  human teammate because the owned set contains only mapped bots.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from goa2.domain.input import InputRequest, InputRequestType, InputResponse, selection_value
from goa2.domain.models import GamePhase
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.phases import planning_open_for_second_card
from goa2.engine.session import GameSession, SessionResult, SessionResultType

from ..agents.base import Agent, PlanningDecision, PlanningKind, plan_from_card_choice

__all__ = [
    "BotDecision",
    "DecisionKind",
    "IllegalBotDecisionError",
    "apply_decision",
    "eligible_hero_ids_for_request",
    "inspect_next_decision",
    "inspect_next_owner",
    "legal_selection_values_for_request",
]


class DecisionKind(StrEnum):
    """Which engine surface a :class:`BotDecision` targets."""

    PLANNING = "PLANNING"
    INPUT = "INPUT"


class IllegalBotDecisionError(ValueError):
    """A mapped bot returned an output the engine would reject.

    Raised by :func:`inspect_next_decision` when:

    - ``choose_card`` returns a card that is not in the hero's hand, or
    - ``choose_card`` returns ``None`` for a hero that has cards and no open
      two-card window (only an empty hand may legally pass; only Emmitt's
      open second-card slot may legally FINISH).

    The exception carries the offending hero and a short description so
    the server bot coordinator can log, quarantine, or fall back to a safer
    bot without inspecting driver internals. It intentionally subclasses
    :class:`ValueError` so callers that already catch ``ValueError`` from
    :func:`~automata.agents.base.plan_from_card_choice` keep working.
    """

    def __init__(self, hero_id: str, reason: str) -> None:
        super().__init__(f"illegal bot decision for {hero_id!r}: {reason}")
        self.hero_id = hero_id
        self.reason = reason


@dataclass(frozen=True)
class BotDecision:
    """One decision, owned by exactly one bot agent, ready to apply.

    - ``PLANNING`` decisions carry a :class:`PlanningDecision` and target the
      hero named by :attr:`hero_id`.
    - ``INPUT`` decisions carry the resolved :attr:`request` (echoed for
      revalidation by the caller) and the raw :attr:`selection` value the
      engine expects. :attr:`hero_id` records which bot the driver used to
      compute the answer; the engine treats the answer as coming from
      ``request.player_id``.
    """

    kind: DecisionKind
    hero_id: HeroID
    planning: PlanningDecision | None = None
    request: InputRequest | None = None
    selection: Any = None

    def __post_init__(self) -> None:
        if self.kind is DecisionKind.PLANNING:
            if self.planning is None:
                raise ValueError("PLANNING BotDecision requires a PlanningDecision")
            if self.request is not None or self.selection is not None:
                raise ValueError("PLANNING BotDecision must not carry request/selection")
        else:  # INPUT
            if self.request is None:
                raise ValueError("INPUT BotDecision requires an InputRequest")
            if self.planning is not None:
                raise ValueError("INPUT BotDecision must not carry a PlanningDecision")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _all_hero_ids(state: GameState) -> list[str]:
    """Every hero id on every team, in team-then-roster order.

    Iteration order is deterministic (dict-of-teams insertion order plus each
    team's heroes list), which matters for reproducible bot-driven games and
    for stable choice of "the first eligible bot" on team-addressed input.
    """
    return [h.id for team in state.teams.values() for h in team.heroes]


def _eligible_hero_ids_for_request(
    state: GameState, request: InputRequest
) -> list[str]:
    """Hero IDs that could legitimately answer ``request``.

    - Direct hero-id addressing → just that hero (if it exists).
    - ``team:XXX`` → every hero on the matching team.
    - ``simultaneous`` (UPGRADE_PHASE etc.) → every hero listed in
      ``context['players']`` (empty context → nobody, since a truly
      simultaneous phase without listed players has no owner).
    - Anything else (unknown delegate) → empty list. We refuse to guess.
    """
    pid = request.player_id
    # Direct hero id — most common path.
    hero = state.get_hero(HeroID(pid)) if pid else None
    if hero is not None:
        return [hero.id]

    if pid.startswith("team:"):
        color = pid.split(":", 1)[1]
        out: list[str] = []
        for team in state.teams.values():
            if team.color.value == color or team.color.name == color:
                out.extend(h.id for h in team.heroes)
        return out

    if pid == "simultaneous":
        players_ctx = request.context.get("players") or {}
        return [str(hid) for hid in players_ctx]

    return []


def eligible_hero_ids_for_request(
    state: GameState, request: InputRequest
) -> list[str]:
    """Public wrapper: hero IDs that could legitimately answer ``request``.

    Server-side callers (the bot coordinator) need to recompute who is
    an eligible responder against **live** state at apply-time, in exactly
    the same way the driver's private inspection does. Sharing the private
    helper keeps ownership resolution in one place — a divergence between
    inspect-time and apply-time is the class of bug the coordinator's stale
    check was written to catch.

    For UPGRADE_PHASE specifically, the caller must additionally require
    ``request.context['players'][hero_id]['remaining'] > 0`` — a hero with
    zero remaining upgrades is listed in ``players`` but not actually a
    responder. The private inspection path applies that filter; public
    consumers should as well when they mean "still owes an answer."
    """
    return _eligible_hero_ids_for_request(state, request)


def _resolve_bot_owner(
    agents: Mapping[str, Agent], eligible_hero_ids: list[str]
) -> str | None:
    """First hero in ``eligible_hero_ids`` that has a mapped bot agent.

    Returns the hero id, or ``None`` if none of the eligible heroes are
    bot-owned. The caller uses ``None`` as the "leave it to the human" signal.
    """
    for hid in eligible_hero_ids:
        if hid in agents:
            return hid
    return None


def _filter_upgrade_request_to_bots(
    request: InputRequest, bot_hero_ids: set[str]
) -> InputRequest:
    """Copy an UPGRADE_PHASE request scoped to bot-owned pending heroes.

    UPGRADE_PHASE's choice space is ``request.context['players']`` — a mapping
    of hero id → {remaining, options}. Handing the raw request to a bot agent
    would let it pick a human's hero. We copy the request with a filtered
    ``players`` map so the agent's own ``choose_input`` only sees bot heroes.

    The original request is left untouched; the caller submits the raw
    selection dict returned by the agent (which the engine validates through
    ``ResolveUpgradesStep._is_legal_upgrade`` regardless of what we pass in).
    """
    players_ctx = dict(request.context.get("players") or {})
    scoped = {
        hid: info for hid, info in players_ctx.items() if hid in bot_hero_ids
    }
    new_context = dict(request.context)
    new_context["players"] = scoped
    return request.model_copy(update={"context": new_context})


def _call_choose_input(
    agent: Agent,
    state: GameState,
    request: InputRequest,
    owned_hero_ids: frozenset[str],
) -> Any:
    """Call ``agent.choose_input`` with the required ``owned_hero_ids`` set.

    The Agent protocol requires every implementation to accept
    ``owned_hero_ids``. We pass it unconditionally: legacy shims that would
    silently omit it are gone. If an implementation drops the kwarg it will
    raise :class:`TypeError` here — that is the correct signal for a
    protocol violation, not something to paper over.
    """
    return agent.choose_input(state, request, owned_hero_ids=owned_hero_ids)


def _pending_request_from_result(result: SessionResult | None) -> InputRequest | None:
    """Extract an :class:`InputRequest` from a session result, if any.

    Only ``INPUT_NEEDED`` carries a pending request; every other result type
    (GAME_OVER / ACTION_COMPLETE / PHASE_CHANGED) yields ``None``.
    """
    if result is None:
        return None
    if result.result_type is SessionResultType.INPUT_NEEDED:
        return result.input_request
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def inspect_next_decision(
    state: GameState,
    agents: Mapping[str, Agent],
    last_result: SessionResult | None = None,
) -> BotDecision | None:
    """Compute the next bot decision, or ``None`` when there is none.

    Parameters
    ----------
    state:
        Live (or cloned) :class:`GameState`. Never mutated.
    agents:
        Map of ``hero_id`` → :class:`Agent`. Heroes not present in this map
        are treated as humans; the driver never falls back to an arbitrary
        agent for an unmapped hero.
    last_result:
        The most recent :class:`SessionResult` from
        :meth:`GameSession.advance` (or the planning helpers). ``None``
        represents "no prior result" — typically the very first tick of a
        game before ``advance()`` has been called. The driver derives the
        pending :class:`InputRequest` from the result internally so callers
        never handle the seam.

    Returns
    -------
    A :class:`BotDecision`, or ``None`` if:

    - the game is over,
    - the pending decision is addressed to a hero/team with no bot agent,
    - PLANNING is idle for bots (every uncommitted hero is human, and every
      two-card-eligible bot has already finished).

    Raises
    ------
    IllegalBotDecisionError
        If a mapped bot returned output the engine would reject (an
        out-of-hand card, or a ``None`` card while the hero holds cards and
        has no open Emmitt two-card window).
    """
    if state.phase == GamePhase.GAME_OVER:
        return None

    pending_request = _pending_request_from_result(last_result)

    # RESOLUTION-family: we must answer the pending request or nobody moves.
    if pending_request is not None:
        return _inspect_input_request(state, agents, pending_request)

    # No pending request. Only PLANNING has bot-initiable work without one.
    if state.phase != GamePhase.PLANNING:
        return None

    return _inspect_planning(state, agents)


def inspect_next_owner(
    state: GameState,
    agents: Mapping[str, Agent],
    last_result: SessionResult | None = None,
) -> str | None:
    """Return the hero_id of the bot that owns the next decision, or ``None``.

    **Cheap** (no policy invocation): this walks the same ordering
    :func:`inspect_next_decision` uses to pick which mapped bot answers next,
    but does **not** call ``agent.choose_card`` / ``agent.choose_input``. The
    coordinator uses it to route a bounded ISMCTS decision only when the
    actual owner is an ISMCTS agent — a Heuristic/Random teammate must not
    drag the whole game into the search semaphore path.

    Ordering must stay bit-for-bit consistent with
    :func:`inspect_next_decision`:

    - **RESOLUTION-family** (pending :class:`InputRequest`): the owner is
      the first mapped bot in eligibility order. For UPGRADE_PHASE we
      additionally require the hero still owes an upgrade
      (``context['players'][hero_id]['remaining'] > 0``).
    - **PLANNING**: iterate every hero in ``_all_hero_ids`` order, return
      the first mapped bot that either (a) has an open Emmitt two-card
      window, or (b) still holds an unresolved planning commit (hand
      non-empty OR empty-hand pass owed).
    - **GAME_OVER** / no matching owner → ``None``.

    A ``None`` result guarantees :func:`inspect_next_decision` will also
    return ``None`` at the moment of the call — no bot owes work.
    """
    if state.phase == GamePhase.GAME_OVER:
        return None

    pending_request = _pending_request_from_result(last_result)
    if pending_request is not None:
        return _input_owner(state, agents, pending_request)

    if state.phase != GamePhase.PLANNING:
        return None

    return _planning_owner(state, agents)


def legal_selection_values_for_request(
    request: InputRequest, can_skip_override: bool | None = None
) -> list[Any]:
    """Enumerate the raw selection values a bot may legally submit.

    Every option is converted through :func:`goa2.domain.input.selection_value`
    — the same shape the engine's response validator expects — plus the
    literal ``"SKIP"`` sentinel when ``request.can_skip`` is true (or the
    override, used by upgrade scoping).

    Returned values are JSON-comparable (dicts / ints / strings). The
    coordinator's driver uses this list to validate a bot's chosen
    selection before it is applied through :func:`apply_decision`; illegal
    output surfaces as :class:`IllegalBotDecisionError` so the bounded
    ISMCTS path can transparently fall back to a Heuristic decision.
    """
    values: list[Any] = [selection_value(opt) for opt in request.options]
    can_skip = request.can_skip if can_skip_override is None else can_skip_override
    if can_skip:
        values.append("SKIP")
    return values


def apply_decision(session: GameSession, decision: BotDecision) -> SessionResult:
    """Apply one :class:`BotDecision` through :class:`GameSession`.

    Callers get back the fresh :class:`SessionResult` so they can immediately
    check for GAME_OVER / a new INPUT_NEEDED / a phase change. The session's
    normal validation (phase gating, hand membership, planning-done
    eligibility) still runs; illegal decisions surface as engine exceptions
    exactly as they would from a human client.
    """
    if decision.kind is DecisionKind.PLANNING:
        plan = decision.planning
        assert plan is not None  # narrowed by __post_init__
        hero_id = HeroID(decision.hero_id)
        if plan.kind is PlanningKind.COMMIT:
            assert plan.card is not None
            return session.commit_card(hero_id, plan.card)
        if plan.kind is PlanningKind.FINISH:
            return session.finish_planning(hero_id)
        # PASS
        return session.pass_turn(hero_id)

    # INPUT
    request = decision.request
    assert request is not None
    response = InputResponse(request_id=request.id, selection=decision.selection)
    return session.advance(response)


# --------------------------------------------------------------------------- #
# Internal: planning / input inspection
# --------------------------------------------------------------------------- #


def _planning_owner(
    state: GameState, agents: Mapping[str, Agent]
) -> str | None:
    """Owner of the next planning decision, without invoking any policy.

    Mirrors :func:`_inspect_planning`'s traversal but stops as soon as a
    bot-mapped hero is discovered to owe planning work — no
    ``choose_card`` call, no hand/plan validation. Because we do not
    execute the policy we cannot detect the "empty-hand-would-PASS"
    case the way :func:`_inspect_planning` does; instead we treat *any*
    unresolved planner (Emmitt second-card window, or not-yet-committed
    first commit) as a work-owing hero. That matches the coordinator's
    contract: the owner may end up finishing with an illegal decision
    (empty hand + None), in which case the bounded fallback path kicks
    in exactly as it does for any other IllegalBotDecisionError.
    """
    for hero_id in _all_hero_ids(state):
        if hero_id not in agents:
            continue
        hero = state.get_hero(HeroID(hero_id))
        if hero is None:
            continue
        # Emmitt / second commit window.
        if planning_open_for_second_card(state, HeroID(hero_id)):
            return hero_id
        # First commit not yet made (matches _inspect_planning's `continue`
        # on already-committed heroes — we do not treat them as owing).
        if hero_id in state.pending_inputs:
            continue
        return hero_id
    return None


def _input_owner(
    state: GameState, agents: Mapping[str, Agent], request: InputRequest
) -> str | None:
    """Owner of the next bot-answered input decision, without invoking policy.

    Reuses the same eligibility resolution :func:`_inspect_input_request`
    uses so hero/team/simultaneous scoping matches exactly. UPGRADE_PHASE
    is filtered to only heroes with a remaining upgrade slot; other input
    types return the first mapped bot in the eligibility order.
    """
    eligible = _eligible_hero_ids_for_request(state, request)
    if not eligible:
        return None
    if request.request_type is InputRequestType.UPGRADE_PHASE:
        players_ctx = request.context.get("players") or {}
        eligible = [
            hid
            for hid in eligible
            if players_ctx.get(hid, {}).get("remaining", 0) > 0
        ]
        if not eligible:
            return None
    return _resolve_bot_owner(agents, eligible)


def _inspect_planning(
    state: GameState, agents: Mapping[str, Agent]
) -> BotDecision | None:
    """Pick the next bot planning move (commit / finish / pass).

    Order of concerns for each hero, in team-roster order:

    1. If already in ``pending_inputs`` AND still open for a second card
       (Emmitt), the bot decides: another commit, or FINISH.
    2. If not yet in ``pending_inputs``, the bot commits or passes (empty
       hand → PASS; legal card → COMMIT; illegal decline → raise).
    3. Otherwise (already committed and closed): skip.

    Returns the first bot-owned hero with pending work, or ``None`` if every
    remaining planner is human. Raises :class:`IllegalBotDecisionError` if a
    mapped bot returns illegal output.
    """
    for hero_id in _all_hero_ids(state):
        agent = agents.get(hero_id)
        if agent is None:
            continue  # Human hero — not our decision.
        hero = state.get_hero(HeroID(hero_id))
        if hero is None:
            continue  # Roster drift — should not happen; skip defensively.

        # Two-card window still open? Emmitt after first commit.
        if planning_open_for_second_card(state, HeroID(hero_id)):
            card = agent.choose_card(state, hero)
            if card is None:
                return BotDecision(
                    kind=DecisionKind.PLANNING,
                    hero_id=HeroID(hero_id),
                    planning=PlanningDecision.finish(),
                )
            if card not in hero.hand:
                raise IllegalBotDecisionError(
                    hero_id,
                    f"choose_card returned card {card.id!r} not in hand "
                    f"({[c.id for c in hero.hand]!r})",
                )
            return BotDecision(
                kind=DecisionKind.PLANNING,
                hero_id=HeroID(hero_id),
                planning=PlanningDecision.commit(card),
            )

        # First commit not yet made.
        if hero_id in state.pending_inputs:
            continue  # Committed & closed — nothing left to do this turn.

        card = agent.choose_card(state, hero)
        if card is None:
            if hero.hand:
                # Passing with cards in hand is illegal in the engine. A bot
                # that does this is buggy — surface it rather than silently
                # skip (previous behavior) or commit something arbitrary.
                raise IllegalBotDecisionError(
                    hero_id,
                    "choose_card returned None but hero still has cards "
                    f"({[c.id for c in hero.hand]!r})",
                )
            return BotDecision(
                kind=DecisionKind.PLANNING,
                hero_id=HeroID(hero_id),
                planning=PlanningDecision.pass_(),
            )

        if card not in hero.hand:
            raise IllegalBotDecisionError(
                hero_id,
                f"choose_card returned card {card.id!r} not in hand "
                f"({[c.id for c in hero.hand]!r})",
            )

        # Sanity: this must round-trip through plan_from_card_choice. Kept as
        # a defensive check so a future refactor of plan_from_card_choice
        # (e.g. new invariants) cannot silently be bypassed here.
        _ = plan_from_card_choice(hero, card)

        return BotDecision(
            kind=DecisionKind.PLANNING,
            hero_id=HeroID(hero_id),
            planning=PlanningDecision.commit(card),
        )

    return None


def _inspect_input_request(
    state: GameState, agents: Mapping[str, Agent], request: InputRequest
) -> BotDecision | None:
    """Resolve an :class:`InputRequest` to a bot answer, or ``None``.

    ``None`` means either:

    - the request is addressed to a hero/team with no bot agent, or
    - it is UPGRADE_PHASE and no bot-owned hero has a pending upgrade.

    The engine treats the answer as coming from ``request.player_id``; the
    driver only needs to pick *which* bot's agent runs the ``choose_input``
    computation, and pass the eligible-responder set explicitly so
    search-backed agents anchor to the right hero.

    **Selection validation.** Before returning a
    :class:`BotDecision`, the driver validates the agent's raw selection
    against the *scoped* request's legal values
    (:func:`legal_selection_values_for_request`). An illegal value raises
    :class:`IllegalBotDecisionError` so the bounded coordinator (which
    catches this error type explicitly) can transparently fall back to
    the cached HeuristicAgent before any engine mutation lands. This
    stops a misbehaving ISMCTS agent from applying an out-of-options
    selection or a phantom "SKIP" on a can-skip=False request.
    """
    eligible = _eligible_hero_ids_for_request(state, request)
    if not eligible:
        return None

    # UPGRADE_PHASE needs extra scoping: only heroes with a pending upgrade
    # slot are actually eligible right now.
    if request.request_type is InputRequestType.UPGRADE_PHASE:
        players_ctx = request.context.get("players") or {}
        eligible = [
            hid
            for hid in eligible
            if players_ctx.get(hid, {}).get("remaining", 0) > 0
        ]
        if not eligible:
            return None
        bot_hero_ids = {hid for hid in eligible if hid in agents}
        if not bot_hero_ids:
            return None  # Only humans owe upgrades — leave it alone.
        # Pick a deterministic bot-owned hero as the driving agent; hand it a
        # request scoped to bot-owned pending heroes so it cannot accidentally
        # target a human.
        owner_hero_id = next(hid for hid in eligible if hid in bot_hero_ids)
        scoped_request = _filter_upgrade_request_to_bots(request, bot_hero_ids)
        agent = agents[owner_hero_id]
        selection = _call_choose_input(
            agent, state, scoped_request, frozenset(bot_hero_ids)
        )
        if selection is None:
            # Legacy UPGRADE_PHASE convention: agent may abstain this tick
            # by returning None. Validation is skipped because the driver
            # is not going to submit anything.
            return None
        _validate_input_selection(scoped_request, selection, owner_hero_id)
        return BotDecision(
            kind=DecisionKind.INPUT,
            hero_id=HeroID(owner_hero_id),
            request=request,
            selection=selection,
        )

    resolved = _resolve_bot_owner(agents, eligible)
    if resolved is None:
        return None
    owner_hero_id = resolved

    # Owned-hero set for strict search agents: bot-mapped heroes among the
    # eligible responders. For a hero-scoped request this is a singleton;
    # for a team-scoped request it is every bot on that team, so search-
    # backed agents will not anchor to a still-uncommitted human teammate.
    owned = frozenset(hid for hid in eligible if hid in agents)
    agent = agents[owner_hero_id]
    selection = _call_choose_input(agent, state, request, owned)
    _validate_input_selection(request, selection, owner_hero_id)
    return BotDecision(
        kind=DecisionKind.INPUT,
        hero_id=HeroID(owner_hero_id),
        request=request,
        selection=selection,
    )


def _validate_input_selection(
    request: InputRequest, selection: Any, hero_id: str
) -> None:
    """Raise :class:`IllegalBotDecisionError` if ``selection`` is not legal.

    A legal selection is:

    - one of the raw values produced by :func:`selection_value` on any of
      ``request.options``, or
    - the literal string ``"SKIP"`` when ``request.can_skip`` is true.

    :func:`legal_selection_values_for_request` builds this set. We compare
    on equality so JSON-shaped hex dicts (``{"q": ..., "r": ..., "s": ...}``)
    match correctly.

    **UPGRADE_PHASE / simultaneous non-branchable requests** are exempt
    from this validator because they do not surface their legal choices
    through ``request.options`` — an UPGRADE_PHASE selection is a
    ``{hero_id, card_id}`` dict validated by the engine against
    ``state.pending_upgrades`` (see
    :class:`goa2.engine.steps.cards.ResolveUpgradesStep._is_legal_upgrade`).
    Enforcing option-based validation here would be a category error and
    would reject every legal upgrade. The engine still catches illegal
    upgrades authoritatively at apply time.
    """
    if request.request_type is InputRequestType.UPGRADE_PHASE:
        return
    if not request.options and request.player_id == "simultaneous":
        # Other simultaneous / broadcast shapes (no options at all) also
        # carry their answer as free-form data validated downstream.
        return
    legal = legal_selection_values_for_request(request)
    if selection in legal:
        return
    raise IllegalBotDecisionError(
        hero_id,
        f"choose_input returned {selection!r} which is not among the "
        f"legal raw selection values for request {request.id!r} "
        f"(request_type={request.request_type.value}, can_skip="
        f"{request.can_skip})",
    )
