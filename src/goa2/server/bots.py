"""Server-side bot coordinator.

One idempotent asynchronous worker per game drives every server-managed AI
hero's decisions through the same engine, persistence, replay, clock, and
broadcast paths as human players.

Invariants (the reason the coordinator exists rather than reusing headless
harness code):

- **One task per game.** Stored in ``ManagedGame.bot_task``.
  :func:`schedule_bot_drive` is safe from any lifecycle site — it is a no-op
  while the previous task is alive, and spawns a fresh one after it exits.
- **Never hold locks during compute.** The worker snapshots the state under
  ``game.lock``, **deep-copies** the pending :class:`SessionResult` /
  :class:`InputRequest` and clones the state, then hands the isolated
  objects to :func:`asyncio.to_thread`. A misbehaving or malicious agent
  cannot mutate live objects, and no lock is held while the CPU work runs.
- **Established lock order on apply.** ``outbound_lock`` → ``game.lock``,
  matching every REST/WS mutation.
- **One decision per locked mutation.** Applied through :class:`GameSession`;
  the engine keeps authority. Stale results are silently dropped after a
  live revalidation that recomputes eligible responders — no half-applied
  mutations, no persistence writes for stale outputs, no phantom broadcasts.
- **Plain ``advance()`` when the bot owns the next work but the engine is
  between requests.** A bot-vs-bot game must be able to resolve mid-turn
  actions without a human/timer nudge. When
  :func:`~automata.runtime.driver.inspect_next_decision` returns ``None``
  during RESOLUTION, the coordinator issues one plain ``session.advance()``
  through the same locked mutation → finalize → save → broadcast path a
  human's REST/WS advance would take. It only exits to a human when the
  live pending request is addressed to a hero/team the game has no bot
  agent for.
- **Recoverable on any failure.** Agent exceptions, illegal engine outputs,
  broadcast errors — the worker logs and exits with the live state
  untouched. Every clock stop is paired with a ``finalize_timed_mutation``
  in ``finally`` so time-control state never drifts.
- **Agent instances live on the ManagedGame.** They are runtime-only (never
  persisted), created lazily with stable game-specific entropy, and cleared
  on ``registry.remove()`` and after every game restore.
- **Live ISMCTS is bounded.** Search runs off-loop with owner-scoped timeouts,
  process-wide concurrency limits, and a per-hero Heuristic fallback.

REST, WebSocket, timer, and restore lifecycle sites call
:func:`schedule_bot_drive` through the shared server mutation seams.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from automata.agents.base import Agent
from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.random_agent import RandomAgent
from automata.runtime.clone import clone_state
from automata.runtime.driver import (
    BotDecision,
    DecisionKind,
    IllegalBotDecisionError,
    apply_decision,
    eligible_hero_ids_for_request,
    inspect_next_decision,
    inspect_next_owner,
)
from automata.search.agent import ISMCTSAgent
from automata.search.config import (
    PROD_QUEUE_TIMEOUT_SECONDS,
    PROD_SEARCH_CONCURRENCY,
    SearchConfig,
)
from goa2.domain.input import InputRequest, InputRequestType
from goa2.domain.models import GamePhase
from goa2.domain.state import GameState
from goa2.domain.time_control import ClockStatus
from goa2.domain.types import HeroID
from goa2.engine.phases import planning_open_for_second_card
from goa2.engine.session import GameSession, SessionResult, SessionResultType
from goa2.server.bot_models import BotSpec, SearchSettings
from goa2.server.registry import GameRegistry, ManagedGame
from goa2.server.time_control import (
    finalize_timed_mutation,
    now_ms,
    set_player_ready,
    stop_clock_for_accepted_decision,
)

logger = logging.getLogger(__name__)

__all__ = [
    "agent_for_spec",
    "auto_ready_bot_heroes",
    "cancel_all_bot_tasks",
    "get_or_build_agents",
    "schedule_bot_drive",
    "start_bot_lifecycle",
]


# --------------------------------------------------------------------------- #
# Bounded ISMCTS execution                                                    #
# --------------------------------------------------------------------------- #
#
# ISMCTS is CPU-bound and can spend hundreds of ms per decision. Running it
# on the event-loop thread would freeze every other coroutine (WebSocket
# broadcasts, REST responses, deadline timers). The coordinator therefore
# runs *every* ISMCTS-involving ``inspect_next_decision`` call inside
# :func:`asyncio.to_thread`, guarded by:
#
# 1. A **process-wide asyncio semaphore** (``_ISMCTS_SEMAPHORE``) sized to
#    :data:`PROD_SEARCH_CONCURRENCY`. Callers queue on the semaphore up to
#    :data:`PROD_QUEUE_TIMEOUT_SECONDS`; a queue timeout is a fallback
#    trigger (not an error).
# 2. An **owner-scoped search timeout** derived from the acting bot's
#    :attr:`SearchSettings.decision_timeout_seconds`. Teammates' budgets do
#    not affect the current decision.
# 3. A **stale-safe release**. The underlying thread runs on an
#    ``asyncio.Future`` returned by :meth:`loop.run_in_executor`; the
#    semaphore is released via ``future.add_done_callback`` when the
#    thread finally exits, NOT when the caller's ``asyncio.wait_for``
#    fires. That means a timed-out search keeps its slot until it really
#    finishes, so concurrency stays honest, and its result is dropped
#    (see below).
# 4. **No late apply**. The compute runs on a deep clone of the state /
#    request (already the coordinator's invariant), so a slow search
#    cannot mutate live objects. If the caller falls back before the
#    search finishes, the future's eventual result is discarded — the
#    fallback decision has already been applied by then.
#
# Structured logging (``bots.ismcts``) emits counters and latencies for
# every observable outcome (queue wait, search latency, timeout, error,
# fallback). We deliberately never log state contents or agent internals.
#
# The semaphore is module-level so it survives across games and across
# request boundaries — this is intentional. It is initialized lazily on
# first use so a test module that imports :mod:`goa2.server.bots` without
# a running event loop does not crash at import time.

_ismcts_semaphore: asyncio.Semaphore | None = None
_ismcts_semaphore_loop: asyncio.AbstractEventLoop | None = None

# Module-level tracker of every in-flight bounded-search future. Used by
# :func:`cancel_all_bot_tasks` on app shutdown to drain outstanding
# executor work before returning — a shutdown that leaves a background
# thread mid-search would leak a semaphore slot into the next process
# and could still (in tests) mutate the shared metrics dict. The
# per-game :attr:`ManagedGame._bot_search_futures` tracker holds the
# same futures scoped to one game so ``registry.remove`` can observe
# them locally.
_in_flight_search_futures: set[asyncio.Future[Any]] = set()


def _get_ismcts_semaphore() -> asyncio.Semaphore:
    """Return the process-wide ISMCTS concurrency semaphore.

    Lazy-initialized on first call so importing this module doesn't require
    a running event loop. If the loop changes across tests (each
    ``asyncio.run`` uses a fresh loop), rebuild the semaphore against the
    new loop — an :class:`asyncio.Semaphore` bound to a dead loop would
    silently deadlock.
    """
    global _ismcts_semaphore, _ismcts_semaphore_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop yet; the semaphore will be built on first use
        # from an actual coroutine. Return an ephemeral one for typing.
        if _ismcts_semaphore is not None:
            return _ismcts_semaphore
        _ismcts_semaphore = asyncio.Semaphore(PROD_SEARCH_CONCURRENCY)
        _ismcts_semaphore_loop = None
        return _ismcts_semaphore
    if _ismcts_semaphore is None or _ismcts_semaphore_loop is not loop:
        _ismcts_semaphore = asyncio.Semaphore(PROD_SEARCH_CONCURRENCY)
        _ismcts_semaphore_loop = loop
    return _ismcts_semaphore


@dataclass
class _IsmctsRunStats:
    """Structured, non-secret observability counters for one run.

    Exposed via module-level :data:`ismcts_metrics` (a live counter map) so
    tests and operators can observe steady-state behavior without parsing
    log lines. Fields deliberately carry only counters and latencies — no
    state, agent identifiers, or request payloads.
    """

    queue_wait_seconds: float = 0.0
    search_seconds: float = 0.0
    fell_back: bool = False
    fallback_reason: str = ""  # "queue_timeout" | "search_timeout" | "error" | ""


@dataclass
class _IsmctsMetrics:
    """Live counters observable by tests and operators.

    Every counter here is safe to log (no secrets, no state contents). The
    coordinator increments these atomically-enough for a single event loop —
    Python bytecode ``+= 1`` is not truly atomic across threads, but every
    increment happens on the event-loop thread.
    """

    total_calls: int = 0
    fallback_queue_timeout: int = 0
    fallback_search_timeout: int = 0
    fallback_error: int = 0
    fallback_invalid_decision: int = 0
    total_queue_wait_seconds: float = 0.0
    total_search_seconds: float = 0.0
    current_queue_depth: int = 0
    peak_queue_depth: int = 0
    late_completions: int = 0  # search finished after caller fell back


ismcts_metrics = _IsmctsMetrics()


def reset_ismcts_metrics() -> None:
    """Test hook: zero every counter in :data:`ismcts_metrics`."""
    global ismcts_metrics
    ismcts_metrics = _IsmctsMetrics()


def _is_ismcts_agent(agent: Agent) -> bool:
    """Whether ``agent`` should be routed through the bounded ISMCTS path.

    Extracted to a module-level function so test suites can monkeypatch a
    duck-typed stub in without subclassing :class:`ISMCTSAgent` (whose
    constructor is heavy). Every ISMCTS-routing predicate in this module
    (owner routing, timeout derivation, fallback swap) consults this one
    function so a test-side override keeps the "which agents are treated
    as ISMCTS" answer consistent everywhere.
    """
    return isinstance(agent, ISMCTSAgent)


def _timeout_for_owner(
    owner_hero_id: str,
    agents: dict[str, Agent],
    specs: dict[str, BotSpec],
) -> float:
    """Decision timeout for the *actual* mapped owner of the next decision.

    The coordinator uses this instead of "minimum timeout across all
    ISMCTS bots" because only the owner's spec should dictate the bound
    for this decision. A Random / Heuristic teammate must never drag an
    ISMCTS owner's compute into a tighter budget, and — critically — the
    presence of an ISMCTS *teammate* must never drag a Heuristic owner
    into the semaphore path (see :func:`_bounded_inspect_next_decision`).

    Defensive fallbacks: if the owner is not in ``agents`` or is not an
    ISMCTS agent, we return the :class:`SearchSettings` default — the
    caller is expected to have already checked ownership and skipped the
    bounded path.
    """
    agent = agents.get(owner_hero_id)
    if agent is None or not _is_ismcts_agent(agent):
        return SearchSettings().decision_timeout_seconds
    spec = specs.get(owner_hero_id)
    if spec is None or spec.search is None:
        return SearchSettings().decision_timeout_seconds
    return spec.search.decision_timeout_seconds


def _fallback_agent_for_hero(game: ManagedGame, hero_id: str) -> Agent:
    """Return the cached :class:`HeuristicAgent` fallback for ``hero_id``.

    Fallbacks are cached **per hero**, not globally per
    game. Two ISMCTS bots on the same team must each have their own
    fallback Heuristic instance so a fallback triggered on hero A does
    not advance hero B's RNG stream. The seed derivation is stable
    across restarts:

        seed(hero) = _game_entropy(game_id) ^ sha1(hero_id)[:8]

    The SHA-1-derived salt is deterministic across processes and keeps each
    hero's fallback RNG independent while retaining game-specific entropy.

    The cache lives on :attr:`ManagedGame._bot_fallback_agents` (runtime
    only, cleared by :meth:`GameRegistry.remove`).
    """
    cache = game._bot_fallback_agents
    if cache is None:
        cache = {}
        game._bot_fallback_agents = cache
    existing = cache.get(hero_id)
    if existing is not None:
        return existing
    # Combine game entropy with a stable per-hero salt so two heroes on
    # the same game get distinct streams. ``hero_id`` is a short stable
    # string; SHA-1 gives us a deterministic derivation independent of
    # Python's hash randomization.
    salt_bytes = hashlib.sha1(hero_id.encode("utf-8")).digest()[:8]
    salt = int.from_bytes(salt_bytes, "big") & ((1 << 63) - 1)
    seed = _game_entropy(game.game_id) ^ salt
    fallback: Agent = HeuristicAgent(seed=seed)
    cache[hero_id] = fallback
    return fallback


def _fallback_agents(agents: dict[str, Agent], game: ManagedGame) -> dict[str, Agent]:
    """Return an ``agents`` copy with every ISMCTS agent replaced by a
    per-hero fallback :class:`HeuristicAgent`.

    Random / Heuristic bots on the same team keep their own agent
    instances. Each replaced ISMCTS agent gets its own seeded
    :class:`HeuristicAgent` (see :func:`_fallback_agent_for_hero`), so
    there is no shared RNG coupling between fallbacks triggered on
    different heroes.
    """
    return {
        hero_id: (
            _fallback_agent_for_hero(game, hero_id)
            if _is_ismcts_agent(agent)
            else agent
        )
        for hero_id, agent in agents.items()
    }


async def _bounded_inspect_next_decision(
    game: ManagedGame,
    cloned_state: GameState,
    agents: dict[str, Agent],
    cloned_last_result: SessionResult | None,
) -> BotDecision | None:
    """Run :func:`inspect_next_decision` off the event loop with bounds.

    Contract:

    - **Owner-scoped routing.** :func:`inspect_next_owner` picks the actual
      mapped bot that will answer next (no policy is invoked); we take the
      bounded ISMCTS path *only* when that specific owner is an ISMCTS
      agent. A Heuristic / Random owner bypasses the semaphore even when
      the game has an ISMCTS teammate, so an ISMCTS bot cannot slow down
      unrelated turns. The timeout is derived only from that owner's
      :class:`BotSpec`.
    - **Never runs on the event-loop thread.** Compute is dispatched via
      :func:`asyncio.get_running_loop().run_in_executor`, which returns an
      :class:`asyncio.Future` we can watch independently from the
      semaphore-holding coroutine.
    - **Queue wait bounded** by :data:`PROD_QUEUE_TIMEOUT_SECONDS`. A queue
      timeout is a fallback trigger, not an error.
    - **Search bounded** by the owner's ``decision_timeout_seconds``. A
      search timeout is a fallback trigger.
    - **Late completion never applies.** The caller falls back immediately;
      the still-running future is left to complete on its executor thread
      and its eventual result is discarded. Because the search operates on
      a deep-cloned state / request, no live server object is at risk.
    - **Semaphore capacity retained until the thread actually finishes.**
      The semaphore is released from ``future.add_done_callback``, not from
      the calling coroutine's ``wait_for`` path — so a timed-out slow
      search still occupies a slot, keeping concurrency honest.
    - **In-flight future tracking.** Every dispatched future is registered
      in the module-level :data:`_in_flight_search_futures` set and on
      ``ManagedGame._bot_search_futures``; the done-callback cleans both
      up. :func:`cancel_all_bot_tasks` drains tracked futures up to its
      shutdown timeout; any still-running thread remains tracked and retains
      its semaphore slot until natural completion.

    When the owner is not ISMCTS, this function is a thin
    ``asyncio.to_thread`` wrapper without bounds.
    """
    owner_hero_id = inspect_next_owner(cloned_state, agents, cloned_last_result)
    owner_agent = agents.get(owner_hero_id) if owner_hero_id is not None else None

    # Owner-based routing: only take the bounded path if the *specific*
    # bot answering next is ISMCTS. A Heuristic teammate of an ISMCTS bot
    # must not be dragged through the semaphore for its own turn.
    if owner_agent is None or not _is_ismcts_agent(owner_agent):
        return await asyncio.to_thread(
            inspect_next_decision, cloned_state, agents, cloned_last_result
        )

    assert owner_hero_id is not None  # narrowed by the owner_agent check
    ismcts_metrics.total_calls += 1
    search_timeout = _timeout_for_owner(owner_hero_id, agents, game.bot_specs)

    sem = _get_ismcts_semaphore()

    # ---- Queue wait ------------------------------------------------------- #
    ismcts_metrics.current_queue_depth += 1
    ismcts_metrics.peak_queue_depth = max(
        ismcts_metrics.peak_queue_depth, ismcts_metrics.current_queue_depth
    )
    queue_start = time.monotonic()
    try:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=PROD_QUEUE_TIMEOUT_SECONDS)
        except TimeoutError:
            queue_wait = time.monotonic() - queue_start
            ismcts_metrics.total_queue_wait_seconds += queue_wait
            ismcts_metrics.fallback_queue_timeout += 1
            logger.info(
                "ismcts: fallback=queue_timeout game=%s owner=%s queue_wait=%.3fs",
                game.game_id,
                owner_hero_id,
                queue_wait,
            )
            return await _fallback_inspect(
                game, cloned_state, agents, cloned_last_result
            )
    finally:
        ismcts_metrics.current_queue_depth -= 1
    queue_wait = time.monotonic() - queue_start
    ismcts_metrics.total_queue_wait_seconds += queue_wait

    # ---- Submit off-loop compute; semaphore released via done_callback ---- #
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(
        None,
        _run_inspect_next_decision,
        cloned_state,
        agents,
        cloned_last_result,
    )
    # Register with both the module-level tracker (for shutdown drain) and
    # the per-game tracker (for registry.remove teardown observability).
    _in_flight_search_futures.add(fut)
    game._bot_search_futures.add(fut)
    search_started_at = time.monotonic()
    finished_flag: dict[str, bool] = {"finished": False}

    def _release(_f: asyncio.Future[Any]) -> None:
        # Release the semaphore only when the underlying thread has
        # actually completed. This is the "retain capacity until the
        # thread actually completes" invariant — a caller-side wait_for
        # timeout must NOT release the slot early.
        try:
            sem.release()
        except Exception:  # defensive: release is normally safe
            logger.exception("ismcts: semaphore release raised (game=%s)", game.game_id)
        # De-register from both trackers. ``discard`` is safe if a test or
        # a shutdown drain has already popped the future.
        _in_flight_search_futures.discard(_f)
        game._bot_search_futures.discard(_f)
        # If the caller already gave up (finished_flag stays False after
        # wait_for), this is a "late completion" — record it.
        if not finished_flag["finished"]:
            ismcts_metrics.late_completions += 1
            logger.info(
                "ismcts: late_completion (dropped) game=%s owner=%s search_wall=%.3fs",
                game.game_id,
                owner_hero_id,
                time.monotonic() - search_started_at,
            )

    fut.add_done_callback(_release)

    # ---- Bounded wait ----------------------------------------------------- #
    try:
        decision = await asyncio.wait_for(
            asyncio.shield(fut),
            timeout=search_timeout,
        )
    except TimeoutError:
        # Timed out. Do NOT cancel the future — it holds the semaphore
        # slot and its state/request are private clones, so letting it
        # run to completion is safe. The done_callback releases the slot
        # when it eventually finishes; we fall back on the caller side.
        search_wait = time.monotonic() - search_started_at
        ismcts_metrics.fallback_search_timeout += 1
        logger.info(
            "ismcts: fallback=search_timeout game=%s owner=%s search_wait=%.3fs timeout=%.3fs",
            game.game_id,
            owner_hero_id,
            search_wait,
            search_timeout,
        )
        return await _fallback_inspect(
            game, cloned_state, agents, cloned_last_result
        )
    except asyncio.CancelledError:
        # The outer bot task was cancelled (shutdown / registry.remove).
        # Do NOT cancel the executor future — it may still be doing work
        # on a clone; letting it finish keeps the semaphore + tracker
        # invariants honest and its result is dropped by the caller
        # anyway. Re-raise so the outer worker exits cleanly.
        raise
    except IllegalBotDecisionError as exc:
        finished_flag["finished"] = True
        search_wait = time.monotonic() - search_started_at
        ismcts_metrics.total_search_seconds += search_wait
        ismcts_metrics.fallback_invalid_decision += 1
        logger.warning(
            "ismcts: fallback=invalid_decision game=%s owner=%s reason=%s search_wall=%.3fs",
            game.game_id,
            owner_hero_id,
            exc.reason,
            search_wait,
        )
        return await _fallback_inspect(
            game, cloned_state, agents, cloned_last_result
        )
    except Exception:
        finished_flag["finished"] = True
        search_wait = time.monotonic() - search_started_at
        ismcts_metrics.total_search_seconds += search_wait
        ismcts_metrics.fallback_error += 1
        logger.exception(
            "ismcts: fallback=error game=%s owner=%s search_wall=%.3fs",
            game.game_id,
            owner_hero_id,
            search_wait,
        )
        return await _fallback_inspect(
            game, cloned_state, agents, cloned_last_result
        )

    finished_flag["finished"] = True
    search_wait = time.monotonic() - search_started_at
    ismcts_metrics.total_search_seconds += search_wait
    logger.debug(
        "ismcts: ok game=%s owner=%s queue_wait=%.3fs search_wall=%.3fs",
        game.game_id,
        owner_hero_id,
        queue_wait,
        search_wait,
    )
    return decision


def _run_inspect_next_decision(
    cloned_state: GameState,
    agents: dict[str, Agent],
    cloned_last_result: SessionResult | None,
) -> BotDecision | None:
    """Thin wrapper for ``run_in_executor``.

    Exists as a named function so tests can monkeypatch it to inject
    canned latencies or errors without depending on ``asyncio.to_thread``
    internals.
    """
    return inspect_next_decision(cloned_state, agents, cloned_last_result)


async def _fallback_inspect(
    game: ManagedGame,
    cloned_state: GameState,
    agents: dict[str, Agent],
    cloned_last_result: SessionResult | None,
) -> BotDecision | None:
    """Recompute the decision with a HeuristicAgent substituted for ISMCTS.

    The fallback still runs on a background thread (Heuristic is cheap but
    should not block the event loop as a matter of policy), and it only
    substitutes for ISMCTS agents — Random / Heuristic bots on the same
    team keep their own agent instances. The cloned state/request are the
    same objects that were handed to the original compute; the caller
    guarantees they are not shared with live state.
    """
    fallback_map = _fallback_agents(agents, game)
    try:
        return await asyncio.to_thread(
            inspect_next_decision, cloned_state, fallback_map, cloned_last_result
        )
    except Exception:
        logger.exception(
            "ismcts: heuristic fallback also failed game=%s", game.game_id
        )
        return None


# --------------------------------------------------------------------------- #
# Agent factory                                                                #
# --------------------------------------------------------------------------- #


def agent_for_spec(spec: BotSpec, seed: int = 0) -> Agent:
    """Instantiate the concrete :class:`Agent` for a persisted :class:`BotSpec`.

    Parameters
    ----------
    spec:
        Serializable bot configuration (see :mod:`goa2.server.bot_models`).
    seed:
        RNG seed passed to seeded agents (Random, Heuristic, ISMCTS).

    Behavior
    --------

    - ``"random"`` / ``"heuristic"`` → the corresponding stateless-ish agent.
    - ``"ismcts"`` → a bounded :class:`ISMCTSAgent` built from
      ``spec.search`` (or the production defaults if ``search`` was
      omitted). The agent instance itself does not enforce the runtime
      timeout — the coordinator wraps every call in
      :func:`_bounded_inspect_next_decision`, which owns the process-wide
      semaphore, queue timeout, and search timeout.

    Raises
    ------
    ValueError
        If ``spec.kind`` is not a supported agent kind.
    """
    kind = spec.kind
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "heuristic":
        return HeuristicAgent(seed=seed)
    if kind == "ismcts":
        settings = spec.search or SearchSettings()
        # Only ``iterations`` and the seed influence the search's internal
        # decisions; ``decision_timeout_seconds`` is enforced at the
        # coordinator boundary (see :func:`_bounded_inspect_next_decision`).
        # This split is intentional: the agent stays deterministic given
        # a fixed seed + iteration budget, which is what
        # ``test_ismcts_is_deterministic`` relies on.
        cfg = SearchConfig(iterations=settings.iterations, seed=seed)
        return ISMCTSAgent(cfg)
    raise ValueError(f"unsupported bot kind: {kind!r}")


def _game_entropy(game_id: str) -> int:
    """Derive a stable, non-negative seed from a game_id.

    We want per-game-stable entropy so restarting a game (same id, same
    persisted spec) produces the same RNG sequence for a rebuilt agent —
    important for reproducibility of a bot's decisions across restarts.
    ``hash(game_id)`` is not sufficient (Python hash randomization); a
    SHA-1 digest reduced to the low 63 bits gives us a stable seed that
    fits in a positive int64.
    """
    digest = hashlib.sha1(game_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def get_or_build_agents(game: ManagedGame) -> dict[str, Agent]:
    """Return this game's cached agent instances, building them if needed.

    Each hero's agent is constructed once per game lifetime (or once per
    restore) with a stable per-hero seed:

        seed(hero) = _game_entropy(game_id) ^ index_in_sorted_bot_specs

    Sorting ``bot_specs`` by hero id makes the offset deterministic given
    the persisted spec set. XOR (rather than add) preserves the low-bit
    entropy of the game hash instead of collapsing it into a small offset.
    Instances are cached on ``game._bot_agents`` (runtime-only field) so a
    long-running game does not re-instantiate an agent per decision, and so
    a stateful policy (e.g. an RNG-based bot) advances its stream naturally
    from one decision to the next.
    """
    cached = game._bot_agents
    if cached is not None:
        return cached
    base = _game_entropy(game.game_id)
    agents: dict[str, Agent] = {}
    for idx, (hero_id, spec) in enumerate(sorted(game.bot_specs.items())):
        agents[hero_id] = agent_for_spec(spec, seed=base ^ idx)
    game._bot_agents = agents
    return agents


# --------------------------------------------------------------------------- #
# Snapshot helpers                                                             #
# --------------------------------------------------------------------------- #


def _snapshot_last_result(
    last_result: SessionResult | None,
) -> SessionResult | None:
    """Deep-copy a :class:`SessionResult` for the background thread.

    ``SessionResult`` is a Pydantic model; ``model_copy(deep=True)`` walks
    the whole graph, including the embedded :class:`InputRequest`, its
    :class:`InputOption` list, and its ``context`` dict. That means any
    agent that (accidentally or maliciously) mutates the received request
    can only mutate its snapshot — the live copy carried on
    ``game.last_result`` and referenced by ``state.input_stack`` is
    untouched.

    ``None`` (game start / non-INPUT_NEEDED) is passed through.
    """
    if last_result is None:
        return None
    return last_result.model_copy(deep=True)


def _snapshot_pending_request(
    last_result: SessionResult | None,
) -> InputRequest | None:
    """Return the pending input request from ``last_result``, if any."""
    if last_result is None:
        return None
    if last_result.result_type is SessionResultType.INPUT_NEEDED:
        return last_result.input_request
    return None


# --------------------------------------------------------------------------- #
# Stale validation                                                             #
# --------------------------------------------------------------------------- #


def _upgrade_still_pending(request: InputRequest, hero_id: str) -> bool:
    """UPGRADE_PHASE: this hero still owes an answer.

    UPGRADE_PHASE puts every player's remaining choice count in
    ``context['players'][hero_id]['remaining']``. A hero that has already
    picked all their upgrades is still listed but with ``remaining == 0``;
    they must not be counted as an eligible responder.
    """
    if request.request_type is not InputRequestType.UPGRADE_PHASE:
        return True
    players_ctx = request.context.get("players") or {}
    info = players_ctx.get(hero_id) or {}
    return int(info.get("remaining", 0)) > 0


def _is_decision_still_valid(
    state: GameState,
    live_last_result: SessionResult | None,
    decision: BotDecision,
    agents: dict[str, Agent],
) -> bool:
    """Whether ``decision`` may still be applied against the live state.

    The coordinator computes on a snapshot; anything can happen between
    snapshot and apply — a human commit, a timer expiry, another bot's
    turn. We must reject any decision that no longer matches the live
    world, without touching persistence.

    PLANNING decisions require:

    - ``state.phase == PLANNING``,
    - the hero still exists,
    - and either an open Emmitt second-card window or the hero is not yet
      in ``pending_inputs`` (a first commit is still valid).

    INPUT decisions require:

    - ``live_last_result`` is INPUT_NEEDED,
    - the live request's ``id`` and ``player_id`` match the decision,
    - ``decision.hero_id`` is *still* an eligible responder to the live
      request (recomputed from live state — team memberships, upgrade
      remaining counts can all have shifted), and
    - ``decision.hero_id`` is still bot-owned in ``agents`` (a mid-flight
      configuration change would otherwise let the coordinator apply a
      decision on behalf of a hero no client currently controls).
    """
    if state.phase == GamePhase.GAME_OVER:
        return False

    if decision.kind is DecisionKind.PLANNING:
        if state.phase != GamePhase.PLANNING:
            return False
        hid = HeroID(decision.hero_id)
        hero = state.get_hero(hid)
        if hero is None:
            return False
        # Bot ownership must still hold (bot_specs cannot legally change
        # mid-game today, but a defensive check keeps the coordinator honest
        # against a future reconfiguration shape change).
        if decision.hero_id not in agents:
            return False
        # Emmitt's second-card commit/finish window?
        if planning_open_for_second_card(state, hid):
            return True
        # First commit: must not already be committed.
        return hid not in state.pending_inputs

    # INPUT — recompute live eligibility.
    request = decision.request
    if request is None:
        return False
    live_request = _snapshot_pending_request(live_last_result)
    if live_request is None:
        return False
    if live_request.id != request.id:
        return False
    if live_request.player_id != request.player_id:
        return False
    # Live eligible responders. Ownership resolution runs against live state
    # so a team-scoped request whose teammate composition shifted (defeats,
    # reshuffles, etc.) is caught here.
    eligible = eligible_hero_ids_for_request(state, live_request)
    if decision.hero_id not in eligible:
        return False
    if decision.hero_id not in agents:
        return False
    # UPGRADE_PHASE: hero must still owe an upgrade.
    return _upgrade_still_pending(live_request, decision.hero_id)


# --------------------------------------------------------------------------- #
# Broadcast / logging / replay helpers                                         #
# --------------------------------------------------------------------------- #


CapturedBroadcast = list[tuple[str | None, Any, dict[str, Any]]]


def _capture_broadcast_for_result(
    game: ManagedGame, result: SessionResult
) -> CapturedBroadcast:
    """Materialize scoped broadcasts for ``result`` while holding the lock."""
    from goa2.server.ws import _capture_broadcast

    events = [ev.model_dump() for ev in result.events] if result.events else None
    return _capture_broadcast(game, events)


async def _send_broadcast(game: ManagedGame, messages: CapturedBroadcast) -> None:
    """Send captured broadcast payloads (already outside ``game.lock``)."""
    if not messages:
        return
    from goa2.server.ws import _send_captured_broadcast

    await _send_captured_broadcast(game, messages)


def _log_result(game: ManagedGame, result: SessionResult) -> None:
    """Same shape :func:`goa2.server.routes_games._log_result` uses."""
    gl = game.game_logger
    if gl is None:
        return
    state = game.session.state
    gl.log_phase_change(result.current_phase.value, state.round, state.turn)
    events = [ev.model_dump() for ev in result.events]
    if events:
        gl.log_events(events)
    if result.input_request:
        gl.log_input_request(result.input_request.to_dict())
    if result.winner:
        gl.log_game_over(result.winner)


def _record_replay(
    game: ManagedGame,
    decision: BotDecision,
    rec_round: int,
    rec_turn: int,
) -> None:
    """Append the applied bot decision to the replay recorder.

    The replay actor is always the decision-making hero (``decision.hero_id``),
    never ``request.player_id``. That distinction matters for team-scoped
    (``"team:RED"``) and simultaneous (``"simultaneous"``, UPGRADE_PHASE)
    requests: the replay must record *which* hero of the team actually
    answered so a rebuilt replay drives the identical hero through the
    same seam. This mirrors ``ws._handle_submit_input`` / REST
    ``submit_input`` where ``hero_id`` is the authenticated submitter, not
    the raw request routing address.
    """
    rec = game.replay_recorder
    if rec is None:
        return
    hero_id_str = str(decision.hero_id)
    if decision.kind is DecisionKind.PLANNING:
        plan = decision.planning
        assert plan is not None
        if plan.card is not None:
            rec.record_commit(hero_id_str, plan.card.id, rec_round, rec_turn)
        elif plan.kind.value == "FINISH":
            rec.record_finish_planning(hero_id_str, rec_round, rec_turn)
        else:  # PASS
            rec.record_pass(hero_id_str, rec_round, rec_turn)
        return
    # INPUT: use the decision maker, not request.player_id.
    rec.record_input(hero_id_str, decision.selection, rec_round, rec_turn)


def _log_action_specific(game: ManagedGame, decision: BotDecision) -> None:
    """Mirror the action-specific logger calls REST/WS mutation handlers make.

    Each REST/WS action handler emits an action-specific log line *before*
    the generic :func:`_log_result` produces the phase / events / winner
    entries. Bots must do the same so a game's log stream is identical
    regardless of whether the mutation came from a human client or the
    coordinator. The exact vocabulary mirrors the handlers:

    - COMMIT  → ``log_card_commit(hero_id, card_id)`` (see
      ``ws._handle_commit_card``, ``routes_games.commit_card``).
    - PASS    → ``log_pass_turn(hero_id)`` (see ``ws._handle_pass_turn``,
      ``routes_games.pass_turn``).
    - FINISH  → no dedicated logger call (matching ``_handle_finish_planning``
      / ``routes_games.planning_done``); only ``_log_result`` fires.
    - INPUT   → ``log_input_response(hero_id, selection)`` (see
      ``ws._handle_submit_input``).

    The ``hero_id`` passed to every logger call is the decision maker
    (``decision.hero_id``) — same actor identity used for the replay entry.
    """
    gl = game.game_logger
    if gl is None:
        return
    hero_id_str = str(decision.hero_id)
    if decision.kind is DecisionKind.PLANNING:
        plan = decision.planning
        assert plan is not None
        if plan.card is not None:
            gl.log_card_commit(hero_id_str, plan.card.id)
        elif plan.kind.value == "PASS":
            gl.log_pass_turn(hero_id_str)
        # FINISH has no dedicated log method by design.
        return
    # INPUT: log the response with the submitter identity.
    gl.log_input_response(hero_id_str, decision.selection)


def _freeze_rollback_for_bot_input(game: ManagedGame, decision: BotDecision) -> None:
    """Freeze rollback before applying a bot ``INPUT`` decision.

    An automatic Resolution/Response answer produced by a server-managed bot
    is externally revealed the same way a timer-driven answer is (see
    :func:`time_control._apply_input_timeout`, which sets exactly the same
    flags before its ``session.advance``). Reusing the engine's existing
    freeze signal ensures that chained resolution steps cannot expose a
    rollback that would reopen or reroll an already externally-decided input.

    Only RESOLUTION-phase INPUT decisions freeze — this mirrors the
    time-control coordinator's own policy: :func:`_apply_input_timeout`
    freezes rollback but :func:`_apply_level_up_timeout` and
    :func:`_apply_planning_timeout` do not. PLANNING decisions are
    committed facedown and UPGRADE_PHASE is between turns, so neither has
    a live resolution snapshot to invalidate.
    """
    if decision.kind is not DecisionKind.INPUT:
        return
    if game.session.state.phase != GamePhase.RESOLUTION:
        return
    game.session.state.execution_context["rollback_frozen"] = True
    game.session._rollback_snapshot = None
    game.session._rollback_actor_id = None


def _stop_clock_for_decision(game: ManagedGame, decision: BotDecision) -> None:
    """Pause the appropriate clock at decision acceptance."""
    if decision.kind is DecisionKind.PLANNING:
        stop_clock_for_accepted_decision(
            game,
            hero_id=str(decision.hero_id),
            completes_planning=True,
        )
        return
    request = decision.request
    assert request is not None
    stop_clock_for_accepted_decision(
        game,
        hero_id=str(decision.hero_id),
        request_id=request.id,
    )


# --------------------------------------------------------------------------- #
# Idle progression: plain advance when the bot owes work but no request yet    #
# --------------------------------------------------------------------------- #


def _current_pending_request(
    last_result: SessionResult | None,
) -> InputRequest | None:
    return _snapshot_pending_request(last_result)


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #


async def _bot_drive_worker(game: ManagedGame, registry: GameRegistry) -> None:
    """Drive bot decisions until the next work belongs to a human or the game ends.

    See module docstring for invariants. The loop is:

    1. Snapshot under ``game.lock``: clone state, deep-copy last_result,
       cache the current agents mapping. Release the lock immediately.
    2. Off-loop compute via ``asyncio.to_thread``. Isolated inputs; no live
       state visible.
    3. If the driver returned a decision → apply under ``outbound_lock`` →
       ``game.lock`` with live revalidation + finally-clause finalize.
    4. If the driver returned ``None``: if a bot still owes the next
       decision (i.e. the engine is between input requests during
       RESOLUTION and we own it), issue one plain ``session.advance()``
       through the same locked-mutation path. Otherwise exit.
    5. Yield to the event loop, then continue.
    """
    if not game.bot_specs:
        return

    # Safety cap: a runaway coordinator (livelocking engine bug, misbehaving
    # bot pair) must not monopolize the event loop even with ``to_thread``.
    # Real games use <10k iterations end-to-end; 100k is a wide margin that
    # still fails loud rather than silently spinning forever.
    max_iterations = 100_000
    iterations = 0

    while iterations < max_iterations:
        iterations += 1

        # Tombstone check: if the game was removed while we were suspended,
        # exit cleanly. Every subsequent locked section re-checks this so a
        # remove landing mid-iteration halts progress before any side
        # effect leaks past the tombstone.
        if game.removed:
            return
        # Defense-in-depth: even though :func:`schedule_bot_drive` gates
        # spawning on the runnable-state invariant, the state may have
        # transitioned out (SUSPENDED, FINISHED) between spawn and this
        # iteration. Re-check here so a worker started while RUNNING but
        # transitioned mid-flight exits cleanly instead of racing a
        # suspended clock.
        if not _is_runnable_for_bots(game):
            return

        # ------------------------------------------------------------------ #
        # 1. Snapshot under game.lock.
        # ------------------------------------------------------------------ #
        try:
            async with game.lock:
                if game.removed:
                    return
                if not _is_runnable_for_bots(game):
                    return
                if game.session.state.phase == GamePhase.GAME_OVER:
                    return
                cloned_state = clone_state(game.session.state)
                cloned_last_result = _snapshot_last_result(game.last_result)
                # Snapshot the agents mapping too — a mid-drive reconfig of
                # ``bot_specs`` would rebuild ``_bot_agents`` on
                # the next call, but this iteration operates on the map we
                # saw under the lock so the stale-check has a fixed target.
                agents = dict(get_or_build_agents(game))
        except asyncio.CancelledError:
            raise

        # ------------------------------------------------------------------ #
        # 2. Compute decision outside locks, on isolated snapshot objects.
        #    :func:`_bounded_inspect_next_decision` runs on a background
        #    thread (never on the event loop). When any ISMCTS bot is in
        #    ``agents`` it additionally enforces the process-wide semaphore,
        #    queue timeout, and per-decision search timeout — falling back
        #    to a cached HeuristicAgent on any bound violation. Random /
        #    Heuristic bots take the plain ``to_thread`` fast path.
        # ------------------------------------------------------------------ #
        try:
            decision = await _bounded_inspect_next_decision(
                game, cloned_state, agents, cloned_last_result
            )
        except asyncio.CancelledError:
            raise
        except IllegalBotDecisionError as exc:
            logger.error(
                "Bot for game %s produced illegal decision (%s); halting drive",
                game.game_id,
                exc,
            )
            return
        except Exception:
            logger.exception(
                "Bot compute failed for game %s; halting drive", game.game_id
            )
            return

        if decision is None:
            # No bot-owned decision. Two cases:
            #
            # (a) Live pending input is addressed to a human / unmapped
            #     hero: exit and let the eventual human mutation reschedule
            #     us. Do NOT advance() blindly — that could consume an
            #     engine step that belongs to a person.
            #
            # (b) No live pending input but the phase is not PLANNING and
            #     not GAME_OVER: the engine has more stack work but hasn't
            #     surfaced the next request. A bot must nudge the engine
            #     with a plain ``session.advance()`` so a bot-vs-bot game
            #     can resolve mid-turn actions without an external caller.
            progressed = await _maybe_plain_advance(game, registry, agents)
            if not progressed:
                return
            await asyncio.sleep(0)
            continue

        # ------------------------------------------------------------------ #
        # 3. Apply under outbound_lock → game.lock.
        # ------------------------------------------------------------------ #
        applied = await _apply_bot_decision(game, registry, decision, agents)
        if applied is None:
            # Stale or failed apply: no side effects landed. Try again on
            # the next scheduled iteration if the caller reschedules; the
            # current worker exits so we don't loop tightly on a broken
            # agent.
            return

        # Terminal state: exit immediately, no further work.
        if applied.result_type is SessionResultType.GAME_OVER:
            return

        # ------------------------------------------------------------------ #
        # 4. Yield to the event loop before scheduling another decision.
        # ------------------------------------------------------------------ #
        await asyncio.sleep(0)

    logger.error(
        "Bot drive worker for game %s exceeded iteration safety limit",
        game.game_id,
    )


async def _apply_bot_decision(
    game: ManagedGame,
    registry: GameRegistry,
    decision: BotDecision,
    agents: dict[str, Agent],
) -> SessionResult | None:
    """Apply one :class:`BotDecision` through the standard locked mutation.

    Returns the fresh :class:`SessionResult` on success, or ``None`` if the
    decision was stale, the engine rejected it, or something in the
    log/save/broadcast chain raised. On any failure path the clock is
    guaranteed to be reconciled (``finalize_timed_mutation`` in ``finally``),
    matching the guarantee REST/WS mutations give.
    """
    result: SessionResult | None = None
    messages: CapturedBroadcast = []

    try:
        async with game.outbound_lock:
            async with game.lock:
                if game.removed:
                    # Tombstone landed while we were computing. Do NOT
                    # persist, log, replay, or broadcast — the game is
                    # being torn down and any side effect after remove is
                    # a defect (stale save file resurrection, phantom
                    # STATE_UPDATE to reconnecting clients, etc.).
                    return None
                if not _is_runnable_for_bots(game):
                    # State transitioned out of RUNNING (SUSPENDED /
                    # FINISHED / GAME_OVER) while we were computing.
                    # Drop the decision silently — the coordinator must
                    # not resume a suspended clock or bypass GAME_OVER.
                    return None
                live_state = game.session.state
                if not _is_decision_still_valid(
                    live_state, game.last_result, decision, agents
                ):
                    logger.debug(
                        "Bot decision for game %s stale on apply; dropping",
                        game.game_id,
                    )
                    return None

                rec_round = live_state.round
                rec_turn = live_state.turn
                session: GameSession = game.session

                # Once we call stop_clock_for_accepted_decision, every exit
                # path from here must run finalize_timed_mutation so the
                # paused clock is reconciled + rescheduled. That mirrors the
                # ws._handle_* exception-restore pattern.
                clock_started = False
                try:
                    _stop_clock_for_decision(game, decision)
                    clock_started = True
                    # Externally-revealed automatic response: freeze rollback
                    # before applying, matching time_control's timeout path.
                    _freeze_rollback_for_bot_input(game, decision)
                    try:
                        result = apply_decision(session, decision)
                    except Exception:
                        logger.exception(
                            "Applying bot decision on game %s raised; halting drive",
                            game.game_id,
                        )
                        return None

                    game.last_result = result
                    try:
                        _record_replay(game, decision, rec_round, rec_turn)
                        _log_action_specific(game, decision)
                        _log_result(game, result)
                    except Exception:
                        logger.exception(
                            "Bot decision recorded but log/replay raised for game %s",
                            game.game_id,
                        )
                        # The engine already mutated; we still want to
                        # broadcast and finalize so clients see the truth.

                    try:
                        messages = _capture_broadcast_for_result(game, result)
                    except Exception:
                        logger.exception(
                            "Bot decision applied but broadcast capture failed for game %s",
                            game.game_id,
                        )
                        messages = []
                finally:
                    if clock_started:
                        # save_game + reconcile_game_clock + schedule_deadline.
                        # Must run even if apply/log/broadcast raised.
                        try:
                            finalize_timed_mutation(game, registry)
                        except Exception:
                            logger.exception(
                                "finalize_timed_mutation failed for bot mutation on game %s",
                                game.game_id,
                            )

            # game.lock released here — outbound_lock still held for send.
            await _send_broadcast(game, messages)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Unexpected error applying bot decision for game %s; halting drive",
            game.game_id,
        )
        return None

    return result


async def _maybe_plain_advance(
    game: ManagedGame,
    registry: GameRegistry,
    agents: dict[str, Agent],
) -> bool:
    """Nudge the engine with one plain ``session.advance()`` if a bot owes work.

    Called only when :func:`inspect_next_decision` returned ``None`` — i.e.
    the driver saw nothing decision-shaped to do. Two shapes:

    - Live pending input is addressed to a hero/team where **no** bot is
      eligible → exit. The next lifecycle event (human mutation, timer,
      restore) will reschedule us.
    - Otherwise (RESOLUTION mid-turn with no pending input, or PLANNING
      already exhausted) → issue one plain ``advance()`` under
      ``outbound_lock`` → ``game.lock``, save + broadcast + reschedule the
      deadline, and return ``True`` so the caller loops to compute the
      next decision.

    PLANNING with ``None`` from the driver means every remaining planning
    slot is human-owned; we exit rather than spin.

    Returns ``True`` if an ``advance()`` was applied, ``False`` if the
    worker should exit.
    """
    result: SessionResult | None = None
    messages: CapturedBroadcast = []

    try:
        async with game.outbound_lock:
            async with game.lock:
                if game.removed:
                    return False
                if not _is_runnable_for_bots(game):
                    # State transitioned out of RUNNING mid-flight; do
                    # not nudge the engine forward on a suspended or
                    # finished game.
                    return False
                state = game.session.state
                phase = state.phase

                if phase == GamePhase.GAME_OVER:
                    return False

                # If there's a live pending input, the driver's None means
                # nobody bot-owned answers it. Exit — a human owes the reply.
                pending = _current_pending_request(game.last_result)
                if pending is not None:
                    return False

                # PLANNING with no pending input: the driver has determined
                # every remaining planning slot is human. Exit.
                if phase == GamePhase.PLANNING:
                    return False

                # RESOLUTION (or another non-planning, non-terminal phase)
                # with no pending input — the engine has more stack work to
                # do. Nudge it exactly one step so the next iteration sees
                # a fresh request / phase.
                #
                # We only nudge when it plausibly leads to bot work. In a
                # mixed human/bot game, an advance() might expose an input
                # for a human — that's still fine: on the *next* loop
                # iteration inspect_next_decision returns None with a
                # pending input, and this function exits (case above).
                _ = agents  # agents set is used indirectly by the caller
                try:
                    result = game.session.advance()
                except Exception:
                    logger.exception(
                        "session.advance() failed during bot idle progression for game %s",
                        game.game_id,
                    )
                    return False

                game.last_result = result
                try:
                    _log_result(game, result)
                except Exception:
                    logger.exception(
                        "Bot idle-advance log failed for game %s", game.game_id
                    )
                try:
                    messages = _capture_broadcast_for_result(game, result)
                except Exception:
                    logger.exception(
                        "Bot idle-advance broadcast capture failed for game %s",
                        game.game_id,
                    )
                    messages = []

                try:
                    finalize_timed_mutation(game, registry)
                except Exception:
                    logger.exception(
                        "finalize_timed_mutation failed for idle-advance on game %s",
                        game.game_id,
                    )

            await _send_broadcast(game, messages)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Unexpected error during bot idle-advance for game %s", game.game_id
        )
        return False

    if result is None:
        return False
    # A GAME_OVER result means "no further work" — caller checks and exits.
    return result.result_type is not SessionResultType.GAME_OVER


# --------------------------------------------------------------------------- #
# Scheduler                                                                    #
# --------------------------------------------------------------------------- #


def _is_runnable_for_bots(game: ManagedGame) -> bool:
    """Whether ``game`` is in a state where bot actions may safely run.

    The invariant: bots may only compute/apply against a game that has
    reached ``clock.status == RUNNING`` if it uses time controls, and any
    non-terminal engine phase otherwise. Every early-exit predicate below
    is a *hard* gate — a violation means the coordinator would either race
    an un-anchored clock (F1 class of bug) or apply an action on a game
    the registry has explicitly ended or removed.

    Rules:

    - **Removed** → ``False``. The registry has evicted the game;
      side effects after removal are always defects.
    - **GAME_OVER** → ``False``. The engine has finalized; further
      mutations are illegal.
    - **No clock (un-timed)** → ``True``. Un-timed matches have no
      readiness handshake; bots may act freely.
    - **Clock ``WAITING_FOR_PLAYERS``** → ``False``. The match has not
      begun; even bots whose ready flag is set must wait for every
      player (typically a human) to ready-up. Any bot compute here
      would race the eventual clock-start reconciliation.
    - **Clock ``SUSPENDED_FOR_INACTIVITY``** → ``False``. The match
      voluntarily paused; bots must not resume it unilaterally.
    - **Clock ``FINISHED``** → ``False``. The clock has ended.
    - **Clock ``RUNNING``** → ``True``.
    """
    if game.removed:
        return False
    if game.session.state.phase == GamePhase.GAME_OVER:
        return False
    clock = game.session.state.clock
    if clock is None:
        return True
    return clock.status == ClockStatus.RUNNING


def schedule_bot_drive(game: ManagedGame, registry: GameRegistry) -> None:
    """Ensure a single bot worker is running (or about to run) for ``game``.

    Idempotent: a call while a previous worker is still alive is a no-op.
    Once the previous worker finishes, the next call spawns a fresh task.

    Callers: game creation, REST/WS mutation
    completion, timer completion, and application restore. Every seam that
    can hand control to a bot should call this — it costs nothing when
    already scheduled and is the only correct way to start bot progress.

    Safe to call outside an active asyncio task if a loop is running (uses
    :func:`asyncio.create_task`). The caller must be inside an event loop.

    **Runnable-state gate.** A single :func:`_is_runnable_for_bots`
    predicate short-circuits the schedule for timed games that have not
    yet reached ``RUNNING`` (``WAITING_FOR_PLAYERS`` /
    ``SUSPENDED_FOR_INACTIVITY``), for finished/terminated games, and for
    removed games. Un-timed games and running timed games proceed. This
    is the central invariant: every REST/WS/timer/lifecycle scheduling
    seam inherits the gate automatically because they all go through
    this function.
    """
    if not game.bot_specs:
        return
    # Central runnable-state gate. See :func:`_is_runnable_for_bots`.
    # Never schedule for a game that has been removed, has ended, or
    # (for timed matches) has not yet reached ``RUNNING``.
    if not _is_runnable_for_bots(game):
        return

    existing = game.bot_task
    if existing is not None and not existing.done():
        return

    task = asyncio.create_task(
        _bot_drive_worker(game, registry),
        name=f"bot-drive-{game.game_id}",
    )
    game.bot_task = task

    def _clear_reference(t: asyncio.Task[None]) -> None:
        if game.bot_task is t:
            game.bot_task = None
        if not t.cancelled():
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "Bot drive task for game %s ended with exception: %r",
                    game.game_id,
                    exc,
                )

    task.add_done_callback(_clear_reference)


# --------------------------------------------------------------------------- #
# Lifecycle hooks                                                              #
# --------------------------------------------------------------------------- #


async def start_bot_lifecycle(
    game: ManagedGame,
    registry: GameRegistry,
    *,
    at_ms: int | None = None,
) -> bool:
    """One-call lifecycle seam for creation and restore.

    Auto-readies every bot hero on a timed match under ``outbound_lock`` →
    ``game.lock``, then — if the ready transition actually started the
    clock — runs the standard finalize (persist + reconcile + schedule the
    initial deadline task) and broadcasts a scoped state update before
    scheduling the bot coordinator. This closes the window where a bot
    could compute against a not-yet-persisted / not-yet-anchored clock.

    Returns whether the ready transition started the clock. Non-timed and
    already-running matches return ``False``. Un-bot games short-circuit
    to ``False`` before acquiring any lock.

    Callers (``create_game`` and the ``lifespan`` restore hook) must not
    call :func:`auto_ready_bot_heroes` + :func:`schedule_bot_drive`
    separately — that pattern skips persistence and broadcast between the
    two calls, which is exactly what led to bots computing against an
    un-anchored clock. This helper is the single blessed seam.
    """
    if not game.bot_specs:
        return False
    if game.removed:
        return False

    started = False
    messages: list[Any] = []
    async with game.outbound_lock:
        async with game.lock:
            if game.removed:
                return False
            started = auto_ready_bot_heroes(game, at_ms=at_ms)
            if started:
                # A ready transition started the clock: persist + reconcile
                # + schedule the initial deadline task before any bot code
                # runs. Otherwise the bot would race a not-yet-scheduled
                # authoritative deadline.
                finalize_timed_mutation(game, registry, at_ms)
                # Materialize a scoped broadcast so every connected client
                # observes the clock transitioning to RUNNING before the
                # first bot mutation lands. Timer events are not attached
                # because no ``TIMER_EXPIRED`` fired — this is a ready
                # transition, not a timeout.
                from goa2.server.ws import _capture_broadcast

                messages = _capture_broadcast(game)

        if messages:
            from goa2.server.ws import _send_captured_broadcast

            await _send_captured_broadcast(game, messages)

    # Schedule the bot task after the ready transition has been persisted
    # and broadcast (or after auto-ready was a no-op on an un-timed game).
    # ``schedule_bot_drive`` still guards against ``game.removed``.
    schedule_bot_drive(game, registry)
    return started


def auto_ready_bot_heroes(game: ManagedGame, at_ms: int | None = None) -> bool:
    """Mark every bot-owned hero as ready on a timed match.

    Timed matches start in :class:`ClockStatus.WAITING_FOR_PLAYERS` and only
    leave it once :func:`set_player_ready` has been called for every hero in
    ``clock.players``. A game full of humans + bots would otherwise stall
    forever because no client submits a ready flag on behalf of a bot. This
    helper is idempotent (a hero already listed in ``ready_hero_ids`` is
    left alone) and returns whether the match transitioned to RUNNING as a
    side effect (i.e. this call was the last-required ready signal).

    Safe to call on non-timed games (returns ``False`` — no clock) and on
    games without any bots (returns ``False``).

    Callers that need the ready transition to be *durable* (persisted and
    reconciled with an initial deadline task) must use
    :func:`start_bot_lifecycle` instead, which wraps this helper with the
    proper finalize + broadcast + scheduler chain.
    """
    if not game.bot_specs:
        return False
    clock = game.session.state.clock
    if clock is None:
        return False
    if clock.status not in {
        ClockStatus.WAITING_FOR_PLAYERS,
        ClockStatus.SUSPENDED_FOR_INACTIVITY,
    }:
        return False
    timestamp = at_ms if at_ms is not None else now_ms()
    started = False
    already_ready = set(clock.ready_hero_ids)
    for hero_id in game.bot_specs:
        if hero_id not in clock.players:
            # A bot hero not part of the clock roster cannot be readied;
            # persistence/roster drift protection — log rather than raise so
            # startup does not fail on a stale save.
            logger.warning(
                "auto_ready_bot_heroes: game %s bot hero %s not in clock roster",
                game.game_id,
                hero_id,
            )
            continue
        if hero_id in already_ready:
            continue
        try:
            if set_player_ready(game, hero_id, True, timestamp):
                started = True
        except ValueError:
            logger.exception(
                "auto_ready_bot_heroes: game %s failed to ready bot hero %s",
                game.game_id,
                hero_id,
            )
    return started


async def cancel_all_bot_tasks(
    registry: GameRegistry, *, drain_timeout_seconds: float = 5.0
) -> None:
    """Cancel every game's bot worker task and drain in-flight searches.

    Called from the FastAPI ``lifespan`` shutdown seam so we never leave an
    orphan bot task running after the app stops. Behavior in order:

    1. Cancel every :attr:`ManagedGame.bot_task` and gather them with
       ``return_exceptions=True`` — the coordinator raising
       :class:`asyncio.CancelledError` at shutdown is the expected flow.
    2. **Drain every tracked in-flight bounded-search future WITHOUT
       cancelling them on timeout.** We use :func:`asyncio.wait` (which
       does not cancel its arguments on timeout, unlike
       :func:`asyncio.wait_for` / :func:`asyncio.gather` in a
       ``wait_for``) so a slow executor thread keeps running to its
       natural completion. That preserves the coordinator's core
       invariant: a semaphore slot is released **only** by the
       done-callback that fires when the underlying thread actually
       finishes — never by shutdown cancelling the future out from
       under it.
    3. On drain timeout, log a warning that some futures are still
       pending. They remain in the trackers (:data:`_in_flight_search_futures`
       and per-game :attr:`ManagedGame._bot_search_futures`) so a later
       reader can still observe them; the done-callback will eventually
       remove them when the executor thread completes on its own.
    4. Post-drain, log any leftover tracked futures as a warning — this
       is expected on shutdown-timeout paths and informational otherwise.

    Cancelling an executor future is safe from asyncio's perspective
    (Python sets the "cancelled" flag), but the underlying thread does
    not stop; more importantly, cancellation prevents the
    done-callback from running with the normal ``result()`` path,
    which is exactly the path that releases the semaphore. We must
    not touch cancellation here — the done-callback is authoritative.
    """
    tasks: list[asyncio.Task[None]] = []
    for game in registry.all_games():
        task = game.bot_task
        if task is None or task.done():
            continue
        task.cancel()
        tasks.append(task)
        game.bot_task = None
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Snapshot the in-flight set before awaiting: the done-callback
    # removes entries from ``_in_flight_search_futures`` and the per-game
    # set, so iterating over the live set while it mutates would risk
    # skipping entries. A snapshot handles that safely.
    pending = list(_in_flight_search_futures)
    if pending:
        # ``asyncio.wait`` does NOT cancel its arguments on timeout —
        # this is the critical difference from ``asyncio.wait_for`` /
        # ``asyncio.gather`` wrapped in ``wait_for``. Any future still
        # in ``still_pending`` after the timeout keeps running on its
        # executor thread; its semaphore slot stays held until the
        # thread finishes and the done-callback fires.
        done, still_pending = await asyncio.wait(
            pending,
            timeout=drain_timeout_seconds,
            return_when=asyncio.ALL_COMPLETED,
        )
        if still_pending:
            logger.warning(
                "cancel_all_bot_tasks: %d bounded-search future(s) did not "
                "drain within %.1fs; leaving them running to natural "
                "completion (tracked, not cancelled)",
                len(still_pending),
                drain_timeout_seconds,
            )
        # Surface any exceptions the drained futures raised so a shutdown
        # doesn't silently swallow an agent bug. We only inspect ``done``
        # here — ``still_pending`` futures cannot have a result yet.
        for fut in done:
            if fut.cancelled():
                # Should not happen: we never cancel these futures. Log
                # defensively rather than let a leaked cancel propagate.
                logger.warning(
                    "cancel_all_bot_tasks: drained future was cancelled "
                    "(unexpected — did another caller cancel it?)"
                )
                continue
            exc = fut.exception()
            if exc is not None:
                logger.warning(
                    "cancel_all_bot_tasks: drained future ended with "
                    "exception: %r",
                    exc,
                )

    # Post-drain leftovers: on the timeout path this is expected (the
    # thread is still running). On the happy path it means a
    # done-callback failed to clear the tracker, which is worth a
    # warning but is not a crash.
    if _in_flight_search_futures:
        logger.warning(
            "cancel_all_bot_tasks: %d future(s) still tracked module-wide "
            "after drain",
            len(_in_flight_search_futures),
        )
    for game in registry.all_games():
        if game._bot_search_futures:
            logger.warning(
                "cancel_all_bot_tasks: game %s still tracks %d in-flight "
                "search future(s) after drain",
                game.game_id,
                len(game._bot_search_futures),
            )
