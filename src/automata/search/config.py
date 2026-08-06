"""Search configuration for the ISMCTS agent.

One knob-bag shared by the driver (`ismcts.py`) and the agent wrapper
(`agent.py`). Defaults are deliberately conservative so a single decision stays
in the low-hundreds-of-ms range on the ~3.4 ms clone cost.

Production limits
-----------------

The constants below are the *server-side* upper bounds enforced when a client
supplies bounded ISMCTS settings on ``POST /games``. They are deliberately
lower than the algorithm's absolute worst case so a mis-configured client
cannot force the coordinator to spend arbitrary time on a single decision.

- ``PROD_MAX_ITERATIONS`` — hard cap on ``iterations``. Empirically a fresh
  planning decision at ~200 iterations completes in a few hundred ms on the
  reference clone cost; 1000 leaves headroom for wider positions without
  ever approaching the event loop's tolerance.
- ``PROD_MAX_DECISION_TIMEOUT_SECONDS`` — hard cap on wall-clock decision
  time. The coordinator races the search against this deadline and falls
  back to the cached ``HeuristicAgent`` on timeout, so this is also the
  worst-case wait a mixed human/bot game will ever observe on a bot turn.
- ``PROD_MIN_ITERATIONS`` / ``PROD_MIN_DECISION_TIMEOUT_SECONDS`` — lower
  bounds so a request cannot degenerate to zero-iteration / zero-timeout
  configurations that would always fall back before search made progress.

These are the values the request-boundary schema validates against; the
coordinator additionally guards against tampering (e.g. restored saves) by
re-validating at agent build time.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Production bounds                                                           #
# --------------------------------------------------------------------------- #

PROD_MIN_ITERATIONS: int = 1
PROD_MAX_ITERATIONS: int = 1000

PROD_MIN_DECISION_TIMEOUT_SECONDS: float = 0.05
PROD_MAX_DECISION_TIMEOUT_SECONDS: float = 5.0

# Default production values applied when a client omits the field.
PROD_DEFAULT_ITERATIONS: int = 200
PROD_DEFAULT_DECISION_TIMEOUT_SECONDS: float = 2.0

# Process-wide concurrency cap on live ISMCTS searches. The coordinator
# guards ``asyncio.to_thread(search)`` with a semaphore of this size; extra
# callers queue on the semaphore up to :data:`PROD_QUEUE_TIMEOUT_SECONDS`,
# then fall back to :class:`HeuristicAgent`. Chosen small: search is
# CPU-bound and Python threads share the GIL, so more than a couple of
# concurrent searches degrade each other without any wall-clock gain.
PROD_SEARCH_CONCURRENCY: int = 2

# How long a caller may wait for a semaphore slot before falling back. This
# is the queue-wait budget (distinct from ``decision_timeout_seconds`` which
# only starts once the search actually runs).
PROD_QUEUE_TIMEOUT_SECONDS: float = 1.0


@dataclass(frozen=True)
class SearchConfig:
    # How many determinized playouts per decision.
    iterations: int = 200

    # UCB1 exploration constant. Rewards are in [0, 1], so ~1.4 (≈√2) is sane.
    uct_c: float = 1.4

    # Depth cutoff: stop a rollout once the round counter has advanced this many
    # times (2 ≈ one full wave resolved), then substitute the value function.
    cutoff_rounds: int = 2

    # Progressive widening: a node with visit count N may reveal at most
    # ⌈C · N^alpha⌉ children. Tames wide positioning nodes (many legal hexes).
    widening_c: float = 2.0
    widening_alpha: float = 0.5

    # evaluate_state is unbounded; squash through tanh(score / scale) into
    # (0, 1). Scale is order-of-magnitude of a meaningful positional edge.
    value_scale: float = 300.0

    # RNG seed for determinization + tie-breaking (reproducible searches).
    seed: int = 0

    # Use a heuristic expansion prior (reveal promising moves first under
    # progressive widening). Disable to fall back to random expansion order.
    use_prior: bool = True

    # PUCT selection: when > 0 and a prior is present, bias tree selection by
    # the prior probability P(a) (AlphaZero-style) instead of plain UCB1. The
    # prior is also used for expansion ordering regardless. 0 disables PUCT
    # (pure UCB1 selection).
    #
    # DEFAULT OFF: measured 2-10 (16.7%) vs plain UCB1 at 8 iters / 12 games.
    # At low iteration budgets a strong prior over-commits and under-explores,
    # while UCB1's force-try-every-child does better. PUCT stays available as a
    # knob for higher-budget / learned-policy experiments (revisit at Rung 3),
    # where a trained P(a) should make it pay off. See docs/plan_ai_ladder.md.
    puct_c: float = 0.0
