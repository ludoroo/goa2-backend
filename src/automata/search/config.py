"""Search configuration for the ISMCTS agent.

One knob-bag shared by the search engine (`ismcts.py`) and the agent wrapper
(`ismcts_agent.py`). Defaults are deliberately conservative so a single decision stays
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

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import CutoffUnit, LeafMode

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

# Opt-in learned-policy root schedule. Product callers may select these values
# without changing classic defaults; offline self-play/evaluation intentionally
# remain unchanged until cross-game evidence is available.
LEARNED_ROOT_PUCT_C: float = 1.5
LEARNED_ROOT_WIDENING_C: float = 1.0
LEARNED_ROOT_WIDENING_ALPHA: float = 0.5


@dataclass(frozen=True)
class SearchConfig:
    # How many determinized playouts per decision.
    iterations: int = 200

    # UCB1 exploration constant. Rewards are in [0, 1], so ~1.4 (≈√2) is sane.
    uct_c: float = 1.4

    # Stop bounded continuation after this many units. ROUNDS preserves the
    # original behavior; DECISIONS bounds the number of controlled choices.
    cutoff_limit: int = 2
    cutoff_unit: CutoffUnit = CutoffUnit.ROUNDS

    # Deterministic fail-closed progression guards. The first bounds one
    # ``_Simulator.advance`` call across session advances and environment
    # planning/input decisions. The second independently bounds consecutive
    # forced/no-legal decisions while descending one determinized tree path.
    # Both defaults leave ample headroom for long valid card chains.
    max_advance_transitions: int = 1024
    max_forced_decisions: int = 256

    # Evaluate a newly expanded state immediately, or first continue it with
    # the separately configured continuation policy up to ``cutoff_limit``.
    leaf_mode: LeafMode = LeafMode.BOUNDED_CONTINUATION

    # Progressive widening: a node with visit count N may reveal at most
    # ⌈C · N^alpha⌉ children. Tames wide positioning nodes (many legal hexes).
    widening_c: float = 2.0
    widening_alpha: float = 0.5

    # Optional root-only overrides. ``None`` preserves the classic behavior:
    # the root uses the same widening and PUCT settings as every other node.
    # Production learned-policy bots set these internally; they are not part
    # of the client-facing SearchSettings contract.
    root_widening_c: float | None = None
    root_widening_alpha: float | None = None

    # Versioned opt-in for deterministic broad-HEX root coverage. ``None``
    # preserves the classic iteration budget and widening behavior exactly.
    # The version is part of immutable config identity so generated search
    # telemetry can distinguish schedules without exposing this internal knob
    # through the client-facing SearchSettings model.
    adaptive_hex_root_schedule_version: int | None = None

    # Note: the leaf-value squash used to live here as ``value_scale``. It now
    # belongs to the LeafEvaluator implementation itself, so a Learned value
    # component with its own bounded output can drop in without a search knob.

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
    # knob for higher-budget / learned-policy experiments,
    # where a trained P(a) should make it pay off. See
    # docs/LEARNED_TRAINING_OPERATIONS.md.
    puct_c: float = 0.0
    root_puct_c: float | None = None

    def __post_init__(self) -> None:
        schedule_version = self.adaptive_hex_root_schedule_version
        if schedule_version is not None and (
            isinstance(schedule_version, bool)
            or not isinstance(schedule_version, int)
            or schedule_version != 1
        ):
            raise ValueError("adaptive_hex_root_schedule_version must be None or integer 1")
        for field_name in ("max_advance_transitions", "max_forced_decisions"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.root_puct_c is not None and (
            not math.isfinite(self.root_puct_c) or self.root_puct_c < 0.0
        ):
            raise ValueError("root_puct_c must be finite and non-negative")
        if self.root_widening_c is not None and (
            not math.isfinite(self.root_widening_c) or self.root_widening_c <= 0.0
        ):
            raise ValueError("root_widening_c must be finite and positive")
        if self.root_widening_alpha is not None and (
            not math.isfinite(self.root_widening_alpha)
            or not 0.0 <= self.root_widening_alpha <= 1.0
        ):
            raise ValueError("root_widening_alpha must be finite and in [0, 1]")


def _strict_int(config: Mapping[str, Any], field: str, *, minimum: int) -> int:
    value = config[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"search-config field {field!r} must be a {qualifier} integer")
    return value


def _strict_float(
    config: Mapping[str, Any],
    field: str,
    *,
    minimum: float,
    minimum_inclusive: bool,
    maximum: float | None = None,
) -> float:
    value = config[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"search-config field {field!r} must be a number")
    parsed = float(value)
    below_minimum = parsed < minimum if minimum_inclusive else parsed <= minimum
    if not math.isfinite(parsed) or below_minimum or (maximum is not None and parsed > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"search-config field {field!r} must be finite and in {bounds}")
    return parsed


def _strict_enum(
    config: Mapping[str, Any], field: str, enum_type: type[CutoffUnit] | type[LeafMode]
) -> CutoffUnit | LeafMode:
    value = config[field]
    if not isinstance(value, str):
        raise ValueError(f"search-config field {field!r} must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"search-config field {field!r} must be one of: {choices}") from exc


def parse_learned_lh_search_config(
    raw: Mapping[str, Any],
) -> tuple[SearchConfig, dict[str, Any]]:
    """Strictly resolve the tracked learned-policy/heuristic-value preset.

    The returned identity contains every resolved field and marks the runtime
    per-side seed source. Unknown fields, implicit type coercions, and a caller-
    supplied seed are rejected so production self-play and arena commands use
    exactly the same complete validation boundary.
    """
    if "seed" in raw:
        raise ValueError("search-config field 'seed' is forbidden; seed comes from agent_seed")
    defaults = SearchConfig(
        cutoff_limit=1,
        cutoff_unit=CutoffUnit.DECISIONS,
        leaf_mode=LeafMode.IMMEDIATE,
        use_prior=True,
        root_puct_c=LEARNED_ROOT_PUCT_C,
        root_widening_c=LEARNED_ROOT_WIDENING_C,
        root_widening_alpha=LEARNED_ROOT_WIDENING_ALPHA,
        max_advance_transitions=1024,
        max_forced_decisions=256,
    )
    allowed = set(asdict(defaults)) - {"seed"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown search-config fields for L/H preset: {unknown!r}")

    resolved: dict[str, Any] = {**asdict(defaults), **raw}
    resolved["iterations"] = _strict_int(resolved, "iterations", minimum=1)
    resolved["cutoff_limit"] = _strict_int(resolved, "cutoff_limit", minimum=0)
    for field in ("max_advance_transitions", "max_forced_decisions"):
        resolved[field] = _strict_int(resolved, field, minimum=1)
    resolved["cutoff_unit"] = _strict_enum(resolved, "cutoff_unit", CutoffUnit)
    resolved["leaf_mode"] = _strict_enum(resolved, "leaf_mode", LeafMode)
    if not isinstance(resolved["use_prior"], bool):
        raise ValueError("search-config field 'use_prior' must be a boolean")

    for field in ("uct_c", "puct_c"):
        resolved[field] = _strict_float(resolved, field, minimum=0.0, minimum_inclusive=True)
    resolved["widening_c"] = _strict_float(
        resolved, "widening_c", minimum=0.0, minimum_inclusive=False
    )
    resolved["widening_alpha"] = _strict_float(
        resolved,
        "widening_alpha",
        minimum=0.0,
        minimum_inclusive=True,
        maximum=1.0,
    )
    for field, minimum, inclusive, maximum in (
        ("root_puct_c", 0.0, True, None),
        ("root_widening_c", 0.0, False, None),
        ("root_widening_alpha", 0.0, True, 1.0),
    ):
        if resolved[field] is not None:
            resolved[field] = _strict_float(
                resolved,
                field,
                minimum=minimum,
                minimum_inclusive=inclusive,
                maximum=maximum,
            )
    schedule = resolved["adaptive_hex_root_schedule_version"]
    if schedule is not None and (
        isinstance(schedule, bool) or not isinstance(schedule, int) or schedule != 1
    ):
        raise ValueError(
            "search-config field 'adaptive_hex_root_schedule_version' must be null or integer 1"
        )

    config = SearchConfig(**resolved)
    identity = asdict(config)
    identity["cutoff_unit"] = config.cutoff_unit.value
    identity["leaf_mode"] = config.leaf_mode.value
    identity["seed"] = "agent_seed"
    return config, identity
