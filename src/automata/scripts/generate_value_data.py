"""Generate a value-training dataset via Heuristic self-play.

Runs :func:`automata.runtime.harness.run_game` over an explicit ``[start, end)``
seed range, driving each game with :class:`automata.agents.heuristic_agent.HeuristicAgent`
on both sides and writing compact per-decision value examples through
:class:`automata.evaluation.value_dataset.ValueDatasetRecorder`.

Design choices
--------------

* **Explicit seed range.** ``--seed-start`` / ``--seed-end`` (half-open) are
  used verbatim; no reserved-eval-seed skipping and no derived offsets.
  Callers pick the exact seeds and are responsible for keeping training and
  evaluation seeds disjoint.

* **Fixed benchmark roster.** RED = Wasp + Xargatha, BLUE = Arien + Brogan
  — the same rosters the evaluation matrix pins as its benchmark. Learned
  value / policy models trained on this dataset therefore share the state
  distribution the eval harness scores them on.

* **Heuristic self-play, fresh per (seed, side).** Every mapped agent is a
  :class:`HeuristicAgent`. Self-play — not a mixed matchup — is what the
  learned value function bootstraps from; both teams following the same
  hand-tuned policy keeps the state distribution centered on
  Heuristic-vs-Heuristic play. We build **fresh** agents per world seed
  (and per side, so RED and BLUE cannot cross-contaminate each other's
  tie-break RNGs). Agent seeds are derived *solely* from the world seed
  and the side label, so generating ``[a, c)`` in one run is byte-for-byte
  identical to concatenating ``[a, b)`` and ``[b, c)`` — no per-game state
  survives between world seeds.

* **``run_game`` seam.** The harness call is looked up as
  :data:`~automata.scripts.generate_value_data.run_game` on this module.
  Tests inject fakes with ``monkeypatch.setattr(gen, "run_game", ...)``;
  keeping the seam here (rather than reaching into ``runtime.harness``)
  makes the CLI directly unit-testable without any real self-play.

* **Source identity, once.** Every game inherits the same
  ``(source_revision, dirty_tree_hash)`` — the working tree does not change
  mid-run — captured up front via :func:`automata.evaluation.cli.source_identity`.

* **Truncate on start.** The output file is created (or truncated to zero
  bytes) up front, before any game runs. This means a stale dataset on
  ``--out`` from a prior run cannot survive even if every game in the new
  range fails to produce rows (all ``max_steps``, unknown-team-only, etc.).
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

from ..agents.heuristic_agent import HeuristicAgent
from ..evaluation.cli import source_identity
from ..evaluation.matchup import hero_id
from ..evaluation.value_dataset import ValueDatasetRecorder

# Re-exported on this module so tests can ``monkeypatch.setattr(gen, "run_game", ...)``
# and steer generator behavior without touching runtime.harness.
from ..runtime.harness import DEFAULT_MAP, run_game  # noqa: F401

# Benchmark roster pinned by the evaluation matrix. Kept in sync with
# :data:`automata.evaluation.cli.RED` / ``BLUE`` deliberately so a learned
# model trained here is scored against the same state distribution.
RED_HEROES: tuple[str, ...] = ("Wasp", "Xargatha")
BLUE_HEROES: tuple[str, ...] = ("Arien", "Brogan")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate value-training examples via Heuristic self-play on the "
            "benchmark roster (Wasp+Xargatha vs Arien+Brogan)."
        )
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL path (truncated at start of run).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        required=True,
        help="Inclusive lower bound of the world-seed range.",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        required=True,
        help="Exclusive upper bound of the world-seed range.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20_000,
        help="Per-game engine step budget passed to run_game.",
    )
    parser.add_argument(
        "--game-type",
        type=str,
        default="QUICK",
        help="Game length passed to run_game (default: QUICK).",
    )
    return parser


def _game_id(seed: int) -> str:
    """Deterministic per-seed game id.

    Zero-padded so a directory-listing over multi-run outputs sorts by seed
    naturally. Six digits comfortably covers realistic training sweeps
    (100k+ games) without ever running out.
    """
    return f"gvd-{seed:06d}"


def agent_seed(world_seed: int, side: str) -> int:
    """Deterministic per-agent seed derived from ``(world_seed, side)``.

    Two agents in the same game must never share the seed we hand them
    (that would break tie-breaks across bot instances); mixing the side
    label into the world seed keeps the mapping deterministic and
    reproducible across reruns while cheaply splitting per-side streams.

    Depends only on its two arguments — no per-game state, no counter —
    so the mapping ``(world_seed, side) -> agent_seed`` is a pure
    function. That is what makes ``main([--seed-start=a, --seed-end=c])``
    byte-identical to concatenating the ``[a, b)`` and ``[b, c)`` runs:
    every world seed's per-side agents come from the same recipe
    regardless of where they land in the outer loop.

    The recipe (sha256 → 8-byte prefix → unsigned int) is chosen to be
    stable across Python versions and platforms; it mirrors the private
    ``_agent_seed`` in :mod:`automata.evaluation.cli` deliberately so the
    two subsystems agree on the derivation.
    """
    blob = f"{world_seed}:{side}".encode()
    digest = hashlib.sha256(blob).digest()[:8]
    return int.from_bytes(digest, "big", signed=False)


def _build_agents(world_seed: int) -> dict[str, HeuristicAgent]:
    """Fresh per-side :class:`HeuristicAgent` map for one world seed.

    Two fresh instances are created — one for RED, one for BLUE — each
    seeded from :func:`agent_seed` so their RNGs are independent and
    reproducible from the world seed alone. Every hero on a side shares
    that side's agent (the agent is stateless w.r.t. per-hero context
    but stateful w.r.t. its tie-break RNG, so sharing across a side is
    the natural "one policy per team" mapping).

    Building fresh instances per world seed is the fix for cross-game
    RNG contamination: a single agent reused across many games threads
    its tie-break RNG state through the whole run, which breaks the
    concatenation property that guarantees ``main`` output over a range
    equals the concatenation of any subrange partition.
    """
    red_agent = HeuristicAgent(agent_seed(world_seed, "RED"))
    blue_agent = HeuristicAgent(agent_seed(world_seed, "BLUE"))
    agents: dict[str, HeuristicAgent] = {}
    for name in RED_HEROES:
        agents[hero_id(name)] = red_agent
    for name in BLUE_HEROES:
        agents[hero_id(name)] = blue_agent
    return agents


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. See the module docstring.

    Returns ``0`` on success. Non-zero returns are reserved for future use
    (e.g. surfacing generation errors); today every seed in the requested
    range is attempted.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.seed_end <= args.seed_start:
        parser.error("--seed-end must be strictly greater than --seed-start")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Explicit truncate/create up front. This guarantees stale rows from a
    # prior run cannot survive on ``--out`` even if every game in the new
    # range fails to produce any rows (all max_steps, unknown-team-only,
    # etc.) — the ValueDatasetRecorder only opens the file on a successful
    # ``record_outcome(reason="game_over")`` flush, so without this step
    # a full range of dropped games would leave the previous dataset in
    # place and silently poison downstream training.
    out_path.write_bytes(b"")

    revision, dirty_hash = source_identity()

    red_list = list(RED_HEROES)
    blue_list = list(BLUE_HEROES)

    for seed in range(args.seed_start, args.seed_end):
        recorder = ValueDatasetRecorder(
            out_path,
            game_id=_game_id(seed),
            world_seed=seed,
            red_heroes=red_list,
            blue_heroes=blue_list,
            source_revision=revision,
            dirty_tree_hash=dirty_hash,
            # File was truncated up front; every game appends its rows so
            # a preceding game's output is preserved regardless of whether
            # this one flushes.
            append=True,
        )
        # Re-resolve ``run_game`` on this module so tests can inject a fake
        # via ``monkeypatch.setattr(gen, "run_game", ...)`` at any point.
        import automata.scripts.generate_value_data as _self

        _self.run_game(
            red_list,
            blue_list,
            _build_agents(seed),
            map_path=DEFAULT_MAP,
            game_type=args.game_type,
            seed=seed,
            max_steps=args.max_steps,
            recorder=recorder,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
