"""Generate resumable terminal labels from ISMCTS cutoff states.

This offline CLI assumes a POSIX main thread so source games can be bounded with
``SIGALRM``/``setitimer``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..evaluation.cli import source_identity
from ..evaluation.features import FEATURE_SCHEMAS
from ..evaluation.matchup import hero_id
from ..evaluation.terminal_labels import TerminalLabelObserver
from ..evaluation.value import HeuristicValue
from ..runtime.harness import DEFAULT_MAP, run_game
from ..search import ISMCTSAgent, SearchConfig

RED_HEROES: tuple[str, ...] = ("Wasp", "Xargatha")
BLUE_HEROES: tuple[str, ...] = ("Arien", "Brogan")


class SourceGameTimeout(TimeoutError):
    """Raised when one source game exceeds its wall-clock budget."""


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


@contextmanager
def _source_game_timeout(seconds: float) -> Iterator[None]:
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        raise RuntimeError("source game timeouts require a POSIX main thread")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise SourceGameTimeout

    signal.signal(signal.SIGALRM, raise_timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate terminal labels from ISMCTS cutoffs.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-end", required=True, type=int)
    parser.add_argument("--iterations", type=_positive, default=4)
    parser.add_argument("--cutoff-rounds", type=_positive, default=2)
    parser.add_argument("--sample-every", type=_positive, default=10)
    parser.add_argument("--max-samples-per-side", type=_positive, default=100)
    parser.add_argument("--source-max-steps", type=_positive, default=20_000)
    parser.add_argument("--source-timeout-seconds", type=_positive_float, default=1800.0)
    parser.add_argument("--continuation-max-steps", type=_positive, default=20_000)
    parser.add_argument("--continuation-max-rounds", type=_positive, default=20)
    parser.add_argument("--game-type", default="QUICK")
    parser.add_argument("--feature-schema", choices=FEATURE_SCHEMAS, default="base-v1")
    return parser


def _identity(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def agent_seed(world_seed: int, side: str) -> int:
    """Return a stable, explicit per-game and per-side search seed."""
    return int(_identity({"world_seed": world_seed, "side": side})[:16], 16)


def _completed_games(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            config_id = row.get("source_config_id")
            seed = row.get("world_seed")
            if isinstance(config_id, str) and isinstance(seed, int):
                completed.add((config_id, seed))
    return completed


def _checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.seed_end <= args.seed_start:
        parser.error("--seed-end must be strictly greater than --seed-start")

    out = Path(args.out)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(f"{out}.games.jsonl")
    revision, dirty_hash = source_identity()
    source_config = {
        "red_heroes": list(RED_HEROES),
        "blue_heroes": list(BLUE_HEROES),
        "iterations": args.iterations,
        "cutoff_rounds": args.cutoff_rounds,
        "sample_every": args.sample_every,
        "max_samples_per_side": args.max_samples_per_side,
        "source_max_steps": args.source_max_steps,
        "source_timeout_seconds": args.source_timeout_seconds,
        "continuation_max_steps": args.continuation_max_steps,
        "continuation_max_rounds": args.continuation_max_rounds,
        "game_type": args.game_type,
        "map_path": DEFAULT_MAP,
        "source_revision": revision,
        "dirty_tree_hash": dirty_hash,
    }
    if args.feature_schema != "base-v1":
        source_config["feature_schema"] = args.feature_schema
    config_id = _identity(source_config)
    completed = _completed_games(checkpoint)
    red_heroes = list(RED_HEROES)
    blue_heroes = list(BLUE_HEROES)

    for world_seed in range(args.seed_start, args.seed_end):
        if (config_id, world_seed) in completed:
            continue
        source_game_id = _identity({"source_config_id": config_id, "world_seed": world_seed})

        agents: dict[str, ISMCTSAgent] = {}
        for side, roster in (("RED", RED_HEROES), ("BLUE", BLUE_HEROES)):
            observer = TerminalLabelObserver(
                out,
                source_game_id=source_game_id,
                world_seed=world_seed,
                agent_label=f"ISMCTS_{side}",
                red_heroes=red_heroes,
                blue_heroes=blue_heroes,
                source_revision=revision,
                dirty_tree_hash=dirty_hash,
                sample_every=args.sample_every,
                max_samples=args.max_samples_per_side,
                continuation_max_steps=args.continuation_max_steps,
                continuation_max_rounds=args.continuation_max_rounds,
                feature_schema=args.feature_schema,
            )
            agent = ISMCTSAgent(
                config=SearchConfig(
                    iterations=args.iterations,
                    cutoff_rounds=args.cutoff_rounds,
                    seed=agent_seed(world_seed, side),
                ),
                value_fn=HeuristicValue(),
                cutoff_observer=observer,
            )
            for name in roster:
                agents[hero_id(name)] = agent

        checkpoint_row: dict[str, Any] = {
            "source_config_id": config_id,
            "source_game_id": source_game_id,
            "world_seed": world_seed,
        }
        try:
            with _source_game_timeout(args.source_timeout_seconds):
                run_game(
                    red_heroes,
                    blue_heroes,
                    agents,
                    map_path=DEFAULT_MAP,
                    game_type=args.game_type,
                    seed=world_seed,
                    max_steps=args.source_max_steps,
                )
        except SourceGameTimeout:
            checkpoint_row.update(reason="wall_clock_timeout", winner=None)
        _checkpoint(
            checkpoint,
            checkpoint_row,
        )
        completed.add((config_id, world_seed))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
