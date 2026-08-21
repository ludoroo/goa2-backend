"""Generate a resumable expert-search policy dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import tempfile
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..evaluation.cli import source_identity
from ..evaluation.matchup import hero_id
from ..evaluation.policy_dataset import PolicyDatasetRecorder
from ..runtime.harness import DEFAULT_MAP, run_game
from ..search import ISMCTSAgent, SearchConfig

RED_HEROES: tuple[str, ...] = ("Wasp", "Xargatha")
BLUE_HEROES: tuple[str, ...] = ("Arien", "Brogan")


class SourceGameTimeout(TimeoutError):
    """Raised when a source game exceeds its wall-clock budget."""


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
    parser = argparse.ArgumentParser(description="Generate expert ISMCTS policy data.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-end", required=True, type=int)
    parser.add_argument("--expert-iterations", type=_positive, default=16)
    parser.add_argument("--expert-cutoff-rounds", type=_positive, default=2)
    parser.add_argument("--source-max-steps", type=_positive, default=20_000)
    parser.add_argument("--source-timeout-seconds", type=_positive_float, default=1800.0)
    parser.add_argument("--game-type", required=True)
    return parser


def _identity(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def agent_seed(world_seed: int, side: str) -> int:
    """Return a stable, explicit per-game and per-side search seed."""
    return int(_identity({"world_seed": world_seed, "side": side})[:16], 16)


def _read_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    successful: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                if type(row.get("config_id")) is not str:
                    raise ValueError("config_id must be a string")
                if type(row.get("game_id")) is not str:
                    raise ValueError("game_id must be a string")
                if type(row.get("world_seed")) is not int:
                    raise ValueError("world_seed must be an integer")
                if type(row.get("completed")) is not bool:
                    raise ValueError("completed must be a boolean")
                if type(row.get("reason")) is not str:
                    raise ValueError("reason must be a string")
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid checkpoint row {line_number}: {exc}") from exc
            if row["completed"] is True and row["reason"] == "game_over":
                successful.add(row["game_id"])
    return successful


def _reconcile_output(path: Path, successful_game_ids: set[str]) -> None:
    """Atomically retain only rows backed by successful checkpoints."""
    if not path.exists():
        return
    retained: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or type(row.get("game_id")) is not str:
                    raise ValueError("row must be an object with a string game_id")
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid policy row {line_number}: {exc}") from exc
            if row["game_id"] in successful_game_ids:
                retained.append(line if line.endswith("\n") else f"{line}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.writelines(retained)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


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
    checkpoint = Path(args.checkpoint)
    revision, dirty_hash = source_identity()
    base_config = SearchConfig(
        iterations=args.expert_iterations,
        cutoff_rounds=args.expert_cutoff_rounds,
    )
    expert_config = asdict(base_config)
    # Search seeds are derived per game and side rather than fixed in this template.
    del expert_config["seed"]
    expert_config.update(
        seed_derivation="sha256(world_seed,side)",
        default_policy="HeuristicAgent",
        value_fn="HeuristicValue",
        prior="HeuristicPrior",
    )
    source_config = {
        "expert_config": expert_config,
        "red_heroes": list(RED_HEROES),
        "blue_heroes": list(BLUE_HEROES),
        "map_path": DEFAULT_MAP,
        "source_revision": revision,
        "dirty_tree_hash": dirty_hash,
        "source_max_steps": args.source_max_steps,
        "source_timeout_seconds": args.source_timeout_seconds,
        "game_type": args.game_type,
    }
    expert_identity = _identity(source_config)
    config_id = _identity({**source_config, "expert_identity": expert_identity})

    successful = _read_checkpoint(checkpoint)
    _reconcile_output(out, successful)
    red_heroes = list(RED_HEROES)
    blue_heroes = list(BLUE_HEROES)

    for world_seed in range(args.seed_start, args.seed_end):
        game_id = _identity({"config_id": config_id, "world_seed": world_seed})
        if game_id in successful:
            continue

        recorder = PolicyDatasetRecorder(
            out,
            game_id=game_id,
            world_seed=world_seed,
            red_heroes=red_heroes,
            blue_heroes=blue_heroes,
            source_revision=revision,
            dirty_tree_hash=dirty_hash,
            expert_config=expert_config,
            expert_identity=expert_identity,
            append=True,
        )
        agents: dict[str, ISMCTSAgent] = {}
        for side, roster in (("RED", RED_HEROES), ("BLUE", BLUE_HEROES)):
            agent = ISMCTSAgent(
                config=replace(base_config, seed=agent_seed(world_seed, side)),
                root_observer=recorder,
            )
            for name in roster:
                agents[hero_id(name)] = agent

        checkpoint_row: dict[str, Any] = {
            "config_id": config_id,
            "game_id": game_id,
            "world_seed": world_seed,
        }
        try:
            with _source_game_timeout(args.source_timeout_seconds):
                result = run_game(
                    red_heroes,
                    blue_heroes,
                    agents,
                    map_path=DEFAULT_MAP,
                    game_type=args.game_type,
                    seed=world_seed,
                    max_steps=args.source_max_steps,
                    recorder=recorder,
                )
        except SourceGameTimeout:
            checkpoint_row.update(completed=False, reason="wall_clock_timeout", winner=None)
        except BaseException:
            checkpoint_row.update(completed=False, reason="exception", winner=None)
            _checkpoint(checkpoint, checkpoint_row)
            raise
        else:
            completed = result.reason == "game_over"
            checkpoint_row.update(
                completed=completed,
                reason=result.reason,
                winner=result.winner,
                rounds=result.rounds,
                turns=result.turns,
                steps=result.steps,
            )
        _checkpoint(checkpoint, checkpoint_row)
        if checkpoint_row["completed"]:
            successful.add(game_id)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
