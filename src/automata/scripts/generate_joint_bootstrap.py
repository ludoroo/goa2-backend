"""Generate deterministic Phase-0 heuristic joint policy/value data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from itertools import groupby
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictInt
from tqdm import tqdm

from automata.agents import HeuristicAgent, PlanningKind
from automata.decision import DecisionDescriptor
from automata.evaluation.provenance import source_identity
from automata.harness.game_runner import DEFAULT_MAP, RunResult, run_game
from automata.models.contracts import canonical_json_bytes
from automata.observation import encode_decision, legal_keys_for_decision
from automata.runtime.driver import BotDecision, DecisionKind
from automata.training.dataset import (
    JointDatasetRecorder,
    JointDatasetRow,
    iter_joint_dataset,
    write_joint_dataset,
)
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT
from goa2.domain.input import InputRequestType
from goa2.domain.state import GameState
from goa2.engine.phases import planning_open_for_second_card

TARGET_RECIPE = "one-hot-exact-choice"
SEED_DERIVATION = "sha256(world_seed,side,phase0-heuristic-bootstrap-v1)"


class SourceGameTimeout(TimeoutError):
    pass


class CheckpointRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    config_id: str
    game_id: str
    world_seed: StrictInt
    completed: bool
    reason: str
    winner: str | None
    rounds: StrictInt | None
    turns: StrictInt | None
    steps: StrictInt | None


def _identity(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def agent_seed(world_seed: int, side: str) -> int:
    """Pure per-world, per-side heuristic RNG seed derivation."""
    return int(
        _identity({"world_seed": world_seed, "side": side, "derivation": SEED_DERIVATION})[:16],
        16,
    )


def _hero_id(name: str) -> str:
    return f"hero_{name.lower().replace(' ', '_')}"


def build_agents(world_seed: int) -> dict[str, HeuristicAgent]:
    """Build fresh agents, shared only among heroes on the same side."""
    scope = PHASE0_EXPERIMENT
    agents: dict[str, HeuristicAgent] = {}
    for side, roster in (("RED", scope.red_heroes), ("BLUE", scope.blue_heroes)):
        agent = HeuristicAgent(agent_seed(world_seed, side))
        agents.update({_hero_id(name): agent for name in roster})
    return agents


class HeuristicJointObserver:
    """Encode exact pre-decision roots and the heuristic's exact chosen action."""

    def __init__(self, recorder: JointDatasetRecorder) -> None:
        self.recorder = recorder

    def record_decision(self, state: GameState, decision: BotDecision) -> None:
        hero = state.get_hero(decision.hero_id)
        if hero is None or hero.team is None:
            raise ValueError(f"unknown decision owner {decision.hero_id!r}")

        if decision.kind is DecisionKind.PLANNING:
            plan = decision.planning
            assert plan is not None
            search_decision = DecisionDescriptor(
                "CARD",
                hero=hero,
                can_finish_planning=planning_open_for_second_card(state, decision.hero_id),
            )
            if plan.kind is PlanningKind.COMMIT:
                assert plan.card is not None
                selected: Any = plan.card.id
            elif plan.kind is PlanningKind.FINISH:
                selected = None
            else:
                # Omit all PASS decisions that cannot be represented by legal policy candidates.
                return
        else:
            request = decision.request
            assert request is not None
            # UPGRADE_PHASE is a structured simultaneous action rather than the
            # branchable candidate contract represented by DecisionObservation v3.
            if request.request_type in {InputRequestType.UPGRADE_PHASE, InputRequestType.NONE}:
                return
            search_decision = DecisionDescriptor("INPUT", request=request)
            selected = decision.selection

        legal = legal_keys_for_decision(search_decision)
        if not legal:
            return
        observation = encode_decision(
            state,
            search_decision,
            legal,
            decision_owner_hero_id=str(decision.hero_id),
            perspective_team=hero.team.value,
        )
        selected_indexes = [
            index
            for index, candidate in enumerate(observation.candidates)
            if candidate.selection == selected
        ]
        if len(selected_indexes) != 1:
            raise ValueError(
                "chosen heuristic selection does not identify exactly one legal candidate"
            )
        selected_index = selected_indexes[0]
        candidate = observation.candidates[selected_index]
        target = tuple(
            1.0 if index == selected_index else 0.0 for index in range(len(observation.candidates))
        )
        self.recorder.record_decision(
            observation=observation,
            policy_source="HEURISTIC",
            policy_target=target,
            selected_candidate_id=candidate.candidate_id,
            selected_selection=candidate.selection,
        )

    def record_outcome(self, *, winner: str | None, rounds: int, reason: str) -> None:
        self.recorder.record_outcome(winner=winner, rounds=rounds, reason=reason)


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-end", required=True, type=int)
    parser.add_argument("--target-source", required=True, choices=("heuristic",))
    parser.add_argument("--target-recipe", required=True, choices=(TARGET_RECIPE,))
    parser.add_argument("--max-steps", required=True, type=_positive)
    parser.add_argument("--timeout-seconds", required=True, type=_positive_float)
    parser.add_argument("--source-revision", help=argparse.SUPPRESS)
    parser.add_argument("--dirty-tree-hash", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false", help="disable progress output"
    )
    return parser


def parse_generator_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse generator CLI options so parent and worker use identical defaults."""
    return _parser().parse_args(argv)


def generator_config(
    args: argparse.Namespace, *, source_revision: str, dirty_tree_hash: str
) -> dict[str, Any]:
    """Build the exact identity-bearing configuration for a parsed generator run."""
    scope = PHASE0_EXPERIMENT
    return {
        "scope": {
            "map_id": scope.map_id,
            "map_path": DEFAULT_MAP,
            "game_type": scope.game_type,
            "red_heroes": scope.red_heroes,
            "blue_heroes": scope.blue_heroes,
            "seed_purpose": "bootstrap",
        },
        "source_revision": source_revision,
        "dirty_tree_hash": dirty_tree_hash,
        "seed_derivation": SEED_DERIVATION,
        "max_steps": args.max_steps,
        "timeout_seconds": args.timeout_seconds,
        "target_source": args.target_source,
        "target_recipe": args.target_recipe,
        "search_config": None,
    }


def generator_config_id(
    args: argparse.Namespace, *, source_revision: str, dirty_tree_hash: str
) -> str:
    return _identity(
        generator_config(
            args,
            source_revision=source_revision,
            dirty_tree_hash=dirty_tree_hash,
        )
    )


@contextmanager
def _source_game_timeout(seconds: float) -> Iterator[None]:
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        raise RuntimeError("source game timeouts require a POSIX main thread")
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise SourceGameTimeout

    signal.signal(signal.SIGALRM, raise_timeout)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _read_checkpoint(path: Path) -> tuple[list[CheckpointRow], bytes]:
    if not path.exists():
        return [], b""
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        boundary = payload.rfind(b"\n")
        payload = payload[: boundary + 1] if boundary >= 0 else b""
    rows: list[CheckpointRow] = []
    for line_number, raw in enumerate(payload.splitlines(), 1):
        if not raw:
            raise ValueError(f"invalid checkpoint row {line_number}: blank row")
        try:
            row = CheckpointRow.model_validate_json(raw)
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint row {line_number}: {exc}") from exc
        if raw != canonical_json_bytes(row):
            raise ValueError(f"invalid checkpoint row {line_number}: non-canonical JSON")
        rows.append(row)
    return rows, payload


def _append_checkpoint(path: Path, row: CheckpointRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fragment_dir(out: Path) -> Path:
    return out.parent / f".{out.name}.games"


def _fragment_path(directory: Path, game_id: str) -> Path:
    return directory / f"{game_id}.jsonl.zst"


def _publish_output(
    out: Path,
    successful: Mapping[str, int],
    *,
    publish: bool = True,
) -> None:
    """Reconcile fragments and optionally publish world-seed ordered bytes."""
    directory = _fragment_dir(out)

    def fragment_for(game_id: str) -> Path | None:
        compressed = _fragment_path(directory, game_id)
        if compressed.exists():
            return compressed
        plain = directory / f"{game_id}.jsonl"
        return plain if plain.exists() else None

    missing_fragments = {game_id for game_id in successful if fragment_for(game_id) is None}
    if missing_fragments:
        if not out.exists() or not out.stat().st_size:
            missing = min(missing_fragments)
            raise RuntimeError(f"completed game {missing} has no recoverable dataset fragment")
        # Validate the complete aggregate before recovering any fragments;
        # iterator failures can occur after rows have already been yielded.
        for _ in iter_joint_dataset(out):
            pass
        recovered_games: set[str] = set()
        for game_id, rows in groupby(iter_joint_dataset(out), lambda row: row.game_id):
            if game_id in recovered_games:
                raise ValueError(f"aggregate game {game_id!r} is not stored contiguously")
            recovered_games.add(game_id)
            if game_id not in missing_fragments:
                for _ in rows:
                    pass
                continue
            write_joint_dataset(_fragment_path(directory, game_id), rows)
        still_missing = missing_fragments - recovered_games
        if still_missing:
            missing = min(still_missing)
            raise RuntimeError(f"completed game {missing} has no recoverable dataset fragment")

    if directory.exists():
        for path in (*directory.glob("*.jsonl"), *directory.glob("*.jsonl.zst")):
            game_id = path.name.removesuffix(".jsonl.zst").removesuffix(".jsonl")
            if game_id not in successful:
                path.unlink()
    fragments: list[tuple[int, str, Path]] = []
    for game_id, seed in successful.items():
        fragment_path = fragment_for(game_id)
        if fragment_path is None:  # pragma: no cover - checked during reconciliation
            raise RuntimeError(f"completed game {game_id} has no recoverable dataset fragment")
        fragments.append((seed, game_id, fragment_path))

    if not publish:
        if not fragments:
            out.unlink(missing_ok=True)
        return

    def ordered_rows() -> Iterator[JointDatasetRow]:
        seed_games: dict[int, str] = {}
        for seed, game_id, path in sorted(fragments):
            previous_game = seed_games.setdefault(seed, game_id)
            if previous_game != game_id:
                raise ValueError(f"world seed {seed} belongs to multiple completed games")
            count = 0
            for row in iter_joint_dataset(path):
                if row.game_id != game_id or row.world_seed != seed:
                    raise ValueError(f"fragment {path} does not contain its named game and seed")
                count += 1
                yield row
            if not count:  # iter_joint_dataset currently reports this first
                raise ValueError(f"fragment {path} is empty")  # pragma: no cover

    if fragments:
        write_joint_dataset(out, ordered_rows())
    else:
        out.unlink(missing_ok=True)


def _checkpoint_row(
    config_id: str, game_id: str, world_seed: int, result: RunResult | None, reason: str
) -> CheckpointRow:
    return CheckpointRow(
        config_id=config_id,
        game_id=game_id,
        world_seed=world_seed,
        completed=result is not None and result.reason == "game_over",
        reason=reason,
        winner=result.winner if result else None,
        rounds=result.rounds if result else None,
        turns=result.turns if result else None,
        steps=result.steps if result else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    owned = PHASE0_EXPERIMENT.seed_registry.range_for("bootstrap")
    if (
        args.seed_start < owned.start
        or args.seed_end > owned.stop
        or args.seed_end <= args.seed_start
    ):
        parser.error(
            f"seed range must be a non-empty half-open subset of bootstrap [{owned.start}, {owned.stop})"
        )

    # No filesystem or game work occurs before scope ownership is accepted.
    out, checkpoint = Path(args.out), Path(args.checkpoint)
    if bool(args.source_revision) != bool(args.dirty_tree_hash):
        parser.error("--source-revision and --dirty-tree-hash must be supplied together")
    if args.source_revision:
        revision, dirty_hash = args.source_revision, args.dirty_tree_hash
    else:
        revision, dirty_hash = source_identity(exclude_paths=(out, checkpoint, _fragment_dir(out)))
    scope = PHASE0_EXPERIMENT
    config = generator_config(
        args,
        source_revision=revision,
        dirty_tree_hash=dirty_hash,
    )
    config_id = generator_config_id(
        args,
        source_revision=revision,
        dirty_tree_hash=dirty_hash,
    )
    generation_id = _identity({"phase": "PHASE0_EXPERIMENT", "generator": config})
    search_config_id = _identity(
        {"target_source": "HEURISTIC", "recipe": TARGET_RECIPE, "search": None}
    )
    rows, canonical_checkpoint = _read_checkpoint(checkpoint)
    if checkpoint.exists() and checkpoint.read_bytes() != canonical_checkpoint:
        _atomic_write(checkpoint, canonical_checkpoint)
    successful = {
        row.game_id: row.world_seed
        for row in rows
        if row.config_id == config_id and row.completed and row.reason == "game_over"
    }
    # Per-game fragments and checkpoints are the durable resume state. Avoid
    # rebuilding the growing aggregate here and after every game; doing so makes
    # generation quadratic in the number of completed games.
    _publish_output(out, successful, publish=False)

    requested_seeds = range(args.seed_start, args.seed_end)
    resumed = sum(seed in requested_seeds for seed in successful.values())
    outcomes = Counter(
        row.winner or "draw"
        for row in rows
        if row.config_id == config_id
        and row.completed
        and row.reason == "game_over"
        and row.world_seed in requested_seeds
    )
    with tqdm(
        total=len(requested_seeds),
        initial=resumed,
        desc="Generating",
        unit="game",
        disable=not args.progress,
    ) as progress:
        progress.set_postfix(dict(outcomes))
        for world_seed in requested_seeds:
            game_id = _identity({"generator_config_id": config_id, "world_seed": world_seed})
            if game_id in successful:
                continue
            fragment = _fragment_path(_fragment_dir(out), game_id)
            fragment.unlink(missing_ok=True)
            recorder = JointDatasetRecorder(
                fragment,
                game_id=game_id,
                world_seed=world_seed,
                map_id=scope.map_id,
                game_type=scope.game_type,
                red_composition=scope.red_heroes,
                blue_composition=scope.blue_heroes,
                generation_id=generation_id,
                source_revision=revision,
                dirty_tree_hash=dirty_hash,
                source_model_digest=None,
                search_config_id=search_config_id,
                generator_config_id=config_id,
            )
            observer = HeuristicJointObserver(recorder)
            result: RunResult | None = None
            reason: Literal["wall_clock_timeout", "exception"] | str = "exception"
            try:
                with _source_game_timeout(args.timeout_seconds):
                    result = run_game(
                        list(scope.red_heroes),
                        list(scope.blue_heroes),
                        build_agents(world_seed),
                        map_path=DEFAULT_MAP,
                        game_type=scope.game_type,
                        seed=world_seed,
                        max_steps=args.max_steps,
                        decision_observer=observer,
                    )
                reason = result.reason
            except SourceGameTimeout:
                reason = "wall_clock_timeout"
            except BaseException:
                recorder.close()
                _append_checkpoint(
                    checkpoint, _checkpoint_row(config_id, game_id, world_seed, None, "exception")
                )
                raise
            finally:
                recorder.close()
            row = _checkpoint_row(config_id, game_id, world_seed, result, reason)
            _append_checkpoint(checkpoint, row)
            if row.completed:
                successful[game_id] = world_seed
                outcomes[row.winner or "draw"] += 1
            else:
                outcomes[row.reason] += 1
            progress.set_postfix(dict(outcomes))
            progress.update()
    _publish_output(out, successful)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
