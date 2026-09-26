"""Generate deterministic Phase-0 heuristic joint policy/value data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, StrictInt, model_validator
from tqdm import tqdm

from automata.agents import HeuristicAgent, PlanningKind
from automata.agents.contracts import Agent, PlanningDecision
from automata.decision import DecisionDescriptor
from automata.evaluation.provenance import repository_root, source_identity
from automata.harness.game_runner import DEFAULT_MAP, RunResult, run_game
from automata.models.contracts import DecisionObservation, canonical_json_bytes
from automata.observation import encode_decision, legal_keys_for_decision
from automata.runtime.driver import BotDecision, DecisionKind
from automata.runtime.outcomes import WinnerSide
from automata.training.dataset import (
    JointDatasetRecorder,
    JointDatasetRow,
    PolicySource,
    iter_joint_dataset,
    write_joint_dataset,
)
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT
from goa2.domain.input import InputRequest, InputRequestType
from goa2.domain.models.card import Card
from goa2.domain.models.unit import Hero
from goa2.domain.state import GameState
from goa2.engine.phases import planning_open_for_second_card

TARGET_RECIPE = "one-hot-exact-choice"
SOFT_CARD_TARGET_RECIPE = "softmax-heuristic-card"
SEED_DERIVATION = "sha256(world_seed,side,phase0-heuristic-bootstrap-v1)"
OUTCOME_CONTRACT = "raw-winner+canonical-side-v1"

_DEFAULT_PILOT_MODE = "heuristic"
_DIVERSE_PILOT_MODE = "diverse"
_DEFAULT_CARD_TARGET_TEMPERATURE = 1.0
_DEFAULT_CARD_TARGET_UNIFORM_MASS = 0.1


@dataclass(frozen=True)
class GameVariant:
    game_type: Literal["QUICK", "LONG"]
    red_heroes: tuple[str, ...]
    blue_heroes: tuple[str, ...]


_ORIGINAL_RED = ("Wasp", "Xargatha")
_ORIGINAL_BLUE = ("Arien", "Brogan")
BALANCED_VARIANTS = (
    GameVariant("QUICK", _ORIGINAL_RED, _ORIGINAL_BLUE),
    GameVariant("QUICK", _ORIGINAL_BLUE, _ORIGINAL_RED),
    GameVariant("LONG", _ORIGINAL_RED, _ORIGINAL_BLUE),
    GameVariant("LONG", _ORIGINAL_BLUE, _ORIGINAL_RED),
)
BALANCED_VARIANT_SCHEDULE = tuple(
    {
        "game_type": variant.game_type,
        "red_heroes": variant.red_heroes,
        "blue_heroes": variant.blue_heroes,
    }
    for variant in BALANCED_VARIANTS
)


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
    winner_side: WinnerSide | None
    rounds: StrictInt | None
    turns: StrictInt | None
    steps: StrictInt | None

    @model_validator(mode="after")
    def _validate_outcome(self) -> CheckpointRow:
        if self.completed and self.reason == "game_over":
            if (self.winner is None) != (self.winner_side is None):
                raise ValueError("terminal winner and winner_side must have matching nullability")
            if (
                self.winner is not None
                and self.winner.upper() in {"RED", "BLUE"}
                and self.winner.upper() != self.winner_side
            ):
                raise ValueError("raw team winner disagrees with winner_side")
        elif self.winner is not None or self.winner_side is not None:
            raise ValueError("incomplete checkpoint rows cannot declare a winner")
        return self


class GeneratorSeedRange(BaseModel):
    """Half-open world-seed range represented by one published dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    start: StrictInt
    end: StrictInt


class GeneratorProvenance(BaseModel):
    """Deterministic sidecar describing the exact resolved generator run."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    seed_range: GeneratorSeedRange
    generator_config_id: str
    generation_id: str
    search_config_id: str
    generator_config: dict[str, Any]
    target_provenance: dict[str, Any]


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


def variant_for_world_seed(world_seed: int, schedule: str = "fixed") -> GameVariant:
    """Return the process-independent game variant assigned to ``world_seed``."""
    if schedule == "fixed":
        scope = PHASE0_EXPERIMENT
        if scope.game_type not in {"QUICK", "LONG"}:
            raise ValueError(f"unsupported engine game type {scope.game_type!r}")
        game_type = cast(Literal["QUICK", "LONG"], scope.game_type)
        return GameVariant(game_type, scope.red_heroes, scope.blue_heroes)
    if schedule != "balanced":
        raise ValueError(f"unknown variant schedule {schedule!r}")
    owned = PHASE0_EXPERIMENT.seed_registry.range_for("bootstrap")
    return BALANCED_VARIANTS[(world_seed - owned.start) % len(BALANCED_VARIANTS)]


class UniformPlanningAgent:
    """Uniform legal planning wrapped around exact heuristic downstream play."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.heuristic = HeuristicAgent(seed)

    def choose_planning(self, state: GameState, hero: Hero) -> PlanningDecision:
        choices: list[Card | None] = list(hero.hand)
        can_finish = planning_open_for_second_card(state, hero.id)
        if can_finish:
            choices.append(None)
        if not choices:
            return PlanningDecision.pass_()
        selected = self._rng.choice(choices)
        return PlanningDecision.finish() if selected is None else PlanningDecision.commit(selected)

    def choose_input(
        self,
        state: GameState,
        request: InputRequest,
        *,
        owned_hero_ids: frozenset[str] | None = None,
        decision_owner_hero_id: str | None = None,
    ) -> Any:
        return self.heuristic.choose_input(
            state,
            request,
            owned_hero_ids=owned_hero_ids,
            decision_owner_hero_id=decision_owner_hero_id,
        )


def build_agents(
    world_seed: int,
    red_heroes: Sequence[str] | None = None,
    blue_heroes: Sequence[str] | None = None,
    *,
    planning_behavior: str = "heuristic",
) -> dict[str, Agent]:
    """Build fresh agents, shared only among heroes on the same side."""
    scope = PHASE0_EXPERIMENT
    red_roster = scope.red_heroes if red_heroes is None else red_heroes
    blue_roster = scope.blue_heroes if blue_heroes is None else blue_heroes
    agents: dict[str, Agent] = {}
    for side, roster in (("RED", red_roster), ("BLUE", blue_roster)):
        seed = agent_seed(world_seed, side)
        if planning_behavior == "heuristic":
            agent: Agent = HeuristicAgent(seed)
        elif planning_behavior == "uniform":
            agent = UniformPlanningAgent(seed)
        else:
            raise ValueError(f"unknown planning behavior {planning_behavior!r}")
        agents.update({_hero_id(name): agent for name in roster})
    return agents


class HeuristicJointObserver:
    """Encode roots plus exact-input or optionally soft heuristic card targets.

    In the diverse pilot, planning behavior is uniform even though planning
    targets are derived from heuristic card scores. Input behavior and targets
    remain the heuristic's exact selected action.
    """

    def __init__(
        self,
        recorder: JointDatasetRecorder,
        *,
        planning_behavior: str = "heuristic",
        target_recipe: str = TARGET_RECIPE,
        card_target_temperature: float = _DEFAULT_CARD_TARGET_TEMPERATURE,
        card_target_uniform_mass: float = _DEFAULT_CARD_TARGET_UNIFORM_MASS,
    ) -> None:
        self.recorder = recorder
        self.planning_behavior = planning_behavior
        self.target_recipe = target_recipe
        self.card_target_temperature = card_target_temperature
        self.card_target_uniform_mass = card_target_uniform_mass
        self._heuristic = HeuristicAgent(0)

    def _card_target(
        self, state: GameState, hero: Hero, observation: DecisionObservation
    ) -> tuple[float, ...]:
        cards = {str(card.id): card for card in hero.hand}
        scores = [
            (
                self._heuristic.score_card(state, hero, cards[str(candidate.selection)])
                if candidate.selection is not None
                else 0.0
            )
            for candidate in observation.candidates
        ]
        maximum = max(scores)
        weights = [math.exp((score - maximum) / self.card_target_temperature) for score in scores]
        total = sum(weights)
        uniform = 1.0 / len(weights)
        return tuple(
            (1.0 - self.card_target_uniform_mass) * weight / total
            + self.card_target_uniform_mass * uniform
            for weight in weights
        )

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
        if decision.kind is DecisionKind.PLANNING and self.target_recipe == SOFT_CARD_TARGET_RECIPE:
            target = self._card_target(state, hero, observation)
        else:
            target = tuple(
                1.0 if index == selected_index else 0.0
                for index in range(len(observation.candidates))
            )
        policy_source: PolicySource = "HEURISTIC"
        if (
            decision.kind is DecisionKind.PLANNING
            and self.planning_behavior == "uniform"
            and self.target_recipe == SOFT_CARD_TARGET_RECIPE
        ):
            policy_source = "UNIFORM_PLANNING_SOFT_HEURISTIC"
        self.recorder.record_decision(
            observation=observation,
            policy_source=policy_source,
            policy_target=target,
            selected_candidate_id=candidate.candidate_id,
            selected_selection=candidate.selection,
        )

    def record_outcome(self, *, winner_side: WinnerSide | None, rounds: int, reason: str) -> None:
        self.recorder.record_outcome(winner_side=winner_side, rounds=rounds, reason=reason)


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


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number between zero and one")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-end", required=True, type=int)
    parser.add_argument("--target-source", required=True, choices=("heuristic",))
    parser.add_argument(
        "--target-recipe",
        choices=(TARGET_RECIPE, SOFT_CARD_TARGET_RECIPE),
        help="policy target recipe (default: exact, or soft card targets in diverse mode)",
    )
    parser.add_argument(
        "--pilot-mode",
        choices=(_DEFAULT_PILOT_MODE, _DIVERSE_PILOT_MODE),
        default=_DEFAULT_PILOT_MODE,
        help="heuristic keeps legacy generation; diverse enables balanced pilot defaults",
    )
    parser.add_argument(
        "--planning-behavior",
        choices=("heuristic", "uniform"),
        help="planning-card behavior (default: heuristic, or uniform in diverse mode)",
    )
    parser.add_argument(
        "--variant-schedule",
        choices=("fixed", "balanced"),
        help="game/composition schedule (default: fixed, or balanced in diverse mode)",
    )
    parser.add_argument(
        "--card-target-temperature",
        type=_positive_float,
        default=_DEFAULT_CARD_TARGET_TEMPERATURE,
    )
    parser.add_argument(
        "--card-target-uniform-mass",
        type=_unit_float,
        default=_DEFAULT_CARD_TARGET_UNIFORM_MASS,
    )
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
    args = _parser().parse_args(argv)
    if args.planning_behavior is None:
        args.planning_behavior = (
            "uniform" if args.pilot_mode == _DIVERSE_PILOT_MODE else "heuristic"
        )
    if args.variant_schedule is None:
        args.variant_schedule = "balanced" if args.pilot_mode == _DIVERSE_PILOT_MODE else "fixed"
    if args.target_recipe is None:
        args.target_recipe = (
            SOFT_CARD_TARGET_RECIPE if args.pilot_mode == _DIVERSE_PILOT_MODE else TARGET_RECIPE
        )
    return args


def generator_config(
    args: argparse.Namespace, *, source_revision: str, dirty_tree_hash: str
) -> dict[str, Any]:
    """Build the exact identity-bearing configuration for a parsed generator run."""
    scope = PHASE0_EXPERIMENT
    config: dict[str, Any] = {
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
        "outcome_contract": OUTCOME_CONTRACT,
        "max_steps": args.max_steps,
        "timeout_seconds": args.timeout_seconds,
        "target_source": args.target_source,
        "target_recipe": args.target_recipe,
        "search_config": None,
    }
    is_legacy = (
        args.pilot_mode == _DEFAULT_PILOT_MODE
        and args.planning_behavior == "heuristic"
        and args.variant_schedule == "fixed"
        and args.target_recipe == TARGET_RECIPE
    )
    if not is_legacy:
        schedule: object = (
            BALANCED_VARIANT_SCHEDULE
            if args.variant_schedule == "balanced"
            else (BALANCED_VARIANT_SCHEDULE[0],)
        )
        config["pilot"] = {
            "mode": args.pilot_mode,
            "planning_behavior": args.planning_behavior,
            "variant_schedule": schedule,
            "card_target": {
                "recipe": args.target_recipe,
                "temperature": args.card_target_temperature,
                "uniform_mass": args.card_target_uniform_mass,
            },
            "input_target_recipe": TARGET_RECIPE,
            "policy_sources": {
                "planning": (
                    "UNIFORM_PLANNING_SOFT_HEURISTIC"
                    if args.planning_behavior == "uniform"
                    and args.target_recipe == SOFT_CARD_TARGET_RECIPE
                    else "HEURISTIC"
                ),
                "input": "HEURISTIC",
            },
        }
    return config


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


def _search_target_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the historical target identity payload used in dataset rows."""
    if args.target_recipe == TARGET_RECIPE:
        return {
            "target_source": "HEURISTIC",
            "recipe": TARGET_RECIPE,
            "search": None,
        }
    return {
        "target_source": "HEURISTIC",
        "card_recipe": args.target_recipe,
        "card_temperature": args.card_target_temperature,
        "card_uniform_mass": args.card_target_uniform_mass,
        "input_recipe": TARGET_RECIPE,
        "search": None,
    }


def build_generator_provenance(
    args: argparse.Namespace,
    *,
    source_revision: str,
    dirty_tree_hash: str,
    seed_start: int,
    seed_end: int,
) -> GeneratorProvenance:
    """Build exact path-independent provenance for a published generator output."""
    config = generator_config(
        args,
        source_revision=source_revision,
        dirty_tree_hash=dirty_tree_hash,
    )
    config_id = _identity(config)
    generation_id = _identity({"phase": "PHASE0_EXPERIMENT", "generator": config})
    search_target = _search_target_config(args)
    planning_source = (
        "UNIFORM_PLANNING_SOFT_HEURISTIC"
        if args.planning_behavior == "uniform" and args.target_recipe == SOFT_CARD_TARGET_RECIPE
        else "HEURISTIC"
    )
    return GeneratorProvenance(
        seed_range=GeneratorSeedRange(start=seed_start, end=seed_end),
        generator_config_id=config_id,
        generation_id=generation_id,
        search_config_id=_identity(search_target),
        generator_config=config,
        target_provenance={
            "planning_behavior": args.planning_behavior,
            "planning_policy_source": planning_source,
            "card_target": {
                "recipe": args.target_recipe,
                "temperature": args.card_target_temperature,
                "uniform_mass": args.card_target_uniform_mass,
            },
            "input_behavior": "heuristic",
            "input_policy_source": "HEURISTIC",
            "input_target_recipe": TARGET_RECIPE,
            "search_identity_payload": search_target,
        },
    )


def generator_provenance_path(output: str | Path) -> Path:
    """Return the documented sidecar path for a generator output."""
    return Path(f"{output}.provenance.json")


def write_generator_provenance(output: str | Path, provenance: GeneratorProvenance) -> None:
    """Atomically publish canonical generator provenance beside a dataset."""
    _atomic_write(generator_provenance_path(output), canonical_json_bytes(provenance))


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
        winner_side=result.winner_side if result else None,
        rounds=result.rounds if result else None,
        turns=result.turns if result else None,
        steps=result.steps if result else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parse_generator_args(sys.argv[1:] if argv is None else argv)
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
        revision, dirty_hash = source_identity(
            exclude_paths=(
                repository_root() / "runs",
                out,
                checkpoint,
                _fragment_dir(out),
                generator_provenance_path(out),
            )
        )
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
    search_config_id = _identity(_search_target_config(args))
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
    outcomes: Counter[str] = Counter(
        row.winner_side or "draw"
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
            variant = variant_for_world_seed(world_seed, args.variant_schedule)
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
                game_type=variant.game_type,
                red_composition=variant.red_heroes,
                blue_composition=variant.blue_heroes,
                generation_id=generation_id,
                source_revision=revision,
                dirty_tree_hash=dirty_hash,
                source_model_digest=None,
                search_config_id=search_config_id,
                generator_config_id=config_id,
            )
            observer = HeuristicJointObserver(
                recorder,
                planning_behavior=args.planning_behavior,
                target_recipe=args.target_recipe,
                card_target_temperature=args.card_target_temperature,
                card_target_uniform_mass=args.card_target_uniform_mass,
            )
            result: RunResult | None = None
            reason: Literal["wall_clock_timeout", "exception"] | str = "exception"
            try:
                with _source_game_timeout(args.timeout_seconds):
                    result = run_game(
                        list(variant.red_heroes),
                        list(variant.blue_heroes),
                        build_agents(
                            world_seed,
                            variant.red_heroes,
                            variant.blue_heroes,
                            planning_behavior=args.planning_behavior,
                        ),
                        map_path=DEFAULT_MAP,
                        game_type=variant.game_type,
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
                outcomes[row.winner_side or "draw"] += 1
            else:
                outcomes[row.reason] += 1
            progress.set_postfix(dict(outcomes))
            progress.update()
    _publish_output(out, successful)
    if out.exists():
        write_generator_provenance(
            out,
            build_generator_provenance(
                args,
                source_revision=revision,
                dirty_tree_hash=dirty_hash,
                seed_start=args.seed_start,
                seed_end=args.seed_end,
            ),
        )
    else:
        generator_provenance_path(out).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
