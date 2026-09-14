"""Framework-neutral contracts for persistent, complete-game self-play workers."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt

from automata.agents.contracts import Agent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.decision import DecisionDescriptor
from automata.harness.game_runner import RunResult, run_game
from automata.observation import encode_decision
from automata.runtime.driver import BotDecision
from automata.search.config import SearchConfig
from automata.search.ismcts.strategy import SearchStrategy, StrategyResult
from automata.search.node import Key
from automata.search.root import RootTarget
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT
from automata.training.io import atomic_write_bytes as _atomic_write
from automata.training.io import canonical_json_bytes as _canonical
from automata.training.io import content_digest as _digest
from automata.training.search_targets import SearchActionTarget
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

from .dataset import JointDatasetRecorder

WORKER_COUNT = 4
SEED_DERIVATION = "sha256(self-play-agent-v1,worker-config,world-seed,side)"


def _copy_json_mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    copied = json.loads(_canonical(dict(value)))
    if not isinstance(copied, dict):  # pragma: no cover - guaranteed by dict(value)
        raise TypeError("configuration must be a JSON object")
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Immutable generation and champion identity shared by every worker."""

    generation_id: str
    parent_model_digest: str
    parent_generation: int
    observation_schema_version: int
    source_revision: str
    dirty_tree_hash: str
    search_config: Mapping[str, JsonValue]
    source_config: Mapping[str, JsonValue]
    max_steps: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.generation_id or not self.source_revision or not self.dirty_tree_hash:
            raise ValueError("generation and source identity fields must be non-empty")
        if len(self.parent_model_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.parent_model_digest
        ):
            raise ValueError("parent_model_digest must be a lowercase SHA-256 digest")
        if self.parent_generation < 0 or self.observation_schema_version <= 0:
            raise ValueError("parent generation and observation schema are invalid")
        if self.max_steps <= 0 or self.timeout_seconds <= 0:
            raise ValueError("worker limits must be positive")
        object.__setattr__(self, "search_config", _copy_json_mapping(self.search_config))
        object.__setattr__(self, "source_config", _copy_json_mapping(self.source_config))

    @property
    def search_config_id(self) -> str:
        return _digest(dict(self.search_config))

    @property
    def source_config_id(self) -> str:
        return _digest(dict(self.source_config))

    @property
    def generator_config_id(self) -> str:
        return _digest(
            {
                "generation_id": self.generation_id,
                "parent_model_digest": self.parent_model_digest,
                "parent_generation": self.parent_generation,
                "observation_schema_version": self.observation_schema_version,
                "source_revision": self.source_revision,
                "dirty_tree_hash": self.dirty_tree_hash,
                "search_config": dict(self.search_config),
                "source_config": dict(self.source_config),
                "max_steps": self.max_steps,
                "timeout_seconds": self.timeout_seconds,
                "seed_derivation": SEED_DERIVATION,
            }
        )


@dataclass(frozen=True, slots=True)
class GameSpec:
    """One deterministic training world assigned to exactly one worker."""

    world_seed: int
    map_id: str
    map_path: str
    game_type: str
    red_composition: tuple[str, ...]
    blue_composition: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.world_seed < 0:
            raise ValueError("world seed must be non-negative")
        if not all((self.map_id, self.map_path, self.game_type)):
            raise ValueError("game scope fields must be non-empty")
        if not self.red_composition or not self.blue_composition:
            raise ValueError("both self-play compositions must be non-empty")

    def game_id(self, config: GenerationConfig) -> str:
        return _digest(
            {
                "generator_config_id": config.generator_config_id,
                "world_seed": self.world_seed,
                "map_id": self.map_id,
                "game_type": self.game_type,
                "red_composition": self.red_composition,
                "blue_composition": self.blue_composition,
            }
        )


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    worker_id: int
    config: GenerationConfig
    games: tuple[GameSpec, ...]

    def __post_init__(self) -> None:
        if self.worker_id < 0:
            raise ValueError("worker_id must be non-negative")
        seeds = tuple(game.world_seed for game in self.games)
        if len(seeds) != len(set(seeds)):
            raise ValueError("a worker cannot contain duplicate world seeds")

    @property
    def worker_config_id(self) -> str:
        return _digest(
            {
                "worker_id": self.worker_id,
                "generator_config_id": self.config.generator_config_id,
                "games": [game.game_id(self.config) for game in self.games],
            }
        )


def build_worker_specs(
    config: GenerationConfig,
    games: Sequence[GameSpec],
    *,
    worker_count: int = WORKER_COUNT,
) -> tuple[WorkerSpec, ...]:
    """Assign sorted training seeds round-robin into deterministic disjoint specs."""
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    ordered = sorted(games, key=lambda game: game.world_seed)
    seeds = [game.world_seed for game in ordered]
    if len(seeds) != len(set(seeds)):
        raise ValueError("self-play game seeds must be globally unique")
    registry = PHASE0_EXPERIMENT.seed_registry
    for seed in seeds:
        try:
            registry.require_seed(seed, purpose="training")
        except ValueError as exc:
            raise ValueError(f"self-play seed {seed} is not owned by training") from exc
    buckets: list[list[GameSpec]] = [[] for _ in range(worker_count)]
    for index, game in enumerate(ordered):
        buckets[index % worker_count].append(game)
    return tuple(
        WorkerSpec(worker_id=index, config=config, games=tuple(bucket))
        for index, bucket in enumerate(buckets)
    )


@dataclass(frozen=True, slots=True)
class LoadedChampionRuntime:
    """Opaque loaded runtime plus the identity validated before any game starts."""

    runtime: object
    model_digest: str
    generation: int
    observation_schema_version: int


class CheckpointRow(BaseModel):
    """A durable receipt for one and only one complete terminal game."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    worker_id: StrictInt = Field(ge=0)
    worker_config_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_config_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_config_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_id: str = Field(min_length=1)
    parent_model_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_generation: StrictInt = Field(ge=0)
    observation_schema_version: StrictInt = Field(gt=0)
    game_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_seed: StrictInt
    fragment_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: StrictInt = Field(gt=0)
    reason: Literal["game_over"] = "game_over"
    winner: Literal["RED", "BLUE"] | None
    rounds: StrictInt | None
    turns: StrictInt | None
    steps: StrictInt | None


class TelemetryEvent(BaseModel):
    """Structured worker telemetry suitable for JSONL logging."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    event: Literal["worker_started", "progress", "game_complete", "timeout", "error"]
    worker_id: int = Field(ge=0)
    game_id: str | None = None
    world_seed: int | None = None
    completed_games: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)
    games_per_second: float = Field(ge=0.0)
    reason: str | None = None
    error_type: str | None = None


class RuntimeLoader(Protocol):
    def __call__(self, config: GenerationConfig) -> LoadedChampionRuntime: ...


class StrategyFactory(Protocol):
    def __call__(
        self, runtime: object, game: GameSpec, side: Literal["RED", "BLUE"], seed: int
    ) -> SearchStrategy: ...


class AgentFactory(Protocol):
    def __call__(self, strategy: SearchStrategy, seed: int) -> Agent: ...


GameRunner = Callable[..., RunResult]
TelemetrySink = Callable[[TelemetryEvent], None]


def agent_seed(config: GenerationConfig, world_seed: int, side: str) -> int:
    return int(
        _digest(
            {
                "derivation": SEED_DERIVATION,
                "generator_config_id": config.generator_config_id,
                "world_seed": world_seed,
                "side": side,
            }
        )[:16],
        16,
    )


class _RecordingStrategy:
    """Record the improved policy at the exact state passed to ``select``."""

    def __init__(self, delegate: SearchStrategy, recorder: JointDatasetRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self.strategy_id = delegate.strategy_id

    def select(
        self,
        state: GameState,
        perspective_team: TeamColor,
        root_target: RootTarget,
        legal_candidates: Sequence[Key],
    ) -> StrategyResult[Key]:
        legal = tuple(legal_candidates)
        result = self._delegate.select(state, perspective_team, root_target, legal)
        if result.candidates != legal:
            raise ValueError("search strategy changed or reordered the legal candidates")
        owner = state.get_hero(HeroID(root_target.decision_owner_hero_id))
        if owner is None:
            raise ValueError("search root decision owner is absent")
        request = None
        if root_target.kind == "INPUT":
            request = next(
                (item for item in reversed(state.input_stack) if item.id == root_target.request_id),
                None,
            )
            if request is None or request.player_id != root_target.player_id:
                raise ValueError("search root request does not match the predecision state")
        decision = DecisionDescriptor(
            root_target.kind,
            hero=owner if root_target.kind == "CARD" else None,
            request=request,
            can_finish_planning=root_target.kind == "CARD" and None in legal,
        )
        observation = encode_decision(
            state,
            decision,
            legal,
            decision_owner_hero_id=owner.id,
            perspective_team=perspective_team.value,
        )
        actions = _aligned_action_stats(result, observation.candidates)
        selected = observation.candidates[result.selected_index]
        self._recorder.record_decision(
            observation=observation,
            policy_source="ISMCTS_VISITS",
            policy_target=tuple(cast(float, action.improved_probability) for action in actions),
            selected_candidate_id=selected.candidate_id,
            selected_selection=selected.selection,
            action_stats=actions,
        )
        return result


def _aligned_action_stats(
    result: StrategyResult[Key], candidates: Sequence[Any]
) -> tuple[SearchActionTarget, ...]:
    statistics = result.search_result
    if statistics is not None:
        if statistics.best_key != result.selected_candidate:
            raise ValueError("search statistics selected candidate disagrees with the action")
        visits = tuple(
            statistics.root.children[key].visits if key in statistics.root.children else 0
            for key in result.candidates
        )
        total_visits = sum(visits)
        if not total_visits:
            raise ValueError("self-play search statistics contain no root visits")
        return tuple(
            SearchActionTarget(
                schema_version=1,
                candidate=candidate,
                sample_count=visits[index],
                mean_value=(
                    statistics.root.children[key].q if key in statistics.root.children else 0.0
                ),
                value_variance=(
                    statistics.root.children[key].value_variance
                    if key in statistics.root.children
                    else 0.0
                ),
                improved_probability=visits[index] / total_visits,
                selected=index == result.selected_index,
            )
            for index, (key, candidate) in enumerate(
                zip(result.candidates, candidates, strict=True)
            )
        )
    raise ValueError("self-play search strategy must return improved action statistics")


class _OutcomeObserver:
    def __init__(self, recorder: JointDatasetRecorder) -> None:
        self._recorder = recorder

    def record_decision(self, state: GameState, decision: BotDecision) -> None:
        del state, decision  # Root decisions are recorded synchronously by _RecordingStrategy.

    def record_outcome(self, *, winner: str | None, rounds: int, reason: str) -> None:
        self._recorder.record_outcome(winner=winner, rounds=rounds, reason=reason)


class SourceGameTimeout(TimeoutError):
    pass


@contextmanager
def _game_timeout(seconds: float) -> Iterator[None]:
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def expire(_signum: int, _frame: Any) -> None:
        raise SourceGameTimeout

    signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


class SelfPlayWorker:
    """Load one champion and run all incomplete games assigned to one worker."""

    def __init__(
        self,
        spec: WorkerSpec,
        *,
        output_dir: str | Path,
        checkpoint_path: str | Path,
        runtime_loader: RuntimeLoader,
        strategy_factory: StrategyFactory,
        agent_factory: AgentFactory | None = None,
        game_runner: GameRunner = run_game,
        telemetry: TelemetrySink | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.spec = spec
        self.output_dir = Path(output_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.runtime_loader = runtime_loader
        self.strategy_factory = strategy_factory
        self.agent_factory = agent_factory or self._default_agent
        self.game_runner = game_runner
        self.telemetry = telemetry or (lambda _event: None)
        self.clock = clock

    @staticmethod
    def _default_agent(strategy: SearchStrategy, seed: int) -> Agent:
        return ISMCTSAgent(SearchConfig(seed=seed), strategy=strategy)

    def run(self) -> int:
        started = self.clock()
        checkpoints = self._load_checkpoints()
        self._reconcile_fragments(checkpoints)
        pending = [
            game for game in self.spec.games if game.game_id(self.spec.config) not in checkpoints
        ]
        if not pending:
            return 0
        loaded = self.runtime_loader(self.spec.config)
        self._validate_runtime(loaded)
        self._emit("worker_started", started, completed=0)
        completed = 0
        for game in pending:
            game_id = game.game_id(self.spec.config)
            fragment = self.fragment_path(game)
            fragment.unlink(missing_ok=True)
            recorder = self._recorder(fragment, game, game_id)
            agents = self._build_agents(loaded.runtime, game, recorder)
            result: RunResult | None = None

            def report_progress(
                _round: int,
                _steps: int,
                *,
                current_game: GameSpec = game,
                completed_before_game: int = completed,
            ) -> None:
                self._emit(
                    "progress",
                    started,
                    completed=completed_before_game,
                    game=current_game,
                )

            try:
                with _game_timeout(self.spec.config.timeout_seconds):
                    result = self.game_runner(
                        list(game.red_composition),
                        list(game.blue_composition),
                        agents,
                        map_path=game.map_path,
                        game_type=game.game_type,
                        seed=game.world_seed,
                        max_steps=self.spec.config.max_steps,
                        decision_observer=_OutcomeObserver(recorder),
                        progress_callback=report_progress,
                    )
            except SourceGameTimeout:
                recorder.close()
                self._emit(
                    "timeout",
                    started,
                    completed=completed,
                    game=game,
                    reason="wall_clock_timeout",
                )
                continue
            except BaseException as exc:
                recorder.close()
                self._emit(
                    "error",
                    started,
                    completed=completed,
                    game=game,
                    reason=str(exc),
                    error_type=type(exc).__name__,
                )
                raise
            finally:
                recorder.close()
            if result.reason != "game_over":
                self._emit(
                    "timeout",
                    started,
                    completed=completed,
                    game=game,
                    reason=result.reason,
                )
                continue
            if not fragment.exists():
                raise RuntimeError("complete self-play game produced no decision fragment")
            row = self._checkpoint_for_fragment(game, game_id, fragment, result)
            self._append_checkpoint(row)
            checkpoints[game_id] = row
            completed += 1
            self._emit("game_complete", started, completed=completed, game=game)
        return completed

    def fragment_path(self, game: GameSpec) -> Path:
        return (
            self.output_dir / f"{game.world_seed:020d}-{game.game_id(self.spec.config)}.jsonl.zst"
        )

    def _recorder(self, path: Path, game: GameSpec, game_id: str) -> JointDatasetRecorder:
        config = self.spec.config
        return JointDatasetRecorder(
            path,
            game_id=game_id,
            world_seed=game.world_seed,
            map_id=game.map_id,
            game_type=game.game_type,
            red_composition=game.red_composition,
            blue_composition=game.blue_composition,
            generation_id=config.generation_id,
            source_revision=config.source_revision,
            dirty_tree_hash=config.dirty_tree_hash,
            source_model_digest=config.parent_model_digest,
            search_config_id=config.search_config_id,
            generator_config_id=config.generator_config_id,
        )

    def _build_agents(
        self, runtime: object, game: GameSpec, recorder: JointDatasetRecorder
    ) -> dict[str, Agent]:
        agents: dict[str, Agent] = {}
        sides: tuple[
            tuple[Literal["RED", "BLUE"], tuple[str, ...]],
            tuple[Literal["RED", "BLUE"], tuple[str, ...]],
        ] = (("RED", game.red_composition), ("BLUE", game.blue_composition))
        for side, composition in sides:
            seed = agent_seed(self.spec.config, game.world_seed, side)
            strategy = self.strategy_factory(runtime, game, side, seed)
            agent = self.agent_factory(_RecordingStrategy(strategy, recorder), seed)
            for hero in composition:
                agents[f"hero_{hero.lower().replace(' ', '_')}"] = agent
        return agents

    def _validate_runtime(self, loaded: LoadedChampionRuntime) -> None:
        config = self.spec.config
        if loaded.model_digest != config.parent_model_digest:
            raise ValueError("loaded runtime model digest does not match worker parent digest")
        if loaded.generation != config.parent_generation:
            raise ValueError("loaded runtime generation does not match worker parent generation")
        if loaded.observation_schema_version != config.observation_schema_version:
            raise ValueError("loaded runtime observation schema does not match worker schema")

    def _load_checkpoints(self) -> dict[str, CheckpointRow]:
        if not self.checkpoint_path.exists():
            return {}
        original = self.checkpoint_path.read_bytes()
        payload = original
        if payload and not payload.endswith(b"\n"):
            boundary = payload.rfind(b"\n")
            payload = payload[: boundary + 1] if boundary >= 0 else b""
            _atomic_write(self.checkpoint_path, payload)
        rows: dict[str, CheckpointRow] = {}
        for line_number, raw in enumerate(payload.splitlines(), 1):
            try:
                row = CheckpointRow.model_validate_json(raw)
            except ValueError as exc:
                raise ValueError(f"invalid self-play checkpoint row {line_number}: {exc}") from exc
            if raw != _canonical(row):
                raise ValueError(
                    f"invalid self-play checkpoint row {line_number}: non-canonical JSON"
                )
            self._validate_checkpoint_identity(row)
            if row.game_id in rows:
                raise ValueError(f"invalid self-play checkpoint row {line_number}: duplicate game")
            rows[row.game_id] = row
        return rows

    def _validate_checkpoint_identity(self, row: CheckpointRow) -> None:
        config = self.spec.config
        expected = {
            "worker_id": self.spec.worker_id,
            "worker_config_id": self.spec.worker_config_id,
            "generator_config_id": config.generator_config_id,
            "source_config_id": config.source_config_id,
            "generation_id": config.generation_id,
            "parent_model_digest": config.parent_model_digest,
            "parent_generation": config.parent_generation,
            "observation_schema_version": config.observation_schema_version,
        }
        if any(getattr(row, field) != value for field, value in expected.items()):
            raise ValueError("self-play checkpoint identity does not match worker specification")
        game = next((item for item in self.spec.games if item.game_id(config) == row.game_id), None)
        if game is None or game.world_seed != row.world_seed:
            raise ValueError("self-play checkpoint contains an unassigned game")

    def _reconcile_fragments(self, checkpoints: dict[str, CheckpointRow]) -> None:
        from .dataset import load_joint_dataset

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for game in self.spec.games:
            game_id = game.game_id(self.spec.config)
            path = self.fragment_path(game)
            checkpoint = checkpoints.get(game_id)
            if checkpoint is not None and not path.exists():
                raise ValueError("complete self-play checkpoint has no dataset fragment")
            if not path.exists():
                continue
            dataset = load_joint_dataset(path)
            self._validate_fragment(dataset, game, game_id)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if checkpoint is not None:
                if checkpoint.fragment_digest != digest or checkpoint.row_count != len(
                    dataset.rows
                ):
                    raise ValueError("self-play checkpoint and fragment disagree")
                continue
            row = self._checkpoint_for_fragment(game, game_id, path, None)
            self._append_checkpoint(row)
            checkpoints[game_id] = row

    def _validate_fragment(self, dataset: Any, game: GameSpec, game_id: str) -> None:
        config = self.spec.config
        if dataset.game_ids != (game_id,) or any(
            row.world_seed != game.world_seed
            or row.generation_id != config.generation_id
            or row.source_model_digest != config.parent_model_digest
            or row.search_config_id != config.search_config_id
            or row.generator_config_id != config.generator_config_id
            or row.observation.schema_version != config.observation_schema_version
            for row in dataset.rows
        ):
            raise ValueError("self-play fragment identity does not match assigned game")

    def _checkpoint_for_fragment(
        self,
        game: GameSpec,
        game_id: str,
        path: Path,
        result: RunResult | None,
    ) -> CheckpointRow:
        from .dataset import load_joint_dataset

        dataset = load_joint_dataset(path)
        self._validate_fragment(dataset, game, game_id)
        config = self.spec.config
        winner = dataset.rows[0].terminal_winner
        return CheckpointRow(
            worker_id=self.spec.worker_id,
            worker_config_id=self.spec.worker_config_id,
            generator_config_id=config.generator_config_id,
            source_config_id=config.source_config_id,
            generation_id=config.generation_id,
            parent_model_digest=config.parent_model_digest,
            parent_generation=config.parent_generation,
            observation_schema_version=config.observation_schema_version,
            game_id=game_id,
            world_seed=game.world_seed,
            fragment_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
            row_count=len(dataset.rows),
            winner=winner,
            rounds=result.rounds if result is not None else None,
            turns=result.turns if result is not None else None,
            steps=result.steps if result is not None else None,
        )

    def _append_checkpoint(self, row: CheckpointRow) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint_path.open("ab") as handle:
            handle.write(_canonical(row) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _emit(
        self,
        event: Literal["worker_started", "progress", "game_complete", "timeout", "error"],
        started: float,
        *,
        completed: int,
        game: GameSpec | None = None,
        reason: str | None = None,
        error_type: str | None = None,
    ) -> None:
        elapsed = max(0.0, self.clock() - started)
        self.telemetry(
            TelemetryEvent(
                event=event,
                worker_id=self.spec.worker_id,
                game_id=game.game_id(self.spec.config) if game else None,
                world_seed=game.world_seed if game else None,
                completed_games=completed,
                elapsed_seconds=elapsed,
                games_per_second=completed / elapsed if elapsed else 0.0,
                reason=reason,
                error_type=error_type,
            )
        )


__all__ = [
    "WORKER_COUNT",
    "CheckpointRow",
    "GameSpec",
    "GenerationConfig",
    "LoadedChampionRuntime",
    "SelfPlayWorker",
    "SourceGameTimeout",
    "TelemetryEvent",
    "WorkerSpec",
    "agent_seed",
    "build_worker_specs",
]
