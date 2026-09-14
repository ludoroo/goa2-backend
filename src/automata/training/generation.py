"""Framework-neutral contracts for persistent, complete-game self-play workers."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
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
from automata.search.ismcts.engine import SearchProgressionError
from automata.search.ismcts.strategy import (
    SearchStrategy,
    StrategyResult,
    VisitSamplingStrategy,
)
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
NAMESPACED_SEED_DERIVATION = "sha256(self-play-agent-random-stream-v1,namespace,world-seed,side)"
RANDOM_STREAM_NAMESPACE_MAX_LENGTH = 128
VISIT_TEMPERATURE_SCHEDULES = ("constant", "round-decay-v1")
VisitTemperatureSchedule = Literal["constant", "round-decay-v1"]


def visit_temperature_provider(
    schedule: VisitTemperatureSchedule | str, base_temperature: float
) -> Callable[[GameState], float]:
    """Resolve a named self-play schedule against the public round number."""
    if schedule not in VISIT_TEMPERATURE_SCHEDULES:
        raise ValueError(f"unknown visit_temperature_schedule {schedule!r}")
    if not math.isfinite(base_temperature) or base_temperature < 0:
        raise ValueError("visit_temperature must be finite and non-negative")

    def temperature(state: GameState) -> float:
        if schedule == "constant" or state.round <= 4:
            return base_temperature
        if state.round <= 8:
            return base_temperature / 2
        return 0.0

    return temperature


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
    visit_temperature: float = 0.0
    visit_temperature_schedule: VisitTemperatureSchedule = "constant"
    decision_timeout_seconds: float | None = None
    random_stream_namespace: str | None = None

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
        if not math.isfinite(self.visit_temperature) or self.visit_temperature < 0:
            raise ValueError("visit_temperature must be finite and non-negative")
        if self.visit_temperature_schedule not in VISIT_TEMPERATURE_SCHEDULES:
            raise ValueError(
                f"unknown visit_temperature_schedule {self.visit_temperature_schedule!r}"
            )
        if self.decision_timeout_seconds is not None and (
            not math.isfinite(self.decision_timeout_seconds) or self.decision_timeout_seconds <= 0
        ):
            raise ValueError("decision_timeout_seconds must be finite and positive when enabled")
        if self.random_stream_namespace is not None and (
            not isinstance(self.random_stream_namespace, str)
            or not self.random_stream_namespace.strip()
            or len(self.random_stream_namespace) > RANDOM_STREAM_NAMESPACE_MAX_LENGTH
        ):
            raise ValueError(
                "random_stream_namespace must be a nonempty string of at most "
                f"{RANDOM_STREAM_NAMESPACE_MAX_LENGTH} characters"
            )
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
        identity: dict[str, JsonValue] = {
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
            "visit_temperature": self.visit_temperature,
            "decision_timeout_seconds": self.decision_timeout_seconds,
            "seed_derivation": SEED_DERIVATION,
        }
        if self.visit_temperature_schedule != "constant":
            # Keep the pre-schedule constant identity byte-for-byte compatible;
            # its semantics are already fully identified by visit_temperature.
            identity["visit_temperature_schedule"] = self.visit_temperature_schedule
        if self.random_stream_namespace is not None:
            identity.update(
                {
                    "random_stream_namespace": self.random_stream_namespace,
                    "seed_derivation": NAMESPACED_SEED_DERIVATION,
                }
            )
        return _digest(identity)


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
    """Bounded structured worker telemetry suitable for JSONL logging.

    Decision events intentionally contain only public identity and aggregate
    search counts. Candidate values, request prompts/options, and state dumps
    are never included.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    event: Literal[
        "worker_started",
        "progress",
        "game_complete",
        "timeout",
        "error",
        "decision_started",
        "decision_completed",
        "decision_timeout",
        "decision_failed",
    ]
    worker_id: int = Field(ge=0)
    game_id: str | None = None
    world_seed: int | None = None
    completed_games: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)
    games_per_second: float = Field(ge=0.0)
    reason: str | None = None
    error_type: str | None = None
    round: int | None = Field(default=None, ge=0)
    steps: int | None = Field(default=None, ge=0)
    phase: str | None = None
    side: Literal["RED", "BLUE"] | None = None
    perspective_team: Literal["RED", "BLUE"] | None = None
    decision_index: int | None = Field(default=None, ge=0)
    decision_owner_hero_id: str | None = None
    root_kind: Literal["CARD", "INPUT"] | None = None
    request_type: str | None = None
    request_id: str | None = None
    legal_count: int | None = Field(default=None, ge=0)
    legal_family: str | None = None
    decision_elapsed_seconds: float | None = Field(default=None, ge=0.0)
    completed_visits: int | None = Field(default=None, ge=0)
    visited_legal_count: int | None = Field(default=None, ge=0)
    legal_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    progression_reason: str | None = None
    progression_phase: str | None = None
    progression_round: int | None = Field(default=None, ge=0)
    progression_actor: str | None = None
    progression_pending_request: str | None = None
    progression_stack_depth: int | None = Field(default=None, ge=0)
    progression_top_step: str | None = None
    progression_transition_counts: dict[str, int] | None = None


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
    if config.random_stream_namespace is None:
        material: dict[str, JsonValue] = {
            "derivation": SEED_DERIVATION,
            "generator_config_id": config.generator_config_id,
            "world_seed": world_seed,
            "side": side,
        }
    else:
        material = {
            "derivation": NAMESPACED_SEED_DERIVATION,
            "namespace": config.random_stream_namespace,
            "world_seed": world_seed,
            "side": side,
        }
    return int(_digest(material)[:16], 16)


DecisionEventSink = Callable[
    [
        Literal["decision_started", "decision_completed", "decision_timeout", "decision_failed"],
        GameState,
        TeamColor,
        RootTarget,
        tuple[Key, ...],
        int,
        float,
        StrategyResult[Key] | None,
        SearchProgressionError | None,
    ],
    None,
]


class _RecordingStrategy:
    """Apply the self-play watchdog and record the exact improved policy root."""

    def __init__(
        self,
        delegate: SearchStrategy,
        recorder: JointDatasetRecorder,
        *,
        decision_timeout_seconds: float | None = None,
        decision_index: Callable[[], int] | None = None,
        decision_events: DecisionEventSink | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self._decision_timeout_seconds = decision_timeout_seconds
        self._decision_index = decision_index or (lambda: 0)
        self._decision_events = decision_events
        self._clock = clock
        self.strategy_id = delegate.strategy_id

    def select(
        self,
        state: GameState,
        perspective_team: TeamColor,
        root_target: RootTarget,
        legal_candidates: Sequence[Key],
    ) -> StrategyResult[Key]:
        legal = tuple(legal_candidates)
        decision_index = self._decision_index()
        started = self._clock()
        self._emit_decision(
            "decision_started",
            state,
            perspective_team,
            root_target,
            legal,
            decision_index,
            0.0,
            None,
        )
        result: StrategyResult[Key] | None = None
        try:
            with _decision_timeout(self._decision_timeout_seconds):
                result = self._delegate.select(state, perspective_team, root_target, legal)
        except SourceDecisionTimeout:
            self._emit_decision(
                "decision_timeout",
                state,
                perspective_team,
                root_target,
                legal,
                decision_index,
                max(0.0, self._clock() - started),
                result,
            )
            raise
        except SearchProgressionError as exc:
            self._emit_decision(
                "decision_failed",
                state,
                perspective_team,
                root_target,
                legal,
                decision_index,
                max(0.0, self._clock() - started),
                result,
                progression_error=exc,
            )
            raise
        assert result is not None
        self._record_result(state, perspective_team, root_target, legal, result)
        self._emit_decision(
            "decision_completed",
            state,
            perspective_team,
            root_target,
            legal,
            decision_index,
            max(0.0, self._clock() - started),
            result,
        )
        return result

    def _record_result(
        self,
        state: GameState,
        perspective_team: TeamColor,
        root_target: RootTarget,
        legal: tuple[Key, ...],
        result: StrategyResult[Key],
    ) -> None:
        if result.candidates != legal:
            raise ValueError("search strategy changed or reordered the legal candidates")
        owner = state.get_hero(HeroID(root_target.decision_owner_hero_id))
        if owner is None:
            raise ValueError("search root decision owner is absent")
        request = None
        if root_target.kind == "INPUT":
            request = root_target.request
            stacked_request = next(
                (item for item in reversed(state.input_stack) if item.id == root_target.request_id),
                None,
            )
            if request is None:
                request = stacked_request
            if (
                request is None
                or request.id != root_target.request_id
                or request.player_id != root_target.player_id
                or (stacked_request is not None and stacked_request != request)
            ):
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

    def _emit_decision(
        self,
        event: Literal[
            "decision_started", "decision_completed", "decision_timeout", "decision_failed"
        ],
        state: GameState,
        perspective_team: TeamColor,
        root_target: RootTarget,
        legal: tuple[Key, ...],
        decision_index: int,
        elapsed: float,
        result: StrategyResult[Key] | None,
        *,
        progression_error: SearchProgressionError | None = None,
    ) -> None:
        if self._decision_events is not None:
            self._decision_events(
                event,
                state,
                perspective_team,
                root_target,
                legal,
                decision_index,
                elapsed,
                result,
                progression_error,
            )


def _aligned_action_stats(
    result: StrategyResult[Key], candidates: Sequence[Any]
) -> tuple[SearchActionTarget, ...]:
    statistics = result.search_result
    if statistics is not None:
        if statistics.best_key not in result.candidates:
            raise ValueError("search statistics best action is outside the legal root")
        if any(key not in result.candidates for key in statistics.root.children):
            raise ValueError("search statistics contain actions outside the legal root")
        visits = tuple(
            statistics.root.children[key].visits if key in statistics.root.children else 0
            for key in result.candidates
        )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in visits
        ):
            raise ValueError("self-play root visits must be non-negative integers")
        total_visits = sum(visits)
        improved_probabilities: tuple[float, ...]
        if not total_visits:
            if len(visits) != 1:
                raise ValueError("self-play search statistics contain no root visits")
            # Search intentionally leaves validated forced roots unvisited. Keep
            # the zero sample/value sentinel while recording the only policy mass.
            improved_probabilities = (1.0,)
        else:
            improved_probabilities = tuple(count / total_visits for count in visits)
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
                improved_probability=improved_probabilities[index],
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
    """The whole self-play game exceeded its wall-clock deadline."""


class SourceDecisionTimeout(TimeoutError):
    """One self-play source-policy decision exceeded its hard deadline."""


@contextmanager
def _alarm_timeout(seconds: float, error_type: type[TimeoutError]) -> Iterator[None]:
    """Install a nestable real-time deadline without extending an outer timer.

    POSIX interval timers are process-global and only safely managed from the
    main thread. A nested deadline therefore arms whichever of its own timeout
    and the existing timer expires first. On exit, elapsed wall time is
    subtracted before restoring the outer timer rather than accidentally
    granting that timer a fresh budget.
    """
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    own_deadline = previous_delay <= 0 or seconds < previous_delay
    delay = seconds if own_deadline else previous_delay

    def expire(signum: int, frame: Any) -> None:
        if own_deadline:
            raise error_type
        if callable(previous_handler):
            previous_handler(signum, frame)
            return
        # An ignored/default outer alarm cannot provide a typed exception to
        # the worker. Fail closed under this scope's timeout instead of letting
        # a stalled source decision continue without any armed watchdog.
        raise error_type

    signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, delay)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = max(0.0, time.monotonic() - started)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0:
            remaining = previous_delay - elapsed
            if remaining <= 0 and previous_interval > 0:
                remaining = previous_interval - ((-remaining) % previous_interval)
            if remaining > 0:
                signal.setitimer(signal.ITIMER_REAL, remaining, previous_interval)


@contextmanager
def _game_timeout(seconds: float) -> Iterator[None]:
    with _alarm_timeout(seconds, SourceGameTimeout):
        yield


@contextmanager
def _decision_timeout(seconds: float | None) -> Iterator[None]:
    if seconds is None:
        yield
        return
    with _alarm_timeout(seconds, SourceDecisionTimeout):
        yield


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
            agents = self._build_agents(loaded.runtime, game, recorder, started, completed)
            result: RunResult | None = None

            def report_progress(
                round_number: int,
                steps: int,
                *,
                current_game: GameSpec = game,
                completed_before_game: int = completed,
            ) -> None:
                self._emit(
                    "progress",
                    started,
                    completed=completed_before_game,
                    game=current_game,
                    round_number=round_number,
                    steps=steps,
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
            except SourceDecisionTimeout:
                try:
                    recorder.close()
                finally:
                    fragment.unlink(missing_ok=True)
                continue
            except SourceGameTimeout:
                try:
                    recorder.close()
                finally:
                    fragment.unlink(missing_ok=True)
                self._emit(
                    "timeout",
                    started,
                    completed=completed,
                    game=game,
                    reason="wall_clock_timeout",
                )
                continue
            except SearchProgressionError as exc:
                try:
                    recorder.close()
                finally:
                    fragment.unlink(missing_ok=True)
                self._emit(
                    "timeout",
                    started,
                    completed=completed,
                    game=game,
                    reason="search_progression",
                    error_type=type(exc).__name__,
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
                fragment.unlink(missing_ok=True)
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
        self,
        runtime: object,
        game: GameSpec,
        recorder: JointDatasetRecorder,
        worker_started: float,
        completed_before_game: int,
    ) -> dict[str, Agent]:
        agents: dict[str, Agent] = {}
        decision_indexes = itertools.count()
        sides: tuple[
            tuple[Literal["RED", "BLUE"], tuple[str, ...]],
            tuple[Literal["RED", "BLUE"], tuple[str, ...]],
        ] = (("RED", game.red_composition), ("BLUE", game.blue_composition))
        for side, composition in sides:
            seed = agent_seed(self.spec.config, game.world_seed, side)
            strategy = self.strategy_factory(runtime, game, side, seed)
            if self.spec.config.visit_temperature_schedule == "constant":
                sampled_strategy = VisitSamplingStrategy(
                    strategy,
                    temperature=self.spec.config.visit_temperature,
                    seed=seed,
                )
            else:
                sampled_strategy = VisitSamplingStrategy(
                    strategy,
                    temperature_provider=visit_temperature_provider(
                        self.spec.config.visit_temperature_schedule,
                        self.spec.config.visit_temperature,
                    ),
                    seed=seed,
                )

            def report_decision(
                event: Literal[
                    "decision_started",
                    "decision_completed",
                    "decision_timeout",
                    "decision_failed",
                ],
                state: GameState,
                perspective_team: TeamColor,
                root_target: RootTarget,
                legal: tuple[Key, ...],
                decision_index: int,
                elapsed: float,
                result: StrategyResult[Key] | None,
                progression_error: SearchProgressionError | None,
                *,
                game_side: Literal["RED", "BLUE"] = side,
            ) -> None:
                self._emit_decision(
                    event,
                    worker_started,
                    completed=completed_before_game,
                    game=game,
                    side=game_side,
                    state=state,
                    perspective_team=perspective_team,
                    root_target=root_target,
                    legal=legal,
                    decision_index=decision_index,
                    decision_elapsed=elapsed,
                    result=result,
                    progression_error=progression_error,
                )

            agent = self.agent_factory(
                _RecordingStrategy(
                    sampled_strategy,
                    recorder,
                    decision_timeout_seconds=self.spec.config.decision_timeout_seconds,
                    decision_index=lambda: next(decision_indexes),
                    decision_events=report_decision,
                    clock=self.clock,
                ),
                seed,
            )
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

    def _emit_decision(
        self,
        event: Literal[
            "decision_started", "decision_completed", "decision_timeout", "decision_failed"
        ],
        started: float,
        *,
        completed: int,
        game: GameSpec,
        side: Literal["RED", "BLUE"],
        state: GameState,
        perspective_team: TeamColor,
        root_target: RootTarget,
        legal: tuple[Key, ...],
        decision_index: int,
        decision_elapsed: float,
        result: StrategyResult[Key] | None,
        progression_error: SearchProgressionError | None,
    ) -> None:
        request = root_target.request or next(
            (item for item in reversed(state.input_stack) if item.id == root_target.request_id),
            None,
        )
        request_type = request.request_type.value if request is not None else None
        completed_visits: int | None = None
        visited_legal_count: int | None = None
        legal_coverage: float | None = None
        if result is not None and result.search_result is not None:
            children = result.search_result.root.children
            visit_counts = tuple(children[key].visits if key in children else 0 for key in legal)
            completed_visits = sum(visit_counts)
            visited_legal_count = sum(count > 0 for count in visit_counts)
            legal_coverage = visited_legal_count / len(legal) if legal else 0.0
        elapsed = max(0.0, self.clock() - started)
        self.telemetry(
            TelemetryEvent(
                event=event,
                worker_id=self.spec.worker_id,
                game_id=game.game_id(self.spec.config),
                world_seed=game.world_seed,
                completed_games=completed,
                elapsed_seconds=elapsed,
                games_per_second=completed / elapsed if elapsed else 0.0,
                reason=(
                    "decision_timeout"
                    if event == "decision_timeout"
                    else "search_progression" if event == "decision_failed" else None
                ),
                error_type=(
                    type(progression_error).__name__ if progression_error is not None else None
                ),
                round=state.round,
                phase=state.phase.value,
                side=side,
                perspective_team=perspective_team.value,
                decision_index=decision_index,
                decision_owner_hero_id=root_target.decision_owner_hero_id,
                root_kind=root_target.kind,
                request_type=request_type,
                request_id=root_target.request_id,
                legal_count=len(legal),
                legal_family=request_type or root_target.kind,
                decision_elapsed_seconds=decision_elapsed,
                completed_visits=completed_visits,
                visited_legal_count=visited_legal_count,
                legal_coverage=legal_coverage,
                progression_reason=(
                    progression_error.reason if progression_error is not None else None
                ),
                progression_phase=(
                    progression_error.phase if progression_error is not None else None
                ),
                progression_round=(
                    progression_error.round if progression_error is not None else None
                ),
                progression_actor=(
                    progression_error.actor if progression_error is not None else None
                ),
                progression_pending_request=(
                    progression_error.pending_request if progression_error is not None else None
                ),
                progression_stack_depth=(
                    progression_error.stack_depth if progression_error is not None else None
                ),
                progression_top_step=(
                    progression_error.top_step if progression_error is not None else None
                ),
                progression_transition_counts=(
                    progression_error.transition_counts if progression_error is not None else None
                ),
            )
        )

    def _emit(
        self,
        event: Literal["worker_started", "progress", "game_complete", "timeout", "error"],
        started: float,
        *,
        completed: int,
        game: GameSpec | None = None,
        reason: str | None = None,
        error_type: str | None = None,
        round_number: int | None = None,
        steps: int | None = None,
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
                round=round_number,
                steps=steps,
            )
        )


__all__ = [
    "VISIT_TEMPERATURE_SCHEDULES",
    "WORKER_COUNT",
    "CheckpointRow",
    "GameSpec",
    "GenerationConfig",
    "LoadedChampionRuntime",
    "SelfPlayWorker",
    "SourceDecisionTimeout",
    "SourceGameTimeout",
    "TelemetryEvent",
    "VisitTemperatureSchedule",
    "WorkerSpec",
    "agent_seed",
    "build_worker_specs",
    "visit_temperature_provider",
]
