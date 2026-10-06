"""Concrete one-game native Gen1 generation through the real harness and search."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.decision import DecisionDescriptor
from automata.harness.game_runner import RunResult, run_game
from automata.models.shared_encoder.gen1_runtime import Gen1SharedEncoderRuntime
from automata.observation import encode_decision
from automata.search.config import SearchConfig
from automata.search.continuation import PriorSamplingContinuationPolicy
from automata.search.contracts import ContinuationPolicy, CutoffUnit, LeafMode, SearchPolicy
from automata.search.heuristic import HeuristicLeafEvaluator, HeuristicPrior
from automata.search.ismcts.strategy import (
    ISMCTSStrategy,
    SearchStrategy,
    StrategyResult,
    VisitSamplingStrategy,
)
from automata.search.learned import LearnedSearchPolicy, LearnedStableValueEvaluator
from automata.search.node import Key
from automata.search.root import RootTarget
from automata.training.native_dataset import NativeGameIdentity, native_game_id
from automata.training.native_gen1 import (
    current_gen1_artifact_scope,
    current_gen1_runtime_requirements,
    load_current_gen1_parent_artifact,
)
from automata.training.native_receipts import (
    NATIVE_COMPLETION_CONTRACT,
    NativeCompletionTarget,
    NativeGameCompletionReceipt,
)
from automata.training.native_recorder import NativeDatasetRecorder
from automata.training.native_splits import NativeSplitConfig, native_seed_purpose
from automata.training.search_targets import search_policy_target_from_result
from goa2.data.heroes import HeroRegistry
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

NativeGenerationSeedPurpose = Literal["bootstrap", "training", "validation"]
NativeTeacherKind = Literal["HEURISTIC_BOOTSTRAP", "GEN1_PARENT"]
NativeSeedStream = Literal[
    "RED_SEARCH",
    "RED_ACTION",
    "RED_ENVIRONMENT",
    "BLUE_SEARCH",
    "BLUE_ACTION",
    "BLUE_ENVIRONMENT",
]

_SEED_RECIPE = "sha256(native-gen1-generation-stream-v1,namespace,world-seed,stream)"
_ROOT_TARGET_ADAPTER = "search-policy-target-from-result-v1"
_ACTION_RECIPE = "visit-sampling-v1"
_MAPS_ROOT = Path(__file__).resolve().parents[2] / "goa2" / "data" / "maps"
_STREAMS: frozenset[str] = frozenset(
    {
        "RED_SEARCH",
        "RED_ACTION",
        "RED_ENVIRONMENT",
        "BLUE_SEARCH",
        "BLUE_ACTION",
        "BLUE_ENVIRONMENT",
    }
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_nonempty(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _strict_positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_nonnegative_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return parsed


def _validate_digest(value: object, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError("GEN1_PARENT requires a source model digest")
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("source model digest must be a lowercase SHA-256 digest")
    if not required:
        raise ValueError("HEURISTIC_BOOTSTRAP cannot carry a source model digest")
    return value


def _validate_search_config(config: object) -> SearchConfig:
    if not isinstance(config, SearchConfig):
        raise TypeError("search_config must be a SearchConfig")
    raw = asdict(config)
    if set(raw) != {field.name for field in fields(SearchConfig)}:  # pragma: no cover
        raise ValueError("search_config fields are incomplete")

    positive_ints = (
        "iterations",
        "max_advance_steps",
        "max_advance_transitions",
        "max_forced_decisions",
    )
    for name in positive_ints:
        _strict_positive_int(raw[name], label=f"search_config.{name}")
    if type(raw["cutoff_limit"]) is not int or raw["cutoff_limit"] < 0:
        raise ValueError("search_config.cutoff_limit must be a non-negative integer")
    if type(raw["seed"]) is not int or raw["seed"] != 0:
        raise ValueError("search_config.seed must be zero; generation derives per-side seeds")
    if type(raw["use_prior"]) is not bool:
        raise ValueError("search_config.use_prior must be a boolean")
    if not isinstance(config.cutoff_unit, CutoffUnit):
        raise ValueError("search_config.cutoff_unit must be a CutoffUnit")
    if not isinstance(config.leaf_mode, LeafMode):
        raise ValueError("search_config.leaf_mode must be a LeafMode")

    for name in ("uct_c", "puct_c"):
        _strict_nonnegative_finite(raw[name], label=f"search_config.{name}")
    for name in ("widening_c",):
        value = _strict_nonnegative_finite(raw[name], label=f"search_config.{name}")
        if value == 0:
            raise ValueError(f"search_config.{name} must be positive")
    for name in ("widening_alpha",):
        value = _strict_nonnegative_finite(raw[name], label=f"search_config.{name}")
        if value > 1:
            raise ValueError(f"search_config.{name} must be at most one")
    for name, positive, bounded in (
        ("root_puct_c", False, False),
        ("root_widening_c", True, False),
        ("root_widening_alpha", False, True),
    ):
        value = raw[name]
        if value is None:
            continue
        parsed = _strict_nonnegative_finite(value, label=f"search_config.{name}")
        if positive and parsed == 0:
            raise ValueError(f"search_config.{name} must be positive")
        if bounded and parsed > 1:
            raise ValueError(f"search_config.{name} must be at most one")

    if config.leaf_mode is not LeafMode.STABLE_TRANSITION:
        raise ValueError("native generation requires STABLE_TRANSITION leaf mode")
    if config.decision_timeout_seconds is not None:
        raise ValueError("native generation search decision timeout must be None")
    if config.request_schedule_version is not None:
        raise ValueError("native generation search request schedule must be None")
    # Reconstruction reruns SearchConfig's own cross-field checks.
    return SearchConfig(**raw)


def _search_identity(config: SearchConfig) -> dict[str, Any]:
    identity = asdict(config)
    identity["cutoff_unit"] = config.cutoff_unit.value
    identity["leaf_mode"] = config.leaf_mode.value
    identity["seed"] = {
        "recipe": _SEED_RECIPE,
        "scope": "per-side SEARCH stream",
    }
    return identity


@dataclass(frozen=True, slots=True)
class NativeGenerationConfig:
    generation_id: str
    source_revision: str
    dirty_tree_hash: str
    teacher_kind: NativeTeacherKind
    source_model_digest: str | None
    search_config: SearchConfig
    split_config: NativeSplitConfig
    random_stream_namespace: str
    visit_temperature: float
    max_steps: int
    max_rounds: int | None

    def __post_init__(self) -> None:
        _validate_generation_config(self)

    @property
    def search_config_id(self) -> str:
        validated = _validate_generation_config(self)
        return _digest({"search_config": _search_identity(validated.search_config)})

    @property
    def generator_config_id(self) -> str:
        validated = _validate_generation_config(self)
        return _digest(
            {
                "teacher": {
                    "kind": validated.teacher_kind,
                    "source_model_digest": validated.source_model_digest,
                },
                "leaf_contract": LeafMode.STABLE_TRANSITION.value,
                "completion_contract": NATIVE_COMPLETION_CONTRACT,
                "root_target_adapter": _ROOT_TARGET_ADAPTER,
                "action_recipe": {
                    "recipe": _ACTION_RECIPE,
                    "visit_temperature": validated.visit_temperature,
                    "zero_temperature": "delegate robust child",
                },
                "seed_derivation": _SEED_RECIPE,
                "random_stream_namespace": validated.random_stream_namespace,
                "max_steps": validated.max_steps,
                "max_rounds": validated.max_rounds,
                "split_config_digest": validated.split_config.digest,
                "search_config_id": _digest(
                    {"search_config": _search_identity(validated.search_config)}
                ),
            }
        )


def _validate_generation_config(config: object) -> NativeGenerationConfig:
    if not isinstance(config, NativeGenerationConfig):
        raise TypeError("config must be a NativeGenerationConfig")
    _strict_nonempty(config.generation_id, label="generation_id")
    _strict_nonempty(config.source_revision, label="source_revision")
    _strict_nonempty(config.dirty_tree_hash, label="dirty_tree_hash")
    _strict_nonempty(config.random_stream_namespace, label="random_stream_namespace")
    if config.teacher_kind not in {"HEURISTIC_BOOTSTRAP", "GEN1_PARENT"}:
        raise ValueError("unknown native teacher kind")
    _validate_digest(
        config.source_model_digest,
        required=config.teacher_kind == "GEN1_PARENT",
    )
    search = _validate_search_config(config.search_config)
    if config.teacher_kind == "GEN1_PARENT" and not search.use_prior:
        raise ValueError("GEN1_PARENT generation requires search_config.use_prior=True")
    if not isinstance(config.split_config, NativeSplitConfig):
        raise TypeError("split_config must be a NativeSplitConfig")
    split = NativeSplitConfig.model_validate(
        config.split_config.model_dump(mode="python"), strict=True
    )
    temperature = _strict_nonnegative_finite(config.visit_temperature, label="visit_temperature")
    max_steps = _strict_positive_int(config.max_steps, label="max_steps")
    if config.max_rounds is not None:
        _strict_positive_int(config.max_rounds, label="max_rounds")
    # Return an independently reconstructed value at public boundaries. Avoid
    # calling the constructor, which would recurse through this validator.
    return _validated_config_copy(
        config,
        search=search,
        split=split,
        temperature=temperature,
        max_steps=max_steps,
    )


def _validated_config_copy(
    config: NativeGenerationConfig,
    *,
    search: SearchConfig,
    split: NativeSplitConfig,
    temperature: float,
    max_steps: int,
) -> NativeGenerationConfig:
    result = object.__new__(NativeGenerationConfig)
    object.__setattr__(result, "generation_id", config.generation_id)
    object.__setattr__(result, "source_revision", config.source_revision)
    object.__setattr__(result, "dirty_tree_hash", config.dirty_tree_hash)
    object.__setattr__(result, "teacher_kind", config.teacher_kind)
    object.__setattr__(result, "source_model_digest", config.source_model_digest)
    object.__setattr__(result, "search_config", search)
    object.__setattr__(result, "split_config", split)
    object.__setattr__(result, "random_stream_namespace", config.random_stream_namespace)
    object.__setattr__(result, "visit_temperature", temperature)
    object.__setattr__(result, "max_steps", max_steps)
    object.__setattr__(result, "max_rounds", config.max_rounds)
    return result


class NativeGenerationGame(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    world_seed: StrictInt = Field(ge=0)
    seed_purpose: NativeGenerationSeedPurpose
    map_id: str = Field(min_length=1)
    game_type: str = Field(min_length=1)
    red_composition: tuple[str, ...] = Field(min_length=1)
    blue_composition: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_scope(self) -> NativeGenerationGame:
        for label, value in (("map_id", self.map_id), ("game_type", self.game_type)):
            if not value.strip():
                raise ValueError(f"{label} must be nonempty")
        roster = (*self.red_composition, *self.blue_composition)
        if any(not hero.strip() for hero in roster):
            raise ValueError("native generation rosters must contain nonempty hero names")
        if len(roster) != len(set(roster)):
            raise ValueError("native generation rosters must contain unique heroes")
        return self


class NativeGenerationOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source_root: Path
    logical_name: str = Field(min_length=1)
    completion_receipt_path: Path


@dataclass(frozen=True, slots=True)
class NativeGenerationResult:
    game: NativeGameIdentity
    outcome: RunResult
    completion_receipt: NativeGameCompletionReceipt | None

    @property
    def completed(self) -> bool:
        return self.outcome.reason == "game_over" and self.completion_receipt is not None


def derive_native_generation_seed(
    config: NativeGenerationConfig,
    *,
    world_seed: int,
    stream: NativeSeedStream,
) -> int:
    """Derive one independent deterministic full-width SHA256 random stream."""
    validated = _validate_generation_config(config)
    if type(world_seed) is not int or world_seed < 0:
        raise ValueError("world_seed must be a non-negative integer")
    if stream not in _STREAMS:
        raise ValueError(f"unknown native generation seed stream {stream!r}")
    payload = {
        "recipe": _SEED_RECIPE,
        "namespace": validated.random_stream_namespace,
        "world_seed": world_seed,
        "stream": stream,
    }
    return int.from_bytes(hashlib.sha256(_canonical_bytes(payload)).digest(), "big")


def _strict_game(game: object) -> NativeGenerationGame:
    if not isinstance(game, NativeGenerationGame):
        raise TypeError("game must be a NativeGenerationGame")
    return NativeGenerationGame.model_validate(game.model_dump(mode="python"), strict=True)


def _strict_output(output: object) -> NativeGenerationOutput:
    if not isinstance(output, NativeGenerationOutput):
        raise TypeError("output must be a NativeGenerationOutput")
    validated = NativeGenerationOutput.model_validate(output.model_dump(mode="python"), strict=True)
    logical = PurePosixPath(validated.logical_name)
    if (
        logical.is_absolute()
        or validated.logical_name != logical.as_posix()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise ValueError("native generation logical name must be normalized relative POSIX")
    if not (
        validated.logical_name.endswith(".jsonl") or validated.logical_name.endswith(".jsonl.zst")
    ):
        raise ValueError("native generation logical name must identify a JSONL file")
    root = Path(os.path.abspath(validated.source_root))
    if root.is_symlink() or not root.is_dir():
        raise ValueError("native generation source root must be an existing regular directory")
    source = root.joinpath(*logical.parts)
    receipt = Path(os.path.abspath(validated.completion_receipt_path))
    if source == receipt or source in receipt.parents or receipt in source.parents:
        raise ValueError("native generation source and completion receipt paths overlap")
    return NativeGenerationOutput(
        source_root=root,
        logical_name=validated.logical_name,
        completion_receipt_path=receipt,
    )


def _validate_game_scope(game: NativeGenerationGame, config: NativeGenerationConfig) -> Path:
    declared = native_seed_purpose(config.split_config, game.world_seed)
    if declared not in {"bootstrap", "training", "validation"}:
        raise ValueError(f"native seed purpose {declared!r} is not eligible for native generation")
    if declared != game.seed_purpose:
        raise ValueError(
            f"native generation seed purpose mismatch: declared {declared!r}, "
            f"requested {game.seed_purpose!r}"
        )

    scope = current_gen1_artifact_scope()
    if game.map_id not in scope.supported_maps:
        raise ValueError(f"unknown current native map {game.map_id!r}")
    if game.game_type not in scope.supported_game_types:
        raise ValueError(f"unknown current native game type {game.game_type!r}")
    unknown = set((*game.red_composition, *game.blue_composition)) - set(scope.supported_heroes)
    if unknown:
        raise ValueError(f"unknown current native hero scope: {sorted(unknown)!r}")
    map_path = _MAPS_ROOT / f"{game.map_id}.json"
    if not map_path.is_file():  # pragma: no cover - scope enumeration guarantees this
        raise ValueError(f"native map source is absent: {game.map_id!r}")
    return map_path


def _game_identity(
    game: NativeGenerationGame, config: NativeGenerationConfig
) -> NativeGameIdentity:
    fields_: dict[str, Any] = {
        "world_seed": game.world_seed,
        "map_id": game.map_id,
        "game_type": game.game_type,
        "red_composition": game.red_composition,
        "blue_composition": game.blue_composition,
        "generation_id": config.generation_id,
        "source_revision": config.source_revision,
        "dirty_tree_hash": config.dirty_tree_hash,
        "source_model_digest": config.source_model_digest,
        "search_config_id": config.search_config_id,
        "generator_config_id": config.generator_config_id,
    }
    return NativeGameIdentity(game_id=native_game_id(**fields_), **fields_)


class _PlayedRootRecorder:
    """Record the sampled result that the agent actually applies to the game."""

    strategy_id = "native-played-root-recorder-v1"

    def __init__(self, delegate: SearchStrategy, recorder: NativeDatasetRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder

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
        if owner is None or owner.team != perspective_team:
            raise ValueError("search root decision owner does not match its perspective")
        if root_target.decision_owner_hero_id not in root_target.owned_hero_ids:
            raise ValueError("search root decision owner is outside root ownership")

        request = None
        if root_target.kind == "CARD":
            if root_target.hero_id != owner.id:
                raise ValueError("card search root does not match its decision owner")
        else:
            stacked = next(
                (item for item in reversed(state.input_stack) if item.id == root_target.request_id),
                None,
            )
            request = root_target.request or stacked
            if (
                request is None
                or request.id != root_target.request_id
                or request.player_id != root_target.player_id
                or (stacked is not None and stacked != request)
            ):
                raise ValueError("input search root does not match the live request")

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
        target = search_policy_target_from_result(result, observation.candidates)
        self._recorder.record_policy(
            observation=observation,
            target=target,
            perspective_team=perspective_team.value,
        )
        return result


def _learned_runtime(
    game: NativeGenerationGame,
    config: NativeGenerationConfig,
    parent_artifact_path: str | Path | None,
) -> Gen1SharedEncoderRuntime | None:
    if config.teacher_kind == "HEURISTIC_BOOTSTRAP":
        if parent_artifact_path is not None:
            raise ValueError("HEURISTIC_BOOTSTRAP cannot use a parent artifact path")
        return None
    if parent_artifact_path is None:
        raise ValueError("GEN1_PARENT requires a parent artifact path")
    assert config.source_model_digest is not None
    loaded = load_current_gen1_parent_artifact(
        parent_artifact_path,
        expected_model_digest=config.source_model_digest,
    )
    requirements = current_gen1_runtime_requirements(
        heroes=(*game.red_composition, *game.blue_composition),
        map_id=game.map_id,
        game_type=game.game_type,
    )
    return Gen1SharedEncoderRuntime(
        model=loaded.model,
        decision_schema=loaded.decision_schema,
        stable_value_schema=loaded.stable_value_schema,
        requirements=requirements,
        supported_heroes=loaded.manifest.supported_heroes,
        supported_maps=loaded.manifest.supported_maps,
        supported_game_types=loaded.manifest.supported_game_types,
        device="cpu",
    )


def _hero_ids(composition: tuple[str, ...]) -> tuple[str, ...]:
    ids: list[str] = []
    for name in composition:
        hero = HeroRegistry.get(name)
        if hero is None:  # pragma: no cover - scope validation guards this
            raise ValueError(f"unknown native generation hero {name!r}")
        ids.append(str(hero.id))
    return tuple(ids)


def _build_agents(
    game: NativeGenerationGame,
    config: NativeGenerationConfig,
    recorder: NativeDatasetRecorder,
    runtime: Gen1SharedEncoderRuntime | None,
) -> dict[str, ISMCTSAgent]:
    agents: dict[str, ISMCTSAgent] = {}
    for side, composition in (
        ("RED", game.red_composition),
        ("BLUE", game.blue_composition),
    ):
        search_seed = derive_native_generation_seed(
            config,
            world_seed=game.world_seed,
            stream=cast(NativeSeedStream, f"{side}_SEARCH"),
        )
        action_seed = derive_native_generation_seed(
            config,
            world_seed=game.world_seed,
            stream=cast(NativeSeedStream, f"{side}_ACTION"),
        )
        environment_seed = derive_native_generation_seed(
            config,
            world_seed=game.world_seed,
            stream=cast(NativeSeedStream, f"{side}_ENVIRONMENT"),
        )
        search_config = replace(config.search_config, seed=search_seed)
        environment = HeuristicAgent(environment_seed)
        prior: SearchPolicy
        continuation: ContinuationPolicy | HeuristicAgent
        leaf: Any
        if runtime is None:
            prior = HeuristicPrior(environment)
            continuation = environment
            leaf = HeuristicLeafEvaluator()
        else:
            prior = LearnedSearchPolicy(runtime)
            continuation = PriorSamplingContinuationPolicy(prior)
            leaf = LearnedStableValueEvaluator(runtime)
        base = ISMCTSStrategy(
            environment_policy=environment,
            config=search_config,
            prior=prior,
            leaf_evaluator=leaf,
            continuation_policy=continuation,
        )
        sampled = VisitSamplingStrategy(
            base,
            temperature=config.visit_temperature,
            seed=action_seed,
        )
        played = _PlayedRootRecorder(sampled, recorder)
        agent = ISMCTSAgent(
            search_config,
            environment_policy=environment,
            continuation_policy=continuation,
            leaf_evaluator=leaf,
            prior=prior,
            strategy=played,
        )
        for hero_id in _hero_ids(composition):
            agents[hero_id] = agent
    return agents


def generate_native_game(
    game: NativeGenerationGame,
    config: NativeGenerationConfig,
    output: NativeGenerationOutput,
    *,
    parent_artifact_path: str | Path | None = None,
) -> NativeGenerationResult:
    """Play exactly one configured game and publish only decisive nonempty evidence."""
    validated_game = _strict_game(game)
    validated_config = _validate_generation_config(config)
    validated_output = _strict_output(output)
    map_path = _validate_game_scope(validated_game, validated_config)
    identity = _game_identity(validated_game, validated_config)

    # Parent verification happens before the recorder can create a spool.
    runtime = _learned_runtime(
        validated_game,
        validated_config,
        parent_artifact_path,
    )
    source = validated_output.source_root.joinpath(
        *PurePosixPath(validated_output.logical_name).parts
    )
    completion_target = NativeCompletionTarget(
        source_root=validated_output.source_root,
        receipt_path=validated_output.completion_receipt_path,
    )
    with NativeDatasetRecorder(
        source,
        game=identity,
        completion_target=completion_target,
    ) as recorder:
        agents = _build_agents(
            validated_game,
            validated_config,
            recorder,
            runtime,
        )
        outcome = run_game(
            list(validated_game.red_composition),
            list(validated_game.blue_composition),
            agents,
            map_path=str(map_path),
            game_type=validated_game.game_type,
            seed=validated_game.world_seed,
            max_steps=validated_config.max_steps,
            max_rounds=validated_config.max_rounds,
            boundary_observer=recorder,
        )
        receipt = recorder.completion_receipt

    if outcome.reason == "game_over":
        # An ordinary game can complete before any recordable root/boundary.
        # The recorder deliberately treats that as empty and unreceipted.
        if receipt is None:
            if source.exists() or validated_output.completion_receipt_path.exists():
                raise RuntimeError(
                    "empty native game encountered unexpected pre-existing or competing "
                    "source/receipt state"
                )
        elif (
            receipt.game != identity
            or not source.is_file()
            or not validated_output.completion_receipt_path.is_file()
        ):
            raise RuntimeError("completed native game publication is incomplete")
    else:
        if (
            receipt is not None
            or source.exists()
            or validated_output.completion_receipt_path.exists()
        ):
            raise RuntimeError(
                "censored native game encountered unexpected pre-existing or competing "
                "source/receipt state"
            )

    return NativeGenerationResult(
        game=identity,
        outcome=outcome,
        completion_receipt=receipt,
    )


__all__ = [
    "NativeGenerationConfig",
    "NativeGenerationGame",
    "NativeGenerationOutput",
    "NativeGenerationResult",
    "NativeGenerationSeedPurpose",
    "NativeSeedStream",
    "NativeTeacherKind",
    "derive_native_generation_seed",
    "generate_native_game",
]
