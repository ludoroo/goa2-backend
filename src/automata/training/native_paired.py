"""Concrete, finite native Gen1 paired gameplay evaluation.

The parent process owns the fixed schedule, plays every declared game once, and
publishes only diagnostic gameplay evidence.  This module deliberately has no
training, replay, recording, promotion, or adaptive scheduling hooks.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from automata.agents.heuristic_agent import HeuristicAgent
from automata.agents.ismcts_agent import ISMCTSAgent
from automata.decision import DecisionDescriptor
from automata.harness.game_runner import RunResult, run_game
from automata.models.contracts import ArtifactError
from automata.models.shared_encoder.gen1_runtime import Gen1SharedEncoderRuntime
from automata.runtime.effects import register_all_effects
from automata.search.continuation import (
    ArgmaxContinuationPolicy,
    PriorSamplingContinuationPolicy,
)
from automata.search.contracts import (
    ComponentInferenceError,
    ComponentUnavailableError,
    ContinuationPolicy,
    SearchContext,
    SearchPolicy,
    score_policy,
)
from automata.search.heuristic import HeuristicLeafEvaluator, HeuristicPrior
from automata.search.ismcts.strategy import (
    ISMCTSStrategy,
    SearchStrategy,
    StrategyResult,
)
from automata.search.learned import LearnedSearchPolicy, LearnedStableValueEvaluator
from automata.search.node import Key
from automata.search.root import RootTarget
from automata.training.io import atomic_write_bytes, fsync_directory
from automata.training.native_gen1 import (
    current_gen1_runtime_requirements,
    load_current_gen1_parent_artifact,
)
from automata.training.native_paired_contracts import (
    NativeEvaluatedArtifactIdentity,
    NativeEvaluationFailure,
    NativeEvaluationFixture,
    NativeEvaluationObservation,
    NativeGameplayComparisonKind,
    NativePairedEvaluationCompletion,
    NativePairedEvaluationManifest,
    NativePairedEvaluationResult,
    NativePlannedEvaluationCase,
    aggregate_native_evaluation,
    create_native_paired_evaluation_manifest,
    load_native_paired_evaluation_manifest,
    load_native_paired_evaluation_result,
    validate_native_evaluation_result,
)
from goa2.data.heroes import HeroRegistry
from goa2.domain.models import TeamColor
from goa2.domain.state import GameState
from goa2.domain.types import HeroID

_MAPS_ROOT = Path(__file__).resolve().parents[2] / "goa2" / "data" / "maps"
_SEED_RECIPE = "native-gen1-paired-evaluation-stream-v1"
NativeEvaluationSeedStream = Literal[
    "RED_SEARCH", "RED_ENVIRONMENT", "BLUE_SEARCH", "BLUE_ENVIRONMENT"
]

_STREAMS = frozenset({"RED_SEARCH", "RED_ENVIRONMENT", "BLUE_SEARCH", "BLUE_ENVIRONMENT"})


# Kept local rather than extending the engine's outcome contract: malformed
# normal returns are evaluation evidence failures, not engine draws.
class _OutcomeNormalizationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _LoadedArtifacts:
    candidate: Any
    parent: Any | None
    identities: tuple[NativeEvaluatedArtifactIdentity, ...]


@dataclass(frozen=True, slots=True)
class _FixtureRuntimes:
    candidate: Gen1SharedEncoderRuntime
    parent: Gen1SharedEncoderRuntime | None


@dataclass(frozen=True, slots=True)
class _AgentComponents:
    """Inspectable recipe used to construct one fresh physical-side agent."""

    agent: ISMCTSAgent
    environment: HeuristicAgent
    prior: SearchPolicy
    continuation: ContinuationPolicy | HeuristicAgent
    leaf: Any
    strategy: SearchStrategy


class _PriorOnlyStrategy:
    """Select the stable first policy argmax without invoking search.

    ``ISMCTSAgent`` remains only the public ownership/routing adapter.  This
    strategy calls the public policy scoring seam directly and intentionally
    returns no search statistics.
    """

    strategy_id = "native-policy-first-argmax-v1"

    def __init__(self, policy: SearchPolicy) -> None:
        self.policy = policy

    def select(
        self,
        state: GameState,
        perspective_team: TeamColor,
        root_target: RootTarget,
        legal_candidates: Sequence[Key],
    ) -> StrategyResult[Key]:
        legal = tuple(legal_candidates)
        if not legal:
            raise ValueError("policy-only selection requires at least one legal candidate")
        if len(legal) == 1:
            return StrategyResult(legal, 0, search_result=None)

        owner = state.get_hero(HeroID(root_target.decision_owner_hero_id))
        if owner is None or owner.team != perspective_team:
            raise ValueError("policy-only decision owner does not match its perspective")
        if root_target.kind == "CARD":
            if root_target.hero_id != str(owner.id):
                raise ValueError("policy-only CARD root does not match its owner")
            decision = DecisionDescriptor(
                "CARD",
                hero=owner,
                can_finish_planning=None in legal,
            )
        else:
            request = root_target.request
            if request is None:
                request = next(
                    (
                        item
                        for item in reversed(state.input_stack)
                        if item.id == root_target.request_id
                    ),
                    None,
                )
            if (
                request is None
                or request.id != root_target.request_id
                or request.player_id != root_target.player_id
            ):
                raise ValueError("policy-only INPUT root does not match the live request")
            decision = DecisionDescriptor("INPUT", request=request)

        context = SearchContext(
            root_viewer_id=root_target.decision_owner_hero_id,
            perspective_team=perspective_team,
            current_owner_id=root_target.decision_owner_hero_id,
            decision=decision,
        )
        scores = score_policy(self.policy, context, state, legal)
        selected = max(range(len(legal)), key=scores.scores.__getitem__)
        return StrategyResult(legal, selected, search_result=None)


def _canonical_seed_payload(
    manifest: NativePairedEvaluationManifest,
    *,
    comparison: NativeGameplayComparisonKind,
    fixture_id: str,
    world_seed: int,
    stream: NativeEvaluationSeedStream,
) -> bytes:
    side, stream_kind = stream.split("_", 1)
    values = (
        _SEED_RECIPE,
        manifest.config.random_stream_namespace,
        comparison,
        fixture_id,
        str(world_seed),
        side,
        stream_kind,
    )
    # Namespace and fixture IDs are caller strings: delimiter joining can alias
    # different tuples when either contains that delimiter. JSON preserves field
    # boundaries while keeping role/leg/artifact identity out of paired streams.
    return json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _derive_native_evaluation_seed_from_validated_manifest(
    manifest: NativePairedEvaluationManifest,
    *,
    comparison: NativeGameplayComparisonKind,
    fixture_id: str,
    world_seed: int,
    stream: NativeEvaluationSeedStream,
) -> int:
    return int.from_bytes(
        hashlib.sha256(
            _canonical_seed_payload(
                manifest,
                comparison=comparison,
                fixture_id=fixture_id,
                world_seed=world_seed,
                stream=stream,
            )
        ).digest(),
        "big",
    )


def derive_native_evaluation_seed(
    manifest: NativePairedEvaluationManifest,
    *,
    comparison: NativeGameplayComparisonKind,
    fixture_id: str,
    world_seed: int,
    stream: NativeEvaluationSeedStream,
) -> int:
    """Derive a deterministic stream keyed only by case setup and physical side."""
    if not isinstance(manifest, NativePairedEvaluationManifest):
        raise TypeError("manifest must be a NativePairedEvaluationManifest")
    validated = NativePairedEvaluationManifest.model_validate(
        manifest.model_dump(mode="python"), strict=True
    )
    if comparison not in validated.config.comparisons:
        raise ValueError("comparison is not declared by the evaluation manifest")
    fixture = next(
        (item for item in validated.config.fixtures if item.fixture_id == fixture_id), None
    )
    if fixture is None:
        raise ValueError("fixture is not declared by the evaluation manifest")
    if type(world_seed) is not int or world_seed not in fixture.world_seeds:
        raise ValueError("world_seed is not declared by the evaluation fixture")
    if stream not in _STREAMS:
        raise ValueError(f"unknown native evaluation seed stream {stream!r}")
    return _derive_native_evaluation_seed_from_validated_manifest(
        validated,
        comparison=comparison,
        fixture_id=fixture_id,
        world_seed=world_seed,
        stream=stream,
    )


def _require_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"{label} path must not contain symlinks")


def _artifact_identity(
    manifest: NativePairedEvaluationManifest,
    *,
    role: Literal["CANDIDATE", "PARENT"],
    artifact_path: Path,
    loaded: Any,
) -> NativeEvaluatedArtifactIdentity:
    artifact_manifest = loaded.manifest
    manifest_payload = (artifact_path / "manifest.json").read_bytes()
    return NativeEvaluatedArtifactIdentity(
        role=role,
        model_digest=artifact_manifest.model_digest,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        model_id=artifact_manifest.model_id,
        runtime_compatibility_version=artifact_manifest.runtime_compatibility_version,
        decision_tensor_schema_digest=artifact_manifest.decision_tensor_schema_digest,
        stable_value_tensor_schema_digest=(artifact_manifest.stable_value_tensor_schema_digest),
        value_semantics=artifact_manifest.value_semantics,
        current_scope_digest=manifest.current_scope.digest,
        device="cpu",
        floating_dtype="float32",
    )


def _preflight(manifest: NativePairedEvaluationManifest) -> _LoadedArtifacts:
    register_all_effects()
    root = manifest.authorities.output_root
    _require_no_symlink_components(root, label="native evaluation output root")
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"native evaluation output root already exists: {root}")
    if not root.parent.is_dir():
        raise FileNotFoundError(
            "native evaluation output root parent must be an existing directory"
        )

    candidate_path = manifest.authorities.candidate_artifact_path
    _require_no_symlink_components(candidate_path, label="native evaluation candidate artifact")
    if not candidate_path.is_dir():
        raise FileNotFoundError("native evaluation candidate artifact must be a directory")
    candidate = load_current_gen1_parent_artifact(
        candidate_path,
        expected_model_digest=manifest.config.candidate_model_digest,
    )

    parent = None
    parent_path = manifest.authorities.parent_artifact_path
    if parent_path is not None:
        _require_no_symlink_components(parent_path, label="native evaluation parent artifact")
        if not parent_path.is_dir():
            raise FileNotFoundError("native evaluation parent artifact must be a directory")
        assert manifest.config.parent_model_digest is not None
        parent = load_current_gen1_parent_artifact(
            parent_path,
            expected_model_digest=manifest.config.parent_model_digest,
        )
        if parent.manifest.model_digest == candidate.manifest.model_digest:
            raise ValueError("candidate and parent model digests must differ")

    # Requirements construction is a deliberate preflight boundary.  It proves
    # every declared fixture is representable before the output root is claimed.
    for fixture in manifest.config.fixtures:
        current_gen1_runtime_requirements(
            heroes=(*fixture.red_composition, *fixture.blue_composition),
            map_id=fixture.map_id,
            game_type=fixture.game_type,
        )

    identities = [
        _artifact_identity(
            manifest,
            role="CANDIDATE",
            artifact_path=candidate_path,
            loaded=candidate,
        )
    ]
    if parent is not None:
        assert parent_path is not None
        identities.append(
            _artifact_identity(
                manifest,
                role="PARENT",
                artifact_path=parent_path,
                loaded=parent,
            )
        )
    return _LoadedArtifacts(candidate, parent, tuple(identities))


def _runtime(loaded: Any, fixture: NativeEvaluationFixture) -> Gen1SharedEncoderRuntime:
    requirements = current_gen1_runtime_requirements(
        heroes=(*fixture.red_composition, *fixture.blue_composition),
        map_id=fixture.map_id,
        game_type=fixture.game_type,
    )
    loaded.model.eval()
    for parameter in loaded.model.parameters():
        parameter.requires_grad_(False)
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


def _build_runtimes(
    manifest: NativePairedEvaluationManifest,
    loaded: _LoadedArtifacts,
) -> dict[str, _FixtureRuntimes]:
    return {
        fixture.fixture_id: _FixtureRuntimes(
            candidate=_runtime(loaded.candidate, fixture),
            parent=(None if loaded.parent is None else _runtime(loaded.parent, fixture)),
        )
        for fixture in manifest.config.fixtures
    }


def _arm_kind(
    comparison: NativeGameplayComparisonKind,
    *,
    candidate_arm: bool,
) -> Literal[
    "CANDIDATE_FULL",
    "HEURISTIC_FULL",
    "CANDIDATE_LEARNED_VALUE",
    "CANDIDATE_HEURISTIC_VALUE",
    "CANDIDATE_POLICY_ONLY",
    "HEURISTIC_POLICY_ONLY",
    "PARENT_FULL",
]:
    if comparison == "CANDIDATE_VS_HEURISTIC_FULL_SEARCH":
        return "CANDIDATE_FULL" if candidate_arm else "HEURISTIC_FULL"
    if comparison == "CANDIDATE_VALUE_VS_HEURISTIC_VALUE_FIXED_CANDIDATE_POLICY":
        return "CANDIDATE_LEARNED_VALUE" if candidate_arm else "CANDIDATE_HEURISTIC_VALUE"
    if comparison == "CANDIDATE_VS_HEURISTIC_POLICY_ONLY":
        return "CANDIDATE_POLICY_ONLY" if candidate_arm else "HEURISTIC_POLICY_ONLY"
    if comparison == "CANDIDATE_VS_GEN1_PARENT_FULL_SEARCH":
        return "CANDIDATE_FULL" if candidate_arm else "PARENT_FULL"
    raise ValueError(f"unsupported native comparison {comparison!r}")


def _build_arm_agent(
    manifest: NativePairedEvaluationManifest,
    planned: NativePlannedEvaluationCase,
    runtimes: _FixtureRuntimes,
    *,
    side: Literal["RED", "BLUE"],
    candidate_arm: bool,
) -> _AgentComponents:
    search_seed = _derive_native_evaluation_seed_from_validated_manifest(
        manifest,
        comparison=planned.comparison,
        fixture_id=planned.fixture_id,
        world_seed=planned.world_seed,
        stream=cast(NativeEvaluationSeedStream, f"{side}_SEARCH"),
    )
    environment_seed = _derive_native_evaluation_seed_from_validated_manifest(
        manifest,
        comparison=planned.comparison,
        fixture_id=planned.fixture_id,
        world_seed=planned.world_seed,
        stream=cast(NativeEvaluationSeedStream, f"{side}_ENVIRONMENT"),
    )
    search_config = replace(manifest.config.search.to_search_config(), seed=search_seed)
    environment = HeuristicAgent(environment_seed)
    kind = _arm_kind(planned.comparison, candidate_arm=candidate_arm)

    runtime: Gen1SharedEncoderRuntime | None = None
    if kind.startswith("CANDIDATE"):
        runtime = runtimes.candidate
    elif kind == "PARENT_FULL":
        runtime = runtimes.parent
        if runtime is None:
            raise ComponentUnavailableError("parent comparison has no parent runtime")

    if kind in {
        "CANDIDATE_FULL",
        "CANDIDATE_LEARNED_VALUE",
        "CANDIDATE_HEURISTIC_VALUE",
        "CANDIDATE_POLICY_ONLY",
        "PARENT_FULL",
    }:
        assert runtime is not None
        prior: SearchPolicy = LearnedSearchPolicy(runtime)
    else:
        prior = HeuristicPrior(environment)

    if kind in {"CANDIDATE_FULL", "CANDIDATE_LEARNED_VALUE", "PARENT_FULL"}:
        assert runtime is not None
        continuation: ContinuationPolicy | HeuristicAgent = PriorSamplingContinuationPolicy(prior)
        leaf: Any = LearnedStableValueEvaluator(runtime)
    elif kind == "CANDIDATE_HEURISTIC_VALUE":
        continuation = PriorSamplingContinuationPolicy(prior)
        leaf = HeuristicLeafEvaluator()
    elif kind == "HEURISTIC_FULL":
        continuation = environment
        leaf = HeuristicLeafEvaluator()
    else:
        # Retained as explicit recipe metadata even though prior-only strategy
        # performs no rollout continuation.
        continuation = ArgmaxContinuationPolicy(prior)
        leaf = None

    if kind in {"CANDIDATE_POLICY_ONLY", "HEURISTIC_POLICY_ONLY"}:
        strategy: SearchStrategy = _PriorOnlyStrategy(prior)
    else:
        strategy = ISMCTSStrategy(
            environment_policy=environment,
            config=search_config,
            prior=prior,
            leaf_evaluator=leaf,
            continuation_policy=continuation,
        )

    agent = ISMCTSAgent(
        search_config,
        environment_policy=environment,
        continuation_policy=continuation,
        leaf_evaluator=leaf,
        prior=prior,
        strategy=strategy,
    )
    return _AgentComponents(agent, environment, prior, continuation, leaf, strategy)


def _hero_ids(composition: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for name in composition:
        hero = HeroRegistry.get(name)
        if hero is None:  # scope validation is expected to guard this
            raise ValueError(f"unknown native evaluation hero {name!r}")
        result.append(str(hero.id))
    return tuple(result)


def _build_case_agents(
    manifest: NativePairedEvaluationManifest,
    planned: NativePlannedEvaluationCase,
    fixture: NativeEvaluationFixture,
    runtimes: _FixtureRuntimes,
) -> tuple[dict[str, ISMCTSAgent], dict[str, _AgentComponents]]:
    by_side: dict[str, _AgentComponents] = {}
    agents: dict[str, ISMCTSAgent] = {}
    for side, composition in (
        ("RED", fixture.red_composition),
        ("BLUE", fixture.blue_composition),
    ):
        candidate_arm = side == planned.candidate_side
        built = _build_arm_agent(
            manifest,
            planned,
            runtimes,
            side=cast(Literal["RED", "BLUE"], side),
            candidate_arm=candidate_arm,
        )
        by_side[side] = built
        for hero_id in _hero_ids(composition):
            agents[hero_id] = built.agent
    return agents, by_side


def _observation(
    planned: NativePlannedEvaluationCase,
    outcome: RunResult,
) -> NativeEvaluationObservation:
    if outcome.reason == "game_over":
        if outcome.winner is None or outcome.winner_side not in {"RED", "BLUE"}:
            raise _OutcomeNormalizationError(
                "game_over requires both raw and normalized RED/BLUE winner"
            )
        status: Literal["COMPLETED", "CENSORED"] = "COMPLETED"
    elif outcome.reason in {"max_steps", "max_rounds"}:
        if outcome.winner is not None or outcome.winner_side is not None:
            raise _OutcomeNormalizationError("censored game cannot declare a winner")
        status = "CENSORED"
    else:
        raise _OutcomeNormalizationError(
            f"unrecognized native evaluation outcome reason {outcome.reason!r}"
        )
    return NativeEvaluationObservation(
        case_id=planned.case_id,
        pair_id=planned.pair_id,
        comparison=planned.comparison,
        fixture_id=planned.fixture_id,
        world_seed=planned.world_seed,
        candidate_side=planned.candidate_side,
        status=status,
        raw_winner=outcome.winner,
        winner_side=outcome.winner_side,
        reason=cast(Literal["game_over", "max_steps", "max_rounds"], outcome.reason),
        rounds=outcome.rounds,
        turns=outcome.turns,
        steps=outcome.steps,
    )


def _publish_new(path: Path, payload: bytes) -> os.stat_result:
    """Durably hard-link new bytes, rolling back only the inode we created."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    owned: os.stat_result | None = None
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
            # Ownership must come from the open temporary fd before linking.
            owned = os.fstat(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"native evaluation path appeared during publication: {path}"
            ) from exc
        fsync_directory(path.parent)
        current = path.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino):
            raise RuntimeError("native evaluation publication was replaced")
        if path.read_bytes() != payload:
            raise RuntimeError("native evaluation publication bytes changed")
        temporary.unlink()
        return owned
    except BaseException as original:
        if owned is not None:
            try:
                _unlink_if_same(path, owned)
            except BaseException as rollback_error:
                original.add_note(
                    "Additionally failed to roll back owned native evaluation publication: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                )
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            original.add_note(
                "Additionally failed to clean native evaluation temporary: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise


def _unlink_if_same(path: Path, identity: os.stat_result) -> bool:
    try:
        current = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
        return False
    path.unlink()
    fsync_directory(path.parent)
    return True


def _replace_owned(
    path: Path,
    payload: bytes,
    *,
    identity: os.stat_result,
    previous: bytes,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("native evaluation result ownership changed")
    current = path.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (
        identity.st_dev,
        identity.st_ino,
    ) or path.read_bytes() != previous:
        raise RuntimeError("native evaluation result ownership changed")
    atomic_write_bytes(path, payload)


def _safe_message(error: BaseException) -> str:
    text = str(error).encode("utf-8", errors="replace").decode("utf-8")
    return text[:1000].strip() or type(error).__name__


def _failure_category(error: BaseException, *, publication: bool) -> Literal[
    "INFERENCE_FAILURE",
    "COMPONENT_UNAVAILABLE",
    "OUTCOME_NORMALIZATION_FAILURE",
    "GAMEPLAY_FAILURE",
    "PUBLICATION_FAILURE",
]:
    if publication:
        return "PUBLICATION_FAILURE"
    if isinstance(error, ComponentInferenceError):
        return "INFERENCE_FAILURE"
    if isinstance(error, (ComponentUnavailableError, ArtifactError)):
        return "COMPONENT_UNAVAILABLE"
    if isinstance(error, _OutcomeNormalizationError) or (
        isinstance(error, ValueError)
        and any(
            marker in str(error)
            for marker in (
                "terminal outcome",
                "terminal winner",
                "winning team",
            )
        )
    ):
        return "OUTCOME_NORMALIZATION_FAILURE"
    return "GAMEPLAY_FAILURE"


def _raise_original_with_status_failure(
    original: BaseException,
    traceback: Any,
    status_error: BaseException,
) -> NoReturn:
    original.add_note(
        "Additionally failed to persist native evaluation FAILED status: "
        f"{type(status_error).__name__}: {_safe_message(status_error)}"
    )
    raise original.with_traceback(traceback) from status_error


def run_native_paired_evaluation(
    manifest: NativePairedEvaluationManifest,
) -> NativePairedEvaluationResult:
    """Play the exact finite paired schedule and publish digest-bound evidence."""
    if not isinstance(manifest, NativePairedEvaluationManifest):
        raise TypeError("manifest must be a NativePairedEvaluationManifest")
    register_all_effects()
    validated = NativePairedEvaluationManifest.model_validate(
        manifest.model_dump(mode="python"), strict=True
    )
    rederived = create_native_paired_evaluation_manifest(validated.config, validated.authorities)
    if validated != rederived:
        raise ValueError("native evaluation manifest is not canonically derived")

    # All model, fixture, and runtime compatibility checks happen before output
    # mutation so setup failures cannot claim an evidence authority.
    loaded = _preflight(validated)
    runtimes = _build_runtimes(validated, loaded)
    root = validated.authorities.output_root
    root.mkdir(exist_ok=False)

    attempted: list[str] = []
    observations: list[NativeEvaluationObservation] = []
    current_case: NativePlannedEvaluationCase | None = None
    result_path = root / "result.json"
    result_identity: os.stat_result | None = None
    result_payload: bytes | None = None
    marker_identity: os.stat_result | None = None
    publication = True
    try:
        fsync_directory(root.parent)
        _publish_new(root / "manifest.json", validated.canonical_bytes())
        publication = False
        fixtures = {fixture.fixture_id: fixture for fixture in validated.config.fixtures}

        for planned in validated.planned_cases:
            current_case = planned
            attempted.append(planned.case_id)
            fixture = fixtures[planned.fixture_id]
            agents, _ = _build_case_agents(
                validated,
                planned,
                fixture,
                runtimes[fixture.fixture_id],
            )
            outcome = run_game(
                list(fixture.red_composition),
                list(fixture.blue_composition),
                agents,
                map_path=str(_MAPS_ROOT / f"{fixture.map_id}.json"),
                game_type=fixture.game_type,
                seed=planned.world_seed,
                max_steps=validated.config.max_steps,
                max_rounds=validated.config.max_rounds,
            )
            observations.append(_observation(planned, outcome))

        # From this point failures concern deriving or publishing terminal
        # evidence, not an unrecorded gameplay case.
        publication = True
        completed_pairs, aggregates = aggregate_native_evaluation(validated, tuple(observations))
        succeeded = NativePairedEvaluationResult(
            status="SUCCEEDED",
            manifest_digest=validated.digest,
            config_digest=validated.config.digest,
            artifact_identities=loaded.identities,
            planned_case_count=len(validated.planned_cases),
            attempted_case_ids=tuple(attempted),
            observations=tuple(observations),
            completed_pairs=completed_pairs,
            aggregates=aggregates,
            failure=None,
        )
        succeeded = validate_native_evaluation_result(validated, succeeded)
        publication = True
        result_payload = succeeded.canonical_bytes()
        result_identity = _publish_new(result_path, result_payload)
        reloaded = load_native_paired_evaluation_result(result_path)
        if reloaded != succeeded:
            raise ValueError("published native evaluation result changed")
        if load_native_paired_evaluation_manifest(root / "manifest.json") != validated:
            raise ValueError("published native evaluation manifest changed")
        validate_native_evaluation_result(validated, reloaded)
        marker = NativePairedEvaluationCompletion(
            manifest_digest=validated.digest,
            result_digest=succeeded.digest,
        )
        marker_identity = _publish_new(root / "complete.json", marker.canonical_bytes())
        return succeeded
    except BaseException as original:
        _, _, traceback = sys.exc_info()
        if marker_identity is not None:
            with suppress(BaseException):
                _unlink_if_same(root / "complete.json", marker_identity)
        try:
            if current_case is None and not publication:
                # Defensively attribute any post-manifest, pre-gameplay setup
                # failure to the first scheduled case. Runtime construction and
                # compatibility checks have already completed before root claim.
                current_case = validated.planned_cases[0]
                attempted.append(current_case.case_id)
            failed = NativePairedEvaluationResult(
                status="FAILED",
                manifest_digest=validated.digest,
                config_digest=validated.config.digest,
                artifact_identities=loaded.identities,
                planned_case_count=len(validated.planned_cases),
                attempted_case_ids=tuple(attempted),
                observations=tuple(observations),
                completed_pairs=(),
                aggregates=(),
                failure=NativeEvaluationFailure(
                    case_id=(None if publication or current_case is None else current_case.case_id),
                    category=_failure_category(original, publication=publication),
                    error_type=type(original).__name__,
                    message=_safe_message(original),
                ),
            )
            failed = validate_native_evaluation_result(validated, failed)
            failed_payload = failed.canonical_bytes()
            if result_identity is None:
                if result_path.exists() or result_path.is_symlink():
                    raise RuntimeError(
                        "native evaluation result path is not owned; refusing to clobber it"
                    )
                _publish_new(result_path, failed_payload)
            else:
                assert result_payload is not None
                _replace_owned(
                    result_path,
                    failed_payload,
                    identity=result_identity,
                    previous=result_payload,
                )
        except BaseException as status_error:
            _raise_original_with_status_failure(original, traceback, status_error)
        raise


__all__ = [
    "NativeEvaluationSeedStream",
    "derive_native_evaluation_seed",
    "run_native_paired_evaluation",
]
