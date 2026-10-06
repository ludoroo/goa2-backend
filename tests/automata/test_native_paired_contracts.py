"""Strict native paired-evaluation contract tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from automata.models.shared_encoder.gen1_model import GEN1_ARCHITECTURE_ID
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.search.contracts import CutoffUnit, LeafMode
from automata.training.native_paired_contracts import (
    NativeEvaluatedArtifactIdentity,
    NativeEvaluationFailure,
    NativeEvaluationFixture,
    NativeEvaluationObservation,
    NativePairedEvaluationAuthorities,
    NativePairedEvaluationCompletion,
    NativePairedEvaluationConfig,
    NativePairedEvaluationResult,
    aggregate_native_evaluation,
    create_native_paired_evaluation_manifest,
    load_completed_native_paired_evaluation,
    load_native_paired_evaluation_manifest,
    validate_native_evaluation_result,
)
from automata.training.native_run_contracts import NativeRunSearchConfig
from automata.training.native_splits import NativeSeedRange, NativeSplitConfig

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_MANDATORY = (
    "CANDIDATE_VS_HEURISTIC_FULL_SEARCH",
    "CANDIDATE_VALUE_VS_HEURISTIC_VALUE_FIXED_CANDIDATE_POLICY",
    "CANDIDATE_VS_HEURISTIC_POLICY_ONLY",
)


def _search(**changes: Any) -> NativeRunSearchConfig:
    values: dict[str, Any] = {
        "iterations": 2,
        "decision_timeout_seconds": None,
        "max_advance_steps": 100,
        "uct_c": 1.4,
        "cutoff_limit": 2,
        "cutoff_unit": CutoffUnit.ROUNDS,
        "max_advance_transitions": 100,
        "max_forced_decisions": 50,
        "leaf_mode": LeafMode.STABLE_TRANSITION,
        "widening_c": 2.0,
        "widening_alpha": 0.5,
        "root_widening_c": None,
        "root_widening_alpha": None,
        "adaptive_hex_root_schedule_version": None,
        "request_schedule_version": None,
        "seed": 0,
        "use_prior": True,
        "puct_c": 1.0,
        "root_puct_c": None,
    }
    values.update(changes)
    return NativeRunSearchConfig(**values)


def _split() -> NativeSplitConfig:
    return NativeSplitConfig(
        namespace="paired-contract-tests",
        salt="fixed",
        validation_fraction=0.2,
        seed_ranges=(NativeSeedRange(purpose="evaluation", start=500, stop=510),),
    )


def _config(**changes: Any) -> NativePairedEvaluationConfig:
    values: dict[str, Any] = {
        "evaluation_id": "paired-contract-test",
        "source_revision": "test-revision",
        "dirty_tree_hash": "clean-test-tree",
        "candidate_model_digest": _DIGEST_A,
        "parent_model_digest": None,
        "split_config": _split(),
        "comparisons": _MANDATORY,
        "fixtures": (
            NativeEvaluationFixture(
                fixture_id="quick-1v1",
                map_id="forgotten_island",
                game_type="QUICK",
                red_composition=("Wasp",),
                blue_composition=("Arien",),
                world_seeds=(500,),
            ),
        ),
        "search": _search(),
        "random_stream_namespace": "paired-contract-streams",
        "max_steps": 30,
        "max_rounds": 3,
    }
    values.update(changes)
    return NativePairedEvaluationConfig(**values)


def _authorities(tmp_path: Path, *, parent: bool = False) -> NativePairedEvaluationAuthorities:
    return NativePairedEvaluationAuthorities(
        output_root=(tmp_path / "evaluation").absolute(),
        candidate_artifact_path=(tmp_path / "candidate").absolute(),
        parent_artifact_path=(tmp_path / "parent").absolute() if parent else None,
    )


def _manifest(tmp_path: Path):
    return create_native_paired_evaluation_manifest(_config(), _authorities(tmp_path))


def _artifact(manifest, role: str = "CANDIDATE") -> NativeEvaluatedArtifactIdentity:
    return NativeEvaluatedArtifactIdentity(
        role=role,
        model_digest=(
            manifest.config.candidate_model_digest
            if role == "CANDIDATE"
            else manifest.config.parent_model_digest
        ),
        manifest_sha256="c" * 64,
        model_id=GEN1_ARCHITECTURE_ID,
        runtime_compatibility_version=1,
        decision_tensor_schema_digest=TensorFeatureSchema.current().digest,
        stable_value_tensor_schema_digest=StableValueTensorSchema.current().digest,
        value_semantics="stable-boundary-outcome-v1",
        current_scope_digest=manifest.current_scope.digest,
        device="cpu",
        floating_dtype="float32",
    )


def _observation(case, *, winner: str | None = "RED", reason: str = "game_over"):
    completed = reason == "game_over"
    return NativeEvaluationObservation(
        case_id=case.case_id,
        pair_id=case.pair_id,
        comparison=case.comparison,
        fixture_id=case.fixture_id,
        world_seed=case.world_seed,
        candidate_side=case.candidate_side,
        status="COMPLETED" if completed else "CENSORED",
        raw_winner=winner if completed else None,
        winner_side=winner if completed else None,
        reason=reason,
        rounds=1,
        turns=2,
        steps=3,
    )


def _successful_result(manifest, observations):
    pairs, aggregates = aggregate_native_evaluation(manifest, observations)
    return NativePairedEvaluationResult(
        status="SUCCEEDED",
        manifest_digest=manifest.digest,
        config_digest=manifest.config.digest,
        artifact_identities=(_artifact(manifest),),
        planned_case_count=len(manifest.planned_cases),
        attempted_case_ids=tuple(case.case_id for case in manifest.planned_cases),
        observations=tuple(observations),
        completed_pairs=pairs,
        aggregates=aggregates,
        failure=None,
    )


def test_manifest_scope_remains_canonical_after_effect_bootstrap_in_fresh_process(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    authorities_path = tmp_path / "authorities.json"
    config_path.write_bytes(_config().canonical_bytes())
    authorities_path.write_bytes(_authorities(tmp_path).canonical_bytes())
    script = """
import sys
from pathlib import Path
from automata.runtime.effects import register_all_effects
from automata.training.native_paired_contracts import (
    NativePairedEvaluationAuthorities,
    NativePairedEvaluationConfig,
    NativePairedEvaluationManifest,
    create_native_paired_evaluation_manifest,
)
config = NativePairedEvaluationConfig.model_validate_json(Path(sys.argv[1]).read_bytes(), strict=True)
authorities = NativePairedEvaluationAuthorities.model_validate_json(
    Path(sys.argv[2]).read_bytes(), strict=True
)
manifest = create_native_paired_evaluation_manifest(config, authorities)
register_all_effects()
revalidated = NativePairedEvaluationManifest.model_validate(
    manifest.model_dump(mode="python"), strict=True
)
assert revalidated == manifest
assert {"Knight", "Rogue"} <= set(manifest.current_scope.supported_heroes)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(authorities_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_manifest_has_canonical_scope_maps_and_fixed_roster_agent_swap(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    assert len(manifest.planned_cases) == 6
    assert (
        tuple(case.candidate_side for case in manifest.planned_cases)
        == (
            "RED",
            "BLUE",
        )
        * 3
    )
    assert len({case.pair_id for case in manifest.planned_cases}) == 3
    assert len({case.case_id for case in manifest.planned_cases}) == 6
    assert manifest.pairing_recipe == "fixed-board-rosters-contestant-side-swap-v1"
    assert tuple(item.map_id for item in manifest.maps) == ("forgotten_island",)
    assert manifest.maps[0].length > 0
    assert manifest.current_scope.supported_heroes == tuple(
        sorted(manifest.current_scope.supported_heroes)
    )
    assert manifest == create_native_paired_evaluation_manifest(_config(), _authorities(tmp_path))


def test_config_digest_is_path_independent_but_manifest_digest_is_authority_bound(
    tmp_path: Path,
) -> None:
    config = _config()
    first = create_native_paired_evaluation_manifest(config, _authorities(tmp_path))
    second = create_native_paired_evaluation_manifest(
        config,
        NativePairedEvaluationAuthorities(
            output_root=(tmp_path / "other-evaluation").absolute(),
            candidate_artifact_path=(tmp_path / "other-candidate").absolute(),
            parent_artifact_path=None,
        ),
    )

    assert first.config.digest == second.config.digest == config.digest
    assert first.digest != second.digest


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"comparisons": _MANDATORY[:2]}, "mandatory"),
        ({"comparisons": (*_MANDATORY, _MANDATORY[0])}, "duplicate"),
        ({"search": _search(use_prior=False)}, "use_prior"),
        ({"max_steps": 0}, "greater than 0"),
        ({"max_rounds": 0}, "greater than 0"),
    ],
)
def test_config_rejects_missing_modes_unpinned_search_and_caps(
    change: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _config(**change)


def test_manifest_rejects_non_evaluation_seeds_unknown_scope_and_parent_mismatch(
    tmp_path: Path,
) -> None:
    train_split = NativeSplitConfig(
        namespace="bad",
        salt="bad",
        validation_fraction=0.2,
        seed_ranges=(NativeSeedRange(purpose="training", start=500, stop=501),),
    )
    with pytest.raises((ValidationError, ValueError), match="evaluation"):
        create_native_paired_evaluation_manifest(
            _config(split_config=train_split), _authorities(tmp_path)
        )

    bad_fixture = _config().fixtures[0].model_copy(update={"map_id": "missing-map"})
    unsafe_config = _config().model_copy(update={"fixtures": (bad_fixture,)})
    with pytest.raises((ValidationError, ValueError), match="map"):
        create_native_paired_evaluation_manifest(unsafe_config, _authorities(tmp_path))

    with pytest.raises(ValueError, match="parent"):
        create_native_paired_evaluation_manifest(
            _config(parent_model_digest=_DIGEST_B), _authorities(tmp_path)
        )


def test_optional_parent_identity_path_and_comparison_form_one_mode(tmp_path: Path) -> None:
    parent_kind = "CANDIDATE_VS_GEN1_PARENT_FULL_SEARCH"
    config = _config(
        parent_model_digest=_DIGEST_B,
        comparisons=(*_MANDATORY, parent_kind),
    )

    manifest = create_native_paired_evaluation_manifest(config, _authorities(tmp_path, parent=True))

    assert manifest.config.comparisons[-1] == parent_kind
    assert len(manifest.planned_cases) == 8
    assert tuple(case.candidate_side for case in manifest.planned_cases[-2:]) == (
        "RED",
        "BLUE",
    )


def test_manifest_rejects_overlapping_or_symlinked_authorities(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with pytest.raises(ValidationError, match="disjoint"):
        NativePairedEvaluationAuthorities(
            output_root=(candidate / "output").absolute(),
            candidate_artifact_path=candidate.absolute(),
            parent_artifact_path=None,
        )

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    authorities = NativePairedEvaluationAuthorities(
        output_root=(tmp_path / "output").absolute(),
        candidate_artifact_path=(link / "candidate").absolute(),
        parent_artifact_path=None,
    )
    with pytest.raises(ValueError, match="symlink"):
        create_native_paired_evaluation_manifest(_config(), authorities)


def test_aggregate_scores_only_two_decisive_mates_and_reports_censoring(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    cases = manifest.planned_cases
    observations = (
        _observation(cases[0], winner="RED"),
        _observation(cases[1], winner="BLUE"),  # candidate wins both => 1
        _observation(cases[2], winner="BLUE"),
        _observation(cases[3], winner="BLUE"),  # split => .5
        _observation(cases[4], winner="RED"),
        _observation(cases[5], winner=None, reason="max_rounds"),
    )

    pairs, aggregates = aggregate_native_evaluation(manifest, observations)

    assert tuple(pair.candidate_pair_score for pair in pairs) == (1.0, 0.5)
    assert aggregates[0].candidate_game_wins_in_completed_pairs == 2
    assert aggregates[0].descriptive_mean_completed_pair_score == 1.0
    assert aggregates[2].completed_pair_count == 0
    assert aggregates[2].censored_pair_count == 1
    assert aggregates[2].excluded_completed_singletons == 1
    assert aggregates[2].candidate_game_wins_in_completed_pairs == 0
    assert aggregates[2].comparator_game_wins_in_completed_pairs == 0
    assert aggregates[2].descriptive_mean_completed_pair_score is None
    assert aggregates[2].censor_reasons == {"max_rounds": 1}


def test_observation_allows_harness_round_limit_counter_and_rejects_later_round(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    case = manifest.planned_cases[0]
    censored = _observation(case, winner=None, reason="max_rounds").model_copy(
        update={"rounds": manifest.config.max_rounds + 1}
    )

    # Exercise the observation budget through aggregation's public validation
    # seam by constructing a complete all-censored schedule below.
    observations = tuple(
        _observation(item, winner=None, reason="max_rounds").model_copy(
            update={"rounds": manifest.config.max_rounds + 1}
        )
        for item in manifest.planned_cases
    )
    pairs, aggregates = aggregate_native_evaluation(manifest, observations)
    assert pairs == ()
    assert all(item.descriptive_mean_completed_pair_score is None for item in aggregates)

    too_late = censored.model_copy(update={"rounds": manifest.config.max_rounds + 2})
    with pytest.raises(ValueError, match="max_rounds"):
        aggregate_native_evaluation(
            manifest,
            (too_late, *observations[1:]),
        )


def test_observation_rejects_fake_draws_winners_on_censors_and_coerced_counts() -> None:
    values = {
        "case_id": "1" * 64,
        "pair_id": "2" * 64,
        "comparison": _MANDATORY[0],
        "fixture_id": "fixture",
        "world_seed": 500,
        "candidate_side": "RED",
        "status": "COMPLETED",
        "raw_winner": None,
        "winner_side": None,
        "reason": "game_over",
        "rounds": 1,
        "turns": 2,
        "steps": 3,
    }
    with pytest.raises(ValidationError, match="winner"):
        NativeEvaluationObservation(**values)
    with pytest.raises(ValidationError):
        NativeEvaluationObservation(
            **{
                **values,
                "status": "CENSORED",
                "reason": "max_steps",
                "raw_winner": "RED",
                "winner_side": "RED",
            }
        )
    with pytest.raises(ValidationError):
        NativeEvaluationObservation(
            **{
                **values,
                "raw_winner": "RED",
                "winner_side": "RED",
                "steps": "3",
            }
        )


def test_result_validator_rederives_success_and_rejects_nested_tampering(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    observations = tuple(_observation(case, winner="RED") for case in manifest.planned_cases)
    result = _successful_result(manifest, observations)

    validated = validate_native_evaluation_result(manifest, result)

    assert validated == result
    assert validated.evaluation_scope == "DECLARED_EVALUATION_SEEDS"
    assert validated.artifact_training_exposure == "UNKNOWN"
    assert validated.strength_claim == "DESCRIPTIVE_COMPLETED_PAIRS_ONLY"

    forged_observation = observations[0].model_copy(update={"case_id": "forged"})
    forged = result.model_copy(update={"observations": (forged_observation, *observations[1:])})
    with pytest.raises((ValidationError, ValueError), match=r"observation|case"):
        validate_native_evaluation_result(manifest, forged)

    forged_aggregate = result.aggregates[0].model_copy(
        update={"candidate_game_wins_in_completed_pairs": 99}
    )
    forged = result.model_copy(update={"aggregates": (forged_aggregate, *result.aggregates[1:])})
    with pytest.raises((ValidationError, ValueError), match="aggregate"):
        validate_native_evaluation_result(manifest, forged)


def test_completed_loader_requires_canonical_digest_bound_evidence_but_not_artifacts(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    observations = tuple(_observation(case, winner="RED") for case in manifest.planned_cases)
    result = _successful_result(manifest, observations)
    root = manifest.authorities.output_root
    root.mkdir()
    (root / "manifest.json").write_bytes(manifest.canonical_bytes())
    (root / "result.json").write_bytes(result.canonical_bytes())
    completion = NativePairedEvaluationCompletion(
        manifest_digest=manifest.digest,
        result_digest=result.digest,
    )
    (root / "complete.json").write_bytes(completion.canonical_bytes())

    assert not manifest.authorities.candidate_artifact_path.exists()
    assert load_completed_native_paired_evaluation(root) == (manifest, result)

    (root / "manifest.json").write_bytes(manifest.canonical_bytes() + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        load_native_paired_evaluation_manifest(root / "manifest.json")


def test_manifest_rejects_unsafe_scope_map_and_plan_copies(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    for update, match in (
        (
            {"current_scope": manifest.current_scope.model_copy(update={"digest": "0" * 64})},
            "scope|digest",
        ),
        (
            {"maps": (manifest.maps[0].model_copy(update={"sha256": "0" * 64}),)},
            "map",
        ),
        (
            {
                "planned_cases": (
                    manifest.planned_cases[0].model_copy(update={"ordinal": 5}),
                    *manifest.planned_cases[1:],
                )
            },
            "planned",
        ),
    ):
        unsafe = manifest.model_copy(update=update)
        with pytest.raises((ValidationError, ValueError), match=rf"{match}"):
            type(manifest).model_validate(unsafe.model_dump(mode="python"), strict=True)


def test_failed_result_retains_exact_attempt_prefix_without_strength_aggregate(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    first, failing = manifest.planned_cases[:2]
    result = NativePairedEvaluationResult(
        status="FAILED",
        manifest_digest=manifest.digest,
        config_digest=manifest.config.digest,
        artifact_identities=(_artifact(manifest),),
        planned_case_count=len(manifest.planned_cases),
        attempted_case_ids=(first.case_id, failing.case_id),
        observations=(_observation(first),),
        completed_pairs=(),
        aggregates=(),
        failure=NativeEvaluationFailure(
            case_id=failing.case_id,
            category="INFERENCE_FAILURE",
            error_type="ComponentInferenceError",
            message="inference failed",
        ),
    )

    assert validate_native_evaluation_result(manifest, result) == result

    unsafe = result.model_copy(
        update={
            "aggregates": _successful_result(
                manifest,
                tuple(_observation(case) for case in manifest.planned_cases),
            ).aggregates
        }
    )
    with pytest.raises((ValidationError, ValueError), match=r"FAILED|aggregate"):
        validate_native_evaluation_result(manifest, unsafe)
