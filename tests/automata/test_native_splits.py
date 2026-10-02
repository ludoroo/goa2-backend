"""Deterministic native seed split ledger contracts."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from automata.training.native_splits import (
    NativeSeedAssignment,
    NativeSeedRange,
    NativeSeedSplitLedger,
    NativeSplitConfig,
    create_native_split_ledger,
    extend_native_split_ledger,
)


def _config() -> NativeSplitConfig:
    return NativeSplitConfig(
        namespace="gen1-self-play",
        salt="2026-10-01",
        validation_fraction=0.2,
        seed_ranges=(
            NativeSeedRange(purpose="training", start=0, stop=100),
            NativeSeedRange(purpose="bootstrap", start=100, stop=200),
            NativeSeedRange(purpose="validation", start=200, stop=210),
            NativeSeedRange(purpose="evaluation", start=300, stop=310),
        ),
    )


def test_hash_recipe_has_stable_canonical_golden_assignment() -> None:
    config = _config()
    ledger = extend_native_split_ledger(create_native_split_ledger(config), (0, 1, 200))

    # The recipe hashes only this domain-separated canonical JSON. In particular,
    # no game ID, generation, cohort, map, composition, or arrival ordinal enters it.
    recipe_bytes = json.dumps(
        {
            "recipe": "native-seed-split-v1",
            "namespace": "gen1-self-play",
            "salt": "2026-10-01",
            "world_seed": 0,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert recipe_bytes == (
        b'{"namespace":"gen1-self-play","recipe":"native-seed-split-v1",'
        b'"salt":"2026-10-01","world_seed":0}'
    )
    assert (
        hashlib.sha256(recipe_bytes).hexdigest()
        == "b9edb74f84903939c99714f9a8dba58c77b20a1b2d9e712cda103e62725985bf"
    )
    assert tuple((item.world_seed, item.split) for item in ledger.assignments) == (
        (0, "train"),
        (1, "validation"),
        (200, "validation"),
    )


def test_config_and_ledger_are_canonical_digestible_and_reloadable() -> None:
    config = _config()
    created = create_native_split_ledger(config)
    extended = extend_native_split_ledger(created, (9, 2, 200))

    assert json.loads(config.canonical_bytes())["recipe"] == "native-seed-split-v1"
    assert config.digest == hashlib.sha256(config.canonical_bytes()).hexdigest()
    assert NativeSplitConfig.model_validate_json(config.canonical_bytes(), strict=True) == config
    assert (
        NativeSeedSplitLedger.model_validate_json(extended.canonical_bytes(), strict=True)
        == extended
    )
    assert extended.digest == hashlib.sha256(extended.canonical_bytes()).hexdigest()
    assert created.assignments == ()
    assert tuple(item.world_seed for item in extended.assignments) == (2, 9, 200)


def test_extension_is_order_independent_idempotent_and_append_only() -> None:
    empty = create_native_split_ledger(_config())
    forward = extend_native_split_ledger(empty, (3, 1, 2, 3, 1))
    reverse = extend_native_split_ledger(empty, (2, 1, 3))
    repeated = extend_native_split_ledger(forward, (1, 2, 3, 3))
    later_generation = extend_native_split_ledger(empty, (3,))

    assert forward == reverse == repeated
    assert forward.split_for_seed(3) == later_generation.split_for_seed(3)
    assert extend_native_split_ledger(forward, (7,)).assignments[:3] == forward.assignments
    assert empty.assignments == ()


def test_dedicated_validation_is_validation_and_excluded_purposes_are_rejected_atomically() -> None:
    ledger = extend_native_split_ledger(create_native_split_ledger(_config()), (4,))
    assert extend_native_split_ledger(ledger, (200,)).split_for_seed(200) == "validation"

    for seed in (300, 999):
        with pytest.raises(ValueError, match=r"not eligible|not declared"):
            extend_native_split_ledger(ledger, (seed,))
        assert tuple(item.world_seed for item in ledger.assignments) == (4,)
    with pytest.raises(ValueError, match="not eligible"):
        extend_native_split_ledger(ledger, (5, 300, 6))
    assert tuple(item.world_seed for item in ledger.assignments) == (4,)

    for purpose in ("evaluation", "arena", "screen", "promotion"):
        config = NativeSplitConfig(
            namespace="excluded",
            salt="v1",
            validation_fraction=0.5,
            seed_ranges=(NativeSeedRange(purpose=purpose, start=10, stop=11),),
        )
        with pytest.raises(ValueError, match="not eligible"):
            extend_native_split_ledger(create_native_split_ledger(config), (10,))


def test_ranges_are_explicit_nonempty_disjoint_and_strict() -> None:
    with pytest.raises(ValidationError):
        NativeSeedRange(purpose="training", start=True, stop=2)
    with pytest.raises(ValidationError):
        NativeSeedRange(purpose="training", start=-1, stop=2)
    with pytest.raises(ValidationError, match="stop"):
        NativeSeedRange(purpose="training", start=2, stop=2)
    with pytest.raises(ValidationError):
        NativeSeedRange(purpose="other", start=0, stop=1)  # type: ignore[arg-type]

    repeated_purpose = NativeSplitConfig(
        namespace="ranges",
        salt="salt",
        validation_fraction=0.5,
        seed_ranges=(
            NativeSeedRange(purpose="training", start=0, stop=2),
            NativeSeedRange(purpose="training", start=2, stop=4),
        ),
    )
    assert len(repeated_purpose.seed_ranges) == 2
    with pytest.raises(ValidationError, match="overlap"):
        NativeSplitConfig(
            namespace="ranges",
            salt="salt",
            validation_fraction=0.5,
            seed_ranges=(
                NativeSeedRange(purpose="training", start=0, stop=3),
                NativeSeedRange(purpose="validation", start=2, stop=4),
            ),
        )


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf")])
def test_config_rejects_invalid_validation_fractions(fraction: float) -> None:
    with pytest.raises(ValidationError):
        NativeSplitConfig(
            namespace="fraction",
            salt="salt",
            validation_fraction=fraction,
            seed_ranges=(NativeSeedRange(purpose="training", start=0, stop=1),),
        )


@pytest.mark.parametrize("field", ["namespace", "salt"])
def test_config_requires_nonempty_namespace_salt_and_ranges(field: str) -> None:
    values = {
        "namespace": "namespace",
        "salt": "salt",
        "validation_fraction": 0.5,
        "seed_ranges": (NativeSeedRange(purpose="training", start=0, stop=1),),
    }
    values[field] = ""
    with pytest.raises(ValidationError):
        NativeSplitConfig(**values)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        NativeSplitConfig(
            namespace="namespace",
            salt="salt",
            validation_fraction=0.5,
            seed_ranges=(),
        )
    with pytest.raises(ValidationError):
        NativeSplitConfig(
            schema_version=True,  # type: ignore[arg-type]
            namespace="namespace",
            salt="salt",
            validation_fraction=0.5,
            seed_ranges=(NativeSeedRange(purpose="training", start=0, stop=1),),
        )


def test_world_seed_inputs_are_strict_and_lookup_requires_enrollment() -> None:
    ledger = create_native_split_ledger(_config())
    for seed in (True, -1, 1.0, "1"):
        with pytest.raises((TypeError, ValueError, ValidationError)):
            extend_native_split_ledger(ledger, (seed,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not enrolled"):
        ledger.split_for_seed(1)
    with pytest.raises((TypeError, ValueError), match="integer"):
        ledger.split_for_seed(True)


def test_assignments_must_be_unique_sorted_and_match_the_recipe() -> None:
    config = _config()
    expected = extend_native_split_ledger(create_native_split_ledger(config), (1, 2))
    assignments = expected.assignments

    with pytest.raises(ValidationError, match=r"sorted|unique"):
        NativeSeedSplitLedger(config=config, assignments=tuple(reversed(assignments)))
    with pytest.raises(ValidationError, match=r"sorted|unique"):
        NativeSeedSplitLedger(config=config, assignments=(assignments[0], assignments[0]))
    with pytest.raises(ValidationError, match="does not match"):
        NativeSeedSplitLedger(
            config=config,
            assignments=(NativeSeedAssignment(world_seed=1, split="train"),),
        )
    with pytest.raises(ValidationError, match=r"eligible|declared"):
        NativeSeedSplitLedger(
            config=config,
            assignments=(NativeSeedAssignment(world_seed=300, split="train"),),
        )


def test_public_operations_revalidate_copied_or_constructed_models_before_trust() -> None:
    config = _config()
    invalid_config = config.model_copy(update={"validation_fraction": float("nan")})
    with pytest.raises(ValidationError):
        create_native_split_ledger(invalid_config)
    with pytest.raises(ValidationError):
        invalid_config.canonical_bytes()
    with pytest.raises(ValidationError):
        _ = invalid_config.digest

    ledger = extend_native_split_ledger(create_native_split_ledger(config), (1,))
    wrong = "train" if ledger.assignments[0].split == "validation" else "validation"
    invalid_ledger = ledger.model_copy(
        update={"assignments": (NativeSeedAssignment.model_construct(world_seed=1, split=wrong),)}
    )
    with pytest.raises(ValidationError, match="does not match"):
        extend_native_split_ledger(invalid_ledger, (2,))
    with pytest.raises(ValidationError, match="does not match"):
        invalid_ledger.canonical_bytes()
    with pytest.raises(ValidationError, match="does not match"):
        _ = invalid_ledger.digest
    with pytest.raises(ValidationError, match="does not match"):
        invalid_ledger.split_for_seed(1)
