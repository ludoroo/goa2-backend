"""Native Gen1 policy/value record and one-game stream contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import zstandard
from pydantic import ValidationError

from automata.decision import DecisionSemanticRole
from automata.models.contracts import (
    DecisionObservation,
    EncodedCandidate,
    LearnedObservation,
    ObservationToken,
    OptionCandidateID,
    StableValueObservation,
    Viewer,
    canonical_json_bytes,
)
from automata.training.native_dataset import (
    NativeBoundaryProvenance,
    NativeGameIdentity,
    PolicyDatasetRecord,
    ValueDatasetRecord,
    iter_native_game_records,
    native_game_id,
    native_sample_id,
    publish_native_game,
)
from automata.training.search_targets import SearchActionTarget, SearchPolicyTarget


def _identity() -> NativeGameIdentity:
    values = {
        "world_seed": 41,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": "generation-1",
        "source_revision": "abc123",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "search-1",
        "generator_config_id": "generator-1",
    }
    return NativeGameIdentity(game_id=native_game_id(**values), **values)


def _candidate(name: str) -> EncodedCandidate:
    return EncodedCandidate(
        schema_version=1,
        candidate_id=OptionCandidateID(schema_version=1, option_id=name),
        selection=name,
    )


def _state(*, actor: bool, perspective: str = "RED") -> LearnedObservation:
    return LearnedObservation(
        schema_version=2,
        viewer=Viewer(
            schema_version=2,
            private_hero_id="hero_wasp",
            perspective_team=perspective,
        ),
        tokens=(
            ObservationToken(
                schema_version=1,
                local_ref="global:0",
                kind="GLOBAL",
                features={
                    "map_id": "forgotten_island",
                    "game_type": "QUICK",
                    "round": 2,
                    "turn": 3,
                },
            ),
            ObservationToken(
                schema_version=1,
                local_ref="hero:hero_wasp",
                kind="HERO",
                features={
                    "hero_id": "hero_wasp",
                    "name": "Wasp",
                    "team_id": "RED",
                    "relation": "SELF",
                    "is_current_actor": actor,
                    "is_decision_owner": actor,
                },
            ),
            ObservationToken(
                schema_version=1,
                local_ref="hero:hero_arien",
                kind="HERO",
                features={
                    "hero_id": "hero_arien",
                    "name": "Arien",
                    "team_id": "BLUE",
                    "relation": "ENEMY",
                    "is_current_actor": False,
                    "is_decision_owner": False,
                },
            ),
        ),
    )


def _policy(*, sample_index: int = 0, policy_index: int = 0) -> PolicyDatasetRecord:
    candidates = (_candidate("hold"), _candidate("advance"))
    observation = DecisionObservation(
        schema_version=4,
        state=_state(actor=True),
        decision_kind="INPUT",
        input_request_type="SELECT_OPTION",
        can_skip=False,
        semantic_role=DecisionSemanticRole.OPTION_SELECTION,
        candidates=candidates,
    )
    target = SearchPolicyTarget(
        schema_version=1,
        actions=(
            SearchActionTarget(
                schema_version=1,
                candidate=candidates[0],
                prior_probability=0.25,
                sample_count=3,
                mean_value=0.75,
                value_variance=0.125,
                improved_probability=0.75,
                selected=True,
            ),
            SearchActionTarget(
                schema_version=1,
                candidate=candidates[1],
                prior_probability=0.75,
                sample_count=1,
                mean_value=0.25,
                value_variance=0.0,
                improved_probability=0.25,
                selected=False,
            ),
        ),
    )
    game = _identity()
    return PolicyDatasetRecord(
        game=game,
        sample_id=native_sample_id(
            game_id=game.game_id, sample_kind="POLICY", sample_index=sample_index
        ),
        sample_index=sample_index,
        policy_index=policy_index,
        perspective_team="RED",
        observation=observation,
        target=target,
    )


def _value(
    *,
    sample_index: int = 0,
    boundary_index: int = 0,
    kind: str = "ACTOR_READY",
    terminal_winner: str | None = "RED",
    value_target: int = 1,
) -> ValueDatasetRecord:
    actor = kind == "ACTOR_READY"
    game = _identity()
    return ValueDatasetRecord(
        game=game,
        sample_id=native_sample_id(
            game_id=game.game_id, sample_kind="VALUE", sample_index=sample_index
        ),
        sample_index=sample_index,
        perspective_team="RED",
        boundary=NativeBoundaryProvenance(
            boundary_index=boundary_index,
            kind=kind,
            round=2,
            turn=3,
            viewer_ref="hero:hero_wasp",
            actor_ref="hero:hero_wasp" if actor else None,
        ),
        observation=StableValueObservation(
            schema_version=1,
            state=_state(actor=actor),
            boundary_kind=kind,
        ),
        terminal_winner=terminal_winner,
        value_target=value_target,
    )


def _replace(model: Any, **changes: Any) -> dict[str, Any]:
    payload = model.model_dump(mode="python")
    payload.update(changes)
    return payload


def test_native_ids_are_namespaced_deterministic_and_identity_is_self_validating() -> None:
    identity = _identity()
    assert len(identity.game_id) == 64
    assert identity.game_id == native_game_id(
        **identity.model_dump(exclude={"game_id"}, mode="python")
    )
    assert native_sample_id(
        game_id=identity.game_id, sample_kind="POLICY", sample_index=0
    ) != native_sample_id(game_id=identity.game_id, sample_kind="VALUE", sample_index=0)

    with pytest.raises(ValidationError, match="game_id"):
        NativeGameIdentity(**_replace(identity, game_id="0" * 64))
    with pytest.raises(ValidationError):
        NativeGameIdentity(**_replace(identity, private_seed_material="secret"))


def test_policy_record_preserves_actual_priors_and_validates_search_evidence() -> None:
    row = _policy()
    restored = PolicyDatasetRecord.model_validate_json(canonical_json_bytes(row), strict=True)

    assert restored.target.actions[0].prior_probability == 0.25
    assert (
        tuple(action.candidate for action in restored.target.actions) == row.observation.candidates
    )

    actions = list(row.target.actions)
    actions[0] = actions[0].model_copy(update={"improved_probability": 0.6})
    actions[1] = actions[1].model_copy(update={"improved_probability": 0.4})
    with pytest.raises(ValidationError, match="visits"):
        PolicyDatasetRecord(**_replace(row, target={"actions": actions}))

    actions = list(row.target.actions)
    actions[1] = actions[1].model_copy(update={"mean_value": -0.1})
    with pytest.raises(ValidationError, match="reward"):
        PolicyDatasetRecord(**_replace(row, target={"actions": actions}))

    actions = list(row.target.actions)
    actions[1] = actions[1].model_copy(
        update={"sample_count": 0, "mean_value": 0.1, "value_variance": 0.0}
    )
    actions[0] = actions[0].model_copy(update={"improved_probability": 1.0})
    actions[1] = actions[1].model_copy(update={"improved_probability": 0.0})
    with pytest.raises(ValidationError, match="unvisited"):
        PolicyDatasetRecord(**_replace(row, target={"actions": actions}))


@pytest.mark.parametrize("invalid", ["opposite-team", "other-private-hero", "no-owner"])
def test_policy_viewer_must_be_its_decision_owner_on_the_declared_team(invalid: str) -> None:
    payload = _policy().model_dump(mode="python")
    graph = payload["observation"]["state"]
    if invalid == "opposite-team":
        payload["perspective_team"] = graph["viewer"]["perspective_team"] = "BLUE"
    elif invalid == "other-private-hero":
        graph["viewer"]["private_hero_id"] = "hero_arien"
    else:
        for token in graph["tokens"]:
            if token["kind"] == "HERO":
                token["features"]["is_decision_owner"] = False

    with pytest.raises(ValueError, match="policy viewer"):
        PolicyDatasetRecord.model_validate(payload)


def test_public_hero_ids_must_be_defined_and_consistent_across_a_game(tmp_path: Path) -> None:
    payload = _value(sample_index=1, boundary_index=1).model_dump(mode="python")
    graph = payload["observation"]["state"]
    graph["viewer"]["private_hero_id"] = "different_hero_wasp"
    for token in graph["tokens"]:
        if token["kind"] == "HERO" and token["features"]["relation"] == "SELF":
            token["features"]["hero_id"] = "different_hero_wasp"
    different = ValueDatasetRecord.model_validate(payload)
    path = tmp_path / "different-roster.jsonl"

    with pytest.raises(ValueError, match="consistent public hero roster"):
        publish_native_game(path, (_value(), different))
    assert not path.exists()

    graph["viewer"]["private_hero_id"] = None
    for token in graph["tokens"]:
        if token["kind"] == "HERO":
            token["features"].pop("hero_id", None)
    with pytest.raises(ValueError, match="hero_id"):
        ValueDatasetRecord.model_validate(payload)


def test_policy_allows_only_singleton_zero_visit_target_with_probability_one() -> None:
    row = _policy()
    candidate = row.observation.candidates[0]
    observation = row.observation.model_copy(update={"candidates": (candidate,)})
    target = SearchPolicyTarget(
        actions=(
            SearchActionTarget(
                candidate=candidate,
                sample_count=0,
                mean_value=0.0,
                value_variance=0.0,
                improved_probability=1.0,
                selected=True,
            ),
        )
    )

    accepted = PolicyDatasetRecord(**_replace(row, observation=observation, target=target))
    assert accepted.target.actions[0].improved_probability == 1.0


def test_value_record_validates_boundary_viewer_actor_and_outcome_orientation() -> None:
    row = _value()
    assert row.boundary.viewer_ref == "hero:hero_wasp"

    with pytest.raises(ValidationError, match="perspective"):
        ValueDatasetRecord(**_replace(row, perspective_team="BLUE"))
    with pytest.raises(ValidationError, match="actor"):
        ValueDatasetRecord(
            **_replace(
                row,
                boundary=row.boundary.model_copy(update={"actor_ref": "hero:hero_arien"}),
            )
        )
    with pytest.raises(ValidationError, match="value target"):
        ValueDatasetRecord(**_replace(row, value_target=-1))

    draw = _value(terminal_winner=None, value_target=0)
    assert draw.value_target == 0


def test_discriminated_records_reject_wrong_row_kind_and_cross_kind_fields(tmp_path: Path) -> None:
    payload = _policy().model_dump(mode="json")
    payload["sample_kind"] = "VALUE"
    path = tmp_path / "wrong.jsonl"
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")

    with pytest.raises(ValueError, match=r"invalid native record|VALUE"):
        tuple(iter_native_game_records(path))


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.zst"])
def test_publish_and_iterate_one_canonical_game_streaming(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"game{suffix}"
    records = (_policy(sample_index=0, policy_index=0), _value(sample_index=1))

    publish_native_game(path, iter(records))

    assert tuple(iter_native_game_records(path)) == records
    if suffix == ".jsonl":
        assert path.read_bytes() == b"".join(canonical_json_bytes(row) + b"\n" for row in records)
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        ("blank", b"\n", "blank"),
        ("crlf", canonical_json_bytes(_policy()) + b"\r\n", "whitespace"),
        ("space", b" " + canonical_json_bytes(_policy()) + b"\n", "whitespace"),
        (
            "noncanonical",
            json.dumps(_policy().model_dump(mode="json")).encode() + b"\n",
            "canonical",
        ),
        ("truncated", canonical_json_bytes(_policy()), "truncated"),
        ("malformed", b'{"sample_kind":"POLICY"\n', "invalid"),
    ],
)
def test_reader_rejects_noncanonical_and_corrupt_plain_jsonl(
    tmp_path: Path, name: str, payload: bytes, message: str
) -> None:
    path = tmp_path / f"{name}.jsonl"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        tuple(iter_native_game_records(path))


def test_reader_rejects_truncated_corrupt_and_trailing_zstd_frames(tmp_path: Path) -> None:
    canonical = canonical_json_bytes(_policy()) + b"\n"
    compressor = zstandard.ZstdCompressor()
    cases = {
        "truncated": compressor.compress(canonical)[:-1],
        "corrupt": b"not-zstd",
        "trailing": compressor.compress(canonical) + compressor.compress(canonical),
    }
    for name, payload in cases.items():
        path = tmp_path / f"{name}.jsonl.zst"
        path.write_bytes(payload)
        with pytest.raises(ValueError, match=r"compressed|zstd|trailing|truncated"):
            tuple(iter_native_game_records(path))


def test_one_game_validation_rejects_indexes_identities_boundaries_and_viewer_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.jsonl"
    cases = (
        (_policy(sample_index=1),),
        (_policy(sample_index=0, policy_index=0), _policy(sample_index=1, policy_index=0)),
        (_value(sample_index=0, boundary_index=0), _value(sample_index=1, boundary_index=0)),
        (_value(sample_index=0, boundary_index=1),),
    )
    for records in cases:
        path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in records))
        with pytest.raises(ValueError, match=r"contiguous|duplicate|boundary|index"):
            tuple(iter_native_game_records(path))

    other_game = _identity().model_copy(update={"generation_id": "other"})
    other_game = NativeGameIdentity(
        **_replace(
            other_game,
            game_id=native_game_id(**other_game.model_dump(exclude={"game_id"}, mode="python")),
        )
    )
    second = _policy(sample_index=1, policy_index=1).model_copy(
        update={
            "game": other_game,
            "sample_id": native_sample_id(
                game_id=other_game.game_id, sample_kind="POLICY", sample_index=1
            ),
        }
    )
    path.write_bytes(canonical_json_bytes(_policy()) + b"\n" + canonical_json_bytes(second) + b"\n")
    with pytest.raises(ValueError, match=r"one game|identity"):
        tuple(iter_native_game_records(path))


def _remap_hero_refs(row: ValueDatasetRecord, refs: dict[str, str]) -> ValueDatasetRecord:
    payload = row.model_dump(mode="python")
    boundary = payload["boundary"]
    boundary["viewer_ref"] = refs[boundary["viewer_ref"]]
    if boundary["actor_ref"] is not None:
        boundary["actor_ref"] = refs[boundary["actor_ref"]]
    for token in payload["observation"]["state"]["tokens"]:
        token["local_ref"] = refs.get(token["local_ref"], token["local_ref"])
    return ValueDatasetRecord.model_validate(payload)


def test_boundary_group_resolves_observation_local_refs_before_comparing_viewers(
    tmp_path: Path,
) -> None:
    red = _remap_hero_refs(_value(), {"hero:hero_wasp": "self", "hero:hero_arien": "other"})
    payload = _value(sample_index=1).model_dump(mode="python")
    payload["perspective_team"] = "BLUE"
    payload["value_target"] = -1
    payload["boundary"]["viewer_ref"] = "hero:hero_arien"
    graph = payload["observation"]["state"]
    graph["viewer"]["private_hero_id"] = "hero_arien"
    graph["viewer"]["perspective_team"] = "BLUE"
    for token in graph["tokens"]:
        if token["kind"] == "HERO":
            token["features"]["relation"] = (
                "SELF" if token["features"]["hero_id"] == "hero_arien" else "ENEMY"
            )
    blue = _remap_hero_refs(
        ValueDatasetRecord.model_validate(payload),
        {"hero:hero_wasp": "other", "hero:hero_arien": "self"},
    )
    path = tmp_path / "local-refs.jsonl"

    publish_native_game(path, (red, blue))

    assert tuple(iter_native_game_records(path)) == (red, blue)


def test_alias_changes_cannot_duplicate_a_viewer_within_a_boundary(tmp_path: Path) -> None:
    first = _value(kind="PLANNING_READY")
    repeated = _remap_hero_refs(
        _value(kind="PLANNING_READY", sample_index=1),
        {"hero:hero_wasp": "different-local-name", "hero:hero_arien": "opponent"},
    )
    path = tmp_path / "duplicated-viewer.jsonl"

    with pytest.raises(ValueError, match="duplicate viewer"):
        publish_native_game(path, (first, repeated))

    assert not path.exists()


def test_reader_rejects_inconsistent_terminal_winner_and_interrupted_boundary_group(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.jsonl"
    first = _value(sample_index=0, boundary_index=0)
    contradictory = _value(
        sample_index=1, boundary_index=1, terminal_winner="BLUE", value_target=-1
    )
    path.write_bytes(
        canonical_json_bytes(first) + b"\n" + canonical_json_bytes(contradictory) + b"\n"
    )
    with pytest.raises(ValueError, match="terminal winner"):
        tuple(iter_native_game_records(path))

    middle = _policy(sample_index=1, policy_index=0)
    repeated = _value(sample_index=2, boundary_index=0)
    path.write_bytes(
        canonical_json_bytes(first)
        + b"\n"
        + canonical_json_bytes(middle)
        + b"\n"
        + canonical_json_bytes(repeated)
        + b"\n"
    )
    with pytest.raises(ValueError, match="boundary"):
        tuple(iter_native_game_records(path))


def test_publish_revalidates_constructed_models_and_never_replaces_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "game.jsonl"
    invalid = _policy().model_copy(update={"sample_id": "0" * 64})
    with pytest.raises(ValueError, match="sample_id"):
        publish_native_game(destination, (invalid,))
    assert not destination.exists()

    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        publish_native_game(destination, (_policy(),))
    assert destination.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".*.tmp"))


def test_publish_destination_race_preserves_competing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "race.jsonl"
    real_link = os.link

    def race_link(
        source: str | bytes | Path, target: str | bytes | Path, *args: Any, **kwargs: Any
    ) -> None:
        Path(target).write_bytes(b"competitor")
        real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", race_link)

    with pytest.raises(FileExistsError):
        publish_native_game(destination, (_policy(),))
    assert destination.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".*.tmp"))
