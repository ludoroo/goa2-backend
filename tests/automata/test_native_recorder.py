"""Whole-game native recording at the public policy and stable-boundary seams.

The real setup states exercise the value encoder and boundary detector together;
no synthetic ``GameStep`` is needed for these recorder contract tests.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import zstandard

from automata.decision import DecisionDescriptor, DecisionSemanticRole
from automata.harness.game_runner import DEFAULT_MAP
from automata.models.contracts import (
    DecisionObservation,
    EncodedCandidate,
    LearnedObservation,
    ObservationToken,
    OptionCandidateID,
    Viewer,
)
from automata.observation import encode_decision
from automata.runtime.effects import register_all_effects
from automata.runtime.value_boundary import (
    StableValueBoundaryKind,
    detect_stable_value_boundary,
)
from automata.training.native_dataset import (
    NativeGameIdentity,
    PolicyDatasetRecord,
    ValueDatasetRecord,
    iter_native_game_records,
    native_game_id,
)
from automata.training.native_recorder import NativeDatasetRecorder
from automata.training.search_targets import SearchActionTarget, SearchPolicyTarget
from goa2.domain.models import CardState, GamePhase
from goa2.domain.types import HeroID
from goa2.engine.phases import resolve_next_action
from goa2.engine.setup import GameSetup


def _game_identity(**changes: Any) -> NativeGameIdentity:
    fields: dict[str, Any] = {
        "world_seed": 73,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": "generation-1",
        "source_revision": "abc123",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "search-config-1",
        "generator_config_id": "generator-config-1",
    }
    fields.update(changes)
    return NativeGameIdentity(game_id=native_game_id(**fields), **fields)


def _state():
    register_all_effects()
    return GameSetup.create_game(
        DEFAULT_MAP,
        ["Wasp"],
        ["Arien"],
        game_type="QUICK",
        seed=73,
    )


def _actor_ready_state():
    state = _state()
    actor = state.get_hero(HeroID("hero_wasp"))
    assert actor is not None and actor.hand
    card = actor.hand.pop()
    card.state = CardState.UNRESOLVED
    card.is_facedown = False
    actor.current_turn_card = card
    state.phase = GamePhase.RESOLUTION
    state.unresolved_hero_ids = [actor.id]
    resolve_next_action(state)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    assert boundary.kind is StableValueBoundaryKind.ACTOR_READY
    return state, boundary


def _candidate(name: str) -> EncodedCandidate:
    return EncodedCandidate(
        schema_version=1,
        candidate_id=OptionCandidateID(schema_version=1, option_id=name),
        selection=name,
        features={"fixture_score": 1.0},
    )


def _policy() -> tuple[DecisionObservation, SearchPolicyTarget]:
    candidates = (_candidate("hold"), _candidate("advance"))
    observation = DecisionObservation(
        schema_version=4,
        state=LearnedObservation(
            schema_version=2,
            viewer=Viewer(
                schema_version=2,
                private_hero_id="hero_wasp",
                perspective_team="RED",
            ),
            tokens=(
                ObservationToken(
                    schema_version=1,
                    local_ref="global:0",
                    kind="GLOBAL",
                    features={
                        "map_id": "forgotten_island",
                        "game_type": "QUICK",
                        "round": 1,
                        "turn": 1,
                    },
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:self",
                    kind="HERO",
                    features={
                        "hero_id": "hero_wasp",
                        "name": "Wasp",
                        "is_decision_owner": True,
                        "team_id": "RED",
                        "relation": "SELF",
                    },
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:enemy",
                    kind="HERO",
                    features={
                        "hero_id": "hero_arien",
                        "name": "Arien",
                        "team_id": "BLUE",
                        "relation": "ENEMY",
                    },
                ),
            ),
        ),
        decision_kind="INPUT",
        input_request_type="SELECT_OPTION",
        can_skip=False,
        semantic_role=DecisionSemanticRole.OPTION_SELECTION,
        candidates=candidates,
    )
    target = SearchPolicyTarget(
        actions=tuple(
            SearchActionTarget(
                candidate=candidate,
                sample_count=visits,
                mean_value=mean,
                value_variance=variance,
                improved_probability=visits / 4,
                selected=index == 0,
            )
            for index, (candidate, visits, mean, variance) in enumerate(
                zip(candidates, (3, 1), (0.75, 0.25), (0.1, 0.0), strict=True)
            )
        )
    )
    return observation, target


def _record_policy(recorder: NativeDatasetRecorder) -> None:
    observation, target = _policy()
    recorder.record_policy(observation=observation, target=target, perspective_team="RED")


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.zst"])
def test_terminal_game_publishes_policy_and_value_rows_only_at_completion(
    tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"native{suffix}"
    state = _state()
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    recorder = NativeDatasetRecorder(path, game=_game_identity())

    _record_policy(recorder)
    recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))

    assert not path.exists()
    assert len(list(tmp_path.iterdir())) == 1
    assert not list(tmp_path.glob("*.jsonl"))
    assert not list(tmp_path.glob("*.jsonl.zst"))
    recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    rows = tuple(iter_native_game_records(path))
    assert [type(row) for row in rows] == [PolicyDatasetRecord, ValueDatasetRecord]
    assert [row.sample_index for row in rows] == [0, 1]
    policy, value = rows
    assert policy.policy_index == 0
    assert value.boundary.boundary_index == 0
    assert value.boundary.viewer_ref == "hero:hero_wasp"
    assert value.value_target == 1
    assert value.terminal_winner == "RED"
    assert list(tmp_path.iterdir()) == [path]
    recorder.close()  # idempotent after successful completion


def test_actor_boundary_records_each_private_view_with_its_own_orientation(
    tmp_path: Path,
) -> None:
    state, boundary = _actor_ready_state()
    path = tmp_path / "views.jsonl"
    recorder = NativeDatasetRecorder(path, game=_game_identity())

    recorder.record_boundary(
        state,
        boundary,
        viewer_hero_ids=("hero_arien", "hero_wasp"),
    )
    recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    rows = tuple(iter_native_game_records(path))
    assert all(isinstance(row, ValueDatasetRecord) for row in rows)
    assert [row.perspective_team for row in rows] == ["BLUE", "RED"]
    assert [row.value_target for row in rows] == [-1, 1]
    assert {row.boundary.boundary_index for row in rows} == {0}
    assert [row.boundary.viewer_ref for row in rows] == [
        "hero:hero_arien",
        "hero:hero_wasp",
    ]
    assert {row.boundary.actor_ref for row in rows} == {"hero:hero_wasp"}


def test_multiple_heroes_keep_declared_roster_order_and_individual_private_views(
    tmp_path: Path,
) -> None:
    register_all_effects()
    red, blue = ("Wasp", "Brogan"), ("Xargatha", "Arien")
    state = GameSetup.create_game(DEFAULT_MAP, list(red), list(blue), game_type="QUICK", seed=73)
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    path = tmp_path / "rosters.jsonl"
    viewers = tuple(sorted(str(hero.id) for team in state.teams.values() for hero in team.heroes))
    game = _game_identity(red_composition=red, blue_composition=blue)
    assert (
        game.game_id
        != _game_identity(red_composition=tuple(reversed(red)), blue_composition=blue).game_id
    )
    owner = state.get_hero(HeroID("hero_wasp"))
    observation = encode_decision(
        state,
        DecisionDescriptor("CARD", hero=owner),
        tuple(card.id for card in owner.hand),
        decision_owner_hero_id=owner.id,
        perspective_team="RED",
    )
    target = SearchPolicyTarget(
        actions=tuple(
            SearchActionTarget(
                candidate=candidate,
                sample_count=1,
                mean_value=0.0,
                value_variance=0.0,
                improved_probability=1 / len(observation.candidates),
                selected=index == 0,
            )
            for index, candidate in enumerate(observation.candidates)
        )
    )
    with NativeDatasetRecorder(path, game=game) as recorder:
        recorder.record_policy(observation=observation, target=target, perspective_team="RED")
        recorder.record_boundary(state, boundary, viewer_hero_ids=viewers)
        recorder.record_outcome(winner_side="BLUE", rounds=1, reason="game_over")

    policy, *values = tuple(iter_native_game_records(path))
    assert policy.observation == observation
    assert policy.game.red_composition == red
    assert policy.game.blue_composition == blue
    assert len(values) == 4
    assert tuple(row.observation.state.viewer.private_hero_id for row in values) == viewers
    assert [row.value_target for row in values] == [1, -1, -1, 1]


def test_recorded_snapshots_do_not_follow_later_input_or_state_mutation(tmp_path: Path) -> None:
    path = tmp_path / "frozen.jsonl"
    recorder = NativeDatasetRecorder(path, game=_game_identity())
    observation, target = _policy()
    state = _state()
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None

    recorder.record_policy(observation=observation, target=target, perspective_team="RED")
    recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))
    observation.state.tokens[0].features["map_id"] = "mutated"
    target.actions[0].candidate.features["fixture_score"] = 99.0
    state.round = 99
    recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    policy, value = tuple(iter_native_game_records(path))
    global_token = next(
        token for token in policy.observation.state.tokens if token.kind == "GLOBAL"
    )
    assert global_token.features["map_id"] == "forgotten_island"
    assert policy.target.actions[0].candidate.features["fixture_score"] == 1.0
    value_global = next(token for token in value.observation.state.tokens if token.kind == "GLOBAL")
    assert value_global.features["round"] == 1


@pytest.mark.parametrize("case", ["unsorted", "duplicate", "unknown", "stale", "identity"])
def test_invalid_boundary_callbacks_poison_and_destroy_the_game_spool(
    tmp_path: Path, case: str
) -> None:
    state, boundary = _actor_ready_state()
    viewers = ("hero_arien", "hero_wasp")
    game = _game_identity()
    if case == "unsorted":
        viewers = tuple(reversed(viewers))
    elif case == "duplicate":
        viewers = ("hero_wasp", "hero_wasp")
    elif case == "unknown":
        viewers = ("hero_arien", "hero_missing")
    elif case == "stale":
        boundary = replace(boundary, turn=boundary.turn + 1)
    else:
        game = _game_identity(map_id="wrong_map")
    path = tmp_path / f"{case}.jsonl"
    recorder = NativeDatasetRecorder(path, game=game)

    with pytest.raises(ValueError):
        recorder.record_boundary(state, boundary, viewer_hero_ids=viewers)

    assert not path.exists()
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="closed"):
        _record_policy(recorder)


def test_empty_viewer_boundary_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    state = _state()
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    recorder = NativeDatasetRecorder(path, game=_game_identity())

    recorder.record_boundary(state, boundary, viewer_hero_ids=())
    recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    assert not path.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("winner", "reason"),
    [(None, "game_over"), ("hero_wasp", "game_over"), ("RED", "max_steps")],
)
def test_malformed_outcomes_raise_and_discard_all_pending_rows(
    tmp_path: Path, winner: str | None, reason: str
) -> None:
    path = tmp_path / "invalid-outcome.jsonl"
    recorder = NativeDatasetRecorder(path, game=_game_identity())
    _record_policy(recorder)

    with pytest.raises(ValueError):
        recorder.record_outcome(winner_side=winner, rounds=2, reason=reason)  # type: ignore[arg-type]

    assert not path.exists()
    assert not list(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="closed"):
        recorder.record_outcome(winner_side="RED", rounds=2, reason="game_over")


@pytest.mark.parametrize("reason", ["max_steps", "timeout", "exception", "interruption"])
def test_censored_outcome_and_unfinished_close_discard_without_publication(
    tmp_path: Path, reason: str
) -> None:
    path = tmp_path / f"{reason}.jsonl"
    with NativeDatasetRecorder(path, game=_game_identity()) as recorder:
        _record_policy(recorder)
        recorder.record_outcome(winner_side=None, rounds=2, reason=reason)

    assert not path.exists()
    assert not list(tmp_path.iterdir())

    unfinished = tmp_path / f"unfinished-{reason}.jsonl"
    with NativeDatasetRecorder(unfinished, game=_game_identity()) as recorder:
        _record_policy(recorder)
    assert not unfinished.exists()


def test_existing_destination_and_publication_race_preserve_existing_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing.jsonl"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        NativeDatasetRecorder(path, game=_game_identity())
    assert path.read_bytes() == b"existing"

    race = tmp_path / "race.jsonl"
    recorder = NativeDatasetRecorder(race, game=_game_identity())
    _record_policy(recorder)
    race.write_bytes(b"racing writer")
    with pytest.raises(FileExistsError):
        recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")
    assert race.read_bytes() == b"racing writer"
    assert set(tmp_path.iterdir()) == {path, race}


@pytest.mark.parametrize("corruption", ["truncated_frame", "missing_rows"])
def test_corrupted_or_incomplete_spool_cannot_publish_a_completed_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    import automata.training.native_recorder as recorder_module

    path = tmp_path / "corrupt.jsonl.zst"
    recorder = NativeDatasetRecorder(path, game=_game_identity())
    _record_policy(recorder)
    state = _state()
    boundary = detect_stable_value_boundary(state)
    assert boundary is not None
    recorder.record_boundary(state, boundary, viewer_hero_ids=("hero_wasp",))
    real_publish = recorder_module.publish_native_game

    def corrupt_then_publish(destination, records):
        spool = next(tmp_path.iterdir())
        if corruption == "truncated_frame":
            spool.write_bytes(spool.read_bytes()[:-1])
        else:
            with (
                spool.open("rb") as source,
                zstandard.ZstdDecompressor().stream_reader(source) as stream,
            ):
                first = stream.read().splitlines(keepends=True)[0]
            spool.write_bytes(zstandard.ZstdCompressor().compress(first))
        real_publish(destination, records)

    monkeypatch.setattr(recorder_module, "publish_native_game", corrupt_then_publish)

    with pytest.raises(ValueError, match=r"truncated|sample count"):
        recorder.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    assert list(tmp_path.iterdir()) == []
    with pytest.raises(RuntimeError, match="closed"):
        _record_policy(recorder)


def test_identical_games_publish_deterministic_plain_and_compressed_bytes(
    tmp_path: Path,
) -> None:
    for suffix in (".jsonl", ".jsonl.zst"):
        paths = (tmp_path / f"one{suffix}", tmp_path / f"two{suffix}")
        for path in paths:
            recorder = NativeDatasetRecorder(path, game=_game_identity())
            _record_policy(recorder)
            recorder.record_outcome(winner_side="BLUE", rounds=2, reason="game_over")
        assert paths[0].read_bytes() == paths[1].read_bytes()
        assert tuple(iter_native_game_records(paths[0])) == tuple(
            iter_native_game_records(paths[1])
        )
