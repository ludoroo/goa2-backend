"""Regression coverage for preserving search priors as metric evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from automata.decision import DecisionSemanticRole
from automata.models.contracts import (
    DecisionObservation,
    EncodedCandidate,
    LearnedObservation,
    ObservationToken,
    OptionCandidateID,
    Viewer,
)
from automata.search.ismcts.engine import RootActionDiagnostic, SearchResult
from automata.search.ismcts.strategy import StrategyResult
from automata.search.node import Node
from automata.training.dataset import JointDatasetRow, joint_decision_id, write_joint_dataset
from automata.training.generation import _aligned_action_stats
from automata.training.indexed_dataset import IndexedDatasetManifest, open_indexed_dataset
from automata.training.metrics import PolicyMetricInput, policy_metrics

_SEARCH_FAVORITE_VARIANCE = 3.9375 / 9 - (5.75 / 9) ** 2


def _candidates(*names: str) -> tuple[EncodedCandidate, ...]:
    return tuple(
        EncodedCandidate(
            schema_version=1,
            candidate_id=OptionCandidateID(schema_version=1, option_id=name),
            selection=name,
        )
        for name in names
    )


def _search_result(
    diagnostics: tuple[RootActionDiagnostic, ...],
) -> StrategyResult[str]:
    root = Node(
        visits=10,
        total_value=6.0,
        total_squared_value=4.0,
        children={
            "prior-favorite": Node(1, 0.25, 0.0625),
            "search-favorite": Node(9, 5.75, 3.9375),
        },
    )
    return StrategyResult(
        candidates=("prior-favorite", "search-favorite", "unvisited"),
        selected_index=1,
        search_result=SearchResult(
            root=root,
            best_key="search-favorite",
            root_action_diagnostics=diagnostics,
        ),
    )


def _complete_diagnostics() -> tuple[RootActionDiagnostic, ...]:
    # Deliberately not in caller candidate order: alignment is by action key.
    return (
        RootActionDiagnostic("unvisited", 0.1, 0, 0.0, 0.0),
        RootActionDiagnostic("search-favorite", 0.1, 9, 5.75 / 9, _SEARCH_FAVORITE_VARIANCE),
        RootActionDiagnostic("prior-favorite", 0.8, 1, 0.25, 0.0),
    )


def _observation(candidates: tuple[EncodedCandidate, ...]) -> DecisionObservation:
    return DecisionObservation(
        schema_version=4,
        state=LearnedObservation(
            schema_version=2,
            viewer=Viewer(schema_version=2, perspective_team="RED"),
            tokens=(
                ObservationToken(
                    schema_version=1,
                    local_ref="global:0",
                    kind="GLOBAL",
                    features={"map_id": "forgotten_island", "game_type": "QUICK"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="team:red",
                    kind="TEAM",
                    features={"team_id": "RED", "relation": "OWN"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="team:blue",
                    kind="TEAM",
                    features={"team_id": "BLUE", "relation": "ENEMY"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:red",
                    kind="HERO",
                    features={"name": "Wasp", "team_id": "RED", "team_ref": "team:red"},
                ),
                ObservationToken(
                    schema_version=1,
                    local_ref="hero:blue",
                    kind="HERO",
                    features={"name": "Arien", "team_id": "BLUE", "team_ref": "team:blue"},
                ),
            ),
        ),
        decision_kind="INPUT",
        input_request_type="SELECT_OPTION",
        can_skip=False,
        semantic_role=DecisionSemanticRole.OPTION_SELECTION,
        candidates=candidates,
    )


def _row(action_stats: tuple[Any, ...]) -> JointDatasetRow:
    observation = _observation(tuple(item.candidate for item in action_stats))
    identity: dict[str, Any] = {
        "game_id": "game-priors",
        "world_seed": 7,
        "map_id": "forgotten_island",
        "game_type": "QUICK",
        "red_composition": ("Wasp",),
        "blue_composition": ("Arien",),
        "generation_id": "generation-1",
        "source_revision": "revision",
        "dirty_tree_hash": "clean",
        "source_model_digest": None,
        "search_config_id": "search-1",
        "generator_config_id": "generator-1",
    }
    return JointDatasetRow(
        schema_version=2,
        decision_id=joint_decision_id(**identity, decision_index=0),
        **identity,
        decision_index=0,
        perspective_team="RED",
        observation=observation,
        policy_source="ISMCTS_VISITS",
        policy_target=tuple(float(item.improved_probability) for item in action_stats),
        selected_candidate_id=observation.candidates[1].candidate_id,
        selected_selection=observation.candidates[1].selection,
        action_stats=action_stats,
        terminal_winner="RED",
        value_target=1,
    )


def test_generation_and_index_preserve_true_priors_including_unvisited_actions(
    tmp_path: Path,
) -> None:
    candidates = _candidates("prior-favorite", "search-favorite", "unvisited")
    actions = _aligned_action_stats(_search_result(_complete_diagnostics()), candidates)

    assert tuple(item.sample_count for item in actions) == (1, 9, 0)
    assert tuple(item.improved_probability for item in actions) == (0.1, 0.9, 0.0)
    assert tuple(item.prior_probability for item in actions) == (0.8, 0.1, 0.1)

    source = tmp_path / "joint.jsonl"
    write_joint_dataset(source, (_row(actions),))
    indexed = open_indexed_dataset(source, tmp_path / "index")
    metadata = next(indexed.iter_game_training_chunks("game-priors")).metric_metadata[0]

    assert metadata.prior_probabilities == (0.8, 0.1, 0.1)
    result = policy_metrics(
        [
            PolicyMetricInput(
                game_id="game-priors",
                candidate_family=metadata.candidate_family,
                target_probabilities=metadata.target_probabilities,
                predicted_logits=(0.0, 1.0, 0.0),
                prior_probabilities=metadata.prior_probabilities,
            )
        ]
    )
    assert result["overall"]["search_overturn_rate"] == 1.0


@pytest.mark.parametrize(
    "diagnostics",
    [
        (),
        (
            RootActionDiagnostic("prior-favorite", None, 1, 0.25, 0.0),
            RootActionDiagnostic("search-favorite", None, 9, 5.75 / 9, _SEARCH_FAVORITE_VARIANCE),
            RootActionDiagnostic("unvisited", None, 0, 0.0, 0.0),
        ),
        (RootActionDiagnostic("prior-favorite", 1.0, 1, 0.25, 0.0),),
    ],
)
def test_unavailable_or_partial_root_priors_do_not_fabricate_evidence(
    diagnostics: tuple[RootActionDiagnostic, ...],
    tmp_path: Path,
) -> None:
    actions = _aligned_action_stats(
        _search_result(diagnostics),
        _candidates("prior-favorite", "search-favorite", "unvisited"),
    )

    assert tuple(item.prior_probability for item in actions) == (None, None, None)
    assert tuple(item.improved_probability for item in actions) == (0.1, 0.9, 0.0)
    source = tmp_path / "missing-priors.jsonl"
    write_joint_dataset(source, (_row(actions),))
    indexed = open_indexed_dataset(source, tmp_path / "index")
    metadata = next(indexed.iter_game_training_chunks("game-priors")).metric_metadata[0]
    assert metadata.prior_probabilities is None


def test_inconsistent_root_diagnostics_are_rejected() -> None:
    diagnostics = (
        RootActionDiagnostic("prior-favorite", 0.8, 2, 0.25, 0.0),
        RootActionDiagnostic("search-favorite", 0.1, 9, 5.75 / 9, 0.0),
        RootActionDiagnostic("unvisited", 0.1, 0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="diagnostics disagree"):
        _aligned_action_stats(
            _search_result(diagnostics),
            _candidates("prior-favorite", "search-favorite", "unvisited"),
        )


def test_index_rebuilds_cache_without_metric_metadata_version(tmp_path: Path) -> None:
    actions = _aligned_action_stats(
        _search_result(_complete_diagnostics()),
        _candidates("prior-favorite", "search-favorite", "unvisited"),
    )
    source = tmp_path / "joint.jsonl"
    cache = tmp_path / "index"
    write_joint_dataset(source, (_row(actions),))
    current = open_indexed_dataset(source, cache)
    stale = current.manifest.model_dump(mode="json")
    stale.pop("metric_metadata_version", None)
    with pytest.raises(ValueError, match="metric_metadata_version"):
        IndexedDatasetManifest.model_validate(stale)
    (cache / "manifest.json").write_text(json.dumps(stale, sort_keys=True, separators=(",", ":")))

    rebuilt = open_indexed_dataset(source, cache)

    assert rebuilt.manifest.metric_metadata_version == 2
    metadata = next(rebuilt.iter_game_training_chunks("game-priors")).metric_metadata[0]
    assert metadata.prior_probabilities == (0.8, 0.1, 0.1)
    assert json.loads((cache / "manifest.json").read_text())["metric_metadata_version"] == 2
