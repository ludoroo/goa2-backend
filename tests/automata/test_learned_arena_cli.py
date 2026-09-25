from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from automata.evaluation.learned_matrix import LearnedMatrixCell
from automata.evaluation.protocol import GameCase
from automata.scripts import run_learned_arena as arena
from automata.search.config import parse_learned_lh_search_config
from automata.search.heuristic import HeuristicLeafEvaluator
from automata.search.learned import LearnedLeafEvaluator


@pytest.fixture
def runner(tmp_path: Path) -> arena.LearnedArenaRunner:
    config, _ = parse_learned_lh_search_config({"iterations": 1})
    return arena.LearnedArenaRunner(
        candidate_artifact=tmp_path / "candidate",
        candidate_digest="a" * 64,
        parent_artifact=tmp_path / "parent",
        parent_digest="b" * 64,
        map_path=tmp_path / "map.json",
        map_id="map",
        game_type="QUICK",
        red_heroes=("Wasp", "Xargatha"),
        blue_heroes=("Arien", "Brogan"),
        max_steps=123,
        search_config=config,
        random_stream_namespace="arena-stratum-v1",
    )


@pytest.mark.parametrize("candidate_side", ["RED", "BLUE"])
def test_runner_is_spawn_picklable_and_binds_one_agent_per_side(
    runner: arena.LearnedArenaRunner,
    monkeypatch: pytest.MonkeyPatch,
    candidate_side: str,
) -> None:
    runner = replace(runner, candidate_matrix_cell=LearnedMatrixCell.LL)
    candidate_runtime = object()
    parent_runtime = object()
    loads: list[tuple[Path, str]] = []
    captured: dict[str, Any] = {}

    def fake_load(path: Path, digest: str, **_scope: Any) -> object:
        loads.append((path, digest))
        return candidate_runtime if digest == "a" * 64 else parent_runtime

    def fake_run_game(
        red: list[str], blue: list[str], agents: dict[str, Any], **kwargs: Any
    ) -> Any:
        captured.update(red=red, blue=blue, agents=agents, kwargs=kwargs)
        return SimpleNamespace(winner="RED", rounds=7, steps=91, reason="game_over")

    monkeypatch.setattr(arena, "_load_pinned_runtime", fake_load)
    monkeypatch.setattr(arena, "run_game", fake_run_game)

    pickle.dumps(runner)
    result = runner(GameCase(case_id="case", world_seed=42, a_side=candidate_side))

    assert result.winner_side == "RED"
    assert loads == [
        (runner.candidate_artifact, runner.candidate_digest),
        (runner.parent_artifact, runner.parent_digest),
    ]
    agents = captured["agents"]
    assert agents["hero_wasp"] is agents["hero_xargatha"]
    assert agents["hero_arien"] is agents["hero_brogan"]
    assert agents["hero_wasp"] is not agents["hero_arien"]
    expected_red = candidate_runtime if candidate_side == "RED" else parent_runtime
    expected_blue = candidate_runtime if candidate_side == "BLUE" else parent_runtime
    assert agents["hero_wasp"]._strategy._prior.runtime is expected_red
    assert agents["hero_arien"]._strategy._prior.runtime is expected_blue
    expected_red_leaf = LearnedLeafEvaluator if candidate_side == "RED" else HeuristicLeafEvaluator
    expected_blue_leaf = (
        LearnedLeafEvaluator if candidate_side == "BLUE" else HeuristicLeafEvaluator
    )
    assert isinstance(agents["hero_wasp"]._strategy._leaf_evaluator, expected_red_leaf)
    assert isinstance(agents["hero_arien"]._strategy._leaf_evaluator, expected_blue_leaf)
    assert captured["kwargs"]["max_steps"] == 123
    assert "recorder" not in captured["kwargs"]


def test_candidate_is_paired_on_both_sides(runner: arena.LearnedArenaRunner) -> None:
    protocol = arena.build_protocol(
        runner,
        world_seeds=(9, 10),
        source_revision="revision",
        dirty_tree_hash="tree",
        case_timeout_seconds=60.0,
    )
    assert [(case.world_seed, case.a_side) for case in protocol.cases()] == [
        (9, "RED"),
        (9, "BLUE"),
        (10, "RED"),
        (10, "BLUE"),
    ]
    assert protocol.agent_a.kind == "learned_ismcts_lh"
    assert protocol.agent_b.kind == "learned_ismcts_lh"
    expected_param_keys = {
        "artifact_digest",
        "continuation_policy",
        "map_id",
        "matrix_cell",
        "random_stream_namespace",
        "runtime_identity",
        "search_config",
        "seed_derivation",
        "value_recipe",
    }
    assert set(protocol.agent_a.params) == expected_param_keys
    assert set(protocol.agent_b.params) == expected_param_keys
    assert protocol.agent_a.params["matrix_cell"] == "L/H"
    assert protocol.agent_b.params["matrix_cell"] == "L/H"
    assert protocol.agent_a.params["value_recipe"] == "public-consequence-v4"
    assert protocol.agent_b.params["value_recipe"] == "public-consequence-v4"
    assert protocol.agent_a.params["continuation_policy"] == "learned-argmax-v1"
    assert protocol.agent_b.params["continuation_policy"] == "learned-argmax-v1"


def test_matrix_cells_bind_leaf_evaluators_and_protocol_identity(
    runner: arena.LearnedArenaRunner,
) -> None:
    runtime = object()
    lh = runner._agent(runtime, LearnedMatrixCell.LH, seed=1)
    ll = runner._agent(runtime, LearnedMatrixCell.LL, seed=1)

    assert isinstance(lh._strategy._leaf_evaluator, HeuristicLeafEvaluator)
    assert isinstance(ll._strategy._leaf_evaluator, LearnedLeafEvaluator)

    kwargs = dict(
        world_seeds=(9,),
        source_revision="revision",
        dirty_tree_hash="tree",
        case_timeout_seconds=60.0,
    )
    baseline = arena.build_protocol(runner, **kwargs)
    ablation = arena.build_protocol(
        replace(runner, candidate_matrix_cell=LearnedMatrixCell.LL), **kwargs
    )

    assert baseline.agent_a.params["matrix_cell"] == "L/H"
    assert baseline.agent_a.params["value_recipe"] == "public-consequence-v4"
    assert ablation.agent_a.params["matrix_cell"] == "L/L"
    assert ablation.agent_a.params["value_recipe"] == "learned-value-head-v1"
    assert ablation.agent_b.params["matrix_cell"] == "L/H"
    assert baseline.identity_digest() != ablation.identity_digest()


def test_arena_records_native_runtime_identity(runner: arena.LearnedArenaRunner) -> None:
    runner.parent_artifact.mkdir()
    identity = {
        "observation_schema_version": 4,
        "runtime_compatibility_version": 2,
        "tensor_schema_digest": "c" * 64,
    }
    (runner.parent_artifact / "manifest.json").write_text(json.dumps(identity))
    protocol = arena.build_protocol(
        runner,
        world_seeds=(9,),
        source_revision="revision",
        dirty_tree_hash="tree",
        case_timeout_seconds=60.0,
    )
    assert protocol.agent_b.params["runtime_identity"] == identity

    identity["tensor_schema_digest"] = "d" * 64
    (runner.parent_artifact / "manifest.json").write_text(json.dumps(identity))
    changed = arena.build_protocol(
        runner,
        world_seeds=(9,),
        source_revision="revision",
        dirty_tree_hash="tree",
        case_timeout_seconds=60.0,
    )
    assert changed.identity_digest() != protocol.identity_digest()


def test_side_search_streams_ignore_artifact_assignment() -> None:
    red = arena.side_search_seed("shared-streams", 1234, "RED")
    blue = arena.side_search_seed("shared-streams", 1234, "BLUE")

    assert red == arena.side_search_seed("shared-streams", 1234, "RED")
    assert red != blue
    # Artifact identity is deliberately not accepted by this public seam.
    assert list(arena.side_search_seed.__annotations__) == [
        "namespace",
        "world_seed",
        "side",
        "return",
    ]


def test_request_schedule_version_changes_arena_protocol_identity(
    runner: arena.LearnedArenaRunner,
) -> None:
    kwargs = dict(
        world_seeds=(1,),
        source_revision="revision",
        dirty_tree_hash="tree",
        case_timeout_seconds=30.0,
    )
    baseline = arena.build_protocol(runner, **kwargs)
    scheduled = arena.build_protocol(
        replace(
            runner,
            search_config=replace(runner.search_config, request_schedule_version=1),
        ),
        **kwargs,
    )

    assert baseline.agent_a.params["search_config"]["request_schedule_version"] is None
    assert scheduled.agent_a.params["search_config"]["request_schedule_version"] == 1
    assert baseline.identity_digest() != scheduled.identity_digest()


def test_protocol_identity_supports_resume_but_changes_for_artifacts_or_namespace(
    runner: arena.LearnedArenaRunner,
) -> None:
    kwargs = dict(
        world_seeds=(1, 2),
        source_revision="revision",
        dirty_tree_hash="tree",
        case_timeout_seconds=30.0,
    )
    baseline = arena.build_protocol(runner, **kwargs)
    resumed = arena.build_protocol(runner, **kwargs)
    assert baseline.identity_digest() == resumed.identity_digest()
    assert [case.case_id for case in baseline.cases()] == [case.case_id for case in resumed.cases()]

    changed_artifact = arena.build_protocol(replace(runner, candidate_digest="c" * 64), **kwargs)
    changed_stream = arena.build_protocol(
        replace(runner, random_stream_namespace="another-stream"), **kwargs
    )
    assert baseline.identity_digest() != changed_artifact.identity_digest()
    assert baseline.identity_digest() != changed_stream.identity_digest()
    assert {case.case_id for case in baseline.cases()}.isdisjoint(
        case.case_id for case in changed_artifact.cases()
    )


def test_pinned_artifact_digest_mismatch_fails_before_runtime_load(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(json.dumps({"model_digest": "a" * 64}))

    with pytest.raises(ValueError, match="pinned digest"):
        arena._load_pinned_runtime(
            artifact,
            "b" * 64,
            map_id="map",
            game_type="QUICK",
            heroes=frozenset({"Wasp"}),
        )


def test_arena_loads_current_runtime_with_pinned_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automata.models.shared_encoder import serving

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "model_digest": "a" * 64,
                "runtime_compatibility_version": 2,
                "observation_schema_version": 4,
            }
        )
    )
    captured: list[Any] = []

    def load_runtime(*_args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs["requirements"])
        return SimpleNamespace(
            schema=SimpleNamespace(observation_schema_version=4),
            supported_maps=frozenset({"map"}),
            supported_game_types=frozenset({"QUICK"}),
            supported_heroes=frozenset({"Wasp"}),
        )

    monkeypatch.setattr(serving, "load_runtime", load_runtime)

    runtime = arena._load_pinned_runtime(
        artifact,
        "a" * 64,
        map_id="map",
        game_type="QUICK",
        heroes=frozenset({"Wasp"}),
    )

    assert runtime.supported_heroes == frozenset({"Wasp"})
    assert captured[0].runtime_compatibility_version == 2
    assert captured[0].observation_schema_version == 4


@pytest.mark.parametrize(
    ("field", "obsolete_version"),
    [("observation_schema_version", 3), ("runtime_compatibility_version", 1)],
)
def test_arena_rejects_obsolete_artifacts_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, obsolete_version: int
) -> None:
    import torch

    from automata.models.contracts import ArtifactError, ArtifactScope
    from automata.models.shared_encoder.artifacts import export_model_artifact
    from automata.models.shared_encoder.model import JointModelConfig, JointPolicyValueModel
    from automata.models.shared_encoder.schema import TensorFeatureSchema

    schema = TensorFeatureSchema.current()
    model = JointPolicyValueModel(
        schema=schema,
        config=JointModelConfig(
            model_version=2,
            schema_digest=schema.digest,
            token_width=4,
            state_width=6,
            candidate_width=4,
            message_passing_layers=1,
        ),
    )
    artifact = tmp_path / "artifact"
    manifest = export_model_artifact(
        artifact,
        model=model,
        schema=schema,
        scope=ArtifactScope(
            supported_heroes=("Wasp",),
            supported_maps=("map",),
            supported_game_types=("QUICK",),
            map_schema_version=1,
            hero_adapter_versions={"generic": 1, "Wasp": 1},
        ),
        runtime_compatibility_version=2,
    )
    payload = manifest.model_dump(mode="json")
    payload[field] = obsolete_version
    (artifact / "manifest.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )

    def unexpected_weights_load(*_args, **_kwargs):
        raise AssertionError("obsolete artifacts must fail before deserializing weights")

    monkeypatch.setattr(torch, "load", unexpected_weights_load)
    with pytest.raises(ArtifactError, match="invalid artifact manifest"):
        arena._load_pinned_runtime(
            artifact,
            manifest.model_digest,
            map_id="map",
            game_type="QUICK",
            heroes=frozenset({"Wasp"}),
        )


def test_pinned_artifact_scope_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from automata.models.shared_encoder import serving

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(json.dumps({"model_digest": "a" * 64}))
    monkeypatch.setattr(
        serving,
        "load_runtime",
        lambda *_args, **_kwargs: SimpleNamespace(
            supported_maps=frozenset(),
            supported_game_types=frozenset({"QUICK"}),
            supported_heroes=frozenset({"Wasp"}),
        ),
    )

    with pytest.raises(ValueError, match="arena map"):
        arena._load_pinned_runtime(
            artifact,
            "a" * 64,
            map_id="map",
            game_type="QUICK",
            heroes=frozenset({"Wasp"}),
        )


@pytest.mark.parametrize(
    ("search_config", "message"),
    [
        ({"iterations": 1, "unknown": 2}, "unknown"),
        ({"iterations": "1"}, "iterations"),
        ({"iterations": 1, "seed": 7}, "seed"),
        ({"iterations": 1, "leaf_mode": "ROLLOUT"}, "leaf_mode"),
    ],
)
def test_cli_rejects_search_config_not_accepted_by_self_play_preset(
    tmp_path: Path,
    search_config: dict[str, Any],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [
        "--candidate-artifact",
        str(tmp_path / "candidate"),
        "--candidate-digest",
        "a" * 64,
        "--parent-artifact",
        str(tmp_path / "parent"),
        "--parent-digest",
        "b" * 64,
        "--checkpoint",
        str(tmp_path / "checkpoint.jsonl"),
        "--map-path",
        str(tmp_path / "map.json"),
        "--game-type",
        "QUICK",
        "--red-heroes",
        "Wasp",
        "Xargatha",
        "--blue-heroes",
        "Arien",
        "Brogan",
        "--seed-start",
        "0",
        "--seed-end",
        "1",
        "--max-steps",
        "100",
        "--timeout-seconds",
        "10",
        "--search-config",
        json.dumps(search_config),
        "--random-stream-namespace",
        "stratum-0",
    ]

    with pytest.raises(SystemExit, match="2"):
        arena.main(args)
    assert message in capsys.readouterr().err
