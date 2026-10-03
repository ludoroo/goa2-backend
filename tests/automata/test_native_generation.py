"""Concrete native generation records only real played roots and live boundaries."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from automata.agents.contracts import PlanningKind
from automata.models.shared_encoder.artifacts import export_gen1_model_artifact
from automata.models.shared_encoder.gen1_model import Gen1ModelConfig, Gen1PolicyValueModel
from automata.models.shared_encoder.gen1_runtime import Gen1SharedEncoderRuntime
from automata.models.shared_encoder.schema import StableValueTensorSchema, TensorFeatureSchema
from automata.runtime.driver import DecisionKind
from automata.search.config import SearchConfig
from automata.search.contracts import LeafMode
from automata.training.native_dataset import (
    PolicyDatasetRecord,
    ValueDatasetRecord,
    iter_native_game_records,
)
from automata.training.native_gen1 import current_gen1_artifact_scope
from automata.training.native_generation import (
    NativeGenerationConfig,
    NativeGenerationGame,
    NativeGenerationOutput,
    derive_native_generation_seed,
    generate_native_game,
)
from automata.training.native_receipts import load_native_game_completion_receipt
from automata.training.native_splits import (
    NativeSeedRange,
    NativeSplitConfig,
    native_seed_purpose,
)
from goa2.domain.models import GamePhase, TargetType
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup
from goa2.engine.steps import ResolveCardStep, SelectStep, StepResult, TriggerGameOverStep


def _split_config() -> NativeSplitConfig:
    return NativeSplitConfig(
        namespace="native-generation-tests",
        salt="fixed",
        validation_fraction=0.2,
        seed_ranges=(
            NativeSeedRange(purpose="bootstrap", start=70, stop=80),
            NativeSeedRange(purpose="training", start=80, stop=90),
            NativeSeedRange(purpose="validation", start=90, stop=100),
            NativeSeedRange(purpose="evaluation", start=100, stop=110),
        ),
    )


def _config(**changes: Any) -> NativeGenerationConfig:
    values: dict[str, Any] = {
        "generation_id": "native-gen1-fixture",
        "source_revision": "fixture-revision",
        "dirty_tree_hash": "clean",
        "teacher_kind": "HEURISTIC_BOOTSTRAP",
        "source_model_digest": None,
        "search_config": SearchConfig(
            iterations=2,
            leaf_mode=LeafMode.STABLE_TRANSITION,
            seed=0,
        ),
        "split_config": _split_config(),
        "random_stream_namespace": "native-generation-test-streams",
        "visit_temperature": 1.0,
        "max_steps": 80,
        "max_rounds": None,
    }
    values.update(changes)
    return NativeGenerationConfig(**values)


def _game(seed: int = 73, purpose: str = "bootstrap") -> NativeGenerationGame:
    return NativeGenerationGame(
        world_seed=seed,
        seed_purpose=purpose,
        map_id="forgotten_island",
        game_type="QUICK",
        red_composition=("Wasp",),
        blue_composition=("Arien",),
    )


def _output(tmp_path: Path, logical_name: str = "games/game.jsonl.zst") -> NativeGenerationOutput:
    root = tmp_path / "source"
    root.mkdir()
    return NativeGenerationOutput(
        source_root=root,
        logical_name=logical_name,
        completion_receipt_path=tmp_path / "receipts" / "game.json",
    )


def _regular_files(root: Path) -> set[Path]:
    """Ignore harmless empty cleanup directories while detecting output files."""
    return {path for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def terminal_card(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    resolved_cards: list[str] = []

    def resolve(step: ResolveCardStep, state, context):
        del context
        hero = state.get_hero(step.hero_id)
        assert hero is not None and hero.current_turn_card is not None
        resolved_cards.append(str(hero.current_turn_card.id))
        return StepResult(
            is_finished=True,
            new_steps=[
                SelectStep(
                    target_type=TargetType.NUMBER,
                    number_options=[1, 2],
                    prompt="Played root choice",
                    override_player_id=str(step.hero_id),
                ),
                TriggerGameOverStep(
                    individual_winner_id=HeroID("hero_wasp"),
                    condition="NATIVE_GENERATION_TEST",
                ),
            ],
        )

    monkeypatch.setattr(ResolveCardStep, "resolve", resolve)
    return resolved_cards


def test_heuristic_generation_records_played_policy_and_live_value_rows(
    tmp_path: Path,
    terminal_card: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import automata.harness.game_runner as game_runner

    output = _output(tmp_path)
    applied: list[tuple[str, str, object]] = []
    real_apply_decision = game_runner.apply_decision

    def capture_applied_decision(session, decision, **kwargs):
        if decision.kind is DecisionKind.PLANNING:
            assert decision.planning is not None
            selected = (
                decision.planning.card.id if decision.planning.kind is PlanningKind.COMMIT else None
            )
            kind = "CARD"
        else:
            selected = decision.selection
            kind = "INPUT"
        result = real_apply_decision(session, decision, **kwargs)
        applied.append((str(decision.hero_id), kind, selected))
        return result

    monkeypatch.setattr(game_runner, "apply_decision", capture_applied_decision)
    config = _config(
        search_config=SearchConfig(
            iterations=8,
            leaf_mode=LeafMode.STABLE_TRANSITION,
            seed=0,
        ),
        visit_temperature=100.0,
    )

    result = generate_native_game(_game(seed=71), config, output)

    assert result.completed
    assert result.outcome.reason == "game_over"
    assert result.outcome.winner_side == "RED"
    assert result.completion_receipt is not None
    assert result.completion_receipt == load_native_game_completion_receipt(
        output.completion_receipt_path
    )
    source = output.source_root / output.logical_name
    rows = tuple(iter_native_game_records(source))
    policies = tuple(row for row in rows if isinstance(row, PolicyDatasetRecord))
    values = tuple(row for row in rows if isinstance(row, ValueDatasetRecord))
    assert policies and values and terminal_card
    assert result.completion_receipt.policy_row_count == len(policies)
    assert result.completion_receipt.value_row_count == len(values)
    assert all(sum(action.selected for action in row.target.actions) == 1 for row in policies)
    assert all(
        tuple(action.candidate for action in row.target.actions) == row.observation.candidates
        for row in policies
    )
    recorded = []
    for row in policies:
        owner = next(
            token.features["hero_id"]
            for token in row.observation.state.tokens
            if token.kind == "HERO" and token.features.get("is_decision_owner") is True
        )
        selected = next(
            action.candidate.selection for action in row.target.actions if action.selected
        )
        recorded.append((owner, row.observation.decision_kind, selected))
    assert recorded == applied
    assert any(
        action.selected
        and action.sample_count < max(item.sample_count for item in row.target.actions)
        for row in policies
        for action in row.target.actions
    ), "the deterministic high-temperature fixture must exercise a sampled non-argmax action"
    selected_cards = {
        action.candidate.selection
        for row in policies
        if row.observation.decision_kind == "CARD"
        for action in row.target.actions
        if action.selected
    }
    assert terminal_card[0] in selected_cards


def test_identical_seed_and_config_reproduce_native_source_bytes(
    tmp_path: Path,
    terminal_card: list[str],
) -> None:
    outputs = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        output = NativeGenerationOutput(
            source_root=root,
            logical_name="game.jsonl.zst",
            completion_receipt_path=tmp_path / f"{name}.completion.json",
        )
        result = generate_native_game(_game(), _config(), output)
        assert result.completed
        outputs.append(output)

    assert (outputs[0].source_root / outputs[0].logical_name).read_bytes() == (
        outputs[1].source_root / outputs[1].logical_name
    ).read_bytes()


def test_exact_gen1_parent_drives_policy_and_stable_value_inference(
    tmp_path: Path,
    terminal_card: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision_schema = TensorFeatureSchema.current()
    value_schema = StableValueTensorSchema.current()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = Gen1PolicyValueModel(
            decision_schema=decision_schema,
            stable_value_schema=value_schema,
            config=Gen1ModelConfig(
                decision_schema_digest=decision_schema.digest,
                stable_value_schema_digest=value_schema.digest,
                token_width=8,
                state_width=12,
                candidate_width=8,
                message_passing_layers=1,
            ),
        )
    artifact = tmp_path / "parent"
    manifest = export_gen1_model_artifact(
        artifact,
        model=model,
        decision_schema=decision_schema,
        stable_value_schema=value_schema,
        scope=current_gen1_artifact_scope(),
    )
    calls = {"policy": 0, "value": 0}
    real_policy = Gen1SharedEncoderRuntime.evaluate_policy
    real_value = Gen1SharedEncoderRuntime.evaluate_stable_value

    def policy(runtime, observation):
        calls["policy"] += 1
        return real_policy(runtime, observation)

    def value(runtime, observation):
        calls["value"] += 1
        return real_value(runtime, observation)

    monkeypatch.setattr(Gen1SharedEncoderRuntime, "evaluate_policy", policy)
    monkeypatch.setattr(Gen1SharedEncoderRuntime, "evaluate_stable_value", value)
    output = _output(tmp_path)
    config = _config(
        teacher_kind="GEN1_PARENT",
        source_model_digest=manifest.model_digest,
    )

    result = generate_native_game(
        _game(),
        config,
        output,
        parent_artifact_path=artifact,
    )

    assert result.completed
    assert result.game.source_model_digest == manifest.model_digest
    assert calls["policy"] > 0
    assert calls["value"] > 0

    inference_root = tmp_path / "inference-source"
    inference_root.mkdir()
    inference_output = NativeGenerationOutput(
        source_root=inference_root,
        logical_name="game.jsonl.zst",
        completion_receipt_path=tmp_path / "inference-completion.json",
    )

    def fail_policy(runtime, observation):
        del runtime, observation
        raise RuntimeError("injected inference failure")

    monkeypatch.setattr(Gen1SharedEncoderRuntime, "evaluate_policy", fail_policy)
    with pytest.raises(RuntimeError, match="injected inference failure"):
        generate_native_game(
            _game(),
            config,
            inference_output,
            parent_artifact_path=artifact,
        )
    assert not _regular_files(inference_root)
    assert not inference_output.completion_receipt_path.exists()

    bad_root = tmp_path / "mismatch-source"
    bad_root.mkdir()
    bad_output = NativeGenerationOutput(
        source_root=bad_root,
        logical_name="game.jsonl.zst",
        completion_receipt_path=tmp_path / "mismatch-completion.json",
    )
    wrong_digest = "0" * 64 if manifest.model_digest != "0" * 64 else "1" * 64
    with pytest.raises(ValueError, match="digest"):
        generate_native_game(
            _game(),
            _config(
                teacher_kind="GEN1_PARENT",
                source_model_digest=wrong_digest,
            ),
            bad_output,
            parent_artifact_path=artifact,
        )
    assert not _regular_files(bad_root)
    assert not bad_output.completion_receipt_path.exists()


def test_generator_config_identity_excludes_per_generation_provenance(
    tmp_path: Path,
    terminal_card: list[str],
) -> None:
    first_config = _config()
    second_config = replace(
        first_config,
        generation_id="another-generation",
        source_revision="another-revision",
        dirty_tree_hash="another-tree-state",
    )
    assert first_config.generator_config_id == second_config.generator_config_id

    results = []
    for name, config in (("first", first_config), ("second", second_config)):
        root = tmp_path / name
        root.mkdir()
        output = NativeGenerationOutput(
            source_root=root,
            logical_name="game.jsonl.zst",
            completion_receipt_path=tmp_path / f"{name}.completion.json",
        )
        results.append(generate_native_game(_game(), config, output))

    assert results[0].game.generator_config_id == results[1].game.generator_config_id
    assert results[0].game.game_id != results[1].game.game_id
    assert results[0].game.generation_id != results[1].game.generation_id


def test_seed_streams_are_domain_separated_and_ids_are_reproducible() -> None:
    config = _config()
    streams = (
        "RED_SEARCH",
        "RED_ACTION",
        "RED_ENVIRONMENT",
        "BLUE_SEARCH",
        "BLUE_ACTION",
        "BLUE_ENVIRONMENT",
    )
    derived = tuple(
        derive_native_generation_seed(config, world_seed=73, stream=stream)  # type: ignore[arg-type]
        for stream in streams
    )

    assert len(set(derived)) == len(derived)
    assert derived == tuple(
        derive_native_generation_seed(config, world_seed=73, stream=stream)  # type: ignore[arg-type]
        for stream in streams
    )
    assert derive_native_generation_seed(config, world_seed=74, stream="RED_SEARCH") != derived[0]
    assert _config().search_config_id == config.search_config_id
    assert _config().generator_config_id == config.generator_config_id
    assert native_seed_purpose(config.split_config, 73) == "bootstrap"


@pytest.mark.parametrize(
    ("game", "config", "match"),
    [
        (_game(purpose="training"), _config(), "purpose"),
        (_game(seed=100, purpose="bootstrap"), _config(), "eligible"),
        (_game(seed=999), _config(), "declared"),
        (_game().model_copy(update={"map_id": "unknown"}), _config(), "map"),
        (_game().model_copy(update={"game_type": "UNKNOWN"}), _config(), "game type"),
        (
            _game().model_copy(update={"red_composition": ("Unknown Hero",)}),
            _config(),
            "hero",
        ),
    ],
)
def test_invalid_scope_and_seed_purpose_reject_before_files(
    tmp_path: Path,
    game: NativeGenerationGame,
    config: NativeGenerationConfig,
    match: str,
) -> None:
    output = _output(tmp_path)

    with pytest.raises((ValueError, ValidationError), match=match):
        generate_native_game(game, config, output)

    assert not (output.source_root / output.logical_name).exists()
    assert not output.completion_receipt_path.exists()
    assert not _regular_files(output.source_root)


@pytest.mark.parametrize(
    ("config", "parent_artifact_path", "match"),
    [
        (_config(), "unexpected-parent", "HEURISTIC_BOOTSTRAP"),
        (
            _config(teacher_kind="GEN1_PARENT", source_model_digest="0" * 64),
            None,
            "GEN1_PARENT",
        ),
    ],
)
def test_teacher_parent_path_mismatch_rejects_before_files(
    tmp_path: Path,
    config: NativeGenerationConfig,
    parent_artifact_path: str | None,
    match: str,
) -> None:
    output = _output(tmp_path)
    before = _regular_files(tmp_path)

    with pytest.raises(ValueError, match=match):
        generate_native_game(
            _game(),
            config,
            output,
            parent_artifact_path=parent_artifact_path,
        )

    assert _regular_files(tmp_path) == before


def test_complete_game_without_recordable_rows_is_unreceipted_and_not_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import automata.harness.game_runner as game_runner

    real_create_game = GameSetup.create_game

    def create_completed_game(**kwargs):
        state = real_create_game(**kwargs)
        state.individual_winner_id = HeroID("hero_wasp")
        state.phase = GamePhase.GAME_OVER
        return state

    monkeypatch.setattr(
        game_runner.GameSetup,
        "create_game",
        staticmethod(create_completed_game),
    )
    output = _output(tmp_path)

    result = generate_native_game(_game(), _config(), output)

    assert result.outcome.reason == "game_over"
    assert not result.completed
    assert result.completion_receipt is None
    assert not (output.source_root / output.logical_name).exists()
    assert not output.completion_receipt_path.exists()
    assert not tuple(output.source_root.rglob("*.pending.zst"))


def test_config_rejects_nonstable_search_parent_mismatch_and_implicit_randomness() -> None:
    with pytest.raises(ValueError, match="STABLE_TRANSITION"):
        _config(search_config=SearchConfig(leaf_mode=LeafMode.IMMEDIATE))
    with pytest.raises(ValueError, match="seed"):
        _config(
            search_config=SearchConfig(
                leaf_mode=LeafMode.STABLE_TRANSITION,
                seed=1,
            )
        )
    with pytest.raises(ValueError, match="timeout"):
        _config(
            search_config=SearchConfig(
                leaf_mode=LeafMode.STABLE_TRANSITION,
                decision_timeout_seconds=1.0,
            )
        )
    with pytest.raises(ValueError, match="digest"):
        _config(source_model_digest="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        _config(teacher_kind="GEN1_PARENT")
    with pytest.raises(ValueError, match="use_prior"):
        _config(
            teacher_kind="GEN1_PARENT",
            source_model_digest="0" * 64,
            search_config=SearchConfig(
                leaf_mode=LeafMode.STABLE_TRANSITION,
                use_prior=False,
            ),
        )


@pytest.mark.parametrize("logical_name", ["../game.jsonl", "/game.jsonl", "game.txt", "."])
def test_unsafe_output_rejects_before_creating_recorder_state(
    tmp_path: Path,
    logical_name: str,
) -> None:
    output = _output(tmp_path, logical_name)
    before = _regular_files(tmp_path)

    with pytest.raises((ValueError, FileNotFoundError), match=r"logical|JSONL|relative|normalized"):
        generate_native_game(_game(), _config(), output)

    assert _regular_files(tmp_path) == before


@pytest.mark.parametrize("empty_game", [False, True], ids=["censored", "empty"])
def test_postcondition_reports_and_preserves_competing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_game: bool,
) -> None:
    import automata.harness.game_runner as game_runner
    import automata.training.native_generation as generation

    output = _output(tmp_path)
    source = output.source_root / output.logical_name
    competitor = b"concurrent-competitor"
    real_record_outcome = generation.NativeDatasetRecorder.record_outcome

    def record_then_compete(recorder, **kwargs):
        real_record_outcome(recorder, **kwargs)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(competitor)

    monkeypatch.setattr(
        generation.NativeDatasetRecorder,
        "record_outcome",
        record_then_compete,
    )
    config = _config(max_steps=1)
    if empty_game:
        real_create_game = GameSetup.create_game

        def create_completed_game(**kwargs):
            state = real_create_game(**kwargs)
            state.individual_winner_id = HeroID("hero_wasp")
            state.phase = GamePhase.GAME_OVER
            return state

        monkeypatch.setattr(
            game_runner.GameSetup,
            "create_game",
            staticmethod(create_completed_game),
        )
        config = _config()

    with pytest.raises(RuntimeError, match="pre-existing or competing"):
        generate_native_game(_game(), config, output)

    assert source.read_bytes() == competitor
    assert not output.completion_receipt_path.exists()
    assert not tuple(output.source_root.rglob("*.pending.zst"))


def test_existing_competing_source_is_preserved(tmp_path: Path) -> None:
    output = _output(tmp_path)
    source = output.source_root / output.logical_name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"competitor")

    with pytest.raises(FileExistsError):
        generate_native_game(_game(), _config(), output)

    assert source.read_bytes() == b"competitor"
    assert not output.completion_receipt_path.exists()
    assert not tuple(output.source_root.rglob("*.pending.zst"))


@pytest.mark.parametrize(
    ("config_changes", "reason"),
    [({"max_steps": 1}, "max_steps"), ({"max_rounds": 1}, "max_rounds")],
)
def test_capped_game_is_censored_without_source_receipt_or_spool(
    tmp_path: Path,
    terminal_card: list[str],
    monkeypatch: pytest.MonkeyPatch,
    config_changes: dict[str, Any],
    reason: str,
) -> None:
    output = _output(tmp_path)
    if reason == "max_rounds":

        def advance_round(step: ResolveCardStep, state, context):
            del step, context
            state.round += 1
            return StepResult(is_finished=True)

        monkeypatch.setattr(ResolveCardStep, "resolve", advance_round)

    result = generate_native_game(_game(), _config(**config_changes), output)

    assert result.outcome.reason == reason
    assert not result.completed
    assert result.completion_receipt is None
    assert not (output.source_root / output.logical_name).exists()
    assert not output.completion_receipt_path.exists()
    assert not tuple(output.source_root.rglob("*.pending.zst"))


def test_search_exception_propagates_after_owned_spool_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import automata.training.native_generation as generation

    output = _output(tmp_path)

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected search failure")

    monkeypatch.setattr(generation.ISMCTSStrategy, "select", fail)
    with pytest.raises(RuntimeError, match="injected search failure"):
        generate_native_game(_game(), _config(), output)

    assert not (output.source_root / output.logical_name).exists()
    assert not output.completion_receipt_path.exists()
    assert not tuple(output.source_root.rglob("*.pending.zst"))


def test_generation_module_has_no_phase0_dependency() -> None:
    source = Path("src/automata/training/native_generation.py").read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any("experiments.phase0" in module for module in imported)


def test_public_seed_purpose_strictly_revalidates_config_and_seed() -> None:
    config = _split_config()
    assert native_seed_purpose(config, 73) == "bootstrap"
    with pytest.raises((TypeError, ValueError)):
        native_seed_purpose(config, True)
    with pytest.raises(ValidationError):
        native_seed_purpose(config.model_copy(update={"validation_fraction": float("nan")}), 73)


def test_generation_models_reject_duplicate_or_non_strict_rosters_and_limits() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _game().model_copy(update={"red_composition": ("Wasp", "Wasp")}).__class__.model_validate(
            {
                **_game().model_dump(mode="python"),
                "red_composition": ("Wasp", "Wasp"),
            },
            strict=True,
        )
    with pytest.raises(ValidationError):
        NativeGenerationGame(
            world_seed=True,
            seed_purpose="bootstrap",
            map_id="forgotten_island",
            game_type="QUICK",
            red_composition=("Wasp",),
            blue_composition=("Arien",),
        )
    with pytest.raises(ValueError, match="positive"):
        replace(_config(), max_steps=0)
