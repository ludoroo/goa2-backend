from __future__ import annotations

import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar

import pytest

from automata.agents.contracts import PlanningDecision
from automata.harness.game_runner import DEFAULT_MAP, RunResult
from automata.runtime.driver import BotDecision, DecisionKind
from automata.runtime.effects import register_all_effects
from automata.training.dataset import (
    JointDatasetRecorder,
    load_joint_dataset,
    write_joint_dataset,
)
from automata.training.experiments.phase0 import PHASE0_EXPERIMENT
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup


def _module() -> Any:
    return importlib.import_module("automata.scripts.generate_joint_bootstrap")


def _args(out: Path, checkpoint: Path, start: int = 10_000, end: int = 10_002) -> list[str]:
    return [
        "--out",
        str(out),
        "--checkpoint",
        str(checkpoint),
        "--seed-start",
        str(start),
        "--seed-end",
        str(end),
        "--target-source",
        "heuristic",
        "--target-recipe",
        "one-hot-exact-choice",
        "--max-steps",
        "123",
        "--timeout-seconds",
        "30",
    ]


class _RecorderSpy:
    calls: ClassVar[list[_RecorderSpy]] = []

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        self.path, self.kwargs, self.closed = Path(path), kwargs, False
        self.calls.append(self)

    def close(self) -> None:
        self.closed = True


def test_generator_excludes_repository_runs_root_from_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    captured: dict[str, tuple[Path, ...]] = {}

    def identity(**kwargs: Any) -> tuple[str, str]:
        captured.update(kwargs)
        return "rev", "dirty"

    monkeypatch.setattr(module, "source_identity", identity)
    monkeypatch.setattr(module, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "run_game",
        lambda *_args, **_kwargs: RunResult(None, 1, 2, 3, "max_steps", winner_side=None),
    )
    out = tmp_path / "other-output" / "joint.jsonl"
    checkpoint = tmp_path / "other-output" / "checkpoint.jsonl"

    assert module.main(_args(out, checkpoint, end=10_001)) == 0
    assert tmp_path / "runs" in captured["exclude_paths"]


def test_generator_uses_fresh_deterministic_agents_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _RecorderSpy.calls = []
    monkeypatch.setattr(module, "JointDatasetRecorder", _RecorderSpy)
    monkeypatch.setattr(module, "source_identity", lambda **_kwargs: ("rev", "dirty"))
    monkeypatch.setattr(module, "_publish_output", lambda *_a, **_kw: None)
    runs: list[dict[str, Any]] = []

    def fake_run(red: Any, blue: Any, agents: Any, **kwargs: Any) -> RunResult:
        runs.append({"red": list(red), "blue": list(blue), "agents": dict(agents), **kwargs})
        return RunResult("hero_wasp", 1, 2, 3, "game_over", winner_side="RED")

    monkeypatch.setattr(module, "run_game", fake_run)
    out, checkpoint = tmp_path / "joint.jsonl", tmp_path / "checkpoint.jsonl"
    args = _args(out, checkpoint)

    assert module.main(args) == 0
    assert [captured["seed"] for captured in runs] == [10_000, 10_001]
    for captured in runs:
        red_agent = captured["agents"]["hero_wasp"]
        blue_agent = captured["agents"]["hero_arien"]
        assert captured["agents"]["hero_xargatha"] is red_agent
        assert captured["agents"]["hero_brogan"] is blue_agent
        assert red_agent is not blue_agent
        assert captured["decision_observer"].recorder in _RecorderSpy.calls
        assert captured["max_steps"] == 123
    assert len({id(captured["agents"]["hero_wasp"]) for captured in runs}) == 2
    assert all(call.path.name.endswith(".jsonl.zst") for call in _RecorderSpy.calls)

    assert module.main(args) == 0
    assert len(runs) == 2
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert all(row["completed"] and row["reason"] == "game_over" for row in rows)
    assert all(row["winner"] == "hero_wasp" and row["winner_side"] == "RED" for row in rows)


def test_diverse_pilot_uniformly_samples_legal_cards_and_finish_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    register_all_effects()
    scope = PHASE0_EXPERIMENT
    state = GameSetup.create_game(
        DEFAULT_MAP,
        list(scope.red_heroes),
        list(scope.blue_heroes),
        game_type=scope.game_type,
        seed=10_000,
    )
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    monkeypatch.setattr(module, "planning_open_for_second_card", lambda *_a: True)

    def choices(seed: int) -> list[tuple[str, str | None]]:
        agent = module.UniformPlanningAgent(seed)
        return [
            (
                decision.kind.value,
                str(decision.card.id) if decision.card is not None else None,
            )
            for _ in range(400)
            for decision in [agent.choose_planning(state, hero)]
        ]

    assert choices(91) == choices(91)
    counts = Counter(choices(91))
    legal = {("COMMIT", str(card.id)) for card in hero.hand} | {("FINISH", None)}
    assert set(counts) == legal
    assert max(counts.values()) - min(counts.values()) < 45

    request = InputRequest(
        id="delegate",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value("first"), InputOption.from_value("second")],
    )
    wrapper = module.UniformPlanningAgent(7)
    assert wrapper.choose_input(state, request) == wrapper.heuristic.choose_input(state, request)


def test_diverse_variant_schedule_is_balanced_and_stable_per_world_seed() -> None:
    module = _module()
    variants = [module.variant_for_world_seed(seed, "balanced") for seed in range(10_000, 10_004)]

    assert [
        (variant.game_type, variant.red_heroes, variant.blue_heroes) for variant in variants
    ] == [
        ("QUICK", ("Wasp", "Xargatha"), ("Arien", "Brogan")),
        ("QUICK", ("Arien", "Brogan"), ("Wasp", "Xargatha")),
        ("LONG", ("Wasp", "Xargatha"), ("Arien", "Brogan")),
        ("LONG", ("Arien", "Brogan"), ("Wasp", "Xargatha")),
    ]
    assert module.variant_for_world_seed(10_004, "balanced") == variants[0]


def test_soft_card_targets_use_temperature_and_uniform_mass_but_inputs_stay_one_hot(
    tmp_path: Path,
) -> None:
    module = _module()
    register_all_effects()
    scope = PHASE0_EXPERIMENT
    state = GameSetup.create_game(
        DEFAULT_MAP,
        list(scope.red_heroes),
        list(scope.blue_heroes),
        game_type=scope.game_type,
        seed=10_000,
    )
    hero = state.get_hero(HeroID("hero_wasp"))
    assert hero is not None
    path = tmp_path / "soft.jsonl"
    recorder = JointDatasetRecorder(
        path,
        game_id="soft",
        world_seed=10_000,
        map_id=scope.map_id,
        game_type=scope.game_type,
        red_composition=scope.red_heroes,
        blue_composition=scope.blue_heroes,
        generation_id="generation",
        source_revision="rev",
        dirty_tree_hash="clean",
        source_model_digest=None,
        search_config_id="soft-target",
        generator_config_id="config",
    )
    observer = module.HeuristicJointObserver(
        recorder,
        planning_behavior="uniform",
        target_recipe=module.SOFT_CARD_TARGET_RECIPE,
        card_target_temperature=0.75,
        card_target_uniform_mass=0.2,
    )
    observer.record_decision(
        state,
        BotDecision(
            kind=DecisionKind.PLANNING,
            hero_id=HeroID("hero_wasp"),
            planning=PlanningDecision.commit(hero.hand[0]),
        ),
    )
    request = InputRequest(
        id="input",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value("a"), InputOption.from_value("b")],
    )
    observer.record_decision(
        state,
        BotDecision(
            kind=DecisionKind.INPUT,
            hero_id=HeroID("hero_wasp"),
            request=request,
            selection="b",
        ),
    )
    observer.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    card_row, input_row = load_joint_dataset(path).rows
    assert card_row.policy_source == "UNIFORM_PLANNING_SOFT_HEURISTIC"
    assert input_row.policy_source == "HEURISTIC"
    assert sum(card_row.policy_target) == pytest.approx(1.0)
    assert all(
        probability >= 0.2 / len(card_row.policy_target) for probability in card_row.policy_target
    )
    assert len(set(card_row.policy_target)) > 1
    assert input_row.policy_target == (0.0, 1.0)


def test_default_config_marks_outcome_contract_and_diverse_knobs_are_identity_bearing(
    tmp_path: Path,
) -> None:
    module = _module()
    default_args = module.parse_generator_args(_args(tmp_path / "out", tmp_path / "checkpoint"))
    default_config = module.generator_config(
        default_args, source_revision="revision", dirty_tree_hash="dirty"
    )
    assert "pilot" not in default_config
    assert default_config["outcome_contract"] == "raw-winner+canonical-side-v1"
    assert default_config["scope"] == {
        "map_id": "forgotten_island",
        "map_path": DEFAULT_MAP,
        "game_type": "QUICK",
        "red_heroes": ("Wasp", "Xargatha"),
        "blue_heroes": ("Arien", "Brogan"),
        "seed_purpose": "bootstrap",
    }

    diverse_argv = [
        value
        for value in _args(tmp_path / "out", tmp_path / "checkpoint")
        if value != "one-hot-exact-choice"
    ]
    recipe_index = diverse_argv.index("--target-recipe")
    diverse_argv.insert(recipe_index + 1, module.SOFT_CARD_TARGET_RECIPE)
    diverse_argv.extend(
        [
            "--pilot-mode",
            "diverse",
            "--card-target-temperature",
            "0.7",
            "--card-target-uniform-mass",
            "0.25",
        ]
    )
    diverse_args = module.parse_generator_args(diverse_argv)
    config = module.generator_config(
        diverse_args, source_revision="revision", dirty_tree_hash="dirty"
    )
    assert config["pilot"]["variant_schedule"] == module.BALANCED_VARIANT_SCHEDULE
    assert config["pilot"]["planning_behavior"] == "uniform"
    assert config["pilot"]["card_target"] == {
        "recipe": module.SOFT_CARD_TARGET_RECIPE,
        "temperature": 0.7,
        "uniform_mass": 0.25,
    }
    changed = module.parse_generator_args([*diverse_argv[:-1], "0.3"])
    assert module.generator_config_id(
        diverse_args, source_revision="revision", dirty_tree_hash="dirty"
    ) != module.generator_config_id(changed, source_revision="revision", dirty_tree_hash="dirty")


def test_diverse_main_runs_one_balanced_variant_per_world_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _RecorderSpy.calls = []
    monkeypatch.setattr(module, "JointDatasetRecorder", _RecorderSpy)
    monkeypatch.setattr(module, "source_identity", lambda **_kwargs: ("rev", "dirty"))

    def publish_output(out: Path, *_args: object, publish: bool = True, **_kwargs: object) -> None:
        if publish:
            out.write_bytes(b"dataset")

    monkeypatch.setattr(module, "_publish_output", publish_output)
    runs: list[dict[str, Any]] = []

    def fake_run(red: Any, blue: Any, agents: Any, **kwargs: Any) -> RunResult:
        runs.append({"red": tuple(red), "blue": tuple(blue), "agents": agents, **kwargs})
        return RunResult("RED", 1, 2, 3, "game_over", winner_side="RED")

    monkeypatch.setattr(module, "run_game", fake_run)
    argv = _args(tmp_path / "out", tmp_path / "checkpoint", 10_000, 10_004)
    recipe = argv.index("one-hot-exact-choice")
    argv[recipe] = module.SOFT_CARD_TARGET_RECIPE
    argv.extend(["--pilot-mode", "diverse"])

    assert module.main(argv) == 0
    assert [(run["game_type"], run["red"], run["blue"]) for run in runs] == [
        (variant.game_type, variant.red_heroes, variant.blue_heroes)
        for variant in module.BALANCED_VARIANTS
    ]
    assert all(
        isinstance(next(iter(run["agents"].values())), module.UniformPlanningAgent) for run in runs
    )
    assert [call.kwargs["world_seed"] for call in _RecorderSpy.calls] == list(range(10_000, 10_004))

    sidecar_path = Path(f"{tmp_path / 'out'}.provenance.json")
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar = json.loads(sidecar_bytes)
    assert sidecar_bytes == module.canonical_json_bytes(
        module.GeneratorProvenance.model_validate(sidecar)
    )
    assert sidecar["schema_version"] == 1
    assert sidecar["seed_range"] == {"start": 10_000, "end": 10_004}
    assert sidecar["generator_config_id"] == _RecorderSpy.calls[0].kwargs["generator_config_id"]
    assert sidecar["generator_config"]["pilot"] == {
        "mode": "diverse",
        "planning_behavior": "uniform",
        "variant_schedule": [
            {
                "game_type": variant.game_type,
                "red_heroes": list(variant.red_heroes),
                "blue_heroes": list(variant.blue_heroes),
            }
            for variant in module.BALANCED_VARIANTS
        ],
        "card_target": {
            "recipe": module.SOFT_CARD_TARGET_RECIPE,
            "temperature": 1.0,
            "uniform_mass": 0.1,
        },
        "input_target_recipe": module.TARGET_RECIPE,
        "policy_sources": {
            "planning": "UNIFORM_PLANNING_SOFT_HEURISTIC",
            "input": "HEURISTIC",
        },
    }
    assert sidecar["target_provenance"]["planning_behavior"] == "uniform"
    assert not list(tmp_path.glob(".out.provenance.json.*"))

    assert module.main(argv) == 0
    assert sidecar_path.read_bytes() == sidecar_bytes


def test_diverse_mode_supplies_all_diverse_defaults(tmp_path: Path) -> None:
    module = _module()
    argv = _args(tmp_path / "out", tmp_path / "checkpoint")
    recipe_flag = argv.index("--target-recipe")
    del argv[recipe_flag : recipe_flag + 2]
    args = module.parse_generator_args([*argv, "--pilot-mode", "diverse"])

    assert args.planning_behavior == "uniform"
    assert args.variant_schedule == "balanced"
    assert args.target_recipe == module.SOFT_CARD_TARGET_RECIPE


def test_rejects_seed_range_outside_bootstrap_before_touching_files(tmp_path: Path) -> None:
    out, checkpoint = tmp_path / "joint.jsonl", tmp_path / "checkpoint.jsonl"
    with pytest.raises(SystemExit):
        _module().main(_args(out, checkpoint, 9_999, 10_001))
    assert not out.exists() and not checkpoint.exists()


def test_checkpoint_tolerates_only_a_truncated_final_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _RecorderSpy.calls = []
    monkeypatch.setattr(module, "JointDatasetRecorder", _RecorderSpy)
    monkeypatch.setattr(module, "source_identity", lambda **_kwargs: ("rev", "dirty"))
    monkeypatch.setattr(module, "_publish_output", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        module,
        "run_game",
        lambda *_a, **_kw: RunResult("RED", 1, 2, 3, "game_over", winner_side="RED"),
    )
    out, checkpoint = tmp_path / "joint.jsonl", tmp_path / "checkpoint.jsonl"
    checkpoint.write_bytes(b'{"broken":')
    assert module.main(_args(out, checkpoint, 10_000, 10_001)) == 0
    assert len(checkpoint.read_text().splitlines()) == 1

    checkpoint.write_bytes(b'{"broken":true}\n{"truncated":')
    with pytest.raises(ValueError, match="checkpoint row 1"):
        module.main(_args(out, checkpoint, 10_000, 10_001))


def test_checkpoint_rejects_legacy_rows_without_explicit_winner_side(tmp_path: Path) -> None:
    module = _module()
    checkpoint = tmp_path / "legacy.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "config_id": "config",
                "game_id": "game",
                "world_seed": 10_000,
                "completed": True,
                "reason": "game_over",
                "winner": "RED",
                "rounds": 1,
                "turns": 1,
                "steps": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="winner_side"):
        module._read_checkpoint(checkpoint)


def test_resume_reconciles_empty_aggregate_from_an_old_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "source_identity", lambda **_kwargs: ("rev", "dirty"))
    runs: list[int] = []

    def incomplete_run(*_args: Any, **kwargs: Any) -> RunResult:
        runs.append(kwargs["seed"])
        return RunResult(None, 1, 2, 123, "max_steps", winner_side=None)

    monkeypatch.setattr(module, "run_game", incomplete_run)
    out, checkpoint = tmp_path / "joint.jsonl.zst", tmp_path / "checkpoint.jsonl"
    write_joint_dataset(out, ())
    module._append_checkpoint(
        checkpoint,
        module.CheckpointRow(
            config_id="old-config",
            game_id="old-game",
            world_seed=10_000,
            completed=True,
            reason="game_over",
            winner="RED",
            winner_side="RED",
            rounds=1,
            turns=2,
            steps=3,
        ),
    )
    stale_fragment = module._fragment_path(module._fragment_dir(out), "old-game")
    stale_fragment.parent.mkdir()
    stale_fragment.write_bytes(b"stale")

    assert module.main(_args(out, checkpoint, 10_000, 10_001)) == 0

    assert runs == [10_000]
    assert not out.exists()
    assert not stale_fragment.exists()


def test_incomplete_games_do_not_publish_an_empty_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "source_identity", lambda **_kwargs: ("rev", "dirty"))
    monkeypatch.setattr(
        module,
        "run_game",
        lambda *_args, **_kwargs: RunResult(None, 1, 2, 123, "max_steps", winner_side=None),
    )
    out, checkpoint = tmp_path / "joint.jsonl.zst", tmp_path / "checkpoint.jsonl"
    provenance = module.generator_provenance_path(out)
    provenance.write_text("stale")

    assert module.main(_args(out, checkpoint, 10_000, 10_001)) == 0

    assert not out.exists()
    assert not provenance.exists()


def test_real_heuristic_decision_records_exact_v4_one_hot_row(tmp_path: Path) -> None:
    module = _module()
    register_all_effects()
    scope = PHASE0_EXPERIMENT
    state = GameSetup.create_game(
        DEFAULT_MAP,
        list(scope.red_heroes),
        list(scope.blue_heroes),
        game_type=scope.game_type,
        seed=10_000,
    )
    from automata.runtime.driver import inspect_next_decision

    decision = inspect_next_decision(state, module.build_agents(10_000))
    assert decision is not None
    path = tmp_path / "one-game.jsonl"
    recorder = JointDatasetRecorder(
        path,
        game_id="game",
        world_seed=10_000,
        map_id=scope.map_id,
        game_type=scope.game_type,
        red_composition=scope.red_heroes,
        blue_composition=scope.blue_heroes,
        generation_id="generation",
        source_revision="rev",
        dirty_tree_hash="clean",
        source_model_digest=None,
        search_config_id="none",
        generator_config_id="config",
    )
    observer = module.HeuristicJointObserver(recorder)
    observer.record_decision(state, decision)
    observer.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    [row] = load_joint_dataset(path).rows
    assert row.observation.schema_version == 4
    selections = [candidate.selection for candidate in row.observation.candidates]
    assert row.selected_selection in selections
    assert row.policy_target == tuple(
        1.0 if candidate.candidate_id == row.selected_candidate_id else 0.0
        for candidate in row.observation.candidates
    )


def test_observer_preserves_finish_and_skip_selections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    register_all_effects()
    scope = PHASE0_EXPERIMENT
    state = GameSetup.create_game(
        DEFAULT_MAP,
        list(scope.red_heroes),
        list(scope.blue_heroes),
        game_type=scope.game_type,
        seed=10_000,
    )
    path = tmp_path / "sentinels.jsonl"
    recorder = JointDatasetRecorder(
        path,
        game_id="sentinels",
        world_seed=10_000,
        map_id=scope.map_id,
        game_type=scope.game_type,
        red_composition=scope.red_heroes,
        blue_composition=scope.blue_heroes,
        generation_id="generation",
        source_revision="rev",
        dirty_tree_hash="clean",
        source_model_digest=None,
        search_config_id="none",
        generator_config_id="config",
    )
    observer = module.HeuristicJointObserver(recorder)
    monkeypatch.setattr(module, "planning_open_for_second_card", lambda *_a: True)
    observer.record_decision(
        state,
        BotDecision(
            kind=DecisionKind.PLANNING,
            hero_id=HeroID("hero_wasp"),
            planning=PlanningDecision.finish(),
        ),
    )
    request = InputRequest(
        id="skip",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value("hold")],
        can_skip=True,
    )
    observer.record_decision(
        state,
        BotDecision(
            kind=DecisionKind.INPUT,
            hero_id=HeroID("hero_wasp"),
            request=request,
            selection="SKIP",
        ),
    )
    observer.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    rows = load_joint_dataset(path).rows
    assert [row.selected_selection for row in rows] == [None, "SKIP"]


def test_observer_preserves_numeric_engine_selection(tmp_path: Path) -> None:
    module = _module()
    register_all_effects()
    scope = PHASE0_EXPERIMENT
    state = GameSetup.create_game(
        DEFAULT_MAP,
        list(scope.red_heroes),
        list(scope.blue_heroes),
        game_type=scope.game_type,
        seed=10_000,
    )
    path = tmp_path / "numeric.jsonl"
    recorder = JointDatasetRecorder(
        path,
        game_id="numeric",
        world_seed=10_000,
        map_id=scope.map_id,
        game_type=scope.game_type,
        red_composition=scope.red_heroes,
        blue_composition=scope.blue_heroes,
        generation_id="generation",
        source_revision="rev",
        dirty_tree_hash="clean",
        source_model_digest=None,
        search_config_id="none",
        generator_config_id="config",
    )
    observer = module.HeuristicJointObserver(recorder)
    request = InputRequest(
        id="number-like-option",
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption.from_value(2), InputOption.from_value(3)],
    )
    observer.record_decision(
        state,
        BotDecision(
            kind=DecisionKind.INPUT,
            hero_id=HeroID("hero_wasp"),
            request=request,
            selection=2,
        ),
    )
    observer.record_outcome(winner_side="RED", rounds=1, reason="game_over")

    [row] = load_joint_dataset(path).rows
    assert row.selected_selection == 2
    assert [candidate.selection for candidate in row.observation.candidates] == [2, 3]
