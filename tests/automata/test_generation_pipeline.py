"""Behavioral contract for the resumable generation-to-candidate coordinator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest

from automata.evaluation.arena import (
    ArenaConfig,
    ArenaOperationalEvidence,
    ArenaStage,
    ArenaStageConfig,
    ArtifactIdentity,
    run_arena,
)
from automata.evaluation.arena_stats import SequentialBoundary, SequentialPlan
from automata.evaluation.promotion_gates import ArtifactLoad, LatencySLO, PromotionGateConfig
from automata.evaluation.protocol import (
    AgentSpec,
    EvaluationGameResult,
    EvaluationProtocol,
    GameCase,
)
from automata.models.contracts import (
    DecisionObservation,
    EncodedCandidate,
    LearnedObservation,
    ObservationToken,
    OptionCandidateID,
    Viewer,
    canonical_json_bytes,
)
from automata.models.shared_encoder.artifacts import (
    ArtifactFile,
    ArtifactTensor,
    LoadedModelArtifact,
    ModelArtifactManifest,
)
from automata.training.dataset import JointDatasetRow, joint_decision_id, load_joint_dataset
from automata.training.generation import CheckpointRow, GameSpec, GenerationConfig, WorkerSpec
from automata.training.generation_pipeline import (
    PipelineDependencies,
    ResumableGenerationPipeline,
    ResumableGenerationPipelineConfig,
    StalePipelineProductError,
    WorkerOutput,
)
from automata.training.policy_iteration import PolicyIterationStatus, run_policy_iteration
from automata.training.registry import CandidateManifest, CandidateMetadata, ChampionRegistry
from automata.training.replay_buffer import ReplayBuffer
from automata.training.splits import JointSplitConfig, split_joint_dataset
from automata.training.trainer import JointTrainingConfig, JointTrainingResult


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _parent(digest: str = "a" * 64) -> ModelArtifactManifest:
    return ModelArtifactManifest.model_construct(
        schema_version=2,
        model_digest=digest,
        observation_schema_version=2,
        map_schema_version=1,
        runtime_compatibility_version=1,
        hero_adapter_versions={"generic": 1, "Wasp": 1, "Arien": 1},
        supported_heroes=("Wasp", "Arien"),
        supported_maps=("forgotten_island",),
        supported_game_types=("QUICK",),
        tensor_schema_id="joint-v1",
        tensor_schema_version=1,
        tensor_schema_digest="b" * 64,
        architecture_config={},
        tensors={"weight": ArtifactTensor(shape=(1,), dtype="float32")},
        files={},
        non_executable_files=(),
    )


def _row(spec: WorkerSpec, game_index: int = 0, *, decision_index: int = 0) -> JointDatasetRow:
    game = spec.games[game_index]
    candidate_id = OptionCandidateID(schema_version=1, option_id="advance")
    observation = DecisionObservation(
        schema_version=3,
        state=LearnedObservation(
            schema_version=2,
            viewer=Viewer(schema_version=2, perspective_team="RED"),
            tokens=(
                ObservationToken(
                    schema_version=1,
                    local_ref="global:0",
                    kind="GLOBAL",
                    features={"map_id": game.map_id, "game_type": game.game_type},
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
        candidates=(
            EncodedCandidate(schema_version=1, candidate_id=candidate_id, selection="advance"),
        ),
    )
    identity: dict[str, Any] = {
        "game_id": game.game_id(spec.config),
        "world_seed": game.world_seed,
        "map_id": game.map_id,
        "game_type": game.game_type,
        "red_composition": game.red_composition,
        "blue_composition": game.blue_composition,
        "generation_id": spec.config.generation_id,
        "source_revision": spec.config.source_revision,
        "dirty_tree_hash": spec.config.dirty_tree_hash,
        "source_model_digest": spec.config.parent_model_digest,
        "search_config_id": spec.config.search_config_id,
        "generator_config_id": spec.config.generator_config_id,
    }
    return JointDatasetRow(
        schema_version=1,
        decision_id=joint_decision_id(**identity, decision_index=decision_index),
        **identity,
        decision_index=decision_index,
        perspective_team="RED",
        observation=observation,
        policy_source="ISMCTS_VISITS",
        policy_target=(1.0,),
        selected_candidate_id=candidate_id,
        selected_selection="advance",
        terminal_winner="RED",
        value_target=1,
    )


class _Registry:
    def __init__(self) -> None:
        self.candidates: dict[str, CandidateManifest] = {}
        self.register_calls = 0

    def candidate(self, digest: str) -> CandidateManifest:
        if digest not in self.candidates:
            raise ValueError("missing candidate")
        return self.candidates[digest]

    def register_candidate(self, source: Path, *, metadata: Any) -> CandidateManifest:
        self.register_calls += 1
        artifact = ModelArtifactManifest.model_validate_json(
            (source / "manifest.json").read_bytes()
        )
        values: dict[str, Any] = {
            "artifact_digest": artifact.model_digest,
            "generation": metadata.generation,
            "parent_champion_digest": metadata.parent_champion_digest,
            "observation_schema_version": artifact.observation_schema_version,
            "hero_adapter_versions": artifact.hero_adapter_versions,
            "map_schema_version": artifact.map_schema_version,
            "training_data_digests": metadata.training_data_digests,
            "source_revision": metadata.source_revision,
            "source_tree_digest": metadata.source_tree_digest,
            "search_config": metadata.search_config,
            "training_config": metadata.training_config,
            "supported_heroes": artifact.supported_heroes,
            "supported_maps": artifact.supported_maps,
            "supported_game_types": artifact.supported_game_types,
            "offline_metrics": metadata.offline_metrics,
            "arena_results": metadata.arena_results,
            "runtime_format": metadata.runtime_format,
            "runtime_compatibility_version": artifact.runtime_compatibility_version,
        }
        digest = hashlib.sha256(_canonical({"schema_version": 1, **values})).hexdigest()
        candidate = CandidateManifest(digest=digest, **values)
        self.candidates[digest] = candidate
        return candidate


def _fixture(tmp_path: Path, *, fail_training: bool = False, crash_stage: str | None = None) -> Any:
    parent = _parent()
    generation = GenerationConfig(
        generation_id="generation-7",
        parent_model_digest=parent.model_digest,
        parent_generation=6,
        observation_schema_version=3,
        source_revision="revision",
        dirty_tree_hash="clean",
        search_config={"iterations": 4},
        source_config={"recipe": "visits-v1"},
        max_steps=20,
        timeout_seconds=5,
    )
    games = tuple(
        GameSpec(seed, "forgotten_island", "map.json", "QUICK", ("Wasp",), ("Arien",))
        for seed in (20_000, 20_001)
    )
    worker = WorkerSpec(0, generation, games)
    root = tmp_path / "pipeline"
    training = JointTrainingConfig(
        dataset_path=root / "training-sample.jsonl.zst",
        split_manifest_path=root / "split.json",
        checkpoint_path=root / "training.pt",
        run_manifest_path=root / "training.json",
        artifact_path=root / "artifact",
        seed=20_001,
        validation_fraction=0.5,
        dataset_seed_purpose="training",
    )
    training_identity = {
        key: value
        for key, value in asdict(training).items()
        if key
        not in {
            "dataset_path",
            "split_manifest_path",
            "checkpoint_path",
            "run_manifest_path",
            "artifact_path",
            "cutoff1_dataset_path",
        }
    }
    calls = {"generate": 0, "train": 0, "validate": 0}

    def generate(spec: WorkerSpec) -> WorkerOutput:
        calls["generate"] += 1
        worker_root = root / f"worker-{spec.worker_id}"
        worker_root.mkdir(parents=True, exist_ok=True)
        fragments: list[Path] = []
        receipts: list[CheckpointRow] = []
        for index, game in enumerate(spec.games):
            fragment = worker_root / f"game-{index}.jsonl"
            fragment.write_bytes(canonical_json_bytes(_row(spec, index)) + b"\n")
            fragments.append(fragment)
            receipts.append(
                CheckpointRow(
                    worker_id=spec.worker_id,
                    worker_config_id=spec.worker_config_id,
                    generator_config_id=spec.config.generator_config_id,
                    source_config_id=spec.config.source_config_id,
                    generation_id=spec.config.generation_id,
                    parent_model_digest=spec.config.parent_model_digest,
                    parent_generation=spec.config.parent_generation,
                    observation_schema_version=spec.config.observation_schema_version,
                    game_id=game.game_id(spec.config),
                    world_seed=game.world_seed,
                    fragment_digest=hashlib.sha256(fragment.read_bytes()).hexdigest(),
                    row_count=1,
                    winner="RED",
                    rounds=1,
                    turns=1,
                    steps=1,
                )
            )
        checkpoint = worker_root / "checkpoint.jsonl"
        checkpoint.write_bytes(b"".join(canonical_json_bytes(item) + b"\n" for item in receipts))
        return WorkerOutput(spec.worker_id, checkpoint, tuple(fragments))

    def train(config: JointTrainingConfig) -> JointTrainingResult:
        calls["train"] += 1
        if fail_training:
            raise RuntimeError("training failed")
        dataset = load_joint_dataset(config.dataset_path)
        splits = split_joint_dataset(
            dataset,
            config=JointSplitConfig(
                seed=config.seed,
                validation_fraction=config.validation_fraction,
                seed_purpose=config.dataset_seed_purpose,
            ),
        )
        config.split_manifest_path.write_bytes(splits.manifest.canonical_bytes())
        metrics = {
            "train": {"policy": {"overall": {"count": 1}}},
            "validation": {"policy": {"overall": {"count": 1}}},
        }
        provenance = {
            "dataset_digest": dataset.digest,
            "split_digest": splits.digest,
            "split_manifest": splits.manifest.model_dump(mode="json"),
            "config": training_identity,
            "dataset_row_count": len(dataset.rows),
            "dataset_game_count": len(dataset.game_ids),
            "metrics": metrics,
            "step": 1,
        }
        config.artifact_path.mkdir(parents=True, exist_ok=True)
        provenance_bytes = _canonical(provenance)
        (config.artifact_path / "provenance.json").write_bytes(provenance_bytes)
        artifact = _parent("c" * 64).model_copy(
            update={
                "files": {
                    "provenance.json": ArtifactFile(
                        length=len(provenance_bytes),
                        sha256=hashlib.sha256(provenance_bytes).hexdigest(),
                    )
                },
                "non_executable_files": ("provenance.json",),
            }
        )
        (config.artifact_path / "manifest.json").write_bytes(canonical_json_bytes(artifact))
        config.run_manifest_path.write_bytes(
            _canonical(
                {
                    "schema_version": 1,
                    "status": "SUCCEEDED",
                    "model_digest": artifact.model_digest,
                    "step": 1,
                    "config": training_identity,
                    "provenance": {
                        key: value
                        for key, value in provenance.items()
                        if key not in {"metrics", "step"}
                    },
                    "metrics": metrics,
                }
            )
        )
        return JointTrainingResult("SUCCEEDED", 1, artifact.model_digest)

    def validate(artifact_path: Path, training_manifest_path: Path) -> dict[str, Any]:
        del artifact_path, training_manifest_path
        calls["validate"] += 1
        return {"validation_loss": 0.25, "passed": True}

    crashed: set[str] = set()

    def boundary(stage: str) -> None:
        if stage == crash_stage and stage not in crashed:
            crashed.add(stage)
            raise RuntimeError("simulated crash")

    config = ResumableGenerationPipelineConfig(
        root=root,
        workers=(worker,),
        parent_artifact=parent,
        parent_candidate_digest="d" * 64,
        candidate_generation=7,
        replay_game_count=2,
        replay_seed=20_002,
        replay_snapshot_digest="e" * 64,
        training=training,
        training_identity=training_identity,
        source_revision="revision",
        source_tree_digest="f" * 64,
    )
    registry = _Registry()
    dependencies = PipelineDependencies(generate, validate, train, boundary)
    return config, registry, dependencies, calls


def _pipeline(fixture: Any) -> ResumableGenerationPipeline:
    config, registry, dependencies, _calls = fixture
    return ResumableGenerationPipeline(
        config,
        replay_buffer=ReplayBuffer(),
        registry=registry,
        dependencies=dependencies,
    )


def _rewrite_artifact_provenance(
    config: ResumableGenerationPipelineConfig, provenance: dict[str, Any]
) -> None:
    payload = _canonical(provenance)
    provenance_path = config.training.artifact_path / "provenance.json"
    provenance_path.write_bytes(payload)
    manifest_path = config.training.artifact_path / "manifest.json"
    manifest = ModelArtifactManifest.model_validate_json(manifest_path.read_bytes())
    manifest = manifest.model_copy(
        update={
            "files": {
                **manifest.files,
                "provenance.json": ArtifactFile(
                    length=len(payload), sha256=hashlib.sha256(payload).hexdigest()
                ),
            }
        }
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def _rewrite_checkpoint_for_fragment(output: WorkerOutput, fragment: Path) -> None:
    receipts = [
        CheckpointRow.model_validate_json(line)
        for line in output.checkpoint_path.read_bytes().splitlines()
    ]
    dataset = load_joint_dataset(fragment)
    rewritten = [
        (
            receipt.model_copy(
                update={
                    "fragment_digest": hashlib.sha256(fragment.read_bytes()).hexdigest(),
                    "row_count": len(dataset.rows),
                }
            )
            if receipt.game_id == dataset.game_ids[0]
            else receipt
        )
        for receipt in receipts
    ]
    output.checkpoint_path.write_bytes(
        b"".join(canonical_json_bytes(receipt) + b"\n" for receipt in rewritten)
    )


def test_runs_all_stages_and_resume_skips_completed_valid_products(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = _pipeline(fixture).run()
    journal_before = fixture[0].journal_path.read_bytes()

    second = _pipeline(fixture).run()

    assert second == first
    assert fixture[3] == {"generate": 1, "train": 1, "validate": 1}
    assert fixture[1].register_calls == 1
    assert fixture[0].journal_path.read_bytes() == journal_before
    assert set(json.loads(journal_before)["stages"]) == {
        "SELF_PLAY",
        "RECONCILE",
        "REPLAY_SAMPLE",
        "TRAIN",
        "OFFLINE_VALIDATE",
        "CANDIDATE",
    }
    sample = json.loads(fixture[0].sample_manifest_path.read_bytes())
    split_digest = hashlib.sha256(fixture[0].training.split_manifest_path.read_bytes()).hexdigest()
    assert first.training_data_digests == (sample["dataset_digest"], split_digest)
    assert (
        first.training_config["replay_sample_digest"]
        == hashlib.sha256(fixture[0].sample_manifest_path.read_bytes()).hexdigest()
    )
    assert first.training_config["split_manifest_digest"] == split_digest


@pytest.mark.parametrize(
    "stage",
    ["SELF_PLAY", "RECONCILE", "REPLAY_SAMPLE", "TRAIN", "OFFLINE_VALIDATE", "CANDIDATE"],
)
def test_crash_after_each_published_product_recovers_without_repeating_stage(
    tmp_path: Path, stage: str
) -> None:
    fixture = _fixture(tmp_path, crash_stage=stage)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _pipeline(fixture).run()
    calls_after_crash = dict(fixture[3])
    registrations_after_crash = fixture[1].register_calls

    candidate = _pipeline(fixture).run()

    assert candidate.digest
    if stage == "SELF_PLAY":
        assert fixture[3]["generate"] == calls_after_crash["generate"]
    if stage == "TRAIN":
        assert fixture[3]["train"] == calls_after_crash["train"]
    if stage == "OFFLINE_VALIDATE":
        assert fixture[3]["validate"] == calls_after_crash["validate"]
    if stage == "CANDIDATE":
        assert fixture[1].register_calls == registrations_after_crash


def test_failed_training_is_journaled_and_never_publishes_candidate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, fail_training=True)

    with pytest.raises(RuntimeError, match="training failed"):
        _pipeline(fixture).run()

    journal = json.loads(fixture[0].journal_path.read_bytes())
    assert journal["stages"]["TRAIN"]["status"] == "FAILED"
    assert fixture[1].register_calls == 0
    assert fixture[3]["validate"] == 0


def test_changed_inputs_and_corrupted_completed_products_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _pipeline(fixture).run()

    changed = replace(fixture[0], replay_snapshot_digest="9" * 64)
    with pytest.raises(StalePipelineProductError, match="inputs changed"):
        ResumableGenerationPipeline(
            changed,
            replay_buffer=ReplayBuffer(),
            registry=fixture[1],
            dependencies=fixture[2],
        )

    fixture[0].validation_path.write_text("{}")
    with pytest.raises(StalePipelineProductError, match="OFFLINE_VALIDATE product"):
        _pipeline(fixture).run()


def test_real_pipeline_registry_and_arena_run_unattended_and_resume_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    base_config, _fake_registry, dependencies, calls = fixture

    def fake_load(path: Path, *, requirements: object) -> Any:
        del requirements
        return LoadedModelArtifact(
            manifest=ModelArtifactManifest.model_validate_json(
                (path / "manifest.json").read_bytes()
            ),
            schema=cast(Any, None),
            config=cast(Any, None),
            model=cast(Any, None),
        )

    monkeypatch.setattr("automata.training.registry.load_model_artifact", fake_load)
    registry = ChampionRegistry(tmp_path / "registry", requirements=object())  # type: ignore[arg-type]
    genesis_source = tmp_path / "genesis"
    genesis_source.mkdir()
    (genesis_source / "manifest.json").write_bytes(
        canonical_json_bytes(base_config.parent_artifact)
    )
    genesis = registry.register_candidate(
        genesis_source,
        metadata=CandidateMetadata(
            generation=0,
            parent_champion_digest=None,
            training_data_digests=("1" * 64,),
            source_revision="revision",
            source_tree_digest="f" * 64,
            search_config={},
            training_config={},
            offline_metrics={"bootstrap": True},
            arena_results={"bootstrap": True},
        ),
    )

    def initialize(target: Any) -> None:
        target.promote(genesis.digest)

    worker = base_config.workers[0]
    worker = replace(
        worker,
        config=replace(worker.config, generation_id="generation-1", parent_generation=0),
    )
    config = replace(
        base_config,
        workers=(worker,),
        parent_candidate_digest=genesis.digest,
        candidate_generation=1,
    )
    pipeline = ResumableGenerationPipeline(
        config,
        replay_buffer=ReplayBuffer(),
        registry=registry,
        dependencies=dependencies,
    )
    gates = PromotionGateConfig(
        max_timeout_or_max_step_rate=0.0,
        latency_slos=(LatencySLO("standard", 100.0, 10.0),),
        practical_margin=0.0,
        required_strata=("all",),
        max_stratum_regression=0.0,
        required_artifact_loads=2,
        expected_artifact_digest="c" * 64,
    )

    def protocol(seeds: tuple[int, ...]) -> EvaluationProtocol:
        return EvaluationProtocol(
            agent_a=AgentSpec("candidate", "fake"),
            agent_b=AgentSpec("champion", "fake"),
            red_heroes=("Wasp",),
            blue_heroes=("Arien",),
            world_seeds=seeds,
            map_path="map.json",
            game_type="QUICK",
            max_steps=20,
            source_revision="revision",
            dirty_tree_hash="clean",
        )

    arena_calls = 0

    def arena(candidate: CandidateManifest, champion: Any, stage_dir: Path) -> Any:
        nonlocal arena_calls
        arena_calls += 1
        plan = SequentialPlan((SequentialBoundary(pair_count=10, alpha=0.4),))
        arena_config = ArenaConfig(
            candidate=ArtifactIdentity(candidate.digest, candidate.artifact_digest),
            champion=ArtifactIdentity(champion.manifest.digest, champion.manifest.artifact_digest),
            candidate_agent=AgentSpec("candidate", "fake"),
            champion_agent=AgentSpec("champion", "fake"),
            smoke=ArenaStageConfig(ArenaStage.SMOKE, protocol((1,)), stage_dir / "smoke.jsonl"),
            screen=ArenaStageConfig(
                ArenaStage.SCREEN, protocol(tuple(range(10))), stage_dir / "screen.jsonl", plan
            ),
            promotion=ArenaStageConfig(
                ArenaStage.PROMOTION,
                protocol(tuple(range(10))),
                stage_dir / "promotion.jsonl",
                plan,
            ),
            promotion_gates=gates,
        )

        def run_case(case: GameCase) -> EvaluationGameResult:
            return EvaluationGameResult(
                case.case_id, case.world_seed, case.a_side, case.a_side, 1, 1, "game_over"
            )

        return run_arena(
            arena_config,
            run_cases={stage: run_case for stage in ArenaStage},
            operational_evidence=ArenaOperationalEvidence(
                0,
                0,
                0,
                {"standard": (1.0,)},
                {"all": 1.0},
                (ArtifactLoad("c" * 64, None), ArtifactLoad("c" * 64, None)),
            ),
        )

    work_dir = tmp_path / "iteration"
    first = run_policy_iteration(
        work_dir=work_dir,
        registry=registry,
        generation_pipeline=pipeline,
        arena_runner=arena,
        promotion_gate_config=gates,
        initialize_champion=initialize,
    )
    manifest_bytes = (work_dir / "generation-manifest.json").read_bytes()
    second = run_policy_iteration(
        work_dir=work_dir,
        registry=registry,
        generation_pipeline=pipeline,
        arena_runner=arena,
        promotion_gate_config=gates,
    )

    assert first.status is second.status is PolicyIterationStatus.PROMOTED
    assert first.manifest == second.manifest
    assert (work_dir / "generation-manifest.json").read_bytes() == manifest_bytes
    assert arena_calls == 1
    assert calls == {"generate": 1, "train": 1, "validate": 1}
    promoted = registry.load_champion().manifest
    assert promoted.digest == first.candidate.digest
    arena_evidence = cast(dict[str, Any], promoted.arena_results["arena_result"])
    promotion_metrics = cast(dict[str, Any], arena_evidence["promotion_metrics"])
    assert promotion_metrics["total_games"] == 42


def test_train_recovery_rejects_stale_artifact_provenance_after_crash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, crash_stage="TRAIN")
    with pytest.raises(RuntimeError, match="simulated crash"):
        _pipeline(fixture).run()
    provenance_path = fixture[0].training.artifact_path / "provenance.json"
    provenance = json.loads(provenance_path.read_bytes())
    provenance["dataset_digest"] = "0" * 64
    _rewrite_artifact_provenance(fixture[0], provenance)

    with pytest.raises(StalePipelineProductError, match="TRAIN product"):
        _pipeline(fixture).run()


def test_offline_validation_rejects_split_manifest_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, crash_stage="TRAIN")
    with pytest.raises(RuntimeError, match="simulated crash"):
        _pipeline(fixture).run()
    split_path = fixture[0].training.split_manifest_path
    split = json.loads(split_path.read_bytes())
    for membership in split["memberships"]:
        membership["split"] = "validation" if membership["split"] == "train" else "train"
    split_path.write_bytes(_canonical(split))

    with pytest.raises(StalePipelineProductError, match="TRAIN product"):
        _pipeline(fixture).run()
    assert fixture[3]["validate"] == 0


def test_reconcile_rejects_conflicting_duplicate_decisions_across_workers(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config, registry, dependencies, calls = fixture
    game = config.workers[0].games[0]
    workers = (
        WorkerSpec(0, config.workers[0].config, (game,)),
        WorkerSpec(1, config.workers[0].config, (game,)),
    )
    original_generate = dependencies.generate_worker

    def generate(spec: WorkerSpec) -> WorkerOutput:
        output = original_generate(spec)
        if spec.worker_id == 1:
            fragment = output.fragment_paths[0]
            conflicting = _row(spec).model_copy(
                update={"terminal_winner": "BLUE", "value_target": -1}
            )
            fragment.write_bytes(canonical_json_bytes(conflicting) + b"\n")
            _rewrite_checkpoint_for_fragment(output, fragment)
        return output

    changed = replace(config, workers=workers, replay_game_count=1)
    conflicting_fixture = (
        changed,
        registry,
        PipelineDependencies(generate, dependencies.offline_validate, dependencies.train_candidate),
        calls,
    )
    with pytest.raises(ValueError, match="conflicting self-play decision"):
        _pipeline(conflicting_fixture).run()


def test_reconcile_rejects_non_contiguous_per_game_indices(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config, registry, dependencies, calls = fixture
    original_generate = dependencies.generate_worker

    def generate(spec: WorkerSpec) -> WorkerOutput:
        output = original_generate(spec)
        fragment = output.fragment_paths[0]
        fragment.write_bytes(canonical_json_bytes(_row(spec, decision_index=1)) + b"\n")
        # Loading the complete fragment is itself the public dataset-contract check.
        receipts = [
            CheckpointRow.model_validate_json(line)
            for line in output.checkpoint_path.read_bytes().splitlines()
        ]
        rewritten = [
            (
                receipt.model_copy(
                    update={"fragment_digest": hashlib.sha256(fragment.read_bytes()).hexdigest()}
                )
                if receipt.game_id == spec.games[0].game_id(spec.config)
                else receipt
            )
            for receipt in receipts
        ]
        output.checkpoint_path.write_bytes(
            b"".join(canonical_json_bytes(receipt) + b"\n" for receipt in rewritten)
        )
        return output

    invalid_fixture = (
        config,
        registry,
        PipelineDependencies(generate, dependencies.offline_validate, dependencies.train_candidate),
        calls,
    )
    with pytest.raises(ValueError, match="decision indexes are not contiguous"):
        _pipeline(invalid_fixture).run()


def test_multi_worker_generation_requires_exact_fragment_coverage(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config, registry, dependencies, calls = fixture
    games = config.workers[0].games
    workers = tuple(
        WorkerSpec(worker_id, config.workers[0].config, (game,))
        for worker_id, game in enumerate(games)
    )
    multi = replace(config, workers=workers)
    valid_fixture = (multi, registry, dependencies, calls)
    assert _pipeline(valid_fixture).run().digest


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_multi_worker_generation_rejects_missing_or_duplicate_fragments(
    tmp_path: Path, fault: str
) -> None:
    fixture = _fixture(tmp_path)
    config, registry, dependencies, calls = fixture
    original_generate = dependencies.generate_worker

    def generate(spec: WorkerSpec) -> WorkerOutput:
        output = original_generate(spec)
        if spec.worker_id != 0:
            return output
        fragments = (
            output.fragment_paths[:-1]
            if fault == "missing"
            else (*output.fragment_paths, output.fragment_paths[0])
        )
        return WorkerOutput(output.worker_id, output.checkpoint_path, fragments)

    invalid_fixture = (
        config,
        registry,
        PipelineDependencies(generate, dependencies.offline_validate, dependencies.train_candidate),
        calls,
    )
    with pytest.raises(ValueError, match="fragment"):
        _pipeline(invalid_fixture).run()
