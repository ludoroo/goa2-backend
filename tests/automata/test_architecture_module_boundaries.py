"""Public module boundaries for Automata contracts and implementations."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


def test_contract_and_implementation_modules_have_explicit_homes() -> None:
    expected = (
        "automata.agents.contracts",
        "automata.models.contracts.observation",
        "automata.models.contracts.candidates",
        "automata.models.contracts.inference",
        "automata.models.contracts.artifacts",
        "automata.models.contracts.compatibility",
        "automata.models.contracts.serialization",
        "automata.models.shared_encoder.artifacts",
        "automata.models.shared_encoder.artifacts.manifest",
        "automata.models.shared_encoder.schema.feature_schema",
        "automata.observation.decision_encoder",
        "automata.observation.graph.encoder",
        "automata.observation.hero_adapters.protocol",
        "automata.observation.hero_adapters.registry",
        "automata.harness.game_runner",
        "automata.harness.trajectory",
        "automata.search.contracts",
        "automata.search.fallback",
        "automata.search.heuristic",
        "automata.search.ismcts.engine",
        "automata.search.ismcts.strategy",
        "automata.search.learned",
        "automata.training.contracts.experiment",
        "automata.training.curriculum",
        "automata.training.dataset",
        "automata.training.experiments.phase0",
        "automata.training.generation",
        "automata.training.generation_pipeline",
        "automata.training.io",
        "automata.training.losses",
        "automata.training.metrics",
        "automata.training.policy_iteration",
        "automata.training.registry",
        "automata.training.replay_buffer",
        "automata.training.search_targets",
        "automata.training.splits",
        "automata.training.trainer",
    )
    retired = (
        "automata.agents.base",
        "automata.models.artifact",
        "automata.models.runtime",
        "automata.models.tensor_schema",
        "automata.models.shared_encoder.artifact",
        "automata.observation.adapters",
        "automata.observation.decision",
        "automata.observation.encoder",
        "automata.search.components",
        "automata.search.prior",
        "automata.search.strategy",
        "automata.evaluation.curriculum",
        "automata.evaluation.io",
        "automata.evaluation.generation",
        "automata.evaluation.generation_pipeline",
        "automata.evaluation.joint_dataset",
        "automata.evaluation.joint_losses",
        "automata.evaluation.joint_metrics",
        "automata.evaluation.joint_splits",
        "automata.evaluation.registry",
        "automata.evaluation.replay_buffer",
        "automata.evaluation.train_joint",
        "automata.models.shared_encoder.experiment",
        "automata.runtime.harness",
        "automata.runtime.trajectory",
        "automata.training.game_runner",
        "automata.training.trajectory",
        "automata.search.action_observation",
        "automata.search.observation",
    )

    assert all(importlib.util.find_spec(name) is not None for name in expected)
    assert all(importlib.util.find_spec(name) is None for name in retired)


def test_model_contract_definitions_are_owned_by_their_public_modules() -> None:
    from automata.models.contracts import artifacts, candidates, inference, observation
    from automata.models.shared_encoder.artifacts import manifest

    owned_types = {
        observation: (
            observation.Viewer,
            observation.PublicSnapshot,
            observation.ObservationToken,
            observation.ObservationRelationship,
            observation.LearnedObservation,
        ),
        candidates: (
            candidates.FinishCandidateID,
            candidates.SkipCandidateID,
            candidates.CardCandidateID,
            candidates.UnitCandidateID,
            candidates.HexCandidateID,
            candidates.NumberCandidateID,
            candidates.OptionCandidateID,
            candidates.ActionCandidateID,
            candidates.EntityCandidateID,
            candidates.EncodedCandidate,
            candidates.DecisionObservation,
        ),
        inference: (inference.PolicyValueOutput, inference.SearchOutcome),
        artifacts: (
            artifacts.ArtifactError,
            artifacts.ArtifactScope,
            artifacts.RuntimeRequirements,
        ),
        manifest: (
            manifest.ArtifactFile,
            manifest.ArtifactTensor,
            manifest.ModelArtifactManifest,
        ),
    }

    for module, types in owned_types.items():
        assert all(type_.__module__ == module.__name__ for type_ in types)
    assert importlib.util.find_spec("automata.models.contracts._definitions") is None


def test_concrete_artifact_manifest_is_not_exported_from_generic_model_packages() -> None:
    import automata.models as models
    import automata.models.contracts as contracts
    from automata.models.contracts import artifacts

    concrete_names = ("ArtifactFile", "ArtifactTensor", "ModelArtifactManifest")

    for module in (models, contracts, artifacts):
        assert all(not hasattr(module, name) for name in concrete_names)


def test_observation_implementation_does_not_import_concrete_search() -> None:
    root = Path(__file__).parents[2] / "src" / "automata" / "observation"
    imports: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        imports.update(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )

    assert not any(name.startswith("automata.search.ismcts") for name in imports)


def test_product_runtime_and_search_do_not_depend_on_offline_packages() -> None:
    root = Path(__file__).parents[2] / "src" / "automata"
    forbidden = ("automata.training", "automata.evaluation", "automata.harness")

    for package in ("runtime", "search"):
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text())
            imports = {
                node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            }
            imports.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            assert not any(name.startswith(forbidden) for name in imports), path


def test_training_and_evaluation_only_couple_through_policy_iteration() -> None:
    root = Path(__file__).parents[2] / "src" / "automata"

    for package, forbidden in (
        ("evaluation", "automata.training"),
        ("training", "automata.evaluation"),
    ):
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text())
            imports = {
                node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            }
            imports.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            if package == "training" and path.name == "policy_iteration.py":
                continue
            assert not any(name.startswith(forbidden) for name in imports), path
