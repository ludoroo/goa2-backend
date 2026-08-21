"""Behavioral contract for learned-policy targeted evaluation integration."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, cast

import pytest

from automata.evaluation import cli
from automata.evaluation.features import FEATURE_NAMES
from automata.evaluation.learned_value import LearnedValue
from automata.evaluation.protocol import (
    AgentSpec,
    EvaluationProtocol,
    EvaluationSummary,
)
from automata.search import POLICY_FEATURE_SCHEMA_ID
from automata.search.learned_policy import LearnedPolicy

RED = ("Wasp", "Xargatha")
BLUE = ("Arien", "Brogan")


def _policy_artifact(*, base_score: float = 0.25) -> dict[str, Any]:
    return {
        "model_version": "gbm-policy-v1",
        "schema_version": 1,
        "policy_feature_schema_id": POLICY_FEATURE_SCHEMA_ID,
        "red_roster": list(RED),
        "blue_roster": list(BLUE),
        "feature_names": ["candidate.finish"],
        "base_score": base_score,
        "learning_rate": 0.5,
        "trees": [{"root": 0, "nodes": [{"value": 1.0}]}],
    }


def _write_policy(path: Path, *, base_score: float = 0.25) -> Path:
    path.write_text(json.dumps(_policy_artifact(base_score=base_score)), encoding="utf-8")
    return path


def _write_value(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "model_version": "logistic-v1",
                "schema_version": 1,
                "red_roster": list(RED),
                "blue_roster": list(BLUE),
                "feature_names": list(FEATURE_NAMES),
                "feature_means": [0.0] * len(FEATURE_NAMES),
                "feature_scales": [1.0] * len(FEATURE_NAMES),
                "coefficients": [0.0] * len(FEATURE_NAMES),
                "intercept": 0.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def _protocol(agent_a: AgentSpec, agent_b: AgentSpec | None = None) -> EvaluationProtocol:
    return EvaluationProtocol(
        agent_a=agent_a,
        agent_b=agent_b or AgentSpec("B", "heuristic"),
        red_heroes=RED,
        blue_heroes=BLUE,
        world_seeds=(0,),
        map_path="src/goa2/data/maps/forgotten_island.json",
        game_type="QUICK",
        max_steps=20_000,
        source_revision="test-rev",
        dirty_tree_hash="clean",
    )


def _capture_targeted(monkeypatch: pytest.MonkeyPatch) -> list[EvaluationProtocol]:
    captured: list[EvaluationProtocol] = []

    def fake_run(protocol: EvaluationProtocol, **_: Any) -> list[Any]:
        captured.append(protocol)
        return []

    monkeypatch.setattr(cli, "source_identity", lambda repo_root=None: ("test-rev", "clean"))
    monkeypatch.setattr(cli, "run_protocol", fake_run)
    monkeypatch.setattr(cli, "summarize", lambda observations: EvaluationSummary())
    return captured


def _build_policy_spec(path: Path) -> AgentSpec:
    builder = cast(Any, cli._build_agent_spec)
    return builder(
        label="A",
        kind="ismcts",
        iterations=4,
        cutoff_rounds=1,
        uct_c=None,
        puct_c=None,
        no_prior=False,
        value_model=None,
        policy_model=str(path),
    )


def test_targeted_cli_accepts_policy_models_for_either_ismcts_side(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = _write_policy(tmp_path / "policy.json")
    captured = _capture_targeted(monkeypatch)

    cli.main(
        [
            "--agent-a",
            "ismcts",
            "--agent-b",
            "ismcts",
            "--a-policy-model",
            str(artifact),
            "--b-policy-model",
            str(artifact),
        ]
    )

    assert len(captured) == 1
    assert captured[0].agent_a.params["policy_model_digest"] == LearnedPolicy(artifact).digest
    assert captured[0].agent_b.params["policy_model_digest"] == LearnedPolicy(artifact).digest


@pytest.mark.parametrize(
    "argv, message",
    [
        (
            ["--agent-a", "heuristic", "--agent-b", "ismcts", "--a-policy-model", "x"],
            "valid only",
        ),
        (
            [
                "--agent-a",
                "ismcts",
                "--agent-b",
                "heuristic",
                "--a-policy-model",
                "x",
                "--a-no-prior",
            ],
            "incompatible",
        ),
    ],
)
def test_policy_cli_rejects_non_ismcts_and_disabled_prior_before_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(
        cli, "run_protocol", lambda *args, **kwargs: pytest.fail("game evaluation started")
    )
    with pytest.raises(SystemExit):
        cli.main(argv)
    assert message in capsys.readouterr().err.lower()


def test_build_spec_eagerly_validates_policy_artifact(tmp_path: Path) -> None:
    malformed = tmp_path / "policy.json"
    malformed.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"(?i)policy.*(missing|required|artifact)"):
        _build_policy_spec(malformed)


def test_policy_path_is_runtime_only_but_behavior_digest_is_identity(tmp_path: Path) -> None:
    first_path = _write_policy(tmp_path / "first.json")
    second_path = _write_policy(tmp_path / "second.json")
    digest = LearnedPolicy(first_path).digest
    first = AgentSpec(
        "A",
        "ismcts",
        {"iterations": 4, "policy_model_path": str(first_path), "policy_model_digest": digest},
    )
    relocated = AgentSpec(
        "A",
        "ismcts",
        {"iterations": 4, "policy_model_path": str(second_path), "policy_model_digest": digest},
    )
    changed = AgentSpec(
        "A",
        "ismcts",
        {"iterations": 4, "policy_model_path": str(second_path), "policy_model_digest": "f" * 64},
    )

    assert first.identity() == relocated.identity()
    assert _protocol(first).identity_digest() == _protocol(relocated).identity_digest()
    assert _protocol(first).identity_digest() != _protocol(changed).identity_digest()


def test_spawn_build_reloads_policy_and_injects_all_optional_collaborators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy_path = _write_policy(tmp_path / "policy.json")
    value_path = _write_value(tmp_path / "value.json")
    telemetry_path = tmp_path / "cutoffs.jsonl"
    captured: dict[str, Any] = {}

    class AgentSpy:
        def __init__(
            self,
            config: Any,
            *,
            prior: Any = None,
            value_fn: Any = None,
            cutoff_observer: Any = None,
        ) -> None:
            captured.update(
                config=config,
                prior=prior,
                value_fn=value_fn,
                cutoff_observer=cutoff_observer,
            )

    spec = AgentSpec(
        "A",
        "ismcts",
        {
            "iterations": 4,
            "cutoff_rounds": 1,
            "use_prior": True,
            "policy_model_path": str(policy_path),
            "policy_model_digest": LearnedPolicy(policy_path).digest,
            "value_model_path": str(value_path),
            "value_model_digest": LearnedValue(value_path).digest,
            "cutoff_telemetry_path": str(telemetry_path),
        },
    )
    monkeypatch.setattr(cli, "ISMCTSAgent", AgentSpy)

    agent = cli.build_agent(spec, seed=91, case_metadata={"case_id": "case-1"})

    assert agent is not None
    assert captured["config"].seed == 91
    assert isinstance(captured["prior"], LearnedPolicy)
    assert isinstance(captured["value_fn"], LearnedValue)
    assert captured["cutoff_observer"] is not None


@pytest.mark.parametrize(
    "params",
    [
        {"policy_model_path": "policy.json"},
        {"policy_model_digest": "a" * 64},
    ],
)
def test_spawn_build_rejects_partial_policy_metadata_clearly(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"(?i)policy.*metadata.*incomplete"):
        cli.build_agent(AgentSpec("A", "ismcts", params), seed=1)


def test_spawn_build_rejects_policy_changed_after_protocol_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _write_policy(tmp_path / "policy.json")
    spec = AgentSpec(
        "A",
        "ismcts",
        {
            "policy_model_path": str(path),
            "policy_model_digest": LearnedPolicy(path).digest,
        },
    )
    _write_policy(path, base_score=9.0)
    monkeypatch.setattr(
        cli, "ISMCTSAgent", lambda *args, **kwargs: pytest.fail("agent constructed")
    )

    with pytest.raises(ValueError, match=r"(?i)(policy|artifact).*(changed|digest)"):
        cli.build_agent(spec, seed=1)


def test_targeted_header_distinguishes_learned_and_heuristic_priors(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = _write_policy(tmp_path / "policy.json")
    digest = LearnedPolicy(path).digest
    learned = AgentSpec(
        "A",
        "ismcts",
        {"policy_model_path": str(path), "policy_model_digest": digest},
    )

    cli._print_targeted_header(
        protocol=_protocol(learned, AgentSpec("B", "ismcts", {"use_prior": True})),
        checkpoint=tmp_path / "checkpoint.jsonl",
        paired_seeds=1,
    )

    output = capsys.readouterr().out.lower()
    assert "learned policy" in output and digest[:12] in output
    assert "heuristic policy" in output


def test_learned_and_heuristic_prior_specs_compare_at_equal_budget_and_pickle(
    tmp_path: Path,
) -> None:
    path = _write_policy(tmp_path / "policy.json")
    learned = _build_policy_spec(path)
    builder = cast(Any, cli._build_agent_spec)
    heuristic = builder(
        label="A",
        kind="ismcts",
        iterations=4,
        cutoff_rounds=1,
        uct_c=None,
        puct_c=None,
        no_prior=False,
        value_model=None,
        policy_model=None,
    )

    assert learned.params["iterations"] == heuristic.params["iterations"] == 4
    assert _protocol(learned).identity_digest() != _protocol(heuristic).identity_digest()
    assert not any(isinstance(value, LearnedPolicy) for value in learned.params.values())
    pickle.dumps(cli.build_case_runner(_protocol(learned, heuristic)))
