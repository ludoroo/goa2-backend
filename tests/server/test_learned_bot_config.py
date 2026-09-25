from __future__ import annotations

import builtins
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from automata.agents.ismcts_agent import ISMCTSAgent
from automata.search.continuation import AgentContinuationPolicy, ArgmaxContinuationPolicy
from automata.search.contracts import LeafMode
from automata.search.heuristic import HeuristicLeafEvaluator, HeuristicPrior
from goa2.engine.setup import GameSetup
from goa2.server.bot_factory import agent_for_spec
from goa2.server.bot_models import BotSpec, ModelArtifactSpec, SearchSettings
from goa2.server.models import CreateBotSpec

MAP = "src/goa2/data/maps/forgotten_island.json"


def _artifact() -> ModelArtifactSpec:
    return ModelArtifactSpec(reference="champions/joint-v1", digest="a" * 64)


@pytest.mark.parametrize(
    ("policy_source", "value_source"),
    [
        ("heuristic", "heuristic"),
        ("learned", "heuristic"),
        ("heuristic", "learned"),
        ("learned", "learned"),
    ],
)
def test_ismcts_api_and_persistence_accept_independent_component_matrix(
    policy_source: str, value_source: str
) -> None:
    artifact = _artifact() if "learned" in {policy_source, value_source} else None
    request = CreateBotSpec.model_validate(
        {
            "kind": "ismcts",
            "search": {
                "policy_source": policy_source,
                "value_source": value_source,
                "leaf_mode": "immediate",
                "horizon": 2,
                "artifact": artifact.model_dump() if artifact else None,
            },
        }
    )
    persisted = BotSpec(kind=request.kind, search=request.search)

    assert persisted.search is not None
    assert persisted.search.policy_source == policy_source
    assert persisted.model_dump(mode="json")["search"]["artifact"] == (
        artifact.model_dump(mode="json") if artifact else None
    )


def test_learned_source_requires_one_pinned_artifact_but_hh_does_not() -> None:
    SearchSettings(policy_source="heuristic", value_source="heuristic")
    with pytest.raises(ValidationError, match="artifact"):
        SearchSettings(policy_source="learned", value_source="heuristic")


@pytest.mark.parametrize(
    "reference", ["champions/model\x00", "champions/model\n", "champions/\x1fmodel"]
)
def test_artifact_reference_rejects_control_characters(reference: str) -> None:
    with pytest.raises(ValidationError, match="safe relative path"):
        ModelArtifactSpec(reference=reference, digest="a" * 64)


def test_hh_factory_does_not_touch_runtime_cache_or_torch(tmp_path: Path) -> None:
    class ExplodingCache:
        def get(self, *args, **kwargs):
            raise AssertionError("H/H must not load a learned-model runtime")

    state = GameSetup.create_game(MAP, ["Razzle"], ["Arien"], game_type="QUICK", seed=3)
    agent = agent_for_spec(
        BotSpec(kind="ismcts", search=SearchSettings()),
        state=state,
        artifact_root=tmp_path,
        runtime_cache=ExplodingCache(),
    )

    assert isinstance(agent, ISMCTSAgent)
    assert agent._cfg.leaf_mode is LeafMode.BOUNDED_CONTINUATION
    assert isinstance(agent._continuation_policy, AgentContinuationPolicy)


@pytest.mark.parametrize("failure", ["missing", "file", "outside_symlink"])
def test_invalid_artifact_path_logs_and_short_circuits_to_heuristics(
    failure: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    candidate = root / _artifact().reference
    candidate.parent.mkdir()
    if failure == "file":
        candidate.write_text("not a model")
    elif failure == "outside_symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        candidate.symlink_to(outside, target_is_directory=True)

    state = GameSetup.create_game(MAP, ["Razzle"], ["Arien"], game_type="QUICK", seed=3)
    spec = BotSpec(
        kind="ismcts",
        search=SearchSettings(
            policy_source="learned",
            value_source="learned",
            artifact=_artifact(),
        ),
    )

    with caplog.at_level(logging.WARNING, logger="goa2.server.bot_factory"):
        agent = agent_for_spec(spec, state=state, artifact_root=root)

    assert isinstance(agent._prior, HeuristicPrior)
    assert isinstance(agent._leaf_evaluator, HeuristicLeafEvaluator)
    assert "falling back to heuristic" in caplog.text.lower()


def test_missing_torch_logs_and_falls_back_without_failing_agent_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    artifact = tmp_path / "champions" / "joint-v1"
    artifact.mkdir(parents=True)
    real_import = builtins.__import__

    def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "automata.models.shared_encoder.serving":
            raise ImportError("No module named 'torch'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    state = GameSetup.create_game(MAP, ["Razzle"], ["Arien"], game_type="QUICK", seed=3)
    spec = BotSpec(
        kind="ismcts",
        search=SearchSettings(
            policy_source="learned",
            value_source="heuristic",
            artifact=_artifact(),
        ),
    )

    with caplog.at_level(logging.WARNING, logger="goa2.server.bot_factory"):
        agent = agent_for_spec(spec, state=state, artifact_root=tmp_path)

    assert isinstance(agent._prior, HeuristicPrior)
    assert isinstance(agent._leaf_evaluator, HeuristicLeafEvaluator)
    assert "torch" in caplog.text


def test_runtime_cache_programmer_value_error_is_not_treated_as_artifact_unavailability(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "champions" / "joint-v1"
    artifact.mkdir(parents=True)

    class InvalidCacheCall:
        def get(self, *args, **kwargs):
            raise ValueError("programmer misuse")

    state = GameSetup.create_game(MAP, ["Razzle"], ["Arien"], game_type="QUICK", seed=3)
    spec = BotSpec(
        kind="ismcts",
        search=SearchSettings(
            policy_source="learned",
            value_source="heuristic",
            artifact=_artifact(),
        ),
    )

    with pytest.raises(ValueError, match="programmer misuse"):
        agent_for_spec(
            spec,
            state=state,
            artifact_root=tmp_path,
            runtime_cache=InvalidCacheCall(),
        )


def test_ll_factory_loads_one_shared_runtime_for_both_components(tmp_path: Path) -> None:
    artifact = tmp_path / "champions" / "joint-v1"
    artifact.mkdir(parents=True)

    class Cache:
        def __init__(self) -> None:
            self.calls = []
            self.runtime = object()

        def get(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.runtime

    cache = Cache()
    state = GameSetup.create_game(MAP, ["Razzle"], ["Arien"], game_type="QUICK", seed=3)
    spec = BotSpec(
        kind="ismcts",
        search=SearchSettings(
            iterations=1,
            policy_source="learned",
            value_source="learned",
            artifact=_artifact(),
        ),
    )

    agent = agent_for_spec(
        spec,
        state=state,
        artifact_root=tmp_path,
        runtime_cache=cache,
    )

    assert len(cache.calls) == 1
    assert cache.calls[0][1]["expected_digest"] == "a" * 64
    assert isinstance(agent, ISMCTSAgent)
    assert isinstance(agent._continuation_policy, ArgmaxContinuationPolicy)
    assert agent._continuation_policy.policy is agent._prior
    assert agent._cfg.root_puct_c is not None and agent._cfg.root_puct_c > 0.0
    assert agent._cfg.root_widening_c is not None
    assert agent._cfg.root_widening_c < agent._cfg.widening_c
    assert agent._cfg.root_widening_alpha == agent._cfg.widening_alpha


def test_learned_value_with_heuristic_policy_keeps_classic_root_defaults(tmp_path: Path) -> None:
    artifact = tmp_path / "champions" / "joint-v1"
    artifact.mkdir(parents=True)

    class Cache:
        def get(self, *args, **kwargs):
            return object()

    state = GameSetup.create_game(MAP, ["Razzle"], ["Arien"], game_type="QUICK", seed=3)
    agent = agent_for_spec(
        BotSpec(
            kind="ismcts",
            search=SearchSettings(
                iterations=1,
                policy_source="heuristic",
                value_source="learned",
                artifact=_artifact(),
            ),
        ),
        state=state,
        artifact_root=tmp_path,
        runtime_cache=Cache(),
    )

    assert isinstance(agent, ISMCTSAgent)
    assert agent._cfg.root_puct_c is None
    assert agent._cfg.root_widening_c is None
    assert agent._cfg.root_widening_alpha is None
