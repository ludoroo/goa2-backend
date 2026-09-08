from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from automata.agents.ismcts_agent import ISMCTSAgent
from automata.search.contracts import LeafMode
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


def test_hh_factory_does_not_touch_runtime_cache_or_torch(tmp_path: Path) -> None:
    class ExplodingCache:
        def get(self, *args, **kwargs):
            raise AssertionError("H/H must not load a neural runtime")

    state = GameSetup.create_game(MAP, ["Razzle"], ["Arien"], game_type="QUICK", seed=3)
    agent = agent_for_spec(
        BotSpec(kind="ismcts", search=SearchSettings()),
        state=state,
        artifact_root=tmp_path,
        runtime_cache=ExplodingCache(),
    )

    assert isinstance(agent, ISMCTSAgent)
    assert agent._cfg.leaf_mode is LeafMode.IMMEDIATE


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

    agent_for_spec(
        spec,
        state=state,
        artifact_root=tmp_path,
        runtime_cache=cache,
    )

    assert len(cache.calls) == 1
    assert cache.calls[0][1]["expected_digest"] == "a" * 64
