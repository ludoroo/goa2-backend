"""Tests for trajectory recording (Seam 4 / T4).

Assert a recorded game is behavior-neutral, emits a decision row per decision
plus one outcome, streams JSONL to disk, and captures restorable snapshots.
"""

from __future__ import annotations

import json

import pytest

from automata.agents.heuristic_agent import HeuristicAgent
from automata.runtime.driver import BotDecision, DecisionKind
from automata.runtime.effects import register_all_effects
from automata.runtime.harness import DEFAULT_MAP, _record_decision, run_game
from automata.runtime.trajectory import InMemoryRecorder, JsonlRecorder
from goa2.domain.input import InputOption, InputRequest, InputRequestType
from goa2.domain.state import GameState
from goa2.domain.types import HeroID
from goa2.engine.setup import GameSetup

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]


def _agents() -> dict[str, HeuristicAgent]:
    a = HeuristicAgent(1)
    return {
        "hero_wasp": a,
        "hero_xargatha": a,
        "hero_arien": a,
        "hero_brogan": a,
    }


def test_inmemory_recorder_captures_decisions_and_outcome() -> None:
    register_all_effects()
    rec = InMemoryRecorder()
    r = run_game(RED, BLUE, _agents(), seed=3, max_steps=400, recorder=rec)
    # At least a few decisions were made, all indexed contiguously.
    assert len(rec.decisions) >= 3
    assert [d["decision_index"] for d in rec.decisions] == list(range(len(rec.decisions)))
    # Every decision is CARD or INPUT and carries a state snapshot + legal keys.
    for d in rec.decisions:
        assert d["decision_kind"] in ("CARD", "INPUT")
        assert isinstance(d["state"], dict)
        assert isinstance(d["legal_keys"], list)
    # Exactly one outcome, consistent with the run result.
    assert rec.outcome is not None
    assert rec.outcome["winner"] == r.winner
    assert rec.outcome["reason"] == r.reason


def test_recording_does_not_change_the_game() -> None:
    register_all_effects()
    plain = run_game(RED, BLUE, _agents(), seed=5, max_steps=400)
    rec = InMemoryRecorder()
    recorded = run_game(RED, BLUE, _agents(), seed=5, max_steps=400, recorder=rec)
    assert (plain.winner, plain.rounds, plain.steps) == (
        recorded.winner,
        recorded.rounds,
        recorded.steps,
    )


@pytest.mark.parametrize(
    ("option_ids", "can_skip", "selection", "expected_legal_keys"),
    [
        (["A", "B"], True, "SKIP", ["A", "B", "SKIP"]),
        (["A"], False, "A", ["A"]),
        (["SKIP"], True, "SKIP", ["SKIP"]),
    ],
)
def test_recorded_skip_legality_matches_request(
    option_ids: list[str],
    can_skip: bool,
    selection: str,
    expected_legal_keys: list[str],
) -> None:
    """Keep this recorder seam deterministic for the serialized row contract.

    Full games cannot reliably surface all three engine-owned request shapes:
    skippable, non-skippable, and an explicit SKIP option.
    """
    state = GameSetup.create_game(DEFAULT_MAP, ["Wasp"], ["Arien"], game_type="QUICK", seed=1)
    request = InputRequest(
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_wasp",
        options=[InputOption(id=value, text=value) for value in option_ids],
        can_skip=can_skip,
    )
    decision = BotDecision(
        kind=DecisionKind.INPUT,
        hero_id=HeroID("hero_wasp"),
        request=request,
        selection=selection,
    )
    recorder = InMemoryRecorder()

    _record_decision(recorder, state, decision)

    assert recorder.decisions[0]["chosen_key"] == selection
    assert recorder.decisions[0]["legal_keys"] == expected_legal_keys


def test_jsonl_recorder_streams_and_reloads(tmp_path) -> None:
    register_all_effects()
    path = tmp_path / "traj.jsonl"
    with JsonlRecorder(path, game_id="g0") as rec:
        run_game(RED, BLUE, _agents(), seed=3, max_steps=400, recorder=rec)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    decisions = [r for r in rows if r["kind"] == "decision"]
    outcomes = [r for r in rows if r["kind"] == "outcome"]
    assert len(decisions) >= 3
    assert len(outcomes) == 1
    assert outcomes[0]["game_id"] == "g0"
    assert outcomes[0]["decisions"] == len(decisions)
    # decision_index is contiguous and game_id is stamped on every row.
    assert [d["decision_index"] for d in decisions] == list(range(len(decisions)))
    assert all(d["game_id"] == "g0" for d in decisions)


def test_snapshot_roundtrips_into_gamestate(tmp_path) -> None:
    register_all_effects()
    rec = InMemoryRecorder()
    run_game(RED, BLUE, _agents(), seed=3, max_steps=200, recorder=rec)
    assert rec.decisions
    snap = rec.decisions[0]["state"]
    # A recorded snapshot must validate back into a GameState with the same
    # round/turn — the contract learned-model data loaders rely on.
    restored = GameState.model_validate(snap)
    assert isinstance(restored, GameState)
    assert restored.round == snap["round"]
    assert restored.turn == snap["turn"]
