"""Behavioral contract for search-cutoff telemetry."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from automata.search.config import SearchConfig
from automata.search.ismcts import Decision, _rollout
from goa2.domain.input import InputRequest, InputRequestType
from goa2.domain.models import TeamColor


class _Policy:
    def choose_input(self, state: Any, request: InputRequest) -> str:
        return "ok"


class _Sim:
    def __init__(self, *, terminal_after_advance: bool = False) -> None:
        self.state = SimpleNamespace(round=3)
        self.our_team = TeamColor.RED
        self.default_policy = _Policy()
        self._terminal_after_advance = terminal_after_advance

    def advance(self, response: Any) -> Decision:
        self.state.round += 1
        if self._terminal_after_advance:
            return Decision(kind="OVER", winner="RED")
        return Decision(kind="INPUT", request=_request("next"))


def _request(request_id: str = "root") -> InputRequest:
    return InputRequest(
        id=request_id,
        request_type=InputRequestType.SELECT_OPTION,
        player_id="hero_test",
        prompt="choose",
        options=[],
    )


def _cfg() -> SearchConfig:
    return SearchConfig(iterations=1, cutoff_rounds=1, seed=0)


def test_nonterminal_cutoff_observer_receives_active_normalized_value() -> None:
    sim = _Sim()
    seen: list[tuple[Any, TeamColor, float]] = []

    def observer(state: Any, team: TeamColor, active_value: float) -> object:
        seen.append((state, team, active_value))
        return object()  # Observers are diagnostic; their return is ignored.

    reward = _rollout(
        sim,
        Decision(kind="INPUT", request=_request()),
        _cfg(),
        lambda state, team: 0.25,
        cutoff_observer=observer,
    )

    assert seen == [(sim.state, TeamColor.RED, 0.25)]
    assert reward == pytest.approx(0.625)


def test_terminal_rollout_does_not_notify_cutoff_observer() -> None:
    seen: list[object] = []
    reward = _rollout(
        _Sim(terminal_after_advance=True),
        Decision(kind="INPUT", request=_request()),
        _cfg(),
        lambda state, team: pytest.fail("terminal rollout evaluated ValueFn"),
        cutoff_observer=lambda state, team, value: seen.append(state),
    )
    assert reward == 1.0
    assert seen == []


def test_absent_cutoff_observer_preserves_deterministic_result() -> None:
    def run() -> float:
        return _rollout(
            _Sim(),
            Decision(kind="INPUT", request=_request()),
            _cfg(),
            lambda state, team: -0.4,
        )

    assert run() == run() == pytest.approx(0.3)


def _telemetry_module() -> Any:
    try:
        return importlib.import_module("automata.evaluation.cutoff_telemetry")
    except ModuleNotFoundError:
        pytest.fail("missing public automata.evaluation.cutoff_telemetry module")


def test_recorder_writes_compact_flushed_jsonl_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    telemetry = _telemetry_module()
    path = tmp_path / "cutoffs.jsonl"
    state = SimpleNamespace(round=9)
    features = [float(i) for i in range(6)]

    monkeypatch.setattr(telemetry, "feature_vector", lambda actual, team: features)

    class Comparator:
        def __call__(self, actual: Any, team: TeamColor) -> float:
            assert actual is state
            return -0.5

    monkeypatch.setattr(telemetry, "HeuristicValue", Comparator)
    recorder = telemetry.CutoffTelemetryRecorder(
        path, case_metadata={"case_id": "tiny-7", "seed": 7}
    )
    recorder(state, TeamColor.BLUE, 0.25)

    # A just-recorded row is readable without closing the recorder.
    row = json.loads(path.read_text().splitlines()[0])
    assert row == {
        "schema_version": telemetry.SCHEMA_VERSION,
        "case_metadata": {"case_id": "tiny-7", "seed": 7},
        "team": "BLUE",
        "round": 9,
        "feature_names": list(telemetry.FEATURE_NAMES),
        "features": features,
        "active_value": 0.25,
        "heuristic_value": -0.5,
        "difference": 0.75,
    }
    assert not {"state", "board", "teams", "execution_stack"} & row.keys()


def _artifact() -> dict[str, Any]:
    count = len(_telemetry_module().FEATURE_NAMES)
    return {
        "model_version": "logistic-v1",
        "schema_version": 1,
        "red_roster": ["red"],
        "blue_roster": ["blue"],
        "feature_names": list(_telemetry_module().FEATURE_NAMES),
        "feature_means": [0.0] * count,
        "feature_scales": [1.0] * count,
        "coefficients": [0.0] * count,
        "intercept": 0.0,
    }


def _analysis_rows() -> list[dict[str, Any]]:
    telemetry = _telemetry_module()
    first_features = [0.0, 2.0, 4.0, -4.0]
    active_values = [-1.0, -0.5, 0.5, 1.0]
    heuristic_values = [1.0, 0.5, -0.5, -1.0]
    rows = []
    for index, (first, active, heuristic) in enumerate(
        zip(first_features, active_values, heuristic_values, strict=True)
    ):
        features = [0.0] * len(telemetry.FEATURE_NAMES)
        features[0] = first
        rows.append(
            {
                "schema_version": telemetry.SCHEMA_VERSION,
                "case_metadata": {"case_id": "synthetic"},
                "team": "RED",
                "round": index + 1,
                "feature_names": list(telemetry.FEATURE_NAMES),
                "features": features,
                "active_value": active,
                "heuristic_value": heuristic,
                "difference": active - heuristic,
            }
        )
    return rows


def test_analyze_reports_hand_calculable_comparison_and_ood_metrics(tmp_path: Path) -> None:
    telemetry = _telemetry_module()
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(_artifact()))

    summary = telemetry.analyze(iter(_analysis_rows()), model_path)

    assert summary["row_count"] == 4
    assert summary["pearson_correlation"] == pytest.approx(-1.0)
    assert summary["mean_abs_difference"] == pytest.approx(1.5)
    assert summary["sign_disagreement_rate"] == pytest.approx(1.0)
    assert summary["active_saturation_rate"] == pytest.approx(0.5)
    assert summary["heuristic_saturation_rate"] == pytest.approx(0.5)
    assert summary["any_feature_ood_rate"] == pytest.approx(0.5)
    first = summary["per_feature"][telemetry.FEATURE_NAMES[0]]
    assert first == pytest.approx({"mean_abs_z": 2.5, "max_abs_z": 4.0, "p95_abs_z": 4.0})
    for name in telemetry.FEATURE_NAMES[1:]:
        assert summary["per_feature"][name] == {
            "mean_abs_z": 0.0,
            "max_abs_z": 0.0,
            "p95_abs_z": 0.0,
        }


def test_analyze_defines_zero_variance_correlation() -> None:
    rows = _analysis_rows()[:2]
    for row in rows:
        row["active_value"] = 0.25
        row["difference"] = 0.25 - row["heuristic_value"]

    try:
        summary = _telemetry_module().analyze(rows, _artifact())
    except ValueError as exc:
        assert "variance" in str(exc).lower()
    else:
        assert summary["pearson_correlation"] == 0.0


def _bad_row(case: str) -> list[dict[str, Any]]:
    rows = _analysis_rows()
    if case == "schema":
        rows[0]["schema_version"] = 999
    elif case == "feature_names":
        rows[0]["feature_names"] = list(reversed(rows[0]["feature_names"]))
    elif case == "feature_count":
        rows[0]["features"] = rows[0]["features"][:-1]
    elif case == "nonfinite_feature":
        rows[0]["features"][0] = float("inf")
    elif case == "nonfinite_value":
        rows[0]["active_value"] = float("nan")
    elif case == "nonfinite_heuristic":
        rows[0]["heuristic_value"] = float("inf")
    elif case == "nonfinite_difference":
        rows[0]["difference"] = float("nan")
    return rows


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "feature_names",
        "feature_count",
        "nonfinite_feature",
        "nonfinite_value",
        "nonfinite_heuristic",
        "nonfinite_difference",
    ],
)
def test_analyze_rejects_malformed_telemetry_rows(case: str) -> None:
    with pytest.raises(ValueError):
        _telemetry_module().analyze(_bad_row(case), _artifact())


def test_analyze_rejects_empty_rows() -> None:
    with pytest.raises(ValueError):
        _telemetry_module().analyze([], _artifact())


@pytest.mark.parametrize("mutation", ["missing_schema", "nonpositive_scale"])
def test_analyze_rejects_artifacts_incompatible_with_learned_value(mutation: str) -> None:
    artifact = _artifact()
    if mutation == "missing_schema":
        artifact.pop("model_version")
    else:
        artifact["feature_scales"][0] = 0.0

    with pytest.raises(ValueError):
        _telemetry_module().analyze(_analysis_rows(), artifact)


def test_analyzer_cli_reads_jsonl_and_prints_sorted_json(tmp_path: Path, capsys: Any) -> None:
    telemetry = _telemetry_module()
    input_path = tmp_path / "cutoffs.jsonl"
    input_path.write_text("".join(json.dumps(row) + "\n" for row in _analysis_rows()))
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(_artifact()))

    assert telemetry.main(["--input", str(input_path), "--model", str(model_path)]) == 0
    output = capsys.readouterr().out.strip()
    decoded = json.loads(output)
    assert decoded["row_count"] == 4
    assert list(decoded) == sorted(decoded)
