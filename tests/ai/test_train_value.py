"""RED behavioral tests for ``automata.evaluation.train_value.main(argv)``.

The CLI validates ValueExample JSONL, splits by game, equalizes each game's
training weight, selects logistic-regression C, and emits a deterministic
LearnedValue-compatible artifact with provenance and held-out metrics.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from automata.evaluation.features import FEATURE_NAMES, RICH_FEATURE_NAMES
from automata.evaluation.value_dataset import SCHEMA_VERSION

RED = ["Wasp", "Xargatha"]
BLUE = ["Arien", "Brogan"]
LIFE_IDX = FEATURE_NAMES.index("life_diff")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _row(*, game_id: str, seed: int, team: str, winner: str, life: float) -> dict[str, Any]:
    features = [0.0] * len(FEATURE_NAMES)
    features[LIFE_IDX] = life
    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": game_id,
        "world_seed": seed,
        "team": team,
        "features": features,
        "feature_names": list(FEATURE_NAMES),
        "winner": winner,
        "red_heroes": list(RED),
        "blue_heroes": list(BLUE),
        "source_revision": "rev-testfake",
        "dirty_tree_hash": "clean",
    }


def _game(game_id: str, seed: int, winner: str, rows: int = 4) -> list[dict[str, Any]]:
    # Features are from the deciding team's perspective. Winner decisions see
    # +3 life_diff and loser decisions -3, independent of fixed board side.
    red_signal = 3.0 if winner == "RED" else -3.0
    return [
        _row(
            game_id=game_id,
            seed=seed,
            team=("RED" if i % 2 == 0 else "BLUE"),
            winner=winner,
            life=(red_signal if i % 2 == 0 else -red_signal),
        )
        for i in range(rows)
    ]


def _dataset(games_per_winner: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(games_per_winner):
        rows.extend(_game(f"red-{i}", 2 * i, "RED"))
        rows.extend(_game(f"blue-{i}", 2 * i + 1, "BLUE"))
    return rows


def _sampled_dataset(games_per_winner: int = 8) -> list[dict[str, Any]]:
    rows = _dataset(games_per_winner)
    for index, row in enumerate(rows):
        row["sample_id"] = f"cutoff-sample-{index}"
        # Independent continuations from one source game may disagree.
        row["winner"] = "RED" if index % 4 < 2 else "BLUE"
    return rows


def _rich(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        signal = row["features"][LIFE_IDX]
        row["feature_schema"] = "rich-v1"
        row["feature_names"] = list(RICH_FEATURE_NAMES)
        row["features"] = [signal, *([0.0] * (len(RICH_FEATURE_NAMES) - 1))]
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _args(
    source: Path,
    out: Path,
    *,
    split_seed: int = 0,
    c_grid: str = "0.1,1.0,10.0",
) -> list[str]:
    return [
        "--input",
        str(source),
        "--out",
        str(out),
        "--split-seed",
        str(split_seed),
        "--val-games",
        "4",
        "--test-games",
        "4",
        "--c-grid",
        c_grid,
    ]


def _main(argv: list[str]) -> int | None:
    from automata.evaluation import train_value

    return train_value.main(argv)


def _train(tmp_path: Path, rows: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    source = _write(tmp_path / "dataset.jsonl", rows)
    out = tmp_path / "model.json"
    assert _main(_args(source, out, **kwargs)) in (None, 0)
    return json.loads(out.read_text(encoding="utf-8"))


def _split(artifact: dict[str, Any]) -> dict[str, set[tuple[str, int]]]:
    split_games = artifact["provenance"]["split_games"]
    assert set(split_games) == {"train", "validation", "test"}
    return {
        name: {(item["game_id"], item["world_seed"]) for item in split_games[name]}
        for name in ("train", "validation", "test")
    }


def test_pyproject_declares_training_group_in_dev_only() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = config.get("dependency-groups", {})

    def has_sklearn(items: list[Any]) -> bool:
        return any(
            "scikit-learn" in str(item).lower() or "sklearn" in str(item).lower() for item in items
        )

    training = groups.get("training")
    assert isinstance(training, list), "missing dependency group: training"
    assert has_sklearn(training), "training group must contain scikit-learn"

    dev = groups.get("dev")
    assert isinstance(dev, list), "missing dependency group: dev"
    assert has_sklearn(dev) or any("training" in str(item) for item in dev)
    assert not has_sklearn(config.get("project", {}).get("dependencies", []))


def test_main_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
        _main(_args(tmp_path / "missing.jsonl", tmp_path / "out.json"))


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": SCHEMA_VERSION + 1},
        {"feature_names": [FEATURE_NAMES[1], FEATURE_NAMES[0], *FEATURE_NAMES[2:]]},
        {"features": [0.0] * (len(FEATURE_NAMES) - 1)},
        {"team": "GREEN"},
        {"winner": None},
    ],
    ids=["schema", "feature-order", "feature-count", "team", "incomplete"],
)
def test_main_rejects_malformed_or_incomplete_rows(
    tmp_path: Path, mutation: dict[str, Any]
) -> None:
    rows = _dataset(4)
    rows[0] = {**rows[0], **mutation}
    with pytest.raises((ValueError, SystemExit)):
        _main(_args(_write(tmp_path / "bad.jsonl", rows), tmp_path / "out.json"))


@pytest.mark.parametrize("problem", ["roster", "winner", "game-id", "world-seed"])
def test_main_rejects_mixed_or_ambiguous_games(tmp_path: Path, problem: str) -> None:
    rows = _dataset(4)
    if problem == "roster":
        rows[0]["red_heroes"] = ["Wasp", "Bain"]
    elif problem == "winner":
        rows[0]["winner"] = "BLUE"
    elif problem == "game-id":
        rows[0]["world_seed"] = 999  # same game_id, conflicting seed
    else:
        rows[0]["game_id"] = "different"  # same seed, conflicting game_id
    with pytest.raises((ValueError, SystemExit)):
        _main(_args(_write(tmp_path / "mixed.jsonl", rows), tmp_path / "out.json"))


def test_main_rejects_single_label_class(tmp_path: Path) -> None:
    rows = [
        _row(game_id=f"game-{i}", seed=i, team="RED", winner="RED", life=3.0) for i in range(10)
    ]
    with pytest.raises((ValueError, SystemExit)):
        _main(_args(_write(tmp_path / "one-class.jsonl", rows), tmp_path / "out.json"))


def test_main_accepts_one_winner_side_with_both_perspective_labels(tmp_path: Path) -> None:
    rows = [row for i in range(10) for row in _game(f"game-{i}", i, "RED")]
    artifact = _train(tmp_path, rows)
    assert artifact["provenance"]["game_count"] == 10


def test_artifact_is_learned_value_compatible(tmp_path: Path) -> None:
    from automata.evaluation.learned_value import LearnedValue

    artifact = _train(tmp_path, _dataset())
    assert artifact["model_version"] == "logistic-v1"
    assert artifact["schema_version"] == 1
    assert artifact["red_roster"] == RED
    assert artifact["blue_roster"] == BLUE
    assert artifact["feature_names"] == list(FEATURE_NAMES)
    for field in ("feature_means", "feature_scales", "coefficients"):
        assert len(artifact[field]) == len(FEATURE_NAMES)
    assert isinstance(artifact["intercept"], (int, float))
    LearnedValue(artifact)


def test_rich_rows_export_schema_and_mixed_schemas_are_rejected(tmp_path: Path) -> None:
    from automata.evaluation.learned_value import LearnedValue

    artifact = _train(tmp_path / "rich", _rich(_dataset()))
    assert artifact["feature_schema"] == "rich-v1"
    assert artifact["feature_names"] == list(RICH_FEATURE_NAMES)
    LearnedValue(artifact)

    mixed = _dataset(4)
    _rich(mixed[:1])
    with pytest.raises((ValueError, SystemExit)):
        _train(tmp_path / "mixed", mixed)


def test_unknown_feature_schema_is_rejected(tmp_path: Path) -> None:
    rows = _rich(_dataset(4))
    for row in rows:
        row["feature_schema"] = "unknown-v1"
    with pytest.raises((ValueError, SystemExit)):
        _train(tmp_path, rows)


def test_artifact_contains_provenance_and_metrics(tmp_path: Path) -> None:
    artifact = _train(tmp_path, _dataset())
    provenance = artifact["provenance"]
    assert len(provenance["dataset_digest"]) == 64
    assert provenance["source_identities"] == [
        {"source_revision": "rev-testfake", "dirty_tree_hash": "clean"}
    ]
    assert provenance["split_seed"] == 0
    assert provenance["game_count"] == 24
    assert provenance["row_count"] == 96
    assert provenance["selected_c"] in (0.1, 1.0, 10.0)

    metrics = artifact["metrics"]
    for field in ("test_log_loss", "brier", "accuracy"):
        assert isinstance(metrics[field], (int, float))
    assert isinstance(metrics["ece"], (int, float))
    assert metrics["test_log_loss"] >= 0.0
    assert 0.0 <= metrics["brier"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_selected_c_comes_from_user_grid(tmp_path: Path) -> None:
    artifact = _train(tmp_path, _dataset(), c_grid="0.25")
    assert artifact["provenance"]["selected_c"] == 0.25


def test_decision_team_labels_learn_separable_signal(tmp_path: Path) -> None:
    artifact = _train(tmp_path, _dataset(20))
    assert artifact["metrics"]["accuracy"] > 0.5


def test_same_dataset_and_config_produces_identical_bytes(tmp_path: Path) -> None:
    source = _write(tmp_path / "dataset.jsonl", _dataset())
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    _main(_args(source, first))
    _main(_args(source, second))
    assert first.read_bytes() == second.read_bytes()


def test_split_is_grouped_complete_disjoint_and_deterministic(tmp_path: Path) -> None:
    rows = _dataset(8)
    source = _write(tmp_path / "dataset.jsonl", rows)
    outs = [tmp_path / f"model-{i}.json" for i in range(3)]
    _main(_args(source, outs[0], split_seed=42))
    _main(_args(source, outs[1], split_seed=42))
    _main(_args(source, outs[2], split_seed=99))
    splits = [_split(json.loads(path.read_text(encoding="utf-8"))) for path in outs]

    assert splits[0] == splits[1]
    assert splits[0] != splits[2]
    train, validation, test = (
        splits[0]["train"],
        splits[0]["validation"],
        splits[0]["test"],
    )
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    expected = {(row["game_id"], row["world_seed"]) for row in rows}
    assert train | validation | test == expected
    assert len(validation) == len(test) == 4


def test_cutoff_samples_with_varied_winners_stay_in_their_source_game_split(
    tmp_path: Path,
) -> None:
    rows = _sampled_dataset()
    artifact = _train(tmp_path, rows)
    splits = _split(artifact)
    expected = {(row["game_id"], row["world_seed"]) for row in rows}

    assert artifact["provenance"]["game_count"] == len(expected)
    assert sum(map(len, splits.values())) == len(expected)
    assert set().union(*splits.values()) == expected
    for source_game in expected:
        assert sum(source_game in split for split in splits.values()) == 1


def test_cutoff_sample_weight_is_equalized_by_source_game(tmp_path: Path) -> None:
    rows = _sampled_dataset()
    base = _train(tmp_path / "base", rows)
    train_key = next(iter(_split(base)["train"]))

    inflated: list[dict[str, Any]] = []
    for row in rows:
        inflated.append(row)
        if (row["game_id"], row["world_seed"]) == train_key:
            inflated.extend(
                {**row, "sample_id": f"{row['sample_id']}-copy-{copy}"} for copy in range(3)
            )
    boosted = _train(tmp_path / "boosted", inflated)

    assert train_key in _split(boosted)["train"]
    for field in ("feature_means", "feature_scales", "coefficients"):
        assert boosted[field] == pytest.approx(base[field], abs=1e-6)
    assert boosted["intercept"] == pytest.approx(base["intercept"], abs=1e-6)


def test_natural_game_rows_still_reject_inconsistent_winner(tmp_path: Path) -> None:
    rows = _dataset(4)
    rows[0]["winner"] = "BLUE" if rows[0]["winner"] == "RED" else "RED"

    with pytest.raises((ValueError, SystemExit)):
        _main(_args(_write(tmp_path / "natural.jsonl", rows), tmp_path / "out.json"))


def test_duplicate_rows_in_one_training_game_do_not_change_model(tmp_path: Path) -> None:
    rows = _dataset()
    base = _train(tmp_path / "base", rows)
    train_key = next(iter(_split(base)["train"]))

    inflated: list[dict[str, Any]] = []
    for row in rows:
        inflated.append(row)
        if (row["game_id"], row["world_seed"]) == train_key:
            inflated.extend(dict(row) for _ in range(3))
    boosted = _train(tmp_path / "boosted", inflated)

    assert train_key in _split(boosted)["train"]
    for field in ("feature_means", "feature_scales", "coefficients"):
        assert boosted[field] == pytest.approx(base[field], abs=1e-6)
    assert boosted["intercept"] == pytest.approx(base["intercept"], abs=1e-6)
    assert boosted["provenance"]["row_count"] > base["provenance"]["row_count"]
