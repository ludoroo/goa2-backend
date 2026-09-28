from pathlib import Path


def test_pr2_runtime_core_has_no_back_edge_to_offline_harness() -> None:
    paths = (
        *Path("src/automata/nn").rglob("*.py"),
        *Path("src/automata/observation").rglob("*.py"),
        *Path("src/automata/search").rglob("*.py"),
        *Path("src/automata/runtime").rglob("*.py"),
    )
    forbidden = ("automata.training", "automata.evaluation", "automata.harness")

    offenders = []
    for path in paths:
        text = path.read_text()
        if any(module in text for module in forbidden):
            offenders.append(str(path))

    assert offenders == []
