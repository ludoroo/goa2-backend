from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from automata.evaluation.provenance import source_identity


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "provenance@example.invalid")
    _git(tmp_path, "config", "user.name", "Provenance Test")
    (tmp_path / "source.py").write_bytes(b"original\n")
    (tmp_path / "rename-me.txt").write_bytes(b"rename contents\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "fixture")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_source_identity_is_repeatable_and_tracks_content_not_only_paths(repository: Path) -> None:
    clean = source_identity()
    assert source_identity() == clean

    source = repository / "source.py"
    source.write_bytes(b"first replacement\n")
    first = source_identity()
    assert source_identity() == first

    source.write_bytes(b"second replacement\n")
    second = source_identity()

    assert first[0] == second[0] == clean[0]
    assert clean[1] != first[1] != second[1]


def test_source_identity_hashes_untracked_path_bytes_and_symlink_target(repository: Path) -> None:
    untracked = repository / "new-source.bin"
    untracked.write_bytes(b"\x00first\xff")
    first = source_identity()
    untracked.write_bytes(b"\x00second\xff")
    second = source_identity()
    assert first[1] != second[1]

    external = repository.parent / "external-source"
    external.write_bytes(b"outside one")
    link = repository / "source-link"
    link.symlink_to(external)
    linked = source_identity()
    external.write_bytes(b"outside two")
    assert source_identity() == linked

    link.unlink()
    link.symlink_to(repository.parent / "different-target")
    assert source_identity()[1] != linked[1]


def test_source_identity_captures_tracked_delete_rename_and_mode(repository: Path) -> None:
    baseline = source_identity()[1]
    (repository / "source.py").unlink()
    deleted = source_identity()[1]
    assert deleted != baseline

    _git(repository, "checkout", "--", "source.py")
    (repository / "rename-me.txt").rename(repository / "renamed.txt")
    renamed = source_identity()[1]
    assert renamed not in {baseline, deleted}

    (repository / "renamed.txt").rename(repository / "rename-me.txt")
    (repository / "source.py").chmod(0o755)
    mode_changed = source_identity()[1]
    assert mode_changed not in {baseline, deleted, renamed}


def test_source_identity_excludes_tracked_and_untracked_output_tree(repository: Path) -> None:
    output = repository / "runs"
    output.mkdir()
    (output / "old-run.jsonl").write_bytes(b"old output")
    baseline = source_identity(exclude_paths=(output,))

    (output / "old-run.jsonl").write_bytes(b"changed old output")
    (output / "current-run.jsonl").write_bytes(b"current output")
    assert source_identity(exclude_paths=(output,)) == baseline

    source = repository / "source.py"
    source.write_bytes(b"real source change")
    changed = source_identity(exclude_paths=(output,))
    assert changed[1] != baseline[1]

    _git(repository, "add", "runs/old-run.jsonl")
    (output / "old-run.jsonl").write_bytes(b"tracked output changed again")
    assert source_identity(exclude_paths=(output,)) == changed


def test_source_identity_rejects_unsafe_exclusions(repository: Path) -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        source_identity(exclude_paths=(repository.parent / "other",))
    with pytest.raises(ValueError, match="repository root"):
        source_identity(exclude_paths=(repository,))

    if os.name != "nt":
        outside = repository.parent / "outside"
        outside.mkdir()
        link = repository / "outside-link"
        link.symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="outside the repository"):
            source_identity(exclude_paths=(link,))
