"""Immutable source identity used by generated Learned evidence."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from collections.abc import Iterable
from pathlib import Path

GATE_FAILURE_EXIT_STATUS = 2

_IDENTITY_FORMAT = b"automata-source-identity-v2"


def repository_root() -> Path:
    """Return the canonical root of the Git repository containing the process cwd."""
    raw = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True
    ).stdout.rstrip(b"\n")
    if not raw:
        raise RuntimeError("git returned an empty repository root")
    return Path(os.fsdecode(raw)).resolve(strict=True)


def _git(repo: Path, *args: bytes) -> bytes:
    command = [b"git", b"-C", os.fsencode(repo), *args]
    return subprocess.run(command, check=True, capture_output=True).stdout


def _relative_exclusions(repo: Path, paths: Iterable[Path]) -> tuple[bytes, ...]:
    exclusions: list[bytes] = []
    cwd = Path.cwd()
    for supplied in paths:
        candidate = supplied if supplied.is_absolute() else cwd / supplied
        try:
            resolved = candidate.resolve(strict=False)
            relative = resolved.relative_to(repo)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"excluded path is outside the repository: {supplied}") from exc
        if not relative.parts:
            raise ValueError("cannot exclude the repository root from source identity")
        exclusions.append(os.fsencode(relative.as_posix()))
    return tuple(sorted(set(exclusions)))


def _is_excluded(path: bytes, exclusions: tuple[bytes, ...]) -> bool:
    return any(path == excluded or path.startswith(excluded + b"/") for excluded in exclusions)


def _validate_git_path(path: bytes) -> None:
    if (
        not path
        or path.startswith(b"/")
        or b"\0" in path
        or any(component in {b"", b".", b".."} for component in path.split(b"/"))
    ):
        raise ValueError(f"git returned an unsafe repository path: {path!r}")


def _tracked_diff(repo: Path, exclusions: tuple[bytes, ...]) -> bytes:
    changed = _git(
        repo,
        b"diff",
        b"--name-only",
        b"-z",
        b"--no-renames",
        b"HEAD",
        b"--",
    ).split(b"\0")
    retained: list[bytes] = []
    for path in changed:
        if not path:
            continue
        _validate_git_path(path)
        if not _is_excluded(path, exclusions):
            retained.append(path)
    if not retained:
        return b""
    pathspecs = [b":(top,literal)" + path for path in sorted(set(retained))]
    return _git(
        repo,
        b"-c",
        b"core.quotePath=true",
        b"-c",
        b"diff.algorithm=myers",
        b"diff",
        b"--binary",
        b"--full-index",
        b"--no-color",
        b"--no-ext-diff",
        b"--no-textconv",
        b"--find-renames=50%",
        b"HEAD",
        b"--",
        *pathspecs,
    )


def _frame(digest: hashlib._Hash, label: bytes, payload: bytes) -> None:
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _start_frame(digest: hashlib._Hash, label: bytes, payload_length: int) -> None:
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(payload_length.to_bytes(8, "big"))


def _safe_untracked_path(repo: Path, relative: bytes) -> tuple[bytes, os.stat_result]:
    _validate_git_path(relative)
    absolute = os.path.join(os.fsencode(repo), *relative.split(b"/"))
    parent = Path(os.fsdecode(os.path.dirname(absolute))).resolve(strict=True)
    try:
        parent.relative_to(repo)
    except ValueError as exc:
        raise ValueError(
            f"untracked path escapes repository through a symlink: {relative!r}"
        ) from exc
    return absolute, os.lstat(absolute)


def _hash_untracked_file(
    digest: hashlib._Hash, repo: Path, relative: bytes, metadata: os.stat_result
) -> None:
    absolute, current = _safe_untracked_path(repo, relative)
    if (current.st_dev, current.st_ino, current.st_mode) != (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
    ):
        raise RuntimeError(f"untracked path changed while hashing: {relative!r}")

    _frame(digest, b"untracked-path", relative)
    if stat.S_ISLNK(current.st_mode):
        _frame(digest, b"untracked-kind", b"symlink")
        _frame(digest, b"untracked-content", os.readlink(absolute))
        return
    if not stat.S_ISREG(current.st_mode):
        raise ValueError(f"unsupported untracked file type: {relative!r}")

    _frame(digest, b"untracked-kind", b"regular")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino) or not stat.S_ISREG(
            opened.st_mode
        ):
            raise RuntimeError(f"untracked path changed while hashing: {relative!r}")
        _start_frame(digest, b"untracked-content", opened.st_size)
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"untracked file changed while hashing: {relative!r}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"untracked file changed while hashing: {relative!r}")
    finally:
        os.close(descriptor)


def source_identity(*, exclude_paths: tuple[Path, ...] = ()) -> tuple[str, str]:
    """Return HEAD and a content digest of non-excluded changes relative to it.

    Tracked changes are represented by a canonical binary Git diff. Untracked,
    non-ignored regular files contribute their repository-relative path and exact
    bytes; symlinks contribute their path and link target without being followed.
    Exclusions must resolve inside the repository and apply recursively.
    """
    repo = repository_root()
    revision_bytes = _git(repo, b"rev-parse", b"--verify", b"HEAD").strip()
    try:
        revision = revision_bytes.decode("ascii")
    except UnicodeDecodeError as exc:  # pragma: no cover - defensive Git contract check
        raise RuntimeError("git returned a non-ASCII HEAD revision") from exc
    exclusions = _relative_exclusions(repo, exclude_paths)

    digest = hashlib.sha256()
    _frame(digest, b"format", _IDENTITY_FORMAT)
    _frame(digest, b"head", revision_bytes)
    _frame(digest, b"tracked-diff", _tracked_diff(repo, exclusions))

    untracked = _git(
        repo,
        b"ls-files",
        b"--others",
        b"--exclude-standard",
        b"-z",
        b"--",
    ).split(b"\0")
    retained: list[tuple[bytes, os.stat_result]] = []
    for relative in untracked:
        if not relative:
            continue
        _validate_git_path(relative)
        if _is_excluded(relative, exclusions):
            continue
        _, metadata = _safe_untracked_path(repo, relative)
        retained.append((relative, metadata))
    for relative, metadata in sorted(retained, key=lambda item: item[0]):
        _hash_untracked_file(digest, repo, relative, metadata)
    return revision, digest.hexdigest()


__all__ = ["GATE_FAILURE_EXIT_STATUS", "repository_root", "source_identity"]
