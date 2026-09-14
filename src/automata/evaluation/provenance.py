"""Immutable source identity used by generated Learned evidence."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

GATE_FAILURE_EXIT_STATUS = 2


def source_identity(*, exclude_paths: tuple[Path, ...] = ()) -> tuple[str, str]:
    """Return HEAD and a stable digest of tracked/untracked working changes."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    command = ["git", "status", "--porcelain=v1", "--untracked-files=all"]
    status = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    excluded = {str(path.resolve()) for path in exclude_paths}
    retained = []
    for line in status.splitlines():
        path = line[3:].split(" -> ")[-1]
        resolved = str(Path(path).resolve())
        if not any(resolved == item or resolved.startswith(f"{item}/") for item in excluded):
            retained.append(line)
    dirty_hash = hashlib.sha256("\n".join(retained).encode()).hexdigest()
    return revision, dirty_hash


__all__ = ["GATE_FAILURE_EXIT_STATUS", "source_identity"]
