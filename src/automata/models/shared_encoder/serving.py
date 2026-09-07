"""Production loading boundary for shared-encoder learned-model artifacts.

PyTorch direct inference is the first serving runtime because it executes the
same exported artifact used by training and the existing runtime. ONNX is
deliberately deferred until a measured parity and performance comparison is
available; no latency advantage is assumed here.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from ..contracts import ArtifactError, RuntimeRequirements
from .runtime import SharedEncoderRuntime


def _requirements_identity(requirements: RuntimeRequirements) -> tuple[object, ...]:
    return (
        requirements.runtime_compatibility_version,
        requirements.observation_schema_version,
        requirements.map_schema_version,
        tuple(sorted(requirements.heroes)),
        requirements.map_id,
        requirements.game_type,
        tuple(sorted(requirements.hero_adapter_versions.items())),
    )


def _artifact_identity(path: Path) -> tuple[str, str]:
    try:
        manifest = json.loads((path / "manifest.json").read_bytes())
        model_digest = manifest["model_digest"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactError("invalid artifact manifest") from exc
    if not isinstance(model_digest, str) or len(model_digest) != 64:
        raise ArtifactError("invalid artifact model digest")

    digest = hashlib.sha256()
    try:
        files = sorted(item for item in path.iterdir() if item.is_file())
        if not files or any(not item.is_file() for item in path.iterdir()):
            raise ArtifactError("artifact must contain files only")
        for item in files:
            digest.update(item.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(item.read_bytes())
            digest.update(b"\0")
    except OSError as exc:
        raise ArtifactError("artifact could not be inspected") from exc
    return model_digest, digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    fingerprint: str
    runtime: SharedEncoderRuntime


class RuntimeCache:
    """Thread-safe load-once cache that rejects mutation of a cached reference.

    Failed loads are not retained, so a repaired artifact may be retried.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[object, ...], _CacheEntry] = {}
        self._references: dict[str, str] = {}

    def get(
        self,
        artifact: str | Path,
        *,
        requirements: RuntimeRequirements,
        expected_digest: str | None = None,
    ) -> SharedEncoderRuntime:
        path = Path(artifact).resolve(strict=True)
        with self._lock:
            model_digest, fingerprint = _artifact_identity(path)
            if expected_digest is not None and model_digest != expected_digest:
                raise ArtifactError("artifact model digest does not match pinned digest")
            previous_fingerprint = self._references.get(str(path))
            if previous_fingerprint is not None and previous_fingerprint != fingerprint:
                raise ArtifactError("cached artifact reference was mutated")
            key = (str(path), model_digest, *_requirements_identity(requirements))
            cached = self._entries.get(key)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    raise ArtifactError("cached artifact reference was mutated")
                return cached.runtime

            runtime = SharedEncoderRuntime.from_artifact(
                path, requirements=requirements, device="cpu"
            )
            _, loaded_fingerprint = _artifact_identity(path)
            if loaded_fingerprint != fingerprint:
                raise ArtifactError("artifact mutated while it was being loaded")
            self._entries[key] = _CacheEntry(fingerprint=fingerprint, runtime=runtime)
            self._references[str(path)] = fingerprint
            return runtime


_PROCESS_RUNTIME_CACHE = RuntimeCache()


def load_runtime(
    artifact: str | Path,
    *,
    requirements: RuntimeRequirements,
    expected_digest: str | None = None,
) -> SharedEncoderRuntime:
    """Return the process-shared runtime for one exact artifact requirement set."""

    return _PROCESS_RUNTIME_CACHE.get(
        artifact,
        requirements=requirements,
        expected_digest=expected_digest,
    )


__all__ = ["RuntimeCache", "load_runtime"]
