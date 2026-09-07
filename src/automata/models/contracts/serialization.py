"""Canonical serialization for versioned learned-model contracts."""

from __future__ import annotations

import json
from typing import Any, TypeVar, get_args

from pydantic import BaseModel

ContractT = TypeVar("ContractT", bound=BaseModel)


def canonical_json_bytes(value: BaseModel) -> bytes:
    """Serialize a contract to finite, stable UTF-8 JSON."""
    return json.dumps(
        value.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def from_canonical_json(model: type[ContractT], payload: bytes | str) -> ContractT:
    """Decode a top-level contract, rejecting versions before nested parsing."""
    try:
        raw: Any = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid canonical JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("top-level contract must be a JSON object")
    version = raw.get("schema_version")
    supported_versions = set(get_args(model.model_fields["schema_version"].annotation))
    if version not in supported_versions:
        raise ValueError(f"unsupported top-level schema version: {version!r}")
    return model.model_validate(raw)


__all__ = ["canonical_json_bytes", "from_canonical_json"]
