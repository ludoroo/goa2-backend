from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from automata.training.io import atomic_write_bytes, canonical_json_bytes, content_digest


class _Label(StrEnum):
    VALUE = "value"


class _Payload(BaseModel):
    label: _Label
    count: int


@dataclass(frozen=True)
class _DataclassPayload:
    label: _Label
    count: int


def test_canonical_json_is_consistent_across_supported_model_types() -> None:
    expected = b'{"count":2,"label":"value"}'

    assert canonical_json_bytes({"label": "value", "count": 2}) == expected
    assert canonical_json_bytes(_Payload(label=_Label.VALUE, count=2)) == expected
    assert canonical_json_bytes(_DataclassPayload(label=_Label.VALUE, count=2)) == expected
    assert content_digest(_Payload(label=_Label.VALUE, count=2)) == content_digest(
        _DataclassPayload(label=_Label.VALUE, count=2)
    )


def test_atomic_write_fsyncs_file_and_parent_directory(tmp_path: Path, monkeypatch) -> None:
    fsynced: list[int] = []
    import automata.training.io as training_io

    real_fsync = training_io.os.fsync

    def recording_fsync(descriptor: int) -> None:
        fsynced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(training_io.os, "fsync", recording_fsync)
    destination = tmp_path / "nested" / "evidence.json"

    atomic_write_bytes(destination, b"complete")

    assert destination.read_bytes() == b"complete"
    assert len(fsynced) == 2
