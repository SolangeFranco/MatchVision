"""Sequence manifest loading and split-leakage validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class ManifestError(ValueError):
    """Raised when a sequence manifest is invalid or leaks a game across splits."""


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    game_id: str
    sequence_id: str
    split: str
    video_path: Path
    annotation_path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    video_sha256: str | None = None
    annotation_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SequenceManifest:
    path: Path
    dataset: str
    release: str
    sequences: tuple[SequenceRecord, ...]
    sha256: str

    def select(self, split: str, limit: int | None = None) -> tuple[SequenceRecord, ...]:
        """Return a stable game/sequence-sorted subset from exactly one split."""
        selected = sorted(
            (sequence for sequence in self.sequences if sequence.split == split),
            key=lambda sequence: (sequence.game_id, sequence.sequence_id),
        )
        if not selected:
            raise ManifestError(f"Manifest contains no sequences for split {split!r}")
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be a positive integer")
            selected = selected[:limit]
        return tuple(selected)


def _validate_cross_split_integrity(sequences: list[dict[str, Any]]) -> None:
    game_splits: dict[str, str] = {}
    sequence_ids: set[str] = set()
    for sequence in sequences:
        game_id = sequence["game_id"]
        split = sequence["split"]
        prior_split = game_splits.setdefault(game_id, split)
        if prior_split != split:
            raise ManifestError(
                f"Game {game_id!r} appears in both {prior_split!r} and {split!r}"
            )
        sequence_id = sequence["sequence_id"]
        if sequence_id in sequence_ids:
            raise ManifestError(f"Duplicate sequence_id {sequence_id!r}")
        sequence_ids.add(sequence_id)


def _resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _verify_hash(path: Path, expected: str | None, field: str) -> None:
    if expected is None:
        return
    if not path.is_file():
        raise ManifestError(f"Cannot verify {field}; file does not exist: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest.lower() != expected.lower():
        raise ManifestError(f"SHA-256 mismatch for {field}: {path}")


def load_sequence_manifest(
    path: str | Path,
    *,
    require_hashes: bool = False,
    verify_files: bool = False,
) -> SequenceManifest:
    """Load a JSON manifest, validate schema/split isolation, and optionally files."""
    manifest_path = Path(path).resolve()
    schema_path = files("matchvision").joinpath(
        "schemas", "sequence-manifest.schema.json"
    )
    try:
        raw_bytes = manifest_path.read_bytes()
        document = json.loads(raw_bytes)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"Could not load manifest or schema: {error}") from error

    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
            f"{error.message}"
            for error in errors
        )
        raise ManifestError(f"Invalid sequence manifest: {details}")

    _validate_cross_split_integrity(document["sequences"])
    records: list[SequenceRecord] = []
    for entry in document["sequences"]:
        video_path = _resolve_path(manifest_path, entry["video_path"])
        annotation_path = _resolve_path(manifest_path, entry["annotation_path"])
        video_hash = entry.get("video_sha256")
        annotation_hash = entry.get("annotation_sha256")
        if require_hashes and (video_hash is None or annotation_hash is None):
            raise ManifestError(
                f"Sequence {entry['sequence_id']!r} must include video_sha256 "
                "and annotation_sha256"
            )
        if verify_files:
            for field, file_path, digest in (
                ("video_path", video_path, video_hash),
                ("annotation_path", annotation_path, annotation_hash),
            ):
                if not file_path.is_file():
                    raise ManifestError(f"{field} does not exist: {file_path}")
                _verify_hash(file_path, digest, field)

        records.append(
            SequenceRecord(
                game_id=entry["game_id"],
                sequence_id=entry["sequence_id"],
                split=entry["split"],
                video_path=video_path,
                annotation_path=annotation_path,
                frame_count=entry["frame_count"],
                fps=float(entry["fps"]),
                width=entry["width"],
                height=entry["height"],
                video_sha256=video_hash,
                annotation_sha256=annotation_hash,
            )
        )

    return SequenceManifest(
        path=manifest_path,
        dataset=document["dataset"],
        release=document["release"],
        sequences=tuple(records),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
