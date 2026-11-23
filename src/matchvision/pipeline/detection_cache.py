"""Atomic, versioned storage for bounded video-detection segments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from matchvision.detection.base import Detection


class DetectionSegmentCache:
    """Cache detector outputs without serializing mutable tracker state."""

    def __init__(
        self,
        directory: str | Path,
        *,
        cache_key: dict[str, Any],
        segment_frames: int,
    ) -> None:
        if segment_frames < 1:
            raise ValueError("segment_frames must be positive")
        self.directory = Path(directory).expanduser().resolve()
        self.segment_frames = segment_frames
        self.directory.mkdir(parents=True, exist_ok=True)
        serialized_key = json.dumps(
            cache_key,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
        normalized_cache_key = json.loads(serialized_key)
        key_bytes = serialized_key.encode("utf-8")
        self.cache_key_sha256 = hashlib.sha256(key_bytes).hexdigest()
        manifest_path = self.directory / "manifest.json"
        manifest = {
            "schema_version": 1,
            "cache_key_sha256": self.cache_key_sha256,
            "segment_frames": segment_frames,
            "cache_key": normalized_cache_key,
        }
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"No se puede leer el manifiesto de checkpoints: {manifest_path}"
                ) from error
            if existing != manifest:
                raise ValueError(
                    "Los checkpoints no coinciden con el video, el detector, "
                    "los pesos o el tamaño de segmento actuales."
                )
        else:
            entries = list(self.directory.iterdir())
            if entries:
                raise ValueError(
                    "La carpeta de checkpoints no está vacía y no contiene "
                    "un manifiesto compatible."
                )
            self._write_atomic(manifest_path, manifest)

    def load(self, frame_start: int) -> list[list[Detection]] | None:
        path = self._segment_path(frame_start)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"El segmento de detecciones no es JSON válido: {path}") from error
        detections = payload.get("detections") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("frame_start") != frame_start
            or not isinstance(detections, list)
            or not 1 <= len(detections) <= self.segment_frames
        ):
            raise ValueError(f"El checkpoint de detecciones no es compatible: {path}")
        return [
            [self._decode_detection(item, path) for item in frame]
            if isinstance(frame, list)
            else self._invalid_segment(path)
            for frame in detections
        ]

    def save(
        self, frame_start: int, detections: list[list[Detection]]
    ) -> None:
        if not 1 <= len(detections) <= self.segment_frames:
            raise ValueError("El segmento guardado no tiene un tamaño válido.")
        path = self._segment_path(frame_start)
        if path.exists():
            raise FileExistsError(f"No se sobrescribió el checkpoint existente: {path}")
        payload = {
            "schema_version": 1,
            "frame_start": frame_start,
            "detections": [
                [
                    {
                        "x1": item.x1,
                        "y1": item.y1,
                        "x2": item.x2,
                        "y2": item.y2,
                        "confidence": item.confidence,
                        "class_id": item.class_id,
                        "label": item.label,
                    }
                    for item in frame
                ]
                for frame in detections
            ],
        }
        self._write_atomic(path, payload)

    def _segment_path(self, frame_start: int) -> Path:
        if frame_start < 0 or frame_start % self.segment_frames:
            raise ValueError("frame_start debe coincidir con un límite de segmento.")
        return self.directory / f"segment-{frame_start:012d}.json"

    @classmethod
    def _decode_detection(cls, item: Any, path: Path) -> Detection:
        if not isinstance(item, dict):
            return cls._invalid_segment(path)
        coordinates = (
            item.get("x1"), item.get("y1"), item.get("x2"), item.get("y2"),
            item.get("confidence"),
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in coordinates
            )
            or isinstance(item.get("class_id"), bool)
            or not isinstance(item.get("class_id"), int)
            or not isinstance(item.get("label"), str)
        ):
            return cls._invalid_segment(path)
        return Detection(
            x1=float(item["x1"]),
            y1=float(item["y1"]),
            x2=float(item["x2"]),
            y2=float(item["y2"]),
            confidence=float(item["confidence"]),
            class_id=item["class_id"],
            label=item["label"],
        )

    @staticmethod
    def _invalid_segment(path: Path) -> Any:
        raise ValueError(f"El formato del checkpoint de detecciones no es válido: {path}")

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    payload, stream, indent=2, sort_keys=True, allow_nan=False,
                    default=str,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
