"""Reader for MOTChallenge and SoccerNet tracking annotations."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True, slots=True)
class MOTAnnotation:
    """One MOT-format box; frame numbers are one-based as in the source file."""

    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float


def read_mot_annotations(source: str | Path | TextIO) -> list[MOTAnnotation]:
    """Parse 9- or 10-column MOTChallenge/SoccerNet CSV annotations.

    Coordinates are top-left ``x, y, width, height``. The final three MOT
    fields are accepted but not used.
    """
    close_after_read = not hasattr(source, "read")
    stream: TextIO
    if close_after_read:
        stream = Path(source).open("r", encoding="utf-8", newline="")
    else:
        stream = source  # type: ignore[assignment]

    annotations: list[MOTAnnotation] = []
    try:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) not in (9, 10):
                raise ValueError(
                    f"Line {line_number}: expected 9 or 10 columns, got {len(row)}"
                )
            try:
                frame = int(row[0])
                track_id = int(row[1])
                x, y, width, height, confidence = map(float, row[2:7])
                unused_fields = [float(value) for value in row[7:]]
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number}: invalid numeric value"
                ) from error

            numeric_values = [x, y, width, height, confidence, *unused_fields]
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError(f"Line {line_number}: values must be finite")
            if frame < 1:
                raise ValueError(f"Line {line_number}: frame must be one-based")
            if track_id < 0:
                raise ValueError(f"Line {line_number}: track ID cannot be negative")
            if width <= 0 or height <= 0:
                raise ValueError(f"Line {line_number}: box dimensions must be positive")

            annotations.append(
                MOTAnnotation(
                    frame=frame,
                    track_id=track_id,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
                )
            )
    finally:
        if close_after_read:
            stream.close()

    return annotations
