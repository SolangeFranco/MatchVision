"""Shared tracking types and interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from matchvision.detection.base import Detection


@dataclass(frozen=True, slots=True)
class TrackedObject:
    """A detection associated with a tracker identity."""

    detection: Detection
    track_id: int


class MultiObjectTracker(Protocol):
    """Frame-oriented tracking adapter interface."""

    @property
    def tracker_name(self) -> str: ...

    @property
    def tracker_version(self) -> str: ...

    @property
    def configuration(self) -> dict[str, Any]: ...

    def update(
        self, detections: list[Detection], frame_index: int
    ) -> list[TrackedObject]: ...
