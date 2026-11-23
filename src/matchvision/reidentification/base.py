"""Shared types for persistent identity association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from matchvision.detection.base import Detection
from matchvision.tracking.base import TrackedObject


@dataclass(frozen=True, slots=True)
class IdentifiedObject:
    """A tracker observation assigned a video-local persistent identity."""

    detection: Detection
    track_id: int
    persistent_id: int
    identity_state: str
    association_confidence: float | None
    history: tuple[tuple[int, int], ...]


class IdentityAssigner(Protocol):
    """Interface for mapping temporal tracks to persistent video identities."""

    @property
    def configuration(self) -> dict[str, Any]: ...

    def update(
        self,
        tracks: list[TrackedObject],
        frame: np.ndarray,
        frame_index: int,
    ) -> list[IdentifiedObject]: ...
