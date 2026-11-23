"""Shared detection data types and interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str


class Detector(Protocol):
    """Interface implemented by video frame detectors."""

    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def model_source(self) -> str: ...

    @property
    def model_license(self) -> str: ...

    @property
    def pretrained_on(self) -> str: ...

    @property
    def class_names(self) -> dict[int, str]: ...

    @property
    def checkpoint_path(self) -> Path | None: ...

    @property
    def configuration(self) -> dict[str, Any]: ...

    def detect(self, frame: np.ndarray) -> list[Detection]: ...
