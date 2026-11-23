"""Asociación temporal conservadora para detecciones del balón."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from matchvision.detection.base import Detection


_BALL_LABELS = {"ball", "football", "soccer ball", "sports ball"}
BallTrackStatus = Literal["observed", "predicted", "missing"]


@dataclass(frozen=True, slots=True)
class BallTrackConfig:
    minimum_confidence: float = 0.05
    maximum_box_area_fraction: float = 0.01
    maximum_aspect_ratio: float = 8.0
    maximum_speed_pixels_per_frame: float = 180.0
    gate_margin_pixels: float = 18.0
    maximum_missed_frames: int = 8
    velocity_smoothing: float = 0.35
    velocity_damping: float = 0.92

    def __post_init__(self) -> None:
        numeric_values = (
            self.minimum_confidence,
            self.maximum_box_area_fraction,
            self.maximum_aspect_ratio,
            self.maximum_speed_pixels_per_frame,
            self.gate_margin_pixels,
            self.velocity_smoothing,
            self.velocity_damping,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("La configuración numérica debe ser finita")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence debe estar entre 0 y 1")
        if not 0 < self.maximum_box_area_fraction <= 1:
            raise ValueError("maximum_box_area_fraction debe estar entre 0 y 1")
        if self.maximum_aspect_ratio < 1:
            raise ValueError("maximum_aspect_ratio debe ser al menos 1")
        if self.maximum_speed_pixels_per_frame <= 0:
            raise ValueError(
                "maximum_speed_pixels_per_frame debe ser positivo"
            )
        if self.gate_margin_pixels < 0:
            raise ValueError("gate_margin_pixels no puede ser negativo")
        if self.maximum_missed_frames < 0:
            raise ValueError("maximum_missed_frames no puede ser negativo")
        if not 0 <= self.velocity_smoothing < 1:
            raise ValueError("velocity_smoothing debe estar entre 0 y 1")
        if not 0 <= self.velocity_damping <= 1:
            raise ValueError("velocity_damping debe estar entre 0 y 1")


@dataclass(frozen=True, slots=True)
class BallTrackObservation:
    frame_index: int
    status: BallTrackStatus
    track_id: int | None
    center_x: float | None
    center_y: float | None
    x1: float | None
    y1: float | None
    x2: float | None
    y2: float | None
    confidence: float | None
    class_id: int | None
    label: str | None
    missed_frames: int
    reacquired: bool = False


@dataclass(slots=True)
class _ActiveBall:
    track_id: int
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    last_observed_x: float
    last_observed_y: float
    width: float
    height: float
    confidence: float
    class_id: int
    label: str
    last_observed_frame: int
    missed_frames: int = 0


class BallTracker:
    """Track a single ball candidate stream and expose predicted observations.

    Candidate class labels must identify a ball unless explicit class IDs are
    supplied. IDs represent continuous track segments, not player identities.
    """

    def __init__(
        self,
        config: BallTrackConfig | None = None,
        *,
        allowed_class_ids: Sequence[int] | None = None,
    ) -> None:
        self.config = config or BallTrackConfig()
        if allowed_class_ids is not None:
            if not allowed_class_ids or any(class_id < 0 for class_id in allowed_class_ids):
                raise ValueError("allowed_class_ids debe contener IDs no negativos")
            self._allowed_class_ids = frozenset(allowed_class_ids)
        else:
            self._allowed_class_ids = None
        self.reset()

    def reset(self) -> None:
        self._active: _ActiveBall | None = None
        self._last_frame_index = -1
        self._next_track_id = 1

    @property
    def allowed_class_ids(self) -> frozenset[int] | None:
        return self._allowed_class_ids

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        frame_shape: tuple[int, ...],
    ) -> BallTrackObservation:
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise ValueError("frame_index debe ser un entero")
        if frame_index < 0:
            raise ValueError("frame_index no puede ser negativo")
        if frame_index != self._last_frame_index + 1:
            raise ValueError(
                f"Se esperaba el fotograma {self._last_frame_index + 1}; "
                f"se recibió {frame_index}"
            )
        if len(frame_shape) < 2:
            raise ValueError("frame_shape debe contener alto y ancho")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("El fotograma debe tener dimensiones positivas")
        self._last_frame_index = frame_index

        candidates = [
            item
            for item in detections
            if self._is_valid_candidate(item, width, height)
        ]

        if self._active is None:
            if not candidates:
                return self._missing(frame_index)
            return self._start_track(
                max(candidates, key=lambda item: item.confidence), frame_index
            )

        candidate = self._associate(candidates)
        if candidate is not None:
            return self._observe(candidate, frame_index)

        if self._active.missed_frames >= self.config.maximum_missed_frames:
            self._active = None
            if candidates:
                return self._start_track(
                    max(candidates, key=lambda item: item.confidence), frame_index
                )
            return self._missing(frame_index)

        prediction_x = self._active.x + self._active.velocity_x
        prediction_y = self._active.y + self._active.velocity_y
        if not (0 <= prediction_x < width and 0 <= prediction_y < height):
            self._active = None
            return self._missing(frame_index)

        self._active.x = prediction_x
        self._active.y = prediction_y
        self._active.velocity_x *= self.config.velocity_damping
        self._active.velocity_y *= self.config.velocity_damping
        self._active.missed_frames += 1
        return self._prediction(frame_index, width, height)

    def _is_valid_candidate(
        self, detection: Detection, frame_width: int, frame_height: int
    ) -> bool:
        if self._allowed_class_ids is None:
            normalized_label = detection.label.lower().replace("_", " ").strip()
            if normalized_label not in _BALL_LABELS:
                return False
        elif detection.class_id not in self._allowed_class_ids:
            return False

        values = (
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            detection.confidence,
        )
        if not all(math.isfinite(value) for value in values):
            return False
        if not 0 <= detection.confidence <= 1:
            return False
        if detection.confidence < self.config.minimum_confidence:
            return False
        box_width = detection.x2 - detection.x1
        box_height = detection.y2 - detection.y1
        if box_width <= 0 or box_height <= 0:
            return False
        if (
            detection.x1 < 0
            or detection.y1 < 0
            or detection.x2 > frame_width
            or detection.y2 > frame_height
        ):
            return False
        if (
            box_width * box_height
            > frame_width
            * frame_height
            * self.config.maximum_box_area_fraction
        ):
            return False
        aspect_ratio = max(box_width / box_height, box_height / box_width)
        return aspect_ratio <= self.config.maximum_aspect_ratio

    def _associate(self, candidates: Sequence[Detection]) -> Detection | None:
        active = self._active
        if active is None or not candidates:
            return None

        elapsed = active.missed_frames + 1
        gate = (
            self.config.maximum_speed_pixels_per_frame * elapsed
            + self.config.gate_margin_pixels
        )
        scored: list[tuple[float, int, Detection]] = []
        for index, detection in enumerate(candidates):
            center_x = (detection.x1 + detection.x2) / 2
            center_y = (detection.y1 + detection.y2) / 2
            distance = math.hypot(center_x - active.x, center_y - active.y)
            if distance <= gate:
                score = distance / max(gate, 1.0) - 0.25 * detection.confidence
                scored.append((score, index, detection))
        if not scored:
            return None
        return min(scored, key=lambda item: (item[0], item[1]))[2]

    def _start_track(
        self, detection: Detection, frame_index: int
    ) -> BallTrackObservation:
        center_x = (detection.x1 + detection.x2) / 2
        center_y = (detection.y1 + detection.y2) / 2
        self._active = _ActiveBall(
            track_id=self._next_track_id,
            x=center_x,
            y=center_y,
            velocity_x=0.0,
            velocity_y=0.0,
            last_observed_x=center_x,
            last_observed_y=center_y,
            width=detection.x2 - detection.x1,
            height=detection.y2 - detection.y1,
            confidence=detection.confidence,
            class_id=detection.class_id,
            label=detection.label,
            last_observed_frame=frame_index,
        )
        self._next_track_id += 1
        return self._observed(detection, frame_index, reacquired=False)

    def _observe(
        self, detection: Detection, frame_index: int
    ) -> BallTrackObservation:
        active = self._active
        if active is None:
            raise RuntimeError("No hay un track activo para asociar")
        reacquired = active.missed_frames > 0
        center_x = (detection.x1 + detection.x2) / 2
        center_y = (detection.y1 + detection.y2) / 2
        elapsed = frame_index - active.last_observed_frame
        measured_velocity_x = (center_x - active.last_observed_x) / elapsed
        measured_velocity_y = (center_y - active.last_observed_y) / elapsed
        smoothing = self.config.velocity_smoothing
        velocity_x = (
            smoothing * active.velocity_x
            + (1 - smoothing) * measured_velocity_x
        )
        velocity_y = (
            smoothing * active.velocity_y
            + (1 - smoothing) * measured_velocity_y
        )
        speed = math.hypot(velocity_x, velocity_y)
        maximum_speed = self.config.maximum_speed_pixels_per_frame
        if speed > maximum_speed:
            scale = maximum_speed / speed
            velocity_x *= scale
            velocity_y *= scale

        active.x = center_x
        active.y = center_y
        active.velocity_x = velocity_x
        active.velocity_y = velocity_y
        active.last_observed_x = center_x
        active.last_observed_y = center_y
        active.width = detection.x2 - detection.x1
        active.height = detection.y2 - detection.y1
        active.confidence = detection.confidence
        active.class_id = detection.class_id
        active.label = detection.label
        active.last_observed_frame = frame_index
        active.missed_frames = 0
        return self._observed(detection, frame_index, reacquired=reacquired)

    def _observed(
        self, detection: Detection, frame_index: int, *, reacquired: bool
    ) -> BallTrackObservation:
        active = self._active
        if active is None:
            raise RuntimeError("No hay un track activo")
        return BallTrackObservation(
            frame_index=frame_index,
            status="observed",
            track_id=active.track_id,
            center_x=(detection.x1 + detection.x2) / 2,
            center_y=(detection.y1 + detection.y2) / 2,
            x1=detection.x1,
            y1=detection.y1,
            x2=detection.x2,
            y2=detection.y2,
            confidence=detection.confidence,
            class_id=detection.class_id,
            label=detection.label,
            missed_frames=0,
            reacquired=reacquired,
        )

    def _prediction(
        self, frame_index: int, frame_width: int, frame_height: int
    ) -> BallTrackObservation:
        active = self._active
        if active is None:
            raise RuntimeError("No hay un track activo para predecir")
        half_width = active.width / 2
        half_height = active.height / 2
        return BallTrackObservation(
            frame_index=frame_index,
            status="predicted",
            track_id=active.track_id,
            center_x=active.x,
            center_y=active.y,
            x1=max(0.0, active.x - half_width),
            y1=max(0.0, active.y - half_height),
            x2=min(float(frame_width), active.x + half_width),
            y2=min(float(frame_height), active.y + half_height),
            confidence=None,
            class_id=active.class_id,
            label=active.label,
            missed_frames=active.missed_frames,
        )

    @staticmethod
    def _missing(frame_index: int) -> BallTrackObservation:
        return BallTrackObservation(
            frame_index=frame_index,
            status="missing",
            track_id=None,
            center_x=None,
            center_y=None,
            x1=None,
            y1=None,
            x2=None,
            y2=None,
            confidence=None,
            class_id=None,
            label=None,
            missed_frames=0,
        )
