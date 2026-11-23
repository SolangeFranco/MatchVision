"""Class-isolated adapter for Ultralytics' ByteTrack implementation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from matchvision.detection.base import Detection
from matchvision.tracking.base import TrackedObject


class _DetectionBatch:
    """NumPy-backed results object accepted by Ultralytics BYTETracker."""

    def __init__(
        self, xyxy: np.ndarray, confidence: np.ndarray, class_ids: np.ndarray
    ) -> None:
        self.xyxy = xyxy
        self.conf = confidence
        self.cls = class_ids.astype(np.float32, copy=False)
        self.xywh = xyxy.copy()
        self.xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
        self.xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
        self.xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        self.xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, selection: np.ndarray) -> _DetectionBatch:
        return _DetectionBatch(
            self.xyxy[selection],
            self.conf[selection],
            self.cls[selection],
        )


class ByteTrackAdapter:
    """Use Ultralytics BYTETracker and keep IDs isolated by model class."""

    def __init__(
        self,
        class_names: dict[int, str],
        *,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
    ) -> None:
        try:
            from ultralytics import __version__ as ultralytics_version
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as error:
            raise RuntimeError(
                "The ByteTrack adapter requires Ultralytics. Install with "
                "`python -m pip install -e \".[vision]\"`."
            ) from error
        if lost_track_buffer < 0:
            raise ValueError("lost_track_buffer cannot be negative")
        if not 0 < track_activation_threshold <= 1:
            raise ValueError("track_activation_threshold must be in (0, 1]")
        if not 0 < minimum_matching_threshold <= 1:
            raise ValueError("minimum_matching_threshold must be in (0, 1]")
        if not class_names:
            raise ValueError("class_names must include at least one tracked class")

        self._version = f"Ultralytics {ultralytics_version}"
        self._configuration = {
            "class_names": class_names.copy(),
            "track_activation_threshold": track_activation_threshold,
            "lost_track_buffer": lost_track_buffer,
            "minimum_matching_threshold": minimum_matching_threshold,
        }
        self._trackers: dict[int, Any] = {}
        for class_id in class_names:
            args = SimpleNamespace(
                track_high_thresh=track_activation_threshold,
                track_low_thresh=min(0.1, track_activation_threshold),
                new_track_thresh=track_activation_threshold,
                track_buffer=lost_track_buffer,
                match_thresh=minimum_matching_threshold,
                fuse_score=True,
            )
            self._trackers[class_id] = BYTETracker(args)
        self._names = class_names.copy()
        self._last_frame_index = -1
        self._global_ids: dict[tuple[int, int], int] = {}
        self._next_global_id = 1

    @property
    def tracker_name(self) -> str:
        return "ByteTrack"

    @property
    def tracker_version(self) -> str:
        return self._version

    @property
    def configuration(self) -> dict[str, Any]:
        return self._configuration.copy()

    def update(
        self, detections: list[Detection], frame_index: int
    ) -> list[TrackedObject]:
        if frame_index != self._last_frame_index + 1:
            raise ValueError(
                f"Expected consecutive frame index {self._last_frame_index + 1}, "
                f"got {frame_index}"
            )
        unknown_class_ids = {item.class_id for item in detections} - set(
            self._trackers
        )
        if unknown_class_ids:
            raise ValueError(
                f"No tracker configured for class IDs {sorted(unknown_class_ids)}"
            )
        self._last_frame_index = frame_index
        tracked: list[TrackedObject] = []

        for class_id, tracker in self._trackers.items():
            selected = [item for item in detections if item.class_id == class_id]
            if selected:
                xyxy = np.asarray(
                    [[item.x1, item.y1, item.x2, item.y2] for item in selected],
                    dtype=np.float32,
                )
                confidence = np.asarray(
                    [item.confidence for item in selected], dtype=np.float32
                )
                classes = np.full(len(selected), class_id, dtype=np.int32)
            else:
                xyxy = np.empty((0, 4), dtype=np.float32)
                confidence = np.empty((0,), dtype=np.float32)
                classes = np.empty((0,), dtype=np.int32)

            output = tracker.update(_DetectionBatch(xyxy, confidence, classes))
            for row in output:
                box = row[:4]
                local_id = int(row[4])
                score = float(row[5])
                output_class_id = int(row[6])
                detection_index = int(row[7])
                if detection_index < 0 or detection_index >= len(selected):
                    raise RuntimeError(
                        f"ByteTrack returned invalid detection index {detection_index}"
                    )
                source_detection = selected[detection_index]
                identity_key = (output_class_id, int(local_id))
                global_id = self._global_ids.get(identity_key)
                if global_id is None:
                    global_id = self._next_global_id
                    self._global_ids[identity_key] = global_id
                    self._next_global_id += 1
                tracked.append(
                    TrackedObject(
                        Detection(
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                            confidence=score,
                            class_id=output_class_id,
                            label=source_detection.label,
                        ),
                        global_id,
                    )
                )

        return tracked
