"""Conservative CPU appearance and motion association for ByteTrack tracklets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from matchvision.detection.base import Detection
from matchvision.reidentification.base import IdentifiedObject
from matchvision.reidentification.osnet import (
    OSNetEmbeddingExtractor,
    cosine_similarity,
    extract_person_crop,
)
from matchvision.tracking.base import TrackedObject


@dataclass(slots=True)
class AppearanceDescriptor:
    color: np.ndarray
    shape: np.ndarray


@dataclass(slots=True)
class _IdentityState:
    persistent_id: int
    class_id: int
    last_seen: int
    scene_epoch: int
    box: np.ndarray
    velocity: np.ndarray
    appearance: AppearanceDescriptor | None
    embedding_gallery: deque[np.ndarray] = field(default_factory=deque)
    temporal_ids: set[int] = field(default_factory=set)
    history: deque[tuple[int, int]] = field(default_factory=deque)


@dataclass(frozen=True, slots=True)
class _Candidate:
    track: TrackedObject
    appearance: AppearanceDescriptor | None
    embedding: np.ndarray | None


class PersistentIdentityAssociator:
    """Associate new tracklets to old identities only with multi-cue evidence.

    HSV/HOG remains the default baseline. Optional OSNet embeddings and a
    weighted hybrid mode use the same temporal, motion, and scale safeguards.
    """

    def __init__(
        self,
        *,
        reid_class_ids: tuple[int, ...] = (0,),
        maximum_reidentification_gap: int = 150,
        active_identity_protection: int = 8,
        minimum_association_confidence: float = 0.82,
        ambiguity_margin: float = 0.06,
        maximum_normalized_motion: float = 5.0,
        history_length: int = 30,
        scene_cut_mean_difference: float = 28.0,
        scene_cut_histogram_distance: float = 0.60,
        reid_method: str = "hsv_hog",
        embedding_weights_path: str | Path | None = None,
        embedding_device: str = "auto",
        embedding_batch_size: int = 16,
        embedding_gallery_size: int = 5,
        embedding_sample_interval: int = 10,
        embedding_weight: float = 0.60,
        minimum_visual_similarity: float = 0.58,
    ) -> None:
        if not reid_class_ids:
            raise ValueError("reid_class_ids must include at least one class")
        if maximum_reidentification_gap < 1:
            raise ValueError("maximum_reidentification_gap must be positive")
        if active_identity_protection < 0:
            raise ValueError("active_identity_protection cannot be negative")
        if not 0 < minimum_association_confidence <= 1:
            raise ValueError("minimum_association_confidence must be in (0, 1]")
        if not 0 <= ambiguity_margin < 1:
            raise ValueError("ambiguity_margin must be in [0, 1)")
        if maximum_normalized_motion <= 0 or history_length < 1:
            raise ValueError("motion limit and history_length must be positive")
        if scene_cut_mean_difference <= 0:
            raise ValueError("scene_cut_mean_difference must be positive")
        if not 0 < scene_cut_histogram_distance <= 1:
            raise ValueError("scene_cut_histogram_distance must be in (0, 1]")
        if reid_method not in {"hsv_hog", "embedding", "hybrid"}:
            raise ValueError("El método ReID debe ser hsv_hog, embedding o hybrid.")
        if reid_method != "hsv_hog" and not embedding_weights_path:
            raise ValueError(
                "Indique los pesos OSNet para el método ReID solicitado."
            )
        if embedding_gallery_size < 1 or embedding_sample_interval < 1:
            raise ValueError(
                "El tamaño de galería y el intervalo de muestreo deben ser positivos."
            )
        if not 0 <= embedding_weight <= 1:
            raise ValueError("El peso de embeddings debe estar entre 0 y 1.")
        if not 0 <= minimum_visual_similarity <= 1:
            raise ValueError("La similitud visual mínima debe estar entre 0 y 1.")

        self._reid_class_ids = frozenset(reid_class_ids)
        self._max_gap = maximum_reidentification_gap
        self._active_protection = active_identity_protection
        self._minimum_confidence = minimum_association_confidence
        self._ambiguity_margin = ambiguity_margin
        self._max_motion = maximum_normalized_motion
        self._history_length = history_length
        self._scene_cut_mean_difference = scene_cut_mean_difference
        self._scene_cut_histogram_distance = scene_cut_histogram_distance
        self._reid_method = reid_method
        self._embedding_gallery_size = embedding_gallery_size
        self._embedding_sample_interval = embedding_sample_interval
        self._embedding_weight = embedding_weight
        self._minimum_visual_similarity = minimum_visual_similarity
        self._embedding_extractor = (
            OSNetEmbeddingExtractor(
                embedding_weights_path,
                device=embedding_device,
                batch_size=embedding_batch_size,
            )
            if reid_method != "hsv_hog"
            else None
        )
        self._identities: dict[int, _IdentityState] = {}
        self._temporal_to_persistent: dict[tuple[int, int], int] = {}
        self._next_persistent_id = 1
        self._last_frame_index = -1
        self._reidentifications = 0
        self._new_identities = 0
        self._scene_epoch = 0
        self._scene_cuts = 0
        self._previous_scene_frame: np.ndarray | None = None
        self._embedding_extraction_seconds = 0.0
        self._appearance_extraction_seconds = 0.0
        self._association_seconds = 0.0

    @property
    def configuration(self) -> dict[str, Any]:
        return {
            "strategy": (
                "HSV/HOG + motion/scale gated Hungarian assignment"
                if self._reid_method == "hsv_hog"
                else f"{self._reid_method} + motion/scale gated Hungarian assignment"
            ),
            "reid_method": self._reid_method,
            "descriptor": {
                "hsv_hog": "OpenCV HSV(16x4) histogram and HOG(64x128)",
                "embedding": "OSNet-AIN 512D L2-normalized embedding",
                "hybrid": (
                    "OSNet-AIN 512D L2-normalized embedding + "
                    "OpenCV HSV(16x4) histogram and HOG(64x128)"
                ),
            }[self._reid_method],
            "embedding": (
                self._embedding_extractor.configuration
                if self._embedding_extractor is not None
                else None
            ),
            "embedding_gallery_size": self._embedding_gallery_size,
            "embedding_sample_interval": self._embedding_sample_interval,
            "embedding_weight": self._embedding_weight,
            "minimum_visual_similarity": self._minimum_visual_similarity,
            "reid_class_ids": sorted(self._reid_class_ids),
            "maximum_reidentification_gap": self._max_gap,
            "active_identity_protection": self._active_protection,
            "minimum_association_confidence": self._minimum_confidence,
            "ambiguity_margin": self._ambiguity_margin,
            "maximum_normalized_motion": self._max_motion,
            "history_length": self._history_length,
            "scene_cut_mean_difference": self._scene_cut_mean_difference,
            "scene_cut_histogram_distance": self._scene_cut_histogram_distance,
            "cross_scene_reidentification": False,
        }

    @property
    def reidentification_count(self) -> int:
        return self._reidentifications

    @property
    def new_identity_count(self) -> int:
        return self._new_identities

    @property
    def scene_cut_count(self) -> int:
        return self._scene_cuts

    @property
    def runtime_metrics(self) -> dict[str, float | int]:
        extractor = self._embedding_extractor
        return {
            "embedding_extraction_seconds": self._embedding_extraction_seconds,
            "embedding_inference_seconds": (
                extractor.inference_seconds if extractor is not None else 0.0
            ),
            "embedding_images_processed": (
                extractor.images_processed if extractor is not None else 0
            ),
            "hsv_hog_extraction_seconds": self._appearance_extraction_seconds,
            "association_seconds": self._association_seconds,
        }

    def update(
        self,
        tracks: list[TrackedObject],
        frame: np.ndarray,
        frame_index: int,
    ) -> list[IdentifiedObject]:
        if frame_index != self._last_frame_index + 1:
            raise ValueError(
                f"Expected consecutive frame index {self._last_frame_index + 1}, "
                f"got {frame_index}"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR frame with shape (height, width, 3)")
        self._last_frame_index = frame_index
        embedding_started = time.perf_counter()
        if self._is_scene_cut(frame):
            self._scene_epoch += 1
            self._scene_cuts += 1
            self._temporal_to_persistent.clear()

        results: list[IdentifiedObject | None] = [None] * len(tracks)
        candidates: list[_Candidate] = []
        keys = [
            (track.detection.class_id, track.track_id)
            for track in tracks
        ]
        embedding_indices: list[int] = []
        embedding_crops: list[np.ndarray] = []
        if self._embedding_extractor is not None:
            for index, track in enumerate(tracks):
                if track.detection.class_id not in self._reid_class_ids:
                    continue
                if (
                    keys[index] not in self._temporal_to_persistent
                    or frame_index % self._embedding_sample_interval == 0
                ):
                    crop = extract_person_crop(
                        frame,
                        (
                            track.detection.x1,
                            track.detection.y1,
                            track.detection.x2,
                            track.detection.y2,
                        ),
                    )
                    if crop is not None:
                        embedding_indices.append(index)
                        embedding_crops.append(crop)
        embeddings: dict[int, np.ndarray] = {}
        if embedding_crops and self._embedding_extractor is not None:
            embeddings = dict(
                zip(
                    embedding_indices,
                    self._embedding_extractor.encode(embedding_crops),
                    strict=True,
                )
            )
        self._embedding_extraction_seconds += time.perf_counter() - embedding_started

        for index, track in enumerate(tracks):
            key = keys[index]
            embedding = embeddings.get(index)
            appearance = (
                self._timed_extract_appearance(frame, track.detection)
                if (
                    track.detection.class_id in self._reid_class_ids
                    and self._reid_method != "embedding"
                )
                else None
            )
            persistent_id = self._temporal_to_persistent.get(key)
            if persistent_id is None:
                candidates.append(_Candidate(track, appearance, embedding))
                continue
            state = self._identities[persistent_id]
            self._update_state(
                state, track.detection, appearance, frame_index, embedding
            )
            results[index] = self._to_identified(
                track, state, identity_state="tracked", confidence=None
            )

        association_started = time.perf_counter()
        candidate_scores, candidate_states = self._score_candidates(
            candidates, frame_index
        )
        assigned: dict[int, tuple[int, float]] = {}
        if candidate_scores.size:
            eligible = candidate_scores >= self._minimum_confidence
            for row in range(candidate_scores.shape[0]):
                row_scores = candidate_scores[row, eligible[row]]
                if row_scores.size > 1:
                    best = float(np.max(row_scores))
                    second = float(np.partition(row_scores, -2)[-2])
                    if best - second < self._ambiguity_margin:
                        eligible[row, :] = False
            for column in range(candidate_scores.shape[1]):
                column_scores = candidate_scores[eligible[:, column], column]
                if column_scores.size > 1:
                    best = float(np.max(column_scores))
                    second = float(np.partition(column_scores, -2)[-2])
                    if best - second < self._ambiguity_margin:
                        eligible[:, column] = False
            cost = np.where(eligible, 1.0 - candidate_scores, 1e6)
            row_indices, column_indices = linear_sum_assignment(cost)
            for row, column in zip(row_indices, column_indices, strict=True):
                if eligible[row, column]:
                    assigned[int(row)] = (
                        int(candidate_states[column]),
                        float(candidate_scores[row, column]),
                    )

        candidate_index = 0
        for index, track in enumerate(tracks):
            if results[index] is not None:
                continue
            candidate = candidates[candidate_index]
            candidate_index += 1
            reidentified = assigned.get(candidate_index - 1)
            if reidentified is None:
                state = self._new_identity(
                    track, candidate.appearance, candidate.embedding, frame_index
                )
                state_name = "new"
                confidence = None
                self._new_identities += 1
            else:
                persistent_id, confidence = reidentified
                state = self._identities[persistent_id]
                self._update_state(
                    state,
                    track.detection,
                    candidate.appearance,
                    frame_index,
                    candidate.embedding,
                )
                state_name = "reidentified"
                self._reidentifications += 1
            key = (track.detection.class_id, track.track_id)
            self._temporal_to_persistent[key] = state.persistent_id
            state.temporal_ids.add(track.track_id)
            results[index] = self._to_identified(
                track, state, identity_state=state_name, confidence=confidence
            )

        self._association_seconds += time.perf_counter() - association_started
        return [item for item in results if item is not None]

    def _timed_extract_appearance(
        self, frame: np.ndarray, detection: Detection
    ) -> AppearanceDescriptor | None:
        started = time.perf_counter()
        result = self._extract_appearance(frame, detection)
        self._appearance_extraction_seconds += time.perf_counter() - started
        return result

    def _score_candidates(
        self, candidates: list[_Candidate], frame_index: int
    ) -> tuple[np.ndarray, list[int]]:
        states = [
            state
            for state in self._identities.values()
            if state.class_id in self._reid_class_ids
            and state.scene_epoch == self._scene_epoch
            and 0 < frame_index - state.last_seen <= self._max_gap
            and frame_index - state.last_seen > self._active_protection
            and (
                state.appearance is not None
                or bool(state.embedding_gallery)
            )
        ]
        states.sort(key=lambda item: item.persistent_id)
        scores = np.full((len(candidates), len(states)), -1.0, dtype=np.float64)
        for row, candidate in enumerate(candidates):
            detection = candidate.track.detection
            if detection.class_id not in self._reid_class_ids:
                continue
            for column, state in enumerate(states):
                gap = frame_index - state.last_seen
                visual_similarity = self._visual_similarity(candidate, state)
                if (
                    visual_similarity is None
                    or visual_similarity < self._minimum_visual_similarity
                ):
                    continue
                motion, scale = self._motion_similarity(
                    detection, state, gap
                )
                normalized_motion = self._normalized_motion(
                    detection, state, gap
                )
                if normalized_motion > self._max_motion:
                    continue
                scores[row, column] = (
                    0.55 * visual_similarity
                    + 0.25 * motion
                    + 0.10 * scale
                    + 0.10 * detection.confidence
                )
        return scores, [state.persistent_id for state in states]

    def _visual_similarity(
        self, candidate: _Candidate, state: _IdentityState
    ) -> float | None:
        hsv_score: float | None = None
        embedding_score: float | None = None
        if (
            self._reid_method in {"hsv_hog", "hybrid"}
            and candidate.appearance is not None
            and state.appearance is not None
        ):
            color, shape = self._appearance_similarity(
                candidate.appearance, state.appearance
            )
            if self._reid_method == "hsv_hog" and (color < 0.30 or shape < 0.30):
                return None
            hsv_score = 0.5 * color + 0.5 * shape
        if (
            self._reid_method in {"embedding", "hybrid"}
            and candidate.embedding is not None
            and state.embedding_gallery
        ):
            similarities = sorted(
                (
                    cosine_similarity(candidate.embedding, gallery)
                    for gallery in state.embedding_gallery
                ),
                reverse=True,
            )
            embedding_score = float(
                np.mean(similarities[:3])
            )
            embedding_score = (embedding_score + 1.0) / 2.0
        if self._reid_method == "hsv_hog":
            return hsv_score
        if self._reid_method == "embedding":
            return embedding_score
        if hsv_score is None or embedding_score is None:
            return None
        return (
            self._embedding_weight * embedding_score
            + (1.0 - self._embedding_weight) * hsv_score
        )

    @staticmethod
    def _appearance_similarity(
        current: AppearanceDescriptor, previous: AppearanceDescriptor
    ) -> tuple[float, float]:
        return compare_appearance_descriptors(current, previous)

    @staticmethod
    def _box(detection: Detection) -> np.ndarray:
        return np.asarray(
            [detection.x1, detection.y1, detection.x2, detection.y2],
            dtype=np.float32,
        )

    @classmethod
    def _normalized_motion(
        cls, detection: Detection, state: _IdentityState, gap: int
    ) -> float:
        box = cls._box(detection)
        old_center = (state.box[:2] + state.box[2:]) * 0.5
        new_center = (box[:2] + box[2:]) * 0.5
        predicted_center = old_center + state.velocity * gap
        scale = max(
            float(state.box[3] - state.box[1]),
            float(box[3] - box[1]),
            12.0,
        )
        uncertainty = scale * (0.75 + 0.08 * gap)
        return float(np.linalg.norm(new_center - predicted_center) / uncertainty)

    @classmethod
    def _motion_similarity(
        cls, detection: Detection, state: _IdentityState, gap: int
    ) -> tuple[float, float]:
        normalized = cls._normalized_motion(detection, state, gap)
        motion = float(np.exp(-0.5 * (normalized / 2.5) ** 2))
        box = cls._box(detection)
        old_area = max(
            float((state.box[2] - state.box[0]) * (state.box[3] - state.box[1])),
            1.0,
        )
        new_area = max(
            float((box[2] - box[0]) * (box[3] - box[1])),
            1.0,
        )
        scale = float(np.exp(-0.5 * abs(np.log(new_area / old_area))))
        return motion, scale

    @staticmethod
    def _extract_appearance(
        frame: np.ndarray, detection: Detection
    ) -> AppearanceDescriptor | None:
        return extract_appearance_descriptor(
            frame,
            (detection.x1, detection.y1, detection.x2, detection.y2),
        )

    def _new_identity(
        self,
        track: TrackedObject,
        appearance: AppearanceDescriptor | None,
        embedding: np.ndarray | None,
        frame_index: int,
    ) -> _IdentityState:
        persistent_id = self._next_persistent_id
        self._next_persistent_id += 1
        state = _IdentityState(
            persistent_id=persistent_id,
            class_id=track.detection.class_id,
            last_seen=frame_index,
            scene_epoch=self._scene_epoch,
            box=self._box(track.detection),
            velocity=np.zeros(2, dtype=np.float32),
            appearance=appearance,
            embedding_gallery=deque(
                maxlen=self._embedding_gallery_size
            ),
            history=deque(maxlen=self._history_length),
        )
        if embedding is not None:
            state.embedding_gallery.append(embedding.copy())
        self._identities[persistent_id] = state
        self._update_history(state, track.detection)
        return state

    def _update_state(
        self,
        state: _IdentityState,
        detection: Detection,
        appearance: AppearanceDescriptor | None,
        frame_index: int,
        embedding: np.ndarray | None = None,
    ) -> None:
        new_box = self._box(detection)
        elapsed = max(frame_index - state.last_seen, 1)
        old_center = (state.box[:2] + state.box[2:]) * 0.5
        new_center = (new_box[:2] + new_box[2:]) * 0.5
        observed_velocity = (new_center - old_center) / elapsed
        state.velocity = 0.5 * state.velocity + 0.5 * observed_velocity
        state.box = new_box
        state.last_seen = frame_index
        if appearance is not None:
            if state.appearance is None:
                state.appearance = appearance
            else:
                state.appearance.color = (
                    0.8 * state.appearance.color + 0.2 * appearance.color
                )
                state.appearance.color /= max(
                    float(state.appearance.color.sum()), 1e-12
                )
                state.appearance.shape = (
                    0.8 * state.appearance.shape + 0.2 * appearance.shape
                )
                norm = float(np.linalg.norm(state.appearance.shape))
                if norm > 0:
                    state.appearance.shape /= norm
        if embedding is not None:
            state.embedding_gallery.append(embedding.copy())
        self._update_history(state, detection)

    @staticmethod
    def _update_history(state: _IdentityState, detection: Detection) -> None:
        center = (
            int(round((detection.x1 + detection.x2) * 0.5)),
            int(round((detection.y1 + detection.y2) * 0.5)),
        )
        if not state.history or state.history[-1] != center:
            state.history.append(center)

    @staticmethod
    def _to_identified(
        track: TrackedObject,
        state: _IdentityState,
        *,
        identity_state: str,
        confidence: float | None,
    ) -> IdentifiedObject:
        return IdentifiedObject(
            detection=track.detection,
            track_id=track.track_id,
            persistent_id=state.persistent_id,
            identity_state=identity_state,
            association_confidence=confidence,
            history=tuple(state.history),
        )

    def _is_scene_cut(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        previous = self._previous_scene_frame
        self._previous_scene_frame = small
        if previous is None:
            return False
        mean_difference = float(
            cv2.absdiff(
                cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), gray
            ).mean()
        )
        previous_histogram = cv2.calcHist(
            [cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)],
            [0],
            None,
            [32],
            [0, 256],
        )
        cv2.normalize(
            previous_histogram,
            previous_histogram,
            alpha=1.0,
            norm_type=cv2.NORM_L1,
        )
        histogram_distance = float(
            cv2.compareHist(
                previous_histogram,
                histogram,
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )
        return (
            mean_difference >= self._scene_cut_mean_difference
            and histogram_distance >= self._scene_cut_histogram_distance
        )

_HOG_DESCRIPTOR = cv2.HOGDescriptor(
    (64, 128),
    (16, 16),
    (8, 8),
    (8, 8),
    9,
)


def extract_appearance_descriptor(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
) -> AppearanceDescriptor | None:
    """Extract the same HSV/HOG descriptor used by the online associator."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a BGR frame with shape (height, width, 3)")
    if len(box) != 4 or not np.isfinite(box).all():
        raise ValueError("box must contain four finite coordinates")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box coordinates must have positive area")
    height, width = frame.shape[:2]
    x1 = max(0, min(width, int(np.floor(x1))))
    y1 = max(0, min(height, int(np.floor(y1))))
    x2 = max(0, min(width, int(np.ceil(x2))))
    y2 = max(0, min(height, int(np.ceil(y2))))
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    torso_start = int(crop.shape[0] * 0.08)
    torso_end = max(torso_start + 1, int(crop.shape[0] * 0.78))
    hsv = cv2.cvtColor(crop[torso_start:torso_end], cv2.COLOR_BGR2HSV)
    color = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256])
    color = color.astype(np.float32).reshape(-1)
    color_sum = float(color.sum())
    if color_sum <= 0:
        return None
    color /= color_sum

    resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    shape = _HOG_DESCRIPTOR.compute(gray).reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(shape))
    if norm <= 0:
        return None
    shape /= norm
    return AppearanceDescriptor(color=color, shape=shape)


def compare_appearance_descriptors(
    first: AppearanceDescriptor,
    second: AppearanceDescriptor,
) -> tuple[float, float]:
    """Return HSV intersection and HOG cosine similarities, not probabilities."""
    color = float(np.minimum(first.color, second.color).sum())
    shape = float(np.clip(np.dot(first.shape, second.shape), 0.0, 1.0))
    return color, shape
