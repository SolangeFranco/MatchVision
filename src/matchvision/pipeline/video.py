"""Frame-by-frame detection, tracking, MOT export, and visualization."""

from __future__ import annotations

import hashlib
import csv
import json
import os
import platform
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from matchvision.detection.base import Detection, Detector
from matchvision.evaluation.mot import evaluate_mot_sequence
from matchvision.reidentification.base import IdentifiedObject, IdentityAssigner
from matchvision.pipeline.detection_cache import DetectionSegmentCache
from matchvision.tracking.base import MultiObjectTracker, TrackedObject


@dataclass(frozen=True, slots=True)
class VideoRunResult:
    run_id: str
    frames_processed: int
    detections_tracked: int
    processing_seconds: float
    processing_fps: float
    mot_path: Path
    annotated_video_path: Path
    metadata_path: Path
    metrics: dict[str, float | int] | None
    trajectory_path: Path | None = None
    tracking_stats_path: Path | None = None
    detections_path: Path | None = None
    track_observations_path: Path | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
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


def _draw_tracks(frame: np.ndarray, tracks: list[TrackedObject]) -> np.ndarray:
    annotated = frame.copy()
    for item in tracks:
        detection = item.detection
        normalized_label = detection.label.lower().replace("_", " ")
        is_ball = normalized_label in {"ball", "sports ball", "soccer ball", "football"}
        color = (0, 165, 255) if is_ball else (60, 220, 60)
        x1, y1, x2, y2 = (
            int(round(detection.x1)),
            int(round(detection.y1)),
            int(round(detection.x2)),
            int(round(detection.y2)),
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{detection.label} ID:{item.track_id} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 7, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _identity_color(persistent_id: int) -> tuple[int, int, int]:
    hue = (persistent_id * 67) % 180
    pixel = np.asarray([[[hue, 210, 235]]], dtype=np.uint8)
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_identified_tracks(
    frame: np.ndarray, tracks: list[IdentifiedObject]
) -> np.ndarray:
    annotated = frame.copy()
    for item in tracks:
        color = _identity_color(item.persistent_id)
        history = item.history
        for start, end in zip(history, history[1:]):
            cv2.line(annotated, start, end, color, 2, cv2.LINE_AA)
        detection = item.detection
        x1, y1, x2, y2 = (
            int(round(detection.x1)),
            int(round(detection.y1)),
            int(round(detection.x2)),
            int(round(detection.y2)),
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        association = (
            f" A:{item.association_confidence:.2f}"
            if item.association_confidence is not None
            else ""
        )
        text = (
            f"{detection.label} T:{item.track_id} P:{item.persistent_id} "
            f"{item.identity_state}{association} D:{detection.confidence:.2f}"
        )
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 7, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


class _IdentityStatisticsAccumulator:
    """Aggregate identity diagnostics without retaining every detection row."""

    def __init__(self) -> None:
        self.tracked_observations = 0
        self.identity_observations: Counter[int] = Counter()
        self.identity_temporal_ids: dict[int, set[int]] = defaultdict(set)
        self.last_frame_by_identity: dict[int, int] = {}
        self.temporal_to_identity: dict[int, set[int]] = defaultdict(set)
        self.identities_by_class: dict[int, set[int]] = defaultdict(set)
        self.temporal_ids_by_class: dict[int, set[int]] = defaultdict(set)
        self.detections_by_frame: Counter[int] = Counter()
        self.temporal_tracklets_merged = 0
        self.gap_events = 0
        self.missing_frames = 0
        self.association_score_count = 0
        self.association_score_sum = 0.0
        self.minimum_association_score: float | None = None

    def add(self, record: dict[str, Any]) -> None:
        persistent_id = int(record["persistent_id"])
        temporal_id = int(record["temporal_track_id"])
        class_id = int(record["class_id"])
        frame_index = int(record["frame"])

        self.tracked_observations += 1
        self.detections_by_frame[frame_index] += 1
        self.temporal_to_identity[temporal_id].add(persistent_id)
        self.identities_by_class[class_id].add(persistent_id)
        self.temporal_ids_by_class[class_id].add(temporal_id)

        temporal_ids = self.identity_temporal_ids[persistent_id]
        if temporal_id not in temporal_ids:
            if temporal_ids:
                self.temporal_tracklets_merged += 1
            temporal_ids.add(temporal_id)

        last_frame = self.last_frame_by_identity.get(persistent_id)
        if last_frame is None or frame_index > last_frame:
            if last_frame is not None and frame_index > last_frame + 1:
                self.gap_events += 1
                self.missing_frames += frame_index - last_frame - 1
            self.identity_observations[persistent_id] += 1
            self.last_frame_by_identity[persistent_id] = frame_index

        score = record["association_confidence"]
        if score != "":
            value = float(score)
            self.association_score_count += 1
            self.association_score_sum += value
            if (
                self.minimum_association_score is None
                or value < self.minimum_association_score
            ):
                self.minimum_association_score = value


def _identity_statistics(
    records: "_IdentityStatisticsAccumulator",
    frames_processed: int,
    identity_assigner: IdentityAssigner,
) -> dict[str, Any]:
    identity_observations = list(records.identity_observations.values())
    tracklets_per_identity = [
        len(track_ids) for track_ids in records.identity_temporal_ids.values()
    ]
    detections_per_frame = list(records.detections_by_frame.values())
    return {
        "frames_processed": frames_processed,
        "tracked_observations": records.tracked_observations,
        "temporal_track_ids": len(records.temporal_to_identity),
        "persistent_identity_ids": len(records.identity_observations),
        "temporal_track_ids_by_class": {
            str(class_id): len(ids)
            for class_id, ids in sorted(records.temporal_ids_by_class.items())
        },
        "persistent_identity_ids_by_class": {
            str(class_id): len(ids)
            for class_id, ids in sorted(records.identities_by_class.items())
        },
        "temporal_tracklets_merged_proxy": records.temporal_tracklets_merged,
        "identities_with_multiple_temporal_tracklets": sum(
            count > 1 for count in tracklets_per_identity
        ),
        "temporal_tracklets_per_identity": {
            "mean": (
                float(np.mean(tracklets_per_identity))
                if tracklets_per_identity
                else 0.0
            ),
            "maximum": max(tracklets_per_identity, default=0),
        },
        "reidentification_association_score": {
            "count": records.association_score_count,
            "mean": (
                records.association_score_sum / records.association_score_count
                if records.association_score_count
                else None
            ),
            "minimum": records.minimum_association_score,
        },
        "reidentification_events": getattr(
            identity_assigner, "reidentification_count", None
        ),
        "new_identity_events": getattr(identity_assigner, "new_identity_count", None),
        "scene_cuts": getattr(identity_assigner, "scene_cut_count", None),
        "persistent_identity_observations": {
            "mean": (
                float(np.mean(identity_observations))
                if identity_observations
                else 0.0
            ),
            "median": (
                float(np.median(identity_observations))
                if identity_observations
                else 0.0
            ),
            "minimum": min(identity_observations, default=0),
            "maximum": max(identity_observations, default=0),
        },
        "within_identity_gap_proxy": {
            "absence_events": records.gap_events,
            "missing_frames": records.missing_frames,
        },
        "detections_per_frame": {
            "mean_over_all_video_frames": (
                records.tracked_observations / frames_processed
                if frames_processed
                else 0.0
            ),
            "mean_nonempty_frame": (
                float(np.mean(detections_per_frame)) if detections_per_frame else 0.0
            ),
            "empty_frames": frames_processed - len(records.detections_by_frame),
            "frames_with_observations": len(records.detections_by_frame),
            "maximum": max(detections_per_frame, default=0),
        },
        "metrics_status": (
            "Descriptive proxies only; not identity accuracy or MOT metrics."
        ),
    }


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    detector: Detector,
    tracker: MultiObjectTracker,
    *,
    ground_truth_path: str | Path | None = None,
    identity_assigner: IdentityAssigner | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_segment_frames: int = 300,
) -> VideoRunResult:
    """Process a local video sequentially and persist MOT, MP4, and run metadata."""
    source = Path(video_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Video has invalid dimensions: {source}")
    output_fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else 25.0
    source_sha256 = _sha256_file(source)
    checkpoint_cache = None
    if checkpoint_dir is not None:
        detector_checkpoint = detector.checkpoint_path
        try:
            checkpoint_cache = DetectionSegmentCache(
                checkpoint_dir,
                cache_key={
                    "video_sha256": source_sha256,
                    "width": width,
                    "height": height,
                    "source_fps": source_fps,
                    "declared_frame_count": declared_frames,
                    "detector": detector.configuration,
                    "model_name": detector.model_name,
                    "model_version": detector.model_version,
                    "model_checkpoint_sha256": (
                        _sha256_file(detector_checkpoint)
                        if detector_checkpoint is not None
                        and detector_checkpoint.is_file()
                        else None
                    ),
                    "class_names": detector.class_names,
                },
                segment_frames=checkpoint_segment_frames,
            )
        except Exception:
            capture.release()
            raise

    run_id = str(uuid.uuid4())
    mot_path = destination / f"{run_id}.txt"
    annotated_path = destination / f"{run_id}_annotated.mp4"
    metadata_path = destination / f"{run_id}.json"
    detections_path = destination / f"{run_id}_detections.csv"
    track_observations_path = destination / f"{run_id}_tracks.csv"
    trajectory_path = (
        destination / f"{run_id}_trajectories.csv"
        if identity_assigner is not None
        else None
    )
    tracking_stats_path = (
        destination / f"{run_id}_tracking_stats.json"
        if identity_assigner is not None
        else None
    )
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not create annotated output: {annotated_path}")

    frames_processed = 0
    tracked_detections = 0
    identity_records = _IdentityStatisticsAccumulator()
    stage_seconds = {
        "video_decode": 0.0,
        "detection": 0.0,
        "tracking": 0.0,
        "reidentification": 0.0,
        "render_and_export": 0.0,
    }
    if checkpoint_cache is not None:
        stage_seconds["detection_cache_read"] = 0.0
        stage_seconds["detection_cache_write"] = 0.0
    active_segment_start: int | None = None
    active_segment_cached = False
    active_segment_detections: list[list[Detection]] = []
    cached_segment_count = 0
    cached_frame_count = 0
    uncached_segment_count = 0
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            mot_stream = stack.enter_context(
                mot_path.open("w", encoding="utf-8", newline="")
            )
            detection_writer = csv.DictWriter(
                stack.enter_context(
                    detections_path.open("w", encoding="utf-8", newline="")
                ),
                fieldnames=[
                    "frame",
                    "class_id",
                    "label",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ],
            )
            detection_writer.writeheader()
            track_observation_writer = csv.DictWriter(
                stack.enter_context(
                    track_observations_path.open(
                        "w", encoding="utf-8", newline=""
                    )
                ),
                fieldnames=[
                    "frame",
                    "track_id",
                    "temporal_track_id",
                    "persistent_id",
                    "class_id",
                    "label",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ],
            )
            track_observation_writer.writeheader()
            trajectory_writer: csv.DictWriter | None = None
            if trajectory_path is not None:
                trajectory_stream = stack.enter_context(
                    trajectory_path.open("w", encoding="utf-8", newline="")
                )
                trajectory_writer = csv.DictWriter(
                    trajectory_stream,
                    fieldnames=[
                        "frame",
                        "temporal_track_id",
                        "persistent_id",
                        "class_id",
                        "label",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "confidence",
                        "identity_state",
                        "association_confidence",
                    ],
                )
                trajectory_writer.writeheader()
            while True:
                stage_started = time.perf_counter()
                success, frame = capture.read()
                stage_seconds["video_decode"] += time.perf_counter() - stage_started
                if not success:
                    break
                if checkpoint_cache is not None:
                    segment_start = (
                        frames_processed // checkpoint_segment_frames
                    ) * checkpoint_segment_frames
                    if segment_start != active_segment_start:
                        if active_segment_start is not None and not active_segment_cached:
                            stage_started = time.perf_counter()
                            checkpoint_cache.save(
                                active_segment_start, active_segment_detections
                            )
                            stage_seconds["detection_cache_write"] += (
                                time.perf_counter() - stage_started
                            )
                            uncached_segment_count += 1
                        stage_started = time.perf_counter()
                        active_segment_detections = checkpoint_cache.load(segment_start)
                        stage_seconds["detection_cache_read"] += (
                            time.perf_counter() - stage_started
                        )
                        active_segment_cached = active_segment_detections is not None
                        if active_segment_cached:
                            cached_segment_count += 1
                            cached_frame_count += len(active_segment_detections)
                            active_segment_detections = active_segment_detections or []
                        else:
                            active_segment_detections = []
                        active_segment_start = segment_start
                stage_started = time.perf_counter()
                if checkpoint_cache is not None and active_segment_cached:
                    segment_offset = frames_processed - active_segment_start
                    if segment_offset >= len(active_segment_detections):
                        raise ValueError(
                            "El checkpoint termina antes que el segmento del video; "
                            "no se reutilizó por seguridad."
                        )
                    detections = active_segment_detections[segment_offset]
                else:
                    detections = detector.detect(frame)
                    if checkpoint_cache is not None:
                        active_segment_detections.append(detections)
                stage_seconds["detection"] += time.perf_counter() - stage_started
                for detection in detections:
                    detection_writer.writerow(
                        {
                            "frame": frames_processed + 1,
                            "class_id": detection.class_id,
                            "label": detection.label,
                            "confidence": detection.confidence,
                            "x1": detection.x1,
                            "y1": detection.y1,
                            "x2": detection.x2,
                            "y2": detection.y2,
                        }
                    )
                stage_started = time.perf_counter()
                tracked = tracker.update(detections, frame_index=frames_processed)
                stage_seconds["tracking"] += time.perf_counter() - stage_started
                tracked_detections += len(tracked)
                stage_started = time.perf_counter()
                identified = None
                if identity_assigner is not None:
                    identified = identity_assigner.update(
                        tracked, frame, frame_index=frames_processed
                    )
                stage_seconds["reidentification"] += (
                    time.perf_counter() - stage_started
                )
                output_tracks: list[TrackedObject | IdentifiedObject] = (
                    identified if identified is not None else tracked
                )
                stage_started = time.perf_counter()
                for track in output_tracks:
                    detection = track.detection
                    output_id = (
                        track.persistent_id
                        if isinstance(track, IdentifiedObject)
                        else track.track_id
                    )
                    x = max(0.0, detection.x1)
                    y = max(0.0, detection.y1)
                    box_width = max(0.0, detection.x2 - detection.x1)
                    box_height = max(0.0, detection.y2 - detection.y1)
                    mot_stream.write(
                        f"{frames_processed + 1},{output_id},"
                        f"{x:.3f},{y:.3f},{box_width:.3f},{box_height:.3f},"
                        f"{detection.confidence:.6f},-1,-1,-1\n"
                    )
                    track_observation_writer.writerow(
                        {
                            "frame": frames_processed + 1,
                            "track_id": output_id,
                            "temporal_track_id": track.track_id,
                            "persistent_id": (
                                track.persistent_id
                                if isinstance(track, IdentifiedObject)
                                else ""
                            ),
                            "class_id": detection.class_id,
                            "label": detection.label,
                            "confidence": detection.confidence,
                            "x1": detection.x1,
                            "y1": detection.y1,
                            "x2": detection.x2,
                            "y2": detection.y2,
                        }
                    )
                    if isinstance(track, IdentifiedObject):
                        record = {
                            "frame": frames_processed + 1,
                            "temporal_track_id": track.track_id,
                            "persistent_id": track.persistent_id,
                            "class_id": detection.class_id,
                            "label": detection.label,
                            "x1": round(detection.x1, 3),
                            "y1": round(detection.y1, 3),
                            "x2": round(detection.x2, 3),
                            "y2": round(detection.y2, 3),
                            "confidence": round(detection.confidence, 6),
                            "identity_state": track.identity_state,
                            "association_confidence": (
                                ""
                                if track.association_confidence is None
                                else round(track.association_confidence, 6)
                            ),
                        }
                        identity_records.add(record)
                        if trajectory_writer is not None:
                            trajectory_writer.writerow(record)
                writer.write(
                    _draw_identified_tracks(frame, identified)
                    if identified is not None
                    else _draw_tracks(frame, tracked)
                )
                stage_seconds["render_and_export"] += (
                    time.perf_counter() - stage_started
                )
                frames_processed += 1
            if (
                checkpoint_cache is not None
                and active_segment_start is not None
                and not active_segment_cached
            ):
                stage_started = time.perf_counter()
                checkpoint_cache.save(
                    active_segment_start, active_segment_detections
                )
                stage_seconds["detection_cache_write"] += (
                    time.perf_counter() - stage_started
                )
                uncached_segment_count += 1
    finally:
        capture.release()
        writer.release()

    processing_seconds = time.perf_counter() - started
    processing_fps = (
        frames_processed / processing_seconds if processing_seconds > 0 else 0.0
    )
    metrics = None
    if ground_truth_path is not None:
        metrics = evaluate_mot_sequence(
            ground_truth_path,
            mot_path,
            frame_count=frames_processed,
        )

    checkpoint_path = detector.checkpoint_path
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "input": {
            "video_path": str(source),
            "video_sha256": source_sha256,
            "width": width,
            "height": height,
            "source_fps": source_fps,
            "declared_frame_count": declared_frames,
            "frames_processed": frames_processed,
        },
        "configuration": {
            "detector": detector.configuration,
            "tracker": tracker.configuration,
        },
        "model": {
            "name": detector.model_name,
            "version": detector.model_version,
            "source": detector.model_source,
            "license": detector.model_license,
            "pretrained_on": detector.pretrained_on,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": (
                _sha256_file(checkpoint_path) if checkpoint_path else None
            ),
            "classes": detector.class_names,
            "runtime_code_license": "AGPL-3.0",
            "runtime_license_reference": "https://github.com/ultralytics/ultralytics/blob/main/LICENSE",
        },
        "tracker": {
            "name": tracker.tracker_name,
            "version": tracker.tracker_version,
        },
        "results": {
            "tracked_detections": tracked_detections,
            "metrics": metrics,
        },
        "timings": {
            "processing_seconds": processing_seconds,
            "processing_fps": processing_fps,
            "stage_seconds": stage_seconds,
            "stage_average_milliseconds_per_frame": {
                name: (
                    seconds * 1000 / frames_processed
                    if frames_processed
                    else 0.0
                )
                for name, seconds in stage_seconds.items()
            },
            "measurement_scope": (
                "Tiempo de pared secuencial por etapa; detección incluye la "
                "sincronización que realice el adaptador del modelo."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "ultralytics": detector.model_version,
            "tracker": tracker.tracker_version,
        },
        "outputs": {
            "mot": str(mot_path),
            "mot_sha256": _sha256_file(mot_path),
            "detections_csv": str(detections_path),
            "detections_csv_sha256": _sha256_file(detections_path),
            "tracking_observations_csv": str(track_observations_path),
            "tracking_observations_csv_sha256": _sha256_file(
                track_observations_path
            ),
            "annotated_video": str(annotated_path),
            "annotated_video_sha256": _sha256_file(annotated_path),
        },
        "notes": [
            (
                "COCO person detections do not distinguish players from referees."
                if {"person", "sports ball"}.issubset(set(detector.class_names.values()))
                else "Role quality depends on the selected model's class annotations."
            ),
            "No performance or accuracy metrics are inferred without ground truth.",
        ],
    }
    if ground_truth_path is not None:
        gt_path = Path(ground_truth_path).resolve()
        metadata["evaluation"] = {
            "ground_truth_path": str(gt_path),
            "ground_truth_sha256": _sha256_file(gt_path),
            "metric_implementation": "TrackEval at pinned project dependency commit",
        }
    if checkpoint_cache is not None:
        metadata["detection_checkpoints"] = {
            "directory": str(checkpoint_cache.directory),
            "segment_frames": checkpoint_segment_frames,
            "cached_segments_reused": cached_segment_count,
            "cached_frames_replayed": cached_frame_count,
            "new_segments_written": uncached_segment_count,
            "resume_strategy": (
                "Se reutilizan detecciones YOLO en segmentos completos; ByteTrack "
                "y ReID se reconstruyen reproduciendo desde el fotograma inicial. "
                "No se serializa estado mutable ni se reanudan salidas parciales."
            ),
            "cache_key_sha256": checkpoint_cache.cache_key_sha256,
        }
    if identity_assigner is not None:
        if trajectory_path is None or tracking_stats_path is None:
            raise RuntimeError("Persistent identity output paths were not initialized")
        identity_stats = _identity_statistics(
            identity_records, frames_processed, identity_assigner
        )
        reidentification_runtime = getattr(
            identity_assigner, "runtime_metrics", None
        )
        if isinstance(reidentification_runtime, dict):
            metadata["timings"]["reidentification_detail"] = (
                reidentification_runtime
            )
        metadata["configuration"]["identity_association"] = (
            identity_assigner.configuration
        )
        metadata["results"]["persistent_identity_statistics"] = identity_stats
        _write_json_atomic(tracking_stats_path, identity_stats)
        metadata["outputs"]["trajectories_csv"] = str(trajectory_path)
        metadata["outputs"]["trajectories_csv_sha256"] = _sha256_file(
            trajectory_path
        )
        metadata["outputs"]["tracking_stats_json"] = str(tracking_stats_path)
        metadata["outputs"]["tracking_stats_json_sha256"] = _sha256_file(
            tracking_stats_path
        )
    _write_json_atomic(metadata_path, metadata)

    return VideoRunResult(
        run_id=run_id,
        frames_processed=frames_processed,
        detections_tracked=tracked_detections,
        processing_seconds=processing_seconds,
        processing_fps=processing_fps,
        mot_path=mot_path,
        annotated_video_path=annotated_path,
        metadata_path=metadata_path,
        metrics=metrics,
        trajectory_path=trajectory_path,
        tracking_stats_path=tracking_stats_path,
        detections_path=detections_path,
        track_observations_path=track_observations_path,
    )
