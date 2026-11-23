"""Frame visualizations for false matches, misses, and identity mismatches."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from matchvision.annotations.mot import MOTAnnotation
from matchvision.evaluation.identity_annotations import (
    evaluation_frames,
    load_identity_annotations,
)
from matchvision.evaluation.mot import load_mot_file


def create_identity_error_visualizations(
    annotation_path: str | Path,
    prediction_path: str | Path,
    video_path: str | Path,
    output_directory: str | Path,
    *,
    run_name: str,
    iou_threshold: float = 0.5,
    maximum_examples: int = 100,
) -> dict[str, Any]:
    """Render up to N exhaustive frames with FP, FN, or identity mismatch boxes."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name may contain only letters, digits, '_' or '-'")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if maximum_examples < 1:
        raise ValueError("maximum_examples must be positive")

    annotation_file = Path(annotation_path).expanduser().resolve()
    prediction_file = Path(prediction_path).expanduser().resolve()
    source_video = Path(video_path).expanduser().resolve()
    for path in (annotation_file, prediction_file, source_video):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
    payload = load_identity_annotations(annotation_file)
    if _sha256_file(source_video) != payload["video"]["sha256"]:
        raise ValueError("The visualization video does not match the annotation SHA-256.")
    frames = evaluation_frames(payload)
    if not frames:
        raise ValueError("No eligible exhaustive frames are available for visualization.")

    frame_count = int(payload["video"]["frame_count"])
    predictions = load_mot_file(prediction_file, frame_count)
    frame_set = set(frames)
    gt_identity = {
        identity: index + 1
        for index, identity in enumerate(
            sorted(
                {
                    observation["identity_id"]
                    for record in payload["frames"]
                    if record["frame"] in frame_set
                    for observation in record["observations"]
                }
            )
        )
    }
    records = {record["frame"]: record for record in payload["frames"]}
    gt_by_frame: dict[int, list[MOTAnnotation]] = defaultdict(list)
    predicted_by_frame: dict[int, list[MOTAnnotation]] = defaultdict(list)
    for frame_number in frames:
        for observation in records[frame_number]["observations"]:
            identity_id = observation["identity_id"]
            x1, y1, x2, y2 = (
                float(observation["x1"]),
                float(observation["y1"]),
                float(observation["x2"]),
                float(observation["y2"]),
            )
            gt_by_frame[frame_number].append(
                MOTAnnotation(
                    frame=frame_number,
                    track_id=gt_identity[identity_id],
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    confidence=1.0,
                )
            )
    for prediction in predictions:
        if prediction.frame in frame_set:
            predicted_by_frame[prediction.frame].append(prediction)

    error_frames: list[dict[str, Any]] = []
    for frame_number in frames:
        gt_rows = gt_by_frame[frame_number]
        predicted_rows = predicted_by_frame[frame_number]
        similarities = np.zeros((len(gt_rows), len(predicted_rows)), dtype=np.float64)
        for row, gt in enumerate(gt_rows):
            for column, predicted in enumerate(predicted_rows):
                similarities[row, column] = _iou(gt, predicted)
        if similarities.size:
            gt_indices, prediction_indices = linear_sum_assignment(1 - similarities)
        else:
            gt_indices, prediction_indices = np.asarray([], dtype=int), np.asarray([], dtype=int)
        matched: list[tuple[int, int, bool]] = []
        matched_gt: set[int] = set()
        matched_predictions: set[int] = set()
        for gt_index, prediction_index in zip(
            gt_indices, prediction_indices, strict=True
        ):
            if similarities[gt_index, prediction_index] < iou_threshold:
                continue
            matched_gt.add(int(gt_index))
            matched_predictions.add(int(prediction_index))
            identity_correct = (
                predicted_rows[prediction_index].track_id
                == gt_rows[gt_index].track_id
            )
            matched.append((int(gt_index), int(prediction_index), identity_correct))
        unmatched_gt = sorted(set(range(len(gt_rows))) - matched_gt)
        unmatched_predictions = sorted(
            set(range(len(predicted_rows))) - matched_predictions
        )
        identity_mismatches = [
            (gt_index, prediction_index)
            for gt_index, prediction_index, identity_correct in matched
            if not identity_correct
        ]
        if unmatched_gt or unmatched_predictions or identity_mismatches:
            error_frames.append(
                {
                    "frame": frame_number,
                    "false_negatives": unmatched_gt,
                    "false_positives": unmatched_predictions,
                    "identity_mismatches": identity_mismatches,
                    "ground_truth": gt_rows,
                    "predictions": predicted_rows,
                    "matches": matched,
                }
            )

    run_directory = Path(output_directory).expanduser().resolve() / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open visualization video: {source_video}")
    image_files: list[str] = []
    try:
        for error in error_frames[:maximum_examples]:
            frame_number = error["frame"]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            success, image = capture.read()
            if not success:
                raise ValueError(
                    f"Could not decode frame {frame_number} from {source_video}"
                )
            if (
                image.shape[1] != int(payload["video"]["width"])
                or image.shape[0] != int(payload["video"]["height"])
            ):
                raise ValueError(
                    f"Decoded video dimensions {image.shape[1]}x{image.shape[0]} "
                    "do not match annotation metadata."
                )
            _draw_error_frame(image, error)
            if image.shape[1] > 1920:
                height = round(image.shape[0] * 1920 / image.shape[1])
                image = cv2.resize(image, (1920, height), interpolation=cv2.INTER_AREA)
            image_path = run_directory / f"frame-{frame_number:06d}.jpg"
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"Could not write error visualization: {image_path}")
            image_files.append(str(image_path))
    finally:
        capture.release()

    report = {
        "schema_version": 1,
        "run_name": run_name,
        "annotation_path": str(annotation_file),
        "annotation_sha256": _sha256_file(annotation_file),
        "prediction_path": str(prediction_file),
        "prediction_sha256": _sha256_file(prediction_file),
        "video_path": str(source_video),
        "video_sha256": _sha256_file(source_video),
        "iou_threshold": iou_threshold,
        "error_frame_count": len(error_frames),
        "visualized_frame_count": len(image_files),
        "error_counts": {
            "false_negatives": sum(
                len(frame["false_negatives"]) for frame in error_frames
            ),
            "false_positives": sum(
                len(frame["false_positives"]) for frame in error_frames
            ),
            "identity_mismatches": sum(
                len(frame["identity_mismatches"]) for frame in error_frames
            ),
        },
        "images": image_files,
        "limitations": (
            "Per-frame IoU/Hungarian matching is used only to illustrate errors; "
            "this image manifest is not a separate metric."
        ),
    }
    manifest_path = run_directory / "visualization-manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["manifest_path"] = str(manifest_path)
    return report


def _iou(left: MOTAnnotation, right: MOTAnnotation) -> float:
    intersection_width = max(
        0.0,
        min(left.x + left.width, right.x + right.width) - max(left.x, right.x),
    )
    intersection_height = max(
        0.0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _draw_error_frame(image: np.ndarray, error: dict[str, Any]) -> None:
    gt_rows = error["ground_truth"]
    predicted_rows = error["predictions"]
    mismatches = {
        prediction_index
        for _, prediction_index in error["identity_mismatches"]
    }
    unmatched_gt = set(error["false_negatives"])
    unmatched_predictions = set(error["false_positives"])
    correct_predictions = {
        prediction_index
        for _, prediction_index, correct in error["matches"]
        if correct
    }
    for index, gt in enumerate(gt_rows):
        color = (255, 170, 0) if index in unmatched_gt else (50, 220, 80)
        if any(gt_index == index for gt_index, _ in error["identity_mismatches"]):
            color = (0, 165, 255)
        _draw_box(
            image,
            gt,
            color,
            f"GT identity {gt.track_id}",
        )
    for index, predicted in enumerate(predicted_rows):
        if index in unmatched_predictions:
            color, label = (0, 0, 255), f"FP track {predicted.track_id}"
        elif index in mismatches:
            color, label = (0, 165, 255), f"ID mismatch {predicted.track_id}"
        elif index in correct_predictions:
            color, label = (50, 220, 80), f"matched {predicted.track_id}"
        else:
            continue
        _draw_box(image, predicted, color, label)
    cv2.putText(
        image,
        f"Frame {error['frame']} | GT green / mismatch orange / FN blue / FP red",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_box(
    image: np.ndarray, annotation: MOTAnnotation, color: tuple[int, int, int], label: str
) -> None:
    x1, y1 = int(round(annotation.x)), int(round(annotation.y))
    x2 = int(round(annotation.x + annotation.width))
    y2 = int(round(annotation.y + annotation.height))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
