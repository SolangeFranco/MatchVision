"""MOT annotation loading, integrity checks, and TrackEval metric adapters."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from matchvision.annotations.mot import MOTAnnotation, read_mot_annotations


class MOTDataError(ValueError):
    """Raised when MOT annotations do not match sequence constraints."""


def load_mot_file(path: str | Path, frame_count: int) -> list[MOTAnnotation]:
    """Read MOT CSV and verify frame range and per-frame identity uniqueness."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    annotations = read_mot_annotations(path)
    occupied: set[tuple[int, int]] = set()
    for annotation in annotations:
        if annotation.frame > frame_count:
            raise MOTDataError(
                f"Frame {annotation.frame} exceeds declared frame_count {frame_count}"
            )
        key = (annotation.frame, annotation.track_id)
        if key in occupied:
            raise MOTDataError(
                f"Duplicate track ID {annotation.track_id} in frame {annotation.frame}"
            )
        occupied.add(key)
    return annotations


def _box_iou(left: MOTAnnotation, right: MOTAnnotation) -> float:
    left_x2 = left.x + left.width
    left_y2 = left.y + left.height
    right_x2 = right.x + right.width
    right_y2 = right.y + right.height
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left.x, right.x))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left.y, right.y))
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _trackeval_data(
    ground_truth: list[MOTAnnotation],
    predictions: list[MOTAnnotation],
    frame_count: int,
) -> dict[str, Any]:
    gt_identity = {
        track_id: index
        for index, track_id in enumerate(sorted({item.track_id for item in ground_truth}))
    }
    pred_identity = {
        track_id: index
        for index, track_id in enumerate(sorted({item.track_id for item in predictions}))
    }
    gt_by_frame: dict[int, list[MOTAnnotation]] = defaultdict(list)
    pred_by_frame: dict[int, list[MOTAnnotation]] = defaultdict(list)
    for item in ground_truth:
        gt_by_frame[item.frame].append(item)
    for item in predictions:
        pred_by_frame[item.frame].append(item)

    gt_ids_by_frame: list[np.ndarray] = []
    pred_ids_by_frame: list[np.ndarray] = []
    similarity_by_frame: list[np.ndarray] = []
    for frame_index in range(1, frame_count + 1):
        gt_boxes = gt_by_frame[frame_index]
        pred_boxes = pred_by_frame[frame_index]
        gt_ids_by_frame.append(
            np.asarray([gt_identity[item.track_id] for item in gt_boxes], dtype=np.int64)
        )
        pred_ids_by_frame.append(
            np.asarray([pred_identity[item.track_id] for item in pred_boxes], dtype=np.int64)
        )
        similarities = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float64)
        for gt_index, gt_box in enumerate(gt_boxes):
            for pred_index, pred_box in enumerate(pred_boxes):
                similarities[gt_index, pred_index] = _box_iou(gt_box, pred_box)
        similarity_by_frame.append(similarities)

    return {
        "num_timesteps": frame_count,
        "num_tracker_dets": len(predictions),
        "num_gt_dets": len(ground_truth),
        "num_tracker_ids": len(pred_identity),
        "num_gt_ids": len(gt_identity),
        "tracker_ids": pred_ids_by_frame,
        "gt_ids": gt_ids_by_frame,
        "similarity_scores": similarity_by_frame,
    }


def evaluate_mot_sequence(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    frame_count: int,
) -> dict[str, float | int]:
    """Evaluate one class-agnostic sequence with the official TrackEval metrics.

    Uses TrackEval's HOTA, CLEAR/MOTA, and Identity/IDF1 metric classes on
    frame-indexed IoU similarities. No class preprocessing is applied.
    """
    ground_truth = load_mot_file(ground_truth_path, frame_count)
    predictions = load_mot_file(prediction_path, frame_count)
    data = _trackeval_data(ground_truth, predictions, frame_count)

    # TrackEval versions pinned for reproducibility use NumPy aliases removed in
    # NumPy 1.24. Add only the legacy aliases it expects when they are absent.
    for name, value in (("float", float), ("int", int)):
        if not hasattr(np, name):
            setattr(np, name, value)

    try:
        from trackeval.metrics import CLEAR, HOTA, Identity
    except ImportError as error:
        raise RuntimeError(
            "TrackEval is required; install the project dependencies with "
            "`python -m pip install -e .`."
        ) from error

    hota_result = HOTA().eval_sequence(data)
    mota_result = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    identity_result = Identity(
        {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    ).eval_sequence(data)
    return {
        "HOTA": float(np.mean(hota_result["HOTA"])),
        "MOTA": float(mota_result["MOTA"]),
        "IDF1": float(identity_result["IDF1"]),
        "ID_Precision": float(identity_result["IDP"]),
        "ID_Recall": float(identity_result["IDR"]),
        "ID_switches": int(mota_result["IDSW"]),
        "Fragmentation": int(mota_result["Frag"]),
        "ID_true_positives": int(identity_result["IDTP"]),
        "ID_false_positives": int(identity_result["IDFP"]),
        "ID_false_negatives": int(identity_result["IDFN"]),
        "ground_truth_detections": len(ground_truth),
        "predicted_detections": len(predictions),
    }
