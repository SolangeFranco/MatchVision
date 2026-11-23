from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.evaluation.detection import (
    DetectionEvaluationError,
    evaluate_detection_predictions,
)
from matchvision.evaluation.identity_annotations import freeze_identity_annotations


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 64)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable")
    try:
        for _ in range(3):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()


def _observation(
    annotation_id: str,
    identity_id: str,
    entity_type: str,
    box: tuple[int, int, int, int],
) -> dict[str, object]:
    return {
        "annotation_id": annotation_id,
        "identity_id": identity_id,
        "entity_type": entity_type,
        "x1": box[0],
        "y1": box[1],
        "x2": box[2],
        "y2": box[3],
        "unknown": False,
        "ambiguous": False,
        "ignore": False,
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    video = tmp_path / "synthetic.avi"
    _write_video(video)
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "video": {
            "path": str(video),
            "sha256": video_hash,
            "frame_count": 3,
            "width": 64,
            "height": 64,
            "fps": 5.0,
        },
        "exhaustive_frames": [1, 2, 3],
        "frames": [
            {
                "frame": 1,
                "exhaustive": True,
                "observations": [
                    _observation("p1-f1", "player-1", "player", (1, 1, 11, 11)),
                    _observation("r1-f1", "referee-1", "referee", (20, 20, 30, 30)),
                ],
            },
            {
                "frame": 2,
                "exhaustive": True,
                "observations": [
                    _observation("p1-f2", "player-1", "player", (1, 1, 11, 11)),
                    _observation("b1-f2", "ball-1", "ball", (40, 40, 45, 45)),
                ],
            },
            {"frame": 3, "exhaustive": True, "observations": []},
        ],
    }
    annotations = tmp_path / "ground-truth.json"
    annotations.write_text(
        json.dumps(freeze_identity_annotations(payload)),
        encoding="utf-8",
    )
    detections = tmp_path / "detections.csv"
    with detections.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
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
        writer.writeheader()
        writer.writerows(
            [
                {
                    "frame": 1,
                    "class_id": 0,
                    "label": "person",
                    "confidence": 0.9,
                    "x1": 4,
                    "y1": 1,
                    "x2": 14,
                    "y2": 11,
                },
                {
                    "frame": 1,
                    "class_id": 32,
                    "label": "sports ball",
                    "confidence": 0.8,
                    "x1": 50,
                    "y1": 50,
                    "x2": 55,
                    "y2": 55,
                },
                {
                    "frame": 2,
                    "class_id": 32,
                    "label": "sports ball",
                    "confidence": 0.8,
                    "x1": 40,
                    "y1": 40,
                    "x2": 45,
                    "y2": 45,
                },
                {
                    "frame": 3,
                    "class_id": 0,
                    "label": "person",
                    "confidence": 0.7,
                    "x1": 1,
                    "y1": 40,
                    "x2": 10,
                    "y2": 55,
                },
            ]
        )
    return video, annotations, detections


def test_detection_metrics_are_class_aware_and_use_iou_threshold(
    tmp_path: Path,
) -> None:
    video, annotations, detections = _fixture(tmp_path)

    report = evaluate_detection_predictions(
        annotations,
        detections,
        video_path=video,
        class_map={"person": "player", "sports ball": "ball"},
        iou_threshold=0.5,
    )

    assert report["video_sha256_verified"] is True
    assert report["overall"]["true_positives"] == 2
    assert report["overall"]["false_positives"] == 2
    assert report["overall"]["false_negatives"] == 2
    assert report["overall"]["precision"] == pytest.approx(0.5)
    assert report["overall"]["recall"] == pytest.approx(0.5)
    assert report["per_class"]["player"]["true_positives"] == 1
    assert report["per_class"]["player"]["false_positives"] == 1
    assert report["per_class"]["player"]["false_negatives"] == 1
    assert report["per_class"]["referee"]["false_negatives"] == 1
    assert report["per_class"]["ball"]["true_positives"] == 1

    strict = evaluate_detection_predictions(
        annotations,
        detections,
        video_path=video,
        class_map={"person": "player", "sports ball": "ball"},
        iou_threshold=0.6,
    )
    assert strict["per_class"]["player"]["true_positives"] == 0
    assert strict["per_class"]["player"]["false_negatives"] == 2
    assert strict["iou_threshold"] == 0.6


def test_detection_requires_explicit_mapping_for_coco_person_class(
    tmp_path: Path,
) -> None:
    video, annotations, detections = _fixture(tmp_path)

    with pytest.raises(DetectionEvaluationError, match="class_map explícito"):
        evaluate_detection_predictions(
            annotations, detections, video_path=video
        )


def test_detection_evaluation_rejects_non_frozen_ground_truth(
    tmp_path: Path,
) -> None:
    video, annotations, detections = _fixture(tmp_path)
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    del payload["dataset"]
    annotations.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="congelado"):
        evaluate_detection_predictions(
            annotations,
            detections,
            video_path=video,
            class_map={"person": "player", "sports ball": "ball"},
        )
