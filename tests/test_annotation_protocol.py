from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.evaluation.annotation_protocol import (
    create_annotation_sampling_plan,
    quality_control_annotations,
    validate_sampling_plan,
)
from matchvision.evaluation.identity_annotations import (
    IdentityAnnotationError,
    freeze_identity_annotations,
)


def _write_video(path: Path, *, frames: int = 30) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable")
    try:
        for index in range(frames):
            frame = np.full((48, 64, 3), index * 5 % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _payload(video: Path, plan: dict[str, object]) -> dict[str, object]:
    frame_records = []
    plan_frames = {item["frame"] for item in plan["planned_frames"]}
    for frame in sorted(plan_frames):
        frame_records.append(
            {
                "frame": frame,
                "exhaustive": True,
                "review_status": "reviewed",
                "review_note": "Revisión humana completa.",
                "observations": [
                    {
                        "annotation_id": f"box-{frame}",
                        "identity_id": "player-1",
                        "entity_type": "player",
                        "x1": 10,
                        "y1": 10,
                        "x2": 30,
                        "y2": 40,
                        "unknown": False,
                        "ambiguous": False,
                        "ignore": False,
                        "identity_status": "confirmed",
                    }
                ],
                "events": [],
            }
        )
    return {
        "schema_version": 1,
        "record_type": "human_identity_ground_truth",
        "video": {
            "path": str(video),
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "frame_count": plan["video"]["frame_count"],
            "width": 64,
            "height": 48,
            "fps": 10.0,
        },
        "sampling_plan": plan,
        "annotation_history": [],
        "exhaustive_frames": sorted(plan_frames),
        "frames": frame_records,
    }


def test_sampling_plan_is_deterministic_video_bound_and_human_only(
    tmp_path: Path,
) -> None:
    video = tmp_path / "synthetic.avi"
    _write_video(video, frames=30)

    first = create_annotation_sampling_plan(
        video,
        interval_frames=10,
        dense_window_seconds=1.0,
        dense_window_count=3,
        hard_cases=[
            {
                "frame": 15,
                "case_type": "crossing",
                "note": "Cruce observado manualmente.",
            }
        ],
    )
    validate_sampling_plan(first)

    assert first["video"]["frame_count"] == 30
    assert first["video"]["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert first["plan_id"].startswith("mv-plan-")
    assert first["protocol"]["annotations_are_human_ground_truth"] is True
    assert first["protocol"]["automatic_predictions_are_labels"] is False
    assert first["hard_cases"] == [
        {"frame": 15, "case_type": "crossing", "note": "Cruce observado manualmente."}
    ]
    assert len(first["dense_windows"]) == 3
    assert 15 in {item["frame"] for item in first["planned_frames"]}
    assert first["planned_frames"] == sorted(
        first["planned_frames"], key=lambda item: item["frame"]
    )


def test_sampling_plan_rejects_tampering(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.avi"
    _write_video(video, frames=30)
    plan = create_annotation_sampling_plan(video, interval_frames=10)

    tampered = json.loads(json.dumps(plan))
    tampered["hard_cases"].append(
        {"frame": 20, "case_type": "complete_occlusion", "note": "Editado."}
    )

    with pytest.raises(IdentityAnnotationError, match="SHA-256"):
        validate_sampling_plan(tampered)


def test_quality_control_rejects_missing_plan_frames_and_can_freeze_complete(
    tmp_path: Path,
) -> None:
    video = tmp_path / "synthetic.avi"
    _write_video(video, frames=30)
    plan = create_annotation_sampling_plan(
        video,
        interval_frames=10,
        dense_window_seconds=1.0,
        dense_window_count=3,
    )
    payload = _payload(video, plan)

    report = quality_control_annotations(payload)
    assert report["ready"] is True
    assert report["sampling_plan_id"] == plan["plan_id"]

    frozen = freeze_identity_annotations(
        payload, frozen_at="2026-11-23T12:00:00+00:00"
    )
    assert frozen["dataset"]["freeze"]["status"] == "frozen"

    incomplete = json.loads(json.dumps(payload))
    incomplete["frames"].pop()
    incomplete["exhaustive_frames"].pop()
    incomplete_report = quality_control_annotations(incomplete)
    assert incomplete_report["ready"] is False
    assert "PLANNED_FRAME_MISSING" in {
        issue["code"] for issue in incomplete_report["issues"]
    }
    with pytest.raises(IdentityAnnotationError, match="sampling protocol"):
        freeze_identity_annotations(incomplete)


def test_quality_control_rejects_not_applicable_without_review_note(
    tmp_path: Path,
) -> None:
    video = tmp_path / "synthetic.avi"
    _write_video(video, frames=30)
    plan = create_annotation_sampling_plan(video, interval_frames=10)
    payload = _payload(video, plan)
    payload["frames"][0]["review_status"] = "not_applicable"
    payload["frames"][0]["review_note"] = ""
    payload["frames"][0]["observations"] = []
    payload["exhaustive_frames"].remove(payload["frames"][0]["frame"])
    payload["frames"][0]["exhaustive"] = False

    report = quality_control_annotations(payload)
    assert report["ready"] is False
    assert "ANNOTATION_SCHEMA_INVALID" in {
        issue["code"] for issue in report["issues"]
    }
