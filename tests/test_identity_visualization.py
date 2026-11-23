from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.evaluation.identity_visualization import (
    create_identity_error_visualizations,
)


def test_renders_identity_mismatch_frame_and_manifest(tmp_path: Path) -> None:
    video_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (100, 100),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    annotations = {
        "schema_version": 1,
        "video": {
            "path": str(video_path),
            "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            "frame_count": 2,
            "width": 100,
            "height": 100,
            "fps": 25.0,
        },
        "exhaustive_frames": [1, 2],
        "frames": [
            {
                "frame": frame,
                "exhaustive": True,
                "observations": [
                    {
                        "annotation_id": f"person-{frame}",
                        "identity_id": "player-1",
                        "x1": 10,
                        "y1": 10,
                        "x2": 30,
                        "y2": 50,
                        "unknown": False,
                        "ambiguous": False,
                        "ignore": False,
                    }
                ],
            }
            for frame in (1, 2)
        ],
    }
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    prediction_path = tmp_path / "prediction.txt"
    prediction_path.write_text(
        "1,1,10,10,20,40,1,-1,-1,-1\n"
        "2,2,10,10,20,40,1,-1,-1,-1\n",
        encoding="utf-8",
    )

    report = create_identity_error_visualizations(
        annotation_path,
        prediction_path,
        video_path,
        tmp_path / "visualizations",
        run_name="reid",
    )

    assert report["error_frame_count"] == 1
    assert report["visualized_frame_count"] == 1
    assert report["error_counts"] == {
        "false_negatives": 0,
        "false_positives": 0,
        "identity_mismatches": 1,
    }
    assert Path(report["images"][0]).name == "frame-000002.jpg"
    assert cv2.imread(report["images"][0]) is not None
    assert Path(report["manifest_path"]).is_file()


def test_error_visualization_rejects_mismatched_video_hash(tmp_path: Path) -> None:
    video_path = tmp_path / "source.avi"
    video_path.write_bytes(b"original")
    video_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video": {
                    "path": str(video_path),
                    "sha256": video_hash,
                    "frame_count": 1,
                    "width": 10,
                    "height": 10,
                    "fps": 25.0,
                },
                "exhaustive_frames": [1],
                "frames": [
                    {
                        "frame": 1,
                        "exhaustive": True,
                        "observations": [
                            {
                                "annotation_id": "ann-1",
                                "identity_id": "player-1",
                                "x1": 1,
                                "y1": 1,
                                "x2": 5,
                                "y2": 5,
                                "unknown": False,
                                "ambiguous": False,
                                "ignore": False,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    video_path.write_bytes(b"changed")
    (tmp_path / "prediction.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        create_identity_error_visualizations(
            annotation_path,
            tmp_path / "prediction.txt",
            video_path,
            tmp_path / "visualizations",
            run_name="baseline",
        )
