from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.evaluation.identity_annotations import freeze_identity_annotations
from matchvision.evaluation.reid_retrieval import (
    evaluate_human_appearance_retrieval,
    main,
)


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (96, 72)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable")
    try:
        for _ in range(3):
            frame = np.zeros((72, 96, 3), dtype=np.uint8)
            frame[10:62, 10:28] = (0, 0, 255)
            frame[10:62, 60:78] = (255, 0, 0)
            writer.write(frame)
    finally:
        writer.release()


def _observation(frame: int, identity: str, x1: int) -> dict[str, object]:
    return {
        "annotation_id": f"{identity}-{frame}",
        "identity_id": identity,
        "entity_type": "player",
        "x1": x1,
        "y1": 10,
        "x2": x1 + 18,
        "y2": 62,
        "unknown": False,
        "ambiguous": False,
        "ignore": False,
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "synthetic.avi"
    _write_video(video)
    video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "video": {
            "path": str(video),
            "sha256": video_sha,
            "frame_count": 3,
            "width": 96,
            "height": 72,
            "fps": 5.0,
        },
        "exhaustive_frames": [1, 2, 3],
        "frames": [
            {
                "frame": frame,
                "exhaustive": True,
                "events": (
                    [
                        {
                            "event_id": "crossing-1",
                            "identity_id": "player-1",
                            "event_type": "crossing",
                        }
                    ]
                    if frame == 2
                    else []
                ),
                "observations": [
                    _observation(frame, "player-1", 10),
                    _observation(frame, "player-2", 60),
                ],
            }
            for frame in (1, 2, 3)
        ],
    }
    annotations = tmp_path / "ground-truth.json"
    annotations.write_text(
        json.dumps(freeze_identity_annotations(payload)),
        encoding="utf-8",
    )
    return video, annotations


def test_retrieval_uses_event_specific_queries_and_unordered_global_pairs(
    tmp_path: Path,
) -> None:
    video, annotations = _inputs(tmp_path)

    report = evaluate_human_appearance_retrieval(
        annotations,
        video,
        methods=("hsv_hog",),
        similarity_threshold=0.99,
        sample_interval=1,
    )

    metrics = report["methods"]["hsv_hog"]["global_retrieval"]
    assert metrics["gallery_pairs"] == 12
    assert metrics["positive_identity_pairs"] == 6
    assert metrics["confusion_matrix"]["true_positive"] == 6
    assert metrics["confusion_matrix"]["false_positive"] == 0
    assert metrics["rank1"] == pytest.approx(1.0)
    assert metrics["mean_average_precision"] == pytest.approx(1.0)

    crossing = report["methods"]["hsv_hog"]["event_scenarios"][
        "crossing_recovery"
    ]
    assert crossing["event_type"] == "crossing"
    assert crossing["queries"] == 1
    assert crossing["gallery_pairs"] == 2
    assert crossing["positive_identity_pairs"] == 1
    assert crossing["rank1"] == pytest.approx(1.0)
    assert report["evaluation_id"]
    assert report["created_at"]
    assert report["hardware"]["logical_cpu_count"]
    assert report["timings"]["sampling_seconds"] >= 0


def test_embedding_methods_are_explicitly_unavailable_without_weights(
    tmp_path: Path,
) -> None:
    video, annotations = _inputs(tmp_path)

    report = evaluate_human_appearance_retrieval(
        annotations,
        video,
        methods=("hsv_hog", "embedding", "hybrid"),
        sample_interval=1,
    )

    assert report["methods"]["hsv_hog"]["available"] is True
    for method in ("embedding", "hybrid"):
        assert report["methods"][method]["available"] is False
        assert "pesos OSNet verificados" in report["methods"][method]["reason"]


def test_retrieval_sampling_is_bounded_balanced_and_event_aware(
    tmp_path: Path,
) -> None:
    video, annotations = _inputs(tmp_path)

    report = evaluate_human_appearance_retrieval(
        annotations,
        video,
        methods=("hsv_hog",),
        sample_interval=1,
        maximum_observations=4,
    )

    assert report["eligible_appearance_observations"] == 6
    assert report["sampled_observations"] == 4
    assert report["observations_excluded_by_maximum"] == 2
    assert 2 in report["sampled_frames"]
    assert report["methods"]["hsv_hog"]["event_scenarios"][
        "crossing_recovery"
    ]["queries"] == 1
    repeated = evaluate_human_appearance_retrieval(
        annotations,
        video,
        methods=("hsv_hog",),
        sample_interval=1,
        maximum_observations=4,
    )
    assert repeated["sampled_frames"] == report["sampled_frames"]


def test_retrieval_cli_writes_a_reproducible_json_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video, annotations = _inputs(tmp_path)
    output = tmp_path / "retrieval-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matchvision-evaluate-reid-retrieval",
            "--annotations",
            str(annotations),
            "--video",
            str(video),
            "--methods",
            "hsv_hog",
            "--sample-interval",
            "1",
            "--maximum-observations",
            "100",
            "--output",
            str(output),
        ],
    )

    main()

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["ground_truth_content_sha256"]
    assert saved["methods"]["hsv_hog"]["global_retrieval"]["rank1"] == pytest.approx(
        1.0
    )
    with pytest.raises(SystemExit):
        main()
