from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from matchvision.tracking.review_cases import select_review_cases


def _write_trajectories(path: Path) -> Path:
    rows = [
        {
            "frame": 1,
            "temporal_track_id": 1,
            "persistent_id": 1,
            "class_id": 0,
            "label": "person",
            "x1": 10,
            "y1": 10,
            "x2": 20,
            "y2": 40,
            "confidence": 0.9,
            "identity_state": "new",
            "association_confidence": "",
        },
        {
            "frame": 2,
            "temporal_track_id": 1,
            "persistent_id": 1,
            "class_id": 0,
            "label": "person",
            "x1": 10,
            "y1": 10,
            "x2": 20,
            "y2": 40,
            "confidence": 0.9,
            "identity_state": "tracked",
            "association_confidence": "",
        },
        {
            "frame": 4,
            "temporal_track_id": 1,
            "persistent_id": 1,
            "class_id": 0,
            "label": "person",
            "x1": 10,
            "y1": 10,
            "x2": 20,
            "y2": 40,
            "confidence": 0.9,
            "identity_state": "tracked",
            "association_confidence": "",
        },
        {
            "frame": 6,
            "temporal_track_id": 2,
            "persistent_id": 1,
            "class_id": 0,
            "label": "person",
            "x1": 11,
            "y1": 10,
            "x2": 21,
            "y2": 40,
            "confidence": 0.9,
            "identity_state": "reidentified",
            "association_confidence": 0.6,
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_selects_reid_gaps_id_restarts_and_close_crossings(
    tmp_path: Path,
) -> None:
    mot = tmp_path / "baseline.txt"
    mot.write_text(
        "2,1,10,10,10,30,1,-1,-1,-1\n"
        "2,2,11,10,10,30,1,-1,-1,-1\n",
        encoding="utf-8",
    )
    trajectories = _write_trajectories(tmp_path / "trajectories.csv")
    video = tmp_path / "video.webm"
    video.write_bytes(b"fixture video")
    output = tmp_path / "review-cases.json"

    result = select_review_cases(
        mot,
        trajectories,
        video_path=video,
        frame_count=10,
        output_path=output,
        maximum_cases=20,
        radius_frames=2,
        minimum_crossing_distance_heights=0.65,
    )

    event_types = {case["event_type"] for case in result["cases"]}
    assert {
        "reid_association",
        "within_track_disappearance",
        "nearby_temporal_id_restart",
        "close_player_crossing_or_occlusion",
    } <= event_types
    assert all(1 <= case["start_frame"] <= case["event_frame"] for case in result["cases"])
    assert all(case["event_frame"] <= case["end_frame"] <= 10 for case in result["cases"])
    assert json.loads(output.read_text(encoding="utf-8"))["cases"] == result["cases"]
    assert result["source_sha256"]["video"]
    assert result["source_sha256"]["baseline_mot"]
    assert "not verified" in result["limitations"]


def test_rejects_trajectory_csv_with_missing_required_columns(tmp_path: Path) -> None:
    mot = tmp_path / "baseline.txt"
    mot.write_text("", encoding="utf-8")
    trajectories = tmp_path / "trajectories.csv"
    trajectories.write_text("frame,id\n1,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        select_review_cases(
            mot,
            trajectories,
            frame_count=1,
            output_path=tmp_path / "cases.json",
        )


def test_rejects_frame_count_or_case_limits_outside_valid_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        select_review_cases(
            tmp_path / "missing.txt",
            tmp_path / "missing.csv",
            frame_count=0,
            output_path=tmp_path / "cases.json",
        )
