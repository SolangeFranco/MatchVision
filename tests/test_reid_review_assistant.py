from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.reidentification.review_assistant import (
    _enrich_proposal_relationships,
    generate_review_proposals,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_video(path: Path, *, scene_cut_at: int | None = None) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (96, 96),
    )
    if not writer.isOpened():
        pytest.skip("El códec MJPG no está disponible en este entorno.")
    for frame_number in range(1, 6):
        background = 235 if scene_cut_at is not None and frame_number >= scene_cut_at else 24
        frame = np.full((96, 96, 3), background, dtype=np.uint8)
        cv2.rectangle(frame, (30, 18), (48, 77), (25, 50, 220), -1)
        cv2.rectangle(frame, (34, 28), (44, 49), (210, 220, 230), -1)
        cv2.line(frame, (31, 55), (47, 70), (10, 15, 20), 3)
        writer.write(frame)
    writer.release()


def _write_trajectory_csv(path: Path) -> None:
    fields = [
        "frame", "temporal_track_id", "persistent_id", "class_id", "label",
        "x1", "y1", "x2", "y2", "confidence", "identity_state",
    ]
    rows = []
    for frame in (1, 2):
        rows.append(
            [frame, 1, 1, 0, "person", 30, 18, 49, 78, 0.9, "tracked"]
        )
    for frame in (4, 5):
        rows.append(
            [frame, 2, 1, 0, "person", 30, 18, 49, 78, 0.9, "reidentified"]
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def test_generates_visual_reid_proposals_without_annotations(tmp_path: Path) -> None:
    video = tmp_path / "review.avi"
    manifest_path = tmp_path / "manifest.json"
    trajectories_path = tmp_path / "trajectories.csv"
    cases_path = tmp_path / "cases.json"
    output = tmp_path / "proposals.json"
    evidence_dir = tmp_path / "evidence"
    _write_video(video)
    _write_trajectory_csv(trajectories_path)
    source_hash = "a" * 64
    manifest = {
        "schema_version": 1,
        "source_sha256": source_hash,
        "output_sha256": _sha256(video),
        "frames_written": 5,
        "source_width": 96,
        "source_height": 96,
        "output_width": 96,
        "output_height": 96,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cases = {
        "schema_version": 1,
        "video_frame_count": 5,
        "video": {"sha256": source_hash},
        "source_sha256": {
            "video": source_hash,
            "persistent_trajectories": _sha256(trajectories_path),
        },
        "cases": [
            {
                "fragment_id": "review-0001",
                "event_type": "nearby_temporal_id_restart",
                "priority": 4,
                "start_frame": 1,
                "event_frame": 3,
                "end_frame": 5,
                "temporal_track_id": 2,
                "candidate_previous_track_id": 1,
            }
        ],
    }
    cases_path.write_text(json.dumps(cases), encoding="utf-8")

    result = generate_review_proposals(
        video,
        manifest_path,
        trajectories_path,
        cases_path,
        output,
        evidence_dir,
    )

    proposal = result["cases"][0]["candidate_associations"][0]
    assert result["schema_version"] == 2
    assert proposal["track_ids"] == [1, 2]
    assert proposal["relation"] in {"existing_reid_link", "candidate_same_player"}
    assert proposal["score_is_probability"] is False
    assert proposal["proposal_id"]
    assert proposal["candidate_group_id"]
    assert result["cases"][0]["no_automatic_identity"] is True
    assert "identity_id" not in proposal
    assert "exhaustive_frames" not in result
    assert result["timings_seconds"]["serialization_included_in_elapsed"] is False
    assert all(
        result["timings_seconds"][key] >= 0
        for key in (
            "input_validation_and_sampling",
            "video_evidence_and_feature_extraction",
            "proposal_scoring_and_grouping",
            "elapsed_before_json_serialization",
        )
    )
    assert len(result["cases"][0]["frame_evidence"]) == 3
    assert all(
        (evidence_dir / Path(item["path"]).name).is_file()
        for item in result["cases"][0]["frame_evidence"]
    )
    assert len(proposal["crop_comparison"]) == 2
    assert {
        sample["frame"]
        for track in result["cases"][0]["track_appearances"]
        for sample in track["samples"]
    }
    assert all(
        (evidence_dir / Path(item["path"]).name).is_file()
        for item in proposal["crop_comparison"]
    )
    assert output.is_file()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["timings_seconds"] == result["timings_seconds"]
    assert evidence_dir.is_dir()
    assert result["proposal_summary"]["candidate_group_count"] == 1

    with pytest.raises(FileExistsError, match="no se sobrescribió"):
        generate_review_proposals(
            video,
            manifest_path,
            trajectories_path,
            cases_path,
            output,
            evidence_dir,
        )


def test_hybrid_review_proposals_record_scores_for_each_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matchvision.reidentification.review_assistant as assistant

    class FakeExtractor:
        configuration = {"model": "fake-osnet", "device": "cpu"}
        inference_seconds = 0.25
        images_processed = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def encode(self, crops: list[np.ndarray]) -> list[np.ndarray]:
            self.images_processed += len(crops)
            return [np.eye(1, 512, dtype=np.float32)[0] for _ in crops]

    monkeypatch.setattr(assistant, "OSNetEmbeddingExtractor", FakeExtractor)
    video = tmp_path / "review.avi"
    manifest_path = tmp_path / "manifest.json"
    trajectories_path = tmp_path / "trajectories.csv"
    cases_path = tmp_path / "cases.json"
    _write_video(video)
    _write_trajectory_csv(trajectories_path)
    source_hash = "c" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha256": source_hash,
                "output_sha256": _sha256(video),
                "frames_written": 5,
                "source_width": 96,
                "source_height": 96,
                "output_width": 96,
                "output_height": 96,
            }
        ),
        encoding="utf-8",
    )
    cases_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video_frame_count": 5,
                "video": {"sha256": source_hash},
                "source_sha256": {
                    "video": source_hash,
                    "persistent_trajectories": _sha256(trajectories_path),
                },
                "cases": [
                    {
                        "fragment_id": "review-hybrid-0001",
                        "event_type": "nearby_temporal_id_restart",
                        "priority": 4,
                        "start_frame": 1,
                        "event_frame": 3,
                        "end_frame": 5,
                        "temporal_track_id": 2,
                        "candidate_previous_track_id": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = generate_review_proposals(
        video,
        manifest_path,
        trajectories_path,
        cases_path,
        tmp_path / "hybrid-proposals.json",
        tmp_path / "hybrid-evidence",
        reid_method="hybrid",
        embedding_weights_path="weights-placeholder.pth",
        embedding_batch_size=2,
    )

    proposal = result["cases"][0]["candidate_associations"][0]
    assert set(proposal["method_scores"]) == {"hsv_hog", "embedding", "hybrid"}
    assert proposal["score_method"] == "hybrid"
    assert proposal["evidence"]["similarity_embedding"] == pytest.approx(1.0)
    assert result["algorithm"]["learned_reid_model_used"] is True
    assert result["timings_seconds"]["embedding_images_processed"] > 0


def test_does_not_propose_associations_across_detected_scene_change(
    tmp_path: Path,
) -> None:
    video = tmp_path / "review.avi"
    manifest_path = tmp_path / "manifest.json"
    trajectories_path = tmp_path / "trajectories.csv"
    cases_path = tmp_path / "cases.json"
    _write_video(video, scene_cut_at=3)
    _write_trajectory_csv(trajectories_path)
    source_hash = "b" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha256": source_hash,
                "output_sha256": _sha256(video),
                "frames_written": 5,
                "source_width": 96,
                "source_height": 96,
                "output_width": 96,
                "output_height": 96,
            }
        ),
        encoding="utf-8",
    )
    cases_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "video_frame_count": 5,
                "video": {"sha256": source_hash},
                "source_sha256": {
                    "video": source_hash,
                    "persistent_trajectories": _sha256(trajectories_path),
                },
                "cases": [
                    {
                        "fragment_id": "review-0001",
                        "event_type": "nearby_temporal_id_restart",
                        "priority": 4,
                        "start_frame": 1,
                        "event_frame": 3,
                        "end_frame": 5,
                        "temporal_track_id": 2,
                        "candidate_previous_track_id": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = generate_review_proposals(
        video,
        manifest_path,
        trajectories_path,
        cases_path,
        tmp_path / "proposals.json",
        tmp_path / "evidence",
    )

    case = result["cases"][0]
    assert case["possible_camera_change_frames"] == [3]
    assert case["associations_excluded_by_scene_change"] == 1
    assert case["candidate_associations"] == []
    assert any("cambio de escena" in text for text in case["ambiguity_reasons"])


def test_candidate_groups_flag_temporal_conflicts_and_repeated_pairs() -> None:
    proposal_12 = {
        "proposal_id": "p1",
        "relation": "candidate_same_player",
        "track_ids": [1, 2],
        "frames": [2, 4],
        "heuristic_score": 0.84,
    }
    proposal_13 = {
        "proposal_id": "p2",
        "relation": "existing_reid_link",
        "track_ids": [1, 3],
        "frames": [2, 5],
        "heuristic_score": 0.83,
    }
    duplicate_12 = {
        **proposal_12,
        "proposal_id": "p3",
        "case_id": "review-0002",
    }
    cases = [
        {
            "candidate_associations": [
                {**proposal_12, "case_id": "review-0001"},
                proposal_13,
            ]
        },
        {"candidate_associations": [duplicate_12]},
    ]
    tracks = {
        1: [{"frame": 1}, {"frame": 2}],
        2: [{"frame": 3}, {"frame": 4}],
        3: [{"frame": 4}, {"frame": 5}],
    }

    groups, duplicates = _enrich_proposal_relationships(cases, tracks)

    assert len(groups) == 1
    assert groups[0]["track_ids"] == [1, 2, 3]
    assert groups[0]["review_status"] == "conflict_requires_review"
    assert groups[0]["temporal_conflicts"] == [
        {"track_ids": [2, 3], "overlap_frames": 1}
    ]
    repeated = next(item for item in duplicates if item["track_ids"] == [1, 2])
    assert repeated["occurrence_count"] == 2
    assert repeated["repeated_across_cases"] is True
    assert all(
        proposal["candidate_group_id"] == groups[0]["candidate_group_id"]
        for case in cases
        for proposal in case["candidate_associations"]
    )
