from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.teams.classification import UniformTeamClassifier


def _kit_frame(color: tuple[int, int, int]) -> np.ndarray:
    frame = np.full((120, 100, 3), (35, 110, 30), dtype=np.uint8)
    cv2.rectangle(frame, (30, 22), (70, 65), color, -1)
    return frame


def _add_samples(
    classifier: UniformTeamClassifier,
    identity: str,
    color: tuple[int, int, int],
) -> None:
    for frame_number in (1, 6, 11):
        classifier.observe(
            _kit_frame(color), identity, (20, 5, 80, 110), frame_number
        )


def test_groups_shirt_colors_without_merging_person_identities() -> None:
    classifier = UniformTeamClassifier()
    for identity in ("blue-1", "blue-2"):
        _add_samples(classifier, identity, (220, 40, 20))
    for identity in ("red-1", "red-2"):
        _add_samples(classifier, identity, (20, 25, 220))

    assignments = classifier.classify()

    assert assignments["blue-1"].team_id == assignments["blue-2"].team_id
    assert assignments["red-1"].team_id == assignments["red-2"].team_id
    assert assignments["blue-1"].team_id != assignments["red-1"].team_id
    assert all(item.state == "assigned" for item in assignments.values())
    assert set(assignments) == {"blue-1", "blue-2", "red-1", "red-2"}


def test_manual_team_correction_takes_precedence_and_can_be_cleared() -> None:
    classifier = UniformTeamClassifier()
    for identity in ("blue-1", "blue-2"):
        _add_samples(classifier, identity, (220, 40, 20))
    for identity in ("red-1", "red-2"):
        _add_samples(classifier, identity, (20, 25, 220))
    previous_team = classifier.classify()["blue-1"].team_id
    corrected_team = "equipo_b" if previous_team == "equipo_a" else "equipo_a"

    classifier.set_manual_assignment("blue-1", corrected_team)
    corrected = classifier.classify()["blue-1"]
    assert corrected.team_id == corrected_team
    assert corrected.state == "manual"
    assert corrected.confidence is None

    classifier.clear_manual_assignment("blue-1")
    assert classifier.classify()["blue-1"].state == "assigned"


def test_identical_kits_remain_unassigned_until_manually_reviewed() -> None:
    classifier = UniformTeamClassifier()
    for identity in ("one", "two", "three", "four"):
        _add_samples(classifier, identity, (40, 180, 60))

    assignments = classifier.classify()

    assert {item.state for item in assignments.values()} == {"insufficient_data"}
    assert all(item.team_id is None for item in assignments.values())
    classifier.set_manual_assignment("one", "equipo_a")
    assert classifier.classify()["one"].state == "manual"


def test_identity_without_a_usable_torso_crop_is_kept_as_unclassified() -> None:
    classifier = UniformTeamClassifier()
    accepted = classifier.observe(
        _kit_frame((20, 25, 220)), "tiny", (20, 5, 21, 6), 1
    )

    assert not accepted
    assert classifier.classify()["tiny"].state == "insufficient_samples"
    assert classifier.classify()["tiny"].sample_count == 0


def test_sampling_is_limited_and_invalid_frame_order_is_rejected() -> None:
    classifier = UniformTeamClassifier(
        sample_interval_frames=5,
        maximum_samples_per_identity=2,
    )
    frame = _kit_frame((20, 25, 220))
    assert classifier.observe(frame, "one", (20, 5, 80, 110), 1)
    assert not classifier.observe(frame, "one", (20, 5, 80, 110), 3)
    assert classifier.observe(frame, "one", (20, 5, 80, 110), 6)
    assert classifier.observe(frame, "one", (20, 5, 80, 110), 11)
    assert classifier.classify()["one"].sample_count == 2
    with pytest.raises(ValueError, match="orden creciente"):
        classifier.observe(frame, "one", (20, 5, 80, 110), 9)


def test_rejects_invalid_team_labels_and_boxes() -> None:
    classifier = UniformTeamClassifier()
    with pytest.raises(ValueError, match="team_id"):
        classifier.set_manual_assignment("one", "árbitro")
    with pytest.raises(ValueError, match="coordenadas"):
        classifier.observe(
            _kit_frame((20, 25, 220)), "one", (0, 0, float("nan"), 10), 1
        )


def test_manual_override_file_is_applied_and_outputs_are_not_overwritten(
    tmp_path: Path,
) -> None:
    from matchvision.teams.cli import classify_trajectory_teams

    video_path = tmp_path / "source.avi"
    trajectory_path = tmp_path / "trajectories.csv"
    output_path = tmp_path / "teams.csv"
    overrides_path = tmp_path / "overrides.json"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (100, 120),
    )
    assert writer.isOpened()
    for _ in range(11):
        writer.write(_kit_frame((40, 180, 60)))
    writer.release()

    with trajectory_path.open("w", encoding="utf-8", newline="") as stream:
        rows = csv.DictWriter(
            stream,
            fieldnames=(
                "frame",
                "temporal_track_id",
                "persistent_id",
                "label",
                "x1",
                "y1",
                "x2",
                "y2",
            ),
        )
        rows.writeheader()
        for frame_number in (1, 6, 11):
            rows.writerow(
                {
                    "frame": frame_number,
                    "temporal_track_id": 1,
                    "persistent_id": 42,
                    "label": "person",
                    "x1": 20,
                    "y1": 5,
                    "x2": 80,
                    "y2": 110,
                }
            )
    overrides_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity_column": "persistent_id",
                "assignments": {"42": "equipo_b"},
            }
        ),
        encoding="utf-8",
    )

    result = classify_trajectory_teams(
        video_path,
        trajectory_path,
        output_path,
        overrides_path=overrides_path,
    )

    with output_path.open(encoding="utf-8", newline="") as stream:
        assignments = list(csv.DictReader(stream))
    assert assignments == [
        {
            "identity_id": "42",
            "team_id": "equipo_b",
            "state": "manual",
            "confidence": "",
            "appearance_samples": "3",
            "assignment_basis": "manual",
        }
    ]
    assert result["inputs"]["video_sha256"]
    assert result["output_csv_sha256"]
    assert Path(str(output_path) + ".json").is_file()
    with pytest.raises(FileExistsError, match="sobrescribir"):
        classify_trajectory_teams(
            video_path,
            trajectory_path,
            output_path,
            overrides_path=overrides_path,
        )
