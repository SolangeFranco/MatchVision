from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.detection.base import Detection
from matchvision.reidentification.appearance import PersistentIdentityAssociator
from matchvision.reidentification.base import IdentifiedObject
from matchvision.tracking.analysis import analyze_mot_file
from matchvision.tracking.base import TrackedObject
from matchvision.pipeline.video import _draw_identified_tracks


def _person(class_id: int, track_id: int, x: int = 20) -> TrackedObject:
    return TrackedObject(
        Detection(
            x1=x,
            y1=10,
            x2=x + 32,
            y2=90,
            confidence=0.95,
            class_id=class_id,
            label="person",
        ),
        track_id=track_id,
    )


def _textured_frame(value: int = 0) -> np.ndarray:
    frame = np.full((120, 100, 3), value, dtype=np.uint8)
    cv2.rectangle(frame, (20, 10), (52, 90), (30, 80, 190), -1)
    cv2.rectangle(frame, (28, 20), (44, 58), (210, 220, 230), -1)
    cv2.line(frame, (21, 60), (50, 80), (10, 20, 30), 4)
    return frame


def test_reidentifies_tracklet_with_matching_appearance_and_motion() -> None:
    associator = PersistentIdentityAssociator(
        active_identity_protection=2,
        maximum_reidentification_gap=50,
    )
    frame = _textured_frame()

    first = associator.update([_person(0, 1)], frame, 0)[0]
    for frame_index in range(1, 10):
        associator.update([], frame, frame_index)
    recovered = associator.update([_person(0, 2, x=21)], frame, 10)[0]

    assert first.identity_state == "new"
    assert recovered.identity_state == "reidentified"
    assert recovered.persistent_id == first.persistent_id
    assert recovered.track_id == 2
    assert recovered.association_confidence is not None
    assert 0.82 <= recovered.association_confidence <= 1
    assert associator.reidentification_count == 1


def test_ambiguous_simultaneous_candidates_are_not_merged() -> None:
    associator = PersistentIdentityAssociator(
        active_identity_protection=1,
        maximum_reidentification_gap=50,
    )
    frame = _textured_frame()
    initial = associator.update([_person(0, 1)], frame, 0)[0]
    associator.update([], frame, 1)
    associator.update([], frame, 2)

    candidates = associator.update(
        [_person(0, 2), _person(0, 3)],
        frame,
        3,
    )

    assert initial.persistent_id not in {item.persistent_id for item in candidates}
    assert len({item.persistent_id for item in candidates}) == 2
    assert associator.reidentification_count == 0


def test_scene_cut_blocks_cross_scene_identity_association() -> None:
    associator = PersistentIdentityAssociator(
        active_identity_protection=0,
        maximum_reidentification_gap=50,
    )
    first = associator.update([_person(0, 1)], _textured_frame(), 0)[0]
    changed_scene = np.full((120, 100, 3), 230, dtype=np.uint8)
    second = associator.update(
        [_person(0, 2)],
        changed_scene,
        1,
    )[0]

    assert second.persistent_id != first.persistent_id
    assert second.identity_state == "new"
    assert associator.scene_cut_count == 1


def test_associator_requires_consecutive_frames() -> None:
    associator = PersistentIdentityAssociator()

    with pytest.raises(ValueError, match="Expected consecutive frame index"):
        associator.update([], _textured_frame(), 1)


def test_identified_track_visualization_displays_both_ids_and_confidence(
    monkeypatch,
) -> None:
    labels: list[str] = []

    def capture_label(
        frame: np.ndarray,
        text: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        del frame, args, kwargs
        labels.append(text)

    monkeypatch.setattr(
        cv2,
        "putText",
        capture_label,
    )
    item = IdentifiedObject(
        detection=Detection(20, 10, 52, 90, 0.9, 0, "person"),
        track_id=12,
        persistent_id=3,
        identity_state="reidentified",
        association_confidence=0.87,
        history=((25, 30), (30, 40)),
    )

    frame = _draw_identified_tracks(_textured_frame(), [item])

    assert frame.shape == (120, 100, 3)
    assert labels == [
        "person T:12 P:3 reidentified A:0.87 D:0.90"
    ]


def test_baseline_mot_analysis_writes_descriptive_outputs(tmp_path: Path) -> None:
    mot = tmp_path / "baseline.txt"
    mot.write_text(
        "1,4,0,0,10,20,0.9,-1,-1,-1\n"
        "2,4,0,0,10,20,0.9,-1,-1,-1\n"
        "4,4,0,0,10,20,0.9,-1,-1,-1\n"
        "1,9,20,0,10,20,0.8,-1,-1,-1\n",
        encoding="utf-8",
    )

    result = analyze_mot_file(mot, frame_count=4, output_dir=tmp_path / "analysis")

    assert result["unique_track_ids"] == 2
    assert result["trajectory_presence_proxy"]["internal_absence_events"] == 1
    assert result["trajectory_presence_proxy"]["excess_segments_over_ids"] == 1
    assert Path(result["per_track_csv"]).is_file()
    assert Path(result["summary_json"]).is_file()
    assert Path(result["visualization_png"]).is_file()
    assert cv2.imread(result["visualization_png"]) is not None
