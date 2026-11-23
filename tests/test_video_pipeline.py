import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.detection.base import Detection
from matchvision.detection.ultralytics_yolo import UltralyticsYOLODetector
from matchvision.pipeline.video import (
    _IdentityStatisticsAccumulator,
    _draw_tracks,
    _identity_statistics,
    process_video,
)
from matchvision.reidentification.appearance import PersistentIdentityAssociator
from matchvision.tracking.base import TrackedObject
from matchvision.tracking.bytetrack import ByteTrackAdapter


class FakeDetector:
    model_name = "fake-model"
    model_version = "test-version"
    model_source = "test fixture"
    model_license = "test-only"
    pretrained_on = "synthetic fixture"
    class_names = {0: "person"}
    checkpoint_path = None
    configuration = {"confidence": 0.25, "image_size": 640, "device": "cpu"}

    def __init__(self) -> None:
        self.detect_calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        self.detect_calls += 1
        return [
            Detection(
                x1=2,
                y1=3,
                x2=14,
                y2=22,
                confidence=0.9,
                class_id=0,
                label="person",
            )
        ]


class FakeTracker:
    tracker_name = "fake-tracker"
    tracker_version = "test-version"
    configuration = {}

    def update(
        self, detections: list[Detection], frame_index: int
    ) -> list[TrackedObject]:
        assert frame_index >= 0
        return [TrackedObject(detection, 5) for detection in detections]


def _write_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable")
    try:
        for value in range(frame_count):
            writer.write(np.full((24, 32, 3), value * 40, dtype=np.uint8))
    finally:
        writer.release()


def test_video_pipeline_writes_mot_video_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.avi"
    _write_video(source)
    ground_truth = tmp_path / "gt.txt"
    ground_truth.write_text(
        "".join(
            f"{frame},5,2,3,12,19,1,-1,-1,-1\n"
            for frame in range(1, 4)
        ),
        encoding="utf-8",
    )

    result = process_video(
        source,
        tmp_path / "output",
        FakeDetector(),
        FakeTracker(),
        ground_truth_path=ground_truth,
    )

    assert result.frames_processed == 3
    assert result.detections_tracked == 3
    mot_rows = result.mot_path.read_text(encoding="utf-8").splitlines()
    assert len(mot_rows) == 3
    assert mot_rows[0].startswith("1,5,")
    assert len(mot_rows[0].split(",")) == 10
    assert result.annotated_video_path.is_file()
    assert result.detections_path is not None and result.detections_path.is_file()
    assert (
        result.track_observations_path is not None
        and result.track_observations_path.is_file()
    )
    detection_rows = result.detections_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(detection_rows) == 4
    assert detection_rows[0] == (
        "frame,class_id,label,confidence,x1,y1,x2,y2"
    )
    assert detection_rows[1].startswith("1,0,person,0.9,")
    track_rows = result.track_observations_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(track_rows) == 4
    assert track_rows[1].startswith("1,5,5,,0,person,0.9,")
    capture = cv2.VideoCapture(str(result.annotated_video_path))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        capture.release()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert Path(metadata["outputs"]["detections_csv"]).is_file()
    assert Path(metadata["outputs"]["tracking_observations_csv"]).is_file()
    assert len(
        metadata["outputs"]["detections_csv_sha256"]
    ) == 64
    assert metadata["model"]["name"] == "fake-model"
    assert metadata["model"]["source"] == "test fixture"
    assert metadata["model"]["license"] == "test-only"
    assert metadata["tracker"]["name"] == "fake-tracker"
    assert metadata["tracker"]["version"] == "test-version"
    assert metadata["environment"]["tracker"] == "test-version"
    assert "supervision" not in metadata["environment"]
    assert metadata["results"]["metrics"]["HOTA"] == pytest.approx(1)
    assert result.metrics is not None
    assert result.metrics["HOTA"] == pytest.approx(1)
    assert result.metrics["MOTA"] == pytest.approx(1)
    assert result.metrics["IDF1"] == pytest.approx(1)
    assert result.tracking_stats_path is None
    assert metadata["timings"]["processing_seconds"] >= 0
    assert set(metadata["timings"]["stage_seconds"]) == {
        "video_decode",
        "detection",
        "tracking",
        "reidentification",
        "render_and_export",
    }
    assert all(
        value >= 0 for value in metadata["timings"]["stage_seconds"].values()
    )
    assert all(
        value >= 0
        for value in metadata["timings"][
            "stage_average_milliseconds_per_frame"
        ].values()
    )


def test_video_pipeline_exports_persistent_identity_trajectories(
    tmp_path: Path,
) -> None:
    source = tmp_path / "persistent-source.avi"
    _write_video(source, frame_count=3)
    result = process_video(
        source,
        tmp_path / "persistent-output",
        FakeDetector(),
        FakeTracker(),
        identity_assigner=PersistentIdentityAssociator(
            scene_cut_mean_difference=1000
        ),
    )

    assert result.trajectory_path is not None and result.trajectory_path.is_file()
    assert (
        result.tracking_stats_path is not None
        and result.tracking_stats_path.is_file()
    )
    tracking_stats = json.loads(
        result.tracking_stats_path.read_text(encoding="utf-8")
    )
    assert tracking_stats["frames_processed"] == 3
    assert tracking_stats["tracked_observations"] == 3
    trajectory_rows = result.trajectory_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(trajectory_rows) == 4
    assert trajectory_rows[0].startswith(
        "frame,temporal_track_id,persistent_id,"
    )
    assert len({row.split(",")[2] for row in trajectory_rows[1:]}) == 1
    mot_rows = result.mot_path.read_text(encoding="utf-8").splitlines()
    assert all(int(row.split(",")[1]) == 1 for row in mot_rows)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["outputs"]["tracking_stats_json"] == str(
        result.tracking_stats_path
    )
    assert Path(metadata["outputs"]["tracking_stats_json"]).is_file()
    assert metadata["results"]["persistent_identity_statistics"][
        "persistent_identity_ids"
    ] == 1
    assert (
        metadata["results"]["persistent_identity_statistics"][
            "tracked_observations"
        ]
        == 3
    )
    assert all(
        value >= 0
        for value in metadata["timings"]["stage_seconds"].values()
    )
    reid_timings = metadata["timings"]["reidentification_detail"]
    assert reid_timings["hsv_hog_extraction_seconds"] > 0
    assert reid_timings["association_seconds"] > 0
    assert reid_timings["embedding_images_processed"] == 0
    assert metadata["configuration"]["identity_association"][
        "cross_scene_reidentification"
    ] is False


def test_streaming_identity_statistics_match_unique_frame_semantics() -> None:
    accumulator = _IdentityStatisticsAccumulator()
    for record in (
        {
            "frame": 1,
            "persistent_id": 1,
            "temporal_track_id": 10,
            "class_id": 0,
            "association_confidence": "",
        },
        {
            "frame": 1,
            "persistent_id": 1,
            "temporal_track_id": 11,
            "class_id": 0,
            "association_confidence": "",
        },
        {
            "frame": 3,
            "persistent_id": 1,
            "temporal_track_id": 11,
            "class_id": 0,
            "association_confidence": 0.75,
        },
        {
            "frame": 5,
            "persistent_id": 2,
            "temporal_track_id": 20,
            "class_id": 0,
            "association_confidence": "",
        },
    ):
        accumulator.add(record)

    stats = _identity_statistics(
        accumulator,
        frames_processed=5,
        identity_assigner=PersistentIdentityAssociator(
            scene_cut_mean_difference=1000
        ),
    )

    assert stats["tracked_observations"] == 4
    assert stats["persistent_identity_ids"] == 2
    assert stats["temporal_tracklets_merged_proxy"] == 1
    assert stats["persistent_identity_observations"] == {
        "mean": 1.5,
        "median": 1.5,
        "minimum": 1,
        "maximum": 2,
    }
    assert stats["within_identity_gap_proxy"] == {
        "absence_events": 1,
        "missing_frames": 1,
    }
    assert stats["detections_per_frame"] == {
        "mean_over_all_video_frames": 0.8,
        "mean_nonempty_frame": pytest.approx(4 / 3),
        "empty_frames": 2,
        "frames_with_observations": 3,
        "maximum": 2,
    }
    assert stats["reidentification_association_score"] == {
        "count": 1,
        "mean": 0.75,
        "minimum": 0.75,
    }


def test_video_annotation_includes_class_id_and_confidence(monkeypatch) -> None:
    texts: list[str] = []

    def capture_label(
        image: np.ndarray,
        text: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        del image, args, kwargs
        texts.append(text)

    monkeypatch.setattr(
        cv2,
        "putText",
        capture_label,
    )
    track = TrackedObject(
        Detection(1, 2, 10, 12, 0.9, 0, "person"),
        track_id=7,
    )

    annotated = _draw_tracks(
        np.zeros((24, 32, 3), dtype=np.uint8),
        [track],
    )

    assert annotated.shape == (24, 32, 3)
    assert texts == ["person ID:7 0.90"]


def test_bytetrack_adapter_preserves_identity_between_frames() -> None:
    pytest.importorskip("ultralytics")
    tracker = ByteTrackAdapter(
        {0: "person"},
        track_activation_threshold=0.1,
        lost_track_buffer=30,
    )
    observed_ids: list[int] = []
    for frame_index in range(4):
        detection = Detection(
            x1=5 + frame_index,
            y1=5,
            x2=20 + frame_index,
            y2=30,
            confidence=0.95,
            class_id=0,
            label="person",
        )
        tracked = tracker.update([detection], frame_index)
        if tracked:
            observed_ids.append(tracked[0].track_id)

    assert len(observed_ids) >= 2
    assert len(set(observed_ids)) == 1


def test_video_pipeline_runs_with_actual_bytetrack_on_fixture_video(
    tmp_path: Path,
) -> None:
    pytest.importorskip("ultralytics")
    source = tmp_path / "bytetrack-input.avi"
    _write_video(source, frame_count=4)
    tracker = ByteTrackAdapter(
        {0: "person"},
        track_activation_threshold=0.1,
        lost_track_buffer=30,
    )

    result = process_video(
        source,
        tmp_path / "bytetrack-output",
        FakeDetector(),
        tracker,
    )

    rows = result.mot_path.read_text(encoding="utf-8").splitlines()
    assert result.frames_processed == 4
    assert len(rows) >= 2
    assert len({row.split(",")[1] for row in rows}) == 1
    assert result.annotated_video_path.is_file()


def test_bytetrack_rejects_nonconsecutive_frames() -> None:
    pytest.importorskip("ultralytics")
    tracker = ByteTrackAdapter({0: "person"})

    with pytest.raises(ValueError, match="Expected consecutive frame index"):
        tracker.update([], frame_index=1)


def test_yolo_adapter_uses_generic_person_and_ball_classes_without_download(
    monkeypatch,
) -> None:
    class ArrayValue:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def cpu(self) -> "ArrayValue":
            return self

        def numpy(self) -> np.ndarray:
            return self.value

    class FakeBoxes:
        xyxy = ArrayValue(np.asarray([[1, 2, 11, 22], [8, 9, 12, 13]]))
        conf = ArrayValue(np.asarray([0.9, 0.8]))
        cls = ArrayValue(np.asarray([0, 32]))

        def __len__(self) -> int:
            return 2

    class FakeModel:
        names = {0: "person", 32: "sports ball", 3: "car"}
        ckpt_path = None

        def predict(self, **kwargs):
            assert kwargs["classes"] == [0, 32]
            return [types.SimpleNamespace(boxes=FakeBoxes())]

    def create_model(model_path: str) -> FakeModel:
        del model_path
        return FakeModel()

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.__version__ = "test"
    fake_ultralytics.YOLO = create_model
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    detector = UltralyticsYOLODetector()
    detections = detector.detect(np.zeros((24, 32, 3), dtype=np.uint8))

    assert detector.class_names == {0: "person", 32: "sports ball"}
    assert detector.pretrained_on == "COCO"
    assert [item.label for item in detections] == ["person", "sports ball"]


def test_video_pipeline_replays_cached_detections_on_resume(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-source.avi"
    _write_video(source, frame_count=5)
    checkpoint_dir = tmp_path / "detection-checkpoints"

    first_detector = FakeDetector()
    first = process_video(
        source,
        tmp_path / "first-output",
        first_detector,
        FakeTracker(),
        checkpoint_dir=checkpoint_dir,
        checkpoint_segment_frames=2,
    )
    assert first_detector.detect_calls == 5

    resumed_detector = FakeDetector()
    resumed = process_video(
        source,
        tmp_path / "resumed-output",
        resumed_detector,
        FakeTracker(),
        checkpoint_dir=checkpoint_dir,
        checkpoint_segment_frames=2,
    )
    metadata = json.loads(resumed.metadata_path.read_text(encoding="utf-8"))
    assert resumed_detector.detect_calls == 0
    assert resumed.frames_processed == first.frames_processed == 5
    assert resumed.detections_tracked == first.detections_tracked
    assert metadata["detection_checkpoints"]["cached_segments_reused"] == 3
    assert metadata["detection_checkpoints"]["cached_frames_replayed"] == 5
    assert metadata["detection_checkpoints"]["new_segments_written"] == 0
    assert metadata["timings"]["stage_seconds"]["detection_cache_read"] >= 0

    incompatible_detector = FakeDetector()
    incompatible_detector.configuration = {
        **FakeDetector.configuration,
        "image_size": 960,
    }
    with pytest.raises(ValueError, match="no coinciden con el video"):
        process_video(
            source,
            tmp_path / "incompatible-output",
            incompatible_detector,
            FakeTracker(),
            checkpoint_dir=checkpoint_dir,
            checkpoint_segment_frames=2,
        )
