import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.ball.cli import process_ball_video
from matchvision.ball.tracking import BallTrackConfig, BallTracker
from matchvision.detection.base import Detection


def _ball(
    center_x: float,
    center_y: float = 20,
    *,
    confidence: float = 0.9,
    class_id: int = 32,
    label: str = "sports ball",
    width: float = 4,
    height: float = 4,
) -> Detection:
    return Detection(
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
        confidence,
        class_id,
        label,
    )


def test_tracker_predicts_during_occlusion_and_reacquires_same_segment() -> None:
    tracker = BallTracker(
        BallTrackConfig(
            maximum_speed_pixels_per_frame=20,
            gate_margin_pixels=4,
            maximum_missed_frames=2,
            velocity_smoothing=0,
            velocity_damping=1,
        )
    )

    first = tracker.update([_ball(10)], 0, (100, 200))
    second = tracker.update([_ball(20)], 1, (100, 200))
    predicted = tracker.update([], 2, (100, 200))
    recovered = tracker.update([_ball(40)], 3, (100, 200))

    assert first.track_id == second.track_id == predicted.track_id == recovered.track_id
    assert predicted.status == "predicted"
    assert predicted.center_x == pytest.approx(30)
    assert predicted.confidence is None
    assert recovered.status == "observed"
    assert recovered.reacquired


def test_tracker_requires_ball_label_unless_class_is_explicit() -> None:
    person = _ball(10, label="person", class_id=0)

    assert BallTracker().update([person], 0, (100, 200)).status == "missing"
    explicit = BallTracker(allowed_class_ids=[0])
    assert explicit.update([person], 0, (100, 200)).status == "observed"


def test_tracker_rejects_oversized_and_extremely_elongated_candidates() -> None:
    tracker = BallTracker()
    oversized = _ball(30, width=50, height=50)
    elongated = _ball(30, width=40, height=2)

    assert tracker.update([oversized], 0, (100, 200)).status == "missing"
    assert tracker.update([elongated], 1, (100, 200)).status == "missing"


def test_tracker_prefers_candidate_near_motion_prediction() -> None:
    tracker = BallTracker(
        BallTrackConfig(
            maximum_speed_pixels_per_frame=100,
            gate_margin_pixels=5,
            velocity_smoothing=0,
        )
    )
    tracker.update([_ball(20)], 0, (100, 200))
    tracker.update([_ball(30)], 1, (100, 200))

    selected = tracker.update(
        [
            _ball(42, confidence=0.2),
            _ball(70, confidence=0.99),
        ],
        2,
        (100, 200),
    )

    assert selected.status == "observed"
    assert selected.center_x == pytest.approx(42)


def test_tracker_starts_a_new_segment_after_prediction_gap_expires() -> None:
    tracker = BallTracker(
        BallTrackConfig(maximum_missed_frames=1, velocity_damping=1)
    )
    first = tracker.update([_ball(10)], 0, (100, 200))
    predicted = tracker.update([], 1, (100, 200))
    missing = tracker.update([], 2, (100, 200))
    restarted = tracker.update([_ball(100)], 3, (100, 200))

    assert first.track_id == predicted.track_id == 1
    assert predicted.status == "predicted"
    assert missing.status == "missing"
    assert restarted.status == "observed"
    assert restarted.track_id == 2
    assert not restarted.reacquired


def test_tracker_rejects_nonconsecutive_frames_and_invalid_configuration() -> None:
    tracker = BallTracker()
    with pytest.raises(ValueError, match="fotograma"):
        tracker.update([], 1, (100, 200))
    with pytest.raises(ValueError, match="maximum_missed_frames"):
        BallTrackConfig(maximum_missed_frames=-1)
    with pytest.raises(ValueError, match="finita"):
        BallTrackConfig(maximum_speed_pixels_per_frame=float("nan"))
    with pytest.raises(ValueError, match="entero"):
        tracker.update([], 0.5, (100, 200))


class _FakeDetector:
    model_name = "fake-ball-model"
    model_version = "1"
    model_source = "fixture"
    model_license = "test-only"
    pretrained_on = "synthetic fixture"
    class_names = {32: "sports ball"}
    checkpoint_path = None
    configuration = {"confidence": 0.1, "image_size": 640, "device": "cpu"}

    def detect(self, frame: np.ndarray) -> list[Detection]:
        del frame
        return [_ball(10, width=2, height=2)]


def _write_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV no pudo crear el video de prueba")
    try:
        for value in range(frame_count):
            writer.write(np.full((24, 32, 3), value * 40, dtype=np.uint8))
    finally:
        writer.release()


def test_process_ball_video_writes_csv_manifest_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "source.avi"
    output_path = tmp_path / "results" / "ball.csv"
    _write_video(video_path)

    result = process_ball_video(
        video_path,
        output_path,
        _FakeDetector(),
        BallTracker(),
    )

    assert result.frames_processed == 3
    assert result.observed_frames == 3
    assert result.predicted_frames == 0
    with output_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {row["estado"] for row in rows} == {"detectada"}
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model"]["name"] == "fake-ball-model"
    assert manifest["tracking"]["observed_frames"] == 3
    assert manifest["output"]["csv_sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    previous_bytes = output_path.read_bytes()
    with pytest.raises(FileExistsError, match="sobrescribir"):
        process_ball_video(
            video_path,
            output_path,
            _FakeDetector(),
            BallTracker(),
        )
    assert output_path.read_bytes() == previous_bytes
