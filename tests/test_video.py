from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.video import get_video_info, iter_video_frames


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable in this environment")
    try:
        for value in (0, 80, 160):
            frame = np.full((24, 32, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_reads_video_metadata_and_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path)

    info = get_video_info(video_path)
    frames = list(iter_video_frames(video_path))

    assert (info.width, info.height, info.frame_count) == (32, 24, 3)
    assert info.duration_seconds == pytest.approx(0.3)
    assert [index for index, _ in frames] == [0, 1, 2]
    assert all(frame.shape == (24, 32, 3) for _, frame in frames)


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_video_info(tmp_path / "missing.mp4")
