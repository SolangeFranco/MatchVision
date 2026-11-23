"""Video metadata and frame iteration utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Metadata reported by the video decoder."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float | None:
        """Return the estimated duration, or None when FPS is unavailable."""
        if self.fps <= 0:
            return None
        return self.frame_count / self.fps


def get_video_info(path: str | Path) -> VideoInfo:
    """Open a video and return decoder metadata.

    Raises:
        FileNotFoundError: If the path does not identify a file.
        ValueError: If OpenCV cannot open the file or metadata is invalid.
    """
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or frame_count < 0:
            raise ValueError(f"Video reports invalid metadata: {video_path}")

        return VideoInfo(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )
    finally:
        capture.release()


def iter_video_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield decoded frames as ``(zero_based_frame_index, BGR image)``.

    The decoder is released even when iteration is stopped early.
    """
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {video_path}")

    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()
