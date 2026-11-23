from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from matchvision.annotations.review_video import (
    ReviewVideoError,
    create_review_video,
)


def _create_source(path: Path, frame_count: int = 5) -> None:
    with av.open(str(path), mode="w", format="matroska") as output:
        stream = output.add_stream("ffv1", rate=Fraction(25, 1))
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 25)
        stream.codec_context.time_base = Fraction(1, 25)
        for index in range(frame_count):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :, 0] = index * 30
            image[:, :, 1] = np.arange(64, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, 25)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def test_review_copy_preserves_frame_count_order_timestamps_and_hashes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    output = tmp_path / "review.mp4"
    _create_source(source)

    manifest = create_review_video(
        source, output, frame_count=4, maximum_width=64, crf=18
    )

    assert manifest["source_codec"] == "ffv1"
    assert manifest["output_codec"] == "h264"
    assert manifest["frames_written"] == 4
    assert manifest["frames_decoded_for_verification"] == 4
    assert manifest["first_decoded_pts"] < manifest["last_decoded_pts"]
    assert Path(manifest["output_path"]).is_file()
    assert Path(str(output) + ".json").is_file()
    assert json.loads(Path(str(output) + ".json").read_text())["output_sha256"]
    with av.open(str(output), mode="r") as result:
        frames = list(result.decode(video=0))
    assert len(frames) == 4


def test_review_copy_does_not_overwrite_existing_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    output = tmp_path / "review.mp4"
    _create_source(source)
    output.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_review_video(source, output, frame_count=2)

    assert output.read_bytes() == b"keep"


def test_review_copy_rejects_source_shorter_than_requested(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    _create_source(source, frame_count=2)

    with pytest.raises(ReviewVideoError, match="decoded only 2 frames"):
        create_review_video(source, tmp_path / "review.mp4", frame_count=3)

    assert not (tmp_path / "review.mp4").exists()
