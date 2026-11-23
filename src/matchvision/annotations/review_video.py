"""Create a browser-compatible, frame-aligned H.264 review copy."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2


class ReviewVideoError(ValueError):
    """Raised when a frame-aligned browser review video cannot be created."""


def create_review_video(
    source_path: str | Path,
    output_path: str | Path,
    *,
    frame_count: int,
    maximum_width: int = 1920,
    crf: int = 22,
) -> dict[str, Any]:
    """Transcode the first ``frame_count`` source frames to CFR H.264 MP4."""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    manifest_path = output.with_suffix(output.suffix + ".json")
    if not source.is_file():
        raise FileNotFoundError(f"Source video not found: {source}")
    if output == source:
        raise ReviewVideoError("Review output must not overwrite the source video.")
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if maximum_width < 2:
        raise ValueError("maximum_width must be at least 2 pixels")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite review output or manifest: {output}"
        )

    source_hash = _sha256_file(source)
    container = av.open(str(source), mode="r")
    if not container.streams.video:
        container.close()
        raise ReviewVideoError(f"No video stream found in {source}")
    input_stream = container.streams.video[0]
    rate = input_stream.average_rate or input_stream.base_rate
    if rate is None or rate <= 0:
        container.close()
        raise ReviewVideoError("Source video has no valid average frame rate.")
    rate = Fraction(rate)
    source_width, source_height = input_stream.width, input_stream.height
    source_codec = input_stream.codec_context.name
    output_width = min(source_width, maximum_width)
    output_width -= output_width % 2
    output_height = int(round(source_height * output_width / source_width))
    output_height -= output_height % 2
    if output_width < 2 or output_height < 2:
        container.close()
        raise ReviewVideoError("Calculated review-video dimensions are invalid.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output: Path | None = None
    frame_index = 0
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}.",
            suffix=".mp4",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_output = Path(temporary.name)

        with av.open(str(temporary_output), mode="w", format="mp4") as writer:
            stream = writer.add_stream("libx264", rate=rate)
            stream.width = output_width
            stream.height = output_height
            stream.pix_fmt = "yuv420p"
            stream.options = {
                "crf": str(crf),
                "preset": "fast",
                "profile": "high",
                "movflags": "+faststart",
            }
            stream.codec_context.gop_size = 30
            time_base = Fraction(rate.denominator, rate.numerator)
            stream.time_base = time_base
            stream.codec_context.time_base = time_base

            for input_frame in container.decode(input_stream):
                if frame_index >= frame_count:
                    break
                image = input_frame.to_ndarray(format="bgr24")
                if (image.shape[1], image.shape[0]) != (
                    source_width,
                    source_height,
                ):
                    raise ReviewVideoError(
                        f"Decoded source frame {frame_index + 1} has unexpected "
                        f"dimensions {image.shape[1]}x{image.shape[0]}."
                    )
                resized = cv2.resize(
                    image,
                    (output_width, output_height),
                    interpolation=cv2.INTER_AREA,
                )
                output_frame = av.VideoFrame.from_ndarray(resized, format="bgr24")
                output_frame = output_frame.reformat(
                    width=output_width,
                    height=output_height,
                    format="yuv420p",
                )
                output_frame.pts = frame_index
                output_frame.time_base = time_base
                for packet in stream.encode(output_frame):
                    writer.mux(packet)
                frame_index += 1
            if frame_index < frame_count:
                raise ReviewVideoError(
                    f"Source decoded only {frame_index} frames; "
                    f"{frame_count} were required."
                )
            for packet in stream.encode():
                writer.mux(packet)
    except Exception:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        raise
    finally:
        container.close()

    if frame_index != frame_count:
        temporary_output.unlink(missing_ok=True)
        raise ReviewVideoError(
            f"Encoded {frame_index} frames; expected exactly {frame_count}."
        )

    decoded_count = 0
    decoded_pts: list[int] = []
    try:
        with av.open(str(temporary_output), mode="r") as verification:
            verification_stream = verification.streams.video[0]
            if verification_stream.codec_context.name != "h264":
                raise ReviewVideoError(
                    "Review copy encoder did not produce H.264 output."
                )
            for decoded_frame in verification.decode(verification_stream):
                decoded_count += 1
                if decoded_frame.pts is not None:
                    decoded_pts.append(decoded_frame.pts)
                if decoded_count <= 3 or decoded_count == frame_count:
                    if decoded_frame.to_ndarray(format="bgr24").shape[:2] != (
                        output_height,
                        output_width,
                    ):
                        raise ReviewVideoError(
                            f"Review output has unexpected dimensions on frame "
                            f"{decoded_count}."
                        )
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise
    if decoded_count != frame_count:
        temporary_output.unlink(missing_ok=True)
        raise ReviewVideoError(
            f"Review copy decodes to {decoded_count} frames; expected {frame_count}."
        )
    if len(decoded_pts) != frame_count or any(
        later <= earlier for earlier, later in zip(decoded_pts, decoded_pts[1:])
    ):
        temporary_output.unlink(missing_ok=True)
        raise ReviewVideoError("Review copy presentation timestamps are not increasing.")

    os.replace(temporary_output, output)
    output_hash = _sha256_file(output)
    manifest = {
        "schema_version": 1,
        "source_path": str(source),
        "source_sha256": source_hash,
        "source_codec": source_codec,
        "source_width": source_width,
        "source_height": source_height,
        "source_fps": float(rate),
        "source_fps_fraction": f"{rate.numerator}/{rate.denominator}",
        "output_path": str(output),
        "output_sha256": output_hash,
        "output_codec": "h264",
        "output_width": output_width,
        "output_height": output_height,
        "output_fps_fraction": f"{rate.numerator}/{rate.denominator}",
        "frames_written": frame_index,
        "frames_decoded_for_verification": decoded_count,
        "pts_monotonic": True,
        "first_decoded_pts": decoded_pts[0],
        "last_decoded_pts": decoded_pts[-1],
        "crf": crf,
        "note": (
            "Frames are sequentially copied from source frames 1 through "
            f"{frame_count}; no frame-rate resampling was applied. Video is "
            "downscaled for browser review; annotations use source coordinates."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
