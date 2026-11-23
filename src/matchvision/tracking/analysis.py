"""Descriptive analysis of tracker MOT exports (without identity ground truth)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from matchvision.annotations.mot import read_mot_annotations


def analyze_mot_file(
    mot_path: str | Path,
    frame_count: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write per-track CSV, descriptive JSON, and a PNG from one MOT file."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    source = Path(mot_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MOT input does not exist: {source}")
    annotations = read_mot_annotations(source)
    if any(item.frame > frame_count for item in annotations):
        raise ValueError("MOT input contains frames beyond frame_count")

    by_track: dict[int, list[int]] = defaultdict(list)
    by_frame: Counter[int] = Counter()
    for item in annotations:
        by_track[item.track_id].append(item.frame)
        by_frame[item.frame] += 1

    tracks: list[dict[str, int]] = []
    total_segments = 0
    internal_gap_events = 0
    missing_frames = 0
    for track_id, observed_frames in sorted(by_track.items()):
        frames = sorted(observed_frames)
        gaps = [right - left - 1 for left, right in zip(frames, frames[1:]) if right - left > 1]
        segments = 1 + len(gaps)
        total_segments += segments
        internal_gap_events += len(gaps)
        missing_frames += sum(gaps)
        tracks.append(
            {
                "track_id": track_id,
                "detections": len(frames),
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "span_frames": frames[-1] - frames[0] + 1,
                "presence_segments_proxy": segments,
                "internal_absence_events": len(gaps),
                "missing_frames_between_observations": sum(gaps),
                "maximum_internal_gap": max(gaps, default=0),
            }
        )

    detections_per_frame = [by_frame.get(frame, 0) for frame in range(1, frame_count + 1)]
    detection_counts = np.asarray(detections_per_frame, dtype=np.int64)
    durations = np.asarray([item["detections"] for item in tracks], dtype=np.int64)
    summary: dict[str, Any] = {
        "input_mot": str(source),
        "frames_processed": frame_count,
        "tracked_detections": len(annotations),
        "unique_track_ids": len(tracks),
        "track_observation_count": {
            "mean": float(np.mean(durations)) if durations.size else 0.0,
            "median": float(np.median(durations)) if durations.size else 0.0,
            "minimum": int(durations.min()) if durations.size else 0,
            "maximum": int(durations.max()) if durations.size else 0,
        },
        "track_span_frames": {
            "mean": (
                float(np.mean([item["span_frames"] for item in tracks]))
                if tracks
                else 0.0
            ),
            "median": (
                float(np.median([item["span_frames"] for item in tracks]))
                if tracks
                else 0.0
            ),
            "minimum": min(
                (item["span_frames"] for item in tracks), default=0
            ),
            "maximum": max(
                (item["span_frames"] for item in tracks), default=0
            ),
        },
        "short_tracks": {
            "five_detections_or_less": int(np.count_nonzero(durations <= 5)),
            "ten_detections_or_less": int(np.count_nonzero(durations <= 10)),
        },
        "trajectory_presence_proxy": {
            "consecutive_presence_segments": total_segments,
            "excess_segments_over_ids": max(total_segments - len(tracks), 0),
            "internal_absence_events": internal_gap_events,
            "missing_frames_between_observations": missing_frames,
            "tracks_with_internal_absence": sum(
                item["internal_absence_events"] > 0 for item in tracks
            ),
        },
        "detections_per_frame": {
            "mean_over_all_frames": (
                float(np.mean(detection_counts)) if detection_counts.size else 0.0
            ),
            "median_nonempty_frame": (
                float(np.median(detection_counts[detection_counts > 0]))
                if np.any(detection_counts > 0)
                else 0.0
            ),
            "maximum": int(detection_counts.max()) if detection_counts.size else 0,
            "empty_frames": int(np.count_nonzero(detection_counts == 0)),
            "frames_with_detections": int(np.count_nonzero(detection_counts > 0)),
        },
        "interpretation": (
            "Descriptive proxies only. Presence gaps and ID counts are not "
            "ground-truth fragmentation, ID switches, MOTA, IDF1, or HOTA."
        ),
    }

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tracks_path = output / "per_track.csv"
    with tracks_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tracks[0]) if tracks else [
            "track_id",
            "detections",
            "first_frame",
            "last_frame",
            "span_frames",
            "presence_segments_proxy",
            "internal_absence_events",
            "missing_frames_between_observations",
            "maximum_internal_gap",
        ])
        writer.writeheader()
        writer.writerows(tracks)
    summary["per_track_csv"] = str(tracks_path)

    json_path = output / "tracking_analysis.json"
    summary["summary_json"] = str(json_path)

    image_path = output / "tracking_analysis.png"
    _write_plot(image_path, detection_counts, durations)
    summary["visualization_png"] = str(image_path)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_plot(
    path: Path, detections_per_frame: np.ndarray, durations: np.ndarray
) -> None:
    canvas = np.full((720, 1280, 3), 248, dtype=np.uint8)
    cv2.putText(
        canvas, "Detections per frame", (50, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA
    )
    _draw_line_chart(
        canvas,
        detections_per_frame,
        x=60,
        y=75,
        width=1150,
        height=240,
        color=(190, 90, 30),
    )
    cv2.putText(
        canvas, "Track observations (count per track)", (50, 375),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA
    )
    _draw_histogram(canvas, durations, x=60, y=410, width=1150, height=245)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write tracking visualization: {path}")


def _draw_line_chart(
    canvas: np.ndarray,
    values: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (210, 210, 210), 1)
    if values.size == 0:
        return
    maximum = max(int(values.max()), 1)
    cv2.putText(
        canvas, str(maximum), (x + 4, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, "0", (x + 4, y + height - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, "frame 1", (x + 4, y + height + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, f"frame {values.size}", (x + width - 95, y + height + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA
    )
    points = [
        (
            x + int(index * width / max(values.size - 1, 1)),
            y + height - int(value * (height - 12) / maximum),
        )
        for index, value in enumerate(values)
    ]
    for start, end in zip(points, points[1:]):
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)


def _draw_histogram(
    canvas: np.ndarray,
    values: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (210, 210, 210), 1)
    if values.size == 0:
        return
    bins = min(30, max(5, int(np.sqrt(values.size))))
    counts, _ = np.histogram(values, bins=bins)
    maximum = max(int(counts.max()), 1)
    cv2.putText(
        canvas, str(maximum), (x + 4, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, f"{int(values.min())} obs", (x + 4, y + height + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, f"{int(values.max())} obs", (x + width - 95, y + height + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA
    )
    bar_width = max(width // bins - 3, 1)
    for index, count in enumerate(counts):
        bar_height = int(count * (height - 10) / maximum)
        left = x + index * width // bins
        cv2.rectangle(
            canvas,
            (left + 1, y + height - bar_height),
            (left + bar_width, y + height - 1),
            (55, 145, 75),
            -1,
        )
