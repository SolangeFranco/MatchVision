"""Automatic prioritization of difficult frames for identity review."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matchvision.annotations.mot import read_mot_annotations


@dataclass(frozen=True, slots=True)
class _Box:
    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


def select_review_cases(
    baseline_mot: str | Path,
    persistent_trajectories: str | Path,
    *,
    frame_count: int,
    output_path: str | Path,
    video_path: str | Path | None = None,
    maximum_cases: int = 60,
    radius_frames: int = 30,
    minimum_crossing_distance_heights: float = 0.65,
) -> dict[str, Any]:
    """Create ranked clips around relinks, track gaps/restarts, and crossings."""
    if frame_count < 1 or maximum_cases < 1 or radius_frames < 1:
        raise ValueError("frame_count, maximum_cases, and radius_frames must be positive")
    if not math.isfinite(minimum_crossing_distance_heights) or (
        minimum_crossing_distance_heights <= 0
    ):
        raise ValueError("minimum_crossing_distance_heights must be finite and positive")
    baseline_path = Path(baseline_mot).expanduser().resolve()
    trajectory_path = Path(persistent_trajectories).expanduser().resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline MOT file not found: {baseline_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Persistent trajectory CSV not found: {trajectory_path}"
        )
    source_video_path = (
        Path(video_path).expanduser().resolve() if video_path is not None else None
    )
    if source_video_path is not None and not source_video_path.is_file():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")

    by_frame: dict[int, list[_Box]] = defaultdict(list)
    for row in read_mot_annotations(baseline_path):
        if row.frame > frame_count:
            raise ValueError("Baseline MOT contains frames beyond frame_count")
        box = _Box(
            row.frame,
            row.track_id,
            row.x,
            row.y,
            row.width,
            row.height,
        )
        by_frame[row.frame].append(box)

    events: list[dict[str, Any]] = []
    with trajectory_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "frame",
            "temporal_track_id",
            "persistent_id",
            "identity_state",
            "association_confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Trajectory CSV is missing columns: {sorted(missing)}"
            )
        trajectories = list(reader)
    trajectory_by_track: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in trajectories:
        frame = int(row["frame"])
        if frame < 1 or frame > frame_count:
            raise ValueError(f"Trajectory frame outside range: {frame}")
        trajectory_by_track[int(row["temporal_track_id"])].append(row)

    for row in trajectories:
        if row["identity_state"] != "reidentified":
            continue
        score = (
            float(row["association_confidence"])
            if row["association_confidence"]
            else 0.0
        )
        events.append(
            {
                "event_frame": int(row["frame"]),
                "priority": 1 if score < 0.86 else 2,
                "event_type": "reid_association",
                "summary": (
                    f"Review proposed persistent ID {row['persistent_id']} "
                    f"for temporal ID {row['temporal_track_id']} (score {score:.3f})."
                ),
                "persistent_id": int(row["persistent_id"]),
                "temporal_track_id": int(row["temporal_track_id"]),
                "association_score": score,
            }
        )

    track_ranges: list[tuple[int, int, int]] = []
    for track_id, observations in trajectory_by_track.items():
        frames = sorted(int(row["frame"]) for row in observations)
        if not frames:
            continue
        track_ranges.append((track_id, frames[0], frames[-1]))
        for left, right in zip(frames, frames[1:]):
            gap = right - left - 1
            if 1 <= gap <= radius_frames * 2:
                events.append(
                    {
                        "event_frame": (left + right) // 2,
                        "priority": 3,
                        "event_type": "within_track_disappearance",
                        "summary": (
                            f"Temporal ID {track_id} has a {gap}-frame detection gap."
                        ),
                        "temporal_track_id": track_id,
                        "gap_frames": gap,
                    }
                )

    sorted_ranges = sorted(track_ranges, key=lambda item: item[1])
    for index, earlier in enumerate(sorted_ranges):
        track_id, _, earlier_end = earlier
        for later in sorted_ranges[index + 1 :]:
            other_id, later_start, _ = later
            if later_start - earlier[1] > radius_frames * 4:
                break
            if track_id == other_id:
                continue
            frame_gap = later_start - earlier_end
            if 1 <= frame_gap <= radius_frames * 2:
                events.append(
                    {
                        "event_frame": (earlier_end + later_start) // 2,
                        "priority": 4,
                        "event_type": "nearby_temporal_id_restart",
                        "summary": (
                            f"Temporal ID {other_id} begins {frame_gap} frames after "
                            f"ID {track_id} ends; review possible fragmentation."
                        ),
                        "temporal_track_id": other_id,
                        "candidate_previous_track_id": track_id,
                        "gap_frames": frame_gap,
                    }
                )

    for frame, boxes in by_frame.items():
        for left_index, left in enumerate(boxes):
            lx, ly = left.center
            for right in boxes[left_index + 1 :]:
                rx, ry = right.center
                scale = max((left.height + right.height) / 2, 1.0)
                distance_heights = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5 / scale
                intersection_width = max(
                    0.0,
                    min(left.x + left.width, right.x + right.width)
                    - max(left.x, right.x),
                )
                intersection_height = max(
                    0.0,
                    min(left.y + left.height, right.y + right.height)
                    - max(left.y, right.y),
                )
                intersection = intersection_width * intersection_height
                union = (
                    left.width * left.height
                    + right.width * right.height
                    - intersection
                )
                overlap_iou = intersection / union if union > 0 else 0.0
                if (
                    distance_heights <= minimum_crossing_distance_heights
                    or overlap_iou >= 0.1
                ):
                    events.append(
                        {
                            "event_frame": frame,
                            "priority": 5,
                            "event_type": "close_player_crossing_or_occlusion",
                            "summary": (
                                f"IDs {left.track_id} and {right.track_id} are "
                                f"{distance_heights:.2f} box-heights apart."
                            ),
                            "track_ids": [left.track_id, right.track_id],
                            "normalized_center_distance": distance_heights,
                            "box_iou": overlap_iou,
                        }
                    )

    events.sort(
        key=lambda item: (
            item["priority"],
            item.get("association_score", 1.0),
            item["event_frame"],
            item["event_type"],
        )
    )
    by_priority: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_priority[event["priority"]].append(event)
    selected: list[dict[str, Any]] = []
    cursors = {priority: 0 for priority in by_priority}
    priorities = sorted(by_priority)
    while len(selected) < maximum_cases:
        added_case = False
        for priority in priorities:
            candidates = by_priority[priority]
            while cursors[priority] < len(candidates):
                event = candidates[cursors[priority]]
                cursors[priority] += 1
                is_duplicate = any(
                    existing["event_type"] == event["event_type"]
                    and abs(existing["event_frame"] - event["event_frame"])
                    <= max(radius_frames // 2, 1)
                    and set(existing.get("track_ids", []))
                    == set(event.get("track_ids", []))
                    and existing.get("temporal_track_id")
                    == event.get("temporal_track_id")
                    for existing in selected
                )
                if is_duplicate:
                    continue
                event_frame = int(event["event_frame"])
                event["fragment_id"] = f"review-{len(selected) + 1:04d}"
                event["start_frame"] = max(1, event_frame - radius_frames)
                event["end_frame"] = min(frame_count, event_frame + radius_frames)
                selected.append(event)
                added_case = True
                break
            if len(selected) >= maximum_cases:
                break
        if not added_case:
            break

    payload = {
        "schema_version": 1,
        "video_frame_count": frame_count,
        "source_files": {
            "baseline_mot": str(baseline_path),
            "persistent_trajectories": str(trajectory_path),
        },
        "source_sha256": {
            "baseline_mot": _sha256_file(baseline_path),
            "persistent_trajectories": _sha256_file(trajectory_path),
            "video": (
                _sha256_file(source_video_path)
                if source_video_path is not None
                else None
            ),
        },
        "video": (
            {
                "path": str(source_video_path),
                "sha256": _sha256_file(source_video_path),
            }
            if source_video_path is not None
            else None
        ),
        "selection_policy": {
            "maximum_cases": maximum_cases,
            "radius_frames": radius_frames,
            "minimum_crossing_distance_heights": minimum_crossing_distance_heights,
            "priority_order": [
                "low-confidence ReID proposals",
                "other ReID proposals",
                "within-track disappearance gaps",
                "nearby temporal ID restarts",
                "close crossings/occlusion proxies",
            ],
            "sampling": "round-robin across available priority levels",
        },
        "cases": selected,
        "limitations": (
            "Cases are review candidates, not verified occlusions, identity "
            "switches, or ground truth. The selection uses predicted boxes and "
            "must be confirmed by an annotator."
        ),
    }
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    payload["output_path"] = str(target)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
