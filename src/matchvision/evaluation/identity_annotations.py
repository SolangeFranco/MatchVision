"""Validation and evaluation for exhaustively identity-annotated video frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from matchvision.annotations.mot import MOTAnnotation
from matchvision.evaluation.mot import evaluate_mot_sequence, load_mot_file


class IdentityAnnotationError(ValueError):
    """Raised for invalid or insufficient identity annotations."""


_ENTITY_TYPES = {"player", "goalkeeper", "referee", "ball", "other"}
_TRAJECTORY_EVENT_TYPES = {
    "appearance",
    "disappearance",
    "occlusion_start",
    "occlusion_end",
    "crossing",
    "camera_change",
}


def _sampling_plan_hash_content(plan: dict[str, Any]) -> dict[str, Any]:
    video = plan["video"]
    return {
        "schema_version": plan["schema_version"],
        "record_type": plan["record_type"],
        "dataset_id": plan["dataset_id"],
        "protocol": plan["protocol"],
        "video": {
            key: video[key]
            for key in ("sha256", "frame_count", "width", "height", "fps")
        },
        "sampling": plan["sampling"],
        "dense_windows": plan["dense_windows"],
        "hard_cases": plan["hard_cases"],
        "planned_frames": plan["planned_frames"],
        "quality_requirements": plan["quality_requirements"],
    }


def _validate_embedded_sampling_plan(payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload.get("sampling_plan")
    if plan is None:
        return {}
    if not isinstance(plan, dict):
        raise IdentityAnnotationError("sampling_plan must be an object")
    if (
        plan.get("schema_version") != 1
        or plan.get("record_type") != "matchvision_annotation_sampling_plan"
        or not isinstance(plan.get("plan_id"), str)
        or not re.fullmatch(r"mv-plan-[0-9a-f]{16}", plan["plan_id"])
        or not re.fullmatch(r"[0-9a-f]{64}", str(plan.get("plan_sha256", "")))
        or not isinstance(plan.get("protocol"), dict)
        or plan["protocol"].get("protocol_id")
        != "matchvision-football-ground-truth"
        or plan["protocol"].get("version") != "1.0"
    ):
        raise IdentityAnnotationError("sampling_plan metadata is invalid")
    canonical = json.dumps(
        _sampling_plan_hash_content(plan),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected_hash = hashlib.sha256(canonical).hexdigest()
    if plan["plan_sha256"] != expected_hash:
        raise IdentityAnnotationError("sampling_plan SHA-256 does not match")
    if plan["plan_id"] != f"mv-plan-{expected_hash[:16]}":
        raise IdentityAnnotationError("sampling_plan plan_id does not match")
    video = payload["video"]
    plan_video = plan["video"]
    for key in ("sha256", "frame_count", "width", "height"):
        if plan_video.get(key) != video[key]:
            raise IdentityAnnotationError(
                f"sampling plan video.{key} does not match Ground Truth"
            )
    if not math.isclose(
        float(plan_video["fps"]), float(video["fps"]), rel_tol=0, abs_tol=1e-6
    ):
        raise IdentityAnnotationError("sampling plan FPS does not match Ground Truth")
    planned_frames = plan.get("planned_frames")
    if not isinstance(planned_frames, list) or not planned_frames:
        raise IdentityAnnotationError("sampling_plan has no planned frames")
    seen: set[int] = set()
    for item in planned_frames:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("frame"), int)
            or isinstance(item.get("frame"), bool)
            or not 1 <= item["frame"] <= video["frame_count"]
            or item["frame"] in seen
            or not isinstance(item.get("reasons"), list)
            or not item["reasons"]
        ):
            raise IdentityAnnotationError("sampling_plan contains an invalid frame")
        seen.add(item["frame"])
    return plan


def _validate_sampling_plan_completion(payload: dict[str, Any]) -> None:
    plan = _validate_embedded_sampling_plan(payload)
    if not plan:
        return
    records = {record["frame"]: record for record in payload["frames"]}
    planned_frames = {item["frame"] for item in plan["planned_frames"]}
    incomplete: list[str] = []
    evaluable: set[int] = set()
    for frame_number in sorted(planned_frames):
        record = records.get(frame_number)
        if record is None:
            incomplete.append(f"missing planned frame {frame_number}")
            continue
        status = record.get("review_status", "pending")
        if status == "pending":
            incomplete.append(f"planned frame {frame_number} is pending")
        elif status == "not_applicable":
            if not str(record.get("review_note", "")).strip():
                incomplete.append(
                    f"not-applicable frame {frame_number} lacks a review_note"
                )
        elif status == "reviewed":
            if not record["exhaustive"]:
                incomplete.append(
                    f"reviewed frame {frame_number} is not exhaustive"
                )
            unresolved = any(
                observation.get("unknown")
                or observation.get("ambiguous")
                or observation.get("ignore")
                for observation in record["observations"]
            )
            if unresolved:
                incomplete.append(
                    f"reviewed frame {frame_number} has unresolved observations"
                )
            elif record["observations"]:
                evaluable.add(frame_number)
        else:
            incomplete.append(f"planned frame {frame_number} has invalid review status")
    unplanned_exhaustive = set(payload["exhaustive_frames"]) - planned_frames
    if unplanned_exhaustive:
        incomplete.append(
            "unplanned exhaustive frames: "
            + ", ".join(str(frame) for frame in sorted(unplanned_exhaustive))
        )
    requirements = plan.get("quality_requirements")
    if not isinstance(requirements, dict):
        raise IdentityAnnotationError("sampling_plan quality_requirements are invalid")
    minimum_evaluable = requirements.get("minimum_evaluable_frames")
    minimum_windows = requirements.get("minimum_dense_windows")
    minimum_window_frames = requirements.get(
        "minimum_evaluable_frames_per_dense_window"
    )
    if not all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for value in (
            minimum_evaluable,
            minimum_windows,
            minimum_window_frames,
        )
    ):
        raise IdentityAnnotationError(
            "sampling_plan quality requirements are invalid"
        )
    if len(evaluable) < minimum_evaluable:
        incomplete.append(
            f"only {len(evaluable)} evaluable frames; "
            f"minimum is {minimum_evaluable}"
        )
    qualifying_windows = 0
    for window in plan.get("dense_windows", []):
        count = sum(
            frame_number in evaluable
            and window["start_frame"] <= frame_number <= window["end_frame"]
            for frame_number in evaluable
        )
        if count >= minimum_window_frames:
            qualifying_windows += 1
    if qualifying_windows < minimum_windows:
        incomplete.append(
            f"only {qualifying_windows} dense windows qualify; "
            f"minimum is {minimum_windows}"
        )
    if incomplete:
        raise IdentityAnnotationError(
            "Ground Truth is incomplete for its sampling protocol: "
            + "; ".join(incomplete)
        )


def load_identity_annotations(path: str | Path) -> dict[str, Any]:
    annotation_path = Path(path).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Identity annotation file not found: {annotation_path}")
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise IdentityAnnotationError(
            f"Invalid identity annotation JSON: {error}"
        ) from error
    validate_identity_annotations(payload)
    return payload


def validate_identity_annotations(payload: Any) -> None:
    """Validate human annotations, entity types, events, and frozen-set integrity."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise IdentityAnnotationError("schema_version must equal 1")
    video = payload.get("video")
    if not isinstance(video, dict):
        raise IdentityAnnotationError("video metadata must be an object")
    for key in ("path", "sha256", "frame_count", "width", "height", "fps"):
        if key not in video:
            raise IdentityAnnotationError(f"video.{key} is required")
    frame_count = video["frame_count"]
    width = video["width"]
    height = video["height"]
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count < 1
    ):
        raise IdentityAnnotationError("video.frame_count must be a positive integer")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise IdentityAnnotationError("video.width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height < 1:
        raise IdentityAnnotationError("video.height must be a positive integer")
    if (
        not isinstance(video["fps"], (int, float))
        or isinstance(video["fps"], bool)
        or not math.isfinite(video["fps"])
        or video["fps"] <= 0
    ):
        raise IdentityAnnotationError("video.fps must be positive")
    if not isinstance(video["path"], str) or not video["path"]:
        raise IdentityAnnotationError("video.path must be nonempty")
    if "record_type" in payload and payload["record_type"] != "human_identity_ground_truth":
        raise IdentityAnnotationError("record_type is not human identity Ground Truth")
    if "annotation_policy" in payload and not isinstance(
        payload["annotation_policy"], str
    ):
        raise IdentityAnnotationError("annotation_policy must be text")
    sha256 = video["sha256"]
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise IdentityAnnotationError("video.sha256 must be a 64-character SHA-256")
    if any(character not in "0123456789abcdef" for character in sha256.lower()):
        raise IdentityAnnotationError("video.sha256 must be hexadecimal")

    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise IdentityAnnotationError("frames must be a list")
    seen_frames: set[int] = set()
    identity_types: dict[str, str] = {}
    seen_event_ids: set[str] = set()
    for frame_record in frames:
        if not isinstance(frame_record, dict):
            raise IdentityAnnotationError("each frames entry must be an object")
        frame_number = frame_record.get("frame")
        if (
            not isinstance(frame_number, int)
            or isinstance(frame_number, bool)
            or frame_number < 1
            or frame_number > frame_count
        ):
            raise IdentityAnnotationError("frame number is outside video bounds")
        if frame_number in seen_frames:
            raise IdentityAnnotationError(f"duplicate frame record: {frame_number}")
        seen_frames.add(frame_number)
        if not isinstance(frame_record.get("exhaustive"), bool):
            raise IdentityAnnotationError(
                f"frames[{frame_number}].exhaustive must be boolean"
            )
        review_status = frame_record.get("review_status", "pending")
        if review_status not in {"pending", "reviewed", "not_applicable"}:
            raise IdentityAnnotationError(
                f"frames[{frame_number}].review_status is invalid"
            )
        review_note = frame_record.get("review_note", "")
        if not isinstance(review_note, str):
            raise IdentityAnnotationError(
                f"frames[{frame_number}].review_note must be text"
            )
        if review_status == "reviewed" and not frame_record["exhaustive"]:
            raise IdentityAnnotationError(
                f"reviewed frame {frame_number} must be exhaustive"
            )
        if review_status == "not_applicable" and not review_note.strip():
            raise IdentityAnnotationError(
                f"not-applicable frame {frame_number} requires a review_note"
            )
        if review_status == "not_applicable" and frame_record.get("observations"):
            raise IdentityAnnotationError(
                f"not-applicable frame {frame_number} cannot contain observations"
            )
        observations = frame_record.get("observations")
        if not isinstance(observations, list):
            raise IdentityAnnotationError(
                f"frames[{frame_number}].observations must be a list"
            )
        annotation_ids: set[str] = set()
        frame_identities: set[str] = set()
        for observation in observations:
            if not isinstance(observation, dict):
                raise IdentityAnnotationError("each observation must be an object")
            annotation_id = observation.get("annotation_id")
            if not isinstance(annotation_id, str) or not annotation_id:
                raise IdentityAnnotationError("observation annotation_id is required")
            if annotation_id in annotation_ids:
                raise IdentityAnnotationError(
                    f"duplicate annotation_id in frame {frame_number}"
                )
            annotation_ids.add(annotation_id)
            entity_type = observation.get("entity_type", "player")
            if not isinstance(entity_type, str) or entity_type not in _ENTITY_TYPES:
                raise IdentityAnnotationError(
                    f"unsupported entity_type in frame {frame_number}"
                )
            identity_id = observation.get("identity_id")
            unknown = observation.get("unknown", False)
            ambiguous = observation.get("ambiguous", False)
            if not isinstance(unknown, bool) or not isinstance(ambiguous, bool):
                raise IdentityAnnotationError(
                    "unknown and ambiguous flags must be booleans"
                )
            ignore = observation.get("ignore", False)
            if not isinstance(ignore, bool):
                raise IdentityAnnotationError("ignore must be boolean")
            if unknown and ambiguous:
                raise IdentityAnnotationError(
                    "an observation cannot be both unknown and ambiguous"
                )
            if unknown and identity_id not in (None, ""):
                raise IdentityAnnotationError(
                    "unknown observations cannot have an identity_id"
                )
            requires_identity = not (
                unknown or ambiguous or ignore
            )
            if requires_identity and (
                not isinstance(identity_id, str) or not identity_id.strip()
            ):
                raise IdentityAnnotationError(
                    "known observations require a nonempty identity_id"
                )
            if (ambiguous or ignore) and identity_id not in (None, ""):
                raise IdentityAnnotationError(
                    "ambiguous or ignored observations cannot have an identity_id"
                )
            identity_status = observation.get("identity_status")
            if identity_status is not None:
                expected_identity_status = (
                    "unknown"
                    if unknown
                    else "pending"
                    if ambiguous
                    else "ignored"
                    if ignore
                    else "confirmed"
                )
                if identity_status != expected_identity_status:
                    raise IdentityAnnotationError(
                        f"identity_status conflicts with flags in frame {frame_number}"
                    )
            if requires_identity:
                previous_type = identity_types.setdefault(identity_id, entity_type)
                if previous_type != entity_type:
                    raise IdentityAnnotationError(
                        f"identity {identity_id} has conflicting entity types"
                    )
                if identity_id in frame_identities:
                    raise IdentityAnnotationError(
                        f"identity {identity_id} appears more than once in frame {frame_number}"
                    )
                frame_identities.add(identity_id)
            coordinates = [observation.get(key) for key in ("x1", "y1", "x2", "y2")]
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(value)
                for value in coordinates
            ):
                raise IdentityAnnotationError("box coordinates must be finite numbers")
            x1, y1, x2, y2 = coordinates
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise IdentityAnnotationError(
                    f"observation box outside image bounds in frame {frame_number}"
                )
        events = frame_record.get("events", [])
        if not isinstance(events, list):
            raise IdentityAnnotationError(
                f"frames[{frame_number}].events must be a list"
            )
        event_ids: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                raise IdentityAnnotationError("each trajectory event must be an object")
            event_id = event.get("event_id")
            identity_id = event.get("identity_id")
            event_type = event.get("event_type")
            if not isinstance(event_id, str) or not event_id:
                raise IdentityAnnotationError("trajectory event_id is required")
            if event_id in event_ids:
                raise IdentityAnnotationError(
                    f"duplicate event_id in frame {frame_number}"
                )
            if event_id in seen_event_ids:
                raise IdentityAnnotationError(
                    f"duplicate event_id across frames: {event_id}"
                )
            if not isinstance(identity_id, str) or not identity_id.strip():
                raise IdentityAnnotationError(
                    "trajectory events require a human identity_id"
                )
            if (
                not isinstance(event_type, str)
                or event_type not in _TRAJECTORY_EVENT_TYPES
            ):
                raise IdentityAnnotationError(
                    f"unsupported trajectory event_type in frame {frame_number}"
                )
            notes = event.get("notes", "")
            if not isinstance(notes, str):
                raise IdentityAnnotationError("trajectory event notes must be text")
            event_ids.add(event_id)
            seen_event_ids.add(event_id)
    exhaustive_frames = payload.get("exhaustive_frames")
    if not isinstance(exhaustive_frames, list) or not all(
        isinstance(frame, int) and not isinstance(frame, bool)
        for frame in exhaustive_frames
    ):
        raise IdentityAnnotationError("exhaustive_frames must be a list")
    if len(exhaustive_frames) != len(set(exhaustive_frames)):
        raise IdentityAnnotationError("exhaustive_frames contains duplicates")
    for frame_number in exhaustive_frames:
        if frame_number < 1 or frame_number > frame_count:
            raise IdentityAnnotationError(
                f"exhaustive frame {frame_number} is outside video bounds"
            )
        if frame_number not in seen_frames:
            raise IdentityAnnotationError(
                f"exhaustive frame {frame_number} has no frame record"
            )
        record = next(item for item in frames if item["frame"] == frame_number)
        if not record["exhaustive"]:
            raise IdentityAnnotationError(
                f"frame {frame_number} is not marked exhaustive"
            )
    marked_exhaustive = {
        record["frame"] for record in frames if record["exhaustive"]
    }
    if marked_exhaustive != set(exhaustive_frames):
        raise IdentityAnnotationError(
            "exhaustive_frames must list exactly the frames marked exhaustive"
        )
    sampling_plan = payload.get("sampling_plan")
    if sampling_plan is not None:
        _validate_embedded_sampling_plan(payload)
    annotation_history = payload.get("annotation_history", [])
    if not isinstance(annotation_history, list):
        raise IdentityAnnotationError("annotation_history must be a list")
    for item in annotation_history:
        if (
            not isinstance(item, dict)
            or item.get("action") not in {"created", "corrected", "deleted"}
            or not isinstance(item.get("timestamp"), str)
            or not isinstance(item.get("frame"), int)
            or isinstance(item.get("frame"), bool)
            or not 1 <= item["frame"] <= frame_count
            or not isinstance(item.get("annotation_id"), str)
            or not item["annotation_id"]
        ):
            raise IdentityAnnotationError(
                "annotation_history contains an invalid entry"
            )
    dataset = payload.get("dataset")
    if dataset is not None:
        if not isinstance(dataset, dict):
            raise IdentityAnnotationError("dataset metadata must be an object")
        freeze = dataset.get("freeze")
        if freeze is not None:
            if (
                not isinstance(freeze, dict)
                or freeze.get("status") != "frozen"
                or not isinstance(freeze.get("frozen_at"), str)
                or not isinstance(freeze.get("policy"), str)
                or not freeze["policy"].strip()
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(freeze.get("content_sha256", ""))
                )
            ):
                raise IdentityAnnotationError("frozen dataset metadata is invalid")
            expected_hash = ground_truth_content_sha256(payload)
            if freeze["content_sha256"] != expected_hash:
                raise IdentityAnnotationError(
                    "frozen Ground Truth content SHA-256 does not match"
                )
            if sampling_plan is not None:
                _validate_sampling_plan_completion(payload)


def _canonicalize_numbers(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise IdentityAnnotationError(
                "Ground Truth content cannot contain non-finite numbers"
            )
        micro_units = (
            Decimal(str(value))
            .quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
            * 1_000_000
        )
        return str(int(micro_units))
    if isinstance(value, list):
        return [_canonicalize_numbers(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _canonicalize_numbers(item)
            for key, item in value.items()
        }
    return value


def _ground_truth_hash_content(payload: dict[str, Any]) -> dict[str, Any]:
    video = payload["video"]
    content = {
        "schema_version": payload["schema_version"],
        "record_type": payload.get("record_type", "human_identity_ground_truth"),
        "annotation_policy": payload.get("annotation_policy", ""),
        "video": {
            key: video[key]
            for key in ("sha256", "frame_count", "width", "height", "fps")
        },
        "exhaustive_frames": sorted(payload["exhaustive_frames"]),
        "frames": sorted(payload["frames"], key=lambda item: item["frame"]),
    }
    for key in ("sampling_plan", "annotation_history"):
        if key in payload:
            content[key] = payload[key]
    return content


def ground_truth_content_sha256(payload: dict[str, Any]) -> str:
    """Hash annotation content independent of machine-specific source paths."""
    canonical = json.dumps(
        _canonicalize_numbers(_ground_truth_hash_content(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def freeze_identity_annotations(
    payload: dict[str, Any], *, frozen_at: str | None = None
) -> dict[str, Any]:
    """Return a validated, immutable-by-integrity Ground Truth test-set snapshot."""
    validate_identity_annotations(payload)
    if not any(
        frame["exhaustive"]
        and frame["observations"]
        and all(
            not observation.get("unknown", False)
            and not observation.get("ambiguous", False)
            and not observation.get("ignore", False)
            for observation in frame["observations"]
        )
        for frame in payload["frames"]
    ):
        raise IdentityAnnotationError(
            "Ground Truth needs an exhaustive frame with resolved identities"
        )
    if payload.get("sampling_plan") is not None:
        _validate_sampling_plan_completion(payload)
    dataset = payload.get("dataset")
    if isinstance(dataset, dict) and dataset.get("freeze") is not None:
        raise IdentityAnnotationError("Ground Truth is already frozen")
    frozen_payload = deepcopy(payload)
    frozen_payload.setdefault("dataset", {})["freeze"] = {
        "status": "frozen",
        "frozen_at": frozen_at or datetime.now(timezone.utc).isoformat(),
        "content_sha256": ground_truth_content_sha256(frozen_payload),
        "policy": (
            "Anotar y revisar sin consultar métricas de los modelos; "
            "conservar este conjunto sin modificaciones para la prueba final."
        ),
    }
    validate_identity_annotations(frozen_payload)
    return frozen_payload


def freeze_identity_annotation_file(
    input_path: str | Path, output_path: str | Path
) -> Path:
    """Freeze a validated annotation file without overwriting an existing file."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"El archivo de salida ya existe: {destination}")
    payload = load_identity_annotations(source)
    frozen_payload = freeze_identity_annotations(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="") as stream:
            json.dump(
                frozen_payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, TypeError, ValueError):
        destination.unlink(missing_ok=True)
        raise
    return destination


def freeze_main() -> None:
    parser = argparse.ArgumentParser(
        description="Congela una exportación humana de Ground Truth sin sobrescribirla."
    )
    parser.add_argument("--input", type=Path, required=True, help="JSON humano validado")
    parser.add_argument("--output", type=Path, required=True, help="Nueva ruta de salida")
    args = parser.parse_args()
    output = freeze_identity_annotation_file(args.input, args.output)
    payload = load_identity_annotations(output)
    print(
        f"Conjunto congelado: {output}\n"
        f"SHA-256 del Ground Truth: "
        f"{payload['dataset']['freeze']['content_sha256']}"
    )


def evaluation_frames(payload: dict[str, Any]) -> list[int]:
    """Return only exhaustive frames with no unknown, ambiguous, or ignored boxes."""
    records = {record["frame"]: record for record in payload["frames"]}
    selected: list[int] = []
    for frame_number in sorted(payload["exhaustive_frames"]):
        record = records[frame_number]
        if any(
            observation.get("unknown", False)
            or observation.get("ambiguous", False)
            or observation.get("ignore", False)
            for observation in record["observations"]
        ):
            continue
        selected.append(frame_number)
    return selected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mot_annotation(
    frame: int,
    track_id: int,
    box: tuple[float, float, float, float],
    confidence: float,
) -> MOTAnnotation:
    x1, y1, x2, y2 = box
    return MOTAnnotation(
        frame=frame,
        track_id=track_id,
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        confidence=confidence,
    )


def _write_mot(path: Path, rows: list[MOTAnnotation]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        for row in rows:
            stream.write(
                f"{row.frame},{row.track_id},{row.x:.6f},{row.y:.6f},"
                f"{row.width:.6f},{row.height:.6f},{row.confidence:.6f},-1,-1,-1\n"
            )


def _identity_association_errors(
    ground_truth: list[MOTAnnotation],
    predictions: list[MOTAnnotation],
    frame_count: int,
    iou_threshold: float,
) -> dict[str, Any]:
    gt_by_frame: dict[int, list[MOTAnnotation]] = defaultdict(list)
    pred_by_frame: dict[int, list[MOTAnnotation]] = defaultdict(list)
    for item in ground_truth:
        gt_by_frame[item.frame].append(item)
    for item in predictions:
        pred_by_frame[item.frame].append(item)

    predicted_to_gt: dict[int, set[int]] = defaultdict(set)
    gt_to_predicted: dict[int, set[int]] = defaultdict(set)
    for frame_number in range(1, frame_count + 1):
        gt_items = gt_by_frame[frame_number]
        pred_items = pred_by_frame[frame_number]
        if not gt_items or not pred_items:
            continue
        similarities = np.zeros((len(gt_items), len(pred_items)), dtype=np.float64)
        for row, gt in enumerate(gt_items):
            gx2, gy2 = gt.x + gt.width, gt.y + gt.height
            for column, pred in enumerate(pred_items):
                px2, py2 = pred.x + pred.width, pred.y + pred.height
                iw = max(0.0, min(gx2, px2) - max(gt.x, pred.x))
                ih = max(0.0, min(gy2, py2) - max(gt.y, pred.y))
                intersection = iw * ih
                union = gt.width * gt.height + pred.width * pred.height - intersection
                similarities[row, column] = intersection / union if union > 0 else 0
        gt_indices, pred_indices = linear_sum_assignment(1.0 - similarities)
        for gt_index, pred_index in zip(gt_indices, pred_indices, strict=True):
            if similarities[gt_index, pred_index] < iou_threshold:
                continue
            gt_id = gt_items[gt_index].track_id
            pred_id = pred_items[pred_index].track_id
            predicted_to_gt[pred_id].add(gt_id)
            gt_to_predicted[gt_id].add(pred_id)
    false_merges = {
        str(pred_id): sorted(gt_ids)
        for pred_id, gt_ids in sorted(predicted_to_gt.items())
        if len(gt_ids) > 1
    }
    identity_splits = {
        str(gt_id): sorted(pred_ids)
        for gt_id, pred_ids in sorted(gt_to_predicted.items())
        if len(pred_ids) > 1
    }
    return {
        "false_association_merge_events": sum(
            len(gt_ids) - 1 for gt_ids in false_merges.values()
        ),
        "ground_truth_identities_never_matched": sorted(
            {item.track_id for item in ground_truth} - gt_to_predicted.keys()
        ),
        "predicted_identities_matched_to_multiple_ground_truth_ids": false_merges,
        "ground_truth_identities_split_across_prediction_ids": identity_splits,
        "definition": (
            "Counts are based only on framewise IoU-matched observations at the "
            "configured threshold; unmatched objects are covered by IDFP/IDFN."
        ),
    }


def evaluate_identity_predictions(
    annotation_path: str | Path,
    prediction_paths: dict[str, str | Path],
    *,
    video_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare prediction files on the same exhaustively labeled frames."""
    if not prediction_paths:
        raise ValueError("At least one named prediction file is required")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in prediction_paths):
        raise ValueError("Prediction names may contain only letters, digits, '_' or '-'")
    payload = load_identity_annotations(annotation_path)
    frames = evaluation_frames(payload)
    if not frames:
        raise IdentityAnnotationError(
            "No eligible exhaustive frames: label complete frames and resolve "
            "unknown/ambiguous/ignored observations before evaluation."
        )
    records = {record["frame"]: record for record in payload["frames"]}
    gt_identity = {
        identity: index + 1
        for index, identity in enumerate(
            sorted(
                {
                    observation["identity_id"]
                    for frame_number in frames
                    for observation in records[frame_number]["observations"]
                }
            )
        )
    }
    if not gt_identity:
        raise IdentityAnnotationError(
            "Eligible frames contain no known identity observations."
        )
    original_to_eval = {
        frame_number: index + 1 for index, frame_number in enumerate(frames)
    }
    gt_rows: list[MOTAnnotation] = []
    for frame_number in frames:
        for observation in records[frame_number]["observations"]:
            gt_rows.append(
                _mot_annotation(
                    original_to_eval[frame_number],
                    gt_identity[observation["identity_id"]],
                    (
                        float(observation["x1"]),
                        float(observation["y1"]),
                        float(observation["x2"]),
                        float(observation["y2"]),
                    ),
                    1.0,
                )
            )

    input_video = Path(video_path).expanduser().resolve()
    if not input_video.is_file():
        raise FileNotFoundError(f"Evaluation video not found: {input_video}")
    video_hash_matches = _sha256_file(input_video) == payload["video"]["sha256"]
    if not video_hash_matches:
        raise IdentityAnnotationError(
            "The annotation video SHA-256 does not match the supplied video."
        )

    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="matchvision-identity-eval-") as temp:
        temporary_dir = Path(temp)
        gt_path = temporary_dir / "ground_truth.txt"
        _write_mot(gt_path, gt_rows)
        for name, prediction_value in prediction_paths.items():
            prediction_path = Path(prediction_value).expanduser().resolve()
            source_rows = load_mot_file(
                prediction_path, frame_count=int(payload["video"]["frame_count"])
            )
            rows_by_eval_frame: list[MOTAnnotation] = []
            for row in source_rows:
                mapped_frame = original_to_eval.get(row.frame)
                if mapped_frame is not None:
                    rows_by_eval_frame.append(
                        MOTAnnotation(
                            frame=mapped_frame,
                            track_id=row.track_id,
                            x=row.x,
                            y=row.y,
                            width=row.width,
                            height=row.height,
                            confidence=row.confidence,
                        )
                    )
            pred_path = temporary_dir / f"{name}.txt"
            _write_mot(pred_path, rows_by_eval_frame)
            metrics = evaluate_mot_sequence(
                gt_path, pred_path, frame_count=len(frames)
            )
            metrics["evaluated_frames"] = len(frames)
            metrics["source_prediction_sha256"] = _sha256_file(prediction_path)
            metrics["identity_association_errors"] = _identity_association_errors(
                gt_rows, rows_by_eval_frame, len(frames), iou_threshold
            )
            results[name] = metrics

    return {
        "schema_version": 1,
        "annotation_path": str(Path(annotation_path).expanduser().resolve()),
        "annotation_sha256": _sha256_file(
            Path(annotation_path).expanduser().resolve()
        ),
        "video": payload["video"],
        "video_sha256_verified": video_hash_matches,
        "video_path_verified": str(input_video),
        "eligible_exhaustive_frames": frames,
        "excluded_exhaustive_frames": sorted(
            set(payload["exhaustive_frames"]) - set(frames)
        ),
        "iou_threshold": iou_threshold,
        "metric_implementation": "TrackEval at pinned project dependency commit",
        "metrics_scope": (
            "Only exhaustive frames without unknown, ambiguous, or ignored boxes; "
            "predictions are filtered to these identical frames."
        ),
        "results": results,
    }


def evaluate_identity_predictions_with_id_metrics(
    annotation_path: str | Path,
    prediction_paths: dict[str, str | Path],
    *,
    video_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate identity runs and return TrackEval IDP/IDR alongside base metrics."""
    # Kept separate as an explicit public entry point for the scientific workflow.
    # TrackEval's published Identity metric is used by evaluate_mot_sequence.
    return evaluate_identity_predictions(
        annotation_path,
        prediction_paths,
        video_path=video_path,
        iou_threshold=iou_threshold,
    )
