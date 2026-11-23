"""Reproducible sampling plans and quality checks for human Ground Truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from matchvision.evaluation.identity_annotations import (
    IdentityAnnotationError,
    evaluation_frames,
    load_identity_annotations,
    validate_identity_annotations,
)

PROTOCOL_ID = "matchvision-football-ground-truth"
PROTOCOL_VERSION = "1.0"
HARD_CASE_TYPES = {
    "crossing",
    "partial_occlusion",
    "complete_occlusion",
    "exit_reentry",
    "scene_change",
    "zoom_change",
    "similar_uniforms",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_hash_content(plan: dict[str, Any]) -> dict[str, Any]:
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


def validate_sampling_plan(plan: Any) -> None:
    if not isinstance(plan, dict):
        raise IdentityAnnotationError("El plan de muestreo debe ser un objeto JSON.")
    if (
        plan.get("schema_version") != 1
        or plan.get("record_type") != "matchvision_annotation_sampling_plan"
        or not isinstance(plan.get("dataset_id"), str)
        or not isinstance(plan.get("protocol"), dict)
        or plan["protocol"].get("protocol_id") != PROTOCOL_ID
        or plan["protocol"].get("version") != PROTOCOL_VERSION
    ):
        raise IdentityAnnotationError("Tipo o versión del plan de anotación no válidos.")
    video = plan.get("video")
    if not isinstance(video, dict):
        raise IdentityAnnotationError("El plan no incluye metadatos de video.")
    for key in ("path", "sha256", "frame_count", "width", "height", "fps"):
        if key not in video:
            raise IdentityAnnotationError(f"El plan no incluye video.{key}.")
    frame_count = video["frame_count"]
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or frame_count < 1
    ):
        raise IdentityAnnotationError("El plan tiene un número de fotogramas inválido.")
    if not isinstance(video["sha256"], str) or len(video["sha256"]) != 64:
        raise IdentityAnnotationError("El plan tiene un SHA-256 de video inválido.")
    if not all(
        isinstance(video[key], int)
        and not isinstance(video[key], bool)
        and video[key] > 0
        for key in ("width", "height")
    ):
        raise IdentityAnnotationError("El plan tiene dimensiones de video inválidas.")
    if (
        not isinstance(video["fps"], (int, float))
        or isinstance(video["fps"], bool)
        or not math.isfinite(video["fps"])
        or video["fps"] <= 0
    ):
        raise IdentityAnnotationError("El plan tiene un FPS inválido.")
    planned_frames = plan.get("planned_frames")
    if not isinstance(planned_frames, list) or not planned_frames:
        raise IdentityAnnotationError("El plan no contiene fotogramas seleccionados.")
    seen_frames: set[int] = set()
    for item in planned_frames:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("frame"), int)
            or isinstance(item.get("frame"), bool)
            or not 1 <= item["frame"] <= frame_count
            or item["frame"] in seen_frames
            or not isinstance(item.get("reasons"), list)
            or not item["reasons"]
            or not all(isinstance(reason, str) and reason for reason in item["reasons"])
        ):
            raise IdentityAnnotationError("El plan contiene un fotograma inválido.")
        seen_frames.add(item["frame"])
    hard_cases = plan.get("hard_cases")
    if not isinstance(hard_cases, list):
        raise IdentityAnnotationError("La lista de casos difíciles no es válida.")
    for case in hard_cases:
        if (
            not isinstance(case, dict)
            or case.get("case_type") not in HARD_CASE_TYPES
            or not isinstance(case.get("frame"), int)
            or isinstance(case.get("frame"), bool)
            or not 1 <= case["frame"] <= frame_count
            or not isinstance(case.get("note"), str)
            or not case["note"].strip()
        ):
            raise IdentityAnnotationError("El plan contiene un caso difícil inválido.")
    windows = plan.get("dense_windows")
    if not isinstance(windows, list):
        raise IdentityAnnotationError("La lista de ventanas densas no es válida.")
    for window in windows:
        if (
            not isinstance(window, dict)
            or not isinstance(window.get("window_id"), str)
            or not isinstance(window.get("start_frame"), int)
            or not isinstance(window.get("end_frame"), int)
            or not 1 <= window["start_frame"] <= window["end_frame"] <= frame_count
        ):
            raise IdentityAnnotationError("El plan contiene una ventana densa inválida.")
    expected_hash = _canonical_sha256(_plan_hash_content(plan))
    if plan.get("plan_sha256") != expected_hash:
        raise IdentityAnnotationError("El SHA-256 del plan de muestreo no coincide.")
    if plan.get("plan_id") != f"mv-plan-{expected_hash[:16]}":
        raise IdentityAnnotationError("El identificador del plan no coincide con su hash.")


def create_annotation_sampling_plan(
    video_path: str | Path,
    *,
    interval_frames: int = 30,
    dense_window_seconds: float = 2.0,
    dense_window_count: int = 3,
    hard_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a deterministic, video-bound frame schedule without annotations."""
    if (
        not isinstance(interval_frames, int)
        or isinstance(interval_frames, bool)
        or interval_frames < 1
    ):
        raise ValueError("interval_frames debe ser un entero positivo.")
    if (
        not isinstance(dense_window_seconds, (int, float))
        or isinstance(dense_window_seconds, bool)
        or not math.isfinite(dense_window_seconds)
        or dense_window_seconds <= 0
    ):
        raise ValueError("dense_window_seconds debe ser positivo.")
    if (
        not isinstance(dense_window_count, int)
        or isinstance(dense_window_count, bool)
        or dense_window_count < 1
    ):
        raise ValueError("dense_window_count debe ser un entero positivo.")

    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"No existe el video: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"No se pudo abrir el video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_frames = 0
    try:
        while True:
            success, _ = capture.read()
            if not success:
                break
            decoded_frames += 1
    finally:
        capture.release()
    if (
        width < 1
        or height < 1
        or not math.isfinite(fps)
        or fps <= 0
        or decoded_frames < 1
    ):
        raise ValueError(f"El video reporta metadatos decodificables inválidos: {source}")

    frame_reasons: dict[int, set[str]] = {}

    def add_frame(frame_number: int, reason: str) -> None:
        frame_reasons.setdefault(frame_number, set()).add(reason)

    temporal_frames = list(range(1, decoded_frames + 1, interval_frames))
    if temporal_frames[-1] != decoded_frames:
        temporal_frames.append(decoded_frames)
    for frame_number in temporal_frames:
        add_frame(frame_number, "muestreo_temporal")

    window_length = min(
        decoded_frames, max(1, int(round(dense_window_seconds * fps)))
    )
    fractions = (
        [0.5]
        if dense_window_count == 1
        else [
            0.1 + 0.8 * index / (dense_window_count - 1)
            for index in range(dense_window_count)
        ]
    )
    windows: list[dict[str, Any]] = []
    seen_ranges: set[tuple[int, int]] = set()
    for index, fraction in enumerate(fractions, start=1):
        center = 1 + round((decoded_frames - 1) * fraction)
        start = max(
            1,
            min(decoded_frames - window_length + 1, center - window_length // 2),
        )
        end = start + window_length - 1
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))
        window_id = f"dense-{len(windows) + 1:02d}"
        windows.append(
            {
                "window_id": window_id,
                "start_frame": start,
                "end_frame": end,
                "location": (
                    "inicio"
                    if fraction < 0.34
                    else "final"
                    if fraction > 0.66
                    else "tramo_central"
                ),
                "purpose": (
                    "Anotar cada fotograma consecutivo para evaluar trayectorias "
                    "y transiciones de identidad dentro de esta ventana."
                ),
            }
        )
        for frame_number in range(start, end + 1):
            add_frame(frame_number, f"ventana_densa:{window_id}")

    validated_hard_cases: list[dict[str, Any]] = []
    for case in hard_cases or []:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("frame"), int)
            or isinstance(case.get("frame"), bool)
            or not 1 <= case["frame"] <= decoded_frames
            or case.get("case_type") not in HARD_CASE_TYPES
            or not isinstance(case.get("note"), str)
            or not case["note"].strip()
        ):
            raise ValueError(
                "Cada caso difícil requiere frame válido, case_type permitido y nota humana."
            )
        normalized = {
            "frame": case["frame"],
            "case_type": case["case_type"],
            "note": case["note"].strip(),
        }
        if normalized in validated_hard_cases:
            raise ValueError("El plan repite un mismo caso difícil.")
        validated_hard_cases.append(normalized)
        add_frame(normalized["frame"], "caso_dificil_humano")
    validated_hard_cases.sort(
        key=lambda item: (item["frame"], item["case_type"], item["note"])
    )

    planned_frames = [
        {"frame": frame, "reasons": sorted(reasons)}
        for frame, reasons in sorted(frame_reasons.items())
    ]
    video_sha256 = _sha256_file(source)
    dataset_id = f"matchvision-{video_sha256[:12]}"
    plan: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "matchvision_annotation_sampling_plan",
        "dataset_id": dataset_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "path": str(video_path),
            "sha256": video_sha256,
            "frame_count": decoded_frames,
            "reported_frame_count": reported_frames,
            "width": width,
            "height": height,
            "fps": fps,
        },
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "version": PROTOCOL_VERSION,
            "annotations_are_human_ground_truth": True,
            "automatic_predictions_are_labels": False,
        },
        "sampling": {
            "interval_frames": interval_frames,
            "approximate_interval_seconds": interval_frames / fps,
            "dense_window_seconds": dense_window_seconds,
            "dense_window_frames": window_length,
            "dense_window_count_requested": dense_window_count,
            "temporal_frames": temporal_frames,
        },
        "dense_windows": windows,
        "hard_cases": validated_hard_cases,
        "planned_frames": planned_frames,
        "quality_requirements": {
            "minimum_evaluable_frames": min(90, len(planned_frames)),
            "minimum_dense_windows": min(2, len(windows)),
            "minimum_evaluable_frames_per_dense_window": min(30, window_length),
        },
        "selection_policy": (
            "Muestreo determinista: fotogramas temporales cada N frames, "
            "fotogramas consecutivos en ventanas distribuidas al inicio, centro "
            "y final, y casos difíciles seleccionados manualmente. El plan no "
            "contiene anotaciones ni predicciones."
        ),
    }
    plan_hash = _canonical_sha256(_plan_hash_content(plan))
    plan["plan_sha256"] = plan_hash
    plan["plan_id"] = f"mv-plan-{plan_hash[:16]}"
    validate_sampling_plan(plan)
    return plan


def _quality_issue(
    severity: str, code: str, message: str, *, frame: int | None = None
) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if frame is not None:
        issue["frame"] = frame
    return issue


def quality_control_annotations(
    payload: dict[str, Any],
    *,
    sampling_plan: dict[str, Any] | None = None,
    validate_schema: bool = True,
) -> dict[str, Any]:
    """Report annotation completeness, consistency and evaluation readiness."""
    issues: list[dict[str, Any]] = []
    if validate_schema:
        try:
            validate_identity_annotations(payload)
        except IdentityAnnotationError as error:
            issues.append(
                _quality_issue("bloqueante", "ANNOTATION_SCHEMA_INVALID", str(error))
            )
            return {
                "schema_version": 1,
                "record_type": "matchvision_ground_truth_quality_report",
                "ready": False,
                "issues": issues,
                "summary": {},
            }

    plan = sampling_plan or payload.get("sampling_plan")
    plan_frames: set[int] = set()
    if plan is not None:
        try:
            validate_sampling_plan(plan)
        except IdentityAnnotationError as error:
            issues.append(
                _quality_issue("bloqueante", "SAMPLING_PLAN_INVALID", str(error))
            )
        else:
            plan_frames = {item["frame"] for item in plan["planned_frames"]}
            video = payload["video"]
            plan_video = plan["video"]
            for key in ("sha256", "frame_count", "width", "height"):
                if video[key] != plan_video[key]:
                    issues.append(
                        _quality_issue(
                            "bloqueante",
                            "VIDEO_PLAN_MISMATCH",
                            f"El metadato video.{key} no coincide con el plan.",
                        )
                    )
            if not math.isclose(
                float(video["fps"]), float(plan_video["fps"]), rel_tol=0, abs_tol=1e-6
            ):
                issues.append(
                    _quality_issue(
                        "bloqueante",
                        "VIDEO_PLAN_MISMATCH",
                        "El FPS del video no coincide con el plan.",
                    )
                )
            if payload.get("sampling_plan") is not None and (
                payload["sampling_plan"].get("plan_id") != plan["plan_id"]
            ):
                issues.append(
                    _quality_issue(
                        "bloqueante",
                        "EMBEDDED_PLAN_MISMATCH",
                        "El plan indicado para el QC no coincide con el plan del Ground Truth.",
                    )
                )
    elif payload.get("dataset", {}).get("freeze") is not None:
        issues.append(
            _quality_issue(
                "bloqueante",
                "MISSING_SAMPLING_PLAN",
                "El conjunto congelado no conserva el protocolo de muestreo.",
            )
        )
    else:
        issues.append(
            _quality_issue(
                "advertencia",
                "NO_SAMPLING_PLAN",
                "Ground Truth legacy sin plan reproducible; se conserva compatibilidad v1.",
            )
        )

    records = {record["frame"]: record for record in payload.get("frames", [])}
    planned_status_counts = {
        "pending": 0,
        "reviewed": 0,
        "not_applicable": 0,
    }
    for frame_number in sorted(plan_frames):
        record = records.get(frame_number)
        if record is None:
            planned_status_counts["pending"] += 1
            issues.append(
                _quality_issue(
                    "bloqueante",
                    "PLANNED_FRAME_MISSING",
                    "El fotograma planificado todavía no fue revisado.",
                    frame=frame_number,
                )
            )
            continue
        status = record.get("review_status", "pending")
        if status not in planned_status_counts:
            issues.append(
                _quality_issue(
                    "bloqueante",
                    "FRAME_REVIEW_STATUS_INVALID",
                    "El estado del fotograma no es válido.",
                    frame=frame_number,
                )
            )
            continue
        planned_status_counts[status] += 1
        if status == "pending":
            issues.append(
                _quality_issue(
                    "bloqueante",
                    "PLANNED_FRAME_PENDING",
                    "El fotograma planificado está pendiente de revisión.",
                    frame=frame_number,
                )
            )
            continue
        if status == "not_applicable":
            if not str(record.get("review_note", "")).strip():
                issues.append(
                    _quality_issue(
                        "bloqueante",
                        "NOT_APPLICABLE_REASON_MISSING",
                        "Un fotograma no aplicable requiere una justificación.",
                        frame=frame_number,
                    )
                )
            continue
        if not record.get("exhaustive"):
            issues.append(
                _quality_issue(
                    "bloqueante",
                    "REVIEWED_FRAME_NOT_EXHAUSTIVE",
                    "Un fotograma planificado revisado debe estar anotado como exhaustivo.",
                    frame=frame_number,
                )
            )
        for observation in record.get("observations", []):
            if "entity_type" not in observation or "identity_status" not in observation:
                issues.append(
                    _quality_issue(
                        "bloqueante",
                        "OBSERVATION_CLASSIFICATION_MISSING",
                        "Cada observación planificada debe tener clasificación y estado de identidad explícitos.",
                        frame=frame_number,
                    )
                )
                break

    planned_exhaustive_extra = sorted(
        set(payload.get("exhaustive_frames", [])) - plan_frames
    ) if plan is not None else []
    for frame_number in planned_exhaustive_extra:
        issues.append(
            _quality_issue(
                "bloqueante",
                "UNPLANNED_EXHAUSTIVE_FRAME",
                "El fotograma exhaustivo no está incluido en el plan reproducible.",
                frame=frame_number,
            )
        )

    all_records = payload.get("frames", [])
    annotation_ids: dict[str, int] = {}
    identity_types: dict[str, set[str]] = {}
    identity_observations: dict[str, int] = {}
    identity_status_counts = {
        "confirmed": 0,
        "unknown": 0,
        "pending": 0,
        "ignored": 0,
    }
    duplicate_event_ids: set[str] = set()
    event_identities: set[str] = set()
    identity_events: dict[str, list[tuple[int, str]]] = {}
    for record in all_records:
        frame_number = record["frame"]
        for observation in record.get("observations", []):
            annotation_id = observation["annotation_id"]
            if annotation_id in annotation_ids:
                issues.append(
                    _quality_issue(
                        "bloqueante",
                        "DUPLICATE_ANNOTATION_ID",
                        "El annotation_id se repite en más de un fotograma.",
                        frame=frame_number,
                    )
                )
            annotation_ids[annotation_id] = frame_number
            identity_id = observation.get("identity_id")
            status = observation.get("identity_status")
            if status is None:
                status = (
                    "unknown"
                    if observation.get("unknown")
                    else "pending"
                    if observation.get("ambiguous")
                    else "ignored"
                    if observation.get("ignore")
                    else "confirmed"
                )
            if status in identity_status_counts:
                identity_status_counts[status] += 1
            if identity_id:
                entity_type = observation.get("entity_type", "player")
                identity_types.setdefault(identity_id, set()).add(entity_type)
                identity_observations[identity_id] = (
                    identity_observations.get(identity_id, 0) + 1
                )
        for event in record.get("events", []):
            event_id = event["event_id"]
            if event_id in duplicate_event_ids:
                issues.append(
                    _quality_issue(
                        "bloqueante",
                        "DUPLICATE_EVENT_ID",
                        "El event_id se repite en el Ground Truth.",
                        frame=frame_number,
                    )
                )
            duplicate_event_ids.add(event_id)
            identity_id = event["identity_id"]
            event_identities.add(identity_id)
            identity_events.setdefault(identity_id, []).append(
                (frame_number, event["event_type"])
            )

    conflicting_identities = sorted(
        identity for identity, types in identity_types.items() if len(types) > 1
    )
    for identity in conflicting_identities:
        issues.append(
            _quality_issue(
                "bloqueante",
                "IDENTITY_TYPE_CONFLICT",
                f"La identidad {identity} tiene más de una clasificación de entidad.",
            )
        )
    for identity in sorted(event_identities - set(identity_observations)):
        issues.append(
            _quality_issue(
                "advertencia",
                "EVENT_IDENTITY_UNOBSERVED",
                f"El evento refiere a {identity}, sin observación resuelta en el archivo.",
            )
        )

    for identity, events in identity_events.items():
        occlusion_open = False
        exited = False
        for frame_number, event_type in sorted(events):
            if event_type == "occlusion_start":
                if occlusion_open:
                    issues.append(
                        _quality_issue(
                            "advertencia",
                            "OCCLUSION_START_WITHOUT_END",
                            f"{identity} inicia otra oclusión antes de cerrar la anterior.",
                            frame=frame_number,
                        )
                    )
                occlusion_open = True
            elif event_type == "occlusion_end":
                if not occlusion_open:
                    issues.append(
                        _quality_issue(
                            "advertencia",
                            "OCCLUSION_END_WITHOUT_START",
                            f"{identity} tiene fin de oclusión sin inicio previo.",
                            frame=frame_number,
                        )
                    )
                occlusion_open = False
            elif event_type == "disappearance":
                exited = True
            elif event_type == "appearance":
                if not exited:
                    issues.append(
                        _quality_issue(
                            "advertencia",
                            "APPEARANCE_WITHOUT_DISAPPEARANCE",
                            f"{identity} reaparece sin una desaparición registrada previamente.",
                            frame=frame_number,
                        )
                    )
                exited = False
        if occlusion_open:
            issues.append(
                _quality_issue(
                    "advertencia",
                    "OCCLUSION_NOT_CLOSED",
                    f"La oclusión de {identity} no tiene evento de fin.",
                )
            )

    evaluable = set(evaluation_frames(payload)) & plan_frames if plan else set(
        evaluation_frames(payload)
    )
    dense_window_counts: dict[str, int] = {}
    if plan is not None:
        requirements = plan["quality_requirements"]
        for window in plan["dense_windows"]:
            count = len(
                evaluable
                & set(range(window["start_frame"], window["end_frame"] + 1))
            )
            dense_window_counts[window["window_id"]] = count
        qualifying_windows = sum(
            count >= requirements["minimum_evaluable_frames_per_dense_window"]
            for count in dense_window_counts.values()
        )
        if len(evaluable) < requirements["minimum_evaluable_frames"]:
            issues.append(
                _quality_issue(
                    "bloqueante",
                    "INSUFFICIENT_EVALUABLE_FRAMES",
                    f"Hay {len(evaluable)} fotogramas evaluables; se requieren "
                    f"{requirements['minimum_evaluable_frames']}.",
                )
            )
        if qualifying_windows < requirements["minimum_dense_windows"]:
            issues.append(
                _quality_issue(
                    "bloqueante",
                    "INSUFFICIENT_DENSE_WINDOWS",
                    f"Solo {qualifying_windows} ventanas densas alcanzan el mínimo; "
                    f"se requieren {requirements['minimum_dense_windows']}.",
                )
            )

    unresolved_evaluable = sorted(
        frame_number
        for frame_number in evaluable
        if any(
            observation.get("unknown")
            or observation.get("ambiguous")
            or observation.get("ignore")
            for observation in records[frame_number]["observations"]
        )
    )
    for frame_number in unresolved_evaluable:
        issues.append(
            _quality_issue(
                "advertencia",
                "FRAME_EXCLUDED_UNRESOLVED_IDENTITY",
                "El fotograma fue marcado exhaustivo, pero una observación sin resolver lo excluye de las métricas.",
                frame=frame_number,
            )
        )
    if plan is not None:
        evaluable = {
            frame_number
            for frame_number in evaluable
            if frame_number not in unresolved_evaluable
        }

    status_counts = {
        "pending": sum(
            record.get("review_status", "pending") == "pending"
            for record in all_records
        ),
        "reviewed": sum(
            record.get("review_status") == "reviewed" for record in all_records
        ),
        "not_applicable": sum(
            record.get("review_status") == "not_applicable"
            for record in all_records
        ),
    }
    blocking = [issue for issue in issues if issue["severity"] == "bloqueante"]
    warnings = [issue for issue in issues if issue["severity"] == "advertencia"]
    return {
        "schema_version": 1,
        "record_type": "matchvision_ground_truth_quality_report",
        "ready": not blocking,
        "status": "bloqueado" if blocking else "listo_con_advertencias" if warnings else "listo",
        "video_sha256": payload.get("video", {}).get("sha256"),
        "sampling_plan_id": plan.get("plan_id") if isinstance(plan, dict) else None,
        "issues": issues,
        "summary": {
            "annotated_frames": len(all_records),
            "planned_frames": len(plan_frames),
            "planned_review_status": planned_status_counts,
            "frame_review_status": status_counts,
            "evaluable_frames": len(evaluable),
            "evaluable_dense_frames": dense_window_counts,
            "identity_count": len(identity_observations),
            "identity_observations": identity_observations,
            "identity_status_counts": identity_status_counts,
            "annotation_history_entries": len(payload.get("annotation_history", [])),
            "hard_cases": len(plan.get("hard_cases", [])) if isinstance(plan, dict) else 0,
            "blocking_issues": len(blocking),
            "warnings": len(warnings),
        },
    }


def _parse_hard_case(value: str) -> dict[str, Any]:
    try:
        frame_text, case_type, note = value.split(":", 2)
        frame = int(frame_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use FRAME:TIPO:NOTA, por ejemplo 175:crossing:dos jugadores se cruzan."
        ) from error
    if case_type not in HARD_CASE_TYPES or not note.strip():
        raise argparse.ArgumentTypeError(
            f"Tipo de caso inválido. Use uno de: {', '.join(sorted(HARD_CASE_TYPES))}."
        )
    return {"frame": frame, "case_type": case_type, "note": note.strip()}


def create_plan_main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea un plan reproducible de muestreo, sin generar anotaciones."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-frames", type=int, default=30)
    parser.add_argument("--dense-window-seconds", type=float, default=2.0)
    parser.add_argument("--dense-window-count", type=int, default=3)
    parser.add_argument(
        "--hard-case",
        type=_parse_hard_case,
        action="append",
        default=[],
        metavar="FRAME:TIPO:NOTA",
        help="Caso ya observado por una persona; nunca se infiere del modelo.",
    )
    args = parser.parse_args()
    try:
        plan = create_annotation_sampling_plan(
            args.video,
            interval_frames=args.interval_frames,
            dense_window_seconds=args.dense_window_seconds,
            dense_window_count=args.dense_window_count,
            hard_cases=args.hard_case,
        )
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, IdentityAnnotationError) as error:
        parser.error(str(error))
        return
    print(
        json.dumps(
            {
                "plan": str(output),
                "dataset_id": plan["dataset_id"],
                "plan_id": plan["plan_id"],
                "video_sha256": plan["video"]["sha256"],
                "decoded_frames": plan["video"]["frame_count"],
                "planned_frames": len(plan["planned_frames"]),
                "dense_windows": plan["dense_windows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def quality_main() -> None:
    parser = argparse.ArgumentParser(
        description="Audita integridad, cobertura y consistencia del Ground Truth."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--sampling-plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = load_identity_annotations(args.annotations)
        plan = None
        if args.sampling_plan:
            plan = json.loads(
                args.sampling_plan.expanduser().resolve().read_text(encoding="utf-8")
            )
        report = quality_control_annotations(payload, sampling_plan=plan)
        if args.output:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["ready"]:
            raise SystemExit(2)
    except (OSError, json.JSONDecodeError, IdentityAnnotationError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    create_plan_main()
