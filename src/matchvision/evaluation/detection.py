"""Class-aware object-detection metrics against frozen human annotations."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from matchvision.evaluation.identity_annotations import (
    IdentityAnnotationError,
    _ENTITY_TYPES,
    evaluation_frames,
    ground_truth_content_sha256,
    load_identity_annotations,
)


class DetectionEvaluationError(ValueError):
    """Raised when detections cannot be compared safely with Ground Truth."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _load_detection_rows(
    path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    class_map: dict[str, str],
) -> list[dict[str, Any]]:
    try:
        stream = path.open("r", encoding="utf-8-sig", newline="")
    except OSError:
        raise
    with stream:
        reader = csv.DictReader(stream)
        required = {"frame", "x1", "y1", "x2", "y2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DetectionEvaluationError(
                f"El CSV de detecciones no contiene columnas: {sorted(missing)}"
            )
        if "label" not in (reader.fieldnames or []) and "class_name" not in (
            reader.fieldnames or []
        ):
            raise DetectionEvaluationError(
                "El CSV de detecciones requiere la columna label o class_name."
            )
        rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                frame = int(row["frame"])
                box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
                confidence = float(row.get("confidence") or 1.0)
            except (TypeError, ValueError) as error:
                raise DetectionEvaluationError(
                    f"Fila {line_number}: coordenadas, fotograma o confianza no válidos."
                ) from error
            label = (row.get("label") or row.get("class_name") or "").strip()
            if not label:
                raise DetectionEvaluationError(
                    f"Fila {line_number}: falta el nombre de clase."
                )
            if frame < 1 or frame > frame_count:
                raise DetectionEvaluationError(
                    f"Fila {line_number}: el fotograma está fuera del video."
                )
            if not all(math.isfinite(value) for value in (*box, confidence)):
                raise DetectionEvaluationError(
                    f"Fila {line_number}: se encontraron valores no finitos."
                )
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise DetectionEvaluationError(
                    f"Fila {line_number}: el recuadro está fuera de los límites del video."
                )
            target_class = class_map.get(label, label)
            rows.append(
                {
                    "frame": frame,
                    "label": label,
                    "entity_type": target_class,
                    "box": box,
                    "confidence": confidence,
                }
            )
    return rows


def evaluate_detection_predictions(
    annotation_path: str | Path,
    prediction_path: str | Path,
    *,
    video_path: str | Path,
    iou_threshold: float = 0.5,
    class_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate raw detector boxes with one-to-one, class-aware IoU matching.

    Only exhaustive frames whose human annotations are all resolved are used.
    ``class_map`` explicitly maps detector labels to Ground Truth entity types;
    unmapped labels must already equal a supported entity type.
    """
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1].")
    annotations_file = Path(annotation_path).expanduser().resolve()
    predictions_file = Path(prediction_path).expanduser().resolve()
    input_video = Path(video_path).expanduser().resolve()
    payload = load_identity_annotations(annotations_file)
    freeze = (payload.get("dataset") or {}).get("freeze")
    if not isinstance(freeze, dict):
        raise IdentityAnnotationError(
            "La evaluación de detección requiere un Ground Truth congelado."
        )
    if not input_video.is_file():
        raise FileNotFoundError(f"No existe el video de evaluación: {input_video}")
    if _sha256_file(input_video) != payload["video"]["sha256"]:
        raise IdentityAnnotationError(
            "El SHA-256 del video no coincide con el Ground Truth."
        )
    if not predictions_file.is_file():
        raise FileNotFoundError(f"No existe el CSV de detecciones: {predictions_file}")

    selected_map = dict(class_map or {})
    invalid_targets = set(selected_map.values()) - _ENTITY_TYPES
    if invalid_targets:
        raise ValueError(
            f"class_map contiene tipos de anotación desconocidos: {sorted(invalid_targets)}"
        )
    frames = evaluation_frames(payload)
    if not frames:
        raise IdentityAnnotationError(
            "No hay fotogramas exhaustivos aptos para evaluación."
        )
    records = {record["frame"]: record for record in payload["frames"]}
    ground_truth: dict[int, dict[str, list[tuple[float, float, float, float]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for frame_number in frames:
        for observation in records[frame_number]["observations"]:
            entity_type = observation.get("entity_type", "player")
            ground_truth[frame_number][entity_type].append(
                tuple(
                    float(observation[key])
                    for key in ("x1", "y1", "x2", "y2")
                )
            )

    rows = _load_detection_rows(
        predictions_file,
        frame_count=int(payload["video"]["frame_count"]),
        width=int(payload["video"]["width"]),
        height=int(payload["video"]["height"]),
        class_map=selected_map,
    )
    predictions: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["entity_type"] not in _ENTITY_TYPES:
            raise DetectionEvaluationError(
                f"La clase «{row['label']}» requiere un class_map explícito."
            )
        if row["frame"] in frames:
            predictions[row["frame"]][row["entity_type"]].append(row)

    classes = sorted(
        {
            entity_type
            for frame in ground_truth.values()
            for entity_type in frame
        }
        | {
            entity_type
            for frame in predictions.values()
            for entity_type in frame
        }
    )
    counts = {
        entity_type: {"ground_truth": 0, "predicted": 0, "true_positives": 0}
        for entity_type in classes
    }
    matched_ious: dict[str, list[float]] = {
        entity_type: [] for entity_type in classes
    }
    for frame_number in frames:
        frame_classes = set(ground_truth[frame_number]) | set(
            predictions[frame_number]
        )
        for entity_type in frame_classes:
            gt_boxes = ground_truth[frame_number][entity_type]
            pred_rows = predictions[frame_number][entity_type]
            counts[entity_type]["ground_truth"] += len(gt_boxes)
            counts[entity_type]["predicted"] += len(pred_rows)
            if not gt_boxes or not pred_rows:
                continue
            similarities = np.asarray(
                [
                    [_iou(gt_box, row["box"]) for row in pred_rows]
                    for gt_box in gt_boxes
                ],
                dtype=np.float64,
            )
            costs = np.where(similarities >= iou_threshold, 1.0 - similarities, 1e6)
            gt_indices, pred_indices = linear_sum_assignment(costs)
            for gt_index, pred_index in zip(gt_indices, pred_indices, strict=True):
                if similarities[gt_index, pred_index] >= iou_threshold:
                    counts[entity_type]["true_positives"] += 1
                    matched_ious[entity_type].append(
                        float(similarities[gt_index, pred_index])
                    )

    per_class: dict[str, Any] = {}
    totals = {"ground_truth": 0, "predicted": 0, "true_positives": 0}
    for entity_type in classes:
        values = counts[entity_type]
        false_positives = values["predicted"] - values["true_positives"]
        false_negatives = values["ground_truth"] - values["true_positives"]
        precision = _ratio(values["true_positives"], values["predicted"])
        recall = _ratio(values["true_positives"], values["ground_truth"])
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall > 0
            else 0.0
            if precision is not None and recall is not None
            else None
        )
        per_class[entity_type] = {
            **values,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": (
                float(np.mean(matched_ious[entity_type]))
                if matched_ious[entity_type]
                else None
            ),
        }
        for key in totals:
            totals[key] += values[key]

    true_positives = totals["true_positives"]
    false_positives = totals["predicted"] - true_positives
    false_negatives = totals["ground_truth"] - true_positives
    precision = _ratio(true_positives, totals["predicted"])
    recall = _ratio(true_positives, totals["ground_truth"])
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else 0.0
        if precision is not None and recall is not None
        else None
    )
    all_ious = [value for values in matched_ious.values() for value in values]
    return {
        "record_type": "matchvision_detection_metrics",
        "annotation_path": str(annotations_file),
        "annotation_sha256": _sha256_file(annotations_file),
        "ground_truth_content_sha256": freeze["content_sha256"],
        "video_sha256": payload["video"]["sha256"],
        "video_sha256_verified": True,
        "prediction_path": str(predictions_file),
        "prediction_sha256": _sha256_file(predictions_file),
        "eligible_exhaustive_frames": frames,
        "excluded_exhaustive_frames": sorted(
            set(payload["exhaustive_frames"]) - set(frames)
        ),
        "iou_threshold": iou_threshold,
        "matching": (
            "Maximum-cardinality one-to-one Hungarian matching among same-class "
            "pairs with IoU >= the configured threshold."
        ),
        "class_map": selected_map,
        "overall": {
            **totals,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": float(np.mean(all_ious)) if all_ious else None,
        },
        "per_class": per_class,
        "limitations": [
            "Las métricas de clase dependen de la correspondencia explícita class_map.",
            "Los fotogramas exhaustivos con observaciones desconocidas, ambiguas o ignoradas se excluyen completos.",
            "Se evalúan detecciones crudas; no se usan cajas extendidas por el tracker.",
            "La precision es null cuando no hay predicciones y la recall es null cuando no hay objetos Ground Truth.",
        ],
    }
