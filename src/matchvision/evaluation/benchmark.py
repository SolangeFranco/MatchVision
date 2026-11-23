"""Reproducible comparison of detector/tracker/ReID pipeline runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from matchvision.evaluation.detection import evaluate_detection_predictions
from matchvision.evaluation.identity_annotations import (
    IdentityAnnotationError,
    _ENTITY_TYPES,
    evaluate_identity_predictions,
    ground_truth_content_sha256,
    load_identity_annotations,
)
from matchvision.evaluation.reid_retrieval import (
    evaluate_human_appearance_retrieval,
)


_REQUIRED_RUNS = {
    "yolo_bytetrack": None,
    "hsv_hog": "hsv_hog",
    "osnet": "embedding",
    "hybrid": "hybrid",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(path: str | Path, run_name: str) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"No existe el registro de ejecución: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"El registro de ejecución no es JSON válido: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"El registro de ejecución debe ser un objeto: {source}")
    if not isinstance(payload.get("input"), dict) or not isinstance(
        payload.get("outputs"), dict
    ):
        raise ValueError(f"Al registro de ejecución le faltan input/outputs: {source}")
    identity_configuration = (
        payload.get("configuration", {}).get("identity_association")
    )
    actual_method = (
        identity_configuration.get("reid_method")
        if isinstance(identity_configuration, dict)
        else None
    )
    expected_method = _REQUIRED_RUNS[run_name]
    if actual_method != expected_method:
        raise ValueError(
            f"La ejecución {run_name} requiere reid_method={expected_method!r}; "
            f"se encontró {actual_method!r}."
        )
    return payload, source


def _resolve_output(run: dict[str, Any], key: str, run_path: Path) -> Path:
    value = run["outputs"].get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"El registro {run_path} no declara outputs.{key}."
        )
    output = Path(value).expanduser()
    if not output.is_absolute():
        output = run_path.parent / output
    output = output.resolve()
    if not output.is_file():
        raise FileNotFoundError(f"No existe el artefacto {key}: {output}")
    hash_key = f"{key}_sha256"
    expected_hash = run["outputs"].get(hash_key)
    if expected_hash is not None and _sha256_file(output) != expected_hash:
        raise ValueError(f"El hash de outputs.{key} no coincide con su registro.")
    return output


def _hardware_snapshot() -> dict[str, Any]:
    return {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def _metric_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_model_benchmark(
    annotation_path: str | Path,
    video_path: str | Path,
    run_paths: dict[str, str | Path],
    *,
    class_map: dict[str, str],
    iou_threshold: float = 0.5,
    dataset_kind: str = "unknown",
    appearance_retrieval_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare four completed runs against the same frozen annotation file."""
    if set(run_paths) != set(_REQUIRED_RUNS):
        raise ValueError(
            "Se requieren exactamente las ejecuciones yolo_bytetrack, hsv_hog, "
            "osnet e hybrid."
        )
    if dataset_kind not in {"real_match", "synthetic", "unknown"}:
        raise ValueError("dataset_kind debe ser real_match, synthetic o unknown.")
    if not isinstance(class_map, dict) or not class_map:
        raise ValueError("class_map debe asignar las clases del detector.")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in class_map.items()):
        raise ValueError("class_map debe relacionar nombres de clase de texto.")
    invalid_targets = set(class_map.values()) - _ENTITY_TYPES
    if invalid_targets:
        raise ValueError(
            f"class_map contiene tipos de anotación desconocidos: {sorted(invalid_targets)}"
        )
    annotation_file = Path(annotation_path).expanduser().resolve()
    input_video = Path(video_path).expanduser().resolve()
    payload = load_identity_annotations(annotation_file)
    freeze = (payload.get("dataset") or {}).get("freeze")
    if not isinstance(freeze, dict):
        raise IdentityAnnotationError(
            "El benchmark requiere un Ground Truth congelado e íntegro."
        )
    if not input_video.is_file():
        raise FileNotFoundError(f"No existe el video: {input_video}")
    video_hash = _sha256_file(input_video)
    if video_hash != payload["video"]["sha256"]:
        raise IdentityAnnotationError(
            "El video no coincide con el SHA-256 del Ground Truth."
        )
    if appearance_retrieval_report is not None and (
        appearance_retrieval_report.get("video_sha256") != video_hash
        or appearance_retrieval_report.get("ground_truth_content_sha256")
        != ground_truth_content_sha256(payload)
    ):
        raise ValueError(
            "El informe de recuperación debe usar el mismo video y Ground Truth."
        )

    runs: dict[str, dict[str, Any]] = {}
    resolved_paths: dict[str, dict[str, Path]] = {}
    common_detector = None
    common_tracker = None
    common_model = None
    for run_name in _REQUIRED_RUNS:
        run, run_path = _load_run(run_paths[run_name], run_name)
        if run["input"].get("video_sha256") != video_hash:
            raise ValueError(f"La ejecución {run_name} procesó otro video.")
        if run["input"].get("frames_processed") != payload["video"]["frame_count"]:
            raise ValueError(
                f"La ejecución {run_name} procesó un número diferente de fotogramas."
            )
        detector_config = run.get("configuration", {}).get("detector")
        tracker_config = run.get("configuration", {}).get("tracker")
        model = run.get("model")
        if not isinstance(detector_config, dict) or not isinstance(
            tracker_config, dict
        ) or not isinstance(model, dict):
            raise ValueError(f"La ejecución {run_name} no registra modelo/configuración.")
        detector_signature = {
            "configuration": detector_config,
            "model_name": model.get("name"),
            "model_version": model.get("version"),
            "checkpoint_sha256": model.get("checkpoint_sha256"),
        }
        if common_detector is None:
            common_detector = detector_signature
            common_tracker = {
                "configuration": tracker_config,
                "name": run.get("tracker", {}).get("name"),
                "version": run.get("tracker", {}).get("version"),
            }
            common_model = detector_signature
        elif (
            detector_signature != common_detector
            or tracker_config != common_tracker["configuration"]
            or run.get("tracker", {}).get("name") != common_tracker["name"]
            or run.get("tracker", {}).get("version") != common_tracker["version"]
        ):
            raise ValueError(
                "Las cuatro ejecuciones deben usar exactamente el mismo detector "
                "y la misma configuración/version de ByteTrack."
            )
        outputs = {
            "mot": _resolve_output(run, "mot", run_path),
            "detections_csv": _resolve_output(run, "detections_csv", run_path),
            "tracking_observations_csv": _resolve_output(
                run, "tracking_observations_csv", run_path
            ),
        }
        runs[run_name] = run
        resolved_paths[run_name] = outputs

    tracking_report = evaluate_identity_predictions(
        annotation_file,
        {name: paths["mot"] for name, paths in resolved_paths.items()},
        video_path=input_video,
        iou_threshold=iou_threshold,
    )
    per_run: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    for run_name in _REQUIRED_RUNS:
        run = runs[run_name]
        detection = evaluate_detection_predictions(
            annotation_file,
            resolved_paths[run_name]["detections_csv"],
            video_path=input_video,
            iou_threshold=iou_threshold,
            class_map=class_map,
        )
        tracking = tracking_report["results"][run_name]
        retrieval_method = _REQUIRED_RUNS[run_name]
        retrieval = (
            appearance_retrieval_report.get("methods", {}).get(retrieval_method)
            if appearance_retrieval_report is not None and retrieval_method
            else None
        )
        global_retrieval = (
            retrieval.get("global_retrieval", {})
            if isinstance(retrieval, dict)
            else {}
        )
        retrieval_available = bool(global_retrieval.get("available"))
        retrieval_unavailable_reason = (
            retrieval.get("reason")
            if isinstance(retrieval, dict)
            else None
        ) or global_retrieval.get("reason") or (
            "No se ejecutó el protocolo independiente de recuperación visual "
            "de apariciones humanas."
            if retrieval is None
            else "No hay consultas con apariciones positivas evaluables."
        )
        identity_metrics = {
            "precision": tracking.get("ID_Precision"),
            "recall": tracking.get("ID_Recall"),
            "f1": _metric_f1(
                tracking.get("ID_Precision"), tracking.get("ID_Recall")
            ),
            "false_association_detections": tracking.get("ID_false_positives"),
            "unrecovered_identity_detections": tracking.get("ID_false_negatives"),
            "identity_switches": tracking.get("ID_switches"),
            "fragmentation": tracking.get("Fragmentation"),
            "identity_errors": tracking.get("identity_association_errors"),
            "rank1": {
                "available": retrieval_available,
                "value": global_retrieval.get("rank1"),
                "source": "apariencia sobre cajas humanas" if retrieval_available else None,
                "reason": None if retrieval_available else retrieval_unavailable_reason,
            },
            "mean_average_precision": {
                "available": retrieval_available,
                "value": global_retrieval.get("mean_average_precision"),
                "source": "apariencia sobre cajas humanas" if retrieval_available else None,
                "reason": None if retrieval_available else retrieval_unavailable_reason,
            },
        }
        timings = run.get("timings", {})
        per_run[run_name] = {
            "run_id": run.get("run_id"),
            "run_metadata_path": str(Path(run_paths[run_name]).expanduser().resolve()),
            "model": run["model"],
            "tracker": run["tracker"],
            "configuration": run.get("configuration", {}),
            "environment": run.get("environment", {}),
            "timings": timings,
            "artifacts": {
                key: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for key, path in resolved_paths[run_name].items()
            },
            "detection": detection,
            "tracking": tracking,
            "reidentification_association": identity_metrics,
            "appearance_retrieval": retrieval,
        }
        table.append(
            {
                "configuracion": run_name,
                "segundos": timings.get("processing_seconds"),
                "fotogramas_por_segundo": timings.get("processing_fps"),
                "precision_deteccion": detection["overall"]["precision"],
                "recall_deteccion": detection["overall"]["recall"],
                "f1_deteccion": detection["overall"]["f1"],
                "HOTA": tracking["HOTA"],
                "MOTA": tracking["MOTA"],
                "IDF1": tracking["IDF1"],
                "precision_asociacion": identity_metrics["precision"],
                "recall_asociacion": identity_metrics["recall"],
                "f1_asociacion": identity_metrics["f1"],
                "Rank-1": identity_metrics["rank1"]["value"],
                "mAP": identity_metrics["mean_average_precision"]["value"],
            }
        )

    return {
        "schema_version": 1,
        "record_type": "matchvision_independent_model_benchmark",
        "benchmark_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_kind_asserted": dataset_kind,
        "real_match_validation_established": dataset_kind == "real_match",
        "video": {
            "path": str(input_video),
            "sha256": video_hash,
            "frame_count": payload["video"]["frame_count"],
        },
        "ground_truth": {
            "path": str(annotation_file),
            "file_sha256": _sha256_file(annotation_file),
            "content_sha256": ground_truth_content_sha256(payload),
            "frozen_at": freeze["frozen_at"],
            "mutation_policy": "read_only",
        },
        "evaluation": {
            "detection_iou_threshold": iou_threshold,
            "tracking_metric_implementation": (
                "TrackEval at the pinned project dependency commit"
            ),
            "tracking_iou_threshold": 0.5,
            "class_map": class_map,
            "appearance_retrieval_protocol": (
                "Cajas de Ground Truth humano; ranking de descriptores, independiente del tracking."
                if appearance_retrieval_report is not None
                else None
            ),
            "eligible_exhaustive_frames": tracking_report[
                "eligible_exhaustive_frames"
            ],
            "excluded_exhaustive_frames": tracking_report[
                "excluded_exhaustive_frames"
            ],
            "identical_ground_truth_for_all_runs": True,
            "limitations": [
                "Los archivos MOT no contienen etiquetas de clase; la evaluación de tracking es global y la comparación de detección es por clase.",
                "Rank-1 y mAP no se derivan de scores de propuestas condicionadas ni de métricas MOT.",
                "El tipo de conjunto es una declaración del operador y no certifica que las anotaciones sean correctas.",
            ],
        },
        "shared_detector": common_model,
        "shared_tracker": common_tracker,
        "hardware_at_evaluation": _hardware_snapshot(),
        "appearance_retrieval_report": appearance_retrieval_report,
        "results": per_run,
        "comparison_table": table,
        "validation_conclusion": (
            "Benchmark del partido con Ground Truth humano congelado."
            if dataset_kind == "real_match"
            else (
                "Prueba sintética; no constituye validación de un partido real."
                if dataset_kind == "synthetic"
                else "El origen del conjunto no está verificado; no afirmar validación de un partido real."
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara detección, tracking e identidad contra Ground Truth congelado."
    )
    parser.add_argument("--annotations", type=Path, required=True, help="JSON humano congelado")
    parser.add_argument("--video", type=Path, required=True, help="Video fuente")
    parser.add_argument(
        "--class-map",
        type=Path,
        required=True,
        help="JSON que mapea etiquetas del detector a tipos humanos",
    )
    parser.add_argument("--baseline", type=Path, required=True, help="Registro YOLO + ByteTrack")
    parser.add_argument("--hsv-hog", type=Path, required=True, help="Registro con HSV/HOG")
    parser.add_argument("--osnet", type=Path, required=True, help="Registro con OSNet")
    parser.add_argument("--hybrid", type=Path, required=True, help="Registro híbrido")
    parser.add_argument("--output", type=Path, required=True, help="JSON de salida")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--appearance-retrieval",
        action="store_true",
        help="Calcula Rank-1/mAP y asociaciones de descriptores sobre cajas humanas.",
    )
    parser.add_argument(
        "--osnet-weights",
        type=Path,
        help="Pesos OSNet verificados; si faltan, se omiten embedding e híbrido.",
    )
    parser.add_argument("--reid-sample-interval", type=int, default=10)
    parser.add_argument("--reid-maximum-observations", type=int, default=500)
    parser.add_argument("--reid-similarity-threshold", type=float, default=0.82)
    parser.add_argument("--reid-embedding-weight", type=float, default=0.60)
    parser.add_argument(
        "--dataset-kind",
        choices=("real_match", "synthetic", "unknown"),
        default="unknown",
        help="Origen declarado del conjunto; unknown es el valor prudente",
    )
    args = parser.parse_args()
    try:
        class_map = json.loads(args.class_map.read_text(encoding="utf-8"))
        if not isinstance(class_map, dict):
            raise ValueError("--class-map debe contener un objeto JSON.")
        appearance_retrieval_report = (
            evaluate_human_appearance_retrieval(
                args.annotations,
                args.video,
                methods=("hsv_hog", "embedding", "hybrid"),
                similarity_threshold=args.reid_similarity_threshold,
                sample_interval=args.reid_sample_interval,
                maximum_observations=args.reid_maximum_observations,
                embedding_weights_path=args.osnet_weights,
                embedding_weight=args.reid_embedding_weight,
            )
            if args.appearance_retrieval
            else None
        )
        report = build_model_benchmark(
            args.annotations,
            args.video,
            {
                "yolo_bytetrack": args.baseline,
                "hsv_hog": args.hsv_hog,
                "osnet": args.osnet,
                "hybrid": args.hybrid,
            },
            class_map=class_map,
            iou_threshold=args.iou_threshold,
            dataset_kind=args.dataset_kind,
            appearance_retrieval_report=appearance_retrieval_report,
        )
        output = args.output.expanduser().resolve()
        csv_output = output.with_suffix(".csv")
        if output.exists() or csv_output.exists():
            raise FileExistsError("No se sobrescriben informes de benchmark existentes.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        with csv_output.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(report["comparison_table"][0])
            )
            writer.writeheader()
            writer.writerows(report["comparison_table"])
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
        return
    print(
        json.dumps(
            {
                "informe": str(output),
                "tabla": str(csv_output),
                "conclusion": report["validation_conclusion"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
