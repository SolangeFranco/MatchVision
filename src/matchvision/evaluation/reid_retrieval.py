"""Independent appearance retrieval metrics from human-labeled video crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from matchvision.evaluation.identity_annotations import (
    IdentityAnnotationError,
    evaluation_frames,
    load_identity_annotations,
)
from matchvision.reidentification.appearance import (
    extract_appearance_descriptor,
)
from matchvision.reidentification.osnet import (
    MODEL_NAME,
    MODEL_SHA256,
    OSNetEmbeddingExtractor,
    extract_person_crop,
)

_METHODS = {"hsv_hog", "embedding", "hybrid"}
_EVENT_SCENARIOS = {
    "appearance": "reappearance_after_exit",
    "occlusion_end": "occlusion_recovery",
    "crossing": "crossing_recovery",
    "camera_change": "camera_change_recovery",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _evenly_spaced(
    candidates: list[tuple[int, dict[str, Any]]], count: int
) -> list[tuple[int, dict[str, Any]]]:
    if count <= 0:
        return []
    if count >= len(candidates):
        return candidates
    indices = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
    return [candidates[int(index)] for index in indices]


def _retrieval_metrics(
    query_indices: list[int],
    samples: list[dict[str, Any]],
    score_rows: Any,
    *,
    threshold: float,
    gallery_policy: Any,
    unordered_pairs: bool,
) -> dict[str, Any]:
    rank1_values: list[float] = []
    average_precisions: list[float] = []
    recovered_by_identity: dict[str, bool] = defaultdict(bool)
    true_positive = false_positive = false_negative = true_negative = 0
    evaluated_pairs = positive_pairs = 0
    candidate_queries = 0
    evaluated_queries = 0
    evaluable_identities: set[str] = set()
    for query_index in query_indices:
        query = samples[query_index]
        scores = score_rows(query_index)
        gallery = gallery_policy(query_index)
        if not gallery:
            continue
        candidate_queries += 1
        relevant = np.asarray(
            [
                samples[index]["identity_id"] == query["identity_id"]
                for index in gallery
            ],
            dtype=bool,
        )
        gallery_scores = scores[np.asarray(gallery, dtype=np.int64)]
        if relevant.any():
            ranked_positions = np.argsort(-gallery_scores, kind="stable")
            ranked_relevance = relevant[ranked_positions]
            rank1_values.append(float(ranked_relevance[0]))
            relevant_ranks = np.flatnonzero(ranked_relevance) + 1
            average_precisions.append(
                float(np.mean(np.arange(1, len(relevant_ranks) + 1) / relevant_ranks))
            )
            evaluated_queries += 1
            matching_scores = gallery_scores[relevant]
            if np.max(matching_scores) >= threshold and ranked_relevance[0]:
                recovered_by_identity[query["identity_id"]] = True
        pair_gallery = (
            [index for index in gallery if index > query_index]
            if unordered_pairs
            else gallery
        )
        for candidate_index in pair_gallery:
            score = scores[candidate_index]
            candidate = samples[candidate_index]
            evaluated_pairs += 1
            actual_same = query["identity_id"] == candidate["identity_id"]
            predicted_same = score >= threshold
            if actual_same:
                evaluable_identities.add(query["identity_id"])
                positive_pairs += 1
                if predicted_same:
                    true_positive += 1
                else:
                    false_negative += 1
            elif predicted_same:
                false_positive += 1
            else:
                true_negative += 1

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else 0.0
        if precision is not None and recall is not None
        else None
    )
    not_recovered = sorted(
        identity
        for identity in evaluable_identities
        if not recovered_by_identity[identity]
    )
    return {
        "available": bool(evaluated_queries),
        "queries": evaluated_queries,
        "candidate_queries": candidate_queries,
        "queries_without_positive_gallery": candidate_queries - evaluated_queries,
        "gallery_pairs": evaluated_pairs,
        "positive_identity_pairs": positive_pairs,
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "association_precision": precision,
        "association_recall": recall,
        "association_f1": f1,
        "false_associations": false_positive,
        "identities_not_recovered": not_recovered,
        "rank1": float(np.mean(rank1_values)) if rank1_values else None,
        "mean_average_precision": (
            float(np.mean(average_precisions)) if average_precisions else None
        ),
        "threshold": threshold,
        "score_semantics": (
            "Similarity for ranking and thresholded pairs; it is not a calibrated probability."
        ),
    }


def _similarity_function(
    method: str,
    samples: list[dict[str, Any]],
    *,
    embedding_weight: float,
) -> Any:
    colors = np.stack([sample["appearance"].color for sample in samples])
    shapes = np.stack([sample["appearance"].shape for sample in samples])
    embeddings = (
        np.stack([sample["embedding"] for sample in samples])
        if method in {"embedding", "hybrid"}
        else None
    )

    def score_row(query_index: int) -> np.ndarray:
        query = samples[query_index]
        color_score = np.minimum(colors, colors[query_index]).sum(axis=1)
        shape_score = np.clip(shapes @ shapes[query_index], 0.0, 1.0)
        hsv_score = 0.5 * color_score + 0.5 * shape_score
        if method == "hsv_hog":
            return hsv_score
        if embeddings is None:
            raise RuntimeError("Embedding features were not loaded.")
        embedding_score = np.clip(
            embeddings @ embeddings[query_index], -1.0, 1.0
        )
        if method == "embedding":
            return (embedding_score + 1.0) / 2.0
        return (
            embedding_weight * ((embedding_score + 1.0) / 2.0)
            + (1.0 - embedding_weight) * hsv_score
        )

    return score_row


def evaluate_human_appearance_retrieval(
    annotation_path: str | Path,
    video_path: str | Path,
    *,
    methods: tuple[str, ...] = ("hsv_hog",),
    similarity_threshold: float = 0.82,
    sample_interval: int = 10,
    maximum_observations: int = 500,
    embedding_weights_path: str | Path | None = None,
    embedding_device: str = "auto",
    embedding_weight: float = 0.60,
) -> dict[str, Any]:
    """Measure appearance retrieval on human boxes, independent of detections.

    Gallery candidates come from other exhaustive frames. Rank-1 and mAP are
    image-retrieval metrics; thresholded association counts are computed once
    for each unordered cross-frame pair. This does not measure detector or
    track-association quality.
    """
    if not methods or set(methods) - _METHODS:
        raise ValueError(f"methods must be selected from {sorted(_METHODS)}.")
    if not math.isfinite(similarity_threshold) or not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in [0, 1].")
    if (
        not isinstance(sample_interval, int)
        or isinstance(sample_interval, bool)
        or sample_interval < 1
    ):
        raise ValueError("sample_interval must be positive.")
    if (
        not isinstance(maximum_observations, int)
        or isinstance(maximum_observations, bool)
        or maximum_observations < 1
    ):
        raise ValueError("maximum_observations must be positive.")
    if not math.isfinite(embedding_weight) or not 0 <= embedding_weight <= 1:
        raise ValueError("embedding_weight must be in [0, 1].")

    evaluation_started = time.perf_counter()
    created_at = datetime.now(timezone.utc).isoformat()
    annotations_file = Path(annotation_path).expanduser().resolve()
    input_video = Path(video_path).expanduser().resolve()
    payload = load_identity_annotations(annotations_file)
    freeze = (payload.get("dataset") or {}).get("freeze")
    if not isinstance(freeze, dict):
        raise IdentityAnnotationError(
            "La recuperación ReID requiere Ground Truth congelado."
        )
    if not input_video.is_file():
        raise FileNotFoundError(f"No existe el video: {input_video}")
    video_sha256 = _sha256_file(input_video)
    if video_sha256 != payload["video"]["sha256"]:
        raise IdentityAnnotationError(
            "El video no coincide con el SHA-256 del Ground Truth."
        )
    eligible_frames = set(evaluation_frames(payload))
    records = {record["frame"]: record for record in payload["frames"]}
    event_frames = {
        record["frame"]
        for record in payload["frames"]
        if record["frame"] in eligible_frames
        and any(
            event.get("event_type") in _EVENT_SCENARIOS
            for event in record.get("events", [])
        )
    }
    sampled_frames = {
        frame
        for frame in eligible_frames
        if (frame - 1) % sample_interval == 0
    } | event_frames
    event_identities = {
        (record["frame"], event.get("identity_id"))
        for record in payload["frames"]
        if record["frame"] in eligible_frames
        for event in record.get("events", [])
        if event.get("event_type") in _EVENT_SCENARIOS
    }
    candidates_by_identity: dict[
        str, list[tuple[int, dict[str, Any]]]
    ] = defaultdict(list)
    for frame_number in sorted(sampled_frames):
        for observation in records[frame_number]["observations"]:
            if observation.get("entity_type", "player") not in {"player", "goalkeeper"}:
                continue
            candidates_by_identity[observation["identity_id"]].append(
                (frame_number, observation)
            )
    candidate_observation_count = sum(
        len(candidates) for candidates in candidates_by_identity.values()
    )
    identity_order = sorted(candidates_by_identity)
    if not identity_order:
        raise IdentityAnnotationError(
            "No hay observaciones humanas de jugadores en fotogramas elegibles."
        )
    if maximum_observations < len(identity_order):
        raise ValueError(
            "maximum_observations debe permitir al menos una aparición por identidad."
        )
    base_quota, remainder = divmod(maximum_observations, len(identity_order))
    selected_observations: list[tuple[int, dict[str, Any]]] = []
    for position, identity_id in enumerate(identity_order):
        quota = base_quota + int(position < remainder)
        candidates = candidates_by_identity[identity_id]
        event_candidates = [
            item for item in candidates if (item[0], identity_id) in event_identities
        ]
        other_candidates = [
            item for item in candidates if (item[0], identity_id) not in event_identities
        ]
        selected_events = _evenly_spaced(
            event_candidates, min(quota, len(event_candidates))
        )
        selected_others = _evenly_spaced(
            other_candidates, quota - len(selected_events)
        )
        selected_observations.extend(selected_events + selected_others)
    selected_observations.sort(
        key=lambda item: (
            item[0],
            item[1]["identity_id"],
            item[1]["annotation_id"],
        )
    )
    selected_frames = sorted({frame for frame, _ in selected_observations})
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"No se pudo abrir el video: {input_video}")
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != (
        int(payload["video"]["width"]),
        int(payload["video"]["height"]),
    ):
        capture.release()
        raise IdentityAnnotationError(
            "Las dimensiones del video no coinciden con el Ground Truth."
        )

    sampling_started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    sampled_indices: dict[tuple[int, str], list[int]] = defaultdict(list)
    try:
        current_frame = 0
        selected_index = 0
        for frame_number in selected_frames:
            while current_frame < frame_number:
                success, frame = capture.read()
                if not success:
                    raise ValueError(
                        f"El video termina antes del fotograma {frame_number}."
                    )
                current_frame += 1
            if current_frame != frame_number:
                raise RuntimeError("La lectura de fotogramas perdió sincronización.")
            while (
                selected_index < len(selected_observations)
                and selected_observations[selected_index][0] == frame_number
            ):
                _, observation = selected_observations[selected_index]
                selected_index += 1
                entity_type = observation.get("entity_type", "player")
                identity_id = observation["identity_id"]
                box = tuple(
                    float(observation[key])
                    for key in ("x1", "y1", "x2", "y2")
                )
                appearance = extract_appearance_descriptor(frame, box)
                crop = extract_person_crop(frame, box)
                if appearance is None or crop is None:
                    continue
                sample_index = len(samples)
                samples.append(
                    {
                        "frame": frame_number,
                        "identity_id": identity_id,
                        "entity_type": entity_type,
                        "appearance": appearance,
                        "crop": crop,
                    }
                )
                sampled_indices[(frame_number, identity_id)].append(sample_index)
    finally:
        capture.release()
    sampling_seconds = time.perf_counter() - sampling_started

    identities = {sample["identity_id"] for sample in samples}
    if not samples or len(identities) < 2:
        raise IdentityAnnotationError(
            "Se requieren apariciones válidas de al menos dos identidades de jugadores."
        )
    embedding_requested = bool(set(methods) & {"embedding", "hybrid"})
    embedding_unavailable_reason = None
    embedding_seconds = None
    resolved_embedding_device = None
    if embedding_requested and embedding_weights_path is None:
        embedding_unavailable_reason = (
            "Indique los pesos OSNet verificados para calcular embeddings."
        )
    elif embedding_requested:
        embedding_started = time.perf_counter()
        extractor = OSNetEmbeddingExtractor(
            embedding_weights_path, device=embedding_device
        )
        resolved_embedding_device = extractor.device
        embeddings = extractor.encode([sample["crop"] for sample in samples])
        for sample, embedding in zip(samples, embeddings, strict=True):
            sample["embedding"] = embedding
        embedding_seconds = time.perf_counter() - embedding_started

    methods_report: dict[str, Any] = {}
    for method in methods:
        method_started = time.perf_counter()
        if method in {"embedding", "hybrid"} and embedding_unavailable_reason:
            methods_report[method] = {
                "available": False,
                "reason": embedding_unavailable_reason,
                "processing_seconds": None,
            }
            continue
        score_row = _similarity_function(
            method, samples, embedding_weight=embedding_weight
        )
        all_indices = list(range(len(samples)))
        global_metrics = _retrieval_metrics(
            all_indices,
            samples,
            score_row,
            threshold=similarity_threshold,
            gallery_policy=lambda query: [
                index
                for index, sample in enumerate(samples)
                if index != query and sample["frame"] != samples[query]["frame"]
            ],
            unordered_pairs=True,
        )
        scenario_queries: dict[str, list[int]] = defaultdict(list)
        for record in payload["frames"]:
            frame_number = record["frame"]
            if frame_number not in eligible_frames:
                continue
            for event in record.get("events", []):
                scenario = _EVENT_SCENARIOS.get(event.get("event_type"))
                if scenario is None:
                    continue
                key = (frame_number, event.get("identity_id"))
                for sample_index in sampled_indices.get(key, []):
                    if any(
                        sample["identity_id"] == event["identity_id"]
                        and sample["frame"] < frame_number
                        for sample in samples
                    ):
                        scenario_queries[scenario].append(sample_index)
        scenarios: dict[str, Any] = {}
        scenario_to_event = {
            scenario: event_type
            for event_type, scenario in _EVENT_SCENARIOS.items()
        }
        for scenario, event_type in scenario_to_event.items():
            queries = sorted(set(scenario_queries[scenario]))
            scenarios[scenario] = _retrieval_metrics(
                queries,
                samples,
                score_row,
                threshold=similarity_threshold,
                gallery_policy=lambda query: [
                    index
                    for index, sample in enumerate(samples)
                    if sample["frame"] < samples[query]["frame"]
                ],
                unordered_pairs=False,
            )
            scenarios[scenario]["event_type"] = event_type
            if not queries:
                scenarios[scenario]["available"] = False
                scenarios[scenario]["reason"] = (
                    "No hay eventos humanos de este tipo con una aparición "
                    "anterior de la misma identidad y una caja consultable."
                )
        methods_report[method] = {
            "available": global_metrics["available"],
            "global_retrieval": global_metrics,
            "event_scenarios": scenarios,
            "configuration": {
                "similarity_threshold": similarity_threshold,
                "sample_interval_frames": sample_interval,
                "maximum_observations": maximum_observations,
                "embedding_weight": (
                    embedding_weight if method == "hybrid" else None
                ),
            },
            "processing_seconds": time.perf_counter() - method_started,
        }

    return {
        "schema_version": 1,
        "record_type": "matchvision_human_box_appearance_retrieval",
        "evaluation_id": str(uuid.uuid4()),
        "created_at": created_at,
        "annotation_path": str(annotations_file),
        "annotation_sha256": _sha256_file(annotations_file),
        "ground_truth_content_sha256": freeze["content_sha256"],
        "video_sha256": video_sha256,
        "video_sha256_verified": True,
        "eligible_exhaustive_frames": sorted(eligible_frames),
        "candidate_sample_frames": sorted(sampled_frames),
        "sampled_frames": selected_frames,
        "eligible_appearance_observations": candidate_observation_count,
        "sampled_observations": len(samples),
        "observations_excluded_by_maximum": (
            candidate_observation_count - len(selected_observations)
        ),
        "excluded_small_or_invalid_crops": len(selected_observations) - len(samples),
        "identity_count": len(identities),
        "evaluated_entity_types": ["player", "goalkeeper"],
        "models": {
            "hsv_hog": {
                "name": "MatchVision HSV/HOG",
                "version": "implementación del proyecto",
            },
            "embedding": {
                "name": MODEL_NAME,
                "weights_sha256_expected": MODEL_SHA256,
                "weights_sha256_used": (
                    MODEL_SHA256 if resolved_embedding_device is not None else None
                ),
                "device_requested": embedding_device,
                "device_resolved": resolved_embedding_device,
            },
        },
        "hardware": {
            "operating_system": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
        },
        "software": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "timings": {
            "sampling_seconds": sampling_seconds,
            "embedding_seconds": embedding_seconds,
            "total_seconds": time.perf_counter() - evaluation_started,
        },
        "methods": methods_report,
        "methodology": {
            "query": "Una caja de jugador/portero anotada por persona.",
            "gallery": (
                "Otras apariciones de fotogramas distintos; Rank-1 y mAP se "
                "calculan solo para consultas con al menos una aparición positiva."
            ),
            "pair_metrics": (
                "Todos los pares no ordenados entre fotogramas distintos, "
                "clasificados por identidad humana con el umbral de similitud configurado."
            ),
            "observation_sampling": (
                "Muestreo uniforme por identidad hasta maximum_observations; "
                "los fotogramas con eventos humanos tienen prioridad."
            ),
            "appearance_only": (
                "Mide el descriptor visual sobre cajas humanas; no mide detección, "
                "ByteTrack ni el asignador completo de identidades."
            ),
            "scenario_events": (
                "Las consultas de recuperación usan únicamente eventos registrados "
                "manualmente; camera_change evalúa continuidad visual entre escenas "
                "editadas, no cámaras físicas sincronizadas."
            ),
            "cross_scene_tracker_reidentification": (
                "El asociador actual deshabilita asociaciones entre escenas; los "
                "resultados de camera_change no deben interpretarse como soporte de esa función."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa recuperación visual con apariciones humanas."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(_METHODS),
        default=["hsv_hog"],
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--maximum-observations", type=int, default=500)
    parser.add_argument("--osnet-weights", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-weight", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate_human_appearance_retrieval(
            args.annotations,
            args.video,
            methods=tuple(args.methods),
            similarity_threshold=args.similarity_threshold,
            sample_interval=args.sample_interval,
            maximum_observations=args.maximum_observations,
            embedding_weights_path=args.osnet_weights,
            embedding_device=args.device,
            embedding_weight=args.embedding_weight,
        )
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"No se sobrescribe el informe existente: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
        return
    print(json.dumps({"informe": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
