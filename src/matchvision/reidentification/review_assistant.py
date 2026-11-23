"""Generate visual, non-ground-truth assistance for manual identity review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from matchvision.reidentification.appearance import (
    AppearanceDescriptor,
    PersistentIdentityAssociator,
    compare_appearance_descriptors,
    extract_appearance_descriptor,
)
from matchvision.reidentification.osnet import (
    MODEL_FILENAME,
    OSNetEmbeddingExtractor,
    cosine_similarity,
)

_MAX_TRACK_SAMPLES = 5
_CANDIDATE_GROUP_SCORE_THRESHOLD = 0.82


def generate_review_proposals(
    review_video_path: str | Path,
    manifest_path: str | Path,
    trajectories_path: str | Path,
    cases_path: str | Path,
    output_path: str | Path,
    evidence_dir: str | Path,
    *,
    maximum_gap_frames: int = 150,
    maximum_candidates_per_case: int = 3,
    reid_method: str = "hsv_hog",
    embedding_weights_path: str | Path | None = None,
    embedding_device: str = "auto",
    embedding_batch_size: int = 16,
    embedding_weight: float = 0.60,
) -> dict[str, Any]:
    """Create review aids without assigning or exporting human identities."""
    review_video = Path(review_video_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    trajectories_file = Path(trajectories_path).expanduser().resolve()
    cases_file = Path(cases_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    evidence_root = Path(evidence_dir).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[3]
    try:
        evidence_relative = evidence_root.relative_to(project_root).as_posix()
    except ValueError:
        evidence_relative = evidence_root.name
    evidence_url_prefix = f"/{evidence_relative}"
    if maximum_gap_frames < 1 or maximum_candidates_per_case < 1:
        raise ValueError("Los límites de fotogramas y candidatos deben ser positivos.")
    if reid_method not in {"hsv_hog", "embedding", "hybrid"}:
        raise ValueError("El método ReID debe ser hsv_hog, embedding o hybrid.")
    if reid_method != "hsv_hog" and not embedding_weights_path:
        raise ValueError("Indique los pesos OSNet para el método ReID solicitado.")
    if not 0 <= embedding_weight <= 1:
        raise ValueError("El peso de embeddings debe estar entre 0 y 1.")
    for path in (review_video, manifest_file, trajectories_file, cases_file):
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró el recurso requerido: {path}")
    if output_file.exists() or evidence_root.exists():
        raise FileExistsError(
            "La salida o el directorio de evidencia ya existe; no se sobrescribió nada."
        )

    generation_started = time.perf_counter()
    preparation_started = generation_started
    manifest = _read_json(manifest_file)
    cases_data = _read_json(cases_file)
    trajectories = _read_trajectories(trajectories_file)
    cases = cases_data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("El archivo de casos debe contener una lista no vacía «cases».")
    frame_count = cases_data.get("video_frame_count")
    if not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("El archivo de casos no declara un número válido de fotogramas.")
    if manifest.get("schema_version") != 1:
        raise ValueError("La versión del manifiesto de revisión no es compatible.")
    if manifest.get("frames_written") != frame_count:
        raise ValueError("El manifiesto y los casos no coinciden en número de fotogramas.")

    source_sha256 = manifest.get("source_sha256")
    review_sha256 = manifest.get("output_sha256")
    if not isinstance(source_sha256, str) or not isinstance(review_sha256, str):
        raise ValueError("El manifiesto debe incluir los hashes de ambos videos.")
    case_video = (cases_data.get("video") or {}).get("sha256")
    if case_video and case_video != source_sha256:
        raise ValueError("El SHA-256 fuente de los casos no coincide con el manifiesto.")
    if cases_data.get("source_sha256", {}).get("video") not in (None, source_sha256):
        raise ValueError("El SHA-256 fuente registrado en los casos no coincide.")
    if _sha256_file(review_video) != review_sha256:
        raise ValueError("El SHA-256 de la copia de revisión no coincide con el manifiesto.")
    if trajectories_file.is_file():
        expected_trajectories = cases_data.get("source_sha256", {}).get(
            "persistent_trajectories"
        )
        if expected_trajectories and _sha256_file(trajectories_file) != expected_trajectories:
            raise ValueError("El CSV de trayectorias no coincide con el archivo de casos.")

    source_width = _positive_int(manifest.get("source_width"), "ancho original")
    source_height = _positive_int(manifest.get("source_height"), "alto original")
    review_width = _positive_int(manifest.get("output_width"), "ancho de revisión")
    review_height = _positive_int(manifest.get("output_height"), "alto de revisión")
    if source_width / source_height != review_width / review_height:
        raise ValueError("El original y la copia de revisión tienen distinta proporción.")

    tracks = _group_tracks(trajectories, frame_count)
    cases = _validate_cases(cases, frame_count, tracks)
    case_pairs = [
        _case_pairs(case, tracks, maximum_gap_frames)
        for case in cases
    ]
    samples_by_track = _sample_rows(tracks, cases, case_pairs)
    sample_rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    embedding_extractor = (
        OSNetEmbeddingExtractor(
            embedding_weights_path,
            device=embedding_device,
            batch_size=embedding_batch_size,
        )
        if reid_method != "hsv_hog"
        else None
    )
    crop_keys: set[tuple[int, int]] = set()
    appearance_track_ids = {
        track_id
        for pairs in case_pairs
        for pair in pairs
        for track_id in (pair["first_track_id"], pair["second_track_id"])
    }
    for track_id in appearance_track_ids:
        crop_keys.update(
            (track_id, row["frame"]) for row in samples_by_track[track_id]
        )
    for track_id, rows in samples_by_track.items():
        for row in rows:
            sample_rows_by_frame[row["frame"]].append(row)
    for pairs in case_pairs:
        for pair in pairs:
            crop_keys.add((pair["first_track_id"], pair["first_frame"]))
            crop_keys.add((pair["second_track_id"], pair["second_frame"]))

    frame_case_tracks: dict[int, set[int]] = defaultdict(set)
    evidence_frames_by_case: dict[str, list[int]] = {}
    for case in cases:
        case_id = case["fragment_id"]
        frames = sorted({case["start_frame"], case["event_frame"], case["end_frame"]})
        evidence_frames_by_case[case_id] = frames
        relevant_tracks = set(case.get("track_ids", []))
        if case.get("temporal_track_id") is not None:
            relevant_tracks.add(case["temporal_track_id"])
        if case.get("candidate_previous_track_id") is not None:
            relevant_tracks.add(case["candidate_previous_track_id"])
        for frame_number in frames:
            frame_case_tracks[frame_number].update(relevant_tracks)

    required_frames = set(frame_case_tracks) | set(sample_rows_by_frame)
    for pairs in case_pairs:
        for pair in pairs:
            required_frames.update((pair["first_frame"], pair["second_frame"]))
    input_and_sampling_seconds = time.perf_counter() - preparation_started
    frame_images: dict[int, str] = {}
    crop_images: dict[tuple[int, int], dict[str, Any]] = {}
    embeddings_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    pending_embedding_keys: list[tuple[int, int]] = []
    pending_embedding_crops: list[np.ndarray] = []
    descriptors: dict[int, list[AppearanceDescriptor]] = defaultdict(list)
    scene_cut_frames: list[int] = []
    scene_detector = PersistentIdentityAssociator(
        maximum_reidentification_gap=maximum_gap_frames
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=False)

    capture = cv2.VideoCapture(str(review_video))
    if not capture.isOpened():
        raise OSError(f"No se pudo abrir la copia de revisión: {review_video}")
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_width, actual_height) != (review_width, review_height):
        capture.release()
        raise ValueError("Las dimensiones decodificadas no coinciden con el manifiesto.")

    evidence_started = time.perf_counter()
    decoded = 0

    def encode_pending_crops() -> None:
        if not pending_embedding_crops or embedding_extractor is None:
            return
        vectors = embedding_extractor.encode(pending_embedding_crops)
        for key, vector in zip(pending_embedding_keys, vectors, strict=True):
            embeddings_by_track[key[0]].append(vector)
        pending_embedding_keys.clear()
        pending_embedding_crops.clear()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            scene_detector.update([], frame, decoded - 1)
            if scene_detector.scene_cut_count > len(scene_cut_frames):
                scene_cut_frames.append(decoded)
            if decoded not in required_frames:
                continue
            for row in sample_rows_by_frame.get(decoded, []):
                box = _review_box(row, source_width, source_height, review_width, review_height)
                descriptor = extract_appearance_descriptor(frame, box)
                if descriptor is not None:
                    descriptors[row["temporal_track_id"]].append(descriptor)
            for track_id, frame_number in sorted(
                key for key in crop_keys if key[1] == decoded
            ):
                row = _row_at_frame(tracks[track_id], frame_number)
                if row is None:
                    continue
                box = _review_box(row, source_width, source_height, review_width, review_height)
                crop, quality = _make_player_crop(frame, box)
                if embedding_extractor is not None:
                    pending_embedding_keys.append((track_id, frame_number))
                    pending_embedding_crops.append(crop)
                    if len(pending_embedding_crops) >= embedding_batch_size:
                        encode_pending_crops()
                name = f"track-{track_id:04d}-frame-{frame_number:04d}.jpg"
                _write_jpeg(evidence_root / name, crop)
                crop_images[(track_id, frame_number)] = {
                    "path": f"{evidence_url_prefix}/{name}",
                    **quality,
                }
            if decoded in frame_case_tracks:
                image = _make_frame_thumbnail(
                    frame,
                    decoded,
                    trajectories,
                    frame_case_tracks[decoded],
                    source_width,
                    source_height,
                )
                name = f"frame-{decoded:04d}.jpg"
                _write_jpeg(evidence_root / name, image)
                frame_images[decoded] = f"{evidence_url_prefix}/{name}"
    finally:
        capture.release()
    encode_pending_crops()

    if decoded != frame_count:
        raise ValueError(
            f"La copia de revisión contiene {decoded} fotogramas decodificados; "
            f"el manifiesto declara {frame_count}."
        )
    evidence_extraction_seconds = time.perf_counter() - evidence_started
    appearance_by_track = {
        track_id: _median_descriptor(items)
        for track_id, items in descriptors.items()
        if items
    }

    scoring_started = time.perf_counter()
    annotated_cases: list[dict[str, Any]] = []
    for case, pairs in zip(cases, case_pairs, strict=True):
        scored_pairs = []
        excluded_scene_pairs = 0
        for pair in pairs:
            first_frame, second_frame = pair["first_frame"], pair["second_frame"]
            if any(first_frame < cut <= second_frame for cut in scene_cut_frames):
                excluded_scene_pairs += 1
                continue
            scored_pairs.append(
                _score_pair(
                    pair,
                    tracks,
                    appearance_by_track,
                    embeddings_by_track=embeddings_by_track,
                    reid_method=reid_method,
                    embedding_weight=embedding_weight,
                )
            )
        scored_pairs = [pair for pair in scored_pairs if pair is not None]
        scored_pairs.sort(
            key=lambda item: (-item["heuristic_score"], item["gap_frames"])
        )
        top_pairs = scored_pairs[:maximum_candidates_per_case]
        for candidate in top_pairs:
            candidate["proposal_id"] = _proposal_id(case, candidate)
            candidate["case_id"] = case["fragment_id"]
            candidate["candidate_group_id"] = None
            candidate["crop_comparison"] = [
                {
                    "track_id": track_id,
                    "frame": frame_number,
                    **crop_images.get((track_id, frame_number), {}),
                }
                for track_id, frame_number in (
                    (candidate["track_ids"][0], candidate["frames"][0]),
                    (candidate["track_ids"][1], candidate["frames"][1]),
                )
            ]
        nearby_cuts = [
            frame for frame in scene_cut_frames
            if case["start_frame"] <= frame <= case["end_frame"]
        ]
        scores = [item["heuristic_score"] for item in top_pairs]
        ambiguous_reasons: list[str] = []
        if len(scores) > 1 and scores[0] - scores[1] < 0.06:
            ambiguous_reasons.append("Las dos mejores propuestas tienen scores cercanos.")
        if any(
            crop.get("sharpness_laplacian_variance", 0) < 12
            for item in top_pairs
            for crop in item["crop_comparison"]
        ):
            ambiguous_reasons.append("Al menos un recorte puede estar desenfocado.")
        if nearby_cuts:
            ambiguous_reasons.append("Se detectó un posible cambio brusco de escena.")
        if excluded_scene_pairs:
            ambiguous_reasons.append(
                "Se descartaron asociaciones que cruzaban un posible cambio de escena."
            )
        if any(
            crop.get("possible_low_resolution", False)
            for item in top_pairs
            for crop in item["crop_comparison"]
        ):
            ambiguous_reasons.append("Al menos un recorte tiene resolución muy baja.")
        if case["event_type"] == "close_player_crossing_or_occlusion":
            ambiguous_reasons.append(
                "La proximidad/oclusión es un indicio geométrico, no una identidad."
            )
        appearance_ids = set(case["event_track_ids"])
        appearance_ids.update(
            track_id
            for proposal in top_pairs
            for track_id in proposal["track_ids"]
        )
        annotated_cases.append(
            {
                "fragment_id": case["fragment_id"],
                "event_type": case["event_type"],
                "frame_evidence": [
                    {"frame": frame, "path": frame_images[frame]}
                    for frame in evidence_frames_by_case[case["fragment_id"]]
                    if frame in frame_images
                ],
                "possible_camera_change_frames": nearby_cuts,
                "associations_excluded_by_scene_change": excluded_scene_pairs,
                "possible_occlusion": (
                    case["event_type"] == "close_player_crossing_or_occlusion"
                ),
                "ambiguity_reasons": ambiguous_reasons,
                "candidate_associations": top_pairs,
                "track_appearances": [
                    {
                        "track_id": track_id,
                        "samples": [
                            {
                                "frame": row["frame"],
                                **crop_images[(track_id, row["frame"])],
                            }
                            for row in samples_by_track.get(track_id, [])
                            if (track_id, row["frame"]) in crop_images
                        ],
                    }
                    for track_id in sorted(appearance_ids)
                ],
                "no_automatic_identity": True,
            }
        )

    candidate_groups, duplicate_groups = _enrich_proposal_relationships(
        annotated_cases, tracks
    )
    group_by_id = {
        group["candidate_group_id"]: group for group in candidate_groups
    }
    for case in annotated_cases:
        group_ids = sorted(
            {
                proposal["candidate_group_id"]
                for proposal in case["candidate_associations"]
                if proposal["candidate_group_id"]
            }
        )
        case["group_overview"] = [
            {
                "candidate_group_id": group_id,
                "track_ids": group_by_id[group_id]["track_ids"],
                "review_status": group_by_id[group_id]["review_status"],
            }
            for group_id in group_ids
        ]
    proposal_scoring_seconds = time.perf_counter() - scoring_started
    payload = {
        "schema_version": 2,
        "proposal_type": "automatic_visual_review_assistance",
        "video": {
            "source_sha256": source_sha256,
            "review_sha256": review_sha256,
            "frame_count": frame_count,
            "source_width": source_width,
            "source_height": source_height,
        },
        "source_sha256": {
            "cases": _sha256_file(cases_file),
            "trajectories": _sha256_file(trajectories_file),
            "manifest": _sha256_file(manifest_file),
        },
        "algorithm": {
            "appearance_descriptor": "histograma HSV del torso (16x4) + HOG (64x128)",
            "score": "0.55 apariencia, 0.25 movimiento, 0.10 escala y 0.10 confianza",
            "method_used_for_ranking": reid_method,
            "embedding_weight_in_hybrid": embedding_weight,
            "embedding_similarity": (
                "media de las tres similitudes coseno más altas de las galerías, "
                "transformadas con (coseno + 1) / 2"
            ),
            "hybrid_visual_similarity": (
                "embedding_weight * similitud_embedding + "
                "(1 - embedding_weight) * similitud_HSV_HOG"
            ),
            "maximum_gap_frames": maximum_gap_frames,
            "maximum_candidates_per_case": maximum_candidates_per_case,
            "score_is_probability": False,
            "scene_change_detector": "mismo detector de cambios bruscos del asociador ReID",
            "learned_reid_model_used": embedding_extractor is not None,
            "embedding_model": (
                embedding_extractor.configuration
                if embedding_extractor is not None
                else None
            ),
            "candidate_group_score_threshold": _CANDIDATE_GROUP_SCORE_THRESHOLD,
            "candidate_groups_are_identity_labels": False,
        },
        "evidence_url_prefix": evidence_url_prefix,
        "cases": annotated_cases,
        "candidate_groups": candidate_groups,
        "duplicate_proposal_groups": duplicate_groups,
        "proposal_summary": {
            "proposal_occurrences": sum(
                len(case["candidate_associations"]) for case in annotated_cases
            ),
            "unique_candidate_track_pairs": len(duplicate_groups),
            "repeated_candidate_track_pairs": sum(
                len(group["proposal_ids"]) > 1 for group in duplicate_groups
            ),
            "candidate_group_count": len(candidate_groups),
            "grouping_policy": (
                "componentes de enlaces de tracks temporales con score >= 0.82; "
                "solo agrupación para priorizar revisión, no identidad"
            ),
        },
        "timings_seconds": {
            "input_validation_and_sampling": input_and_sampling_seconds,
            "video_evidence_and_feature_extraction": evidence_extraction_seconds,
            "proposal_scoring_and_grouping": proposal_scoring_seconds,
            "embedding_inference": (
                embedding_extractor.inference_seconds
                if embedding_extractor is not None
                else 0.0
            ),
            "embedding_images_processed": (
                embedding_extractor.images_processed
                if embedding_extractor is not None
                else 0
            ),
            "elapsed_before_json_serialization": (
                time.perf_counter() - generation_started
            ),
            "serialization_included_in_elapsed": False,
        },
        "limitations": [
            "Las asociaciones son sugerencias visuales, no identidades humanas ni ground truth.",
            "Los scores son heurísticos y no probabilidades calibradas.",
            "Los grupos se forman solo con scores >= 0.82 y separación temporal observada; siguen siendo candidatos, no identidades.",
            "HSV/HOG puede confundir uniformes similares, recortes pequeños, borrosos u ocluidos.",
            "Los embeddings OSNet/MSMT17 no están validados para tomas de fútbol.",
            "Los cambios de cámara y las oclusiones son indicios automáticos que requieren confirmación.",
            "La evidencia no modifica ni prellena anotaciones.",
        ],
    }
    output_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _proposal_id(case: dict[str, Any], proposal: dict[str, Any]) -> str:
    value = "|".join(
        (
            case["fragment_id"],
            proposal["relation"],
            ",".join(str(item) for item in proposal["track_ids"]),
            ",".join(str(item) for item in proposal["frames"]),
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _enrich_proposal_relationships(
    cases: list[dict[str, Any]],
    tracks: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proposal_rows = []
    for case_index, case in enumerate(cases, start=1):
        for proposal in case["candidate_associations"]:
            proposal.setdefault(
                "case_id", case.get("fragment_id", f"case-{case_index:04d}")
            )
            proposal_rows.append(proposal)
    pair_occurrences: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    graph: dict[int, set[int]] = defaultdict(set)
    edge_scores: dict[tuple[int, int], float] = {}
    for proposal in proposal_rows:
        if proposal["relation"] not in {
            "candidate_same_player",
            "existing_reid_link",
        }:
            continue
        pair = tuple(sorted(proposal["track_ids"]))
        pair_occurrences[pair].append(proposal)
        if proposal["heuristic_score"] < _CANDIDATE_GROUP_SCORE_THRESHOLD:
            continue
        first_id, second_id = pair
        graph[first_id].add(second_id)
        graph[second_id].add(first_id)
        edge_scores[pair] = max(
            edge_scores.get(pair, 0.0), proposal["heuristic_score"]
        )

    duplicate_groups: list[dict[str, Any]] = []
    duplicate_id_by_pair: dict[tuple[int, int], str] = {}
    for pair, occurrences in sorted(pair_occurrences.items()):
        group_id = f"par-{pair[0]}-{pair[1]}"
        duplicate_id_by_pair[pair] = group_id
        proposal_ids = sorted(item["proposal_id"] for item in occurrences)
        duplicate_groups.append(
            {
                "duplicate_group_id": group_id,
                "track_ids": list(pair),
                "proposal_ids": proposal_ids,
                "occurrence_count": len(occurrences),
                "repeated_across_cases": len(
                    {item["case_id"] for item in occurrences}
                ) > 1,
                "heuristic_scores": sorted(
                    {item["heuristic_score"] for item in occurrences}, reverse=True
                ),
            }
        )
        for proposal in occurrences:
            proposal["duplicate_group_id"] = group_id
            proposal["duplicate_occurrence_count"] = len(occurrences)

    frame_sets = {
        track_id: {row["frame"] for row in rows}
        for track_id, rows in tracks.items()
    }
    visited: set[int] = set()
    candidate_groups: list[dict[str, Any]] = []
    group_by_pair: dict[tuple[int, int], str] = {}
    for start in sorted(graph):
        if start in visited:
            continue
        stack = [start]
        component: set[int] = set()
        while stack:
            track_id = stack.pop()
            if track_id in component:
                continue
            component.add(track_id)
            visited.add(track_id)
            stack.extend(graph[track_id] - component)
        track_ids = sorted(component)
        conflicts = [
            {
                "track_ids": [first_id, second_id],
                "overlap_frames": len(frame_sets[first_id] & frame_sets[second_id]),
            }
            for index, first_id in enumerate(track_ids)
            for second_id in track_ids[index + 1 :]
            if frame_sets[first_id] & frame_sets[second_id]
        ]
        group_id = f"grupo-candidato-{len(candidate_groups) + 1:03d}"
        group_edges = [
            (first_id, second_id)
            for first_id in track_ids
            for second_id in graph[first_id]
            if first_id < second_id
        ]
        group = {
            "candidate_group_id": group_id,
            "track_ids": track_ids,
            "proposal_count": sum(
                len(pair_occurrences[pair])
                for pair in group_edges
            ),
            "unique_link_count": len(group_edges),
            "mean_heuristic_score": round(
                sum(edge_scores[pair] for pair in group_edges) / len(group_edges), 4
            ),
            "temporal_conflicts": conflicts,
            "review_status": (
                "conflict_requires_review" if conflicts else "candidate_only"
            ),
            "is_identity": False,
        }
        candidate_groups.append(group)
        for pair in group_edges:
            group_by_pair[pair] = group_id

    for proposal in proposal_rows:
        if proposal["relation"] in {"candidate_same_player", "existing_reid_link"}:
            pair = tuple(sorted(proposal["track_ids"]))
            proposal["candidate_group_id"] = group_by_pair.get(pair)

    return candidate_groups, duplicate_groups


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {path}.")
    return data


def _read_trajectories(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "frame", "temporal_track_id", "persistent_id", "class_id",
            "x1", "y1", "x2", "y2", "confidence",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Al CSV le faltan columnas: {sorted(missing)}.")
        rows = []
        for line_number, row in enumerate(reader, 2):
            try:
                converted = {
                    "frame": int(row["frame"]),
                    "temporal_track_id": int(row["temporal_track_id"]),
                    "persistent_id": int(row["persistent_id"]),
                    "class_id": int(row["class_id"]),
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "confidence": float(row["confidence"]),
                    "identity_state": row.get("identity_state", ""),
                }
            except (TypeError, ValueError) as error:
                raise ValueError(f"Fila {line_number} del CSV no válida.") from error
            if (
                converted["frame"] < 1
                or converted["temporal_track_id"] < 1
                or converted["class_id"] < 0
                or not all(math.isfinite(converted[key]) for key in ("x1", "y1", "x2", "y2", "confidence"))
                or converted["x2"] <= converted["x1"]
                or converted["y2"] <= converted["y1"]
            ):
                raise ValueError(f"Fila {line_number} del CSV contiene valores no válidos.")
            rows.append(converted)
    if not rows:
        raise ValueError("El CSV de trayectorias está vacío.")
    return rows


def _group_tracks(
    rows: list[dict[str, Any]], frame_count: int
) -> dict[int, list[dict[str, Any]]]:
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["frame"] > frame_count:
            raise ValueError("El CSV contiene fotogramas fuera del video.")
        if row["class_id"] == 0:
            tracks[row["temporal_track_id"]].append(row)
    for observations in tracks.values():
        observations.sort(key=lambda row: row["frame"])
    if not tracks:
        raise ValueError("No hay trayectorias de personas (clase 0) para analizar.")
    return dict(tracks)


def _validate_cases(
    cases: list[Any],
    frame_count: int,
    tracks: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    validated = []
    for index, value in enumerate(cases, 1):
        if not isinstance(value, dict):
            raise ValueError(f"El caso {index} no es un objeto.")
        case_id = value.get("fragment_id")
        start, event, end = (
            value.get("start_frame"),
            value.get("event_frame"),
            value.get("end_frame"),
        )
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen_ids
            or not all(isinstance(item, int) for item in (start, event, end))
            or not 1 <= start <= event <= end <= frame_count
        ):
            raise ValueError(f"El intervalo o ID del caso {index} no es válido.")
        seen_ids.add(case_id)
        case = dict(value)
        track_ids = set()
        for key in ("temporal_track_id", "candidate_previous_track_id"):
            track_id = case.get(key)
            if track_id is not None:
                if track_id not in tracks:
                    raise ValueError(f"El caso {case_id} referencia un track inexistente.")
                track_ids.add(track_id)
        for track_id in case.get("track_ids", []):
            if track_id not in tracks:
                raise ValueError(f"El caso {case_id} referencia un track inexistente.")
            track_ids.add(track_id)
        case["event_track_ids"] = sorted(track_ids)
        validated.append(case)
    return validated


def _case_pairs(
    case: dict[str, Any],
    tracks: dict[int, list[dict[str, Any]]],
    maximum_gap_frames: int,
) -> list[dict[str, Any]]:
    event_type = case["event_type"]
    if event_type == "close_player_crossing_or_occlusion":
        ids = case.get("track_ids", [])
        if len(ids) != 2:
            return []
        first = _nearest_row(tracks[ids[0]], case["event_frame"], 5)
        second = _nearest_row(tracks[ids[1]], case["event_frame"], 5)
        if first and second:
            return [_pair(ids[0], ids[1], first, second, "simultaneous_tracks")]
        return []

    if event_type == "within_track_disappearance":
        track_id = case["temporal_track_id"]
        before = _nearest_row(
            [row for row in tracks[track_id] if row["frame"] <= case["event_frame"]],
            case["event_frame"],
            maximum_gap_frames,
        )
        after = _nearest_row(
            [row for row in tracks[track_id] if row["frame"] > case["event_frame"]],
            case["event_frame"],
            maximum_gap_frames,
        )
        if before and after:
            return [_pair(track_id, track_id, before, after, "same_track_continuity")]
        return []

    target_id = case.get("temporal_track_id")
    if target_id not in tracks:
        return []
    target = tracks[target_id]
    target_start = target[0]["frame"]
    candidates: list[dict[str, Any]] = []
    explicit_previous = case.get("candidate_previous_track_id")
    for previous_id, observations in tracks.items():
        if previous_id == target_id:
            continue
        previous_end = observations[-1]["frame"]
        gap = target_start - previous_end
        explicit = previous_id == explicit_previous
        same_persistent = (
            case.get("persistent_id") is not None
            and any(row["persistent_id"] == case["persistent_id"] for row in target)
            and any(row["persistent_id"] == case["persistent_id"] for row in observations)
        )
        if (0 <= gap <= maximum_gap_frames or explicit) and (
            previous_end <= target_start
        ):
            candidates.append(
                _pair(
                    previous_id,
                    target_id,
                    observations[-1],
                    target[0],
                    "existing_reid_link" if same_persistent else "candidate_same_player",
                )
            )
    return candidates


def _pair(
    first_id: int,
    second_id: int,
    first: dict[str, Any],
    second: dict[str, Any],
    relation: str,
) -> dict[str, Any]:
    return {
        "first_track_id": first_id,
        "second_track_id": second_id,
        "first_frame": first["frame"],
        "second_frame": second["frame"],
        "relation": relation,
    }


def _sample_rows(
    tracks: dict[int, list[dict[str, Any]]],
    cases: list[dict[str, Any]],
    case_pairs: list[list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    sample_frames: dict[int, set[int]] = {}
    for track_id, rows in tracks.items():
        indexes = np.linspace(
            0, len(rows) - 1, num=min(_MAX_TRACK_SAMPLES, len(rows)), dtype=int
        )
        sample_frames[track_id] = {rows[int(index)]["frame"] for index in indexes}
        sample_frames[track_id].update((rows[0]["frame"], rows[-1]["frame"]))
    for case, pairs in zip(cases, case_pairs, strict=True):
        for pair in pairs:
            sample_frames[pair["first_track_id"]].add(pair["first_frame"])
            sample_frames[pair["second_track_id"]].add(pair["second_frame"])
        if case["event_type"] == "close_player_crossing_or_occlusion":
            for track_id in case.get("track_ids", []):
                row = _nearest_row(tracks[track_id], case["event_frame"], 5)
                if row:
                    sample_frames[track_id].add(row["frame"])
    return {
        track_id: [
            row for row in rows if row["frame"] in sample_frames[track_id]
        ]
        for track_id, rows in tracks.items()
    }


def _score_pair(
    pair: dict[str, Any],
    tracks: dict[int, list[dict[str, Any]]],
    descriptors: dict[int, AppearanceDescriptor],
    *,
    embeddings_by_track: dict[int, list[np.ndarray]] | None = None,
    reid_method: str = "hsv_hog",
    embedding_weight: float = 0.60,
) -> dict[str, Any] | None:
    first_id, second_id = pair["first_track_id"], pair["second_track_id"]
    first, second = descriptors.get(first_id), descriptors.get(second_id)
    first_row = _row_at_frame(tracks[first_id], pair["first_frame"])
    second_row = _row_at_frame(tracks[second_id], pair["second_frame"])
    if first_row is None or second_row is None:
        return None
    color: float | None = None
    shape: float | None = None
    appearance: float | None = None
    if first is not None and second is not None:
        color, shape = compare_appearance_descriptors(first, second)
        appearance = 0.5 * color + 0.5 * shape
    embedding_similarity = None
    if embeddings_by_track is not None:
        embedding_similarity = _embedding_gallery_similarity(
            embeddings_by_track.get(first_id, []),
            embeddings_by_track.get(second_id, []),
        )
    motion, scale = _motion_and_scale(
        tracks[first_id], first_row, second_row
    )
    confidence = (first_row["confidence"] + second_row["confidence"]) / 2
    method_scores: dict[str, float] = {}
    if appearance is not None:
        method_scores["hsv_hog"] = (
            0.55 * appearance
            + 0.25 * motion
            + 0.10 * scale
            + 0.10 * confidence
        )
    if embedding_similarity is not None:
        method_scores["embedding"] = (
            0.55 * embedding_similarity
            + 0.25 * motion
            + 0.10 * scale
            + 0.10 * confidence
        )
    if appearance is not None and embedding_similarity is not None:
        hybrid_visual = (
            embedding_weight * embedding_similarity
            + (1.0 - embedding_weight) * appearance
        )
        method_scores["hybrid"] = (
            0.55 * hybrid_visual
            + 0.25 * motion
            + 0.10 * scale
            + 0.10 * confidence
        )
    score = method_scores.get(reid_method)
    if reid_method == "embedding":
        selected_visual = embedding_similarity
    elif reid_method == "hybrid":
        selected_visual = (
            embedding_weight * embedding_similarity
            + (1.0 - embedding_weight) * appearance
            if embedding_similarity is not None and appearance is not None
            else None
        )
    else:
        selected_visual = appearance
    if score is None or selected_visual is None:
        return None
    gap = max(0, pair["second_frame"] - pair["first_frame"])
    return {
        "track_ids": [first_id, second_id],
        "frames": [pair["first_frame"], pair["second_frame"]],
        "relation": pair["relation"],
        "score_method": reid_method,
        "heuristic_score": round(float(np.clip(score, 0, 1)), 4),
        "method_scores": {
            method: round(float(np.clip(value, 0, 1)), 4)
            for method, value in method_scores.items()
        },
        "confidence_level": (
            "alta, heurística" if score >= 0.82
            else "media, heurística" if score >= 0.70
            else "baja, heurística"
        ),
        "gap_frames": gap,
        "evidence": {
            "similarity_hsv": round(color, 4) if color is not None else None,
            "similarity_hog": round(shape, 4) if shape is not None else None,
            "similarity_appearance": (
                round(appearance, 4) if appearance is not None else None
            ),
            "similarity_embedding": (
                round(embedding_similarity, 4)
                if embedding_similarity is not None
                else None
            ),
            "similarity_visual_used": round(selected_visual, 4),
            "similarity_motion": round(motion, 4),
            "similarity_scale": round(scale, 4),
            "mean_detection_confidence": round(confidence, 4),
        },
        "score_is_probability": False,
    }


def _embedding_gallery_similarity(
    first: list[np.ndarray], second: list[np.ndarray]
) -> float | None:
    if not first or not second:
        return None
    similarities = sorted(
        (cosine_similarity(left, right) for left in first for right in second),
        reverse=True,
    )
    top_similarities = similarities[:3]
    return (float(np.mean(top_similarities)) + 1.0) / 2.0


def _motion_and_scale(
    earlier_track: list[dict[str, Any]],
    earlier: dict[str, Any],
    later: dict[str, Any],
) -> tuple[float, float]:
    recent = [
        row for row in earlier_track
        if row["frame"] <= earlier["frame"]
    ][-5:]
    velocities = []
    for left, right in zip(recent, recent[1:]):
        elapsed = right["frame"] - left["frame"]
        if elapsed > 0:
            velocities.append(
                (
                    (right["x1"] + right["x2"] - left["x1"] - left["x2"]) / (2 * elapsed),
                    (right["y1"] + right["y2"] - left["y1"] - left["y2"]) / (2 * elapsed),
                )
            )
    velocity = np.mean(velocities, axis=0) if velocities else np.zeros(2)
    gap = max(later["frame"] - earlier["frame"], 1)
    old_center = np.asarray(
        ((earlier["x1"] + earlier["x2"]) / 2, (earlier["y1"] + earlier["y2"]) / 2)
    )
    new_center = np.asarray(
        ((later["x1"] + later["x2"]) / 2, (later["y1"] + later["y2"]) / 2)
    )
    predicted = old_center + velocity * gap
    height = max(earlier["y2"] - earlier["y1"], later["y2"] - later["y1"], 12)
    normalized_motion = float(
        np.linalg.norm(new_center - predicted) / (height * (0.75 + 0.08 * gap))
    )
    motion = math.exp(-0.5 * (normalized_motion / 2.5) ** 2)
    old_area = max(
        (earlier["x2"] - earlier["x1"]) * (earlier["y2"] - earlier["y1"]), 1
    )
    new_area = max(
        (later["x2"] - later["x1"]) * (later["y2"] - later["y1"]), 1
    )
    scale = math.exp(-0.5 * abs(math.log(new_area / old_area)))
    return motion, scale


def _median_descriptor(
    items: list[AppearanceDescriptor],
) -> AppearanceDescriptor:
    color = np.median(np.stack([item.color for item in items]), axis=0)
    color /= max(float(color.sum()), 1e-12)
    shape = np.median(np.stack([item.shape for item in items]), axis=0)
    norm = float(np.linalg.norm(shape))
    if norm > 0:
        shape /= norm
    return AppearanceDescriptor(color=color.astype(np.float32), shape=shape.astype(np.float32))


def _review_box(
    row: dict[str, Any],
    source_width: int,
    source_height: int,
    review_width: int,
    review_height: int,
) -> tuple[float, float, float, float]:
    return (
        row["x1"] * review_width / source_width,
        row["y1"] * review_height / source_height,
        row["x2"] * review_width / source_width,
        row["y2"] * review_height / source_height,
    )


def _make_player_crop(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * 0.15, (y2 - y1) * 0.08
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    source = frame[top:bottom, left:right]
    if source.size == 0:
        source = np.zeros((16, 8, 3), dtype=np.uint8)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    scale = min(192 / source.shape[1], 288 / source.shape[0])
    resized = cv2.resize(
        source,
        (max(1, round(source.shape[1] * scale)), max(1, round(source.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.zeros((288, 192, 3), dtype=np.uint8)
    x_offset = (192 - resized.shape[1]) // 2
    y_offset = (288 - resized.shape[0]) // 2
    canvas[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = resized
    return canvas, {
        "crop_height_pixels": int(source.shape[0]),
        "sharpness_laplacian_variance": round(sharpness, 2),
        "possible_low_resolution": source.shape[0] < 32,
    }


def _make_frame_thumbnail(
    frame: np.ndarray,
    frame_number: int,
    trajectories: list[dict[str, Any]],
    relevant_track_ids: set[int],
    source_width: int,
    source_height: int,
) -> np.ndarray:
    thumbnail = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    scale_x, scale_y = 640 / source_width, 360 / source_height
    for row in trajectories:
        if row["frame"] != frame_number:
            continue
        track_id = row["temporal_track_id"]
        color = (0, 220, 255) if track_id in relevant_track_ids else (110, 150, 170)
        x1 = round(row["x1"] * scale_x)
        y1 = round(row["y1"] * scale_y)
        x2 = round(row["x2"] * scale_x)
        y2 = round(row["y2"] * scale_y)
        cv2.rectangle(thumbnail, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            thumbnail,
            f"T:{track_id}",
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        thumbnail,
        f"Fotograma {frame_number}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return thumbnail


def _nearest_row(
    rows: list[dict[str, Any]], frame_number: int, maximum_distance: int
) -> dict[str, Any] | None:
    if not rows:
        return None
    row = min(rows, key=lambda item: abs(item["frame"] - frame_number))
    return row if abs(row["frame"] - frame_number) <= maximum_distance else None


def _row_at_frame(
    rows: list[dict[str, Any]], frame_number: int
) -> dict[str, Any] | None:
    return next((row for row in rows if row["frame"] == frame_number), None)


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"El {label} del manifiesto no es válido.")
    return value


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"No se sobrescribió evidencia existente: {path}")
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise OSError(f"No se pudo codificar la imagen de evidencia: {path.name}")
    path.write_bytes(encoded.tobytes())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    identity_review = root / "outputs" / "identity-review"
    video_runs = root / "outputs" / "video-runs"
    parser = argparse.ArgumentParser(
        description="Genera evidencia visual para revisar manualmente propuestas ReID."
    )
    parser.add_argument(
        "--review-video", type=Path,
        default=identity_review / "review-video-browser.mp4",
        help="Copia de video compatible con navegador y alineada con el original",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=identity_review / "review-video-browser.mp4.json",
        help="Manifiesto de la copia de revisión",
    )
    parser.add_argument(
        "--trajectories", type=Path,
        default=video_runs / "4052ed58-6c37-4a2c-b62a-920dfe1e700b_trajectories.csv",
        help="CSV de trayectorias del experimento",
    )
    parser.add_argument(
        "--cases", type=Path,
        default=identity_review / "review-cases-final.json",
        help="JSON de casos priorizados",
    )
    parser.add_argument(
        "--output", type=Path,
        default=identity_review / "reid-review-proposals-v4.json",
        help="Archivo JSON de propuestas nuevo; no se sobrescribe",
    )
    parser.add_argument(
        "--evidence-dir", type=Path,
        default=identity_review / "reid-review-evidence-v4",
        help="Directorio nuevo para fotogramas y recortes comparativos",
    )
    parser.add_argument("--maximum-gap-frames", type=int, default=150)
    parser.add_argument("--maximum-candidates-per-case", type=int, default=3)
    parser.add_argument(
        "--reid-method",
        choices=("hsv_hog", "embedding", "hybrid"),
        default="hsv_hog",
        help="Método usado para ordenar candidatos; HSV/HOG por defecto",
    )
    parser.add_argument(
        "--embedding-weights-path",
        type=Path,
        default=Path("models") / MODEL_FILENAME,
        help="Checkpoint OSNet fijado y verificado",
    )
    parser.add_argument(
        "--embedding-device",
        default="auto",
        help="Dispositivo OSNet: auto, cpu, cuda o mps",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-weight", type=float, default=0.60)
    args = parser.parse_args()
    try:
        result = generate_review_proposals(
            args.review_video,
            args.manifest,
            args.trajectories,
            args.cases,
            args.output,
            args.evidence_dir,
            maximum_gap_frames=args.maximum_gap_frames,
            maximum_candidates_per_case=args.maximum_candidates_per_case,
            reid_method=args.reid_method,
            embedding_weights_path=args.embedding_weights_path,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            embedding_weight=args.embedding_weight,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
        return
    print(
        json.dumps(
            {
                "proposals": str(args.output.resolve()),
                "evidence_directory": str(args.evidence_dir.resolve()),
                "cases": len(result["cases"]),
                "scene_change_frames": sorted(
                    {
                        frame
                        for case in result["cases"]
                        for frame in case["possible_camera_change_frames"]
                    }
                ),
                "candidate_associations": sum(
                    len(case["candidate_associations"]) for case in result["cases"]
                ),
                "timings_seconds": result["timings_seconds"],
                "ground_truth_created": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
