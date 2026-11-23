"""Evaluation of ReID proposals against explicitly human-validated identities."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any


_LINK_RELATIONS = {"candidate_same_player", "existing_reid_link"}
_REID_METHODS = ("hsv_hog", "embedding", "hybrid")


class ReIDMetricsError(ValueError):
    """Raised when review exports cannot safely support ReID evaluation."""


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"El archivo no existe: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReIDMetricsError(f"JSON no válido en {source}: {error}") from error
    if not isinstance(payload, dict):
        raise ReIDMetricsError(f"Se esperaba un objeto JSON en {source}.")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_reid_proposals(
    review: dict[str, Any],
    proposals: dict[str, Any],
    *,
    actual_proposals_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare generated candidate links with manually validated track labels.

    Only ``review.identities[*].tracks`` is treated as ground truth. Proposal
    decisions alone, heuristic scores, and candidate groups are never labels.
    """
    if review.get("record_type") != "human_review_with_manual_ground_truth":
        raise ReIDMetricsError(
            "La revisión no contiene el tipo de exportación humana esperado."
        )
    if proposals.get("schema_version") != 2 or not isinstance(
        proposals.get("cases"), list
    ):
        raise ReIDMetricsError("El archivo de propuestas ReID no es compatible.")
    review_video = review.get("video")
    proposal_video = proposals.get("video")
    video_hash = review_video.get("sha256") if isinstance(review_video, dict) else None
    proposal_video_hash = (
        proposal_video.get("source_sha256")
        if isinstance(proposal_video, dict)
        else None
    )
    if not isinstance(video_hash, str) or video_hash != proposal_video_hash:
        raise ReIDMetricsError(
            "El SHA-256 del video de las propuestas y la revisión no coincide."
        )
    expected_proposals_hash = review.get("proposal_file_sha256")
    if (
        actual_proposals_sha256 is not None
        and expected_proposals_hash != actual_proposals_sha256
    ):
        raise ReIDMetricsError(
            "El SHA-256 del archivo de propuestas no coincide con la revisión humana."
        )

    identities = review.get("identities")
    if not isinstance(identities, list):
        raise ReIDMetricsError("La revisión no contiene la lista de identidades.")
    review_records = review.get("review_records")
    if not isinstance(review_records, list):
        raise ReIDMetricsError("La revisión no contiene estados de propuestas.")
    decisions: dict[str, dict[str, Any]] = {}
    for record in review_records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("proposal_id"), str)
            or record["proposal_id"] in decisions
        ):
            raise ReIDMetricsError("El registro de decisiones contiene IDs inválidos o duplicados.")
        decisions[record["proposal_id"]] = record
    labels: dict[str, str] = {}
    identity_ids: set[str] = set()
    for identity in identities:
        if not isinstance(identity, dict):
            raise ReIDMetricsError("Cada identidad debe ser un objeto JSON.")
        identity_id = identity.get("identity_id")
        tracks = identity.get("tracks")
        if (
            not isinstance(identity_id, str)
            or not identity_id.strip()
            or identity_id in identity_ids
            or identity.get("status") != "validated_manual"
            or identity.get("validation_source")
            != "explicit_human_track_assignment"
            or not isinstance(tracks, list)
            or not tracks
        ):
            raise ReIDMetricsError(
                "Ground Truth solo admite identidades únicas con asignación humana validada."
            )
        if identity_id.upper().startswith("TEST-ONLY-"):
            raise ReIDMetricsError(
                f"La identidad {identity_id} está reservada para pruebas y no se evalúa como Ground Truth."
            )
        identity_ids.add(identity_id)
        for track in tracks:
            if (
                not isinstance(track, dict)
                or track.get("status") != "validated_manual"
                or track.get("decision") != "human_confirmed_identity_assignment"
                or not isinstance(track.get("evidence_frames"), list)
                or not track["evidence_frames"]
                or not isinstance(track.get("evidence_proposals"), list)
                or not track["evidence_proposals"]
            ):
                raise ReIDMetricsError(
                    "Cada track necesita una asignación humana y fotogramas de evidencia."
                )
            track_id = _normalize_track_id(track.get("track_id"))
            if track_id in labels:
                raise ReIDMetricsError(
                    f"El track T:{track_id} está asignado a más de una identidad."
                )
            labels[track_id] = identity_id
            for evidence in track["evidence_proposals"]:
                if (
                    not isinstance(evidence, dict)
                    or not isinstance(evidence.get("proposal_id"), str)
                    or evidence.get("decision")
                    != "human_confirmed_identity_assignment"
                    or evidence.get("proposal_file_sha256")
                    != expected_proposals_hash
                    or not isinstance(evidence.get("evidence_frames"), list)
                    or not evidence["evidence_frames"]
                    or any(
                        not isinstance(frame, int)
                        or isinstance(frame, bool)
                        or frame < 1
                        for frame in evidence["evidence_frames"]
                    )
                ):
                    raise ReIDMetricsError(
                        f"La procedencia de T:{track_id} no es una asignación humana válida."
                    )
                human_decision = decisions.get(evidence["proposal_id"])
                if (
                    human_decision is None
                    or human_decision.get("decision") != "accepted"
                    or human_decision.get("decision_source") != "human_review"
                    or track_id
                    not in {
                        _normalize_track_id(item)
                        for item in human_decision.get("track_ids", [])
                    }
                ):
                    raise ReIDMetricsError(
                        f"La evidencia {evidence['proposal_id']} de T:{track_id} no fue aceptada por una persona."
                    )

    candidate_scores: dict[tuple[str, str], float] = {}
    proposals_by_id: dict[str, dict[str, Any]] = {}
    for case in proposals["cases"]:
        if not isinstance(case, dict) or not isinstance(
            case.get("candidate_associations"), list
        ):
            raise ReIDMetricsError("El archivo de propuestas contiene un caso no válido.")
        for proposal in case["candidate_associations"]:
            if (
                not isinstance(proposal, dict)
                or proposal.get("relation") not in _LINK_RELATIONS
            ):
                continue
            proposal_id = proposal.get("proposal_id")
            if isinstance(proposal_id, str):
                if proposal_id in proposals_by_id:
                    raise ReIDMetricsError(
                        f"ID de propuesta duplicado: {proposal_id}."
                    )
                proposals_by_id[proposal_id] = proposal
            track_ids = proposal.get("track_ids")
            if not isinstance(track_ids, list) or len(track_ids) != 2:
                raise ReIDMetricsError("Una propuesta de enlace no tiene dos tracks.")
            pair = tuple(sorted(_normalize_track_id(item) for item in track_ids))
            if pair[0] == pair[1]:
                raise ReIDMetricsError("Una propuesta no puede enlazar un track consigo mismo.")
            score = proposal.get("heuristic_score")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
            ):
                continue
            candidate_scores[pair] = max(candidate_scores.get(pair, -math.inf), float(score))

    for identity in identities:
        for track in identity["tracks"]:
            track_id = _normalize_track_id(track["track_id"])
            for evidence in track["evidence_proposals"]:
                proposal = proposals_by_id.get(evidence["proposal_id"])
                proposal_track_ids = (
                    proposal.get("track_ids", []) if proposal else []
                )
                human_decision = decisions[evidence["proposal_id"]]
                decision_track_ids = human_decision.get("track_ids")
                decision_frames = human_decision.get("frames")
                track_index = next(
                    (
                        index
                        for index, value in enumerate(proposal_track_ids)
                        if _normalize_track_id(value) == track_id
                    ),
                    None,
                )
                proposal_frames = proposal.get("frames", []) if proposal else []
                if (
                    proposal is None
                    or proposal.get("relation") not in _LINK_RELATIONS
                    or track_index is None
                    or not isinstance(decision_track_ids, list)
                    or sorted(
                        _normalize_track_id(value) for value in decision_track_ids
                    )
                    != sorted(
                        _normalize_track_id(value) for value in proposal_track_ids
                    )
                    or decision_frames != proposal.get("frames")
                    or human_decision.get("case_id") != proposal.get("case_id")
                    or len(evidence["evidence_frames"]) != 1
                    or track_index >= len(proposal_frames)
                    or evidence["evidence_frames"][0] != proposal_frames[track_index]
                ):
                    raise ReIDMetricsError(
                        f"La propuesta {evidence['proposal_id']} no valida la evidencia de T:{track_id}."
                    )

    track_ids = sorted(labels)
    all_pairs = list(combinations(track_ids, 2))
    tp = fp = fn = tn = 0
    for pair in all_pairs:
        actual_same = labels[pair[0]] == labels[pair[1]]
        predicted_same = pair in candidate_scores
        if actual_same and predicted_same:
            tp += 1
        elif actual_same:
            fn += 1
        elif predicted_same:
            fp += 1
        else:
            tn += 1

    positive_pairs = tp + fn
    negative_pairs = fp + tn
    predicted_pairs = tp + fp
    precision = _ratio(tp, predicted_pairs)
    recall = _ratio(tp, positive_pairs)
    f1 = (
        0.0
        if precision == 0 and recall == 0
        else _ratio(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    fpr = _ratio(fp, negative_pairs)
    fnr = _ratio(fn, positive_pairs)
    top1, top5, query_count = _top_k_accuracy(labels, candidate_scores)
    enough_pair_labels = positive_pairs > 0 and negative_pairs > 0

    unavailable_reasons = []
    if len(labels) < 2:
        unavailable_reasons.append("Se necesitan al menos dos tracks confirmados.")
    if positive_pairs == 0:
        unavailable_reasons.append(
            "Faltan dos o más tracks humanos confirmados con la misma identidad."
        )
    if negative_pairs == 0:
        unavailable_reasons.append(
            "Faltan tracks confirmados con identidades humanas distintas."
        )
    if query_count == 0:
        unavailable_reasons.append(
            "No hay consultas con un track positivo para medir Top-1/Top-5."
        )

    return {
        "schema_version": 1,
        "record_type": "reid_metrics_against_manual_ground_truth",
        "status": "sufficient_for_proposal_conditioned_diagnostic"
        if enough_pair_labels
        else "insufficient_ground_truth",
        "evaluation_scope": {
            "type": "diagnostico_condicionado_a_propuestas",
            "unit": "pares no ordenados de tracks etiquetados manualmente",
            "prediction": (
                "Existe al menos una propuesta candidate_same_player o "
                "existing_reid_link entre los dos tracks."
            ),
            "generalization_claim_allowed": False,
            "reason": (
                "Las identidades de esta interfaz se construyen confirmando "
                "propuestas; por tanto, el conjunto etiquetado no es una muestra "
                "independiente de las propuestas y no demuestra rendimiento "
                "general del ReID."
            ),
        },
        "video_sha256": video_hash,
        "proposal_file_sha256": expected_proposals_hash,
        "ground_truth_source": "explicit_human_track_assignments_only",
        "counts": {
            "validated_tracks": len(labels),
            "validated_identities": len(identity_ids),
            "labeled_track_pairs": len(all_pairs),
            "positive_identity_pairs": positive_pairs,
            "negative_identity_pairs": negative_pairs,
            "candidate_pairs_with_scores": len(candidate_scores),
            "retrieval_queries": query_count,
        },
        "confusion_matrix": {
            "same_identity": {"predicted_link": tp, "not_predicted": fn},
            "different_identity": {"predicted_link": fp, "not_predicted": tn},
        },
        "metrics": {
            "precision": _metric(precision, predicted_pairs > 0, "No hay enlaces candidatos predichos."),
            "recall": _metric(recall, positive_pairs > 0, "No hay pares positivos confirmados."),
            "f1": _metric(f1, precision is not None and recall is not None, "Precision o Recall no se pueden calcular."),
            "false_positive_rate": _metric(fpr, negative_pairs > 0, "No hay pares negativos confirmados."),
            "false_negative_rate": _metric(fnr, positive_pairs > 0, "No hay pares positivos confirmados."),
            "top1_accuracy": _metric(top1, query_count > 0, "No hay consultas positivas para Top-1."),
            "top5_accuracy": _metric(top5, query_count > 0, "No hay consultas positivas para Top-5."),
        },
        "limitations": [
            "La clase positiva es la presencia de una propuesta candidate_same_player o existing_reid_link entre tracks confirmados.",
            "La matriz de confusión usa todos los pares de tracks que tienen identidad humana confirmada; el universo no incluye tracks no anotados.",
            "Top-1/Top-5 ordena solo candidatos con score presentes en los casos muestreados, no todas las apariciones del partido.",
            "Los scores se usan únicamente para ordenar candidatos; no son probabilidades calibradas.",
            "Las métricas no miden calidad de detección, localización ni cobertura de tracking.",
        ],
        "missing_data": unavailable_reasons,
    }


def evaluate_reid_methods(
    review: dict[str, Any],
    proposals: dict[str, Any],
    *,
    actual_proposals_sha256: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare method scores against independently assigned human track labels.

    The result is deliberately candidate-conditioned: it evaluates only pairs
    in the supplied candidate-association set, and is not a generalization
    estimate for unseen tracks or matches.
    """
    if review.get("record_type") != "human_review_with_manual_ground_truth":
        raise ReIDMetricsError("La revisión humana no tiene un tipo compatible.")
    if proposals.get("schema_version") != 2 or not isinstance(
        proposals.get("cases"), list
    ):
        raise ReIDMetricsError("El archivo de propuestas ReID no es compatible.")
    review_video = review.get("video")
    proposal_video = proposals.get("video")
    video_hash = review_video.get("sha256") if isinstance(review_video, dict) else None
    if (
        not isinstance(video_hash, str)
        or not isinstance(proposal_video, dict)
        or proposal_video.get("source_sha256") != video_hash
    ):
        raise ReIDMetricsError(
            "El SHA-256 del video de las propuestas y la revisión no coincide."
        )
    expected_proposals_hash = review.get("proposal_file_sha256")
    if (
        actual_proposals_sha256 is not None
        and expected_proposals_hash != actual_proposals_sha256
    ):
        raise ReIDMetricsError(
            "El SHA-256 del archivo de propuestas no coincide con la revisión humana."
        )
    selected_thresholds = {method: 0.82 for method in _REID_METHODS}
    if thresholds:
        unknown = set(thresholds) - set(_REID_METHODS)
        if unknown:
            raise ReIDMetricsError(
                f"Métodos de evaluación desconocidos: {', '.join(sorted(unknown))}."
            )
        selected_thresholds.update(thresholds)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in selected_thresholds.values()
    ):
        raise ReIDMetricsError("Los umbrales de método deben estar entre 0 y 1.")

    identities = review.get("identities")
    if not isinstance(identities, list):
        raise ReIDMetricsError("La revisión no contiene identidades.")
    labels: dict[str, str] = {}
    identity_ids: set[str] = set()
    for identity in identities:
        if not isinstance(identity, dict):
            raise ReIDMetricsError("Cada identidad debe ser un objeto JSON.")
        identity_id = identity.get("identity_id")
        tracks = identity.get("tracks")
        if (
            not isinstance(identity_id, str)
            or not identity_id.strip()
            or identity_id.upper().startswith("TEST-ONLY-")
            or identity_id in identity_ids
            or identity.get("status") != "validated_manual"
            or identity.get("validation_source")
            not in {
                "independent_human_track_annotations",
                "explicit_human_track_assignment",
            }
            or not isinstance(tracks, list)
        ):
            raise ReIDMetricsError(
                "Ground Truth requiere identidades humanas únicas y validadas."
            )
        identity_ids.add(identity_id)
        for track in tracks:
            if not isinstance(track, dict):
                raise ReIDMetricsError("Cada track anotado debe ser un objeto JSON.")
            if (
                track.get("annotation_source")
                != "independent_human_track_annotation"
                or track.get("status") != "validated_manual"
                or track.get("decision") != "human_confirmed_identity_assignment"
                or not isinstance(track.get("evidence_frames"), list)
                or not track["evidence_frames"]
                or any(
                    isinstance(frame, bool)
                    or not isinstance(frame, int)
                    or frame < 1
                    for frame in track["evidence_frames"]
                )
            ):
                continue
            track_id = _normalize_track_id(track.get("track_id"))
            if track_id in labels:
                raise ReIDMetricsError(
                    f"El track T:{track_id} está asignado a más de una identidad."
                )
            labels[track_id] = identity_id

    candidate_scores: dict[str, dict[tuple[str, str], float]] = {
        method: {} for method in _REID_METHODS
    }
    for case in proposals["cases"]:
        if not isinstance(case, dict) or not isinstance(
            case.get("candidate_associations"), list
        ):
            raise ReIDMetricsError("El archivo de propuestas contiene un caso no válido.")
        for proposal in case["candidate_associations"]:
            if not isinstance(proposal, dict) or proposal.get(
                "relation"
            ) not in _LINK_RELATIONS:
                continue
            track_ids = proposal.get("track_ids")
            method_scores = proposal.get("method_scores")
            if not isinstance(track_ids, list) or len(track_ids) != 2:
                raise ReIDMetricsError("Una propuesta de enlace no tiene dos tracks.")
            pair = tuple(sorted(_normalize_track_id(item) for item in track_ids))
            if pair[0] == pair[1]:
                raise ReIDMetricsError(
                    "Una propuesta no puede enlazar un track consigo mismo."
                )
            if not isinstance(method_scores, dict):
                continue
            for method in _REID_METHODS:
                value = method_scores.get(method)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                ):
                    continue
                candidate_scores[method][pair] = max(
                    candidate_scores[method].get(pair, -math.inf), float(value)
                )

    all_pairs = list(combinations(sorted(labels), 2))
    positive_pairs = sum(labels[first] == labels[second] for first, second in all_pairs)
    negative_pairs = len(all_pairs) - positive_pairs
    reports: dict[str, Any] = {}
    for method in _REID_METHODS:
        scores = candidate_scores[method]
        predicted = {
            pair for pair, score in scores.items()
            if score >= selected_thresholds[method]
        }
        tp = sum(
            labels[first] == labels[second] and (first, second) in predicted
            for first, second in all_pairs
        )
        fp = sum(
            labels[first] != labels[second] and (first, second) in predicted
            for first, second in all_pairs
        )
        fn = positive_pairs - tp
        tn = negative_pairs - fp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, positive_pairs)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        top1, top5, mean_ap, query_count = _retrieval_metrics(
            labels, scores
        )
        reports[method] = {
            "threshold": selected_thresholds[method],
            "counts": {
                "scored_candidate_pairs": len(scores),
                "predicted_links": len(predicted),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "retrieval_queries": query_count,
            },
            "confusion_matrix": {
                "same_identity": {"predicted_link": tp, "not_predicted": fn},
                "different_identity": {"predicted_link": fp, "not_predicted": tn},
            },
            "metrics": {
                "precision": _metric(precision, tp + fp > 0, "No se predijeron enlaces."),
                "recall": _metric(recall, positive_pairs > 0, "No hay pares positivos."),
                "f1": _metric(f1, 2 * tp + fp + fn > 0, "F1 no disponible."),
                "false_positive_rate": _metric(
                    _ratio(fp, negative_pairs), negative_pairs > 0,
                    "No hay pares negativos."
                ),
                "false_negative_rate": _metric(
                    _ratio(fn, positive_pairs), positive_pairs > 0,
                    "No hay pares positivos."
                ),
                "top1_accuracy": _metric(top1, query_count > 0, "No hay consultas con positivos."),
                "top5_accuracy": _metric(top5, query_count > 0, "No hay consultas con positivos."),
                "mean_average_precision": _metric(
                    mean_ap, query_count > 0, "No hay consultas con positivos."
                ),
            },
        }

    enough_labels = len(labels) >= 2 and positive_pairs > 0 and negative_pairs > 0
    return {
        "schema_version": 1,
        "record_type": "independent_reid_method_comparison",
        "status": "sufficient_for_candidate_conditioned_diagnostic"
        if enough_labels
        else "insufficient_independent_ground_truth",
        "evaluation_scope": {
            "type": "diagnostico_independiente_condicionado_a_candidatos",
            "unit": "pares de tracks con etiquetas humanas independientes",
            "ground_truth_source": "independent_human_track_annotation",
            "candidate_universe": (
                "Pares candidate_same_player/existing_reid_link presentes en el "
                "archivo de propuestas; pares sin propuesta cuentan como no enlazados."
            ),
            "generalization_claim_allowed": False,
            "reason": (
                "Las anotaciones directas son independientes de las propuestas, pero "
                "el universo de candidatos procede de un único partido y no incluye "
                "todos los pares posibles ni partidos independientes."
            ),
        },
        "video_sha256": video_hash,
        "proposal_file_sha256": expected_proposals_hash,
        "counts": {
            "independently_validated_tracks": len(labels),
            "independently_validated_identities": len(set(labels.values())),
            "labeled_track_pairs": len(all_pairs),
            "positive_identity_pairs": positive_pairs,
            "negative_identity_pairs": negative_pairs,
        },
        "methods": reports,
        "missing_data": [] if enough_labels else [
            "Se requieren tracks independientes de una misma identidad y de identidades distintas."
        ],
        "limitations": [
            "Comparación de scores sobre candidatos generados; no estima generalización.",
            "Los umbrales son configurables y los scores no se interpretan como probabilidades.",
            "No mide detección, calidad de tracking, MOTA, IDF1 ni HOTA.",
            "Un solo partido no permite separar entrenamiento, validación y prueba por partido.",
        ],
    }


def _normalize_track_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ReIDMetricsError("Los IDs de track deben ser cadenas o enteros.")
    track_id = str(value).strip()
    if not track_id:
        raise ReIDMetricsError("El ID de track no puede estar vacío.")
    if track_id.isdecimal():
        return str(int(track_id))
    return track_id


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _metric(value: float | None, available: bool, reason: str) -> dict[str, Any]:
    return {
        "value": value if available else None,
        "available": available,
        "reason": None if available else reason,
    }


def _top_k_accuracy(
    labels: dict[str, str],
    candidate_scores: dict[tuple[str, str], float],
) -> tuple[float | None, float | None, int]:
    tracks_by_identity: dict[str, list[str]] = {}
    for track_id, identity_id in labels.items():
        tracks_by_identity.setdefault(identity_id, []).append(track_id)
    queries = [
        track_id
        for members in tracks_by_identity.values()
        if len(members) > 1
        for track_id in members
    ]
    if not queries:
        return None, None, 0
    top1_hits = 0
    top5_hits = 0
    for query in queries:
        ranked = []
        for pair, score in candidate_scores.items():
            if query not in pair:
                continue
            candidate = pair[1] if pair[0] == query else pair[0]
            ranked.append((score, candidate))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        relevant = {
            track_id
            for track_id, identity_id in labels.items()
            if track_id != query and identity_id == labels[query]
        }
        top1_hits += bool(ranked and ranked[0][1] in relevant)
        top5_hits += any(candidate in relevant for _, candidate in ranked[:5])
    return top1_hits / len(queries), top5_hits / len(queries), len(queries)


def _retrieval_metrics(
    labels: dict[str, str],
    candidate_scores: dict[tuple[str, str], float],
) -> tuple[float | None, float | None, float | None, int]:
    members_by_identity: dict[str, list[str]] = {}
    for track_id, identity_id in labels.items():
        members_by_identity.setdefault(identity_id, []).append(track_id)
    queries = [
        track_id
        for members in members_by_identity.values()
        if len(members) > 1
        for track_id in members
    ]
    if not queries:
        return None, None, None, 0

    top1_hits = 0
    top5_hits = 0
    average_precisions = []
    for query in queries:
        relevant = {
            track_id
            for track_id, identity_id in labels.items()
            if track_id != query and identity_id == labels[query]
        }
        ranked = []
        for pair, score in candidate_scores.items():
            if query not in pair:
                continue
            candidate = pair[1] if pair[0] == query else pair[0]
            ranked.append((score, candidate))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        ranked_candidates = [candidate for _, candidate in ranked]
        top1_hits += bool(ranked_candidates and ranked_candidates[0] in relevant)
        top5_hits += any(candidate in relevant for candidate in ranked_candidates[:5])
        hits = 0
        precision_sum = 0.0
        for rank, candidate in enumerate(ranked_candidates, start=1):
            if candidate in relevant:
                hits += 1
                precision_sum += hits / rank
        average_precisions.append(precision_sum / len(relevant))
    count = len(queries)
    return (
        top1_hits / count,
        top5_hits / count,
        sum(average_precisions) / count,
        count,
    )
