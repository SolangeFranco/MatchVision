from __future__ import annotations

import pytest

from matchvision.evaluation.reid_metrics import (
    ReIDMetricsError,
    evaluate_reid_methods,
    evaluate_reid_proposals,
)


def _review_export() -> dict:
    evidence = {
        "p12": {1: 10, 2: 20},
        "p13": {1: 10, 3: 30},
        "p24": {2: 20, 4: 40},
    }

    def track_record(track_id: int, evidence_ids: list[str]) -> dict:
        records = [
            {
                "proposal_id": proposal_id,
                "decision": "human_confirmed_identity_assignment",
                "proposal_file_sha256": "b" * 64,
                "evidence_frames": [evidence[proposal_id][track_id]],
            }
            for proposal_id in evidence_ids
        ]
        return {
            "track_id": track_id,
            "status": "validated_manual",
            "decision": "human_confirmed_identity_assignment",
            "evidence_frames": sorted({
                frame for item in records for frame in item["evidence_frames"]
            }),
            "evidence_proposals": records,
        }

    proposal_track_ids = {
        "p12": [1, 2],
        "p13": [1, 3],
        "p24": [2, 4],
    }
    return {
        "record_type": "human_review_with_manual_ground_truth",
        "video": {"sha256": "a" * 64},
        "proposal_file_sha256": "b" * 64,
        "identities": [
            {
                "identity_id": "jugador-01",
                "status": "validated_manual",
                "validation_source": "explicit_human_track_assignment",
                "tracks": [
                    track_record(1, ["p12", "p13"]),
                    track_record(2, ["p12", "p24"]),
                ],
            },
            {
                "identity_id": "jugador-02",
                "status": "validated_manual",
                "validation_source": "explicit_human_track_assignment",
                "tracks": [
                    track_record(3, ["p13"]),
                    track_record(4, ["p24"]),
                ],
            },
        ],
        "review_records": [
            {
                "proposal_id": proposal_id,
                "decision": "accepted",
                "decision_source": "human_review",
                "track_ids": track_ids,
                "frames": {
                    "p12": [10, 20],
                    "p13": [10, 30],
                    "p24": [20, 40],
                }[proposal_id],
                "case_id": "case-1",
            }
            for proposal_id, track_ids in proposal_track_ids.items()
        ],
    }


def _proposals() -> dict:
    return {
        "schema_version": 2,
        "video": {"source_sha256": "a" * 64},
        "cases": [
            {
                "candidate_associations": [
                    {
                        "proposal_id": "p12",
                        "case_id": "case-1",
                        "relation": "candidate_same_player",
                        "track_ids": [1, 2],
                        "frames": [10, 20],
                        "heuristic_score": 0.9,
                    },
                    {
                        "proposal_id": "p13",
                        "case_id": "case-1",
                        "relation": "existing_reid_link",
                        "track_ids": [1, 3],
                        "frames": [10, 30],
                        "heuristic_score": 0.8,
                    },
                    {
                        "proposal_id": "p24",
                        "case_id": "case-1",
                        "relation": "candidate_same_player",
                        "track_ids": [2, 4],
                        "frames": [20, 40],
                        "heuristic_score": 0.85,
                    },
                ]
            }
        ],
    }


def test_evaluates_only_manually_validated_track_labels() -> None:
    report = evaluate_reid_proposals(
        _review_export(),
        _proposals(),
        actual_proposals_sha256="b" * 64,
    )

    assert report["status"] == "sufficient_for_proposal_conditioned_diagnostic"
    assert report["counts"] == {
        "validated_tracks": 4,
        "validated_identities": 2,
        "labeled_track_pairs": 6,
        "positive_identity_pairs": 2,
        "negative_identity_pairs": 4,
        "candidate_pairs_with_scores": 3,
        "retrieval_queries": 4,
    }
    assert report["confusion_matrix"] == {
        "same_identity": {"predicted_link": 1, "not_predicted": 1},
        "different_identity": {"predicted_link": 2, "not_predicted": 2},
    }
    assert report["metrics"]["precision"]["value"] == pytest.approx(1 / 3)
    assert report["metrics"]["recall"]["value"] == pytest.approx(0.5)
    assert report["metrics"]["f1"]["value"] == pytest.approx(0.4)
    assert report["metrics"]["false_positive_rate"]["value"] == pytest.approx(0.5)
    assert report["metrics"]["false_negative_rate"]["value"] == pytest.approx(0.5)
    assert report["metrics"]["top1_accuracy"]["value"] == pytest.approx(0.5)
    assert report["metrics"]["top5_accuracy"]["value"] == pytest.approx(0.5)
    assert report["evaluation_scope"]["type"] == "diagnostico_condicionado_a_propuestas"
    assert report["evaluation_scope"]["generalization_claim_allowed"] is False


def test_accepted_proposal_without_identity_assignment_is_not_ground_truth() -> None:
    review = _review_export()
    review["identities"] = []

    report = evaluate_reid_proposals(
        review, _proposals(), actual_proposals_sha256="b" * 64
    )

    assert report["status"] == "insufficient_ground_truth"
    assert report["counts"]["validated_tracks"] == 0
    assert report["counts"]["candidate_pairs_with_scores"] == 3
    assert report["metrics"]["precision"]["available"] is False
    assert report["metrics"]["recall"]["available"] is False
    assert report["missing_data"]


def test_rejects_duplicate_track_identity_assignment() -> None:
    review = _review_export()
    review["identities"][1]["tracks"][0]["track_id"] = "2"

    with pytest.raises(ReIDMetricsError, match="asignado a más de una identidad"):
        evaluate_reid_proposals(
            review, _proposals(), actual_proposals_sha256="b" * 64
        )


def test_rejects_hash_mismatch() -> None:
    with pytest.raises(ReIDMetricsError, match="SHA-256 del archivo de propuestas"):
        evaluate_reid_proposals(
            _review_export(), _proposals(), actual_proposals_sha256="c" * 64
        )


def test_rejects_video_hash_mismatch() -> None:
    proposals = _proposals()
    proposals["video"]["source_sha256"] = "d" * 64

    with pytest.raises(ReIDMetricsError, match="SHA-256 del video"):
        evaluate_reid_proposals(
            _review_export(), proposals, actual_proposals_sha256="b" * 64
        )


def test_rejects_test_only_identity_as_ground_truth() -> None:
    review = _review_export()
    review["identities"][0]["identity_id"] = "TEST-ONLY-REID-E2E"

    with pytest.raises(ReIDMetricsError, match="reservada para pruebas"):
        evaluate_reid_proposals(
            review, _proposals(), actual_proposals_sha256="b" * 64
        )


def _independent_review_export() -> dict:
    review = _review_export()
    for identity in review["identities"]:
        identity["validation_source"] = "independent_human_track_annotations"
        for track in identity["tracks"]:
            track["annotation_source"] = "independent_human_track_annotation"
            track["evidence_proposals"] = []
    return review


def _method_proposals() -> dict:
    proposals = _proposals()
    values = {
        "p12": {"hsv_hog": 0.91, "embedding": 0.79, "hybrid": 0.88},
        "p13": {"hsv_hog": 0.50, "embedding": 0.88, "hybrid": 0.60},
        "p24": {"hsv_hog": 0.80, "embedding": 0.85, "hybrid": 0.81},
    }
    for case in proposals["cases"]:
        for candidate in case["candidate_associations"]:
            candidate["method_scores"] = values[candidate["proposal_id"]]
    return proposals


def test_compares_methods_against_independent_track_annotations() -> None:
    report = evaluate_reid_methods(
        _independent_review_export(),
        _method_proposals(),
        actual_proposals_sha256="b" * 64,
    )

    assert report["status"] == "sufficient_for_candidate_conditioned_diagnostic"
    assert report["counts"]["independently_validated_tracks"] == 4
    assert report["evaluation_scope"]["generalization_claim_allowed"] is False
    assert report["methods"]["hsv_hog"]["confusion_matrix"] == {
        "same_identity": {"predicted_link": 1, "not_predicted": 1},
        "different_identity": {"predicted_link": 0, "not_predicted": 4},
    }
    assert report["methods"]["hsv_hog"]["metrics"]["precision"]["value"] == 1
    assert report["methods"]["hsv_hog"]["metrics"]["recall"]["value"] == 0.5
    assert report["methods"]["hsv_hog"]["metrics"]["top1_accuracy"]["value"] == 0.5
    assert report["methods"]["hsv_hog"]["metrics"]["mean_average_precision"]["value"] == 0.5
    assert report["methods"]["embedding"]["confusion_matrix"] == {
        "same_identity": {"predicted_link": 0, "not_predicted": 2},
        "different_identity": {"predicted_link": 2, "not_predicted": 2},
    }


def test_independent_comparison_ignores_proposal_supported_assignments() -> None:
    review = _independent_review_export()
    review["identities"][0]["tracks"][0]["annotation_source"] = (
        "proposal_supported_human_assignment"
    )

    report = evaluate_reid_methods(review, _method_proposals())

    assert report["counts"]["independently_validated_tracks"] == 3


def test_independent_comparison_rejects_test_identity() -> None:
    review = _independent_review_export()
    review["identities"][0]["identity_id"] = "TEST-ONLY-REID-E2E"

    with pytest.raises(ReIDMetricsError, match="Ground Truth requiere identidades"):
        evaluate_reid_methods(review, _method_proposals())
