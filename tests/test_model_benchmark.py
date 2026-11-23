from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from matchvision.evaluation.benchmark import build_model_benchmark
from matchvision.evaluation.identity_annotations import (
    freeze_identity_annotations,
    ground_truth_content_sha256,
    load_identity_annotations,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 64)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable")
    try:
        for _ in range(2):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    video = tmp_path / "synthetic.avi"
    _write_video(video)
    video_sha = _sha256(video)
    payload = {
        "schema_version": 1,
        "video": {
            "path": str(video),
            "sha256": video_sha,
            "frame_count": 2,
            "width": 64,
            "height": 64,
            "fps": 5.0,
        },
        "exhaustive_frames": [1, 2],
        "frames": [
            {
                "frame": frame,
                "exhaustive": True,
                "observations": [
                    {
                        "annotation_id": f"player-{frame}",
                        "identity_id": "player-1",
                        "entity_type": "player",
                        "x1": 10,
                        "y1": 10,
                        "x2": 30,
                        "y2": 50,
                        "unknown": False,
                        "ambiguous": False,
                        "ignore": False,
                    }
                ],
            }
            for frame in (1, 2)
        ],
    }
    annotations = tmp_path / "ground-truth.json"
    annotations.write_text(
        json.dumps(freeze_identity_annotations(payload)),
        encoding="utf-8",
    )
    run_names = {
        "yolo_bytetrack": None,
        "hsv_hog": "hsv_hog",
        "osnet": "embedding",
        "hybrid": "hybrid",
    }
    run_paths: dict[str, Path] = {}
    for run_name, method in run_names.items():
        mot = tmp_path / f"{run_name}.txt"
        mot.write_text(
            "1,1,10,10,20,40,1,-1,-1,-1\n"
            "2,1,10,10,20,40,1,-1,-1,-1\n",
            encoding="utf-8",
        )
        detections = tmp_path / f"{run_name}-detections.csv"
        with detections.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "frame",
                    "class_id",
                    "label",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                ],
            )
            writer.writeheader()
            for frame in (1, 2):
                writer.writerow(
                    {
                        "frame": frame,
                        "class_id": 0,
                        "label": "player",
                        "confidence": 0.9,
                        "x1": 10,
                        "y1": 10,
                        "x2": 30,
                        "y2": 50,
                    }
                )
        tracks = tmp_path / f"{run_name}-tracks.csv"
        tracks.write_text(
            "frame,track_id,temporal_track_id,persistent_id,class_id,label,confidence,x1,y1,x2,y2\n"
            "1,1,1,,0,player,0.9,10,10,30,50\n"
            "2,1,1,,0,player,0.9,10,10,30,50\n",
            encoding="utf-8",
        )
        outputs = {
            "mot": str(mot),
            "mot_sha256": _sha256(mot),
            "detections_csv": str(detections),
            "detections_csv_sha256": _sha256(detections),
            "tracking_observations_csv": str(tracks),
            "tracking_observations_csv_sha256": _sha256(tracks),
        }
        config: dict[str, object] = {
            "detector": {"confidence": 0.25, "image_size": 640, "device": "cpu"},
            "tracker": {"threshold": 0.8},
        }
        if method:
            config["identity_association"] = {"reid_method": method}
        run = {
            "schema_version": 1,
            "run_id": run_name,
            "input": {"video_sha256": video_sha, "frames_processed": 2},
            "configuration": config,
            "model": {
                "name": "yolo-test.pt",
                "version": "test-version",
                "checkpoint_sha256": "a" * 64,
            },
            "tracker": {"name": "ByteTrack", "version": "test-version"},
            "timings": {"processing_seconds": 1.0, "processing_fps": 2.0},
            "environment": {"python": "test"},
            "outputs": outputs,
        }
        run_path = tmp_path / f"{run_name}.json"
        run_path.write_text(json.dumps(run), encoding="utf-8")
        run_paths[run_name] = run_path
    return video, annotations, run_paths


def test_benchmark_compares_four_runs_against_identical_frozen_ground_truth(
    tmp_path: Path,
) -> None:
    video, annotations, runs = _prepare_inputs(tmp_path)

    report = build_model_benchmark(
        annotations,
        video,
        runs,
        class_map={"player": "player"},
        dataset_kind="synthetic",
    )

    assert report["evaluation"]["identical_ground_truth_for_all_runs"] is True
    assert report["ground_truth"]["mutation_policy"] == "read_only"
    assert report["validation_conclusion"].startswith("Prueba sintética")
    assert len(report["comparison_table"]) == 4
    for row in report["comparison_table"]:
        assert row["precision_deteccion"] == pytest.approx(1.0)
        assert row["HOTA"] == pytest.approx(1.0)
        assert row["IDF1"] == pytest.approx(1.0)
        assert row["Rank-1"] is None


def test_benchmark_rejects_mismatched_detector_configuration(tmp_path: Path) -> None:
    video, annotations, runs = _prepare_inputs(tmp_path)
    path = runs["osnet"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["configuration"]["detector"]["confidence"] = 0.5
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="exactamente el mismo detector"):
        build_model_benchmark(
            annotations,
            video,
            runs,
            class_map={"player": "player"},
            dataset_kind="synthetic",
        )


def test_benchmark_keeps_appearance_retrieval_separate_and_maps_methods(
    tmp_path: Path,
) -> None:
    video, annotations, runs = _prepare_inputs(tmp_path)
    ground_truth = load_identity_annotations(annotations)
    methods = {
        "hsv_hog": {"available": True, "global_retrieval": {"available": True, "rank1": 0.8, "mean_average_precision": 0.7}},
        "embedding": {"available": True, "global_retrieval": {"available": True, "rank1": 0.9, "mean_average_precision": 0.85}},
        "hybrid": {"available": True, "global_retrieval": {"available": True, "rank1": 0.85, "mean_average_precision": 0.8}},
    }
    retrieval_report = {
        "video_sha256": ground_truth["video"]["sha256"],
        "ground_truth_content_sha256": ground_truth_content_sha256(ground_truth),
        "methods": methods,
    }

    report = build_model_benchmark(
        annotations,
        video,
        runs,
        class_map={"player": "player"},
        dataset_kind="synthetic",
        appearance_retrieval_report=retrieval_report,
    )

    rows = {row["configuracion"]: row for row in report["comparison_table"]}
    assert rows["yolo_bytetrack"]["Rank-1"] is None
    assert rows["hsv_hog"]["Rank-1"] == pytest.approx(0.8)
    assert rows["osnet"]["Rank-1"] == pytest.approx(0.9)
    assert rows["hybrid"]["mAP"] == pytest.approx(0.8)
    assert report["results"]["osnet"]["appearance_retrieval"] == methods["embedding"]
