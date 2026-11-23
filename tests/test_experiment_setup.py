import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from matchvision.evaluation.cli import main as evaluate_main
from matchvision.evaluation.mot import MOTDataError, evaluate_mot_sequence, load_mot_file
from matchvision.experiments.config import ConfigurationError, load_benchmark_config
from matchvision.experiments.manifest import ManifestError, load_sequence_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence_manifest(directory: Path, sequences: list[dict[str, object]]) -> Path:
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "dataset": "test-dataset",
                "release": "test-release",
                "sequences": sequences,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _sequence_entry(
    directory: Path, game_id: str, sequence_id: str, split: str
) -> dict[str, object]:
    video_path = directory / f"{sequence_id}.mp4"
    annotation_path = directory / f"{sequence_id}.txt"
    video_path.write_bytes(b"synthetic-video-placeholder")
    annotation_path.write_text("1,1,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8")
    return {
        "game_id": game_id,
        "sequence_id": sequence_id,
        "split": split,
        "video_path": video_path.name,
        "annotation_path": annotation_path.name,
        "frame_count": 1,
        "fps": 25,
        "width": 32,
        "height": 24,
        "video_sha256": _sha256(video_path),
        "annotation_sha256": _sha256(annotation_path),
    }


def test_repository_benchmark_config_validates() -> None:
    config = load_benchmark_config(PROJECT_ROOT / "configs" / "benchmark.yaml")

    assert config["experiment"]["dataset"]["split_by"] == "match"


def test_invalid_yaml_config_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("experiment: []\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid benchmark configuration"):
        load_benchmark_config(bad_config)


def test_manifest_validates_and_selects_deterministically(tmp_path: Path) -> None:
    entries = [
        _sequence_entry(tmp_path, "game-b", "seq-b", "validation"),
        _sequence_entry(tmp_path, "game-a", "seq-a", "validation"),
    ]
    manifest_path = _sequence_manifest(tmp_path, entries)
    manifest = load_sequence_manifest(
        manifest_path, require_hashes=True, verify_files=True
    )

    assert [item.sequence_id for item in manifest.select("validation", limit=1)] == [
        "seq-a"
    ]


def test_manifest_rejects_game_leakage_across_splits(tmp_path: Path) -> None:
    entries = [
        _sequence_entry(tmp_path, "same-game", "seq-train", "train"),
        _sequence_entry(tmp_path, "same-game", "seq-val", "validation"),
    ]
    manifest_path = _sequence_manifest(tmp_path, entries)

    with pytest.raises(ManifestError, match="appears in both"):
        load_sequence_manifest(manifest_path)


def test_manifest_verifies_declared_hashes(tmp_path: Path) -> None:
    entry = _sequence_entry(tmp_path, "game-a", "seq-a", "validation")
    video_path = tmp_path / str(entry["video_path"])
    video_path.write_bytes(b"changed")
    manifest_path = _sequence_manifest(tmp_path, [entry])

    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        load_sequence_manifest(manifest_path, verify_files=True)


def test_mot_loader_rejects_duplicate_identity_per_frame(tmp_path: Path) -> None:
    annotation_path = tmp_path / "gt.txt"
    annotation_path.write_text(
        "1,7,0,0,10,10,1,-1,-1,-1\n1,7,1,1,10,10,1,-1,-1,-1\n",
        encoding="utf-8",
    )

    with pytest.raises(MOTDataError, match="Duplicate track ID"):
        load_mot_file(annotation_path, frame_count=1)


def test_trackeval_metrics_score_perfect_mot_sequence(tmp_path: Path) -> None:
    ground_truth = tmp_path / "gt.txt"
    predictions = tmp_path / "pred.txt"
    row = "1,7,0,0,10,10,1,-1,-1,-1\n2,7,1,0,10,10,1,-1,-1,-1\n"
    ground_truth.write_text(row, encoding="utf-8")
    predictions.write_text(row, encoding="utf-8")

    metrics = evaluate_mot_sequence(ground_truth, predictions, frame_count=2)

    assert metrics["HOTA"] == pytest.approx(1.0)
    assert metrics["MOTA"] == pytest.approx(1.0)
    assert metrics["IDF1"] == pytest.approx(1.0)
    assert metrics["ID_switches"] == 0


def test_trackeval_metrics_detect_identity_switch(tmp_path: Path) -> None:
    ground_truth = tmp_path / "gt.txt"
    predictions = tmp_path / "pred.txt"
    ground_truth.write_text(
        "1,7,0,0,10,10,1,-1,-1,-1\n2,7,0,0,10,10,1,-1,-1,-1\n",
        encoding="utf-8",
    )
    predictions.write_text(
        "1,5,0,0,10,10,1,-1,-1,-1\n2,6,0,0,10,10,1,-1,-1,-1\n",
        encoding="utf-8",
    )

    metrics = evaluate_mot_sequence(ground_truth, predictions, frame_count=2)

    assert metrics["ID_switches"] == 1
    assert metrics["MOTA"] == pytest.approx(0.5)
    assert metrics["IDF1"] == pytest.approx(0.5)


def test_cli_saves_reproducible_json_and_csv(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    entry = _sequence_entry(data_dir, "game-a", "seq-a", "validation")
    gt_path = data_dir / str(entry["annotation_path"])
    gt_path.write_text("1,7,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8")
    entry["annotation_sha256"] = _sha256(gt_path)
    manifest_path = _sequence_manifest(data_dir, [entry])

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "seq-a.txt").write_text(
        gt_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "benchmark.yaml").read_text(encoding="utf-8")
    )
    config["experiment"]["dataset"]["sequence_manifest"] = str(manifest_path)
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_dir = tmp_path / "results"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matchvision-evaluate",
            "--config",
            str(config_path),
            "--predictions",
            str(predictions_dir),
            "--output",
            str(output_dir),
            "--limit",
            "1",
            "--detector-version",
            "detector-test",
            "--tracker-version",
            "tracker-test",
        ],
    )
    evaluate_main()

    json_files = list(output_dir.glob("*.json"))
    csv_files = list(output_dir.glob("*.csv"))
    assert len(json_files) == len(csv_files) == 1
    record = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert record["dataset"]["name"] == "test-dataset"
    assert record["dataset"]["release"] == "test-release"
    assert record["metrics"]["macro_average_by_sequence"]["IDF1"] == pytest.approx(1)
    assert record["models"]["detector_version"] == "detector-test"
    assert record["models"]["tracker_version"] == "tracker-test"
    assert record["timings"]["total_seconds"] >= 0
