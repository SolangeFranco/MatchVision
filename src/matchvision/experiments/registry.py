"""Structured, append-only experiment result storage."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def create_experiment_record(
    *,
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    manifest_dataset: str,
    manifest_release: str,
    sequences: list[dict[str, Any]],
    total_seconds: float,
    detector_version: str,
    tracker_version: str,
) -> dict[str, Any]:
    """Create a JSON-compatible record with inputs, versions, scores and timings."""
    dataset_config = config["experiment"]["dataset"]
    return {
        "schema_version": 1,
        "experiment_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "name": config["experiment"]["name"],
            "seed": config["experiment"]["random_seed"],
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256_file(config_path),
            "config": config,
        },
        "dataset": {
            "name": manifest_dataset,
            "release": manifest_release,
            "split": dataset_config["split"],
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "sequence_count": len(sequences),
            "sequences": sequences,
        },
        "models": {
            "detectors": config["detectors"],
            "trackers": config["trackers"],
            "detector_version": detector_version,
            "tracker_version": tracker_version,
        },
        "metrics": {
            "macro_average_by_sequence": _macro_average(sequences),
            "per_sequence": [
                {"sequence_id": item["sequence_id"], **item["metrics"]}
                for item in sequences
            ],
        },
        "timings": {
            "total_seconds": total_seconds,
            "per_sequence_seconds": {
                item["sequence_id"]: item["elapsed_seconds"] for item in sequences
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "packages": {
                name: _package_version(name)
                for name in ("trackeval", "numpy", "scipy", "PyYAML", "jsonschema")
            },
            "trackeval_source_commit": "12c8791b303e0a0b50f753af204249e622d0281a",
        },
        "status": "completed",
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _macro_average(sequences: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = ("HOTA", "MOTA", "IDF1", "ID_Precision", "ID_Recall")
    return {
        metric: sum(float(item["metrics"][metric]) for item in sequences) / len(sequences)
        for metric in metric_names
    }


def save_experiment(output_dir: str | Path, record: dict[str, Any]) -> Path:
    """Atomically save experiment JSON and a per-sequence CSV score table."""
    output_path = Path(output_dir)
    experiment_id = record["experiment_id"]
    json_path = output_path / f"{experiment_id}.json"
    json_content = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(json_path, json_content)

    csv_path = output_path / f"{experiment_id}.csv"
    from io import StringIO

    stream = StringIO(newline="")
    columns = [
        "sequence_id",
        "game_id",
        "HOTA",
        "MOTA",
        "IDF1",
        "ID_Precision",
        "ID_Recall",
        "ID_switches",
        "Fragmentation",
        "ID_true_positives",
        "ID_false_positives",
        "ID_false_negatives",
        "ground_truth_detections",
        "predicted_detections",
        "elapsed_seconds",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for item in record["dataset"]["sequences"]:
        writer.writerow(
            {
                "sequence_id": item["sequence_id"],
                "game_id": item["game_id"],
                **item["metrics"],
                "elapsed_seconds": item["elapsed_seconds"],
            }
        )
    _atomic_write(csv_path, stream.getvalue())
    return json_path
