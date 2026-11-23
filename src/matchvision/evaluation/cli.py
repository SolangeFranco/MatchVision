"""CLI for reproducible evaluation of existing MOT prediction files."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from matchvision.evaluation.mot import evaluate_mot_sequence
from matchvision.experiments.config import load_benchmark_config
from matchvision.experiments.manifest import load_sequence_manifest
from matchvision.experiments.registry import create_experiment_record, save_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa predicciones MOT existentes; no ejecuta inferencia."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Carpeta con un archivo MOT <sequence_id>.txt por secuencia",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--detector-version", default="unspecified", help="Versión del detector"
    )
    parser.add_argument(
        "--tracker-version", default="unspecified", help="Versión del tracker"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evalúa las primeras N secuencias en orden estable de partido/secuencia",
    )
    args = parser.parse_args()

    try:
        config_path = args.config.resolve()
        config = load_benchmark_config(config_path)
        manifest_path = args.manifest or Path(
            config["experiment"]["dataset"]["sequence_manifest"]
        )
        manifest = load_sequence_manifest(
            manifest_path,
            require_hashes=config["experiment"]["dataset"]["sequence_hashes_required"],
            verify_files=True,
        )
        split = config["experiment"]["dataset"]["split"]
        sequences = manifest.select(split, args.limit)
        evaluated: list[dict[str, object]] = []
        run_started = time.perf_counter()
        for sequence in sequences:
            prediction_path = args.predictions / f"{sequence.sequence_id}.txt"
            if not prediction_path.is_file():
                raise FileNotFoundError(
                    f"Prediction file missing for {sequence.sequence_id}: {prediction_path}"
                )
            sequence_started = time.perf_counter()
            metrics = evaluate_mot_sequence(
                sequence.annotation_path,
                prediction_path,
                sequence.frame_count,
            )
            evaluated.append(
                {
                    "game_id": sequence.game_id,
                    "sequence_id": sequence.sequence_id,
                    "frame_count": sequence.frame_count,
                    "fps": sequence.fps,
                    "width": sequence.width,
                    "height": sequence.height,
                    "video_sha256": sequence.video_sha256,
                    "annotation_sha256": _hash_file(sequence.annotation_path),
                    "prediction_path": str(prediction_path.resolve()),
                    "prediction_sha256": _hash_file(prediction_path),
                    "metrics": metrics,
                    "elapsed_seconds": time.perf_counter() - sequence_started,
                }
            )
        total_seconds = time.perf_counter() - run_started
        output_dir = args.output or Path(config["outputs"]["directory"])
        record = create_experiment_record(
            config_path=config_path,
            config=config,
            manifest_path=manifest.path,
            manifest_sha256=manifest.sha256,
            manifest_dataset=manifest.dataset,
            manifest_release=manifest.release,
            sequences=evaluated,
            total_seconds=total_seconds,
            detector_version=args.detector_version,
            tracker_version=args.tracker_version,
        )
        saved_path = save_experiment(output_dir, record)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
        return

    print(
        json.dumps(
            {
                "experiment_id": record["experiment_id"],
                "sequences_evaluated": len(evaluated),
                "results_json": str(saved_path.resolve()),
                "results_csv": str(saved_path.with_suffix(".csv").resolve()),
                "macro_average_by_sequence": record["metrics"]["macro_average_by_sequence"],
            },
            indent=2,
        )
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
