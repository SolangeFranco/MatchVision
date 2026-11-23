"""CLI para detectar y seguir el balón en un video local."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matchvision.ball.tracking import BallTrackConfig, BallTracker
from matchvision.detection.base import Detector
from matchvision.video import get_video_info, iter_video_frames


_CSV_FIELDS = [
    "fotograma",
    "segmento_id",
    "estado",
    "centro_x",
    "centro_y",
    "x1",
    "y1",
    "x2",
    "y2",
    "confianza_deteccion",
    "clase_id",
    "etiqueta_modelo",
    "fotogramas_sin_deteccion",
    "reaparecido",
]


@dataclass(frozen=True, slots=True)
class BallTrackingRun:
    run_id: str
    frames_processed: int
    observed_frames: int
    predicted_frames: int
    missing_frames: int
    processing_seconds: float
    csv_path: Path
    manifest_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spanish_status(status: str) -> str:
    return {
        "observed": "detectada",
        "predicted": "predicha",
        "missing": "sin_deteccion",
    }[status]


def process_ball_video(
    video_path: str | Path,
    output_csv: str | Path,
    detector: Detector,
    tracker: BallTracker,
) -> BallTrackingRun:
    """Procesa un video y guarda posiciones observadas y predichas del balón."""
    source = Path(video_path).expanduser().resolve()
    destination = Path(output_csv).expanduser().resolve()
    manifest_path = destination.with_suffix(destination.suffix + ".json")
    if not source.is_file():
        raise FileNotFoundError(f"No existe el video de entrada: {source}")
    if destination.suffix.lower() != ".csv":
        raise ValueError("La salida de seguimiento debe tener extensión .csv")
    if destination == source or manifest_path == source:
        raise ValueError("La salida no puede sobrescribir el video de entrada")
    if destination.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Se rechaza sobrescribir la salida o el manifiesto: {destination}"
        )

    video_info = get_video_info(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    temporary_csv: Path | None = None
    temporary_manifest: Path | None = None
    published_csv = False
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    frames_processed = 0
    checkpoint_path = detector.checkpoint_path

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_csv = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for frame_index, frame in iter_video_frames(source):
                observation = tracker.update(
                    detector.detect(frame),
                    frame_index=frame_index,
                    frame_shape=frame.shape[:2],
                )
                counts[observation.status] += 1
                writer.writerow(
                    {
                        "fotograma": frame_index + 1,
                        "segmento_id": (
                            "" if observation.track_id is None else observation.track_id
                        ),
                        "estado": _spanish_status(observation.status),
                        "centro_x": _number(observation.center_x),
                        "centro_y": _number(observation.center_y),
                        "x1": _number(observation.x1),
                        "y1": _number(observation.y1),
                        "x2": _number(observation.x2),
                        "y2": _number(observation.y2),
                        "confianza_deteccion": _number(observation.confidence),
                        "clase_id": (
                            "" if observation.class_id is None else observation.class_id
                        ),
                        "etiqueta_modelo": observation.label or "",
                        "fotogramas_sin_deteccion": observation.missed_frames,
                        "reaparecido": str(observation.reacquired).lower(),
                    }
                )
                frames_processed += 1
        if frames_processed == 0:
            raise ValueError("El video no contiene fotogramas decodificables")

        processing_seconds = time.perf_counter() - started
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {
                "video_path": str(source),
                "video_sha256": _sha256_file(source),
                "width": video_info.width,
                "height": video_info.height,
                "source_fps": video_info.fps,
                "declared_frame_count": video_info.frame_count,
                "frames_processed": frames_processed,
            },
            "model": {
                "name": detector.model_name,
                "version": detector.model_version,
                "source": detector.model_source,
                "license": detector.model_license,
                "pretrained_on": detector.pretrained_on,
                "checkpoint_path": (
                    str(checkpoint_path) if checkpoint_path is not None else None
                ),
                "checkpoint_sha256": (
                    _sha256_file(checkpoint_path)
                    if checkpoint_path is not None
                    else None
                ),
                "classes": detector.class_names,
                "configuration": detector.configuration,
            },
            "tracking": {
                "configuration": asdict(tracker.config),
                "allowed_class_ids": (
                    sorted(tracker.allowed_class_ids)
                    if tracker.allowed_class_ids is not None
                    else None
                ),
                "observed_frames": counts["observed"],
                "predicted_frames": counts["predicted"],
                "missing_frames": counts["missing"],
                "processing_seconds": processing_seconds,
            },
            "output": {
                "csv_path": str(destination),
                "csv_sha256": _sha256_file(temporary_csv),
            },
            "limitations": [
                "Las posiciones detectadas dependen del modelo y no son ground truth.",
                "Las posiciones predichas son estimaciones temporales, no detecciones.",
                "Los ID son segmentos de seguimiento del balón, no identidades de jugadores.",
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_manifest = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

        os.rename(temporary_csv, destination)
        published_csv = True
        os.rename(temporary_manifest, manifest_path)
    except Exception:
        if published_csv:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if temporary_csv is not None:
            temporary_csv.unlink(missing_ok=True)
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)

    return BallTrackingRun(
        run_id=run_id,
        frames_processed=frames_processed,
        observed_frames=counts["observed"],
        predicted_frames=counts["predicted"],
        missing_frames=counts["missing"],
        processing_seconds=processing_seconds,
        csv_path=destination,
        manifest_path=manifest_path,
    )


def _number(value: float | None) -> str | float:
    return "" if value is None else round(value, 3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta y sigue el balón en un video local."
    )
    parser.add_argument("video", type=Path, help="Ruta al video de entrada")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ball-tracking"),
        help="Carpeta de resultados (predeterminada: outputs/ball-tracking)",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Pesos de Ultralytics; se permiten pesos especializados con procedencia",
    )
    parser.add_argument("--model-source", help="Procedencia de los pesos")
    parser.add_argument("--model-license", help="Licencia de los pesos")
    parser.add_argument("--pretrained-on", help="Datos de entrenamiento declarados")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--device", help="Por ejemplo: cpu, 0 o cuda:0")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="IDs de clase que se inferirán con el modelo",
    )
    parser.add_argument(
        "--ball-class-ids",
        type=int,
        nargs="+",
        help="IDs de clase que representan el balón en los pesos elegidos",
    )
    parser.add_argument("--maximum-missed-frames", type=int, default=8)
    parser.add_argument(
        "--maximum-speed-pixels-per-frame", type=float, default=180.0
    )
    args = parser.parse_args()

    try:
        from matchvision.detection.ultralytics_yolo import UltralyticsYOLODetector

        detector = UltralyticsYOLODetector(
            args.model,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
            classes=args.classes,
            model_source=args.model_source,
            model_license=args.model_license,
            pretrained_on=args.pretrained_on,
        )
        if args.ball_class_ids is not None:
            unknown_ids = set(args.ball_class_ids) - set(detector.class_names)
            if unknown_ids:
                raise ValueError(
                    f"El modelo no tiene las clases de balón {sorted(unknown_ids)}"
                )
        elif not any(
            name.lower().replace("_", " ").strip()
            in {"ball", "football", "soccer ball", "sports ball"}
            for name in detector.class_names.values()
        ):
            raise ValueError(
                "El modelo no declara una clase de balón reconocida; "
                "indica --ball-class-ids para seleccionar la clase correcta."
            )

        tracker = BallTracker(
            BallTrackConfig(
                minimum_confidence=args.confidence,
                maximum_missed_frames=args.maximum_missed_frames,
                maximum_speed_pixels_per_frame=(
                    args.maximum_speed_pixels_per_frame
                ),
            ),
            allowed_class_ids=args.ball_class_ids,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        output_csv = args.output / f"{uuid.uuid4()}_ball_tracks.csv"
        result = process_ball_video(args.video, output_csv, detector, tracker)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
        return

    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "fotogramas_procesados": result.frames_processed,
                "fotogramas_con_deteccion": result.observed_frames,
                "fotogramas_predichos": result.predicted_frames,
                "fotogramas_sin_deteccion": result.missing_frames,
                "segundos_de_procesamiento": round(result.processing_seconds, 3),
                "salida_csv": str(result.csv_path),
                "manifiesto": str(result.manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
