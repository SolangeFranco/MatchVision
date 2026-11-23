"""CLI for local video detection and ByteTrack tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from matchvision.detection.ultralytics_yolo import UltralyticsYOLODetector
from matchvision.pipeline.video import process_video
from matchvision.reidentification.appearance import PersistentIdentityAssociator
from matchvision.tracking.bytetrack import ByteTrackAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta objetos y ejecuta tracking en un video local."
    )
    parser.add_argument("video", type=Path, help="Ruta al video de entrada")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/video-runs"),
        help="Carpeta para los resultados (predeterminada: outputs/video-runs)",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="Pesos del detector")
    parser.add_argument("--model-source", help="Procedencia de los pesos")
    parser.add_argument("--model-license", help="Licencia de los pesos")
    parser.add_argument("--pretrained-on", help="Dataset de preentrenamiento")
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Confianza mínima"
    )
    parser.add_argument(
        "--image-size", type=int, default=1280, help="Tamaño de inferencia"
    )
    parser.add_argument(
        "--device", help="Dispositivo de inferencia; por ejemplo cpu, 0 o cuda:0"
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        help="IDs de clase; por defecto selecciona personas y balón si existen",
    )
    parser.add_argument(
        "--ground-truth", type=Path, help="Anotaciones MOT de referencia"
    )
    parser.add_argument(
        "--tracker-activation-threshold",
        type=float,
        default=0.25,
        help="Umbral de activación de tracks",
    )
    parser.add_argument(
        "--lost-track-buffer",
        type=int,
        default=30,
        help="Fotogramas que se conservan los tracks perdidos",
    )
    parser.add_argument(
        "--matching-threshold",
        type=float,
        default=0.8,
        help="Umbral mínimo de asociación",
    )
    parser.add_argument(
        "--persistent-identities",
        action="store_true",
        help="Propone asociaciones conservadoras de apariencia/movimiento sobre ByteTrack",
    )
    parser.add_argument(
        "--reid-method",
        choices=("hsv_hog", "embedding", "hybrid"),
        help="Método opcional de asociación; por defecto usa el configurado en YAML",
    )
    parser.add_argument(
        "--reid-class-ids",
        type=int,
        nargs="+",
        help="IDs de clase elegibles para ReID; reemplaza la configuración YAML",
    )
    parser.add_argument(
        "--reid-config",
        type=Path,
        default=Path("configs/reidentification.yaml"),
        help="Ruta a la configuración YAML de ReID",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Carpeta durable para caché segmentada de detecciones; al reanudar "
            "se reproduce el video desde el inicio para reconstruir tracking y ReID"
        ),
    )
    parser.add_argument(
        "--checkpoint-segment-frames",
        type=int,
        default=300,
        help="Fotogramas por segmento de caché (predeterminado: 300)",
    )
    args = parser.parse_args()

    try:
        from matchvision.video import get_video_info

        get_video_info(args.video)
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
        tracker = ByteTrackAdapter(
            detector.class_names,
            track_activation_threshold=args.tracker_activation_threshold,
            lost_track_buffer=args.lost_track_buffer,
            minimum_matching_threshold=args.matching_threshold,
        )
        identity_assigner = None
        if args.persistent_identities:
            people_class_ids = tuple(
                class_id
                for class_id, label in detector.class_names.items()
                if label.lower().replace("_", " ")
                in {"person", "player", "football player", "referee", "goalkeeper"}
            )
            config_path = args.reid_config.expanduser().resolve()
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"ReID configuration does not exist: {config_path}"
                )
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as error:
                raise ValueError(f"Invalid ReID YAML configuration: {error}") from error
            if not isinstance(config, dict):
                raise ValueError("ReID configuration must be a YAML mapping")
            allowed = {
                "reid_class_ids",
                "maximum_reidentification_gap",
                "active_identity_protection",
                "minimum_association_confidence",
                "ambiguity_margin",
                "maximum_normalized_motion",
                "history_length",
                "scene_cut_mean_difference",
                "scene_cut_histogram_distance",
                "reid_method",
                "embedding_weights_path",
                "embedding_device",
                "embedding_batch_size",
                "embedding_gallery_size",
                "embedding_sample_interval",
                "embedding_weight",
                "minimum_visual_similarity",
            }
            unknown = set(config) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown ReID configuration keys: {sorted(unknown)}"
                )
            configured_classes = config.get("reid_class_ids")
            if configured_classes is not None and (
                not isinstance(configured_classes, list)
                or not all(isinstance(item, int) for item in configured_classes)
            ):
                raise ValueError("reid_class_ids must be a list of integer class IDs")
            config["reid_class_ids"] = tuple(
                args.reid_class_ids
                or config.get("reid_class_ids")
                or people_class_ids
            )
            if args.reid_method is not None:
                config["reid_method"] = args.reid_method
            identity_assigner = PersistentIdentityAssociator(**config)
        result = process_video(
            args.video,
            args.output,
            detector,
            tracker,
            ground_truth_path=args.ground_truth,
            identity_assigner=identity_assigner,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_segment_frames=args.checkpoint_segment_frames,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
        return

    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "frames_processed": result.frames_processed,
                "tracked_detections": result.detections_tracked,
                "processing_seconds": result.processing_seconds,
                "processing_fps": result.processing_fps,
                "mot": str(result.mot_path.resolve()),
                "annotated_video": str(result.annotated_video_path.resolve()),
                "detections": (
                    str(result.detections_path.resolve())
                    if result.detections_path
                    else None
                ),
                "tracking_observations": (
                    str(result.track_observations_path.resolve())
                    if result.track_observations_path
                    else None
                ),
                "metadata": str(result.metadata_path.resolve()),
                "trajectories": (
                    str(result.trajectory_path.resolve())
                    if result.trajectory_path
                    else None
                ),
                "tracking_stats": (
                    str(result.tracking_stats_path.resolve())
                    if result.tracking_stats_path
                    else None
                ),
                "metrics": result.metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
