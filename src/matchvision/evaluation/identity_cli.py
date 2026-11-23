"""CLI for comparing tracking runs against manually labeled identity frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matchvision.evaluation.identity_annotations import (
    evaluate_identity_predictions,
)
from matchvision.evaluation.identity_visualization import (
    create_identity_error_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compara MOT baseline y ReID sobre los mismos fotogramas anotados "
            "exhaustivamente."
        )
    )
    parser.add_argument(
        "--annotations", type=Path, required=True, help="JSON de anotaciones humanas"
    )
    parser.add_argument("--video", type=Path, required=True, help="Video original")
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Predicciones MOT baseline"
    )
    parser.add_argument(
        "--reid", type=Path, required=True, help="Predicciones MOT con ReID"
    )
    parser.add_argument("--output", type=Path, required=True, help="Ruta del informe")
    parser.add_argument(
        "--visualizations",
        type=Path,
        help="Carpeta opcional para imágenes de error y manifiestos",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Umbral IoU para asociar predicciones y anotaciones",
    )
    args = parser.parse_args()

    try:
        output_path = args.output.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {output_path}")
        report = evaluate_identity_predictions(
            args.annotations,
            {"baseline": args.baseline, "reid": args.reid},
            video_path=args.video,
            iou_threshold=args.iou_threshold,
        )
        report["prediction_paths"] = {
            "baseline": str(args.baseline.expanduser().resolve()),
            "reid": str(args.reid.expanduser().resolve()),
        }
        if args.visualizations is not None:
            report["error_visualizations"] = {
                name: create_identity_error_visualizations(
                    args.annotations,
                    prediction_path,
                    args.video,
                    args.visualizations,
                    run_name=name,
                    iou_threshold=args.iou_threshold,
                )
                for name, prediction_path in (
                    ("baseline", args.baseline),
                    ("reid", args.reid),
                )
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
        return

    print(
        json.dumps(
            {
                "report": str(output_path),
                "video_sha256_verified": report["video_sha256_verified"],
                "eligible_frames": len(report["eligible_exhaustive_frames"]),
                "results": report["results"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
