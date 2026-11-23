"""CLI for selecting difficult tracking frames for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matchvision.tracking.review_cases import select_review_cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selecciona eventos de tracking para revisar identidades manualmente."
    )
    parser.add_argument(
        "--baseline-mot", type=Path, required=True, help="Predicciones MOT baseline"
    )
    parser.add_argument(
        "--trajectories", type=Path, required=True, help="CSV de trayectorias"
    )
    parser.add_argument("--video", type=Path, required=True, help="Video de entrada")
    parser.add_argument(
        "--frame-count", type=int, required=True, help="Cantidad de fotogramas"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON de casos priorizados"
    )
    parser.add_argument(
        "--maximum-cases", type=int, default=60, help="Máximo de casos seleccionados"
    )
    parser.add_argument(
        "--radius-frames", type=int, default=30, help="Radio de revisión en fotogramas"
    )
    parser.add_argument(
        "--minimum-crossing-distance-heights",
        type=float,
        default=0.65,
        help="Distancia mínima entre cruces, expresada en alturas de caja",
    )
    args = parser.parse_args()

    try:
        output_path = args.output.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing case list: {output_path}")
        payload = select_review_cases(
            args.baseline_mot,
            args.trajectories,
            video_path=args.video,
            frame_count=args.frame_count,
            output_path=output_path,
            maximum_cases=args.maximum_cases,
            radius_frames=args.radius_frames,
            minimum_crossing_distance_heights=args.minimum_crossing_distance_heights,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
        return

    print(
        json.dumps(
            {
                "output": str(output_path),
                "selected_cases": len(payload["cases"]),
                "event_types": sorted(
                    {case["event_type"] for case in payload["cases"]}
                ),
                "limitations": payload["limitations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
