"""CLI for descriptive summaries of MOT tracking output."""

from __future__ import annotations

import argparse
import json

from matchvision.tracking.analysis import analyze_mot_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume resultados de tracking sin métricas de ground truth."
    )
    parser.add_argument("mot", help="Archivo de predicciones MOT")
    parser.add_argument(
        "--frame-count", required=True, type=int, help="Cantidad de fotogramas"
    )
    parser.add_argument("--output", required=True, help="Carpeta de resultados")
    args = parser.parse_args()
    try:
        summary = analyze_mot_file(args.mot, args.frame_count, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
        return
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
