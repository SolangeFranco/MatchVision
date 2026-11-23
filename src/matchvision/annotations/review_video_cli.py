"""CLI for creating browser-compatible identity annotation previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matchvision.annotations.review_video import create_review_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crea una copia H.264 compatible con navegador y conserva el orden de fotogramas."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Video original")
    parser.add_argument("--output", type=Path, required=True, help="Ruta del video de revisión")
    parser.add_argument(
        "--frame-count", type=int, required=True, help="Cantidad de fotogramas a copiar"
    )
    parser.add_argument(
        "--maximum-width", type=int, default=1920, help="Ancho máximo de salida"
    )
    parser.add_argument(
        "--crf", type=int, default=22, help="Calidad H.264 (CRF de 0 a 51)"
    )
    args = parser.parse_args()
    try:
        manifest = create_review_video(
            args.input,
            args.output,
            frame_count=args.frame_count,
            maximum_width=args.maximum_width,
            crf=args.crf,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
        return
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
