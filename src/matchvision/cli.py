"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from matchvision.video import get_video_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consulta los metadatos de un video de partido."
    )
    parser.add_argument("video", type=Path, help="Ruta al video de entrada")
    args = parser.parse_args()

    info = get_video_info(args.video)
    result = asdict(info)
    result["path"] = str(result["path"])
    result["duration_seconds"] = info.duration_seconds
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
