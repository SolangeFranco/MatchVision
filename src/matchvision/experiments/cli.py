"""CLI for validating experiment configuration and sequence manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matchvision.experiments.config import load_benchmark_config
from matchvision.experiments.manifest import load_sequence_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valida el YAML del benchmark y, de forma opcional, un manifiesto de secuencias."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark.yaml"),
        help="Ruta a la configuración YAML del benchmark",
    )
    parser.add_argument(
        "--manifest", type=Path, help="Manifiesto de secuencias para validar"
    )
    args = parser.parse_args()

    try:
        config = load_benchmark_config(args.config)
        manifest = None
        if args.manifest is not None:
            manifest = load_sequence_manifest(
                args.manifest,
                require_hashes=config["experiment"]["dataset"]["sequence_hashes_required"],
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))
        return

    print(
        json.dumps(
            {
                "config": str(args.config.resolve()),
                "config_valid": True,
                "manifest": str(manifest.path) if manifest else None,
                "manifest_valid": manifest is not None,
                "dataset": manifest.dataset if manifest else None,
                "release": manifest.release if manifest else None,
                "sequences": len(manifest.sequences) if manifest else None,
                "manifest_sha256": manifest.sha256 if manifest else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
