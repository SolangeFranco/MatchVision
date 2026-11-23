"""Download the pinned OSNet-AIN checkpoint without replacing existing files."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from matchvision.reidentification.osnet import (
    MODEL_FILENAME,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    MODEL_URL,
)


def download_weights(destination: str | Path) -> Path:
    target = Path(destination).expanduser().resolve()
    if target.exists():
        if target.is_file() and _sha256_file(target) == MODEL_SHA256:
            return target
        raise FileExistsError(f"No se sobrescribió el archivo existente: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download")
    if temporary.exists():
        raise FileExistsError(f"Ya existe una descarga temporal: {temporary}")
    try:
        request = urllib.request.Request(
            MODEL_URL, headers={"User-Agent": "MatchVision/0.1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if temporary.stat().st_size != MODEL_SIZE_BYTES:
            raise ValueError("El tamaño descargado no coincide con el modelo fijado.")
        if _sha256_file(temporary) != MODEL_SHA256:
            raise ValueError("El SHA-256 descargado no coincide con el modelo fijado.")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga los pesos OSNet-AIN fijados y verifica su integridad."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / MODEL_FILENAME,
        help="Ruta de salida; no se sobrescriben archivos existentes",
    )
    args = parser.parse_args()
    path = download_weights(args.output)
    print(f"Pesos OSNet verificados: {path}")


if __name__ == "__main__":
    main()
