"""Clasifica equipos de fútbol a partir del color de camiseta de los tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from matchvision.teams.classification import UniformTeamClassifier


def classify_trajectory_teams(
    video_path: str | Path,
    trajectories_path: str | Path,
    output_path: str | Path,
    *,
    identity_column: str = "persistent_id",
    overrides_path: str | Path | None = None,
) -> dict[str, Any]:
    """Classify trajectory identities and write a new CSV plus provenance JSON."""
    video = Path(video_path).expanduser().resolve()
    trajectories = Path(trajectories_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    manifest_path = output.with_suffix(output.suffix + ".json")
    if not video.is_file():
        raise FileNotFoundError(f"No se encontró el video: {video}")
    if not trajectories.is_file():
        raise FileNotFoundError(f"No se encontró el CSV de trayectorias: {trajectories}")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            f"No se sobrescribirán resultados de equipos existentes: {output} o {manifest_path}"
        )
    if identity_column not in {"persistent_id", "temporal_track_id"}:
        raise ValueError(
            "identity_column debe ser persistent_id o temporal_track_id"
        )

    observations = _read_observations(trajectories, identity_column)
    if not observations:
        raise ValueError("El CSV de trayectorias no contiene observaciones de personas")
    grouped: dict[int, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    for frame, identity, box in observations:
        grouped[frame].append((identity, box))

    classifier = UniformTeamClassifier()
    overrides = (
        Path(overrides_path).expanduser().resolve()
        if overrides_path is not None
        else None
    )
    if overrides is not None:
        for identity, team in _load_overrides(overrides, identity_column).items():
            classifier.set_manual_assignment(identity, team)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"No se pudo abrir el video: {video}")
    frame_number = 0
    try:
        last_frame = max(grouped)
        while frame_number < last_frame:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(
                    f"El video termina en el fotograma {frame_number}; las trayectorias requieren el {last_frame}"
                )
            frame_number += 1
            for identity, box in grouped.get(frame_number, ()):
                classifier.observe(frame, identity, box, frame_number)
    finally:
        capture.release()

    assignments = classifier.classify()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = _temporary_path(output)
    try:
        with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "identity_id",
                    "team_id",
                    "state",
                    "confidence",
                    "appearance_samples",
                    "assignment_basis",
                ),
            )
            writer.writeheader()
            for assignment in assignments.values():
                writer.writerow(
                    {
                        "identity_id": assignment.identity_id,
                        "team_id": assignment.team_id or "",
                        "state": assignment.state,
                        "confidence": (
                            f"{assignment.confidence:.6f}"
                            if assignment.confidence is not None
                            else ""
                        ),
                        "appearance_samples": assignment.sample_count,
                        "assignment_basis": (
                            "manual" if assignment.state == "manual" else "estimada"
                        ),
                    }
                )
        if output.exists():
            raise FileExistsError(f"No se sobrescribirá el resultado existente: {output}")
        os.rename(temporary_csv, output)
    finally:
        temporary_csv.unlink(missing_ok=True)

    states: dict[str, int] = defaultdict(int)
    for assignment in assignments.values():
        states[assignment.state] += 1
    payload: dict[str, Any] = {
        "schema_version": 1,
        "method": classifier.configuration,
        "identity_column": identity_column,
        "inputs": {
            "video_path": str(video),
            "video_sha256": _sha256_file(video),
            "trajectories_path": str(trajectories),
            "trajectories_sha256": _sha256_file(trajectories),
            "manual_overrides_path": str(overrides) if overrides else None,
            "manual_overrides_sha256": (
                _sha256_file(overrides) if overrides is not None else None
            ),
        },
        "output_csv": str(output),
        "output_csv_sha256": _sha256_file(output),
        "input_observations": len(observations),
        "identity_count": len(assignments),
        "identities_with_usable_appearance": sum(
            assignment.sample_count > 0 for assignment in assignments.values()
        ),
        "identities_with_sufficient_samples": sum(
            assignment.sample_count >= classifier.minimum_samples
            for assignment in assignments.values()
        ),
        "assignment_states": dict(sorted(states.items())),
        "limitations": [
            "Equipo es una estimación visual no supervisada basada en color de camiseta; no es una identidad individual.",
            "El modelo no identifica árbitros ni resuelve uniformes similares; las asignaciones ambiguas o atípicas requieren revisión.",
            "Los ID de entrada se usan solo como claves de agregación y no se validan como identidades humanas.",
            "La confianza es un margen geométrico del agrupamiento, no una probabilidad.",
        ],
    }
    temporary_manifest = _temporary_path(manifest_path)
    try:
        temporary_manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if manifest_path.exists():
            raise FileExistsError(
                f"No se sobrescribirá el manifiesto existente: {manifest_path}"
            )
        os.rename(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return payload


def _read_observations(
    path: Path,
    identity_column: str,
) -> list[tuple[int, str, tuple[float, float, float, float]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"frame", identity_column, "x1", "y1", "x2", "y2"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Al CSV de trayectorias le faltan columnas: {sorted(missing)}")
        rows: list[tuple[int, str, tuple[float, float, float, float]]] = []
        for line_number, row in enumerate(reader, start=2):
            label = row.get("label", "person").strip().lower().replace("_", " ")
            if label and label not in {
                "person",
                "player",
                "football player",
                "goalkeeper",
                "referee",
            }:
                continue
            identity = (row.get(identity_column) or "").strip()
            if not identity:
                continue
            try:
                frame = int(row["frame"])
                box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Fotograma, identidad o coordenadas no válidas en la línea {line_number}"
                ) from error
            if frame < 1 or len(box) != 4 or not all(math.isfinite(value) for value in box):
                raise ValueError(f"Fotograma o recuadro no válido en la línea {line_number}")
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"El recuadro tiene dimensiones no positivas en la línea {line_number}")
            rows.append((frame, identity, box))
    return rows


def _load_overrides(path: Path, identity_column: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de correcciones manuales: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON de correcciones manuales no válido: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Las correcciones manuales deben usar schema_version 1")
    if payload.get("identity_column") != identity_column:
        raise ValueError(
            f"Las correcciones deben usar identity_column {identity_column!r}"
        )
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Las correcciones deben incluir un objeto assignments")
    allowed_teams = {"equipo_a", "equipo_b"}
    for identity, team in assignments.items():
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Los ID de las correcciones deben ser textos no vacíos")
        if not isinstance(team, str) or team not in allowed_teams:
            raise ValueError(f"El equipo de la corrección debe ser uno de {sorted(allowed_teams)}")
    return assignments


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    return Path(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estima los dos equipos a partir del color de camiseta de los tracks."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--identity-column",
        choices=("persistent_id", "temporal_track_id"),
        default="persistent_id",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        help="JSON de correcciones manuales de equipo; se conserva el original",
    )
    args = parser.parse_args()
    try:
        payload = classify_trajectory_teams(
            args.video,
            args.trajectories,
            args.output,
            identity_column=args.identity_column,
            overrides_path=args.overrides,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
