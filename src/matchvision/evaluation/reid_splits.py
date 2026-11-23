"""Leakage-resistant match-grouped ReID dataset split manifests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


_SPLITS = ("train", "validation", "test")


def build_grouped_split_manifest(
    records: list[dict[str, Any]], *, seed: int = 42
) -> dict[str, Any]:
    """Assign whole match/identity components to disjoint data splits."""
    matches: dict[str, dict[str, Any]] = {}
    video_to_match: dict[str, str] = {}
    identity_to_matches: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Cada partido debe ser un objeto JSON.")
        match_id = record.get("match_id")
        video_sha256 = record.get("video_sha256")
        identity_ids = record.get("identity_ids")
        if (
            not isinstance(match_id, str)
            or not match_id.strip()
            or match_id in matches
            or not isinstance(video_sha256, str)
            or len(video_sha256) != 64
            or not isinstance(identity_ids, list)
            or not identity_ids
            or any(not isinstance(item, str) or not item.strip() for item in identity_ids)
        ):
            raise ValueError("Registro de partido incompleto o duplicado.")
        previous_match = video_to_match.get(video_sha256)
        if previous_match is not None and previous_match != match_id:
            raise ValueError(
                f"El video {video_sha256} aparece en más de un partido."
            )
        if any(identity.upper().startswith("TEST-ONLY-") for identity in identity_ids):
            raise ValueError("Las identidades TEST-ONLY no se admiten en un split.")
        video_to_match[video_sha256] = match_id
        identities = set(identity_ids)
        matches[match_id] = {
            "match_id": match_id,
            "video_sha256": video_sha256,
            "identity_ids": sorted(identities),
        }
        for identity in identities:
            identity_to_matches.setdefault(identity, set()).add(match_id)

    parent = {match_id: match_id for match_id in matches}

    def find(match_id: str) -> str:
        while parent[match_id] != match_id:
            parent[match_id] = parent[parent[match_id]]
            match_id = parent[match_id]
        return match_id

    for shared_matches in identity_to_matches.values():
        ordered = sorted(shared_matches)
        for match_id in ordered[1:]:
            parent[find(match_id)] = find(ordered[0])

    components: dict[str, list[str]] = {}
    for match_id in sorted(matches):
        components.setdefault(find(match_id), []).append(match_id)
    units = sorted(components.values(), key=lambda unit: tuple(unit))
    random.Random(seed).shuffle(units)

    if len(units) < 3:
        assignments = {
            match_id: "unassigned"
            for match_id in sorted(matches)
        }
        status = "insufficient_independent_groups"
    else:
        tie_break_order = {tuple(unit): index for index, unit in enumerate(units)}
        units.sort(
            key=lambda unit: (-len(unit), tie_break_order[tuple(unit)])
        )
        assigned_units: dict[str, list[list[str]]] = {
            split: [] for split in _SPLITS
        }
        split_sizes = {split: 0 for split in _SPLITS}
        next_split = 0
        for unit in units:
            if any(not assigned_units[split] for split in _SPLITS):
                selected = next(
                    split
                    for split in _SPLITS[next_split:] + _SPLITS[:next_split]
                    if not assigned_units[split]
                )
            else:
                minimum_size = min(split_sizes.values())
                selected = next(
                    split
                    for split in _SPLITS[next_split:] + _SPLITS[:next_split]
                    if split_sizes[split] == minimum_size
                )
            assigned_units[selected].append(unit)
            split_sizes[selected] += len(unit)
            next_split = (_SPLITS.index(selected) + 1) % len(_SPLITS)
        assignments = {
            match_id: split
            for split, split_units in assigned_units.items()
            for unit in split_units
            for match_id in unit
        }
        status = "ready"

    split_by_identity: dict[str, str] = {}
    for match_id, record in matches.items():
        split = assignments[match_id]
        for identity in record["identity_ids"]:
            previous_split = split_by_identity.get(identity)
            if split != "unassigned" and previous_split not in {None, split}:
                raise RuntimeError(
                    f"La identidad {identity} se filtró entre particiones."
                )
            split_by_identity[identity] = split

    return {
        "schema_version": 1,
        "record_type": "grouped_reid_split_manifest",
        "status": status,
        "seed": seed,
        "grouping": [
            "match_id",
            "video_sha256",
            "shared_identity_id",
        ],
        "split_fractions_target": {
            "train": 1 / 3,
            "validation": 1 / 3,
            "test": 1 / 3,
        },
        "assignments": [
            {**matches[match_id], "split": assignments[match_id]}
            for match_id in sorted(matches)
        ],
        "independent_groups": len(units),
        "limitations": [
            "Los fotogramas, tracks y apariciones de un partido nunca se separan.",
            "Partidos conectados por una identidad humana se mantienen en el mismo grupo.",
            "Con menos de tres grupos independientes no se asignan train/validation/test.",
            "Las proporciones son objetivos aproximados; la integridad de grupos tiene prioridad.",
        ],
    }


def _read_review(path: Path, match_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("record_type") != "human_review_with_manual_ground_truth":
        raise ValueError(f"{path} no es una exportación de revisión humana.")
    video = payload.get("video")
    identities = payload.get("identities")
    if not isinstance(video, dict) or not isinstance(identities, list):
        raise ValueError(f"{path} no contiene video e identidades.")
    identity_ids = sorted({
        identity["identity_id"]
        for identity in identities
        if isinstance(identity, dict)
        and identity.get("status") == "validated_manual"
        and any(
            isinstance(track, dict)
            and track.get("annotation_source")
            == "independent_human_track_annotation"
            for track in identity.get("tracks", [])
        )
    })
    if not identity_ids:
        raise ValueError(
            f"{path} no contiene asignaciones independientes; no se crea un split."
        )
    return {
        "match_id": match_id,
        "video_sha256": video.get("sha256"),
        "identity_ids": identity_ids,
        "review_path": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea un manifiesto de splits ReID agrupados por partido e identidad."
    )
    parser.add_argument(
        "--review",
        action="append",
        required=True,
        metavar="MATCH_ID=JSON",
        help="Exportación de revisión; repetir por partido",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = []
        for item in args.review:
            if "=" not in item:
                raise ValueError("--review debe usar el formato MATCH_ID=JSON.")
            match_id, raw_path = item.split("=", 1)
            records.append(_read_review(Path(raw_path).expanduser().resolve(), match_id))
        manifest = build_grouped_split_manifest(records, seed=args.seed)
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"No se sobrescribió el manifiesto existente: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
        return
    print(json.dumps({
        "manifest": str(output),
        "status": manifest["status"],
        "independent_groups": manifest["independent_groups"],
        "splits": {
            split: sum(item["split"] == split for item in manifest["assignments"])
            for split in (*_SPLITS, "unassigned")
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
