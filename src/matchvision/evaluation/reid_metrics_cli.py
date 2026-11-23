"""CLI for evaluating ReID candidates against a human review export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matchvision.evaluation.reid_metrics import (
    evaluate_reid_methods,
    evaluate_reid_proposals,
    load_json_object,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa propuestas ReID exclusivamente frente a asignaciones humanas "
            "confirmadas."
        )
    )
    parser.add_argument(
        "--review",
        type=Path,
        required=True,
        help="JSON exportado de revisión humana y Ground Truth",
    )
    parser.add_argument(
        "--proposals", type=Path, required=True, help="JSON de propuestas ReID"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Ruta nueva para el informe JSON"
    )
    parser.add_argument(
        "--mode",
        choices=("proposal-conditioned", "independent-methods"),
        default="proposal-conditioned",
        help=(
            "Evaluación histórica condicionada a propuestas o comparación de "
            "scores contra asignaciones directas independientes"
        ),
    )
    parser.add_argument("--threshold-hsv-hog", type=float, default=0.82)
    parser.add_argument("--threshold-embedding", type=float, default=0.82)
    parser.add_argument("--threshold-hybrid", type=float, default=0.82)
    args = parser.parse_args()

    try:
        output_path = args.output.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(
                f"No se sobrescribió el informe existente: {output_path}"
            )
        review = load_json_object(args.review)
        proposals = load_json_object(args.proposals)
        actual_proposals_sha256 = sha256_file(args.proposals)
        if args.mode == "independent-methods":
            report = evaluate_reid_methods(
                review,
                proposals,
                actual_proposals_sha256=actual_proposals_sha256,
                thresholds={
                    "hsv_hog": args.threshold_hsv_hog,
                    "embedding": args.threshold_embedding,
                    "hybrid": args.threshold_hybrid,
                },
            )
        else:
            report = evaluate_reid_proposals(
                review,
                proposals,
                actual_proposals_sha256=actual_proposals_sha256,
            )
        report["input_files"] = {
            "review": str(args.review.expanduser().resolve()),
            "proposals": str(args.proposals.expanduser().resolve()),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
        return

    print(
        json.dumps(
            {
                "report": str(output_path),
                "status": report["status"],
                "counts": report["counts"],
                "missing_data": report["missing_data"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
