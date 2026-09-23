#!/usr/bin/env python3
"""
Find all PDF files in input directory and extract requirements from each file and export to Excel format.
"""

# main.py

import argparse
import sys
from pathlib import Path

# Add repo root and src to path so we can import our modules
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from config.loader import load_config
from src.evaluation.quality_evaluator import EvaluationInputError, evaluate_quality
from src.requirement_extraction.requirement_extractor import (
    PartialExtractionError,
    RequirementExtractor,
)
from src.utils.logging_config import configure_logging
from src.vlm_service import VLMServiceManager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Requirement Extractor")
    parser.add_argument(
        "command",
        nargs="?",
        default="extract",
        choices=[
            "extract",
            "prepare-pdf",
            "evaluate",
            "check-vlm-service",
            "start-vlm-service",
        ],
        help="Command to execute",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--references-dir",
        type=Path,
        help="Directory containing one reference workbook per document",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="Directory containing extraction run output directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where evaluation reports will be written",
    )
    parser.add_argument(
        "--document-inventory",
        type=Path,
        help="Optional CSV mapping document_id, pdf_path, reference_xlsx, and notes",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.command == "evaluate":
        missing_args = [
            name
            for name in ("references_dir", "runs_dir", "output_dir")
            if getattr(args, name) is None
        ]
        if missing_args:
            raise SystemExit(
                f"evaluate requires: {', '.join('--' + name.replace('_', '-') for name in missing_args)}"
            )
        try:
            evaluate_quality(
                references_dir=args.references_dir,
                runs_dir=args.runs_dir,
                output_dir=args.output_dir,
                document_inventory=args.document_inventory,
            )
        except EvaluationInputError as exc:
            raise SystemExit(f"Evaluation input error: {exc}") from exc
        print(f"Evaluation reports written to {args.output_dir}")
        return

    config = load_config(args.config)
    configure_logging(config.logging)
    vlm_service = VLMServiceManager(config.parser)

    if args.command == "check-vlm-service":
        vlm_service.ensure_healthy()
        print(f"VLM service reachable at {config.parser.vlm_server_url}")
        return

    if args.command == "start-vlm-service":
        raise SystemExit(vlm_service.start_server())

    extractor = RequirementExtractor(config, command_name=args.command)
    try:
        extractor.run()
    except (
        PartialExtractionError,
        FileExistsError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
