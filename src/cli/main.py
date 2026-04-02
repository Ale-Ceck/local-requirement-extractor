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
from src.requirement_extraction.requirement_extractor import RequirementExtractor
from src.vlm_service import VLMServiceManager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Requirement Extractor"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="extract",
        choices=["extract", "check-vlm-service", "start-vlm-service"],
        help="Command to execute",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    vlm_service = VLMServiceManager(config.parser)

    if args.command == "check-vlm-service":
        vlm_service.ensure_healthy()
        print(f"VLM service reachable at {config.parser.vlm_server_url}")
        return

    if args.command == "start-vlm-service":
        raise SystemExit(vlm_service.start_server())

    extractor = RequirementExtractor(config)
    extractor.run()


if __name__ == "__main__":
    main()
