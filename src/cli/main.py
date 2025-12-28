#!/usr/bin/env python3
"""
Find all PDF files in input directory and extract requirements from each file and export to Excel format.
"""
# main.py

import argparse
import sys
from pathlib import Path

# Add src to path so we can import our modules
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from config.loader import load_config
from requirement_extraction.requirement_extractor import RequirementExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Requirement Extractor"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load configuration (ONCE)
    config = load_config(args.config)

    # 2. Assemble application
    extractor = RequirementExtractor(config)

    # 3. Run
    extractor.run()


if __name__ == "__main__":
    main()
