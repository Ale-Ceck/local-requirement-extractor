from __future__ import annotations

from config.schema import AppConfig
from src.pdf_processing.legacy_markdown_parser import LegacyMarkdownParser
from src.pdf_processing.paddleocr_parser import PaddleOCRVLParser


def create_document_parser(config: AppConfig, toc_pruning_plan=None):
    if config.parser.backend == "paddleocr_vl":
        return PaddleOCRVLParser(
            config.parser,
            output_directory=config.output.directory,
            toc_pruning_plan=toc_pruning_plan,
        )
    if config.parser.backend == "legacy_markdown":
        return LegacyMarkdownParser(config.pdf)
    raise ValueError(f"Unsupported parser backend: {config.parser.backend}")
