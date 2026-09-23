from __future__ import annotations

from pathlib import Path

from config.schema import PDFConfig
from src.pdf_processing.models import SemanticDocument, SemanticPage, SemanticSegment
from src.pdf_processing.pdf_to_markdown import convert_pdf_to_markdown


class LegacyMarkdownParser:
    """Compatibility parser used while migrating to structured OCR parsing."""

    def __init__(self, config: PDFConfig):
        self.config = config

    def parse_pdf(self, pdf_path: str) -> SemanticDocument:
        markdown_path = convert_pdf_to_markdown(pdf_path, output_dir=self.config.markdown_output_dir, config=self.config)
        markdown_text = Path(markdown_path).read_text(encoding="utf-8")
        segment = SemanticSegment(
            segment_id="legacy-markdown-1",
            segment_kind="text",
            paddle_label="legacy_markdown",
            page=0,
            text_content=markdown_text,
            text_markdown=markdown_text,
            source_block_ids=["legacy-markdown-1"],
            included_for_extraction=True,
        )
        page = SemanticPage(page=0, width=0, height=0, segments=[segment])
        return SemanticDocument(source_document=str(pdf_path), pages=[page], metadata={"markdown_path": markdown_path})
