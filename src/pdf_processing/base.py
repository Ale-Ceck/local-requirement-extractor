from __future__ import annotations

from typing import Protocol

from src.pdf_processing.models import SemanticDocument


class DocumentParser(Protocol):
    def parse_pdf(self, pdf_path: str) -> SemanticDocument:
        ...
