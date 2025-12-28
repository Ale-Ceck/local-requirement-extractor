# requirement_extractor.py

from pathlib import Path
from typing import Iterable

from config.schema import AppConfig
from llm_integration.ollama_client import OllamaClient
from utils.markdown_splitter import MarkdownSplitter
from pdf_processing import PDFProcessor
from excel_writer import ExcelWriter


class RequirementExtractor:
    """
    Coordinates the full extraction pipeline.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        # Subsystems receive ONLY what they need
        self.ollama_client = OllamaClient(config.ollama)
        self.splitter = MarkdownSplitter(config.chunking)
        self.pdf_processor = PDFProcessor(config.pdf)
        self.excel_writer = ExcelWriter(config.output)

    def run(self) -> None:
        input_path = Path(self.config.input.path)

        if self.config.input.mode == "pdf":
            markdown_files = self._process_pdfs(input_path)
        else:
            markdown_files = self._collect_markdown_files(input_path)

        all_requirements = []

        for md_file in markdown_files:
            chunks = self.splitter.split(md_file)
            requirements = self._extract_from_chunks(chunks)
            all_requirements.extend(requirements)

        self.excel_writer.write(all_requirements)

    # ---------------------------
    # Internal steps
    # ---------------------------

    def _process_pdfs(self, path: Path) -> Iterable[Path]:
        return self.pdf_processor.convert_to_markdown(
            path,
            recursive=self.config.input.recursive,
        )

    def _collect_markdown_files(self, path: Path) -> Iterable[Path]:
        if path.is_file():
            return [path]

        extensions = self.config.input.file_extensions
        files: list[Path] = []

        for ext in extensions:
            files.extend(path.rglob(f"*{ext}"))

        return files

    def _extract_from_chunks(self, chunks: Iterable[str]):
        results = []

        for chunk in chunks:
            extracted = self.ollama_client.extract_requirements(chunk)
            if extracted or self.config.extraction.allow_empty_results:
                results.extend(extracted)

        return results
