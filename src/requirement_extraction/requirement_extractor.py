# requirement_extractor.py

import concurrent.futures
import json
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.documents import Document

from config.schema import AppConfig
from src.data_models.requirement import RequirementList
from src.llm_integration.ollama_client import OllamaClient
from src.llm_integration.prompt_templates import get_prompt
from src.pdf_processing.pdf_to_markdown import convert_pdf_to_markdown
from src.requirement_extraction.excel_writer import ExcelWriter
from src.utils.logging_config import setup_logger
from src.utils.markdown_splitter import MarkdownSplitter

logger = setup_logger(__name__)


class RequirementExtractor:
    """Extract requirements from PDF or Markdown documents using LLM processing."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize the requirement extractor with centralized configuration."""
        self.config = config
        self.model_name = config.extraction.model_name
        self.max_workers = max(1, config.parallel.max_workers)
        self.parallel_enabled = config.parallel.enabled
        self.ollama_client = OllamaClient(config.ollama)
        self.splitter = MarkdownSplitter(config.chunking)
        self.excel_writer = ExcelWriter(config.output)

    def run(self) -> None:
        """Run the extraction pipeline based on configuration."""
        input_path = Path(self.config.input.path)

        if self.config.input.mode == "pdf":
            source_files = self._collect_pdfs(input_path)
            all_requirements = self._extract_from_pdfs(source_files)
        else:
            source_files = self._collect_markdown_files(input_path)
            all_requirements = self._extract_from_markdown_files(source_files)

        self.excel_writer.write(RequirementList(all_requirements))

    def extract_requirements_from_pdf(
        self,
        pdf_path: str,
        markdown_dir: Optional[str] = None,
    ) -> RequirementList:
        """
        Extract requirements from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            markdown_dir: Optional directory for markdown output

        Returns:
            RequirementList with extracted requirements
        """
        try:
            logger.info(f"Starting requirement extraction from PDF: {pdf_path}")

            if markdown_dir is None:
                markdown_dir = self.config.pdf.markdown_output_dir

            markdown_path = convert_pdf_to_markdown(
                pdf_path,
                output_dir=markdown_dir,
                config=self.config.pdf,
            )
            logger.info(f"PDF converted to markdown: {markdown_path}")

            return self.extract_requirements_from_markdown(markdown_path)
        except Exception as e:
            logger.error(f"Error extracting requirements from PDF: {e}")
            return RequirementList([])

    def extract_requirements_from_markdown(self, md_path: str) -> RequirementList:
        """
        Extract requirements from a Markdown file.

        Args:
            md_path: Path to the Markdown file to process

        Returns:
            RequirementList with extracted requirements
        """
        try:
            logger.info(f"Starting requirement extraction from Markdown: {md_path}")

            requirement_schema = json.dumps(RequirementList.model_json_schema())
            docs = self.splitter.split_markdown(markdown_path=md_path)

            if self.parallel_enabled:
                all_requirements = self.process_chunks_parallel(docs, requirement_schema)
            else:
                all_requirements = self.process_chunks_sequential(docs, requirement_schema)

            requirements = RequirementList(all_requirements)
            logger.info(f"Successfully extracted {len(requirements.root)} requirements")
            return requirements
        except Exception as e:
            logger.error(f"Error extracting requirements from markdown: {e}")
            raise

    def _collect_pdfs(self, path: Path) -> Iterable[Path]:
        if path.is_file():
            return [path]

        if self.config.input.recursive:
            return list(path.rglob("*.pdf"))

        return list(path.glob("*.pdf"))

    def _collect_markdown_files(self, path: Path) -> Iterable[Path]:
        if path.is_file():
            return [path]

        extensions = self.config.input.file_extensions
        files: list[Path] = []

        for ext in extensions:
            if self.config.input.recursive:
                files.extend(path.rglob(f"*{ext}"))
            else:
                files.extend(path.glob(f"*{ext}"))

        return files

    def _extract_from_pdfs(self, pdf_files: Iterable[Path]) -> list:
        all_requirements = []

        for pdf_file in pdf_files:
            requirements = self.extract_requirements_from_pdf(str(pdf_file))
            if requirements.root or self.config.extraction.allow_empty_results:
                all_requirements.extend(requirements.root)

        return all_requirements

    def _extract_from_markdown_files(self, markdown_files: Iterable[Path]) -> list:
        all_requirements = []

        for md_file in markdown_files:
            requirements = self.extract_requirements_from_markdown(str(md_file))
            if requirements.root or self.config.extraction.allow_empty_results:
                all_requirements.extend(requirements.root)

        return all_requirements

    def process_single_chunk(self, doc: Document, requirement_schema: str) -> RequirementList:
        """
        Process a single document chunk to extract requirements.

        Args:
            doc: Document chunk from langchain splitter
            requirement_schema: JSON schema for requirements

        Returns:
            RequirementList with extracted requirements from this chunk
        """
        try:
            prompt = get_prompt(
                "requirement_extraction",
                doc.page_content,
                requirement_schema,
                include_few_shot=True,
            )

            response = self.ollama_client.get_structured_response(
                prompt,
                model_name=self.model_name,
            )

            if response is None:
                logger.error("Failed to get response from LLM for chunk")
                return RequirementList([])

            return self.parse_llm_response(response)
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            return RequirementList([])

    def process_chunks_parallel(self, docs: list[Document], requirement_schema: str) -> list:
        """
        Process document chunks in parallel to extract requirements.

        Args:
            docs: List of document chunks from langchain splitter
            requirement_schema: JSON schema for requirements

        Returns:
            List of Requirement objects from all chunks
        """
        all_requirements = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.process_single_chunk, doc, requirement_schema): i
                for i, doc in enumerate(docs)
            }

            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    chunk_requirements = future.result()
                    logger.info(
                        "Processed chunk %s/%s: found %s requirements",
                        chunk_index + 1,
                        len(docs),
                        len(chunk_requirements),
                    )
                    if chunk_requirements.root or self.config.extraction.allow_empty_results:
                        all_requirements.extend(chunk_requirements.root)
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index + 1}: {e}")

        return all_requirements

    def process_chunks_sequential(self, docs: list[Document], requirement_schema: str) -> list:
        """Process document chunks sequentially to extract requirements."""
        all_requirements = []

        for index, doc in enumerate(docs):
            chunk_requirements = self.process_single_chunk(doc, requirement_schema)
            logger.info(
                "Processed chunk %s/%s: found %s requirements",
                index + 1,
                len(docs),
                len(chunk_requirements),
            )
            if chunk_requirements.root or self.config.extraction.allow_empty_results:
                all_requirements.extend(chunk_requirements.root)

        return all_requirements

    def parse_llm_response(self, json_response: str) -> RequirementList:
        """
        Parse LLM JSON response into validated requirements.

        Args:
            json_response: JSON string from LLM

        Returns:
            RequirementList object

        Raises:
            ValueError: If JSON is malformed
        """
        try:
            data = json.loads(json_response)

            if isinstance(data, dict):
                if "requirements" in data:
                    data = data["requirements"]
                elif "requirement" in data:
                    data = [data["requirement"]]
                else:
                    if "code" in data and "description" in data:
                        data = [data]
                    else:
                        data = []

            if not isinstance(data, list):
                data = []

            return RequirementList(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {e}") from e


def extract_requirements_from_pdf(
    pdf_path: str,
    config: AppConfig,
    markdown_dir: Optional[str] = None,
) -> RequirementList:
    """Convenience function to extract requirements from a PDF file."""
    extractor = RequirementExtractor(config=config)
    return extractor.extract_requirements_from_pdf(pdf_path, markdown_dir)


def extract_requirements_from_markdown(md_path: str, config: AppConfig) -> RequirementList:
    """Convenience function to extract requirements from a Markdown file."""
    extractor = RequirementExtractor(config=config)
    return extractor.extract_requirements_from_markdown(md_path)
