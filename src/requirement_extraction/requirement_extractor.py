import concurrent.futures
import copy
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional
from uuid import uuid4

from langchain_core.documents import Document

from config.schema import AppConfig
from src.data_models.requirement import (
    Requirement,
    RequirementExtractionResult,
    RequirementList,
)
from src.llm_integration.ollama_client import OllamaClient
from src.llm_integration.prompt_templates import (
    get_prompt,
    get_requirement_extraction_prompt_fingerprint,
)
from src.pdf_processing.factory import create_document_parser
from src.pdf_processing.models import SemanticSegment
from src.requirement_extraction.chunk_cache import load_chunk_cache, write_chunk_cache
from src.requirement_extraction.document_chunker import (
    AnchoredMarkdownChunk,
    SemanticDocumentChunker,
)
from src.requirement_extraction.excel_writer import ExcelWriter
from src.requirement_extraction.run_safety import (
    collect_input_files,
    validate_output_directory,
)
from src.utils.logging_config import setup_logger
from src.utils.markdown_splitter import MarkdownSplitter
from src.utils.reproducibility import (
    collect_file_metadata,
    collect_git_metadata,
    collect_runtime_metadata,
)

logger = setup_logger(__name__)


class PartialExtractionError(RuntimeError):
    """Useful exports were saved, but one or more extraction results were lost."""


class RequirementExtractor:
    """Extract requirements from PDF or Markdown documents using LLM processing."""

    def __init__(
        self,
        config: AppConfig,
        ollama_client: Optional[OllamaClient] = None,
        document_parser=None,
        toc_pruning_plan: Optional[dict] = None,
        markdown_splitter: Optional[MarkdownSplitter] = None,
        chunker: Optional[SemanticDocumentChunker] = None,
        excel_writer: Optional[ExcelWriter] = None,
        command_name: str = "extract",
        write_run_manifest: bool = True,
        run_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self.model_name = config.extraction.model_name
        self.max_workers = max(1, config.parallel.max_workers)
        self.parallel_enabled = config.parallel.enabled
        self.command_name = command_name
        self.write_run_manifest = write_run_manifest
        self.run_id = run_id or self._new_run_id()
        self.ollama_client = ollama_client or OllamaClient(config.ollama)
        self.toc_pruning_plan = toc_pruning_plan
        self.document_parser = document_parser or create_document_parser(
            config, toc_pruning_plan=toc_pruning_plan
        )
        self.markdown_splitter = markdown_splitter or MarkdownSplitter(config.chunking)
        self.chunker = chunker or SemanticDocumentChunker(config.chunking)
        self.excel_writer = excel_writer or ExcelWriter(config.output)
        self._run_stats_lock = Lock()
        self._run_started_at_perf = time.perf_counter()
        self.run_stats = self._new_run_stats()
        self._source_files: dict[str, list[Path]] = {}
        self._pdf_artifact_dirs: dict[str, Path] = {}

    def run(self) -> None:
        """Execute the configured pipeline and always persist terminal run state."""
        # A rejected destination belongs to a previous run. Do not replace its
        # manifest with the failure metadata for this attempted invocation.
        validate_output_directory(self.config)
        try:
            self._run_pipeline()
        except Exception as exc:
            if self.write_run_manifest:
                try:
                    self._write_failed_run_manifest(exc)
                except Exception as manifest_exc:
                    logger.error(
                        "Failed to persist failed run metadata: %s", manifest_exc
                    )
            raise

        partial_error = self._partial_extraction_error()
        if partial_error is not None:
            # The pipeline has already exported results and persisted its partial
            # manifest. Do not relabel it as a fatal failure in the handler above.
            if not self.write_run_manifest:
                self._finish_run_stats(status="partial", error=partial_error)
            raise partial_error

    def _run_pipeline(self) -> None:
        input_path = Path(self.config.input.path)
        if self.command_name == "prepare-pdf":
            if self.config.input.mode != "pdf":
                raise ValueError("prepare-pdf only supports input.mode=pdf.")
            source_files = self._collect_pdfs(input_path)
            self._prepare_pdfs(source_files)
            if self.write_run_manifest:
                self._write_run_manifest(
                    pdf_sources=source_files,
                    used_live_ocr=True,
                    used_chunk_cache_replay=False,
                )
            return

        if self.config.input.mode == "pdf":
            source_files = self._collect_pdfs(input_path)
            all_requirements = self._extract_from_pdfs(source_files)
            requirement_list = RequirementList(
                self._finalize_requirements(all_requirements)
            )
            self._set_stat("requirements_written", len(requirement_list))
            self.excel_writer.write(requirement_list)
            if self.write_run_manifest:
                self._write_run_manifest(
                    pdf_sources=source_files,
                    used_live_ocr=True,
                    used_chunk_cache_replay=False,
                )
            return

        if self.config.input.mode == "chunk_cache":
            cache_files = self._collect_chunk_cache_files(input_path)
            all_requirements = self._extract_from_chunk_cache_files(cache_files)
            requirement_list = RequirementList(
                self._finalize_requirements(all_requirements)
            )
            self._set_stat("requirements_written", len(requirement_list))
            self.excel_writer.write(requirement_list)
            if self.write_run_manifest:
                self._write_run_manifest(
                    chunk_cache_sources=cache_files,
                    used_live_ocr=False,
                    used_chunk_cache_replay=True,
                )
            return

        if self.config.input.mode == "markdown":
            source_files = self._collect_markdown_files(input_path)
            all_requirements = self._extract_from_markdown_files(source_files)
            requirement_list = RequirementList(
                self._finalize_requirements(all_requirements)
            )
            self._set_stat("requirements_written", len(requirement_list))
            self.excel_writer.write(requirement_list)
            if self.write_run_manifest:
                self._write_run_manifest(
                    markdown_sources=source_files,
                    used_live_ocr=False,
                    used_chunk_cache_replay=False,
                )
            return

        raise ValueError(f"Unsupported input mode: {self.config.input.mode}")

    def prepare_pdf_chunks(
        self, pdf_path: str, *, chunk_id_prefix: str = ""
    ) -> list[AnchoredMarkdownChunk]:
        logger.info("Preparing PDF without extraction: %s", pdf_path)
        semantic_document = self.document_parser.parse_pdf(pdf_path)
        chunks = self.chunker.chunk_document(semantic_document)
        if chunk_id_prefix:
            for chunk in chunks:
                chunk.chunk_id = f"{chunk_id_prefix}{chunk.chunk_id}"
        self._increment_stat("chunks_prepared", len(chunks))
        self._persist_anchored_markdown(pdf_path, chunks)
        self._persist_chunk_cache(pdf_path, chunks)
        return chunks

    def prepare_pdf(self, pdf_path: str) -> list[AnchoredMarkdownChunk]:
        if self._batch_mode_enabled():
            return self.prepare_pdf_in_batches(pdf_path)
        return self.prepare_pdf_chunks(pdf_path)

    def extract_requirements_from_pdf(self, pdf_path: str) -> RequirementList:
        logger.info("Starting requirement extraction from PDF: %s", pdf_path)
        chunks = self.prepare_pdf_chunks(pdf_path)
        return self.extract_requirements_from_chunks(chunks)

    def extract_requirements_from_chunk_cache(self, cache_path: str) -> RequirementList:
        logger.info("Starting requirement extraction from chunk cache: %s", cache_path)
        chunks = load_chunk_cache(cache_path)
        self._increment_stat("chunks_loaded_from_cache", len(chunks))
        return self.extract_requirements_from_chunks(chunks)

    def extract_requirements_from_markdown(self, md_path: str) -> RequirementList:
        logger.info("Starting requirement extraction from Markdown: %s", md_path)
        docs = self.markdown_splitter.split_markdown(markdown_path=md_path)
        chunks = self._markdown_docs_to_chunks(md_path, docs)
        return self.extract_requirements_from_chunks(chunks)

    def extract_requirements_from_chunks(
        self, chunks: list[AnchoredMarkdownChunk]
    ) -> RequirementList:
        requirement_schema = RequirementExtractionResult.model_json_schema()
        if self.parallel_enabled:
            requirements = self.process_chunks_parallel(chunks, requirement_schema)
        else:
            requirements = self.process_chunks_sequential(chunks, requirement_schema)
        return RequirementList(self._finalize_requirements(requirements))

    def _collect_pdfs(self, path: Path) -> Iterable[Path]:
        sources = collect_input_files(
            path, suffixes=(".pdf",), recursive=self.config.input.recursive
        )
        self._source_files["pdf"] = sources
        return sources

    def _collect_markdown_files(self, path: Path) -> Iterable[Path]:
        suffixes = tuple(self.config.input.file_extensions)
        if suffixes == (".pdf",):
            suffixes = (".md", ".markdown")
        sources = collect_input_files(
            path, suffixes=suffixes, recursive=self.config.input.recursive
        )
        self._source_files["markdown"] = sources
        return sources

    def _collect_chunk_cache_files(self, path: Path) -> Iterable[Path]:
        sources = collect_input_files(
            path, suffixes=(".chunks.json",), recursive=self.config.input.recursive
        )
        self._source_files["chunk_cache"] = sources
        return sources

    def _extract_from_pdfs(self, pdf_files: Iterable[Path]) -> list[Requirement]:
        pdf_files = list(pdf_files)
        all_requirements: list[Requirement] = []
        for pdf_file in pdf_files:
            self._increment_stat("documents_processed")
            extractor = (
                self
                if len(pdf_files) == 1
                else self._spawn_document_extractor(pdf_file)
            )
            try:
                if extractor._batch_mode_enabled():
                    requirements = extractor.extract_requirements_from_pdf_in_batches(
                        str(pdf_file)
                    )
                else:
                    requirements = extractor.extract_requirements_from_pdf(
                        str(pdf_file)
                    )
            finally:
                if extractor is not self:
                    self._merge_run_stats(extractor.run_stats)
            if requirements.root or self.config.extraction.allow_empty_results:
                all_requirements.extend(requirements.root)
        return all_requirements

    def _extract_from_markdown_files(
        self, markdown_files: Iterable[Path]
    ) -> list[Requirement]:
        all_requirements: list[Requirement] = []
        for md_file in markdown_files:
            self._increment_stat("documents_processed")
            requirements = self.extract_requirements_from_markdown(str(md_file))
            if requirements.root or self.config.extraction.allow_empty_results:
                all_requirements.extend(requirements.root)
        return all_requirements

    def _extract_from_chunk_cache_files(
        self, cache_files: Iterable[Path]
    ) -> list[Requirement]:
        all_requirements: list[Requirement] = []
        for cache_file in cache_files:
            self._increment_stat("documents_processed")
            requirements = self.extract_requirements_from_chunk_cache(str(cache_file))
            if requirements.root or self.config.extraction.allow_empty_results:
                all_requirements.extend(requirements.root)
        return all_requirements

    def _prepare_pdfs(self, pdf_files: Iterable[Path]) -> None:
        pdf_files = list(pdf_files)
        for pdf_file in pdf_files:
            self._increment_stat("documents_processed")
            extractor = (
                self
                if len(pdf_files) == 1
                else self._spawn_document_extractor(pdf_file)
            )
            try:
                extractor.prepare_pdf(str(pdf_file))
            finally:
                if extractor is not self:
                    self._merge_run_stats(extractor.run_stats)

    def _spawn_document_extractor(self, pdf_path: Path) -> "RequirementExtractor":
        """Keep each PDF's images, OCR input and pruning report in its own directory."""
        config = copy.deepcopy(self.config)
        config.input.path = str(pdf_path)
        directory = Path(self.config.output.directory) / "documents" / pdf_path.stem
        config.output.directory = str(directory)
        self._pdf_artifact_dirs[str(pdf_path)] = directory
        return RequirementExtractor(
            config=config,
            ollama_client=self.ollama_client,
            command_name=self.command_name,
            write_run_manifest=False,
        )

    def process_single_chunk(
        self, chunk: AnchoredMarkdownChunk, requirement_schema: dict
    ) -> RequirementList:
        self._increment_stat("chunks_processed")
        try:
            prompt = get_prompt(
                "requirement_extraction",
                chunk.markdown_text,
                json.dumps(requirement_schema),
                include_few_shot=True,
            )
            response = self.ollama_client.get_structured_response(
                prompt,
                model_name=self.model_name,
                response_schema=requirement_schema,
            )
            if response is None:
                logger.error(
                    "Failed to get response from LLM for chunk %s", chunk.chunk_id
                )
                self._increment_stat("llm_response_failures")
                return RequirementList([])
            requirements = self.parse_llm_response(response)
            enriched = self._apply_chunk_provenance(requirements, chunk)
            self._increment_stat("requirements_accepted", len(enriched))
            return enriched
        except Exception as exc:
            logger.error("Error processing chunk %s: %s", chunk.chunk_id, exc)
            self._increment_stat("chunk_processing_errors")
            return RequirementList([])

    def process_chunks_parallel(
        self, chunks: list[AnchoredMarkdownChunk], requirement_schema: dict
    ) -> list[Requirement]:
        all_requirements: list[Requirement] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_chunk = {
                executor.submit(
                    self.process_single_chunk, chunk, requirement_schema
                ): chunk
                for chunk in chunks
            }
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_requirements = future.result()
                    logger.info(
                        "Processed chunk %s: found %s requirements",
                        chunk.chunk_id,
                        len(chunk_requirements),
                    )
                    if (
                        chunk_requirements.root
                        or self.config.extraction.allow_empty_results
                    ):
                        all_requirements.extend(chunk_requirements.root)
                    if not chunk_requirements.root:
                        self._increment_stat("empty_chunk_results")
                except Exception as exc:
                    logger.error("Error processing chunk %s: %s", chunk.chunk_id, exc)
                    self._increment_stat("chunk_processing_errors")
        return all_requirements

    def process_chunks_sequential(
        self, chunks: list[AnchoredMarkdownChunk], requirement_schema: dict
    ) -> list[Requirement]:
        all_requirements: list[Requirement] = []
        for chunk in chunks:
            chunk_requirements = self.process_single_chunk(chunk, requirement_schema)
            logger.info(
                "Processed chunk %s: found %s requirements",
                chunk.chunk_id,
                len(chunk_requirements),
            )
            if chunk_requirements.root or self.config.extraction.allow_empty_results:
                all_requirements.extend(chunk_requirements.root)
            if not chunk_requirements.root:
                self._increment_stat("empty_chunk_results")
        return all_requirements

    def parse_llm_response(self, json_response: str) -> RequirementList:
        try:
            data = json.loads(json_response)
            if isinstance(data, dict):
                if "requirements" in data:
                    data = data["requirements"]
                elif "requirement" in data:
                    data = [data["requirement"]]
                elif "code" in data or "description" in data:
                    data = [data]
                else:
                    raise ValueError(
                        "LLM response has no recognized requirements field."
                    )
            if not isinstance(data, list):
                raise ValueError("LLM requirements response must be a list.")
            extracted = RequirementExtractionResult.model_validate(data)
            valid_items = [
                item
                for item in extracted.root
                if item.code is not None and item.description is not None
            ]
            dropped_count = len(extracted.root) - len(valid_items)
            if dropped_count:
                logger.warning(
                    "Dropped %s extracted item(s) because every requirement must include both code and description.",
                    dropped_count,
                )
                self._increment_stat("invalid_items_dropped", dropped_count)
            return RequirementList(
                [
                    Requirement(
                        code=item.code,
                        description=item.description,
                        source_segment_ids=list(item.source_segment_ids),
                    )
                    for item in valid_items
                ]
            )
        except json.JSONDecodeError as exc:
            self._increment_stat("json_parse_failures")
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

    def _markdown_docs_to_chunks(
        self, md_path: str, docs: list[Document]
    ) -> list[AnchoredMarkdownChunk]:
        chunks: list[AnchoredMarkdownChunk] = []
        for index, doc in enumerate(docs, start=1):
            section_parts = []
            header_1 = doc.metadata.get("Header 1")
            header_2 = doc.metadata.get("Header 2")
            if header_1:
                section_parts.append(header_1)
            if header_2:
                section_parts.append(header_2)
            elif isinstance(doc.metadata.get("merged_from"), list):
                merged_headers = [
                    header
                    for header in doc.metadata["merged_from"]
                    if isinstance(header, str) and header and header != "UNKNOWN"
                ]
                if merged_headers:
                    section_parts.append(merged_headers[0])
            chunk_text = doc.page_content.strip()
            chunks.append(
                AnchoredMarkdownChunk(
                    chunk_id=f"markdown-{index}",
                    markdown_text=chunk_text,
                    source_document=md_path,
                    source_page_start=None,
                    source_page_end=None,
                    segment_ids=[],
                    source_block_ids=[],
                    source_section=" / ".join(section_parts) if section_parts else None,
                    source_text_excerpt=chunk_text[:400],
                    source_bbox_list=[],
                    source_regions=[],
                    source_segments=[],
                )
            )
        return chunks

    def _apply_chunk_provenance(
        self, requirements: RequirementList, chunk: AnchoredMarkdownChunk
    ) -> RequirementList:
        enriched = []
        known_segments = {
            segment.segment_id: segment for segment in chunk.source_segments
        }
        known_regions = {
            region.get("segment_id"): region for region in chunk.source_regions
        }

        for requirement in requirements:
            valid_segment_ids = []
            invalid_segment_ids = []
            for segment_id in requirement.source_segment_ids:
                if segment_id in known_segments:
                    if segment_id not in valid_segment_ids:
                        valid_segment_ids.append(segment_id)
                else:
                    invalid_segment_ids.append(segment_id)

            resolved_segments = [
                known_segments[segment_id] for segment_id in valid_segment_ids
            ]
            resolved_regions = [
                known_regions[segment_id]
                for segment_id in valid_segment_ids
                if segment_id in known_regions
            ]
            review_status = self._derive_review_status(
                existing_status=requirement.review_status,
                valid_segment_ids=valid_segment_ids,
                invalid_segment_ids=invalid_segment_ids,
            )

            if resolved_segments:
                source_page_start = min(
                    segment.page_number for segment in resolved_segments
                )
                source_page_end = max(
                    segment.page_number for segment in resolved_segments
                )
                source_block_ids = [
                    block_id
                    for segment in resolved_segments
                    for block_id in segment.source_block_ids
                ]
                source_section = (
                    self._resolve_section(resolved_segments) or chunk.source_section
                )
                source_text_excerpt = self._resolve_excerpt(resolved_segments)
                source_bbox_list = [
                    list(segment.bbox)
                    for segment in resolved_segments
                    if segment.bbox is not None
                ]
                source_regions = resolved_regions
            else:
                source_page_start = chunk.source_page_start
                source_page_end = chunk.source_page_end
                source_block_ids = list(chunk.source_block_ids)
                source_section = chunk.source_section
                source_text_excerpt = chunk.source_text_excerpt
                source_bbox_list = list(chunk.source_bbox_list)
                source_regions = list(chunk.source_regions)

            enriched.append(
                Requirement(
                    code=requirement.code,
                    description=requirement.description,
                    source_segment_ids=valid_segment_ids,
                    source_document=requirement.source_document
                    or chunk.source_document,
                    source_chunk_id=requirement.source_chunk_id or chunk.chunk_id,
                    source_page_start=requirement.source_page_start
                    or source_page_start,
                    source_page_end=requirement.source_page_end or source_page_end,
                    source_block_ids=requirement.source_block_ids or source_block_ids,
                    source_section=requirement.source_section or source_section,
                    source_text_excerpt=requirement.source_text_excerpt
                    or source_text_excerpt,
                    source_bbox_list=requirement.source_bbox_list or source_bbox_list,
                    source_regions=requirement.source_regions or source_regions,
                    confidence=requirement.confidence,
                    review_status=review_status,
                )
            )
        return RequirementList(enriched)

    def _resolve_section(self, segments: list[SemanticSegment]) -> Optional[str]:
        for segment in reversed(segments):
            label = segment.section_label()
            if label:
                return label
        return None

    def _resolve_excerpt(self, segments: list[SemanticSegment]) -> str:
        excerpt = "\n\n".join(
            segment.text_content.strip()
            for segment in segments
            if segment.text_content.strip()
        )
        return excerpt[:400]

    def _derive_review_status(
        self,
        existing_status: Optional[str],
        valid_segment_ids: list[str],
        invalid_segment_ids: list[str],
    ) -> Optional[str]:
        if existing_status:
            return existing_status
        if invalid_segment_ids and valid_segment_ids:
            return "partial_invalid_source_segment_ids"
        if invalid_segment_ids and not valid_segment_ids:
            return "invalid_source_segment_ids"
        if not valid_segment_ids:
            if self.config.extraction.allow_uncited_results:
                return "uncited_source_segments"
            return "needs_source_segment_review"
        return None

    def _deduplicate_requirements(
        self, requirements: list[Requirement]
    ) -> list[Requirement]:
        if not self.config.extraction.deduplicate_requirements:
            return self._sort_requirements(requirements)
        seen = set()
        unique_requirements = []
        for requirement in self._sort_requirements(requirements):
            code = (
                requirement.code.upper()
                if requirement.code and self.config.extraction.normalize_codes
                else requirement.code
            )
            key = (requirement.source_document, code, requirement.description)
            if key in seen:
                continue
            seen.add(key)
            unique_requirements.append(requirement)
        return unique_requirements

    def _finalize_requirements(
        self, requirements: list[Requirement]
    ) -> list[Requirement]:
        deduplicated = self._deduplicate_requirements(requirements)
        return self._sort_requirements(deduplicated)

    def _sort_requirements(self, requirements: list[Requirement]) -> list[Requirement]:
        return sorted(requirements, key=self._requirement_sort_key)

    def _requirement_sort_key(self, requirement: Requirement) -> tuple:
        page_start = (
            requirement.source_page_start
            if requirement.source_page_start is not None
            else float("inf")
        )
        page_end = (
            requirement.source_page_end
            if requirement.source_page_end is not None
            else float("inf")
        )
        first_block_order = self._first_block_order_for_requirement(requirement)
        chunk_key = self._natural_sort_key(requirement.source_chunk_id)
        code = (
            requirement.code.upper()
            if requirement.code and self.config.extraction.normalize_codes
            else (requirement.code or "")
        )
        return (page_start, page_end, first_block_order, chunk_key, code)

    def _first_block_order_for_requirement(self, requirement: Requirement) -> float:
        earliest_page = requirement.source_page_start
        candidate_orders: list[int] = []
        for region in requirement.source_regions:
            page_number = region.get("page_number")
            block_order = region.get("block_order")
            if block_order is None:
                continue
            if earliest_page is None or page_number == earliest_page:
                candidate_orders.append(int(block_order))
        if not candidate_orders:
            return float("inf")
        return float(min(candidate_orders))

    def _natural_sort_key(self, value: Optional[str]) -> tuple:
        if value is None:
            return (float("inf"),)
        parts: list[int | str] = []
        for token in self._split_digits(value):
            parts.append(int(token) if token.isdigit() else token.lower())
        return tuple(parts)

    def _split_digits(self, value: str) -> list[str]:
        current = []
        last_is_digit: Optional[bool] = None
        parts: list[str] = []
        for char in value:
            is_digit = char.isdigit()
            if last_is_digit is None or is_digit == last_is_digit:
                current.append(char)
            else:
                parts.append("".join(current))
                current = [char]
            last_is_digit = is_digit
        if current:
            parts.append("".join(current))
        return parts

    def _persist_anchored_markdown(
        self, pdf_path: str, chunks: list[AnchoredMarkdownChunk]
    ) -> Optional[str]:
        if not self.config.parser.persist_anchored_markdown:
            return None
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(pdf_path).stem}.anchored.md"
        output_path.write_text(
            "\n\n".join(
                chunk.markdown_text.strip()
                for chunk in chunks
                if chunk.markdown_text.strip()
            ).rstrip()
            + "\n",
            encoding="utf-8",
        )
        return str(output_path)

    def _persist_chunk_cache(
        self, source_path: str, chunks: list[AnchoredMarkdownChunk]
    ) -> str:
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(source_path).stem}.chunks.json"
        return write_chunk_cache(chunks, output_path, source_document=source_path)

    def prepare_pdf_in_batches(self, pdf_path: str) -> list[AnchoredMarkdownChunk]:
        page_count = self._count_pdf_pages(pdf_path)
        batch_windows = self._build_batch_windows(page_count)
        pdf_stem = Path(pdf_path).stem
        batch_root = (
            Path(self.config.output.directory)
            / self.config.parser.batch_output_subdir
            / pdf_stem
        )
        batch_root.mkdir(parents=True, exist_ok=True)
        toc_pruning_plan = self._build_batch_toc_pruning_plan(pdf_path)
        self._persist_batch_toc_pruning_report(toc_pruning_plan)

        merged_chunks: list[AnchoredMarkdownChunk] = []
        anchored_parts: list[str] = []

        for start_page, end_page in batch_windows:
            slice_dir = batch_root / f"p{start_page:03d}-{end_page:03d}"
            batch_config = self._build_batch_config(slice_dir, start_page, end_page)
            batch_extractor = self._spawn_batch_extractor(
                batch_config, toc_pruning_plan=toc_pruning_plan
            )
            try:
                batch_chunks = batch_extractor.prepare_pdf_chunks(
                    pdf_path,
                    chunk_id_prefix=f"p{start_page:03d}-{end_page:03d}-",
                )
            finally:
                self._merge_run_stats(batch_extractor.run_stats)
            merged_chunks.extend(batch_chunks)

            anchored_path = slice_dir / f"{pdf_stem}.anchored.md"
            if anchored_path.exists():
                anchored_parts.append(anchored_path.read_text(encoding="utf-8").strip())

        self._persist_combined_batch_anchored_markdown(pdf_path, anchored_parts)
        self._persist_combined_batch_chunk_cache(pdf_path, merged_chunks)
        return merged_chunks

    def extract_requirements_from_pdf_in_batches(
        self, pdf_path: str
    ) -> RequirementList:
        pdf_stem = Path(pdf_path).stem
        all_requirements: list[Requirement] = []
        page_count = self._count_pdf_pages(pdf_path)
        batch_windows = self._build_batch_windows(page_count)
        batch_root = (
            Path(self.config.output.directory)
            / self.config.parser.batch_output_subdir
            / pdf_stem
        )
        toc_pruning_plan = self._build_batch_toc_pruning_plan(pdf_path)
        self._persist_batch_toc_pruning_report(toc_pruning_plan)
        merged_chunks: list[AnchoredMarkdownChunk] = []
        anchored_parts: list[str] = []

        for start_page, end_page in batch_windows:
            slice_dir = batch_root / f"p{start_page:03d}-{end_page:03d}"
            batch_config = self._build_batch_config(slice_dir, start_page, end_page)
            batch_extractor = self._spawn_batch_extractor(
                batch_config, toc_pruning_plan=toc_pruning_plan
            )
            try:
                batch_chunks = batch_extractor.prepare_pdf_chunks(
                    pdf_path,
                    chunk_id_prefix=f"p{start_page:03d}-{end_page:03d}-",
                )
                batch_requirements = batch_extractor.extract_requirements_from_chunks(
                    batch_chunks
                )
                batch_extractor.excel_writer.write(batch_requirements)
            finally:
                self._merge_run_stats(batch_extractor.run_stats)
            merged_chunks.extend(batch_chunks)

            if batch_requirements.root or self.config.extraction.allow_empty_results:
                all_requirements.extend(batch_requirements.root)

            anchored_path = slice_dir / f"{pdf_stem}.anchored.md"
            if anchored_path.exists():
                anchored_parts.append(anchored_path.read_text(encoding="utf-8").strip())

        self._persist_combined_batch_anchored_markdown(pdf_path, anchored_parts)
        self._persist_combined_batch_chunk_cache(pdf_path, merged_chunks)
        return RequirementList(self._finalize_requirements(all_requirements))

    def _batch_mode_enabled(self) -> bool:
        return (self.config.parser.batch_page_count or 0) > 0

    def _count_pdf_pages(self, pdf_path: str) -> int:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required to count PDF pages for batch mode."
            ) from exc

        with fitz.open(pdf_path) as document:
            return document.page_count

    def _build_batch_windows(self, page_count: int) -> list[tuple[int, int]]:
        batch_page_count = self.config.parser.batch_page_count
        if batch_page_count is None or batch_page_count <= 0:
            raise ValueError(
                "parser.batch_page_count must be a positive integer when batch mode is enabled."
            )

        start_page = max(self.config.parser.page_start or 1, 1)
        end_page = self.config.parser.page_end or page_count
        end_page = min(end_page, page_count)

        if self.config.parser.max_pages is not None:
            end_page = min(
                end_page, start_page + max(self.config.parser.max_pages - 1, 0)
            )

        if start_page > page_count:
            raise ValueError(
                f"Configured parser.page_start={start_page} is beyond the document page count ({page_count})."
            )

        windows: list[tuple[int, int]] = []
        current = start_page
        while current <= end_page:
            window_end = min(current + batch_page_count - 1, end_page)
            windows.append((current, window_end))
            current = window_end + 1
        return windows

    def _build_batch_config(
        self, slice_dir: Path, start_page: int, end_page: int
    ) -> AppConfig:
        batch_config = copy.deepcopy(self.config)
        batch_config.output.directory = str(slice_dir)
        batch_config.parser.page_start = start_page
        batch_config.parser.page_end = end_page
        batch_config.parser.max_pages = None
        batch_config.parser.batch_page_count = None
        return batch_config

    def _spawn_batch_extractor(
        self,
        batch_config: AppConfig,
        *,
        toc_pruning_plan: Optional[dict] = None,
    ) -> "RequirementExtractor":
        return RequirementExtractor(
            config=batch_config,
            ollama_client=self.ollama_client,
            toc_pruning_plan=toc_pruning_plan,
            command_name=self.command_name,
            write_run_manifest=False,
        )

    def _persist_combined_batch_anchored_markdown(
        self, pdf_path: str, anchored_parts: list[str]
    ) -> None:
        if not self.config.parser.persist_anchored_markdown or not anchored_parts:
            return
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(pdf_path).stem}.anchored.md"
        output_path.write_text(
            "\n\n".join(part for part in anchored_parts if part).rstrip() + "\n",
            encoding="utf-8",
        )

    def _persist_combined_batch_chunk_cache(
        self, pdf_path: str, chunks: list[AnchoredMarkdownChunk]
    ) -> str:
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(pdf_path).stem}.chunks.json"
        return write_chunk_cache(chunks, output_path, source_document=pdf_path)

    def _build_batch_toc_pruning_plan(self, pdf_path: str) -> Optional[dict]:
        if self.config.parser.toc_section_pruning_mode == "off":
            return None
        build_plan = getattr(self.document_parser, "build_toc_pruning_plan", None)
        if not callable(build_plan):
            return None
        try:
            return build_plan(pdf_path)
        except Exception as exc:
            logger.warning("Failed to build TOC pruning plan for %s: %s", pdf_path, exc)
            return None

    def _persist_batch_toc_pruning_report(
        self, toc_pruning_plan: Optional[dict]
    ) -> None:
        if not toc_pruning_plan:
            return
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.config.parser.toc_pruning_report_filename
        output_path.write_text(json.dumps(toc_pruning_plan, indent=2), encoding="utf-8")

    def _new_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return f"run-{timestamp}-{uuid4().hex[:8]}"

    def _new_run_stats(self) -> dict:
        return {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "duration_seconds": None,
            "documents_processed": 0,
            "chunks_prepared": 0,
            "chunks_loaded_from_cache": 0,
            "chunks_processed": 0,
            "requirements_accepted": 0,
            "requirements_written": 0,
            "invalid_items_dropped": 0,
            "json_parse_failures": 0,
            "llm_response_failures": 0,
            "chunk_processing_errors": 0,
            "empty_chunk_results": 0,
            "error_type": None,
            "error_message": None,
        }

    def _increment_stat(self, key: str, amount: int = 1) -> None:
        with self._run_stats_lock:
            self.run_stats[key] = int(self.run_stats.get(key, 0) or 0) + amount

    def _set_stat(self, key: str, value) -> None:
        with self._run_stats_lock:
            self.run_stats[key] = value

    def _merge_run_stats(self, child_stats: dict) -> None:
        for key, value in child_stats.items():
            if key in {"status", "started_at", "finished_at", "duration_seconds"}:
                continue
            if isinstance(value, int):
                self._increment_stat(key, value)

    def _finish_run_stats(
        self, status: str = "completed", error: Optional[Exception] = None
    ) -> dict:
        with self._run_stats_lock:
            self.run_stats["status"] = status
            self.run_stats["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.run_stats["duration_seconds"] = round(
                time.perf_counter() - self._run_started_at_perf, 3
            )
            self.run_stats["error_type"] = (
                type(error).__name__ if error is not None else None
            )
            self.run_stats["error_message"] = str(error) if error is not None else None
            return dict(self.run_stats)

    def _partial_extraction_error(self) -> PartialExtractionError | None:
        """Describe data loss without conflating it with valid empty responses."""
        counters = (
            "json_parse_failures",
            "llm_response_failures",
            "chunk_processing_errors",
            "invalid_items_dropped",
        )
        errors = {
            key: int(self.run_stats.get(key, 0) or 0)
            for key in counters
            if self.run_stats.get(key, 0)
        }
        if not errors:
            return None
        details = ", ".join(f"{key}={value}" for key, value in errors.items())
        return PartialExtractionError(
            f"Extraction is partial; available results were exported ({details})."
        )

    def _write_run_stats(
        self, status: str = "completed", error: Optional[Exception] = None
    ) -> dict:
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = self._finish_run_stats(status=status, error=error)
        (output_dir / "run-stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        return stats

    def _write_failed_run_manifest(self, error: Exception) -> None:
        mode = self.config.input.mode
        self._write_run_manifest(
            pdf_sources=self._source_files.get("pdf", []),
            markdown_sources=self._source_files.get("markdown", []),
            chunk_cache_sources=self._source_files.get("chunk_cache", []),
            used_live_ocr=mode == "pdf",
            used_chunk_cache_replay=mode == "chunk_cache",
            status="failed",
            error=error,
        )

    def _write_run_manifest(
        self,
        *,
        pdf_sources: Iterable[Path] = (),
        markdown_sources: Iterable[Path] = (),
        chunk_cache_sources: Iterable[Path] = (),
        used_live_ocr: bool,
        used_chunk_cache_replay: bool,
        status: str = "completed",
        error: Optional[Exception] = None,
    ) -> None:
        if status == "completed":
            partial_error = self._partial_extraction_error()
            if partial_error is not None:
                status, error = "partial", partial_error
        pdf_sources = tuple(pdf_sources)
        markdown_sources = tuple(markdown_sources)
        chunk_cache_sources = tuple(chunk_cache_sources)
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "run-manifest.json"
        run_stats = self._write_run_stats(status=status, error=error)
        artifacts = self._build_artifact_manifest_entries(
            pdf_sources=pdf_sources,
            markdown_sources=markdown_sources,
            chunk_cache_sources=chunk_cache_sources,
        )
        manifest = {
            "schema_version": 2,
            "run_id": self.run_id,
            "command": self.command_name,
            "input": {
                "mode": self.config.input.mode,
                "path": self.config.input.path,
                "recursive": self.config.input.recursive,
                "pdf_sources": [str(path) for path in pdf_sources],
                "markdown_sources": [str(path) for path in markdown_sources],
                "chunk_cache_sources": [str(path) for path in chunk_cache_sources],
            },
            "execution": {
                "used_live_ocr": used_live_ocr,
                "used_chunk_cache_replay": used_chunk_cache_replay,
                "status": status,
                "error": (
                    {"type": type(error).__name__, "message": str(error)}
                    if error is not None
                    else None
                ),
            },
            "parser": asdict(self.config.parser),
            "chunking": asdict(self.config.chunking),
            "extraction": asdict(self.config.extraction),
            "parallel": asdict(self.config.parallel),
            "ollama": asdict(self.config.ollama),
            "output": asdict(self.config.output),
            "logging": asdict(self.config.logging),
            "run_stats": run_stats,
            "artifacts": artifacts,
            "reproducibility": self._build_reproducibility_metadata(
                pdf_sources=pdf_sources,
                markdown_sources=markdown_sources,
                chunk_cache_sources=chunk_cache_sources,
                artifacts=artifacts,
            ),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _build_reproducibility_metadata(
        self,
        *,
        pdf_sources: Iterable[Path],
        markdown_sources: Iterable[Path],
        chunk_cache_sources: Iterable[Path],
        artifacts: dict,
    ) -> dict:
        input_paths: list[str | Path] = [self.config.input.path]
        input_paths.extend(pdf_sources)
        input_paths.extend(markdown_sources)
        input_paths.extend(chunk_cache_sources)

        artifact_paths = list(artifacts.get("exports", {}).values())
        for source_entry in artifacts.get("pdf_sources", []):
            artifact_paths.extend(
                value
                for key, value in source_entry.items()
                if key != "source" and isinstance(value, str)
            )

        model_metadata: dict = {"name": self.model_name, "digest": None}
        if self.command_name != "prepare-pdf" and self.run_stats["chunks_processed"]:
            resolver = getattr(self.ollama_client, "get_model_metadata", None)
            if callable(resolver):
                try:
                    resolved_metadata = resolver(self.model_name)
                    if isinstance(resolved_metadata, dict):
                        model_metadata.update(resolved_metadata)
                except Exception as exc:
                    logger.warning(
                        "Could not record metadata for model %s: %s",
                        self.model_name,
                        exc,
                    )

        repo_root = Path(__file__).resolve().parents[2]
        return {
            "git": collect_git_metadata(repo_root),
            "runtime": collect_runtime_metadata(),
            "prompt": get_requirement_extraction_prompt_fingerprint(),
            "model": model_metadata,
            "inputs": collect_file_metadata(input_paths),
            "artifacts": collect_file_metadata(artifact_paths),
        }

    def _build_artifact_manifest_entries(
        self,
        *,
        pdf_sources: Iterable[Path],
        markdown_sources: Iterable[Path],
        chunk_cache_sources: Iterable[Path],
    ) -> dict:
        output_dir = Path(self.config.output.directory)
        exports = {}
        for filename_key, filename, enabled in (
            (
                "requirements_xlsx",
                "requirements.xlsx",
                self.command_name != "prepare-pdf",
            ),
            (
                "requirements_review_json",
                self.config.output.review_artifact_filename,
                self.command_name != "prepare-pdf"
                and self.config.output.write_review_artifact,
            ),
            (
                "requirements_review_md",
                self.config.output.review_markdown_filename,
                self.command_name != "prepare-pdf"
                and self.config.output.write_review_markdown,
            ),
            (
                "requirements_review_html",
                self.config.output.review_html_filename,
                self.command_name != "prepare-pdf"
                and self.config.output.write_review_html,
            ),
            ("run_stats", "run-stats.json", True),
        ):
            path = output_dir / filename
            if enabled and path.is_file():
                exports[filename_key] = str(path)

        pdf_entries = [
            self._build_pdf_source_artifact_entry(path) for path in pdf_sources
        ]
        markdown_entries = [
            {
                "source": str(path),
            }
            for path in markdown_sources
        ]
        chunk_cache_entries = [
            {
                "source": str(path),
            }
            for path in chunk_cache_sources
        ]
        artifacts = {
            "output_directory": str(output_dir),
            "exports": exports,
            "pdf_sources": pdf_entries,
            "markdown_sources": markdown_entries,
            "chunk_cache_sources": chunk_cache_entries,
        }
        return artifacts

    def _build_pdf_source_artifact_entry(self, pdf_path: Path) -> dict:
        output_dir = self._pdf_artifact_dirs.get(
            str(pdf_path), Path(self.config.output.directory)
        )
        pdf_stem = pdf_path.stem
        entry = {
            "source": str(pdf_path),
            "anchored_markdown": str(output_dir / f"{pdf_stem}.anchored.md"),
            "chunk_cache": str(output_dir / f"{pdf_stem}.chunks.json"),
            "toc_pruning_report": str(
                output_dir / self.config.parser.toc_pruning_report_filename
            ),
        }
        if self._batch_mode_enabled():
            batch_root = output_dir / self.config.parser.batch_output_subdir / pdf_stem
            entry["batch_root"] = str(batch_root)
            try:
                entry["page_windows"] = [
                    {"start_page": start_page, "end_page": end_page}
                    for start_page, end_page in self._build_batch_windows(
                        self._count_pdf_pages(str(pdf_path))
                    )
                ]
            except Exception as exc:
                # A corrupt PDF may itself be the cause of a failed run.
                entry["page_windows"] = []
                logger.warning(
                    "Could not record page windows for %s: %s", pdf_path, exc
                )
        else:
            entry["ocr_input_pdf"] = str(output_dir / "ocr-input.pdf")
            entry["page_images_dir"] = str(output_dir / "page-images")
            entry["ocr_pages_dir"] = str(output_dir / "ocr-pages")
        # Record artifacts actually produced, including in partial or failed runs.
        return {
            key: value
            for key, value in entry.items()
            if key == "source" or not isinstance(value, str) or Path(value).exists()
        }


def extract_requirements_from_pdf(pdf_path: str, config: AppConfig) -> RequirementList:
    extractor = RequirementExtractor(config=config)
    return extractor.extract_requirements_from_pdf(pdf_path)
