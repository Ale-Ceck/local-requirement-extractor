from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from contextlib import suppress
from typing import Any, Dict, Iterable, List, Optional, Sequence

from config.schema import ParserConfig
from src.pdf_processing.models import SemanticDocument, SemanticPage, SemanticSegment
from src.utils.logging_config import setup_logger
from src.vlm_service import VLMServiceManager

logger = setup_logger(__name__)

NOTE_PREFIXES = ("note", "warning", "exception", "constraint", "caution")
TOC_SCAN_PAGE_LIMIT = 12
TOC_MIN_ENTRY_COUNT = 3
TOC_ENTRY_PATTERNS = (
    re.compile(r"^\s*(?P<title>.+?)\s*\.{2,}\s*(?P<page>\d+)\s*$"),
    re.compile(r"^\s*(?P<title>.+?)\s{2,}(?P<page>\d+)\s*$"),
)


class PaddleOCRVLParser:
    """Structured PDF parser backed by PaddleOCR-VL when the optional dependency is installed."""

    def __init__(
        self,
        config: ParserConfig,
        output_directory: Optional[str] = None,
        toc_pruning_plan: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.output_directory = Path(output_directory) if output_directory else None
        self._toc_pruning_plan = copy.deepcopy(toc_pruning_plan) if toc_pruning_plan is not None else None
        self._pipeline = None
        self._vlm_service = VLMServiceManager(config)
        self._rasterized_pages_dir: Optional[tempfile.TemporaryDirectory] = None
        self._sliced_pdfs_dir: Optional[tempfile.TemporaryDirectory] = None

    def parse_pdf(self, pdf_path: str) -> SemanticDocument:
        pipeline = self._get_pipeline()
        parse_input_path, page_offset = self._prepare_pdf_input(Path(pdf_path))
        predict_input_path = self._persist_parse_input(parse_input_path)
        try:
            result = pipeline.predict(
                input=str(predict_input_path),
                **self._build_predict_kwargs(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse PDF '{pdf_path}' using PaddleOCR-VL service "
                f"'{self.config.vlm_server_url}': {exc}"
            ) from exc
        return self._build_document(
            pdf_path,
            result,
            page_offset=page_offset,
            parse_source_path=predict_input_path,
        )

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR-VL support requires the optional 'paddleocr[doc-parser]' dependency."
            ) from exc

        if self.config.vlm_backend.endswith("server"):
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            self._vlm_service.ensure_healthy()

        try:
            self._pipeline = PaddleOCRVL(**self._build_pipeline_kwargs())
        except ModuleNotFoundError as exc:
            if exc.name == "paddle" or "No module named 'paddle'" in str(exc):
                raise RuntimeError(
                    "Failed to initialize PaddleOCR-VL because 'paddlepaddle' is not installed "
                    "in the active environment."
                ) from exc
            raise RuntimeError(
                f"Failed to initialize PaddleOCR-VL for backend '{self.config.vlm_backend}' "
                f"at '{self.config.vlm_server_url}': {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize PaddleOCR-VL for backend '{self.config.vlm_backend}' "
                f"at '{self.config.vlm_server_url}': {exc}"
            ) from exc
        return self._pipeline

    def _build_pipeline_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "vl_rec_backend": self.config.vlm_backend,
            "vl_rec_server_url": self.config.vlm_server_url,
            "vl_rec_api_model_name": self.config.vlm_api_model_name,
        }
        if self.config.vlm_api_key:
            kwargs["vl_rec_api_key"] = self.config.vlm_api_key
        return kwargs

    def _build_predict_kwargs(self) -> Dict[str, Any]:
        return {
            "use_doc_orientation_classify": self.config.use_doc_orientation_classify,
            "use_doc_unwarping": self.config.use_doc_unwarping,
            "use_layout_detection": self.config.layout_detection,
        }

    def _build_document(
        self,
        pdf_path: str,
        raw_result: Any,
        *,
        page_offset: int = 0,
        parse_source_path: Optional[Path] = None,
    ) -> SemanticDocument:
        parse_source = parse_source_path or Path(pdf_path)
        page_image_paths, ocr_page_image_paths = self._resolve_page_image_paths(str(parse_source), raw_result)
        pages: List[SemanticPage] = []
        document_section: List[str] = []

        for page_index, page_payload in enumerate(self._iter_pages(raw_result)):
            page_image_path = page_image_paths.get(page_index + 1)
            ocr_page_image_path = ocr_page_image_paths.get(page_index + 1)
            page_width = float(self._get_value(page_payload, "width", default=0) or 0)
            page_height = float(self._get_value(page_payload, "height", default=0) or 0)
            layout_boxes = self._extract_layout_boxes(page_payload)
            table_regions = self._extract_table_regions(page_payload, layout_boxes)
            segments: List[SemanticSegment] = []
            absolute_page = page_index + page_offset

            for block_index, block_payload in enumerate(self._iter_blocks(page_payload), start=1):
                paddle_label = self._extract_label(block_payload)
                text_content = self._extract_text(block_payload)
                if not text_content:
                    continue

                bbox = self._extract_bbox(block_payload)
                matched_layout_box = self._match_layout_box(bbox, layout_boxes)
                polygon_points = self._extract_polygon_points(block_payload, matched_layout_box)
                heading_level = self._extract_heading_level(paddle_label, text_content)
                included_for_extraction, exclusion_reason = self._classify_segment(
                    label=paddle_label,
                    text=text_content,
                    page=absolute_page,
                    heading_level=heading_level,
                )

                section_path = list(document_section)
                if included_for_extraction and heading_level is not None:
                    document_section = self._update_section_path(document_section, text_content, heading_level)
                    section_path = list(document_section)

                source_block_id = self._extract_source_block_id(block_payload, absolute_page, block_index)
                segment_id = self._build_segment_id(absolute_page, block_index, self._extract_group_id(block_payload))
                segment_kind = self._infer_segment_kind(paddle_label)
                block_order = self._extract_block_order(block_payload, matched_layout_box, block_index)
                text_markdown = self._render_markdown(
                    label=paddle_label,
                    text=text_content,
                    heading_level=heading_level,
                )

                segments.append(
                    SemanticSegment(
                        segment_id=segment_id,
                        segment_kind=segment_kind,
                        paddle_label=paddle_label,
                        page=absolute_page,
                        text_content=text_content,
                        text_markdown=text_markdown,
                        bbox=bbox,
                        bbox_norm=self._normalize_bbox(bbox, page_width, page_height),
                        polygon_points=polygon_points,
                        source_block_ids=[source_block_id],
                        group_id=self._extract_group_id(block_payload),
                        block_order=block_order,
                        section_path=section_path,
                        heading_level=heading_level,
                        included_for_extraction=included_for_extraction,
                        exclusion_reason=exclusion_reason,
                        confidence=self._extract_confidence(block_payload),
                        metadata=self._build_segment_metadata(
                            block_payload=block_payload,
                            page_width=page_width,
                            page_height=page_height,
                            matched_layout_box=matched_layout_box,
                            table_regions=table_regions,
                            page_image_path=page_image_path,
                            ocr_page_image_path=ocr_page_image_path,
                        ),
                    )
                )

            pages.append(
                SemanticPage(
                    page=absolute_page,
                    width=page_width,
                    height=page_height,
                    segments=segments,
                    metadata=self._build_page_metadata(
                        page_payload=page_payload,
                        layout_boxes=layout_boxes,
                        table_regions=table_regions,
                        page_image_path=page_image_path,
                        ocr_page_image_path=ocr_page_image_path,
                    ),
                )
            )

        document = SemanticDocument(
            source_document=str(Path(pdf_path)),
            pages=pages,
            metadata={
                "parser_backend": "paddleocr_vl",
                "page_offset": page_offset,
                "parse_source_path": str(parse_source),
                "ignored_paddle_labels": list(self.config.ignored_paddle_labels),
                "include_note_like_segments": self.config.include_note_like_segments,
                "exclude_front_matter": self.config.exclude_front_matter,
                "toc_section_pruning_mode": self.config.toc_section_pruning_mode,
                "toc_excluded_section_titles": list(self.config.toc_excluded_section_titles),
            },
        )
        self._apply_toc_section_pruning(document)
        return document

    def _prepare_pdf_input(self, pdf_path: Path) -> tuple[Path, int]:
        if pdf_path.suffix.lower() != ".pdf":
            return pdf_path, 0

        if self.config.page_start is None and self.config.page_end is None and self.config.max_pages is None:
            return pdf_path, 0

        start_index = max((self.config.page_start or 1) - 1, 0)
        end_index = self.config.page_end - 1 if self.config.page_end is not None else None
        if end_index is None and self.config.max_pages is not None:
            end_index = start_index + max(self.config.max_pages - 1, 0)

        sliced_path = self._slice_pdf_pages(pdf_path, start_index, end_index)
        return sliced_path, start_index

    def _slice_pdf_pages(self, pdf_path: Path, start_index: int, end_index: Optional[int]) -> Path:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required to slice PDFs when parser.page_start/page_end/max_pages is configured."
            ) from exc

        if self._sliced_pdfs_dir is None:
            self._sliced_pdfs_dir = tempfile.TemporaryDirectory(prefix="requirement-extractor-slices-")

        source_id = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:12]
        output_path = Path(self._sliced_pdfs_dir.name) / (
            f"{pdf_path.stem}-{source_id}-p{start_index + 1:03d}"
            f"-p{(end_index + 1) if end_index is not None else 'end'}.pdf"
        )

        src = fitz.open(str(pdf_path))
        try:
            page_count = len(src)
            if page_count == 0:
                raise RuntimeError(f"PDF '{pdf_path}' has no pages to parse.")
            effective_end = page_count - 1 if end_index is None else min(end_index, page_count - 1)
            if start_index >= page_count:
                raise RuntimeError(
                    f"Configured parser.page_start={start_index + 1} is beyond the document page count ({page_count})."
                )
            if effective_end < start_index:
                raise RuntimeError(
                    f"Configured PDF page range is invalid: start={start_index + 1}, end={effective_end + 1}."
                )

            sliced = fitz.open()
            try:
                sliced.insert_pdf(src, from_page=start_index, to_page=effective_end)
                sliced.save(str(output_path))
            finally:
                sliced.close()
        finally:
            src.close()

        return output_path

    def _resolve_page_image_paths(
        self,
        pdf_path: str,
        raw_result: Any,
    ) -> tuple[Dict[int, Optional[str]], Dict[int, Optional[str]]]:
        source_path = Path(pdf_path)
        suffix = source_path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            return {1: str(source_path)}, {}
        if suffix != ".pdf":
            return {}, {}

        page_count = len(list(self._iter_pages(raw_result)))
        page_image_paths = self._rasterize_pdf_pages(source_path, page_count)
        ocr_page_image_paths = self._persist_native_page_images(raw_result)
        if not page_image_paths and ocr_page_image_paths:
            page_image_paths = dict(ocr_page_image_paths)
        return page_image_paths, ocr_page_image_paths

    def _persist_native_page_images(self, raw_result: Any) -> Dict[int, Optional[str]]:
        output_dir = self._resolve_native_pages_output_dir()
        if output_dir is None:
            return {}

        page_results = list(self._iter_pages(raw_result))
        if not page_results:
            return {}

        output_dir.mkdir(parents=True, exist_ok=True)
        for existing_page in output_dir.glob("page-*.png"):
            existing_page.unlink()

        page_image_paths: Dict[int, Optional[str]] = {}
        for page_index, page_result in enumerate(page_results, start=1):
            save_to_img = getattr(page_result, "save_to_img", None)
            if not callable(save_to_img):
                return {}

            scratch_dir = output_dir / f".page-{page_index:03d}"
            if scratch_dir.exists():
                shutil.rmtree(scratch_dir)
            scratch_dir.mkdir(parents=True, exist_ok=True)

            try:
                save_to_img(save_path=str(scratch_dir))
                persisted_image = self._select_saved_page_image(scratch_dir)
                if persisted_image is None:
                    return {}
                target_path = output_dir / f"page-{page_index:03d}.png"
                shutil.copyfile(persisted_image, target_path)
                page_image_paths[page_index] = str(target_path)
            except Exception as exc:
                logger.warning("Failed to persist Paddle native OCR image for page %s: %s", page_index, exc)
                return {}
            finally:
                with suppress(Exception):
                    shutil.rmtree(scratch_dir)

        return page_image_paths

    def _rasterize_pdf_pages(self, pdf_path: Path, page_count: int) -> Dict[int, Optional[str]]:
        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF is not available; skipping PDF page rasterization for %s", pdf_path)
            return {}

        output_dir = self._resolve_rasterized_pages_output_dir(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for existing_page in output_dir.glob("page-*.png"):
            existing_page.unlink()

        page_image_paths: Dict[int, Optional[str]] = {}
        try:
            pdf_document = fitz.open(str(pdf_path))
            render_count = min(page_count, len(pdf_document))
            matrix = fitz.Matrix(2.0, 2.0)
            for page_index in range(render_count):
                pixmap = pdf_document.load_page(page_index).get_pixmap(matrix=matrix, alpha=False)
                output_path = output_dir / f"page-{page_index + 1:03d}.png"
                pixmap.save(str(output_path))
                page_image_paths[page_index + 1] = str(output_path)
            pdf_document.close()
        except Exception as exc:
            logger.warning("Failed to rasterize PDF pages for %s: %s", pdf_path, exc)
            return {}

        return page_image_paths

    def _persist_parse_input(self, parse_input_path: Path) -> Path:
        if self.output_directory is None or parse_input_path.suffix.lower() != ".pdf":
            return parse_input_path

        self.output_directory.mkdir(parents=True, exist_ok=True)
        persisted_path = self.output_directory / "ocr-input.pdf"
        if parse_input_path.resolve() == persisted_path.resolve():
            return parse_input_path

        shutil.copyfile(parse_input_path, persisted_path)
        return persisted_path

    def _resolve_rasterized_pages_output_dir(self, pdf_path: Path) -> Path:
        if self.output_directory is not None:
            return self.output_directory / "page-images"

        if self._rasterized_pages_dir is None:
            self._rasterized_pages_dir = tempfile.TemporaryDirectory(prefix="requirement-extractor-pages-")

        doc_id = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:12]
        return Path(self._rasterized_pages_dir.name) / f"{pdf_path.stem}-{doc_id}"

    def _resolve_native_pages_output_dir(self) -> Optional[Path]:
        if self.output_directory is None:
            return None
        return self.output_directory / "ocr-pages"

    def _select_saved_page_image(self, scratch_dir: Path) -> Optional[Path]:
        candidates = sorted(
            path
            for path in scratch_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not candidates:
            return None
        return candidates[0]

    def _apply_toc_section_pruning(self, document: SemanticDocument) -> None:
        report = self._resolve_toc_pruning_report(document)
        page_matches = self._build_toc_excluded_page_matches(report)

        pruning_applied = False
        excluded_segment_count = 0
        skipped_reason: Optional[str] = None

        if self.config.toc_section_pruning_mode == "off":
            skipped_reason = "mode_off"
        elif not report["toc_reliable"]:
            skipped_reason = report.get("toc_reliability_reason") or "toc_unreliable"
        elif self.config.toc_section_pruning_mode == "audit":
            skipped_reason = "audit_only"
        elif self.config.toc_section_pruning_mode == "enforce":
            pruning_applied = True
            for page in document.pages:
                page_match = page_matches.get(page.page_number)
                if page_match is None:
                    continue
                page.metadata["toc_excluded_section"] = {
                    "title": page_match["title"],
                    "normalized_title": page_match["normalized_title"],
                    "matched_excluded_rule": page_match["matched_excluded_rule"],
                    "range_start_page": page_match["range_start_page"],
                    "range_end_page": page_match["range_end_page"],
                }
                for segment in page.segments:
                    if not segment.included_for_extraction:
                        continue
                    segment.included_for_extraction = False
                    segment.exclusion_reason = f"excluded_toc_section:{page_match['normalized_title']}"
                    excluded_segment_count += 1
        else:
            skipped_reason = f"unsupported_mode:{self.config.toc_section_pruning_mode}"

        report["pruning_applied"] = pruning_applied
        report["pruning_skipped_reason"] = skipped_reason
        report["excluded_segment_count"] = excluded_segment_count
        report["excluded_page_count"] = len(page_matches) if pruning_applied else 0
        document.metadata["toc_section_pruning"] = report
        report_path = self._persist_toc_pruning_report(report)
        if report_path is not None:
            document.metadata["toc_pruning_report_path"] = str(report_path)

    def build_toc_pruning_plan(self, pdf_path: str) -> Dict[str, Any]:
        if self._toc_pruning_plan is not None:
            return copy.deepcopy(self._toc_pruning_plan)

        full_page_count = self._count_pdf_pages(pdf_path)
        scan_config = copy.deepcopy(self.config)
        scan_config.toc_section_pruning_mode = "audit"
        scan_config.page_start = max(scan_config.page_start or 1, 1)
        scan_config.page_end = None
        scan_config.max_pages = TOC_SCAN_PAGE_LIMIT
        scan_config.batch_page_count = None
        scan_parser = PaddleOCRVLParser(scan_config, output_directory=None)
        document = scan_parser.parse_pdf(pdf_path)
        report_copy = scan_parser._build_toc_pruning_report(
            document,
            document_page_count=full_page_count,
            mode_override=self.config.toc_section_pruning_mode,
        )
        report_copy["plan_source"] = "toc_preflight_scan"
        report_copy["plan_pdf_path"] = str(Path(pdf_path))
        return report_copy

    def _resolve_toc_pruning_report(self, document: SemanticDocument) -> Dict[str, Any]:
        if self._toc_pruning_plan is None:
            return self._build_toc_pruning_report(document)

        report = copy.deepcopy(self._toc_pruning_plan)
        page_numbers = [page.page_number for page in document.pages]
        window_start = min(page_numbers, default=None)
        window_end = max(page_numbers, default=None)

        report["plan_source"] = report.get("plan_source") or "precomputed"
        report["source_document"] = document.source_document
        report["document_page_window"] = {
            "start_page": window_start,
            "end_page": window_end,
        }
        report["window_excluded_ranges"] = [
            excluded
            for excluded in report.get("excluded_ranges", [])
            if window_start is not None
            and window_end is not None
            and not (
                int(excluded["end_page"]) < window_start
                or int(excluded["start_page"]) > window_end
            )
        ]
        return report

    def _build_toc_pruning_report(
        self,
        document: SemanticDocument,
        *,
        document_page_count: Optional[int] = None,
        mode_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        page_count = document_page_count or max((page.page_number for page in document.pages), default=0)
        toc_entries, toc_pages = self._extract_toc_entries(document)
        toc_reliable, toc_reliability_reason = self._assess_toc_reliability(
            toc_entries=toc_entries,
            toc_pages=toc_pages,
            page_count=page_count,
        )
        ranged_entries = self._build_toc_ranges(toc_entries, page_count)

        excluded_ranges = []
        for entry in ranged_entries:
            matched_rule = self._match_toc_excluded_section_title(entry["normalized_title"])
            entry["matched_excluded_rule"] = matched_rule
            entry["would_exclude"] = bool(matched_rule)
            if entry["would_exclude"]:
                excluded_ranges.append(
                    {
                        "title": entry["title"],
                        "normalized_title": entry["normalized_title"],
                        "source_toc_page": entry["source_toc_page"],
                        "start_page": entry["range_start_page"],
                        "end_page": entry["range_end_page"],
                        "matched_excluded_rule": matched_rule,
                    }
                )

        return {
            "mode": mode_override or self.config.toc_section_pruning_mode,
            "source_document": document.source_document,
            "toc_scan_page_limit": min(len(document.pages), TOC_SCAN_PAGE_LIMIT),
            "toc_pages": toc_pages,
            "toc_reliable": toc_reliable,
            "toc_reliability_reason": toc_reliability_reason,
            "toc_entry_count": len(ranged_entries),
            "toc_entries": ranged_entries,
            "excluded_ranges": excluded_ranges,
        }

    def _count_pdf_pages(self, pdf_path: str) -> Optional[int]:
        pdf_file = Path(pdf_path)
        if pdf_file.suffix.lower() != ".pdf":
            return None

        try:
            import fitz
        except ImportError:
            logger.warning("PyMuPDF is not available; falling back to scanned TOC page count for %s", pdf_path)
            return None

        try:
            with fitz.open(str(pdf_file)) as pdf_document:
                return len(pdf_document)
        except Exception as exc:
            logger.warning("Failed to count PDF pages for TOC pruning plan %s: %s", pdf_path, exc)
            return None

    def _extract_toc_entries(self, document: SemanticDocument) -> tuple[List[Dict[str, Any]], List[int]]:
        toc_entries: List[Dict[str, Any]] = []
        toc_pages: List[int] = []
        seen_entries: set[tuple[str, int, int]] = set()
        toc_started = False

        for page in document.pages[:TOC_SCAN_PAGE_LIMIT]:
            lines = self._collect_page_lines(page)
            page_entries = []
            for line in lines:
                entry = self._parse_toc_entry_line(line)
                if entry is None:
                    continue
                entry["source_toc_page"] = page.page_number
                page_entries.append(entry)

            has_toc_heading = any(self._looks_like_toc_heading(segment.text_content) for segment in page.segments)
            is_toc_page = has_toc_heading or len(page_entries) >= TOC_MIN_ENTRY_COUNT or (toc_started and len(page_entries) >= 2)
            if not is_toc_page:
                if toc_started:
                    break
                continue

            toc_started = True
            toc_pages.append(page.page_number)
            for entry in page_entries:
                key = (entry["normalized_title"], entry["start_page"], entry["section_level"])
                if key in seen_entries:
                    continue
                seen_entries.add(key)
                toc_entries.append(entry)

        return toc_entries, toc_pages

    def _collect_page_lines(self, page: SemanticPage) -> List[str]:
        lines: List[str] = []
        ordered_segments = sorted(
            page.segments,
            key=lambda segment: (
                segment.block_order if segment.block_order is not None else 10_000,
                segment.segment_id,
            ),
        )
        for segment in ordered_segments:
            for raw_line in segment.text_content.splitlines():
                stripped = raw_line.strip()
                if stripped:
                    lines.append(stripped)
        return lines

    def _parse_toc_entry_line(self, line: str) -> Optional[Dict[str, Any]]:
        for pattern in TOC_ENTRY_PATTERNS:
            match = pattern.match(line)
            if match is None:
                continue

            title = match.group("title").strip(" \t.-")
            if not title or re.fullmatch(r"[\d.\s]+", title):
                return None

            normalized_title = self._normalize_section_title(title)
            if not normalized_title or re.fullmatch(r"[\d\s]+", normalized_title):
                return None
            if self._is_non_section_toc_entry(title, normalized_title):
                return None

            return {
                "title": title,
                "normalized_title": normalized_title,
                "start_page": int(match.group("page")),
                "section_level": self._infer_toc_section_level(title),
            }
        return None

    def _is_non_section_toc_entry(self, title: str, normalized_title: str) -> bool:
        if normalized_title in {"contents", "table of contents"}:
            return False

        return bool(
            re.match(r"^\s*figures?\b", title, flags=re.IGNORECASE)
            or re.match(r"^\s*tables?\b", title, flags=re.IGNORECASE)
        )

    def _looks_like_toc_heading(self, text: str) -> bool:
        normalized = self._normalize_section_title(text)
        return normalized in {"contents", "table of contents"}

    def _infer_toc_section_level(self, title: str) -> int:
        numbered_heading = re.match(r"^\s*(\d+(?:\.\d+)*)\b", title)
        if numbered_heading:
            return len(numbered_heading.group(1).split("."))
        if re.match(r"^\s*(annex|appendix)\b", title, flags=re.IGNORECASE):
            return 1
        return 1

    def _normalize_section_title(self, title: str) -> str:
        normalized = title.strip()
        normalized = re.sub(r"^\s*[#•\-\u2022]+\s*", "", normalized)
        normalized = re.sub(r"^\s*(\d+(?:\.\d+)*)\b[\s.:_-]*", "", normalized)
        normalized = re.sub(r"[^\w\s]+", " ", normalized.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _assess_toc_reliability(
        self,
        *,
        toc_entries: List[Dict[str, Any]],
        toc_pages: List[int],
        page_count: int,
    ) -> tuple[bool, str]:
        if not toc_pages:
            return False, "no_toc_pages_detected"
        if len(toc_entries) < TOC_MIN_ENTRY_COUNT:
            return False, "insufficient_toc_entries"
        if page_count <= 0:
            return False, "invalid_page_count"

        previous_page = 0
        monotonic_violations = 0
        for entry in toc_entries:
            start_page = entry["start_page"]
            if start_page < 1 or start_page > page_count:
                return False, "toc_page_out_of_range"
            if start_page < previous_page:
                monotonic_violations += 1
            previous_page = max(previous_page, start_page)
            if not entry["normalized_title"]:
                return False, "empty_toc_title"

        allowed_violations = max(1, len(toc_entries) // 5)
        if monotonic_violations > allowed_violations:
            return False, "toc_pages_not_monotonic"

        return True, "ok"

    def _build_toc_ranges(self, toc_entries: List[Dict[str, Any]], page_count: int) -> List[Dict[str, Any]]:
        ranged_entries: List[Dict[str, Any]] = []
        for index, entry in enumerate(toc_entries):
            range_end_page = page_count
            for later in toc_entries[index + 1 :]:
                if later["section_level"] <= entry["section_level"] and later["start_page"] > entry["start_page"]:
                    range_end_page = later["start_page"] - 1
                    break
            annotated = dict(entry)
            annotated["range_start_page"] = entry["start_page"]
            annotated["range_end_page"] = max(entry["start_page"], range_end_page)
            ranged_entries.append(annotated)
        return ranged_entries

    def _match_toc_excluded_section_title(self, normalized_title: str) -> Optional[str]:
        matched_rule: Optional[str] = None
        matched_rule_length = -1
        for raw_rule in self.config.toc_excluded_section_titles:
            normalized_rule = self._normalize_section_title(raw_rule)
            if normalized_rule and normalized_rule in normalized_title and len(normalized_rule) > matched_rule_length:
                matched_rule = raw_rule
                matched_rule_length = len(normalized_rule)
        return matched_rule

    def _build_toc_excluded_page_matches(self, report: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        page_matches: Dict[int, Dict[str, Any]] = {}
        if not report.get("toc_reliable"):
            return page_matches
        for entry in report.get("toc_entries", []):
            if not entry.get("would_exclude"):
                continue
            start_page = int(entry["range_start_page"])
            end_page = int(entry["range_end_page"])
            for page_number in range(start_page, end_page + 1):
                page_matches.setdefault(page_number, entry)
        return page_matches

    def _persist_toc_pruning_report(self, report: Dict[str, Any]) -> Optional[Path]:
        if self.output_directory is None:
            return None
        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_path = self.output_directory / self.config.toc_pruning_report_filename
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return output_path

    def _iter_pages(self, raw_result: Any) -> Iterable[Any]:
        if raw_result is None:
            return []
        if isinstance(raw_result, list):
            return [item for item in raw_result if self._is_page_payload(item)]
        if self._is_page_payload(raw_result):
            pages = self._get_value(raw_result, "pages")
            if isinstance(pages, list):
                return [item for item in pages if self._is_page_payload(item)]
            return [raw_result]
        if hasattr(raw_result, "to_dict"):
            return self._iter_pages(raw_result.to_dict())
        if hasattr(raw_result, "json"):
            return self._iter_pages(raw_result.json)
        return []

    def _iter_blocks(self, page_payload: Any) -> Iterable[Any]:
        for key in ("parsing_res_list", "blocks", "layout", "texts", "paragraphs"):
            blocks = self._get_value(page_payload, key)
            if isinstance(blocks, list):
                return [item for item in blocks if self._is_block_payload(item)]
        return []

    def _extract_text(self, block_payload: Any) -> str:
        for key in ("block_content", "content", "text", "markdown"):
            value = self._get_value(block_payload, key)
            if isinstance(value, str):
                return value.strip()
        return ""

    def _extract_label(self, block_payload: Any) -> str:
        for key in ("block_label", "label", "type", "block_type"):
            value = self._get_value(block_payload, key)
            if value is not None:
                return str(value)
        return "text"

    def _extract_bbox(self, block_payload: Any) -> Optional[tuple[float, float, float, float]]:
        bbox = (
            self._get_value(block_payload, "block_bbox")
            or self._get_value(block_payload, "bbox")
            or self._get_value(block_payload, "box")
        )
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return tuple(float(value) for value in bbox)
        return None

    def _extract_confidence(self, block_payload: Any) -> Optional[float]:
        for key in ("score", "confidence", "block_score"):
            confidence = self._get_value(block_payload, key)
            if confidence is not None:
                return float(confidence)
        return None

    def _extract_group_id(self, block_payload: Any) -> Optional[int]:
        value = self._get_value(block_payload, "group_id")
        if value is None:
            return None
        return int(value)

    def _extract_block_order(
        self,
        block_payload: Any,
        matched_layout_box: Optional[Dict[str, Any]],
        fallback: int,
    ) -> Optional[int]:
        value = self._get_value(block_payload, "block_order")
        if value is None and matched_layout_box is not None:
            value = matched_layout_box.get("order")
        if value is None:
            return fallback
        return int(value)

    def _extract_heading_level(self, paddle_label: str, text: str) -> Optional[int]:
        normalized_label = (paddle_label or "").lower()
        stripped = text.strip()
        if not stripped:
            return None
        if normalized_label == "doc_title":
            return 1
        if normalized_label not in {"paragraph_title", "title"}:
            return None

        numbered_heading = re.match(r"^\s*(\d+(?:\.\d+)*)\b", stripped)
        if numbered_heading:
            depth = len(numbered_heading.group(1).split("."))
            return min(depth + 1, 6)
        if self._looks_like_heading_text(stripped):
            return 2
        return None

    def _infer_segment_kind(self, paddle_label: str) -> str:
        normalized_label = paddle_label.lower()
        if normalized_label == "table":
            return "table"
        if "image" in normalized_label or normalized_label in {"figure", "chart"}:
            return "figure"
        return "text"

    def _render_markdown(self, label: str, text: str, heading_level: Optional[int]) -> str:
        if heading_level is not None:
            return f"{'#' * heading_level} {text.strip()}"
        if label.lower() == "table":
            return text.strip()
        return text.strip()

    def _classify_segment(
        self,
        label: str,
        text: str,
        page: int,
        heading_level: Optional[int],
    ) -> tuple[bool, Optional[str]]:
        normalized_label = label.lower()
        stripped = text.strip()
        if not stripped:
            return False, "empty"

        if self.config.exclude_front_matter and self._looks_like_front_matter(stripped, page):
            return False, "front_matter"

        if normalized_label in {value.lower() for value in self.config.ignored_paddle_labels}:
            if self._should_include_note_like_segment(normalized_label, stripped):
                return True, None
            return False, f"ignored_label:{normalized_label}"

        if self._infer_segment_kind(normalized_label) == "figure":
            return False, "figure_omitted"

        if self._is_note_like(normalized_label, stripped):
            if self.config.include_note_like_segments == "never":
                return False, "note_like_excluded"
            if self.config.include_note_like_segments == "always":
                return True, None
            if self._looks_like_requirement_note(stripped):
                return True, None
            return False, "note_like_not_relevant"

        if heading_level is not None and self._looks_like_front_matter_heading(stripped, page):
            return False, "front_matter_heading"

        return True, None

    def _should_include_note_like_segment(self, label: str, text: str) -> bool:
        if self.config.include_note_like_segments == "never":
            return False
        if self.config.include_note_like_segments == "always":
            return True
        return self._is_note_like(label, text) and self._looks_like_requirement_note(text)

    def _is_note_like(self, label: str, text: str) -> bool:
        normalized_label = label.lower()
        lower_text = text.lower().lstrip()
        return normalized_label in {"footnote", "vision_footnote", "aside_text"} or lower_text.startswith(NOTE_PREFIXES)

    def _looks_like_requirement_note(self, text: str) -> bool:
        lowered = text.lower()
        return (
            lowered.startswith(NOTE_PREFIXES)
            or " shall " in f" {lowered} "
            or " must " in f" {lowered} "
            or " required " in f" {lowered} "
        )

    def _looks_like_front_matter(self, text: str, page: int) -> bool:
        lowered = text.lower()
        if "table of contents" in lowered or lowered.strip() == "contents":
            return True
        if page > 1:
            return False
        if re.search(r"\.{2,}\s*\d+\s*$", text, flags=re.MULTILINE):
            return True
        if re.search(r"^\s*\d+(?:\.\d+)*\s+.+\s+\d+\s*$", text, flags=re.MULTILINE):
            return True
        return False

    def _looks_like_front_matter_heading(self, text: str, page: int) -> bool:
        return self._looks_like_front_matter(text, page)

    def _looks_like_heading_text(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.isupper() and len(stripped.split()) <= 2:
            return False
        if "/" in stripped:
            return True
        return len(stripped.split()) >= 2

    def _update_section_path(self, current_section: List[str], heading_text: str, heading_level: int) -> List[str]:
        current = list(current_section)
        target_depth = max(heading_level - 1, 0)
        if target_depth == 0:
            return [heading_text]
        current = current[:target_depth - 1]
        current.append(heading_text)
        return current

    def _extract_source_block_id(self, block_payload: Any, page: int, index: int) -> str:
        for key in ("block_id", "id"):
            value = self._get_value(block_payload, key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, int):
                return str(value)
        return f"p{page + 1:03d}-b{index:03d}"

    def _build_segment_id(self, page: int, index: int, group_id: Optional[int]) -> str:
        if group_id is not None:
            return f"seg-p{page:03d}-g{group_id:03d}-i{index:03d}"
        return f"seg-p{page:03d}-i{index:03d}"

    def _normalize_bbox(
        self,
        bbox: Optional[tuple[float, float, float, float]],
        width: float,
        height: float,
    ) -> Optional[tuple[float, float, float, float]]:
        if bbox is None or width <= 0 or height <= 0:
            return None
        x1, y1, x2, y2 = bbox
        return (x1 / width, y1 / height, x2 / width, y2 / height)

    def _build_page_metadata(
        self,
        page_payload: Any,
        layout_boxes: List[Dict[str, Any]],
        table_regions: List[Dict[str, Any]],
        page_image_path: Optional[str],
        ocr_page_image_path: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "parser_backend": "paddleocr_vl",
            "page_index": self._get_value(page_payload, "page_index"),
            "page_count": self._get_value(page_payload, "page_count"),
            "layout_boxes": layout_boxes,
            "table_regions": table_regions,
            "page_image_path": page_image_path,
            "ocr_page_image_path": ocr_page_image_path,
            "model_settings": self._sanitize_value(self._get_value(page_payload, "model_settings")),
        }

    def _build_segment_metadata(
        self,
        block_payload: Any,
        page_width: float,
        page_height: float,
        matched_layout_box: Optional[Dict[str, Any]],
        table_regions: List[Dict[str, Any]],
        page_image_path: Optional[str],
        ocr_page_image_path: Optional[str],
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "parser_backend": "paddleocr_vl",
            "page_width": page_width,
            "page_height": page_height,
            "page_image_path": page_image_path,
            "ocr_page_image_path": ocr_page_image_path,
        }
        if matched_layout_box is not None:
            metadata.update(
                {
                    "layout_cls_id": matched_layout_box.get("cls_id"),
                    "layout_order": matched_layout_box.get("order"),
                    "layout_score": matched_layout_box.get("score"),
                    "layout_label": matched_layout_box.get("label"),
                }
            )
        bbox = self._extract_bbox(block_payload)
        if bbox is not None:
            matching_tables = [
                region["bbox"]
                for region in table_regions
                if self._bboxes_overlap(bbox, tuple(region["bbox"]))
            ]
            if matching_tables:
                metadata["table_bboxes"] = matching_tables
        return metadata

    def _extract_layout_boxes(self, page_payload: Any) -> List[Dict[str, Any]]:
        layout_res = self._get_value(page_payload, "layout_det_res")
        boxes = self._get_value(layout_res, "boxes", default=[])
        if not isinstance(boxes, list):
            return []
        serialized_boxes: List[Dict[str, Any]] = []
        for box in boxes:
            coordinate = self._get_value(box, "coordinate")
            if not isinstance(coordinate, Sequence) or len(coordinate) != 4:
                continue
            serialized_boxes.append(
                {
                    "cls_id": self._get_value(box, "cls_id"),
                    "label": self._get_value(box, "label"),
                    "order": self._get_value(box, "order"),
                    "score": self._coerce_float(self._get_value(box, "score")),
                    "bbox": [float(value) for value in coordinate],
                    "polygon_points": self._sanitize_value(self._get_value(box, "polygon_points")),
                }
            )
        return serialized_boxes

    def _extract_table_regions(
        self,
        page_payload: Any,
        layout_boxes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        table_regions: List[Dict[str, Any]] = [
            {
                "bbox": list(box["bbox"]),
                "label": box.get("label"),
                "score": box.get("score"),
            }
            for box in layout_boxes
            if box.get("label") == "table"
        ]
        table_res_list = self._get_value(page_payload, "table_res_list", default=[])
        if isinstance(table_res_list, list):
            for table_res in table_res_list:
                bbox = self._extract_bbox(table_res)
                if bbox is None:
                    continue
                serialized_bbox = [float(value) for value in bbox]
                if any(region["bbox"] == serialized_bbox for region in table_regions):
                    continue
                table_regions.append(
                    {
                        "bbox": serialized_bbox,
                        "label": self._get_value(table_res, "label") or "table",
                        "score": self._coerce_float(
                            self._get_value(table_res, "score") or self._get_value(table_res, "confidence")
                        ),
                    }
                )
        return table_regions

    def _match_layout_box(
        self,
        bbox: Optional[tuple[float, float, float, float]],
        layout_boxes: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if bbox is None:
            return None
        for box in layout_boxes:
            layout_bbox = tuple(box["bbox"])
            if self._bboxes_close(bbox, layout_bbox):
                return box
        return None

    def _extract_polygon_points(
        self,
        block_payload: Any,
        matched_layout_box: Optional[Dict[str, Any]],
    ) -> List[List[float]]:
        for key in ("block_polygon_points", "polygon_points"):
            value = self._get_value(block_payload, key)
            if isinstance(value, list):
                return self._sanitize_value(value)
        if matched_layout_box is not None and isinstance(matched_layout_box.get("polygon_points"), list):
            return self._sanitize_value(matched_layout_box["polygon_points"])
        return []

    def _bboxes_close(self, left: tuple[float, float, float, float], right: tuple[float, float, float, float], tolerance: float = 2.0) -> bool:
        return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))

    def _bboxes_overlap(self, left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
        left_x1, left_y1, left_x2, left_y2 = left
        right_x1, right_y1, right_x2, right_y2 = right
        overlap_x1 = max(left_x1, right_x1)
        overlap_y1 = max(left_y1, right_y1)
        overlap_x2 = min(left_x2, right_x2)
        overlap_y2 = min(left_y2, right_y2)
        return overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2

    def _sanitize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): self._sanitize_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_value(item) for item in value]
        return str(value)

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        return float(value)

    def _get_value(self, payload: Any, key: str, default: Any = None) -> Any:
        if isinstance(payload, dict):
            return payload.get(key, default)
        if hasattr(payload, "get"):
            try:
                return payload.get(key, default)
            except TypeError:
                pass
        if hasattr(payload, key):
            return getattr(payload, key)
        return default

    def _is_page_payload(self, payload: Any) -> bool:
        return isinstance(payload, dict) or hasattr(payload, "keys") or hasattr(payload, "get")

    def _is_block_payload(self, payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            or hasattr(payload, "keys")
            or hasattr(payload, "get")
            or hasattr(payload, "content")
            or hasattr(payload, "text")
        )
