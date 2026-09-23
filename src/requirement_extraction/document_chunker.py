from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config.schema import ChunkingConfig
from src.pdf_processing.models import SemanticDocument, SemanticPage, SemanticSegment


@dataclass
class AnchoredMarkdownChunk:
    chunk_id: str
    markdown_text: str
    source_document: str
    source_page_start: Optional[int]
    source_page_end: Optional[int]
    segment_ids: List[str] = field(default_factory=list)
    source_block_ids: List[str] = field(default_factory=list)
    source_section: Optional[str] = None
    source_text_excerpt: Optional[str] = None
    source_bbox_list: List[List[float]] = field(default_factory=list)
    source_regions: List[dict] = field(default_factory=list)
    source_segments: List[SemanticSegment] = field(default_factory=list, repr=False)

    @property
    def text(self) -> str:
        return self.markdown_text


class SemanticDocumentChunker:
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def build_extraction_segments(self, document: SemanticDocument) -> List[SemanticSegment]:
        return [
            segment
            for page in document.pages
            for segment in page.segments
            if segment.included_for_extraction and segment.text_markdown.strip()
        ]

    def chunk_document(self, document: SemanticDocument) -> List[AnchoredMarkdownChunk]:
        max_chars = self.config.max_chunk_chars or 4000
        segments = self.build_extraction_segments(document)
        semantic_units = self._build_semantic_units(segments)
        page_lookup = {page.page: page for page in document.pages}
        chunks: List[AnchoredMarkdownChunk] = []
        current_segments: List[SemanticSegment] = []
        current_chars = 0

        for unit in semantic_units:
            unit_chars = self._unit_char_count(unit)
            if current_segments and current_chars + unit_chars > max_chars:
                chunks.append(self._build_chunk(document.source_document, chunks, current_segments, page_lookup))
                current_segments = []
                current_chars = 0
            current_segments.extend(unit)
            current_chars += unit_chars

        if current_segments:
            chunks.append(self._build_chunk(document.source_document, chunks, current_segments, page_lookup))

        return chunks

    def render_anchored_markdown(self, segments: List[SemanticSegment]) -> str:
        return "\n\n".join(self._render_segment(segment) for segment in segments if segment.text_markdown.strip()).strip()

    def _build_semantic_units(self, segments: List[SemanticSegment]) -> List[List[SemanticSegment]]:
        units: List[List[SemanticSegment]] = []
        current_unit: List[SemanticSegment] = []

        for segment in segments:
            if segment.heading_level is not None and current_unit:
                units.append(current_unit)
                current_unit = [segment]
                continue
            current_unit.append(segment)

        if current_unit:
            units.append(current_unit)

        return units

    def _unit_char_count(self, segments: List[SemanticSegment]) -> int:
        return sum(len(self._render_segment(segment)) for segment in segments)

    def _render_segment(self, segment: SemanticSegment) -> str:
        return f"<a id=\"{segment.segment_id}\"></a>\n{segment.text_markdown.strip()}".strip()

    def _build_chunk(
        self,
        source_document: str,
        existing_chunks: List[AnchoredMarkdownChunk],
        segments: List[SemanticSegment],
        page_lookup: dict[int, SemanticPage],
    ) -> AnchoredMarkdownChunk:
        markdown_text = self.render_anchored_markdown(segments)
        bbox_list = [list(segment.bbox) for segment in segments if segment.bbox is not None]
        section = self._infer_section(segments)
        return AnchoredMarkdownChunk(
            chunk_id=f"chunk-{len(existing_chunks) + 1}",
            markdown_text=markdown_text,
            source_document=source_document,
            source_page_start=min((segment.page_number for segment in segments), default=None),
            source_page_end=max((segment.page_number for segment in segments), default=None),
            segment_ids=[segment.segment_id for segment in segments],
            source_block_ids=[block_id for segment in segments for block_id in segment.source_block_ids],
            source_section=" / ".join(section) if section else None,
            source_text_excerpt=self._build_excerpt(segments),
            source_bbox_list=bbox_list,
            source_regions=self._build_source_regions(segments, page_lookup),
            source_segments=list(segments),
        )

    def _build_excerpt(self, segments: List[SemanticSegment]) -> str:
        excerpt = "\n\n".join(segment.text_content.strip() for segment in segments if segment.text_content.strip())
        return excerpt[:400]

    def _infer_section(self, segments: List[SemanticSegment]) -> Optional[List[str]]:
        for segment in reversed(segments):
            if segment.section_path:
                return segment.section_path
        return None

    def _build_source_regions(
        self,
        segments: List[SemanticSegment],
        page_lookup: dict[int, SemanticPage],
    ) -> List[dict]:
        regions: List[dict] = []
        for segment in segments:
            page = page_lookup.get(segment.page)
            page_width = page.width if page else 0.0
            page_height = page.height if page else 0.0
            region = {
                "segment_id": segment.segment_id,
                "segment_kind": segment.segment_kind,
                "block_id": segment.source_block_ids[0] if segment.source_block_ids else None,
                "block_type": segment.segment_kind,
                "paddle_label": segment.paddle_label,
                "page": segment.page,
                "page_number": segment.page_number,
                "page_width": page_width,
                "page_height": page_height,
                "page_image_path": segment.metadata.get("page_image_path"),
                "ocr_page_image_path": segment.metadata.get("ocr_page_image_path"),
                "bbox": list(segment.bbox) if segment.bbox is not None else None,
                "bbox_norm": list(segment.bbox_norm) if segment.bbox_norm is not None else None,
                "polygon_points": segment.polygon_points or None,
                "group_id": segment.group_id,
                "block_order": segment.block_order,
                "source_block_ids": list(segment.source_block_ids),
                "section_path": list(segment.section_path),
                "text_markdown": segment.text_markdown,
                "confidence": segment.confidence,
            }
            regions.append(region)
        return regions
