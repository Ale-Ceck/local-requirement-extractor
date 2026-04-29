from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pdf_processing.models import SemanticSegment
from src.requirement_extraction.document_chunker import AnchoredMarkdownChunk


CHUNK_CACHE_VERSION = 1


def write_chunk_cache(
    chunks: list[AnchoredMarkdownChunk],
    output_path: str | Path,
    *,
    source_document: str | None = None,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHUNK_CACHE_VERSION,
        "source_document": source_document or (chunks[0].source_document if chunks else None),
        "chunk_count": len(chunks),
        "chunks": [_serialize_chunk(chunk) for chunk in chunks],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def load_chunk_cache(cache_path: str | Path) -> list[AnchoredMarkdownChunk]:
    path = Path(cache_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != CHUNK_CACHE_VERSION:
        raise ValueError(
            f"Unsupported chunk cache version in {path}: {payload.get('version')!r}"
        )

    chunks_payload = payload.get("chunks", [])
    if not isinstance(chunks_payload, list):
        raise ValueError(f"Invalid chunk cache payload in {path}: 'chunks' must be a list.")

    return [_deserialize_chunk(chunk_payload) for chunk_payload in chunks_payload]


def _serialize_chunk(chunk: AnchoredMarkdownChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "markdown_text": chunk.markdown_text,
        "source_document": chunk.source_document,
        "source_page_start": chunk.source_page_start,
        "source_page_end": chunk.source_page_end,
        "segment_ids": list(chunk.segment_ids),
        "source_block_ids": list(chunk.source_block_ids),
        "source_section": chunk.source_section,
        "source_text_excerpt": chunk.source_text_excerpt,
        "source_bbox_list": [list(bbox) for bbox in chunk.source_bbox_list],
        "source_regions": list(chunk.source_regions),
        "source_segments": [_serialize_segment(segment) for segment in chunk.source_segments],
    }


def _deserialize_chunk(payload: dict[str, Any]) -> AnchoredMarkdownChunk:
    return AnchoredMarkdownChunk(
        chunk_id=str(payload.get("chunk_id") or ""),
        markdown_text=str(payload.get("markdown_text") or ""),
        source_document=str(payload.get("source_document") or ""),
        source_page_start=_optional_int(payload.get("source_page_start")),
        source_page_end=_optional_int(payload.get("source_page_end")),
        segment_ids=_string_list(payload.get("segment_ids")),
        source_block_ids=_string_list(payload.get("source_block_ids")),
        source_section=_optional_str(payload.get("source_section")),
        source_text_excerpt=_optional_str(payload.get("source_text_excerpt")),
        source_bbox_list=_bbox_list(payload.get("source_bbox_list")),
        source_regions=list(payload.get("source_regions") or []),
        source_segments=[
            _deserialize_segment(segment_payload)
            for segment_payload in payload.get("source_segments", [])
            if isinstance(segment_payload, dict)
        ],
    )


def _serialize_segment(segment: SemanticSegment) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "segment_kind": segment.segment_kind,
        "paddle_label": segment.paddle_label,
        "page": segment.page,
        "text_content": segment.text_content,
        "text_markdown": segment.text_markdown,
        "bbox": list(segment.bbox) if segment.bbox is not None else None,
        "bbox_norm": list(segment.bbox_norm) if segment.bbox_norm is not None else None,
        "polygon_points": list(segment.polygon_points),
        "source_block_ids": list(segment.source_block_ids),
        "group_id": segment.group_id,
        "block_order": segment.block_order,
        "section_path": list(segment.section_path),
        "heading_level": segment.heading_level,
        "included_for_extraction": segment.included_for_extraction,
        "exclusion_reason": segment.exclusion_reason,
        "confidence": segment.confidence,
        "metadata": dict(segment.metadata),
    }


def _deserialize_segment(payload: dict[str, Any]) -> SemanticSegment:
    bbox = payload.get("bbox")
    bbox_norm = payload.get("bbox_norm")
    return SemanticSegment(
        segment_id=str(payload.get("segment_id") or ""),
        segment_kind=str(payload.get("segment_kind") or ""),
        paddle_label=str(payload.get("paddle_label") or ""),
        page=int(payload.get("page") or 0),
        text_content=str(payload.get("text_content") or ""),
        text_markdown=str(payload.get("text_markdown") or ""),
        bbox=tuple(float(value) for value in bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
        bbox_norm=tuple(float(value) for value in bbox_norm)
        if isinstance(bbox_norm, list) and len(bbox_norm) == 4
        else None,
        polygon_points=_polygon_points(payload.get("polygon_points")),
        source_block_ids=_string_list(payload.get("source_block_ids")),
        group_id=_optional_int(payload.get("group_id")),
        block_order=_optional_int(payload.get("block_order")),
        section_path=_string_list(payload.get("section_path")),
        heading_level=_optional_int(payload.get("heading_level")),
        included_for_extraction=bool(payload.get("included_for_extraction", True)),
        exclusion_reason=_optional_str(payload.get("exclusion_reason")),
        confidence=_optional_float(payload.get("confidence")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _bbox_list(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    normalized: list[list[float]] = []
    for bbox in value:
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        normalized.append([float(item) for item in bbox])
    return normalized


def _polygon_points(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    normalized: list[list[float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            continue
        normalized.append([float(point[0]), float(point[1])])
    return normalized


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
