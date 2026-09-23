from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from src.pdf_processing.models import SemanticSegment
from src.requirement_extraction.document_chunker import AnchoredMarkdownChunk

CHUNK_CACHE_VERSION = 1


def write_chunk_cache(
    chunks: list[AnchoredMarkdownChunk],
    output_path: str | Path,
    *,
    source_document: str | None = None,
) -> str:
    """Write a versioned cache after validating every chunk and its provenance."""
    path = Path(output_path)
    payload = {
        "version": CHUNK_CACHE_VERSION,
        "source_document": source_document
        or (chunks[0].source_document if chunks else None),
        "chunk_count": len(chunks),
        "chunks": [_serialize_chunk(chunk) for chunk in chunks],
    }
    _deserialize_cache(payload, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def load_chunk_cache(cache_path: str | Path) -> list[AnchoredMarkdownChunk]:
    """Load a v1 cache, rejecting malformed data with its file and field location.

    Optional provenance fields may be absent, but supplied fields must be valid.
    Legacy batch caches with repeated chunk IDs must be regenerated.
    """
    path = Path(cache_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid chunk cache JSON in {path}: {exc}") from exc
    return _deserialize_cache(payload, path)


def _deserialize_cache(payload: Any, path: Path) -> list[AnchoredMarkdownChunk]:
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid chunk cache in {path}: root must be an object.")
    if (
        type(payload.get("version")) is not int
        or payload["version"] != CHUNK_CACHE_VERSION
    ):
        raise ValueError(
            f"Unsupported chunk cache version in {path}: {payload.get('version')!r}"
        )

    chunks_payload = payload.get("chunks")
    if not isinstance(chunks_payload, list):
        raise ValueError(
            f"Invalid chunk cache payload in {path}: 'chunks' must be a list."
        )

    count = payload.get("chunk_count")
    if type(count) is not int or count != len(chunks_payload):
        raise ValueError(
            f"Invalid chunk cache in {path}: 'chunk_count' must equal "
            f"the number of chunks ({len(chunks_payload)})."
        )

    chunks: list[AnchoredMarkdownChunk] = []
    seen_ids: set[str] = set()
    for index, chunk_payload in enumerate(chunks_payload):
        try:
            chunk = _deserialize_chunk(chunk_payload)
            if chunk.chunk_id in seen_ids:
                raise ValueError(
                    f"duplicate chunk_id {chunk.chunk_id!r}; regenerate legacy batch caches"
                )
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)
        except ValueError as exc:
            raise ValueError(
                f"Invalid chunk cache in {path}, chunks[{index}]: {exc}"
            ) from exc
    return chunks


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
        "source_segments": [
            _serialize_segment(segment) for segment in chunk.source_segments
        ],
    }


def _deserialize_chunk(payload: Any) -> AnchoredMarkdownChunk:
    if not isinstance(payload, dict):
        raise ValueError("chunk must be an object")
    segment_ids = _string_list(payload.get("segment_ids", []), "segment_ids")
    _require_unique(segment_ids, "segment_ids")
    segments_payload = _object_list(
        payload.get("source_segments", []), "source_segments"
    )
    source_segments = []
    for index, segment_payload in enumerate(segments_payload):
        try:
            source_segments.append(_deserialize_segment(segment_payload))
        except ValueError as exc:
            raise ValueError(f"source_segments[{index}]: {exc}") from exc
    source_ids = [segment.segment_id for segment in source_segments]
    _require_unique(source_ids, "source_segments.segment_id")
    if set(segment_ids) != set(source_ids):
        raise ValueError("segment_ids must match source_segments.segment_id")

    page_start = _optional_int(payload.get("source_page_start"), "source_page_start")
    page_end = _optional_int(payload.get("source_page_end"), "source_page_end")
    if page_start is not None and page_end is not None and page_start > page_end:
        raise ValueError("source_page_start must not exceed source_page_end")

    return AnchoredMarkdownChunk(
        chunk_id=_required_str(payload.get("chunk_id"), "chunk_id"),
        markdown_text=_required_str(payload.get("markdown_text"), "markdown_text"),
        source_document=_required_str(
            payload.get("source_document"), "source_document"
        ),
        source_page_start=page_start,
        source_page_end=page_end,
        segment_ids=segment_ids,
        source_block_ids=_string_list(
            payload.get("source_block_ids", []), "source_block_ids"
        ),
        source_section=_optional_str(payload.get("source_section"), "source_section"),
        source_text_excerpt=_optional_str(
            payload.get("source_text_excerpt"), "source_text_excerpt"
        ),
        source_bbox_list=_coordinate_list(
            payload.get("source_bbox_list", []), 4, "source_bbox_list"
        ),
        source_regions=_object_list(
            payload.get("source_regions", []), "source_regions"
        ),
        source_segments=source_segments,
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
    page = _optional_int(payload.get("page", 0), "page")
    if page is None or page < 0:
        raise ValueError("page must be a nonnegative integer")
    included = payload.get("included_for_extraction", True)
    if not isinstance(included, bool):
        raise ValueError("included_for_extraction must be a boolean")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    return SemanticSegment(
        segment_id=_required_str(payload.get("segment_id"), "segment_id"),
        segment_kind=_text(payload.get("segment_kind", ""), "segment_kind"),
        paddle_label=_text(payload.get("paddle_label", ""), "paddle_label"),
        page=page,
        text_content=_text(payload.get("text_content", ""), "text_content"),
        text_markdown=_text(payload.get("text_markdown", ""), "text_markdown"),
        bbox=_optional_bbox(payload.get("bbox"), "bbox"),
        bbox_norm=_optional_bbox(payload.get("bbox_norm"), "bbox_norm"),
        polygon_points=_coordinate_list(
            payload.get("polygon_points", []), 2, "polygon_points"
        ),
        source_block_ids=_string_list(
            payload.get("source_block_ids", []), "source_block_ids"
        ),
        group_id=_optional_int(payload.get("group_id"), "group_id"),
        block_order=_optional_int(payload.get("block_order"), "block_order"),
        section_path=_string_list(payload.get("section_path", []), "section_path"),
        heading_level=_optional_int(payload.get("heading_level"), "heading_level"),
        included_for_extraction=included,
        exclusion_reason=_optional_str(
            payload.get("exclusion_reason"), "exclusion_reason"
        ),
        confidence=_optional_float(payload.get("confidence"), "confidence"),
        metadata=dict(metadata),
    )


def _coordinate_list(value: Any, size: int, field: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    normalized: list[list[float]] = []
    for coordinates in value:
        if not isinstance(coordinates, list) or len(coordinates) != size:
            raise ValueError(f"{field} entries must contain {size} coordinates")
        normalized.append([_number(item, field) for item in coordinates])
    return normalized


def _optional_bbox(value: Any, field: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    x1, y1, x2, y2 = _coordinate_list([value], 4, field)[0]
    return x1, y1, x2, y2


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [_required_str(item, field) for item in value]


def _object_list(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field} must be a list of objects")
    return cast(list[dict[str, Any]], value)


def _require_unique(values: list[str], field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} contains duplicate IDs")


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer or null")
    return value


def _optional_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field)


def _number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must contain finite numbers")
    return float(value)


def _optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _required_str(value: Any, field: str) -> str:
    text = _text(value, field)
    if not text.strip():
        raise ValueError(f"{field} must not be empty")
    return text
