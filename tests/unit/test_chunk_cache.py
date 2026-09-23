"""Contract tests for reusable chunk caches and their provenance."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.pdf_processing.models import SemanticSegment
from src.requirement_extraction.chunk_cache import load_chunk_cache, write_chunk_cache
from src.requirement_extraction.document_chunker import AnchoredMarkdownChunk


@pytest.fixture
def cached_chunk() -> AnchoredMarkdownChunk:
    """Return a chunk containing every optional provenance field."""
    segment = SemanticSegment(
        segment_id="segment-1",
        segment_kind="text",
        paddle_label="text",
        page=2,
        text_content="REQ-001 The system shall preserve provenance.",
        text_markdown="**REQ-001** The system shall preserve provenance.",
        bbox=(10.0, 20.0, 100.0, 200.0),
        bbox_norm=(0.1, 0.1, 1.0, 1.0),
        polygon_points=[[10.0, 20.0], [100.0, 200.0]],
        source_block_ids=["block-1"],
        group_id=1,
        block_order=2,
        section_path=["Requirements"],
        heading_level=2,
        included_for_extraction=True,
        exclusion_reason=None,
        confidence=0.98,
        metadata={"page_image_path": "pages/page-3.png"},
    )
    return AnchoredMarkdownChunk(
        chunk_id="chunk-1",
        markdown_text='<a id="segment-1"></a>\n' + segment.text_markdown,
        source_document="document.pdf",
        source_page_start=3,
        source_page_end=3,
        segment_ids=["segment-1"],
        source_block_ids=["block-1"],
        source_section="Requirements",
        source_text_excerpt=segment.text_content,
        source_bbox_list=[[10.0, 20.0, 100.0, 200.0]],
        source_regions=[{"segment_id": "segment-1", "page": 2}],
        source_segments=[segment],
    )


@pytest.fixture
def cache_payload(
    tmp_path: Path, cached_chunk: AnchoredMarkdownChunk
) -> dict[str, Any]:
    """Build the exact payload emitted by the current writer."""
    path = tmp_path / "valid.chunks.json"
    write_chunk_cache([cached_chunk], path)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _load_payload(tmp_path: Path, payload: Any) -> list[AnchoredMarkdownChunk]:
    path = tmp_path / "input.chunks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_chunk_cache(path)


def test_chunk_cache_roundtrip_preserves_all_provenance(
    tmp_path: Path, cached_chunk: AnchoredMarkdownChunk
) -> None:
    path = tmp_path / "nested" / "document.chunks.json"
    assert write_chunk_cache([cached_chunk], path) == str(path)
    assert load_chunk_cache(path) == [cached_chunk]


def test_empty_cache_and_chunks_without_optional_provenance_are_valid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.chunks.json"
    write_chunk_cache([], path, source_document="empty.pdf")
    assert load_chunk_cache(path) == []
    chunk = _load_payload(
        tmp_path,
        {
            "version": 1,
            "chunk_count": 1,
            "chunks": [
                {
                    "chunk_id": "plain",
                    "markdown_text": "Text",
                    "source_document": "plain.md",
                }
            ],
        },
    )[0]
    assert chunk.source_segments == []
    assert chunk.source_page_start is None


@pytest.mark.parametrize(
    "payload,reason",
    [
        ([], "root must be an object"),
        ({"version": 2}, "Unsupported chunk cache version"),
        ({"version": True}, "Unsupported chunk cache version"),
        ({"version": 1}, "'chunks' must be a list"),
        ({"version": 1, "chunks": {}}, "'chunks' must be a list"),
        ({"version": 1, "chunks": []}, "'chunk_count' must equal"),
        ({"version": 1, "chunk_count": 1, "chunks": []}, "'chunk_count' must equal"),
        (
            {"version": 1, "chunk_count": False, "chunks": []},
            "'chunk_count' must equal",
        ),
        ({"version": 1, "chunk_count": 1, "chunks": [None]}, "chunk must be an object"),
    ],
)
def test_rejects_invalid_cache_structure(
    tmp_path: Path, payload: Any, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason) as exc_info:
        _load_payload(tmp_path, payload)
    assert "input.chunks.json" in str(exc_info.value)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("chunk_id", " ", "chunk_id must not be empty"),
        ("markdown_text", "\n", "markdown_text must not be empty"),
        ("source_document", "", "source_document must not be empty"),
        ("chunk_id", 42, "chunk_id must be a string"),
        ("segment_ids", "segment-1", "segment_ids must be a list"),
        ("segment_ids", [None], "segment_ids must be a string"),
        (
            "segment_ids",
            ["segment-1", "segment-1"],
            "segment_ids contains duplicate IDs",
        ),
        ("segment_ids", ["unknown"], "segment_ids must match"),
        ("source_segments", ["bad"], "source_segments must be a list of objects"),
        ("source_segments", [], "segment_ids must match"),
        ("source_bbox_list", None, "source_bbox_list must be a list"),
        (
            "source_bbox_list",
            [[1, 2]],
            "source_bbox_list entries must contain 4 coordinates",
        ),
        (
            "source_bbox_list",
            [[1, 2, 3, "bad"]],
            "source_bbox_list must contain finite numbers",
        ),
        ("source_regions", "invalid", "source_regions must be a list of objects"),
        ("source_page_start", 4, "source_page_start must not exceed"),
        ("source_page_start", 1.5, "source_page_start must be an integer"),
        ("source_page_start", True, "source_page_start must be an integer"),
    ],
)
def test_rejects_invalid_chunks_without_silent_data_loss(
    tmp_path: Path, cache_payload: dict[str, Any], field: str, value: Any, reason: str
) -> None:
    cache_payload["chunks"][0][field] = value
    with pytest.raises(ValueError, match=reason) as exc_info:
        _load_payload(tmp_path, cache_payload)
    assert "chunks[0]" in str(exc_info.value)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("segment_id", "", "segment_id must not be empty"),
        ("page", -1, "page must be a nonnegative integer"),
        ("page", None, "page must be a nonnegative integer"),
        (
            "included_for_extraction",
            "false",
            "included_for_extraction must be a boolean",
        ),
        ("metadata", [], "metadata must be an object"),
        ("bbox", [1, 2, 3], "bbox entries must contain 4 coordinates"),
        ("polygon_points", [[1]], "polygon_points entries must contain 2 coordinates"),
        ("confidence", float("nan"), "confidence must contain finite numbers"),
    ],
)
def test_rejects_invalid_source_segments(
    tmp_path: Path, cache_payload: dict[str, Any], field: str, value: Any, reason: str
) -> None:
    cache_payload["chunks"][0]["source_segments"][0][field] = value
    with pytest.raises(ValueError, match=reason) as exc_info:
        _load_payload(tmp_path, cache_payload)
    assert "source_segments[0]" in str(exc_info.value)


def test_rejects_duplicate_segments_and_legacy_duplicate_chunk_ids(
    tmp_path: Path, cache_payload: dict[str, Any]
) -> None:
    duplicate_chunk_payload = deepcopy(cache_payload)
    duplicate_chunk_payload["chunks"].append(deepcopy(cache_payload["chunks"][0]))
    duplicate_chunk_payload["chunk_count"] = 2
    with pytest.raises(
        ValueError, match="duplicate chunk_id.*regenerate legacy batch caches"
    ):
        _load_payload(tmp_path, duplicate_chunk_payload)
    segments = cache_payload["chunks"][0]["source_segments"]
    segments.append(deepcopy(segments[0]))
    with pytest.raises(
        ValueError, match="source_segments.segment_id contains duplicate IDs"
    ):
        _load_payload(tmp_path, cache_payload)


def test_invalid_write_does_not_replace_an_existing_cache(
    tmp_path: Path, cached_chunk: AnchoredMarkdownChunk
) -> None:
    path = tmp_path / "document.chunks.json"
    write_chunk_cache([cached_chunk], path)
    original_bytes = path.read_bytes()
    cached_chunk.markdown_text = ""
    with pytest.raises(ValueError, match="markdown_text must not be empty"):
        write_chunk_cache([cached_chunk], path)
    assert path.read_bytes() == original_bytes


def test_invalid_json_reports_cache_path(tmp_path: Path) -> None:
    path = tmp_path / "broken.chunks.json"
    path.write_text('{"chunks":', encoding="utf-8")
    with pytest.raises(
        ValueError, match="Invalid chunk cache JSON.*broken.chunks.json"
    ):
        load_chunk_cache(path)
