from config.schema import ChunkingConfig
from src.pdf_processing.models import SemanticDocument, SemanticPage, SemanticSegment
from src.requirement_extraction.document_chunker import SemanticDocumentChunker


def _segment(
    segment_id: str,
    *,
    page: int,
    text_content: str,
    text_markdown: str,
    section_path=None,
    heading_level=None,
    segment_kind="text",
    paddle_label="text",
    included_for_extraction=True,
    bbox=(0.0, 0.0, 100.0, 20.0),
):
    return SemanticSegment(
        segment_id=segment_id,
        segment_kind=segment_kind,
        paddle_label=paddle_label,
        page=page,
        text_content=text_content,
        text_markdown=text_markdown,
        bbox=bbox,
        bbox_norm=(0.0, 0.0, 0.1, 0.1),
        source_block_ids=[segment_id.replace("seg", "blk")],
        section_path=section_path or [],
        heading_level=heading_level,
        included_for_extraction=included_for_extraction,
        metadata={"page_width": 1000.0, "page_height": 1400.0, "page_image_path": None},
    )


def test_chunk_document_preserves_section_block_ids_and_segment_ids():
    document = SemanticDocument(
        source_document="spec.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-1",
                        page=0,
                        text_content="Section A",
                        text_markdown="## Section A",
                        paddle_label="paragraph_title",
                        heading_level=2,
                        section_path=["Section A"],
                    ),
                    _segment(
                        "seg-2",
                        page=0,
                        text_content="REQ-001 The system shall record telemetry.",
                        text_markdown="REQ-001 The system shall record telemetry.",
                        section_path=["Section A"],
                    ),
                    _segment(
                        "seg-3",
                        page=0,
                        text_content="REQ-002 The system shall persist telemetry for 24h.",
                        text_markdown="REQ-002 The system shall persist telemetry for 24h.",
                        section_path=["Section A"],
                    ),
                ],
            )
        ],
    )

    chunker = SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=500))
    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_document == "spec.pdf"
    assert chunk.source_section == "Section A"
    assert chunk.segment_ids == ["seg-1", "seg-2", "seg-3"]
    assert chunk.source_block_ids == ["blk-1", "blk-2", "blk-3"]
    assert chunk.source_page_start == 1
    assert chunk.source_page_end == 1
    assert "<a id=\"seg-1\"></a>" in chunk.markdown_text


def test_chunk_document_does_not_orphan_heading_segments():
    document = SemanticDocument(
        source_document="spec.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-1",
                        page=0,
                        text_content="REQ-001 The system shall record telemetry.",
                        text_markdown="REQ-001 The system shall record telemetry.",
                    ),
                    _segment(
                        "seg-2",
                        page=1,
                        text_content="3.1 Interfaces",
                        text_markdown="## 3.1 Interfaces",
                        heading_level=2,
                        paddle_label="paragraph_title",
                        section_path=["3.1 Interfaces"],
                    ),
                    _segment(
                        "seg-3",
                        page=1,
                        text_content="REQ-002 The system shall expose telemetry over SpaceWire.",
                        text_markdown="REQ-002 The system shall expose telemetry over SpaceWire.",
                        section_path=["3.1 Interfaces"],
                    ),
                ],
            )
        ],
    )

    chunker = SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=80))
    chunks = chunker.chunk_document(document)

    assert len(chunks) == 2
    assert chunks[1].segment_ids == ["seg-2", "seg-3"]
    assert chunks[1].source_page_start == 2
    assert chunks[1].source_page_end == 2


def test_chunk_document_keeps_multi_segment_requirement_together_when_section_exceeds_limit():
    document = SemanticDocument(
        source_document="spec.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-1",
                        page=0,
                        text_content="HAA-121 / CREATED / T.A",
                        text_markdown="## HAA-121 / CREATED / T.A",
                        heading_level=2,
                        paddle_label="paragraph_title",
                        section_path=["HAA-121 / CREATED / T.A"],
                    ),
                    _segment(
                        "seg-2",
                        page=0,
                        text_content="The nominal mass of the HAA shall not exceed 14.5 kg.",
                        text_markdown="The nominal mass of the HAA shall not exceed 14.5 kg.",
                        section_path=["HAA-121 / CREATED / T.A"],
                    ),
                    _segment(
                        "seg-3",
                        page=0,
                        text_content="The value includes harness and mounting bracket contributions.",
                        text_markdown="The value includes harness and mounting bracket contributions.",
                        section_path=["HAA-121 / CREATED / T.A"],
                    ),
                ],
            )
        ],
    )

    chunker = SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=40))
    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].segment_ids == ["seg-1", "seg-2", "seg-3"]
    assert chunks[0].source_section == "HAA-121 / CREATED / T.A"


def test_chunk_document_keeps_table_with_current_heading_section():
    document = SemanticDocument(
        source_document="spec.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-1",
                        page=0,
                        text_content="4.2.3 Thermal Interface Matrix",
                        text_markdown="## 4.2.3 Thermal Interface Matrix",
                        heading_level=2,
                        paddle_label="paragraph_title",
                        section_path=["4.2.3 Thermal Interface Matrix"],
                    ),
                    _segment(
                        "seg-2",
                        page=0,
                        text_content="The following conductance values apply at the interface.",
                        text_markdown="The following conductance values apply at the interface.",
                        section_path=["4.2.3 Thermal Interface Matrix"],
                    ),
                    _segment(
                        "seg-3",
                        page=0,
                        text_content="<table><tr><td>Surface</td><td>Conductance</td></tr></table>",
                        text_markdown="<table><tr><td>Surface</td><td>Conductance</td></tr></table>",
                        segment_kind="table",
                        paddle_label="table",
                        section_path=["4.2.3 Thermal Interface Matrix"],
                    ),
                    _segment(
                        "seg-4",
                        page=0,
                        text_content="4.2.4 Survival Heaters",
                        text_markdown="## 4.2.4 Survival Heaters",
                        heading_level=2,
                        paddle_label="paragraph_title",
                        section_path=["4.2.4 Survival Heaters"],
                    ),
                    _segment(
                        "seg-5",
                        page=0,
                        text_content="Heater control shall remain independent from survival monitoring.",
                        text_markdown="Heater control shall remain independent from survival monitoring.",
                        section_path=["4.2.4 Survival Heaters"],
                    ),
                ],
            )
        ],
    )

    chunker = SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=50))
    chunks = chunker.chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].segment_ids == ["seg-1", "seg-2", "seg-3"]
    assert chunks[0].source_section == "4.2.3 Thermal Interface Matrix"
    assert chunks[1].segment_ids == ["seg-4", "seg-5"]


def test_chunk_document_skips_segments_excluded_from_extraction():
    document = SemanticDocument(
        source_document="spec.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-1",
                        page=0,
                        text_content="Table of Contents",
                        text_markdown="## Table of Contents",
                        heading_level=2,
                        included_for_extraction=False,
                    ),
                    _segment(
                        "seg-2",
                        page=0,
                        text_content="REQ-001 The system shall record telemetry.",
                        text_markdown="REQ-001 The system shall record telemetry.",
                    ),
                ],
            )
        ],
    )

    chunker = SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=200))
    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].segment_ids == ["seg-2"]
    assert "Table of Contents" not in chunks[0].markdown_text
