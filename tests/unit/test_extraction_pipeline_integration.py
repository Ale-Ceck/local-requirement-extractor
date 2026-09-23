import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from config.schema import (
    AppConfig,
    ChunkingConfig,
    ExtractionConfig,
    InputConfig,
    LoggingConfig,
    OllamaConfig,
    OutputConfig,
    ParallelConfig,
    ParserConfig,
    PDFConfig,
)
from src.pdf_processing.models import SemanticDocument, SemanticPage, SemanticSegment
from src.requirement_extraction.requirement_extractor import RequirementExtractor


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "parsed_documents" / "sample_parsed_document.json"


class FixtureParser:
    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path

    def parse_pdf(self, pdf_path: str) -> SemanticDocument:
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        pages = []
        for page_payload in payload["pages"]:
            segments = [
                SemanticSegment(
                    segment_id=segment_payload["segment_id"],
                    segment_kind=segment_payload["segment_kind"],
                    paddle_label=segment_payload["paddle_label"],
                    page=segment_payload["page"],
                    text_content=segment_payload["text_content"],
                    text_markdown=segment_payload["text_markdown"],
                    bbox=tuple(segment_payload["bbox"]) if segment_payload.get("bbox") else None,
                    bbox_norm=tuple(segment_payload["bbox_norm"]) if segment_payload.get("bbox_norm") else None,
                    polygon_points=segment_payload.get("polygon_points", []),
                    source_block_ids=list(segment_payload.get("source_block_ids", [])),
                    group_id=segment_payload.get("group_id"),
                    block_order=segment_payload.get("block_order"),
                    section_path=list(segment_payload.get("section_path", [])),
                    heading_level=segment_payload.get("heading_level"),
                    included_for_extraction=segment_payload.get("included_for_extraction", True),
                    exclusion_reason=segment_payload.get("exclusion_reason"),
                    metadata=segment_payload.get("metadata", {}),
                )
                for segment_payload in page_payload["segments"]
            ]
            pages.append(
                SemanticPage(
                    page=page_payload["page"],
                    width=page_payload["width"],
                    height=page_payload["height"],
                    segments=segments,
                )
            )
        return SemanticDocument(source_document=payload["source_document"], pages=pages)


class FixtureOllamaClient:
    def __init__(self):
        self.calls = 0

    def get_structured_response(self, prompt, model_name, response_schema=None):
        responses = [
            json.dumps(
                [
                    {
                        "code": "REQ-100",
                        "description": "The instrument shall provide thermal telemetry.",
                        "source_segment_ids": ["seg-p000-i002"],
                    },
                    {
                        "code": "REQ-101",
                        "description": "The instrument shall retain telemetry for 24 hours.",
                        "source_segment_ids": ["seg-p000-i003"],
                    },
                ]
            ),
            json.dumps(
                [
                    {
                        "code": "REQ-102",
                        "description": "The instrument shall expose telemetry over SpaceWire.",
                        "source_segment_ids": ["seg-p001-i002"],
                    },
                ]
            ),
        ]
        response = responses[self.calls]
        self.calls += 1
        return response


def build_config(input_path: str, output_dir: str) -> AppConfig:
    return AppConfig(
        input=InputConfig(path=input_path, mode="pdf", recursive=False),
        output=OutputConfig(directory=output_dir, include_metadata=True, write_review_artifact=True),
        pdf=PDFConfig(),
        parser=ParserConfig(backend="paddleocr_vl", persist_anchored_markdown=True),
        chunking=ChunkingConfig(max_chunk_chars=200),
        extraction=ExtractionConfig(model_name="test-model"),
        parallel=ParallelConfig(enabled=False, max_workers=1),
        ollama=OllamaConfig(),
        logging=LoggingConfig(),
    )


def test_run_creates_excel_review_artifact_and_anchored_markdown_from_semantic_fixture():
    with TemporaryDirectory() as temp_dir:
        input_pdf_path = Path(temp_dir) / "fixture-input.pdf"
        input_pdf_path.write_bytes(b"%PDF-1.4\n% fixture\n")

        config = build_config(str(input_pdf_path), temp_dir)
        extractor = RequirementExtractor(
            config=config,
            ollama_client=FixtureOllamaClient(),
            document_parser=FixtureParser(FIXTURE_PATH),
        )

        extractor.run()

        excel_path = Path(temp_dir) / "requirements.xlsx"
        review_path = Path(temp_dir) / "requirements.review.json"
        anchored_markdown_path = Path(temp_dir) / "fixture-input.anchored.md"

        assert excel_path.exists()
        assert review_path.exists()
        assert anchored_markdown_path.exists()
        dataframe = pd.read_excel(excel_path)
        review_payload = json.loads(review_path.read_text(encoding="utf-8"))

    assert len(dataframe) == 3
    assert dataframe["Requirement Code"].tolist() == ["REQ-100", "REQ-101", "REQ-102"]
    assert dataframe["Page Start"].tolist() == [1, 1, 2]
    assert dataframe["Source Segment IDs"].tolist() == [
        "seg-p000-i002",
        "seg-p000-i003",
        "seg-p001-i002",
    ]
    assert review_payload["requirements"][2]["source_block_ids"] == ["blk-p002-i002"]
    assert review_payload["requirements"][2]["source_section"] == "3.1 Interfaces"
