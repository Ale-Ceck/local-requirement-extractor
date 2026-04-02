import json
from pathlib import Path
from tempfile import TemporaryDirectory

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
from src.data_models.requirement import RequirementList
from src.pdf_processing.models import SemanticDocument, SemanticPage, SemanticSegment
from src.requirement_extraction.document_chunker import SemanticDocumentChunker
from src.requirement_extraction.requirement_extractor import RequirementExtractor


class FakeOllamaClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def get_structured_response(self, prompt, model_name, response_schema=None):
        self.prompts.append(
            {
                "prompt": prompt,
                "model_name": model_name,
                "response_schema": response_schema,
            }
        )
        return self.responses.pop(0)


class FakeParser:
    def __init__(self, semantic_document):
        self.semantic_document = semantic_document
        self.pdf_paths = []

    def parse_pdf(self, pdf_path):
        self.pdf_paths.append(pdf_path)
        return self.semantic_document


class WindowedFakeParser:
    def __init__(self, semantic_document, config):
        self.semantic_document = semantic_document
        self.config = config

    def parse_pdf(self, pdf_path):
        start_page = self.config.parser.page_start or 1
        end_page = self.config.parser.page_end or start_page
        selected_pages = [
            SemanticPage(
                page=page.page,
                width=page.width,
                height=page.height,
                segments=list(page.segments),
                metadata=dict(page.metadata),
            )
            for page in self.semantic_document.pages
            if start_page <= page.page_number <= end_page
        ]
        return SemanticDocument(source_document=pdf_path, pages=selected_pages)


def build_config(output_dir):
    return AppConfig(
        input=InputConfig(path="data/input", mode="pdf", recursive=False),
        output=OutputConfig(directory=output_dir, include_metadata=True),
        pdf=PDFConfig(),
        parser=ParserConfig(backend="legacy_markdown"),
        chunking=ChunkingConfig(max_chunk_chars=200),
        extraction=ExtractionConfig(model_name="test-model", allow_uncited_results=False),
        parallel=ParallelConfig(enabled=False, max_workers=1),
        ollama=OllamaConfig(),
        logging=LoggingConfig(),
    )


def build_batch_config(output_dir):
    return AppConfig(
        input=InputConfig(path="data/input/sample.pdf", mode="pdf", recursive=False),
        output=OutputConfig(directory=output_dir, include_metadata=True),
        pdf=PDFConfig(),
        parser=ParserConfig(
            backend="paddleocr_vl",
            batch_page_count=2,
            batch_output_subdir="batch-runs",
            persist_anchored_markdown=True,
        ),
        chunking=ChunkingConfig(max_chunk_chars=200),
        extraction=ExtractionConfig(model_name="test-model", allow_uncited_results=False),
        parallel=ParallelConfig(enabled=False, max_workers=1),
        ollama=OllamaConfig(),
        logging=LoggingConfig(),
    )


def _segment(
    segment_id: str,
    *,
    page: int,
    text_content: str,
    text_markdown: str,
    section_path=None,
    heading_level=None,
    bbox=None,
    paddle_label="text",
):
    return SemanticSegment(
        segment_id=segment_id,
        segment_kind="text",
        paddle_label=paddle_label,
        page=page,
        text_content=text_content,
        text_markdown=text_markdown,
        bbox=bbox,
        bbox_norm=(0.1, 0.2, 0.3, 0.4) if bbox is not None else None,
        polygon_points=[[100.0, 200.0], [320.0, 200.0], [320.0, 230.0], [100.0, 230.0]] if bbox is not None else [],
        source_block_ids=[segment_id.replace("seg", "blk")],
        section_path=section_path or [],
        heading_level=heading_level,
        included_for_extraction=True,
        metadata={
            "page_width": 1000.0,
            "page_height": 1400.0,
            "page_image_path": None,
            "layout_order": 2,
            "layout_cls_id": 22,
        },
    )


def test_extract_requirements_from_pdf_resolves_cited_segment_provenance():
    semantic_document = SemanticDocument(
        source_document="sample.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-1",
                        page=0,
                        text_content="3.1 Functional Requirements",
                        text_markdown="## 3.1 Functional Requirements",
                        heading_level=2,
                        paddle_label="paragraph_title",
                        section_path=["3.1 Functional Requirements"],
                    ),
                    _segment(
                        "seg-2",
                        page=0,
                        text_content="REQ-001 The system shall authenticate users.",
                        text_markdown="REQ-001 The system shall authenticate users.",
                        section_path=["3.1 Functional Requirements"],
                        bbox=(100.0, 200.0, 320.0, 230.0),
                    ),
                ],
            )
        ],
    )

    fake_parser = FakeParser(semantic_document)
    fake_client = FakeOllamaClient(
        [
            json.dumps(
                [
                    {
                        "code": "REQ-001",
                        "description": "The system shall authenticate users.",
                        "source_segment_ids": ["seg-1", "seg-2"],
                    }
                ]
            )
        ]
    )

    with TemporaryDirectory() as temp_dir:
        extractor = RequirementExtractor(
            config=build_config(temp_dir),
            ollama_client=fake_client,
            document_parser=fake_parser,
            chunker=SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=400)),
        )

        requirements = extractor.extract_requirements_from_pdf("sample.pdf")
        anchored_markdown_path = Path(temp_dir) / "sample.anchored.md"
        assert anchored_markdown_path.exists()

    assert fake_parser.pdf_paths == ["sample.pdf"]
    assert len(fake_client.prompts) == 1
    assert "<a id=\"seg-1\"></a>" in fake_client.prompts[0]["prompt"]
    assert "<a id=\"seg-2\"></a>" in fake_client.prompts[0]["prompt"]
    assert isinstance(requirements, RequirementList)
    assert len(requirements) == 1
    requirement = requirements[0]
    assert requirement.code == "REQ-001"
    assert requirement.source_segment_ids == ["seg-1", "seg-2"]
    assert requirement.source_document == "sample.pdf"
    assert requirement.source_page_start == 1
    assert requirement.source_page_end == 1
    assert requirement.source_block_ids == ["blk-1", "blk-2"]
    assert requirement.source_section == "3.1 Functional Requirements"
    assert "authenticate users" in (requirement.source_text_excerpt or "")
    assert requirement.review_status is None
    assert requirement.source_regions[1]["segment_id"] == "seg-2"
    assert requirement.source_regions[1]["bbox_norm"] == [0.1, 0.2, 0.3, 0.4]


def test_extract_requirements_from_pdf_marks_invalid_segment_ids_for_review():
    semantic_document = SemanticDocument(
        source_document="sample.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-1",
                        page=0,
                        text_content="REQ-001 The system shall authenticate users.",
                        text_markdown="REQ-001 The system shall authenticate users.",
                        bbox=(100.0, 200.0, 320.0, 230.0),
                    ),
                ],
            )
        ],
    )

    fake_parser = FakeParser(semantic_document)
    fake_client = FakeOllamaClient(
        [
            json.dumps(
                [
                    {
                        "code": "REQ-001",
                        "description": "The system shall authenticate users.",
                        "source_segment_ids": ["seg-999"],
                    }
                ]
            )
        ]
    )

    with TemporaryDirectory() as temp_dir:
        extractor = RequirementExtractor(
            config=build_config(temp_dir),
            ollama_client=fake_client,
            document_parser=fake_parser,
            chunker=SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=400)),
        )

        requirements = extractor.extract_requirements_from_pdf("sample.pdf")

    requirement = requirements[0]
    assert requirement.source_segment_ids == []
    assert requirement.review_status == "invalid_source_segment_ids"
    assert requirement.source_chunk_id == "chunk-1"


def test_extract_requirements_from_markdown_uses_chunk_metadata():
    with TemporaryDirectory() as temp_dir:
        markdown_path = Path(temp_dir) / "input.md"
        markdown_path.write_text(
            "# System Requirements\n\n## Security\n\nREQ-002 The system shall encrypt data.\n",
            encoding="utf-8",
        )

        fake_client = FakeOllamaClient(
            [
                json.dumps(
                    [
                        {
                            "code": "REQ-002",
                            "description": "The system shall encrypt data.",
                            "source_segment_ids": [],
                        }
                    ]
                )
            ]
        )

        extractor = RequirementExtractor(
            config=build_config(temp_dir),
            ollama_client=fake_client,
        )

        requirements = extractor.extract_requirements_from_markdown(str(markdown_path))

    assert len(requirements) == 1
    requirement = requirements[0]
    assert requirement.code == "REQ-002"
    assert requirement.source_document == str(markdown_path)


def test_run_in_pdf_batch_mode_writes_root_and_per_slice_outputs():
    semantic_document = SemanticDocument(
        source_document="sample.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-1",
                        page=0,
                        text_content="1. Requirements",
                        text_markdown="# 1. Requirements",
                        heading_level=1,
                        paddle_label="doc_title",
                        section_path=["1. Requirements"],
                    ),
                    _segment(
                        "seg-2",
                        page=0,
                        text_content="REQ-001 First requirement.",
                        text_markdown="REQ-001 First requirement.",
                        section_path=["1. Requirements"],
                        bbox=(100.0, 200.0, 320.0, 230.0),
                    ),
                ],
            ),
            SemanticPage(
                page=1,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-3",
                        page=1,
                        text_content="REQ-002 Second requirement.",
                        text_markdown="REQ-002 Second requirement.",
                        section_path=["1. Requirements"],
                        bbox=(100.0, 240.0, 320.0, 270.0),
                    ),
                ],
            ),
            SemanticPage(
                page=2,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-4",
                        page=2,
                        text_content="2. Interfaces",
                        text_markdown="## 2. Interfaces",
                        heading_level=2,
                        paddle_label="paragraph_title",
                        section_path=["2. Interfaces"],
                    ),
                    _segment(
                        "seg-5",
                        page=2,
                        text_content="REQ-003 Third requirement.",
                        text_markdown="REQ-003 Third requirement.",
                        section_path=["2. Interfaces"],
                        bbox=(100.0, 300.0, 320.0, 330.0),
                    ),
                ],
            ),
        ],
    )

    fake_client = FakeOllamaClient(
        [
            json.dumps(
                [
                    {
                        "code": "REQ-001",
                        "description": "First requirement.",
                        "source_segment_ids": ["seg-2"],
                    },
                    {
                        "code": "REQ-002",
                        "description": "Second requirement.",
                        "source_segment_ids": ["seg-3"],
                    },
                ]
            ),
            json.dumps(
                [
                    {
                        "code": "REQ-003",
                        "description": "Third requirement.",
                        "source_segment_ids": ["seg-5"],
                    }
                ]
            ),
        ]
    )

    class BatchTestRequirementExtractor(RequirementExtractor):
        def _count_pdf_pages(self, pdf_path: str) -> int:
            return 3

        def _spawn_batch_extractor(self, batch_config: AppConfig, *, toc_pruning_plan=None) -> RequirementExtractor:
            return RequirementExtractor(
                config=batch_config,
                ollama_client=self.ollama_client,
                document_parser=WindowedFakeParser(semantic_document, batch_config),
                toc_pruning_plan=toc_pruning_plan,
            )

    with TemporaryDirectory() as temp_dir:
        input_pdf_path = Path(temp_dir) / "sample.pdf"
        input_pdf_path.write_bytes(b"%PDF-1.4\n% fixture\n")

        config = build_batch_config(temp_dir)
        config.input.path = str(input_pdf_path)

        extractor = BatchTestRequirementExtractor(
            config=config,
            ollama_client=fake_client,
            document_parser=WindowedFakeParser(semantic_document, config),
        )

        extractor.run()

        root_output_dir = Path(temp_dir)
        assert (root_output_dir / "requirements.xlsx").exists()
        assert (root_output_dir / "requirements.review.json").exists()
        assert (root_output_dir / "requirements.review.md").exists()
        assert (root_output_dir / "requirements.review.html").exists()
        assert (root_output_dir / "sample.anchored.md").exists()

        batch_root = root_output_dir / "batch-runs" / "sample"
        first_slice = batch_root / "p001-002"
        second_slice = batch_root / "p003-003"

        for slice_dir in (first_slice, second_slice):
            assert (slice_dir / "requirements.xlsx").exists()
            assert (slice_dir / "requirements.review.json").exists()
            assert (slice_dir / "requirements.review.md").exists()
            assert (slice_dir / "requirements.review.html").exists()
            assert (slice_dir / "sample.anchored.md").exists()


def test_batch_mode_builds_toc_plan_once_and_persists_root_report():
    semantic_document = SemanticDocument(
        source_document="sample.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[_segment("seg-1", page=0, text_content="REQ-001 First requirement.", text_markdown="REQ-001 First requirement.")],
            ),
            SemanticPage(
                page=1,
                width=1000,
                height=1400,
                segments=[_segment("seg-2", page=1, text_content="REQ-002 Second requirement.", text_markdown="REQ-002 Second requirement.")],
            ),
            SemanticPage(
                page=2,
                width=1000,
                height=1400,
                segments=[_segment("seg-3", page=2, text_content="REQ-003 Third requirement.", text_markdown="REQ-003 Third requirement.")],
            ),
        ],
    )

    fake_client = FakeOllamaClient(
        [
            json.dumps([{"code": "REQ-001", "description": "First requirement.", "source_segment_ids": ["seg-1"]}]),
            json.dumps(
                [
                    {"code": "REQ-002", "description": "Second requirement.", "source_segment_ids": ["seg-2"]},
                    {"code": "REQ-003", "description": "Third requirement.", "source_segment_ids": ["seg-3"]},
                ]
            ),
        ]
    )

    toc_plan = {
        "mode": "audit",
        "source_document": "sample.pdf",
        "toc_reliable": True,
        "toc_pages": [1],
        "toc_entry_count": 2,
        "toc_entries": [
            {
                "title": "Summary",
                "normalized_title": "summary",
                "start_page": 1,
                "section_level": 1,
                "range_start_page": 1,
                "range_end_page": 1,
                "matched_excluded_rule": "summary",
                "would_exclude": True,
            },
            {
                "title": "Requirements",
                "normalized_title": "requirements",
                "start_page": 2,
                "section_level": 1,
                "range_start_page": 2,
                "range_end_page": 3,
                "matched_excluded_rule": None,
                "would_exclude": False,
            },
        ],
        "excluded_ranges": [
            {
                "title": "Summary",
                "normalized_title": "summary",
                "source_toc_page": 1,
                "start_page": 1,
                "end_page": 1,
                "matched_excluded_rule": "summary",
            }
        ],
        "plan_source": "toc_preflight_scan",
    }

    class FakeParserWithTocPlan(WindowedFakeParser):
        def __init__(self, semantic_document, config):
            super().__init__(semantic_document, config)
            self.toc_plan_requests = []

        def build_toc_pruning_plan(self, pdf_path):
            self.toc_plan_requests.append(pdf_path)
            return dict(toc_plan)

    class CapturingBatchRequirementExtractor(RequirementExtractor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.spawned_plans = []

        def _count_pdf_pages(self, pdf_path: str) -> int:
            return 3

        def _spawn_batch_extractor(self, batch_config: AppConfig, *, toc_pruning_plan=None) -> RequirementExtractor:
            self.spawned_plans.append(toc_pruning_plan)
            return RequirementExtractor(
                config=batch_config,
                ollama_client=self.ollama_client,
                document_parser=WindowedFakeParser(semantic_document, batch_config),
                toc_pruning_plan=toc_pruning_plan,
            )

    with TemporaryDirectory() as temp_dir:
        input_pdf_path = Path(temp_dir) / "sample.pdf"
        input_pdf_path.write_bytes(b"%PDF-1.4\n% fixture\n")

        config = build_batch_config(temp_dir)
        config.input.path = str(input_pdf_path)

        fake_parser = FakeParserWithTocPlan(semantic_document, config)
        extractor = CapturingBatchRequirementExtractor(
            config=config,
            ollama_client=fake_client,
            document_parser=fake_parser,
        )

        extractor.run()

        assert fake_parser.toc_plan_requests == [str(input_pdf_path)]
        assert extractor.spawned_plans == [toc_plan, toc_plan]

        report_path = Path(temp_dir) / "toc-pruning-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["excluded_ranges"][0]["title"] == "Summary"
        assert report["plan_source"] == "toc_preflight_scan"


def test_parse_llm_response_accepts_wrapper_shapes():
    with TemporaryDirectory() as temp_dir:
        extractor = RequirementExtractor(
            config=build_config(temp_dir),
            ollama_client=FakeOllamaClient([]),
        )

        result = extractor.parse_llm_response(
            json.dumps(
                {
                    "requirements": [
                        {
                            "code": "REQ-100",
                            "description": "Wrapped requirement",
                            "source_segment_ids": ["seg-1"],
                        },
                    ]
                }
            )
        )

    assert len(result) == 1
    assert result[0].code == "REQ-100"
    assert result[0].source_segment_ids == ["seg-1"]


def test_parse_llm_response_rejects_items_missing_code_or_description():
    with TemporaryDirectory() as temp_dir:
        extractor = RequirementExtractor(
            config=build_config(temp_dir),
            ollama_client=FakeOllamaClient([]),
        )

        result = extractor.parse_llm_response(
            json.dumps(
                [
                    {
                        "code": "REQ-100",
                        "description": "Valid requirement",
                        "source_segment_ids": ["seg-1"],
                    },
                    {
                        "code": None,
                        "description": "Missing code",
                        "source_segment_ids": ["seg-2"],
                    },
                    {
                        "code": "REQ-101",
                        "description": None,
                        "source_segment_ids": ["seg-3"],
                    },
                    {
                        "code": "   ",
                        "description": "Blank code",
                        "source_segment_ids": ["seg-4"],
                    },
                    {
                        "code": "REQ-102",
                        "description": "   ",
                        "source_segment_ids": ["seg-5"],
                    },
                ]
            )
        )

    assert len(result) == 1
    assert result[0].code == "REQ-100"
    assert result[0].description == "Valid requirement"
