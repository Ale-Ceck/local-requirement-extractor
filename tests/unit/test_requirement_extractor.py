import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

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
from src.data_models.requirement import Requirement, RequirementList
from src.pdf_processing.models import SemanticDocument, SemanticPage, SemanticSegment
from src.requirement_extraction.chunk_cache import load_chunk_cache
from src.requirement_extraction.document_chunker import SemanticDocumentChunker
from src.requirement_extraction.requirement_extractor import (
    PartialExtractionError,
    RequirementExtractor,
)


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

    def get_model_metadata(self, model_name):
        return {"name": model_name, "digest": "sha256:test-model"}


class FailingMetadataOllamaClient(FakeOllamaClient):
    def get_model_metadata(self, model_name):
        raise RuntimeError(f"metadata unavailable for {model_name}")


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


class ExplodingParser:
    def parse_pdf(self, pdf_path):
        raise AssertionError(
            f"parse_pdf should not be called during chunk-cache replay: {pdf_path}"
        )


class FailingParser:
    def parse_pdf(self, pdf_path):
        raise RuntimeError(f"synthetic parser failure: {pdf_path}")


def build_config(output_dir):
    return AppConfig(
        input=InputConfig(path="data/input", mode="pdf", recursive=False),
        output=OutputConfig(directory=output_dir, include_metadata=True),
        pdf=PDFConfig(),
        parser=ParserConfig(backend="legacy_markdown"),
        chunking=ChunkingConfig(max_chunk_chars=200),
        extraction=ExtractionConfig(
            model_name="test-model", allow_uncited_results=False
        ),
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
        extraction=ExtractionConfig(
            model_name="test-model", allow_uncited_results=False
        ),
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
        polygon_points=(
            [[100.0, 200.0], [320.0, 200.0], [320.0, 230.0], [100.0, 230.0]]
            if bbox is not None
            else []
        ),
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
    assert '<a id="seg-1"></a>' in fake_client.prompts[0]["prompt"]
    assert '<a id="seg-2"></a>' in fake_client.prompts[0]["prompt"]
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


def test_prepare_pdf_command_writes_chunk_cache_and_manifest_without_calling_ollama():
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
            )
        ],
    )

    fake_parser = FakeParser(semantic_document)
    fake_client = FakeOllamaClient([])

    with TemporaryDirectory() as temp_dir:
        input_pdf_path = Path(temp_dir) / "sample.pdf"
        input_pdf_path.write_bytes(b"%PDF-1.4\n% fixture\n")

        config = build_config(temp_dir)
        config.input.path = str(input_pdf_path)
        extractor = RequirementExtractor(
            config=config,
            ollama_client=fake_client,
            document_parser=fake_parser,
            chunker=SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=400)),
            command_name="prepare-pdf",
        )

        extractor.run()

        chunk_cache_path = Path(temp_dir) / "sample.chunks.json"
        manifest_path = Path(temp_dir) / "run-manifest.json"
        assert chunk_cache_path.exists()
        assert (Path(temp_dir) / "sample.anchored.md").exists()
        assert manifest_path.exists()
        assert (Path(temp_dir) / "run-stats.json").exists()
        assert not (Path(temp_dir) / "requirements.xlsx").exists()

        cached_chunks = load_chunk_cache(chunk_cache_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert fake_client.prompts == []
    assert fake_parser.pdf_paths == [str(input_pdf_path)]
    assert len(cached_chunks) == 1
    assert cached_chunks[0].segment_ids == ["seg-1", "seg-2"]
    assert manifest["command"] == "prepare-pdf"
    assert manifest["schema_version"] == 2
    assert manifest["run_id"].startswith("run-")
    assert manifest["execution"]["used_live_ocr"] is True
    assert manifest["execution"]["status"] == "completed"
    assert manifest["execution"]["used_chunk_cache_replay"] is False
    assert manifest["run_stats"]["documents_processed"] == 1
    assert manifest["run_stats"]["chunks_prepared"] == 1
    assert manifest["run_stats"]["chunks_processed"] == 0
    assert manifest["artifacts"]["pdf_sources"][0]["chunk_cache"].endswith(
        "sample.chunks.json"
    )
    assert manifest["reproducibility"]["git"]["commit"]
    assert len(manifest["reproducibility"]["prompt"]["sha256"]) == 64
    assert manifest["reproducibility"]["inputs"][0]["sha256"]


def test_failed_run_writes_terminal_manifest_and_stats(tmp_path: Path) -> None:
    input_pdf_path = tmp_path / "sample.pdf"
    input_pdf_path.write_bytes(b"%PDF-1.4\n% fixture\n")
    output_dir = tmp_path / "failed-run"
    config = build_config(str(output_dir))
    config.input.path = str(input_pdf_path)
    extractor = RequirementExtractor(
        config=config,
        ollama_client=FailingMetadataOllamaClient([]),
        document_parser=FailingParser(),
        command_name="extract",
    )

    with pytest.raises(RuntimeError, match="synthetic parser failure"):
        extractor.run()

    manifest = json.loads(
        (output_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    run_stats = json.loads((output_dir / "run-stats.json").read_text(encoding="utf-8"))
    assert manifest["execution"]["status"] == "failed"
    assert manifest["execution"]["error"]["type"] == "RuntimeError"
    assert manifest["reproducibility"]["model"] == {
        "name": "test-model",
        "digest": None,
    }
    assert run_stats["status"] == "failed"
    assert run_stats["error_type"] == "RuntimeError"
    assert "synthetic parser failure" in run_stats["error_message"]
    assert manifest["artifacts"]["pdf_sources"] == [{"source": str(input_pdf_path)}]


def test_run_can_replay_extraction_from_chunk_cache_without_parser():
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
                        text_content="REQ-001 First requirement.",
                        text_markdown="REQ-001 First requirement.",
                        bbox=(100.0, 200.0, 320.0, 230.0),
                    ),
                ],
            )
        ],
    )

    with TemporaryDirectory() as temp_dir:
        prepare_output_dir = Path(temp_dir) / "prepared"
        replay_output_dir = Path(temp_dir) / "replay"
        input_pdf_path = Path(temp_dir) / "sample.pdf"
        input_pdf_path.write_bytes(b"%PDF-1.4\n% fixture\n")

        prepare_config = build_config(str(prepare_output_dir))
        prepare_config.input.path = str(input_pdf_path)
        prepare_extractor = RequirementExtractor(
            config=prepare_config,
            ollama_client=FakeOllamaClient([]),
            document_parser=FakeParser(semantic_document),
            chunker=SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=400)),
            command_name="prepare-pdf",
        )
        prepare_extractor.run()

        chunk_cache_path = prepare_output_dir / "sample.chunks.json"
        replay_config = build_config(str(replay_output_dir))
        replay_config.input.mode = "chunk_cache"
        replay_config.input.path = str(chunk_cache_path)
        replay_client = FakeOllamaClient(
            [
                json.dumps(
                    [
                        {
                            "code": "REQ-001",
                            "description": "First requirement.",
                            "source_segment_ids": ["seg-1"],
                        }
                    ]
                )
            ]
        )
        replay_extractor = RequirementExtractor(
            config=replay_config,
            ollama_client=replay_client,
            document_parser=ExplodingParser(),
        )

        replay_extractor.run()

        manifest = json.loads(
            (replay_output_dir / "run-manifest.json").read_text(encoding="utf-8")
        )
        run_stats = json.loads(
            (replay_output_dir / "run-stats.json").read_text(encoding="utf-8")
        )
        requirements_exists = (replay_output_dir / "requirements.xlsx").exists()
        review_exists = (replay_output_dir / "requirements.review.json").exists()

    assert requirements_exists is True
    assert review_exists is True
    assert manifest["command"] == "extract"
    assert manifest["input"]["mode"] == "chunk_cache"
    assert manifest["execution"]["used_live_ocr"] is False
    assert manifest["execution"]["used_chunk_cache_replay"] is True
    assert manifest["run_stats"]["chunks_loaded_from_cache"] == 1
    assert manifest["reproducibility"]["model"]["digest"] == "sha256:test-model"
    assert run_stats["chunks_processed"] == 1
    assert run_stats["requirements_written"] == 1


def test_extract_requirements_from_chunks_sorts_results_by_document_position():
    fake_client = FakeOllamaClient(
        [
            json.dumps(
                [
                    {
                        "code": "REQ-020",
                        "description": "Later requirement.",
                        "source_segment_ids": ["seg-20"],
                    }
                ]
            ),
            json.dumps(
                [
                    {
                        "code": "REQ-010",
                        "description": "Earlier requirement.",
                        "source_segment_ids": ["seg-10"],
                    }
                ]
            ),
        ]
    )

    with TemporaryDirectory() as temp_dir:
        extractor = RequirementExtractor(
            config=build_config(temp_dir),
            ollama_client=fake_client,
        )

        later_segment = _segment(
            "seg-20",
            page=2,
            text_content="REQ-020 Later requirement.",
            text_markdown="REQ-020 Later requirement.",
            bbox=(100.0, 300.0, 320.0, 330.0),
        )
        earlier_segment = _segment(
            "seg-10",
            page=1,
            text_content="REQ-010 Earlier requirement.",
            text_markdown="REQ-010 Earlier requirement.",
            bbox=(100.0, 200.0, 320.0, 230.0),
        )

        chunks = [
            SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=400))._build_chunk(
                "sample.pdf",
                [],
                [later_segment],
                {
                    2: SemanticPage(
                        page=2, width=1000, height=1400, segments=[later_segment]
                    ),
                },
            ),
            SemanticDocumentChunker(ChunkingConfig(max_chunk_chars=400))._build_chunk(
                "sample.pdf",
                [],
                [earlier_segment],
                {
                    1: SemanticPage(
                        page=1, width=1000, height=1400, segments=[earlier_segment]
                    ),
                },
            ),
        ]

        requirements = extractor.extract_requirements_from_chunks(chunks)

    assert [requirement.code for requirement in requirements] == ["REQ-010", "REQ-020"]


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

        def _spawn_batch_extractor(
            self, batch_config: AppConfig, *, toc_pruning_plan=None
        ) -> RequirementExtractor:
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
        assert (root_output_dir / "sample.chunks.json").exists()
        assert (root_output_dir / "run-manifest.json").exists()

        batch_root = root_output_dir / "batch-runs" / "sample"
        first_slice = batch_root / "p001-002"
        second_slice = batch_root / "p003-003"

        for slice_dir in (first_slice, second_slice):
            assert (slice_dir / "requirements.xlsx").exists()
            assert (slice_dir / "requirements.review.json").exists()
            assert (slice_dir / "requirements.review.md").exists()
            assert (slice_dir / "requirements.review.html").exists()
            assert (slice_dir / "sample.anchored.md").exists()
            assert (slice_dir / "sample.chunks.json").exists()

        root_chunks = load_chunk_cache(root_output_dir / "sample.chunks.json")
        assert [chunk.chunk_id for chunk in root_chunks] == [
            "p001-002-chunk-1",
            "p003-003-chunk-1",
        ]
        root_review = json.loads(
            (root_output_dir / "requirements.review.json").read_text(encoding="utf-8")
        )
        assert [item["source_chunk_id"] for item in root_review["requirements"]] == [
            "p001-002-chunk-1",
            "p001-002-chunk-1",
            "p003-003-chunk-1",
        ]
        for slice_dir in (first_slice, second_slice):
            slice_chunks = load_chunk_cache(slice_dir / "sample.chunks.json")
            assert all(
                chunk.chunk_id.startswith(f"{slice_dir.name}-")
                for chunk in slice_chunks
            )


def test_batch_mode_builds_toc_plan_once_and_persists_root_report():
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
                        text_content="REQ-001 First requirement.",
                        text_markdown="REQ-001 First requirement.",
                    )
                ],
            ),
            SemanticPage(
                page=1,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-2",
                        page=1,
                        text_content="REQ-002 Second requirement.",
                        text_markdown="REQ-002 Second requirement.",
                    )
                ],
            ),
            SemanticPage(
                page=2,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-3",
                        page=2,
                        text_content="REQ-003 Third requirement.",
                        text_markdown="REQ-003 Third requirement.",
                    )
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
                        "source_segment_ids": ["seg-1"],
                    }
                ]
            ),
            json.dumps(
                [
                    {
                        "code": "REQ-002",
                        "description": "Second requirement.",
                        "source_segment_ids": ["seg-2"],
                    },
                    {
                        "code": "REQ-003",
                        "description": "Third requirement.",
                        "source_segment_ids": ["seg-3"],
                    },
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

        def _spawn_batch_extractor(
            self, batch_config: AppConfig, *, toc_pruning_plan=None
        ) -> RequirementExtractor:
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


def test_deduplication_preserves_identical_requirements_from_different_documents(
    tmp_path: Path,
) -> None:
    extractor = RequirementExtractor(
        build_config(str(tmp_path)),
        ollama_client=FakeOllamaClient([]),
        document_parser=ExplodingParser(),
    )
    requirements = [
        Requirement(
            code=code,
            description="Shared requirement.",
            source_document=document,
        )
        for document, code in [
            ("issue-1.pdf", "REQ-1"),
            ("issue-1.pdf", "req-1"),
            ("issue-2.pdf", "REQ-1"),
        ]
    ]

    finalized = extractor._finalize_requirements(requirements)

    assert [item.source_document for item in finalized] == [
        "issue-1.pdf",
        "issue-2.pdf",
    ]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    ("response", "error_counter", "accepted_count"),
    [
        (None, "llm_response_failures", 0),
        ("not json", "json_parse_failures", 0),
        ('{"unexpected": []}', "chunk_processing_errors", 0),
        ('{"requirements": null}', "chunk_processing_errors", 0),
        (
            '[{"code": 42, "description": "Invalid code type"}]',
            "chunk_processing_errors",
            0,
        ),
        (
            '[{"code": "REQ-1", "description": "Valid", "source_segment_ids": ["seg-1"]},'
            '{"code": null, "description": "Missing code"}]',
            "invalid_items_dropped",
            1,
        ),
    ],
)
def test_run_exports_partial_results_and_raises_on_extraction_data_loss(
    tmp_path: Path,
    parallel: bool,
    response: str | None,
    error_counter: str,
    accepted_count: int,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF-1.4\n% fixture\n")
    output_dir = tmp_path / "output"
    config = build_config(str(output_dir))
    config.input.path = str(input_path)
    config.parallel.enabled = parallel
    document = SemanticDocument(
        source_document=str(input_path),
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _segment(
                        "seg-1",
                        page=0,
                        text_content="REQ-1 Valid",
                        text_markdown="REQ-1 Valid",
                    )
                ],
            )
        ],
    )
    extractor = RequirementExtractor(
        config,
        ollama_client=FakeOllamaClient([response]),
        document_parser=FakeParser(document),
    )

    with pytest.raises(PartialExtractionError, match=error_counter):
        extractor.run()

    manifest = json.loads((output_dir / "run-manifest.json").read_text())
    stats = json.loads((output_dir / "run-stats.json").read_text())
    review = json.loads((output_dir / "requirements.review.json").read_text())
    assert (output_dir / "requirements.xlsx").exists()
    assert len(review["requirements"]) == accepted_count
    assert manifest["execution"]["status"] == "partial"
    assert manifest["execution"]["error"]["type"] == "PartialExtractionError"
    assert manifest["input"]["pdf_sources"] == [str(input_path)]
    assert manifest["run_stats"] == stats
    assert stats["status"] == "partial"
    assert stats[error_counter] == 1
    assert stats["requirements_written"] == accepted_count


@pytest.mark.parametrize("command_name", ["prepare-pdf", "extract"])
def test_batch_failure_preserves_child_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command_name: str
) -> None:
    class FailingBatchExtractor(RequirementExtractor):
        def prepare_pdf_chunks(self, pdf_path: str, *, chunk_id_prefix: str = ""):
            self._increment_stat("chunks_prepared", 2)
            raise RuntimeError("synthetic batch persistence failure")

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF-1.4\n% fixture\n")
    config = build_batch_config(str(tmp_path / "output"))
    config.input.path = str(input_path)
    extractor = RequirementExtractor(
        config,
        ollama_client=FakeOllamaClient([]),
        document_parser=ExplodingParser(),
        command_name=command_name,
    )
    monkeypatch.setattr(extractor, "_count_pdf_pages", lambda _: 3)
    monkeypatch.setattr(
        extractor,
        "_spawn_batch_extractor",
        lambda batch_config, **kwargs: FailingBatchExtractor(
            batch_config,
            ollama_client=FakeOllamaClient([]),
            document_parser=ExplodingParser(),
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic batch persistence failure"):
        extractor.run()

    stats = json.loads((tmp_path / "output" / "run-stats.json").read_text())
    assert stats["status"] == "failed"
    assert stats["chunks_prepared"] == 2
