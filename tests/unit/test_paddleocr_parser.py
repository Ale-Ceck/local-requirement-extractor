import json
import re
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types

import pytest

from config.schema import ParserConfig
from src.pdf_processing.models import SemanticDocument, SemanticPage, SemanticSegment
from src.pdf_processing.paddleocr_parser import PaddleOCRVLParser
from src.requirement_extraction.document_chunker import SemanticDocumentChunker


class FakeBlock:
    def __init__(self, *, label=None, block_label=None, content=None, block_content=None, bbox=None, score=None, group_id=None, block_order=None):
        self.label = label
        self.block_label = block_label
        self.content = content
        self.block_content = block_content
        self.bbox = bbox
        self.score = score
        self.group_id = group_id
        self.block_order = block_order


class FakePageResult(dict):
    def save_to_img(self, save_path: str):
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "visualized.png").write_bytes(b"fake-png")


def test_build_pipeline_kwargs_uses_mlx_vlm_service_configuration():
    config = ParserConfig(
        language="en",
        vlm_backend="mlx-vlm-server",
        vlm_server_url="http://localhost:8111/",
        vlm_api_model_name="mlx-community/PaddleOCR-VL-1.5-bf16",
    )

    parser = PaddleOCRVLParser(config)

    kwargs = parser._build_pipeline_kwargs()

    assert "lang" not in kwargs
    assert kwargs["vl_rec_backend"] == "mlx-vlm-server"
    assert kwargs["vl_rec_server_url"] == "http://localhost:8111/"
    assert kwargs["vl_rec_api_model_name"] == "mlx-community/PaddleOCR-VL-1.5-bf16"


def test_build_predict_kwargs_uses_supported_runtime_flags():
    config = ParserConfig(
        layout_detection=True,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=False,
    )

    parser = PaddleOCRVLParser(config)

    kwargs = parser._build_predict_kwargs()

    assert kwargs == {
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": True,
        "use_layout_detection": True,
    }


def test_get_pipeline_wraps_constructor_errors(monkeypatch):
    class BrokenPaddleOCRVL:
        def __init__(self, **kwargs):
            raise ValueError("Unknown argument: lang")

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        types.SimpleNamespace(PaddleOCRVL=BrokenPaddleOCRVL),
    )

    parser = PaddleOCRVLParser(ParserConfig(vlm_server_url="http://localhost:8111/"))
    monkeypatch.setattr(parser._vlm_service, "ensure_healthy", lambda: None)

    with pytest.raises(RuntimeError, match="Failed to initialize PaddleOCR-VL"):
        parser._get_pipeline()


def test_get_pipeline_surfaces_missing_paddle_dependency(monkeypatch):
    class BrokenPaddleOCRVL:
        def __init__(self, **kwargs):
            raise ModuleNotFoundError("No module named 'paddle'")

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        types.SimpleNamespace(PaddleOCRVL=BrokenPaddleOCRVL),
    )

    parser = PaddleOCRVLParser(ParserConfig(vlm_server_url="http://localhost:8111/"))
    monkeypatch.setattr(parser._vlm_service, "ensure_healthy", lambda: None)

    with pytest.raises(RuntimeError, match="paddlepaddle"):
        parser._get_pipeline()


def test_parse_pdf_wraps_predict_errors(monkeypatch):
    class BrokenPipeline:
        def predict(self, **kwargs):
            raise ConnectionError("service unavailable")

    parser = PaddleOCRVLParser(ParserConfig(vlm_server_url="http://localhost:8111/"))
    monkeypatch.setattr(parser, "_get_pipeline", lambda: BrokenPipeline())

    with pytest.raises(RuntimeError, match="Failed to parse PDF"):
        parser.parse_pdf("sample.pdf")


def test_build_document_uses_semantic_segments_and_section_context(monkeypatch):
    parser = PaddleOCRVLParser(ParserConfig())
    monkeypatch.setattr(
        parser,
        "_resolve_page_image_paths",
        lambda pdf_path, raw_result: ({1: "/tmp/page-1.png"}, {1: "/tmp/ocr-page-1.png"}),
    )

    raw_result = [
        {
            "width": 596,
            "height": 842,
            "page_index": 0,
            "page_count": 1,
            "layout_det_res": {
                "boxes": [
                    {
                        "cls_id": 17,
                        "label": "paragraph_title",
                        "order": 3,
                        "score": 0.783,
                        "coordinate": [70, 121, 134, 132],
                        "polygon_points": [[70, 121], [134, 121], [134, 132], [70, 132]],
                    },
                    {
                        "cls_id": 22,
                        "label": "text",
                        "order": 4,
                        "score": 0.851,
                        "coordinate": [99, 155, 348, 167],
                        "polygon_points": [[99, 155], [348, 155], [348, 167], [99, 167]],
                    },
                ]
            },
            "parsing_res_list": [
                FakeBlock(
                    label="paragraph_title",
                    content="4.1.2 Mass",
                    bbox=[70, 121, 134, 132],
                    score=0.78,
                    group_id=0,
                    block_order=1,
                ),
                FakeBlock(
                    label="text",
                    content="The HAA mass shall not exceed 17.5 kg.",
                    bbox=[99, 155, 348, 167],
                    score=0.85,
                    group_id=1,
                    block_order=2,
                ),
            ],
        }
    ]

    document = parser._build_document("sample.pdf", raw_result)

    assert len(document.pages) == 1
    assert document.pages[0].width == 596
    assert document.pages[0].height == 842
    assert len(document.pages[0].segments) == 2
    assert document.pages[0].metadata["page_image_path"] == "/tmp/page-1.png"
    assert document.pages[0].metadata["ocr_page_image_path"] == "/tmp/ocr-page-1.png"

    heading = document.pages[0].segments[0]
    paragraph = document.pages[0].segments[1]

    assert heading.segment_kind == "text"
    assert heading.paddle_label == "paragraph_title"
    assert heading.text_content == "4.1.2 Mass"
    assert heading.text_markdown.endswith("4.1.2 Mass")
    assert heading.section_path == ["4.1.2 Mass"]
    assert heading.included_for_extraction is True
    assert heading.metadata["layout_order"] == 3
    assert heading.polygon_points == [[70, 121], [134, 121], [134, 132], [70, 132]]

    assert paragraph.paddle_label == "text"
    assert paragraph.section_path == ["4.1.2 Mass"]
    assert paragraph.bbox == (99.0, 155.0, 348.0, 167.0)
    assert paragraph.confidence == 0.85
    assert paragraph.metadata["layout_cls_id"] == 22
    assert paragraph.bbox_norm == pytest.approx(
        (99.0 / 596.0, 155.0 / 842.0, 348.0 / 596.0, 167.0 / 842.0)
    )


def test_build_document_keeps_branding_title_out_of_section_lineage(monkeypatch):
    parser = PaddleOCRVLParser(ParserConfig())
    monkeypatch.setattr(parser, "_resolve_page_image_paths", lambda pdf_path, raw_result: ({}, {}))

    raw_result = [
        {
            "width": 596,
            "height": 842,
            "parsing_res_list": [
                FakeBlock(
                    label="paragraph_title",
                    content="AIRBUS",
                    bbox=[88, 60, 176, 80],
                ),
                FakeBlock(
                    label="text",
                    content="Reference: JUI-ADSF-SYS-RS-000204 Issue: 4 Date: 29.11.2019",
                    bbox=[70, 101, 344, 112],
                ),
            ],
        }
    ]

    document = parser._build_document("sample.pdf", raw_result)

    assert document.pages[0].segments[0].heading_level is None
    assert document.pages[0].segments[0].section_path == []
    assert document.pages[0].segments[1].section_path == []


def test_build_document_marks_segments_that_overlap_table_regions(monkeypatch):
    parser = PaddleOCRVLParser(ParserConfig())
    monkeypatch.setattr(parser, "_resolve_page_image_paths", lambda pdf_path, raw_result: ({}, {}))

    raw_result = [
        {
            "width": 596,
            "height": 842,
            "layout_det_res": {
                "boxes": [
                    {
                        "cls_id": 21,
                        "label": "table",
                        "order": None,
                        "score": 0.94,
                        "coordinate": [146, 693, 448, 745],
                        "polygon_points": [[146, 693], [448, 693], [448, 745], [146, 745]],
                    }
                ]
            },
            "parsing_res_list": [
                FakeBlock(
                    label="table",
                    content="<table><tr><td>Surface</td><td>Conductance</td></tr></table>",
                    bbox=[146, 693, 448, 745],
                    score=0.94,
                ),
            ],
        }
    ]

    document = parser._build_document("sample.pdf", raw_result)

    assert document.pages[0].metadata["table_regions"] == [
        {"bbox": [146.0, 693.0, 448.0, 745.0], "label": "table", "score": 0.94}
    ]
    assert document.pages[0].segments[0].metadata["table_bboxes"] == [[146.0, 693.0, 448.0, 745.0]]


def test_build_document_excludes_default_ignored_labels_but_keeps_relevant_note(monkeypatch):
    parser = PaddleOCRVLParser(ParserConfig())
    monkeypatch.setattr(parser, "_resolve_page_image_paths", lambda pdf_path, raw_result: ({}, {}))

    raw_result = [
        {
            "width": 596,
            "height": 842,
            "parsing_res_list": [
                FakeBlock(
                    label="header",
                    content="PROPRIETARY & CONFIDENTIAL",
                    bbox=[0, 0, 100, 20],
                ),
                FakeBlock(
                    label="vision_footnote",
                    content="Note: the timing shall be defined by [AD 028].",
                    bbox=[10, 100, 200, 140],
                ),
            ],
        }
    ]

    document = parser._build_document("sample.pdf", raw_result)

    assert document.pages[0].segments[0].included_for_extraction is False
    assert document.pages[0].segments[0].exclusion_reason == "ignored_label:header"
    assert document.pages[0].segments[1].included_for_extraction is True
    assert document.pages[0].segments[1].exclusion_reason is None


def test_build_document_excludes_front_matter_segments(monkeypatch):
    parser = PaddleOCRVLParser(ParserConfig(exclude_front_matter=True))
    monkeypatch.setattr(parser, "_resolve_page_image_paths", lambda pdf_path, raw_result: ({}, {}))

    raw_result = [
        {
            "width": 596,
            "height": 842,
            "parsing_res_list": [
                FakeBlock(
                    label="paragraph_title",
                    content="Table of Contents",
                    bbox=[0, 0, 200, 20],
                ),
                FakeBlock(
                    label="text",
                    content="3.3 Documentation delivery ........ 13",
                    bbox=[0, 30, 260, 50],
                ),
            ],
        }
    ]

    document = parser._build_document("sample.pdf", raw_result)

    assert all(segment.included_for_extraction is False for segment in document.pages[0].segments)
    assert document.pages[0].segments[0].exclusion_reason == "front_matter"


def test_build_document_attaches_rasterized_pdf_page_images(monkeypatch):
    parser = PaddleOCRVLParser(ParserConfig())
    monkeypatch.setattr(
        parser,
        "_resolve_page_image_paths",
        lambda pdf_path, raw_result: ({1: "/tmp/page-1.png"}, {1: "/tmp/ocr-page-1.png"}),
    )

    raw_result = [
        {
            "width": 596,
            "height": 842,
            "parsing_res_list": [
                FakeBlock(
                    label="text",
                    content="REQ-001 The system shall authenticate users.",
                    bbox=[99, 155, 348, 167],
                    score=0.85,
                ),
            ],
        }
    ]

    document = parser._build_document("sample.pdf", raw_result)

    assert document.pages[0].metadata["page_image_path"] == "/tmp/page-1.png"
    assert document.pages[0].metadata["ocr_page_image_path"] == "/tmp/ocr-page-1.png"
    assert document.pages[0].segments[0].metadata["page_image_path"] == "/tmp/page-1.png"
    assert document.pages[0].segments[0].metadata["ocr_page_image_path"] == "/tmp/ocr-page-1.png"


def test_generated_anchored_markdown_matches_example_12_fixture(monkeypatch):
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures" / "paddle_examples"
    json_path = fixture_dir / "example_12_res.json"
    markdown_path = fixture_dir / "example_12.md"

    parser = PaddleOCRVLParser(ParserConfig())
    monkeypatch.setattr(parser, "_resolve_page_image_paths", lambda pdf_path, raw_result: ({}, {}))

    raw_result = json.loads(json_path.read_text(encoding="utf-8"))
    document = parser._build_document("example.pdf", raw_result)

    chunker = SemanticDocumentChunker()
    rendered_markdown = chunker.render_anchored_markdown(chunker.build_extraction_segments(document))
    expected_markdown = markdown_path.read_text(encoding="utf-8")

    assert _normalize_markdown(rendered_markdown) == _normalize_markdown(expected_markdown)
    assert "PROPRIETARY & CONFIDENTIAL" not in rendered_markdown
    assert "Page 13/27" not in rendered_markdown
    assert "Note: the need for these reviews shall be established" in rendered_markdown


def test_prepare_pdf_input_slices_requested_page_range():
    fitz = pytest.importorskip("fitz")

    with TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "sample.pdf"
        document = fitz.open()
        for _ in range(4):
            document.new_page(width=200, height=300)
        document.save(str(pdf_path))
        document.close()

        parser = PaddleOCRVLParser(ParserConfig(page_start=2, page_end=3))

        sliced_path, page_offset = parser._prepare_pdf_input(pdf_path)

        sliced_document = fitz.open(str(sliced_path))
        try:
            assert page_offset == 1
            assert len(sliced_document) == 2
        finally:
            sliced_document.close()


def test_persist_parse_input_copies_sliced_pdf_into_output_directory():
    fitz = pytest.importorskip("fitz")

    with TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "sample.pdf"
        artifact_dir = Path(temp_dir) / "artifacts"
        document = fitz.open()
        for _ in range(4):
            document.new_page(width=200, height=300)
        document.save(str(pdf_path))
        document.close()

        parser = PaddleOCRVLParser(ParserConfig(page_start=2, page_end=3), output_directory=str(artifact_dir))
        sliced_path, page_offset = parser._prepare_pdf_input(pdf_path)
        persisted_path = parser._persist_parse_input(sliced_path)

        persisted_document = fitz.open(str(persisted_path))
        try:
            assert page_offset == 1
            assert persisted_path == artifact_dir / "ocr-input.pdf"
            assert persisted_path.exists()
            assert len(persisted_document) == 2
        finally:
            persisted_document.close()


def test_rasterize_pdf_pages_persists_images_in_output_directory():
    fitz = pytest.importorskip("fitz")

    with TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "sample.pdf"
        artifact_dir = Path(temp_dir) / "artifacts"
        document = fitz.open()
        for _ in range(2):
            document.new_page(width=200, height=300)
        document.save(str(pdf_path))
        document.close()

        parser = PaddleOCRVLParser(ParserConfig(), output_directory=str(artifact_dir))
        page_image_paths = parser._rasterize_pdf_pages(pdf_path, page_count=2)

        assert page_image_paths == {
            1: str(artifact_dir / "page-images" / "page-001.png"),
            2: str(artifact_dir / "page-images" / "page-002.png"),
        }
        assert (artifact_dir / "page-images" / "page-001.png").exists()
        assert (artifact_dir / "page-images" / "page-002.png").exists()


def test_resolve_page_image_paths_uses_raw_pages_for_review_and_keeps_native_ocr_visualizations(monkeypatch):
    parser = PaddleOCRVLParser(ParserConfig(), output_directory="/tmp/test-artifacts")
    monkeypatch.setattr(
        parser,
        "_rasterize_pdf_pages",
        lambda pdf_path, page_count: {1: "/tmp/page-images/page-001.png", 2: "/tmp/page-images/page-002.png"},
    )

    raw_result = [
        FakePageResult(
            {
                "width": 596,
                "height": 842,
                "parsing_res_list": [],
            }
        ),
        FakePageResult(
            {
                "width": 596,
                "height": 842,
                "parsing_res_list": [],
            }
        ),
    ]

    with TemporaryDirectory() as temp_dir:
        parser.output_directory = Path(temp_dir)
        page_paths, ocr_paths = parser._resolve_page_image_paths(str(Path(temp_dir) / "sample.pdf"), raw_result)

        assert page_paths == {
            1: "/tmp/page-images/page-001.png",
            2: "/tmp/page-images/page-002.png",
        }
        assert ocr_paths == {
            1: str(Path(temp_dir) / "ocr-pages" / "page-001.png"),
            2: str(Path(temp_dir) / "ocr-pages" / "page-002.png"),
        }
        assert (Path(temp_dir) / "ocr-pages" / "page-001.png").read_bytes() == b"fake-png"
        assert (Path(temp_dir) / "ocr-pages" / "page-002.png").read_bytes() == b"fake-png"


def test_build_document_offsets_absolute_page_numbers(monkeypatch):
    parser = PaddleOCRVLParser(ParserConfig())
    monkeypatch.setattr(
        parser,
        "_resolve_page_image_paths",
        lambda pdf_path, raw_result: ({1: "/tmp/page-1.png"}, {1: "/tmp/ocr-page-1.png"}),
    )

    raw_result = [
        {
            "width": 596,
            "height": 842,
            "parsing_res_list": [
                FakeBlock(
                    label="text",
                    content="REQ-001 The system shall authenticate users.",
                    bbox=[99, 155, 348, 167],
                    score=0.85,
                ),
            ],
        }
    ]

    document = parser._build_document("sample.pdf", raw_result, page_offset=4)

    assert document.pages[0].page == 4
    assert document.pages[0].page_number == 5
    assert document.pages[0].segments[0].page == 4
    assert document.pages[0].segments[0].page_number == 5


def test_parse_pdf_uses_persisted_ocr_input_path_for_predict_and_metadata(monkeypatch):
    fitz = pytest.importorskip("fitz")

    class FakePipeline:
        def __init__(self):
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return []

    with TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "sample.pdf"
        artifact_dir = Path(temp_dir) / "artifacts"
        document = fitz.open()
        document.new_page(width=200, height=300)
        document.save(str(pdf_path))
        document.close()

        parser = PaddleOCRVLParser(ParserConfig(), output_directory=str(artifact_dir))
        fake_pipeline = FakePipeline()
        monkeypatch.setattr(parser, "_get_pipeline", lambda: fake_pipeline)
        monkeypatch.setattr(
            parser,
            "_build_document",
            lambda pdf_path, raw_result, *, page_offset=0, parse_source_path=None: {
                "pdf_path": pdf_path,
                "page_offset": page_offset,
                "parse_source_path": str(parse_source_path),
            },
        )

        result = parser.parse_pdf(str(pdf_path))

        assert fake_pipeline.calls[0]["input"] == str(artifact_dir / "ocr-input.pdf")
        assert result["parse_source_path"] == str(artifact_dir / "ocr-input.pdf")


def _normalize_markdown(value: str) -> str:
    normalized = re.sub(r'<a id="[^"]+"></a>\s*', "", value)
    normalized = re.sub(r"<table[^>]*>", "<table>", normalized)
    normalized = re.sub(r"<td[^>]*>", "<td>", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _toc_segment(segment_id: str, page: int, text: str) -> SemanticSegment:
    return SemanticSegment(
        segment_id=segment_id,
        segment_kind="text",
        paddle_label="text",
        page=page,
        text_content=text,
        text_markdown=text,
        source_block_ids=[segment_id.replace("seg", "blk")],
        included_for_extraction=True,
    )


def _build_toc_test_document() -> SemanticDocument:
    return SemanticDocument(
        source_document="sample.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _toc_segment("seg-001", 0, "Contents"),
                    _toc_segment(
                        "seg-002",
                        0,
                        "\n".join(
                            [
                                "1 Introduction and Scope ..... 2",
                                "2 Documents ..... 4",
                                "3 Functional and Performance Requirements ..... 6",
                                "Annex A Section Cross Reference ..... 10",
                            ]
                        ),
                    ),
                ],
            ),
            SemanticPage(page=1, width=1000, height=1400, segments=[_toc_segment("seg-003", 1, "Intro content")]),
            SemanticPage(page=2, width=1000, height=1400, segments=[_toc_segment("seg-004", 2, "Still intro")]),
            SemanticPage(page=3, width=1000, height=1400, segments=[_toc_segment("seg-005", 3, "Documents content")]),
            SemanticPage(page=4, width=1000, height=1400, segments=[_toc_segment("seg-006", 4, "Documents continuation")]),
            SemanticPage(
                page=5,
                width=1000,
                height=1400,
                segments=[_toc_segment("seg-007", 5, "HAA-54 The system shall provide measurements.")],
            ),
            SemanticPage(
                page=6,
                width=1000,
                height=1400,
                segments=[_toc_segment("seg-008", 6, "HAA-56 The system shall warm up in 36 h.")],
            ),
            SemanticPage(
                page=7,
                width=1000,
                height=1400,
                segments=[_toc_segment("seg-009", 7, "Further requirement content.")],
            ),
            SemanticPage(
                page=8,
                width=1000,
                height=1400,
                segments=[_toc_segment("seg-010", 8, "Requirement content still present.")],
            ),
            SemanticPage(
                page=9,
                width=1000,
                height=1400,
                segments=[_toc_segment("seg-011", 9, "Cross reference appendix content.")],
            ),
        ],
    )


def test_build_toc_pruning_report_marks_excluded_ranges():
    parser = PaddleOCRVLParser(ParserConfig(toc_section_pruning_mode="audit"))

    report = parser._build_toc_pruning_report(_build_toc_test_document())

    assert report["toc_reliable"] is True
    assert report["toc_pages"] == [1]
    assert report["toc_entry_count"] == 4

    by_title = {entry["normalized_title"]: entry for entry in report["toc_entries"]}
    assert by_title["introduction and scope"]["matched_excluded_rule"] == "introduction and scope"
    assert by_title["documents"]["matched_excluded_rule"] == "documents"
    assert by_title["functional and performance requirements"]["matched_excluded_rule"] is None
    assert by_title["annex a section cross reference"]["matched_excluded_rule"] == "section cross reference"
    assert by_title["introduction and scope"]["range_start_page"] == 2
    assert by_title["introduction and scope"]["range_end_page"] == 3
    assert by_title["documents"]["range_start_page"] == 4
    assert by_title["documents"]["range_end_page"] == 5
    assert by_title["annex a section cross reference"]["range_start_page"] == 10
    assert by_title["annex a section cross reference"]["range_end_page"] == 10


def test_parse_toc_entry_line_ignores_figure_and_table_index_entries():
    parser = PaddleOCRVLParser(ParserConfig())

    assert parser._parse_toc_entry_line("Figure 1.1-1: JUICE Mission ..... 8") is None
    assert parser._parse_toc_entry_line("Table 4.1-1: Qualification levels ..... 18") is None
    assert parser._parse_toc_entry_line("3 Functional and Performance Requirements ..... 12") == {
        "title": "3 Functional and Performance Requirements",
        "normalized_title": "functional and performance requirements",
        "start_page": 12,
        "section_level": 1,
    }


def test_build_toc_pruning_report_ignores_figure_and_table_entries_for_reliability():
    parser = PaddleOCRVLParser(ParserConfig(toc_section_pruning_mode="audit"))
    document = SemanticDocument(
        source_document="sample.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _toc_segment("seg-001", 0, "Contents"),
                    _toc_segment(
                        "seg-002",
                        0,
                        "\n".join(
                            [
                                "1 Introduction and Scope ..... 7",
                                "2 Documents ..... 11",
                                "7 Product Assurance Requirements ..... 24",
                                "Distribution List ..... 28",
                                "Figure 1.1-1: JUICE Mission ..... 8",
                                "Table 4.1-1: Qualification levels ..... 18",
                            ]
                        ),
                    ),
                ],
            )
        ],
    )

    report = parser._build_toc_pruning_report(document, document_page_count=29)

    assert report["toc_reliable"] is True
    assert report["toc_reliability_reason"] == "ok"
    normalized_titles = [entry["normalized_title"] for entry in report["toc_entries"]]
    assert "figure 1 1 1 juice mission" not in normalized_titles
    assert "table 4 1 1 qualification levels" not in normalized_titles
    assert normalized_titles == [
        "introduction and scope",
        "documents",
        "product assurance requirements",
        "distribution list",
    ]


def test_build_toc_pruning_plan_uses_full_pdf_page_count_for_preflight_scan(monkeypatch):
    sparse_toc_document = SemanticDocument(
        source_document="sample.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[
                    _toc_segment("seg-001", 0, "Contents"),
                    _toc_segment(
                        "seg-002",
                        0,
                        "\n".join(
                            [
                                "1 Introduction and Scope ..... 7",
                                "2 Documents ..... 11",
                                "Annex A Section Cross Reference ..... 29",
                            ]
                        ),
                    ),
                ],
            )
        ],
    )

    monkeypatch.setattr(PaddleOCRVLParser, "_count_pdf_pages", lambda self, pdf_path: 29)
    monkeypatch.setattr(PaddleOCRVLParser, "parse_pdf", lambda self, pdf_path: sparse_toc_document)

    parser = PaddleOCRVLParser(ParserConfig(toc_section_pruning_mode="enforce"))

    report = parser.build_toc_pruning_plan("sample.pdf")

    assert report["mode"] == "enforce"
    assert report["toc_reliable"] is True
    assert report["toc_reliability_reason"] == "ok"
    by_title = {entry["normalized_title"]: entry for entry in report["toc_entries"]}
    assert by_title["annex a section cross reference"]["range_start_page"] == 29
    assert by_title["annex a section cross reference"]["range_end_page"] == 29


def test_apply_toc_pruning_in_audit_mode_writes_report_without_excluding_segments():
    config = ParserConfig(
        toc_section_pruning_mode="audit",
        toc_pruning_report_filename="toc-pruning-report.json",
    )
    with TemporaryDirectory() as temp_dir:
        parser = PaddleOCRVLParser(config, output_directory=temp_dir)
        document = _build_toc_test_document()

        parser._apply_toc_section_pruning(document)

        assert all(
            segment.included_for_extraction
            for page in document.pages
            for segment in page.segments
        )
        report_path = Path(temp_dir) / "toc-pruning-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["pruning_applied"] is False
        assert report["pruning_skipped_reason"] == "audit_only"
        assert any(item["matched_excluded_rule"] == "documents" for item in report["toc_entries"])


def test_apply_toc_pruning_in_enforce_mode_excludes_matching_section_pages():
    config = ParserConfig(toc_section_pruning_mode="enforce")
    parser = PaddleOCRVLParser(config)
    document = _build_toc_test_document()

    parser._apply_toc_section_pruning(document)

    page_by_number = {page.page_number: page for page in document.pages}
    assert page_by_number[2].segments[0].included_for_extraction is False
    assert page_by_number[4].segments[0].included_for_extraction is False
    assert page_by_number[6].segments[0].included_for_extraction is True
    assert page_by_number[10].segments[0].included_for_extraction is False
    assert page_by_number[2].segments[0].exclusion_reason == "excluded_toc_section:introduction and scope"
    assert document.metadata["toc_section_pruning"]["pruning_applied"] is True
    assert document.metadata["toc_section_pruning"]["excluded_page_count"] == 5


def test_apply_toc_pruning_skips_when_toc_is_unreliable():
    config = ParserConfig(toc_section_pruning_mode="enforce")
    parser = PaddleOCRVLParser(config)
    document = SemanticDocument(
        source_document="sample.pdf",
        pages=[
            SemanticPage(
                page=0,
                width=1000,
                height=1400,
                segments=[_toc_segment("seg-001", 0, "Only one TOC-like line ..... 2")],
            ),
            SemanticPage(
                page=1,
                width=1000,
                height=1400,
                segments=[_toc_segment("seg-002", 1, "Valid requirement content.")],
            ),
        ],
    )

    parser._apply_toc_section_pruning(document)

    assert document.pages[1].segments[0].included_for_extraction is True
    report = document.metadata["toc_section_pruning"]
    assert report["toc_reliable"] is False
    assert report["pruning_applied"] is False
    assert report["pruning_skipped_reason"] == "no_toc_pages_detected"


def test_apply_toc_pruning_uses_precomputed_plan_for_slice_documents():
    toc_plan = {
        "mode": "audit",
        "source_document": "sample.pdf",
        "toc_reliable": True,
        "toc_reliability_reason": "ok",
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
                "range_end_page": 4,
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

    parser = PaddleOCRVLParser(ParserConfig(toc_section_pruning_mode="enforce"), toc_pruning_plan=toc_plan)
    document = SemanticDocument(
        source_document="sample.pdf",
        pages=[
            SemanticPage(page=0, width=1000, height=1400, segments=[_toc_segment("seg-001", 0, "Summary body")]),
            SemanticPage(page=1, width=1000, height=1400, segments=[_toc_segment("seg-002", 1, "Requirement body")]),
        ],
    )

    parser._apply_toc_section_pruning(document)

    assert document.pages[0].segments[0].included_for_extraction is False
    assert document.pages[1].segments[0].included_for_extraction is True
    report = document.metadata["toc_section_pruning"]
    assert report["plan_source"] == "toc_preflight_scan"
    assert report["document_page_window"] == {"start_page": 1, "end_page": 2}
    assert report["window_excluded_ranges"][0]["title"] == "Summary"
