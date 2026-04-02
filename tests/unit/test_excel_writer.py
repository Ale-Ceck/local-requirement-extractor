import json
import base64
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pandas as pd

from config.schema import OutputConfig
from src.data_models.requirement import Requirement, RequirementList
from src.requirement_extraction.excel_writer import ExcelWriter, write_to_excel


def test_write_to_excel_includes_traceability_columns_when_enabled():
    output_config = OutputConfig(include_metadata=True)
    requirements = RequirementList(
        [
            Requirement(
                code="REQ-001",
                description="System shall authenticate users.",
                source_document="spec.pdf",
                source_page_start=2,
                source_page_end=3,
                source_section="Security",
                source_text_excerpt="REQ-001 System shall authenticate users.",
            )
        ]
    )

    with NamedTemporaryFile(suffix=".xlsx") as temp_file:
        write_to_excel(requirements, temp_file.name, output_config)
        dataframe = pd.read_excel(temp_file.name)

    assert list(dataframe.columns) == [
        "Requirement Code",
        "Description",
        "Source Document",
        "Page Start",
        "Page End",
        "Section",
        "Source Excerpt",
        "Source Segment IDs",
        "Block IDs",
        "Review Status",
        "Confidence",
    ]
    assert dataframe.iloc[0]["Source Document"] == "spec.pdf"
    assert dataframe.iloc[0]["Page Start"] == 2


def test_write_to_excel_can_hide_traceability_columns():
    output_config = OutputConfig(include_metadata=False)
    requirements = RequirementList([Requirement(code="REQ-001", description="System shall log events.")])

    with NamedTemporaryFile(suffix=".xlsx") as temp_file:
        write_to_excel(requirements, temp_file.name, output_config)
        dataframe = pd.read_excel(temp_file.name)

    assert list(dataframe.columns) == ["Requirement Code", "Description"]


def test_excel_writer_creates_companion_review_artifact():
    output_config = OutputConfig(
        directory="unused",
        include_metadata=True,
        write_review_artifact=True,
        write_review_markdown=True,
        write_review_html=True,
    )
    requirements = RequirementList(
        [
            Requirement(
                code="REQ-010",
                description="System shall expose provenance.",
                source_segment_ids=["seg-10", "seg-11"],
                source_document="spec.pdf",
                source_chunk_id="chunk-10",
                source_page_start=5,
                source_page_end=5,
                source_block_ids=["b-1", "b-2"],
                source_section="Review",
                source_text_excerpt="REQ-010 System shall expose provenance.",
                source_regions=[
                    {
                        "segment_id": "seg-11",
                        "segment_kind": "text",
                        "block_id": "b-2",
                        "page": 4,
                        "page_number": 5,
                        "page_width": 596.0,
                        "page_height": 842.0,
                        "block_type": "text",
                        "bbox": [100.0, 200.0, 320.0, 230.0],
                        "bbox_norm": [0.16, 0.23, 0.54, 0.27],
                        "polygon_points": [[100.0, 200.0], [320.0, 200.0], [320.0, 230.0], [100.0, 230.0]],
                        "paddle_label": "text",
                        "group_id": 7,
                        "block_order": 2,
                        "source_block_ids": ["b-2"],
                        "section_path": ["Review"],
                        "text_markdown": "REQ-010 System shall expose provenance.",
                    }
                ],
            )
        ]
    )

    with TemporaryDirectory() as temp_dir:
        page_image_path = Path(temp_dir) / "page-5.png"
        ocr_page_image_path = Path(temp_dir) / "ocr-page-5.png"
        raw_bytes = b"raw-page-image"
        ocr_bytes = b"ocr-page-image"
        page_image_path.write_bytes(raw_bytes)
        ocr_page_image_path.write_bytes(ocr_bytes)
        requirements[0].source_regions[0]["page_image_path"] = str(page_image_path)
        requirements[0].source_regions[0]["ocr_page_image_path"] = str(ocr_page_image_path)
        excel_path = Path(temp_dir) / "requirements.xlsx"
        writer = ExcelWriter(output_config)

        writer.write(requirements, str(excel_path))

        artifact_path = excel_path.with_suffix(".review.json")
        markdown_path = excel_path.with_suffix(".review.md")
        html_path = excel_path.with_suffix(".review.html")
        assert artifact_path.exists()
        assert markdown_path.exists()
        assert html_path.exists()
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        markdown = markdown_path.read_text(encoding="utf-8")
        html = html_path.read_text(encoding="utf-8")

    assert payload["requirements"][0]["code"] == "REQ-010"
    assert payload["requirements"][0]["source_chunk_id"] == "chunk-10"
    assert payload["requirements"][0]["source_segment_ids"] == ["seg-10", "seg-11"]
    assert payload["requirements"][0]["source_regions"][0]["page_number"] == 5
    assert "REQ-010" in markdown
    assert "Segment IDs: seg-10, seg-11" in markdown
    assert "Page Range: 5-5" in markdown
    assert "Regions: 1" in markdown
    assert "First Region: page 5" in markdown
    assert "<svg" in html
    assert "Page 5" in html
    assert "polygon" in html
    assert f"data:image/png;base64,{base64.b64encode(raw_bytes).decode('ascii')}" in html
    assert base64.b64encode(ocr_bytes).decode("ascii") not in html
