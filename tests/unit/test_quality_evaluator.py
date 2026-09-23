import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.quality_evaluator import (
    EvaluationInputError,
    ExtractedRequirement,
    ReferenceDocument,
    ReferenceRequirement,
    RunResult,
    evaluate_quality,
    evaluate_runs,
    load_document_inventory,
    load_reference_workbook,
    load_runs,
    normalize_code,
    normalize_description,
)


def _write_reference(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_excel(path, index=False)


def _write_run(
    run_dir: Path,
    *,
    model_name: str,
    requirements: list[dict],
    command: str = "extract",
    review_filename: str = "requirements.review.json",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    review_path = run_dir / review_filename
    review_path.write_text(
        json.dumps({"requirements": requirements}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "run-manifest.json").write_text(
        json.dumps(
            {
                "command": command,
                "run_id": run_dir.name,
                "extraction": {"model_name": model_name},
                "parser": {"toc_section_pruning_mode": "enforce"},
                "chunking": {"max_chunk_chars": 4000},
                "parallel": {"enabled": False, "max_workers": 1},
                "run_stats": {
                    "status": "completed",
                    "duration_seconds": 12.5,
                    "json_parse_failures": 1,
                    "llm_response_failures": 0,
                    "chunk_processing_errors": 0,
                    "empty_chunk_results": 2,
                },
                "artifacts": {
                    "output_directory": str(run_dir),
                    "exports": {
                        "requirements_review_json": str(review_path),
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_reference_loader_reads_and_normalizes_required_columns(tmp_path: Path) -> None:
    reference_path = tmp_path / "doc-a.xlsx"
    _write_reference(
        reference_path,
        [
            {"CODE": " req-1 ", "DESCRIPTIONS": " First\nrequirement "},
            {"CODE": "REQ-2", "DESCRIPTIONS": "Second requirement"},
        ],
    )

    document = load_reference_workbook(
        reference_path,
        document_id="doc-a",
        pdf_path="doc-a.pdf",
        notes=None,
    )

    assert [requirement.code for requirement in document.requirements] == [
        "REQ-1",
        "REQ-2",
    ]
    assert document.requirements[0].description == "First requirement"
    assert document.duplicate_codes == []
    assert normalize_code(" req-1 ") == "REQ-1"
    assert normalize_description("A\n\nB\tC") == "A B C"


def test_reference_loader_rejects_duplicate_codes_and_empty_descriptions(
    tmp_path: Path,
) -> None:
    duplicate_path = tmp_path / "duplicates.xlsx"
    _write_reference(
        duplicate_path,
        [
            {"CODE": "REQ-1", "DESCRIPTIONS": "First"},
            {"CODE": " req-1 ", "DESCRIPTIONS": "Duplicate"},
        ],
    )
    with pytest.raises(EvaluationInputError, match="duplicate requirement codes"):
        load_reference_workbook(
            duplicate_path, document_id="doc-a", pdf_path=None, notes=None
        )

    empty_description_path = tmp_path / "empty-description.xlsx"
    _write_reference(
        empty_description_path,
        [{"CODE": "REQ-1", "DESCRIPTIONS": "  "}],
    )
    with pytest.raises(EvaluationInputError, match="empty description"):
        load_reference_workbook(
            empty_description_path, document_id="doc-a", pdf_path=None, notes=None
        )


def test_document_inventory_rejects_empty_and_duplicate_document_ids(
    tmp_path: Path,
) -> None:
    empty_inventory = tmp_path / "empty.csv"
    empty_inventory.write_text(
        "document_id,pdf_path,reference_xlsx,notes\n", encoding="utf-8"
    )
    with pytest.raises(EvaluationInputError, match="does not contain any documents"):
        load_document_inventory(empty_inventory)

    duplicate_inventory = tmp_path / "duplicates.csv"
    duplicate_inventory.write_text(
        "\n".join(
            [
                "document_id,pdf_path,reference_xlsx,notes",
                "DOC-A,doc-a.pdf,doc-a.xlsx,first",
                "DOC-A,doc-b.pdf,doc-b.xlsx,duplicate",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationInputError, match="duplicate document_id"):
        load_document_inventory(duplicate_inventory)


def test_evaluate_quality_writes_reports_and_scores_strict_quality(
    tmp_path: Path,
) -> None:
    references_dir = tmp_path / "references"
    runs_dir = tmp_path / "runs"
    output_dir = tmp_path / "quality"
    references_dir.mkdir()
    reference_path = references_dir / "doc-a.xlsx"
    _write_reference(
        reference_path,
        [
            {"CODE": "REQ-1", "DESCRIPTIONS": "First requirement."},
            {"CODE": "REQ-2", "DESCRIPTIONS": "Second requirement."},
        ],
    )
    _write_run(
        runs_dir / "run-a",
        model_name="test-model",
        requirements=[
            {
                "code": "REQ-1",
                "description": "Incomplete requirement.",
                "source_document": "doc-a.pdf",
                "source_segment_ids": ["seg-1"],
            },
            {
                "code": "REQ-3",
                "description": "Unexpected requirement.",
                "source_document": "doc-a.pdf",
                "source_segment_ids": [],
                "review_status": "needs_source_segment_review",
            },
            {
                "code": "REQ-3",
                "description": "Unexpected requirement duplicate.",
                "source_document": "doc-a.pdf",
                "source_segment_ids": ["seg-4"],
            },
        ],
    )

    result = evaluate_quality(
        references_dir=references_dir,
        runs_dir=runs_dir,
        output_dir=output_dir,
    )

    summary = result.summary_rows[0]
    assert summary["true_positive_count"] == 1
    assert summary["false_positive_count"] == 2
    assert summary["false_negative_count"] == 1
    assert round(summary["code_f1"], 3) == 0.4
    assert summary["description_exact_match_count"] == 0
    assert summary["description_exact_rate"] == 0.0
    assert summary["strict_f1"] == 0.0
    assert summary["citation_coverage"] == pytest.approx(2 / 3)
    assert summary["provenance_review_count"] == 1
    assert summary["duration_seconds"] == 12.5
    assert summary["json_parse_failures"] == 1
    assert result.duplicate_rows == [
        {
            "run_id": "run-a",
            "model_name": "test-model",
            "document_id": "doc-a",
            "source": "extraction",
            "code": "REQ-3",
        }
    ]
    assert (output_dir / "quality-summary.md").exists()
    assert (output_dir / "quality-analysis.xlsx").exists()
    assert (output_dir / "quality-analysis.json").exists()
    json_report = json.loads(
        (output_dir / "quality-analysis.json").read_text(encoding="utf-8")
    )
    assert json_report["metadata"]["policy_version"] == "2026-09-17.1"
    assert len(json_report["metadata"]["references"][0]["sha256"]) == 64
    assert "strict_metrics" in json_report["policy"]
    assert {"policy", "metadata", "summary", "matched"}.issubset(
        pd.ExcelFile(output_dir / "quality-analysis.xlsx").sheet_names
    )


def test_evaluate_runs_rejects_unmapped_documents() -> None:
    references = {
        "doc-a": ReferenceDocument(
            document_id="doc-a",
            reference_path="doc-a.xlsx",
            pdf_path="doc-a.pdf",
            notes=None,
            requirements=[
                ReferenceRequirement(code="REQ-1", description="First", row_number=2)
            ],
        )
    }
    extracted = ExtractedRequirement(
        code="REQ-X",
        description="Unexpected",
        source_document="doc-b.pdf",
        source_page_start=1,
        source_page_end=1,
        source_section=None,
        source_segment_ids=["seg-x"],
        review_status=None,
    )
    run = RunResult(
        run_id="run-a",
        run_dir=Path("run-a"),
        manifest={"extraction": {"model_name": "test-model"}},
        requirements_by_document={"doc-b": [extracted]},
    )

    with pytest.raises(EvaluationInputError, match="unmapped document IDs.*doc-b"):
        evaluate_runs(references=references, runs=[run])


def test_load_runs_skips_prepare_runs_and_prefers_local_artifacts(
    tmp_path: Path,
) -> None:
    original_dir = tmp_path / "original"
    _write_run(
        original_dir,
        model_name="test-model",
        requirements=[
            {
                "code": "WRONG",
                "description": "Stale artifact",
                "source_document": "doc-a.pdf",
                "source_segment_ids": ["seg-stale"],
            }
        ],
    )

    runs_dir = tmp_path / "runs"
    copied_run = runs_dir / "copied-run"
    _write_run(
        copied_run,
        model_name="test-model",
        requirements=[
            {
                "code": "REQ-1",
                "description": "Local artifact",
                "source_document": "doc-a.pdf",
                "source_segment_ids": ["seg-local"],
            }
        ],
    )
    manifest_path = copied_run / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["exports"]["requirements_review_json"] = str(
        original_dir / "requirements.review.json"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    prepare_dir = runs_dir / "prepared-only"
    prepare_dir.mkdir(parents=True)
    (prepare_dir / "run-manifest.json").write_text(
        json.dumps({"command": "prepare-pdf", "run_id": "prepared-only"}),
        encoding="utf-8",
    )

    runs = load_runs(runs_dir, document_key_map={"doc-a": "doc-a"})

    assert [run.run_id for run in runs] == ["copied-run"]
    assert runs[0].requirements_by_document["doc-a"][0].code == "REQ-1"


@pytest.mark.skipif(
    not Path("data/test/example_reference.xlsx").is_file()
    or not Path(
        "data/test/output/examples-replay-20260415-gemma4-e4b/run-manifest.json"
    ).is_file(),
    reason="Optional historical run and workbook are local data, not repository fixtures.",
)
def test_existing_gemma4_example_matches_reference_code_set(tmp_path: Path) -> None:
    output_dir = tmp_path / "quality"
    inventory_path = tmp_path / "inventory.csv"
    inventory_path.write_text(
        "\n".join(
            [
                "document_id,pdf_path,reference_xlsx,notes",
                "examples,data/test/examples.pdf,data/test/example_reference.xlsx,existing fixture",
            ]
        ),
        encoding="utf-8",
    )

    result = evaluate_quality(
        references_dir="data/test",
        runs_dir="data/test/output/examples-replay-20260415-gemma4-e4b",
        output_dir=output_dir,
        document_inventory=inventory_path,
    )

    summary = result.summary_rows[0]
    assert summary["model_name"] == "gemma4:e4b"
    assert summary["reference_count"] == 22
    assert summary["extracted_count"] == 22
    assert summary["true_positive_count"] == 22
    assert summary["false_positive_count"] == 0
    assert summary["false_negative_count"] == 0
    assert summary["exact_description_mismatch_count"] == 4
    assert summary["code_f1"] == 1.0
    assert summary["description_exact_match_count"] == 18
    assert summary["description_exact_rate"] == pytest.approx(18 / 22)
    assert summary["strict_f1"] == pytest.approx(18 / 22)
