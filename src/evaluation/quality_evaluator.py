from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.reproducibility import (
    collect_file_metadata,
    collect_git_metadata,
    collect_runtime_metadata,
)

REFERENCE_CODE_COLUMN = "CODE"
REFERENCE_DESCRIPTION_COLUMN = "DESCRIPTIONS"
EVALUATION_POLICY_VERSION = "2026-09-17.1"
PROVENANCE_REVIEW_STATUSES = {
    "partial_invalid_source_segment_ids",
    "invalid_source_segment_ids",
    "uncited_source_segments",
    "needs_source_segment_review",
}
EVALUATION_POLICY: dict[str, str] = {
    "reference_codes": "Codes must be non-empty and unique within each reference document.",
    "reference_descriptions": (
        "Descriptions must be non-empty. Exact matching preserves case and punctuation after whitespace normalization."
    ),
    "extracted_duplicates": "Every extracted occurrence after the first normalized code is counted as a false positive.",
    "code_metrics": "Code precision, recall, and F1 compare normalized code identity only.",
    "strict_metrics": (
        "Strict true positives require both a matching normalized code and an exact normalized description. "
        "Description mismatches count as one strict false positive and one strict false negative."
    ),
    "document_mapping": "Every extracted requirement must map to exactly one inventory document ID.",
}


class EvaluationInputError(ValueError):
    """Raised when evaluation inputs are incomplete, ambiguous, or inconsistent."""


@dataclass
class ReferenceRequirement:
    code: str
    description: str
    row_number: int


@dataclass
class ReferenceDocument:
    document_id: str
    reference_path: str
    pdf_path: str | None
    notes: str | None
    requirements: list[ReferenceRequirement]
    duplicate_codes: list[str] = field(default_factory=list)


@dataclass
class ExtractedRequirement:
    code: str
    description: str
    source_document: str | None
    source_page_start: int | None
    source_page_end: int | None
    source_section: str | None
    source_segment_ids: list[str]
    review_status: str | None


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    manifest: dict[str, Any]
    requirements_by_document: dict[str, list[ExtractedRequirement]]


@dataclass
class DocumentEvaluation:
    run_id: str
    model_name: str | None
    document_id: str
    reference_path: str
    extracted_count: int
    reference_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    precision: float
    recall: float
    f1: float
    strict_true_positive_count: int
    strict_false_positive_count: int
    strict_false_negative_count: int
    strict_precision: float
    strict_recall: float
    strict_f1: float
    description_exact_match_count: int
    exact_description_mismatch_count: int
    description_exact_rate: float
    average_description_similarity: float | None
    cited_extracted_count: int
    uncited_extracted_count: int
    citation_coverage: float
    provenance_review_count: int
    provenance_review_rate: float
    reference_duplicate_codes: list[str]
    extracted_duplicate_codes: list[str]


@dataclass
class EvaluationResult:
    summary_rows: list[dict[str, Any]]
    document_rows: list[dict[str, Any]]
    missing_rows: list[dict[str, Any]]
    unexpected_rows: list[dict[str, Any]]
    mismatch_rows: list[dict[str, Any]]
    duplicate_rows: list[dict[str, Any]]
    matched_rows: list[dict[str, Any]]
    policy: dict[str, str] = field(default_factory=lambda: dict(EVALUATION_POLICY))
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_quality(
    *,
    references_dir: str | Path,
    runs_dir: str | Path,
    output_dir: str | Path,
    document_inventory: str | Path | None = None,
) -> EvaluationResult:
    references_path = Path(references_dir)
    runs_path = Path(runs_dir)
    output_path = Path(output_dir)

    inventory = (
        load_document_inventory(document_inventory) if document_inventory else None
    )
    references, document_key_map = load_reference_documents(
        references_path, inventory=inventory
    )
    runs = load_runs(runs_path, document_key_map=document_key_map)

    result = evaluate_runs(references=references, runs=runs)
    repo_root = Path(__file__).resolve().parents[2]
    result.metadata = {
        "policy_version": EVALUATION_POLICY_VERSION,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "git": collect_git_metadata(repo_root),
        "runtime": collect_runtime_metadata(),
        "references": collect_file_metadata(
            document.reference_path for document in references.values()
        ),
        "run_manifests": collect_file_metadata(
            run.run_dir / "run-manifest.json" for run in runs
        ),
    }
    write_evaluation_outputs(result, output_path)
    return result


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value or "").strip().upper()


def normalize_description(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value or "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_document_key(value: str | Path | None) -> str:
    if value is None:
        return ""
    stem = Path(str(value)).stem
    normalized = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return normalized


def description_similarity(reference: str, extracted: str) -> float:
    return SequenceMatcher(
        None, normalize_description(reference), normalize_description(extracted)
    ).ratio()


def load_document_inventory(path: str | Path | None) -> list[dict[str, str]]:
    """Load and validate the explicit document-to-reference mapping."""
    if path is None:
        return []
    inventory_path = Path(path)
    if not inventory_path.is_file():
        raise EvaluationInputError(f"Document inventory not found: {inventory_path}")
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"document_id", "pdf_path", "reference_xlsx"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise EvaluationInputError(
                f"Document inventory is missing required columns: {sorted(missing)}"
            )
        rows: list[dict[str, str]] = []
        seen_document_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            document_id = (row.get("document_id") or "").strip()
            reference_xlsx = (row.get("reference_xlsx") or "").strip()
            pdf_path = (row.get("pdf_path") or "").strip()
            notes = (row.get("notes") or "").strip()
            if not any((document_id, reference_xlsx, pdf_path, notes)):
                continue
            if not document_id or not reference_xlsx:
                raise EvaluationInputError(
                    f"Document inventory row {row_number} requires non-empty document_id and reference_xlsx."
                )
            if document_id in seen_document_ids:
                raise EvaluationInputError(
                    f"Document inventory contains duplicate document_id: {document_id}"
                )
            seen_document_ids.add(document_id)
            rows.append(
                {
                    "document_id": document_id,
                    "pdf_path": pdf_path,
                    "reference_xlsx": reference_xlsx,
                    "notes": notes,
                    "_inventory_dir": str(inventory_path.parent),
                }
            )
    if not rows:
        raise EvaluationInputError(
            f"Document inventory does not contain any documents: {inventory_path}"
        )
    return rows


def load_reference_documents(
    references_dir: str | Path,
    *,
    inventory: list[dict[str, str]] | None = None,
) -> tuple[dict[str, ReferenceDocument], dict[str, str]]:
    """Load validated reference workbooks and build an unambiguous key map."""
    references_path = Path(references_dir)
    documents: dict[str, ReferenceDocument] = {}
    document_key_map: dict[str, str] = {}

    if inventory is not None:
        for row in inventory:
            reference_path = _resolve_inventory_path(
                row["reference_xlsx"], row["_inventory_dir"], references_path
            )
            document = load_reference_workbook(
                reference_path,
                document_id=row["document_id"],
                pdf_path=row.get("pdf_path") or None,
                notes=row.get("notes") or None,
            )
            if document.document_id in documents:
                raise EvaluationInputError(
                    f"Duplicate reference document_id: {document.document_id}"
                )
            documents[document.document_id] = document
            _register_document_keys(document_key_map, document)
        return documents, document_key_map

    workbook_paths = sorted(
        path
        for path in references_path.rglob("*.xlsx")
        if not path.name.startswith("~$")
    )
    for workbook_path in workbook_paths:
        document = load_reference_workbook(
            workbook_path,
            document_id=normalize_document_key(workbook_path),
            pdf_path=None,
            notes=None,
        )
        if document.document_id in documents:
            raise EvaluationInputError(
                f"Reference workbooks resolve to the same document_id {document.document_id!r}; use an inventory."
            )
        documents[document.document_id] = document
        _register_document_keys(document_key_map, document)

    if not documents:
        raise EvaluationInputError(
            f"No reference workbooks found under {references_path}"
        )
    return documents, document_key_map


def load_reference_workbook(
    path: str | Path,
    *,
    document_id: str,
    pdf_path: str | None,
    notes: str | None,
) -> ReferenceDocument:
    """Load one gold reference workbook under the strict evaluation policy."""
    workbook_path = Path(path)
    if not workbook_path.is_file():
        raise EvaluationInputError(f"Reference workbook not found: {workbook_path}")
    dataframe = pd.read_excel(workbook_path, sheet_name=0)
    missing_columns = [
        column
        for column in (REFERENCE_CODE_COLUMN, REFERENCE_DESCRIPTION_COLUMN)
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise EvaluationInputError(
            f"{workbook_path} is missing required columns: {missing_columns}"
        )

    requirements: list[ReferenceRequirement] = []
    code_rows: dict[str, list[int]] = {}
    for index, row in dataframe.iterrows():
        row_number = index + 2
        code = normalize_code(row[REFERENCE_CODE_COLUMN])
        description = normalize_description(row[REFERENCE_DESCRIPTION_COLUMN])
        if not code and not description:
            continue
        if not code:
            raise EvaluationInputError(
                f"{workbook_path} row {row_number} has a description but no requirement code."
            )
        if not description:
            raise EvaluationInputError(
                f"{workbook_path} row {row_number} has an empty description for code {code}."
            )
        requirements.append(
            ReferenceRequirement(
                code=code,
                description=description,
                row_number=row_number,
            )
        )
        code_rows.setdefault(code, []).append(row_number)

    duplicate_codes = sorted(code for code, rows in code_rows.items() if len(rows) > 1)
    if duplicate_codes:
        locations = {code: code_rows[code] for code in duplicate_codes}
        raise EvaluationInputError(
            f"{workbook_path} contains duplicate requirement codes after normalization: {locations}"
        )
    if not requirements:
        raise EvaluationInputError(
            f"Reference workbook contains no requirements: {workbook_path}"
        )
    return ReferenceDocument(
        document_id=document_id,
        reference_path=str(workbook_path),
        pdf_path=pdf_path,
        notes=notes,
        requirements=requirements,
        duplicate_codes=duplicate_codes,
    )


def load_runs(
    runs_dir: str | Path,
    *,
    document_key_map: dict[str, str],
) -> list[RunResult]:
    """Load completed extraction runs while excluding preparation-only manifests."""
    runs_path = Path(runs_dir)
    manifest_paths = (
        [runs_path / "run-manifest.json"]
        if (runs_path / "run-manifest.json").exists()
        else []
    )
    if not manifest_paths:
        manifest_paths = sorted(runs_path.rglob("run-manifest.json"))
    if not manifest_paths:
        raise EvaluationInputError(
            f"No run-manifest.json files found under {runs_path}"
        )

    runs: list[RunResult] = []
    seen_run_ids: set[str] = set()
    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        command = manifest.get("command")
        if command == "prepare-pdf":
            continue
        if command not in (None, "extract"):
            continue
        run = load_run(
            manifest_path.parent, document_key_map=document_key_map, manifest=manifest
        )
        if run.run_id in seen_run_ids:
            raise EvaluationInputError(
                f"Duplicate run_id {run.run_id!r} under {runs_path}"
            )
        seen_run_ids.add(run.run_id)
        runs.append(run)

    if not runs:
        raise EvaluationInputError(
            f"No completed extraction runs found under {runs_path}"
        )
    return runs


def load_run(
    run_dir: str | Path,
    *,
    document_key_map: dict[str, str],
    manifest: dict[str, Any] | None = None,
) -> RunResult:
    """Load and validate a self-contained extraction run."""
    path = Path(run_dir)
    manifest_path = path / "run-manifest.json"
    manifest = manifest or _read_json(manifest_path)
    run_stats = _load_run_stats(path, manifest)
    status = run_stats.get("status") or manifest.get("execution", {}).get("status")
    if status not in (None, "completed"):
        raise EvaluationInputError(f"Run {path} is not completed; status={status!r}.")
    review_path = _resolve_review_artifact_path(path, manifest)
    review_payload = _read_json(review_path)
    raw_requirements = review_payload.get("requirements")
    if not isinstance(raw_requirements, list):
        raise EvaluationInputError(
            f"{review_path} must contain a list field named 'requirements'."
        )
    requirements_by_document: dict[str, list[ExtractedRequirement]] = {}

    for item_index, payload in enumerate(raw_requirements, start=1):
        if not isinstance(payload, dict):
            raise EvaluationInputError(
                f"{review_path} requirement #{item_index} must be a JSON object."
            )
        code = normalize_code(payload.get("code"))
        if not code:
            raise EvaluationInputError(
                f"{review_path} requirement #{item_index} has an empty code."
            )
        description = _require_string(
            payload.get("description"), "description", review_path, item_index
        )
        source_document = payload.get("source_document")
        if not isinstance(source_document, str) or not source_document.strip():
            raise EvaluationInputError(
                f"{review_path} requirement #{item_index} has no source_document; document mapping is required."
            )
        document_key = normalize_document_key(source_document)
        document_id = document_key_map.get(document_key, document_key)
        raw_segment_ids = payload.get("source_segment_ids") or []
        if not isinstance(raw_segment_ids, list) or not all(
            isinstance(value, str) for value in raw_segment_ids
        ):
            raise EvaluationInputError(
                f"{review_path} requirement #{item_index} source_segment_ids must be a list of strings."
            )
        review_status = payload.get("review_status")
        if review_status is not None and not isinstance(review_status, str):
            raise EvaluationInputError(
                f"{review_path} requirement #{item_index} review_status must be a string or null."
            )
        requirements_by_document.setdefault(document_id, []).append(
            ExtractedRequirement(
                code=code,
                description=normalize_description(description),
                source_document=source_document.strip(),
                source_page_start=payload.get("source_page_start"),
                source_page_end=payload.get("source_page_end"),
                source_section=payload.get("source_section"),
                source_segment_ids=[
                    value.strip() for value in raw_segment_ids if value.strip()
                ],
                review_status=(
                    review_status.strip()
                    if isinstance(review_status, str) and review_status.strip()
                    else None
                ),
            )
        )

    manifest_run_id = manifest.get("run_id")
    run_id = (
        manifest_run_id.strip()
        if isinstance(manifest_run_id, str) and manifest_run_id.strip()
        else path.name
    )
    return RunResult(
        run_id=run_id,
        run_dir=path,
        manifest=manifest,
        requirements_by_document=requirements_by_document,
    )


def evaluate_runs(
    *,
    references: dict[str, ReferenceDocument],
    runs: Iterable[RunResult],
) -> EvaluationResult:
    """Evaluate runs according to the explicit code, description, and provenance policy."""
    summary_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    unexpected_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []

    for run in runs:
        model_name = run.manifest.get("extraction", {}).get("model_name")
        unmapped_document_ids = sorted(
            set(run.requirements_by_document) - set(references)
        )
        if unmapped_document_ids:
            raise EvaluationInputError(
                f"Run {run.run_id!r} contains unmapped document IDs: {unmapped_document_ids}. "
                "Add them to the document inventory or correct source_document metadata."
            )
        run_totals = {
            "true_positive_count": 0,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "reference_count": 0,
            "extracted_count": 0,
            "strict_true_positive_count": 0,
            "strict_false_positive_count": 0,
            "strict_false_negative_count": 0,
            "description_exact_match_count": 0,
            "exact_description_mismatch_count": 0,
            "cited_extracted_count": 0,
            "uncited_extracted_count": 0,
            "provenance_review_count": 0,
        }
        all_similarities: list[float] = []

        for document_id, reference in sorted(references.items()):
            extracted_requirements = run.requirements_by_document.get(document_id, [])
            doc_result = evaluate_document(
                run_id=run.run_id,
                model_name=model_name,
                reference=reference,
                extracted_requirements=extracted_requirements,
            )
            document_rows.append(_document_result_row(doc_result))

            for key in run_totals:
                run_totals[key] += getattr(doc_result, key)
            if doc_result.average_description_similarity is not None:
                doc_matched_count = doc_result.true_positive_count
                all_similarities.extend(
                    [doc_result.average_description_similarity] * doc_matched_count
                )

            _append_duplicate_rows(duplicate_rows, doc_result)
            _append_detail_rows(
                missing_rows=missing_rows,
                unexpected_rows=unexpected_rows,
                mismatch_rows=mismatch_rows,
                matched_rows=matched_rows,
                run=run,
                model_name=model_name,
                reference=reference,
                extracted_requirements=extracted_requirements,
            )

        precision, recall, f1 = _precision_recall_f1(
            run_totals["true_positive_count"],
            run_totals["false_positive_count"],
            run_totals["false_negative_count"],
        )
        strict_precision, strict_recall, strict_f1 = _precision_recall_f1(
            run_totals["strict_true_positive_count"],
            run_totals["strict_false_positive_count"],
            run_totals["strict_false_negative_count"],
        )
        description_exact_rate = _safe_ratio(
            run_totals["description_exact_match_count"],
            run_totals["true_positive_count"],
        )
        citation_coverage = _safe_ratio(
            run_totals["cited_extracted_count"],
            run_totals["extracted_count"],
        )
        provenance_review_rate = _safe_ratio(
            run_totals["provenance_review_count"],
            run_totals["extracted_count"],
        )
        run_stats = _load_run_stats(run.run_dir, run.manifest)
        summary_rows.append(
            {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "model_name": model_name,
                "document_count": len(references),
                **run_totals,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "code_precision": precision,
                "code_recall": recall,
                "code_f1": f1,
                "strict_precision": strict_precision,
                "strict_recall": strict_recall,
                "strict_f1": strict_f1,
                "description_exact_rate": description_exact_rate,
                "average_description_similarity": _average(all_similarities),
                "citation_coverage": citation_coverage,
                "provenance_review_rate": provenance_review_rate,
                "toc_section_pruning_mode": run.manifest.get("parser", {}).get(
                    "toc_section_pruning_mode"
                ),
                "chunking_max_chunk_chars": run.manifest.get("chunking", {}).get(
                    "max_chunk_chars"
                ),
                "parallel_enabled": run.manifest.get("parallel", {}).get("enabled"),
                "parallel_max_workers": run.manifest.get("parallel", {}).get(
                    "max_workers"
                ),
                "run_status": run_stats.get("status"),
                "duration_seconds": run_stats.get("duration_seconds"),
                "json_parse_failures": int(
                    run_stats.get("json_parse_failures", 0) or 0
                ),
                "llm_response_failures": int(
                    run_stats.get("llm_response_failures", 0) or 0
                ),
                "chunk_processing_errors": int(
                    run_stats.get("chunk_processing_errors", 0) or 0
                ),
                "empty_chunk_results": int(
                    run_stats.get("empty_chunk_results", 0) or 0
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            row["strict_f1"],
            row["code_f1"],
            row["description_exact_rate"],
            row["code_recall"],
            -row["false_positive_count"],
            -row["provenance_review_count"],
        ),
        reverse=True,
    )
    return EvaluationResult(
        summary_rows=summary_rows,
        document_rows=document_rows,
        missing_rows=missing_rows,
        unexpected_rows=unexpected_rows,
        mismatch_rows=mismatch_rows,
        duplicate_rows=duplicate_rows,
        matched_rows=matched_rows,
    )


def evaluate_document(
    *,
    run_id: str,
    model_name: str | None,
    reference: ReferenceDocument,
    extracted_requirements: list[ExtractedRequirement],
) -> DocumentEvaluation:
    """Evaluate one document with occurrence-aware duplicate penalties."""
    reference_by_code = _first_by_code(reference.requirements)
    extracted_by_code = _first_by_code(extracted_requirements)
    reference_codes = set(reference_by_code)
    extracted_codes = set(extracted_by_code)
    matched_codes = sorted(reference_codes & extracted_codes)
    unexpected_codes = extracted_codes - reference_codes
    missing_codes = reference_codes - extracted_codes
    duplicate_occurrence_count = len(extracted_requirements) - len(extracted_by_code)
    similarities = [
        description_similarity(
            reference_by_code[code].description, extracted_by_code[code].description
        )
        for code in matched_codes
    ]
    exact_mismatches = [
        code
        for code in matched_codes
        if normalize_description(reference_by_code[code].description)
        != normalize_description(extracted_by_code[code].description)
    ]
    exact_match_count = len(matched_codes) - len(exact_mismatches)
    false_positive_count = len(unexpected_codes) + duplicate_occurrence_count
    precision, recall, f1 = _precision_recall_f1(
        len(matched_codes),
        false_positive_count,
        len(missing_codes),
    )
    strict_true_positive_count = exact_match_count
    strict_false_positive_count = false_positive_count + len(exact_mismatches)
    strict_false_negative_count = len(missing_codes) + len(exact_mismatches)
    strict_precision, strict_recall, strict_f1 = _precision_recall_f1(
        strict_true_positive_count,
        strict_false_positive_count,
        strict_false_negative_count,
    )
    cited_extracted_count = sum(
        bool(requirement.source_segment_ids) for requirement in extracted_requirements
    )
    extracted_count = len(extracted_requirements)
    provenance_review_count = sum(
        requirement.review_status in PROVENANCE_REVIEW_STATUSES
        for requirement in extracted_requirements
    )
    return DocumentEvaluation(
        run_id=run_id,
        model_name=model_name,
        document_id=reference.document_id,
        reference_path=reference.reference_path,
        extracted_count=extracted_count,
        reference_count=len(reference.requirements),
        true_positive_count=len(matched_codes),
        false_positive_count=false_positive_count,
        false_negative_count=len(missing_codes),
        precision=precision,
        recall=recall,
        f1=f1,
        strict_true_positive_count=strict_true_positive_count,
        strict_false_positive_count=strict_false_positive_count,
        strict_false_negative_count=strict_false_negative_count,
        strict_precision=strict_precision,
        strict_recall=strict_recall,
        strict_f1=strict_f1,
        description_exact_match_count=exact_match_count,
        exact_description_mismatch_count=len(exact_mismatches),
        description_exact_rate=_safe_ratio(exact_match_count, len(matched_codes)),
        average_description_similarity=_average(similarities),
        cited_extracted_count=cited_extracted_count,
        uncited_extracted_count=extracted_count - cited_extracted_count,
        citation_coverage=_safe_ratio(cited_extracted_count, extracted_count),
        provenance_review_count=provenance_review_count,
        provenance_review_rate=_safe_ratio(provenance_review_count, extracted_count),
        reference_duplicate_codes=reference.duplicate_codes,
        extracted_duplicate_codes=_duplicate_codes(extracted_requirements),
    )


def write_evaluation_outputs(result: EvaluationResult, output_dir: str | Path) -> None:
    """Write machine-readable and human-readable evaluation reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "quality-analysis.json").write_text(
        json.dumps(
            {
                "metadata": result.metadata,
                "policy": result.policy,
                "summary": result.summary_rows,
                "by_document": result.document_rows,
                "missing": result.missing_rows,
                "unexpected": result.unexpected_rows,
                "description_mismatches": result.mismatch_rows,
                "duplicates": result.duplicate_rows,
                "matched": result.matched_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_path / "quality-summary.md").write_text(
        _render_markdown_summary(result), encoding="utf-8"
    )
    with pd.ExcelWriter(output_path / "quality-analysis.xlsx") as writer:
        _write_sheet(
            writer,
            "policy",
            [
                {"rule": key, "definition": value}
                for key, value in result.policy.items()
            ],
        )
        _write_sheet(writer, "metadata", _flatten_metadata_rows(result.metadata))
        _write_sheet(writer, "summary", result.summary_rows)
        _write_sheet(writer, "by_document", result.document_rows)
        _write_sheet(writer, "missing", result.missing_rows)
        _write_sheet(writer, "unexpected", result.unexpected_rows)
        _write_sheet(writer, "desc_mismatches", result.mismatch_rows)
        _write_sheet(writer, "duplicates", result.duplicate_rows)
        _write_sheet(writer, "matched", result.matched_rows)


def _append_detail_rows(
    *,
    missing_rows: list[dict[str, Any]],
    unexpected_rows: list[dict[str, Any]],
    mismatch_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    run: RunResult,
    model_name: str | None,
    reference: ReferenceDocument,
    extracted_requirements: list[ExtractedRequirement],
) -> None:
    reference_by_code = _first_by_code(reference.requirements)
    extracted_by_code = _first_by_code(extracted_requirements)
    reference_codes = set(reference_by_code)
    extracted_codes = set(extracted_by_code)

    for code in sorted(reference_codes - extracted_codes):
        missing_rows.append(
            {
                "run_id": run.run_id,
                "model_name": model_name,
                "document_id": reference.document_id,
                "code": code,
                "reference_description": reference_by_code[code].description,
            }
        )
    for code in sorted(extracted_codes - reference_codes):
        extracted = extracted_by_code[code]
        unexpected_rows.append(
            {
                "run_id": run.run_id,
                "model_name": model_name,
                "document_id": reference.document_id,
                "code": code,
                "extracted_description": extracted.description,
                "source_document": extracted.source_document,
                "page_start": extracted.source_page_start,
                "page_end": extracted.source_page_end,
                "section": extracted.source_section,
            }
        )
    for code in sorted(reference_codes & extracted_codes):
        reference_requirement = reference_by_code[code]
        extracted = extracted_by_code[code]
        similarity = description_similarity(
            reference_requirement.description, extracted.description
        )
        matched_row = {
            "run_id": run.run_id,
            "model_name": model_name,
            "document_id": reference.document_id,
            "code": code,
            "description_exact_match": normalize_description(
                reference_requirement.description
            )
            == normalize_description(extracted.description),
            "description_similarity": similarity,
            "reference_description": reference_requirement.description,
            "extracted_description": extracted.description,
            "source_document": extracted.source_document,
            "page_start": extracted.source_page_start,
            "page_end": extracted.source_page_end,
            "section": extracted.source_section,
            "source_segment_ids": ", ".join(extracted.source_segment_ids),
            "review_status": extracted.review_status,
        }
        matched_rows.append(matched_row)
        if not matched_row["description_exact_match"]:
            mismatch_rows.append(matched_row)


def _append_duplicate_rows(
    rows: list[dict[str, Any]], result: DocumentEvaluation
) -> None:
    for source, codes in (
        ("reference", result.reference_duplicate_codes),
        ("extraction", result.extracted_duplicate_codes),
    ):
        for code in codes:
            rows.append(
                {
                    "run_id": result.run_id,
                    "model_name": result.model_name,
                    "document_id": result.document_id,
                    "source": source,
                    "code": code,
                }
            )


def _document_result_row(result: DocumentEvaluation) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "model_name": result.model_name,
        "document_id": result.document_id,
        "reference_path": result.reference_path,
        "reference_count": result.reference_count,
        "extracted_count": result.extracted_count,
        "true_positive_count": result.true_positive_count,
        "false_positive_count": result.false_positive_count,
        "false_negative_count": result.false_negative_count,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "code_precision": result.precision,
        "code_recall": result.recall,
        "code_f1": result.f1,
        "strict_true_positive_count": result.strict_true_positive_count,
        "strict_false_positive_count": result.strict_false_positive_count,
        "strict_false_negative_count": result.strict_false_negative_count,
        "strict_precision": result.strict_precision,
        "strict_recall": result.strict_recall,
        "strict_f1": result.strict_f1,
        "description_exact_match_count": result.description_exact_match_count,
        "exact_description_mismatch_count": result.exact_description_mismatch_count,
        "description_exact_rate": result.description_exact_rate,
        "average_description_similarity": result.average_description_similarity,
        "cited_extracted_count": result.cited_extracted_count,
        "uncited_extracted_count": result.uncited_extracted_count,
        "citation_coverage": result.citation_coverage,
        "provenance_review_count": result.provenance_review_count,
        "provenance_review_rate": result.provenance_review_rate,
        "reference_duplicate_codes": ", ".join(result.reference_duplicate_codes),
        "extracted_duplicate_codes": ", ".join(result.extracted_duplicate_codes),
    }


def _render_markdown_summary(result: EvaluationResult) -> str:
    lines = [
        "# Extraction Quality Summary",
        "",
        f"Evaluation policy: `{result.metadata.get('policy_version', EVALUATION_POLICY_VERSION)}`",
        "",
    ]
    if result.summary_rows:
        best = result.summary_rows[0]
        lines.extend(
            [
                f"Best run: `{best['run_id']}`",
                f"Primary metric: strict code-and-description F1 = `{best['strict_f1']:.3f}`",
                f"Code-level F1 = `{best['code_f1']:.3f}`",
                "",
            ]
        )
    lines.extend(["## Evaluation Policy", ""])
    for key, value in result.policy.items():
        lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    lines.extend(["", "## Run Results", ""])
    lines.extend(
        [
            "| Run | Model | Docs | TP | FP | FN | Code F1 | Strict F1 | Exact Desc. Rate | Citation Coverage | Review Flags |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.summary_rows:
        lines.append(
            "| {run_id} | {model_name} | {document_count} | {tp} | {fp} | {fn} | {code_f1:.3f} | {strict_f1:.3f} | {exact_rate:.3f} | {citation:.3f} | {review_count} |".format(
                run_id=row["run_id"],
                model_name=row.get("model_name") or "",
                document_count=row["document_count"],
                tp=row["true_positive_count"],
                fp=row["false_positive_count"],
                fn=row["false_negative_count"],
                code_f1=row["code_f1"],
                strict_f1=row["strict_f1"],
                exact_rate=row["description_exact_rate"],
                citation=row["citation_coverage"],
                review_count=row["provenance_review_count"],
            )
        )
    lines.extend(["", "## Document Results", ""])
    lines.extend(
        [
            "| Run | Document | TP | FP | FN | Code F1 | Strict F1 | Exact Desc. Rate | Citation Coverage |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.document_rows:
        lines.append(
            "| {run_id} | {document_id} | {tp} | {fp} | {fn} | {code_f1:.3f} | {strict_f1:.3f} | {exact_rate:.3f} | {citation:.3f} |".format(
                run_id=row["run_id"],
                document_id=row["document_id"],
                tp=row["true_positive_count"],
                fp=row["false_positive_count"],
                fn=row["false_negative_count"],
                code_f1=row["code_f1"],
                strict_f1=row["strict_f1"],
                exact_rate=row["description_exact_rate"],
                citation=row["citation_coverage"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_sheet(
    writer: pd.ExcelWriter, sheet_name: str, rows: list[dict[str, Any]]
) -> None:
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe = dataframe.map(_excel_safe_value)
    dataframe.to_excel(writer, sheet_name=sheet_name, index=False)


def _flatten_metadata_rows(
    metadata: dict[str, Any], prefix: str = ""
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, value in metadata.items():
        qualified_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(_flatten_metadata_rows(value, prefix=qualified_key))
        else:
            rendered = (
                json.dumps(value, sort_keys=True)
                if isinstance(value, list)
                else str(value)
            )
            rows.append({"key": qualified_key, "value": rendered})
    return rows


def _excel_safe_value(value: object) -> object:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def _resolve_inventory_path(
    path_value: str, inventory_dir: str, references_dir: Path
) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    inventory_relative = Path(inventory_dir) / path
    if inventory_relative.exists():
        return inventory_relative
    return references_dir / path


def _register_document_keys(
    document_key_map: dict[str, str], document: ReferenceDocument
) -> None:
    for value in (document.document_id, document.reference_path, document.pdf_path):
        key = normalize_document_key(value)
        if key:
            existing_document_id = document_key_map.get(key)
            if (
                existing_document_id is not None
                and existing_document_id != document.document_id
            ):
                raise EvaluationInputError(
                    f"Document key {key!r} maps to both {existing_document_id!r} and {document.document_id!r}."
                )
            document_key_map[key] = document.document_id


def _resolve_review_artifact_path(run_dir: Path, manifest: dict[str, Any]) -> Path:
    exported = (
        manifest.get("artifacts", {}).get("exports", {}).get("requirements_review_json")
    )
    candidates: list[Path] = []
    if exported:
        exported_path = Path(exported)
        candidates.append(run_dir / exported_path.name)
        if not exported_path.is_absolute():
            nested_candidate = run_dir / exported_path
            if _is_within(nested_candidate, run_dir):
                candidates.append(nested_candidate)
    candidates.append(run_dir / "requirements.review.json")
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    raise EvaluationInputError(
        f"Could not find a local requirements.review.json for extraction run {run_dir}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(
            f"Could not read valid JSON from {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvaluationInputError(f"Expected a JSON object in {path}.")
    return payload


def _require_string(value: object, field_name: str, path: Path, item_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationInputError(
            f"{path} requirement #{item_index} has an empty or invalid {field_name}."
        )
    return value


def _load_run_stats(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_stats = manifest.get("run_stats")
    if isinstance(manifest_stats, dict):
        return manifest_stats
    stats_path = run_dir / "run-stats.json"
    if stats_path.is_file():
        return _read_json(stats_path)
    return {}


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _first_by_code(requirements: Iterable[Any]) -> dict[str, Any]:
    by_code: dict[str, Any] = {}
    for requirement in requirements:
        code = normalize_code(getattr(requirement, "code", None))
        if code and code not in by_code:
            by_code[code] = requirement
    return by_code


def _duplicate_codes(requirements: Iterable[Any]) -> list[str]:
    counts: dict[str, int] = {}
    for requirement in requirements:
        code = normalize_code(getattr(requirement, "code", None))
        if code:
            counts[code] = counts.get(code, 0) + 1
    return sorted(code for code, count in counts.items() if count > 1)


def _precision_recall_f1(
    true_positive: int, false_positive: int, false_negative: int
) -> tuple[float, float, float]:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
