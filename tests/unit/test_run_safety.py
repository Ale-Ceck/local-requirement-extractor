"""Regression checks for source discovery and non-destructive run startup."""

import json
from pathlib import Path

import pytest

from config.schema import (
    AppConfig,
    InputConfig,
    OutputConfig,
    ParallelConfig,
    ParserConfig,
)
from src.pdf_processing.models import SemanticDocument, SemanticPage, SemanticSegment
from src.requirement_extraction.chunk_cache import load_chunk_cache
from src.requirement_extraction.requirement_extractor import RequirementExtractor
from src.requirement_extraction.run_safety import (
    collect_input_files,
    validate_output_directory,
)


def test_discovery_rejects_missing_empty_and_wrong_type_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        collect_input_files(
            tmp_path / "missing.pdf", suffixes=(".pdf",), recursive=False
        )
    with pytest.raises(ValueError, match="No input files"):
        collect_input_files(tmp_path, suffixes=(".pdf",), recursive=False)
    wrong = tmp_path / "data.txt"
    wrong.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match="No input files"):
        collect_input_files(wrong, suffixes=(".pdf",), recursive=False)


def test_discovery_is_sorted_and_rejects_recursive_collisions(tmp_path: Path) -> None:
    for name in ("z.PDF", "a.pdf"):
        (tmp_path / name).write_bytes(b"fixture")
    assert [
        path.name
        for path in collect_input_files(tmp_path, suffixes=(".pdf",), recursive=False)
    ] == ["a.pdf", "z.PDF"]
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "A.pdf").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="same artifact name"):
        collect_input_files(tmp_path, suffixes=(".pdf",), recursive=True)


@pytest.mark.parametrize(
    "artifact",
    [
        "run-manifest.json",
        "requirements.xlsx",
        "old.chunks.json",
        "old.anchored.md",
        "ocr-input.pdf",
    ],
)
def test_existing_outputs_are_protected(tmp_path: Path, artifact: str) -> None:
    config = AppConfig(
        input=InputConfig(path="source.pdf"),
        output=OutputConfig(directory=str(tmp_path)),
    )
    path = tmp_path / artifact
    path.write_bytes(b"previous-run")
    with pytest.raises(FileExistsError, match="fresh output.directory"):
        validate_output_directory(config)
    assert path.read_bytes() == b"previous-run"
    config.output.overwrite_existing = True
    validate_output_directory(config)


def test_unrelated_files_do_not_block_first_run(tmp_path: Path) -> None:
    (tmp_path / "source.pdf").write_bytes(b"input")
    config = AppConfig(
        input=InputConfig(path=str(tmp_path / "source.pdf")),
        output=OutputConfig(directory=str(tmp_path)),
    )
    validate_output_directory(config)


@pytest.mark.parametrize("value", ["false", "true", 0, 1])
def test_overwrite_requires_a_real_boolean(tmp_path: Path, value: object) -> None:
    manifest = tmp_path / "run-manifest.json"
    manifest.write_bytes(b"previous-run")
    config = AppConfig(
        input=InputConfig(path="source.pdf"),
        output=OutputConfig(directory=str(tmp_path)),
    )
    config.output.overwrite_existing = value  # type: ignore[assignment]
    with pytest.raises(ValueError, match="output.overwrite_existing"):
        validate_output_directory(config)
    assert manifest.read_bytes() == b"previous-run"


class StubModel:
    def __init__(self) -> None:
        self.calls = 0

    def get_structured_response(self, *args, **kwargs) -> str:
        self.calls += 1
        return '[{"code":"REQ-1","description":"Shared requirement.","source_segment_ids":["seg-1"]}]'


class ArtifactParser:
    """Record each source in artifacts normally emitted by the real OCR adapter."""

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)

    def parse_pdf(self, pdf_path: str) -> SemanticDocument:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "ocr-input.pdf").write_bytes(Path(pdf_path).read_bytes())
        images = self.directory / "page-images"
        images.mkdir(exist_ok=True)
        image = images / "page-001.png"
        image.write_bytes(Path(pdf_path).name.encode())
        return SemanticDocument(
            source_document=pdf_path,
            pages=[
                SemanticPage(
                    page=0,
                    width=100,
                    height=100,
                    segments=[
                        SemanticSegment(
                            segment_id="seg-1",
                            segment_kind="text",
                            paddle_label="text",
                            page=0,
                            text_content="REQ-1 Shared requirement.",
                            text_markdown="REQ-1 Shared requirement.",
                            metadata={"page_image_path": str(image)},
                        )
                    ],
                )
            ],
        )


@pytest.mark.parametrize("command", ["prepare-pdf", "extract"])
@pytest.mark.parametrize("batch_size", [None, 1])
def test_multi_document_runs_keep_distinct_artifacts_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    batch_size: int | None,
) -> None:
    inputs, output = tmp_path / "inputs", tmp_path / "output"
    inputs.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (inputs / name).write_bytes(name.encode())
    config = AppConfig(
        input=InputConfig(path=str(inputs), recursive=False),
        output=OutputConfig(directory=str(output)),
        parser=ParserConfig(
            batch_page_count=batch_size, toc_section_pruning_mode="off"
        ),
        parallel=ParallelConfig(enabled=False),
    )
    monkeypatch.setattr(
        "src.requirement_extraction.requirement_extractor.create_document_parser",
        lambda config, **kwargs: ArtifactParser(config.output.directory),
    )
    monkeypatch.setattr(RequirementExtractor, "_count_pdf_pages", lambda *args: 1)
    client = StubModel()
    extractor = RequirementExtractor(config, ollama_client=client, command_name=command)
    extractor.run()
    manifest = json.loads((output / "run-manifest.json").read_text())
    assert manifest["execution"]["status"] == "completed"
    assert manifest["run_stats"]["documents_processed"] == 2
    assert manifest["run_stats"]["chunks_prepared"] == 2
    assert client.calls == (2 if command == "extract" else 0)
    for entry in manifest["artifacts"]["pdf_sources"]:
        source = Path(entry["source"])
        assert "toc_pruning_report" not in entry
        for key, value in entry.items():
            if key != "source" and isinstance(value, str):
                assert Path(value).exists(), (key, value)
        document_dir = output / "documents" / source.stem
        cache_path = Path(entry["chunk_cache"])
        assert cache_path.parent == document_dir
        cache = load_chunk_cache(cache_path)
        assert len(cache) == 1
        assert cache[0].source_document == str(source)
        image_path = Path(cache[0].source_regions[0]["page_image_path"])
        assert image_path.is_relative_to(document_dir)
        assert image_path.read_bytes() == source.name.encode()
        ocr_input = document_dir / (
            f"batches/{source.stem}/p001-001/ocr-input.pdf"
            if batch_size
            else "ocr-input.pdf"
        )
        assert ocr_input.read_bytes() == source.read_bytes()
    if command == "extract":
        review = json.loads((output / "requirements.review.json").read_text())
        assert len(review["requirements"]) == 2
        assert {item["source_document"] for item in review["requirements"]} == {
            str(inputs / "a.pdf"),
            str(inputs / "b.pdf"),
        }
    old_manifest = (output / "run-manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        RequirementExtractor(config, ollama_client=client, command_name=command).run()
    assert (output / "run-manifest.json").read_bytes() == old_manifest


def test_missing_input_produces_failed_manifest_without_model_calls(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    config = AppConfig(
        input=InputConfig(path=str(tmp_path / "missing.pdf")),
        output=OutputConfig(directory=str(output)),
    )
    model = StubModel()
    with pytest.raises(FileNotFoundError):
        RequirementExtractor(
            config, ollama_client=model, document_parser=ArtifactParser(str(output))
        ).run()
    manifest = json.loads((output / "run-manifest.json").read_text())
    assert manifest["execution"]["status"] == "failed"
    assert set(manifest["artifacts"]["exports"]) == {"run_stats"}
    assert model.calls == 0
