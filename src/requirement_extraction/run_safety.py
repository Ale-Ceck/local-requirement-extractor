"""Validate run inputs and protect existing pipeline artifacts before execution."""

from pathlib import Path

from config.schema import AppConfig


def validate_output_directory(config: AppConfig) -> None:
    """Reject occupied run destinations unless overwriting was explicitly configured.

    Raises:
        FileExistsError: The destination contains pipeline artifacts or is a file.
        ValueError: The overwrite option is not an actual boolean.
    """
    if not isinstance(config.output.overwrite_existing, bool):
        raise ValueError("output.overwrite_existing must be a boolean.")
    directory = Path(config.output.directory)
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"Output directory is not a directory: {directory}")
    if config.output.overwrite_existing is True or not directory.exists():
        return
    names = {
        "run-manifest.json",
        "run-stats.json",
        "requirements.xlsx",
        "ocr-input.pdf",
        "page-images",
        "ocr-pages",
        "documents",
        config.parser.batch_output_subdir,
        config.parser.toc_pruning_report_filename,
        config.output.review_artifact_filename,
        config.output.review_markdown_filename,
        config.output.review_html_filename,
    }
    occupied = [
        str(directory / name)
        for name in sorted(names)
        if name and (directory / name).exists()
    ]
    occupied.extend(str(path) for path in sorted(directory.glob("*.chunks.json")))
    occupied.extend(str(path) for path in sorted(directory.glob("*.anchored.md")))
    if occupied:
        raise FileExistsError(
            f"Output directory already contains pipeline artifacts: {directory}. "
            "Choose a fresh output.directory for this run, or explicitly set "
            "output.overwrite_existing=true to reuse it."
        )


def collect_input_files(
    path: Path, *, suffixes: tuple[str, ...], recursive: bool
) -> list[Path]:
    """Return sorted, unambiguous source files or fail before invoking a model.

    Raises:
        FileNotFoundError: The input path does not exist.
        ValueError: No matching files exist, or two files share an artifact stem.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    candidates = (
        [path] if path.is_file() else path.rglob("*") if recursive else path.glob("*")
    )
    normalized_suffixes = tuple(suffix.lower() for suffix in suffixes)
    files = sorted(
        candidate
        for candidate in candidates
        if candidate.is_file() and candidate.name.lower().endswith(normalized_suffixes)
    )
    if not files:
        raise ValueError(f"No input files matching {suffixes} found at {path}")
    stems: dict[str, Path] = {}
    for source in files:
        key = source.stem.casefold()
        if key in stems:
            raise ValueError(
                f"Input files have the same artifact name: {stems[key]} and {source}. "
                "Use distinct filenames or separate runs."
            )
        stems[key] = source
    return files
