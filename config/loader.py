from pathlib import Path
from typing import Any

import yaml

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


def _require_keys(data: dict[str, Any], keys: list[str]) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def _validate_choice(
    section: str, field_name: str, value: Any, allowed: set[str]
) -> None:
    if not isinstance(value, str):
        raise ValueError(
            f"{section}.{field_name} must be one of {sorted(allowed)}; got {value!r}. "
            "Quote YAML values such as 'off' if needed."
        )
    if value not in allowed:
        raise ValueError(
            f"{section}.{field_name} must be one of {sorted(allowed)}; got {value!r}."
        )


def _validate_config(config: AppConfig) -> None:
    if not isinstance(config.output.overwrite_existing, bool):
        raise ValueError("output.overwrite_existing must be a boolean.")
    for name, value in (
        ("input.path", config.input.path),
        ("output.directory", config.output.directory),
        ("extraction.model_name", config.extraction.model_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")
    _validate_choice(
        "input", "mode", config.input.mode, {"pdf", "markdown", "chunk_cache"}
    )
    _validate_choice(
        "parser",
        "toc_section_pruning_mode",
        config.parser.toc_section_pruning_mode,
        {"off", "audit", "enforce"},
    )
    _validate_choice(
        "parser",
        "include_note_like_segments",
        config.parser.include_note_like_segments,
        {"never", "conditional", "always"},
    )
    _validate_choice(
        "parser", "backend", config.parser.backend, {"paddleocr_vl", "legacy_markdown"}
    )
    _validate_choice("parallel", "backend", config.parallel.backend, {"thread"})
    for name, value in (
        ("parser.batch_page_count", config.parser.batch_page_count),
        ("parser.page_start", config.parser.page_start),
        ("parser.page_end", config.parser.page_end),
        ("parser.max_pages", config.parser.max_pages),
        ("chunking.max_chunk_chars", config.chunking.max_chunk_chars),
        ("chunking.min_chunk_chars", config.chunking.min_chunk_chars),
    ):
        _require_positive_integer(name, value, optional=True)
    _require_positive_integer("parallel.max_workers", config.parallel.max_workers)
    _require_positive_integer("ollama.timeout_seconds", config.ollama.timeout_seconds)
    if (
        config.parser.page_start is not None
        and config.parser.page_end is not None
        and config.parser.page_end < config.parser.page_start
    ):
        raise ValueError(
            "parser.page_end must be greater than or equal to parser.page_start."
        )
    if config.ollama.temperature is not None and (
        isinstance(config.ollama.temperature, bool)
        or not isinstance(config.ollama.temperature, (int, float))
        or not 0.0 <= config.ollama.temperature <= 2.0
    ):
        raise ValueError("ollama.temperature must be between 0.0 and 2.0.")
    if config.ollama.seed is not None and (
        isinstance(config.ollama.seed, bool)
        or not isinstance(config.ollama.seed, int)
        or config.ollama.seed < 0
    ):
        raise ValueError("ollama.seed must be a non-negative integer or null.")


def _require_positive_integer(name: str, value: Any, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{name} must be a positive integer" + (" or null." if optional else ".")
        )


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    _require_keys(raw, ["input"])
    for section in (
        "input",
        "output",
        "pdf",
        "parser",
        "chunking",
        "extraction",
        "parallel",
        "ollama",
        "logging",
    ):
        if section in raw and not isinstance(raw[section], dict):
            raise ValueError(f"{section} must be a YAML mapping.")

    config = AppConfig(
        input=InputConfig(**raw["input"]),
        output=OutputConfig(**raw.get("output", {})),
        pdf=PDFConfig(**raw.get("pdf", {})),
        parser=ParserConfig(**raw.get("parser", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        extraction=ExtractionConfig(**raw.get("extraction", {})),
        parallel=ParallelConfig(**raw.get("parallel", {})),
        ollama=OllamaConfig(**raw.get("ollama", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
    )
    _validate_config(config)
    return config
