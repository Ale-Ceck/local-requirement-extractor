import yaml
from pathlib import Path
from typing import Any, Dict, Union

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


def _require_keys(data: Dict[str, Any], keys: list[str]):
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def load_config(path: Union[str, Path]) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    _require_keys(raw, ["input"])

    return AppConfig(
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
