import yaml
from pathlib import Path
from typing import Any, Dict

from config.schema import AppConfig, InputConfig


def _require_keys(data: Dict[str, Any], keys: list[str]):
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _require_keys(raw, ["input"])

    return AppConfig(
        input=InputConfig(**raw["input"]),
        output=raw.get("output", {}),
        pdf=raw.get("pdf", {}),
        chunking=raw.get("chunking", {}),
        extraction=raw.get("extraction", {}),
        parallel=raw.get("parallel", {}),
        ollama=raw.get("ollama", {}),
        logging=raw.get("logging", {}),
    )
