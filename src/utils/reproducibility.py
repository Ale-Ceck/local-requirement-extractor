"""Helpers for recording the exact software and input state of a run."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

RUNTIME_PACKAGES = (
    "langchain-core",
    "ollama",
    "openpyxl",
    "paddleocr",
    "paddlepaddle",
    "pandas",
    "pydantic",
    "PyMuPDF",
    "PyYAML",
    "semchunk",
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_metadata(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Collect stable path, size, and checksum metadata for existing files."""
    metadata: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        path_key = str(path)
        if path_key in seen_paths or not path.is_file():
            continue
        seen_paths.add(path_key)
        metadata.append(
            {
                "path": path_key,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return metadata


def collect_git_metadata(repo_root: str | Path) -> dict[str, Any]:
    """Collect the checked-out revision and dirty state without mutating Git."""
    root = Path(repo_root)
    commit = _run_git(root, "rev-parse", "HEAD")
    branch = _run_git(root, "branch", "--show-current")
    status = _run_git(root, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status) if status is not None else None,
    }


def collect_runtime_metadata() -> dict[str, Any]:
    """Collect interpreter, platform, and critical dependency versions."""
    dependencies: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": dependencies,
    }


def _run_git(repo_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
