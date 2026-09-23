from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from config.loader import load_config


def test_load_config_reads_parser_section():
    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "input:",
                    "  path: ./data/input",
                    "  mode: pdf",
                    "parser:",
                    "  backend: paddleocr_vl",
                    "  language: en",
                    "  vlm_backend: mlx-vlm-server",
                    "  vlm_server_url: http://localhost:8111/",
                    "  vlm_api_model_name: mlx-community/PaddleOCR-VL-1.5-bf16",
                    "  vlm_server_command: mlx_vlm.server --port 8111",
                    "  batch_page_count: 5",
                    "  batch_output_subdir: batch-runs",
                    "  page_start: 2",
                    "  page_end: 4",
                    "  ignored_paddle_labels:",
                    "    - number",
                    "    - header",
                    "  include_note_like_segments: conditional",
                    "  exclude_front_matter: true",
                    "  toc_section_pruning_mode: audit",
                    "  toc_excluded_section_titles:",
                    "    - contents",
                    "    - appendix",
                    "  toc_pruning_report_filename: toc-pruning-report.json",
                    "  persist_anchored_markdown: true",
                    "chunking:",
                    "  max_chunk_chars: 5000",
                    "extraction:",
                    "  allow_uncited_results: false",
                ]
            ),
            encoding="utf-8",
        )

        config = load_config(config_path)

    assert config.parser.backend == "paddleocr_vl"
    assert config.parser.language == "en"
    assert config.parser.vlm_backend == "mlx-vlm-server"
    assert config.parser.vlm_server_url == "http://localhost:8111/"
    assert config.parser.vlm_api_model_name == "mlx-community/PaddleOCR-VL-1.5-bf16"
    assert config.parser.vlm_server_command == "mlx_vlm.server --port 8111"
    assert config.parser.batch_page_count == 5
    assert config.parser.batch_output_subdir == "batch-runs"
    assert config.parser.page_start == 2
    assert config.parser.page_end == 4
    assert config.parser.ignored_paddle_labels == ["number", "header"]
    assert config.parser.include_note_like_segments == "conditional"
    assert config.parser.exclude_front_matter is True
    assert config.parser.toc_section_pruning_mode == "audit"
    assert config.parser.toc_excluded_section_titles == ["contents", "appendix"]
    assert config.parser.toc_pruning_report_filename == "toc-pruning-report.json"
    assert config.parser.persist_anchored_markdown is True
    assert config.chunking.max_chunk_chars == 5000
    assert config.extraction.allow_uncited_results is False


def test_profile_configs_load():
    for relative_path in (
        "profiles/local-stable.yaml",
        "profiles/local-debug.yaml",
        "profiles/replay-extract.yaml",
    ):
        config = load_config(relative_path)
        assert config.input.path

    replay_config = load_config("profiles/replay-extract.yaml")
    assert replay_config.input.mode == "chunk_cache"
    assert replay_config.chunking.max_chunk_chars == 4000
    assert replay_config.ollama.temperature == 0.0
    assert replay_config.ollama.seed == 42


def test_load_config_rejects_yaml_boolean_for_enum_like_values():
    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "input:",
                    "  path: ./data/input",
                    "  mode: pdf",
                    "parser:",
                    "  toc_section_pruning_mode: off",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="parser.toc_section_pruning_mode"):
            load_config(config_path)


def test_load_config_rejects_unsupported_input_mode():
    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "input:",
                    "  path: ./data/input",
                    "  mode: spreadsheet",
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="input.mode"):
            load_config(config_path)


@pytest.mark.parametrize(
    "ollama_lines,error_pattern",
    [
        (["  temperature: 2.5"], "ollama.temperature"),
        (["  seed: -1"], "ollama.seed"),
        (["  seed: true"], "ollama.seed"),
        (["  temperature: true"], "ollama.temperature"),
        (["  temperature: warm"], "ollama.temperature"),
    ],
)
def test_load_config_rejects_invalid_reproducibility_options(
    ollama_lines: list[str], error_pattern: str
) -> None:
    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "input:",
                    "  path: ./data/input",
                    "ollama:",
                    *ollama_lines,
                ]
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=error_pattern):
            load_config(config_path)


@pytest.mark.parametrize(
    "settings,error_pattern",
    [
        ("parser:\n  page_start: 5\n  page_end: 2", "parser.page_end"),
        ("parser:\n  batch_page_count: 0", "parser.batch_page_count"),
        ("parser:\n  max_pages: -1", "parser.max_pages"),
        ("chunking:\n  max_chunk_chars: true", "chunking.max_chunk_chars"),
        ("parallel:\n  max_workers: 0", "parallel.max_workers"),
        ("parallel:\n  backend: process", "parallel.backend"),
        ("output:\n  directory: ''", "output.directory"),
        ('output:\n  overwrite_existing: "false"', "output.overwrite_existing"),
        ('output:\n  overwrite_existing: "true"', "output.overwrite_existing"),
        ("output:\n  overwrite_existing: 0", "output.overwrite_existing"),
        ("output:\n  overwrite_existing: 1", "output.overwrite_existing"),
        ("parser: []", "parser must be a YAML mapping"),
    ],
)
def test_load_config_rejects_invalid_execution_settings(
    tmp_path: Path, settings: str, error_pattern: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(f"input:\n  path: input.pdf\n{settings}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=error_pattern):
        load_config(path)
