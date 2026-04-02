from pathlib import Path
from tempfile import TemporaryDirectory

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
