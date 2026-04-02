# Local Requirement Extractor

Extract requirement codes and descriptions from technical PDFs with local OCR and a local text model, then export the results with traceability metadata.

## Supported Workflow

The only supported entrypoint is `src/cli/main.py`.

The canonical PDF path is:

`PDF -> Paddle semantic segments -> anchored markdown -> Ollama extraction with cited segment IDs -> provenance-enriched outputs`

Current support boundaries:

- `src/cli/main.py` is the supported CLI
- `src/pdf_processing/paddleocr_parser.py` is the canonical PDF parser
- Markdown input is still supported as a compatibility path
- standalone scripts under `src/cli/` such as `converter.py`, `extractor.py`, `tester.py`, and `vision.py` are experimental helpers and not part of the supported workflow

## Runtime Requirements

- Python `3.10+`
- Apple Silicon macOS for the intended local development/runtime path
- Ollama running locally at `http://localhost:11434`
- a locally available Ollama model such as `llama3:latest`
- a running PaddleOCR-VL VLM inference service

The current local development virtualenv is Python `3.13.7`.

The repo is aligned to the Apple Silicon PaddleOCR-VL service-backed path described in the official PaddleOCR guide:

https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-Apple-Silicon.html#1-environment-preparation

## Quickstart

1. Create a Python `3.10+` virtual environment.
2. Install `paddlepaddle` and `paddleocr[doc-parser]` for the Apple Silicon path.
3. Install the repo dependencies.
4. Ensure the VLM service is reachable.
5. Ensure Ollama is running locally with the configured model.
6. Review `config.yaml`.
7. Run the CLI.

Example:

```bash
python src/cli/main.py extract --config config.yaml
```

Useful service commands:

```bash
python src/cli/main.py check-vlm-service --config config.yaml
python src/cli/main.py start-vlm-service --config config.yaml
```

## Key Configuration

The pipeline is driven by `config.yaml`.

Important sections:

- `input`: input path, mode, recursion
- `output`: output directory and review-artifact switches
- `parser`: PaddleOCR-VL runtime, pruning, and PDF windowing settings
- `chunking`: chunk-size behavior for extraction
- `extraction`: model name, deduplication, and uncited-result policy
- `ollama`: local model host and timeout

Important parser fields:

- `parser.backend`: parser backend, currently `paddleocr_vl`
- `parser.vlm_backend`: VLM backend used by PaddleOCR-VL
- `parser.vlm_server_url`: VLM inference service base URL
- `parser.vlm_api_model_name`: model name exposed by the VLM service
- `parser.ignored_paddle_labels`: labels excluded from the extraction input
- `parser.exclude_front_matter`: coarse front-matter exclusion
- `parser.toc_section_pruning_mode`: `off`, `audit`, or `enforce`
- `parser.toc_excluded_section_titles`: normalized section-title rules for TOC pruning
- `parser.persist_anchored_markdown`: persist the exact LLM-facing PDF input
- `parser.batch_page_count`: split long PDFs into bounded page windows
- `parser.batch_output_subdir`: per-window output directory name
- `parser.page_start` / `parser.page_end` / `parser.max_pages`: bounded PDF parsing controls

The default parser settings are aligned for the configured Apple Silicon service-backed path:

- `parser.backend: paddleocr_vl`
- `parser.vlm_backend: mlx-vlm-server`
- `parser.vlm_server_url: http://localhost:8111/`
- `parser.vlm_api_model_name: mlx-community/PaddleOCR-VL-1.5-bf16`
- `parser.vlm_server_command: mlx_vlm.server --port 8111`

## Outputs

Each run writes:

- `requirements.xlsx`: tabular requirement list
- `requirements.review.json`: full enriched requirement objects with provenance
- `requirements.review.md`: compact human-readable review report
- `requirements.review.html`: static HTML review artifact with page-region overlays
- `<input-stem>.anchored.md`: exact anchored markdown sent to the extractor for PDF inputs

PDF runs also write:

- `ocr-input.pdf`: the exact PDF passed to PaddleOCR-VL
- `toc-pruning-report.json`: TOC detection and pruning report

When page images are available, the pipeline persists both:

- `page-images/page-001.png` ...: clean rasterized input-page backgrounds used by the HTML review artifact
- `ocr-pages/page-001.png` ...: PaddleOCR-VL native visual outputs with labels and bounding boxes for OCR inspection

When `parser.batch_page_count` is enabled, the run produces:

- merged document-level outputs in the root output directory
- full per-window outputs under `batches/<pdf-stem>/p001-005/`-style directories

Each batch directory contains the same artifact set for that window.

## Code Layout

- `src/cli/`: supported entrypoint plus experimental helpers
- `src/pdf_processing/`: parser abstraction, parser factory, Paddle parser, and compatibility parser
- `src/requirement_extraction/`: chunking, extraction orchestration, and export writers
- `src/llm_integration/`: Ollama client and prompts
- `src/data_models/`: extraction and enriched requirement models
- `config/`: config schema and loader
- `tests/unit/`: architecture-aligned tests

## Notes

- TOC-based section pruning is intentionally coarse and only removes high-confidence non-requirement sections.
- The HTML artifact is static and review-oriented, not an interactive PDF viewer.
- Page-image generation currently exists to support provenance and review artifacts, not as a separate export workflow.
