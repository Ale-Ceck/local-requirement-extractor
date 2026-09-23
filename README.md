# Local Requirement Extractor

Extract requirement codes and descriptions from technical PDFs with local OCR and a local text model, then export the results with traceability metadata.

The consolidated implementation is described in [docs/implementation.md](docs/implementation.md)
(Italian), with verification evidence and remaining limits in
[docs/validation.md](docs/validation.md). The available real-document references
are currently partial: technical regression checks are separate from extraction-quality experiments.

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
3. Install the repo dependencies. `requirements.lock` captures the current Apple Silicon / Python 3.13.7 environment; `requirements.txt` contains broad dependency constraints. A fresh environment has been rebuilt from `requirements-dev.lock` (which includes the runtime lock) on that platform. Other Python versions and platforms have not been verified.
4. Ensure the VLM service is reachable.
5. Ensure Ollama is running locally with the configured model.
6. Review `config.yaml`, especially the input path and a fresh output directory for this run.
7. Run the CLI.

Example:

```bash
python src/cli/main.py extract --config config.yaml
```

Startup rejects missing inputs, empty input selections, colliding source filenames,
and destinations containing pipeline artifacts. Prefer a separate `output.directory`
for each run. `output.overwrite_existing: true` explicitly allows reuse, but does not
remove stale files or support concurrent runs sharing a directory.

Two-stage replay workflow:

```bash
python src/cli/main.py prepare-pdf --config config.yaml
python src/cli/main.py extract --config profiles/replay-extract.yaml
```

Useful service commands:

```bash
python src/cli/main.py check-vlm-service --config config.yaml
python src/cli/main.py start-vlm-service --config config.yaml
```

Quality evaluation:

```bash
python src/cli/main.py evaluate \
  --references-dir data/references \
  --runs-dir data/output/experiments \
  --output-dir data/output/quality \
  --document-inventory profiles/document_inventory.csv
```

The inventory is mandatory when stable document identity matters. Each non-empty row must provide a unique `document_id` and `reference_xlsx`; `pdf_path` provides the mapping from extraction provenance to that stable ID. Empty inventories, duplicate IDs, ambiguous filename keys, missing references, duplicate reference codes, and empty reference descriptions are rejected before scoring.

Evaluation assumes complete references within the selected scope. With partial
references, a correct extracted requirement can be counted as unexpected and a
complete description as a mismatch. Do not interpret those scores as global
precision, recall or F1 until the annotation scope and completeness are established.

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

Important input modes:

- `input.mode: pdf`: full OCR + chunking + extraction pipeline
- `input.mode: chunk_cache`: extraction-only replay from `<input-stem>.chunks.json`
- `input.mode: markdown`: compatibility path for direct Markdown input

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
- `<input-stem>.anchored.md`: included PDF segments and anchors (prompt instructions and few-shot examples are added separately)
- `<input-stem>.chunks.json`: serialized extraction-ready chunk cache for replay runs
- `run-manifest.json`: effective run settings, input mode, model, batch windows, and artifact paths
- `run-stats.json`: extraction counters and timing for repeatable evaluation

Top-level manifests use schema version 2 and record a unique run ID, terminal status, failure details, Git commit and dirty state, Python/platform and dependency versions, extraction-prompt checksum, model name and best-effort model digest, plus SHA-256 checksums for available inputs and main artifacts. Failed runs also attempt to write terminal `run-manifest.json` and `run-stats.json` files.

Statuses are `completed`, `partial` and `failed`. Chunk/model/schema errors or
dropped invalid items make the run `partial`: available results are exported, but
the CLI exits nonzero. Fatal pipeline errors produce `failed`. Valid empty model
responses are not technical errors. Neither `completed` nor a structurally valid
export establishes extraction completeness. The evaluator rejects `partial` and
`failed` runs rather than ranking them as completed runs.

PDF runs also write:

- `ocr-input.pdf`: the exact PDF passed to PaddleOCR-VL
- `toc-pruning-report.json`: TOC detection and pruning report

When page images are available, the pipeline persists both:

- `page-images/page-001.png` ...: clean rasterized input-page backgrounds used by the HTML review artifact
- `ocr-pages/page-001.png` ...: PaddleOCR-VL native visual outputs with labels and bounding boxes for OCR inspection

When `parser.batch_page_count` is enabled, the run produces:

- merged document-level outputs in the root output directory
- full per-window outputs under `batches/<pdf-stem>/p001-005/`-style directories
- per-window chunk caches in those slice directories plus a merged root `<input-stem>.chunks.json`

Preparation writes per-window OCR and cache artifacts; extraction additionally
writes per-window requirement exports. Run manifests and aggregate statistics
are top-level artifacts. Chunk IDs include their page window and remain identical
between caches and exported provenance.

When one run processes multiple PDFs, document-specific artifacts are isolated
under `documents/<input-stem>/`, including any batch subdirectories. Aggregate
requirement exports and the run manifest stay at the root. Identical requirements
from different source documents are preserved by deduplication.

## Experiment Profiles

The repo ships three baseline profiles under `profiles/`:

- `local-stable.yaml`: recommended local full run with batch mode and TOC pruning enforced
- `local-debug.yaml`: sequential debug profile with verbose logging
- `replay-extract.yaml`: extraction-only replay from a previously prepared chunk cache

The main experiment levers are:

- `extraction.model_name`
- `parallel.enabled`
- `parallel.max_workers`
- `chunking.max_chunk_chars`
- `parser.batch_page_count`
- `parser.toc_section_pruning_mode`
- `parser.page_start` / `parser.page_end` / `parser.max_pages`
- `ollama.temperature` / `ollama.seed`

The stable and replay profiles use `temperature: 0.0` and `seed: 42`. The manifest also records the resolved local model digest when the Ollama client exposes it; use digest-pinned model builds rather than mutable `latest` tags for long-lived studies.

Quality evaluation expects one reference workbook per document. The first sheet must contain `CODE` and `DESCRIPTIONS` columns. The scoring policy is:

- reference codes are normalized, non-empty, and unique per document
- reference descriptions are mandatory and compared exactly after whitespace normalization
- `code_f1` measures code retrieval only
- `strict_f1` requires both the code and normalized description to match; a description mismatch contributes one strict false positive and one strict false negative
- repeated extracted codes after the first occurrence count as false positives
- citation coverage reports how many extracted requirements cite at least one source segment
- provenance review counts report invalid, partial, or missing source citations
- every extracted source document must map to exactly one inventory document; unmapped output aborts evaluation

Runs are ranked by strict F1 before code F1. Evaluation outputs are `quality-summary.md`, `quality-analysis.xlsx`, and `quality-analysis.json`; all include the policy and reproducibility metadata.

The local `data/test/examples.pdf` contains requirements also used in the prompt's
few-shot examples. Treat it as development and smoke-test material, not an
independent evaluation set. Use separate documents for final quality measurements.

For controlled model comparisons, run `prepare-pdf` once per document, preserve the resulting chunk cache, and replay every model/prompt configuration against the same cache. Keep full PDF runs as a separate end-to-end check so OCR variance is not mixed with text-model variance.

Replay validates the cache before extraction, including chunk counts, unique IDs
and segment identity. Legacy batch caches with repeated chunk IDs must be
regenerated. Changing chunking settings in a replay profile does not rebuild an
existing cache; prepare a new cache to compare chunking strategies.

## Offline Verification

Install the tools in `requirements-dev.lock` into the development environment,
then run from the repository root:

```bash
bash scripts/verify_pipeline.sh
```

This checks formatting, core lint rules, strict typing for the selected boundary
modules, the regression suite, installed dependency consistency and whitespace.
Use `PIPELINE_PYTHON=/absolute/path/to/python` to select another existing environment.
The full legacy orchestrator is not yet clean under strict typing.

Tests use parser/model doubles and local fixtures. The optional historical Gemma
regression is skipped when its ignored local dataset is absent. These checks do
not invoke live PaddleOCR/Ollama services or validate real-document accuracy.
CLI logging respects `logging.level`, `to_console`, `to_file` and `file_path`;
importing modules does not create a log file.

## Code Layout

- `src/cli/`: supported entrypoint plus experimental helpers
- `src/pdf_processing/`: parser abstraction, parser factory, Paddle parser, and compatibility parser
- `src/requirement_extraction/`: chunking, extraction orchestration, and export writers
- `src/evaluation/`: reference loading, run comparison, and quality report generation
- `src/llm_integration/`: Ollama client and prompts
- `src/data_models/`: extraction and enriched requirement models
- `config/`: config schema and loader
- `tests/unit/`: architecture-aligned tests

## Notes

- TOC-based section pruning is intentionally coarse and only removes high-confidence non-requirement sections.
- The HTML artifact is static and review-oriented, not an interactive PDF viewer.
- Page-image generation currently exists to support provenance and review artifacts, not as a separate export workflow.

## License

The project uses the [MIT License](LICENSE), preserved from the main branch.
