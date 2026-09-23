#!/usr/bin/env bash
# Run the offline consolidation checks from any working directory.
set -euo pipefail

pipeline_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$pipeline_root"
pipeline_python="${PIPELINE_PYTHON:-$pipeline_root/.venv/bin/python}"

checked_files=(
  config/loader.py
  src/cli/main.py
  src/requirement_extraction/requirement_extractor.py
  src/requirement_extraction/chunk_cache.py
  src/requirement_extraction/run_safety.py
  src/utils/logging_config.py
  src/llm_integration/prompt_templates.py
  tests/unit/test_cli_main.py
  tests/unit/test_config_loader.py
  tests/unit/test_requirement_extractor.py
  tests/unit/test_chunk_cache.py
  tests/unit/test_run_safety.py
  tests/unit/test_logging_config.py
  tests/unit/test_quality_evaluator.py
  tests/unit/test_prompt_templates.py
)

"$pipeline_python" -m black --check "${checked_files[@]}"
"$pipeline_python" -m ruff check --select F,I "${checked_files[@]}"
"$pipeline_python" -m mypy --strict --ignore-missing-imports --follow-imports=skip \
  config/loader.py \
  src/requirement_extraction/chunk_cache.py \
  src/requirement_extraction/run_safety.py \
  src/utils/logging_config.py \
  src/utils/reproducibility.py \
  src/evaluation/quality_evaluator.py \
  src/llm_integration/ollama_client.py
"$pipeline_python" -m pytest -q
"$pipeline_python" -m pip check
git diff --check
