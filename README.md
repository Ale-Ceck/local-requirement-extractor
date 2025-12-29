# Local Requirement Extractor

Extract requirement codes and descriptions from technical documents using a local LLM (Ollama), then export results to Excel.

## What this does
- Ingests PDFs or Markdown files
- Converts PDF to Markdown
- Splits Markdown into LLM-sized chunks
- Calls a local Ollama model to extract requirement entries as JSON
- Writes results to an Excel file

This is an offline-first pipeline designed for local processing (no hosted LLM required).

## Quick start
1. Create and activate a virtual environment
2. Install Python dependencies
3. Ensure Ollama is running locally and a model is available
4. Edit `config.yaml`
5. Run the CLI

Example:

```bash
python src/cli/main.py --config config.yaml
```

## Requirements
- Python 3.10+ (tested locally with newer versions)
- Ollama running at `http://localhost:11434`
- An Ollama model available (e.g., `llama3:latest`)

## Configuration
The pipeline is driven by `config.yaml`. Key settings:

- `input.path`: PDF file or directory
- `input.mode`: `pdf` or `markdown`
- `input.recursive`: scan directories recursively
- `output.directory`: where the Excel output is written
- `pdf.markdown_output_dir`: where intermediate Markdown goes
- `chunking`: header-based chunking and merge behavior
- `extraction.model_name`: Ollama model name
- `parallel`: enable/disable threaded chunk processing
- `ollama`: host/timeout and sampling settings

See `config/schema.py` for all fields and defaults.

## How it works
1. **PDF to Markdown**: PDFs are converted to Markdown using `pymupdf4llm`.
2. **Chunking**: Markdown is split by headers and merged to fit LLM context windows.
3. **LLM extraction**: Each chunk is sent to Ollama with a strict JSON schema.
4. **Aggregation**: Results are merged into a single list.
5. **Export**: An Excel file is written with requirement code and description columns.

## Project layout (src)
- `src/cli/`: CLI entrypoints and helper scripts
- `src/requirement_extraction/`: main extraction pipeline and Excel writer
- `src/llm_integration/`: Ollama client and prompt templates
- `src/pdf_processing/`: PDF to Markdown conversion
- `src/utils/`: logging, file utilities, markdown splitter
- `src/data_models/`: Pydantic models for requirements

## Notes and caveats
- The main entrypoint is `src/cli/main.py`. Other scripts in `src/cli/` are experimental utilities.
- Output is a simple Excel file with two columns: requirement code and description.
- The LLM is instructed to return strict JSON; malformed responses are rejected.

## Troubleshooting
- If Ollama is not running or the model is missing, requests will fail. Start Ollama and pull a model before running the CLI.
- If no requirements are found, the output Excel will still be created (with headers only).

## License
TBD
