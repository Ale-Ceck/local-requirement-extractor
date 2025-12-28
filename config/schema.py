from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------- Input ----------

@dataclass
class InputConfig:
    path: str
    mode: str = "pdf"  # pdf | markdown
    recursive: bool = True
    file_extensions: List[str] = field(default_factory=lambda: [".pdf"])


# ---------- Output ----------

@dataclass
class OutputConfig:
    directory: str = "./data/output"
    overwrite_existing: bool = False
    include_metadata: bool = True
    sheet_name: str = "Requirements"


# ---------- PDF ----------

@dataclass
class PDFConfig:
    markdown_output_dir: Optional[str] = "./data/markdown_output"
    force_rebuild_markdown: bool = False
    cleanup_markdown: bool = True
    keep_intermediate_files: bool = False


# ---------- Chunking ----------

@dataclass
class ChunkingConfig:
    headers_to_split_on: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
    )
    strip_headers: bool = False
    max_chunk_chars: Optional[int] = None
    min_chunk_chars: Optional[int] = None
    merge_small_chunks: bool = True


# ---------- Extraction ----------

@dataclass
class ExtractionConfig:
    model_name: str = "llama3:latest"
    deduplicate_requirements: bool = True
    normalize_codes: bool = True
    allow_empty_results: bool = False


# ---------- Parallel ----------

@dataclass
class ParallelConfig:
    enabled: bool = True
    max_workers: int = 5
    backend: str = "thread"
    chunk_batch_size: int = 1


# ---------- Ollama ----------

@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    timeout_seconds: int = 300
    keep_alive: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


# ---------- Logging ----------

@dataclass
class LoggingConfig:
    level: str = "INFO"
    to_console: bool = True
    to_file: bool = False
    file_path: str = "./myapp.log"
    log_llm_prompts: bool = False
    log_llm_responses: bool = False


# ---------- Root ----------

@dataclass
class AppConfig:
    input: InputConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
