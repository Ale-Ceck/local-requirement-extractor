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
    write_review_artifact: bool = True
    review_artifact_filename: str = "requirements.review.json"
    write_review_markdown: bool = True
    review_markdown_filename: str = "requirements.review.md"
    write_review_html: bool = True
    review_html_filename: str = "requirements.review.html"


# ---------- PDF ----------

@dataclass
class PDFConfig:
    markdown_output_dir: Optional[str] = "./data/markdown_output"
    force_rebuild_markdown: bool = False
    cleanup_markdown: bool = True
    keep_intermediate_files: bool = False


# ---------- Parser ----------

@dataclass
class ParserConfig:
    backend: str = "paddleocr_vl"
    language: str = "en"
    vlm_backend: str = "mlx-vlm-server"
    vlm_server_url: str = "http://localhost:8111/"
    vlm_api_model_name: str = "mlx-community/PaddleOCR-VL-1.5-bf16"
    vlm_api_key: Optional[str] = None
    vlm_server_command: str = "mlx_vlm.server --port 8111"
    batch_page_count: Optional[int] = None
    batch_output_subdir: str = "batches"
    max_pages: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    layout_detection: bool = True
    extract_tables: bool = True
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    ignored_paddle_labels: List[str] = field(
        default_factory=lambda: [
            "number",
            "footnote",
            "header",
            "header_image",
            "footer",
            "footer_image",
            "aside_text",
        ]
    )
    include_note_like_segments: str = "conditional"
    exclude_front_matter: bool = True
    toc_section_pruning_mode: str = "audit"
    toc_excluded_section_titles: List[str] = field(
        default_factory=lambda: [
            "contents",
            "table of contents",
            "summary",
            "document change log",
            "introduction",
            "introduction and scope",
            "scope",
            "summary description",
            "acronym list",
            "documents",
            "applicable documents",
            "export control information",
            "requirement section cross reference",
            "section cross reference",
            "distribution list",
            "configuration management",
            "annex",
            "appendix",
        ]
    )
    toc_pruning_report_filename: str = "toc-pruning-report.json"
    persist_anchored_markdown: bool = True


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
    allow_uncited_results: bool = False


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
    parser: ParserConfig = field(default_factory=ParserConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
