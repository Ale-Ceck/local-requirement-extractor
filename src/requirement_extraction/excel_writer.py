# src/requirement_extractor/excel_writer.py
from pathlib import Path
from typing import Optional

import pandas as pd

from config.schema import OutputConfig
from src.data_models.requirement import RequirementList
from src.requirement_extraction.review_artifact_writer import ReviewArtifactWriter
from src.requirement_extraction.review_html_writer import ReviewHTMLWriter
from src.requirement_extraction.review_markdown_writer import ReviewMarkdownWriter
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

class ExcelWriter:
    """Write extracted requirements to an Excel file using centralized config."""

    def __init__(self, config: OutputConfig):
        self.config = config
        self.review_artifact_writer = ReviewArtifactWriter(config)
        self.review_markdown_writer = ReviewMarkdownWriter(config)
        self.review_html_writer = ReviewHTMLWriter(config)

    def write(self, requirement_list: RequirementList, output_path: Optional[str] = None) -> None:
        if output_path is None:
            output_path = self._default_output_path()
        write_to_excel(requirement_list, output_path, config=self.config)
        if self.config.write_review_artifact:
            self.review_artifact_writer.write(
                requirement_list,
                artifact_path=self._review_artifact_path(output_path),
            )
        if self.config.write_review_markdown:
            self.review_markdown_writer.write(
                requirement_list,
                output_path=self._review_markdown_path(output_path),
            )
        if self.config.write_review_html:
            self.review_html_writer.write(
                requirement_list,
                output_path=self._review_html_path(output_path),
            )

    def _default_output_path(self) -> str:
        output_dir = Path(self.config.directory)
        return str(output_dir / "requirements.xlsx")

    def _review_artifact_path(self, output_path: str) -> str:
        if self.config.review_artifact_filename and output_path == self._default_output_path():
            return str(Path(self.config.directory) / self.config.review_artifact_filename)
        return str(Path(output_path).with_suffix(".review.json"))

    def _review_markdown_path(self, output_path: str) -> str:
        if self.config.review_markdown_filename and output_path == self._default_output_path():
            return str(Path(self.config.directory) / self.config.review_markdown_filename)
        return str(Path(output_path).with_suffix(".review.md"))

    def _review_html_path(self, output_path: str) -> str:
        if self.config.review_html_filename and output_path == self._default_output_path():
            return str(Path(self.config.directory) / self.config.review_html_filename)
        return str(Path(output_path).with_suffix(".review.html"))


def write_to_excel(
    requirement_list: RequirementList,
    output_path: Optional[str],
    config: Optional[OutputConfig] = None,
) -> None:
    """Writes the extracted requirements to an Excel file.
    
    Args:
        requirement_list: RequirementList containing requirements to export
        output_path: Path where the Excel file should be saved
        config: Centralized output configuration (optional)
        
    Raises:
        ValueError: If requirement_list is empty or output_path is invalid
        OSError: If file cannot be written due to permissions or disk space
    """
    config = config or OutputConfig()

    # Input validation
    if not isinstance(requirement_list, RequirementList):
        raise ValueError("requirement_list must be a RequirementList instance")

    if output_path is None:
        output_path = str(Path(config.directory) / "requirements.xlsx")

    if not output_path or not str(output_path).strip():
        raise ValueError("output_path cannot be empty")
    
    if requirement_list.is_empty():
        logger.warning("RequirementList is empty, creating Excel file with headers only")
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensuring output directory exists: {output_dir}")
    
    try:
        data = {
            "Requirement Code": [req.code for req in requirement_list],
            "Description": [req.description for req in requirement_list],
        }

        if config.include_metadata:
            data.update(
                {
                    "Source Document": [req.source_document for req in requirement_list],
                    "Page Start": [req.source_page_start for req in requirement_list],
                    "Page End": [req.source_page_end for req in requirement_list],
                    "Section": [req.source_section for req in requirement_list],
                    "Source Excerpt": [req.source_text_excerpt for req in requirement_list],
                    "Source Segment IDs": [", ".join(req.source_segment_ids) for req in requirement_list],
                    "Block IDs": [", ".join(req.source_block_ids) for req in requirement_list],
                    "Review Status": [req.review_status for req in requirement_list],
                    "Confidence": [req.confidence for req in requirement_list],
                }
            )

        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)
        
        logger.info(f"Successfully wrote {len(requirement_list)} requirements to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to write Excel file to {output_path}: {str(e)}")
        raise OSError(f"Failed to write Excel file: {str(e)}") from e
