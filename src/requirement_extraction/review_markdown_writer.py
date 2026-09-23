from pathlib import Path
from typing import Optional

from config.schema import OutputConfig
from src.data_models.requirement import Requirement, RequirementList
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ReviewMarkdownWriter:
    """Write a human-readable Markdown review report for extracted requirements."""

    def __init__(self, config: OutputConfig):
        self.config = config

    def write(self, requirement_list: RequirementList, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = str(Path(self.config.directory) / self.config.review_markdown_filename)
        write_review_markdown(requirement_list, output_path)
        return output_path


def write_review_markdown(requirement_list: RequirementList, output_path: str) -> None:
    if not isinstance(requirement_list, RequirementList):
        raise ValueError("requirement_list must be a RequirementList instance")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Requirement Review Report",
        "",
        f"Total Requirements: {len(requirement_list)}",
        "",
    ]

    for index, requirement in enumerate(requirement_list, start=1):
        lines.extend(_render_requirement(index, requirement))

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    logger.info("Successfully wrote review markdown to %s", path)


def _render_requirement(index: int, requirement: Requirement) -> list[str]:
    segment_ids = ", ".join(requirement.source_segment_ids) if requirement.source_segment_ids else "(none)"
    block_ids = ", ".join(requirement.source_block_ids) if requirement.source_block_ids else "(none)"
    page_start = requirement.source_page_start if requirement.source_page_start is not None else "?"
    page_end = requirement.source_page_end if requirement.source_page_end is not None else page_start
    excerpt = requirement.source_text_excerpt or "(no excerpt)"
    section = requirement.source_section or "(no section)"
    source_document = requirement.source_document or "(unknown source)"
    region_count = len(requirement.source_regions)
    first_region = requirement.source_regions[0] if requirement.source_regions else None
    if first_region and first_region.get("bbox") is not None:
        first_region_line = (
            f"First Region: page {first_region.get('page_number', '?')}, "
            f"bbox {first_region['bbox']}"
        )
    else:
        first_region_line = "First Region: (none)"

    return [
        f"## {index}. {requirement.code or '(no code)'}",
        "",
        f"Description: {requirement.description or '(no description)'}",
        f"Source Document: {source_document}",
        f"Page Range: {page_start}-{page_end}",
        f"Section: {section}",
        f"Segment IDs: {segment_ids}",
        f"Block IDs: {block_ids}",
        f"Regions: {region_count}",
        first_region_line,
        "",
        "Excerpt:",
        "",
        f"> {excerpt}",
        "",
    ]
