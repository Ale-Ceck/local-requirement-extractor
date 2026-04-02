from __future__ import annotations

import base64
import mimetypes
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Iterable, Optional

from config.schema import OutputConfig
from src.data_models.requirement import Requirement, RequirementList
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ReviewHTMLWriter:
    """Write a static HTML review artifact with page-region overlays."""

    def __init__(self, config: OutputConfig):
        self.config = config

    def write(self, requirement_list: RequirementList, output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = str(Path(self.config.directory) / self.config.review_html_filename)
        write_review_html(requirement_list, output_path)
        return output_path


def write_review_html(requirement_list: RequirementList, output_path: str) -> None:
    if not isinstance(requirement_list, RequirementList):
        raise ValueError("requirement_list must be a RequirementList instance")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_html(requirement_list), encoding="utf-8")
    logger.info("Successfully wrote review HTML to %s", path)


def _render_html(requirement_list: RequirementList) -> str:
    cards = "\n".join(_render_requirement_card(index, requirement) for index, requirement in enumerate(requirement_list, start=1))
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Requirement Review</title>
  <style>
    :root {
      --bg: #f4f1e8;
      --panel: #fffdf7;
      --ink: #1f1f1a;
      --muted: #6b665a;
      --accent: #b04632;
      --line: #d9d1be;
      --page: #fcfaf3;
      --highlight-fill: rgba(176, 70, 50, 0.16);
      --highlight-stroke: #b04632;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      color: var(--ink);
      background: linear-gradient(180deg, #efe8d8 0%, var(--bg) 100%);
    }
    main {
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    h1, h2, h3, p { margin-top: 0; }
    .intro {
      margin-bottom: 28px;
      padding: 20px 24px;
      background: rgba(255, 253, 247, 0.86);
      border: 1px solid var(--line);
      border-radius: 18px;
    }
    .grid {
      display: grid;
      gap: 20px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 14px 32px rgba(31, 31, 26, 0.06);
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px 16px;
      margin: 14px 0 18px;
      font-size: 0.95rem;
    }
    .meta div { color: var(--muted); }
    .meta strong { color: var(--ink); }
    .excerpt {
      padding: 14px 16px;
      border-left: 4px solid var(--accent);
      background: #faf5eb;
      margin-bottom: 18px;
      white-space: pre-wrap;
    }
    .pages {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }
    .page-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: #fffef9;
    }
    .page-frame {
      width: 100%;
      border: 1px solid var(--line);
      background: var(--page);
      border-radius: 10px;
      overflow: hidden;
    }
    svg {
      width: 100%;
      height: auto;
      display: block;
      background:
        linear-gradient(0deg, rgba(0,0,0,0.02), rgba(0,0,0,0.02)),
        repeating-linear-gradient(
          180deg,
          transparent,
          transparent 26px,
          rgba(0,0,0,0.03) 27px
        );
    }
    .region-list {
      margin: 12px 0 0;
      padding-left: 18px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .empty {
      color: var(--muted);
      font-style: italic;
    }
  </style>
</head>
<body>
  <main>
    <section class="intro">
      <h1>Requirement Review</h1>
      <p>Total Requirements: """ + str(len(requirement_list)) + """</p>
      <p>This artifact renders the stored provenance geometry directly from <code>source_regions</code>.</p>
    </section>
    <section class="grid">
""" + cards + """
    </section>
  </main>
</body>
</html>
"""


def _render_requirement_card(index: int, requirement: Requirement) -> str:
    code = escape(requirement.code or "(no code)")
    description = escape(requirement.description or "(no description)")
    source_document = escape(requirement.source_document or "(unknown source)")
    source_section = escape(requirement.source_section or "(no section)")
    excerpt = escape(requirement.source_text_excerpt or "(no excerpt)")
    segment_ids = escape(", ".join(requirement.source_segment_ids) if requirement.source_segment_ids else "(none)")
    block_ids = escape(", ".join(requirement.source_block_ids) if requirement.source_block_ids else "(none)")
    page_range = f"{requirement.source_page_start or '?'}-{requirement.source_page_end or requirement.source_page_start or '?'}"
    page_views = _render_page_views(requirement)

    return f"""
      <article class="card">
        <h2>{index}. {code}</h2>
        <p>{description}</p>
        <div class="meta">
          <div><strong>Source Document:</strong> {source_document}</div>
          <div><strong>Page Range:</strong> {escape(page_range)}</div>
          <div><strong>Section:</strong> {source_section}</div>
          <div><strong>Segment IDs:</strong> {segment_ids}</div>
          <div><strong>Block IDs:</strong> {block_ids}</div>
          <div><strong>Regions:</strong> {len(requirement.source_regions)}</div>
        </div>
        <div class="excerpt">{excerpt}</div>
        {page_views}
      </article>
    """


def _render_page_views(requirement: Requirement) -> str:
    grouped_regions: dict[int, list[dict]] = defaultdict(list)
    for region in requirement.source_regions:
        page_number = region.get("page_number")
        if isinstance(page_number, int):
            grouped_regions[page_number].append(region)

    if not grouped_regions:
        return '<p class="empty">No page-region geometry is available for this requirement.</p>'

    views = "\n".join(_render_single_page_view(page_number, regions) for page_number, regions in sorted(grouped_regions.items()))
    return f'<div class="pages">{views}</div>'


def _render_single_page_view(page_number: int, regions: Iterable[dict]) -> str:
    regions = list(regions)
    page_width = max(float(region.get("page_width") or 0.0) for region in regions) or 1000.0
    page_height = max(float(region.get("page_height") or 0.0) for region in regions) or 1400.0
    page_image_href = _resolve_page_image_href(regions)
    shapes = []
    items = []
    for region in regions:
        shape = _render_region_shape(region)
        if shape:
            shapes.append(shape)
        items.append(
            "<li>"
            f"{escape(region.get('segment_id') or '(no segment)')} | "
            f"{escape(region.get('block_id') or '(no block)')} | "
            f"{escape(region.get('block_type') or '(unknown)')} | "
            f"label={escape(str(region.get('paddle_label') or '(none)'))}"
            "</li>"
        )

    return f"""
      <section class="page-card">
        <h3>Page {page_number}</h3>
        <div class="page-frame">
          <svg viewBox="0 0 {page_width} {page_height}" role="img" aria-label="Page {page_number} region map">
            <rect x="0" y="0" width="{page_width}" height="{page_height}" fill="transparent" stroke="#d9d1be" />
            {_render_page_background(page_image_href, page_width, page_height)}
            {''.join(shapes)}
          </svg>
        </div>
        <ul class="region-list">
          {''.join(items)}
        </ul>
      </section>
    """


def _render_region_shape(region: dict) -> str:
    polygon_points = region.get("polygon_points")
    if isinstance(polygon_points, list) and polygon_points:
        normalized_points = []
        for point in polygon_points:
            if isinstance(point, list) and len(point) == 2:
                normalized_points.append(f"{float(point[0])},{float(point[1])}")
        if normalized_points:
            return (
                f'<polygon points="{" ".join(normalized_points)}" '
                'fill="var(--highlight-fill)" stroke="var(--highlight-stroke)" stroke-width="2" />'
            )

    bbox = region.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        return (
            f'<rect x="{x1}" y="{y1}" width="{max(x2 - x1, 1.0)}" height="{max(y2 - y1, 1.0)}" '
            'fill="var(--highlight-fill)" stroke="var(--highlight-stroke)" stroke-width="2" />'
        )
    return ""


def _resolve_page_image_href(regions: Iterable[dict]) -> Optional[str]:
    for region in regions:
        for key in ("page_image_path", "ocr_page_image_path"):
            page_image_path = region.get(key)
            if not isinstance(page_image_path, str) or not page_image_path.strip():
                continue
            path = Path(page_image_path)
            if not path.exists() or not path.is_file():
                continue
            mime_type, _ = mimetypes.guess_type(path.name)
            mime_type = mime_type or "application/octet-stream"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"
    return None


def _render_page_background(page_image_href: Optional[str], page_width: float, page_height: float) -> str:
    if not page_image_href:
        return ""
    return (
        f'<image href="{page_image_href}" x="0" y="0" width="{page_width}" height="{page_height}" '
        'preserveAspectRatio="none" opacity="0.96" />'
    )
