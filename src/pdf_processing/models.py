from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


BBox = Tuple[float, float, float, float]
PolygonPoints = List[List[float]]


@dataclass
class SemanticSegment:
    segment_id: str
    segment_kind: str
    paddle_label: str
    page: int
    text_content: str
    text_markdown: str
    bbox: Optional[BBox] = None
    bbox_norm: Optional[BBox] = None
    polygon_points: PolygonPoints = field(default_factory=list)
    source_block_ids: List[str] = field(default_factory=list)
    group_id: Optional[int] = None
    block_order: Optional[int] = None
    section_path: List[str] = field(default_factory=list)
    heading_level: Optional[int] = None
    included_for_extraction: bool = True
    exclusion_reason: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_number(self) -> int:
        return self.page + 1

    def section_label(self) -> Optional[str]:
        if not self.section_path:
            return None
        return " / ".join(self.section_path)

@dataclass
class SemanticPage:
    page: int
    width: float
    height: float
    segments: List[SemanticSegment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def page_number(self) -> int:
        return self.page + 1


@dataclass
class SemanticDocument:
    source_document: str
    pages: List[SemanticPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def iter_segments(self) -> List[SemanticSegment]:
        segments: List[SemanticSegment] = []
        for page in self.pages:
            segments.extend(page.segments)
        return segments


# Compatibility models kept for the markdown-compatibility path and legacy tests.
@dataclass
class ParsedBlock:
    block_id: str
    page_number: int
    text: str
    block_type: str = "paragraph"
    bbox: Optional[BBox] = None
    reading_order: int = 0
    confidence: Optional[float] = None
    section_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPage:
    page_number: int
    width: float
    height: float
    blocks: List[ParsedBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    source_document: str
    pages: List[ParsedPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
