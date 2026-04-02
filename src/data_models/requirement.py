from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, RootModel, field_validator


class RequirementExtractionItem(BaseModel):
    code: Optional[str] = Field(default=None, description="The code of the requirement")
    description: Optional[str] = Field(default=None, description="The description of the requirement")
    source_segment_ids: List[str] = Field(
        default_factory=list,
        description="Stable source segment identifiers cited by the extractor",
    )

    @field_validator("code", "description", mode="before")
    @classmethod
    def strip_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("source_segment_ids", mode="before")
    @classmethod
    def normalize_source_segment_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if not isinstance(value, list):
            return []
        normalized: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized


class RequirementExtractionResult(RootModel[List[RequirementExtractionItem]]):
    root: List[RequirementExtractionItem]


class Requirement(RequirementExtractionItem):
    source_document: Optional[str] = Field(default=None, description="Source document path or identifier")
    source_chunk_id: Optional[str] = Field(default=None, description="Chunk identifier used during extraction")
    source_page_start: Optional[int] = Field(default=None, description="First source page for the requirement")
    source_page_end: Optional[int] = Field(default=None, description="Last source page for the requirement")
    source_block_ids: List[str] = Field(default_factory=list, description="Source block identifiers")
    source_section: Optional[str] = Field(default=None, description="Logical section path for the requirement")
    source_text_excerpt: Optional[str] = Field(default=None, description="Supporting excerpt from the source")
    source_bbox_list: List[List[float]] = Field(default_factory=list, description="Bounding boxes for the source evidence")
    source_regions: List[Dict[str, Any]] = Field(default_factory=list, description="Region-level source evidence")
    confidence: Optional[float] = Field(default=None, description="Confidence score if available")
    review_status: Optional[str] = Field(default=None, description="Manual review status")

    @field_validator("source_document", "source_chunk_id", "source_section", "source_text_excerpt", "review_status", mode="before")
    @classmethod
    def strip_metadata_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class RequirementList(RootModel[List[Requirement]]):
    root: List[Requirement]

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return len(self.root)

    def __getitem__(self, index):
        return self.root[index]

    def is_empty(self) -> bool:
        return len(self.root) == 0

    def get_codes(self) -> List[str]:
        return [req.code for req in self.root if req.code]

    def merge(self, other: "RequirementList") -> "RequirementList":
        return RequirementList(list(self.root) + list(other.root))
