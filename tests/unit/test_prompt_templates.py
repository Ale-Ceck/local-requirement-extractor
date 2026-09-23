import json
import re

import pytest

from src.data_models.requirement import RequirementExtractionResult
from src.llm_integration.prompt_templates import (
    FEW_SHOT_REQUIREMENT_EXAMPLES,
    get_prompt,
)


@pytest.mark.parametrize(
    "example",
    re.split(r"Example \d+:", FEW_SHOT_REQUIREMENT_EXAMPLES)[1:],
    ids=["bullets", "prose", "notes", "figure-reference"],
)
def test_few_shot_outputs_are_valid_json_with_supported_citations(example):
    document, output = example.split("Output JSON:", 1)
    items = RequirementExtractionResult.model_validate(json.loads(output))
    anchors = re.findall(r'<a id="([^"]+)"></a>', document)

    assert len(items.root) == 1
    requirement = items.root[0]
    assert requirement.code
    assert requirement.description
    assert requirement.source_segment_ids
    assert len(set(requirement.source_segment_ids)) == len(
        requirement.source_segment_ids
    )
    positions = [
        anchors.index(segment_id) for segment_id in requirement.source_segment_ids
    ]
    assert positions == sorted(positions)


def test_prose_example_preserves_continuation_and_stops_at_new_section():
    example = FEW_SHOT_REQUIREMENT_EXAMPLES.split("Example 2:")[1].split("Example 3:")[
        0
    ]
    document, output = example.split("Output JSON:", 1)
    requirement = json.loads(output)[0]

    assert (
        "\nNote: readings shall use the same reference point."
        in requirement["description"]
    )
    assert requirement["source_segment_ids"] == [
        "seg-thermal-002",
        "seg-thermal-003",
        "seg-thermal-004",
    ]
    assert "#### 2.2 Verification procedure" in document
    assert (
        "The verification procedure is described separately."
        not in requirement["description"]
    )


def test_requirement_prompt_instructs_model_to_extract_contiguous_multi_span_bodies():
    prompt = get_prompt(
        "requirement_extraction",
        '<a id="seg-1"></a>\n## HAA-127 / CREATED / T',
        json.dumps(RequirementExtractionResult.model_json_schema()),
        include_few_shot=True,
    )

    assert "contiguous requirement body that follows it" in prompt
    assert (
        "Anchors inserted between spans do **not** break the same requirement body"
        in prompt
    )
    assert "consecutive prose lines, bullet lines, note lines" in prompt
    assert (
        "Stop the current requirement description only when you reach the next requirement code or a new section heading"
        in prompt
    )
    assert "include every relevant ID in order" in prompt


def test_requirement_prompt_few_shot_examples_cover_multi_span_failure_modes():
    prompt = get_prompt(
        "requirement_extraction",
        '<a id="seg-1"></a>\n## HAA-127 / CREATED / T',
        json.dumps(RequirementExtractionResult.model_json_schema()),
        include_few_shot=True,
    )

    assert "HAA-127" in prompt
    assert "• HAA ACU design max: 10 W" in prompt
    assert "REQ-THERM-001" in prompt
    assert "Note: readings shall use the same reference point." in prompt
    assert "HAA-313" in prompt
    assert "Note 2: thermistors shall be of Type 3 as specified in [AD 03]." in prompt
    assert "HAA-72" in prompt
    assert "Figure 3.3-7" in prompt
