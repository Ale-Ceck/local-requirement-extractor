import json

from src.data_models.requirement import RequirementExtractionResult
from src.llm_integration.prompt_templates import get_prompt


def test_requirement_prompt_instructs_model_to_extract_contiguous_multi_span_bodies():
    prompt = get_prompt(
        "requirement_extraction",
        '<a id="seg-1"></a>\n## HAA-127 / CREATED / T',
        json.dumps(RequirementExtractionResult.model_json_schema()),
        include_few_shot=True,
    )

    assert "contiguous requirement body that follows it" in prompt
    assert "Anchors inserted between spans do **not** break the same requirement body" in prompt
    assert "consecutive prose lines, bullet lines, note lines" in prompt
    assert "Stop the current requirement description only when you reach the next requirement code or a new section heading" in prompt
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
    assert "HAA-131" in prompt
    assert "If necessary, a radiative TRP could be defined for radiative influence measure." in prompt
    assert "HAA-313" in prompt
    assert "Note 2: thermistors shall be of Type 3 as specified in [AD 03]." in prompt
    assert "HAA-72" in prompt
    assert "Figure 3.3-7" in prompt
