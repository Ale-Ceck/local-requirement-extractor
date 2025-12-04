from typing import Dict, Any

# ==============================================================================
# 1. Core Prompt Template for Requirement Extraction
#    This is the primary prompt for extracting requirements.
#    It instructs the LLM on its role, task, constraints, and output format.
# ==============================================================================

# Define a function to generate the main extraction prompt.
# This allows for dynamic injection of context (like ground truth schema).
def get_requirement_extraction_prompt(text_content: str,
                                      output_schema_json: str, # JSON schema string of Requirement Pydantic model
                                      few_shot_examples: str = "" # String containing few-shot examples
                                      ) -> str:
    """
    Generates the prompt for extracting requirements from a given text.

    Args:
        text_content (str): The content of the document (plain text or Markdown).
        output_schema_json (str): JSON schema representing the expected output format (e.g., from RequirementList.model_json_schema()).
        few_shot_examples (str): Optional string containing few-shot examples to guide the LLM.

    Returns:
        str: The complete prompt string.
    """
    # System Instruction: Define the LLM's role and overall objective
    system_instruction = """
    [System]
    You are an automated system specialized in parsing technical documents to extract requirements. Your responses must be precise and strictly follow the provided format.
    """

    # Task Instruction: Detail the specific task
    task_instruction = f"""
    [Task]
    Your task is to identify and extract all requirement entries from the provided text in the [Document] section. A requirement consists of:
    1.  A unique **code** (e.g., REQ-123, HAA-54).
    2.  A **description** detailing what the system must do.

    [Output Instructions]
    1.  Produce a single, valid JSON array of objects.
    2.  Each object in the array must contain two keys: "code" and "description".
    3.  If no requirements are found in the text, output an empty JSON array: `[]`.
    4.  Do not include any text or explanations outside of the JSON array.

    [Extraction Rules]
    - **Source Material:** Process **ONLY** the text inside the `[Document]` section. The examples are for learning and must be ignored in the final output.
    - **Association:** Each requirement code must be paired with the specific description text that directly follows it. **Do not** reuse the same description for different codes.
    - **Verbatim Extraction:** The requirement code and description must be extracted exactly as they appear in the text, without any summarization, paraphrasing, or alteration.
    - **Completeness:** Ensure the entire text of the requirement description is included, even if it spans multiple lines or paragraphs.
    - **Fidelity:** Only extract requirements that are explicitly present in the text. Do not invent or infer requirements.
    """

    # Output Format Instruction: Crucially, tell the LLM the expected JSON schema.
    # We use triple backticks for multi-line JSON strings in the prompt.
    output_format_instruction = f"""
    <output_format>
        You MUST respond STRICTLY in JSON format, as a list of objects.
        Each object in the list MUST conform to the following JSON schema:
        <json>
        {output_schema_json}
        </json>
        Ensure the JSON is perfectly formed and can be parsed by a machine.
        DO NOT include any other text or conversational elements outside the JSON.
    </output_format>
    """

    # Few-shot examples (if provided)
    examples_section = ""
    if few_shot_examples:
        examples_section = f"""
    [Examples for Learning]
    The following are examples to show you the correct output format. Do not include them in your final output.
        {few_shot_examples}
    [End of Examples]

    Now, apply the rules above to extract all requirements from the document below.
    """

    # Combine all parts of the prompt
    #{output_format_instruction.strip()}
    full_prompt = f"""
    {system_instruction.strip()}

    {task_instruction.strip()}

    {examples_section.strip()}

    ---
    Document Content to Analyze:
    [Document]
        {text_content.strip()}
    """
    return full_prompt.strip()

# ==============================================================================
# 2. Few-Shot Examples (Optional but Recommended)
#    Store few-shot examples separately. These can be generated or hand-crafted.
#    Use a format that's easy to embed into the main prompt.
# ==============================================================================

# Example structure for few-shot examples. 
FEW_SHOT_REQUIREMENT_EXAMPLES = """
Example 1:
"## 3 FUNCTIONAL AND PERFORMANCE REQUIREMENTS

## 3.1 HAA Functions

## HAA-54 / CREATED / R

HAA shall provide acceleration measurements in three orthogonal axes.

## 3.2 HAA Switch On and Operating

## HAA-56 / CREATED / T

64

65

66

The HAA full performances shall be achieved within 36 h after switch-on.
"

Output JSON:
[
    {
        "code": "HAA-54",
        "description": "HAA shall provide acceleration measurements in three orthogonal axes."
    },
    {
        "code": "HAA-56",
        "description": "The HAA full performances shall be achieved within 36 h after switch-on."
    }
]

Example 2:
"
## HAA-57 / CREATED / T

● Reference: NIE-ADSF-SYS-RS-029004 ●● Issue: 4 ●●● Date: 29.11.2039

## PROPRIETARY &amp; CONFIDENTIAL INFORMATION

67

68

The HAA shall include dedicated heaters with 90 W peak power consumption (60 W average) to warm up the internal shock absorbers within 3600 s to cope with NIE shock levels.

## 3.3 Conditions for Performance Requirements

## 3.3.1 General Conditions

## HAA-60 / CREATED / T,A

The performance requirements shall be met under worst case conditions over the lifetime from BOL to EOL in the presence of all known effects influencing this performance to include (where appropriate) but not restricted to environmental, launch loads, spacecraft dynamic conditions, ageing, radiation environment, bias, noise, scale factor, quantisation, temperature effects, unit conversion factor errors, 1g to 0g effects and SEU's.

## HAA-62 / CREATED / T,A

The performances specified in this section shall be met for the temperatures operating range as defined in [AD 03] at unit Temperature Reference Point (TRP).
"
Output JSON:
[
    {
        "code": "HAA-57",
        "description": "The HAA shall include dedicated heaters with 90 W peak power consumption (60 W average) to warm up the internal shock absorbers within 3600 s to cope with NIE shock levels."
    },
    {
        "code": "HAA-60",
        "description": "The performance requirements shall be met under worst case conditions over the lifetime from BOL to EOL in the presence of all known effects influencing this performance to include (where appropriate) but not restricted to environmental, launch loads, spacecraft dynamic conditions, ageing, radiation environment, bias, noise, scale factor, quantisation, temperature effects, unit conversion factor errors, 1g to 0g effects and SEU's."
    },
    {
        "code": "HAA-62",
        "description": "The performances specified in this section shall be met for the temperatures operating range as defined in [AD 03] at unit Temperature Reference Point (TRP)."
    }
]
"""

# ==============================================================================
# 3. Utility for getting prompt versions or specific prompt types
#    If you have multiple types of prompts (e.g., for summary, for validation, etc.)
# ==============================================================================

def get_prompt(prompt_name: str, text_content: str, output_schema_json: str, include_few_shot: bool = True) -> str:
    """
    Retrieves a specific prompt template by name.

    Args:
        prompt_name (str): The name of the prompt template (e.g., "requirement_extraction").
        text_content (str): The document content to inject.
        output_schema_json (str): The JSON schema for validation.
        include_few_shot (bool): Whether to include few-shot examples.

    Returns:
        str: The requested prompt string.

    Raises:
        ValueError: If the prompt_name is not recognized.
    """
    few_shot_examples_str = FEW_SHOT_REQUIREMENT_EXAMPLES if include_few_shot else ""

    if prompt_name == "requirement_extraction":
        return get_requirement_extraction_prompt(
            text_content,
            output_schema_json,
            few_shot_examples_str
        )
    # Add more elif blocks for other prompt types if needed in the future
    # elif prompt_name == "summary":
    #     return get_summary_prompt(text_content)
    else:
        raise ValueError(f"Unknown prompt name: {prompt_name}")

# ==============================================================================
# 4. Example Usage (for testing/demonstration)
# ==============================================================================

if __name__ == "__main__":
    from src.data_models.requirement import Requirement
    import json

    # Get the JSON schema from your Pydantic model
    requirement_schema = json.dumps(Requirement.model_json_schema(), indent=2)

    sample_doc_content = """
    This document outlines the system requirements.
    The user interface shall be intuitive (UI-001).
    All data shall be backed up daily (BCK-002).
    """

    # Get the prompt with few-shot examples
    prompt_with_examples = get_prompt(
        "requirement_extraction",
        sample_doc_content,
        requirement_schema,
        include_few_shot=True
    )
    print("--- Prompt with Few-Shot Examples ---")
    print(prompt_with_examples)
    print("\n" + "="*80 + "\n")

    # Get the prompt without few-shot examples
    prompt_without_examples = get_prompt(
        "requirement_extraction",
        sample_doc_content,
        requirement_schema,
        include_few_shot=False
    )
    print("--- Prompt Without Few-Shot Examples ---")
    print(prompt_without_examples)