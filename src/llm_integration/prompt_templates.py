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
    3.  A list of **source_segment_ids** containing the anchor IDs that support that requirement.

    [Output Instructions]
    1.  Produce a single, valid JSON array of objects.
    2.  Each object in the array must contain three keys: "code", "description", and "source_segment_ids".
    3.  If no requirements are found in the text, output an empty JSON array: `[]`.
    4.  Do not include any text or explanations outside of the JSON array.

    [Extraction Rules]
    - **Source Material:** Process **ONLY** the text inside the `[Document]` section. The examples are for learning and must be ignored in the final output.
    - **Anchors:** The document contains source anchors formatted as `<a id="SEGMENT_ID"></a>`. These anchors identify the semantic source segments. Use them to populate `source_segment_ids`.
    - **Association:** Each requirement code must be paired with the contiguous requirement body that follows it, even when that body is split across multiple anchored segments. **Do not** reuse the same description for different codes.
    - **Continuation Spans:** Anchors inserted between spans do **not** break the same requirement body. After a requirement code, continue collecting the description from consecutive anchored segments that belong to the same requirement body.
    - **Included Continuations:** Treat consecutive prose lines, bullet lines, note lines, and same-body continuation lines as part of the same description when they follow the requirement code and remain under the same requirement body.
    - **Stop Boundaries:** Stop the current requirement description only when you reach the next requirement code or a new section heading that starts a different document section.
    - **Verbatim Extraction:** The requirement code and description must be extracted exactly as they appear in the text, without any summarization, paraphrasing, or alteration.
    - **Completeness:** Ensure the entire text of the requirement description is included, even if it spans multiple lines, bullets, notes, or paragraphs across multiple anchored segments.
    - **Fidelity:** Only extract requirements that are explicitly present in the text. Do not invent or infer requirements.
    - **Citation Fidelity:** Every extracted requirement must cite the anchor IDs for all segments that directly support the full requirement body. If a requirement spans multiple anchored segments, include every relevant ID in order. Never invent anchor IDs.
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

    {output_format_instruction.strip()}

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
"<a id=\"seg-p127-i001\"></a>
#### 4.2.1 Heat Dissipation

<a id=\"seg-p127-i002\"></a>
## HAA-127 / CREATED / T

<a id=\"seg-p127-i003\"></a>
The following operational average and maximum dissipation shall be respected:

<a id=\"seg-p127-i004\"></a>
• HAA ADA average: 13 W

<a id=\"seg-p127-i005\"></a>
• HAA ACU average: 9 W

<a id=\"seg-p127-i006\"></a>
• HAA ADA design max: 19.5 W

<a id=\"seg-p127-i007\"></a>
• HAA ACU design max: 10 W

<a id=\"seg-p127-i008\"></a>
#### 4.2.2 Radiative and conductive interfaces
"

Output JSON:
[
    {
        "code": "HAA-127",
        "description": "The following operational average and maximum dissipation shall be respected:\n• HAA ADA average: 13 W\n• HAA ACU average: 9 W\n• HAA ADA design max: 19.5 W\n• HAA ACU design max: 10 W",
        "source_segment_ids": ["seg-p127-i002", "seg-p127-i003", "seg-p127-i004", "seg-p127-i005", "seg-p127-i006", "seg-p127-i007"]
    }
]

Example 2:
"
<a id=\"seg-p131-i001\"></a>
#### 4.2.3 Environment temperature range

<a id=\"seg-p131-i002\"></a>
## HAA-131 / CREATED / R

<a id=\"seg-p131-i003\"></a>
The Temperature Reference Point (TRP) for the HAA shall be defined by the Supplier at a single HAA interface point and shall be approved by Prime.

<a id=\"seg-p131-i004\"></a>
If necessary, a radiative TRP could be defined for radiative influence measure.

<a id=\"seg-p131-i005\"></a>
The HAA shall withstand the temperature ranges at unit TRP given in the table below:
"
Output JSON:
[
    {
        "code": "HAA-131",
        "description": "The Temperature Reference Point (TRP) for the HAA shall be defined by the Supplier at a single HAA interface point and shall be approved by Prime.\nIf necessary, a radiative TRP could be defined for radiative influence measure.",
        "source_segment_ids": ["seg-p131-i002", "seg-p131-i003", "seg-p131-i004"]
    }
]

Example 3:
"
<a id=\"seg-p313-i001\"></a>
## HAA-313 / CREATED / R

<a id=\"seg-p313-i002\"></a>
The HAA shall be designed to work with 1 heater line (Nominal + Redundant) controlled by spacecraft. For that purpose, heaters and 3 thermistors shall be procured and installed by the supplier and can be connected to spacecraft active thermal control.

<a id=\"seg-p313-i003\"></a>
Note 1: the control will be performed with an ON/OFF coarse temperature control.

<a id=\"seg-p313-i004\"></a>
Note 2: thermistors shall be of Type 3 as specified in [AD 03].

<a id=\"seg-p313-i005\"></a>
#### 4.2.5 Thermal Model
"
Output JSON:
[
    {
        "code": "HAA-313",
        "description": "The HAA shall be designed to work with 1 heater line (Nominal + Redundant) controlled by spacecraft. For that purpose, heaters and 3 thermistors shall be procured and installed by the supplier and can be connected to spacecraft active thermal control.\nNote 1: the control will be performed with an ON/OFF coarse temperature control.\nNote 2: thermistors shall be of Type 3 as specified in [AD 03].",
        "source_segment_ids": ["seg-p313-i001", "seg-p313-i002", "seg-p313-i003", "seg-p313-i004"]
    }
]

Example 4:
"
<a id=\"seg-p072-i001\"></a>
##### 3.3.2.2 Microvibration Environment

<a id=\"seg-p072-i002\"></a>
## HAA-72 / CREATED / A,R

<a id=\"seg-p072-i003\"></a>
The HAA shall meet its performance requirements during the radio science experiments when exposed to a microvibration envelope given in Figure 3.3-3, Figure 3.3-4, Figure 3.3-5, Figure 3.3-6, Figure 3.3-7, and Figure 3.3-8.

<a id=\"seg-p072-i004\"></a>
Figure 3.3-3: MGA Y microstepping
"
Output JSON:
[
    {
        "code": "HAA-72",
        "description": "The HAA shall meet its performance requirements during the radio science experiments when exposed to a microvibration envelope given in Figure 3.3-3, Figure 3.3-4, Figure 3.3-5, Figure 3.3-6, Figure 3.3-7, and Figure 3.3-8.",
        "source_segment_ids": ["seg-p072-i002", "seg-p072-i003"]
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
