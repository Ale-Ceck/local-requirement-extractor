#!/usr/bin/env python3
"""
Requirement extractor tester. Used to fine-tune the module and to debug the operations.
"""
import json
import sys
from pathlib import Path

# Add src to path so we can import our modules
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from utils.logging_config import setup_logger
from utils import file_operations as fo
from llm_integration.ollama_client import get_client
from requirement_extraction.excel_writer import write_to_excel

from utils.markdown_splitter import split_markdown
from data_models.requirement import RequirementList, Requirement
from llm_integration.prompt_templates import get_prompt

# Get a logger instance
logger = setup_logger(__name__)


def parse_llm_response(response_str: str) -> RequirementList:
    """Parse LLM JSON response into RequirementList object."""
    try:
        # Parse JSON string
        response_data = json.loads(response_str)
        
        # Handle different response formats
        if isinstance(response_data, list):
            # Direct list of requirements
            requirements_data = response_data
        elif isinstance(response_data, dict) and 'root' in response_data:
            # Wrapped in root (matching RequirementList structure)
            requirements_data = response_data['root']
        else:
            logger.warning(f"Unexpected response format: {type(response_data)}")
            return RequirementList([])
        
        # Convert to Requirement objects
        requirements = []
        for req_data in requirements_data:
            if isinstance(req_data, dict):
                # Ensure we have the required fields
                code = req_data.get('code')
                description = req_data.get('description')
                
                if code or description:  # At least one field should be present
                    requirements.append(Requirement(code=code, description=description))
        
        return RequirementList(requirements)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response: {response_str}")
        return RequirementList([])
    except Exception as e:
        logger.error(f"Error processing LLM response: {e}")
        return RequirementList([])


def tester():
    """Extract requirements from all Markdown files in data/input and export to Excel format."""

    # Instantiate the OllamaClient
    ollama_client = get_client()
    
    # Check if server is running
    if not ollama_client.is_server_running():
        logger.error("Ollama server is not running. Please start the server first.")
        print("❌ Ollama server is not running. Please start the server first.")
        return
    
    # Define paths
    md_file = Path("data/test/cleaned_requirements.md")
    output_dir = Path("data/test")
    
    if not fo.dir_exists(output_dir):
        logger.error(f"Output directory not found: {output_dir}")
        return
    
    if not md_file.exists():
        logger.error(f"Markdown file not found: {md_file}")
        print(f"❌ Markdown file not found: {md_file}")
        return
    
    try:
        logger.info(f"Processing: {md_file.name}")

        # Extract requirements from Markdown
        logger.info(f"Starting requirement extraction from Markdown: {md_file}")

        requirement_schema = RequirementList.model_json_schema()

        # Chunk the markdown file
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2")
        ]
        chunks = split_markdown(markdown_path=md_file, headers_to_split_on=headers_to_split_on)
        docs = [doc.page_content for doc in chunks]

# DEBUGGING: salvo i documenti splittati per vedere come sono
        serialized_docs = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in chunks
        ]
        output_path = output_dir / "documents.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialized_docs, f, ensure_ascii= False, indent=2)
        quit
####
        
        # Initialize containers for results
        all_requirements = RequirementList([])
        all_raw_responses = []
        
        logger.info(f"Processing {len(docs)} document chunks")
        
        # Process chunks 
        for i, doc in enumerate(docs):
            try:
                logger.info(f"Processing chunk {i+1}/{len(docs)}")
                
                # Get the chunk content (assuming doc has a 'page_content' or similar attribute)
                chunk_content = getattr(doc, 'page_content', str(doc))
                
                prompt = get_prompt("requirement_extraction", chunk_content, requirement_schema, include_few_shot=True)
                print(prompt)
                
                # Get structured response from LLM
                response_str = ollama_client.get_structured_response(prompt, model_name="gemma3:27b")
                
                if response_str:
                    # Store raw response for debugging
                    all_raw_responses.append({
                        'chunk_index': i,
                        'chunk_preview': chunk_content[:100] + "..." if len(chunk_content) > 100 else chunk_content,
                        'raw_response': response_str
                    })
                    
                    # Parse the response into RequirementList
                    chunk_requirements = parse_llm_response(response_str)
                    
                    if not chunk_requirements.is_empty():
                        # Merge with existing requirements
                        all_requirements = all_requirements.merge(chunk_requirements)
                        logger.info(f"Extracted {len(chunk_requirements)} requirements from chunk {i+1}")
                    else:
                        logger.info(f"No requirements found in chunk {i+1}")
                else:
                    logger.warning(f"No response received for chunk {i+1}")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                continue  # Continue with next chunk instead of failing completely

        logger.info(f"Successfully extracted {len(all_requirements)} total requirements from {md_file.name}")
        
        # Save raw responses for debugging
        answer_file_path = output_dir / "raw_responses.json"
        try:
            with open(answer_file_path, "w", encoding="utf-8") as f:
                json.dump(all_raw_responses, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved raw LLM responses to {answer_file_path}")
        except Exception as e:
            logger.error(f"Failed to save raw responses file: {e}")
        
        # Save processed requirements as JSON
        requirements_json_path = output_dir / "extracted_requirements.json"
        try:
            with open(requirements_json_path, "w", encoding="utf-8") as f:
                json.dump([req.model_dump() for req in all_requirements], f, ensure_ascii=False, indent=2)
            logger.info(f"Saved processed requirements to {requirements_json_path}")
        except Exception as e:
            logger.error(f"Failed to save requirements JSON: {e}")
        
        # Generate Excel output path
        excel_filename = f"extracted_requirements_{md_file.stem}.xlsx"
        excel_output_path = output_dir / excel_filename
        
        # Export to Excel (only if we have requirements)
        if not all_requirements.is_empty():
            try:
                logger.info(f"Exporting requirements to Excel: {excel_filename}")
                write_to_excel(requirement_list=all_requirements, output_path=str(excel_output_path))
                logger.info(f"Successfully exported {len(all_requirements)} requirements to {excel_output_path}")
            except Exception as e:
                logger.error(f"Failed to export to Excel: {e}")
                print(f"❌ Failed to export to Excel: {str(e)}")
        
        # Print summary for user
        print(f"\n✅ Processed {md_file.name}:")
        print(f"  - Processed {len(docs)} document chunks")
        print(f"  - Extracted {len(all_requirements)} total requirements")
        if not all_requirements.is_empty():
            print(f"  - Exported to: {excel_output_path}")
            codes = [code for code in all_requirements.get_codes() if code]
            if codes:
                print(f"  - Sample requirement codes: {', '.join(codes[:5])}")
                if len(codes) > 5:
                    print(f"    ... and {len(codes) - 5} more")
        else:
            print("  - No requirements found in the document")
        
    except Exception as e:
        logger.error(f"Failed to process {md_file.name}: {e}")
        print(f"\n❌ Failed to process {md_file.name}: {str(e)}")


def main():
    """Try to extract the requirements passing the md file"""
    md_file = Path("data/test/examples.md")
    with open(md_file, 'r') as file:
        content = file.read()

    task = """
            Your task is to identify and extract all requirement entries from the provided text in the attached document. A requirement consists of:
                1.  A unique code (e.g., REQ-123, HAA-54).
                2.  A description detailing what the system must do.

                [Output Instructions]
                1.  Produce a single, valid JSON array of objects.
                2.  Each object in the array must contain two keys: "code" and "description".
                3.  If no requirements are found in the text, output an empty JSON array: `[]`.
                4.  Do not include any text or explanations outside of the JSON array.

            Example:

            # HAA-54 / CREATED / R : HAA shall provide acceleration measurements in three orthogonal axes.

            ## 3.2 HAA Switch On and Operating

            ## HAA-56 / CREATED / T :  The HAA full performances shall be achieved within 36 h after switch-on.

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
            """
    my_prompt = f"{task} {content}"
    ollama_client = get_client()
    response = ollama_client.get_structured_response(my_prompt, model_name="gemma3:12b")
    print(response)



if __name__ == "__main__":
    main()
    #print(json.dumps(Requirement.model_json_schema(), indent=2))