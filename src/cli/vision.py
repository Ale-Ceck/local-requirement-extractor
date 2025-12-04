import json
import sys
from pathlib import Path
import concurrent.futures

# Add src to path so we can import our modules
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from utils.logging_config import setup_logger
from utils import file_operations as fo
from llm_integration.ollama_client import get_client
from requirement_extraction.excel_writer import write_to_excel
from data_models.requirement import RequirementList, Requirement

import base64
import io
import os
import fitz
from PIL import Image

def pdf_to_base64_images(pdf_path: str, output_dir: str = None):
    """
    Converts each page of a PDF into a base64-encoded PNG image.
    Optionally saves each page as a PNG file.

    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str, optional): Directory to save PNG images. If None, images are not saved.

    Returns:
        list[str]: A list of base64-encoded PNG strings, one per page.
    """
    pdf_document = fitz.open(pdf_path)
    base64_images = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Save to buffer for base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(base64_str)

        # Optionally save as PNG file
        if output_dir:
            output_path = os.path.join(output_dir, f"page_{page_number + 1}.png")
            img.save(output_path, format="PNG")

    return base64_images



# Get a logger instance
logger = setup_logger(__name__)

# Instantiate the OllamaClient
ollama_client = get_client()

# Define paths
input_dir = Path("data/test")
output_dir = Path("data/test/vision")

prompt = """Your task is to identify and extract all requirement entries from the provided text in the attached document. A requirement consists of:
    1.  A unique code (e.g., REQ-123, HAA-54).
    2.  A description detailing what the system must do. Include relative notes and tables.

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
    ]"""



def call_llm_with_image(image_path: str, pdf_name: str, idx: int):
    print(f"Sending request for {pdf_name} - image {idx+1}: {image_path}")

    for image in image_path:
        with open(image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = ollama_client.get_structured_response(prompt=prompt, model_name="gemma3:12b",images=image_path) #[image_b64])
    
    return {
        "pdf": pdf_name,
        "page_index": idx + 1,
        "image": str(image_path),
        "response": response
    }


def process_pdf(pdf_path: Path, output_dir: Path):
    pdf_name = pdf_path.stem
    pdf_dir = output_dir / pdf_name
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Check if folder is empty → convert if needed
    if not any(pdf_dir.iterdir()):
        print(f"Converting {pdf_name} to images...")
        images = []
        images = pdf_to_base64_images(pdf_path=pdf_path, output_dir=pdf_dir)
    else:
        print(f"Skipping conversion for {pdf_name}, images already exist.")

    # Collect images
    image_files = sorted(pdf_dir.glob("*.png"))
    if not image_files:
        print(f"No images found for {pdf_name}, skipping LLM requests.")
        return

    results = []
    """
#prova
    response = ollama_client.get_structured_response(prompt=prompt, model_name="gemma3:12b",images=image_files)

    answer_file_path = pdf_dir / "raw_responses.json"
    try:
        with open(answer_file_path, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved raw LLM responses to {answer_file_path}")
    except Exception as e:
        logger.error(f"Failed to save raw responses file: {e}")

"""
    # Process in batches of 4
    for batch_start in range(0, len(image_files), 2):
        batch = image_files[batch_start: batch_start + 2]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(call_llm_with_image, batch, pdf_name, batch_start)
                #for i, img in enumerate(batch)
            ]
            batch_results = [f.result() for f in futures]
            results.extend(batch_results)
    # Save responses in JSONL format
    responses_file = pdf_dir / "responses.jsonl"
    with open(responses_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Finished processing {pdf_name}, saved {len(results)} responses → {responses_file}")
    return results



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
    

def export_excel():
    # Generate Excel output path
    excel_filename = f"extracted_requirements_examples.xlsx"
    excel_output_path = output_dir / excel_filename

    answer_file_path = output_dir / "raw_responses.json"
    with open(answer_file_path, 'r') as file:
        answer = json.load(file)

    all_requirements = parse_llm_response(answer)
    
    # Export to Excel (only if we have requirements)
    if not all_requirements.is_empty():
        try:
            logger.info(f"Exporting requirements to Excel: {excel_filename}")
            write_to_excel(requirement_list=all_requirements, output_path=str(excel_output_path))
            logger.info(f"Successfully exported {len(all_requirements)} requirements to {excel_output_path}")
        except Exception as e:
            logger.error(f"Failed to export to Excel: {e}")
            print(f"❌ Failed to export to Excel: {str(e)}")

    
def main():
    if not fo.dir_exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    if not fo.dir_exists(output_dir):
        logger.error(f"Output directory not found: {output_dir}")
        return
    # Find all PDF files in input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in data/input directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to convert")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            process_pdf(pdf_file, output_dir)
          
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
            print(f"\n✗ Failed to process {pdf_file.name}: {str(e)}")



if __name__ == "__main__":
    main()


