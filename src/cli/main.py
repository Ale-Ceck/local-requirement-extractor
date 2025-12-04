#!/usr/bin/env python3
"""
Find all PDF files in input directory and extract requirements from each file and export to Excel format.
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from utils.logging_config import setup_logger
from utils import file_operations as fo
from llm_integration.ollama_client import get_client
from requirement_extraction.requirement_extractor import extract_requirements_from_pdf
from requirement_extraction.excel_writer import write_to_excel

# Get a logger instance
logger = setup_logger("main")


def main():
    """Extract requirements from all PDFs in data/input and export to Excel format."""

    # Instantiate the OllamaClient
    ollama_client = get_client()
    
    # Define paths
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    
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
            
            # Extract requirements from PDF
            requirements = extract_requirements_from_pdf(str(pdf_file))
            logger.info(f"Successfully extracted {len(requirements)} requirements from {pdf_file.name}")
            
            # Generate Excel output path
            excel_filename = pdf_file.stem + "_requirements.xlsx"
            excel_output_path = output_dir / excel_filename
            
            # Export to Excel
            logger.info(f"Exporting requirements to Excel: {excel_filename}")
            write_to_excel(requirement_list=requirements, output_path=str(excel_output_path))
            logger.info(f"Successfully exported {len(requirements)} requirements to {excel_output_path}")
            
            # Print summary for user
            print(f"\n✓ Processed {pdf_file.name}:")
            print(f"  - Extracted {len(requirements)} requirements")
            print(f"  - Exported to: {excel_output_path}")
            if not requirements.is_empty():
                print(f"  - Requirement codes: {', '.join(requirements.get_codes())}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
            print(f"\n✗ Failed to process {pdf_file.name}: {str(e)}")

    

if __name__ == "__main__":
    main()