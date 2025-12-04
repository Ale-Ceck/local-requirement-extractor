#!/usr/bin/env python3
"""
Find all Markdown files in input directory and extract all requirements for each file and export to Excel format.
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from utils.logging_config import setup_logger
from utils import file_operations as fo
from llm_integration.ollama_client import get_client
from requirement_extraction.requirement_extractor import extract_requirements_from_markdown
from requirement_extraction.excel_writer import write_to_excel

# Get a logger instance
logger = setup_logger(__name__)


def extractor():
    """Extract requirements from all Markdown files in data/input and export to Excel format."""

    # Instantiate the OllamaClient
    ollama_client = get_client()
    
    # Define paths
    input_dir = Path("data/docling_markdown_output")
    output_dir = Path("data/docling_output")
    
    if not fo.dir_exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    if not fo.dir_exists(output_dir):
        logger.error(f"Output directory not found: {output_dir}")
        return
    
    # Find all Markdown files in input directory
    md_files = list(input_dir.glob("*.md"))
    
    if not md_files:
        logger.warning("No Markdown files found in data/docling_markdown_output directory")
        return
    
    logger.info(f"Found {len(md_files)} Markdown file(s) to convert")
    
    # Process each Markdown file
    for md_file in md_files:
        try:
            logger.info(f"Processing: {md_file.name}")
            
            # Extract requirements from Markdown
            requirements = extract_requirements_from_markdown(str(md_file))
            logger.info(f"Successfully extracted {len(requirements)} requirements from {md_file.name}")
            
            # Generate Excel output path
            excel_filename = md_file.stem + "_requirements.xlsx"
            excel_output_path = output_dir / excel_filename
            
            # Export to Excel
            logger.info(f"Exporting requirements to Excel: {excel_filename}")
            write_to_excel(requirement_list=requirements, output_path=str(excel_output_path))
            logger.info(f"Successfully exported {len(requirements)} requirements to {excel_output_path}")
            
            # Print summary for user
            print(f"\n✓ Processed {md_file.name}:")
            print(f"  - Extracted {len(requirements)} requirements")
            print(f"  - Exported to: {excel_output_path}")
            if not requirements.is_empty():
                print(f"  - Requirement codes: {', '.join(requirements.get_codes())}")
            
        except Exception as e:
            logger.error(f"Failed to process {md_file.name}: {e}")
            print(f"\n✗ Failed to process {md_file.name}: {str(e)}")

    

if __name__ == "__main__":
    extractor()