#!/usr/bin/env python3
"""
Tool that converts pdf files in markdown format
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from utils.logging_config import setup_logger
from utils import file_operations as fo
#from pdf_processing.pdf_to_markdown import convert_pdf_to_markdown
from pdf_processing.docling_converter import convert_pdf_to_markdown


# Get a logger instance
logger = setup_logger("converter")


def converter():
    """Extract requirements from all PDFs in data/input and export to Excel format."""

    # Define paths
    input_dir = Path("data/test")
    #markdown_dir = Path("data/markdown_output")
    #markdown_dir = Path("data/docling_markdown_output")
    markdown_dir = input_dir
    
    if not fo.dir_exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    if not fo.dir_exists(markdown_dir):
        logger.error(f"Output directory not found: {markdown_dir}")
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
            #Convert PDF to markdown
            markdown_path = convert_pdf_to_markdown(pdf_file, markdown_dir)
            logger.info(f"PDF converted to markdown:{markdown_path}")

        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
            print(f"\n✗ Failed to process {pdf_file.name}: {str(e)}")

    


if __name__ == "__main__":
    converter()