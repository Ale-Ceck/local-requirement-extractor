import pymupdf4llm
import pathlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(pdf_path: str, output_dir: Optional[str] = "data/markdown_output") -> str:
    """
    Convert PDF to Markdown format using pymupdf4llm library.
    
    Args:
        pdf_path: Path to the PDF file to convert
        output_dir: Directory to save the markdown file (optional)
    
    Returns:
        Path to the generated markdown file
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the PDF path is invalid
        Exception: For other conversion errors
    """
    try:
        # Validate input
        pdf_path_obj = pathlib.Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path_obj.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        logger.info(f"Starting PDF to Markdown conversion for: {pdf_path}")
        
        # Convert PDF to markdown
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Determine output path
        output_dir_path = pathlib.Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / f"{pdf_path_obj.stem}.md"
        
        # Write markdown to file
        output_path.write_text(md_text, encoding='utf-8')
        
        logger.info(f"Successfully converted PDF to Markdown: {output_path}")
        return str(output_path)
        
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise
    except Exception as e:
        logger.error(f"Error converting PDF to Markdown: {e}")
        raise