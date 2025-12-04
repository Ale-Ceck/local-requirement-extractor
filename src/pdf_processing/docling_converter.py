import pathlib
import time

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from utils.logging_config import setup_logger
from typing import Optional

logger = setup_logger(__name__)


def convert_pdf_to_markdown(pdf_path: str, output_dir: Optional[str] = "data/docling_markdown_output") -> str:
    """
    Convert PDF to Markdown format using docling library.
    
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

        artifacts_path = "/Users/ceck/.cache/docling/models"
        
        # Convert PDF to markdown

        pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["en"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.AUTO
        )

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        start_time = time.time()
        conv_result = doc_converter.convert(pdf_path_obj)
        end_time = time.time() - start_time

        logger.info(f"Document {pdf_path_obj.stem} converted in {end_time:.2f} seconds.")
        md_text = conv_result.document.export_to_markdown(image_placeholder="", include_annotations=False)
        
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
