# src/requirement_extractor/excel_writer.py
import pandas as pd
from pathlib import Path
from data_models.requirement import RequirementList
from utils.logging_config import setup_logger

# Setup logger
logger = setup_logger(__name__)

def write_to_excel(requirement_list: RequirementList, output_path: str):
    """Writes the extracted requirements to an Excel file.
    
    Args:
        requirement_list: RequirementList containing requirements to export
        output_path: Path where the Excel file should be saved
        
    Raises:
        ValueError: If requirement_list is empty or output_path is invalid
        OSError: If file cannot be written due to permissions or disk space
    """
    # Input validation
    if not isinstance(requirement_list, RequirementList):
        raise ValueError("requirement_list must be a RequirementList instance")
    
    if not output_path or not output_path.strip():
        raise ValueError("output_path cannot be empty")
    
    if requirement_list.is_empty():
        logger.warning("RequirementList is empty, creating Excel file with headers only")
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensuring output directory exists: {output_dir}")
    
    try:
        # Extract data from RequirementList (iterate directly over the list)
        data = {
            "Requirement Code": [req.code for req in requirement_list],
            "Description": [req.description for req in requirement_list],
        }
        
        # Create DataFrame and write to Excel
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)
        
        logger.info(f"Successfully wrote {len(requirement_list)} requirements to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to write Excel file to {output_path}: {str(e)}")
        raise OSError(f"Failed to write Excel file: {str(e)}") from e