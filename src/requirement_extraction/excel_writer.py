# src/requirement_extractor/excel_writer.py
from pathlib import Path
from typing import Optional

import pandas as pd

from config.schema import OutputConfig
from data_models.requirement import RequirementList
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

class ExcelWriter:
    """Write extracted requirements to an Excel file using centralized config."""

    def __init__(self, config: OutputConfig):
        self.config = config

    def write(self, requirement_list: RequirementList, output_path: Optional[str] = None) -> None:
        if output_path is None:
            output_path = self._default_output_path()
        write_to_excel(requirement_list, output_path, config=self.config)

    def _default_output_path(self) -> str:
        output_dir = Path(self.config.directory)
        return str(output_dir / "requirements.xlsx")


def write_to_excel(
    requirement_list: RequirementList,
    output_path: Optional[str],
    config: Optional[OutputConfig] = None,
) -> None:
    """Writes the extracted requirements to an Excel file.
    
    Args:
        requirement_list: RequirementList containing requirements to export
        output_path: Path where the Excel file should be saved
        config: Centralized output configuration (optional)
        
    Raises:
        ValueError: If requirement_list is empty or output_path is invalid
        OSError: If file cannot be written due to permissions or disk space
    """
    config = config or OutputConfig()

    # Input validation
    if not isinstance(requirement_list, RequirementList):
        raise ValueError("requirement_list must be a RequirementList instance")

    if output_path is None:
        output_path = str(Path(config.directory) / "requirements.xlsx")

    if not output_path or not str(output_path).strip():
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
