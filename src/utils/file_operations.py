# General utility functions for interacting with the file system (e.g., checking if a file exists, creating directories, listing files).

from pathlib import Path
from typing import List, Optional
import shutil
from utils.logging_config import setup_logger

logger = setup_logger("file_operations")

def file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    exists = Path(filepath).is_file()
    logger.debug(f"Checked if file exists: {filepath} -> {exists}")
    return exists

def dir_exists(dirpath: str) -> bool:
    """Check if a directory exists."""
    exists = Path(dirpath).is_dir()
    logger.debug(f"Checked if directory exists: {dirpath} -> {exists}")
    return exists

def create_dir(dirpath: str) -> None:
    """Create a directory and any missing parent directories."""
    path = Path(dirpath)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dirpath}")
    else:
        logger.debug(f"Directory already exists: {dirpath}")

def list_files(dirpath: str, extension: Optional[str] = None) -> List[str]:
    """List all files in a directory, optionally filtering by extension."""
    path = Path(dirpath)
    if not path.exists():
        logger.warning(f"Directory does not exist: {dirpath}")
        return []
    
    if extension:
        files = [str(f) for f in path.glob(f'*.{extension.lstrip(".")}')]
    else:
        files = [str(f) for f in path.iterdir() if f.is_file()]
    
    logger.debug(f"Listed files in {dirpath} with extension={extension}: {files}")
    return files

def delete_file(filepath: str) -> None:
    """Delete a file if it exists."""
    path = Path(filepath)
    if path.is_file():
        path.unlink()
        logger.info(f"Deleted file: {filepath}")
    else:
        logger.debug(f"File not found for deletion: {filepath}")

def copy_file(src: str, dst: str) -> None:
    """Copy a file from src to dst."""
    shutil.copy2(src, dst)
    logger.info(f"Copied file from {src} to {dst}")
