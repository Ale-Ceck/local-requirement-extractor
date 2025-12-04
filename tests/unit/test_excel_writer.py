import pytest
import pandas as pd
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import sys
import tempfile
import os

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from requirement_extraction.excel_writer import write_to_excel   # type: ignore
from data_models.requirement import Requirement, RequirementList # type: ignore
from pydantic import ValidationError


class TestExcelWriter:
    """Test cases for Excel writer functionality."""
    
    @pytest.fixture
    def single_requirement_list(self):
        """Create a RequirementList with a single requirement."""
        return RequirementList([
            Requirement(code="REQ-001", description="System must authenticate users")
        ])
    
    @pytest.fixture
    def multiple_requirements_list(self):
        """Create a RequirementList with multiple requirements."""
        return RequirementList([
            Requirement(code="REQ-001", description="System must authenticate users"),
            Requirement(code="REQ-002", description="System must store user data securely"),
            Requirement(code="REQ-003", description="System must provide audit logging")
        ])
    
    @pytest.fixture
    def empty_requirement_list(self):
        """Create an empty RequirementList."""
        return RequirementList([])
    
    def test_write_single_requirement_to_excel(self, single_requirement_list):
        """Test writing a single requirement to Excel."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                write_to_excel(single_requirement_list, tmp_file.name)
                
                # Verify file was created
                assert os.path.exists(tmp_file.name)
                
                # Verify file contents
                df = pd.read_excel(tmp_file.name)
                assert len(df) == 1
                assert df.iloc[0]['Requirement Code'] == "REQ-001"
                assert df.iloc[0]['Description'] == "System must authenticate users"
                assert list(df.columns) == ["Requirement Code", "Description"]
                
            finally:
                # Cleanup
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_write_multiple_requirements_to_excel(self, multiple_requirements_list):
        """Test writing multiple requirements to Excel."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                write_to_excel(multiple_requirements_list, tmp_file.name)
                
                # Verify file was created
                assert os.path.exists(tmp_file.name)
                
                # Verify file contents
                df = pd.read_excel(tmp_file.name)
                assert len(df) == 3
                assert df.iloc[0]['Requirement Code'] == "REQ-001"
                assert df.iloc[1]['Requirement Code'] == "REQ-002" 
                assert df.iloc[2]['Requirement Code'] == "REQ-003"
                assert df.iloc[0]['Description'] == "System must authenticate users"
                assert df.iloc[1]['Description'] == "System must store user data securely"
                assert df.iloc[2]['Description'] == "System must provide audit logging"
                
            finally:
                # Cleanup
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_write_empty_requirement_list_to_excel(self, empty_requirement_list):
        """Test writing an empty requirement list to Excel."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                write_to_excel(empty_requirement_list, tmp_file.name)
                
                # Verify file was created
                assert os.path.exists(tmp_file.name)
                
                # Verify file contents (should have headers but no data)
                df = pd.read_excel(tmp_file.name)
                assert len(df) == 0
                assert list(df.columns) == ["Requirement Code", "Description"]
                
            finally:
                # Cleanup
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_output_directory_creation(self, single_requirement_list):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a nested path that doesn't exist
            output_path = os.path.join(tmp_dir, "new_dir", "subdir", "output.xlsx")
            
            write_to_excel(single_requirement_list, output_path)
            
            # Verify file was created and directory structure exists
            assert os.path.exists(output_path)
            assert os.path.exists(os.path.dirname(output_path))
    
    def test_invalid_requirement_list_type(self):
        """Test error handling for invalid requirement list type."""
        with pytest.raises(ValueError) as exc_info:
            write_to_excel("not a requirement list", "output.xlsx")
        
        assert "requirement_list must be a RequirementList instance" in str(exc_info.value)
    
    def test_empty_output_path(self, single_requirement_list):
        """Test error handling for empty output path."""
        with pytest.raises(ValueError) as exc_info:
            write_to_excel(single_requirement_list, "")
        
        assert "output_path cannot be empty" in str(exc_info.value)
    
    def test_whitespace_only_output_path(self, single_requirement_list):
        """Test error handling for whitespace-only output path."""
        with pytest.raises(ValueError) as exc_info:
            write_to_excel(single_requirement_list, "   ")
        
        assert "output_path cannot be empty" in str(exc_info.value)
    
    def test_none_output_path(self, single_requirement_list):
        """Test error handling for None output path."""
        with pytest.raises(ValueError) as exc_info:
            write_to_excel(single_requirement_list, None)
        
        assert "output_path cannot be empty" in str(exc_info.value)
    
    @patch('requirement_extraction.excel_writer.logger')
    def test_logging_success(self, mock_logger, single_requirement_list):
        """Test that success is logged correctly."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                write_to_excel(single_requirement_list, tmp_file.name)
                
                # Verify info log was called
                mock_logger.info.assert_called()
                info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
                assert any("Successfully wrote 1 requirements" in call for call in info_calls)
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    @patch('requirement_extraction.excel_writer.logger')
    def test_logging_empty_list_warning(self, mock_logger, empty_requirement_list):
        """Test that warning is logged for empty requirement list."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                write_to_excel(empty_requirement_list, tmp_file.name)
                
                # Verify warning log was called
                mock_logger.warning.assert_called_once_with(
                    "RequirementList is empty, creating Excel file with headers only"
                )
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    @patch('pandas.DataFrame.to_excel')
    @patch('requirement_extraction.excel_writer.logger')
    def test_error_handling_file_write_failure(self, mock_logger, mock_to_excel, single_requirement_list):
        """Test error handling when Excel file writing fails."""
        # Mock pandas to_excel to raise an exception
        mock_to_excel.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(OSError) as exc_info:
            write_to_excel(single_requirement_list, "output.xlsx")
        
        assert "Failed to write Excel file" in str(exc_info.value)
        assert "Permission denied" in str(exc_info.value)
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert "Failed to write Excel file to output.xlsx" in error_call
    
    def test_requirement_data_integrity(self, multiple_requirements_list):
        """Test that requirement data is preserved correctly in Excel."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                write_to_excel(multiple_requirements_list, tmp_file.name)
                
                # Read back the Excel file
                df = pd.read_excel(tmp_file.name)
                
                # Verify all original requirements are present
                original_codes = [req.code for req in multiple_requirements_list]
                original_descriptions = [req.description for req in multiple_requirements_list]
                
                excel_codes = df['Requirement Code'].tolist()
                excel_descriptions = df['Description'].tolist()
                
                assert excel_codes == original_codes
                assert excel_descriptions == original_descriptions
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_special_characters_in_requirements(self):
        """Test handling of special characters in requirement text."""
        requirements_with_special_chars = RequirementList([
            Requirement(code="REQ-001", description="System must handle 'quotes' and \"double quotes\""),
            Requirement(code="REQ-002", description="System must process Unicode: é, ñ, 中文"),
            Requirement(code="REQ-003", description="System must handle newlines\nand tabs\t")
        ])
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                write_to_excel(requirements_with_special_chars, tmp_file.name)
                
                # Verify file was created and data preserved
                df = pd.read_excel(tmp_file.name)
                assert len(df) == 3
                assert "quotes" in df.iloc[0]['Description']
                assert "Unicode" in df.iloc[1]['Description']
                assert "newlines" in df.iloc[2]['Description']
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


class TestExcelWriterIntegration:
    """Integration tests for Excel writer with real RequirementList operations."""
    
    def test_write_after_requirement_list_merge(self):
        """Test writing Excel after merging RequirementLists."""
        list1 = RequirementList([
            Requirement(code="REQ-001", description="First requirement")
        ])
        list2 = RequirementList([
            Requirement(code="REQ-002", description="Second requirement")
        ])
        
        merged_list = list1.merge(list2)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            try:
                write_to_excel(merged_list, tmp_file.name)
                
                df = pd.read_excel(tmp_file.name)
                assert len(df) == 2
                assert df.iloc[0]['Requirement Code'] == "REQ-001"
                assert df.iloc[1]['Requirement Code'] == "REQ-002"
                
            finally:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)