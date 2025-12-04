import pytest
import json
import sys
import concurrent.futures
import time
from unittest.mock import Mock, patch, mock_open, MagicMock, ANY
from pathlib import Path
from pydantic import ValidationError

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from requirement_extraction.requirement_extractor import RequirementExtractor, extract_requirements_from_pdf # type: ignore
from data_models.requirement import Requirement, RequirementList # type: ignore


class TestRequirementExtractor:
    """Test cases for RequirementExtractor class."""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client for testing."""
        with patch('requirement_extraction.requirement_extractor.get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def mock_pdf_conversion(self):
        """Mock PDF to markdown conversion."""
        with patch('requirement_extraction.requirement_extractor.convert_pdf_to_markdown') as mock_convert:
            mock_convert.return_value = "test_output.md"
            yield mock_convert
    
    @pytest.fixture
    def mock_prompt_template(self):
        """Mock prompt template generation."""
        with patch('requirement_extraction.requirement_extractor.get_prompt') as mock_get_prompt:
            mock_get_prompt.return_value = "Test prompt"
            yield mock_get_prompt
    
    @pytest.fixture
    def extractor(self, mock_ollama_client):
        """Create a RequirementExtractor instance with mocked dependencies."""
        return RequirementExtractor(model_name="test-model")
    
    def test_init(self, mock_ollama_client):
        """Test RequirementExtractor initialization."""
        extractor = RequirementExtractor(model_name="custom-model")
        assert extractor.model_name == "custom-model"
        assert extractor.ollama_client == mock_ollama_client
    
    def test_init_default_model(self, mock_ollama_client):
        """Test RequirementExtractor initialization with default model."""
        extractor = RequirementExtractor()
        assert extractor.model_name == "mistral:7b"
        assert extractor.max_workers == 3  # Default value
    
    def test_init_with_max_workers(self, mock_ollama_client):
        """Test RequirementExtractor initialization with custom max_workers."""
        extractor = RequirementExtractor(model_name="test-model", max_workers=5)
        assert extractor.model_name == "test-model"
        assert extractor.max_workers == 5
        assert extractor.ollama_client == mock_ollama_client
    
    @patch('builtins.open', new_callable=mock_open, read_data="# Test Document\nSome markdown content")
    @patch('requirement_extraction.requirement_extractor.MarkdownHeaderTextSplitter')
    def test_extract_requirements_empty_response(self, mock_splitter, mock_file, extractor, 
                                                mock_pdf_conversion, mock_prompt_template, mock_ollama_client):
        """Test case 1: Empty response from LLM."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_splitter.return_value.split_text.return_value = [mock_doc]
        mock_ollama_client.get_structured_response.return_value = "[]"
        
        # Test
        result = extractor.extract_requirements_from_pdf("test.pdf")
        
        # Assertions
        assert isinstance(result, RequirementList)
        assert len(result.root) == 0
        assert result.is_empty()
        mock_pdf_conversion.assert_called_once_with("test.pdf", None)
        mock_ollama_client.get_structured_response.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, read_data="# Test Document\nSome markdown content")
    @patch('requirement_extraction.requirement_extractor.MarkdownHeaderTextSplitter')
    def test_extract_requirements_single_requirement(self, mock_splitter, mock_file, extractor,
                                                   mock_pdf_conversion, mock_prompt_template, mock_ollama_client):
        """Test case 2: Single requirement response from LLM."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.page_content = "REQ-001: System must authenticate users"
        mock_splitter.return_value.split_text.return_value = [mock_doc]
        
        single_req_response = json.dumps([{
            "code": "REQ-001",
            "description": "System must authenticate users"
        }])
        mock_ollama_client.get_structured_response.return_value = single_req_response
        
        # Test
        result = extractor.extract_requirements_from_pdf("test.pdf")
        
        # Assertions
        assert isinstance(result, RequirementList)
        assert len(result.root) == 1
        assert result.root[0].code == "REQ-001"
        assert result.root[0].description == "System must authenticate users"
        assert not result.is_empty()
    
    @patch('builtins.open', new_callable=mock_open, read_data="# Test Document\nSome markdown content")
    @patch('requirement_extraction.requirement_extractor.MarkdownHeaderTextSplitter')
    def test_extract_requirements_multiple_requirements(self, mock_splitter, mock_file, extractor,
                                                      mock_pdf_conversion, mock_prompt_template, mock_ollama_client):
        """Test case 3: Multiple requirements response from LLM."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.page_content = "REQ-001: Auth requirement\nREQ-002: Data requirement"
        mock_splitter.return_value.split_text.return_value = [mock_doc]
        
        multiple_req_response = json.dumps([
            {
                "code": "REQ-001",
                "description": "System must authenticate users"
            },
            {
                "code": "REQ-002", 
                "description": "System must store user data securely"
            },
            {
                "code": "REQ-003",
                "description": "System must provide audit logging"
            }
        ])
        mock_ollama_client.get_structured_response.return_value = multiple_req_response
        
        # Test
        result = extractor.extract_requirements_from_pdf("test.pdf")
        
        # Assertions
        assert isinstance(result, RequirementList)
        assert len(result.root) == 3
        assert result.root[0].code == "REQ-001"
        assert result.root[1].code == "REQ-002"
        assert result.root[2].code == "REQ-003"
        assert result.get_codes() == ["REQ-001", "REQ-002", "REQ-003"]
    
    @patch('builtins.open', new_callable=mock_open, read_data="# Test Document\nSome markdown content")
    @patch('requirement_extraction.requirement_extractor.MarkdownHeaderTextSplitter')
    def test_extract_requirements_none_response(self, mock_splitter, mock_file, extractor,
                                              mock_pdf_conversion, mock_prompt_template, mock_ollama_client):
        """Test case for None response from LLM."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_splitter.return_value.split_text.return_value = [mock_doc]
        mock_ollama_client.get_structured_response.return_value = None
        
        # Test
        result = extractor.extract_requirements_from_pdf("test.pdf")
        
        # Assertions - should return empty RequirementList when response is None
        assert isinstance(result, RequirementList)
        assert len(result.root) == 0
        assert result.is_empty()
    
    @patch('builtins.open', new_callable=mock_open, read_data="# Test Document\nSome markdown content")
    @patch('requirement_extraction.requirement_extractor.MarkdownHeaderTextSplitter')
    def test_extract_requirements_error_handling(self, mock_splitter, mock_file, extractor,
                                                mock_pdf_conversion, mock_prompt_template, mock_ollama_client):
        """Test case 5: Error handling during extraction."""
        # Setup mocks to raise exception
        mock_pdf_conversion.side_effect = Exception("PDF conversion failed")
        
        # Test
        with pytest.raises(Exception) as exc_info:
            extractor.extract_requirements_from_pdf("test.pdf")
        
        # Assertions
        assert "PDF conversion failed" in str(exc_info.value)
    
    @patch('builtins.open', new_callable=mock_open, read_data="# Test Document\nSome markdown content")
    @patch('requirement_extraction.requirement_extractor.MarkdownHeaderTextSplitter')
    def test_extract_requirements_json_validation_error(self, mock_splitter, mock_file, extractor,
                                                       mock_pdf_conversion, mock_prompt_template, mock_ollama_client):
        """Test error handling for invalid JSON response - now handled gracefully."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_splitter.return_value.split_text.return_value = [mock_doc]
        
        # Invalid JSON response
        mock_ollama_client.get_structured_response.return_value = "invalid json"
        
        # Test - with concurrent processing, invalid JSON is handled gracefully
        result = extractor.extract_requirements_from_pdf("test.pdf")
        
        # Assertions - should return empty list instead of raising exception
        assert isinstance(result, RequirementList)
        assert len(result.root) == 0
        assert result.is_empty()


class TestRequirementExtractorResponseParsing:
    """Test different LLM response formats - Test case 4."""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client for testing."""
        with patch('requirement_extraction.requirement_extractor.get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def extractor(self, mock_ollama_client):
        """Create a RequirementExtractor instance for testing."""
        return RequirementExtractor(model_name="test-model")
    
    def test_parse_llm_response_list_format(self, extractor):
        """Test parsing when LLM returns direct list format."""
        json_response = json.dumps([
            {"code": "REQ-001", "description": "Test requirement 1"},
            {"code": "REQ-002", "description": "Test requirement 2"}
        ])
        
        result = extractor.parse_llm_response(json_response)
        
        assert len(result.root) == 2
        assert result.root[0].code == "REQ-001"
        assert result.root[1].code == "REQ-002"
    
    def test_parse_llm_response_requirements_wrapper(self, extractor):
        """Test parsing when LLM returns {"requirements": [...]} format."""
        json_response = json.dumps({
            "requirements": [
                {"code": "REQ-001", "description": "Test requirement 1"}
            ]
        })
        
        result = extractor.parse_llm_response(json_response)
        
        assert len(result.root) == 1
        assert result.root[0].code == "REQ-001"
    
    def test_parse_llm_response_single_requirement_wrapper(self, extractor):
        """Test parsing when LLM returns {"requirement": {...}} format."""
        json_response = json.dumps({
            "requirement": {"code": "REQ-001", "description": "Test requirement 1"}
        })
        
        result = extractor.parse_llm_response(json_response)
        
        assert len(result.root) == 1
        assert result.root[0].code == "REQ-001"
    
    def test_parse_llm_response_direct_object(self, extractor):
        """Test parsing when LLM returns direct object with code/description."""
        json_response = json.dumps({
            "code": "REQ-001", 
            "description": "Test requirement 1"
        })
        
        result = extractor.parse_llm_response(json_response)
        
        assert len(result.root) == 1
        assert result.root[0].code == "REQ-001"
    
    def test_parse_llm_response_empty_dict(self, extractor):
        """Test parsing when LLM returns empty dictionary."""
        json_response = json.dumps({})
        
        result = extractor.parse_llm_response(json_response)
        
        assert len(result.root) == 0
        assert result.is_empty()
    
    def test_parse_llm_response_invalid_json(self, extractor):
        """Test error handling for invalid JSON."""
        json_response = "invalid json string"
        
        with pytest.raises(ValueError) as exc_info:
            extractor.parse_llm_response(json_response)
        
        assert "Invalid JSON from LLM" in str(exc_info.value)
    
    def test_parse_llm_response_non_list_non_dict(self, extractor):
        """Test parsing when LLM returns non-list, non-dict format."""
        json_response = json.dumps("string response")
        
        result = extractor.parse_llm_response(json_response)
        
        assert len(result.root) == 0
        assert result.is_empty()


class TestConvenienceFunction:
    """Test the convenience function extract_requirements_from_pdf."""
    
    @patch('requirement_extraction.requirement_extractor.RequirementExtractor')
    def test_extract_requirements_from_pdf_function(self, mock_extractor_class):
        """Test the convenience function calls extractor correctly."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_requirements_from_pdf.return_value = RequirementList([])
        
        # Test with default parameters
        result = extract_requirements_from_pdf("test.pdf")
        
        # Assertions - now includes max_workers parameter
        mock_extractor_class.assert_called_once_with(
            model_name="DeepSeek-R1-Distill-Llama-8B-Q4_K_M:latest",
            max_workers=3
        )
        mock_extractor.extract_requirements_from_pdf.assert_called_once_with("test.pdf", "data/markdown_output")
    
    @patch('requirement_extraction.requirement_extractor.RequirementExtractor')
    def test_extract_requirements_from_pdf_function_custom_params(self, mock_extractor_class):
        """Test the convenience function with custom parameters."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_requirements_from_pdf.return_value = RequirementList([])
        
        # Test with custom parameters
        result = extract_requirements_from_pdf("test.pdf", model_name="custom-model", markdown_dir="custom/dir")
        
        # Assertions - now includes max_workers parameter
        mock_extractor_class.assert_called_once_with(
            model_name="custom-model",
            max_workers=3
        )
        mock_extractor.extract_requirements_from_pdf.assert_called_once_with("test.pdf", "custom/dir")


class TestValidationAndEdgeCases:
    """Test validation and edge cases."""
    
    def test_requirement_list_unique_codes_validation(self):
        """Test that RequirementList enforces unique codes."""
        # This should raise a validation error due to duplicate codes
        with pytest.raises(ValidationError):
            RequirementList([
                Requirement(code="REQ-001", description="First requirement"),
                Requirement(code="REQ-001", description="Duplicate code requirement")
            ])
    
    def test_requirement_empty_code_validation(self):
        """Test that Requirement validates non-empty code."""
        with pytest.raises(ValidationError):
            Requirement(code="", description="Valid description")
    
    def test_requirement_empty_description_validation(self):
        """Test that Requirement validates non-empty description."""
        with pytest.raises(ValidationError):
            Requirement(code="REQ-001", description="")
    
    def test_requirement_whitespace_trimming(self):
        """Test that Requirement trims whitespace from code and description."""
        req = Requirement(code="  REQ-001  ", description="  Test description  ")
        assert req.code == "REQ-001"
        assert req.description == "Test description"


class TestConcurrentProcessing:
    """Test cases for concurrent processing functionality."""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client for testing."""
        with patch('requirement_extraction.requirement_extractor.get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def mock_prompt_template(self):
        """Mock prompt template generation."""
        with patch('requirement_extraction.requirement_extractor.get_prompt') as mock_get_prompt:
            mock_get_prompt.return_value = "Test prompt"
            yield mock_get_prompt
    
    @pytest.fixture
    def extractor(self, mock_ollama_client):
        """Create a RequirementExtractor instance with mocked dependencies."""
        return RequirementExtractor(model_name="test-model", max_workers=2)
    
    def test_process_single_chunk_success(self, extractor, mock_prompt_template, mock_ollama_client):
        """Test successful processing of a single chunk."""
        # Setup mock document chunk
        mock_doc = Mock()
        mock_doc.page_content = "REQ-001: Test requirement"
        
        # Setup mock response
        llm_response = json.dumps([{
            "code": "REQ-001",
            "description": "Test requirement"
        }])
        mock_ollama_client.get_structured_response.return_value = llm_response
        
        # Test
        result = extractor.process_single_chunk(mock_doc, {"type": "object"})
        
        # Assertions
        assert isinstance(result, RequirementList)
        assert len(result.root) == 1
        assert result.root[0].code == "REQ-001"
        assert result.root[0].description == "Test requirement"
        mock_prompt_template.assert_called_once()
        mock_ollama_client.get_structured_response.assert_called_once()
    
    def test_process_single_chunk_llm_failure(self, extractor, mock_prompt_template, mock_ollama_client):
        """Test handling of LLM failure in single chunk processing."""
        # Setup mock document chunk
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        
        # Mock LLM returning None (failure)
        mock_ollama_client.get_structured_response.return_value = None
        
        # Test
        result = extractor.process_single_chunk(mock_doc, {"type": "object"})
        
        # Assertions
        assert isinstance(result, RequirementList)
        assert len(result.root) == 0
        assert result.is_empty()
    
    def test_process_single_chunk_exception_handling(self, extractor, mock_prompt_template, mock_ollama_client):
        """Test exception handling in single chunk processing."""
        # Setup mock document chunk
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        
        # Mock prompt template raising exception
        mock_prompt_template.side_effect = Exception("Prompt generation failed")
        
        # Test
        result = extractor.process_single_chunk(mock_doc, {"type": "object"})
        
        # Assertions
        assert isinstance(result, RequirementList)
        assert len(result.root) == 0
        assert result.is_empty()
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_process_chunks_parallel_success(self, mock_executor, extractor, mock_prompt_template, mock_ollama_client):
        """Test successful parallel processing of multiple chunks."""
        # Setup mock documents
        mock_docs = []
        for i in range(3):
            mock_doc = Mock()
            mock_doc.page_content = f"REQ-00{i+1}: Test requirement {i+1}"
            mock_docs.append(mock_doc)
        
        # Setup mock futures and results
        mock_futures = []
        mock_results = []
        for i in range(3):
            mock_future = Mock()
            mock_requirement_list = RequirementList([
                Requirement(code=f"REQ-00{i+1}", description=f"Test requirement {i+1}")
            ])
            mock_future.result.return_value = mock_requirement_list
            mock_futures.append(mock_future)
            mock_results.append(mock_requirement_list)
        
        # Setup mock executor
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.side_effect = mock_futures
        
        # Mock as_completed to return futures in order
        with patch('concurrent.futures.as_completed', return_value=mock_futures):
            # Create a mapping from futures to indices
            future_to_chunk = {mock_futures[i]: i for i in range(3)}
            
            # Mock the future_to_chunk mapping in the method
            with patch.object(extractor, 'process_single_chunk', side_effect=mock_results):
                # Test
                result = extractor.process_chunks_parallel(mock_docs, {"type": "object"})
        
        # Assertions
        assert len(result) == 3
        assert all(isinstance(req, Requirement) for req in result)
        mock_executor_instance.submit.assert_called()
        assert mock_executor_instance.submit.call_count == 3
    
    def test_process_chunks_parallel_with_failures(self, extractor, mock_prompt_template, mock_ollama_client):
        """Test parallel processing with some chunk failures."""
        # Setup mock documents
        mock_docs = []
        for i in range(3):
            mock_doc = Mock()
            mock_doc.page_content = f"Content {i+1}"
            mock_docs.append(mock_doc)
        
        # Mock process_single_chunk to simulate mixed success/failure
        def mock_process_chunk(doc, schema):
            if "Content 2" in doc.page_content:
                # Simulate failure for second chunk
                return RequirementList([])
            else:
                # Success for other chunks
                content_num = doc.page_content.split()[-1]
                return RequirementList([
                    Requirement(code=f"REQ-00{content_num}", description=f"Test requirement {content_num}")
                ])
        
        with patch.object(extractor, 'process_single_chunk', side_effect=mock_process_chunk):
            # Test
            result = extractor.process_chunks_parallel(mock_docs, {"type": "object"})
        
        # Assertions - should have 2 requirements (chunk 2 failed)
        assert len(result) == 2
        codes = [req.code for req in result]
        assert "REQ-001" in codes
        assert "REQ-003" in codes
    
    def test_process_chunks_parallel_empty_docs(self, extractor):
        """Test parallel processing with empty document list."""
        # Test
        result = extractor.process_chunks_parallel([], {"type": "object"})
        
        # Assertions
        assert isinstance(result, list)
        assert len(result) == 0
    
    @patch('builtins.open', new_callable=mock_open, read_data="# Test Document\nSome markdown content")
    @patch('requirement_extraction.requirement_extractor.MarkdownHeaderTextSplitter')
    @patch('requirement_extraction.requirement_extractor.convert_pdf_to_markdown')
    def test_extract_requirements_uses_parallel_processing(self, mock_pdf_conversion, mock_splitter, 
                                                         mock_file, extractor, mock_prompt_template, mock_ollama_client):
        """Test that extract_requirements_from_pdf uses parallel processing."""
        # Setup mocks
        mock_pdf_conversion.return_value = "test.md"
        
        mock_docs = []
        for i in range(2):
            mock_doc = Mock()
            mock_doc.page_content = f"REQ-00{i+1}: Test requirement {i+1}"
            mock_docs.append(mock_doc)
        
        mock_splitter.return_value.split_text.return_value = mock_docs
        
        # Mock parallel processing to return known results
        expected_requirements = [
            Requirement(code="REQ-001", description="Test requirement 1"),
            Requirement(code="REQ-002", description="Test requirement 2")
        ]
        
        with patch.object(extractor, 'process_chunks_parallel', return_value=expected_requirements) as mock_parallel:
            # Test
            result = extractor.extract_requirements_from_pdf("test.pdf")
            
            # Assertions
            assert isinstance(result, RequirementList)
            assert len(result.root) == 2
            mock_parallel.assert_called_once_with(mock_docs, ANY)  # ANY for schema
    
    def test_max_workers_configuration(self, mock_ollama_client):
        """Test that max_workers is properly configured."""
        # Test different max_workers values
        extractor1 = RequirementExtractor(max_workers=1)
        extractor2 = RequirementExtractor(max_workers=5)
        extractor3 = RequirementExtractor()  # Default
        
        assert extractor1.max_workers == 1
        assert extractor2.max_workers == 5
        assert extractor3.max_workers == 3  # Default value


class TestConcurrentProcessingIntegration:
    """Integration tests for concurrent processing."""
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client for testing."""
        with patch('requirement_extraction.requirement_extractor.get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def mock_prompt_template(self):
        """Mock prompt template generation."""
        with patch('requirement_extraction.requirement_extractor.get_prompt') as mock_get_prompt:
            mock_get_prompt.return_value = "Test prompt"
            yield mock_get_prompt
    
    @patch('builtins.open', new_callable=mock_open, read_data="# Test Document\nSome markdown content")
    @patch('requirement_extraction.requirement_extractor.MarkdownHeaderTextSplitter')
    @patch('requirement_extraction.requirement_extractor.convert_pdf_to_markdown')
    def test_parallel_vs_sequential_behavior(self, mock_pdf_conversion, mock_splitter, mock_file,
                                           mock_prompt_template, mock_ollama_client):
        """Test that parallel processing produces same results as sequential would."""
        # Setup common mocks
        mock_pdf_conversion.return_value = "test.md"
        
        # Create multiple document chunks
        mock_docs = []
        expected_responses = []
        for i in range(4):
            mock_doc = Mock()
            mock_doc.page_content = f"Content chunk {i+1}"
            mock_docs.append(mock_doc)
            
            # Expected LLM response for each chunk
            response = json.dumps([{
                "code": f"REQ-00{i+1}",
                "description": f"Requirement from chunk {i+1}"
            }])
            expected_responses.append(response)
        
        mock_splitter.return_value.split_text.return_value = mock_docs
        
        # Mock LLM to return different responses for different chunks
        mock_ollama_client.get_structured_response.side_effect = expected_responses
        
        # Test parallel processing
        extractor = RequirementExtractor(model_name="test-model", max_workers=2)
        result = extractor.extract_requirements_from_pdf("test.pdf")
        
        # Assertions
        assert isinstance(result, RequirementList)
        assert len(result.root) == 4
        
        # Check that all expected requirements are present (order may vary due to concurrency)
        result_codes = [req.code for req in result.root]
        expected_codes = ["REQ-001", "REQ-002", "REQ-003", "REQ-004"]
        assert set(result_codes) == set(expected_codes)
    
    def test_concurrent_error_isolation(self, mock_prompt_template, mock_ollama_client):
        """Test that errors in one chunk don't affect processing of other chunks."""
        extractor = RequirementExtractor(model_name="test-model", max_workers=3)
        
        # Create test chunks
        mock_docs = []
        for i in range(3):
            mock_doc = Mock()
            mock_doc.page_content = f"Content {i+1}"
            mock_docs.append(mock_doc)
        
        # Mock LLM responses - second chunk will fail
        responses = [
            json.dumps([{"code": "REQ-001", "description": "Success 1"}]),
            None,  # This will cause chunk 2 to fail
            json.dumps([{"code": "REQ-003", "description": "Success 3"}])
        ]
        mock_ollama_client.get_structured_response.side_effect = responses
        
        # Test
        result = extractor.process_chunks_parallel(mock_docs, {"type": "object"})
        
        # Assertions - should have 2 successful results despite one failure
        assert len(result) == 2
        codes = [req.code for req in result]
        assert "REQ-001" in codes
        assert "REQ-003" in codes


class TestConvenienceFunctionConcurrency:
    """Test the convenience function with concurrency parameters."""
    
    @patch('requirement_extraction.requirement_extractor.RequirementExtractor')
    def test_extract_requirements_from_pdf_with_max_workers(self, mock_extractor_class):
        """Test the convenience function with max_workers parameter."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_requirements_from_pdf.return_value = RequirementList([])
        
        # Test with custom max_workers
        result = extract_requirements_from_pdf("test.pdf", max_workers=5)
        
        # Assertions
        mock_extractor_class.assert_called_once_with(
            model_name="DeepSeek-R1-Distill-Llama-8B-Q4_K_M:latest",
            max_workers=5
        )
        mock_extractor.extract_requirements_from_pdf.assert_called_once_with("test.pdf", "data/markdown_output")
    
    @patch('requirement_extraction.requirement_extractor.RequirementExtractor')
    def test_extract_requirements_from_pdf_default_max_workers(self, mock_extractor_class):
        """Test the convenience function with default max_workers."""
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_requirements_from_pdf.return_value = RequirementList([])
        
        # Test with default max_workers
        result = extract_requirements_from_pdf("test.pdf")
        
        # Assertions
        mock_extractor_class.assert_called_once_with(
            model_name="DeepSeek-R1-Distill-Llama-8B-Q4_K_M:latest",
            max_workers=3  # Default value
        )