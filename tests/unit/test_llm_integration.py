import pytest
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path so we can import our modules
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from llm_integration.ollama_client import OllamaClient, get_client # type: ignore
from llm_integration.prompt_templates import get_requirement_extraction_prompt, get_prompt # type: ignore


class TestOllamaClient:
    """Test suite for OllamaClient class."""
    
    def setup_method(self):
        """Setup run before each test method."""
        self.host = "http://localhost:11434"
        self.timeout = 60
        self.client = OllamaClient(host=self.host, timeout=self.timeout)
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Reset the global client instance
        import llm_integration.ollama_client # type: ignore
        llm_integration.ollama_client._client = None
    
    def test_init_default_parameters(self):
        """Test OllamaClient initialization with default parameters."""
        client = OllamaClient()
        assert client.host == "http://localhost:11434"
        assert client.timeout == 60
        assert client.client is not None
    
    def test_init_custom_parameters(self):
        """Test OllamaClient initialization with custom parameters."""
        custom_host = "http://custom-host:8080"
        custom_timeout = 120
        
        client = OllamaClient(host=custom_host, timeout=custom_timeout)
        assert client.host == custom_host
        assert client.timeout == custom_timeout
    
    @patch('llm_integration.ollama_client.Client')
    def test_init_creates_client_instance(self, mock_client_class):
        """Test that Client instance is created during initialization."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        client = OllamaClient(host=self.host, timeout=self.timeout)
        
        mock_client_class.assert_called_once_with(host=self.host)
        assert client.client == mock_client_instance


class TestOllamaClientServerStatus:
    """Test server status checking methods."""
    
    def setup_method(self):
        """Setup run before each test method."""
        self.client = OllamaClient()
    
    def test_is_server_running_success(self):
        """Test server status check when server is running."""
        self.client.client = Mock()
        self.client.client.ps.return_value = {"models": []}
        
        result = self.client.is_server_running()
        
        assert result is True
        self.client.client.ps.assert_called_once()
    
    def test_is_server_running_failure(self):
        """Test server status check when server is not accessible."""
        self.client.client = Mock()
        self.client.client.ps.side_effect = Exception("Connection refused")
        
        result = self.client.is_server_running()
        
        assert result is False
        self.client.client.ps.assert_called_once()
    
    @patch('llm_integration.ollama_client.logger')
    def test_is_server_running_logs_error(self, mock_logger):
        """Test that server connection errors are logged."""
        self.client.client = Mock()
        error_msg = "Connection refused"
        self.client.client.ps.side_effect = Exception(error_msg)
        
        self.client.is_server_running()
        
        mock_logger.error.assert_called_once_with(f"Ollama server not accessible: {error_msg}")


class TestOllamaClientModels:
    """Test model-related methods."""
    
    def setup_method(self):
        """Setup run before each test method."""
        self.client = OllamaClient()
    
    def test_get_available_models_success(self):
        """Test getting available models when server responds correctly."""
        mock_models = {
            'models': [
                Mock(model='mistral:7b'),
                Mock(model='gemma3:27b'),
                Mock(model='llama2:13b')
            ]
        }
        self.client.client = Mock()
        self.client.client.list.return_value = mock_models
        
        result = self.client.get_available_models()
        
        expected = ['mistral:7b', 'gemma3:27b', 'llama2:13b']
        assert result == expected
        self.client.client.list.assert_called_once()
    
    def test_get_available_models_empty_list(self):
        """Test getting available models when no models are available."""
        self.client.client = Mock()
        self.client.client.list.return_value = {'models': []}
        
        result = self.client.get_available_models()
        
        assert result == []
    
    def test_get_available_models_failure(self):
        """Test getting available models when server request fails."""
        self.client.client = Mock()
        self.client.client.list.side_effect = Exception("Server error")
        
        result = self.client.get_available_models()
        
        assert result == []
    
    @patch('llm_integration.ollama_client.logger')
    def test_get_available_models_logs_error(self, mock_logger):
        """Test that model listing errors are logged."""
        self.client.client = Mock()
        error_msg = "Server error"
        self.client.client.list.side_effect = Exception(error_msg)
        
        self.client.get_available_models()
        
        mock_logger.error.assert_called_once_with(f"Failed to get available models: {error_msg}")
    
    @patch.object(OllamaClient, 'get_available_models')
    def test_is_model_available_true(self, mock_get_models):
        """Test model availability check when model exists."""
        mock_get_models.return_value = ['mistral:7b', 'gemma3:27b']
        
        result = self.client.is_model_available('mistral:7b')
        
        assert result is True
        mock_get_models.assert_called_once()
    
    @patch.object(OllamaClient, 'get_available_models')
    def test_is_model_available_false(self, mock_get_models):
        """Test model availability check when model doesn't exist."""
        mock_get_models.return_value = ['mistral:7b', 'gemma3:27b']
        
        result = self.client.is_model_available('nonexistent:model')
        
        assert result is False
        mock_get_models.assert_called_once()
    
    def test_pull_model_success(self):
        """Test successful model pulling."""
        self.client.client = Mock()
        self.client.client.pull.return_value = True
        model_name = 'mistral:7b'
        
        result = self.client.pull_model(model_name)
        
        assert result is True
        self.client.client.pull.assert_called_once_with(model_name)
    
    def test_pull_model_failure(self):
        """Test failed model pulling."""
        self.client.client = Mock()
        self.client.client.pull.side_effect = Exception("Pull failed")
        model_name = 'mistral:7b'
        
        result = self.client.pull_model(model_name)
        
        assert result is False
        self.client.client.pull.assert_called_once_with(model_name)
    
    @patch('llm_integration.ollama_client.logger')
    def test_pull_model_logs_info_and_error(self, mock_logger):
        """Test that model pulling logs appropriate messages."""
        self.client.client = Mock()
        error_msg = "Pull failed"
        self.client.client.pull.side_effect = Exception(error_msg)
        model_name = 'mistral:7b'
        
        self.client.pull_model(model_name)
        
        mock_logger.info.assert_called_once_with(f"Pulling model: {model_name}")
        mock_logger.error.assert_called_once_with(f"Failed to pull model {model_name}: {error_msg}")


class TestOllamaClientLLMResponse:
    """Test LLM response methods."""
    
    def setup_method(self):
        """Setup run before each test method."""
        self.client = OllamaClient()
        self.model_name = 'mistral:7b'
        self.prompt = "What is the capital of France?"
    
    @patch.object(OllamaClient, 'is_model_available')
    def test_get_llm_response_success(self, mock_is_available):
        """Test successful LLM response."""
        mock_is_available.return_value = True
        
        self.client.client = Mock()
        mock_response = {
            'response': 'Paris is the capital of France.'
        }
        self.client.client.generate.return_value = mock_response
        
        result = self.client.get_llm_response(self.prompt, self.model_name)
        
        assert result == 'Paris is the capital of France.'
        self.client.client.generate.assert_called_once()
    
    @patch.object(OllamaClient, 'is_model_available')
    @patch.object(OllamaClient, 'pull_model')
    def test_get_llm_response_model_not_available_pull_success(self, mock_pull, mock_is_available):
        """Test LLM response when model is not available but pull succeeds."""
        mock_is_available.return_value = False
        mock_pull.return_value = True
        
        self.client.client = Mock()
        mock_response = {
            'response': 'Paris is the capital of France.'
        }
        self.client.client.generate.return_value = mock_response
        
        result = self.client.get_llm_response(self.prompt, self.model_name)
        
        assert result == 'Paris is the capital of France.'
        mock_pull.assert_called_once_with(self.model_name)
    
    @patch.object(OllamaClient, 'is_model_available')
    @patch.object(OllamaClient, 'pull_model')
    def test_get_llm_response_model_not_available_pull_fails(self, mock_pull, mock_is_available):
        """Test LLM response when model is not available and pull fails."""
        mock_is_available.return_value = False
        mock_pull.return_value = False
        
        result = self.client.get_llm_response(self.prompt, self.model_name)
        
        assert result is None
        mock_pull.assert_called_once_with(self.model_name)
    
    @patch.object(OllamaClient, 'is_model_available')
    def test_get_llm_response_with_system_prompt(self, mock_is_available):
        """Test LLM response with system prompt."""
        mock_is_available.return_value = True
        
        self.client.client = Mock()
        mock_response = {
            'response': 'Response with system prompt.'
        }
        self.client.client.generate.return_value = mock_response
        
        system_prompt = "You are a helpful assistant."
        result = self.client.get_llm_response(
            self.prompt, 
            self.model_name, 
            system_prompt=system_prompt
        )
        
        assert result == 'Response with system prompt.'
        
        # Verify the system prompt was passed
        call_args = self.client.client.generate.call_args
        assert call_args[1]['system'] == system_prompt
    
    @patch.object(OllamaClient, 'is_model_available')
    def test_get_llm_response_custom_temperature(self, mock_is_available):
        """Test LLM response with custom temperature."""
        mock_is_available.return_value = True
        
        self.client.client = Mock()
        mock_response = {
            'response': 'Response with custom temperature.'
        }
        self.client.client.generate.return_value = mock_response
        
        custom_temp = 0.8
        result = self.client.get_llm_response(
            self.prompt, 
            self.model_name, 
            temperature=custom_temp
        )
        
        assert result == 'Response with custom temperature.'
        
        # Verify temperature is passed correctly
        call_args = self.client.client.generate.call_args
        options = call_args[1]['options']
        assert options['temperature'] == custom_temp
    
    @patch.object(OllamaClient, 'is_model_available')
    def test_get_llm_response_invalid_response_format(self, mock_is_available):
        """Test LLM response with invalid response format."""
        mock_is_available.return_value = True
        
        self.client.client = Mock()
        mock_response = {'invalid': 'format'}
        self.client.client.generate.return_value = mock_response
        
        result = self.client.get_llm_response(self.prompt, self.model_name)
        
        assert result is None
    
    @patch.object(OllamaClient, 'is_model_available')
    def test_get_llm_response_retry_mechanism(self, mock_is_available):
        """Test LLM response retry mechanism."""
        mock_is_available.return_value = True
        
        self.client.client = Mock()
        # First two calls fail, third succeeds
        self.client.client.generate.side_effect = [
            Exception("First attempt failed"),
            Exception("Second attempt failed"),
            {'response': 'Success on third attempt'}
        ]
        
        result = self.client.get_llm_response(self.prompt, self.model_name, max_retries=3)
        
        assert result == 'Success on third attempt'
        assert self.client.client.generate.call_count == 3
    
    @patch.object(OllamaClient, 'is_model_available')
    def test_get_llm_response_all_retries_fail(self, mock_is_available):
        """Test LLM response when all retries fail."""
        mock_is_available.return_value = True
        
        self.client.client = Mock()
        self.client.client.generate.side_effect = Exception("All attempts failed")
        
        result = self.client.get_llm_response(self.prompt, self.model_name, max_retries=2)
        
        assert result is None
        assert self.client.client.generate.call_count == 2


class TestOllamaClientStructuredResponse:
    """Test structured JSON response methods."""
    
    def setup_method(self):
        """Setup run before each test method."""
        self.client = OllamaClient()
        self.model_name = 'mistral:7b'
        self.prompt = "Return a JSON response"
    
    @patch.object(OllamaClient, 'is_model_available')
    def test_get_structured_response_success(self, mock_is_available):
        """Test successful structured response."""
        mock_is_available.return_value = True
        
        self.client.client = Mock()
        json_data = {'key': 'value', 'number': 42}
        mock_response = {
            'response': json.dumps(json_data)
        }
        self.client.client.generate.return_value = mock_response
        
        result = self.client.get_structured_response(self.prompt, self.model_name)
        
        assert result == json.dumps(json_data)
    
    @patch.object(OllamaClient, 'is_model_available')
    def test_get_structured_response_with_system_prompt(self, mock_is_available):
        """Test structured response with system prompt modification."""
        mock_is_available.return_value = True
        
        self.client.client = Mock()
        mock_response = {
            'response': '{"test": "data"}'
        }
        self.client.client.generate.return_value = mock_response
        
        system_prompt = "You are a JSON generator."
        result = self.client.get_structured_response(
            self.prompt, 
            self.model_name, 
            system_prompt=system_prompt
        )
        
        assert result == '{"test": "data"}'
        
        # Verify the system prompt was modified to include JSON instruction
        call_args = self.client.client.generate.call_args
        assert "Respond using JSON." in call_args[1]['system']
    
    @patch.object(OllamaClient, 'is_model_available')
    def test_get_structured_response_format_json(self, mock_is_available):
        """Test that structured response uses JSON format."""
        mock_is_available.return_value = True
        
        self.client.client = Mock()
        mock_response = {
            'response': '{"formatted": "json"}'
        }
        self.client.client.generate.return_value = mock_response
        
        result = self.client.get_structured_response(self.prompt, self.model_name)
        
        # Verify format="json" is passed
        call_args = self.client.client.generate.call_args
        assert call_args[1]['format'] == 'json'
    
    @patch.object(OllamaClient, 'is_model_available')
    def test_get_structured_response_none_response(self, mock_is_available):
        """Test structured response when LLM returns None."""
        mock_is_available.return_value = True
        
        self.client.client = Mock()
        self.client.client.generate.return_value = None
        
        result = self.client.get_structured_response(self.prompt, self.model_name)
        
        assert result is None


class TestOllamaClientGlobalInstance:
    """Test the global client instance functionality."""
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Reset the global client instance
        import llm_integration.ollama_client # type: ignore
        llm_integration.ollama_client._client = None
    
    def test_get_client_creates_instance(self):
        """Test that get_client creates a new instance when none exists."""
        client = get_client()
        
        assert isinstance(client, OllamaClient)
        assert client.host == "http://localhost:11434"
        assert client.timeout == 60
    
    def test_get_client_returns_same_instance(self):
        """Test that get_client returns the same instance on subsequent calls."""
        client1 = get_client()
        client2 = get_client()
        
        assert client1 is client2
    
    def test_get_client_with_custom_parameters(self):
        """Test get_client with custom host and timeout."""
        custom_host = "http://custom:8080"
        custom_timeout = 120
        
        client = get_client(host=custom_host, timeout=custom_timeout)
        
        assert client.host == custom_host
        assert client.timeout == custom_timeout
    
    def test_get_client_ignores_parameters_on_subsequent_calls(self):
        """Test that get_client ignores parameters when instance already exists."""
        # First call creates instance with default parameters
        client1 = get_client()
        original_host = client1.host
        
        # Second call with different parameters should return same instance
        client2 = get_client(host="http://different:9090", timeout=999)
        
        assert client1 is client2
        assert client2.host == original_host


class TestPromptTemplates:
    """Test suite for prompt_templates module."""
    
    def test_get_requirement_extraction_prompt_basic_functionality(self):
        """Test basic functionality of get_requirement_extraction_prompt."""
        test_content = "This is test content with REQ-001: Test requirement"
        test_schema = '{"type": "object", "properties": {"code": {"type": "string"}}}'
        
        prompt = get_requirement_extraction_prompt(test_content, test_schema)
        
        # Check that the prompt is a string
        assert isinstance(prompt, str)
        
        # Check that the prompt is not empty
        assert len(prompt.strip()) > 0
        
        # Check that content is included in the prompt
        assert test_content in prompt
        
        # Check that schema is included in the prompt
        assert test_schema in prompt
    
    def test_get_requirement_extraction_prompt_contains_required_elements(self):
        """Test that the prompt contains all required elements."""
        test_content = "REQ-001: System must authenticate users"
        test_schema = '{"type": "object"}'
        
        prompt = get_requirement_extraction_prompt(test_content, test_schema)
        
        # Check for key instruction elements
        assert "extract" in prompt.lower()
        assert "requirements" in prompt.lower()
        assert "code" in prompt.lower()
        assert "description" in prompt.lower()
        
        # Check for format instructions
        assert "JSON" in prompt
        assert "schema" in prompt.lower()
    
    def test_get_requirement_extraction_prompt_with_few_shot_examples(self):
        """Test prompt generation with few-shot examples."""
        test_content = "Test content"
        test_schema = '{"type": "object"}'
        few_shot_examples = "Example 1: REQ-001 -> Test requirement"
        
        prompt = get_requirement_extraction_prompt(test_content, test_schema, few_shot_examples)
        
        # Check that examples are included
        assert few_shot_examples in prompt
        assert "examples" in prompt.lower()
    
    def test_get_prompt_function_requirement_extraction(self):
        """Test the get_prompt function for requirement extraction."""
        test_content = "Test content"
        test_schema = '{"type": "object"}'
        
        prompt = get_prompt("requirement_extraction", test_content, test_schema)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert test_content in prompt
        assert test_schema in prompt
    
    def test_get_prompt_function_with_few_shot_examples(self):
        """Test get_prompt function with few-shot examples enabled."""
        test_content = "Test content"
        test_schema = '{"type": "object"}'
        
        prompt = get_prompt("requirement_extraction", test_content, test_schema, include_few_shot=True)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # Should include the predefined few-shot examples
        assert "Example" in prompt
    
    def test_get_prompt_function_without_few_shot_examples(self):
        """Test get_prompt function with few-shot examples disabled."""
        test_content = "Test content"
        test_schema = '{"type": "object"}'
        
        prompt_with_examples = get_prompt("requirement_extraction", test_content, test_schema, include_few_shot=True)
        prompt_without_examples = get_prompt("requirement_extraction", test_content, test_schema, include_few_shot=False)
        
        # Prompt without examples should be shorter
        assert len(prompt_without_examples) < len(prompt_with_examples)
    
    def test_get_prompt_function_unknown_prompt_name(self):
        """Test get_prompt function with unknown prompt name."""
        test_content = "Test content"
        test_schema = '{"type": "object"}'
        
        with pytest.raises(ValueError) as exc_info:
            get_prompt("unknown_prompt", test_content, test_schema)
        
        assert "Unknown prompt name" in str(exc_info.value)


# Parametrized tests for different scenarios
class TestOllamaClientParametrized:
    """Parametrized tests for various scenarios."""
    
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 0.5, 0.8, 1.0])
    def test_temperature_values(self, temperature):
        """Test different temperature values."""
        client = OllamaClient()
        client.client = Mock()
        
        mock_response = {'response': 'test response'}
        client.client.generate.return_value = mock_response
        
        with patch.object(client, 'is_model_available', return_value=True):
            result = client.get_llm_response("test", "model", temperature=temperature)
            
            assert result == 'test response'
            call_args = client.client.generate.call_args
            assert call_args[1]['options']['temperature'] == temperature
    
    @pytest.mark.parametrize("max_retries", [1, 2, 3, 5])
    def test_retry_values(self, max_retries):
        """Test different retry values."""
        client = OllamaClient()
        client.client = Mock()
        client.client.generate.side_effect = Exception("Always fails")
        
        with patch.object(client, 'is_model_available', return_value=True):
            result = client.get_llm_response("test", "model", max_retries=max_retries)
            
            assert result is None
            assert client.client.generate.call_count == max_retries


if __name__ == "__main__":
    # Run tests with: python -m pytest test_llm_integration.py -v
    pytest.main([__file__, "-v"])