from unittest.mock import Mock, patch

from config.schema import OllamaConfig
from src.data_models.requirement import RequirementExtractionResult
from src.llm_integration.ollama_client import OllamaClient


@patch("src.llm_integration.ollama_client.Client")
def test_get_structured_response_uses_provided_response_schema(mock_client_class):
    mock_client = Mock()
    mock_client.generate.return_value = {"response": "[]"}
    mock_client_class.return_value = mock_client

    client = OllamaClient(OllamaConfig())
    schema = RequirementExtractionResult.model_json_schema()

    with patch.object(OllamaClient, "is_model_available", return_value=True):
        client.get_structured_response(
            prompt="Extract requirements",
            model_name="test-model",
            response_schema=schema,
        )

    generate_kwargs = mock_client.generate.call_args.kwargs
    assert generate_kwargs["format"] == schema
    assert generate_kwargs["options"]["temperature"] == 0.0
    assert generate_kwargs["options"]["seed"] == 42


@patch("src.llm_integration.ollama_client.Client")
def test_get_model_metadata_returns_exact_digest(mock_client_class):
    mock_client = Mock()
    mock_client.list.return_value = {
        "models": [
            {"model": "other-model", "digest": "sha256:other"},
            {"model": "test-model", "digest": "sha256:exact"},
        ]
    }
    mock_client_class.return_value = mock_client

    client = OllamaClient(OllamaConfig())

    assert client.get_model_metadata("test-model") == {
        "name": "test-model",
        "digest": "sha256:exact",
    }
