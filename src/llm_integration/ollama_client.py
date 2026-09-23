from typing import Any

from ollama import Client

from config.schema import OllamaConfig
from src.data_models.requirement import RequirementExtractionResult
from src.utils import logging_config

logger = logging_config.setup_logger(__name__)


class OllamaClient:
    """Client for interacting with Ollama server for LLM processing."""

    def __init__(self, config: OllamaConfig):
        self.config = config
        self._client = Client(host=config.host, timeout=config.timeout_seconds)

    @property
    def client(self) -> Client:
        """Expose the underlying Ollama client."""
        return self._client

    def _build_options(self, override_temperature: float | None) -> dict[str, Any]:
        """Combine request options with configuration defaults."""
        temperature = (
            override_temperature
            if override_temperature is not None
            else self.config.temperature
        )
        options: dict[str, Any] = {
            "temperature": temperature if temperature is not None else 0.1
        }

        if self.config.top_p is not None:
            options["top_p"] = self.config.top_p

        if self.config.max_tokens is not None:
            options["num_predict"] = self.config.max_tokens

        if self.config.seed is not None:
            options["seed"] = self.config.seed

        return options

    def get_model_metadata(self, model_name: str) -> dict[str, Any]:
        """Return best-effort immutable metadata for a locally installed model."""
        metadata: dict[str, Any] = {"name": model_name, "digest": None}
        try:
            response = self.client.list()
            models = getattr(response, "models", None)
            if models is None and isinstance(response, dict):
                models = response.get("models", [])
            for model in models or []:
                candidate_name = getattr(model, "model", None)
                candidate_digest = getattr(model, "digest", None)
                if isinstance(model, dict):
                    candidate_name = (
                        candidate_name or model.get("model") or model.get("name")
                    )
                    candidate_digest = candidate_digest or model.get("digest")
                if candidate_name == model_name:
                    metadata["digest"] = candidate_digest
                    break
        except Exception as exc:
            logger.warning("Could not resolve digest for model %s: %s", model_name, exc)
        return metadata

    def _keep_alive_value(self) -> float | str | None:
        """Translate keep-alive preference into the value expected by Ollama."""
        return None if self.config.keep_alive else 0.0

    def is_server_running(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            self.client.ps()
            return True
        except Exception as e:
            logger.error(f"Ollama server not accessible: {e}")
            return False

    def get_available_models(self) -> list[str]:
        """Get list of available models on the Ollama server."""
        try:
            response = self.client.list()
            models = getattr(response, "models", None)
            if models is None and isinstance(response, dict):
                models = response.get("models", [])
            return [
                (
                    str(
                        getattr(model, "model", None)
                        or model.get("model")
                        or model.get("name")
                    )
                    if isinstance(model, dict)
                    else str(model.model)
                )
                for model in models or []
            ]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        available_models = self.get_available_models()
        return model_name in available_models

    def pull_model(self, model_name: str) -> bool:
        """Pull a model if it's not available locally."""
        try:
            logger.info(f"Pulling model: {model_name}")
            self.client.pull(model_name)
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    def get_llm_response(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> str | None:
        """Get response from LLM with error handling and retries.

        Args:
            prompt: User prompt to send to the model
            model_name: Name of the model to use
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0). Falls back to config.
            max_retries: Maximum number of retry attempts

        Returns:
            Model response as string, or None if failed
        """

        if not self.is_model_available(model_name):
            logger.warning(f"Model {model_name} not available. Attempting to pull...")
            if not self.pull_model(model_name):
                logger.error(f"Failed to pull model {model_name}")
                return None

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Sending request to model {model_name} (attempt {attempt + 1})"
                )
                response = self.client.generate(
                    model=model_name,
                    system=system_prompt or "",
                    prompt=prompt,
                    stream=False,
                    options=self._build_options(temperature),
                    keep_alive=self._keep_alive_value(),
                )

                if response and "response" in response:
                    content = response["response"]
                    logger.info(f"Received response from {model_name}")
                    return str(content)
                else:
                    logger.error(f"Invalid response format from {model_name}")

            except Exception as e:
                logger.error(
                    f"Error getting response from {model_name} (attempt {attempt + 1}): {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get response after {max_retries} attempts")
                    return None
        return None

    def get_structured_response(
        self,
        prompt: str,
        model_name: str,
        system_prompt: str | None = "",
        images: list[str] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_retries: int = 3,
    ) -> str | None:
        """Get structured JSON response from LLM. Output format is set to JSON and the model is instruct to respond in JSON.

        Args:
            prompt: User prompt to send to the model
            model_name: Name of the model to use
            system_prompt: Optional system prompt. Will include instruction to respond in JSON.
            images: Optional list of images. Input data for multimodal models
            temperature: Sampling temperature (0.0 to 1.0). Falls back to config.
            max_retries: Maximum number of retry attempts

        Returns:
            Parsed JSON response as dict, or None if failed
        """

        if not self.is_model_available(model_name):
            logger.warning(f"Model {model_name} not available. Attempting to pull...")
            if not self.pull_model(model_name):
                logger.error(f"Failed to pull model {model_name}")
                return None

        system_prompt = (system_prompt or "") + " Respond using JSON."

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Sending request to model {model_name} (attempt {attempt + 1})"
                )
                response = self.client.generate(
                    model=model_name,
                    system=system_prompt,
                    prompt=prompt,
                    stream=False,
                    images=images,
                    options=self._build_options(temperature),
                    keep_alive=self._keep_alive_value(),
                    format=response_schema
                    or RequirementExtractionResult.model_json_schema(),
                )

                if response and "response" in response:
                    content = response["response"]
                    logger.info(f"Received response from {model_name}")
                    return str(content)
                else:
                    logger.error(f"Invalid response format from {model_name}")

            except Exception as e:
                logger.error(
                    f"Error getting response from {model_name} (attempt {attempt + 1}): {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get response after {max_retries} attempts")
                    return None
        return None


def get_client(config: OllamaConfig) -> OllamaClient:
    """
    Factory function.
    Creates an OllamaClient using provided configuration.
    """
    return OllamaClient(config)
