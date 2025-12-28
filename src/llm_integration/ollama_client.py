from typing import Dict, Any, Optional, List
from ollama import Client
from config.schema import OllamaConfig
from utils import logging_config
from data_models.requirement import RequirementList

logger = logging_config.setup_logger(__name__)


class OllamaClient:
    """Client for interacting with Ollama server for LLM processing."""
    def __init__(self, config: OllamaConfig):
        self.config = config
        self._client = Client(
            host=config.host,
            timeout=config.timeout_seconds
        )

    @property
    def client(self) -> Client:
        """Expose the underlying Ollama client."""
        return self._client

    def _build_options(self, override_temperature: Optional[float]) -> Dict[str, Any]:
        """Combine request options with configuration defaults."""
        temperature = (
            override_temperature
            if override_temperature is not None
            else self.config.temperature
        )
        options: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else 0.1
        }

        if self.config.top_p is not None:
            options["top_p"] = self.config.top_p

        if self.config.max_tokens is not None:
            options["num_predict"] = self.config.max_tokens

        return options

    def _keep_alive_value(self) -> Optional[int]:
        """Translate keep-alive preference into the value expected by Ollama."""
        return None if self.config.keep_alive else 0

    def is_server_running(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            self.client.ps()
            return True
        except Exception as e:
            logger.error(f"Ollama server not accessible: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models on the Ollama server."""
        try:
            models = self.client.list() 
        #   models=[Model(model='mistral:7b', ...), Model(model='gemma3:27b', ...), ...]
            return [model.model for model in models['models']]
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
    
    def get_llm_response(self, prompt: str, model_name: str, 
                        system_prompt: Optional[str] = None,
                        temperature: Optional[float] = None,
                        max_retries: int = 3) -> Optional[str]:
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
                logger.info(f"Sending request to model {model_name} (attempt {attempt + 1})")
                response = self.client.generate(
                    model=model_name,
                    system=system_prompt,
                    prompt=prompt,
                    stream=False,
                    options=self._build_options(temperature),
                    keep_alive=self._keep_alive_value(),
                )

                if response and 'response' in response:
                    content = response['response']
                    logger.info(f"Received response from {model_name}")
                    return content
                else:
                    logger.error(f"Invalid response format from {model_name}")
                    
            except Exception as e:
                logger.error(f"Error getting response from {model_name} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get response after {max_retries} attempts")
                    return None

    
    def get_structured_response(self, prompt: str, model_name: str,
                              system_prompt: Optional[str] = "",
                              images: Optional[List[str]] = None,
                              temperature: Optional[float] = None,
                              max_retries: int = 3) -> Optional[str]:#Dict[str, Any]]:
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
            
        system_prompt += " Respond using JSON."
          
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to model {model_name} (attempt {attempt + 1})")
                response = self.client.generate(
                    model=model_name,
                    system=system_prompt,
                    prompt=prompt,
                    stream=False,
                    images=images,
                    options=self._build_options(temperature),
                    keep_alive=self._keep_alive_value(),
                    format=RequirementList.model_json_schema()
                )

                if response and 'response' in response:
                    content = response['response']
                    logger.info(f"Received response from {model_name}")
                    return content
                else:
                    logger.error(f"Invalid response format from {model_name}")
                    
            except Exception as e:
                logger.error(f"Error getting response from {model_name} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get response after {max_retries} attempts")
                    return None

def get_client(config: OllamaConfig) -> OllamaClient:
    """
    Factory function.
    Creates an OllamaClient using provided configuration.
    """
    return OllamaClient(config)
