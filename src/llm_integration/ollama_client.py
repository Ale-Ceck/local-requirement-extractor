import json
from typing import Dict, Any, Optional, List
from ollama import Client
from utils import logging_config
from data_models.requirement import RequirementList

logger = logging_config.setup_logger(__name__)


class OllamaClient:
    """Client for interacting with Ollama server for LLM processing."""
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 300):
        """Initialize the Ollama client.
        
        Args:
            host: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.host = host
        self.timeout = timeout
        self.client = Client(host=host)
    
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
                        temperature: float = 0.1,
                        max_retries: int = 3) -> Optional[str]:
        """Get response from LLM with error handling and retries.
        
        Args:
            prompt: User prompt to send to the model
            model_name: Name of the model to use
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
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
                    options={
                        "temperature": temperature
                    },
                    keep_alive=0,
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
                              temperature: float = 0.1,
                              max_retries: int = 3) -> Optional[str]:#Dict[str, Any]]:
        """Get structured JSON response from LLM. Output format is set to JSON and the model is instruct to respond in JSON.
        
        Args:
            prompt: User prompt to send to the model
            model_name: Name of the model to use
            system_prompt: Optional system prompt. Will include instruction to respond in JSON.
            images: Optional list of images. Input data for multimodal models
            temperature: Sampling temperature (0.0 to 1.0)
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
                    options={
                        "temperature": temperature
                    },
                    #keep_alive=0,
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

# Global client instance
_client = None

def get_client(host: str = "http://localhost:11434", timeout: int = 300) -> OllamaClient:
    """Get or create global OllamaClient instance."""
    global _client
    if _client is None:
        _client = OllamaClient(host=host, timeout=timeout)
    return _client
'''
This pattern is commonly referred to as the "singleton pattern" and is particularly useful in scenarios
where maintaining a single connection or shared resource is critical, such as interacting with external APIs or servers.
The purpose of this design is to ensure that only one instance of OllamaClient is created and reused throughout the 
application, which can be beneficial for managing resources and maintaining consistent state.
'''