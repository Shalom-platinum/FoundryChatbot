"""
Service module for interacting with Microsoft Foundry Local
"""
import logging
from typing import List, Dict, Optional, Generator
from openai import OpenAI
from foundry_local import FoundryLocalManager

logger = logging.getLogger(__name__)


class FoundryService:
    """Service class for Microsoft Foundry Local AI operations"""
    
    _instance = None
    _fl_manager = None
    _client = None
    
    def __new__(cls):
        """Singleton pattern to reuse connection"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize Foundry Local manager and OpenAI client"""
        try:
            self._fl_manager = FoundryLocalManager()
            self._fl_manager.start_service()
            self._client = OpenAI(
                base_url=self._fl_manager.endpoint,
                api_key=self._fl_manager.api_key
            )
            logger.info("Foundry Local service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Foundry Local: {e}")
            raise

    def get_available_models(self) -> List[Dict]:
        """Get list of available models from Foundry Local"""
        try:
            models = self._fl_manager.list_cached_models()
            return [
                {
                    'id': model.id,
                    'alias': model.alias if hasattr(model, 'alias') else model.id,
                    'name': getattr(model, 'name', model.id),
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []

    def get_model_id(self, alias_or_id: str) -> str:
        """Get actual model ID from alias or return as-is if it's an ID"""
        models = self.get_available_models()
        for model in models:
            if model['alias'] == alias_or_id or model['id'] == alias_or_id:
                return model['id']
        # Return first available model if not found
        if models:
            return models[0]['id']
        raise ValueError("No models available")

    def chat_completion(
        self,
        messages: List[Dict],
        model: str = 'phi-4-mini',
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str | Generator:
        """
        Send a chat completion request to Foundry Local
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model alias or ID to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response
            
        Returns:
            Assistant's response text or a generator for streaming
        """
        try:
            model_id = self.get_model_id(model)
            
            if stream:
                return self._stream_completion(messages, model_id, temperature, max_tokens)
            
            response = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise

    def _stream_completion(
        self,
        messages: List[Dict],
        model_id: str,
        temperature: float,
        max_tokens: int
    ) -> Generator:
        """Stream chat completion response"""
        try:
            response = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Stream completion error: {e}")
            raise

    def summarize_text(self, text: str, model: str = 'phi-4-mini') -> str:
        """
        Summarize a block of text
        
        Args:
            text: Text to summarize
            model: Model to use for summarization
            
        Returns:
            Summary of the text
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text. Provide a concise, well-structured summary that captures the key points."
            },
            {
                "role": "user",
                "content": f"Please summarize the following text:\n\n{text}"
            }
        ]
        return self.chat_completion(messages, model=model)

    def is_service_running(self) -> bool:
        """Check if Foundry Local service is running"""
        try:
            models = self._fl_manager.list_cached_models()
            return len(models) > 0
        except Exception:
            return False


# Global instance
foundry_service = None

def get_foundry_service() -> FoundryService:
    """Get or create the Foundry service instance"""
    global foundry_service
    if foundry_service is None:
        foundry_service = FoundryService()
    return foundry_service
