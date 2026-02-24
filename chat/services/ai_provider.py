"""
AI Provider abstraction layer for supporting multiple AI backends.
Supports Foundry Local and Azure OpenAI.
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Generator, Optional

logger = logging.getLogger(__name__)


class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def get_available_models(self) -> List[Dict]:
        """Get list of available models"""
        pass
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str | Generator:
        """Send a chat completion request"""
        pass
    
    @abstractmethod
    def summarize_text(self, text: str, model: str) -> str:
        """Summarize a block of text"""
        pass
    
    @abstractmethod
    def is_service_running(self) -> bool:
        """Check if the service is available"""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name"""
        pass


def _create_httpx_client():
    """Create an httpx client without proxy settings to avoid initialization errors"""
    import httpx
    return httpx.Client(
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=True
    )


class FoundryLocalProvider(AIProvider):
    """Foundry Local AI provider"""
    
    def __init__(self):
        from openai import OpenAI
        from foundry_local import FoundryLocalManager
        
        self._fl_manager = FoundryLocalManager()
        self._fl_manager.start_service()
        self._client = OpenAI(
            base_url=self._fl_manager.endpoint,
            api_key=self._fl_manager.api_key,
            http_client=_create_httpx_client()
        )
        logger.info("Foundry Local provider initialized")
    
    @property
    def provider_name(self) -> str:
        return "foundry_local"
    
    def get_available_models(self) -> List[Dict]:
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
            logger.error(f"Error getting Foundry Local models: {e}")
            return []
    
    def _get_model_id(self, alias_or_id: str) -> str:
        """Get actual model ID from alias"""
        models = self.get_available_models()
        for model in models:
            if model['alias'] == alias_or_id or model['id'] == alias_or_id:
                return model['id']
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
        try:
            model_id = self._get_model_id(model)
            
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
            logger.error(f"Foundry Local chat completion error: {e}")
            raise
    
    def _stream_completion(
        self,
        messages: List[Dict],
        model_id: str,
        temperature: float,
        max_tokens: int
    ) -> Generator:
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
        try:
            models = self._fl_manager.list_cached_models()
            return len(models) > 0
        except Exception:
            return False


class AzureOpenAIProvider(AIProvider):
    """Azure OpenAI provider"""
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None
    ):
        from openai import AzureOpenAI
        
        self._endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self._api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self._api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        self._deployment_name = deployment_name or os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')
        
        if not self._endpoint or not self._api_key:
            raise ValueError("Azure OpenAI endpoint and API key are required")
        
        self._client = AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
            http_client=_create_httpx_client()
        )
        logger.info("Azure OpenAI provider initialized")
    
    @property
    def provider_name(self) -> str:
        return "azure_openai"
    
    def get_available_models(self) -> List[Dict]:
        """Return configured Azure OpenAI models/deployments"""
        # Azure OpenAI uses deployments, return the configured one
        # Additional deployments can be configured via environment variables
        deployments = []
        
        # Primary deployment
        if self._deployment_name:
            deployments.append({
                'id': self._deployment_name,
                'alias': self._deployment_name,
                'name': f"Azure OpenAI - {self._deployment_name}",
            })
        
        # Additional deployments from environment
        additional = os.getenv('AZURE_OPENAI_ADDITIONAL_DEPLOYMENTS', '')
        if additional:
            for deploy in additional.split(','):
                deploy = deploy.strip()
                if deploy and deploy != self._deployment_name:
                    deployments.append({
                        'id': deploy,
                        'alias': deploy,
                        'name': f"Azure OpenAI - {deploy}",
                    })
        
        return deployments
    
    def _is_reasoning_model(self, model_name: str) -> bool:
        """Check if the model is a reasoning model (o1, o3, o4 series)"""
        if not model_name:
            return False
        model_lower = model_name.lower()
        # Reasoning models start with 'o' followed by a number
        return model_lower.startswith(('o1', 'o3', 'o4'))
    
    def chat_completion(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str | Generator:
        try:
            deployment = model or self._deployment_name
            
            if stream:
                return self._stream_completion(messages, deployment, temperature, max_tokens)
            
            # Build request parameters
            params = {
                'model': deployment,
                'messages': messages,
            }
            
            # Reasoning models (o1, o3, o4) use different parameters
            if self._is_reasoning_model(deployment):
                params['max_completion_tokens'] = max_tokens
                # Reasoning models don't support temperature
            else:
                params['max_tokens'] = max_tokens
                params['temperature'] = temperature
            
            response = self._client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure OpenAI chat completion error: {e}")
            raise
    
    def _stream_completion(
        self,
        messages: List[Dict],
        deployment: str,
        temperature: float,
        max_tokens: int
    ) -> Generator:
        try:
            # Build request parameters
            params = {
                'model': deployment,
                'messages': messages,
                'stream': True,
            }
            
            # Reasoning models (o1, o3, o4) use different parameters
            if self._is_reasoning_model(deployment):
                params['max_completion_tokens'] = max_tokens
            else:
                params['max_tokens'] = max_tokens
                params['temperature'] = temperature
            
            response = self._client.chat.completions.create(**params)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Azure OpenAI stream error: {e}")
            raise
    
    def summarize_text(self, text: str, model: str = None) -> str:
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
        try:
            # Simple test request to check connectivity
            self._client.models.list()
            return True
        except Exception:
            # Azure OpenAI doesn't support listing models, try a minimal completion
            try:
                self._client.chat.completions.create(
                    model=self._deployment_name,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                return True
            except Exception:
                return False


class AIProviderManager:
    """
    Manages AI provider instances and selection.
    Supports switching between providers via environment variable or user settings.
    """
    
    PROVIDER_FOUNDRY_LOCAL = 'foundry_local'
    PROVIDER_AZURE_OPENAI = 'azure_openai'
    
    _providers: Dict[str, AIProvider] = {}
    _default_provider: str = None
    
    @classmethod
    def get_default_provider_name(cls) -> str:
        """Get the default provider name from environment or fallback"""
        if cls._default_provider:
            return cls._default_provider
        
        # Check environment variable
        env_provider = os.getenv('AI_PROVIDER', '').lower()
        if env_provider in [cls.PROVIDER_FOUNDRY_LOCAL, cls.PROVIDER_AZURE_OPENAI]:
            cls._default_provider = env_provider
        else:
            # Auto-detect based on available configuration
            if os.getenv('AZURE_OPENAI_ENDPOINT') and os.getenv('AZURE_OPENAI_API_KEY'):
                cls._default_provider = cls.PROVIDER_AZURE_OPENAI
            else:
                cls._default_provider = cls.PROVIDER_FOUNDRY_LOCAL
        
        return cls._default_provider
    
    @classmethod
    def get_provider(cls, provider_name: Optional[str] = None) -> AIProvider:
        """
        Get an AI provider instance.
        
        Args:
            provider_name: Name of the provider ('foundry_local' or 'azure_openai').
                          If None, uses the default provider.
        
        Returns:
            AIProvider instance
        """
        if provider_name is None:
            provider_name = cls.get_default_provider_name()
        
        # Return cached provider if available
        if provider_name in cls._providers:
            return cls._providers[provider_name]
        
        # Create new provider instance
        if provider_name == cls.PROVIDER_FOUNDRY_LOCAL:
            provider = cls._create_foundry_provider()
        elif provider_name == cls.PROVIDER_AZURE_OPENAI:
            provider = cls._create_azure_provider()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        cls._providers[provider_name] = provider
        return provider
    
    @classmethod
    def _create_foundry_provider(cls) -> FoundryLocalProvider:
        """Create Foundry Local provider"""
        try:
            return FoundryLocalProvider()
        except Exception as e:
            logger.error(f"Failed to create Foundry Local provider: {e}")
            raise
    
    @classmethod
    def _create_azure_provider(cls) -> AzureOpenAIProvider:
        """Create Azure OpenAI provider"""
        try:
            return AzureOpenAIProvider()
        except Exception as e:
            logger.error(f"Failed to create Azure OpenAI provider: {e}")
            raise
    
    @classmethod
    def is_provider_available(cls, provider_name: str) -> bool:
        """Check if a provider can be initialized"""
        try:
            if provider_name == cls.PROVIDER_AZURE_OPENAI:
                return bool(os.getenv('AZURE_OPENAI_ENDPOINT') and os.getenv('AZURE_OPENAI_API_KEY'))
            elif provider_name == cls.PROVIDER_FOUNDRY_LOCAL:
                # Try to import foundry_local to check availability
                try:
                    import foundry_local
                    return True
                except ImportError:
                    return False
            return False
        except Exception:
            return False
    
    @classmethod
    def get_available_providers(cls) -> List[Dict]:
        """Get list of available providers with their status"""
        providers = []
        
        # Check Foundry Local
        foundry_available = cls.is_provider_available(cls.PROVIDER_FOUNDRY_LOCAL)
        providers.append({
            'id': cls.PROVIDER_FOUNDRY_LOCAL,
            'name': 'Foundry Local',
            'available': foundry_available,
            'description': 'Local AI models via Microsoft Foundry Local'
        })
        
        # Check Azure OpenAI
        azure_available = cls.is_provider_available(cls.PROVIDER_AZURE_OPENAI)
        providers.append({
            'id': cls.PROVIDER_AZURE_OPENAI,
            'name': 'Azure OpenAI',
            'available': azure_available,
            'description': 'Cloud AI via Azure OpenAI Service'
        })
        
        return providers
    
    @classmethod
    def clear_cache(cls):
        """Clear cached provider instances"""
        cls._providers.clear()
        cls._default_provider = None


# Convenience function for backward compatibility
def get_ai_provider(provider_name: Optional[str] = None) -> AIProvider:
    """Get an AI provider instance"""
    return AIProviderManager.get_provider(provider_name)
