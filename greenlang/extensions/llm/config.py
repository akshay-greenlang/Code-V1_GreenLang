"""
LLM Configuration
=================

Configuration classes for LLM integration.

Author: AI Team
Created: 2025-11-21
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List


class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    BEDROCK = "bedrock"
    VERTEXAI = "vertexai"
    MOCK = "mock"


@dataclass
class ModelCapabilities:
    """Capabilities of an LLM model."""
    max_context_length: int = 4096
    supports_functions: bool = False
    supports_vision: bool = False
    supports_streaming: bool = False
    supports_json_mode: bool = False
    languages: List[str] = field(default_factory=lambda: ["en"])
    specializations: List[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    # Model settings
    provider: ModelProvider = ModelProvider.MOCK
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    organization: Optional[str] = None

    # Request settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    # Default completion settings
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 3600

    # Safety and filtering
    content_filter: bool = True
    pii_detection: bool = True
    max_input_length: int = 10000

    # Model capabilities
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    # Additional settings
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""
        if self.provider != ModelProvider.MOCK and not self.api_key:
            raise ValueError(f"API key required for {self.provider.value}")

        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")

        if self.max_tokens < 1:
            raise ValueError("Max tokens must be positive")

        if self.timeout < 1:
            raise ValueError("Timeout must be positive")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'LLMConfig':
        """Create config from dictionary."""
        # Handle provider enum
        if 'provider' in config and isinstance(config['provider'], str):
            config['provider'] = ModelProvider(config['provider'])

        # Handle capabilities
        if 'capabilities' in config and isinstance(config['capabilities'], dict):
            config['capabilities'] = ModelCapabilities(**config['capabilities'])

        return cls(**config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'provider': self.provider.value,
            'model': self.model,
            'api_key': '***' if self.api_key else None,  # Mask API key
            'api_base': self.api_base,
            'organization': self.organization,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'cache_enabled': self.cache_enabled,
            'cache_ttl': self.cache_ttl,
            'content_filter': self.content_filter,
            'pii_detection': self.pii_detection,
            'max_input_length': self.max_input_length,
            'capabilities': {
                'max_context_length': self.capabilities.max_context_length,
                'supports_functions': self.capabilities.supports_functions,
                'supports_vision': self.capabilities.supports_vision,
                'supports_streaming': self.capabilities.supports_streaming,
                'supports_json_mode': self.capabilities.supports_json_mode,
                'languages': self.capabilities.languages,
                'specializations': self.capabilities.specializations
            },
            'metadata': self.metadata
        }


# Predefined configurations for common models
PRESET_CONFIGS = {
    "gpt-4": LLMConfig(
        provider=ModelProvider.OPENAI,
        model="gpt-4",
        capabilities=ModelCapabilities(
            max_context_length=8192,
            supports_functions=True,
            supports_vision=False,
            supports_streaming=True,
            supports_json_mode=True,
            languages=["en", "es", "fr", "de", "zh", "ja"],
            specializations=["reasoning", "coding", "analysis"]
        )
    ),
    "gpt-4-vision": LLMConfig(
        provider=ModelProvider.OPENAI,
        model="gpt-4-vision-preview",
        capabilities=ModelCapabilities(
            max_context_length=128000,
            supports_functions=True,
            supports_vision=True,
            supports_streaming=True,
            supports_json_mode=True,
            languages=["en", "es", "fr", "de", "zh", "ja"],
            specializations=["vision", "reasoning", "analysis"]
        )
    ),
    "claude-3": LLMConfig(
        provider=ModelProvider.ANTHROPIC,
        model="claude-3-opus",
        capabilities=ModelCapabilities(
            max_context_length=200000,
            supports_functions=False,
            supports_vision=True,
            supports_streaming=True,
            supports_json_mode=False,
            languages=["en", "es", "fr", "de", "zh", "ja"],
            specializations=["reasoning", "coding", "analysis", "creativity"]
        )
    ),
    "mock": LLMConfig(
        provider=ModelProvider.MOCK,
        model="mock-model",
        capabilities=ModelCapabilities(
            max_context_length=4096,
            supports_functions=True,
            supports_vision=False,
            supports_streaming=False,
            supports_json_mode=True,
            languages=["en"],
            specializations=["testing"]
        )
    )
}


def get_preset_config(preset: str) -> LLMConfig:
    """Get a preset configuration."""
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[preset]