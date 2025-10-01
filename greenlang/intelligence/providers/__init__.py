"""
LLM Provider Adapters

Multi-provider strategy:
- OpenAI (GPT-4, GPT-3.5) - Primary
- Anthropic (Claude-2, Claude-3) - Backup
- FakeProvider (Demo mode) - Zero-config for open source developers
- Extensible to add more providers

All providers implement LLMProvider ABC for consistent interface.
"""

from greenlang.intelligence.providers.base import (
    LLMProvider,
    LLMProviderConfig,
    LLMCapabilities,
)
from greenlang.intelligence.providers.errors import (
    ProviderAuthError,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderServerError,
    ProviderBadRequest,
)
from greenlang.intelligence.providers.fake import FakeProvider

__all__ = [
    "LLMProvider",
    "LLMProviderConfig",
    "LLMCapabilities",
    "FakeProvider",
    "ProviderAuthError",
    "ProviderRateLimit",
    "ProviderTimeout",
    "ProviderServerError",
    "ProviderBadRequest",
]
