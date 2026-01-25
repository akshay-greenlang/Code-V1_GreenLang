# -*- coding: utf-8 -*-
"""
LLM Provider Adapters - Multi-Tier Intelligence System

GreenLang's intelligence is organized in tiers for the open-source ecosystem:

TIER 2 (BYOK - Bring Your Own Key):
- OpenAI (GPT-4, GPT-4o) - High quality, requires API key
- Anthropic (Claude-3) - High quality, requires API key

TIER 1 (Local LLM - Free, Private):
- Ollama (Llama 3, Mistral, Phi-3) - Real LLM, no API key needed
- Runs locally, data stays private

TIER 0 (Deterministic - Always Available):
- Template-based, rule-driven intelligence
- Works immediately after pip install
- No dependencies, no cost

SMART ROUTER:
- Automatically selects best available tier
- Graceful fallback on failures
- Consistent API regardless of backend

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

# Core providers
from greenlang.intelligence.providers.fake import FakeProvider
from greenlang.intelligence.providers.deterministic import DeterministicProvider
from greenlang.intelligence.providers.ollama import OllamaProvider

# Smart router
from greenlang.intelligence.providers.router import (
    SmartProviderRouter,
    IntelligenceTier,
    TierStatus,
    RouterConfig,
)

__all__ = [
    # Base classes
    "LLMProvider",
    "LLMProviderConfig",
    "LLMCapabilities",

    # Providers by tier
    "DeterministicProvider",  # Tier 0
    "OllamaProvider",         # Tier 1
    "FakeProvider",           # Legacy demo mode

    # Smart router (recommended)
    "SmartProviderRouter",
    "IntelligenceTier",
    "TierStatus",
    "RouterConfig",

    # Errors
    "ProviderAuthError",
    "ProviderRateLimit",
    "ProviderTimeout",
    "ProviderServerError",
    "ProviderBadRequest",
]
