"""
GreenLang Intelligence Layer

AI-native intelligence framework for climate calculations with:
- Multi-provider LLM support (OpenAI, Anthropic, Demo)
- Zero-config setup (works without API keys!)
- Tool-first numerics (no hallucinated numbers)
- Budget enforcement & cost tracking
- Deterministic execution for audit trails
- Hallucination detection & prompt injection defense

Quick Start (No API Key Required!):
    from greenlang.intelligence import create_provider, ChatSession
    from greenlang.intelligence.schemas.messages import ChatMessage, Role
    from greenlang.intelligence.runtime.budget import Budget

    # Works immediately - no setup required!
    provider = create_provider()  # Auto-detects (demo if no keys)
    session = ChatSession(provider)

    response = await session.chat(
        messages=[ChatMessage(role=Role.user, content="What's the grid intensity in CA?")],
        budget=Budget(max_usd=0.50)
    )
    print(response.text)

Upgrade to Production:
    export OPENAI_API_KEY=sk-...
    # Provider automatically uses real LLM instead of demo mode

Week 1 (INTL-101) Foundation:
- LLM provider abstraction
- Zero-config setup with FakeProvider
- Tool runtime & JSON schema validation
- Budget & telemetry
- Security (PromptGuard, HallucinationDetector)
"""

from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.schemas.tools import ToolDef, ToolCall, ToolChoice
from greenlang.intelligence.schemas.responses import ChatResponse, Usage, FinishReason, ProviderInfo
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.intelligence.runtime.session import ChatSession
from greenlang.intelligence.providers.base import LLMProvider, LLMProviderConfig
from greenlang.intelligence.factory import (
    create_provider,
    has_openai_key,
    has_anthropic_key,
    has_any_api_key,
)
from greenlang.intelligence.config import IntelligenceConfig, get_recommended_model
from greenlang.intelligence.verification import (
    HallucinationDetector,
    HallucinationDetected,
    NumericClaim,
    Citation,
)
from greenlang.intelligence.security import PromptGuard, PromptInjectionDetected

__version__ = "0.2.0"

__all__ = [
    # Core schemas
    "ChatMessage",
    "Role",
    "ToolDef",
    "ToolCall",
    "ToolChoice",
    "ChatResponse",
    "Usage",
    "FinishReason",
    "ProviderInfo",
    # Budget
    "Budget",
    "BudgetExceeded",
    # Session
    "ChatSession",
    # Provider abstraction
    "LLMProvider",
    "LLMProviderConfig",
    # Factory (main entry point)
    "create_provider",
    "has_openai_key",
    "has_anthropic_key",
    "has_any_api_key",
    # Configuration
    "IntelligenceConfig",
    "get_recommended_model",
    # Security
    "HallucinationDetector",
    "HallucinationDetected",
    "NumericClaim",
    "Citation",
    "PromptGuard",
    "PromptInjectionDetected",
]
