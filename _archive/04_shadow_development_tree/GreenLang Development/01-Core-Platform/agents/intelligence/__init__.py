# -*- coding: utf-8 -*-
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
    from greenlang.agents.intelligence.schemas.messages import ChatMessage, Role
    from greenlang.agents.intelligence.runtime.budget import Budget

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

Phase 5 (AI Optimization):
- Semantic caching with vector embeddings (>30% cache hit rate)
- Prompt compression (>20% token reduction)
- Streaming responses with SSE
- Model fallback chains with circuit breaker
- Quality validation and confidence scoring
- Budget tracking and enforcement
- Request batching for throughput optimization
"""

from greenlang.agents.intelligence.schemas.messages import ChatMessage, Role
from greenlang.agents.intelligence.schemas.tools import ToolDef, ToolCall, ToolChoice
from greenlang.agents.intelligence.schemas.responses import ChatResponse, Usage, FinishReason, ProviderInfo
from greenlang.agents.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.agents.intelligence.runtime.session import ChatSession
from greenlang.agents.intelligence.providers.base import LLMProvider, LLMProviderConfig
from greenlang.agents.intelligence.factory import (
    create_provider,
    has_openai_key,
    has_anthropic_key,
    has_any_api_key,
)
from greenlang.agents.intelligence.config import IntelligenceConfig, get_recommended_model
from greenlang.agents.intelligence.verification import (
    HallucinationDetector,
    HallucinationDetected,
    NumericClaim,
    Citation,
)
from greenlang.agents.intelligence.security import PromptGuard, PromptInjectionDetected

# Phase 5: AI Optimization
from greenlang.agents.intelligence.semantic_cache import (
    SemanticCache,
    get_global_cache,
    CacheEntry,
    CacheMetrics,
)
from greenlang.agents.intelligence.cache_warming import (
    CacheWarmer,
    warm_cache_on_startup,
    COMMON_QUERIES,
)
from greenlang.agents.intelligence.prompt_compression import (
    PromptCompressor,
    CompressionResult,
    get_compression_metrics,
)
from greenlang.agents.intelligence.streaming import (
    stream_chat_completion,
    stream_to_sse,
    StreamToken,
    StreamingProvider,
)
from greenlang.agents.intelligence.fallback import (
    FallbackManager,
    ModelConfig,
    DEFAULT_FALLBACK_CHAIN,
    CircuitBreaker,
)
from greenlang.agents.intelligence.quality_check import (
    QualityChecker,
    QualityScore,
)
from greenlang.agents.intelligence.budget import (
    BudgetTracker,
    Budget as BudgetConfig,
    BudgetExceededError,
    Usage as UsageRecord,
)
from greenlang.agents.intelligence.request_batching import (
    RequestBatcher,
    AdaptiveBatcher,
)

__version__ = "0.3.0"

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
    # Phase 5: AI Optimization
    "SemanticCache",
    "get_global_cache",
    "CacheEntry",
    "CacheMetrics",
    "CacheWarmer",
    "warm_cache_on_startup",
    "COMMON_QUERIES",
    "PromptCompressor",
    "CompressionResult",
    "get_compression_metrics",
    "stream_chat_completion",
    "stream_to_sse",
    "StreamToken",
    "StreamingProvider",
    "FallbackManager",
    "ModelConfig",
    "DEFAULT_FALLBACK_CHAIN",
    "CircuitBreaker",
    "QualityChecker",
    "QualityScore",
    "BudgetTracker",
    "BudgetConfig",
    "BudgetExceededError",
    "UsageRecord",
    "RequestBatcher",
    "AdaptiveBatcher",
]
