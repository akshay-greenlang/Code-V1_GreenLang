# -*- coding: utf-8 -*-
"""
DEPRECATED: This module is a compatibility shim.

Please update your imports from:
    from greenlang.intelligence import ...
To:
    from greenlang.agents.intelligence import ...

This shim will be removed in v0.5.0.
"""
import warnings

warnings.warn(
    "greenlang.intelligence is deprecated. "
    "Use greenlang.agents.intelligence instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from greenlang.agents.intelligence import *
from greenlang.agents.intelligence import (
    # Core schemas
    ChatMessage,
    Role,
    ToolDef,
    ToolCall,
    ToolChoice,
    ChatResponse,
    Usage,
    FinishReason,
    ProviderInfo,
    # Budget
    Budget,
    BudgetExceeded,
    # Session
    ChatSession,
    # Provider abstraction
    LLMProvider,
    LLMProviderConfig,
    # Factory (main entry point)
    create_provider,
    has_openai_key,
    has_anthropic_key,
    has_any_api_key,
    # Configuration
    IntelligenceConfig,
    get_recommended_model,
    # Security
    HallucinationDetector,
    HallucinationDetected,
    NumericClaim,
    Citation,
    PromptGuard,
    PromptInjectionDetected,
    # Phase 5: AI Optimization
    SemanticCache,
    get_global_cache,
    CacheEntry,
    CacheMetrics,
    CacheWarmer,
    warm_cache_on_startup,
    COMMON_QUERIES,
    PromptCompressor,
    CompressionResult,
    get_compression_metrics,
    stream_chat_completion,
    stream_to_sse,
    StreamToken,
    StreamingProvider,
    FallbackManager,
    ModelConfig,
    DEFAULT_FALLBACK_CHAIN,
    CircuitBreaker,
    QualityChecker,
    QualityScore,
    BudgetTracker,
    BudgetConfig,
    BudgetExceededError,
    UsageRecord,
    RequestBatcher,
    AdaptiveBatcher,
)
