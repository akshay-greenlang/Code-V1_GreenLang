"""
GreenLang LLM Integration - Production-ready LLM infrastructure.

This package provides comprehensive LLM integration with:
- Multi-provider support (Anthropic, OpenAI, custom providers)
- Intelligent routing with automatic failover
- Circuit breaker protection
- Rate limiting and cost tracking
- Health monitoring
- Budget management

Example:
    >>> from llm import LLMRouter, CostTracker, RoutingStrategy
    >>> from llm.providers import AnthropicProvider, OpenAIProvider
    >>>
    >>> # Initialize router with cost optimization
    >>> router = LLMRouter(strategy=RoutingStrategy.LEAST_COST)
    >>>
    >>> # Register providers
    >>> router.register_provider("anthropic", AnthropicProvider(...))
    >>> router.register_provider("openai", OpenAIProvider(...))
    >>>
    >>> # Initialize cost tracking
    >>> tracker = CostTracker()
    >>> tracker.set_budget("tenant-123", monthly_limit_usd=1000.0)
    >>>
    >>> # Generate with automatic provider selection
    >>> response = await router.generate(request)
    >>>
    >>> # Track costs
    >>> tracker.track_usage(
    ...     provider=response.provider,
    ...     tenant_id="tenant-123",
    ...     agent_id="esg-agent",
    ...     model_id=response.model_id,
    ...     usage=response.usage
    ... )
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerStats,
    CircuitState,
)
from .cost_tracker import (
    BudgetConfig,
    BudgetStatus,
    CostRecord,
    CostSummary,
    CostTracker,
)
from .llm_router import (
    LLMRouter,
    ProviderMetrics,
    RegisteredProvider,
    RoutingStrategy,
)
from .providers.anthropic_provider import AnthropicProvider
from .providers.base_provider import (
    AuthenticationError,
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
    InvalidRequestError,
    ProviderError,
    ProviderHealth,
    RateLimitError,
    ServiceUnavailableError,
    TokenUsage,
)
from .providers.openai_provider import OpenAIProvider

__all__ = [
    # Core components
    "LLMRouter",
    "CostTracker",
    "CircuitBreaker",
    # Providers
    "AnthropicProvider",
    "OpenAIProvider",
    "BaseLLMProvider",
    # Router types
    "RoutingStrategy",
    "ProviderMetrics",
    "RegisteredProvider",
    # Cost tracking types
    "CostRecord",
    "CostSummary",
    "BudgetConfig",
    "BudgetStatus",
    # Circuit breaker types
    "CircuitState",
    "CircuitBreakerStats",
    "CircuitBreakerOpenError",
    # Request/Response types
    "GenerationRequest",
    "GenerationResponse",
    "TokenUsage",
    "ProviderHealth",
    # Exceptions
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "InvalidRequestError",
    "ServiceUnavailableError",
]

__version__ = "1.0.0"
