# -*- coding: utf-8 -*-
"""
Provider Router - Intelligent Provider/Model Selection

Routes queries to optimal provider/model based on:
- Query complexity (simple calc vs complex analysis)
- Budget constraints (cost per query)
- Latency requirements (realtime vs batch)
- Climate domain context (regional data availability)

Architecture:
    Query → analyze_complexity() → select_provider() → (provider, model)

Routing Strategy:
- Simple calculations: OpenAI GPT-4o-mini (cheap, fast)
- Complex scenario analysis: Anthropic Claude-3.5-Sonnet (reasoning)
- Multi-step planning: OpenAI o1-preview (chain-of-thought)
- Large context: Anthropic Claude-3 (200K context window)

Cost Table (Q4 2025):
- gpt-4o-mini: $0.15/1M input, $0.60/1M output → ~$0.0002/query
- gpt-4o: $5/1M input, $15/1M output → ~$0.01/query
- claude-3-sonnet: $3/1M input, $15/1M output → ~$0.008/query
- claude-3-opus: $15/1M input, $75/1M output → ~$0.04/query

Example:
    router = ProviderRouter()

    # Simple query
    provider_name, model = router.select_provider(
        query_type="simple_calc",
        budget_cents=5,
        latency_req="realtime"
    )
    # Returns: ("openai", "gpt-4o-mini")

    # Complex analysis
    provider_name, model = router.select_provider(
        query_type="complex_analysis",
        budget_cents=50,
        latency_req="batch"
    )
    # Returns: ("anthropic", "claude-3-sonnet-20240229")
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query complexity classification"""

    SIMPLE_CALC = "simple_calc"  # Basic calculations, lookups
    STANDARD_QUERY = "standard_query"  # Typical questions
    COMPLEX_ANALYSIS = "complex_analysis"  # Multi-step reasoning
    SCENARIO_PLANNING = "scenario_planning"  # Long-term projections
    LARGE_CONTEXT = "large_context"  # Requires >32K context


class LatencyRequirement(str, Enum):
    """Latency tolerance"""

    REALTIME = "realtime"  # <3s (user-facing)
    INTERACTIVE = "interactive"  # <10s (dashboards)
    BATCH = "batch"  # <60s (background jobs)


class ProviderRouter:
    """
    Intelligent provider/model selection for cost and performance optimization

    Routes queries based on:
    1. Query complexity (simple → cheap, complex → capable)
    2. Budget (cents available per query)
    3. Latency requirement (realtime → fast model, batch → capable model)
    4. Climate context (regional data, sector-specific knowledge)

    Usage:
        router = ProviderRouter()

        # For simple query
        provider, model = router.select_provider(
            query_type=QueryType.SIMPLE_CALC,
            budget_cents=5,
            latency_req=LatencyRequirement.REALTIME
        )

        # For complex scenario
        provider, model = router.select_provider(
            query_type=QueryType.COMPLEX_ANALYSIS,
            budget_cents=50,
            latency_req=LatencyRequirement.BATCH,
            climate_context=ClimateContext(region="US-CA", sector="buildings")
        )
    """

    def __init__(self):
        """Initialize router with provider/model registry"""
        # Model registry: (provider, model) → (cost_per_1k_tokens, latency_score, capability_score)
        # Cost: USD per 1000 tokens (avg of input+output)
        # Latency: 1-10 scale (1=fastest, 10=slowest)
        # Capability: 1-10 scale (1=basic, 10=most capable)
        self.models = {
            ("openai", "gpt-4o-mini"): {
                "cost_per_1k_tokens": 0.000375,  # ($0.15 + $0.60) / 2 / 1000
                "latency_score": 2,
                "capability_score": 6,
                "context_window": 128_000,
            },
            ("openai", "gpt-4o"): {
                "cost_per_1k_tokens": 0.01,  # ($5 + $15) / 2 / 1000
                "latency_score": 4,
                "capability_score": 8,
                "context_window": 128_000,
            },
            ("openai", "gpt-4-turbo"): {
                "cost_per_1k_tokens": 0.02,  # ($10 + $30) / 2 / 1000
                "latency_score": 6,
                "capability_score": 9,
                "context_window": 128_000,
            },
            ("anthropic", "claude-3-haiku-20240307"): {
                "cost_per_1k_tokens": 0.000625,  # ($0.25 + $1.25) / 2 / 1000
                "latency_score": 1,
                "capability_score": 5,
                "context_window": 200_000,
            },
            ("anthropic", "claude-3-sonnet-20240229"): {
                "cost_per_1k_tokens": 0.009,  # ($3 + $15) / 2 / 1000
                "latency_score": 3,
                "capability_score": 8,
                "context_window": 200_000,
            },
            ("anthropic", "claude-3-opus-20240229"): {
                "cost_per_1k_tokens": 0.045,  # ($15 + $75) / 2 / 1000
                "latency_score": 7,
                "capability_score": 10,
                "context_window": 200_000,
            },
        }

    def select_provider(
        self,
        query_type: str | QueryType,
        budget_cents: int,
        latency_req: str | LatencyRequirement = LatencyRequirement.INTERACTIVE,
        climate_context: Optional[Any] = None,
        estimated_tokens: int = 2000,
    ) -> Tuple[str, str]:
        """
        Select optimal provider and model

        Routing logic:
        1. Filter models that fit budget
        2. Filter models that meet latency requirement
        3. Filter models that have required capabilities
        4. Select cheapest model that meets requirements

        Args:
            query_type: Query complexity (simple_calc, complex_analysis, etc.)
            budget_cents: Budget in cents (e.g., 5 = $0.05)
            latency_req: Latency requirement (realtime, interactive, batch)
            climate_context: Optional ClimateContext (unused for now, future use)
            estimated_tokens: Estimated token count (default: 2000)

        Returns:
            Tuple of (provider_name, model_name)

        Raises:
            ValueError: If no model meets requirements

        Example:
            >>> router = ProviderRouter()
            >>> provider, model = router.select_provider(
            ...     query_type="simple_calc",
            ...     budget_cents=5,
            ...     latency_req="realtime"
            ... )
            >>> provider
            'openai'
            >>> "mini" in model
            True
        """
        # Convert to enums
        if isinstance(query_type, str):
            query_type = QueryType(query_type)
        if isinstance(latency_req, str):
            latency_req = LatencyRequirement(latency_req)

        # Get required capability level
        required_capability = self._get_required_capability(query_type)

        # Get latency threshold
        latency_threshold = self._get_latency_threshold(latency_req)

        # Convert budget to USD
        budget_usd = budget_cents / 100.0

        # Calculate max cost per 1k tokens
        max_cost_per_1k = budget_usd / (estimated_tokens / 1000.0)

        logger.debug(
            f"Routing query: type={query_type.value}, budget=${budget_usd:.3f}, "
            f"latency={latency_req.value}, required_capability={required_capability}, "
            f"max_cost_per_1k=${max_cost_per_1k:.6f}"
        )

        # Filter candidates
        candidates = []

        for (provider, model), specs in self.models.items():
            # Check budget
            if specs["cost_per_1k_tokens"] > max_cost_per_1k:
                logger.debug(
                    f"  {provider}/{model}: SKIP (cost ${specs['cost_per_1k_tokens']:.6f} > ${max_cost_per_1k:.6f})"
                )
                continue

            # Check latency
            if specs["latency_score"] > latency_threshold:
                logger.debug(
                    f"  {provider}/{model}: SKIP (latency {specs['latency_score']} > {latency_threshold})"
                )
                continue

            # Check capability
            if specs["capability_score"] < required_capability:
                logger.debug(
                    f"  {provider}/{model}: SKIP (capability {specs['capability_score']} < {required_capability})"
                )
                continue

            logger.debug(
                f"  {provider}/{model}: CANDIDATE (cost=${specs['cost_per_1k_tokens']:.6f}, "
                f"latency={specs['latency_score']}, capability={specs['capability_score']})"
            )

            candidates.append((provider, model, specs))

        if not candidates:
            raise ValueError(
                f"No model meets requirements: query_type={query_type.value}, "
                f"budget=${budget_usd:.3f}, latency={latency_req.value}. "
                f"Try increasing budget or relaxing latency requirement."
            )

        # Select cheapest candidate
        candidates.sort(key=lambda x: x[2]["cost_per_1k_tokens"])
        provider, model, specs = candidates[0]

        logger.info(
            f"Selected provider: {provider}/{model} "
            f"(cost=${specs['cost_per_1k_tokens']:.6f}/1k tokens, "
            f"latency={specs['latency_score']}, capability={specs['capability_score']})"
        )

        return provider, model

    def _get_required_capability(self, query_type: QueryType) -> int:
        """
        Map query type to required capability score

        Args:
            query_type: Query complexity

        Returns:
            Required capability score (1-10)
        """
        mapping = {
            QueryType.SIMPLE_CALC: 5,  # Basic models OK
            QueryType.STANDARD_QUERY: 6,  # Mid-tier models
            QueryType.COMPLEX_ANALYSIS: 8,  # High-capability models
            QueryType.SCENARIO_PLANNING: 9,  # Top-tier reasoning
            QueryType.LARGE_CONTEXT: 7,  # Need large context window
        }
        return mapping.get(query_type, 6)

    def _get_latency_threshold(self, latency_req: LatencyRequirement) -> int:
        """
        Map latency requirement to max latency score

        Args:
            latency_req: Latency tolerance

        Returns:
            Max latency score (1-10)
        """
        mapping = {
            LatencyRequirement.REALTIME: 3,  # Very fast models only
            LatencyRequirement.INTERACTIVE: 5,  # Moderate speed OK
            LatencyRequirement.BATCH: 10,  # Any speed OK
        }
        return mapping.get(latency_req, 5)

    def estimate_cost(
        self,
        provider: str,
        model: str,
        estimated_tokens: int = 2000,
    ) -> float:
        """
        Estimate cost for provider/model

        Args:
            provider: Provider name
            model: Model name
            estimated_tokens: Token count

        Returns:
            Estimated cost in USD

        Example:
            >>> router = ProviderRouter()
            >>> cost = router.estimate_cost("openai", "gpt-4o-mini", 2000)
            >>> cost < 0.001
            True
        """
        specs = self.models.get((provider, model))
        if not specs:
            raise ValueError(f"Unknown model: {provider}/{model}")

        return specs["cost_per_1k_tokens"] * (estimated_tokens / 1000.0)

    def get_available_models(self) -> list[Tuple[str, str]]:
        """
        Get list of available (provider, model) pairs

        Returns:
            List of (provider, model) tuples

        Example:
            >>> router = ProviderRouter()
            >>> models = router.get_available_models()
            >>> ("openai", "gpt-4o-mini") in models
            True
        """
        return list(self.models.keys())


# Global router instance
_global_router: Optional[ProviderRouter] = None


def get_global_router() -> ProviderRouter:
    """
    Get global provider router (singleton)

    Returns:
        Global ProviderRouter instance

    Example:
        router = get_global_router()
        provider, model = router.select_provider(...)
    """
    global _global_router
    if _global_router is None:
        _global_router = ProviderRouter()
    return _global_router
