"""
Model Router - Intelligent routing with cost optimization and fallback logic.

This module implements smart routing to select the best model based on:
- Task requirements (capabilities needed)
- Cost constraints (max cost per 1k tokens)
- Performance constraints (max latency)
- Certification requirements (zero-hallucination certified)
- Fallback logic (if primary model fails)

Example:
    >>> router = ModelRouter(model_registry)
    >>> criteria = RoutingCriteria(
    ...     capability=ModelCapability.CODE_GENERATION,
    ...     max_cost_per_1k_tokens=0.005,
    ...     certified_only=True
    ... )
    >>> model = router.select_model(criteria)
    >>> response = await router.invoke_with_fallback(model.id, prompt)
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
import logging
import asyncio
from enum import Enum

from greenlang.registry.model_registry import (
    model_registry,
    ModelProvider,
    ModelCapability,
    ModelMetadata
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class RoutingStrategy(str, Enum):
    """Routing strategy options."""
    LOWEST_COST = "lowest_cost"
    LOWEST_LATENCY = "lowest_latency"
    HIGHEST_QUALITY = "highest_quality"
    BALANCED = "balanced"


class RoutingCriteria(BaseModel):
    """Criteria for model selection."""

    # Requirements
    capability: Optional[ModelCapability] = Field(
        None,
        description="Required capability"
    )
    capabilities: Optional[List[ModelCapability]] = Field(
        None,
        description="Multiple required capabilities"
    )

    # Constraints
    max_cost_per_1k_tokens: Optional[float] = Field(
        None,
        ge=0.0,
        description="Maximum cost per 1k tokens"
    )
    max_latency_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="Maximum latency in milliseconds"
    )
    min_context_window: Optional[int] = Field(
        None,
        ge=0,
        description="Minimum context window size"
    )

    # Filters
    provider: Optional[ModelProvider] = Field(
        None,
        description="Specific provider to use"
    )
    certified_only: bool = Field(
        True,
        description="Only use zero-hallucination certified models"
    )

    # Strategy
    strategy: RoutingStrategy = Field(
        RoutingStrategy.BALANCED,
        description="Routing strategy"
    )

    # Fallback
    enable_fallback: bool = Field(
        True,
        description="Enable fallback to alternative models"
    )
    max_fallback_attempts: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum fallback attempts"
    )


class RoutingDecision(BaseModel):
    """Result of routing decision."""

    primary_model: ModelMetadata = Field(..., description="Selected primary model")
    fallback_models: List[ModelMetadata] = Field(
        default_factory=list,
        description="Fallback models (in order)"
    )
    reason: str = Field(..., description="Selection reason")
    alternatives_considered: int = Field(..., description="Number of models considered")
    strategy_used: RoutingStrategy


class InvocationResult(BaseModel):
    """Result of model invocation with routing."""

    model_id: str = Field(..., description="Model used")
    response: str = Field(..., description="Model response")
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    fallback_used: bool = Field(False, description="Whether fallback was used")
    attempts: int = Field(1, description="Number of attempts")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# MODEL ROUTER
# ============================================================================

class ModelRouter:
    """
    Intelligent model router with cost optimization and fallback logic.

    The router:
    1. Selects best model based on criteria
    2. Attempts invocation with primary model
    3. Falls back to alternative models if primary fails
    4. Tracks routing decisions and performance
    5. Optimizes for cost when multiple models are suitable

    Example:
        >>> router = ModelRouter()
        >>> criteria = RoutingCriteria(
        ...     capability=ModelCapability.TEXT_GENERATION,
        ...     max_cost_per_1k_tokens=0.005
        ... )
        >>> decision = router.route(criteria)
        >>> print(f"Selected: {decision.primary_model.name}")
    """

    def __init__(self, registry=None):
        """Initialize model router."""
        self.registry = registry or model_registry
        self.routing_history: List[Dict[str, Any]] = []

    def route(self, criteria: RoutingCriteria) -> RoutingDecision:
        """
        Route request to best model based on criteria.

        Args:
            criteria: Routing criteria

        Returns:
            Routing decision with primary and fallback models

        Raises:
            ValueError: If no suitable model found
        """
        logger.info(f"Routing request with strategy: {criteria.strategy}")

        # Get candidate models
        candidates = self._get_candidate_models(criteria)

        if not candidates:
            raise ValueError(
                f"No models found matching criteria: {criteria.dict()}"
            )

        logger.info(f"Found {len(candidates)} candidate models")

        # Select primary model based on strategy
        primary_model = self._select_by_strategy(candidates, criteria.strategy)

        # Select fallback models
        fallback_models = []
        if criteria.enable_fallback:
            fallback_models = [
                m for m in candidates
                if m.id != primary_model.id
            ][:criteria.max_fallback_attempts]

        decision = RoutingDecision(
            primary_model=primary_model,
            fallback_models=fallback_models,
            reason=self._generate_selection_reason(primary_model, criteria),
            alternatives_considered=len(candidates),
            strategy_used=criteria.strategy
        )

        # Track decision
        self.routing_history.append({
            "timestamp": datetime.utcnow(),
            "criteria": criteria.dict(),
            "decision": decision.dict()
        })

        logger.info(
            f"Routed to {primary_model.name} "
            f"({len(fallback_models)} fallbacks available)"
        )

        return decision

    def select_model(self, criteria: RoutingCriteria) -> ModelMetadata:
        """
        Select best model (shortcut that returns only primary model).

        Args:
            criteria: Routing criteria

        Returns:
            Selected model metadata
        """
        decision = self.route(criteria)
        return decision.primary_model

    async def invoke_with_fallback(
        self,
        criteria: RoutingCriteria,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> InvocationResult:
        """
        Invoke model with automatic fallback on failure.

        Args:
            criteria: Routing criteria
            prompt: Input prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Invocation result with response and metadata
        """
        decision = self.route(criteria)

        # Try primary model
        models_to_try = [decision.primary_model] + decision.fallback_models
        errors = []

        for attempt, model in enumerate(models_to_try, 1):
            try:
                logger.info(f"Attempt {attempt}: Invoking {model.name}")

                result = await self._invoke_model(
                    model_id=model.id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                return InvocationResult(
                    model_id=model.id,
                    response=result["response"],
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    latency_ms=result["latency_ms"],
                    cost_usd=result["cost_usd"],
                    fallback_used=attempt > 1,
                    attempts=attempt,
                    errors=errors
                )

            except Exception as e:
                error_msg = f"{model.name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Invocation failed: {error_msg}")

                if attempt >= len(models_to_try):
                    # All attempts exhausted
                    raise RuntimeError(
                        f"All {attempt} invocation attempts failed. Errors: {errors}"
                    )

        # Should never reach here
        raise RuntimeError("Unexpected error in fallback logic")

    def _get_candidate_models(self, criteria: RoutingCriteria) -> List[ModelMetadata]:
        """Get candidate models matching criteria."""
        # Start with all models or filter by provider
        if criteria.provider:
            candidates = self.registry.list_models(provider=criteria.provider)
        else:
            candidates = list(self.registry.models.values())

        # Filter by certification
        if criteria.certified_only:
            candidates = [
                m for m in candidates
                if m.certified_for_zero_hallucination
            ]

        # Filter by single capability
        if criteria.capability:
            candidates = [
                m for m in candidates
                if criteria.capability in m.capabilities
            ]

        # Filter by multiple capabilities
        if criteria.capabilities:
            candidates = [
                m for m in candidates
                if all(cap in m.capabilities for cap in criteria.capabilities)
            ]

        # Filter by cost constraint
        if criteria.max_cost_per_1k_tokens:
            candidates = [
                m for m in candidates
                if m.avg_cost_per_1k_tokens
                and m.avg_cost_per_1k_tokens <= criteria.max_cost_per_1k_tokens
            ]

        # Filter by latency constraint
        if criteria.max_latency_ms:
            candidates = [
                m for m in candidates
                if m.avg_latency_ms
                and m.avg_latency_ms <= criteria.max_latency_ms
            ]

        # Filter by context window
        if criteria.min_context_window:
            candidates = [
                m for m in candidates
                if m.context_window >= criteria.min_context_window
            ]

        return candidates

    def _select_by_strategy(
        self,
        candidates: List[ModelMetadata],
        strategy: RoutingStrategy
    ) -> ModelMetadata:
        """Select best model based on strategy."""
        if not candidates:
            raise ValueError("No candidate models available")

        if strategy == RoutingStrategy.LOWEST_COST:
            # Sort by cost (lowest first)
            candidates.sort(
                key=lambda m: m.avg_cost_per_1k_tokens or float('inf')
            )
            return candidates[0]

        elif strategy == RoutingStrategy.LOWEST_LATENCY:
            # Sort by latency (lowest first)
            candidates.sort(
                key=lambda m: m.avg_latency_ms or float('inf')
            )
            return candidates[0]

        elif strategy == RoutingStrategy.HIGHEST_QUALITY:
            # Prefer certified models, then lowest cost
            certified = [m for m in candidates if m.certified_for_zero_hallucination]
            if certified:
                return certified[0]
            return candidates[0]

        elif strategy == RoutingStrategy.BALANCED:
            # Balance cost and quality
            # Score = cost_score * 0.5 + quality_score * 0.5
            def balanced_score(model: ModelMetadata) -> float:
                cost_score = model.avg_cost_per_1k_tokens or 0.0
                quality_score = -1.0 if model.certified_for_zero_hallucination else 0.0
                return cost_score * 0.5 + quality_score * 0.5

            candidates.sort(key=balanced_score)
            return candidates[0]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _generate_selection_reason(
        self,
        model: ModelMetadata,
        criteria: RoutingCriteria
    ) -> str:
        """Generate human-readable selection reason."""
        reasons = []

        if criteria.strategy == RoutingStrategy.LOWEST_COST:
            reasons.append(
                f"lowest cost (${model.avg_cost_per_1k_tokens:.6f}/1k tokens)"
            )
        elif criteria.strategy == RoutingStrategy.LOWEST_LATENCY:
            reasons.append(
                f"lowest latency ({model.avg_latency_ms:.1f}ms avg)"
            )
        elif criteria.strategy == RoutingStrategy.HIGHEST_QUALITY:
            reasons.append("highest quality")
        elif criteria.strategy == RoutingStrategy.BALANCED:
            reasons.append("best cost/quality balance")

        if model.certified_for_zero_hallucination:
            reasons.append("zero-hallucination certified")

        return f"Selected {model.name}: " + ", ".join(reasons)

    async def _invoke_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Invoke model (mock implementation).

        Returns:
            Dictionary with response, tokens, latency, cost
        """
        import time

        # TODO: Replace with actual LLM SDK calls
        model = self.registry.get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        start_time = time.time()

        # Mock response
        response = f"[{model.name}] Processed: {prompt[:50]}..."

        # Mock token counts
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost_usd = 0.0
        if model.avg_cost_per_1k_tokens:
            cost_usd = (total_tokens / 1000.0) * model.avg_cost_per_1k_tokens

        latency_ms = (time.time() - start_time) * 1000

        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd
        }

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.routing_history:
            return {
                "total_routes": 0,
                "models_used": {},
                "strategies_used": {}
            }

        models_used = {}
        strategies_used = {}

        for entry in self.routing_history:
            model_id = entry["decision"]["primary_model"]["id"]
            strategy = entry["decision"]["strategy_used"]

            models_used[model_id] = models_used.get(model_id, 0) + 1
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1

        return {
            "total_routes": len(self.routing_history),
            "models_used": models_used,
            "strategies_used": strategies_used,
            "last_route_time": self.routing_history[-1]["timestamp"]
        }


# ============================================================================
# COST OPTIMIZER
# ============================================================================

class CostOptimizer:
    """
    Optimizes cost by selecting cheaper models when acceptable.

    Uses heuristics to determine when a cheaper model is sufficient:
    - Prompt complexity analysis
    - Task type classification
    - Historical performance data
    """

    def __init__(self, registry=None):
        """Initialize cost optimizer."""
        self.registry = registry or model_registry

    def should_use_cheaper_model(
        self,
        prompt: str,
        capability: ModelCapability,
        quality_threshold: float = 0.95
    ) -> bool:
        """
        Determine if a cheaper model is acceptable.

        Args:
            prompt: Input prompt
            capability: Required capability
            quality_threshold: Minimum acceptable quality (0-1)

        Returns:
            True if cheaper model should be used
        """
        # Heuristics for when cheaper models are sufficient:

        # 1. Short prompts (< 100 characters)
        if len(prompt) < 100:
            return True

        # 2. Simple queries
        simple_keywords = ["what is", "define", "list", "summarize"]
        if any(keyword in prompt.lower() for keyword in simple_keywords):
            return True

        # 3. Non-critical capabilities
        non_critical = [
            ModelCapability.TEXT_GENERATION,
        ]
        if capability in non_critical:
            return True

        # Default: use primary model
        return False

    def optimize_routing_criteria(
        self,
        base_criteria: RoutingCriteria,
        prompt: str
    ) -> RoutingCriteria:
        """
        Optimize routing criteria based on prompt analysis.

        Args:
            base_criteria: Base routing criteria
            prompt: Input prompt

        Returns:
            Optimized routing criteria
        """
        optimized = base_criteria.copy()

        # If cheaper model is acceptable, reduce cost constraint
        if base_criteria.capability and self.should_use_cheaper_model(
            prompt,
            base_criteria.capability
        ):
            # Prefer lower cost
            optimized.strategy = RoutingStrategy.LOWEST_COST
            logger.info("Cost optimizer: Using cheaper model for simple task")

        return optimized


# ============================================================================
# LOAD BALANCER
# ============================================================================

class LoadBalancer:
    """
    Load balances requests across multiple models.

    Useful for:
    - Distributing load across models
    - Avoiding rate limits
    - A/B testing
    """

    def __init__(self, models: List[str], weights: Optional[List[float]] = None):
        """
        Initialize load balancer.

        Args:
            models: List of model IDs
            weights: Optional weights for each model (must sum to 1.0)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.models) != len(self.weights):
            raise ValueError("models and weights must have same length")

        if abs(sum(self.weights) - 1.0) > 0.01:
            raise ValueError("weights must sum to 1.0")

        self.request_count = 0

    def select_model(self) -> str:
        """
        Select model using weighted round-robin.

        Returns:
            Selected model ID
        """
        import random

        # Weighted random selection
        selected = random.choices(self.models, weights=self.weights, k=1)[0]
        self.request_count += 1

        return selected
