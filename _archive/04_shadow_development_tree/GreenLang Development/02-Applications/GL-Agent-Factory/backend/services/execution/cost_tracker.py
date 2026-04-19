"""
Cost Tracker

This module provides execution cost tracking for billing and cost optimization.
It tracks LLM token usage, compute time, and storage costs.

Example:
    >>> tracker = CostTracker()
    >>> cost = tracker.calculate_execution_cost(
    ...     execution_id="exec-123",
    ...     duration_ms=1500,
    ...     llm_tokens_input=1000,
    ...     llm_tokens_output=500
    ... )
    >>> print(f"Total cost: ${cost.total_usd:.4f}")
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMModel(str, Enum):
    """Supported LLM models with pricing."""

    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"


class CostBreakdown(BaseModel):
    """Detailed cost breakdown for an execution."""

    execution_id: str = Field(..., description="Execution identifier")
    compute_cost_usd: float = Field(0.0, description="Compute cost in USD")
    llm_cost_usd: float = Field(0.0, description="LLM API cost in USD")
    storage_cost_usd: float = Field(0.0, description="Storage cost in USD")
    total_usd: float = Field(0.0, description="Total cost in USD")

    # Detailed metrics
    compute_seconds: float = Field(0.0, description="Compute time in seconds")
    llm_tokens_input: int = Field(0, description="LLM input tokens")
    llm_tokens_output: int = Field(0, description="LLM output tokens")
    llm_tokens_cached: int = Field(0, description="LLM cached tokens")
    storage_bytes: int = Field(0, description="Storage used in bytes")

    model_used: Optional[str] = Field(None, description="LLM model used")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class CostTracker:
    """
    Cost Tracker for execution billing.

    This class tracks and calculates costs for:
    - Compute time (CPU seconds)
    - LLM API calls (token-based)
    - Storage usage (S3 bytes)

    The pricing is configurable and supports multiple LLM providers.

    Attributes:
        pricing: Cost rates for different resources
        aggregated_costs: Running totals by tenant/agent

    Example:
        >>> tracker = CostTracker()
        >>> cost = tracker.calculate_execution_cost(
        ...     execution_id="exec-1",
        ...     duration_ms=2000,
        ...     llm_tokens_input=1500,
        ...     llm_tokens_output=750
        ... )
        >>> print(f"LLM cost: ${cost.llm_cost_usd:.4f}")
    """

    # Default pricing rates
    DEFAULT_PRICING = {
        # Compute pricing (USD per second)
        "compute_per_second": 0.00002,

        # LLM pricing (USD per 1K tokens)
        "llm": {
            LLMModel.CLAUDE_3_5_SONNET: {
                "input_per_1k": 0.003,
                "output_per_1k": 0.015,
                "cached_per_1k": 0.0003,
            },
            LLMModel.CLAUDE_3_OPUS: {
                "input_per_1k": 0.015,
                "output_per_1k": 0.075,
                "cached_per_1k": 0.0015,
            },
            LLMModel.CLAUDE_3_HAIKU: {
                "input_per_1k": 0.00025,
                "output_per_1k": 0.00125,
                "cached_per_1k": 0.000025,
            },
            LLMModel.GPT_4_TURBO: {
                "input_per_1k": 0.01,
                "output_per_1k": 0.03,
                "cached_per_1k": 0.001,
            },
            LLMModel.GPT_4O: {
                "input_per_1k": 0.0025,
                "output_per_1k": 0.01,
                "cached_per_1k": 0.00025,
            },
        },

        # Storage pricing (USD per GB per month)
        "storage_per_gb_month": 0.023,
    }

    def __init__(
        self,
        pricing: Optional[Dict[str, Any]] = None,
        default_model: LLMModel = LLMModel.CLAUDE_3_5_SONNET,
    ):
        """
        Initialize the CostTracker.

        Args:
            pricing: Custom pricing rates (overrides defaults)
            default_model: Default LLM model for pricing
        """
        self.pricing = {**self.DEFAULT_PRICING, **(pricing or {})}
        self.default_model = default_model

        # Aggregated costs by tenant
        self._tenant_costs: Dict[str, float] = {}
        # Aggregated costs by agent
        self._agent_costs: Dict[str, float] = {}

    def calculate_execution_cost(
        self,
        execution_id: str,
        duration_ms: float = 0,
        llm_tokens_input: int = 0,
        llm_tokens_output: int = 0,
        llm_tokens_cached: int = 0,
        storage_bytes: int = 0,
        model: Optional[LLMModel] = None,
    ) -> CostBreakdown:
        """
        Calculate the total cost for an execution.

        Args:
            execution_id: Unique execution identifier
            duration_ms: Compute duration in milliseconds
            llm_tokens_input: Number of LLM input tokens
            llm_tokens_output: Number of LLM output tokens
            llm_tokens_cached: Number of cached LLM tokens
            storage_bytes: Storage used in bytes
            model: LLM model used (defaults to default_model)

        Returns:
            Detailed cost breakdown

        Example:
            >>> cost = tracker.calculate_execution_cost(
            ...     execution_id="exec-123",
            ...     duration_ms=2500,
            ...     llm_tokens_input=1000,
            ...     llm_tokens_output=500
            ... )
            >>> cost.total_usd > 0
            True
        """
        model = model or self.default_model
        compute_seconds = duration_ms / 1000.0

        # Calculate compute cost
        compute_cost = compute_seconds * self.pricing["compute_per_second"]

        # Calculate LLM cost
        llm_cost = self._calculate_llm_cost(
            model,
            llm_tokens_input,
            llm_tokens_output,
            llm_tokens_cached,
        )

        # Calculate storage cost (prorated per second)
        storage_gb = storage_bytes / (1024 * 1024 * 1024)
        storage_cost = (storage_gb * self.pricing["storage_per_gb_month"]) / (30 * 24 * 3600)
        storage_cost = storage_cost * compute_seconds

        total_cost = compute_cost + llm_cost + storage_cost

        breakdown = CostBreakdown(
            execution_id=execution_id,
            compute_cost_usd=round(compute_cost, 8),
            llm_cost_usd=round(llm_cost, 8),
            storage_cost_usd=round(storage_cost, 8),
            total_usd=round(total_cost, 8),
            compute_seconds=compute_seconds,
            llm_tokens_input=llm_tokens_input,
            llm_tokens_output=llm_tokens_output,
            llm_tokens_cached=llm_tokens_cached,
            storage_bytes=storage_bytes,
            model_used=model.value if model else None,
        )

        logger.debug(
            f"Calculated cost for {execution_id}: "
            f"${breakdown.total_usd:.6f} "
            f"(compute: ${breakdown.compute_cost_usd:.6f}, "
            f"llm: ${breakdown.llm_cost_usd:.6f}, "
            f"storage: ${breakdown.storage_cost_usd:.6f})"
        )

        return breakdown

    def _calculate_llm_cost(
        self,
        model: LLMModel,
        tokens_input: int,
        tokens_output: int,
        tokens_cached: int,
    ) -> float:
        """
        Calculate LLM API cost based on token usage.

        Args:
            model: LLM model used
            tokens_input: Input tokens
            tokens_output: Output tokens
            tokens_cached: Cached tokens (cheaper rate)

        Returns:
            Total LLM cost in USD
        """
        model_pricing = self.pricing["llm"].get(model)
        if not model_pricing:
            logger.warning(f"Unknown model {model}, using default pricing")
            model_pricing = self.pricing["llm"][self.default_model]

        input_cost = (tokens_input / 1000) * model_pricing["input_per_1k"]
        output_cost = (tokens_output / 1000) * model_pricing["output_per_1k"]
        cached_cost = (tokens_cached / 1000) * model_pricing["cached_per_1k"]

        return input_cost + output_cost + cached_cost

    def track_tenant_cost(
        self,
        tenant_id: str,
        cost: CostBreakdown,
    ) -> float:
        """
        Track cost for a tenant (for quota/billing).

        Args:
            tenant_id: Tenant identifier
            cost: Cost breakdown to add

        Returns:
            Updated total cost for tenant
        """
        if tenant_id not in self._tenant_costs:
            self._tenant_costs[tenant_id] = 0.0

        self._tenant_costs[tenant_id] += cost.total_usd

        logger.debug(
            f"Tenant {tenant_id} total cost: ${self._tenant_costs[tenant_id]:.4f}"
        )
        return self._tenant_costs[tenant_id]

    def track_agent_cost(
        self,
        agent_id: str,
        cost: CostBreakdown,
    ) -> float:
        """
        Track cost for an agent (for analytics).

        Args:
            agent_id: Agent identifier
            cost: Cost breakdown to add

        Returns:
            Updated total cost for agent
        """
        if agent_id not in self._agent_costs:
            self._agent_costs[agent_id] = 0.0

        self._agent_costs[agent_id] += cost.total_usd
        return self._agent_costs[agent_id]

    def get_tenant_total(self, tenant_id: str) -> float:
        """Get total cost for a tenant."""
        return self._tenant_costs.get(tenant_id, 0.0)

    def get_agent_total(self, agent_id: str) -> float:
        """Get total cost for an agent."""
        return self._agent_costs.get(agent_id, 0.0)

    def estimate_cost(
        self,
        expected_duration_ms: float = 1000,
        expected_llm_tokens: int = 1000,
        model: Optional[LLMModel] = None,
    ) -> float:
        """
        Estimate cost for a planned execution.

        Args:
            expected_duration_ms: Expected duration in ms
            expected_llm_tokens: Expected total LLM tokens
            model: LLM model to use

        Returns:
            Estimated cost in USD
        """
        # Assume 70% input, 30% output ratio
        tokens_input = int(expected_llm_tokens * 0.7)
        tokens_output = int(expected_llm_tokens * 0.3)

        breakdown = self.calculate_execution_cost(
            execution_id="estimate",
            duration_ms=expected_duration_ms,
            llm_tokens_input=tokens_input,
            llm_tokens_output=tokens_output,
            model=model,
        )

        return breakdown.total_usd

    def reset_tracking(self) -> None:
        """Reset all tracked costs."""
        self._tenant_costs.clear()
        self._agent_costs.clear()
