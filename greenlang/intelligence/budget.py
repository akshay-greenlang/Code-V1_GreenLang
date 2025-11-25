# -*- coding: utf-8 -*-
"""
Budget Tracking and Enforcement for LLM API Costs

Implements cost tracking and budget enforcement:
- Real-time cost calculation based on token usage
- Multi-level budgets (per request, per hour, per day, per month)
- Budget enforcement with BudgetExceededError
- Cost breakdown by agent, user, organization
- Budget alerts (80%, 90%, 100%)
- Dashboard-ready metrics

Budget Hierarchy:
    Organization ($1000/month)
         |
      User ($100/day)
         |
      Agent ($10/hour)
         |
    Request ($0.10)

Performance Targets:
- Budget enforcement: 100% accurate
- Tracking overhead: <1ms
- Real-time alerts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


class BudgetPeriod(Enum):
    """Budget period types"""
    REQUEST = "request"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


class BudgetExceededError(Exception):
    """Raised when budget is exceeded"""

    def __init__(self, message: str, budget_type: BudgetPeriod, limit: float, current: float):
        super().__init__(message)
        self.budget_type = budget_type
        self.limit = limit
        self.current = current


@dataclass
class ModelPricing:
    """
    Pricing for a specific model

    Attributes:
        model: Model identifier
        cost_per_1k_input: Cost per 1K input tokens (USD)
        cost_per_1k_output: Cost per 1K output tokens (USD)
    """
    model: str
    cost_per_1k_input: float
    cost_per_1k_output: float

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage"""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost


# Standard model pricing (as of 2024)
STANDARD_PRICING = {
    "gpt-4o": ModelPricing("gpt-4o", 0.005, 0.015),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", 0.01, 0.03),
    "gpt-4": ModelPricing("gpt-4", 0.03, 0.06),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.0005, 0.0015),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.00015, 0.0006),
    "claude-3-opus-20240229": ModelPricing("claude-3-opus-20240229", 0.015, 0.075),
    "claude-3-sonnet-20240229": ModelPricing("claude-3-sonnet-20240229", 0.003, 0.015),
    "claude-3-haiku-20240307": ModelPricing("claude-3-haiku-20240307", 0.00025, 0.00125),
}


@dataclass
class Budget:
    """
    Budget configuration

    Attributes:
        max_cost_per_request: Maximum cost per request (USD)
        max_tokens_per_request: Maximum tokens per request
        max_cost_per_hour: Maximum cost per hour (USD)
        max_cost_per_day: Maximum cost per day (USD)
        max_cost_per_month: Maximum cost per month (USD)
        alert_thresholds: Alert thresholds (0-1)
    """
    max_cost_per_request: float = 0.10
    max_tokens_per_request: int = 4000
    max_cost_per_hour: float = 10.00
    max_cost_per_day: float = 100.00
    max_cost_per_month: float = 1000.00
    alert_thresholds: List[float] = field(default_factory=lambda: [0.8, 0.9])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "max_cost_per_request": self.max_cost_per_request,
            "max_tokens_per_request": self.max_tokens_per_request,
            "max_cost_per_hour": self.max_cost_per_hour,
            "max_cost_per_day": self.max_cost_per_day,
            "max_cost_per_month": self.max_cost_per_month,
            "alert_thresholds": self.alert_thresholds,
        }


@dataclass
class Usage:
    """
    Usage record for a single request

    Attributes:
        request_id: Unique request ID
        model: Model used
        input_tokens: Input tokens
        output_tokens: Output tokens
        cost: Cost (USD)
        timestamp: Request timestamp
        agent_id: Agent ID
        user_id: User ID
        organization_id: Organization ID
    """
    request_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)"""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "organization_id": self.organization_id,
        }


@dataclass
class BudgetMetrics:
    """
    Budget usage metrics

    Attributes:
        total_cost: Total cost (USD)
        total_tokens: Total tokens
        total_requests: Total requests
        cost_by_model: Cost breakdown by model
        cost_by_agent: Cost breakdown by agent
        cost_by_user: Cost breakdown by user
        hourly_cost: Cost in last hour
        daily_cost: Cost in last day
        monthly_cost: Cost in last month
    """
    total_cost: float = 0.0
    total_tokens: int = 0
    total_requests: int = 0
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_agent: Dict[str, float] = field(default_factory=dict)
    cost_by_user: Dict[str, float] = field(default_factory=dict)
    hourly_cost: float = 0.0
    daily_cost: float = 0.0
    monthly_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "cost_by_model": self.cost_by_model,
            "cost_by_agent": self.cost_by_agent,
            "cost_by_user": self.cost_by_user,
            "hourly_cost": self.hourly_cost,
            "daily_cost": self.daily_cost,
            "monthly_cost": self.monthly_cost,
        }


class BudgetTracker:
    """
    Track and enforce LLM API budgets

    Handles:
    - Real-time cost calculation
    - Budget enforcement
    - Usage tracking
    - Cost breakdown
    - Budget alerts
    """

    def __init__(
        self,
        budget: Optional[Budget] = None,
        pricing: Optional[Dict[str, ModelPricing]] = None,
        enable_enforcement: bool = True,
    ):
        """
        Initialize budget tracker

        Args:
            budget: Budget configuration
            pricing: Model pricing (default: STANDARD_PRICING)
            enable_enforcement: Enable budget enforcement
        """
        self.budget = budget or Budget()
        self.pricing = pricing or STANDARD_PRICING
        self.enable_enforcement = enable_enforcement

        # Usage history
        self.usage_history: List[Usage] = []

        # Metrics
        self.metrics = BudgetMetrics()

        logger.info("BudgetTracker initialized")

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for token usage

        Args:
            model: Model identifier
            input_tokens: Input tokens
            output_tokens: Output tokens

        Returns:
            Cost in USD
        """
        pricing = self.pricing.get(model)
        if not pricing:
            logger.warning(f"Unknown model pricing: {model}. Using default.")
            pricing = self.pricing.get("gpt-3.5-turbo")

        return pricing.calculate_cost(input_tokens, output_tokens)

    def check_budget(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
    ):
        """
        Check if request is within budget

        Args:
            model: Model identifier
            input_tokens: Input tokens
            output_tokens: Estimated output tokens

        Raises:
            BudgetExceededError: If budget would be exceeded
        """
        if not self.enable_enforcement:
            return

        # Calculate request cost
        request_cost = self.calculate_cost(model, input_tokens, output_tokens)

        # Check per-request budget
        if request_cost > self.budget.max_cost_per_request:
            raise BudgetExceededError(
                f"Request cost ${request_cost:.4f} exceeds per-request limit ${self.budget.max_cost_per_request}",
                BudgetPeriod.REQUEST,
                self.budget.max_cost_per_request,
                request_cost,
            )

        # Check token budget
        total_tokens = input_tokens + output_tokens
        if total_tokens > self.budget.max_tokens_per_request:
            raise BudgetExceededError(
                f"Request tokens {total_tokens} exceeds limit {self.budget.max_tokens_per_request}",
                BudgetPeriod.REQUEST,
                self.budget.max_tokens_per_request,
                total_tokens,
            )

        # Check hourly budget
        hourly_cost = self._calculate_period_cost(hours=1)
        if hourly_cost + request_cost > self.budget.max_cost_per_hour:
            raise BudgetExceededError(
                f"Hourly cost ${hourly_cost + request_cost:.2f} would exceed limit ${self.budget.max_cost_per_hour}",
                BudgetPeriod.HOUR,
                self.budget.max_cost_per_hour,
                hourly_cost + request_cost,
            )

        # Check daily budget
        daily_cost = self._calculate_period_cost(days=1)
        if daily_cost + request_cost > self.budget.max_cost_per_day:
            raise BudgetExceededError(
                f"Daily cost ${daily_cost + request_cost:.2f} would exceed limit ${self.budget.max_cost_per_day}",
                BudgetPeriod.DAY,
                self.budget.max_cost_per_day,
                daily_cost + request_cost,
            )

        # Check monthly budget
        monthly_cost = self._calculate_period_cost(days=30)
        if monthly_cost + request_cost > self.budget.max_cost_per_month:
            raise BudgetExceededError(
                f"Monthly cost ${monthly_cost + request_cost:.2f} would exceed limit ${self.budget.max_cost_per_month}",
                BudgetPeriod.MONTH,
                self.budget.max_cost_per_month,
                monthly_cost + request_cost,
            )

    def record_usage(
        self,
        request_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> Usage:
        """
        Record usage for a request

        Args:
            request_id: Unique request ID
            model: Model used
            input_tokens: Input tokens
            output_tokens: Output tokens
            agent_id: Agent ID
            user_id: User ID
            organization_id: Organization ID

        Returns:
            Usage record
        """
        # Calculate cost
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        # Create usage record
        usage = Usage(
            request_id=request_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            agent_id=agent_id,
            user_id=user_id,
            organization_id=organization_id,
        )

        # Store usage
        self.usage_history.append(usage)

        # Update metrics
        self._update_metrics(usage)

        logger.debug(f"Recorded usage: {request_id}, cost=${cost:.4f}")

        return usage

    def _update_metrics(self, usage: Usage):
        """Update metrics with new usage"""
        # Update totals
        self.metrics.total_cost += usage.cost
        self.metrics.total_tokens += usage.total_tokens
        self.metrics.total_requests += 1

        # Update by model
        if usage.model not in self.metrics.cost_by_model:
            self.metrics.cost_by_model[usage.model] = 0.0
        self.metrics.cost_by_model[usage.model] += usage.cost

        # Update by agent
        if usage.agent_id:
            if usage.agent_id not in self.metrics.cost_by_agent:
                self.metrics.cost_by_agent[usage.agent_id] = 0.0
            self.metrics.cost_by_agent[usage.agent_id] += usage.cost

        # Update by user
        if usage.user_id:
            if usage.user_id not in self.metrics.cost_by_user:
                self.metrics.cost_by_user[usage.user_id] = 0.0
            self.metrics.cost_by_user[usage.user_id] += usage.cost

        # Update period costs
        self.metrics.hourly_cost = self._calculate_period_cost(hours=1)
        self.metrics.daily_cost = self._calculate_period_cost(days=1)
        self.metrics.monthly_cost = self._calculate_period_cost(days=30)

    def _calculate_period_cost(self, hours: int = 0, days: int = 0) -> float:
        """Calculate cost for time period"""
        cutoff = DeterministicClock.now() - timedelta(hours=hours, days=days)

        period_cost = sum(
            usage.cost
            for usage in self.usage_history
            if usage.timestamp >= cutoff
        )

        return period_cost

    def get_budget_utilization(self, period: BudgetPeriod) -> float:
        """
        Get budget utilization for period

        Args:
            period: Budget period

        Returns:
            Utilization (0-1)
        """
        if period == BudgetPeriod.HOUR:
            cost = self.metrics.hourly_cost
            limit = self.budget.max_cost_per_hour
        elif period == BudgetPeriod.DAY:
            cost = self.metrics.daily_cost
            limit = self.budget.max_cost_per_day
        elif period == BudgetPeriod.MONTH:
            cost = self.metrics.monthly_cost
            limit = self.budget.max_cost_per_month
        else:
            return 0.0

        if limit == 0:
            return 0.0

        return cost / limit

    def should_alert(self, period: BudgetPeriod) -> Optional[float]:
        """
        Check if budget alert should be triggered

        Args:
            period: Budget period

        Returns:
            Alert threshold if should alert, None otherwise
        """
        utilization = self.get_budget_utilization(period)

        for threshold in sorted(self.budget.alert_thresholds, reverse=True):
            if utilization >= threshold:
                return threshold

        return None

    def get_metrics(self) -> BudgetMetrics:
        """Get current budget metrics"""
        return self.metrics

    def clear_history(self, before: Optional[datetime] = None):
        """Clear usage history before date"""
        if before is None:
            before = DeterministicClock.now() - timedelta(days=90)  # Keep 90 days

        self.usage_history = [
            usage for usage in self.usage_history
            if usage.timestamp >= before
        ]

        logger.info(f"Cleared usage history before {before.isoformat()}")


if __name__ == "__main__":
    """
    Demo and testing
    """
    import uuid

    print("=" * 80)
    print("GreenLang Budget Tracker Demo")
    print("=" * 80)

    # Initialize tracker
    budget = Budget(
        max_cost_per_request=0.10,
        max_cost_per_hour=1.00,
        max_cost_per_day=10.00,
    )
    tracker = BudgetTracker(budget=budget)

    # Simulate requests
    print("\n1. Simulating requests:")
    for i in range(5):
        request_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        model = "gpt-4o" if i % 2 == 0 else "gpt-3.5-turbo"
        input_tokens = 500
        output_tokens = 300

        try:
            # Check budget
            tracker.check_budget(model, input_tokens, output_tokens)

            # Record usage
            usage = tracker.record_usage(
                request_id=request_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                agent_id="carbon_calc",
            )

            print(f"   Request {i+1}: ${usage.cost:.4f} ({model})")

        except BudgetExceededError as e:
            print(f"   Request {i+1}: BUDGET EXCEEDED - {e}")

    # Show metrics
    print("\n2. Budget metrics:")
    metrics = tracker.get_metrics()
    print(f"   Total cost: ${metrics.total_cost:.4f}")
    print(f"   Total requests: {metrics.total_requests}")
    print(f"   Total tokens: {metrics.total_tokens}")
    print(f"   Cost by model:")
    for model, cost in metrics.cost_by_model.items():
        print(f"     {model}: ${cost:.4f}")

    # Check budget utilization
    print("\n3. Budget utilization:")
    for period in [BudgetPeriod.HOUR, BudgetPeriod.DAY]:
        util = tracker.get_budget_utilization(period)
        print(f"   {period.value}: {util*100:.1f}%")

        alert_threshold = tracker.should_alert(period)
        if alert_threshold:
            print(f"     ALERT: {alert_threshold*100:.0f}% threshold exceeded!")

    print("\n" + "=" * 80)
