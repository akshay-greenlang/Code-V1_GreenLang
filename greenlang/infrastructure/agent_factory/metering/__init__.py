"""
Agent Factory Metering Module - INFRA-010

Provides cost tracking, budget management, resource quotas, and billing
event emission for agent execution in the GreenLang Climate OS platform.
Enables per-agent and per-tenant cost visibility, spend limits, and
downstream accounting integration.

Public API:
    - CostTracker: Per-agent cost tracking with background batch writes.
    - CostEntry: Single cost record dataclass.
    - CostCategory: Cost category enumeration.
    - CostRates: Configurable rates per resource unit.
    - CostSummary: Aggregated cost summary.
    - BudgetManager: Budget enforcement with alert thresholds.
    - Budget: Budget definition dataclass.
    - BudgetCheckResult: Outcome of a budget check.
    - BudgetExceededError: Raised when a hard budget is exceeded.
    - ResourceQuotaEnforcer: Execution quota enforcement via Redis.
    - ResourceQuota: Quota definition dataclass.
    - QuotaExceededError: Raised when a quota is exceeded.
    - BillingEventEmitter: Event emission for downstream accounting.
    - BillingEvent: Single billing event.
    - BillingEventType: Event type enumeration.

Example:
    >>> from greenlang.infrastructure.agent_factory.metering import (
    ...     CostTracker, BudgetManager, ResourceQuotaEnforcer,
    ... )
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.metering.billing_events import (
    BillingEvent,
    BillingEventEmitter,
    BillingEventType,
)
from greenlang.infrastructure.agent_factory.metering.budget import (
    Budget,
    BudgetCheckResult,
    BudgetExceededError,
    BudgetManager,
    BudgetPeriod,
)
from greenlang.infrastructure.agent_factory.metering.cost_tracker import (
    CostCategory,
    CostEntry,
    CostRates,
    CostSummary,
    CostTracker,
)
from greenlang.infrastructure.agent_factory.metering.resource_quotas import (
    QuotaCheckResult,
    QuotaExceededError,
    ResourceQuota,
    ResourceQuotaEnforcer,
)

__all__ = [
    # Cost Tracker
    "CostCategory",
    "CostEntry",
    "CostRates",
    "CostSummary",
    "CostTracker",
    # Budget
    "Budget",
    "BudgetCheckResult",
    "BudgetExceededError",
    "BudgetManager",
    "BudgetPeriod",
    # Quotas
    "QuotaCheckResult",
    "QuotaExceededError",
    "ResourceQuota",
    "ResourceQuotaEnforcer",
    # Billing Events
    "BillingEvent",
    "BillingEventEmitter",
    "BillingEventType",
]
