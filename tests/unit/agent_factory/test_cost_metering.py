# -*- coding: utf-8 -*-
"""
Unit tests for Agent Factory Cost Metering: cost tracking, budget
management, resource quotas, and billing event emission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ============================================================================
# Inline Implementations (contract definitions)
# ============================================================================


@dataclass
class CostRecord:
    agent_key: str
    tenant_id: str
    cost_usd: float
    resource_type: str  # "compute", "storage", "api_call"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostTracker:
    def __init__(self) -> None:
        self._records: List[CostRecord] = []

    def record(self, record: CostRecord) -> None:
        self._records.append(record)

    def query_by_agent(
        self,
        agent_key: str,
        since: Optional[float] = None,
    ) -> List[CostRecord]:
        results = [r for r in self._records if r.agent_key == agent_key]
        if since is not None:
            results = [r for r in results if r.timestamp >= since]
        return results

    def query_by_tenant(
        self,
        tenant_id: str,
        since: Optional[float] = None,
    ) -> List[CostRecord]:
        results = [r for r in self._records if r.tenant_id == tenant_id]
        if since is not None:
            results = [r for r in results if r.timestamp >= since]
        return results

    def total_cost_by_agent(self, agent_key: str) -> float:
        return sum(r.cost_usd for r in self._records if r.agent_key == agent_key)

    def total_cost_by_tenant(self, tenant_id: str) -> float:
        return sum(r.cost_usd for r in self._records if r.tenant_id == tenant_id)


class BudgetStatus(str, Enum):
    WITHIN = "within"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass
class BudgetConfig:
    tenant_id: str
    monthly_limit_usd: float
    warning_threshold_pct: float = 0.80  # 80%
    hard_limit_pct: float = 1.0  # 100%
    period_start: float = field(default_factory=time.time)


class BudgetManager:
    def __init__(self, cost_tracker: CostTracker) -> None:
        self._tracker = cost_tracker
        self._budgets: Dict[str, BudgetConfig] = {}
        self._events: List[Dict[str, Any]] = []

    def set_budget(self, config: BudgetConfig) -> None:
        self._budgets[config.tenant_id] = config

    def check_budget(self, tenant_id: str) -> BudgetStatus:
        config = self._budgets.get(tenant_id)
        if config is None:
            return BudgetStatus.WITHIN

        spent = self._tracker.total_cost_by_tenant(tenant_id)
        ratio = spent / config.monthly_limit_usd if config.monthly_limit_usd > 0 else 0

        if ratio >= config.hard_limit_pct:
            self._emit_event(tenant_id, "budget_blocked", spent)
            return BudgetStatus.BLOCKED
        if ratio >= config.warning_threshold_pct:
            self._emit_event(tenant_id, "budget_warning", spent)
            return BudgetStatus.WARNING
        return BudgetStatus.WITHIN

    def allow_execution(self, tenant_id: str) -> bool:
        return self.check_budget(tenant_id) != BudgetStatus.BLOCKED

    def reset_period(self, tenant_id: str) -> None:
        config = self._budgets.get(tenant_id)
        if config:
            self._budgets[tenant_id] = BudgetConfig(
                tenant_id=config.tenant_id,
                monthly_limit_usd=config.monthly_limit_usd,
                warning_threshold_pct=config.warning_threshold_pct,
                hard_limit_pct=config.hard_limit_pct,
                period_start=time.time(),
            )

    def _emit_event(self, tenant_id: str, event_type: str, spent: float) -> None:
        self._events.append({
            "tenant_id": tenant_id,
            "event_type": event_type,
            "spent_usd": spent,
            "timestamp": time.time(),
        })

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)


@dataclass
class ResourceQuota:
    tenant_id: str
    max_agents: int = 50
    max_concurrent_executions: int = 20
    max_storage_gb: float = 100.0


class ResourceQuotaManager:
    def __init__(self) -> None:
        self._quotas: Dict[str, ResourceQuota] = {}
        self._usage: Dict[str, Dict[str, float]] = {}

    def set_quota(self, quota: ResourceQuota) -> None:
        self._quotas[quota.tenant_id] = quota
        self._usage.setdefault(quota.tenant_id, {
            "agents": 0,
            "concurrent": 0,
            "storage_gb": 0.0,
        })

    def record_usage(self, tenant_id: str, resource: str, amount: float) -> None:
        usage = self._usage.get(tenant_id, {})
        usage[resource] = usage.get(resource, 0) + amount
        self._usage[tenant_id] = usage

    def check_quota(self, tenant_id: str, resource: str) -> bool:
        quota = self._quotas.get(tenant_id)
        if quota is None:
            return True
        usage = self._usage.get(tenant_id, {})
        limits = {
            "agents": quota.max_agents,
            "concurrent": quota.max_concurrent_executions,
            "storage_gb": quota.max_storage_gb,
        }
        limit = limits.get(resource, float("inf"))
        return usage.get(resource, 0) < limit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cost_tracker() -> CostTracker:
    return CostTracker()


@pytest.fixture
def budget_manager(cost_tracker: CostTracker) -> BudgetManager:
    return BudgetManager(cost_tracker)


@pytest.fixture
def quota_manager() -> ResourceQuotaManager:
    return ResourceQuotaManager()


# ============================================================================
# Tests
# ============================================================================


class TestCostTracker:
    """Tests for cost tracking operations."""

    def test_cost_tracker_record_cost(
        self, cost_tracker: CostTracker
    ) -> None:
        """Recording a cost creates a retrievable record."""
        record = CostRecord(
            agent_key="agent-a",
            tenant_id="tenant-1",
            cost_usd=0.05,
            resource_type="compute",
        )
        cost_tracker.record(record)
        assert cost_tracker.total_cost_by_agent("agent-a") == pytest.approx(0.05)

    def test_cost_tracker_query_by_agent(
        self, cost_tracker: CostTracker
    ) -> None:
        """Querying by agent returns only matching records."""
        cost_tracker.record(CostRecord("agent-a", "t1", 0.10, "compute"))
        cost_tracker.record(CostRecord("agent-b", "t1", 0.20, "compute"))
        cost_tracker.record(CostRecord("agent-a", "t1", 0.15, "storage"))

        results = cost_tracker.query_by_agent("agent-a")
        assert len(results) == 2
        assert cost_tracker.total_cost_by_agent("agent-a") == pytest.approx(0.25)

    def test_cost_tracker_query_by_tenant(
        self, cost_tracker: CostTracker
    ) -> None:
        """Querying by tenant returns all tenant records."""
        cost_tracker.record(CostRecord("a", "tenant-1", 0.10, "compute"))
        cost_tracker.record(CostRecord("b", "tenant-1", 0.20, "compute"))
        cost_tracker.record(CostRecord("c", "tenant-2", 0.30, "compute"))

        results = cost_tracker.query_by_tenant("tenant-1")
        assert len(results) == 2
        assert cost_tracker.total_cost_by_tenant("tenant-1") == pytest.approx(0.30)

    def test_cost_tracker_empty_returns_zero(
        self, cost_tracker: CostTracker
    ) -> None:
        """Total cost for nonexistent agent is zero."""
        assert cost_tracker.total_cost_by_agent("ghost") == 0.0


class TestBudgetManager:
    """Tests for budget enforcement."""

    def test_budget_manager_allow_within_budget(
        self, budget_manager: BudgetManager, cost_tracker: CostTracker
    ) -> None:
        """Execution is allowed when spending is within budget."""
        budget_manager.set_budget(BudgetConfig("tenant-1", monthly_limit_usd=100.0))
        cost_tracker.record(CostRecord("a", "tenant-1", 10.0, "compute"))

        assert budget_manager.allow_execution("tenant-1") is True
        assert budget_manager.check_budget("tenant-1") == BudgetStatus.WITHIN

    def test_budget_manager_warn_at_threshold(
        self, budget_manager: BudgetManager, cost_tracker: CostTracker
    ) -> None:
        """Warning is triggered at 80% of budget."""
        budget_manager.set_budget(
            BudgetConfig("tenant-1", monthly_limit_usd=100.0, warning_threshold_pct=0.80)
        )
        cost_tracker.record(CostRecord("a", "tenant-1", 85.0, "compute"))

        status = budget_manager.check_budget("tenant-1")
        assert status == BudgetStatus.WARNING
        assert budget_manager.allow_execution("tenant-1") is True

    def test_budget_manager_block_at_hard_limit(
        self, budget_manager: BudgetManager, cost_tracker: CostTracker
    ) -> None:
        """Execution is blocked when budget is exceeded."""
        budget_manager.set_budget(
            BudgetConfig("tenant-1", monthly_limit_usd=100.0)
        )
        cost_tracker.record(CostRecord("a", "tenant-1", 105.0, "compute"))

        assert budget_manager.allow_execution("tenant-1") is False
        assert budget_manager.check_budget("tenant-1") == BudgetStatus.BLOCKED

    def test_budget_manager_period_reset(
        self, budget_manager: BudgetManager
    ) -> None:
        """Resetting the period updates the period_start timestamp."""
        budget_manager.set_budget(
            BudgetConfig("tenant-1", monthly_limit_usd=100.0)
        )
        budget_manager.reset_period("tenant-1")
        # Should still work normally after reset
        assert budget_manager.check_budget("tenant-1") == BudgetStatus.WITHIN

    def test_budget_manager_no_budget_allows(
        self, budget_manager: BudgetManager
    ) -> None:
        """Tenants without a budget are always allowed."""
        assert budget_manager.allow_execution("no-budget-tenant") is True

    def test_billing_event_emission(
        self, budget_manager: BudgetManager, cost_tracker: CostTracker
    ) -> None:
        """Billing events are emitted when thresholds are crossed."""
        budget_manager.set_budget(
            BudgetConfig("tenant-1", monthly_limit_usd=100.0)
        )
        cost_tracker.record(CostRecord("a", "tenant-1", 105.0, "compute"))

        budget_manager.check_budget("tenant-1")
        assert len(budget_manager.events) >= 1
        assert budget_manager.events[-1]["event_type"] == "budget_blocked"


class TestResourceQuotaManager:
    """Tests for resource quota enforcement."""

    def test_resource_quota_allow(
        self, quota_manager: ResourceQuotaManager
    ) -> None:
        """Resource usage within quota is allowed."""
        quota_manager.set_quota(ResourceQuota("tenant-1", max_agents=10))
        quota_manager.record_usage("tenant-1", "agents", 5)
        assert quota_manager.check_quota("tenant-1", "agents") is True

    def test_resource_quota_exceeded(
        self, quota_manager: ResourceQuotaManager
    ) -> None:
        """Resource usage exceeding quota is blocked."""
        quota_manager.set_quota(ResourceQuota("tenant-1", max_agents=5))
        quota_manager.record_usage("tenant-1", "agents", 5)
        assert quota_manager.check_quota("tenant-1", "agents") is False

    def test_resource_quota_no_quota_allows(
        self, quota_manager: ResourceQuotaManager
    ) -> None:
        """Tenants without a quota are always allowed."""
        assert quota_manager.check_quota("unknown", "agents") is True
