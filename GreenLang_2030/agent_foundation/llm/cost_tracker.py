"""
CostTracker - LLM cost tracking and budget management.

This module provides comprehensive cost tracking for LLM usage across providers,
tenants, agents, and time periods with budget alerts and reporting.

Key Features:
- Track costs per provider, tenant, agent, and model
- Time-series cost aggregation (hourly, daily, monthly)
- Budget limits with configurable alerts (80%, 90%, 100%)
- Cost forecasting and trend analysis
- Export to CSV/JSON for external analysis
- Thread-safe implementation
- Real-time cost monitoring
- Multi-dimensional cost breakdowns

Cost Dimensions:
- Provider: anthropic, openai, etc.
- Tenant: customer/organization ID
- Agent: specific agent instance
- Model: claude-3-opus, gpt-4, etc.
- Time: hourly, daily, monthly aggregations

Example:
    >>> tracker = CostTracker()
    >>> tracker.set_budget("tenant-123", 1000.0, alert_thresholds=[0.8, 0.9])
    >>>
    >>> # Track usage
    >>> tracker.track_usage(
    ...     provider="anthropic",
    ...     tenant_id="tenant-123",
    ...     agent_id="esg-agent",
    ...     model_id="claude-3-opus",
    ...     usage=token_usage
    ... )
    >>>
    >>> # Check budget
    >>> status = tracker.check_budget("tenant-123")
    >>> if status.alert_triggered:
    ...     print(f"Warning: {status.percentage_used:.1f}% of budget used")
    >>>
    >>> # Export report
    >>> tracker.export_csv("costs_2024.csv", start_date="2024-01-01")
"""

import csv
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from .providers.base_provider import TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """Single cost tracking record."""

    timestamp: datetime
    provider: str
    tenant_id: str
    agent_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    cached_tokens: int = 0
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostRecord":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class BudgetConfig:
    """Budget configuration for a tenant."""

    tenant_id: str
    monthly_limit_usd: float
    alert_thresholds: List[float] = field(
        default_factory=lambda: [0.8, 0.9, 1.0]
    )  # 80%, 90%, 100%
    alerts_triggered: List[float] = field(default_factory=list)
    current_month_cost: float = 0.0
    last_reset: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BudgetStatus:
    """Budget status for a tenant."""

    tenant_id: str
    monthly_limit_usd: float
    current_cost_usd: float
    remaining_usd: float
    percentage_used: float
    alert_triggered: bool
    alert_level: Optional[float]
    days_remaining_in_month: int
    projected_monthly_cost: float


@dataclass
class CostSummary:
    """Cost summary for reporting."""

    total_cost_usd: float
    total_tokens: int
    total_requests: int
    breakdown_by_provider: Dict[str, float]
    breakdown_by_tenant: Dict[str, float]
    breakdown_by_agent: Dict[str, float]
    breakdown_by_model: Dict[str, float]
    time_period: str
    start_date: datetime
    end_date: datetime


class CostTracker:
    """
    Production-ready cost tracker for LLM usage.

    Tracks all LLM costs across multiple dimensions with budget management,
    alerting, and comprehensive reporting capabilities.

    Attributes:
        records: List of all cost records
        budgets: Budget configurations by tenant
        alert_callbacks: Callbacks to invoke on budget alerts
    """

    def __init__(self, auto_reset_budgets: bool = True):
        """
        Initialize cost tracker.

        Args:
            auto_reset_budgets: Automatically reset monthly budgets (default: True)
        """
        self.auto_reset_budgets = auto_reset_budgets

        # Storage
        self._records: List[CostRecord] = []
        self._budgets: Dict[str, BudgetConfig] = {}
        self._lock = Lock()

        # Alert callbacks
        self._alert_callbacks: List[callable] = []

        # Aggregated metrics (cache for performance)
        self._aggregations: Dict[str, Any] = defaultdict(lambda: defaultdict(float))

        self._logger = logging.getLogger(f"{__name__}.tracker")
        self._logger.info("Initialized CostTracker")

    def track_usage(
        self,
        provider: str,
        tenant_id: str,
        agent_id: str,
        model_id: str,
        usage: TokenUsage,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostRecord:
        """
        Track LLM usage and costs.

        Args:
            provider: Provider name (anthropic, openai, etc.)
            tenant_id: Tenant/customer ID
            agent_id: Agent instance ID
            model_id: Model identifier
            usage: Token usage and costs
            request_id: Optional request ID for tracing
            metadata: Additional metadata

        Returns:
            Created cost record
        """
        record = CostRecord(
            timestamp=datetime.utcnow(),
            provider=provider,
            tenant_id=tenant_id,
            agent_id=agent_id,
            model_id=model_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            input_cost_usd=usage.input_cost_usd,
            output_cost_usd=usage.output_cost_usd,
            total_cost_usd=usage.total_cost_usd,
            cached_tokens=usage.cached_tokens,
            request_id=request_id,
            metadata=metadata or {},
        )

        with self._lock:
            # Add record
            self._records.append(record)

            # Update aggregations
            self._aggregations["by_provider"][provider] += usage.total_cost_usd
            self._aggregations["by_tenant"][tenant_id] += usage.total_cost_usd
            self._aggregations["by_agent"][agent_id] += usage.total_cost_usd
            self._aggregations["by_model"][model_id] += usage.total_cost_usd

            # Update budget tracking
            if tenant_id in self._budgets:
                budget = self._budgets[tenant_id]

                # Auto-reset monthly budgets
                if self.auto_reset_budgets:
                    self._check_and_reset_budget(budget)

                budget.current_month_cost += usage.total_cost_usd

                # Check budget alerts
                self._check_budget_alerts(budget)

        self._logger.debug(
            f"Tracked usage: provider={provider}, tenant={tenant_id}, "
            f"cost=${usage.total_cost_usd:.4f}"
        )

        return record

    def set_budget(
        self,
        tenant_id: str,
        monthly_limit_usd: float,
        alert_thresholds: Optional[List[float]] = None,
    ) -> None:
        """
        Set or update budget for a tenant.

        Args:
            tenant_id: Tenant ID
            monthly_limit_usd: Monthly budget limit in USD
            alert_thresholds: Alert thresholds as percentages (e.g., [0.8, 0.9, 1.0])
        """
        if alert_thresholds is None:
            alert_thresholds = [0.8, 0.9, 1.0]

        # Validate thresholds
        alert_thresholds = sorted([t for t in alert_thresholds if 0 < t <= 1.0])

        with self._lock:
            if tenant_id in self._budgets:
                # Update existing budget
                budget = self._budgets[tenant_id]
                budget.monthly_limit_usd = monthly_limit_usd
                budget.alert_thresholds = alert_thresholds
            else:
                # Create new budget
                budget = BudgetConfig(
                    tenant_id=tenant_id,
                    monthly_limit_usd=monthly_limit_usd,
                    alert_thresholds=alert_thresholds,
                )
                self._budgets[tenant_id] = budget

        self._logger.info(
            f"Set budget for {tenant_id}: ${monthly_limit_usd:.2f}/month, "
            f"thresholds={alert_thresholds}"
        )

    def check_budget(self, tenant_id: str) -> Optional[BudgetStatus]:
        """
        Check budget status for a tenant.

        Args:
            tenant_id: Tenant ID

        Returns:
            Budget status or None if no budget configured
        """
        with self._lock:
            if tenant_id not in self._budgets:
                return None

            budget = self._budgets[tenant_id]

            # Auto-reset if needed
            if self.auto_reset_budgets:
                self._check_and_reset_budget(budget)

            # Calculate status
            percentage_used = (
                (budget.current_month_cost / budget.monthly_limit_usd) * 100
                if budget.monthly_limit_usd > 0
                else 0
            )

            remaining = budget.monthly_limit_usd - budget.current_month_cost

            # Check if any alert threshold crossed
            alert_triggered = False
            alert_level = None
            for threshold in budget.alert_thresholds:
                if percentage_used >= threshold * 100:
                    alert_triggered = True
                    alert_level = threshold

            # Calculate days remaining in month
            now = datetime.utcnow()
            if now.month == 12:
                next_month = datetime(now.year + 1, 1, 1)
            else:
                next_month = datetime(now.year, now.month + 1, 1)
            days_remaining = (next_month - now).days

            # Project monthly cost
            days_in_month = (next_month - datetime(now.year, now.month, 1)).days
            days_elapsed = days_in_month - days_remaining
            if days_elapsed > 0:
                projected = (budget.current_month_cost / days_elapsed) * days_in_month
            else:
                projected = budget.current_month_cost

            return BudgetStatus(
                tenant_id=tenant_id,
                monthly_limit_usd=budget.monthly_limit_usd,
                current_cost_usd=budget.current_month_cost,
                remaining_usd=remaining,
                percentage_used=percentage_used,
                alert_triggered=alert_triggered,
                alert_level=alert_level,
                days_remaining_in_month=days_remaining,
                projected_monthly_cost=projected,
            )

    def _check_and_reset_budget(self, budget: BudgetConfig) -> None:
        """Check if budget should be reset (new month)."""
        now = datetime.utcnow()
        last_reset = budget.last_reset

        # Check if we're in a new month
        if now.year > last_reset.year or now.month > last_reset.month:
            self._logger.info(
                f"Resetting budget for {budget.tenant_id} "
                f"(previous cost: ${budget.current_month_cost:.2f})"
            )
            budget.current_month_cost = 0.0
            budget.alerts_triggered = []
            budget.last_reset = now

    def _check_budget_alerts(self, budget: BudgetConfig) -> None:
        """Check and trigger budget alerts."""
        if budget.monthly_limit_usd <= 0:
            return

        percentage = budget.current_month_cost / budget.monthly_limit_usd

        for threshold in budget.alert_thresholds:
            # Check if threshold crossed and not already alerted
            if percentage >= threshold and threshold not in budget.alerts_triggered:
                budget.alerts_triggered.append(threshold)

                alert_msg = (
                    f"Budget alert for {budget.tenant_id}: "
                    f"{percentage * 100:.1f}% used "
                    f"(${budget.current_month_cost:.2f}/${budget.monthly_limit_usd:.2f})"
                )

                self._logger.warning(alert_msg)

                # Trigger callbacks
                for callback in self._alert_callbacks:
                    try:
                        callback(budget.tenant_id, threshold, budget.current_month_cost)
                    except Exception as e:
                        self._logger.error(f"Alert callback failed: {str(e)}")

    def register_alert_callback(self, callback: callable) -> None:
        """
        Register callback for budget alerts.

        Args:
            callback: Function with signature (tenant_id, threshold, current_cost)
        """
        self._alert_callbacks.append(callback)
        self._logger.info(f"Registered alert callback: {callback.__name__}")

    def get_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
    ) -> CostSummary:
        """
        Get cost summary for a time period.

        Args:
            start_date: Start date (default: beginning of records)
            end_date: End date (default: now)
            tenant_id: Filter by tenant ID (optional)

        Returns:
            Cost summary
        """
        with self._lock:
            # Filter records
            filtered = self._records

            if start_date:
                filtered = [r for r in filtered if r.timestamp >= start_date]
            if end_date:
                filtered = [r for r in filtered if r.timestamp <= end_date]
            if tenant_id:
                filtered = [r for r in filtered if r.tenant_id == tenant_id]

            if not filtered:
                return CostSummary(
                    total_cost_usd=0.0,
                    total_tokens=0,
                    total_requests=0,
                    breakdown_by_provider={},
                    breakdown_by_tenant={},
                    breakdown_by_agent={},
                    breakdown_by_model={},
                    time_period="custom",
                    start_date=start_date or datetime.utcnow(),
                    end_date=end_date or datetime.utcnow(),
                )

            # Calculate aggregations
            total_cost = sum(r.total_cost_usd for r in filtered)
            total_tokens = sum(r.total_tokens for r in filtered)

            by_provider = defaultdict(float)
            by_tenant = defaultdict(float)
            by_agent = defaultdict(float)
            by_model = defaultdict(float)

            for record in filtered:
                by_provider[record.provider] += record.total_cost_usd
                by_tenant[record.tenant_id] += record.total_cost_usd
                by_agent[record.agent_id] += record.total_cost_usd
                by_model[record.model_id] += record.total_cost_usd

            # Determine time period
            actual_start = min(r.timestamp for r in filtered)
            actual_end = max(r.timestamp for r in filtered)

            return CostSummary(
                total_cost_usd=total_cost,
                total_tokens=total_tokens,
                total_requests=len(filtered),
                breakdown_by_provider=dict(by_provider),
                breakdown_by_tenant=dict(by_tenant),
                breakdown_by_agent=dict(by_agent),
                breakdown_by_model=dict(by_model),
                time_period="custom",
                start_date=actual_start,
                end_date=actual_end,
            )

    def get_time_series(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "daily",
        tenant_id: Optional[str] = None,
    ) -> List[Tuple[datetime, float]]:
        """
        Get time series cost data.

        Args:
            start_date: Start date
            end_date: End date
            granularity: Time granularity (hourly, daily, monthly)
            tenant_id: Filter by tenant ID (optional)

        Returns:
            List of (timestamp, cost) tuples
        """
        with self._lock:
            # Filter records
            filtered = [
                r
                for r in self._records
                if start_date <= r.timestamp <= end_date
                and (not tenant_id or r.tenant_id == tenant_id)
            ]

            # Group by time bucket
            buckets = defaultdict(float)

            for record in filtered:
                if granularity == "hourly":
                    bucket = record.timestamp.replace(minute=0, second=0, microsecond=0)
                elif granularity == "daily":
                    bucket = record.timestamp.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                elif granularity == "monthly":
                    bucket = record.timestamp.replace(
                        day=1, hour=0, minute=0, second=0, microsecond=0
                    )
                else:
                    raise ValueError(f"Invalid granularity: {granularity}")

                buckets[bucket] += record.total_cost_usd

            # Sort by timestamp
            return sorted(buckets.items())

    def export_csv(
        self,
        file_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Export cost records to CSV.

        Args:
            file_path: Output CSV file path
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
        """
        with self._lock:
            # Filter records
            filtered = self._records

            if start_date:
                filtered = [r for r in filtered if r.timestamp >= start_date]
            if end_date:
                filtered = [r for r in filtered if r.timestamp <= end_date]

            # Write CSV
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", newline="", encoding="utf-8") as f:
                if not filtered:
                    self._logger.warning("No records to export")
                    return

                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "provider",
                        "tenant_id",
                        "agent_id",
                        "model_id",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "input_cost_usd",
                        "output_cost_usd",
                        "total_cost_usd",
                        "cached_tokens",
                        "request_id",
                    ],
                )
                writer.writeheader()

                for record in filtered:
                    writer.writerow(
                        {
                            "timestamp": record.timestamp.isoformat(),
                            "provider": record.provider,
                            "tenant_id": record.tenant_id,
                            "agent_id": record.agent_id,
                            "model_id": record.model_id,
                            "input_tokens": record.input_tokens,
                            "output_tokens": record.output_tokens,
                            "total_tokens": record.total_tokens,
                            "input_cost_usd": f"{record.input_cost_usd:.6f}",
                            "output_cost_usd": f"{record.output_cost_usd:.6f}",
                            "total_cost_usd": f"{record.total_cost_usd:.6f}",
                            "cached_tokens": record.cached_tokens,
                            "request_id": record.request_id or "",
                        }
                    )

            self._logger.info(f"Exported {len(filtered)} records to {file_path}")

    def export_json(
        self,
        file_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_summary: bool = True,
    ) -> None:
        """
        Export cost records to JSON.

        Args:
            file_path: Output JSON file path
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            include_summary: Include summary statistics (default: True)
        """
        with self._lock:
            # Filter records
            filtered = self._records

            if start_date:
                filtered = [r for r in filtered if r.timestamp >= start_date]
            if end_date:
                filtered = [r for r in filtered if r.timestamp <= end_date]

            # Build export data
            export_data = {
                "records": [r.to_dict() for r in filtered],
                "record_count": len(filtered),
                "export_timestamp": datetime.utcnow().isoformat(),
            }

            if include_summary:
                summary = self.get_summary(start_date, end_date)
                export_data["summary"] = {
                    "total_cost_usd": summary.total_cost_usd,
                    "total_tokens": summary.total_tokens,
                    "total_requests": summary.total_requests,
                    "breakdown_by_provider": summary.breakdown_by_provider,
                    "breakdown_by_tenant": summary.breakdown_by_tenant,
                    "breakdown_by_agent": summary.breakdown_by_agent,
                    "breakdown_by_model": summary.breakdown_by_model,
                    "start_date": summary.start_date.isoformat(),
                    "end_date": summary.end_date.isoformat(),
                }

            # Write JSON
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)

            self._logger.info(f"Exported {len(filtered)} records to {file_path}")

    def clear_records(self, before_date: Optional[datetime] = None) -> int:
        """
        Clear cost records.

        Args:
            before_date: Clear records before this date (default: clear all)

        Returns:
            Number of records cleared
        """
        with self._lock:
            if before_date:
                kept = [r for r in self._records if r.timestamp >= before_date]
                cleared = len(self._records) - len(kept)
                self._records = kept
            else:
                cleared = len(self._records)
                self._records = []
                self._aggregations.clear()

            self._logger.info(f"Cleared {cleared} cost records")
            return cleared


# Example usage
if __name__ == "__main__":
    import time

    # Initialize tracker
    tracker = CostTracker()

    # Set budget with alerts at 80%, 90%, 100%
    tracker.set_budget("tenant-123", monthly_limit_usd=1000.0, alert_thresholds=[0.8, 0.9, 1.0])

    # Register alert callback
    def on_budget_alert(tenant_id: str, threshold: float, current_cost: float):
        print(f"ALERT: Tenant {tenant_id} reached {threshold * 100:.0f}% of budget (${current_cost:.2f})")

    tracker.register_alert_callback(on_budget_alert)

    # Simulate usage
    print("=== Simulating LLM Usage ===")
    for i in range(5):
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            input_cost_usd=0.015,  # $0.015/1K * 1000 tokens
            output_cost_usd=0.0375,  # $0.075/1K * 500 tokens
            total_cost_usd=0.0525,
        )

        tracker.track_usage(
            provider="anthropic",
            tenant_id="tenant-123",
            agent_id="esg-analyzer",
            model_id="claude-3-opus-20240229",
            usage=usage,
            request_id=f"req-{i+1}",
        )

        print(f"Request {i+1}: ${usage.total_cost_usd:.4f}")
        time.sleep(0.1)

    # Check budget
    print("\n=== Budget Status ===")
    status = tracker.check_budget("tenant-123")
    if status:
        print(f"Monthly limit: ${status.monthly_limit_usd:.2f}")
        print(f"Current cost: ${status.current_cost_usd:.2f}")
        print(f"Remaining: ${status.remaining_usd:.2f}")
        print(f"Usage: {status.percentage_used:.1f}%")
        print(f"Projected monthly: ${status.projected_monthly_cost:.2f}")

    # Get summary
    print("\n=== Cost Summary ===")
    summary = tracker.get_summary()
    print(f"Total cost: ${summary.total_cost_usd:.4f}")
    print(f"Total tokens: {summary.total_tokens:,}")
    print(f"Total requests: {summary.total_requests}")
    print(f"\nBy Provider:")
    for provider, cost in summary.breakdown_by_provider.items():
        print(f"  {provider}: ${cost:.4f}")
    print(f"\nBy Agent:")
    for agent, cost in summary.breakdown_by_agent.items():
        print(f"  {agent}: ${cost:.4f}")

    # Export to CSV
    print("\n=== Exporting Data ===")
    tracker.export_csv("cost_report.csv")
    tracker.export_json("cost_report.json", include_summary=True)
    print("Exported to cost_report.csv and cost_report.json")
