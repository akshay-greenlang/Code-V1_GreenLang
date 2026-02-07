"""
Budget Manager - Agent Factory Metering (INFRA-010)

Manages per-agent and per-tenant budgets with configurable periods
(daily, weekly, monthly). Checks budgets before execution and emits
alerts when spending approaches or exceeds thresholds. Uses Redis
for fast spend lookups and PostgreSQL for persistent budget definitions.

Classes:
    - BudgetPeriod: Budget period enumeration.
    - Budget: Budget definition dataclass.
    - BudgetCheckResult: Outcome of a budget check (allow/warn/block).
    - BudgetManager: Core budget management service.

Example:
    >>> manager = BudgetManager(redis_client, db_pool)
    >>> budget = Budget(agent_key="intake", tenant_id="acme", period=BudgetPeriod.DAILY, amount_usd=10.0)
    >>> result = await manager.check_budget("intake", "acme")
    >>> if result.action == "block":
    ...     raise BudgetExceededError(result.message)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget Period
# ---------------------------------------------------------------------------


class BudgetPeriod(str, Enum):
    """Budget period enumeration."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# ---------------------------------------------------------------------------
# Budget Definition
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    """Budget definition for an agent or tenant.

    Attributes:
        budget_id: Unique identifier.
        agent_key: Agent this budget applies to (empty for tenant-level).
        tenant_id: Tenant this budget applies to.
        period: Budget period (daily, weekly, monthly).
        amount_usd: Maximum allowed spend for the period.
        spent_usd: Current spend within the period.
        alert_threshold_pct: Percentage at which a warning is emitted (0.0-1.0).
        is_hard_limit: If True, execution is blocked when budget is exceeded.
        period_start: Start of the current budget period (UTC).
        period_end: End of the current budget period (UTC).
    """

    budget_id: str = ""
    agent_key: str = ""
    tenant_id: str = ""
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    amount_usd: float = 0.0
    spent_usd: float = 0.0
    alert_threshold_pct: float = 0.8
    is_hard_limit: bool = False
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Budget Check Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetCheckResult:
    """Outcome of a budget check.

    Attributes:
        action: One of 'allow', 'warn', 'block'.
        budget: The budget that was checked.
        spent_usd: Current spend in the period.
        remaining_usd: Remaining budget.
        utilisation_pct: Percentage of budget used (0.0-1.0).
        message: Human-readable description.
    """

    action: str
    budget: Budget
    spent_usd: float
    remaining_usd: float
    utilisation_pct: float
    message: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised when a hard budget limit is exceeded.

    Attributes:
        agent_key: The agent whose budget was exceeded.
        tenant_id: The tenant.
        budget_amount: The budget limit.
        spent_amount: The current spend.
    """

    def __init__(
        self,
        agent_key: str,
        tenant_id: str,
        budget_amount: float,
        spent_amount: float,
    ) -> None:
        self.agent_key = agent_key
        self.tenant_id = tenant_id
        self.budget_amount = budget_amount
        self.spent_amount = spent_amount
        super().__init__(
            f"Budget exceeded for agent '{agent_key}' tenant '{tenant_id}': "
            f"${spent_amount:.4f}/${budget_amount:.4f}"
        )


# ---------------------------------------------------------------------------
# Budget Manager
# ---------------------------------------------------------------------------


class BudgetManager:
    """Per-agent and per-tenant budget management.

    Uses Redis for fast spend lookups with TTLs matching the budget period.
    Budget definitions are stored in PostgreSQL. Alerts are emitted via
    configurable callbacks.

    Attributes:
        _redis: Async Redis client.
        _pool: Async PostgreSQL connection pool.
    """

    _KEY_PREFIX = "gl:budget:"

    def __init__(
        self,
        redis_client: Any,
        db_pool: Any,
        on_alert: Optional[Callable[[BudgetCheckResult], Any]] = None,
    ) -> None:
        """Initialize the budget manager.

        Args:
            redis_client: Async Redis client (redis.asyncio).
            db_pool: Async PostgreSQL connection pool.
            on_alert: Optional callback for budget alerts.
        """
        self._redis = redis_client
        self._pool = db_pool
        self._on_alert = on_alert
        self._budgets: Dict[str, Budget] = {}
        logger.info("BudgetManager initialised")

    # ------------------------------------------------------------------
    # Budget CRUD
    # ------------------------------------------------------------------

    async def set_budget(self, budget: Budget) -> Budget:
        """Create or update a budget definition.

        Args:
            budget: Budget definition.

        Returns:
            The saved budget.
        """
        key = self._budget_key(budget.agent_key, budget.tenant_id)
        if not budget.budget_id:
            budget.budget_id = key

        # Calculate period boundaries
        now = datetime.now(timezone.utc)
        budget.period_start, budget.period_end = self._period_boundaries(
            budget.period, now,
        )

        self._budgets[key] = budget
        logger.info(
            "BudgetManager: set budget %s -> $%.2f/%s (hard=%s)",
            key, budget.amount_usd, budget.period.value, budget.is_hard_limit,
        )
        return budget

    async def get_budget(
        self,
        agent_key: str,
        tenant_id: str,
    ) -> Optional[Budget]:
        """Retrieve a budget definition.

        Args:
            agent_key: Agent key.
            tenant_id: Tenant ID.

        Returns:
            Budget if found, None otherwise.
        """
        key = self._budget_key(agent_key, tenant_id)
        return self._budgets.get(key)

    async def remove_budget(self, agent_key: str, tenant_id: str) -> bool:
        """Remove a budget definition.

        Args:
            agent_key: Agent key.
            tenant_id: Tenant ID.

        Returns:
            True if a budget was removed.
        """
        key = self._budget_key(agent_key, tenant_id)
        if key in self._budgets:
            del self._budgets[key]
            await self._redis.delete(f"{self._KEY_PREFIX}{key}:spend")
            return True
        return False

    # ------------------------------------------------------------------
    # Budget Checking
    # ------------------------------------------------------------------

    async def check_budget(
        self,
        agent_key: str,
        tenant_id: str,
    ) -> BudgetCheckResult:
        """Check whether execution is allowed under the current budget.

        Args:
            agent_key: Agent key.
            tenant_id: Tenant ID.

        Returns:
            BudgetCheckResult with action = allow/warn/block.
        """
        budget = await self.get_budget(agent_key, tenant_id)
        if budget is None:
            # No budget defined - always allow
            return BudgetCheckResult(
                action="allow",
                budget=Budget(agent_key=agent_key, tenant_id=tenant_id),
                spent_usd=0.0,
                remaining_usd=float("inf"),
                utilisation_pct=0.0,
                message="No budget defined, execution allowed.",
            )

        # Check if period has expired and reset if needed
        now = datetime.now(timezone.utc)
        if budget.period_end is not None and now > budget.period_end:
            await self._reset_period(budget, now)

        # Get current spend from Redis
        spent = await self._get_current_spend(agent_key, tenant_id)
        budget.spent_usd = spent

        remaining = max(0.0, budget.amount_usd - spent)
        utilisation = spent / budget.amount_usd if budget.amount_usd > 0 else 0.0

        if utilisation >= 1.0 and budget.is_hard_limit:
            result = BudgetCheckResult(
                action="block",
                budget=budget,
                spent_usd=spent,
                remaining_usd=0.0,
                utilisation_pct=utilisation,
                message=(
                    f"Hard budget limit exceeded: ${spent:.4f}/${budget.amount_usd:.2f} "
                    f"({utilisation * 100:.1f}%). Execution blocked."
                ),
            )
            await self._emit_alert(result)
            return result

        if utilisation >= budget.alert_threshold_pct:
            result = BudgetCheckResult(
                action="warn",
                budget=budget,
                spent_usd=spent,
                remaining_usd=remaining,
                utilisation_pct=utilisation,
                message=(
                    f"Budget warning: ${spent:.4f}/${budget.amount_usd:.2f} "
                    f"({utilisation * 100:.1f}%). Threshold: {budget.alert_threshold_pct * 100:.0f}%."
                ),
            )
            await self._emit_alert(result)
            return result

        return BudgetCheckResult(
            action="allow",
            budget=budget,
            spent_usd=spent,
            remaining_usd=remaining,
            utilisation_pct=utilisation,
            message=f"Within budget: ${spent:.4f}/${budget.amount_usd:.2f}.",
        )

    async def record_spend(
        self,
        agent_key: str,
        tenant_id: str,
        amount_usd: float,
    ) -> float:
        """Record spending against a budget.

        Atomically increments the spend counter in Redis.

        Args:
            agent_key: Agent key.
            tenant_id: Tenant ID.
            amount_usd: Amount spent in USD.

        Returns:
            New total spend for the current period.
        """
        redis_key = f"{self._KEY_PREFIX}{self._budget_key(agent_key, tenant_id)}:spend"

        # Use INCRBYFLOAT for atomic increment
        # Store amount as integer cents for precision
        amount_cents = int(amount_usd * 1_000_000)
        new_total_cents = await self._redis.incrby(redis_key, amount_cents)
        new_total = new_total_cents / 1_000_000

        # Set TTL based on budget period
        budget = await self.get_budget(agent_key, tenant_id)
        if budget is not None and budget.period_end is not None:
            ttl_seconds = int(
                (budget.period_end - datetime.now(timezone.utc)).total_seconds()
            )
            if ttl_seconds > 0:
                await self._redis.expire(redis_key, ttl_seconds)

        return new_total

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_current_spend(
        self, agent_key: str, tenant_id: str,
    ) -> float:
        """Get current spend from Redis."""
        redis_key = f"{self._KEY_PREFIX}{self._budget_key(agent_key, tenant_id)}:spend"
        raw = await self._redis.get(redis_key)
        if raw is None:
            return 0.0
        return int(raw) / 1_000_000

    async def _reset_period(self, budget: Budget, now: datetime) -> None:
        """Reset the budget period and clear spend counters."""
        budget.period_start, budget.period_end = self._period_boundaries(
            budget.period, now,
        )
        budget.spent_usd = 0.0
        redis_key = (
            f"{self._KEY_PREFIX}"
            f"{self._budget_key(budget.agent_key, budget.tenant_id)}:spend"
        )
        await self._redis.delete(redis_key)
        logger.info(
            "BudgetManager: period reset for %s (new period: %s to %s)",
            budget.budget_id,
            budget.period_start.isoformat() if budget.period_start else "N/A",
            budget.period_end.isoformat() if budget.period_end else "N/A",
        )

    async def _emit_alert(self, result: BudgetCheckResult) -> None:
        """Emit an alert via the registered callback."""
        if self._on_alert is None:
            return
        try:
            outcome = self._on_alert(result)
            if asyncio.iscoroutine(outcome):
                await outcome
        except Exception as exc:
            logger.error("BudgetManager: alert callback failed: %s", exc)

    @staticmethod
    def _budget_key(agent_key: str, tenant_id: str) -> str:
        """Generate a canonical budget key."""
        return f"{tenant_id}:{agent_key}" if tenant_id else agent_key

    @staticmethod
    def _period_boundaries(
        period: BudgetPeriod, now: datetime,
    ) -> tuple[datetime, datetime]:
        """Calculate the start and end of the current budget period.

        Args:
            period: Budget period type.
            now: Current time (UTC).

        Returns:
            Tuple of (period_start, period_end) as UTC datetimes.
        """
        if period == BudgetPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            # Week starts on Monday
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end = start + timedelta(weeks=1)
        else:  # MONTHLY
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Move to first day of next month
            if now.month == 12:
                end = start.replace(year=now.year + 1, month=1)
            else:
                end = start.replace(month=now.month + 1)
        return start, end


__all__ = [
    "Budget",
    "BudgetCheckResult",
    "BudgetExceededError",
    "BudgetManager",
    "BudgetPeriod",
]
