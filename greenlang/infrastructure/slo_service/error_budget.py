# -*- coding: utf-8 -*-
"""
Error Budget Calculator - OBS-005: SLO/SLI Definitions & Error Budget Management

Computes error budget consumption, remaining budget, budget status,
forecasts exhaustion dates, and records budget snapshots.  Supports
Redis caching for real-time budget lookups.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from greenlang.infrastructure.slo_service.models import (
    BudgetStatus,
    ErrorBudget,
    SLO,
)


# ---------------------------------------------------------------------------
# Budget calculation
# ---------------------------------------------------------------------------


def calculate_error_budget(
    slo: SLO,
    current_sli: float,
    budget_thresholds: Optional[Dict[str, float]] = None,
) -> ErrorBudget:
    """Calculate the error budget for an SLO given its current SLI.

    Args:
        slo: SLO definition.
        current_sli: Current SLI value as a ratio (0.0 to 1.0).
        budget_thresholds: Custom thresholds for budget status.

    Returns:
        ErrorBudget instance.
    """
    thresholds = budget_thresholds or {
        "warning": 20.0,
        "critical": 50.0,
        "exhausted": 100.0,
    }

    total_minutes = slo.window_minutes
    error_budget_fraction = slo.error_budget_fraction
    total_budget_minutes = total_minutes * error_budget_fraction

    # SLI is a ratio 0..1, target is percentage e.g. 99.9
    target_ratio = slo.target / 100.0
    actual_error_rate = 1.0 - current_sli
    allowed_error_rate = 1.0 - target_ratio

    if allowed_error_rate <= 0:
        consumed_percent = 100.0 if actual_error_rate > 0 else 0.0
    else:
        consumed_percent = min(
            (actual_error_rate / allowed_error_rate) * 100.0, 100.0
        )

    consumed_minutes = total_budget_minutes * (consumed_percent / 100.0)
    remaining_minutes = max(total_budget_minutes - consumed_minutes, 0.0)
    remaining_percent = max(100.0 - consumed_percent, 0.0)

    status = _determine_budget_status(consumed_percent, thresholds)

    return ErrorBudget(
        slo_id=slo.slo_id,
        total_minutes=total_budget_minutes,
        consumed_minutes=consumed_minutes,
        remaining_minutes=remaining_minutes,
        remaining_percent=remaining_percent,
        consumed_percent=consumed_percent,
        status=status,
        sli_value=current_sli * 100.0,
        window=slo.window.value,
    )


def _determine_budget_status(
    consumed_percent: float,
    thresholds: Dict[str, float],
) -> BudgetStatus:
    """Determine budget health status from consumed percentage.

    Args:
        consumed_percent: Percentage of budget consumed (0-100).
        thresholds: Threshold mapping.

    Returns:
        BudgetStatus enum value.
    """
    if consumed_percent >= thresholds.get("exhausted", 100.0):
        return BudgetStatus.EXHAUSTED
    elif consumed_percent >= thresholds.get("critical", 50.0):
        return BudgetStatus.CRITICAL
    elif consumed_percent >= thresholds.get("warning", 20.0):
        return BudgetStatus.WARNING
    else:
        return BudgetStatus.HEALTHY


def budget_consumption_rate(
    consumed_percent: float,
    elapsed_minutes: float,
) -> float:
    """Calculate the rate of budget consumption per hour.

    Args:
        consumed_percent: Percentage of budget consumed.
        elapsed_minutes: Minutes elapsed since window start.

    Returns:
        Consumption rate in percent per hour.
    """
    if elapsed_minutes <= 0:
        return 0.0
    return (consumed_percent / elapsed_minutes) * 60.0


def forecast_exhaustion(
    consumed_percent: float,
    consumption_rate_per_hour: float,
) -> Optional[datetime]:
    """Forecast when the error budget will be exhausted.

    Args:
        consumed_percent: Current consumed percentage.
        consumption_rate_per_hour: Rate of consumption in % per hour.

    Returns:
        Estimated exhaustion datetime, or None if no exhaustion projected.
    """
    if consumption_rate_per_hour <= 0:
        return None

    remaining = 100.0 - consumed_percent
    if remaining <= 0:
        return datetime.now(timezone.utc)

    hours_remaining = remaining / consumption_rate_per_hour
    return datetime.now(timezone.utc) + timedelta(hours=hours_remaining)


def check_budget_policy(
    budget: ErrorBudget,
    policy: str,
) -> Dict[str, Any]:
    """Check budget against the configured exhaustion policy.

    Args:
        budget: Error budget state.
        policy: Policy action (``freeze_deployments``, ``alert_only``, ``none``).

    Returns:
        Dictionary with policy evaluation result.
    """
    result: Dict[str, Any] = {
        "budget_status": budget.status.value,
        "consumed_percent": budget.consumed_percent,
        "policy": policy,
        "action_required": False,
        "action": "none",
    }

    if budget.status == BudgetStatus.EXHAUSTED:
        if policy == "freeze_deployments":
            result["action_required"] = True
            result["action"] = "freeze_deployments"
        elif policy == "alert_only":
            result["action_required"] = True
            result["action"] = "alert_only"
        else:
            result["action_required"] = False
            result["action"] = "none"
    elif budget.status == BudgetStatus.CRITICAL:
        if policy in ("freeze_deployments", "alert_only"):
            result["action_required"] = True
            result["action"] = "alert_only"

    return result


# ---------------------------------------------------------------------------
# Redis caching helpers
# ---------------------------------------------------------------------------


class BudgetCache:
    """Redis-backed cache for error budget snapshots.

    Attributes:
        _redis: Redis client instance.
        _prefix: Key prefix for budget entries.
        _ttl: Cache TTL in seconds.
    """

    def __init__(
        self,
        redis_client: Any,
        prefix: str = "gl:slo:budget:",
        ttl_seconds: int = 60,
    ) -> None:
        """Initialize the budget cache.

        Args:
            redis_client: Redis client.
            prefix: Key prefix.
            ttl_seconds: Cache TTL.
        """
        self._redis = redis_client
        self._prefix = prefix
        self._ttl = ttl_seconds

    def _key(self, slo_id: str) -> str:
        """Build the Redis key for an SLO budget."""
        return f"{self._prefix}{slo_id}"

    def get(self, slo_id: str) -> Optional[ErrorBudget]:
        """Get a cached budget snapshot.

        Args:
            slo_id: SLO identifier.

        Returns:
            ErrorBudget or None on cache miss.
        """
        try:
            data = self._redis.get(self._key(slo_id))
            if data is None:
                return None
            return ErrorBudget.from_dict(json.loads(data))
        except Exception as exc:
            logger.warning("Redis cache get error: %s", exc)
            return None

    def set(self, budget: ErrorBudget) -> bool:
        """Cache a budget snapshot.

        Args:
            budget: ErrorBudget to cache.

        Returns:
            True on success, False on error.
        """
        try:
            key = self._key(budget.slo_id)
            data = json.dumps(budget.to_dict(), default=str)
            self._redis.setex(key, self._ttl, data)
            return True
        except Exception as exc:
            logger.warning("Redis cache set error: %s", exc)
            return False

    def invalidate(self, slo_id: str) -> bool:
        """Invalidate a cached budget.

        Args:
            slo_id: SLO identifier.

        Returns:
            True if key was deleted.
        """
        try:
            return bool(self._redis.delete(self._key(slo_id)))
        except Exception as exc:
            logger.warning("Redis cache invalidate error: %s", exc)
            return False
