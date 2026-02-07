"""
Resource Quota Enforcer - Agent Factory Metering (INFRA-010)

Enforces per-agent execution quotas using atomic Redis counters.
Supports max concurrent executions, daily execution limits, and
hourly execution limits. Quotas reset automatically at period boundaries.

Classes:
    - ResourceQuota: Quota definition dataclass.
    - QuotaCheckResult: Outcome of a quota check.
    - QuotaExceededError: Raised when a quota is exceeded.
    - ResourceQuotaEnforcer: Core quota enforcement service.

Example:
    >>> enforcer = ResourceQuotaEnforcer(redis_client)
    >>> quota = ResourceQuota(max_daily_executions=1000)
    >>> enforcer.set_quota("intake-agent", quota)
    >>> result = await enforcer.check_and_acquire("intake-agent")
    >>> if not result.allowed:
    ...     raise QuotaExceededError(result.reason)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quota Definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceQuota:
    """Quota definition for an agent.

    Attributes:
        max_concurrent_executions: Maximum simultaneous executions.
        max_daily_executions: Maximum executions per day (UTC midnight reset).
        max_hourly_executions: Maximum executions per hour.
    """

    max_concurrent_executions: int = 50
    max_daily_executions: int = 10000
    max_hourly_executions: int = 1000


# ---------------------------------------------------------------------------
# Check Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuotaCheckResult:
    """Outcome of a quota check.

    Attributes:
        allowed: Whether the execution is allowed.
        agent_key: Agent that was checked.
        concurrent_count: Current concurrent execution count.
        daily_count: Executions so far today.
        hourly_count: Executions so far this hour.
        reason: Human-readable explanation if not allowed.
        quota: The quota that was applied.
    """

    allowed: bool
    agent_key: str
    concurrent_count: int
    daily_count: int
    hourly_count: int
    reason: str = ""
    quota: Optional[ResourceQuota] = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class QuotaExceededError(Exception):
    """Raised when an execution quota is exceeded.

    Attributes:
        agent_key: The agent whose quota was exceeded.
        quota_type: Which quota was exceeded (concurrent/daily/hourly).
        current_count: Current count for the quota.
        max_count: Maximum allowed count.
    """

    def __init__(
        self,
        agent_key: str,
        quota_type: str,
        current_count: int,
        max_count: int,
    ) -> None:
        self.agent_key = agent_key
        self.quota_type = quota_type
        self.current_count = current_count
        self.max_count = max_count
        super().__init__(
            f"Quota exceeded for '{agent_key}': {quota_type} "
            f"({current_count}/{max_count})"
        )


# ---------------------------------------------------------------------------
# Resource Quota Enforcer
# ---------------------------------------------------------------------------


class ResourceQuotaEnforcer:
    """Enforces per-agent execution quotas via atomic Redis operations.

    Uses Redis INCR/DECR for atomic counter management with TTL-based
    automatic resets at period boundaries.

    Attributes:
        _redis: Async Redis client.
    """

    _KEY_PREFIX = "gl:quota:"

    def __init__(self, redis_client: Any) -> None:
        """Initialize the quota enforcer.

        Args:
            redis_client: Async Redis client (redis.asyncio).
        """
        self._redis = redis_client
        self._quotas: Dict[str, ResourceQuota] = {}
        logger.info("ResourceQuotaEnforcer initialised")

    # ------------------------------------------------------------------
    # Quota Configuration
    # ------------------------------------------------------------------

    def set_quota(self, agent_key: str, quota: ResourceQuota) -> None:
        """Set the quota for an agent.

        Args:
            agent_key: The agent key.
            quota: Quota definition.
        """
        self._quotas[agent_key] = quota
        logger.info(
            "Quota set for '%s': concurrent=%d, daily=%d, hourly=%d",
            agent_key,
            quota.max_concurrent_executions,
            quota.max_daily_executions,
            quota.max_hourly_executions,
        )

    def get_quota(self, agent_key: str) -> Optional[ResourceQuota]:
        """Get the quota for an agent.

        Args:
            agent_key: The agent key.

        Returns:
            ResourceQuota if defined, None otherwise.
        """
        return self._quotas.get(agent_key)

    def remove_quota(self, agent_key: str) -> bool:
        """Remove the quota for an agent.

        Args:
            agent_key: The agent key.

        Returns:
            True if a quota was removed.
        """
        return self._quotas.pop(agent_key, None) is not None

    # ------------------------------------------------------------------
    # Check and Acquire
    # ------------------------------------------------------------------

    async def check_and_acquire(self, agent_key: str) -> QuotaCheckResult:
        """Check quotas and atomically acquire execution slots.

        This is the primary method called before agent execution. It
        atomically increments all counters and checks against limits.
        If any limit is exceeded, the counters are rolled back.

        Args:
            agent_key: The agent to check.

        Returns:
            QuotaCheckResult indicating whether execution is allowed.
        """
        quota = self._quotas.get(agent_key)
        if quota is None:
            return QuotaCheckResult(
                allowed=True,
                agent_key=agent_key,
                concurrent_count=0,
                daily_count=0,
                hourly_count=0,
                reason="No quota defined.",
            )

        # Atomically increment all counters
        concurrent_key = f"{self._KEY_PREFIX}{agent_key}:concurrent"
        daily_key = f"{self._KEY_PREFIX}{agent_key}:daily:{self._daily_suffix()}"
        hourly_key = f"{self._KEY_PREFIX}{agent_key}:hourly:{self._hourly_suffix()}"

        # Use pipeline for atomic multi-key operations
        pipe = self._redis.pipeline()
        pipe.incr(concurrent_key)
        pipe.incr(daily_key)
        pipe.incr(hourly_key)
        results = await pipe.execute()

        concurrent_count = int(results[0])
        daily_count = int(results[1])
        hourly_count = int(results[2])

        # Set TTLs on period counters
        await self._ensure_ttl(daily_key, self._seconds_until_next_day())
        await self._ensure_ttl(hourly_key, self._seconds_until_next_hour())

        # Check limits
        violation = self._check_limits(
            quota, concurrent_count, daily_count, hourly_count,
        )

        if violation:
            # Rollback all increments
            pipe = self._redis.pipeline()
            pipe.decr(concurrent_key)
            pipe.decr(daily_key)
            pipe.decr(hourly_key)
            await pipe.execute()

            return QuotaCheckResult(
                allowed=False,
                agent_key=agent_key,
                concurrent_count=concurrent_count - 1,
                daily_count=daily_count - 1,
                hourly_count=hourly_count - 1,
                reason=violation,
                quota=quota,
            )

        return QuotaCheckResult(
            allowed=True,
            agent_key=agent_key,
            concurrent_count=concurrent_count,
            daily_count=daily_count,
            hourly_count=hourly_count,
            quota=quota,
        )

    async def release(self, agent_key: str) -> None:
        """Release a concurrent execution slot.

        Called after agent execution completes (success or failure).

        Args:
            agent_key: The agent that completed.
        """
        concurrent_key = f"{self._KEY_PREFIX}{agent_key}:concurrent"
        new_val = await self._redis.decr(concurrent_key)
        # Prevent going negative
        if new_val < 0:
            await self._redis.set(concurrent_key, 0)

    # ------------------------------------------------------------------
    # Status Query
    # ------------------------------------------------------------------

    async def get_current_usage(self, agent_key: str) -> QuotaCheckResult:
        """Get current usage without acquiring.

        Args:
            agent_key: The agent to query.

        Returns:
            QuotaCheckResult with current counts (allowed is always True).
        """
        quota = self._quotas.get(agent_key)

        concurrent_key = f"{self._KEY_PREFIX}{agent_key}:concurrent"
        daily_key = f"{self._KEY_PREFIX}{agent_key}:daily:{self._daily_suffix()}"
        hourly_key = f"{self._KEY_PREFIX}{agent_key}:hourly:{self._hourly_suffix()}"

        pipe = self._redis.pipeline()
        pipe.get(concurrent_key)
        pipe.get(daily_key)
        pipe.get(hourly_key)
        results = await pipe.execute()

        return QuotaCheckResult(
            allowed=True,
            agent_key=agent_key,
            concurrent_count=int(results[0] or 0),
            daily_count=int(results[1] or 0),
            hourly_count=int(results[2] or 0),
            quota=quota,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _check_limits(
        quota: ResourceQuota,
        concurrent: int,
        daily: int,
        hourly: int,
    ) -> str:
        """Check if any limits are exceeded.

        Returns empty string if all limits pass, or a violation message.
        """
        if concurrent > quota.max_concurrent_executions:
            return (
                f"Concurrent execution limit exceeded: "
                f"{concurrent}/{quota.max_concurrent_executions}"
            )
        if daily > quota.max_daily_executions:
            return (
                f"Daily execution limit exceeded: "
                f"{daily}/{quota.max_daily_executions}"
            )
        if hourly > quota.max_hourly_executions:
            return (
                f"Hourly execution limit exceeded: "
                f"{hourly}/{quota.max_hourly_executions}"
            )
        return ""

    async def _ensure_ttl(self, key: str, ttl_seconds: int) -> None:
        """Set TTL on a key if it does not already have one."""
        current_ttl = await self._redis.ttl(key)
        if current_ttl < 0:  # No TTL set (-1) or key doesn't exist (-2)
            await self._redis.expire(key, ttl_seconds)

    @staticmethod
    def _daily_suffix() -> str:
        """Generate today's date suffix for daily keys."""
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    @staticmethod
    def _hourly_suffix() -> str:
        """Generate the current hour suffix for hourly keys."""
        return datetime.now(timezone.utc).strftime("%Y%m%d%H")

    @staticmethod
    def _seconds_until_next_day() -> int:
        """Calculate seconds remaining until UTC midnight."""
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        return int((tomorrow - now).total_seconds()) + 1

    @staticmethod
    def _seconds_until_next_hour() -> int:
        """Calculate seconds remaining until the next hour boundary."""
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0,
        )
        return int((next_hour - now).total_seconds()) + 1


__all__ = [
    "QuotaCheckResult",
    "QuotaExceededError",
    "ResourceQuota",
    "ResourceQuotaEnforcer",
]
