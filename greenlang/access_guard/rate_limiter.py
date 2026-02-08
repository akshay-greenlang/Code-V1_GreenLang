# -*- coding: utf-8 -*-
"""
Rate Limiter - AGENT-FOUND-006: Access & Policy Guard

Token bucket rate limiter with per-minute, per-hour, and per-day limits.
Supports role-based overrides and auto-reset on time window expiry.

Zero-Hallucination Guarantees:
    - Deterministic token bucket algorithm
    - No probabilistic rate decisions
    - Complete quota visibility

Example:
    >>> from greenlang.access_guard.rate_limiter import RateLimiter
    >>> from greenlang.access_guard.models import RateLimitConfig
    >>> limiter = RateLimiter(RateLimitConfig())
    >>> allowed, reason = limiter.check_rate_limit("tenant-1", "user-1")
    >>> print(allowed)  # True

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-006 Access & Policy Guard
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from greenlang.access_guard.models import RateLimitConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token bucket data structure
# ---------------------------------------------------------------------------


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting.

    Tracks request counts within minute, hour, and day time windows.
    Windows auto-reset when the elapsed time exceeds the window size.

    Attributes:
        tokens: Available burst tokens.
        last_update: Timestamp of last token update.
        requests_this_minute: Requests consumed in current minute window.
        requests_this_hour: Requests consumed in current hour window.
        requests_this_day: Requests consumed in current day window.
        minute_start: Start timestamp of the current minute window.
        hour_start: Start timestamp of the current hour window.
        day_start: Start timestamp of the current day window.
    """
    tokens: float
    last_update: float
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    requests_this_day: int = 0
    minute_start: float = 0.0
    hour_start: float = 0.0
    day_start: float = 0.0


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Token bucket rate limiter with per-minute, per-hour, and per-day limits.

    Thread-safe implementation for multi-tenant environments. Supports
    role-based limit overrides defined in the RateLimitConfig.

    Attributes:
        config: Rate limit configuration.

    Example:
        >>> limiter = RateLimiter(RateLimitConfig(requests_per_minute=50))
        >>> allowed, reason = limiter.check_rate_limit("t1", "u1")
        >>> quota = limiter.get_remaining_quota("t1", "u1")
    """

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize the RateLimiter.

        Args:
            config: Rate limit configuration with default limits and
                role-based overrides.
        """
        self.config = config
        self._buckets: Dict[str, RateLimitBucket] = {}
        self._lock = threading.Lock()
        logger.info(
            "RateLimiter initialized: rpm=%d, rph=%d, rpd=%d, burst=%d",
            config.requests_per_minute,
            config.requests_per_hour,
            config.requests_per_day,
            config.burst_limit,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_rate_limit(
        self,
        tenant_id: str,
        principal_id: str,
        role: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a request is within rate limits.

        Acquires a lock, refreshes time windows, checks all three
        rate limit tiers, and increments counters atomically.

        Args:
            tenant_id: Tenant identifier.
            principal_id: Principal identifier.
            role: Optional role for role-specific limits.

        Returns:
            Tuple of (allowed, denial_reason). denial_reason is None
            when allowed is True.
        """
        key = self._bucket_key(tenant_id, principal_id)
        rpm, rph, rpd = self._effective_limits(role)

        with self._lock:
            bucket = self._get_or_create_bucket(key)
            now = time.time()

            # Reset counters if time windows have expired
            if now - bucket.minute_start >= 60:
                bucket.requests_this_minute = 0
                bucket.minute_start = now

            if now - bucket.hour_start >= 3600:
                bucket.requests_this_hour = 0
                bucket.hour_start = now

            if now - bucket.day_start >= 86400:
                bucket.requests_this_day = 0
                bucket.day_start = now

            # Check limits
            if bucket.requests_this_minute >= rpm:
                reason = f"Rate limit exceeded: {rpm} requests per minute"
                logger.debug("Rate limited %s: %s", key, reason)
                return False, reason

            if bucket.requests_this_hour >= rph:
                reason = f"Rate limit exceeded: {rph} requests per hour"
                logger.debug("Rate limited %s: %s", key, reason)
                return False, reason

            if bucket.requests_this_day >= rpd:
                reason = f"Rate limit exceeded: {rpd} requests per day"
                logger.debug("Rate limited %s: %s", key, reason)
                return False, reason

            # Increment counters
            bucket.requests_this_minute += 1
            bucket.requests_this_hour += 1
            bucket.requests_this_day += 1

        return True, None

    def get_remaining_quota(
        self,
        tenant_id: str,
        principal_id: str,
        role: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get remaining quota for a principal.

        Args:
            tenant_id: Tenant identifier.
            principal_id: Principal identifier.
            role: Optional role for role-specific limits.

        Returns:
            Dictionary with remaining_per_minute, remaining_per_hour,
            and remaining_per_day.
        """
        key = self._bucket_key(tenant_id, principal_id)
        rpm, rph, rpd = self._effective_limits(role)

        with self._lock:
            bucket = self._get_or_create_bucket(key)

        return {
            "remaining_per_minute": max(0, rpm - bucket.requests_this_minute),
            "remaining_per_hour": max(0, rph - bucket.requests_this_hour),
            "remaining_per_day": max(0, rpd - bucket.requests_this_day),
        }

    def reset_limits(
        self, tenant_id: str, principal_id: str,
    ) -> None:
        """Reset rate limits for a specific principal.

        Args:
            tenant_id: Tenant identifier.
            principal_id: Principal identifier.
        """
        key = self._bucket_key(tenant_id, principal_id)
        with self._lock:
            if key in self._buckets:
                del self._buckets[key]
        logger.info("Reset rate limits for %s", key)

    def get_all_buckets(self) -> Dict[str, Dict[str, int]]:
        """Get a snapshot of all active rate limit buckets.

        Returns:
            Dictionary mapping bucket keys to their current counters.
        """
        with self._lock:
            result: Dict[str, Dict[str, int]] = {}
            for key, bucket in self._buckets.items():
                result[key] = {
                    "requests_this_minute": bucket.requests_this_minute,
                    "requests_this_hour": bucket.requests_this_hour,
                    "requests_this_day": bucket.requests_this_day,
                }
            return result

    @property
    def count(self) -> int:
        """Return the number of active rate limit buckets."""
        return len(self._buckets)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bucket_key(self, tenant_id: str, principal_id: str) -> str:
        """Generate unique bucket key for tenant+principal.

        Args:
            tenant_id: Tenant identifier.
            principal_id: Principal identifier.

        Returns:
            Combined bucket key string.
        """
        return f"{tenant_id}:{principal_id}"

    def _get_or_create_bucket(self, key: str) -> RateLimitBucket:
        """Get or create a rate limit bucket (call under lock).

        Args:
            key: Bucket key.

        Returns:
            The existing or newly created bucket.
        """
        if key not in self._buckets:
            now = time.time()
            self._buckets[key] = RateLimitBucket(
                tokens=self.config.burst_limit,
                last_update=now,
                minute_start=now,
                hour_start=now,
                day_start=now,
            )
        return self._buckets[key]

    def _effective_limits(
        self, role: Optional[str],
    ) -> Tuple[int, int, int]:
        """Compute effective rate limits, applying role overrides.

        Args:
            role: Optional role for override lookup.

        Returns:
            Tuple of (rpm, rph, rpd).
        """
        rpm = self.config.requests_per_minute
        rph = self.config.requests_per_hour
        rpd = self.config.requests_per_day

        if role and role in self.config.role_overrides:
            overrides = self.config.role_overrides[role]
            rpm = overrides.get("requests_per_minute", rpm)
            rph = overrides.get("requests_per_hour", rph)
            rpd = overrides.get("requests_per_day", rpd)

        return rpm, rph, rpd


__all__ = [
    "RateLimiter",
    "RateLimitBucket",
]
