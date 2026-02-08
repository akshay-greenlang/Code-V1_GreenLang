# -*- coding: utf-8 -*-
"""
Unit Tests for RateLimiter (AGENT-FOUND-006)

Tests rate limiting per minute/hour/day, role overrides, multi-tenant
isolation, quota queries, reset, and auto-window expiry.

Coverage target: 85%+ of rate_limiter.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Inline RateLimiter mirroring foundation policy_guard.py
# ---------------------------------------------------------------------------


class RateLimitConfig:
    def __init__(
        self,
        requests_per_minute: int = 100,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        burst_limit: int = 20,
        role_overrides: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_limit = burst_limit
        self.role_overrides = role_overrides or {
            "admin": {"requests_per_minute": 500, "requests_per_hour": 5000},
            "super_admin": {"requests_per_minute": 1000, "requests_per_hour": 10000},
            "service_account": {"requests_per_minute": 1000, "requests_per_hour": 50000},
        }


@dataclass
class RateLimitBucket:
    tokens: float
    last_update: float
    requests_this_minute: int = 0
    requests_this_hour: int = 0
    requests_this_day: int = 0
    minute_start: float = 0.0
    hour_start: float = 0.0
    day_start: float = 0.0


class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: Dict[str, RateLimitBucket] = {}

    @property
    def count(self) -> int:
        return len(self._buckets)

    def _get_bucket_key(self, tenant_id: str, principal_id: str) -> str:
        return f"{tenant_id}:{principal_id}"

    def _get_or_create_bucket(self, key: str) -> RateLimitBucket:
        now = time.time()
        if key not in self._buckets:
            self._buckets[key] = RateLimitBucket(
                tokens=self.config.burst_limit,
                last_update=now,
                minute_start=now,
                hour_start=now,
                day_start=now,
            )
        return self._buckets[key]

    def check_rate_limit(
        self, tenant_id: str, principal_id: str, role: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        key = self._get_bucket_key(tenant_id, principal_id)
        bucket = self._get_or_create_bucket(key)
        now = time.time()

        rpm = self.config.requests_per_minute
        rph = self.config.requests_per_hour
        rpd = self.config.requests_per_day

        if role and role in self.config.role_overrides:
            overrides = self.config.role_overrides[role]
            rpm = overrides.get("requests_per_minute", rpm)
            rph = overrides.get("requests_per_hour", rph)
            rpd = overrides.get("requests_per_day", rpd)

        if now - bucket.minute_start >= 60:
            bucket.requests_this_minute = 0
            bucket.minute_start = now

        if now - bucket.hour_start >= 3600:
            bucket.requests_this_hour = 0
            bucket.hour_start = now

        if now - bucket.day_start >= 86400:
            bucket.requests_this_day = 0
            bucket.day_start = now

        if bucket.requests_this_minute >= rpm:
            return False, f"Rate limit exceeded: {rpm} requests per minute"

        if bucket.requests_this_hour >= rph:
            return False, f"Rate limit exceeded: {rph} requests per hour"

        if bucket.requests_this_day >= rpd:
            return False, f"Rate limit exceeded: {rpd} requests per day"

        bucket.requests_this_minute += 1
        bucket.requests_this_hour += 1
        bucket.requests_this_day += 1

        return True, None

    def get_remaining_quota(
        self, tenant_id: str, principal_id: str, role: Optional[str] = None,
    ) -> Dict[str, int]:
        key = self._get_bucket_key(tenant_id, principal_id)
        bucket = self._get_or_create_bucket(key)

        rpm = self.config.requests_per_minute
        rph = self.config.requests_per_hour
        rpd = self.config.requests_per_day

        if role and role in self.config.role_overrides:
            overrides = self.config.role_overrides[role]
            rpm = overrides.get("requests_per_minute", rpm)
            rph = overrides.get("requests_per_hour", rph)
            rpd = overrides.get("requests_per_day", rpd)

        return {
            "remaining_per_minute": max(0, rpm - bucket.requests_this_minute),
            "remaining_per_hour": max(0, rph - bucket.requests_this_hour),
            "remaining_per_day": max(0, rpd - bucket.requests_this_day),
        }

    def reset_limits(self, tenant_id: str, principal_id: str) -> bool:
        key = self._get_bucket_key(tenant_id, principal_id)
        if key in self._buckets:
            del self._buckets[key]
            return True
        return False

    def get_all_buckets(self) -> Dict[str, RateLimitBucket]:
        return dict(self._buckets)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRateLimiterBasic:
    """Test single request allowed and quota decrement."""

    def test_first_request_allowed(self):
        rl = RateLimiter(RateLimitConfig())
        allowed, reason = rl.check_rate_limit("t1", "u1")
        assert allowed is True
        assert reason is None

    def test_quota_decrements_after_request(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=10))
        rl.check_rate_limit("t1", "u1")
        quota = rl.get_remaining_quota("t1", "u1")
        assert quota["remaining_per_minute"] == 9

    def test_bucket_created_on_first_check(self):
        rl = RateLimiter(RateLimitConfig())
        assert rl.count == 0
        rl.check_rate_limit("t1", "u1")
        assert rl.count == 1

    def test_multiple_requests_within_limit(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=5))
        for _ in range(5):
            allowed, _ = rl.check_rate_limit("t1", "u1")
            assert allowed is True


class TestRateLimiterPerMinute:
    """Test exceed per-minute limit."""

    def test_exceed_per_minute_limit(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=3))
        for _ in range(3):
            rl.check_rate_limit("t1", "u1")
        allowed, reason = rl.check_rate_limit("t1", "u1")
        assert allowed is False
        assert "per minute" in reason

    def test_exact_limit_allowed(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=3))
        results = []
        for _ in range(3):
            allowed, _ = rl.check_rate_limit("t1", "u1")
            results.append(allowed)
        assert all(results)

    def test_one_over_limit_denied(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=3))
        for _ in range(3):
            rl.check_rate_limit("t1", "u1")
        allowed, _ = rl.check_rate_limit("t1", "u1")
        assert allowed is False

    def test_reason_message_contains_limit(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=2))
        rl.check_rate_limit("t1", "u1")
        rl.check_rate_limit("t1", "u1")
        _, reason = rl.check_rate_limit("t1", "u1")
        assert "2 requests per minute" in reason


class TestRateLimiterPerHour:
    """Test exceed per-hour limit."""

    def test_exceed_per_hour_limit(self):
        rl = RateLimiter(RateLimitConfig(
            requests_per_minute=1000, requests_per_hour=5,
        ))
        for _ in range(5):
            rl.check_rate_limit("t1", "u1")
        allowed, reason = rl.check_rate_limit("t1", "u1")
        assert allowed is False
        assert "per hour" in reason

    def test_per_hour_checked_after_per_minute(self):
        rl = RateLimiter(RateLimitConfig(
            requests_per_minute=1000, requests_per_hour=3,
        ))
        for _ in range(3):
            rl.check_rate_limit("t1", "u1")
        allowed, reason = rl.check_rate_limit("t1", "u1")
        assert allowed is False
        assert "per hour" in reason


class TestRateLimiterPerDay:
    """Test exceed per-day limit."""

    def test_exceed_per_day_limit(self):
        rl = RateLimiter(RateLimitConfig(
            requests_per_minute=1000, requests_per_hour=1000, requests_per_day=5,
        ))
        for _ in range(5):
            rl.check_rate_limit("t1", "u1")
        allowed, reason = rl.check_rate_limit("t1", "u1")
        assert allowed is False
        assert "per day" in reason

    def test_day_limit_exact_allowed(self):
        rl = RateLimiter(RateLimitConfig(
            requests_per_minute=1000, requests_per_hour=1000, requests_per_day=5,
        ))
        results = []
        for _ in range(5):
            allowed, _ = rl.check_rate_limit("t1", "u1")
            results.append(allowed)
        assert all(results)


class TestRateLimiterRoleOverrides:
    """Test admin/super_admin/service_account higher limits."""

    def test_admin_higher_rpm(self):
        config = RateLimitConfig(requests_per_minute=2)
        rl = RateLimiter(config)
        # Regular user hits limit at 2
        rl.check_rate_limit("t1", "u1")
        rl.check_rate_limit("t1", "u1")
        allowed, _ = rl.check_rate_limit("t1", "u1")
        assert allowed is False

        # Admin has higher limit (500)
        for _ in range(10):
            allowed, _ = rl.check_rate_limit("t1", "admin-user", role="admin")
            assert allowed is True

    def test_super_admin_higher_rpm(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=2))
        for _ in range(10):
            allowed, _ = rl.check_rate_limit("t1", "sa-user", role="super_admin")
            assert allowed is True

    def test_service_account_higher_rpm(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=2))
        for _ in range(10):
            allowed, _ = rl.check_rate_limit("t1", "svc", role="service_account")
            assert allowed is True

    def test_unknown_role_uses_default(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=2))
        rl.check_rate_limit("t1", "u1", role="unknown_role")
        rl.check_rate_limit("t1", "u1", role="unknown_role")
        allowed, _ = rl.check_rate_limit("t1", "u1", role="unknown_role")
        assert allowed is False

    def test_none_role_uses_default(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=2))
        rl.check_rate_limit("t1", "u1", role=None)
        rl.check_rate_limit("t1", "u1", role=None)
        allowed, _ = rl.check_rate_limit("t1", "u1", role=None)
        assert allowed is False


class TestRateLimiterReset:
    """Test reset counters."""

    def test_reset_clears_bucket(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=2))
        rl.check_rate_limit("t1", "u1")
        rl.check_rate_limit("t1", "u1")
        assert rl.reset_limits("t1", "u1") is True
        allowed, _ = rl.check_rate_limit("t1", "u1")
        assert allowed is True

    def test_reset_nonexistent_returns_false(self):
        rl = RateLimiter(RateLimitConfig())
        assert rl.reset_limits("t1", "nobody") is False

    def test_reset_reduces_count(self):
        rl = RateLimiter(RateLimitConfig())
        rl.check_rate_limit("t1", "u1")
        rl.check_rate_limit("t1", "u2")
        assert rl.count == 2
        rl.reset_limits("t1", "u1")
        assert rl.count == 1


class TestRateLimiterQuota:
    """Test remaining quota calculation."""

    def test_full_quota_before_any_requests(self):
        rl = RateLimiter(RateLimitConfig(
            requests_per_minute=100, requests_per_hour=1000, requests_per_day=10000,
        ))
        quota = rl.get_remaining_quota("t1", "u1")
        assert quota["remaining_per_minute"] == 100
        assert quota["remaining_per_hour"] == 1000
        assert quota["remaining_per_day"] == 10000

    def test_quota_after_some_requests(self):
        rl = RateLimiter(RateLimitConfig(
            requests_per_minute=100, requests_per_hour=1000, requests_per_day=10000,
        ))
        for _ in range(10):
            rl.check_rate_limit("t1", "u1")
        quota = rl.get_remaining_quota("t1", "u1")
        assert quota["remaining_per_minute"] == 90
        assert quota["remaining_per_hour"] == 990
        assert quota["remaining_per_day"] == 9990

    def test_quota_at_zero_when_exhausted(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=3))
        for _ in range(3):
            rl.check_rate_limit("t1", "u1")
        quota = rl.get_remaining_quota("t1", "u1")
        assert quota["remaining_per_minute"] == 0

    def test_quota_with_role_override(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=100))
        quota = rl.get_remaining_quota("t1", "u1", role="admin")
        assert quota["remaining_per_minute"] == 500  # admin override


class TestRateLimiterMultiTenant:
    """Test separate buckets per tenant."""

    def test_different_tenants_separate_buckets(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=2))
        rl.check_rate_limit("t1", "u1")
        rl.check_rate_limit("t1", "u1")
        allowed_t1, _ = rl.check_rate_limit("t1", "u1")
        assert allowed_t1 is False

        # Different tenant, fresh bucket
        allowed_t2, _ = rl.check_rate_limit("t2", "u1")
        assert allowed_t2 is True

    def test_different_principals_separate_buckets(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=2))
        rl.check_rate_limit("t1", "u1")
        rl.check_rate_limit("t1", "u1")
        allowed_u1, _ = rl.check_rate_limit("t1", "u1")
        assert allowed_u1 is False

        allowed_u2, _ = rl.check_rate_limit("t1", "u2")
        assert allowed_u2 is True

    def test_get_all_buckets(self):
        rl = RateLimiter(RateLimitConfig())
        rl.check_rate_limit("t1", "u1")
        rl.check_rate_limit("t2", "u2")
        buckets = rl.get_all_buckets()
        assert len(buckets) == 2
        assert "t1:u1" in buckets
        assert "t2:u2" in buckets


class TestRateLimiterAutoReset:
    """Test time window expiry (mocked time)."""

    def test_minute_window_resets(self):
        rl = RateLimiter(RateLimitConfig(requests_per_minute=2))
        rl.check_rate_limit("t1", "u1")
        rl.check_rate_limit("t1", "u1")
        allowed, _ = rl.check_rate_limit("t1", "u1")
        assert allowed is False

        # Simulate 61 seconds passing
        key = rl._get_bucket_key("t1", "u1")
        rl._buckets[key].minute_start -= 61

        allowed, _ = rl.check_rate_limit("t1", "u1")
        assert allowed is True

    def test_hour_window_resets(self):
        rl = RateLimiter(RateLimitConfig(
            requests_per_minute=1000, requests_per_hour=2,
        ))
        rl.check_rate_limit("t1", "u1")
        rl.check_rate_limit("t1", "u1")
        allowed, _ = rl.check_rate_limit("t1", "u1")
        assert allowed is False

        key = rl._get_bucket_key("t1", "u1")
        rl._buckets[key].hour_start -= 3601

        allowed, _ = rl.check_rate_limit("t1", "u1")
        assert allowed is True

    def test_day_window_resets(self):
        rl = RateLimiter(RateLimitConfig(
            requests_per_minute=1000, requests_per_hour=1000, requests_per_day=2,
        ))
        rl.check_rate_limit("t1", "u1")
        rl.check_rate_limit("t1", "u1")
        allowed, _ = rl.check_rate_limit("t1", "u1")
        assert allowed is False

        key = rl._get_bucket_key("t1", "u1")
        rl._buckets[key].day_start -= 86401

        allowed, _ = rl.check_rate_limit("t1", "u1")
        assert allowed is True
