# -*- coding: utf-8 -*-
"""
Unit tests for the Tier-Based Rate Limiting Middleware.

Tests cover:
- Sliding-window algorithm correctness
- Per-tier limits (community, pro, enterprise, internal)
- Export endpoint limits (separate, tighter window)
- Rate-limit response headers (X-RateLimit-*)
- HTTP 429 with Retry-After when exceeded
- Thread safety
- Redis backend (mocked)
- Edge cases (unknown tier, missing user_id, disabled limiter)

Author: GreenLang Framework Team
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.factors.middleware.rate_limiter import (
    RateLimitConfig,
    TierRateLimiter,
    _InMemoryBackend,
    _TIER_SPECS,
    _GENERAL_WINDOW_SECONDS,
    _EXPORT_WINDOW_SECONDS,
    apply_export_rate_limit,
    apply_rate_limit,
    configure_limiter,
    get_rate_limiter,
)


# ---------------------------------------------------------------------------
# Helpers: Fake Request / Response for unit testing without FastAPI
# ---------------------------------------------------------------------------


class FakeClient:
    """Simulates request.client for IP extraction."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.host = host


class FakeURL:
    """Simulates request.url for path extraction in log messages."""

    def __init__(self, path: str = "/api/v1/factors") -> None:
        self.path = path


class FakeRequest:
    """Minimal request stub for rate limiter testing."""

    def __init__(
        self,
        client_host: str = "127.0.0.1",
        path: str = "/api/v1/factors",
    ) -> None:
        self.client = FakeClient(client_host)
        self.url = FakeURL(path)
        self.state = _FakeState()


class _FakeState:
    """Mutable state object (mirrors Starlette request.state)."""
    pass


class FakeResponse:
    """Minimal response stub that captures headers."""

    def __init__(self) -> None:
        self.headers: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def limiter() -> TierRateLimiter:
    """Fresh in-memory rate limiter for each test."""
    return TierRateLimiter(RateLimitConfig(enabled=True))


@pytest.fixture()
def disabled_limiter() -> TierRateLimiter:
    """Rate limiter with enabled=False."""
    return TierRateLimiter(RateLimitConfig(enabled=False))


@pytest.fixture()
def req() -> FakeRequest:
    return FakeRequest()


@pytest.fixture()
def resp() -> FakeResponse:
    return FakeResponse()


# ---------------------------------------------------------------------------
# Tests: In-memory backend
# ---------------------------------------------------------------------------


class TestInMemoryBackend:
    """Tests for _InMemoryBackend sliding-window logic."""

    def test_first_request_allowed(self) -> None:
        backend = _InMemoryBackend()
        allowed, remaining, reset_at = backend.record_and_check(
            "user1", time.time(), 60, 10
        )
        assert allowed is True
        assert remaining == 9

    def test_requests_up_to_limit_allowed(self) -> None:
        backend = _InMemoryBackend()
        now = time.time()
        for i in range(10):
            allowed, remaining, _ = backend.record_and_check(
                "user1", now + i * 0.001, 60, 10
            )
            assert allowed is True
            assert remaining == 10 - i - 1

    def test_request_beyond_limit_denied(self) -> None:
        backend = _InMemoryBackend()
        now = time.time()
        # Fill up to limit.
        for i in range(10):
            backend.record_and_check("user1", now + i * 0.001, 60, 10)
        # 11th request should be denied.
        allowed, remaining, reset_at = backend.record_and_check(
            "user1", now + 0.011, 60, 10
        )
        assert allowed is False
        assert remaining == 0
        assert reset_at > now

    def test_expired_entries_are_pruned(self) -> None:
        backend = _InMemoryBackend()
        now = time.time()
        # Fill up to limit in the past.
        for i in range(10):
            backend.record_and_check("user1", now - 61 + i * 0.001, 60, 10)
        # All should have expired; next request is allowed.
        allowed, remaining, _ = backend.record_and_check(
            "user1", now, 60, 10
        )
        assert allowed is True
        assert remaining == 9

    def test_sliding_window_partial_expiry(self) -> None:
        backend = _InMemoryBackend()
        now = time.time()
        # Add 5 requests 30 seconds ago.
        for i in range(5):
            backend.record_and_check("user1", now - 30 + i * 0.001, 60, 10)
        # Add 5 more now.
        for i in range(5):
            backend.record_and_check("user1", now + i * 0.001, 60, 10)
        # Should be at limit.
        allowed, remaining, _ = backend.record_and_check(
            "user1", now + 0.006, 60, 10
        )
        assert allowed is False
        assert remaining == 0

    def test_separate_keys_independent(self) -> None:
        backend = _InMemoryBackend()
        now = time.time()
        # Fill user1 to limit.
        for i in range(10):
            backend.record_and_check("user1", now + i * 0.001, 60, 10)
        # user2 should still be allowed.
        allowed, remaining, _ = backend.record_and_check(
            "user2", now, 60, 10
        )
        assert allowed is True
        assert remaining == 9

    def test_clear_resets_all_windows(self) -> None:
        backend = _InMemoryBackend()
        now = time.time()
        for i in range(10):
            backend.record_and_check("user1", now + i * 0.001, 60, 10)
        backend.clear()
        allowed, remaining, _ = backend.record_and_check(
            "user1", now + 1, 60, 10
        )
        assert allowed is True
        assert remaining == 9

    def test_reset_epoch_is_correct(self) -> None:
        backend = _InMemoryBackend()
        t0 = time.time()
        backend.record_and_check("user1", t0, 60, 10)
        _, _, reset_at = backend.record_and_check("user1", t0 + 1, 60, 10)
        # Reset should be first entry time + window.
        assert abs(reset_at - (t0 + 60)) < 0.01


# ---------------------------------------------------------------------------
# Tests: TierRateLimiter.check_general
# ---------------------------------------------------------------------------


class TestCheckGeneral:
    """Tests for the general (per-minute) rate limit check."""

    def test_community_tier_limit_60(self, limiter, req, resp) -> None:
        """Community tier allows 60 requests per minute."""
        for i in range(60):
            limiter.check_general(req, resp, f"user_{i}", "community")
        # 60 requests from the SAME user should fail on the 61st.
        for i in range(60):
            limiter.check_general(req, resp, "user_single", "community")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            limiter.check_general(req, resp, "user_single", "community")
        assert exc_info.value.status_code == 429

    def test_pro_tier_limit_600(self, limiter, req, resp) -> None:
        """Pro tier allows 600 requests per minute."""
        for i in range(600):
            limiter.check_general(req, resp, "pro_user", "pro")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            limiter.check_general(req, resp, "pro_user", "pro")
        assert exc_info.value.status_code == 429

    def test_enterprise_tier_limit_6000(self, limiter, req, resp) -> None:
        """Enterprise tier allows 6000 requests per minute."""
        # Test with a smaller batch to avoid slow test.
        spec = _TIER_SPECS["enterprise"]
        assert spec.requests_per_minute == 6000

    def test_internal_tier_limit_60000(self, limiter, req, resp) -> None:
        """Internal tier allows 60000 requests per minute."""
        spec = _TIER_SPECS["internal"]
        assert spec.requests_per_minute == 60000

    def test_unknown_tier_defaults_to_community(
        self, limiter, req, resp
    ) -> None:
        """Unknown tiers fall back to community limits."""
        for i in range(60):
            limiter.check_general(req, resp, "unknown_user", "gold")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            limiter.check_general(req, resp, "unknown_user", "gold")
        assert exc_info.value.status_code == 429

    def test_headers_set_on_success(self, limiter, req, resp) -> None:
        """Successful request should set X-RateLimit-* headers."""
        limiter.check_general(req, resp, "test_user", "community")
        assert "X-RateLimit-Limit" in resp.headers
        assert resp.headers["X-RateLimit-Limit"] == "60"
        assert "X-RateLimit-Remaining" in resp.headers
        assert int(resp.headers["X-RateLimit-Remaining"]) == 59
        assert "X-RateLimit-Reset" in resp.headers
        assert int(resp.headers["X-RateLimit-Reset"]) > 0

    def test_headers_on_429(self, limiter, req) -> None:
        """429 response should include Retry-After header."""
        resp = FakeResponse()
        for i in range(60):
            limiter.check_general(req, resp, "hdr_user", "community")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            resp2 = FakeResponse()
            limiter.check_general(req, resp2, "hdr_user", "community")
        exc = exc_info.value
        assert "Retry-After" in exc.headers
        assert int(exc.headers["Retry-After"]) >= 1
        assert exc.headers["X-RateLimit-Remaining"] == "0"

    def test_disabled_limiter_allows_everything(
        self, disabled_limiter, req, resp
    ) -> None:
        """When disabled, all requests pass through."""
        for i in range(200):
            disabled_limiter.check_general(req, resp, "user", "community")
        # No exception should be raised.

    def test_tier_normalization_case_insensitive(
        self, limiter, req, resp
    ) -> None:
        """Tier matching is case-insensitive and strips whitespace."""
        limiter.check_general(req, resp, "user", "  PRO  ")
        assert resp.headers["X-RateLimit-Limit"] == "600"


# ---------------------------------------------------------------------------
# Tests: TierRateLimiter.check_export
# ---------------------------------------------------------------------------


class TestCheckExport:
    """Tests for the export-specific rate limit check."""

    def test_community_export_limit_1(self, limiter, req, resp) -> None:
        """Community tier allows 1 export per 15 minutes."""
        limiter.check_export(req, resp, "export_user", "community")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            limiter.check_export(req, resp, "export_user", "community")
        assert exc_info.value.status_code == 429
        assert "Export rate limit" in exc_info.value.detail

    def test_pro_export_limit_5(self, limiter, req, resp) -> None:
        """Pro tier allows 5 exports per 15 minutes."""
        for i in range(5):
            limiter.check_export(req, resp, "pro_export", "pro")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            limiter.check_export(req, resp, "pro_export", "pro")
        assert exc_info.value.status_code == 429

    def test_enterprise_export_limit_20(self, limiter, req, resp) -> None:
        """Enterprise tier allows 20 exports per 15 minutes."""
        for i in range(20):
            limiter.check_export(req, resp, "ent_export", "enterprise")
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            limiter.check_export(req, resp, "ent_export", "enterprise")
        assert exc_info.value.status_code == 429

    def test_export_headers_set(self, limiter, req, resp) -> None:
        """Export check sets correct rate-limit headers."""
        limiter.check_export(req, resp, "export_hdr", "pro")
        assert resp.headers["X-RateLimit-Limit"] == "5"
        assert int(resp.headers["X-RateLimit-Remaining"]) == 4

    def test_export_window_is_15_minutes(self) -> None:
        """Verify the export window constant is 900 seconds."""
        assert _EXPORT_WINDOW_SECONDS == 900

    def test_general_window_is_60_seconds(self) -> None:
        """Verify the general window constant is 60 seconds."""
        assert _GENERAL_WINDOW_SECONDS == 60


# ---------------------------------------------------------------------------
# Tests: apply_rate_limit / apply_export_rate_limit helpers
# ---------------------------------------------------------------------------


class TestApplyHelpers:
    """Tests for the module-level apply_* helper functions."""

    def test_apply_rate_limit_uses_user_context(self) -> None:
        """apply_rate_limit extracts user_id and tier from user dict."""
        # Configure a fresh limiter for isolation.
        configure_limiter(RateLimitConfig(enabled=True))

        req = FakeRequest()
        resp = FakeResponse()
        user = {"user_id": "helper_user", "tier": "pro"}
        apply_rate_limit(req, resp, user)
        assert resp.headers["X-RateLimit-Limit"] == "600"

    def test_apply_rate_limit_defaults_anonymous(self) -> None:
        """Missing user_id defaults to 'anonymous'."""
        configure_limiter(RateLimitConfig(enabled=True))

        req = FakeRequest()
        resp = FakeResponse()
        user = {}  # No user_id, no tier.
        apply_rate_limit(req, resp, user)
        assert resp.headers["X-RateLimit-Limit"] == "60"  # Community default.

    def test_apply_export_rate_limit_checks_both(self) -> None:
        """apply_export_rate_limit checks both general and export limits."""
        configure_limiter(RateLimitConfig(enabled=True))

        req = FakeRequest()
        resp = FakeResponse()
        user = {"user_id": "both_check_user", "tier": "community"}
        apply_export_rate_limit(req, resp, user)
        # Headers should reflect the export limit (set last).
        assert resp.headers["X-RateLimit-Limit"] == "1"

    def test_apply_export_rate_limit_triggers_429_on_export(self) -> None:
        """Export limit blocks after 1 request for community tier."""
        configure_limiter(RateLimitConfig(enabled=True))

        req = FakeRequest()
        user = {"user_id": "export_block_user", "tier": "community"}

        resp1 = FakeResponse()
        apply_export_rate_limit(req, resp1, user)

        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            resp2 = FakeResponse()
            apply_export_rate_limit(req, resp2, user)
        assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# Tests: get_rate_limiter / configure_limiter
# ---------------------------------------------------------------------------


class TestSingletonManagement:
    """Tests for the module-level singleton lifecycle."""

    def test_get_rate_limiter_returns_instance(self) -> None:
        """get_rate_limiter always returns a TierRateLimiter."""
        limiter = get_rate_limiter()
        assert isinstance(limiter, TierRateLimiter)

    def test_get_rate_limiter_returns_same_instance(self) -> None:
        """Consecutive calls return the same object."""
        a = get_rate_limiter()
        b = get_rate_limiter()
        assert a is b

    def test_configure_limiter_replaces_singleton(self) -> None:
        """configure_limiter creates a new instance."""
        old = get_rate_limiter()
        new = configure_limiter(RateLimitConfig(enabled=False))
        assert new is not old
        assert new.enabled is False
        assert get_rate_limiter() is new

    def test_configure_limiter_with_redis(self) -> None:
        """configure_limiter accepts a Redis client."""
        mock_redis = MagicMock()
        limiter = configure_limiter(
            RateLimitConfig(redis_client=mock_redis, enabled=True)
        )
        assert limiter.enabled is True


# ---------------------------------------------------------------------------
# Tests: Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests that concurrent access does not cause data corruption."""

    def test_concurrent_requests_respect_limit(self) -> None:
        """Multiple threads hitting the same user should not exceed limit."""
        limiter = TierRateLimiter(RateLimitConfig(enabled=True))
        allowed_count = 0
        denied_count = 0
        lock = threading.Lock()

        def make_request():
            nonlocal allowed_count, denied_count
            req = FakeRequest()
            resp = FakeResponse()
            try:
                limiter.check_general(req, resp, "thread_user", "community")
                with lock:
                    allowed_count += 1
            except Exception:
                with lock:
                    denied_count += 1

        threads = [threading.Thread(target=make_request) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Community limit is 60, so at most 60 should pass.
        assert allowed_count <= 60
        assert allowed_count + denied_count == 100
        assert denied_count >= 40


# ---------------------------------------------------------------------------
# Tests: Tier spec verification
# ---------------------------------------------------------------------------


class TestTierSpecs:
    """Verify tier specifications match the developer guide."""

    def test_community_spec(self) -> None:
        spec = _TIER_SPECS["community"]
        assert spec.requests_per_minute == 60
        assert spec.burst == 10
        assert spec.exports_per_15min == 1

    def test_pro_spec(self) -> None:
        spec = _TIER_SPECS["pro"]
        assert spec.requests_per_minute == 600
        assert spec.burst == 50
        assert spec.exports_per_15min == 5

    def test_enterprise_spec(self) -> None:
        spec = _TIER_SPECS["enterprise"]
        assert spec.requests_per_minute == 6000
        assert spec.burst == 200
        assert spec.exports_per_15min == 20

    def test_internal_spec(self) -> None:
        spec = _TIER_SPECS["internal"]
        assert spec.requests_per_minute == 60000
        assert spec.burst == 1000
        assert spec.exports_per_15min == 200


# ---------------------------------------------------------------------------
# Tests: clear method
# ---------------------------------------------------------------------------


class TestClear:
    """Tests for the limiter.clear() method."""

    def test_clear_resets_counters(self, limiter, req) -> None:
        """After clear(), limits are reset."""
        for i in range(60):
            resp = FakeResponse()
            limiter.check_general(req, resp, "clear_user", "community")
        limiter.clear()
        resp = FakeResponse()
        # Should succeed after clear.
        limiter.check_general(req, resp, "clear_user", "community")
        assert int(resp.headers["X-RateLimit-Remaining"]) == 59


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_empty_tier_string(self, limiter, req, resp) -> None:
        """Empty tier string defaults to community."""
        limiter.check_general(req, resp, "edge_user", "")
        assert resp.headers["X-RateLimit-Limit"] == "60"

    def test_none_tier(self, limiter, req, resp) -> None:
        """None tier defaults to community."""
        limiter.check_general(req, resp, "edge_user2", None)
        assert resp.headers["X-RateLimit-Limit"] == "60"

    def test_whitespace_tier(self, limiter, req, resp) -> None:
        """Whitespace-only tier defaults to community."""
        limiter.check_general(req, resp, "edge_user3", "   ")
        assert resp.headers["X-RateLimit-Limit"] == "60"

    def test_mixed_case_tier(self, limiter, req, resp) -> None:
        """Mixed-case tier is normalized correctly."""
        limiter.check_general(req, resp, "edge_user4", "EnTeRpRiSe")
        assert resp.headers["X-RateLimit-Limit"] == "6000"

    def test_exact_limit_boundary(self, limiter, req) -> None:
        """Exactly at the limit, the last request should succeed."""
        for i in range(60):
            resp = FakeResponse()
            limiter.check_general(req, resp, "boundary_user", "community")
        # The 60th call should have succeeded with remaining=0.
        assert resp.headers["X-RateLimit-Remaining"] == "0"

    def test_general_and_export_are_separate_windows(
        self, limiter, req
    ) -> None:
        """General and export rate limits use separate tracking keys."""
        # Fill general limit.
        for i in range(60):
            resp = FakeResponse()
            limiter.check_general(req, resp, "sep_user", "community")
        # Export should still work (separate key).
        resp2 = FakeResponse()
        limiter.check_export(req, resp2, "sep_user", "community")
        assert resp2.headers["X-RateLimit-Limit"] == "1"

    def test_different_users_are_independent(self, limiter, req) -> None:
        """Different user_ids have independent rate limits."""
        for i in range(60):
            resp = FakeResponse()
            limiter.check_general(req, resp, "user_a", "community")
        # user_b should still be allowed.
        resp_b = FakeResponse()
        limiter.check_general(req, resp_b, "user_b", "community")
        assert int(resp_b.headers["X-RateLimit-Remaining"]) == 59
