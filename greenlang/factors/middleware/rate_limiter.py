# -*- coding: utf-8 -*-
"""
Tier-Based Rate Limiting Middleware for GreenLang Factors API
=============================================================

Implements a sliding-window rate limiter that enforces per-tier
request limits for the Factors API.  Each incoming request is
identified by ``user_id`` (extracted from JWT claims or API key
context) and the caller's tier determines the allowed throughput.

Rate-limit tiers (requests per minute):

    +--------------+------+-------+--------------+
    | Tier         | RPM  | Burst | Export/15min  |
    +--------------+------+-------+--------------+
    | Community    |   60 |    10 |            1 |
    | Pro          |  600 |    50 |            5 |
    | Enterprise   | 6000 |   200 |           20 |
    | Internal     |60000 |  1000 |          200 |
    +--------------+------+-------+--------------+

Every response receives the standard rate-limit headers:

- ``X-RateLimit-Limit``     -- the tier's per-minute cap
- ``X-RateLimit-Remaining`` -- requests left in the current window
- ``X-RateLimit-Reset``     -- UTC epoch when the window resets

When the limit is exceeded the middleware returns **HTTP 429** with
a ``Retry-After`` header (seconds until the oldest tracked request
expires from the window).

Storage back-ends:

- **In-memory** (default): ``dict``-based, protected by
  ``threading.Lock``.  Suitable for single-instance deployments.
- **Redis** (optional): pass a ``redis.Redis`` instance via
  ``RateLimitConfig.redis_client`` for horizontally-scaled
  deployments.  Uses a sorted-set sliding window.

Author: GreenLang Framework Team
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request, Response, status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TierSpec:
    """Immutable rate-limit parameters for a single tier."""

    requests_per_minute: int
    burst: int
    exports_per_15min: int


# Canonical tier definitions from the developer guide.
_TIER_SPECS: Dict[str, _TierSpec] = {
    "community": _TierSpec(requests_per_minute=60, burst=10, exports_per_15min=1),
    "pro": _TierSpec(requests_per_minute=600, burst=50, exports_per_15min=5),
    "enterprise": _TierSpec(requests_per_minute=6000, burst=200, exports_per_15min=20),
    "internal": _TierSpec(requests_per_minute=60000, burst=1000, exports_per_15min=200),
}

# Window durations in seconds.
_GENERAL_WINDOW_SECONDS: int = 60
_EXPORT_WINDOW_SECONDS: int = 15 * 60  # 15 minutes


@dataclass
class RateLimitConfig:
    """
    Configuration for the tier-based rate limiter.

    Attributes:
        redis_client: Optional ``redis.Redis`` instance.  When provided
            the limiter uses Redis sorted sets instead of in-memory
            storage, enabling multi-instance deployments.
        enabled: Master kill-switch.  When ``False`` no limiting is
            applied (useful for load-test environments).
    """

    redis_client: Any = None
    enabled: bool = True


# ---------------------------------------------------------------------------
# Storage back-ends
# ---------------------------------------------------------------------------


class _InMemoryBackend:
    """
    Thread-safe in-memory sliding-window storage.

    Each key maps to a deque of ``float`` timestamps.  Expired
    entries are pruned on every ``record_and_check`` call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: Dict[str, Deque[float]] = {}

    def record_and_check(
        self,
        key: str,
        now: float,
        window_seconds: int,
        max_requests: int,
    ) -> Tuple[bool, int, float]:
        """
        Record a request and check against the limit.

        Args:
            key: Composite key, e.g. ``"rl:general:{user_id}"``.
            now: Current epoch timestamp (``time.time()``).
            window_seconds: Sliding-window size in seconds.
            max_requests: Maximum requests allowed in the window.

        Returns:
            Tuple of ``(allowed, remaining, reset_epoch)``.
            ``allowed`` is ``True`` when the request is within limits.
            ``remaining`` is the count of requests still available.
            ``reset_epoch`` is the UTC epoch second when the oldest
            tracked request will drop out of the window.
        """
        cutoff = now - window_seconds

        with self._lock:
            dq = self._windows.get(key)
            if dq is None:
                dq = deque()
                self._windows[key] = dq

            # Evict expired timestamps from the left.
            while dq and dq[0] <= cutoff:
                dq.popleft()

            count = len(dq)

            if count >= max_requests:
                # Oldest entry determines when the window opens up.
                reset_at = dq[0] + window_seconds
                return False, 0, reset_at

            dq.append(now)
            remaining = max(0, max_requests - count - 1)
            reset_at = dq[0] + window_seconds if dq else now + window_seconds
            return True, remaining, reset_at

    def clear(self) -> None:
        """Remove all tracked windows (useful for testing)."""
        with self._lock:
            self._windows.clear()


class _RedisBackend:
    """
    Redis sorted-set sliding-window storage.

    Each key maps to a sorted set where scores are timestamps.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def record_and_check(
        self,
        key: str,
        now: float,
        window_seconds: int,
        max_requests: int,
    ) -> Tuple[bool, int, float]:
        """Record a request in Redis and check against the limit."""
        cutoff = now - window_seconds
        pipe = self._client.pipeline(transaction=True)

        try:
            # Remove expired entries.
            pipe.zremrangebyscore(key, "-inf", cutoff)
            # Count current entries.
            pipe.zcard(key)
            # Add the new entry (unique member via timestamp + thread id).
            member = f"{now}:{threading.get_ident()}"
            pipe.zadd(key, {member: now})
            # Set key TTL slightly beyond the window for auto-cleanup.
            pipe.expire(key, window_seconds + 10)
            # Get the oldest entry score.
            pipe.zrange(key, 0, 0, withscores=True)

            results = pipe.execute()
        except Exception:
            logger.exception(
                "Redis rate-limit pipeline failed; allowing request"
            )
            return True, max_requests - 1, now + window_seconds

        current_count = results[1]  # zcard result (before zadd)

        if current_count >= max_requests:
            # Remove the entry we just added -- request is denied.
            try:
                self._client.zrem(key, member)
            except Exception:
                pass
            oldest_entries = results[4]
            if oldest_entries:
                reset_at = oldest_entries[0][1] + window_seconds
            else:
                reset_at = now + window_seconds
            return False, 0, reset_at

        oldest_entries = results[4]
        if oldest_entries:
            reset_at = oldest_entries[0][1] + window_seconds
        else:
            reset_at = now + window_seconds
        remaining = max(0, max_requests - current_count - 1)
        return True, remaining, reset_at

    def clear(self) -> None:
        """No-op for Redis (keys expire automatically)."""


# ---------------------------------------------------------------------------
# Core limiter
# ---------------------------------------------------------------------------


class TierRateLimiter:
    """
    Tier-aware sliding-window rate limiter.

    Designed to be instantiated once at application startup and
    shared across all routes.  Two public methods are exposed for
    use as FastAPI dependencies:

    - ``check_general(request, response, user_id, tier)``
    - ``check_export(request, response, user_id, tier)``

    Example::

        limiter = TierRateLimiter()

        @router.get("/factors")
        async def list_factors(
            request: Request,
            response: Response,
            current_user: dict = Depends(get_current_user),
        ):
            limiter.check_general(request, response,
                                  current_user["user_id"],
                                  current_user.get("tier", "community"))
            ...
    """

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        self._config = config or RateLimitConfig()
        if self._config.redis_client is not None:
            self._backend: Any = _RedisBackend(self._config.redis_client)
        else:
            self._backend = _InMemoryBackend()

    @property
    def enabled(self) -> bool:
        """Whether rate limiting is currently active."""
        return self._config.enabled

    # -- public methods -----------------------------------------------------

    def check_general(
        self,
        request: Request,
        response: Response,
        user_id: str,
        tier: str,
    ) -> None:
        """
        Enforce the general per-minute rate limit.

        Args:
            request: The incoming FastAPI request.
            response: The outgoing FastAPI response (headers are set).
            user_id: Caller identity string.
            tier: Caller tier (community / pro / enterprise / internal).

        Raises:
            HTTPException: 429 when the rate limit is exceeded.
        """
        if not self._config.enabled:
            return

        spec = self._resolve_spec(tier)
        now = time.time()
        key = f"rl:general:{user_id}"

        allowed, remaining, reset_at = self._backend.record_and_check(
            key, now, _GENERAL_WINDOW_SECONDS, spec.requests_per_minute,
        )

        _set_headers(response, spec.requests_per_minute, remaining, reset_at)

        if not allowed:
            retry_after = max(1, int(reset_at - now))
            logger.warning(
                "Rate limit exceeded: user=%s tier=%s limit=%d path=%s",
                user_id, tier, spec.requests_per_minute, request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {spec.requests_per_minute} "
                    f"requests per {_GENERAL_WINDOW_SECONDS}s. "
                    f"Upgrade your tier for higher limits."
                ),
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(spec.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_at)),
                },
            )

    def check_export(
        self,
        request: Request,
        response: Response,
        user_id: str,
        tier: str,
    ) -> None:
        """
        Enforce the tighter export rate limit (per 15 minutes).

        This should be called **in addition to** ``check_general``
        on export endpoints.

        Args:
            request: The incoming FastAPI request.
            response: The outgoing FastAPI response (headers are set).
            user_id: Caller identity string.
            tier: Caller tier (community / pro / enterprise / internal).

        Raises:
            HTTPException: 429 when the export rate limit is exceeded.
        """
        if not self._config.enabled:
            return

        spec = self._resolve_spec(tier)
        now = time.time()
        key = f"rl:export:{user_id}"

        allowed, remaining, reset_at = self._backend.record_and_check(
            key, now, _EXPORT_WINDOW_SECONDS, spec.exports_per_15min,
        )

        # Override the general headers with export-specific ones.
        _set_headers(response, spec.exports_per_15min, remaining, reset_at)

        if not allowed:
            retry_after = max(1, int(reset_at - now))
            logger.warning(
                "Export rate limit exceeded: user=%s tier=%s "
                "limit=%d/15min path=%s",
                user_id, tier, spec.exports_per_15min, request.url.path,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Export rate limit exceeded: {spec.exports_per_15min} "
                    f"exports per 15 minutes. "
                    f"Upgrade your tier for higher limits."
                ),
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(spec.exports_per_15min),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_at)),
                },
            )

    def clear(self) -> None:
        """Reset all tracked windows (for testing)."""
        self._backend.clear()

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _resolve_spec(tier: str) -> _TierSpec:
        """Normalize tier string and return its spec."""
        normalized = tier.lower().strip() if tier else "community"
        return _TIER_SPECS.get(normalized, _TIER_SPECS["community"])


# ---------------------------------------------------------------------------
# Header helper
# ---------------------------------------------------------------------------


def _set_headers(
    response: Response,
    limit: int,
    remaining: int,
    reset_at: float,
) -> None:
    """Attach standard rate-limit headers to the response."""
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(int(reset_at))


# ---------------------------------------------------------------------------
# Module-level singleton and configuration
# ---------------------------------------------------------------------------

_default_limiter: Optional[TierRateLimiter] = None
_limiter_init_lock = threading.Lock()


def get_rate_limiter() -> TierRateLimiter:
    """
    Return the module-level singleton rate limiter, creating it
    lazily with default (in-memory) configuration.

    Returns:
        The shared ``TierRateLimiter`` instance.
    """
    global _default_limiter
    if _default_limiter is None:
        with _limiter_init_lock:
            if _default_limiter is None:
                _default_limiter = TierRateLimiter()
    return _default_limiter


def configure_limiter(config: RateLimitConfig) -> TierRateLimiter:
    """
    Configure (or reconfigure) the module-level rate limiter.

    Call this during application startup to inject a Redis client
    or to disable rate limiting entirely.

    Args:
        config: Rate limiter configuration.

    Returns:
        The configured ``TierRateLimiter`` instance.
    """
    global _default_limiter
    with _limiter_init_lock:
        _default_limiter = TierRateLimiter(config)
    return _default_limiter


# ---------------------------------------------------------------------------
# FastAPI dependency helpers
# ---------------------------------------------------------------------------


def apply_rate_limit(
    request: Request,
    response: Response,
    current_user: Dict[str, Any],
) -> None:
    """
    Apply the general rate limit for the given user context.

    This is a synchronous helper intended to be called at the top
    of a route handler after ``current_user`` has been resolved::

        @router.get("/factors")
        async def list_factors(
            request: Request,
            response: Response,
            current_user: dict = Depends(get_current_user),
        ):
            apply_rate_limit(request, response, current_user)
            ...

    Args:
        request: The incoming FastAPI request.
        response: The outgoing FastAPI response.
        current_user: User context dict with ``user_id`` and ``tier``.
    """
    limiter = get_rate_limiter()
    user_id = current_user.get("user_id", "anonymous")
    tier = current_user.get("tier", "community")
    limiter.check_general(request, response, user_id, tier)


def apply_export_rate_limit(
    request: Request,
    response: Response,
    current_user: Dict[str, Any],
) -> None:
    """
    Apply both general and export rate limits for the given user.

    This is a synchronous helper intended to be called at the top
    of export-related route handlers::

        @router.get("/factors/export")
        async def export_factors(
            request: Request,
            response: Response,
            current_user: dict = Depends(get_current_user),
        ):
            apply_export_rate_limit(request, response, current_user)
            ...

    Args:
        request: The incoming FastAPI request.
        response: The outgoing FastAPI response.
        current_user: User context dict with ``user_id`` and ``tier``.
    """
    limiter = get_rate_limiter()
    user_id = current_user.get("user_id", "anonymous")
    tier = current_user.get("tier", "community")
    limiter.check_general(request, response, user_id, tier)
    limiter.check_export(request, response, user_id, tier)


# ---------------------------------------------------------------------------
# Class-shape middleware (BaseHTTPMiddleware) for use with
# `app.add_middleware(RateLimitMiddleware)` from factors_app.py.
#
# Launch-tier overrides per FY27 Factors checklist (CTO spec § "Commercial
# packaging"):
#   - community         : 60   req/min
#   - developer_pro     : 1000 req/min
#   - consulting        : 3000 req/min
#   - enterprise        : unlimited (bypass)
#   - internal          : unlimited (bypass)
# ---------------------------------------------------------------------------

from starlette.middleware.base import BaseHTTPMiddleware as _BaseHTTPMiddleware
from starlette.responses import JSONResponse as _JSONResponse


_LAUNCH_TIER_SPECS: Dict[str, _TierSpec] = {
    "community":          _TierSpec(requests_per_minute=60,    burst=10,  exports_per_15min=1),
    "developer_pro":      _TierSpec(requests_per_minute=1000,  burst=100, exports_per_15min=10),
    "pro":                _TierSpec(requests_per_minute=1000,  burst=100, exports_per_15min=10),
    "consulting":         _TierSpec(requests_per_minute=3000,  burst=200, exports_per_15min=30),
    "consulting_platform":_TierSpec(requests_per_minute=3000,  burst=200, exports_per_15min=30),
}

_UNLIMITED_TIERS = {"enterprise", "internal"}


class RateLimitMiddleware(_BaseHTTPMiddleware):
    """Sliding-window rate limiter applied at /v1.

    Uses the existing :class:`TierRateLimiter` backend but overrides the
    per-tier RPM caps with the launch tiers above. Public routes
    (/v1/health, /openapi.json, /metrics, /docs, /redoc) and unlimited
    tiers bypass the limiter entirely.
    """

    BYPASS = {"/v1/health", "/openapi.json", "/docs", "/redoc", "/metrics", "/"}

    def __init__(self, app, *, config: Optional[RateLimitConfig] = None) -> None:
        super().__init__(app)
        self._limiter = TierRateLimiter(config or RateLimitConfig())

    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in self.BYPASS:
            return await call_next(request)
        if not self._limiter.enabled:
            return await call_next(request)

        user = getattr(request.state, "user", None) or {}
        tier = (user.get("tier") or "community").lower()
        if tier in _UNLIMITED_TIERS:
            return await call_next(request)

        spec = _LAUNCH_TIER_SPECS.get(tier, _LAUNCH_TIER_SPECS["community"])
        user_id = user.get("user_id") or user.get("api_key") or request.client.host

        now = time.time()
        key = f"rl:v1:{user_id}"
        allowed, remaining, reset_at = self._limiter._backend.record_and_check(  # type: ignore[attr-defined]
            key, now, _GENERAL_WINDOW_SECONDS, spec.requests_per_minute,
        )
        if not allowed:
            retry_after = max(1, int(reset_at - now))
            return _JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": (
                        f"Tier '{tier}' allows {spec.requests_per_minute} req/min."
                        " Upgrade for higher limits."
                    ),
                    "tier": tier,
                    "limit": spec.requests_per_minute,
                    "retry_after_seconds": retry_after,
                    "upgrade_url": "https://greenlang.ai/pricing",
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(spec.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_at)),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(spec.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_at))
        return response
