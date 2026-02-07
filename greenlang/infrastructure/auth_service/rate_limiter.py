# -*- coding: utf-8 -*-
"""
Auth Rate Limiter - JWT Authentication Service (SEC-001)

Sliding-window rate limiting for authentication endpoints using Redis
sorted sets.  Each request is recorded as a timestamped member; stale
entries are pruned on every check to maintain an accurate count.

Supports per-endpoint limits with sensible defaults for login, token
refresh, validation, password reset, and MFA verification flows.

Classes:
    - RateLimitConfig: Immutable per-endpoint rate limit configuration.
    - RateLimitExceeded: Exception raised when a limit is exceeded.
    - AuthRateLimiter: Core sliding-window rate limiter.

Example:
    >>> limiter = AuthRateLimiter(redis_client=redis)
    >>> await limiter.check_login_rate("192.168.1.10")  # passes
    >>> await limiter.check_login_rate("192.168.1.10")  # passes
    >>> # ... after exceeding the limit ...
    >>> await limiter.check_login_rate("192.168.1.10")
    RateLimitExceeded: 10 requests per 60s exceeded; retry after 42s

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Redis key prefix
# ---------------------------------------------------------------------------

_RATE_LIMIT_PREFIX = "gl:auth:rl:"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitConfig:
    """Immutable per-endpoint rate limit configuration.

    All values are *requests per minute* unless noted otherwise.

    Attributes:
        login_per_ip_per_minute: Max login attempts per IP per minute.
        token_per_ip_per_minute: Max token-issue requests per IP per minute.
        refresh_per_ip_per_minute: Max token-refresh requests per IP per minute.
        validate_per_ip_per_minute: Max token-validation requests per IP per minute.
        password_reset_per_ip_per_minute: Max password-reset requests per IP per minute.
        mfa_verify_per_ip_per_minute: Max MFA verification requests per IP per minute.
    """

    login_per_ip_per_minute: int = 10
    token_per_ip_per_minute: int = 20
    refresh_per_ip_per_minute: int = 30
    validate_per_ip_per_minute: int = 1000
    password_reset_per_ip_per_minute: int = 3
    mfa_verify_per_ip_per_minute: int = 10


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class RateLimitExceeded(Exception):
    """Raised when a rate limit is exceeded.

    Attributes:
        limit: The configured limit that was exceeded.
        window_seconds: The sliding-window duration in seconds.
        retry_after: Seconds until the oldest request in the window expires.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after: int,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        super().__init__(
            f"{limit} requests per {window_seconds}s exceeded; "
            f"retry after {retry_after}s"
        )


# ---------------------------------------------------------------------------
# AuthRateLimiter
# ---------------------------------------------------------------------------


class AuthRateLimiter:
    """Sliding-window rate limiter backed by Redis sorted sets.

    Each tracked request is inserted into a sorted set keyed by endpoint
    and client identifier.  On every check, entries outside the window are
    pruned, and the remaining cardinality is compared against the limit.

    Attributes:
        config: Rate-limit configuration.

    Example:
        >>> limiter = AuthRateLimiter(redis_client=redis)
        >>> await limiter.check_login_rate("10.0.0.1")
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        redis_client: Any = None,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            config: Rate-limit configuration.  Uses defaults if ``None``.
            redis_client: Async Redis client (``redis.asyncio.Redis``).
        """
        self.config = config or RateLimitConfig()
        self._redis = redis_client

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60,
    ) -> bool:
        """Check whether a request is within the rate limit.

        If the limit is exceeded, raises ``RateLimitExceeded``.

        The method uses a Redis sorted set where each member is a unique
        request ID scored by its Unix timestamp.  Stale members outside
        the window are pruned atomically.

        Args:
            key: Fully-qualified Redis key (including prefix and identifiers).
            limit: Maximum number of requests allowed in the window.
            window_seconds: Sliding-window size in seconds.

        Returns:
            ``True`` if the request is allowed.

        Raises:
            RateLimitExceeded: If the request would exceed the limit.
        """
        if self._redis is None:
            logger.debug("No Redis client; rate limiting is disabled")
            return True

        now = time.time()
        cutoff = now - window_seconds
        member = f"{now}:{uuid.uuid4().hex[:8]}"

        pipe = self._redis.pipeline(transaction=True)
        pipe.zremrangebyscore(key, "-inf", cutoff)
        pipe.zadd(key, {member: now})
        pipe.zcard(key)
        pipe.expire(key, window_seconds + 10)
        results = await pipe.execute()

        current_count: int = results[2]

        if current_count > limit:
            # Determine when the oldest entry in the window will expire
            oldest_entries = await self._redis.zrange(key, 0, 0, withscores=True)
            retry_after = window_seconds
            if oldest_entries:
                oldest_score = oldest_entries[0][1]
                retry_after = max(1, int(oldest_score + window_seconds - now))

            logger.warning(
                "Rate limit exceeded: key=%s count=%d limit=%d retry_after=%ds",
                key,
                current_count,
                limit,
                retry_after,
            )
            raise RateLimitExceeded(
                limit=limit,
                window_seconds=window_seconds,
                retry_after=retry_after,
            )

        logger.debug(
            "Rate limit check passed: key=%s count=%d/%d",
            key,
            current_count,
            limit,
        )
        return True

    # ------------------------------------------------------------------
    # Endpoint-specific helpers
    # ------------------------------------------------------------------

    async def check_login_rate(self, ip_address: str) -> None:
        """Check login rate limit for an IP address.

        Args:
            ip_address: Client IP address.

        Raises:
            RateLimitExceeded: If the login rate limit is exceeded.
        """
        key = f"{_RATE_LIMIT_PREFIX}login:{ip_address}"
        await self.check_rate_limit(
            key,
            limit=self.config.login_per_ip_per_minute,
            window_seconds=60,
        )

    async def check_token_rate(self, ip_address: str) -> None:
        """Check token-issue rate limit for an IP address.

        Args:
            ip_address: Client IP address.

        Raises:
            RateLimitExceeded: If the token rate limit is exceeded.
        """
        key = f"{_RATE_LIMIT_PREFIX}token:{ip_address}"
        await self.check_rate_limit(
            key,
            limit=self.config.token_per_ip_per_minute,
            window_seconds=60,
        )

    async def check_refresh_rate(self, ip_address: str) -> None:
        """Check token-refresh rate limit for an IP address.

        Args:
            ip_address: Client IP address.

        Raises:
            RateLimitExceeded: If the refresh rate limit is exceeded.
        """
        key = f"{_RATE_LIMIT_PREFIX}refresh:{ip_address}"
        await self.check_rate_limit(
            key,
            limit=self.config.refresh_per_ip_per_minute,
            window_seconds=60,
        )

    async def check_validation_rate(self, ip_address: str) -> None:
        """Check token-validation rate limit for an IP address.

        Args:
            ip_address: Client IP address.

        Raises:
            RateLimitExceeded: If the validation rate limit is exceeded.
        """
        key = f"{_RATE_LIMIT_PREFIX}validate:{ip_address}"
        await self.check_rate_limit(
            key,
            limit=self.config.validate_per_ip_per_minute,
            window_seconds=60,
        )

    async def check_password_reset_rate(self, ip_address: str) -> None:
        """Check password-reset rate limit for an IP address.

        Args:
            ip_address: Client IP address.

        Raises:
            RateLimitExceeded: If the password-reset rate limit is exceeded.
        """
        key = f"{_RATE_LIMIT_PREFIX}pwreset:{ip_address}"
        await self.check_rate_limit(
            key,
            limit=self.config.password_reset_per_ip_per_minute,
            window_seconds=60,
        )

    async def check_mfa_rate(self, ip_address: str) -> None:
        """Check MFA verification rate limit for an IP address.

        Args:
            ip_address: Client IP address.

        Raises:
            RateLimitExceeded: If the MFA rate limit is exceeded.
        """
        key = f"{_RATE_LIMIT_PREFIX}mfa:{ip_address}"
        await self.check_rate_limit(
            key,
            limit=self.config.mfa_verify_per_ip_per_minute,
            window_seconds=60,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    async def get_remaining(
        self,
        key: str,
        limit: int,
        window_seconds: int = 60,
    ) -> int:
        """Get the number of remaining requests in the current window.

        Args:
            key: Fully-qualified Redis key.
            limit: Configured limit for the endpoint.
            window_seconds: Sliding-window size in seconds.

        Returns:
            Number of remaining allowed requests (>= 0).
        """
        if self._redis is None:
            return limit

        cutoff = time.time() - window_seconds
        await self._redis.zremrangebyscore(key, "-inf", cutoff)
        current = await self._redis.zcard(key)

        return max(0, limit - current)

    async def reset(self, key: str) -> None:
        """Reset a rate-limit counter (admin action).

        Args:
            key: Fully-qualified Redis key to clear.
        """
        if self._redis is None:
            return

        await self._redis.delete(key)
        logger.info("Rate limit counter reset: %s", key)

    async def reset_for_ip(self, ip_address: str) -> None:
        """Reset all rate-limit counters for a specific IP address.

        Removes keys for all endpoint types (login, token, refresh,
        validate, password reset, MFA).

        Args:
            ip_address: Client IP address to reset.
        """
        if self._redis is None:
            return

        endpoints = ["login", "token", "refresh", "validate", "pwreset", "mfa"]
        keys = [f"{_RATE_LIMIT_PREFIX}{ep}:{ip_address}" for ep in endpoints]

        if keys:
            await self._redis.delete(*keys)
            logger.info("All rate limit counters reset for IP %s", ip_address)
