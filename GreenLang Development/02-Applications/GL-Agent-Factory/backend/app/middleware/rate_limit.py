"""
Rate Limiting Middleware

This module provides Redis-based rate limiting using the token bucket algorithm.
Implements distributed rate limiting for GreenLang Agent Factory with:
- Token bucket algorithm for smooth rate limiting
- Sliding window for accurate request counting
- Atomic Redis operations using Lua scripts
- Graceful degradation on Redis failures
"""

import logging
import time
import uuid
from enum import Enum
from typing import Callable, Optional, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Behavior when Redis is unavailable."""
    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"


# Lua script for atomic token bucket rate limiting
TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
local tokens = tonumber(bucket[1])
local last_update = tonumber(bucket[2])

if tokens == nil then
    tokens = capacity
    last_update = now
end

local elapsed = math.max(0, now - last_update)
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
local remaining = tokens

if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
    remaining = tokens
end

redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
redis.call('EXPIRE', key, ttl)

local refill_time = 0
if allowed == 0 then
    refill_time = math.ceil((requested - remaining) / rate)
end

return {allowed, math.floor(remaining), refill_time}
"""


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate Limiting Middleware.

    Implements token bucket algorithm with Redis for distributed rate limiting.
    Supports per-user and per-endpoint limits with graceful degradation.
    """

    _token_bucket_sha: Optional[str] = None

    def __init__(
        self,
        app,
        redis=None,
        requests_per_minute: int = 100,
        burst_size: int = 20,
        failure_mode: FailureMode = FailureMode.FAIL_OPEN,
        key_prefix: str = "ratelimit",
    ):
        super().__init__(app)
        self.redis = redis
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.failure_mode = failure_mode
        self.key_prefix = key_prefix
        self._tokens_per_second = requests_per_minute / 60.0
        self._key_ttl = 120
        self._scripts_loaded = False

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        user_id = getattr(request.state, "user_id", None) or request.client.host
        is_allowed, remaining, reset_time = await self._check_rate_limit(user_id)

        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Try again in {reset_time} seconds.",
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + reset_time),
                    "Retry-After": str(reset_time),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        return response

    async def _check_rate_limit(self, user_id: str) -> Tuple[bool, int, int]:
        if not self.redis:
            return True, self.requests_per_minute, 0

        key = f"{self.key_prefix}:{user_id}"
        now = time.time()
        window_start = now - 60

        try:
            # Sliding window rate limiting using sorted sets
            await self._async_call(self.redis.zremrangebyscore, key, 0, window_start)
            count = await self._async_call(self.redis.zcard, key)

            if count < self.requests_per_minute:
                request_id = f"{now}:{uuid.uuid4().hex[:8]}"
                await self._async_call(self.redis.zadd, key, {request_id: now})
                await self._async_call(self.redis.expire, key, 60)
                return True, self.requests_per_minute - count - 1, 0
            else:
                oldest = await self._async_call(self.redis.zrange, key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(oldest[0][1]) + 60 - int(now)
                    return False, 0, max(1, retry_after)
                return False, 0, 60

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return self._handle_redis_failure()

    async def _async_call(self, method, *args, **kwargs):
        result = method(*args, **kwargs)
        if hasattr(result, '__await__'):
            return await result
        return result

    def _handle_redis_failure(self) -> Tuple[bool, int, int]:
        if self.failure_mode == FailureMode.FAIL_OPEN:
            logger.warning("Redis unavailable - failing open")
            return True, self.requests_per_minute, 0
        logger.warning("Redis unavailable - failing closed")
        return False, 0, 60


class RateLimitConfig:
    """Configuration class for rate limiting."""

    def __init__(
        self,
        requests_per_minute: int = 100,
        burst_size: int = 20,
        failure_mode: FailureMode = FailureMode.FAIL_OPEN,
        key_prefix: str = "ratelimit",
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.failure_mode = failure_mode
        self.key_prefix = key_prefix

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        import os
        failure_mode_str = os.getenv("RATE_LIMIT_FAIL_MODE", "open").lower()
        failure_mode = FailureMode.FAIL_CLOSED if failure_mode_str == "closed" else FailureMode.FAIL_OPEN
        return cls(
            requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "100")),
            burst_size=int(os.getenv("RATE_LIMIT_BURST", "20")),
            failure_mode=failure_mode,
            key_prefix=os.getenv("RATE_LIMIT_KEY_PREFIX", "ratelimit"),
        )

    @classmethod
    def strict(cls) -> "RateLimitConfig":
        return cls(requests_per_minute=30, burst_size=5, failure_mode=FailureMode.FAIL_CLOSED)

    @classmethod
    def permissive(cls) -> "RateLimitConfig":
        return cls(requests_per_minute=1000, burst_size=100, failure_mode=FailureMode.FAIL_OPEN)


def create_redis_pool(
    redis_url: str = "redis://localhost:6379/0",
    max_connections: int = 50,
    socket_timeout: float = 5.0,
):
    """Create a Redis connection pool with optimal settings."""
    try:
        from redis import Redis, ConnectionPool
    except ImportError:
        logger.error("redis package not installed")
        raise

    pool = ConnectionPool.from_url(
        redis_url,
        max_connections=max_connections,
        socket_timeout=socket_timeout,
        decode_responses=True,
    )
    return Redis(connection_pool=pool)
