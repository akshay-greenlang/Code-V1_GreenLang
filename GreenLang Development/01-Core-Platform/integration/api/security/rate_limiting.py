"""
Rate Limiting for GreenLang API

This module implements rate limiting using token bucket algorithm
with Redis backend for distributed systems support.

Example:
    >>> from greenlang.api.security.rate_limiting import RateLimiter
    >>> limiter = RateLimiter(redis_url="redis://localhost:6379")
    >>> @limiter.limit("10/minute")
    >>> async def api_endpoint():
    >>>     return {"status": "ok"}
"""

import time
import hashlib
import json
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum

from fastapi import Request, Response, HTTPException, status
from pydantic import BaseModel, Field, validator
import logging
import asyncio

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

logger = logging.getLogger(__name__)


class RateLimitPeriod(str, Enum):
    """Supported rate limit periods."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    redis_url: Optional[str] = Field(None, description="Redis connection URL")
    default_limit: int = Field(100, ge=1, description="Default request limit")
    default_period: RateLimitPeriod = Field(
        RateLimitPeriod.MINUTE,
        description="Default rate limit period"
    )
    per_ip_limits: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-IP rate limits (pattern -> limit)"
    )
    per_user_limits: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-user rate limits (pattern -> limit)"
    )
    endpoint_limits: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-endpoint rate limits (path -> limit)"
    )
    burst_multiplier: float = Field(
        1.5, ge=1.0,
        description="Burst allowance multiplier"
    )
    include_retry_after: bool = Field(
        True,
        description="Include Retry-After header in responses"
    )
    key_prefix: str = Field(
        "greenlang:ratelimit:",
        description="Redis key prefix"
    )
    enable_distributed: bool = Field(
        True,
        description="Enable distributed rate limiting via Redis"
    )


class TokenBucket:
    """
    Token bucket algorithm implementation for rate limiting.

    Attributes:
        capacity: Maximum number of tokens in bucket
        refill_rate: Tokens added per second
        tokens: Current number of tokens
        last_refill: Last refill timestamp
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum bucket capacity
            refill_rate: Token refill rate per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            Tuple of (success, wait_time_seconds)
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0

        # Calculate wait time for tokens to be available
        tokens_needed = tokens - self.tokens
        wait_time = tokens_needed / self.refill_rate

        return False, wait_time

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def reset(self):
        """Reset bucket to full capacity."""
        self.tokens = float(self.capacity)
        self.last_refill = time.time()


class RateLimiter:
    """
    Rate limiter implementation with Redis backend support.

    This class provides rate limiting using token bucket algorithm
    with support for distributed systems via Redis.

    Attributes:
        config: Rate limiting configuration
        redis: Redis client for distributed rate limiting
        local_buckets: Local token buckets for in-memory rate limiting

    Example:
        >>> limiter = RateLimiter(config)
        >>> @app.post("/api/data")
        >>> @limiter.limit("10/minute")
        >>> async def create_data(data: DataModel):
        >>>     return {"created": True}
    """

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter with configuration."""
        self.config = config
        self.redis: Optional[Redis] = None
        self.local_buckets: Dict[str, TokenBucket] = {}
        self._cleanup_interval = 300  # Clean local buckets every 5 minutes
        self._last_cleanup = time.time()

        # Initialize Redis if configured
        if config.enable_distributed and config.redis_url and REDIS_AVAILABLE:
            try:
                self.redis = aioredis.from_url(
                    config.redis_url,
                    decode_responses=True
                )
                logger.info("Redis-backed rate limiting initialized")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using local rate limiting.")
                self.redis = None
        else:
            logger.info("Using local in-memory rate limiting")

    def parse_rate_limit(self, limit_string: str) -> Tuple[int, float]:
        """
        Parse rate limit string.

        Args:
            limit_string: Rate limit string (e.g., "100/minute")

        Returns:
            Tuple of (limit, period_seconds)

        Raises:
            ValueError: If limit string is invalid
        """
        try:
            limit_str, period_str = limit_string.split("/")
            limit = int(limit_str)

            period_map = {
                "second": 1,
                "minute": 60,
                "hour": 3600,
                "day": 86400
            }

            period_seconds = period_map.get(period_str.lower())
            if not period_seconds:
                raise ValueError(f"Invalid period: {period_str}")

            return limit, float(period_seconds)

        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid rate limit format: {limit_string}") from e

    def _get_identifier(self, request: Request) -> str:
        """
        Get unique identifier for rate limiting.

        Args:
            request: FastAPI request object

        Returns:
            Unique identifier string
        """
        # Try to get user ID from request (if authenticated)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def _get_cache_key(self, identifier: str, endpoint: str) -> str:
        """
        Generate cache key for rate limit tracking.

        Args:
            identifier: User or IP identifier
            endpoint: API endpoint path

        Returns:
            Cache key string
        """
        # Create deterministic key
        key_data = f"{identifier}:{endpoint}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()[:8]
        return f"{self.config.key_prefix}{key_hash}"

    async def _check_redis_limit(
        self,
        key: str,
        limit: int,
        period: float
    ) -> Tuple[bool, int, float]:
        """
        Check rate limit using Redis.

        Args:
            key: Redis key for rate limiting
            limit: Request limit
            period: Period in seconds

        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        if not self.redis:
            return True, limit, 0

        try:
            # Use Redis pipeline for atomic operations
            async with self.redis.pipeline() as pipe:
                now = time.time()
                window_start = now - period

                # Remove old entries outside window
                await pipe.zremrangebyscore(key, 0, window_start)

                # Count requests in current window
                await pipe.zcard(key)

                # Add current request
                await pipe.zadd(key, {str(now): now})

                # Set expiry on key
                await pipe.expire(key, int(period))

                results = await pipe.execute()

                request_count = results[1]  # zcard result

                if request_count <= limit:
                    remaining = limit - request_count
                    reset_time = window_start + period
                    return True, remaining, reset_time
                else:
                    # Calculate when oldest request will expire
                    oldest = await self.redis.zrange(key, 0, 0, withscores=True)
                    if oldest:
                        reset_time = oldest[0][1] + period
                    else:
                        reset_time = now + period

                    return False, 0, reset_time

        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fall back to allowing request on Redis error
            return True, limit, 0

    def _check_local_limit(
        self,
        key: str,
        limit: int,
        period: float
    ) -> Tuple[bool, int, float]:
        """
        Check rate limit using local token bucket.

        Args:
            key: Bucket key
            limit: Request limit
            period: Period in seconds

        Returns:
            Tuple of (allowed, remaining, wait_time)
        """
        # Clean up old buckets periodically
        if time.time() - self._last_cleanup > self._cleanup_interval:
            self._cleanup_local_buckets()

        # Get or create bucket
        if key not in self.local_buckets:
            refill_rate = limit / period
            capacity = int(limit * self.config.burst_multiplier)
            self.local_buckets[key] = TokenBucket(capacity, refill_rate)

        bucket = self.local_buckets[key]
        allowed, wait_time = bucket.consume()

        remaining = int(bucket.tokens)
        return allowed, remaining, wait_time

    def _cleanup_local_buckets(self):
        """Remove unused local buckets to prevent memory leak."""
        # Simple cleanup: remove buckets not used recently
        # In production, would track last access time
        if len(self.local_buckets) > 1000:
            # Keep only 500 most recent buckets
            keys_to_remove = list(self.local_buckets.keys())[:-500]
            for key in keys_to_remove:
                del self.local_buckets[key]

        self._last_cleanup = time.time()

    def _get_limit_for_request(self, request: Request) -> str:
        """
        Get rate limit for specific request.

        Args:
            request: FastAPI request object

        Returns:
            Rate limit string
        """
        path = request.url.path

        # Check endpoint-specific limits
        for endpoint_pattern, limit in self.config.endpoint_limits.items():
            if path.startswith(endpoint_pattern):
                return limit

        # Check user-specific limits
        user_id = getattr(request.state, "user_id", None)
        if user_id and user_id in self.config.per_user_limits:
            return self.config.per_user_limits[user_id]

        # Check IP-specific limits
        if request.client:
            client_ip = request.client.host
            for ip_pattern, limit in self.config.per_ip_limits.items():
                if client_ip.startswith(ip_pattern):
                    return limit

        # Use default limit
        return f"{self.config.default_limit}/{self.config.default_period.value}"

    async def check_rate_limit(
        self,
        request: Request,
        limit_override: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limit.

        Args:
            request: FastAPI request object
            limit_override: Optional limit override

        Returns:
            Tuple of (allowed, metadata)
        """
        # Get rate limit for request
        limit_string = limit_override or self._get_limit_for_request(request)
        limit, period = self.parse_rate_limit(limit_string)

        # Get identifier and cache key
        identifier = self._get_identifier(request)
        endpoint = request.url.path
        cache_key = self._get_cache_key(identifier, endpoint)

        # Check rate limit
        if self.redis and self.config.enable_distributed:
            allowed, remaining, reset_time = await self._check_redis_limit(
                cache_key, limit, period
            )
        else:
            allowed, remaining, wait_time = self._check_local_limit(
                cache_key, limit, period
            )
            reset_time = time.time() + wait_time if not allowed else 0

        metadata = {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "retry_after": int(reset_time - time.time()) if not allowed else None
        }

        return allowed, metadata

    def limit(self, rate_limit: str):
        """
        Decorator for rate limiting endpoints.

        Args:
            rate_limit: Rate limit string (e.g., "10/minute")

        Example:
            >>> @limiter.limit("10/minute")
            >>> async def endpoint():
            >>>     return {"status": "ok"}
        """
        def decorator(func):
            async def wrapper(request: Request, *args, **kwargs):
                allowed, metadata = await self.check_rate_limit(
                    request,
                    limit_override=rate_limit
                )

                if not allowed:
                    # Add rate limit headers
                    headers = {
                        "X-RateLimit-Limit": str(metadata["limit"]),
                        "X-RateLimit-Remaining": str(metadata["remaining"]),
                        "X-RateLimit-Reset": str(int(metadata["reset"]))
                    }

                    if self.config.include_retry_after and metadata["retry_after"]:
                        headers["Retry-After"] = str(metadata["retry_after"])

                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Rate limit exceeded",
                        headers=headers
                    )

                # Add rate limit headers to response
                response = await func(request, *args, **kwargs)

                if isinstance(response, Response):
                    response.headers["X-RateLimit-Limit"] = str(metadata["limit"])
                    response.headers["X-RateLimit-Remaining"] = str(metadata["remaining"])
                    response.headers["X-RateLimit-Reset"] = str(int(metadata.get("reset", 0)))

                return response

            return wrapper
        return decorator


class RateLimitMiddleware:
    """
    FastAPI middleware for rate limiting.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(RateLimitMiddleware, config=rate_limit_config)
    """

    def __init__(self, app, config: RateLimitConfig):
        """Initialize rate limit middleware."""
        self.app = app
        self.limiter = RateLimiter(config)

    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        allowed, metadata = await self.limiter.check_rate_limit(request)

        # Add rate limit headers
        response = await call_next(request) if allowed else Response(
            content="Rate limit exceeded",
            status_code=429
        )

        response.headers["X-RateLimit-Limit"] = str(metadata["limit"])
        response.headers["X-RateLimit-Remaining"] = str(metadata["remaining"])
        response.headers["X-RateLimit-Reset"] = str(int(metadata.get("reset", 0)))

        if not allowed and self.limiter.config.include_retry_after:
            response.headers["Retry-After"] = str(metadata["retry_after"])

        return response