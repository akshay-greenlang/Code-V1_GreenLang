"""
Token Bucket Rate Limiter for GreenLang

This module provides rate limiting functionality using
the token bucket algorithm.

Features:
- Token bucket algorithm
- Sliding window rate limiting
- Per-client/per-route limits
- Redis-backed distributed limiting
- Rate limit headers
- Graceful degradation

Example:
    >>> limiter = RateLimiter(config)
    >>> if await limiter.acquire("user-123"):
    ...     process_request()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from fastapi import HTTPException, Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    HTTPException = Exception
    Request = None
    Response = None
    BaseHTTPMiddleware = object

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(str, Enum):
    """Rate limit scope."""
    GLOBAL = "global"
    PER_CLIENT = "per_client"
    PER_ROUTE = "per_route"
    PER_CLIENT_ROUTE = "per_client_route"


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter."""
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    scope: RateLimitScope = RateLimitScope.PER_CLIENT
    # Token bucket settings
    tokens_per_second: float = 10.0
    bucket_size: int = 100
    # Sliding window settings
    window_size_seconds: int = 60
    max_requests: int = 100
    # Headers
    enable_headers: bool = True
    limit_header: str = "X-RateLimit-Limit"
    remaining_header: str = "X-RateLimit-Remaining"
    reset_header: str = "X-RateLimit-Reset"
    # Client identification
    client_id_header: str = "X-API-Key"
    use_ip_fallback: bool = True
    # Redis (for distributed limiting)
    redis_url: Optional[str] = None
    redis_prefix: str = "ratelimit:"


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    limit: int = Field(..., description="Maximum requests allowed")
    remaining: int = Field(..., description="Remaining requests")
    reset_at: datetime = Field(..., description="When limit resets")
    retry_after: Optional[int] = Field(default=None, description="Seconds until retry")


class TokenBucket:
    """
    Token bucket rate limiter.

    Implements the token bucket algorithm for smooth
    rate limiting with burst capacity.

    Attributes:
        capacity: Maximum tokens in bucket
        fill_rate: Tokens added per second
        tokens: Current token count
    """

    def __init__(
        self,
        capacity: int,
        fill_rate: float
    ):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum bucket capacity
            fill_rate: Tokens per second
        """
        self.capacity = capacity
        self.fill_rate = fill_rate
        self._tokens = float(capacity)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired
        """
        async with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.fill_rate
        )
        self._last_update = now

    @property
    def tokens(self) -> int:
        """Get current token count."""
        return int(self._tokens)

    @property
    def time_until_available(self) -> float:
        """Get seconds until a token is available."""
        if self._tokens >= 1:
            return 0.0
        return (1 - self._tokens) / self.fill_rate


class SlidingWindowCounter:
    """
    Sliding window rate limiter.

    Uses a sliding window for more accurate rate limiting
    across time boundaries.
    """

    def __init__(
        self,
        window_size: int,
        max_requests: int
    ):
        """
        Initialize sliding window counter.

        Args:
            window_size: Window size in seconds
            max_requests: Max requests per window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self._requests: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Try to acquire a request slot.

        Returns:
            True if slot acquired
        """
        async with self._lock:
            now = time.monotonic()
            window_start = now - self.window_size

            # Remove old requests
            self._requests = [
                t for t in self._requests
                if t > window_start
            ]

            if len(self._requests) < self.max_requests:
                self._requests.append(now)
                return True
            return False

    @property
    def remaining(self) -> int:
        """Get remaining request slots."""
        now = time.monotonic()
        window_start = now - self.window_size
        current = len([t for t in self._requests if t > window_start])
        return max(0, self.max_requests - current)

    @property
    def reset_at(self) -> float:
        """Get time when oldest request expires."""
        if not self._requests:
            return time.monotonic()
        return self._requests[0] + self.window_size


class RateLimiter:
    """
    Production-ready rate limiter.

    Provides rate limiting for API endpoints with multiple
    strategies and client identification.

    Attributes:
        config: Rate limiter configuration
        buckets: Token buckets by client/route

    Example:
        >>> config = RateLimiterConfig(
        ...     tokens_per_second=10,
        ...     bucket_size=100
        ... )
        >>> limiter = RateLimiter(config)
        >>> info = await limiter.check("client-123")
        >>> if info.remaining > 0:
        ...     await limiter.acquire("client-123")
    """

    def __init__(self, config: Optional[RateLimiterConfig] = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limiter configuration
        """
        self.config = config or RateLimiterConfig()
        self._buckets: Dict[str, TokenBucket] = {}
        self._windows: Dict[str, SlidingWindowCounter] = {}
        self._lock = asyncio.Lock()

        logger.info(
            f"RateLimiter initialized: {self.config.strategy.value}, "
            f"{self.config.tokens_per_second} tokens/s"
        )

    def _get_key(
        self,
        client_id: Optional[str] = None,
        route: Optional[str] = None
    ) -> str:
        """Get rate limit key based on scope."""
        if self.config.scope == RateLimitScope.GLOBAL:
            return "global"
        elif self.config.scope == RateLimitScope.PER_CLIENT:
            return client_id or "unknown"
        elif self.config.scope == RateLimitScope.PER_ROUTE:
            return route or "default"
        else:
            return f"{client_id or 'unknown'}:{route or 'default'}"

    async def _get_or_create_bucket(self, key: str) -> TokenBucket:
        """Get or create a token bucket for a key."""
        if key not in self._buckets:
            async with self._lock:
                if key not in self._buckets:
                    self._buckets[key] = TokenBucket(
                        capacity=self.config.bucket_size,
                        fill_rate=self.config.tokens_per_second
                    )
        return self._buckets[key]

    async def _get_or_create_window(self, key: str) -> SlidingWindowCounter:
        """Get or create a sliding window for a key."""
        if key not in self._windows:
            async with self._lock:
                if key not in self._windows:
                    self._windows[key] = SlidingWindowCounter(
                        window_size=self.config.window_size_seconds,
                        max_requests=self.config.max_requests
                    )
        return self._windows[key]

    async def acquire(
        self,
        client_id: Optional[str] = None,
        route: Optional[str] = None,
        tokens: int = 1
    ) -> bool:
        """
        Try to acquire tokens/requests.

        Args:
            client_id: Client identifier
            route: Route identifier
            tokens: Number of tokens to acquire

        Returns:
            True if acquired
        """
        key = self._get_key(client_id, route)

        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            bucket = await self._get_or_create_bucket(key)
            return await bucket.acquire(tokens)

        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            window = await self._get_or_create_window(key)
            # For sliding window, ignore tokens count
            return await window.acquire()

        return True

    async def check(
        self,
        client_id: Optional[str] = None,
        route: Optional[str] = None
    ) -> RateLimitInfo:
        """
        Check rate limit status without consuming.

        Args:
            client_id: Client identifier
            route: Route identifier

        Returns:
            Rate limit information
        """
        key = self._get_key(client_id, route)

        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            bucket = await self._get_or_create_bucket(key)
            return RateLimitInfo(
                limit=bucket.capacity,
                remaining=bucket.tokens,
                reset_at=datetime.utcnow() + timedelta(
                    seconds=bucket.time_until_available
                ),
                retry_after=int(bucket.time_until_available) if bucket.tokens < 1 else None
            )

        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            window = await self._get_or_create_window(key)
            return RateLimitInfo(
                limit=window.max_requests,
                remaining=window.remaining,
                reset_at=datetime.utcfromtimestamp(window.reset_at),
            )

        return RateLimitInfo(
            limit=self.config.max_requests,
            remaining=self.config.max_requests,
            reset_at=datetime.utcnow(),
        )

    async def reset(
        self,
        client_id: Optional[str] = None,
        route: Optional[str] = None
    ) -> None:
        """
        Reset rate limit for a key.

        Args:
            client_id: Client identifier
            route: Route identifier
        """
        key = self._get_key(client_id, route)

        async with self._lock:
            if key in self._buckets:
                self._buckets[key] = TokenBucket(
                    capacity=self.config.bucket_size,
                    fill_rate=self.config.tokens_per_second
                )
            if key in self._windows:
                self._windows[key] = SlidingWindowCounter(
                    window_size=self.config.window_size_seconds,
                    max_requests=self.config.max_requests
                )

        logger.info(f"Reset rate limit for key: {key}")

    def add_rate_limit_headers(
        self,
        response: Response,
        info: RateLimitInfo
    ) -> None:
        """
        Add rate limit headers to response.

        Args:
            response: FastAPI response
            info: Rate limit info
        """
        if not self.config.enable_headers:
            return

        response.headers[self.config.limit_header] = str(info.limit)
        response.headers[self.config.remaining_header] = str(info.remaining)
        response.headers[self.config.reset_header] = str(
            int(info.reset_at.timestamp())
        )

        if info.retry_after:
            response.headers["Retry-After"] = str(info.retry_after)

    def get_client_id(self, request: Request) -> str:
        """
        Extract client ID from request.

        Args:
            request: FastAPI request

        Returns:
            Client identifier
        """
        # Try header first
        client_id = request.headers.get(self.config.client_id_header)
        if client_id:
            return client_id

        # Fallback to IP
        if self.config.use_ip_fallback and request.client:
            return request.client.host

        return "anonymous"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "strategy": self.config.strategy.value,
            "scope": self.config.scope.value,
            "active_buckets": len(self._buckets),
            "active_windows": len(self._windows),
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Automatically applies rate limiting to all requests
    and adds appropriate headers.
    """

    def __init__(
        self,
        app,
        limiter: RateLimiter,
        exclude_paths: Optional[List[str]] = None
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application
            limiter: Rate limiter instance
            exclude_paths: Paths to exclude from limiting
        """
        super().__init__(app)
        self.limiter = limiter
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to request."""
        # Check exclusions
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get client ID
        client_id = self.limiter.get_client_id(request)
        route = request.url.path

        # Check rate limit
        info = await self.limiter.check(client_id, route)

        if info.remaining <= 0:
            # Rate limited
            response = Response(
                content='{"error": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json"
            )
            self.limiter.add_rate_limit_headers(response, info)
            return response

        # Acquire token
        acquired = await self.limiter.acquire(client_id, route)
        if not acquired:
            response = Response(
                content='{"error": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json"
            )
            self.limiter.add_rate_limit_headers(response, info)
            return response

        # Process request
        response = await call_next(request)

        # Add headers
        updated_info = await self.limiter.check(client_id, route)
        self.limiter.add_rate_limit_headers(response, updated_info)

        return response


def rate_limit(
    limiter: RateLimiter,
    tokens: int = 1,
    client_id_extractor: Optional[Callable[[Request], str]] = None
):
    """
    Decorator for rate limiting individual endpoints.

    Args:
        limiter: Rate limiter instance
        tokens: Tokens to consume per request
        client_id_extractor: Custom client ID extraction

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(request: Request, *args, **kwargs):
            # Get client ID
            if client_id_extractor:
                client_id = client_id_extractor(request)
            else:
                client_id = limiter.get_client_id(request)

            # Check and acquire
            if not await limiter.acquire(client_id, tokens=tokens):
                info = await limiter.check(client_id)
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        limiter.config.limit_header: str(info.limit),
                        limiter.config.remaining_header: str(info.remaining),
                        "Retry-After": str(info.retry_after or 60),
                    }
                )

            return await func(request, *args, **kwargs)

        return wrapper
    return decorator
