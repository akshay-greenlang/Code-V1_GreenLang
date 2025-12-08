"""
Token Bucket Rate Limiter for GreenLang

This module provides rate limiting functionality using
multiple rate limiting algorithms.

Features:
- Token bucket algorithm (burst-friendly)
- Sliding window rate limiting (accurate boundary behavior)
- Fixed window rate limiting (simple, memory-efficient)
- Leaky bucket rate limiting (smooth output rate)
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
import collections
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

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
    """
    Configuration for rate limiter.

    Attributes:
        strategy: Rate limiting algorithm to use
        scope: Scope for rate limiting (global, per-client, per-route, etc.)
        tokens_per_second: Token fill rate for token bucket and leak rate for leaky bucket
        bucket_size: Maximum capacity for token bucket and leaky bucket
        window_size_seconds: Window size for sliding and fixed window strategies
        max_requests: Maximum requests per window for sliding/fixed window strategies
        enable_headers: Whether to add rate limit headers to responses
        limit_header: Name of the limit header
        remaining_header: Name of the remaining header
        reset_header: Name of the reset header
        client_id_header: Header to extract client ID from
        use_ip_fallback: Whether to fall back to IP address for client ID
        redis_url: Redis URL for distributed rate limiting (enables Redis backend)
        redis_prefix: Prefix for Redis keys
        redis_db: Redis database number
        redis_pool_size: Maximum Redis connection pool size
        redis_timeout: Redis operation timeout in seconds
        redis_key_ttl: TTL for Redis keys in seconds
        fallback_to_memory: Whether to fall back to in-memory on Redis failure
    """
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    scope: RateLimitScope = RateLimitScope.PER_CLIENT

    # Token bucket / Leaky bucket settings
    tokens_per_second: float = 10.0
    bucket_size: int = 100

    # Window settings (sliding and fixed)
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

    # Redis settings (for distributed limiting)
    redis_url: Optional[str] = None
    redis_prefix: str = "ratelimit:"
    redis_db: int = 0
    redis_pool_size: int = 10
    redis_timeout: float = 5.0
    redis_key_ttl: int = 3600
    fallback_to_memory: bool = True


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


class FixedWindowCounter:
    """
    Fixed window rate limiter.

    Implements a simple fixed time window rate limiting algorithm.
    Windows are aligned to time boundaries (e.g., every minute, every hour).

    Advantages:
    - Memory efficient (single counter per window)
    - Simple to understand and implement
    - Predictable reset times

    Disadvantages:
    - Burst at window boundaries (up to 2x rate momentarily)

    Attributes:
        window_size: Window size in seconds
        max_requests: Maximum requests allowed per window
    """

    def __init__(
        self,
        window_size: int,
        max_requests: int
    ):
        """
        Initialize fixed window counter.

        Args:
            window_size: Window size in seconds (e.g., 60 for 1 minute)
            max_requests: Maximum requests per window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self._counter: int = 0
        self._window_start: float = self._get_window_start(time.monotonic())
        self._lock = asyncio.Lock()

    def _get_window_start(self, timestamp: float) -> float:
        """
        Get the start time of the window containing the timestamp.

        Args:
            timestamp: Monotonic timestamp

        Returns:
            Window start timestamp (aligned to window boundary)
        """
        return (timestamp // self.window_size) * self.window_size

    async def acquire(self, count: int = 1) -> bool:
        """
        Try to acquire request slots.

        Args:
            count: Number of request slots to acquire

        Returns:
            True if slots acquired, False if limit exceeded
        """
        async with self._lock:
            now = time.monotonic()
            current_window = self._get_window_start(now)

            # Check if we've moved to a new window
            if current_window != self._window_start:
                # Reset counter for new window
                self._window_start = current_window
                self._counter = 0

            # Check if we have capacity
            if self._counter + count <= self.max_requests:
                self._counter += count
                return True

            return False

    @property
    def remaining(self) -> int:
        """
        Get remaining request slots in current window.

        Returns:
            Number of remaining slots
        """
        now = time.monotonic()
        current_window = self._get_window_start(now)

        # If we're in a new window, full capacity available
        if current_window != self._window_start:
            return self.max_requests

        return max(0, self.max_requests - self._counter)

    @property
    def current_count(self) -> int:
        """
        Get current request count in window.

        Returns:
            Current count
        """
        now = time.monotonic()
        current_window = self._get_window_start(now)

        if current_window != self._window_start:
            return 0

        return self._counter

    @property
    def reset_at(self) -> float:
        """
        Get timestamp when current window resets.

        Returns:
            Monotonic timestamp of window reset
        """
        now = time.monotonic()
        current_window = self._get_window_start(now)
        return current_window + self.window_size

    @property
    def time_until_reset(self) -> float:
        """
        Get seconds until window reset.

        Returns:
            Seconds until reset
        """
        return max(0.0, self.reset_at - time.monotonic())


class LeakyBucket:
    """
    Leaky bucket rate limiter.

    Implements the leaky bucket algorithm for smooth, constant-rate
    request processing. Requests are queued and "leak" out at a
    constant rate.

    Advantages:
    - Smooth, constant output rate
    - Good for downstream systems with fixed processing capacity
    - Prevents bursts

    Disadvantages:
    - Adds latency (requests may wait in queue)
    - Queue can fill up during sustained high traffic

    Attributes:
        capacity: Maximum bucket (queue) capacity
        leak_rate: Requests leaked (processed) per second
    """

    def __init__(
        self,
        capacity: int,
        leak_rate: float
    ):
        """
        Initialize leaky bucket.

        Args:
            capacity: Maximum queue size (bucket capacity)
            leak_rate: Requests processed per second (leak rate)
        """
        if leak_rate <= 0:
            raise ValueError("leak_rate must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.leak_rate = leak_rate
        self._water_level: float = 0.0  # Current "water" in bucket
        self._last_leak: float = time.monotonic()
        self._lock = asyncio.Lock()

    def _leak(self) -> None:
        """
        Drain water from bucket based on elapsed time.

        Called internally before any bucket operation.
        """
        now = time.monotonic()
        elapsed = now - self._last_leak

        # Calculate how much water has leaked
        leaked = elapsed * self.leak_rate
        self._water_level = max(0.0, self._water_level - leaked)
        self._last_leak = now

    async def acquire(self, amount: float = 1.0) -> bool:
        """
        Try to add a request to the bucket.

        If there's room in the bucket, the request is accepted.
        If the bucket is full (overflow), the request is rejected.

        Args:
            amount: Amount of "water" to add (default 1.0 per request)

        Returns:
            True if request accepted, False if bucket overflowed
        """
        async with self._lock:
            self._leak()

            # Check if adding this request would overflow
            if self._water_level + amount <= self.capacity:
                self._water_level += amount
                return True

            return False

    @property
    def current_level(self) -> float:
        """
        Get current water level in bucket.

        Returns:
            Current level (0 to capacity)
        """
        # Calculate current level without modifying state
        now = time.monotonic()
        elapsed = now - self._last_leak
        leaked = elapsed * self.leak_rate
        return max(0.0, self._water_level - leaked)

    @property
    def available_capacity(self) -> float:
        """
        Get available capacity in bucket.

        Returns:
            Available capacity
        """
        return max(0.0, self.capacity - self.current_level)

    @property
    def remaining(self) -> int:
        """
        Get remaining request slots (integer).

        Returns:
            Number of remaining slots
        """
        return int(self.available_capacity)

    @property
    def time_until_available(self) -> float:
        """
        Get seconds until at least one slot is available.

        Returns:
            Seconds until a request can be accepted
        """
        current = self.current_level
        if current < self.capacity:
            return 0.0

        # Calculate time to drain enough for 1 request
        excess = current - (self.capacity - 1)
        return excess / self.leak_rate

    @property
    def queue_depth(self) -> int:
        """
        Get approximate queue depth (pending requests).

        Returns:
            Number of pending requests
        """
        return int(self.current_level)

    async def wait_and_acquire(
        self,
        amount: float = 1.0,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Wait for capacity and then acquire.

        This method will wait until there's room in the bucket,
        providing backpressure instead of immediate rejection.

        Args:
            amount: Amount to acquire
            timeout: Maximum seconds to wait (None for no timeout)

        Returns:
            True if acquired, False if timeout
        """
        start = time.monotonic()

        while True:
            if await self.acquire(amount):
                return True

            # Calculate wait time
            wait_time = self.time_until_available

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed + wait_time > timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)

            # Wait for some water to leak
            await asyncio.sleep(min(wait_time, 0.1))  # Max 100ms between checks


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
        self._token_buckets: Dict[str, TokenBucket] = {}
        self._sliding_windows: Dict[str, SlidingWindowCounter] = {}
        self._fixed_windows: Dict[str, FixedWindowCounter] = {}
        self._leaky_buckets: Dict[str, LeakyBucket] = {}
        self._lock = asyncio.Lock()

        # For backwards compatibility
        self._buckets = self._token_buckets
        self._windows = self._sliding_windows

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
        if key not in self._token_buckets:
            async with self._lock:
                if key not in self._token_buckets:
                    self._token_buckets[key] = TokenBucket(
                        capacity=self.config.bucket_size,
                        fill_rate=self.config.tokens_per_second
                    )
        return self._token_buckets[key]

    async def _get_or_create_window(self, key: str) -> SlidingWindowCounter:
        """Get or create a sliding window for a key."""
        if key not in self._sliding_windows:
            async with self._lock:
                if key not in self._sliding_windows:
                    self._sliding_windows[key] = SlidingWindowCounter(
                        window_size=self.config.window_size_seconds,
                        max_requests=self.config.max_requests
                    )
        return self._sliding_windows[key]

    async def _get_or_create_fixed_window(self, key: str) -> FixedWindowCounter:
        """Get or create a fixed window counter for a key."""
        if key not in self._fixed_windows:
            async with self._lock:
                if key not in self._fixed_windows:
                    self._fixed_windows[key] = FixedWindowCounter(
                        window_size=self.config.window_size_seconds,
                        max_requests=self.config.max_requests
                    )
        return self._fixed_windows[key]

    async def _get_or_create_leaky_bucket(self, key: str) -> LeakyBucket:
        """Get or create a leaky bucket for a key."""
        if key not in self._leaky_buckets:
            async with self._lock:
                if key not in self._leaky_buckets:
                    self._leaky_buckets[key] = LeakyBucket(
                        capacity=self.config.bucket_size,
                        leak_rate=self.config.tokens_per_second
                    )
        return self._leaky_buckets[key]

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
            # For sliding window, each request counts as one
            return await window.acquire()

        elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            fixed_window = await self._get_or_create_fixed_window(key)
            return await fixed_window.acquire(tokens)

        elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
            leaky_bucket = await self._get_or_create_leaky_bucket(key)
            return await leaky_bucket.acquire(float(tokens))

        # Unknown strategy - allow (fail open)
        logger.warning(f"Unknown rate limit strategy: {self.config.strategy}")
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
            reset_timestamp = window.reset_at
            # Handle monotonic time by converting to wall clock
            now_monotonic = time.monotonic()
            seconds_until_reset = max(0, reset_timestamp - now_monotonic)
            return RateLimitInfo(
                limit=window.max_requests,
                remaining=window.remaining,
                reset_at=datetime.utcnow() + timedelta(seconds=seconds_until_reset),
            )

        elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            fixed_window = await self._get_or_create_fixed_window(key)
            return RateLimitInfo(
                limit=fixed_window.max_requests,
                remaining=fixed_window.remaining,
                reset_at=datetime.utcnow() + timedelta(
                    seconds=fixed_window.time_until_reset
                ),
                retry_after=int(fixed_window.time_until_reset) + 1 if fixed_window.remaining <= 0 else None
            )

        elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
            leaky_bucket = await self._get_or_create_leaky_bucket(key)
            time_until_avail = leaky_bucket.time_until_available
            return RateLimitInfo(
                limit=leaky_bucket.capacity,
                remaining=leaky_bucket.remaining,
                reset_at=datetime.utcnow() + timedelta(
                    seconds=time_until_avail
                ),
                retry_after=int(time_until_avail) + 1 if leaky_bucket.remaining <= 0 else None
            )

        # Default fallback
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
            # Reset token bucket if exists
            if key in self._token_buckets:
                self._token_buckets[key] = TokenBucket(
                    capacity=self.config.bucket_size,
                    fill_rate=self.config.tokens_per_second
                )

            # Reset sliding window if exists
            if key in self._sliding_windows:
                self._sliding_windows[key] = SlidingWindowCounter(
                    window_size=self.config.window_size_seconds,
                    max_requests=self.config.max_requests
                )

            # Reset fixed window if exists
            if key in self._fixed_windows:
                self._fixed_windows[key] = FixedWindowCounter(
                    window_size=self.config.window_size_seconds,
                    max_requests=self.config.max_requests
                )

            # Reset leaky bucket if exists
            if key in self._leaky_buckets:
                self._leaky_buckets[key] = LeakyBucket(
                    capacity=self.config.bucket_size,
                    leak_rate=self.config.tokens_per_second
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
            "active_token_buckets": len(self._token_buckets),
            "active_sliding_windows": len(self._sliding_windows),
            "active_fixed_windows": len(self._fixed_windows),
            "active_leaky_buckets": len(self._leaky_buckets),
            # Backwards compatibility
            "active_buckets": len(self._token_buckets),
            "active_windows": len(self._sliding_windows),
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
