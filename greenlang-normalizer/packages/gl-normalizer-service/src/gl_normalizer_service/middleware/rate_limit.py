"""
Rate limiting middleware for GL Normalizer Service.

This module provides rate limiting middleware to protect the API from abuse
and ensure fair usage across tenants. Supports multiple rate limit strategies
and backends.

Rate Limit Strategies:
    - Fixed Window: Simple count per time window
    - Sliding Window: More accurate, rolling time window
    - Token Bucket: Allows bursts within limits

Backends:
    - Memory: In-process storage (single instance only)
    - Redis: Distributed rate limiting (production)

Usage:
    >>> from fastapi import FastAPI
    >>> from gl_normalizer_service.middleware.rate_limit import RateLimitMiddleware
    >>>
    >>> app = FastAPI()
    >>> app.add_middleware(
    ...     RateLimitMiddleware,
    ...     requests_per_window=100,
    ...     window_seconds=60
    ... )
"""

import time
from collections import defaultdict
from typing import Callable, Optional

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from gl_normalizer_service.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with configurable limits.

    Applies rate limits based on client identification (API key, user ID,
    or IP address). Supports per-endpoint and global limits.

    Attributes:
        requests_per_window: Maximum requests per time window
        window_seconds: Time window duration in seconds
        exempt_paths: Paths exempt from rate limiting
        backend: Rate limit storage backend

    Example:
        >>> app.add_middleware(
        ...     RateLimitMiddleware,
        ...     requests_per_window=100,
        ...     window_seconds=60,
        ...     exempt_paths={"/v1/health"}
        ... )
    """

    def __init__(
        self,
        app,
        requests_per_window: int = 100,
        window_seconds: int = 60,
        exempt_paths: Optional[set[str]] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: ASGI application
            requests_per_window: Max requests per window
            window_seconds: Window duration in seconds
            exempt_paths: Paths exempt from rate limiting
            settings: Application settings
        """
        super().__init__(app)
        self.settings = settings or get_settings()

        # Use settings if not explicitly provided
        self.requests_per_window = (
            requests_per_window
            if requests_per_window != 100
            else self.settings.rate_limit_requests
        )
        self.window_seconds = (
            window_seconds
            if window_seconds != 60
            else self.settings.rate_limit_window
        )

        self.exempt_paths = exempt_paths or {
            "/v1/health",
            "/v1/ready",
            "/v1/live",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

        # In-memory rate limit storage (use Redis in production)
        self._counters: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "window_start": 0}
        )

        # Per-endpoint limits (can be customized)
        self._endpoint_limits: dict[str, tuple[int, int]] = {
            "/v1/normalize": (100, 60),  # 100 req/min
            "/v1/normalize/batch": (20, 60),  # 20 req/min (heavier operation)
            "/v1/jobs": (50, 60),  # 50 req/min
        }

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request through rate limit middleware.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response from handler or rate limit error
        """
        # Check if rate limiting is enabled
        if not self.settings.rate_limit_enabled:
            return await call_next(request)

        path = request.url.path

        # Skip exempt paths
        if self._is_exempt(path):
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Get applicable limits
        limit, window = self._get_limits(path)

        # Check rate limit
        allowed, remaining, reset_at = self._check_rate_limit(
            client_id=client_id,
            path=path,
            limit=limit,
            window=window,
        )

        # Add rate limit headers to all responses
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(max(0, remaining)),
            "X-RateLimit-Reset": str(int(reset_at)),
            "X-RateLimit-Window": str(window),
        }

        if not allowed:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.warning(
                "rate_limit_exceeded",
                client_id=client_id[:20] if len(client_id) > 20 else client_id,
                path=path,
                limit=limit,
                window=window,
                request_id=request_id,
            )

            return JSONResponse(
                status_code=429,
                content={
                    "api_revision": self.settings.api_revision,
                    "error": {
                        "code": "GLNORM-008",
                        "message": f"Rate limit exceeded. Limit: {limit} requests per {window} seconds.",
                        "details": {
                            "limit": limit,
                            "window_seconds": window,
                            "retry_after": int(reset_at - time.time()),
                        },
                    },
                    "request_id": request_id,
                },
                headers={
                    **headers,
                    "Retry-After": str(int(reset_at - time.time())),
                },
            )

        # Process request and add headers to response
        response = await call_next(request)

        # Add rate limit headers
        for key, value in headers.items():
            response.headers[key] = value

        return response

    def _is_exempt(self, path: str) -> bool:
        """
        Check if path is exempt from rate limiting.

        Args:
            path: Request path

        Returns:
            True if exempt
        """
        return path in self.exempt_paths

    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.

        Priority:
        1. API key ID (if authenticated with API key)
        2. User ID (if authenticated with JWT)
        3. Tenant ID + IP (fallback)
        4. IP address (last resort)

        Args:
            request: HTTP request

        Returns:
            Client identifier string
        """
        # Check for authenticated user
        if hasattr(request.state, "user"):
            user = request.state.user
            if hasattr(user, "api_key_id") and user.api_key_id:
                return f"key:{user.api_key_id}"
            if hasattr(user, "id"):
                return f"user:{user.id}"
            if hasattr(user, "tenant_id"):
                return f"tenant:{user.tenant_id}:{self._get_client_ip(request)}"

        # Fall back to IP address
        return f"ip:{self._get_client_ip(request)}"

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.

        Handles X-Forwarded-For for proxied requests.

        Args:
            request: HTTP request

        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take first IP in chain
            return forwarded.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client
        if request.client:
            return request.client.host

        return "unknown"

    def _get_limits(self, path: str) -> tuple[int, int]:
        """
        Get rate limits for a path.

        Args:
            path: Request path

        Returns:
            Tuple of (requests_per_window, window_seconds)
        """
        # Check for endpoint-specific limits
        for endpoint, limits in self._endpoint_limits.items():
            if path.startswith(endpoint):
                return limits

        # Return default limits
        return self.requests_per_window, self.window_seconds

    def _check_rate_limit(
        self,
        client_id: str,
        path: str,
        limit: int,
        window: int,
    ) -> tuple[bool, int, float]:
        """
        Check if request is within rate limits (fixed window).

        Args:
            client_id: Client identifier
            path: Request path
            limit: Maximum requests
            window: Time window in seconds

        Returns:
            Tuple of (allowed, remaining, reset_timestamp)
        """
        now = time.time()
        key = f"{client_id}:{path}"

        counter = self._counters[key]

        # Reset window if expired
        if now - counter["window_start"] >= window:
            counter["count"] = 0
            counter["window_start"] = now

        # Calculate reset time
        reset_at = counter["window_start"] + window

        # Check limit
        if counter["count"] >= limit:
            return False, 0, reset_at

        # Increment counter
        counter["count"] += 1
        remaining = limit - counter["count"]

        return True, remaining, reset_at


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for more accurate limiting.

    Uses a sliding window algorithm that provides smoother rate limiting
    compared to fixed windows, avoiding the "thundering herd" problem
    at window boundaries.

    Example:
        >>> limiter = SlidingWindowRateLimiter(limit=100, window=60)
        >>> allowed = limiter.is_allowed("client_123")
    """

    def __init__(self, limit: int = 100, window: int = 60):
        """
        Initialize sliding window rate limiter.

        Args:
            limit: Maximum requests per window
            window: Window size in seconds
        """
        self.limit = limit
        self.window = window
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> tuple[bool, int, float]:
        """
        Check if request is allowed.

        Args:
            client_id: Client identifier

        Returns:
            Tuple of (allowed, remaining, reset_timestamp)
        """
        now = time.time()
        window_start = now - self.window

        # Get requests for this client
        requests = self._requests[client_id]

        # Remove expired requests
        requests[:] = [ts for ts in requests if ts > window_start]

        # Calculate remaining
        remaining = self.limit - len(requests)

        # Check limit
        if remaining <= 0:
            # Reset time is when oldest request expires
            reset_at = requests[0] + self.window if requests else now + self.window
            return False, 0, reset_at

        # Record this request
        requests.append(now)

        return True, remaining - 1, now + self.window


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for burst-friendly limiting.

    Allows short bursts of traffic while maintaining long-term limits.
    Tokens are added at a fixed rate and consumed per request.

    Example:
        >>> limiter = TokenBucketRateLimiter(capacity=100, refill_rate=1.67)
        >>> allowed = limiter.is_allowed("client_123")
    """

    def __init__(
        self,
        capacity: int = 100,
        refill_rate: float = 1.67,  # tokens per second (100/60)
    ):
        """
        Initialize token bucket rate limiter.

        Args:
            capacity: Maximum bucket capacity (burst size)
            refill_rate: Token refill rate per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: dict[str, dict] = defaultdict(
            lambda: {"tokens": capacity, "last_update": time.time()}
        )

    def is_allowed(self, client_id: str, tokens: int = 1) -> tuple[bool, int, float]:
        """
        Check if request is allowed and consume tokens.

        Args:
            client_id: Client identifier
            tokens: Number of tokens to consume (default 1)

        Returns:
            Tuple of (allowed, remaining, time_to_refill)
        """
        now = time.time()
        bucket = self._buckets[client_id]

        # Refill tokens based on time elapsed
        elapsed = now - bucket["last_update"]
        refill = elapsed * self.refill_rate
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + refill)
        bucket["last_update"] = now

        # Check if enough tokens
        if bucket["tokens"] < tokens:
            # Calculate time until enough tokens
            needed = tokens - bucket["tokens"]
            time_to_refill = needed / self.refill_rate
            return False, int(bucket["tokens"]), now + time_to_refill

        # Consume tokens
        bucket["tokens"] -= tokens
        return True, int(bucket["tokens"]), now


class TieredRateLimiter:
    """
    Tiered rate limiter with different limits per subscription tier.

    Applies different rate limits based on user's subscription tier
    or service level.

    Example:
        >>> limiter = TieredRateLimiter()
        >>> allowed = limiter.is_allowed("client_123", tier="enterprise")
    """

    # Default tier limits (requests per minute)
    DEFAULT_TIERS = {
        "free": (10, 60),  # 10 req/min
        "starter": (100, 60),  # 100 req/min
        "professional": (500, 60),  # 500 req/min
        "enterprise": (2000, 60),  # 2000 req/min
    }

    def __init__(self, tier_limits: Optional[dict[str, tuple[int, int]]] = None):
        """
        Initialize tiered rate limiter.

        Args:
            tier_limits: Dict of tier name to (limit, window) tuples
        """
        self.tier_limits = tier_limits or self.DEFAULT_TIERS
        self._limiters: dict[str, SlidingWindowRateLimiter] = {}

    def _get_limiter(self, tier: str) -> SlidingWindowRateLimiter:
        """Get or create limiter for tier."""
        if tier not in self._limiters:
            limit, window = self.tier_limits.get(tier, self.tier_limits["free"])
            self._limiters[tier] = SlidingWindowRateLimiter(limit=limit, window=window)
        return self._limiters[tier]

    def is_allowed(
        self, client_id: str, tier: str = "free"
    ) -> tuple[bool, int, float, str]:
        """
        Check if request is allowed for tier.

        Args:
            client_id: Client identifier
            tier: Subscription tier

        Returns:
            Tuple of (allowed, remaining, reset_timestamp, tier)
        """
        limiter = self._get_limiter(tier)
        allowed, remaining, reset_at = limiter.is_allowed(client_id)
        return allowed, remaining, reset_at, tier
