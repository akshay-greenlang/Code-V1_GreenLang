# -*- coding: utf-8 -*-
"""
Enterprise Rate Limiting Middleware for GreenLang

This module provides comprehensive rate limiting capabilities with
tenant-aware quotas, distributed rate limiting, and multiple strategies.
"""

import time
import hashlib
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
from functools import wraps
from greenlang.determinism import deterministic_random


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""

    requests_per_second: Optional[float] = None
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    tenant_specific: bool = True
    endpoint_specific: bool = True

    def get_limit(self, window: str = "second") -> Optional[float]:
        """Get limit for specific time window"""
        if window == "second" and self.requests_per_second:
            return self.requests_per_second
        elif window == "minute" and self.requests_per_minute:
            return self.requests_per_minute / 60.0
        elif window == "hour" and self.requests_per_hour:
            return self.requests_per_hour / 3600.0
        elif window == "day" and self.requests_per_day:
            return self.requests_per_day / 86400.0
        return None


@dataclass
class RateLimitStatus:
    """Rate limit status for a client"""

    allowed: bool
    limit: int
    remaining: int
    reset_at: datetime
    retry_after: Optional[int] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers"""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_at.timestamp())),
        }
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
        return headers


class RateLimiter:
    """
    Base rate limiter class with multiple strategy support.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.tokens: Dict[str, float] = defaultdict(lambda: config.burst_size)
        self.last_update: Dict[str, float] = defaultdict(time.time)
        self.lock = threading.Lock()

    def get_client_id(
        self,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> str:
        """Generate unique client identifier"""
        parts = []

        if self.config.tenant_specific and tenant_id:
            parts.append(f"tenant:{tenant_id}")

        if user_id:
            parts.append(f"user:{user_id}")
        elif ip_address:
            parts.append(f"ip:{ip_address}")

        if self.config.endpoint_specific and endpoint:
            parts.append(f"endpoint:{endpoint}")

        client_id = ":".join(parts) if parts else "global"
        return hashlib.md5(client_id.encode()).hexdigest()

    def check_rate_limit(self, client_id: str) -> RateLimitStatus:
        """Check if request is within rate limit"""
        if self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._check_fixed_window(client_id)
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(client_id)
        elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(client_id)
        elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._check_leaky_bucket(client_id)
        elif self.config.strategy == RateLimitStrategy.ADAPTIVE:
            return self._check_adaptive(client_id)
        else:
            return RateLimitStatus(
                allowed=True, limit=0, remaining=0, reset_at=DeterministicClock.utcnow()
            )

    def _check_fixed_window(self, client_id: str) -> RateLimitStatus:
        """Fixed window rate limiting"""
        with self.lock:
            now = time.time()
            window_start = int(now / 60) * 60  # 1-minute windows

            # Clean old requests
            self.requests[client_id] = deque(
                [ts for ts in self.requests[client_id] if ts >= window_start]
            )

            limit = self.config.requests_per_minute or 60
            current_count = len(self.requests[client_id])

            if current_count < limit:
                self.requests[client_id].append(now)
                return RateLimitStatus(
                    allowed=True,
                    limit=limit,
                    remaining=limit - current_count - 1,
                    reset_at=datetime.fromtimestamp(window_start + 60),
                )
            else:
                return RateLimitStatus(
                    allowed=False,
                    limit=limit,
                    remaining=0,
                    reset_at=datetime.fromtimestamp(window_start + 60),
                    retry_after=int(window_start + 60 - now),
                )

    def _check_sliding_window(self, client_id: str) -> RateLimitStatus:
        """Sliding window rate limiting"""
        with self.lock:
            now = time.time()
            window_size = 60.0  # 1-minute window

            # Remove requests outside the window
            cutoff = now - window_size
            self.requests[client_id] = deque(
                [ts for ts in self.requests[client_id] if ts > cutoff]
            )

            limit = self.config.requests_per_minute or 60
            current_count = len(self.requests[client_id])

            if current_count < limit:
                self.requests[client_id].append(now)
                return RateLimitStatus(
                    allowed=True,
                    limit=limit,
                    remaining=limit - current_count - 1,
                    reset_at=datetime.fromtimestamp(now + window_size),
                )
            else:
                oldest_request = min(self.requests[client_id])
                retry_after = int(oldest_request + window_size - now)
                return RateLimitStatus(
                    allowed=False,
                    limit=limit,
                    remaining=0,
                    reset_at=datetime.fromtimestamp(oldest_request + window_size),
                    retry_after=retry_after,
                )

    def _check_token_bucket(self, client_id: str) -> RateLimitStatus:
        """Token bucket rate limiting"""
        with self.lock:
            now = time.time()
            rate = self.config.requests_per_second or 1.0
            capacity = self.config.burst_size

            # Refill tokens based on time elapsed
            time_passed = now - self.last_update[client_id]
            self.tokens[client_id] = min(
                capacity, self.tokens[client_id] + time_passed * rate
            )
            self.last_update[client_id] = now

            if self.tokens[client_id] >= 1:
                self.tokens[client_id] -= 1
                return RateLimitStatus(
                    allowed=True,
                    limit=capacity,
                    remaining=int(self.tokens[client_id]),
                    reset_at=datetime.fromtimestamp(
                        now + (capacity - self.tokens[client_id]) / rate
                    ),
                )
            else:
                retry_after = int((1 - self.tokens[client_id]) / rate)
                return RateLimitStatus(
                    allowed=False,
                    limit=capacity,
                    remaining=0,
                    reset_at=datetime.fromtimestamp(now + retry_after),
                    retry_after=retry_after,
                )

    def _check_leaky_bucket(self, client_id: str) -> RateLimitStatus:
        """Leaky bucket rate limiting"""
        with self.lock:
            now = time.time()
            rate = self.config.requests_per_second or 1.0
            capacity = self.config.burst_size

            # Leak tokens based on time elapsed
            time_passed = now - self.last_update[client_id]
            leaked = time_passed * rate

            # Update bucket level
            current_level = max(0, len(self.requests[client_id]) - leaked)

            if current_level < capacity:
                self.requests[client_id].append(now)
                self.last_update[client_id] = now
                return RateLimitStatus(
                    allowed=True,
                    limit=capacity,
                    remaining=int(capacity - current_level - 1),
                    reset_at=datetime.fromtimestamp(now + (current_level + 1) / rate),
                )
            else:
                retry_after = int((current_level - capacity + 1) / rate)
                return RateLimitStatus(
                    allowed=False,
                    limit=capacity,
                    remaining=0,
                    reset_at=datetime.fromtimestamp(now + retry_after),
                    retry_after=retry_after,
                )

    def _check_adaptive(self, client_id: str) -> RateLimitStatus:
        """Adaptive rate limiting based on system load"""
        # Start with token bucket and adjust based on system metrics
        base_status = self._check_token_bucket(client_id)

        # Adjust limits based on system load (placeholder for actual metrics)
        system_load = self._get_system_load()
        if system_load > 0.8:  # High load
            base_status.limit = int(base_status.limit * 0.5)
            base_status.remaining = min(base_status.remaining, base_status.limit)

        return base_status

    def _get_system_load(self) -> float:
        """Get current system load (placeholder)"""
        # In production, this would check actual system metrics
        import random

        return deterministic_random().random()


class TenantRateLimiter(RateLimiter):
    """
    Tenant-aware rate limiter with different limits per subscription tier.
    """

    def __init__(self):
        # Default config for unknown tenants
        super().__init__(
            RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            )
        )

        # Tier-specific configurations
        self.tier_configs = {
            "FREE": RateLimitConfig(
                requests_per_minute=20,
                requests_per_hour=500,
                requests_per_day=5000,
                burst_size=5,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
            ),
            "STARTER": RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=2000,
                requests_per_day=20000,
                burst_size=10,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "PROFESSIONAL": RateLimitConfig(
                requests_per_minute=200,
                requests_per_hour=10000,
                requests_per_day=100000,
                burst_size=50,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "ENTERPRISE": RateLimitConfig(
                requests_per_minute=1000,
                requests_per_hour=50000,
                requests_per_day=1000000,
                burst_size=200,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
            "CUSTOM": RateLimitConfig(
                requests_per_minute=None,  # No limits
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            ),
        }

    def get_tenant_config(
        self, tenant_id: str, subscription_tier: str
    ) -> RateLimitConfig:
        """Get rate limit config for tenant"""
        return self.tier_configs.get(subscription_tier, self.config)

    def check_tenant_limit(
        self, tenant_id: str, subscription_tier: str, endpoint: Optional[str] = None
    ) -> RateLimitStatus:
        """Check rate limit for specific tenant"""
        config = self.get_tenant_config(tenant_id, subscription_tier)

        # No limits for custom tier
        if subscription_tier == "CUSTOM":
            return RateLimitStatus(
                allowed=True,
                limit=999999,
                remaining=999999,
                reset_at=DeterministicClock.utcnow() + timedelta(hours=1),
            )

        # Use tenant-specific config
        self.config = config
        client_id = self.get_client_id(tenant_id=tenant_id, endpoint=endpoint)
        return self.check_rate_limit(client_id)


class RateLimitMiddleware:
    """
    Rate limiting middleware for web applications.
    """

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter

    def __call__(self, request_handler: Callable) -> Callable:
        """Wrap request handler with rate limiting"""

        @wraps(request_handler)
        async def wrapped_handler(request, *args, **kwargs):
            # Extract client information from request
            tenant_id = request.headers.get("X-Tenant-ID")
            user_id = request.headers.get("X-User-ID")
            ip_address = request.remote_addr
            endpoint = request.path

            # Check rate limit
            client_id = self.rate_limiter.get_client_id(
                tenant_id=tenant_id,
                user_id=user_id,
                ip_address=ip_address,
                endpoint=endpoint,
            )

            status = self.rate_limiter.check_rate_limit(client_id)

            # Add rate limit headers to response
            request.rate_limit_headers = status.to_headers()

            if not status.allowed:
                # Rate limit exceeded
                return (
                    {"error": "Rate limit exceeded", "retry_after": status.retry_after},
                    429,
                    status.to_headers(),
                )

            # Process request
            return await request_handler(request, *args, **kwargs)

        return wrapped_handler


def rate_limit(
    requests_per_minute: int = 60,
    requests_per_hour: Optional[int] = None,
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
):
    """
    Decorator for rate limiting functions.

    Args:
        requests_per_minute: Maximum requests per minute
        requests_per_hour: Maximum requests per hour
        strategy: Rate limiting strategy to use
    """
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        strategy=strategy,
    )
    limiter = RateLimiter(config)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get client identifier (simplified for decorator)
            client_id = "default"

            status = limiter.check_rate_limit(client_id)
            if not status.allowed:
                raise Exception(
                    f"Rate limit exceeded. Retry after {status.retry_after} seconds"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Global rate limiter instance
_global_rate_limiter: Optional[TenantRateLimiter] = None


def get_rate_limiter() -> TenantRateLimiter:
    """Get global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = TenantRateLimiter()
    return _global_rate_limiter
