# -*- coding: utf-8 -*-
"""
FastAPI Dependency Injection for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module provides dependency injection for the FastAPI service, including:
- SchemaValidator: Main validation engine
- SchemaRegistry: Schema resolution
- IRCacheService: Compiled schema caching
- MetricsCollector: Prometheus-compatible metrics
- RateLimiter: Request rate limiting
- RequestContext: Request tracing and context

All dependencies are designed for thread-safety and can be shared across
concurrent requests.

Example:
    >>> from fastapi import Depends
    >>> from greenlang.schema.api.dependencies import get_validator
    >>>
    >>> @app.post("/validate")
    >>> async def validate(validator=Depends(get_validator)):
    ...     result = validator.validate(payload, schema_ref)

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.3
"""

from __future__ import annotations

import logging
import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Deque, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import Depends, Header, HTTPException, Request, status

from greenlang.schema.constants import (
    MAX_BATCH_ITEMS,
    MAX_PAYLOAD_BYTES,
    SCHEMA_CACHE_MAX_SIZE,
    SCHEMA_CACHE_TTL_SECONDS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SERVICE VERSION
# =============================================================================

SERVICE_VERSION = "1.0.0"


# =============================================================================
# API CONFIGURATION
# =============================================================================


@dataclass
class APIConfig:
    """
    Configuration for the Schema Validator API.

    Attributes:
        service_name: Name of the service.
        version: Service version.
        debug: Enable debug mode.
        rate_limit_requests: Max requests per window.
        rate_limit_window_seconds: Rate limit window size.
        max_payload_bytes: Maximum payload size.
        max_batch_items: Maximum batch items.
        cache_max_size: Maximum cache entries.
        cache_ttl_seconds: Cache TTL in seconds.
        require_api_key: Whether API key is required.
        allowed_api_keys: Set of allowed API keys.

    Example:
        >>> config = APIConfig()
        >>> config.is_rate_limit_enabled()
        True
    """

    service_name: str = "greenlang-schema-validator"
    version: str = SERVICE_VERSION
    debug: bool = False

    # Rate limiting
    rate_limit_requests: int = 1000
    rate_limit_window_seconds: int = 60

    # Size limits
    max_payload_bytes: int = MAX_PAYLOAD_BYTES
    max_batch_items: int = MAX_BATCH_ITEMS

    # Cache settings
    cache_max_size: int = SCHEMA_CACHE_MAX_SIZE
    cache_ttl_seconds: int = SCHEMA_CACHE_TTL_SECONDS

    # Authentication
    require_api_key: bool = False
    allowed_api_keys: frozenset = field(default_factory=frozenset)

    def is_rate_limit_enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self.rate_limit_requests > 0

    @classmethod
    def from_env(cls) -> "APIConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            GL_SCHEMA_API_DEBUG: Enable debug mode
            GL_SCHEMA_API_RATE_LIMIT: Max requests per minute
            GL_SCHEMA_API_REQUIRE_KEY: Require API key
            GL_SCHEMA_API_KEYS: Comma-separated API keys

        Returns:
            APIConfig instance
        """
        api_keys_str = os.environ.get("GL_SCHEMA_API_KEYS", "")
        api_keys = frozenset(
            k.strip() for k in api_keys_str.split(",") if k.strip()
        )

        return cls(
            debug=os.environ.get("GL_SCHEMA_API_DEBUG", "").lower() == "true",
            rate_limit_requests=int(
                os.environ.get("GL_SCHEMA_API_RATE_LIMIT", "1000")
            ),
            require_api_key=os.environ.get(
                "GL_SCHEMA_API_REQUIRE_KEY", ""
            ).lower() == "true",
            allowed_api_keys=api_keys,
        )


# =============================================================================
# REQUEST CONTEXT
# =============================================================================


@dataclass
class RequestContext:
    """
    Context for the current request.

    Provides request-scoped information including trace ID, timing,
    and authentication context.

    Attributes:
        trace_id: Unique request trace ID.
        start_time: Request start timestamp.
        api_key: API key used (if authenticated).
        client_ip: Client IP address.
        user_agent: Client user agent.

    Example:
        >>> context = RequestContext(trace_id="abc123")
        >>> context.elapsed_ms()
        15.5
    """

    trace_id: str = field(default_factory=lambda: str(uuid4()))
    start_time: float = field(default_factory=time.perf_counter)
    api_key: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None

    def elapsed_ms(self) -> float:
        """Get elapsed time since request start in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "elapsed_ms": self.elapsed_ms(),
            "client_ip": self.client_ip,
        }


# =============================================================================
# METRICS COLLECTOR
# =============================================================================


class MetricsCollector:
    """
    Thread-safe metrics collector for Prometheus-compatible metrics.

    Collects metrics for:
    - Validation requests (total, success, failed)
    - Batch validation (requests, items)
    - Cache performance (hits, misses, size)
    - Latency statistics (average, p95)

    Thread Safety:
        All methods use locks for thread-safe updates.

    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.record_validation(valid=True, latency_ms=15.0)
        >>> metrics.get_metrics()
        {'validations_total': 1, ...}
    """

    def __init__(self, latency_window_size: int = 1000):
        """
        Initialize metrics collector.

        Args:
            latency_window_size: Number of latency samples to keep for percentile calculation.
        """
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Counters
        self._validations_total = 0
        self._validations_success = 0
        self._validations_failed = 0
        self._batch_validations_total = 0
        self._batch_items_total = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_size = 0

        # Latency tracking (sliding window)
        self._latency_samples: Deque[float] = deque(maxlen=latency_window_size)
        self._latency_sum = 0.0

        logger.debug("MetricsCollector initialized")

    def record_validation(self, valid: bool, latency_ms: float) -> None:
        """
        Record a validation request.

        Args:
            valid: Whether validation passed.
            latency_ms: Validation latency in milliseconds.
        """
        with self._lock:
            self._validations_total += 1
            if valid:
                self._validations_success += 1
            else:
                self._validations_failed += 1

            # Update latency tracking
            if len(self._latency_samples) == self._latency_samples.maxlen:
                # Remove oldest sample from sum
                oldest = self._latency_samples[0]
                self._latency_sum -= oldest
            self._latency_samples.append(latency_ms)
            self._latency_sum += latency_ms

    def record_batch_validation(
        self, item_count: int, latency_ms: float
    ) -> None:
        """
        Record a batch validation request.

        Args:
            item_count: Number of items in batch.
            latency_ms: Total batch latency in milliseconds.
        """
        with self._lock:
            self._batch_validations_total += 1
            self._batch_items_total += item_count
            self._latency_samples.append(latency_ms)
            self._latency_sum += latency_ms

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._cache_misses += 1

    def update_cache_size(self, size: int) -> None:
        """
        Update current cache size.

        Args:
            size: Current number of cached entries.
        """
        with self._lock:
            self._cache_size = size

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics as dictionary.

        Returns:
            Dictionary with all metrics.
        """
        with self._lock:
            uptime = time.time() - self._start_time

            # Calculate averages
            sample_count = len(self._latency_samples)
            avg_latency = (
                self._latency_sum / sample_count if sample_count > 0 else 0.0
            )

            # Calculate p95 latency
            p95_latency = 0.0
            if sample_count > 0:
                sorted_samples = sorted(self._latency_samples)
                p95_index = int(sample_count * 0.95)
                p95_latency = sorted_samples[min(p95_index, sample_count - 1)]

            return {
                "validations_total": self._validations_total,
                "validations_success": self._validations_success,
                "validations_failed": self._validations_failed,
                "batch_validations_total": self._batch_validations_total,
                "batch_items_total": self._batch_items_total,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_size": self._cache_size,
                "avg_validation_time_ms": round(avg_latency, 2),
                "p95_validation_time_ms": round(p95_latency, 2),
                "uptime_seconds": round(uptime, 2),
            }

    def get_uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self._start_time


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """
    Thread-safe sliding window rate limiter.

    Implements a token bucket algorithm with sliding window for
    rate limiting API requests.

    Thread Safety:
        All methods use locks for thread-safe updates.

    Example:
        >>> limiter = RateLimiter(max_requests=100, window_seconds=60)
        >>> if limiter.is_allowed("client-ip"):
        ...     process_request()
        ... else:
        ...     return 429
    """

    def __init__(self, max_requests: int = 1000, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests per window.
            window_seconds: Window size in seconds.
        """
        self._lock = threading.Lock()
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: Dict[str, Deque[float]] = {}

        logger.debug(
            f"RateLimiter initialized: {max_requests} req/{window_seconds}s"
        )

    def is_allowed(self, client_key: str) -> Tuple[bool, int]:
        """
        Check if request is allowed for client.

        Args:
            client_key: Client identifier (IP, API key, etc.)

        Returns:
            Tuple of (allowed, remaining_requests)
        """
        now = time.time()
        cutoff = now - self._window_seconds

        with self._lock:
            # Get or create request history for client
            if client_key not in self._requests:
                self._requests[client_key] = deque()

            requests = self._requests[client_key]

            # Remove expired requests
            while requests and requests[0] < cutoff:
                requests.popleft()

            # Check limit
            if len(requests) >= self._max_requests:
                return False, 0

            # Record request
            requests.append(now)
            remaining = self._max_requests - len(requests)

            return True, remaining

    def get_retry_after(self, client_key: str) -> int:
        """
        Get seconds until rate limit resets.

        Args:
            client_key: Client identifier.

        Returns:
            Seconds until oldest request expires.
        """
        with self._lock:
            if client_key not in self._requests:
                return 0

            requests = self._requests[client_key]
            if not requests:
                return 0

            oldest = requests[0]
            retry_after = int(oldest + self._window_seconds - time.time())
            return max(0, retry_after)

    def cleanup(self) -> None:
        """Remove expired entries from all clients."""
        now = time.time()
        cutoff = now - self._window_seconds

        with self._lock:
            # Clean up each client's requests
            empty_clients = []
            for client_key, requests in self._requests.items():
                while requests and requests[0] < cutoff:
                    requests.popleft()
                if not requests:
                    empty_clients.append(client_key)

            # Remove clients with no recent requests
            for client_key in empty_clients:
                del self._requests[client_key]


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Global configuration (loaded from environment)
_config: Optional[APIConfig] = None

# Global metrics collector
_metrics: Optional[MetricsCollector] = None

# Global rate limiter
_rate_limiter: Optional[RateLimiter] = None

# Service start time
_start_time: float = time.time()


def _get_config() -> APIConfig:
    """Get or create global configuration."""
    global _config
    if _config is None:
        _config = APIConfig.from_env()
    return _config


def _get_metrics() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def _get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    config = _get_config()
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests,
            window_seconds=config.rate_limit_window_seconds,
        )
    return _rate_limiter


# =============================================================================
# FASTAPI DEPENDENCIES
# =============================================================================


async def get_config() -> APIConfig:
    """
    Get API configuration.

    Returns:
        APIConfig instance.
    """
    return _get_config()


async def get_metrics() -> MetricsCollector:
    """
    Get metrics collector.

    Returns:
        MetricsCollector instance.
    """
    return _get_metrics()


async def get_rate_limiter() -> RateLimiter:
    """
    Get rate limiter.

    Returns:
        RateLimiter instance.
    """
    return _get_rate_limiter()


async def get_request_context(
    request: Request,
    x_trace_id: Optional[str] = Header(None, alias="X-Trace-ID"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> RequestContext:
    """
    Create request context with tracing.

    Args:
        request: FastAPI request object.
        x_trace_id: Optional trace ID from header.
        x_api_key: Optional API key from header.

    Returns:
        RequestContext for this request.
    """
    trace_id = x_trace_id or str(uuid4())

    # Get client IP (handle proxies)
    client_ip = request.client.host if request.client else None
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    return RequestContext(
        trace_id=trace_id,
        api_key=x_api_key,
        client_ip=client_ip,
        user_agent=request.headers.get("User-Agent"),
    )


async def check_rate_limit(
    request: Request,
    context: RequestContext = Depends(get_request_context),
    config: APIConfig = Depends(get_config),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    """
    Check rate limit for request.

    Raises:
        HTTPException: 429 if rate limit exceeded.
    """
    if not config.is_rate_limit_enabled():
        return

    # Use client IP as rate limit key
    client_key = context.client_ip or "anonymous"

    allowed, remaining = limiter.is_allowed(client_key)

    if not allowed:
        retry_after = limiter.get_retry_after(client_key)
        logger.warning(
            f"Rate limit exceeded for {client_key} [{context.trace_id}]"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": "Too many requests",
                "retry_after": retry_after,
                "trace_id": context.trace_id,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Remaining": "0",
            },
        )


async def check_api_key(
    context: RequestContext = Depends(get_request_context),
    config: APIConfig = Depends(get_config),
) -> None:
    """
    Validate API key if required.

    Raises:
        HTTPException: 401 if API key is required but missing/invalid.
    """
    if not config.require_api_key:
        return

    if not context.api_key:
        logger.warning(f"Missing API key [{context.trace_id}]")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "unauthorized",
                "message": "API key required",
                "trace_id": context.trace_id,
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if context.api_key not in config.allowed_api_keys:
        logger.warning(f"Invalid API key [{context.trace_id}]")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "unauthorized",
                "message": "Invalid API key",
                "trace_id": context.trace_id,
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def get_validator():
    """
    Get schema validator instance.

    Returns:
        SchemaValidator instance for validation operations.

    Note:
        The validator is created on-demand and uses cached schemas
        for performance.
    """
    # Import here to avoid circular imports
    from greenlang.schema.validator.core import SchemaValidator

    # Return a validator instance
    # In production, this would be a singleton or use connection pooling
    return SchemaValidator()


async def get_compiler():
    """
    Get schema compiler instance.

    Returns:
        SchemaCompiler instance for compilation operations.
    """
    # Import here to avoid circular imports
    from greenlang.schema.compiler.compiler import SchemaCompiler

    return SchemaCompiler()


async def get_registry():
    """
    Get schema registry instance.

    Returns:
        SchemaRegistry instance for schema resolution.

    Note:
        Returns None if no registry is configured.
        The API will use inline schemas in this case.
    """
    # Import here to avoid circular imports
    try:
        from greenlang.schema.registry.resolver import SchemaRegistry

        # In production, this would be configured from environment
        registry_url = os.environ.get("GL_SCHEMA_REGISTRY_URL")
        if registry_url:
            # Return configured registry
            return SchemaRegistry(base_url=registry_url)
    except ImportError:
        pass

    return None


def get_uptime_seconds() -> float:
    """
    Get service uptime in seconds.

    Returns:
        Uptime in seconds since service start.
    """
    return time.time() - _start_time


def reset_metrics() -> None:
    """
    Reset all metrics.

    Used primarily for testing.
    """
    global _metrics
    _metrics = MetricsCollector()


def reset_rate_limiter() -> None:
    """
    Reset rate limiter.

    Used primarily for testing.
    """
    global _rate_limiter
    config = _get_config()
    _rate_limiter = RateLimiter(
        max_requests=config.rate_limit_requests,
        window_seconds=config.rate_limit_window_seconds,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Configuration
    "APIConfig",
    "SERVICE_VERSION",

    # Context
    "RequestContext",

    # Metrics
    "MetricsCollector",

    # Rate limiting
    "RateLimiter",

    # Dependencies
    "get_config",
    "get_metrics",
    "get_rate_limiter",
    "get_request_context",
    "check_rate_limit",
    "check_api_key",
    "get_validator",
    "get_compiler",
    "get_registry",
    "get_uptime_seconds",

    # Testing utilities
    "reset_metrics",
    "reset_rate_limiter",
]
