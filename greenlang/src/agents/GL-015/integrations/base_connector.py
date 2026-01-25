"""
Base Connector Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides abstract base class with common functionality for all insulation inspection
system connectors including connection pooling (aiohttp), retry logic with
exponential backoff, rate limiting, circuit breaker pattern, health monitoring,
caching, authentication management, and error handling.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import asyncio
import functools
import hashlib
import json
import logging
import random
import time
import uuid
from collections import deque

import aiohttp
from pydantic import BaseModel, Field, ConfigDict, field_validator

# Configure module logger
logger = logging.getLogger(__name__)

# Generic type for cached values
T = TypeVar("T")


# =============================================================================
# Enumerations
# =============================================================================


class ConnectionState(str, Enum):
    """Connection state enumeration for insulation inspection system connectors."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


class CircuitState(str, Enum):
    """Circuit breaker state enumeration."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing state, requests rejected immediately
    HALF_OPEN = "half_open"  # Testing state, limited requests allowed


class HealthStatus(str, Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class ConnectorType(str, Enum):
    """Types of connectors supported by GL-015 INSULSCAN."""

    THERMAL_CAMERA = "thermal_camera"  # FLIR, Fluke, Testo, Optris, InfraTec
    CMMS = "cmms"  # SAP PM, IBM Maximo, Oracle EAM
    ASSET_MANAGEMENT = "asset_management"  # Equipment registry, inventory
    WEATHER_SERVICE = "weather_service"  # OpenWeatherMap, NOAA
    AGENT_COORDINATOR = "agent_coordinator"  # GL-001, GL-006, GL-014
    REST_API = "rest_api"  # Generic REST API
    DATABASE = "database"  # SQL/NoSQL databases
    FILE_STORAGE = "file_storage"  # Image storage, NAS


class DataQualityLevel(str, Enum):
    """Data quality classification levels."""

    EXCELLENT = "excellent"  # 95-100% quality score
    GOOD = "good"  # 80-94% quality score
    ACCEPTABLE = "acceptable"  # 60-79% quality score
    POOR = "poor"  # 40-59% quality score
    UNACCEPTABLE = "unacceptable"  # < 40% quality score


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class AuthenticationType(str, Enum):
    """Authentication types supported by connectors."""

    NONE = "none"
    BASIC = "basic"
    BEARER_TOKEN = "bearer_token"
    API_KEY = "api_key"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
    OAUTH2_PASSWORD = "oauth2_password"
    CERTIFICATE = "certificate"
    DIGEST = "digest"
    CUSTOM = "custom"


class ImageFormat(str, Enum):
    """Supported thermal image formats."""

    JPEG = "jpeg"
    TIFF = "tiff"
    PNG = "png"
    RAW = "raw"
    RADIOMETRIC_JPEG = "radiometric_jpeg"  # FLIR radiometric JPEG
    SEQ = "seq"  # FLIR sequence format
    FFF = "fff"  # FLIR file format
    IS2 = "is2"  # InfraTec format
    TMX = "tmx"  # Testo format


# =============================================================================
# Custom Exceptions
# =============================================================================


class ConnectorError(Exception):
    """Base exception for connector errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class ConnectionError(ConnectorError):
    """Connection-related errors."""
    pass


class AuthenticationError(ConnectorError):
    """Authentication-related errors."""
    pass


class TimeoutError(ConnectorError):
    """Timeout-related errors."""
    pass


class ValidationError(ConnectorError):
    """Data validation errors."""
    pass


class CircuitOpenError(ConnectorError):
    """Circuit breaker is open."""
    pass


class RetryExhaustedError(ConnectorError):
    """All retry attempts exhausted."""
    pass


class ConfigurationError(ConnectorError):
    """Configuration-related errors."""
    pass


class DataQualityError(ConnectorError):
    """Data quality threshold not met."""
    pass


class RateLimitError(ConnectorError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.retry_after_seconds = retry_after_seconds


class ProtocolError(ConnectorError):
    """Protocol-specific errors."""
    pass


class ImageProcessingError(ConnectorError):
    """Thermal image processing errors."""
    pass


class CameraConnectionError(ConnectorError):
    """Thermal camera connection errors."""
    pass


class CalibrationError(ConnectorError):
    """Camera calibration errors."""
    pass


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class BaseConnectorConfig(BaseModel):
    """Base configuration for all insulation inspection system connectors."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    connector_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the connector instance"
    )
    connector_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name for the connector"
    )
    connector_type: ConnectorType = Field(
        ...,
        description="Type of connector"
    )
    enabled: bool = Field(
        default=True,
        description="Whether the connector is enabled"
    )
    description: str = Field(
        default="",
        max_length=1000,
        description="Description of the connector"
    )

    # Connection settings
    connection_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Connection timeout in seconds"
    )
    read_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Read timeout in seconds"
    )
    write_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Write timeout in seconds"
    )

    # Retry settings
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts"
    )
    retry_base_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Base delay for exponential backoff"
    )
    retry_max_delay_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay between retries"
    )
    retry_exponential_base: float = Field(
        default=2.0,
        ge=1.5,
        le=4.0,
        description="Exponential base for backoff calculation"
    )
    retry_jitter_factor: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Jitter factor for retry delay randomization"
    )

    # Circuit breaker settings
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Whether circuit breaker is enabled"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of failures before opening circuit"
    )
    circuit_breaker_success_threshold: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of successes to close circuit"
    )
    circuit_breaker_timeout_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Time before attempting to close circuit"
    )
    circuit_breaker_half_open_max_requests: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum requests allowed in half-open state"
    )

    # Connection pool settings (aiohttp)
    pool_min_size: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum pool size"
    )
    pool_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum pool size"
    )
    pool_acquire_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Timeout for acquiring connection from pool"
    )
    pool_idle_timeout_seconds: float = Field(
        default=300.0,
        ge=60.0,
        le=3600.0,
        description="Timeout for idle connections"
    )
    pool_keepalive_timeout_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Keepalive timeout for connections"
    )

    # Cache settings
    cache_enabled: bool = Field(
        default=True,
        description="Whether caching is enabled"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        le=86400,
        description="Cache time-to-live in seconds"
    )
    cache_max_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Maximum number of cached items"
    )
    cache_warmup_enabled: bool = Field(
        default=False,
        description="Whether to warm up cache on initialization"
    )

    # Rate limiting settings
    rate_limit_enabled: bool = Field(
        default=True,
        description="Whether rate limiting is enabled"
    )
    rate_limit_requests_per_second: float = Field(
        default=10.0,
        ge=0.1,
        le=1000.0,
        description="Maximum requests per second"
    )
    rate_limit_burst_size: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum burst size for rate limiting"
    )
    rate_limit_strategy: RateLimitStrategy = Field(
        default=RateLimitStrategy.TOKEN_BUCKET,
        description="Rate limiting strategy to use"
    )

    # Health check settings
    health_check_enabled: bool = Field(
        default=True,
        description="Whether health checks are enabled"
    )
    health_check_interval_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Interval between health checks"
    )
    health_check_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout for health check"
    )
    health_check_failure_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive failures before marking unhealthy"
    )

    # Reconnection settings
    reconnect_enabled: bool = Field(
        default=True,
        description="Whether automatic reconnection is enabled"
    )
    reconnect_max_attempts: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum reconnection attempts"
    )
    reconnect_initial_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial delay before first reconnection attempt"
    )
    reconnect_max_delay_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=3600.0,
        description="Maximum delay between reconnection attempts"
    )

    # Data validation settings
    data_validation_enabled: bool = Field(
        default=True,
        description="Whether data validation is enabled"
    )
    data_quality_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum data quality threshold (0-1)"
    )

    # Logging and metrics
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level"
    )
    metrics_enabled: bool = Field(
        default=True,
        description="Whether Prometheus metrics are enabled"
    )
    audit_logging_enabled: bool = Field(
        default=True,
        description="Whether audit logging is enabled"
    )
    trace_enabled: bool = Field(
        default=False,
        description="Whether distributed tracing is enabled"
    )

    @field_validator('pool_max_size')
    @classmethod
    def validate_pool_max_size(cls, v: int, info) -> int:
        """Validate pool_max_size is greater than or equal to pool_min_size."""
        min_size = info.data.get('pool_min_size', 1)
        if v < min_size:
            raise ValueError(f'pool_max_size ({v}) must be >= pool_min_size ({min_size})')
        return v


class ConnectionInfo(BaseModel):
    """Information about a connection."""

    model_config = ConfigDict(frozen=True)

    connection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique connection identifier"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Connection creation timestamp"
    )
    last_used_at: Optional[datetime] = Field(
        default=None,
        description="Last time connection was used"
    )
    last_health_check_at: Optional[datetime] = Field(
        default=None,
        description="Last health check timestamp"
    )
    state: ConnectionState = Field(
        default=ConnectionState.DISCONNECTED,
        description="Current connection state"
    )
    remote_address: Optional[str] = Field(
        default=None,
        description="Remote address of the connection"
    )
    remote_port: Optional[int] = Field(
        default=None,
        description="Remote port of the connection"
    )
    protocol: Optional[str] = Field(
        default=None,
        description="Protocol used (HTTP, HTTPS, ONVIF, etc.)"
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of errors on this connection"
    )
    request_count: int = Field(
        default=0,
        ge=0,
        description="Number of requests on this connection"
    )
    bytes_sent: int = Field(
        default=0,
        ge=0,
        description="Total bytes sent"
    )
    bytes_received: int = Field(
        default=0,
        ge=0,
        description="Total bytes received"
    )


class HealthCheckResult(BaseModel):
    """Result of a health check operation."""

    model_config = ConfigDict(frozen=True)

    status: HealthStatus = Field(
        ...,
        description="Health status"
    )
    checked_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of health check"
    )
    latency_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Latency of health check in milliseconds"
    )
    message: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Additional health check message"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional health check details"
    )
    component_health: Dict[str, HealthStatus] = Field(
        default_factory=dict,
        description="Health status of individual components"
    )
    consecutive_failures: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive health check failures"
    )


class DataQualityResult(BaseModel):
    """Result of data quality validation."""

    model_config = ConfigDict(frozen=True)

    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall data quality score (0-1)"
    )
    quality_level: DataQualityLevel = Field(
        ...,
        description="Quality level classification"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of quality issues found"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual quality metrics"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of quality check"
    )


class MetricsSnapshot(BaseModel):
    """Snapshot of connector metrics."""

    model_config = ConfigDict(frozen=True)

    connector_id: str = Field(..., description="Connector ID")
    connector_name: str = Field(..., description="Connector name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Request metrics
    total_requests: int = Field(default=0, ge=0)
    successful_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    retried_requests: int = Field(default=0, ge=0)

    # Timing metrics
    avg_response_time_ms: float = Field(default=0.0, ge=0)
    min_response_time_ms: float = Field(default=0.0, ge=0)
    max_response_time_ms: float = Field(default=0.0, ge=0)
    p95_response_time_ms: float = Field(default=0.0, ge=0)
    p99_response_time_ms: float = Field(default=0.0, ge=0)

    # Circuit breaker
    circuit_state: CircuitState = Field(default=CircuitState.CLOSED)
    circuit_open_count: int = Field(default=0, ge=0)

    # Rate limiting
    rate_limit_hits: int = Field(default=0, ge=0)
    current_rate: float = Field(default=0.0, ge=0)

    # Cache
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    cache_size: int = Field(default=0, ge=0)

    # Connection pool
    pool_active_connections: int = Field(default=0, ge=0)
    pool_idle_connections: int = Field(default=0, ge=0)
    pool_waiting_requests: int = Field(default=0, ge=0)

    # Data quality
    avg_data_quality_score: float = Field(default=0.0, ge=0, le=1)
    data_quality_failures: int = Field(default=0, ge=0)

    # Image processing (specific to thermal cameras)
    images_processed: int = Field(default=0, ge=0)
    images_failed: int = Field(default=0, ge=0)
    avg_image_processing_time_ms: float = Field(default=0.0, ge=0)


class AuditLogEntry(BaseModel):
    """Audit log entry for connector operations."""

    model_config = ConfigDict(frozen=True)

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    connector_id: str = Field(..., description="Connector ID")
    operation: str = Field(..., description="Operation name")
    status: str = Field(..., description="Operation status")
    duration_ms: Optional[float] = Field(default=None, ge=0)
    user_id: Optional[str] = Field(default=None)
    details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)


# =============================================================================
# LRU Cache Implementation
# =============================================================================


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL support."""

    key: str
    value: T
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with TTL support."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: int = 300
    ) -> None:
        """Initialize LRU cache."""
        self._max_size = max_size
        self._default_ttl = default_ttl_seconds
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._access_order: deque = deque()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            entry.access_count += 1
            entry.last_accessed = time.time()
            self._hits += 1

            # Move to end of access order
            try:
                self._access_order.remove(key)
            except ValueError:
                pass
            self._access_order.append(key)

            return entry.value

    async def set(
        self,
        key: str,
        value: T,
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Set value in cache."""
        async with self._lock:
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expires_at = time.time() + ttl if ttl > 0 else None

            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at
            )

            # Evict if at capacity
            while len(self._cache) >= self._max_size and self._access_order:
                oldest_key = self._access_order.popleft()
                self._cache.pop(oldest_key, None)

            self._cache[key] = entry
            self._access_order.append(key)

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()

    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        async with self._lock:
            expired_keys = [
                k for k, v in self._cache.items()
                if v.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
            return len(expired_keys)

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# =============================================================================
# Rate Limiter Implementation
# =============================================================================


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""

    def __init__(
        self,
        rate: float,
        burst_size: int
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            rate: Tokens per second (rate limit)
            burst_size: Maximum burst capacity
        """
        self._rate = rate
        self._burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update

            # Replenish tokens
            self._tokens = min(
                self._burst_size,
                self._tokens + elapsed * self._rate
            )
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait_for_token(self, tokens: int = 1) -> float:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Time waited in seconds
        """
        start = time.time()

        while not await self.acquire(tokens):
            # Calculate wait time
            async with self._lock:
                deficit = tokens - self._tokens
                wait_time = deficit / self._rate
            await asyncio.sleep(min(wait_time, 1.0))

        return time.time() - start

    @property
    def available_tokens(self) -> float:
        """Current available tokens."""
        return self._tokens

    @property
    def current_rate(self) -> float:
        """Configured rate."""
        return self._rate


class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation."""

    def __init__(
        self,
        rate: float,
        window_size_seconds: float = 1.0
    ) -> None:
        """
        Initialize sliding window rate limiter.

        Args:
            rate: Maximum requests per window
            window_size_seconds: Window size in seconds
        """
        self._rate = rate
        self._window_size = window_size_seconds
        self._requests: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to make a request."""
        async with self._lock:
            now = time.time()
            window_start = now - self._window_size

            # Remove old requests
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()

            if len(self._requests) < self._rate:
                self._requests.append(now)
                return True
            return False

    @property
    def current_count(self) -> int:
        """Current request count in window."""
        return len(self._requests)


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_seconds: float = 60.0,
        half_open_max_requests: int = 3
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening
            success_threshold: Successes to close
            timeout_seconds: Time before half-open
            half_open_max_requests: Max requests in half-open
        """
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._timeout = timeout_seconds
        self._half_open_max = half_open_max_requests

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_requests = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_requests = 0
        self._rejected_requests = 0
        self._open_count = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    async def can_execute(self) -> bool:
        """Check if request can be executed."""
        async with self._lock:
            self._total_requests += 1

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_requests = 0
                    logger.info("Circuit breaker entering half-open state")
                    return True
                self._rejected_requests += 1
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests < self._half_open_max:
                    self._half_open_requests += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker closed")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    async def record_failure(self) -> None:
        """Record failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._open_count += 1
                logger.warning("Circuit breaker re-opened from half-open")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    self._open_count += 1
                    logger.warning(
                        f"Circuit breaker opened after {self._failure_count} failures"
                    )

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset circuit."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self._timeout

    async def force_open(self) -> None:
        """Force circuit to open state."""
        async with self._lock:
            self._state = CircuitState.OPEN
            self._open_count += 1
            logger.warning("Circuit breaker force-opened")

    async def force_close(self) -> None:
        """Force circuit to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            logger.info("Circuit breaker force-closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_requests": self._total_requests,
            "rejected_requests": self._rejected_requests,
            "open_count": self._open_count,
        }


# =============================================================================
# Connection Pool Implementation (aiohttp)
# =============================================================================


class ConnectionPool:
    """
    Connection pool manager using aiohttp.

    Manages HTTP/HTTPS connections with configurable pool size,
    keepalive settings, and connection lifecycle management.
    """

    def __init__(
        self,
        min_size: int = 1,
        max_size: int = 10,
        keepalive_timeout: float = 60.0,
        idle_timeout: float = 300.0
    ) -> None:
        """
        Initialize connection pool.

        Args:
            min_size: Minimum connections to maintain
            max_size: Maximum connections allowed
            keepalive_timeout: TCP keepalive timeout
            idle_timeout: Idle connection timeout
        """
        self._min_size = min_size
        self._max_size = max_size
        self._keepalive_timeout = keepalive_timeout
        self._idle_timeout = idle_timeout

        self._connector: Optional[aiohttp.TCPConnector] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        async with self._lock:
            if self._initialized:
                return

            self._connector = aiohttp.TCPConnector(
                limit=self._max_size,
                limit_per_host=self._max_size,
                keepalive_timeout=self._keepalive_timeout,
                enable_cleanup_closed=True,
                force_close=False,
            )

            timeout = aiohttp.ClientTimeout(
                total=300,
                connect=30,
                sock_read=60,
                sock_connect=30
            )

            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout
            )

            self._initialized = True
            logger.info(f"Connection pool initialized (max_size={self._max_size})")

    async def get_session(self) -> aiohttp.ClientSession:
        """Get the aiohttp session."""
        if not self._initialized:
            await self.initialize()
        return self._session

    async def close(self) -> None:
        """Close the connection pool."""
        async with self._lock:
            if self._session:
                await self._session.close()
                self._session = None
            if self._connector:
                await self._connector.close()
                self._connector = None
            self._initialized = False
            logger.info("Connection pool closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        if not self._connector:
            return {
                "initialized": False,
                "active_connections": 0,
                "idle_connections": 0,
            }

        return {
            "initialized": self._initialized,
            "max_size": self._max_size,
            "active_connections": len(self._connector._acquired),
            "available_connections": self._connector.limit - len(self._connector._acquired),
        }


# =============================================================================
# Metrics Collector
# =============================================================================


class MetricsCollector:
    """Collects and aggregates connector metrics."""

    def __init__(self, connector_id: str, connector_name: str) -> None:
        """Initialize metrics collector."""
        self._connector_id = connector_id
        self._connector_name = connector_name
        self._lock = asyncio.Lock()

        # Request metrics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._retried_requests = 0

        # Timing metrics
        self._response_times: deque = deque(maxlen=1000)

        # Rate limiting metrics
        self._rate_limit_hits = 0

        # Data quality metrics
        self._quality_scores: deque = deque(maxlen=100)
        self._quality_failures = 0

        # Image processing metrics
        self._images_processed = 0
        self._images_failed = 0
        self._image_processing_times: deque = deque(maxlen=100)

    async def record_request(
        self,
        success: bool,
        response_time_ms: float,
        retried: bool = False
    ) -> None:
        """Record a request."""
        async with self._lock:
            self._total_requests += 1
            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1
            if retried:
                self._retried_requests += 1
            self._response_times.append(response_time_ms)

    async def record_rate_limit_hit(self) -> None:
        """Record a rate limit hit."""
        async with self._lock:
            self._rate_limit_hits += 1

    async def record_data_quality(
        self,
        quality_score: float,
        passed: bool
    ) -> None:
        """Record data quality check."""
        async with self._lock:
            self._quality_scores.append(quality_score)
            if not passed:
                self._quality_failures += 1

    async def record_image_processing(
        self,
        success: bool,
        processing_time_ms: float
    ) -> None:
        """Record image processing metrics."""
        async with self._lock:
            if success:
                self._images_processed += 1
            else:
                self._images_failed += 1
            self._image_processing_times.append(processing_time_ms)

    def get_snapshot(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        cache: Optional[LRUCache] = None,
        pool: Optional[ConnectionPool] = None
    ) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        response_times = list(self._response_times)

        if response_times:
            sorted_times = sorted(response_times)
            avg_time = sum(response_times) / len(response_times)
            min_time = sorted_times[0]
            max_time = sorted_times[-1]
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            p95_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else max_time
            p99_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else max_time
        else:
            avg_time = min_time = max_time = p95_time = p99_time = 0.0

        quality_scores = list(self._quality_scores)
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        image_times = list(self._image_processing_times)
        avg_image_time = sum(image_times) / len(image_times) if image_times else 0.0

        cache_stats = cache.get_stats() if cache else {}
        pool_stats = pool.get_stats() if pool else {}
        cb_stats = circuit_breaker.get_stats() if circuit_breaker else {}

        return MetricsSnapshot(
            connector_id=self._connector_id,
            connector_name=self._connector_name,
            total_requests=self._total_requests,
            successful_requests=self._successful_requests,
            failed_requests=self._failed_requests,
            retried_requests=self._retried_requests,
            avg_response_time_ms=avg_time,
            min_response_time_ms=min_time,
            max_response_time_ms=max_time,
            p95_response_time_ms=p95_time,
            p99_response_time_ms=p99_time,
            circuit_state=CircuitState(cb_stats.get("state", "closed")),
            circuit_open_count=cb_stats.get("open_count", 0),
            rate_limit_hits=self._rate_limit_hits,
            cache_hits=cache_stats.get("hits", 0),
            cache_misses=cache_stats.get("misses", 0),
            cache_size=cache_stats.get("size", 0),
            pool_active_connections=pool_stats.get("active_connections", 0),
            pool_idle_connections=pool_stats.get("available_connections", 0),
            avg_data_quality_score=avg_quality,
            data_quality_failures=self._quality_failures,
            images_processed=self._images_processed,
            images_failed=self._images_failed,
            avg_image_processing_time_ms=avg_image_time,
        )


# =============================================================================
# Audit Logger
# =============================================================================


class AuditLogger:
    """Logs connector operations for audit trail."""

    def __init__(
        self,
        connector_id: str,
        enabled: bool = True
    ) -> None:
        """Initialize audit logger."""
        self._connector_id = connector_id
        self._enabled = enabled
        self._logger = logging.getLogger(f"{__name__}.audit.{connector_id}")
        self._entries: deque = deque(maxlen=10000)

    async def log(
        self,
        operation: str,
        status: str,
        duration_ms: Optional[float] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> AuditLogEntry:
        """Log an operation."""
        entry = AuditLogEntry(
            connector_id=self._connector_id,
            operation=operation,
            status=status,
            duration_ms=duration_ms,
            user_id=user_id,
            details=details or {},
            error_message=error_message
        )

        if self._enabled:
            self._entries.append(entry)
            log_msg = f"[AUDIT] {operation}: {status}"
            if duration_ms:
                log_msg += f" ({duration_ms:.2f}ms)"
            if error_message:
                self._logger.warning(f"{log_msg} - {error_message}")
            else:
                self._logger.info(log_msg)

        return entry

    def get_recent_entries(self, count: int = 100) -> List[AuditLogEntry]:
        """Get recent audit entries."""
        return list(self._entries)[-count:]


# =============================================================================
# Data Validator
# =============================================================================


class DataValidator:
    """Validates data quality for connector responses."""

    def __init__(
        self,
        quality_threshold: float = 0.6,
        enabled: bool = True
    ) -> None:
        """Initialize data validator."""
        self._threshold = quality_threshold
        self._enabled = enabled

    def validate(
        self,
        data: Any,
        schema: Optional[Dict[str, Any]] = None
    ) -> DataQualityResult:
        """
        Validate data quality.

        Args:
            data: Data to validate
            schema: Optional schema for validation

        Returns:
            Data quality result
        """
        if not self._enabled:
            return DataQualityResult(
                quality_score=1.0,
                quality_level=DataQualityLevel.EXCELLENT,
                issues=[],
                metrics={}
            )

        issues = []
        metrics = {}

        # Check for None/empty data
        if data is None:
            issues.append("Data is None")
            metrics["completeness"] = 0.0
        elif isinstance(data, (list, dict)) and len(data) == 0:
            issues.append("Data is empty")
            metrics["completeness"] = 0.0
        else:
            metrics["completeness"] = 1.0

        # Check for NaN/Inf values (if numeric)
        if isinstance(data, (int, float)):
            import math
            if math.isnan(data) or math.isinf(data):
                issues.append("Data contains NaN or Inf")
                metrics["validity"] = 0.0
            else:
                metrics["validity"] = 1.0

        # Calculate overall score
        if metrics:
            quality_score = sum(metrics.values()) / len(metrics)
        else:
            quality_score = 1.0

        # Determine quality level
        if quality_score >= 0.95:
            quality_level = DataQualityLevel.EXCELLENT
        elif quality_score >= 0.80:
            quality_level = DataQualityLevel.GOOD
        elif quality_score >= 0.60:
            quality_level = DataQualityLevel.ACCEPTABLE
        elif quality_score >= 0.40:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.UNACCEPTABLE

        return DataQualityResult(
            quality_score=quality_score,
            quality_level=quality_level,
            issues=issues,
            metrics=metrics
        )

    def meets_threshold(self, result: DataQualityResult) -> bool:
        """Check if data quality meets threshold."""
        return result.quality_score >= self._threshold


# =============================================================================
# Retry Decorator
# =============================================================================


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter_factor: float = 0.1,
    retryable_exceptions: Tuple[type, ...] = (ConnectionError, TimeoutError)
):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter_factor: Random jitter factor (0-1)
        retryable_exceptions: Exceptions to retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    if attempt >= max_retries:
                        logger.error(
                            f"Retry exhausted after {max_retries + 1} attempts: {e}"
                        )
                        raise RetryExhaustedError(
                            f"All {max_retries + 1} attempts failed",
                            details={"last_error": str(e)}
                        )

                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    jitter = delay * jitter_factor * random.random()
                    delay += jitter

                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)

            raise RetryExhaustedError(
                "Retry logic completed without success or exception",
                details={"last_error": str(last_exception) if last_exception else None}
            )

        return wrapper
    return decorator


# =============================================================================
# Base Connector Abstract Class
# =============================================================================


class BaseConnector(ABC):
    """
    Abstract base class for GL-015 INSULSCAN connectors.

    Provides common functionality including:
    - Connection pooling (aiohttp)
    - Retry logic with exponential backoff
    - Rate limiting (token bucket)
    - Circuit breaker pattern
    - Health check interface
    - Caching with LRU eviction
    - Metrics collection
    - Audit logging
    - Data validation
    """

    def __init__(self, config: BaseConnectorConfig) -> None:
        """
        Initialize base connector.

        Args:
            config: Connector configuration
        """
        self._config = config
        self._logger = logging.getLogger(
            f"{__name__}.{config.connector_type.value}.{config.connector_id}"
        )

        # Set log level
        log_level = getattr(logging, config.log_level, logging.INFO)
        self._logger.setLevel(log_level)

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._connection_info: Optional[ConnectionInfo] = None

        # Initialize components
        self._pool = ConnectionPool(
            min_size=config.pool_min_size,
            max_size=config.pool_max_size,
            keepalive_timeout=config.pool_keepalive_timeout_seconds,
            idle_timeout=config.pool_idle_timeout_seconds
        )

        self._cache = LRUCache[Any](
            max_size=config.cache_max_size,
            default_ttl_seconds=config.cache_ttl_seconds
        ) if config.cache_enabled else None

        self._rate_limiter = TokenBucketRateLimiter(
            rate=config.rate_limit_requests_per_second,
            burst_size=config.rate_limit_burst_size
        ) if config.rate_limit_enabled else None

        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            success_threshold=config.circuit_breaker_success_threshold,
            timeout_seconds=config.circuit_breaker_timeout_seconds,
            half_open_max_requests=config.circuit_breaker_half_open_max_requests
        ) if config.circuit_breaker_enabled else None

        self._metrics = MetricsCollector(
            connector_id=config.connector_id,
            connector_name=config.connector_name
        ) if config.metrics_enabled else None

        self._audit_logger = AuditLogger(
            connector_id=config.connector_id,
            enabled=config.audit_logging_enabled
        )

        self._data_validator = DataValidator(
            quality_threshold=config.data_quality_threshold,
            enabled=config.data_validation_enabled
        )

        # Health check state
        self._last_health_check: Optional[HealthCheckResult] = None
        self._consecutive_health_failures = 0
        self._health_check_task: Optional[asyncio.Task] = None

    @property
    def config(self) -> BaseConnectorConfig:
        """Get connector configuration."""
        return self._config

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def circuit_state(self) -> CircuitState:
        """Get circuit breaker state."""
        if self._circuit_breaker:
            return self._circuit_breaker.state
        return CircuitState.CLOSED

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the external system.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the external system.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on the connection.

        Must be implemented by subclasses.

        Returns:
            Health check result
        """
        pass

    @abstractmethod
    async def validate_configuration(self) -> bool:
        """
        Validate connector configuration.

        Must be implemented by subclasses.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass

    # =========================================================================
    # Connection Lifecycle
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize the connector and all components."""
        self._logger.info(f"Initializing connector: {self._config.connector_name}")

        # Validate configuration
        await self.validate_configuration()

        # Initialize connection pool
        await self._pool.initialize()

        # Start health check task if enabled
        if self._config.health_check_enabled:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

        self._state = ConnectionState.CONNECTING

    async def shutdown(self) -> None:
        """Shutdown the connector and cleanup resources."""
        self._logger.info(f"Shutting down connector: {self._config.connector_name}")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Disconnect
        await self.disconnect()

        # Close connection pool
        await self._pool.close()

        # Clear cache
        if self._cache:
            await self._cache.clear()

        self._state = ConnectionState.DISCONNECTED

    async def reconnect(self) -> None:
        """Attempt to reconnect to the external system."""
        if not self._config.reconnect_enabled:
            raise ConfigurationError("Reconnection is disabled")

        self._state = ConnectionState.RECONNECTING
        delay = self._config.reconnect_initial_delay_seconds

        for attempt in range(self._config.reconnect_max_attempts):
            try:
                self._logger.info(
                    f"Reconnection attempt {attempt + 1}/{self._config.reconnect_max_attempts}"
                )
                await self.disconnect()
                await self.connect()
                self._logger.info("Reconnection successful")
                return

            except Exception as e:
                self._logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")

                if attempt < self._config.reconnect_max_attempts - 1:
                    await asyncio.sleep(delay)
                    delay = min(
                        delay * 2,
                        self._config.reconnect_max_delay_seconds
                    )

        self._state = ConnectionState.ERROR
        raise ConnectionError(
            f"Failed to reconnect after {self._config.reconnect_max_attempts} attempts"
        )

    # =========================================================================
    # Health Check
    # =========================================================================

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                if self._state not in [
                    ConnectionState.CONNECTED,
                    ConnectionState.DEGRADED
                ]:
                    continue

                result = await self.health_check()
                self._last_health_check = result

                if result.status == HealthStatus.HEALTHY:
                    self._consecutive_health_failures = 0
                    if self._state == ConnectionState.DEGRADED:
                        self._state = ConnectionState.CONNECTED
                else:
                    self._consecutive_health_failures += 1

                    if self._consecutive_health_failures >= self._config.health_check_failure_threshold:
                        self._state = ConnectionState.DEGRADED
                        self._logger.warning(
                            f"Connector degraded after {self._consecutive_health_failures} "
                            "consecutive health check failures"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health check error: {e}")
                self._consecutive_health_failures += 1

    # =========================================================================
    # Request Execution with Protection
    # =========================================================================

    async def execute_with_protection(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        operation_name: str = "operation",
        use_cache: bool = False,
        cache_key: Optional[str] = None,
        validate_result: bool = True
    ) -> T:
        """
        Execute an operation with all protection mechanisms.

        Args:
            operation: Async operation to execute
            operation_name: Name for logging/metrics
            use_cache: Whether to use caching
            cache_key: Cache key (required if use_cache=True)
            validate_result: Whether to validate result quality

        Returns:
            Operation result

        Raises:
            CircuitOpenError: If circuit breaker is open
            RateLimitError: If rate limit exceeded
            DataQualityError: If data quality check fails
        """
        start_time = time.time()

        try:
            # Check circuit breaker
            if self._circuit_breaker and not await self._circuit_breaker.can_execute():
                raise CircuitOpenError(
                    f"Circuit breaker is open for {self._config.connector_name}"
                )

            # Check rate limiter
            if self._rate_limiter and not await self._rate_limiter.acquire():
                if self._metrics:
                    await self._metrics.record_rate_limit_hit()
                raise RateLimitError(
                    f"Rate limit exceeded for {self._config.connector_name}"
                )

            # Check cache
            if use_cache and cache_key and self._cache:
                cached_value = await self._cache.get(cache_key)
                if cached_value is not None:
                    self._logger.debug(f"Cache hit for {cache_key}")
                    return cached_value

            # Execute operation
            result = await operation()

            # Validate result
            if validate_result:
                quality_result = self._data_validator.validate(result)
                if self._metrics:
                    await self._metrics.record_data_quality(
                        quality_result.quality_score,
                        self._data_validator.meets_threshold(quality_result)
                    )
                if not self._data_validator.meets_threshold(quality_result):
                    raise DataQualityError(
                        f"Data quality below threshold: {quality_result.quality_score:.2f}",
                        details={"issues": quality_result.issues}
                    )

            # Cache result
            if use_cache and cache_key and self._cache:
                await self._cache.set(cache_key, result)

            # Record success
            if self._circuit_breaker:
                await self._circuit_breaker.record_success()

            duration_ms = (time.time() - start_time) * 1000
            if self._metrics:
                await self._metrics.record_request(True, duration_ms)

            await self._audit_logger.log(
                operation=operation_name,
                status="success",
                duration_ms=duration_ms
            )

            return result

        except (CircuitOpenError, RateLimitError, DataQualityError):
            raise

        except Exception as e:
            # Record failure
            if self._circuit_breaker:
                await self._circuit_breaker.record_failure()

            duration_ms = (time.time() - start_time) * 1000
            if self._metrics:
                await self._metrics.record_request(False, duration_ms)

            await self._audit_logger.log(
                operation=operation_name,
                status="failure",
                duration_ms=duration_ms,
                error_message=str(e)
            )

            raise

    def _generate_cache_key(
        self,
        *args,
        **kwargs
    ) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps(
            {"args": args, "kwargs": kwargs},
            sort_keys=True,
            default=str
        )
        return hashlib.md5(key_data.encode()).hexdigest()

    # =========================================================================
    # Metrics and Monitoring
    # =========================================================================

    def get_metrics(self) -> Optional[MetricsSnapshot]:
        """Get current metrics snapshot."""
        if not self._metrics:
            return None
        return self._metrics.get_snapshot(
            circuit_breaker=self._circuit_breaker,
            cache=self._cache,
            pool=self._pool
        )

    def get_recent_audit_logs(self, count: int = 100) -> List[AuditLogEntry]:
        """Get recent audit log entries."""
        return self._audit_logger.get_recent_entries(count)


# =============================================================================
# Context Manager for Connectors
# =============================================================================


class ConnectorContextManager:
    """Context manager for connector lifecycle."""

    def __init__(self, connector: BaseConnector) -> None:
        """Initialize context manager."""
        self._connector = connector

    async def __aenter__(self) -> BaseConnector:
        """Enter context and connect."""
        await self._connector.initialize()
        await self._connector.connect()
        return self._connector

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and disconnect."""
        await self._connector.shutdown()


# =============================================================================
# Factory Function
# =============================================================================


def create_connector_config(
    connector_name: str,
    connector_type: ConnectorType,
    **kwargs
) -> BaseConnectorConfig:
    """
    Factory function to create connector configuration.

    Args:
        connector_name: Name of the connector
        connector_type: Type of connector
        **kwargs: Additional configuration options

    Returns:
        Configured BaseConnectorConfig instance
    """
    return BaseConnectorConfig(
        connector_name=connector_name,
        connector_type=connector_type,
        **kwargs
    )
