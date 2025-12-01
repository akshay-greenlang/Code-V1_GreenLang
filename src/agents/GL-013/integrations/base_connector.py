"""
Base Connector Module for GL-013 PREDICTMAINT (Predictive Maintenance Agent).

Provides abstract base class with common functionality for all predictive maintenance
system connectors including connection pooling, retry logic, circuit breaker pattern,
health monitoring, caching, rate limiting, and error handling.

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
import time
import uuid
from collections import deque

from pydantic import BaseModel, Field, ConfigDict, field_validator

# Configure module logger
logger = logging.getLogger(__name__)

# Generic type for cached values
T = TypeVar("T")


# =============================================================================
# Enumerations
# =============================================================================


class ConnectionState(str, Enum):
    """Connection state enumeration for predictive maintenance connectors."""

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
    """Types of connectors supported by GL-013 PREDICTMAINT."""

    CMMS = "cmms"  # Computerized Maintenance Management System
    CONDITION_MONITORING = "condition_monitoring"
    IOT_SENSOR = "iot_sensor"
    AGENT_COORDINATOR = "agent_coordinator"
    VIBRATION_ANALYZER = "vibration_analyzer"
    THERMAL_IMAGING = "thermal_imaging"
    OIL_ANALYSIS = "oil_analysis"
    MOTOR_CURRENT = "motor_current"
    ULTRASONIC = "ultrasonic"
    SCADA = "scada"


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


# =============================================================================
# Pydantic Models
# =============================================================================


class BaseConnectorConfig(BaseModel):
    """Base configuration for all predictive maintenance connectors."""

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

    # Connection pool settings
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
        description="Protocol used (HTTP, HTTPS, MQTT, OPC-UA, etc.)"
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
    completeness_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Data completeness score"
    )
    validity_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Data validity score"
    )
    consistency_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Data consistency score"
    )
    timeliness_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Data timeliness score"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of data quality issues found"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving data quality"
    )


class MetricsSnapshot(BaseModel):
    """Snapshot of connector metrics for Prometheus export."""

    model_config = ConfigDict(frozen=True)

    connector_id: str = Field(..., description="Connector identifier")
    connector_type: ConnectorType = Field(..., description="Connector type")
    connector_name: str = Field(default="", description="Connector name")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Metrics timestamp"
    )

    # Connection state
    connection_state: ConnectionState = Field(
        default=ConnectionState.DISCONNECTED,
        description="Current connection state"
    )

    # Connection pool metrics
    total_connections: int = Field(default=0, ge=0)
    active_connections: int = Field(default=0, ge=0)
    idle_connections: int = Field(default=0, ge=0)
    connection_errors: int = Field(default=0, ge=0)

    # Request metrics
    total_requests: int = Field(default=0, ge=0)
    successful_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    retried_requests: int = Field(default=0, ge=0)
    rate_limited_requests: int = Field(default=0, ge=0)

    # Latency metrics (in milliseconds)
    avg_latency_ms: float = Field(default=0.0, ge=0)
    min_latency_ms: float = Field(default=0.0, ge=0)
    max_latency_ms: float = Field(default=0.0, ge=0)
    p50_latency_ms: float = Field(default=0.0, ge=0)
    p95_latency_ms: float = Field(default=0.0, ge=0)
    p99_latency_ms: float = Field(default=0.0, ge=0)

    # Throughput metrics
    requests_per_second: float = Field(default=0.0, ge=0)
    bytes_sent_per_second: float = Field(default=0.0, ge=0)
    bytes_received_per_second: float = Field(default=0.0, ge=0)

    # Circuit breaker metrics
    circuit_state: CircuitState = Field(default=CircuitState.CLOSED)
    circuit_open_count: int = Field(default=0, ge=0)
    circuit_half_open_count: int = Field(default=0, ge=0)

    # Rate limiting metrics
    rate_limit_remaining: int = Field(default=0, ge=0)
    rate_limit_reset_at: Optional[datetime] = Field(default=None)

    # Cache metrics
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    cache_size: int = Field(default=0, ge=0)
    cache_hit_rate: float = Field(default=0.0, ge=0, le=1.0)

    # Health metrics
    health_status: HealthStatus = Field(default=HealthStatus.UNKNOWN)
    health_check_latency_ms: float = Field(default=0.0, ge=0)
    consecutive_health_failures: int = Field(default=0, ge=0)

    # Data quality metrics
    avg_data_quality_score: float = Field(default=1.0, ge=0, le=1.0)
    data_validation_failures: int = Field(default=0, ge=0)

    # Error metrics
    error_count: int = Field(default=0, ge=0)
    last_error: Optional[str] = Field(default=None)
    last_error_at: Optional[datetime] = Field(default=None)

    # Reconnection metrics
    reconnection_attempts: int = Field(default=0, ge=0)
    successful_reconnections: int = Field(default=0, ge=0)
    failed_reconnections: int = Field(default=0, ge=0)


class AuditLogEntry(BaseModel):
    """Audit log entry for compliance tracking."""

    model_config = ConfigDict(frozen=True)

    entry_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique entry identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Entry timestamp"
    )
    connector_id: str = Field(..., description="Connector identifier")
    connector_type: ConnectorType = Field(..., description="Connector type")
    operation: str = Field(..., description="Operation performed")
    status: str = Field(..., description="Operation status")
    user_id: Optional[str] = Field(default=None, description="User ID if applicable")
    source_ip: Optional[str] = Field(default=None, description="Source IP address")
    target_system: Optional[str] = Field(default=None, description="Target system")
    request_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Request data (sanitized)"
    )
    response_summary: Optional[str] = Field(
        default=None,
        description="Summary of response"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    duration_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Operation duration in milliseconds"
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Distributed trace ID"
    )
    span_id: Optional[str] = Field(
        default=None,
        description="Distributed span ID"
    )


# =============================================================================
# Exceptions
# =============================================================================


class ConnectorError(Exception):
    """Base exception for connector errors."""

    def __init__(
        self,
        message: str,
        connector_id: Optional[str] = None,
        connector_type: Optional[ConnectorType] = None,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.connector_id = connector_id
        self.connector_type = connector_type
        self.details = details or {}
        self.retryable = retryable
        self.timestamp = datetime.utcnow()


class ConnectionError(ConnectorError):
    """Error establishing or maintaining connection."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, retryable=True, **kwargs)


class AuthenticationError(ConnectorError):
    """Authentication or authorization error."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, retryable=False, **kwargs)


class TimeoutError(ConnectorError):
    """Operation timed out."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, retryable=True, **kwargs)


class ValidationError(ConnectorError):
    """Data validation error."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, retryable=False, **kwargs)


class CircuitOpenError(ConnectorError):
    """Circuit breaker is open, requests rejected."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, retryable=False, **kwargs)


class RetryExhaustedError(ConnectorError):
    """All retry attempts exhausted."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, retryable=False, **kwargs)


class ConfigurationError(ConnectorError):
    """Configuration error."""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, retryable=False, **kwargs)


class DataQualityError(ConnectorError):
    """Data quality validation failed."""

    def __init__(self, message: str, quality_result: Optional[DataQualityResult] = None, **kwargs) -> None:
        super().__init__(message, retryable=False, **kwargs)
        self.quality_result = quality_result


class RateLimitError(ConnectorError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after_seconds: Optional[float] = None, **kwargs) -> None:
        super().__init__(message, retryable=True, **kwargs)
        self.retry_after_seconds = retry_after_seconds


class ProtocolError(ConnectorError):
    """Protocol-level error (OPC-UA, MQTT, Modbus, etc.)."""

    def __init__(self, message: str, protocol: str, **kwargs) -> None:
        super().__init__(message, retryable=True, **kwargs)
        self.protocol = protocol


# =============================================================================
# Cache Implementation
# =============================================================================


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL support and access tracking."""

    key: str
    value: T
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 300
    access_count: int = 0
    last_accessed_at: float = field(default_factory=time.time)
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at

    def access(self) -> T:
        """Access the cached value and update tracking."""
        self.access_count += 1
        self.last_accessed_at = time.time()
        return self.value


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with TTL support and eviction policies."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 300,
        max_memory_bytes: Optional[int] = None,
    ) -> None:
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            max_memory_bytes: Optional maximum memory limit
        """
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._max_memory_bytes = max_memory_bytes
        self._current_memory_bytes = 0
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._current_memory_bytes -= entry.size_bytes
                self._misses += 1
                return None

            self._hits += 1
            return entry.access()

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
        size_bytes: int = 0,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
            size_bytes: Size of value in bytes for memory tracking
        """
        async with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory_bytes -= old_entry.size_bytes
                del self._cache[key]

            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                await self._evict_lru_unlocked()

            # Evict if memory limit exceeded
            if self._max_memory_bytes:
                while (self._current_memory_bytes + size_bytes > self._max_memory_bytes
                       and self._cache):
                    await self._evict_lru_unlocked()

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl or self._default_ttl,
                size_bytes=size_bytes,
            )
            self._current_memory_bytes += size_bytes

    async def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted
        """
        async with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._current_memory_bytes -= entry.size_bytes
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._current_memory_bytes = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                entry = self._cache[key]
                self._current_memory_bytes -= entry.size_bytes
                del self._cache[key]
            return len(expired_keys)

    async def _evict_lru_unlocked(self) -> None:
        """Evict least recently used entry (must hold lock)."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed_at
        )
        entry = self._cache[lru_key]
        self._current_memory_bytes -= entry.size_bytes
        del self._cache[lru_key]
        self._evictions += 1

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "memory_bytes": self._current_memory_bytes,
            "max_memory_bytes": self._max_memory_bytes,
        }


# =============================================================================
# Rate Limiter Implementation
# =============================================================================


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""

    def __init__(
        self,
        rate: float,
        burst_size: int,
    ) -> None:
        """
        Initialize token bucket rate limiter.

        Args:
            rate: Tokens per second
            burst_size: Maximum bucket size
        """
        self._rate = rate
        self._burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = time.time()
        self._lock = asyncio.Lock()
        self._rejected_count = 0

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False if rate limited
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._last_update = now

            # Refill tokens
            self._tokens = min(
                self._burst_size,
                self._tokens + elapsed * self._rate
            )

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            else:
                self._rejected_count += 1
                return False

    async def wait_for_token(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds

        Returns:
            True if tokens were acquired, False if timed out
        """
        start_time = time.time()
        while True:
            if await self.acquire(tokens):
                return True

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            # Calculate wait time
            async with self._lock:
                wait_time = (tokens - self._tokens) / self._rate

            # Cap wait time
            if timeout is not None:
                wait_time = min(wait_time, timeout - (time.time() - start_time))

            if wait_time > 0:
                await asyncio.sleep(wait_time)

    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        return int(self._tokens)

    @property
    def rejected_count(self) -> int:
        """Get count of rejected requests."""
        return self._rejected_count

    def reset(self) -> None:
        """Reset the rate limiter."""
        self._tokens = float(self._burst_size)
        self._last_update = time.time()
        self._rejected_count = 0


class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation."""

    def __init__(
        self,
        rate: float,
        window_seconds: float = 1.0,
    ) -> None:
        """
        Initialize sliding window rate limiter.

        Args:
            rate: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self._rate = int(rate)
        self._window_seconds = window_seconds
        self._requests: deque = deque()
        self._lock = asyncio.Lock()
        self._rejected_count = 0

    async def acquire(self) -> bool:
        """
        Try to acquire a request slot.

        Returns:
            True if request is allowed, False if rate limited
        """
        async with self._lock:
            now = time.time()
            window_start = now - self._window_seconds

            # Remove old requests
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()

            if len(self._requests) < self._rate:
                self._requests.append(now)
                return True
            else:
                self._rejected_count += 1
                return False

    @property
    def remaining(self) -> int:
        """Get remaining request slots in current window."""
        now = time.time()
        window_start = now - self._window_seconds
        current_count = sum(1 for t in self._requests if t >= window_start)
        return max(0, self._rate - current_count)

    @property
    def rejected_count(self) -> int:
        """Get count of rejected requests."""
        return self._rejected_count


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing state, requests are rejected immediately
    - HALF_OPEN: Testing state, limited requests allowed to test recovery
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_seconds: float = 60.0,
        half_open_max_requests: int = 3,
        name: str = "circuit_breaker",
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            success_threshold: Successes to close circuit from half-open
            timeout_seconds: Time before attempting to close circuit
            half_open_max_requests: Max requests in half-open state
            name: Name for logging
        """
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._timeout_seconds = timeout_seconds
        self._half_open_max_requests = half_open_max_requests
        self._name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: float = time.time()
        self._open_count = 0
        self._half_open_count = 0
        self._lock = asyncio.Lock()

        # Track failure history for analysis
        self._failure_history: deque = deque(maxlen=100)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self._state == CircuitState.OPEN

    @property
    def open_count(self) -> int:
        """Get number of times circuit has opened."""
        return self._open_count

    @property
    def half_open_count(self) -> int:
        """Get number of times circuit entered half-open state."""
        return self._half_open_count

    @property
    def time_in_current_state(self) -> float:
        """Get time spent in current state in seconds."""
        return time.time() - self._last_state_change

    async def can_execute(self) -> bool:
        """
        Check if request can be executed.

        Returns:
            True if request can proceed
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self._timeout_seconds:
                        self._state = CircuitState.HALF_OPEN
                        self._last_state_change = time.time()
                        self._success_count = 0
                        self._half_open_requests = 0
                        self._half_open_count += 1
                        logger.info(
                            f"Circuit breaker '{self._name}' transitioning to HALF_OPEN "
                            f"after {elapsed:.1f}s"
                        )
                        return True
                return False

            # HALF_OPEN state - allow limited requests
            if self._half_open_requests < self._half_open_max_requests:
                self._half_open_requests += 1
                return True
            return False

    async def record_success(self) -> None:
        """Record successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._last_state_change = time.time()
                    self._failure_count = 0
                    logger.info(
                        f"Circuit breaker '{self._name}' closed after "
                        f"{self._success_count} successes"
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """
        Record failed request.

        Args:
            error: Optional exception that caused the failure
        """
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._failure_history.append({
                "timestamp": self._last_failure_time,
                "error": str(error) if error else None,
                "state": self._state.value,
            })

            if self._state == CircuitState.HALF_OPEN:
                # Immediately open on failure in half-open state
                self._state = CircuitState.OPEN
                self._last_state_change = time.time()
                self._open_count += 1
                logger.warning(
                    f"Circuit breaker '{self._name}' opened from HALF_OPEN "
                    f"after failure: {error}"
                )
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_state_change = time.time()
                    self._open_count += 1
                    logger.warning(
                        f"Circuit breaker '{self._name}' opened after "
                        f"{self._failure_count} failures: {error}"
                    )

    async def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._last_state_change = time.time()
            self._failure_count = 0
            self._success_count = 0
            self._half_open_requests = 0
            self._last_failure_time = None

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "open_count": self._open_count,
            "half_open_count": self._half_open_count,
            "time_in_current_state_seconds": self.time_in_current_state,
            "failure_history_size": len(self._failure_history),
        }


# =============================================================================
# Connection Pool Implementation
# =============================================================================


@dataclass
class PooledConnection(Generic[T]):
    """Wrapper for pooled connections with metadata."""

    connection: T
    connection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0
    error_count: int = 0
    healthy: bool = True

    @property
    def age_seconds(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_used_at


class ConnectionPool(Generic[T]):
    """
    Generic async connection pool.

    Manages a pool of reusable connections with health checking,
    automatic reconnection, and idle timeout handling.
    """

    def __init__(
        self,
        factory: Callable[[], Coroutine[Any, Any, T]],
        min_size: int = 1,
        max_size: int = 10,
        acquire_timeout: float = 30.0,
        idle_timeout: float = 300.0,
        health_check_interval: float = 30.0,
        validation_func: Optional[Callable[[T], Coroutine[Any, Any, bool]]] = None,
    ) -> None:
        """
        Initialize connection pool.

        Args:
            factory: Async factory function to create new connections
            min_size: Minimum pool size
            max_size: Maximum pool size
            acquire_timeout: Timeout for acquiring connection
            idle_timeout: Timeout for idle connections
            health_check_interval: Interval between health checks
            validation_func: Optional function to validate connections
        """
        self._factory = factory
        self._min_size = min_size
        self._max_size = max_size
        self._acquire_timeout = acquire_timeout
        self._idle_timeout = idle_timeout
        self._health_check_interval = health_check_interval
        self._validation_func = validation_func

        self._pool: asyncio.Queue[PooledConnection[T]] = asyncio.Queue(maxsize=max_size)
        self._all_connections: List[PooledConnection[T]] = []
        self._in_use: Set[str] = set()
        self._lock = asyncio.Lock()
        self._initialized = False
        self._closed = False

        # Statistics
        self._total_created = 0
        self._total_destroyed = 0
        self._acquire_count = 0
        self._acquire_timeout_count = 0

    async def initialize(self) -> None:
        """Initialize pool with minimum connections."""
        async with self._lock:
            if self._initialized:
                return

            for _ in range(self._min_size):
                try:
                    conn = await self._create_connection()
                    await self._pool.put(conn)
                except Exception as e:
                    logger.error(f"Failed to create initial connection: {e}")

            self._initialized = True
            logger.info(
                f"Connection pool initialized with {self._pool.qsize()} connections "
                f"(min={self._min_size}, max={self._max_size})"
            )

    async def _create_connection(self) -> PooledConnection[T]:
        """Create a new pooled connection."""
        conn = await self._factory()
        pooled = PooledConnection(connection=conn)
        self._all_connections.append(pooled)
        self._total_created += 1
        return pooled

    async def acquire(self) -> PooledConnection[T]:
        """
        Acquire connection from pool.

        Returns:
            Pooled connection

        Raises:
            TimeoutError: If acquire times out
        """
        if self._closed:
            raise ConnectionError("Connection pool is closed")

        if not self._initialized:
            await self.initialize()

        self._acquire_count += 1
        start_time = time.time()

        try:
            # Try to get from pool with timeout
            try:
                pooled = await asyncio.wait_for(
                    self._pool.get(),
                    timeout=min(self._acquire_timeout / 2, 5.0),
                )
            except asyncio.TimeoutError:
                pooled = None

            # Validate existing connection or create new one
            if pooled is not None:
                # Validate connection
                is_valid = True
                if self._validation_func:
                    try:
                        is_valid = await self._validation_func(pooled.connection)
                    except Exception:
                        is_valid = False

                if is_valid and pooled.healthy:
                    pooled.last_used_at = time.time()
                    pooled.use_count += 1
                    self._in_use.add(pooled.connection_id)
                    return pooled
                else:
                    # Connection invalid, try to create new one
                    await self._destroy_connection(pooled)
                    pooled = None

            # Create new connection if under max
            async with self._lock:
                current_size = len(self._all_connections)
                if current_size < self._max_size:
                    pooled = await self._create_connection()
                    pooled.last_used_at = time.time()
                    pooled.use_count += 1
                    self._in_use.add(pooled.connection_id)
                    return pooled

            # Wait for connection to become available
            remaining_timeout = self._acquire_timeout - (time.time() - start_time)
            if remaining_timeout <= 0:
                self._acquire_timeout_count += 1
                raise TimeoutError(
                    f"Timeout acquiring connection after {self._acquire_timeout}s"
                )

            try:
                pooled = await asyncio.wait_for(
                    self._pool.get(),
                    timeout=remaining_timeout,
                )
                pooled.last_used_at = time.time()
                pooled.use_count += 1
                self._in_use.add(pooled.connection_id)
                return pooled
            except asyncio.TimeoutError:
                self._acquire_timeout_count += 1
                raise TimeoutError(
                    f"Timeout acquiring connection after {self._acquire_timeout}s"
                )

        except TimeoutError:
            raise
        except Exception as e:
            raise ConnectionError(f"Failed to acquire connection: {e}")

    async def release(self, pooled: PooledConnection[T], healthy: bool = True) -> None:
        """
        Release connection back to pool.

        Args:
            pooled: Pooled connection to release
            healthy: Whether connection is still healthy
        """
        if pooled.connection_id in self._in_use:
            self._in_use.discard(pooled.connection_id)

        if not healthy or self._closed:
            await self._destroy_connection(pooled)
            return

        pooled.healthy = healthy
        pooled.last_used_at = time.time()

        try:
            self._pool.put_nowait(pooled)
        except asyncio.QueueFull:
            # Pool is full, destroy connection
            await self._destroy_connection(pooled)

    async def _destroy_connection(self, pooled: PooledConnection[T]) -> None:
        """Destroy a connection and remove from tracking."""
        try:
            if hasattr(pooled.connection, 'close'):
                if asyncio.iscoroutinefunction(pooled.connection.close):
                    await pooled.connection.close()
                else:
                    pooled.connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

        async with self._lock:
            if pooled in self._all_connections:
                self._all_connections.remove(pooled)
        self._total_destroyed += 1

    async def cleanup_idle(self) -> int:
        """
        Remove idle connections exceeding timeout.

        Returns:
            Number of connections removed
        """
        removed = 0
        async with self._lock:
            current_size = len(self._all_connections)

            # Don't remove if at or below minimum
            if current_size <= self._min_size:
                return 0

            # Check idle connections
            now = time.time()
            to_remove: List[PooledConnection[T]] = []

            for pooled in self._all_connections:
                if pooled.connection_id not in self._in_use:
                    if pooled.idle_seconds > self._idle_timeout:
                        if current_size - len(to_remove) > self._min_size:
                            to_remove.append(pooled)

            for pooled in to_remove:
                await self._destroy_connection(pooled)
                removed += 1

        return removed

    async def close(self) -> None:
        """Close all connections in pool."""
        self._closed = True

        async with self._lock:
            # Close all connections
            while self._all_connections:
                pooled = self._all_connections.pop()
                try:
                    if hasattr(pooled.connection, 'close'):
                        if asyncio.iscoroutinefunction(pooled.connection.close):
                            await pooled.connection.close()
                        else:
                            pooled.connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")

            # Clear pool queue
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except asyncio.QueueEmpty:
                    break

            self._initialized = False

    @property
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "total": len(self._all_connections),
            "available": self._pool.qsize(),
            "in_use": len(self._in_use),
            "min_size": self._min_size,
            "max_size": self._max_size,
            "total_created": self._total_created,
            "total_destroyed": self._total_destroyed,
            "acquire_count": self._acquire_count,
            "acquire_timeout_count": self._acquire_timeout_count,
        }


# =============================================================================
# Retry Logic Implementation
# =============================================================================


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter_factor: float = 0.1,
    retryable_exceptions: Tuple[type, ...] = (Exception,),
    non_retryable_exceptions: Tuple[type, ...] = (),
) -> Callable:
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential calculation
        jitter_factor: Factor for delay randomization (0-1)
        retryable_exceptions: Tuple of exceptions that trigger retry
        non_retryable_exceptions: Tuple of exceptions that should not retry

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except non_retryable_exceptions as e:
                    # Don't retry these
                    raise
                except retryable_exceptions as e:
                    last_exception = e

                    # Check if error is retryable
                    if isinstance(e, ConnectorError) and not e.retryable:
                        raise

                    if attempt == max_retries:
                        logger.error(
                            f"All {max_retries} retry attempts exhausted for "
                            f"{func.__name__}: {e}"
                        )
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay,
                    )

                    # Add jitter
                    import random
                    jitter = delay * jitter_factor * random.random()
                    delay += jitter

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                        f"{func.__name__}: {e}. Retrying in {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)

            raise RetryExhaustedError(
                f"All retry attempts exhausted for {func.__name__}",
                details={"last_error": str(last_exception), "attempts": max_retries + 1},
            )

        return wrapper

    return decorator


# =============================================================================
# Metrics Collector
# =============================================================================


class MetricsCollector:
    """
    Collects and exposes Prometheus-compatible metrics.

    Tracks request counts, latencies, errors, and other operational metrics.
    """

    def __init__(
        self,
        connector_id: str,
        connector_type: ConnectorType,
        connector_name: str = "",
    ) -> None:
        """
        Initialize metrics collector.

        Args:
            connector_id: Connector identifier
            connector_type: Type of connector
            connector_name: Human-readable connector name
        """
        self._connector_id = connector_id
        self._connector_type = connector_type
        self._connector_name = connector_name

        # Request counters
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._retried_requests = 0
        self._rate_limited_requests = 0

        # Latencies (in ms)
        self._latencies: deque = deque(maxlen=1000)
        self._min_latency: float = float('inf')
        self._max_latency: float = 0.0

        # Throughput tracking
        self._request_timestamps: deque = deque(maxlen=1000)
        self._bytes_sent: deque = deque(maxlen=1000)
        self._bytes_received: deque = deque(maxlen=1000)

        # Error tracking
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._last_error_at: Optional[datetime] = None
        self._errors_by_type: Dict[str, int] = {}

        # Data quality tracking
        self._quality_scores: deque = deque(maxlen=100)
        self._validation_failures = 0

        # Reconnection tracking
        self._reconnection_attempts = 0
        self._successful_reconnections = 0
        self._failed_reconnections = 0

        self._lock = asyncio.Lock()
        self._start_time = time.time()

    async def record_request(
        self,
        success: bool,
        latency_ms: float,
        retried: bool = False,
        rate_limited: bool = False,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        bytes_sent: int = 0,
        bytes_received: int = 0,
    ) -> None:
        """
        Record request metrics.

        Args:
            success: Whether request succeeded
            latency_ms: Request latency in milliseconds
            retried: Whether request was retried
            rate_limited: Whether request was rate limited
            error: Error message if failed
            error_type: Type of error
            bytes_sent: Bytes sent in request
            bytes_received: Bytes received in response
        """
        async with self._lock:
            now = time.time()
            self._total_requests += 1
            self._request_timestamps.append(now)

            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1
                self._error_count += 1
                self._last_error = error
                self._last_error_at = datetime.utcnow()
                if error_type:
                    self._errors_by_type[error_type] = self._errors_by_type.get(error_type, 0) + 1

            if retried:
                self._retried_requests += 1

            if rate_limited:
                self._rate_limited_requests += 1

            # Update latency stats
            self._latencies.append(latency_ms)
            self._min_latency = min(self._min_latency, latency_ms)
            self._max_latency = max(self._max_latency, latency_ms)

            # Track throughput
            if bytes_sent > 0:
                self._bytes_sent.append((now, bytes_sent))
            if bytes_received > 0:
                self._bytes_received.append((now, bytes_received))

    async def record_data_quality(self, score: float, validation_failed: bool = False) -> None:
        """Record data quality metrics."""
        async with self._lock:
            self._quality_scores.append(score)
            if validation_failed:
                self._validation_failures += 1

    async def record_reconnection(self, success: bool) -> None:
        """Record reconnection attempt."""
        async with self._lock:
            self._reconnection_attempts += 1
            if success:
                self._successful_reconnections += 1
            else:
                self._failed_reconnections += 1

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _calculate_rate(self, timestamps: deque, window_seconds: float = 60.0) -> float:
        """Calculate rate per second over time window."""
        if not timestamps:
            return 0.0
        now = time.time()
        cutoff = now - window_seconds
        count = sum(1 for t in timestamps if t >= cutoff)
        return count / window_seconds

    def _calculate_byte_rate(self, data: deque, window_seconds: float = 60.0) -> float:
        """Calculate byte rate per second over time window."""
        if not data:
            return 0.0
        now = time.time()
        cutoff = now - window_seconds
        total_bytes = sum(b for t, b in data if t >= cutoff)
        return total_bytes / window_seconds

    async def get_snapshot(
        self,
        connection_state: ConnectionState = ConnectionState.DISCONNECTED,
        circuit_state: CircuitState = CircuitState.CLOSED,
        circuit_open_count: int = 0,
        circuit_half_open_count: int = 0,
        cache_stats: Optional[Dict[str, Any]] = None,
        connection_stats: Optional[Dict[str, Any]] = None,
        rate_limiter_remaining: int = 0,
        health_status: HealthStatus = HealthStatus.UNKNOWN,
        health_latency_ms: float = 0.0,
        consecutive_health_failures: int = 0,
    ) -> MetricsSnapshot:
        """
        Get current metrics snapshot.

        Returns:
            Metrics snapshot for Prometheus export
        """
        async with self._lock:
            latency_list = list(self._latencies)
            avg_latency = sum(latency_list) / len(latency_list) if latency_list else 0.0

            avg_quality = (
                sum(self._quality_scores) / len(self._quality_scores)
                if self._quality_scores else 1.0
            )

            cache_hit_rate = 0.0
            if cache_stats:
                cache_hit_rate = cache_stats.get("hit_rate", 0.0)

            return MetricsSnapshot(
                connector_id=self._connector_id,
                connector_type=self._connector_type,
                connector_name=self._connector_name,
                connection_state=connection_state,
                total_connections=connection_stats.get("total", 0) if connection_stats else 0,
                active_connections=connection_stats.get("in_use", 0) if connection_stats else 0,
                idle_connections=connection_stats.get("available", 0) if connection_stats else 0,
                connection_errors=connection_stats.get("acquire_timeout_count", 0) if connection_stats else 0,
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                retried_requests=self._retried_requests,
                rate_limited_requests=self._rate_limited_requests,
                avg_latency_ms=avg_latency,
                min_latency_ms=self._min_latency if self._min_latency != float('inf') else 0.0,
                max_latency_ms=self._max_latency,
                p50_latency_ms=self._percentile(latency_list, 50),
                p95_latency_ms=self._percentile(latency_list, 95),
                p99_latency_ms=self._percentile(latency_list, 99),
                requests_per_second=self._calculate_rate(self._request_timestamps),
                bytes_sent_per_second=self._calculate_byte_rate(self._bytes_sent),
                bytes_received_per_second=self._calculate_byte_rate(self._bytes_received),
                circuit_state=circuit_state,
                circuit_open_count=circuit_open_count,
                circuit_half_open_count=circuit_half_open_count,
                rate_limit_remaining=rate_limiter_remaining,
                cache_hits=cache_stats.get("hits", 0) if cache_stats else 0,
                cache_misses=cache_stats.get("misses", 0) if cache_stats else 0,
                cache_size=cache_stats.get("size", 0) if cache_stats else 0,
                cache_hit_rate=cache_hit_rate,
                health_status=health_status,
                health_check_latency_ms=health_latency_ms,
                consecutive_health_failures=consecutive_health_failures,
                avg_data_quality_score=avg_quality,
                data_validation_failures=self._validation_failures,
                error_count=self._error_count,
                last_error=self._last_error,
                last_error_at=self._last_error_at,
                reconnection_attempts=self._reconnection_attempts,
                successful_reconnections=self._successful_reconnections,
                failed_reconnections=self._failed_reconnections,
            )


# =============================================================================
# Audit Logger
# =============================================================================


class AuditLogger:
    """
    Audit logger for compliance tracking.

    Logs all operations with sanitized data for regulatory compliance.
    """

    def __init__(
        self,
        connector_id: str,
        connector_type: ConnectorType,
        enabled: bool = True,
    ) -> None:
        """
        Initialize audit logger.

        Args:
            connector_id: Connector identifier
            connector_type: Type of connector
            enabled: Whether audit logging is enabled
        """
        self._connector_id = connector_id
        self._connector_type = connector_type
        self._enabled = enabled
        self._logger = logging.getLogger(f"audit.{connector_type.value}")

        # Sensitive field patterns to sanitize
        self._sensitive_fields = {
            "password", "secret", "token", "key", "credential",
            "authorization", "api_key", "apikey", "auth", "bearer",
            "private", "certificate", "cert", "pem", "ssh",
        }

    def _sanitize_data(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Sanitize sensitive data from dictionary.

        Args:
            data: Data to sanitize

        Returns:
            Sanitized data
        """
        if data is None:
            return None

        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(field in key_lower for field in self._sensitive_fields):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    async def log_operation(
        self,
        operation: str,
        status: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        target_system: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Log an operation for audit purposes.

        Args:
            operation: Operation name
            status: Operation status
            user_id: Optional user identifier
            source_ip: Optional source IP
            target_system: Optional target system
            request_data: Optional request data (will be sanitized)
            response_summary: Optional response summary
            error_message: Optional error message
            duration_ms: Optional duration in milliseconds
            trace_id: Optional distributed trace ID
            span_id: Optional distributed span ID

        Returns:
            Audit log entry
        """
        entry = AuditLogEntry(
            connector_id=self._connector_id,
            connector_type=self._connector_type,
            operation=operation,
            status=status,
            user_id=user_id,
            source_ip=source_ip,
            target_system=target_system,
            request_data=self._sanitize_data(request_data) if self._enabled else None,
            response_summary=response_summary,
            error_message=error_message,
            duration_ms=duration_ms,
            trace_id=trace_id,
            span_id=span_id,
        )

        if not self._enabled:
            return entry

        # Log to audit logger
        log_data = entry.model_dump(exclude_none=True)
        log_data["timestamp"] = entry.timestamp.isoformat()

        if status == "success":
            self._logger.info(json.dumps(log_data))
        elif status == "failure":
            self._logger.error(json.dumps(log_data))
        else:
            self._logger.warning(json.dumps(log_data))

        return entry


# =============================================================================
# Data Validator
# =============================================================================


class DataValidator:
    """
    Validates data quality and consistency.

    Provides comprehensive data quality scoring and validation.
    """

    def __init__(
        self,
        quality_threshold: float = 0.6,
        required_fields: Optional[List[str]] = None,
        field_validators: Optional[Dict[str, Callable[[Any], bool]]] = None,
    ) -> None:
        """
        Initialize data validator.

        Args:
            quality_threshold: Minimum acceptable quality score
            required_fields: List of required field names
            field_validators: Dict of field name to validation function
        """
        self._quality_threshold = quality_threshold
        self._required_fields = required_fields or []
        self._field_validators = field_validators or {}

    def validate(self, data: Dict[str, Any]) -> DataQualityResult:
        """
        Validate data and compute quality score.

        Args:
            data: Data to validate

        Returns:
            Data quality result
        """
        issues: List[str] = []
        recommendations: List[str] = []

        # Check completeness
        completeness_score = self._check_completeness(data, issues, recommendations)

        # Check validity
        validity_score = self._check_validity(data, issues, recommendations)

        # Check consistency
        consistency_score = self._check_consistency(data, issues, recommendations)

        # Check timeliness
        timeliness_score = self._check_timeliness(data, issues, recommendations)

        # Calculate overall score (weighted average)
        weights = {
            "completeness": 0.30,
            "validity": 0.35,
            "consistency": 0.20,
            "timeliness": 0.15,
        }
        quality_score = (
            completeness_score * weights["completeness"] +
            validity_score * weights["validity"] +
            consistency_score * weights["consistency"] +
            timeliness_score * weights["timeliness"]
        )

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
            completeness_score=completeness_score,
            validity_score=validity_score,
            consistency_score=consistency_score,
            timeliness_score=timeliness_score,
            issues=issues,
            recommendations=recommendations,
        )

    def _check_completeness(
        self,
        data: Dict[str, Any],
        issues: List[str],
        recommendations: List[str],
    ) -> float:
        """Check data completeness."""
        if not self._required_fields:
            return 1.0

        present = sum(1 for field in self._required_fields if field in data and data[field] is not None)
        score = present / len(self._required_fields)

        missing = [f for f in self._required_fields if f not in data or data[f] is None]
        if missing:
            issues.append(f"Missing required fields: {', '.join(missing)}")
            recommendations.append("Ensure all required fields are populated")

        return score

    def _check_validity(
        self,
        data: Dict[str, Any],
        issues: List[str],
        recommendations: List[str],
    ) -> float:
        """Check data validity using field validators."""
        if not self._field_validators:
            return 1.0

        valid_count = 0
        total_count = 0

        for field, validator in self._field_validators.items():
            if field in data:
                total_count += 1
                try:
                    if validator(data[field]):
                        valid_count += 1
                    else:
                        issues.append(f"Field '{field}' failed validation")
                except Exception as e:
                    issues.append(f"Field '{field}' validation error: {str(e)}")

        return valid_count / total_count if total_count > 0 else 1.0

    def _check_consistency(
        self,
        data: Dict[str, Any],
        issues: List[str],
        recommendations: List[str],
    ) -> float:
        """Check data consistency (type consistency, format consistency)."""
        # Basic type consistency check
        score = 1.0

        for key, value in data.items():
            # Check for mixed None values in lists
            if isinstance(value, list) and value:
                non_none = [v for v in value if v is not None]
                if len(non_none) != len(value):
                    score *= 0.95
                    issues.append(f"Field '{key}' contains None values in list")

        return score

    def _check_timeliness(
        self,
        data: Dict[str, Any],
        issues: List[str],
        recommendations: List[str],
    ) -> float:
        """Check data timeliness (how recent the data is)."""
        # Look for timestamp fields
        timestamp_fields = ["timestamp", "created_at", "updated_at", "time", "date"]

        for field in timestamp_fields:
            if field in data:
                try:
                    if isinstance(data[field], datetime):
                        age = datetime.utcnow() - data[field]
                        if age > timedelta(hours=24):
                            return 0.7
                        elif age > timedelta(hours=1):
                            return 0.9
                        return 1.0
                except Exception:
                    pass

        return 1.0

    def validate_and_raise(self, data: Dict[str, Any]) -> DataQualityResult:
        """
        Validate data and raise exception if below threshold.

        Args:
            data: Data to validate

        Returns:
            Data quality result

        Raises:
            DataQualityError: If quality is below threshold
        """
        result = self.validate(data)

        if result.quality_score < self._quality_threshold:
            raise DataQualityError(
                f"Data quality score {result.quality_score:.2f} below threshold "
                f"{self._quality_threshold:.2f}",
                quality_result=result,
            )

        return result


# =============================================================================
# Abstract Base Connector
# =============================================================================


class BaseConnector(ABC):
    """
    Abstract base class for all predictive maintenance connectors.

    Provides common functionality:
    - Connection state management
    - Connection pooling
    - Retry logic with exponential backoff
    - Circuit breaker pattern for fault tolerance
    - Health monitoring with periodic checks
    - Automatic reconnection
    - Caching layer
    - Rate limiting
    - Data validation and normalization
    - Error handling
    - Audit logging for compliance
    - Prometheus metrics

    All concrete connectors must implement the abstract methods.
    """

    def __init__(self, config: BaseConnectorConfig) -> None:
        """
        Initialize base connector.

        Args:
            config: Connector configuration
        """
        self._config = config
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = asyncio.Lock()

        # Initialize circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            success_threshold=config.circuit_breaker_success_threshold,
            timeout_seconds=config.circuit_breaker_timeout_seconds,
            half_open_max_requests=config.circuit_breaker_half_open_max_requests,
            name=f"{config.connector_type.value}_{config.connector_id[:8]}",
        )

        # Initialize cache
        self._cache: LRUCache[Any] = LRUCache(
            max_size=config.cache_max_size,
            default_ttl=config.cache_ttl_seconds,
        )

        # Initialize rate limiter
        if config.rate_limit_strategy == RateLimitStrategy.TOKEN_BUCKET:
            self._rate_limiter = TokenBucketRateLimiter(
                rate=config.rate_limit_requests_per_second,
                burst_size=config.rate_limit_burst_size,
            )
        else:
            self._rate_limiter = SlidingWindowRateLimiter(
                rate=config.rate_limit_requests_per_second,
            )

        # Initialize metrics collector
        self._metrics = MetricsCollector(
            connector_id=config.connector_id,
            connector_type=config.connector_type,
            connector_name=config.connector_name,
        )

        # Initialize audit logger
        self._audit_logger = AuditLogger(
            connector_id=config.connector_id,
            connector_type=config.connector_type,
            enabled=config.audit_logging_enabled,
        )

        # Initialize data validator
        self._data_validator = DataValidator(
            quality_threshold=config.data_quality_threshold,
        )

        # Configure logging
        self._logger = logging.getLogger(
            f"connector.{config.connector_type.value}.{config.connector_id[:8]}"
        )
        self._logger.setLevel(getattr(logging, config.log_level))

        # Health check state
        self._last_health_check: Optional[HealthCheckResult] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._consecutive_health_failures = 0

        # Reconnection state
        self._reconnection_task: Optional[asyncio.Task] = None
        self._reconnection_attempt = 0

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

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
    def is_healthy(self) -> bool:
        """Check if connector is healthy."""
        if self._last_health_check is None:
            return False
        return self._last_health_check.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @property
    def circuit_state(self) -> CircuitState:
        """Get circuit breaker state."""
        return self._circuit_breaker.state

    # -------------------------------------------------------------------------
    # Abstract Methods - Must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the target system.

        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the target system.

        Should gracefully close all connections and release resources.
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on the connection.

        Returns:
            Health check result with status and details
        """
        pass

    @abstractmethod
    async def validate_configuration(self) -> bool:
        """
        Validate connector configuration.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        pass

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    async def _set_state(self, new_state: ConnectionState) -> None:
        """Set connection state with logging."""
        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            if old_state != new_state:
                self._logger.info(
                    f"Connection state changed: {old_state.value} -> {new_state.value}"
                )

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize the connector.

        Validates configuration and establishes initial connection.
        """
        self._logger.info(
            f"Initializing connector {self._config.connector_name} "
            f"({self._config.connector_type.value})"
        )

        # Validate configuration
        await self.validate_configuration()

        # Connect
        await self._set_state(ConnectionState.CONNECTING)
        try:
            await self.connect()
            await self._set_state(ConnectionState.CONNECTED)
        except Exception as e:
            await self._set_state(ConnectionState.ERROR)
            raise

        # Start background tasks
        if self._config.health_check_enabled:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop()
        )

        # Warm up cache if enabled
        if self._config.cache_warmup_enabled:
            await self._warmup_cache()

        self._logger.info(
            f"Connector {self._config.connector_name} initialized successfully"
        )

    async def shutdown(self) -> None:
        """
        Shutdown the connector gracefully.

        Stops background tasks, disconnects, and releases resources.
        """
        self._logger.info(
            f"Shutting down connector {self._config.connector_name}"
        )

        # Cancel background tasks
        for task in [self._health_check_task, self._reconnection_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Disconnect
        await self.disconnect()
        await self._set_state(ConnectionState.DISCONNECTED)

        # Clear cache
        await self._cache.clear()

        self._logger.info(
            f"Connector {self._config.connector_name} shutdown complete"
        )

    async def _warmup_cache(self) -> None:
        """Warm up cache with frequently accessed data. Override in subclass."""
        pass

    # -------------------------------------------------------------------------
    # Background Tasks
    # -------------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                start_time = time.time()
                try:
                    self._last_health_check = await asyncio.wait_for(
                        self.health_check(),
                        timeout=self._config.health_check_timeout_seconds,
                    )
                    latency_ms = (time.time() - start_time) * 1000

                    if self._last_health_check.status == HealthStatus.HEALTHY:
                        self._consecutive_health_failures = 0
                    elif self._last_health_check.status == HealthStatus.UNHEALTHY:
                        self._consecutive_health_failures += 1
                        self._logger.warning(
                            f"Health check failed ({self._consecutive_health_failures}): "
                            f"{self._last_health_check.message}"
                        )

                        # Trigger reconnection if threshold exceeded
                        if self._consecutive_health_failures >= self._config.health_check_failure_threshold:
                            await self._trigger_reconnection()

                except asyncio.TimeoutError:
                    self._consecutive_health_failures += 1
                    self._logger.warning(
                        f"Health check timed out ({self._consecutive_health_failures})"
                    )
                    self._last_health_check = HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        message="Health check timed out",
                        consecutive_failures=self._consecutive_health_failures,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health check error: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup (cache, idle connections)."""
        cleanup_interval = 60.0  # Run cleanup every minute

        while True:
            try:
                await asyncio.sleep(cleanup_interval)

                # Cleanup expired cache entries
                expired = await self._cache.cleanup_expired()
                if expired > 0:
                    self._logger.debug(f"Cleaned up {expired} expired cache entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Cleanup error: {e}")

    # -------------------------------------------------------------------------
    # Reconnection Logic
    # -------------------------------------------------------------------------

    async def _trigger_reconnection(self) -> None:
        """Trigger reconnection attempt."""
        if not self._config.reconnect_enabled:
            return

        if self._reconnection_task and not self._reconnection_task.done():
            return  # Already reconnecting

        self._reconnection_task = asyncio.create_task(
            self._reconnection_loop()
        )

    async def _reconnection_loop(self) -> None:
        """Reconnection loop with exponential backoff."""
        await self._set_state(ConnectionState.RECONNECTING)
        self._reconnection_attempt = 0

        while self._reconnection_attempt < self._config.reconnect_max_attempts:
            self._reconnection_attempt += 1

            # Calculate delay with exponential backoff
            delay = min(
                self._config.reconnect_initial_delay_seconds * (2 ** (self._reconnection_attempt - 1)),
                self._config.reconnect_max_delay_seconds,
            )

            self._logger.info(
                f"Reconnection attempt {self._reconnection_attempt}/"
                f"{self._config.reconnect_max_attempts} in {delay:.1f}s"
            )

            await asyncio.sleep(delay)

            try:
                await self.disconnect()
                await self.connect()
                await self._set_state(ConnectionState.CONNECTED)
                self._consecutive_health_failures = 0
                await self._metrics.record_reconnection(success=True)
                self._logger.info("Reconnection successful")
                return

            except Exception as e:
                await self._metrics.record_reconnection(success=False)
                self._logger.warning(
                    f"Reconnection attempt {self._reconnection_attempt} failed: {e}"
                )

        await self._set_state(ConnectionState.ERROR)
        self._logger.error(
            f"All {self._config.reconnect_max_attempts} reconnection attempts failed"
        )

    # -------------------------------------------------------------------------
    # Request Execution with Protection
    # -------------------------------------------------------------------------

    async def execute_with_protection(
        self,
        operation: Callable[..., Coroutine[Any, Any, T]],
        *args,
        operation_name: str = "operation",
        use_cache: bool = False,
        cache_key: Optional[str] = None,
        cache_ttl: Optional[int] = None,
        validate_result: bool = False,
        **kwargs,
    ) -> T:
        """
        Execute operation with full protection (circuit breaker, rate limiting, retry).

        Args:
            operation: Async operation to execute
            *args: Positional arguments for operation
            operation_name: Name for logging/metrics
            use_cache: Whether to use caching
            cache_key: Optional cache key (auto-generated if not provided)
            cache_ttl: Optional cache TTL override
            validate_result: Whether to validate result data quality
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result

        Raises:
            CircuitOpenError: If circuit is open
            RateLimitError: If rate limited
            RetryExhaustedError: If all retries fail
        """
        # Check circuit breaker
        if self._config.circuit_breaker_enabled:
            if not await self._circuit_breaker.can_execute():
                raise CircuitOpenError(
                    f"Circuit breaker is open for connector {self._config.connector_name}",
                    connector_id=self._config.connector_id,
                    connector_type=self._config.connector_type,
                )

        # Check rate limit
        if self._config.rate_limit_enabled:
            if not await self._rate_limiter.acquire():
                raise RateLimitError(
                    f"Rate limit exceeded for connector {self._config.connector_name}",
                    connector_id=self._config.connector_id,
                    connector_type=self._config.connector_type,
                )

        # Check cache
        if use_cache and self._config.cache_enabled:
            if cache_key is None:
                cache_key = self._generate_cache_key(operation_name, *args, **kwargs)

            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached

        start_time = time.time()
        last_exception: Optional[Exception] = None

        # Execute with retry
        for attempt in range(self._config.max_retries + 1):
            try:
                result = await operation(*args, **kwargs)

                # Record success
                latency_ms = (time.time() - start_time) * 1000
                await self._circuit_breaker.record_success()
                await self._metrics.record_request(
                    success=True,
                    latency_ms=latency_ms,
                    retried=attempt > 0,
                )

                # Validate result if requested
                if validate_result and isinstance(result, dict):
                    quality_result = self._data_validator.validate(result)
                    await self._metrics.record_data_quality(
                        score=quality_result.quality_score,
                        validation_failed=quality_result.quality_score < self._config.data_quality_threshold,
                    )

                # Cache result
                if use_cache and self._config.cache_enabled and cache_key:
                    await self._cache.set(cache_key, result, cache_ttl)

                # Audit log
                await self._audit_logger.log_operation(
                    operation=operation_name,
                    status="success",
                    duration_ms=latency_ms,
                )

                return result

            except Exception as e:
                last_exception = e

                # Check if retryable
                retryable = True
                if isinstance(e, ConnectorError):
                    retryable = e.retryable

                if not retryable or attempt == self._config.max_retries:
                    # Record failure
                    latency_ms = (time.time() - start_time) * 1000
                    await self._circuit_breaker.record_failure(e)
                    await self._metrics.record_request(
                        success=False,
                        latency_ms=latency_ms,
                        retried=attempt > 0,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

                    # Audit log
                    await self._audit_logger.log_operation(
                        operation=operation_name,
                        status="failure",
                        error_message=str(e),
                        duration_ms=latency_ms,
                    )

                    if not retryable:
                        raise
                    break

                # Calculate retry delay
                delay = min(
                    self._config.retry_base_delay_seconds * (self._config.retry_exponential_base ** attempt),
                    self._config.retry_max_delay_seconds,
                )

                # Add jitter
                import random
                jitter = delay * self._config.retry_jitter_factor * random.random()
                delay += jitter

                self._logger.warning(
                    f"Attempt {attempt + 1}/{self._config.max_retries + 1} failed for "
                    f"{operation_name}: {e}. Retrying in {delay:.2f}s"
                )

                await asyncio.sleep(delay)

        raise RetryExhaustedError(
            f"All retry attempts exhausted for {operation_name}",
            connector_id=self._config.connector_id,
            connector_type=self._config.connector_type,
            details={"last_error": str(last_exception), "attempts": self._config.max_retries + 1},
        )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _generate_cache_key(self, operation: str, *args, **kwargs) -> str:
        """
        Generate cache key from operation and arguments.

        Args:
            operation: Operation name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        key_data = json.dumps({
            "operation": operation,
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        }, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    async def get_metrics(self) -> MetricsSnapshot:
        """
        Get current metrics snapshot.

        Returns:
            Metrics snapshot for Prometheus export
        """
        return await self._metrics.get_snapshot(
            connection_state=self._state,
            circuit_state=self._circuit_breaker.state,
            circuit_open_count=self._circuit_breaker.open_count,
            circuit_half_open_count=self._circuit_breaker.half_open_count,
            cache_stats=self._cache.stats,
            rate_limiter_remaining=self._rate_limiter.remaining,
            health_status=self._last_health_check.status if self._last_health_check else HealthStatus.UNKNOWN,
            health_latency_ms=self._last_health_check.latency_ms or 0.0 if self._last_health_check else 0.0,
            consecutive_health_failures=self._consecutive_health_failures,
        )

    async def get_health_status(self) -> HealthCheckResult:
        """
        Get latest health status.

        Returns:
            Latest health check result
        """
        if self._last_health_check is None:
            return await self.health_check()
        return self._last_health_check

    async def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state."""
        await self._circuit_breaker.reset()
        self._logger.info(
            f"Circuit breaker reset for connector {self._config.connector_name}"
        )

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        await self._cache.clear()
        self._logger.info(
            f"Cache cleared for connector {self._config.connector_name}"
        )

    def __repr__(self) -> str:
        """String representation of connector."""
        return (
            f"<{self.__class__.__name__}("
            f"id={self._config.connector_id[:8]}, "
            f"name={self._config.connector_name}, "
            f"type={self._config.connector_type.value}, "
            f"state={self._state.value})>"
        )


# =============================================================================
# Context Manager Support
# =============================================================================


class ConnectorContextManager:
    """
    Async context manager for connector lifecycle.

    Usage:
        async with ConnectorContextManager(connector) as conn:
            await conn.some_operation()
    """

    def __init__(self, connector: BaseConnector) -> None:
        """
        Initialize context manager.

        Args:
            connector: Connector instance
        """
        self._connector = connector

    async def __aenter__(self) -> BaseConnector:
        """Enter context - initialize connector."""
        await self._connector.initialize()
        return self._connector

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context - shutdown connector."""
        await self._connector.shutdown()


# =============================================================================
# Factory Function
# =============================================================================


def create_connector_config(
    connector_name: str,
    connector_type: ConnectorType,
    **overrides,
) -> BaseConnectorConfig:
    """
    Factory function to create connector configuration with sensible defaults.

    Args:
        connector_name: Human-readable name
        connector_type: Type of connector
        **overrides: Configuration overrides

    Returns:
        Connector configuration
    """
    return BaseConnectorConfig(
        connector_name=connector_name,
        connector_type=connector_type,
        **overrides,
    )
