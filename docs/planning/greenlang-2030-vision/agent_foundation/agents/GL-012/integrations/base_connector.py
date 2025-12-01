# -*- coding: utf-8 -*-
"""
Base Connector Module for GL-012 STEAMQUAL (SteamQualityController).

Provides abstract base class with common functionality for all steam quality
monitoring and control connectors including connection pooling, retry logic,
circuit breaker pattern, health monitoring, caching, and error handling.

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
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
    Tuple,
)
import asyncio
import functools
import hashlib
import json
import logging
import time
import uuid

from pydantic import BaseModel, Field, ConfigDict

# Configure module logger
logger = logging.getLogger(__name__)

# Generic type for cached values
T = TypeVar("T")


# =============================================================================
# Enumerations
# =============================================================================


class ConnectionState(str, Enum):
    """Connection state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class CircuitState(str, Enum):
    """Circuit breaker state enumeration."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class HealthStatus(str, Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ConnectorType(str, Enum):
    """Types of connectors supported for GL-012 STEAMQUAL."""

    STEAM_QUALITY_METER = "steam_quality_meter"
    CONTROL_VALVE = "control_valve"
    DESUPERHEATER = "desuperheater"
    SCADA = "scada"
    TEMPERATURE_SENSOR = "temperature_sensor"
    PRESSURE_SENSOR = "pressure_sensor"
    FLOW_METER = "flow_meter"


class ProtocolType(str, Enum):
    """Supported communication protocols."""

    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPC_UA = "opc_ua"
    OPC_DA = "opc_da"
    HART = "hart"
    PROFIBUS = "profibus"
    ETHERNET_IP = "ethernet_ip"
    PROFINET = "profinet"


# =============================================================================
# Pydantic Models
# =============================================================================


class BaseConnectorConfig(BaseModel):
    """Base configuration for all GL-012 STEAMQUAL connectors."""

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

    # Connection settings
    host: str = Field(
        default="localhost",
        description="Host address for the connection"
    )
    port: int = Field(
        default=502,
        ge=1,
        le=65535,
        description="Port number for the connection"
    )
    protocol: ProtocolType = Field(
        default=ProtocolType.MODBUS_TCP,
        description="Communication protocol"
    )
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

    # TLS/Security settings
    tls_enabled: bool = Field(
        default=False,
        description="Whether TLS encryption is enabled"
    )
    cert_path: Optional[str] = Field(
        default=None,
        description="Path to TLS certificate"
    )
    key_path: Optional[str] = Field(
        default=None,
        description="Path to TLS private key"
    )
    ca_path: Optional[str] = Field(
        default=None,
        description="Path to CA certificate"
    )

    # Authentication
    username: Optional[str] = Field(
        default=None,
        description="Username for authentication"
    )
    # Note: Password should be retrieved from secure vault, never stored in config

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

    # Health check settings
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
    state: ConnectionState = Field(
        default=ConnectionState.DISCONNECTED,
        description="Current connection state"
    )
    remote_address: Optional[str] = Field(
        default=None,
        description="Remote address of the connection"
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of errors on this connection"
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


class MetricsSnapshot(BaseModel):
    """Snapshot of connector metrics."""

    model_config = ConfigDict(frozen=True)

    connector_id: str = Field(..., description="Connector identifier")
    connector_type: ConnectorType = Field(..., description="Connector type")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Metrics timestamp"
    )

    # Connection metrics
    total_connections: int = Field(default=0, ge=0)
    active_connections: int = Field(default=0, ge=0)
    idle_connections: int = Field(default=0, ge=0)

    # Request metrics
    total_requests: int = Field(default=0, ge=0)
    successful_requests: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    retried_requests: int = Field(default=0, ge=0)

    # Latency metrics
    avg_latency_ms: float = Field(default=0.0, ge=0)
    p50_latency_ms: float = Field(default=0.0, ge=0)
    p95_latency_ms: float = Field(default=0.0, ge=0)
    p99_latency_ms: float = Field(default=0.0, ge=0)

    # Circuit breaker metrics
    circuit_state: CircuitState = Field(default=CircuitState.CLOSED)
    circuit_open_count: int = Field(default=0, ge=0)

    # Cache metrics
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    cache_size: int = Field(default=0, ge=0)

    # Error metrics
    error_count: int = Field(default=0, ge=0)
    last_error: Optional[str] = Field(default=None)
    last_error_at: Optional[datetime] = Field(default=None)


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
    ) -> None:
        super().__init__(message)
        self.message = message
        self.connector_id = connector_id
        self.connector_type = connector_type
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class ConnectionError(ConnectorError):
    """Error establishing or maintaining connection."""
    pass


class AuthenticationError(ConnectorError):
    """Authentication or authorization error."""
    pass


class TimeoutError(ConnectorError):
    """Operation timed out."""
    pass


class ValidationError(ConnectorError):
    """Data validation error."""
    pass


class CircuitOpenError(ConnectorError):
    """Circuit breaker is open, requests rejected."""
    pass


class RetryExhaustedError(ConnectorError):
    """All retry attempts exhausted."""
    pass


class ConfigurationError(ConnectorError):
    """Configuration error."""
    pass


class DataQualityError(ConnectorError):
    """Data quality validation failed."""
    pass


class SafetyInterlockError(ConnectorError):
    """Safety interlock triggered."""
    pass


class CalibrationError(ConnectorError):
    """Calibration operation failed."""
    pass


class CommunicationError(ConnectorError):
    """Communication with device failed."""
    pass


# =============================================================================
# Cache Implementation
# =============================================================================


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL support."""

    key: str
    value: T
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 300
    access_count: int = 0
    last_accessed_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    def access(self) -> T:
        """Access the cached value."""
        self.access_count += 1
        self.last_accessed_at = time.time()
        return self.value


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300) -> None:
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

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
                self._misses += 1
                return None

            self._hits += 1
            return entry.access()

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
        """
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                await self._evict_lru()

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl or self._default_ttl,
            )

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
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed_at
        )
        del self._cache[lru_key]

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
        }


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
        name: str = "circuit_breaker",
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            success_threshold: Successes to close circuit from half-open
            timeout_seconds: Time before attempting to close circuit
            name: Name for logging
        """
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._timeout_seconds = timeout_seconds
        self._name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._open_count = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def open_count(self) -> int:
        """Get number of times circuit has opened."""
        return self._open_count

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
                        self._success_count = 0
                        logger.info(
                            f"Circuit breaker '{self._name}' transitioning to HALF_OPEN"
                        )
                        return True
                return False

            # HALF_OPEN state - allow limited requests
            return True

    async def record_success(self) -> None:
        """Record successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(
                        f"Circuit breaker '{self._name}' closed after "
                        f"{self._success_count} successes"
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """
        Record failed request.

        Args:
            error: Optional exception that caused the failure
        """
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Immediately open on failure in half-open state
                self._state = CircuitState.OPEN
                self._open_count += 1
                logger.warning(
                    f"Circuit breaker '{self._name}' opened from HALF_OPEN "
                    f"after failure: {error}"
                )
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    self._open_count += 1
                    logger.warning(
                        f"Circuit breaker '{self._name}' opened after "
                        f"{self._failure_count} failures: {error}"
                    )

    async def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None


# =============================================================================
# Connection Pool Implementation
# =============================================================================


class ConnectionPool(Generic[T]):
    """
    Generic async connection pool.

    Manages a pool of reusable connections with health checking
    and automatic reconnection.
    """

    def __init__(
        self,
        factory: Callable[[], T],
        min_size: int = 1,
        max_size: int = 10,
        acquire_timeout: float = 30.0,
        health_check_interval: float = 30.0,
    ) -> None:
        """
        Initialize connection pool.

        Args:
            factory: Factory function to create new connections
            min_size: Minimum pool size
            max_size: Maximum pool size
            acquire_timeout: Timeout for acquiring connection
            health_check_interval: Interval between health checks
        """
        self._factory = factory
        self._min_size = min_size
        self._max_size = max_size
        self._acquire_timeout = acquire_timeout
        self._health_check_interval = health_check_interval

        self._pool: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self._connections: List[T] = []
        self._in_use: int = 0
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize pool with minimum connections."""
        async with self._lock:
            if self._initialized:
                return

            for _ in range(self._min_size):
                conn = await asyncio.to_thread(self._factory)
                self._connections.append(conn)
                await self._pool.put(conn)

            self._initialized = True
            logger.info(
                f"Connection pool initialized with {self._min_size} connections"
            )

    async def acquire(self) -> T:
        """
        Acquire connection from pool.

        Returns:
            Connection from pool

        Raises:
            TimeoutError: If acquire times out
        """
        if not self._initialized:
            await self.initialize()

        try:
            conn = await asyncio.wait_for(
                self._pool.get(),
                timeout=self._acquire_timeout,
            )
            async with self._lock:
                self._in_use += 1
            return conn
        except asyncio.TimeoutError:
            # Try to create new connection if under max
            async with self._lock:
                if len(self._connections) < self._max_size:
                    conn = await asyncio.to_thread(self._factory)
                    self._connections.append(conn)
                    self._in_use += 1
                    return conn

            raise TimeoutError(
                f"Timeout acquiring connection after {self._acquire_timeout}s"
            )

    async def release(self, conn: T) -> None:
        """
        Release connection back to pool.

        Args:
            conn: Connection to release
        """
        async with self._lock:
            self._in_use -= 1
        await self._pool.put(conn)

    async def close(self) -> None:
        """Close all connections in pool."""
        async with self._lock:
            while not self._pool.empty():
                conn = await self._pool.get()
                if hasattr(conn, "close"):
                    if asyncio.iscoroutinefunction(conn.close):
                        await conn.close()
                    else:
                        conn.close()

            self._connections.clear()
            self._initialized = False

    @property
    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "total": len(self._connections),
            "available": self._pool.qsize(),
            "in_use": self._in_use,
            "max_size": self._max_size,
        }


# =============================================================================
# Retry Logic Implementation
# =============================================================================


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential calculation
        retryable_exceptions: Tuple of exceptions that trigger retry

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
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"All {max_retries} retry attempts exhausted for "
                            f"{func.__name__}: {e}"
                        )
                        break

                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay,
                    )
                    # Add jitter (10-20% of delay)
                    jitter = delay * (0.1 + 0.1 * (hash(str(e)) % 10) / 10)
                    delay += jitter

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                        f"{func.__name__}: {e}. Retrying in {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)

            raise RetryExhaustedError(
                f"All retry attempts exhausted for {func.__name__}",
                details={"last_error": str(last_exception)},
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

    def __init__(self, connector_id: str, connector_type: ConnectorType) -> None:
        """
        Initialize metrics collector.

        Args:
            connector_id: Connector identifier
            connector_type: Type of connector
        """
        self._connector_id = connector_id
        self._connector_type = connector_type

        # Counters
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._retried_requests = 0

        # Latencies (in ms)
        self._latencies: List[float] = []
        self._max_latency_samples = 1000

        # Errors
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._last_error_at: Optional[datetime] = None

        self._lock = asyncio.Lock()

    async def record_request(
        self,
        success: bool,
        latency_ms: float,
        retried: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """
        Record request metrics.

        Args:
            success: Whether request succeeded
            latency_ms: Request latency in milliseconds
            retried: Whether request was retried
            error: Error message if failed
        """
        async with self._lock:
            self._total_requests += 1

            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1
                self._error_count += 1
                self._last_error = error
                self._last_error_at = datetime.utcnow()

            if retried:
                self._retried_requests += 1

            # Store latency
            self._latencies.append(latency_ms)
            if len(self._latencies) > self._max_latency_samples:
                self._latencies = self._latencies[-self._max_latency_samples:]

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    async def get_snapshot(
        self,
        circuit_state: CircuitState = CircuitState.CLOSED,
        circuit_open_count: int = 0,
        cache_stats: Optional[Dict[str, Any]] = None,
        connection_stats: Optional[Dict[str, int]] = None,
    ) -> MetricsSnapshot:
        """
        Get current metrics snapshot.

        Args:
            circuit_state: Current circuit breaker state
            circuit_open_count: Number of times circuit opened
            cache_stats: Optional cache statistics
            connection_stats: Optional connection pool statistics

        Returns:
            Metrics snapshot
        """
        async with self._lock:
            avg_latency = (
                sum(self._latencies) / len(self._latencies)
                if self._latencies else 0.0
            )

            return MetricsSnapshot(
                connector_id=self._connector_id,
                connector_type=self._connector_type,
                total_connections=connection_stats.get("total", 0) if connection_stats else 0,
                active_connections=connection_stats.get("in_use", 0) if connection_stats else 0,
                idle_connections=connection_stats.get("available", 0) if connection_stats else 0,
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                retried_requests=self._retried_requests,
                avg_latency_ms=avg_latency,
                p50_latency_ms=self._percentile(self._latencies, 50),
                p95_latency_ms=self._percentile(self._latencies, 95),
                p99_latency_ms=self._percentile(self._latencies, 99),
                circuit_state=circuit_state,
                circuit_open_count=circuit_open_count,
                cache_hits=cache_stats.get("hits", 0) if cache_stats else 0,
                cache_misses=cache_stats.get("misses", 0) if cache_stats else 0,
                cache_size=cache_stats.get("size", 0) if cache_stats else 0,
                error_count=self._error_count,
                last_error=self._last_error,
                last_error_at=self._last_error_at,
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
            "authorization", "api_key", "apikey", "auth",
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
            else:
                sanitized[key] = value

        return sanitized

    async def log_operation(
        self,
        operation: str,
        status: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> AuditLogEntry:
        """
        Log an operation for audit purposes.

        Args:
            operation: Operation name
            status: Operation status
            user_id: Optional user identifier
            source_ip: Optional source IP
            request_data: Optional request data (will be sanitized)
            response_summary: Optional response summary
            error_message: Optional error message
            duration_ms: Optional duration in milliseconds

        Returns:
            Audit log entry
        """
        if not self._enabled:
            return AuditLogEntry(
                connector_id=self._connector_id,
                connector_type=self._connector_type,
                operation=operation,
                status=status,
            )

        entry = AuditLogEntry(
            connector_id=self._connector_id,
            connector_type=self._connector_type,
            operation=operation,
            status=status,
            user_id=user_id,
            source_ip=source_ip,
            request_data=self._sanitize_data(request_data),
            response_summary=response_summary,
            error_message=error_message,
            duration_ms=duration_ms,
        )

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
# Abstract Base Connector
# =============================================================================


class BaseConnector(ABC):
    """
    Abstract base class for all GL-012 STEAMQUAL connectors.

    Provides common functionality:
    - Connection pooling
    - Retry logic with exponential backoff
    - Circuit breaker pattern for fault tolerance
    - Health monitoring
    - Caching layer
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

        # Initialize circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            success_threshold=config.circuit_breaker_success_threshold,
            timeout_seconds=config.circuit_breaker_timeout_seconds,
            name=f"{config.connector_type.value}_{config.connector_id}",
        )

        # Initialize cache
        self._cache: LRUCache[Any] = LRUCache(
            max_size=config.cache_max_size,
            default_ttl=config.cache_ttl_seconds,
        )

        # Initialize metrics collector
        self._metrics = MetricsCollector(
            connector_id=config.connector_id,
            connector_type=config.connector_type,
        )

        # Initialize audit logger
        self._audit_logger = AuditLogger(
            connector_id=config.connector_id,
            connector_type=config.connector_type,
            enabled=config.audit_logging_enabled,
        )

        # Configure logging
        self._logger = logging.getLogger(
            f"connector.{config.connector_type.value}.{config.connector_id}"
        )
        self._logger.setLevel(getattr(logging, config.log_level))

        # Health check state
        self._last_health_check: Optional[HealthCheckResult] = None
        self._health_check_task: Optional[asyncio.Task] = None

        # Connection tracking
        self._connection: Optional[Any] = None
        self._connection_info: Optional[ConnectionInfo] = None

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
        return self._circuit_breaker.state

    # -------------------------------------------------------------------------
    # Abstract Methods - Must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Establish connection to the target device/system.

        Args:
            config: Optional additional connection configuration

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the target device/system.

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
    # Common Methods
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
        await self.connect()

        # Start health check task
        if self._config.health_check_interval_seconds > 0:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )

        self._logger.info(
            f"Connector {self._config.connector_name} initialized successfully"
        )

    async def shutdown(self) -> None:
        """
        Shutdown the connector gracefully.

        Stops health checks, disconnects, and releases resources.
        """
        self._logger.info(
            f"Shutting down connector {self._config.connector_name}"
        )

        # Stop health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Disconnect
        await self.disconnect()

        # Clear cache
        await self._cache.clear()

        self._logger.info(
            f"Connector {self._config.connector_name} shutdown complete"
        )

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)
                self._last_health_check = await self.health_check()

                if self._last_health_check.status == HealthStatus.UNHEALTHY:
                    self._logger.warning(
                        f"Health check failed: {self._last_health_check.message}"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health check error: {e}")

    async def execute_with_circuit_breaker(
        self,
        operation: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute operation with circuit breaker protection.

        Args:
            operation: Async operation to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result

        Raises:
            CircuitOpenError: If circuit is open
        """
        if not self._config.circuit_breaker_enabled:
            return await operation(*args, **kwargs)

        if not await self._circuit_breaker.can_execute():
            raise CircuitOpenError(
                f"Circuit breaker is open for connector {self._config.connector_name}",
                connector_id=self._config.connector_id,
                connector_type=self._config.connector_type,
            )

        start_time = time.time()
        try:
            result = await operation(*args, **kwargs)
            await self._circuit_breaker.record_success()

            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=True,
                latency_ms=latency_ms,
            )

            return result

        except Exception as e:
            await self._circuit_breaker.record_failure(e)

            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )

            raise

    async def get_cached(
        self,
        key: str,
        fetch_func: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get value from cache or fetch if not present.

        Args:
            key: Cache key
            fetch_func: Function to fetch value if not cached
            ttl: Optional TTL override

        Returns:
            Cached or fetched value
        """
        if not self._config.cache_enabled:
            return await fetch_func()

        # Try cache first
        cached = await self._cache.get(key)
        if cached is not None:
            return cached

        # Fetch and cache
        value = await fetch_func()
        await self._cache.set(key, value, ttl)
        return value

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    async def get_metrics(self) -> MetricsSnapshot:
        """
        Get current metrics snapshot.

        Returns:
            Metrics snapshot
        """
        return await self._metrics.get_snapshot(
            circuit_state=self._circuit_breaker.state,
            circuit_open_count=self._circuit_breaker.open_count,
            cache_stats=self._cache.stats,
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
            f"id={self._config.connector_id}, "
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
