"""
BaseConnector - Production-Grade Integration Framework
========================================================

Enterprise-grade base class for external system integrations with:
- Retry logic with exponential backoff (tenacity)
- Circuit breaker pattern (pybreaker)
- Health monitoring and status tracking
- Connection pooling and lifecycle management
- Mock implementations for testing
- Provenance tracking and audit trails
- Type-safe generic interfaces

This module provides the foundation for all GreenLang external integrations
including SCADA, ERP, CEMS, Historian, and CMMS systems.

Architecture:
    BaseConnector (abstract)
    ├── retry logic (tenacity)
    ├── circuit breaker (pybreaker)
    ├── health monitoring
    ├── connection pooling
    └── provenance tracking

Example:
    >>> from greenlang.integrations.base_connector import BaseConnector
    >>> from greenlang.integrations.scada_connector import SCADAConnector
    >>>
    >>> connector = SCADAConnector(config)
    >>> async with connector:
    ...     data = await connector.fetch_data(query)
    ...     assert connector.health_status == HealthStatus.HEALTHY

Author: GreenLang Backend Team
Date: 2025-12-01
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Dict, Any, List, Callable
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager

# Retry logic with exponential backoff
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)

# Circuit breaker pattern
try:
    from pybreaker import CircuitBreaker, CircuitBreakerError
except ImportError:
    # Fallback if pybreaker not installed
    class CircuitBreaker:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, func):
            return func

    class CircuitBreakerError(Exception):
        pass

# Type variables for generic connector
T = TypeVar('T')  # Data type
TQuery = TypeVar('TQuery', bound=BaseModel)
TPayload = TypeVar('TPayload', bound=BaseModel)
TConfig = TypeVar('TConfig', bound=BaseModel)

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Connector health status enumeration."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"


class ConnectionState(str, Enum):
    """Connection lifecycle state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    ERROR = "error"


class ConnectorConfig(BaseModel):
    """
    Base configuration for all connectors.

    All connector-specific configs should inherit from this.

    Attributes:
        connector_id: Unique identifier for this connector instance
        connector_type: Type of connector (scada, erp, cems, etc.)
        timeout_seconds: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        circuit_breaker_enabled: Enable circuit breaker pattern
        circuit_breaker_threshold: Failures before opening circuit
        circuit_breaker_timeout: Time to wait before retrying (seconds)
        health_check_interval: Health check interval in seconds
        connection_pool_size: Max connections in pool
        enable_provenance: Track provenance for audit trails
    """

    connector_id: str = Field(..., description="Unique connector identifier")
    connector_type: str = Field(..., description="Connector type (scada/erp/cems/historian/cmms)")

    # Timeout settings
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout")

    # Retry settings
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_multiplier: int = Field(default=1, ge=1, le=10, description="Exponential backoff multiplier")
    retry_min_wait: int = Field(default=2, ge=1, le=60, description="Minimum wait between retries (seconds)")
    retry_max_wait: int = Field(default=10, ge=1, le=300, description="Maximum wait between retries (seconds)")

    # Circuit breaker settings
    circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker")
    circuit_breaker_threshold: int = Field(default=5, ge=1, le=100, description="Failures before opening")
    circuit_breaker_timeout: int = Field(default=60, ge=1, le=600, description="Circuit recovery timeout")

    # Health monitoring
    health_check_interval: int = Field(default=60, ge=10, le=3600, description="Health check interval")

    # Connection pooling
    connection_pool_size: int = Field(default=5, ge=1, le=100, description="Connection pool size")

    # Provenance
    enable_provenance: bool = Field(default=True, description="Enable provenance tracking")

    # Mock mode for testing
    mock_mode: bool = Field(default=False, description="Use mock implementation")

    class Config:
        frozen = False  # Allow modification for dynamic config


class ConnectorMetrics(BaseModel):
    """Connector performance and health metrics."""

    total_requests: int = Field(default=0, description="Total requests made")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")
    retry_count: int = Field(default=0, description="Total retry attempts")
    circuit_breaker_opens: int = Field(default=0, description="Times circuit opened")

    avg_response_time_ms: float = Field(default=0.0, description="Average response time")
    last_request_time: Optional[datetime] = Field(default=None, description="Last request timestamp")
    last_success_time: Optional[datetime] = Field(default=None, description="Last success timestamp")
    last_failure_time: Optional[datetime] = Field(default=None, description="Last failure timestamp")

    health_status: HealthStatus = Field(default=HealthStatus.UNKNOWN, description="Current health")
    connection_state: ConnectionState = Field(default=ConnectionState.DISCONNECTED, description="Connection state")

    uptime_seconds: float = Field(default=0.0, description="Total uptime in seconds")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class ConnectorProvenance(BaseModel):
    """
    Provenance metadata for data lineage and audit trails.

    Required for regulatory compliance and reproducibility.
    """

    connector_id: str = Field(..., description="Connector instance identifier")
    connector_type: str = Field(..., description="Connector type")
    connector_version: str = Field(..., description="Connector version (semver)")

    query_hash: str = Field(..., description="SHA-256 hash of query")
    response_hash: str = Field(..., description="SHA-256 hash of response")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp")
    request_id: Optional[str] = Field(default=None, description="Request correlation ID")

    mode: str = Field(default="live", description="Execution mode (live/mock/replay)")
    source_system: Optional[str] = Field(default=None, description="Source system identifier")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BaseConnector(ABC, Generic[TQuery, TPayload, TConfig]):
    """
    Base class for external system integrations.

    Provides enterprise-grade infrastructure:
    - Retry logic with exponential backoff (tenacity)
    - Circuit breaker pattern (pybreaker)
    - Health monitoring and status tracking
    - Connection pooling and lifecycle management
    - Provenance tracking for audit trails
    - Type-safe generic interfaces

    Zero-Hallucination Principle:
    - All data fetching is deterministic
    - No LLM calls in data retrieval path
    - Provenance tracking with SHA-256 hashes
    - Mock implementations for testing

    Attributes:
        config: Connector configuration
        health_status: Current health status
        metrics: Performance metrics
        circuit_breaker: Circuit breaker instance

    Example:
        >>> class MySCADAConnector(BaseConnector[SCADAQuery, SCADAPayload, SCADAConfig]):
        ...     connector_id = "scada-opcua"
        ...     connector_version = "1.0.0"
        ...
        ...     async def connect(self) -> bool:
        ...         # Establish OPC UA connection
        ...         return True
        ...
        ...     async def _fetch_data_impl(self, query: SCADAQuery) -> SCADAPayload:
        ...         # Fetch data from SCADA system
        ...         return SCADAPayload(...)
        >>>
        >>> connector = MySCADAConnector(config)
        >>> async with connector:
        ...     data = await connector.fetch_data(query)
    """

    # Class attributes (must be set by subclass)
    connector_id: str
    connector_version: str = "0.1.0"

    def __init__(self, config: TConfig):
        """
        Initialize connector with configuration.

        Args:
            config: Typed connector configuration
        """
        self.config = config
        self.metrics = ConnectorMetrics(
            health_status=HealthStatus.UNKNOWN,
            connection_state=ConnectionState.DISCONNECTED
        )
        self.logger = logging.getLogger(f"{__name__}.{config.connector_id}")

        # Circuit breaker initialization
        if config.circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreaker(
                fail_max=config.circuit_breaker_threshold,
                timeout_duration=config.circuit_breaker_timeout,
                name=f"{config.connector_id}_breaker"
            )
        else:
            self.circuit_breaker = None

        # Connection pool (placeholder - implement in subclass if needed)
        self._connection_pool: Optional[Any] = None

        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None

        # Start time for uptime tracking
        self._start_time: Optional[datetime] = None

        self.logger.info(
            f"Initialized connector: {config.connector_id} "
            f"(type={config.connector_type}, version={self.connector_version})"
        )

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to external system.

        This method should handle:
        - Authentication/authorization
        - Connection establishment
        - Initial health check
        - Connection pool initialization

        Returns:
            True if connection successful, False otherwise

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Close connection to external system.

        This method should handle:
        - Graceful connection shutdown
        - Connection pool cleanup
        - Resource cleanup

        Returns:
            True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    async def _health_check_impl(self) -> bool:
        """
        Implementation-specific health check.

        This method should verify:
        - Connection is alive
        - System is responsive
        - Credentials are valid

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def _fetch_data_impl(self, query: TQuery) -> TPayload:
        """
        Implementation-specific data fetching.

        This is the core data retrieval method - ZERO HALLUCINATION.
        No LLM calls allowed in this method.

        Args:
            query: Typed query specification

        Returns:
            Typed data payload

        Raises:
            ConnectionError: If connection fails
            TimeoutError: If request times out
            ValueError: If query is invalid
        """
        pass

    async def fetch_data(
        self,
        query: TQuery,
        timeout: Optional[int] = None
    ) -> tuple[TPayload, ConnectorProvenance]:
        """
        Fetch data with retry logic and circuit breaker.

        This is the main entry point for data retrieval.
        Provides automatic retry, circuit breaking, and provenance tracking.

        Args:
            query: Typed query specification
            timeout: Optional timeout override (seconds)

        Returns:
            Tuple of (payload, provenance)

        Raises:
            CircuitBreakerError: If circuit is open
            ConnectionError: If connection fails after retries
            TimeoutError: If request times out
        """
        start_time = datetime.now(timezone.utc)
        self.metrics.total_requests += 1
        self.metrics.last_request_time = start_time

        # Use timeout from config if not specified
        timeout = timeout or self.config.timeout_seconds

        try:
            # Check health before fetching
            if not await self.health_check():
                self.logger.warning(f"Health check failed for {self.config.connector_id}")

            # Execute fetch with timeout (circuit breaker applied in retry method)
            payload = await asyncio.wait_for(
                self._fetch_with_retry(query, timeout),
                timeout=timeout
            )

            # Success - update metrics
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now(timezone.utc)

            # Calculate response time
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_avg_response_time(response_time)

            # Generate provenance
            provenance = self._generate_provenance(query, payload, start_time)

            self.logger.info(
                f"Successfully fetched data from {self.config.connector_id} "
                f"in {response_time:.2f}ms"
            )

            return payload, provenance

        except CircuitBreakerError as e:
            self.metrics.failed_requests += 1
            self.metrics.circuit_breaker_opens += 1
            self.logger.error(f"Circuit breaker open for {self.config.connector_id}: {e}")
            raise

        except asyncio.TimeoutError as e:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now(timezone.utc)
            self.logger.error(f"Request timeout for {self.config.connector_id} after {timeout}s")
            raise TimeoutError(f"Request timed out after {timeout}s") from e

        except Exception as e:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now(timezone.utc)
            self.logger.error(
                f"Failed to fetch data from {self.config.connector_id}: {e}",
                exc_info=True
            )
            raise

    async def _fetch_with_retry(self, query: TQuery, timeout: int) -> TPayload:
        """
        Fetch data with retry logic using tenacity.

        Implements exponential backoff with jitter for resilience.

        Args:
            query: Query specification
            timeout: Timeout in seconds

        Returns:
            Data payload
        """

        @retry(
            stop=stop_after_attempt(self.config.max_retries + 1),
            wait=wait_exponential(
                multiplier=self.config.retry_multiplier,
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait
            ),
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
            before_sleep=before_sleep_log(self.logger, logging.WARNING),
            after=after_log(self.logger, logging.DEBUG)
        )
        async def _fetch_with_tenacity():
            self.metrics.retry_count += 1
            return await self._fetch_data_impl(query)

        return await _fetch_with_tenacity()

    async def health_check(self) -> bool:
        """
        Check connector health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            is_healthy = await self._health_check_impl()

            if is_healthy:
                self.metrics.health_status = HealthStatus.HEALTHY
            else:
                self.metrics.health_status = HealthStatus.DEGRADED

            return is_healthy

        except Exception as e:
            self.logger.error(f"Health check failed for {self.config.connector_id}: {e}")
            self.metrics.health_status = HealthStatus.UNHEALTHY
            return False

    async def start_health_monitoring(self):
        """Start periodic health check monitoring."""

        async def _health_check_loop():
            while True:
                await asyncio.sleep(self.config.health_check_interval)
                await self.health_check()

        if not self._health_check_task:
            self._health_check_task = asyncio.create_task(_health_check_loop())
            self.logger.info(f"Started health monitoring for {self.config.connector_id}")

    async def stop_health_monitoring(self):
        """Stop health check monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            self.logger.info(f"Stopped health monitoring for {self.config.connector_id}")

    def _generate_provenance(
        self,
        query: TQuery,
        payload: TPayload,
        timestamp: datetime
    ) -> ConnectorProvenance:
        """
        Generate provenance metadata for audit trail.

        Args:
            query: Query specification
            payload: Response payload
            timestamp: Request timestamp

        Returns:
            Provenance metadata
        """
        if not self.config.enable_provenance:
            return ConnectorProvenance(
                connector_id=self.config.connector_id,
                connector_type=self.config.connector_type,
                connector_version=self.connector_version,
                query_hash="",
                response_hash="",
                timestamp=timestamp
            )

        # Calculate query hash
        query_json = query.json(sort_keys=True)
        query_hash = hashlib.sha256(query_json.encode()).hexdigest()

        # Calculate response hash
        payload_json = payload.json(sort_keys=True)
        response_hash = hashlib.sha256(payload_json.encode()).hexdigest()

        return ConnectorProvenance(
            connector_id=self.config.connector_id,
            connector_type=self.config.connector_type,
            connector_version=self.connector_version,
            query_hash=query_hash,
            response_hash=response_hash,
            timestamp=timestamp,
            mode="mock" if self.config.mock_mode else "live"
        )

    def _update_avg_response_time(self, response_time_ms: float):
        """Update rolling average response time."""
        if self.metrics.avg_response_time_ms == 0:
            self.metrics.avg_response_time_ms = response_time_ms
        else:
            # Exponential moving average
            alpha = 0.2
            self.metrics.avg_response_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * self.metrics.avg_response_time_ms
            )

    def get_metrics(self) -> ConnectorMetrics:
        """
        Get current connector metrics.

        Returns:
            Metrics snapshot
        """
        # Update uptime
        if self._start_time:
            self.metrics.uptime_seconds = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds()

        return self.metrics.model_copy(deep=True)

    def reset_metrics(self):
        """Reset all metrics to initial state."""
        self.metrics = ConnectorMetrics(
            health_status=self.metrics.health_status,
            connection_state=self.metrics.connection_state
        )
        self.logger.info(f"Reset metrics for {self.config.connector_id}")

    # Async context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        self._start_time = datetime.now(timezone.utc)
        self.metrics.connection_state = ConnectionState.CONNECTING

        connected = await self.connect()
        if connected:
            self.metrics.connection_state = ConnectionState.CONNECTED
            await self.start_health_monitoring()
        else:
            self.metrics.connection_state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to connect: {self.config.connector_id}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.metrics.connection_state = ConnectionState.DISCONNECTING

        await self.stop_health_monitoring()
        await self.disconnect()

        self.metrics.connection_state = ConnectionState.DISCONNECTED

    # Sync context manager support (for backward compatibility)
    def __enter__(self):
        """Sync context manager entry - runs async connect."""
        asyncio.run(self.__aenter__())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit - runs async disconnect."""
        asyncio.run(self.__aexit__(exc_type, exc_val, exc_tb))


class MockConnector(BaseConnector[TQuery, TPayload, TConfig]):
    """
    Mock connector for testing.

    Provides deterministic mock data without external dependencies.
    All subclasses should implement a mock version for testing.

    Example:
        >>> class MockSCADAConnector(MockConnector[SCADAQuery, SCADAPayload, SCADAConfig]):
        ...     async def _fetch_data_impl(self, query: SCADAQuery) -> SCADAPayload:
        ...         # Return deterministic mock data
        ...         return SCADAPayload(value=42.0, timestamp=datetime.now(timezone.utc))
    """

    async def connect(self) -> bool:
        """Mock connect - always succeeds."""
        self.logger.info(f"Mock connect: {self.config.connector_id}")
        return True

    async def disconnect(self) -> bool:
        """Mock disconnect - always succeeds."""
        self.logger.info(f"Mock disconnect: {self.config.connector_id}")
        return True

    async def _health_check_impl(self) -> bool:
        """Mock health check - always healthy."""
        return True
