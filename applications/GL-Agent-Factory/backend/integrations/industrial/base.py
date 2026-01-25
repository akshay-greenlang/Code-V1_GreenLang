"""
Base Industrial Connector Interface.

This module provides the abstract base class for all industrial protocol
connectors, ensuring consistent interface, error handling, metrics collection,
and security across all industrial integrations.

Features:
    - Unified connection interface
    - Health checking with configurable intervals
    - Metrics collection and reporting
    - Rate limiting with token bucket algorithm
    - TLS/SSL security configuration
    - Automatic reconnection with backoff
    - Connection pooling support

Example:
    >>> class MyConnector(BaseIndustrialConnector):
    ...     async def _do_connect(self) -> bool:
    ...         # Implementation
    ...     async def _do_disconnect(self) -> None:
    ...         # Implementation
    ...     async def read_tags(self, tag_ids: List[str]) -> BatchReadResponse:
    ...         # Implementation
"""

import asyncio
import logging
import ssl
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, SecretStr

from .data_models import (
    BatchReadRequest,
    BatchReadResponse,
    BatchWriteRequest,
    BatchWriteResponse,
    ConnectionMetrics,
    ConnectionState,
    HistoricalQuery,
    HistoricalResult,
    SubscriptionConfig,
    SubscriptionStatus,
    TagMetadata,
    TagValue,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Models
# =============================================================================


class SecurityMode(str, Enum):
    """Security mode for connections."""

    NONE = "none"
    SIGN = "sign"
    SIGN_AND_ENCRYPT = "sign_and_encrypt"


class AuthenticationType(str, Enum):
    """Authentication type for connections."""

    NONE = "none"
    USERNAME_PASSWORD = "username_password"
    CERTIFICATE = "certificate"
    TOKEN = "token"
    API_KEY = "api_key"


class TLSConfig(BaseModel):
    """
    TLS/SSL configuration for secure connections.

    Attributes:
        enabled: Enable TLS
        cert_path: Client certificate path
        key_path: Client key path
        ca_path: CA certificate path
        verify_server: Verify server certificate
        check_hostname: Check server hostname
        min_version: Minimum TLS version
    """

    enabled: bool = Field(True, description="Enable TLS")
    cert_path: Optional[str] = Field(None, description="Client certificate path")
    key_path: Optional[str] = Field(None, description="Client private key path")
    ca_path: Optional[str] = Field(None, description="CA certificate path")
    verify_server: bool = Field(True, description="Verify server certificate")
    check_hostname: bool = Field(True, description="Check server hostname")
    min_version: str = Field("TLSv1.2", description="Minimum TLS version")

    def create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """
        Create SSL context from configuration.

        Returns:
            Configured SSL context or None if TLS disabled
        """
        if not self.enabled:
            return None

        # Determine minimum TLS version
        min_versions = {
            "TLSv1.2": ssl.TLSVersion.TLSv1_2,
            "TLSv1.3": ssl.TLSVersion.TLSv1_3,
        }
        min_ver = min_versions.get(self.min_version, ssl.TLSVersion.TLSv1_2)

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = min_ver

        if not self.verify_server:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        else:
            context.check_hostname = self.check_hostname
            context.verify_mode = ssl.CERT_REQUIRED

        if self.ca_path:
            context.load_verify_locations(self.ca_path)

        if self.cert_path and self.key_path:
            context.load_cert_chain(self.cert_path, self.key_path)

        return context


class RateLimitConfig(BaseModel):
    """
    Rate limiting configuration using token bucket algorithm.

    Attributes:
        enabled: Enable rate limiting
        requests_per_second: Maximum requests per second
        burst_size: Maximum burst size
        retry_after_ms: Wait time when rate limited
    """

    enabled: bool = Field(True, description="Enable rate limiting")
    requests_per_second: float = Field(100.0, gt=0, description="Requests per second")
    burst_size: int = Field(50, gt=0, description="Burst size")
    retry_after_ms: int = Field(100, ge=0, description="Retry delay in ms")


class ReconnectConfig(BaseModel):
    """
    Automatic reconnection configuration.

    Attributes:
        enabled: Enable automatic reconnection
        max_attempts: Maximum reconnection attempts (0 = infinite)
        initial_delay_ms: Initial delay before first retry
        max_delay_ms: Maximum delay between retries
        backoff_multiplier: Exponential backoff multiplier
    """

    enabled: bool = Field(True, description="Enable auto-reconnect")
    max_attempts: int = Field(0, ge=0, description="Max attempts (0=infinite)")
    initial_delay_ms: int = Field(1000, ge=100, description="Initial delay ms")
    max_delay_ms: int = Field(60000, ge=1000, description="Max delay ms")
    backoff_multiplier: float = Field(2.0, ge=1.0, description="Backoff multiplier")


class BaseConnectorConfig(BaseModel):
    """
    Base configuration for industrial connectors.

    Attributes:
        host: Server hostname or IP
        port: Server port
        timeout_seconds: Operation timeout
        auth_type: Authentication type
        username: Username for auth
        password: Password for auth
        tls: TLS configuration
        rate_limit: Rate limiting configuration
        reconnect: Reconnection configuration
    """

    host: str = Field(..., description="Server hostname or IP")
    port: int = Field(..., description="Server port")
    timeout_seconds: float = Field(30.0, gt=0, description="Operation timeout")

    # Authentication
    auth_type: AuthenticationType = Field(
        AuthenticationType.NONE,
        description="Authentication type"
    )
    username: Optional[str] = Field(None, description="Username")
    password: Optional[SecretStr] = Field(None, description="Password")
    api_key: Optional[SecretStr] = Field(None, description="API key")

    # Security
    tls: TLSConfig = Field(default_factory=TLSConfig, description="TLS config")

    # Rate limiting
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description="Rate limit config"
    )

    # Reconnection
    reconnect: ReconnectConfig = Field(
        default_factory=ReconnectConfig,
        description="Reconnect config"
    )

    # Health check
    health_check_interval_seconds: float = Field(
        30.0,
        gt=0,
        description="Health check interval"
    )

    # Connection name for logging
    name: str = Field("industrial_connector", description="Connector name")


# =============================================================================
# Rate Limiter
# =============================================================================


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for controlling request rates.

    Implements the token bucket algorithm for smooth rate limiting
    with support for burst traffic.
    """

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter."""
        self.config = config
        self.tokens = float(config.burst_size)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if rate limited
        """
        if not self.config.enabled:
            return True

        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill

            # Refill tokens
            self.tokens = min(
                self.config.burst_size,
                self.tokens + elapsed * self.config.requests_per_second
            )
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait_and_acquire(self, tokens: int = 1) -> None:
        """
        Wait until tokens available and acquire.

        Args:
            tokens: Number of tokens to acquire
        """
        while not await self.acquire(tokens):
            await asyncio.sleep(self.config.retry_after_ms / 1000)


# =============================================================================
# Base Connector
# =============================================================================


class BaseIndustrialConnector(ABC):
    """
    Abstract base class for industrial protocol connectors.

    Provides common functionality for all industrial integrations
    including connection management, metrics, rate limiting, and security.

    Subclasses must implement:
        - _do_connect(): Establish protocol-specific connection
        - _do_disconnect(): Close protocol-specific connection
        - read_tags(): Read tag values
        - write_tags(): Write tag values (optional)

    Attributes:
        config: Connector configuration
        state: Current connection state
        metrics: Connection metrics

    Example:
        >>> class OPCUAConnector(BaseIndustrialConnector):
        ...     async def _do_connect(self) -> bool:
        ...         self._client = Client(self.config.endpoint_url)
        ...         await self._client.connect()
        ...         return True
    """

    def __init__(self, config: BaseConnectorConfig):
        """
        Initialize the industrial connector.

        Args:
            config: Connector configuration
        """
        self.config = config
        self._state = ConnectionState.DISCONNECTED
        self._metrics = ConnectionMetrics()
        self._rate_limiter = TokenBucketRateLimiter(config.rate_limit)
        self._ssl_context: Optional[ssl.SSLContext] = None
        self._health_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._subscriptions: Dict[str, SubscriptionStatus] = {}
        self._tag_metadata_cache: Dict[str, TagMetadata] = {}
        self._connected_at: Optional[datetime] = None
        self._lock = asyncio.Lock()

        logger.info(
            f"Initialized {self.__class__.__name__} for {config.host}:{config.port}"
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def metrics(self) -> ConnectionMetrics:
        """Get current metrics."""
        return self._metrics

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Connect to the industrial system.

        Establishes connection with TLS if configured, starts health
        checking, and initializes metrics.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        async with self._lock:
            if self._state == ConnectionState.CONNECTED:
                logger.warning(f"{self.config.name}: Already connected")
                return True

            self._state = ConnectionState.CONNECTING
            self._shutdown_event.clear()

            logger.info(
                f"{self.config.name}: Connecting to {self.config.host}:{self.config.port}"
            )

            try:
                # Create SSL context if needed
                if self.config.tls.enabled:
                    self._ssl_context = self.config.tls.create_ssl_context()

                # Perform protocol-specific connection
                success = await self._do_connect()

                if success:
                    self._state = ConnectionState.CONNECTED
                    self._connected_at = datetime.utcnow()
                    self._metrics.state = ConnectionState.CONNECTED
                    self._metrics.connected_since = self._connected_at

                    # Start health check task
                    self._start_health_check()

                    logger.info(f"{self.config.name}: Connected successfully")
                    return True
                else:
                    raise ConnectionError("Connection failed")

            except Exception as e:
                self._state = ConnectionState.ERROR
                self._metrics.state = ConnectionState.ERROR
                self._metrics.last_error = str(e)
                self._metrics.last_error_time = datetime.utcnow()

                logger.error(f"{self.config.name}: Connection failed: {e}")

                # Start reconnection if enabled
                if self.config.reconnect.enabled:
                    self._start_reconnect()

                raise ConnectionError(f"Failed to connect: {e}")

    async def disconnect(self) -> None:
        """
        Disconnect from the industrial system.

        Stops health checking, closes subscriptions, and cleans up resources.
        """
        async with self._lock:
            if self._state == ConnectionState.DISCONNECTED:
                return

            self._state = ConnectionState.CLOSING
            self._shutdown_event.set()

            logger.info(f"{self.config.name}: Disconnecting...")

            try:
                # Stop background tasks
                await self._stop_health_check()
                await self._stop_reconnect()

                # Close all subscriptions
                for sub_id in list(self._subscriptions.keys()):
                    await self.unsubscribe(sub_id)

                # Perform protocol-specific disconnect
                await self._do_disconnect()

            except Exception as e:
                logger.error(f"{self.config.name}: Error during disconnect: {e}")

            finally:
                self._state = ConnectionState.DISCONNECTED
                self._metrics.state = ConnectionState.DISCONNECTED
                self._connected_at = None

                logger.info(f"{self.config.name}: Disconnected")

    async def reconnect(self) -> bool:
        """
        Force reconnection to the system.

        Returns:
            True if reconnection successful
        """
        logger.info(f"{self.config.name}: Reconnecting...")

        try:
            await self.disconnect()
        except Exception:
            pass

        return await self.connect()

    @asynccontextmanager
    async def connection(self):
        """
        Context manager for connection lifecycle.

        Example:
            >>> async with connector.connection():
            ...     values = await connector.read_tags(["tag1"])
        """
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    async def _do_connect(self) -> bool:
        """
        Perform protocol-specific connection.

        Returns:
            True if connection successful

        Raises:
            Exception: If connection fails
        """
        pass

    @abstractmethod
    async def _do_disconnect(self) -> None:
        """Perform protocol-specific disconnection."""
        pass

    @abstractmethod
    async def read_tags(
        self,
        tag_ids: List[str],
    ) -> BatchReadResponse:
        """
        Read multiple tag values.

        Args:
            tag_ids: List of tag identifiers to read

        Returns:
            BatchReadResponse with values and errors

        Raises:
            ConnectionError: If not connected
        """
        pass

    async def write_tags(
        self,
        request: BatchWriteRequest,
    ) -> BatchWriteResponse:
        """
        Write multiple tag values.

        Default implementation returns error for all writes.
        Override in subclasses that support writing.

        Args:
            request: Batch write request

        Returns:
            BatchWriteResponse with results
        """
        return BatchWriteResponse(
            errors={tag: "Write not supported" for tag in request.writes.keys()}
        )

    # =========================================================================
    # Tag Operations
    # =========================================================================

    async def read_tag(self, tag_id: str) -> Optional[TagValue]:
        """
        Read a single tag value.

        Args:
            tag_id: Tag identifier

        Returns:
            TagValue or None if read failed
        """
        response = await self.read_tags([tag_id])
        return response.values.get(tag_id)

    async def write_tag(
        self,
        tag_id: str,
        value: Any,
        validate: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Write a single tag value.

        Args:
            tag_id: Tag identifier
            value: Value to write
            validate: Validate against range limits

        Returns:
            Tuple of (success, error_message)
        """
        request = BatchWriteRequest(
            writes={tag_id: value},
            validate_ranges=validate,
        )
        response = await self.write_tags(request)

        if tag_id in response.errors:
            return False, response.errors[tag_id]
        return response.success.get(tag_id, False), None

    async def get_tag_metadata(
        self,
        tag_id: str,
        use_cache: bool = True,
    ) -> Optional[TagMetadata]:
        """
        Get metadata for a tag.

        Args:
            tag_id: Tag identifier
            use_cache: Use cached metadata if available

        Returns:
            TagMetadata or None if not found
        """
        if use_cache and tag_id in self._tag_metadata_cache:
            return self._tag_metadata_cache[tag_id]

        # Default implementation - override in subclasses
        return None

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def subscribe(
        self,
        config: SubscriptionConfig,
        callback: Callable[[TagValue], None],
    ) -> str:
        """
        Create subscription for real-time data.

        Args:
            config: Subscription configuration
            callback: Callback function for data updates

        Returns:
            Subscription identifier

        Raises:
            NotImplementedError: If subscriptions not supported
        """
        raise NotImplementedError("Subscriptions not supported by this connector")

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Remove a subscription.

        Args:
            subscription_id: Subscription identifier

        Returns:
            True if unsubscribed successfully
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    def get_subscriptions(self) -> Dict[str, SubscriptionStatus]:
        """Get all active subscriptions."""
        return self._subscriptions.copy()

    # =========================================================================
    # Historical Data
    # =========================================================================

    async def read_history(
        self,
        query: HistoricalQuery,
    ) -> Dict[str, HistoricalResult]:
        """
        Read historical data.

        Default implementation raises NotImplementedError.
        Override in subclasses that support historical data.

        Args:
            query: Historical query specification

        Returns:
            Dictionary of tag_id to HistoricalResult

        Raises:
            NotImplementedError: If historical not supported
        """
        raise NotImplementedError("Historical data not supported by this connector")

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health check result dictionary
        """
        try:
            # Try to read a known tag or perform ping
            healthy = await self._do_health_check()

            return {
                "healthy": healthy,
                "state": self._state.value,
                "connected_since": (
                    self._connected_at.isoformat() if self._connected_at else None
                ),
                "metrics": {
                    "total_requests": self._metrics.total_requests,
                    "failed_requests": self._metrics.failed_requests,
                    "success_rate": self._metrics.success_rate,
                    "avg_response_ms": self._metrics.avg_response_ms,
                    "reconnect_count": self._metrics.reconnect_count,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "healthy": False,
                "state": ConnectionState.ERROR.value,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _do_health_check(self) -> bool:
        """
        Perform protocol-specific health check.

        Default implementation checks connection state.
        Override for protocol-specific health checks.

        Returns:
            True if healthy
        """
        return self._state == ConnectionState.CONNECTED

    def _start_health_check(self) -> None:
        """Start background health check task."""
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_check_loop())

    async def _stop_health_check(self) -> None:
        """Stop background health check task."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                if self._shutdown_event.is_set():
                    break

                result = await self.health_check()

                if not result.get("healthy", False):
                    logger.warning(
                        f"{self.config.name}: Health check failed: "
                        f"{result.get('error', 'Unknown')}"
                    )

                    # Trigger reconnection if needed
                    if self.config.reconnect.enabled:
                        self._start_reconnect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self.config.name}: Health check error: {e}")

    # =========================================================================
    # Reconnection
    # =========================================================================

    def _start_reconnect(self) -> None:
        """Start background reconnection task."""
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _stop_reconnect(self) -> None:
        """Stop background reconnection task."""
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

    async def _reconnect_loop(self) -> None:
        """Background reconnection loop with exponential backoff."""
        config = self.config.reconnect
        attempt = 0
        delay_ms = config.initial_delay_ms

        while not self._shutdown_event.is_set():
            if config.max_attempts > 0 and attempt >= config.max_attempts:
                logger.error(
                    f"{self.config.name}: Max reconnection attempts reached"
                )
                break

            self._state = ConnectionState.RECONNECTING
            self._metrics.state = ConnectionState.RECONNECTING
            attempt += 1

            logger.info(
                f"{self.config.name}: Reconnection attempt {attempt} "
                f"(delay: {delay_ms}ms)"
            )

            await asyncio.sleep(delay_ms / 1000)

            try:
                # Try to reconnect
                success = await self._do_connect()

                if success:
                    self._state = ConnectionState.CONNECTED
                    self._metrics.state = ConnectionState.CONNECTED
                    self._metrics.reconnect_count += 1
                    self._connected_at = datetime.utcnow()
                    self._metrics.connected_since = self._connected_at

                    logger.info(
                        f"{self.config.name}: Reconnected successfully "
                        f"(attempt {attempt})"
                    )

                    # Restart health check
                    self._start_health_check()
                    return

            except Exception as e:
                logger.warning(
                    f"{self.config.name}: Reconnection attempt {attempt} "
                    f"failed: {e}"
                )

            # Calculate next delay with exponential backoff
            delay_ms = min(
                int(delay_ms * config.backoff_multiplier),
                config.max_delay_ms
            )

    # =========================================================================
    # Metrics
    # =========================================================================

    def _record_request(
        self,
        success: bool,
        response_time_ms: float,
    ) -> None:
        """
        Record request metrics.

        Args:
            success: Whether request succeeded
            response_time_ms: Response time in milliseconds
        """
        self._metrics.total_requests += 1

        if not success:
            self._metrics.failed_requests += 1

        # Update average response time (exponential moving average)
        alpha = 0.1
        self._metrics.avg_response_ms = (
            alpha * response_time_ms +
            (1 - alpha) * self._metrics.avg_response_ms
        )

    async def _rate_limited_request(
        self,
        request_func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute request with rate limiting.

        Args:
            request_func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from request_func
        """
        await self._rate_limiter.wait_and_acquire()

        start_time = time.monotonic()
        success = False

        try:
            result = await request_func(*args, **kwargs)
            success = True
            return result

        finally:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_request(success, elapsed_ms)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _validate_connected(self) -> None:
        """
        Validate connection is established.

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected:
            raise ConnectionError(
                f"{self.config.name}: Not connected to "
                f"{self.config.host}:{self.config.port}"
            )

    def get_status(self) -> Dict[str, Any]:
        """
        Get connector status information.

        Returns:
            Status dictionary
        """
        return {
            "name": self.config.name,
            "host": self.config.host,
            "port": self.config.port,
            "state": self._state.value,
            "connected_since": (
                self._connected_at.isoformat() if self._connected_at else None
            ),
            "metrics": {
                "total_requests": self._metrics.total_requests,
                "failed_requests": self._metrics.failed_requests,
                "success_rate": self._metrics.success_rate,
                "avg_response_ms": self._metrics.avg_response_ms,
                "reconnect_count": self._metrics.reconnect_count,
            },
            "subscriptions": len(self._subscriptions),
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "SecurityMode",
    "AuthenticationType",
    "TLSConfig",
    "RateLimitConfig",
    "ReconnectConfig",
    "BaseConnectorConfig",
    # Rate limiting
    "TokenBucketRateLimiter",
    # Base connector
    "BaseIndustrialConnector",
]
