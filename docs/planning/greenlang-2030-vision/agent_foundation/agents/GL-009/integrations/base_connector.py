"""
Base Connector for GL-009 THERMALIQ Integrations.

Provides abstract base class and common functionality for all industrial connectors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ConnectorStatus(Enum):
    """Connector connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    DEGRADED = "degraded"


@dataclass
class ConnectorHealth:
    """Health status of a connector."""
    status: ConnectorStatus
    is_healthy: bool
    last_successful_read: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    uptime_seconds: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "last_successful_read": self.last_successful_read.isoformat() if self.last_successful_read else None,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "uptime_seconds": self.uptime_seconds,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class BaseConnector(ABC):
    """
    Abstract base class for all GL-009 THERMALIQ integration connectors.

    Provides:
    - Connection management with auto-reconnect
    - Health monitoring
    - Error handling and logging
    - Retry logic with exponential backoff
    - Audit logging
    """

    def __init__(
        self,
        connector_id: str,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
        connection_timeout_seconds: float = 30.0,
        enable_audit_logging: bool = True,
    ):
        """
        Initialize base connector.

        Args:
            connector_id: Unique identifier for this connector
            max_retries: Maximum retry attempts for operations
            retry_delay_seconds: Initial delay between retries (exponential backoff)
            connection_timeout_seconds: Connection timeout
            enable_audit_logging: Enable audit logging for operations
        """
        self.connector_id = connector_id
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.connection_timeout_seconds = connection_timeout_seconds
        self.enable_audit_logging = enable_audit_logging

        # Connection state
        self._status = ConnectorStatus.DISCONNECTED
        self._connected_at: Optional[datetime] = None
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._last_successful_read: Optional[datetime] = None

        # Reconnection control
        self._reconnect_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Audit logging
        self._audit_logger = logging.getLogger(f"{__name__}.audit.{connector_id}")

    @property
    def status(self) -> ConnectorStatus:
        """Get current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._status == ConnectorStatus.CONNECTED

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the remote system.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the remote system.

        Returns:
            True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    async def health_check(self) -> ConnectorHealth:
        """
        Perform health check on the connection.

        Returns:
            ConnectorHealth object with status information
        """
        pass

    @abstractmethod
    async def read(self, **kwargs) -> Any:
        """
        Read data from the connected system.

        Args:
            **kwargs: Implementation-specific parameters

        Returns:
            Data read from the system
        """
        pass

    async def connect_with_retry(self) -> bool:
        """
        Connect with automatic retry using exponential backoff.

        Returns:
            True if connection successful, False otherwise
        """
        self._status = ConnectorStatus.CONNECTING
        delay = self.retry_delay_seconds

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"[{self.connector_id}] Connection attempt {attempt}/{self.max_retries}")

                success = await asyncio.wait_for(
                    self.connect(),
                    timeout=self.connection_timeout_seconds
                )

                if success:
                    self._status = ConnectorStatus.CONNECTED
                    self._connected_at = datetime.now()
                    self._error_count = 0
                    self._last_error = None
                    logger.info(f"[{self.connector_id}] Connected successfully")
                    self._audit_log("connect", {"success": True, "attempt": attempt})
                    return True

            except asyncio.TimeoutError:
                error_msg = f"Connection timeout after {self.connection_timeout_seconds}s"
                logger.warning(f"[{self.connector_id}] {error_msg}")
                self._last_error = error_msg

            except Exception as e:
                error_msg = f"Connection failed: {str(e)}"
                logger.error(f"[{self.connector_id}] {error_msg}")
                self._last_error = error_msg

            # Exponential backoff
            if attempt < self.max_retries:
                logger.info(f"[{self.connector_id}] Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        self._status = ConnectorStatus.FAILED
        self._error_count += 1
        self._audit_log("connect", {"success": False, "error": self._last_error})
        return False

    async def reconnect(self) -> bool:
        """
        Reconnect to the remote system.

        Returns:
            True if reconnection successful, False otherwise
        """
        logger.info(f"[{self.connector_id}] Reconnecting...")
        self._status = ConnectorStatus.RECONNECTING

        # Disconnect first
        try:
            await self.disconnect()
        except Exception as e:
            logger.warning(f"[{self.connector_id}] Disconnect during reconnect failed: {e}")

        # Connect with retry
        return await self.connect_with_retry()

    async def start_auto_reconnect(self, check_interval_seconds: float = 60.0):
        """
        Start automatic reconnection monitoring.

        Args:
            check_interval_seconds: Interval between health checks
        """
        logger.info(f"[{self.connector_id}] Starting auto-reconnect monitor")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(check_interval_seconds)

                # Perform health check
                health = await self.health_check()

                if not health.is_healthy and self._status != ConnectorStatus.RECONNECTING:
                    logger.warning(f"[{self.connector_id}] Unhealthy connection detected, reconnecting...")
                    await self.reconnect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.connector_id}] Auto-reconnect error: {e}")

    def stop_auto_reconnect(self):
        """Stop automatic reconnection monitoring."""
        self._shutdown_event.set()
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()

    async def execute_with_retry(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with retry logic.

        Args:
            operation: Async function to execute
            operation_name: Name for logging
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of the operation

        Raises:
            Exception: If all retries fail
        """
        delay = self.retry_delay_seconds

        for attempt in range(1, self.max_retries + 1):
            try:
                result = await operation(*args, **kwargs)
                self._last_successful_read = datetime.now()
                self._audit_log(operation_name, {"success": True, "attempt": attempt})
                return result

            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                logger.error(f"[{self.connector_id}] {operation_name} failed (attempt {attempt}/{self.max_retries}): {e}")

                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    self._audit_log(operation_name, {"success": False, "error": str(e)})
                    raise

    def _audit_log(self, operation: str, data: Dict[str, Any]):
        """
        Write audit log entry.

        Args:
            operation: Operation name
            data: Operation data
        """
        if not self.enable_audit_logging:
            return

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "connector_id": self.connector_id,
            "operation": operation,
            "data": data,
        }

        self._audit_logger.info(f"AUDIT: {audit_entry}")

    async def get_health(self) -> ConnectorHealth:
        """
        Get current health status.

        Returns:
            ConnectorHealth object
        """
        uptime = 0.0
        if self._connected_at:
            uptime = (datetime.now() - self._connected_at).total_seconds()

        return ConnectorHealth(
            status=self._status,
            is_healthy=self.is_connected and self._error_count < 10,
            last_successful_read=self._last_successful_read,
            last_error=self._last_error,
            error_count=self._error_count,
            uptime_seconds=uptime,
            metadata={"connector_id": self.connector_id}
        )

    @asynccontextmanager
    async def connection(self):
        """
        Context manager for connection lifecycle.

        Usage:
            async with connector.connection():
                data = await connector.read()
        """
        try:
            if not self.is_connected:
                await self.connect_with_retry()
            yield self
        finally:
            await self.disconnect()

    async def shutdown(self):
        """Graceful shutdown of connector."""
        logger.info(f"[{self.connector_id}] Shutting down...")
        self.stop_auto_reconnect()
        await self.disconnect()
        logger.info(f"[{self.connector_id}] Shutdown complete")
