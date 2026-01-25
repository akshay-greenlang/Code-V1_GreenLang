"""
GL-004 BURNMASTER - Connection Manager

Centralized connection management for all integration endpoints.

Features:
    - Manage multiple connections (DCS, BMS, Historian, OPC-UA)
    - Connection health monitoring with heartbeat
    - Automatic reconnection on failure
    - Failover to backup connections
    - Connection event logging and alerting

Author: GreenLang Combustion Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConnectionType(str, Enum):
    """Types of managed connections."""
    DCS = "dcs"
    BMS = "bms"
    HISTORIAN = "historian"
    OPCUA = "opcua"


class ConnectionState(str, Enum):
    """Connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    DEGRADED = "degraded"


class ConnectionEventType(str, Enum):
    """Types of connection events."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    RECONNECTED = "reconnected"
    FAILED = "failed"
    FAILOVER = "failover"
    HEALTH_CHECK_FAILED = "health_check_failed"
    HEALTH_RESTORED = "health_restored"


@dataclass
class ConnectionConfig:
    """Configuration for a managed connection."""
    connection_id: str
    connection_type: ConnectionType
    name: str
    endpoint: str
    is_primary: bool = True
    backup_endpoint: Optional[str] = None
    health_check_interval_seconds: float = 30.0
    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 10
    timeout_seconds: float = 30.0
    enabled: bool = True


@dataclass
class ConnectionHealth:
    """Health status of a connection."""
    connection_id: str
    is_healthy: bool
    state: ConnectionState
    last_successful_communication: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    latency_ms: float = 0.0
    error_message: Optional[str] = None
    uptime_seconds: float = 0.0
    connected_since: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "is_healthy": self.is_healthy,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "latency_ms": self.latency_ms,
            "uptime_seconds": self.uptime_seconds,
        }


@dataclass
class ConnectionEvent:
    """Connection event for logging and alerting."""
    event_id: str
    connection_id: str
    event_type: ConnectionEventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "connection_id": self.connection_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "severity": self.severity,
        }


@dataclass
class ManagedConnection:
    """Internal representation of a managed connection."""
    config: ConnectionConfig
    connector: Any  # The actual connector instance
    health: ConnectionHealth
    is_active: bool = False
    reconnect_attempts: int = 0
    health_check_task: Optional[asyncio.Task] = None
    using_backup: bool = False


@dataclass
class OverallHealth:
    """Overall health status across all connections."""
    is_healthy: bool
    total_connections: int
    healthy_connections: int
    degraded_connections: int
    failed_connections: int
    connection_states: Dict[str, ConnectionState]
    critical_connections_healthy: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "total_connections": self.total_connections,
            "healthy_connections": self.healthy_connections,
            "degraded_connections": self.degraded_connections,
            "failed_connections": self.failed_connections,
            "critical_connections_healthy": self.critical_connections_healthy,
        }


class ConnectionManager:
    """
    Centralized connection manager for all integration endpoints.

    Manages:
    - DCS connections
    - BMS connections
    - Historian connections
    - OPC-UA connections

    Features:
    - Health monitoring with configurable intervals
    - Automatic reconnection with exponential backoff
    - Failover to backup endpoints
    - Event logging and alerting
    """

    def __init__(self, event_callback: Optional[Callable[[ConnectionEvent], None]] = None):
        """
        Initialize connection manager.

        Args:
            event_callback: Optional callback for connection events
        """
        self._connections: Dict[str, ManagedConnection] = {}
        self._event_callback = event_callback
        self._event_history: List[ConnectionEvent] = []
        self._lock = asyncio.Lock()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "total_connections": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "reconnections": 0,
            "failovers": 0,
            "health_checks": 0,
        }

        # Critical connections that must be healthy
        self._critical_connections: List[str] = []

        logger.info("ConnectionManager initialized")

    async def register_connection(
        self,
        config: ConnectionConfig,
        connector: Any,
        is_critical: bool = False,
    ) -> None:
        """
        Register a connection for management.

        Args:
            config: Connection configuration
            connector: Connector instance (DCSConnector, OPCUAClient, etc.)
            is_critical: Whether this is a critical connection
        """
        async with self._lock:
            health = ConnectionHealth(
                connection_id=config.connection_id,
                is_healthy=False,
                state=ConnectionState.DISCONNECTED,
            )

            managed = ManagedConnection(
                config=config,
                connector=connector,
                health=health,
            )

            self._connections[config.connection_id] = managed
            self._stats["total_connections"] += 1

            if is_critical:
                self._critical_connections.append(config.connection_id)

            logger.info(f"Registered connection: {config.connection_id} ({config.connection_type.value})")

    async def unregister_connection(self, connection_id: str) -> None:
        """Remove a connection from management."""
        async with self._lock:
            if connection_id in self._connections:
                managed = self._connections[connection_id]

                # Stop health check task
                if managed.health_check_task:
                    managed.health_check_task.cancel()

                # Disconnect
                try:
                    await managed.connector.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting {connection_id}: {e}")

                del self._connections[connection_id]

                if connection_id in self._critical_connections:
                    self._critical_connections.remove(connection_id)

                logger.info(f"Unregistered connection: {connection_id}")

    async def manage_connections(self) -> Dict[str, bool]:
        """
        Start managing all registered connections.

        Connects all enabled connections and starts health monitoring.

        Returns:
            Dict of connection_id to connection success status
        """
        self._running = True
        results = {}

        for conn_id, managed in self._connections.items():
            if managed.config.enabled:
                success = await self._connect(conn_id)
                results[conn_id] = success

                # Start health monitoring for this connection
                managed.health_check_task = asyncio.create_task(
                    self._health_check_loop(conn_id)
                )

        # Start overall monitor
        self._monitor_task = asyncio.create_task(self._overall_monitor())

        logger.info(f"Started managing {len(results)} connections")
        return results

    async def stop_managing(self) -> None:
        """Stop managing connections and disconnect all."""
        self._running = False

        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

        # Stop all health checks and disconnect
        for conn_id, managed in self._connections.items():
            if managed.health_check_task:
                managed.health_check_task.cancel()
                managed.health_check_task = None

            try:
                await managed.connector.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting {conn_id}: {e}")

            managed.health.state = ConnectionState.DISCONNECTED
            managed.health.is_healthy = False
            managed.is_active = False

        logger.info("Stopped managing all connections")

    async def monitor_connection_health(self) -> OverallHealth:
        """
        Get overall health status of all connections.

        Returns:
            OverallHealth with aggregated status
        """
        healthy_count = 0
        degraded_count = 0
        failed_count = 0
        connection_states = {}

        for conn_id, managed in self._connections.items():
            connection_states[conn_id] = managed.health.state

            if managed.health.state == ConnectionState.CONNECTED:
                healthy_count += 1
            elif managed.health.state == ConnectionState.DEGRADED:
                degraded_count += 1
            else:
                failed_count += 1

        # Check critical connections
        critical_healthy = all(
            self._connections[conn_id].health.is_healthy
            for conn_id in self._critical_connections
            if conn_id in self._connections
        )

        total = len(self._connections)
        is_healthy = (healthy_count == total) and critical_healthy

        return OverallHealth(
            is_healthy=is_healthy,
            total_connections=total,
            healthy_connections=healthy_count,
            degraded_connections=degraded_count,
            failed_connections=failed_count,
            connection_states=connection_states,
            critical_connections_healthy=critical_healthy,
        )

    async def reconnect_on_failure(self, connection_id: str) -> bool:
        """
        Attempt to reconnect a failed connection.

        Args:
            connection_id: ID of connection to reconnect

        Returns:
            True if reconnection successful
        """
        if connection_id not in self._connections:
            logger.warning(f"Connection {connection_id} not found")
            return False

        managed = self._connections[connection_id]
        managed.health.state = ConnectionState.RECONNECTING

        self._log_event(
            connection_id,
            ConnectionEventType.RECONNECTING,
            f"Attempting reconnection (attempt {managed.reconnect_attempts + 1})",
            severity="warning",
        )

        # Exponential backoff
        delay = min(
            managed.config.reconnect_delay_seconds * (2 ** managed.reconnect_attempts),
            60.0  # Max 60 seconds
        )

        await asyncio.sleep(delay)

        success = await self._connect(connection_id)

        if success:
            managed.reconnect_attempts = 0
            self._stats["reconnections"] += 1

            self._log_event(
                connection_id,
                ConnectionEventType.RECONNECTED,
                "Reconnection successful",
                severity="info",
            )
        else:
            managed.reconnect_attempts += 1

            if managed.reconnect_attempts >= managed.config.max_reconnect_attempts:
                # Try failover
                if managed.config.backup_endpoint:
                    await self.failover_to_backup(connection_id)
                else:
                    managed.health.state = ConnectionState.FAILED
                    self._log_event(
                        connection_id,
                        ConnectionEventType.FAILED,
                        f"Max reconnection attempts ({managed.config.max_reconnect_attempts}) exceeded",
                        severity="critical",
                    )

        return success

    async def failover_to_backup(self, connection_id: str) -> bool:
        """
        Failover to backup endpoint.

        Args:
            connection_id: ID of connection to failover

        Returns:
            True if failover successful
        """
        if connection_id not in self._connections:
            logger.warning(f"Connection {connection_id} not found")
            return False

        managed = self._connections[connection_id]

        if not managed.config.backup_endpoint:
            logger.warning(f"No backup endpoint configured for {connection_id}")
            return False

        self._log_event(
            connection_id,
            ConnectionEventType.FAILOVER,
            f"Failing over to backup endpoint: {managed.config.backup_endpoint}",
            severity="warning",
        )

        # Swap endpoints
        original_endpoint = managed.config.endpoint
        managed.config.endpoint = managed.config.backup_endpoint
        managed.config.backup_endpoint = original_endpoint
        managed.using_backup = not managed.using_backup
        managed.reconnect_attempts = 0

        success = await self._connect(connection_id)

        if success:
            self._stats["failovers"] += 1
            logger.info(f"Failover successful for {connection_id}")
        else:
            # Swap back
            managed.config.backup_endpoint = managed.config.endpoint
            managed.config.endpoint = original_endpoint
            managed.using_backup = not managed.using_backup

            self._log_event(
                connection_id,
                ConnectionEventType.FAILED,
                "Failover failed, backup endpoint unreachable",
                severity="critical",
            )

        return success

    def log_connection_events(self) -> List[ConnectionEvent]:
        """Get recent connection events."""
        return self._event_history[-100:]

    async def _connect(self, connection_id: str) -> bool:
        """Internal method to connect a managed connection."""
        if connection_id not in self._connections:
            return False

        managed = self._connections[connection_id]
        managed.health.state = ConnectionState.CONNECTING

        try:
            # Call connector's connect method
            # The connector should have a connect() method
            connector = managed.connector

            if hasattr(connector, 'connect'):
                # Build config based on connection type
                await self._do_connect(managed)

            managed.health.state = ConnectionState.CONNECTED
            managed.health.is_healthy = True
            managed.health.connected_since = datetime.now(timezone.utc)
            managed.health.consecutive_failures = 0
            managed.is_active = True
            self._stats["successful_connections"] += 1

            self._log_event(
                connection_id,
                ConnectionEventType.CONNECTED,
                f"Connected to {managed.config.endpoint}",
            )

            return True

        except Exception as e:
            managed.health.state = ConnectionState.FAILED
            managed.health.is_healthy = False
            managed.health.error_message = str(e)
            managed.health.consecutive_failures += 1
            managed.is_active = False
            self._stats["failed_connections"] += 1

            self._log_event(
                connection_id,
                ConnectionEventType.FAILED,
                f"Connection failed: {e}",
                severity="error",
            )

            return False

    async def _do_connect(self, managed: ManagedConnection) -> None:
        """Execute the actual connection based on connection type."""
        connector = managed.connector
        config = managed.config

        # Simulate connection for different types
        if config.connection_type == ConnectionType.DCS:
            if hasattr(connector, 'connect'):
                from .dcs_connector import DCSConfig, DCSType
                dcs_config = DCSConfig(
                    dcs_type=DCSType.HONEYWELL_EXPERION,
                    host=config.endpoint,
                )
                await connector.connect(dcs_config)

        elif config.connection_type == ConnectionType.HISTORIAN:
            if hasattr(connector, 'connect'):
                from .historian_connector import HistorianConfig, HistorianType
                hist_config = HistorianConfig(
                    historian_type=HistorianType.OSISOFT_PI,
                    host=config.endpoint,
                )
                await connector.connect(hist_config)

        elif config.connection_type == ConnectionType.OPCUA:
            if hasattr(connector, 'connect'):
                from .opcua_client import SecurityConfig
                security = SecurityConfig()
                await connector.connect(config.endpoint, security)

        elif config.connection_type == ConnectionType.BMS:
            # BMS is typically read-only and doesn't require explicit connect
            pass

    async def _health_check_loop(self, connection_id: str) -> None:
        """Health check loop for a connection."""
        while self._running and connection_id in self._connections:
            managed = self._connections[connection_id]

            if not managed.config.enabled:
                await asyncio.sleep(managed.config.health_check_interval_seconds)
                continue

            try:
                await self._check_connection_health(connection_id)
                self._stats["health_checks"] += 1

            except Exception as e:
                logger.error(f"Health check error for {connection_id}: {e}")

            await asyncio.sleep(managed.config.health_check_interval_seconds)

    async def _check_connection_health(self, connection_id: str) -> None:
        """Check health of a single connection."""
        if connection_id not in self._connections:
            return

        managed = self._connections[connection_id]
        now = datetime.now(timezone.utc)

        # Update last health check time
        managed.health.last_health_check = now

        # Check if connector is connected
        connector = managed.connector
        is_connected = False

        if hasattr(connector, 'is_connected'):
            is_connected = connector.is_connected
        elif hasattr(connector, '_status'):
            is_connected = str(connector._status) == "connected"

        if is_connected:
            managed.health.is_healthy = True
            managed.health.state = ConnectionState.CONNECTED
            managed.health.last_successful_communication = now
            managed.health.consecutive_failures = 0

            # Calculate uptime
            if managed.health.connected_since:
                managed.health.uptime_seconds = (now - managed.health.connected_since).total_seconds()

            # Simulate latency check
            managed.health.latency_ms = 50.0  # Simulated

        else:
            managed.health.consecutive_failures += 1

            if managed.health.consecutive_failures >= 3:
                managed.health.is_healthy = False

                if managed.health.state == ConnectionState.CONNECTED:
                    self._log_event(
                        connection_id,
                        ConnectionEventType.HEALTH_CHECK_FAILED,
                        f"Health check failed {managed.health.consecutive_failures} times",
                        severity="warning",
                    )

                    # Trigger reconnection
                    asyncio.create_task(self.reconnect_on_failure(connection_id))

    async def _overall_monitor(self) -> None:
        """Overall monitoring task."""
        while self._running:
            try:
                health = await self.monitor_connection_health()

                if not health.critical_connections_healthy:
                    logger.warning("Critical connections unhealthy!")

            except Exception as e:
                logger.error(f"Overall monitor error: {e}")

            await asyncio.sleep(60)  # Check every minute

    def _log_event(
        self,
        connection_id: str,
        event_type: ConnectionEventType,
        message: str,
        severity: str = "info",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a connection event."""
        import uuid
        event = ConnectionEvent(
            event_id=str(uuid.uuid4()),
            connection_id=connection_id,
            event_type=event_type,
            message=message,
            severity=severity,
            details=details or {},
        )

        self._event_history.append(event)

        # Trim history
        if len(self._event_history) > 1000:
            self._event_history = self._event_history[-500:]

        # Log to logger
        log_method = getattr(logger, severity, logger.info)
        log_method(f"[{connection_id}] {event_type.value}: {message}")

        # Call event callback
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def get_connection(self, connection_id: str) -> Optional[Any]:
        """Get a connector by ID."""
        if connection_id in self._connections:
            return self._connections[connection_id].connector
        return None

    def get_connection_health(self, connection_id: str) -> Optional[ConnectionHealth]:
        """Get health status for a specific connection."""
        if connection_id in self._connections:
            return self._connections[connection_id].health
        return None

    def get_all_connections(self) -> Dict[str, ConnectionConfig]:
        """Get all registered connection configs."""
        return {
            conn_id: managed.config
            for conn_id, managed in self._connections.items()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get connection manager statistics."""
        return {
            **self._stats,
            "active_connections": len([
                c for c in self._connections.values()
                if c.is_active
            ]),
            "critical_connections": len(self._critical_connections),
            "event_history_size": len(self._event_history),
        }
