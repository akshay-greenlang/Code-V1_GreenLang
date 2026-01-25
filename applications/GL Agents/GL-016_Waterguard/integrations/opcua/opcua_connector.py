"""
GL-016 Waterguard OPC-UA Connector

Production-grade OPC-UA client using asyncua library for connecting to
industrial control systems. Features certificate-based authentication,
automatic reconnection, subscription management, and health monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, SecretStr

from integrations.opcua.opcua_schemas import (
    OPCUAQuality,
    SubscriptionGroup,
    TagSubscription,
    TagValue,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Connection State
# =============================================================================

class ConnectionState(Enum):
    """OPC-UA connection states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    FAILED = auto()
    CLOSED = auto()


# =============================================================================
# Configuration
# =============================================================================

class SecurityMode(str, Enum):
    """OPC-UA security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class SecurityPolicy(str, Enum):
    """OPC-UA security policies."""
    NONE = "None"
    BASIC128RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256SHA256 = "Basic256Sha256"
    AES128_SHA256_RSAOAEP = "Aes128_Sha256_RsaOaep"
    AES256_SHA256_RSAPSS = "Aes256_Sha256_RsaPss"


class AuthenticationType(str, Enum):
    """OPC-UA authentication types."""
    ANONYMOUS = "anonymous"
    USERNAME_PASSWORD = "username_password"
    CERTIFICATE = "certificate"


class OPCUAConfig(BaseModel):
    """
    OPC-UA connection configuration.

    Certificates and credentials should be loaded from a secure vault.
    """

    # Connection
    endpoint_url: str = Field(
        ...,
        description="OPC-UA server endpoint URL"
    )
    application_name: str = Field(
        default="Waterguard-GL016",
        description="Client application name"
    )
    application_uri: str = Field(
        default="urn:waterguard:gl016:client",
        description="Client application URI"
    )

    # Security
    security_mode: SecurityMode = Field(
        default=SecurityMode.SIGN_AND_ENCRYPT,
        description="Security mode"
    )
    security_policy: SecurityPolicy = Field(
        default=SecurityPolicy.BASIC256SHA256,
        description="Security policy"
    )
    authentication_type: AuthenticationType = Field(
        default=AuthenticationType.CERTIFICATE,
        description="Authentication type"
    )

    # Certificates (paths or PEM content)
    client_certificate_path: Optional[str] = Field(
        default=None,
        description="Path to client certificate"
    )
    client_private_key_path: Optional[str] = Field(
        default=None,
        description="Path to client private key"
    )
    server_certificate_path: Optional[str] = Field(
        default=None,
        description="Path to server certificate for trust"
    )
    trusted_certificates_dir: Optional[str] = Field(
        default=None,
        description="Directory containing trusted certificates"
    )

    # Username/password auth
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[SecretStr] = Field(default=None, description="Password")

    # Timeouts
    connection_timeout_seconds: int = Field(
        default=30,
        description="Connection timeout"
    )
    request_timeout_seconds: int = Field(
        default=30,
        description="Request timeout"
    )
    session_timeout_seconds: int = Field(
        default=3600,
        description="Session timeout"
    )

    # Reconnection
    auto_reconnect: bool = Field(
        default=True,
        description="Enable automatic reconnection"
    )
    reconnect_delay_seconds: float = Field(
        default=5.0,
        description="Initial reconnect delay"
    )
    reconnect_max_delay_seconds: float = Field(
        default=300.0,
        description="Maximum reconnect delay"
    )
    reconnect_backoff_multiplier: float = Field(
        default=2.0,
        description="Reconnect delay multiplier"
    )
    max_reconnect_attempts: int = Field(
        default=0,
        description="Max reconnect attempts (0 = unlimited)"
    )

    # Subscription defaults
    default_sampling_interval_ms: int = Field(
        default=1000,
        description="Default sampling interval"
    )
    default_publishing_interval_ms: int = Field(
        default=1000,
        description="Default publishing interval"
    )

    class Config:
        extra = "forbid"


# =============================================================================
# Connection Health
# =============================================================================

@dataclass
class ConnectionHealth:
    """Connection health telemetry."""

    state: ConnectionState = ConnectionState.DISCONNECTED
    last_connected: Optional[datetime] = None
    last_disconnected: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    reconnect_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_uptime_seconds: float = 0.0
    current_session_start: Optional[datetime] = None

    # Read/write stats
    reads_total: int = 0
    reads_success: int = 0
    reads_failed: int = 0
    writes_total: int = 0
    writes_success: int = 0
    writes_failed: int = 0
    avg_read_latency_ms: float = 0.0
    avg_write_latency_ms: float = 0.0

    @property
    def uptime_percent(self) -> float:
        """Calculate uptime percentage."""
        if self.successful_connections == 0:
            return 0.0
        total_time = (
            self.successful_connections *
            self.total_uptime_seconds /
            max(1, self.successful_connections)
        )
        return min(100.0, (self.total_uptime_seconds / max(1, total_time)) * 100)

    @property
    def read_success_rate(self) -> float:
        """Calculate read success rate."""
        if self.reads_total == 0:
            return 100.0
        return (self.reads_success / self.reads_total) * 100


# =============================================================================
# OPC-UA Connector
# =============================================================================

class WaterguardOPCUAConnector:
    """
    Production OPC-UA connector for Waterguard GL-016.

    Features:
    - Certificate-based authentication
    - Automatic reconnection with exponential backoff
    - Subscription management with callbacks
    - Connection health monitoring
    - Thread-safe async operations

    Example:
        config = OPCUAConfig(endpoint_url="opc.tcp://localhost:4840")
        async with WaterguardOPCUAConnector(config) as connector:
            # Subscribe to tags
            await connector.subscribe(subscriptions, callback)

            # Read values
            values = await connector.read_tags(["ns=2;s=AI_PO4_001"])

            # Write values
            await connector.write_tag("ns=2;s=AO_PUMP_001", 50.0)
    """

    def __init__(
        self,
        config: OPCUAConfig,
        on_state_change: Optional[Callable[[ConnectionState], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Initialize OPC-UA connector.

        Args:
            config: Connection configuration
            on_state_change: Callback for state changes
            on_error: Callback for errors
        """
        self.config = config
        self._on_state_change = on_state_change
        self._on_error = on_error

        # Connection state
        self._client: Optional[Any] = None
        self._state = ConnectionState.DISCONNECTED
        self._health = ConnectionHealth()
        self._lock = asyncio.Lock()

        # Subscriptions
        self._subscriptions: Dict[UUID, Any] = {}
        self._monitored_items: Dict[str, Any] = {}
        self._callbacks: Dict[str, Callable[[TagValue], None]] = {}

        # Reconnection
        self._reconnect_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._current_reconnect_delay = config.reconnect_delay_seconds

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def health(self) -> ConnectionHealth:
        """Get connection health."""
        return self._health

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    async def connect(self) -> None:
        """
        Establish connection to OPC-UA server.

        Raises:
            ConnectionError: If connection fails
        """
        async with self._lock:
            if self._state == ConnectionState.CONNECTED:
                return

            self._set_state(ConnectionState.CONNECTING)

            try:
                # Import asyncua here for optional dependency
                from asyncua import Client
                from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256

                # Create client
                self._client = Client(url=self.config.endpoint_url)
                self._client.application_uri = self.config.application_uri
                self._client.name = self.config.application_name

                # Configure security
                await self._configure_security()

                # Set timeouts
                self._client.session_timeout = self.config.session_timeout_seconds * 1000

                # Connect
                await asyncio.wait_for(
                    self._client.connect(),
                    timeout=self.config.connection_timeout_seconds
                )

                # Update state
                self._set_state(ConnectionState.CONNECTED)
                self._health.last_connected = datetime.utcnow()
                self._health.current_session_start = datetime.utcnow()
                self._health.successful_connections += 1
                self._health.reconnect_attempts = 0
                self._current_reconnect_delay = self.config.reconnect_delay_seconds

                logger.info(f"Connected to OPC-UA server: {self.config.endpoint_url}")

            except asyncio.TimeoutError:
                self._handle_connection_error("Connection timeout")
                raise ConnectionError("Connection timeout")

            except Exception as e:
                self._handle_connection_error(str(e))
                raise ConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from OPC-UA server."""
        async with self._lock:
            # Stop reconnection task
            self._shutdown_event.set()
            if self._reconnect_task:
                self._reconnect_task.cancel()
                try:
                    await self._reconnect_task
                except asyncio.CancelledError:
                    pass

            # Clear subscriptions
            self._subscriptions.clear()
            self._monitored_items.clear()
            self._callbacks.clear()

            # Disconnect client
            if self._client:
                try:
                    await self._client.disconnect()
                except Exception as e:
                    logger.warning(f"Error during disconnect: {e}")
                finally:
                    self._client = None

            # Update state
            self._set_state(ConnectionState.CLOSED)
            self._health.last_disconnected = datetime.utcnow()

            # Calculate uptime
            if self._health.current_session_start:
                session_duration = (
                    datetime.utcnow() - self._health.current_session_start
                ).total_seconds()
                self._health.total_uptime_seconds += session_duration
                self._health.current_session_start = None

            logger.info("Disconnected from OPC-UA server")

    async def __aenter__(self) -> "WaterguardOPCUAConnector":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def _configure_security(self) -> None:
        """Configure security settings on client."""
        if self.config.security_mode == SecurityMode.NONE:
            return

        if self.config.authentication_type == AuthenticationType.CERTIFICATE:
            # Load certificates
            if self.config.client_certificate_path and self.config.client_private_key_path:
                await self._client.set_security(
                    self._get_security_policy(),
                    certificate=self.config.client_certificate_path,
                    private_key=self.config.client_private_key_path,
                    server_certificate=self.config.server_certificate_path,
                    mode=self._get_security_mode(),
                )

        elif self.config.authentication_type == AuthenticationType.USERNAME_PASSWORD:
            if self.config.username and self.config.password:
                self._client.set_user(self.config.username)
                self._client.set_password(self.config.password.get_secret_value())

    def _get_security_policy(self) -> Any:
        """Get security policy class."""
        from asyncua.crypto.security_policies import (
            SecurityPolicyBasic256Sha256,
            SecurityPolicyBasic256,
            SecurityPolicyBasic128Rsa15,
        )

        policies = {
            SecurityPolicy.BASIC256SHA256: SecurityPolicyBasic256Sha256,
            SecurityPolicy.BASIC256: SecurityPolicyBasic256,
            SecurityPolicy.BASIC128RSA15: SecurityPolicyBasic128Rsa15,
        }
        return policies.get(self.config.security_policy)

    def _get_security_mode(self) -> Any:
        """Get security mode enum."""
        from asyncua.ua import MessageSecurityMode

        modes = {
            SecurityMode.NONE: MessageSecurityMode.None_,
            SecurityMode.SIGN: MessageSecurityMode.Sign,
            SecurityMode.SIGN_AND_ENCRYPT: MessageSecurityMode.SignAndEncrypt,
        }
        return modes.get(self.config.security_mode, MessageSecurityMode.None_)

    def _set_state(self, new_state: ConnectionState) -> None:
        """Update connection state and notify."""
        old_state = self._state
        self._state = new_state
        self._health.state = new_state

        if old_state != new_state:
            logger.info(f"Connection state changed: {old_state.name} -> {new_state.name}")
            if self._on_state_change:
                try:
                    self._on_state_change(new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")

    def _handle_connection_error(self, error: str) -> None:
        """Handle connection error."""
        self._health.last_error = error
        self._health.last_error_time = datetime.utcnow()
        self._health.failed_connections += 1
        self._set_state(ConnectionState.FAILED)

        if self._on_error:
            try:
                self._on_error(ConnectionError(error))
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    async def _start_reconnection(self) -> None:
        """Start automatic reconnection task."""
        if not self.config.auto_reconnect:
            return

        if self._reconnect_task and not self._reconnect_task.done():
            return

        self._reconnect_task = asyncio.create_task(self._reconnection_loop())

    async def _reconnection_loop(self) -> None:
        """Reconnection loop with exponential backoff."""
        while not self._shutdown_event.is_set():
            if self._state == ConnectionState.CONNECTED:
                await asyncio.sleep(1)
                continue

            if self.config.max_reconnect_attempts > 0:
                if self._health.reconnect_attempts >= self.config.max_reconnect_attempts:
                    logger.error("Max reconnection attempts reached")
                    self._set_state(ConnectionState.FAILED)
                    return

            self._set_state(ConnectionState.RECONNECTING)
            self._health.reconnect_attempts += 1

            logger.info(
                f"Reconnection attempt {self._health.reconnect_attempts} "
                f"in {self._current_reconnect_delay:.1f}s"
            )

            await asyncio.sleep(self._current_reconnect_delay)

            try:
                await self.connect()

                # Restore subscriptions
                await self._restore_subscriptions()

            except Exception as e:
                logger.warning(f"Reconnection failed: {e}")

                # Increase delay with backoff
                self._current_reconnect_delay = min(
                    self._current_reconnect_delay * self.config.reconnect_backoff_multiplier,
                    self.config.reconnect_max_delay_seconds
                )

    async def _restore_subscriptions(self) -> None:
        """Restore subscriptions after reconnection."""
        # This would re-subscribe to previously registered tags
        # Implementation depends on stored subscription state
        pass

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def read_tag(self, node_id: str) -> TagValue:
        """
        Read a single tag value.

        Args:
            node_id: OPC-UA node ID

        Returns:
            Tag value with quality

        Raises:
            RuntimeError: If not connected
        """
        values = await self.read_tags([node_id])
        return values[0]

    async def read_tags(self, node_ids: List[str]) -> List[TagValue]:
        """
        Read multiple tag values.

        Args:
            node_ids: List of OPC-UA node IDs

        Returns:
            List of tag values with quality
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to OPC-UA server")

        start_time = time.time()
        self._health.reads_total += len(node_ids)

        try:
            from asyncua import ua

            # Parse node IDs
            nodes = []
            for node_id in node_ids:
                node = self._client.get_node(node_id)
                nodes.append(node)

            # Read values
            results = await self._client.read_values(nodes)

            # Build TagValue objects
            tag_values = []
            now = datetime.utcnow()

            for i, (node_id, result) in enumerate(zip(node_ids, results)):
                # Get data value with timestamps and quality
                try:
                    dv = await nodes[i].read_data_value()

                    tag_value = TagValue(
                        node_id=node_id,
                        tag_name=node_id.split(";")[-1].replace("s=", ""),
                        value=dv.Value.Value if dv.Value else None,
                        source_timestamp=dv.SourceTimestamp or now,
                        server_timestamp=dv.ServerTimestamp or now,
                        quality=OPCUAQuality.from_status_code(
                            dv.StatusCode.value if dv.StatusCode else 0
                        ),
                        status_code=dv.StatusCode.value if dv.StatusCode else 0,
                    )
                    tag_values.append(tag_value)
                    self._health.reads_success += 1

                except Exception as e:
                    # Create bad quality value on error
                    tag_values.append(TagValue(
                        node_id=node_id,
                        tag_name=node_id.split(";")[-1].replace("s=", ""),
                        value=None,
                        source_timestamp=now,
                        server_timestamp=now,
                        quality=OPCUAQuality.BAD,
                        status_code=0x80000000,
                    ))
                    self._health.reads_failed += 1
                    logger.warning(f"Failed to read {node_id}: {e}")

            # Update latency
            elapsed_ms = (time.time() - start_time) * 1000
            self._health.avg_read_latency_ms = (
                (self._health.avg_read_latency_ms + elapsed_ms) / 2
            )

            return tag_values

        except Exception as e:
            self._health.reads_failed += len(node_ids)
            logger.error(f"Batch read failed: {e}")
            raise

    # =========================================================================
    # Write Operations
    # =========================================================================

    async def write_tag(
        self,
        node_id: str,
        value: Union[float, int, bool, str],
        check_quality: bool = True,
    ) -> bool:
        """
        Write a value to a tag.

        Args:
            node_id: OPC-UA node ID
            value: Value to write
            check_quality: Check current quality before write

        Returns:
            True if write successful

        Raises:
            RuntimeError: If not connected
            ValueError: If quality check fails
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to OPC-UA server")

        # Optionally check current quality - NEVER overwrite bad with "last good"
        if check_quality:
            current = await self.read_tag(node_id)
            if current.quality.is_bad:
                raise ValueError(
                    f"Cannot write to {node_id}: current quality is {current.quality.value}. "
                    "Fix the underlying issue before writing."
                )

        start_time = time.time()
        self._health.writes_total += 1

        try:
            from asyncua import ua

            node = self._client.get_node(node_id)

            # Determine data type
            dv = ua.DataValue(ua.Variant(value))
            await node.write_value(dv)

            self._health.writes_success += 1

            # Update latency
            elapsed_ms = (time.time() - start_time) * 1000
            self._health.avg_write_latency_ms = (
                (self._health.avg_write_latency_ms + elapsed_ms) / 2
            )

            logger.debug(f"Write successful: {node_id} = {value}")
            return True

        except Exception as e:
            self._health.writes_failed += 1
            logger.error(f"Write failed for {node_id}: {e}")
            raise

    # =========================================================================
    # Subscription Operations
    # =========================================================================

    async def subscribe(
        self,
        subscriptions: List[TagSubscription],
        callback: Callable[[TagValue], None],
    ) -> UUID:
        """
        Subscribe to tag value changes.

        Args:
            subscriptions: List of tag subscriptions
            callback: Callback function for value changes

        Returns:
            Subscription group ID
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to OPC-UA server")

        group_id = uuid4()

        try:
            from asyncua import ua

            # Create subscription
            subscription = await self._client.create_subscription(
                period=subscriptions[0].publishing_interval_ms if subscriptions else 1000,
                handler=self._create_subscription_handler(callback),
            )

            self._subscriptions[group_id] = subscription

            # Add monitored items
            for sub in subscriptions:
                node = self._client.get_node(sub.node_id)

                monitoring_params = ua.MonitoringParameters(
                    SamplingInterval=sub.sampling_interval_ms,
                    QueueSize=sub.queue_size,
                    DiscardOldest=sub.discard_oldest,
                )

                # Add deadband filter if configured
                if sub.deadband_value > 0:
                    if sub.deadband_type == "absolute":
                        monitoring_params.Filter = ua.DataChangeFilter(
                            Trigger=ua.DataChangeTrigger.StatusValue,
                            DeadbandType=ua.DeadbandType.Absolute,
                            DeadbandValue=sub.deadband_value,
                        )
                    elif sub.deadband_type == "percent":
                        monitoring_params.Filter = ua.DataChangeFilter(
                            Trigger=ua.DataChangeTrigger.StatusValue,
                            DeadbandType=ua.DeadbandType.Percent,
                            DeadbandValue=sub.deadband_value,
                        )

                monitored_item = await subscription.subscribe_data_change(
                    node,
                    monitoring_params,
                )

                self._monitored_items[sub.node_id] = monitored_item
                self._callbacks[sub.node_id] = callback

            logger.info(
                f"Created subscription {group_id} with {len(subscriptions)} items"
            )
            return group_id

        except Exception as e:
            logger.error(f"Failed to create subscription: {e}")
            raise

    async def unsubscribe(self, group_id: UUID) -> None:
        """
        Unsubscribe from a subscription group.

        Args:
            group_id: Subscription group ID
        """
        if group_id in self._subscriptions:
            subscription = self._subscriptions[group_id]
            try:
                await subscription.delete()
            except Exception as e:
                logger.warning(f"Error deleting subscription: {e}")
            del self._subscriptions[group_id]
            logger.info(f"Deleted subscription {group_id}")

    def _create_subscription_handler(
        self,
        callback: Callable[[TagValue], None],
    ) -> Any:
        """Create subscription handler class."""

        class SubscriptionHandler:
            def __init__(self, connector: WaterguardOPCUAConnector, cb: Callable):
                self.connector = connector
                self.callback = cb

            def datachange_notification(self, node, val, data):
                """Handle data change notification."""
                try:
                    now = datetime.utcnow()
                    node_id = node.nodeid.to_string()

                    tag_value = TagValue(
                        node_id=node_id,
                        tag_name=node_id.split(";")[-1].replace("s=", ""),
                        value=val,
                        source_timestamp=data.monitored_item.Value.SourceTimestamp or now,
                        server_timestamp=data.monitored_item.Value.ServerTimestamp or now,
                        quality=OPCUAQuality.from_status_code(
                            data.monitored_item.Value.StatusCode.value
                            if data.monitored_item.Value.StatusCode
                            else 0
                        ),
                        status_code=(
                            data.monitored_item.Value.StatusCode.value
                            if data.monitored_item.Value.StatusCode
                            else 0
                        ),
                    )

                    self.callback(tag_value)

                except Exception as e:
                    logger.error(f"Error in subscription handler: {e}")

        return SubscriptionHandler(self, callback)
