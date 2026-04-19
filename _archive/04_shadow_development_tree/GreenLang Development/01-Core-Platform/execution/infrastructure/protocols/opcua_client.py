"""
OPC-UA Client Implementation for GreenLang Agents

This module provides a production-ready OPC-UA client for connecting
to industrial automation systems and collecting data.

Features:
- Automatic reconnection with exponential backoff
- Connection pooling
- Subscription management
- Historical data queries
- Method calls
- Security configuration

Example:
    >>> client = OPCUAClient(config)
    >>> await client.connect()
    >>> value = await client.read_value("ns=2;s=Temperature")
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from asyncua import Client, ua
    from asyncua.common.subscription import Subscription
    ASYNCUA_AVAILABLE = True
except ImportError:
    ASYNCUA_AVAILABLE = False
    Client = None
    ua = None

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class OPCUAClientConfig:
    """Configuration for OPC-UA client."""
    endpoint: str = "opc.tcp://localhost:4840/greenlang/"
    security_policy: str = "Basic256Sha256"
    security_mode: str = "SignAndEncrypt"
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    timeout_ms: int = 30000
    keepalive_interval_ms: int = 10000
    reconnect_interval_ms: int = 5000
    max_reconnect_attempts: int = 10
    subscription_publishing_interval_ms: int = 1000


class DataChangeNotification(BaseModel):
    """Data change notification model."""
    node_id: str = Field(..., description="Node that changed")
    value: Any = Field(..., description="New value")
    source_timestamp: datetime = Field(..., description="Source timestamp")
    server_timestamp: datetime = Field(..., description="Server timestamp")
    status_code: int = Field(default=0, description="Status code")


class HistoricalDataPoint(BaseModel):
    """Historical data point model."""
    timestamp: datetime = Field(..., description="Data point timestamp")
    value: Any = Field(..., description="Data point value")
    status_code: int = Field(default=0, description="Quality status code")


class OPCUAClient:
    """
    Production-ready OPC-UA client for GreenLang agents.

    This client provides reliable connection to OPC-UA servers with
    automatic reconnection, subscription management, and historical
    data access.

    Attributes:
        config: Client configuration
        state: Current connection state
        client: AsyncUA client instance
        subscriptions: Active subscriptions

    Example:
        >>> config = OPCUAClientConfig(
        ...     endpoint="opc.tcp://plc.factory.local:4840/",
        ...     username="reader",
        ...     password="secret"
        ... )
        >>> client = OPCUAClient(config)
        >>> async with client:
        ...     value = await client.read_value("ns=2;s=Sensor1")
    """

    def __init__(self, config: OPCUAClientConfig):
        """
        Initialize OPC-UA client.

        Args:
            config: Client configuration

        Raises:
            ImportError: If asyncua is not installed
        """
        if not ASYNCUA_AVAILABLE:
            raise ImportError(
                "asyncua is required for OPC-UA support. "
                "Install with: pip install asyncua"
            )

        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.client: Optional[Client] = None
        self.subscriptions: Dict[str, Subscription] = {}
        self._data_change_handlers: Dict[str, List[Callable]] = {}
        self._reconnect_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._shutdown = False

        logger.info(f"OPCUAClient initialized for endpoint: {config.endpoint}")

    async def connect(self) -> None:
        """
        Connect to the OPC-UA server.

        Establishes connection with configured security settings
        and starts keepalive monitoring.

        Raises:
            ConnectionError: If connection fails
        """
        if self.state == ConnectionState.CONNECTED:
            logger.warning("Already connected")
            return

        self.state = ConnectionState.CONNECTING
        self._shutdown = False

        try:
            self.client = Client(
                self.config.endpoint,
                timeout=self.config.timeout_ms / 1000
            )

            # Configure security
            await self._configure_security()

            # Configure authentication
            if self.config.username and self.config.password:
                self.client.set_user(self.config.username)
                self.client.set_password(self.config.password)

            # Connect
            await self.client.connect()
            self.state = ConnectionState.CONNECTED

            # Start keepalive
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

            logger.info(f"Connected to OPC-UA server: {self.config.endpoint}")

        except Exception as e:
            self.state = ConnectionState.ERROR
            logger.error(f"Connection failed: {e}", exc_info=True)
            raise ConnectionError(f"Failed to connect to {self.config.endpoint}: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect from the OPC-UA server gracefully.

        Cancels subscriptions and cleans up resources.
        """
        self._shutdown = True

        # Cancel tasks
        if self._keepalive_task:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        # Delete subscriptions
        for sub_id, subscription in self.subscriptions.items():
            try:
                await subscription.delete()
                logger.debug(f"Deleted subscription {sub_id}")
            except Exception as e:
                logger.warning(f"Error deleting subscription {sub_id}: {e}")

        self.subscriptions.clear()

        # Disconnect
        if self.client:
            try:
                await self.client.disconnect()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        self.state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from OPC-UA server")

    async def _configure_security(self) -> None:
        """Configure client security settings."""
        if self.config.certificate_path and self.config.private_key_path:
            await self.client.load_client_certificate(self.config.certificate_path)
            await self.client.load_private_key(self.config.private_key_path)

            # Set security policy
            policy_map = {
                "None": ua.SecurityPolicyType.NoSecurity,
                "Basic256Sha256": ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt,
            }
            policy = policy_map.get(
                self.config.security_policy,
                ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt
            )
            self.client.set_security_string(f"{self.config.security_policy},{self.config.security_mode}")

    async def _keepalive_loop(self) -> None:
        """Keepalive loop to monitor connection health."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.keepalive_interval_ms / 1000)

                if self.state != ConnectionState.CONNECTED:
                    continue

                # Read server status as keepalive
                try:
                    server_node = self.client.get_node(ua.ObjectIds.Server_ServerStatus)
                    await server_node.read_value()
                except Exception as e:
                    logger.warning(f"Keepalive failed: {e}")
                    await self._handle_connection_loss()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Keepalive loop error: {e}")

    async def _handle_connection_loss(self) -> None:
        """Handle connection loss and attempt reconnection."""
        if self._shutdown or self.state == ConnectionState.RECONNECTING:
            return

        self.state = ConnectionState.RECONNECTING
        logger.warning("Connection lost, attempting reconnection...")

        for attempt in range(self.config.max_reconnect_attempts):
            if self._shutdown:
                break

            try:
                # Exponential backoff
                delay = min(
                    self.config.reconnect_interval_ms * (2 ** attempt) / 1000,
                    60  # Max 60 seconds
                )
                await asyncio.sleep(delay)

                # Attempt reconnection
                await self.client.connect()
                self.state = ConnectionState.CONNECTED

                # Resubscribe to all active subscriptions
                await self._resubscribe_all()

                logger.info(f"Reconnected after {attempt + 1} attempts")
                return

            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")

        self.state = ConnectionState.ERROR
        logger.error("Max reconnection attempts reached")

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all previously active subscriptions."""
        old_handlers = dict(self._data_change_handlers)
        self.subscriptions.clear()

        for node_id, handlers in old_handlers.items():
            for handler in handlers:
                await self.subscribe(node_id, handler)

    async def read_value(self, node_id: str) -> Any:
        """
        Read current value from a node.

        Args:
            node_id: OPC-UA node identifier

        Returns:
            Current node value

        Raises:
            ConnectionError: If not connected
            ValueError: If node not found
        """
        self._ensure_connected()

        try:
            node = self.client.get_node(node_id)
            value = await node.read_value()
            logger.debug(f"Read value from {node_id}: {value}")
            return value

        except Exception as e:
            logger.error(f"Failed to read {node_id}: {e}")
            raise ValueError(f"Failed to read node {node_id}: {e}") from e

    async def read_values(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Read multiple values in a single request.

        Args:
            node_ids: List of node identifiers

        Returns:
            Dictionary mapping node_id to value
        """
        self._ensure_connected()

        results = {}
        nodes = [self.client.get_node(nid) for nid in node_ids]

        values = await self.client.read_values(nodes)

        for node_id, value in zip(node_ids, values):
            results[node_id] = value

        return results

    async def write_value(
        self,
        node_id: str,
        value: Any,
        source_timestamp: Optional[datetime] = None
    ) -> str:
        """
        Write a value to a node.

        Args:
            node_id: OPC-UA node identifier
            value: Value to write
            source_timestamp: Optional source timestamp

        Returns:
            Provenance hash of the write operation

        Raises:
            ConnectionError: If not connected
            ValueError: If write fails
        """
        self._ensure_connected()

        try:
            node = self.client.get_node(node_id)

            # Create data value
            dv = ua.DataValue(ua.Variant(value))
            if source_timestamp:
                dv.SourceTimestamp = source_timestamp

            await node.write_value(dv)

            # Calculate provenance hash
            provenance_str = f"{node_id}:{value}:{datetime.utcnow().isoformat()}"
            provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

            logger.debug(f"Wrote value to {node_id}: {value}")
            return provenance_hash

        except Exception as e:
            logger.error(f"Failed to write to {node_id}: {e}")
            raise ValueError(f"Failed to write to node {node_id}: {e}") from e

    async def subscribe(
        self,
        node_id: str,
        callback: Callable[[DataChangeNotification], None],
        publishing_interval_ms: Optional[int] = None
    ) -> str:
        """
        Subscribe to data changes on a node.

        Args:
            node_id: Node to subscribe to
            callback: Function to call on data change
            publishing_interval_ms: Custom publishing interval

        Returns:
            Subscription ID
        """
        self._ensure_connected()

        interval = publishing_interval_ms or self.config.subscription_publishing_interval_ms

        # Create subscription if needed
        sub_id = str(uuid4())
        subscription = await self.client.create_subscription(
            interval,
            self._create_data_change_handler(sub_id)
        )

        # Subscribe to node
        node = self.client.get_node(node_id)
        handle = await subscription.subscribe_data_change(node)

        self.subscriptions[sub_id] = subscription

        # Store handler
        if node_id not in self._data_change_handlers:
            self._data_change_handlers[node_id] = []
        self._data_change_handlers[node_id].append(callback)

        logger.info(f"Subscribed to {node_id} with ID {sub_id}")
        return sub_id

    def _create_data_change_handler(self, sub_id: str) -> Callable:
        """Create a data change handler for a subscription."""
        async def handler(node, val, data):
            node_id = node.nodeid.to_string()
            notification = DataChangeNotification(
                node_id=node_id,
                value=val,
                source_timestamp=data.monitored_item.Value.SourceTimestamp or datetime.utcnow(),
                server_timestamp=data.monitored_item.Value.ServerTimestamp or datetime.utcnow(),
                status_code=data.monitored_item.Value.StatusCode.value
            )

            # Call all registered handlers
            for cb in self._data_change_handlers.get(node_id, []):
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(notification)
                    else:
                        cb(notification)
                except Exception as e:
                    logger.error(f"Data change handler error: {e}")

        return handler

    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Unsubscribe from a subscription.

        Args:
            subscription_id: ID of subscription to remove
        """
        if subscription_id in self.subscriptions:
            subscription = self.subscriptions.pop(subscription_id)
            await subscription.delete()
            logger.info(f"Unsubscribed from {subscription_id}")

    async def read_history(
        self,
        node_id: str,
        start_time: datetime,
        end_time: datetime,
        num_values: int = 1000
    ) -> List[HistoricalDataPoint]:
        """
        Read historical data from a node.

        Args:
            node_id: Node to read history from
            start_time: Start of time range
            end_time: End of time range
            num_values: Maximum number of values

        Returns:
            List of historical data points
        """
        self._ensure_connected()

        try:
            node = self.client.get_node(node_id)

            history = await node.read_raw_history(
                start_time,
                end_time,
                numvalues=num_values
            )

            data_points = []
            for dv in history:
                data_points.append(HistoricalDataPoint(
                    timestamp=dv.SourceTimestamp or datetime.utcnow(),
                    value=dv.Value.Value,
                    status_code=dv.StatusCode.value if dv.StatusCode else 0
                ))

            logger.info(f"Read {len(data_points)} historical values from {node_id}")
            return data_points

        except Exception as e:
            logger.error(f"Failed to read history from {node_id}: {e}")
            raise

    async def call_method(
        self,
        object_id: str,
        method_id: str,
        arguments: List[Any]
    ) -> List[Any]:
        """
        Call a method on the server.

        Args:
            object_id: Object containing the method
            method_id: Method node ID
            arguments: Method arguments

        Returns:
            Method return values
        """
        self._ensure_connected()

        try:
            object_node = self.client.get_node(object_id)
            method_node = self.client.get_node(method_id)

            result = await object_node.call_method(method_node, *arguments)

            logger.info(f"Called method {method_id} with {len(arguments)} args")
            return result if isinstance(result, list) else [result]

        except Exception as e:
            logger.error(f"Method call failed: {e}")
            raise

    async def browse(self, node_id: str = None) -> List[Dict[str, Any]]:
        """
        Browse nodes under a given node.

        Args:
            node_id: Parent node ID (None for root)

        Returns:
            List of child node information
        """
        self._ensure_connected()

        if node_id:
            node = self.client.get_node(node_id)
        else:
            node = self.client.nodes.root

        children = await node.get_children()
        results = []

        for child in children:
            browse_name = await child.read_browse_name()
            display_name = await child.read_display_name()
            node_class = await child.read_node_class()

            results.append({
                "node_id": child.nodeid.to_string(),
                "browse_name": browse_name.to_string(),
                "display_name": display_name.Text,
                "node_class": str(node_class),
            })

        return results

    def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError(
                f"Not connected to OPC-UA server (state: {self.state})"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get client statistics.

        Returns:
            Dictionary containing client statistics
        """
        return {
            "state": self.state.value,
            "endpoint": self.config.endpoint,
            "active_subscriptions": len(self.subscriptions),
            "monitored_nodes": len(self._data_change_handlers),
        }

    async def __aenter__(self) -> "OPCUAClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
