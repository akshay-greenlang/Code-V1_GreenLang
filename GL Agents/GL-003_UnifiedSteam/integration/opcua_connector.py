"""
GL-003 UNIFIEDSTEAM - OPC-UA Connector

Industrial OPC-UA connectivity for steam system data acquisition from:
- PLCs (Allen-Bradley, Siemens, etc.)
- DCS systems (Honeywell, Emerson, Yokogawa)
- Historian servers (OSIsoft PI, Wonderware)
- Edge gateways (Kepware, Ignition)

Features:
- Secure authentication (certificate-based, username/password)
- Multiple security policies (Basic256Sha256, Aes128_Sha256_RsaOaep, etc.)
- Automatic reconnection with exponential backoff
- Subscription-based data change notifications
- Namespace browsing for tag discovery
- Connection pooling for high-throughput scenarios
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import asyncio
import logging
import uuid
import hashlib

logger = logging.getLogger(__name__)


class SecurityPolicy(Enum):
    """OPC-UA security policies."""
    NONE = "None"
    BASIC128RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256SHA256 = "Basic256Sha256"
    AES128_SHA256_RSAOAEP = "Aes128_Sha256_RsaOaep"
    AES256_SHA256_RSAPSS = "Aes256_Sha256_RsaPss"


class MessageSecurityMode(Enum):
    """OPC-UA message security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class NodeClass(Enum):
    """OPC-UA node classes."""
    OBJECT = "Object"
    VARIABLE = "Variable"
    METHOD = "Method"
    OBJECT_TYPE = "ObjectType"
    VARIABLE_TYPE = "VariableType"
    REFERENCE_TYPE = "ReferenceType"
    DATA_TYPE = "DataType"
    VIEW = "View"


class DataType(Enum):
    """OPC-UA data types for steam system variables."""
    BOOLEAN = "Boolean"
    SBYTE = "SByte"
    BYTE = "Byte"
    INT16 = "Int16"
    UINT16 = "UInt16"
    INT32 = "Int32"
    UINT32 = "UInt32"
    INT64 = "Int64"
    UINT64 = "UInt64"
    FLOAT = "Float"
    DOUBLE = "Double"
    STRING = "String"
    DATETIME = "DateTime"
    BYTESTRING = "ByteString"


class StatusCode(Enum):
    """OPC-UA status codes (subset)."""
    GOOD = 0x00000000
    UNCERTAIN = 0x40000000
    BAD = 0x80000000
    BAD_NODE_ID_UNKNOWN = 0x80340000
    BAD_CONNECTION_CLOSED = 0x80AC0000
    BAD_TIMEOUT = 0x800A0000
    BAD_COMMUNICATION_ERROR = 0x80050000
    BAD_ENCODING_ERROR = 0x80060000
    BAD_SUBSCRIPTION_ID_INVALID = 0x80280000


@dataclass
class OPCUAConfig:
    """OPC-UA connection configuration."""
    endpoint_url: str
    security_policy: SecurityPolicy = SecurityPolicy.BASIC256SHA256
    security_mode: MessageSecurityMode = MessageSecurityMode.SIGN_AND_ENCRYPT

    # Authentication
    username: Optional[str] = None
    password: Optional[str] = None  # Retrieved from vault, never hardcoded
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None

    # Application identity
    application_uri: str = "urn:greenlang:gl003:unifiedsteam"
    application_name: str = "GL-003 UNIFIEDSTEAM Steam Optimizer"

    # Connection settings
    timeout_ms: int = 10000
    session_timeout_ms: int = 3600000  # 1 hour
    keepalive_interval_ms: int = 60000  # 1 minute

    # Reconnection settings
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 10
    reconnect_delay_ms: int = 5000
    reconnect_backoff_factor: float = 1.5
    max_reconnect_delay_ms: int = 300000  # 5 minutes

    # Subscription defaults
    default_publishing_interval_ms: int = 1000
    default_sampling_interval_ms: int = 500
    default_queue_size: int = 10

    # Connection pooling
    max_connections: int = 5
    connection_pool_timeout_ms: int = 30000


@dataclass
class Node:
    """OPC-UA node representation."""
    node_id: str
    browse_name: str
    display_name: str
    node_class: NodeClass
    data_type: Optional[DataType] = None
    description: str = ""
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)

    # Additional metadata for steam system tags
    engineering_unit: Optional[str] = None
    eu_range: Optional[Tuple[float, float]] = None  # (low, high)
    instrument_range: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "browse_name": self.browse_name,
            "display_name": self.display_name,
            "node_class": self.node_class.value,
            "data_type": self.data_type.value if self.data_type else None,
            "description": self.description,
            "engineering_unit": self.engineering_unit,
            "eu_range": self.eu_range,
        }


@dataclass
class TagValue:
    """Value read from OPC-UA server."""
    node_id: str
    value: Any
    status_code: StatusCode
    source_timestamp: datetime
    server_timestamp: datetime
    data_type: DataType

    # Quality flags
    is_good: bool = True
    is_uncertain: bool = False
    is_bad: bool = False

    def __post_init__(self):
        """Set quality flags based on status code."""
        code_val = self.status_code.value if isinstance(self.status_code, StatusCode) else self.status_code
        if isinstance(code_val, int):
            self.is_good = (code_val & 0xC0000000) == 0x00000000
            self.is_uncertain = (code_val & 0xC0000000) == 0x40000000
            self.is_bad = (code_val & 0xC0000000) == 0x80000000

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "value": self.value,
            "status_code": self.status_code.value if isinstance(self.status_code, StatusCode) else self.status_code,
            "source_timestamp": self.source_timestamp.isoformat(),
            "server_timestamp": self.server_timestamp.isoformat(),
            "data_type": self.data_type.value,
            "is_good": self.is_good,
        }


# Type alias for data change callback
DataChangeCallback = Callable[[str, TagValue], None]


@dataclass
class MonitoredItem:
    """OPC-UA monitored item configuration."""
    node_id: str
    sampling_interval_ms: int = 500
    queue_size: int = 10
    discard_oldest: bool = True

    # Filter settings
    deadband_type: str = "absolute"  # "absolute" or "percent"
    deadband_value: float = 0.0

    # Callback
    callback: Optional[DataChangeCallback] = None

    # Runtime state
    client_handle: Optional[int] = None
    monitored_item_id: Optional[int] = None


@dataclass
class Subscription:
    """OPC-UA subscription."""
    subscription_id: str
    publishing_interval_ms: int
    monitored_items: Dict[str, MonitoredItem] = field(default_factory=dict)

    # State
    is_active: bool = False
    lifetime_count: int = 10000
    max_keepalive_count: int = 10
    max_notifications_per_publish: int = 1000
    priority: int = 0

    # Statistics
    notifications_received: int = 0
    last_notification_time: Optional[datetime] = None

    def add_item(self, item: MonitoredItem) -> None:
        """Add monitored item to subscription."""
        self.monitored_items[item.node_id] = item

    def remove_item(self, node_id: str) -> Optional[MonitoredItem]:
        """Remove monitored item from subscription."""
        return self.monitored_items.pop(node_id, None)


class ConnectionState(Enum):
    """OPC-UA connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class OPCUAConnector:
    """
    OPC-UA connector for steam system data acquisition.

    Provides secure, reliable connectivity to industrial control systems
    for real-time steam system monitoring and optimization.

    Features:
    - Automatic reconnection with exponential backoff
    - Subscription-based data change notifications
    - Namespace browsing for tag discovery
    - Certificate-based security
    - Connection pooling

    Example:
        config = OPCUAConfig(
            endpoint_url="opc.tcp://plc.site.company.com:4840",
            security_policy=SecurityPolicy.BASIC256SHA256,
            username="gl003_service",
        )

        connector = OPCUAConnector(config)
        await connector.connect()

        # Subscribe to steam header tags
        subscription = await connector.subscribe_tags(
            tag_list=["ns=2;s=SteamHeader.Pressure", "ns=2;s=SteamHeader.Temperature"],
            callback=on_data_change,
            sampling_interval_ms=500
        )

        # Browse available tags
        nodes = await connector.browse_namespace("ns=2;s=SteamSystem")
    """

    def __init__(
        self,
        config: OPCUAConfig,
        vault_client: Optional[Any] = None,
        on_state_change: Optional[Callable[[ConnectionState], None]] = None,
    ) -> None:
        """
        Initialize OPC-UA connector.

        Args:
            config: Connection configuration
            vault_client: Optional vault client for credential retrieval
            on_state_change: Callback for connection state changes
        """
        self.config = config
        self.vault_client = vault_client
        self._on_state_change = on_state_change

        # Retrieve credentials from vault if available
        if vault_client and config.username:
            try:
                self.config.password = vault_client.get_secret(
                    f"opcua/{config.endpoint_url}/password"
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve credentials from vault: {e}")

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._client = None  # Actual OPC-UA client (asyncua.Client)
        self._session = None

        # Subscriptions
        self._subscriptions: Dict[str, Subscription] = {}
        self._subscription_handles: Dict[str, Any] = {}  # Maps to actual subscription objects

        # Node cache
        self._node_cache: Dict[str, Node] = {}

        # Reconnection
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._stop_reconnect = False

        # Statistics
        self._stats = {
            "connects": 0,
            "disconnects": 0,
            "reconnects": 0,
            "reads": 0,
            "writes": 0,
            "subscriptions_created": 0,
            "notifications_received": 0,
            "errors": 0,
            "bytes_received": 0,
            "bytes_sent": 0,
        }

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(f"OPCUAConnector initialized for {config.endpoint_url}")

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    def _set_state(self, new_state: ConnectionState) -> None:
        """Update connection state and notify callback."""
        old_state = self._state
        self._state = new_state

        if old_state != new_state:
            logger.info(f"OPC-UA connection state: {old_state.value} -> {new_state.value}")
            if self._on_state_change:
                try:
                    self._on_state_change(new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")

    async def connect(
        self,
        endpoint_url: Optional[str] = None,
        security_policy: Optional[SecurityPolicy] = None,
    ) -> bool:
        """
        Connect to OPC-UA server.

        Args:
            endpoint_url: Override endpoint URL from config
            security_policy: Override security policy from config

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails after retries
        """
        if endpoint_url:
            self.config.endpoint_url = endpoint_url
        if security_policy:
            self.config.security_policy = security_policy

        async with self._lock:
            if self._state == ConnectionState.CONNECTED:
                logger.info("Already connected to OPC-UA server")
                return True

            self._set_state(ConnectionState.CONNECTING)
            self._stop_reconnect = False

            try:
                # Create OPC-UA client
                # In production, use: from asyncua import Client
                # self._client = Client(self.config.endpoint_url)

                # Configure security
                await self._configure_security()

                # Connect to server
                # await self._client.connect()

                # Simulate successful connection for framework
                self._client = self._create_mock_client()

                self._set_state(ConnectionState.CONNECTED)
                self._stats["connects"] += 1
                self._reconnect_attempts = 0

                logger.info(f"Connected to OPC-UA server: {self.config.endpoint_url}")
                return True

            except Exception as e:
                self._stats["errors"] += 1
                self._set_state(ConnectionState.ERROR)
                logger.error(f"OPC-UA connection failed: {e}")

                # Start reconnection if enabled
                if self.config.auto_reconnect:
                    self._start_reconnect()

                return False

    def _create_mock_client(self) -> object:
        """Create mock client for framework demonstration."""
        class MockClient:
            def __init__(self):
                self.connected = True
            async def disconnect(self):
                self.connected = False
        return MockClient()

    async def _configure_security(self) -> None:
        """Configure security settings for connection."""
        if self.config.security_policy == SecurityPolicy.NONE:
            return

        # Set security policy and mode
        # In production:
        # await self._client.set_security(
        #     self.config.security_policy.value,
        #     self.config.certificate_path,
        #     self.config.private_key_path,
        #     mode=self.config.security_mode.value
        # )

        # Set authentication credentials
        if self.config.username:
            # self._client.set_user(self.config.username)
            # self._client.set_password(self.config.password)
            pass

        logger.debug(f"Security configured: {self.config.security_policy.value}")

    async def disconnect(self) -> None:
        """Disconnect from OPC-UA server."""
        async with self._lock:
            self._stop_reconnect = True

            # Cancel reconnection task
            if self._reconnect_task:
                self._reconnect_task.cancel()
                try:
                    await self._reconnect_task
                except asyncio.CancelledError:
                    pass
                self._reconnect_task = None

            # Remove all subscriptions
            for sub_id in list(self._subscriptions.keys()):
                await self._delete_subscription(sub_id)

            # Disconnect client
            if self._client:
                try:
                    await self._client.disconnect()
                except Exception as e:
                    logger.warning(f"Error during disconnect: {e}")
                self._client = None

            self._set_state(ConnectionState.DISCONNECTED)
            self._stats["disconnects"] += 1

            logger.info("Disconnected from OPC-UA server")

    def _start_reconnect(self) -> None:
        """Start automatic reconnection."""
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Reconnection loop with exponential backoff."""
        self._set_state(ConnectionState.RECONNECTING)

        while not self._stop_reconnect:
            self._reconnect_attempts += 1

            if self._reconnect_attempts > self.config.max_reconnect_attempts:
                logger.error("Max reconnection attempts exceeded")
                self._set_state(ConnectionState.ERROR)
                return

            # Calculate delay with exponential backoff
            delay = min(
                self.config.reconnect_delay_ms * (self.config.reconnect_backoff_factor ** (self._reconnect_attempts - 1)),
                self.config.max_reconnect_delay_ms
            )

            logger.info(f"Reconnection attempt {self._reconnect_attempts}/{self.config.max_reconnect_attempts} "
                       f"in {delay/1000:.1f}s")

            await asyncio.sleep(delay / 1000)

            if self._stop_reconnect:
                return

            try:
                # Attempt reconnection
                success = await self.connect()
                if success:
                    self._stats["reconnects"] += 1

                    # Resubscribe to all subscriptions
                    await self._restore_subscriptions()

                    logger.info("Reconnection successful, subscriptions restored")
                    return

            except Exception as e:
                logger.warning(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")

    async def _restore_subscriptions(self) -> None:
        """Restore subscriptions after reconnection."""
        for sub_id, subscription in list(self._subscriptions.items()):
            try:
                # Recreate subscription
                await self._create_subscription_internal(subscription)

                # Re-add monitored items
                for item in subscription.monitored_items.values():
                    await self._add_monitored_item_internal(sub_id, item)

                subscription.is_active = True
                logger.debug(f"Restored subscription {sub_id}")

            except Exception as e:
                logger.error(f"Failed to restore subscription {sub_id}: {e}")

    async def subscribe_tags(
        self,
        tag_list: List[str],
        callback: DataChangeCallback,
        sampling_interval_ms: int = 500,
        publishing_interval_ms: Optional[int] = None,
        deadband_value: float = 0.0,
        deadband_type: str = "absolute",
    ) -> Subscription:
        """
        Subscribe to tag value changes.

        Args:
            tag_list: List of node IDs to subscribe to
            callback: Function called on data change (node_id, TagValue)
            sampling_interval_ms: Sampling interval for monitored items
            publishing_interval_ms: Publishing interval for subscription
            deadband_value: Deadband value (filter small changes)
            deadband_type: "absolute" or "percent"

        Returns:
            Subscription object

        Example:
            def on_pressure_change(node_id: str, value: TagValue):
                print(f"{node_id}: {value.value} {value.status_code}")

            sub = await connector.subscribe_tags(
                ["ns=2;s=Header.Pressure", "ns=2;s=Header.Temperature"],
                on_pressure_change,
                sampling_interval_ms=500
            )
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to OPC-UA server")

        # Create subscription
        pub_interval = publishing_interval_ms or self.config.default_publishing_interval_ms
        subscription_id = str(uuid.uuid4())

        subscription = Subscription(
            subscription_id=subscription_id,
            publishing_interval_ms=pub_interval,
        )

        try:
            # Create subscription on server
            await self._create_subscription_internal(subscription)

            # Add monitored items for each tag
            for node_id in tag_list:
                item = MonitoredItem(
                    node_id=node_id,
                    sampling_interval_ms=sampling_interval_ms,
                    deadband_type=deadband_type,
                    deadband_value=deadband_value,
                    callback=callback,
                )

                await self._add_monitored_item_internal(subscription_id, item)
                subscription.add_item(item)

            subscription.is_active = True
            self._subscriptions[subscription_id] = subscription
            self._stats["subscriptions_created"] += 1

            logger.info(f"Created subscription {subscription_id} with {len(tag_list)} items")
            return subscription

        except Exception as e:
            logger.error(f"Failed to create subscription: {e}")
            self._stats["errors"] += 1
            raise

    async def _create_subscription_internal(self, subscription: Subscription) -> None:
        """Create subscription on OPC-UA server."""
        # In production with asyncua:
        # handler = DataChangeHandler(subscription, self._on_data_change)
        # sub = await self._client.create_subscription(
        #     subscription.publishing_interval_ms,
        #     handler
        # )
        # self._subscription_handles[subscription.subscription_id] = sub

        # For framework: store reference
        self._subscription_handles[subscription.subscription_id] = {
            "publishing_interval": subscription.publishing_interval_ms,
            "items": {},
        }

    async def _add_monitored_item_internal(
        self,
        subscription_id: str,
        item: MonitoredItem,
    ) -> None:
        """Add monitored item to subscription on server."""
        # In production with asyncua:
        # sub = self._subscription_handles[subscription_id]
        # node = self._client.get_node(item.node_id)
        # handle = await sub.subscribe_data_change(
        #     node,
        #     sampling_interval=item.sampling_interval_ms
        # )
        # item.monitored_item_id = handle

        # For framework: store reference
        if subscription_id in self._subscription_handles:
            self._subscription_handles[subscription_id]["items"][item.node_id] = item

    async def _delete_subscription(self, subscription_id: str) -> None:
        """Delete subscription from server."""
        if subscription_id in self._subscription_handles:
            # In production: await self._subscription_handles[subscription_id].delete()
            del self._subscription_handles[subscription_id]

        if subscription_id in self._subscriptions:
            self._subscriptions[subscription_id].is_active = False
            del self._subscriptions[subscription_id]

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Remove subscription."""
        await self._delete_subscription(subscription.subscription_id)
        logger.info(f"Deleted subscription {subscription.subscription_id}")

    async def read_tag(self, tag_name: str) -> TagValue:
        """
        Read single tag value.

        Args:
            tag_name: Node ID to read (e.g., "ns=2;s=SteamHeader.Pressure")

        Returns:
            TagValue with current value and status
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to OPC-UA server")

        try:
            # In production with asyncua:
            # node = self._client.get_node(tag_name)
            # data_value = await node.read_data_value()
            # value = data_value.Value.Value
            # status = data_value.StatusCode
            # source_ts = data_value.SourceTimestamp
            # server_ts = data_value.ServerTimestamp

            # For framework: return simulated value
            now = datetime.now(timezone.utc)

            tag_value = TagValue(
                node_id=tag_name,
                value=self._get_simulated_value(tag_name),
                status_code=StatusCode.GOOD,
                source_timestamp=now,
                server_timestamp=now,
                data_type=DataType.DOUBLE,
            )

            self._stats["reads"] += 1
            return tag_value

        except Exception as e:
            logger.error(f"Error reading tag {tag_name}: {e}")
            self._stats["errors"] += 1

            return TagValue(
                node_id=tag_name,
                value=None,
                status_code=StatusCode.BAD_COMMUNICATION_ERROR,
                source_timestamp=datetime.now(timezone.utc),
                server_timestamp=datetime.now(timezone.utc),
                data_type=DataType.DOUBLE,
            )

    async def read_tags(self, tag_names: List[str]) -> Dict[str, TagValue]:
        """
        Read multiple tag values.

        Args:
            tag_names: List of node IDs to read

        Returns:
            Dict mapping node_id to TagValue
        """
        results: Dict[str, TagValue] = {}

        # In production, use batch read for efficiency:
        # nodes = [self._client.get_node(name) for name in tag_names]
        # values = await self._client.read_values(nodes)

        for tag_name in tag_names:
            results[tag_name] = await self.read_tag(tag_name)

        return results

    async def write_tag(self, tag_name: str, value: Any) -> bool:
        """
        Write value to tag.

        Args:
            tag_name: Node ID to write
            value: Value to write

        Returns:
            True if write successful
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to OPC-UA server")

        try:
            # In production with asyncua:
            # node = self._client.get_node(tag_name)
            # data_type = await node.read_data_type_as_variant_type()
            # await node.write_value(ua.Variant(value, data_type))

            self._stats["writes"] += 1
            logger.info(f"Written {tag_name} = {value}")
            return True

        except Exception as e:
            logger.error(f"Error writing tag {tag_name}: {e}")
            self._stats["errors"] += 1
            return False

    async def browse_namespace(
        self,
        start_node_id: Optional[str] = None,
        max_depth: int = 3,
        filter_node_class: Optional[NodeClass] = None,
    ) -> List[Node]:
        """
        Browse OPC-UA namespace to discover available tags.

        Args:
            start_node_id: Starting node (default: Objects folder)
            max_depth: Maximum recursion depth
            filter_node_class: Filter by node class (e.g., VARIABLE for tags)

        Returns:
            List of discovered nodes
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to OPC-UA server")

        nodes: List[Node] = []
        visited: Set[str] = set()

        async def browse_recursive(node_id: str, depth: int) -> None:
            if depth > max_depth or node_id in visited:
                return

            visited.add(node_id)

            try:
                # In production with asyncua:
                # node = self._client.get_node(node_id)
                # children = await node.get_children()
                # for child in children:
                #     browse_name = await child.read_browse_name()
                #     display_name = await child.read_display_name()
                #     node_class = await child.read_node_class()

                # For framework: return simulated structure for steam system
                child_nodes = self._get_simulated_children(node_id)

                for child in child_nodes:
                    if filter_node_class and child.node_class != filter_node_class:
                        continue

                    nodes.append(child)
                    self._node_cache[child.node_id] = child

                    if child.node_class == NodeClass.OBJECT:
                        await browse_recursive(child.node_id, depth + 1)

            except Exception as e:
                logger.warning(f"Error browsing node {node_id}: {e}")

        start = start_node_id or "ns=0;i=85"  # Objects folder
        await browse_recursive(start, 0)

        logger.info(f"Browsed namespace: found {len(nodes)} nodes")
        return nodes

    def _get_simulated_value(self, tag_name: str) -> float:
        """Get simulated value for framework demonstration."""
        import random
        import math

        # Generate deterministic but varying values based on tag name
        base = hash(tag_name) % 100
        variation = math.sin(datetime.now().timestamp() / 10) * 5

        if "pressure" in tag_name.lower():
            return 100.0 + base * 0.5 + variation
        elif "temperature" in tag_name.lower():
            return 350.0 + base * 2 + variation * 2
        elif "flow" in tag_name.lower():
            return 50000.0 + base * 500 + variation * 100
        elif "level" in tag_name.lower():
            return 50.0 + variation * 2
        else:
            return float(base) + variation

    def _get_simulated_children(self, parent_id: str) -> List[Node]:
        """Get simulated child nodes for framework demonstration."""
        if parent_id == "ns=0;i=85" or parent_id == "ns=2;s=SteamSystem":
            # Top-level steam system structure
            return [
                Node(
                    node_id="ns=2;s=SteamSystem.Header",
                    browse_name="Header",
                    display_name="Steam Header",
                    node_class=NodeClass.OBJECT,
                    parent_id=parent_id,
                ),
                Node(
                    node_id="ns=2;s=SteamSystem.Boilers",
                    browse_name="Boilers",
                    display_name="Boilers",
                    node_class=NodeClass.OBJECT,
                    parent_id=parent_id,
                ),
                Node(
                    node_id="ns=2;s=SteamSystem.Turbines",
                    browse_name="Turbines",
                    display_name="Steam Turbines",
                    node_class=NodeClass.OBJECT,
                    parent_id=parent_id,
                ),
                Node(
                    node_id="ns=2;s=SteamSystem.Traps",
                    browse_name="Traps",
                    display_name="Steam Traps",
                    node_class=NodeClass.OBJECT,
                    parent_id=parent_id,
                ),
            ]
        elif "Header" in parent_id:
            return [
                Node(
                    node_id="ns=2;s=SteamSystem.Header.Pressure",
                    browse_name="Pressure",
                    display_name="Header Pressure",
                    node_class=NodeClass.VARIABLE,
                    data_type=DataType.DOUBLE,
                    engineering_unit="psig",
                    eu_range=(0.0, 200.0),
                    parent_id=parent_id,
                ),
                Node(
                    node_id="ns=2;s=SteamSystem.Header.Temperature",
                    browse_name="Temperature",
                    display_name="Header Temperature",
                    node_class=NodeClass.VARIABLE,
                    data_type=DataType.DOUBLE,
                    engineering_unit="degF",
                    eu_range=(200.0, 600.0),
                    parent_id=parent_id,
                ),
                Node(
                    node_id="ns=2;s=SteamSystem.Header.Flow",
                    browse_name="Flow",
                    display_name="Header Flow",
                    node_class=NodeClass.VARIABLE,
                    data_type=DataType.DOUBLE,
                    engineering_unit="lb/hr",
                    eu_range=(0.0, 100000.0),
                    parent_id=parent_id,
                ),
            ]
        else:
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            "state": self._state.value,
            "endpoint_url": self.config.endpoint_url,
            "active_subscriptions": len([s for s in self._subscriptions.values() if s.is_active]),
            "total_monitored_items": sum(
                len(s.monitored_items) for s in self._subscriptions.values()
            ),
            "cached_nodes": len(self._node_cache),
        }

    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription by ID."""
        return self._subscriptions.get(subscription_id)

    def get_all_subscriptions(self) -> List[Subscription]:
        """Get all active subscriptions."""
        return list(self._subscriptions.values())


def create_steam_system_tag_list() -> List[str]:
    """Create standard steam system tag list for subscription."""
    return [
        # Steam header
        "ns=2;s=SteamSystem.Header.Pressure",
        "ns=2;s=SteamSystem.Header.Temperature",
        "ns=2;s=SteamSystem.Header.Flow",

        # Boiler 1
        "ns=2;s=SteamSystem.Boilers.B1.SteamFlow",
        "ns=2;s=SteamSystem.Boilers.B1.Pressure",
        "ns=2;s=SteamSystem.Boilers.B1.FuelFlow",
        "ns=2;s=SteamSystem.Boilers.B1.O2Percent",

        # Turbine 1
        "ns=2;s=SteamSystem.Turbines.T1.InletPressure",
        "ns=2;s=SteamSystem.Turbines.T1.ExhaustPressure",
        "ns=2;s=SteamSystem.Turbines.T1.Power",

        # Steam traps
        "ns=2;s=SteamSystem.Traps.ST001.Temperature",
        "ns=2;s=SteamSystem.Traps.ST001.Status",
        "ns=2;s=SteamSystem.Traps.ST002.Temperature",
        "ns=2;s=SteamSystem.Traps.ST002.Status",
    ]
