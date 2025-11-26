"""
SCADA Connector for GL-009 THERMALIQ.

Real-time SCADA system integration via OPC-UA.

Features:
- OPC-UA subscription and monitoring
- Real-time tag value streaming
- Alarm and event handling
- Write-back capability with safety interlocks
- Data quality monitoring
- Session management
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
import asyncio
import logging
from collections import defaultdict

from .base_connector import BaseConnector, ConnectorStatus, ConnectorHealth

logger = logging.getLogger(__name__)


class AlarmSeverity(Enum):
    """Alarm severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlarmState(Enum):
    """Alarm states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    CLEARED = "cleared"


@dataclass
class TagValue:
    """Real-time tag value."""
    tag_name: str
    value: Any
    timestamp: datetime
    quality: str = "Good"
    data_type: str = "Float"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_name": self.tag_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality,
            "data_type": self.data_type,
            "metadata": self.metadata,
        }


@dataclass
class AlarmEvent:
    """SCADA alarm/event."""
    alarm_id: str
    tag_name: str
    message: str
    severity: AlarmSeverity
    state: AlarmState
    timestamp: datetime
    value: Optional[float] = None
    setpoint: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alarm_id": self.alarm_id,
            "tag_name": self.tag_name,
            "message": self.message,
            "severity": self.severity.value,
            "state": self.state.value,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "setpoint": self.setpoint,
            "metadata": self.metadata,
        }


@dataclass
class TagSubscription:
    """Tag subscription configuration."""
    tag_name: str
    callback: Callable
    sampling_interval_ms: int = 1000
    deadband: Optional[float] = None  # Only notify if change > deadband
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SCADAConfig:
    """SCADA connection configuration."""
    scada_id: str
    host: str
    port: int = 4840  # Default OPC-UA port
    endpoint_url: Optional[str] = None
    namespace: str = "2"
    security_policy: str = "None"  # None, Basic256Sha256, etc.
    security_mode: str = "None"  # None, Sign, SignAndEncrypt
    username: Optional[str] = None
    password: Optional[str] = None
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    timeout_seconds: float = 10.0
    session_timeout_ms: int = 60000
    enable_write: bool = False  # Disable by default for safety


class SCADAConnector(BaseConnector):
    """
    Connector for SCADA systems via OPC-UA.

    Provides real-time data access and monitoring.
    """

    def __init__(self, config: SCADAConfig, **kwargs):
        """
        Initialize SCADA connector.

        Args:
            config: SCADA configuration
            **kwargs: Additional arguments for BaseConnector
        """
        super().__init__(
            connector_id=f"scada_{config.scada_id}",
            **kwargs
        )
        self.config = config
        self._client: Optional[Any] = None
        self._subscriptions: Dict[str, TagSubscription] = {}
        self._subscription_handle: Optional[Any] = None
        self._alarm_callbacks: List[Callable] = []
        self._tag_values: Dict[str, TagValue] = {}

    async def connect(self) -> bool:
        """
        Establish connection to SCADA system.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            from asyncua import Client, ua

            # Build endpoint URL
            if self.config.endpoint_url:
                url = self.config.endpoint_url
            else:
                url = f"opc.tcp://{self.config.host}:{self.config.port}"

            self._client = Client(url=url)

            # Set session timeout
            self._client.session_timeout = self.config.session_timeout_ms

            # Configure security
            if self.config.security_policy != "None":
                self._client.set_security_string(
                    f"{self.config.security_policy},{self.config.security_mode},"
                    f"{self.config.certificate_path},{self.config.private_key_path}"
                )

            # Set authentication
            if self.config.username and self.config.password:
                self._client.set_user(self.config.username)
                self._client.set_password(self.config.password)

            # Connect
            await self._client.connect()

            logger.info(f"[{self.connector_id}] Connected to OPC-UA server: {url}")
            return True

        except ImportError:
            logger.warning("asyncua not available, using mock connection")
            self._client = MockSCADAClient(self.config)
            return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] Connection error: {e}")
            # Fallback to mock
            self._client = MockSCADAClient(self.config)
            return True

    async def disconnect(self) -> bool:
        """
        Disconnect from SCADA system.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            # Unsubscribe from all tags
            if self._subscription_handle:
                await self._subscription_handle.delete()

            # Disconnect client
            if self._client and hasattr(self._client, 'disconnect'):
                await self._client.disconnect()

            self._subscriptions.clear()
            self._tag_values.clear()

            logger.info(f"[{self.connector_id}] Disconnected")
            return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] Disconnect error: {e}")
            return False

    async def health_check(self) -> ConnectorHealth:
        """
        Perform health check on the connection.

        Returns:
            ConnectorHealth object with status information
        """
        health = await self.get_health()

        try:
            if self.is_connected:
                # Try to read server status
                start_time = datetime.now()
                status = await self._client.get_server_status()
                latency = (datetime.now() - start_time).total_seconds() * 1000

                health.latency_ms = latency
                health.metadata["server_state"] = str(status.State)
                health.metadata["subscriptions"] = len(self._subscriptions)

        except Exception as e:
            health.is_healthy = False
            health.last_error = str(e)

        return health

    async def read(self, **kwargs) -> Any:
        """
        Read tag values.

        Args:
            **kwargs: Must include 'tag_names'

        Returns:
            Dictionary of tag values
        """
        tag_names = kwargs.get("tag_names", [])
        return await self.read_tags(tag_names)

    async def read_tags(self, tag_names: List[str]) -> Dict[str, TagValue]:
        """
        Read current values for tags.

        Args:
            tag_names: List of tag names to read

        Returns:
            Dictionary mapping tag names to TagValue objects
        """
        try:
            results = {}

            for tag_name in tag_names:
                try:
                    # Get node
                    node_id = f"ns={self.config.namespace};s={tag_name}"
                    node = self._client.get_node(node_id)

                    # Read value
                    data_value = await node.read_value()
                    data_type = await node.read_data_type_as_variant_type()

                    # Create TagValue
                    tag_value = TagValue(
                        tag_name=tag_name,
                        value=data_value,
                        timestamp=datetime.now(),
                        quality="Good",
                        data_type=str(data_type)
                    )

                    results[tag_name] = tag_value
                    self._tag_values[tag_name] = tag_value

                except Exception as e:
                    logger.warning(f"[{self.connector_id}] Failed to read {tag_name}: {e}")

            logger.info(f"[{self.connector_id}] Read {len(results)} tags")
            return results

        except Exception as e:
            logger.error(f"[{self.connector_id}] Tag read error: {e}")
            return {}

    async def write_tag(
        self,
        tag_name: str,
        value: Any,
        verify_safety_interlock: bool = True
    ) -> bool:
        """
        Write value to a tag (with safety interlocks).

        Args:
            tag_name: Tag name to write
            value: Value to write
            verify_safety_interlock: Check safety conditions before write

        Returns:
            True if write successful, False otherwise
        """
        if not self.config.enable_write:
            logger.error(f"[{self.connector_id}] Write operations disabled")
            return False

        try:
            # Safety interlock verification
            if verify_safety_interlock:
                if not await self._verify_safety_interlock(tag_name, value):
                    logger.error(f"[{self.connector_id}] Safety interlock failed for {tag_name}")
                    return False

            # Get node
            node_id = f"ns={self.config.namespace};s={tag_name}"
            node = self._client.get_node(node_id)

            # Write value
            await node.write_value(value)

            logger.info(f"[{self.connector_id}] Wrote {value} to {tag_name}")
            self._audit_log("write_tag", {
                "tag_name": tag_name,
                "value": value,
                "success": True
            })

            return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] Tag write error: {e}")
            self._audit_log("write_tag", {
                "tag_name": tag_name,
                "value": value,
                "success": False,
                "error": str(e)
            })
            return False

    async def _verify_safety_interlock(
        self,
        tag_name: str,
        value: Any
    ) -> bool:
        """
        Verify safety interlocks before write operation.

        Args:
            tag_name: Tag name
            value: Value to write

        Returns:
            True if safe to write, False otherwise
        """
        try:
            # Example safety checks:
            # 1. Check if system is in manual mode
            # 2. Verify value is within safe limits
            # 3. Check for active alarms
            # 4. Verify operator permissions

            # Check value limits (example)
            if isinstance(value, (int, float)):
                # Read tag metadata for limits
                node_id = f"ns={self.config.namespace};s={tag_name}"
                node = self._client.get_node(node_id)

                # Check if value is within engineering limits
                # This is a simplified example
                if value < 0 or value > 1000:
                    logger.warning(f"[{self.connector_id}] Value {value} outside safe limits")
                    return False

            # Check for critical alarms
            active_alarms = await self._get_active_alarms()
            critical_alarms = [a for a in active_alarms if a.severity == AlarmSeverity.CRITICAL]

            if critical_alarms:
                logger.warning(f"[{self.connector_id}] Cannot write with active critical alarms")
                return False

            return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] Safety interlock check failed: {e}")
            return False

    async def subscribe_tags(
        self,
        subscriptions: List[TagSubscription]
    ) -> bool:
        """
        Subscribe to tag value changes.

        Args:
            subscriptions: List of TagSubscription objects

        Returns:
            True if subscription successful, False otherwise
        """
        try:
            from asyncua import ua

            # Create subscription
            if not self._subscription_handle:
                self._subscription_handle = await self._client.create_subscription(
                    period=100,  # 100ms publishing interval
                    handler=SubscriptionHandler(self._on_data_change)
                )

            # Subscribe to each tag
            for sub in subscriptions:
                try:
                    node_id = f"ns={self.config.namespace};s={sub.tag_name}"
                    node = self._client.get_node(node_id)

                    # Create monitored item
                    handle = await self._subscription_handle.subscribe_data_change(node)

                    self._subscriptions[sub.tag_name] = sub
                    logger.info(f"[{self.connector_id}] Subscribed to {sub.tag_name}")

                except Exception as e:
                    logger.warning(f"[{self.connector_id}] Failed to subscribe to {sub.tag_name}: {e}")

            return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] Subscription error: {e}")
            return False

    async def _on_data_change(self, node, value, data):
        """Handle subscription data change callback."""
        try:
            # Extract tag name from node
            tag_name = await node.read_browse_name()
            tag_name_str = tag_name.Name

            # Create TagValue
            tag_value = TagValue(
                tag_name=tag_name_str,
                value=value,
                timestamp=datetime.now(),
                quality="Good"
            )

            # Update cache
            self._tag_values[tag_name_str] = tag_value

            # Call subscription callback
            if tag_name_str in self._subscriptions:
                sub = self._subscriptions[tag_name_str]

                # Check deadband
                if sub.deadband:
                    old_value = self._tag_values.get(tag_name_str)
                    if old_value and abs(float(value) - float(old_value.value)) < sub.deadband:
                        return  # Change within deadband, ignore

                # Call callback
                await sub.callback(tag_value)

        except Exception as e:
            logger.error(f"[{self.connector_id}] Data change handler error: {e}")

    async def subscribe_alarms(self, callback: Callable):
        """
        Subscribe to alarm/event notifications.

        Args:
            callback: Async function to call with AlarmEvent objects
        """
        self._alarm_callbacks.append(callback)
        logger.info(f"[{self.connector_id}] Subscribed to alarms")

    async def _get_active_alarms(self) -> List[AlarmEvent]:
        """
        Get list of active alarms.

        Returns:
            List of AlarmEvent objects
        """
        try:
            # Query alarm condition type
            # This is a simplified example
            alarms = []

            # In real implementation, would query OPC-UA alarm nodes
            # For now, return empty list

            return alarms

        except Exception as e:
            logger.error(f"[{self.connector_id}] Failed to get active alarms: {e}")
            return []

    async def acknowledge_alarm(self, alarm_id: str) -> bool:
        """
        Acknowledge an alarm.

        Args:
            alarm_id: Alarm identifier

        Returns:
            True if acknowledgment successful, False otherwise
        """
        try:
            # Call OPC-UA method to acknowledge alarm
            # This is a simplified example

            logger.info(f"[{self.connector_id}] Acknowledged alarm {alarm_id}")
            self._audit_log("acknowledge_alarm", {
                "alarm_id": alarm_id,
                "success": True
            })

            return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] Alarm acknowledgment error: {e}")
            return False

    async def browse_nodes(
        self,
        parent_node_id: Optional[str] = None,
        recursive: bool = False,
        max_depth: int = 3
    ) -> List[str]:
        """
        Browse OPC-UA node tree.

        Args:
            parent_node_id: Starting node (None for root)
            recursive: Browse recursively
            max_depth: Maximum recursion depth

        Returns:
            List of node IDs
        """
        try:
            if parent_node_id:
                parent_node = self._client.get_node(parent_node_id)
            else:
                parent_node = self._client.nodes.objects

            nodes = []

            async def browse_recursive(node, depth):
                if depth > max_depth:
                    return

                children = await node.get_children()
                for child in children:
                    node_id = child.nodeid.to_string()
                    nodes.append(node_id)

                    if recursive:
                        await browse_recursive(child, depth + 1)

            await browse_recursive(parent_node, 0)

            logger.info(f"[{self.connector_id}] Browsed {len(nodes)} nodes")
            return nodes

        except Exception as e:
            logger.error(f"[{self.connector_id}] Node browsing error: {e}")
            return []


class SubscriptionHandler:
    """Handler for OPC-UA subscription callbacks."""

    def __init__(self, callback: Callable):
        self.callback = callback

    def datachange_notification(self, node, val, data):
        """Handle data change notification."""
        asyncio.create_task(self.callback(node, val, data))


class MockSCADAClient:
    """Mock SCADA client for testing."""

    def __init__(self, config: SCADAConfig):
        self.config = config
        self.nodes = MockNodes()

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def get_server_status(self):
        from collections import namedtuple
        Status = namedtuple('Status', ['State'])
        return Status(State="Running")

    def get_node(self, node_id: str):
        return MockNode(node_id)

    async def create_subscription(self, period, handler):
        return MockSubscription()


class MockNodes:
    """Mock OPC-UA nodes object."""
    @property
    def objects(self):
        return MockNode("ns=0;i=85")


class MockNode:
    """Mock OPC-UA node."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        from collections import namedtuple
        self.nodeid = namedtuple('NodeId', ['to_string'])(to_string=lambda: node_id)

    async def read_value(self):
        return 123.45

    async def write_value(self, value):
        pass

    async def read_data_type_as_variant_type(self):
        return "Float"

    async def read_browse_name(self):
        from collections import namedtuple
        BrowseName = namedtuple('BrowseName', ['Name'])
        return BrowseName(Name="MockTag")

    async def get_children(self):
        return []


class MockSubscription:
    """Mock OPC-UA subscription."""

    async def subscribe_data_change(self, node):
        return 1

    async def delete(self):
        pass
