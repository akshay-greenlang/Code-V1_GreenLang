# -*- coding: utf-8 -*-
"""
Mock OPC-UA Server for Integration Testing
===========================================

Provides a fully-featured mock OPC-UA server that simulates:
- Node browsing and discovery
- Value reading (single and batch)
- Value writing with timestamps
- Subscription and data change callbacks
- Historical data queries
- Security mode handling
- Connection/reconnection behavior

Usage:
    >>> server = MockOPCUAServer()
    >>> await server.start()
    >>> server.add_node(MockOPCUANode("ns=2;s=Temp", value=85.5))
    >>> value = await server.read_value("ns=2;s=Temp")

Author: GreenLang Test Engineering Team
Date: December 2025
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class SecurityMode(str, Enum):
    """OPC-UA security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class NodeClass(str, Enum):
    """OPC-UA node classes."""
    VARIABLE = "Variable"
    OBJECT = "Object"
    METHOD = "Method"
    VIEW = "View"


@dataclass
class HistoricalDataPoint:
    """Historical data point."""
    timestamp: datetime
    value: Any
    status_code: int = 0


@dataclass
class MockOPCUANode:
    """
    Mock OPC-UA node for testing.

    Represents a single OPC-UA node with all metadata and history tracking.
    """
    node_id: str
    browse_name: str = ""
    display_name: str = ""
    value: Any = None
    data_type: str = "Double"
    node_class: NodeClass = NodeClass.VARIABLE
    access_level: int = 3  # Read/Write
    historizing: bool = True
    history: List[HistoricalDataPoint] = field(default_factory=list)
    subscriptions: Dict[str, Callable] = field(default_factory=dict)
    source_timestamp: Optional[datetime] = None
    server_timestamp: Optional[datetime] = None
    status_code: int = 0

    def __post_init__(self):
        if not self.browse_name:
            self.browse_name = self.node_id.split(";")[-1].replace("s=", "")
        if not self.display_name:
            self.display_name = self.browse_name
        if self.source_timestamp is None:
            self.source_timestamp = datetime.utcnow()
        if self.server_timestamp is None:
            self.server_timestamp = datetime.utcnow()


@dataclass
class Subscription:
    """OPC-UA subscription tracking."""
    subscription_id: str
    client_id: str
    node_ids: List[str]
    publishing_interval_ms: int
    callbacks: Dict[str, Callable]
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_published: Optional[datetime] = None
    active: bool = True


class MockOPCUAServer:
    """
    Mock OPC-UA server for integration testing.

    Provides a complete simulation of OPC-UA server behavior for testing
    client implementations without requiring actual OPC-UA infrastructure.

    Features:
    - Full node hierarchy with browsing
    - Value reading and writing
    - Subscription management with callbacks
    - Historical data simulation
    - Security mode simulation
    - Connection state management
    - Error injection for testing error handling

    Attributes:
        endpoint: Server endpoint URL
        nodes: Dictionary of registered nodes
        subscriptions: Active subscriptions
        security_mode: Current security mode
        running: Server running state
    """

    def __init__(
        self,
        endpoint: str = "opc.tcp://localhost:4840/greenlang/",
        security_mode: SecurityMode = SecurityMode.NONE
    ):
        """
        Initialize mock OPC-UA server.

        Args:
            endpoint: Server endpoint URL
            security_mode: Security mode to simulate
        """
        self.endpoint = endpoint
        self.security_mode = security_mode
        self.nodes: Dict[str, MockOPCUANode] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.running: bool = False
        self._namespace_index: int = 2
        self._error_mode: bool = False
        self._latency_ms: int = 0
        self._subscription_tasks: Dict[str, asyncio.Task] = {}

        # Initialize default namespace structure
        self._setup_default_structure()

        logger.info(f"MockOPCUAServer initialized at {endpoint}")

    def _setup_default_structure(self) -> None:
        """Setup default OPC-UA namespace structure."""
        # Server status node
        self.add_node(MockOPCUANode(
            node_id="ns=0;i=2256",
            browse_name="ServerStatus",
            display_name="Server Status",
            value={"state": "Running", "startTime": datetime.utcnow().isoformat()},
            node_class=NodeClass.VARIABLE
        ))

        # Process heat nodes
        process_nodes = [
            MockOPCUANode(
                node_id="ns=2;s=ProcessHeat/Temperature",
                browse_name="Temperature",
                display_name="Process Temperature",
                value=85.5,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=ProcessHeat/Pressure",
                browse_name="Pressure",
                display_name="Process Pressure",
                value=2.5,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=ProcessHeat/FlowRate",
                browse_name="FlowRate",
                display_name="Flow Rate",
                value=150.0,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=ProcessHeat/FuelConsumption",
                browse_name="FuelConsumption",
                display_name="Fuel Consumption",
                value=45.2,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=ProcessHeat/EmissionsFactor",
                browse_name="EmissionsFactor",
                display_name="Emissions Factor",
                value=2.68,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=ProcessHeat/TotalEmissions",
                browse_name="TotalEmissions",
                display_name="Total CO2 Emissions",
                value=121.14,
                data_type="Double"
            ),
            MockOPCUANode(
                node_id="ns=2;s=ProcessHeat/Status",
                browse_name="Status",
                display_name="System Status",
                value="RUNNING",
                data_type="String"
            ),
            MockOPCUANode(
                node_id="ns=2;s=ProcessHeat/Efficiency",
                browse_name="Efficiency",
                display_name="Thermal Efficiency",
                value=0.92,
                data_type="Double"
            ),
        ]

        for node in process_nodes:
            self.add_node(node)

    async def start(self) -> None:
        """Start the mock OPC-UA server."""
        if self.running:
            logger.warning("Server already running")
            return

        self.running = True
        logger.info(f"MockOPCUAServer started at {self.endpoint}")

    async def stop(self) -> None:
        """Stop the mock OPC-UA server."""
        if not self.running:
            return

        # Cancel all subscription tasks
        for task in self._subscription_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._subscription_tasks.clear()
        self.subscriptions.clear()
        self.connected_clients.clear()
        self.running = False

        logger.info("MockOPCUAServer stopped")

    def add_node(self, node: MockOPCUANode) -> None:
        """Add a node to the server."""
        self.nodes[node.node_id] = node
        logger.debug(f"Added node: {node.node_id}")

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the server."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.debug(f"Removed node: {node_id}")

    async def connect_client(
        self,
        client_id: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        certificate: Optional[bytes] = None
    ) -> bool:
        """
        Connect a client to the server.

        Args:
            client_id: Unique client identifier
            username: Optional username for authentication
            password: Optional password for authentication
            certificate: Optional client certificate

        Returns:
            True if connection successful
        """
        if not self.running:
            raise ConnectionError("Server not running")

        if self._error_mode:
            raise ConnectionError("Simulated connection error")

        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000)

        # Simulate security validation
        if self.security_mode != SecurityMode.NONE:
            if not certificate and not (username and password):
                raise PermissionError("Authentication required")

        self.connected_clients[client_id] = {
            "username": username,
            "connected_at": datetime.utcnow(),
            "security_mode": self.security_mode.value
        }

        logger.info(f"Client {client_id} connected")
        return True

    async def disconnect_client(self, client_id: str) -> None:
        """Disconnect a client from the server."""
        if client_id in self.connected_clients:
            # Remove client subscriptions
            subs_to_remove = [
                sub_id for sub_id, sub in self.subscriptions.items()
                if sub.client_id == client_id
            ]
            for sub_id in subs_to_remove:
                await self.delete_subscription(sub_id)

            del self.connected_clients[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def read_value(self, node_id: str) -> Any:
        """
        Read value from a node.

        Args:
            node_id: OPC-UA node identifier

        Returns:
            Current node value

        Raises:
            ValueError: If node not found
        """
        self._check_running()

        if self._error_mode:
            raise Exception("Simulated read error")

        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000)

        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        return self.nodes[node_id].value

    async def read_values(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Read multiple node values in a batch.

        Args:
            node_ids: List of node identifiers

        Returns:
            Dictionary mapping node_id to value
        """
        self._check_running()

        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000)

        results = {}
        for node_id in node_ids:
            if node_id in self.nodes:
                results[node_id] = self.nodes[node_id].value
            else:
                results[node_id] = None

        return results

    async def write_value(
        self,
        node_id: str,
        value: Any,
        source_timestamp: Optional[datetime] = None
    ) -> str:
        """
        Write value to a node.

        Args:
            node_id: OPC-UA node identifier
            value: Value to write
            source_timestamp: Optional source timestamp

        Returns:
            Provenance hash of the write operation
        """
        self._check_running()

        if self._error_mode:
            raise Exception("Simulated write error")

        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        node = self.nodes[node_id]

        # Check write access
        if not (node.access_level & 0x02):
            raise PermissionError(f"Node {node_id} is read-only")

        old_value = node.value
        node.value = value
        node.source_timestamp = source_timestamp or datetime.utcnow()
        node.server_timestamp = datetime.utcnow()

        # Add to history if historizing
        if node.historizing:
            node.history.append(HistoricalDataPoint(
                timestamp=node.server_timestamp,
                value=value
            ))

        # Notify subscribers
        await self._notify_subscribers(node_id, value)

        # Calculate provenance hash
        provenance_str = f"{node_id}:{value}:{node.server_timestamp.isoformat()}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        logger.debug(f"Wrote {value} to {node_id}")
        return provenance_hash

    async def browse(
        self,
        node_id: Optional[str] = None,
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Browse child nodes.

        Args:
            node_id: Parent node ID (None for all nodes)
            recursive: Whether to browse recursively

        Returns:
            List of node information dictionaries
        """
        self._check_running()

        results = []

        if node_id is None:
            # Return all nodes
            for nid, node in self.nodes.items():
                results.append({
                    "node_id": nid,
                    "browse_name": node.browse_name,
                    "display_name": node.display_name,
                    "node_class": node.node_class.value,
                    "data_type": node.data_type
                })
        else:
            # Filter by prefix (simulating hierarchy)
            prefix = node_id.replace("s=", "").replace("ns=2;", "")
            for nid, node in self.nodes.items():
                if prefix in nid and nid != node_id:
                    results.append({
                        "node_id": nid,
                        "browse_name": node.browse_name,
                        "display_name": node.display_name,
                        "node_class": node.node_class.value,
                        "data_type": node.data_type
                    })

        return results

    async def create_subscription(
        self,
        client_id: str,
        node_ids: List[str],
        callback: Callable,
        publishing_interval_ms: int = 1000
    ) -> str:
        """
        Create a subscription for data changes.

        Args:
            client_id: Client identifier
            node_ids: Nodes to subscribe to
            callback: Function to call on data change
            publishing_interval_ms: Publishing interval in milliseconds

        Returns:
            Subscription ID
        """
        self._check_running()

        sub_id = str(uuid4())

        subscription = Subscription(
            subscription_id=sub_id,
            client_id=client_id,
            node_ids=node_ids,
            publishing_interval_ms=publishing_interval_ms,
            callbacks={nid: callback for nid in node_ids}
        )

        self.subscriptions[sub_id] = subscription

        # Register callbacks on nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].subscriptions[sub_id] = callback

        logger.info(f"Created subscription {sub_id} for {node_ids}")
        return sub_id

    async def delete_subscription(self, subscription_id: str) -> None:
        """Delete a subscription."""
        if subscription_id not in self.subscriptions:
            return

        subscription = self.subscriptions[subscription_id]

        # Remove callbacks from nodes
        for node_id in subscription.node_ids:
            if node_id in self.nodes:
                self.nodes[node_id].subscriptions.pop(subscription_id, None)

        # Cancel task if exists
        if subscription_id in self._subscription_tasks:
            self._subscription_tasks[subscription_id].cancel()
            del self._subscription_tasks[subscription_id]

        del self.subscriptions[subscription_id]
        logger.info(f"Deleted subscription {subscription_id}")

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
            node_id: Node identifier
            start_time: Start of time range
            end_time: End of time range
            num_values: Maximum values to return

        Returns:
            List of historical data points
        """
        self._check_running()

        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        node = self.nodes[node_id]

        if not node.historizing:
            raise ValueError(f"Node {node_id} does not have historizing enabled")

        # Filter history by time range
        filtered = [
            dp for dp in node.history
            if start_time <= dp.timestamp <= end_time
        ]

        return filtered[:num_values]

    async def _notify_subscribers(self, node_id: str, value: Any) -> None:
        """Notify all subscribers of a node value change."""
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]

        for sub_id, callback in node.subscriptions.items():
            try:
                notification = {
                    "node_id": node_id,
                    "value": value,
                    "source_timestamp": node.source_timestamp,
                    "server_timestamp": node.server_timestamp,
                    "status_code": node.status_code
                }

                if asyncio.iscoroutinefunction(callback):
                    await callback(notification)
                else:
                    callback(notification)
            except Exception as e:
                logger.error(f"Subscription callback error: {e}")

    def _check_running(self) -> None:
        """Check if server is running."""
        if not self.running:
            raise ConnectionError("Server not running")

    # ==========================================================================
    # Test Helper Methods
    # ==========================================================================

    def simulate_disconnect(self) -> None:
        """Simulate server going offline."""
        self.running = False
        logger.info("Simulated server disconnect")

    def simulate_reconnect(self) -> None:
        """Simulate server coming back online."""
        self.running = True
        logger.info("Simulated server reconnect")

    def enable_error_mode(self) -> None:
        """Enable error simulation mode."""
        self._error_mode = True
        logger.info("Error mode enabled")

    def disable_error_mode(self) -> None:
        """Disable error simulation mode."""
        self._error_mode = False
        logger.info("Error mode disabled")

    def set_latency(self, latency_ms: int) -> None:
        """Set simulated network latency."""
        self._latency_ms = latency_ms
        logger.info(f"Latency set to {latency_ms}ms")

    def simulate_value_change(self, node_id: str, new_value: Any) -> None:
        """
        Simulate an external value change (e.g., from PLC).

        Useful for testing subscription callbacks.
        """
        if node_id in self.nodes:
            asyncio.create_task(self.write_value(node_id, new_value))

    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "endpoint": self.endpoint,
            "running": self.running,
            "security_mode": self.security_mode.value,
            "node_count": len(self.nodes),
            "connected_clients": len(self.connected_clients),
            "active_subscriptions": len(self.subscriptions),
            "error_mode": self._error_mode,
            "latency_ms": self._latency_ms
        }
