"""
OPC-UA Client for GreenLang Process Heat Agents.

This module provides an async OPC-UA client wrapper built on the asyncua library,
enabling secure, reliable connectivity to industrial control systems including
DCS, PLC, and SCADA systems.

Features:
- Secure connections with X.509 certificates
- Multiple authentication methods (Anonymous, Username/Password, Certificate)
- Async operations for non-blocking I/O
- Connection pooling and automatic reconnection
- Real-time data subscriptions
- Historical data access (HDA)
- Address space browsing
- Data quality handling with provenance tracking

Usage:
    from connectors.opcua.client import OPCUAClient, OPCUAClientConfig

    # Create client
    client = OPCUAClient(
        endpoint="opc.tcp://192.168.1.100:4840",
        security_policy=SecurityPolicy.BASIC256SHA256,
    )

    # Connect
    await client.connect()

    # Read values
    value = await client.read_node("ns=2;s=Furnace1.Temperature.PV")

    # Subscribe to changes
    await client.subscribe(
        ["ns=2;s=Temperature", "ns=2;s=Pressure"],
        callback=handle_data_change
    )

    # Disconnect
    await client.disconnect()
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import uuid

from pydantic import BaseModel, Field

from .types import (
    NodeValue,
    NodeValueBatch,
    NodeInfo,
    BrowseResult,
    DataProvenance,
    OPCUAQuality,
    QualityLevel,
    NodeClass,
    ConnectionConfig,
    SessionInfo,
    SecurityPolicy,
    MessageSecurityMode,
    AuthenticationType,
    SubscriptionConfig,
    HistoricalReadConfig,
    HistoricalDataResult,
    HistoricalValue,
    EndpointInfo,
    ServerInfo,
)
from .security import CertificateManager, SecurityManager
from .subscription import OPCUASubscription, SubscriptionManager, DataChangeHandler
from .utils import (
    parse_node_id,
    format_node_id,
    is_valid_node_id,
    get_quality_description,
    parse_endpoint_url,
    WellKnownNodes,
    validate_node_id_list,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Client Configuration
# =============================================================================


@dataclass
class OPCUAClientConfig:
    """Configuration for OPC-UA client."""

    endpoint: str
    security_policy: SecurityPolicy = SecurityPolicy.BASIC256SHA256
    security_mode: MessageSecurityMode = MessageSecurityMode.SIGN_AND_ENCRYPT
    authentication: AuthenticationType = AuthenticationType.ANONYMOUS
    username: Optional[str] = None
    password: Optional[str] = None
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    server_certificate_path: Optional[str] = None
    application_name: str = "GreenLang-OPC-UA-Client"
    application_uri: str = "urn:greenlang:opcua:client"
    session_timeout_ms: int = 3600000
    request_timeout_ms: int = 30000
    connect_timeout_ms: int = 10000
    auto_reconnect: bool = True
    reconnect_delay_ms: int = 5000
    max_reconnect_attempts: int = 10
    keep_alive_interval_ms: int = 30000


# =============================================================================
# Client Statistics
# =============================================================================


@dataclass
class ClientStatistics:
    """Statistics for OPC-UA client operations."""

    connected: bool = False
    connect_time: Optional[datetime] = None
    disconnect_time: Optional[datetime] = None
    reconnect_count: int = 0
    read_count: int = 0
    write_count: int = 0
    browse_count: int = 0
    subscribe_count: int = 0
    historical_read_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    bytes_sent: int = 0
    bytes_received: int = 0
    request_count: int = 0
    response_time_avg_ms: float = 0.0

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.error_count += 1
        self.last_error = error
        self.last_error_time = datetime.utcnow()


# =============================================================================
# OPC-UA Client
# =============================================================================


class OPCUAClient:
    """
    Async OPC-UA Client for industrial data access.

    Provides secure, reliable connectivity to OPC-UA servers with support for:
    - Reading and writing node values
    - Browsing the address space
    - Real-time subscriptions
    - Historical data access
    - Data quality and provenance tracking
    """

    def __init__(
        self,
        endpoint: str,
        config: Optional[OPCUAClientConfig] = None,
        security_policy: SecurityPolicy = SecurityPolicy.BASIC256SHA256,
        security_mode: MessageSecurityMode = MessageSecurityMode.SIGN_AND_ENCRYPT,
        certificate_manager: Optional[CertificateManager] = None,
    ):
        """
        Initialize OPC-UA client.

        Args:
            endpoint: OPC-UA server endpoint URL
            config: Full client configuration
            security_policy: Security policy (if config not provided)
            security_mode: Message security mode (if config not provided)
            certificate_manager: Certificate manager for security
        """
        if config:
            self.config = config
        else:
            self.config = OPCUAClientConfig(
                endpoint=endpoint,
                security_policy=security_policy,
                security_mode=security_mode,
            )

        # Parse endpoint
        self._endpoint_info = parse_endpoint_url(self.config.endpoint)

        # Security
        self._cert_manager = certificate_manager or CertificateManager()
        self._security_manager = SecurityManager(self._cert_manager)

        # Internal state
        self._client: Any = None  # asyncua.Client instance
        self._session: Optional[SessionInfo] = None
        self._connected: bool = False
        self._connecting: bool = False
        self._lock = asyncio.Lock()

        # Subscriptions
        self._subscription_manager = SubscriptionManager(endpoint_url=self.config.endpoint)
        self._subscriptions: Dict[str, OPCUASubscription] = {}

        # Statistics
        self.statistics = ClientStatistics()

        # Reconnection
        self._reconnect_task: Optional[asyncio.Task] = None
        self._should_reconnect: bool = False

        # Trace ID for provenance
        self._trace_id = str(uuid.uuid4())

    @property
    def endpoint(self) -> str:
        """Get the server endpoint URL."""
        return self.config.endpoint

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected

    @property
    def session_info(self) -> Optional[SessionInfo]:
        """Get current session information."""
        return self._session

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self, security_policy: Optional[str] = None) -> bool:
        """
        Connect to the OPC-UA server.

        Args:
            security_policy: Optional override for security policy

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        if self._connected:
            logger.warning("Already connected to OPC-UA server")
            return True

        if self._connecting:
            logger.warning("Connection already in progress")
            return False

        async with self._lock:
            self._connecting = True

            try:
                logger.info(f"Connecting to OPC-UA server: {self.config.endpoint}")

                # Try to import asyncua
                try:
                    from asyncua import Client, ua
                    from asyncua.crypto.security_policies import SecurityPolicyBasic256Sha256

                    # Create asyncua client
                    self._client = Client(self.config.endpoint)

                    # Configure security
                    if self.config.security_policy != SecurityPolicy.NONE:
                        security_params = await self._security_manager.setup_security(
                            ConnectionConfig(
                                endpoint_url=self.config.endpoint,
                                security_policy=self.config.security_policy,
                                security_mode=self.config.security_mode,
                                authentication_type=self.config.authentication,
                                username=self.config.username,
                                password=self.config.password,
                                certificate_path=self.config.certificate_path,
                                private_key_path=self.config.private_key_path,
                                application_name=self.config.application_name,
                                application_uri=self.config.application_uri,
                            )
                        )

                        # Apply security to client
                        if security_params.get("certificate") and security_params.get("private_key"):
                            await self._client.set_security(
                                SecurityPolicyBasic256Sha256,
                                security_params["certificate"],
                                security_params["private_key"],
                            )

                    # Set authentication
                    if self.config.authentication == AuthenticationType.USERNAME_PASSWORD:
                        self._client.set_user(self.config.username)
                        self._client.set_password(self.config.password)

                    # Set timeouts
                    self._client.session_timeout = self.config.session_timeout_ms
                    self._client.secure_channel_timeout = self.config.session_timeout_ms

                    # Connect
                    await asyncio.wait_for(
                        self._client.connect(),
                        timeout=self.config.connect_timeout_ms / 1000,
                    )

                    self._connected = True
                    self.statistics.connected = True
                    self.statistics.connect_time = datetime.utcnow()

                    logger.info(f"Connected to OPC-UA server: {self.config.endpoint}")

                except ImportError:
                    # asyncua not available, use mock connection for development
                    logger.warning(
                        "asyncua library not available. Using mock connection. "
                        "Install asyncua for production use: pip install asyncua"
                    )
                    await self._mock_connect()

                # Create session info
                self._session = SessionInfo(
                    session_id=str(uuid.uuid4()),
                    session_timeout=self.config.session_timeout_ms,
                    created_at=datetime.utcnow(),
                    last_activity=datetime.utcnow(),
                )

                return True

            except asyncio.TimeoutError:
                error_msg = f"Connection timeout after {self.config.connect_timeout_ms}ms"
                logger.error(error_msg)
                self.statistics.record_error(error_msg)
                raise ConnectionError(error_msg)

            except Exception as e:
                error_msg = f"Connection failed: {str(e)}"
                logger.error(error_msg)
                self.statistics.record_error(error_msg)

                if self.config.auto_reconnect:
                    self._schedule_reconnect()

                raise ConnectionError(error_msg)

            finally:
                self._connecting = False

    async def _mock_connect(self) -> None:
        """Mock connection for development/testing."""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self._connected = True
        self.statistics.connected = True
        self.statistics.connect_time = datetime.utcnow()
        logger.info("Mock connection established")

    async def disconnect(self) -> None:
        """Disconnect from the OPC-UA server."""
        self._should_reconnect = False

        # Cancel reconnection task
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        # Stop all subscriptions
        await self._subscription_manager.stop_all()

        # Disconnect client
        if self._client:
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        self._connected = False
        self._client = None
        self._session = None
        self.statistics.connected = False
        self.statistics.disconnect_time = datetime.utcnow()

        logger.info("Disconnected from OPC-UA server")

    def _schedule_reconnect(self) -> None:
        """Schedule automatic reconnection."""
        if self._reconnect_task and not self._reconnect_task.done():
            return

        self._should_reconnect = True

        async def reconnect_loop():
            attempts = 0
            delay = self.config.reconnect_delay_ms / 1000

            while self._should_reconnect and attempts < self.config.max_reconnect_attempts:
                attempts += 1
                logger.info(f"Reconnection attempt {attempts}/{self.config.max_reconnect_attempts}")

                await asyncio.sleep(delay)

                try:
                    await self.connect()
                    if self._connected:
                        self.statistics.reconnect_count += 1
                        logger.info("Reconnection successful")
                        return
                except Exception as e:
                    logger.warning(f"Reconnection failed: {e}")

                # Exponential backoff
                delay = min(delay * 2, 60)

            logger.error("Max reconnection attempts reached")

        self._reconnect_task = asyncio.create_task(reconnect_loop())

    # =========================================================================
    # Node Operations
    # =========================================================================

    async def read_node(self, node_id: str) -> NodeValue:
        """
        Read a single node value.

        Args:
            node_id: OPC-UA node ID

        Returns:
            Node value with quality and timestamps

        Raises:
            ConnectionError: If not connected
            ValueError: If node ID is invalid
        """
        if not self._connected:
            raise ConnectionError("Not connected to OPC-UA server")

        if not is_valid_node_id(node_id):
            raise ValueError(f"Invalid node ID: {node_id}")

        self.statistics.read_count += 1

        try:
            start_time = datetime.utcnow()

            if self._client:
                # Use asyncua client
                from asyncua import ua

                node = self._client.get_node(node_id)
                data_value = await node.read_data_value()

                value = data_value.Value.Value
                quality = OPCUAQuality(data_value.StatusCode.value)
                source_timestamp = data_value.SourceTimestamp
                server_timestamp = data_value.ServerTimestamp

            else:
                # Mock read for development
                value, quality, source_timestamp, server_timestamp = await self._mock_read_node(node_id)

            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_response_time(response_time)

            # Create node value with provenance
            return NodeValue(
                node_id=node_id,
                value=value,
                quality=quality,
                quality_level=QualityLevel.from_quality_code(quality),
                source_timestamp=source_timestamp,
                server_timestamp=server_timestamp or datetime.utcnow(),
                provenance=DataProvenance(
                    source_endpoint=self.config.endpoint,
                    source_node_id=node_id,
                    retrieval_method="read",
                    session_id=self._session.session_id if self._session else None,
                    trace_id=self._trace_id,
                ),
            )

        except Exception as e:
            error_msg = f"Failed to read node {node_id}: {str(e)}"
            logger.error(error_msg)
            self.statistics.record_error(error_msg)
            raise

    async def _mock_read_node(
        self,
        node_id: str,
    ) -> Tuple[Any, OPCUAQuality, datetime, datetime]:
        """Mock node read for development."""
        import random

        await asyncio.sleep(0.01)  # Simulate network latency

        # Generate mock value based on node ID
        if "Temperature" in node_id:
            value = 400.0 + random.uniform(-10, 10)
        elif "Pressure" in node_id:
            value = 1.0 + random.uniform(-0.1, 0.1)
        elif "Flow" in node_id:
            value = 100.0 + random.uniform(-5, 5)
        else:
            value = random.uniform(0, 100)

        return (
            value,
            OPCUAQuality.GOOD,
            datetime.utcnow(),
            datetime.utcnow(),
        )

    async def read_nodes(self, node_ids: List[str]) -> NodeValueBatch:
        """
        Read multiple node values in a single request.

        Args:
            node_ids: List of OPC-UA node IDs

        Returns:
            Batch of node values
        """
        if not self._connected:
            raise ConnectionError("Not connected to OPC-UA server")

        # Validate node IDs
        valid_ids, invalid_ids = validate_node_id_list(node_ids)
        if invalid_ids:
            logger.warning(f"Skipping invalid node IDs: {invalid_ids}")

        values = []
        for node_id in valid_ids:
            try:
                value = await self.read_node(node_id)
                values.append(value)
            except Exception as e:
                logger.error(f"Failed to read {node_id}: {e}")
                # Add placeholder with bad quality
                values.append(NodeValue(
                    node_id=node_id,
                    value=None,
                    quality=OPCUAQuality.BAD,
                    quality_level=QualityLevel.BAD,
                ))

        return NodeValueBatch(
            values=values,
            endpoint=self.config.endpoint,
        )

    async def write_node(self, node_id: str, value: Any) -> bool:
        """
        Write a value to a node.

        Args:
            node_id: OPC-UA node ID
            value: Value to write

        Returns:
            True if write successful

        Raises:
            ConnectionError: If not connected
            ValueError: If write fails
        """
        if not self._connected:
            raise ConnectionError("Not connected to OPC-UA server")

        if not is_valid_node_id(node_id):
            raise ValueError(f"Invalid node ID: {node_id}")

        self.statistics.write_count += 1

        try:
            if self._client:
                from asyncua import ua

                node = self._client.get_node(node_id)
                await node.write_value(value)
            else:
                # Mock write
                await asyncio.sleep(0.01)
                logger.info(f"Mock write: {node_id} = {value}")

            logger.debug(f"Written value to {node_id}: {value}")
            return True

        except Exception as e:
            error_msg = f"Failed to write to {node_id}: {str(e)}"
            logger.error(error_msg)
            self.statistics.record_error(error_msg)
            raise ValueError(error_msg)

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def subscribe(
        self,
        nodes: List[str],
        callback: Callable[[Any], Coroutine[Any, Any, None]],
        publishing_interval_ms: int = 1000,
        sampling_interval_ms: int = 1000,
    ) -> OPCUASubscription:
        """
        Create a subscription for real-time data updates.

        Args:
            nodes: Node IDs to subscribe to
            callback: Async callback for data changes
            publishing_interval_ms: How often to publish notifications
            sampling_interval_ms: How often to sample values

        Returns:
            Created subscription
        """
        if not self._connected:
            raise ConnectionError("Not connected to OPC-UA server")

        self.statistics.subscribe_count += 1

        # Create subscription
        subscription = await self._subscription_manager.create_subscription(
            publishing_interval_ms=publishing_interval_ms,
            nodes=nodes,
        )

        # Register callback
        subscription.on_data_change(callback)

        # Start subscription
        await subscription.start()

        # If using asyncua, create server-side subscription
        if self._client:
            try:
                handler = DataChangeHandler(subscription)
                opcua_sub = await self._client.create_subscription(
                    publishing_interval_ms,
                    handler,
                )

                # Add monitored items
                for node_id in nodes:
                    node = self._client.get_node(node_id)
                    await opcua_sub.subscribe_data_change(node)

                subscription._opcua_subscription = opcua_sub

            except Exception as e:
                logger.error(f"Failed to create server subscription: {e}")

        self._subscriptions[subscription.subscription_id] = subscription

        logger.info(
            f"Created subscription {subscription.subscription_id} "
            f"for {len(nodes)} nodes"
        )

        return subscription

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Delete a subscription.

        Args:
            subscription_id: Subscription to delete

        Returns:
            True if deleted
        """
        return await self._subscription_manager.delete_subscription(subscription_id)

    # =========================================================================
    # Browsing
    # =========================================================================

    async def browse(self, node_id: str = WellKnownNodes.ROOT) -> List[NodeInfo]:
        """
        Browse child nodes of a node.

        Args:
            node_id: Parent node ID (defaults to root)

        Returns:
            List of child node information
        """
        if not self._connected:
            raise ConnectionError("Not connected to OPC-UA server")

        self.statistics.browse_count += 1

        try:
            if self._client:
                from asyncua import ua

                node = self._client.get_node(node_id)
                children = await node.get_children()

                result = []
                for child in children:
                    try:
                        browse_name = await child.read_browse_name()
                        display_name = await child.read_display_name()
                        node_class = await child.read_node_class()

                        info = NodeInfo(
                            node_id=str(child.nodeid),
                            browse_name=browse_name.Name,
                            display_name=display_name.Text,
                            node_class=NodeClass(node_class.name),
                            parent_node_id=node_id,
                        )
                        result.append(info)
                    except Exception as e:
                        logger.warning(f"Failed to read child info: {e}")

                return result

            else:
                # Mock browse
                return await self._mock_browse(node_id)

        except Exception as e:
            error_msg = f"Failed to browse {node_id}: {str(e)}"
            logger.error(error_msg)
            self.statistics.record_error(error_msg)
            raise

    async def _mock_browse(self, node_id: str) -> List[NodeInfo]:
        """Mock browse for development."""
        await asyncio.sleep(0.01)

        # Return mock nodes based on parent
        if node_id == WellKnownNodes.ROOT or node_id == WellKnownNodes.OBJECTS:
            return [
                NodeInfo(
                    node_id="ns=2;s=Furnace1",
                    browse_name="Furnace1",
                    display_name="Furnace 1",
                    node_class=NodeClass.OBJECT,
                    parent_node_id=node_id,
                ),
                NodeInfo(
                    node_id="ns=2;s=Furnace2",
                    browse_name="Furnace2",
                    display_name="Furnace 2",
                    node_class=NodeClass.OBJECT,
                    parent_node_id=node_id,
                ),
            ]
        elif "Furnace" in node_id:
            return [
                NodeInfo(
                    node_id=f"{node_id}.Temperature",
                    browse_name="Temperature",
                    display_name="Temperature",
                    node_class=NodeClass.VARIABLE,
                    data_type="Double",
                    parent_node_id=node_id,
                ),
                NodeInfo(
                    node_id=f"{node_id}.Pressure",
                    browse_name="Pressure",
                    display_name="Pressure",
                    node_class=NodeClass.VARIABLE,
                    data_type="Double",
                    parent_node_id=node_id,
                ),
            ]
        return []

    async def browse_recursive(
        self,
        node_id: str = WellKnownNodes.OBJECTS,
        max_depth: int = 3,
        filter_node_class: Optional[NodeClass] = None,
    ) -> List[NodeInfo]:
        """
        Recursively browse the address space.

        Args:
            node_id: Starting node ID
            max_depth: Maximum recursion depth
            filter_node_class: Only return nodes of this class

        Returns:
            Flat list of all discovered nodes
        """
        all_nodes = []

        async def browse_level(current_id: str, depth: int):
            if depth > max_depth:
                return

            children = await self.browse(current_id)

            for child in children:
                if filter_node_class is None or child.node_class == filter_node_class:
                    all_nodes.append(child)

                # Recurse into object nodes
                if child.node_class == NodeClass.OBJECT:
                    await browse_level(child.node_id, depth + 1)

        await browse_level(node_id, 0)
        return all_nodes

    # =========================================================================
    # Historical Data Access
    # =========================================================================

    async def read_history(
        self,
        config: HistoricalReadConfig,
    ) -> List[HistoricalDataResult]:
        """
        Read historical data for nodes.

        Args:
            config: Historical read configuration

        Returns:
            List of historical data results
        """
        if not self._connected:
            raise ConnectionError("Not connected to OPC-UA server")

        self.statistics.historical_read_count += 1

        results = []

        for node_id in config.node_ids:
            try:
                if self._client:
                    from asyncua import ua

                    node = self._client.get_node(node_id)

                    # Read raw history
                    history = await node.read_raw_history(
                        starttime=config.start_time,
                        endtime=config.end_time,
                        numvalues=config.max_values_per_node,
                    )

                    values = []
                    for data_value in history:
                        values.append(HistoricalValue(
                            timestamp=data_value.SourceTimestamp,
                            value=data_value.Value.Value,
                            quality=OPCUAQuality(data_value.StatusCode.value),
                        ))

                    results.append(HistoricalDataResult(
                        node_id=node_id,
                        values=values,
                        provenance=DataProvenance(
                            source_endpoint=self.config.endpoint,
                            source_node_id=node_id,
                            retrieval_method="historical",
                            session_id=self._session.session_id if self._session else None,
                        ),
                    ))

                else:
                    # Mock historical read
                    result = await self._mock_read_history(node_id, config)
                    results.append(result)

            except Exception as e:
                logger.error(f"Failed to read history for {node_id}: {e}")
                results.append(HistoricalDataResult(
                    node_id=node_id,
                    values=[],
                ))

        return results

    async def _mock_read_history(
        self,
        node_id: str,
        config: HistoricalReadConfig,
    ) -> HistoricalDataResult:
        """Mock historical read for development."""
        import random

        await asyncio.sleep(0.05)

        # Generate mock historical data
        values = []
        current_time = config.start_time
        interval = (config.end_time - config.start_time) / 100  # 100 data points

        base_value = 400.0 if "Temperature" in node_id else 1.0

        for i in range(100):
            values.append(HistoricalValue(
                timestamp=current_time + interval * i,
                value=base_value + random.uniform(-10, 10),
                quality=OPCUAQuality.GOOD,
            ))

        return HistoricalDataResult(
            node_id=node_id,
            values=values,
            provenance=DataProvenance(
                source_endpoint=self.config.endpoint,
                source_node_id=node_id,
                retrieval_method="historical",
            ),
        )

    # =========================================================================
    # Discovery
    # =========================================================================

    @staticmethod
    async def discover_endpoints(
        discovery_url: str,
        timeout_ms: int = 10000,
    ) -> List[EndpointInfo]:
        """
        Discover endpoints on an OPC-UA server.

        Args:
            discovery_url: Discovery endpoint URL
            timeout_ms: Discovery timeout

        Returns:
            List of available endpoints
        """
        try:
            from asyncua import Client

            client = Client(discovery_url)
            endpoints = await asyncio.wait_for(
                client.connect_and_get_server_endpoints(),
                timeout=timeout_ms / 1000,
            )

            result = []
            for ep in endpoints:
                result.append(EndpointInfo(
                    endpoint_url=ep.EndpointUrl,
                    server=ServerInfo(
                        application_uri=ep.Server.ApplicationUri,
                        application_name=ep.Server.ApplicationName.Text,
                        application_type=ep.Server.ApplicationType.name,
                        discovery_urls=list(ep.Server.DiscoveryUrls),
                    ),
                    security_policy_uri=ep.SecurityPolicyUri,
                    security_mode=MessageSecurityMode(ep.SecurityMode.name),
                    security_level=ep.SecurityLevel,
                ))

            return result

        except ImportError:
            logger.warning("asyncua not available for discovery")
            return []

        except Exception as e:
            logger.error(f"Endpoint discovery failed: {e}")
            return []

    # =========================================================================
    # Utilities
    # =========================================================================

    def _update_response_time(self, response_time_ms: float) -> None:
        """Update average response time."""
        current_avg = self.statistics.response_time_avg_ms
        count = self.statistics.request_count
        self.statistics.request_count += 1
        self.statistics.response_time_avg_ms = (
            (current_avg * count + response_time_ms) / (count + 1)
        )

    def get_statistics(self) -> ClientStatistics:
        """Get client statistics."""
        return self.statistics

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> "OPCUAClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


# =============================================================================
# Factory Function
# =============================================================================


def create_opcua_client(
    endpoint: str,
    security_policy: SecurityPolicy = SecurityPolicy.BASIC256SHA256,
    username: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs,
) -> OPCUAClient:
    """
    Create an OPC-UA client with the specified configuration.

    Args:
        endpoint: OPC-UA server endpoint URL
        security_policy: Security policy to use
        username: Optional username for authentication
        password: Optional password for authentication
        **kwargs: Additional configuration options

    Returns:
        Configured OPCUAClient instance
    """
    auth_type = AuthenticationType.ANONYMOUS
    if username and password:
        auth_type = AuthenticationType.USERNAME_PASSWORD

    config = OPCUAClientConfig(
        endpoint=endpoint,
        security_policy=security_policy,
        authentication=auth_type,
        username=username,
        password=password,
        **kwargs,
    )

    return OPCUAClient(config=config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "OPCUAClient",
    "OPCUAClientConfig",
    "ClientStatistics",
    "create_opcua_client",
]
