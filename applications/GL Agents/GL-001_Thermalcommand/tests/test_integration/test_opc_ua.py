"""
Integration Tests: OPC-UA Connectivity

Tests OPC-UA client connectivity and data exchange including:
- Connection establishment and management
- Node reading and writing
- Subscription management
- Security configuration
- Error handling and reconnection

Reference: GL-001 Specification Section 11.3
Target Coverage: 85%+
"""

import pytest
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import AsyncMock, MagicMock, patch
from enum import Enum


# =============================================================================
# OPC-UA Classes (Simulated Production Code)
# =============================================================================

class OPCUAConnectionError(Exception):
    """Raised when OPC-UA connection fails."""
    pass


class OPCUAReadError(Exception):
    """Raised when OPC-UA read operation fails."""
    pass


class OPCUAWriteError(Exception):
    """Raised when OPC-UA write operation fails."""
    pass


class SecurityMode(Enum):
    """OPC-UA security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class MessageSecurityMode(Enum):
    """OPC-UA message security modes."""
    NONE = 0
    SIGN = 1
    SIGN_AND_ENCRYPT = 2


@dataclass
class OPCUAConfig:
    """OPC-UA client configuration."""
    endpoint_url: str
    security_mode: SecurityMode = SecurityMode.SIGN_AND_ENCRYPT
    security_policy: str = "Basic256Sha256"
    username: Optional[str] = None
    password: Optional[str] = None
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    timeout: float = 30.0
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 3


@dataclass
class OPCUANodeValue:
    """Value read from OPC-UA node."""
    node_id: str
    value: Any
    timestamp: datetime
    quality: str
    source_timestamp: Optional[datetime] = None


@dataclass
class OPCUASubscription:
    """OPC-UA subscription details."""
    subscription_id: str
    node_ids: List[str]
    publishing_interval: float
    callback: Callable
    active: bool = True


class OPCUAClient:
    """OPC-UA client for SCADA integration."""

    def __init__(self, config: OPCUAConfig):
        self.config = config
        self.connected = False
        self.subscriptions: Dict[str, OPCUASubscription] = {}
        self._session = None
        self._reconnect_count = 0

    async def connect(self) -> bool:
        """Establish connection to OPC-UA server."""
        if self.connected:
            return True

        try:
            # Simulate connection
            self._session = MagicMock()
            self.connected = True
            self._reconnect_count = 0
            return True
        except Exception as e:
            raise OPCUAConnectionError(f"Failed to connect: {str(e)}")

    async def disconnect(self) -> bool:
        """Disconnect from OPC-UA server."""
        if not self.connected:
            return True

        try:
            # Cancel all subscriptions
            for sub_id in list(self.subscriptions.keys()):
                await self.unsubscribe(sub_id)

            self._session = None
            self.connected = False
            return True
        except Exception as e:
            raise OPCUAConnectionError(f"Failed to disconnect: {str(e)}")

    async def read_node(self, node_id: str) -> OPCUANodeValue:
        """Read value from OPC-UA node."""
        if not self.connected:
            raise OPCUAReadError("Not connected to OPC-UA server")

        try:
            # Simulate node read
            return OPCUANodeValue(
                node_id=node_id,
                value=100.0,
                timestamp=datetime.now(),
                quality="Good"
            )
        except Exception as e:
            raise OPCUAReadError(f"Failed to read node {node_id}: {str(e)}")

    async def read_nodes(self, node_ids: List[str]) -> List[OPCUANodeValue]:
        """Read values from multiple OPC-UA nodes."""
        if not self.connected:
            raise OPCUAReadError("Not connected to OPC-UA server")

        results = []
        for node_id in node_ids:
            results.append(await self.read_node(node_id))
        return results

    async def write_node(self, node_id: str, value: Any) -> bool:
        """Write value to OPC-UA node."""
        if not self.connected:
            raise OPCUAWriteError("Not connected to OPC-UA server")

        try:
            # Simulate node write
            return True
        except Exception as e:
            raise OPCUAWriteError(f"Failed to write node {node_id}: {str(e)}")

    async def subscribe(self, node_ids: List[str], callback: Callable,
                       publishing_interval: float = 1000) -> str:
        """Create subscription to node changes."""
        if not self.connected:
            raise OPCUAConnectionError("Not connected to OPC-UA server")

        subscription_id = f"sub_{len(self.subscriptions)}"
        self.subscriptions[subscription_id] = OPCUASubscription(
            subscription_id=subscription_id,
            node_ids=node_ids,
            publishing_interval=publishing_interval,
            callback=callback
        )
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Cancel subscription."""
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].active = False
            del self.subscriptions[subscription_id]
            return True
        return False

    async def get_endpoints(self) -> List[Dict[str, Any]]:
        """Get available server endpoints."""
        return [
            {
                "url": self.config.endpoint_url,
                "security_mode": self.config.security_mode.value,
                "security_policy": self.config.security_policy
            }
        ]

    async def browse_nodes(self, parent_node_id: str = "i=84") -> List[Dict[str, str]]:
        """Browse child nodes of specified node."""
        if not self.connected:
            raise OPCUAConnectionError("Not connected to OPC-UA server")

        # Simulate browse results
        return [
            {"node_id": "ns=2;s=Boiler.Temperature", "browse_name": "Temperature"},
            {"node_id": "ns=2;s=Boiler.Pressure", "browse_name": "Pressure"},
            {"node_id": "ns=2;s=Boiler.FlowRate", "browse_name": "FlowRate"}
        ]

    async def reconnect(self) -> bool:
        """Attempt to reconnect to server."""
        self._reconnect_count += 1

        if self._reconnect_count > self.config.max_reconnect_attempts:
            raise OPCUAConnectionError("Max reconnection attempts exceeded")

        self.connected = False
        await asyncio.sleep(self.config.reconnect_interval)
        return await self.connect()


# =============================================================================
# Test Classes
# =============================================================================

@pytest.mark.integration
class TestOPCUAConnection:
    """Test OPC-UA connection management."""

    @pytest.fixture
    def config(self):
        """Create OPC-UA configuration."""
        return OPCUAConfig(
            endpoint_url="opc.tcp://localhost:4840",
            security_mode=SecurityMode.SIGN_AND_ENCRYPT,
            timeout=30.0
        )

    @pytest.fixture
    def client(self, config):
        """Create OPC-UA client."""
        return OPCUAClient(config)

    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Test successful connection."""
        result = await client.connect()

        assert result == True
        assert client.connected == True

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, client):
        """Test connecting when already connected."""
        await client.connect()
        result = await client.connect()

        assert result == True
        assert client.connected == True

    @pytest.mark.asyncio
    async def test_disconnect_success(self, client):
        """Test successful disconnection."""
        await client.connect()
        result = await client.disconnect()

        assert result == True
        assert client.connected == False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, client):
        """Test disconnecting when not connected."""
        result = await client.disconnect()

        assert result == True

    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect(self, client):
        """Test reconnecting after disconnect."""
        await client.connect()
        await client.disconnect()
        result = await client.connect()

        assert result == True
        assert client.connected == True

    @pytest.mark.asyncio
    async def test_get_endpoints(self, client):
        """Test getting server endpoints."""
        endpoints = await client.get_endpoints()

        assert len(endpoints) > 0
        assert "url" in endpoints[0]
        assert "security_mode" in endpoints[0]


@pytest.mark.integration
class TestOPCUAReadWrite:
    """Test OPC-UA read/write operations."""

    @pytest.fixture
    async def connected_client(self):
        """Create and connect OPC-UA client."""
        config = OPCUAConfig(endpoint_url="opc.tcp://localhost:4840")
        client = OPCUAClient(config)
        await client.connect()
        yield client
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_read_node(self, connected_client):
        """Test reading single node."""
        result = await connected_client.read_node("ns=2;s=Boiler.Temperature")

        assert result.node_id == "ns=2;s=Boiler.Temperature"
        assert result.value is not None
        assert result.quality == "Good"
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_read_multiple_nodes(self, connected_client):
        """Test reading multiple nodes."""
        node_ids = [
            "ns=2;s=Boiler.Temperature",
            "ns=2;s=Boiler.Pressure",
            "ns=2;s=Boiler.FlowRate"
        ]
        results = await connected_client.read_nodes(node_ids)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.node_id == node_ids[i]

    @pytest.mark.asyncio
    async def test_read_when_disconnected_fails(self):
        """Test reading when not connected raises error."""
        config = OPCUAConfig(endpoint_url="opc.tcp://localhost:4840")
        client = OPCUAClient(config)

        with pytest.raises(OPCUAReadError):
            await client.read_node("ns=2;s=Test")

    @pytest.mark.asyncio
    async def test_write_node(self, connected_client):
        """Test writing to node."""
        result = await connected_client.write_node(
            "ns=2;s=Boiler.Setpoint",
            value=450.0
        )

        assert result == True

    @pytest.mark.asyncio
    async def test_write_when_disconnected_fails(self):
        """Test writing when not connected raises error."""
        config = OPCUAConfig(endpoint_url="opc.tcp://localhost:4840")
        client = OPCUAClient(config)

        with pytest.raises(OPCUAWriteError):
            await client.write_node("ns=2;s=Test", 100.0)


@pytest.mark.integration
class TestOPCUASubscriptions:
    """Test OPC-UA subscription management."""

    @pytest.fixture
    async def connected_client(self):
        """Create and connect OPC-UA client."""
        config = OPCUAConfig(endpoint_url="opc.tcp://localhost:4840")
        client = OPCUAClient(config)
        await client.connect()
        yield client
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_create_subscription(self, connected_client):
        """Test creating subscription."""
        callback = MagicMock()
        node_ids = ["ns=2;s=Boiler.Temperature"]

        sub_id = await connected_client.subscribe(
            node_ids=node_ids,
            callback=callback,
            publishing_interval=1000
        )

        assert sub_id is not None
        assert sub_id in connected_client.subscriptions
        assert connected_client.subscriptions[sub_id].active == True

    @pytest.mark.asyncio
    async def test_cancel_subscription(self, connected_client):
        """Test canceling subscription."""
        callback = MagicMock()
        sub_id = await connected_client.subscribe(
            node_ids=["ns=2;s=Test"],
            callback=callback
        )

        result = await connected_client.unsubscribe(sub_id)

        assert result == True
        assert sub_id not in connected_client.subscriptions

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_subscription(self, connected_client):
        """Test canceling subscription that doesn't exist."""
        result = await connected_client.unsubscribe("nonexistent_sub")

        assert result == False

    @pytest.mark.asyncio
    async def test_multiple_subscriptions(self, connected_client):
        """Test multiple simultaneous subscriptions."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        sub_id1 = await connected_client.subscribe(["ns=2;s=Node1"], callback1)
        sub_id2 = await connected_client.subscribe(["ns=2;s=Node2"], callback2)

        assert sub_id1 != sub_id2
        assert len(connected_client.subscriptions) == 2

    @pytest.mark.asyncio
    async def test_subscriptions_cleaned_on_disconnect(self, connected_client):
        """Test subscriptions are cleaned up on disconnect."""
        callback = MagicMock()
        await connected_client.subscribe(["ns=2;s=Test"], callback)

        await connected_client.disconnect()

        assert len(connected_client.subscriptions) == 0


@pytest.mark.integration
class TestOPCUABrowsing:
    """Test OPC-UA node browsing."""

    @pytest.fixture
    async def connected_client(self):
        """Create and connect OPC-UA client."""
        config = OPCUAConfig(endpoint_url="opc.tcp://localhost:4840")
        client = OPCUAClient(config)
        await client.connect()
        yield client
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_browse_nodes(self, connected_client):
        """Test browsing child nodes."""
        nodes = await connected_client.browse_nodes()

        assert len(nodes) > 0
        assert "node_id" in nodes[0]
        assert "browse_name" in nodes[0]

    @pytest.mark.asyncio
    async def test_browse_when_disconnected_fails(self):
        """Test browsing when not connected raises error."""
        config = OPCUAConfig(endpoint_url="opc.tcp://localhost:4840")
        client = OPCUAClient(config)

        with pytest.raises(OPCUAConnectionError):
            await client.browse_nodes()


@pytest.mark.integration
class TestOPCUAReconnection:
    """Test OPC-UA reconnection logic."""

    @pytest.fixture
    def config(self):
        """Create OPC-UA configuration with fast reconnect."""
        return OPCUAConfig(
            endpoint_url="opc.tcp://localhost:4840",
            reconnect_interval=0.1,
            max_reconnect_attempts=3
        )

    @pytest.mark.asyncio
    async def test_reconnect_success(self, config):
        """Test successful reconnection."""
        client = OPCUAClient(config)
        await client.connect()
        client.connected = False  # Simulate connection loss

        result = await client.reconnect()

        assert result == True
        assert client.connected == True

    @pytest.mark.asyncio
    async def test_reconnect_attempts_tracked(self, config):
        """Test reconnection attempts are tracked."""
        client = OPCUAClient(config)

        # Multiple reconnects
        for _ in range(2):
            client.connected = False
            await client.reconnect()

        assert client._reconnect_count == 2

    @pytest.mark.asyncio
    async def test_reconnect_max_attempts_exceeded(self, config):
        """Test max reconnection attempts exceeded."""
        client = OPCUAClient(config)
        client._reconnect_count = config.max_reconnect_attempts

        with pytest.raises(OPCUAConnectionError) as exc_info:
            await client.reconnect()

        assert "Max reconnection" in str(exc_info.value)


@pytest.mark.integration
class TestOPCUASecurity:
    """Test OPC-UA security configuration."""

    def test_security_mode_none(self):
        """Test creating client with no security."""
        config = OPCUAConfig(
            endpoint_url="opc.tcp://localhost:4840",
            security_mode=SecurityMode.NONE
        )

        client = OPCUAClient(config)

        assert client.config.security_mode == SecurityMode.NONE

    def test_security_mode_sign_and_encrypt(self):
        """Test creating client with sign and encrypt security."""
        config = OPCUAConfig(
            endpoint_url="opc.tcp://localhost:4840",
            security_mode=SecurityMode.SIGN_AND_ENCRYPT,
            security_policy="Basic256Sha256",
            certificate_path="/path/to/cert.pem",
            private_key_path="/path/to/key.pem"
        )

        client = OPCUAClient(config)

        assert client.config.security_mode == SecurityMode.SIGN_AND_ENCRYPT
        assert client.config.security_policy == "Basic256Sha256"

    def test_authentication_config(self):
        """Test creating client with username/password auth."""
        config = OPCUAConfig(
            endpoint_url="opc.tcp://localhost:4840",
            username="admin",
            password="secret"
        )

        client = OPCUAClient(config)

        assert client.config.username == "admin"
        assert client.config.password == "secret"
