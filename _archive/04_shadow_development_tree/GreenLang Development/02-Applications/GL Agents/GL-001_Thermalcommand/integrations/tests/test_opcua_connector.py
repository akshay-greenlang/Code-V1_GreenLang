# -*- coding: utf-8 -*-
"""
Unit Tests for OPC-UA Connector

Tests comprehensive validation of OPC-UA connection functionality:
- Connection lifecycle
- Circuit breaker behavior
- Subscription management
- Data buffering
- Reconnection logic
- Connection pooling

Author: GL-BackendDeveloper
Version: 1.0.0
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from integrations.opcua_connector import (
    ConnectionState,
    CircuitBreakerState,
    CircuitBreaker,
    DataBuffer,
    OPCUASubscriptionManager,
    ConnectionPool,
    OPCUAConnector,
)
from integrations.opcua_schemas import (
    OPCUAConnectionConfig,
    OPCUASubscriptionConfig,
    OPCUATagConfig,
    OPCUADataPoint,
    OPCUAQualityCode,
    OPCUASecurityConfig,
    TagMetadata,
    TagDataType,
    SecurityMode,
    SecurityPolicy,
)


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreaker:
    """Test circuit breaker fault tolerance."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker instance."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout_s=10,
            half_open_max_calls=2,
        )

    @pytest.mark.asyncio
    async def test_initial_state_closed(self, circuit_breaker):
        """Test circuit breaker starts in CLOSED state."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert await circuit_breaker.can_execute()

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self, circuit_breaker):
        """Test circuit breaker opens after threshold failures."""
        # Record failures
        for _ in range(3):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert not await circuit_breaker.can_execute()

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, circuit_breaker):
        """Test success resets failure count."""
        await circuit_breaker.record_failure()
        await circuit_breaker.record_failure()
        await circuit_breaker.record_success()

        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_after_recovery_timeout(self, circuit_breaker):
        """Test transition to HALF_OPEN after recovery timeout."""
        # Open the circuit
        for _ in range(3):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Simulate time passing (mock the last_failure_time)
        circuit_breaker.last_failure_time = (
            circuit_breaker.last_failure_time - circuit_breaker.recovery_timeout_s - 1
        )

        # Should now transition to HALF_OPEN
        assert await circuit_breaker.can_execute()
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_successful_half_open(self, circuit_breaker):
        """Test circuit closes after successful HALF_OPEN period."""
        # Open the circuit
        for _ in range(3):
            await circuit_breaker.record_failure()

        # Manually set to HALF_OPEN
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN
        circuit_breaker.success_count = 0

        # Record successful calls
        await circuit_breaker.record_success()
        await circuit_breaker.record_success()

        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_reopens_on_half_open_failure(self, circuit_breaker):
        """Test circuit reopens on failure during HALF_OPEN."""
        circuit_breaker.state = CircuitBreakerState.HALF_OPEN

        await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitBreakerState.OPEN

    def test_get_state(self, circuit_breaker):
        """Test state serialization."""
        state = circuit_breaker.get_state()

        assert "state" in state
        assert "failure_count" in state
        assert "success_count" in state


# =============================================================================
# DATA BUFFER TESTS
# =============================================================================

class TestDataBuffer:
    """Test data buffer functionality."""

    @pytest.fixture
    def data_buffer(self):
        """Create data buffer instance."""
        return DataBuffer(max_size=100, retention_hours=24)

    @pytest.fixture
    def sample_data_point(self):
        """Create sample data point."""
        now = datetime.now(timezone.utc)
        return OPCUADataPoint(
            tag_id="test_tag",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.value",
            value=42.0,
            data_type=TagDataType.DOUBLE,
            source_timestamp=now,
            server_timestamp=now,
            quality_code=OPCUAQualityCode.GOOD,
        )

    @pytest.mark.asyncio
    async def test_add_data_point(self, data_buffer, sample_data_point):
        """Test adding data point to buffer."""
        seq = await data_buffer.add(sample_data_point)

        assert seq == 1
        assert sample_data_point.sequence_number == 1

    @pytest.mark.asyncio
    async def test_sequence_numbers_increment(self, data_buffer, sample_data_point):
        """Test sequence numbers increment correctly."""
        seq1 = await data_buffer.add(sample_data_point)
        seq2 = await data_buffer.add(sample_data_point)
        seq3 = await data_buffer.add(sample_data_point)

        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3

    @pytest.mark.asyncio
    async def test_get_recent(self, data_buffer, sample_data_point):
        """Test getting recent data points."""
        await data_buffer.add(sample_data_point)

        recent = await data_buffer.get_recent(minutes=60)

        assert len(recent) == 1
        assert recent[0].tag_id == "test_tag"

    @pytest.mark.asyncio
    async def test_get_recent_with_tag_filter(self, data_buffer):
        """Test getting recent data points filtered by tag."""
        now = datetime.now(timezone.utc)

        # Add different tags
        for tag_id in ["tag_a", "tag_b", "tag_a"]:
            dp = OPCUADataPoint(
                tag_id=tag_id,
                node_id="ns=2;s=Test",
                canonical_name=f"test.{tag_id}.value",
                value=42.0,
                data_type=TagDataType.DOUBLE,
                source_timestamp=now,
                server_timestamp=now,
            )
            await data_buffer.add(dp)

        recent_a = await data_buffer.get_recent(tag_id="tag_a", minutes=60)
        recent_b = await data_buffer.get_recent(tag_id="tag_b", minutes=60)

        assert len(recent_a) == 2
        assert len(recent_b) == 1

    @pytest.mark.asyncio
    async def test_buffer_respects_max_size(self, data_buffer, sample_data_point):
        """Test buffer respects max size limit."""
        # Fill buffer beyond max size
        for _ in range(150):
            await data_buffer.add(sample_data_point)

        stats = await data_buffer.get_stats()
        assert stats["current_size"] <= data_buffer.max_size

    @pytest.mark.asyncio
    async def test_get_by_sequence(self, data_buffer, sample_data_point):
        """Test getting data points by sequence range."""
        for _ in range(5):
            await data_buffer.add(sample_data_point)

        result = await data_buffer.get_by_sequence(2, 4)

        assert len(result) == 3
        sequences = [dp.sequence_number for dp in result]
        assert sequences == [2, 3, 4]


# =============================================================================
# SUBSCRIPTION MANAGER TESTS
# =============================================================================

class TestOPCUASubscriptionManager:
    """Test OPC-UA subscription management."""

    @pytest.fixture
    def mock_connector(self):
        """Create mock connector."""
        connector = MagicMock()
        connector.is_connected.return_value = True
        return connector

    @pytest.fixture
    def subscription_manager(self, mock_connector):
        """Create subscription manager instance."""
        return OPCUASubscriptionManager(mock_connector)

    @pytest.fixture
    def sample_subscription_config(self):
        """Create sample subscription configuration."""
        metadata = TagMetadata(
            tag_id="test_tag",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.value",
            display_name="Test Tag",
            data_type=TagDataType.DOUBLE,
        )
        tag_config = OPCUATagConfig(metadata=metadata)

        return OPCUASubscriptionConfig(
            name="test_subscription",
            publishing_interval_ms=1000,
            tag_configs=[tag_config],
        )

    @pytest.mark.asyncio
    async def test_create_subscription(
        self, subscription_manager, sample_subscription_config
    ):
        """Test subscription creation."""
        subscription = await subscription_manager.create_subscription(
            sample_subscription_config
        )

        assert subscription.status == "active"
        assert subscription.is_connected
        assert len(subscription.monitored_item_handles) == 1

    @pytest.mark.asyncio
    async def test_create_subscription_not_connected(
        self, subscription_manager, sample_subscription_config
    ):
        """Test subscription fails when not connected."""
        subscription_manager.connector.is_connected.return_value = False

        with pytest.raises(ConnectionError):
            await subscription_manager.create_subscription(sample_subscription_config)

    @pytest.mark.asyncio
    async def test_delete_subscription(
        self, subscription_manager, sample_subscription_config
    ):
        """Test subscription deletion."""
        subscription = await subscription_manager.create_subscription(
            sample_subscription_config
        )

        result = await subscription_manager.delete_subscription(
            sample_subscription_config.subscription_id
        )

        assert result is True
        assert sample_subscription_config.subscription_id not in subscription_manager.subscriptions

    @pytest.mark.asyncio
    async def test_process_data_change(
        self, subscription_manager, sample_subscription_config
    ):
        """Test data change notification processing."""
        subscription = await subscription_manager.create_subscription(
            sample_subscription_config
        )

        now = datetime.now(timezone.utc)
        data_point = await subscription_manager.process_data_change(
            subscription_id=sample_subscription_config.subscription_id,
            tag_id="test_tag",
            value=42.0,
            source_timestamp=now,
            server_timestamp=now,
            quality=OPCUAQualityCode.GOOD,
        )

        assert data_point.value == 42.0
        assert data_point.provenance_hash is not None
        assert subscription.notification_count == 1

    @pytest.mark.asyncio
    async def test_callback_registration(
        self, subscription_manager, sample_subscription_config
    ):
        """Test callback registration and invocation."""
        callback_data = []

        def test_callback(data_point):
            callback_data.append(data_point)

        subscription_manager.register_callback("test_tag", test_callback)

        await subscription_manager.create_subscription(sample_subscription_config)

        now = datetime.now(timezone.utc)
        await subscription_manager.process_data_change(
            subscription_id=sample_subscription_config.subscription_id,
            tag_id="test_tag",
            value=42.0,
            source_timestamp=now,
            server_timestamp=now,
            quality=OPCUAQualityCode.GOOD,
        )

        assert len(callback_data) == 1
        assert callback_data[0].value == 42.0

    @pytest.mark.asyncio
    async def test_get_recent_data(
        self, subscription_manager, sample_subscription_config
    ):
        """Test getting recent data from buffer."""
        await subscription_manager.create_subscription(sample_subscription_config)

        now = datetime.now(timezone.utc)
        await subscription_manager.process_data_change(
            subscription_id=sample_subscription_config.subscription_id,
            tag_id="test_tag",
            value=42.0,
            source_timestamp=now,
            server_timestamp=now,
            quality=OPCUAQualityCode.GOOD,
        )

        recent = await subscription_manager.get_recent_data(minutes=60)

        assert len(recent) == 1


# =============================================================================
# CONNECTION POOL TESTS
# =============================================================================

class TestConnectionPool:
    """Test connection pool functionality."""

    @pytest.fixture
    def connection_pool(self):
        """Create connection pool instance."""
        return ConnectionPool(max_connections=5)

    @pytest.fixture
    def sample_connection_config(self):
        """Create sample connection configuration."""
        return OPCUAConnectionConfig(
            name="test_connection",
            endpoint_url="opc.tcp://localhost:4840",
            security=OPCUASecurityConfig(
                security_mode=SecurityMode.NONE,
                security_policy=SecurityPolicy.NONE,
            ),
        )

    @pytest.mark.asyncio
    async def test_add_connection(self, connection_pool, sample_connection_config):
        """Test adding connection to pool."""
        connector = await connection_pool.add_connection(
            "conn1", sample_connection_config
        )

        assert connector is not None
        assert "conn1" in connection_pool.connections
        assert connection_pool.health_status["conn1"]

    @pytest.mark.asyncio
    async def test_get_connection(self, connection_pool, sample_connection_config):
        """Test getting connection from pool."""
        await connection_pool.add_connection("conn1", sample_connection_config)

        connector = await connection_pool.get_connection("conn1")

        assert connector is not None

    @pytest.mark.asyncio
    async def test_get_any_healthy_connection(
        self, connection_pool, sample_connection_config
    ):
        """Test getting any healthy connection."""
        await connection_pool.add_connection("conn1", sample_connection_config)

        connector = await connection_pool.get_connection()  # No specific ID

        assert connector is not None

    @pytest.mark.asyncio
    async def test_remove_connection(self, connection_pool, sample_connection_config):
        """Test removing connection from pool."""
        await connection_pool.add_connection("conn1", sample_connection_config)

        result = await connection_pool.remove_connection("conn1")

        assert result is True
        assert "conn1" not in connection_pool.connections

    @pytest.mark.asyncio
    async def test_pool_max_connections(self, connection_pool, sample_connection_config):
        """Test pool enforces max connections."""
        # Fill pool
        for i in range(5):
            await connection_pool.add_connection(
                f"conn{i}", sample_connection_config
            )

        # Should fail to add more
        with pytest.raises(ValueError, match="pool full"):
            await connection_pool.add_connection("conn5", sample_connection_config)

    @pytest.mark.asyncio
    async def test_health_check(self, connection_pool, sample_connection_config):
        """Test health check on all connections."""
        await connection_pool.add_connection("conn1", sample_connection_config)

        health = await connection_pool.health_check()

        assert "conn1" in health
        assert health["conn1"]

    def test_get_stats(self, connection_pool):
        """Test pool statistics."""
        stats = connection_pool.get_stats()

        assert "total_connections" in stats
        assert "max_connections" in stats
        assert "healthy_connections" in stats


# =============================================================================
# OPC-UA CONNECTOR TESTS
# =============================================================================

class TestOPCUAConnector:
    """Test main OPC-UA connector."""

    @pytest.fixture
    def connector_config(self):
        """Create connector configuration."""
        return OPCUAConnectionConfig(
            name="test_connector",
            endpoint_url="opc.tcp://localhost:4840",
            security=OPCUASecurityConfig(
                security_mode=SecurityMode.NONE,
                security_policy=SecurityPolicy.NONE,
            ),
            auto_reconnect=True,
            reconnect_interval_ms=1000,
            max_reconnect_attempts=3,
        )

    @pytest.fixture
    def connector(self, connector_config):
        """Create connector instance."""
        return OPCUAConnector(connector_config)

    def test_initial_state(self, connector):
        """Test connector initial state."""
        assert connector.state == ConnectionState.DISCONNECTED
        assert not connector.is_connected()

    @pytest.mark.asyncio
    async def test_connect(self, connector):
        """Test connection establishment."""
        result = await connector.connect()

        assert result is True
        assert connector.state == ConnectionState.CONNECTED
        assert connector.is_connected()

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Test disconnection."""
        await connector.connect()
        result = await connector.disconnect()

        assert result is True
        assert connector.state == ConnectionState.DISCONNECTED
        assert not connector.is_connected()

    @pytest.mark.asyncio
    async def test_already_connected(self, connector):
        """Test connect when already connected."""
        await connector.connect()
        result = await connector.connect()

        assert result is True  # Should still return True

    @pytest.mark.asyncio
    async def test_read_tag(self, connector):
        """Test reading tag value."""
        await connector.connect()

        data_point = await connector.read_tag("ns=2;s=Test")

        assert data_point is not None
        assert data_point.provenance_hash is not None

    @pytest.mark.asyncio
    async def test_read_tag_not_connected(self, connector):
        """Test read fails when not connected."""
        with pytest.raises(ConnectionError):
            await connector.read_tag("ns=2;s=Test")

    @pytest.mark.asyncio
    async def test_create_subscription(self, connector):
        """Test subscription creation through connector."""
        await connector.connect()

        metadata = TagMetadata(
            tag_id="test_tag",
            node_id="ns=2;s=Test",
            canonical_name="test.tag.value",
            display_name="Test",
            data_type=TagDataType.DOUBLE,
        )
        config = OPCUASubscriptionConfig(
            name="test_sub",
            tag_configs=[OPCUATagConfig(metadata=metadata)],
        )

        subscription = await connector.create_subscription(config)

        assert subscription is not None
        assert subscription.status == "active"

    @pytest.mark.asyncio
    async def test_context_manager(self, connector_config):
        """Test async context manager usage."""
        async with OPCUAConnector(connector_config) as connector:
            assert connector.is_connected()

        # After context exit, should be disconnected
        assert not connector.is_connected()

    def test_get_stats(self, connector):
        """Test connector statistics."""
        stats = connector.get_stats()

        assert "connection_id" in stats
        assert "endpoint_url" in stats
        assert "state" in stats
        assert "circuit_breaker" in stats


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestOPCUAConnectorIntegration:
    """Integration tests for OPC-UA connector."""

    @pytest.fixture
    def connector_with_subscription(self, connector_config):
        """Create connector with active subscription."""
        return OPCUAConnector(connector_config)

    @pytest.mark.asyncio
    async def test_full_workflow(self, connector_with_subscription):
        """Test complete workflow: connect, subscribe, receive data, disconnect."""
        connector = connector_with_subscription

        # Connect
        await connector.connect()
        assert connector.is_connected()

        # Create subscription
        metadata = TagMetadata(
            tag_id="steam_pressure",
            node_id="ns=2;s=Steam.Pressure",
            canonical_name="steam.header.pressure",
            display_name="Steam Pressure",
            data_type=TagDataType.DOUBLE,
        )
        config = OPCUASubscriptionConfig(
            name="steam_monitoring",
            tag_configs=[OPCUATagConfig(metadata=metadata)],
        )
        subscription = await connector.create_subscription(config)

        # Register callback
        received_data = []

        def callback(dp):
            received_data.append(dp)

        connector.register_data_callback("steam_pressure", callback)

        # Simulate data change
        now = datetime.now(timezone.utc)
        await connector.subscription_manager.process_data_change(
            subscription_id=config.subscription_id,
            tag_id="steam_pressure",
            value=15.5,
            source_timestamp=now,
            server_timestamp=now,
            quality=OPCUAQualityCode.GOOD,
        )

        # Verify callback was invoked
        assert len(received_data) == 1
        assert received_data[0].value == 15.5

        # Disconnect
        await connector.disconnect()
        assert not connector.is_connected()
