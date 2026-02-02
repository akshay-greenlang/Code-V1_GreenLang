# -*- coding: utf-8 -*-
"""
OPC-UA Integration Tests for Process Heat Agents
=================================================

Comprehensive integration tests for OPC-UA client functionality:
- Connection establishment with mock server
- Node reading (single, batch)
- Node writing with provenance tracking
- Subscription and data change callbacks
- Reconnection after disconnect
- Security modes testing
- Historical data queries

Test Coverage Target: 85%+

References:
- greenlang/infrastructure/protocols/opcua_client.py
- greenlang/infrastructure/protocols/opcua_server.py

Author: GreenLang Test Engineering Team
Date: December 2025
"""

import asyncio
import hashlib
import logging
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from tests.integration.protocols.conftest import (
    MockOPCUAServer,
    MockOPCUANode,
    SecurityMode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Test Class: OPC-UA Connection Tests
# =============================================================================


class TestOPCUAConnection:
    """Test OPC-UA connection establishment and management."""

    @pytest.mark.asyncio
    async def test_connection_establishment(self, running_mock_opcua_server):
        """Test successful connection to OPC-UA server."""
        server = running_mock_opcua_server
        client_id = "test-client-001"

        # Connect client
        connected = await server.connect_client(client_id)

        assert connected is True
        assert client_id in server.connected_clients
        assert server.connected_clients[client_id]["connected_at"] is not None

    @pytest.mark.asyncio
    async def test_connection_with_authentication(self, running_mock_opcua_server):
        """Test connection with username/password authentication."""
        server = running_mock_opcua_server
        server.set_security_mode(SecurityMode.SIGN)

        client_id = "authenticated-client"

        # Connection should succeed with credentials
        connected = await server.connect_client(
            client_id,
            username="admin",
            password="secret123"
        )

        assert connected is True
        assert server.connected_clients[client_id]["username"] == "admin"

    @pytest.mark.asyncio
    async def test_connection_fails_without_auth_when_required(self, running_mock_opcua_server):
        """Test connection fails without auth when security enabled."""
        server = running_mock_opcua_server
        server.set_security_mode(SecurityMode.SIGN_AND_ENCRYPT)

        client_id = "unauthenticated-client"

        # Connection should fail without credentials
        with pytest.raises(PermissionError, match="Authentication required"):
            await server.connect_client(client_id)

    @pytest.mark.asyncio
    async def test_connection_fails_when_server_down(self, mock_opcua_server):
        """Test connection fails when server is not running."""
        server = mock_opcua_server
        # Server not started

        with pytest.raises(ConnectionError, match="Server not running"):
            await server.connect_client("test-client")

    @pytest.mark.asyncio
    async def test_disconnect_client(self, running_mock_opcua_server):
        """Test client disconnection."""
        server = running_mock_opcua_server
        client_id = "disconnect-test-client"

        await server.connect_client(client_id)
        assert client_id in server.connected_clients

        await server.disconnect_client(client_id)
        assert client_id not in server.connected_clients

    @pytest.mark.asyncio
    async def test_connection_with_latency(self, running_mock_opcua_server):
        """Test connection with simulated network latency."""
        server = running_mock_opcua_server
        server.set_latency(100)  # 100ms latency

        client_id = "latency-test-client"
        start_time = datetime.utcnow()

        await server.connect_client(client_id)

        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        assert elapsed >= 100  # Should take at least 100ms

    @pytest.mark.asyncio
    async def test_connection_error_mode(self, running_mock_opcua_server):
        """Test connection fails in error mode."""
        server = running_mock_opcua_server
        server.enable_error_mode()

        with pytest.raises(ConnectionError, match="Simulated connection error"):
            await server.connect_client("error-test-client")


# =============================================================================
# Test Class: OPC-UA Node Reading Tests
# =============================================================================


class TestOPCUANodeReading:
    """Test OPC-UA node reading functionality."""

    @pytest.mark.asyncio
    async def test_read_single_value(self, running_mock_opcua_server):
        """Test reading a single node value."""
        server = running_mock_opcua_server

        value = await server.read_value("ns=2;s=Temperature")

        assert value == 85.5

    @pytest.mark.asyncio
    async def test_read_multiple_values_batch(self, running_mock_opcua_server):
        """Test batch reading of multiple node values."""
        server = running_mock_opcua_server

        node_ids = [
            "ns=2;s=Temperature",
            "ns=2;s=Pressure",
            "ns=2;s=FlowRate"
        ]

        values = await server.read_values(node_ids)

        assert len(values) == 3
        assert values["ns=2;s=Temperature"] == 85.5
        assert values["ns=2;s=Pressure"] == 2.5
        assert values["ns=2;s=FlowRate"] == 150.0

    @pytest.mark.asyncio
    async def test_read_nonexistent_node(self, running_mock_opcua_server):
        """Test reading a non-existent node raises error."""
        server = running_mock_opcua_server

        with pytest.raises(ValueError, match="not found"):
            await server.read_value("ns=2;s=NonExistent")

    @pytest.mark.asyncio
    async def test_read_string_value(self, running_mock_opcua_server):
        """Test reading a string node value."""
        server = running_mock_opcua_server

        value = await server.read_value("ns=2;s=Status")

        assert value == "RUNNING"

    @pytest.mark.asyncio
    async def test_read_with_latency(self, running_mock_opcua_server):
        """Test reading with simulated latency."""
        server = running_mock_opcua_server
        server.set_latency(50)  # 50ms latency

        start_time = datetime.utcnow()
        await server.read_value("ns=2;s=Temperature")
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

        assert elapsed >= 50

    @pytest.mark.asyncio
    async def test_read_in_error_mode(self, running_mock_opcua_server):
        """Test reading fails in error mode."""
        server = running_mock_opcua_server
        server.enable_error_mode()

        with pytest.raises(Exception, match="Simulated read error"):
            await server.read_value("ns=2;s=Temperature")

    @pytest.mark.asyncio
    async def test_batch_read_with_missing_nodes(self, running_mock_opcua_server):
        """Test batch read returns None for missing nodes."""
        server = running_mock_opcua_server

        node_ids = [
            "ns=2;s=Temperature",
            "ns=2;s=NonExistent",
            "ns=2;s=Pressure"
        ]

        values = await server.read_values(node_ids)

        assert values["ns=2;s=Temperature"] == 85.5
        assert values["ns=2;s=NonExistent"] is None
        assert values["ns=2;s=Pressure"] == 2.5

    @pytest.mark.asyncio
    async def test_read_all_process_heat_nodes(self, running_mock_opcua_server):
        """Test reading all process heat related nodes."""
        server = running_mock_opcua_server

        process_heat_nodes = [
            "ns=2;s=Temperature",
            "ns=2;s=Pressure",
            "ns=2;s=FlowRate",
            "ns=2;s=FuelConsumption",
            "ns=2;s=EmissionsFactor",
            "ns=2;s=TotalEmissions",
            "ns=2;s=Status",
            "ns=2;s=Efficiency"
        ]

        values = await server.read_values(process_heat_nodes)

        assert len(values) == 8
        assert values["ns=2;s=Temperature"] == 85.5
        assert values["ns=2;s=EmissionsFactor"] == 2.68
        assert values["ns=2;s=TotalEmissions"] == 121.14
        assert values["ns=2;s=Efficiency"] == 0.92


# =============================================================================
# Test Class: OPC-UA Node Writing Tests
# =============================================================================


class TestOPCUANodeWriting:
    """Test OPC-UA node writing functionality."""

    @pytest.mark.asyncio
    async def test_write_single_value(self, running_mock_opcua_server):
        """Test writing a single node value."""
        server = running_mock_opcua_server

        provenance_hash = await server.write_value("ns=2;s=Temperature", 90.0)

        # Verify value was written
        value = await server.read_value("ns=2;s=Temperature")
        assert value == 90.0

        # Verify provenance hash is valid SHA-256
        assert len(provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in provenance_hash)

    @pytest.mark.asyncio
    async def test_write_with_timestamp(self, running_mock_opcua_server):
        """Test writing with source timestamp."""
        server = running_mock_opcua_server

        source_ts = datetime(2025, 1, 15, 10, 30, 0)

        await server.write_value(
            "ns=2;s=Temperature",
            95.0,
            source_timestamp=source_ts
        )

        node = server.nodes["ns=2;s=Temperature"]
        assert node.value == 95.0
        assert node.source_timestamp == source_ts

    @pytest.mark.asyncio
    async def test_write_updates_history(self, running_mock_opcua_server):
        """Test that writing updates node history."""
        server = running_mock_opcua_server

        initial_value = await server.read_value("ns=2;s=Temperature")

        # Write new value
        await server.write_value("ns=2;s=Temperature", 100.0)

        # Check history
        node = server.nodes["ns=2;s=Temperature"]
        assert len(node.history) > 0
        last_history = node.history[-1]
        assert last_history["value"] == 100.0
        assert last_history["old_value"] == initial_value

    @pytest.mark.asyncio
    async def test_write_nonexistent_node(self, running_mock_opcua_server):
        """Test writing to non-existent node raises error."""
        server = running_mock_opcua_server

        with pytest.raises(ValueError, match="not found"):
            await server.write_value("ns=2;s=NonExistent", 123.0)

    @pytest.mark.asyncio
    async def test_write_in_error_mode(self, running_mock_opcua_server):
        """Test writing fails in error mode."""
        server = running_mock_opcua_server
        server.enable_error_mode()

        with pytest.raises(Exception, match="Simulated write error"):
            await server.write_value("ns=2;s=Temperature", 100.0)

    @pytest.mark.asyncio
    async def test_write_provenance_determinism(self, running_mock_opcua_server):
        """Test that same write produces consistent provenance format."""
        server = running_mock_opcua_server

        # Two writes with same value should produce different hashes
        # (due to timestamp difference)
        hash1 = await server.write_value("ns=2;s=Temperature", 80.0)
        hash2 = await server.write_value("ns=2;s=Temperature", 80.0)

        # Hashes should be different (different timestamps)
        assert hash1 != hash2

        # Both should be valid SHA-256 hashes
        assert len(hash1) == 64
        assert len(hash2) == 64

    @pytest.mark.asyncio
    async def test_write_multiple_values_sequentially(self, running_mock_opcua_server):
        """Test writing multiple values in sequence."""
        server = running_mock_opcua_server

        writes = [
            ("ns=2;s=Temperature", 95.0),
            ("ns=2;s=Pressure", 3.0),
            ("ns=2;s=FlowRate", 200.0),
        ]

        for node_id, value in writes:
            await server.write_value(node_id, value)

        # Verify all writes
        assert await server.read_value("ns=2;s=Temperature") == 95.0
        assert await server.read_value("ns=2;s=Pressure") == 3.0
        assert await server.read_value("ns=2;s=FlowRate") == 200.0


# =============================================================================
# Test Class: OPC-UA Subscription Tests
# =============================================================================


class TestOPCUASubscriptions:
    """Test OPC-UA subscription and data change callback functionality."""

    @pytest.mark.asyncio
    async def test_create_subscription(self, running_mock_opcua_server):
        """Test creating a subscription to a node."""
        server = running_mock_opcua_server
        received_notifications = []

        async def callback(notification):
            received_notifications.append(notification)

        sub_id = await server.subscribe(
            client_id="test-client",
            node_id="ns=2;s=Temperature",
            callback=callback
        )

        assert sub_id is not None
        assert sub_id in server.subscriptions

    @pytest.mark.asyncio
    async def test_subscription_callback_on_value_change(self, running_mock_opcua_server):
        """Test subscription callback is triggered on value change."""
        server = running_mock_opcua_server
        received_notifications = []

        async def callback(notification):
            received_notifications.append(notification)

        await server.subscribe(
            client_id="test-client",
            node_id="ns=2;s=Temperature",
            callback=callback
        )

        # Trigger value change
        await server.write_value("ns=2;s=Temperature", 100.0)

        # Callback should have been triggered
        assert len(received_notifications) == 1
        assert received_notifications[0]["node_id"] == "ns=2;s=Temperature"
        assert received_notifications[0]["value"] == 100.0

    @pytest.mark.asyncio
    async def test_multiple_subscriptions_same_node(self, running_mock_opcua_server):
        """Test multiple subscriptions to the same node."""
        server = running_mock_opcua_server
        notifications1 = []
        notifications2 = []

        async def callback1(notification):
            notifications1.append(notification)

        async def callback2(notification):
            notifications2.append(notification)

        await server.subscribe("client-1", "ns=2;s=Temperature", callback1)
        await server.subscribe("client-2", "ns=2;s=Temperature", callback2)

        # Trigger value change
        await server.write_value("ns=2;s=Temperature", 90.0)

        # Both callbacks should receive notification
        assert len(notifications1) == 1
        assert len(notifications2) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, running_mock_opcua_server):
        """Test unsubscribing from a node."""
        server = running_mock_opcua_server
        received_notifications = []

        async def callback(notification):
            received_notifications.append(notification)

        sub_id = await server.subscribe(
            client_id="test-client",
            node_id="ns=2;s=Temperature",
            callback=callback
        )

        # Unsubscribe
        await server.unsubscribe(sub_id)

        # Trigger value change
        await server.write_value("ns=2;s=Temperature", 100.0)

        # Callback should NOT have been triggered
        assert len(received_notifications) == 0

    @pytest.mark.asyncio
    async def test_subscription_notification_contains_timestamps(self, running_mock_opcua_server):
        """Test subscription notifications contain timestamps."""
        server = running_mock_opcua_server
        received_notifications = []

        async def callback(notification):
            received_notifications.append(notification)

        await server.subscribe(
            client_id="test-client",
            node_id="ns=2;s=Temperature",
            callback=callback
        )

        await server.write_value("ns=2;s=Temperature", 88.0)

        assert len(received_notifications) == 1
        notification = received_notifications[0]
        assert "source_timestamp" in notification
        assert "server_timestamp" in notification
        assert notification["source_timestamp"] is not None
        assert notification["server_timestamp"] is not None

    @pytest.mark.asyncio
    async def test_subscribe_to_nonexistent_node(self, running_mock_opcua_server):
        """Test subscribing to non-existent node raises error."""
        server = running_mock_opcua_server

        async def callback(notification):
            pass

        with pytest.raises(ValueError, match="not found"):
            await server.subscribe(
                client_id="test-client",
                node_id="ns=2;s=NonExistent",
                callback=callback
            )

    @pytest.mark.asyncio
    async def test_sync_callback_handling(self, running_mock_opcua_server):
        """Test synchronous callback handling."""
        server = running_mock_opcua_server
        received_notifications = []

        def sync_callback(notification):
            received_notifications.append(notification)

        await server.subscribe(
            client_id="test-client",
            node_id="ns=2;s=Temperature",
            callback=sync_callback
        )

        await server.write_value("ns=2;s=Temperature", 77.0)

        assert len(received_notifications) == 1
        assert received_notifications[0]["value"] == 77.0


# =============================================================================
# Test Class: OPC-UA Reconnection Tests
# =============================================================================


class TestOPCUAReconnection:
    """Test OPC-UA reconnection behavior."""

    @pytest.mark.asyncio
    async def test_server_disconnect_detection(self, running_mock_opcua_server):
        """Test detection of server disconnect."""
        server = running_mock_opcua_server

        # Simulate disconnect
        server.simulate_disconnect()

        assert server.running is False

        # Operations should fail
        with pytest.raises(ConnectionError, match="Server not running"):
            await server.read_value("ns=2;s=Temperature")

    @pytest.mark.asyncio
    async def test_server_reconnect_restores_operations(self, running_mock_opcua_server):
        """Test server reconnection restores operations."""
        server = running_mock_opcua_server

        # Disconnect
        server.simulate_disconnect()

        # Reconnect
        server.simulate_reconnect()

        # Operations should work again
        value = await server.read_value("ns=2;s=Temperature")
        assert value is not None

    @pytest.mark.asyncio
    async def test_auto_disconnect_after_n_operations(self, running_mock_opcua_server):
        """Test server disconnects after N operations."""
        server = running_mock_opcua_server
        server.set_disconnect_after(3)

        # First 3 operations should succeed
        await server.read_value("ns=2;s=Temperature")
        await server.read_value("ns=2;s=Pressure")

        # Third operation triggers disconnect
        with pytest.raises(ConnectionError):
            await server.read_value("ns=2;s=FlowRate")

    @pytest.mark.asyncio
    async def test_client_disconnect_clears_subscriptions(self, running_mock_opcua_server):
        """Test client disconnect clears its subscriptions."""
        server = running_mock_opcua_server

        async def callback(notification):
            pass

        client_id = "subscription-client"
        await server.connect_client(client_id)

        sub_id = await server.subscribe(
            client_id=client_id,
            node_id="ns=2;s=Temperature",
            callback=callback
        )

        assert sub_id in server.subscriptions

        # Disconnect client
        await server.disconnect_client(client_id)

        # Subscription should be removed
        assert sub_id not in server.subscriptions


# =============================================================================
# Test Class: OPC-UA Security Modes
# =============================================================================


class TestOPCUASecurityModes:
    """Test OPC-UA security mode configurations."""

    @pytest.mark.asyncio
    async def test_no_security_mode(self, running_mock_opcua_server):
        """Test connection with no security."""
        server = running_mock_opcua_server
        server.set_security_mode(SecurityMode.NONE)

        # Should connect without credentials
        connected = await server.connect_client("anonymous-client")
        assert connected is True

    @pytest.mark.asyncio
    async def test_sign_security_mode(self, running_mock_opcua_server):
        """Test connection with Sign security mode."""
        server = running_mock_opcua_server
        server.set_security_mode(SecurityMode.SIGN)

        # Should require authentication
        with pytest.raises(PermissionError):
            await server.connect_client("unauthenticated")

        # Should succeed with credentials
        connected = await server.connect_client(
            "authenticated",
            username="user",
            password="pass"
        )
        assert connected is True

    @pytest.mark.asyncio
    async def test_sign_and_encrypt_security_mode(self, running_mock_opcua_server):
        """Test connection with SignAndEncrypt security mode."""
        server = running_mock_opcua_server
        server.set_security_mode(SecurityMode.SIGN_AND_ENCRYPT)

        # Should require authentication
        with pytest.raises(PermissionError):
            await server.connect_client("unauthenticated")

        # Should succeed with certificate (simulated by credentials)
        connected = await server.connect_client(
            "cert-client",
            certificate=b"fake-certificate"
        )
        assert connected is True

    @pytest.mark.asyncio
    async def test_security_mode_recorded_in_client_info(self, running_mock_opcua_server):
        """Test security mode is recorded in client connection info."""
        server = running_mock_opcua_server
        server.set_security_mode(SecurityMode.SIGN_AND_ENCRYPT)

        await server.connect_client(
            "secure-client",
            username="admin",
            password="secret"
        )

        client_info = server.connected_clients["secure-client"]
        assert client_info["security_mode"] == "SignAndEncrypt"


# =============================================================================
# Test Class: OPC-UA Browse Tests
# =============================================================================


class TestOPCUABrowse:
    """Test OPC-UA node browsing functionality."""

    @pytest.mark.asyncio
    async def test_browse_all_nodes(self, running_mock_opcua_server):
        """Test browsing all available nodes."""
        server = running_mock_opcua_server

        nodes = await server.browse()

        assert len(nodes) > 0

        # Should contain process heat nodes
        node_ids = [n["node_id"] for n in nodes]
        assert "ns=2;s=Temperature" in node_ids
        assert "ns=2;s=Pressure" in node_ids

    @pytest.mark.asyncio
    async def test_browse_returns_node_metadata(self, running_mock_opcua_server):
        """Test browse returns proper node metadata."""
        server = running_mock_opcua_server

        nodes = await server.browse()

        temp_node = next(n for n in nodes if n["node_id"] == "ns=2;s=Temperature")

        assert temp_node["browse_name"] == "Temperature"
        assert temp_node["display_name"] == "Process Temperature"
        assert temp_node["data_type"] == "Double"


# =============================================================================
# Test Class: OPC-UA Historical Data Tests
# =============================================================================


class TestOPCUAHistoricalData:
    """Test OPC-UA historical data queries."""

    @pytest.mark.asyncio
    async def test_read_history_after_writes(self, running_mock_opcua_server):
        """Test reading historical data after multiple writes."""
        server = running_mock_opcua_server

        # Write multiple values to create history
        await server.write_value("ns=2;s=Temperature", 80.0)
        await asyncio.sleep(0.01)
        await server.write_value("ns=2;s=Temperature", 85.0)
        await asyncio.sleep(0.01)
        await server.write_value("ns=2;s=Temperature", 90.0)

        # Read history
        start_time = datetime.utcnow() - timedelta(minutes=5)
        end_time = datetime.utcnow() + timedelta(minutes=5)

        history = await server.read_history(
            "ns=2;s=Temperature",
            start_time,
            end_time
        )

        assert len(history) >= 3  # At least our 3 writes
        values = [h["value"] for h in history]
        assert 80.0 in values
        assert 85.0 in values
        assert 90.0 in values

    @pytest.mark.asyncio
    async def test_read_history_time_filter(self, running_mock_opcua_server):
        """Test historical data is filtered by time range."""
        server = running_mock_opcua_server

        # Write a value
        await server.write_value("ns=2;s=Temperature", 100.0)

        # Query with future time range (should return nothing)
        start_time = datetime.utcnow() + timedelta(hours=1)
        end_time = datetime.utcnow() + timedelta(hours=2)

        history = await server.read_history(
            "ns=2;s=Temperature",
            start_time,
            end_time
        )

        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_read_history_nonexistent_node(self, running_mock_opcua_server):
        """Test reading history from non-existent node."""
        server = running_mock_opcua_server

        with pytest.raises(ValueError, match="not found"):
            await server.read_history(
                "ns=2;s=NonExistent",
                datetime.utcnow() - timedelta(hours=1),
                datetime.utcnow()
            )

    @pytest.mark.asyncio
    async def test_read_history_non_historizing_node(self, running_mock_opcua_server):
        """Test reading history from node without historizing enabled."""
        server = running_mock_opcua_server

        # Add node without historizing
        server.add_node(MockOPCUANode(
            node_id="ns=2;s=NoHistory",
            browse_name="NoHistory",
            display_name="No History Node",
            value=50.0,
            historizing=False
        ))

        with pytest.raises(ValueError, match="historizing"):
            await server.read_history(
                "ns=2;s=NoHistory",
                datetime.utcnow() - timedelta(hours=1),
                datetime.utcnow()
            )


# =============================================================================
# Test Class: OPC-UA Statistics Tests
# =============================================================================


class TestOPCUAStatistics:
    """Test OPC-UA server statistics."""

    @pytest.mark.asyncio
    async def test_server_statistics(self, running_mock_opcua_server):
        """Test server statistics are accurate."""
        server = running_mock_opcua_server

        stats = server.get_statistics()

        assert stats["running"] is True
        assert stats["endpoint"] == "opc.tcp://localhost:4840/greenlang/"
        assert stats["node_count"] > 0
        assert stats["connected_clients"] == 0
        assert stats["active_subscriptions"] == 0

    @pytest.mark.asyncio
    async def test_statistics_update_on_connection(self, running_mock_opcua_server):
        """Test statistics update on client connection."""
        server = running_mock_opcua_server

        await server.connect_client("stats-client")

        stats = server.get_statistics()
        assert stats["connected_clients"] == 1

    @pytest.mark.asyncio
    async def test_statistics_update_on_subscription(self, running_mock_opcua_server):
        """Test statistics update on subscription."""
        server = running_mock_opcua_server

        async def callback(notification):
            pass

        await server.subscribe("client", "ns=2;s=Temperature", callback)

        stats = server.get_statistics()
        assert stats["active_subscriptions"] == 1


# =============================================================================
# Test Class: OPC-UA Performance Tests
# =============================================================================


@pytest.mark.performance
class TestOPCUAPerformance:
    """Performance tests for OPC-UA operations."""

    @pytest.mark.asyncio
    async def test_read_latency(self, running_mock_opcua_server, performance_timer):
        """Test read operation latency."""
        server = running_mock_opcua_server
        num_reads = 100

        for _ in range(num_reads):
            performance_timer.start()
            await server.read_value("ns=2;s=Temperature")
            performance_timer.stop()

        # Average read should be very fast (mock server)
        assert performance_timer.average_ms < 10  # <10ms average

    @pytest.mark.asyncio
    async def test_batch_read_throughput(self, running_mock_opcua_server, throughput_calculator):
        """Test batch read throughput."""
        server = running_mock_opcua_server

        node_ids = [
            "ns=2;s=Temperature",
            "ns=2;s=Pressure",
            "ns=2;s=FlowRate",
            "ns=2;s=EmissionsFactor"
        ]

        throughput_calculator.start()

        for _ in range(100):
            values = await server.read_values(node_ids)
            throughput_calculator.record_message(len(str(values)))

        stats = throughput_calculator.get_throughput()

        # Should handle many reads per second
        assert stats["messages_per_sec"] > 100

    @pytest.mark.asyncio
    async def test_write_latency(self, running_mock_opcua_server, performance_timer):
        """Test write operation latency."""
        server = running_mock_opcua_server
        num_writes = 50

        for i in range(num_writes):
            performance_timer.start()
            await server.write_value("ns=2;s=Temperature", 80.0 + i)
            performance_timer.stop()

        # Average write should be fast
        assert performance_timer.average_ms < 20  # <20ms average

    @pytest.mark.asyncio
    async def test_subscription_notification_latency(self, running_mock_opcua_server):
        """Test subscription notification latency."""
        server = running_mock_opcua_server
        notification_times = []

        async def callback(notification):
            notification_times.append(datetime.utcnow())

        await server.subscribe("perf-client", "ns=2;s=Temperature", callback)

        # Trigger multiple value changes
        for i in range(10):
            write_time = datetime.utcnow()
            await server.write_value("ns=2;s=Temperature", 80.0 + i)

        # All notifications should have been received
        assert len(notification_times) == 10


# =============================================================================
# Test Class: OPC-UA Process Heat Integration
# =============================================================================


class TestOPCUAProcessHeatIntegration:
    """Integration tests for process heat data via OPC-UA."""

    @pytest.mark.asyncio
    async def test_read_all_process_heat_data(
        self,
        running_mock_opcua_server,
        sample_process_heat_data
    ):
        """Test reading all process heat related data."""
        server = running_mock_opcua_server

        # Read process heat values
        temp = await server.read_value("ns=2;s=Temperature")
        pressure = await server.read_value("ns=2;s=Pressure")
        flow_rate = await server.read_value("ns=2;s=FlowRate")
        fuel_consumption = await server.read_value("ns=2;s=FuelConsumption")
        emission_factor = await server.read_value("ns=2;s=EmissionsFactor")
        total_emissions = await server.read_value("ns=2;s=TotalEmissions")
        efficiency = await server.read_value("ns=2;s=Efficiency")

        # Validate expected values
        assert temp == pytest.approx(85.5, rel=0.01)
        assert pressure == pytest.approx(2.5, rel=0.01)
        assert flow_rate == pytest.approx(150.0, rel=0.01)
        assert fuel_consumption == pytest.approx(45.2, rel=0.01)
        assert emission_factor == pytest.approx(2.68, rel=0.01)
        assert total_emissions == pytest.approx(121.14, rel=0.01)
        assert efficiency == pytest.approx(0.92, rel=0.01)

    @pytest.mark.asyncio
    async def test_calculate_emissions_from_opcua_data(self, running_mock_opcua_server):
        """Test calculating emissions from OPC-UA process heat data."""
        server = running_mock_opcua_server

        # Read fuel consumption and emission factor
        fuel_consumption = await server.read_value("ns=2;s=FuelConsumption")
        emission_factor = await server.read_value("ns=2;s=EmissionsFactor")

        # Calculate expected emissions
        calculated_emissions = fuel_consumption * emission_factor

        # Read actual emissions
        total_emissions = await server.read_value("ns=2;s=TotalEmissions")

        # Should match (allowing for rounding)
        assert calculated_emissions == pytest.approx(total_emissions, rel=0.01)

    @pytest.mark.asyncio
    async def test_subscribe_to_process_heat_changes(self, running_mock_opcua_server):
        """Test subscribing to process heat value changes."""
        server = running_mock_opcua_server
        temperature_changes = []
        emission_changes = []

        async def temp_callback(notification):
            temperature_changes.append(notification["value"])

        async def emission_callback(notification):
            emission_changes.append(notification["value"])

        await server.subscribe("heat-client", "ns=2;s=Temperature", temp_callback)
        await server.subscribe("heat-client", "ns=2;s=TotalEmissions", emission_callback)

        # Simulate process changes
        await server.write_value("ns=2;s=Temperature", 95.0)
        await server.write_value("ns=2;s=TotalEmissions", 150.0)

        assert 95.0 in temperature_changes
        assert 150.0 in emission_changes
