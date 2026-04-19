# -*- coding: utf-8 -*-
"""
MQTT Integration Tests for Process Heat Agents
===============================================

Comprehensive integration tests for MQTT pub/sub messaging:
- Connect/disconnect with mock broker
- Publish with QoS 0, 1, 2
- Subscribe and message receive
- Topic wildcards (+ and #)
- Retained messages
- Last Will and Testament (LWT)
- Reconnection behavior
- Clean session handling

Test Coverage Target: 85%+

References:
- greenlang/infrastructure/protocols/mqtt_client.py

Author: GreenLang Test Engineering Team
Date: December 2025
"""

import asyncio
import json
import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from tests.integration.protocols.conftest import (
    MockMQTTBroker,
    MockMQTTMessage,
    QoSLevel,
)


# =============================================================================
# Test Class: MQTT Connection Tests
# =============================================================================


class TestMQTTConnection:
    """Test MQTT connection establishment and management."""

    @pytest.mark.asyncio
    async def test_connect_to_broker(self, mock_mqtt_broker):
        """Test successful connection to MQTT broker."""
        broker = mock_mqtt_broker

        connected = await broker.connect(client_id="test-client-001")

        assert connected is True
        assert "test-client-001" in broker.connected_clients

    @pytest.mark.asyncio
    async def test_connect_with_authentication(self, mock_mqtt_broker):
        """Test connection with username/password authentication."""
        broker = mock_mqtt_broker

        connected = await broker.connect(
            client_id="auth-client",
            username="admin",
            password="secret123"
        )

        assert connected is True
        client_info = broker.connected_clients["auth-client"]
        assert client_info["username"] == "admin"

    @pytest.mark.asyncio
    async def test_connect_with_clean_session(self, mock_mqtt_broker):
        """Test connection with clean session flag."""
        broker = mock_mqtt_broker

        await broker.connect(
            client_id="clean-session-client",
            clean_session=True
        )

        client_info = broker.connected_clients["clean-session-client"]
        assert client_info["clean_session"] is True

    @pytest.mark.asyncio
    async def test_connect_without_clean_session(self, mock_mqtt_broker):
        """Test connection without clean session (persistent session)."""
        broker = mock_mqtt_broker

        await broker.connect(
            client_id="persistent-client",
            clean_session=False
        )

        client_info = broker.connected_clients["persistent-client"]
        assert client_info["clean_session"] is False

    @pytest.mark.asyncio
    async def test_graceful_disconnect(self, connected_mock_mqtt_broker):
        """Test graceful disconnection from broker."""
        broker = connected_mock_mqtt_broker

        assert "test-client" in broker.connected_clients

        await broker.disconnect("test-client", graceful=True)

        assert "test-client" not in broker.connected_clients

    @pytest.mark.asyncio
    async def test_connection_fails_when_broker_down(self, mock_mqtt_broker):
        """Test connection fails when broker is not running."""
        broker = mock_mqtt_broker
        broker.simulate_disconnect()

        with pytest.raises(ConnectionError, match="not running"):
            await broker.connect("offline-client")

    @pytest.mark.asyncio
    async def test_multiple_clients_connect(self, mock_mqtt_broker):
        """Test multiple clients can connect simultaneously."""
        broker = mock_mqtt_broker

        await broker.connect("client-1")
        await broker.connect("client-2")
        await broker.connect("client-3")

        assert len(broker.connected_clients) == 3


# =============================================================================
# Test Class: MQTT Publish Tests
# =============================================================================


class TestMQTTPublish:
    """Test MQTT message publishing functionality."""

    @pytest.mark.asyncio
    async def test_publish_message(self, connected_mock_mqtt_broker):
        """Test publishing a message."""
        broker = connected_mock_mqtt_broker

        provenance_hash = await broker.publish(
            client_id="test-client",
            topic="process-heat/temperature",
            payload=b'{"value": 85.5}'
        )

        assert len(broker.message_history) == 1
        assert broker.message_history[0].topic == "process-heat/temperature"
        assert len(provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_publish_qos_0(self, connected_mock_mqtt_broker):
        """Test publishing with QoS 0 (at most once)."""
        broker = connected_mock_mqtt_broker

        await broker.publish(
            client_id="test-client",
            topic="sensors/temp",
            payload=b"85.5",
            qos=QoSLevel.AT_MOST_ONCE
        )

        msg = broker.message_history[0]
        assert msg.qos == 0

    @pytest.mark.asyncio
    async def test_publish_qos_1(self, connected_mock_mqtt_broker):
        """Test publishing with QoS 1 (at least once)."""
        broker = connected_mock_mqtt_broker

        await broker.publish(
            client_id="test-client",
            topic="sensors/temp",
            payload=b"85.5",
            qos=QoSLevel.AT_LEAST_ONCE
        )

        msg = broker.message_history[0]
        assert msg.qos == 1

    @pytest.mark.asyncio
    async def test_publish_qos_2(self, connected_mock_mqtt_broker):
        """Test publishing with QoS 2 (exactly once)."""
        broker = connected_mock_mqtt_broker

        await broker.publish(
            client_id="test-client",
            topic="sensors/temp",
            payload=b"85.5",
            qos=QoSLevel.EXACTLY_ONCE
        )

        msg = broker.message_history[0]
        assert msg.qos == 2

    @pytest.mark.asyncio
    async def test_publish_retained_message(self, connected_mock_mqtt_broker):
        """Test publishing a retained message."""
        broker = connected_mock_mqtt_broker

        await broker.publish(
            client_id="test-client",
            topic="status/system",
            payload=b"RUNNING",
            retain=True
        )

        # Check retained message is stored
        assert "status/system" in broker.retained_messages
        retained = broker.retained_messages["status/system"]
        assert retained.payload == b"RUNNING"
        assert retained.retain is True

    @pytest.mark.asyncio
    async def test_clear_retained_message(self, connected_mock_mqtt_broker):
        """Test clearing a retained message with empty payload."""
        broker = connected_mock_mqtt_broker

        # First, set a retained message
        await broker.publish(
            client_id="test-client",
            topic="status/system",
            payload=b"RUNNING",
            retain=True
        )

        assert "status/system" in broker.retained_messages

        # Clear with empty payload
        await broker.publish(
            client_id="test-client",
            topic="status/system",
            payload=b"",
            retain=True
        )

        assert "status/system" not in broker.retained_messages

    @pytest.mark.asyncio
    async def test_publish_when_not_connected(self, mock_mqtt_broker):
        """Test publishing fails when not connected."""
        broker = mock_mqtt_broker

        with pytest.raises(ConnectionError, match="not connected"):
            await broker.publish(
                client_id="disconnected-client",
                topic="test/topic",
                payload=b"test"
            )

    @pytest.mark.asyncio
    async def test_publish_json_payload(self, connected_mock_mqtt_broker):
        """Test publishing JSON payload."""
        broker = connected_mock_mqtt_broker

        payload = json.dumps({
            "temperature": 85.5,
            "pressure": 2.5,
            "timestamp": datetime.utcnow().isoformat()
        }).encode()

        await broker.publish(
            client_id="test-client",
            topic="process-heat/data",
            payload=payload
        )

        msg = broker.message_history[0]
        data = json.loads(msg.payload.decode())
        assert data["temperature"] == 85.5


# =============================================================================
# Test Class: MQTT Subscribe Tests
# =============================================================================


class TestMQTTSubscribe:
    """Test MQTT subscription functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_to_topic(self, connected_mock_mqtt_broker):
        """Test subscribing to a topic."""
        broker = connected_mock_mqtt_broker
        received_messages = []

        async def callback(msg):
            received_messages.append(msg)

        await broker.subscribe(
            client_id="test-client",
            topic="sensors/temperature",
            callback=callback
        )

        assert "sensors/temperature" in broker.subscriptions

    @pytest.mark.asyncio
    async def test_receive_message_after_subscribe(self, connected_mock_mqtt_broker):
        """Test receiving messages after subscribing."""
        broker = connected_mock_mqtt_broker
        received_messages = []

        async def callback(msg):
            received_messages.append(msg)

        await broker.subscribe("test-client", "sensors/temp", callback=callback)

        # Publish a message
        await broker.publish(
            client_id="test-client",
            topic="sensors/temp",
            payload=b"85.5"
        )

        assert len(received_messages) == 1
        assert received_messages[0].payload == b"85.5"

    @pytest.mark.asyncio
    async def test_subscribe_with_single_level_wildcard(self, connected_mock_mqtt_broker):
        """Test subscription with single-level wildcard (+)."""
        broker = connected_mock_mqtt_broker
        received_messages = []

        async def callback(msg):
            received_messages.append(msg)

        # Subscribe with + wildcard
        await broker.subscribe(
            client_id="test-client",
            topic="sensors/+/temperature",
            callback=callback
        )

        # Publish to matching topics
        await broker.publish("test-client", "sensors/boiler1/temperature", b"85")
        await broker.publish("test-client", "sensors/boiler2/temperature", b"90")

        assert len(received_messages) == 2

    @pytest.mark.asyncio
    async def test_subscribe_with_multi_level_wildcard(self, connected_mock_mqtt_broker):
        """Test subscription with multi-level wildcard (#)."""
        broker = connected_mock_mqtt_broker
        received_messages = []

        async def callback(msg):
            received_messages.append(msg)

        # Subscribe with # wildcard
        await broker.subscribe(
            client_id="test-client",
            topic="process-heat/#",
            callback=callback
        )

        # Publish to various matching topics
        await broker.publish("test-client", "process-heat/temperature", b"85")
        await broker.publish("test-client", "process-heat/sensors/pressure", b"2.5")
        await broker.publish("test-client", "process-heat/a/b/c", b"data")

        assert len(received_messages) == 3

    @pytest.mark.asyncio
    async def test_subscribe_with_qos(self, connected_mock_mqtt_broker):
        """Test subscription with specified QoS level."""
        broker = connected_mock_mqtt_broker
        received_messages = []

        async def callback(msg):
            received_messages.append(msg)

        await broker.subscribe(
            client_id="test-client",
            topic="high-priority/data",
            qos=QoSLevel.EXACTLY_ONCE,
            callback=callback
        )

        # Publish with QoS 2
        await broker.publish(
            client_id="test-client",
            topic="high-priority/data",
            payload=b"critical",
            qos=QoSLevel.EXACTLY_ONCE
        )

        assert len(received_messages) == 1
        assert received_messages[0].qos == 2

    @pytest.mark.asyncio
    async def test_qos_downgrade(self, connected_mock_mqtt_broker):
        """Test QoS is downgraded to minimum of pub and sub QoS."""
        broker = connected_mock_mqtt_broker
        received_messages = []

        async def callback(msg):
            received_messages.append(msg)

        # Subscribe with QoS 1
        await broker.subscribe(
            client_id="test-client",
            topic="qos-test",
            qos=QoSLevel.AT_LEAST_ONCE,
            callback=callback
        )

        # Publish with QoS 2
        await broker.publish(
            client_id="test-client",
            topic="qos-test",
            payload=b"test",
            qos=QoSLevel.EXACTLY_ONCE
        )

        # Effective QoS should be 1 (minimum)
        assert received_messages[0].qos == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, connected_mock_mqtt_broker):
        """Test unsubscribing from a topic."""
        broker = connected_mock_mqtt_broker
        received_messages = []

        async def callback(msg):
            received_messages.append(msg)

        await broker.subscribe("test-client", "test/topic", callback=callback)
        await broker.unsubscribe("test-client", "test/topic")

        # Publish after unsubscribe
        await broker.publish("test-client", "test/topic", b"test")

        # Should not receive message
        assert len(received_messages) == 0

    @pytest.mark.asyncio
    async def test_receive_retained_message_on_subscribe(self, connected_mock_mqtt_broker):
        """Test receiving retained message on new subscription."""
        broker = connected_mock_mqtt_broker
        received_messages = []

        # First, publish a retained message
        await broker.publish(
            client_id="test-client",
            topic="status/current",
            payload=b"RUNNING",
            retain=True
        )

        async def callback(msg):
            received_messages.append(msg)

        # Subscribe (should receive retained message)
        await broker.subscribe("test-client", "status/current", callback=callback)

        assert len(received_messages) == 1
        assert received_messages[0].retain is True
        assert received_messages[0].payload == b"RUNNING"


# =============================================================================
# Test Class: MQTT Last Will and Testament Tests
# =============================================================================


class TestMQTTLastWillTestament:
    """Test MQTT Last Will and Testament (LWT) functionality."""

    @pytest.mark.asyncio
    async def test_set_will_message(self, mock_mqtt_broker):
        """Test setting a will message on connect."""
        broker = mock_mqtt_broker

        await broker.connect(
            client_id="lwt-client",
            will_topic="client/status",
            will_message=b"OFFLINE",
            will_qos=1
        )

        assert "lwt-client" in broker.will_messages
        will = broker.will_messages["lwt-client"]
        assert will.topic == "client/status"
        assert will.payload == b"OFFLINE"
        assert will.qos == 1

    @pytest.mark.asyncio
    async def test_will_published_on_ungraceful_disconnect(self, mock_mqtt_broker):
        """Test will message is published on ungraceful disconnect."""
        broker = mock_mqtt_broker
        received_messages = []

        # Connect subscriber
        await broker.connect("subscriber")

        async def callback(msg):
            received_messages.append(msg)

        await broker.subscribe("subscriber", "client/status", callback=callback)

        # Connect client with will
        await broker.connect(
            client_id="monitored-client",
            will_topic="client/status",
            will_message=b"DISCONNECTED",
            will_qos=1
        )

        # Ungraceful disconnect
        await broker.disconnect("monitored-client", graceful=False)

        # Will message should be published
        assert len(received_messages) == 1
        assert received_messages[0].payload == b"DISCONNECTED"

    @pytest.mark.asyncio
    async def test_will_not_published_on_graceful_disconnect(self, mock_mqtt_broker):
        """Test will message is NOT published on graceful disconnect."""
        broker = mock_mqtt_broker
        received_messages = []

        # Connect subscriber
        await broker.connect("subscriber")

        async def callback(msg):
            received_messages.append(msg)

        await broker.subscribe("subscriber", "client/status", callback=callback)

        # Connect client with will
        await broker.connect(
            client_id="monitored-client",
            will_topic="client/status",
            will_message=b"DISCONNECTED",
            will_qos=1
        )

        # Graceful disconnect
        await broker.disconnect("monitored-client", graceful=True)

        # Will message should NOT be published
        assert len(received_messages) == 0

    @pytest.mark.asyncio
    async def test_will_with_retain_flag(self, mock_mqtt_broker):
        """Test will message with retain flag."""
        broker = mock_mqtt_broker

        await broker.connect(
            client_id="retain-will-client",
            will_topic="device/status",
            will_message=b"OFFLINE",
            will_qos=1,
            will_retain=True
        )

        will = broker.will_messages["retain-will-client"]
        assert will.retain is True


# =============================================================================
# Test Class: MQTT Reconnection Tests
# =============================================================================


class TestMQTTReconnection:
    """Test MQTT reconnection behavior."""

    @pytest.mark.asyncio
    async def test_broker_disconnect_detection(self, connected_mock_mqtt_broker):
        """Test detection of broker disconnect."""
        broker = connected_mock_mqtt_broker

        broker.simulate_disconnect()

        assert broker.running is False

    @pytest.mark.asyncio
    async def test_broker_reconnect(self, mock_mqtt_broker):
        """Test reconnecting after broker comes back online."""
        broker = mock_mqtt_broker

        await broker.connect("reconnect-client")
        broker.simulate_disconnect()
        broker.simulate_reconnect()

        # Should be able to connect again
        connected = await broker.connect("new-client")
        assert connected is True

    @pytest.mark.asyncio
    async def test_will_messages_triggered_on_broker_disconnect(self, mock_mqtt_broker):
        """Test all will messages are triggered when broker disconnects."""
        broker = mock_mqtt_broker
        received_messages = []

        # Connect multiple clients with will messages
        await broker.connect(
            "client-1",
            will_topic="clients/status",
            will_message=b"client-1-offline"
        )
        await broker.connect(
            "client-2",
            will_topic="clients/status",
            will_message=b"client-2-offline"
        )

        # Subscribe to will messages topic
        await broker.connect("monitor")

        async def callback(msg):
            received_messages.append(msg)

        await broker.subscribe("monitor", "clients/status", callback=callback)

        # Simulate broker disconnect (triggers will messages)
        broker.simulate_disconnect()

        # Wait for async tasks
        await asyncio.sleep(0.1)

        # Both will messages should be published
        assert len(received_messages) >= 2


# =============================================================================
# Test Class: MQTT Topic Matching Tests
# =============================================================================


class TestMQTTTopicMatching:
    """Test MQTT topic pattern matching."""

    @pytest.mark.asyncio
    async def test_exact_topic_match(self, connected_mock_mqtt_broker):
        """Test exact topic matching."""
        broker = connected_mock_mqtt_broker

        assert broker._topic_matches("sensor/temp", "sensor/temp") is True
        assert broker._topic_matches("sensor/temp", "sensor/pressure") is False

    @pytest.mark.asyncio
    async def test_single_level_wildcard_match(self, connected_mock_mqtt_broker):
        """Test single-level wildcard (+) matching."""
        broker = connected_mock_mqtt_broker

        assert broker._topic_matches("sensor/+/temp", "sensor/boiler/temp") is True
        assert broker._topic_matches("sensor/+/temp", "sensor/pump/temp") is True
        assert broker._topic_matches("sensor/+/temp", "sensor/boiler/pressure") is False
        assert broker._topic_matches("+/data", "device/data") is True

    @pytest.mark.asyncio
    async def test_multi_level_wildcard_match(self, connected_mock_mqtt_broker):
        """Test multi-level wildcard (#) matching."""
        broker = connected_mock_mqtt_broker

        assert broker._topic_matches("sensor/#", "sensor") is False
        assert broker._topic_matches("sensor/#", "sensor/temp") is True
        assert broker._topic_matches("sensor/#", "sensor/a/b/c") is True
        assert broker._topic_matches("#", "any/topic/here") is True

    @pytest.mark.asyncio
    async def test_combined_wildcards(self, connected_mock_mqtt_broker):
        """Test combined wildcards matching."""
        broker = connected_mock_mqtt_broker

        assert broker._topic_matches("+/sensor/#", "boiler/sensor/temp") is True
        assert broker._topic_matches("+/sensor/#", "pump/sensor/pressure/raw") is True


# =============================================================================
# Test Class: MQTT Message Count Tests
# =============================================================================


class TestMQTTMessageCount:
    """Test MQTT message counting functionality."""

    @pytest.mark.asyncio
    async def test_total_message_count(self, connected_mock_mqtt_broker):
        """Test total message count."""
        broker = connected_mock_mqtt_broker

        await broker.publish("test-client", "topic1", b"msg1")
        await broker.publish("test-client", "topic2", b"msg2")
        await broker.publish("test-client", "topic1", b"msg3")

        assert broker.get_message_count() == 3

    @pytest.mark.asyncio
    async def test_message_count_by_topic(self, connected_mock_mqtt_broker):
        """Test message count for specific topic."""
        broker = connected_mock_mqtt_broker

        await broker.publish("test-client", "sensors/temp", b"85")
        await broker.publish("test-client", "sensors/temp", b"86")
        await broker.publish("test-client", "sensors/pressure", b"2.5")

        assert broker.get_message_count("sensors/temp") == 2
        assert broker.get_message_count("sensors/pressure") == 1


# =============================================================================
# Test Class: MQTT Statistics Tests
# =============================================================================


class TestMQTTStatistics:
    """Test MQTT broker statistics."""

    @pytest.mark.asyncio
    async def test_broker_statistics(self, connected_mock_mqtt_broker):
        """Test broker statistics are accurate."""
        broker = connected_mock_mqtt_broker

        stats = broker.get_statistics()

        assert stats["running"] is True
        assert stats["connected_clients"] == 1
        assert stats["host"] == "localhost"
        assert stats["port"] == 1883

    @pytest.mark.asyncio
    async def test_statistics_update_on_subscribe(self, connected_mock_mqtt_broker):
        """Test statistics update on subscription."""
        broker = connected_mock_mqtt_broker

        await broker.subscribe("test-client", "topic1", callback=lambda m: None)
        await broker.subscribe("test-client", "topic2", callback=lambda m: None)

        stats = broker.get_statistics()
        assert stats["active_subscriptions"] == 2

    @pytest.mark.asyncio
    async def test_statistics_retained_messages(self, connected_mock_mqtt_broker):
        """Test statistics track retained messages."""
        broker = connected_mock_mqtt_broker

        await broker.publish("test-client", "status/1", b"on", retain=True)
        await broker.publish("test-client", "status/2", b"off", retain=True)

        stats = broker.get_statistics()
        assert stats["retained_messages"] == 2


# =============================================================================
# Test Class: MQTT Performance Tests
# =============================================================================


@pytest.mark.performance
class TestMQTTPerformance:
    """Performance tests for MQTT operations."""

    @pytest.mark.asyncio
    async def test_publish_throughput(
        self,
        connected_mock_mqtt_broker,
        throughput_calculator
    ):
        """Test publish throughput."""
        broker = connected_mock_mqtt_broker

        throughput_calculator.start()

        for i in range(100):
            payload = f'{{"value": {i}}}'.encode()
            await broker.publish("test-client", "perf/test", payload)
            throughput_calculator.record_message(len(payload))

        stats = throughput_calculator.get_throughput()
        assert stats["messages_per_sec"] > 100

    @pytest.mark.asyncio
    async def test_subscribe_callback_latency(
        self,
        connected_mock_mqtt_broker,
        performance_timer
    ):
        """Test subscribe callback latency."""
        broker = connected_mock_mqtt_broker
        received = []

        async def callback(msg):
            received.append(msg)

        await broker.subscribe("test-client", "latency/test", callback=callback)

        for _ in range(50):
            performance_timer.start()
            await broker.publish("test-client", "latency/test", b"test")
            performance_timer.stop()

        assert performance_timer.average_ms < 10

    @pytest.mark.asyncio
    async def test_high_volume_subscriptions(self, connected_mock_mqtt_broker):
        """Test handling many subscriptions."""
        broker = connected_mock_mqtt_broker
        callbacks = {}

        for i in range(100):
            async def cb(msg, idx=i):
                pass
            await broker.subscribe("test-client", f"topic/{i}", callback=cb)
            callbacks[i] = cb

        stats = broker.get_statistics()
        assert stats["active_subscriptions"] == 100


# =============================================================================
# Test Class: MQTT Process Heat Integration
# =============================================================================


class TestMQTTProcessHeatIntegration:
    """Integration tests for process heat data via MQTT."""

    @pytest.mark.asyncio
    async def test_publish_process_heat_data(
        self,
        connected_mock_mqtt_broker,
        sample_process_heat_data
    ):
        """Test publishing process heat data."""
        broker = connected_mock_mqtt_broker

        payload = json.dumps(sample_process_heat_data).encode()

        await broker.publish(
            client_id="test-client",
            topic="process-heat/data",
            payload=payload,
            qos=1
        )

        assert broker.get_message_count("process-heat/data") == 1

    @pytest.mark.asyncio
    async def test_subscribe_to_process_heat_topics(
        self,
        connected_mock_mqtt_broker,
        sample_mqtt_topics
    ):
        """Test subscribing to process heat topics."""
        broker = connected_mock_mqtt_broker
        received_data = {}

        async def temp_callback(msg):
            received_data["temperature"] = msg.payload.decode()

        async def pressure_callback(msg):
            received_data["pressure"] = msg.payload.decode()

        await broker.subscribe("test-client", sample_mqtt_topics["temperature"], temp_callback)
        await broker.subscribe("test-client", sample_mqtt_topics["pressure"], pressure_callback)

        # Publish data
        await broker.publish("test-client", sample_mqtt_topics["temperature"], b"85.5")
        await broker.publish("test-client", sample_mqtt_topics["pressure"], b"2.5")

        assert received_data["temperature"] == "85.5"
        assert received_data["pressure"] == "2.5"

    @pytest.mark.asyncio
    async def test_emissions_event_workflow(self, connected_mock_mqtt_broker):
        """Test emissions event workflow via MQTT."""
        broker = connected_mock_mqtt_broker
        emissions_events = []

        async def emissions_callback(msg):
            event = json.loads(msg.payload.decode())
            emissions_events.append(event)

        await broker.subscribe(
            "test-client",
            "process-heat/calculated/emissions",
            callback=emissions_callback
        )

        # Publish emissions event
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "boiler_1",
            "emission_type": "CO2",
            "value_kg": 121.14,
            "calculation_method": "mass_balance"
        }

        await broker.publish(
            client_id="test-client",
            topic="process-heat/calculated/emissions",
            payload=json.dumps(event).encode(),
            qos=2
        )

        assert len(emissions_events) == 1
        assert emissions_events[0]["value_kg"] == 121.14

    @pytest.mark.asyncio
    async def test_alarm_topic_subscription(self, connected_mock_mqtt_broker):
        """Test subscribing to alarm topics with wildcard."""
        broker = connected_mock_mqtt_broker
        alarms = []

        async def alarm_callback(msg):
            alarms.append({
                "topic": msg.topic,
                "payload": msg.payload.decode()
            })

        # Subscribe to all alarms
        await broker.subscribe(
            "test-client",
            "process-heat/alarms/#",
            callback=alarm_callback
        )

        # Publish various alarms
        await broker.publish("test-client", "process-heat/alarms/high-temp", b"WARNING")
        await broker.publish("test-client", "process-heat/alarms/low-pressure", b"CRITICAL")

        assert len(alarms) == 2

    @pytest.mark.asyncio
    async def test_status_retained_message(self, connected_mock_mqtt_broker):
        """Test status retained message for system state."""
        broker = connected_mock_mqtt_broker

        # Publish retained status
        await broker.publish(
            client_id="test-client",
            topic="process-heat/status",
            payload=b'{"state": "RUNNING", "uptime_hours": 125}',
            retain=True
        )

        # New subscriber should receive status immediately
        received_status = []

        async def status_callback(msg):
            received_status.append(json.loads(msg.payload.decode()))

        await broker.subscribe("test-client", "process-heat/status", callback=status_callback)

        assert len(received_status) == 1
        assert received_status[0]["state"] == "RUNNING"

    @pytest.mark.asyncio
    async def test_command_response_workflow(self, connected_mock_mqtt_broker):
        """Test command-response workflow for process control."""
        broker = connected_mock_mqtt_broker
        commands = []
        responses = []

        async def command_callback(msg):
            commands.append(json.loads(msg.payload.decode()))
            # Simulate response
            response = {"status": "OK", "command_id": commands[-1]["command_id"]}
            await broker.publish(
                "test-client",
                "process-heat/response",
                json.dumps(response).encode()
            )

        async def response_callback(msg):
            responses.append(json.loads(msg.payload.decode()))

        await broker.subscribe("test-client", "process-heat/commands/+", command_callback)
        await broker.subscribe("test-client", "process-heat/response", response_callback)

        # Send command
        command = {
            "command_id": "cmd_001",
            "action": "set_temperature",
            "value": 90.0
        }

        await broker.publish(
            client_id="test-client",
            topic="process-heat/commands/temperature",
            payload=json.dumps(command).encode()
        )

        assert len(commands) == 1
        assert len(responses) == 1
        assert responses[0]["command_id"] == "cmd_001"
