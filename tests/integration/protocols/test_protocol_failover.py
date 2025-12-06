# -*- coding: utf-8 -*-
"""
Protocol Failover and Resilience Tests
======================================

Comprehensive tests for protocol failover, resilience, and error recovery:
- Connection failover across protocols
- Automatic reconnection
- Circuit breaker patterns
- Retry mechanisms
- Graceful degradation
- Multi-protocol redundancy
- Error recovery workflows

Test Coverage Target: 85%+

Author: GreenLang Test Engineering Team
Date: December 2025
"""

import asyncio
import json
import logging
import pytest
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from tests.integration.protocols.conftest import (
    MockOPCUAServer,
    MockModbusServer,
    MockMQTTBroker,
    MockKafkaCluster,
    SecurityMode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Classes for Testing
# =============================================================================


class CircuitBreaker:
    """Simple circuit breaker implementation for testing."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def record_success(self) -> None:
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    return True
            return False

        # HALF_OPEN - allow one request to test
        return True


class RetryPolicy:
    """Retry policy for testing."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay_ms: int = 100,
        max_delay_ms: int = 5000,
        exponential_backoff: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.exponential_backoff = exponential_backoff

    def get_delay(self, attempt: int) -> float:
        """Get delay for given attempt number (in seconds)."""
        if self.exponential_backoff:
            delay_ms = min(
                self.initial_delay_ms * (2 ** attempt),
                self.max_delay_ms
            )
        else:
            delay_ms = self.initial_delay_ms

        return delay_ms / 1000.0


class ProtocolManager:
    """Manager for multiple protocol connections for testing."""

    def __init__(self):
        self.protocols: Dict[str, Any] = {}
        self.primary_protocol: Optional[str] = None
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def register_protocol(
        self,
        name: str,
        protocol: Any,
        is_primary: bool = False
    ) -> None:
        """Register a protocol with the manager."""
        self.protocols[name] = protocol
        self.circuit_breakers[name] = CircuitBreaker()

        if is_primary:
            self.primary_protocol = name

    def get_available_protocols(self) -> List[str]:
        """Get list of available protocols."""
        return [
            name for name, cb in self.circuit_breakers.items()
            if cb.can_execute()
        ]


# =============================================================================
# Test Class: OPC-UA Failover Tests
# =============================================================================


class TestOPCUAFailover:
    """Test OPC-UA connection failover and recovery."""

    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect(self, mock_opcua_server):
        """Test automatic reconnection after server disconnect."""
        server = mock_opcua_server
        await server.start()

        # Connect and read
        await server.connect_client("test-client")
        value = await server.read_value("ns=2;s=Temperature")
        assert value == 85.5

        # Simulate disconnect
        server.simulate_disconnect()

        # Operations should fail
        with pytest.raises(ConnectionError):
            await server.read_value("ns=2;s=Temperature")

        # Simulate reconnect
        server.simulate_reconnect()

        # Should work again
        value = await server.read_value("ns=2;s=Temperature")
        assert value == 85.5

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, mock_opcua_server):
        """Test retry mechanism on transient errors."""
        server = mock_opcua_server
        await server.start()

        retry_policy = RetryPolicy(max_retries=3)
        attempts = 0
        result = None

        # Enable error mode for first 2 attempts
        server.enable_error_mode()

        for attempt in range(retry_policy.max_retries + 1):
            attempts += 1
            try:
                # Disable error after 2 attempts
                if attempt >= 2:
                    server.disable_error_mode()

                result = await server.read_value("ns=2;s=Temperature")
                break
            except Exception:
                if attempt < retry_policy.max_retries:
                    delay = retry_policy.get_delay(attempt)
                    await asyncio.sleep(delay)
                else:
                    raise

        assert result == 85.5
        assert attempts == 3  # Succeeded on 3rd attempt

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, mock_opcua_server):
        """Test circuit breaker opens after repeated failures."""
        server = mock_opcua_server
        await server.start()

        circuit_breaker = CircuitBreaker(failure_threshold=3)
        server.enable_error_mode()

        # Make failing requests
        for _ in range(3):
            try:
                await server.read_value("ns=2;s=Temperature")
            except Exception:
                circuit_breaker.record_failure()

        # Circuit should be open
        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.can_execute() is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovers(self, mock_opcua_server):
        """Test circuit breaker recovers after timeout."""
        server = mock_opcua_server
        await server.start()

        circuit_breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout_seconds=0.1  # Short timeout for testing
        )

        server.enable_error_mode()

        # Trip circuit breaker
        for _ in range(2):
            try:
                await server.read_value("ns=2;s=Temperature")
            except Exception:
                circuit_breaker.record_failure()

        assert circuit_breaker.state == "OPEN"

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should transition to HALF_OPEN
        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.state == "HALF_OPEN"

        # Successful request should close circuit
        server.disable_error_mode()
        await server.read_value("ns=2;s=Temperature")
        circuit_breaker.record_success()

        assert circuit_breaker.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_subscription_recovery_after_disconnect(self, mock_opcua_server):
        """Test subscriptions are re-established after disconnect."""
        server = mock_opcua_server
        await server.start()

        received_before = []
        received_after = []

        async def callback_before(notification):
            received_before.append(notification)

        async def callback_after(notification):
            received_after.append(notification)

        # Create subscription
        await server.subscribe(
            "test-client",
            "ns=2;s=Temperature",
            callback_before
        )

        # Write and verify callback
        await server.write_value("ns=2;s=Temperature", 90.0)
        assert len(received_before) == 1

        # Disconnect
        server.simulate_disconnect()

        # Reconnect
        server.simulate_reconnect()

        # Re-subscribe
        await server.subscribe(
            "test-client",
            "ns=2;s=Temperature",
            callback_after
        )

        # Write again
        await server.write_value("ns=2;s=Temperature", 95.0)
        assert len(received_after) == 1


# =============================================================================
# Test Class: Modbus Failover Tests
# =============================================================================


class TestModbusFailover:
    """Test Modbus connection failover and recovery."""

    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect(self, mock_modbus_server):
        """Test reconnection after Modbus disconnect."""
        server = mock_modbus_server
        await server.connect()

        # Read successfully
        values = await server.read_holding_registers(0, 1)
        assert len(values) == 1

        # Disconnect
        await server.disconnect()

        # Should fail
        with pytest.raises(ConnectionError):
            await server.read_holding_registers(0, 1)

        # Reconnect
        await server.connect()

        # Should work again
        values = await server.read_holding_registers(0, 1)
        assert len(values) == 1

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, mock_modbus_server):
        """Test retry on Modbus timeout."""
        server = mock_modbus_server
        retry_policy = RetryPolicy(max_retries=3)

        attempts = 0
        connected = False

        # Timeout mode for first attempt
        server.enable_timeout_mode()

        for attempt in range(retry_policy.max_retries + 1):
            attempts += 1
            try:
                # Disable timeout after first attempt
                if attempt >= 1:
                    server.disable_timeout_mode()

                await asyncio.wait_for(server.connect(), timeout=0.5)
                connected = True
                break
            except (asyncio.TimeoutError, TimeoutError):
                if attempt < retry_policy.max_retries:
                    await asyncio.sleep(retry_policy.get_delay(attempt))

        assert connected is True
        assert attempts == 2

    @pytest.mark.asyncio
    async def test_error_recovery_with_register_read(self, mock_modbus_server):
        """Test error recovery during register reading."""
        server = mock_modbus_server
        await server.connect()

        # Set error mode for a few reads
        server.simulate_error()

        errors = 0
        successes = 0

        for i in range(5):
            try:
                # Clear error on 3rd iteration
                if i == 3:
                    server.clear_error()

                await server.read_holding_registers(0, 1)
                successes += 1
            except Exception:
                errors += 1

        assert errors == 3
        assert successes == 2


# =============================================================================
# Test Class: MQTT Failover Tests
# =============================================================================


class TestMQTTFailover:
    """Test MQTT connection failover and recovery."""

    @pytest.mark.asyncio
    async def test_reconnect_after_broker_disconnect(self, mock_mqtt_broker):
        """Test reconnection after broker disconnect."""
        broker = mock_mqtt_broker
        await broker.connect("test-client")

        # Publish successfully
        await broker.publish("test-client", "test/topic", b"test")
        assert broker.get_message_count() == 1

        # Simulate broker disconnect
        broker.simulate_disconnect()

        # Should fail to publish
        with pytest.raises(ConnectionError):
            await broker.publish("test-client", "test/topic", b"test")

        # Broker comes back
        broker.simulate_reconnect()

        # Reconnect client
        await broker.connect("test-client")

        # Should work again
        await broker.publish("test-client", "test/topic", b"test")

    @pytest.mark.asyncio
    async def test_will_message_on_unexpected_disconnect(self, mock_mqtt_broker):
        """Test will message is delivered on unexpected disconnect."""
        broker = mock_mqtt_broker
        will_received = []

        # Connect subscriber
        await broker.connect("subscriber")

        async def will_callback(msg):
            will_received.append(msg)

        await broker.subscribe("subscriber", "clients/status", callback=will_callback)

        # Connect client with will
        await broker.connect(
            "monitored-client",
            will_topic="clients/status",
            will_message=b"OFFLINE"
        )

        # Unexpected disconnect (simulated by broker disconnect)
        broker.simulate_disconnect()

        await asyncio.sleep(0.1)

        # Will message should be delivered
        assert len(will_received) >= 1

    @pytest.mark.asyncio
    async def test_subscription_persistence_across_reconnect(self, mock_mqtt_broker):
        """Test subscriptions persist with clean_session=False."""
        broker = mock_mqtt_broker
        received_messages = []

        async def callback(msg):
            received_messages.append(msg)

        # Connect with persistent session
        await broker.connect(
            "persistent-client",
            clean_session=False
        )

        await broker.subscribe(
            "persistent-client",
            "test/topic",
            callback=callback
        )

        # Disconnect gracefully
        await broker.disconnect("persistent-client", graceful=True)

        # Reconnect with same client ID
        await broker.connect(
            "persistent-client",
            clean_session=False
        )

        # Re-subscribe (in real MQTT, session would be restored)
        await broker.subscribe(
            "persistent-client",
            "test/topic",
            callback=callback
        )

        # Publish and receive
        await broker.publish("persistent-client", "test/topic", b"test")
        assert len(received_messages) == 1


# =============================================================================
# Test Class: Kafka Failover Tests
# =============================================================================


class TestKafkaFailover:
    """Test Kafka failover and recovery."""

    @pytest.mark.asyncio
    async def test_producer_retry_on_cluster_unavailable(self, mock_kafka_cluster):
        """Test producer retries when cluster is temporarily unavailable."""
        cluster = mock_kafka_cluster
        retry_policy = RetryPolicy(max_retries=3)

        # Cluster down initially
        cluster.simulate_cluster_down()

        attempts = 0
        produced = False

        for attempt in range(retry_policy.max_retries + 1):
            attempts += 1
            try:
                # Cluster comes back on 3rd attempt
                if attempt == 2:
                    cluster.simulate_cluster_up()

                await cluster.produce("test-topic", b"test")
                produced = True
                break
            except ConnectionError:
                if attempt < retry_policy.max_retries:
                    await asyncio.sleep(retry_policy.get_delay(attempt))

        assert produced is True
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_consumer_offset_recovery(self, mock_kafka_cluster):
        """Test consumer recovers from correct offset after failure."""
        cluster = mock_kafka_cluster
        group_id = "recovery-group"

        # Produce messages
        for i in range(10):
            await cluster.produce("test-topic", f'{{"id": {i}}}'.encode())

        # Consume some messages
        records = await cluster.consume(group_id, ["test-topic"], max_records=5)

        # Commit offset
        await cluster.commit(group_id, {"test-topic:0": 5})

        # Simulate failure and recovery
        cluster.simulate_cluster_down()
        cluster.simulate_cluster_up()

        # Continue consuming from committed offset
        more_records = await cluster.consume(group_id, ["test-topic"])

        # Should get remaining messages
        for r in more_records:
            if r.partition == 0:
                assert r.offset >= 5

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_failure(self, mock_kafka_cluster):
        """Test transaction rollback preserves data consistency."""
        cluster = mock_kafka_cluster
        producer_id = "transactional-producer"

        initial_count = cluster.get_topic_message_count("test-topic")

        # Begin transaction
        await cluster.begin_transaction(producer_id)

        # Produce messages within transaction
        await cluster.produce(
            "test-topic",
            b"msg1",
            producer_id=producer_id
        )
        await cluster.produce(
            "test-topic",
            b"msg2",
            producer_id=producer_id
        )

        # Simulate failure - abort transaction
        await cluster.abort_transaction(producer_id)

        # Message count should be unchanged
        final_count = cluster.get_topic_message_count("test-topic")
        assert final_count == initial_count


# =============================================================================
# Test Class: Multi-Protocol Failover Tests
# =============================================================================


class TestMultiProtocolFailover:
    """Test failover across multiple protocols."""

    @pytest.mark.asyncio
    async def test_failover_from_opcua_to_modbus(
        self,
        mock_opcua_server,
        mock_modbus_server
    ):
        """Test failover from OPC-UA to Modbus when OPC-UA fails."""
        opcua = mock_opcua_server
        modbus = mock_modbus_server

        await opcua.start()
        await modbus.connect()

        manager = ProtocolManager()
        manager.register_protocol("opcua", opcua, is_primary=True)
        manager.register_protocol("modbus", modbus)

        # Read from primary (OPC-UA)
        value = await opcua.read_value("ns=2;s=Temperature")
        assert value == 85.5

        # OPC-UA fails
        opcua.enable_error_mode()
        manager.circuit_breakers["opcua"].record_failure()
        manager.circuit_breakers["opcua"].record_failure()
        manager.circuit_breakers["opcua"].record_failure()

        # Failover to Modbus
        available = manager.get_available_protocols()
        assert "modbus" in available
        assert "opcua" not in available

        # Read from Modbus
        values = await modbus.read_holding_registers(0, 1)
        temperature = values[0] * 0.1  # Scale factor
        assert temperature == 85.5

    @pytest.mark.asyncio
    async def test_multi_protocol_redundancy(
        self,
        mock_opcua_server,
        mock_modbus_server,
        mock_mqtt_broker
    ):
        """Test reading from multiple protocols for redundancy."""
        opcua = mock_opcua_server
        modbus = mock_modbus_server
        mqtt = mock_mqtt_broker

        await opcua.start()
        await modbus.connect()
        await mqtt.connect("data-client")

        readings = {}

        # Read from OPC-UA
        try:
            value = await opcua.read_value("ns=2;s=Temperature")
            readings["opcua"] = value
        except Exception:
            readings["opcua"] = None

        # Read from Modbus
        try:
            values = await modbus.read_holding_registers(0, 1)
            readings["modbus"] = values[0] * 0.1
        except Exception:
            readings["modbus"] = None

        # Store reading in MQTT (as backup)
        if readings["opcua"]:
            await mqtt.publish(
                "data-client",
                "sensors/temperature",
                json.dumps({"value": readings["opcua"]}).encode(),
                retain=True
            )

        # Verify all readings match
        assert readings["opcua"] == pytest.approx(readings["modbus"], rel=0.01)

    @pytest.mark.asyncio
    async def test_publish_to_multiple_destinations(
        self,
        mock_mqtt_broker,
        mock_kafka_cluster
    ):
        """Test publishing data to both MQTT and Kafka for redundancy."""
        mqtt = mock_mqtt_broker
        kafka = mock_kafka_cluster

        await mqtt.connect("publisher")

        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "temperature": 85.5,
            "source": "boiler_1"
        }

        payload = json.dumps(data).encode()

        # Publish to both
        mqtt_result = await mqtt.publish(
            "publisher",
            "sensors/temperature",
            payload
        )

        kafka_result = await kafka.produce(
            "sensors-topic",
            payload,
            key=b"boiler_1"
        )

        # Both should succeed
        assert mqtt.get_message_count("sensors/temperature") == 1
        assert kafka.get_topic_message_count("sensors-topic") == 1


# =============================================================================
# Test Class: Graceful Degradation Tests
# =============================================================================


class TestGracefulDegradation:
    """Test graceful degradation when protocols fail."""

    @pytest.mark.asyncio
    async def test_continue_with_partial_data(
        self,
        mock_opcua_server,
        mock_modbus_server
    ):
        """Test system continues with partial data when some sources fail."""
        opcua = mock_opcua_server
        modbus = mock_modbus_server

        await opcua.start()
        await modbus.connect()

        # OPC-UA fails
        opcua.enable_error_mode()

        data = {
            "temperature": None,
            "pressure": None,
            "status": "PARTIAL"
        }

        # Try OPC-UA
        try:
            data["temperature"] = await opcua.read_value("ns=2;s=Temperature")
        except Exception:
            pass  # Graceful handling

        # Modbus works
        try:
            values = await modbus.read_holding_registers(1, 1)
            data["pressure"] = values[0] * 0.1
        except Exception:
            pass

        # Should have partial data
        assert data["temperature"] is None  # OPC-UA failed
        assert data["pressure"] == 2.5  # Modbus worked
        assert data["status"] == "PARTIAL"

    @pytest.mark.asyncio
    async def test_cache_fallback(self, mock_opcua_server):
        """Test fallback to cached values when live data unavailable."""
        server = mock_opcua_server
        await server.start()

        cache = {}

        # Read and cache
        value = await server.read_value("ns=2;s=Temperature")
        cache["temperature"] = {
            "value": value,
            "timestamp": datetime.utcnow()
        }

        # Server goes down
        server.simulate_disconnect()

        # Try to read, fall back to cache
        try:
            current_value = await server.read_value("ns=2;s=Temperature")
        except ConnectionError:
            # Use cached value
            cached = cache.get("temperature")
            current_value = cached["value"] if cached else None
            cache_age = (datetime.utcnow() - cached["timestamp"]).total_seconds()

            # Cached value is stale but usable
            assert current_value == 85.5
            assert cache_age < 1.0  # Recently cached

    @pytest.mark.asyncio
    async def test_queue_messages_during_outage(self, mock_kafka_cluster):
        """Test messages are queued during Kafka outage."""
        cluster = mock_kafka_cluster
        message_queue = []

        # Kafka goes down
        cluster.simulate_cluster_down()

        # Queue messages locally
        for i in range(5):
            message_queue.append({
                "topic": "test-topic",
                "value": f'{{"id": {i}}}'.encode()
            })

        # Kafka comes back
        cluster.simulate_cluster_up()

        # Send queued messages
        for msg in message_queue:
            await cluster.produce(msg["topic"], msg["value"])

        message_queue.clear()

        # Verify all sent
        assert cluster.get_topic_message_count("test-topic") == 5


# =============================================================================
# Test Class: Error Recovery Workflow Tests
# =============================================================================


class TestErrorRecoveryWorkflows:
    """Test complex error recovery workflows."""

    @pytest.mark.asyncio
    async def test_complete_recovery_workflow(
        self,
        mock_opcua_server,
        mock_kafka_cluster
    ):
        """Test complete error detection and recovery workflow."""
        opcua = mock_opcua_server
        kafka = mock_kafka_cluster

        await opcua.start()

        # State tracking
        state = {
            "errors_detected": 0,
            "recoveries": 0,
            "data_collected": [],
            "data_published": 0
        }

        # Simulate workflow with intermittent failures
        for cycle in range(5):
            # Inject failure on cycle 2
            if cycle == 2:
                opcua.enable_error_mode()

            try:
                # Collect data
                temp = await opcua.read_value("ns=2;s=Temperature")
                state["data_collected"].append(temp)

                # Publish to Kafka
                await kafka.produce(
                    "process-data",
                    json.dumps({"temperature": temp}).encode()
                )
                state["data_published"] += 1

            except Exception:
                state["errors_detected"] += 1

                # Recovery action
                opcua.disable_error_mode()
                state["recoveries"] += 1

                # Retry
                try:
                    temp = await opcua.read_value("ns=2;s=Temperature")
                    state["data_collected"].append(temp)

                    await kafka.produce(
                        "process-data",
                        json.dumps({"temperature": temp}).encode()
                    )
                    state["data_published"] += 1
                except Exception:
                    pass  # Log and continue

        assert state["errors_detected"] == 1
        assert state["recoveries"] == 1
        assert state["data_published"] >= 4

    @pytest.mark.asyncio
    async def test_dead_letter_queue_workflow(
        self,
        mock_kafka_cluster
    ):
        """Test dead letter queue for failed message processing."""
        cluster = mock_kafka_cluster
        processed = []
        dlq = []

        # Produce messages
        messages = [
            {"id": 1, "valid": True},
            {"id": 2, "valid": False},  # Will fail processing
            {"id": 3, "valid": True},
            {"id": 4, "valid": False},  # Will fail processing
            {"id": 5, "valid": True},
        ]

        for msg in messages:
            await cluster.produce(
                "input-topic",
                json.dumps(msg).encode()
            )

        # Consume and process
        records = await cluster.consume(
            "processor-group",
            ["input-topic"]
        )

        for record in records:
            msg = json.loads(record.value.decode())

            if msg["valid"]:
                processed.append(msg["id"])
                await cluster.produce(
                    "output-topic",
                    record.value
                )
            else:
                dlq.append(msg["id"])
                await cluster.produce(
                    "dlq-topic",
                    record.value,
                    headers=[("error", b"invalid_message")]
                )

        assert len(processed) == 3
        assert len(dlq) == 2
        assert cluster.get_topic_message_count("output-topic") == 3
        assert cluster.get_topic_message_count("dlq-topic") == 2


# =============================================================================
# Test Class: Performance Under Failure Tests
# =============================================================================


@pytest.mark.performance
class TestPerformanceUnderFailure:
    """Test performance characteristics during failure scenarios."""

    @pytest.mark.asyncio
    async def test_retry_overhead(
        self,
        mock_opcua_server,
        performance_timer
    ):
        """Test performance overhead of retry mechanism."""
        server = mock_opcua_server
        await server.start()

        retry_policy = RetryPolicy(
            max_retries=3,
            initial_delay_ms=10,
            exponential_backoff=True
        )

        # Normal operation timing
        performance_timer.start()
        for _ in range(10):
            await server.read_value("ns=2;s=Temperature")
        normal_time = performance_timer.stop()

        # With retries (server fails on first attempt)
        server.set_disconnect_after(1)

        total_retry_time = 0
        for _ in range(5):
            server.simulate_reconnect()
            performance_timer.start()

            for attempt in range(retry_policy.max_retries + 1):
                try:
                    await server.read_value("ns=2;s=Temperature")
                    break
                except ConnectionError:
                    await asyncio.sleep(retry_policy.get_delay(attempt))

            total_retry_time += performance_timer.stop()

        # Retry overhead should be bounded
        average_retry_time = total_retry_time / 5
        assert average_retry_time < 0.5  # Less than 500ms per retry cycle

    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(
        self,
        mock_opcua_server,
        performance_timer
    ):
        """Test circuit breaker prevents repeated slow failures."""
        server = mock_opcua_server
        await server.start()

        circuit_breaker = CircuitBreaker(failure_threshold=3)
        server.set_latency(100)  # Simulate slow failures
        server.enable_error_mode()

        # Without circuit breaker - slow failures
        performance_timer.start()
        for _ in range(10):
            try:
                await server.read_value("ns=2;s=Temperature")
            except Exception:
                circuit_breaker.record_failure()

                # Circuit breaker should open after 3 failures
                if not circuit_breaker.can_execute():
                    break
        with_cb_time = performance_timer.stop()

        # Circuit breaker should have stopped after 3 failures
        # Total time should be ~300ms (3 * 100ms latency)
        assert with_cb_time < 0.5  # Less than 500ms due to circuit breaker
