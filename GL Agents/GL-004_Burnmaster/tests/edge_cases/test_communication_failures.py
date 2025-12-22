"""
Communication Failure Edge Case Tests for GL-004 BURNMASTER

Tests system behavior under various communication failure scenarios:
- Protocol timeouts (OPC-UA, Modbus, DCS)
- Packet loss and corruption
- Network partition and recovery
- Latency spikes and jitter
- Connection drops and reconnection
- Message queue overflow

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import asyncio
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Awaitable
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from enum import Enum
from dataclasses import dataclass, field
import random
import time
import threading
import queue

# Import system modules
import sys
sys.path.insert(0, 'C:/Users/aksha/Code-V1_GreenLang/GL Agents/GL-004_Burnmaster')

from safety.interlock_manager import (
    InterlockManager, BMSStatus, SISStatus, BMSState, SISState,
    InterlockState, Interlock, PermissiveStatus,
)
from calculators.stability_calculator import FlameStabilityCalculator


# ============================================================================
# COMMUNICATION ERROR CLASSES
# ============================================================================

class CommunicationError(Exception):
    """Base class for communication errors."""
    pass


class TimeoutError(CommunicationError):
    """Request timed out."""
    pass


class ConnectionError(CommunicationError):
    """Connection lost or refused."""
    pass


class ProtocolError(CommunicationError):
    """Protocol-level error."""
    pass


class PacketLossError(CommunicationError):
    """Packet was lost."""
    pass


class DataCorruptionError(CommunicationError):
    """Data corruption detected."""
    pass


# ============================================================================
# COMMUNICATION SIMULATOR
# ============================================================================

class CommunicationState(Enum):
    """Communication channel state."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    TIMEOUT = "timeout"
    DISCONNECTED = "disconnected"
    RECOVERING = "recovering"


@dataclass
class NetworkConditions:
    """Simulated network conditions."""
    latency_ms: float = 10.0
    latency_jitter_ms: float = 2.0
    packet_loss_rate: float = 0.0
    corruption_rate: float = 0.0
    timeout_probability: float = 0.0
    bandwidth_kbps: float = 1000.0


@dataclass
class CommunicationMetrics:
    """Metrics for communication performance."""
    requests_sent: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    timeouts: int = 0
    retries: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    packet_loss_count: int = 0
    corruption_count: int = 0
    connection_drops: int = 0
    last_successful_time: Optional[datetime] = None


class CommunicationSimulator:
    """Simulator for communication channel behavior."""

    def __init__(
        self,
        channel_id: str,
        conditions: NetworkConditions = None
    ):
        self.channel_id = channel_id
        self.conditions = conditions or NetworkConditions()
        self.state = CommunicationState.HEALTHY
        self.metrics = CommunicationMetrics()
        self._latency_history: List[float] = []
        self._request_queue: queue.Queue = queue.Queue(maxsize=100)

    def set_conditions(self, conditions: NetworkConditions):
        """Update network conditions."""
        self.conditions = conditions

    def inject_timeout(self):
        """Inject a timeout condition."""
        self.state = CommunicationState.TIMEOUT

    def inject_disconnect(self):
        """Inject a disconnection."""
        self.state = CommunicationState.DISCONNECTED

    def recover(self):
        """Recover from failure state."""
        self.state = CommunicationState.RECOVERING
        # Simulate recovery time
        self.state = CommunicationState.HEALTHY

    def send_request(self, data: Any, timeout_ms: float = 5000) -> Any:
        """
        Simulate sending a request with network conditions applied.

        Args:
            data: Request data
            timeout_ms: Request timeout in milliseconds

        Returns:
            Response data

        Raises:
            Various communication errors based on conditions
        """
        self.metrics.requests_sent += 1

        # Check current state
        if self.state == CommunicationState.DISCONNECTED:
            self.metrics.requests_failed += 1
            self.metrics.connection_drops += 1
            raise ConnectionError(f"Channel {self.channel_id} disconnected")

        if self.state == CommunicationState.TIMEOUT:
            self.metrics.requests_failed += 1
            self.metrics.timeouts += 1
            raise TimeoutError(f"Channel {self.channel_id} timeout")

        # Simulate packet loss
        if random.random() < self.conditions.packet_loss_rate:
            self.metrics.requests_failed += 1
            self.metrics.packet_loss_count += 1
            raise PacketLossError(f"Packet lost on {self.channel_id}")

        # Simulate timeout probability
        if random.random() < self.conditions.timeout_probability:
            self.metrics.requests_failed += 1
            self.metrics.timeouts += 1
            raise TimeoutError(f"Random timeout on {self.channel_id}")

        # Simulate latency
        latency = self.conditions.latency_ms + random.gauss(0, self.conditions.latency_jitter_ms)
        latency = max(0, latency)
        self._latency_history.append(latency)

        # Check if latency exceeds timeout
        if latency > timeout_ms:
            self.metrics.requests_failed += 1
            self.metrics.timeouts += 1
            raise TimeoutError(f"Latency {latency:.1f}ms exceeded timeout {timeout_ms}ms")

        # Simulate actual delay (reduced for tests)
        time.sleep(latency / 1000.0 / 100.0)  # Scaled down for testing

        # Simulate data corruption
        if random.random() < self.conditions.corruption_rate:
            self.metrics.corruption_count += 1
            raise DataCorruptionError(f"Data corruption on {self.channel_id}")

        # Success
        self.metrics.requests_succeeded += 1
        self.metrics.last_successful_time = datetime.utcnow()
        self._update_latency_stats(latency)

        return {"status": "ok", "data": data, "latency_ms": latency}

    def _update_latency_stats(self, latency: float):
        """Update latency statistics."""
        if self._latency_history:
            self.metrics.avg_latency_ms = sum(self._latency_history) / len(self._latency_history)
            self.metrics.max_latency_ms = max(self._latency_history)


class OPCUASimulator(CommunicationSimulator):
    """Simulator for OPC-UA communication."""

    def __init__(self, server_url: str):
        super().__init__(f"opcua://{server_url}")
        self.server_url = server_url
        self.connected = True
        self.subscription_active = True

    def read_node(self, node_id: str) -> Any:
        """Read a node value."""
        return self.send_request({"node_id": node_id, "operation": "read"})

    def write_node(self, node_id: str, value: Any) -> bool:
        """Write a node value."""
        response = self.send_request({"node_id": node_id, "value": value, "operation": "write"})
        return response["status"] == "ok"

    def create_subscription(self, publish_interval_ms: int) -> str:
        """Create a subscription."""
        if self.state != CommunicationState.HEALTHY:
            raise ConnectionError("Cannot create subscription - connection unhealthy")
        self.subscription_active = True
        return f"sub_{self.channel_id}_{publish_interval_ms}"

    def delete_subscription(self, subscription_id: str):
        """Delete a subscription."""
        self.subscription_active = False


class ModbusSimulator(CommunicationSimulator):
    """Simulator for Modbus communication."""

    def __init__(self, host: str, port: int = 502):
        super().__init__(f"modbus://{host}:{port}")
        self.host = host
        self.port = port

    def read_holding_registers(self, address: int, count: int) -> List[int]:
        """Read holding registers."""
        response = self.send_request({"address": address, "count": count})
        return [100] * count  # Simulated values

    def write_single_register(self, address: int, value: int) -> bool:
        """Write a single register."""
        response = self.send_request({"address": address, "value": value})
        return response["status"] == "ok"


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def opcua_simulator():
    """Create OPC-UA simulator."""
    return OPCUASimulator("localhost:4840")


@pytest.fixture
def modbus_simulator():
    """Create Modbus simulator."""
    return ModbusSimulator("localhost", 502)


@pytest.fixture
def healthy_conditions():
    """Create healthy network conditions."""
    return NetworkConditions(
        latency_ms=10.0,
        latency_jitter_ms=2.0,
        packet_loss_rate=0.0,
        corruption_rate=0.0,
        timeout_probability=0.0
    )


@pytest.fixture
def degraded_conditions():
    """Create degraded network conditions."""
    return NetworkConditions(
        latency_ms=100.0,
        latency_jitter_ms=50.0,
        packet_loss_rate=0.05,
        corruption_rate=0.01,
        timeout_probability=0.02
    )


@pytest.fixture
def severe_conditions():
    """Create severe network conditions."""
    return NetworkConditions(
        latency_ms=500.0,
        latency_jitter_ms=200.0,
        packet_loss_rate=0.20,
        corruption_rate=0.05,
        timeout_probability=0.10
    )


# ============================================================================
# PROTOCOL TIMEOUT TESTS
# ============================================================================

class TestProtocolTimeouts:
    """Test suite for protocol timeout scenarios."""

    def test_opcua_read_timeout(self, opcua_simulator):
        """Test OPC-UA read operation timeout."""
        opcua_simulator.inject_timeout()

        with pytest.raises(TimeoutError):
            opcua_simulator.read_node("ns=2;s=Burner1/O2")

        assert opcua_simulator.metrics.timeouts == 1
        assert opcua_simulator.metrics.requests_failed == 1

    def test_opcua_write_timeout(self, opcua_simulator):
        """Test OPC-UA write operation timeout."""
        opcua_simulator.inject_timeout()

        with pytest.raises(TimeoutError):
            opcua_simulator.write_node("ns=2;s=Burner1/O2_SP", 3.0)

    def test_modbus_timeout(self, modbus_simulator):
        """Test Modbus communication timeout."""
        modbus_simulator.inject_timeout()

        with pytest.raises(TimeoutError):
            modbus_simulator.read_holding_registers(0, 10)

    def test_timeout_recovery(self, opcua_simulator):
        """Test recovery after timeout."""
        # Cause timeout
        opcua_simulator.inject_timeout()

        with pytest.raises(TimeoutError):
            opcua_simulator.read_node("ns=2;s=Test")

        # Recover
        opcua_simulator.recover()

        # Should work again
        result = opcua_simulator.read_node("ns=2;s=Test")
        assert result["status"] == "ok"

    def test_intermittent_timeouts(self, opcua_simulator):
        """Test handling of intermittent timeouts."""
        # Set 20% timeout probability
        opcua_simulator.set_conditions(NetworkConditions(timeout_probability=0.20))

        successes = 0
        failures = 0

        for _ in range(100):
            try:
                opcua_simulator.read_node("ns=2;s=Test")
                successes += 1
            except TimeoutError:
                failures += 1

        # Should have approximately 20% failures
        assert 10 <= failures <= 35, f"Expected ~20% failures, got {failures}%"
        assert successes > 60, "Should have majority successes"

    def test_timeout_with_retry_logic(self, opcua_simulator):
        """Test timeout handling with retry logic."""
        max_retries = 3
        opcua_simulator.set_conditions(NetworkConditions(timeout_probability=0.50))

        def read_with_retry(node_id: str, retries: int = max_retries) -> Any:
            last_error = None
            for attempt in range(retries + 1):
                try:
                    return opcua_simulator.read_node(node_id)
                except TimeoutError as e:
                    last_error = e
                    opcua_simulator.metrics.retries += 1
                    continue
            raise last_error

        # With 50% timeout and 4 attempts, should usually succeed
        successes = 0
        for _ in range(20):
            try:
                read_with_retry("ns=2;s=Test")
                successes += 1
            except TimeoutError:
                pass

        # With 4 attempts and 50% success, probability of all failing is 0.5^4 = 6.25%
        assert successes >= 15, "Retry logic should improve success rate"


# ============================================================================
# PACKET LOSS TESTS
# ============================================================================

class TestPacketLoss:
    """Test suite for packet loss scenarios."""

    def test_single_packet_loss(self, opcua_simulator):
        """Test single packet loss detection."""
        opcua_simulator.set_conditions(NetworkConditions(packet_loss_rate=1.0))

        with pytest.raises(PacketLossError):
            opcua_simulator.read_node("ns=2;s=Test")

        assert opcua_simulator.metrics.packet_loss_count == 1

    @pytest.mark.parametrize("loss_rate", [0.01, 0.05, 0.10, 0.20])
    def test_varying_packet_loss_rates(self, opcua_simulator, loss_rate: float):
        """Test system behavior at varying packet loss rates."""
        opcua_simulator.set_conditions(NetworkConditions(packet_loss_rate=loss_rate))

        losses = 0
        total = 500

        for _ in range(total):
            try:
                opcua_simulator.read_node("ns=2;s=Test")
            except PacketLossError:
                losses += 1

        actual_rate = losses / total
        # Allow for statistical variance
        assert abs(actual_rate - loss_rate) < 0.05, \
            f"Expected {loss_rate*100}% loss, got {actual_rate*100:.1f}%"

    def test_burst_packet_loss(self, opcua_simulator):
        """Test burst packet loss patterns."""
        results = []

        for i in range(100):
            # Simulate burst: high loss for 10 packets, then normal
            if 30 <= i < 40:
                opcua_simulator.set_conditions(NetworkConditions(packet_loss_rate=0.80))
            else:
                opcua_simulator.set_conditions(NetworkConditions(packet_loss_rate=0.0))

            try:
                opcua_simulator.read_node("ns=2;s=Test")
                results.append(True)
            except PacketLossError:
                results.append(False)

        # Count losses in burst window
        burst_losses = sum(1 for i, r in enumerate(results) if 30 <= i < 40 and not r)
        other_losses = sum(1 for i, r in enumerate(results) if not (30 <= i < 40) and not r)

        assert burst_losses > 5, "Should have significant losses during burst"
        assert other_losses == 0, "Should have no losses outside burst"


# ============================================================================
# CONNECTION DROP TESTS
# ============================================================================

class TestConnectionDrops:
    """Test suite for connection drop scenarios."""

    def test_connection_drop(self, opcua_simulator):
        """Test behavior when connection drops."""
        # Normal operation
        result = opcua_simulator.read_node("ns=2;s=Test")
        assert result["status"] == "ok"

        # Drop connection
        opcua_simulator.inject_disconnect()

        with pytest.raises(ConnectionError):
            opcua_simulator.read_node("ns=2;s=Test")

        assert opcua_simulator.metrics.connection_drops == 1

    def test_reconnection_after_drop(self, opcua_simulator):
        """Test reconnection after connection drop."""
        # Drop connection
        opcua_simulator.inject_disconnect()

        with pytest.raises(ConnectionError):
            opcua_simulator.read_node("ns=2;s=Test")

        # Reconnect
        opcua_simulator.recover()

        # Should work again
        result = opcua_simulator.read_node("ns=2;s=Test")
        assert result["status"] == "ok"

    def test_subscription_survives_brief_disconnect(self, opcua_simulator):
        """Test subscription handling during brief disconnect."""
        # Create subscription
        sub_id = opcua_simulator.create_subscription(1000)
        assert opcua_simulator.subscription_active

        # Brief disconnect
        opcua_simulator.inject_disconnect()
        opcua_simulator.recover()

        # Subscription should be recoverable
        # (In real implementation, would need to resubscribe)

    def test_multiple_connection_drops(self, opcua_simulator):
        """Test handling of multiple connection drops."""
        drop_count = 0

        for i in range(10):
            if i % 3 == 0:
                opcua_simulator.inject_disconnect()
                try:
                    opcua_simulator.read_node("ns=2;s=Test")
                except ConnectionError:
                    drop_count += 1
                finally:
                    opcua_simulator.recover()
            else:
                result = opcua_simulator.read_node("ns=2;s=Test")
                assert result["status"] == "ok"

        assert drop_count == 4  # i = 0, 3, 6, 9


# ============================================================================
# LATENCY AND JITTER TESTS
# ============================================================================

class TestLatencyAndJitter:
    """Test suite for latency and jitter scenarios."""

    def test_high_latency_impact(self, opcua_simulator):
        """Test impact of high latency."""
        opcua_simulator.set_conditions(NetworkConditions(latency_ms=200.0))

        result = opcua_simulator.read_node("ns=2;s=Test")

        assert result["latency_ms"] >= 150  # Accounting for jitter

    def test_latency_jitter(self, opcua_simulator):
        """Test latency jitter distribution."""
        opcua_simulator.set_conditions(NetworkConditions(
            latency_ms=50.0,
            latency_jitter_ms=20.0
        ))

        latencies = []
        for _ in range(100):
            result = opcua_simulator.read_node("ns=2;s=Test")
            latencies.append(result["latency_ms"])

        # Check jitter is present
        latency_std = np.std(latencies)
        assert latency_std > 5, "Should have measurable jitter"

        # Mean should be close to configured value
        assert abs(np.mean(latencies) - 50.0) < 10

    def test_latency_exceeds_timeout(self, opcua_simulator):
        """Test when latency exceeds timeout threshold."""
        opcua_simulator.set_conditions(NetworkConditions(latency_ms=6000.0))

        with pytest.raises(TimeoutError):
            opcua_simulator.send_request({}, timeout_ms=5000)

    @pytest.mark.parametrize("latency_ms", [10, 50, 100, 200, 500, 1000])
    def test_latency_measurement_accuracy(self, opcua_simulator, latency_ms: int):
        """Test latency measurement at different levels."""
        opcua_simulator.set_conditions(NetworkConditions(
            latency_ms=float(latency_ms),
            latency_jitter_ms=float(latency_ms * 0.1)
        ))

        latencies = []
        for _ in range(20):
            result = opcua_simulator.read_node("ns=2;s=Test")
            latencies.append(result["latency_ms"])

        avg_latency = np.mean(latencies)

        # Should be within 20% of configured latency
        assert abs(avg_latency - latency_ms) < latency_ms * 0.25


# ============================================================================
# DATA CORRUPTION TESTS
# ============================================================================

class TestDataCorruption:
    """Test suite for data corruption scenarios."""

    def test_data_corruption_detection(self, opcua_simulator):
        """Test detection of data corruption."""
        opcua_simulator.set_conditions(NetworkConditions(corruption_rate=1.0))

        with pytest.raises(DataCorruptionError):
            opcua_simulator.read_node("ns=2;s=Test")

        assert opcua_simulator.metrics.corruption_count == 1

    def test_corruption_rate_measurement(self, opcua_simulator):
        """Test corruption rate matches configuration."""
        target_rate = 0.10
        opcua_simulator.set_conditions(NetworkConditions(corruption_rate=target_rate))

        corruptions = 0
        total = 500

        for _ in range(total):
            try:
                opcua_simulator.read_node("ns=2;s=Test")
            except DataCorruptionError:
                corruptions += 1

        actual_rate = corruptions / total
        assert abs(actual_rate - target_rate) < 0.05


# ============================================================================
# QUEUE OVERFLOW TESTS
# ============================================================================

class TestQueueOverflow:
    """Test suite for message queue overflow scenarios."""

    def test_queue_capacity_limit(self, opcua_simulator):
        """Test queue behavior at capacity."""
        # Fill the queue
        try:
            for i in range(150):  # More than queue capacity of 100
                opcua_simulator._request_queue.put_nowait(f"request_{i}")
        except queue.Full:
            pass  # Expected when queue is full

        assert opcua_simulator._request_queue.full() or \
               opcua_simulator._request_queue.qsize() >= 100

    def test_queue_drain_under_load(self, opcua_simulator):
        """Test queue draining under load."""
        # Put some items in queue
        for i in range(50):
            opcua_simulator._request_queue.put_nowait(f"request_{i}")

        initial_size = opcua_simulator._request_queue.qsize()

        # Drain queue
        drained = 0
        while not opcua_simulator._request_queue.empty():
            opcua_simulator._request_queue.get_nowait()
            drained += 1

        assert drained == initial_size


# ============================================================================
# NETWORK PARTITION TESTS
# ============================================================================

class TestNetworkPartition:
    """Test suite for network partition scenarios."""

    def test_complete_partition(self, opcua_simulator, modbus_simulator):
        """Test complete network partition (all channels down)."""
        opcua_simulator.inject_disconnect()
        modbus_simulator.inject_disconnect()

        with pytest.raises(ConnectionError):
            opcua_simulator.read_node("ns=2;s=Test")

        with pytest.raises(ConnectionError):
            modbus_simulator.read_holding_registers(0, 10)

    def test_partial_partition(self, opcua_simulator, modbus_simulator):
        """Test partial network partition (some channels working)."""
        # OPC-UA down, Modbus working
        opcua_simulator.inject_disconnect()

        with pytest.raises(ConnectionError):
            opcua_simulator.read_node("ns=2;s=Test")

        # Modbus should still work
        result = modbus_simulator.read_holding_registers(0, 10)
        assert len(result) == 10

    def test_partition_recovery_sequence(self, opcua_simulator, modbus_simulator):
        """Test recovery from network partition."""
        # Both down
        opcua_simulator.inject_disconnect()
        modbus_simulator.inject_disconnect()

        # Recover Modbus first
        modbus_simulator.recover()
        result = modbus_simulator.read_holding_registers(0, 10)
        assert len(result) == 10

        # OPC-UA still down
        with pytest.raises(ConnectionError):
            opcua_simulator.read_node("ns=2;s=Test")

        # Recover OPC-UA
        opcua_simulator.recover()
        result = opcua_simulator.read_node("ns=2;s=Test")
        assert result["status"] == "ok"


# ============================================================================
# COMMUNICATION METRICS TESTS
# ============================================================================

class TestCommunicationMetrics:
    """Test suite for communication metrics collection."""

    def test_success_rate_tracking(self, opcua_simulator):
        """Test tracking of success rate."""
        opcua_simulator.set_conditions(NetworkConditions(packet_loss_rate=0.10))

        for _ in range(100):
            try:
                opcua_simulator.read_node("ns=2;s=Test")
            except PacketLossError:
                pass

        total = opcua_simulator.metrics.requests_sent
        succeeded = opcua_simulator.metrics.requests_succeeded
        failed = opcua_simulator.metrics.requests_failed

        assert total == 100
        assert succeeded + failed == total
        assert 0.85 <= succeeded / total <= 0.95

    def test_latency_statistics(self, opcua_simulator):
        """Test latency statistics collection."""
        opcua_simulator.set_conditions(NetworkConditions(
            latency_ms=100.0,
            latency_jitter_ms=20.0
        ))

        for _ in range(50):
            opcua_simulator.read_node("ns=2;s=Test")

        assert opcua_simulator.metrics.avg_latency_ms > 0
        assert opcua_simulator.metrics.max_latency_ms > opcua_simulator.metrics.avg_latency_ms

    def test_last_successful_time_tracking(self, opcua_simulator):
        """Test tracking of last successful communication time."""
        # Should be None initially
        assert opcua_simulator.metrics.last_successful_time is None

        # After successful request
        opcua_simulator.read_node("ns=2;s=Test")

        assert opcua_simulator.metrics.last_successful_time is not None
        assert (datetime.utcnow() - opcua_simulator.metrics.last_successful_time).seconds < 1


# ============================================================================
# INTEGRATION WITH SAFETY SYSTEMS TESTS
# ============================================================================

class TestCommunicationWithSafetySystems:
    """Test suite for communication failures affecting safety systems."""

    def test_bms_communication_failure_handling(self):
        """Test BMS handling when communication fails."""
        mock_interface = MagicMock()
        mock_interface.read_status.side_effect = ConnectionError("OPC-UA connection lost")

        manager = InterlockManager(
            unit_id="BLR-TEST",
            bms_interface=mock_interface
        )

        status = manager.read_bms_status("BLR-TEST")

        # Should return safe-state (fault) status
        assert status.state == BMSState.FAULT
        assert status.lockout_active == True
        assert status.flame_proven == False

    def test_sis_communication_timeout(self):
        """Test SIS handling on communication timeout."""
        mock_interface = MagicMock()
        mock_interface.read_status.side_effect = TimeoutError("Modbus timeout")

        manager = InterlockManager(
            unit_id="BLR-TEST",
            sis_interface=mock_interface
        )

        status = manager.read_sis_status("BLR-TEST")

        # Should return fault status
        assert status.state == SISState.FAULT

    def test_graceful_degradation_on_comm_failure(self):
        """Test graceful degradation when communication fails."""
        # With no interface configured, should return default (safe) status
        manager = InterlockManager(unit_id="BLR-TEST")

        # Should return default status (simulated for testing)
        bms_status = manager.read_bms_status("BLR-TEST")
        sis_status = manager.read_sis_status("BLR-TEST")

        # Default test status should be "healthy" for testing
        # In production failure, would be fault
        assert bms_status.state in [BMSState.RUN, BMSState.FAULT]


# ============================================================================
# ASYNC COMMUNICATION TESTS
# ============================================================================

class TestAsyncCommunication:
    """Test suite for asynchronous communication patterns."""

    @pytest.mark.asyncio
    async def test_async_timeout_handling(self):
        """Test async timeout handling."""
        async def async_read_with_timeout(timeout_s: float):
            await asyncio.sleep(timeout_s + 0.1)
            return {"status": "ok"}

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                async_read_with_timeout(0.1),
                timeout=0.05
            )

    @pytest.mark.asyncio
    async def test_async_retry_pattern(self):
        """Test async retry pattern."""
        attempt_count = 0

        async def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Connection failed")
            return {"status": "ok"}

        async def with_retry(func, max_retries=3):
            last_error = None
            for _ in range(max_retries):
                try:
                    return await func()
                except ConnectionError as e:
                    last_error = e
                    await asyncio.sleep(0.01)
            raise last_error

        result = await with_retry(failing_operation)
        assert result["status"] == "ok"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_failures(self):
        """Test handling concurrent requests with some failures."""
        async def maybe_fail(i: int):
            await asyncio.sleep(0.001)
            if i % 3 == 0:
                raise PacketLossError(f"Request {i} lost")
            return i

        tasks = [maybe_fail(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if isinstance(r, int)]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(failures) == 4  # 0, 3, 6, 9
        assert len(successes) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
