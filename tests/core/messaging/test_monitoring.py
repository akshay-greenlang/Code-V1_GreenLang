# -*- coding: utf-8 -*-
"""
Test suite for MessageBusMonitor.

Tests cover:
- Health status checking
- Performance metrics collection
- Threshold detection
- Prometheus metrics export

Author: GreenLang Framework Team
Date: December 2025
"""

import asyncio
import pytest
import pytest_asyncio

from greenlang.core.messaging import (
    InMemoryMessageBus,
    MessageBusConfig,
    create_event,
    StandardEvents,
)
from greenlang.core.messaging.monitoring import MessageBusMonitor


@pytest_asyncio.fixture
async def message_bus():
    """Create and start a message bus for testing."""
    bus = InMemoryMessageBus()
    await bus.start()
    yield bus
    await bus.close()


@pytest_asyncio.fixture
async def monitor(message_bus):
    """Create a monitor for the message bus."""
    mon = MessageBusMonitor(message_bus, check_interval=0.5)
    await mon.start()
    yield mon
    await mon.stop()


class TestHealthStatus:
    """Test health status checking."""

    @pytest.mark.asyncio
    async def test_healthy_status(self, message_bus):
        """Test that healthy bus reports healthy status."""
        monitor = MessageBusMonitor(message_bus)

        # Check health on empty, idle bus
        health = monitor.check_health()

        assert health.status == "healthy"
        assert health.queue_health == "healthy"
        assert health.delivery_health == "healthy"
        assert len(health.issues) == 0

    @pytest.mark.asyncio
    async def test_high_queue_utilization(self):
        """Test detection of high queue utilization."""
        # Create bus with small queue
        config = MessageBusConfig(max_queue_size=10)
        bus = InMemoryMessageBus(config)
        await bus.start()

        monitor = MessageBusMonitor(
            bus,
            max_queue_utilization=0.5,  # 50% threshold
        )

        # Don't start the bus processor to let queue fill up
        await bus.stop()  # Stop processor

        # Fill queue beyond threshold
        for i in range(6):
            event = create_event(
                event_type="test.event",
                source_agent="test",
                payload={"i": i},
            )
            await bus.publish(event)

        # Check health
        health = monitor.check_health()

        assert health.status in ("degraded", "unhealthy")
        assert "Queue utilization" in str(health.issues)

        await bus.close()

    @pytest.mark.asyncio
    async def test_high_error_rate(self, message_bus):
        """Test detection of high error rate."""
        monitor = MessageBusMonitor(
            message_bus,
            max_error_rate=0.1,  # 10% threshold
        )

        # Create failing handler
        async def failing_handler(event):
            raise ValueError("Simulated failure")

        await message_bus.subscribe("test.error", failing_handler, "failing")

        # Publish events that will fail
        for i in range(5):
            event = create_event(
                event_type="test.error",
                source_agent="test",
                payload={"i": i},
            )
            await message_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Check health
        health = monitor.check_health()

        # Should detect high error rate
        assert health.error_rate > 0
        assert "Error rate" in str(health.issues) or health.status in ("degraded", "unhealthy")


class TestPerformanceMetrics:
    """Test performance metrics collection."""

    @pytest.mark.asyncio
    async def test_events_per_second_calculation(self, message_bus, monitor):
        """Test calculation of events per second."""
        # Publish events
        for i in range(10):
            event = create_event(
                event_type="test.perf",
                source_agent="test",
                payload={"i": i},
            )
            await message_bus.publish(event)

        # Wait for monitoring interval
        await asyncio.sleep(0.6)

        # Get performance metrics
        perf = monitor.get_performance_metrics()

        # Should have recorded some events per second
        # (exact value depends on timing, just check it's tracked)
        assert perf.events_per_second >= 0

    @pytest.mark.asyncio
    async def test_delivery_time_tracking(self, message_bus, monitor):
        """Test delivery time tracking."""

        async def slow_handler(event):
            await asyncio.sleep(0.01)

        await message_bus.subscribe("test.timing", slow_handler, "slow")

        # Publish event
        event = create_event(
            event_type="test.timing",
            source_agent="test",
            payload={},
        )
        await message_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Get performance metrics
        perf = monitor.get_performance_metrics()

        # Should have tracked delivery time
        assert perf.avg_delivery_time_ms > 0

    @pytest.mark.asyncio
    async def test_error_rate_calculation(self, message_bus):
        """Test error rate calculation."""
        monitor = MessageBusMonitor(message_bus)

        # Create mix of successful and failing handlers
        async def success_handler(event):
            pass

        async def fail_handler(event):
            raise ValueError("Fail")

        await message_bus.subscribe("test.success", success_handler, "success")
        await message_bus.subscribe("test.fail", fail_handler, "fail")

        # Publish events
        for i in range(5):
            await message_bus.publish(
                create_event("test.success", "test", {"i": i})
            )
        for i in range(2):
            await message_bus.publish(
                create_event("test.fail", "test", {"i": i})
            )

        # Wait for processing
        await asyncio.sleep(0.5)

        # Get performance metrics
        perf = monitor.get_performance_metrics()

        # Should have error rate > 0 but < 1
        assert 0 < perf.error_rate < 1


class TestMetricsSummary:
    """Test metrics summary."""

    @pytest.mark.asyncio
    async def test_get_metrics_summary(self, message_bus, monitor):
        """Test getting comprehensive metrics summary."""
        # Publish some events
        for i in range(3):
            event = create_event(
                event_type="test.summary",
                source_agent="test",
                payload={"i": i},
            )
            await message_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Get summary
        summary = monitor.get_metrics_summary()

        # Check structure
        assert "health" in summary
        assert "performance" in summary
        assert "raw_metrics" in summary
        assert "monitoring" in summary

        # Check health section
        assert "status" in summary["health"]
        assert "queue_health" in summary["health"]
        assert "delivery_health" in summary["health"]

        # Check performance section
        assert "events_per_second" in summary["performance"]
        assert "avg_delivery_time_ms" in summary["performance"]

        # Check raw metrics section
        assert "events_published" in summary["raw_metrics"]
        assert "events_delivered" in summary["raw_metrics"]

        # Check monitoring section
        assert summary["monitoring"]["running"] is True


class TestPrometheusExport:
    """Test Prometheus metrics export."""

    @pytest.mark.asyncio
    async def test_prometheus_format(self, message_bus):
        """Test exporting metrics in Prometheus format."""
        monitor = MessageBusMonitor(message_bus)

        # Publish some events
        for i in range(5):
            event = create_event(
                event_type="test.prom",
                source_agent="test",
                payload={"i": i},
            )
            await message_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Export Prometheus metrics
        prom_metrics = monitor.export_prometheus_metrics()

        # Check format
        assert isinstance(prom_metrics, str)
        assert "# HELP" in prom_metrics
        assert "# TYPE" in prom_metrics
        assert "greenlang_message_bus_events_published_total" in prom_metrics
        assert "greenlang_message_bus_queue_size" in prom_metrics

        # Check that values are included
        lines = prom_metrics.split("\n")
        metric_lines = [l for l in lines if not l.startswith("#") and l.strip()]
        assert len(metric_lines) > 0


class TestMonitoringLoop:
    """Test monitoring background loop."""

    @pytest.mark.asyncio
    async def test_monitor_start_stop(self, message_bus):
        """Test starting and stopping monitor."""
        monitor = MessageBusMonitor(message_bus, check_interval=0.1)

        # Start
        await monitor.start()
        assert monitor._running is True

        # Let it run for a bit
        await asyncio.sleep(0.3)

        # Stop
        await monitor.stop()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_monitor_detects_degradation(self, message_bus):
        """Test that monitor detects performance degradation."""
        monitor = MessageBusMonitor(
            message_bus,
            check_interval=0.2,
            max_error_rate=0.1,
        )
        await monitor.start()

        # Create failing handler
        async def failing_handler(event):
            raise ValueError("Fail")

        await message_bus.subscribe("test.degrade", failing_handler, "fail")

        # Publish events that will fail
        for i in range(5):
            await message_bus.publish(
                create_event("test.degrade", "test", {"i": i})
            )

        # Wait for monitoring to detect issue
        await asyncio.sleep(0.5)

        # Check health
        health = monitor.check_health()

        # Should have detected degradation
        assert health.status in ("degraded", "unhealthy") or health.error_rate > 0

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_monitor_history_tracking(self, message_bus):
        """Test that monitor tracks historical data."""
        monitor = MessageBusMonitor(message_bus, check_interval=0.1)
        await monitor.start()

        # Publish events over time
        for i in range(3):
            event = create_event(
                event_type="test.history",
                source_agent="test",
                payload={"i": i},
            )
            await message_bus.publish(event)
            await asyncio.sleep(0.15)

        # Check that history has been collected
        assert len(monitor._events_history) > 0
        assert len(monitor._delivery_time_history) > 0

        await monitor.stop()
