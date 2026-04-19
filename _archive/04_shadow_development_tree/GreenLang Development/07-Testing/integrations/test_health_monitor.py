"""
Tests for Health Monitor
=========================

Tests for connector health monitoring and metrics collection.

Author: GreenLang Backend Team
Date: 2025-12-01
"""

import pytest
import asyncio
from datetime import datetime, timezone
from pydantic import BaseModel

from greenlang.integrations.base_connector import (
    BaseConnector,
    ConnectorConfig,
    HealthStatus,
)
from greenlang.integrations.health_monitor import (
    HealthMonitor,
    HealthCheckResult,
    AggregatedHealth,
)


# Test models
class TestQuery(BaseModel):
    value: int = 1


class TestPayload(BaseModel):
    result: int = 1


class TestConfig(ConnectorConfig):
    pass


class HealthyConnector(BaseConnector[TestQuery, TestPayload, TestConfig]):
    """Always healthy connector."""

    connector_id = "healthy-conn"
    connector_version = "1.0.0"

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> bool:
        return True

    async def _health_check_impl(self) -> bool:
        return True

    async def _fetch_data_impl(self, query: TestQuery) -> TestPayload:
        return TestPayload(result=1)


class UnhealthyConnector(BaseConnector[TestQuery, TestPayload, TestConfig]):
    """Always unhealthy connector."""

    connector_id = "unhealthy-conn"
    connector_version = "1.0.0"

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> bool:
        return True

    async def _health_check_impl(self) -> bool:
        return False

    async def _fetch_data_impl(self, query: TestQuery) -> TestPayload:
        return TestPayload(result=1)


class TestHealthMonitor:
    """Test suite for HealthMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create health monitor."""
        return HealthMonitor(
            check_interval=1,
            enable_prometheus=False
        )

    @pytest.fixture
    def healthy_connector(self):
        """Create healthy connector."""
        config = TestConfig(
            connector_id="healthy-1",
            connector_type="test"
        )
        return HealthyConnector(config)

    @pytest.fixture
    def unhealthy_connector(self):
        """Create unhealthy connector."""
        config = TestConfig(
            connector_id="unhealthy-1",
            connector_type="test"
        )
        return UnhealthyConnector(config)

    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.check_interval == 1
        assert len(monitor._connectors) == 0

    def test_register_connector(self, monitor, healthy_connector):
        """Test connector registration."""
        monitor.register_connector(healthy_connector)

        assert "healthy-1" in monitor._connectors
        assert monitor._connectors["healthy-1"] == healthy_connector

    def test_unregister_connector(self, monitor, healthy_connector):
        """Test connector unregistration."""
        monitor.register_connector(healthy_connector)
        assert "healthy-1" in monitor._connectors

        monitor.unregister_connector("healthy-1")
        assert "healthy-1" not in monitor._connectors

    @pytest.mark.asyncio
    async def test_check_connector_health(self, monitor, healthy_connector):
        """Test checking single connector health."""
        await healthy_connector.connect()
        monitor.register_connector(healthy_connector)

        result = await monitor.check_connector_health("healthy-1")

        assert isinstance(result, HealthCheckResult)
        assert result.connector_id == "healthy-1"
        assert result.health_status == HealthStatus.HEALTHY
        assert result.latency_ms >= 0
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_check_unhealthy_connector(self, monitor, unhealthy_connector):
        """Test checking unhealthy connector."""
        await unhealthy_connector.connect()
        monitor.register_connector(unhealthy_connector)

        result = await monitor.check_connector_health("unhealthy-1")

        assert result.health_status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_all_connectors(self, monitor, healthy_connector, unhealthy_connector):
        """Test checking all connectors."""
        await healthy_connector.connect()
        await unhealthy_connector.connect()

        monitor.register_connector(healthy_connector)
        monitor.register_connector(unhealthy_connector)

        results = await monitor.check_all_connectors()

        assert len(results) == 2
        assert "healthy-1" in results
        assert "unhealthy-1" in results

        assert results["healthy-1"].health_status == HealthStatus.HEALTHY
        assert results["unhealthy-1"].health_status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_get_health_status(self, monitor, healthy_connector):
        """Test getting health status."""
        await healthy_connector.connect()
        monitor.register_connector(healthy_connector)

        # Initially no status
        status = monitor.get_health_status("healthy-1")
        assert status is None

        # After check
        await monitor.check_connector_health("healthy-1")
        status = monitor.get_health_status("healthy-1")
        assert status is not None
        assert status.health_status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_aggregated_health(self, monitor, healthy_connector, unhealthy_connector):
        """Test aggregated health statistics."""
        await healthy_connector.connect()
        await unhealthy_connector.connect()

        monitor.register_connector(healthy_connector)
        monitor.register_connector(unhealthy_connector)

        await monitor.check_all_connectors()

        agg_health = monitor.get_aggregated_health()

        assert isinstance(agg_health, AggregatedHealth)
        assert agg_health.total_connectors == 2
        assert agg_health.healthy_count == 1
        assert agg_health.degraded_count == 1
        assert agg_health.overall_health == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_history(self, monitor, healthy_connector):
        """Test health history tracking."""
        await healthy_connector.connect()
        monitor.register_connector(healthy_connector)

        # Perform multiple checks
        await monitor.check_connector_health("healthy-1")
        await asyncio.sleep(0.1)
        await monitor.check_connector_health("healthy-1")

        history = monitor.get_health_history("healthy-1")

        assert len(history) == 2
        assert all(isinstance(r, HealthCheckResult) for r in history)

    @pytest.mark.asyncio
    async def test_health_history_time_filter(self, monitor, healthy_connector):
        """Test health history with time filter."""
        await healthy_connector.connect()
        monitor.register_connector(healthy_connector)

        await monitor.check_connector_health("healthy-1")

        # Get history for last hour
        history = monitor.get_health_history("healthy-1", hours=1)
        assert len(history) >= 1

        # Get history for 0 hours (should be empty)
        history = monitor.get_health_history("healthy-1", hours=0)
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitor, healthy_connector):
        """Test starting and stopping monitoring."""
        await healthy_connector.connect()
        monitor.register_connector(healthy_connector)

        # Start monitoring
        await monitor.start_monitoring()
        assert monitor._running is True
        assert monitor._monitoring_task is not None

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Should have performed at least one check
        status = monitor.get_health_status("healthy-1")
        assert status is not None

        # Stop monitoring
        await monitor.stop_monitoring()
        assert monitor._running is False
        assert monitor._monitoring_task is None

    @pytest.mark.asyncio
    async def test_get_all_metrics(self, monitor, healthy_connector):
        """Test getting all connector metrics."""
        await healthy_connector.connect()
        monitor.register_connector(healthy_connector)

        # Perform a fetch to generate metrics
        query = TestQuery(value=1)
        await healthy_connector.fetch_data(query)

        all_metrics = monitor.get_all_metrics()

        assert "healthy-1" in all_metrics
        metrics = all_metrics["healthy-1"]
        assert metrics.total_requests > 0


class TestHealthCheckResult:
    """Test HealthCheckResult model."""

    def test_health_check_result_creation(self):
        """Test creating health check result."""
        result = HealthCheckResult(
            connector_id="test-conn",
            health_status=HealthStatus.HEALTHY,
            latency_ms=15.5
        )

        assert result.connector_id == "test-conn"
        assert result.health_status == HealthStatus.HEALTHY
        assert result.latency_ms == 15.5
        assert result.error_message is None


class TestAggregatedHealth:
    """Test AggregatedHealth model."""

    def test_aggregated_health_creation(self):
        """Test creating aggregated health."""
        agg = AggregatedHealth(
            total_connectors=10,
            healthy_count=8,
            degraded_count=1,
            unhealthy_count=1,
            overall_health=HealthStatus.DEGRADED
        )

        assert agg.total_connectors == 10
        assert agg.healthy_count == 8
        assert agg.overall_health == HealthStatus.DEGRADED
