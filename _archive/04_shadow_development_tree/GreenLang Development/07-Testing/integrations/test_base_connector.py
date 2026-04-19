"""
Tests for BaseConnector Framework
==================================

Comprehensive tests for base connector functionality including:
- Connection lifecycle
- Health monitoring
- Metrics tracking
- Provenance generation
- Context manager support

Author: GreenLang Backend Team
Date: 2025-12-01
"""

import pytest
import asyncio
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Tuple

from greenlang.integrations.base_connector import (
    BaseConnector,
    MockConnector,
    ConnectorConfig,
    ConnectorMetrics,
    ConnectorProvenance,
    HealthStatus,
    ConnectionState,
)


# Test models
class TestQuery(BaseModel):
    """Test query model."""

    query_id: str = Field(..., description="Query identifier")
    value: int = Field(default=42, description="Test value")


class TestPayload(BaseModel):
    """Test payload model."""

    result: int = Field(..., description="Result value")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TestConfig(ConnectorConfig):
    """Test connector configuration."""

    test_param: str = Field(default="test", description="Test parameter")


# Concrete test connector
class TestConnector(BaseConnector[TestQuery, TestPayload, TestConfig]):
    """Test connector implementation."""

    connector_id = "test-connector"
    connector_version = "1.0.0"

    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.connect_called = False
        self.disconnect_called = False
        self.fetch_count = 0

    async def connect(self) -> bool:
        """Test connect."""
        self.connect_called = True
        return True

    async def disconnect(self) -> bool:
        """Test disconnect."""
        self.disconnect_called = True
        return True

    async def _health_check_impl(self) -> bool:
        """Test health check."""
        return True

    async def _fetch_data_impl(self, query: TestQuery) -> TestPayload:
        """Test fetch implementation."""
        self.fetch_count += 1
        return TestPayload(result=query.value * 2)


class TestMockConnector(MockConnector[TestQuery, TestPayload, TestConfig]):
    """Test mock connector."""

    connector_id = "mock-connector"
    connector_version = "1.0.0"

    async def _fetch_data_impl(self, query: TestQuery) -> TestPayload:
        """Mock fetch implementation."""
        return TestPayload(result=100)


class TestBaseConnector:
    """Test suite for BaseConnector."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TestConfig(
            connector_id="test-connector-1",
            connector_type="test",
            max_retries=3,
            timeout_seconds=10,
            circuit_breaker_enabled=True
        )

    @pytest.fixture
    def connector(self, config):
        """Create test connector."""
        return TestConnector(config)

    @pytest.fixture
    def query(self):
        """Create test query."""
        return TestQuery(query_id="test-001", value=21)

    @pytest.mark.asyncio
    async def test_connector_initialization(self, connector, config):
        """Test connector initialization."""
        assert connector.config == config
        assert connector.metrics.health_status == HealthStatus.UNKNOWN
        assert connector.metrics.connection_state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, connector):
        """Test connection lifecycle."""
        # Initially disconnected
        assert not connector.connect_called

        # Connect
        result = await connector.connect()
        assert result is True
        assert connector.connect_called

        # Disconnect
        result = await connector.disconnect()
        assert result is True
        assert connector.disconnect_called

    @pytest.mark.asyncio
    async def test_health_check(self, connector):
        """Test health check functionality."""
        await connector.connect()

        result = await connector.health_check()
        assert result is True
        assert connector.metrics.health_status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_fetch_data(self, connector, query):
        """Test data fetching."""
        await connector.connect()

        payload, provenance = await connector.fetch_data(query)

        # Verify payload
        assert isinstance(payload, TestPayload)
        assert payload.result == 42  # 21 * 2

        # Verify provenance
        assert isinstance(provenance, ConnectorProvenance)
        assert provenance.connector_id == connector.config.connector_id
        assert provenance.connector_type == "test"
        assert provenance.query_hash
        assert provenance.response_hash

        # Verify metrics updated
        assert connector.metrics.total_requests == 1
        assert connector.metrics.successful_requests == 1
        assert connector.metrics.failed_requests == 0
        assert connector.fetch_count == 1

    @pytest.mark.asyncio
    async def test_fetch_data_timeout(self, config):
        """Test fetch with timeout."""

        class SlowConnector(TestConnector):
            async def _fetch_data_impl(self, query: TestQuery) -> TestPayload:
                await asyncio.sleep(5)  # Longer than timeout
                return TestPayload(result=42)

        connector = SlowConnector(config)
        await connector.connect()

        query = TestQuery(query_id="slow", value=1)

        with pytest.raises(TimeoutError):
            await connector.fetch_data(query, timeout=1)

        # Verify metrics
        assert connector.metrics.failed_requests == 1

    @pytest.mark.asyncio
    async def test_async_context_manager(self, connector, query):
        """Test async context manager."""
        async with connector as conn:
            assert conn.connect_called
            assert connector.metrics.connection_state == ConnectionState.CONNECTED

            # Fetch data within context
            payload, _ = await conn.fetch_data(query)
            assert payload.result == 42

        # After context exit
        assert connector.disconnect_called
        assert connector.metrics.connection_state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, connector, query):
        """Test metrics are properly tracked."""
        await connector.connect()

        # Initial state
        initial_metrics = connector.get_metrics()
        assert initial_metrics.total_requests == 0
        assert initial_metrics.avg_response_time_ms == 0

        # Make requests
        await connector.fetch_data(query)
        await connector.fetch_data(query)

        # Check metrics
        metrics = connector.get_metrics()
        assert metrics.total_requests == 2
        assert metrics.successful_requests == 2
        assert metrics.avg_response_time_ms > 0
        assert metrics.last_success_time is not None

    @pytest.mark.asyncio
    async def test_metrics_reset(self, connector, query):
        """Test metrics reset."""
        await connector.connect()
        await connector.fetch_data(query)

        # Verify metrics exist
        assert connector.metrics.total_requests == 1

        # Reset
        connector.reset_metrics()

        # Verify reset
        assert connector.metrics.total_requests == 0
        assert connector.metrics.successful_requests == 0
        assert connector.metrics.avg_response_time_ms == 0

    @pytest.mark.asyncio
    async def test_provenance_tracking(self, connector, query):
        """Test provenance tracking."""
        await connector.connect()

        payload, prov = await connector.fetch_data(query)

        # Verify provenance fields
        assert prov.connector_id == connector.config.connector_id
        assert prov.connector_type == "test"
        assert prov.connector_version == "1.0.0"
        assert prov.query_hash
        assert prov.response_hash
        assert prov.timestamp
        assert prov.mode == "live"

    @pytest.mark.asyncio
    async def test_provenance_disabled(self):
        """Test with provenance disabled."""
        config = TestConfig(
            connector_id="test-no-prov",
            connector_type="test",
            enable_provenance=False
        )
        connector = TestConnector(config)
        await connector.connect()

        query = TestQuery(query_id="test", value=10)
        payload, prov = await connector.fetch_data(query)

        # Provenance should have empty hashes
        assert prov.query_hash == ""
        assert prov.response_hash == ""

    @pytest.mark.asyncio
    async def test_mock_connector(self):
        """Test mock connector."""
        config = TestConfig(
            connector_id="mock-test",
            connector_type="test",
            mock_mode=True
        )
        connector = TestMockConnector(config)

        async with connector:
            query = TestQuery(query_id="test", value=1)
            payload, prov = await connector.fetch_data(query)

            # Mock should return fixed value
            assert payload.result == 100
            assert prov.mode == "mock"


class TestRetryLogic:
    """Test retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry logic on transient failures."""

        class FlakyConnector(TestConnector):
            def __init__(self, config):
                super().__init__(config)
                self.attempt_count = 0

            async def _fetch_data_impl(self, query: TestQuery) -> TestPayload:
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise ConnectionError("Transient error")
                return TestPayload(result=42)

        config = TestConfig(
            connector_id="flaky",
            connector_type="test",
            max_retries=3
        )
        connector = FlakyConnector(config)
        await connector.connect()

        query = TestQuery(query_id="test", value=1)
        payload, _ = await connector.fetch_data(query)

        # Should succeed after retries
        assert payload.result == 42
        assert connector.attempt_count == 3
        assert connector.metrics.retry_count > 0

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry exhaustion."""

        class AlwaysFailConnector(TestConnector):
            async def _fetch_data_impl(self, query: TestQuery) -> TestPayload:
                raise ConnectionError("Permanent error")

        config = TestConfig(
            connector_id="always-fail",
            connector_type="test",
            max_retries=2
        )
        connector = AlwaysFailConnector(config)
        await connector.connect()

        query = TestQuery(query_id="test", value=1)

        with pytest.raises(ConnectionError):
            await connector.fetch_data(query)

        assert connector.metrics.failed_requests == 1


class TestHealthMonitoring:
    """Test health monitoring."""

    @pytest.mark.asyncio
    async def test_health_monitoring_start_stop(self, connector):
        """Test starting and stopping health monitoring."""
        await connector.connect()

        # Start monitoring
        await connector.start_health_monitoring()
        assert connector._health_check_task is not None

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop monitoring
        await connector.stop_health_monitoring()
        assert connector._health_check_task is None

    @pytest.mark.asyncio
    async def test_health_status_tracking(self):
        """Test health status tracking."""

        class UnhealthyConnector(TestConnector):
            def __init__(self, config):
                super().__init__(config)
                self.is_healthy = True

            async def _health_check_impl(self) -> bool:
                return self.is_healthy

        config = TestConfig(
            connector_id="unhealthy-test",
            connector_type="test"
        )
        connector = UnhealthyConnector(config)
        await connector.connect()

        # Initially healthy
        result = await connector.health_check()
        assert result is True
        assert connector.metrics.health_status == HealthStatus.HEALTHY

        # Make unhealthy
        connector.is_healthy = False
        result = await connector.health_check()
        assert result is False
        assert connector.metrics.health_status == HealthStatus.DEGRADED
