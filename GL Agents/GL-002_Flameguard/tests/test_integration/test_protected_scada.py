"""
GL-002 FLAMEGUARD - Protected SCADA Connector Integration Tests

Integration tests for the protected SCADA connector with circuit breakers,
caching, and graceful degradation.

Test Coverage Target: 85%+
"""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitOpenError,
)
from integration.scada_connector import (
    SCADAConnectionConfig,
    SCADAProtocol,
    TagMapping,
    TagValue,
    TagQuality,
    DataType,
)
from integration.protected_scada_connector import (
    ProtectedSCADAConnector,
    FallbackConfig,
    FallbackStrategy,
    DegradedModeLevel,
    CachedValue,
    ProtectedModbusClient,
    ProtectedOPCUAClient,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def scada_config() -> SCADAConnectionConfig:
    """Create test SCADA configuration."""
    return SCADAConnectionConfig(
        protocol=SCADAProtocol.MODBUS_TCP,
        host="localhost",
        port=502,
        timeout_ms=1000,
    )


@pytest.fixture
def fallback_config() -> FallbackConfig:
    """Create test fallback configuration."""
    return FallbackConfig(
        strategy=FallbackStrategy.LAST_KNOWN_VALUE,
        cache_max_age_s=60.0,
        safe_defaults={
            "pressure": 100.0,
            "temperature": 200.0,
        },
    )


@pytest.fixture
def breaker_config() -> CircuitBreakerConfig:
    """Create test circuit breaker configuration."""
    return CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout_s=0.5,
        half_open_max_calls=2,
        success_threshold=1,
    )


@pytest.fixture
def protected_connector(scada_config, fallback_config, breaker_config):
    """Create protected SCADA connector for testing."""
    # Reset registry for clean state
    CircuitBreakerRegistry.reset_instance()

    connector = ProtectedSCADAConnector(
        config=scada_config,
        fallback_config=fallback_config,
        breaker_config=breaker_config,
    )

    # Add test tag mappings
    connector.add_tag(TagMapping(
        scada_tag="HR100",
        internal_name="pressure",
        data_type=DataType.FLOAT32,
        low_limit=0.0,
        high_limit=200.0,
    ))
    connector.add_tag(TagMapping(
        scada_tag="HR102",
        internal_name="temperature",
        data_type=DataType.FLOAT32,
        low_limit=0.0,
        high_limit=500.0,
    ))

    return connector


@pytest.fixture
def sample_tag_values() -> dict:
    """Sample tag values for testing."""
    return {
        "pressure": TagValue(
            tag="pressure",
            value=125.5,
            quality=TagQuality.GOOD,
            timestamp=datetime.now(timezone.utc),
        ),
        "temperature": TagValue(
            tag="temperature",
            value=350.0,
            quality=TagQuality.GOOD,
            timestamp=datetime.now(timezone.utc),
        ),
    }


# =============================================================================
# CACHED VALUE TESTS
# =============================================================================

class TestCachedValue:
    """Tests for CachedValue class."""

    def test_cached_value_creation(self):
        """Test creating a cached value."""
        cached = CachedValue(
            tag="test_tag",
            value=100.0,
            quality=TagQuality.GOOD,
            timestamp=datetime.now(timezone.utc),
        )

        assert cached.tag == "test_tag"
        assert cached.value == 100.0
        assert cached.cache_hits == 0
        assert cached.source == "scada"

    def test_is_stale(self):
        """Test staleness detection."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        cached = CachedValue(
            tag="test",
            value=100.0,
            quality=TagQuality.GOOD,
            timestamp=old_time,
            cached_at=old_time,
        )

        assert cached.is_stale(max_age_s=60.0)
        assert not cached.is_stale(max_age_s=300.0)

    def test_to_tag_value(self):
        """Test conversion to TagValue."""
        cached = CachedValue(
            tag="test",
            value=100.0,
            quality=TagQuality.GOOD,
            timestamp=datetime.now(timezone.utc),
        )

        # With cached marking
        tag_val = cached.to_tag_value(mark_as_cached=True)
        assert tag_val.quality == TagQuality.LAST_KNOWN

        # Without cached marking
        tag_val = cached.to_tag_value(mark_as_cached=False)
        assert tag_val.quality == TagQuality.GOOD


# =============================================================================
# CONNECTION TESTS
# =============================================================================

class TestProtectedConnectorConnection:
    """Tests for connection handling."""

    @pytest.mark.asyncio
    async def test_successful_connection(self, protected_connector):
        """Test successful SCADA connection."""
        with patch.object(
            protected_connector._connector,
            'connect',
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await protected_connector.connect()

            assert result is True
            assert protected_connector.metrics.connection_attempts == 1
            assert protected_connector.metrics.recovery_successes == 1

    @pytest.mark.asyncio
    async def test_failed_connection(self, protected_connector):
        """Test failed SCADA connection."""
        with patch.object(
            protected_connector._connector,
            'connect',
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await protected_connector.connect()

            assert result is False
            assert protected_connector.metrics.connection_failures == 1

    @pytest.mark.asyncio
    async def test_connection_circuit_opens(self, protected_connector):
        """Test connection circuit breaker opens on repeated failures."""
        with patch.object(
            protected_connector._connector,
            'connect',
            new_callable=AsyncMock,
            side_effect=Exception("Connection refused"),
        ):
            # First two attempts fail and open circuit
            for _ in range(3):
                result = await protected_connector.connect()
                assert result is False

            # Circuit should be open
            assert protected_connector._connection_breaker.is_open


# =============================================================================
# READ TESTS
# =============================================================================

class TestProtectedConnectorRead:
    """Tests for read operations."""

    @pytest.mark.asyncio
    async def test_successful_read(self, protected_connector, sample_tag_values):
        """Test successful tag read."""
        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
            return_value=sample_tag_values,
        ):
            result = await protected_connector.read_tags_safe(["pressure", "temperature"])

            assert "pressure" in result
            assert "temperature" in result
            assert result["pressure"].value == 125.5
            assert protected_connector.metrics.successful_reads == 1

    @pytest.mark.asyncio
    async def test_read_with_fallback_on_failure(
        self,
        protected_connector,
        sample_tag_values,
    ):
        """Test fallback to cached values on read failure."""
        # First, populate cache with successful read
        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
            return_value=sample_tag_values,
        ):
            await protected_connector.read_tags_safe(["pressure"])

        # Now simulate failure
        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
            side_effect=Exception("Read failed"),
        ):
            result = await protected_connector.read_tags_safe(["pressure"], use_cache=True)

            # Should get cached value
            assert "pressure" in result
            assert result["pressure"].quality == TagQuality.LAST_KNOWN

    @pytest.mark.asyncio
    async def test_read_safe_default_fallback(self, protected_connector):
        """Test fallback to safe defaults when no cached value."""
        # Configure fallback to use safe defaults
        protected_connector.fallback_config.strategy = FallbackStrategy.SAFE_DEFAULT

        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
            side_effect=Exception("Read failed"),
        ):
            result = await protected_connector.read_tags_safe(["pressure"], use_cache=True)

            assert "pressure" in result
            assert result["pressure"].value == 100.0  # Safe default

    @pytest.mark.asyncio
    async def test_read_in_cached_only_mode(
        self,
        protected_connector,
        sample_tag_values,
    ):
        """Test reading in cached-only degraded mode."""
        # Populate cache
        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
            return_value=sample_tag_values,
        ):
            await protected_connector.read_tags_safe(["pressure"])

        # Set to cached-only mode
        protected_connector.degraded_mode = DegradedModeLevel.CACHED_ONLY

        # Should return cached value without calling connector
        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
        ) as mock_read:
            result = await protected_connector.read_tags_safe(["pressure"])

            # Connector should not be called
            mock_read.assert_not_called()
            assert "pressure" in result

    @pytest.mark.asyncio
    async def test_read_circuit_opens_on_failures(self, protected_connector):
        """Test read circuit breaker opens on repeated failures."""
        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
            side_effect=Exception("Read error"),
        ):
            # Trigger failures
            for _ in range(3):
                await protected_connector.read_tags_safe(["pressure"], use_cache=False)

            assert protected_connector._read_breaker.is_open


# =============================================================================
# WRITE TESTS
# =============================================================================

class TestProtectedConnectorWrite:
    """Tests for write operations."""

    @pytest.mark.asyncio
    async def test_successful_write(self, protected_connector):
        """Test successful tag write."""
        with patch.object(
            protected_connector._connector,
            'write_tag',
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await protected_connector.write_tag_safe("pressure", 150.0)

            assert result is True
            assert protected_connector.metrics.successful_writes == 1

    @pytest.mark.asyncio
    async def test_write_blocked_in_read_only_mode(self, protected_connector):
        """Test writes are blocked in read-only degraded mode."""
        protected_connector.degraded_mode = DegradedModeLevel.READ_ONLY

        result = await protected_connector.write_tag_safe("pressure", 150.0)

        assert result is False
        assert protected_connector.metrics.failed_writes == 1

    @pytest.mark.asyncio
    async def test_write_blocked_in_offline_mode(self, protected_connector):
        """Test writes are blocked in offline degraded mode."""
        protected_connector.degraded_mode = DegradedModeLevel.OFFLINE

        result = await protected_connector.write_tag_safe("pressure", 150.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_write_circuit_opens_on_failures(self, protected_connector):
        """Test write circuit breaker opens on repeated failures."""
        with patch.object(
            protected_connector._connector,
            'write_tag',
            new_callable=AsyncMock,
            side_effect=Exception("Write error"),
        ):
            for _ in range(3):
                await protected_connector.write_tag_safe("pressure", 150.0)

            assert protected_connector._write_breaker.is_open


# =============================================================================
# DEGRADED MODE TESTS
# =============================================================================

class TestDegradedMode:
    """Tests for degraded mode handling."""

    def test_initial_mode_is_normal(self, protected_connector):
        """Test initial degraded mode is NORMAL."""
        assert protected_connector.degraded_mode == DegradedModeLevel.NORMAL

    def test_mode_changes_when_connection_breaker_opens(self, protected_connector):
        """Test mode changes to OFFLINE when connection breaker opens."""
        protected_connector._connection_breaker.force_open()
        protected_connector._update_degraded_mode()

        assert protected_connector.degraded_mode == DegradedModeLevel.OFFLINE

    def test_mode_changes_when_read_breaker_opens(self, protected_connector):
        """Test mode changes to LIMITED when read breaker opens."""
        protected_connector._read_breaker.force_open()
        protected_connector._update_degraded_mode()

        assert protected_connector.degraded_mode == DegradedModeLevel.LIMITED

    def test_mode_changes_when_write_breaker_opens(self, protected_connector):
        """Test mode changes to READ_ONLY when write breaker opens."""
        protected_connector._write_breaker.force_open()
        protected_connector._update_degraded_mode()

        assert protected_connector.degraded_mode == DegradedModeLevel.READ_ONLY

    def test_mode_changes_when_both_read_write_open(self, protected_connector):
        """Test mode changes to CACHED_ONLY when both breakers open."""
        protected_connector._read_breaker.force_open()
        protected_connector._write_breaker.force_open()
        protected_connector._update_degraded_mode()

        assert protected_connector.degraded_mode == DegradedModeLevel.CACHED_ONLY

    def test_mode_callback_invoked(self, protected_connector):
        """Test degraded mode change callback is invoked."""
        mode_changes = []

        def on_mode_change(mode):
            mode_changes.append(mode)

        protected_connector._on_degraded_mode_change = on_mode_change

        protected_connector._connection_breaker.force_open()
        protected_connector._update_degraded_mode()

        assert len(mode_changes) == 1
        assert mode_changes[0] == DegradedModeLevel.OFFLINE


# =============================================================================
# CACHE TESTS
# =============================================================================

class TestCaching:
    """Tests for value caching."""

    @pytest.mark.asyncio
    async def test_values_are_cached(self, protected_connector, sample_tag_values):
        """Test that successful reads are cached."""
        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
            return_value=sample_tag_values,
        ):
            await protected_connector.read_tags_safe(["pressure"])

        cache_status = protected_connector.get_cache_status()
        assert cache_status["size"] == 1
        assert "pressure" in cache_status["entries"]

    @pytest.mark.asyncio
    async def test_cache_hit_tracking(self, protected_connector, sample_tag_values):
        """Test cache hit tracking."""
        # Populate cache
        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
            return_value=sample_tag_values,
        ):
            await protected_connector.read_tags_safe(["pressure"])

        # Force use of cache
        protected_connector.degraded_mode = DegradedModeLevel.CACHED_ONLY
        await protected_connector.read_tags_safe(["pressure"])

        assert protected_connector.metrics.cache_hits > 0

    def test_clear_cache(self, protected_connector):
        """Test clearing the cache."""
        protected_connector._value_cache["test"] = CachedValue(
            tag="test",
            value=100.0,
            quality=TagQuality.GOOD,
            timestamp=datetime.now(timezone.utc),
        )

        protected_connector.clear_cache()

        assert len(protected_connector._value_cache) == 0


# =============================================================================
# STATUS AND METRICS TESTS
# =============================================================================

class TestStatusAndMetrics:
    """Tests for status and metrics."""

    def test_get_status(self, protected_connector):
        """Test get_status method."""
        status = protected_connector.get_status()

        assert "connected" in status
        assert "healthy" in status
        assert "degraded_mode" in status
        assert "circuit_breakers" in status
        assert "metrics" in status

    def test_get_circuit_status(self, protected_connector):
        """Test get_circuit_status method."""
        status = protected_connector.get_circuit_status()

        assert "connection" in status
        assert "read" in status
        assert "write" in status

    def test_is_healthy(self, protected_connector):
        """Test is_healthy method."""
        assert protected_connector.is_healthy()

        protected_connector._read_breaker.force_open()
        assert not protected_connector.is_healthy()

    def test_reset_circuit_breakers(self, protected_connector):
        """Test resetting circuit breakers."""
        protected_connector._read_breaker.force_open()
        protected_connector._write_breaker.force_open()
        protected_connector._update_degraded_mode()

        protected_connector.reset_circuit_breakers()

        assert protected_connector._read_breaker.is_closed
        assert protected_connector._write_breaker.is_closed
        assert protected_connector.degraded_mode == DegradedModeLevel.NORMAL

    def test_provenance_hash(self, protected_connector):
        """Test provenance hash calculation."""
        hash1 = protected_connector.get_provenance_hash()
        assert len(hash1) == 64

        # Hash changes with state
        protected_connector._read_breaker.force_open()
        protected_connector._update_degraded_mode()
        hash2 = protected_connector.get_provenance_hash()

        assert hash1 != hash2


# =============================================================================
# MODBUS CLIENT TESTS
# =============================================================================

class TestProtectedModbusClient:
    """Tests for ProtectedModbusClient."""

    @pytest.mark.asyncio
    async def test_client_creation(self):
        """Test creating protected Modbus client."""
        client = ProtectedModbusClient(
            host="localhost",
            port=502,
            unit_id=1,
        )

        assert client._connector is not None

    @pytest.mark.asyncio
    async def test_client_status(self):
        """Test getting client status."""
        client = ProtectedModbusClient(
            host="localhost",
            port=502,
        )

        status = client.get_status()

        assert "connected" in status
        assert "circuit_breakers" in status


# =============================================================================
# OPC-UA CLIENT TESTS
# =============================================================================

class TestProtectedOPCUAClient:
    """Tests for ProtectedOPCUAClient."""

    @pytest.mark.asyncio
    async def test_client_creation(self):
        """Test creating protected OPC-UA client."""
        client = ProtectedOPCUAClient(
            endpoint_url="opc.tcp://localhost:4840",
        )

        assert client._connector is not None

    @pytest.mark.asyncio
    async def test_client_status(self):
        """Test getting client status."""
        client = ProtectedOPCUAClient(
            endpoint_url="opc.tcp://localhost:4840",
        )

        status = client.get_status()

        assert "connected" in status
        assert "circuit_breakers" in status


# =============================================================================
# RECOVERY TESTS
# =============================================================================

class TestRecovery:
    """Tests for recovery scenarios."""

    @pytest.mark.asyncio
    async def test_recovery_after_timeout(self, protected_connector, sample_tag_values):
        """Test automatic recovery after timeout."""
        # Open read breaker
        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
            side_effect=Exception("Error"),
        ):
            for _ in range(3):
                await protected_connector.read_tags_safe(["pressure"], use_cache=False)

        assert protected_connector._read_breaker.is_open

        # Wait for recovery timeout
        await asyncio.sleep(0.6)

        # Should allow recovery attempt
        with patch.object(
            protected_connector._connector,
            'read_tags',
            new_callable=AsyncMock,
            return_value=sample_tag_values,
        ):
            result = await protected_connector.read_tags_safe(["pressure"])

        assert "pressure" in result
        # Circuit should be closed or half-open
        assert protected_connector._read_breaker.state in [
            CircuitState.CLOSED,
            CircuitState.HALF_OPEN,
        ]
