"""
Unit tests for SCADA integration module.

Tests OPC-UA and Modbus connectivity, tag operations, alarm management,
and connection health monitoring for flue gas analyzers.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.scada_integration import (
    SCADAClient,
    SCADAConfig,
    FlueGasTag,
    TagDataPoint,
    AlarmData,
    ConnectionProtocol,
    ParameterType,
    MeasurementLocation,
    TagType,
    AlarmSeverity,
    AlarmState,
    AnalyzerType,
    create_scada_client,
    create_standard_flue_gas_tags,
    create_abb_ao2000_tags,
    create_sick_marsic_tags,
    create_horiba_pg_tags,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def opcua_config():
    """OPC-UA configuration fixture."""
    return SCADAConfig(
        protocol=ConnectionProtocol.OPC_UA,
        analyzer_type=AnalyzerType.ABB_AO2000,
        host="192.168.1.100",
        port=4840,
        endpoint_url="opc.tcp://192.168.1.100:4840",
        namespace_index=2,
        connection_timeout=5.0,
        reconnect_interval=2.0,
        max_reconnect_attempts=3,
    )


@pytest.fixture
def modbus_config():
    """Modbus TCP configuration fixture."""
    return SCADAConfig(
        protocol=ConnectionProtocol.MODBUS_TCP,
        analyzer_type=AnalyzerType.HORIBA_PG,
        host="192.168.1.101",
        port=502,
        modbus_unit_id=1,
        modbus_timeout=3.0,
        connection_timeout=5.0,
    )


@pytest.fixture
def sample_flue_gas_tags():
    """Sample flue gas tags fixture."""
    return [
        FlueGasTag(
            tag_name="FG_O2_STACK",
            parameter_type=ParameterType.OXYGEN,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="%",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=25.0,
            low_alarm_limit=1.0,
            high_alarm_limit=10.0,
        ),
        FlueGasTag(
            tag_name="FG_CO_STACK",
            parameter_type=ParameterType.CARBON_MONOXIDE,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="ppm",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=5000.0,
            high_alarm_limit=400.0,
        ),
        FlueGasTag(
            tag_name="AIR_DAMPER_POS",
            parameter_type=ParameterType.DAMPER_POSITION,
            location=MeasurementLocation.FORCED_DRAFT_FAN_OUTLET,
            tag_type=TagType.ANALOG_OUTPUT,
            engineering_unit="%",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=100.0,
        ),
    ]


# =============================================================================
# Configuration Tests
# =============================================================================


def test_scada_config_creation():
    """Test SCADA configuration creation."""
    config = SCADAConfig(
        protocol=ConnectionProtocol.OPC_UA,
        host="localhost",
        port=4840,
    )

    assert config.protocol == ConnectionProtocol.OPC_UA
    assert config.host == "localhost"
    assert config.port == 4840
    assert config.connection_timeout == 10.0
    assert config.enable_buffering is True


def test_scada_config_validation():
    """Test configuration validation."""
    # Invalid port
    with pytest.raises(Exception):
        SCADAConfig(
            protocol=ConnectionProtocol.OPC_UA,
            host="localhost",
            port=70000,  # Invalid port
        )

    # Invalid timeout
    with pytest.raises(Exception):
        SCADAConfig(
            protocol=ConnectionProtocol.OPC_UA,
            host="localhost",
            connection_timeout=-1.0,  # Invalid timeout
        )


def test_flue_gas_tag_creation():
    """Test flue gas tag creation."""
    tag = FlueGasTag(
        tag_name="FG_O2_STACK",
        parameter_type=ParameterType.OXYGEN,
        location=MeasurementLocation.STACK_OUTLET,
        engineering_unit="%",
        raw_min=0.0,
        raw_max=16384.0,
        scaled_min=0.0,
        scaled_max=25.0,
    )

    assert tag.tag_name == "FG_O2_STACK"
    assert tag.parameter_type == ParameterType.OXYGEN
    assert tag.location == MeasurementLocation.STACK_OUTLET
    assert tag.engineering_unit == "%"


# =============================================================================
# Client Initialization Tests
# =============================================================================


def test_client_initialization_opcua(opcua_config):
    """Test SCADA client initialization with OPC-UA."""
    client = SCADAClient(opcua_config)

    assert client._config == opcua_config
    assert not client.is_connected()
    assert len(client._tags) == 0


def test_client_initialization_modbus(modbus_config):
    """Test SCADA client initialization with Modbus."""
    client = SCADAClient(modbus_config)

    assert client._config == modbus_config
    assert not client.is_connected()


def test_factory_function():
    """Test factory function."""
    client = create_scada_client(
        analyzer_type=AnalyzerType.SICK_MARSIC,
        protocol=ConnectionProtocol.OPC_UA,
        host="192.168.1.100",
        port=4840,
    )

    assert isinstance(client, SCADAClient)
    assert client._config.analyzer_type == AnalyzerType.SICK_MARSIC
    assert client._config.protocol == ConnectionProtocol.OPC_UA


# =============================================================================
# Tag Management Tests
# =============================================================================


def test_register_single_tag(opcua_config, sample_flue_gas_tags):
    """Test registering single tag."""
    client = SCADAClient(opcua_config)
    tag = sample_flue_gas_tags[0]

    client.register_tag(tag)

    assert len(client._tags) == 1
    assert client.get_tag_definition(tag.tag_name) == tag


def test_register_multiple_tags(opcua_config, sample_flue_gas_tags):
    """Test registering multiple tags."""
    client = SCADAClient(opcua_config)

    client.register_tags(sample_flue_gas_tags)

    assert len(client._tags) == len(sample_flue_gas_tags)


def test_get_tags_by_location(opcua_config, sample_flue_gas_tags):
    """Test getting tags by location."""
    client = SCADAClient(opcua_config)
    client.register_tags(sample_flue_gas_tags)

    stack_tags = client.get_tags_by_location(MeasurementLocation.STACK_OUTLET)

    assert len(stack_tags) == 2
    assert all(tag.location == MeasurementLocation.STACK_OUTLET for tag in stack_tags)


def test_get_tags_by_parameter(opcua_config, sample_flue_gas_tags):
    """Test getting tags by parameter type."""
    client = SCADAClient(opcua_config)
    client.register_tags(sample_flue_gas_tags)

    o2_tags = client.get_tags_by_parameter(ParameterType.OXYGEN)

    assert len(o2_tags) == 1
    assert o2_tags[0].parameter_type == ParameterType.OXYGEN


# =============================================================================
# Connection Tests
# =============================================================================


@pytest.mark.asyncio
async def test_opcua_connection_success(opcua_config):
    """Test successful OPC-UA connection."""
    client = SCADAClient(opcua_config)

    with patch('integrations.scada_integration.Client') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.connect = AsyncMock()

        result = await client.connect()

        assert result is True
        assert client.is_connected()
        mock_client.connect.assert_called_once()


@pytest.mark.asyncio
async def test_modbus_connection_success(modbus_config):
    """Test successful Modbus connection."""
    client = SCADAClient(modbus_config)

    with patch('integrations.scada_integration.AsyncModbusTcpClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.connect = AsyncMock()
        mock_client.connected = True

        result = await client.connect()

        assert result is True
        assert client.is_connected()


@pytest.mark.asyncio
async def test_connection_failure_with_retry(opcua_config):
    """Test connection failure with retry logic."""
    client = SCADAClient(opcua_config)

    with patch('integrations.scada_integration.Client') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.connect = AsyncMock(side_effect=ConnectionError("Connection failed"))

        result = await client.connect()

        assert result is False
        assert not client.is_connected()
        assert client._reconnect_task is not None


@pytest.mark.asyncio
async def test_disconnect(opcua_config):
    """Test disconnection."""
    client = SCADAClient(opcua_config)
    client._connected = True
    client._client = AsyncMock()

    await client.disconnect()

    assert not client.is_connected()


# =============================================================================
# Tag Reading Tests
# =============================================================================


@pytest.mark.asyncio
async def test_read_opcua_tag(opcua_config, sample_flue_gas_tags):
    """Test reading OPC-UA tag."""
    client = SCADAClient(opcua_config)
    client.register_tag(sample_flue_gas_tags[0])
    client._connected = True

    with patch.object(client, '_read_opcua_tag', return_value=8192.0):
        data_point = await client.read_tag("FG_O2_STACK", use_cache=False)

        assert data_point.tag_name == "FG_O2_STACK"
        assert data_point.value == 12.5  # Scaled value
        assert data_point.engineering_unit == "%"
        assert data_point.quality == "GOOD"


@pytest.mark.asyncio
async def test_read_modbus_tag(modbus_config):
    """Test reading Modbus tag."""
    client = SCADAClient(modbus_config)

    tag = FlueGasTag(
        tag_name="30001",
        parameter_type=ParameterType.OXYGEN,
        location=MeasurementLocation.STACK_OUTLET,
        engineering_unit="%",
        raw_min=0,
        raw_max=20000,
        scaled_min=0.0,
        scaled_max=25.0,
    )

    client.register_tag(tag)
    client._connected = True

    with patch.object(client, '_read_modbus_tag', return_value=10000):
        data_point = await client.read_tag("30001", use_cache=False)

        assert data_point.tag_name == "30001"
        assert abs(data_point.value - 12.5) < 0.1


@pytest.mark.asyncio
async def test_read_with_caching(opcua_config, sample_flue_gas_tags):
    """Test tag reading with caching."""
    client = SCADAClient(opcua_config)
    client.register_tag(sample_flue_gas_tags[0])
    client._connected = True

    with patch.object(client, '_read_opcua_tag', return_value=8192.0) as mock_read:
        # First read
        data_point1 = await client.read_tag("FG_O2_STACK", use_cache=True)

        # Second read (should use cache)
        data_point2 = await client.read_tag("FG_O2_STACK", use_cache=True)

        # Should only read once
        assert mock_read.call_count == 1
        assert data_point1.value == data_point2.value


@pytest.mark.asyncio
async def test_read_multiple_tags(opcua_config, sample_flue_gas_tags):
    """Test reading multiple tags."""
    client = SCADAClient(opcua_config)
    client.register_tags(sample_flue_gas_tags)
    client._connected = True

    with patch.object(client, 'read_tag') as mock_read:
        mock_read.return_value = TagDataPoint(
            tag_name="test",
            value=10.0,
            engineering_unit="%",
        )

        tag_names = [tag.tag_name for tag in sample_flue_gas_tags]
        results = await client.read_tags(tag_names)

        assert len(results) == len(sample_flue_gas_tags)


@pytest.mark.asyncio
async def test_read_tag_not_connected(opcua_config, sample_flue_gas_tags):
    """Test reading tag when not connected."""
    client = SCADAClient(opcua_config)
    client.register_tag(sample_flue_gas_tags[0])

    with pytest.raises(ConnectionError, match="Not connected"):
        await client.read_tag("FG_O2_STACK")


# =============================================================================
# Tag Writing Tests
# =============================================================================


@pytest.mark.asyncio
async def test_write_opcua_tag(opcua_config, sample_flue_gas_tags):
    """Test writing to OPC-UA tag."""
    client = SCADAClient(opcua_config)
    damper_tag = sample_flue_gas_tags[2]  # AIR_DAMPER_POS
    client.register_tag(damper_tag)
    client._connected = True

    with patch.object(client, '_write_opcua_tag', return_value=True):
        result = await client.write_tag("AIR_DAMPER_POS", 50.0)

        assert result is True


@pytest.mark.asyncio
async def test_write_modbus_tag(modbus_config):
    """Test writing to Modbus tag."""
    client = SCADAClient(modbus_config)

    damper_tag = FlueGasTag(
        tag_name="40001",
        parameter_type=ParameterType.DAMPER_POSITION,
        location=MeasurementLocation.FORCED_DRAFT_FAN_OUTLET,
        tag_type=TagType.ANALOG_OUTPUT,
        engineering_unit="%",
        raw_min=0,
        raw_max=16384,
        scaled_min=0.0,
        scaled_max=100.0,
    )

    client.register_tag(damper_tag)
    client._connected = True

    with patch.object(client, '_write_modbus_tag', return_value=True):
        result = await client.write_tag("40001", 75.0)

        assert result is True


@pytest.mark.asyncio
async def test_write_buffering_when_disconnected(opcua_config, sample_flue_gas_tags):
    """Test write buffering when disconnected."""
    client = SCADAClient(opcua_config)
    damper_tag = sample_flue_gas_tags[2]
    client.register_tag(damper_tag)

    # Not connected
    result = await client.write_tag("AIR_DAMPER_POS", 50.0)

    assert result is False
    assert len(client._disconnect_buffer) == 1
    assert client._stats["buffered_points"] == 1


@pytest.mark.asyncio
async def test_write_readonly_tag(opcua_config, sample_flue_gas_tags):
    """Test writing to read-only tag."""
    client = SCADAClient(opcua_config)
    o2_tag = sample_flue_gas_tags[0]  # ANALOG_INPUT
    client.register_tag(o2_tag)
    client._connected = True

    with pytest.raises(ValueError, match="not writable"):
        await client.write_tag("FG_O2_STACK", 5.0)


# =============================================================================
# Value Scaling Tests
# =============================================================================


def test_value_scaling(opcua_config, sample_flue_gas_tags):
    """Test value scaling."""
    client = SCADAClient(opcua_config)
    tag = sample_flue_gas_tags[0]

    # Test scaling: 8192 (mid-point) should scale to 12.5%
    scaled = client._scale_value(8192.0, tag)
    assert abs(scaled - 12.5) < 0.1

    # Test minimum
    scaled = client._scale_value(0.0, tag)
    assert scaled == 0.0

    # Test maximum
    scaled = client._scale_value(16384.0, tag)
    assert scaled == 25.0


def test_boolean_scaling(opcua_config):
    """Test boolean value scaling."""
    client = SCADAClient(opcua_config)
    tag = FlueGasTag(
        tag_name="TEST_BOOL",
        parameter_type=ParameterType.DAMPER_POSITION,
        location=MeasurementLocation.STACK_OUTLET,
        tag_type=TagType.DIGITAL_INPUT,
        engineering_unit="bool",
        raw_min=0,
        raw_max=1,
        scaled_min=0,
        scaled_max=1,
    )

    assert client._scale_value(True, tag) == 1.0
    assert client._scale_value(False, tag) == 0.0


# =============================================================================
# Alarm Management Tests
# =============================================================================


def test_high_alarm_generation(opcua_config, sample_flue_gas_tags):
    """Test high alarm generation."""
    client = SCADAClient(opcua_config)
    tag = sample_flue_gas_tags[0]  # O2 with high_alarm_limit=10.0

    data_point = TagDataPoint(
        tag_name="FG_O2_STACK",
        value=12.0,  # Above high alarm limit
        engineering_unit="%",
    )

    client._check_tag_limits(data_point, tag)

    assert len(client.get_active_alarms()) == 1
    alarm = client.get_active_alarms()[0]
    assert alarm.severity == AlarmSeverity.CRITICAL
    assert alarm.tag_name == "FG_O2_STACK"


def test_low_alarm_generation(opcua_config, sample_flue_gas_tags):
    """Test low alarm generation."""
    client = SCADAClient(opcua_config)
    tag = sample_flue_gas_tags[0]  # O2 with low_alarm_limit=1.0

    data_point = TagDataPoint(
        tag_name="FG_O2_STACK",
        value=0.5,  # Below low alarm limit
        engineering_unit="%",
    )

    client._check_tag_limits(data_point, tag)

    assert len(client.get_active_alarms()) == 1
    alarm = client.get_active_alarms()[0]
    assert alarm.severity == AlarmSeverity.CRITICAL


def test_alarm_clearing(opcua_config, sample_flue_gas_tags):
    """Test alarm clearing."""
    client = SCADAClient(opcua_config)
    tag = sample_flue_gas_tags[0]

    # Generate alarm
    data_point_high = TagDataPoint(
        tag_name="FG_O2_STACK",
        value=12.0,
        engineering_unit="%",
    )
    client._check_tag_limits(data_point_high, tag)
    assert len(client.get_active_alarms()) == 1

    # Clear alarm
    data_point_normal = TagDataPoint(
        tag_name="FG_O2_STACK",
        value=5.0,  # Normal value
        engineering_unit="%",
    )
    client._check_tag_limits(data_point_normal, tag)
    assert len(client.get_active_alarms()) == 0


@pytest.mark.asyncio
async def test_alarm_acknowledgment(opcua_config, sample_flue_gas_tags):
    """Test alarm acknowledgment."""
    client = SCADAClient(opcua_config)
    tag = sample_flue_gas_tags[0]

    # Generate alarm
    data_point = TagDataPoint(
        tag_name="FG_O2_STACK",
        value=12.0,
        engineering_unit="%",
    )
    client._check_tag_limits(data_point, tag)

    alarms = client.get_active_alarms()
    alarm_id = alarms[0].alarm_id

    # Acknowledge
    result = await client.acknowledge_alarm(
        alarm_id,
        acknowledged_by="operator@example.com",
        notes="Investigating high O2"
    )

    assert result is True
    alarm = client._active_alarms[alarm_id]
    assert alarm.state == AlarmState.ACKNOWLEDGED
    assert alarm.acknowledged_by == "operator@example.com"


# =============================================================================
# Subscription Tests
# =============================================================================


@pytest.mark.asyncio
async def test_tag_subscription(opcua_config, sample_flue_gas_tags):
    """Test tag subscription."""
    client = SCADAClient(opcua_config)
    client.register_tag(sample_flue_gas_tags[0])
    client._connected = True

    callback_called = False
    received_value = None

    def callback(data_point: TagDataPoint):
        nonlocal callback_called, received_value
        callback_called = True
        received_value = data_point.value

    with patch.object(client, 'read_tag') as mock_read:
        mock_read.return_value = TagDataPoint(
            tag_name="FG_O2_STACK",
            value=5.5,
            engineering_unit="%",
        )

        await client.subscribe_tag("FG_O2_STACK", callback)

        # Wait for callback
        await asyncio.sleep(0.1)

        assert "FG_O2_STACK" in client._subscriptions


@pytest.mark.asyncio
async def test_tag_unsubscribe(opcua_config, sample_flue_gas_tags):
    """Test tag unsubscription."""
    client = SCADAClient(opcua_config)
    client.register_tag(sample_flue_gas_tags[0])
    client._connected = True

    def callback(data_point: TagDataPoint):
        pass

    with patch.object(client, 'read_tag'):
        await client.subscribe_tag("FG_O2_STACK", callback)
        assert "FG_O2_STACK" in client._subscriptions

        await client.unsubscribe_tag("FG_O2_STACK")
        assert "FG_O2_STACK" not in client._subscriptions


# =============================================================================
# Historical Data Tests
# =============================================================================


@pytest.mark.asyncio
async def test_historical_data_buffering(opcua_config, sample_flue_gas_tags):
    """Test historical data buffering."""
    client = SCADAClient(opcua_config)
    client.register_tag(sample_flue_gas_tags[0])
    client._connected = True

    with patch.object(client, '_read_opcua_tag', return_value=8192.0):
        # Read multiple times
        for _ in range(5):
            await client.read_tag("FG_O2_STACK", use_cache=False)
            await asyncio.sleep(0.01)

        # Check buffer
        buffer = client._historical_buffer["FG_O2_STACK"]
        assert len(buffer) == 5


@pytest.mark.asyncio
async def test_get_historical_data(opcua_config, sample_flue_gas_tags):
    """Test retrieving historical data."""
    client = SCADAClient(opcua_config)
    client.register_tag(sample_flue_gas_tags[0])

    # Add historical data
    now = datetime.now(timezone.utc)
    for i in range(10):
        data_point = TagDataPoint(
            tag_name="FG_O2_STACK",
            timestamp=now - timedelta(minutes=10-i),
            value=5.0 + i * 0.5,
            engineering_unit="%",
        )
        client._historical_buffer["FG_O2_STACK"].append(data_point)

    # Retrieve data
    start_time = now - timedelta(minutes=8)
    end_time = now - timedelta(minutes=2)

    historical = await client.get_historical_data(
        "FG_O2_STACK",
        start_time,
        end_time
    )

    assert len(historical) > 0
    assert all(start_time <= dp.timestamp <= end_time for dp in historical)


# =============================================================================
# Statistics and Health Tests
# =============================================================================


def test_get_statistics(opcua_config):
    """Test getting statistics."""
    client = SCADAClient(opcua_config)

    stats = client.get_statistics()

    assert "reads" in stats
    assert "writes" in stats
    assert "errors" in stats
    assert "connected" in stats
    assert stats["reads"] == 0
    assert stats["writes"] == 0


@pytest.mark.asyncio
async def test_health_check(opcua_config):
    """Test health check."""
    client = SCADAClient(opcua_config)

    health = await client.health_check()

    assert "healthy" in health
    assert "connected" in health
    assert "protocol" in health
    assert "analyzer_type" in health
    assert health["protocol"] == "opc_ua"


# =============================================================================
# Predefined Tag Tests
# =============================================================================


def test_create_standard_tags():
    """Test creating standard flue gas tags."""
    tags = create_standard_flue_gas_tags()

    assert len(tags) > 0
    assert any(tag.parameter_type == ParameterType.OXYGEN for tag in tags)
    assert any(tag.parameter_type == ParameterType.CARBON_MONOXIDE for tag in tags)
    assert any(tag.parameter_type == ParameterType.NOX for tag in tags)


def test_create_abb_tags():
    """Test creating ABB AO2000-specific tags."""
    tags = create_abb_ao2000_tags()

    assert len(tags) > 0
    assert any("AO2000" in tag.tag_name for tag in tags)


def test_create_sick_tags():
    """Test creating SICK MARSIC-specific tags."""
    tags = create_sick_marsic_tags()

    assert len(tags) > 0
    assert any("MARSIC" in tag.tag_name for tag in tags)


def test_create_horiba_tags():
    """Test creating Horiba PG-specific tags (Modbus)."""
    tags = create_horiba_pg_tags()

    assert len(tags) > 0
    # Horiba uses Modbus register addresses
    assert all(tag.tag_name.isdigit() for tag in tags)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_complete_workflow(opcua_config, sample_flue_gas_tags):
    """Test complete workflow: connect, read, write, disconnect."""
    client = SCADAClient(opcua_config)
    client.register_tags(sample_flue_gas_tags)

    with patch('integrations.scada_integration.Client') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.connect = AsyncMock()

        # Connect
        await client.connect()
        assert client.is_connected()

        # Read tags
        with patch.object(client, '_read_opcua_tag', return_value=8192.0):
            data = await client.read_tag("FG_O2_STACK")
            assert data.value > 0

        # Write tag
        with patch.object(client, '_write_opcua_tag', return_value=True):
            result = await client.write_tag("AIR_DAMPER_POS", 50.0)
            assert result is True

        # Check statistics
        stats = client.get_statistics()
        assert stats["reads"] > 0
        assert stats["writes"] > 0

        # Disconnect
        await client.disconnect()
        assert not client.is_connected()
