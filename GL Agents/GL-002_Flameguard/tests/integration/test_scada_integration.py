"""
GL-002 FLAMEGUARD - SCADA Integration Tests

Tests for SCADA/DCS protocol connectivity including:
- Modbus TCP communication
- OPC-UA connectivity
- Tag mapping and scaling
- Store-and-forward buffering
- Connection management
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import struct
import sys

sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])

from integration.scada_connector import (
    SCADAConnector,
    SCADAConnectionConfig,
    SCADAProtocol,
    ModbusTCPHandler,
    OPCUAHandler,
    TagMapping,
    TagValue,
    TagQuality,
    DataType,
    ProtocolHandler,
    create_boiler_tag_mappings,
)


# =============================================================================
# SCADA CONNECTION CONFIG TESTS
# =============================================================================


class TestSCADAConnectionConfig:
    """Test SCADA connection configuration."""

    def test_modbus_config(self):
        """Test Modbus TCP configuration."""
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="192.168.1.100",
            port=502,
        )

        assert config.protocol == SCADAProtocol.MODBUS_TCP
        assert config.host == "192.168.1.100"
        assert config.port == 502
        assert config.unit_id == 1  # Default

    def test_opcua_config(self):
        """Test OPC-UA configuration."""
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.OPC_UA,
            host="192.168.1.100",
            port=4840,
            endpoint_url="opc.tcp://192.168.1.100:4840",
            security_policy="Basic256Sha256",
        )

        assert config.protocol == SCADAProtocol.OPC_UA
        assert config.endpoint_url == "opc.tcp://192.168.1.100:4840"
        assert config.security_policy == "Basic256Sha256"

    def test_default_values(self):
        """Test default configuration values."""
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )

        assert config.timeout_ms == 5000
        assert config.retry_count == 3
        assert config.poll_interval_ms == 1000
        assert config.enable_store_forward is True
        assert config.buffer_size == 10000


class TestSCADAProtocolEnum:
    """Test SCADA protocol enumeration."""

    def test_protocol_values(self):
        """Test protocol enum values."""
        assert SCADAProtocol.MODBUS_TCP.value == "modbus_tcp"
        assert SCADAProtocol.MODBUS_RTU.value == "modbus_rtu"
        assert SCADAProtocol.OPC_UA.value == "opc_ua"
        assert SCADAProtocol.DNP3.value == "dnp3"


# =============================================================================
# TAG MAPPING TESTS
# =============================================================================


class TestTagMapping:
    """Test tag mapping functionality."""

    def test_default_tag_mapping(self):
        """Test default tag mapping values."""
        mapping = TagMapping(
            scada_tag="HR100",
            internal_name="drum_pressure",
        )

        assert mapping.scada_tag == "HR100"
        assert mapping.internal_name == "drum_pressure"
        assert mapping.data_type == DataType.FLOAT32
        assert mapping.scale_factor == 1.0
        assert mapping.offset == 0.0
        assert mapping.read_only is True

    def test_tag_mapping_with_scaling(self):
        """Test tag mapping with scaling."""
        mapping = TagMapping(
            scada_tag="HR200",
            internal_name="temperature",
            data_type=DataType.INT16,
            scale_factor=0.1,
            offset=-40.0,
            unit="degC",
        )

        assert mapping.scale_factor == 0.1
        assert mapping.offset == -40.0

    def test_tag_mapping_with_limits(self):
        """Test tag mapping with validation limits."""
        mapping = TagMapping(
            scada_tag="HR300",
            internal_name="pressure",
            low_limit=0.0,
            high_limit=200.0,
        )

        assert mapping.low_limit == 0.0
        assert mapping.high_limit == 200.0

    def test_writable_tag(self):
        """Test writable tag mapping."""
        mapping = TagMapping(
            scada_tag="HR500",
            internal_name="setpoint",
            read_only=False,
        )

        assert mapping.read_only is False


class TestBoilerTagMappings:
    """Test standard boiler tag mappings."""

    def test_create_mappings(self):
        """Test creating standard boiler mappings."""
        mappings = create_boiler_tag_mappings()

        assert len(mappings) > 0
        assert any(m.internal_name == "drum_pressure" for m in mappings)
        assert any(m.internal_name == "o2_percent" for m in mappings)
        assert any(m.internal_name == "o2_setpoint" for m in mappings)

    def test_mappings_have_descriptions(self):
        """Test mappings have descriptions."""
        mappings = create_boiler_tag_mappings()

        for mapping in mappings:
            assert mapping.description != "" or mapping.internal_name != ""

    def test_writable_setpoints(self):
        """Test setpoint tags are writable."""
        mappings = create_boiler_tag_mappings()

        o2_setpoint = next(m for m in mappings if m.internal_name == "o2_setpoint")
        assert o2_setpoint.read_only is False


# =============================================================================
# TAG VALUE TESTS
# =============================================================================


class TestTagValue:
    """Test TagValue dataclass."""

    def test_tag_value_creation(self):
        """Test creating tag value."""
        value = TagValue(
            tag="test_tag",
            value=123.45,
            quality=TagQuality.GOOD,
            timestamp=datetime.now(timezone.utc),
        )

        assert value.tag == "test_tag"
        assert value.value == 123.45
        assert value.quality == TagQuality.GOOD

    def test_tag_value_to_dict(self):
        """Test converting tag value to dict."""
        ts = datetime.now(timezone.utc)
        value = TagValue(
            tag="test_tag",
            value=100.0,
            quality=TagQuality.GOOD,
            timestamp=ts,
        )

        d = value.to_dict()

        assert d["tag"] == "test_tag"
        assert d["value"] == 100.0
        assert d["quality"] == "good"
        assert d["timestamp"] == ts.isoformat()


class TestTagQualityEnum:
    """Test tag quality enumeration."""

    def test_quality_values(self):
        """Test quality enum values."""
        assert TagQuality.GOOD.value == "good"
        assert TagQuality.BAD.value == "bad"
        assert TagQuality.UNCERTAIN.value == "uncertain"
        assert TagQuality.COMM_FAILURE.value == "comm_failure"


class TestDataTypeEnum:
    """Test data type enumeration."""

    def test_data_type_values(self):
        """Test data type enum values."""
        assert DataType.BOOL.value == "bool"
        assert DataType.INT16.value == "int16"
        assert DataType.INT32.value == "int32"
        assert DataType.FLOAT32.value == "float32"
        assert DataType.FLOAT64.value == "float64"


# =============================================================================
# MODBUS TCP HANDLER TESTS
# =============================================================================


class TestModbusTCPHandler:
    """Test Modbus TCP protocol handler."""

    @pytest.fixture
    def modbus_config(self):
        return SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
            timeout_ms=1000,
        )

    @pytest.fixture
    def handler(self, modbus_config):
        return ModbusTCPHandler(modbus_config)

    def test_initialization(self, handler):
        """Test handler initialization."""
        assert handler._connected is False
        assert handler._transaction_id == 0

    def test_add_register(self, handler):
        """Test adding register mapping."""
        handler.add_register("HR100", 100, 2, DataType.FLOAT32)

        assert "HR100" in handler._register_map
        assert handler._register_map["HR100"] == (100, 2, DataType.FLOAT32)

    @pytest.mark.asyncio
    async def test_connect_success(self, handler):
        """Test successful connection."""
        with patch("asyncio.open_connection") as mock_open:
            mock_reader = AsyncMock()
            mock_writer = MagicMock()
            mock_open.return_value = (mock_reader, mock_writer)

            result = await handler.connect()

            assert result is True
            assert handler._connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, handler):
        """Test connection failure."""
        with patch("asyncio.open_connection") as mock_open:
            mock_open.side_effect = ConnectionRefusedError()

            result = await handler.connect()

            assert result is False
            assert handler._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, handler):
        """Test disconnection."""
        # Set up connected state
        handler._connected = True
        handler._writer = MagicMock()
        handler._writer.close = MagicMock()
        handler._writer.wait_closed = AsyncMock()

        await handler.disconnect()

        assert handler._connected is False

    @pytest.mark.asyncio
    async def test_read_tags_not_connected(self, handler):
        """Test reading tags when not connected."""
        handler.add_register("HR100", 100, 2, DataType.FLOAT32)

        result = await handler.read_tags(["HR100"])

        assert result["HR100"].quality == TagQuality.NOT_CONNECTED

    @pytest.mark.asyncio
    async def test_read_tags_unknown_tag(self, handler):
        """Test reading unknown tag."""
        handler._connected = True

        result = await handler.read_tags(["UNKNOWN"])

        assert result["UNKNOWN"].quality == TagQuality.CONFIG_ERROR

    def test_is_connected(self, handler):
        """Test connection status check."""
        assert handler.is_connected() is False

        handler._connected = True
        assert handler.is_connected() is True


class TestModbusValueConversion:
    """Test Modbus value conversion."""

    @pytest.fixture
    def handler(self):
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        return ModbusTCPHandler(config)

    def test_convert_bool(self, handler):
        """Test boolean conversion."""
        result = handler._convert_value([1], DataType.BOOL)
        assert result is True

        result = handler._convert_value([0], DataType.BOOL)
        assert result is False

    def test_convert_int16(self, handler):
        """Test INT16 conversion."""
        # 0x7FFF = 32767
        result = handler._convert_value([0x7FFF], DataType.INT16)
        assert result == 32767

        # 0xFFFF = -1 (signed)
        result = handler._convert_value([0xFFFF], DataType.INT16)
        assert result == -1

    def test_convert_uint16(self, handler):
        """Test UINT16 conversion."""
        result = handler._convert_value([0xFFFF], DataType.UINT16)
        assert result == 65535

    def test_convert_float32(self, handler):
        """Test FLOAT32 conversion."""
        # Pack 123.456 as big-endian float
        packed = struct.pack(">f", 123.456)
        registers = struct.unpack(">HH", packed)

        result = handler._convert_value(list(registers), DataType.FLOAT32)
        assert abs(result - 123.456) < 0.001

    def test_value_to_registers_float32(self, handler):
        """Test converting float to registers."""
        registers = handler._value_to_registers(123.456, DataType.FLOAT32)

        assert len(registers) == 2

        # Convert back
        packed = struct.pack(">HH", *registers)
        value = struct.unpack(">f", packed)[0]
        assert abs(value - 123.456) < 0.001

    def test_value_to_registers_int16(self, handler):
        """Test converting int16 to registers."""
        registers = handler._value_to_registers(-100, DataType.INT16)

        assert len(registers) == 1


# =============================================================================
# OPC-UA HANDLER TESTS
# =============================================================================


class TestOPCUAHandler:
    """Test OPC-UA protocol handler."""

    @pytest.fixture
    def opcua_config(self):
        return SCADAConnectionConfig(
            protocol=SCADAProtocol.OPC_UA,
            host="localhost",
            port=4840,
            endpoint_url="opc.tcp://localhost:4840",
        )

    @pytest.fixture
    def handler(self, opcua_config):
        return OPCUAHandler(opcua_config)

    def test_initialization(self, handler):
        """Test handler initialization."""
        assert handler._connected is False
        assert handler._node_map == {}

    def test_add_node(self, handler):
        """Test adding node mapping."""
        handler.add_node("temperature", "ns=2;s=Temperature")

        assert "temperature" in handler._node_map

    @pytest.mark.asyncio
    async def test_connect(self, handler):
        """Test connection (stub implementation)."""
        result = await handler.connect()

        assert result is True
        assert handler._connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, handler):
        """Test disconnection."""
        handler._connected = True

        await handler.disconnect()

        assert handler._connected is False

    @pytest.mark.asyncio
    async def test_read_tags(self, handler):
        """Test reading tags (stub implementation)."""
        handler._connected = True

        result = await handler.read_tags(["temperature"])

        assert "temperature" in result
        assert result["temperature"].quality == TagQuality.GOOD

    @pytest.mark.asyncio
    async def test_read_tags_not_connected(self, handler):
        """Test reading when not connected."""
        handler._connected = False

        result = await handler.read_tags(["temperature"])

        assert result["temperature"].quality == TagQuality.NOT_CONNECTED

    @pytest.mark.asyncio
    async def test_write_tag(self, handler):
        """Test writing tag (stub implementation)."""
        handler._connected = True

        result = await handler.write_tag("setpoint", 100.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_write_tag_not_connected(self, handler):
        """Test writing when not connected."""
        handler._connected = False

        result = await handler.write_tag("setpoint", 100.0)

        assert result is False


# =============================================================================
# SCADA CONNECTOR TESTS
# =============================================================================


class TestSCADAConnector:
    """Test main SCADA connector class."""

    @pytest.fixture
    def modbus_connector(self):
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        return SCADAConnector(config)

    @pytest.fixture
    def opcua_connector(self):
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.OPC_UA,
            host="localhost",
            port=4840,
            endpoint_url="opc.tcp://localhost:4840",
        )
        return SCADAConnector(config)

    def test_modbus_initialization(self, modbus_connector):
        """Test Modbus connector initialization."""
        assert isinstance(modbus_connector._handler, ModbusTCPHandler)

    def test_opcua_initialization(self, opcua_connector):
        """Test OPC-UA connector initialization."""
        assert isinstance(opcua_connector._handler, OPCUAHandler)

    def test_add_tag(self, modbus_connector):
        """Test adding tag mapping."""
        mapping = TagMapping(
            scada_tag="HR100",
            internal_name="pressure",
            data_type=DataType.FLOAT32,
        )

        modbus_connector.add_tag(mapping)

        assert "pressure" in modbus_connector._tag_mappings

    @pytest.mark.asyncio
    async def test_connect(self, modbus_connector):
        """Test connector connect."""
        with patch.object(modbus_connector._handler, "connect", return_value=True):
            result = await modbus_connector.connect()

            assert result is True

    @pytest.mark.asyncio
    async def test_disconnect(self, modbus_connector):
        """Test connector disconnect."""
        modbus_connector._handler.disconnect = AsyncMock()

        await modbus_connector.disconnect()

        modbus_connector._handler.disconnect.assert_called_once()

    def test_is_connected(self, modbus_connector):
        """Test connection status."""
        modbus_connector._handler._connected = True

        assert modbus_connector.is_connected() is True


class TestSCADAConnectorPolling:
    """Test SCADA connector polling functionality."""

    @pytest.fixture
    def connector(self):
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
            poll_interval_ms=100,
        )
        return SCADAConnector(config)

    @pytest.mark.asyncio
    async def test_start_polling(self, connector):
        """Test starting polling loop."""
        await connector.start_polling()

        assert connector._polling is True
        assert connector._poll_task is not None

        await connector.stop_polling()

    @pytest.mark.asyncio
    async def test_stop_polling(self, connector):
        """Test stopping polling loop."""
        await connector.start_polling()
        await connector.stop_polling()

        assert connector._polling is False

    @pytest.mark.asyncio
    async def test_double_start_polling(self, connector):
        """Test starting polling twice."""
        await connector.start_polling()
        task1 = connector._poll_task

        await connector.start_polling()
        task2 = connector._poll_task

        # Should be same task
        assert task1 == task2

        await connector.stop_polling()


class TestSCADAConnectorScaling:
    """Test SCADA connector value scaling."""

    @pytest.fixture
    def connector(self):
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        connector = SCADAConnector(config)

        # Add tag with scaling
        mapping = TagMapping(
            scada_tag="HR100",
            internal_name="temperature",
            scale_factor=0.1,
            offset=-40.0,
        )
        connector.add_tag(mapping)

        return connector

    def test_process_values_scaling(self, connector):
        """Test value scaling during processing."""
        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=500,  # Raw value
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        }

        processed = connector._process_values(raw_values)

        # Expected: 500 * 0.1 + (-40) = 10
        assert processed["temperature"].value == 10.0

    def test_process_values_quality_change(self, connector):
        """Test quality change callback."""
        quality_changes = []

        def on_quality_change(name, quality):
            quality_changes.append((name, quality))

        connector._on_quality_change = on_quality_change

        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=500,
                quality=TagQuality.BAD,
                timestamp=datetime.now(timezone.utc),
            )
        }

        connector._process_values(raw_values)

        assert len(quality_changes) == 1
        assert quality_changes[0][1] == TagQuality.BAD


class TestSCADAConnectorValidation:
    """Test SCADA connector value validation."""

    @pytest.fixture
    def connector(self):
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        connector = SCADAConnector(config)

        mapping = TagMapping(
            scada_tag="HR100",
            internal_name="pressure",
            low_limit=0.0,
            high_limit=200.0,
        )
        connector.add_tag(mapping)

        return connector

    def test_value_below_limit(self, connector):
        """Test value below low limit."""
        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=-10.0,  # Below limit
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        }

        processed = connector._process_values(raw_values)

        assert processed["pressure"].quality == TagQuality.SENSOR_FAILURE

    def test_value_above_limit(self, connector):
        """Test value above high limit."""
        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=250.0,  # Above limit
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        }

        processed = connector._process_values(raw_values)

        assert processed["pressure"].quality == TagQuality.SENSOR_FAILURE


class TestSCADAConnectorDeadband:
    """Test SCADA connector deadband filtering."""

    @pytest.fixture
    def connector(self):
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        connector = SCADAConnector(config)

        mapping = TagMapping(
            scada_tag="HR100",
            internal_name="temperature",
            deadband=1.0,
        )
        connector.add_tag(mapping)

        return connector

    def test_within_deadband_filtered(self, connector):
        """Test values within deadband are filtered."""
        # First value
        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=100.0,
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        }
        connector._process_values(raw_values)

        # Second value within deadband
        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=100.5,  # Within 1.0 deadband
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        }
        processed = connector._process_values(raw_values)

        # Should be empty (filtered out)
        assert "temperature" not in processed

    def test_outside_deadband_passed(self, connector):
        """Test values outside deadband are passed."""
        # First value
        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=100.0,
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        }
        connector._process_values(raw_values)

        # Second value outside deadband
        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=102.0,  # Outside 1.0 deadband
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        }
        processed = connector._process_values(raw_values)

        assert "temperature" in processed


class TestSCADAConnectorReadWrite:
    """Test SCADA connector read/write operations."""

    @pytest.fixture
    def connector(self):
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        connector = SCADAConnector(config)

        # Read-only tag
        connector.add_tag(TagMapping(
            scada_tag="HR100",
            internal_name="measurement",
            read_only=True,
        ))

        # Writable tag
        connector.add_tag(TagMapping(
            scada_tag="HR200",
            internal_name="setpoint",
            read_only=False,
        ))

        return connector

    @pytest.mark.asyncio
    async def test_read_single_tag(self, connector):
        """Test reading single tag."""
        connector._handler.read_tags = AsyncMock(return_value={
            "HR100": TagValue(
                tag="HR100",
                value=75.0,
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        })

        result = await connector.read_tag("measurement")

        assert result is not None
        assert result.value == 75.0

    @pytest.mark.asyncio
    async def test_read_unknown_tag(self, connector):
        """Test reading unknown tag."""
        result = await connector.read_tag("unknown")

        assert result is None

    @pytest.mark.asyncio
    async def test_write_tag_readonly(self, connector):
        """Test writing to read-only tag fails."""
        connector._handler._connected = True

        result = await connector.write_tag("measurement", 100.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_write_tag_writable(self, connector):
        """Test writing to writable tag."""
        connector._handler.write_tag = AsyncMock(return_value=True)
        connector.read_tag = AsyncMock(return_value=TagValue(
            tag="setpoint",
            value=100.0,
            quality=TagQuality.GOOD,
            timestamp=datetime.now(timezone.utc),
        ))

        result = await connector.write_tag("setpoint", 100.0, verify=True)

        assert result is True


class TestSCADAConnectorStatistics:
    """Test SCADA connector statistics."""

    @pytest.fixture
    def connector(self):
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        return SCADAConnector(config)

    def test_initial_statistics(self, connector):
        """Test initial statistics values."""
        stats = connector.get_statistics()

        assert stats["reads"] == 0
        assert stats["writes"] == 0
        assert stats["errors"] == 0
        assert stats["reconnects"] == 0

    def test_get_tag_status(self, connector):
        """Test getting tag status."""
        connector.add_tag(TagMapping(
            scada_tag="HR100",
            internal_name="temperature",
        ))

        status = connector.get_tag_status()

        assert "temperature" in status
        assert "scada_tag" in status["temperature"]


# =============================================================================
# STORE AND FORWARD TESTS
# =============================================================================


class TestStoreAndForward:
    """Test store-and-forward buffer functionality."""

    @pytest.fixture
    def connector(self):
        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
            enable_store_forward=True,
            buffer_size=100,
        )
        return SCADAConnector(config)

    @pytest.mark.asyncio
    async def test_buffer_values(self, connector):
        """Test values are buffered."""
        values = {
            "test": TagValue(
                tag="test",
                value=100.0,
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        }

        await connector._buffer_values(values)

        assert len(connector._buffer) == 1

    @pytest.mark.asyncio
    async def test_buffer_trimming(self, connector):
        """Test buffer trimming at max size."""
        for i in range(150):
            values = {
                "test": TagValue(
                    tag="test",
                    value=float(i),
                    quality=TagQuality.GOOD,
                    timestamp=datetime.now(timezone.utc),
                )
            }
            await connector._buffer_values(values)

        # Should be trimmed to 100
        assert len(connector._buffer) == 100
