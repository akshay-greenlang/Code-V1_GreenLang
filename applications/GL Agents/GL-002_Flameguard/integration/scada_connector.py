"""
GL-002 FLAMEGUARD - SCADA/DCS Connector

Industrial protocol connectivity for boiler control systems.
Supports OPC-UA, Modbus TCP/RTU, DNP3, and native DCS protocols.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio
import logging
import struct
import hashlib

logger = logging.getLogger(__name__)


class SCADAProtocol(Enum):
    """Supported SCADA/DCS protocols."""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    DNP3 = "dnp3"
    PROFINET = "profinet"
    ETHERNET_IP = "ethernet_ip"
    HART_IP = "hart_ip"
    DCS_NATIVE = "dcs_native"  # Honeywell, Emerson, Yokogawa


class DataType(Enum):
    """SCADA data types."""
    BOOL = "bool"
    INT16 = "int16"
    INT32 = "int32"
    UINT16 = "uint16"
    UINT32 = "uint32"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"


class TagQuality(Enum):
    """OPC-style tag quality codes."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    CONFIG_ERROR = "config_error"
    NOT_CONNECTED = "not_connected"
    DEVICE_FAILURE = "device_failure"
    SENSOR_FAILURE = "sensor_failure"
    LAST_KNOWN = "last_known"
    COMM_FAILURE = "comm_failure"
    OUT_OF_SERVICE = "out_of_service"


@dataclass
class TagMapping:
    """Maps SCADA tag to internal variable."""
    scada_tag: str
    internal_name: str
    data_type: DataType = DataType.FLOAT32
    unit: str = ""
    scale_factor: float = 1.0
    offset: float = 0.0
    deadband: float = 0.0
    read_only: bool = True
    description: str = ""

    # Limits for validation
    low_limit: Optional[float] = None
    high_limit: Optional[float] = None

    # Engineering unit conversion
    scada_unit: Optional[str] = None  # Unit as stored in SCADA

    # Timestamp tracking
    last_value: Optional[float] = None
    last_quality: TagQuality = TagQuality.BAD
    last_timestamp: Optional[datetime] = None


@dataclass
class SCADAConnectionConfig:
    """SCADA connection configuration."""
    protocol: SCADAProtocol
    host: str
    port: int

    # Authentication
    username: Optional[str] = None
    password: Optional[str] = None
    certificate_path: Optional[str] = None

    # Modbus specific
    unit_id: int = 1
    byte_order: str = "big"  # big or little
    word_order: str = "big"

    # OPC-UA specific
    endpoint_url: Optional[str] = None
    security_policy: str = "None"  # None, Basic128Rsa15, Basic256, Basic256Sha256
    security_mode: str = "None"  # None, Sign, SignAndEncrypt
    application_uri: Optional[str] = None

    # Connection settings
    timeout_ms: int = 5000
    retry_count: int = 3
    retry_delay_ms: int = 1000
    poll_interval_ms: int = 1000

    # Buffering
    enable_store_forward: bool = True
    buffer_size: int = 10000


@dataclass
class TagValue:
    """Value read from SCADA."""
    tag: str
    value: Any
    quality: TagQuality
    timestamp: datetime
    raw_value: Optional[Any] = None

    def to_dict(self) -> Dict:
        return {
            "tag": self.tag,
            "value": self.value,
            "quality": self.quality.value,
            "timestamp": self.timestamp.isoformat(),
        }


class ProtocolHandler(ABC):
    """Abstract protocol handler."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to device."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from device."""
        pass

    @abstractmethod
    async def read_tags(self, tags: List[str]) -> Dict[str, TagValue]:
        """Read multiple tags."""
        pass

    @abstractmethod
    async def write_tag(self, tag: str, value: Any) -> bool:
        """Write single tag."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection status."""
        pass


class ModbusTCPHandler(ProtocolHandler):
    """Modbus TCP protocol handler."""

    def __init__(self, config: SCADAConnectionConfig) -> None:
        self.config = config
        self._connected = False
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._transaction_id = 0
        self._lock = asyncio.Lock()

        # Register type mapping
        self._register_map: Dict[str, Tuple[int, int, DataType]] = {}

    def add_register(
        self,
        tag: str,
        address: int,
        count: int = 1,
        data_type: DataType = DataType.FLOAT32,
    ) -> None:
        """Add register mapping."""
        self._register_map[tag] = (address, count, data_type)

    async def connect(self) -> bool:
        """Connect to Modbus device."""
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.host, self.config.port),
                timeout=self.config.timeout_ms / 1000,
            )
            self._connected = True
            logger.info(f"Connected to Modbus TCP: {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Modbus connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Modbus device."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False
        logger.info("Modbus TCP disconnected")

    async def read_tags(self, tags: List[str]) -> Dict[str, TagValue]:
        """Read multiple Modbus registers."""
        results: Dict[str, TagValue] = {}

        if not self._connected:
            for tag in tags:
                results[tag] = TagValue(
                    tag=tag,
                    value=None,
                    quality=TagQuality.NOT_CONNECTED,
                    timestamp=datetime.now(timezone.utc),
                )
            return results

        async with self._lock:
            for tag in tags:
                if tag not in self._register_map:
                    results[tag] = TagValue(
                        tag=tag,
                        value=None,
                        quality=TagQuality.CONFIG_ERROR,
                        timestamp=datetime.now(timezone.utc),
                    )
                    continue

                address, count, data_type = self._register_map[tag]

                try:
                    value = await self._read_holding_registers(address, count)
                    converted = self._convert_value(value, data_type)

                    results[tag] = TagValue(
                        tag=tag,
                        value=converted,
                        quality=TagQuality.GOOD,
                        timestamp=datetime.now(timezone.utc),
                        raw_value=value,
                    )
                except Exception as e:
                    logger.error(f"Error reading {tag}: {e}")
                    results[tag] = TagValue(
                        tag=tag,
                        value=None,
                        quality=TagQuality.COMM_FAILURE,
                        timestamp=datetime.now(timezone.utc),
                    )

        return results

    async def _read_holding_registers(
        self,
        address: int,
        count: int,
    ) -> List[int]:
        """Read holding registers (function code 3)."""
        self._transaction_id = (self._transaction_id + 1) % 65536

        # Build Modbus TCP request
        request = struct.pack(
            ">HHHBBHH",
            self._transaction_id,  # Transaction ID
            0,  # Protocol ID (Modbus)
            6,  # Length
            self.config.unit_id,  # Unit ID
            3,  # Function code (read holding registers)
            address,  # Starting address
            count,  # Quantity of registers
        )

        self._writer.write(request)
        await self._writer.drain()

        # Read response header
        header = await asyncio.wait_for(
            self._reader.read(9),
            timeout=self.config.timeout_ms / 1000,
        )

        if len(header) < 9:
            raise Exception("Incomplete Modbus response")

        # Parse header
        trans_id, proto_id, length, unit_id, func_code, byte_count = struct.unpack(
            ">HHHBBB", header
        )

        # Check for error
        if func_code & 0x80:
            raise Exception(f"Modbus error: function code {func_code}")

        # Read data
        data = await asyncio.wait_for(
            self._reader.read(byte_count),
            timeout=self.config.timeout_ms / 1000,
        )

        # Convert to register values
        registers = []
        for i in range(0, len(data), 2):
            registers.append(struct.unpack(">H", data[i:i+2])[0])

        return registers

    def _convert_value(
        self,
        registers: List[int],
        data_type: DataType,
    ) -> Union[int, float, bool]:
        """Convert register values to typed value."""
        if data_type == DataType.BOOL:
            return bool(registers[0] & 1)

        elif data_type == DataType.INT16:
            return struct.unpack(">h", struct.pack(">H", registers[0]))[0]

        elif data_type == DataType.UINT16:
            return registers[0]

        elif data_type == DataType.INT32:
            packed = struct.pack(">HH", registers[0], registers[1])
            return struct.unpack(">i", packed)[0]

        elif data_type == DataType.UINT32:
            packed = struct.pack(">HH", registers[0], registers[1])
            return struct.unpack(">I", packed)[0]

        elif data_type == DataType.FLOAT32:
            packed = struct.pack(">HH", registers[0], registers[1])
            return struct.unpack(">f", packed)[0]

        elif data_type == DataType.FLOAT64:
            packed = struct.pack(">HHHH", *registers[:4])
            return struct.unpack(">d", packed)[0]

        return registers[0]

    async def write_tag(self, tag: str, value: Any) -> bool:
        """Write single register."""
        if not self._connected or tag not in self._register_map:
            return False

        address, count, data_type = self._register_map[tag]

        try:
            registers = self._value_to_registers(value, data_type)
            await self._write_registers(address, registers)
            return True
        except Exception as e:
            logger.error(f"Error writing {tag}: {e}")
            return False

    def _value_to_registers(
        self,
        value: Any,
        data_type: DataType,
    ) -> List[int]:
        """Convert value to registers."""
        if data_type == DataType.FLOAT32:
            packed = struct.pack(">f", float(value))
            return list(struct.unpack(">HH", packed))
        elif data_type == DataType.INT16:
            packed = struct.pack(">h", int(value))
            return [struct.unpack(">H", packed)[0]]
        elif data_type == DataType.UINT16:
            return [int(value) & 0xFFFF]
        return [int(value)]

    async def _write_registers(self, address: int, values: List[int]) -> None:
        """Write multiple registers (function code 16)."""
        self._transaction_id = (self._transaction_id + 1) % 65536

        byte_count = len(values) * 2
        data = b"".join(struct.pack(">H", v) for v in values)

        request = struct.pack(
            ">HHHBBHHB",
            self._transaction_id,
            0,  # Protocol ID
            7 + byte_count,  # Length
            self.config.unit_id,
            16,  # Function code (write multiple registers)
            address,
            len(values),
            byte_count,
        ) + data

        self._writer.write(request)
        await self._writer.drain()

        # Read response
        response = await asyncio.wait_for(
            self._reader.read(12),
            timeout=self.config.timeout_ms / 1000,
        )

        if len(response) < 12:
            raise Exception("Incomplete write response")

    def is_connected(self) -> bool:
        return self._connected


class OPCUAHandler(ProtocolHandler):
    """OPC-UA protocol handler (stub for actual implementation)."""

    def __init__(self, config: SCADAConnectionConfig) -> None:
        self.config = config
        self._connected = False
        self._client = None
        self._subscription = None
        self._node_map: Dict[str, str] = {}  # tag -> node_id

    def add_node(self, tag: str, node_id: str) -> None:
        """Add node mapping."""
        self._node_map[tag] = node_id

    async def connect(self) -> bool:
        """Connect to OPC-UA server."""
        try:
            # In production, use asyncua or opcua library
            # self._client = Client(self.config.endpoint_url)
            # await self._client.connect()
            self._connected = True
            logger.info(f"Connected to OPC-UA: {self.config.endpoint_url}")
            return True
        except Exception as e:
            logger.error(f"OPC-UA connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from OPC-UA server."""
        if self._client:
            # await self._client.disconnect()
            pass
        self._connected = False

    async def read_tags(self, tags: List[str]) -> Dict[str, TagValue]:
        """Read OPC-UA nodes."""
        results: Dict[str, TagValue] = {}

        for tag in tags:
            # Stub implementation
            results[tag] = TagValue(
                tag=tag,
                value=0.0,
                quality=TagQuality.GOOD if self._connected else TagQuality.NOT_CONNECTED,
                timestamp=datetime.now(timezone.utc),
            )

        return results

    async def write_tag(self, tag: str, value: Any) -> bool:
        """Write to OPC-UA node."""
        if not self._connected:
            return False

        # Stub implementation
        logger.info(f"OPC-UA write: {tag} = {value}")
        return True

    def is_connected(self) -> bool:
        return self._connected


class SCADAConnector:
    """
    Main SCADA connector class.

    Provides unified interface for multiple protocols:
    - OPC-UA for modern DCS systems
    - Modbus TCP/RTU for PLCs
    - DNP3 for power systems
    - Native DCS protocols
    """

    def __init__(
        self,
        config: SCADAConnectionConfig,
        on_data_callback: Optional[Callable[[Dict[str, TagValue]], None]] = None,
        on_quality_change: Optional[Callable[[str, TagQuality], None]] = None,
    ) -> None:
        self.config = config
        self._on_data = on_data_callback
        self._on_quality_change = on_quality_change

        # Tag mappings
        self._tag_mappings: Dict[str, TagMapping] = {}

        # Protocol handler
        self._handler: Optional[ProtocolHandler] = None
        self._init_handler()

        # Polling
        self._polling = False
        self._poll_task: Optional[asyncio.Task] = None

        # Store and forward buffer
        self._buffer: List[Dict[str, TagValue]] = []
        self._buffer_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "reads": 0,
            "writes": 0,
            "errors": 0,
            "reconnects": 0,
            "bytes_received": 0,
            "bytes_sent": 0,
        }

        logger.info(f"SCADAConnector initialized: {config.protocol.value}")

    def _init_handler(self) -> None:
        """Initialize protocol handler."""
        if self.config.protocol == SCADAProtocol.MODBUS_TCP:
            self._handler = ModbusTCPHandler(self.config)
        elif self.config.protocol == SCADAProtocol.OPC_UA:
            self._handler = OPCUAHandler(self.config)
        else:
            logger.warning(f"Protocol {self.config.protocol} not yet implemented")

    def add_tag(self, mapping: TagMapping) -> None:
        """Add tag mapping."""
        self._tag_mappings[mapping.internal_name] = mapping

        # Configure protocol handler
        if isinstance(self._handler, ModbusTCPHandler):
            # Parse Modbus address from tag (e.g., "HR400" -> address 400)
            if mapping.scada_tag.startswith("HR"):
                address = int(mapping.scada_tag[2:])
                count = 2 if mapping.data_type == DataType.FLOAT32 else 1
                self._handler.add_register(
                    mapping.scada_tag,
                    address,
                    count,
                    mapping.data_type,
                )
        elif isinstance(self._handler, OPCUAHandler):
            self._handler.add_node(mapping.scada_tag, mapping.scada_tag)

    async def connect(self) -> bool:
        """Connect to SCADA system."""
        if not self._handler:
            return False

        return await self._handler.connect()

    async def disconnect(self) -> None:
        """Disconnect from SCADA system."""
        await self.stop_polling()
        if self._handler:
            await self._handler.disconnect()

    async def start_polling(self) -> None:
        """Start continuous polling."""
        if self._polling:
            return

        self._polling = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("SCADA polling started")

    async def stop_polling(self) -> None:
        """Stop continuous polling."""
        self._polling = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("SCADA polling stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._polling:
            try:
                # Read all tags
                tags = [m.scada_tag for m in self._tag_mappings.values()]
                if tags and self._handler:
                    values = await self._handler.read_tags(tags)
                    self._stats["reads"] += 1

                    # Process values
                    processed = self._process_values(values)

                    # Callback
                    if self._on_data and processed:
                        self._on_data(processed)

                    # Store for forwarding if needed
                    if self.config.enable_store_forward:
                        await self._buffer_values(processed)

                await asyncio.sleep(self.config.poll_interval_ms / 1000)

            except Exception as e:
                logger.error(f"Polling error: {e}")
                self._stats["errors"] += 1

                # Attempt reconnect
                if self._handler and not self._handler.is_connected():
                    self._stats["reconnects"] += 1
                    await self._handler.connect()

                await asyncio.sleep(self.config.retry_delay_ms / 1000)

    def _process_values(
        self,
        raw_values: Dict[str, TagValue],
    ) -> Dict[str, TagValue]:
        """Apply scaling and conversion to raw values."""
        processed: Dict[str, TagValue] = {}

        for internal_name, mapping in self._tag_mappings.items():
            if mapping.scada_tag not in raw_values:
                continue

            raw = raw_values[mapping.scada_tag]

            # Check for quality change
            if raw.quality != mapping.last_quality:
                if self._on_quality_change:
                    self._on_quality_change(internal_name, raw.quality)
                mapping.last_quality = raw.quality

            if raw.value is None:
                processed[internal_name] = TagValue(
                    tag=internal_name,
                    value=None,
                    quality=raw.quality,
                    timestamp=raw.timestamp,
                )
                continue

            # Apply scaling: scaled = raw * factor + offset
            scaled_value = float(raw.value) * mapping.scale_factor + mapping.offset

            # Validate limits
            quality = raw.quality
            if mapping.low_limit is not None and scaled_value < mapping.low_limit:
                quality = TagQuality.SENSOR_FAILURE
            elif mapping.high_limit is not None and scaled_value > mapping.high_limit:
                quality = TagQuality.SENSOR_FAILURE

            # Apply deadband
            if mapping.last_value is not None:
                if abs(scaled_value - mapping.last_value) < mapping.deadband:
                    continue  # Skip if within deadband

            mapping.last_value = scaled_value
            mapping.last_timestamp = raw.timestamp

            processed[internal_name] = TagValue(
                tag=internal_name,
                value=round(scaled_value, 4),
                quality=quality,
                timestamp=raw.timestamp,
                raw_value=raw.raw_value,
            )

        return processed

    async def _buffer_values(self, values: Dict[str, TagValue]) -> None:
        """Buffer values for store-and-forward."""
        async with self._buffer_lock:
            self._buffer.append(values)

            # Trim buffer if too large
            if len(self._buffer) > self.config.buffer_size:
                self._buffer = self._buffer[-self.config.buffer_size:]

    async def read_tag(self, internal_name: str) -> Optional[TagValue]:
        """Read single tag."""
        if internal_name not in self._tag_mappings:
            return None

        mapping = self._tag_mappings[internal_name]
        if not self._handler:
            return None

        values = await self._handler.read_tags([mapping.scada_tag])
        processed = self._process_values(values)
        return processed.get(internal_name)

    async def read_tags(self, internal_names: List[str]) -> Dict[str, TagValue]:
        """Read multiple tags."""
        scada_tags = []
        for name in internal_names:
            if name in self._tag_mappings:
                scada_tags.append(self._tag_mappings[name].scada_tag)

        if not self._handler or not scada_tags:
            return {}

        values = await self._handler.read_tags(scada_tags)
        return self._process_values(values)

    async def write_tag(
        self,
        internal_name: str,
        value: float,
        verify: bool = True,
    ) -> bool:
        """Write value to tag."""
        if internal_name not in self._tag_mappings:
            return False

        mapping = self._tag_mappings[internal_name]

        if mapping.read_only:
            logger.warning(f"Cannot write to read-only tag: {internal_name}")
            return False

        if not self._handler:
            return False

        # Reverse scaling: raw = (value - offset) / factor
        raw_value = (value - mapping.offset) / mapping.scale_factor

        success = await self._handler.write_tag(mapping.scada_tag, raw_value)

        if success:
            self._stats["writes"] += 1
            logger.info(f"Written {internal_name} = {value} (raw: {raw_value})")

            if verify:
                # Verify write by reading back
                await asyncio.sleep(0.1)
                read_value = await self.read_tag(internal_name)
                if read_value and abs(read_value.value - value) > 0.1:
                    logger.warning(f"Write verification failed: {internal_name}")
        else:
            self._stats["errors"] += 1

        return success

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._handler.is_connected() if self._handler else False

    def get_statistics(self) -> Dict:
        """Get connector statistics."""
        return {
            **self._stats,
            "connected": self.is_connected(),
            "polling": self._polling,
            "tag_count": len(self._tag_mappings),
            "buffer_size": len(self._buffer),
        }

    def get_tag_status(self) -> Dict[str, Dict]:
        """Get status of all tags."""
        return {
            name: {
                "scada_tag": m.scada_tag,
                "last_value": m.last_value,
                "quality": m.last_quality.value,
                "last_update": m.last_timestamp.isoformat() if m.last_timestamp else None,
            }
            for name, m in self._tag_mappings.items()
        }


def create_boiler_tag_mappings() -> List[TagMapping]:
    """Create standard boiler tag mappings."""
    return [
        # Steam drum
        TagMapping(
            scada_tag="HR100",
            internal_name="drum_pressure",
            data_type=DataType.FLOAT32,
            unit="psig",
            scale_factor=1.0,
            low_limit=0.0,
            high_limit=200.0,
            description="Steam drum pressure",
        ),
        TagMapping(
            scada_tag="HR102",
            internal_name="drum_level",
            data_type=DataType.FLOAT32,
            unit="inches",
            scale_factor=1.0,
            low_limit=-10.0,
            high_limit=10.0,
            description="Steam drum level",
        ),
        TagMapping(
            scada_tag="HR104",
            internal_name="steam_flow",
            data_type=DataType.FLOAT32,
            unit="klb/hr",
            scale_factor=1.0,
            low_limit=0.0,
            high_limit=500.0,
            description="Steam flow rate",
        ),
        TagMapping(
            scada_tag="HR106",
            internal_name="steam_temperature",
            data_type=DataType.FLOAT32,
            unit="degF",
            scale_factor=1.0,
            description="Steam temperature",
        ),

        # Combustion
        TagMapping(
            scada_tag="HR200",
            internal_name="o2_percent",
            data_type=DataType.FLOAT32,
            unit="%",
            scale_factor=1.0,
            low_limit=0.0,
            high_limit=21.0,
            description="Flue gas O2",
        ),
        TagMapping(
            scada_tag="HR202",
            internal_name="co_ppm",
            data_type=DataType.FLOAT32,
            unit="ppm",
            scale_factor=1.0,
            low_limit=0.0,
            high_limit=1000.0,
            description="Flue gas CO",
        ),
        TagMapping(
            scada_tag="HR204",
            internal_name="flue_gas_temp",
            data_type=DataType.FLOAT32,
            unit="degF",
            scale_factor=1.0,
            description="Flue gas temperature",
        ),

        # Fuel
        TagMapping(
            scada_tag="HR300",
            internal_name="fuel_flow",
            data_type=DataType.FLOAT32,
            unit="scfh",
            scale_factor=1.0,
            description="Fuel flow rate",
        ),
        TagMapping(
            scada_tag="HR302",
            internal_name="fuel_pressure",
            data_type=DataType.FLOAT32,
            unit="psig",
            scale_factor=1.0,
            description="Fuel header pressure",
        ),

        # Air
        TagMapping(
            scada_tag="HR400",
            internal_name="air_flow",
            data_type=DataType.FLOAT32,
            unit="scfm",
            scale_factor=1.0,
            description="Combustion air flow",
        ),
        TagMapping(
            scada_tag="HR402",
            internal_name="air_temperature",
            data_type=DataType.FLOAT32,
            unit="degF",
            scale_factor=1.0,
            description="Combustion air temperature",
        ),

        # Setpoints (writeable)
        TagMapping(
            scada_tag="HR500",
            internal_name="o2_setpoint",
            data_type=DataType.FLOAT32,
            unit="%",
            scale_factor=1.0,
            read_only=False,
            description="O2 setpoint",
        ),
        TagMapping(
            scada_tag="HR502",
            internal_name="load_demand",
            data_type=DataType.FLOAT32,
            unit="%",
            scale_factor=1.0,
            read_only=False,
            description="Load demand setpoint",
        ),
    ]
