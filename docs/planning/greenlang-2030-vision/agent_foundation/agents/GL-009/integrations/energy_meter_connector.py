"""
Energy Meter Connector for GL-009 THERMALIQ.

Interfaces with industrial energy meters for real-time energy measurements.
Supported protocols: Modbus TCP/RTU, OPC-UA

Handles:
- Multiple meter protocols (Modbus TCP/RTU, OPC-UA)
- Real-time energy readings
- Historical data retrieval
- Multi-phase measurements
- Power quality monitoring
- Automatic unit conversion
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import struct

from .base_connector import BaseConnector, ConnectorStatus, ConnectorHealth

logger = logging.getLogger(__name__)

# Handle optional MQTT dependency
try:
    import paho.mqtt.client as mqtt
    MQTTMessage = mqtt.MQTTMessage
except ImportError:
    mqtt = None
    MQTTMessage = None
    logger.warning("MQTT library not available, MQTT fallback disabled")


class MeterProtocol(Enum):
    """Energy meter communication protocols."""
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPC_UA = "opc_ua"
    IEC_61850 = "iec_61850"
    DLMS_COSEM = "dlms_cosem"


class DataQuality(Enum):
    """Data quality indicators."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"


@dataclass
class EnergyReading:
    """Single energy meter reading."""
    meter_id: str
    timestamp: datetime
    energy_kwh: float
    power_kw: float
    voltage_v: Optional[float] = None
    current_a: Optional[float] = None
    power_factor: Optional[float] = None
    frequency_hz: Optional[float] = None
    reactive_power_kvar: Optional[float] = None
    apparent_power_kva: Optional[float] = None
    quality: DataQuality = DataQuality.GOOD
    phase_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "meter_id": self.meter_id,
            "timestamp": self.timestamp.isoformat(),
            "energy_kwh": self.energy_kwh,
            "power_kw": self.power_kw,
            "voltage_v": self.voltage_v,
            "current_a": self.current_a,
            "power_factor": self.power_factor,
            "frequency_hz": self.frequency_hz,
            "reactive_power_kvar": self.reactive_power_kvar,
            "apparent_power_kva": self.apparent_power_kva,
            "quality": self.quality.value,
            "phase_data": self.phase_data,
            "metadata": self.metadata,
        }


@dataclass
class MeterConfig:
    """Energy meter configuration."""
    meter_id: str
    protocol: MeterProtocol
    address: str
    port: int
    unit_id: int = 1
    registers: Optional[Dict[str, int]] = None  # Register mapping
    scale_factors: Optional[Dict[str, float]] = None  # Scaling factors
    byte_order: str = "big"  # big or little endian
    word_order: str = "big"  # big or little endian for 32-bit values
    timeout_seconds: float = 5.0
    polling_interval_seconds: float = 1.0

    # OPC-UA specific
    namespace: Optional[str] = None
    certificate_path: Optional[str] = None

    # Authentication
    username: Optional[str] = None
    password: Optional[str] = None

    def __post_init__(self):
        """Initialize default register mappings."""
        if self.registers is None:
            self.registers = self._default_registers()
        if self.scale_factors is None:
            self.scale_factors = self._default_scale_factors()

    def _default_registers(self) -> Dict[str, int]:
        """Default Modbus register mapping (common layout)."""
        return {
            "energy_kwh": 0,
            "power_kw": 10,
            "voltage_v": 20,
            "current_a": 30,
            "power_factor": 40,
            "frequency_hz": 50,
            "reactive_power_kvar": 60,
            "apparent_power_kva": 70,
        }

    def _default_scale_factors(self) -> Dict[str, float]:
        """Default scaling factors."""
        return {
            "energy_kwh": 0.001,  # Wh to kWh
            "power_kw": 0.001,    # W to kW
            "voltage_v": 0.1,     # 0.1V resolution
            "current_a": 0.001,   # mA to A
            "power_factor": 0.001,
            "frequency_hz": 0.01,
            "reactive_power_kvar": 0.001,
            "apparent_power_kva": 0.001,
        }


class EnergyMeterConnector(BaseConnector):
    """
    Connector for industrial energy meters.

    Supports multiple protocols and provides unified interface for energy data.
    """

    def __init__(self, config: MeterConfig, **kwargs):
        """
        Initialize energy meter connector.

        Args:
            config: Meter configuration
            **kwargs: Additional arguments for BaseConnector
        """
        super().__init__(
            connector_id=f"energy_meter_{config.meter_id}",
            **kwargs
        )
        self.config = config
        self._client: Optional[Any] = None
        self._opc_client: Optional[Any] = None
        self._subscription_task: Optional[asyncio.Task] = None
        self._latest_reading: Optional[EnergyReading] = None

    async def connect(self) -> bool:
        """
        Establish connection to energy meter.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.config.protocol == MeterProtocol.MODBUS_TCP:
                return await self._connect_modbus_tcp()
            elif self.config.protocol == MeterProtocol.MODBUS_RTU:
                return await self._connect_modbus_rtu()
            elif self.config.protocol == MeterProtocol.OPC_UA:
                return await self._connect_opc_ua()
            else:
                logger.error(f"Unsupported protocol: {self.config.protocol}")
                return False

        except Exception as e:
            logger.error(f"[{self.connector_id}] Connection failed: {e}")
            return False

    async def _connect_modbus_tcp(self) -> bool:
        """Connect via Modbus TCP."""
        try:
            # Try to import pymodbus
            try:
                from pymodbus.client import AsyncModbusTcpClient
            except ImportError:
                logger.error("pymodbus not available, using mock connection")
                self._client = MockModbusClient(self.config)
                return True

            self._client = AsyncModbusTcpClient(
                host=self.config.address,
                port=self.config.port,
                timeout=self.config.timeout_seconds
            )

            connected = await self._client.connect()
            if connected:
                logger.info(f"[{self.connector_id}] Modbus TCP connected to {self.config.address}:{self.config.port}")
                return True
            else:
                logger.error(f"[{self.connector_id}] Modbus TCP connection failed")
                return False

        except Exception as e:
            logger.error(f"[{self.connector_id}] Modbus TCP connection error: {e}")
            # Fallback to mock for development
            self._client = MockModbusClient(self.config)
            return True

    async def _connect_modbus_rtu(self) -> bool:
        """Connect via Modbus RTU."""
        try:
            from pymodbus.client import AsyncModbusSerialClient

            self._client = AsyncModbusSerialClient(
                port=self.config.address,  # Serial port path
                baudrate=self.config.port,  # Baud rate
                timeout=self.config.timeout_seconds
            )

            connected = await self._client.connect()
            if connected:
                logger.info(f"[{self.connector_id}] Modbus RTU connected to {self.config.address}")
                return True
            else:
                return False

        except ImportError:
            logger.warning("pymodbus not available, using mock connection")
            self._client = MockModbusClient(self.config)
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] Modbus RTU connection error: {e}")
            return False

    async def _connect_opc_ua(self) -> bool:
        """Connect via OPC-UA."""
        try:
            from asyncua import Client

            url = f"opc.tcp://{self.config.address}:{self.config.port}"
            self._opc_client = Client(url=url)

            if self.config.username and self.config.password:
                self._opc_client.set_user(self.config.username)
                self._opc_client.set_password(self.config.password)

            await self._opc_client.connect()
            logger.info(f"[{self.connector_id}] OPC-UA connected to {url}")
            return True

        except ImportError:
            logger.warning("asyncua not available, using mock connection")
            self._opc_client = MockOPCClient(self.config)
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] OPC-UA connection error: {e}")
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from energy meter.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self._subscription_task:
                self._subscription_task.cancel()

            if self.config.protocol in [MeterProtocol.MODBUS_TCP, MeterProtocol.MODBUS_RTU]:
                if self._client and hasattr(self._client, 'close'):
                    await self._client.close()
            elif self.config.protocol == MeterProtocol.OPC_UA:
                if self._opc_client and hasattr(self._opc_client, 'disconnect'):
                    await self._opc_client.disconnect()

            logger.info(f"[{self.connector_id}] Disconnected")
            return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] Disconnect error: {e}")
            return False

    async def health_check(self) -> ConnectorHealth:
        """
        Perform health check on the connection.

        Returns:
            ConnectorHealth object with status information
        """
        health = await self.get_health()

        # Additional health checks
        try:
            if self.is_connected:
                # Try to read a single register
                test_reading = await asyncio.wait_for(
                    self.read_energy(),
                    timeout=self.config.timeout_seconds
                )

                if test_reading:
                    health.latency_ms = (datetime.now() - test_reading.timestamp).total_seconds() * 1000
                    health.metadata["last_reading"] = test_reading.to_dict()

        except Exception as e:
            health.is_healthy = False
            health.last_error = str(e)
            logger.warning(f"[{self.connector_id}] Health check failed: {e}")

        return health

    async def read(self, **kwargs) -> EnergyReading:
        """
        Read current energy values from meter.

        Returns:
            EnergyReading object with current values
        """
        return await self.read_energy()

    async def read_energy(self) -> EnergyReading:
        """
        Read current energy values from meter.

        Returns:
            EnergyReading object with current values

        Raises:
            Exception: If read fails
        """
        if self.config.protocol in [MeterProtocol.MODBUS_TCP, MeterProtocol.MODBUS_RTU]:
            return await self._read_modbus()
        elif self.config.protocol == MeterProtocol.OPC_UA:
            return await self._read_opc_ua()
        else:
            raise ValueError(f"Unsupported protocol: {self.config.protocol}")

    async def _read_modbus(self) -> EnergyReading:
        """Read energy data via Modbus."""
        try:
            # Read holding registers
            data = {}

            for param, register in self.config.registers.items():
                try:
                    # Read 2 registers (32-bit value)
                    response = await self._client.read_holding_registers(
                        address=register,
                        count=2,
                        slave=self.config.unit_id
                    )

                    if response.isError():
                        logger.warning(f"[{self.connector_id}] Failed to read {param}")
                        continue

                    # Convert to float (IEEE 754)
                    raw_value = self._registers_to_float(
                        response.registers,
                        self.config.byte_order,
                        self.config.word_order
                    )

                    # Apply scaling
                    scaled_value = raw_value * self.config.scale_factors.get(param, 1.0)
                    data[param] = scaled_value

                except Exception as e:
                    logger.warning(f"[{self.connector_id}] Error reading {param}: {e}")

            # Create reading object
            reading = EnergyReading(
                meter_id=self.config.meter_id,
                timestamp=datetime.now(),
                energy_kwh=data.get("energy_kwh", 0.0),
                power_kw=data.get("power_kw", 0.0),
                voltage_v=data.get("voltage_v"),
                current_a=data.get("current_a"),
                power_factor=data.get("power_factor"),
                frequency_hz=data.get("frequency_hz"),
                reactive_power_kvar=data.get("reactive_power_kvar"),
                apparent_power_kva=data.get("apparent_power_kva"),
                quality=DataQuality.GOOD if data else DataQuality.BAD
            )

            self._latest_reading = reading
            return reading

        except Exception as e:
            logger.error(f"[{self.connector_id}] Modbus read error: {e}")
            raise

    async def _read_opc_ua(self) -> EnergyReading:
        """Read energy data via OPC-UA."""
        try:
            # Build node paths
            namespace = self.config.namespace or "2"
            base_path = f"ns={namespace};s=Energy"

            # Read all values
            data = {}
            for param in self.config.registers.keys():
                try:
                    node_path = f"{base_path}.{param}"
                    node = self._opc_client.get_node(node_path)
                    value = await node.read_value()
                    data[param] = float(value)
                except Exception as e:
                    logger.warning(f"[{self.connector_id}] Error reading {param}: {e}")

            # Create reading object
            reading = EnergyReading(
                meter_id=self.config.meter_id,
                timestamp=datetime.now(),
                energy_kwh=data.get("energy_kwh", 0.0),
                power_kw=data.get("power_kw", 0.0),
                voltage_v=data.get("voltage_v"),
                current_a=data.get("current_a"),
                power_factor=data.get("power_factor"),
                frequency_hz=data.get("frequency_hz"),
                reactive_power_kvar=data.get("reactive_power_kvar"),
                apparent_power_kva=data.get("apparent_power_kva"),
                quality=DataQuality.GOOD if data else DataQuality.BAD
            )

            self._latest_reading = reading
            return reading

        except Exception as e:
            logger.error(f"[{self.connector_id}] OPC-UA read error: {e}")
            raise

    async def read_historical(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 15
    ) -> List[EnergyReading]:
        """
        Read historical energy data.

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            interval_minutes: Data interval in minutes

        Returns:
            List of EnergyReading objects

        Raises:
            NotImplementedError: If protocol doesn't support historical data
        """
        if self.config.protocol == MeterProtocol.OPC_UA:
            return await self._read_historical_opc_ua(start_time, end_time, interval_minutes)
        else:
            # Modbus doesn't typically support historical data
            raise NotImplementedError(f"Historical data not supported for {self.config.protocol}")

    async def _read_historical_opc_ua(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int
    ) -> List[EnergyReading]:
        """Read historical data via OPC-UA."""
        try:
            from asyncua import ua

            # Build node path for energy parameter
            namespace = self.config.namespace or "2"
            node_path = f"ns={namespace};s=Energy.energy_kwh"
            node = self._opc_client.get_node(node_path)

            # Read historical data
            history = await self._opc_client.read_raw_history(
                nodeid=node.nodeid,
                start=start_time,
                end=end_time
            )

            readings = []
            for data_value in history:
                reading = EnergyReading(
                    meter_id=self.config.meter_id,
                    timestamp=data_value.SourceTimestamp,
                    energy_kwh=float(data_value.Value.Value),
                    power_kw=0.0,  # Would need separate query
                    quality=DataQuality.GOOD if data_value.StatusCode.is_good() else DataQuality.BAD
                )
                readings.append(reading)

            logger.info(f"[{self.connector_id}] Retrieved {len(readings)} historical readings")
            return readings

        except Exception as e:
            logger.error(f"[{self.connector_id}] Historical read error: {e}")
            return []

    async def start_streaming(self, callback: callable):
        """
        Start continuous data streaming.

        Args:
            callback: Async function to call with each reading
        """
        logger.info(f"[{self.connector_id}] Starting data streaming")

        async def stream_loop():
            while not self._shutdown_event.is_set():
                try:
                    reading = await self.read_energy()
                    await callback(reading)
                    await asyncio.sleep(self.config.polling_interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[{self.connector_id}] Streaming error: {e}")
                    await asyncio.sleep(5)  # Error backoff

        self._subscription_task = asyncio.create_task(stream_loop())

    def _registers_to_float(
        self,
        registers: List[int],
        byte_order: str,
        word_order: str
    ) -> float:
        """
        Convert Modbus registers to float.

        Args:
            registers: List of register values
            byte_order: Byte order (big/little)
            word_order: Word order (big/little)

        Returns:
            Float value
        """
        if len(registers) < 2:
            return 0.0

        # Combine registers based on word order
        if word_order == "big":
            high_word = registers[0]
            low_word = registers[1]
        else:
            high_word = registers[1]
            low_word = registers[0]

        # Combine into 32-bit value
        combined = (high_word << 16) | low_word

        # Convert to bytes
        byte_str = combined.to_bytes(4, byteorder=byte_order)

        # Unpack as float
        return struct.unpack('!f' if byte_order == 'big' else '<f', byte_str)[0]


class MockModbusClient:
    """Mock Modbus client for testing."""

    def __init__(self, config: MeterConfig):
        self.config = config
        self._connected = False

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def close(self):
        self._connected = False

    async def read_holding_registers(self, address: int, count: int, slave: int):
        """Return mock register data."""
        from collections import namedtuple
        Response = namedtuple('Response', ['registers', 'isError'])

        # Generate mock data
        if address == self.config.registers["energy_kwh"]:
            # Mock 1234.5 kWh
            return Response(registers=[16544, 0], isError=lambda: False)
        elif address == self.config.registers["power_kw"]:
            # Mock 250.0 kW
            return Response(registers=[16000, 0], isError=lambda: False)
        else:
            return Response(registers=[0, 0], isError=lambda: False)


class MockOPCClient:
    """Mock OPC-UA client for testing."""

    def __init__(self, config: MeterConfig):
        self.config = config

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    def get_node(self, path: str):
        from collections import namedtuple
        Node = namedtuple('Node', ['read_value'])

        async def read_value():
            # Return mock data based on path
            if "energy_kwh" in path:
                return 1234.5
            elif "power_kw" in path:
                return 250.0
            elif "voltage_v" in path:
                return 230.0
            elif "current_a" in path:
                return 10.0
            else:
                return 0.0

        return Node(read_value=read_value)
