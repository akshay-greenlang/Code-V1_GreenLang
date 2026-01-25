"""
Modbus TCP/RTU Gateway for GreenLang Agents

This module provides a production-ready Modbus gateway supporting
both TCP and RTU protocols for industrial device communication.

Features:
- Modbus TCP client
- Modbus RTU over TCP
- Serial RTU support
- Register mapping
- Data type conversions
- Automatic reconnection
- Polling and monitoring

Example:
    >>> gateway = ModbusGateway(config)
    >>> await gateway.connect()
    >>> values = await gateway.read_holding_registers(0, 10)
"""

import asyncio
import hashlib
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from pymodbus.client import AsyncModbusTcpClient, AsyncModbusSerialClient
    from pymodbus.exceptions import ModbusException
    from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
    from pymodbus.constants import Endian
    PYMODBUS_AVAILABLE = True
except ImportError:
    PYMODBUS_AVAILABLE = False
    AsyncModbusTcpClient = None
    AsyncModbusSerialClient = None
    ModbusException = Exception

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModbusProtocol(str, Enum):
    """Modbus protocol types."""
    TCP = "tcp"
    RTU = "rtu"
    RTU_OVER_TCP = "rtu_over_tcp"


class DataType(str, Enum):
    """Modbus data types for decoding."""
    UINT16 = "uint16"
    INT16 = "int16"
    UINT32 = "uint32"
    INT32 = "int32"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    BITS = "bits"


class ByteOrder(str, Enum):
    """Byte order for multi-register values."""
    BIG_ENDIAN = "big"
    LITTLE_ENDIAN = "little"
    BIG_ENDIAN_SWAP = "big_swap"  # Big endian word, little endian byte
    LITTLE_ENDIAN_SWAP = "little_swap"  # Little endian word, big endian byte


@dataclass
class ModbusGatewayConfig:
    """Configuration for Modbus gateway."""
    # Connection settings
    protocol: ModbusProtocol = ModbusProtocol.TCP
    host: str = "localhost"
    port: int = 502
    # RTU settings
    serial_port: Optional[str] = None
    baudrate: int = 9600
    parity: str = "N"
    stopbits: int = 1
    bytesize: int = 8
    # Common settings
    unit_id: int = 1
    timeout: float = 3.0
    retries: int = 3
    retry_delay: float = 0.5
    # Data settings
    byte_order: ByteOrder = ByteOrder.BIG_ENDIAN
    word_order: ByteOrder = ByteOrder.BIG_ENDIAN
    # Reconnection
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10


class RegisterMapping(BaseModel):
    """Mapping for a Modbus register."""
    name: str = Field(..., description="Register name")
    address: int = Field(..., ge=0, description="Register address")
    data_type: DataType = Field(default=DataType.UINT16, description="Data type")
    scale: float = Field(default=1.0, description="Scale factor")
    offset: float = Field(default=0.0, description="Offset value")
    unit: Optional[str] = Field(default=None, description="Engineering unit")
    read_only: bool = Field(default=True, description="Read-only flag")

    def register_count(self) -> int:
        """Get number of registers for this data type."""
        counts = {
            DataType.UINT16: 1,
            DataType.INT16: 1,
            DataType.UINT32: 2,
            DataType.INT32: 2,
            DataType.FLOAT32: 2,
            DataType.FLOAT64: 4,
            DataType.BITS: 1,
        }
        return counts.get(self.data_type, 1)


class ModbusValue(BaseModel):
    """Modbus value with metadata."""
    name: str = Field(..., description="Value name")
    address: int = Field(..., description="Register address")
    raw_value: Any = Field(..., description="Raw register value")
    scaled_value: float = Field(..., description="Scaled engineering value")
    unit: Optional[str] = Field(default=None, description="Engineering unit")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    quality: str = Field(default="good", description="Data quality")
    provenance_hash: str = Field(default="", description="Provenance hash")


class ModbusGateway:
    """
    Production-ready Modbus TCP/RTU gateway.

    This gateway provides reliable communication with Modbus devices
    including data type conversions and register mapping.

    Attributes:
        config: Gateway configuration
        register_map: Configured register mappings
        client: Underlying Modbus client

    Example:
        >>> config = ModbusGatewayConfig(
        ...     protocol=ModbusProtocol.TCP,
        ...     host="192.168.1.100",
        ...     port=502
        ... )
        >>> gateway = ModbusGateway(config)
        >>> async with gateway:
        ...     value = await gateway.read_register("temperature")
    """

    def __init__(self, config: ModbusGatewayConfig):
        """
        Initialize Modbus gateway.

        Args:
            config: Gateway configuration

        Raises:
            ImportError: If pymodbus is not installed
        """
        if not PYMODBUS_AVAILABLE:
            raise ImportError(
                "pymodbus is required for Modbus support. "
                "Install with: pip install pymodbus"
            )

        self.config = config
        self.register_map: Dict[str, RegisterMapping] = {}
        self._client: Optional[Any] = None
        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._poll_callbacks: Dict[str, List[Callable]] = {}

        logger.info(
            f"ModbusGateway initialized: {config.protocol.value} "
            f"{config.host}:{config.port}"
        )

    async def connect(self) -> None:
        """
        Connect to the Modbus device.

        Establishes connection based on protocol configuration.

        Raises:
            ConnectionError: If connection fails
        """
        if self._connected:
            logger.warning("Already connected")
            return

        self._shutdown = False

        try:
            if self.config.protocol == ModbusProtocol.TCP:
                self._client = AsyncModbusTcpClient(
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout,
                    retries=self.config.retries,
                    retry_on_empty=True,
                )
            elif self.config.protocol == ModbusProtocol.RTU:
                self._client = AsyncModbusSerialClient(
                    port=self.config.serial_port,
                    baudrate=self.config.baudrate,
                    parity=self.config.parity,
                    stopbits=self.config.stopbits,
                    bytesize=self.config.bytesize,
                    timeout=self.config.timeout,
                )
            else:
                # RTU over TCP
                self._client = AsyncModbusTcpClient(
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout,
                    framer="rtu",
                )

            connected = await self._client.connect()
            if not connected:
                raise ConnectionError("Failed to establish connection")

            self._connected = True
            logger.info(f"Connected to Modbus device at {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Connection failed: {e}", exc_info=True)
            raise ConnectionError(f"Modbus connection failed: {e}") from e

    async def disconnect(self) -> None:
        """
        Disconnect from the Modbus device gracefully.
        """
        self._shutdown = True

        # Cancel polling tasks
        for task in self._poll_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._poll_tasks.clear()

        # Cancel reconnect task
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        # Close client
        if self._client:
            self._client.close()

        self._connected = False
        logger.info("Disconnected from Modbus device")

    async def _handle_connection_loss(self) -> None:
        """Handle connection loss and attempt reconnection."""
        if self._shutdown or not self.config.auto_reconnect:
            return

        self._connected = False
        logger.warning("Connection lost, attempting reconnection...")

        for attempt in range(self.config.max_reconnect_attempts):
            if self._shutdown:
                break

            try:
                await asyncio.sleep(
                    self.config.reconnect_delay * (1.5 ** attempt)
                )
                await self.connect()
                logger.info(f"Reconnected after {attempt + 1} attempts")
                return

            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")

        logger.error("Max reconnection attempts reached")

    def add_register(self, mapping: RegisterMapping) -> None:
        """
        Add a register mapping.

        Args:
            mapping: Register mapping definition
        """
        self.register_map[mapping.name] = mapping
        logger.debug(f"Added register mapping: {mapping.name} @ {mapping.address}")

    def add_registers(self, mappings: List[RegisterMapping]) -> None:
        """
        Add multiple register mappings.

        Args:
            mappings: List of register mappings
        """
        for mapping in mappings:
            self.add_register(mapping)

    async def read_coils(
        self,
        address: int,
        count: int = 1,
        unit: Optional[int] = None
    ) -> List[bool]:
        """
        Read coil registers (function code 01).

        Args:
            address: Starting address
            count: Number of coils to read
            unit: Slave unit ID (defaults to config)

        Returns:
            List of boolean coil values
        """
        self._ensure_connected()
        unit_id = unit or self.config.unit_id

        try:
            result = await self._client.read_coils(
                address=address,
                count=count,
                slave=unit_id
            )

            if result.isError():
                raise ModbusException(f"Read coils error: {result}")

            return result.bits[:count]

        except Exception as e:
            logger.error(f"Read coils failed: {e}")
            await self._handle_connection_loss()
            raise

    async def read_discrete_inputs(
        self,
        address: int,
        count: int = 1,
        unit: Optional[int] = None
    ) -> List[bool]:
        """
        Read discrete input registers (function code 02).

        Args:
            address: Starting address
            count: Number of inputs to read
            unit: Slave unit ID

        Returns:
            List of boolean input values
        """
        self._ensure_connected()
        unit_id = unit or self.config.unit_id

        try:
            result = await self._client.read_discrete_inputs(
                address=address,
                count=count,
                slave=unit_id
            )

            if result.isError():
                raise ModbusException(f"Read discrete inputs error: {result}")

            return result.bits[:count]

        except Exception as e:
            logger.error(f"Read discrete inputs failed: {e}")
            await self._handle_connection_loss()
            raise

    async def read_holding_registers(
        self,
        address: int,
        count: int = 1,
        unit: Optional[int] = None
    ) -> List[int]:
        """
        Read holding registers (function code 03).

        Args:
            address: Starting address
            count: Number of registers to read
            unit: Slave unit ID

        Returns:
            List of register values
        """
        self._ensure_connected()
        unit_id = unit or self.config.unit_id

        try:
            result = await self._client.read_holding_registers(
                address=address,
                count=count,
                slave=unit_id
            )

            if result.isError():
                raise ModbusException(f"Read holding registers error: {result}")

            return result.registers

        except Exception as e:
            logger.error(f"Read holding registers failed: {e}")
            await self._handle_connection_loss()
            raise

    async def read_input_registers(
        self,
        address: int,
        count: int = 1,
        unit: Optional[int] = None
    ) -> List[int]:
        """
        Read input registers (function code 04).

        Args:
            address: Starting address
            count: Number of registers to read
            unit: Slave unit ID

        Returns:
            List of register values
        """
        self._ensure_connected()
        unit_id = unit or self.config.unit_id

        try:
            result = await self._client.read_input_registers(
                address=address,
                count=count,
                slave=unit_id
            )

            if result.isError():
                raise ModbusException(f"Read input registers error: {result}")

            return result.registers

        except Exception as e:
            logger.error(f"Read input registers failed: {e}")
            await self._handle_connection_loss()
            raise

    async def write_coil(
        self,
        address: int,
        value: bool,
        unit: Optional[int] = None
    ) -> str:
        """
        Write single coil (function code 05).

        Args:
            address: Coil address
            value: Boolean value to write
            unit: Slave unit ID

        Returns:
            Provenance hash
        """
        self._ensure_connected()
        unit_id = unit or self.config.unit_id

        try:
            result = await self._client.write_coil(
                address=address,
                value=value,
                slave=unit_id
            )

            if result.isError():
                raise ModbusException(f"Write coil error: {result}")

            # Calculate provenance
            provenance_str = f"coil:{address}:{value}:{datetime.utcnow().isoformat()}"
            return hashlib.sha256(provenance_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Write coil failed: {e}")
            raise

    async def write_register(
        self,
        address: int,
        value: int,
        unit: Optional[int] = None
    ) -> str:
        """
        Write single holding register (function code 06).

        Args:
            address: Register address
            value: Value to write (0-65535)
            unit: Slave unit ID

        Returns:
            Provenance hash
        """
        self._ensure_connected()
        unit_id = unit or self.config.unit_id

        try:
            result = await self._client.write_register(
                address=address,
                value=value,
                slave=unit_id
            )

            if result.isError():
                raise ModbusException(f"Write register error: {result}")

            provenance_str = f"register:{address}:{value}:{datetime.utcnow().isoformat()}"
            return hashlib.sha256(provenance_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Write register failed: {e}")
            raise

    async def write_registers(
        self,
        address: int,
        values: List[int],
        unit: Optional[int] = None
    ) -> str:
        """
        Write multiple holding registers (function code 16).

        Args:
            address: Starting address
            values: List of values to write
            unit: Slave unit ID

        Returns:
            Provenance hash
        """
        self._ensure_connected()
        unit_id = unit or self.config.unit_id

        try:
            result = await self._client.write_registers(
                address=address,
                values=values,
                slave=unit_id
            )

            if result.isError():
                raise ModbusException(f"Write registers error: {result}")

            provenance_str = f"registers:{address}:{values}:{datetime.utcnow().isoformat()}"
            return hashlib.sha256(provenance_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Write registers failed: {e}")
            raise

    async def read_register_value(self, name: str) -> ModbusValue:
        """
        Read a mapped register with data type conversion.

        Args:
            name: Register mapping name

        Returns:
            ModbusValue with scaled engineering value
        """
        mapping = self.register_map.get(name)
        if not mapping:
            raise KeyError(f"Register mapping '{name}' not found")

        count = mapping.register_count()
        registers = await self.read_holding_registers(mapping.address, count)

        # Decode based on data type
        raw_value = self._decode_registers(registers, mapping.data_type)

        # Apply scaling
        scaled_value = raw_value * mapping.scale + mapping.offset

        # Calculate provenance
        provenance_str = f"{name}:{mapping.address}:{raw_value}:{datetime.utcnow().isoformat()}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        return ModbusValue(
            name=name,
            address=mapping.address,
            raw_value=raw_value,
            scaled_value=scaled_value,
            unit=mapping.unit,
            provenance_hash=provenance_hash
        )

    async def read_all_registers(self) -> Dict[str, ModbusValue]:
        """
        Read all mapped registers.

        Returns:
            Dictionary of register values
        """
        results = {}
        for name in self.register_map:
            try:
                results[name] = await self.read_register_value(name)
            except Exception as e:
                logger.error(f"Failed to read register {name}: {e}")
        return results

    def _decode_registers(
        self,
        registers: List[int],
        data_type: DataType
    ) -> Union[int, float]:
        """Decode register values based on data type."""
        # Determine byte order
        if self.config.byte_order == ByteOrder.BIG_ENDIAN:
            byte_order = Endian.BIG
        else:
            byte_order = Endian.LITTLE

        if self.config.word_order == ByteOrder.BIG_ENDIAN:
            word_order = Endian.BIG
        else:
            word_order = Endian.LITTLE

        decoder = BinaryPayloadDecoder.fromRegisters(
            registers,
            byteorder=byte_order,
            wordorder=word_order
        )

        if data_type == DataType.UINT16:
            return decoder.decode_16bit_uint()
        elif data_type == DataType.INT16:
            return decoder.decode_16bit_int()
        elif data_type == DataType.UINT32:
            return decoder.decode_32bit_uint()
        elif data_type == DataType.INT32:
            return decoder.decode_32bit_int()
        elif data_type == DataType.FLOAT32:
            return decoder.decode_32bit_float()
        elif data_type == DataType.FLOAT64:
            return decoder.decode_64bit_float()
        else:
            return registers[0]

    async def start_polling(
        self,
        name: str,
        interval_ms: int,
        callback: Callable[[ModbusValue], None]
    ) -> None:
        """
        Start polling a register at regular intervals.

        Args:
            name: Register mapping name
            interval_ms: Polling interval in milliseconds
            callback: Function to call with each value
        """
        if name in self._poll_tasks:
            logger.warning(f"Polling already active for {name}")
            return

        if name not in self._poll_callbacks:
            self._poll_callbacks[name] = []
        self._poll_callbacks[name].append(callback)

        async def poll_loop():
            while not self._shutdown:
                try:
                    value = await self.read_register_value(name)
                    for cb in self._poll_callbacks.get(name, []):
                        try:
                            if asyncio.iscoroutinefunction(cb):
                                await cb(value)
                            else:
                                cb(value)
                        except Exception as e:
                            logger.error(f"Poll callback error: {e}")

                except Exception as e:
                    logger.error(f"Polling error for {name}: {e}")

                await asyncio.sleep(interval_ms / 1000)

        self._poll_tasks[name] = asyncio.create_task(poll_loop())
        logger.info(f"Started polling {name} every {interval_ms}ms")

    async def stop_polling(self, name: str) -> None:
        """
        Stop polling a register.

        Args:
            name: Register mapping name
        """
        if name in self._poll_tasks:
            self._poll_tasks[name].cancel()
            try:
                await self._poll_tasks[name]
            except asyncio.CancelledError:
                pass
            del self._poll_tasks[name]
            logger.info(f"Stopped polling {name}")

    def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if not self._connected:
            raise ConnectionError("Not connected to Modbus device")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get gateway statistics.

        Returns:
            Dictionary containing gateway statistics
        """
        return {
            "connected": self._connected,
            "protocol": self.config.protocol.value,
            "host": self.config.host,
            "port": self.config.port,
            "unit_id": self.config.unit_id,
            "registered_mappings": len(self.register_map),
            "active_polls": list(self._poll_tasks.keys()),
        }

    async def __aenter__(self) -> "ModbusGateway":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
