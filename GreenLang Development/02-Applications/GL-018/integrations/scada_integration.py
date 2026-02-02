"""
SCADA Integration Module for GL-018 FLUEFLOW

Provides comprehensive OPC-UA and Modbus integration for flue gas analyzers
and combustion optimization systems. Supports real-time monitoring of flue gas
composition, temperature, pressure, and flow parameters with control capabilities
for air/fuel ratio optimization.

Supported Analyzers:
- ABB AO2000 series (OPC-UA)
- SICK MARSIC series (OPC-UA)
- Emerson Rosemount X-STREAM (OPC-UA/Modbus)
- Siemens ULTRAMAT (OPC-UA/Profinet)
- Horiba PG series (Modbus TCP)
- Fuji Electric ZRJ (Modbus RTU/TCP)
- Yokogawa AV550G (OPC-UA)

Monitored Parameters:
- O2 percentage (combustion air optimization)
- CO2 percentage (combustion efficiency)
- CO concentration (ppm) - incomplete combustion indicator
- NOx concentration (ppm) - emissions compliance
- Stack temperature (flue gas temperature)
- Stack pressure (draft control)
- Flue gas flow rate (mass/volumetric)
- Fuel flow rate (primary fuel)
- Air flow rate (combustion air)

Control Setpoints:
- Air damper position (modulating control)
- Fuel valve position (flow control)
- Target O2 setpoint (optimization target)

Author: GreenLang Integrations Engineering Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import asyncio
import logging
import time
from collections import defaultdict, deque

from pydantic import BaseModel, Field, ConfigDict, field_validator

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class TagType(str, Enum):
    """Types of SCADA tags."""
    ANALOG_INPUT = "analog_input"
    ANALOG_OUTPUT = "analog_output"
    DIGITAL_INPUT = "digital_input"
    DIGITAL_OUTPUT = "digital_output"
    CALCULATED = "calculated"
    ALARM = "alarm"


class ParameterType(str, Enum):
    """Flue gas and combustion parameter types."""
    # Gas composition
    OXYGEN = "oxygen"
    CARBON_DIOXIDE = "carbon_dioxide"
    CARBON_MONOXIDE = "carbon_monoxide"
    NOX = "nitrogen_oxides"
    SO2 = "sulfur_dioxide"
    METHANE = "methane"
    TOTAL_HYDROCARBONS = "total_hydrocarbons"

    # Physical parameters
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW_RATE = "flow_rate"
    VELOCITY = "velocity"
    HUMIDITY = "humidity"

    # Flow measurements
    FUEL_FLOW = "fuel_flow"
    AIR_FLOW = "air_flow"
    FLUE_GAS_FLOW = "flue_gas_flow"

    # Control parameters
    DAMPER_POSITION = "damper_position"
    VALVE_POSITION = "valve_position"
    FAN_SPEED = "fan_speed"

    # Calculated values
    EXCESS_AIR = "excess_air"
    COMBUSTION_EFFICIENCY = "combustion_efficiency"
    AIR_FUEL_RATIO = "air_fuel_ratio"
    HEAT_LOSS = "heat_loss"
    LAMBDA = "lambda_value"


class MeasurementLocation(str, Enum):
    """Measurement point locations."""
    # Stack locations
    STACK_OUTLET = "stack_outlet"
    ECONOMIZER_OUTLET = "economizer_outlet"
    AIR_HEATER_OUTLET = "air_heater_outlet"
    BOILER_OUTLET = "boiler_outlet"

    # Air side
    FORCED_DRAFT_FAN_INLET = "fd_fan_inlet"
    FORCED_DRAFT_FAN_OUTLET = "fd_fan_outlet"
    WINDBOX = "windbox"
    PRIMARY_AIR = "primary_air"
    SECONDARY_AIR = "secondary_air"

    # Fuel side
    FUEL_INLET = "fuel_inlet"
    BURNER_INLET = "burner_inlet"

    # Exhaust
    INDUCED_DRAFT_FAN_INLET = "id_fan_inlet"
    INDUCED_DRAFT_FAN_OUTLET = "id_fan_outlet"

    # Ambient
    AMBIENT = "ambient"


class AnalyzerType(str, Enum):
    """Types of flue gas analyzers."""
    ABB_AO2000 = "abb_ao2000"
    SICK_MARSIC = "sick_marsic"
    EMERSON_XSTREAM = "emerson_xstream"
    SIEMENS_ULTRAMAT = "siemens_ultramat"
    HORIBA_PG = "horiba_pg"
    FUJI_ZRJ = "fuji_zrj"
    YOKOGAWA_AV550G = "yokogawa_av550g"
    GENERIC_OPC_UA = "generic_opcua"
    GENERIC_MODBUS = "generic_modbus"


class AlarmSeverity(str, Enum):
    """Alarm severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlarmState(str, Enum):
    """Alarm states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    CLEARED = "cleared"
    SHELVED = "shelved"


class ConnectionProtocol(str, Enum):
    """Supported communication protocols."""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    PROFINET = "profinet"
    ETHERNET_IP = "ethernet_ip"


# =============================================================================
# Pydantic Models
# =============================================================================


class SCADAConfig(BaseModel):
    """Configuration for SCADA client."""

    model_config = ConfigDict(extra="forbid")

    # Connection settings
    protocol: ConnectionProtocol = Field(
        default=ConnectionProtocol.OPC_UA,
        description="Communication protocol"
    )
    host: str = Field(..., description="SCADA server host")
    port: int = Field(default=4840, ge=1, le=65535, description="Server port")

    # Analyzer identification
    analyzer_type: AnalyzerType = Field(
        default=AnalyzerType.GENERIC_OPC_UA,
        description="Type of flue gas analyzer"
    )

    # OPC-UA specific
    endpoint_url: Optional[str] = Field(
        default=None,
        description="OPC-UA endpoint URL"
    )
    namespace_index: int = Field(
        default=2,
        ge=0,
        description="OPC-UA namespace index"
    )

    # Modbus specific
    modbus_unit_id: int = Field(default=1, ge=1, le=247, description="Modbus unit ID")
    modbus_timeout: float = Field(default=3.0, ge=0.1, le=30.0, description="Modbus timeout")

    # Authentication
    username: Optional[str] = Field(default=None, description="Username for authentication")
    password: Optional[str] = Field(default=None, description="Password for authentication")
    certificate_path: Optional[str] = Field(default=None, description="Path to client certificate")
    private_key_path: Optional[str] = Field(default=None, description="Path to private key")

    # Connection management
    connection_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Connection timeout seconds"
    )
    reconnect_interval: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Reconnection interval seconds"
    )
    max_reconnect_attempts: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max reconnection attempts"
    )

    # Data subscription
    subscription_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Subscription interval milliseconds"
    )
    queue_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Subscription queue size"
    )

    # Historical data
    enable_historical_access: bool = Field(
        default=True,
        description="Enable historical data access"
    )
    historical_buffer_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Historical data buffer hours"
    )

    # Performance
    batch_read_size: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Batch read size for multiple tags"
    )
    cache_ttl_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Tag value cache TTL"
    )

    # Data buffering during disconnection
    enable_buffering: bool = Field(
        default=True,
        description="Enable data buffering during disconnections"
    )
    buffer_max_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum buffer size for disconnected data"
    )


class FlueGasTag(BaseModel):
    """Flue gas measurement tag definition."""

    model_config = ConfigDict(extra="allow")

    tag_name: str = Field(..., description="Tag name/identifier in SCADA")
    parameter_type: ParameterType = Field(..., description="Type of parameter")
    location: MeasurementLocation = Field(..., description="Measurement location")
    tag_type: TagType = Field(default=TagType.ANALOG_INPUT, description="Tag type")

    # Units and scaling
    engineering_unit: str = Field(..., description="Engineering unit (%, ppm, C, Pa, etc.)")
    raw_min: float = Field(default=0.0, description="Raw value minimum")
    raw_max: float = Field(default=100.0, description="Raw value maximum")
    scaled_min: float = Field(default=0.0, description="Scaled value minimum")
    scaled_max: float = Field(default=100.0, description="Scaled value maximum")

    # Quality limits
    low_alarm_limit: Optional[float] = Field(default=None, description="Low alarm limit")
    low_warning_limit: Optional[float] = Field(default=None, description="Low warning limit")
    high_warning_limit: Optional[float] = Field(default=None, description="High warning limit")
    high_alarm_limit: Optional[float] = Field(default=None, description="High alarm limit")

    # Operational settings
    deadband: float = Field(default=0.0, ge=0.0, description="Change deadband")
    update_rate_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Update rate milliseconds"
    )

    # Data quality
    enable_quality_check: bool = Field(default=True, description="Enable quality checking")
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Value timeout"
    )

    # Analyzer-specific
    analyzer_channel: Optional[int] = Field(default=None, description="Analyzer channel number")
    sample_point: Optional[str] = Field(default=None, description="Sample point identifier")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TagDataPoint(BaseModel):
    """Single tag data point with timestamp and quality."""

    model_config = ConfigDict(extra="allow")

    tag_name: str = Field(..., description="Tag name")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp"
    )
    value: Union[float, int, bool, str] = Field(..., description="Tag value")
    raw_value: Optional[Union[float, int]] = Field(default=None, description="Raw value")
    quality: str = Field(default="GOOD", description="Data quality indicator")
    status_code: Optional[int] = Field(default=None, description="Status code")

    # Engineering data
    engineering_unit: Optional[str] = Field(default=None, description="Engineering unit")
    parameter_type: Optional[ParameterType] = Field(default=None, description="Parameter type")
    location: Optional[MeasurementLocation] = Field(default=None, description="Location")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AlarmData(BaseModel):
    """Alarm/event data."""

    model_config = ConfigDict(extra="allow")

    alarm_id: str = Field(..., description="Alarm identifier")
    tag_name: str = Field(..., description="Associated tag")
    severity: AlarmSeverity = Field(..., description="Alarm severity")
    state: AlarmState = Field(default=AlarmState.ACTIVE, description="Alarm state")

    message: str = Field(..., description="Alarm message")
    description: Optional[str] = Field(default=None, description="Detailed description")

    # Timestamps
    activated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Activation timestamp"
    )
    acknowledged_at: Optional[datetime] = Field(default=None, description="Acknowledgment time")
    cleared_at: Optional[datetime] = Field(default=None, description="Clear time")

    # Values
    current_value: Optional[float] = Field(default=None, description="Current value")
    setpoint: Optional[float] = Field(default=None, description="Setpoint value")
    deviation: Optional[float] = Field(default=None, description="Deviation from setpoint")

    # Acknowledgment
    acknowledged_by: Optional[str] = Field(default=None, description="Who acknowledged")
    notes: Optional[str] = Field(default=None, description="Acknowledgment notes")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# SCADA Client Implementation
# =============================================================================


class SCADAClient:
    """
    SCADA/DCS client for flue gas analyzer integration.

    Provides:
    - Real-time tag monitoring and subscription
    - Historical data retrieval
    - Setpoint writing (damper, valve control)
    - Alarm management
    - Connection pooling and auto-reconnection
    - Data buffering during disconnections
    - Async operations
    """

    def __init__(self, config: SCADAConfig) -> None:
        """
        Initialize SCADA client.

        Args:
            config: SCADA configuration
        """
        self._config = config
        self._logger = logger

        # Connection state
        self._connected = False
        self._client = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0

        # Tag registry
        self._tags: Dict[str, FlueGasTag] = {}
        self._tag_cache: Dict[str, Tuple[TagDataPoint, float]] = {}  # (value, cache_time)

        # Subscriptions
        self._subscriptions: Dict[str, asyncio.Task] = {}
        self._subscription_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Historical data buffer
        self._historical_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )

        # Disconnection buffer (for data during outages)
        self._disconnect_buffer: deque = deque(maxlen=config.buffer_max_size)
        self._is_buffering = False

        # Alarms
        self._active_alarms: Dict[str, AlarmData] = {}
        self._alarm_history: deque = deque(maxlen=1000)

        # Statistics
        self._stats = {
            "reads": 0,
            "writes": 0,
            "errors": 0,
            "reconnections": 0,
            "buffered_points": 0,
            "last_successful_read": None,
            "last_successful_write": None,
        }

        # Health monitoring
        self._last_heartbeat = time.time()
        self._heartbeat_task: Optional[asyncio.Task] = None

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Establish connection to SCADA system.

        Returns:
            True if connection successful
        """
        try:
            self._logger.info(
                f"Connecting to {self._config.analyzer_type.value} analyzer: "
                f"{self._config.protocol.value} at {self._config.host}:{self._config.port}"
            )

            if self._config.protocol == ConnectionProtocol.OPC_UA:
                await self._connect_opcua()
            elif self._config.protocol in [
                ConnectionProtocol.MODBUS_TCP,
                ConnectionProtocol.MODBUS_RTU
            ]:
                await self._connect_modbus()
            else:
                raise ValueError(f"Unsupported protocol: {self._config.protocol}")

            self._connected = True
            self._reconnect_attempts = 0
            self._last_heartbeat = time.time()

            # Flush disconnect buffer if we were buffering
            if self._is_buffering:
                await self._flush_disconnect_buffer()
                self._is_buffering = False

            # Start heartbeat monitoring
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            self._logger.info(
                f"SCADA connection established to {self._config.analyzer_type.value}"
            )
            return True

        except Exception as e:
            self._logger.error(f"Failed to connect to SCADA: {e}")
            self._connected = False
            self._stats["errors"] += 1

            # Enable buffering if configured
            if self._config.enable_buffering:
                self._is_buffering = True

            # Schedule reconnection
            if self._reconnect_attempts < self._config.max_reconnect_attempts:
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())

            return False

    async def _connect_opcua(self) -> None:
        """Establish OPC-UA connection."""
        try:
            from asyncua import Client

            # Build endpoint URL
            if self._config.endpoint_url:
                url = self._config.endpoint_url
            else:
                url = f"opc.tcp://{self._config.host}:{self._config.port}"

            self._client = Client(url=url)

            # Set security if certificates provided
            if self._config.certificate_path and self._config.private_key_path:
                await self._client.set_security_string(
                    f"Basic256Sha256,SignAndEncrypt,"
                    f"{self._config.certificate_path},{self._config.private_key_path}"
                )

            # Set authentication
            if self._config.username and self._config.password:
                self._client.set_user(self._config.username)
                self._client.set_password(self._config.password)

            # Connect with timeout
            await asyncio.wait_for(
                self._client.connect(),
                timeout=self._config.connection_timeout
            )

            self._logger.info(f"OPC-UA client connected to {url}")

        except ImportError:
            raise ImportError(
                "asyncua package required for OPC-UA. Install with: pip install asyncua"
            )
        except asyncio.TimeoutError:
            raise ConnectionError("OPC-UA connection timeout")
        except Exception as e:
            raise ConnectionError(f"OPC-UA connection failed: {e}")

    async def _connect_modbus(self) -> None:
        """Establish Modbus connection."""
        try:
            from pymodbus.client import AsyncModbusTcpClient, AsyncModbusSerialClient

            if self._config.protocol == ConnectionProtocol.MODBUS_TCP:
                self._client = AsyncModbusTcpClient(
                    host=self._config.host,
                    port=self._config.port,
                    timeout=self._config.modbus_timeout,
                )
            else:  # MODBUS_RTU
                self._client = AsyncModbusSerialClient(
                    port=self._config.host,  # Serial port path
                    timeout=self._config.modbus_timeout,
                )

            await self._client.connect()

            if not self._client.connected:
                raise ConnectionError("Modbus client failed to connect")

            self._logger.info(
                f"Modbus client connected to {self._config.host}:{self._config.port}"
            )

        except ImportError:
            raise ImportError(
                "pymodbus package required for Modbus. Install with: pip install pymodbus"
            )
        except Exception as e:
            raise ConnectionError(f"Modbus connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from SCADA system."""
        self._logger.info("Disconnecting from SCADA system...")

        # Stop background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        # Cancel subscriptions
        for task in self._subscriptions.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._subscriptions.clear()

        # Disconnect client
        if self._client:
            try:
                if self._config.protocol == ConnectionProtocol.OPC_UA:
                    await self._client.disconnect()
                elif self._config.protocol in [
                    ConnectionProtocol.MODBUS_TCP,
                    ConnectionProtocol.MODBUS_RTU
                ]:
                    self._client.close()
            except Exception as e:
                self._logger.warning(f"Error during disconnect: {e}")

        self._connected = False
        self._logger.info("SCADA disconnected")

    async def _reconnect_loop(self) -> None:
        """Background task for reconnection attempts."""
        while self._reconnect_attempts < self._config.max_reconnect_attempts:
            self._reconnect_attempts += 1
            self._logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/"
                f"{self._config.max_reconnect_attempts}"
            )

            await asyncio.sleep(self._config.reconnect_interval)

            if await self.connect():
                self._stats["reconnections"] += 1
                break

        if not self._connected:
            self._logger.error("Max reconnection attempts reached. Giving up.")

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat monitoring."""
        while self._connected:
            try:
                await asyncio.sleep(30.0)

                # Check connection health
                if time.time() - self._last_heartbeat > 60.0:
                    self._logger.warning("Heartbeat timeout detected")
                    self._connected = False
                    await self.connect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Heartbeat error: {e}")

    async def _flush_disconnect_buffer(self) -> None:
        """Flush buffered data after reconnection."""
        if not self._disconnect_buffer:
            return

        self._logger.info(
            f"Flushing {len(self._disconnect_buffer)} buffered data points"
        )

        # Process buffered writes
        flushed = 0
        while self._disconnect_buffer:
            try:
                buffered_item = self._disconnect_buffer.popleft()
                tag_name = buffered_item["tag_name"]
                value = buffered_item["value"]

                await self.write_tag(tag_name, value)
                flushed += 1

            except Exception as e:
                self._logger.error(f"Error flushing buffered item: {e}")

        self._logger.info(f"Successfully flushed {flushed} buffered data points")

    def is_connected(self) -> bool:
        """Check if connected to SCADA system."""
        return self._connected

    # =========================================================================
    # Tag Management
    # =========================================================================

    def register_tag(self, tag: FlueGasTag) -> None:
        """
        Register a flue gas tag for monitoring.

        Args:
            tag: Tag definition
        """
        self._tags[tag.tag_name] = tag
        self._logger.debug(f"Registered tag: {tag.tag_name}")

    def register_tags(self, tags: List[FlueGasTag]) -> None:
        """Register multiple tags."""
        for tag in tags:
            self.register_tag(tag)

    def get_tag_definition(self, tag_name: str) -> Optional[FlueGasTag]:
        """Get tag definition."""
        return self._tags.get(tag_name)

    def get_all_tags(self) -> List[FlueGasTag]:
        """Get all registered tags."""
        return list(self._tags.values())

    def get_tags_by_location(
        self,
        location: MeasurementLocation
    ) -> List[FlueGasTag]:
        """Get tags for specific location."""
        return [tag for tag in self._tags.values() if tag.location == location]

    def get_tags_by_parameter(
        self,
        parameter_type: ParameterType
    ) -> List[FlueGasTag]:
        """Get tags for specific parameter type."""
        return [
            tag for tag in self._tags.values()
            if tag.parameter_type == parameter_type
        ]

    # =========================================================================
    # Reading Tag Values
    # =========================================================================

    async def read_tag(self, tag_name: str, use_cache: bool = True) -> TagDataPoint:
        """
        Read single tag value.

        Args:
            tag_name: Tag name to read
            use_cache: Whether to use cached value if available

        Returns:
            Tag data point
        """
        # Check cache first
        if use_cache and tag_name in self._tag_cache:
            cached_value, cache_time = self._tag_cache[tag_name]
            if time.time() - cache_time < self._config.cache_ttl_seconds:
                return cached_value

        if not self._connected:
            raise ConnectionError("Not connected to SCADA system")

        tag_def = self._tags.get(tag_name)
        if not tag_def:
            raise ValueError(f"Unknown tag: {tag_name}")

        try:
            if self._config.protocol == ConnectionProtocol.OPC_UA:
                value = await self._read_opcua_tag(tag_name)
            elif self._config.protocol in [
                ConnectionProtocol.MODBUS_TCP,
                ConnectionProtocol.MODBUS_RTU
            ]:
                value = await self._read_modbus_tag(tag_name, tag_def)
            else:
                raise ValueError(f"Unsupported protocol: {self._config.protocol}")

            # Scale value
            scaled_value = self._scale_value(value, tag_def)

            # Create data point
            data_point = TagDataPoint(
                tag_name=tag_name,
                timestamp=datetime.now(timezone.utc),
                value=scaled_value,
                raw_value=value,
                quality="GOOD",
                engineering_unit=tag_def.engineering_unit,
                parameter_type=tag_def.parameter_type,
                location=tag_def.location,
            )

            # Check limits
            self._check_tag_limits(data_point, tag_def)

            # Update cache
            self._tag_cache[tag_name] = (data_point, time.time())

            # Add to historical buffer
            if self._config.enable_historical_access:
                self._historical_buffer[tag_name].append(data_point)

            self._stats["reads"] += 1
            self._stats["last_successful_read"] = datetime.now(timezone.utc)
            self._last_heartbeat = time.time()

            return data_point

        except Exception as e:
            self._logger.error(f"Error reading tag {tag_name}: {e}")
            self._stats["errors"] += 1
            raise

    async def _read_opcua_tag(self, tag_name: str) -> Union[float, int, bool]:
        """Read OPC-UA tag value."""
        try:
            # Get node
            node_id = f"ns={self._config.namespace_index};s={tag_name}"
            node = self._client.get_node(node_id)

            # Read value
            value = await node.read_value()

            return value

        except Exception as e:
            raise ValueError(f"OPC-UA read error for {tag_name}: {e}")

    async def _read_modbus_tag(
        self,
        tag_name: str,
        tag_def: FlueGasTag
    ) -> Union[float, int]:
        """Read Modbus tag value."""
        try:
            # Parse tag address (format: "40001" for holding register)
            address = int(tag_name)

            if tag_def.tag_type == TagType.ANALOG_INPUT:
                # Read input register
                result = await self._client.read_input_registers(
                    address=address,
                    count=1,
                    slave=self._config.modbus_unit_id
                )
            elif tag_def.tag_type == TagType.ANALOG_OUTPUT:
                # Read holding register
                result = await self._client.read_holding_registers(
                    address=address,
                    count=1,
                    slave=self._config.modbus_unit_id
                )
            elif tag_def.tag_type == TagType.DIGITAL_INPUT:
                # Read discrete input
                result = await self._client.read_discrete_inputs(
                    address=address,
                    count=1,
                    slave=self._config.modbus_unit_id
                )
            else:
                # Read coil
                result = await self._client.read_coils(
                    address=address,
                    count=1,
                    slave=self._config.modbus_unit_id
                )

            if result.isError():
                raise ValueError(f"Modbus error: {result}")

            return result.registers[0] if hasattr(result, 'registers') else result.bits[0]

        except Exception as e:
            raise ValueError(f"Modbus read error for {tag_name}: {e}")

    def _scale_value(self, raw_value: Union[float, int], tag_def: FlueGasTag) -> float:
        """Scale raw value to engineering units."""
        if isinstance(raw_value, bool):
            return 1.0 if raw_value else 0.0

        # Linear scaling
        raw_range = tag_def.raw_max - tag_def.raw_min
        scaled_range = tag_def.scaled_max - tag_def.scaled_min

        if raw_range == 0:
            return tag_def.scaled_min

        scaled = (
            (raw_value - tag_def.raw_min) / raw_range * scaled_range
            + tag_def.scaled_min
        )

        return round(scaled, 3)

    def _check_tag_limits(
        self,
        data_point: TagDataPoint,
        tag_def: FlueGasTag
    ) -> None:
        """Check tag value against limits and generate alarms."""
        value = data_point.value

        if not isinstance(value, (int, float)):
            return

        # Check critical alarms
        if tag_def.high_alarm_limit and value > tag_def.high_alarm_limit:
            self._generate_alarm(
                tag_def.tag_name,
                AlarmSeverity.CRITICAL,
                f"High alarm: {value} > {tag_def.high_alarm_limit} {tag_def.engineering_unit}",
                value,
                tag_def.high_alarm_limit
            )
        elif tag_def.low_alarm_limit and value < tag_def.low_alarm_limit:
            self._generate_alarm(
                tag_def.tag_name,
                AlarmSeverity.CRITICAL,
                f"Low alarm: {value} < {tag_def.low_alarm_limit} {tag_def.engineering_unit}",
                value,
                tag_def.low_alarm_limit
            )
        # Check warnings
        elif tag_def.high_warning_limit and value > tag_def.high_warning_limit:
            self._generate_alarm(
                tag_def.tag_name,
                AlarmSeverity.HIGH,
                f"High warning: {value} > {tag_def.high_warning_limit} {tag_def.engineering_unit}",
                value,
                tag_def.high_warning_limit
            )
        elif tag_def.low_warning_limit and value < tag_def.low_warning_limit:
            self._generate_alarm(
                tag_def.tag_name,
                AlarmSeverity.HIGH,
                f"Low warning: {value} < {tag_def.low_warning_limit} {tag_def.engineering_unit}",
                value,
                tag_def.low_warning_limit
            )
        else:
            # Clear any existing alarm for this tag
            self._clear_alarm(tag_def.tag_name)

    async def read_tags(
        self,
        tag_names: List[str],
        use_cache: bool = True
    ) -> Dict[str, TagDataPoint]:
        """
        Read multiple tags efficiently.

        Args:
            tag_names: List of tag names
            use_cache: Whether to use cached values

        Returns:
            Dictionary of tag_name -> data_point
        """
        results = {}

        # Process in batches
        batch_size = self._config.batch_read_size
        for i in range(0, len(tag_names), batch_size):
            batch = tag_names[i:i+batch_size]

            # Read batch concurrently
            tasks = [self.read_tag(tag, use_cache) for tag in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for tag_name, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self._logger.error(f"Error reading {tag_name}: {result}")
                else:
                    results[tag_name] = result

        return results

    # =========================================================================
    # Writing Tag Values (Setpoints)
    # =========================================================================

    async def write_tag(
        self,
        tag_name: str,
        value: Union[float, int, bool]
    ) -> bool:
        """
        Write value to tag (setpoint control).

        Args:
            tag_name: Tag name to write
            value: Value to write

        Returns:
            True if write successful
        """
        tag_def = self._tags.get(tag_name)
        if not tag_def:
            raise ValueError(f"Unknown tag: {tag_name}")

        if tag_def.tag_type not in [TagType.ANALOG_OUTPUT, TagType.DIGITAL_OUTPUT]:
            raise ValueError(f"Tag {tag_name} is not writable")

        # Buffer write if disconnected
        if not self._connected:
            if self._config.enable_buffering:
                self._disconnect_buffer.append({
                    "tag_name": tag_name,
                    "value": value,
                    "timestamp": datetime.now(timezone.utc)
                })
                self._stats["buffered_points"] += 1
                self._logger.warning(
                    f"Buffering write for {tag_name}={value} (disconnected)"
                )
                return False
            else:
                raise ConnectionError("Not connected to SCADA system")

        try:
            self._logger.info(f"Writing {value} to tag {tag_name}")

            if self._config.protocol == ConnectionProtocol.OPC_UA:
                success = await self._write_opcua_tag(tag_name, value)
            elif self._config.protocol in [
                ConnectionProtocol.MODBUS_TCP,
                ConnectionProtocol.MODBUS_RTU
            ]:
                success = await self._write_modbus_tag(tag_name, tag_def, value)
            else:
                raise ValueError(f"Unsupported protocol: {self._config.protocol}")

            if success:
                self._stats["writes"] += 1
                self._stats["last_successful_write"] = datetime.now(timezone.utc)
                self._last_heartbeat = time.time()

            return success

        except Exception as e:
            self._logger.error(f"Error writing tag {tag_name}: {e}")
            self._stats["errors"] += 1
            return False

    async def _write_opcua_tag(
        self,
        tag_name: str,
        value: Union[float, int, bool]
    ) -> bool:
        """Write to OPC-UA tag."""
        try:
            node_id = f"ns={self._config.namespace_index};s={tag_name}"
            node = self._client.get_node(node_id)

            await node.write_value(value)
            return True

        except Exception as e:
            self._logger.error(f"OPC-UA write error for {tag_name}: {e}")
            return False

    async def _write_modbus_tag(
        self,
        tag_name: str,
        tag_def: FlueGasTag,
        value: Union[float, int, bool]
    ) -> bool:
        """Write to Modbus tag."""
        try:
            address = int(tag_name)

            if tag_def.tag_type == TagType.ANALOG_OUTPUT:
                result = await self._client.write_register(
                    address=address,
                    value=int(value),
                    slave=self._config.modbus_unit_id
                )
            else:  # DIGITAL_OUTPUT
                result = await self._client.write_coil(
                    address=address,
                    value=bool(value),
                    slave=self._config.modbus_unit_id
                )

            return not result.isError()

        except Exception as e:
            self._logger.error(f"Modbus write error for {tag_name}: {e}")
            return False

    # =========================================================================
    # Historical Data
    # =========================================================================

    async def get_historical_data(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        max_points: int = 1000
    ) -> List[TagDataPoint]:
        """
        Retrieve historical data for tag.

        Args:
            tag_name: Tag name
            start_time: Start time
            end_time: End time
            max_points: Maximum points to return

        Returns:
            List of historical data points
        """
        if not self._config.enable_historical_access:
            raise ValueError("Historical data access not enabled")

        # Get from buffer
        buffer = self._historical_buffer.get(tag_name, deque())

        # Filter by time range
        filtered = [
            dp for dp in buffer
            if start_time <= dp.timestamp <= end_time
        ]

        # Limit number of points
        if len(filtered) > max_points:
            # Sample evenly
            step = len(filtered) // max_points
            filtered = filtered[::step]

        return filtered

    # =========================================================================
    # Subscription/Monitoring
    # =========================================================================

    async def subscribe_tag(
        self,
        tag_name: str,
        callback: Callable[[TagDataPoint], None]
    ) -> None:
        """
        Subscribe to tag value changes.

        Args:
            tag_name: Tag name to subscribe
            callback: Callback function for value updates
        """
        if tag_name not in self._tags:
            raise ValueError(f"Unknown tag: {tag_name}")

        # Add callback
        self._subscription_callbacks[tag_name].append(callback)

        # Start monitoring task if not already running
        if tag_name not in self._subscriptions:
            task = asyncio.create_task(self._monitor_tag(tag_name))
            self._subscriptions[tag_name] = task
            self._logger.info(f"Started subscription for tag: {tag_name}")

    async def _monitor_tag(self, tag_name: str) -> None:
        """Background task to monitor tag value."""
        tag_def = self._tags[tag_name]
        interval = tag_def.update_rate_ms / 1000.0
        last_value = None

        while tag_name in self._subscriptions:
            try:
                # Read current value
                data_point = await self.read_tag(tag_name, use_cache=False)

                # Check if value changed (outside deadband)
                if last_value is None or abs(data_point.value - last_value) > tag_def.deadband:
                    # Notify callbacks
                    for callback in self._subscription_callbacks[tag_name]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(data_point)
                            else:
                                callback(data_point)
                        except Exception as e:
                            self._logger.error(f"Callback error for {tag_name}: {e}")

                    last_value = data_point.value

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitor error for {tag_name}: {e}")
                await asyncio.sleep(interval)

    async def unsubscribe_tag(self, tag_name: str) -> None:
        """Unsubscribe from tag."""
        if tag_name in self._subscriptions:
            task = self._subscriptions.pop(tag_name)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._subscription_callbacks.pop(tag_name, None)
        self._logger.info(f"Unsubscribed from tag: {tag_name}")

    # =========================================================================
    # Alarm Management
    # =========================================================================

    def _generate_alarm(
        self,
        tag_name: str,
        severity: AlarmSeverity,
        message: str,
        current_value: float,
        setpoint: float
    ) -> None:
        """Generate alarm for tag."""
        alarm_id = f"{tag_name}_{severity.value}"

        if alarm_id not in self._active_alarms:
            alarm = AlarmData(
                alarm_id=alarm_id,
                tag_name=tag_name,
                severity=severity,
                state=AlarmState.ACTIVE,
                message=message,
                current_value=current_value,
                setpoint=setpoint,
                deviation=abs(current_value - setpoint),
            )

            self._active_alarms[alarm_id] = alarm
            self._alarm_history.append(alarm)

            self._logger.warning(f"ALARM: {message}")

    def _clear_alarm(self, tag_name: str) -> None:
        """Clear alarms for tag."""
        to_clear = [
            alarm_id for alarm_id in self._active_alarms
            if self._active_alarms[alarm_id].tag_name == tag_name
        ]

        for alarm_id in to_clear:
            alarm = self._active_alarms.pop(alarm_id)
            alarm.state = AlarmState.CLEARED
            alarm.cleared_at = datetime.now(timezone.utc)

    def get_active_alarms(self) -> List[AlarmData]:
        """Get all active alarms."""
        return list(self._active_alarms.values())

    def get_alarm_history(self, limit: int = 100) -> List[AlarmData]:
        """Get alarm history."""
        return list(self._alarm_history)[-limit:]

    async def acknowledge_alarm(
        self,
        alarm_id: str,
        acknowledged_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """Acknowledge an alarm."""
        if alarm_id in self._active_alarms:
            alarm = self._active_alarms[alarm_id]
            alarm.state = AlarmState.ACKNOWLEDGED
            alarm.acknowledged_at = datetime.now(timezone.utc)
            alarm.acknowledged_by = acknowledged_by
            alarm.notes = notes

            self._logger.info(f"Alarm {alarm_id} acknowledged by {acknowledged_by}")
            return True

        return False

    # =========================================================================
    # Statistics and Health
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "registered_tags": len(self._tags),
            "active_subscriptions": len(self._subscriptions),
            "active_alarms": len(self._active_alarms),
            "cached_tags": len(self._tag_cache),
            "buffering": self._is_buffering,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            "healthy": self._connected,
            "connected": self._connected,
            "protocol": self._config.protocol.value,
            "analyzer_type": self._config.analyzer_type.value,
            "host": self._config.host,
            "port": self._config.port,
            "last_heartbeat_age": time.time() - self._last_heartbeat,
            "statistics": self.get_statistics(),
        }

        return health


# =============================================================================
# Factory Function
# =============================================================================


def create_scada_client(
    analyzer_type: AnalyzerType = AnalyzerType.GENERIC_OPC_UA,
    protocol: ConnectionProtocol = ConnectionProtocol.OPC_UA,
    host: str = "localhost",
    port: int = 4840,
    **kwargs
) -> SCADAClient:
    """
    Factory function to create SCADA client.

    Args:
        analyzer_type: Type of flue gas analyzer
        protocol: Communication protocol
        host: SCADA server host
        port: Server port
        **kwargs: Additional configuration options

    Returns:
        Configured SCADA client
    """
    config = SCADAConfig(
        analyzer_type=analyzer_type,
        protocol=protocol,
        host=host,
        port=port,
        **kwargs
    )

    return SCADAClient(config)


# =============================================================================
# Predefined Tag Definitions for Common Flue Gas Analyzers
# =============================================================================


def create_standard_flue_gas_tags() -> List[FlueGasTag]:
    """Create standard flue gas analyzer tags for combustion optimization."""
    tags = [
        # Oxygen measurement
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
            low_warning_limit=2.0,
            high_warning_limit=8.0,
            high_alarm_limit=10.0,
            deadband=0.1,
        ),

        # CO2 measurement
        FlueGasTag(
            tag_name="FG_CO2_STACK",
            parameter_type=ParameterType.CARBON_DIOXIDE,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="%",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=20.0,
            low_alarm_limit=8.0,
            low_warning_limit=10.0,
            high_warning_limit=16.0,
            high_alarm_limit=18.0,
            deadband=0.1,
        ),

        # CO measurement
        FlueGasTag(
            tag_name="FG_CO_STACK",
            parameter_type=ParameterType.CARBON_MONOXIDE,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="ppm",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=5000.0,
            high_warning_limit=100.0,
            high_alarm_limit=400.0,
            deadband=5.0,
        ),

        # NOx measurement
        FlueGasTag(
            tag_name="FG_NOX_STACK",
            parameter_type=ParameterType.NOX,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="ppm",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=1000.0,
            high_warning_limit=200.0,
            high_alarm_limit=400.0,
            deadband=5.0,
        ),

        # Stack temperature
        FlueGasTag(
            tag_name="FG_TEMP_STACK",
            parameter_type=ParameterType.TEMPERATURE,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="°C",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=500.0,
            low_alarm_limit=80.0,
            low_warning_limit=100.0,
            high_warning_limit=250.0,
            high_alarm_limit=300.0,
            deadband=1.0,
        ),

        # Stack pressure
        FlueGasTag(
            tag_name="FG_PRESSURE_STACK",
            parameter_type=ParameterType.PRESSURE,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="Pa",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=-500.0,
            scaled_max=500.0,
            low_alarm_limit=-200.0,
            low_warning_limit=-100.0,
            high_warning_limit=100.0,
            high_alarm_limit=200.0,
            deadband=5.0,
        ),

        # Flue gas flow rate
        FlueGasTag(
            tag_name="FG_FLOW_STACK",
            parameter_type=ParameterType.FLUE_GAS_FLOW,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="kg/s",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=100.0,
            low_alarm_limit=5.0,
            low_warning_limit=10.0,
            high_warning_limit=80.0,
            high_alarm_limit=90.0,
            deadband=0.5,
        ),

        # Fuel flow rate
        FlueGasTag(
            tag_name="FUEL_FLOW",
            parameter_type=ParameterType.FUEL_FLOW,
            location=MeasurementLocation.FUEL_INLET,
            engineering_unit="kg/h",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=10000.0,
            low_alarm_limit=500.0,
            low_warning_limit=1000.0,
            high_warning_limit=8000.0,
            high_alarm_limit=9000.0,
            deadband=10.0,
        ),

        # Air flow rate
        FlueGasTag(
            tag_name="AIR_FLOW",
            parameter_type=ParameterType.AIR_FLOW,
            location=MeasurementLocation.FORCED_DRAFT_FAN_OUTLET,
            engineering_unit="kg/h",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=100000.0,
            low_alarm_limit=5000.0,
            low_warning_limit=10000.0,
            high_warning_limit=80000.0,
            high_alarm_limit=90000.0,
            deadband=100.0,
        ),

        # Air damper position (control output)
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
            low_alarm_limit=10.0,
            high_alarm_limit=95.0,
            deadband=0.5,
        ),

        # Fuel valve position (control output)
        FlueGasTag(
            tag_name="FUEL_VALVE_POS",
            parameter_type=ParameterType.VALVE_POSITION,
            location=MeasurementLocation.FUEL_INLET,
            tag_type=TagType.ANALOG_OUTPUT,
            engineering_unit="%",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=100.0,
            low_alarm_limit=5.0,
            high_alarm_limit=95.0,
            deadband=0.5,
        ),

        # O2 setpoint (control setpoint)
        FlueGasTag(
            tag_name="O2_SETPOINT",
            parameter_type=ParameterType.OXYGEN,
            location=MeasurementLocation.STACK_OUTLET,
            tag_type=TagType.ANALOG_OUTPUT,
            engineering_unit="%",
            raw_min=0.0,
            raw_max=16384.0,
            scaled_min=0.0,
            scaled_max=10.0,
            low_alarm_limit=1.5,
            low_warning_limit=2.0,
            high_warning_limit=6.0,
            high_alarm_limit=8.0,
            deadband=0.05,
        ),
    ]

    return tags


# =============================================================================
# Analyzer-Specific Tag Mappings
# =============================================================================


def create_abb_ao2000_tags() -> List[FlueGasTag]:
    """Create ABB AO2000 analyzer-specific tags."""
    tags = create_standard_flue_gas_tags()

    # ABB-specific tags
    abb_tags = [
        FlueGasTag(
            tag_name="AO2000.Channel1.O2",
            parameter_type=ParameterType.OXYGEN,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="%",
            raw_min=0.0,
            raw_max=25.0,
            scaled_min=0.0,
            scaled_max=25.0,
            analyzer_channel=1,
        ),
        FlueGasTag(
            tag_name="AO2000.Channel2.CO",
            parameter_type=ParameterType.CARBON_MONOXIDE,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="ppm",
            raw_min=0.0,
            raw_max=5000.0,
            scaled_min=0.0,
            scaled_max=5000.0,
            analyzer_channel=2,
        ),
    ]

    return tags + abb_tags


def create_sick_marsic_tags() -> List[FlueGasTag]:
    """Create SICK MARSIC analyzer-specific tags."""
    tags = create_standard_flue_gas_tags()

    # SICK-specific tags
    sick_tags = [
        FlueGasTag(
            tag_name="MARSIC.O2.Value",
            parameter_type=ParameterType.OXYGEN,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="%",
            raw_min=0.0,
            raw_max=21.0,
            scaled_min=0.0,
            scaled_max=21.0,
        ),
        FlueGasTag(
            tag_name="MARSIC.NOx.Value",
            parameter_type=ParameterType.NOX,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="ppm",
            raw_min=0.0,
            raw_max=1000.0,
            scaled_min=0.0,
            scaled_max=1000.0,
        ),
    ]

    return tags + sick_tags


def create_horiba_pg_tags() -> List[FlueGasTag]:
    """Create Horiba PG analyzer-specific tags (Modbus)."""
    # Horiba PG uses Modbus registers
    tags = [
        FlueGasTag(
            tag_name="30001",  # Modbus holding register
            parameter_type=ParameterType.OXYGEN,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="%",
            raw_min=0,
            raw_max=20000,
            scaled_min=0.0,
            scaled_max=25.0,
        ),
        FlueGasTag(
            tag_name="30002",
            parameter_type=ParameterType.CARBON_DIOXIDE,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="%",
            raw_min=0,
            raw_max=20000,
            scaled_min=0.0,
            scaled_max=20.0,
        ),
        FlueGasTag(
            tag_name="30003",
            parameter_type=ParameterType.CARBON_MONOXIDE,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="ppm",
            raw_min=0,
            raw_max=50000,
            scaled_min=0.0,
            scaled_max=5000.0,
        ),
        FlueGasTag(
            tag_name="30004",
            parameter_type=ParameterType.NOX,
            location=MeasurementLocation.STACK_OUTLET,
            engineering_unit="ppm",
            raw_min=0,
            raw_max=10000,
            scaled_min=0.0,
            scaled_max=1000.0,
        ),
    ]

    return tags
