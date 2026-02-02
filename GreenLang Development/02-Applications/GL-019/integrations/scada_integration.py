"""
SCADA/DCS Integration Module for GL-019 HEATSCHEDULER

Provides comprehensive OPC-UA and Modbus integration for heating equipment
monitoring and control. Supports furnaces, boilers, heat exchangers, and other
thermal processing equipment for optimal heat scheduling.

Supported Equipment Types:
- Industrial furnaces (batch, continuous)
- Steam boilers (fire-tube, water-tube)
- Hot water boilers
- Heat exchangers
- Industrial ovens
- Kilns and dryers
- Heat treatment equipment

Monitored Parameters:
- Equipment availability and status
- Temperature readings (process, zone, ambient)
- Power consumption (kW, kWh)
- Fuel consumption rates
- Operating efficiency
- Maintenance status

Control Capabilities:
- Temperature setpoint control
- Power setpoint control
- Start/stop commands
- Operating mode selection

Protocols:
- OPC UA (Unified Architecture)
- Modbus TCP
- Modbus RTU

Author: GreenLang Data Integration Engineering Team
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


class ConnectionProtocol(str, Enum):
    """Supported communication protocols."""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"


class EquipmentType(str, Enum):
    """Types of heating equipment."""
    # Furnaces
    BATCH_FURNACE = "batch_furnace"
    CONTINUOUS_FURNACE = "continuous_furnace"
    HEAT_TREATMENT_FURNACE = "heat_treatment_furnace"
    MELTING_FURNACE = "melting_furnace"
    REHEATING_FURNACE = "reheating_furnace"
    ANNEALING_FURNACE = "annealing_furnace"

    # Boilers
    STEAM_BOILER = "steam_boiler"
    HOT_WATER_BOILER = "hot_water_boiler"
    FIRE_TUBE_BOILER = "fire_tube_boiler"
    WATER_TUBE_BOILER = "water_tube_boiler"
    ELECTRIC_BOILER = "electric_boiler"

    # Heat exchangers
    SHELL_TUBE_HX = "shell_tube_heat_exchanger"
    PLATE_HX = "plate_heat_exchanger"
    AIR_COOLED_HX = "air_cooled_heat_exchanger"

    # Ovens and dryers
    INDUSTRIAL_OVEN = "industrial_oven"
    CONVEYOR_OVEN = "conveyor_oven"
    BATCH_OVEN = "batch_oven"
    TUNNEL_DRYER = "tunnel_dryer"
    ROTARY_DRYER = "rotary_dryer"

    # Other
    KILN = "kiln"
    AUTOCLAVE = "autoclave"
    HOT_AIR_GENERATOR = "hot_air_generator"
    THERMAL_OIL_HEATER = "thermal_oil_heater"
    GENERIC = "generic"


class EquipmentStatus(str, Enum):
    """Equipment operating status."""
    OFF = "off"
    STANDBY = "standby"
    HEATING_UP = "heating_up"
    RUNNING = "running"
    COOLING_DOWN = "cooling_down"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    EMERGENCY_STOP = "emergency_stop"
    UNAVAILABLE = "unavailable"


class OperatingMode(str, Enum):
    """Equipment operating modes."""
    AUTO = "auto"
    MANUAL = "manual"
    REMOTE = "remote"
    LOCAL = "local"
    ENERGY_SAVING = "energy_saving"
    BOOST = "boost"
    SCHEDULED = "scheduled"


class TagType(str, Enum):
    """Types of SCADA tags."""
    ANALOG_INPUT = "analog_input"
    ANALOG_OUTPUT = "analog_output"
    DIGITAL_INPUT = "digital_input"
    DIGITAL_OUTPUT = "digital_output"
    CALCULATED = "calculated"
    STATUS = "status"


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


# =============================================================================
# Pydantic Models - Configuration
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
    username: Optional[str] = Field(default=None, description="Username")
    password: Optional[str] = Field(default=None, description="Password")
    certificate_path: Optional[str] = Field(default=None, description="Certificate path")
    private_key_path: Optional[str] = Field(default=None, description="Private key path")

    # Connection management
    connection_timeout: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Connection timeout"
    )
    reconnect_interval: float = Field(
        default=5.0, ge=1.0, le=60.0, description="Reconnection interval"
    )
    max_reconnect_attempts: int = Field(
        default=10, ge=1, le=100, description="Max reconnection attempts"
    )

    # Data subscription
    subscription_interval_ms: int = Field(
        default=1000, ge=100, le=60000, description="Subscription interval"
    )
    queue_size: int = Field(default=100, ge=10, le=1000, description="Queue size")

    # Performance
    batch_read_size: int = Field(default=50, ge=1, le=500, description="Batch read size")
    cache_ttl_seconds: float = Field(default=1.0, ge=0.1, le=60.0, description="Cache TTL")

    # Buffering
    enable_buffering: bool = Field(default=True, description="Enable data buffering")
    buffer_max_size: int = Field(default=10000, ge=100, le=100000, description="Max buffer size")


# =============================================================================
# Pydantic Models - Equipment and Tags
# =============================================================================


class HeatingEquipment(BaseModel):
    """Heating equipment definition."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_name: str = Field(..., description="Equipment name")
    equipment_type: EquipmentType = Field(..., description="Equipment type")

    # Location
    plant_code: str = Field(..., description="Plant code")
    area: Optional[str] = Field(default=None, description="Plant area")
    line: Optional[str] = Field(default=None, description="Production line")

    # Specifications
    rated_power_kw: float = Field(default=0.0, ge=0, description="Rated power (kW)")
    max_temperature_c: float = Field(default=0.0, ge=0, description="Max temperature (C)")
    min_temperature_c: float = Field(default=0.0, ge=0, description="Min temperature (C)")
    ramp_rate_c_per_min: Optional[float] = Field(default=None, description="Ramp rate")
    thermal_mass_kwh_per_c: Optional[float] = Field(default=None, description="Thermal mass")

    # Heating parameters
    fuel_type: Optional[str] = Field(default=None, description="Fuel type")
    heating_medium: Optional[str] = Field(default=None, description="Heating medium")
    efficiency_percent: float = Field(default=85.0, ge=0, le=100, description="Efficiency")

    # Scheduling constraints
    min_run_time_minutes: int = Field(default=0, ge=0, description="Min run time")
    min_off_time_minutes: int = Field(default=0, ge=0, description="Min off time")
    startup_time_minutes: int = Field(default=0, ge=0, description="Startup time")
    cooldown_time_minutes: int = Field(default=0, ge=0, description="Cooldown time")

    # Current status (updated from SCADA)
    current_status: EquipmentStatus = Field(default=EquipmentStatus.UNAVAILABLE)
    current_temperature_c: Optional[float] = Field(default=None)
    current_power_kw: Optional[float] = Field(default=None)
    operating_mode: OperatingMode = Field(default=OperatingMode.MANUAL)
    available: bool = Field(default=True, description="Available for scheduling")

    # Maintenance
    next_maintenance: Optional[datetime] = Field(default=None)
    last_maintenance: Optional[datetime] = Field(default=None)
    maintenance_due: bool = Field(default=False)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class EquipmentTag(BaseModel):
    """SCADA tag for heating equipment."""

    model_config = ConfigDict(extra="allow")

    tag_name: str = Field(..., description="Tag name in SCADA")
    equipment_id: str = Field(..., description="Associated equipment ID")
    tag_type: TagType = Field(default=TagType.ANALOG_INPUT)
    parameter: str = Field(..., description="Parameter name")

    # Engineering units
    engineering_unit: str = Field(default="", description="Engineering unit")
    raw_min: float = Field(default=0.0, description="Raw value minimum")
    raw_max: float = Field(default=100.0, description="Raw value maximum")
    scaled_min: float = Field(default=0.0, description="Scaled value minimum")
    scaled_max: float = Field(default=100.0, description="Scaled value maximum")

    # Alarm limits
    low_alarm_limit: Optional[float] = Field(default=None)
    low_warning_limit: Optional[float] = Field(default=None)
    high_warning_limit: Optional[float] = Field(default=None)
    high_alarm_limit: Optional[float] = Field(default=None)

    # Settings
    deadband: float = Field(default=0.0, ge=0.0, description="Change deadband")
    update_rate_ms: int = Field(default=1000, ge=100, le=60000)
    read_only: bool = Field(default=True, description="Read-only tag")

    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Pydantic Models - Data
# =============================================================================


class EquipmentReading(BaseModel):
    """Equipment status reading."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Status
    status: EquipmentStatus = Field(..., description="Operating status")
    operating_mode: Optional[OperatingMode] = Field(default=None)
    available: bool = Field(default=True)

    # Run state
    running: bool = Field(default=False)
    run_hours: Optional[float] = Field(default=None)
    starts_count: Optional[int] = Field(default=None)

    # Faults
    fault_active: bool = Field(default=False)
    fault_code: Optional[str] = Field(default=None)
    fault_message: Optional[str] = Field(default=None)

    # Maintenance
    maintenance_due: bool = Field(default=False)
    hours_until_maintenance: Optional[float] = Field(default=None)

    quality: str = Field(default="good")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TemperatureReading(BaseModel):
    """Temperature reading from equipment."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Temperature values
    process_temperature_c: Optional[float] = Field(default=None, description="Process temp")
    setpoint_temperature_c: Optional[float] = Field(default=None, description="Setpoint")

    # Zone temperatures (for multi-zone equipment)
    zone_temperatures: Dict[str, float] = Field(
        default_factory=dict, description="Zone temperatures"
    )

    # Additional readings
    inlet_temperature_c: Optional[float] = Field(default=None)
    outlet_temperature_c: Optional[float] = Field(default=None)
    ambient_temperature_c: Optional[float] = Field(default=None)
    delta_temperature_c: Optional[float] = Field(default=None)

    # Rate of change
    ramp_rate_c_per_min: Optional[float] = Field(default=None)

    # Control status
    temperature_achieved: bool = Field(default=False)
    heating_active: bool = Field(default=False)
    cooling_active: bool = Field(default=False)

    quality: str = Field(default="good")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PowerReading(BaseModel):
    """Power consumption reading from equipment."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Power measurements
    active_power_kw: Optional[float] = Field(default=None, description="Active power")
    reactive_power_kvar: Optional[float] = Field(default=None, description="Reactive power")
    apparent_power_kva: Optional[float] = Field(default=None, description="Apparent power")
    power_factor: Optional[float] = Field(default=None, description="Power factor")

    # Energy
    energy_today_kwh: Optional[float] = Field(default=None)
    energy_total_kwh: Optional[float] = Field(default=None)
    energy_this_batch_kwh: Optional[float] = Field(default=None)

    # Demand
    demand_kw: Optional[float] = Field(default=None)
    peak_demand_kw: Optional[float] = Field(default=None)

    # Fuel (for fuel-fired equipment)
    fuel_flow_rate: Optional[float] = Field(default=None)
    fuel_consumption_today: Optional[float] = Field(default=None)

    # Efficiency
    efficiency_percent: Optional[float] = Field(default=None)
    specific_energy_kwh_per_unit: Optional[float] = Field(default=None)

    quality: str = Field(default="good")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ControlSetpoint(BaseModel):
    """Control setpoint for equipment."""

    model_config = ConfigDict(extra="allow")

    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Temperature control
    temperature_setpoint_c: Optional[float] = Field(default=None)
    zone_setpoints: Dict[str, float] = Field(default_factory=dict)

    # Power control
    power_limit_kw: Optional[float] = Field(default=None)
    power_setpoint_percent: Optional[float] = Field(default=None)

    # Operating mode
    operating_mode: Optional[OperatingMode] = Field(default=None)

    # Commands
    start_command: bool = Field(default=False)
    stop_command: bool = Field(default=False)
    emergency_stop: bool = Field(default=False)

    # Scheduling
    schedule_enabled: bool = Field(default=False)
    scheduled_start_time: Optional[datetime] = Field(default=None)
    scheduled_end_time: Optional[datetime] = Field(default=None)

    # Applied status
    applied: bool = Field(default=False)
    applied_at: Optional[datetime] = Field(default=None)

    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlarmData(BaseModel):
    """Alarm/event data."""

    model_config = ConfigDict(extra="allow")

    alarm_id: str = Field(..., description="Alarm identifier")
    equipment_id: str = Field(..., description="Equipment identifier")
    tag_name: Optional[str] = Field(default=None, description="Associated tag")
    severity: AlarmSeverity = Field(..., description="Alarm severity")
    state: AlarmState = Field(default=AlarmState.ACTIVE)

    message: str = Field(..., description="Alarm message")
    description: Optional[str] = Field(default=None)

    # Timestamps
    activated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    acknowledged_at: Optional[datetime] = Field(default=None)
    cleared_at: Optional[datetime] = Field(default=None)

    # Values
    current_value: Optional[float] = Field(default=None)
    limit_value: Optional[float] = Field(default=None)

    # Acknowledgment
    acknowledged_by: Optional[str] = Field(default=None)
    notes: Optional[str] = Field(default=None)

    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# SCADA Client Implementation
# =============================================================================


class SCADAClient:
    """
    SCADA/DCS client for heating equipment integration.

    Provides:
    - Equipment status monitoring
    - Temperature and power readings
    - Control setpoint writing
    - Alarm management
    - Connection resilience
    """

    def __init__(self, config: SCADAConfig) -> None:
        """Initialize SCADA client."""
        self._config = config
        self._logger = logger

        # Connection state
        self._connected = False
        self._client: Any = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0

        # Equipment registry
        self._equipment: Dict[str, HeatingEquipment] = {}
        self._tags: Dict[str, EquipmentTag] = {}

        # Data cache
        self._status_cache: Dict[str, Tuple[EquipmentReading, float]] = {}
        self._temp_cache: Dict[str, Tuple[TemperatureReading, float]] = {}
        self._power_cache: Dict[str, Tuple[PowerReading, float]] = {}

        # Subscriptions
        self._subscriptions: Dict[str, asyncio.Task] = {}
        self._status_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._temp_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._power_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Alarms
        self._active_alarms: Dict[str, AlarmData] = {}
        self._alarm_history: deque = deque(maxlen=1000)

        # Buffering
        self._write_buffer: deque = deque(maxlen=config.buffer_max_size)
        self._is_buffering = False

        # Statistics
        self._stats = {
            "reads": 0,
            "writes": 0,
            "errors": 0,
            "reconnections": 0,
            "buffered_writes": 0,
            "last_successful_read": None,
            "last_successful_write": None,
        }

        # Heartbeat
        self._last_heartbeat = time.time()
        self._heartbeat_task: Optional[asyncio.Task] = None

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """Establish connection to SCADA system."""
        try:
            self._logger.info(
                f"Connecting to SCADA: {self._config.protocol.value} "
                f"at {self._config.host}:{self._config.port}"
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

            # Flush buffered writes
            if self._is_buffering:
                await self._flush_write_buffer()
                self._is_buffering = False

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            self._logger.info("SCADA connection established")
            return True

        except Exception as e:
            self._logger.error(f"SCADA connection failed: {e}")
            self._connected = False
            self._stats["errors"] += 1

            if self._config.enable_buffering:
                self._is_buffering = True

            if self._reconnect_attempts < self._config.max_reconnect_attempts:
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())

            return False

    async def _connect_opcua(self) -> None:
        """Establish OPC-UA connection."""
        try:
            from asyncua import Client

            if self._config.endpoint_url:
                url = self._config.endpoint_url
            else:
                url = f"opc.tcp://{self._config.host}:{self._config.port}"

            self._client = Client(url=url)

            # Security settings
            if self._config.certificate_path and self._config.private_key_path:
                await self._client.set_security_string(
                    f"Basic256Sha256,SignAndEncrypt,"
                    f"{self._config.certificate_path},{self._config.private_key_path}"
                )

            # Authentication
            if self._config.username and self._config.password:
                self._client.set_user(self._config.username)
                self._client.set_password(self._config.password)

            await asyncio.wait_for(
                self._client.connect(),
                timeout=self._config.connection_timeout
            )

            self._logger.info(f"OPC-UA connected: {url}")

        except ImportError:
            raise ImportError("asyncua package required. Install with: pip install asyncua")
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
            else:
                self._client = AsyncModbusSerialClient(
                    port=self._config.host,
                    timeout=self._config.modbus_timeout,
                )

            await self._client.connect()

            if not self._client.connected:
                raise ConnectionError("Modbus connection failed")

            self._logger.info(f"Modbus connected: {self._config.host}:{self._config.port}")

        except ImportError:
            raise ImportError("pymodbus package required. Install with: pip install pymodbus")
        except Exception as e:
            raise ConnectionError(f"Modbus connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from SCADA system."""
        self._logger.info("Disconnecting from SCADA...")

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
                else:
                    self._client.close()
            except Exception as e:
                self._logger.warning(f"Error during disconnect: {e}")

        self._connected = False
        self._logger.info("SCADA disconnected")

    async def _reconnect_loop(self) -> None:
        """Background reconnection task."""
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
            self._logger.error("Max reconnection attempts reached")

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat monitoring."""
        while self._connected:
            try:
                await asyncio.sleep(30.0)

                if time.time() - self._last_heartbeat > 60.0:
                    self._logger.warning("Heartbeat timeout")
                    self._connected = False
                    await self.connect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Heartbeat error: {e}")

    async def _flush_write_buffer(self) -> None:
        """Flush buffered writes after reconnection."""
        if not self._write_buffer:
            return

        self._logger.info(f"Flushing {len(self._write_buffer)} buffered writes")

        flushed = 0
        while self._write_buffer:
            try:
                item = self._write_buffer.popleft()
                await self._apply_setpoint(item["equipment_id"], item["setpoint"])
                flushed += 1
            except Exception as e:
                self._logger.error(f"Error flushing buffered write: {e}")

        self._logger.info(f"Flushed {flushed} buffered writes")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    # =========================================================================
    # Equipment Management
    # =========================================================================

    def register_equipment(self, equipment: HeatingEquipment) -> None:
        """Register heating equipment for monitoring."""
        self._equipment[equipment.equipment_id] = equipment
        self._logger.debug(f"Registered equipment: {equipment.equipment_id}")

    def register_equipment_list(self, equipment_list: List[HeatingEquipment]) -> None:
        """Register multiple equipment."""
        for equipment in equipment_list:
            self.register_equipment(equipment)

    def register_tag(self, tag: EquipmentTag) -> None:
        """Register equipment tag."""
        self._tags[tag.tag_name] = tag
        self._logger.debug(f"Registered tag: {tag.tag_name}")

    def register_tags(self, tags: List[EquipmentTag]) -> None:
        """Register multiple tags."""
        for tag in tags:
            self.register_tag(tag)

    def get_equipment(self, equipment_id: str) -> Optional[HeatingEquipment]:
        """Get equipment by ID."""
        return self._equipment.get(equipment_id)

    def get_all_equipment(self) -> List[HeatingEquipment]:
        """Get all registered equipment."""
        return list(self._equipment.values())

    def get_equipment_by_type(self, equipment_type: EquipmentType) -> List[HeatingEquipment]:
        """Get equipment by type."""
        return [
            eq for eq in self._equipment.values()
            if eq.equipment_type == equipment_type
        ]

    def get_available_equipment(self) -> List[HeatingEquipment]:
        """Get equipment available for scheduling."""
        return [eq for eq in self._equipment.values() if eq.available]

    # =========================================================================
    # Reading Equipment Data
    # =========================================================================

    async def read_equipment_status(
        self,
        equipment_id: str,
        use_cache: bool = True
    ) -> EquipmentReading:
        """
        Read equipment status.

        Args:
            equipment_id: Equipment identifier
            use_cache: Use cached value if available

        Returns:
            Equipment status reading
        """
        # Check cache
        if use_cache and equipment_id in self._status_cache:
            cached, cache_time = self._status_cache[equipment_id]
            if time.time() - cache_time < self._config.cache_ttl_seconds:
                return cached

        if not self._connected:
            raise ConnectionError("Not connected to SCADA")

        equipment = self._equipment.get(equipment_id)
        if not equipment:
            raise ValueError(f"Unknown equipment: {equipment_id}")

        try:
            # Read status tags for this equipment
            status_tags = [
                tag for tag in self._tags.values()
                if tag.equipment_id == equipment_id and tag.parameter in [
                    "status", "running", "fault", "mode", "available"
                ]
            ]

            reading = EquipmentReading(
                equipment_id=equipment_id,
                timestamp=datetime.now(timezone.utc),
                status=EquipmentStatus.UNAVAILABLE,
            )

            for tag in status_tags:
                value = await self._read_tag(tag)
                if tag.parameter == "status":
                    reading.status = self._parse_status(value)
                elif tag.parameter == "running":
                    reading.running = bool(value)
                elif tag.parameter == "fault":
                    reading.fault_active = bool(value)
                elif tag.parameter == "mode":
                    reading.operating_mode = self._parse_mode(value)
                elif tag.parameter == "available":
                    reading.available = bool(value)

            # Update equipment record
            equipment.current_status = reading.status
            equipment.operating_mode = reading.operating_mode or OperatingMode.MANUAL
            equipment.available = reading.available

            # Cache result
            self._status_cache[equipment_id] = (reading, time.time())

            self._stats["reads"] += 1
            self._stats["last_successful_read"] = datetime.now(timezone.utc)
            self._last_heartbeat = time.time()

            return reading

        except Exception as e:
            self._logger.error(f"Error reading equipment status: {e}")
            self._stats["errors"] += 1
            raise

    async def read_temperature(
        self,
        equipment_id: str,
        use_cache: bool = True
    ) -> TemperatureReading:
        """
        Read equipment temperature.

        Args:
            equipment_id: Equipment identifier
            use_cache: Use cached value if available

        Returns:
            Temperature reading
        """
        # Check cache
        if use_cache and equipment_id in self._temp_cache:
            cached, cache_time = self._temp_cache[equipment_id]
            if time.time() - cache_time < self._config.cache_ttl_seconds:
                return cached

        if not self._connected:
            raise ConnectionError("Not connected to SCADA")

        equipment = self._equipment.get(equipment_id)
        if not equipment:
            raise ValueError(f"Unknown equipment: {equipment_id}")

        try:
            # Read temperature tags
            temp_tags = [
                tag for tag in self._tags.values()
                if tag.equipment_id == equipment_id and "temp" in tag.parameter.lower()
            ]

            reading = TemperatureReading(
                equipment_id=equipment_id,
                timestamp=datetime.now(timezone.utc),
            )

            for tag in temp_tags:
                value = await self._read_tag(tag)
                scaled_value = self._scale_value(value, tag)

                if tag.parameter == "process_temperature":
                    reading.process_temperature_c = scaled_value
                elif tag.parameter == "setpoint_temperature":
                    reading.setpoint_temperature_c = scaled_value
                elif tag.parameter == "inlet_temperature":
                    reading.inlet_temperature_c = scaled_value
                elif tag.parameter == "outlet_temperature":
                    reading.outlet_temperature_c = scaled_value
                elif tag.parameter.startswith("zone_"):
                    zone_name = tag.parameter.replace("zone_", "").replace("_temperature", "")
                    reading.zone_temperatures[zone_name] = scaled_value

            # Update equipment
            if reading.process_temperature_c is not None:
                equipment.current_temperature_c = reading.process_temperature_c

            # Check if temperature achieved
            if reading.process_temperature_c and reading.setpoint_temperature_c:
                reading.temperature_achieved = (
                    abs(reading.process_temperature_c - reading.setpoint_temperature_c) < 5.0
                )

            # Cache result
            self._temp_cache[equipment_id] = (reading, time.time())

            self._stats["reads"] += 1
            self._stats["last_successful_read"] = datetime.now(timezone.utc)

            return reading

        except Exception as e:
            self._logger.error(f"Error reading temperature: {e}")
            self._stats["errors"] += 1
            raise

    async def read_power(
        self,
        equipment_id: str,
        use_cache: bool = True
    ) -> PowerReading:
        """
        Read equipment power consumption.

        Args:
            equipment_id: Equipment identifier
            use_cache: Use cached value if available

        Returns:
            Power reading
        """
        # Check cache
        if use_cache and equipment_id in self._power_cache:
            cached, cache_time = self._power_cache[equipment_id]
            if time.time() - cache_time < self._config.cache_ttl_seconds:
                return cached

        if not self._connected:
            raise ConnectionError("Not connected to SCADA")

        equipment = self._equipment.get(equipment_id)
        if not equipment:
            raise ValueError(f"Unknown equipment: {equipment_id}")

        try:
            # Read power tags
            power_tags = [
                tag for tag in self._tags.values()
                if tag.equipment_id == equipment_id and tag.parameter in [
                    "active_power", "reactive_power", "power_factor",
                    "energy_total", "demand", "efficiency"
                ]
            ]

            reading = PowerReading(
                equipment_id=equipment_id,
                timestamp=datetime.now(timezone.utc),
            )

            for tag in power_tags:
                value = await self._read_tag(tag)
                scaled_value = self._scale_value(value, tag)

                if tag.parameter == "active_power":
                    reading.active_power_kw = scaled_value
                elif tag.parameter == "reactive_power":
                    reading.reactive_power_kvar = scaled_value
                elif tag.parameter == "power_factor":
                    reading.power_factor = scaled_value
                elif tag.parameter == "energy_total":
                    reading.energy_total_kwh = scaled_value
                elif tag.parameter == "demand":
                    reading.demand_kw = scaled_value
                elif tag.parameter == "efficiency":
                    reading.efficiency_percent = scaled_value

            # Update equipment
            if reading.active_power_kw is not None:
                equipment.current_power_kw = reading.active_power_kw

            # Cache result
            self._power_cache[equipment_id] = (reading, time.time())

            self._stats["reads"] += 1
            self._stats["last_successful_read"] = datetime.now(timezone.utc)

            return reading

        except Exception as e:
            self._logger.error(f"Error reading power: {e}")
            self._stats["errors"] += 1
            raise

    async def _read_tag(self, tag: EquipmentTag) -> Union[float, int, bool]:
        """Read single tag value."""
        try:
            if self._config.protocol == ConnectionProtocol.OPC_UA:
                node_id = f"ns={self._config.namespace_index};s={tag.tag_name}"
                node = self._client.get_node(node_id)
                value = await node.read_value()
                return value

            else:  # Modbus
                # Parse register address from tag name
                address = int(tag.tag_name) if tag.tag_name.isdigit() else 0

                if tag.tag_type == TagType.ANALOG_INPUT:
                    result = await self._client.read_input_registers(
                        address=address,
                        count=1,
                        slave=self._config.modbus_unit_id
                    )
                elif tag.tag_type == TagType.ANALOG_OUTPUT:
                    result = await self._client.read_holding_registers(
                        address=address,
                        count=1,
                        slave=self._config.modbus_unit_id
                    )
                elif tag.tag_type == TagType.DIGITAL_INPUT:
                    result = await self._client.read_discrete_inputs(
                        address=address,
                        count=1,
                        slave=self._config.modbus_unit_id
                    )
                else:
                    result = await self._client.read_coils(
                        address=address,
                        count=1,
                        slave=self._config.modbus_unit_id
                    )

                if result.isError():
                    raise ValueError(f"Modbus error: {result}")

                return result.registers[0] if hasattr(result, 'registers') else result.bits[0]

        except Exception as e:
            self._logger.error(f"Error reading tag {tag.tag_name}: {e}")
            raise

    def _scale_value(self, raw_value: Union[float, int], tag: EquipmentTag) -> float:
        """Scale raw value to engineering units."""
        if isinstance(raw_value, bool):
            return 1.0 if raw_value else 0.0

        raw_range = tag.raw_max - tag.raw_min
        scaled_range = tag.scaled_max - tag.scaled_min

        if raw_range == 0:
            return tag.scaled_min

        scaled = (
            (raw_value - tag.raw_min) / raw_range * scaled_range
            + tag.scaled_min
        )

        return round(scaled, 3)

    def _parse_status(self, value: Union[int, str]) -> EquipmentStatus:
        """Parse status value to enum."""
        if isinstance(value, str):
            try:
                return EquipmentStatus(value.lower())
            except ValueError:
                pass

        status_map = {
            0: EquipmentStatus.OFF,
            1: EquipmentStatus.STANDBY,
            2: EquipmentStatus.HEATING_UP,
            3: EquipmentStatus.RUNNING,
            4: EquipmentStatus.COOLING_DOWN,
            5: EquipmentStatus.MAINTENANCE,
            6: EquipmentStatus.FAULT,
            7: EquipmentStatus.EMERGENCY_STOP,
        }
        return status_map.get(int(value), EquipmentStatus.UNAVAILABLE)

    def _parse_mode(self, value: Union[int, str]) -> OperatingMode:
        """Parse mode value to enum."""
        if isinstance(value, str):
            try:
                return OperatingMode(value.lower())
            except ValueError:
                pass

        mode_map = {
            0: OperatingMode.AUTO,
            1: OperatingMode.MANUAL,
            2: OperatingMode.REMOTE,
            3: OperatingMode.LOCAL,
            4: OperatingMode.ENERGY_SAVING,
        }
        return mode_map.get(int(value), OperatingMode.MANUAL)

    # =========================================================================
    # Writing Setpoints
    # =========================================================================

    async def write_setpoint(
        self,
        equipment_id: str,
        setpoint: ControlSetpoint
    ) -> bool:
        """
        Write control setpoint to equipment.

        Args:
            equipment_id: Equipment identifier
            setpoint: Control setpoint

        Returns:
            True if write successful
        """
        equipment = self._equipment.get(equipment_id)
        if not equipment:
            raise ValueError(f"Unknown equipment: {equipment_id}")

        # Buffer if disconnected
        if not self._connected:
            if self._config.enable_buffering:
                self._write_buffer.append({
                    "equipment_id": equipment_id,
                    "setpoint": setpoint,
                    "timestamp": datetime.now(timezone.utc)
                })
                self._stats["buffered_writes"] += 1
                self._logger.warning(f"Buffering setpoint for {equipment_id}")
                return False
            else:
                raise ConnectionError("Not connected to SCADA")

        return await self._apply_setpoint(equipment_id, setpoint)

    async def _apply_setpoint(
        self,
        equipment_id: str,
        setpoint: ControlSetpoint
    ) -> bool:
        """Apply setpoint to equipment."""
        try:
            # Find writable tags for this equipment
            writable_tags = [
                tag for tag in self._tags.values()
                if tag.equipment_id == equipment_id and not tag.read_only
            ]

            success = True

            for tag in writable_tags:
                value = None

                if tag.parameter == "temperature_setpoint" and setpoint.temperature_setpoint_c:
                    value = setpoint.temperature_setpoint_c
                elif tag.parameter == "power_limit" and setpoint.power_limit_kw:
                    value = setpoint.power_limit_kw
                elif tag.parameter == "operating_mode" and setpoint.operating_mode:
                    value = self._mode_to_value(setpoint.operating_mode)
                elif tag.parameter == "start_command" and setpoint.start_command:
                    value = 1
                elif tag.parameter == "stop_command" and setpoint.stop_command:
                    value = 1
                elif tag.parameter == "emergency_stop" and setpoint.emergency_stop:
                    value = 1

                if value is not None:
                    result = await self._write_tag(tag, value)
                    if not result:
                        success = False

            if success:
                setpoint.applied = True
                setpoint.applied_at = datetime.now(timezone.utc)
                self._stats["writes"] += 1
                self._stats["last_successful_write"] = datetime.now(timezone.utc)
                self._logger.info(f"Setpoint applied to {equipment_id}")

            return success

        except Exception as e:
            self._logger.error(f"Error applying setpoint: {e}")
            self._stats["errors"] += 1
            return False

    async def _write_tag(
        self,
        tag: EquipmentTag,
        value: Union[float, int, bool]
    ) -> bool:
        """Write value to tag."""
        try:
            if self._config.protocol == ConnectionProtocol.OPC_UA:
                node_id = f"ns={self._config.namespace_index};s={tag.tag_name}"
                node = self._client.get_node(node_id)
                await node.write_value(value)
                return True

            else:  # Modbus
                address = int(tag.tag_name) if tag.tag_name.isdigit() else 0

                if tag.tag_type == TagType.ANALOG_OUTPUT:
                    result = await self._client.write_register(
                        address=address,
                        value=int(value),
                        slave=self._config.modbus_unit_id
                    )
                else:
                    result = await self._client.write_coil(
                        address=address,
                        value=bool(value),
                        slave=self._config.modbus_unit_id
                    )

                return not result.isError()

        except Exception as e:
            self._logger.error(f"Error writing tag {tag.tag_name}: {e}")
            return False

    def _mode_to_value(self, mode: OperatingMode) -> int:
        """Convert operating mode to numeric value."""
        mode_map = {
            OperatingMode.AUTO: 0,
            OperatingMode.MANUAL: 1,
            OperatingMode.REMOTE: 2,
            OperatingMode.LOCAL: 3,
            OperatingMode.ENERGY_SAVING: 4,
            OperatingMode.BOOST: 5,
            OperatingMode.SCHEDULED: 6,
        }
        return mode_map.get(mode, 1)

    # =========================================================================
    # Subscriptions
    # =========================================================================

    async def subscribe_status(
        self,
        equipment_id: str,
        callback: Callable[[EquipmentReading], None]
    ) -> None:
        """Subscribe to equipment status updates."""
        self._status_callbacks[equipment_id].append(callback)

        if equipment_id not in self._subscriptions:
            task = asyncio.create_task(self._monitor_equipment(equipment_id))
            self._subscriptions[equipment_id] = task

    async def subscribe_temperature(
        self,
        equipment_id: str,
        callback: Callable[[TemperatureReading], None]
    ) -> None:
        """Subscribe to temperature updates."""
        self._temp_callbacks[equipment_id].append(callback)

        if equipment_id not in self._subscriptions:
            task = asyncio.create_task(self._monitor_equipment(equipment_id))
            self._subscriptions[equipment_id] = task

    async def subscribe_power(
        self,
        equipment_id: str,
        callback: Callable[[PowerReading], None]
    ) -> None:
        """Subscribe to power updates."""
        self._power_callbacks[equipment_id].append(callback)

        if equipment_id not in self._subscriptions:
            task = asyncio.create_task(self._monitor_equipment(equipment_id))
            self._subscriptions[equipment_id] = task

    async def _monitor_equipment(self, equipment_id: str) -> None:
        """Background task to monitor equipment."""
        interval = self._config.subscription_interval_ms / 1000.0

        while equipment_id in self._subscriptions:
            try:
                # Read status if subscribed
                if equipment_id in self._status_callbacks:
                    status = await self.read_equipment_status(equipment_id, use_cache=False)
                    for callback in self._status_callbacks[equipment_id]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(status)
                            else:
                                callback(status)
                        except Exception as e:
                            self._logger.error(f"Status callback error: {e}")

                # Read temperature if subscribed
                if equipment_id in self._temp_callbacks:
                    temp = await self.read_temperature(equipment_id, use_cache=False)
                    for callback in self._temp_callbacks[equipment_id]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(temp)
                            else:
                                callback(temp)
                        except Exception as e:
                            self._logger.error(f"Temperature callback error: {e}")

                # Read power if subscribed
                if equipment_id in self._power_callbacks:
                    power = await self.read_power(equipment_id, use_cache=False)
                    for callback in self._power_callbacks[equipment_id]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(power)
                            else:
                                callback(power)
                        except Exception as e:
                            self._logger.error(f"Power callback error: {e}")

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitor error for {equipment_id}: {e}")
                await asyncio.sleep(interval)

    async def unsubscribe(self, equipment_id: str) -> None:
        """Unsubscribe from equipment updates."""
        if equipment_id in self._subscriptions:
            task = self._subscriptions.pop(equipment_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._status_callbacks.pop(equipment_id, None)
        self._temp_callbacks.pop(equipment_id, None)
        self._power_callbacks.pop(equipment_id, None)

    # =========================================================================
    # Alarms
    # =========================================================================

    def get_active_alarms(self) -> List[AlarmData]:
        """Get all active alarms."""
        return list(self._active_alarms.values())

    def get_equipment_alarms(self, equipment_id: str) -> List[AlarmData]:
        """Get alarms for specific equipment."""
        return [
            alarm for alarm in self._active_alarms.values()
            if alarm.equipment_id == equipment_id
        ]

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
            "registered_equipment": len(self._equipment),
            "registered_tags": len(self._tags),
            "active_subscriptions": len(self._subscriptions),
            "active_alarms": len(self._active_alarms),
            "buffered_writes": len(self._write_buffer),
            "is_buffering": self._is_buffering,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self._connected,
            "connected": self._connected,
            "protocol": self._config.protocol.value,
            "host": self._config.host,
            "port": self._config.port,
            "last_heartbeat_age": time.time() - self._last_heartbeat,
            "statistics": self.get_statistics(),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_scada_client(
    protocol: ConnectionProtocol = ConnectionProtocol.OPC_UA,
    host: str = "localhost",
    port: int = 4840,
    **kwargs
) -> SCADAClient:
    """
    Factory function to create SCADA client.

    Args:
        protocol: Communication protocol
        host: SCADA server host
        port: Server port
        **kwargs: Additional configuration options

    Returns:
        Configured SCADA client
    """
    config = SCADAConfig(
        protocol=protocol,
        host=host,
        port=port,
        **kwargs
    )

    return SCADAClient(config)


def create_heating_equipment_tags(
    equipment_id: str,
    equipment_type: EquipmentType = EquipmentType.GENERIC,
    tag_prefix: str = ""
) -> List[EquipmentTag]:
    """
    Create standard tags for heating equipment.

    Args:
        equipment_id: Equipment identifier
        equipment_type: Type of heating equipment
        tag_prefix: Prefix for tag names

    Returns:
        List of equipment tags
    """
    prefix = f"{tag_prefix}{equipment_id}_" if tag_prefix else f"{equipment_id}_"

    tags = [
        # Status tags
        EquipmentTag(
            tag_name=f"{prefix}STATUS",
            equipment_id=equipment_id,
            tag_type=TagType.ANALOG_INPUT,
            parameter="status",
            engineering_unit="",
            raw_min=0,
            raw_max=10,
            scaled_min=0,
            scaled_max=10,
        ),
        EquipmentTag(
            tag_name=f"{prefix}RUNNING",
            equipment_id=equipment_id,
            tag_type=TagType.DIGITAL_INPUT,
            parameter="running",
            engineering_unit="",
        ),
        EquipmentTag(
            tag_name=f"{prefix}FAULT",
            equipment_id=equipment_id,
            tag_type=TagType.DIGITAL_INPUT,
            parameter="fault",
            engineering_unit="",
        ),
        EquipmentTag(
            tag_name=f"{prefix}MODE",
            equipment_id=equipment_id,
            tag_type=TagType.ANALOG_INPUT,
            parameter="mode",
            engineering_unit="",
        ),
        EquipmentTag(
            tag_name=f"{prefix}AVAILABLE",
            equipment_id=equipment_id,
            tag_type=TagType.DIGITAL_INPUT,
            parameter="available",
            engineering_unit="",
        ),

        # Temperature tags
        EquipmentTag(
            tag_name=f"{prefix}PROCESS_TEMP",
            equipment_id=equipment_id,
            tag_type=TagType.ANALOG_INPUT,
            parameter="process_temperature",
            engineering_unit="C",
            raw_min=0,
            raw_max=16384,
            scaled_min=0,
            scaled_max=1000,
        ),
        EquipmentTag(
            tag_name=f"{prefix}TEMP_SP",
            equipment_id=equipment_id,
            tag_type=TagType.ANALOG_OUTPUT,
            parameter="temperature_setpoint",
            engineering_unit="C",
            raw_min=0,
            raw_max=16384,
            scaled_min=0,
            scaled_max=1000,
            read_only=False,
        ),

        # Power tags
        EquipmentTag(
            tag_name=f"{prefix}POWER_KW",
            equipment_id=equipment_id,
            tag_type=TagType.ANALOG_INPUT,
            parameter="active_power",
            engineering_unit="kW",
            raw_min=0,
            raw_max=16384,
            scaled_min=0,
            scaled_max=5000,
        ),
        EquipmentTag(
            tag_name=f"{prefix}ENERGY_KWH",
            equipment_id=equipment_id,
            tag_type=TagType.ANALOG_INPUT,
            parameter="energy_total",
            engineering_unit="kWh",
            raw_min=0,
            raw_max=65535,
            scaled_min=0,
            scaled_max=1000000,
        ),
        EquipmentTag(
            tag_name=f"{prefix}POWER_LIMIT",
            equipment_id=equipment_id,
            tag_type=TagType.ANALOG_OUTPUT,
            parameter="power_limit",
            engineering_unit="kW",
            raw_min=0,
            raw_max=16384,
            scaled_min=0,
            scaled_max=5000,
            read_only=False,
        ),

        # Control tags
        EquipmentTag(
            tag_name=f"{prefix}MODE_CMD",
            equipment_id=equipment_id,
            tag_type=TagType.ANALOG_OUTPUT,
            parameter="operating_mode",
            engineering_unit="",
            read_only=False,
        ),
        EquipmentTag(
            tag_name=f"{prefix}START_CMD",
            equipment_id=equipment_id,
            tag_type=TagType.DIGITAL_OUTPUT,
            parameter="start_command",
            engineering_unit="",
            read_only=False,
        ),
        EquipmentTag(
            tag_name=f"{prefix}STOP_CMD",
            equipment_id=equipment_id,
            tag_type=TagType.DIGITAL_OUTPUT,
            parameter="stop_command",
            engineering_unit="",
            read_only=False,
        ),
        EquipmentTag(
            tag_name=f"{prefix}ESTOP",
            equipment_id=equipment_id,
            tag_type=TagType.DIGITAL_OUTPUT,
            parameter="emergency_stop",
            engineering_unit="",
            read_only=False,
        ),
    ]

    # Add zone temperature tags for furnaces
    if equipment_type in [
        EquipmentType.BATCH_FURNACE,
        EquipmentType.CONTINUOUS_FURNACE,
        EquipmentType.HEAT_TREATMENT_FURNACE,
    ]:
        for zone in range(1, 4):
            tags.append(EquipmentTag(
                tag_name=f"{prefix}ZONE{zone}_TEMP",
                equipment_id=equipment_id,
                tag_type=TagType.ANALOG_INPUT,
                parameter=f"zone_{zone}_temperature",
                engineering_unit="C",
                raw_min=0,
                raw_max=16384,
                scaled_min=0,
                scaled_max=1000,
            ))

    return tags
