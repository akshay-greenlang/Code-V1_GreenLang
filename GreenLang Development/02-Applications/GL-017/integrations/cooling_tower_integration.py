"""
GL-017 CONDENSYNC Cooling Tower Integration Module

PLC communication for cooling tower control including fan speed control,
basin temperature monitoring, blowdown coordination, and multi-cell load balancing.

Features:
- Modbus TCP / EtherNet/IP communication
- Fan speed control with VFD integration
- Basin temperature monitoring
- Blowdown rate coordination
- Weather compensation
- Multi-cell load balancing

Author: GreenLang AI Platform
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import uuid
import struct

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class CoolingTowerError(Exception):
    """Base exception for cooling tower integration."""
    pass


class PLCConnectionError(CoolingTowerError):
    """Raised when PLC connection fails."""
    pass


class PLCReadError(CoolingTowerError):
    """Raised when reading from PLC fails."""
    pass


class PLCWriteError(CoolingTowerError):
    """Raised when writing to PLC fails."""
    pass


class LoadBalancingError(CoolingTowerError):
    """Raised when load balancing calculation fails."""
    pass


# =============================================================================
# Enums and Constants
# =============================================================================

class CellState(Enum):
    """Cooling tower cell operating states."""
    OFF = "off"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAULT = "fault"
    MAINTENANCE = "maintenance"
    STANDBY = "standby"


class FanMode(Enum):
    """Fan operating mode."""
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VFD = "vfd"  # Variable frequency drive
    AUTO = "auto"


class BlowdownMode(Enum):
    """Blowdown control mode."""
    MANUAL = "manual"
    TIMER = "timer"
    CONDUCTIVITY = "conductivity"
    AUTO = "auto"


class WeatherCondition(Enum):
    """Weather condition classifications."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    FOGGY = "foggy"
    WINDY = "windy"
    FREEZING = "freezing"


class CommunicationProtocol(Enum):
    """PLC communication protocols."""
    MODBUS_TCP = "modbus_tcp"
    ETHERNET_IP = "ethernet_ip"
    PROFINET = "profinet"


# =============================================================================
# Data Models
# =============================================================================

class CoolingTowerConfig(BaseModel):
    """Configuration for cooling tower integration."""

    plc_host: str = Field(
        ...,
        description="PLC IP address or hostname"
    )
    plc_port: int = Field(
        default=502,
        description="PLC port (502 for Modbus)"
    )
    protocol: CommunicationProtocol = Field(
        default=CommunicationProtocol.MODBUS_TCP,
        description="Communication protocol"
    )
    unit_id: int = Field(
        default=1,
        description="Modbus unit ID"
    )
    number_of_cells: int = Field(
        default=4,
        description="Number of cooling tower cells"
    )
    connection_timeout: float = Field(
        default=5.0,
        description="Connection timeout in seconds"
    )
    read_timeout: float = Field(
        default=2.0,
        description="Read timeout in seconds"
    )
    write_timeout: float = Field(
        default=2.0,
        description="Write timeout in seconds"
    )
    poll_interval: float = Field(
        default=1.0,
        description="Polling interval in seconds"
    )
    max_reconnect_attempts: int = Field(
        default=5,
        description="Maximum reconnection attempts"
    )
    reconnect_delay: float = Field(
        default=5.0,
        description="Delay between reconnection attempts"
    )

    # Register addresses (Modbus)
    base_register_address: int = Field(
        default=40001,
        description="Base register address for cooling tower data"
    )

    # Performance parameters
    design_approach: float = Field(
        default=5.0,
        description="Design approach temperature (deg C)"
    )
    design_range: float = Field(
        default=10.0,
        description="Design range temperature (deg C)"
    )
    design_wet_bulb: float = Field(
        default=25.0,
        description="Design wet bulb temperature (deg C)"
    )

    @validator("plc_port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


@dataclass
class CellStatus:
    """Status of a single cooling tower cell."""

    cell_id: int
    state: CellState
    fan_mode: FanMode
    fan_speed_percent: float  # 0-100%
    fan_motor_current: float  # Amps
    fan_motor_temperature: float  # deg C
    vibration_level: float  # mm/s
    runtime_hours: float
    last_maintenance: Optional[datetime] = None
    fault_code: int = 0
    fault_description: str = ""

    def is_healthy(self) -> bool:
        """Check if cell is in healthy state."""
        return (
            self.state in [CellState.RUNNING, CellState.STANDBY, CellState.OFF] and
            self.fault_code == 0 and
            self.vibration_level < 5.0 and
            self.fan_motor_temperature < 80.0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cell_id": self.cell_id,
            "state": self.state.value,
            "fan_mode": self.fan_mode.value,
            "fan_speed_percent": self.fan_speed_percent,
            "fan_motor_current": self.fan_motor_current,
            "fan_motor_temperature": self.fan_motor_temperature,
            "vibration_level": self.vibration_level,
            "runtime_hours": self.runtime_hours,
            "last_maintenance": (
                self.last_maintenance.isoformat() if self.last_maintenance else None
            ),
            "fault_code": self.fault_code,
            "fault_description": self.fault_description,
            "is_healthy": self.is_healthy()
        }


@dataclass
class FanSpeedCommand:
    """Command to set fan speed."""

    cell_id: int
    speed_percent: float  # 0-100%
    ramp_time_seconds: float = 30.0
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "GL-017-CONDENSYNC"

    def validate(self) -> None:
        """Validate command parameters."""
        if not 0 <= self.speed_percent <= 100:
            raise ValueError("Speed percent must be between 0 and 100")
        if self.ramp_time_seconds < 0:
            raise ValueError("Ramp time must be non-negative")


@dataclass
class BasinTemperature:
    """Basin temperature measurements."""

    hot_water_temp: float  # deg C - from condenser
    cold_water_temp: float  # deg C - to condenser
    approach: float  # deg C - cold water - wet bulb
    range_temp: float  # deg C - hot water - cold water
    wet_bulb_temp: float  # deg C
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def efficiency(self) -> float:
        """Calculate cooling tower efficiency."""
        if self.range_temp + self.approach == 0:
            return 0.0
        return (self.range_temp / (self.range_temp + self.approach)) * 100


@dataclass
class BlowdownConfig:
    """Blowdown control configuration."""

    mode: BlowdownMode
    target_cycles_of_concentration: float = 5.0
    max_conductivity_us_cm: float = 2500.0
    min_blowdown_flow_m3h: float = 5.0
    max_blowdown_flow_m3h: float = 50.0
    timer_interval_hours: float = 4.0
    timer_duration_minutes: float = 15.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "target_cycles_of_concentration": self.target_cycles_of_concentration,
            "max_conductivity_us_cm": self.max_conductivity_us_cm,
            "min_blowdown_flow_m3h": self.min_blowdown_flow_m3h,
            "max_blowdown_flow_m3h": self.max_blowdown_flow_m3h,
            "timer_interval_hours": self.timer_interval_hours,
            "timer_duration_minutes": self.timer_duration_minutes
        }


@dataclass
class WeatherData:
    """Weather data for compensation."""

    ambient_temp: float  # deg C
    wet_bulb_temp: float  # deg C
    relative_humidity: float  # %
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    condition: WeatherCondition
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def is_freezing_risk(self) -> bool:
        """Check if there is freezing risk."""
        return self.ambient_temp < 5.0 or self.wet_bulb_temp < 2.0


# =============================================================================
# Modbus Register Mapping
# =============================================================================

@dataclass
class ModbusRegisterMap:
    """Modbus register mapping for cooling tower."""

    # Base addresses (offset from base_register_address)
    # Basin temperatures
    HOT_WATER_TEMP: int = 0
    COLD_WATER_TEMP: int = 1
    WET_BULB_TEMP: int = 2

    # Water chemistry
    BASIN_CONDUCTIVITY: int = 10
    BASIN_PH: int = 11
    MAKEUP_FLOW: int = 12
    BLOWDOWN_FLOW: int = 13

    # Cell status (per cell, offset = cell_id * 20)
    CELL_STATE: int = 100  # +cell_offset
    FAN_MODE: int = 101
    FAN_SPEED_FEEDBACK: int = 102
    FAN_SPEED_SETPOINT: int = 103
    FAN_MOTOR_CURRENT: int = 104
    FAN_MOTOR_TEMP: int = 105
    VIBRATION: int = 106
    RUNTIME_HOURS_HIGH: int = 107
    RUNTIME_HOURS_LOW: int = 108
    FAULT_CODE: int = 109

    # Control registers (write)
    FAN_SPEED_CMD: int = 200  # +cell_offset
    FAN_MODE_CMD: int = 201
    CELL_START_CMD: int = 202
    CELL_STOP_CMD: int = 203

    # Weather/ambient
    AMBIENT_TEMP: int = 300
    HUMIDITY: int = 301
    WIND_SPEED: int = 302

    CELL_REGISTER_OFFSET: int = 20  # Offset per cell


# =============================================================================
# Cooling Tower Integration Class
# =============================================================================

class CoolingTowerIntegration:
    """
    PLC communication for cooling tower control.

    Provides:
    - Fan speed control with VFD integration
    - Basin temperature monitoring
    - Multi-cell load balancing
    - Blowdown rate coordination
    - Weather compensation
    - Automatic reconnection
    """

    def __init__(self, config: CoolingTowerConfig):
        """
        Initialize cooling tower integration.

        Args:
            config: Cooling tower configuration
        """
        self.config = config
        self.register_map = ModbusRegisterMap()

        self._client = None
        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None

        # Current state
        self._cell_statuses: Dict[int, CellStatus] = {}
        self._basin_temp: Optional[BasinTemperature] = None
        self._weather_data: Optional[WeatherData] = None
        self._blowdown_config = BlowdownConfig(mode=BlowdownMode.AUTO)

        # Water chemistry
        self._conductivity: float = 0.0
        self._ph: float = 7.0
        self._makeup_flow: float = 0.0
        self._blowdown_flow: float = 0.0

        # Statistics
        self._stats = {
            "reads_total": 0,
            "reads_success": 0,
            "reads_failed": 0,
            "writes_total": 0,
            "writes_success": 0,
            "writes_failed": 0,
            "reconnections": 0,
            "last_poll": None
        }

        logger.info(
            f"Cooling Tower Integration initialized for {config.plc_host}:{config.plc_port}"
        )

    @property
    def is_connected(self) -> bool:
        """Check if connected to PLC."""
        return self._connected

    async def connect(self) -> None:
        """
        Establish connection to cooling tower PLC.

        Raises:
            PLCConnectionError: If connection fails
        """
        logger.info(f"Connecting to cooling tower PLC at {self.config.plc_host}")

        try:
            await self._create_connection()
            self._connected = True

            # Start polling task
            self._poll_task = asyncio.create_task(self._poll_loop())

            logger.info("Successfully connected to cooling tower PLC")

        except Exception as e:
            logger.error(f"Failed to connect to cooling tower PLC: {e}")
            raise PLCConnectionError(f"Connection failed: {e}")

    async def _create_connection(self) -> None:
        """Create PLC connection based on protocol."""
        if self.config.protocol == CommunicationProtocol.MODBUS_TCP:
            await self._create_modbus_connection()
        elif self.config.protocol == CommunicationProtocol.ETHERNET_IP:
            await self._create_ethernet_ip_connection()
        else:
            raise PLCConnectionError(
                f"Unsupported protocol: {self.config.protocol}"
            )

    async def _create_modbus_connection(self) -> None:
        """Create Modbus TCP connection."""
        # Simulated connection for implementation
        self._client = {
            "host": self.config.plc_host,
            "port": self.config.plc_port,
            "unit_id": self.config.unit_id,
            "connected": False
        }

        await asyncio.sleep(0.1)  # Simulate connection delay
        self._client["connected"] = True

    async def _create_ethernet_ip_connection(self) -> None:
        """Create EtherNet/IP connection."""
        # Simulated connection for implementation
        self._client = {
            "host": self.config.plc_host,
            "port": self.config.plc_port,
            "connected": False
        }

        await asyncio.sleep(0.1)
        self._client["connected"] = True

    async def disconnect(self) -> None:
        """Disconnect from cooling tower PLC."""
        logger.info("Disconnecting from cooling tower PLC")

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._client:
            self._client["connected"] = False
            self._client = None

        self._connected = False
        logger.info("Disconnected from cooling tower PLC")

    async def _poll_loop(self) -> None:
        """Continuous polling loop for cooling tower data."""
        while self._connected:
            try:
                await self._poll_all_data()
                self._stats["last_poll"] = datetime.utcnow()
            except Exception as e:
                logger.error(f"Polling error: {e}")

            await asyncio.sleep(self.config.poll_interval)

    async def _poll_all_data(self) -> None:
        """Poll all cooling tower data."""
        # Poll basin temperatures
        await self._poll_basin_temperatures()

        # Poll each cell status
        for cell_id in range(self.config.number_of_cells):
            await self._poll_cell_status(cell_id)

        # Poll water chemistry
        await self._poll_water_chemistry()

    async def _poll_basin_temperatures(self) -> None:
        """Poll basin temperature data."""
        try:
            # Simulated register reads
            hot_water = await self._read_register(self.register_map.HOT_WATER_TEMP)
            cold_water = await self._read_register(self.register_map.COLD_WATER_TEMP)
            wet_bulb = await self._read_register(self.register_map.WET_BULB_TEMP)

            self._basin_temp = BasinTemperature(
                hot_water_temp=hot_water / 10.0,  # Scale from register
                cold_water_temp=cold_water / 10.0,
                wet_bulb_temp=wet_bulb / 10.0,
                approach=(cold_water - wet_bulb) / 10.0,
                range_temp=(hot_water - cold_water) / 10.0
            )

            self._stats["reads_success"] += 1

        except Exception as e:
            self._stats["reads_failed"] += 1
            logger.error(f"Failed to poll basin temperatures: {e}")

    async def _poll_cell_status(self, cell_id: int) -> None:
        """Poll status for a single cell."""
        try:
            offset = cell_id * self.register_map.CELL_REGISTER_OFFSET

            state_val = await self._read_register(
                self.register_map.CELL_STATE + offset
            )
            fan_mode_val = await self._read_register(
                self.register_map.FAN_MODE + offset
            )
            fan_speed = await self._read_register(
                self.register_map.FAN_SPEED_FEEDBACK + offset
            )
            motor_current = await self._read_register(
                self.register_map.FAN_MOTOR_CURRENT + offset
            )
            motor_temp = await self._read_register(
                self.register_map.FAN_MOTOR_TEMP + offset
            )
            vibration = await self._read_register(
                self.register_map.VIBRATION + offset
            )
            runtime_high = await self._read_register(
                self.register_map.RUNTIME_HOURS_HIGH + offset
            )
            runtime_low = await self._read_register(
                self.register_map.RUNTIME_HOURS_LOW + offset
            )
            fault_code = await self._read_register(
                self.register_map.FAULT_CODE + offset
            )

            runtime_hours = (runtime_high << 16) | runtime_low

            self._cell_statuses[cell_id] = CellStatus(
                cell_id=cell_id,
                state=self._map_cell_state(state_val),
                fan_mode=self._map_fan_mode(fan_mode_val),
                fan_speed_percent=fan_speed / 10.0,
                fan_motor_current=motor_current / 10.0,
                fan_motor_temperature=motor_temp / 10.0,
                vibration_level=vibration / 100.0,
                runtime_hours=runtime_hours,
                fault_code=fault_code,
                fault_description=self._get_fault_description(fault_code)
            )

            self._stats["reads_success"] += 1

        except Exception as e:
            self._stats["reads_failed"] += 1
            logger.error(f"Failed to poll cell {cell_id} status: {e}")

    async def _poll_water_chemistry(self) -> None:
        """Poll water chemistry data."""
        try:
            self._conductivity = (
                await self._read_register(self.register_map.BASIN_CONDUCTIVITY)
            )
            self._ph = (
                await self._read_register(self.register_map.BASIN_PH)
            ) / 100.0
            self._makeup_flow = (
                await self._read_register(self.register_map.MAKEUP_FLOW)
            ) / 10.0
            self._blowdown_flow = (
                await self._read_register(self.register_map.BLOWDOWN_FLOW)
            ) / 10.0

            self._stats["reads_success"] += 1

        except Exception as e:
            self._stats["reads_failed"] += 1
            logger.error(f"Failed to poll water chemistry: {e}")

    async def _read_register(self, address: int) -> int:
        """Read a single Modbus register."""
        self._stats["reads_total"] += 1

        # Simulated register read
        import random
        await asyncio.sleep(0.005)  # Simulate network delay

        # Return simulated values based on address
        simulated_values = {
            self.register_map.HOT_WATER_TEMP: random.randint(350, 400),
            self.register_map.COLD_WATER_TEMP: random.randint(280, 320),
            self.register_map.WET_BULB_TEMP: random.randint(220, 260),
            self.register_map.BASIN_CONDUCTIVITY: random.randint(1500, 2000),
            self.register_map.BASIN_PH: random.randint(700, 800),
            self.register_map.MAKEUP_FLOW: random.randint(100, 200),
            self.register_map.BLOWDOWN_FLOW: random.randint(50, 100),
        }

        return simulated_values.get(address, random.randint(0, 1000))

    async def _write_register(self, address: int, value: int) -> None:
        """Write a single Modbus register."""
        self._stats["writes_total"] += 1

        # Simulated register write
        await asyncio.sleep(0.01)  # Simulate network delay
        self._stats["writes_success"] += 1

    def _map_cell_state(self, value: int) -> CellState:
        """Map register value to CellState."""
        state_map = {
            0: CellState.OFF,
            1: CellState.STARTING,
            2: CellState.RUNNING,
            3: CellState.STOPPING,
            4: CellState.FAULT,
            5: CellState.MAINTENANCE,
            6: CellState.STANDBY
        }
        return state_map.get(value, CellState.OFF)

    def _map_fan_mode(self, value: int) -> FanMode:
        """Map register value to FanMode."""
        mode_map = {
            0: FanMode.OFF,
            1: FanMode.LOW,
            2: FanMode.MEDIUM,
            3: FanMode.HIGH,
            4: FanMode.VFD,
            5: FanMode.AUTO
        }
        return mode_map.get(value, FanMode.OFF)

    def _get_fault_description(self, fault_code: int) -> str:
        """Get fault description from code."""
        fault_descriptions = {
            0: "",
            1: "Motor overload",
            2: "High vibration",
            3: "Motor overtemperature",
            4: "VFD fault",
            5: "Communication failure",
            6: "Low oil level",
            7: "Bearing failure",
            8: "Belt slip",
            9: "Gearbox fault"
        }
        return fault_descriptions.get(fault_code, f"Unknown fault ({fault_code})")

    async def set_fan_speed(self, command: FanSpeedCommand) -> bool:
        """
        Set fan speed for a cell.

        Args:
            command: Fan speed command

        Returns:
            True if successful

        Raises:
            PLCWriteError: If write fails
        """
        try:
            command.validate()

            if command.cell_id >= self.config.number_of_cells:
                raise PLCWriteError(f"Invalid cell ID: {command.cell_id}")

            offset = command.cell_id * self.register_map.CELL_REGISTER_OFFSET
            speed_register = int(command.speed_percent * 10)

            await self._write_register(
                self.register_map.FAN_SPEED_CMD + offset,
                speed_register
            )

            logger.info(
                f"Set fan speed for cell {command.cell_id} to {command.speed_percent}% "
                f"(command_id={command.command_id})"
            )

            return True

        except Exception as e:
            self._stats["writes_failed"] += 1
            logger.error(f"Failed to set fan speed: {e}")
            raise PLCWriteError(f"Failed to set fan speed: {e}")

    async def set_all_fan_speeds(
        self,
        speed_percent: float,
        exclude_cells: Optional[List[int]] = None
    ) -> Dict[int, bool]:
        """
        Set fan speed for all cells.

        Args:
            speed_percent: Target speed percentage
            exclude_cells: Cells to exclude

        Returns:
            Dictionary mapping cell IDs to success status
        """
        exclude_cells = exclude_cells or []
        results = {}

        for cell_id in range(self.config.number_of_cells):
            if cell_id in exclude_cells:
                continue

            command = FanSpeedCommand(
                cell_id=cell_id,
                speed_percent=speed_percent
            )

            try:
                success = await self.set_fan_speed(command)
                results[cell_id] = success
            except Exception as e:
                logger.error(f"Failed to set speed for cell {cell_id}: {e}")
                results[cell_id] = False

        return results

    async def balance_cell_load(
        self,
        total_duty_kw: float,
        weather_data: Optional[WeatherData] = None
    ) -> List[FanSpeedCommand]:
        """
        Calculate optimal fan speeds for load balancing.

        Args:
            total_duty_kw: Total cooling duty required (kW)
            weather_data: Current weather conditions

        Returns:
            List of fan speed commands for each cell
        """
        commands = []

        # Get healthy cells
        healthy_cells = [
            cell_id for cell_id, status in self._cell_statuses.items()
            if status.is_healthy() and status.state != CellState.MAINTENANCE
        ]

        if not healthy_cells:
            raise LoadBalancingError("No healthy cells available")

        # Calculate required fan speeds based on duty
        # Using simplified cooling tower performance model
        design_capacity_per_cell_kw = 5000.0  # Configurable
        total_design_capacity = design_capacity_per_cell_kw * len(healthy_cells)

        load_factor = min(total_duty_kw / total_design_capacity, 1.0)

        # Weather compensation
        if weather_data:
            compensation = self._calculate_weather_compensation(weather_data)
            load_factor *= compensation

        # Calculate base speed
        base_speed = self._load_to_fan_speed(load_factor)

        # Adjust for cell runtime (favor cells with less runtime)
        runtime_adjustments = self._calculate_runtime_adjustments(healthy_cells)

        for cell_id in healthy_cells:
            adjusted_speed = base_speed * runtime_adjustments.get(cell_id, 1.0)
            adjusted_speed = max(0.0, min(100.0, adjusted_speed))

            commands.append(FanSpeedCommand(
                cell_id=cell_id,
                speed_percent=adjusted_speed
            ))

        return commands

    def _load_to_fan_speed(self, load_factor: float) -> float:
        """Convert load factor to fan speed percentage."""
        # Fan laws: flow proportional to speed, heat transfer proportional to flow^0.8
        # Simplified relationship
        if load_factor <= 0:
            return 0.0
        elif load_factor >= 1.0:
            return 100.0
        else:
            # Cubic relationship approximation
            return (load_factor ** 0.33) * 100.0

    def _calculate_weather_compensation(self, weather_data: WeatherData) -> float:
        """Calculate weather compensation factor."""
        compensation = 1.0

        # Higher wet bulb = more fan speed needed
        wet_bulb_deviation = weather_data.wet_bulb_temp - self.config.design_wet_bulb
        compensation += wet_bulb_deviation * 0.02

        # High humidity reduces evaporation
        if weather_data.relative_humidity > 80:
            compensation += 0.1

        # Wind can assist cooling
        if weather_data.wind_speed > 5.0:
            compensation -= 0.05

        # Freezing protection
        if weather_data.is_freezing_risk():
            compensation = max(compensation, 0.3)  # Minimum speed for freeze protection

        return max(0.5, min(1.5, compensation))

    def _calculate_runtime_adjustments(
        self,
        cell_ids: List[int]
    ) -> Dict[int, float]:
        """Calculate runtime-based adjustments for load balancing."""
        adjustments = {}

        if not cell_ids:
            return adjustments

        runtimes = {
            cell_id: self._cell_statuses[cell_id].runtime_hours
            for cell_id in cell_ids
        }

        avg_runtime = sum(runtimes.values()) / len(runtimes)

        if avg_runtime == 0:
            return {cell_id: 1.0 for cell_id in cell_ids}

        for cell_id, runtime in runtimes.items():
            # Favor cells with less runtime
            ratio = runtime / avg_runtime
            if ratio > 1.2:
                adjustments[cell_id] = 0.9  # Reduce load
            elif ratio < 0.8:
                adjustments[cell_id] = 1.1  # Increase load
            else:
                adjustments[cell_id] = 1.0

        return adjustments

    async def set_blowdown_config(self, config: BlowdownConfig) -> bool:
        """
        Set blowdown control configuration.

        Args:
            config: Blowdown configuration

        Returns:
            True if successful
        """
        try:
            self._blowdown_config = config
            # Write to PLC registers (simulated)
            logger.info(f"Updated blowdown configuration: {config.to_dict()}")
            return True
        except Exception as e:
            logger.error(f"Failed to set blowdown config: {e}")
            return False

    async def get_blowdown_recommendation(self) -> Dict[str, Any]:
        """
        Get blowdown recommendation based on current conditions.

        Returns:
            Dictionary with blowdown recommendations
        """
        cycles_of_concentration = self._conductivity / max(
            self._makeup_flow * 10, 1
        )  # Simplified calculation

        recommendation = {
            "current_conductivity": self._conductivity,
            "target_conductivity": self._blowdown_config.max_conductivity_us_cm,
            "current_cycles": cycles_of_concentration,
            "target_cycles": self._blowdown_config.target_cycles_of_concentration,
            "action": "none",
            "recommended_flow_m3h": 0.0
        }

        if self._conductivity > self._blowdown_config.max_conductivity_us_cm:
            recommendation["action"] = "increase_blowdown"
            recommendation["recommended_flow_m3h"] = min(
                self._blowdown_config.max_blowdown_flow_m3h,
                self._blowdown_flow * 1.5
            )
        elif self._conductivity < self._blowdown_config.max_conductivity_us_cm * 0.7:
            recommendation["action"] = "decrease_blowdown"
            recommendation["recommended_flow_m3h"] = max(
                self._blowdown_config.min_blowdown_flow_m3h,
                self._blowdown_flow * 0.7
            )

        return recommendation

    def update_weather_data(self, weather_data: WeatherData) -> None:
        """Update current weather data for compensation."""
        self._weather_data = weather_data
        logger.debug(f"Weather data updated: {weather_data}")

    def get_cell_status(self, cell_id: int) -> Optional[CellStatus]:
        """Get status for a specific cell."""
        return self._cell_statuses.get(cell_id)

    def get_all_cell_statuses(self) -> Dict[int, CellStatus]:
        """Get status for all cells."""
        return dict(self._cell_statuses)

    def get_basin_temperature(self) -> Optional[BasinTemperature]:
        """Get current basin temperature data."""
        return self._basin_temp

    def get_water_chemistry(self) -> Dict[str, float]:
        """Get current water chemistry data."""
        return {
            "conductivity_us_cm": self._conductivity,
            "ph": self._ph,
            "makeup_flow_m3h": self._makeup_flow,
            "blowdown_flow_m3h": self._blowdown_flow
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "cells_monitored": len(self._cell_statuses),
            "healthy_cells": sum(
                1 for s in self._cell_statuses.values() if s.is_healthy()
            )
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on cooling tower integration.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "connected": self._connected,
            "timestamp": datetime.utcnow().isoformat()
        }

        if not self._connected:
            health["status"] = "unhealthy"
            health["reason"] = "Not connected to PLC"
            return health

        # Check cell statuses
        unhealthy_cells = [
            cell_id for cell_id, status in self._cell_statuses.items()
            if not status.is_healthy()
        ]

        if unhealthy_cells:
            health["status"] = "degraded"
            health["unhealthy_cells"] = unhealthy_cells
            health["reason"] = f"Cells {unhealthy_cells} are unhealthy"

        # Check data freshness
        if self._stats["last_poll"]:
            age = (datetime.utcnow() - self._stats["last_poll"]).total_seconds()
            if age > self.config.poll_interval * 5:
                health["status"] = "degraded"
                health["data_stale"] = True
                health["reason"] = f"Data is {age:.1f}s old"

        return health
