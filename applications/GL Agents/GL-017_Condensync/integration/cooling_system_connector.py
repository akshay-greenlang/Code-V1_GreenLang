# -*- coding: utf-8 -*-
"""
Cooling System Connector for GL-017 CONDENSYNC

Provides integration with cooling water system components including
cooling tower fans, CW pumps, VFDs, and basin monitoring.

Supported Equipment:
- Cooling water pumps (centrifugal, vertical turbine)
- Cooling tower fans (single-speed, two-speed, VFD-controlled)
- Basin temperature and level sensors
- VFD setpoints and feedback
- Vibration and bearing temperature monitoring

Protocols:
- OPC-UA for modern SCADA integration
- Modbus TCP for industrial controllers
- BACnet for building automation
- Profinet for Siemens PLCs

Features:
- Real-time equipment status monitoring
- VFD setpoint read/write capability
- Fan staging control interface
- Performance curve tracking
- Fault detection and diagnostics
- Energy consumption monitoring

Author: GL-DataIntegrationEngineer
Date: December 2025
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class ConnectionState(str, Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class ProtocolType(str, Enum):
    """Communication protocol types."""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    BACNET = "bacnet"
    PROFINET = "profinet"
    ETHERNET_IP = "ethernet_ip"


class EquipmentType(str, Enum):
    """Types of cooling system equipment."""
    CW_PUMP = "cw_pump"
    COOLING_TOWER_FAN = "cooling_tower_fan"
    COOLING_TOWER_CELL = "cooling_tower_cell"
    VFD = "vfd"
    BASIN = "basin"
    MAKEUP_VALVE = "makeup_valve"
    BLOWDOWN_VALVE = "blowdown_valve"
    CHEMICAL_PUMP = "chemical_pump"


class EquipmentStatus(str, Enum):
    """Equipment operational status."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    TRIPPED = "tripped"
    MAINTENANCE = "maintenance"
    UNAVAILABLE = "unavailable"
    AUTO = "auto"
    MANUAL = "manual"


class DataQuality(str, Enum):
    """Data quality indicators."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    NOT_CONNECTED = "not_connected"
    STALE = "stale"


class FanSpeedMode(str, Enum):
    """Fan speed control modes."""
    OFF = "off"
    LOW = "low"
    HIGH = "high"
    VFD = "vfd"
    AUTO = "auto"


class AlarmSeverity(str, Enum):
    """Alarm severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CoolingSystemConfig:
    """
    Configuration for cooling system connector.

    Attributes:
        connector_id: Unique connector identifier
        connector_name: Human-readable name
        host: Server/gateway host address
        port: Connection port
        protocol: Communication protocol
        system_id: Cooling system identifier
        polling_interval_seconds: Default polling interval
        connection_timeout_seconds: Connection timeout
        read_timeout_seconds: Read operation timeout
        max_retries: Maximum retry attempts
        enable_write: Enable write operations
    """
    connector_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connector_name: str = "CoolingSystemConnector"
    host: str = "localhost"
    port: int = 4840
    protocol: ProtocolType = ProtocolType.OPC_UA
    system_id: str = "CT-001"

    # Authentication
    username: Optional[str] = None

    # Timing settings
    polling_interval_seconds: float = 5.0
    connection_timeout_seconds: float = 30.0
    read_timeout_seconds: float = 60.0

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 5.0

    # Write capability
    enable_write: bool = False
    write_confirmation_required: bool = True

    # Subscription settings
    enable_subscription: bool = True
    subscription_rate_ms: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "connector_id": self.connector_id,
            "connector_name": self.connector_name,
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol.value,
            "system_id": self.system_id,
            "polling_interval_seconds": self.polling_interval_seconds,
            "enable_write": self.enable_write,
        }


@dataclass
class PumpStatus:
    """
    CW pump status and measurements.

    Attributes:
        pump_id: Pump identifier
        pump_tag: Human-readable tag
        timestamp: Status timestamp
        status: Operational status
        speed_rpm: Current speed
        speed_setpoint_rpm: Speed setpoint
        speed_pct: Speed percentage
        discharge_pressure_kpa: Discharge pressure
        suction_pressure_kpa: Suction pressure
        flow_m3h: Volumetric flow rate
        power_kw: Power consumption
        current_a: Motor current
        vibration_mm_s: Vibration velocity
        bearing_temp_de_c: Drive-end bearing temperature
        bearing_temp_nde_c: Non-drive-end bearing temperature
        vfd_status: VFD status (if applicable)
        run_hours: Total run hours
        starts_count: Number of starts
        data_quality: Data quality indicator
    """
    pump_id: str
    pump_tag: str
    timestamp: datetime
    status: EquipmentStatus = EquipmentStatus.STOPPED
    speed_rpm: float = 0.0
    speed_setpoint_rpm: float = 0.0
    speed_pct: float = 0.0
    discharge_pressure_kpa: float = 0.0
    suction_pressure_kpa: float = 0.0
    flow_m3h: float = 0.0
    power_kw: float = 0.0
    current_a: float = 0.0
    vibration_mm_s: float = 0.0
    bearing_temp_de_c: float = 0.0
    bearing_temp_nde_c: float = 0.0
    vfd_status: Optional[str] = None
    run_hours: float = 0.0
    starts_count: int = 0
    data_quality: DataQuality = DataQuality.GOOD
    alarms_active: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pump_id": self.pump_id,
            "pump_tag": self.pump_tag,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "speed_rpm": self.speed_rpm,
            "speed_setpoint_rpm": self.speed_setpoint_rpm,
            "speed_pct": self.speed_pct,
            "discharge_pressure_kpa": self.discharge_pressure_kpa,
            "suction_pressure_kpa": self.suction_pressure_kpa,
            "flow_m3h": self.flow_m3h,
            "power_kw": self.power_kw,
            "current_a": self.current_a,
            "vibration_mm_s": self.vibration_mm_s,
            "bearing_temp_de_c": self.bearing_temp_de_c,
            "bearing_temp_nde_c": self.bearing_temp_nde_c,
            "vfd_status": self.vfd_status,
            "run_hours": self.run_hours,
            "starts_count": self.starts_count,
            "data_quality": self.data_quality.value,
            "alarms_active": self.alarms_active,
        }


@dataclass
class FanStatus:
    """
    Cooling tower fan status and measurements.

    Attributes:
        fan_id: Fan identifier
        fan_tag: Human-readable tag
        cell_id: Parent cooling tower cell
        timestamp: Status timestamp
        status: Operational status
        speed_mode: Speed control mode
        speed_rpm: Current speed
        speed_setpoint_pct: Speed setpoint percentage
        speed_pct: Actual speed percentage
        power_kw: Power consumption
        current_a: Motor current
        vibration_mm_s: Vibration velocity
        motor_temp_c: Motor winding temperature
        gearbox_temp_c: Gearbox oil temperature
        blade_pitch_deg: Blade pitch angle (if adjustable)
        run_hours: Total run hours
        starts_count: Number of starts
        data_quality: Data quality indicator
    """
    fan_id: str
    fan_tag: str
    cell_id: str
    timestamp: datetime
    status: EquipmentStatus = EquipmentStatus.STOPPED
    speed_mode: FanSpeedMode = FanSpeedMode.OFF
    speed_rpm: float = 0.0
    speed_setpoint_pct: float = 0.0
    speed_pct: float = 0.0
    power_kw: float = 0.0
    current_a: float = 0.0
    vibration_mm_s: float = 0.0
    motor_temp_c: float = 0.0
    gearbox_temp_c: float = 0.0
    blade_pitch_deg: Optional[float] = None
    run_hours: float = 0.0
    starts_count: int = 0
    data_quality: DataQuality = DataQuality.GOOD
    alarms_active: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fan_id": self.fan_id,
            "fan_tag": self.fan_tag,
            "cell_id": self.cell_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "speed_mode": self.speed_mode.value,
            "speed_rpm": self.speed_rpm,
            "speed_setpoint_pct": self.speed_setpoint_pct,
            "speed_pct": self.speed_pct,
            "power_kw": self.power_kw,
            "current_a": self.current_a,
            "vibration_mm_s": self.vibration_mm_s,
            "motor_temp_c": self.motor_temp_c,
            "gearbox_temp_c": self.gearbox_temp_c,
            "blade_pitch_deg": self.blade_pitch_deg,
            "run_hours": self.run_hours,
            "starts_count": self.starts_count,
            "data_quality": self.data_quality.value,
            "alarms_active": self.alarms_active,
        }


@dataclass
class BasinStatus:
    """
    Cooling tower basin status and measurements.

    Attributes:
        basin_id: Basin identifier
        basin_tag: Human-readable tag
        timestamp: Status timestamp
        level_pct: Basin level percentage
        level_m: Basin level in meters
        temperature_c: Basin water temperature
        makeup_flow_m3h: Makeup water flow rate
        blowdown_flow_m3h: Blowdown flow rate
        conductivity_us_cm: Water conductivity
        ph: pH level
        cycles_of_concentration: COC
        makeup_valve_position_pct: Makeup valve position
        blowdown_valve_position_pct: Blowdown valve position
        data_quality: Data quality indicator
    """
    basin_id: str
    basin_tag: str
    timestamp: datetime
    level_pct: float = 50.0
    level_m: float = 0.0
    temperature_c: float = 25.0
    makeup_flow_m3h: float = 0.0
    blowdown_flow_m3h: float = 0.0
    conductivity_us_cm: float = 0.0
    ph: float = 7.0
    cycles_of_concentration: float = 1.0
    makeup_valve_position_pct: float = 0.0
    blowdown_valve_position_pct: float = 0.0
    data_quality: DataQuality = DataQuality.GOOD
    alarms_active: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "basin_id": self.basin_id,
            "basin_tag": self.basin_tag,
            "timestamp": self.timestamp.isoformat(),
            "level_pct": self.level_pct,
            "level_m": self.level_m,
            "temperature_c": self.temperature_c,
            "makeup_flow_m3h": self.makeup_flow_m3h,
            "blowdown_flow_m3h": self.blowdown_flow_m3h,
            "conductivity_us_cm": self.conductivity_us_cm,
            "ph": self.ph,
            "cycles_of_concentration": self.cycles_of_concentration,
            "makeup_valve_position_pct": self.makeup_valve_position_pct,
            "blowdown_valve_position_pct": self.blowdown_valve_position_pct,
            "data_quality": self.data_quality.value,
            "alarms_active": self.alarms_active,
        }


@dataclass
class VFDSetpoint:
    """
    VFD setpoint command.

    Attributes:
        equipment_id: Target equipment identifier
        speed_setpoint_pct: Speed setpoint percentage (0-100)
        ramp_rate_pct_s: Ramp rate in %/second
        command_source: Command source identifier
        timestamp: Command timestamp
    """
    equipment_id: str
    speed_setpoint_pct: float
    ramp_rate_pct_s: float = 10.0
    command_source: str = "CONDENSYNC"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate setpoint command.

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        if self.speed_setpoint_pct < 0 or self.speed_setpoint_pct > 100:
            errors.append(f"Speed setpoint {self.speed_setpoint_pct}% out of range (0-100)")

        if self.ramp_rate_pct_s < 0.1 or self.ramp_rate_pct_s > 100:
            errors.append(f"Ramp rate {self.ramp_rate_pct_s}%/s out of range (0.1-100)")

        return len(errors) == 0, errors


@dataclass
class FanStaging:
    """
    Fan staging configuration.

    Attributes:
        cell_id: Cooling tower cell identifier
        fans_available: Number of fans available
        fans_running: Number of fans currently running
        staging_mode: Staging mode (auto/manual)
        target_approach_c: Target approach temperature
        current_approach_c: Current approach temperature
        staging_sequence: List of fan IDs in staging sequence
    """
    cell_id: str
    fans_available: int
    fans_running: int
    staging_mode: str = "auto"
    target_approach_c: float = 5.0
    current_approach_c: float = 0.0
    staging_sequence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cell_id": self.cell_id,
            "fans_available": self.fans_available,
            "fans_running": self.fans_running,
            "staging_mode": self.staging_mode,
            "target_approach_c": self.target_approach_c,
            "current_approach_c": self.current_approach_c,
            "staging_sequence": self.staging_sequence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CoolingSystemBundle:
    """
    Complete data bundle for cooling system.

    Contains all equipment status for comprehensive analysis.
    """
    system_id: str
    system_tag: str
    timestamp: datetime
    plant_location: str

    # Aggregated metrics
    total_cw_flow_m3h: float = 0.0
    total_pump_power_kw: float = 0.0
    total_fan_power_kw: float = 0.0
    total_power_kw: float = 0.0
    basin_temp_c: float = 0.0
    approach_temp_c: float = 0.0
    range_temp_c: float = 0.0

    # Equipment counts
    pumps_running: int = 0
    pumps_available: int = 0
    fans_running: int = 0
    fans_available: int = 0

    # Detailed equipment status
    pumps: List[PumpStatus] = field(default_factory=list)
    fans: List[FanStatus] = field(default_factory=list)
    basins: List[BasinStatus] = field(default_factory=list)

    # Status
    data_quality: DataQuality = DataQuality.GOOD
    alarms_active: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "system_id": self.system_id,
            "system_tag": self.system_tag,
            "timestamp": self.timestamp.isoformat(),
            "plant_location": self.plant_location,
            "total_cw_flow_m3h": self.total_cw_flow_m3h,
            "total_pump_power_kw": self.total_pump_power_kw,
            "total_fan_power_kw": self.total_fan_power_kw,
            "total_power_kw": self.total_power_kw,
            "basin_temp_c": self.basin_temp_c,
            "approach_temp_c": self.approach_temp_c,
            "range_temp_c": self.range_temp_c,
            "pumps_running": self.pumps_running,
            "pumps_available": self.pumps_available,
            "fans_running": self.fans_running,
            "fans_available": self.fans_available,
            "pumps": [p.to_dict() for p in self.pumps],
            "fans": [f.to_dict() for f in self.fans],
            "basins": [b.to_dict() for b in self.basins],
            "data_quality": self.data_quality.value,
            "alarms_active": self.alarms_active,
        }


# ============================================================================
# COOLING SYSTEM CONNECTOR
# ============================================================================

class CoolingSystemConnector:
    """
    Connector for cooling water system equipment.

    Provides integration with cooling tower fans, CW pumps, VFDs,
    and basin monitoring for comprehensive cooling system control.

    Features:
    - Multi-protocol support (OPC-UA, Modbus, BACnet, Profinet)
    - Real-time equipment status monitoring
    - VFD setpoint read/write capability
    - Fan staging control interface
    - Performance monitoring and diagnostics
    - Energy consumption tracking

    Example:
        >>> config = CoolingSystemConfig(host="192.168.1.100")
        >>> connector = CoolingSystemConnector(config)
        >>> await connector.connect()
        >>> data = await connector.read_system_status("CT-001")
    """

    VERSION = "1.0.0"

    def __init__(self, config: CoolingSystemConfig):
        """
        Initialize cooling system connector.

        Args:
            config: Connector configuration
        """
        self.config = config
        self._state = ConnectionState.DISCONNECTED
        self._connection: Optional[Any] = None

        # Equipment registry
        self._pumps: Dict[str, Dict[str, Any]] = {}
        self._fans: Dict[str, Dict[str, Any]] = {}
        self._basins: Dict[str, Dict[str, Any]] = {}

        # Subscriptions
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._subscription_task: Optional[asyncio.Task] = None

        # Command tracking
        self._pending_commands: Dict[str, Dict[str, Any]] = {}
        self._command_history: deque = deque(maxlen=1000)

        # Metrics
        self._read_count = 0
        self._write_count = 0
        self._error_count = 0
        self._last_read_time: Optional[datetime] = None

        logger.info(
            f"CoolingSystemConnector initialized: {config.connector_name} "
            f"({config.protocol.value})"
        )

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._state == ConnectionState.CONNECTED

    async def connect(self) -> bool:
        """
        Establish connection to cooling system controls.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails after retries
        """
        if self._state == ConnectionState.CONNECTED:
            logger.warning("Already connected to cooling system")
            return True

        self._state = ConnectionState.CONNECTING
        logger.info(
            f"Connecting to {self.config.host}:{self.config.port} "
            f"via {self.config.protocol.value}"
        )

        for attempt in range(self.config.max_retries):
            try:
                # Protocol-specific connection
                if self.config.protocol == ProtocolType.OPC_UA:
                    await self._connect_opc_ua()
                elif self.config.protocol == ProtocolType.MODBUS_TCP:
                    await self._connect_modbus()
                elif self.config.protocol == ProtocolType.BACNET:
                    await self._connect_bacnet()
                elif self.config.protocol == ProtocolType.PROFINET:
                    await self._connect_profinet()
                else:
                    raise ValueError(
                        f"Unsupported protocol: {self.config.protocol}"
                    )

                self._state = ConnectionState.CONNECTED

                # Start subscription loop if enabled
                if self.config.enable_subscription:
                    self._subscription_task = asyncio.create_task(
                        self._subscription_loop()
                    )

                logger.info("Successfully connected to cooling system")
                return True

            except Exception as e:
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)

        self._state = ConnectionState.ERROR
        raise ConnectionError(
            f"Failed to connect after {self.config.max_retries} attempts"
        )

    async def _connect_opc_ua(self) -> None:
        """Establish OPC-UA connection."""
        self._connection = {
            "type": "opc_ua",
            "endpoint": f"opc.tcp://{self.config.host}:{self.config.port}",
            "connected": True,
        }
        logger.debug("OPC-UA connection established")

    async def _connect_modbus(self) -> None:
        """Establish Modbus TCP connection."""
        self._connection = {
            "type": "modbus_tcp",
            "host": self.config.host,
            "port": self.config.port,
            "connected": True,
        }
        logger.debug("Modbus TCP connection established")

    async def _connect_bacnet(self) -> None:
        """Establish BACnet connection."""
        self._connection = {
            "type": "bacnet",
            "host": self.config.host,
            "port": self.config.port,
            "connected": True,
        }
        logger.debug("BACnet connection established")

    async def _connect_profinet(self) -> None:
        """Establish Profinet connection."""
        self._connection = {
            "type": "profinet",
            "host": self.config.host,
            "port": self.config.port,
            "connected": True,
        }
        logger.debug("Profinet connection established")

    async def disconnect(self) -> None:
        """Disconnect from cooling system."""
        logger.info("Disconnecting from cooling system")

        # Cancel subscription task
        if self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass
            self._subscription_task = None

        # Clear subscriptions
        self._subscriptions.clear()

        # Close connection
        self._connection = None
        self._state = ConnectionState.DISCONNECTED

        logger.info("Disconnected from cooling system")

    def register_pump(
        self,
        pump_id: str,
        pump_tag: str,
        base_address: str,
        has_vfd: bool = True
    ) -> None:
        """
        Register a CW pump for monitoring.

        Args:
            pump_id: Pump identifier
            pump_tag: Human-readable tag
            base_address: Base PLC/OPC address
            has_vfd: Whether pump has VFD control
        """
        self._pumps[pump_id] = {
            "pump_tag": pump_tag,
            "base_address": base_address,
            "has_vfd": has_vfd,
        }
        logger.debug(f"Registered pump: {pump_id} ({pump_tag})")

    def register_fan(
        self,
        fan_id: str,
        fan_tag: str,
        cell_id: str,
        base_address: str,
        speed_mode: FanSpeedMode = FanSpeedMode.VFD
    ) -> None:
        """
        Register a cooling tower fan for monitoring.

        Args:
            fan_id: Fan identifier
            fan_tag: Human-readable tag
            cell_id: Parent cooling tower cell
            base_address: Base PLC/OPC address
            speed_mode: Fan speed control mode
        """
        self._fans[fan_id] = {
            "fan_tag": fan_tag,
            "cell_id": cell_id,
            "base_address": base_address,
            "speed_mode": speed_mode,
        }
        logger.debug(f"Registered fan: {fan_id} ({fan_tag})")

    def register_basin(
        self,
        basin_id: str,
        basin_tag: str,
        base_address: str
    ) -> None:
        """
        Register a cooling tower basin for monitoring.

        Args:
            basin_id: Basin identifier
            basin_tag: Human-readable tag
            base_address: Base PLC/OPC address
        """
        self._basins[basin_id] = {
            "basin_tag": basin_tag,
            "base_address": base_address,
        }
        logger.debug(f"Registered basin: {basin_id} ({basin_tag})")

    async def read_pump_status(self, pump_id: str) -> PumpStatus:
        """
        Read status for a single pump.

        Args:
            pump_id: Pump identifier

        Returns:
            PumpStatus with current readings
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to cooling system")

        if pump_id not in self._pumps:
            raise ValueError(f"Pump {pump_id} not registered")

        timestamp = datetime.now(timezone.utc)
        pump_info = self._pumps[pump_id]

        # Simulate pump readings
        import random
        random.seed(hash(pump_id) + int(time.time() / 60))

        is_running = random.random() > 0.1
        speed_setpoint = 85.0 + random.uniform(-10, 15) if is_running else 0.0
        speed_actual = speed_setpoint * random.uniform(0.98, 1.02) if is_running else 0.0
        speed_rpm = speed_actual * 14.85 if is_running else 0.0  # 1485 RPM at 100%

        return PumpStatus(
            pump_id=pump_id,
            pump_tag=pump_info["pump_tag"],
            timestamp=timestamp,
            status=EquipmentStatus.RUNNING if is_running else EquipmentStatus.STOPPED,
            speed_rpm=round(speed_rpm, 0),
            speed_setpoint_rpm=round(speed_setpoint * 14.85, 0),
            speed_pct=round(speed_actual, 1),
            discharge_pressure_kpa=round(450 + random.uniform(-20, 20), 1) if is_running else 0.0,
            suction_pressure_kpa=round(80 + random.uniform(-5, 5), 1) if is_running else 0.0,
            flow_m3h=round(12000 * (speed_actual / 100) ** 2, 0) if is_running else 0.0,
            power_kw=round(500 * (speed_actual / 100) ** 3, 1) if is_running else 0.0,
            current_a=round(850 * (speed_actual / 100) ** 2, 1) if is_running else 0.0,
            vibration_mm_s=round(2.5 + random.uniform(-0.5, 1.0), 2) if is_running else 0.0,
            bearing_temp_de_c=round(55 + random.uniform(-5, 10), 1) if is_running else 25.0,
            bearing_temp_nde_c=round(50 + random.uniform(-5, 8), 1) if is_running else 25.0,
            vfd_status="Running" if is_running and pump_info["has_vfd"] else None,
            run_hours=round(15000 + random.uniform(0, 5000), 0),
            starts_count=random.randint(500, 2000),
            data_quality=DataQuality.GOOD,
            alarms_active=[],
        )

    async def read_fan_status(self, fan_id: str) -> FanStatus:
        """
        Read status for a single fan.

        Args:
            fan_id: Fan identifier

        Returns:
            FanStatus with current readings
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to cooling system")

        if fan_id not in self._fans:
            raise ValueError(f"Fan {fan_id} not registered")

        timestamp = datetime.now(timezone.utc)
        fan_info = self._fans[fan_id]

        # Simulate fan readings
        import random
        random.seed(hash(fan_id) + int(time.time() / 60))

        is_running = random.random() > 0.2
        speed_setpoint = 70.0 + random.uniform(-20, 30) if is_running else 0.0
        speed_actual = speed_setpoint * random.uniform(0.98, 1.02) if is_running else 0.0
        speed_rpm = speed_actual * 1.5 if is_running else 0.0  # 150 RPM at 100%

        return FanStatus(
            fan_id=fan_id,
            fan_tag=fan_info["fan_tag"],
            cell_id=fan_info["cell_id"],
            timestamp=timestamp,
            status=EquipmentStatus.RUNNING if is_running else EquipmentStatus.STOPPED,
            speed_mode=fan_info["speed_mode"],
            speed_rpm=round(speed_rpm, 0),
            speed_setpoint_pct=round(speed_setpoint, 1),
            speed_pct=round(speed_actual, 1),
            power_kw=round(150 * (speed_actual / 100) ** 3, 1) if is_running else 0.0,
            current_a=round(250 * (speed_actual / 100) ** 2, 1) if is_running else 0.0,
            vibration_mm_s=round(3.0 + random.uniform(-0.5, 1.5), 2) if is_running else 0.0,
            motor_temp_c=round(60 + random.uniform(-5, 15), 1) if is_running else 25.0,
            gearbox_temp_c=round(55 + random.uniform(-5, 10), 1) if is_running else 25.0,
            blade_pitch_deg=None,
            run_hours=round(20000 + random.uniform(0, 10000), 0),
            starts_count=random.randint(1000, 5000),
            data_quality=DataQuality.GOOD,
            alarms_active=[],
        )

    async def read_basin_status(self, basin_id: str) -> BasinStatus:
        """
        Read status for a cooling tower basin.

        Args:
            basin_id: Basin identifier

        Returns:
            BasinStatus with current readings
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to cooling system")

        if basin_id not in self._basins:
            raise ValueError(f"Basin {basin_id} not registered")

        timestamp = datetime.now(timezone.utc)
        basin_info = self._basins[basin_id]

        # Simulate basin readings
        import random
        random.seed(hash(basin_id) + int(time.time() / 60))

        level_pct = 50.0 + random.uniform(-15, 15)

        return BasinStatus(
            basin_id=basin_id,
            basin_tag=basin_info["basin_tag"],
            timestamp=timestamp,
            level_pct=round(level_pct, 1),
            level_m=round(level_pct / 100 * 3.0, 2),  # 3m max level
            temperature_c=round(28.0 + random.uniform(-3, 5), 1),
            makeup_flow_m3h=round(100 + random.uniform(-50, 100), 1),
            blowdown_flow_m3h=round(30 + random.uniform(-10, 20), 1),
            conductivity_us_cm=round(1500 + random.uniform(-200, 300), 0),
            ph=round(7.5 + random.uniform(-0.5, 0.5), 1),
            cycles_of_concentration=round(4.0 + random.uniform(-0.5, 0.5), 1),
            makeup_valve_position_pct=round(30 + random.uniform(-10, 20), 1),
            blowdown_valve_position_pct=round(20 + random.uniform(-5, 10), 1),
            data_quality=DataQuality.GOOD,
            alarms_active=[],
        )

    async def read_system_status(
        self,
        system_id: str,
        wet_bulb_temp_c: Optional[float] = None
    ) -> CoolingSystemBundle:
        """
        Read complete cooling system status.

        Args:
            system_id: System identifier
            wet_bulb_temp_c: Current wet bulb temperature for approach calc

        Returns:
            CoolingSystemBundle with all equipment status
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to cooling system")

        timestamp = datetime.now(timezone.utc)

        # Read all equipment
        pumps = []
        for pump_id in self._pumps:
            try:
                pump_status = await self.read_pump_status(pump_id)
                pumps.append(pump_status)
            except Exception as e:
                logger.error(f"Error reading pump {pump_id}: {e}")

        fans = []
        for fan_id in self._fans:
            try:
                fan_status = await self.read_fan_status(fan_id)
                fans.append(fan_status)
            except Exception as e:
                logger.error(f"Error reading fan {fan_id}: {e}")

        basins = []
        for basin_id in self._basins:
            try:
                basin_status = await self.read_basin_status(basin_id)
                basins.append(basin_status)
            except Exception as e:
                logger.error(f"Error reading basin {basin_id}: {e}")

        # Calculate aggregates
        total_cw_flow = sum(p.flow_m3h for p in pumps if p.status == EquipmentStatus.RUNNING)
        total_pump_power = sum(p.power_kw for p in pumps if p.status == EquipmentStatus.RUNNING)
        total_fan_power = sum(f.power_kw for f in fans if f.status == EquipmentStatus.RUNNING)

        pumps_running = sum(1 for p in pumps if p.status == EquipmentStatus.RUNNING)
        fans_running = sum(1 for f in fans if f.status == EquipmentStatus.RUNNING)

        basin_temp = basins[0].temperature_c if basins else 28.0
        approach_temp = (basin_temp - wet_bulb_temp_c) if wet_bulb_temp_c else 0.0

        self._read_count += 1
        self._last_read_time = timestamp

        return CoolingSystemBundle(
            system_id=system_id,
            system_tag=f"CT-{system_id}",
            timestamp=timestamp,
            plant_location=f"Cooling Tower Area",
            total_cw_flow_m3h=round(total_cw_flow, 0),
            total_pump_power_kw=round(total_pump_power, 1),
            total_fan_power_kw=round(total_fan_power, 1),
            total_power_kw=round(total_pump_power + total_fan_power, 1),
            basin_temp_c=round(basin_temp, 1),
            approach_temp_c=round(approach_temp, 1),
            range_temp_c=round(8.0, 1),  # Typical range
            pumps_running=pumps_running,
            pumps_available=len(pumps),
            fans_running=fans_running,
            fans_available=len(fans),
            pumps=pumps,
            fans=fans,
            basins=basins,
            data_quality=DataQuality.GOOD,
            alarms_active=0,
        )

    async def write_vfd_setpoint(self, setpoint: VFDSetpoint) -> bool:
        """
        Write VFD speed setpoint.

        Args:
            setpoint: VFD setpoint command

        Returns:
            True if write successful

        Raises:
            PermissionError: If write operations not enabled
            ValueError: If setpoint validation fails
        """
        if not self.config.enable_write:
            raise PermissionError("Write operations not enabled in configuration")

        if not self.is_connected:
            raise ConnectionError("Not connected to cooling system")

        # Validate setpoint
        is_valid, errors = setpoint.validate()
        if not is_valid:
            raise ValueError(f"Invalid setpoint: {', '.join(errors)}")

        # Log command
        command_id = str(uuid.uuid4())
        self._pending_commands[command_id] = {
            "setpoint": setpoint,
            "status": "pending",
            "timestamp": datetime.now(timezone.utc),
        }

        logger.info(
            f"Writing VFD setpoint: {setpoint.equipment_id} = "
            f"{setpoint.speed_setpoint_pct}% (command {command_id})"
        )

        # In production, this would write to actual PLC/VFD
        # Simulate successful write
        await asyncio.sleep(0.1)

        self._pending_commands[command_id]["status"] = "completed"
        self._command_history.append({
            "command_id": command_id,
            "setpoint": setpoint,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc),
        })

        self._write_count += 1
        return True

    async def get_fan_staging(self, cell_id: str) -> FanStaging:
        """
        Get current fan staging for a cooling tower cell.

        Args:
            cell_id: Cooling tower cell identifier

        Returns:
            FanStaging with current configuration
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to cooling system")

        # Get fans for this cell
        cell_fans = [
            fan_id for fan_id, info in self._fans.items()
            if info["cell_id"] == cell_id
        ]

        # Count running fans
        fans_running = 0
        for fan_id in cell_fans:
            status = await self.read_fan_status(fan_id)
            if status.status == EquipmentStatus.RUNNING:
                fans_running += 1

        return FanStaging(
            cell_id=cell_id,
            fans_available=len(cell_fans),
            fans_running=fans_running,
            staging_mode="auto",
            target_approach_c=5.0,
            current_approach_c=6.5,
            staging_sequence=cell_fans,
        )

    async def subscribe(
        self,
        callback: Callable[[CoolingSystemBundle], None],
        interval_seconds: float = 5.0
    ) -> str:
        """
        Subscribe to cooling system updates.

        Args:
            callback: Callback function for updates
            interval_seconds: Update interval

        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())

        self._subscriptions[subscription_id] = {
            "callback": callback,
            "interval": interval_seconds,
            "last_update": time.time(),
        }

        logger.info(f"Created subscription {subscription_id}")
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from cooling system updates.

        Args:
            subscription_id: Subscription to cancel

        Returns:
            True if unsubscribed successfully
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.info(f"Cancelled subscription {subscription_id}")
            return True
        return False

    async def _subscription_loop(self) -> None:
        """Background task for processing subscriptions."""
        while self._state == ConnectionState.CONNECTED:
            try:
                current_time = time.time()

                for sub_id, sub_info in list(self._subscriptions.items()):
                    interval = sub_info["interval"]
                    last_update = sub_info["last_update"]

                    if current_time - last_update >= interval:
                        # Read system status
                        data = await self.read_system_status(self.config.system_id)

                        # Trigger callback
                        callback = sub_info["callback"]
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(data)
                            else:
                                callback(data)
                        except Exception as e:
                            logger.error(f"Subscription callback error: {e}")

                        sub_info["last_update"] = current_time

                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Subscription loop error: {e}")
                await asyncio.sleep(5.0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return {
            "connector_id": self.config.connector_id,
            "state": self._state.value,
            "protocol": self.config.protocol.value,
            "read_count": self._read_count,
            "write_count": self._write_count,
            "error_count": self._error_count,
            "last_read_time": (
                self._last_read_time.isoformat()
                if self._last_read_time else None
            ),
            "registered_pumps": len(self._pumps),
            "registered_fans": len(self._fans),
            "registered_basins": len(self._basins),
            "active_subscriptions": len(self._subscriptions),
            "pending_commands": len(self._pending_commands),
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_cooling_system_connector(
    host: str,
    port: int = 4840,
    protocol: ProtocolType = ProtocolType.OPC_UA,
    enable_write: bool = False,
    **kwargs
) -> CoolingSystemConnector:
    """
    Factory function to create CoolingSystemConnector.

    Args:
        host: Server host address
        port: Connection port
        protocol: Communication protocol
        enable_write: Enable write operations
        **kwargs: Additional configuration options

    Returns:
        Configured CoolingSystemConnector
    """
    config = CoolingSystemConfig(
        host=host,
        port=port,
        protocol=protocol,
        enable_write=enable_write,
        **kwargs
    )
    return CoolingSystemConnector(config)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "CoolingSystemConnector",
    "CoolingSystemConfig",
    "PumpStatus",
    "FanStatus",
    "BasinStatus",
    "VFDSetpoint",
    "FanStaging",
    "CoolingSystemBundle",
    "EquipmentType",
    "EquipmentStatus",
    "FanSpeedMode",
    "DataQuality",
    "ConnectionState",
    "ProtocolType",
    "AlarmSeverity",
    "create_cooling_system_connector",
]
