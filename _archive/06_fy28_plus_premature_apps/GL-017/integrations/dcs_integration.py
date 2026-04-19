"""
GL-017 CONDENSYNC DCS Integration Module

Distributed Control System interface for condenser controls including
cascade control coordination, interlock monitoring, and mode selection.

Features:
- DCS interface for condenser controls
- Cascade control coordination
- Interlock status monitoring
- Mode selection (manual/auto)
- Control valve position feedback

Author: GreenLang AI Platform
Version: 1.0.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class DCSConnectionError(Exception):
    """Raised when DCS connection fails."""
    pass


class DCSControlError(Exception):
    """Raised when control operation fails."""
    pass


class DCSInterlockError(Exception):
    """Raised when interlock prevents operation."""
    pass


class DCSModeError(Exception):
    """Raised when mode change is not permitted."""
    pass


# =============================================================================
# Enums and Constants
# =============================================================================

class ControlMode(Enum):
    """Control loop operating modes."""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"
    REMOTE = "remote"
    LOCAL = "local"
    OUT_OF_SERVICE = "out_of_service"


class InterlockState(Enum):
    """Interlock states."""
    NORMAL = "normal"
    TRIPPED = "tripped"
    BYPASSED = "bypassed"
    FAULT = "fault"
    TESTING = "testing"


class ValveAction(Enum):
    """Control valve fail-safe action."""
    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"
    FAIL_LAST = "fail_last"


class AlarmPriority(Enum):
    """DCS alarm priorities."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class ControllerType(Enum):
    """Types of controllers."""
    PID = "pid"
    SPLIT_RANGE = "split_range"
    RATIO = "ratio"
    FEEDFORWARD = "feedforward"
    CASCADE_PRIMARY = "cascade_primary"
    CASCADE_SECONDARY = "cascade_secondary"


# =============================================================================
# Data Models
# =============================================================================

class DCSConfig(BaseModel):
    """Configuration for DCS integration."""

    host: str = Field(
        ...,
        description="DCS server hostname or IP"
    )
    port: int = Field(
        default=4840,
        description="DCS server port"
    )
    namespace: str = Field(
        default="Condenser",
        description="DCS namespace for condenser controls"
    )
    username: Optional[str] = Field(
        default=None,
        description="DCS authentication username"
    )
    password: Optional[str] = Field(
        default=None,
        description="DCS authentication password"
    )
    connection_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds"
    )
    request_timeout: float = Field(
        default=5.0,
        description="Request timeout in seconds"
    )
    polling_interval: float = Field(
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

    # Control limits
    min_output_change_rate: float = Field(
        default=1.0,
        description="Minimum output change rate (%/sec)"
    )
    max_output_change_rate: float = Field(
        default=10.0,
        description="Maximum output change rate (%/sec)"
    )


@dataclass
class InterlockStatus:
    """Status of an interlock."""

    interlock_id: str
    name: str
    state: InterlockState
    description: str
    trip_value: Optional[float] = None
    reset_value: Optional[float] = None
    current_value: Optional[float] = None
    trip_timestamp: Optional[datetime] = None
    bypass_timestamp: Optional[datetime] = None
    bypass_reason: Optional[str] = None
    associated_equipment: List[str] = field(default_factory=list)

    def is_safe_to_operate(self) -> bool:
        """Check if interlock allows operation."""
        return self.state in [InterlockState.NORMAL, InterlockState.BYPASSED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interlock_id": self.interlock_id,
            "name": self.name,
            "state": self.state.value,
            "description": self.description,
            "trip_value": self.trip_value,
            "reset_value": self.reset_value,
            "current_value": self.current_value,
            "trip_timestamp": (
                self.trip_timestamp.isoformat() if self.trip_timestamp else None
            ),
            "bypass_timestamp": (
                self.bypass_timestamp.isoformat() if self.bypass_timestamp else None
            ),
            "bypass_reason": self.bypass_reason,
            "associated_equipment": self.associated_equipment,
            "safe_to_operate": self.is_safe_to_operate()
        }


@dataclass
class ValvePosition:
    """Control valve position and status."""

    valve_id: str
    name: str
    position_percent: float  # 0-100%
    setpoint_percent: float  # 0-100%
    mode: ControlMode
    fail_action: ValveAction
    is_open: bool
    is_closed: bool
    in_transit: bool
    feedback_fault: bool
    motor_fault: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def deviation(self) -> float:
        """Calculate position deviation from setpoint."""
        return abs(self.position_percent - self.setpoint_percent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valve_id": self.valve_id,
            "name": self.name,
            "position_percent": self.position_percent,
            "setpoint_percent": self.setpoint_percent,
            "mode": self.mode.value,
            "fail_action": self.fail_action.value,
            "is_open": self.is_open,
            "is_closed": self.is_closed,
            "in_transit": self.in_transit,
            "feedback_fault": self.feedback_fault,
            "motor_fault": self.motor_fault,
            "deviation": self.deviation(),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CascadeController:
    """Cascade control loop status."""

    controller_id: str
    name: str
    controller_type: ControllerType
    mode: ControlMode

    # Process variable (PV)
    pv: float
    pv_units: str
    pv_high_limit: float
    pv_low_limit: float

    # Setpoint (SP)
    sp: float
    sp_high_limit: float
    sp_low_limit: float
    sp_source: str  # "local", "remote", "cascade"

    # Output (OP)
    op: float  # 0-100%
    op_high_limit: float
    op_low_limit: float

    # Tuning parameters
    kp: float  # Proportional gain
    ki: float  # Integral time (seconds)
    kd: float  # Derivative time (seconds)

    # Status
    is_tracking: bool = False
    is_clamped: bool = False
    alarm_active: bool = False
    error: float = 0.0

    # Cascade specific
    cascade_primary_id: Optional[str] = None
    cascade_secondary_ids: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "controller_id": self.controller_id,
            "name": self.name,
            "controller_type": self.controller_type.value,
            "mode": self.mode.value,
            "pv": self.pv,
            "pv_units": self.pv_units,
            "sp": self.sp,
            "op": self.op,
            "error": self.error,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "is_tracking": self.is_tracking,
            "is_clamped": self.is_clamped,
            "alarm_active": self.alarm_active,
            "cascade_primary_id": self.cascade_primary_id,
            "cascade_secondary_ids": self.cascade_secondary_ids,
            "timestamp": self.timestamp.isoformat()
        }


# =============================================================================
# Condenser Control Tags
# =============================================================================

@dataclass
class CondenserControlTags:
    """DCS tag definitions for condenser controls."""

    # Cooling water control valves
    CW_INLET_VALVE: str = "Condenser.CW.InletValve"
    CW_OUTLET_VALVE: str = "Condenser.CW.OutletValve"
    CW_BYPASS_VALVE: str = "Condenser.CW.BypassValve"

    # Condensate control
    HOTWELL_LEVEL_CONTROLLER: str = "Condenser.Hotwell.LevelController"
    MAKEUP_VALVE: str = "Condenser.Hotwell.MakeupValve"
    CONDENSATE_RECIRC_VALVE: str = "Condenser.Condensate.RecircValve"

    # Vacuum control
    VACUUM_CONTROLLER: str = "Condenser.Vacuum.Controller"
    AIR_EJECTOR_STEAM_VALVE: str = "Condenser.AirRemoval.EjectorSteamValve"
    VACUUM_BREAKER: str = "Condenser.Vacuum.BreakerValve"

    # Interlocks
    LOW_CW_FLOW_INTERLOCK: str = "Condenser.Interlocks.LowCWFlow"
    HIGH_HOTWELL_LEVEL_INTERLOCK: str = "Condenser.Interlocks.HighHotwellLevel"
    LOW_HOTWELL_LEVEL_INTERLOCK: str = "Condenser.Interlocks.LowHotwellLevel"
    HIGH_VACUUM_INTERLOCK: str = "Condenser.Interlocks.HighVacuum"
    LOW_VACUUM_INTERLOCK: str = "Condenser.Interlocks.LowVacuum"
    CONDENSER_TUBE_LEAK_INTERLOCK: str = "Condenser.Interlocks.TubeLeak"


# =============================================================================
# DCS Integration Class
# =============================================================================

class DCSIntegration:
    """
    Distributed Control System interface for condenser controls.

    Provides:
    - Control valve position feedback and setpoint writing
    - Cascade control coordination
    - Interlock status monitoring
    - Mode selection (manual/auto/cascade)
    - Automatic reconnection
    """

    def __init__(self, config: DCSConfig):
        """
        Initialize DCS integration.

        Args:
            config: DCS configuration
        """
        self.config = config
        self.tags = CondenserControlTags()

        self._client = None
        self._connected = False
        self._poll_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        # Current state
        self._valve_positions: Dict[str, ValvePosition] = {}
        self._controllers: Dict[str, CascadeController] = {}
        self._interlocks: Dict[str, InterlockStatus] = {}

        # Callbacks
        self._interlock_callbacks: List[Callable[[InterlockStatus], None]] = []
        self._alarm_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Statistics
        self._stats = {
            "reads_total": 0,
            "reads_success": 0,
            "reads_failed": 0,
            "writes_total": 0,
            "writes_success": 0,
            "writes_failed": 0,
            "reconnections": 0,
            "interlock_trips": 0,
            "last_poll": None
        }

        logger.info(f"DCS Integration initialized for {config.host}:{config.port}")

    @property
    def is_connected(self) -> bool:
        """Check if connected to DCS."""
        return self._connected

    async def connect(self) -> None:
        """
        Establish connection to DCS.

        Raises:
            DCSConnectionError: If connection fails
        """
        logger.info(f"Connecting to DCS at {self.config.host}:{self.config.port}")

        try:
            await self._create_connection()
            self._connected = True

            # Initialize controllers and interlocks
            await self._initialize_controllers()
            await self._initialize_interlocks()

            # Start polling
            self._poll_task = asyncio.create_task(self._poll_loop())

            logger.info("Successfully connected to DCS")

        except Exception as e:
            logger.error(f"Failed to connect to DCS: {e}")
            raise DCSConnectionError(f"Connection failed: {e}")

    async def _create_connection(self) -> None:
        """Create DCS connection."""
        # Simulated connection
        self._client = {
            "host": self.config.host,
            "port": self.config.port,
            "namespace": self.config.namespace,
            "connected": False
        }

        await asyncio.sleep(0.1)  # Simulate connection delay
        self._client["connected"] = True

    async def disconnect(self) -> None:
        """Disconnect from DCS."""
        logger.info("Disconnecting from DCS")

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        if self._client:
            self._client["connected"] = False
            self._client = None

        self._connected = False
        logger.info("Disconnected from DCS")

    async def _initialize_controllers(self) -> None:
        """Initialize cascade controllers."""
        # Hotwell level controller
        self._controllers["hotwell_level"] = CascadeController(
            controller_id="LC-001",
            name="Hotwell Level Controller",
            controller_type=ControllerType.CASCADE_PRIMARY,
            mode=ControlMode.CASCADE,
            pv=50.0, pv_units="%", pv_high_limit=100.0, pv_low_limit=0.0,
            sp=50.0, sp_high_limit=80.0, sp_low_limit=20.0, sp_source="local",
            op=50.0, op_high_limit=100.0, op_low_limit=0.0,
            kp=2.0, ki=120.0, kd=0.0,
            cascade_secondary_ids=["FC-001", "FC-002"]
        )

        # Vacuum controller
        self._controllers["vacuum"] = CascadeController(
            controller_id="PC-001",
            name="Vacuum Pressure Controller",
            controller_type=ControllerType.PID,
            mode=ControlMode.AUTO,
            pv=5.0, pv_units="kPa abs", pv_high_limit=20.0, pv_low_limit=0.0,
            sp=5.0, sp_high_limit=15.0, sp_low_limit=3.0, sp_source="remote",
            op=40.0, op_high_limit=100.0, op_low_limit=0.0,
            kp=1.5, ki=60.0, kd=10.0
        )

        # Cooling water temperature controller
        self._controllers["cw_temp"] = CascadeController(
            controller_id="TC-001",
            name="CW Outlet Temperature Controller",
            controller_type=ControllerType.CASCADE_PRIMARY,
            mode=ControlMode.AUTO,
            pv=35.0, pv_units="degC", pv_high_limit=50.0, pv_low_limit=20.0,
            sp=35.0, sp_high_limit=45.0, sp_low_limit=25.0, sp_source="cascade",
            op=60.0, op_high_limit=100.0, op_low_limit=0.0,
            kp=1.0, ki=180.0, kd=30.0
        )

    async def _initialize_interlocks(self) -> None:
        """Initialize interlock statuses."""
        interlocks = [
            InterlockStatus(
                interlock_id="IL-001",
                name="Low CW Flow",
                state=InterlockState.NORMAL,
                description="Low cooling water flow protection",
                trip_value=30000.0,  # m3/h
                reset_value=35000.0,
                associated_equipment=["CW Pumps", "Condenser"]
            ),
            InterlockStatus(
                interlock_id="IL-002",
                name="High Hotwell Level",
                state=InterlockState.NORMAL,
                description="High hotwell level protection",
                trip_value=90.0,  # %
                reset_value=80.0,
                associated_equipment=["Makeup Valve", "Condensate Pumps"]
            ),
            InterlockStatus(
                interlock_id="IL-003",
                name="Low Hotwell Level",
                state=InterlockState.NORMAL,
                description="Low hotwell level protection",
                trip_value=10.0,  # %
                reset_value=20.0,
                associated_equipment=["Condensate Pumps"]
            ),
            InterlockStatus(
                interlock_id="IL-004",
                name="High Vacuum",
                state=InterlockState.NORMAL,
                description="High vacuum (low pressure) protection",
                trip_value=2.0,  # kPa abs
                reset_value=3.0,
                associated_equipment=["Turbine", "Vacuum Breaker"]
            ),
            InterlockStatus(
                interlock_id="IL-005",
                name="Low Vacuum",
                state=InterlockState.NORMAL,
                description="Low vacuum (high pressure) protection",
                trip_value=15.0,  # kPa abs
                reset_value=12.0,
                associated_equipment=["Turbine", "Air Ejectors"]
            ),
            InterlockStatus(
                interlock_id="IL-006",
                name="Condenser Tube Leak",
                state=InterlockState.NORMAL,
                description="Condenser tube leak detection",
                trip_value=50.0,  # Conductivity uS/cm
                reset_value=30.0,
                associated_equipment=["Condenser", "Water Treatment"]
            )
        ]

        for interlock in interlocks:
            self._interlocks[interlock.interlock_id] = interlock

    async def _poll_loop(self) -> None:
        """Continuous polling loop."""
        while self._connected:
            try:
                await self._poll_controllers()
                await self._poll_valves()
                await self._poll_interlocks()
                self._stats["last_poll"] = datetime.utcnow()
            except Exception as e:
                logger.error(f"DCS polling error: {e}")

            await asyncio.sleep(self.config.polling_interval)

    async def _poll_controllers(self) -> None:
        """Poll controller statuses."""
        import random

        for controller in self._controllers.values():
            # Simulate reading controller values
            controller.pv += random.uniform(-0.5, 0.5)
            controller.error = controller.sp - controller.pv
            controller.op = max(0, min(100, controller.op + controller.error * 0.1))
            controller.timestamp = datetime.utcnow()

        self._stats["reads_success"] += 1

    async def _poll_valves(self) -> None:
        """Poll valve positions."""
        import random

        valve_defs = [
            (self.tags.CW_INLET_VALVE, "CW Inlet Valve", ValveAction.FAIL_OPEN),
            (self.tags.CW_OUTLET_VALVE, "CW Outlet Valve", ValveAction.FAIL_OPEN),
            (self.tags.MAKEUP_VALVE, "Makeup Valve", ValveAction.FAIL_CLOSED),
            (self.tags.AIR_EJECTOR_STEAM_VALVE, "Air Ejector Steam", ValveAction.FAIL_CLOSED),
        ]

        for valve_id, name, fail_action in valve_defs:
            position = random.uniform(45, 55)
            setpoint = 50.0

            self._valve_positions[valve_id] = ValvePosition(
                valve_id=valve_id,
                name=name,
                position_percent=position,
                setpoint_percent=setpoint,
                mode=ControlMode.AUTO,
                fail_action=fail_action,
                is_open=position > 90,
                is_closed=position < 10,
                in_transit=abs(position - setpoint) > 2,
                feedback_fault=False,
                motor_fault=False
            )

        self._stats["reads_success"] += 1

    async def _poll_interlocks(self) -> None:
        """Poll interlock statuses."""
        import random

        for interlock in self._interlocks.values():
            # Simulate checking interlock values
            if interlock.trip_value and interlock.reset_value:
                interlock.current_value = random.uniform(
                    interlock.reset_value * 0.8,
                    interlock.reset_value * 1.2
                )

                # Check for trip condition (simplified)
                if interlock.state == InterlockState.NORMAL:
                    if interlock.name == "High Hotwell Level":
                        if interlock.current_value > interlock.trip_value:
                            await self._trip_interlock(interlock)
                    elif interlock.name in ["Low CW Flow", "Low Hotwell Level"]:
                        if interlock.current_value < interlock.trip_value:
                            await self._trip_interlock(interlock)

        self._stats["reads_success"] += 1

    async def _trip_interlock(self, interlock: InterlockStatus) -> None:
        """Trip an interlock."""
        interlock.state = InterlockState.TRIPPED
        interlock.trip_timestamp = datetime.utcnow()
        self._stats["interlock_trips"] += 1

        logger.warning(
            f"Interlock tripped: {interlock.name} "
            f"(value={interlock.current_value}, trip={interlock.trip_value})"
        )

        # Notify callbacks
        for callback in self._interlock_callbacks:
            try:
                callback(interlock)
            except Exception as e:
                logger.error(f"Error in interlock callback: {e}")

    async def get_controller(self, controller_key: str) -> Optional[CascadeController]:
        """
        Get controller status.

        Args:
            controller_key: Controller identifier key

        Returns:
            CascadeController or None
        """
        return self._controllers.get(controller_key)

    async def get_all_controllers(self) -> Dict[str, CascadeController]:
        """Get all controller statuses."""
        return dict(self._controllers)

    async def set_controller_mode(
        self,
        controller_key: str,
        mode: ControlMode
    ) -> bool:
        """
        Set controller operating mode.

        Args:
            controller_key: Controller identifier
            mode: Target operating mode

        Returns:
            True if successful

        Raises:
            DCSModeError: If mode change not permitted
        """
        controller = self._controllers.get(controller_key)
        if not controller:
            raise DCSControlError(f"Controller not found: {controller_key}")

        # Check if mode change is safe
        if not await self._is_mode_change_safe(controller, mode):
            raise DCSModeError(
                f"Mode change from {controller.mode.value} to {mode.value} not permitted"
            )

        old_mode = controller.mode
        controller.mode = mode

        logger.info(
            f"Controller {controller.name} mode changed: "
            f"{old_mode.value} -> {mode.value}"
        )

        self._stats["writes_success"] += 1
        return True

    async def _is_mode_change_safe(
        self,
        controller: CascadeController,
        target_mode: ControlMode
    ) -> bool:
        """Check if mode change is safe."""
        # Don't allow cascade if secondary controllers not ready
        if target_mode == ControlMode.CASCADE:
            for sec_id in controller.cascade_secondary_ids:
                sec_controller = self._controllers.get(sec_id)
                if sec_controller and sec_controller.mode != ControlMode.AUTO:
                    return False

        # Check interlocks
        for interlock in self._interlocks.values():
            if not interlock.is_safe_to_operate():
                if any(
                    eq in interlock.associated_equipment
                    for eq in ["Condenser", "Turbine"]
                ):
                    return False

        return True

    async def set_controller_setpoint(
        self,
        controller_key: str,
        setpoint: float
    ) -> bool:
        """
        Set controller setpoint.

        Args:
            controller_key: Controller identifier
            setpoint: Target setpoint value

        Returns:
            True if successful

        Raises:
            DCSControlError: If setpoint change fails
        """
        controller = self._controllers.get(controller_key)
        if not controller:
            raise DCSControlError(f"Controller not found: {controller_key}")

        # Validate setpoint limits
        if not controller.sp_low_limit <= setpoint <= controller.sp_high_limit:
            raise DCSControlError(
                f"Setpoint {setpoint} out of range "
                f"[{controller.sp_low_limit}, {controller.sp_high_limit}]"
            )

        # Check mode allows remote setpoint
        if controller.mode not in [ControlMode.AUTO, ControlMode.CASCADE, ControlMode.REMOTE]:
            raise DCSControlError(
                f"Cannot change setpoint in {controller.mode.value} mode"
            )

        old_sp = controller.sp
        controller.sp = setpoint

        logger.info(
            f"Controller {controller.name} setpoint changed: "
            f"{old_sp} -> {setpoint} {controller.pv_units}"
        )

        self._stats["writes_success"] += 1
        return True

    async def get_valve_position(self, valve_id: str) -> Optional[ValvePosition]:
        """
        Get valve position.

        Args:
            valve_id: Valve identifier

        Returns:
            ValvePosition or None
        """
        return self._valve_positions.get(valve_id)

    async def get_all_valve_positions(self) -> Dict[str, ValvePosition]:
        """Get all valve positions."""
        return dict(self._valve_positions)

    async def set_valve_setpoint(
        self,
        valve_id: str,
        setpoint_percent: float
    ) -> bool:
        """
        Set valve position setpoint.

        Args:
            valve_id: Valve identifier
            setpoint_percent: Target position (0-100%)

        Returns:
            True if successful
        """
        valve = self._valve_positions.get(valve_id)
        if not valve:
            raise DCSControlError(f"Valve not found: {valve_id}")

        if valve.mode != ControlMode.MANUAL:
            raise DCSControlError(f"Valve {valve_id} not in manual mode")

        if not 0 <= setpoint_percent <= 100:
            raise DCSControlError("Setpoint must be between 0 and 100%")

        old_sp = valve.setpoint_percent
        valve.setpoint_percent = setpoint_percent

        logger.info(
            f"Valve {valve.name} setpoint changed: {old_sp}% -> {setpoint_percent}%"
        )

        self._stats["writes_success"] += 1
        return True

    async def get_interlock_status(
        self,
        interlock_id: str
    ) -> Optional[InterlockStatus]:
        """
        Get interlock status.

        Args:
            interlock_id: Interlock identifier

        Returns:
            InterlockStatus or None
        """
        return self._interlocks.get(interlock_id)

    async def get_all_interlocks(self) -> Dict[str, InterlockStatus]:
        """Get all interlock statuses."""
        return dict(self._interlocks)

    async def check_all_interlocks_safe(self) -> Tuple[bool, List[InterlockStatus]]:
        """
        Check if all interlocks are safe.

        Returns:
            Tuple of (all_safe, list of unsafe interlocks)
        """
        unsafe = [
            il for il in self._interlocks.values()
            if not il.is_safe_to_operate()
        ]

        return len(unsafe) == 0, unsafe

    async def reset_interlock(self, interlock_id: str) -> bool:
        """
        Reset a tripped interlock.

        Args:
            interlock_id: Interlock identifier

        Returns:
            True if reset successful
        """
        interlock = self._interlocks.get(interlock_id)
        if not interlock:
            raise DCSControlError(f"Interlock not found: {interlock_id}")

        if interlock.state != InterlockState.TRIPPED:
            logger.warning(f"Interlock {interlock_id} not in tripped state")
            return False

        # Check reset conditions
        if interlock.current_value and interlock.reset_value:
            # Check if value has returned to safe range
            if interlock.name in ["Low CW Flow", "Low Hotwell Level"]:
                if interlock.current_value < interlock.reset_value:
                    raise DCSControlError(
                        f"Cannot reset: value {interlock.current_value} "
                        f"below reset threshold {interlock.reset_value}"
                    )
            else:
                if interlock.current_value > interlock.reset_value:
                    raise DCSControlError(
                        f"Cannot reset: value {interlock.current_value} "
                        f"above reset threshold {interlock.reset_value}"
                    )

        interlock.state = InterlockState.NORMAL
        logger.info(f"Interlock {interlock.name} reset successfully")

        self._stats["writes_success"] += 1
        return True

    def register_interlock_callback(
        self,
        callback: Callable[[InterlockStatus], None]
    ) -> None:
        """Register callback for interlock trips."""
        self._interlock_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "controllers_count": len(self._controllers),
            "valves_count": len(self._valve_positions),
            "interlocks_count": len(self._interlocks),
            "tripped_interlocks": sum(
                1 for il in self._interlocks.values()
                if il.state == InterlockState.TRIPPED
            )
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

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
            health["reason"] = "Not connected to DCS"
            return health

        # Check interlocks
        all_safe, unsafe = await self.check_all_interlocks_safe()
        if not all_safe:
            health["status"] = "degraded"
            health["unsafe_interlocks"] = [il.interlock_id for il in unsafe]

        # Check data freshness
        if self._stats["last_poll"]:
            age = (datetime.utcnow() - self._stats["last_poll"]).total_seconds()
            if age > self.config.polling_interval * 5:
                health["status"] = "degraded"
                health["data_stale"] = True

        return health
