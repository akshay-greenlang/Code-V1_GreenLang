# -*- coding: utf-8 -*-
"""
Control Valve Connector for GL-012 STEAMQUAL (SteamQualityController).

Provides integration with control valve actuators for steam system control including:
- Globe valves
- Butterfly valves
- Ball valves

Supports multiple actuator types:
- Pneumatic actuators
- Electric actuators
- Hydraulic actuators

Features:
- Position control and feedback
- Emergency close functionality
- Valve diagnostics and health monitoring
- Safety interlocks
- Connection pooling and retry logic

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ConfigDict

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConnectionState,
    ConnectorType,
    ProtocolType,
    HealthStatus,
    HealthCheckResult,
    ConnectionError,
    ConfigurationError,
    CommunicationError,
    SafetyInterlockError,
    with_retry,
    CircuitState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class ValveType(str, Enum):
    """Types of control valves supported."""

    GLOBE = "globe"
    BUTTERFLY = "butterfly"
    BALL = "ball"
    GATE = "gate"
    CHECK = "check"
    DIAPHRAGM = "diaphragm"


class ActuatorType(str, Enum):
    """Types of valve actuators supported."""

    PNEUMATIC = "pneumatic"
    ELECTRIC = "electric"
    HYDRAULIC = "hydraulic"
    MANUAL = "manual"
    SOLENOID = "solenoid"


class ValveStatus(str, Enum):
    """Valve operational status."""

    OPEN = "open"
    CLOSED = "closed"
    TRAVELING = "traveling"  # Moving between positions
    FAULT = "fault"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"
    LOCKED_OUT = "locked_out"


class ActuatorStatus(str, Enum):
    """Actuator operational status."""

    OPERATIONAL = "operational"
    FAULT = "fault"
    LOW_SUPPLY = "low_supply"  # Low air/hydraulic pressure
    OVERLOAD = "overload"
    COMMUNICATION_LOST = "communication_lost"
    CALIBRATING = "calibrating"


class FailSafeAction(str, Enum):
    """Fail-safe action on loss of signal."""

    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"
    FAIL_IN_PLACE = "fail_in_place"


class InterlockType(str, Enum):
    """Types of safety interlocks."""

    HIGH_PRESSURE = "high_pressure"
    LOW_PRESSURE = "low_pressure"
    HIGH_TEMPERATURE = "high_temperature"
    LOW_TEMPERATURE = "low_temperature"
    HIGH_LEVEL = "high_level"
    LOW_LEVEL = "low_level"
    EMERGENCY_STOP = "emergency_stop"
    MANUAL_OVERRIDE = "manual_override"


# =============================================================================
# Data Models
# =============================================================================


class ControlValveConfig(BaseConnectorConfig):
    """Configuration for control valve connector."""

    model_config = ConfigDict(extra="forbid")

    connector_type: ConnectorType = Field(
        default=ConnectorType.CONTROL_VALVE,
        frozen=True
    )

    # Valve identification
    valve_id: str = Field(
        ...,
        description="Unique valve identifier"
    )
    valve_tag: str = Field(
        ...,
        description="Valve tag number (e.g., FV-101)"
    )
    valve_type: ValveType = Field(
        default=ValveType.GLOBE,
        description="Type of valve"
    )
    actuator_type: ActuatorType = Field(
        default=ActuatorType.PNEUMATIC,
        description="Type of actuator"
    )

    # Valve specifications
    size_inches: float = Field(
        default=4.0,
        ge=0.5,
        le=48.0,
        description="Valve size in inches"
    )
    cv_rating: float = Field(
        default=100.0,
        ge=0.1,
        description="Valve Cv (flow coefficient)"
    )
    pressure_rating_bar: float = Field(
        default=25.0,
        ge=1.0,
        description="Pressure rating in bar"
    )
    temperature_rating_c: float = Field(
        default=300.0,
        description="Temperature rating in Celsius"
    )

    # Safety settings
    fail_safe_action: FailSafeAction = Field(
        default=FailSafeAction.FAIL_CLOSED,
        description="Fail-safe action"
    )
    enable_safety_interlocks: bool = Field(
        default=True,
        description="Enable safety interlocks"
    )
    emergency_close_enabled: bool = Field(
        default=True,
        description="Enable emergency close function"
    )

    # Position limits
    position_min_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum position limit"
    )
    position_max_percent: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Maximum position limit"
    )

    # Stroke settings
    stroke_time_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Full stroke time in seconds"
    )
    deadband_percent: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Position deadband"
    )

    # Modbus settings
    slave_address: int = Field(
        default=1,
        ge=1,
        le=247,
        description="Modbus slave address"
    )

    # Interlock settings
    high_pressure_setpoint_bar: Optional[float] = Field(
        default=None,
        description="High pressure interlock setpoint"
    )
    low_pressure_setpoint_bar: Optional[float] = Field(
        default=None,
        description="Low pressure interlock setpoint"
    )
    high_temperature_setpoint_c: Optional[float] = Field(
        default=None,
        description="High temperature interlock setpoint"
    )


class ValveStatusReading(BaseModel):
    """Valve status information."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Status timestamp"
    )
    valve_id: str = Field(
        ...,
        description="Valve identifier"
    )
    valve_tag: str = Field(
        ...,
        description="Valve tag"
    )
    status: ValveStatus = Field(
        ...,
        description="Current valve status"
    )
    position_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current position (0-100%)"
    )
    setpoint_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Target setpoint (0-100%)"
    )
    deviation_percent: float = Field(
        default=0.0,
        description="Deviation from setpoint"
    )
    actuator_status: ActuatorStatus = Field(
        default=ActuatorStatus.OPERATIONAL,
        description="Actuator status"
    )
    is_open: bool = Field(
        ...,
        description="Whether valve is open"
    )
    is_closed: bool = Field(
        ...,
        description="Whether valve is closed"
    )
    is_traveling: bool = Field(
        ...,
        description="Whether valve is traveling"
    )
    interlocks_active: List[str] = Field(
        default_factory=list,
        description="Active interlocks"
    )
    alarms: List[str] = Field(
        default_factory=list,
        description="Active alarms"
    )


class ValveDiagnostics(BaseModel):
    """Valve diagnostic information."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Diagnostics timestamp"
    )
    valve_id: str = Field(
        ...,
        description="Valve identifier"
    )

    # Actuator diagnostics
    supply_pressure_bar: Optional[float] = Field(
        default=None,
        description="Actuator supply pressure (pneumatic/hydraulic)"
    )
    supply_pressure_min_bar: Optional[float] = Field(
        default=None,
        description="Minimum required supply pressure"
    )
    motor_current_amps: Optional[float] = Field(
        default=None,
        description="Motor current (electric actuators)"
    )
    motor_temperature_c: Optional[float] = Field(
        default=None,
        description="Motor temperature (electric actuators)"
    )

    # Position feedback
    position_feedback_ma: Optional[float] = Field(
        default=None,
        description="Position feedback signal (4-20mA)"
    )
    position_error_percent: Optional[float] = Field(
        default=None,
        description="Position error"
    )

    # Stroke metrics
    total_strokes: int = Field(
        default=0,
        ge=0,
        description="Total stroke count"
    )
    total_travel_percent: float = Field(
        default=0.0,
        ge=0.0,
        description="Total travel accumulated"
    )
    last_stroke_time_seconds: Optional[float] = Field(
        default=None,
        description="Last stroke time"
    )

    # Valve health
    seat_leakage_detected: bool = Field(
        default=False,
        description="Seat leakage detected"
    )
    packing_leakage_detected: bool = Field(
        default=False,
        description="Packing leakage detected"
    )
    stiction_detected: bool = Field(
        default=False,
        description="Valve stiction detected"
    )
    hysteresis_percent: Optional[float] = Field(
        default=None,
        description="Measured hysteresis"
    )

    # Maintenance
    last_maintenance_date: Optional[datetime] = Field(
        default=None,
        description="Last maintenance date"
    )
    next_maintenance_due: Optional[datetime] = Field(
        default=None,
        description="Next maintenance due date"
    )
    maintenance_required: bool = Field(
        default=False,
        description="Maintenance required flag"
    )

    # Overall health score
    health_score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Overall valve health score"
    )


class SafetyInterlockState(BaseModel):
    """Safety interlock state."""

    model_config = ConfigDict(frozen=True)

    interlock_type: InterlockType = Field(
        ...,
        description="Type of interlock"
    )
    active: bool = Field(
        default=False,
        description="Whether interlock is active"
    )
    setpoint: Optional[float] = Field(
        default=None,
        description="Interlock setpoint"
    )
    current_value: Optional[float] = Field(
        default=None,
        description="Current process value"
    )
    bypass_enabled: bool = Field(
        default=False,
        description="Whether bypass is enabled"
    )


# =============================================================================
# Control Valve Connector
# =============================================================================


class ControlValveConnector(BaseConnector):
    """
    Connector for control valve actuators.

    Provides valve position control, status monitoring, diagnostics,
    and safety interlock management.

    Features:
    - Position control with feedback
    - Multiple actuator type support
    - Safety interlocks
    - Emergency close functionality
    - Valve diagnostics and health monitoring
    - Connection pooling and retry logic
    """

    def __init__(self, config: ControlValveConfig) -> None:
        """
        Initialize control valve connector.

        Args:
            config: Valve configuration
        """
        super().__init__(config)
        self.valve_config: ControlValveConfig = config

        # Current state
        self._current_position: float = 0.0
        self._target_position: float = 0.0
        self._valve_status: ValveStatus = ValveStatus.UNKNOWN
        self._actuator_status: ActuatorStatus = ActuatorStatus.OPERATIONAL

        # Safety interlocks
        self._interlocks: Dict[InterlockType, SafetyInterlockState] = {}
        self._initialize_interlocks()

        # Position history
        self._position_history: deque = deque(maxlen=1000)

        # Diagnostics
        self._stroke_count: int = 0
        self._total_travel: float = 0.0

        # Connection state
        self._modbus_client: Optional[Any] = None

        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None

        self._logger.info(
            f"Initialized ControlValveConnector for valve {config.valve_tag} "
            f"(type={config.valve_type.value}, actuator={config.actuator_type.value})"
        )

    def _initialize_interlocks(self) -> None:
        """Initialize safety interlocks from configuration."""
        if self.valve_config.high_pressure_setpoint_bar:
            self._interlocks[InterlockType.HIGH_PRESSURE] = SafetyInterlockState(
                interlock_type=InterlockType.HIGH_PRESSURE,
                setpoint=self.valve_config.high_pressure_setpoint_bar,
            )

        if self.valve_config.low_pressure_setpoint_bar:
            self._interlocks[InterlockType.LOW_PRESSURE] = SafetyInterlockState(
                interlock_type=InterlockType.LOW_PRESSURE,
                setpoint=self.valve_config.low_pressure_setpoint_bar,
            )

        if self.valve_config.high_temperature_setpoint_c:
            self._interlocks[InterlockType.HIGH_TEMPERATURE] = SafetyInterlockState(
                interlock_type=InterlockType.HIGH_TEMPERATURE,
                setpoint=self.valve_config.high_temperature_setpoint_c,
            )

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self, valve_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Establish connection to the control valve.

        Args:
            valve_config: Optional additional configuration

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        if self._state == ConnectionState.CONNECTED:
            self._logger.warning(f"Already connected to valve {self.valve_config.valve_tag}")
            return True

        self._state = ConnectionState.CONNECTING

        try:
            # Apply any additional config
            if valve_config:
                for key, value in valve_config.items():
                    if hasattr(self.valve_config, key):
                        setattr(self.valve_config, key, value)

            # Connect based on protocol
            if self.valve_config.protocol == ProtocolType.MODBUS_TCP:
                await self._connect_modbus_tcp()
            elif self.valve_config.protocol == ProtocolType.OPC_UA:
                await self._connect_opc_ua()
            elif self.valve_config.protocol == ProtocolType.HART:
                await self._connect_hart()
            else:
                raise ConfigurationError(
                    f"Unsupported protocol: {self.valve_config.protocol}",
                    connector_id=self._config.connector_id,
                )

            # Read initial state
            await self._read_valve_state()

            self._state = ConnectionState.CONNECTED

            # Start monitoring task
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            self._logger.info(
                f"Connected to valve {self.valve_config.valve_tag} via "
                f"{self.valve_config.protocol.value}"
            )

            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
                response_summary=f"Connected to {self.valve_config.valve_tag}",
            )

            return True

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Failed to connect to valve {self.valve_config.valve_tag}: {e}")

            await self._audit_logger.log_operation(
                operation="connect",
                status="failure",
                error_message=str(e),
            )

            raise ConnectionError(
                f"Failed to connect to valve: {e}",
                connector_id=self._config.connector_id,
                details={"valve_tag": self.valve_config.valve_tag},
            )

    async def _connect_modbus_tcp(self) -> None:
        """Establish Modbus TCP connection."""
        # In production, would use pymodbus
        self._modbus_client = {
            "type": "modbus_tcp",
            "host": self.valve_config.host,
            "port": self.valve_config.port,
            "connected": True,
            "slave_address": self.valve_config.slave_address,
        }
        self._logger.debug(
            f"Modbus TCP connection established to {self.valve_config.host}:"
            f"{self.valve_config.port}"
        )

    async def _connect_opc_ua(self) -> None:
        """Establish OPC-UA connection."""
        # Placeholder for OPC-UA connection
        self._logger.debug("OPC-UA connection established")

    async def _connect_hart(self) -> None:
        """Establish HART connection."""
        # Placeholder for HART connection
        self._logger.debug("HART connection established")

    async def disconnect(self) -> None:
        """Disconnect from the control valve."""
        self._logger.info(f"Disconnecting from valve {self.valve_config.valve_tag}")

        # Cancel monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        # Close connection
        self._modbus_client = None
        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def validate_configuration(self) -> bool:
        """Validate valve configuration."""
        # Validate position limits
        if self.valve_config.position_min_percent >= self.valve_config.position_max_percent:
            raise ConfigurationError(
                "Position min must be less than max",
                connector_id=self._config.connector_id,
            )

        # Validate fail-safe configuration
        if (self.valve_config.fail_safe_action == FailSafeAction.FAIL_OPEN and
                self.valve_config.position_max_percent < 100.0):
            self._logger.warning(
                "Fail-open configured but max position limit is below 100%"
            )

        self._logger.debug("Configuration validated successfully")
        return True

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on the valve connection."""
        start_time = time.time()

        try:
            if self._state != ConnectionState.CONNECTED:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="Valve not connected",
                    latency_ms=0.0,
                )

            # Read current status
            status = await self.get_status()
            diagnostics = await self.read_diagnostics()
            latency_ms = (time.time() - start_time) * 1000

            # Determine health based on status and diagnostics
            if status.status == ValveStatus.FAULT:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    message="Valve in fault state",
                    details={
                        "valve_status": status.status.value,
                        "alarms": status.alarms,
                    },
                )
            elif diagnostics.maintenance_required or diagnostics.stiction_detected:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency_ms,
                    message="Valve requires attention",
                    details={
                        "health_score": diagnostics.health_score,
                        "maintenance_required": diagnostics.maintenance_required,
                        "stiction_detected": diagnostics.stiction_detected,
                    },
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    message="Valve operational",
                    details={
                        "position": status.position_percent,
                        "health_score": diagnostics.health_score,
                    },
                )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}",
            )

    # -------------------------------------------------------------------------
    # Monitoring Loop
    # -------------------------------------------------------------------------

    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop for valve state."""
        while self._state == ConnectionState.CONNECTED:
            try:
                await self._read_valve_state()
                await asyncio.sleep(0.5)  # 500ms monitoring interval

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1.0)

    async def _read_valve_state(self) -> None:
        """Read current valve state from device."""
        # In production, would read from actual device
        # Simulate realistic behavior

        import random

        # Update position (simulate gradual movement)
        if abs(self._current_position - self._target_position) > self.valve_config.deadband_percent:
            # Calculate movement per cycle
            movement_rate = 100.0 / self.valve_config.stroke_time_seconds * 0.5
            if self._current_position < self._target_position:
                self._current_position = min(
                    self._current_position + movement_rate,
                    self._target_position
                )
            else:
                self._current_position = max(
                    self._current_position - movement_rate,
                    self._target_position
                )

            self._valve_status = ValveStatus.TRAVELING
        else:
            # At setpoint
            if self._current_position <= 1.0:
                self._valve_status = ValveStatus.CLOSED
            elif self._current_position >= 99.0:
                self._valve_status = ValveStatus.OPEN
            else:
                # Intermediate position, no longer traveling
                if self._valve_status == ValveStatus.TRAVELING:
                    self._valve_status = ValveStatus.OPEN if self._current_position > 50 else ValveStatus.CLOSED

        # Add small noise to position feedback
        position_with_noise = self._current_position + random.uniform(-0.2, 0.2)
        position_with_noise = max(0.0, min(100.0, position_with_noise))

        # Record position
        self._position_history.append({
            "timestamp": datetime.utcnow(),
            "position": position_with_noise,
            "setpoint": self._target_position,
            "status": self._valve_status.value,
        })

    # -------------------------------------------------------------------------
    # Position Control
    # -------------------------------------------------------------------------

    @with_retry(max_retries=3, base_delay=0.5)
    async def set_position(self, position_percent: float) -> bool:
        """
        Set valve position.

        Args:
            position_percent: Target position (0-100%)

        Returns:
            True if command accepted

        Raises:
            SafetyInterlockError: If interlock prevents operation
            CommunicationError: If communication fails
        """
        if self._state != ConnectionState.CONNECTED:
            raise CommunicationError(
                "Not connected to valve",
                connector_id=self._config.connector_id,
            )

        # Validate position
        if position_percent < self.valve_config.position_min_percent:
            position_percent = self.valve_config.position_min_percent
            self._logger.warning(
                f"Position limited to minimum: {position_percent}%"
            )
        elif position_percent > self.valve_config.position_max_percent:
            position_percent = self.valve_config.position_max_percent
            self._logger.warning(
                f"Position limited to maximum: {position_percent}%"
            )

        # Check safety interlocks
        if self.valve_config.enable_safety_interlocks:
            await self._check_interlocks()

        # Track stroke
        if abs(position_percent - self._current_position) > self.valve_config.deadband_percent:
            self._stroke_count += 1
            self._total_travel += abs(position_percent - self._current_position)

        # Set target position
        previous_position = self._target_position
        self._target_position = position_percent

        self._logger.info(
            f"Valve {self.valve_config.valve_tag}: Position setpoint changed "
            f"from {previous_position:.1f}% to {position_percent:.1f}%"
        )

        # In production, would write to actual device
        # await self._write_position_setpoint(position_percent)

        await self._audit_logger.log_operation(
            operation="set_position",
            status="success",
            request_data={"position_percent": position_percent},
            response_summary=f"Position set to {position_percent:.1f}%",
        )

        return True

    async def get_position(self) -> float:
        """
        Get current valve position.

        Returns:
            Current position (0-100%)
        """
        if self._state != ConnectionState.CONNECTED:
            raise CommunicationError(
                "Not connected to valve",
                connector_id=self._config.connector_id,
            )

        return self._current_position

    async def get_status(self) -> ValveStatusReading:
        """
        Get current valve status.

        Returns:
            Complete valve status
        """
        if self._state != ConnectionState.CONNECTED:
            return ValveStatusReading(
                valve_id=self.valve_config.valve_id,
                valve_tag=self.valve_config.valve_tag,
                status=ValveStatus.UNKNOWN,
                position_percent=0.0,
                setpoint_percent=0.0,
                actuator_status=ActuatorStatus.COMMUNICATION_LOST,
                is_open=False,
                is_closed=False,
                is_traveling=False,
            )

        # Check for active interlocks
        active_interlocks = [
            il.interlock_type.value for il in self._interlocks.values()
            if il.active
        ]

        return ValveStatusReading(
            valve_id=self.valve_config.valve_id,
            valve_tag=self.valve_config.valve_tag,
            status=self._valve_status,
            position_percent=self._current_position,
            setpoint_percent=self._target_position,
            deviation_percent=abs(self._current_position - self._target_position),
            actuator_status=self._actuator_status,
            is_open=self._current_position >= 99.0,
            is_closed=self._current_position <= 1.0,
            is_traveling=self._valve_status == ValveStatus.TRAVELING,
            interlocks_active=active_interlocks,
            alarms=[],
        )

    # -------------------------------------------------------------------------
    # Emergency Close
    # -------------------------------------------------------------------------

    async def emergency_close(self) -> bool:
        """
        Emergency close the valve immediately.

        This bypasses normal position limits and immediately commands
        the valve to close.

        Returns:
            True if emergency close command accepted

        Raises:
            CommunicationError: If communication fails
        """
        if not self.valve_config.emergency_close_enabled:
            self._logger.warning(
                f"Emergency close not enabled for valve {self.valve_config.valve_tag}"
            )
            return False

        self._logger.warning(
            f"EMERGENCY CLOSE initiated for valve {self.valve_config.valve_tag}"
        )

        try:
            # In production, would send emergency close command to device
            # This typically activates a hardware interlock/shutdown

            # Set target to fully closed
            self._target_position = 0.0
            self._current_position = 0.0
            self._valve_status = ValveStatus.CLOSED

            await self._audit_logger.log_operation(
                operation="emergency_close",
                status="success",
                response_summary=f"Emergency close executed on {self.valve_config.valve_tag}",
            )

            return True

        except Exception as e:
            self._logger.error(f"Emergency close failed: {e}")

            await self._audit_logger.log_operation(
                operation="emergency_close",
                status="failure",
                error_message=str(e),
            )

            raise CommunicationError(
                f"Emergency close failed: {e}",
                connector_id=self._config.connector_id,
            )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    async def read_diagnostics(self) -> ValveDiagnostics:
        """
        Read valve diagnostic information.

        Returns:
            Complete valve diagnostics
        """
        if self._state != ConnectionState.CONNECTED:
            return ValveDiagnostics(
                valve_id=self.valve_config.valve_id,
                health_score=0.0,
            )

        # Calculate health score
        health_score = 100.0

        # Deduct for high stroke count
        if self._stroke_count > 100000:
            health_score -= 20
        elif self._stroke_count > 50000:
            health_score -= 10

        # Deduct for high total travel
        if self._total_travel > 1000000:  # % cumulative
            health_score -= 15
        elif self._total_travel > 500000:
            health_score -= 5

        import random

        # Simulate some diagnostic values
        supply_pressure = None
        motor_current = None

        if self.valve_config.actuator_type == ActuatorType.PNEUMATIC:
            supply_pressure = 6.0 + random.uniform(-0.3, 0.3)  # ~6 bar air supply
        elif self.valve_config.actuator_type == ActuatorType.ELECTRIC:
            motor_current = 2.5 + random.uniform(-0.2, 0.2)  # ~2.5A motor current

        return ValveDiagnostics(
            valve_id=self.valve_config.valve_id,
            supply_pressure_bar=supply_pressure,
            supply_pressure_min_bar=4.5 if supply_pressure else None,
            motor_current_amps=motor_current,
            motor_temperature_c=45.0 + random.uniform(-5, 10) if motor_current else None,
            position_feedback_ma=4.0 + (self._current_position / 100.0) * 16.0,
            position_error_percent=abs(self._current_position - self._target_position),
            total_strokes=self._stroke_count,
            total_travel_percent=self._total_travel,
            last_stroke_time_seconds=self.valve_config.stroke_time_seconds * 0.95,
            seat_leakage_detected=False,
            packing_leakage_detected=False,
            stiction_detected=random.random() < 0.05,  # 5% chance
            hysteresis_percent=0.5 + random.uniform(0, 0.5),
            maintenance_required=health_score < 70,
            health_score=max(0.0, health_score),
        )

    # -------------------------------------------------------------------------
    # Safety Interlocks
    # -------------------------------------------------------------------------

    async def _check_interlocks(self) -> None:
        """
        Check safety interlocks before allowing valve operation.

        Raises:
            SafetyInterlockError: If any interlock is active
        """
        active_interlocks = []

        for interlock_type, interlock in self._interlocks.items():
            if interlock.active and not interlock.bypass_enabled:
                active_interlocks.append(interlock_type.value)

        if active_interlocks:
            raise SafetyInterlockError(
                f"Safety interlocks active: {', '.join(active_interlocks)}",
                connector_id=self._config.connector_id,
                details={"active_interlocks": active_interlocks},
            )

    async def update_interlock(
        self,
        interlock_type: InterlockType,
        current_value: float,
    ) -> bool:
        """
        Update interlock state with current process value.

        Args:
            interlock_type: Type of interlock to update
            current_value: Current process value

        Returns:
            True if interlock is now active
        """
        if interlock_type not in self._interlocks:
            return False

        interlock = self._interlocks[interlock_type]
        was_active = interlock.active

        # Check if interlock should be active
        if interlock.setpoint is not None:
            if interlock_type in (InterlockType.HIGH_PRESSURE, InterlockType.HIGH_TEMPERATURE, InterlockType.HIGH_LEVEL):
                active = current_value > interlock.setpoint
            else:
                active = current_value < interlock.setpoint
        else:
            active = False

        # Update interlock state
        self._interlocks[interlock_type] = SafetyInterlockState(
            interlock_type=interlock_type,
            active=active,
            setpoint=interlock.setpoint,
            current_value=current_value,
            bypass_enabled=interlock.bypass_enabled,
        )

        # Log if state changed
        if active and not was_active:
            self._logger.warning(
                f"Interlock {interlock_type.value} ACTIVATED for valve "
                f"{self.valve_config.valve_tag} (value={current_value}, "
                f"setpoint={interlock.setpoint})"
            )

            await self._audit_logger.log_operation(
                operation="interlock_activated",
                status="warning",
                request_data={
                    "interlock_type": interlock_type.value,
                    "current_value": current_value,
                    "setpoint": interlock.setpoint,
                },
            )

        elif not active and was_active:
            self._logger.info(
                f"Interlock {interlock_type.value} CLEARED for valve "
                f"{self.valve_config.valve_tag}"
            )

        return active

    def get_interlock_states(self) -> Dict[str, SafetyInterlockState]:
        """Get all interlock states."""
        return {k.value: v for k, v in self._interlocks.items()}


# =============================================================================
# Factory Function
# =============================================================================


def create_control_valve_connector(
    valve_id: str,
    valve_tag: str,
    host: str,
    port: int = 502,
    valve_type: ValveType = ValveType.GLOBE,
    actuator_type: ActuatorType = ActuatorType.PNEUMATIC,
    **kwargs,
) -> ControlValveConnector:
    """
    Factory function to create a control valve connector.

    Args:
        valve_id: Unique valve identifier
        valve_tag: Valve tag number (e.g., FV-101)
        host: Valve controller host address
        port: Port number
        valve_type: Type of valve
        actuator_type: Type of actuator
        **kwargs: Additional configuration options

    Returns:
        Configured ControlValveConnector instance
    """
    config = ControlValveConfig(
        connector_name=f"ControlValve_{valve_tag}",
        valve_id=valve_id,
        valve_tag=valve_tag,
        host=host,
        port=port,
        valve_type=valve_type,
        actuator_type=actuator_type,
        **kwargs,
    )

    return ControlValveConnector(config)
