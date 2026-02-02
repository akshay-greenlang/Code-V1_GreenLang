"""
GL-001 ThermalCommand Orchestrator - Cascade Control Module

Master-slave PID cascade control hierarchy implementation for
process heat systems. Provides coordinated control across multiple
equipment units with proper gain scheduling and anti-windup.

Key Features:
    - Master-slave PID cascade architecture
    - Gain scheduling based on operating region
    - Anti-windup with back-calculation
    - Feedforward compensation
    - Bumpless transfer between modes
    - Auto/Manual/Cascade mode switching
    - Rate limiting and output clamping
    - Comprehensive audit trail

Reference Standards:
    - ISA-5.1 Instrumentation Symbols
    - ISA-88 Batch Control
    - IEC 61131-3 PLC Programming

Example:
    >>> from greenlang.agents.process_heat.gl_001_thermal_command.cascade_control import (
    ...     CascadeController, PIDController, PIDTuning
    ... )
    >>>
    >>> # Create master temperature controller
    >>> master = PIDController(
    ...     name="TC-101",
    ...     tuning=PIDTuning(kp=2.0, ki=0.1, kd=0.5),
    ...     output_min=0.0, output_max=100.0
    ... )
    >>>
    >>> # Create slave flow controller
    >>> slave = PIDController(
    ...     name="FC-101",
    ...     tuning=PIDTuning(kp=1.0, ki=0.5, kd=0.0),
    ...     output_min=0.0, output_max=100.0
    ... )
    >>>
    >>> # Create cascade
    >>> cascade = CascadeController(master=master, slave=slave)
    >>> output = cascade.calculate(temp_pv=450.0, temp_sp=500.0, flow_pv=75.0)

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import math
import uuid

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ControlMode(str, Enum):
    """Controller operating mode."""
    AUTO = "auto"           # Automatic control active
    MANUAL = "manual"       # Manual output control
    CASCADE = "cascade"     # Cascade from master controller
    REMOTE = "remote"       # Remote setpoint from DCS
    LOCAL = "local"         # Local setpoint
    TRACK = "track"         # Tracking external signal


class ControlAction(str, Enum):
    """Controller action direction."""
    DIRECT = "direct"       # Increasing PV requires increasing output
    REVERSE = "reverse"     # Increasing PV requires decreasing output


class AlarmPriority(str, Enum):
    """Alarm priority levels per ISA-18.2."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class GainScheduleRegion(str, Enum):
    """Operating regions for gain scheduling."""
    STARTUP = "startup"
    LOW_LOAD = "low_load"
    NORMAL = "normal"
    HIGH_LOAD = "high_load"
    SHUTDOWN = "shutdown"


# =============================================================================
# DATA MODELS
# =============================================================================

class PIDTuning(BaseModel):
    """
    PID controller tuning parameters.

    Standard form: output = Kp * (e + 1/Ti * integral(e) + Td * de/dt)
    """
    kp: float = Field(
        default=1.0,
        description="Proportional gain"
    )
    ki: float = Field(
        default=0.1,
        ge=0,
        description="Integral gain (1/Ti)"
    )
    kd: float = Field(
        default=0.0,
        ge=0,
        description="Derivative gain"
    )
    td_filter: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Derivative filter coefficient (0-1)"
    )
    anti_windup_gain: float = Field(
        default=1.0,
        ge=0,
        description="Anti-windup back-calculation gain"
    )

    @property
    def ti(self) -> float:
        """Get integral time Ti = 1/Ki."""
        return 1.0 / self.ki if self.ki > 0 else float('inf')


class GainScheduleEntry(BaseModel):
    """Gain schedule entry for a specific operating region."""
    region: GainScheduleRegion = Field(..., description="Operating region")
    pv_low: float = Field(..., description="Lower PV bound for this region")
    pv_high: float = Field(..., description="Upper PV bound for this region")
    tuning: PIDTuning = Field(..., description="Tuning for this region")
    transition_rate: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Rate of transition to these gains (0-1)"
    )


class ControllerState(BaseModel):
    """Internal state of a PID controller."""
    integral: float = Field(default=0.0, description="Integral accumulator")
    derivative_filter: float = Field(default=0.0, description="Filtered derivative")
    last_error: float = Field(default=0.0, description="Previous error for derivative")
    last_pv: float = Field(default=0.0, description="Previous PV for derivative on PV")
    last_output: float = Field(default=0.0, description="Previous output")
    last_calculation_time: Optional[datetime] = Field(
        default=None,
        description="Time of last calculation"
    )


class ControllerAlarm(BaseModel):
    """Controller alarm definition."""
    alarm_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Alarm identifier"
    )
    name: str = Field(..., description="Alarm name")
    priority: AlarmPriority = Field(..., description="Alarm priority")
    setpoint: float = Field(..., description="Alarm setpoint")
    deadband: float = Field(default=0.0, ge=0, description="Alarm deadband")
    direction: str = Field(default="high", description="high or low")
    delay_seconds: float = Field(default=0.0, ge=0, description="Time delay")
    active: bool = Field(default=False, description="Alarm currently active")
    acknowledged: bool = Field(default=False, description="Alarm acknowledged")
    activation_time: Optional[datetime] = Field(default=None, description="When activated")


class ControlOutput(BaseModel):
    """Output from a control calculation."""
    output_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Output calculation ID"
    )
    controller_name: str = Field(..., description="Controller name")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )
    setpoint: float = Field(..., description="Setpoint value")
    process_value: float = Field(..., description="Process value")
    error: float = Field(..., description="Error (SP - PV)")
    output: float = Field(..., description="Controller output (0-100%)")
    output_clamped: bool = Field(
        default=False,
        description="Was output clamped?"
    )
    mode: ControlMode = Field(..., description="Operating mode")
    p_term: float = Field(default=0.0, description="Proportional contribution")
    i_term: float = Field(default=0.0, description="Integral contribution")
    d_term: float = Field(default=0.0, description="Derivative contribution")
    feedforward: float = Field(default=0.0, description="Feedforward contribution")
    dt_seconds: float = Field(default=0.0, ge=0, description="Time step")
    gain_region: Optional[GainScheduleRegion] = Field(
        default=None,
        description="Active gain schedule region"
    )
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{self.controller_name}|{self.timestamp.isoformat()}|"
            f"{self.setpoint:.6f}|{self.process_value:.6f}|{self.output:.6f}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


class CascadeOutput(BaseModel):
    """Output from a cascade control calculation."""
    cascade_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Cascade calculation ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )
    master_output: ControlOutput = Field(..., description="Master controller output")
    slave_output: ControlOutput = Field(..., description="Slave controller output")
    final_output: float = Field(..., description="Final control output")
    cascade_active: bool = Field(
        default=True,
        description="Is cascade mode active?"
    )
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")


# =============================================================================
# PID CONTROLLER
# =============================================================================

class PIDController:
    """
    Industrial PID Controller with anti-windup and gain scheduling.

    Implements a full-featured PID controller suitable for process heat
    applications with:
    - Configurable P, PI, PD, or PID action
    - Derivative on PV (not error) to avoid setpoint kick
    - Anti-windup via back-calculation
    - Output rate limiting
    - Bumpless transfer between modes
    - Gain scheduling
    - Feedforward support

    Example:
        >>> tuning = PIDTuning(kp=2.0, ki=0.1, kd=0.5)
        >>> controller = PIDController(
        ...     name="TC-101",
        ...     tuning=tuning,
        ...     output_min=0.0,
        ...     output_max=100.0,
        ...     action=ControlAction.REVERSE
        ... )
        >>> output = controller.calculate(setpoint=500.0, pv=450.0)
        >>> print(f"Output: {output.output:.1f}%")
    """

    def __init__(
        self,
        name: str,
        tuning: PIDTuning,
        output_min: float = 0.0,
        output_max: float = 100.0,
        action: ControlAction = ControlAction.REVERSE,
        sample_time_seconds: float = 1.0,
        rate_limit_per_second: Optional[float] = None,
        derivative_on_pv: bool = True,
        enable_gain_scheduling: bool = False
    ) -> None:
        """
        Initialize PID Controller.

        Args:
            name: Controller tag name
            tuning: PID tuning parameters
            output_min: Minimum output value
            output_max: Maximum output value
            action: Controller action (direct/reverse)
            sample_time_seconds: Expected sample time
            rate_limit_per_second: Maximum output change rate
            derivative_on_pv: Use derivative on PV (recommended)
            enable_gain_scheduling: Enable gain scheduling
        """
        self.name = name
        self.tuning = tuning
        self.output_min = output_min
        self.output_max = output_max
        self.action = action
        self.sample_time = sample_time_seconds
        self.rate_limit = rate_limit_per_second
        self.derivative_on_pv = derivative_on_pv
        self.enable_gain_scheduling = enable_gain_scheduling

        # Internal state
        self._state = ControllerState()
        self._mode = ControlMode.AUTO
        self._manual_output = 50.0
        self._setpoint = 0.0
        self._gain_schedules: List[GainScheduleEntry] = []
        self._active_tuning = tuning
        self._alarms: List[ControllerAlarm] = []
        self._feedforward_value = 0.0
        self._history: List[ControlOutput] = []

        # Tracking for bumpless transfer
        self._tracking_value: Optional[float] = None

        logger.info(
            "PID Controller initialized: %s (Kp=%.2f, Ki=%.2f, Kd=%.2f)",
            name, tuning.kp, tuning.ki, tuning.kd
        )

    # =========================================================================
    # CORE CALCULATION
    # =========================================================================

    def calculate(
        self,
        setpoint: float,
        pv: float,
        dt_seconds: Optional[float] = None,
        feedforward: float = 0.0
    ) -> ControlOutput:
        """
        Calculate PID output.

        Args:
            setpoint: Desired setpoint
            pv: Current process value
            dt_seconds: Time step (None uses elapsed time)
            feedforward: Feedforward contribution

        Returns:
            ControlOutput with calculation details
        """
        now = datetime.now(timezone.utc)

        # Calculate dt
        if dt_seconds is None:
            if self._state.last_calculation_time:
                dt_seconds = (now - self._state.last_calculation_time).total_seconds()
            else:
                dt_seconds = self.sample_time

        dt_seconds = max(dt_seconds, 0.001)  # Prevent division by zero

        # Update setpoint
        self._setpoint = setpoint
        self._feedforward_value = feedforward

        # Check for mode
        if self._mode == ControlMode.MANUAL:
            return self._manual_calculate(setpoint, pv, dt_seconds, now)

        # Update gain scheduling if enabled
        if self.enable_gain_scheduling and self._gain_schedules:
            self._update_active_gains(pv)

        # Calculate error
        error = setpoint - pv
        if self.action == ControlAction.REVERSE:
            error = -error

        # Proportional term
        p_term = self._active_tuning.kp * error

        # Integral term with anti-windup
        self._state.integral += self._active_tuning.ki * error * dt_seconds
        i_term = self._state.integral

        # Derivative term (on PV for stability)
        if self.derivative_on_pv:
            d_pv = (pv - self._state.last_pv) / dt_seconds
            d_term = -self._active_tuning.kd * d_pv  # Negative because on PV
        else:
            d_error = (error - self._state.last_error) / dt_seconds
            d_term = self._active_tuning.kd * d_error

        # Apply derivative filter
        self._state.derivative_filter = (
            self._active_tuning.td_filter * d_term +
            (1 - self._active_tuning.td_filter) * self._state.derivative_filter
        )
        d_term = self._state.derivative_filter

        # Calculate raw output
        raw_output = p_term + i_term + d_term + feedforward

        # Clamp output
        output_clamped = False
        output = raw_output
        if output > self.output_max:
            output = self.output_max
            output_clamped = True
        elif output < self.output_min:
            output = self.output_min
            output_clamped = True

        # Anti-windup via back-calculation
        if output_clamped:
            windup_correction = (output - raw_output) * self._active_tuning.anti_windup_gain
            self._state.integral += windup_correction

        # Apply rate limiting
        if self.rate_limit is not None:
            max_change = self.rate_limit * dt_seconds
            change = output - self._state.last_output
            if abs(change) > max_change:
                output = self._state.last_output + math.copysign(max_change, change)

        # Update state
        self._state.last_error = error
        self._state.last_pv = pv
        self._state.last_output = output
        self._state.last_calculation_time = now

        # Determine gain region
        gain_region = None
        if self.enable_gain_scheduling and self._gain_schedules:
            for schedule in self._gain_schedules:
                if schedule.pv_low <= pv <= schedule.pv_high:
                    gain_region = schedule.region
                    break

        # Build result
        result = ControlOutput(
            controller_name=self.name,
            timestamp=now,
            setpoint=setpoint,
            process_value=pv,
            error=error if self.action == ControlAction.DIRECT else -error,
            output=output,
            output_clamped=output_clamped,
            mode=self._mode,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            feedforward=feedforward,
            dt_seconds=dt_seconds,
            gain_region=gain_region,
        )

        # Store history
        self._history.append(result)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        # Check alarms
        self._check_alarms(pv)

        return result

    def _manual_calculate(
        self,
        setpoint: float,
        pv: float,
        dt_seconds: float,
        now: datetime
    ) -> ControlOutput:
        """Calculate in manual mode."""
        # In manual mode, track the output for bumpless transfer
        error = setpoint - pv
        if self.action == ControlAction.REVERSE:
            error = -error

        # Back-calculate integral to match manual output
        p_term = self._active_tuning.kp * error
        d_term = 0.0  # No derivative in manual
        self._state.integral = self._manual_output - p_term

        # Update state
        self._state.last_error = error
        self._state.last_pv = pv
        self._state.last_output = self._manual_output
        self._state.last_calculation_time = now

        return ControlOutput(
            controller_name=self.name,
            timestamp=now,
            setpoint=setpoint,
            process_value=pv,
            error=error if self.action == ControlAction.DIRECT else -error,
            output=self._manual_output,
            output_clamped=False,
            mode=ControlMode.MANUAL,
            p_term=p_term,
            i_term=self._state.integral,
            d_term=0.0,
            feedforward=0.0,
            dt_seconds=dt_seconds,
        )

    # =========================================================================
    # MODE CONTROL
    # =========================================================================

    def set_mode(self, mode: ControlMode) -> None:
        """
        Set controller mode with bumpless transfer.

        Args:
            mode: New operating mode
        """
        if mode == self._mode:
            return

        old_mode = self._mode

        # Bumpless transfer preparation
        if mode == ControlMode.AUTO and old_mode == ControlMode.MANUAL:
            # Back-calculate integral for bumpless transfer
            pass  # Already handled in _manual_calculate

        elif mode == ControlMode.MANUAL and old_mode == ControlMode.AUTO:
            # Set manual output to current auto output
            self._manual_output = self._state.last_output

        elif mode == ControlMode.CASCADE:
            # Prepare for external setpoint
            pass

        self._mode = mode
        logger.info(
            "Controller %s mode changed: %s -> %s",
            self.name, old_mode.value, mode.value
        )

    def set_manual_output(self, output: float) -> None:
        """Set manual output value."""
        self._manual_output = max(self.output_min, min(self.output_max, output))

    @property
    def mode(self) -> ControlMode:
        """Get current mode."""
        return self._mode

    # =========================================================================
    # GAIN SCHEDULING
    # =========================================================================

    def add_gain_schedule(self, entry: GainScheduleEntry) -> None:
        """Add a gain schedule entry."""
        self._gain_schedules.append(entry)
        # Sort by pv_low for efficient lookup
        self._gain_schedules.sort(key=lambda x: x.pv_low)
        logger.info(
            "Gain schedule added for %s: %s (PV %.1f-%.1f)",
            self.name, entry.region.value, entry.pv_low, entry.pv_high
        )

    def _update_active_gains(self, pv: float) -> None:
        """Update active tuning based on current PV."""
        for schedule in self._gain_schedules:
            if schedule.pv_low <= pv <= schedule.pv_high:
                # Smooth transition to new gains
                rate = schedule.transition_rate
                self._active_tuning = PIDTuning(
                    kp=self._active_tuning.kp * (1 - rate) + schedule.tuning.kp * rate,
                    ki=self._active_tuning.ki * (1 - rate) + schedule.tuning.ki * rate,
                    kd=self._active_tuning.kd * (1 - rate) + schedule.tuning.kd * rate,
                    td_filter=schedule.tuning.td_filter,
                    anti_windup_gain=schedule.tuning.anti_windup_gain,
                )
                return

    # =========================================================================
    # ALARMS
    # =========================================================================

    def add_alarm(self, alarm: ControllerAlarm) -> None:
        """Add an alarm to the controller."""
        self._alarms.append(alarm)
        logger.info(
            "Alarm added to %s: %s (%s, setpoint=%.1f)",
            self.name, alarm.name, alarm.priority.value, alarm.setpoint
        )

    def _check_alarms(self, pv: float) -> None:
        """Check all alarms against current PV."""
        now = datetime.now(timezone.utc)

        for alarm in self._alarms:
            if alarm.direction == "high":
                in_alarm = pv > alarm.setpoint
                clear_point = alarm.setpoint - alarm.deadband
                is_clear = pv < clear_point
            else:
                in_alarm = pv < alarm.setpoint
                clear_point = alarm.setpoint + alarm.deadband
                is_clear = pv > clear_point

            if in_alarm and not alarm.active:
                # Alarm activating
                if alarm.delay_seconds > 0 and alarm.activation_time is None:
                    alarm.activation_time = now
                elif alarm.activation_time is None or (
                    now - alarm.activation_time
                ).total_seconds() >= alarm.delay_seconds:
                    alarm.active = True
                    alarm.acknowledged = False
                    logger.warning(
                        "ALARM ACTIVATED: %s.%s (PV=%.1f, SP=%.1f)",
                        self.name, alarm.name, pv, alarm.setpoint
                    )

            elif is_clear and alarm.active:
                alarm.active = False
                alarm.activation_time = None
                logger.info(
                    "Alarm cleared: %s.%s",
                    self.name, alarm.name
                )

    def get_active_alarms(self) -> List[ControllerAlarm]:
        """Get all active alarms."""
        return [a for a in self._alarms if a.active]

    # =========================================================================
    # STATE AND HISTORY
    # =========================================================================

    def reset(self) -> None:
        """Reset controller state."""
        self._state = ControllerState()
        self._active_tuning = self.tuning
        logger.info("Controller %s reset", self.name)

    def get_history(self, limit: int = 100) -> List[ControlOutput]:
        """Get calculation history."""
        return list(reversed(self._history[-limit:]))

    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "name": self.name,
            "mode": self._mode.value,
            "setpoint": self._setpoint,
            "last_output": self._state.last_output,
            "last_pv": self._state.last_pv,
            "integral": self._state.integral,
            "active_tuning": {
                "kp": self._active_tuning.kp,
                "ki": self._active_tuning.ki,
                "kd": self._active_tuning.kd,
            },
            "active_alarms": len(self.get_active_alarms()),
        }


# =============================================================================
# CASCADE CONTROLLER
# =============================================================================

class CascadeController:
    """
    Master-Slave Cascade Controller.

    Implements a cascade control structure where the master controller
    output becomes the setpoint for the slave controller. This provides
    improved rejection of disturbances affecting the inner loop.

    Typical use cases:
    - Temperature (master) -> Flow (slave) for fuel control
    - Pressure (master) -> Flow (slave) for steam control
    - Level (master) -> Flow (slave) for feedwater control

    Features:
    - Automatic cascade/manual mode switching
    - Slave controller ratio adjustment
    - Master output limiting for slave protection
    - Bumpless cascade-to-manual transfer
    - Comprehensive audit trail

    Example:
        >>> # Master: Temperature controller
        >>> master = PIDController(
        ...     name="TIC-101",
        ...     tuning=PIDTuning(kp=2.0, ki=0.1, kd=0.5),
        ...     output_min=0.0, output_max=100.0
        ... )
        >>>
        >>> # Slave: Flow controller
        >>> slave = PIDController(
        ...     name="FIC-101",
        ...     tuning=PIDTuning(kp=1.0, ki=0.5, kd=0.0),
        ...     output_min=0.0, output_max=100.0
        ... )
        >>>
        >>> cascade = CascadeController(master=master, slave=slave)
        >>> cascade.set_master_setpoint(500.0)
        >>> result = cascade.calculate(
        ...     master_pv=450.0,
        ...     slave_pv=75.0
        ... )
    """

    def __init__(
        self,
        master: PIDController,
        slave: PIDController,
        slave_sp_min: Optional[float] = None,
        slave_sp_max: Optional[float] = None,
        ratio: float = 1.0
    ) -> None:
        """
        Initialize Cascade Controller.

        Args:
            master: Master (outer loop) controller
            slave: Slave (inner loop) controller
            slave_sp_min: Minimum slave setpoint (clamp)
            slave_sp_max: Maximum slave setpoint (clamp)
            ratio: Ratio applied to master output for slave SP
        """
        self.master = master
        self.slave = slave
        self.slave_sp_min = slave_sp_min or slave.output_min
        self.slave_sp_max = slave_sp_max or slave.output_max
        self.ratio = ratio

        self._master_setpoint = 0.0
        self._cascade_active = True
        self._history: List[CascadeOutput] = []

        logger.info(
            "Cascade controller initialized: %s (master) -> %s (slave)",
            master.name, slave.name
        )

    # =========================================================================
    # CORE CALCULATION
    # =========================================================================

    def calculate(
        self,
        master_pv: float,
        slave_pv: float,
        master_feedforward: float = 0.0,
        slave_feedforward: float = 0.0,
        dt_seconds: Optional[float] = None
    ) -> CascadeOutput:
        """
        Calculate cascade control output.

        Args:
            master_pv: Master process value
            slave_pv: Slave process value
            master_feedforward: Feedforward to master
            slave_feedforward: Feedforward to slave
            dt_seconds: Time step

        Returns:
            CascadeOutput with both controller outputs
        """
        # Calculate master controller
        master_output = self.master.calculate(
            setpoint=self._master_setpoint,
            pv=master_pv,
            dt_seconds=dt_seconds,
            feedforward=master_feedforward
        )

        # Determine slave setpoint from master output
        if self._cascade_active and self.master.mode != ControlMode.MANUAL:
            slave_sp = master_output.output * self.ratio
            # Clamp slave setpoint
            slave_sp = max(self.slave_sp_min, min(self.slave_sp_max, slave_sp))
        else:
            # In manual or decoupled mode, slave uses its own setpoint
            slave_sp = self.slave._setpoint

        # Calculate slave controller
        slave_output = self.slave.calculate(
            setpoint=slave_sp,
            pv=slave_pv,
            dt_seconds=dt_seconds,
            feedforward=slave_feedforward
        )

        # Build cascade output
        result = CascadeOutput(
            timestamp=datetime.now(timezone.utc),
            master_output=master_output,
            slave_output=slave_output,
            final_output=slave_output.output,
            cascade_active=self._cascade_active,
        )

        # Calculate provenance
        provenance_str = (
            f"{self.master.name}|{self.slave.name}|"
            f"{result.timestamp.isoformat()}|"
            f"{result.final_output:.4f}"
        )
        result.provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        # Store history
        self._history.append(result)
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        return result

    # =========================================================================
    # SETPOINT AND MODE CONTROL
    # =========================================================================

    def set_master_setpoint(self, setpoint: float) -> None:
        """Set master controller setpoint."""
        self._master_setpoint = setpoint
        logger.debug(
            "Cascade %s master SP set to %.2f",
            self.master.name, setpoint
        )

    def set_cascade_active(self, active: bool) -> None:
        """Enable or disable cascade mode."""
        if active != self._cascade_active:
            self._cascade_active = active
            if active:
                # Entering cascade - bumpless transfer
                self.slave.set_mode(ControlMode.CASCADE)
            else:
                # Exiting cascade - slave to auto with current SP
                self.slave.set_mode(ControlMode.AUTO)

            logger.info(
                "Cascade %s -> %s: cascade mode %s",
                self.master.name, self.slave.name,
                "enabled" if active else "disabled"
            )

    def set_ratio(self, ratio: float) -> None:
        """Set cascade ratio."""
        self.ratio = ratio
        logger.info(
            "Cascade %s ratio set to %.2f",
            self.master.name, ratio
        )

    # =========================================================================
    # STATUS AND HISTORY
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get cascade controller status."""
        return {
            "master": self.master.get_status(),
            "slave": self.slave.get_status(),
            "cascade_active": self._cascade_active,
            "ratio": self.ratio,
            "master_setpoint": self._master_setpoint,
        }

    def get_history(self, limit: int = 100) -> List[CascadeOutput]:
        """Get calculation history."""
        return list(reversed(self._history[-limit:]))


# =============================================================================
# CASCADE COORDINATOR
# =============================================================================

class CascadeCoordinator:
    """
    Coordinator for multiple cascade control loops.

    Manages a system of cascade controllers with coordinated
    setpoint changes and mode transitions.

    Example:
        >>> coordinator = CascadeCoordinator()
        >>> coordinator.add_cascade("furnace_1", cascade_1)
        >>> coordinator.add_cascade("furnace_2", cascade_2)
        >>> coordinator.set_all_master_setpoints(500.0)
        >>> results = coordinator.calculate_all(pv_data)
    """

    def __init__(self, name: str = "CascadeCoordinator") -> None:
        """Initialize cascade coordinator."""
        self.name = name
        self._cascades: Dict[str, CascadeController] = {}
        self._calculation_history: List[Dict[str, CascadeOutput]] = []

        logger.info("CascadeCoordinator initialized: %s", name)

    def add_cascade(self, cascade_id: str, cascade: CascadeController) -> None:
        """Add a cascade controller."""
        self._cascades[cascade_id] = cascade
        logger.info(
            "Cascade added to coordinator: %s (%s -> %s)",
            cascade_id, cascade.master.name, cascade.slave.name
        )

    def remove_cascade(self, cascade_id: str) -> bool:
        """Remove a cascade controller."""
        if cascade_id in self._cascades:
            del self._cascades[cascade_id]
            return True
        return False

    def calculate_all(
        self,
        pv_data: Dict[str, Tuple[float, float]],
        dt_seconds: Optional[float] = None
    ) -> Dict[str, CascadeOutput]:
        """
        Calculate all cascades.

        Args:
            pv_data: Dict of cascade_id -> (master_pv, slave_pv)
            dt_seconds: Time step

        Returns:
            Dict of cascade_id -> CascadeOutput
        """
        results = {}

        for cascade_id, cascade in self._cascades.items():
            if cascade_id in pv_data:
                master_pv, slave_pv = pv_data[cascade_id]
                results[cascade_id] = cascade.calculate(
                    master_pv=master_pv,
                    slave_pv=slave_pv,
                    dt_seconds=dt_seconds
                )

        self._calculation_history.append(results)
        if len(self._calculation_history) > 1000:
            self._calculation_history = self._calculation_history[-500:]

        return results

    def set_all_master_setpoints(self, setpoint: float) -> None:
        """Set master setpoint for all cascades."""
        for cascade in self._cascades.values():
            cascade.set_master_setpoint(setpoint)

    def set_all_cascade_active(self, active: bool) -> None:
        """Enable/disable cascade mode for all."""
        for cascade in self._cascades.values():
            cascade.set_cascade_active(active)

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all cascades."""
        return {
            cascade_id: cascade.get_status()
            for cascade_id, cascade in self._cascades.items()
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_temperature_flow_cascade(
    tag_prefix: str,
    temp_sp_default: float = 500.0,
    flow_max: float = 100.0
) -> CascadeController:
    """
    Factory function to create temperature-to-flow cascade.

    Common application: Fuel firing rate control based on
    process temperature.

    Args:
        tag_prefix: Tag prefix for controllers (e.g., "FZ-101")
        temp_sp_default: Default temperature setpoint
        flow_max: Maximum flow setpoint

    Returns:
        Configured CascadeController
    """
    # Master: Temperature controller (slow, outer loop)
    master = PIDController(
        name=f"TIC-{tag_prefix}",
        tuning=PIDTuning(kp=2.0, ki=0.05, kd=1.0),
        output_min=0.0,
        output_max=100.0,
        action=ControlAction.REVERSE,
        sample_time_seconds=10.0,  # Slow loop
        rate_limit_per_second=5.0,  # 5%/sec max
    )

    # Slave: Flow controller (fast, inner loop)
    slave = PIDController(
        name=f"FIC-{tag_prefix}",
        tuning=PIDTuning(kp=1.0, ki=0.5, kd=0.0),
        output_min=0.0,
        output_max=100.0,
        action=ControlAction.REVERSE,
        sample_time_seconds=1.0,  # Fast loop
    )

    cascade = CascadeController(
        master=master,
        slave=slave,
        slave_sp_min=0.0,
        slave_sp_max=flow_max,
    )

    cascade.set_master_setpoint(temp_sp_default)

    return cascade


def create_pressure_flow_cascade(
    tag_prefix: str,
    pressure_sp_default: float = 150.0,
    flow_max: float = 100.0
) -> CascadeController:
    """
    Factory function to create pressure-to-flow cascade.

    Common application: Steam pressure control with feedforward.

    Args:
        tag_prefix: Tag prefix for controllers
        pressure_sp_default: Default pressure setpoint (psig)
        flow_max: Maximum flow setpoint

    Returns:
        Configured CascadeController
    """
    # Master: Pressure controller
    master = PIDController(
        name=f"PIC-{tag_prefix}",
        tuning=PIDTuning(kp=1.5, ki=0.1, kd=0.3),
        output_min=0.0,
        output_max=100.0,
        action=ControlAction.REVERSE,
        sample_time_seconds=5.0,
    )

    # Slave: Flow controller
    slave = PIDController(
        name=f"FIC-{tag_prefix}",
        tuning=PIDTuning(kp=1.0, ki=0.8, kd=0.0),
        output_min=0.0,
        output_max=100.0,
        action=ControlAction.REVERSE,
        sample_time_seconds=1.0,
    )

    cascade = CascadeController(
        master=master,
        slave=slave,
        slave_sp_min=0.0,
        slave_sp_max=flow_max,
    )

    cascade.set_master_setpoint(pressure_sp_default)

    return cascade
