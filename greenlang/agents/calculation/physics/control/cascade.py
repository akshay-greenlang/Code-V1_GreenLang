"""
Cascade Control Framework

Zero-Hallucination Cascade Control Implementation

This module implements cascade control structures for industrial
process control applications.

References:
    - Seborg, Edgar, Mellichamp: Process Dynamics and Control
    - ISA-5.1: Instrumentation Symbols and Identification
    - ISA-95: Enterprise-Control System Integration

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Callable
from enum import Enum
import hashlib
import math


class ControlMode(Enum):
    """Controller operating modes."""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"
    REMOTE = "remote"


class ControlAction(Enum):
    """Controller action direction."""
    DIRECT = "direct"  # Output increases when PV > SP
    REVERSE = "reverse"  # Output increases when PV < SP


@dataclass
class PIDParameters:
    """PID controller tuning parameters."""
    kp: float  # Proportional gain
    ki: float  # Integral gain (1/s)
    kd: float  # Derivative gain (s)
    output_min: float = 0.0
    output_max: float = 100.0
    deadband: float = 0.0
    setpoint_tracking: bool = True

    def to_dict(self) -> Dict:
        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "output_min": self.output_min,
            "output_max": self.output_max
        }


@dataclass
class ControllerState:
    """Current state of a controller."""
    setpoint: Decimal
    process_variable: Decimal
    output: Decimal
    error: Decimal
    integral: Decimal
    derivative: Decimal
    mode: ControlMode
    is_saturated: bool
    last_update_time: float


@dataclass
class CascadeLoopResult:
    """
    Cascade control loop calculation results.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Primary (master) loop
    primary_setpoint: Decimal
    primary_pv: Decimal
    primary_output: Decimal  # This becomes secondary setpoint
    primary_error: Decimal

    # Secondary (slave) loop
    secondary_setpoint: Decimal
    secondary_pv: Decimal
    secondary_output: Decimal  # Final control output
    secondary_error: Decimal

    # Loop status
    primary_mode: str
    secondary_mode: str
    cascade_active: bool

    # Performance metrics
    iae_primary: Decimal  # Integral of Absolute Error
    iae_secondary: Decimal

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        return {
            "primary_output": float(self.primary_output),
            "secondary_output": float(self.secondary_output),
            "cascade_active": self.cascade_active,
            "provenance_hash": self.provenance_hash
        }


class PIDController:
    """
    Discrete PID Controller Implementation.

    ZERO-HALLUCINATION GUARANTEE:
    - Deterministic calculations
    - Anti-windup protection
    - Bumpless transfer between modes
    - Complete state tracking
    """

    def __init__(
        self,
        name: str,
        parameters: PIDParameters,
        action: ControlAction = ControlAction.REVERSE,
        sample_time_s: float = 1.0
    ):
        self.name = name
        self.params = parameters
        self.action = action
        self.dt = Decimal(str(sample_time_s))

        # State variables
        self.integral = Decimal("0")
        self.prev_error = Decimal("0")
        self.prev_pv = Decimal("0")
        self.output = Decimal("50")  # Default 50%
        self.mode = ControlMode.AUTO
        self.manual_output = Decimal("50")

        # Anti-windup
        self.is_saturated = False

        # Performance tracking
        self.iae = Decimal("0")  # Integral of Absolute Error

    def _apply_precision(self, value: Decimal, precision: int = 4) -> Decimal:
        """Apply precision rounding."""
        quantize_str = "0." + "0" * precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def calculate(
        self,
        setpoint: float,
        process_variable: float,
        external_reset: Optional[float] = None
    ) -> Decimal:
        """
        Calculate PID output.

        ZERO-HALLUCINATION: Deterministic PID calculation.

        Args:
            setpoint: Desired setpoint value
            process_variable: Current process measurement
            external_reset: External integral reset (for cascade)

        Returns:
            Controller output (0-100%)
        """
        sp = Decimal(str(setpoint))
        pv = Decimal(str(process_variable))

        # If in manual mode, return manual output
        if self.mode == ControlMode.MANUAL:
            return self._apply_precision(self.manual_output)

        # Calculate error
        if self.action == ControlAction.REVERSE:
            error = sp - pv  # Increase output when PV < SP
        else:
            error = pv - sp  # Increase output when PV > SP

        # Deadband
        if abs(error) < Decimal(str(self.params.deadband)):
            error = Decimal("0")

        # PID gains as Decimal
        kp = Decimal(str(self.params.kp))
        ki = Decimal(str(self.params.ki))
        kd = Decimal(str(self.params.kd))

        # Proportional term
        p_term = kp * error

        # Integral term with anti-windup
        if not self.is_saturated:
            self.integral += ki * error * self.dt

        # External reset (for cascade slave)
        if external_reset is not None:
            self.integral = Decimal(str(external_reset))

        i_term = self.integral

        # Derivative term (on PV to avoid derivative kick)
        d_pv = (pv - self.prev_pv) / self.dt if self.dt > 0 else Decimal("0")
        d_term = -kd * d_pv

        # Calculate output
        output = p_term + i_term + d_term

        # Output limits
        out_min = Decimal(str(self.params.output_min))
        out_max = Decimal(str(self.params.output_max))

        if output > out_max:
            output = out_max
            self.is_saturated = True
        elif output < out_min:
            output = out_min
            self.is_saturated = True
        else:
            self.is_saturated = False

        # Update state
        self.prev_error = error
        self.prev_pv = pv
        self.output = output

        # Update IAE
        self.iae += abs(error) * self.dt

        return self._apply_precision(output)

    def set_mode(self, mode: ControlMode, bumpless: bool = True) -> None:
        """
        Set controller mode with optional bumpless transfer.

        Args:
            mode: New control mode
            bumpless: If True, perform bumpless transfer
        """
        if bumpless and mode == ControlMode.AUTO and self.mode == ControlMode.MANUAL:
            # Bumpless transfer: set integral to match current output
            self.integral = self.manual_output

        self.mode = mode

    def set_manual_output(self, output: float) -> None:
        """Set manual output value."""
        self.manual_output = Decimal(str(output))

    def get_state(self) -> ControllerState:
        """Get current controller state."""
        return ControllerState(
            setpoint=Decimal("0"),  # Set by caller
            process_variable=self.prev_pv,
            output=self.output,
            error=self.prev_error,
            integral=self.integral,
            derivative=Decimal("0"),
            mode=self.mode,
            is_saturated=self.is_saturated,
            last_update_time=0
        )


class CascadeController:
    """
    Cascade Control Implementation.

    ZERO-HALLUCINATION GUARANTEE:
    - Deterministic cascade control
    - Proper master-slave coordination
    - Anti-windup propagation
    - Mode handling

    Structure:
        Primary (Master) Controller -> Secondary (Slave) Controller -> Process

    The primary controller output becomes the setpoint for the secondary.
    """

    def __init__(
        self,
        name: str,
        primary_params: PIDParameters,
        secondary_params: PIDParameters,
        primary_action: ControlAction = ControlAction.REVERSE,
        secondary_action: ControlAction = ControlAction.REVERSE,
        sample_time_s: float = 1.0
    ):
        """
        Initialize cascade controller.

        Args:
            name: Controller name
            primary_params: Primary (master) loop parameters
            secondary_params: Secondary (slave) loop parameters
            primary_action: Primary controller action
            secondary_action: Secondary controller action
            sample_time_s: Sample time in seconds
        """
        self.name = name
        self.dt = sample_time_s

        # Create controllers
        self.primary = PIDController(
            name=f"{name}_Primary",
            parameters=primary_params,
            action=primary_action,
            sample_time_s=sample_time_s
        )

        self.secondary = PIDController(
            name=f"{name}_Secondary",
            parameters=secondary_params,
            action=secondary_action,
            sample_time_s=sample_time_s
        )

        # Cascade mode
        self.cascade_active = True

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "controller": self.name,
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def calculate(
        self,
        primary_setpoint: float,
        primary_pv: float,
        secondary_pv: float
    ) -> CascadeLoopResult:
        """
        Execute cascade control calculation.

        ZERO-HALLUCINATION: Deterministic cascade calculation.

        Args:
            primary_setpoint: Setpoint for primary (master) loop
            primary_pv: Process variable for primary loop
            secondary_pv: Process variable for secondary (slave) loop

        Returns:
            CascadeLoopResult with complete state
        """
        # Primary controller calculation
        primary_output = self.primary.calculate(primary_setpoint, primary_pv)

        # Secondary setpoint comes from primary output (scaled if needed)
        if self.cascade_active:
            secondary_sp = float(primary_output)
        else:
            # If cascade broken, secondary runs independently
            secondary_sp = float(self.secondary.output)

        # Secondary controller calculation
        secondary_output = self.secondary.calculate(secondary_sp, secondary_pv)

        # Calculate errors
        primary_error = Decimal(str(primary_setpoint)) - Decimal(str(primary_pv))
        secondary_error = Decimal(str(secondary_sp)) - Decimal(str(secondary_pv))

        # Anti-windup propagation
        # If secondary is saturated, prevent primary integral windup
        if self.secondary.is_saturated:
            self.primary.is_saturated = True

        # Create provenance
        inputs = {
            "primary_sp": str(primary_setpoint),
            "primary_pv": str(primary_pv),
            "secondary_pv": str(secondary_pv)
        }
        outputs = {
            "primary_out": str(primary_output),
            "secondary_out": str(secondary_output)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return CascadeLoopResult(
            primary_setpoint=Decimal(str(primary_setpoint)),
            primary_pv=Decimal(str(primary_pv)),
            primary_output=primary_output,
            primary_error=primary_error,
            secondary_setpoint=Decimal(str(secondary_sp)),
            secondary_pv=Decimal(str(secondary_pv)),
            secondary_output=secondary_output,
            secondary_error=secondary_error,
            primary_mode=self.primary.mode.value,
            secondary_mode=self.secondary.mode.value,
            cascade_active=self.cascade_active,
            iae_primary=self.primary.iae,
            iae_secondary=self.secondary.iae,
            provenance_hash=provenance_hash
        )

    def set_cascade_mode(self, active: bool, bumpless: bool = True) -> None:
        """
        Enable or disable cascade mode.

        Args:
            active: True to enable cascade
            bumpless: Perform bumpless transfer
        """
        if not active and self.cascade_active and bumpless:
            # Breaking cascade - hold secondary setpoint
            self.secondary.manual_output = self.secondary.output

        self.cascade_active = active

    def set_primary_mode(self, mode: ControlMode) -> None:
        """Set primary controller mode."""
        self.primary.set_mode(mode)

    def set_secondary_mode(self, mode: ControlMode) -> None:
        """Set secondary controller mode."""
        self.secondary.set_mode(mode)
        if mode == ControlMode.MANUAL:
            self.cascade_active = False


# Convenience functions
def create_temperature_cascade(
    primary_kp: float = 2.0,
    primary_ki: float = 0.01,
    secondary_kp: float = 1.0,
    secondary_ki: float = 0.1
) -> CascadeController:
    """
    Create cascade controller for temperature control.

    Typical application: Reactor temperature controlling jacket flow.

    Example:
        >>> cascade = create_temperature_cascade()
        >>> result = cascade.calculate(
        ...     primary_setpoint=150.0,  # Reactor temp SP
        ...     primary_pv=148.0,        # Reactor temp
        ...     secondary_pv=45.0        # Jacket flow
        ... )
        >>> print(f"Flow output: {result.secondary_output}%")
    """
    primary_params = PIDParameters(
        kp=primary_kp,
        ki=primary_ki,
        kd=0.0,
        output_min=0.0,
        output_max=100.0
    )

    secondary_params = PIDParameters(
        kp=secondary_kp,
        ki=secondary_ki,
        kd=0.0,
        output_min=0.0,
        output_max=100.0
    )

    return CascadeController(
        name="Temperature_Cascade",
        primary_params=primary_params,
        secondary_params=secondary_params
    )


def create_level_cascade(
    primary_kp: float = 1.0,
    primary_ki: float = 0.005,
    secondary_kp: float = 0.5,
    secondary_ki: float = 0.05
) -> CascadeController:
    """
    Create cascade controller for level control.

    Typical application: Drum level controlling feed flow.
    """
    primary_params = PIDParameters(
        kp=primary_kp,
        ki=primary_ki,
        kd=0.0,
        output_min=0.0,
        output_max=100.0
    )

    secondary_params = PIDParameters(
        kp=secondary_kp,
        ki=secondary_ki,
        kd=0.0,
        output_min=0.0,
        output_max=100.0
    )

    return CascadeController(
        name="Level_Cascade",
        primary_params=primary_params,
        secondary_params=secondary_params
    )
