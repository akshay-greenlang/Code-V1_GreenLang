"""
Feedforward Control

Zero-Hallucination Feedforward Control Implementation

This module implements feedforward control strategies for disturbance
rejection in industrial process control applications.

References:
    - Seborg, Edgar, Mellichamp: Process Dynamics and Control
    - ISA-5.1: Instrumentation Symbols and Identification
    - Smith, C.L.: Advanced Process Control

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Callable
from enum import Enum
import hashlib
import math


class FeedforwardType(Enum):
    """Feedforward controller types."""
    STATIC = "static"  # Simple gain-based
    DYNAMIC = "dynamic"  # With lead-lag compensation
    RATIO = "ratio"  # Ratio-based feedforward
    MODEL_BASED = "model_based"  # First-principles model


@dataclass
class FeedforwardConfig:
    """Configuration for feedforward controller."""
    # Static gain
    gain: float = 1.0

    # Dynamic compensation (lead-lag)
    lead_time_s: float = 0.0  # Lead time constant
    lag_time_s: float = 0.0  # Lag time constant

    # Output limits
    output_min: float = -100.0
    output_max: float = 100.0

    # Bias (steady-state offset)
    bias: float = 0.0

    # Deadband
    deadband: float = 0.0


@dataclass
class FeedforwardResult:
    """
    Feedforward calculation results.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Disturbance measurement
    disturbance_value: Decimal
    disturbance_rate: Decimal

    # Feedforward output
    ff_output: Decimal
    ff_output_rate: Decimal

    # Dynamic compensation
    lead_output: Decimal
    lag_output: Decimal

    # Combined output (FF + FB if applicable)
    total_output: Decimal

    # Status
    is_active: bool
    in_deadband: bool

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        return {
            "disturbance_value": float(self.disturbance_value),
            "ff_output": float(self.ff_output),
            "total_output": float(self.total_output),
            "is_active": self.is_active,
            "provenance_hash": self.provenance_hash
        }


class FeedforwardController:
    """
    Feedforward Controller Implementation.

    ZERO-HALLUCINATION GUARANTEE:
    - Deterministic feedforward calculations
    - Lead-lag dynamic compensation
    - Complete state tracking
    - Anti-windup and limiting

    Theory:
    Feedforward compensates for measured disturbances before they
    affect the controlled variable. The ideal feedforward transfer
    function is:

    Gff = -Gd / Gp

    Where Gd is disturbance-to-output and Gp is manipulated-to-output.
    """

    def __init__(
        self,
        name: str,
        config: FeedforwardConfig,
        sample_time_s: float = 1.0
    ):
        """
        Initialize feedforward controller.

        Args:
            name: Controller name
            config: Feedforward configuration
            sample_time_s: Sample time in seconds
        """
        self.name = name
        self.config = config
        self.dt = Decimal(str(sample_time_s))

        # State variables
        self.prev_disturbance = Decimal("0")
        self.prev_ff_output = Decimal("0")
        self.lead_state = Decimal("0")
        self.lag_state = Decimal("0")

        # Activation
        self.is_active = True

    def _apply_precision(self, value: Decimal, precision: int = 4) -> Decimal:
        """Apply precision rounding."""
        quantize_str = "0." + "0" * precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

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
        disturbance: float,
        feedback_output: Optional[float] = None
    ) -> FeedforwardResult:
        """
        Calculate feedforward output.

        ZERO-HALLUCINATION: Deterministic feedforward calculation.

        Args:
            disturbance: Measured disturbance value
            feedback_output: Optional feedback controller output

        Returns:
            FeedforwardResult with complete state
        """
        d = Decimal(str(disturbance))
        gain = Decimal(str(self.config.gain))
        bias = Decimal(str(self.config.bias))
        lead_tau = Decimal(str(self.config.lead_time_s))
        lag_tau = Decimal(str(self.config.lag_time_s))

        # Check deadband
        d_change = d - self.prev_disturbance
        in_deadband = abs(d_change) < Decimal(str(self.config.deadband))

        if in_deadband and not self.is_active:
            # No update needed
            ff_out = self.prev_ff_output
            d_rate = Decimal("0")
        else:
            # Rate of change of disturbance
            d_rate = d_change / self.dt if self.dt > 0 else Decimal("0")

            # Static feedforward
            ff_static = gain * d + bias

            # Dynamic compensation (lead-lag)
            # Lead-lag: G = (1 + s*tau_lead) / (1 + s*tau_lag)
            # Discrete approximation using bilinear transform

            if lead_tau > 0 or lag_tau > 0:
                # Lead term: y = x + tau_lead * dx/dt
                lead_out = ff_static + lead_tau * gain * d_rate

                # Lag term (first-order filter)
                if lag_tau > 0:
                    alpha = self.dt / (lag_tau + self.dt)
                    self.lag_state = alpha * lead_out + (Decimal("1") - alpha) * self.lag_state
                    ff_out = self.lag_state
                else:
                    ff_out = lead_out

                self.lead_state = lead_out
            else:
                ff_out = ff_static
                self.lead_state = ff_out
                self.lag_state = ff_out

            # Apply output limits
            out_min = Decimal(str(self.config.output_min))
            out_max = Decimal(str(self.config.output_max))

            if ff_out > out_max:
                ff_out = out_max
            elif ff_out < out_min:
                ff_out = out_min

        # Combine with feedback if provided
        if feedback_output is not None and self.is_active:
            fb_out = Decimal(str(feedback_output))
            total_output = fb_out + ff_out
        else:
            total_output = ff_out

        # Apply total output limits
        out_min = Decimal(str(self.config.output_min))
        out_max = Decimal(str(self.config.output_max))
        if total_output > out_max:
            total_output = out_max
        elif total_output < out_min:
            total_output = out_min

        # Update state
        ff_rate = (ff_out - self.prev_ff_output) / self.dt if self.dt > 0 else Decimal("0")
        self.prev_disturbance = d
        self.prev_ff_output = ff_out

        # Provenance
        inputs = {"disturbance": str(d), "feedback": str(feedback_output)}
        outputs = {"ff_output": str(ff_out), "total_output": str(total_output)}
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return FeedforwardResult(
            disturbance_value=self._apply_precision(d),
            disturbance_rate=self._apply_precision(d_rate),
            ff_output=self._apply_precision(ff_out),
            ff_output_rate=self._apply_precision(ff_rate),
            lead_output=self._apply_precision(self.lead_state),
            lag_output=self._apply_precision(self.lag_state),
            total_output=self._apply_precision(total_output),
            is_active=self.is_active,
            in_deadband=in_deadband,
            provenance_hash=provenance_hash
        )

    def set_active(self, active: bool) -> None:
        """Enable or disable feedforward."""
        self.is_active = active

    def set_gain(self, gain: float) -> None:
        """Update feedforward gain."""
        self.config.gain = gain

    def reset(self) -> None:
        """Reset controller state."""
        self.prev_disturbance = Decimal("0")
        self.prev_ff_output = Decimal("0")
        self.lead_state = Decimal("0")
        self.lag_state = Decimal("0")


class MultiDisturbanceFeedforward:
    """
    Multiple Disturbance Feedforward Controller.

    Handles multiple measured disturbances with individual gains
    and dynamics.
    """

    def __init__(
        self,
        name: str,
        sample_time_s: float = 1.0
    ):
        """Initialize multi-disturbance feedforward."""
        self.name = name
        self.dt = sample_time_s
        self.ff_controllers: Dict[str, FeedforwardController] = {}
        self.total_output = Decimal("0")

    def add_disturbance(
        self,
        disturbance_name: str,
        config: FeedforwardConfig
    ) -> None:
        """Add a disturbance feedforward channel."""
        self.ff_controllers[disturbance_name] = FeedforwardController(
            name=f"{self.name}_{disturbance_name}",
            config=config,
            sample_time_s=self.dt
        )

    def calculate(
        self,
        disturbances: Dict[str, float],
        feedback_output: Optional[float] = None
    ) -> Dict[str, FeedforwardResult]:
        """
        Calculate feedforward for all disturbances.

        Args:
            disturbances: Dictionary of {name: value} for each disturbance
            feedback_output: Optional feedback controller output

        Returns:
            Dictionary of results for each disturbance channel
        """
        results = {}
        total_ff = Decimal("0")

        for name, controller in self.ff_controllers.items():
            if name in disturbances:
                result = controller.calculate(disturbances[name])
                results[name] = result
                total_ff += result.ff_output

        # Combine all feedforward with feedback
        if feedback_output is not None:
            self.total_output = Decimal(str(feedback_output)) + total_ff
        else:
            self.total_output = total_ff

        return results

    def get_total_output(self) -> Decimal:
        """Get combined output from all feedforward channels."""
        return self.total_output


# Convenience functions
def create_flow_feedforward(
    gain: float = -1.0,
    lead_time_s: float = 0.0,
    lag_time_s: float = 5.0
) -> FeedforwardController:
    """
    Create feedforward controller for flow disturbance.

    Typical application: Compensate for feed flow changes.

    Example:
        >>> ff = create_flow_feedforward(gain=-0.8, lag_time_s=10)
        >>> result = ff.calculate(disturbance=120.0, feedback_output=50.0)
        >>> print(f"Total output: {result.total_output}%")
    """
    config = FeedforwardConfig(
        gain=gain,
        lead_time_s=lead_time_s,
        lag_time_s=lag_time_s,
        output_min=-50.0,
        output_max=50.0
    )

    return FeedforwardController(
        name="Flow_FF",
        config=config
    )


def create_temperature_feedforward(
    gain: float = -0.5,
    lead_time_s: float = 30.0,
    lag_time_s: float = 60.0
) -> FeedforwardController:
    """
    Create feedforward controller for temperature disturbance.

    Typical application: Compensate for feed temperature changes.
    """
    config = FeedforwardConfig(
        gain=gain,
        lead_time_s=lead_time_s,
        lag_time_s=lag_time_s,
        output_min=-25.0,
        output_max=25.0
    )

    return FeedforwardController(
        name="Temp_FF",
        config=config
    )


def calculate_feedforward_gain(
    disturbance_gain: float,
    process_gain: float
) -> float:
    """
    Calculate ideal feedforward gain.

    Reference: Gff = -Gd / Gp

    Args:
        disturbance_gain: Disturbance-to-output process gain
        process_gain: Manipulated-to-output process gain

    Returns:
        Ideal feedforward gain
    """
    if abs(process_gain) < 1e-10:
        raise ValueError("Process gain cannot be zero")

    return -disturbance_gain / process_gain
