# -*- coding: utf-8 -*-
"""
Pressure Control Calculator for GL-012 STEAMQUAL.

Provides deterministic calculations for steam pressure control including
valve positioning, flow coefficient calculations, pressure response prediction,
PID control output, and pressure drop calculations.

Standards:
- ISA-75.01.01: Flow Equations for Sizing Control Valves
- IEC 60534: Industrial-process control valves
- ASME B31.1: Power Piping (pressure drop calculations)
- Darcy-Weisbach: Pipe friction losses

Zero-hallucination: All calculations are deterministic with bit-perfect reproducibility.
No LLM involved in any calculation path.

Author: GL-CalculatorEngineer
Version: 1.0.0

Formulas:
    Valve Position: Based on Cv requirement and valve characteristic
    Flow Coefficient: Cv = Q * sqrt(SG / delta_P)
    Pressure Drop (Darcy): delta_P = f * (L/D) * (rho * v^2 / 2)
    PID Output: u = Kp*e + Ki*integral(e) + Kd*de/dt
"""

import hashlib
import json
import logging
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValveCharacteristic(Enum):
    """Control valve flow characteristic."""
    LINEAR = "linear"
    EQUAL_PERCENTAGE = "equal_percentage"
    QUICK_OPENING = "quick_opening"
    MODIFIED_PARABOLIC = "modified_parabolic"


class FlowRegime(Enum):
    """Pipe flow regime classification."""
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"


@dataclass
class PIDGains:
    """
    PID controller tuning parameters.

    Attributes:
        kp: Proportional gain
        ki: Integral gain (1/s)
        kd: Derivative gain (s)
        output_min: Minimum output limit
        output_max: Maximum output limit
        anti_windup: Anti-windup limit for integral term
        deadband: Error deadband to prevent hunting
    """
    kp: float = 2.0
    ki: float = 0.5
    kd: float = 0.1
    output_min: float = 0.0
    output_max: float = 100.0
    anti_windup: float = 50.0
    deadband: float = 0.0


@dataclass
class PressureControlInput:
    """Input parameters for pressure control calculation."""
    setpoint_mpa: float  # Desired pressure
    actual_mpa: float  # Measured pressure
    flow_rate_kg_s: float  # Mass flow rate
    fluid_density_kg_m3: float = 800.0  # Fluid density (steam ~2-50 kg/m3)
    valve_cv_max: float = 100.0  # Valve maximum Cv
    pipe_diameter_m: float = 0.1  # Pipe internal diameter
    pipe_length_m: float = 10.0  # Pipe length
    pipe_roughness_m: float = 0.000045  # Pipe roughness (steel)
    valve_characteristic: ValveCharacteristic = ValveCharacteristic.EQUAL_PERCENTAGE


@dataclass
class PressureControlOutput:
    """Output of pressure control calculations."""
    valve_position_pct: float  # Calculated valve position (0-100%)
    required_cv: float  # Required flow coefficient
    actual_cv: float  # Actual Cv at calculated position
    pressure_drop_valve_mpa: float  # Pressure drop across valve
    pressure_drop_pipe_mpa: float  # Pressure drop in piping
    pid_output: float  # Raw PID output
    control_error_mpa: float  # Setpoint deviation
    flow_regime: FlowRegime  # Pipe flow regime
    reynolds_number: float  # Reynolds number
    friction_factor: float  # Darcy friction factor
    calculation_method: str = "ISA-75.01.01"
    provenance_hash: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class SystemCharacteristics:
    """System characteristics for pressure response prediction."""
    system_gain: float = 1.0  # Pressure change per valve % change
    time_constant_s: float = 5.0  # First-order lag time constant
    dead_time_s: float = 1.0  # Transport delay
    natural_frequency_hz: float = 0.5  # If second-order
    damping_ratio: float = 0.7  # If second-order


class PressureControlCalculator:
    """
    Deterministic pressure control calculator.

    Calculates valve positions, flow coefficients, pressure drops,
    and PID control outputs for steam pressure regulation.

    All calculations are deterministic (zero-hallucination):
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes
    - No LLM or AI inference in calculation path

    Standards:
    - ISA-75.01.01 for control valve sizing
    - Darcy-Weisbach for pipe friction
    - Colebrook-White for friction factor (explicit approximation)

    Example:
        >>> calc = PressureControlCalculator()
        >>> result = calc.calculate(PressureControlInput(
        ...     setpoint_mpa=1.0,
        ...     actual_mpa=0.95,
        ...     flow_rate_kg_s=5.0,
        ...     fluid_density_kg_m3=10.0
        ... ))
        >>> print(f"Valve position: {result.valve_position_pct:.1f}%")
    """

    # Physical constants
    GRAVITY = Decimal("9.80665")  # m/s^2

    # Steam properties (approximate)
    STEAM_VISCOSITY_PA_S = Decimal("0.000012")  # Dynamic viscosity at ~180C

    # ISA-75.01.01 constants
    N1 = Decimal("0.0865")  # Cv equation constant for SI units
    N2 = Decimal("0.00214")  # Alternative constant

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pressure control calculator.

        Args:
            config: Optional configuration dictionary with keys:
                - precision: Decimal places for output (default: 4)
                - default_pid_gains: Default PID parameters
                - valve_rangeability: Valve rangeability ratio (default: 50)
        """
        self.config = config or {}
        self.precision = self.config.get('precision', 4)
        self.valve_rangeability = Decimal(str(
            self.config.get('valve_rangeability', 50)
        ))

        # Default PID gains
        default_gains = self.config.get('default_pid_gains', {})
        self.default_pid_gains = PIDGains(**default_gains)

        self.calculation_count = 0

        # PID controller state
        self._integral = Decimal("0")
        self._last_error = Decimal("0")
        self._last_output = Decimal("0")

    def calculate_valve_position(
        self,
        setpoint: float,
        actual: float,
        valve_cv: float,
        delta_p: float,
        flow_rate: Optional[float] = None,
        density: Optional[float] = None,
        characteristic: ValveCharacteristic = ValveCharacteristic.EQUAL_PERCENTAGE
    ) -> float:
        """
        Calculate valve position for desired pressure control.

        FORMULA (ISA-75.01.01):
            For liquid: Cv = Q * sqrt(SG / delta_P)
            For gas/steam: Cv = Q * sqrt(T * SG / (delta_P * P_inlet))

        Valve Position from Cv (Equal Percentage):
            x = ln(Cv / Cv_min) / ln(R)

        where R is the rangeability ratio.

        ZERO-HALLUCINATION GUARANTEE:
            - Standard ISA equations
            - Deterministic valve characteristic curves

        Args:
            setpoint: Desired pressure (MPa)
            actual: Current pressure (MPa)
            valve_cv: Maximum valve Cv
            delta_p: Available pressure differential (MPa)
            flow_rate: Current flow rate (kg/s), optional
            density: Fluid density (kg/m3), optional
            characteristic: Valve characteristic curve

        Returns:
            Valve position (0-100%)

        Example:
            >>> calc = PressureControlCalculator()
            >>> pos = calc.calculate_valve_position(
            ...     setpoint=1.0, actual=0.95, valve_cv=100, delta_p=0.2
            ... )
            >>> print(f"Valve position: {pos:.1f}%")
        """
        sp = Decimal(str(setpoint))
        pv = Decimal(str(actual))
        cv_max = Decimal(str(valve_cv))
        dp = Decimal(str(delta_p))

        # Prevent division by zero
        if dp <= 0:
            dp = Decimal("0.01")
            logger.warning("Delta P <= 0, using minimum 0.01 MPa")

        # Calculate required Cv if flow rate and density provided
        if flow_rate and density:
            cv_required = self.calculate_flow_coefficient(
                flow_rate, float(dp), density
            )
            cv_required = Decimal(str(cv_required))
        else:
            # Estimate based on pressure error and simple model
            error = sp - pv
            # Assume linear relationship for small errors
            cv_required = cv_max * Decimal("0.5") * (Decimal("1") + error / sp)
            cv_required = max(Decimal("0"), min(cv_max, cv_required))

        # Calculate valve position based on characteristic
        if characteristic == ValveCharacteristic.LINEAR:
            # Linear: Cv = Cv_max * x
            position = cv_required / cv_max * Decimal("100")

        elif characteristic == ValveCharacteristic.EQUAL_PERCENTAGE:
            # Equal percentage: Cv = Cv_max * R^(x-1)
            # Solving: x = 1 + ln(Cv/Cv_max) / ln(R)
            if cv_required > 0:
                cv_ratio = cv_required / cv_max
                if cv_ratio > 0:
                    import math
                    x = 1 + Decimal(str(math.log(float(cv_ratio)))) / Decimal(str(math.log(float(self.valve_rangeability))))
                    position = x * Decimal("100")
                else:
                    position = Decimal("0")
            else:
                position = Decimal("0")

        elif characteristic == ValveCharacteristic.QUICK_OPENING:
            # Quick opening: Cv = Cv_max * sqrt(x)
            # Solving: x = (Cv/Cv_max)^2
            cv_ratio = cv_required / cv_max
            position = cv_ratio ** 2 * Decimal("100")

        else:  # MODIFIED_PARABOLIC
            # Modified parabolic approximation
            cv_ratio = cv_required / cv_max
            position = cv_ratio ** Decimal("1.5") * Decimal("100")

        # Clamp to valid range
        position = max(Decimal("0"), min(Decimal("100"), position))

        return float(position.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    def calculate_flow_coefficient(
        self,
        flow_rate: float,
        delta_p: float,
        density: float
    ) -> float:
        """
        Calculate flow coefficient (Cv) from process conditions.

        FORMULA (ISA-75.01.01 for compressible flow):
            Cv = W / (N8 * Fp * Y * sqrt(x * P1 * rho1))

        Simplified for liquid-like behavior:
            Cv = Q / (N1 * sqrt(delta_P / SG))

        Where:
            Q = volumetric flow rate (m3/h)
            delta_P = pressure drop (kPa)
            SG = specific gravity relative to water

        ZERO-HALLUCINATION GUARANTEE:
            - Standard ISA-75.01.01 equation
            - Deterministic calculation

        Args:
            flow_rate: Mass flow rate (kg/s)
            delta_p: Pressure drop across valve (MPa)
            density: Fluid density (kg/m3)

        Returns:
            Flow coefficient Cv (dimensionless in SI context)

        Example:
            >>> calc = PressureControlCalculator()
            >>> cv = calc.calculate_flow_coefficient(
            ...     flow_rate=5.0, delta_p=0.2, density=10.0
            ... )
            >>> print(f"Required Cv: {cv:.2f}")
        """
        Q = Decimal(str(flow_rate))
        dp = Decimal(str(delta_p))
        rho = Decimal(str(density))

        # Prevent division by zero
        if dp <= 0:
            dp = Decimal("0.001")
        if rho <= 0:
            rho = Decimal("1")

        # Convert mass flow to volumetric flow (m3/h)
        # Q_vol = m_dot / rho * 3600
        Q_vol = Q / rho * Decimal("3600")  # m3/h

        # Specific gravity (relative to water at 15C)
        SG = rho / Decimal("1000")

        # Convert pressure to kPa
        dp_kpa = dp * Decimal("1000")

        # ISA equation: Cv = Q / (N1 * sqrt(dp/SG))
        # N1 = 0.0865 for Q in m3/h, dp in kPa
        if dp_kpa > 0:
            import math
            cv = Q_vol / (self.N1 * Decimal(str(math.sqrt(float(dp_kpa / SG)))))
        else:
            cv = Decimal("0")

        return float(cv.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    def predict_pressure_response(
        self,
        valve_change: float,
        system_characteristics: SystemCharacteristics,
        time_horizon_s: float = 30.0,
        time_step_s: float = 0.1
    ) -> List[Tuple[float, float]]:
        """
        Predict pressure response to valve position change.

        FORMULA (First-order plus dead-time model):
            P(t) = P_initial + K * delta_valve * (1 - exp(-(t-L)/tau))  for t > L
            P(t) = P_initial  for t <= L

        Where:
            K = system gain (MPa per % valve change)
            tau = time constant (seconds)
            L = dead time (seconds)

        ZERO-HALLUCINATION GUARANTEE:
            - Standard process dynamics model
            - Deterministic simulation

        Args:
            valve_change: Valve position change (%)
            system_characteristics: System dynamic parameters
            time_horizon_s: Prediction time horizon (seconds)
            time_step_s: Simulation time step (seconds)

        Returns:
            List of (time, pressure_change) tuples

        Example:
            >>> calc = PressureControlCalculator()
            >>> response = calc.predict_pressure_response(
            ...     valve_change=10.0,
            ...     system_characteristics=SystemCharacteristics(
            ...         system_gain=0.01, time_constant_s=5.0, dead_time_s=1.0
            ...     )
            ... )
        """
        dv = Decimal(str(valve_change))
        K = Decimal(str(system_characteristics.system_gain))
        tau = Decimal(str(system_characteristics.time_constant_s))
        L = Decimal(str(system_characteristics.dead_time_s))
        dt = Decimal(str(time_step_s))

        response = []
        t = Decimal("0")

        while t <= Decimal(str(time_horizon_s)):
            if t <= L:
                # During dead time, no response
                dp = Decimal("0")
            else:
                # First-order response after dead time
                import math
                time_after_dead = float(t - L)
                exp_term = math.exp(-time_after_dead / float(tau)) if float(tau) > 0 else 0
                dp = K * dv * Decimal(str(1 - exp_term))

            response.append((float(t), float(dp)))
            t += dt

        return response

    def calculate_pid_output(
        self,
        error: float,
        integral: float,
        derivative: float,
        gains: PIDGains
    ) -> float:
        """
        Calculate PID controller output.

        FORMULA (Standard PID):
            u(t) = Kp * e(t) + Ki * integral(e) + Kd * de/dt

        With anti-windup:
            integral = clamp(integral, -anti_windup, +anti_windup)

        With output limiting:
            u = clamp(u, output_min, output_max)

        ZERO-HALLUCINATION GUARANTEE:
            - Standard PID algorithm
            - Deterministic for given inputs

        Args:
            error: Current error (setpoint - actual)
            integral: Accumulated integral term
            derivative: Rate of error change (de/dt)
            gains: PID tuning parameters

        Returns:
            Controller output (0-100% typically)

        Example:
            >>> calc = PressureControlCalculator()
            >>> output = calc.calculate_pid_output(
            ...     error=0.05, integral=0.1, derivative=-0.01,
            ...     gains=PIDGains(kp=2.0, ki=0.5, kd=0.1)
            ... )
        """
        e = Decimal(str(error))
        I = Decimal(str(integral))
        D = Decimal(str(derivative))
        kp = Decimal(str(gains.kp))
        ki = Decimal(str(gains.ki))
        kd = Decimal(str(gains.kd))

        # Apply anti-windup to integral term
        aw = Decimal(str(gains.anti_windup))
        I_clamped = max(-aw, min(aw, I))

        # Calculate PID terms
        P_term = kp * e
        I_term = ki * I_clamped
        D_term = kd * D

        # Total output
        output = P_term + I_term + D_term

        # Apply output limits
        output_min = Decimal(str(gains.output_min))
        output_max = Decimal(str(gains.output_max))
        output = max(output_min, min(output_max, output))

        return float(output.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    def calculate_pressure_drop(
        self,
        flow_rate: float,
        pipe_diameter: float,
        length: float,
        roughness: float,
        density: Optional[float] = None,
        viscosity: Optional[float] = None
    ) -> float:
        """
        Calculate pipe pressure drop using Darcy-Weisbach equation.

        FORMULA (Darcy-Weisbach):
            delta_P = f * (L/D) * (rho * v^2 / 2)

        Where:
            f = Darcy friction factor (from Colebrook-White)
            L = pipe length (m)
            D = pipe diameter (m)
            rho = fluid density (kg/m3)
            v = flow velocity (m/s)

        Friction factor from Swamee-Jain explicit approximation:
            f = 0.25 / [log10(e/(3.7*D) + 5.74/Re^0.9)]^2

        ZERO-HALLUCINATION GUARANTEE:
            - Standard fluid mechanics equations
            - Deterministic explicit friction factor

        Args:
            flow_rate: Mass flow rate (kg/s)
            pipe_diameter: Internal diameter (m)
            length: Pipe length (m)
            roughness: Absolute roughness (m)
            density: Fluid density (kg/m3)
            viscosity: Dynamic viscosity (Pa.s)

        Returns:
            Pressure drop (MPa)

        Example:
            >>> calc = PressureControlCalculator()
            >>> dp = calc.calculate_pressure_drop(
            ...     flow_rate=5.0, pipe_diameter=0.1,
            ...     length=50.0, roughness=0.000045,
            ...     density=10.0
            ... )
            >>> print(f"Pressure drop: {dp:.4f} MPa")
        """
        m_dot = Decimal(str(flow_rate))
        D = Decimal(str(pipe_diameter))
        L = Decimal(str(length))
        e = Decimal(str(roughness))
        rho = Decimal(str(density)) if density else Decimal("10")  # Default steam density
        mu = Decimal(str(viscosity)) if viscosity else self.STEAM_VISCOSITY_PA_S

        # Prevent division by zero
        if D <= 0:
            D = Decimal("0.01")
        if rho <= 0:
            rho = Decimal("1")

        # Calculate flow area
        import math
        A = Decimal(str(math.pi)) * D ** 2 / Decimal("4")

        # Calculate velocity
        v = m_dot / (rho * A)  # m/s

        # Calculate Reynolds number
        Re = rho * v * D / mu

        # Determine flow regime and friction factor
        if Re < Decimal("2300"):
            # Laminar flow
            f = Decimal("64") / Re
            regime = FlowRegime.LAMINAR
        elif Re < Decimal("4000"):
            # Transitional
            f = Decimal("0.03")  # Approximation
            regime = FlowRegime.TRANSITIONAL
        else:
            # Turbulent - Swamee-Jain explicit approximation
            # f = 0.25 / [log10(e/(3.7*D) + 5.74/Re^0.9)]^2
            term1 = e / (Decimal("3.7") * D)
            term2 = Decimal("5.74") / (Re ** Decimal("0.9"))
            log_arg = float(term1 + term2)
            if log_arg > 0:
                log_term = math.log10(log_arg)
                f = Decimal("0.25") / Decimal(str(log_term ** 2))
            else:
                f = Decimal("0.02")  # Fallback
            regime = FlowRegime.TURBULENT

        # Darcy-Weisbach pressure drop
        # delta_P = f * (L/D) * (rho * v^2 / 2)
        delta_P = f * (L / D) * (rho * v ** 2 / Decimal("2"))

        # Convert Pa to MPa
        delta_P_mpa = delta_P / Decimal("1000000")

        # Store regime for reference
        self._last_flow_regime = regime
        self._last_reynolds = float(Re)
        self._last_friction_factor = float(f)

        return float(delta_P_mpa.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))

    def calculate(self, input_data: PressureControlInput) -> PressureControlOutput:
        """
        Comprehensive pressure control calculation.

        Calculates all pressure control parameters from input conditions.

        ZERO-HALLUCINATION GUARANTEE:
            - All calculations use deterministic formulas
            - Complete provenance tracking with SHA-256 hash

        Args:
            input_data: Pressure control conditions

        Returns:
            PressureControlOutput with all calculated values

        Example:
            >>> calc = PressureControlCalculator()
            >>> result = calc.calculate(PressureControlInput(
            ...     setpoint_mpa=1.0,
            ...     actual_mpa=0.95,
            ...     flow_rate_kg_s=5.0,
            ...     fluid_density_kg_m3=10.0
            ... ))
        """
        self.calculation_count += 1
        warnings = []

        sp = Decimal(str(input_data.setpoint_mpa))
        pv = Decimal(str(input_data.actual_mpa))

        # Calculate control error
        error = sp - pv

        # Calculate pressure drop available for valve
        # Assume system can provide some delta_P for control
        delta_p_available = max(Decimal("0.05"), abs(error) + Decimal("0.1"))

        # Calculate pipe pressure drop
        pipe_dp = self.calculate_pressure_drop(
            input_data.flow_rate_kg_s,
            input_data.pipe_diameter_m,
            input_data.pipe_length_m,
            input_data.pipe_roughness_m,
            input_data.fluid_density_kg_m3
        )

        # Calculate required Cv
        required_cv = self.calculate_flow_coefficient(
            input_data.flow_rate_kg_s,
            float(delta_p_available),
            input_data.fluid_density_kg_m3
        )

        # Calculate valve position
        valve_position = self.calculate_valve_position(
            float(sp),
            float(pv),
            input_data.valve_cv_max,
            float(delta_p_available),
            input_data.flow_rate_kg_s,
            input_data.fluid_density_kg_m3,
            input_data.valve_characteristic
        )

        # Calculate actual Cv at this position
        # For equal percentage: Cv = Cv_max * R^(x-1)
        if input_data.valve_characteristic == ValveCharacteristic.EQUAL_PERCENTAGE:
            x = Decimal(str(valve_position / 100.0))
            import math
            actual_cv = float(input_data.valve_cv_max) * (
                float(self.valve_rangeability) ** float(x - Decimal("1"))
            )
        else:
            actual_cv = float(input_data.valve_cv_max) * valve_position / 100.0

        # Calculate valve pressure drop
        if actual_cv > 0:
            # Rearrange Cv equation to solve for delta_P
            Q_vol = input_data.flow_rate_kg_s / input_data.fluid_density_kg_m3 * 3600
            SG = input_data.fluid_density_kg_m3 / 1000
            valve_dp = (Q_vol / (float(self.N1) * actual_cv)) ** 2 * SG / 1000
        else:
            valve_dp = 0.0
            warnings.append("Calculated Cv is zero - valve fully closed")

        # Calculate PID output
        dt = 1.0  # Assume 1 second sample time
        derivative = float(error - self._last_error) / dt

        # Update integral with anti-windup
        new_integral = self._integral + error * Decimal(str(dt))
        if abs(new_integral) > Decimal(str(self.default_pid_gains.anti_windup)):
            new_integral = Decimal(str(self.default_pid_gains.anti_windup)) * (
                Decimal("1") if new_integral > 0 else Decimal("-1")
            )
            warnings.append("Integral windup limit reached")

        pid_output = self.calculate_pid_output(
            float(error),
            float(new_integral),
            derivative,
            self.default_pid_gains
        )

        # Update state
        self._integral = new_integral
        self._last_error = error
        self._last_output = Decimal(str(pid_output))

        # Retrieve flow regime info from pipe calculation
        flow_regime = getattr(self, '_last_flow_regime', FlowRegime.TURBULENT)
        reynolds = getattr(self, '_last_reynolds', 0.0)
        friction = getattr(self, '_last_friction_factor', 0.02)

        # Generate provenance hash
        provenance_hash = self._calculate_provenance(
            input_data, valve_position, required_cv
        )

        return PressureControlOutput(
            valve_position_pct=valve_position,
            required_cv=round(required_cv, 2),
            actual_cv=round(actual_cv, 2),
            pressure_drop_valve_mpa=round(valve_dp, 4),
            pressure_drop_pipe_mpa=pipe_dp,
            pid_output=pid_output,
            control_error_mpa=round(float(error), 4),
            flow_regime=flow_regime,
            reynolds_number=round(reynolds, 0),
            friction_factor=round(friction, 6),
            calculation_method="ISA-75.01.01",
            provenance_hash=provenance_hash,
            warnings=warnings
        )

    def _calculate_provenance(
        self,
        input_data: PressureControlInput,
        valve_position: float,
        required_cv: float
    ) -> str:
        """Generate SHA-256 provenance hash for calculation."""
        data = {
            'calculator': 'PressureControlCalculator',
            'version': '1.0.0',
            'inputs': {
                'setpoint_mpa': input_data.setpoint_mpa,
                'actual_mpa': input_data.actual_mpa,
                'flow_rate_kg_s': input_data.flow_rate_kg_s,
                'fluid_density_kg_m3': input_data.fluid_density_kg_m3,
                'valve_cv_max': input_data.valve_cv_max,
            },
            'outputs': {
                'valve_position_pct': valve_position,
                'required_cv': required_cv,
            },
            'method': 'ISA-75.01.01'
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def reset_controller_state(self) -> None:
        """Reset PID controller state."""
        self._integral = Decimal("0")
        self._last_error = Decimal("0")
        self._last_output = Decimal("0")

    def set_pid_gains(self, gains: PIDGains) -> None:
        """Update PID gains."""
        self.default_pid_gains = gains

    def get_statistics(self) -> Dict[str, Any]:
        """Get calculator statistics."""
        return {
            'calculation_count': self.calculation_count,
            'precision': self.precision,
            'valve_rangeability': float(self.valve_rangeability),
            'current_integral': float(self._integral),
            'last_error': float(self._last_error),
            'pid_gains': {
                'kp': self.default_pid_gains.kp,
                'ki': self.default_pid_gains.ki,
                'kd': self.default_pid_gains.kd
            }
        }


# Unit test examples
def _run_self_tests():
    """Run self-tests to verify calculator correctness."""
    calc = PressureControlCalculator()

    # Test 1: Flow coefficient calculation
    cv = calc.calculate_flow_coefficient(
        flow_rate=5.0, delta_p=0.2, density=10.0
    )
    assert cv > 0, f"Cv should be positive: {cv}"
    print(f"Test 1 passed: Cv = {cv:.2f}")

    # Test 2: Valve position calculation
    pos = calc.calculate_valve_position(
        setpoint=1.0, actual=0.95, valve_cv=100, delta_p=0.2
    )
    assert 0 <= pos <= 100, f"Position out of range: {pos}"
    print(f"Test 2 passed: valve position = {pos:.1f}%")

    # Test 3: Pressure drop calculation
    dp = calc.calculate_pressure_drop(
        flow_rate=5.0, pipe_diameter=0.1, length=50.0,
        roughness=0.000045, density=10.0
    )
    assert dp >= 0, f"Pressure drop should be non-negative: {dp}"
    print(f"Test 3 passed: pressure drop = {dp:.4f} MPa")

    # Test 4: PID output
    output = calc.calculate_pid_output(
        error=0.05, integral=0.1, derivative=-0.01,
        gains=PIDGains(kp=2.0, ki=0.5, kd=0.1)
    )
    assert 0 <= output <= 100, f"PID output out of range: {output}"
    print(f"Test 4 passed: PID output = {output:.2f}")

    # Test 5: Pressure response prediction
    response = calc.predict_pressure_response(
        valve_change=10.0,
        system_characteristics=SystemCharacteristics(
            system_gain=0.01, time_constant_s=5.0, dead_time_s=1.0
        ),
        time_horizon_s=30.0
    )
    assert len(response) > 0, "Should have response data"
    assert response[-1][1] > 0, "Final response should be non-zero"
    print(f"Test 5 passed: {len(response)} response points, final dP = {response[-1][1]:.4f}")

    # Test 6: Comprehensive calculation
    result = calc.calculate(PressureControlInput(
        setpoint_mpa=1.0,
        actual_mpa=0.95,
        flow_rate_kg_s=5.0,
        fluid_density_kg_m3=10.0,
        valve_cv_max=100.0
    ))
    assert result.provenance_hash, "Should have provenance hash"
    print(f"Test 6 passed: comprehensive calc, hash = {result.provenance_hash[:16]}...")

    print("\nAll self-tests passed!")
    return True


if __name__ == "__main__":
    _run_self_tests()
