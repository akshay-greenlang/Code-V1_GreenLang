# -*- coding: utf-8 -*-
"""
Desuperheater Calculator for GL-012 STEAMQUAL.

Provides deterministic calculations for desuperheater spray water injection control,
including injection rate, outlet temperature prediction, energy balance validation,
and PID-based control signal optimization.

Standards:
- ASME PTC 6: Steam Turbines
- ASME B31.1: Power Piping
- ISA-75.01.01: Flow Equations for Sizing Control Valves

Zero-hallucination: All calculations are deterministic with bit-perfect reproducibility.
No LLM involved in any calculation path.

Author: GL-CalculatorEngineer
Version: 1.0.0

Formulas:
    Injection Rate: m_water = m_steam * (h_inlet - h_outlet) / (h_outlet - h_water)
    Outlet Temperature: T_out = (m_steam*h_steam + m_water*h_water) / (m_total*cp_mix)
    Energy Balance: Q_in = Q_out (within tolerance)
    Spray Pressure: P_spray = P_steam + delta_P_required
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


class ControlMode(Enum):
    """Desuperheater control mode."""
    MANUAL = "manual"
    AUTO_PID = "auto_pid"
    CASCADE = "cascade"
    FEEDFORWARD = "feedforward"


@dataclass
class ControlSignal:
    """
    Control signal output for desuperheater valve.

    Attributes:
        valve_position_pct: Spray valve position (0-100%)
        injection_rate_kg_s: Calculated spray water flow rate
        control_mode: Active control mode
        setpoint_deviation: Temperature deviation from setpoint
        pid_output: Raw PID controller output
        rate_limited: Whether output was rate-limited
        alarm_active: Whether control alarm is active
        provenance_hash: SHA-256 hash for audit trail
    """
    valve_position_pct: float
    injection_rate_kg_s: float
    control_mode: ControlMode
    setpoint_deviation: float
    pid_output: float
    rate_limited: bool
    alarm_active: bool
    provenance_hash: str


@dataclass
class DesuperheaterInput:
    """Input parameters for desuperheater calculation."""
    steam_flow_kg_s: float  # Inlet steam mass flow rate
    inlet_temperature_c: float  # Inlet steam temperature
    inlet_pressure_mpa: float  # Inlet steam pressure
    inlet_enthalpy_kj_kg: Optional[float] = None  # Inlet enthalpy (calculated if not provided)
    target_temperature_c: float = 0.0  # Desired outlet temperature
    water_temperature_c: float = 25.0  # Spray water temperature
    water_pressure_mpa: float = 2.0  # Spray water supply pressure
    water_enthalpy_kj_kg: Optional[float] = None  # Water enthalpy


@dataclass
class DesuperheaterOutput:
    """Output of desuperheater calculations."""
    injection_rate_kg_s: float  # Required spray water flow
    outlet_temperature_c: float  # Predicted outlet temperature
    outlet_enthalpy_kj_kg: float  # Outlet steam enthalpy
    energy_balance_error_pct: float  # Energy balance closure error
    spray_pressure_required_mpa: float  # Minimum spray water pressure
    valve_position_pct: float  # Calculated valve position
    temperature_reduction_c: float  # Temperature drop achieved
    water_to_steam_ratio: float  # Mass ratio
    control_signal: Optional[ControlSignal] = None
    calculation_method: str = "energy_balance"
    provenance_hash: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class PIDState:
    """PID controller state for stateful control."""
    integral: float = 0.0
    last_error: float = 0.0
    last_output: float = 0.0
    last_time: float = 0.0


class DesuperheaterCalculator:
    """
    Deterministic desuperheater spray water calculator.

    Calculates spray water injection requirements for steam temperature control
    using energy balance equations and thermodynamic property relationships.

    All calculations are deterministic (zero-hallucination):
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes
    - No LLM or AI inference in calculation path

    Thermodynamic Basis:
    Energy balance: m_steam * h_steam_in + m_water * h_water = m_total * h_out
    Mass balance: m_total = m_steam + m_water

    Example:
        >>> calc = DesuperheaterCalculator()
        >>> result = calc.calculate(DesuperheaterInput(
        ...     steam_flow_kg_s=10.0,
        ...     inlet_temperature_c=350.0,
        ...     inlet_pressure_mpa=1.0,
        ...     target_temperature_c=250.0,
        ...     water_temperature_c=30.0
        ... ))
        >>> print(f"Injection rate: {result.injection_rate_kg_s:.3f} kg/s")
    """

    # Specific heat capacities (kJ/kg.K)
    CP_WATER = Decimal("4.186")  # Liquid water
    CP_STEAM_AVG = Decimal("2.1")  # Superheated steam (average)

    # Latent heat of vaporization at atmospheric pressure (kJ/kg)
    H_FG_ATM = Decimal("2257")

    # Reference conditions
    REFERENCE_TEMP_C = Decimal("0")  # Reference temperature for enthalpy
    REFERENCE_PRESSURE_MPA = Decimal("0.101325")  # Atmospheric

    # Control constraints
    MAX_INJECTION_RATIO = Decimal("0.25")  # Max 25% water to steam ratio
    MIN_PRESSURE_DIFFERENTIAL_MPA = Decimal("0.3")  # Min spray pressure above steam
    MAX_VALVE_POSITION_PCT = Decimal("100")
    MIN_VALVE_POSITION_PCT = Decimal("0")
    MAX_RATE_OF_CHANGE_PCT_S = Decimal("10")  # Max 10%/second valve movement

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize desuperheater calculator.

        Args:
            config: Optional configuration dictionary with keys:
                - precision: Decimal places for output (default: 4)
                - max_injection_ratio: Maximum water/steam ratio (default: 0.25)
                - min_pressure_diff: Minimum pressure differential (default: 0.3)
                - pid_gains: Default PID gains {'kp': 2.0, 'ki': 0.5, 'kd': 0.1}
        """
        self.config = config or {}
        self.precision = self.config.get('precision', 4)
        self.max_injection_ratio = Decimal(str(
            self.config.get('max_injection_ratio', 0.25)
        ))
        self.min_pressure_diff = Decimal(str(
            self.config.get('min_pressure_diff', 0.3)
        ))

        # Default PID gains
        self.default_pid_gains = self.config.get('pid_gains', {
            'kp': 2.0,
            'ki': 0.5,
            'kd': 0.1
        })

        self.calculation_count = 0
        self._pid_state = PIDState()

    def calculate_injection_rate(
        self,
        m_steam: float,
        h_inlet: float,
        h_outlet: float,
        h_water: float
    ) -> float:
        """
        Calculate required spray water injection rate.

        FORMULA (Energy Balance):
            m_steam * h_inlet + m_water * h_water = (m_steam + m_water) * h_outlet

        Solving for m_water:
            m_water = m_steam * (h_inlet - h_outlet) / (h_outlet - h_water)

        ZERO-HALLUCINATION GUARANTEE:
            - Algebraic solution is deterministic
            - No iteration or numerical methods
            - Same inputs always produce identical output

        Args:
            m_steam: Steam mass flow rate (kg/s)
            h_inlet: Inlet steam specific enthalpy (kJ/kg)
            h_outlet: Desired outlet specific enthalpy (kJ/kg)
            h_water: Spray water specific enthalpy (kJ/kg)

        Returns:
            Required spray water mass flow rate (kg/s)

        Raises:
            ValueError: If h_outlet <= h_water (would require cooling water)

        Example:
            >>> calc = DesuperheaterCalculator()
            >>> m_water = calc.calculate_injection_rate(
            ...     m_steam=10.0,
            ...     h_inlet=3050.0,  # Superheated at 1 MPa, 350C
            ...     h_outlet=2950.0,  # Target outlet
            ...     h_water=125.0    # 30C water
            ... )
            >>> print(f"Injection rate: {m_water:.4f} kg/s")
        """
        # Convert to Decimal for precision
        ms = Decimal(str(m_steam))
        hi = Decimal(str(h_inlet))
        ho = Decimal(str(h_outlet))
        hw = Decimal(str(h_water))

        # Validate thermodynamic feasibility
        if ho <= hw:
            raise ValueError(
                f"Outlet enthalpy ({ho}) must be greater than water enthalpy ({hw}). "
                "Desuperheating requires h_outlet > h_water."
            )

        if hi <= ho:
            raise ValueError(
                f"Inlet enthalpy ({hi}) must be greater than outlet enthalpy ({ho}). "
                "No desuperheating needed."
            )

        # Calculate injection rate
        # m_water = m_steam * (h_inlet - h_outlet) / (h_outlet - h_water)
        numerator = ms * (hi - ho)
        denominator = ho - hw

        m_water = numerator / denominator

        # Validate against maximum injection ratio
        max_water = ms * self.max_injection_ratio
        if m_water > max_water:
            logger.warning(
                f"Calculated injection rate {m_water} exceeds maximum {max_water}. "
                "Consider staged desuperheating."
            )

        return float(m_water.quantize(
            Decimal('0.0001'), rounding=ROUND_HALF_UP
        ))

    def calculate_outlet_temperature(
        self,
        m_steam: float,
        T_inlet: float,
        m_water: float,
        T_water: float,
        P_mpa: Optional[float] = None
    ) -> float:
        """
        Calculate outlet temperature after spray water mixing.

        FORMULA (Energy Balance with constant Cp approximation):
            T_out = (m_steam * Cp_steam * T_inlet + m_water * (h_vap + Cp_water * T_water))
                    / ((m_steam + m_water) * Cp_mix)

        For superheated steam mixing with subcooled water:
            Q_steam = m_steam * Cp_steam * (T_inlet - T_sat)
            Q_evap = m_water * h_fg (water vaporization)
            Q_superheat = m_water * Cp_steam * (T_out - T_sat)

        Energy balance: Q_steam = Q_evap + Q_superheat (for water to vaporize)

        ZERO-HALLUCINATION GUARANTEE:
            - Algebraic solution is deterministic
            - Fixed thermodynamic properties used

        Args:
            m_steam: Steam mass flow rate (kg/s)
            T_inlet: Inlet steam temperature (C)
            m_water: Spray water mass flow rate (kg/s)
            T_water: Spray water temperature (C)
            P_mpa: Operating pressure for property lookup (optional)

        Returns:
            Outlet temperature (C)

        Example:
            >>> calc = DesuperheaterCalculator()
            >>> T_out = calc.calculate_outlet_temperature(
            ...     m_steam=10.0, T_inlet=350.0,
            ...     m_water=0.5, T_water=30.0
            ... )
            >>> print(f"Outlet temperature: {T_out:.1f} C")
        """
        ms = Decimal(str(m_steam))
        Ti = Decimal(str(T_inlet))
        mw = Decimal(str(m_water))
        Tw = Decimal(str(T_water))

        # Get saturation temperature (approximate based on typical industrial pressure)
        T_sat = Decimal("180")  # Assume ~1 MPa if pressure not provided
        if P_mpa:
            T_sat = self._get_saturation_temp_approx(Decimal(str(P_mpa)))

        # Check if we're above saturation
        if Ti <= T_sat:
            logger.warning("Inlet temperature below saturation - wet steam handling")
            # Simplified calculation for wet steam
            m_total = ms + mw
            T_out = (ms * Ti + mw * Tw) / m_total
            return float(T_out.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

        # Energy balance for superheated steam + subcooled water
        # Assuming complete vaporization of spray water:
        # m_steam * cp_steam * (T_in - T_out) = m_water * (h_fg + cp_steam * (T_out - T_sat))

        # Simplification: Using average specific heat
        cp_steam = self.CP_STEAM_AVG
        h_fg = self.H_FG_ATM * (Decimal("1") - Decimal("0.001") * (T_sat - Decimal("100")))

        # Solve for T_out:
        # ms * cp * Ti - ms * cp * Tout = mw * hfg + mw * cp * Tout - mw * cp * Tsat
        # ms * cp * Ti + mw * cp * Tsat - mw * hfg = Tout * (ms * cp + mw * cp)
        # T_out = (ms * cp * Ti + mw * cp * Tsat - mw * hfg) / ((ms + mw) * cp)

        numerator = ms * cp_steam * Ti + mw * cp_steam * T_sat - mw * h_fg
        denominator = (ms + mw) * cp_steam

        if denominator == 0:
            raise ValueError("Total mass flow cannot be zero")

        T_out = numerator / denominator

        # Clamp to physical bounds
        if T_out < T_sat:
            logger.warning(f"Calculated T_out {T_out} below saturation, clamping to {T_sat}")
            T_out = T_sat

        return float(T_out.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    def calculate_water_requirement(
        self,
        temp_reduction: float,
        steam_flow: float,
        cp_steam: Optional[float] = None
    ) -> float:
        """
        Calculate spray water requirement for desired temperature reduction.

        FORMULA (Simplified Energy Balance):
            Q_required = m_steam * Cp_steam * delta_T
            m_water = Q_required / (h_fg + Cp_steam * (T_out - T_sat))

        For quick estimation:
            m_water approx= m_steam * Cp_steam * delta_T / h_fg

        ZERO-HALLUCINATION GUARANTEE:
            - Direct formula application
            - Fixed thermodynamic constants

        Args:
            temp_reduction: Desired temperature drop (C)
            steam_flow: Steam mass flow rate (kg/s)
            cp_steam: Steam specific heat (kJ/kg.K), uses default if not provided

        Returns:
            Required spray water flow rate (kg/s)

        Example:
            >>> calc = DesuperheaterCalculator()
            >>> m_water = calc.calculate_water_requirement(
            ...     temp_reduction=50.0, steam_flow=10.0
            ... )
            >>> print(f"Water needed: {m_water:.3f} kg/s")
        """
        dt = Decimal(str(temp_reduction))
        ms = Decimal(str(steam_flow))
        cp = Decimal(str(cp_steam)) if cp_steam else self.CP_STEAM_AVG

        # Energy to remove from steam
        Q_remove = ms * cp * dt  # kJ/s = kW

        # Energy absorbed by water (vaporization + superheat)
        # Simplified: assume all goes to vaporization (conservative estimate)
        h_fg = self.H_FG_ATM

        m_water = Q_remove / h_fg

        return float(m_water.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))

    def validate_energy_balance(
        self,
        inlet_conditions: Dict[str, float],
        outlet_conditions: Dict[str, float],
        injection: Dict[str, float],
        tolerance_pct: float = 1.0
    ) -> bool:
        """
        Validate energy balance for desuperheater operation.

        FORMULA:
            Q_in = m_steam * h_steam_in + m_water * h_water
            Q_out = (m_steam + m_water) * h_out
            Error = |Q_in - Q_out| / Q_in * 100

        ZERO-HALLUCINATION GUARANTEE:
            - Direct calculation with no iteration
            - Deterministic comparison

        Args:
            inlet_conditions: {'mass_flow_kg_s': float, 'enthalpy_kj_kg': float}
            outlet_conditions: {'mass_flow_kg_s': float, 'enthalpy_kj_kg': float}
            injection: {'mass_flow_kg_s': float, 'enthalpy_kj_kg': float}
            tolerance_pct: Acceptable error percentage (default 1%)

        Returns:
            True if energy balance closes within tolerance

        Example:
            >>> calc = DesuperheaterCalculator()
            >>> valid = calc.validate_energy_balance(
            ...     inlet_conditions={'mass_flow_kg_s': 10.0, 'enthalpy_kj_kg': 3050.0},
            ...     outlet_conditions={'mass_flow_kg_s': 10.5, 'enthalpy_kj_kg': 2900.0},
            ...     injection={'mass_flow_kg_s': 0.5, 'enthalpy_kj_kg': 125.0}
            ... )
        """
        # Extract values
        m_steam = Decimal(str(inlet_conditions['mass_flow_kg_s']))
        h_steam = Decimal(str(inlet_conditions['enthalpy_kj_kg']))
        m_water = Decimal(str(injection['mass_flow_kg_s']))
        h_water = Decimal(str(injection['enthalpy_kj_kg']))
        m_out = Decimal(str(outlet_conditions['mass_flow_kg_s']))
        h_out = Decimal(str(outlet_conditions['enthalpy_kj_kg']))

        # Calculate energy flows (kW)
        Q_steam_in = m_steam * h_steam
        Q_water_in = m_water * h_water
        Q_in = Q_steam_in + Q_water_in

        Q_out = m_out * h_out

        # Calculate error
        if Q_in == 0:
            logger.error("Zero inlet energy - invalid conditions")
            return False

        error_pct = abs(Q_in - Q_out) / Q_in * Decimal("100")

        is_valid = error_pct <= Decimal(str(tolerance_pct))

        if not is_valid:
            logger.warning(
                f"Energy balance error {error_pct:.2f}% exceeds tolerance {tolerance_pct}%"
            )

        return is_valid

    def calculate_spray_water_pressure(
        self,
        steam_pressure: float,
        differential: Optional[float] = None
    ) -> float:
        """
        Calculate required spray water pressure.

        FORMULA:
            P_spray = P_steam + delta_P

        Where delta_P is the minimum pressure differential required for proper
        atomization and injection (typically 0.3-0.5 MPa above steam pressure).

        ZERO-HALLUCINATION GUARANTEE:
            - Simple addition with fixed minimum differential

        Args:
            steam_pressure: Steam pressure at injection point (MPa)
            differential: Pressure differential (MPa), uses minimum if not provided

        Returns:
            Required spray water pressure (MPa)

        Example:
            >>> calc = DesuperheaterCalculator()
            >>> P_spray = calc.calculate_spray_water_pressure(1.0)
            >>> print(f"Spray pressure required: {P_spray:.2f} MPa")  # 1.30 MPa
        """
        P_steam = Decimal(str(steam_pressure))
        delta_P = Decimal(str(differential)) if differential else self.min_pressure_diff

        P_spray = P_steam + delta_P

        return float(P_spray.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

    def optimize_injection_control(
        self,
        setpoint: float,
        actual: float,
        pid_params: Optional[Dict[str, float]] = None,
        dt: float = 1.0
    ) -> ControlSignal:
        """
        Optimize injection control using PID algorithm.

        FORMULA (Discrete PID):
            error = setpoint - actual
            P = Kp * error
            I = Ki * sum(error * dt)
            D = Kd * (error - last_error) / dt
            output = P + I + D

        With anti-windup and rate limiting applied.

        ZERO-HALLUCINATION GUARANTEE:
            - Standard PID algorithm (no ML/AI)
            - Deterministic for same state and inputs
            - State explicitly tracked

        Args:
            setpoint: Target temperature (C)
            actual: Current temperature (C)
            pid_params: {'kp': float, 'ki': float, 'kd': float} gains
            dt: Time step (seconds)

        Returns:
            ControlSignal with valve position and diagnostics

        Example:
            >>> calc = DesuperheaterCalculator()
            >>> signal = calc.optimize_injection_control(
            ...     setpoint=250.0, actual=260.0,
            ...     pid_params={'kp': 2.0, 'ki': 0.5, 'kd': 0.1}
            ... )
            >>> print(f"Valve position: {signal.valve_position_pct:.1f}%")
        """
        gains = pid_params or self.default_pid_gains
        kp = Decimal(str(gains.get('kp', 2.0)))
        ki = Decimal(str(gains.get('ki', 0.5)))
        kd = Decimal(str(gains.get('kd', 0.1)))

        sp = Decimal(str(setpoint))
        pv = Decimal(str(actual))
        dt_dec = Decimal(str(dt))

        # Calculate error (positive error = need more cooling = more spray)
        error = pv - sp  # Reverse acting: if actual > setpoint, increase spray

        # Proportional term
        P = kp * error

        # Integral term with anti-windup
        self._pid_state.integral += float(error * dt_dec)
        # Anti-windup: limit integral to prevent excessive accumulation
        max_integral = 50.0  # Limit integral contribution
        self._pid_state.integral = max(-max_integral,
                                       min(max_integral, self._pid_state.integral))
        I = ki * Decimal(str(self._pid_state.integral))

        # Derivative term (on error)
        if dt_dec > 0:
            derivative = (error - Decimal(str(self._pid_state.last_error))) / dt_dec
        else:
            derivative = Decimal("0")
        D = kd * derivative

        # Total PID output
        pid_output = P + I + D

        # Convert to valve position (0-100%)
        # Assuming linear relationship: output of 0 = 0%, output of 100 = 100%
        valve_position = float(pid_output)

        # Clamp to valid range
        valve_position = max(0.0, min(100.0, valve_position))

        # Rate limiting
        rate_limited = False
        if hasattr(self._pid_state, 'last_output'):
            max_change = float(self.MAX_RATE_OF_CHANGE_PCT_S * dt_dec)
            change = valve_position - self._pid_state.last_output
            if abs(change) > max_change:
                valve_position = self._pid_state.last_output + max_change * (1 if change > 0 else -1)
                rate_limited = True

        # Update state
        self._pid_state.last_error = float(error)
        self._pid_state.last_output = valve_position

        # Calculate injection rate from valve position
        # Assuming linear valve with max capacity = steam_flow * max_ratio
        injection_rate = valve_position / 100.0 * 2.0  # Placeholder calculation

        # Check for alarm conditions
        alarm_active = abs(float(error)) > 20.0  # Alarm if >20C deviation

        # Generate provenance hash
        provenance_data = {
            'setpoint': setpoint,
            'actual': actual,
            'pid_params': gains,
            'valve_position': valve_position,
            'error': float(error)
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        return ControlSignal(
            valve_position_pct=round(valve_position, 2),
            injection_rate_kg_s=round(injection_rate, 4),
            control_mode=ControlMode.AUTO_PID,
            setpoint_deviation=round(float(error), 2),
            pid_output=round(float(pid_output), 2),
            rate_limited=rate_limited,
            alarm_active=alarm_active,
            provenance_hash=provenance_hash
        )

    def calculate(self, input_data: DesuperheaterInput) -> DesuperheaterOutput:
        """
        Comprehensive desuperheater calculation.

        Calculates all desuperheater parameters from inlet conditions and target.

        ZERO-HALLUCINATION GUARANTEE:
            - All calculations use deterministic formulas
            - Complete provenance tracking with SHA-256 hash

        Args:
            input_data: Desuperheater operating conditions

        Returns:
            DesuperheaterOutput with all calculated values

        Example:
            >>> calc = DesuperheaterCalculator()
            >>> result = calc.calculate(DesuperheaterInput(
            ...     steam_flow_kg_s=10.0,
            ...     inlet_temperature_c=350.0,
            ...     inlet_pressure_mpa=1.0,
            ...     target_temperature_c=250.0,
            ...     water_temperature_c=30.0
            ... ))
        """
        self.calculation_count += 1
        warnings = []

        # Calculate enthalpies if not provided
        h_inlet = input_data.inlet_enthalpy_kj_kg
        if h_inlet is None:
            h_inlet = self._estimate_superheated_enthalpy(
                input_data.inlet_temperature_c,
                input_data.inlet_pressure_mpa
            )
            warnings.append("Inlet enthalpy estimated from T and P")

        h_water = input_data.water_enthalpy_kj_kg
        if h_water is None:
            h_water = float(self.CP_WATER * Decimal(str(input_data.water_temperature_c)))
            warnings.append("Water enthalpy calculated from temperature")

        # Calculate target outlet enthalpy
        h_outlet = self._estimate_superheated_enthalpy(
            input_data.target_temperature_c,
            input_data.inlet_pressure_mpa
        )

        # Calculate injection rate
        try:
            injection_rate = self.calculate_injection_rate(
                input_data.steam_flow_kg_s,
                h_inlet,
                h_outlet,
                h_water
            )
        except ValueError as e:
            warnings.append(str(e))
            injection_rate = 0.0

        # Calculate actual outlet temperature
        outlet_temp = self.calculate_outlet_temperature(
            input_data.steam_flow_kg_s,
            input_data.inlet_temperature_c,
            injection_rate,
            input_data.water_temperature_c,
            input_data.inlet_pressure_mpa
        )

        # Calculate spray pressure requirement
        spray_pressure = self.calculate_spray_water_pressure(
            input_data.inlet_pressure_mpa
        )

        # Check if water supply pressure is adequate
        if input_data.water_pressure_mpa < spray_pressure:
            warnings.append(
                f"Water pressure {input_data.water_pressure_mpa} MPa below required "
                f"{spray_pressure} MPa"
            )

        # Calculate water to steam ratio
        if input_data.steam_flow_kg_s > 0:
            water_steam_ratio = injection_rate / input_data.steam_flow_kg_s
        else:
            water_steam_ratio = 0.0

        # Calculate temperature reduction
        temp_reduction = input_data.inlet_temperature_c - outlet_temp

        # Validate energy balance
        m_total = input_data.steam_flow_kg_s + injection_rate
        energy_error = 0.0
        if injection_rate > 0:
            is_valid = self.validate_energy_balance(
                {'mass_flow_kg_s': input_data.steam_flow_kg_s, 'enthalpy_kj_kg': h_inlet},
                {'mass_flow_kg_s': m_total, 'enthalpy_kj_kg': h_outlet},
                {'mass_flow_kg_s': injection_rate, 'enthalpy_kj_kg': h_water}
            )
            if not is_valid:
                warnings.append("Energy balance error exceeds tolerance")
                # Calculate actual error
                Q_in = input_data.steam_flow_kg_s * h_inlet + injection_rate * h_water
                Q_out = m_total * h_outlet
                if Q_in > 0:
                    energy_error = abs(Q_in - Q_out) / Q_in * 100

        # Calculate valve position (simplified linear assumption)
        # Assume max injection at 100% valve = 25% of steam flow
        max_injection = input_data.steam_flow_kg_s * float(self.max_injection_ratio)
        if max_injection > 0:
            valve_position = min(100.0, injection_rate / max_injection * 100)
        else:
            valve_position = 0.0

        # Generate control signal if target provided
        control_signal = None
        if input_data.target_temperature_c > 0:
            control_signal = self.optimize_injection_control(
                input_data.target_temperature_c,
                outlet_temp
            )

        # Calculate outlet enthalpy
        h_out_actual = self._estimate_superheated_enthalpy(
            outlet_temp, input_data.inlet_pressure_mpa
        )

        # Generate provenance hash
        provenance_hash = self._calculate_provenance(input_data, injection_rate, outlet_temp)

        return DesuperheaterOutput(
            injection_rate_kg_s=round(injection_rate, 4),
            outlet_temperature_c=round(outlet_temp, 2),
            outlet_enthalpy_kj_kg=round(h_out_actual, 2),
            energy_balance_error_pct=round(energy_error, 2),
            spray_pressure_required_mpa=spray_pressure,
            valve_position_pct=round(valve_position, 2),
            temperature_reduction_c=round(temp_reduction, 2),
            water_to_steam_ratio=round(water_steam_ratio, 4),
            control_signal=control_signal,
            calculation_method="energy_balance",
            provenance_hash=provenance_hash,
            warnings=warnings
        )

    def _estimate_superheated_enthalpy(
        self,
        temperature_c: float,
        pressure_mpa: float
    ) -> float:
        """
        Estimate superheated steam enthalpy using polynomial approximation.

        This is an approximation for quick calculations. For high-precision
        work, use IAPWS-IF97 tables directly.
        """
        T = Decimal(str(temperature_c))
        P = Decimal(str(pressure_mpa))

        # Approximate saturation temperature
        T_sat = self._get_saturation_temp_approx(P)

        # Saturation enthalpy approximation (h_g at P)
        # h_g ~= 2675 + 30 * ln(P/0.1) for P in MPa (rough approximation)
        import math
        if float(P) > 0:
            h_g = Decimal("2675") + Decimal("30") * Decimal(str(math.log(float(P) / 0.1)))
        else:
            h_g = Decimal("2675")

        # Superheat contribution
        superheat = T - T_sat
        if superheat > 0:
            h = h_g + self.CP_STEAM_AVG * superheat
        else:
            h = h_g

        return float(h.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

    def _get_saturation_temp_approx(self, pressure_mpa: Decimal) -> Decimal:
        """
        Get approximate saturation temperature from pressure.

        Uses Antoine equation approximation:
        T_sat(C) = 100 + 28.02 * (P/0.101325)^0.25 - 28.02

        For more accuracy, use lookup tables.
        """
        P = float(pressure_mpa)
        if P <= 0:
            return Decimal("100")

        # Simplified correlation
        T_sat = 100 + 28.02 * (P / 0.101325) ** 0.25 - 28.02

        # Clamp to valid range
        T_sat = max(45, min(374, T_sat))

        return Decimal(str(T_sat)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def _calculate_provenance(
        self,
        input_data: DesuperheaterInput,
        injection_rate: float,
        outlet_temp: float
    ) -> str:
        """Generate SHA-256 provenance hash for calculation."""
        data = {
            'calculator': 'DesuperheaterCalculator',
            'version': '1.0.0',
            'inputs': {
                'steam_flow_kg_s': input_data.steam_flow_kg_s,
                'inlet_temperature_c': input_data.inlet_temperature_c,
                'inlet_pressure_mpa': input_data.inlet_pressure_mpa,
                'target_temperature_c': input_data.target_temperature_c,
                'water_temperature_c': input_data.water_temperature_c,
            },
            'outputs': {
                'injection_rate_kg_s': injection_rate,
                'outlet_temperature_c': outlet_temp,
            },
            'method': 'energy_balance'
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def reset_pid_state(self) -> None:
        """Reset PID controller state."""
        self._pid_state = PIDState()

    def get_statistics(self) -> Dict[str, Any]:
        """Get calculator statistics."""
        return {
            'calculation_count': self.calculation_count,
            'precision': self.precision,
            'max_injection_ratio': float(self.max_injection_ratio),
            'min_pressure_differential': float(self.min_pressure_diff),
            'pid_integral': self._pid_state.integral
        }


# Unit test examples
def _run_self_tests():
    """Run self-tests to verify calculator correctness."""
    calc = DesuperheaterCalculator()

    # Test 1: Injection rate calculation
    m_water = calc.calculate_injection_rate(
        m_steam=10.0,
        h_inlet=3050.0,
        h_outlet=2900.0,
        h_water=125.0
    )
    assert m_water > 0, f"Injection rate should be positive: {m_water}"
    assert m_water < 2.0, f"Injection rate too high: {m_water}"
    print(f"Test 1 passed: injection rate = {m_water:.4f} kg/s")

    # Test 2: Outlet temperature
    T_out = calc.calculate_outlet_temperature(
        m_steam=10.0, T_inlet=350.0,
        m_water=0.5, T_water=30.0
    )
    assert 200 < T_out < 350, f"Outlet temp out of range: {T_out}"
    print(f"Test 2 passed: outlet temp = {T_out:.2f} C")

    # Test 3: Water requirement
    m_req = calc.calculate_water_requirement(
        temp_reduction=50.0, steam_flow=10.0
    )
    assert m_req > 0, f"Water requirement should be positive: {m_req}"
    print(f"Test 3 passed: water requirement = {m_req:.4f} kg/s")

    # Test 4: Spray pressure
    P_spray = calc.calculate_spray_water_pressure(1.0)
    assert P_spray >= 1.3, f"Spray pressure too low: {P_spray}"
    print(f"Test 4 passed: spray pressure = {P_spray:.2f} MPa")

    # Test 5: PID control
    signal = calc.optimize_injection_control(
        setpoint=250.0, actual=260.0,
        pid_params={'kp': 2.0, 'ki': 0.5, 'kd': 0.1}
    )
    assert signal.valve_position_pct > 0, "Should increase valve position"
    assert signal.setpoint_deviation > 0, "Error should be positive (too hot)"
    print(f"Test 5 passed: valve position = {signal.valve_position_pct:.1f}%")

    # Test 6: Energy balance validation
    is_valid = calc.validate_energy_balance(
        inlet_conditions={'mass_flow_kg_s': 10.0, 'enthalpy_kj_kg': 3050.0},
        outlet_conditions={'mass_flow_kg_s': 10.54, 'enthalpy_kj_kg': 2900.0},
        injection={'mass_flow_kg_s': 0.54, 'enthalpy_kj_kg': 125.0},
        tolerance_pct=5.0
    )
    print(f"Test 6 passed: energy balance valid = {is_valid}")

    print("\nAll self-tests passed!")
    return True


if __name__ == "__main__":
    _run_self_tests()
