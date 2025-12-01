# -*- coding: utf-8 -*-
"""
Unit Tests for DesuperheaterCalculator.

This module provides comprehensive tests for the DesuperheaterCalculator class,
covering injection rate calculations, outlet temperature predictions, energy
balance validation, spray water pressure calculations, and PID control response.

Coverage Target: 95%+
Standards Compliance:
- ASME PTC 12.4: Steam-to-Steam Jet Ejectors
- ASME PTC 19.11: Steam and Water Sampling
- ISA-75.01: Control Valve Sizing

Test Categories:
1. Injection rate calculations
2. Outlet temperature predictions
3. Energy balance validation
4. Spray water pressure calculations
5. Control signal generation
6. PID control response
7. Mass and energy conservation verification

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import math
import time
from pathlib import Path
from decimal import Decimal
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from enum import Enum

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import test fixtures from conftest
from conftest import (
    SteamState,
    DesuperheaterConfig,
    generate_provenance_hash,
    assert_within_tolerance,
    assert_deterministic,
)


# =============================================================================
# ENUMS AND DATACLASSES FOR TESTING
# =============================================================================

class ControlMode(Enum):
    """Desuperheater control mode."""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"


@dataclass
class DesuperheaterInput:
    """Input data for desuperheater calculation."""
    inlet_temperature_c: float
    inlet_pressure_bar: float
    inlet_flow_kg_s: float
    target_temperature_c: float
    spray_water_temp_c: float
    spray_water_pressure_bar: float
    current_valve_position: float = 50.0
    control_mode: ControlMode = ControlMode.AUTO


@dataclass
class DesuperheaterOutput:
    """Output data from desuperheater calculation."""
    injection_rate_kg_s: float
    outlet_temperature_c: float
    outlet_flow_kg_s: float
    energy_balance_error_percent: float
    valve_position_percent: float
    control_signal: float
    superheat_reduction_c: float
    spray_utilization_percent: float
    provenance_hash: str
    calculation_time_ms: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class PIDState:
    """PID controller internal state."""
    integral: float = 0.0
    previous_error: float = 0.0
    output: float = 0.0
    saturated: bool = False


# =============================================================================
# MOCK CALCULATOR IMPLEMENTATION
# =============================================================================

class DesuperheaterCalculator:
    """
    Desuperheater calculator for steam temperature control.

    Calculates spray water injection rates to reduce steam superheat
    to target temperature using energy balance equations.
    """

    # Steam property constants (simplified)
    STEAM_CP_KJ_KG_K = 2.0  # Specific heat of superheated steam
    WATER_CP_KJ_KG_K = 4.186  # Specific heat of water
    WATER_HFG_KJ_KG = 2257.0  # Latent heat of vaporization

    def __init__(self, config: DesuperheaterConfig = None):
        """Initialize desuperheater calculator."""
        self.config = config or DesuperheaterConfig()
        self.calculation_count = 0
        self.pid_state = PIDState()

    def calculate_injection_rate(
        self,
        inlet_temp_c: float,
        inlet_flow_kg_s: float,
        target_temp_c: float,
        spray_temp_c: float,
        inlet_pressure_bar: float
    ) -> float:
        """
        Calculate required spray water injection rate.

        Based on energy balance:
        m_steam * cp_steam * (T_in - T_out) = m_spray * (cp_water * (T_sat - T_spray) + hfg)

        Args:
            inlet_temp_c: Inlet steam temperature
            inlet_flow_kg_s: Inlet steam flow rate
            target_temp_c: Target outlet temperature
            spray_temp_c: Spray water temperature
            inlet_pressure_bar: System pressure

        Returns:
            Required injection rate in kg/s
        """
        if inlet_temp_c <= target_temp_c:
            return 0.0  # No cooling needed

        if inlet_flow_kg_s <= 0:
            raise ValueError("Inlet flow must be positive")

        # Saturation temperature at operating pressure (simplified)
        t_sat = 100.0 + 58.0 * math.log10(inlet_pressure_bar)

        # Energy required to cool steam
        delta_t_steam = inlet_temp_c - target_temp_c
        q_cooling = inlet_flow_kg_s * self.STEAM_CP_KJ_KG_K * delta_t_steam

        # Energy absorbed by spray water
        # Water heats from spray temp to saturation, then evaporates
        delta_t_water = t_sat - spray_temp_c
        h_absorbed_per_kg = (
            self.WATER_CP_KJ_KG_K * delta_t_water +
            self.WATER_HFG_KJ_KG
        )

        # Required injection rate
        if h_absorbed_per_kg <= 0:
            raise ValueError("Invalid spray water temperature")

        injection_rate = q_cooling / h_absorbed_per_kg

        # Apply limits
        injection_rate = max(
            self.config.min_injection_rate_kg_s,
            min(self.config.max_injection_rate_kg_s, injection_rate)
        )

        return injection_rate

    def predict_outlet_temperature(
        self,
        inlet_temp_c: float,
        inlet_flow_kg_s: float,
        injection_rate_kg_s: float,
        spray_temp_c: float,
        inlet_pressure_bar: float
    ) -> float:
        """
        Predict outlet temperature given injection rate.

        Args:
            inlet_temp_c: Inlet steam temperature
            inlet_flow_kg_s: Inlet steam flow rate
            injection_rate_kg_s: Spray water injection rate
            spray_temp_c: Spray water temperature
            inlet_pressure_bar: System pressure

        Returns:
            Predicted outlet temperature in Celsius
        """
        if injection_rate_kg_s <= 0:
            return inlet_temp_c  # No cooling

        t_sat = 100.0 + 58.0 * math.log10(inlet_pressure_bar)

        # Energy absorbed by spray water
        delta_t_water = t_sat - spray_temp_c
        q_absorbed = injection_rate_kg_s * (
            self.WATER_CP_KJ_KG_K * delta_t_water +
            self.WATER_HFG_KJ_KG
        )

        # Temperature drop in steam
        total_flow = inlet_flow_kg_s + injection_rate_kg_s
        delta_t_steam = q_absorbed / (total_flow * self.STEAM_CP_KJ_KG_K)

        outlet_temp = inlet_temp_c - delta_t_steam

        # Cannot go below saturation
        return max(t_sat + 1.0, outlet_temp)

    def calculate_energy_balance(
        self,
        inlet_temp_c: float,
        inlet_flow_kg_s: float,
        outlet_temp_c: float,
        outlet_flow_kg_s: float,
        injection_rate_kg_s: float,
        spray_temp_c: float,
        inlet_pressure_bar: float
    ) -> Tuple[float, float]:
        """
        Validate energy balance and return error percentage.

        Returns:
            Tuple of (energy_in_kw, energy_error_percent)
        """
        t_sat = 100.0 + 58.0 * math.log10(inlet_pressure_bar)

        # Energy in from steam
        h_steam_in = self.STEAM_CP_KJ_KG_K * inlet_temp_c  # Simplified
        energy_steam_in = inlet_flow_kg_s * h_steam_in

        # Energy in from spray water
        h_water = self.WATER_CP_KJ_KG_K * spray_temp_c
        energy_water_in = injection_rate_kg_s * h_water

        # Total energy in
        energy_in = energy_steam_in + energy_water_in

        # Energy out
        h_steam_out = self.STEAM_CP_KJ_KG_K * outlet_temp_c
        energy_out = outlet_flow_kg_s * h_steam_out

        # Energy balance error
        if energy_in > 0:
            error_percent = abs(energy_out - energy_in) / energy_in * 100
        else:
            error_percent = 0.0

        return energy_in, error_percent

    def calculate_spray_pressure_required(
        self,
        system_pressure_bar: float,
        injection_rate_kg_s: float
    ) -> float:
        """
        Calculate required spray water pressure.

        Spray pressure must be higher than system pressure for injection.
        Typically 20-30% higher plus pressure drop through nozzles.

        Args:
            system_pressure_bar: Steam system pressure
            injection_rate_kg_s: Required injection rate

        Returns:
            Required spray water pressure in bar
        """
        # Base pressure differential (20% above system)
        base_differential = system_pressure_bar * 0.20

        # Nozzle pressure drop (proportional to flow squared)
        nozzle_cv = self.config.nozzle_cv * self.config.nozzle_count
        if nozzle_cv > 0:
            # Delta_P = (Q/Cv)^2 where Q in m3/hr
            # Simplified: assume 1 bar per (kg/s)^2 / Cv
            nozzle_dp = (injection_rate_kg_s ** 2) / (nozzle_cv ** 2) * 10
        else:
            nozzle_dp = 5.0  # Default

        required_pressure = system_pressure_bar + base_differential + nozzle_dp

        return required_pressure

    def calculate_pid_output(
        self,
        setpoint: float,
        process_value: float,
        dt_seconds: float = 1.0
    ) -> float:
        """
        Calculate PID controller output for temperature control.

        Args:
            setpoint: Target temperature
            process_value: Actual temperature
            dt_seconds: Time step

        Returns:
            Control output (valve position adjustment)
        """
        kp = self.config.pid_kp
        ki = self.config.pid_ki
        kd = self.config.pid_kd

        error = setpoint - process_value

        # Proportional term
        p_term = kp * error

        # Integral term (with anti-windup)
        if not self.pid_state.saturated:
            self.pid_state.integral += error * dt_seconds
        i_term = ki * self.pid_state.integral

        # Derivative term
        d_error = (error - self.pid_state.previous_error) / dt_seconds
        d_term = kd * d_error

        # Total output
        output = p_term + i_term + d_term

        # Apply limits
        output_limited = max(0.0, min(100.0, output + 50.0))  # Bias at 50%
        self.pid_state.saturated = (output_limited == 0.0 or output_limited == 100.0)

        # Store for next iteration
        self.pid_state.previous_error = error
        self.pid_state.output = output

        return output_limited

    def generate_control_signal(
        self,
        target_temp_c: float,
        actual_temp_c: float,
        current_valve_pos: float,
        rate_limit_percent_s: float = 5.0
    ) -> Tuple[float, float]:
        """
        Generate control signal with rate limiting.

        Args:
            target_temp_c: Target temperature
            actual_temp_c: Actual temperature
            current_valve_pos: Current valve position
            rate_limit_percent_s: Max rate of change per second

        Returns:
            Tuple of (new_valve_position, control_signal)
        """
        pid_output = self.calculate_pid_output(target_temp_c, actual_temp_c)

        # Rate limit
        delta = pid_output - current_valve_pos
        if abs(delta) > rate_limit_percent_s:
            delta = rate_limit_percent_s * (1 if delta > 0 else -1)

        new_position = current_valve_pos + delta
        new_position = max(0.0, min(100.0, new_position))

        control_signal = pid_output - 50.0  # Remove bias for signal

        return new_position, control_signal

    def calculate(self, input_data: DesuperheaterInput) -> DesuperheaterOutput:
        """
        Perform complete desuperheater calculation.

        Args:
            input_data: DesuperheaterInput with all operating parameters

        Returns:
            DesuperheaterOutput with calculated values
        """
        start_time = time.perf_counter()
        self.calculation_count += 1
        warnings = []

        # Calculate injection rate
        injection_rate = self.calculate_injection_rate(
            inlet_temp_c=input_data.inlet_temperature_c,
            inlet_flow_kg_s=input_data.inlet_flow_kg_s,
            target_temp_c=input_data.target_temperature_c,
            spray_temp_c=input_data.spray_water_temp_c,
            inlet_pressure_bar=input_data.inlet_pressure_bar
        )

        # Check spray pressure adequacy
        required_pressure = self.calculate_spray_pressure_required(
            input_data.inlet_pressure_bar, injection_rate
        )
        if input_data.spray_water_pressure_bar < required_pressure:
            warnings.append(
                f"Spray pressure {input_data.spray_water_pressure_bar} bar "
                f"below required {required_pressure:.1f} bar"
            )

        # Predict outlet temperature
        outlet_temp = self.predict_outlet_temperature(
            inlet_temp_c=input_data.inlet_temperature_c,
            inlet_flow_kg_s=input_data.inlet_flow_kg_s,
            injection_rate_kg_s=injection_rate,
            spray_temp_c=input_data.spray_water_temp_c,
            inlet_pressure_bar=input_data.inlet_pressure_bar
        )

        # Calculate outlet flow
        outlet_flow = input_data.inlet_flow_kg_s + injection_rate

        # Calculate energy balance error
        _, energy_error = self.calculate_energy_balance(
            inlet_temp_c=input_data.inlet_temperature_c,
            inlet_flow_kg_s=input_data.inlet_flow_kg_s,
            outlet_temp_c=outlet_temp,
            outlet_flow_kg_s=outlet_flow,
            injection_rate_kg_s=injection_rate,
            spray_temp_c=input_data.spray_water_temp_c,
            inlet_pressure_bar=input_data.inlet_pressure_bar
        )

        # Generate control signal
        valve_pos, control_signal = self.generate_control_signal(
            target_temp_c=input_data.target_temperature_c,
            actual_temp_c=outlet_temp,
            current_valve_pos=input_data.current_valve_position
        )

        # Calculate metrics
        superheat_reduction = input_data.inlet_temperature_c - outlet_temp
        spray_utilization = (injection_rate / self.config.max_injection_rate_kg_s) * 100

        # Generate provenance hash
        hash_data = {
            'inlet_temp_c': input_data.inlet_temperature_c,
            'inlet_pressure_bar': input_data.inlet_pressure_bar,
            'inlet_flow_kg_s': input_data.inlet_flow_kg_s,
            'injection_rate_kg_s': round(injection_rate, 10),
            'outlet_temp_c': round(outlet_temp, 6),
        }
        provenance_hash = generate_provenance_hash(hash_data)

        end_time = time.perf_counter()
        calc_time_ms = (end_time - start_time) * 1000

        return DesuperheaterOutput(
            injection_rate_kg_s=injection_rate,
            outlet_temperature_c=outlet_temp,
            outlet_flow_kg_s=outlet_flow,
            energy_balance_error_percent=energy_error,
            valve_position_percent=valve_pos,
            control_signal=control_signal,
            superheat_reduction_c=superheat_reduction,
            spray_utilization_percent=spray_utilization,
            provenance_hash=provenance_hash,
            calculation_time_ms=calc_time_ms,
            warnings=warnings
        )


# =============================================================================
# TEST CLASS: INJECTION RATE CALCULATIONS
# =============================================================================

class TestInjectionRateCalculations:
    """Test suite for spray water injection rate calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DesuperheaterCalculator()

    @pytest.fixture
    def config(self):
        """Create custom configuration."""
        return DesuperheaterConfig(
            max_injection_rate_kg_s=10.0,
            min_injection_rate_kg_s=0.1
        )

    @pytest.mark.unit
    def test_injection_rate_normal_operation(self, calculator):
        """Test injection rate for normal operating conditions."""
        rate = calculator.calculate_injection_rate(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            target_temp_c=300.0,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert rate > 0, "Injection rate should be positive"
        assert rate < calculator.config.max_injection_rate_kg_s
        # Typical values 2-5 kg/s for 50K reduction
        assert 1.0 < rate < 8.0

    @pytest.mark.unit
    def test_injection_rate_no_cooling_needed(self, calculator):
        """Test injection rate when no cooling is needed."""
        rate = calculator.calculate_injection_rate(
            inlet_temp_c=300.0,
            inlet_flow_kg_s=50.0,
            target_temp_c=350.0,  # Target above inlet
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert rate == 0.0, "No injection needed when target > inlet"

    @pytest.mark.unit
    def test_injection_rate_equal_temps(self, calculator):
        """Test injection rate when temperatures are equal."""
        rate = calculator.calculate_injection_rate(
            inlet_temp_c=300.0,
            inlet_flow_kg_s=50.0,
            target_temp_c=300.0,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert rate == 0.0

    @pytest.mark.unit
    def test_injection_rate_max_limit(self, calculator):
        """Test injection rate is limited to maximum."""
        rate = calculator.calculate_injection_rate(
            inlet_temp_c=500.0,  # Very high inlet
            inlet_flow_kg_s=100.0,  # High flow
            target_temp_c=280.0,  # Large reduction
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert rate <= calculator.config.max_injection_rate_kg_s

    @pytest.mark.unit
    def test_injection_rate_proportional_to_temp_diff(self, calculator):
        """Test injection rate increases with temperature difference."""
        rate_small = calculator.calculate_injection_rate(
            inlet_temp_c=320.0,
            inlet_flow_kg_s=50.0,
            target_temp_c=300.0,  # 20C reduction
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        rate_large = calculator.calculate_injection_rate(
            inlet_temp_c=400.0,
            inlet_flow_kg_s=50.0,
            target_temp_c=300.0,  # 100C reduction
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert rate_large > rate_small

    @pytest.mark.unit
    def test_injection_rate_proportional_to_flow(self, calculator):
        """Test injection rate increases with inlet flow."""
        rate_low_flow = calculator.calculate_injection_rate(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=25.0,
            target_temp_c=300.0,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        rate_high_flow = calculator.calculate_injection_rate(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=75.0,
            target_temp_c=300.0,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert rate_high_flow > rate_low_flow
        # Should scale roughly linearly
        ratio = rate_high_flow / rate_low_flow
        assert 2.5 < ratio < 3.5

    @pytest.mark.unit
    def test_injection_rate_invalid_flow_raises_error(self, calculator):
        """Test that zero or negative flow raises error."""
        with pytest.raises(ValueError, match="Inlet flow must be positive"):
            calculator.calculate_injection_rate(
                inlet_temp_c=350.0,
                inlet_flow_kg_s=0.0,
                target_temp_c=300.0,
                spray_temp_c=105.0,
                inlet_pressure_bar=40.0
            )


# =============================================================================
# TEST CLASS: OUTLET TEMPERATURE PREDICTIONS
# =============================================================================

class TestOutletTemperaturePredictions:
    """Test suite for outlet temperature prediction calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DesuperheaterCalculator()

    @pytest.mark.unit
    def test_outlet_temp_no_injection(self, calculator):
        """Test outlet equals inlet with no injection."""
        outlet = calculator.predict_outlet_temperature(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            injection_rate_kg_s=0.0,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert outlet == 350.0

    @pytest.mark.unit
    def test_outlet_temp_decreases_with_injection(self, calculator):
        """Test outlet temperature decreases with injection."""
        outlet = calculator.predict_outlet_temperature(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            injection_rate_kg_s=3.0,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert outlet < 350.0

    @pytest.mark.unit
    def test_outlet_temp_above_saturation(self, calculator):
        """Test outlet temperature stays above saturation."""
        outlet = calculator.predict_outlet_temperature(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            injection_rate_kg_s=10.0,  # High injection
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        # Saturation at 40 bar ~ 250C
        assert outlet > 250.0

    @pytest.mark.unit
    def test_outlet_temp_higher_injection_lower_temp(self, calculator):
        """Test higher injection gives lower outlet temperature."""
        outlet_low = calculator.predict_outlet_temperature(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            injection_rate_kg_s=2.0,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        outlet_high = calculator.predict_outlet_temperature(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            injection_rate_kg_s=5.0,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert outlet_high < outlet_low

    @pytest.mark.unit
    @pytest.mark.parametrize("injection_rate,expected_temp_range", [
        (1.0, (330, 345)),
        (3.0, (300, 320)),
        (5.0, (280, 300)),
        (8.0, (260, 285)),
    ])
    def test_outlet_temp_parametrized(
        self, calculator, injection_rate, expected_temp_range
    ):
        """Parametrized test for outlet temperature vs injection rate."""
        outlet = calculator.predict_outlet_temperature(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            injection_rate_kg_s=injection_rate,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert expected_temp_range[0] < outlet < expected_temp_range[1]


# =============================================================================
# TEST CLASS: ENERGY BALANCE VALIDATION
# =============================================================================

class TestEnergyBalanceValidation:
    """Test suite for energy balance validation calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DesuperheaterCalculator()

    @pytest.mark.unit
    def test_energy_balance_error_small(self, calculator):
        """Test energy balance error is small for consistent values."""
        # Use calculator to get consistent values
        injection_rate = calculator.calculate_injection_rate(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            target_temp_c=300.0,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        outlet_temp = calculator.predict_outlet_temperature(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            injection_rate_kg_s=injection_rate,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        _, error = calculator.calculate_energy_balance(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            outlet_temp_c=outlet_temp,
            outlet_flow_kg_s=50.0 + injection_rate,
            injection_rate_kg_s=injection_rate,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        # Energy balance error should be small
        assert error < 10.0, f"Energy balance error {error}% too high"

    @pytest.mark.unit
    def test_mass_conservation(self, calculator):
        """Test mass is conserved (outlet = inlet + spray)."""
        input_data = DesuperheaterInput(
            inlet_temperature_c=350.0,
            inlet_pressure_bar=40.0,
            inlet_flow_kg_s=50.0,
            target_temperature_c=300.0,
            spray_water_temp_c=105.0,
            spray_water_pressure_bar=60.0
        )

        result = calculator.calculate(input_data)

        expected_outlet_flow = 50.0 + result.injection_rate_kg_s
        assert_within_tolerance(
            result.outlet_flow_kg_s, expected_outlet_flow, 0.001,
            "Mass conservation"
        )

    @pytest.mark.unit
    def test_energy_balance_returns_positive_energy(self, calculator):
        """Test energy calculation returns positive values."""
        energy_in, error = calculator.calculate_energy_balance(
            inlet_temp_c=350.0,
            inlet_flow_kg_s=50.0,
            outlet_temp_c=300.0,
            outlet_flow_kg_s=53.0,
            injection_rate_kg_s=3.0,
            spray_temp_c=105.0,
            inlet_pressure_bar=40.0
        )

        assert energy_in > 0
        assert error >= 0


# =============================================================================
# TEST CLASS: SPRAY WATER PRESSURE CALCULATIONS
# =============================================================================

class TestSprayWaterPressureCalculations:
    """Test suite for spray water pressure calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DesuperheaterCalculator()

    @pytest.mark.unit
    def test_spray_pressure_above_system(self, calculator):
        """Test required spray pressure is above system pressure."""
        required = calculator.calculate_spray_pressure_required(
            system_pressure_bar=40.0,
            injection_rate_kg_s=3.0
        )

        assert required > 40.0

    @pytest.mark.unit
    def test_spray_pressure_increases_with_flow(self, calculator):
        """Test required pressure increases with injection rate."""
        p_low = calculator.calculate_spray_pressure_required(
            system_pressure_bar=40.0,
            injection_rate_kg_s=1.0
        )

        p_high = calculator.calculate_spray_pressure_required(
            system_pressure_bar=40.0,
            injection_rate_kg_s=5.0
        )

        assert p_high > p_low

    @pytest.mark.unit
    def test_spray_pressure_scales_with_system(self, calculator):
        """Test required pressure scales with system pressure."""
        p_low_system = calculator.calculate_spray_pressure_required(
            system_pressure_bar=20.0,
            injection_rate_kg_s=3.0
        )

        p_high_system = calculator.calculate_spray_pressure_required(
            system_pressure_bar=60.0,
            injection_rate_kg_s=3.0
        )

        assert p_high_system > p_low_system

    @pytest.mark.unit
    @pytest.mark.parametrize("system_p,injection,expected_min", [
        (10.0, 1.0, 12.0),
        (20.0, 2.0, 24.0),
        (40.0, 3.0, 48.0),
        (60.0, 5.0, 72.0),
    ])
    def test_spray_pressure_parametrized(
        self, calculator, system_p, injection, expected_min
    ):
        """Parametrized test for spray pressure requirements."""
        required = calculator.calculate_spray_pressure_required(
            system_pressure_bar=system_p,
            injection_rate_kg_s=injection
        )

        assert required >= expected_min


# =============================================================================
# TEST CLASS: CONTROL SIGNAL GENERATION
# =============================================================================

class TestControlSignalGeneration:
    """Test suite for control signal generation."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DesuperheaterCalculator()

    @pytest.mark.unit
    def test_control_signal_positive_error(self, calculator):
        """Test control signal for positive error (need more cooling)."""
        valve_pos, signal = calculator.generate_control_signal(
            target_temp_c=300.0,
            actual_temp_c=350.0,  # Above target
            current_valve_pos=50.0
        )

        # Should increase valve opening (more spray)
        assert valve_pos > 50.0

    @pytest.mark.unit
    def test_control_signal_negative_error(self, calculator):
        """Test control signal for negative error (too much cooling)."""
        valve_pos, signal = calculator.generate_control_signal(
            target_temp_c=300.0,
            actual_temp_c=280.0,  # Below target
            current_valve_pos=50.0
        )

        # Should decrease valve opening (less spray)
        assert valve_pos < 50.0

    @pytest.mark.unit
    def test_control_signal_at_setpoint(self, calculator):
        """Test control signal when at setpoint."""
        valve_pos, signal = calculator.generate_control_signal(
            target_temp_c=300.0,
            actual_temp_c=300.0,  # At target
            current_valve_pos=50.0
        )

        # Should stay near current position
        assert abs(valve_pos - 50.0) < 5.0

    @pytest.mark.unit
    def test_control_signal_rate_limited(self, calculator):
        """Test control signal is rate limited."""
        valve_pos, signal = calculator.generate_control_signal(
            target_temp_c=300.0,
            actual_temp_c=400.0,  # Large error
            current_valve_pos=50.0,
            rate_limit_percent_s=5.0
        )

        # Change should not exceed rate limit
        assert abs(valve_pos - 50.0) <= 5.0

    @pytest.mark.unit
    def test_valve_position_within_limits(self, calculator):
        """Test valve position stays within 0-100%."""
        # Try to drive below 0
        valve_pos, _ = calculator.generate_control_signal(
            target_temp_c=300.0,
            actual_temp_c=200.0,  # Way below target
            current_valve_pos=2.0
        )
        assert valve_pos >= 0.0

        # Try to drive above 100
        valve_pos, _ = calculator.generate_control_signal(
            target_temp_c=300.0,
            actual_temp_c=500.0,  # Way above target
            current_valve_pos=98.0
        )
        assert valve_pos <= 100.0


# =============================================================================
# TEST CLASS: PID CONTROL RESPONSE
# =============================================================================

class TestPIDControlResponse:
    """Test suite for PID controller response."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DesuperheaterCalculator()

    @pytest.mark.unit
    def test_pid_proportional_term(self, calculator):
        """Test PID proportional term contribution."""
        # Reset PID state
        calculator.pid_state = PIDState()

        output = calculator.calculate_pid_output(
            setpoint=300.0,
            process_value=290.0,  # 10 degree error
            dt_seconds=1.0
        )

        # With Kp=2.0, P term = 2.0 * 10 = 20
        # Output = 50 (bias) + 20 + I + D
        assert output > 50.0

    @pytest.mark.unit
    def test_pid_integral_accumulates(self, calculator):
        """Test PID integral term accumulates over time."""
        calculator.pid_state = PIDState()

        # Multiple iterations with constant error
        outputs = []
        for _ in range(5):
            output = calculator.calculate_pid_output(
                setpoint=300.0,
                process_value=295.0,  # Constant 5 degree error
                dt_seconds=1.0
            )
            outputs.append(output)

        # Output should increase due to integral
        assert outputs[-1] > outputs[0]

    @pytest.mark.unit
    def test_pid_derivative_response(self, calculator):
        """Test PID derivative term responds to rate of change."""
        calculator.pid_state = PIDState()

        # First call
        calculator.calculate_pid_output(300.0, 310.0, 1.0)

        # Second call with different error (error changed from -10 to -5)
        output = calculator.calculate_pid_output(300.0, 305.0, 1.0)

        # Derivative should contribute (error decreased, so positive D term)
        assert output > 50.0

    @pytest.mark.unit
    def test_pid_anti_windup(self, calculator):
        """Test PID anti-windup prevents integral windup."""
        calculator.pid_state = PIDState()

        # Drive to saturation
        for _ in range(20):
            output = calculator.calculate_pid_output(
                setpoint=300.0,
                process_value=200.0,  # Large error
                dt_seconds=1.0
            )

        # Should saturate at 100%
        assert output == 100.0
        assert calculator.pid_state.saturated

    @pytest.mark.unit
    def test_pid_output_limits(self, calculator):
        """Test PID output is within 0-100%."""
        calculator.pid_state = PIDState()

        # Large positive error
        output = calculator.calculate_pid_output(300.0, 100.0, 1.0)
        assert 0.0 <= output <= 100.0

        # Large negative error
        calculator.pid_state = PIDState()
        output = calculator.calculate_pid_output(300.0, 500.0, 1.0)
        assert 0.0 <= output <= 100.0


# =============================================================================
# TEST CLASS: FULL CALCULATION WORKFLOW
# =============================================================================

class TestFullDesuperheaterCalculation:
    """Test suite for complete desuperheater calculation workflow."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DesuperheaterCalculator()

    @pytest.mark.unit
    def test_complete_calculation_normal(self, calculator):
        """Test complete calculation for normal operation."""
        input_data = DesuperheaterInput(
            inlet_temperature_c=350.0,
            inlet_pressure_bar=40.0,
            inlet_flow_kg_s=50.0,
            target_temperature_c=300.0,
            spray_water_temp_c=105.0,
            spray_water_pressure_bar=60.0
        )

        result = calculator.calculate(input_data)

        assert result.injection_rate_kg_s > 0
        assert result.outlet_temperature_c < 350.0
        assert result.outlet_flow_kg_s > 50.0
        assert result.superheat_reduction_c > 0
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_complete_calculation_low_pressure_warning(self, calculator):
        """Test calculation generates warning for low spray pressure."""
        input_data = DesuperheaterInput(
            inlet_temperature_c=350.0,
            inlet_pressure_bar=40.0,
            inlet_flow_kg_s=50.0,
            target_temperature_c=300.0,
            spray_water_temp_c=105.0,
            spray_water_pressure_bar=30.0  # Too low
        )

        result = calculator.calculate(input_data)

        assert len(result.warnings) > 0
        assert any('pressure' in w.lower() for w in result.warnings)

    @pytest.mark.unit
    def test_calculation_count_increments(self, calculator):
        """Test calculation counter increments."""
        initial = calculator.calculation_count

        input_data = DesuperheaterInput(
            inlet_temperature_c=350.0,
            inlet_pressure_bar=40.0,
            inlet_flow_kg_s=50.0,
            target_temperature_c=300.0,
            spray_water_temp_c=105.0,
            spray_water_pressure_bar=60.0
        )

        calculator.calculate(input_data)
        calculator.calculate(input_data)

        assert calculator.calculation_count == initial + 2


# =============================================================================
# TEST CLASS: DETERMINISM
# =============================================================================

class TestDesuperheaterDeterminism:
    """Test suite for determinism verification."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DesuperheaterCalculator()

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_injection_rate_determinism(self, calculator):
        """Test injection rate is deterministic."""
        results = []
        for _ in range(100):
            rate = calculator.calculate_injection_rate(
                inlet_temp_c=350.0,
                inlet_flow_kg_s=50.0,
                target_temp_c=300.0,
                spray_temp_c=105.0,
                inlet_pressure_bar=40.0
            )
            results.append(rate)

        assert_deterministic(results, "Injection rate")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_full_calculation_determinism(self, calculator):
        """Test full calculation is deterministic."""
        input_data = DesuperheaterInput(
            inlet_temperature_c=350.0,
            inlet_pressure_bar=40.0,
            inlet_flow_kg_s=50.0,
            target_temperature_c=300.0,
            spray_water_temp_c=105.0,
            spray_water_pressure_bar=60.0
        )

        # Reset PID state for consistent results
        calculator.pid_state = PIDState()
        results = []
        for _ in range(50):
            calculator.pid_state = PIDState()  # Reset each time
            result = calculator.calculate(input_data)
            results.append(result.injection_rate_kg_s)

        assert_deterministic(results, "Full calculation")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_provenance_hash_reproducibility(self, calculator):
        """Test provenance hash is reproducible."""
        input_data = DesuperheaterInput(
            inlet_temperature_c=350.0,
            inlet_pressure_bar=40.0,
            inlet_flow_kg_s=50.0,
            target_temperature_c=300.0,
            spray_water_temp_c=105.0,
            spray_water_pressure_bar=60.0
        )

        calculator.pid_state = PIDState()
        result1 = calculator.calculate(input_data)

        calculator.pid_state = PIDState()
        result2 = calculator.calculate(input_data)

        assert result1.provenance_hash == result2.provenance_hash


# =============================================================================
# TEST CLASS: PERFORMANCE
# =============================================================================

class TestDesuperheaterPerformance:
    """Test suite for performance benchmarks."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DesuperheaterCalculator()

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_calculation_under_10ms(self, calculator, benchmark_targets):
        """Test single calculation completes under 10ms."""
        input_data = DesuperheaterInput(
            inlet_temperature_c=350.0,
            inlet_pressure_bar=40.0,
            inlet_flow_kg_s=50.0,
            target_temperature_c=300.0,
            spray_water_temp_c=105.0,
            spray_water_pressure_bar=60.0
        )

        result = calculator.calculate(input_data)

        assert result.calculation_time_ms < benchmark_targets['desuperheater_calculation_ms']

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_calculations(self, calculator, performance_timer):
        """Test batch of 100 calculations."""
        input_data = DesuperheaterInput(
            inlet_temperature_c=350.0,
            inlet_pressure_bar=40.0,
            inlet_flow_kg_s=50.0,
            target_temperature_c=300.0,
            spray_water_temp_c=105.0,
            spray_water_pressure_bar=60.0
        )

        with performance_timer() as timer:
            for _ in range(100):
                calculator.calculate(input_data)

        # 100 calculations should complete in under 500ms
        assert timer.elapsed_ms < 500.0
