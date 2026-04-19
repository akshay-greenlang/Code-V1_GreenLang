"""
GL-022 SuperheaterControlAgent - Calculator/Formula Tests

This module provides comprehensive tests for:
- Thermodynamic calculations (IAPWS-IF97 compliance)
- Saturation temperature calculations
- Superheat calculations
- Steam enthalpy calculations
- Spray water flow calculations
- Valve position calculations
- PID parameter calculations
- Energy loss calculations
- Provenance hash generation

Target: 85%+ coverage for formulas.py

Standards tested against:
- IAPWS-IF97 (International Association for Properties of Water and Steam)
- ASME PTC 4 (Performance Test Code for Fired Steam Generators)
"""

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# Add agent paths
AGENT_BASE_PATH = Path(__file__).parent.parent.parent.parent.parent.parent
BACKEND_AGENT_PATH = AGENT_BASE_PATH / "GL-Agent-Factory" / "backend" / "agents"
sys.path.insert(0, str(AGENT_BASE_PATH))
sys.path.insert(0, str(BACKEND_AGENT_PATH))

try:
    from gl_022_superheater_control.formulas import (
        calculate_saturation_temperature,
        calculate_superheat,
        calculate_steam_enthalpy,
        calculate_spray_water_flow,
        calculate_valve_position,
        calculate_pid_parameters,
        calculate_spray_energy_loss,
        calculate_thermal_efficiency_impact,
        generate_calculation_hash,
        bar_to_psi,
        celsius_to_fahrenheit,
        kg_s_to_lb_hr,
        SteamProperties,
    )
    FORMULAS_AVAILABLE = True
except ImportError:
    FORMULAS_AVAILABLE = False

# Skip all tests if formulas not available
pytestmark = pytest.mark.skipif(not FORMULAS_AVAILABLE, reason="Formulas module not available")


# =============================================================================
# SATURATION TEMPERATURE TESTS
# =============================================================================

class TestSaturationTemperature:
    """Tests for calculate_saturation_temperature function."""

    @pytest.mark.unit
    def test_saturation_temp_at_1_bar(self):
        """Test saturation temperature at 1 bar (near 100C)."""
        t_sat = calculate_saturation_temperature(1.0)

        # At 1 bar (atmospheric), saturation temp should be ~100C
        assert 99.0 <= t_sat <= 101.0

    @pytest.mark.unit
    def test_saturation_temp_at_10_bar(self):
        """Test saturation temperature at 10 bar (~180C)."""
        t_sat = calculate_saturation_temperature(10.0)

        # IAPWS-IF97: T_sat at 10 bar = 179.88C
        assert 175.0 <= t_sat <= 185.0

    @pytest.mark.unit
    def test_saturation_temp_at_40_bar(self):
        """Test saturation temperature at 40 bar (~250C)."""
        t_sat = calculate_saturation_temperature(40.0)

        # IAPWS-IF97: T_sat at 40 bar = 250.35C
        assert 245.0 <= t_sat <= 255.0

    @pytest.mark.unit
    def test_saturation_temp_at_100_bar(self):
        """Test saturation temperature at 100 bar (~311C)."""
        t_sat = calculate_saturation_temperature(100.0)

        # IAPWS-IF97: T_sat at 100 bar = 311.0C
        assert 305.0 <= t_sat <= 320.0

    @pytest.mark.unit
    @pytest.mark.parametrize("pressure,expected_t_sat,tolerance", [
        (1.0, 99.97, 2.0),
        (5.0, 151.84, 3.0),
        (10.0, 179.88, 3.0),
        (20.0, 212.38, 4.0),
        (40.0, 250.35, 4.0),
        (60.0, 275.58, 5.0),
        (100.0, 311.0, 6.0),
        (150.0, 342.16, 8.0),
    ])
    def test_saturation_temp_iapws_reference(self, pressure, expected_t_sat, tolerance):
        """
        Test saturation temperature against IAPWS-IF97 reference values.

        Uses polynomial approximation, so some deviation is expected.
        """
        t_sat = calculate_saturation_temperature(pressure)

        assert t_sat == pytest.approx(expected_t_sat, abs=tolerance), \
            f"At {pressure} bar: got {t_sat}C, expected ~{expected_t_sat}C"

    @pytest.mark.unit
    def test_saturation_temp_monotonic_increasing(self):
        """Test saturation temperature increases with pressure."""
        pressures = [1, 5, 10, 20, 40, 60, 80, 100, 150, 200]
        temperatures = [calculate_saturation_temperature(p) for p in pressures]

        for i in range(len(temperatures) - 1):
            assert temperatures[i] < temperatures[i + 1], \
                f"T_sat should increase: {temperatures[i]} < {temperatures[i + 1]}"

    @pytest.mark.unit
    def test_saturation_temp_zero_pressure_raises(self):
        """Test zero pressure raises ValueError."""
        with pytest.raises(ValueError, match="Pressure must be positive"):
            calculate_saturation_temperature(0.0)

    @pytest.mark.unit
    def test_saturation_temp_negative_pressure_raises(self):
        """Test negative pressure raises ValueError."""
        with pytest.raises(ValueError, match="Pressure must be positive"):
            calculate_saturation_temperature(-10.0)

    @pytest.mark.unit
    def test_saturation_temp_returns_rounded(self):
        """Test result is rounded to 2 decimal places."""
        t_sat = calculate_saturation_temperature(40.0)

        # Check it's a float rounded to 2 decimals
        t_sat_str = str(t_sat)
        if "." in t_sat_str:
            decimal_places = len(t_sat_str.split(".")[1])
            assert decimal_places <= 2


# =============================================================================
# SUPERHEAT CALCULATION TESTS
# =============================================================================

class TestSuperheatCalculation:
    """Tests for calculate_superheat function."""

    @pytest.mark.unit
    def test_superheat_positive(self):
        """Test superheat is positive when steam is superheated."""
        # At 40 bar, T_sat ~ 250C, so 400C steam has ~150C superheat
        superheat = calculate_superheat(400.0, 40.0)

        assert superheat > 0
        assert 145.0 <= superheat <= 155.0

    @pytest.mark.unit
    def test_superheat_zero_at_saturation(self):
        """Test superheat is zero at saturation temperature."""
        t_sat = calculate_saturation_temperature(40.0)
        superheat = calculate_superheat(t_sat, 40.0)

        assert superheat == pytest.approx(0.0, abs=0.1)

    @pytest.mark.unit
    def test_superheat_negative_wet_steam(self):
        """Test superheat is negative for wet steam."""
        # At 40 bar, T_sat ~ 250C, so 200C is wet steam
        superheat = calculate_superheat(200.0, 40.0)

        assert superheat < 0

    @pytest.mark.unit
    @pytest.mark.parametrize("steam_temp,pressure,expected_superheat", [
        (300.0, 10.0, 120.0),   # 300C at 10 bar (T_sat~180C)
        (400.0, 40.0, 150.0),   # 400C at 40 bar (T_sat~250C)
        (500.0, 60.0, 224.0),   # 500C at 60 bar (T_sat~276C)
        (550.0, 100.0, 239.0),  # 550C at 100 bar (T_sat~311C)
    ])
    def test_superheat_various_conditions(self, steam_temp, pressure, expected_superheat):
        """Test superheat calculation at various conditions."""
        superheat = calculate_superheat(steam_temp, pressure)

        # Allow 10C tolerance due to approximations
        assert superheat == pytest.approx(expected_superheat, abs=10.0)

    @pytest.mark.unit
    def test_superheat_returns_rounded(self):
        """Test superheat is rounded to 2 decimal places."""
        superheat = calculate_superheat(450.0, 40.0)

        superheat_str = str(superheat)
        if "." in superheat_str:
            decimal_places = len(superheat_str.split(".")[1])
            assert decimal_places <= 2


# =============================================================================
# STEAM ENTHALPY TESTS
# =============================================================================

class TestSteamEnthalpy:
    """Tests for calculate_steam_enthalpy function."""

    @pytest.mark.unit
    def test_enthalpy_superheated_steam(self):
        """Test enthalpy calculation for superheated steam."""
        # At 40 bar, 400C superheated steam
        enthalpy = calculate_steam_enthalpy(400.0, 40.0)

        # Should be > 2800 kJ/kg for superheated steam
        assert enthalpy > 2800

    @pytest.mark.unit
    def test_enthalpy_increases_with_temperature(self):
        """Test enthalpy increases with temperature at constant pressure."""
        temps = [300, 350, 400, 450, 500]
        enthalpies = [calculate_steam_enthalpy(t, 40.0) for t in temps]

        for i in range(len(enthalpies) - 1):
            assert enthalpies[i] < enthalpies[i + 1]

    @pytest.mark.unit
    def test_enthalpy_wet_steam_returns_saturation(self):
        """Test wet steam returns saturated vapor enthalpy."""
        # Temperature below saturation at 40 bar
        enthalpy_wet = calculate_steam_enthalpy(200.0, 40.0)
        t_sat = calculate_saturation_temperature(40.0)
        enthalpy_sat = calculate_steam_enthalpy(t_sat, 40.0)

        # Wet steam should return saturation enthalpy
        assert enthalpy_wet == pytest.approx(enthalpy_sat, abs=1.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("temp,pressure,min_h,max_h", [
        (200.0, 10.0, 2750, 2900),   # Low superheat
        (300.0, 20.0, 2950, 3100),   # Medium pressure
        (400.0, 40.0, 3100, 3300),   # High superheat
        (500.0, 60.0, 3300, 3500),   # Very high superheat
    ])
    def test_enthalpy_reasonable_range(self, temp, pressure, min_h, max_h):
        """Test enthalpy falls within reasonable range."""
        enthalpy = calculate_steam_enthalpy(temp, pressure)

        assert min_h <= enthalpy <= max_h, \
            f"Enthalpy {enthalpy} outside expected range [{min_h}, {max_h}]"

    @pytest.mark.unit
    def test_enthalpy_difference_for_control(self):
        """Test enthalpy difference for temperature control scenario."""
        # This simulates the enthalpy reduction needed for desuperheating
        h_in = calculate_steam_enthalpy(450.0, 40.0)
        h_out = calculate_steam_enthalpy(400.0, 40.0)

        delta_h = h_in - h_out

        # For 50C temperature drop with Cp~2.1 kJ/kg.K
        # Expected: ~105 kJ/kg
        assert 80 <= delta_h <= 130


# =============================================================================
# SPRAY WATER FLOW CALCULATION TESTS
# =============================================================================

class TestSprayWaterFlow:
    """Tests for calculate_spray_water_flow function."""

    @pytest.mark.unit
    def test_spray_flow_positive_when_cooling_needed(self):
        """Test spray flow is positive when temperature reduction needed."""
        spray_flow, energy = calculate_spray_water_flow(
            steam_flow_kg_s=20.0,
            steam_temp_in_c=450.0,
            steam_temp_target_c=400.0,
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        assert spray_flow > 0
        assert energy > 0

    @pytest.mark.unit
    def test_spray_flow_zero_when_no_cooling(self):
        """Test spray flow is zero when no cooling needed."""
        spray_flow, energy = calculate_spray_water_flow(
            steam_flow_kg_s=20.0,
            steam_temp_in_c=400.0,
            steam_temp_target_c=400.0,  # Already at target
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        assert spray_flow == 0.0
        assert energy == 0.0

    @pytest.mark.unit
    def test_spray_flow_zero_when_below_target(self):
        """Test spray flow is zero when temp below target."""
        spray_flow, energy = calculate_spray_water_flow(
            steam_flow_kg_s=20.0,
            steam_temp_in_c=380.0,  # Below target
            steam_temp_target_c=400.0,
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        assert spray_flow == 0.0
        assert energy == 0.0

    @pytest.mark.unit
    def test_spray_flow_increases_with_temp_difference(self):
        """Test spray flow increases with larger temperature difference."""
        spray_small, _ = calculate_spray_water_flow(
            steam_flow_kg_s=20.0,
            steam_temp_in_c=420.0,
            steam_temp_target_c=400.0,  # 20C difference
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        spray_large, _ = calculate_spray_water_flow(
            steam_flow_kg_s=20.0,
            steam_temp_in_c=480.0,
            steam_temp_target_c=400.0,  # 80C difference
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        assert spray_large > spray_small

    @pytest.mark.unit
    def test_spray_flow_scales_with_steam_flow(self):
        """Test spray flow scales proportionally with steam flow."""
        spray_10, _ = calculate_spray_water_flow(
            steam_flow_kg_s=10.0,
            steam_temp_in_c=450.0,
            steam_temp_target_c=400.0,
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        spray_20, _ = calculate_spray_water_flow(
            steam_flow_kg_s=20.0,
            steam_temp_in_c=450.0,
            steam_temp_target_c=400.0,
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        # Should be approximately double
        assert spray_20 == pytest.approx(spray_10 * 2, rel=0.05)

    @pytest.mark.unit
    def test_spray_flow_returns_tuple(self):
        """Test function returns tuple of (flow, energy)."""
        result = calculate_spray_water_flow(
            steam_flow_kg_s=20.0,
            steam_temp_in_c=450.0,
            steam_temp_target_c=400.0,
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.unit
    def test_spray_flow_rounded(self):
        """Test spray flow is rounded to 4 decimal places."""
        spray_flow, energy = calculate_spray_water_flow(
            steam_flow_kg_s=20.0,
            steam_temp_in_c=450.0,
            steam_temp_target_c=400.0,
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        flow_str = str(spray_flow)
        if "." in flow_str:
            decimal_places = len(flow_str.split(".")[1])
            assert decimal_places <= 4

    @pytest.mark.unit
    def test_spray_flow_energy_balance(self):
        """Test energy balance is maintained."""
        steam_flow = 20.0
        t_in = 450.0
        t_out = 400.0
        spray_water_temp = 100.0
        pressure = 40.0

        spray_flow, energy_absorbed = calculate_spray_water_flow(
            steam_flow_kg_s=steam_flow,
            steam_temp_in_c=t_in,
            steam_temp_target_c=t_out,
            spray_water_temp_c=spray_water_temp,
            steam_pressure_bar=pressure
        )

        # Energy absorbed should be positive
        assert energy_absorbed > 0

        # Verify energy balance (approximately)
        h_in = calculate_steam_enthalpy(t_in, pressure)
        h_out = calculate_steam_enthalpy(t_out, pressure)

        # Energy removed from steam = steam_flow * (h_in - h_out)
        energy_removed = steam_flow * (h_in - h_out)

        # This should approximately equal energy absorbed by spray
        assert energy_absorbed == pytest.approx(energy_removed, rel=0.1)


# =============================================================================
# VALVE POSITION CALCULATION TESTS
# =============================================================================

class TestValvePosition:
    """Tests for calculate_valve_position function."""

    @pytest.mark.unit
    def test_valve_position_zero_flow(self):
        """Test valve position is 0% at zero flow."""
        position = calculate_valve_position(0.0, 10.0)

        assert position == 0.0

    @pytest.mark.unit
    def test_valve_position_full_flow(self):
        """Test valve position is 100% at max flow."""
        position = calculate_valve_position(10.0, 10.0)

        assert position == 100.0

    @pytest.mark.unit
    def test_valve_position_half_flow(self):
        """Test valve position is 50% at half flow."""
        position = calculate_valve_position(5.0, 10.0)

        assert position == 50.0

    @pytest.mark.unit
    @pytest.mark.parametrize("required,max_flow,expected_position", [
        (0.0, 10.0, 0.0),
        (1.0, 10.0, 10.0),
        (2.5, 10.0, 25.0),
        (5.0, 10.0, 50.0),
        (7.5, 10.0, 75.0),
        (10.0, 10.0, 100.0),
    ])
    def test_valve_position_linear(self, required, max_flow, expected_position):
        """Test valve position follows linear characteristic."""
        position = calculate_valve_position(required, max_flow)

        assert position == pytest.approx(expected_position, abs=0.1)

    @pytest.mark.unit
    def test_valve_position_clamped_at_100(self):
        """Test valve position is clamped at 100%."""
        position = calculate_valve_position(15.0, 10.0)  # Exceeds max

        assert position == 100.0

    @pytest.mark.unit
    def test_valve_position_clamped_at_0(self):
        """Test valve position is clamped at 0%."""
        position = calculate_valve_position(-5.0, 10.0)  # Negative

        assert position == 0.0

    @pytest.mark.unit
    def test_valve_position_zero_max_flow(self):
        """Test valve position is 0% when max flow is 0."""
        position = calculate_valve_position(5.0, 0.0)

        assert position == 0.0

    @pytest.mark.unit
    def test_valve_position_negative_max_flow(self):
        """Test valve position is 0% when max flow is negative."""
        position = calculate_valve_position(5.0, -10.0)

        assert position == 0.0

    @pytest.mark.unit
    def test_valve_position_rounded(self):
        """Test valve position is rounded to 1 decimal place."""
        position = calculate_valve_position(3.33333, 10.0)

        position_str = str(position)
        if "." in position_str:
            decimal_places = len(position_str.split(".")[1])
            assert decimal_places <= 1


# =============================================================================
# PID PARAMETER CALCULATION TESTS
# =============================================================================

class TestPIDParameters:
    """Tests for calculate_pid_parameters function."""

    @pytest.mark.unit
    def test_pid_returns_all_params(self):
        """Test PID calculation returns all required parameters."""
        params = calculate_pid_parameters()

        required_keys = ["kp", "ki", "kd", "deadband_c", "max_rate_c_per_min"]
        for key in required_keys:
            assert key in params

    @pytest.mark.unit
    def test_pid_kp_positive(self):
        """Test Kp is always positive."""
        params = calculate_pid_parameters(60.0, 10.0, 120.0)

        assert params["kp"] > 0

    @pytest.mark.unit
    def test_pid_ki_positive(self):
        """Test Ki is always positive."""
        params = calculate_pid_parameters(60.0, 10.0, 120.0)

        assert params["ki"] > 0

    @pytest.mark.unit
    def test_pid_kd_non_negative(self):
        """Test Kd is non-negative."""
        params = calculate_pid_parameters(60.0, 10.0, 120.0)

        assert params["kd"] >= 0

    @pytest.mark.unit
    def test_pid_lambda_tuning_formula(self):
        """Test Lambda tuning formula is correctly implemented."""
        tau = 60.0
        theta = 10.0
        lambda_cl = 120.0

        params = calculate_pid_parameters(tau, theta, lambda_cl)

        # Lambda tuning formulas
        expected_kp = tau / (1.0 * (lambda_cl + theta))
        expected_ki = expected_kp / tau
        expected_kd = expected_kp * theta / 2

        assert params["kp"] == pytest.approx(expected_kp, rel=0.001)
        assert params["ki"] == pytest.approx(expected_ki, rel=0.001)
        assert params["kd"] == pytest.approx(expected_kd, rel=0.001)

    @pytest.mark.unit
    def test_pid_default_values(self):
        """Test default parameter values."""
        params = calculate_pid_parameters()

        assert params["deadband_c"] == 1.0
        assert params["max_rate_c_per_min"] == 5.0

    @pytest.mark.unit
    @pytest.mark.parametrize("tau,theta,lambda_cl", [
        (30.0, 5.0, 60.0),
        (60.0, 10.0, 120.0),
        (90.0, 15.0, 180.0),
        (120.0, 20.0, 240.0),
    ])
    def test_pid_various_dynamics(self, tau, theta, lambda_cl):
        """Test PID calculation with various process dynamics."""
        params = calculate_pid_parameters(tau, theta, lambda_cl)

        # All should be positive
        assert params["kp"] > 0
        assert params["ki"] > 0
        assert params["kd"] >= 0


# =============================================================================
# ENERGY LOSS CALCULATION TESTS
# =============================================================================

class TestSprayEnergyLoss:
    """Tests for calculate_spray_energy_loss function."""

    @pytest.mark.unit
    def test_energy_loss_zero_spray(self):
        """Test energy loss is zero with no spray."""
        loss = calculate_spray_energy_loss(0.0, 100.0)

        assert loss == 0.0

    @pytest.mark.unit
    def test_energy_loss_positive(self):
        """Test energy loss is positive with spray."""
        loss = calculate_spray_energy_loss(1.0, 100.0)  # 1 kg/s, 100 kJ/kg

        assert loss == 100.0  # 100 kW

    @pytest.mark.unit
    def test_energy_loss_scales_linearly(self):
        """Test energy loss scales linearly with flow."""
        loss_1 = calculate_spray_energy_loss(1.0, 100.0)
        loss_2 = calculate_spray_energy_loss(2.0, 100.0)

        assert loss_2 == 2 * loss_1

    @pytest.mark.unit
    def test_energy_loss_rounded(self):
        """Test energy loss is rounded to 2 decimal places."""
        loss = calculate_spray_energy_loss(1.5555, 100.123)

        loss_str = str(loss)
        if "." in loss_str:
            decimal_places = len(loss_str.split(".")[1])
            assert decimal_places <= 2


# =============================================================================
# THERMAL EFFICIENCY IMPACT TESTS
# =============================================================================

class TestThermalEfficiencyImpact:
    """Tests for calculate_thermal_efficiency_impact function."""

    @pytest.mark.unit
    def test_efficiency_impact_zero_loss(self):
        """Test efficiency impact is zero with no spray loss."""
        impact = calculate_thermal_efficiency_impact(0.0, 10000.0)

        assert impact == 0.0

    @pytest.mark.unit
    def test_efficiency_impact_calculation(self):
        """Test efficiency impact calculation."""
        # 100 kW spray loss / 10000 kW fuel = 1% impact
        impact = calculate_thermal_efficiency_impact(100.0, 10000.0)

        assert impact == 1.0

    @pytest.mark.unit
    def test_efficiency_impact_zero_fuel(self):
        """Test efficiency impact is zero when fuel input is zero."""
        impact = calculate_thermal_efficiency_impact(100.0, 0.0)

        assert impact == 0.0

    @pytest.mark.unit
    def test_efficiency_impact_negative_fuel(self):
        """Test efficiency impact is zero when fuel input is negative."""
        impact = calculate_thermal_efficiency_impact(100.0, -10000.0)

        assert impact == 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize("spray_loss,fuel_input,expected_impact", [
        (50.0, 10000.0, 0.5),
        (100.0, 10000.0, 1.0),
        (200.0, 10000.0, 2.0),
        (100.0, 5000.0, 2.0),
    ])
    def test_efficiency_impact_various_values(self, spray_loss, fuel_input, expected_impact):
        """Test efficiency impact with various values."""
        impact = calculate_thermal_efficiency_impact(spray_loss, fuel_input)

        assert impact == pytest.approx(expected_impact, rel=0.001)


# =============================================================================
# PROVENANCE HASH TESTS
# =============================================================================

class TestProvenanceHash:
    """Tests for generate_calculation_hash function."""

    @pytest.mark.unit
    def test_hash_returns_64_char_hex(self):
        """Test hash returns 64-character hexadecimal string."""
        inputs = {"temp": 400.0, "pressure": 40.0}
        outputs = {"spray_flow": 1.5}

        hash_value = generate_calculation_hash(inputs, outputs)

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    @pytest.mark.unit
    def test_hash_deterministic(self):
        """Test hash is deterministic for same inputs."""
        inputs = {"temp": 400.0, "pressure": 40.0}
        outputs = {"spray_flow": 1.5}

        hash1 = generate_calculation_hash(inputs, outputs)
        hash2 = generate_calculation_hash(inputs, outputs)

        assert hash1 == hash2

    @pytest.mark.unit
    def test_hash_different_for_different_inputs(self):
        """Test hash changes with different inputs."""
        inputs1 = {"temp": 400.0, "pressure": 40.0}
        inputs2 = {"temp": 401.0, "pressure": 40.0}  # 1 degree different
        outputs = {"spray_flow": 1.5}

        hash1 = generate_calculation_hash(inputs1, outputs)
        hash2 = generate_calculation_hash(inputs2, outputs)

        assert hash1 != hash2

    @pytest.mark.unit
    def test_hash_different_for_different_outputs(self):
        """Test hash changes with different outputs."""
        inputs = {"temp": 400.0, "pressure": 40.0}
        outputs1 = {"spray_flow": 1.5}
        outputs2 = {"spray_flow": 1.6}

        hash1 = generate_calculation_hash(inputs, outputs1)
        hash2 = generate_calculation_hash(inputs, outputs2)

        assert hash1 != hash2

    @pytest.mark.unit
    def test_hash_includes_metadata(self):
        """Test hash includes formula version and standard."""
        inputs = {"temp": 400.0}
        outputs = {"result": 1.0}

        # The hash should include formula_version and standard
        hash_value = generate_calculation_hash(inputs, outputs)

        # Manually verify the structure
        data = {
            "inputs": inputs,
            "outputs": outputs,
            "formula_version": "1.0.0",
            "standard": "IAPWS-IF97"
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        expected_hash = hashlib.sha256(json_str.encode()).hexdigest()

        assert hash_value == expected_hash

    @pytest.mark.unit
    def test_hash_order_independent_keys(self):
        """Test hash is same regardless of key order in input dicts."""
        inputs1 = {"a": 1, "b": 2, "c": 3}
        inputs2 = {"c": 3, "a": 1, "b": 2}  # Different order
        outputs = {"x": 10}

        hash1 = generate_calculation_hash(inputs1, outputs)
        hash2 = generate_calculation_hash(inputs2, outputs)

        # Should be same due to sort_keys=True
        assert hash1 == hash2

    @pytest.mark.unit
    def test_hash_with_complex_values(self):
        """Test hash handles complex nested values."""
        inputs = {
            "temperatures": [400.0, 401.0, 402.0],
            "nested": {"inner": 100.0}
        }
        outputs = {"results": [1.0, 2.0, 3.0]}

        # Should not raise an error
        hash_value = generate_calculation_hash(inputs, outputs)

        assert len(hash_value) == 64


# =============================================================================
# UNIT CONVERSION TESTS
# =============================================================================

class TestUnitConversions:
    """Tests for unit conversion helper functions."""

    @pytest.mark.unit
    @pytest.mark.parametrize("bar,expected_psi", [
        (1.0, 14.50),
        (10.0, 145.04),
        (40.0, 580.15),
        (100.0, 1450.38),
    ])
    def test_bar_to_psi(self, bar, expected_psi):
        """Test bar to PSI conversion."""
        psi = bar_to_psi(bar)

        assert psi == pytest.approx(expected_psi, rel=0.01)

    @pytest.mark.unit
    @pytest.mark.parametrize("celsius,expected_fahrenheit", [
        (0.0, 32.0),
        (100.0, 212.0),
        (200.0, 392.0),
        (400.0, 752.0),
    ])
    def test_celsius_to_fahrenheit(self, celsius, expected_fahrenheit):
        """Test Celsius to Fahrenheit conversion."""
        fahrenheit = celsius_to_fahrenheit(celsius)

        assert fahrenheit == pytest.approx(expected_fahrenheit, rel=0.01)

    @pytest.mark.unit
    @pytest.mark.parametrize("kg_s,expected_lb_hr", [
        (1.0, 7936.64),
        (10.0, 79366.4),
        (0.1, 793.66),
    ])
    def test_kg_s_to_lb_hr(self, kg_s, expected_lb_hr):
        """Test kg/s to lb/hr conversion."""
        lb_hr = kg_s_to_lb_hr(kg_s)

        assert lb_hr == pytest.approx(expected_lb_hr, rel=0.01)


# =============================================================================
# INTEGRATION TESTS FOR CALCULATION CHAINS
# =============================================================================

class TestCalculationChains:
    """Integration tests for calculation chains."""

    @pytest.mark.unit
    def test_full_spray_control_calculation(self):
        """Test full spray control calculation chain."""
        # Input conditions
        steam_flow = 20.0  # kg/s
        steam_temp_in = 450.0  # C
        steam_temp_target = 400.0  # C
        spray_water_temp = 100.0  # C
        pressure = 40.0  # bar
        max_spray = 5.0  # kg/s

        # Step 1: Calculate saturation temperature
        t_sat = calculate_saturation_temperature(pressure)
        assert t_sat > 0

        # Step 2: Calculate superheat
        superheat = calculate_superheat(steam_temp_in, pressure)
        assert superheat > 0

        # Step 3: Calculate required spray flow
        spray_flow, energy = calculate_spray_water_flow(
            steam_flow, steam_temp_in, steam_temp_target,
            spray_water_temp, pressure
        )
        assert spray_flow > 0

        # Step 4: Calculate valve position
        valve_pos = calculate_valve_position(spray_flow, max_spray)
        assert 0 <= valve_pos <= 100

        # Step 5: Calculate energy loss
        h_in = calculate_steam_enthalpy(steam_temp_in, pressure)
        h_out = calculate_steam_enthalpy(steam_temp_target, pressure)
        energy_loss = calculate_spray_energy_loss(spray_flow, h_in - h_out)
        assert energy_loss >= 0

        # Step 6: Generate provenance hash
        inputs = {
            "steam_temp": steam_temp_in,
            "target_temp": steam_temp_target,
            "pressure": pressure
        }
        outputs = {
            "spray_flow": spray_flow,
            "valve_position": valve_pos
        }
        calc_hash = generate_calculation_hash(inputs, outputs)
        assert len(calc_hash) == 64

    @pytest.mark.unit
    def test_calculation_chain_determinism(self):
        """Test calculation chain produces same results every time."""
        def run_calculation():
            steam_flow = 20.0
            steam_temp_in = 450.0
            steam_temp_target = 400.0
            spray_water_temp = 100.0
            pressure = 40.0

            t_sat = calculate_saturation_temperature(pressure)
            superheat = calculate_superheat(steam_temp_in, pressure)
            spray_flow, energy = calculate_spray_water_flow(
                steam_flow, steam_temp_in, steam_temp_target,
                spray_water_temp, pressure
            )
            valve_pos = calculate_valve_position(spray_flow, 5.0)

            return {
                "t_sat": t_sat,
                "superheat": superheat,
                "spray_flow": spray_flow,
                "valve_pos": valve_pos
            }

        results = [run_calculation() for _ in range(10)]

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_very_low_pressure(self):
        """Test calculations at very low pressure (near atmospheric)."""
        t_sat = calculate_saturation_temperature(1.01)
        assert t_sat == pytest.approx(100.0, abs=2.0)

    @pytest.mark.unit
    def test_very_high_pressure(self):
        """Test calculations at very high pressure (200 bar)."""
        t_sat = calculate_saturation_temperature(200.0)
        assert t_sat > 350  # Should be > 365C

    @pytest.mark.unit
    def test_tiny_temperature_difference(self):
        """Test spray calculation with tiny temperature difference."""
        spray_flow, _ = calculate_spray_water_flow(
            steam_flow_kg_s=20.0,
            steam_temp_in_c=400.1,
            steam_temp_target_c=400.0,  # Only 0.1C difference
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        # Should be very small but positive
        assert spray_flow >= 0
        assert spray_flow < 0.1

    @pytest.mark.unit
    def test_large_temperature_difference(self):
        """Test spray calculation with large temperature difference."""
        spray_flow, _ = calculate_spray_water_flow(
            steam_flow_kg_s=20.0,
            steam_temp_in_c=600.0,
            steam_temp_target_c=300.0,  # 300C difference
            spray_water_temp_c=50.0,
            steam_pressure_bar=40.0
        )

        # Should require significant spray
        assert spray_flow > 2.0

    @pytest.mark.unit
    def test_very_small_steam_flow(self):
        """Test calculations with very small steam flow."""
        spray_flow, _ = calculate_spray_water_flow(
            steam_flow_kg_s=0.1,  # Very small
            steam_temp_in_c=450.0,
            steam_temp_target_c=400.0,
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        assert spray_flow >= 0
        assert spray_flow < 0.1  # Should be proportionally small

    @pytest.mark.unit
    def test_zero_steam_flow(self):
        """Test calculations with zero steam flow."""
        spray_flow, _ = calculate_spray_water_flow(
            steam_flow_kg_s=0.0,
            steam_temp_in_c=450.0,
            steam_temp_target_c=400.0,
            spray_water_temp_c=100.0,
            steam_pressure_bar=40.0
        )

        assert spray_flow == 0.0
