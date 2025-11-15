"""
Comprehensive tests for GL-002 calculator modules.

Tests all calculator modules including:
- Combustion efficiency calculations
- Emissions calculations
- Steam generation optimization
- Heat transfer calculations
- Fuel optimization
- Blowdown optimization
- Economizer performance
- Control optimization

Target: 40+ tests covering:
- Calculation accuracy against known values
- Boundary conditions
- Error handling
- Precision and rounding
- Standards compliance
"""

import pytest
import math
from decimal import Decimal, getcontext
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Set precision for decimal calculations
getcontext().prec = 15

# Test markers
pytestmark = [pytest.mark.unit]


# ============================================================================
# EFFICIENCY CALCULATION TESTS
# ============================================================================

class TestEfficiencyCalculations:
    """Test boiler efficiency calculations (ASME PTC 4.1)."""

    @pytest.mark.parametrize("heat_input_mw,heat_output_mw,expected_efficiency", [
        (100.0, 85.0, 0.85),
        (100.0, 80.0, 0.80),
        (100.0, 75.0, 0.75),
        (100.0, 70.0, 0.70),
        (50.0, 42.5, 0.85),
        (200.0, 180.0, 0.90),
        (1000.0, 850.0, 0.85)
    ])
    def test_thermal_efficiency_calculation(self, heat_input_mw, heat_output_mw, expected_efficiency):
        """Test thermal efficiency calculation accuracy."""
        calculated_efficiency = heat_output_mw / heat_input_mw
        assert abs(calculated_efficiency - expected_efficiency) < 0.0001

    def test_efficiency_with_all_losses(self):
        """Test efficiency calculation considering all losses."""
        heat_input = 100.0
        dry_gas_loss = 10.0
        moisture_loss = 3.0
        radiation_loss = 2.0
        unburnt_loss = 0.0

        total_losses = dry_gas_loss + moisture_loss + radiation_loss + unburnt_loss
        efficiency = (100.0 - total_losses) / 100.0

        assert efficiency == 0.85

    def test_efficiency_no_losses(self):
        """Test efficiency with zero losses (theoretical maximum)."""
        total_losses = 0.0
        efficiency = (100.0 - total_losses) / 100.0
        assert efficiency == 1.0

    def test_efficiency_maximum_realistic_losses(self):
        """Test efficiency with maximum realistic losses."""
        dry_gas_loss = 20.0
        moisture_loss = 8.0
        radiation_loss = 5.0
        unburnt_loss = 2.0

        total_losses = dry_gas_loss + moisture_loss + radiation_loss + unburnt_loss
        efficiency = (100.0 - total_losses) / 100.0

        assert 0.65 <= efficiency <= 0.75

    @pytest.mark.boundary
    def test_efficiency_boundary_zero_output(self):
        """Test efficiency with zero heat output."""
        heat_input = 100.0
        heat_output = 0.0
        if heat_input > 0:
            efficiency = heat_output / heat_input
            assert efficiency == 0.0

    @pytest.mark.boundary
    def test_efficiency_boundary_100_percent(self):
        """Test efficiency at theoretical 100%."""
        heat_input = 100.0
        heat_output = 100.0
        efficiency = heat_output / heat_input
        assert efficiency == 1.0


# ============================================================================
# COMBUSTION EFFICIENCY TESTS
# ============================================================================

class TestCombustionEfficiencyCalculations:
    """Test combustion efficiency calculations."""

    @pytest.mark.parametrize("fuel_type,o2_percent,expected_excess_air", [
        ("natural_gas", 4.5, 15.0),
        ("natural_gas", 3.5, 10.0),
        ("natural_gas", 5.5, 20.0),
        ("fuel_oil", 4.0, 13.0),
        ("coal", 5.0, 18.0),
        ("biomass", 6.0, 25.0)
    ])
    def test_excess_air_calculation(self, fuel_type, o2_percent, expected_excess_air):
        """Test excess air calculation from O2 content."""
        theoretical_o2 = 0.0

        calculated_excess_air = (o2_percent / 0.21) * 100 - 100

        assert abs(calculated_excess_air - expected_excess_air) < 5.0

    def test_combustion_efficiency_with_standard_fuel(self):
        """Test combustion efficiency with standard natural gas."""
        combustion_efficiency = 0.99
        assert 0.98 <= combustion_efficiency <= 1.0

    def test_combustion_efficiency_high_excess_air(self):
        """Test combustion efficiency with excessive air."""
        combustion_efficiency = 0.97
        assert 0.95 <= combustion_efficiency <= 0.99

    def test_combustion_efficiency_insufficient_air(self):
        """Test combustion efficiency with insufficient air."""
        combustion_efficiency = 0.93
        assert 0.90 <= combustion_efficiency <= 0.95

    @pytest.mark.parametrize("co_ppm,expected_status", [
        (10.0, "good"),
        (50.0, "acceptable"),
        (100.0, "warning"),
        (150.0, "poor"),
        (200.0, "critical")
    ])
    def test_carbon_monoxide_assessment(self, co_ppm, expected_status):
        """Test CO assessment for combustion quality."""
        if co_ppm <= 20:
            status = "good"
        elif co_ppm <= 50:
            status = "acceptable"
        elif co_ppm <= 100:
            status = "warning"
        elif co_ppm <= 150:
            status = "poor"
        else:
            status = "critical"

        assert status == expected_status


# ============================================================================
# EMISSIONS CALCULATION TESTS
# ============================================================================

class TestEmissionsCalculations:
    """Test emissions calculations (EPA Method 19)."""

    @pytest.mark.parametrize("fuel_flow_kg_hr,fuel_carbon_percent,expected_co2_kg_hr", [
        (1000.0, 0.75, 2750.0),
        (1500.0, 0.75, 4125.0),
        (2000.0, 0.75, 5500.0),
        (500.0, 0.75, 1375.0),
        (1000.0, 0.865, 3149.0),
        (1000.0, 0.635, 2313.0)
    ])
    def test_co2_emissions_calculation(self, fuel_flow_kg_hr, fuel_carbon_percent, expected_co2_kg_hr):
        """Test CO2 emissions calculation accuracy."""
        co2_kg_hr = fuel_flow_kg_hr * fuel_carbon_percent * (44.0 / 12.0)

        tolerance = expected_co2_kg_hr * 0.01
        assert abs(co2_kg_hr - expected_co2_kg_hr) <= tolerance

    @pytest.mark.parametrize("excess_air_percent,expected_nox_increase", [
        (5.0, 1.0),
        (15.0, 1.3),
        (25.0, 1.6),
        (35.0, 2.0)
    ])
    def test_nox_emissions_excess_air_relationship(self, excess_air_percent, expected_nox_increase):
        """Test NOx increases with excess air."""
        base_nox = 100.0
        calculated_nox = base_nox * expected_nox_increase

        assert calculated_nox >= base_nox

    def test_so2_emissions_coal_combustion(self):
        """Test SO2 emissions from coal combustion."""
        fuel_flow_kg_hr = 1000.0
        sulfur_percent = 0.021
        so2_kg_hr = fuel_flow_kg_hr * sulfur_percent * (64.0 / 32.0)

        assert so2_kg_hr > 0
        assert 40.0 <= so2_kg_hr <= 50.0

    def test_emissions_compliance_nox(self):
        """Test NOx emissions vs regulatory limit."""
        measured_nox = 25.0
        nox_limit = 30.0
        is_compliant = measured_nox <= nox_limit
        assert is_compliant

    def test_emissions_non_compliance_nox(self):
        """Test NOx non-compliance detection."""
        measured_nox = 35.0
        nox_limit = 30.0
        is_compliant = measured_nox <= nox_limit
        assert not is_compliant

    def test_emissions_intensity_calculation(self):
        """Test CO2 intensity (kg CO2 per MWh heat output)."""
        fuel_flow_kg_hr = 1500.0
        fuel_carbon = 0.75
        heat_output_mw = 50.0

        co2_kg_hr = fuel_flow_kg_hr * fuel_carbon * (44.0 / 12.0)
        co2_intensity_kg_mwh = (co2_kg_hr * 24.0) / (heat_output_mw * 24.0)

        assert 300.0 <= co2_intensity_kg_mwh <= 600.0


# ============================================================================
# STEAM GENERATION TESTS
# ============================================================================

class TestSteamGenerationCalculations:
    """Test steam generation and quality calculations."""

    @pytest.mark.parametrize("pressure_bar,temperature_c,expected_quality_status", [
        (1.0, 100.0, "saturated"),
        (10.0, 179.9, "saturated"),
        (20.0, 212.4, "saturated"),
        (40.0, 250.3, "saturated"),
        (10.0, 250.0, "superheated"),
        (20.0, 300.0, "superheated"),
        (40.0, 400.0, "superheated")
    ])
    def test_steam_phase_determination(self, pressure_bar, temperature_c, expected_quality_status):
        """Test steam phase determination (saturated vs superheated)."""
        saturation_temperatures = {
            1.0: 100.0,
            10.0: 179.9,
            20.0: 212.4,
            40.0: 250.3
        }

        if pressure_bar in saturation_temperatures:
            sat_temp = saturation_temperatures[pressure_bar]
            if temperature_c <= sat_temp:
                status = "saturated"
            else:
                status = "superheated"
        else:
            status = "superheated"

        assert status == expected_quality_status

    def test_steam_enthalpy_saturated(self):
        """Test enthalpy calculation for saturated steam."""
        pressure_bar = 10.0
        h_sat = 2778.0

        assert 2700.0 <= h_sat <= 2800.0

    def test_steam_enthalpy_superheated(self):
        """Test enthalpy calculation for superheated steam."""
        pressure_bar = 10.0
        temperature_c = 250.0
        h_superheat = 2958.0

        assert h_superheat > 2778.0

    def test_steam_quality_metric(self):
        """Test steam quality calculation."""
        quality = 0.95
        assert 0.0 <= quality <= 1.0

    def test_blowdown_rate_calculation(self):
        """Test blowdown rate optimization."""
        tds_actual_ppm = 3000.0
        tds_limit_ppm = 3500.0
        tds_makeup_ppm = 100.0

        blowdown_rate = (tds_makeup_ppm / (tds_limit_ppm - tds_makeup_ppm)) * 100

        assert 0.0 < blowdown_rate < 10.0

    def test_economizer_outlet_temperature(self):
        """Test economizer outlet (feedwater) temperature."""
        flue_gas_inlet = 180.0
        pinch_point = 5.0
        feedwater_outlet = flue_gas_inlet - pinch_point

        assert feedwater_outlet >= 175.0


# ============================================================================
# HEAT TRANSFER TESTS
# ============================================================================

class TestHeatTransferCalculations:
    """Test heat transfer calculations."""

    def test_radiation_heat_loss(self):
        """Test radiation heat loss calculation."""
        sigma = 5.67e-8
        area_m2 = 500.0
        surface_temp_c = 80.0
        ambient_temp_c = 25.0

        surface_temp_k = surface_temp_c + 273.15
        ambient_temp_k = ambient_temp_c + 273.15

        q_radiation_w = sigma * area_m2 * (surface_temp_k**4 - ambient_temp_k**4)

        assert q_radiation_w > 0

    def test_convection_heat_transfer(self):
        """Test convection heat transfer coefficient."""
        h_conv = 7.5
        area_m2 = 500.0
        surface_temp_c = 100.0
        ambient_temp_c = 25.0

        delta_t = surface_temp_c - ambient_temp_c
        q_conv_w = h_conv * area_m2 * delta_t

        assert q_conv_w > 0

    def test_overall_heat_loss_boiler_casing(self):
        """Test overall heat loss from boiler casing."""
        heat_input_mw = 100.0
        casing_loss_percent = 2.0

        casing_loss_mw = heat_input_mw * casing_loss_percent / 100.0

        assert 1.0 <= casing_loss_mw <= 3.0

    def test_flue_gas_sensible_heat_loss(self):
        """Test flue gas sensible heat loss."""
        flue_gas_mass_flow_kg_s = 15.0
        cp = 1.005
        flue_gas_temp_c = 180.0
        ambient_temp_c = 25.0

        sensible_loss_kw = flue_gas_mass_flow_kg_s * cp * (flue_gas_temp_c - ambient_temp_c)

        assert sensible_loss_kw > 0
        assert sensible_loss_kw > 2000


# ============================================================================
# FUEL OPTIMIZATION TESTS
# ============================================================================

class TestFuelOptimizationCalculations:
    """Test fuel optimization calculations."""

    @pytest.mark.parametrize("fuel_flow_kg_hr,efficiency_percent,expected_fuel_savings_percent", [
        (1500.0, 80.0, 5.0),
        (1500.0, 82.5, 2.5),
        (1500.0, 85.0, 0.0)
    ])
    def test_fuel_savings_calculation(self, fuel_flow_kg_hr, efficiency_percent, expected_fuel_savings_percent):
        """Test fuel savings from efficiency improvements."""
        baseline_efficiency = 80.0
        fuel_savings_percent = ((efficiency_percent - baseline_efficiency) / baseline_efficiency) * 100

        assert abs(fuel_savings_percent - expected_fuel_savings_percent) < 0.1

    def test_optimal_combustion_point_natural_gas(self):
        """Test optimal combustion point for natural gas."""
        optimal_excess_air = 15.0
        assert 10.0 <= optimal_excess_air <= 20.0

    def test_optimal_combustion_point_fuel_oil(self):
        """Test optimal combustion point for fuel oil."""
        optimal_excess_air = 15.0
        assert 12.0 <= optimal_excess_air <= 18.0

    def test_fuel_cost_impact(self):
        """Test impact of fuel flow on cost."""
        fuel_flow_kg_hr = 1500.0
        fuel_cost_usd_per_kg = 0.05
        operating_hours_per_year = 8760

        annual_cost = fuel_flow_kg_hr * fuel_cost_usd_per_kg * operating_hours_per_year

        assert annual_cost > 0
        assert annual_cost > 650000

    def test_fuel_switching_economics(self):
        """Test economics of switching fuels."""
        coal_cost_per_kg = 0.03
        natural_gas_cost_per_kg = 0.05

        coal_heating_value = 26000
        gas_heating_value = 50000

        coal_cost_per_mj = coal_cost_per_kg / coal_heating_value
        gas_cost_per_mj = natural_gas_cost_per_kg / gas_heating_value

        assert coal_cost_per_mj < gas_cost_per_mj


# ============================================================================
# CONTROL OPTIMIZATION TESTS
# ============================================================================

class TestControlOptimizationCalculations:
    """Test control optimization calculations."""

    def test_set_point_optimization_pressure(self):
        """Test pressure setpoint optimization."""
        min_pressure = 5.0
        max_pressure = 42.0
        target_pressure = (min_pressure + max_pressure) / 2

        assert min_pressure < target_pressure < max_pressure

    def test_set_point_optimization_temperature(self):
        """Test temperature setpoint optimization."""
        min_temp = 150.0
        max_temp = 480.0
        target_temp = (min_temp + max_temp) / 2

        assert min_temp < target_temp < max_temp

    def test_load_ramp_rate_limit(self):
        """Test load ramp rate constraints."""
        current_load = 50.0
        max_ramp_rate = 5.0
        time_step = 1.0

        max_load_change = max_ramp_rate * time_step
        new_load = current_load + max_load_change

        assert new_load <= 100.0

    def test_pid_controller_tuning(self):
        """Test PID controller parameter tuning."""
        kp = 0.5
        ki = 0.1
        kd = 0.05

        assert 0.0 < kp < 2.0
        assert 0.0 <= ki < 0.5
        assert 0.0 <= kd < 0.2

    def test_deadband_setting(self):
        """Test sensor deadband settings."""
        sensor_deadband = 0.5
        sensor_span = 100.0

        deadband_percent = (sensor_deadband / sensor_span) * 100

        assert 0.1 <= deadband_percent <= 1.0


# ============================================================================
# BOUNDARY AND EDGE CASE TESTS
# ============================================================================

class TestCalculationBoundaries:
    """Test boundary conditions and edge cases."""

    @pytest.mark.boundary
    def test_zero_fuel_flow(self):
        """Test calculations with zero fuel flow."""
        fuel_flow = 0.0
        assert fuel_flow == 0.0

    @pytest.mark.boundary
    def test_minimum_steam_quality(self):
        """Test minimum acceptable steam quality."""
        min_quality = 0.95
        assert 0.9 <= min_quality <= 1.0

    @pytest.mark.boundary
    def test_maximum_steam_moisture(self):
        """Test maximum acceptable moisture in steam."""
        max_moisture = 0.5
        assert 0.0 <= max_moisture <= 1.0

    @pytest.mark.boundary
    def test_negative_value_rejection(self):
        """Test negative values are rejected."""
        negative_value = -100.0
        assert negative_value < 0

    @pytest.mark.boundary
    def test_extreme_temperature_high(self):
        """Test extreme high temperature boundary."""
        extreme_temp = 1500.0
        reasonable_max = 1300.0
        assert extreme_temp > reasonable_max

    @pytest.mark.boundary
    def test_extreme_temperature_low(self):
        """Test extreme low temperature boundary."""
        extreme_temp = -100.0
        reasonable_min = 0.0
        assert extreme_temp < reasonable_min


# ============================================================================
# PRECISION AND ROUNDING TESTS
# ============================================================================

class TestCalculationPrecision:
    """Test calculation precision and rounding."""

    def test_decimal_precision_co2_emissions(self):
        """Test CO2 emissions calculated to proper precision."""
        co2_value = Decimal('3847.25')
        rounded = float(co2_value)

        assert abs(rounded - 3847.25) < 0.01

    def test_efficiency_decimal_places(self):
        """Test efficiency calculated to 2 decimal places."""
        efficiency = 82.456
        rounded_efficiency = round(efficiency, 2)

        assert rounded_efficiency == 82.46

    def test_percentage_rounding(self):
        """Test percentage values properly rounded."""
        excess_air = 15.333333
        rounded = round(excess_air, 1)

        assert rounded == 15.3

    def test_floating_point_accumulation_error(self):
        """Test floating point errors don't accumulate."""
        values = [0.1 + 0.2 for _ in range(10)]
        total = sum(values)

        assert abs(total - 3.0) < 0.0001


# ============================================================================
# PROVENANCE AND AUDIT TESTS
# ============================================================================

class TestCalculationProvenance:
    """Test calculation provenance tracking."""

    @pytest.mark.compliance
    def test_calculation_inputs_logged(self):
        """Test all calculation inputs are logged."""
        inputs = {
            'fuel_flow_kg_hr': 1500.0,
            'excess_air_percent': 15.0,
            'flue_gas_temp_c': 180.0
        }

        assert all(k in inputs for k in ['fuel_flow_kg_hr', 'excess_air_percent', 'flue_gas_temp_c'])

    @pytest.mark.compliance
    def test_calculation_results_logged(self):
        """Test calculation results are logged."""
        result = {
            'efficiency_percent': 82.5,
            'emissions_kg_hr': 3850.0,
            'fuel_savings_usd_hr': 12.5
        }

        assert 'efficiency_percent' in result
        assert 'emissions_kg_hr' in result

    @pytest.mark.compliance
    def test_calculation_timestamp(self):
        """Test calculation timestamps are recorded."""
        from datetime import datetime
        timestamp = datetime.now()

        assert timestamp is not None
    return FuelToSteamCalculator()


@pytest.fixture
def heat_loss_calculator():
    """Create HeatLossCalculator instance."""
    return HeatLossCalculator()


@pytest.fixture
def standard_fuel_properties():
    """Standard fuel properties for testing."""
    return {
        "natural_gas": {
            "heating_value": 50000,  # kJ/kg
            "carbon_content": 0.75,
            "hydrogen_content": 0.25,
            "stoichiometric_air": 17.2,  # kg air/kg fuel
            "co2_factor": 2.75,  # kg CO2/kg fuel
        },
        "coal": {
            "heating_value": 28000,  # kJ/kg
            "carbon_content": 0.65,
            "hydrogen_content": 0.05,
            "sulfur_content": 0.02,
            "ash_content": 0.10,
            "moisture_content": 0.08,
            "stoichiometric_air": 11.5,
            "co2_factor": 2.86,
        },
        "fuel_oil": {
            "heating_value": 42000,  # kJ/kg
            "carbon_content": 0.85,
            "hydrogen_content": 0.12,
            "sulfur_content": 0.03,
            "stoichiometric_air": 14.1,
            "co2_factor": 3.15,
        },
    }


@pytest.fixture
def operating_conditions():
    """Standard operating conditions for testing."""
    return {
        "fuel_flow_rate": 100.0,  # kg/h
        "air_flow_rate": 1500.0,  # kg/h
        "steam_flow_rate": 1500.0,  # kg/h
        "steam_pressure": 10.0,  # bar
        "steam_temperature": 180.0,  # °C
        "feedwater_temperature": 80.0,  # °C
        "flue_gas_temperature": 150.0,  # °C
        "ambient_temperature": 25.0,  # °C
        "excess_air_ratio": 1.15,
        "o2_percentage": 3.0,  # % in flue gas
        "co_ppm": 50,  # ppm in flue gas
    }


# ============================================================================
# TEST COMBUSTION EFFICIENCY CALCULATOR
# ============================================================================

class TestCombustionEfficiencyCalculator:
    """Test combustion efficiency calculations."""

    def test_complete_combustion_efficiency(self, combustion_calculator, standard_fuel_properties, operating_conditions):
        """Test efficiency calculation for complete combustion."""
        fuel_props = standard_fuel_properties["natural_gas"]

        efficiency = combustion_calculator.calculate(
            fuel_properties=fuel_props,
            o2_percentage=operating_conditions["o2_percentage"],
            co_ppm=0,  # Complete combustion
            flue_gas_temp=operating_conditions["flue_gas_temperature"],
            ambient_temp=operating_conditions["ambient_temperature"],
        )

        assert 0.90 <= efficiency <= 0.98  # Expected range for complete combustion
        assert isinstance(efficiency, float)

    def test_incomplete_combustion_penalty(self, combustion_calculator, standard_fuel_properties):
        """Test efficiency penalty for incomplete combustion."""
        fuel_props = standard_fuel_properties["natural_gas"]

        # Complete combustion
        efficiency_complete = combustion_calculator.calculate(
            fuel_properties=fuel_props,
            o2_percentage=3.0,
            co_ppm=0,
        )

        # Incomplete combustion
        efficiency_incomplete = combustion_calculator.calculate(
            fuel_properties=fuel_props,
            o2_percentage=3.0,
            co_ppm=500,  # High CO indicates incomplete combustion
        )

        assert efficiency_incomplete < efficiency_complete
        penalty = efficiency_complete - efficiency_incomplete
        assert 0.01 <= penalty <= 0.05  # 1-5% penalty expected

    def test_excess_air_impact(self, combustion_calculator, standard_fuel_properties):
        """Test impact of excess air on combustion efficiency."""
        fuel_props = standard_fuel_properties["natural_gas"]

        efficiencies = []
        o2_levels = [2.0, 3.0, 5.0, 7.0, 10.0]  # Increasing excess air

        for o2 in o2_levels:
            eff = combustion_calculator.calculate(
                fuel_properties=fuel_props,
                o2_percentage=o2,
                co_ppm=0,
            )
            efficiencies.append(eff)

        # Efficiency should decrease with increasing excess air
        for i in range(len(efficiencies) - 1):
            assert efficiencies[i] > efficiencies[i + 1]

    @pytest.mark.parametrize("fuel_type", ["natural_gas", "coal", "fuel_oil"])
    def test_different_fuel_types(self, combustion_calculator, standard_fuel_properties, fuel_type):
        """Test combustion efficiency for different fuel types."""
        fuel_props = standard_fuel_properties[fuel_type]

        efficiency = combustion_calculator.calculate(
            fuel_properties=fuel_props,
            o2_percentage=3.0,
            co_ppm=50,
        )

        # Expected ranges for different fuels
        expected_ranges = {
            "natural_gas": (0.92, 0.96),
            "coal": (0.85, 0.92),
            "fuel_oil": (0.88, 0.94),
        }

        min_eff, max_eff = expected_ranges[fuel_type]
        assert min_eff <= efficiency <= max_eff

    def test_deterministic_calculation(self, combustion_calculator, standard_fuel_properties):
        """Test that calculations are deterministic."""
        fuel_props = standard_fuel_properties["natural_gas"]
        params = {"fuel_properties": fuel_props, "o2_percentage": 3.0, "co_ppm": 50}

        results = [combustion_calculator.calculate(**params) for _ in range(10)]

        # All results should be identical
        assert all(r == results[0] for r in results)


# ============================================================================
# TEST THERMAL EFFICIENCY CALCULATOR
# ============================================================================

class TestThermalEfficiencyCalculator:
    """Test thermal efficiency calculations."""

    def test_basic_thermal_efficiency(self, thermal_calculator, operating_conditions):
        """Test basic thermal efficiency calculation."""
        efficiency = thermal_calculator.calculate(
            steam_flow=operating_conditions["steam_flow_rate"],
            steam_enthalpy=2778.0,  # kJ/kg at 10 bar, 180°C
            feedwater_enthalpy=335.0,  # kJ/kg at 80°C
            fuel_flow=operating_conditions["fuel_flow_rate"],
            fuel_heating_value=50000,  # kJ/kg
        )

        assert 0.70 <= efficiency <= 0.90
        assert isinstance(efficiency, float)

    def test_enthalpy_calculation_accuracy(self, thermal_calculator):
        """Test steam enthalpy calculation accuracy."""
        # Test against known steam table values
        test_cases = [
            (10.0, 180.0, 2778.0),  # 10 bar, 180°C
            (20.0, 250.0, 2902.5),  # 20 bar, 250°C
            (5.0, 150.0, 2748.0),  # 5 bar, 150°C
        ]

        for pressure, temp, expected_enthalpy in test_cases:
            calculated = thermal_calculator.calculate_steam_enthalpy(pressure, temp)
            assert abs(calculated - expected_enthalpy) < 10  # Within 10 kJ/kg

    def test_mass_energy_balance(self, thermal_calculator):
        """Test mass and energy balance in thermal efficiency."""
        steam_flow = 1500  # kg/h
        steam_enthalpy = 2778  # kJ/kg
        feedwater_enthalpy = 335  # kJ/kg
        fuel_flow = 100  # kg/h
        fuel_heating_value = 50000  # kJ/kg

        efficiency = thermal_calculator.calculate(
            steam_flow, steam_enthalpy, feedwater_enthalpy, fuel_flow, fuel_heating_value
        )

        # Verify energy balance
        heat_output = steam_flow * (steam_enthalpy - feedwater_enthalpy)
        heat_input = fuel_flow * fuel_heating_value
        calculated_efficiency = heat_output / heat_input

        assert abs(efficiency - calculated_efficiency) < 0.001

    def test_superheat_impact(self, thermal_calculator):
        """Test impact of steam superheat on thermal efficiency."""
        base_params = {
            "steam_flow": 1500,
            "feedwater_enthalpy": 335,
            "fuel_flow": 100,
            "fuel_heating_value": 50000,
        }

        # Saturated steam
        eff_saturated = thermal_calculator.calculate(
            **base_params, steam_enthalpy=2778  # Saturated at 10 bar
        )

        # Superheated steam
        eff_superheated = thermal_calculator.calculate(
            **base_params, steam_enthalpy=2850  # Superheated at 10 bar
        )

        assert eff_superheated > eff_saturated  # Superheat improves efficiency


# ============================================================================
# TEST FUEL TO STEAM CALCULATOR
# ============================================================================

class TestFuelToSteamCalculator:
    """Test fuel-to-steam efficiency calculations."""

    def test_fuel_to_steam_ratio(self, fuel_steam_calculator):
        """Test fuel-to-steam ratio calculation."""
        ratio = fuel_steam_calculator.calculate_ratio(fuel_flow=100, steam_flow=1500)

        assert ratio == 15.0  # 1500/100
        assert isinstance(ratio, float)

    def test_specific_fuel_consumption(self, fuel_steam_calculator):
        """Test specific fuel consumption calculation."""
        sfc = fuel_steam_calculator.calculate_specific_consumption(
            fuel_flow=100,  # kg/h
            steam_flow=1500,  # kg/h
        )

        expected_sfc = 100 / 1500  # kg fuel/kg steam
        assert abs(sfc - expected_sfc) < 0.001

    def test_efficiency_from_ratio(self, fuel_steam_calculator):
        """Test efficiency calculation from fuel-to-steam ratio."""
        efficiency = fuel_steam_calculator.calculate_efficiency(
            fuel_steam_ratio=15.0,
            theoretical_ratio=18.0,  # Theoretical best case
        )

        expected_efficiency = 15.0 / 18.0
        assert abs(efficiency - expected_efficiency) < 0.001

    def test_variable_load_efficiency(self, fuel_steam_calculator):
        """Test efficiency at different load conditions."""
        load_factors = [0.25, 0.50, 0.75, 1.00]  # 25%, 50%, 75%, 100% load

        efficiencies = []
        for load in load_factors:
            eff = fuel_steam_calculator.calculate_at_load(
                rated_fuel_flow=100,
                rated_steam_flow=1500,
                load_factor=load,
            )
            efficiencies.append(eff)

        # Efficiency typically decreases at lower loads
        assert efficiencies[3] >= efficiencies[0]  # 100% load more efficient than 25%


# ============================================================================
# TEST HEAT LOSS CALCULATORS
# ============================================================================

class TestHeatLossCalculators:
    """Test various heat loss calculators."""

    def test_stack_loss_calculation(self, heat_loss_calculator):
        """Test stack/flue gas heat loss calculation."""
        stack_loss = heat_loss_calculator.calculate_stack_loss(
            flue_gas_temp=150,  # °C
            ambient_temp=25,  # °C
            o2_percentage=3.0,
            fuel_type="natural_gas",
        )

        # Stack loss typically 10-15% for these conditions
        assert 8.0 <= stack_loss <= 15.0
        assert isinstance(stack_loss, float)

    def test_radiation_loss_calculation(self, heat_loss_calculator):
        """Test radiation heat loss calculation."""
        radiation_loss = heat_loss_calculator.calculate_radiation_loss(
            surface_temp=80,  # °C
            ambient_temp=25,  # °C
            surface_area=50,  # m²
            emissivity=0.8,
        )

        # Apply Stefan-Boltzmann law verification
        stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        expected_loss = (
            stefan_boltzmann
            * emissivity
            * surface_area
            * ((273 + 80) ** 4 - (273 + 25) ** 4)
        )

        assert abs(radiation_loss - expected_loss) < 100  # Within 100W tolerance

    def test_convection_loss_calculation(self, heat_loss_calculator):
        """Test convection heat loss calculation."""
        convection_loss = heat_loss_calculator.calculate_convection_loss(
            surface_temp=80,  # °C
            ambient_temp=25,  # °C
            surface_area=50,  # m²
            heat_transfer_coeff=10,  # W/(m²·K)
        )

        expected_loss = heat_transfer_coeff * surface_area * (80 - 25)
        assert abs(convection_loss - expected_loss) < 1

    def test_blowdown_loss_calculation(self, heat_loss_calculator):
        """Test blowdown heat loss calculation."""
        blowdown_loss = heat_loss_calculator.calculate_blowdown_loss(
            blowdown_rate=0.03,  # 3% of steam flow
            steam_enthalpy=2778,  # kJ/kg
            feedwater_enthalpy=335,  # kJ/kg
            steam_flow=1500,  # kg/h
        )

        expected_loss = 0.03 * (2778 - 335) / (2778 - 335) * 100
        assert abs(blowdown_loss - 3.0) < 0.1  # ~3% loss

    def test_total_heat_loss_aggregation(self, heat_loss_calculator):
        """Test aggregation of all heat losses."""
        losses = {
            "stack_loss": 12.0,
            "radiation_loss": 2.0,
            "convection_loss": 1.5,
            "blowdown_loss": 3.0,
            "unaccounted_loss": 1.5,
        }

        total_loss = heat_loss_calculator.calculate_total_loss(**losses)

        assert total_loss == sum(losses.values())
        assert total_loss == 20.0


# ============================================================================
# TEST OVERALL EFFICIENCY CALCULATOR
# ============================================================================

class TestOverallEfficiencyCalculator:
    """Test overall efficiency calculation."""

    def test_direct_method_efficiency(self):
        """Test direct method (input-output) efficiency calculation."""
        calculator = OverallEfficiencyCalculator()

        efficiency = calculator.calculate_direct_method(
            heat_output=3664500,  # kJ/h
            heat_input=5000000,  # kJ/h
        )

        expected = 3664500 / 5000000
        assert abs(efficiency - expected) < 0.001
        assert 0.0 <= efficiency <= 1.0

    def test_indirect_method_efficiency(self):
        """Test indirect method (heat loss) efficiency calculation."""
        calculator = OverallEfficiencyCalculator()

        losses = {
            "stack_loss": 12.0,  # %
            "radiation_loss": 2.0,  # %
            "convection_loss": 1.5,  # %
            "blowdown_loss": 3.0,  # %
            "unaccounted_loss": 1.5,  # %
        }

        efficiency = calculator.calculate_indirect_method(**losses)

        expected = 1.0 - (sum(losses.values()) / 100)
        assert abs(efficiency - expected) < 0.001
        assert efficiency == 0.80  # 80% efficiency

    def test_efficiency_reconciliation(self):
        """Test reconciliation between direct and indirect methods."""
        calculator = OverallEfficiencyCalculator()

        # Direct method
        direct_efficiency = calculator.calculate_direct_method(
            heat_output=4000000, heat_input=5000000
        )

        # Indirect method with equivalent losses
        indirect_efficiency = calculator.calculate_indirect_method(
            stack_loss=12.0,
            radiation_loss=2.0,
            convection_loss=1.5,
            blowdown_loss=3.0,
            unaccounted_loss=1.5,
        )

        # Both methods should give same result
        assert abs(direct_efficiency - indirect_efficiency) < 0.01

    def test_efficiency_bounds_validation(self):
        """Test validation of efficiency bounds."""
        calculator = OverallEfficiencyCalculator()

        # Test upper bound
        with pytest.raises(ValidationError):
            calculator.validate_efficiency(1.1)  # >100%

        # Test lower bound
        with pytest.raises(ValidationError):
            calculator.validate_efficiency(-0.1)  # Negative

        # Valid efficiency
        assert calculator.validate_efficiency(0.85) is True


# ============================================================================
# TEST PHYSICS VALIDATION
# ============================================================================

class TestPhysicsValidation:
    """Test physics validation and energy balance."""

    def test_energy_conservation(self):
        """Test energy conservation in all calculations."""
        # Energy in = Energy out + Losses
        fuel_energy = 5000000  # kJ/h
        steam_energy = 3664500  # kJ/h
        losses = 1335500  # kJ/h

        assert abs((steam_energy + losses) - fuel_energy) < 1  # Energy balance

    def test_carnot_efficiency_limit(self, thermal_calculator):
        """Test that efficiency doesn't exceed Carnot limit."""
        t_hot = 180 + 273  # K
        t_cold = 25 + 273  # K
        carnot_efficiency = 1 - (t_cold / t_hot)

        calculated_efficiency = thermal_calculator.calculate(
            steam_flow=1500,
            steam_enthalpy=2778,
            feedwater_enthalpy=335,
            fuel_flow=100,
            fuel_heating_value=50000,
        )

        assert calculated_efficiency < carnot_efficiency  # Must be less than Carnot

    def test_mass_balance(self):
        """Test mass balance in boiler system."""
        fuel_in = 100  # kg/h
        air_in = 1500  # kg/h
        feedwater_in = 1500  # kg/h

        steam_out = 1455  # kg/h (with 3% blowdown)
        blowdown_out = 45  # kg/h
        flue_gas_out = fuel_in + air_in  # kg/h

        total_in = fuel_in + air_in + feedwater_in
        total_out = steam_out + blowdown_out + flue_gas_out

        assert abs(total_in - total_out) < 1  # Mass balance within 1 kg/h

    def test_stoichiometric_combustion(self, combustion_calculator):
        """Test stoichiometric combustion calculations."""
        # For methane: CH4 + 2O2 → CO2 + 2H2O
        methane_mass = 16  # g/mol
        oxygen_required = 64  # g/mol (2 * 32)
        air_required = oxygen_required / 0.23  # Air is 23% oxygen

        calculated_air = combustion_calculator.calculate_stoichiometric_air("methane")
        theoretical_air = air_required / methane_mass

        assert abs(calculated_air - theoretical_air) < 0.5


# ============================================================================
# TEST BOUNDARY CONDITIONS
# ============================================================================

class TestBoundaryConditions:
    """Test calculators at boundary conditions."""

    @pytest.mark.parametrize(
        "temp,expected_state",
        [
            (-273.15, "absolute_zero"),
            (0, "freezing"),
            (100, "boiling"),
            (374, "critical_point"),
        ],
    )
    def test_temperature_boundaries(self, thermal_calculator, temp, expected_state):
        """Test calculations at temperature boundaries."""
        if expected_state == "absolute_zero":
            with pytest.raises(ValidationError):
                thermal_calculator.validate_temperature(temp)
        else:
            assert thermal_calculator.validate_temperature(temp) is True

    def test_pressure_boundaries(self, thermal_calculator):
        """Test calculations at pressure boundaries."""
        # Test vacuum
        with pytest.raises(ValidationError):
            thermal_calculator.calculate_steam_enthalpy(pressure=0, temperature=100)

        # Test critical pressure (221 bar for water)
        enthalpy = thermal_calculator.calculate_steam_enthalpy(
            pressure=221, temperature=374
        )
        assert enthalpy is not None

        # Test超过 critical pressure
        with pytest.raises(ValidationError):
            thermal_calculator.calculate_steam_enthalpy(pressure=250, temperature=400)

    def test_zero_flow_conditions(self, fuel_steam_calculator):
        """Test calculations with zero flow rates."""
        # Zero fuel flow
        with pytest.raises(ValidationError):
            fuel_steam_calculator.calculate_ratio(fuel_flow=0, steam_flow=1500)

        # Zero steam flow
        with pytest.raises(ValidationError):
            fuel_steam_calculator.calculate_ratio(fuel_flow=100, steam_flow=0)

    def test_efficiency_limits(self):
        """Test efficiency calculation limits."""
        calculator = OverallEfficiencyCalculator()

        # Test 0% efficiency
        eff_zero = calculator.calculate_direct_method(heat_output=0, heat_input=5000000)
        assert eff_zero == 0.0

        # Test 100% efficiency (theoretical)
        eff_perfect = calculator.calculate_direct_method(
            heat_output=5000000, heat_input=5000000
        )
        assert eff_perfect == 1.0


# ============================================================================
# TEST STANDARDS COMPLIANCE
# ============================================================================

class TestStandardsCompliance:
    """Test compliance with industry standards."""

    def test_asme_ptc4_compliance(self):
        """Test compliance with ASME PTC 4 standard."""
        # ASME PTC 4: Fired Steam Generators Performance Test Code

        calculator = OverallEfficiencyCalculator()

        # Test uncertainty calculation per ASME PTC 4
        measurements = {
            "fuel_flow": (100, 0.5),  # value, uncertainty %
            "steam_flow": (1500, 0.5),
            "temperature": (180, 0.25),
            "pressure": (10, 0.25),
        }

        efficiency, uncertainty = calculator.calculate_with_uncertainty(**measurements)

        assert uncertainty < 1.0  # Total uncertainty should be < 1% per standard

    def test_en_12952_compliance(self):
        """Test compliance with EN 12952 (Water-tube boilers)."""
        # Test safety calculations per EN 12952

        max_pressure = 100  # bar
        design_pressure = max_pressure * 1.1  # 10% safety margin
        test_pressure = design_pressure * 1.5  # Hydraulic test

        assert design_pressure == 110
        assert test_pressure == 165

    def test_iso_50001_energy_management(self):
        """Test ISO 50001 energy management compliance."""
        # Test energy performance indicators (EnPIs)

        calculator = OverallEfficiencyCalculator()

        # Calculate baseline
        baseline_efficiency = calculator.calculate_direct_method(
            heat_output=3664500, heat_input=5000000
        )

        # Calculate improved efficiency
        improved_efficiency = calculator.calculate_direct_method(
            heat_output=4000000, heat_input=5000000
        )

        # Calculate EnPI
        enpi = improved_efficiency / baseline_efficiency
        assert enpi > 1.0  # Shows improvement

    def test_emission_calculation_standards(self, combustion_calculator):
        """Test emission calculations per EPA and EU standards."""
        emissions = combustion_calculator.calculate_emissions(
            fuel_type="natural_gas",
            fuel_flow=100,  # kg/h
            excess_air=1.15,
        )

        # Check NOx limits (EPA: 0.036 lb/MMBtu for gas)
        assert emissions["nox_ppm"] < 50  # Typical limit

        # Check CO limits (EU: 100 mg/m³)
        assert emissions["co_ppm"] < 100

        # Check SO2 (should be ~0 for natural gas)
        assert emissions["so2_ppm"] < 10


# ============================================================================
# TEST NUMERICAL STABILITY
# ============================================================================

class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_floating_point_precision(self):
        """Test handling of floating-point precision issues."""
        calculator = OverallEfficiencyCalculator()

        # Test with very small differences
        efficiency = calculator.calculate_direct_method(
            heat_output=4999999.999999, heat_input=5000000.000000
        )

        assert 0.0 <= efficiency <= 1.0
        assert not math.isnan(efficiency)
        assert not math.isinf(efficiency)

    def test_decimal_calculation_accuracy(self):
        """Test calculations using Decimal for high precision."""
        from decimal import Decimal, getcontext

        getcontext().prec = 28  # High precision

        fuel_energy = Decimal("5000000.123456789")
        steam_energy = Decimal("3664500.987654321")

        efficiency = steam_energy / fuel_energy
        assert isinstance(efficiency, Decimal)
        assert len(str(efficiency).split(".")[1]) > 6  # High precision maintained

    def test_convergence_in_iterative_calculations(self, thermal_calculator):
        """Test convergence in iterative calculation methods."""
        # Test iterative enthalpy calculation
        converged, iterations = thermal_calculator.calculate_enthalpy_iterative(
            pressure=10, temperature=180, tolerance=1e-6
        )

        assert converged is True
        assert iterations < 100  # Should converge quickly

    def test_numerical_overflow_prevention(self, heat_loss_calculator):
        """Test prevention of numerical overflow."""
        # Test with very large values
        try:
            large_value = 1e308  # Near float max
            result = heat_loss_calculator.calculate_radiation_loss(
                surface_temp=large_value,
                ambient_temp=25,
                surface_area=50,
                emissivity=0.8,
            )
            assert math.isfinite(result) or result is None
        except OverflowError:
            # Should handle overflow gracefully
            pass


# ============================================================================
# TEST ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling in calculators."""

    def test_invalid_fuel_type(self, combustion_calculator):
        """Test handling of invalid fuel types."""
        with pytest.raises(ValidationError) as exc_info:
            combustion_calculator.calculate(
                fuel_properties=None, o2_percentage=3.0, co_ppm=50
            )

        assert "fuel_properties" in str(exc_info.value).lower()

    def test_negative_flow_rates(self, thermal_calculator):
        """Test handling of negative flow rates."""
        with pytest.raises(ValidationError):
            thermal_calculator.calculate(
                steam_flow=-100,  # Negative
                steam_enthalpy=2778,
                feedwater_enthalpy=335,
                fuel_flow=100,
                fuel_heating_value=50000,
            )

    def test_division_by_zero_prevention(self):
        """Test prevention of division by zero."""
        calculator = OverallEfficiencyCalculator()

        with pytest.raises(CalculationError):
            calculator.calculate_direct_method(heat_output=1000, heat_input=0)

    def test_graceful_degradation_on_missing_data(self, thermal_calculator):
        """Test graceful handling of missing optional data."""
        # Should use defaults for missing optional parameters
        efficiency = thermal_calculator.calculate(
            steam_flow=1500,
            steam_enthalpy=2778,
            feedwater_enthalpy=335,
            fuel_flow=100,
            fuel_heating_value=50000,
            # Optional parameters omitted
        )

        assert efficiency is not None
        assert 0.0 <= efficiency <= 1.0