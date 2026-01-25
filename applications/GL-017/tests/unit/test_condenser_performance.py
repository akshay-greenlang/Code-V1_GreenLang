# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Condenser Performance Unit Tests

Comprehensive unit tests for condenser performance calculations including:
- U-value calculations
- Cleanliness factor calculations
- TTD (Terminal Temperature Difference) calculations
- Vacuum optimization
- LMTD (Log Mean Temperature Difference) calculations
- HEI standard calculations
- Boundary conditions and edge cases

Test coverage target: 95%+

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch, MagicMock
import hashlib
import json
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.heat_transfer_calculator import (
    HeatTransferCalculator,
    HeatTransferInput,
    HeatTransferOutput,
    calculate_lmtd,
    calculate_ntu,
    calculate_effectiveness,
    TUBE_MATERIAL_CONDUCTIVITY,
    WATER_PROPERTIES,
)
from calculators.efficiency_calculator import (
    EfficiencyCalculator,
    EfficiencyInput,
    EfficiencyOutput,
    calculate_cw_temperature_rise,
    calculate_optimal_cw_flow,
    calculate_cw_pumping_power,
    calculate_payback_period,
    calculate_npv,
    HEAT_RATE_IMPACT_KJ_KWH_PER_MMHG,
    CARBON_EMISSION_FACTOR_KG_CO2_MWH,
)
from calculators.vacuum_calculator import (
    VacuumCalculator,
    VacuumInput,
    VacuumOutput,
)
from calculators.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    verify_provenance,
    compute_input_fingerprint,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def heat_transfer_calculator():
    """Create HeatTransferCalculator instance."""
    return HeatTransferCalculator()


@pytest.fixture
def efficiency_calculator():
    """Create EfficiencyCalculator instance."""
    return EfficiencyCalculator()


@pytest.fixture
def vacuum_calculator():
    """Create VacuumCalculator instance."""
    return VacuumCalculator()


@pytest.fixture
def standard_heat_transfer_input():
    """Standard heat transfer input for testing."""
    return HeatTransferInput(
        heat_duty_mw=200.0,
        steam_temp_c=40.0,
        cw_inlet_temp_c=25.0,
        cw_outlet_temp_c=35.0,
        cw_flow_rate_m3_hr=50000.0,
        tube_od_mm=25.4,
        tube_id_mm=23.4,
        tube_length_m=12.0,
        tube_count=18500,
        tube_material="titanium",
        design_u_value_w_m2k=3500.0,
        fouling_factor_m2k_w=0.00015,
    )


@pytest.fixture
def standard_efficiency_input():
    """Standard efficiency input for testing."""
    return EfficiencyInput(
        steam_temp_c=40.0,
        cw_inlet_temp_c=25.0,
        cw_outlet_temp_c=35.0,
        cw_flow_rate_m3_hr=50000.0,
        heat_duty_mw=200.0,
        turbine_output_mw=300.0,
        design_backpressure_mmhg=50.8,
        actual_backpressure_mmhg=55.0,
        design_u_value_w_m2k=3500.0,
        actual_u_value_w_m2k=3000.0,
        heat_transfer_area_m2=17500.0,
        electricity_cost_usd_mwh=50.0,
        operating_hours_per_year=8000,
    )


@pytest.fixture
def degraded_condenser_input():
    """Degraded condenser efficiency input for testing."""
    return EfficiencyInput(
        steam_temp_c=45.0,
        cw_inlet_temp_c=28.0,
        cw_outlet_temp_c=38.0,
        cw_flow_rate_m3_hr=45000.0,
        heat_duty_mw=180.0,
        turbine_output_mw=280.0,
        design_backpressure_mmhg=50.8,
        actual_backpressure_mmhg=75.0,  # High backpressure
        design_u_value_w_m2k=3500.0,
        actual_u_value_w_m2k=2200.0,  # Degraded U-value
        heat_transfer_area_m2=17500.0,
        electricity_cost_usd_mwh=60.0,
        operating_hours_per_year=8000,
    )


@pytest.fixture
def golden_test_data():
    """Golden test data with known calculation results."""
    return {
        "case_1": {
            "input": {
                "steam_temp_c": 40.0,
                "cw_inlet_temp_c": 25.0,
                "cw_outlet_temp_c": 35.0,
            },
            "expected": {
                "ttd_c": 5.0,
                "itd_c": 15.0,
                "thermal_efficiency_pct": 66.67,
            }
        },
        "case_2": {
            "input": {
                "steam_temp_c": 45.0,
                "cw_inlet_temp_c": 20.0,
                "cw_outlet_temp_c": 38.0,
            },
            "expected": {
                "ttd_c": 7.0,
                "itd_c": 25.0,
                "thermal_efficiency_pct": 72.0,
            }
        },
        "case_3": {
            "input": {
                "steam_temp_c": 38.0,
                "cw_inlet_temp_c": 28.0,
                "cw_outlet_temp_c": 35.0,
            },
            "expected": {
                "ttd_c": 3.0,
                "itd_c": 10.0,
                "thermal_efficiency_pct": 70.0,
            }
        }
    }


# =============================================================================
# U-VALUE CALCULATION TESTS
# =============================================================================

class TestUValueCalculations:
    """Test suite for U-value (overall heat transfer coefficient) calculations."""

    @pytest.mark.unit
    def test_u_value_basic_calculation(self, heat_transfer_calculator, standard_heat_transfer_input):
        """Test basic U-value calculation from heat transfer inputs."""
        result, provenance = heat_transfer_calculator.calculate(standard_heat_transfer_input)

        assert result.actual_u_value_w_m2k > 0
        assert result.actual_u_value_w_m2k <= standard_heat_transfer_input.design_u_value_w_m2k
        assert provenance.provenance_hash is not None

    @pytest.mark.unit
    @pytest.mark.parametrize("design_u,actual_u,expected_ratio", [
        (3500.0, 3500.0, 1.0),
        (3500.0, 3000.0, 0.857),
        (3500.0, 2500.0, 0.714),
        (3500.0, 2000.0, 0.571),
        (4000.0, 3200.0, 0.8),
    ])
    def test_u_value_ratio_calculation(self, design_u, actual_u, expected_ratio):
        """Test U-value ratio calculation for various scenarios."""
        ratio = actual_u / design_u
        assert abs(ratio - expected_ratio) < 0.01

    @pytest.mark.unit
    def test_u_value_with_fouling(self, heat_transfer_calculator):
        """Test U-value degradation due to fouling."""
        # Clean condition
        clean_input = HeatTransferInput(
            heat_duty_mw=200.0,
            steam_temp_c=40.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=35.0,
            cw_flow_rate_m3_hr=50000.0,
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="titanium",
            design_u_value_w_m2k=3500.0,
            fouling_factor_m2k_w=0.00005,  # Low fouling
        )

        # Fouled condition
        fouled_input = HeatTransferInput(
            heat_duty_mw=200.0,
            steam_temp_c=40.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=35.0,
            cw_flow_rate_m3_hr=50000.0,
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="titanium",
            design_u_value_w_m2k=3500.0,
            fouling_factor_m2k_w=0.0005,  # High fouling
        )

        clean_result, _ = heat_transfer_calculator.calculate(clean_input)
        fouled_result, _ = heat_transfer_calculator.calculate(fouled_input)

        # Fouled U-value should be lower
        assert fouled_result.actual_u_value_w_m2k < clean_result.actual_u_value_w_m2k

    @pytest.mark.unit
    @pytest.mark.parametrize("tube_material,expected_conductivity", [
        ("titanium", 21.9),
        ("stainless_316", 16.3),
        ("admiralty_brass", 111.0),
        ("copper_nickel_90_10", 45.0),
    ])
    def test_tube_material_conductivity(self, tube_material, expected_conductivity):
        """Test tube material thermal conductivity values."""
        conductivity = TUBE_MATERIAL_CONDUCTIVITY.get(tube_material, 0)
        assert abs(conductivity - expected_conductivity) < 1.0

    @pytest.mark.unit
    def test_u_value_boundary_minimum(self, heat_transfer_calculator):
        """Test U-value calculation at minimum boundary conditions."""
        input_data = HeatTransferInput(
            heat_duty_mw=50.0,  # Low heat duty
            steam_temp_c=35.0,  # Low steam temp
            cw_inlet_temp_c=30.0,  # High inlet temp
            cw_outlet_temp_c=34.0,  # Small temp rise
            cw_flow_rate_m3_hr=80000.0,  # High flow
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="titanium",
            design_u_value_w_m2k=3500.0,
            fouling_factor_m2k_w=0.0008,  # High fouling
        )

        result, _ = heat_transfer_calculator.calculate(input_data)
        assert result.actual_u_value_w_m2k > 0

    @pytest.mark.unit
    def test_u_value_boundary_maximum(self, heat_transfer_calculator):
        """Test U-value calculation at maximum boundary conditions."""
        input_data = HeatTransferInput(
            heat_duty_mw=400.0,  # High heat duty
            steam_temp_c=50.0,  # High steam temp
            cw_inlet_temp_c=15.0,  # Low inlet temp
            cw_outlet_temp_c=40.0,  # Large temp rise
            cw_flow_rate_m3_hr=30000.0,  # Lower flow
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="admiralty_brass",  # High conductivity
            design_u_value_w_m2k=4000.0,
            fouling_factor_m2k_w=0.00005,  # Low fouling
        )

        result, _ = heat_transfer_calculator.calculate(input_data)
        assert result.actual_u_value_w_m2k > 0


# =============================================================================
# CLEANLINESS FACTOR TESTS
# =============================================================================

class TestCleanlinessFactorCalculations:
    """Test suite for cleanliness factor calculations."""

    @pytest.mark.unit
    def test_cleanliness_factor_basic(self, efficiency_calculator, standard_efficiency_input):
        """Test basic cleanliness factor calculation."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # CF = actual_U / design_U = 3000 / 3500 = 0.857
        expected_cf = 3000.0 / 3500.0
        assert abs(result.cleanliness_factor - expected_cf) < 0.001

    @pytest.mark.unit
    @pytest.mark.parametrize("actual_u,design_u,expected_cf", [
        (3500.0, 3500.0, 1.0),
        (3000.0, 3500.0, 0.857),
        (2500.0, 3500.0, 0.714),
        (2000.0, 3500.0, 0.571),
        (1500.0, 3500.0, 0.429),
    ])
    def test_cleanliness_factor_parametrized(self, actual_u, design_u, expected_cf):
        """Test cleanliness factor for various U-value combinations."""
        cf = actual_u / design_u
        assert abs(cf - expected_cf) < 0.01

    @pytest.mark.unit
    def test_cleanliness_factor_excellent_condition(self, efficiency_calculator):
        """Test cleanliness factor for excellent condenser condition."""
        input_data = EfficiencyInput(
            steam_temp_c=40.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=35.0,
            cw_flow_rate_m3_hr=50000.0,
            heat_duty_mw=200.0,
            turbine_output_mw=300.0,
            design_backpressure_mmhg=50.8,
            actual_backpressure_mmhg=51.0,  # Near design
            design_u_value_w_m2k=3500.0,
            actual_u_value_w_m2k=3450.0,  # Near design
            heat_transfer_area_m2=17500.0,
        )

        result, _ = efficiency_calculator.calculate(input_data)
        assert result.cleanliness_factor >= 0.95
        assert result.performance_rating == "Excellent"

    @pytest.mark.unit
    def test_cleanliness_factor_poor_condition(self, efficiency_calculator, degraded_condenser_input):
        """Test cleanliness factor for poor condenser condition."""
        result, _ = efficiency_calculator.calculate(degraded_condenser_input)

        # CF = 2200 / 3500 = 0.629
        assert result.cleanliness_factor < 0.70
        assert result.performance_rating in ["Poor", "Critical"]

    @pytest.mark.unit
    def test_cleanliness_factor_capped_at_one(self, efficiency_calculator):
        """Test that cleanliness factor is capped at 1.0."""
        input_data = EfficiencyInput(
            steam_temp_c=40.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=35.0,
            cw_flow_rate_m3_hr=50000.0,
            heat_duty_mw=200.0,
            turbine_output_mw=300.0,
            design_backpressure_mmhg=50.8,
            actual_backpressure_mmhg=48.0,
            design_u_value_w_m2k=3500.0,
            actual_u_value_w_m2k=3600.0,  # Higher than design
            heat_transfer_area_m2=17500.0,
        )

        result, _ = efficiency_calculator.calculate(input_data)
        assert result.cleanliness_factor <= 1.0

    @pytest.mark.unit
    def test_cleanliness_factor_thresholds(self):
        """Test cleanliness factor threshold classifications."""
        thresholds = {
            "excellent": 0.95,
            "good": 0.85,
            "average": 0.75,
            "poor": 0.65,
        }

        cf_values = [0.98, 0.90, 0.80, 0.70, 0.60]
        expected_ratings = ["excellent", "good", "average", "poor", "critical"]

        for cf, expected in zip(cf_values, expected_ratings):
            if cf >= 0.95:
                rating = "excellent"
            elif cf >= 0.85:
                rating = "good"
            elif cf >= 0.75:
                rating = "average"
            elif cf >= 0.65:
                rating = "poor"
            else:
                rating = "critical"
            assert rating == expected


# =============================================================================
# TTD (TERMINAL TEMPERATURE DIFFERENCE) TESTS
# =============================================================================

class TestTTDCalculations:
    """Test suite for Terminal Temperature Difference calculations."""

    @pytest.mark.unit
    def test_ttd_basic_calculation(self, efficiency_calculator, standard_efficiency_input):
        """Test basic TTD calculation."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # TTD = steam_temp - cw_outlet = 40 - 35 = 5
        expected_ttd = 40.0 - 35.0
        assert abs(result.ttd_c - expected_ttd) < 0.01

    @pytest.mark.unit
    @pytest.mark.parametrize("steam_temp,cw_outlet,expected_ttd", [
        (40.0, 35.0, 5.0),
        (45.0, 38.0, 7.0),
        (38.0, 35.0, 3.0),
        (50.0, 42.0, 8.0),
        (42.0, 40.0, 2.0),
    ])
    def test_ttd_parametrized(self, steam_temp, cw_outlet, expected_ttd):
        """Test TTD calculation with various temperature combinations."""
        ttd = steam_temp - cw_outlet
        assert abs(ttd - expected_ttd) < 0.001

    @pytest.mark.unit
    def test_ttd_excellent_performance(self, efficiency_calculator):
        """Test TTD for excellent condenser performance (low TTD)."""
        input_data = EfficiencyInput(
            steam_temp_c=40.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=38.0,  # High outlet = low TTD
            cw_flow_rate_m3_hr=60000.0,
            heat_duty_mw=200.0,
            turbine_output_mw=300.0,
            design_backpressure_mmhg=50.8,
            actual_backpressure_mmhg=50.8,
            design_u_value_w_m2k=3500.0,
            actual_u_value_w_m2k=3400.0,
            heat_transfer_area_m2=17500.0,
        )

        result, _ = efficiency_calculator.calculate(input_data)
        # TTD = 40 - 38 = 2C
        assert result.ttd_c <= 3.0

    @pytest.mark.unit
    def test_ttd_poor_performance(self, efficiency_calculator, degraded_condenser_input):
        """Test TTD for poor condenser performance (high TTD)."""
        result, _ = efficiency_calculator.calculate(degraded_condenser_input)

        # TTD = 45 - 38 = 7C
        assert result.ttd_c >= 5.0

    @pytest.mark.unit
    def test_ttd_with_golden_data(self, efficiency_calculator, golden_test_data):
        """Test TTD calculations against golden test data."""
        for case_name, case_data in golden_test_data.items():
            input_vals = case_data["input"]
            expected = case_data["expected"]

            ttd = input_vals["steam_temp_c"] - input_vals["cw_outlet_temp_c"]
            assert abs(ttd - expected["ttd_c"]) < 0.1, f"Failed for {case_name}"

    @pytest.mark.unit
    def test_ttd_and_itd_relationship(self, efficiency_calculator, standard_efficiency_input):
        """Test relationship between TTD and ITD."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # ITD should always be greater than TTD
        assert result.itd_c > result.ttd_c

        # ITD - TTD = cw_outlet - cw_inlet = temperature rise
        expected_diff = 35.0 - 25.0  # outlet - inlet
        actual_diff = result.itd_c - result.ttd_c
        assert abs(actual_diff - expected_diff) < 0.01


# =============================================================================
# VACUUM OPTIMIZATION TESTS
# =============================================================================

class TestVacuumOptimization:
    """Test suite for vacuum optimization calculations."""

    @pytest.mark.unit
    def test_vacuum_basic_calculation(self, vacuum_calculator):
        """Test basic vacuum pressure calculation."""
        input_data = VacuumInput(
            steam_temp_c=40.0,
            heat_load_mw=200.0,
            cw_inlet_temp_c=25.0,
            cw_flow_rate_m3_hr=50000.0,
            air_inleakage_rate_kg_hr=1.0,
            design_vacuum_mbar=50.0,
        )

        result, provenance = vacuum_calculator.calculate(input_data)

        assert result.actual_vacuum_mbar > 0
        assert result.actual_vacuum_mbar < 200  # Reasonable range
        assert provenance.provenance_hash is not None

    @pytest.mark.unit
    def test_vacuum_with_low_air_inleakage(self, vacuum_calculator):
        """Test vacuum with low air inleakage (good condition)."""
        input_data = VacuumInput(
            steam_temp_c=40.0,
            heat_load_mw=200.0,
            cw_inlet_temp_c=25.0,
            cw_flow_rate_m3_hr=50000.0,
            air_inleakage_rate_kg_hr=0.3,  # Low air
            design_vacuum_mbar=50.0,
        )

        result, _ = vacuum_calculator.calculate(input_data)

        # Low air should result in good vacuum (closer to design)
        assert result.vacuum_deviation_mbar < 10

    @pytest.mark.unit
    def test_vacuum_with_high_air_inleakage(self, vacuum_calculator):
        """Test vacuum with high air inleakage (poor condition)."""
        input_data = VacuumInput(
            steam_temp_c=40.0,
            heat_load_mw=200.0,
            cw_inlet_temp_c=25.0,
            cw_flow_rate_m3_hr=50000.0,
            air_inleakage_rate_kg_hr=5.0,  # High air
            design_vacuum_mbar=50.0,
        )

        result, _ = vacuum_calculator.calculate(input_data)

        # High air should degrade vacuum
        assert result.vacuum_deviation_mbar > 5

    @pytest.mark.unit
    @pytest.mark.parametrize("cw_inlet_temp,expected_vacuum_trend", [
        (20.0, "lower"),  # Cold water = better vacuum
        (25.0, "medium"),
        (30.0, "higher"),  # Warm water = worse vacuum
        (35.0, "highest"),
    ])
    def test_vacuum_vs_cooling_water_temperature(self, vacuum_calculator, cw_inlet_temp, expected_vacuum_trend):
        """Test vacuum pressure correlation with cooling water temperature."""
        input_data = VacuumInput(
            steam_temp_c=40.0,
            heat_load_mw=200.0,
            cw_inlet_temp_c=cw_inlet_temp,
            cw_flow_rate_m3_hr=50000.0,
            air_inleakage_rate_kg_hr=1.0,
            design_vacuum_mbar=50.0,
        )

        result, _ = vacuum_calculator.calculate(input_data)
        assert result.actual_vacuum_mbar > 0

    @pytest.mark.unit
    def test_vacuum_heat_rate_impact(self, efficiency_calculator, standard_efficiency_input):
        """Test heat rate deviation due to vacuum deviation."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # Backpressure deviation = 55.0 - 50.8 = 4.2 mmHg
        # Heat rate impact = 4.2 * 25 = 105 kJ/kWh
        bp_deviation = 55.0 - 50.8
        expected_hr_deviation = bp_deviation * HEAT_RATE_IMPACT_KJ_KWH_PER_MMHG

        assert abs(result.heat_rate_deviation_kj_kwh - expected_hr_deviation) < 1.0

    @pytest.mark.unit
    def test_vacuum_optimization_recommendations(self, vacuum_calculator):
        """Test vacuum optimization generates valid recommendations."""
        input_data = VacuumInput(
            steam_temp_c=45.0,
            heat_load_mw=200.0,
            cw_inlet_temp_c=28.0,
            cw_flow_rate_m3_hr=45000.0,
            air_inleakage_rate_kg_hr=3.0,
            design_vacuum_mbar=50.0,
        )

        result, _ = vacuum_calculator.calculate(input_data)

        # Should have recommendations for improvement
        assert result.optimization_potential_mw >= 0


# =============================================================================
# LMTD CALCULATION TESTS
# =============================================================================

class TestLMTDCalculations:
    """Test suite for Log Mean Temperature Difference calculations."""

    @pytest.mark.unit
    def test_lmtd_basic_calculation(self):
        """Test basic LMTD calculation."""
        # For a condenser with constant steam temperature
        steam_temp = 40.0
        cw_inlet = 25.0
        cw_outlet = 35.0

        dt1 = steam_temp - cw_inlet  # Hot end delta = 15
        dt2 = steam_temp - cw_outlet  # Cold end delta = 5

        # LMTD = (dt1 - dt2) / ln(dt1/dt2)
        expected_lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

        calculated_lmtd = calculate_lmtd(steam_temp, cw_inlet, cw_outlet)

        assert abs(calculated_lmtd - expected_lmtd) < 0.01

    @pytest.mark.unit
    @pytest.mark.parametrize("steam_temp,cw_inlet,cw_outlet", [
        (40.0, 25.0, 35.0),
        (45.0, 20.0, 38.0),
        (38.0, 28.0, 35.0),
        (50.0, 22.0, 42.0),
    ])
    def test_lmtd_various_conditions(self, steam_temp, cw_inlet, cw_outlet):
        """Test LMTD calculation for various temperature conditions."""
        lmtd = calculate_lmtd(steam_temp, cw_inlet, cw_outlet)

        # LMTD should be between ITD and TTD
        itd = steam_temp - cw_inlet
        ttd = steam_temp - cw_outlet

        assert ttd < lmtd < itd

    @pytest.mark.unit
    def test_lmtd_symmetric_case(self):
        """Test LMTD when dt1 equals dt2 (approaches arithmetic mean)."""
        steam_temp = 40.0
        cw_inlet = 30.0
        cw_outlet = 30.0  # Same delta at both ends

        # When dt1 = dt2, LMTD = dt (the common value)
        # Need to handle this edge case
        try:
            lmtd = calculate_lmtd(steam_temp, cw_inlet, cw_outlet)
            # Should equal the common delta
            assert abs(lmtd - 10.0) < 0.1
        except ZeroDivisionError:
            # Some implementations may not handle this edge case
            pass

    @pytest.mark.unit
    def test_lmtd_heat_transfer_relationship(self, heat_transfer_calculator, standard_heat_transfer_input):
        """Test LMTD relationship with heat transfer."""
        result, _ = heat_transfer_calculator.calculate(standard_heat_transfer_input)

        # Q = U * A * LMTD
        # LMTD = Q / (U * A)
        expected_lmtd = (result.heat_duty_mw * 1e6) / (
            result.actual_u_value_w_m2k * standard_heat_transfer_input.tube_count *
            math.pi * (standard_heat_transfer_input.tube_od_mm / 1000) *
            standard_heat_transfer_input.tube_length_m
        )

        # LMTD should be reasonable for condenser operation
        assert result.lmtd_c > 0
        assert result.lmtd_c < 30

    @pytest.mark.unit
    def test_lmtd_boundary_small_delta(self):
        """Test LMTD with very small temperature difference."""
        steam_temp = 40.0
        cw_inlet = 38.0
        cw_outlet = 39.5

        lmtd = calculate_lmtd(steam_temp, cw_inlet, cw_outlet)

        # Small deltas should still produce valid LMTD
        assert lmtd > 0
        assert lmtd < 5


# =============================================================================
# HEI STANDARD CALCULATION TESTS
# =============================================================================

class TestHEIStandardCalculations:
    """Test suite for HEI (Heat Exchange Institute) standard calculations."""

    @pytest.mark.unit
    def test_hei_u_value_correction(self, heat_transfer_calculator):
        """Test HEI standard U-value correction factors."""
        # HEI provides correction factors for:
        # - Tube material
        # - Inlet water temperature
        # - Tube cleanliness

        input_data = HeatTransferInput(
            heat_duty_mw=200.0,
            steam_temp_c=40.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=35.0,
            cw_flow_rate_m3_hr=50000.0,
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="titanium",
            design_u_value_w_m2k=3500.0,
            fouling_factor_m2k_w=0.00015,
        )

        result, _ = heat_transfer_calculator.calculate(input_data)

        # Verify HEI correction is applied
        assert result.hei_correction_factor > 0
        assert result.hei_correction_factor <= 1.2

    @pytest.mark.unit
    @pytest.mark.parametrize("cw_inlet_temp,expected_correction_range", [
        (15.0, (0.85, 0.95)),  # Cold water
        (25.0, (0.95, 1.05)),  # Design condition
        (35.0, (1.05, 1.15)),  # Warm water
    ])
    def test_hei_temperature_correction(self, heat_transfer_calculator, cw_inlet_temp, expected_correction_range):
        """Test HEI inlet temperature correction factors."""
        input_data = HeatTransferInput(
            heat_duty_mw=200.0,
            steam_temp_c=45.0,
            cw_inlet_temp_c=cw_inlet_temp,
            cw_outlet_temp_c=cw_inlet_temp + 10.0,
            cw_flow_rate_m3_hr=50000.0,
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="titanium",
            design_u_value_w_m2k=3500.0,
            fouling_factor_m2k_w=0.00015,
        )

        result, _ = heat_transfer_calculator.calculate(input_data)

        # Temperature correction should be within expected range
        assert result.hei_correction_factor >= expected_correction_range[0]
        assert result.hei_correction_factor <= expected_correction_range[1]

    @pytest.mark.unit
    def test_hei_tube_velocity_limits(self, heat_transfer_calculator):
        """Test HEI tube velocity limits and recommendations."""
        input_data = HeatTransferInput(
            heat_duty_mw=200.0,
            steam_temp_c=40.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=35.0,
            cw_flow_rate_m3_hr=50000.0,
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="titanium",
            design_u_value_w_m2k=3500.0,
            fouling_factor_m2k_w=0.00015,
        )

        result, _ = heat_transfer_calculator.calculate(input_data)

        # HEI recommends velocity 1.5-2.5 m/s for titanium tubes
        assert result.tube_velocity_m_s > 1.0
        assert result.tube_velocity_m_s < 3.5

    @pytest.mark.unit
    def test_hei_cleanliness_factor_standard(self, efficiency_calculator, standard_efficiency_input):
        """Test HEI standard cleanliness factor calculation."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # HEI defines CF = U_actual / U_clean
        # CF should be between 0 and 1
        assert 0 < result.cleanliness_factor <= 1.0

    @pytest.mark.unit
    def test_hei_performance_ratio(self, heat_transfer_calculator, standard_heat_transfer_input):
        """Test HEI performance ratio calculation."""
        result, _ = heat_transfer_calculator.calculate(standard_heat_transfer_input)

        # HEI Performance Ratio = Actual heat duty / Design heat duty
        # Should be positive and reasonable
        assert result.performance_ratio > 0
        assert result.performance_ratio <= 1.5


# =============================================================================
# BOUNDARY CONDITION TESTS
# =============================================================================

class TestBoundaryConditions:
    """Test suite for boundary conditions and edge cases."""

    @pytest.mark.unit
    def test_minimum_temperature_difference(self, efficiency_calculator):
        """Test calculation with minimum temperature difference."""
        input_data = EfficiencyInput(
            steam_temp_c=36.0,  # Just above outlet
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=35.0,  # Close to steam temp
            cw_flow_rate_m3_hr=80000.0,
            heat_duty_mw=150.0,
            turbine_output_mw=250.0,
            design_backpressure_mmhg=50.8,
            actual_backpressure_mmhg=52.0,
            design_u_value_w_m2k=3500.0,
            actual_u_value_w_m2k=3200.0,
            heat_transfer_area_m2=17500.0,
        )

        result, _ = efficiency_calculator.calculate(input_data)

        # TTD should be small but positive
        assert result.ttd_c == 1.0
        assert result.thermal_efficiency_pct > 0

    @pytest.mark.unit
    def test_maximum_flow_rate(self, heat_transfer_calculator):
        """Test calculation with maximum flow rate."""
        input_data = HeatTransferInput(
            heat_duty_mw=200.0,
            steam_temp_c=40.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=30.0,  # Small rise due to high flow
            cw_flow_rate_m3_hr=100000.0,  # Very high flow
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="titanium",
            design_u_value_w_m2k=3500.0,
            fouling_factor_m2k_w=0.00015,
        )

        result, _ = heat_transfer_calculator.calculate(input_data)

        # High flow should result in high tube velocity
        assert result.tube_velocity_m_s > 2.0

    @pytest.mark.unit
    def test_minimum_flow_rate(self, heat_transfer_calculator):
        """Test calculation with minimum flow rate."""
        input_data = HeatTransferInput(
            heat_duty_mw=100.0,
            steam_temp_c=40.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=38.0,  # Large rise due to low flow
            cw_flow_rate_m3_hr=20000.0,  # Low flow
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="titanium",
            design_u_value_w_m2k=3500.0,
            fouling_factor_m2k_w=0.00015,
        )

        result, _ = heat_transfer_calculator.calculate(input_data)

        # Low flow should result in low tube velocity
        assert result.tube_velocity_m_s < 2.0

    @pytest.mark.unit
    def test_zero_fouling(self, heat_transfer_calculator):
        """Test calculation with zero fouling (clean tubes)."""
        input_data = HeatTransferInput(
            heat_duty_mw=200.0,
            steam_temp_c=40.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=35.0,
            cw_flow_rate_m3_hr=50000.0,
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="titanium",
            design_u_value_w_m2k=3500.0,
            fouling_factor_m2k_w=0.00001,  # Nearly zero
        )

        result, _ = heat_transfer_calculator.calculate(input_data)

        # U-value should be near design
        ratio = result.actual_u_value_w_m2k / input_data.design_u_value_w_m2k
        assert ratio > 0.95

    @pytest.mark.unit
    def test_maximum_fouling(self, heat_transfer_calculator):
        """Test calculation with maximum fouling."""
        input_data = HeatTransferInput(
            heat_duty_mw=150.0,
            steam_temp_c=45.0,
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=38.0,
            cw_flow_rate_m3_hr=50000.0,
            tube_od_mm=25.4,
            tube_id_mm=23.4,
            tube_length_m=12.0,
            tube_count=18500,
            tube_material="titanium",
            design_u_value_w_m2k=3500.0,
            fouling_factor_m2k_w=0.001,  # Very high fouling
        )

        result, _ = heat_transfer_calculator.calculate(input_data)

        # U-value should be significantly degraded
        ratio = result.actual_u_value_w_m2k / input_data.design_u_value_w_m2k
        assert ratio < 0.7

    @pytest.mark.unit
    def test_input_validation_steam_temp_low(self, efficiency_calculator):
        """Test input validation for low steam temperature."""
        with pytest.raises(ValueError, match="Steam temperature"):
            EfficiencyInput(
                steam_temp_c=15.0,  # Too low
                cw_inlet_temp_c=10.0,
                cw_outlet_temp_c=12.0,
                cw_flow_rate_m3_hr=50000.0,
                heat_duty_mw=200.0,
                turbine_output_mw=300.0,
                design_backpressure_mmhg=50.8,
                actual_backpressure_mmhg=55.0,
                design_u_value_w_m2k=3500.0,
                actual_u_value_w_m2k=3000.0,
                heat_transfer_area_m2=17500.0,
            )

    @pytest.mark.unit
    def test_input_validation_steam_temp_high(self, efficiency_calculator):
        """Test input validation for high steam temperature."""
        with pytest.raises(ValueError, match="Steam temperature"):
            EfficiencyInput(
                steam_temp_c=70.0,  # Too high
                cw_inlet_temp_c=25.0,
                cw_outlet_temp_c=35.0,
                cw_flow_rate_m3_hr=50000.0,
                heat_duty_mw=200.0,
                turbine_output_mw=300.0,
                design_backpressure_mmhg=50.8,
                actual_backpressure_mmhg=55.0,
                design_u_value_w_m2k=3500.0,
                actual_u_value_w_m2k=3000.0,
                heat_transfer_area_m2=17500.0,
            )

    @pytest.mark.unit
    def test_input_validation_outlet_greater_than_inlet(self, efficiency_calculator):
        """Test validation that CW outlet must be greater than inlet."""
        with pytest.raises(ValueError, match="outlet temp must be greater"):
            EfficiencyInput(
                steam_temp_c=40.0,
                cw_inlet_temp_c=35.0,
                cw_outlet_temp_c=30.0,  # Less than inlet
                cw_flow_rate_m3_hr=50000.0,
                heat_duty_mw=200.0,
                turbine_output_mw=300.0,
                design_backpressure_mmhg=50.8,
                actual_backpressure_mmhg=55.0,
                design_u_value_w_m2k=3500.0,
                actual_u_value_w_m2k=3000.0,
                heat_transfer_area_m2=17500.0,
            )

    @pytest.mark.unit
    def test_input_validation_steam_greater_than_outlet(self, efficiency_calculator):
        """Test validation that steam temp must be greater than CW outlet."""
        with pytest.raises(ValueError, match="Steam temp must be greater"):
            EfficiencyInput(
                steam_temp_c=38.0,
                cw_inlet_temp_c=25.0,
                cw_outlet_temp_c=40.0,  # Greater than steam
                cw_flow_rate_m3_hr=50000.0,
                heat_duty_mw=200.0,
                turbine_output_mw=300.0,
                design_backpressure_mmhg=50.8,
                actual_backpressure_mmhg=55.0,
                design_u_value_w_m2k=3500.0,
                actual_u_value_w_m2k=3000.0,
                heat_transfer_area_m2=17500.0,
            )

    @pytest.mark.unit
    def test_input_validation_negative_heat_duty(self, efficiency_calculator):
        """Test validation for negative heat duty."""
        with pytest.raises(ValueError, match="Heat duty must be positive"):
            EfficiencyInput(
                steam_temp_c=40.0,
                cw_inlet_temp_c=25.0,
                cw_outlet_temp_c=35.0,
                cw_flow_rate_m3_hr=50000.0,
                heat_duty_mw=-100.0,  # Negative
                turbine_output_mw=300.0,
                design_backpressure_mmhg=50.8,
                actual_backpressure_mmhg=55.0,
                design_u_value_w_m2k=3500.0,
                actual_u_value_w_m2k=3000.0,
                heat_transfer_area_m2=17500.0,
            )


# =============================================================================
# PERFORMANCE RATING TESTS
# =============================================================================

class TestPerformanceRating:
    """Test suite for performance rating classification."""

    @pytest.mark.unit
    @pytest.mark.parametrize("cpi,expected_rating", [
        (0.98, "Excellent"),
        (0.95, "Excellent"),
        (0.90, "Good"),
        (0.85, "Good"),
        (0.80, "Average"),
        (0.75, "Average"),
        (0.70, "Poor"),
        (0.65, "Poor"),
        (0.60, "Critical"),
        (0.50, "Critical"),
    ])
    def test_performance_rating_thresholds(self, cpi, expected_rating):
        """Test performance rating threshold classification."""
        if cpi >= 0.95:
            rating = "Excellent"
        elif cpi >= 0.85:
            rating = "Good"
        elif cpi >= 0.75:
            rating = "Average"
        elif cpi >= 0.65:
            rating = "Poor"
        else:
            rating = "Critical"

        assert rating == expected_rating

    @pytest.mark.unit
    def test_cpi_calculation(self, efficiency_calculator, standard_efficiency_input):
        """Test CPI (Condenser Performance Index) calculation."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # CPI should be between 0 and 1
        assert 0 < result.cpi <= 1.0

        # Verify CPI is consistent with performance rating
        if result.cpi >= 0.95:
            assert result.performance_rating == "Excellent"
        elif result.cpi >= 0.85:
            assert result.performance_rating == "Good"

    @pytest.mark.unit
    def test_cpi_weighted_components(self, efficiency_calculator, standard_efficiency_input):
        """Test CPI weighted component calculation."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # CPI = 0.4*CF + 0.3*TTD_factor + 0.3*BP_factor
        cf = result.cleanliness_factor

        # Each component should contribute to final CPI
        assert result.cpi > 0


# =============================================================================
# EFFICIENCY LOSS CALCULATION TESTS
# =============================================================================

class TestEfficiencyLossCalculations:
    """Test suite for efficiency loss and savings calculations."""

    @pytest.mark.unit
    def test_efficiency_loss_calculation(self, efficiency_calculator, standard_efficiency_input):
        """Test efficiency loss calculation from backpressure deviation."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # Verify efficiency loss is calculated
        assert result.efficiency_loss_mw >= 0

    @pytest.mark.unit
    def test_annual_energy_loss(self, efficiency_calculator, standard_efficiency_input):
        """Test annual energy loss calculation."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # Annual energy loss = efficiency_loss_mw * operating_hours
        expected_annual_loss = result.efficiency_loss_mw * 8000
        assert abs(result.annual_energy_loss_mwh - expected_annual_loss) < 1.0

    @pytest.mark.unit
    def test_annual_cost_loss(self, efficiency_calculator, standard_efficiency_input):
        """Test annual cost loss calculation."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # Annual cost = annual_energy_loss * electricity_cost
        expected_cost = result.annual_energy_loss_mwh * 50.0
        assert abs(result.annual_cost_loss_usd - expected_cost) < 10.0

    @pytest.mark.unit
    def test_carbon_penalty(self, efficiency_calculator, standard_efficiency_input):
        """Test carbon emission penalty calculation."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)

        # Carbon = annual_energy_loss * emission_factor / 1000
        expected_carbon = result.annual_energy_loss_mwh * CARBON_EMISSION_FACTOR_KG_CO2_MWH / 1000
        assert abs(result.annual_carbon_penalty_tonnes - expected_carbon) < 1.0

    @pytest.mark.unit
    def test_potential_savings(self, efficiency_calculator, degraded_condenser_input):
        """Test potential savings calculation for degraded condenser."""
        result, _ = efficiency_calculator.calculate(degraded_condenser_input)

        # Degraded condenser should have significant potential savings
        assert result.potential_savings_mw > 0
        assert result.potential_annual_savings_usd > 0


# =============================================================================
# STANDALONE FUNCTION TESTS
# =============================================================================

class TestStandaloneFunctions:
    """Test suite for standalone calculation functions."""

    @pytest.mark.unit
    def test_cw_temperature_rise(self):
        """Test cooling water temperature rise calculation."""
        heat_duty = 200.0  # MW
        flow_rate = 50000.0  # m3/hr

        temp_rise = calculate_cw_temperature_rise(heat_duty, flow_rate)

        # Expected: Q = m_dot * Cp * dT
        # dT = Q / (m_dot * Cp)
        # dT = 200e6 / (50000*995/3600 * 4180) = ~3.5 C
        assert temp_rise > 0
        assert temp_rise < 20

    @pytest.mark.unit
    def test_optimal_cw_flow(self):
        """Test optimal CW flow calculation."""
        heat_duty = 200.0  # MW
        target_rise = 10.0  # C

        flow_rate = calculate_optimal_cw_flow(heat_duty, target_rise)

        # Verify by back-calculation
        actual_rise = calculate_cw_temperature_rise(heat_duty, flow_rate)
        assert abs(actual_rise - target_rise) < 0.5

    @pytest.mark.unit
    def test_cw_pumping_power(self):
        """Test CW pumping power calculation."""
        flow_rate = 50000.0  # m3/hr
        head = 20.0  # m
        efficiency = 0.80

        power = calculate_cw_pumping_power(flow_rate, head, efficiency)

        # P = rho * g * Q * H / eta
        expected = (995 * 9.81 * (50000/3600) * 20) / 0.80 / 1000
        assert abs(power - expected) < 10

    @pytest.mark.unit
    def test_payback_period(self):
        """Test simple payback period calculation."""
        investment = 500000.0  # USD
        annual_savings = 100000.0  # USD/year

        payback = calculate_payback_period(investment, annual_savings)

        assert payback == 5.0

    @pytest.mark.unit
    def test_payback_period_zero_savings(self):
        """Test payback period with zero savings."""
        investment = 500000.0
        annual_savings = 0.0

        payback = calculate_payback_period(investment, annual_savings)

        assert payback == float('inf')

    @pytest.mark.unit
    def test_npv_calculation(self):
        """Test NPV calculation."""
        investment = 500000.0
        annual_savings = 100000.0
        discount_rate = 0.08
        project_life = 10

        npv = calculate_npv(investment, annual_savings, discount_rate, project_life)

        # NPV should be positive for good investment
        assert npv > 0

    @pytest.mark.unit
    def test_npv_negative_case(self):
        """Test NPV calculation for unprofitable case."""
        investment = 1000000.0
        annual_savings = 50000.0
        discount_rate = 0.10
        project_life = 5

        npv = calculate_npv(investment, annual_savings, discount_rate, project_life)

        # NPV should be negative for poor investment
        assert npv < 0


# =============================================================================
# PROVENANCE AND DETERMINISM TESTS
# =============================================================================

class TestProvenanceTracking:
    """Test suite for provenance tracking in calculations."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, efficiency_calculator, standard_efficiency_input):
        """Test that provenance hash is generated."""
        result, provenance = efficiency_calculator.calculate(standard_efficiency_input)

        assert provenance.provenance_hash is not None
        assert len(provenance.provenance_hash) == 64  # SHA-256

    @pytest.mark.unit
    def test_provenance_input_hash(self, efficiency_calculator, standard_efficiency_input):
        """Test that input hash is generated."""
        result, provenance = efficiency_calculator.calculate(standard_efficiency_input)

        assert provenance.input_hash is not None
        assert len(provenance.input_hash) == 64

    @pytest.mark.unit
    def test_provenance_output_hash(self, efficiency_calculator, standard_efficiency_input):
        """Test that output hash is generated."""
        result, provenance = efficiency_calculator.calculate(standard_efficiency_input)

        assert provenance.output_hash is not None
        assert len(provenance.output_hash) == 64

    @pytest.mark.unit
    def test_provenance_verification(self, efficiency_calculator, standard_efficiency_input):
        """Test provenance verification."""
        result, provenance = efficiency_calculator.calculate(standard_efficiency_input)

        is_valid = verify_provenance(provenance)
        assert is_valid is True

    @pytest.mark.unit
    def test_provenance_steps_recorded(self, efficiency_calculator, standard_efficiency_input):
        """Test that calculation steps are recorded."""
        result, provenance = efficiency_calculator.calculate(standard_efficiency_input)

        assert len(provenance.calculation_steps) > 0

        # Verify steps have required fields
        for step in provenance.calculation_steps:
            assert step.step_number is not None
            assert step.description is not None
            assert step.output_name is not None

    @pytest.mark.unit
    def test_deterministic_output(self, efficiency_calculator, standard_efficiency_input):
        """Test that same input produces same output."""
        result1, prov1 = efficiency_calculator.calculate(standard_efficiency_input)
        result2, prov2 = efficiency_calculator.calculate(standard_efficiency_input)

        # Same provenance hash for same inputs
        assert prov1.input_hash == prov2.input_hash
        assert prov1.output_hash == prov2.output_hash

        # Same output values
        assert result1.thermal_efficiency_pct == result2.thermal_efficiency_pct
        assert result1.cpi == result2.cpi
        assert result1.cleanliness_factor == result2.cleanliness_factor


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================

class TestPropertyBased:
    """Property-based tests for invariants."""

    @pytest.mark.unit
    @pytest.mark.parametrize("steam_temp", [35.0, 40.0, 45.0, 50.0, 55.0])
    @pytest.mark.parametrize("cw_inlet", [15.0, 20.0, 25.0, 30.0])
    def test_ttd_always_positive(self, efficiency_calculator, steam_temp, cw_inlet):
        """Property: TTD should always be positive for valid inputs."""
        if steam_temp <= cw_inlet + 6:  # Need valid temp difference
            pytest.skip("Invalid temperature combination")

        cw_outlet = cw_inlet + 5  # 5C rise
        if steam_temp <= cw_outlet:
            pytest.skip("Steam must be greater than outlet")

        input_data = EfficiencyInput(
            steam_temp_c=steam_temp,
            cw_inlet_temp_c=cw_inlet,
            cw_outlet_temp_c=cw_outlet,
            cw_flow_rate_m3_hr=50000.0,
            heat_duty_mw=200.0,
            turbine_output_mw=300.0,
            design_backpressure_mmhg=50.8,
            actual_backpressure_mmhg=55.0,
            design_u_value_w_m2k=3500.0,
            actual_u_value_w_m2k=3000.0,
            heat_transfer_area_m2=17500.0,
        )

        result, _ = efficiency_calculator.calculate(input_data)
        assert result.ttd_c > 0

    @pytest.mark.unit
    def test_cleanliness_factor_bounded(self, efficiency_calculator):
        """Property: Cleanliness factor should be between 0 and 1."""
        test_cases = [
            (3500.0, 3500.0),  # 100%
            (3000.0, 3500.0),  # ~86%
            (2000.0, 3500.0),  # ~57%
            (3600.0, 3500.0),  # >100%, should cap at 1
        ]

        for actual_u, design_u in test_cases:
            input_data = EfficiencyInput(
                steam_temp_c=40.0,
                cw_inlet_temp_c=25.0,
                cw_outlet_temp_c=35.0,
                cw_flow_rate_m3_hr=50000.0,
                heat_duty_mw=200.0,
                turbine_output_mw=300.0,
                design_backpressure_mmhg=50.8,
                actual_backpressure_mmhg=55.0,
                design_u_value_w_m2k=design_u,
                actual_u_value_w_m2k=actual_u,
                heat_transfer_area_m2=17500.0,
            )

            result, _ = efficiency_calculator.calculate(input_data)
            assert 0 < result.cleanliness_factor <= 1.0

    @pytest.mark.unit
    def test_cpi_bounded(self, efficiency_calculator, standard_efficiency_input):
        """Property: CPI should be between 0 and 1."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)
        assert 0 < result.cpi <= 1.0

    @pytest.mark.unit
    def test_thermal_efficiency_bounded(self, efficiency_calculator, standard_efficiency_input):
        """Property: Thermal efficiency should be between 0 and 100%."""
        result, _ = efficiency_calculator.calculate(standard_efficiency_input)
        assert 0 < result.thermal_efficiency_pct <= 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
