# -*- coding: utf-8 -*-
"""
Unit tests for GL-024 Air Preheater Agent Calculations Module

Tests zero-hallucination calculation engine with physics-based validations.
All test values derived from ASME PTC 4.3 and standard heat transfer references.

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
import math
from typing import Dict, Any

from greenlang.agents.process_heat.gl_024_air_preheater.calculations import (
    AirPreheaterCalculator,
    EffectivenessResult,
    NTUResult,
    HeatDutyResult,
    LMTDResult,
    XRatioResult,
    LeakageResult,
)
from greenlang.agents.process_heat.gl_024_air_preheater.config import (
    AirPreheaterConfig,
    PreheaterType,
    AirPreheaterType,
    create_test_config,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator():
    """Create calculator with default config."""
    config = create_test_config("APH-TEST-001")
    return AirPreheaterCalculator(config)


@pytest.fixture
def typical_regenerative_conditions():
    """Typical operating conditions for Ljungstrom regenerative preheater."""
    return {
        "gas_inlet_temp_f": 700.0,
        "gas_outlet_temp_f": 300.0,
        "air_inlet_temp_f": 80.0,
        "air_outlet_temp_f": 580.0,
        "gas_flow_lb_hr": 500000.0,
        "air_flow_lb_hr": 480000.0,
        "o2_inlet_pct": 3.0,
        "o2_outlet_pct": 4.5,
    }


@pytest.fixture
def coal_fired_conditions():
    """Typical conditions for coal-fired application."""
    return {
        "so3_ppm": 10.0,
        "h2o_vol_pct": 8.0,
        "so2_ppm": 1000.0,
        "excess_air_pct": 20.0,
        "gas_pressure_psia": 14.5,
    }


# =============================================================================
# EFFECTIVENESS CALCULATION TESTS
# =============================================================================

class TestEffectivenessCalculation:
    """Test suite for effectiveness calculations."""

    @pytest.mark.unit
    def test_calculate_effectiveness_basic(self, calculator, typical_regenerative_conditions):
        """Test basic effectiveness calculation."""
        result = calculator.calculate_effectiveness(
            gas_inlet_temp_f=typical_regenerative_conditions["gas_inlet_temp_f"],
            gas_outlet_temp_f=typical_regenerative_conditions["gas_outlet_temp_f"],
            air_inlet_temp_f=typical_regenerative_conditions["air_inlet_temp_f"],
            air_outlet_temp_f=typical_regenerative_conditions["air_outlet_temp_f"],
        )

        assert isinstance(result, EffectivenessResult)
        # Effectiveness should be between 0 and 1
        assert 0 < result.effectiveness < 1
        # For these conditions, expect ~65-75% effectiveness
        assert 0.60 < result.effectiveness < 0.80

    @pytest.mark.unit
    def test_effectiveness_gas_side_calculation(self, calculator):
        """Test gas-side effectiveness calculation."""
        # Gas-side effectiveness = (T_gas_in - T_gas_out) / (T_gas_in - T_air_in)
        result = calculator.calculate_effectiveness(
            gas_inlet_temp_f=700.0,
            gas_outlet_temp_f=300.0,
            air_inlet_temp_f=80.0,
            air_outlet_temp_f=600.0,
        )

        # (700 - 300) / (700 - 80) = 400 / 620 = 0.645
        expected_gas_effectiveness = 400.0 / 620.0
        assert abs(result.hot_side_effectiveness - expected_gas_effectiveness) < 0.01

    @pytest.mark.unit
    def test_effectiveness_air_side_calculation(self, calculator):
        """Test air-side effectiveness calculation."""
        # Air-side effectiveness = (T_air_out - T_air_in) / (T_gas_in - T_air_in)
        result = calculator.calculate_effectiveness(
            gas_inlet_temp_f=700.0,
            gas_outlet_temp_f=300.0,
            air_inlet_temp_f=80.0,
            air_outlet_temp_f=600.0,
        )

        # (600 - 80) / (700 - 80) = 520 / 620 = 0.839
        expected_air_effectiveness = 520.0 / 620.0
        assert abs(result.cold_side_effectiveness - expected_air_effectiveness) < 0.01

    @pytest.mark.unit
    def test_effectiveness_capacity_ratio(self, calculator):
        """Test capacity ratio calculation."""
        result = calculator.calculate_effectiveness(
            gas_inlet_temp_f=700.0,
            gas_outlet_temp_f=300.0,
            air_inlet_temp_f=80.0,
            air_outlet_temp_f=600.0,
        )

        # Capacity ratio should be C_min/C_max
        assert 0 < result.capacity_ratio <= 1

    @pytest.mark.unit
    def test_effectiveness_methodology_documented(self, calculator):
        """Test that methodology is documented in result."""
        result = calculator.calculate_effectiveness(
            gas_inlet_temp_f=700.0,
            gas_outlet_temp_f=300.0,
            air_inlet_temp_f=80.0,
            air_outlet_temp_f=600.0,
        )

        assert "ASME PTC 4.3" in result.methodology or "epsilon-NTU" in result.methodology


# =============================================================================
# NTU CALCULATION TESTS
# =============================================================================

class TestNTUCalculation:
    """Test suite for NTU calculations."""

    @pytest.mark.unit
    def test_calculate_ntu_regenerative(self, calculator):
        """Test NTU calculation for regenerative preheater."""
        result = calculator.calculate_ntu(
            effectiveness=0.75,
            capacity_ratio=0.9,
            preheater_type=PreheaterType.REGENERATIVE,
        )

        assert isinstance(result, NTUResult)
        # For 75% effectiveness with Cr=0.9, NTU should be ~2.5-3.5
        assert 2.0 < result.ntu < 4.0

    @pytest.mark.unit
    def test_ntu_increases_with_effectiveness(self, calculator):
        """Test that NTU increases with effectiveness."""
        ntu_low = calculator.calculate_ntu(
            effectiveness=0.50,
            capacity_ratio=0.9,
            preheater_type=PreheaterType.REGENERATIVE,
        )
        ntu_high = calculator.calculate_ntu(
            effectiveness=0.80,
            capacity_ratio=0.9,
            preheater_type=PreheaterType.REGENERATIVE,
        )

        assert ntu_high.ntu > ntu_low.ntu


# =============================================================================
# HEAT DUTY CALCULATION TESTS
# =============================================================================

class TestHeatDutyCalculation:
    """Test suite for heat duty calculations."""

    @pytest.mark.unit
    def test_calculate_heat_duty_gas_side(self, calculator):
        """Test gas-side heat duty calculation."""
        result = calculator.calculate_heat_duty(
            flow_rate_lb_hr=500000.0,
            inlet_temp_f=700.0,
            outlet_temp_f=300.0,
            fluid_type="flue_gas",
        )

        assert isinstance(result, HeatDutyResult)
        # Q = m_dot * Cp * dT
        # With Cp ~ 0.25 BTU/lb-F, Q = 500000 * 0.25 * 400 = 50 MMBTU/hr
        assert 40.0 < result.heat_duty_mmbtu_hr < 60.0

    @pytest.mark.unit
    def test_calculate_heat_duty_air_side(self, calculator):
        """Test air-side heat duty calculation."""
        result = calculator.calculate_heat_duty(
            flow_rate_lb_hr=480000.0,
            inlet_temp_f=80.0,
            outlet_temp_f=600.0,
            fluid_type="air",
        )

        assert isinstance(result, HeatDutyResult)
        # Q = m_dot * Cp * dT
        # With Cp ~ 0.24 BTU/lb-F, Q = 480000 * 0.24 * 520 = ~60 MMBTU/hr
        assert 50.0 < result.heat_duty_mmbtu_hr < 70.0

    @pytest.mark.unit
    def test_heat_duty_formula_documented(self, calculator):
        """Test that formula is documented."""
        result = calculator.calculate_heat_duty(
            flow_rate_lb_hr=500000.0,
            inlet_temp_f=700.0,
            outlet_temp_f=300.0,
            fluid_type="flue_gas",
        )

        assert "Q = m_dot" in result.methodology or "Cp" in result.methodology


# =============================================================================
# LMTD CALCULATION TESTS
# =============================================================================

class TestLMTDCalculation:
    """Test suite for LMTD calculations."""

    @pytest.mark.unit
    def test_calculate_lmtd_counterflow(self, calculator):
        """Test LMTD calculation for counterflow arrangement."""
        result = calculator.calculate_lmtd(
            gas_inlet_temp_f=700.0,
            gas_outlet_temp_f=300.0,
            air_inlet_temp_f=80.0,
            air_outlet_temp_f=600.0,
        )

        assert isinstance(result, LMTDResult)
        # dT1 = 700 - 600 = 100F (hot end)
        # dT2 = 300 - 80 = 220F (cold end)
        # LMTD = (220 - 100) / ln(220/100) = 120 / 0.788 = 152F
        assert 140.0 < result.lmtd_f < 170.0

    @pytest.mark.unit
    def test_lmtd_approach_temperatures(self, calculator):
        """Test approach temperature calculations."""
        result = calculator.calculate_lmtd(
            gas_inlet_temp_f=700.0,
            gas_outlet_temp_f=300.0,
            air_inlet_temp_f=80.0,
            air_outlet_temp_f=600.0,
        )

        # Hot end approach = T_gas_in - T_air_out = 700 - 600 = 100F
        assert abs(result.delta_t1_f - 100.0) < 5.0
        # Cold end approach = T_gas_out - T_air_in = 300 - 80 = 220F
        assert abs(result.delta_t2_f - 220.0) < 5.0


# =============================================================================
# X-RATIO CALCULATION TESTS
# =============================================================================

class TestXRatioCalculation:
    """Test suite for X-ratio calculations (regenerative preheaters)."""

    @pytest.mark.unit
    def test_calculate_x_ratio(self, calculator, typical_regenerative_conditions):
        """Test X-ratio calculation."""
        result = calculator.calculate_x_ratio(
            gas_inlet_temp_f=typical_regenerative_conditions["gas_inlet_temp_f"],
            gas_outlet_temp_f=typical_regenerative_conditions["gas_outlet_temp_f"],
            air_inlet_temp_f=typical_regenerative_conditions["air_inlet_temp_f"],
            air_outlet_temp_f=typical_regenerative_conditions["air_outlet_temp_f"],
            o2_inlet_pct=typical_regenerative_conditions["o2_inlet_pct"],
            o2_outlet_pct=typical_regenerative_conditions["o2_outlet_pct"],
        )

        assert isinstance(result, XRatioResult)
        # X-ratio typically 0.8-1.2 for regenerative preheaters
        assert 0.7 < result.x_ratio < 1.3


# =============================================================================
# LEAKAGE CALCULATION TESTS
# =============================================================================

class TestLeakageCalculation:
    """Test suite for leakage calculations per ASME PTC 4.3."""

    @pytest.mark.unit
    def test_calculate_leakage_o2_method(self, calculator):
        """Test O2 rise method leakage calculation."""
        result = calculator.calculate_leakage_o2_method(
            o2_inlet_pct=3.0,
            o2_outlet_pct=4.5,
            gas_flow_rate_lb_hr=500000.0,
            air_flow_rate_lb_hr=480000.0,
        )

        assert isinstance(result, LeakageResult)
        # O2 rise of 1.5% indicates significant leakage (~8-12%)
        assert 5.0 < result.air_to_gas_leakage_pct < 15.0

    @pytest.mark.unit
    def test_leakage_o2_rise(self, calculator):
        """Test O2 rise calculation."""
        result = calculator.calculate_leakage_o2_method(
            o2_inlet_pct=3.0,
            o2_outlet_pct=4.5,
            gas_flow_rate_lb_hr=500000.0,
            air_flow_rate_lb_hr=480000.0,
        )

        assert result.o2_rise_pct == pytest.approx(1.5, abs=0.1)

    @pytest.mark.unit
    def test_zero_leakage_no_o2_rise(self, calculator):
        """Test that no O2 rise means minimal leakage."""
        result = calculator.calculate_leakage_o2_method(
            o2_inlet_pct=3.0,
            o2_outlet_pct=3.0,  # No O2 rise
            gas_flow_rate_lb_hr=500000.0,
            air_flow_rate_lb_hr=480000.0,
        )

        # Should be very low leakage
        assert result.air_to_gas_leakage_pct < 1.0


# =============================================================================
# ACID DEW POINT CALCULATION TESTS
# =============================================================================

class TestAcidDewPointCalculation:
    """Test suite for acid dew point calculations."""

    @pytest.mark.unit
    def test_verhoff_banchero_calculation(self, calculator, coal_fired_conditions):
        """Test Verhoff-Banchero acid dew point calculation."""
        result = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_vol_pct=coal_fired_conditions["h2o_vol_pct"],
            so3_ppm=coal_fired_conditions["so3_ppm"],
        )

        # For coal with 10 ppm SO3 and 8% H2O, expect ADP ~260-290F
        assert 250.0 < result.acid_dew_point_f < 310.0

    @pytest.mark.unit
    def test_okkes_calculation(self, calculator, coal_fired_conditions):
        """Test Okkes acid dew point calculation."""
        result = calculator.calculate_acid_dew_point_okkes(
            so2_ppm=coal_fired_conditions["so2_ppm"],
            h2o_vol_pct=coal_fired_conditions["h2o_vol_pct"],
            excess_air_pct=coal_fired_conditions["excess_air_pct"],
        )

        # Okkes method should give similar results
        assert 250.0 < result.acid_dew_point_f < 320.0

    @pytest.mark.unit
    def test_higher_so3_increases_adp(self, calculator):
        """Test that higher SO3 increases acid dew point."""
        adp_low = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_vol_pct=8.0,
            so3_ppm=5.0,
        )
        adp_high = calculator.calculate_acid_dew_point_verhoff_banchero(
            h2o_vol_pct=8.0,
            so3_ppm=20.0,
        )

        assert adp_high.acid_dew_point_f > adp_low.acid_dew_point_f


# =============================================================================
# WATER DEW POINT CALCULATION TESTS
# =============================================================================

class TestWaterDewPointCalculation:
    """Test suite for water dew point calculations."""

    @pytest.mark.unit
    def test_water_dew_point_calculation(self, calculator):
        """Test water dew point calculation."""
        result = calculator.calculate_water_dew_point(
            h2o_vol_pct=8.0,
            pressure_psia=14.7,
        )

        # For 8% moisture at atmospheric pressure, expect ~115-135F
        assert 110.0 < result.water_dew_point_f < 140.0

    @pytest.mark.unit
    def test_higher_moisture_increases_wdp(self, calculator):
        """Test that higher moisture increases water dew point."""
        wdp_low = calculator.calculate_water_dew_point(
            h2o_vol_pct=5.0,
            pressure_psia=14.7,
        )
        wdp_high = calculator.calculate_water_dew_point(
            h2o_vol_pct=15.0,
            pressure_psia=14.7,
        )

        assert wdp_high.water_dew_point_f > wdp_low.water_dew_point_f


# =============================================================================
# CLEANLINESS FACTOR TESTS
# =============================================================================

class TestCleanlinessFactorCalculation:
    """Test suite for cleanliness factor calculations."""

    @pytest.mark.unit
    def test_clean_preheater_factor(self, calculator):
        """Test cleanliness factor for clean preheater."""
        result = calculator.calculate_cleanliness_factor(
            current_effectiveness=0.80,
            design_effectiveness=0.80,
            current_ua=200000.0,
            design_ua=200000.0,
        )

        # Clean preheater should have factor ~1.0
        assert 0.95 < result.cleanliness_factor <= 1.0

    @pytest.mark.unit
    def test_fouled_preheater_factor(self, calculator):
        """Test cleanliness factor for fouled preheater."""
        result = calculator.calculate_cleanliness_factor(
            current_effectiveness=0.65,
            design_effectiveness=0.80,
            current_ua=150000.0,
            design_ua=200000.0,
        )

        # Fouled preheater should have factor < 0.85
        assert result.cleanliness_factor < 0.85


# =============================================================================
# ENERGY SAVINGS CALCULATION TESTS
# =============================================================================

class TestEnergySavingsCalculation:
    """Test suite for energy savings calculations."""

    @pytest.mark.unit
    def test_energy_savings_from_improvement(self, calculator):
        """Test energy savings from effectiveness improvement."""
        result = calculator.calculate_energy_savings(
            current_effectiveness=0.70,
            achievable_effectiveness=0.80,
            fuel_flow_mmbtu_hr=100.0,
            fuel_cost_per_mmbtu=5.0,
        )

        # 10% effectiveness improvement should yield significant savings
        assert result.annual_savings_mmbtu > 0
        assert result.annual_cost_savings_usd > 0
        assert result.efficiency_gain_pct > 0

    @pytest.mark.unit
    def test_no_improvement_no_savings(self, calculator):
        """Test that no improvement yields no savings."""
        result = calculator.calculate_energy_savings(
            current_effectiveness=0.80,
            achievable_effectiveness=0.80,
            fuel_flow_mmbtu_hr=100.0,
            fuel_cost_per_mmbtu=5.0,
        )

        assert result.annual_savings_mmbtu == pytest.approx(0, abs=0.1)


# =============================================================================
# BOUNDARY CONDITION TESTS
# =============================================================================

class TestBoundaryConditions:
    """Test suite for boundary and edge conditions."""

    @pytest.mark.unit
    def test_minimum_temperature_difference(self, calculator):
        """Test with minimum temperature difference."""
        result = calculator.calculate_effectiveness(
            gas_inlet_temp_f=300.0,
            gas_outlet_temp_f=250.0,  # Small difference
            air_inlet_temp_f=80.0,
            air_outlet_temp_f=200.0,
        )

        # Should still calculate valid effectiveness
        assert 0 < result.effectiveness < 1

    @pytest.mark.unit
    def test_high_temperature_operation(self, calculator):
        """Test with high temperature operation."""
        result = calculator.calculate_effectiveness(
            gas_inlet_temp_f=900.0,  # High gas inlet
            gas_outlet_temp_f=350.0,
            air_inlet_temp_f=100.0,
            air_outlet_temp_f=700.0,
        )

        assert 0 < result.effectiveness < 1

    @pytest.mark.unit
    def test_low_load_operation(self, calculator):
        """Test with low flow rates (low load)."""
        result = calculator.calculate_heat_duty(
            flow_rate_lb_hr=100000.0,  # Low flow
            inlet_temp_f=500.0,
            outlet_temp_f=280.0,
            fluid_type="flue_gas",
        )

        # Should still calculate valid heat duty
        assert result.heat_duty_mmbtu_hr > 0
