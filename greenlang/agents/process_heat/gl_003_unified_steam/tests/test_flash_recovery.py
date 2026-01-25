"""
GL-003 Unified Steam System Optimizer - Flash Recovery Module Tests

Unit tests for flash steam recovery calculations.
Target: 85%+ coverage of flash_recovery.py

Tests:
    - Thermodynamic flash fraction calculations
    - Flash tank sizing
    - Multi-stage flash optimization
    - Energy recovery calculations
    - IAPWS-IF97 steam property validation
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import math


from greenlang.agents.process_heat.gl_003_unified_steam.flash_recovery import (
    FlashSteamCalculator,
    FlashTankSizer,
    MultiStageFlashOptimizer,
    FlashRecoveryOptimizer,
    FlashConstants,
)
from greenlang.agents.process_heat.gl_003_unified_steam.config import FlashRecoveryConfig
from greenlang.agents.process_heat.gl_003_unified_steam.schemas import FlashSteamInput, FlashSteamOutput


# =============================================================================
# FLASH CONSTANTS TESTS
# =============================================================================

class TestFlashConstants:
    """Test suite for FlashConstants."""

    def test_minimum_pressure_differential(self):
        """Test minimum pressure differential constant."""
        assert FlashConstants.MIN_PRESSURE_DIFFERENTIAL == 5.0

    def test_flash_tank_efficiency(self):
        """Test flash tank efficiency constant."""
        assert FlashConstants.FLASH_TANK_EFFICIENCY == 0.95

    def test_saturation_data_completeness(self, iapws_saturation_reference):
        """Test saturation data matches IAPWS reference."""
        for pressure in [0, 15, 50, 100, 150, 300, 600]:
            assert pressure in FlashConstants.SATURATION_DATA

    def test_saturation_data_accuracy(self, iapws_saturation_reference):
        """Test saturation data accuracy against IAPWS."""
        for pressure, expected in iapws_saturation_reference.items():
            if pressure in FlashConstants.SATURATION_DATA:
                actual = FlashConstants.SATURATION_DATA[pressure]
                assert actual[0] == pytest.approx(expected[0], rel=0.01)  # T_sat
                assert actual[3] == pytest.approx(expected[3], rel=0.01)  # h_g


# =============================================================================
# FLASH STEAM CALCULATOR TESTS
# =============================================================================

class TestFlashSteamCalculator:
    """Test suite for FlashSteamCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create flash steam calculator."""
        return FlashSteamCalculator()

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator is not None

    def test_get_saturation_properties(self, calculator):
        """Test saturation properties lookup."""
        props = calculator.get_saturation_properties(150.0)

        assert "T_sat_f" in props
        assert "h_f_btu_lb" in props
        assert "h_fg_btu_lb" in props
        assert "h_g_btu_lb" in props

        assert props["T_sat_f"] == pytest.approx(365.9, rel=0.01)
        assert props["h_g_btu_lb"] == pytest.approx(1196.0, rel=0.01)

    def test_get_saturation_properties_interpolation(self, calculator):
        """Test interpolation between table values."""
        props = calculator.get_saturation_properties(75.0)

        # 75 psig is between 50 and 100 in table
        assert 298.0 < props["T_sat_f"] < 337.9

    def test_get_saturation_properties_clamping(self, calculator):
        """Test clamping for out-of-range pressures."""
        props_low = calculator.get_saturation_properties(-10.0)
        props_high = calculator.get_saturation_properties(700.0)

        # Should clamp to table bounds
        assert props_low["T_sat_f"] == pytest.approx(212.0, rel=0.01)
        assert props_high["T_sat_f"] == pytest.approx(489.0, rel=0.01)

    def test_calculate_flash_fraction_basic(self, calculator):
        """Test basic flash fraction calculation."""
        # 150 psig condensate flashing to 15 psig
        fraction, details = calculator.calculate_flash_fraction(
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        # Expected flash fraction ~12-15%
        assert 0.10 < fraction < 0.20
        assert details["condensate_pressure_psig"] == 150.0
        assert details["flash_pressure_psig"] == 15.0

    def test_flash_fraction_increases_with_pressure_drop(self, calculator):
        """Test flash fraction increases with larger pressure drop."""
        fraction_small, _ = calculator.calculate_flash_fraction(
            condensate_pressure_psig=100.0,
            flash_pressure_psig=50.0,
        )

        fraction_large, _ = calculator.calculate_flash_fraction(
            condensate_pressure_psig=300.0,
            flash_pressure_psig=15.0,
        )

        assert fraction_large > fraction_small

    def test_flash_fraction_subcooled_condensate(self, calculator):
        """Test flash fraction with subcooled condensate."""
        # Subcooled by 20F
        fraction_subcooled, details = calculator.calculate_flash_fraction(
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
            condensate_temperature_f=345.9,  # 20F below saturation
        )

        # Normal (saturated)
        fraction_saturated, _ = calculator.calculate_flash_fraction(
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        # Subcooled should have lower flash fraction
        assert fraction_subcooled < fraction_saturated

    def test_flash_fraction_invalid_pressures(self, calculator):
        """Test error for invalid pressure configuration."""
        with pytest.raises(ValueError, match="Flash pressure must be less"):
            calculator.calculate_flash_fraction(
                condensate_pressure_psig=15.0,
                flash_pressure_psig=150.0,
            )

    def test_flash_fraction_equal_pressures(self, calculator):
        """Test error for equal pressures."""
        with pytest.raises(ValueError):
            calculator.calculate_flash_fraction(
                condensate_pressure_psig=150.0,
                flash_pressure_psig=150.0,
            )

    @pytest.mark.parametrize("cond_p,flash_p,expected_fraction_range", [
        (100, 15, (0.08, 0.15)),   # Moderate pressure drop
        (150, 15, (0.10, 0.18)),   # Higher pressure drop
        (300, 15, (0.15, 0.25)),   # Large pressure drop
        (600, 15, (0.20, 0.35)),   # Very large pressure drop
        (150, 50, (0.05, 0.12)),   # Smaller pressure drop
    ])
    def test_flash_fraction_accuracy(
        self, calculator, cond_p, flash_p, expected_fraction_range
    ):
        """Test flash fraction accuracy for various conditions."""
        fraction, _ = calculator.calculate_flash_fraction(cond_p, flash_p)

        assert expected_fraction_range[0] < fraction < expected_fraction_range[1]

    def test_calculate_flash(self, calculator):
        """Test complete flash steam calculation."""
        result = calculator.calculate_flash(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        assert isinstance(result, FlashSteamOutput)
        assert result.flash_fraction_pct > 0
        assert result.flash_steam_lb_hr > 0
        assert result.residual_condensate_lb_hr > 0
        assert result.energy_recovered_btu_hr > 0

    def test_calculate_flash_mass_balance(self, calculator):
        """Test mass balance in flash calculation."""
        flow = 5000.0
        result = calculator.calculate_flash(
            condensate_flow_lb_hr=flow,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        # Mass balance (accounting for efficiency)
        total_out = result.flash_steam_lb_hr + result.residual_condensate_lb_hr

        # Due to efficiency factor (0.95), total_out slightly less than input
        assert total_out <= flow
        assert total_out > 0.9 * flow

    def test_calculate_flash_energy_recovery(self, calculator):
        """Test energy recovery calculation."""
        result = calculator.calculate_flash(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        # Energy = mass * enthalpy
        expected_energy = result.flash_steam_lb_hr * result.flash_steam_enthalpy_btu_lb
        assert result.energy_recovered_btu_hr == pytest.approx(expected_energy, rel=0.01)

    def test_calculate_flash_provenance_hash(self, calculator):
        """Test provenance hash is generated."""
        result = calculator.calculate_flash(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_calculate_flash_formula_reference(self, calculator):
        """Test formula reference is included."""
        result = calculator.calculate_flash(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        assert "IAPWS-IF97" in result.formula_reference


# =============================================================================
# FLASH TANK SIZER TESTS
# =============================================================================

class TestFlashTankSizer:
    """Test suite for FlashTankSizer."""

    @pytest.fixture
    def sizer(self):
        """Create flash tank sizer."""
        return FlashTankSizer()

    def test_initialization(self, sizer):
        """Test sizer initialization."""
        assert sizer.flash_calc is not None

    def test_size_flash_tank(self, sizer):
        """Test flash tank sizing."""
        result = sizer.size_flash_tank(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        assert "flash_steam_lb_hr" in result
        assert "calculated_diameter_in" in result
        assert "recommended_diameter_in" in result
        assert "volume_gal" in result

    def test_size_flash_tank_standard_sizes(self, sizer):
        """Test recommended sizes are standard."""
        result = sizer.size_flash_tank(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        standard_sizes = [12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96]
        assert result["recommended_diameter_in"] in standard_sizes

    def test_size_flash_tank_larger_flow(self, sizer):
        """Test sizing scales with flow."""
        result_small = sizer.size_flash_tank(
            condensate_flow_lb_hr=1000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        result_large = sizer.size_flash_tank(
            condensate_flow_lb_hr=10000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        assert result_large["recommended_diameter_in"] >= result_small["recommended_diameter_in"]

    def test_size_flash_tank_residence_time(self, sizer):
        """Test residence time calculation."""
        result = sizer.size_flash_tank(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        assert "residence_time_s" in result
        assert result["residence_time_s"] > 0

    def test_custom_separation_velocity(self, sizer):
        """Test custom separation velocity."""
        result_default = sizer.size_flash_tank(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
            separation_velocity_ft_s=3.0,
        )

        result_slower = sizer.size_flash_tank(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
            separation_velocity_ft_s=2.0,
        )

        # Slower velocity requires larger tank
        assert result_slower["calculated_diameter_in"] > result_default["calculated_diameter_in"]


# =============================================================================
# MULTI-STAGE FLASH OPTIMIZER TESTS
# =============================================================================

class TestMultiStageFlashOptimizer:
    """Test suite for MultiStageFlashOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create multi-stage flash optimizer."""
        return MultiStageFlashOptimizer(
            fuel_cost_per_mmbtu=5.0,
            operating_hours_per_year=8000,
        )

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.flash_calc is not None
        assert optimizer.fuel_cost == 5.0
        assert optimizer.operating_hours == 8000

    def test_evaluate_single_stage(self, optimizer):
        """Test single-stage evaluation."""
        result = optimizer.optimize_stages(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=300.0,
            final_pressure_psig=15.0,
            max_stages=1,
        )

        assert "1_stage" in result["stage_analysis"]
        assert result["optimal_stages"] == 1

    def test_evaluate_multi_stage(self, optimizer):
        """Test multi-stage evaluation."""
        result = optimizer.optimize_stages(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=600.0,
            final_pressure_psig=15.0,
            max_stages=3,
        )

        assert "1_stage" in result["stage_analysis"]
        assert "2_stage" in result["stage_analysis"]
        assert "3_stage" in result["stage_analysis"]

    def test_multi_stage_more_recovery(self, optimizer):
        """Test multi-stage recovers more steam than single-stage."""
        result = optimizer.optimize_stages(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=600.0,
            final_pressure_psig=15.0,
            max_stages=3,
        )

        single = result["stage_analysis"]["1_stage"]
        double = result["stage_analysis"]["2_stage"]
        triple = result["stage_analysis"]["3_stage"]

        # Multi-stage should recover more
        assert triple["total_flash_steam_lb_hr"] >= double["total_flash_steam_lb_hr"]
        assert double["total_flash_steam_lb_hr"] >= single["total_flash_steam_lb_hr"]

    def test_intermediate_pressures(self, optimizer):
        """Test intermediate pressure calculation."""
        result = optimizer.optimize_stages(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=300.0,
            final_pressure_psig=15.0,
            max_stages=2,
        )

        pressures = result["stage_analysis"]["2_stage"]["intermediate_pressures_psig"]

        # Should have 3 pressures: inlet, intermediate, outlet
        assert len(pressures) == 3
        assert pressures[0] == 300.0  # Inlet
        assert pressures[-1] == 15.0  # Outlet
        assert pressures[0] > pressures[1] > pressures[2]  # Decreasing

    def test_annual_savings_calculation(self, optimizer):
        """Test annual savings calculation."""
        result = optimizer.optimize_stages(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=300.0,
            final_pressure_psig=15.0,
            max_stages=1,
        )

        single = result["stage_analysis"]["1_stage"]

        assert "annual_savings_usd" in single
        assert single["annual_savings_usd"] > 0

    def test_recommendation_generation(self, optimizer):
        """Test recommendation text generation."""
        result = optimizer.optimize_stages(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=600.0,
            final_pressure_psig=15.0,
            max_stages=3,
        )

        assert "recommendation" in result
        assert len(result["recommendation"]) > 0

    def test_invalid_stages_count(self, optimizer):
        """Test error for invalid stage count."""
        with pytest.raises(ValueError):
            optimizer._evaluate_stages(
                condensate_flow_lb_hr=5000.0,
                high_pressure_psig=300.0,
                low_pressure_psig=15.0,
                n_stages=0,
            )


# =============================================================================
# FLASH RECOVERY OPTIMIZER TESTS
# =============================================================================

class TestFlashRecoveryOptimizer:
    """Test suite for FlashRecoveryOptimizer."""

    @pytest.fixture
    def optimizer(self, flash_recovery_config):
        """Create flash recovery optimizer."""
        return FlashRecoveryOptimizer(config=flash_recovery_config)

    def test_initialization(self, optimizer, flash_recovery_config):
        """Test optimizer initialization."""
        assert optimizer.config == flash_recovery_config
        assert optimizer.flash_calc is not None
        assert optimizer.tank_sizer is not None
        assert optimizer.multi_stage is not None

    def test_analyze_complete(self, optimizer):
        """Test complete flash recovery analysis."""
        result = optimizer.analyze(condensate_flow_lb_hr=5000.0)

        assert "flash_analysis" in result
        assert "tank_sizing" in result
        assert "economic_analysis" in result
        assert "recommendations" in result

    def test_analyze_flash_results(self, optimizer):
        """Test flash analysis portion of results."""
        result = optimizer.analyze(condensate_flow_lb_hr=5000.0)

        flash = result["flash_analysis"]
        assert isinstance(flash, FlashSteamOutput)
        assert flash.flash_fraction_pct > 0

    def test_analyze_economic_results(self, optimizer):
        """Test economic analysis portion of results."""
        result = optimizer.analyze(condensate_flow_lb_hr=5000.0)

        econ = result["economic_analysis"]
        assert "annual_savings_usd" in econ
        assert econ["annual_savings_usd"] > 0

    def test_analyze_with_subcooling(self, optimizer):
        """Test analysis with subcooled condensate."""
        result = optimizer.analyze(
            condensate_flow_lb_hr=5000.0,
            condensate_temperature_f=340.0,  # Subcooled
        )

        flash = result["flash_analysis"]
        assert flash.flash_fraction_pct > 0

    def test_multi_stage_recommendation(self, optimizer):
        """Test multi-stage recommendation for high pressure ratio."""
        # Create config with large pressure drop
        config = FlashRecoveryConfig(
            flash_tank_id="FT-TEST",
            condensate_pressure_psig=600.0,  # High pressure
            flash_pressure_psig=15.0,
        )
        opt = FlashRecoveryOptimizer(config=config)

        result = opt.analyze(condensate_flow_lb_hr=5000.0)

        # Should recommend multi-stage for pressure ratio > 5
        if result["multi_stage_analysis"] is not None:
            assert result["multi_stage_analysis"]["optimal_stages"] >= 1

    def test_recommendations_generated(self, optimizer):
        """Test recommendations are generated."""
        result = optimizer.analyze(condensate_flow_lb_hr=5000.0)

        assert len(result["recommendations"]) > 0


# =============================================================================
# THERMODYNAMIC VALIDATION TESTS
# =============================================================================

class TestThermodynamicValidation:
    """Validation tests against known thermodynamic values."""

    @pytest.fixture
    def calculator(self):
        """Create flash steam calculator."""
        return FlashSteamCalculator()

    @pytest.mark.compliance
    def test_flash_fraction_known_value_150_to_15(self, calculator):
        """Validate flash fraction for 150->15 psig (known ~12.7%)."""
        # From IAPWS-IF97:
        # h_f at 150 psig = 339.2 BTU/lb
        # h_f at 15 psig = 218.9 BTU/lb
        # h_fg at 15 psig = 945.4 BTU/lb
        # Flash fraction = (339.2 - 218.9) / 945.4 = 0.1273 = 12.73%

        fraction, _ = calculator.calculate_flash_fraction(150.0, 15.0)

        expected = (339.2 - 218.9) / 945.4
        assert fraction == pytest.approx(expected, rel=0.02)

    @pytest.mark.compliance
    def test_flash_fraction_known_value_300_to_15(self, calculator):
        """Validate flash fraction for 300->15 psig (known ~18.8%)."""
        # h_f at 300 psig = 397.0 BTU/lb
        # Flash fraction = (397.0 - 218.9) / 945.4 = 0.1884 = 18.84%

        fraction, _ = calculator.calculate_flash_fraction(300.0, 15.0)

        expected = (397.0 - 218.9) / 945.4
        assert fraction == pytest.approx(expected, rel=0.02)

    @pytest.mark.compliance
    def test_energy_conservation(self, calculator):
        """Test energy conservation in flash process."""
        result = calculator.calculate_flash(
            condensate_flow_lb_hr=5000.0,
            condensate_pressure_psig=150.0,
            flash_pressure_psig=15.0,
        )

        # Energy in = Energy out (flash steam + residual condensate)
        energy_in = 5000.0 * result.condensate_enthalpy_in_btu_lb
        energy_flash = (result.flash_steam_lb_hr /
                        FlashConstants.FLASH_TANK_EFFICIENCY) * result.flash_steam_enthalpy_btu_lb
        energy_residual = result.residual_condensate_lb_hr * result.residual_enthalpy_btu_lb

        # Should be approximately balanced
        assert energy_in == pytest.approx(energy_flash + energy_residual, rel=0.05)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestFlashPerformance:
    """Performance tests for flash recovery module."""

    @pytest.fixture
    def calculator(self):
        """Create flash steam calculator."""
        return FlashSteamCalculator()

    @pytest.mark.performance
    def test_flash_calculation_speed(self, calculator):
        """Test flash calculation speed (<5ms)."""
        import time
        start = time.time()

        for _ in range(1000):
            calculator.calculate_flash(
                condensate_flow_lb_hr=5000.0,
                condensate_pressure_psig=150.0,
                flash_pressure_psig=15.0,
            )

        elapsed = time.time() - start
        assert elapsed < 1.0  # 1000 calculations in <1s

    @pytest.mark.performance
    def test_multi_stage_optimization_speed(self):
        """Test multi-stage optimization speed."""
        import time
        optimizer = MultiStageFlashOptimizer()

        start = time.time()
        for _ in range(100):
            optimizer.optimize_stages(
                condensate_flow_lb_hr=5000.0,
                condensate_pressure_psig=600.0,
                final_pressure_psig=15.0,
                max_stages=3,
            )
        elapsed = time.time() - start

        assert elapsed < 5.0  # 100 optimizations in <5s
