# -*- coding: utf-8 -*-
"""
GL-007 Heat Transfer Calculator Tests
=====================================

Unit tests for GL-007 heat transfer calculations.
Tests radiant, convective, LMTD, and overall HTC calculations.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
import math

from greenlang.agents.process_heat.gl_007_furnace_optimizer.heat_transfer import (
    FurnaceHeatTransfer,
    HeatTransferConstants,
    create_heat_transfer_calculator,
)
from greenlang.agents.process_heat.gl_007_furnace_optimizer.schemas import (
    HeatTransferAnalysis,
)


class TestHeatTransferConstants:
    """Tests for heat transfer engineering constants."""

    def test_stefan_boltzmann_constant(self):
        """Test Stefan-Boltzmann constant value."""
        # Standard value: 0.1714e-8 Btu/hr-ft2-R^4
        assert HeatTransferConstants.STEFAN_BOLTZMANN_BTU_HR_FT2_R4 > 0
        assert abs(HeatTransferConstants.STEFAN_BOLTZMANN_BTU_HR_FT2_R4 - 0.1714e-8) < 1e-10

    def test_rankine_offset(self):
        """Test Rankine temperature offset."""
        assert HeatTransferConstants.RANKINE_OFFSET == 459.67

    def test_emissivity_values(self):
        """Test emissivity constants are in valid range."""
        assert 0 < HeatTransferConstants.EMISSIVITY_REFRACTORY <= 1
        assert 0 < HeatTransferConstants.EMISSIVITY_STEEL_OXIDIZED <= 1
        assert 0 < HeatTransferConstants.EMISSIVITY_STEEL_CLEAN <= 1
        assert 0 < HeatTransferConstants.EMISSIVITY_FLAME <= 1

    def test_thermal_conductivities(self):
        """Test thermal conductivity constants."""
        assert HeatTransferConstants.K_REFRACTORY > 0
        assert HeatTransferConstants.K_CARBON_STEEL > 0
        # Steel conducts better than refractory
        assert HeatTransferConstants.K_CARBON_STEEL > HeatTransferConstants.K_REFRACTORY

    def test_htc_values(self):
        """Test heat transfer coefficient constants."""
        # Natural convection < forced convection
        assert (HeatTransferConstants.HTC_NATURAL_CONVECTION_AIR <
                HeatTransferConstants.HTC_FORCED_CONVECTION_AIR)
        # Boiling > single phase
        assert (HeatTransferConstants.HTC_BOILING_WATER >
                HeatTransferConstants.HTC_PROCESS_GAS)

    def test_fouling_factors(self):
        """Test fouling factor constants."""
        assert HeatTransferConstants.FOULING_CLEAN < HeatTransferConstants.FOULING_LIGHT
        assert HeatTransferConstants.FOULING_LIGHT < HeatTransferConstants.FOULING_MODERATE
        assert HeatTransferConstants.FOULING_MODERATE < HeatTransferConstants.FOULING_HEAVY
        assert HeatTransferConstants.FOULING_HEAVY < HeatTransferConstants.FOULING_SEVERE


class TestFurnaceHeatTransfer:
    """Tests for FurnaceHeatTransfer class."""

    @pytest.fixture
    def calculator(self):
        """Create heat transfer calculator instance."""
        return FurnaceHeatTransfer(provenance_enabled=True)

    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator is not None
        assert calculator.VERSION == "1.0.0"
        assert calculator.provenance_enabled is True

    def test_initialization_without_provenance(self):
        """Test calculator without provenance tracking."""
        calc = FurnaceHeatTransfer(provenance_enabled=False)
        assert calc._provenance_tracker is None


class TestRadiantHeatTransfer:
    """Tests for radiant heat transfer calculations."""

    @pytest.fixture
    def calculator(self):
        """Create heat transfer calculator instance."""
        return FurnaceHeatTransfer()

    def test_radiant_heat_basic(self, calculator):
        """Test basic radiant heat transfer calculation."""
        result = calculator.calculate_radiant_heat_transfer(
            gas_temp_f=2500.0,
            surface_temp_f=800.0,
            area_ft2=500.0,
        )

        assert "q_radiant_btu_hr" in result
        assert "q_radiant_mmbtu_hr" in result
        assert "htc_radiant_btu_hr_ft2_f" in result
        assert result["q_radiant_btu_hr"] > 0

    def test_radiant_heat_increases_with_temp(self, calculator):
        """Test radiant heat increases with temperature difference."""
        result_low = calculator.calculate_radiant_heat_transfer(
            gas_temp_f=2000.0,
            surface_temp_f=800.0,
            area_ft2=500.0,
        )
        result_high = calculator.calculate_radiant_heat_transfer(
            gas_temp_f=3000.0,
            surface_temp_f=800.0,
            area_ft2=500.0,
        )

        # Higher gas temp = more radiant heat
        assert result_high["q_radiant_btu_hr"] > result_low["q_radiant_btu_hr"]

    def test_radiant_heat_increases_with_area(self, calculator):
        """Test radiant heat increases with area."""
        result_small = calculator.calculate_radiant_heat_transfer(
            gas_temp_f=2500.0,
            surface_temp_f=800.0,
            area_ft2=250.0,
        )
        result_large = calculator.calculate_radiant_heat_transfer(
            gas_temp_f=2500.0,
            surface_temp_f=800.0,
            area_ft2=500.0,
        )

        # Double area = double heat
        ratio = result_large["q_radiant_btu_hr"] / result_small["q_radiant_btu_hr"]
        assert abs(ratio - 2.0) < 0.1

    def test_radiant_heat_emissivity_effect(self, calculator):
        """Test emissivity effect on radiant heat."""
        result_low = calculator.calculate_radiant_heat_transfer(
            gas_temp_f=2500.0,
            surface_temp_f=800.0,
            area_ft2=500.0,
            emissivity=0.5,
        )
        result_high = calculator.calculate_radiant_heat_transfer(
            gas_temp_f=2500.0,
            surface_temp_f=800.0,
            area_ft2=500.0,
            emissivity=1.0,
        )

        # Higher emissivity = more heat transfer
        assert result_high["q_radiant_btu_hr"] > result_low["q_radiant_btu_hr"]

    def test_radiant_heat_view_factor_effect(self, calculator):
        """Test view factor effect on radiant heat."""
        result_partial = calculator.calculate_radiant_heat_transfer(
            gas_temp_f=2500.0,
            surface_temp_f=800.0,
            area_ft2=500.0,
            view_factor=0.5,
        )
        result_full = calculator.calculate_radiant_heat_transfer(
            gas_temp_f=2500.0,
            surface_temp_f=800.0,
            area_ft2=500.0,
            view_factor=1.0,
        )

        # View factor = 0.5 means half the heat
        ratio = result_partial["q_radiant_btu_hr"] / result_full["q_radiant_btu_hr"]
        assert abs(ratio - 0.5) < 0.1


class TestConvectiveHeatTransfer:
    """Tests for convective heat transfer calculations."""

    @pytest.fixture
    def calculator(self):
        """Create heat transfer calculator instance."""
        return FurnaceHeatTransfer()

    def test_convective_heat_basic(self, calculator):
        """Test basic convective heat transfer calculation."""
        result = calculator.calculate_convective_heat_transfer(
            gas_temp_f=500.0,
            surface_temp_f=200.0,
            area_ft2=1000.0,
            htc_btu_hr_ft2_f=5.0,
        )

        assert "q_convective_btu_hr" in result
        assert "q_convective_mmbtu_hr" in result
        assert result["q_convective_btu_hr"] > 0

    def test_convective_heat_calculation(self, calculator):
        """Test convective heat calculation accuracy."""
        result = calculator.calculate_convective_heat_transfer(
            gas_temp_f=500.0,
            surface_temp_f=200.0,
            area_ft2=1000.0,
            htc_btu_hr_ft2_f=5.0,
        )

        # Q = h * A * dT = 5 * 1000 * 300 = 1,500,000 Btu/hr
        expected = 5.0 * 1000.0 * (500.0 - 200.0)
        assert abs(result["q_convective_btu_hr"] - expected) < 1.0

    def test_convective_heat_increases_with_htc(self, calculator):
        """Test convective heat increases with HTC."""
        result_low = calculator.calculate_convective_heat_transfer(
            gas_temp_f=500.0,
            surface_temp_f=200.0,
            area_ft2=1000.0,
            htc_btu_hr_ft2_f=3.0,
        )
        result_high = calculator.calculate_convective_heat_transfer(
            gas_temp_f=500.0,
            surface_temp_f=200.0,
            area_ft2=1000.0,
            htc_btu_hr_ft2_f=10.0,
        )

        assert result_high["q_convective_btu_hr"] > result_low["q_convective_btu_hr"]


class TestLMTDCalculation:
    """Tests for Log Mean Temperature Difference calculations."""

    @pytest.fixture
    def calculator(self):
        """Create heat transfer calculator instance."""
        return FurnaceHeatTransfer()

    def test_lmtd_counterflow(self, calculator):
        """Test LMTD for counterflow arrangement."""
        result = calculator.calculate_lmtd(
            t_hot_in_f=500.0,
            t_hot_out_f=300.0,
            t_cold_in_f=100.0,
            t_cold_out_f=250.0,
            flow_arrangement="counterflow",
        )

        assert "lmtd_f" in result
        assert result["lmtd_f"] > 0
        assert result["valid"] is True

    def test_lmtd_parallel_flow(self, calculator):
        """Test LMTD for parallel flow arrangement."""
        result = calculator.calculate_lmtd(
            t_hot_in_f=500.0,
            t_hot_out_f=300.0,
            t_cold_in_f=100.0,
            t_cold_out_f=200.0,
            flow_arrangement="parallel",
        )

        assert result["lmtd_f"] > 0
        assert result["valid"] is True

    def test_lmtd_calculation_accuracy(self, calculator):
        """Test LMTD calculation accuracy."""
        # dT1 = 500 - 250 = 250, dT2 = 300 - 100 = 200
        # LMTD = (250 - 200) / ln(250/200) = 50 / 0.223 = 224.1
        result = calculator.calculate_lmtd(
            t_hot_in_f=500.0,
            t_hot_out_f=300.0,
            t_cold_in_f=100.0,
            t_cold_out_f=250.0,
            flow_arrangement="counterflow",
        )

        dt1 = 500.0 - 250.0  # 250
        dt2 = 300.0 - 100.0  # 200
        expected_lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

        assert abs(result["lmtd_f"] - expected_lmtd) < 1.0

    def test_lmtd_equal_dt(self, calculator):
        """Test LMTD when dT1 equals dT2."""
        result = calculator.calculate_lmtd(
            t_hot_in_f=500.0,
            t_hot_out_f=300.0,
            t_cold_in_f=100.0,
            t_cold_out_f=300.0,
            flow_arrangement="counterflow",
        )

        # When dT1 = dT2, LMTD = dT (arithmetic mean)
        assert result["lmtd_f"] > 0
        assert result["valid"] is True

    def test_lmtd_temperature_cross(self, calculator):
        """Test LMTD with temperature cross (invalid)."""
        result = calculator.calculate_lmtd(
            t_hot_in_f=200.0,
            t_hot_out_f=100.0,
            t_cold_in_f=150.0,
            t_cold_out_f=300.0,  # Cold out > Hot in = cross
            flow_arrangement="counterflow",
        )

        assert result["valid"] is False


class TestOverallHTC:
    """Tests for overall heat transfer coefficient calculations."""

    @pytest.fixture
    def calculator(self):
        """Create heat transfer calculator instance."""
        return FurnaceHeatTransfer()

    def test_overall_htc_basic(self, calculator):
        """Test basic overall HTC calculation."""
        result = calculator.calculate_overall_htc(
            htc_inside_btu_hr_ft2_f=100.0,
            htc_outside_btu_hr_ft2_f=5.0,
        )

        assert "u_overall_btu_hr_ft2_f" in result
        assert "u_clean_btu_hr_ft2_f" in result
        assert result["u_overall_btu_hr_ft2_f"] > 0

    def test_overall_htc_limited_by_lowest(self, calculator):
        """Test overall HTC is limited by lowest coefficient."""
        result = calculator.calculate_overall_htc(
            htc_inside_btu_hr_ft2_f=100.0,  # High
            htc_outside_btu_hr_ft2_f=5.0,   # Low
        )

        # Overall HTC should be less than the lower value
        assert result["u_overall_btu_hr_ft2_f"] < 5.0

    def test_overall_htc_fouling_effect(self, calculator):
        """Test fouling reduces overall HTC."""
        result_clean = calculator.calculate_overall_htc(
            htc_inside_btu_hr_ft2_f=100.0,
            htc_outside_btu_hr_ft2_f=5.0,
            fouling_inside=0.0,
            fouling_outside=0.0,
        )
        result_fouled = calculator.calculate_overall_htc(
            htc_inside_btu_hr_ft2_f=100.0,
            htc_outside_btu_hr_ft2_f=5.0,
            fouling_inside=0.002,
            fouling_outside=0.002,
        )

        # Fouling reduces HTC
        assert result_fouled["u_overall_btu_hr_ft2_f"] < result_clean["u_overall_btu_hr_ft2_f"]

    def test_overall_htc_resistance_breakdown(self, calculator):
        """Test resistance breakdown sums to 100%."""
        result = calculator.calculate_overall_htc(
            htc_inside_btu_hr_ft2_f=100.0,
            htc_outside_btu_hr_ft2_f=5.0,
            fouling_inside=0.001,
            fouling_outside=0.001,
        )

        total_pct = (
            result["pct_inside_resistance"]
            + result["pct_fouling_resistance"]
            + result["pct_wall_resistance"]
            + result["pct_outside_resistance"]
        )

        assert abs(total_pct - 100.0) < 1.0


class TestWallLoss:
    """Tests for wall heat loss calculations."""

    @pytest.fixture
    def calculator(self):
        """Create heat transfer calculator instance."""
        return FurnaceHeatTransfer()

    def test_wall_loss_basic(self, calculator):
        """Test basic wall loss calculation."""
        result = calculator.calculate_wall_loss(
            t_inside_f=1800.0,
            t_ambient_f=77.0,
            wall_area_ft2=1000.0,
        )

        assert "q_loss_btu_hr_ft2" in result
        assert "q_loss_total_mmbtu_hr" in result
        assert result["q_loss_total_btu_hr"] > 0

    def test_wall_loss_increases_with_temp(self, calculator):
        """Test wall loss increases with temperature difference."""
        result_low = calculator.calculate_wall_loss(
            t_inside_f=1000.0,
            t_ambient_f=77.0,
            wall_area_ft2=1000.0,
        )
        result_high = calculator.calculate_wall_loss(
            t_inside_f=2000.0,
            t_ambient_f=77.0,
            wall_area_ft2=1000.0,
        )

        assert result_high["q_loss_total_btu_hr"] > result_low["q_loss_total_btu_hr"]

    def test_wall_loss_with_custom_layers(self, calculator):
        """Test wall loss with custom insulation layers."""
        # Better insulation = lower loss
        result_thin = calculator.calculate_wall_loss(
            t_inside_f=1800.0,
            t_ambient_f=77.0,
            wall_area_ft2=1000.0,
            wall_layers=[
                {"thickness_in": 2.0, "conductivity_btu_hr_ft_f": 0.5},
            ],
        )
        result_thick = calculator.calculate_wall_loss(
            t_inside_f=1800.0,
            t_ambient_f=77.0,
            wall_area_ft2=1000.0,
            wall_layers=[
                {"thickness_in": 6.0, "conductivity_btu_hr_ft_f": 0.5},
            ],
        )

        assert result_thick["q_loss_total_btu_hr"] < result_thin["q_loss_total_btu_hr"]

    def test_wall_surface_temperature(self, calculator):
        """Test outside wall surface temperature."""
        result = calculator.calculate_wall_loss(
            t_inside_f=1800.0,
            t_ambient_f=77.0,
            wall_area_ft2=1000.0,
        )

        # Surface temp should be between inside and ambient
        assert 77.0 < result["t_wall_outside_f"] < 1800.0


class TestHeatTransferAnalysis:
    """Tests for comprehensive heat transfer analysis."""

    @pytest.fixture
    def calculator(self):
        """Create heat transfer calculator instance."""
        return FurnaceHeatTransfer()

    def test_analyze_heat_transfer_basic(self, calculator):
        """Test basic heat transfer analysis."""
        result = calculator.analyze_heat_transfer(
            design_duty_mmbtu_hr=50.0,
            actual_duty_mmbtu_hr=43.5,
            furnace_temp_f=1800.0,
            flue_gas_temp_f=450.0,
            process_inlet_temp_f=200.0,
            process_outlet_temp_f=600.0,
            radiant_area_ft2=500.0,
            convective_area_ft2=1000.0,
        )

        assert isinstance(result, HeatTransferAnalysis)
        assert result.analysis_id is not None
        assert result.design_duty_mmbtu_hr == 50.0
        assert result.actual_duty_mmbtu_hr == 43.5

    def test_analyze_heat_transfer_duty_ratio(self, calculator):
        """Test duty ratio calculation."""
        result = calculator.analyze_heat_transfer(
            design_duty_mmbtu_hr=50.0,
            actual_duty_mmbtu_hr=43.5,
            furnace_temp_f=1800.0,
            flue_gas_temp_f=450.0,
            process_inlet_temp_f=200.0,
            process_outlet_temp_f=600.0,
            radiant_area_ft2=500.0,
            convective_area_ft2=1000.0,
        )

        expected_ratio = 43.5 / 50.0 * 100
        assert abs(result.duty_ratio_pct - expected_ratio) < 0.1

    def test_analyze_heat_transfer_fouling_detection(self, calculator):
        """Test fouling detection based on HTC degradation."""
        result = calculator.analyze_heat_transfer(
            design_duty_mmbtu_hr=50.0,
            actual_duty_mmbtu_hr=35.0,  # Significantly lower
            furnace_temp_f=1800.0,
            flue_gas_temp_f=450.0,
            process_inlet_temp_f=200.0,
            process_outlet_temp_f=600.0,
            radiant_area_ft2=500.0,
            convective_area_ft2=1000.0,
            design_htc_btu_hr_ft2_f=8.0,
        )

        assert result.fouling_severity in ["clean", "light", "moderate", "heavy", "severe"]

    def test_analyze_heat_transfer_provenance(self, calculator):
        """Test provenance hash is generated."""
        result = calculator.analyze_heat_transfer(
            design_duty_mmbtu_hr=50.0,
            actual_duty_mmbtu_hr=43.5,
            furnace_temp_f=1800.0,
            flue_gas_temp_f=450.0,
            process_inlet_temp_f=200.0,
            process_outlet_temp_f=600.0,
            radiant_area_ft2=500.0,
            convective_area_ft2=1000.0,
        )

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_analyze_heat_transfer_recommendations(self, calculator):
        """Test recommendations are generated."""
        result = calculator.analyze_heat_transfer(
            design_duty_mmbtu_hr=50.0,
            actual_duty_mmbtu_hr=30.0,  # Low duty suggests problems
            furnace_temp_f=1800.0,
            flue_gas_temp_f=500.0,
            process_inlet_temp_f=200.0,
            process_outlet_temp_f=600.0,
            radiant_area_ft2=500.0,
            convective_area_ft2=1000.0,
        )

        assert isinstance(result.recommendations, list)


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_heat_transfer_calculator_default(self):
        """Test default factory creation."""
        calc = create_heat_transfer_calculator()
        assert calc is not None
        assert calc.provenance_enabled is True

    def test_create_heat_transfer_calculator_custom(self):
        """Test custom factory creation."""
        calc = create_heat_transfer_calculator(
            provenance_enabled=False,
            precision=2,
        )
        assert calc.provenance_enabled is False
        assert calc.precision == 2


class TestDeterminism:
    """Tests for deterministic calculation behavior."""

    @pytest.fixture
    def calculator(self):
        """Create heat transfer calculator instance."""
        return FurnaceHeatTransfer()

    def test_radiant_heat_determinism(self, calculator):
        """Test radiant heat calculation is deterministic."""
        results = []
        for _ in range(5):
            result = calculator.calculate_radiant_heat_transfer(
                gas_temp_f=2500.0,
                surface_temp_f=800.0,
                area_ft2=500.0,
            )
            results.append(result["q_radiant_btu_hr"])

        assert len(set(results)) == 1  # All results identical

    def test_lmtd_determinism(self, calculator):
        """Test LMTD calculation is deterministic."""
        results = []
        for _ in range(5):
            result = calculator.calculate_lmtd(
                t_hot_in_f=500.0,
                t_hot_out_f=300.0,
                t_cold_in_f=100.0,
                t_cold_out_f=250.0,
            )
            results.append(result["lmtd_f"])

        assert len(set(results)) == 1  # All results identical

    def test_analysis_determinism(self, calculator):
        """Test full analysis is deterministic."""
        results = []
        for _ in range(3):
            result = calculator.analyze_heat_transfer(
                design_duty_mmbtu_hr=50.0,
                actual_duty_mmbtu_hr=43.5,
                furnace_temp_f=1800.0,
                flue_gas_temp_f=450.0,
                process_inlet_temp_f=200.0,
                process_outlet_temp_f=600.0,
                radiant_area_ft2=500.0,
                convective_area_ft2=1000.0,
            )
            results.append(result.lmtd_f)

        assert len(set(results)) == 1  # All results identical
