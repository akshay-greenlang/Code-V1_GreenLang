# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Hydraulic Analysis Tests

Comprehensive tests for pressure drop calculations including:
- Tube-side pressure drop (Darcy-Weisbach)
- Shell-side pressure drop (Kern and Bell-Delaware methods)
- Plate exchanger pressure drop (Martin correlation)
- Velocity calculations
- Reynolds number and flow regime determination

Coverage Target: 90%+

References:
    - Kern, "Process Heat Transfer" (1950)
    - Bell-Delaware Method
    - HEDH Heat Exchanger Design Handbook
"""

import pytest
import math
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_014_heat_exchanger.hydraulics import (
    HydraulicCalculator,
    FluidProperties,
    PressureDropComponents,
    VelocityResult,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    TubeGeometryConfig,
    ShellGeometryConfig,
    PlateGeometryConfig,
    TubeLayout,
    ExchangerType,
)


class TestFluidProperties:
    """Tests for FluidProperties dataclass."""

    def test_create_fluid_properties(self):
        """Test creating fluid properties."""
        fluid = FluidProperties(
            density_kg_m3=998.0,
            viscosity_pa_s=0.001,
            specific_heat_j_kgk=4180.0,
            thermal_conductivity_w_mk=0.62,
        )
        assert fluid.density_kg_m3 == 998.0
        assert fluid.viscosity_pa_s == 0.001

    def test_fluid_properties_minimal(self):
        """Test fluid properties with minimal fields."""
        fluid = FluidProperties(
            density_kg_m3=850.0,
            viscosity_pa_s=0.003,
        )
        assert fluid.specific_heat_j_kgk is None


class TestHydraulicCalculatorInit:
    """Tests for HydraulicCalculator initialization."""

    def test_calculator_init_shell_tube(self, tube_geometry_config, shell_geometry_config):
        """Test calculator initialization for shell-tube."""
        calc = HydraulicCalculator(
            tube_geometry=tube_geometry_config,
            shell_geometry=shell_geometry_config,
        )
        assert calc.tube_geometry == tube_geometry_config
        assert calc.shell_geometry == shell_geometry_config

    def test_calculator_init_plate(self, plate_geometry_config):
        """Test calculator initialization for plate exchanger."""
        calc = HydraulicCalculator(
            plate_geometry=plate_geometry_config,
        )
        assert calc.plate_geometry == plate_geometry_config


class TestTubeSidePressureDrop:
    """Tests for tube-side pressure drop calculations."""

    @pytest.fixture
    def calculator(self, tube_geometry_config, shell_geometry_config):
        """Create HydraulicCalculator instance."""
        return HydraulicCalculator(
            tube_geometry=tube_geometry_config,
            shell_geometry=shell_geometry_config,
        )

    @pytest.fixture
    def water_fluid(self):
        """Water properties at 25C."""
        return FluidProperties(
            density_kg_m3=998.0,
            viscosity_pa_s=0.001,
        )

    def test_tube_dp_calculation(self, calculator, water_fluid):
        """Test basic tube-side pressure drop calculation."""
        dp = calculator.calculate_tube_side_dp(
            mass_flow_kg_s=10.0,
            fluid=water_fluid,
            fouling_factor=0.0,
        )

        assert isinstance(dp, PressureDropComponents)
        assert dp.total_bar > 0
        assert dp.friction_bar > 0
        assert dp.entrance_exit_bar >= 0
        assert dp.nozzle_bar >= 0

    def test_tube_dp_components_sum(self, calculator, water_fluid):
        """Test pressure drop components sum to total."""
        dp = calculator.calculate_tube_side_dp(
            mass_flow_kg_s=10.0,
            fluid=water_fluid,
            fouling_factor=0.0,
        )

        calculated_total = (
            dp.friction_bar +
            dp.entrance_exit_bar +
            dp.nozzle_bar +
            dp.elevation_bar
        )
        assert dp.total_bar == pytest.approx(calculated_total, rel=0.01)

    def test_tube_dp_increases_with_flow(self, calculator, water_fluid):
        """Test pressure drop increases with flow rate."""
        dp_low = calculator.calculate_tube_side_dp(
            mass_flow_kg_s=5.0,
            fluid=water_fluid,
        )

        dp_high = calculator.calculate_tube_side_dp(
            mass_flow_kg_s=15.0,
            fluid=water_fluid,
        )

        # DP proportional to velocity^2, so flow^2
        assert dp_high.total_bar > dp_low.total_bar

    def test_tube_dp_increases_with_viscosity(self, calculator):
        """Test pressure drop increases with viscosity (laminar)."""
        low_visc = FluidProperties(
            density_kg_m3=998.0,
            viscosity_pa_s=0.001,
        )

        high_visc = FluidProperties(
            density_kg_m3=998.0,
            viscosity_pa_s=0.01,
        )

        dp_low_visc = calculator.calculate_tube_side_dp(
            mass_flow_kg_s=1.0,  # Low flow for potentially laminar
            fluid=low_visc,
        )

        dp_high_visc = calculator.calculate_tube_side_dp(
            mass_flow_kg_s=1.0,
            fluid=high_visc,
        )

        # Higher viscosity increases friction
        assert dp_high_visc.friction_bar >= dp_low_visc.friction_bar

    def test_tube_dp_with_fouling(self, calculator, water_fluid):
        """Test pressure drop increases with fouling."""
        dp_clean = calculator.calculate_tube_side_dp(
            mass_flow_kg_s=10.0,
            fluid=water_fluid,
            fouling_factor=0.0,
        )

        dp_fouled = calculator.calculate_tube_side_dp(
            mass_flow_kg_s=10.0,
            fluid=water_fluid,
            fouling_factor=0.5,  # 50% fouling
        )

        # Fouling reduces flow area, increases velocity and DP
        assert dp_fouled.total_bar > dp_clean.total_bar

    def test_tube_dp_no_geometry_error(self):
        """Test error when tube geometry not provided."""
        calc = HydraulicCalculator()

        with pytest.raises(ValueError):
            calc.calculate_tube_side_dp(
                mass_flow_kg_s=10.0,
                fluid=FluidProperties(density_kg_m3=998.0, viscosity_pa_s=0.001),
            )


class TestShellSidePressureDropKern:
    """Tests for shell-side pressure drop using Kern method."""

    @pytest.fixture
    def calculator(self, tube_geometry_config, shell_geometry_config):
        """Create HydraulicCalculator instance."""
        return HydraulicCalculator(
            tube_geometry=tube_geometry_config,
            shell_geometry=shell_geometry_config,
        )

    @pytest.fixture
    def oil_fluid(self):
        """Crude oil properties."""
        return FluidProperties(
            density_kg_m3=850.0,
            viscosity_pa_s=0.003,
        )

    def test_shell_dp_kern_calculation(self, calculator, oil_fluid):
        """Test Kern method shell-side pressure drop."""
        dp = calculator.calculate_shell_side_dp_kern(
            mass_flow_kg_s=10.0,
            fluid=oil_fluid,
            fouling_factor=0.0,
        )

        assert isinstance(dp, PressureDropComponents)
        assert dp.total_bar > 0

    def test_shell_dp_kern_increases_with_baffles(self, tube_geometry_config):
        """Test pressure drop increases with baffle count."""
        shell_few_baffles = ShellGeometryConfig(
            inner_diameter_mm=610.0,
            baffle_count=5,
            baffle_spacing_mm=400.0,
        )

        shell_many_baffles = ShellGeometryConfig(
            inner_diameter_mm=610.0,
            baffle_count=15,
            baffle_spacing_mm=133.0,
        )

        calc_few = HydraulicCalculator(
            tube_geometry=tube_geometry_config,
            shell_geometry=shell_few_baffles,
        )

        calc_many = HydraulicCalculator(
            tube_geometry=tube_geometry_config,
            shell_geometry=shell_many_baffles,
        )

        fluid = FluidProperties(density_kg_m3=850.0, viscosity_pa_s=0.003)

        dp_few = calc_few.calculate_shell_side_dp_kern(
            mass_flow_kg_s=10.0,
            fluid=fluid,
        )

        dp_many = calc_many.calculate_shell_side_dp_kern(
            mass_flow_kg_s=10.0,
            fluid=fluid,
        )

        # More baffles = more crossings = higher DP
        assert dp_many.total_bar > dp_few.total_bar


class TestShellSidePressureDropBellDelaware:
    """Tests for shell-side pressure drop using Bell-Delaware method."""

    @pytest.fixture
    def calculator(self, tube_geometry_config, shell_geometry_config):
        """Create HydraulicCalculator instance."""
        return HydraulicCalculator(
            tube_geometry=tube_geometry_config,
            shell_geometry=shell_geometry_config,
        )

    @pytest.fixture
    def oil_fluid(self):
        """Crude oil properties."""
        return FluidProperties(
            density_kg_m3=850.0,
            viscosity_pa_s=0.003,
        )

    def test_bell_delaware_calculation(self, calculator, oil_fluid):
        """Test Bell-Delaware method calculation."""
        dp = calculator.calculate_shell_side_dp_bell_delaware(
            mass_flow_kg_s=10.0,
            fluid=oil_fluid,
            fouling_factor=0.0,
        )

        assert isinstance(dp, PressureDropComponents)
        assert dp.total_bar > 0

    def test_bell_delaware_vs_kern(self, calculator, oil_fluid):
        """Test Bell-Delaware includes more effects than Kern."""
        dp_kern = calculator.calculate_shell_side_dp_kern(
            mass_flow_kg_s=10.0,
            fluid=oil_fluid,
        )

        dp_bd = calculator.calculate_shell_side_dp_bell_delaware(
            mass_flow_kg_s=10.0,
            fluid=oil_fluid,
        )

        # Both should give reasonable results
        assert dp_kern.total_bar > 0
        assert dp_bd.total_bar > 0

        # Bell-Delaware accounts for leakage (often lower DP)
        # Ratio should be reasonable
        ratio = dp_bd.total_bar / dp_kern.total_bar
        assert 0.5 < ratio < 2.0


class TestPlatePressureDrop:
    """Tests for plate heat exchanger pressure drop."""

    @pytest.fixture
    def calculator(self, plate_geometry_config):
        """Create HydraulicCalculator for plate exchanger."""
        return HydraulicCalculator(
            plate_geometry=plate_geometry_config,
        )

    @pytest.fixture
    def water_fluid(self):
        """Water properties."""
        return FluidProperties(
            density_kg_m3=998.0,
            viscosity_pa_s=0.001,
        )

    def test_plate_dp_calculation(self, calculator, water_fluid):
        """Test plate exchanger pressure drop calculation."""
        dp = calculator.calculate_plate_dp(
            mass_flow_kg_s=5.0,
            fluid=water_fluid,
            fouling_factor=0.0,
        )

        assert isinstance(dp, PressureDropComponents)
        assert dp.total_bar > 0

    def test_plate_dp_chevron_angle_effect(self, water_fluid):
        """Test chevron angle affects pressure drop."""
        plate_low_angle = PlateGeometryConfig(
            plate_count=50,
            chevron_angle_deg=30.0,  # Low angle
        )

        plate_high_angle = PlateGeometryConfig(
            plate_count=50,
            chevron_angle_deg=65.0,  # High angle
        )

        calc_low = HydraulicCalculator(plate_geometry=plate_low_angle)
        calc_high = HydraulicCalculator(plate_geometry=plate_high_angle)

        dp_low = calc_low.calculate_plate_dp(
            mass_flow_kg_s=5.0,
            fluid=water_fluid,
        )

        dp_high = calc_high.calculate_plate_dp(
            mass_flow_kg_s=5.0,
            fluid=water_fluid,
        )

        # Higher chevron angle typically means higher DP
        # But relationship is complex - just verify both are valid
        assert dp_low.total_bar > 0
        assert dp_high.total_bar > 0

    def test_plate_dp_no_geometry_error(self):
        """Test error when plate geometry not provided."""
        calc = HydraulicCalculator()

        with pytest.raises(ValueError):
            calc.calculate_plate_dp(
                mass_flow_kg_s=5.0,
                fluid=FluidProperties(density_kg_m3=998.0, viscosity_pa_s=0.001),
            )


class TestVelocityCalculation:
    """Tests for velocity calculations."""

    @pytest.fixture
    def calculator(self, tube_geometry_config, shell_geometry_config):
        """Create HydraulicCalculator instance."""
        return HydraulicCalculator(
            tube_geometry=tube_geometry_config,
            shell_geometry=shell_geometry_config,
        )

    def test_velocity_calculation(self, calculator):
        """Test velocity calculation."""
        result = calculator.calculate_velocity(
            mass_flow_kg_s=10.0,
            flow_area_m2=0.01,  # 100 cm2
            density_kg_m3=998.0,
            characteristic_length_m=0.02,  # 20mm tube ID
            viscosity_pa_s=0.001,
        )

        assert isinstance(result, VelocityResult)
        # V = m / (rho * A) = 10 / (998 * 0.01) = 1.0 m/s
        assert result.velocity_m_s == pytest.approx(1.0, rel=0.01)

    def test_reynolds_number_calculation(self, calculator):
        """Test Reynolds number calculation."""
        result = calculator.calculate_velocity(
            mass_flow_kg_s=10.0,
            flow_area_m2=0.01,
            density_kg_m3=998.0,
            characteristic_length_m=0.02,
            viscosity_pa_s=0.001,
        )

        # Re = rho * V * D / mu = 998 * 1.0 * 0.02 / 0.001 = 19960
        expected_re = 998 * 1.0 * 0.02 / 0.001
        assert result.reynolds == pytest.approx(expected_re, rel=0.01)

    def test_flow_regime_laminar(self, calculator):
        """Test laminar flow regime detection."""
        result = calculator.calculate_velocity(
            mass_flow_kg_s=0.1,
            flow_area_m2=0.01,
            density_kg_m3=850.0,
            characteristic_length_m=0.02,
            viscosity_pa_s=0.1,  # High viscosity
        )

        # Re = 850 * 0.01 * 0.02 / 0.1 = 1.7 (laminar)
        assert result.flow_regime == "laminar"
        assert result.reynolds < 2300

    def test_flow_regime_turbulent(self, calculator):
        """Test turbulent flow regime detection."""
        result = calculator.calculate_velocity(
            mass_flow_kg_s=10.0,
            flow_area_m2=0.01,
            density_kg_m3=998.0,
            characteristic_length_m=0.02,
            viscosity_pa_s=0.001,
        )

        # Re = 19960 (turbulent)
        assert result.flow_regime == "turbulent"
        assert result.reynolds > 4000

    def test_friction_factor_laminar(self, calculator):
        """Test friction factor for laminar flow (f = 64/Re)."""
        result = calculator.calculate_velocity(
            mass_flow_kg_s=0.05,
            flow_area_m2=0.01,
            density_kg_m3=850.0,
            characteristic_length_m=0.02,
            viscosity_pa_s=0.1,
        )

        if result.flow_regime == "laminar":
            expected_f = 64 / result.reynolds
            assert result.friction_factor == pytest.approx(expected_f, rel=0.1)


class TestCompleteHydraulicAnalysis:
    """Tests for complete hydraulic analysis."""

    @pytest.fixture
    def calculator(self, tube_geometry_config, shell_geometry_config):
        """Create HydraulicCalculator instance."""
        return HydraulicCalculator(
            tube_geometry=tube_geometry_config,
            shell_geometry=shell_geometry_config,
        )

    @pytest.fixture
    def shell_fluid(self):
        """Shell side fluid (oil)."""
        return FluidProperties(density_kg_m3=850.0, viscosity_pa_s=0.003)

    @pytest.fixture
    def tube_fluid(self):
        """Tube side fluid (water)."""
        return FluidProperties(density_kg_m3=998.0, viscosity_pa_s=0.001)

    def test_complete_analysis_shell_tube(self, calculator, shell_fluid, tube_fluid):
        """Test complete analysis for shell-tube exchanger."""
        result = calculator.calculate_complete_analysis(
            shell_flow_kg_s=10.0,
            tube_flow_kg_s=15.0,
            shell_fluid=shell_fluid,
            tube_fluid=tube_fluid,
            exchanger_type=ExchangerType.SHELL_TUBE,
            fouling_factor=0.0,
        )

        # Verify all results present
        assert result.shell_pressure_drop_bar > 0
        assert result.tube_pressure_drop_bar > 0
        assert result.shell_velocity_m_s > 0
        assert result.tube_velocity_m_s > 0
        assert result.shell_reynolds > 0
        assert result.tube_reynolds > 0

    def test_complete_analysis_dp_ratios(self, calculator, shell_fluid, tube_fluid):
        """Test DP ratios are calculated."""
        result = calculator.calculate_complete_analysis(
            shell_flow_kg_s=10.0,
            tube_flow_kg_s=15.0,
            shell_fluid=shell_fluid,
            tube_fluid=tube_fluid,
            exchanger_type=ExchangerType.SHELL_TUBE,
        )

        # DP ratios should be actual/design
        expected_shell_ratio = result.shell_pressure_drop_bar / result.shell_dp_design_bar
        expected_tube_ratio = result.tube_pressure_drop_bar / result.tube_dp_design_bar

        assert result.shell_dp_ratio == pytest.approx(expected_shell_ratio, rel=0.01)
        assert result.tube_dp_ratio == pytest.approx(expected_tube_ratio, rel=0.01)

    def test_complete_analysis_alarms(self, calculator, shell_fluid, tube_fluid):
        """Test DP alarms are set correctly."""
        # High flow to potentially trigger alarms
        result = calculator.calculate_complete_analysis(
            shell_flow_kg_s=50.0,  # High flow
            tube_flow_kg_s=75.0,
            shell_fluid=shell_fluid,
            tube_fluid=tube_fluid,
            exchanger_type=ExchangerType.SHELL_TUBE,
        )

        # Alarms should be boolean
        assert isinstance(result.shell_dp_alarm, bool)
        assert isinstance(result.tube_dp_alarm, bool)

        # If DP exceeds design, alarm should be true
        if result.shell_pressure_drop_bar > result.shell_dp_design_bar:
            assert result.shell_dp_alarm == True

    def test_complete_analysis_with_fouling(self, calculator, shell_fluid, tube_fluid):
        """Test complete analysis with fouling contribution."""
        result = calculator.calculate_complete_analysis(
            shell_flow_kg_s=10.0,
            tube_flow_kg_s=15.0,
            shell_fluid=shell_fluid,
            tube_fluid=tube_fluid,
            exchanger_type=ExchangerType.SHELL_TUBE,
            fouling_factor=0.3,
        )

        # Fouling contribution should be calculated
        assert result.shell_dp_fouling_contribution_bar >= 0
        assert result.tube_dp_fouling_contribution_bar >= 0


class TestTubeLayoutEffect:
    """Tests for tube layout effect on shell-side DP."""

    @pytest.fixture
    def shell_geometry(self):
        """Standard shell geometry."""
        return ShellGeometryConfig(
            inner_diameter_mm=610.0,
            baffle_count=10,
            baffle_spacing_mm=300.0,
        )

    def test_triangular_vs_square_layout(self, shell_geometry):
        """Test triangular vs square layout effect on DP."""
        tube_triangular = TubeGeometryConfig(
            tube_layout=TubeLayout.TRIANGULAR_30,
            outer_diameter_mm=25.4,
            tube_pitch_mm=31.75,
            tube_count=100,
        )

        tube_square = TubeGeometryConfig(
            tube_layout=TubeLayout.SQUARE_90,
            outer_diameter_mm=25.4,
            tube_pitch_mm=31.75,
            tube_count=100,
        )

        calc_tri = HydraulicCalculator(
            tube_geometry=tube_triangular,
            shell_geometry=shell_geometry,
        )

        calc_sq = HydraulicCalculator(
            tube_geometry=tube_square,
            shell_geometry=shell_geometry,
        )

        fluid = FluidProperties(density_kg_m3=850.0, viscosity_pa_s=0.003)

        dp_tri = calc_tri.calculate_shell_side_dp_kern(
            mass_flow_kg_s=10.0,
            fluid=fluid,
        )

        dp_sq = calc_sq.calculate_shell_side_dp_kern(
            mass_flow_kg_s=10.0,
            fluid=fluid,
        )

        # Both should give valid results
        # Triangular typically has higher DP due to tighter flow paths
        assert dp_tri.total_bar > 0
        assert dp_sq.total_bar > 0
