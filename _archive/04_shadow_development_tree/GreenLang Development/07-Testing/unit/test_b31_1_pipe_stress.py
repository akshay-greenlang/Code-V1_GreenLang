"""
Unit Tests for ASME B31.1 Pipe Stress Calculations

Tests for power piping stress analysis per ASME B31.1 including:
- Hoop stress from internal pressure
- Longitudinal stress calculations
- Sustained stress per Equation 11A
- Thermal expansion stress range per Equation 13
- Allowable stress range per Equation 1
- Minimum wall thickness per Equation 3
- Complete stress analysis

Reference: ASME B31.1-2022 Power Piping

Author: GreenLang Engineering Team
License: MIT
"""

import pytest
from decimal import Decimal
import math

from greenlang.calculations.asme.b31_1_pipe_stress import (
    ASMEB311PipeStress,
    B311StressResult,
    MinimumThicknessResult,
    PipeGeometry,
    LoadData,
    PipeMaterial,
    PipeSchedule,
    LoadCategory,
    pipe_hoop_stress,
    pipe_sustained_stress,
    pipe_expansion_stress,
    pipe_allowable_stress_range,
    pipe_minimum_thickness,
    analyze_pipe_stress,
)


class TestHoopStress:
    """Test hoop stress calculations."""

    def test_basic_hoop_stress_calculation(self):
        """
        Test basic hoop stress from internal pressure.

        Reference: Barlow formula S_h = P * D_o / (2 * t)
        """
        calc = ASMEB311PipeStress()

        # NPS 12 Sch 40 pipe at 10 MPa
        # D_o = 323.9 mm, t = 9.53 mm
        stress = calc.calculate_hoop_stress(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53
        )

        # Expected: S_h = 10 * 323.9 / (2 * 9.53) = 169.9 MPa
        expected = Decimal("10") * Decimal("323.9") / (Decimal("2") * Decimal("9.53"))

        assert abs(float(stress) - float(expected)) < 0.5

    def test_hoop_stress_convenience_function(self):
        """Test convenience function for hoop stress."""
        stress = pipe_hoop_stress(10.0, 323.9, 9.53)

        assert stress > Decimal("0")
        assert stress < Decimal("300")  # Reasonable range

    def test_hoop_stress_zero_thickness_raises(self):
        """Test that zero wall thickness raises error."""
        calc = ASMEB311PipeStress()

        with pytest.raises(ValueError, match="Wall thickness must be positive"):
            calc.calculate_hoop_stress(10.0, 323.9, 0.0)

    def test_hoop_stress_proportionality(self):
        """Test hoop stress is proportional to pressure."""
        calc = ASMEB311PipeStress()

        stress_10 = calc.calculate_hoop_stress(10.0, 323.9, 9.53)
        stress_20 = calc.calculate_hoop_stress(20.0, 323.9, 9.53)

        # Doubling pressure should double stress
        ratio = float(stress_20) / float(stress_10)
        assert abs(ratio - 2.0) < 0.01


class TestLongitudinalStress:
    """Test longitudinal stress calculations."""

    def test_pressure_only_longitudinal_stress(self):
        """
        Test longitudinal stress from pressure only.

        S_L = P * D_o / (4 * t) for pressure only
        """
        calc = ASMEB311PipeStress()

        stress = calc.calculate_longitudinal_stress(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53,
            axial_force_n=0.0,
            bending_moment_nm=0.0
        )

        # Expected: S_L = 10 * 323.9 / (4 * 9.53) = 84.95 MPa
        # This is exactly half the hoop stress
        hoop = calc.calculate_hoop_stress(10.0, 323.9, 9.53)

        assert abs(float(stress) - float(hoop) / 2) < 0.5

    def test_longitudinal_stress_with_bending(self):
        """Test longitudinal stress includes bending contribution."""
        calc = ASMEB311PipeStress()

        stress_no_bending = calc.calculate_longitudinal_stress(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53,
            axial_force_n=0.0,
            bending_moment_nm=0.0
        )

        stress_with_bending = calc.calculate_longitudinal_stress(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53,
            axial_force_n=0.0,
            bending_moment_nm=100000.0  # N-mm
        )

        # With bending, stress should be higher
        assert stress_with_bending > stress_no_bending


class TestSustainedStress:
    """Test sustained stress calculations per B31.1 Eq. 11A."""

    def test_sustained_stress_pressure_only(self):
        """Test sustained stress with pressure only."""
        calc = ASMEB311PipeStress()

        stress = calc.calculate_sustained_stress(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53,
            axial_force_n=0.0,
            bending_moment_nm=0.0,
            stress_intensification_factor=1.0
        )

        # Should equal pressure longitudinal stress
        p_stress = Decimal("10") * Decimal("323.9") / (Decimal("4") * Decimal("9.53"))
        assert abs(float(stress) - float(p_stress)) < 0.5

    def test_sustained_stress_with_sif(self):
        """Test that SIF increases bending stress contribution."""
        calc = ASMEB311PipeStress()

        stress_sif_1 = calc.calculate_sustained_stress(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53,
            axial_force_n=0.0,
            bending_moment_nm=50000.0,
            stress_intensification_factor=1.0
        )

        stress_sif_2 = calc.calculate_sustained_stress(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53,
            axial_force_n=0.0,
            bending_moment_nm=50000.0,
            stress_intensification_factor=2.0
        )

        # Higher SIF should increase stress
        assert stress_sif_2 > stress_sif_1

    def test_sustained_stress_convenience_function(self):
        """Test convenience function for sustained stress."""
        stress = pipe_sustained_stress(10.0, 323.9, 9.53, 0, 50000, 1.5)

        assert stress > Decimal("0")


class TestExpansionStressRange:
    """Test expansion stress range calculations per B31.1 Eq. 13."""

    def test_bending_only_expansion_stress(self):
        """Test expansion stress from bending only."""
        calc = ASMEB311PipeStress()

        stress = calc.calculate_expansion_stress_range(
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53,
            thermal_bending_moment_nm=100000.0,
            torsional_moment_nm=0.0,
            stress_intensification_factor=1.0
        )

        # S_E = sqrt(S_b^2 + 4*S_t^2) with S_t = 0
        # S_E = S_b = i * M_c / Z
        assert stress > Decimal("0")

    def test_combined_expansion_stress(self):
        """
        Test combined bending and torsion.

        S_E = sqrt(S_b^2 + 4*S_t^2)
        """
        calc = ASMEB311PipeStress()

        stress_bending_only = calc.calculate_expansion_stress_range(
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53,
            thermal_bending_moment_nm=100000.0,
            torsional_moment_nm=0.0,
            stress_intensification_factor=1.0
        )

        stress_combined = calc.calculate_expansion_stress_range(
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53,
            thermal_bending_moment_nm=100000.0,
            torsional_moment_nm=50000.0,
            stress_intensification_factor=1.0
        )

        # Combined should be higher
        assert stress_combined > stress_bending_only

    def test_expansion_stress_convenience_function(self):
        """Test convenience function for expansion stress."""
        stress = pipe_expansion_stress(323.9, 9.53, 100000, 20000, 1.5)

        assert stress > Decimal("0")


class TestAllowableStressRange:
    """Test allowable stress range calculations per B31.1 Eq. 1."""

    def test_basic_allowable_range(self):
        """
        Test basic allowable stress range.

        S_A = f * (1.25 * S_c + 0.25 * S_h)
        """
        calc = ASMEB311PipeStress()

        s_a = calc.calculate_allowable_stress_range(
            allowable_stress_cold_mpa=117.9,
            allowable_stress_hot_mpa=93.1,
            stress_reduction_factor=1.0
        )

        # Expected: 1.0 * (1.25 * 117.9 + 0.25 * 93.1) = 170.65 MPa
        expected = Decimal("1.0") * (
            Decimal("1.25") * Decimal("117.9") +
            Decimal("0.25") * Decimal("93.1")
        )

        assert abs(float(s_a) - float(expected)) < 0.5

    def test_stress_reduction_factor_effect(self):
        """Test that stress reduction factor reduces allowable range."""
        calc = ASMEB311PipeStress()

        s_a_f1 = calc.calculate_allowable_stress_range(117.9, 93.1, 1.0)
        s_a_f09 = calc.calculate_allowable_stress_range(117.9, 93.1, 0.9)

        # f = 0.9 should reduce by 10%
        ratio = float(s_a_f09) / float(s_a_f1)
        assert abs(ratio - 0.9) < 0.01

    def test_stress_reduction_factor_lookup(self):
        """Test stress reduction factor table lookup."""
        calc = ASMEB311PipeStress()

        # Per B31.1 Table 102.3.2(c)
        assert calc.get_stress_reduction_factor(7000) == Decimal("1.0")
        assert calc.get_stress_reduction_factor(10000) == Decimal("0.9")
        assert calc.get_stress_reduction_factor(20000) == Decimal("0.8")
        assert calc.get_stress_reduction_factor(40000) == Decimal("0.7")
        assert calc.get_stress_reduction_factor(80000) == Decimal("0.6")
        assert calc.get_stress_reduction_factor(200000) == Decimal("0.5")

    def test_invalid_f_factor_raises(self):
        """Test that invalid f factor raises error."""
        calc = ASMEB311PipeStress()

        with pytest.raises(ValueError, match="Stress reduction factor"):
            calc.calculate_allowable_stress_range(117.9, 93.1, 0.0)

        with pytest.raises(ValueError, match="Stress reduction factor"):
            calc.calculate_allowable_stress_range(117.9, 93.1, 1.5)


class TestMinimumWallThickness:
    """Test minimum wall thickness calculations per B31.1 Eq. 3."""

    def test_basic_thickness_calculation(self):
        """
        Test basic minimum thickness.

        t = P * D_o / (2 * (S*E + P*y))
        """
        calc = ASMEB311PipeStress()

        result = calc.calculate_minimum_wall_thickness(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            allowable_stress_mpa=117.9,
            joint_efficiency=1.0,
            y_factor=0.4,
            mill_tolerance_percent=12.5,
            corrosion_allowance_mm=0.0
        )

        assert isinstance(result, MinimumThicknessResult)
        assert result.pressure_design_thickness_mm > Decimal("0")
        assert result.total_minimum_thickness_mm > result.pressure_design_thickness_mm

    def test_thickness_with_corrosion_allowance(self):
        """Test that corrosion allowance increases thickness."""
        calc = ASMEB311PipeStress()

        result_no_ca = calc.calculate_minimum_wall_thickness(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            allowable_stress_mpa=117.9,
            corrosion_allowance_mm=0.0
        )

        result_with_ca = calc.calculate_minimum_wall_thickness(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            allowable_stress_mpa=117.9,
            corrosion_allowance_mm=3.0
        )

        # Corrosion allowance should add directly
        diff = float(result_with_ca.required_thickness_mm -
                    result_no_ca.required_thickness_mm)
        assert abs(diff - 3.0) < 0.1

    def test_joint_efficiency_effect(self):
        """Test that lower joint efficiency increases required thickness."""
        calc = ASMEB311PipeStress()

        result_e1 = calc.calculate_minimum_wall_thickness(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            allowable_stress_mpa=117.9,
            joint_efficiency=1.0
        )

        result_e085 = calc.calculate_minimum_wall_thickness(
            internal_pressure_mpa=10.0,
            outside_diameter_mm=323.9,
            allowable_stress_mpa=117.9,
            joint_efficiency=0.85
        )

        # Lower efficiency should require more thickness
        assert result_e085.pressure_design_thickness_mm > result_e1.pressure_design_thickness_mm

    def test_convenience_function(self):
        """Test convenience function for minimum thickness."""
        result = pipe_minimum_thickness(10.0, 323.9, 117.9, 1.5)

        assert isinstance(result, MinimumThicknessResult)
        assert result.provenance_hash != ""


class TestCompleteStressAnalysis:
    """Test complete stress analysis."""

    def test_complete_analysis_acceptable(self):
        """Test complete analysis for acceptable piping."""
        calc = ASMEB311PipeStress()

        geometry = PipeGeometry(
            outside_diameter_mm=323.9,
            wall_thickness_mm=15.0,  # Thick pipe
            mill_tolerance_percent=12.5,
            corrosion_allowance_mm=0.0
        )

        loads = LoadData(
            internal_pressure_mpa=5.0,  # Moderate pressure
            bending_moment_nm=10000.0,
            thermal_bending_moment_nm=50000.0
        )

        result = calc.analyze_piping_stress(
            geometry=geometry,
            loads=loads,
            material=PipeMaterial.CARBON_STEEL_A106_B,
            design_temperature_c=260.0,  # 500F
            stress_intensification_factor=1.0
        )

        assert isinstance(result, B311StressResult)
        assert result.sustained_acceptable
        assert result.expansion_acceptable
        assert result.overall_acceptable
        assert result.provenance_hash != ""

    def test_complete_analysis_high_stress(self):
        """Test complete analysis with high stress conditions."""
        calc = ASMEB311PipeStress()

        geometry = PipeGeometry(
            outside_diameter_mm=323.9,
            wall_thickness_mm=5.0,  # Thin pipe
            mill_tolerance_percent=12.5
        )

        loads = LoadData(
            internal_pressure_mpa=15.0,  # High pressure
            bending_moment_nm=100000.0,  # High bending
            thermal_bending_moment_nm=200000.0
        )

        result = calc.analyze_piping_stress(
            geometry=geometry,
            loads=loads,
            material=PipeMaterial.CARBON_STEEL_A106_B,
            design_temperature_c=371.0,  # 700F
            stress_intensification_factor=2.0  # High SIF
        )

        # At least one stress should exceed limits
        assert (
            result.sustained_stress_ratio > Decimal("1") or
            result.expansion_stress_ratio > Decimal("1") or
            not result.overall_acceptable
        )

    def test_stress_ratios_consistency(self):
        """Test that stress ratios are calculated correctly."""
        result = analyze_pipe_stress(
            od_mm=323.9,
            wall_mm=10.0,
            pressure_mpa=8.0,
            material="carbon_steel_a106_b",
            temperature_c=316.0,
            bending_moment_nm=20000,
            thermal_moment_nm=80000,
            sif=1.5
        )

        # Stress ratio = actual / allowable
        sustained_ratio = (
            float(result.sustained_stress_mpa) /
            float(result.allowable_sustained_mpa)
        )
        assert abs(sustained_ratio - float(result.sustained_stress_ratio)) < 0.01

    def test_different_materials(self):
        """Test analysis with different materials."""
        calc = ASMEB311PipeStress()

        geometry = PipeGeometry(
            outside_diameter_mm=219.1,
            wall_thickness_mm=8.18
        )

        loads = LoadData(
            internal_pressure_mpa=12.0,
            thermal_bending_moment_nm=80000.0
        )

        # Carbon steel at moderate temp
        result_cs = calc.analyze_piping_stress(
            geometry=geometry,
            loads=loads,
            material=PipeMaterial.CARBON_STEEL_A106_B,
            design_temperature_c=316.0
        )

        # P22 (higher allowable at elevated temp)
        result_p22 = calc.analyze_piping_stress(
            geometry=geometry,
            loads=loads,
            material=PipeMaterial.LOW_ALLOY_P22,
            design_temperature_c=482.0  # Higher temp
        )

        # P22 should have higher allowables at elevated temp
        assert result_p22.allowable_stress_hot_mpa > Decimal("50")


class TestMaterialProperties:
    """Test material property lookups."""

    def test_allowable_stress_lookup(self):
        """Test allowable stress interpolation."""
        calc = ASMEB311PipeStress()

        # At table temperature
        s = calc.get_allowable_stress(PipeMaterial.CARBON_STEEL_A106_B, 260.0)
        assert s == Decimal("117.90")  # Should match table

        # Between table temperatures
        s = calc.get_allowable_stress(PipeMaterial.CARBON_STEEL_A106_B, 350.0)
        assert Decimal("100") < s < Decimal("115")

    def test_elastic_modulus_lookup(self):
        """Test elastic modulus interpolation."""
        calc = ASMEB311PipeStress()

        e = calc.get_elastic_modulus(PipeMaterial.CARBON_STEEL_A106_B, 21.0)
        assert Decimal("200000") < e < Decimal("210000")

        # Higher temp = lower modulus
        e_hot = calc.get_elastic_modulus(PipeMaterial.CARBON_STEEL_A106_B, 400.0)
        e_cold = calc.get_elastic_modulus(PipeMaterial.CARBON_STEEL_A106_B, 21.0)
        assert e_hot < e_cold


class TestProvenanceTracking:
    """Test provenance hash generation."""

    def test_provenance_hash_generated(self):
        """Test that provenance hash is generated."""
        result = analyze_pipe_stress(
            od_mm=323.9,
            wall_mm=9.53,
            pressure_mpa=10.0
        )

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_provenance_hash_deterministic(self):
        """Test that same inputs produce same hash."""
        result1 = analyze_pipe_stress(
            od_mm=323.9,
            wall_mm=9.53,
            pressure_mpa=10.0,
            bending_moment_nm=50000
        )

        result2 = analyze_pipe_stress(
            od_mm=323.9,
            wall_mm=9.53,
            pressure_mpa=10.0,
            bending_moment_nm=50000
        )

        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_hash_changes_with_input(self):
        """Test that different inputs produce different hash."""
        result1 = analyze_pipe_stress(
            od_mm=323.9,
            wall_mm=9.53,
            pressure_mpa=10.0
        )

        result2 = analyze_pipe_stress(
            od_mm=323.9,
            wall_mm=9.53,
            pressure_mpa=11.0  # Different pressure
        )

        assert result1.provenance_hash != result2.provenance_hash


class TestPipeGeometry:
    """Test PipeGeometry dataclass."""

    def test_geometry_properties(self):
        """Test calculated geometry properties."""
        geom = PipeGeometry(
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53,
            mill_tolerance_percent=12.5,
            corrosion_allowance_mm=1.5
        )

        # Inside diameter
        assert abs(geom.inside_diameter_mm - (323.9 - 2*9.53)) < 0.01

        # Mean diameter
        assert abs(geom.mean_diameter_mm - (323.9 - 9.53)) < 0.01

        # Minimum wall
        expected_min = 9.53 * (1 - 0.125)
        assert abs(geom.minimum_wall_thickness_mm - expected_min) < 0.01

        # Effective wall
        expected_eff = expected_min - 1.5
        assert abs(geom.effective_wall_thickness_mm - expected_eff) < 0.01

    def test_metal_area(self):
        """Test metal cross-sectional area calculation."""
        geom = PipeGeometry(
            outside_diameter_mm=323.9,
            wall_thickness_mm=9.53
        )

        d_o = 323.9
        d_i = 323.9 - 2*9.53
        expected_area = math.pi / 4 * (d_o**2 - d_i**2)

        assert abs(geom.metal_area_mm2 - expected_area) < 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_thin_wall(self):
        """Test with very thin wall thickness."""
        calc = ASMEB311PipeStress()

        stress = calc.calculate_hoop_stress(
            internal_pressure_mpa=1.0,
            outside_diameter_mm=100.0,
            wall_thickness_mm=0.5  # Very thin
        )

        # Should still calculate
        assert stress > Decimal("0")

    def test_very_high_pressure(self):
        """Test with high pressure."""
        calc = ASMEB311PipeStress()

        stress = calc.calculate_hoop_stress(
            internal_pressure_mpa=50.0,  # High pressure
            outside_diameter_mm=100.0,
            wall_thickness_mm=10.0
        )

        assert stress > Decimal("200")  # Should be high

    def test_temperature_extrapolation(self):
        """Test material properties at extreme temperatures."""
        calc = ASMEB311PipeStress()

        # Very low temperature (uses lowest table value)
        s_low = calc.get_allowable_stress(PipeMaterial.CARBON_STEEL_A106_B, -10.0)
        assert s_low > Decimal("0")

        # Very high temperature (uses highest table value)
        s_high = calc.get_allowable_stress(PipeMaterial.CARBON_STEEL_A106_B, 600.0)
        assert s_high > Decimal("0")
        assert s_high < s_low  # Higher temp = lower allowable
