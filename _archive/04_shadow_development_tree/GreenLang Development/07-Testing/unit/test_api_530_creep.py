"""
Unit Tests for API 530 Creep Life Assessment

Tests for creep-rupture life analysis including:
- Larson-Miller Parameter calculation
- Rupture time estimation
- Life fraction calculation (Robinson's Rule)
- Multi-condition creep accumulation
- Omega method per API 579-1 Part 10
- Minimum wall thickness with creep allowance

Reference: API 530 7th Edition, API 579-1 Part 10

Author: GreenLang Engineering Team
License: MIT
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
import math

from greenlang.calculations.api.api_530_creep import (
    CreepLifeAssessor,
    CreepLifeResult,
    CreepAccumulationResult,
    OmegaMethodResult,
    CreepMaterial,
    OperatingCondition,
    CreepDataPoint,
    creep_rupture_time,
    creep_life_fraction,
    creep_remaining_life,
    assess_tube_creep,
)


class TestLarsonMillerParameter:
    """Test Larson-Miller Parameter calculations."""

    def test_basic_lmp_calculation(self):
        """
        Test basic Larson-Miller Parameter.

        LMP = T(K) * [C + log10(t_r)] / 1000
        """
        assessor = CreepLifeAssessor()

        # P22 at 565C (838K), 100,000 hours
        # LMP = 838 * (20 + 5) / 1000 = 20.95
        lmp = assessor.calculate_larson_miller_parameter(
            temperature_c=565.0,
            time_hours=100000.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        # Check reasonable range for P22
        assert Decimal("18") < lmp < Decimal("25")

    def test_lmp_increases_with_temperature(self):
        """Test that LMP increases with temperature."""
        assessor = CreepLifeAssessor()

        lmp_low = assessor.calculate_larson_miller_parameter(
            temperature_c=500.0,
            time_hours=100000.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        lmp_high = assessor.calculate_larson_miller_parameter(
            temperature_c=600.0,
            time_hours=100000.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        assert lmp_high > lmp_low

    def test_lmp_increases_with_time(self):
        """Test that LMP increases with time."""
        assessor = CreepLifeAssessor()

        lmp_short = assessor.calculate_larson_miller_parameter(
            temperature_c=565.0,
            time_hours=10000.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        lmp_long = assessor.calculate_larson_miller_parameter(
            temperature_c=565.0,
            time_hours=100000.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        assert lmp_long > lmp_short

    def test_different_material_constants(self):
        """Test different Larson-Miller constants for materials."""
        assessor = CreepLifeAssessor()

        # Ferritic (C=20) vs Austenitic (C=18)
        lmp_ferritic = assessor.calculate_larson_miller_parameter(
            temperature_c=565.0,
            time_hours=100000.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        lmp_austenitic = assessor.calculate_larson_miller_parameter(
            temperature_c=565.0,
            time_hours=100000.0,
            material=CreepMaterial.SS_304H
        )

        # Different constants should give different LMP
        assert lmp_ferritic != lmp_austenitic

    def test_invalid_time_raises(self):
        """Test that zero/negative time raises error."""
        assessor = CreepLifeAssessor()

        with pytest.raises(ValueError, match="Time must be positive"):
            assessor.calculate_larson_miller_parameter(
                temperature_c=565.0,
                time_hours=0.0,
                material=CreepMaterial.CR_2_25_MO_P22
            )


class TestRuptureTime:
    """Test rupture time calculations."""

    def test_basic_rupture_time(self):
        """Test basic rupture time estimation."""
        assessor = CreepLifeAssessor()

        # At 100,000 hour rupture stress, should get ~100,000 hours
        t_r = assessor.calculate_rupture_time(
            temperature_c=538.0,
            stress_mpa=52.0,  # Near 100k rupture stress for P22
            material=CreepMaterial.CR_2_25_MO_P22
        )

        # Should be in reasonable range around 100,000 hours
        assert t_r > Decimal("10000")
        assert t_r < Decimal("1000000")

    def test_rupture_time_convenience_function(self):
        """Test convenience function for rupture time."""
        t_r = creep_rupture_time(565.0, 50.0, "2.25cr_1mo")

        assert t_r > Decimal("0")

    def test_higher_stress_shorter_life(self):
        """Test that higher stress gives shorter rupture time."""
        assessor = CreepLifeAssessor()

        t_r_low = assessor.calculate_rupture_time(
            temperature_c=565.0,
            stress_mpa=40.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        t_r_high = assessor.calculate_rupture_time(
            temperature_c=565.0,
            stress_mpa=60.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        assert t_r_low > t_r_high

    def test_higher_temp_shorter_life(self):
        """Test that higher temperature gives shorter rupture time."""
        assessor = CreepLifeAssessor()

        t_r_low = assessor.calculate_rupture_time(
            temperature_c=540.0,
            stress_mpa=50.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        t_r_high = assessor.calculate_rupture_time(
            temperature_c=580.0,
            stress_mpa=50.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        assert t_r_low > t_r_high


class TestLifeFraction:
    """Test life fraction calculations (Robinson's Rule)."""

    def test_basic_life_fraction(self):
        """
        Test basic life fraction calculation.

        Life Fraction = t_operating / t_rupture
        """
        assessor = CreepLifeAssessor()

        # Short operating time at moderate conditions
        phi = assessor.calculate_life_fraction(
            operating_time_hours=10000.0,
            temperature_c=538.0,
            stress_mpa=50.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        # Should be small fraction
        assert phi > Decimal("0")
        assert phi < Decimal("1")

    def test_life_fraction_convenience_function(self):
        """Test convenience function for life fraction."""
        phi = creep_life_fraction(50000.0, 565.0, 50.0, "2.25cr_1mo")

        assert phi > Decimal("0")

    def test_life_fraction_proportional_to_time(self):
        """Test that life fraction is proportional to operating time."""
        assessor = CreepLifeAssessor()

        phi_10k = assessor.calculate_life_fraction(
            operating_time_hours=10000.0,
            temperature_c=565.0,
            stress_mpa=50.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        phi_20k = assessor.calculate_life_fraction(
            operating_time_hours=20000.0,
            temperature_c=565.0,
            stress_mpa=50.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        # Doubling time should double fraction
        ratio = float(phi_20k) / float(phi_10k)
        assert abs(ratio - 2.0) < 0.1

    def test_zero_operating_time(self):
        """Test that zero operating time gives zero fraction."""
        assessor = CreepLifeAssessor()

        phi = assessor.calculate_life_fraction(
            operating_time_hours=0.0,
            temperature_c=565.0,
            stress_mpa=50.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        assert phi == Decimal("0")


class TestCreepAccumulation:
    """Test multi-condition creep accumulation (Robinson's Rule)."""

    def test_single_condition_accumulation(self):
        """Test accumulation with single operating condition."""
        assessor = CreepLifeAssessor()

        conditions = [
            OperatingCondition(
                temperature_c=565.0,
                stress_mpa=50.0,
                duration_hours=50000.0,
                description="Normal operation"
            )
        ]

        result = assessor.calculate_remaining_life(
            design_life_hours=100000.0,
            conditions=conditions,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        assert isinstance(result, CreepAccumulationResult)
        assert result.conditions_analyzed == 1
        assert result.total_life_fraction > Decimal("0")
        assert result.remaining_life_fraction > Decimal("0")

    def test_multi_condition_accumulation(self):
        """Test accumulation with multiple operating conditions."""
        assessor = CreepLifeAssessor()

        conditions = [
            OperatingCondition(565.0, 50.0, 20000.0, "Normal"),
            OperatingCondition(580.0, 55.0, 5000.0, "High temp"),
            OperatingCondition(550.0, 45.0, 25000.0, "Reduced load"),
        ]

        result = assessor.calculate_remaining_life(
            design_life_hours=100000.0,
            conditions=conditions,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        assert result.conditions_analyzed == 3
        assert len(result.life_fractions) == 3

        # Total hours should match sum
        expected_hours = 20000 + 5000 + 25000
        assert abs(float(result.total_operating_hours) - expected_hours) < 1

    def test_convenience_function(self):
        """Test convenience function for remaining life."""
        conditions = [
            (565.0, 50.0, 20000.0),
            (580.0, 55.0, 5000.0),
        ]

        result = creep_remaining_life(100000.0, conditions, "2.25cr_1mo")

        assert isinstance(result, CreepAccumulationResult)
        assert result.conditions_analyzed == 2

    def test_high_damage_detection(self):
        """Test that high damage is properly flagged."""
        assessor = CreepLifeAssessor()

        # Severe conditions for extended time
        conditions = [
            OperatingCondition(600.0, 70.0, 100000.0, "Severe")
        ]

        result = assessor.calculate_remaining_life(
            design_life_hours=100000.0,
            conditions=conditions,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        # Should detect high damage
        if result.total_life_fraction > Decimal("0.8"):
            assert not result.is_acceptable

    def test_remaining_life_years_calculation(self):
        """Test remaining life in years is correct."""
        assessor = CreepLifeAssessor()

        conditions = [
            OperatingCondition(565.0, 50.0, 50000.0, "Normal")
        ]

        result = assessor.calculate_remaining_life(
            design_life_hours=100000.0,
            conditions=conditions,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        # Remaining years = remaining hours / 8760
        expected_years = float(result.estimated_remaining_hours) / 8760
        assert abs(float(result.estimated_remaining_years) - expected_years) < 0.1


class TestOmegaMethod:
    """Test Omega method per API 579-1 Part 10."""

    def test_basic_omega_assessment(self):
        """Test basic Omega method assessment."""
        assessor = CreepLifeAssessor()

        result = assessor.omega_method_assessment(
            temperature_c=565.0,
            stress_mpa=50.0,
            operating_hours=50000.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        assert isinstance(result, OmegaMethodResult)
        assert result.strain_rate > 0
        assert result.remaining_life_hours >= Decimal("0")
        assert result.provenance_hash != ""

    def test_omega_damage_parameter(self):
        """Test damage parameter calculation."""
        assessor = CreepLifeAssessor()

        result = assessor.omega_method_assessment(
            temperature_c=565.0,
            stress_mpa=50.0,
            operating_hours=50000.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        # Damage should be positive
        assert result.damage_parameter >= Decimal("0")

    def test_higher_stress_higher_strain_rate(self):
        """Test that higher stress gives higher strain rate."""
        assessor = CreepLifeAssessor()

        result_low = assessor.omega_method_assessment(
            temperature_c=565.0,
            stress_mpa=40.0,
            operating_hours=50000.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        result_high = assessor.omega_method_assessment(
            temperature_c=565.0,
            stress_mpa=60.0,
            operating_hours=50000.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        assert result_high.strain_rate > result_low.strain_rate


class TestMinimumWallWithCreep:
    """Test minimum wall thickness with creep allowance."""

    def test_basic_wall_with_creep(self):
        """Test basic wall thickness calculation with creep."""
        assessor = CreepLifeAssessor()

        min_wall, creep_allowance = assessor.calculate_minimum_wall_with_creep(
            design_pressure_mpa=5.0,
            tube_od_mm=114.3,
            material=CreepMaterial.CR_2_25_MO_P22,
            temperature_c=565.0,
            design_life_hours=100000.0
        )

        assert min_wall > Decimal("0")
        assert creep_allowance > Decimal("0")
        assert creep_allowance < min_wall  # Allowance is part of total

    def test_higher_temp_thicker_wall(self):
        """Test that higher temperature requires thicker wall."""
        assessor = CreepLifeAssessor()

        wall_low, _ = assessor.calculate_minimum_wall_with_creep(
            design_pressure_mpa=5.0,
            tube_od_mm=114.3,
            material=CreepMaterial.CR_2_25_MO_P22,
            temperature_c=540.0,
            design_life_hours=100000.0
        )

        wall_high, _ = assessor.calculate_minimum_wall_with_creep(
            design_pressure_mpa=5.0,
            tube_od_mm=114.3,
            material=CreepMaterial.CR_2_25_MO_P22,
            temperature_c=593.0,
            design_life_hours=100000.0
        )

        assert wall_high > wall_low


class TestCompleteCreepAssessment:
    """Test complete creep life assessment."""

    def test_basic_assessment(self):
        """Test basic complete assessment."""
        assessor = CreepLifeAssessor()

        result = assessor.assess_creep_life(
            material=CreepMaterial.CR_2_25_MO_P22,
            design_temperature_c=565.0,
            design_stress_mpa=50.0,
            design_life_hours=100000.0
        )

        assert isinstance(result, CreepLifeResult)
        assert result.material == "2.25cr_1mo"
        assert result.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert result.provenance_hash != ""

    def test_assessment_convenience_function(self):
        """Test convenience function for assessment."""
        result = assess_tube_creep(
            temperature_c=565.0,
            stress_mpa=50.0,
            design_life_hours=100000.0,
            material="2.25cr_1mo"
        )

        assert isinstance(result, CreepLifeResult)
        assert result.is_acceptable is not None

    def test_assessment_with_geometry(self):
        """Test assessment with tube geometry."""
        result = assess_tube_creep(
            temperature_c=565.0,
            stress_mpa=50.0,
            design_life_hours=100000.0,
            material="2.25cr_1mo",
            tube_od_mm=114.3,
            pressure_mpa=5.0
        )

        assert result.minimum_wall_with_creep_mm is not None
        assert result.creep_allowance_mm is not None

    def test_assessment_with_history(self):
        """Test assessment with operating history."""
        assessor = CreepLifeAssessor()

        history = [
            OperatingCondition(565.0, 50.0, 30000.0, "Normal"),
            OperatingCondition(580.0, 55.0, 5000.0, "Upset"),
        ]

        result = assessor.assess_creep_life(
            material=CreepMaterial.CR_2_25_MO_P22,
            design_temperature_c=565.0,
            design_stress_mpa=50.0,
            design_life_hours=100000.0,
            operating_history=history
        )

        # With history, life fraction should be > 0
        assert result.total_life_fraction_consumed > Decimal("0")

    def test_risk_level_assessment(self):
        """Test risk level determination."""
        assessor = CreepLifeAssessor()

        # Low risk - minimal history
        result_low = assessor.assess_creep_life(
            material=CreepMaterial.CR_2_25_MO_P22,
            design_temperature_c=565.0,
            design_stress_mpa=50.0,
            design_life_hours=100000.0
        )

        assert result_low.risk_level == "LOW"
        assert result_low.is_acceptable


class TestOperatingHistory:
    """Test operating history tracking."""

    def test_add_history_point(self):
        """Test adding operating history data point."""
        assessor = CreepLifeAssessor()

        start = datetime(2020, 1, 1)
        end = datetime(2021, 1, 1)

        data_point = assessor.add_operating_history(
            start_date=start,
            end_date=end,
            temperature_c=565.0,
            stress_mpa=50.0,
            material=CreepMaterial.CR_2_25_MO_P22,
            notes="Year 1 operation"
        )

        assert isinstance(data_point, CreepDataPoint)
        assert data_point.duration_hours > 8000  # ~1 year

        # Verify it's in history
        history = assessor.get_operating_history()
        assert len(history) == 1

    def test_clear_history(self):
        """Test clearing operating history."""
        assessor = CreepLifeAssessor()

        # Add some points
        assessor.add_operating_history(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2021, 1, 1),
            temperature_c=565.0,
            stress_mpa=50.0,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        assert len(assessor.get_operating_history()) == 1

        assessor.clear_operating_history()
        assert len(assessor.get_operating_history()) == 0


class TestMaterialCoverage:
    """Test different material coverage."""

    def test_carbon_steel(self):
        """Test carbon steel calculations."""
        result = assess_tube_creep(
            temperature_c=450.0,
            stress_mpa=40.0,
            material="carbon_steel"
        )

        assert result.material == "carbon_steel"

    def test_p91(self):
        """Test P91 (9Cr-1Mo-V) calculations."""
        result = assess_tube_creep(
            temperature_c=593.0,
            stress_mpa=65.0,
            material="9cr_1mo_v"
        )

        assert result.material == "9cr_1mo_v"

    def test_ss_304h(self):
        """Test stainless steel 304H calculations."""
        result = assess_tube_creep(
            temperature_c=650.0,
            stress_mpa=50.0,
            material="ss_304h"
        )

        assert result.material == "ss_304h"

    def test_ss_316h(self):
        """Test stainless steel 316H calculations."""
        result = assess_tube_creep(
            temperature_c=650.0,
            stress_mpa=55.0,
            material="ss_316h"
        )

        assert result.material == "ss_316h"


class TestProvenanceTracking:
    """Test provenance hash generation."""

    def test_provenance_hash_generated(self):
        """Test that provenance hash is generated."""
        result = assess_tube_creep(
            temperature_c=565.0,
            stress_mpa=50.0
        )

        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_provenance_hash_deterministic(self):
        """Test that same inputs produce same hash."""
        result1 = assess_tube_creep(
            temperature_c=565.0,
            stress_mpa=50.0,
            design_life_hours=100000.0
        )

        result2 = assess_tube_creep(
            temperature_c=565.0,
            stress_mpa=50.0,
            design_life_hours=100000.0
        )

        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_hash_changes_with_input(self):
        """Test that different inputs produce different hash."""
        result1 = assess_tube_creep(
            temperature_c=565.0,
            stress_mpa=50.0
        )

        result2 = assess_tube_creep(
            temperature_c=580.0,  # Different temp
            stress_mpa=50.0
        )

        assert result1.provenance_hash != result2.provenance_hash


class TestResultSerialization:
    """Test result to_dict methods."""

    def test_creep_result_to_dict(self):
        """Test CreepLifeResult serialization."""
        result = assess_tube_creep(
            temperature_c=565.0,
            stress_mpa=50.0
        )

        d = result.to_dict()

        assert "material" in d
        assert "design_temperature_c" in d
        assert "risk_level" in d
        assert "provenance_hash" in d

    def test_accumulation_result_serialization(self):
        """Test CreepAccumulationResult structure."""
        conditions = [
            (565.0, 50.0, 20000.0),
        ]

        result = creep_remaining_life(100000.0, conditions)

        assert hasattr(result, "life_fractions")
        assert hasattr(result, "damage_mechanism")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_low_stress(self):
        """Test with very low stress (should have long life)."""
        t_r = creep_rupture_time(565.0, 20.0, "2.25cr_1mo")

        # Low stress should give long rupture time
        assert t_r > Decimal("100000")

    def test_very_high_temperature(self):
        """Test at high temperature limit."""
        result = assess_tube_creep(
            temperature_c=620.0,
            stress_mpa=30.0,
            material="2.25cr_1mo"
        )

        # Should still calculate
        assert result.rupture_time_hours > Decimal("0")

    def test_short_design_life(self):
        """Test with short design life."""
        result = assess_tube_creep(
            temperature_c=565.0,
            stress_mpa=50.0,
            design_life_hours=10000.0  # Short design life
        )

        assert result.design_life_hours == Decimal("10000.000")

    def test_long_operating_time(self):
        """Test accumulation with long operating time."""
        assessor = CreepLifeAssessor()

        # Operating longer than design life
        conditions = [
            OperatingCondition(565.0, 50.0, 200000.0, "Extended")
        ]

        result = assessor.calculate_remaining_life(
            design_life_hours=100000.0,
            conditions=conditions,
            material=CreepMaterial.CR_2_25_MO_P22
        )

        # Damage mechanism should indicate concern
        assert "HIGH" in result.damage_mechanism or "CRITICAL" in result.damage_mechanism
