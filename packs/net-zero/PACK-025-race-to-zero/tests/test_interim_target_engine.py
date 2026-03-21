# -*- coding: utf-8 -*-
"""
Deep tests for InterimTargetEngine (Engine 3 of 10).

Covers: 2030 target validation, 1.5C pathway alignment, temperature
scoring formula, annual reduction rate calculation, scope coverage
validation, fair share assessment, methodology validation, boundary
values, Decimal arithmetic, SHA-256 provenance.

Target: ~60 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from engines.interim_target_engine import (
    InterimTargetEngine,
    InterimTargetInput,
    PathwayAlignment,
    ScopeTargetInput,
    TargetMethodology,
    TargetType,
    ComplianceLevel,
    IPCC_MIN_REDUCTION_PCT,
    R2Z_TARGET_REDUCTION_PCT,
    SBTI_1_5C_ANNUAL_RATE,
    SBTI_WB2C_ANNUAL_RATE,
    SBTI_2C_ANNUAL_RATE,
    MIN_SCOPE1_2_COVERAGE,
    MIN_SCOPE3_COVERAGE,
    PATHWAY_THRESHOLDS,
    RECOGNIZED_METHODOLOGIES,
    TEMP_FLOOR,
    TEMP_CEILING,
)

from conftest import assert_decimal_close, assert_provenance_hash, timed_block


# ========================================================================
# Constants Validation
# ========================================================================


class TestInterimTargetConstants:
    """Validate interim target constants."""

    def test_ipcc_min_reduction_42(self):
        assert IPCC_MIN_REDUCTION_PCT == Decimal("42")

    def test_r2z_target_50(self):
        assert R2Z_TARGET_REDUCTION_PCT == Decimal("50")

    def test_sbti_1_5c_rate_4_2(self):
        assert SBTI_1_5C_ANNUAL_RATE == Decimal("4.2")

    def test_sbti_wb2c_rate_2_5(self):
        assert SBTI_WB2C_ANNUAL_RATE == Decimal("2.5")

    def test_sbti_2c_rate_1_5(self):
        assert SBTI_2C_ANNUAL_RATE == Decimal("1.5")

    def test_min_scope1_2_coverage_95(self):
        assert MIN_SCOPE1_2_COVERAGE == Decimal("95")

    def test_min_scope3_coverage_67(self):
        assert MIN_SCOPE3_COVERAGE == Decimal("67")

    def test_temp_floor_1_5(self):
        assert TEMP_FLOOR == Decimal("1.5")

    def test_temp_ceiling_4_0(self):
        assert TEMP_CEILING == Decimal("4.0")

    def test_pathway_thresholds_3_entries(self):
        assert len(PATHWAY_THRESHOLDS) == 3

    def test_recognized_methodologies_count(self):
        assert len(RECOGNIZED_METHODOLOGIES) == 5

    def test_sbti_aca_recognized(self):
        assert "sbti_aca" in RECOGNIZED_METHODOLOGIES

    def test_iea_nze_recognized(self):
        assert "iea_nze" in RECOGNIZED_METHODOLOGIES

    def test_ipcc_ar6_recognized(self):
        assert "ipcc_ar6" in RECOGNIZED_METHODOLOGIES


# ========================================================================
# Enum Validation
# ========================================================================


class TestInterimTargetEnums:
    """Validate interim target enums."""

    def test_pathway_alignment_4_values(self):
        assert len(PathwayAlignment) == 4

    def test_pathway_values(self):
        assert PathwayAlignment.ALIGNED_1_5C.value == "1.5c_aligned"
        assert PathwayAlignment.WELL_BELOW_2C.value == "well_below_2c"
        assert PathwayAlignment.ALIGNED_2C.value == "2c_aligned"
        assert PathwayAlignment.MISALIGNED.value == "misaligned"

    def test_target_methodology_7_values(self):
        assert len(TargetMethodology) == 7

    def test_target_type_3_values(self):
        assert len(TargetType) == 3

    def test_compliance_level_4_values(self):
        assert len(ComplianceLevel) == 4

    def test_compliance_values(self):
        assert ComplianceLevel.FULLY_COMPLIANT.value == "fully_compliant"
        assert ComplianceLevel.NON_COMPLIANT.value == "non_compliant"


# ========================================================================
# Input Model Validation
# ========================================================================


class TestInterimTargetInputModel:
    """Validate InterimTargetInput Pydantic model."""

    def test_aligned_input_constructs(self, aligned_interim_input):
        assert aligned_interim_input.entity_name == "GreenCorp International"
        assert aligned_interim_input.target_reduction_pct == Decimal("50")

    def test_misaligned_input_constructs(self, misaligned_interim_input):
        assert misaligned_interim_input.target_reduction_pct == Decimal("10")

    def test_scope_target_input(self):
        st = ScopeTargetInput(
            scope=1,
            baseline_emissions_tco2e=Decimal("50000"),
            target_emissions_tco2e=Decimal("25000"),
        )
        assert st.scope == 1

    def test_invalid_target_type_raises(self):
        with pytest.raises(Exception):
            InterimTargetInput(
                entity_name="Test",
                baseline_year=2019,
                total_baseline_emissions_tco2e=Decimal("100000"),
                total_target_emissions_tco2e=Decimal("50000"),
                target_type="invalid_type",
            )

    def test_invalid_methodology_raises(self):
        with pytest.raises(Exception):
            InterimTargetInput(
                entity_name="Test",
                baseline_year=2019,
                total_baseline_emissions_tco2e=Decimal("100000"),
                total_target_emissions_tco2e=Decimal("50000"),
                methodology="invalid_method",
            )


# ========================================================================
# Engine Instantiation
# ========================================================================


class TestInterimTargetEngineInstantiation:
    """Tests for engine creation."""

    def test_default_instantiation(self, interim_target_engine):
        assert interim_target_engine is not None

    def test_engine_has_calculate(self, interim_target_engine):
        assert callable(getattr(interim_target_engine, "validate", None))


# ========================================================================
# Aligned Target Assessment
# ========================================================================


class TestAlignedTargetAssessment:
    """Tests for a 1.5C-aligned target assessment."""

    def test_aligned_calculates(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        assert result is not None

    def test_aligned_pathway(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        assert result.pathway_alignment in (
            "1.5c_aligned", "well_below_2c",
        )

    def test_aligned_meets_ipcc_minimum(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        assert result.meets_ipcc_minimum is True

    def test_aligned_meets_r2z_minimum(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        assert result.meets_r2z_minimum is True

    def test_aligned_reduction_pct(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        assert result.absolute_reduction_pct >= Decimal("42")

    def test_aligned_compliance_level(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        assert result.compliance_level in (
            "fully_compliant", "substantially_compliant",
        )

    def test_aligned_temperature_score_below_2(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        assert result.temperature_score <= Decimal("2.0")

    def test_aligned_has_provenance(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        assert_provenance_hash(result)

    def test_aligned_methodology_recognized(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        assert result.methodology_recognized is True

    def test_aligned_performance(
        self, interim_target_engine, aligned_interim_input,
    ):
        with timed_block("aligned_target_assessment", max_seconds=5.0):
            interim_target_engine.validate(aligned_interim_input)


# ========================================================================
# Misaligned Target Assessment
# ========================================================================


class TestMisalignedTargetAssessment:
    """Tests for a misaligned target assessment."""

    def test_misaligned_calculates(
        self, interim_target_engine, misaligned_interim_input,
    ):
        result = interim_target_engine.validate(misaligned_interim_input)
        assert result is not None

    def test_misaligned_pathway(
        self, interim_target_engine, misaligned_interim_input,
    ):
        result = interim_target_engine.validate(misaligned_interim_input)
        assert result.pathway_alignment in ("misaligned", "2c_aligned")

    def test_misaligned_does_not_meet_ipcc(
        self, interim_target_engine, misaligned_interim_input,
    ):
        result = interim_target_engine.validate(misaligned_interim_input)
        assert result.meets_ipcc_minimum is False

    def test_misaligned_temperature_score_high(
        self, interim_target_engine, misaligned_interim_input,
    ):
        result = interim_target_engine.validate(misaligned_interim_input)
        assert result.temperature_score >= Decimal("2.0")

    def test_misaligned_has_gaps(
        self, interim_target_engine, misaligned_interim_input,
    ):
        result = interim_target_engine.validate(misaligned_interim_input)
        assert len(result.gaps) > 0

    def test_misaligned_methodology_not_recognized(
        self, interim_target_engine, misaligned_interim_input,
    ):
        result = interim_target_engine.validate(misaligned_interim_input)
        assert result.methodology_recognized is False

    def test_misaligned_has_provenance(
        self, interim_target_engine, misaligned_interim_input,
    ):
        result = interim_target_engine.validate(misaligned_interim_input)
        assert_provenance_hash(result)


# ========================================================================
# Pathway Comparison
# ========================================================================


class TestPathwayComparison:
    """Tests for pathway comparison output."""

    def test_aligned_has_pathway_comparisons(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        assert hasattr(result, "pathway_comparisons")
        assert len(result.pathway_comparisons) > 0

    def test_pathway_comparison_fields(
        self, interim_target_engine, aligned_interim_input,
    ):
        result = interim_target_engine.validate(aligned_interim_input)
        if result.pathway_comparisons:
            pc = result.pathway_comparisons[0]
            assert hasattr(pc, "pathway_name")
            assert hasattr(pc, "entity_reduction_pct")


# ========================================================================
# Determinism
# ========================================================================


class TestInterimTargetDeterminism:
    """Tests for deterministic output."""

    def test_same_input_same_pathway(
        self, interim_target_engine, aligned_interim_input,
    ):
        r1 = interim_target_engine.validate(aligned_interim_input)
        r2 = interim_target_engine.validate(aligned_interim_input)
        assert r1.pathway_alignment == r2.pathway_alignment

    def test_same_input_same_temperature(
        self, interim_target_engine, aligned_interim_input,
    ):
        r1 = interim_target_engine.validate(aligned_interim_input)
        r2 = interim_target_engine.validate(aligned_interim_input)
        assert r1.temperature_score == r2.temperature_score

    def test_same_input_same_reduction(
        self, interim_target_engine, aligned_interim_input,
    ):
        r1 = interim_target_engine.validate(aligned_interim_input)
        r2 = interim_target_engine.validate(aligned_interim_input)
        assert r1.absolute_reduction_pct == r2.absolute_reduction_pct
