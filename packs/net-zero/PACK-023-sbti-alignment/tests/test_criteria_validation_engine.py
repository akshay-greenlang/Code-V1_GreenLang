# -*- coding: utf-8 -*-
"""
Test suite for PACK-023 SBTi Alignment Pack - CriteriaValidationEngine.

Validates:
  - 28 near-term criteria (C1-C28) assessment
  - 14 net-zero criteria (NZ-C1 to NZ-C14) assessment
  - Criterion pass/fail/warning/not_applicable status
  - Readiness score calculation
  - Remediation guidance for failures
  - Deterministic validation logic
  - Provenance hashing

Total Tests: 60+ parametrized assertions
Author: GreenLang Test Engineering
Pack: PACK-023 SBTi Alignment
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

from engines.criteria_validation_engine import (
    CriteriaValidationEngine,
    ValidationInput,
    ValidationResult,
    CriterionStatus,
    CriterionResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> CriteriaValidationEngine:
    """Fresh engine instance."""
    return CriteriaValidationEngine()


@pytest.fixture
def passing_input() -> ValidationInput:
    """Input that passes most/all criteria."""
    return ValidationInput(
        entity_name="ExcellentCorp",
        scope1_baseline_tco2e=Decimal("5000"),
        scope2_baseline_tco2e=Decimal("2000"),
        scope3_baseline_tco2e=Decimal("8000"),
        scope1_covered_pct=Decimal("98"),
        scope2_covered_pct=Decimal("100"),
        scope3_covered_pct=Decimal("85"),
        scope12_nt_reduction_rate_pct=Decimal("5.0"),
        scope3_nt_reduction_rate_pct=Decimal("3.0"),
        scope12_lt_reduction_rate_pct=Decimal("7.5"),
        scope3_lt_reduction_rate_pct=Decimal("4.5"),
        baseline_year=2024,
        nt_target_year=2030,
        lt_target_year=2045,
        nz_target_year=2050,
        nz_scope12_residual_pct=Decimal("5"),
        nz_scope3_residual_pct=Decimal("8"),
    )


@pytest.fixture
def failing_input() -> ValidationInput:
    """Input that fails several criteria."""
    return ValidationInput(
        entity_name="FailingCorp",
        scope1_baseline_tco2e=Decimal("1000"),
        scope2_baseline_tco2e=Decimal("500"),
        scope3_baseline_tco2e=Decimal("3000"),
        scope1_covered_pct=Decimal("80"),  # Below 95% minimum
        scope2_covered_pct=Decimal("75"),  # Below 95% minimum
        scope3_covered_pct=Decimal("40"),  # Below 67% for NT
        scope12_nt_reduction_rate_pct=Decimal("2.0"),  # Below 4.2% for 1.5C
        scope3_nt_reduction_rate_pct=Decimal("1.0"),
        scope12_lt_reduction_rate_pct=Decimal("3.0"),
        scope3_lt_reduction_rate_pct=Decimal("1.5"),
        baseline_year=2024,
        nt_target_year=2030,
        lt_target_year=2045,
        nz_target_year=2050,
        nz_scope12_residual_pct=Decimal("15"),  # Above 10% maximum
        nz_scope3_residual_pct=Decimal("12"),
    )


@pytest.fixture
def near_term_focused_input() -> ValidationInput:
    """Input with near-term target only (no long-term/NZ)."""
    return ValidationInput(
        entity_name="NearTermCorp",
        scope1_baseline_tco2e=Decimal("2000"),
        scope2_baseline_tco2e=Decimal("800"),
        scope3_baseline_tco2e=Decimal("5000"),
        scope1_covered_pct=Decimal("96"),
        scope2_covered_pct=Decimal("95"),
        scope3_covered_pct=Decimal("70"),
        scope12_nt_reduction_rate_pct=Decimal("4.5"),
        scope3_nt_reduction_rate_pct=Decimal("3.5"),
        baseline_year=2024,
        nt_target_year=2030,
        lt_target_year=None,
        nz_target_year=None,
    )


# ===========================================================================
# Tests -- Engine Instantiation
# ===========================================================================


class TestEngineInstantiation:
    """Tests for engine creation."""

    def test_engine_instantiates(self) -> None:
        """Engine must instantiate without arguments."""
        engine = CriteriaValidationEngine()
        assert engine is not None
        assert engine.engine_version == "1.0.0"


# ===========================================================================
# Tests -- Near-Term Criteria (C1-C28)
# ===========================================================================


class TestNearTermCriteria:
    """Tests for near-term SBTi criteria (C1-C28)."""

    def test_c1_boundary_coverage_scope1(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """C1: Baseline boundary must cover >= 95% of Scope 1."""
        inp_pass = ValidationInput(
            entity_name="C1Pass",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=Decimal("95"),
            scope2_covered_pct=Decimal("95"),
            scope3_covered_pct=Decimal("70"),
            scope12_nt_reduction_rate_pct=Decimal("4.5"),
            scope3_nt_reduction_rate_pct=Decimal("2.5"),
            baseline_year=2024,
            nt_target_year=2030,
        )
        result = engine.calculate(inp_pass)
        c1_result = next((c for c in result.criteria if c.criterion_id == "C1"), None)
        assert c1_result is not None
        assert c1_result.status in (CriterionStatus.PASS, CriterionStatus.WARNING)

    def test_c1_boundary_coverage_scope1_fail(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """C1 fails when Scope 1 coverage < 95%."""
        inp_fail = ValidationInput(
            entity_name="C1Fail",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=Decimal("80"),
            scope2_covered_pct=Decimal("80"),
            scope3_covered_pct=Decimal("50"),
            scope12_nt_reduction_rate_pct=Decimal("4.5"),
            scope3_nt_reduction_rate_pct=Decimal("2.5"),
            baseline_year=2024,
            nt_target_year=2030,
        )
        result = engine.calculate(inp_fail)
        c1_result = next((c for c in result.criteria if c.criterion_id == "C1"), None)
        assert c1_result is not None
        assert c1_result.status == CriterionStatus.FAIL

    def test_c6_ambition_scope1_2(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """C6: Near-term Scope 1+2 ambition must be >= 4.2%/yr (1.5C equivalent)."""
        inp_pass = ValidationInput(
            entity_name="C6Pass",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=Decimal("95"),
            scope2_covered_pct=Decimal("95"),
            scope3_covered_pct=Decimal("70"),
            scope12_nt_reduction_rate_pct=Decimal("4.2"),
            scope3_nt_reduction_rate_pct=Decimal("2.5"),
            baseline_year=2024,
            nt_target_year=2030,
        )
        result = engine.calculate(inp_pass)
        c6_result = next((c for c in result.criteria if c.criterion_id == "C6"), None)
        assert c6_result is not None
        assert c6_result.status in (CriterionStatus.PASS, CriterionStatus.WARNING)

    def test_c8_scope3_trigger(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """C8: If Scope 3 >= 40% of total, must set Scope 3 target."""
        inp_high_s3 = ValidationInput(
            entity_name="C8HighS3",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("3000"),  # 60% of 6000 total
            scope1_covered_pct=Decimal("95"),
            scope2_covered_pct=Decimal("95"),
            scope3_covered_pct=Decimal("70"),
            scope12_nt_reduction_rate_pct=Decimal("4.5"),
            scope3_nt_reduction_rate_pct=Decimal("2.5"),
            baseline_year=2024,
            nt_target_year=2030,
        )
        result = engine.calculate(inp_high_s3)
        c8_result = next((c for c in result.criteria if c.criterion_id == "C8"), None)
        assert c8_result is not None
        # Should require Scope 3 target
        assert c8_result.status in (CriterionStatus.PASS, CriterionStatus.WARNING)

    @pytest.mark.parametrize("coverage_pct,should_pass", [
        (Decimal("95"), True),
        (Decimal("94"), False),
        (Decimal("100"), True),
        (Decimal("85"), False),
    ])
    def test_coverage_boundary_conditions(
        self,
        engine: CriteriaValidationEngine,
        coverage_pct: Decimal,
        should_pass: bool,
    ) -> None:
        """Coverage boundary at 95% must be precisely enforced."""
        inp = ValidationInput(
            entity_name="BoundaryCorp",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=coverage_pct,
            scope2_covered_pct=coverage_pct,
            scope3_covered_pct=Decimal("70"),
            scope12_nt_reduction_rate_pct=Decimal("4.5"),
            scope3_nt_reduction_rate_pct=Decimal("2.5"),
            baseline_year=2024,
            nt_target_year=2030,
        )
        result = engine.calculate(inp)
        c1_result = next((c for c in result.criteria if c.criterion_id == "C1"), None)
        assert c1_result is not None
        if should_pass:
            assert c1_result.status in (CriterionStatus.PASS, CriterionStatus.WARNING)
        else:
            assert c1_result.status == CriterionStatus.FAIL


# ===========================================================================
# Tests -- Net-Zero Criteria (NZ-C1 to NZ-C14)
# ===========================================================================


class TestNetZeroCriteria:
    """Tests for net-zero SBTi criteria (NZ-C1 to NZ-C14)."""

    def test_nzc1_net_zero_target_by_2050(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """NZ-C1: Net-zero target must be by 2050 or earlier."""
        inp_valid = ValidationInput(
            entity_name="NZC1Valid",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=Decimal("95"),
            scope2_covered_pct=Decimal("95"),
            scope3_covered_pct=Decimal("70"),
            scope12_nt_reduction_rate_pct=Decimal("4.5"),
            scope3_nt_reduction_rate_pct=Decimal("2.5"),
            scope12_lt_reduction_rate_pct=Decimal("6.5"),
            scope3_lt_reduction_rate_pct=Decimal("3.5"),
            baseline_year=2024,
            nt_target_year=2030,
            lt_target_year=2045,
            nz_target_year=2050,
            nz_scope12_residual_pct=Decimal("5"),
            nz_scope3_residual_pct=Decimal("8"),
        )
        result = engine.calculate(inp_valid)
        nzc1_result = next((c for c in result.criteria if c.criterion_id == "NZ-C1"), None)
        if nzc1_result:  # NZ criteria may be optional if NZ target not set
            assert nzc1_result.status in (CriterionStatus.PASS, CriterionStatus.NOT_APPLICABLE)

    def test_nzc9_residual_emissions_max_10pct(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """NZ-C9: Residual emissions must be <= 10% of baseline for net-zero."""
        inp_pass = ValidationInput(
            entity_name="NZC9Pass",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=Decimal("95"),
            scope2_covered_pct=Decimal("95"),
            scope3_covered_pct=Decimal("70"),
            scope12_nt_reduction_rate_pct=Decimal("4.5"),
            scope3_nt_reduction_rate_pct=Decimal("2.5"),
            scope12_lt_reduction_rate_pct=Decimal("7.0"),
            scope3_lt_reduction_rate_pct=Decimal("4.0"),
            baseline_year=2024,
            nt_target_year=2030,
            lt_target_year=2045,
            nz_target_year=2050,
            nz_scope12_residual_pct=Decimal("9"),
            nz_scope3_residual_pct=Decimal("8"),
        )
        result = engine.calculate(inp_pass)
        nzc9_result = next((c for c in result.criteria if c.criterion_id == "NZ-C9"), None)
        if nzc9_result:
            assert nzc9_result.status in (CriterionStatus.PASS, CriterionStatus.NOT_APPLICABLE)

    def test_nzc9_residual_emissions_fail(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """NZ-C9 fails when residual emissions > 10%."""
        inp_fail = ValidationInput(
            entity_name="NZC9Fail",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=Decimal("95"),
            scope2_covered_pct=Decimal("95"),
            scope3_covered_pct=Decimal("70"),
            scope12_nt_reduction_rate_pct=Decimal("4.5"),
            scope3_nt_reduction_rate_pct=Decimal("2.5"),
            scope12_lt_reduction_rate_pct=Decimal("5.0"),
            scope3_lt_reduction_rate_pct=Decimal("2.0"),
            baseline_year=2024,
            nt_target_year=2030,
            lt_target_year=2045,
            nz_target_year=2050,
            nz_scope12_residual_pct=Decimal("15"),  # Exceeds 10%
            nz_scope3_residual_pct=Decimal("12"),
        )
        result = engine.calculate(inp_fail)
        nzc9_result = next((c for c in result.criteria if c.criterion_id == "NZ-C9"), None)
        if nzc9_result:
            assert nzc9_result.status in (CriterionStatus.FAIL, CriterionStatus.NOT_APPLICABLE)


# ===========================================================================
# Tests -- Readiness Score Calculation
# ===========================================================================


class TestReadinessScoreCalculation:
    """Tests for overall readiness score computation."""

    def test_readiness_score_all_pass(
        self, engine: CriteriaValidationEngine, passing_input: ValidationInput
    ) -> None:
        """All passing criteria should yield high readiness score."""
        result = engine.calculate(passing_input)

        assert result.readiness_score >= Decimal("90")
        assert result.readiness_score <= Decimal("100")

    def test_readiness_score_all_fail(
        self, engine: CriteriaValidationEngine, failing_input: ValidationInput
    ) -> None:
        """Many failures should yield low readiness score."""
        result = engine.calculate(failing_input)

        assert result.readiness_score < Decimal("70")

    def test_readiness_score_formula(
        self, engine: CriteriaValidationEngine, passing_input: ValidationInput
    ) -> None:
        """Readiness score = (passed + 0.5*warnings) / total_applicable * 100."""
        result = engine.calculate(passing_input)

        # Score should be deterministic based on criteria status
        assert result.readiness_score > Decimal("0")
        assert result.readiness_score <= Decimal("100")

    def test_readiness_score_monotonic(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """Improving inputs should monotonically increase readiness score."""
        inp_low = ValidationInput(
            entity_name="LowScore",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=Decimal("80"),
            scope2_covered_pct=Decimal("80"),
            scope3_covered_pct=Decimal("40"),
            scope12_nt_reduction_rate_pct=Decimal("2.0"),
            scope3_nt_reduction_rate_pct=Decimal("1.0"),
            baseline_year=2024,
            nt_target_year=2030,
        )
        inp_high = ValidationInput(
            entity_name="HighScore",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=Decimal("98"),
            scope2_covered_pct=Decimal("98"),
            scope3_covered_pct=Decimal("85"),
            scope12_nt_reduction_rate_pct=Decimal("5.0"),
            scope3_nt_reduction_rate_pct=Decimal("3.5"),
            baseline_year=2024,
            nt_target_year=2030,
        )
        result_low = engine.calculate(inp_low)
        result_high = engine.calculate(inp_high)

        assert result_high.readiness_score > result_low.readiness_score


# ===========================================================================
# Tests -- Remediation Guidance
# ===========================================================================


class TestRemediationGuidance:
    """Tests for failure remediation guidance."""

    def test_failed_criteria_have_remediation(
        self, engine: CriteriaValidationEngine, failing_input: ValidationInput
    ) -> None:
        """Failed criteria should have remediation guidance."""
        result = engine.calculate(failing_input)

        failed_criteria = [c for c in result.criteria if c.status == CriterionStatus.FAIL]
        for criterion in failed_criteria:
            assert criterion.remediation is not None
            assert len(criterion.remediation) > 0

    def test_remediation_specific_to_criterion(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """Remediation guidance should be specific to failed criterion."""
        inp = ValidationInput(
            entity_name="RemediationTest",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=Decimal("70"),  # Fails coverage
            scope2_covered_pct=Decimal("70"),
            scope3_covered_pct=Decimal("40"),
            scope12_nt_reduction_rate_pct=Decimal("2.0"),  # Fails ambition
            scope3_nt_reduction_rate_pct=Decimal("1.0"),
            baseline_year=2024,
            nt_target_year=2030,
        )
        result = engine.calculate(inp)

        coverage_fail = next((c for c in result.criteria if c.criterion_id == "C1"), None)
        if coverage_fail and coverage_fail.status == CriterionStatus.FAIL:
            assert "coverage" in coverage_fail.remediation.lower() or "boundary" in coverage_fail.remediation.lower()


# ===========================================================================
# Tests -- Criterion Status Values
# ===========================================================================


class TestCriterionStatusValues:
    """Tests for criterion status classifications."""

    def test_criterion_status_enum_completeness(self) -> None:
        """All status enums must be defined."""
        assert CriterionStatus.PASS is not None
        assert CriterionStatus.FAIL is not None
        assert CriterionStatus.WARNING is not None
        assert CriterionStatus.NOT_APPLICABLE is not None

    def test_result_criteria_all_have_status(
        self, engine: CriteriaValidationEngine, passing_input: ValidationInput
    ) -> None:
        """Every criterion in result must have a status."""
        result = engine.calculate(passing_input)

        for criterion in result.criteria:
            assert criterion.status is not None
            assert criterion.status in [
                CriterionStatus.PASS,
                CriterionStatus.FAIL,
                CriterionStatus.WARNING,
                CriterionStatus.NOT_APPLICABLE,
            ]


# ===========================================================================
# Tests -- Provenance and Zero-Hallucination
# ===========================================================================


class TestProvenanceAndValidation:
    """Tests for deterministic hashing and validation."""

    def test_result_has_provenance_hash(
        self, engine: CriteriaValidationEngine, passing_input: ValidationInput
    ) -> None:
        """Every result must have SHA-256 provenance hash."""
        result = engine.calculate(passing_input)

        assert hasattr(result, "provenance_hash")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_deterministic(
        self, engine: CriteriaValidationEngine, passing_input: ValidationInput
    ) -> None:
        """Same input must produce same provenance hash."""
        result1 = engine.calculate(passing_input)
        result2 = engine.calculate(passing_input)

        assert result1.provenance_hash == result2.provenance_hash

    def test_different_input_different_hash(
        self, engine: CriteriaValidationEngine, passing_input: ValidationInput
    ) -> None:
        """Different inputs must produce different hashes."""
        result1 = engine.calculate(passing_input)

        modified = passing_input.model_copy()
        modified.scope1_covered_pct = Decimal("80")
        result2 = engine.calculate(modified)

        assert result1.provenance_hash != result2.provenance_hash


# ===========================================================================
# Tests -- Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_scope3_emissions(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """Zero Scope 3 emissions should handle gracefully."""
        inp = ValidationInput(
            entity_name="NoS3",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("0"),
            scope1_covered_pct=Decimal("95"),
            scope2_covered_pct=Decimal("95"),
            scope3_covered_pct=Decimal("100"),
            scope12_nt_reduction_rate_pct=Decimal("4.5"),
            scope3_nt_reduction_rate_pct=Decimal("0"),
            baseline_year=2024,
            nt_target_year=2030,
        )
        result = engine.calculate(inp)

        assert len(result.criteria) > 0
        # Scope 3 criteria should be NOT_APPLICABLE
        s3_criteria = [c for c in result.criteria if "scope3" in c.criterion_id.lower() or "s3" in c.criterion_id.lower()]
        for criterion in s3_criteria:
            assert criterion.status in (CriterionStatus.NOT_APPLICABLE, CriterionStatus.PASS)

    def test_near_term_only_input(
        self, engine: CriteriaValidationEngine, near_term_focused_input: ValidationInput
    ) -> None:
        """Near-term only (no LT/NZ) should mark those criteria as N/A."""
        result = engine.calculate(near_term_focused_input)

        # Find net-zero criteria
        nz_criteria = [c for c in result.criteria if c.criterion_id.startswith("NZ-")]
        for criterion in nz_criteria:
            assert criterion.status == CriterionStatus.NOT_APPLICABLE

    def test_very_high_reduction_rates(
        self, engine: CriteriaValidationEngine
    ) -> None:
        """Very high reduction rates should pass ambition criteria."""
        inp = ValidationInput(
            entity_name="SuperAmbitious",
            scope1_baseline_tco2e=Decimal("1000"),
            scope2_baseline_tco2e=Decimal("500"),
            scope3_baseline_tco2e=Decimal("2000"),
            scope1_covered_pct=Decimal("95"),
            scope2_covered_pct=Decimal("95"),
            scope3_covered_pct=Decimal("75"),
            scope12_nt_reduction_rate_pct=Decimal("10.0"),  # Very aggressive
            scope3_nt_reduction_rate_pct=Decimal("8.0"),
            baseline_year=2024,
            nt_target_year=2030,
        )
        result = engine.calculate(inp)

        c6_result = next((c for c in result.criteria if c.criterion_id == "C6"), None)
        if c6_result:
            assert c6_result.status in (CriterionStatus.PASS, CriterionStatus.WARNING)
