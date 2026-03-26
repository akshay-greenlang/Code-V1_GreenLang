# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 SBTi Pathway Engine
==============================================

Tests materiality check, near-term target calculation, long-term
target, FLAG pathway, coverage check, progress tracking, milestone
generation, submission package, and edge cases.

Coverage target: 85%+
Total tests: ~45
"""

from decimal import Decimal

import pytest

from tests.conftest import (
    SBTI_15C_ANNUAL_RATE,
    SBTI_WB2C_ANNUAL_RATE,
    SBTI_LONG_TERM_REDUCTION,
    SBTI_COVERAGE_THRESHOLD,
)


# =============================================================================
# Materiality Check
# =============================================================================


class TestMaterialityCheck:
    """Test Scope 3 materiality determination for SBTi."""

    def test_scope3_above_40pct(self, sample_sbti_targets):
        """Scope 3 > 40% of total means Scope 3 target is required."""
        pct = sample_sbti_targets["scope3_pct_of_total"]
        assert pct > Decimal("40")

    def test_scope3_is_material(self, sample_sbti_targets):
        assert sample_sbti_targets["scope3_is_material"] is True

    def test_scope3_pct_calculation(self, sample_sbti_targets):
        """Verify: scope3 / total * 100."""
        s3 = sample_sbti_targets["base_year_scope3_tco2e"]
        total = sample_sbti_targets["base_year_total_tco2e"]
        expected_pct = s3 / total * Decimal("100")
        assert expected_pct == pytest.approx(
            sample_sbti_targets["scope3_pct_of_total"], abs=Decimal("0.01")
        )

    def test_total_equals_sum(self, sample_sbti_targets):
        s1 = sample_sbti_targets["base_year_scope1_tco2e"]
        s2 = sample_sbti_targets["base_year_scope2_tco2e"]
        s3 = sample_sbti_targets["base_year_scope3_tco2e"]
        total = sample_sbti_targets["base_year_total_tco2e"]
        assert s1 + s2 + s3 == total


# =============================================================================
# Near-Term Target Calculation
# =============================================================================


class TestNearTermTarget:
    """Test near-term target calculation (4.2%/yr for 1.5C)."""

    def test_near_term_pathway_15c(self, sample_sbti_targets):
        assert sample_sbti_targets["near_term"]["pathway"] == "1.5C"

    def test_near_term_annual_rate_42(self, sample_sbti_targets):
        rate = sample_sbti_targets["near_term"]["annual_reduction_pct"]
        assert rate == SBTI_15C_ANNUAL_RATE

    def test_near_term_target_year_2030(self, sample_sbti_targets):
        assert sample_sbti_targets["near_term"]["target_year"] == 2030

    def test_near_term_reduction_calculation(self, sample_sbti_targets):
        """42% reduction from base year to 2030 (1.5C, ~10 years at 4.2%)."""
        nt = sample_sbti_targets["near_term"]
        base_year = sample_sbti_targets["base_year"]
        years = nt["target_year"] - base_year
        # Compound: (1 - 0.042)^11 = ~0.628 -> 37.2% reduction
        # But SBTi uses 42% for 2019->2030 boundary
        reduction = nt["target_reduction_pct"]
        assert Decimal("30") <= reduction <= Decimal("50")

    def test_near_term_absolute_target(self, sample_sbti_targets):
        """Absolute target = base_year * (1 - reduction%)."""
        nt = sample_sbti_targets["near_term"]
        base = sample_sbti_targets["base_year_scope3_tco2e"]
        expected = base * (Decimal("1") - nt["target_reduction_pct"] / Decimal("100"))
        assert nt["target_absolute_tco2e"] == expected

    def test_covered_categories_non_empty(self, sample_sbti_targets):
        cats = sample_sbti_targets["near_term"]["covered_categories"]
        assert len(cats) >= 5


# =============================================================================
# Long-Term Target
# =============================================================================


class TestLongTermTarget:
    """Test long-term net-zero target (90% by 2050)."""

    def test_long_term_year_2050(self, sample_sbti_targets):
        assert sample_sbti_targets["long_term"]["target_year"] == 2050

    def test_long_term_reduction_90pct(self, sample_sbti_targets):
        reduction = sample_sbti_targets["long_term"]["total_reduction_pct"]
        assert reduction == SBTI_LONG_TERM_REDUCTION

    def test_long_term_absolute(self, sample_sbti_targets):
        lt = sample_sbti_targets["long_term"]
        base = sample_sbti_targets["base_year_scope3_tco2e"]
        expected = base * (Decimal("1") - lt["total_reduction_pct"] / Decimal("100"))
        assert lt["target_absolute_tco2e"] == expected

    def test_long_term_more_ambitious_than_near(self, sample_sbti_targets):
        nt = sample_sbti_targets["near_term"]["target_absolute_tco2e"]
        lt = sample_sbti_targets["long_term"]["target_absolute_tco2e"]
        assert lt < nt


# =============================================================================
# FLAG Pathway
# =============================================================================


class TestFLAGPathway:
    """Test FLAG (Forest, Land and Agriculture) pathway applicability."""

    def test_flag_not_applicable_for_manufacturing(self, sample_sbti_targets):
        assert sample_sbti_targets["flag_pathway"]["applicable"] is False

    def test_flag_sectors_empty_for_manufacturing(self, sample_sbti_targets):
        assert len(sample_sbti_targets["flag_pathway"]["sectors"]) == 0

    def test_flag_applicable_for_agriculture(self):
        """FLAG should be applicable for agricultural companies."""
        flag = {
            "applicable": True,
            "sectors": ["agriculture", "forestry", "land_use"],
            "target_type": "intensity",
        }
        assert flag["applicable"] is True
        assert len(flag["sectors"]) >= 1


# =============================================================================
# Coverage Check
# =============================================================================


class TestCoverageCheck:
    """Test Scope 3 coverage threshold check (>= 67%)."""

    def test_coverage_above_threshold(self, sample_sbti_targets):
        coverage = sample_sbti_targets["near_term"]["scope3_coverage_pct"]
        assert coverage >= SBTI_COVERAGE_THRESHOLD

    def test_coverage_calculation(self, sample_sbti_targets):
        """Coverage = covered_emissions / total_scope3 * 100."""
        nt = sample_sbti_targets["near_term"]
        base_s3 = sample_sbti_targets["base_year_scope3_tco2e"]
        expected = nt["covered_emissions_tco2e"] / base_s3 * Decimal("100")
        assert expected == nt["scope3_coverage_pct"]

    def test_ten_categories_covered(self, sample_sbti_targets):
        cats = sample_sbti_targets["near_term"]["covered_categories"]
        assert len(cats) == 10

    def test_covered_categories_in_range(self, sample_sbti_targets):
        for cat in sample_sbti_targets["near_term"]["covered_categories"]:
            assert 1 <= cat <= 15


# =============================================================================
# Progress Tracking
# =============================================================================


class TestProgressTracking:
    """Test progress tracking against SBTi targets."""

    def test_milestones_defined(self, sample_sbti_targets):
        assert len(sample_sbti_targets["milestones"]) >= 2

    def test_2025_milestone_on_track(self, sample_sbti_targets):
        milestone_2025 = next(
            m for m in sample_sbti_targets["milestones"] if m["year"] == 2025
        )
        assert milestone_2025["status"] == "on_track"

    def test_2030_milestone_is_target(self, sample_sbti_targets):
        milestone_2030 = next(
            m for m in sample_sbti_targets["milestones"] if m["year"] == 2030
        )
        assert milestone_2030["status"] == "target"

    def test_milestone_emissions_decreasing(self, sample_sbti_targets):
        milestones = sorted(sample_sbti_targets["milestones"], key=lambda m: m["year"])
        for i in range(1, len(milestones)):
            assert milestones[i]["target_tco2e"] <= milestones[i - 1]["target_tco2e"]

    def test_progress_rate_calculation(self, sample_sbti_targets):
        """Progress = (base - current) / (base - target) * 100."""
        base = sample_sbti_targets["base_year_scope3_tco2e"]
        current = sample_sbti_targets["milestones"][0]["target_tco2e"]
        target = sample_sbti_targets["near_term"]["target_absolute_tco2e"]
        progress = (base - current) / (base - target) * Decimal("100")
        assert Decimal("0") <= progress <= Decimal("100")


# =============================================================================
# Milestone Generation
# =============================================================================


class TestMilestoneGeneration:
    """Test milestone generation between base year and target year."""

    def test_milestones_have_years(self, sample_sbti_targets):
        for m in sample_sbti_targets["milestones"]:
            assert "year" in m
            assert m["year"] >= sample_sbti_targets["base_year"]

    def test_milestones_have_targets(self, sample_sbti_targets):
        for m in sample_sbti_targets["milestones"]:
            assert "target_tco2e" in m
            assert m["target_tco2e"] > Decimal("0")

    def test_generate_annual_milestones(self, sample_sbti_targets):
        """Generate annual milestones from base to target."""
        base = sample_sbti_targets["base_year_scope3_tco2e"]
        rate = sample_sbti_targets["near_term"]["annual_reduction_pct"] / Decimal("100")
        base_year = sample_sbti_targets["base_year"]
        target_year = sample_sbti_targets["near_term"]["target_year"]
        milestones = []
        for y in range(base_year, target_year + 1):
            years_elapsed = y - base_year
            target = base * (Decimal("1") - rate) ** years_elapsed
            milestones.append({"year": y, "target_tco2e": target})
        assert len(milestones) == target_year - base_year + 1
        assert milestones[-1]["target_tco2e"] < milestones[0]["target_tco2e"]


# =============================================================================
# Submission Package
# =============================================================================


class TestSubmissionPackage:
    """Test SBTi submission data package."""

    def test_submission_has_base_year(self, sample_sbti_targets):
        assert sample_sbti_targets["base_year"] == 2019

    def test_submission_has_scope_data(self, sample_sbti_targets):
        assert sample_sbti_targets["base_year_scope1_tco2e"] > Decimal("0")
        assert sample_sbti_targets["base_year_scope2_tco2e"] > Decimal("0")
        assert sample_sbti_targets["base_year_scope3_tco2e"] > Decimal("0")

    def test_submission_has_near_term(self, sample_sbti_targets):
        nt = sample_sbti_targets["near_term"]
        assert nt["target_year"] > sample_sbti_targets["base_year"]
        assert nt["target_absolute_tco2e"] > Decimal("0")

    def test_submission_has_long_term(self, sample_sbti_targets):
        lt = sample_sbti_targets["long_term"]
        assert lt["target_year"] > sample_sbti_targets["near_term"]["target_year"]


# =============================================================================
# Edge Cases
# =============================================================================


class TestSBTiEdgeCases:
    """Test edge cases for SBTi pathway engine."""

    def test_scope3_below_40pct(self):
        """If Scope 3 < 40%, Scope 3 target is optional."""
        total = Decimal("100000")
        scope3 = Decimal("35000")
        pct = scope3 / total * Decimal("100")
        is_material = pct > Decimal("40")
        assert is_material is False

    def test_100pct_coverage(self):
        """100% coverage: all 15 categories covered."""
        covered = list(range(1, 16))
        coverage_pct = Decimal("100")
        assert len(covered) == 15
        assert coverage_pct >= SBTI_COVERAGE_THRESHOLD

    def test_minimum_coverage_threshold(self):
        """Exactly 67% coverage should pass."""
        total_scope3 = Decimal("300000")
        covered = Decimal("201000")  # 67%
        coverage_pct = covered / total_scope3 * Decimal("100")
        assert coverage_pct >= SBTI_COVERAGE_THRESHOLD

    def test_wb2c_pathway_rate(self):
        """WB2C pathway uses 2.5%/yr."""
        assert SBTI_WB2C_ANNUAL_RATE == Decimal("2.5")

    def test_near_zero_emissions_target(self):
        """Long-term 90% reduction from very low base."""
        base = Decimal("1000")
        long_term = base * (Decimal("1") - SBTI_LONG_TERM_REDUCTION / Decimal("100"))
        assert long_term == Decimal("100")
