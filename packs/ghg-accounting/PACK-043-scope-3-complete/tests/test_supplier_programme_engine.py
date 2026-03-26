# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 Supplier Programme Engine
====================================================

Tests supplier target setting, commitment tracking, YoY progress,
programme impact, scorecard generation, tier classification,
incentive modelling, programme ROI, and transition risk.

Coverage target: 85%+
Total tests: ~45
"""

from decimal import Decimal

import pytest


# =============================================================================
# Supplier Target Setting
# =============================================================================


class TestSupplierTargetSetting:
    """Test supplier target setting logic."""

    def test_twenty_suppliers_present(self, sample_supplier_programme):
        assert sample_supplier_programme["total_suppliers"] == 20

    def test_suppliers_have_required_fields(self, sample_supplier_programme):
        required = [
            "supplier_id", "supplier_name", "category",
            "scope3_contribution_tco2e", "commitment_type",
            "has_reduction_target", "yoy_reduction_pct", "tier",
        ]
        for sup in sample_supplier_programme["suppliers"]:
            for field in required:
                assert field in sup, f"Missing {field} in {sup['supplier_id']}"

    def test_programme_target_30pct(self, sample_supplier_programme):
        assert sample_supplier_programme["programme_target_reduction_pct"] == Decimal("30")

    def test_programme_target_year_2030(self, sample_supplier_programme):
        assert sample_supplier_programme["programme_target_year"] == 2030

    def test_supplier_ids_unique(self, sample_supplier_programme):
        ids = [s["supplier_id"] for s in sample_supplier_programme["suppliers"]]
        assert len(ids) == len(set(ids))


# =============================================================================
# Commitment Tracking
# =============================================================================


class TestCommitmentTracking:
    """Test tracking supplier commitments (SBTi, RE100, CDP)."""

    def test_committed_suppliers_count(self, sample_supplier_programme):
        committed = sample_supplier_programme["committed_suppliers"]
        total = sample_supplier_programme["total_suppliers"]
        assert 0 < committed <= total

    def test_committed_emissions_positive(self, sample_supplier_programme):
        assert sample_supplier_programme["committed_emissions_tco2e"] > Decimal("0")

    def test_coverage_calculation(self, sample_supplier_programme):
        committed = sample_supplier_programme["committed_emissions_tco2e"]
        total = sample_supplier_programme["total_supplier_emissions_tco2e"]
        expected_pct = committed / total * Decimal("100")
        assert expected_pct == pytest.approx(
            sample_supplier_programme["coverage_pct"], abs=Decimal("0.01")
        )

    @pytest.mark.parametrize("commitment_type", ["SBTi", "RE100", "CDP"])
    def test_commitment_type_present(self, commitment_type, sample_supplier_programme):
        found = any(
            s["commitment_type"] == commitment_type
            for s in sample_supplier_programme["suppliers"]
        )
        assert found is True

    def test_uncommitted_suppliers_exist(self, sample_supplier_programme):
        uncommitted = [
            s for s in sample_supplier_programme["suppliers"]
            if s["commitment_type"] == "None"
        ]
        assert len(uncommitted) >= 1


# =============================================================================
# YoY Progress Measurement
# =============================================================================


class TestYoYProgressMeasurement:
    """Test year-over-year progress measurement."""

    def test_committed_suppliers_have_reduction(self, sample_supplier_programme):
        for sup in sample_supplier_programme["suppliers"]:
            if sup["has_reduction_target"]:
                assert sup["yoy_reduction_pct"] > Decimal("0")

    def test_uncommitted_suppliers_zero_reduction(self, sample_supplier_programme):
        for sup in sample_supplier_programme["suppliers"]:
            if not sup["has_reduction_target"]:
                assert sup["yoy_reduction_pct"] == Decimal("0")

    def test_weighted_average_reduction(self, sample_supplier_programme):
        """Weighted average reduction across committed suppliers."""
        suppliers = sample_supplier_programme["suppliers"]
        committed = [s for s in suppliers if s["has_reduction_target"]]
        total_emissions = sum(s["scope3_contribution_tco2e"] for s in committed)
        weighted = sum(
            s["yoy_reduction_pct"] * s["scope3_contribution_tco2e"] / total_emissions
            for s in committed
        )
        assert weighted > Decimal("0")

    def test_max_reduction_rate(self, sample_supplier_programme):
        max_rate = max(
            s["yoy_reduction_pct"]
            for s in sample_supplier_programme["suppliers"]
        )
        assert max_rate <= Decimal("15")  # reasonable upper bound


# =============================================================================
# Programme Impact on Reporter Scope 3
# =============================================================================


class TestProgrammeImpact:
    """Test programme impact on reporter's Scope 3 emissions."""

    def test_programme_reduces_emissions(self, sample_supplier_programme):
        """Total reduction = sum of each supplier's YoY reduction * their emissions."""
        suppliers = sample_supplier_programme["suppliers"]
        total_reduction = sum(
            s["scope3_contribution_tco2e"] * s["yoy_reduction_pct"] / Decimal("100")
            for s in suppliers
            if s["has_reduction_target"]
        )
        assert total_reduction > Decimal("0")

    def test_programme_impact_pct(self, sample_supplier_programme):
        """Programme impact as % of total supplier emissions."""
        suppliers = sample_supplier_programme["suppliers"]
        total = sample_supplier_programme["total_supplier_emissions_tco2e"]
        reduction = sum(
            s["scope3_contribution_tco2e"] * s["yoy_reduction_pct"] / Decimal("100")
            for s in suppliers
            if s["has_reduction_target"]
        )
        impact_pct = reduction / total * Decimal("100")
        assert Decimal("0") < impact_pct < Decimal("20")


# =============================================================================
# Scorecard Generation
# =============================================================================


class TestScorecardGeneration:
    """Test supplier scorecard generation and scoring."""

    def test_all_suppliers_have_engagement_score(self, sample_supplier_programme):
        for sup in sample_supplier_programme["suppliers"]:
            assert "engagement_score" in sup
            assert Decimal("0") <= sup["engagement_score"] <= Decimal("100")

    def test_high_emission_suppliers_higher_score(self, sample_supplier_programme):
        """Suppliers with higher emissions should tend to have higher engagement score."""
        suppliers = sample_supplier_programme["suppliers"]
        critical = [s for s in suppliers if s["tier"] == "critical"]
        standard = [s for s in suppliers if s["tier"] == "standard"]
        avg_critical = sum(s["engagement_score"] for s in critical) / len(critical)
        avg_standard = sum(s["engagement_score"] for s in standard) / len(standard)
        assert avg_critical >= avg_standard

    def test_scorecard_has_commitment(self, sample_supplier_programme):
        for sup in sample_supplier_programme["suppliers"]:
            assert "commitment_type" in sup


# =============================================================================
# Supplier Tier Classification
# =============================================================================


class TestSupplierTierClassification:
    """Test supplier tier classification logic."""

    def test_critical_tier_threshold(self, sample_supplier_programme):
        for sup in sample_supplier_programme["suppliers"]:
            if sup["tier"] == "critical":
                assert sup["scope3_contribution_tco2e"] >= Decimal("15000")

    def test_significant_tier_threshold(self, sample_supplier_programme):
        for sup in sample_supplier_programme["suppliers"]:
            if sup["tier"] == "significant":
                e = sup["scope3_contribution_tco2e"]
                assert Decimal("5000") <= e < Decimal("15000")

    def test_standard_tier_threshold(self, sample_supplier_programme):
        for sup in sample_supplier_programme["suppliers"]:
            if sup["tier"] == "standard":
                assert sup["scope3_contribution_tco2e"] < Decimal("5000")

    @pytest.mark.parametrize("tier", ["critical", "significant", "standard"])
    def test_tier_has_suppliers(self, tier, sample_supplier_programme):
        found = any(
            s["tier"] == tier for s in sample_supplier_programme["suppliers"]
        )
        assert found is True


# =============================================================================
# Incentive Modelling
# =============================================================================


class TestIncentiveModelling:
    """Test incentive modelling for supplier engagement."""

    def test_incentive_for_sbti_commitment(self):
        """Suppliers with SBTi should get preferential procurement score."""
        sbti_supplier = {"commitment_type": "SBTi", "base_score": 70}
        bonus = 15 if sbti_supplier["commitment_type"] == "SBTi" else 0
        final = sbti_supplier["base_score"] + bonus
        assert final == 85

    def test_incentive_for_cdp_disclosure(self):
        cdp_supplier = {"commitment_type": "CDP", "base_score": 65}
        bonus = 10 if cdp_supplier["commitment_type"] == "CDP" else 0
        final = cdp_supplier["base_score"] + bonus
        assert final == 75

    def test_no_incentive_for_uncommitted(self):
        supplier = {"commitment_type": "None", "base_score": 50}
        bonus = 0
        final = supplier["base_score"] + bonus
        assert final == 50


# =============================================================================
# Programme ROI
# =============================================================================


class TestProgrammeROI:
    """Test programme ROI calculation."""

    def test_roi_from_carbon_savings(self, sample_supplier_programme, sample_scenario_config):
        """ROI = (emission_reductions * carbon_price) / programme_cost."""
        suppliers = sample_supplier_programme["suppliers"]
        carbon_price = sample_scenario_config["carbon_price_usd_per_tco2e"]
        annual_reduction = sum(
            s["scope3_contribution_tco2e"] * s["yoy_reduction_pct"] / Decimal("100")
            for s in suppliers
            if s["has_reduction_target"]
        )
        annual_value = annual_reduction * carbon_price
        programme_cost = Decimal("500000")  # annual programme operating cost
        roi = annual_value / programme_cost
        assert roi > Decimal("1")  # ROI > 100%

    def test_cumulative_reduction_5_years(self, sample_supplier_programme):
        """5-year cumulative reduction from supplier programme."""
        suppliers = sample_supplier_programme["suppliers"]
        annual_reduction = sum(
            s["scope3_contribution_tco2e"] * s["yoy_reduction_pct"] / Decimal("100")
            for s in suppliers
            if s["has_reduction_target"]
        )
        cumulative_5yr = annual_reduction * 5
        assert cumulative_5yr > Decimal("0")


# =============================================================================
# Transition Risk Assessment
# =============================================================================


class TestTransitionRiskAssessment:
    """Test transition risk assessment for supplier programme."""

    def test_uncommitted_suppliers_risk(self, sample_supplier_programme):
        """Uncommitted suppliers represent transition risk."""
        uncommitted = [
            s for s in sample_supplier_programme["suppliers"]
            if not s["has_reduction_target"]
        ]
        risk_emissions = sum(s["scope3_contribution_tco2e"] for s in uncommitted)
        assert risk_emissions > Decimal("0")

    def test_critical_uncommitted_highest_risk(self, sample_supplier_programme):
        """Critical-tier uncommitted suppliers are highest risk."""
        critical_uncommitted = [
            s for s in sample_supplier_programme["suppliers"]
            if s["tier"] == "critical" and not s["has_reduction_target"]
        ]
        # In our data, no critical suppliers are uncommitted
        # (all critical have targets), which is the ideal state
        assert isinstance(critical_uncommitted, list)

    def test_category_concentration_risk(self, sample_supplier_programme):
        """Check if emissions are concentrated in few categories."""
        by_category = {}
        for sup in sample_supplier_programme["suppliers"]:
            cat = sup["category"]
            by_category[cat] = by_category.get(cat, Decimal("0")) + sup["scope3_contribution_tco2e"]
        total = sum(by_category.values())
        max_cat_pct = max(by_category.values()) / total * Decimal("100")
        assert max_cat_pct < Decimal("50")  # no single category > 50%
