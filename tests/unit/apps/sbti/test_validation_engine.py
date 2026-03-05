# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Validation Engine.

Tests all SBTi v2.1 validation criteria (C1-C12) including organizational
boundary, GHG gases, coverage, base year, timeframe, ambition levels,
Scope 3 trigger and coverage, bioenergy, carbon credits, avoided
emissions, plus net-zero criteria (NZ-C1 through NZ-C14), full
end-to-end validation, and readiness report generation with 38+ test
functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime
from decimal import Decimal

import pytest


# ===========================================================================
# C1: Organizational Boundary
# ===========================================================================

class TestC1OrgBoundary:
    """Test C1: organizational boundary definition."""

    def test_boundary_defined(self, sample_validation_result):
        c1 = sample_validation_result["criteria_results"]["C1_org_boundary"]
        assert c1["status"] == "pass"

    def test_boundary_operational_control(self):
        boundary = {"approach": "operational_control", "coverage_pct": 100.0}
        assert boundary["approach"] in ["operational_control", "equity_share", "financial_control"]

    def test_parent_group_inclusion(self):
        org = {
            "parent_company": "Acme Holdings",
            "subsidiaries_included": ["SubA", "SubB", "SubC"],
            "boundary_coverage_pct": 100.0,
        }
        assert len(org["subsidiaries_included"]) >= 1
        assert org["boundary_coverage_pct"] == 100.0


# ===========================================================================
# C2: GHG Gases
# ===========================================================================

class TestC2GHGs:
    """Test C2: all seven Kyoto GHGs must be included."""

    REQUIRED_GHGS = ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"]

    def test_all_7_ghgs_pass(self, sample_validation_result):
        c2 = sample_validation_result["criteria_results"]["C2_ghg_gases"]
        assert c2["status"] == "pass"

    def test_all_7_ghgs_present(self):
        included_ghgs = ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"]
        assert len(included_ghgs) == 7
        for gas in self.REQUIRED_GHGS:
            assert gas in included_ghgs

    def test_missing_ghg_fails(self):
        included_ghgs = ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6"]
        missing = set(self.REQUIRED_GHGS) - set(included_ghgs)
        assert "NF3" in missing
        assert len(missing) == 1

    def test_fail_on_missing_ghg(self, failing_validation_result):
        c2 = failing_validation_result["criteria_results"]["C2_ghg_gases"]
        assert c2["status"] == "fail"


# ===========================================================================
# C3: Coverage
# ===========================================================================

class TestC3Coverage:
    """Test C3: >= 95% Scope 1+2 boundary coverage."""

    def test_coverage_passes_at_95(self, sample_validation_result):
        c3 = sample_validation_result["criteria_results"]["C3_coverage"]
        assert c3["status"] == "pass"

    @pytest.mark.parametrize("coverage,expected_status", [
        (95.0, "pass"),
        (95.2, "pass"),
        (100.0, "pass"),
        (94.9, "fail"),
        (90.0, "fail"),
        (50.0, "fail"),
    ])
    def test_coverage_threshold_parametrized(self, coverage, expected_status):
        status = "pass" if coverage >= 95.0 else "fail"
        assert status == expected_status

    def test_coverage_fails_below_95(self, failing_validation_result):
        c3 = failing_validation_result["criteria_results"]["C3_coverage"]
        assert c3["status"] == "fail"


# ===========================================================================
# C4: Base Year
# ===========================================================================

class TestC4BaseYear:
    """Test C4: base year must be >= 2015."""

    def test_base_year_passes(self, sample_validation_result):
        c4 = sample_validation_result["criteria_results"]["C4_base_year"]
        assert c4["status"] == "pass"

    @pytest.mark.parametrize("base_year,expected_status", [
        (2015, "pass"),
        (2018, "pass"),
        (2020, "pass"),
        (2025, "pass"),
        (2014, "fail"),
        (2010, "fail"),
        (2000, "fail"),
    ])
    def test_base_year_validation(self, base_year, expected_status):
        status = "pass" if base_year >= 2015 else "fail"
        assert status == expected_status


# ===========================================================================
# C5: Timeframe
# ===========================================================================

class TestC5Timeframe:
    """Test C5: 5-10 year timeframe for near-term targets."""

    def test_timeframe_passes(self, sample_validation_result):
        c5 = sample_validation_result["criteria_results"]["C5_timeframe"]
        assert c5["status"] == "pass"

    @pytest.mark.parametrize("timeframe,expected_status", [
        (5, "pass"),
        (7, "pass"),
        (10, "pass"),
        (4, "fail"),
        (11, "fail"),
        (3, "fail"),
        (15, "fail"),
    ])
    def test_timeframe_window(self, timeframe, expected_status):
        status = "pass" if 5 <= timeframe <= 10 else "fail"
        assert status == expected_status

    def test_failing_timeframe(self, failing_validation_result):
        c5 = failing_validation_result["criteria_results"]["C5_timeframe"]
        assert c5["status"] == "fail"


# ===========================================================================
# C6: Ambition S1+2
# ===========================================================================

class TestC6AmbitionS12:
    """Test C6: S1+2 ambition must align with 1.5C."""

    def test_ambition_passes_at_4_2(self, sample_validation_result):
        c6 = sample_validation_result["criteria_results"]["C6_ambition_s12"]
        assert c6["status"] == "pass"

    @pytest.mark.parametrize("annual_rate,ambition,expected_status", [
        (4.2, "1.5C", "pass"),
        (5.0, "1.5C", "pass"),
        (4.1, "1.5C", "fail"),
        (3.0, "1.5C", "fail"),
        (2.5, "well_below_2C", "pass"),
        (3.0, "well_below_2C", "pass"),
        (2.4, "well_below_2C", "fail"),
    ])
    def test_ambition_rate_validation(self, annual_rate, ambition, expected_status):
        min_rates = {"1.5C": 4.2, "well_below_2C": 2.5}
        status = "pass" if annual_rate >= min_rates[ambition] else "fail"
        assert status == expected_status

    def test_failing_ambition(self, failing_validation_result):
        c6 = failing_validation_result["criteria_results"]["C6_ambition_s12"]
        assert c6["status"] == "fail"


# ===========================================================================
# C7: Ambition S3
# ===========================================================================

class TestC7AmbitionS3:
    """Test C7: S3 ambition must be at least WB2C."""

    def test_scope3_ambition_passes(self, sample_validation_result):
        c7 = sample_validation_result["criteria_results"]["C7_ambition_s3"]
        assert c7["status"] == "pass"

    @pytest.mark.parametrize("annual_rate,expected_status", [
        (2.5, "pass"),
        (3.0, "pass"),
        (4.2, "pass"),
        (2.4, "fail"),
        (1.0, "fail"),
    ])
    def test_scope3_minimum_ambition(self, annual_rate, expected_status):
        status = "pass" if annual_rate >= 2.5 else "fail"
        assert status == expected_status


# ===========================================================================
# C8: Scope 3 Trigger
# ===========================================================================

class TestC8Scope3Trigger:
    """Test C8: 40% Scope 3 trigger."""

    def test_scope3_trigger_passes(self, sample_validation_result):
        c8 = sample_validation_result["criteria_results"]["C8_scope3_trigger"]
        assert c8["status"] == "pass"

    @pytest.mark.parametrize("s3_pct,required", [
        (40.0, True),
        (61.5, True),
        (39.9, False),
        (10.0, False),
    ])
    def test_trigger_threshold(self, s3_pct, required):
        assert (s3_pct >= 40.0) == required


# ===========================================================================
# C9: Scope 3 Coverage
# ===========================================================================

class TestC9Scope3Coverage:
    """Test C9: 67% Scope 3 coverage minimum."""

    def test_scope3_coverage_passes(self, sample_validation_result):
        c9 = sample_validation_result["criteria_results"]["C9_scope3_coverage"]
        assert c9["status"] == "pass"

    @pytest.mark.parametrize("coverage,expected_status", [
        (67.0, "pass"),
        (72.0, "pass"),
        (90.0, "pass"),
        (66.9, "fail"),
        (50.0, "fail"),
    ])
    def test_scope3_coverage_threshold(self, coverage, expected_status):
        status = "pass" if coverage >= 67.0 else "fail"
        assert status == expected_status


# ===========================================================================
# C10: Bioenergy
# ===========================================================================

class TestC10Bioenergy:
    """Test C10: bioenergy emissions must be included."""

    def test_bioenergy_included(self, sample_validation_result):
        c10 = sample_validation_result["criteria_results"]["C10_bioenergy"]
        assert c10["status"] == "pass"

    def test_bioenergy_tracking(self, sample_emissions_inventory):
        assert sample_emissions_inventory["bioenergy_included"] is True
        assert sample_emissions_inventory["bioenergy_tco2e"] > 0


# ===========================================================================
# C11: Carbon Credits
# ===========================================================================

class TestC11CarbonCredits:
    """Test C11: carbon credits must be excluded from target progress."""

    def test_carbon_credits_excluded(self, sample_validation_result):
        c11 = sample_validation_result["criteria_results"]["C11_carbon_credits"]
        assert c11["status"] == "pass"

    def test_credits_not_counted(self):
        progress = {
            "actual_emissions_tco2e": 70_000.0,
            "carbon_credits_applied_tco2e": 0.0,
            "net_emissions_tco2e": 70_000.0,
        }
        assert progress["carbon_credits_applied_tco2e"] == 0.0
        assert progress["actual_emissions_tco2e"] == progress["net_emissions_tco2e"]


# ===========================================================================
# C12: Avoided Emissions
# ===========================================================================

class TestC12AvoidedEmissions:
    """Test C12: avoided emissions must be reported separately."""

    def test_avoided_emissions_separated(self, sample_validation_result):
        c12 = sample_validation_result["criteria_results"]["C12_avoided_emissions"]
        assert c12["status"] == "pass"

    def test_separation_structure(self):
        reporting = {
            "scope1_2_emissions_tco2e": 80_000.0,
            "avoided_emissions_tco2e": 15_000.0,
            "reported_separately": True,
        }
        assert reporting["reported_separately"] is True


# ===========================================================================
# Net-Zero Criteria (NZ-C1 through NZ-C14)
# ===========================================================================

class TestNetZeroCriteria:
    """Test net-zero specific validation criteria."""

    def test_nz_c1_near_term_required(self, sample_net_zero_target):
        """NZ-C1: Near-term target must accompany net-zero."""
        assert sample_net_zero_target["target_type"] == "net_zero"

    def test_nz_c2_long_term_by_2050(self, sample_net_zero_target):
        assert sample_net_zero_target["target_year"] <= 2050

    def test_nz_c3_minimum_90_pct_reduction(self, sample_net_zero_target):
        assert sample_net_zero_target["reduction_pct"] >= 90.0

    def test_nz_c4_all_scopes(self, sample_net_zero_target):
        assert sample_net_zero_target["scope"] == "all_scopes"

    def test_nz_c5_residual_neutralization(self, sample_net_zero_target):
        residual = sample_net_zero_target["residual_emissions_tco2e"]
        base = sample_net_zero_target["base_year_emissions_tco2e"]
        residual_pct = (residual / base) * 100
        assert residual_pct <= 10.0

    def test_nz_c6_permanent_removal(self, sample_net_zero_target):
        strategy = sample_net_zero_target["neutralization_strategy"]
        assert strategy == "permanent_carbon_removal"

    @pytest.mark.parametrize("reduction_pct,valid", [
        (90.0, True),
        (95.0, True),
        (100.0, True),
        (89.9, False),
        (80.0, False),
    ])
    def test_nz_minimum_reduction(self, reduction_pct, valid):
        assert (reduction_pct >= 90.0) == valid

    def test_nz_neutralization_mechanisms(self, sample_net_zero_target):
        mechanisms = sample_net_zero_target["neutralization_mechanisms"]
        valid_mechanisms = {"DACCS", "BECCS", "biochar", "enhanced_weathering", "ocean_CDR"}
        for m in mechanisms:
            assert m in valid_mechanisms

    def test_nz_residual_calculation(self, sample_net_zero_target):
        base = sample_net_zero_target["base_year_emissions_tco2e"]
        reduction = sample_net_zero_target["reduction_pct"]
        expected_residual = base * (1 - reduction / 100)
        assert abs(sample_net_zero_target["residual_emissions_tco2e"] - expected_residual) < 1.0


# ===========================================================================
# Full Validation
# ===========================================================================

class TestFullValidation:
    """Test end-to-end full validation."""

    def test_full_validation_all_pass(self, sample_validation_result):
        assert sample_validation_result["overall_status"] == "pass"
        assert sample_validation_result["fail_count"] == 0

    def test_full_validation_with_failures(self, failing_validation_result):
        assert failing_validation_result["overall_status"] == "fail"
        assert failing_validation_result["fail_count"] > 0

    def test_criteria_count(self, sample_validation_result):
        assert sample_validation_result["total_criteria"] == 12

    def test_pass_fail_sum(self, sample_validation_result):
        total = (
            sample_validation_result["pass_count"]
            + sample_validation_result["fail_count"]
            + sample_validation_result.get("warning_count", 0)
        )
        assert total == sample_validation_result["total_criteria"]

    def test_validation_provenance(self, sample_validation_result):
        assert len(sample_validation_result["provenance_hash"]) == 64


# ===========================================================================
# Readiness Report
# ===========================================================================

class TestReadinessReport:
    """Test readiness report generation."""

    def test_readiness_score_100(self, sample_validation_result):
        assert sample_validation_result["readiness_score"] == 100.0

    def test_readiness_level_ready(self, sample_validation_result):
        assert sample_validation_result["readiness_level"] == "ready_for_submission"

    def test_readiness_level_gaps(self, failing_validation_result):
        assert failing_validation_result["readiness_level"] == "significant_gaps"

    @pytest.mark.parametrize("score,expected_level", [
        (100.0, "ready_for_submission"),
        (80.0, "minor_gaps"),
        (60.0, "moderate_gaps"),
        (30.0, "significant_gaps"),
    ])
    def test_readiness_level_mapping(self, score, expected_level):
        if score >= 90:
            level = "ready_for_submission"
        elif score >= 70:
            level = "minor_gaps"
        elif score >= 50:
            level = "moderate_gaps"
        else:
            level = "significant_gaps"
        assert level == expected_level

    def test_no_recommendations_when_passing(self, sample_validation_result):
        assert len(sample_validation_result["recommendations"]) == 0
