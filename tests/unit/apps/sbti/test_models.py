# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Platform Domain Models.

Tests all domain models including organization, emissions inventory, targets,
pathways, validation results, Scope 3 screening, FLAG assessments, progress
records, temperature scores, FI portfolios, recalculations, five-year
reviews, and gap assessments with 35+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import date, datetime
from decimal import Decimal

import pytest


# ===========================================================================
# Helper tests
# ===========================================================================

class TestHelpers:
    """Test utility functions from conftest."""

    def test_new_id_returns_string(self):
        from conftest import _new_id
        result = _new_id()
        assert isinstance(result, str)
        assert len(result) == 36

    def test_new_id_unique(self):
        from conftest import _new_id
        ids = {_new_id() for _ in range(200)}
        assert len(ids) == 200

    def test_sha256_deterministic(self):
        from conftest import _sha256
        h1 = _sha256("sbti_test_payload")
        h2 = _sha256("sbti_test_payload")
        assert h1 == h2

    def test_sha256_length_64(self):
        from conftest import _sha256
        result = _sha256("sbti data")
        assert len(result) == 64

    def test_sha256_different_inputs(self):
        from conftest import _sha256
        h1 = _sha256("input_alpha")
        h2 = _sha256("input_beta")
        assert h1 != h2


# ===========================================================================
# Organization model tests
# ===========================================================================

class TestOrganization:
    """Test organization model creation and validation."""

    def test_create_organization(self, sample_organization):
        assert sample_organization["name"] == "Acme Manufacturing Corp"
        assert sample_organization["sector"] == "manufacturing"
        assert len(sample_organization["id"]) == 36

    def test_organization_sector_classification(self, sample_organization):
        assert sample_organization["isic_code"] == "2394"
        assert sample_organization["nace_code"] == "C23.51"
        assert sample_organization["naics_code"] == "327310"

    def test_organization_oecd_status(self, sample_organization):
        assert sample_organization["oecd_member"] is True

    def test_organization_sbti_status(self, sample_organization):
        assert sample_organization["sbti_status"] == "committed"
        assert isinstance(sample_organization["commitment_date"], date)

    def test_financial_organization(self, financial_organization):
        assert financial_organization["sector"] == "financial_services"
        assert financial_organization["industry"] == "Commercial Banking"

    def test_flag_organization(self, flag_organization):
        assert flag_organization["sector"] == "food_agriculture"
        assert flag_organization["isic_code"] == "0111"

    def test_organization_timestamps(self, sample_organization):
        assert isinstance(sample_organization["created_at"], datetime)
        assert isinstance(sample_organization["updated_at"], datetime)


# ===========================================================================
# Emissions inventory model tests
# ===========================================================================

class TestEmissionsInventory:
    """Test emissions inventory model."""

    def test_create_inventory(self, sample_emissions_inventory):
        assert sample_emissions_inventory["reporting_year"] == 2024
        assert sample_emissions_inventory["scope1_tco2e"] == 50_000.0

    def test_scope3_percentage_calculation(self, sample_emissions_inventory):
        total = sample_emissions_inventory["total_s123_tco2e"]
        s3 = sample_emissions_inventory["scope3_tco2e"]
        expected_pct = round((s3 / total) * 100, 2)
        assert abs(sample_emissions_inventory["scope3_pct_of_total"] - expected_pct) < 0.1

    def test_scope3_categories_complete(self, sample_emissions_inventory):
        categories = sample_emissions_inventory["scope3_categories"]
        assert len(categories) == 15
        for cat_num in range(1, 16):
            assert cat_num in categories

    def test_scope3_categories_sum(self, sample_emissions_inventory):
        cat_sum = sum(sample_emissions_inventory["scope3_categories"].values())
        assert cat_sum == sample_emissions_inventory["scope3_tco2e"]

    def test_flag_percentage(self, sample_emissions_inventory):
        flag_pct = sample_emissions_inventory["flag_pct_of_total"]
        assert flag_pct < 20.0  # Below FLAG threshold

    def test_bioenergy_included(self, sample_emissions_inventory):
        assert sample_emissions_inventory["bioenergy_included"] is True
        assert sample_emissions_inventory["bioenergy_tco2e"] > 0

    def test_inventory_provenance(self, sample_emissions_inventory):
        assert len(sample_emissions_inventory["provenance_hash"]) == 64

    def test_high_scope3_inventory(self, high_scope3_inventory):
        assert high_scope3_inventory["scope3_pct_of_total"] >= 40.0

    def test_low_scope3_inventory(self, low_scope3_inventory):
        assert low_scope3_inventory["scope3_pct_of_total"] < 40.0

    def test_high_flag_inventory(self, high_flag_inventory):
        assert high_flag_inventory["flag_pct_of_total"] >= 20.0


# ===========================================================================
# Target model tests
# ===========================================================================

class TestTarget:
    """Test target model creation and validation."""

    def test_create_near_term_target(self, sample_near_term_target):
        assert sample_near_term_target["target_type"] == "near_term"
        assert sample_near_term_target["scope"] == "scope_1_2"
        assert sample_near_term_target["ambition_level"] == "1.5C"

    def test_create_long_term_target(self, sample_long_term_target):
        assert sample_long_term_target["target_type"] == "long_term"
        assert sample_long_term_target["target_year"] == 2050
        assert sample_long_term_target["reduction_pct"] == 90.0

    def test_create_net_zero_target(self, sample_net_zero_target):
        assert sample_net_zero_target["target_type"] == "net_zero"
        assert sample_net_zero_target["residual_emissions_tco2e"] > 0
        assert sample_net_zero_target["neutralization_strategy"] == "permanent_carbon_removal"

    def test_create_scope3_target(self, sample_scope3_target):
        assert sample_scope3_target["scope"] == "scope_3"
        assert len(sample_scope3_target["scope3_categories_included"]) > 0

    def test_create_flag_target(self, sample_flag_target):
        assert sample_flag_target["is_flag_target"] is True
        assert sample_flag_target["flag_commodity"] == "cattle"
        assert sample_flag_target["deforestation_commitment"] is True

    def test_create_intensity_target(self, sample_intensity_target):
        assert sample_intensity_target["method"] == "intensity_physical"
        assert sample_intensity_target["intensity_metric"] is not None

    def test_target_status_default(self, sample_near_term_target):
        assert sample_near_term_target["status"] == "draft"

    def test_target_annual_rate(self, sample_near_term_target):
        expected_rate = sample_near_term_target["reduction_pct"] / (
            sample_near_term_target["target_year"] - sample_near_term_target["base_year"]
        )
        assert abs(sample_near_term_target["linear_annual_reduction_pct"] - expected_rate) < 0.01

    def test_target_provenance(self, sample_near_term_target):
        assert len(sample_near_term_target["provenance_hash"]) == 64

    def test_net_zero_neutralization_mechanisms(self, sample_net_zero_target):
        mechanisms = sample_net_zero_target["neutralization_mechanisms"]
        assert len(mechanisms) >= 1
        assert "DACCS" in mechanisms

    def test_target_valid_status_values(self):
        valid = {"draft", "pending_validation", "submitted", "validated",
                 "approved", "active", "expired", "withdrawn"}
        assert "draft" in valid
        assert "validated" in valid


# ===========================================================================
# Pathway model tests
# ===========================================================================

class TestPathway:
    """Test pathway model creation and validation."""

    def test_create_aca_pathway(self, sample_pathway):
        assert sample_pathway["pathway_type"] == "aca"
        assert sample_pathway["ambition_level"] == "1.5C"

    def test_pathway_milestones(self, sample_pathway):
        milestones = sample_pathway["milestones"]
        assert len(milestones) == 10  # 2021-2030
        # Each year should be lower than the previous
        prev_value = sample_pathway["base_emissions_tco2e"]
        for year in sorted(milestones.keys()):
            assert milestones[year] < prev_value
            prev_value = milestones[year]

    def test_pathway_uncertainty_bounds(self, sample_pathway):
        assert sample_pathway["uncertainty_lower_pct"] < sample_pathway["uncertainty_upper_pct"]

    def test_sda_pathway(self, sda_pathway):
        assert sda_pathway["pathway_type"] == "sda"
        assert sda_pathway["sector"] == "cement"
        assert sda_pathway["convergence_intensity_2050"] < sda_pathway["base_intensity"]

    def test_flag_commodity_pathway(self, flag_commodity_pathway):
        assert flag_commodity_pathway["pathway_type"] == "flag_commodity"
        assert flag_commodity_pathway["commodity"] == "cattle"
        assert flag_commodity_pathway["annual_reduction_rate_pct"] == 3.03


# ===========================================================================
# Validation result model tests
# ===========================================================================

class TestValidationResult:
    """Test validation result model."""

    def test_create_passing_result(self, sample_validation_result):
        assert sample_validation_result["overall_status"] == "pass"
        assert sample_validation_result["pass_count"] == 12
        assert sample_validation_result["fail_count"] == 0

    def test_readiness_calculation(self, sample_validation_result):
        score = sample_validation_result["readiness_score"]
        assert score == 100.0
        assert sample_validation_result["readiness_level"] == "ready_for_submission"

    def test_failing_result(self, failing_validation_result):
        assert failing_validation_result["overall_status"] == "fail"
        assert failing_validation_result["fail_count"] == 4
        assert failing_validation_result["readiness_score"] < 50.0

    def test_criteria_results_present(self, sample_validation_result):
        criteria = sample_validation_result["criteria_results"]
        assert "C1_org_boundary" in criteria
        assert "C6_ambition_s12" in criteria

    def test_validation_provenance(self, sample_validation_result):
        assert len(sample_validation_result["provenance_hash"]) == 64


# ===========================================================================
# Scope 3 screening model tests
# ===========================================================================

class TestScope3Screening:
    """Test Scope 3 screening model."""

    def test_trigger_assessment_required(self, sample_scope3_screening):
        assert sample_scope3_screening["scope3_target_required"] is True
        assert sample_scope3_screening["scope3_pct_of_total"] >= 40.0

    def test_category_breakdown_complete(self, sample_scope3_screening):
        categories = sample_scope3_screening["category_breakdown"]
        assert len(categories) == 15

    def test_hotspot_categories(self, sample_scope3_screening):
        hotspots = sample_scope3_screening["hotspot_categories"]
        assert len(hotspots) > 0
        # Top category should be Cat 1 (Purchased Goods)
        assert 1 in hotspots

    def test_coverage_recommendation(self, sample_scope3_screening):
        assert sample_scope3_screening["recommended_coverage_pct"] >= 67.0

    def test_screening_provenance(self, sample_scope3_screening):
        assert len(sample_scope3_screening["provenance_hash"]) == 64


# ===========================================================================
# FLAG assessment model tests
# ===========================================================================

class TestFLAGAssessment:
    """Test FLAG assessment model."""

    def test_flag_trigger_required(self, sample_flag_assessment):
        assert sample_flag_assessment["flag_target_required"] is True
        assert sample_flag_assessment["flag_pct_of_total"] >= 20.0

    def test_flag_not_triggered(self, non_flag_assessment):
        assert non_flag_assessment["flag_target_required"] is False
        assert non_flag_assessment["flag_pct_of_total"] < 20.0

    def test_commodity_breakdown(self, sample_flag_assessment):
        commodities = sample_flag_assessment["commodity_breakdown"]
        assert "cattle" in commodities
        total_pct = sum(c["pct"] for c in commodities.values())
        assert abs(total_pct - 100.0) < 0.1

    def test_deforestation_commitment(self, sample_flag_assessment):
        assert sample_flag_assessment["deforestation_commitment"] is True
        assert isinstance(sample_flag_assessment["deforestation_target_date"], date)

    def test_flag_long_term(self, sample_flag_assessment):
        assert sample_flag_assessment["long_term_reduction_pct"] == 72.0


# ===========================================================================
# Progress record model tests
# ===========================================================================

class TestProgressRecord:
    """Test progress record model."""

    def test_create_progress(self, sample_progress_record):
        assert sample_progress_record["reporting_year"] == 2024
        assert sample_progress_record["actual_emissions_tco2e"] > 0

    def test_variance_calculation(self, sample_progress_record):
        expected = sample_progress_record["expected_emissions_tco2e"]
        actual = sample_progress_record["actual_emissions_tco2e"]
        variance_tco2e = actual - expected
        assert abs(sample_progress_record["variance_tco2e"] - abs(variance_tco2e)) < 1.0

    def test_rag_status_amber(self, sample_progress_record):
        assert sample_progress_record["rag_status"] == "amber"
        assert sample_progress_record["on_track"] is False

    def test_rag_status_green(self, on_track_progress):
        assert on_track_progress["rag_status"] == "green"
        assert on_track_progress["on_track"] is True

    def test_scope_breakdown(self, sample_progress_record):
        breakdown = sample_progress_record["scope_breakdown"]
        assert "scope_1" in breakdown
        assert "scope_2" in breakdown


# ===========================================================================
# Temperature score model tests
# ===========================================================================

class TestTemperatureScore:
    """Test temperature score model."""

    def test_overall_score_range(self, sample_temperature_score):
        score = sample_temperature_score["overall_score_c"]
        assert 1.0 <= score <= 4.0

    def test_scope_temperatures(self, sample_temperature_score):
        assert sample_temperature_score["scope1_score_c"] <= sample_temperature_score["scope3_score_c"]

    def test_high_temperature_score(self, high_temperature_score):
        assert high_temperature_score["overall_score_c"] >= 3.0
        assert high_temperature_score["peer_percentile"] > 50

    def test_sector_comparison(self, sample_temperature_score):
        assert sample_temperature_score["overall_score_c"] < sample_temperature_score["sector_average_c"]


# ===========================================================================
# FI portfolio model tests
# ===========================================================================

class TestFIPortfolio:
    """Test FI portfolio model."""

    def test_create_portfolio(self, sample_fi_portfolio):
        assert sample_fi_portfolio["portfolio_name"] == "Corporate Lending Portfolio"
        assert sample_fi_portfolio["total_financed_emissions_tco2e"] > 0

    def test_holdings_present(self, sample_fi_portfolio):
        holdings = sample_fi_portfolio["holdings"]
        assert len(holdings) == 3

    def test_coverage_calculation(self, sample_fi_portfolio):
        holdings = sample_fi_portfolio["holdings"]
        with_target = sum(1 for h in holdings if h["has_sbti_target"])
        coverage_pct = (with_target / len(holdings)) * 100
        assert abs(sample_fi_portfolio["coverage_with_sbti_pct"] - coverage_pct) < 1.0

    def test_waci(self, sample_fi_portfolio):
        assert sample_fi_portfolio["waci"] > 0
        assert sample_fi_portfolio["waci_unit"] == "tCO2e per USD million invested"

    def test_pcaf_quality(self, sample_fi_portfolio):
        assert 1.0 <= sample_fi_portfolio["pcaf_avg_data_quality"] <= 5.0

    def test_coverage_path(self, sample_fi_portfolio):
        path = sample_fi_portfolio["fi_coverage_path"]
        assert path[2040] == 100.0
        # Should be monotonically increasing
        years = sorted(path.keys())
        for i in range(1, len(years)):
            assert path[years[i]] >= path[years[i - 1]]


# ===========================================================================
# Recalculation model tests
# ===========================================================================

class TestRecalculation:
    """Test recalculation model."""

    def test_threshold_exceeded(self, sample_recalculation):
        assert sample_recalculation["exceeds_threshold"] is True
        assert sample_recalculation["change_pct"] > sample_recalculation["recalculation_threshold_pct"]

    def test_threshold_not_exceeded(self, minor_recalculation):
        assert minor_recalculation["exceeds_threshold"] is False
        assert minor_recalculation["revalidation_required"] is False

    def test_revalidation_required(self, sample_recalculation):
        assert sample_recalculation["revalidation_required"] is True

    def test_audit_trail(self, sample_recalculation):
        trail = sample_recalculation["audit_trail"]
        assert len(trail) >= 2


# ===========================================================================
# Five-year review model tests
# ===========================================================================

class TestFiveYearReview:
    """Test five-year review model."""

    def test_review_dates(self, sample_five_year_review):
        trigger = sample_five_year_review["review_trigger_date"]
        original = sample_five_year_review["original_validation_date"]
        assert (trigger.year - original.year) == 5

    def test_deadline_calculation(self, sample_five_year_review):
        trigger = sample_five_year_review["review_trigger_date"]
        deadline = sample_five_year_review["review_deadline"]
        assert (deadline.year - trigger.year) == 1

    def test_notification_schedule(self, sample_five_year_review):
        schedule = sample_five_year_review["notification_schedule"]
        assert len(schedule) >= 4
        assert schedule[0]["months_before"] == 12

    def test_review_status(self, sample_five_year_review):
        assert sample_five_year_review["review_status"] == "upcoming"

    def test_readiness_score(self, sample_five_year_review):
        assert 0 <= sample_five_year_review["readiness_score"] <= 100.0


# ===========================================================================
# Gap assessment model tests
# ===========================================================================

class TestGapAssessment:
    """Test gap assessment model."""

    def test_create_gap_assessment(self, sample_gap_assessment):
        assert sample_gap_assessment["readiness_level"] == "moderate_gaps"
        assert sample_gap_assessment["overall_readiness_score"] == 65.0

    def test_data_gaps(self, sample_gap_assessment):
        gaps = sample_gap_assessment["data_gaps"]
        assert len(gaps) >= 1
        severities = {g["severity"] for g in gaps}
        assert "high" in severities

    def test_ambition_gaps(self, sample_gap_assessment):
        gaps = sample_gap_assessment["ambition_gaps"]
        assert len(gaps) >= 1
        assert gaps[0]["shortfall_pct"] > 0

    def test_action_plan_ordered(self, sample_gap_assessment):
        plan = sample_gap_assessment["action_plan"]
        assert len(plan) >= 2
        # Should be ordered by priority
        for i in range(1, len(plan)):
            assert plan[i]["priority"] >= plan[i - 1]["priority"]

    def test_peer_benchmark(self, sample_gap_assessment):
        benchmark = sample_gap_assessment["peer_benchmark"]
        assert benchmark["sector"] == "manufacturing"
        assert benchmark["org_percentile"] > 0

    def test_gap_provenance(self, sample_gap_assessment):
        assert len(sample_gap_assessment["provenance_hash"]) == 64
