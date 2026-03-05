# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Platform Domain Models.

Tests all domain models including organizations (non-financial, financial,
insurance, asset manager), economic activities, NACE mappings, eligibility
screening results, SC assessments, TSC evaluations, DNSH assessments,
safeguard assessments, KPI calculations, activity financials, CapEx plans,
GAR calculations, exposures, alignment results, portfolio alignments,
reports, evidence items, regulatory versions, data quality scores, gap
assessments, and gap items with 60+ test functions.

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
        h1 = _sha256("taxonomy_test_payload")
        h2 = _sha256("taxonomy_test_payload")
        assert h1 == h2

    def test_sha256_length_64(self):
        from conftest import _sha256
        result = _sha256("taxonomy data")
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

    def test_create_non_financial_org(self, sample_org):
        assert sample_org["name"] == "EuroManufacturing GmbH"
        assert sample_org["entity_type"] == "non_financial"
        assert len(sample_org["id"]) == 36

    def test_org_sector(self, sample_org):
        assert sample_org["sector"] == "manufacturing"
        assert sample_org["country"] == "DE"

    def test_org_lei(self, sample_org):
        assert sample_org["lei"] == "529900HNOAA1KXQJUQ27"
        assert len(sample_org["lei"]) == 20

    def test_org_reporting_flags(self, sample_org):
        assert sample_org["nfrd_reporting"] is True
        assert sample_org["csrd_reporting"] is True

    def test_org_financial_data(self, sample_org):
        assert sample_org["annual_revenue"] == Decimal("2500000000.00")
        assert sample_org["total_assets"] == Decimal("4200000000.00")
        assert sample_org["employee_count"] == 8500

    def test_org_settings(self, sample_org):
        settings = sample_org["settings"]
        assert settings["default_period"] == "FY2025"
        assert settings["base_currency"] == "EUR"

    def test_org_timestamps(self, sample_org):
        assert isinstance(sample_org["created_at"], datetime)
        assert isinstance(sample_org["updated_at"], datetime)

    def test_financial_institution(self, sample_financial_institution):
        assert sample_financial_institution["entity_type"] == "financial"
        assert sample_financial_institution["sector"] == "banking"
        assert sample_financial_institution["country"] == "NL"
        assert sample_financial_institution["settings"]["gar_enabled"] is True

    def test_insurance_entity(self, sample_insurance_entity):
        assert sample_insurance_entity["entity_type"] == "insurance"
        assert sample_insurance_entity["sector"] == "insurance"
        assert sample_insurance_entity["country"] == "FR"

    def test_asset_manager(self, sample_asset_manager):
        assert sample_asset_manager["entity_type"] == "asset_manager"
        assert sample_asset_manager["sector"] == "asset_management"
        assert sample_asset_manager["country"] == "LU"


# ===========================================================================
# Economic activity model tests
# ===========================================================================

class TestEconomicActivity:
    """Test economic activity model."""

    def test_activity_count(self, sample_activities):
        assert len(sample_activities) == 13

    def test_activity_code_format(self, sample_activities):
        for activity in sample_activities:
            assert len(activity["activity_code"]) >= 3
            assert "_" in activity["activity_code"]

    def test_solar_pv_activity(self, single_activity):
        assert single_activity["activity_code"] == "CCM_4.1"
        assert single_activity["sector"] == "energy"
        assert single_activity["activity_type"] == "own_performance"
        assert single_activity["delegated_act"] == "climate"
        assert "climate_mitigation" in single_activity["objectives"]

    def test_transitional_activity_type(self, transitional_activity):
        assert transitional_activity["activity_code"] == "CCM_3.3"
        assert transitional_activity["activity_type"] == "transitional"

    def test_enabling_activity_type(self, enabling_activity):
        assert enabling_activity["activity_code"] == "CCM_8.1"
        assert enabling_activity["activity_type"] == "enabling"

    def test_nace_codes_present(self, sample_activities):
        for activity in sample_activities:
            assert len(activity["nace_codes"]) >= 1

    def test_sc_criteria_present(self, sample_activities):
        for activity in sample_activities:
            assert "type" in activity["sc_criteria"]

    def test_objectives_coverage(self, sample_activities):
        all_objectives = set()
        for activity in sample_activities:
            all_objectives.update(activity["objectives"])
        expected = {
            "climate_mitigation", "climate_adaptation",
            "water_marine", "circular_economy",
            "pollution_prevention", "biodiversity",
        }
        assert all_objectives == expected

    def test_delegated_acts(self, sample_activities):
        acts = {a["delegated_act"] for a in sample_activities}
        assert "climate" in acts
        assert "environmental" in acts

    def test_effective_dates(self, sample_activities):
        for activity in sample_activities:
            assert isinstance(activity["effective_date"], date)
            assert activity["effective_date"].year >= 2022


# ===========================================================================
# NACE mapping model tests
# ===========================================================================

class TestNACEMapping:
    """Test NACE mapping model."""

    def test_nace_mapping_count(self, sample_nace_mappings):
        assert len(sample_nace_mappings) == 8

    def test_nace_levels(self, sample_nace_mappings):
        levels = {m["nace_level"] for m in sample_nace_mappings}
        assert levels == {1, 2, 3, 4}

    def test_level4_has_activities(self, sample_nace_mappings):
        level4 = [m for m in sample_nace_mappings if m["nace_level"] == 4]
        for mapping in level4:
            assert len(mapping["taxonomy_activities"]) >= 1

    def test_electricity_nace_mapping(self, sample_nace_mappings):
        d3511 = next(m for m in sample_nace_mappings if m["nace_code"] == "D35.11")
        assert "CCM_4.1" in d3511["taxonomy_activities"]
        assert "CCM_4.3" in d3511["taxonomy_activities"]

    def test_parent_code_reference(self, sample_nace_mappings):
        for mapping in sample_nace_mappings:
            if mapping["nace_level"] == 1:
                assert mapping["parent_code"] is None
            else:
                assert mapping["parent_code"] is not None


# ===========================================================================
# Eligibility screening model tests
# ===========================================================================

class TestEligibilityScreening:
    """Test eligibility screening model."""

    def test_screening_counts(self, sample_screening):
        total = sample_screening["total_activities"]
        eligible = sample_screening["eligible_count"]
        not_eligible = sample_screening["not_eligible_count"]
        de_minimis = sample_screening["de_minimis_excluded"]
        assert total == eligible + not_eligible + de_minimis

    def test_screening_status(self, sample_screening):
        assert sample_screening["status"] == "completed"

    def test_screening_period(self, sample_screening):
        assert sample_screening["period"] == "FY2025"

    def test_screening_results(self, sample_screening_results):
        assert len(sample_screening_results) == 5
        eligible = [r for r in sample_screening_results if r["eligible"]]
        assert len(eligible) == 3

    def test_screening_confidence(self, sample_screening_results):
        for result in sample_screening_results:
            assert result["confidence"] >= 0
            assert result["confidence"] <= 100

    def test_de_minimis_result(self, sample_screening_results):
        de_minimis = [r for r in sample_screening_results if r["de_minimis"]]
        assert len(de_minimis) == 1
        assert de_minimis[0]["eligible"] is False


# ===========================================================================
# SC assessment model tests
# ===========================================================================

class TestSCAssessment:
    """Test Substantial Contribution assessment model."""

    def test_sc_assessment_pass(self, sample_sc_assessment):
        assert sample_sc_assessment["overall_pass"] is True
        assert sample_sc_assessment["sc_type"] == "own_performance"

    def test_sc_assessment_objective(self, sample_sc_assessment):
        assert sample_sc_assessment["objective"] == "climate_mitigation"

    def test_sc_threshold_checks(self, sample_sc_assessment):
        checks = sample_sc_assessment["threshold_checks"]
        assert checks["life_cycle_ghg"]["pass"] is True
        assert checks["life_cycle_ghg"]["actual_gco2e_kwh"] < checks["life_cycle_ghg"]["threshold_gco2e_kwh"]

    def test_sc_evidence_items(self, sample_sc_assessment):
        assert len(sample_sc_assessment["evidence_items"]) == 2

    def test_cement_sc_transitional(self, cement_sc_assessment):
        assert cement_sc_assessment["sc_type"] == "transitional"
        assert cement_sc_assessment["overall_pass"] is True

    def test_steel_sc_failing(self, steel_sc_assessment):
        assert steel_sc_assessment["overall_pass"] is False
        checks = steel_sc_assessment["threshold_checks"]
        assert checks["hot_metal_emissions"]["pass"] is False

    def test_sc_provenance(self, sample_sc_assessment):
        assert len(sample_sc_assessment["provenance_hash"]) == 64


# ===========================================================================
# TSC evaluation model tests
# ===========================================================================

class TestTSCEvaluation:
    """Test Technical Screening Criteria evaluation model."""

    def test_tsc_evaluations_count(self, sample_tsc_evaluations):
        assert len(sample_tsc_evaluations) == 2

    def test_tsc_numeric_criterion(self, sample_tsc_evaluations):
        ghg_eval = sample_tsc_evaluations[0]
        assert ghg_eval["threshold_value"] == Decimal("100.0000")
        assert ghg_eval["actual_value"] == Decimal("25.4000")
        assert ghg_eval["pass_result"] is True
        assert ghg_eval["unit"] == "gCO2e/kWh"

    def test_tsc_qualitative_criterion(self, sample_tsc_evaluations):
        cert_eval = sample_tsc_evaluations[1]
        assert cert_eval["threshold_value"] is None
        assert cert_eval["pass_result"] is True


# ===========================================================================
# DNSH assessment model tests
# ===========================================================================

class TestDNSHAssessment:
    """Test DNSH assessment model."""

    def test_dnsh_pass(self, sample_dnsh_assessment):
        assert sample_dnsh_assessment["overall_pass"] is True
        assert sample_dnsh_assessment["sc_objective"] == "climate_mitigation"

    def test_dnsh_objective_results_all_pass(self, sample_dnsh_assessment):
        results = sample_dnsh_assessment["objective_results"]
        for obj, result in results.items():
            assert result["status"] in ("pass", "not_applicable")

    def test_dnsh_fail(self, failing_dnsh_assessment):
        assert failing_dnsh_assessment["overall_pass"] is False
        results = failing_dnsh_assessment["objective_results"]
        assert results["pollution_prevention"]["status"] == "fail"

    def test_dnsh_objective_detail_records(self, sample_dnsh_objective_results):
        assert len(sample_dnsh_objective_results) == 5
        objectives = {r["objective"] for r in sample_dnsh_objective_results}
        expected = {"climate_adaptation", "water_marine", "circular_economy", "pollution_prevention", "biodiversity"}
        assert objectives == expected


# ===========================================================================
# Climate risk assessment model tests
# ===========================================================================

class TestClimateRiskAssessment:
    """Test climate risk assessment model."""

    def test_climate_risk_status(self, sample_climate_risk_assessment):
        assert sample_climate_risk_assessment["overall_status"] == "managed"

    def test_physical_risks_structure(self, sample_climate_risk_assessment):
        risks = sample_climate_risk_assessment["physical_risks"]
        assert "chronic" in risks
        assert "acute" in risks
        assert len(risks["chronic"]) == 2
        assert len(risks["acute"]) == 2

    def test_adaptation_solutions(self, sample_climate_risk_assessment):
        solutions = sample_climate_risk_assessment["adaptation_solutions"]
        assert len(solutions) == 2
        assert all("cost_eur" in s for s in solutions)

    def test_residual_risks_acceptable(self, sample_climate_risk_assessment):
        residual = sample_climate_risk_assessment["residual_risks"]
        for risk, detail in residual.items():
            assert detail["acceptable"] is True


# ===========================================================================
# Minimum safeguard model tests
# ===========================================================================

class TestMinimumSafeguards:
    """Test minimum safeguard model."""

    def test_safeguard_all_pass(self, sample_safeguard_assessment):
        assert sample_safeguard_assessment["overall_pass"] is True
        topics = sample_safeguard_assessment["topics"]
        assert len(topics) == 4
        for topic_name, result in topics.items():
            assert result["overall"] is True

    def test_safeguard_fail(self, failing_safeguard_assessment):
        assert failing_safeguard_assessment["overall_pass"] is False
        topics = failing_safeguard_assessment["topics"]
        assert topics["anti_corruption"]["overall"] is False

    def test_safeguard_evidence(self, sample_safeguard_assessment):
        evidence = sample_safeguard_assessment["evidence_items"]
        assert len(evidence) == 4

    def test_safeguard_topic_details(self, sample_safeguard_topic_results):
        assert len(sample_safeguard_topic_results) == 4
        topics = {r["topic"] for r in sample_safeguard_topic_results}
        assert topics == {"human_rights", "anti_corruption", "taxation", "fair_competition"}


# ===========================================================================
# KPI calculation model tests
# ===========================================================================

class TestKPICalculation:
    """Test KPI calculation model."""

    def test_turnover_kpi(self, sample_kpi_data):
        assert sample_kpi_data["kpi_type"] == "turnover"
        assert sample_kpi_data["kpi_percentage"] == Decimal("42.0000")

    def test_kpi_amounts_consistent(self, sample_kpi_data):
        aligned = sample_kpi_data["aligned_amount"]
        eligible = sample_kpi_data["eligible_amount"]
        total = sample_kpi_data["total_amount"]
        assert aligned <= eligible
        assert eligible <= total

    def test_kpi_percentage_correct(self, sample_kpi_data):
        aligned = float(sample_kpi_data["aligned_amount"])
        total = float(sample_kpi_data["total_amount"])
        expected_pct = round(aligned / total * 100, 4)
        assert abs(float(sample_kpi_data["kpi_percentage"]) - expected_pct) < 0.01

    def test_capex_kpi(self, sample_capex_kpi):
        assert sample_capex_kpi["kpi_type"] == "capex"
        assert sample_capex_kpi["kpi_percentage"] == Decimal("52.5000")

    def test_opex_kpi(self, sample_opex_kpi):
        assert sample_opex_kpi["kpi_type"] == "opex"
        assert sample_opex_kpi["kpi_percentage"] == Decimal("25.5000")

    def test_objective_breakdown(self, sample_kpi_data):
        breakdown = sample_kpi_data["objective_breakdown"]
        assert "climate_mitigation" in breakdown
        assert breakdown["climate_mitigation"]["percentage"] == 38.0


# ===========================================================================
# Activity financials model tests
# ===========================================================================

class TestActivityFinancials:
    """Test activity financial data model."""

    def test_activity_financials_count(self, sample_activity_financials):
        assert len(sample_activity_financials) == 3

    def test_eligible_activity_data(self, sample_activity_financials):
        solar = sample_activity_financials[0]
        assert solar["eligible"] is True
        assert solar["aligned"] is True
        assert solar["objective"] == "climate_mitigation"

    def test_non_eligible_activity(self, sample_activity_financials):
        non_eligible = sample_activity_financials[2]
        assert non_eligible["eligible"] is False
        assert non_eligible["aligned"] is False
        assert non_eligible["objective"] is None


# ===========================================================================
# CapEx plan model tests
# ===========================================================================

class TestCapExPlan:
    """Test CapEx plan model."""

    def test_capex_plan_timeline(self, sample_capex_plan):
        assert sample_capex_plan["start_year"] == 2025
        assert sample_capex_plan["end_year"] == 2030
        assert sample_capex_plan["end_year"] >= sample_capex_plan["start_year"]

    def test_capex_plan_approved(self, sample_capex_plan):
        assert sample_capex_plan["management_approved"] is True
        assert isinstance(sample_capex_plan["approved_date"], date)

    def test_capex_plan_amounts(self, sample_capex_plan):
        planned = sample_capex_plan["planned_amounts"]
        assert len(planned) == 6
        assert all(int(year) >= 2025 for year in planned.keys())


# ===========================================================================
# GAR calculation model tests
# ===========================================================================

class TestGARCalculation:
    """Test GAR calculation model."""

    def test_gar_stock(self, sample_gar_data):
        assert sample_gar_data["gar_type"] == "stock"
        assert sample_gar_data["gar_percentage"] == Decimal("18.7500")

    def test_gar_aligned_le_covered(self, sample_gar_data):
        assert sample_gar_data["aligned_assets"] <= sample_gar_data["covered_assets"]

    def test_gar_flow(self, sample_gar_flow):
        assert sample_gar_flow["gar_type"] == "flow"
        assert sample_gar_flow["gar_percentage"] > sample_gar_flow["gar_percentage"] * 0  # positive

    def test_btar(self, sample_btar_data):
        assert sample_btar_data["metadata"]["ratio_type"] == "btar"

    def test_gar_sector_breakdown(self, sample_gar_data):
        sectors = sample_gar_data["sector_breakdown"]
        assert len(sectors) == 5
        assert "energy" in sectors


# ===========================================================================
# Exposure model tests
# ===========================================================================

class TestExposure:
    """Test financial exposure model."""

    def test_exposure_count(self, sample_exposures):
        assert len(sample_exposures) == 8

    def test_exposure_types(self, sample_exposures):
        types = {e["exposure_type"] for e in sample_exposures}
        assert "corporate_loan" in types
        assert "retail_mortgage" in types
        assert "auto_loan" in types
        assert "project_finance" in types
        assert "green_bond" in types

    def test_mortgage_epc_rating(self, sample_exposures):
        mortgages = [e for e in sample_exposures if e["exposure_type"] == "retail_mortgage"]
        assert len(mortgages) == 3
        epc_a = [m for m in mortgages if m["epc_rating"] == "A"]
        assert len(epc_a) == 1
        assert epc_a[0]["taxonomy_aligned"] is True

    def test_auto_loan_co2(self, sample_exposures):
        auto_loans = [e for e in sample_exposures if e["exposure_type"] == "auto_loan"]
        assert len(auto_loans) == 2
        ev = [a for a in auto_loans if a["co2_gkm"] == Decimal("0.00")]
        assert len(ev) == 1
        assert ev[0]["taxonomy_aligned"] is True

    def test_alignment_pct_range(self, sample_exposures):
        for exp in sample_exposures:
            assert exp["alignment_pct"] >= 0
            assert exp["alignment_pct"] <= 100


# ===========================================================================
# Alignment result model tests
# ===========================================================================

class TestAlignmentResult:
    """Test alignment result model."""

    def test_full_alignment(self, sample_alignment_result):
        assert sample_alignment_result["aligned"] is True
        assert sample_alignment_result["eligible"] is True
        assert sample_alignment_result["sc_pass"] is True
        assert sample_alignment_result["dnsh_pass"] is True
        assert sample_alignment_result["ms_pass"] is True

    def test_aligned_requires_all_steps(self, sample_alignment_result):
        # aligned = eligible AND sc AND dnsh AND ms
        r = sample_alignment_result
        if r["aligned"]:
            assert r["eligible"] and r["sc_pass"] and r["dnsh_pass"] and r["ms_pass"]

    def test_partial_alignment(self, partial_alignment_result):
        assert partial_alignment_result["aligned"] is False
        assert partial_alignment_result["dnsh_pass"] is False


# ===========================================================================
# Portfolio alignment model tests
# ===========================================================================

class TestPortfolioAlignment:
    """Test portfolio alignment model."""

    def test_portfolio_counts(self, sample_portfolio_alignment):
        pa = sample_portfolio_alignment
        assert pa["aligned_count"] <= pa["eligible_count"]
        assert pa["eligible_count"] <= pa["total_activities"]

    def test_portfolio_percentage(self, sample_portfolio_alignment):
        pa = sample_portfolio_alignment
        assert pa["alignment_percentage"] >= 0
        assert pa["alignment_percentage"] <= 100

    def test_kpi_summary(self, sample_portfolio_alignment):
        kpi = sample_portfolio_alignment["kpi_summary"]
        assert "turnover" in kpi
        assert "capex" in kpi
        assert "opex" in kpi


# ===========================================================================
# Report model tests
# ===========================================================================

class TestReport:
    """Test report model."""

    def test_article_8_report(self, sample_report):
        assert sample_report["template"] == "article_8_turnover"
        assert sample_report["format"] == "excel"
        assert sample_report["status"] == "generated"

    def test_eba_report(self, sample_eba_report):
        assert sample_eba_report["template"] == "eba_template_7"
        content = sample_eba_report["content"]
        assert "gar_stock" in content
        assert "gar_flow" in content


# ===========================================================================
# Evidence model tests
# ===========================================================================

class TestEvidence:
    """Test evidence item model."""

    def test_evidence_count(self, sample_evidence):
        assert len(sample_evidence) == 4

    def test_evidence_types(self, sample_evidence):
        types = {e["evidence_type"] for e in sample_evidence}
        assert "certification" in types
        assert "report" in types
        assert "declaration" in types
        assert "data_extract" in types

    def test_verified_evidence(self, sample_evidence):
        verified = [e for e in sample_evidence if e["verified"]]
        assert len(verified) == 3


# ===========================================================================
# Regulatory version model tests
# ===========================================================================

class TestRegulatoryVersion:
    """Test regulatory version model."""

    def test_version_count(self, sample_regulatory_versions):
        assert len(sample_regulatory_versions) == 4

    def test_active_versions(self, sample_regulatory_versions):
        active = [v for v in sample_regulatory_versions if v["status"] == "active"]
        assert len(active) == 3

    def test_superseded_version(self, sample_regulatory_versions):
        superseded = [v for v in sample_regulatory_versions if v["status"] == "superseded"]
        assert len(superseded) == 1
        assert superseded[0]["version_number"] == "1.0"
        assert superseded[0]["delegated_act"] == "climate"


# ===========================================================================
# Data quality model tests
# ===========================================================================

class TestDataQuality:
    """Test data quality score model."""

    def test_data_quality_score(self, sample_data_quality):
        assert sample_data_quality["overall_score"] == Decimal("78.50")
        assert sample_data_quality["grade"] == "B+"

    def test_data_quality_dimensions(self, sample_data_quality):
        dims = sample_data_quality["dimensions"]
        expected = {"completeness", "accuracy", "timeliness", "consistency", "traceability"}
        assert set(dims.keys()) == expected

    def test_improvement_actions(self, sample_data_quality):
        actions = sample_data_quality["improvement_actions"]
        assert len(actions) == 3
        assert all("priority" in a for a in actions)


# ===========================================================================
# Gap assessment model tests
# ===========================================================================

class TestGapAssessment:
    """Test gap assessment model."""

    def test_gap_counts(self, sample_gap_assessment):
        assert sample_gap_assessment["total_gaps"] == 8
        assert sample_gap_assessment["high_priority"] == 3
        assert sample_gap_assessment["high_priority"] <= sample_gap_assessment["total_gaps"]

    def test_gap_categories(self, sample_gap_assessment):
        cats = sample_gap_assessment["gap_categories"]
        assert "sc" in cats
        assert "dnsh" in cats
        assert "data" in cats

    def test_gap_items(self, sample_gap_items):
        assert len(sample_gap_items) == 5
        categories = {item["category"] for item in sample_gap_items}
        assert "sc" in categories
        assert "dnsh" in categories
        assert "data" in categories
        assert "regulatory" in categories
        assert "safeguard" in categories

    def test_gap_item_priorities(self, sample_gap_items):
        priorities = {item["priority"] for item in sample_gap_items}
        assert "critical" in priorities
        assert "high" in priorities

    def test_gap_item_deadlines(self, sample_gap_items):
        for item in sample_gap_items:
            assert isinstance(item["deadline"], date)


# ===========================================================================
# Configuration model tests
# ===========================================================================

class TestConfiguration:
    """Test application configuration."""

    def test_config_objectives(self, sample_config):
        assert len(sample_config["environmental_objectives"]) == 6

    def test_config_kpi_types(self, sample_config):
        assert sample_config["kpi_types"] == ["turnover", "capex", "opex"]

    def test_config_thresholds(self, sample_config):
        assert sample_config["de_minimis_threshold_pct"] == 5.0
        assert sample_config["sc_confidence_threshold"] == 80.0

    def test_config_gar_settings(self, sample_config):
        assert sample_config["gar_stock_enabled"] is True
        assert sample_config["gar_flow_enabled"] is True
        assert sample_config["epc_alignment_threshold"] == "B"
        assert sample_config["auto_loan_co2_threshold_gkm"] == 50.0
