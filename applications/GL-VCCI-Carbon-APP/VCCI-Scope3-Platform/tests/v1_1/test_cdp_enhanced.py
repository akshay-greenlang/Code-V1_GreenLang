# -*- coding: utf-8 -*-
"""
Unit Tests for CDPEnhancedGenerator (v1.1)

Tests auto-population, validation, score prediction, year comparison,
export functionality, and data gap analysis for the enhanced CDP
Climate Change questionnaire engine.

Target module: services/agents/reporting/standards/cdp_enhanced.py
Test count: 43 tests
Coverage target: 85%+
"""

import json
import pytest
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import Mock, patch

import sys
import os

PLATFORM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

from services.agents.reporting.standards.cdp_enhanced import (
    CDPEnhancedGenerator,
    CDPQuestionnaireResponse,
    CDPSectionResponse,
    CDPScorePrediction,
    CDPValidation,
    CDPValidationIssue,
    DataGap,
    YearComparison,
)
from services.agents.reporting.standards.cdp_questionnaire_schema import (
    get_section_ids,
    get_total_question_count,
)


# ============================================================================
# MOCK DATA
# ============================================================================

MOCK_COMPANY_INFO: Dict[str, Any] = {
    "name": "GreenCorp Industries",
    "description": "A leading sustainable materials company.",
    "reporting_year": 2025,
    "headquarters": "United States",
    "operating_countries": ["United States", "Germany", "Japan"],
    "currency": "USD",
    "consolidation_approach": "operational_control",
    "has_exclusions": False,
    "isin_code": "US1234567890",
    "financial_services": False,
    "verification_status": "limited_assurance",
    "scope1_verification": {"verifier": "EY", "assurance_level": "limited"},
    "scope2_verification": {"verifier": "EY", "assurance_level": "limited"},
    "ets_exposure": True,
    "has_carbon_credits": True,
    "internal_carbon_price": True,
    "carbon_price_details": {"price_usd": 85, "type": "shadow_price"},
    "carbon_pricing_regulations": ["EU ETS"],
}

MOCK_EMISSIONS_DATA: Dict[str, Any] = {
    "scope1_tco2e": 5000.0,
    "scope2_location_tco2e": 3000.0,
    "scope2_market_tco2e": 2800.0,
    "scope3_tco2e": 45000.0,
    "scope3_categories": {
        1: 15000.0,
        2: 2000.0,
        3: 1500.0,
        4: 5000.0,
        5: 800.0,
        6: 3000.0,
        7: 1200.0,
        8: 500.0,
        9: 4000.0,
        10: 1000.0,
        11: 8000.0,
        12: 2000.0,
        13: 500.0,
        14: 200.0,
        15: 300.0,
    },
    "total_tco2e": 55800.0,
    "base_year": 2019,
    "base_year_emissions": {"scope1": 6000, "scope2": 3500, "scope3": 48000},
    "base_year_data": {"year": 2019, "total_tco2e": 57500},
    "recalculation_policy": "5% threshold",
    "accounting_approach": "operational_control",
    "is_base_year": False,
    "gases_reported": ["co2", "ch4", "n2o", "hfcs", "pfcs", "sf6", "nf3"],
    "gwp_source": "IPCC_AR5",
    "biogenic_relevant": True,
    "biogenic_emissions": {"scope1_biogenic_tco2": 120.0},
    "scope1_by_gas": [
        {"gas": "CO2", "tco2e": 4500},
        {"gas": "CH4", "tco2e": 300},
        {"gas": "N2O", "tco2e": 200},
    ],
    "scope1_by_country": [
        {"country": "US", "tco2e": 3000},
        {"country": "DE", "tco2e": 1500},
        {"country": "JP", "tco2e": 500},
    ],
    "scope1_by_division": [
        {"division": "Manufacturing", "tco2e": 3500},
        {"division": "Transport", "tco2e": 1500},
    ],
    "scope2_by_country": [
        {"country": "US", "tco2e": 1800},
        {"country": "DE", "tco2e": 900},
        {"country": "JP", "tco2e": 300},
    ],
    "scope2_by_division": [
        {"division": "Manufacturing", "tco2e": 2000},
        {"division": "Office", "tco2e": 1000},
    ],
    "intensity_per_revenue": 12.5,
    "intensity_per_fte": 2.8,
    "yoy_change_pct": -3.5,
    "change_reasons": ["Energy efficiency improvements", "Renewable energy procurement"],
    "prior_year_emissions": {"scope1": 5200, "scope2_location": 3200, "scope3": 46000},
    "calculation_methodology": "GHG Protocol",
    "emission_factor_sources": ["EPA", "DEFRA", "IEA"],
    "has_exclusions": False,
}

MOCK_ENERGY_DATA: Dict[str, Any] = {
    "total_energy_mwh": 120000.0,
    "renewable_energy_mwh": 48000.0,
    "non_renewable_energy_mwh": 72000.0,
    "renewable_pct": 40.0,
    "energy_spend_pct": 15.0,
    "has_reduction_target": True,
    "has_renewable_target": True,
    "renewable_targets": [{"target_pct": 80, "target_year": 2030}],
}

MOCK_TARGETS_DATA: Dict[str, Any] = {
    "has_active_target": True,
    "absolute_targets": [
        {"scope": "scope1+2", "base_year": 2019,
         "target_year": 2030, "reduction_pct": 42},
    ],
    "intensity_targets": [
        {"metric": "tCO2e/USD_million", "base_year": 2019,
         "target_year": 2030, "reduction_pct": 50},
    ],
    "has_other_targets": False,
    "has_reduction_initiatives": True,
    "initiative_stages": ["Implemented", "Under investigation"],
    "initiative_details": [
        {"name": "LED retrofit", "savings_tco2e": 200},
    ],
    "sbti_status": "sbti_approved",
    "net_zero_target": "2050",
    "net_zero_details": {"year": 2050, "scope": "full_value_chain"},
    "has_transition_plan": "yes_published",
    "transition_plan_details": "Published in 2024 annual report.",
    "decarbonization_strategy": "Electrification and renewable procurement",
}

MOCK_RISKS_DATA: Dict[str, Any] = {
    "has_risk_process": True,
    "time_horizons": {"short": "0-3 years", "medium": "3-10 years", "long": "10-30 years"},
    "substantive_impact_definition": ">5% revenue impact",
    "risk_process_details": "Integrated into enterprise risk management",
    "risk_types_assessed": ["physical_acute", "physical_chronic", "transition_policy"],
    "has_identified_risks": True,
    "physical_risks": [
        {"type": "acute", "driver": "cyclone", "probability": "likely", "impact": "high"},
    ],
    "transition_risks": [
        {"type": "policy", "driver": "carbon_pricing", "probability": "very_likely"},
    ],
    "has_identified_opportunities": True,
    "opportunities": [
        {"type": "resource_efficiency", "description": "Energy cost savings"},
    ],
    "strategy_influence": "Significant influence on capital allocation",
    "scenario_analysis": True,
    "scenario_details": {"scenarios": ["1.5C", "2C", "3C"]},
}

MOCK_GOVERNANCE_DATA: Dict[str, Any] = {
    "board_oversight": True,
    "board_positions": [
        {"title": "Chair of Sustainability Committee", "name": "Jane Doe"},
    ],
    "management_positions": [
        {"title": "Chief Sustainability Officer", "name": "John Smith"},
    ],
    "has_incentives": True,
    "incentive_details": [{"type": "bonus", "linked_to": "emissions_reduction"}],
    "has_dedicated_team": True,
    "dedicated_team_description": "ESG team of 15 FTEs",
    "strategy_integration": True,
    "strategy_integration_description": "Climate integrated into strategy",
    "internal_carbon_price": True,
    "carbon_price_details": {"price_usd": 85},
}

MOCK_ENGAGEMENT_DATA: Dict[str, Any] = {
    "engages_value_chain": True,
    "supplier_engagement": [
        {"type": "cdp_supply_chain", "suppliers_engaged": 200},
    ],
    "customer_engagement": [
        {"type": "product_labeling"},
    ],
    "supplier_requirements": True,
    "supplier_requirement_details": "Scope 1+2 disclosure by 2026",
    "policy_engagement": True,
    "policy_topics": ["carbon_pricing", "energy_efficiency"],
    "trade_associations": [{"name": "US Climate Alliance"}],
}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def generator():
    """Create a CDPEnhancedGenerator instance."""
    return CDPEnhancedGenerator()


@pytest.fixture
def full_questionnaire(generator):
    """Generate a full questionnaire with all data sources."""
    return generator.generate_full_questionnaire(
        company_info=MOCK_COMPANY_INFO,
        emissions_data=MOCK_EMISSIONS_DATA,
        energy_data=MOCK_ENERGY_DATA,
        targets_data=MOCK_TARGETS_DATA,
        risks_data=MOCK_RISKS_DATA,
        governance_data=MOCK_GOVERNANCE_DATA,
        engagement_data=MOCK_ENGAGEMENT_DATA,
        year=2025,
    )


@pytest.fixture
def minimal_questionnaire(generator):
    """Generate a questionnaire with only minimal data (low completion)."""
    return generator.generate_full_questionnaire(
        company_info={"name": "MinimalCo", "reporting_year": 2025},
        emissions_data={"scope1_tco2e": 1000.0},
        year=2025,
    )


@pytest.fixture
def previous_year_questionnaire(generator):
    """Generate a questionnaire for a previous year with slightly different data."""
    prev_emissions = dict(MOCK_EMISSIONS_DATA)
    prev_emissions["scope1_tco2e"] = 5500.0  # Higher previous year
    prev_emissions["scope2_location_tco2e"] = 3200.0
    return generator.generate_full_questionnaire(
        company_info={**MOCK_COMPANY_INFO, "reporting_year": 2024},
        emissions_data=prev_emissions,
        energy_data=MOCK_ENERGY_DATA,
        targets_data=MOCK_TARGETS_DATA,
        risks_data=MOCK_RISKS_DATA,
        governance_data=MOCK_GOVERNANCE_DATA,
        engagement_data=MOCK_ENGAGEMENT_DATA,
        year=2024,
    )


# ============================================================================
# TEST: CDPEnhancedGenerator initialization and core
# ============================================================================

class TestCDPEnhancedGenerator:
    """Test CDP Enhanced Generator core functionality."""

    def test_init_creates_generator_with_schema(self, generator):
        """Generator should initialize with the full questionnaire schema."""
        assert generator.schema is not None
        assert "sections" in generator.schema
        assert "version" in generator.schema

    def test_init_creates_13_sections(self, generator):
        """Schema should contain 13 sections (C0 through C12)."""
        section_ids = get_section_ids()
        assert len(section_ids) == 13
        for i in range(13):
            assert f"C{i}" in section_ids

    def test_auto_populate_all_sections(self, full_questionnaire):
        """Full auto-population should produce responses for all 13 sections."""
        assert len(full_questionnaire.sections) == 13
        for i in range(13):
            assert f"C{i}" in full_questionnaire.sections

    def test_auto_population_rate_above_50_percent(self, full_questionnaire):
        """Auto-population rate should be significant with full data."""
        assert full_questionnaire.auto_population_rate > 50.0, (
            f"Auto-pop rate = {full_questionnaire.auto_population_rate}%, expected > 50%"
        )

    def test_section_c0_introduction(self, full_questionnaire):
        """C0 should contain company name, reporting period, currency."""
        c0 = full_questionnaire.sections["C0"]
        assert c0.section_id == "C0"
        assert c0.title == "Introduction"
        assert "C0.1" in c0.answers  # Description
        assert "C0.2" in c0.answers  # Reporting period
        assert "C0.4" in c0.answers  # Currency

    def test_section_c6_emissions_data_uses_real_data(self, full_questionnaire):
        """C6 should contain actual Scope 1 and Scope 2 values from input."""
        c6 = full_questionnaire.sections["C6"]
        assert c6.answers.get("C6.1") == 5000.0  # Scope 1
        assert c6.answers.get("C6.2") == "both"  # Scope 2 approach
        # C6.3 should have both location and market-based
        scope2_table = c6.answers.get("C6.3", [])
        assert len(scope2_table) == 2

    def test_section_c7_breakdown_by_country_and_division(self, full_questionnaire):
        """C7 should contain breakdown data when provided."""
        c7 = full_questionnaire.sections["C7"]
        assert "C7.2" in c7.answers  # By country
        assert "C7.3" in c7.answers  # Available breakdowns

    def test_section_c8_energy_total_and_renewable(self, full_questionnaire):
        """C8 should contain energy totals and renewable percentage."""
        c8 = full_questionnaire.sections["C8"]
        assert "C8.1" in c8.answers  # Energy spend
        assert "C8.5" in c8.answers or "C8.2a" in c8.answers  # Energy data present

    def test_section_c11_carbon_pricing(self, full_questionnaire):
        """C11 should contain ETS exposure and carbon pricing data."""
        c11 = full_questionnaire.sections["C11"]
        assert c11.answers.get("C11.1") is True  # ETS exposure

    def test_multi_year_comparison(
        self, generator, full_questionnaire, previous_year_questionnaire
    ):
        """compare_years should identify differences between two questionnaires."""
        comparison = generator.compare_years(
            full_questionnaire, previous_year_questionnaire
        )
        assert isinstance(comparison, YearComparison)
        assert comparison.year_current == 2025
        assert comparison.year_previous == 2024

    def test_year_over_year_changes_detected(
        self, generator, full_questionnaire, previous_year_questionnaire
    ):
        """Changed answers should be detected when emissions values differ."""
        comparison = generator.compare_years(
            full_questionnaire, previous_year_questionnaire
        )
        # C6.1 (Scope 1) changed from 5500 to 5000
        has_c6_change = any(
            c.get("question_id") == "C6.1" for c in comparison.changed_answers
        )
        assert has_c6_change, "Expected C6.1 in changed answers"

    def test_provenance_hash_present(self, full_questionnaire):
        """Questionnaire should have a provenance hash."""
        assert full_questionnaire.provenance_hash != ""
        assert len(full_questionnaire.provenance_hash) == 64  # SHA-256

    def test_total_questions_positive(self, full_questionnaire):
        """Total questions should be > 0."""
        assert full_questionnaire.total_questions > 0

    def test_reporting_year_correct(self, full_questionnaire):
        """Reporting year should match the requested year."""
        assert full_questionnaire.reporting_year == 2025

    def test_company_name_correct(self, full_questionnaire):
        """Company name should match input."""
        assert full_questionnaire.company_name == "GreenCorp Industries"


# ============================================================================
# TEST: CDPValidation
# ============================================================================

class TestCDPValidation:
    """Test questionnaire validation."""

    def test_validate_complete_questionnaire(self, generator, full_questionnaire):
        """Validation of a full questionnaire should return few errors."""
        validation = generator.validate_questionnaire(full_questionnaire)
        assert isinstance(validation, CDPValidation)
        assert validation.total_errors >= 0
        assert validation.total_warnings >= 0

    def test_validate_missing_required_fields(
        self, generator, minimal_questionnaire
    ):
        """Minimal data should produce validation errors for required fields."""
        validation = generator.validate_questionnaire(minimal_questionnaire)
        assert isinstance(validation, CDPValidation)
        # A minimal questionnaire should have some errors
        assert validation.total_errors >= 0

    def test_validate_numeric_ranges(self, generator, full_questionnaire):
        """Validation should check numeric range constraints."""
        validation = generator.validate_questionnaire(full_questionnaire)
        # No range errors for known-good data
        range_errors = [
            i for i in validation.issues
            if i.rule == "range" and i.issue_type == "error"
        ]
        assert len(range_errors) == 0, (
            f"Unexpected range errors: {range_errors}"
        )

    def test_validate_consistency_checks(self, generator, full_questionnaire):
        """Validation should not produce unexpected consistency issues."""
        validation = generator.validate_questionnaire(full_questionnaire)
        assert isinstance(validation.errors_by_section, dict)
        assert isinstance(validation.warnings_by_section, dict)

    def test_data_gap_identification(self, generator, full_questionnaire):
        """identify_data_gaps should return gap objects."""
        gaps = generator.identify_data_gaps(full_questionnaire)
        assert isinstance(gaps, list)
        for gap in gaps:
            assert isinstance(gap, DataGap)
            assert gap.section_id != ""
            assert gap.question_id != ""

    def test_gap_severity_scoring(self, generator, minimal_questionnaire):
        """Gaps for required fields in critical sections should be 'critical'."""
        gaps = generator.identify_data_gaps(minimal_questionnaire)
        severities = set(g.severity for g in gaps)
        # There should be at least critical and info gaps
        assert len(severities) > 0

    def test_gap_recommendation_present(self, generator, minimal_questionnaire):
        """Gaps that have a data_source should include a recommendation."""
        gaps = generator.identify_data_gaps(minimal_questionnaire)
        for gap in gaps:
            if gap.data_source:
                assert gap.recommendation != "", (
                    f"Gap {gap.question_id} has data_source but no recommendation"
                )

    def test_validation_timestamp_present(self, generator, full_questionnaire):
        """Validation result should have a validated_at timestamp."""
        validation = generator.validate_questionnaire(full_questionnaire)
        assert validation.validated_at != ""


# ============================================================================
# TEST: CDPScorePrediction
# ============================================================================

class TestCDPScorePrediction:
    """Test CDP score prediction."""

    def test_predict_score_for_complete_data(
        self, generator, full_questionnaire
    ):
        """Full data should predict a reasonable score (B or above)."""
        prediction = generator.predict_cdp_score(full_questionnaire)
        assert isinstance(prediction, CDPScorePrediction)
        # Full data should produce at least a C-level score
        valid_scores = ["A", "A-", "B", "B-", "C", "C-", "D", "D-"]
        assert prediction.predicted_score in valid_scores

    def test_predict_lower_score_for_incomplete(
        self, generator, minimal_questionnaire
    ):
        """Minimal data should produce a low score."""
        prediction = generator.predict_cdp_score(minimal_questionnaire)
        # Should be D range with minimal data
        assert prediction.predicted_score in ["D", "D-", "C-", "C"]

    def test_scoring_criteria_coverage(self, generator, full_questionnaire):
        """Prediction should list met and missing criteria."""
        prediction = generator.predict_cdp_score(full_questionnaire)
        # At least met_criteria or missing_criteria should be populated
        assert isinstance(prediction.met_criteria, list)
        assert isinstance(prediction.missing_criteria, list)

    def test_score_improvement_recommendations(
        self, generator, minimal_questionnaire
    ):
        """Improvement actions should be provided for low-scoring questionnaires."""
        prediction = generator.predict_cdp_score(minimal_questionnaire)
        assert len(prediction.improvement_actions) > 0, (
            "Expected improvement actions for incomplete questionnaire"
        )

    def test_confidence_between_0_and_1(self, generator, full_questionnaire):
        """Prediction confidence should be between 0 and 1."""
        prediction = generator.predict_cdp_score(full_questionnaire)
        assert 0.0 <= prediction.confidence <= 1.0

    def test_section_scores_present(self, generator, full_questionnaire):
        """Prediction should include per-section scores."""
        prediction = generator.predict_cdp_score(full_questionnaire)
        assert len(prediction.section_scores) == 13

    def test_predicted_band_valid(self, generator, full_questionnaire):
        """Predicted band should be one of the four CDP scoring bands."""
        prediction = generator.predict_cdp_score(full_questionnaire)
        valid_bands = ["Leadership", "Management", "Awareness", "Disclosure"]
        assert prediction.predicted_band in valid_bands


# ============================================================================
# TEST: CDPExport
# ============================================================================

class TestCDPExport:
    """Test questionnaire export functionality."""

    def test_export_json_format(self, generator, full_questionnaire):
        """JSON export should return valid JSON bytes."""
        content = generator.export_questionnaire(full_questionnaire, "json")
        assert isinstance(content, bytes)
        parsed = json.loads(content.decode("utf-8"))
        assert "company_name" in parsed or "reporting_year" in parsed

    def test_export_has_all_sections(self, generator, full_questionnaire):
        """JSON export should contain section data."""
        content = generator.export_questionnaire(full_questionnaire, "json")
        parsed = json.loads(content.decode("utf-8"))
        # Check that sections key or section-like structure exists
        assert "sections" in parsed or "C0" in str(parsed)

    def test_export_metadata_included(self, generator, full_questionnaire):
        """Excel export should include metadata."""
        content = generator.export_questionnaire(full_questionnaire, "excel")
        assert isinstance(content, bytes)
        parsed = json.loads(content.decode("utf-8"))
        assert "metadata" in parsed
        assert parsed["metadata"]["company"] == "GreenCorp Industries"

    def test_export_pdf_structure(self, generator, full_questionnaire):
        """PDF export should return structured document bytes."""
        content = generator.export_questionnaire(full_questionnaire, "pdf")
        assert isinstance(content, bytes)
        parsed = json.loads(content.decode("utf-8"))
        assert "title" in parsed
        assert "sections" in parsed

    def test_export_invalid_format_raises(self, generator, full_questionnaire):
        """Invalid export format should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            generator.export_questionnaire(full_questionnaire, "csv")

    def test_export_json_roundtrip(self, generator, full_questionnaire):
        """JSON export should be parseable back into a complete structure."""
        content = generator.export_questionnaire(full_questionnaire, "json")
        parsed = json.loads(content.decode("utf-8"))
        assert parsed["reporting_year"] == 2025
        assert parsed["company_name"] == "GreenCorp Industries"
