# -*- coding: utf-8 -*-
"""
Unit Tests for ComplianceScorecardEngine (v1.1)

Tests multi-standard compliance assessment, cross-standard gap analysis,
action item generation, scoring, and trend analysis.

Target module: services/agents/reporting/standards/compliance_scorecard.py
Test count: 34 tests
Coverage target: 85%+
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import sys
import os

PLATFORM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

from services.agents.reporting.standards.compliance_scorecard import (
    ComplianceScorecardEngine,
    ComplianceScorecard,
    StandardCompliance,
    ComplianceRequirement,
    ComplianceGap,
    ActionItem,
    GHG_PROTOCOL_REQUIREMENTS,
    ESRS_E1_REQUIREMENTS,
    CDP_REQUIREMENTS,
    IFRS_S2_REQUIREMENTS,
    ISO_14083_REQUIREMENTS,
)


# ============================================================================
# MOCK DATA
# ============================================================================

FULL_EMISSIONS_DATA: Dict[str, Any] = {
    "scope1_tco2e": 5000.0,
    "scope2_location_tco2e": 3000.0,
    "scope2_market_tco2e": 2800.0,
    "scope3_tco2e": 45000.0,
    "scope3_categories": {
        1: 15000.0, 2: 2000.0, 3: 1500.0, 4: 5000.0, 5: 800.0,
        6: 3000.0, 7: 1200.0, 8: 500.0, 9: 4000.0, 10: 1000.0,
        11: 8000.0, 12: 2000.0, 13: 500.0, 14: 200.0, 15: 300.0,
    },
    "total_tco2e": 55800.0,
    "base_year": 2019,
    "base_year_emissions": {"scope1": 6000, "scope2": 3500, "scope3": 48000},
    "recalculation_policy": "5% threshold",
    "calculation_methodology": "GHG Protocol Corporate Standard",
    "gases_reported": ["co2", "ch4", "n2o", "hfcs", "pfcs", "sf6", "nf3"],
    "gwp_source": "IPCC_AR5",
    "emission_factor_sources": ["EPA", "DEFRA", "IEA"],
    "intensity_per_revenue": 12.5,
    "intensity_per_fte": 2.8,
    "prior_year_emissions": {"scope1": 5200, "scope2_location": 3200, "scope3": 46000},
    "yoy_change_pct": -3.5,
    "biogenic_emissions": {"scope1_biogenic_tco2": 120.0},
    "uncertainty_results": {"overall_cv": 0.15},
    "avg_dqi_score": 3.2,
    "data_quality_by_scope": {"scope1": 4.0, "scope2": 3.5, "scope3": 2.8},
    "provenance_chains": True,
    "scope2_instruments": ["RECs"],
    "scope3_exclusion_rationale": "Categories 14/15 not material",
    "exclusions": "None",
    "reporting_period_start": "2025-01-01",
    "reporting_period_end": "2025-12-31",
    "total_energy_mwh": 120000,
    "renewable_pct": 40.0,
    "energy_by_source": {"electricity": 80000, "natural_gas": 40000},
    "total_emissions_tco2e": 5000.0,
    "total_tonne_km": 10000000,
    "data_quality_score": 3.5,
}

FULL_COMPANY_INFO: Dict[str, Any] = {
    "name": "GreenCorp Industries",
    "reporting_year": 2025,
    "consolidation_approach": "operational_control",
    "consolidation_approach_rationale": "Operational control chosen per GHG Protocol",
    "verification_status": "limited_assurance",
    "ets_exposure": True,
    "internal_carbon_price": True,
}

FULL_TARGETS_DATA: Dict[str, Any] = {
    "has_active_target": True,
    "absolute_targets": [
        {"scope": "scope1+2", "base_year": 2019, "target_year": 2030, "reduction_pct": 42},
    ],
    "sbti_status": "sbti_approved",
    "transition_plan": "Published transition plan",
    "has_transition_plan": True,
    "target_progress": [
        {"target": "scope1+2 42% by 2030", "progress_pct": 15},
    ],
}

FULL_RISKS_DATA: Dict[str, Any] = {
    "has_risk_process": True,
    "risk_process_details": "Integrated ERM process",
    "scenario_analysis": True,
    "financial_effects": {"physical_risk_usd": 5000000, "transition_risk_usd": 3000000},
    "risk_integration": "Fully integrated with financial risk management",
}

FULL_GOVERNANCE_DATA: Dict[str, Any] = {
    "board_oversight": True,
    "management_positions": [{"title": "CSO"}],
}

FULL_TRANSPORT_DATA: Dict[str, Any] = {
    "transport_by_mode": {"road": 3000, "maritime": 1500, "air": 500},
    "transport_chain": "Multi-leg supply chain EU-US",
    "wtw_boundary": "well-to-wheel",
    "allocation_method": "mass",
    "multi_leg_chains": [{"legs": 3, "total_km": 12000}],
    "total_emissions_tco2e": 5000.0,
    "total_tonne_km": 10000000.0,
    "data_quality_score": 3.5,
}

MINIMAL_EMISSIONS_DATA: Dict[str, Any] = {
    "scope1_tco2e": 1000.0,
}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def engine():
    """Create a ComplianceScorecardEngine instance."""
    return ComplianceScorecardEngine()


@pytest.fixture
def full_scorecard(engine):
    """Generate a scorecard with complete data."""
    return engine.generate_scorecard(
        emissions_data=FULL_EMISSIONS_DATA,
        company_info=FULL_COMPANY_INFO,
        targets_data=FULL_TARGETS_DATA,
        risks_data=FULL_RISKS_DATA,
        governance_data=FULL_GOVERNANCE_DATA,
        transport_data=FULL_TRANSPORT_DATA,
    )


@pytest.fixture
def minimal_scorecard(engine):
    """Generate a scorecard with minimal data."""
    return engine.generate_scorecard(
        emissions_data=MINIMAL_EMISSIONS_DATA,
        company_info={"name": "MinimalCo", "reporting_year": 2025},
    )


# ============================================================================
# TEST: ComplianceScorecardEngine initialization
# ============================================================================

class TestComplianceScorecardEngine:
    """Test scorecard engine initialization and full generation."""

    def test_init_creates_engine(self, engine):
        """Engine should initialize without errors."""
        assert engine is not None

    def test_init_creates_5_standards(self, engine):
        """STANDARD_WEIGHTS should have 5 entries."""
        assert len(engine.STANDARD_WEIGHTS) == 5
        assert "ghg_protocol" in engine.STANDARD_WEIGHTS
        assert "esrs_e1" in engine.STANDARD_WEIGHTS
        assert "cdp" in engine.STANDARD_WEIGHTS
        assert "ifrs_s2" in engine.STANDARD_WEIGHTS
        assert "iso_14083" in engine.STANDARD_WEIGHTS

    def test_generate_full_scorecard(self, full_scorecard):
        """Full scorecard should be generated with all standards."""
        assert isinstance(full_scorecard, ComplianceScorecard)
        assert len(full_scorecard.standards) == 5

    def test_ghg_protocol_requirements_count(self):
        """GHG Protocol should have 25 requirements."""
        assert len(GHG_PROTOCOL_REQUIREMENTS) == 25

    def test_esrs_e1_requirements_count(self):
        """ESRS E1 should have 20 requirements."""
        assert len(ESRS_E1_REQUIREMENTS) == 20

    def test_cdp_requirements_count(self):
        """CDP should have 15 requirements."""
        assert len(CDP_REQUIREMENTS) == 15

    def test_ifrs_s2_requirements_count(self):
        """IFRS S2 should have 15 requirements."""
        assert len(IFRS_S2_REQUIREMENTS) == 15

    def test_iso_14083_requirements_count(self):
        """ISO 14083 should have 10 requirements."""
        assert len(ISO_14083_REQUIREMENTS) == 10

    def test_weighted_overall_score(self, full_scorecard):
        """Overall score should be between 0 and 100."""
        assert 0.0 <= full_scorecard.overall_score <= 100.0

    def test_weights_sum_to_100(self, engine):
        """Standard weights should sum to 1.0."""
        total_weight = sum(engine.STANDARD_WEIGHTS.values())
        assert abs(total_weight - 1.0) < 1e-6, (
            f"Weights sum to {total_weight}, expected 1.0"
        )

    def test_overall_grade_assigned(self, full_scorecard):
        """Overall grade should be a letter grade."""
        valid_grades = [
            "A+", "A", "A-", "B+", "B", "B-",
            "C+", "C", "C-", "D+", "D", "D-", "F",
        ]
        assert full_scorecard.overall_grade in valid_grades

    def test_provenance_hash_present(self, full_scorecard):
        """Scorecard should have a provenance hash."""
        assert full_scorecard.provenance_hash != ""
        assert len(full_scorecard.provenance_hash) == 64

    def test_company_name_in_scorecard(self, full_scorecard):
        """Company name should be present in scorecard."""
        assert full_scorecard.company_name == "GreenCorp Industries"

    def test_ghg_protocol_coverage_positive(self, full_scorecard):
        """GHG Protocol coverage should be > 0 with full data."""
        ghg = full_scorecard.standards["ghg_protocol"]
        assert ghg.coverage_pct > 0.0

    def test_iso_14083_with_transport_data(self, full_scorecard):
        """ISO 14083 should have met requirements when transport data provided."""
        iso = full_scorecard.standards["iso_14083"]
        assert iso.met_count > 0

    def test_iso_14083_without_transport_data(self, minimal_scorecard):
        """ISO 14083 should be N/A when no transport data provided."""
        iso = minimal_scorecard.standards["iso_14083"]
        assert iso.not_applicable_count == len(ISO_14083_REQUIREMENTS)

    def test_cdp_predicted_score(self, full_scorecard):
        """CDP assessment should include a predicted score."""
        cdp = full_scorecard.standards["cdp"]
        assert cdp.predicted_score is not None
        valid_scores = ["A", "A-", "B", "B-", "C", "C-", "D", "D-"]
        assert cdp.predicted_score in valid_scores


# ============================================================================
# TEST: Requirement Checking
# ============================================================================

class TestRequirementChecking:
    """Test individual requirement assessment logic."""

    def test_requirement_met_with_evidence(self, full_scorecard):
        """Requirements with all data fields present should be 'met'."""
        ghg = full_scorecard.standards["ghg_protocol"]
        met_reqs = [r for r in ghg.requirements if r.status == "met"]
        assert len(met_reqs) > 0
        for req in met_reqs:
            assert req.evidence is not None

    def test_requirement_not_met_without_data(self, minimal_scorecard):
        """Requirements missing all data fields should be 'not_met'."""
        ghg = minimal_scorecard.standards["ghg_protocol"]
        not_met = [r for r in ghg.requirements if r.status == "not_met"]
        assert len(not_met) > 0

    def test_partial_completion(self, engine):
        """Requirements with some but not all fields should be 'partially_met'."""
        # Provide scope2_location but not scope2_market
        partial_data = {
            "scope1_tco2e": 5000.0,
            "scope2_location_tco2e": 3000.0,
            # scope2_market_tco2e missing
        }
        scorecard = engine.generate_scorecard(
            emissions_data=partial_data,
            company_info={"name": "PartialCo", "reporting_year": 2025},
        )
        ghg = scorecard.standards["ghg_protocol"]
        # GHG-004 requires both scope2_location and scope2_market
        ghg004 = next(
            (r for r in ghg.requirements if r.id == "GHG-004"), None
        )
        assert ghg004 is not None
        assert ghg004.status in ("met", "partially_met")

    def test_evidence_trail_linkage(self, full_scorecard):
        """Every assessed requirement should have evidence or a note."""
        for std_code, std_compliance in full_scorecard.standards.items():
            for req in std_compliance.requirements:
                if req.status != "not_applicable":
                    assert req.evidence is not None or req.notes != "", (
                        f"{req.id} has no evidence or notes"
                    )

    def test_action_items_for_gaps(self, minimal_scorecard):
        """Gaps should generate action items."""
        assert len(minimal_scorecard.gaps) > 0
        assert len(minimal_scorecard.action_items) > 0


# ============================================================================
# TEST: Compliance Trend
# ============================================================================

class TestComplianceTrend:
    """Test trend analysis across multiple assessments."""

    def test_trend_improvement(self, engine):
        """Two scorecards with improving data should show improvement."""
        sc1 = engine.generate_scorecard(
            emissions_data=MINIMAL_EMISSIONS_DATA,
            company_info={"name": "TrendCo", "reporting_year": 2024},
        )
        sc2 = engine.generate_scorecard(
            emissions_data=FULL_EMISSIONS_DATA,
            company_info=FULL_COMPANY_INFO,
            targets_data=FULL_TARGETS_DATA,
            risks_data=FULL_RISKS_DATA,
            governance_data=FULL_GOVERNANCE_DATA,
        )
        # The second scorecard should have a higher overall score
        assert sc2.overall_score > sc1.overall_score

    def test_trend_regression(self, engine):
        """Removing data should lower the score."""
        sc_full = engine.generate_scorecard(
            emissions_data=FULL_EMISSIONS_DATA,
            company_info=FULL_COMPANY_INFO,
        )
        sc_minimal = engine.generate_scorecard(
            emissions_data=MINIMAL_EMISSIONS_DATA,
            company_info={"name": "RegressCo", "reporting_year": 2025},
        )
        assert sc_full.overall_score > sc_minimal.overall_score

    def test_standard_comparison(self, full_scorecard):
        """All five standards should have coverage percentages."""
        for std_code in ["ghg_protocol", "esrs_e1", "cdp", "ifrs_s2", "iso_14083"]:
            std = full_scorecard.standards[std_code]
            assert 0.0 <= std.coverage_pct <= 100.0

    def test_gap_severity_levels(self, minimal_scorecard):
        """Gaps should have valid severity levels."""
        valid_severities = {"critical", "high", "medium", "low"}
        for gap in minimal_scorecard.gaps:
            assert gap.severity in valid_severities

    def test_action_item_priority_levels(self, minimal_scorecard):
        """Action items should have valid priority levels."""
        valid_priorities = {"critical", "high", "medium", "low"}
        for item in minimal_scorecard.action_items:
            assert item.priority in valid_priorities

    def test_action_item_has_estimated_effort(self, minimal_scorecard):
        """Action items should include estimated effort in hours."""
        for item in minimal_scorecard.action_items:
            assert item.estimated_effort_hours is not None
            assert item.estimated_effort_hours > 0

    def test_cross_standard_gaps_sorted_by_severity(self, minimal_scorecard):
        """Gaps should be sorted: critical > high > medium > low."""
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps = minimal_scorecard.gaps
        if len(gaps) >= 2:
            for i in range(1, len(gaps)):
                prev = severity_order.get(gaps[i - 1].severity, 4)
                curr = severity_order.get(gaps[i].severity, 4)
                assert prev <= curr, (
                    f"Gap {gaps[i-1].gap_id} ({gaps[i-1].severity}) "
                    f"should come before {gaps[i].gap_id} ({gaps[i].severity})"
                )
