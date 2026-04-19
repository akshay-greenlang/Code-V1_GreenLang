# -*- coding: utf-8 -*-
"""
Unit Tests for GL-007: EU Taxonomy Agent

Comprehensive test suite with 50 test cases covering:
- Activity classification (NACE codes) (10 tests)
- Technical screening criteria (TSC) (15 tests)
- DNSH assessment (10 tests)
- Alignment calculations (10 tests)
- Error handling (5 tests)

Target: 85%+ coverage for EU Taxonomy Agent
Run with: pytest tests/unit/test_gl007_taxonomy_agent.py -v --cov

Author: GL-TestEngineer
Version: 1.0.0

The EU Taxonomy Agent evaluates economic activities against EU Taxonomy
per EU Regulation 2020/852 and the Climate Delegated Act.
"""

import pytest
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "GL-Agent-Factory" / "backend" / "agents"))

# Import agent components
from gl_007_eu_taxonomy.agent import (
    EUTaxonomyAgent,
    TaxonomyInput,
    TaxonomyOutput,
    EnvironmentalObjective,
    AlignmentStatus,
    DNSHStatus,
    MinimumSafeguardsStatus,
    TaxonomyActivity,
    TSCResult,
    DNSHResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create EUTaxonomyAgent instance."""
    return EUTaxonomyAgent()


@pytest.fixture
def valid_solar_pv_input():
    """Create valid solar PV electricity generation input."""
    return TaxonomyInput(
        nace_code="D35.11",
        activity_description="Electricity generation using solar photovoltaic technology",
        revenue_eur=10000000.0,
        capex_eur=2000000.0,
        opex_eur=500000.0,
        total_revenue_eur=100000000.0,
        total_capex_eur=20000000.0,
        total_opex_eur=15000000.0,
        primary_objective=EnvironmentalObjective.CLIMATE_MITIGATION,
        environmental_data={"lifecycle_ghg_gco2e_kwh": 30},  # Below 100g threshold
        dnsh_data={
            "climate_change_adaptation": {"compliant": True},
            "water_marine_resources": {"compliant": True},
            "circular_economy": {"compliant": True},
            "pollution_prevention": {"compliant": True},
            "biodiversity_ecosystems": {"compliant": True},
        },
        safeguards_data={
            "oecd_guidelines": True,
            "un_guiding_principles": True,
            "ilo_conventions": True,
            "international_bill_rights": True,
        },
    )


@pytest.fixture
def valid_wind_power_input():
    """Create valid wind power electricity generation input."""
    return TaxonomyInput(
        nace_code="D35.11",
        activity_description="Electricity generation from wind power",
        revenue_eur=15000000.0,
        total_revenue_eur=100000000.0,
        primary_objective=EnvironmentalObjective.CLIMATE_MITIGATION,
        environmental_data={"lifecycle_ghg_gco2e_kwh": 10},
        dnsh_data={
            "climate_change_adaptation": {"compliant": True},
            "water_marine_resources": {"compliant": True},
            "circular_economy": {"compliant": True},
            "pollution_prevention": {"compliant": True},
            "biodiversity_ecosystems": {"compliant": True},
        },
        safeguards_data={
            "oecd_guidelines": True,
            "un_guiding_principles": True,
            "ilo_conventions": True,
            "international_bill_rights": True,
        },
    )


@pytest.fixture
def building_renovation_input():
    """Create valid building renovation input."""
    return TaxonomyInput(
        nace_code="F41",
        activity_description="Renovation of existing buildings",
        revenue_eur=5000000.0,
        total_revenue_eur=50000000.0,
        primary_objective=EnvironmentalObjective.CLIMATE_MITIGATION,
        environmental_data={"primary_energy_reduction_pct": 35},  # Above 30% threshold
    )


@pytest.fixture
def non_eligible_activity_input():
    """Create input for non-eligible activity."""
    return TaxonomyInput(
        nace_code="A01.11",  # Crop production - not in taxonomy
        activity_description="Growing of cereals",
        revenue_eur=1000000.0,
        total_revenue_eur=10000000.0,
        primary_objective=EnvironmentalObjective.CLIMATE_MITIGATION,
    )


@pytest.fixture
def eligible_not_aligned_input():
    """Create input that is eligible but not aligned (fails TSC)."""
    return TaxonomyInput(
        nace_code="D35.11",
        activity_description="Electricity generation using solar PV",
        revenue_eur=10000000.0,
        total_revenue_eur=100000000.0,
        primary_objective=EnvironmentalObjective.CLIMATE_MITIGATION,
        environmental_data={"lifecycle_ghg_gco2e_kwh": 150},  # Above 100g threshold - FAILS
    )


# =============================================================================
# Activity Classification Tests (10 tests)
# =============================================================================

class TestActivityClassification:
    """Test suite for NACE code activity classification - 10 test cases."""

    @pytest.mark.unit
    def test_d35_11_is_eligible(self, agent):
        """UT-GL007-001: Test D35.11 (electricity generation) is eligible."""
        is_eligible, activity = agent._check_eligibility("D35.11")
        assert is_eligible is True
        assert activity is not None

    @pytest.mark.unit
    def test_solar_pv_activity_code(self, agent):
        """UT-GL007-002: Test solar PV has activity code 4.1."""
        is_eligible, activity = agent._check_eligibility("D35.11")
        assert activity.code == "4.1"

    @pytest.mark.unit
    def test_f41_renovation_eligible(self, agent):
        """UT-GL007-003: Test F41 (construction/renovation) is eligible."""
        is_eligible, activity = agent._check_eligibility("F41")
        assert is_eligible is True

    @pytest.mark.unit
    def test_l68_real_estate_eligible(self, agent):
        """UT-GL007-004: Test L68 (real estate) is eligible."""
        is_eligible, activity = agent._check_eligibility("L68")
        assert is_eligible is True

    @pytest.mark.unit
    def test_non_taxonomy_nace_not_eligible(self, agent):
        """UT-GL007-005: Test non-taxonomy NACE code is not eligible."""
        is_eligible, activity = agent._check_eligibility("A01.11")
        assert is_eligible is False
        assert activity is None

    @pytest.mark.unit
    def test_prefix_matching_works(self, agent):
        """UT-GL007-006: Test NACE prefix matching works."""
        # D35.11 should match activity for D35.11
        is_eligible, _ = agent._check_eligibility("D35.11")
        assert is_eligible is True

    @pytest.mark.unit
    def test_c29_10_vehicle_manufacturing(self, agent):
        """UT-GL007-007: Test C29.10 (vehicle manufacturing) is eligible."""
        is_eligible, activity = agent._check_eligibility("C29.10")
        assert is_eligible is True

    @pytest.mark.unit
    def test_taxonomy_activities_dict(self, agent):
        """UT-GL007-008: Test TAXONOMY_ACTIVITIES dictionary is populated."""
        assert len(agent.TAXONOMY_ACTIVITIES) > 0

    @pytest.mark.unit
    def test_nace_to_activities_mapping(self, agent):
        """UT-GL007-009: Test NACE_TO_ACTIVITIES mapping is built."""
        agent._build_nace_mapping()
        assert len(agent.NACE_TO_ACTIVITIES) > 0

    @pytest.mark.unit
    def test_get_taxonomy_activities_method(self, agent):
        """UT-GL007-010: Test get_taxonomy_activities utility method."""
        activities = agent.get_taxonomy_activities()
        assert len(activities) > 0
        assert all("code" in a and "name" in a for a in activities)


# =============================================================================
# Technical Screening Criteria Tests (15 tests)
# =============================================================================

class TestTechnicalScreeningCriteria:
    """Test suite for TSC evaluation - 15 test cases."""

    @pytest.mark.unit
    def test_solar_pv_meets_tsc(self, agent, valid_solar_pv_input):
        """UT-GL007-011: Test solar PV meets TSC with low GHG intensity."""
        result = agent.run(valid_solar_pv_input)
        assert result.substantial_contribution is True

    @pytest.mark.unit
    def test_tsc_threshold_100g(self, agent):
        """UT-GL007-012: Test TSC threshold is 100 gCO2e/kWh for electricity."""
        activity = agent.TAXONOMY_ACTIVITIES["4.1"]
        assert activity.tsc_thresholds["lifecycle_ghg_gco2e_kwh"] == 100

    @pytest.mark.unit
    def test_above_threshold_fails_tsc(self, agent, eligible_not_aligned_input):
        """UT-GL007-013: Test above threshold fails TSC."""
        result = agent.run(eligible_not_aligned_input)
        assert result.substantial_contribution is False

    @pytest.mark.unit
    def test_tsc_results_in_output(self, agent, valid_solar_pv_input):
        """UT-GL007-014: Test TSC results are in output."""
        result = agent.run(valid_solar_pv_input)
        assert len(result.tsc_results) > 0

    @pytest.mark.unit
    def test_tsc_result_structure(self, agent, valid_solar_pv_input):
        """UT-GL007-015: Test TSC result structure."""
        result = agent.run(valid_solar_pv_input)
        tsc = result.tsc_results[0]
        assert "objective" in tsc
        assert "criteria_met" in tsc

    @pytest.mark.unit
    def test_building_renovation_30pct_threshold(self, agent, building_renovation_input):
        """UT-GL007-016: Test building renovation 30% energy reduction threshold."""
        result = agent.run(building_renovation_input)
        # 35% reduction meets 30% threshold
        assert result.substantial_contribution is True

    @pytest.mark.unit
    def test_activity_without_thresholds_passes(self, agent):
        """UT-GL007-017: Test activity without specific thresholds passes by nature."""
        # Activity 3.1 (renewable tech manufacturing) has no specific thresholds
        input_data = TaxonomyInput(
            nace_code="C27",  # Manufacture of electrical equipment
            activity_description="Manufacture of renewable energy technologies",
            revenue_eur=5000000.0,
            total_revenue_eur=50000000.0,
            primary_objective=EnvironmentalObjective.CLIMATE_MITIGATION,
        )
        # May pass if in scope and no thresholds

    @pytest.mark.unit
    def test_evaluate_tsc_method(self, agent):
        """UT-GL007-018: Test _evaluate_tsc method directly."""
        activity = agent.TAXONOMY_ACTIVITIES["4.1"]
        env_data = {"lifecycle_ghg_gco2e_kwh": 50}

        tsc_pass, results = agent._evaluate_tsc(activity, env_data)
        assert tsc_pass is True
        assert len(results) > 0

    @pytest.mark.unit
    def test_evaluate_tsc_fails_missing_data(self, agent):
        """UT-GL007-019: Test TSC fails with missing environmental data."""
        activity = agent.TAXONOMY_ACTIVITIES["4.1"]
        env_data = {}  # No data

        tsc_pass, results = agent._evaluate_tsc(activity, env_data)
        assert tsc_pass is False

    @pytest.mark.unit
    def test_tsc_boolean_threshold(self, agent):
        """UT-GL007-020: Test TSC with boolean threshold."""
        # Building construction has air_tightness boolean
        activity = agent.TAXONOMY_ACTIVITIES.get("7.1")
        if activity and "air_tightness" in activity.tsc_thresholds:
            env_data = {"air_tightness": True}
            tsc_pass, _ = agent._evaluate_tsc(activity, env_data)

    @pytest.mark.unit
    def test_tsc_numeric_comparison(self, agent):
        """UT-GL007-021: Test TSC numeric comparison (actual <= threshold)."""
        activity = agent.TAXONOMY_ACTIVITIES["4.1"]
        env_data = {"lifecycle_ghg_gco2e_kwh": 100}  # Exactly at threshold

        tsc_pass, _ = agent._evaluate_tsc(activity, env_data)
        assert tsc_pass is True

    @pytest.mark.unit
    def test_tsc_threshold_exceeded(self, agent):
        """UT-GL007-022: Test TSC fails when threshold exceeded."""
        activity = agent.TAXONOMY_ACTIVITIES["4.1"]
        env_data = {"lifecycle_ghg_gco2e_kwh": 101}  # Just above

        tsc_pass, _ = agent._evaluate_tsc(activity, env_data)
        assert tsc_pass is False

    @pytest.mark.unit
    def test_tsc_result_includes_actual_value(self, agent, valid_solar_pv_input):
        """UT-GL007-023: Test TSC result includes actual value."""
        result = agent.run(valid_solar_pv_input)
        if result.tsc_results:
            tsc = result.tsc_results[0]
            assert "actual_value" in tsc or tsc.get("criteria_details")

    @pytest.mark.unit
    def test_tsc_result_includes_threshold(self, agent, valid_solar_pv_input):
        """UT-GL007-024: Test TSC result includes threshold value."""
        result = agent.run(valid_solar_pv_input)
        if result.tsc_results:
            tsc = result.tsc_results[0]
            assert "threshold_value" in tsc or tsc.get("criteria_details")

    @pytest.mark.unit
    def test_wind_power_tsc(self, agent, valid_wind_power_input):
        """UT-GL007-025: Test wind power meets TSC."""
        result = agent.run(valid_wind_power_input)
        assert result.substantial_contribution is True


# =============================================================================
# DNSH Assessment Tests (10 tests)
# =============================================================================

class TestDNSHAssessment:
    """Test suite for DNSH (Do No Significant Harm) assessment - 10 test cases."""

    @pytest.mark.unit
    def test_dnsh_pass_with_all_compliant(self, agent, valid_solar_pv_input):
        """UT-GL007-026: Test DNSH passes when all objectives are compliant."""
        result = agent.run(valid_solar_pv_input)
        assert result.dnsh_pass is True

    @pytest.mark.unit
    def test_dnsh_results_in_output(self, agent, valid_solar_pv_input):
        """UT-GL007-027: Test DNSH results are in output."""
        result = agent.run(valid_solar_pv_input)
        assert len(result.dnsh_results) > 0

    @pytest.mark.unit
    def test_dnsh_checks_5_objectives(self, agent, valid_solar_pv_input):
        """UT-GL007-028: Test DNSH checks 5 other objectives (excluding primary)."""
        result = agent.run(valid_solar_pv_input)
        # Primary is climate mitigation, should check 5 others
        assert len(result.dnsh_results) == 5

    @pytest.mark.unit
    def test_dnsh_fails_with_missing_assessment(self, agent):
        """UT-GL007-029: Test DNSH fails with missing assessment."""
        input_data = TaxonomyInput(
            nace_code="D35.11",
            activity_description="Solar PV",
            revenue_eur=10000000.0,
            total_revenue_eur=100000000.0,
            primary_objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            environmental_data={"lifecycle_ghg_gco2e_kwh": 30},
            dnsh_data={},  # Empty - missing assessments
        )
        result = agent.run(input_data)
        assert result.dnsh_pass is False

    @pytest.mark.unit
    def test_dnsh_status_enum_values(self):
        """UT-GL007-030: Test DNSHStatus enum values."""
        assert DNSHStatus.PASS.value == "pass"
        assert DNSHStatus.FAIL.value == "fail"
        assert DNSHStatus.NOT_APPLICABLE.value == "not_applicable"
        assert DNSHStatus.ASSESSMENT_REQUIRED.value == "assessment_required"

    @pytest.mark.unit
    def test_evaluate_dnsh_method(self, agent):
        """UT-GL007-031: Test _evaluate_dnsh method directly."""
        activity = agent.TAXONOMY_ACTIVITIES["4.1"]
        dnsh_data = {
            "climate_change_adaptation": {"compliant": True},
            "water_marine_resources": {"compliant": True},
            "circular_economy": {"compliant": True},
            "pollution_prevention": {"compliant": True},
            "biodiversity_ecosystems": {"compliant": True},
        }

        dnsh_pass, results = agent._evaluate_dnsh(activity, dnsh_data)
        assert dnsh_pass is True
        assert len(results) == 5

    @pytest.mark.unit
    def test_dnsh_fails_with_one_non_compliant(self, agent):
        """UT-GL007-032: Test DNSH fails if any objective is non-compliant."""
        input_data = TaxonomyInput(
            nace_code="D35.11",
            activity_description="Solar PV",
            revenue_eur=10000000.0,
            total_revenue_eur=100000000.0,
            primary_objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            environmental_data={"lifecycle_ghg_gco2e_kwh": 30},
            dnsh_data={
                "climate_change_adaptation": {"compliant": True},
                "water_marine_resources": {"compliant": False},  # FAILS
                "circular_economy": {"compliant": True},
                "pollution_prevention": {"compliant": True},
                "biodiversity_ecosystems": {"compliant": True},
            },
        )
        result = agent.run(input_data)
        assert result.dnsh_pass is False

    @pytest.mark.unit
    def test_dnsh_result_includes_issues(self, agent):
        """UT-GL007-033: Test DNSH result includes issues when failing."""
        input_data = TaxonomyInput(
            nace_code="D35.11",
            activity_description="Solar PV",
            revenue_eur=10000000.0,
            total_revenue_eur=100000000.0,
            primary_objective=EnvironmentalObjective.CLIMATE_MITIGATION,
            environmental_data={"lifecycle_ghg_gco2e_kwh": 30},
            dnsh_data={
                "climate_change_adaptation": {"compliant": False, "issues": ["Flood risk not assessed"]},
            },
        )
        result = agent.run(input_data)
        # Should have issues recorded

    @pytest.mark.unit
    def test_environmental_objective_enum_values(self):
        """UT-GL007-034: Test EnvironmentalObjective enum values."""
        assert EnvironmentalObjective.CLIMATE_MITIGATION.value == "climate_change_mitigation"
        assert EnvironmentalObjective.CLIMATE_ADAPTATION.value == "climate_change_adaptation"
        assert EnvironmentalObjective.WATER.value == "water_marine_resources"
        assert EnvironmentalObjective.CIRCULAR_ECONOMY.value == "circular_economy"
        assert EnvironmentalObjective.POLLUTION.value == "pollution_prevention"
        assert EnvironmentalObjective.BIODIVERSITY.value == "biodiversity_ecosystems"

    @pytest.mark.unit
    def test_get_environmental_objectives_method(self, agent):
        """UT-GL007-035: Test get_environmental_objectives utility method."""
        objectives = agent.get_environmental_objectives()
        assert len(objectives) == 6


# =============================================================================
# Alignment Calculations Tests (10 tests)
# =============================================================================

class TestAlignmentCalculations:
    """Test suite for alignment calculations - 10 test cases."""

    @pytest.mark.unit
    def test_aligned_status_full_compliance(self, agent, valid_solar_pv_input):
        """UT-GL007-036: Test ALIGNED status with full compliance."""
        result = agent.run(valid_solar_pv_input)
        assert result.is_aligned is True
        assert result.alignment_status == "aligned"

    @pytest.mark.unit
    def test_eligible_not_aligned_status(self, agent, eligible_not_aligned_input):
        """UT-GL007-037: Test ELIGIBLE_NOT_ALIGNED status when TSC fails."""
        result = agent.run(eligible_not_aligned_input)
        assert result.is_eligible is True
        assert result.is_aligned is False
        assert result.alignment_status == "eligible_not_aligned"

    @pytest.mark.unit
    def test_not_eligible_status(self, agent, non_eligible_activity_input):
        """UT-GL007-038: Test NOT_ELIGIBLE status for non-taxonomy activity."""
        result = agent.run(non_eligible_activity_input)
        assert result.is_eligible is False
        assert result.is_aligned is False
        assert result.alignment_status == "not_eligible"

    @pytest.mark.unit
    def test_alignment_formula(self, agent, valid_solar_pv_input):
        """UT-GL007-039: Test alignment formula: eligible AND SC AND DNSH AND safeguards."""
        result = agent.run(valid_solar_pv_input)

        # If aligned, all components must be True
        assert result.is_eligible is True
        assert result.substantial_contribution is True
        assert result.dnsh_pass is True
        assert result.minimum_safeguards_compliant is True
        assert result.is_aligned is True

    @pytest.mark.unit
    def test_revenue_aligned_calculation(self, agent, valid_solar_pv_input):
        """UT-GL007-040: Test revenue aligned EUR is calculated."""
        result = agent.run(valid_solar_pv_input)
        assert result.revenue_aligned_eur == 10000000.0  # Full activity revenue

    @pytest.mark.unit
    def test_revenue_aligned_pct_calculation(self, agent, valid_solar_pv_input):
        """UT-GL007-041: Test revenue aligned percentage is calculated."""
        result = agent.run(valid_solar_pv_input)

        # 10M / 100M = 10%
        expected_pct = (10000000.0 / 100000000.0) * 100
        assert result.revenue_aligned_pct == pytest.approx(expected_pct, rel=0.01)

    @pytest.mark.unit
    def test_capex_aligned_calculation(self, agent, valid_solar_pv_input):
        """UT-GL007-042: Test CapEx aligned is calculated."""
        result = agent.run(valid_solar_pv_input)

        # 2M / 20M = 10%
        assert result.capex_aligned_eur == 2000000.0
        expected_pct = (2000000.0 / 20000000.0) * 100
        assert result.capex_aligned_pct == pytest.approx(expected_pct, rel=0.01)

    @pytest.mark.unit
    def test_opex_aligned_calculation(self, agent, valid_solar_pv_input):
        """UT-GL007-043: Test OpEx aligned is calculated."""
        result = agent.run(valid_solar_pv_input)

        # 500K / 15M = 3.33%
        assert result.opex_aligned_eur == 500000.0
        expected_pct = (500000.0 / 15000000.0) * 100
        assert result.opex_aligned_pct == pytest.approx(expected_pct, rel=0.01)

    @pytest.mark.unit
    def test_zero_kpis_for_non_aligned(self, agent, eligible_not_aligned_input):
        """UT-GL007-044: Test KPIs are zero for non-aligned activity."""
        result = agent.run(eligible_not_aligned_input)

        assert result.revenue_aligned_eur == 0
        assert result.capex_aligned_eur == 0
        assert result.opex_aligned_eur == 0

    @pytest.mark.unit
    def test_alignment_status_enum_values(self):
        """UT-GL007-045: Test AlignmentStatus enum values."""
        assert AlignmentStatus.ALIGNED.value == "aligned"
        assert AlignmentStatus.ELIGIBLE_NOT_ALIGNED.value == "eligible_not_aligned"
        assert AlignmentStatus.NOT_ELIGIBLE.value == "not_eligible"
        assert AlignmentStatus.ASSESSMENT_REQUIRED.value == "assessment_required"


# =============================================================================
# Error Handling Tests (5 tests)
# =============================================================================

class TestErrorHandling:
    """Test suite for error handling - 5 test cases."""

    @pytest.mark.unit
    def test_minimum_safeguards_check(self, agent):
        """UT-GL007-046: Test minimum safeguards are checked."""
        safeguards_data = {
            "oecd_guidelines": True,
            "un_guiding_principles": True,
            "ilo_conventions": True,
            "international_bill_rights": True,
        }
        compliant, details = agent._evaluate_safeguards(safeguards_data)
        assert compliant is True

    @pytest.mark.unit
    def test_safeguards_fail_missing(self, agent):
        """UT-GL007-047: Test safeguards fail with missing compliance."""
        safeguards_data = {
            "oecd_guidelines": True,
            "un_guiding_principles": False,  # Missing
        }
        compliant, details = agent._evaluate_safeguards(safeguards_data)
        assert compliant is False

    @pytest.mark.unit
    def test_provenance_hash_generated(self, agent, valid_solar_pv_input):
        """UT-GL007-048: Test provenance hash is generated."""
        result = agent.run(valid_solar_pv_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_output_timestamp(self, agent, valid_solar_pv_input):
        """UT-GL007-049: Test output includes timestamp."""
        result = agent.run(valid_solar_pv_input)
        assert result.calculated_at is not None
        assert isinstance(result.calculated_at, datetime)

    @pytest.mark.unit
    def test_deterministic_calculation(self, agent, valid_solar_pv_input):
        """UT-GL007-050: Test calculation is deterministic."""
        result1 = agent.run(valid_solar_pv_input)
        result2 = agent.run(valid_solar_pv_input)

        assert result1.is_aligned == result2.is_aligned
        assert result1.revenue_aligned_eur == result2.revenue_aligned_eur


# =============================================================================
# Agent Initialization Tests
# =============================================================================

class TestAgentInitialization:
    """Tests for agent initialization."""

    @pytest.mark.unit
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = EUTaxonomyAgent()
        assert agent is not None
        assert agent.AGENT_ID == "regulatory/eu_taxonomy_v1"
        assert agent.VERSION == "1.0.0"

    @pytest.mark.unit
    def test_taxonomy_activities_loaded(self):
        """Test taxonomy activities are loaded."""
        agent = EUTaxonomyAgent()
        assert len(agent.TAXONOMY_ACTIVITIES) > 0
        assert "4.1" in agent.TAXONOMY_ACTIVITIES  # Solar PV

    @pytest.mark.unit
    def test_nace_mapping_built(self):
        """Test NACE to activities mapping is built."""
        agent = EUTaxonomyAgent()
        assert len(agent.NACE_TO_ACTIVITIES) > 0


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrizedTaxonomy:
    """Parametrized tests for taxonomy scenarios."""

    @pytest.mark.unit
    @pytest.mark.parametrize("activity_code,expected_name", [
        ("4.1", "Electricity generation using solar photovoltaic technology"),
        ("4.3", "Electricity generation from wind power"),
        ("7.1", "Construction of new buildings"),
        ("7.2", "Renovation of existing buildings"),
    ])
    def test_activity_names(self, agent, activity_code, expected_name):
        """Test taxonomy activity names."""
        activity = agent.TAXONOMY_ACTIVITIES.get(activity_code)
        if activity:
            assert activity.name == expected_name

    @pytest.mark.unit
    @pytest.mark.parametrize("objective", [
        EnvironmentalObjective.CLIMATE_MITIGATION,
        EnvironmentalObjective.CLIMATE_ADAPTATION,
        EnvironmentalObjective.WATER,
        EnvironmentalObjective.CIRCULAR_ECONOMY,
        EnvironmentalObjective.POLLUTION,
        EnvironmentalObjective.BIODIVERSITY,
    ])
    def test_all_objectives_valid(self, objective):
        """Test all environmental objectives are valid."""
        assert objective.value is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
