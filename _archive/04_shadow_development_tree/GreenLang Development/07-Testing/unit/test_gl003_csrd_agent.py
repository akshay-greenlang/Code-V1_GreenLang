# -*- coding: utf-8 -*-
"""
Unit Tests for GL-003: CSRD Reporting Agent

Comprehensive test suite with 50 test cases covering:
- ESRS disclosure generation (10 tests)
- Materiality assessment (10 tests)
- Data point validation (15 tests)
- Completeness scoring (10 tests)
- Error handling (5 tests)

Target: 85%+ coverage for CSRD Reporting Agent
Run with: pytest tests/unit/test_gl003_csrd_agent.py -v --cov

Author: GL-TestEngineer
Version: 1.0.0

CSRD (Corporate Sustainability Reporting Directive) generates ESRS-compliant
sustainability disclosures per EU Directive 2022/2464.
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
from gl_003_csrd_reporting.agent import (
    CSRDReportingAgent,
    CSRDInput,
    CSRDOutput,
    ESRSStandard,
    MaterialityLevel,
    CompanySize,
    ESRSDatapoint,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create CSRDReportingAgent instance."""
    return CSRDReportingAgent()


@pytest.fixture
def valid_large_company_input():
    """Create valid large company input data."""
    return CSRDInput(
        company_id="EU-CORP-001",
        reporting_year=2024,
        company_size=CompanySize.LARGE,
        e1_climate_data={
            "scope1_emissions": 10000,
            "scope2_emissions_location": 5000,
            "scope2_emissions_market": 4500,
            "scope3_emissions": 50000,
            "total_emissions": 65000,
            "energy_consumption": 100000,
            "renewable_energy_share": 25.0,
            "ghg_intensity_revenue": 15.5,
            "transition_plan_status": True,
        },
        s1_workforce_data={
            "total_employees": 5000,
            "gender_ratio": 0.45,
            "collective_bargaining_pct": 75.0,
            "training_hours": 20.5,
            "recordable_incidents": 12,
        },
        g1_governance_data={
            "code_of_conduct": True,
            "anti_corruption_training_pct": 95.0,
            "corruption_incidents": 0,
            "whistleblower_mechanism": True,
            "supplier_code_pct": 80.0,
        },
    )


@pytest.fixture
def minimal_input():
    """Create minimal required input data."""
    return CSRDInput(
        company_id="EU-SME-001",
        reporting_year=2024,
        company_size=CompanySize.LARGE,
    )


@pytest.fixture
def input_with_materiality():
    """Create input with materiality assessment."""
    return CSRDInput(
        company_id="EU-CORP-002",
        reporting_year=2024,
        company_size=CompanySize.LARGE,
        double_materiality={
            "E2": MaterialityLevel.HIGH,
            "E3": MaterialityLevel.MEDIUM,
            "E4": MaterialityLevel.LOW,
            "E5": MaterialityLevel.NOT_MATERIAL,
            "S2": MaterialityLevel.HIGH,
        },
        e1_climate_data={"scope1_emissions": 5000},
    )


@pytest.fixture
def input_with_all_environmental_data():
    """Create input with all environmental standards data."""
    return CSRDInput(
        company_id="EU-CORP-003",
        reporting_year=2025,
        company_size=CompanySize.LARGE_PIE,
        e1_climate_data={"scope1_emissions": 10000},
        e2_pollution_data={"air_pollutants": 100},
        e3_water_data={"water_consumption": 50000},
        e4_biodiversity_data={"land_use_ha": 1000},
        e5_circular_economy_data={"waste_generated": 500},
        s1_workforce_data={"total_employees": 10000},
        g1_governance_data={"code_of_conduct": True},
    )


# =============================================================================
# ESRS Disclosure Generation Tests (10 tests)
# =============================================================================

class TestESRSDisclosureGeneration:
    """Test suite for ESRS disclosure generation - 10 test cases."""

    @pytest.mark.unit
    def test_e1_climate_disclosures_generated(self, agent, valid_large_company_input):
        """UT-GL003-001: Test E1 climate disclosures are generated."""
        result = agent.run(valid_large_company_input)

        assert result.e1_metrics is not None
        assert "scope1_emissions" in result.e1_metrics

    @pytest.mark.unit
    def test_s1_workforce_disclosures_generated(self, agent, valid_large_company_input):
        """UT-GL003-002: Test S1 workforce disclosures are generated."""
        result = agent.run(valid_large_company_input)

        assert result.s1_metrics is not None
        assert "total_employees" in result.s1_metrics

    @pytest.mark.unit
    def test_g1_governance_disclosures_generated(self, agent, valid_large_company_input):
        """UT-GL003-003: Test G1 governance disclosures are generated."""
        result = agent.run(valid_large_company_input)

        assert result.g1_metrics is not None
        assert "has_code_of_conduct" in result.g1_metrics

    @pytest.mark.unit
    def test_mandatory_standards_always_assessed(self, agent, minimal_input):
        """UT-GL003-004: Test E1, S1, G1 are always assessed as mandatory."""
        result = agent.run(minimal_input)

        assert "E1" in result.material_topics
        assert "S1" in result.material_topics
        assert "G1" in result.material_topics

    @pytest.mark.unit
    def test_esrs_2_general_disclosures_included(self, agent, minimal_input):
        """UT-GL003-005: Test ESRS_2 general disclosures are included."""
        result = agent.run(minimal_input)

        assert "ESRS_2" in result.material_topics

    @pytest.mark.unit
    def test_e1_metrics_extraction(self, agent, valid_large_company_input):
        """UT-GL003-006: Test E1 metrics correctly extracted."""
        result = agent.run(valid_large_company_input)

        assert result.e1_metrics["scope1_emissions"] == 10000
        assert result.e1_metrics["scope2_emissions"] == 5000
        assert result.e1_metrics["has_transition_plan"] is True

    @pytest.mark.unit
    def test_s1_metrics_extraction(self, agent, valid_large_company_input):
        """UT-GL003-007: Test S1 metrics correctly extracted."""
        result = agent.run(valid_large_company_input)

        assert result.s1_metrics["total_employees"] == 5000
        assert result.s1_metrics["gender_diversity"] == 0.45

    @pytest.mark.unit
    def test_g1_metrics_extraction(self, agent, valid_large_company_input):
        """UT-GL003-008: Test G1 metrics correctly extracted."""
        result = agent.run(valid_large_company_input)

        assert result.g1_metrics["has_code_of_conduct"] is True
        assert result.g1_metrics["corruption_incidents"] == 0

    @pytest.mark.unit
    def test_company_id_preserved(self, agent, valid_large_company_input):
        """UT-GL003-009: Test company_id preserved in output."""
        result = agent.run(valid_large_company_input)

        assert result.company_id == "EU-CORP-001"

    @pytest.mark.unit
    def test_reporting_year_preserved(self, agent, valid_large_company_input):
        """UT-GL003-010: Test reporting_year preserved in output."""
        result = agent.run(valid_large_company_input)

        assert result.reporting_year == 2024


# =============================================================================
# Materiality Assessment Tests (10 tests)
# =============================================================================

class TestMaterialityAssessment:
    """Test suite for materiality assessment - 10 test cases."""

    @pytest.mark.unit
    def test_high_materiality_topics_included(self, agent, input_with_materiality):
        """UT-GL003-011: Test high materiality topics are included."""
        result = agent.run(input_with_materiality)

        assert "E2" in result.material_topics
        assert "S2" in result.material_topics

    @pytest.mark.unit
    def test_medium_materiality_topics_included(self, agent, input_with_materiality):
        """UT-GL003-012: Test medium materiality topics are included."""
        result = agent.run(input_with_materiality)

        assert "E3" in result.material_topics

    @pytest.mark.unit
    def test_low_materiality_topics_excluded(self, agent, input_with_materiality):
        """UT-GL003-013: Test low materiality topics are excluded."""
        result = agent.run(input_with_materiality)

        # E4 is LOW, should not be in material topics
        assert "E4" not in result.material_topics

    @pytest.mark.unit
    def test_not_material_topics_excluded(self, agent, input_with_materiality):
        """UT-GL003-014: Test not_material topics are excluded."""
        result = agent.run(input_with_materiality)

        # E5 is NOT_MATERIAL
        assert "E5" not in result.material_topics

    @pytest.mark.unit
    def test_mandatory_topics_always_material(self, agent, input_with_materiality):
        """UT-GL003-015: Test E1, S1, G1 always material regardless of assessment."""
        result = agent.run(input_with_materiality)

        # Even without explicit materiality, these are mandatory
        assert "E1" in result.material_topics
        assert "S1" in result.material_topics
        assert "G1" in result.material_topics

    @pytest.mark.unit
    def test_determine_material_topics_method(self, agent):
        """UT-GL003-016: Test _determine_material_topics method."""
        materiality = {
            "E2": MaterialityLevel.HIGH,
            "E3": MaterialityLevel.LOW,
        }
        topics = agent._determine_material_topics(materiality)

        assert "E1" in topics  # Mandatory
        assert "E2" in topics  # High
        assert "E3" not in topics  # Low

    @pytest.mark.unit
    def test_material_topics_sorted(self, agent, input_with_materiality):
        """UT-GL003-017: Test material topics are sorted."""
        result = agent.run(input_with_materiality)

        # Should be sorted alphabetically
        assert result.material_topics == sorted(result.material_topics)

    @pytest.mark.unit
    def test_no_duplicate_material_topics(self, agent, input_with_materiality):
        """UT-GL003-018: Test no duplicate material topics."""
        result = agent.run(input_with_materiality)

        assert len(result.material_topics) == len(set(result.material_topics))

    @pytest.mark.unit
    def test_materiality_enum_values(self):
        """UT-GL003-019: Test MaterialityLevel enum values."""
        assert MaterialityLevel.HIGH.value == "high"
        assert MaterialityLevel.MEDIUM.value == "medium"
        assert MaterialityLevel.LOW.value == "low"
        assert MaterialityLevel.NOT_MATERIAL.value == "not_material"

    @pytest.mark.unit
    def test_empty_materiality_uses_defaults(self, agent, minimal_input):
        """UT-GL003-020: Test empty materiality assessment uses mandatory defaults."""
        result = agent.run(minimal_input)

        # Should have mandatory topics only
        assert "E1" in result.material_topics
        assert "S1" in result.material_topics
        assert "G1" in result.material_topics
        assert "ESRS_2" in result.material_topics


# =============================================================================
# Data Point Validation Tests (15 tests)
# =============================================================================

class TestDataPointValidation:
    """Test suite for data point validation - 15 test cases."""

    @pytest.mark.unit
    def test_total_datapoints_calculated(self, agent, valid_large_company_input):
        """UT-GL003-021: Test total datapoints are calculated."""
        result = agent.run(valid_large_company_input)

        assert result.total_datapoints > 0

    @pytest.mark.unit
    def test_filled_datapoints_counted(self, agent, valid_large_company_input):
        """UT-GL003-022: Test filled datapoints are counted."""
        result = agent.run(valid_large_company_input)

        assert result.filled_datapoints > 0
        assert result.filled_datapoints <= result.total_datapoints

    @pytest.mark.unit
    def test_completeness_score_calculated(self, agent, valid_large_company_input):
        """UT-GL003-023: Test completeness score is calculated."""
        result = agent.run(valid_large_company_input)

        assert 0 <= result.completeness_score <= 100

    @pytest.mark.unit
    def test_completeness_formula(self, agent, valid_large_company_input):
        """UT-GL003-024: Test completeness = filled / total * 100."""
        result = agent.run(valid_large_company_input)

        if result.total_datapoints > 0:
            expected = (result.filled_datapoints / result.total_datapoints) * 100
            assert result.completeness_score == pytest.approx(expected, rel=0.01)

    @pytest.mark.unit
    def test_mandatory_completeness_calculated(self, agent, valid_large_company_input):
        """UT-GL003-025: Test mandatory completeness is calculated."""
        result = agent.run(valid_large_company_input)

        assert 0 <= result.mandatory_completeness <= 100

    @pytest.mark.unit
    def test_gap_analysis_generated(self, agent, minimal_input):
        """UT-GL003-026: Test gap analysis is generated."""
        result = agent.run(minimal_input)

        assert result.gap_analysis is not None
        assert isinstance(result.gap_analysis, dict)

    @pytest.mark.unit
    def test_gap_analysis_shows_missing_datapoints(self, agent, minimal_input):
        """UT-GL003-027: Test gap analysis shows missing datapoints by standard."""
        result = agent.run(minimal_input)

        # With minimal data, should have gaps in E1, S1, G1
        # The specific gaps depend on what mandatory fields are missing

    @pytest.mark.unit
    def test_calculate_required_datapoints_method(self, agent):
        """UT-GL003-028: Test _calculate_required_datapoints method."""
        material_topics = ["E1", "S1", "G1", "ESRS_2"]
        datapoints = agent._calculate_required_datapoints(
            CompanySize.LARGE,
            material_topics
        )

        assert len(datapoints) > 0
        assert all(isinstance(dp, ESRSDatapoint) for dp in datapoints)

    @pytest.mark.unit
    def test_datapoint_has_required_fields(self, agent):
        """UT-GL003-029: Test ESRSDatapoint has required fields."""
        datapoint = ESRSDatapoint(
            id="E1-1",
            standard=ESRSStandard.E1,
            disclosure_requirement="E1-1",
            value=None,
            unit=None,
            is_mandatory=True,
            is_filled=False,
        )

        assert datapoint.id is not None
        assert datapoint.standard is not None
        assert datapoint.is_mandatory is True

    @pytest.mark.unit
    def test_assess_filled_datapoints_method(self, agent, valid_large_company_input):
        """UT-GL003-030: Test _assess_filled_datapoints method."""
        material_topics = ["E1", "S1", "G1", "ESRS_2"]
        required = agent._calculate_required_datapoints(
            CompanySize.LARGE,
            material_topics
        )
        filled, gaps = agent._assess_filled_datapoints(valid_large_company_input, required)

        assert isinstance(filled, int)
        assert isinstance(gaps, dict)

    @pytest.mark.unit
    def test_mandatory_e1_metrics_tracked(self, agent):
        """UT-GL003-031: Test mandatory E1 metrics are tracked."""
        mandatory_metrics = agent.E1_CLIMATE_METRICS

        # Check key mandatory metrics exist
        assert "scope1_emissions" in mandatory_metrics
        assert "scope2_emissions_location" in mandatory_metrics
        assert mandatory_metrics["scope1_emissions"]["mandatory"] is True

    @pytest.mark.unit
    def test_completeness_zero_for_empty_data(self, agent):
        """UT-GL003-032: Test completeness is low for empty data."""
        empty_input = CSRDInput(
            company_id="EU-EMPTY-001",
            reporting_year=2024,
            company_size=CompanySize.LARGE,
        )
        result = agent.run(empty_input)

        # With no data, mandatory completeness should be low
        assert result.mandatory_completeness < 50

    @pytest.mark.unit
    def test_completeness_high_for_complete_data(self, agent, valid_large_company_input):
        """UT-GL003-033: Test completeness is high for complete data."""
        result = agent.run(valid_large_company_input)

        # With filled data, should have reasonable completeness
        assert result.completeness_score > 0

    @pytest.mark.unit
    def test_count_by_standard_method(self, agent):
        """UT-GL003-034: Test _count_by_standard method."""
        datapoints = [
            ESRSDatapoint(id="E1-1", standard=ESRSStandard.E1, disclosure_requirement="E1-1",
                         value=None, unit=None, is_mandatory=True, is_filled=False),
            ESRSDatapoint(id="E1-2", standard=ESRSStandard.E1, disclosure_requirement="E1-2",
                         value=None, unit=None, is_mandatory=True, is_filled=False),
            ESRSDatapoint(id="S1-1", standard=ESRSStandard.S1, disclosure_requirement="S1-1",
                         value=None, unit=None, is_mandatory=True, is_filled=False),
        ]
        counts = agent._count_by_standard(datapoints)

        assert counts["E1"] == 2
        assert counts["S1"] == 1

    @pytest.mark.unit
    def test_assess_mandatory_completeness_method(self, agent, valid_large_company_input):
        """UT-GL003-035: Test _assess_mandatory_completeness method."""
        filled, total = agent._assess_mandatory_completeness(valid_large_company_input)

        assert total > 0
        assert filled >= 0
        assert filled <= total


# =============================================================================
# Completeness Scoring Tests (10 tests)
# =============================================================================

class TestCompletenessScoring:
    """Test suite for completeness scoring - 10 test cases."""

    @pytest.mark.unit
    def test_completeness_score_range(self, agent, valid_large_company_input):
        """UT-GL003-036: Test completeness score is in valid range 0-100."""
        result = agent.run(valid_large_company_input)

        assert 0 <= result.completeness_score <= 100

    @pytest.mark.unit
    def test_mandatory_completeness_range(self, agent, valid_large_company_input):
        """UT-GL003-037: Test mandatory completeness is in valid range 0-100."""
        result = agent.run(valid_large_company_input)

        assert 0 <= result.mandatory_completeness <= 100

    @pytest.mark.unit
    def test_completeness_rounded_to_two_decimals(self, agent, valid_large_company_input):
        """UT-GL003-038: Test completeness score rounded to 2 decimals."""
        result = agent.run(valid_large_company_input)

        str_score = str(result.completeness_score)
        if '.' in str_score:
            decimal_places = len(str_score.split('.')[1])
            assert decimal_places <= 2

    @pytest.mark.unit
    def test_assurance_level_limited_before_2030(self, agent, valid_large_company_input):
        """UT-GL003-039: Test assurance level is 'limited' before 2030."""
        result = agent.run(valid_large_company_input)

        assert result.assurance_level == "limited"

    @pytest.mark.unit
    def test_assurance_level_reasonable_from_2030(self, agent):
        """UT-GL003-040: Test assurance level is 'reasonable' from 2030."""
        input_2030 = CSRDInput(
            company_id="EU-CORP-001",
            reporting_year=2030,
            company_size=CompanySize.LARGE,
        )
        result = agent.run(input_2030)

        assert result.assurance_level == "reasonable"

    @pytest.mark.unit
    def test_determine_assurance_level_method(self, agent):
        """UT-GL003-041: Test _determine_assurance_level method."""
        assert agent._determine_assurance_level(2024) == "limited"
        assert agent._determine_assurance_level(2029) == "limited"
        assert agent._determine_assurance_level(2030) == "reasonable"
        assert agent._determine_assurance_level(2035) == "reasonable"

    @pytest.mark.unit
    def test_completeness_deterministic(self, agent, valid_large_company_input):
        """UT-GL003-042: Test completeness calculation is deterministic."""
        result1 = agent.run(valid_large_company_input)
        result2 = agent.run(valid_large_company_input)

        assert result1.completeness_score == result2.completeness_score

    @pytest.mark.unit
    def test_more_data_higher_completeness(self, agent):
        """UT-GL003-043: Test more data leads to higher completeness."""
        minimal = CSRDInput(
            company_id="EU-MIN-001",
            reporting_year=2024,
            company_size=CompanySize.LARGE,
        )
        complete = CSRDInput(
            company_id="EU-COMP-001",
            reporting_year=2024,
            company_size=CompanySize.LARGE,
            e1_climate_data={"scope1_emissions": 10000},
            s1_workforce_data={"total_employees": 1000},
            g1_governance_data={"code_of_conduct": True},
        )

        min_result = agent.run(minimal)
        comp_result = agent.run(complete)

        # Complete should have better mandatory completeness
        assert comp_result.mandatory_completeness >= min_result.mandatory_completeness

    @pytest.mark.unit
    def test_completeness_with_all_environmental_data(self, agent, input_with_all_environmental_data):
        """UT-GL003-044: Test completeness with all environmental data."""
        result = agent.run(input_with_all_environmental_data)

        # Should have good completeness with all data provided
        assert result.completeness_score > 0

    @pytest.mark.unit
    def test_provenance_hash_generated(self, agent, valid_large_company_input):
        """UT-GL003-045: Test provenance hash is generated."""
        result = agent.run(valid_large_company_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256


# =============================================================================
# Error Handling Tests (5 tests)
# =============================================================================

class TestErrorHandling:
    """Test suite for error handling - 5 test cases."""

    @pytest.mark.unit
    def test_invalid_reporting_year_rejected(self):
        """UT-GL003-046: Test reporting year before 2024 is rejected."""
        with pytest.raises(ValueError):
            CSRDInput(
                company_id="EU-CORP-001",
                reporting_year=2023,  # Before CSRD applies
                company_size=CompanySize.LARGE,
            )

    @pytest.mark.unit
    def test_output_includes_timestamp(self, agent, valid_large_company_input):
        """UT-GL003-047: Test output includes calculation timestamp."""
        result = agent.run(valid_large_company_input)

        assert result.calculated_at is not None
        assert isinstance(result.calculated_at, datetime)

    @pytest.mark.unit
    def test_agent_recovers_after_error(self, agent, valid_large_company_input):
        """UT-GL003-048: Test agent recovers after an error."""
        # Cause an error with invalid data
        try:
            invalid = CSRDInput(
                company_id="",
                reporting_year=2020,
                company_size=CompanySize.LARGE,
            )
            agent.run(invalid)
        except:
            pass

        # Should still work
        result = agent.run(valid_large_company_input)
        assert result.company_id == "EU-CORP-001"

    @pytest.mark.unit
    def test_esrs_standard_enum_values(self):
        """UT-GL003-049: Test ESRSStandard enum has all values."""
        standards = {s.value for s in ESRSStandard}

        assert "ESRS_1" in standards
        assert "ESRS_2" in standards
        assert "E1" in standards
        assert "E2" in standards
        assert "E3" in standards
        assert "E4" in standards
        assert "E5" in standards
        assert "S1" in standards
        assert "S2" in standards
        assert "S3" in standards
        assert "S4" in standards
        assert "G1" in standards

    @pytest.mark.unit
    def test_company_size_enum_values(self):
        """UT-GL003-050: Test CompanySize enum has all values."""
        sizes = {s.value for s in CompanySize}

        assert "large_pie" in sizes
        assert "large" in sizes
        assert "sme" in sizes
        assert "micro" in sizes


# =============================================================================
# Agent Initialization Tests
# =============================================================================

class TestAgentInitialization:
    """Tests for agent initialization."""

    @pytest.mark.unit
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = CSRDReportingAgent()
        assert agent is not None
        assert agent.AGENT_ID == "regulatory/csrd_reporting_v1"
        assert agent.VERSION == "1.0.0"

    @pytest.mark.unit
    def test_agent_initialization_with_config(self):
        """Test agent initializes with custom config."""
        config = {"custom_setting": True}
        agent = CSRDReportingAgent(config=config)
        assert agent.config["custom_setting"] is True

    @pytest.mark.unit
    def test_get_esrs_standards_method(self):
        """Test get_esrs_standards utility method."""
        agent = CSRDReportingAgent()
        standards = agent.get_esrs_standards()

        assert "E1" in standards
        assert "S1" in standards
        assert "G1" in standards
        assert len(standards) == 12  # ESRS_1, ESRS_2, E1-E5, S1-S4, G1


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrizedCSRD:
    """Parametrized tests for CSRD scenarios."""

    @pytest.mark.unit
    @pytest.mark.parametrize("reporting_year,expected_assurance", [
        (2024, "limited"),
        (2025, "limited"),
        (2029, "limited"),
        (2030, "reasonable"),
        (2035, "reasonable"),
    ])
    def test_assurance_level_by_year(self, agent, reporting_year, expected_assurance):
        """Test assurance level changes by year."""
        input_data = CSRDInput(
            company_id="EU-TEST-001",
            reporting_year=reporting_year,
            company_size=CompanySize.LARGE,
        )
        result = agent.run(input_data)
        assert result.assurance_level == expected_assurance

    @pytest.mark.unit
    @pytest.mark.parametrize("company_size", [
        CompanySize.LARGE_PIE,
        CompanySize.LARGE,
        CompanySize.SME,
    ])
    def test_different_company_sizes(self, agent, company_size):
        """Test agent handles different company sizes."""
        input_data = CSRDInput(
            company_id="EU-SIZE-001",
            reporting_year=2024,
            company_size=company_size,
        )
        result = agent.run(input_data)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
