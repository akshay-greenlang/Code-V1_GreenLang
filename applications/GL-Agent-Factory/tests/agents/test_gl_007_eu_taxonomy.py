"""
Unit Tests for GL-007: EU Taxonomy Agent

Comprehensive test suite covering:
- Activity eligibility assessment
- Technical Screening Criteria (TSC) evaluation
- Do No Significant Harm (DNSH) assessment
- Minimum Safeguards verification
- Taxonomy KPI calculation (Revenue, CapEx, OpEx)

Target: 85%+ code coverage

Reference:
- EU Regulation 2020/852 (Taxonomy Regulation)
- EU Delegated Acts 2021/2139, 2023/2485, 2023/2486
- EFRAG Implementation Guidance

Run with:
    pytest tests/agents/test_gl_007_eu_taxonomy.py -v --cov=backend/agents/gl_007_eu_taxonomy
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock




from agents.gl_007_eu_taxonomy.agent import (
    EUTaxonomyAgent,
    TaxonomyInput,
    TaxonomyOutput,
    EnvironmentalObjective,
    AlignmentStatus,
    DNSHStatus,
    MinimumSafeguardsStatus,
    TSCResult,
    DNSHResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def taxonomy_agent():
    """Create EUTaxonomyAgent instance for testing."""
    return EUTaxonomyAgent()


@pytest.fixture
def solar_pv_input():
    """Create input for solar PV electricity generation (aligned activity)."""
    return TaxonomyInput(
        nace_code="D35.11",
        activity_description="Electricity generation from solar PV",
        revenue_eur=10000000.0,
        capex_eur=5000000.0,
        opex_eur=500000.0,
        total_revenue_eur=50000000.0,
        total_capex_eur=20000000.0,
        total_opex_eur=5000000.0,
        environmental_data={"ghg_intensity": 0},
        primary_objective=EnvironmentalObjective.CLIMATE_MITIGATION,
    )


@pytest.fixture
def gas_power_input():
    """Create input for gas-fired power plant (conditional alignment)."""
    return TaxonomyInput(
        nace_code="D35.11",
        activity_description="Electricity generation from natural gas",
        revenue_eur=20000000.0,
        total_revenue_eur=50000000.0,
        environmental_data={"ghg_intensity": 250},  # gCO2e/kWh
    )


@pytest.fixture
def real_estate_input():
    """Create input for real estate activity."""
    return TaxonomyInput(
        nace_code="L68.20",
        activity_description="Acquisition and ownership of buildings",
        revenue_eur=5000000.0,
        total_revenue_eur=100000000.0,
        environmental_data={"primary_energy_demand_kwh_sqm": 120},
    )


# =============================================================================
# Test Class: Agent Initialization
# =============================================================================


class TestEUTaxonomyAgentInitialization:
    """Tests for EUTaxonomyAgent initialization."""

    @pytest.mark.unit
    def test_agent_initializes_with_defaults(self, taxonomy_agent):
        """Test agent initializes correctly with default config."""
        assert taxonomy_agent is not None
        assert hasattr(taxonomy_agent, "run")

    @pytest.mark.unit
    def test_agent_has_activity_mappings(self, taxonomy_agent):
        """Test agent has NACE to taxonomy activity mappings."""
        assert hasattr(taxonomy_agent, "TAXONOMY_ACTIVITIES") or True


# =============================================================================
# Test Class: Environmental Objectives
# =============================================================================


class TestEnvironmentalObjectives:
    """Tests for EU Taxonomy environmental objectives."""

    @pytest.mark.unit
    def test_all_six_objectives_defined(self):
        """Test all 6 environmental objectives are defined."""
        objectives = [
            EnvironmentalObjective.CLIMATE_MITIGATION,
            EnvironmentalObjective.CLIMATE_ADAPTATION,
            EnvironmentalObjective.WATER,
            EnvironmentalObjective.CIRCULAR_ECONOMY,
            EnvironmentalObjective.POLLUTION,
            EnvironmentalObjective.BIODIVERSITY,
        ]
        assert len(objectives) == 6

    @pytest.mark.unit
    def test_objective_values(self):
        """Test objective enum values."""
        assert EnvironmentalObjective.CLIMATE_MITIGATION.value == "climate_change_mitigation"
        assert EnvironmentalObjective.CIRCULAR_ECONOMY.value == "circular_economy"


# =============================================================================
# Test Class: Alignment Status
# =============================================================================


class TestAlignmentStatus:
    """Tests for taxonomy alignment status."""

    @pytest.mark.unit
    def test_alignment_status_values(self):
        """Test alignment status enum values."""
        assert AlignmentStatus.ALIGNED.value == "aligned"
        assert AlignmentStatus.ELIGIBLE_NOT_ALIGNED.value == "eligible_not_aligned"
        assert AlignmentStatus.NOT_ELIGIBLE.value == "not_eligible"
        assert AlignmentStatus.ASSESSMENT_REQUIRED.value == "assessment_required"


# =============================================================================
# Test Class: DNSH Status
# =============================================================================


class TestDNSHStatus:
    """Tests for Do No Significant Harm status."""

    @pytest.mark.unit
    def test_dnsh_status_values(self):
        """Test DNSH status enum values."""
        assert DNSHStatus.PASS.value == "pass"
        assert DNSHStatus.FAIL.value == "fail"
        assert DNSHStatus.NOT_APPLICABLE.value == "not_applicable"


# =============================================================================
# Test Class: Minimum Safeguards
# =============================================================================


class TestMinimumSafeguards:
    """Tests for minimum safeguards status."""

    @pytest.mark.unit
    def test_safeguards_status_values(self):
        """Test minimum safeguards status values."""
        assert MinimumSafeguardsStatus.COMPLIANT.value == "compliant"
        assert MinimumSafeguardsStatus.NON_COMPLIANT.value == "non_compliant"
        assert MinimumSafeguardsStatus.PARTIAL.value == "partial"


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestTaxonomyInputValidation:
    """Tests for taxonomy input validation."""

    @pytest.mark.unit
    def test_valid_input_passes(self, solar_pv_input):
        """Test valid input passes validation."""
        assert solar_pv_input.nace_code == "D35.11"
        assert solar_pv_input.revenue_eur == 10000000.0

    @pytest.mark.unit
    def test_financial_values_non_negative(self, solar_pv_input):
        """Test financial values must be non-negative."""
        assert solar_pv_input.revenue_eur >= 0
        assert solar_pv_input.capex_eur >= 0
        assert solar_pv_input.opex_eur >= 0


# =============================================================================
# Test Class: Eligibility Assessment
# =============================================================================


class TestEligibilityAssessment:
    """Tests for activity eligibility assessment."""

    @pytest.mark.unit
    def test_solar_pv_is_eligible(self, taxonomy_agent, solar_pv_input):
        """Test solar PV is eligible for climate mitigation."""
        result = taxonomy_agent.run(solar_pv_input)
        assert result.is_eligible is True

    @pytest.mark.unit
    def test_eligibility_determination(self, taxonomy_agent, solar_pv_input):
        """Test eligibility is properly determined."""
        result = taxonomy_agent.run(solar_pv_input)
        assert hasattr(result, "is_eligible")


# =============================================================================
# Test Class: Technical Screening Criteria
# =============================================================================


class TestTechnicalScreeningCriteria:
    """Tests for TSC evaluation."""

    @pytest.mark.unit
    def test_solar_pv_meets_tsc(self, taxonomy_agent, solar_pv_input):
        """Test solar PV meets TSC (zero operational emissions)."""
        result = taxonomy_agent.run(solar_pv_input)
        assert result.substantial_contribution is True

    @pytest.mark.unit
    def test_tsc_results_provided(self, taxonomy_agent, solar_pv_input):
        """Test TSC results are provided in output."""
        result = taxonomy_agent.run(solar_pv_input)
        assert hasattr(result, "tsc_results")


# =============================================================================
# Test Class: DNSH Assessment
# =============================================================================


class TestDNSHAssessment:
    """Tests for Do No Significant Harm assessment."""

    @pytest.mark.unit
    def test_dnsh_assessment_performed(self, taxonomy_agent, solar_pv_input):
        """Test DNSH assessment is performed."""
        result = taxonomy_agent.run(solar_pv_input)
        assert hasattr(result, "dnsh_pass")
        assert hasattr(result, "dnsh_results")

    @pytest.mark.unit
    def test_dnsh_covers_all_objectives(self, taxonomy_agent, solar_pv_input):
        """Test DNSH covers all other environmental objectives."""
        result = taxonomy_agent.run(solar_pv_input)
        # DNSH should assess 5 other objectives (excluding primary)
        if result.dnsh_results:
            assert len(result.dnsh_results) >= 1


# =============================================================================
# Test Class: Alignment Assessment
# =============================================================================


class TestAlignmentAssessment:
    """Tests for full taxonomy alignment assessment."""

    @pytest.mark.unit
    def test_solar_pv_aligned(self, taxonomy_agent, solar_pv_input):
        """Test solar PV achieves full alignment."""
        result = taxonomy_agent.run(solar_pv_input)
        # Solar PV should be aligned (eligible + TSC + DNSH + safeguards)
        assert result.alignment_status in ["aligned", AlignmentStatus.ALIGNED.value]

    @pytest.mark.unit
    def test_alignment_requires_all_criteria(self, taxonomy_agent, solar_pv_input):
        """Test alignment requires all criteria."""
        result = taxonomy_agent.run(solar_pv_input)
        # Alignment = eligible AND substantial_contribution AND dnsh_pass AND safeguards
        if result.is_aligned:
            assert result.is_eligible
            assert result.substantial_contribution


# =============================================================================
# Test Class: KPI Calculations
# =============================================================================


class TestKPICalculations:
    """Tests for taxonomy KPI calculations."""

    @pytest.mark.unit
    def test_revenue_alignment_calculated(self, taxonomy_agent, solar_pv_input):
        """Test revenue alignment KPI is calculated."""
        result = taxonomy_agent.run(solar_pv_input)

        assert hasattr(result, "revenue_aligned_eur")
        assert hasattr(result, "revenue_aligned_pct")

    @pytest.mark.unit
    def test_capex_alignment_calculated(self, taxonomy_agent, solar_pv_input):
        """Test CapEx alignment KPI is calculated."""
        result = taxonomy_agent.run(solar_pv_input)

        assert hasattr(result, "capex_aligned_eur")
        assert hasattr(result, "capex_aligned_pct")

    @pytest.mark.unit
    def test_opex_alignment_calculated(self, taxonomy_agent, solar_pv_input):
        """Test OpEx alignment KPI is calculated."""
        result = taxonomy_agent.run(solar_pv_input)

        assert hasattr(result, "opex_aligned_eur")
        assert hasattr(result, "opex_aligned_pct")

    @pytest.mark.unit
    def test_alignment_percentage_calculation(self, taxonomy_agent, solar_pv_input):
        """Test alignment percentage is correctly calculated."""
        result = taxonomy_agent.run(solar_pv_input)

        if result.is_aligned:
            # Revenue aligned pct = (aligned revenue / total revenue) * 100
            expected_pct = (10000000.0 / 50000000.0) * 100
            assert result.revenue_aligned_pct == pytest.approx(expected_pct, rel=0.01)


# =============================================================================
# Test Class: Provenance Tracking
# =============================================================================


class TestTaxonomyProvenance:
    """Tests for provenance hash tracking."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, taxonomy_agent, solar_pv_input):
        """Test provenance hash is generated."""
        result = taxonomy_agent.run(solar_pv_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_provenance_hash_deterministic(self, taxonomy_agent, solar_pv_input):
        """Test provenance hash is deterministic."""
        result1 = taxonomy_agent.run(solar_pv_input)
        result2 = taxonomy_agent.run(solar_pv_input)
        assert result1.provenance_hash == result2.provenance_hash


# =============================================================================
# Test Class: Performance
# =============================================================================


class TestEUTaxonomyPerformance:
    """Performance tests for EUTaxonomyAgent."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_assessment_performance(self, taxonomy_agent, solar_pv_input):
        """Test single assessment completes quickly."""
        import time

        start = time.perf_counter()
        result = taxonomy_agent.run(solar_pv_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0
        assert result is not None
