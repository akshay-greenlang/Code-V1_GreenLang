"""
Unit Tests for GL-012: Carbon Offset Verification Agent

Comprehensive test coverage (35 golden tests) for the Carbon Offset Agent including:
- Basic agent initialization and configuration
- Registry validation (Verra VCS, Gold Standard, ACR, CAR, Plan Vivo, Puro.earth)
- ICVCM Core Carbon Principles quality scoring (5 dimensions)
- Credit verification checks
- Article 6.4 compliance assessment
- Portfolio analysis (removal vs avoidance, vintage distribution, diversification)
- Price benchmarking
- Risk assessment
- Edge cases and validation

Test coverage target: 85%+
"""

import hashlib
import json
import pytest
from datetime import datetime
from typing import Dict, List, Any

from .agent import (
    CarbonOffsetAgent,
    CarbonOffsetInput,
    CarbonOffsetOutput,
    CarbonCredit,
    ProjectDetails,
    CarbonRegistry,
    ProjectType,
    VerificationStatus,
    RiskLevel,
    RetirementStatus,
    CorrespondingAdjustmentStatus,
    Article6AuthorizationStatus,
    CreditCategory,
    PriceTier,
    ICVCMScoreBreakdown,
    VerificationCheck,
    CreditVerificationResult,
    RiskAssessment,
    RegistryCreditVerification,
    Article6Compliance,
    PortfolioAnalysis,
    PriceBenchmark,
    PortfolioPriceSummary,
    REGISTRY_STANDARDS,
    PROJECT_TYPE_PROFILES,
    PRICE_BENCHMARKS,
    REMOVAL_PROJECT_TYPES,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent() -> CarbonOffsetAgent:
    """Create a CarbonOffsetAgent instance for testing."""
    return CarbonOffsetAgent()


@pytest.fixture
def sample_credit() -> CarbonCredit:
    """Create a sample carbon credit."""
    return CarbonCredit(
        serial_number="VCS-1234-2023-001",
        vintage_year=2023,
        quantity_tco2e=100.0,
        retirement_status=RetirementStatus.ACTIVE,
        corresponding_adjustment=CorrespondingAdjustmentStatus.APPLIED,
        article6_authorization=Article6AuthorizationStatus.AUTHORIZED,
        unit_price_usd=15.0,
    )


@pytest.fixture
def sample_credits() -> List[CarbonCredit]:
    """Create sample carbon credits for testing."""
    return [
        CarbonCredit(
            serial_number="VCS-1234-2023-001",
            vintage_year=2023,
            quantity_tco2e=100.0,
            retirement_status=RetirementStatus.ACTIVE,
            unit_price_usd=15.0,
        ),
        CarbonCredit(
            serial_number="VCS-1234-2022-002",
            vintage_year=2022,
            quantity_tco2e=200.0,
            retirement_status=RetirementStatus.ACTIVE,
            unit_price_usd=12.0,
        ),
        CarbonCredit(
            serial_number="VCS-1234-2021-003",
            vintage_year=2021,
            quantity_tco2e=150.0,
            retirement_status=RetirementStatus.ACTIVE,
            unit_price_usd=10.0,
        ),
    ]


@pytest.fixture
def sample_project_details() -> ProjectDetails:
    """Create sample project details."""
    return ProjectDetails(
        project_id="VCS-1234",
        project_name="Amazon Rainforest Protection",
        project_type=ProjectType.REDD_PLUS,
        registry=CarbonRegistry.VERRA_VCS,
        country="BR",
        region="Amazonas",
        start_date="2020-01-01",
        crediting_period_start="2020-01-01",
        crediting_period_end="2040-12-31",
        methodology="VM0007",
        methodology_version="1.6",
        verification_body="SCS Global Services",
        last_verification_date="2023-06-15",
        total_credits_issued=1500000.0,
        buffer_pool_contribution=20.0,
        sdg_contributions=[13, 15, 1, 8],
    )


@pytest.fixture
def basic_input(sample_credit: CarbonCredit) -> CarbonOffsetInput:
    """Create basic input for simple tests."""
    return CarbonOffsetInput(
        project_id="VCS-1234",
        registry=CarbonRegistry.VERRA_VCS,
        credits=[sample_credit],
    )


@pytest.fixture
def comprehensive_input(
    sample_credits: List[CarbonCredit],
    sample_project_details: ProjectDetails,
) -> CarbonOffsetInput:
    """Create comprehensive input for full tests."""
    return CarbonOffsetInput(
        project_id="VCS-1234",
        registry=CarbonRegistry.VERRA_VCS,
        credits=sample_credits,
        project_details=sample_project_details,
        verification_purpose="due_diligence",
        require_corresponding_adjustment=False,
        vintage_cutoff_years=5,
    )


@pytest.fixture
def removal_project_input() -> CarbonOffsetInput:
    """Create input for carbon removal project."""
    return CarbonOffsetInput(
        project_id="DAC-001",
        registry=CarbonRegistry.PURO_EARTH,
        credits=[
            CarbonCredit(
                serial_number="PURO-DAC-2024-001",
                vintage_year=2024,
                quantity_tco2e=50.0,
                retirement_status=RetirementStatus.ACTIVE,
                unit_price_usd=500.0,
            )
        ],
        project_details=ProjectDetails(
            project_id="DAC-001",
            project_name="Direct Air Capture Facility",
            project_type=ProjectType.DIRECT_AIR_CAPTURE,
            registry=CarbonRegistry.PURO_EARTH,
            country="IS",
            verification_body="DNV GL",
            sdg_contributions=[13, 9, 7],
        ),
    )


# =============================================================================
# Test 1-5: Basic Agent Tests
# =============================================================================


class TestAgentInitialization:
    """Tests for agent initialization and configuration."""

    def test_1_agent_initialization(self, agent: CarbonOffsetAgent):
        """Test 1: Agent initializes correctly with default config."""
        assert agent is not None
        assert agent.AGENT_ID == "offsets/carbon_verification_v1"
        assert agent.VERSION == "1.0.0"
        assert agent.registry_standards is not None
        assert agent.project_profiles is not None

    def test_2_agent_with_custom_config(self):
        """Test 2: Agent initialization with custom config."""
        config = {"custom_setting": "value", "debug": True}
        agent = CarbonOffsetAgent(config=config)
        assert agent.config == config
        assert agent.config["custom_setting"] == "value"

    def test_3_icvcm_weights_loaded(self, agent: CarbonOffsetAgent):
        """Test 3: ICVCM weights are correctly configured."""
        weights = agent.ICVCM_WEIGHTS
        assert weights["additionality"] == 0.30
        assert weights["permanence"] == 0.25
        assert weights["mrv"] == 0.20
        assert weights["cobenefits"] == 0.15
        assert weights["governance"] == 0.10
        # Weights should sum to 1.0
        assert sum(weights.values()) == 1.0

    def test_4_quality_thresholds_defined(self, agent: CarbonOffsetAgent):
        """Test 4: Quality rating thresholds are defined."""
        thresholds = agent.QUALITY_THRESHOLDS
        assert thresholds["excellent"] == 80
        assert thresholds["good"] == 65
        assert thresholds["acceptable"] == 50
        assert thresholds["poor"] == 35
        assert thresholds["unacceptable"] == 0

    def test_5_registry_standards_loaded(self, agent: CarbonOffsetAgent):
        """Test 5: All registry standards are loaded."""
        assert len(agent.registry_standards) == 6
        assert CarbonRegistry.VERRA_VCS in agent.registry_standards
        assert CarbonRegistry.GOLD_STANDARD in agent.registry_standards
        assert CarbonRegistry.ACR in agent.registry_standards
        assert CarbonRegistry.CAR in agent.registry_standards
        assert CarbonRegistry.PLAN_VIVO in agent.registry_standards
        assert CarbonRegistry.PURO_EARTH in agent.registry_standards


# =============================================================================
# Test 6-10: Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_6_valid_basic_input(self, basic_input: CarbonOffsetInput):
        """Test 6: Valid basic input passes validation."""
        assert basic_input.project_id == "VCS-1234"
        assert basic_input.registry == CarbonRegistry.VERRA_VCS
        assert len(basic_input.credits) == 1

    def test_7_valid_comprehensive_input(self, comprehensive_input: CarbonOffsetInput):
        """Test 7: Comprehensive input passes validation."""
        assert len(comprehensive_input.credits) == 3
        assert comprehensive_input.project_details is not None
        assert comprehensive_input.project_details.project_type == ProjectType.REDD_PLUS

    def test_8_credit_validation(self):
        """Test 8: Carbon credit validation works correctly."""
        # Valid credit
        credit = CarbonCredit(
            serial_number="VCS-TEST-001",
            vintage_year=2023,
            quantity_tco2e=100.0,
        )
        assert credit.vintage_year == 2023
        assert credit.quantity_tco2e == 100.0

        # Invalid vintage year should raise
        with pytest.raises(ValueError):
            CarbonCredit(
                serial_number="VCS-TEST-001",
                vintage_year=1999,  # Before 2000
                quantity_tco2e=100.0,
            )

    def test_9_empty_serial_number_rejected(self):
        """Test 9: Empty serial number is rejected."""
        with pytest.raises(ValueError):
            CarbonCredit(
                serial_number="   ",
                vintage_year=2023,
                quantity_tco2e=100.0,
            )

    def test_10_negative_quantity_rejected(self):
        """Test 10: Negative quantity is rejected."""
        with pytest.raises(ValueError):
            CarbonCredit(
                serial_number="VCS-TEST-001",
                vintage_year=2023,
                quantity_tco2e=-100.0,
            )


# =============================================================================
# Test 11-15: ICVCM Quality Scoring Tests
# =============================================================================


class TestICVCMScoring:
    """Tests for ICVCM Core Carbon Principles scoring."""

    def test_11_icvcm_scores_calculated(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 11: ICVCM scores are calculated correctly."""
        result = agent.run(comprehensive_input)

        assert result.icvcm_scores is not None
        assert 0 <= result.icvcm_scores.additionality_score <= 100
        assert 0 <= result.icvcm_scores.permanence_score <= 100
        assert 0 <= result.icvcm_scores.mrv_score <= 100
        assert 0 <= result.icvcm_scores.cobenefits_score <= 100
        assert 0 <= result.icvcm_scores.governance_score <= 100

    def test_12_weighted_total_formula(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 12: Weighted total follows ICVCM formula."""
        result = agent.run(comprehensive_input)
        scores = result.icvcm_scores

        expected_total = (
            scores.additionality_score * 0.30
            + scores.permanence_score * 0.25
            + scores.mrv_score * 0.20
            + scores.cobenefits_score * 0.15
            + scores.governance_score * 0.10
        )

        assert abs(scores.weighted_total - expected_total) < 0.1

    def test_13_removal_projects_higher_additionality(
        self,
        agent: CarbonOffsetAgent,
        removal_project_input: CarbonOffsetInput,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 13: Carbon removal projects score higher on additionality."""
        removal_result = agent.run(removal_project_input)
        avoidance_result = agent.run(comprehensive_input)

        # DAC should have higher additionality than REDD+
        assert (
            removal_result.icvcm_scores.additionality_score
            >= avoidance_result.icvcm_scores.additionality_score
        )

    def test_14_gold_standard_higher_cobenefits(self, agent: CarbonOffsetAgent):
        """Test 14: Gold Standard projects score higher on co-benefits."""
        gs_input = CarbonOffsetInput(
            project_id="GS-1234",
            registry=CarbonRegistry.GOLD_STANDARD,
            credits=[
                CarbonCredit(
                    serial_number="GS-1234-2023-001",
                    vintage_year=2023,
                    quantity_tco2e=100.0,
                )
            ],
            project_details=ProjectDetails(
                project_id="GS-1234",
                project_name="Clean Cookstoves Kenya",
                project_type=ProjectType.COOKSTOVES,
                registry=CarbonRegistry.GOLD_STANDARD,
                country="KE",
                sdg_contributions=[1, 3, 5, 7, 13],
            ),
        )

        vcs_input = CarbonOffsetInput(
            project_id="VCS-1234",
            registry=CarbonRegistry.VERRA_VCS,
            credits=[
                CarbonCredit(
                    serial_number="VCS-1234-2023-001",
                    vintage_year=2023,
                    quantity_tco2e=100.0,
                )
            ],
        )

        gs_result = agent.run(gs_input)
        vcs_result = agent.run(vcs_input)

        assert gs_result.icvcm_scores.cobenefits_score >= vcs_result.icvcm_scores.cobenefits_score

    def test_15_quality_rating_assignment(self, agent: CarbonOffsetAgent):
        """Test 15: Quality rating is assigned based on score."""
        assert agent._get_quality_rating(85) == "excellent"
        assert agent._get_quality_rating(70) == "good"
        assert agent._get_quality_rating(55) == "acceptable"
        assert agent._get_quality_rating(40) == "poor"
        assert agent._get_quality_rating(20) == "unacceptable"


# =============================================================================
# Test 16-20: Registry Verification Tests
# =============================================================================


class TestRegistryVerification:
    """Tests for registry API verification."""

    def test_16_verra_credit_verification(self, agent: CarbonOffsetAgent):
        """Test 16: Verra VCS credit verification works."""
        result = agent.verify_credit_with_registry(
            serial_number="VCS-1234-2023-001",
            registry=CarbonRegistry.VERRA_VCS,
        )

        assert result.verified is True
        assert result.project_exists is True
        assert result.registry == CarbonRegistry.VERRA_VCS
        assert result.api_response_code == 200

    def test_17_gold_standard_credit_verification(self, agent: CarbonOffsetAgent):
        """Test 17: Gold Standard credit verification works."""
        result = agent.verify_credit_with_registry(
            serial_number="GS-5678-2023-001",
            registry=CarbonRegistry.GOLD_STANDARD,
        )

        assert result.verified is True
        assert result.project_exists is True

    def test_18_mismatched_registry_format(self, agent: CarbonOffsetAgent):
        """Test 18: Mismatched serial number format is detected."""
        # VCS serial number against Gold Standard registry
        result = agent.verify_credit_with_registry(
            serial_number="VCS-1234-2023-001",
            registry=CarbonRegistry.GOLD_STANDARD,
        )

        assert result.verified is False
        assert result.project_exists is False
        assert result.api_response_code == 404
        assert result.error_message is not None

    def test_19_all_registries_verifiable(self, agent: CarbonOffsetAgent):
        """Test 19: All registries can verify credits."""
        test_cases = [
            ("VCS-001", CarbonRegistry.VERRA_VCS),
            ("GS-001", CarbonRegistry.GOLD_STANDARD),
            ("ACR-001", CarbonRegistry.ACR),
            ("CAR-001", CarbonRegistry.CAR),
            ("PV-001", CarbonRegistry.PLAN_VIVO),
            ("PURO-001", CarbonRegistry.PURO_EARTH),
        ]

        for serial, registry in test_cases:
            result = agent.verify_credit_with_registry(serial, registry)
            assert result.registry == registry

    def test_20_registry_verification_in_full_run(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 20: Registry verification included in full run."""
        result = agent.run(comprehensive_input)

        assert len(result.registry_verifications) == len(comprehensive_input.credits)
        for verification in result.registry_verifications:
            assert verification.registry == comprehensive_input.registry


# =============================================================================
# Test 21-25: Article 6.4 Compliance Tests
# =============================================================================


class TestArticle6Compliance:
    """Tests for Article 6.4 compliance assessment."""

    def test_21_article6_compliance_assessed(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 21: Article 6.4 compliance is assessed for all credits."""
        result = agent.run(comprehensive_input)

        assert len(result.article6_compliance) == len(comprehensive_input.credits)
        assert result.article6_overall_score >= 0
        assert result.article6_overall_score <= 100

    def test_22_authorized_credit_high_score(self, agent: CarbonOffsetAgent):
        """Test 22: Fully authorized credits get high compliance score."""
        credit = CarbonCredit(
            serial_number="VCS-AUTH-001",
            vintage_year=2023,
            quantity_tco2e=100.0,
            article6_authorization=Article6AuthorizationStatus.AUTHORIZED,
            corresponding_adjustment=CorrespondingAdjustmentStatus.APPLIED,
        )

        input_data = CarbonOffsetInput(
            project_id="VCS-AUTH",
            registry=CarbonRegistry.VERRA_VCS,
            credits=[credit],
            project_details=ProjectDetails(
                project_id="VCS-AUTH",
                project_name="Authorized Project",
                project_type=ProjectType.RENEWABLE_ENERGY,
                registry=CarbonRegistry.VERRA_VCS,
                country="BR",
            ),
        )

        result = agent.run(input_data)
        assert result.article6_overall_score >= 90

    def test_23_unauthorized_credit_low_score(self, agent: CarbonOffsetAgent):
        """Test 23: Unauthorized credits get lower compliance score."""
        credit = CarbonCredit(
            serial_number="VCS-UNAUTH-001",
            vintage_year=2023,
            quantity_tco2e=100.0,
            article6_authorization=Article6AuthorizationStatus.NOT_AUTHORIZED,
            corresponding_adjustment=CorrespondingAdjustmentStatus.NOT_APPLIED,
        )

        input_data = CarbonOffsetInput(
            project_id="VCS-UNAUTH",
            registry=CarbonRegistry.VERRA_VCS,
            credits=[credit],
            require_corresponding_adjustment=True,
        )

        result = agent.run(input_data)
        assert result.article6_overall_score < 50

    def test_24_itmo_eligibility_check(self, agent: CarbonOffsetAgent):
        """Test 24: ITMO eligibility is correctly determined."""
        # Eligible: authorized + CA applied
        eligible_credit = CarbonCredit(
            serial_number="VCS-ITMO-001",
            vintage_year=2023,
            quantity_tco2e=100.0,
            article6_authorization=Article6AuthorizationStatus.AUTHORIZED,
            corresponding_adjustment=CorrespondingAdjustmentStatus.APPLIED,
        )

        input_data = CarbonOffsetInput(
            project_id="VCS-ITMO",
            registry=CarbonRegistry.VERRA_VCS,
            credits=[eligible_credit],
            project_details=ProjectDetails(
                project_id="VCS-ITMO",
                project_name="ITMO Eligible Project",
                project_type=ProjectType.RENEWABLE_ENERGY,
                registry=CarbonRegistry.VERRA_VCS,
                country="IN",
            ),
        )

        result = agent.run(input_data)
        assert result.article6_compliance[0].itmo_eligible is True

    def test_25_compliance_issues_tracked(self, agent: CarbonOffsetAgent):
        """Test 25: Compliance issues are tracked correctly."""
        credit = CarbonCredit(
            serial_number="VCS-ISSUES-001",
            vintage_year=2023,
            quantity_tco2e=100.0,
            article6_authorization=Article6AuthorizationStatus.PENDING_AUTHORIZATION,
            corresponding_adjustment=CorrespondingAdjustmentStatus.PENDING,
        )

        input_data = CarbonOffsetInput(
            project_id="VCS-ISSUES",
            registry=CarbonRegistry.VERRA_VCS,
            credits=[credit],
        )

        result = agent.run(input_data)
        compliance = result.article6_compliance[0]

        assert len(compliance.compliance_issues) > 0


# =============================================================================
# Test 26-30: Portfolio Analysis Tests
# =============================================================================


class TestPortfolioAnalysis:
    """Tests for portfolio analysis functionality."""

    def test_26_portfolio_analysis_calculated(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 26: Portfolio analysis is calculated correctly."""
        result = agent.run(comprehensive_input)

        assert result.portfolio_analysis is not None
        assert result.portfolio_analysis.total_credits_tco2e > 0

    def test_27_removal_vs_avoidance_mix(
        self,
        agent: CarbonOffsetAgent,
        removal_project_input: CarbonOffsetInput,
    ):
        """Test 27: Removal vs avoidance mix is calculated."""
        result = agent.run(removal_project_input)
        portfolio = result.portfolio_analysis

        # DAC project should be 100% removal
        assert portfolio.removal_percentage == 100.0
        assert portfolio.removal_credits_tco2e == portfolio.total_credits_tco2e
        assert portfolio.avoidance_credits_tco2e == 0.0

    def test_28_vintage_distribution(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 28: Vintage distribution is calculated correctly."""
        result = agent.run(comprehensive_input)
        portfolio = result.portfolio_analysis

        # Should have 3 different vintages (2021, 2022, 2023)
        assert len(portfolio.vintage_distribution) == 3
        assert portfolio.oldest_vintage_year == 2021
        assert portfolio.newest_vintage_year == 2023

    def test_29_diversification_score(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 29: Diversification score is calculated."""
        result = agent.run(comprehensive_input)
        portfolio = result.portfolio_analysis

        assert 0 <= portfolio.diversification_score <= 100
        # Single registry, single project type = low diversification
        assert len(portfolio.concentration_risks) > 0

    def test_30_portfolio_recommendations(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 30: Portfolio recommendations are generated."""
        result = agent.run(comprehensive_input)
        portfolio = result.portfolio_analysis

        # Should have recommendations for improvement
        assert portfolio.portfolio_recommendations is not None


# =============================================================================
# Test 31-35: Price Benchmarking and Full Run Tests
# =============================================================================


class TestPriceBenchmarking:
    """Tests for price benchmarking functionality."""

    def test_31_price_benchmark_calculated(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 31: Price benchmarking is calculated."""
        result = agent.run(comprehensive_input)

        assert result.price_summary is not None
        assert len(result.price_summary.credit_benchmarks) == len(comprehensive_input.credits)

    def test_32_price_tiers_assigned(self, agent: CarbonOffsetAgent):
        """Test 32: Price tiers are correctly assigned."""
        benchmarks = agent.get_price_benchmarks(ProjectType.DIRECT_AIR_CAPTURE)

        # DAC should have premium pricing
        assert benchmarks["low"] >= 200
        assert benchmarks["high"] >= 500

        # REDD+ should have lower pricing
        redd_benchmarks = agent.get_price_benchmarks(ProjectType.REDD_PLUS)
        assert redd_benchmarks["mid"] < benchmarks["mid"]

    def test_33_full_verification_run(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 33: Full verification run completes successfully."""
        result = agent.run(comprehensive_input)

        assert result is not None
        assert isinstance(result, CarbonOffsetOutput)
        assert result.verification_status in VerificationStatus
        assert result.quality_score >= 0
        assert result.quality_score <= 100
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_34_provenance_tracking(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 34: Provenance tracking is complete."""
        result1 = agent.run(comprehensive_input)
        result2 = agent.run(comprehensive_input)

        # Each run should have unique provenance hash (includes timestamp)
        assert result1.provenance_hash != result2.provenance_hash
        assert result1.processing_time_ms > 0

    def test_35_risk_assessment_complete(
        self,
        agent: CarbonOffsetAgent,
        comprehensive_input: CarbonOffsetInput,
    ):
        """Test 35: Risk assessment covers all dimensions."""
        result = agent.run(comprehensive_input)
        risk = result.risk_assessment

        assert risk.overall_risk in RiskLevel
        assert risk.reversal_risk in RiskLevel
        assert risk.permanence_risk in RiskLevel
        assert risk.double_counting_risk in RiskLevel
        assert risk.regulatory_risk in RiskLevel
        assert risk.reputational_risk in RiskLevel


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_credit_verification(self, agent: CarbonOffsetAgent):
        """Test single credit verification API."""
        result = agent.verify_credit(
            serial_number="VCS-SINGLE-001",
            registry=CarbonRegistry.VERRA_VCS,
            vintage_year=2023,
            quantity_tco2e=50.0,
        )

        assert result is not None
        assert result.serial_number == "VCS-SINGLE-001"

    def test_retired_credits_handled(self, agent: CarbonOffsetAgent):
        """Test handling of retired credits."""
        credit = CarbonCredit(
            serial_number="VCS-RETIRED-001",
            vintage_year=2022,
            quantity_tco2e=100.0,
            retirement_status=RetirementStatus.RETIRED,
            retirement_date="2023-12-01",
            retirement_beneficiary="Test Company",
        )

        input_data = CarbonOffsetInput(
            project_id="VCS-RETIRED",
            registry=CarbonRegistry.VERRA_VCS,
            credits=[credit],
        )

        result = agent.run(input_data)

        # Retired credits should be flagged
        assert result.verification_status in {
            VerificationStatus.REJECTED,
            VerificationStatus.NEEDS_REVIEW,
        }

    def test_cancelled_credits_rejected(self, agent: CarbonOffsetAgent):
        """Test that cancelled credits are rejected."""
        credit = CarbonCredit(
            serial_number="VCS-CANCELLED-001",
            vintage_year=2022,
            quantity_tco2e=100.0,
            retirement_status=RetirementStatus.CANCELLED,
        )

        input_data = CarbonOffsetInput(
            project_id="VCS-CANCELLED",
            registry=CarbonRegistry.VERRA_VCS,
            credits=[credit],
        )

        result = agent.run(input_data)
        assert result.verification_status == VerificationStatus.REJECTED

    def test_old_vintage_warning(self, agent: CarbonOffsetAgent):
        """Test warning for old vintage credits."""
        credit = CarbonCredit(
            serial_number="VCS-OLD-001",
            vintage_year=2015,
            quantity_tco2e=100.0,
        )

        input_data = CarbonOffsetInput(
            project_id="VCS-OLD",
            registry=CarbonRegistry.VERRA_VCS,
            credits=[credit],
            vintage_cutoff_years=5,
        )

        result = agent.run(input_data)

        # Should have vintage warning
        vintage_check = next(
            (c for c in result.verification_checks if c.check_name == "vintage_validation"),
            None,
        )
        assert vintage_check is not None
        assert vintage_check.passed is False

    def test_all_project_types_supported(self, agent: CarbonOffsetAgent):
        """Test that all project types are supported."""
        project_types = agent.get_project_types()

        assert len(project_types) > 0
        for pt in ProjectType:
            assert any(p["id"] == pt.value for p in project_types)


# =============================================================================
# Data Integrity Tests
# =============================================================================


class TestDataIntegrity:
    """Tests for reference data integrity."""

    def test_registry_standards_complete(self):
        """Test all registry standards have required fields."""
        required_fields = [
            "registry",
            "name",
            "website",
            "methodology_count",
            "buffer_pool_required",
            "third_party_verification_required",
            "governance_score_base",
        ]

        for registry, standard in REGISTRY_STANDARDS.items():
            for field in required_fields:
                assert hasattr(standard, field), f"Missing {field} in {registry}"

    def test_project_profiles_complete(self):
        """Test all project profiles have required fields."""
        required_fields = [
            "base_additionality",
            "permanence_risk",
            "reversal_buffer_minimum",
            "typical_crediting_period",
            "monitoring_complexity",
        ]

        for project_type, profile in PROJECT_TYPE_PROFILES.items():
            for field in required_fields:
                assert field in profile, f"Missing {field} in {project_type}"

    def test_price_benchmarks_reasonable(self):
        """Test price benchmarks are reasonable."""
        for project_type, prices in PRICE_BENCHMARKS.items():
            assert prices["low"] > 0
            assert prices["mid"] >= prices["low"]
            assert prices["high"] >= prices["mid"]

            # DAC and tech removal should be premium
            if project_type in {
                ProjectType.DIRECT_AIR_CAPTURE,
                ProjectType.BIOENERGY_CCS,
                ProjectType.ENHANCED_WEATHERING,
            }:
                assert prices["mid"] >= 100

    def test_removal_types_classified(self):
        """Test removal project types are correctly classified."""
        assert ProjectType.DIRECT_AIR_CAPTURE in REMOVAL_PROJECT_TYPES
        assert ProjectType.BIOCHAR in REMOVAL_PROJECT_TYPES
        assert ProjectType.AFFORESTATION in REMOVAL_PROJECT_TYPES

        # Technology avoidance should NOT be removal
        assert ProjectType.RENEWABLE_ENERGY not in REMOVAL_PROJECT_TYPES
        assert ProjectType.ENERGY_EFFICIENCY not in REMOVAL_PROJECT_TYPES


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
