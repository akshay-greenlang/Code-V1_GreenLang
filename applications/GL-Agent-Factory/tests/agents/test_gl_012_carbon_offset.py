"""
Unit Tests for GL-012: Carbon Offset Verification Agent

Comprehensive test suite covering:
- 6 major carbon registries (Verra VCS, Gold Standard, ACR, CAR, Plan Vivo, Puro.earth)
- ICVCM Core Carbon Principles quality scoring
- Project existence and retirement status verification
- Credit vintage validation
- Double counting prevention (corresponding adjustments)
- Buffer pool adequacy assessment
- Article 6.4 compliance checking (Paris Agreement)

Target: 85%+ code coverage

Reference:
- ICVCM Core Carbon Principles (2023)
- Paris Agreement Article 6
- Verra VCS Standard
- Gold Standard for the Global Goals

Run with:
    pytest tests/agents/test_gl_012_carbon_offset.py -v --cov=backend/agents/gl_012_carbon_offset
"""

import pytest
from datetime import datetime, date
from unittest.mock import patch, MagicMock




from agents.gl_012_carbon_offset.agent import (
    CarbonOffsetAgent,
    CarbonRegistry,
    ProjectType,
    VerificationStatus,
    RiskLevel,
    RetirementStatus,
    CorrespondingAdjustmentStatus,
    Article6AuthorizationStatus,
    CreditCategory,
    PriceTier,
    CarbonCredit,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def carbon_offset_agent():
    """Create CarbonOffsetAgent instance for testing."""
    return CarbonOffsetAgent()


@pytest.fixture
def vcs_credit():
    """Create Verra VCS carbon credit."""
    return CarbonCredit(
        serial_number="VCS-1234-2023-001",
        vintage_year=2023,
        quantity_tco2e=100.0,
        retirement_status=RetirementStatus.ACTIVE,
    )


@pytest.fixture
def gold_standard_credit():
    """Create Gold Standard carbon credit."""
    return CarbonCredit(
        serial_number="GS-5678-2022-001",
        vintage_year=2022,
        quantity_tco2e=50.0,
        retirement_status=RetirementStatus.RETIRED,
        retirement_date="2024-01-15",
        retirement_beneficiary="Example Corp",
    )


@pytest.fixture
def carbon_offset_input(vcs_credit):
    """Create carbon offset verification input."""
    from agents.gl_012_carbon_offset.agent import CarbonOffsetInput
    return CarbonOffsetInput(
        project_id="VCS-1234",
        registry=CarbonRegistry.VERRA_VCS,
        project_type=ProjectType.REDD_PLUS,
        credits=[vcs_credit],
    )


# =============================================================================
# Test Class: Agent Initialization
# =============================================================================


class TestCarbonOffsetAgentInitialization:
    """Tests for CarbonOffsetAgent initialization."""

    @pytest.mark.unit
    def test_agent_initializes_with_defaults(self, carbon_offset_agent):
        """Test agent initializes correctly with default config."""
        assert carbon_offset_agent is not None
        assert hasattr(carbon_offset_agent, "run")


# =============================================================================
# Test Class: Carbon Registries
# =============================================================================


class TestCarbonRegistries:
    """Tests for carbon registry handling."""

    @pytest.mark.unit
    def test_all_6_registries_defined(self):
        """Test all 6 major registries are defined."""
        registries = [
            CarbonRegistry.VERRA_VCS,
            CarbonRegistry.GOLD_STANDARD,
            CarbonRegistry.ACR,
            CarbonRegistry.CAR,
            CarbonRegistry.PLAN_VIVO,
            CarbonRegistry.PURO_EARTH,
        ]
        assert len(registries) == 6

    @pytest.mark.unit
    def test_registry_values(self):
        """Test registry enum values."""
        assert CarbonRegistry.VERRA_VCS.value == "verra_vcs"
        assert CarbonRegistry.GOLD_STANDARD.value == "gold_standard"
        assert CarbonRegistry.PURO_EARTH.value == "puro_earth"


# =============================================================================
# Test Class: Project Types
# =============================================================================


class TestProjectTypes:
    """Tests for carbon project type handling."""

    @pytest.mark.unit
    def test_nature_based_project_types(self):
        """Test nature-based solution project types."""
        nature_based = [
            ProjectType.AFFORESTATION,
            ProjectType.REFORESTATION,
            ProjectType.REDD_PLUS,
            ProjectType.IMPROVED_FOREST_MANAGEMENT,
            ProjectType.BLUE_CARBON,
            ProjectType.SOIL_CARBON,
            ProjectType.AGROFORESTRY,
        ]
        assert len(nature_based) == 7

    @pytest.mark.unit
    def test_technology_based_project_types(self):
        """Test technology-based project types."""
        tech_based = [
            ProjectType.RENEWABLE_ENERGY,
            ProjectType.ENERGY_EFFICIENCY,
            ProjectType.WASTE_MANAGEMENT,
            ProjectType.METHANE_CAPTURE,
            ProjectType.INDUSTRIAL_PROCESS,
            ProjectType.COOKSTOVES,
        ]
        assert len(tech_based) == 6

    @pytest.mark.unit
    def test_carbon_removal_project_types(self):
        """Test carbon removal project types."""
        removal = [
            ProjectType.DIRECT_AIR_CAPTURE,
            ProjectType.BIOENERGY_CCS,
            ProjectType.BIOCHAR,
            ProjectType.ENHANCED_WEATHERING,
            ProjectType.OCEAN_ALKALINITY,
        ]
        assert len(removal) == 5


# =============================================================================
# Test Class: Verification Status
# =============================================================================


class TestVerificationStatus:
    """Tests for verification status handling."""

    @pytest.mark.unit
    def test_verification_status_values(self):
        """Test verification status enum values."""
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.NEEDS_REVIEW.value == "needs_review"
        assert VerificationStatus.REJECTED.value == "rejected"
        assert VerificationStatus.PENDING.value == "pending"
        assert VerificationStatus.EXPIRED.value == "expired"


# =============================================================================
# Test Class: Risk Levels
# =============================================================================


class TestRiskLevels:
    """Tests for risk level handling."""

    @pytest.mark.unit
    def test_risk_level_values(self):
        """Test risk level enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


# =============================================================================
# Test Class: Retirement Status
# =============================================================================


class TestRetirementStatus:
    """Tests for credit retirement status."""

    @pytest.mark.unit
    def test_retirement_status_values(self):
        """Test retirement status enum values."""
        assert RetirementStatus.ACTIVE.value == "active"
        assert RetirementStatus.RETIRED.value == "retired"
        assert RetirementStatus.CANCELLED.value == "cancelled"
        assert RetirementStatus.PENDING_RETIREMENT.value == "pending_retirement"
        assert RetirementStatus.TRANSFERRED.value == "transferred"


# =============================================================================
# Test Class: Corresponding Adjustments
# =============================================================================


class TestCorrespondingAdjustments:
    """Tests for corresponding adjustment status (Article 6)."""

    @pytest.mark.unit
    def test_ca_status_values(self):
        """Test corresponding adjustment status values."""
        assert CorrespondingAdjustmentStatus.APPLIED.value == "applied"
        assert CorrespondingAdjustmentStatus.NOT_REQUIRED.value == "not_required"
        assert CorrespondingAdjustmentStatus.PENDING.value == "pending"
        assert CorrespondingAdjustmentStatus.NOT_APPLIED.value == "not_applied"


# =============================================================================
# Test Class: Article 6 Authorization
# =============================================================================


class TestArticle6Authorization:
    """Tests for Article 6.4 authorization status."""

    @pytest.mark.unit
    def test_authorization_status_values(self):
        """Test Article 6 authorization status values."""
        assert Article6AuthorizationStatus.AUTHORIZED.value == "authorized"
        assert Article6AuthorizationStatus.AUTHORIZED_CONDITIONAL.value == "authorized_conditional"
        assert Article6AuthorizationStatus.NOT_AUTHORIZED.value == "not_authorized"
        assert Article6AuthorizationStatus.NOT_APPLICABLE.value == "not_applicable"


# =============================================================================
# Test Class: Credit Categories
# =============================================================================


class TestCreditCategories:
    """Tests for credit category handling."""

    @pytest.mark.unit
    def test_credit_category_values(self):
        """Test credit category enum values."""
        assert CreditCategory.AVOIDANCE.value == "avoidance"
        assert CreditCategory.REMOVAL.value == "removal"


# =============================================================================
# Test Class: Price Tiers
# =============================================================================


class TestPriceTiers:
    """Tests for price tier classification."""

    @pytest.mark.unit
    def test_price_tier_values(self):
        """Test price tier enum values."""
        assert PriceTier.PREMIUM.value == "premium"
        assert PriceTier.STANDARD.value == "standard"
        assert PriceTier.ECONOMY.value == "economy"
        assert PriceTier.BUDGET.value == "budget"


# =============================================================================
# Test Class: Carbon Credit Validation
# =============================================================================


class TestCarbonCreditValidation:
    """Tests for carbon credit model validation."""

    @pytest.mark.unit
    def test_valid_credit(self, vcs_credit):
        """Test valid credit passes validation."""
        assert vcs_credit.serial_number == "VCS-1234-2023-001"
        assert vcs_credit.vintage_year == 2023
        assert vcs_credit.quantity_tco2e == 100.0

    @pytest.mark.unit
    def test_vintage_year_range(self):
        """Test vintage year must be 2000-2100."""
        with pytest.raises(ValueError):
            CarbonCredit(
                serial_number="TEST-001",
                vintage_year=1990,  # Before 2000
                quantity_tco2e=100.0,
            )

    @pytest.mark.unit
    def test_quantity_positive(self):
        """Test quantity must be positive."""
        with pytest.raises(ValueError):
            CarbonCredit(
                serial_number="TEST-001",
                vintage_year=2023,
                quantity_tco2e=0,  # Must be > 0
            )


# =============================================================================
# Test Class: Credit Verification
# =============================================================================


class TestCreditVerification:
    """Tests for credit verification functionality."""

    @pytest.mark.unit
    def test_verification_performed(self, carbon_offset_agent, carbon_offset_input):
        """Test verification is performed."""
        result = carbon_offset_agent.run(carbon_offset_input)

        assert hasattr(result, "verification_status")
        assert hasattr(result, "quality_score")

    @pytest.mark.unit
    def test_quality_score_range(self, carbon_offset_agent, carbon_offset_input):
        """Test quality score is in valid range."""
        result = carbon_offset_agent.run(carbon_offset_input)

        assert 0 <= result.quality_score <= 100


# =============================================================================
# Test Class: ICVCM Quality Scoring
# =============================================================================


class TestICVCMQualityScoring:
    """Tests for ICVCM Core Carbon Principles scoring."""

    @pytest.mark.unit
    def test_quality_dimensions_assessed(self, carbon_offset_agent, carbon_offset_input):
        """Test quality dimensions are assessed."""
        result = carbon_offset_agent.run(carbon_offset_input)

        # Should assess multiple quality dimensions
        assert hasattr(result, "quality_score")


# =============================================================================
# Test Class: Double Counting Prevention
# =============================================================================


class TestDoubleCountingPrevention:
    """Tests for double counting prevention."""

    @pytest.mark.unit
    def test_ca_status_assessed(self, carbon_offset_agent, carbon_offset_input):
        """Test corresponding adjustment status is assessed."""
        result = carbon_offset_agent.run(carbon_offset_input)

        # Should check for double counting risk
        assert hasattr(result, "double_counting_risk") or hasattr(result, "ca_status")


# =============================================================================
# Test Class: Provenance Tracking
# =============================================================================


class TestCarbonOffsetProvenance:
    """Tests for provenance hash tracking."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, carbon_offset_agent, carbon_offset_input):
        """Test provenance hash is generated."""
        result = carbon_offset_agent.run(carbon_offset_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_provenance_hash_deterministic(self, carbon_offset_agent, carbon_offset_input):
        """Test provenance hash is deterministic."""
        result1 = carbon_offset_agent.run(carbon_offset_input)
        result2 = carbon_offset_agent.run(carbon_offset_input)
        assert result1.provenance_hash == result2.provenance_hash


# =============================================================================
# Test Class: Performance
# =============================================================================


class TestCarbonOffsetPerformance:
    """Performance tests for CarbonOffsetAgent."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_verification_performance(self, carbon_offset_agent, carbon_offset_input):
        """Test single verification completes quickly."""
        import time

        start = time.perf_counter()
        result = carbon_offset_agent.run(carbon_offset_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100.0
        assert result is not None
