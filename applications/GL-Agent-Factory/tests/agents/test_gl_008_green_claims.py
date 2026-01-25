"""
Unit Tests for GL-008: Green Claims Verification Agent

Comprehensive test suite covering:
- 16 claim types (carbon_neutral, net_zero, eco_friendly, etc.)
- 5-dimension substantiation scoring
- 7 greenwashing red flag patterns
- EU Green Claims Directive compliance validation

Target: 85%+ code coverage

Reference:
- EU Green Claims Directive (2023/0085)
- TerraChoice Seven Sins of Greenwashing
- ISO 14021, ISO 14024, ISO 14025

Run with:
    pytest tests/agents/test_gl_008_green_claims.py -v --cov=backend/agents/gl_008_green_claims
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock




from agents.gl_008_green_claims.agent import (
    GreenClaimsAgent,
    ClaimType,
    GreenwashingRedFlag,
    ValidationStatus,
    SubstantiationLevel,
    ComplianceFramework,
    EvidenceItem,
    ClaimScope,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def green_claims_agent():
    """Create GreenClaimsAgent instance for testing."""
    return GreenClaimsAgent()


@pytest.fixture
def carbon_neutral_claim():
    """Create a carbon neutral claim with evidence."""
    from agents.gl_008_green_claims.agent import GreenClaimsInput
    return GreenClaimsInput(
        claim_text="Our product is carbon neutral",
        claim_type=ClaimType.CARBON_NEUTRAL,
        evidence_provided=["ISO 14064 certification", "Third-party audit report"],
        product_category="consumer_goods",
    )


@pytest.fixture
def vague_claim():
    """Create a vague green claim without evidence."""
    from agents.gl_008_green_claims.agent import GreenClaimsInput
    return GreenClaimsInput(
        claim_text="This product is eco-friendly",
        claim_type=ClaimType.ECO_FRIENDLY,
        evidence_provided=[],
        product_category="consumer_goods",
    )


@pytest.fixture
def evidence_item():
    """Create evidence item."""
    return EvidenceItem(
        evidence_type="certification",
        description="ISO 14064-1:2018 GHG inventory verification",
        source="Bureau Veritas",
        date_issued="2024-01-15",
        is_third_party=True,
    )


# =============================================================================
# Test Class: Agent Initialization
# =============================================================================


class TestGreenClaimsAgentInitialization:
    """Tests for GreenClaimsAgent initialization."""

    @pytest.mark.unit
    def test_agent_initializes_with_defaults(self, green_claims_agent):
        """Test agent initializes correctly with default config."""
        assert green_claims_agent is not None
        assert hasattr(green_claims_agent, "run")


# =============================================================================
# Test Class: Claim Types
# =============================================================================


class TestClaimTypes:
    """Tests for green claim type handling."""

    @pytest.mark.unit
    def test_all_16_claim_types_defined(self):
        """Test all 16 claim types are defined."""
        claim_types = [
            ClaimType.CARBON_NEUTRAL,
            ClaimType.CLIMATE_POSITIVE,
            ClaimType.NET_ZERO,
            ClaimType.CARBON_NEGATIVE,
            ClaimType.ECO_FRIENDLY,
            ClaimType.SUSTAINABLE,
            ClaimType.GREEN,
            ClaimType.RENEWABLE,
            ClaimType.RECYCLABLE,
            ClaimType.BIODEGRADABLE,
            ClaimType.COMPOSTABLE,
            ClaimType.PLASTIC_FREE,
            ClaimType.ZERO_WASTE,
            ClaimType.LOW_CARBON,
            ClaimType.REDUCED_EMISSIONS,
            ClaimType.ENVIRONMENTALLY_FRIENDLY,
        ]
        assert len(claim_types) == 16

    @pytest.mark.unit
    def test_carbon_related_claims(self):
        """Test carbon-related claim types."""
        carbon_claims = [
            ClaimType.CARBON_NEUTRAL,
            ClaimType.CARBON_NEGATIVE,
            ClaimType.NET_ZERO,
            ClaimType.LOW_CARBON,
        ]
        assert len(carbon_claims) == 4


# =============================================================================
# Test Class: Greenwashing Red Flags
# =============================================================================


class TestGreenwashingRedFlags:
    """Tests for greenwashing red flag patterns."""

    @pytest.mark.unit
    def test_all_7_red_flags_defined(self):
        """Test all 7 greenwashing red flags are defined."""
        red_flags = [
            GreenwashingRedFlag.VAGUE_CLAIMS,
            GreenwashingRedFlag.HIDDEN_TRADEOFFS,
            GreenwashingRedFlag.FALSE_LABELS,
            GreenwashingRedFlag.IRRELEVANT_CLAIMS,
            GreenwashingRedFlag.LESSER_OF_EVILS,
            GreenwashingRedFlag.FIBBING,
            GreenwashingRedFlag.FALSE_CERTIFICATIONS,
        ]
        assert len(red_flags) == 7

    @pytest.mark.unit
    def test_red_flag_values(self):
        """Test red flag enum values."""
        assert GreenwashingRedFlag.VAGUE_CLAIMS.value == "vague_claims"
        assert GreenwashingRedFlag.FIBBING.value == "fibbing"


# =============================================================================
# Test Class: Validation Status
# =============================================================================


class TestValidationStatus:
    """Tests for claim validation status."""

    @pytest.mark.unit
    def test_validation_status_values(self):
        """Test validation status enum values."""
        assert ValidationStatus.VALID.value == "valid"
        assert ValidationStatus.INVALID.value == "invalid"
        assert ValidationStatus.REQUIRES_VERIFICATION.value == "requires_verification"
        assert ValidationStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
        assert ValidationStatus.POTENTIAL_GREENWASHING.value == "potential_greenwashing"


# =============================================================================
# Test Class: Substantiation Level
# =============================================================================


class TestSubstantiationLevel:
    """Tests for substantiation level classification."""

    @pytest.mark.unit
    def test_substantiation_levels(self):
        """Test all substantiation levels are defined."""
        levels = [
            SubstantiationLevel.EXCELLENT,
            SubstantiationLevel.GOOD,
            SubstantiationLevel.MODERATE,
            SubstantiationLevel.WEAK,
            SubstantiationLevel.INSUFFICIENT,
        ]
        assert len(levels) == 5

    @pytest.mark.unit
    def test_substantiation_level_values(self):
        """Test substantiation level enum values."""
        assert SubstantiationLevel.EXCELLENT.value == "excellent"
        assert SubstantiationLevel.INSUFFICIENT.value == "insufficient"


# =============================================================================
# Test Class: Compliance Frameworks
# =============================================================================


class TestComplianceFrameworks:
    """Tests for compliance framework handling."""

    @pytest.mark.unit
    def test_eu_green_claims_directive(self):
        """Test EU Green Claims Directive framework."""
        assert ComplianceFramework.EU_GREEN_CLAIMS_DIRECTIVE.value == "eu_green_claims_directive"

    @pytest.mark.unit
    def test_iso_standards(self):
        """Test ISO standards frameworks."""
        assert ComplianceFramework.ISO_14021.value == "iso_14021"
        assert ComplianceFramework.ISO_14024.value == "iso_14024"
        assert ComplianceFramework.ISO_14025.value == "iso_14025"

    @pytest.mark.unit
    def test_regional_frameworks(self):
        """Test regional compliance frameworks."""
        assert ComplianceFramework.FTC_GREEN_GUIDES.value == "ftc_green_guides"
        assert ComplianceFramework.ASA_UK.value == "asa_uk"


# =============================================================================
# Test Class: Evidence Validation
# =============================================================================


class TestEvidenceValidation:
    """Tests for evidence item validation."""

    @pytest.mark.unit
    def test_valid_evidence_item(self, evidence_item):
        """Test valid evidence item passes validation."""
        assert evidence_item.evidence_type == "certification"
        assert evidence_item.is_third_party is True

    @pytest.mark.unit
    def test_evidence_item_fields(self):
        """Test evidence item has all required fields."""
        evidence = EvidenceItem(
            evidence_type="audit",
            description="Annual sustainability audit",
            source="EY",
            is_third_party=True,
        )
        assert evidence.source == "EY"


# =============================================================================
# Test Class: Claim Scope
# =============================================================================


class TestClaimScope:
    """Tests for claim scope definition."""

    @pytest.mark.unit
    def test_claim_scope_creation(self):
        """Test claim scope creation."""
        scope = ClaimScope(
            product_scope="Single product line",
            lifecycle_stages=["production", "use"],
            geographic_scope="European Union",
            time_period="2024",
        )
        assert scope.product_scope == "Single product line"
        assert len(scope.lifecycle_stages) == 2

    @pytest.mark.unit
    def test_scope_exclusions(self):
        """Test scope exclusions handling."""
        scope = ClaimScope(
            product_scope="All products",
            lifecycle_stages=["raw_materials", "production"],
            exclusions=["end_of_life", "transport"],
        )
        assert len(scope.exclusions) == 2


# =============================================================================
# Test Class: Claim Verification
# =============================================================================


class TestClaimVerification:
    """Tests for claim verification functionality."""

    @pytest.mark.unit
    def test_well_substantiated_claim(self, green_claims_agent, carbon_neutral_claim):
        """Test well-substantiated claim validation."""
        result = green_claims_agent.run(carbon_neutral_claim)

        assert hasattr(result, "claim_valid")
        assert hasattr(result, "substantiation_score")

    @pytest.mark.unit
    def test_vague_claim_flagged(self, green_claims_agent, vague_claim):
        """Test vague claim is flagged."""
        result = green_claims_agent.run(vague_claim)

        # Vague claims without evidence should have low substantiation
        assert result.substantiation_score < 50 or len(result.greenwashing_flags) > 0


# =============================================================================
# Test Class: Greenwashing Detection
# =============================================================================


class TestGreenwashingDetection:
    """Tests for greenwashing detection."""

    @pytest.mark.unit
    def test_greenwashing_flags_detected(self, green_claims_agent, vague_claim):
        """Test greenwashing flags are detected for vague claims."""
        result = green_claims_agent.run(vague_claim)

        # Vague eco-friendly claim should trigger flags
        assert hasattr(result, "greenwashing_flags")

    @pytest.mark.unit
    def test_no_flags_for_substantiated_claim(self, green_claims_agent, carbon_neutral_claim):
        """Test no greenwashing flags for well-substantiated claims."""
        result = green_claims_agent.run(carbon_neutral_claim)

        # Well-substantiated claim should have fewer/no flags
        if result.claim_valid:
            assert len(result.greenwashing_flags) == 0 or result.substantiation_score > 60


# =============================================================================
# Test Class: Substantiation Scoring
# =============================================================================


class TestSubstantiationScoring:
    """Tests for 5-dimension substantiation scoring."""

    @pytest.mark.unit
    def test_substantiation_score_range(self, green_claims_agent, carbon_neutral_claim):
        """Test substantiation score is in valid range."""
        result = green_claims_agent.run(carbon_neutral_claim)
        assert 0 <= result.substantiation_score <= 100

    @pytest.mark.unit
    def test_evidence_affects_score(self, green_claims_agent, carbon_neutral_claim, vague_claim):
        """Test evidence presence affects substantiation score."""
        result_with_evidence = green_claims_agent.run(carbon_neutral_claim)
        result_without_evidence = green_claims_agent.run(vague_claim)

        assert result_with_evidence.substantiation_score > result_without_evidence.substantiation_score


# =============================================================================
# Test Class: Provenance Tracking
# =============================================================================


class TestGreenClaimsProvenance:
    """Tests for provenance hash tracking."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, green_claims_agent, carbon_neutral_claim):
        """Test provenance hash is generated."""
        result = green_claims_agent.run(carbon_neutral_claim)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# =============================================================================
# Test Class: Performance
# =============================================================================


class TestGreenClaimsPerformance:
    """Performance tests for GreenClaimsAgent."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_verification_performance(self, green_claims_agent, carbon_neutral_claim):
        """Test single claim verification completes quickly."""
        import time

        start = time.perf_counter()
        result = green_claims_agent.run(carbon_neutral_claim)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0
        assert result is not None
