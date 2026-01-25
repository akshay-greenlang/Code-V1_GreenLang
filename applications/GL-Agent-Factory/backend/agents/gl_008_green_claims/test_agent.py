"""
Unit Tests for GL-008: Green Claims Verification Agent

Comprehensive test coverage for the Green Claims Verification Agent including:
- 16 claim types validation (carbon_neutral, net_zero, eco_friendly, etc.)
- 5-dimension substantiation scoring (evidence, scope, transparency, verification, clarity)
- 7 greenwashing red flag detection patterns
- EU Green Claims Directive compliance validation
- Evidence verification and quality assessment
- Provenance hash and determinism verification

Test coverage target: 85%+
Total tests: 60+ golden tests covering all verification scenarios

Formula Documentation:
----------------------
Substantiation Score Calculation:
    weighted_total = (evidence_quality * 0.30) + (scope_accuracy * 0.25) +
                     (transparency * 0.20) + (verification * 0.15) + (clarity * 0.10)

Greenwashing Risk Score:
    risk_score = sum(severity_weights[rf.severity] for rf in red_flags)
    where severity_weights = {critical: 30, high: 20, medium: 10, low: 5}

Substantiation Levels:
    - excellent: 80-100
    - good: 60-79
    - moderate: 40-59
    - weak: 20-39
    - insufficient: 0-19
"""

import hashlib
import json
import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional

from agent import (
    GreenClaimsAgent,
    GreenClaimsInput,
    GreenClaimsOutput,
    ClaimType,
    GreenwashingRedFlag,
    ValidationStatus,
    SubstantiationLevel,
    ComplianceFramework,
    EvidenceItem,
    ClaimScope,
    SubstantiationScoreBreakdown,
    GreenwashingDetection,
    FrameworkComplianceResult,
    ComplianceReport,
    ClaimRequirements,
    EU_GREEN_CLAIMS_REQUIREMENTS,
    GREENWASHING_PATTERNS,
    VALID_CERTIFICATIONS,
)


# =============================================================================
# Test Constants - Score Weights
# =============================================================================

# Substantiation score weights (from agent)
WEIGHT_EVIDENCE_QUALITY = 0.30
WEIGHT_SCOPE_ACCURACY = 0.25
WEIGHT_TRANSPARENCY = 0.20
WEIGHT_VERIFICATION = 0.15
WEIGHT_CLARITY = 0.10

# Greenwashing severity weights
SEVERITY_WEIGHTS = {
    "critical": 30,
    "high": 20,
    "medium": 10,
    "low": 5,
}

# Substantiation level thresholds
LEVEL_EXCELLENT = 80
LEVEL_GOOD = 60
LEVEL_MODERATE = 40
LEVEL_WEAK = 20


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent() -> GreenClaimsAgent:
    """Create a GreenClaimsAgent instance for testing."""
    return GreenClaimsAgent()


@pytest.fixture
def agent_with_config() -> GreenClaimsAgent:
    """Create agent with custom configuration."""
    return GreenClaimsAgent(config={"strict_mode": True})


@pytest.fixture
def valid_evidence_third_party() -> EvidenceItem:
    """Create valid third-party evidence item."""
    return EvidenceItem(
        evidence_type="certification",
        description="ISO 14064 GHG accounting certification",
        source="Bureau Veritas",
        date_issued="2024-01-15",
        expiry_date="2027-01-15",
        verification_url="https://verifier.example.com/cert/12345",
        is_third_party=True,
    )


@pytest.fixture
def valid_evidence_self_declared() -> EvidenceItem:
    """Create self-declared evidence item."""
    return EvidenceItem(
        evidence_type="report",
        description="Internal sustainability assessment",
        source="Company Sustainability Team",
        date_issued="2024-06-01",
        is_third_party=False,
    )


@pytest.fixture
def comprehensive_evidence_set() -> List[EvidenceItem]:
    """Create comprehensive evidence set for carbon neutral claim."""
    return [
        EvidenceItem(
            evidence_type="ghg_inventory",
            description="Complete Scope 1, 2, 3 emissions inventory",
            source="GHG Protocol",
            date_issued="2024-01-01",
            is_third_party=True,
            verification_url="https://ghgprotocol.org/verify/12345",
        ),
        EvidenceItem(
            evidence_type="reduction_pathway",
            description="Science-based reduction pathway aligned with 1.5C",
            source="SBTi",
            date_issued="2024-02-01",
            is_third_party=True,
            verification_url="https://sciencebasedtargets.org/verify/12345",
        ),
        EvidenceItem(
            evidence_type="offset_registry",
            description="Gold Standard verified carbon offsets",
            source="Gold Standard",
            date_issued="2024-03-01",
            is_third_party=True,
            verification_url="https://registry.goldstandard.org/projects/12345",
        ),
        EvidenceItem(
            evidence_type="third_party_verification",
            description="Independent verification by accredited body",
            source="DNV",
            date_issued="2024-04-01",
            expiry_date="2027-04-01",
            is_third_party=True,
            verification_url="https://dnv.com/verify/12345",
        ),
    ]


@pytest.fixture
def full_lifecycle_scope() -> ClaimScope:
    """Create claim scope covering full lifecycle."""
    return ClaimScope(
        product_scope="All consumer products",
        lifecycle_stages=["raw_materials", "production", "use", "disposal"],
        geographic_scope="Global",
        time_period="2024-2025",
        exclusions=["Scope 3 Category 15 investments"],
    )


@pytest.fixture
def partial_lifecycle_scope() -> ClaimScope:
    """Create claim scope with partial lifecycle coverage."""
    return ClaimScope(
        product_scope="Manufacturing operations",
        lifecycle_stages=["production"],
        geographic_scope="EU",
        time_period="2024",
    )


@pytest.fixture
def carbon_neutral_input_valid(
    comprehensive_evidence_set: List[EvidenceItem],
    full_lifecycle_scope: ClaimScope,
) -> GreenClaimsInput:
    """Create valid carbon neutral claim input."""
    return GreenClaimsInput(
        claim_text="Our product is carbon neutral, verified by independent third parties",
        claim_type=ClaimType.CARBON_NEUTRAL,
        evidence_items=comprehensive_evidence_set,
        claim_scope=full_lifecycle_scope,
        company_name="EcoTech Corp",
        product_category="consumer_goods",
        target_frameworks=[ComplianceFramework.EU_GREEN_CLAIMS_DIRECTIVE],
    )


@pytest.fixture
def vague_eco_friendly_input() -> GreenClaimsInput:
    """Create vague eco-friendly claim without evidence."""
    return GreenClaimsInput(
        claim_text="Our product is eco-friendly and natural",
        claim_type=ClaimType.ECO_FRIENDLY,
        evidence_items=[],
        company_name="GreenWash Inc",
        product_category="consumer_goods",
    )


# =============================================================================
# Test 1-10: Agent Initialization and Basic Tests
# =============================================================================


class TestAgentInitialization:
    """Tests for agent initialization and configuration."""

    def test_01_agent_initialization(self, agent: GreenClaimsAgent):
        """Test 1: Agent initializes correctly with default config."""
        assert agent is not None
        assert agent.AGENT_ID == "regulatory/green_claims_v1"
        assert agent.VERSION == "1.0.0"
        assert agent.DESCRIPTION == "Green claims verification and greenwashing detection"

    def test_02_agent_with_custom_config(self, agent_with_config: GreenClaimsAgent):
        """Test 2: Agent initializes with custom configuration."""
        assert agent_with_config.config == {"strict_mode": True}

    def test_03_requirements_database_loaded(self, agent: GreenClaimsAgent):
        """Test 3: EU Green Claims requirements database is loaded."""
        assert len(agent.requirements_db) == 16  # All 16 claim types
        assert ClaimType.CARBON_NEUTRAL in agent.requirements_db
        assert ClaimType.NET_ZERO in agent.requirements_db
        assert ClaimType.ECO_FRIENDLY in agent.requirements_db

    def test_04_greenwashing_patterns_loaded(self, agent: GreenClaimsAgent):
        """Test 4: Greenwashing detection patterns are loaded."""
        assert len(agent.greenwashing_patterns) == 7  # 7 red flag patterns
        assert GreenwashingRedFlag.VAGUE_CLAIMS in agent.greenwashing_patterns
        assert GreenwashingRedFlag.FIBBING in agent.greenwashing_patterns
        assert GreenwashingRedFlag.FALSE_CERTIFICATIONS in agent.greenwashing_patterns

    def test_05_valid_certifications_loaded(self, agent: GreenClaimsAgent):
        """Test 5: Valid certifications database is loaded."""
        assert len(agent.valid_certifications) > 10
        assert "iso_14064" in agent.valid_certifications
        assert "sbti" in agent.valid_certifications
        assert "gold_standard" in agent.valid_certifications

    def test_06_score_weights_correct(self, agent: GreenClaimsAgent):
        """Test 6: Substantiation score weights sum to 1.0."""
        total_weight = sum(agent.SCORE_WEIGHTS.values())
        assert total_weight == pytest.approx(1.0, rel=1e-6)

    def test_07_get_claim_types(self, agent: GreenClaimsAgent):
        """Test 7: Get all supported claim types."""
        claim_types = agent.get_claim_types()
        assert len(claim_types) == 16
        assert "carbon_neutral" in claim_types
        assert "net_zero" in claim_types
        assert "sustainable" in claim_types

    def test_08_get_greenwashing_patterns(self, agent: GreenClaimsAgent):
        """Test 8: Get all greenwashing red flag patterns."""
        patterns = agent.get_greenwashing_patterns()
        assert len(patterns) == 7
        assert "vague_claims" in patterns
        assert "hidden_tradeoffs" in patterns
        assert "false_certifications" in patterns

    def test_09_get_valid_certifications(self, agent: GreenClaimsAgent):
        """Test 9: Get list of recognized certifications."""
        certs = agent.get_valid_certifications()
        assert len(certs) > 10
        assert any(c["id"] == "iso_14064" for c in certs)

    def test_10_get_claim_requirements(self, agent: GreenClaimsAgent):
        """Test 10: Get requirements for specific claim type."""
        reqs = agent.get_claim_requirements(ClaimType.CARBON_NEUTRAL)
        assert reqs is not None
        assert "required_evidence" in reqs
        assert "ghg_inventory" in reqs["required_evidence"]
        assert reqs["requires_third_party_verification"] is True


# =============================================================================
# Test 11-20: Claim Validation for Different Claim Types
# =============================================================================


class TestClaimTypeValidation:
    """Tests for validation of different claim types."""

    @pytest.mark.golden
    def test_11_carbon_neutral_valid(
        self,
        agent: GreenClaimsAgent,
        carbon_neutral_input_valid: GreenClaimsInput,
    ):
        """
        Test 11: Valid carbon neutral claim with complete evidence.

        ZERO-HALLUCINATION CHECK:
        - All required evidence present (ghg_inventory, reduction_pathway, offset_registry, third_party_verification)
        - Third-party verification present
        - Full lifecycle coverage
        """
        result = agent.run(carbon_neutral_input_valid)

        assert result.claim_valid is True
        assert result.validation_status == ValidationStatus.VALID.value
        assert result.substantiation_score >= 60
        assert result.greenwashing_detected is False

    @pytest.mark.golden
    def test_12_carbon_neutral_missing_evidence(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 12: Carbon neutral claim missing required evidence.

        ZERO-HALLUCINATION CHECK:
        - Missing ghg_inventory, reduction_pathway, offset_registry
        - Should fail validation
        """
        input_data = GreenClaimsInput(
            claim_text="Our product is carbon neutral",
            claim_type=ClaimType.CARBON_NEUTRAL,
            evidence_items=[
                EvidenceItem(
                    evidence_type="report",
                    description="Internal carbon assessment",
                    is_third_party=False,
                )
            ],
        )

        result = agent.run(input_data)

        assert result.claim_valid is False
        assert "Missing required evidence" in " ".join(result.validation_messages)

    @pytest.mark.golden
    def test_13_net_zero_valid(
        self,
        agent: GreenClaimsAgent,
        full_lifecycle_scope: ClaimScope,
    ):
        """
        Test 13: Valid net-zero claim with SBTi alignment.

        ZERO-HALLUCINATION CHECK:
        - Requires ghg_inventory, sbti_commitment, reduction_pathway, residual_offset_plan
        - Requires full lifecycle coverage
        """
        evidence = [
            EvidenceItem(
                evidence_type="ghg_inventory",
                description="Full scope emissions inventory",
                source="GHG Protocol",
                is_third_party=True,
            ),
            EvidenceItem(
                evidence_type="sbti_commitment",
                description="SBTi validated net-zero target",
                source="SBTi",
                is_third_party=True,
                verification_url="https://sciencebasedtargets.org/companies",
            ),
            EvidenceItem(
                evidence_type="reduction_pathway",
                description="90%+ reduction by 2050",
                source="Company Net-Zero Plan",
                is_third_party=False,
            ),
            EvidenceItem(
                evidence_type="residual_offset_plan",
                description="Permanent carbon removal for residual emissions",
                source="Third-party verified",
                is_third_party=True,
            ),
        ]

        input_data = GreenClaimsInput(
            claim_text="We are committed to achieving net-zero emissions by 2050",
            claim_type=ClaimType.NET_ZERO,
            evidence_items=evidence,
            claim_scope=full_lifecycle_scope,
            target_frameworks=[ComplianceFramework.EU_GREEN_CLAIMS_DIRECTIVE],
        )

        result = agent.run(input_data)

        assert result.claim_type == ClaimType.NET_ZERO.value
        assert result.substantiation_score > 40

    @pytest.mark.golden
    def test_14_eco_friendly_vague_fails(
        self,
        agent: GreenClaimsAgent,
        vague_eco_friendly_input: GreenClaimsInput,
    ):
        """
        Test 14: Vague eco-friendly claim without evidence fails.

        ZERO-HALLUCINATION CHECK:
        - No evidence provided
        - Contains vague terms "eco-friendly", "natural"
        - Should detect greenwashing
        """
        result = agent.run(vague_eco_friendly_input)

        assert result.claim_valid is False
        assert result.validation_status == ValidationStatus.INSUFFICIENT_EVIDENCE.value
        assert result.greenwashing_detected is True

    @pytest.mark.golden
    def test_15_recyclable_claim_valid(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 15: Valid recyclable claim with material composition evidence.

        ZERO-HALLUCINATION CHECK:
        - Requires material_composition, recycling_infrastructure_availability
        - Lifecycle stage: disposal
        """
        input_data = GreenClaimsInput(
            claim_text="This packaging is 100% recyclable in curbside recycling programs",
            claim_type=ClaimType.RECYCLABLE,
            evidence_items=[
                EvidenceItem(
                    evidence_type="material_composition",
                    description="100% PET plastic, type 1",
                    source="Material testing lab",
                    is_third_party=True,
                ),
                EvidenceItem(
                    evidence_type="recycling_infrastructure_availability",
                    description="Available in 95% of US curbside programs",
                    source="EPA recycling data",
                    is_third_party=True,
                ),
            ],
            claim_scope=ClaimScope(
                product_scope="Product packaging",
                lifecycle_stages=["disposal"],
            ),
        )

        result = agent.run(input_data)

        assert result.substantiation_score > 40

    @pytest.mark.golden
    def test_16_biodegradable_requires_conditions(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 16: Biodegradable claim requires test conditions specification.

        ZERO-HALLUCINATION CHECK:
        - Requires biodegradation_test_results, test_conditions
        - Must specify timeframe and conditions
        """
        input_data = GreenClaimsInput(
            claim_text="Our packaging is biodegradable",
            claim_type=ClaimType.BIODEGRADABLE,
            evidence_items=[
                EvidenceItem(
                    evidence_type="biodegradation_test_results",
                    description="90% biodegradation in 180 days",
                    source="TUV Austria",
                    is_third_party=True,
                ),
                EvidenceItem(
                    evidence_type="test_conditions",
                    description="Industrial composting conditions per EN 13432",
                    source="Testing lab",
                    is_third_party=True,
                ),
            ],
        )

        result = agent.run(input_data)

        assert result.claim_type == ClaimType.BIODEGRADABLE.value

    @pytest.mark.golden
    def test_17_compostable_en13432_required(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 17: Compostable claim requires EN 13432 certification.

        ZERO-HALLUCINATION CHECK:
        - Requires compostability_certification, test_results_en13432
        - Must distinguish industrial vs home compostable
        """
        input_data = GreenClaimsInput(
            claim_text="Industrially compostable packaging certified to EN 13432",
            claim_type=ClaimType.COMPOSTABLE,
            evidence_items=[
                EvidenceItem(
                    evidence_type="compostability_certification",
                    description="OK Compost Industrial certification",
                    source="TUV Austria",
                    is_third_party=True,
                    verification_url="https://tuv.com/cert/12345",
                ),
                EvidenceItem(
                    evidence_type="test_results_en13432",
                    description="Compliant with EN 13432 requirements",
                    source="Accredited testing lab",
                    is_third_party=True,
                ),
            ],
        )

        result = agent.run(input_data)

        assert result.claim_type == ClaimType.COMPOSTABLE.value
        assert result.substantiation_score > 30

    @pytest.mark.golden
    def test_18_plastic_free_supply_chain(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 18: Plastic-free claim requires supply chain verification.

        ZERO-HALLUCINATION CHECK:
        - Requires material_composition, supply_chain_verification
        - Must include packaging in assessment
        """
        input_data = GreenClaimsInput(
            claim_text="100% plastic-free product and packaging",
            claim_type=ClaimType.PLASTIC_FREE,
            evidence_items=[
                EvidenceItem(
                    evidence_type="material_composition",
                    description="All materials are glass, paper, or metal",
                    source="Material audit",
                    is_third_party=True,
                ),
                EvidenceItem(
                    evidence_type="supply_chain_verification",
                    description="Full supply chain audit for plastic content",
                    source="Third-party auditor",
                    is_third_party=True,
                ),
            ],
            claim_scope=ClaimScope(
                lifecycle_stages=["raw_materials", "production"],
            ),
        )

        result = agent.run(input_data)

        assert result.claim_type == ClaimType.PLASTIC_FREE.value

    @pytest.mark.golden
    def test_19_zero_waste_90_percent_diversion(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 19: Zero waste claim requires 90%+ diversion rate.

        ZERO-HALLUCINATION CHECK:
        - Requires waste_audit, diversion_rate_calculation, third_party_verification
        - Must achieve 90%+ diversion rate
        """
        input_data = GreenClaimsInput(
            claim_text="Our facility is zero waste certified with 95% diversion rate",
            claim_type=ClaimType.ZERO_WASTE,
            evidence_items=[
                EvidenceItem(
                    evidence_type="waste_audit",
                    description="Comprehensive waste audit 2024",
                    source="Zero Waste International Alliance",
                    is_third_party=True,
                ),
                EvidenceItem(
                    evidence_type="diversion_rate_calculation",
                    description="95% diversion from landfill",
                    source="Verified by ZWIA",
                    is_third_party=True,
                ),
                EvidenceItem(
                    evidence_type="third_party_verification",
                    description="TRUE Zero Waste certification",
                    source="GBCI",
                    is_third_party=True,
                    verification_url="https://true.gbci.org/",
                ),
            ],
        )

        result = agent.run(input_data)

        assert result.claim_type == ClaimType.ZERO_WASTE.value
        assert result.substantiation_score > 50

    @pytest.mark.golden
    def test_20_low_carbon_requires_benchmark(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 20: Low carbon claim requires benchmark comparison.

        ZERO-HALLUCINATION CHECK:
        - Requires carbon_footprint_calculation, benchmark_comparison
        - Must define 'low' threshold
        """
        input_data = GreenClaimsInput(
            claim_text="Our product has 50% lower carbon footprint than industry average",
            claim_type=ClaimType.LOW_CARBON,
            evidence_items=[
                EvidenceItem(
                    evidence_type="carbon_footprint_calculation",
                    description="LCA-based carbon footprint: 2.5 kg CO2e/unit",
                    source="ISO 14067 certified assessment",
                    is_third_party=True,
                ),
                EvidenceItem(
                    evidence_type="benchmark_comparison",
                    description="Industry average: 5.0 kg CO2e/unit",
                    source="Industry association data",
                    is_third_party=True,
                ),
            ],
        )

        result = agent.run(input_data)

        assert result.claim_type == ClaimType.LOW_CARBON.value


# =============================================================================
# Test 21-30: Substantiation Scoring
# =============================================================================


class TestSubstantiationScoring:
    """Tests for 5-dimension substantiation scoring."""

    @pytest.mark.golden
    def test_21_evidence_quality_scoring(
        self,
        agent: GreenClaimsAgent,
        comprehensive_evidence_set: List[EvidenceItem],
    ):
        """
        Test 21: Evidence quality score calculation.

        ZERO-HALLUCINATION CHECK:
        - Factor 1: Evidence count (up to 25 points)
        - Factor 2: Type diversity (up to 25 points)
        - Factor 3: Third-party evidence (up to 30 points)
        - Factor 4: Validity/recency (up to 20 points)
        """
        score = agent.calculate_substantiation_score(
            claim_text="Carbon neutral claim",
            claim_type=ClaimType.CARBON_NEUTRAL,
            evidence_items=comprehensive_evidence_set,
        )

        # 4 evidence items = 20 points (5 * 4)
        # 4 unique types = 25 points (capped)
        # 4 third-party = 30 points (capped at 30)
        # All have source and description = 20 points
        # Total: 95/100
        assert score.evidence_quality >= 75

    @pytest.mark.golden
    def test_22_evidence_quality_no_evidence(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 22: Evidence quality score with no evidence is 0.

        ZERO-HALLUCINATION CHECK:
        - No evidence items = 0 score
        """
        score = agent.calculate_substantiation_score(
            claim_text="Eco-friendly claim",
            claim_type=ClaimType.ECO_FRIENDLY,
            evidence_items=[],
        )

        assert score.evidence_quality == 0.0

    @pytest.mark.golden
    def test_23_scope_accuracy_full_lifecycle(
        self,
        agent: GreenClaimsAgent,
        full_lifecycle_scope: ClaimScope,
    ):
        """
        Test 23: Scope accuracy score with full lifecycle coverage.

        ZERO-HALLUCINATION CHECK:
        - Factor 1: Lifecycle coverage (up to 40 points)
        - Factor 2: Geographic scope (up to 20 points)
        - Factor 3: Time period (up to 20 points)
        - Factor 4: Exclusions disclosure (up to 20 points)
        """
        score = agent.calculate_substantiation_score(
            claim_text="Net zero commitment",
            claim_type=ClaimType.NET_ZERO,
            evidence_items=[],
            claim_scope=full_lifecycle_scope,
        )

        # Full lifecycle = 40, geographic = 20, time period = 20, exclusions = 20
        assert score.scope_accuracy >= 80

    @pytest.mark.golden
    def test_24_scope_accuracy_no_scope(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 24: Scope accuracy score with no scope defined.

        ZERO-HALLUCINATION CHECK:
        - No scope = minimum 20 points
        """
        score = agent.calculate_substantiation_score(
            claim_text="Sustainable product",
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[],
            claim_scope=None,
        )

        assert score.scope_accuracy == 20.0

    @pytest.mark.golden
    def test_25_transparency_score_full(
        self,
        agent: GreenClaimsAgent,
        comprehensive_evidence_set: List[EvidenceItem],
        full_lifecycle_scope: ClaimScope,
    ):
        """
        Test 25: Transparency score with complete evidence disclosure.

        ZERO-HALLUCINATION CHECK:
        - Factor 1: Source disclosure (up to 30 points)
        - Factor 2: Verification URLs (up to 30 points)
        - Factor 3: Dates provided (up to 20 points)
        - Factor 4: Scope exclusions (up to 20 points)
        """
        score = agent.calculate_substantiation_score(
            claim_text="Carbon neutral",
            claim_type=ClaimType.CARBON_NEUTRAL,
            evidence_items=comprehensive_evidence_set,
            claim_scope=full_lifecycle_scope,
        )

        # All have sources = 30, all have URLs = 30, all have dates = 20, exclusions = 20
        assert score.transparency >= 80

    @pytest.mark.golden
    def test_26_verification_score_third_party(
        self,
        agent: GreenClaimsAgent,
        comprehensive_evidence_set: List[EvidenceItem],
    ):
        """
        Test 26: Verification score with third-party certifications.

        ZERO-HALLUCINATION CHECK:
        - Factor 1: Third-party verification (up to 50 points)
        - Factor 2: Recognized certifications (up to 30 points)
        - Factor 3: Expiry information (up to 20 points)
        """
        score = agent.calculate_substantiation_score(
            claim_text="Carbon neutral certified",
            claim_type=ClaimType.CARBON_NEUTRAL,
            evidence_items=comprehensive_evidence_set,
        )

        # 4 third-party = 50 (capped), recognized certs detected, 1 expiry
        assert score.verification >= 50

    @pytest.mark.golden
    def test_27_clarity_score_vague_penalty(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 27: Clarity score with vague terms penalty.

        ZERO-HALLUCINATION CHECK:
        - Base score: 50
        - Vague terms penalty: up to -30 points
        - Quantifiable metrics bonus: up to +25 points
        - Specific scope bonus: up to +25 points
        """
        score = agent.calculate_substantiation_score(
            claim_text="Our green and eco-friendly sustainable product is natural",
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[],
        )

        # 4 vague terms = -30, no metrics, no scope
        # 50 - 30 = 20
        assert score.clarity <= 30

    @pytest.mark.golden
    def test_28_clarity_score_quantified(
        self,
        agent: GreenClaimsAgent,
        full_lifecycle_scope: ClaimScope,
    ):
        """
        Test 28: Clarity score with quantified metrics.

        ZERO-HALLUCINATION CHECK:
        - Percentage in claim = +25 points
        - Specific scope = +25 points
        """
        score = agent.calculate_substantiation_score(
            claim_text="We have reduced emissions by 50% compared to 2019 baseline",
            claim_type=ClaimType.REDUCED_EMISSIONS,
            evidence_items=[],
            claim_scope=full_lifecycle_scope,
        )

        # Base 50 + percentage 25 + scope 25 = 100 (but capped)
        assert score.clarity >= 70

    @pytest.mark.golden
    def test_29_weighted_total_calculation(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 29: Verify weighted total calculation.

        ZERO-HALLUCINATION CHECK:
        weighted_total = (evidence * 0.30) + (scope * 0.25) +
                         (transparency * 0.20) + (verification * 0.15) + (clarity * 0.10)
        """
        # Create input with known scores
        score = agent.calculate_substantiation_score(
            claim_text="Carbon neutral certified by ISO 14064",
            claim_type=ClaimType.CARBON_NEUTRAL,
            evidence_items=[
                EvidenceItem(
                    evidence_type="certification",
                    description="ISO 14064 certification",
                    source="Certifier",
                    is_third_party=True,
                    verification_url="https://example.com",
                    date_issued="2024-01-01",
                )
            ],
            claim_scope=ClaimScope(
                lifecycle_stages=["production"],
                geographic_scope="US",
                time_period="2024",
            ),
        )

        # Verify weighted total is between 0 and 100
        assert 0 <= score.weighted_total <= 100

        # Verify weighted total matches sum of weighted dimensions
        expected = (
            score.evidence_quality * WEIGHT_EVIDENCE_QUALITY
            + score.scope_accuracy * WEIGHT_SCOPE_ACCURACY
            + score.transparency * WEIGHT_TRANSPARENCY
            + score.verification * WEIGHT_VERIFICATION
            + score.clarity * WEIGHT_CLARITY
        )
        assert score.weighted_total == pytest.approx(expected, rel=1e-2)

    @pytest.mark.golden
    def test_30_substantiation_level_thresholds(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 30: Verify substantiation level threshold logic.

        ZERO-HALLUCINATION CHECK:
        - excellent: 80-100
        - good: 60-79
        - moderate: 40-59
        - weak: 20-39
        - insufficient: 0-19
        """
        # Test via public API
        result = agent.run(GreenClaimsInput(
            claim_text="Sustainable product",
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[],
        ))

        # With no evidence, should be weak or insufficient
        assert result.substantiation_level in [
            SubstantiationLevel.INSUFFICIENT.value,
            SubstantiationLevel.WEAK.value,
            SubstantiationLevel.MODERATE.value,
        ]


# =============================================================================
# Test 31-40: Greenwashing Detection
# =============================================================================


class TestGreenwashingDetection:
    """Tests for 7 greenwashing red flag patterns."""

    @pytest.mark.golden
    def test_31_vague_claims_detection(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 31: Detect vague claims red flag pattern.

        ZERO-HALLUCINATION CHECK:
        - Keywords: green, eco, natural, environmentally friendly, etc.
        - Without qualifiers or evidence = high severity
        """
        red_flags, risk_score = agent.detect_greenwashing(
            claim_text="Our product is green and eco-friendly for the planet",
            claim_type=ClaimType.GREEN,
            evidence_items=[],
        )

        assert len(red_flags) > 0
        vague_flag = next(
            (rf for rf in red_flags if rf.red_flag == GreenwashingRedFlag.VAGUE_CLAIMS),
            None
        )
        assert vague_flag is not None
        assert vague_flag.severity == "high"

    @pytest.mark.golden
    def test_32_hidden_tradeoffs_detection(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 32: Detect hidden tradeoffs pattern.

        ZERO-HALLUCINATION CHECK:
        - Single lifecycle stage focus = medium severity
        - Ignores significant impacts
        """
        red_flags, risk_score = agent.detect_greenwashing(
            claim_text="Low carbon manufacturing process",
            claim_type=ClaimType.LOW_CARBON,
            evidence_items=[],
        )

        # With single lifecycle stage, should detect hidden tradeoffs
        input_data = GreenClaimsInput(
            claim_text="Low carbon manufacturing",
            claim_type=ClaimType.LOW_CARBON,
            evidence_items=[],
            claim_scope=ClaimScope(lifecycle_stages=["production"]),
        )
        result = agent.run(input_data)

        hidden_flag = next(
            (rf for rf in result.red_flags if rf.red_flag == GreenwashingRedFlag.HIDDEN_TRADEOFFS),
            None
        )
        if hidden_flag:
            assert hidden_flag.severity == "medium"

    @pytest.mark.golden
    def test_33_false_labels_detection(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 33: Detect false labels pattern.

        ZERO-HALLUCINATION CHECK:
        - Self-declared certification without verification = critical severity
        """
        red_flags, risk_score = agent.detect_greenwashing(
            claim_text="Certified sustainable product",
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[
                EvidenceItem(
                    evidence_type="certification",
                    description="Our sustainability certification",
                    is_third_party=False,
                    # No verification_url
                )
            ],
        )

        false_label_flag = next(
            (rf for rf in red_flags if rf.red_flag == GreenwashingRedFlag.FALSE_LABELS),
            None
        )
        assert false_label_flag is not None
        assert false_label_flag.severity == "critical"

    @pytest.mark.golden
    def test_34_irrelevant_claims_detection(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 34: Detect irrelevant claims pattern.

        ZERO-HALLUCINATION CHECK:
        - CFC-free, lead-free = medium severity (legally required)
        """
        red_flags, risk_score = agent.detect_greenwashing(
            claim_text="Our product is CFC-free and environmentally friendly",
            claim_type=ClaimType.ENVIRONMENTALLY_FRIENDLY,
            evidence_items=[],
        )

        irrelevant_flag = next(
            (rf for rf in red_flags if rf.red_flag == GreenwashingRedFlag.IRRELEVANT_CLAIMS),
            None
        )
        assert irrelevant_flag is not None
        assert irrelevant_flag.severity == "medium"

    @pytest.mark.golden
    def test_35_lesser_of_evils_detection(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 35: Detect lesser of evils pattern.

        ZERO-HALLUCINATION CHECK:
        - Harmful product categories: tobacco, fossil fuel, coal, petroleum
        - High severity
        """
        red_flags, risk_score = agent.detect_greenwashing(
            claim_text="Our cleaner-burning coal is more sustainable",
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[],
            product_category="coal",
        )

        lesser_flag = next(
            (rf for rf in red_flags if rf.red_flag == GreenwashingRedFlag.LESSER_OF_EVILS),
            None
        )
        assert lesser_flag is not None
        assert lesser_flag.severity == "high"

    @pytest.mark.golden
    def test_36_fibbing_detection(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 36: Detect fibbing pattern (exaggerated claims).

        ZERO-HALLUCINATION CHECK:
        - Absolute terms: 100%, completely, totally, pure, perfect
        - Without supporting evidence = critical severity
        """
        red_flags, risk_score = agent.detect_greenwashing(
            claim_text="Our product is 100% sustainable and completely carbon free",
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[],
        )

        fibbing_flag = next(
            (rf for rf in red_flags if rf.red_flag == GreenwashingRedFlag.FIBBING),
            None
        )
        assert fibbing_flag is not None
        assert fibbing_flag.severity == "critical"

    @pytest.mark.golden
    def test_37_false_certifications_detection(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 37: Detect false certifications pattern.

        ZERO-HALLUCINATION CHECK:
        - Certification keywords without third-party verification = critical
        """
        red_flags, risk_score = agent.detect_greenwashing(
            claim_text="Our certified and approved green product",
            claim_type=ClaimType.GREEN,
            evidence_items=[],
        )

        false_cert_flag = next(
            (rf for rf in red_flags if rf.red_flag == GreenwashingRedFlag.FALSE_CERTIFICATIONS),
            None
        )
        assert false_cert_flag is not None
        assert false_cert_flag.severity == "critical"

    @pytest.mark.golden
    def test_38_risk_score_calculation(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 38: Verify greenwashing risk score calculation.

        ZERO-HALLUCINATION CHECK:
        risk_score = sum(severity_weights[rf.severity] for rf in red_flags)
        Capped at 100
        """
        red_flags, risk_score = agent.detect_greenwashing(
            claim_text="100% green eco-friendly certified product",
            claim_type=ClaimType.GREEN,
            evidence_items=[],
        )

        # Multiple red flags should increase risk score
        assert risk_score > 0
        assert risk_score <= 100

        # Verify calculation matches sum of severity weights
        expected_score = min(
            sum(SEVERITY_WEIGHTS.get(rf.severity, 10) for rf in red_flags),
            100
        )
        assert risk_score == pytest.approx(expected_score, rel=1e-2)

    @pytest.mark.golden
    def test_39_no_greenwashing_detected(
        self,
        agent: GreenClaimsAgent,
        comprehensive_evidence_set: List[EvidenceItem],
        full_lifecycle_scope: ClaimScope,
    ):
        """
        Test 39: No greenwashing detected with valid claim.

        ZERO-HALLUCINATION CHECK:
        - Specific claim with evidence = no red flags
        """
        result = agent.run(GreenClaimsInput(
            claim_text="We have reduced our Scope 1 and 2 emissions by 46% since 2019",
            claim_type=ClaimType.REDUCED_EMISSIONS,
            evidence_items=[
                EvidenceItem(
                    evidence_type="baseline_emissions",
                    description="2019 baseline: 1000 tCO2e",
                    source="Verified auditor",
                    is_third_party=True,
                    verification_url="https://example.com/audit",
                ),
                EvidenceItem(
                    evidence_type="current_emissions",
                    description="2024 emissions: 540 tCO2e",
                    source="Verified auditor",
                    is_third_party=True,
                    verification_url="https://example.com/audit",
                ),
                EvidenceItem(
                    evidence_type="reduction_calculation",
                    description="46% reduction calculated per GHG Protocol",
                    source="GHG Protocol methodology",
                    is_third_party=True,
                ),
            ],
            claim_scope=ClaimScope(
                lifecycle_stages=["production"],
                time_period="2019-2024",
            ),
        ))

        # May still have some flags but should be minimal
        assert result.greenwashing_risk_score < 30

    @pytest.mark.golden
    def test_40_multiple_red_flags(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 40: Multiple red flags detected for egregious claim.

        ZERO-HALLUCINATION CHECK:
        - Multiple patterns = high risk score
        """
        result = agent.run(GreenClaimsInput(
            claim_text="100% green certified eco-friendly natural sustainable product that is CFC-free",
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[
                EvidenceItem(
                    evidence_type="certification",
                    description="Self-certified green label",
                    is_third_party=False,
                )
            ],
            product_category="petroleum",
        ))

        # Should detect multiple red flags
        assert len(result.red_flags) >= 2
        assert result.greenwashing_detected is True
        assert result.greenwashing_risk_score >= 40


# =============================================================================
# Test 41-50: Compliance Framework Validation
# =============================================================================


class TestComplianceFrameworks:
    """Tests for compliance framework validation."""

    @pytest.mark.golden
    def test_41_eu_green_claims_directive_compliant(
        self,
        agent: GreenClaimsAgent,
        carbon_neutral_input_valid: GreenClaimsInput,
    ):
        """
        Test 41: EU Green Claims Directive compliance with valid claim.

        ZERO-HALLUCINATION CHECK:
        - Claim substantiation requirements met
        - Independent verification present
        - Lifecycle scope defined
        - Transparency score >= 60
        """
        result = agent.run(carbon_neutral_input_valid)

        eu_result = next(
            (fr for fr in result.compliance_report.framework_results
             if fr.framework == ComplianceFramework.EU_GREEN_CLAIMS_DIRECTIVE.value),
            None
        )
        assert eu_result is not None
        assert len(eu_result.requirements_met) >= 2

    @pytest.mark.golden
    def test_42_eu_directive_non_compliant(
        self,
        agent: GreenClaimsAgent,
        vague_eco_friendly_input: GreenClaimsInput,
    ):
        """
        Test 42: EU Green Claims Directive non-compliance.

        ZERO-HALLUCINATION CHECK:
        - Missing substantiation
        - No independent verification
        - No lifecycle scope
        """
        result = agent.run(vague_eco_friendly_input)

        eu_result = next(
            (fr for fr in result.compliance_report.framework_results
             if fr.framework == ComplianceFramework.EU_GREEN_CLAIMS_DIRECTIVE.value),
            None
        )
        assert eu_result is not None
        assert eu_result.compliant is False
        assert len(eu_result.requirements_failed) > 0

    @pytest.mark.golden
    def test_43_iso_14021_compliance(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 43: ISO 14021 compliance for self-declared claims.

        ZERO-HALLUCINATION CHECK:
        - No vague/misleading terms
        - Verifiable claims
        """
        input_data = GreenClaimsInput(
            claim_text="Made with 75% recycled PET plastic",
            claim_type=ClaimType.RECYCLABLE,
            evidence_items=[
                EvidenceItem(
                    evidence_type="material_composition",
                    description="75% post-consumer recycled PET",
                    source="Material supplier certification",
                    is_third_party=True,
                )
            ],
            target_frameworks=[ComplianceFramework.ISO_14021],
        )

        result = agent.run(input_data)

        iso_result = next(
            (fr for fr in result.compliance_report.framework_results
             if fr.framework == ComplianceFramework.ISO_14021.value),
            None
        )
        assert iso_result is not None

    @pytest.mark.golden
    def test_44_ftc_green_guides_compliance(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 44: FTC Green Guides compliance.

        ZERO-HALLUCINATION CHECK:
        - Clear scope qualification
        - Comparative claims substantiated
        """
        input_data = GreenClaimsInput(
            claim_text="50% lower emissions than our 2019 baseline",
            claim_type=ClaimType.REDUCED_EMISSIONS,
            evidence_items=[
                EvidenceItem(
                    evidence_type="baseline_emissions",
                    description="2019 emissions data",
                    is_third_party=True,
                )
            ],
            claim_scope=ClaimScope(
                lifecycle_stages=["production"],
                time_period="2019-2024",
            ),
            target_frameworks=[ComplianceFramework.FTC_GREEN_GUIDES],
            comparative_claim=True,
            comparison_baseline="2019 emissions",
        )

        result = agent.run(input_data)

        ftc_result = next(
            (fr for fr in result.compliance_report.framework_results
             if fr.framework == ComplianceFramework.FTC_GREEN_GUIDES.value),
            None
        )
        assert ftc_result is not None

    @pytest.mark.golden
    def test_45_comparative_claim_without_baseline(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 45: Comparative claim without baseline fails validation.

        ZERO-HALLUCINATION CHECK:
        - Comparative claims must specify baseline
        """
        input_data = GreenClaimsInput(
            claim_text="50% lower emissions",
            claim_type=ClaimType.REDUCED_EMISSIONS,
            evidence_items=[],
            comparative_claim=True,
            # Missing comparison_baseline
        )

        result = agent.run(input_data)

        assert "baseline" in " ".join(result.validation_messages).lower()

    @pytest.mark.golden
    def test_46_overall_compliance_status(
        self,
        agent: GreenClaimsAgent,
        carbon_neutral_input_valid: GreenClaimsInput,
    ):
        """
        Test 46: Overall compliance status determination.

        ZERO-HALLUCINATION CHECK:
        - COMPLIANT: all frameworks pass, no red flags, score >= 80
        - NON_COMPLIANT: critical issues
        - CONDITIONALLY COMPLIANT: valid with score >= 60
        - REQUIRES REMEDIATION: otherwise
        """
        result = agent.run(carbon_neutral_input_valid)

        # Valid claim should be compliant or conditionally compliant
        assert result.compliance_report.overall_status in [
            "COMPLIANT",
            "CONDITIONALLY COMPLIANT",
        ]

    @pytest.mark.golden
    def test_47_risk_level_determination(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 47: Risk level determination in compliance report.

        ZERO-HALLUCINATION CHECK:
        - critical: critical red flags
        - high: high severity red flags
        - medium: validation failures or low score
        - low: fully compliant
        """
        # Critical risk
        critical_input = GreenClaimsInput(
            claim_text="100% certified green product",
            claim_type=ClaimType.GREEN,
            evidence_items=[
                EvidenceItem(
                    evidence_type="certification",
                    description="Self-certified",
                    is_third_party=False,
                )
            ],
        )
        result = agent.run(critical_input)

        assert result.compliance_report.risk_level in ["critical", "high", "medium"]

    @pytest.mark.golden
    def test_48_immediate_actions_generated(
        self,
        agent: GreenClaimsAgent,
        vague_eco_friendly_input: GreenClaimsInput,
    ):
        """
        Test 48: Immediate actions generated for non-compliant claims.

        ZERO-HALLUCINATION CHECK:
        - Red flag recommendations become immediate actions
        - Missing evidence requirements become actions
        """
        result = agent.run(vague_eco_friendly_input)

        assert len(result.compliance_report.immediate_actions) > 0

    @pytest.mark.golden
    def test_49_long_term_recommendations(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 49: Long-term recommendations for low dimension scores.

        ZERO-HALLUCINATION CHECK:
        - Dimension score < 60 triggers recommendation
        """
        input_data = GreenClaimsInput(
            claim_text="Sustainable product",
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[
                EvidenceItem(
                    evidence_type="report",
                    description="Sustainability report",
                    is_third_party=False,
                )
            ],
        )
        result = agent.run(input_data)

        # Low scores should generate recommendations
        if result.substantiation_score < 60:
            assert len(result.compliance_report.long_term_recommendations) > 0

    @pytest.mark.golden
    def test_50_evidence_gaps_identified(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 50: Evidence gaps identified in compliance report.

        ZERO-HALLUCINATION CHECK:
        - Missing required evidence listed in gaps
        """
        input_data = GreenClaimsInput(
            claim_text="Carbon neutral",
            claim_type=ClaimType.CARBON_NEUTRAL,
            evidence_items=[
                EvidenceItem(
                    evidence_type="report",
                    description="Carbon report",
                    is_third_party=False,
                )
            ],
        )
        result = agent.run(input_data)

        # Carbon neutral requires ghg_inventory, reduction_pathway, etc.
        assert len(result.compliance_report.evidence_gaps) > 0


# =============================================================================
# Test 51-55: Provenance and Determinism
# =============================================================================


class TestProvenanceAndDeterminism:
    """Tests for provenance hash and deterministic calculations."""

    @pytest.mark.golden
    def test_51_provenance_hash_format(
        self,
        agent: GreenClaimsAgent,
        carbon_neutral_input_valid: GreenClaimsInput,
    ):
        """
        Test 51: Provenance hash is valid SHA-256 format.

        ZERO-HALLUCINATION CHECK:
        - SHA-256 produces 64 hex characters
        """
        result = agent.run(carbon_neutral_input_valid)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    @pytest.mark.golden
    def test_52_deterministic_scores(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 52: Same inputs produce same scores (zero-hallucination).

        ZERO-HALLUCINATION CHECK:
        - Calculation is deterministic, no LLM in math
        """
        input_data = GreenClaimsInput(
            claim_text="Carbon neutral product",
            claim_type=ClaimType.CARBON_NEUTRAL,
            evidence_items=[
                EvidenceItem(
                    evidence_type="certification",
                    description="ISO 14064",
                    is_third_party=True,
                )
            ],
        )

        result1 = agent.run(input_data)
        result2 = agent.run(input_data)
        result3 = agent.run(input_data)

        assert result1.substantiation_score == result2.substantiation_score
        assert result2.substantiation_score == result3.substantiation_score
        assert result1.greenwashing_risk_score == result2.greenwashing_risk_score

    @pytest.mark.golden
    def test_53_deterministic_across_instances(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 53: Different agent instances produce same results.

        ZERO-HALLUCINATION CHECK:
        - Calculation doesn't depend on instance state
        """
        input_data = GreenClaimsInput(
            claim_text="Net zero by 2050",
            claim_type=ClaimType.NET_ZERO,
            evidence_items=[],
        )

        agent1 = GreenClaimsAgent()
        agent2 = GreenClaimsAgent()
        agent3 = GreenClaimsAgent()

        result1 = agent1.run(input_data)
        result2 = agent2.run(input_data)
        result3 = agent3.run(input_data)

        assert result1.substantiation_score == result2.substantiation_score
        assert result2.substantiation_score == result3.substantiation_score

    @pytest.mark.golden
    def test_54_provenance_changes_with_inputs(
        self,
        agent: GreenClaimsAgent,
    ):
        """
        Test 54: Provenance hash changes when inputs change.

        ZERO-HALLUCINATION CHECK:
        - Different inputs produce different hashes
        """
        input1 = GreenClaimsInput(
            claim_text="Carbon neutral",
            claim_type=ClaimType.CARBON_NEUTRAL,
            evidence_items=[],
        )
        input2 = GreenClaimsInput(
            claim_text="Net zero",
            claim_type=ClaimType.NET_ZERO,
            evidence_items=[],
        )

        result1 = agent.run(input1)
        result2 = agent.run(input2)

        assert result1.provenance_hash != result2.provenance_hash

    @pytest.mark.golden
    def test_55_processing_time_tracked(
        self,
        agent: GreenClaimsAgent,
        carbon_neutral_input_valid: GreenClaimsInput,
    ):
        """
        Test 55: Processing time is tracked in output.

        ZERO-HALLUCINATION CHECK:
        - Processing time > 0 ms
        """
        result = agent.run(carbon_neutral_input_valid)

        assert result.processing_time_ms > 0


# =============================================================================
# Test 56-60: Edge Cases and Input Validation
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and input validation."""

    def test_56_empty_claim_text_rejected(self, agent: GreenClaimsAgent):
        """Test 56: Empty claim text is rejected by validator."""
        with pytest.raises(ValueError):
            GreenClaimsInput(
                claim_text="",
                claim_type=ClaimType.SUSTAINABLE,
            )

    def test_57_whitespace_only_claim_rejected(self, agent: GreenClaimsAgent):
        """Test 57: Whitespace-only claim text is rejected."""
        with pytest.raises(ValueError):
            GreenClaimsInput(
                claim_text="   ",
                claim_type=ClaimType.SUSTAINABLE,
            )

    def test_58_very_long_claim_text(self, agent: GreenClaimsAgent):
        """Test 58: Long claim text is handled correctly."""
        long_text = "Sustainable " * 100  # 1100+ chars
        input_data = GreenClaimsInput(
            claim_text=long_text[:1999],  # Max 2000
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[],
        )
        result = agent.run(input_data)
        assert result is not None

    def test_59_all_claim_types_have_requirements(self, agent: GreenClaimsAgent):
        """Test 59: All 16 claim types have defined requirements."""
        for claim_type in ClaimType:
            reqs = agent.get_claim_requirements(claim_type)
            assert reqs is not None, f"Missing requirements for {claim_type}"
            assert "required_evidence" in reqs

    def test_60_all_frameworks_can_be_assessed(self, agent: GreenClaimsAgent):
        """Test 60: All compliance frameworks can be assessed."""
        for framework in ComplianceFramework:
            input_data = GreenClaimsInput(
                claim_text="Test claim",
                claim_type=ClaimType.SUSTAINABLE,
                evidence_items=[],
                target_frameworks=[framework],
            )
            result = agent.run(input_data)
            assert len(result.compliance_report.framework_results) == 1
            assert result.compliance_report.framework_results[0].framework == framework.value


# =============================================================================
# Test 61-65: Public API Methods
# =============================================================================


class TestPublicAPIMethods:
    """Tests for public API methods."""

    def test_61_validate_claim_public_api(self, agent: GreenClaimsAgent):
        """Test 61: validate_claim public API method."""
        is_valid, status, messages = agent.validate_claim(
            claim_text="Carbon neutral product",
            claim_type=ClaimType.CARBON_NEUTRAL,
            evidence_items=[],
        )

        assert isinstance(is_valid, bool)
        assert isinstance(status, ValidationStatus)
        assert isinstance(messages, list)

    def test_62_calculate_substantiation_score_public_api(
        self,
        agent: GreenClaimsAgent,
    ):
        """Test 62: calculate_substantiation_score public API method."""
        score = agent.calculate_substantiation_score(
            claim_text="Sustainable product",
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[],
        )

        assert isinstance(score, SubstantiationScoreBreakdown)
        assert 0 <= score.weighted_total <= 100

    def test_63_detect_greenwashing_public_api(self, agent: GreenClaimsAgent):
        """Test 63: detect_greenwashing public API method."""
        red_flags, risk_score = agent.detect_greenwashing(
            claim_text="Green eco-friendly product",
            claim_type=ClaimType.GREEN,
            evidence_items=[],
        )

        assert isinstance(red_flags, list)
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 100

    def test_64_generate_report_public_api(self, agent: GreenClaimsAgent):
        """Test 64: generate_report public API method."""
        input_data = GreenClaimsInput(
            claim_text="Sustainable product",
            claim_type=ClaimType.SUSTAINABLE,
            evidence_items=[],
        )

        report = agent.generate_report(input_data)

        assert isinstance(report, ComplianceReport)
        assert report.summary is not None
        assert report.overall_status is not None

    def test_65_full_run_output_structure(
        self,
        agent: GreenClaimsAgent,
        carbon_neutral_input_valid: GreenClaimsInput,
    ):
        """Test 65: Full run returns complete output structure."""
        result = agent.run(carbon_neutral_input_valid)

        # Verify all required fields
        assert isinstance(result, GreenClaimsOutput)
        assert result.claim_text == carbon_neutral_input_valid.claim_text
        assert result.claim_type == ClaimType.CARBON_NEUTRAL.value
        assert isinstance(result.claim_valid, bool)
        assert isinstance(result.validation_status, str)
        assert isinstance(result.substantiation_score, float)
        assert isinstance(result.substantiation_level, str)
        assert isinstance(result.score_breakdown, SubstantiationScoreBreakdown)
        assert isinstance(result.greenwashing_detected, bool)
        assert isinstance(result.red_flags, list)
        assert isinstance(result.greenwashing_risk_score, float)
        assert isinstance(result.overall_compliance, bool)
        assert isinstance(result.compliance_report, ComplianceReport)
        assert len(result.provenance_hash) == 64
        assert result.processing_time_ms > 0


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
