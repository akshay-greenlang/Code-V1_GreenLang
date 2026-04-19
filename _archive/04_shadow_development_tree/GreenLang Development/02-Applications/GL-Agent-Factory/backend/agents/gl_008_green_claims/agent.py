"""
GL-008: Green Claims Verification Agent

This module implements the Green Claims Verification Agent that validates
environmental claims against the EU Green Claims Directive (2023/0085) and
detects potential greenwashing patterns.

The agent supports:
- 16 claim types (carbon_neutral, net_zero, eco_friendly, etc.)
- 5-dimension substantiation scoring (evidence, scope, transparency, verification, clarity)
- 7 greenwashing red flag patterns
- EU Green Claims Directive compliance validation
- Complete SHA-256 provenance tracking

Example:
    >>> agent = GreenClaimsAgent()
    >>> result = agent.run(GreenClaimsInput(
    ...     claim_text="Our product is carbon neutral",
    ...     claim_type=ClaimType.CARBON_NEUTRAL,
    ...     evidence_provided=["ISO 14064 certification", "Third-party audit report"],
    ...     product_category="consumer_goods"
    ... ))
    >>> print(f"Claim valid: {result.data.claim_valid}")
    >>> print(f"Substantiation score: {result.data.substantiation_score}")
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class ClaimType(str, Enum):
    """
    Supported green claim types.

    These 16 claim types cover the most common environmental marketing claims
    and are validated against EU Green Claims Directive requirements.
    """

    CARBON_NEUTRAL = "carbon_neutral"
    CLIMATE_POSITIVE = "climate_positive"
    NET_ZERO = "net_zero"
    CARBON_NEGATIVE = "carbon_negative"
    ECO_FRIENDLY = "eco_friendly"
    SUSTAINABLE = "sustainable"
    GREEN = "green"
    RENEWABLE = "renewable"
    RECYCLABLE = "recyclable"
    BIODEGRADABLE = "biodegradable"
    COMPOSTABLE = "compostable"
    PLASTIC_FREE = "plastic_free"
    ZERO_WASTE = "zero_waste"
    LOW_CARBON = "low_carbon"
    REDUCED_EMISSIONS = "reduced_emissions"
    ENVIRONMENTALLY_FRIENDLY = "environmentally_friendly"


class GreenwashingRedFlag(str, Enum):
    """
    Greenwashing red flag patterns based on TerraChoice Seven Sins.

    These patterns indicate potential greenwashing in environmental claims.
    """

    VAGUE_CLAIMS = "vague_claims"
    HIDDEN_TRADEOFFS = "hidden_tradeoffs"
    FALSE_LABELS = "false_labels"
    IRRELEVANT_CLAIMS = "irrelevant_claims"
    LESSER_OF_EVILS = "lesser_of_evils"
    FIBBING = "fibbing"
    FALSE_CERTIFICATIONS = "false_certifications"


class ValidationStatus(str, Enum):
    """Claim validation status."""

    VALID = "valid"
    INVALID = "invalid"
    REQUIRES_VERIFICATION = "requires_verification"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    POTENTIAL_GREENWASHING = "potential_greenwashing"


class SubstantiationLevel(str, Enum):
    """Substantiation level based on score."""

    EXCELLENT = "excellent"  # 80-100
    GOOD = "good"  # 60-79
    MODERATE = "moderate"  # 40-59
    WEAK = "weak"  # 20-39
    INSUFFICIENT = "insufficient"  # 0-19


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks for green claims."""

    EU_GREEN_CLAIMS_DIRECTIVE = "eu_green_claims_directive"
    ISO_14021 = "iso_14021"
    ISO_14024 = "iso_14024"
    ISO_14025 = "iso_14025"
    FTC_GREEN_GUIDES = "ftc_green_guides"
    ASA_UK = "asa_uk"


# =============================================================================
# Pydantic Models
# =============================================================================


class EvidenceItem(BaseModel):
    """Evidence item supporting a green claim."""

    evidence_type: str = Field(
        ...,
        description="Type of evidence (certification, audit, report, measurement)"
    )
    description: str = Field(..., description="Description of evidence")
    source: Optional[str] = Field(None, description="Source or issuer of evidence")
    date_issued: Optional[str] = Field(None, description="Date evidence was issued")
    expiry_date: Optional[str] = Field(None, description="Expiry date if applicable")
    verification_url: Optional[str] = Field(None, description="URL for verification")
    is_third_party: bool = Field(False, description="Whether evidence is third-party verified")


class ClaimScope(BaseModel):
    """Scope definition for a green claim."""

    product_scope: Optional[str] = Field(None, description="Product or product line")
    lifecycle_stages: List[str] = Field(
        default_factory=list,
        description="Lifecycle stages covered (raw_materials, production, use, disposal)"
    )
    geographic_scope: Optional[str] = Field(None, description="Geographic coverage")
    time_period: Optional[str] = Field(None, description="Time period the claim covers")
    exclusions: List[str] = Field(default_factory=list, description="Excluded scopes")


class GreenClaimsInput(BaseModel):
    """
    Input model for Green Claims Verification Agent.

    Attributes:
        claim_text: The actual claim text as made
        claim_type: Type of environmental claim
        evidence_items: List of supporting evidence
        claim_scope: Scope of the claim
        company_name: Name of company making the claim
        product_category: Product category
        target_frameworks: Compliance frameworks to validate against
    """

    claim_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The environmental claim text"
    )
    claim_type: ClaimType = Field(..., description="Type of environmental claim")
    evidence_items: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence supporting the claim"
    )
    claim_scope: Optional[ClaimScope] = Field(
        None,
        description="Scope of the claim"
    )
    company_name: Optional[str] = Field(None, description="Company making the claim")
    product_category: Optional[str] = Field(None, description="Product category")
    target_frameworks: List[ComplianceFramework] = Field(
        default_factory=lambda: [ComplianceFramework.EU_GREEN_CLAIMS_DIRECTIVE],
        description="Compliance frameworks to validate against"
    )
    comparative_claim: bool = Field(
        False,
        description="Whether this is a comparative claim"
    )
    comparison_baseline: Optional[str] = Field(
        None,
        description="Baseline for comparison if comparative claim"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("claim_text")
    def validate_claim_text(cls, v: str) -> str:
        """Validate claim text is not empty or just whitespace."""
        v = v.strip()
        if not v:
            raise ValueError("Claim text cannot be empty")
        return v

    @validator("comparison_baseline", always=True)
    def validate_comparison_baseline(cls, v: Optional[str], values: Dict) -> Optional[str]:
        """Ensure comparison baseline is provided for comparative claims."""
        if values.get("comparative_claim", False) and not v:
            logger.warning("Comparative claim without baseline provided")
        return v


class SubstantiationScoreBreakdown(BaseModel):
    """Breakdown of substantiation score by dimension."""

    evidence_quality: float = Field(
        ...,
        ge=0,
        le=100,
        description="Evidence quality score (30% weight)"
    )
    scope_accuracy: float = Field(
        ...,
        ge=0,
        le=100,
        description="Scope accuracy score (25% weight)"
    )
    transparency: float = Field(
        ...,
        ge=0,
        le=100,
        description="Transparency score (20% weight)"
    )
    verification: float = Field(
        ...,
        ge=0,
        le=100,
        description="Verification score (15% weight)"
    )
    clarity: float = Field(
        ...,
        ge=0,
        le=100,
        description="Clarity score (10% weight)"
    )
    weighted_total: float = Field(
        ...,
        ge=0,
        le=100,
        description="Weighted total score"
    )


class GreenwashingDetection(BaseModel):
    """Greenwashing detection result."""

    red_flag: GreenwashingRedFlag = Field(..., description="Type of red flag detected")
    severity: str = Field(..., description="Severity: low, medium, high, critical")
    description: str = Field(..., description="Description of the issue")
    evidence: List[str] = Field(default_factory=list, description="Evidence of the issue")
    recommendation: str = Field(..., description="Recommendation to address the issue")


class FrameworkComplianceResult(BaseModel):
    """Compliance result for a specific framework."""

    framework: str = Field(..., description="Framework assessed")
    compliant: bool = Field(..., description="Whether claim is compliant")
    requirements_met: List[str] = Field(
        default_factory=list,
        description="Requirements that are met"
    )
    requirements_failed: List[str] = Field(
        default_factory=list,
        description="Requirements that failed"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for compliance"
    )


class ComplianceReport(BaseModel):
    """Full compliance report with recommendations."""

    summary: str = Field(..., description="Executive summary")
    overall_status: str = Field(..., description="Overall compliance status")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    framework_results: List[FrameworkComplianceResult] = Field(
        default_factory=list,
        description="Results by framework"
    )
    immediate_actions: List[str] = Field(
        default_factory=list,
        description="Immediate actions required"
    )
    long_term_recommendations: List[str] = Field(
        default_factory=list,
        description="Long-term recommendations"
    )
    evidence_gaps: List[str] = Field(
        default_factory=list,
        description="Missing evidence"
    )


class GreenClaimsOutput(BaseModel):
    """
    Output model for Green Claims Verification Agent.

    Comprehensive assessment of green claim validity and compliance.
    """

    # Claim identification
    claim_text: str = Field(..., description="Original claim text")
    claim_type: str = Field(..., description="Claim type assessed")

    # Validation results
    claim_valid: bool = Field(..., description="Whether claim is valid")
    validation_status: str = Field(..., description="Detailed validation status")
    validation_messages: List[str] = Field(
        default_factory=list,
        description="Validation messages"
    )

    # Substantiation scoring
    substantiation_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall substantiation score (0-100)"
    )
    substantiation_level: str = Field(
        ...,
        description="Substantiation level"
    )
    score_breakdown: SubstantiationScoreBreakdown = Field(
        ...,
        description="Score breakdown by dimension"
    )

    # Greenwashing detection
    greenwashing_detected: bool = Field(
        ...,
        description="Whether greenwashing patterns detected"
    )
    red_flags: List[GreenwashingDetection] = Field(
        default_factory=list,
        description="Detected greenwashing red flags"
    )
    greenwashing_risk_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Greenwashing risk score (0-100)"
    )

    # Compliance
    frameworks_assessed: List[str] = Field(
        default_factory=list,
        description="Frameworks assessed"
    )
    overall_compliance: bool = Field(..., description="Overall compliance status")
    compliance_report: ComplianceReport = Field(
        ...,
        description="Detailed compliance report"
    )

    # Audit trail
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Validation Rules and Thresholds
# =============================================================================


class ClaimRequirements(BaseModel):
    """Requirements for a specific claim type."""

    claim_type: ClaimType
    required_evidence: List[str]
    minimum_lifecycle_coverage: List[str]
    requires_third_party_verification: bool
    requires_offsetting_disclosure: bool
    specific_requirements: List[str]


# EU Green Claims Directive Requirements Database
EU_GREEN_CLAIMS_REQUIREMENTS: Dict[ClaimType, ClaimRequirements] = {
    ClaimType.CARBON_NEUTRAL: ClaimRequirements(
        claim_type=ClaimType.CARBON_NEUTRAL,
        required_evidence=[
            "ghg_inventory",
            "reduction_pathway",
            "offset_registry",
            "third_party_verification"
        ],
        minimum_lifecycle_coverage=["production", "use"],
        requires_third_party_verification=True,
        requires_offsetting_disclosure=True,
        specific_requirements=[
            "Emissions must be quantified per ISO 14064",
            "Reduction targets must be science-based",
            "Offsets must be from verified registry",
            "Must disclose offset percentage vs actual reductions"
        ]
    ),
    ClaimType.CLIMATE_POSITIVE: ClaimRequirements(
        claim_type=ClaimType.CLIMATE_POSITIVE,
        required_evidence=[
            "ghg_inventory",
            "net_negative_calculation",
            "offset_registry",
            "third_party_verification"
        ],
        minimum_lifecycle_coverage=["raw_materials", "production", "use", "disposal"],
        requires_third_party_verification=True,
        requires_offsetting_disclosure=True,
        specific_requirements=[
            "Must demonstrate net negative emissions",
            "Full lifecycle assessment required",
            "Offsets must exceed emissions by verifiable margin",
            "Must disclose calculation methodology"
        ]
    ),
    ClaimType.NET_ZERO: ClaimRequirements(
        claim_type=ClaimType.NET_ZERO,
        required_evidence=[
            "ghg_inventory",
            "sbti_commitment",
            "reduction_pathway",
            "residual_offset_plan"
        ],
        minimum_lifecycle_coverage=["raw_materials", "production", "use", "disposal"],
        requires_third_party_verification=True,
        requires_offsetting_disclosure=True,
        specific_requirements=[
            "Must align with SBTi Net-Zero Standard",
            "90%+ emissions reduction before offsetting",
            "Only residual emissions can be offset",
            "Must include Scope 3 emissions"
        ]
    ),
    ClaimType.CARBON_NEGATIVE: ClaimRequirements(
        claim_type=ClaimType.CARBON_NEGATIVE,
        required_evidence=[
            "ghg_inventory",
            "carbon_removal_verification",
            "third_party_audit"
        ],
        minimum_lifecycle_coverage=["raw_materials", "production", "use", "disposal"],
        requires_third_party_verification=True,
        requires_offsetting_disclosure=True,
        specific_requirements=[
            "Must remove more CO2 than emitted",
            "Carbon removals must be verified",
            "Must use permanent removal methods",
            "Full lifecycle emissions must be included"
        ]
    ),
    ClaimType.ECO_FRIENDLY: ClaimRequirements(
        claim_type=ClaimType.ECO_FRIENDLY,
        required_evidence=[
            "environmental_assessment",
            "lifecycle_assessment"
        ],
        minimum_lifecycle_coverage=["production"],
        requires_third_party_verification=False,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "Must specify which environmental aspects",
            "Cannot be used as generic claim",
            "Must provide quantifiable evidence",
            "Must disclose scope limitations"
        ]
    ),
    ClaimType.SUSTAINABLE: ClaimRequirements(
        claim_type=ClaimType.SUSTAINABLE,
        required_evidence=[
            "sustainability_report",
            "environmental_assessment",
            "social_assessment"
        ],
        minimum_lifecycle_coverage=["production"],
        requires_third_party_verification=False,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "Must cover environmental, social, and economic aspects",
            "Cannot be used as generic claim",
            "Must specify sustainability dimensions addressed",
            "Must provide quantifiable metrics"
        ]
    ),
    ClaimType.GREEN: ClaimRequirements(
        claim_type=ClaimType.GREEN,
        required_evidence=[
            "environmental_assessment"
        ],
        minimum_lifecycle_coverage=["production"],
        requires_third_party_verification=False,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "HIGH RISK: Extremely vague term",
            "Must be qualified with specific attributes",
            "Should specify environmental benefit",
            "Recommend avoiding this term alone"
        ]
    ),
    ClaimType.RENEWABLE: ClaimRequirements(
        claim_type=ClaimType.RENEWABLE,
        required_evidence=[
            "renewable_source_certification",
            "energy_certificates"
        ],
        minimum_lifecycle_coverage=["production"],
        requires_third_party_verification=True,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "Must specify percentage from renewable sources",
            "Energy certificates required (GO, REC)",
            "Must disclose calculation methodology",
            "Must specify renewable source type"
        ]
    ),
    ClaimType.RECYCLABLE: ClaimRequirements(
        claim_type=ClaimType.RECYCLABLE,
        required_evidence=[
            "material_composition",
            "recycling_infrastructure_availability"
        ],
        minimum_lifecycle_coverage=["disposal"],
        requires_third_party_verification=False,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "Must specify percentage that is recyclable",
            "Must consider actual recycling infrastructure",
            "Must disclose recycling conditions",
            "Cannot claim if <reasonable recycling access"
        ]
    ),
    ClaimType.BIODEGRADABLE: ClaimRequirements(
        claim_type=ClaimType.BIODEGRADABLE,
        required_evidence=[
            "biodegradation_test_results",
            "test_conditions"
        ],
        minimum_lifecycle_coverage=["disposal"],
        requires_third_party_verification=True,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "Must specify timeframe for biodegradation",
            "Must specify conditions (industrial, home, marine)",
            "Test results per EN 13432 or equivalent",
            "Cannot claim marine biodegradable without proof"
        ]
    ),
    ClaimType.COMPOSTABLE: ClaimRequirements(
        claim_type=ClaimType.COMPOSTABLE,
        required_evidence=[
            "compostability_certification",
            "test_results_en13432"
        ],
        minimum_lifecycle_coverage=["disposal"],
        requires_third_party_verification=True,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "Must specify industrial vs home compostable",
            "Certification per EN 13432 required",
            "Must disclose composting infrastructure availability",
            "Cannot claim if no composting infrastructure"
        ]
    ),
    ClaimType.PLASTIC_FREE: ClaimRequirements(
        claim_type=ClaimType.PLASTIC_FREE,
        required_evidence=[
            "material_composition",
            "supply_chain_verification"
        ],
        minimum_lifecycle_coverage=["raw_materials", "production"],
        requires_third_party_verification=False,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "Must be 100% plastic-free including packaging",
            "Must verify supply chain",
            "Must include bio-based plastics in assessment",
            "Must disclose scope (product only vs packaging)"
        ]
    ),
    ClaimType.ZERO_WASTE: ClaimRequirements(
        claim_type=ClaimType.ZERO_WASTE,
        required_evidence=[
            "waste_audit",
            "diversion_rate_calculation",
            "third_party_verification"
        ],
        minimum_lifecycle_coverage=["production"],
        requires_third_party_verification=True,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "Must achieve 90%+ diversion rate",
            "Must follow Zero Waste International Alliance definition",
            "Must include all waste streams",
            "Must disclose calculation methodology"
        ]
    ),
    ClaimType.LOW_CARBON: ClaimRequirements(
        claim_type=ClaimType.LOW_CARBON,
        required_evidence=[
            "carbon_footprint_calculation",
            "benchmark_comparison"
        ],
        minimum_lifecycle_coverage=["production"],
        requires_third_party_verification=False,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "Must quantify carbon footprint",
            "Must compare to industry benchmark",
            "Must define 'low' threshold",
            "Must disclose calculation methodology"
        ]
    ),
    ClaimType.REDUCED_EMISSIONS: ClaimRequirements(
        claim_type=ClaimType.REDUCED_EMISSIONS,
        required_evidence=[
            "baseline_emissions",
            "current_emissions",
            "reduction_calculation"
        ],
        minimum_lifecycle_coverage=["production"],
        requires_third_party_verification=False,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "Must specify reduction percentage",
            "Must specify baseline year/product",
            "Must use consistent methodology",
            "Must disclose scope of reduction"
        ]
    ),
    ClaimType.ENVIRONMENTALLY_FRIENDLY: ClaimRequirements(
        claim_type=ClaimType.ENVIRONMENTALLY_FRIENDLY,
        required_evidence=[
            "environmental_assessment",
            "lifecycle_assessment"
        ],
        minimum_lifecycle_coverage=["production"],
        requires_third_party_verification=False,
        requires_offsetting_disclosure=False,
        specific_requirements=[
            "HIGH RISK: Vague term under EU Directive",
            "Must qualify with specific environmental benefits",
            "Must provide quantifiable evidence",
            "Recommend using more specific claims"
        ]
    ),
}


# Greenwashing Pattern Detection Rules
GREENWASHING_PATTERNS: Dict[GreenwashingRedFlag, Dict[str, Any]] = {
    GreenwashingRedFlag.VAGUE_CLAIMS: {
        "keywords": [
            "green", "eco", "natural", "environmentally friendly",
            "earth friendly", "eco-conscious", "sustainable",
            "planet friendly", "nature-friendly"
        ],
        "without_qualifiers": True,
        "severity": "high",
        "description": "Claim uses vague, undefined terms without specific substantiation"
    },
    GreenwashingRedFlag.HIDDEN_TRADEOFFS: {
        "indicators": [
            "focuses_on_single_attribute",
            "ignores_significant_impacts",
            "partial_lifecycle_only"
        ],
        "severity": "medium",
        "description": "Claim emphasizes one attribute while hiding other environmental impacts"
    },
    GreenwashingRedFlag.FALSE_LABELS: {
        "indicators": [
            "fake_certification_logo",
            "self_declared_label",
            "unrecognized_certification"
        ],
        "severity": "critical",
        "description": "Claim uses fake or misleading certification labels"
    },
    GreenwashingRedFlag.IRRELEVANT_CLAIMS: {
        "indicators": [
            "legally_required_anyway",
            "industry_standard_practice",
            "already_banned_substance"
        ],
        "examples": ["CFC-free", "lead-free paint"],
        "severity": "medium",
        "description": "Claim highlights something required by law or industry standard"
    },
    GreenwashingRedFlag.LESSER_OF_EVILS: {
        "indicators": [
            "harmful_product_category",
            "comparative_within_bad_category"
        ],
        "severity": "high",
        "description": "Claim makes harmful product seem green by comparison"
    },
    GreenwashingRedFlag.FIBBING: {
        "indicators": [
            "false_claim",
            "misrepresented_data",
            "exaggerated_benefits"
        ],
        "severity": "critical",
        "description": "Claim is outright false or significantly exaggerated"
    },
    GreenwashingRedFlag.FALSE_CERTIFICATIONS: {
        "indicators": [
            "fake_third_party_endorsement",
            "expired_certification",
            "misused_certification"
        ],
        "severity": "critical",
        "description": "Claim falsely suggests third-party certification"
    },
}


# Valid certifications database
VALID_CERTIFICATIONS: Dict[str, Dict[str, Any]] = {
    "iso_14064": {"name": "ISO 14064", "scope": "ghg_accounting", "issuer": "ISO"},
    "iso_14067": {"name": "ISO 14067", "scope": "carbon_footprint", "issuer": "ISO"},
    "pef": {"name": "Product Environmental Footprint", "scope": "lifecycle", "issuer": "EU"},
    "en_13432": {"name": "EN 13432", "scope": "compostability", "issuer": "CEN"},
    "sbti": {"name": "Science Based Targets", "scope": "emissions_reduction", "issuer": "SBTi"},
    "gold_standard": {"name": "Gold Standard", "scope": "carbon_offsets", "issuer": "Gold Standard"},
    "verra_vcs": {"name": "Verified Carbon Standard", "scope": "carbon_offsets", "issuer": "Verra"},
    "eu_ecolabel": {"name": "EU Ecolabel", "scope": "environmental_performance", "issuer": "EU"},
    "blue_angel": {"name": "Blue Angel", "scope": "environmental_performance", "issuer": "RAL"},
    "nordic_swan": {"name": "Nordic Swan", "scope": "environmental_performance", "issuer": "Nordic"},
    "cradle_to_cradle": {"name": "Cradle to Cradle", "scope": "circular_design", "issuer": "C2C"},
    "fsc": {"name": "Forest Stewardship Council", "scope": "forestry", "issuer": "FSC"},
    "pefc": {"name": "Programme for Endorsement of Forest Certification", "scope": "forestry", "issuer": "PEFC"},
    "gots": {"name": "Global Organic Textile Standard", "scope": "textiles", "issuer": "GOTS"},
    "oeko_tex": {"name": "OEKO-TEX", "scope": "textiles", "issuer": "OEKO-TEX"},
}


# =============================================================================
# Green Claims Agent Implementation
# =============================================================================


class GreenClaimsAgent:
    """
    GL-008: Green Claims Verification Agent.

    This agent validates environmental claims against the EU Green Claims
    Directive using zero-hallucination deterministic rules:
    - Claim validation against explicit requirements
    - Substantiation scoring with weighted dimensions
    - Greenwashing pattern detection
    - Compliance report generation

    Aligned with:
    - EU Green Claims Directive (2023/0085)
    - ISO 14021 (Environmental labels - Self-declared claims)
    - ISO 14024 (Type I environmental labelling)
    - FTC Green Guides (US)

    Attributes:
        requirements_db: Database of claim requirements
        greenwashing_patterns: Greenwashing detection patterns
        valid_certifications: Database of valid certifications

    Example:
        >>> agent = GreenClaimsAgent()
        >>> result = agent.run(GreenClaimsInput(
        ...     claim_text="Our product is carbon neutral",
        ...     claim_type=ClaimType.CARBON_NEUTRAL,
        ...     evidence_items=[
        ...         EvidenceItem(
        ...             evidence_type="certification",
        ...             description="ISO 14064 certification",
        ...             is_third_party=True
        ...         )
        ...     ]
        ... ))
        >>> assert result.substantiation_score >= 0
    """

    AGENT_ID = "regulatory/green_claims_v1"
    VERSION = "1.0.0"
    DESCRIPTION = "Green claims verification and greenwashing detection"

    # Substantiation score weights
    SCORE_WEIGHTS = {
        "evidence_quality": 0.30,
        "scope_accuracy": 0.25,
        "transparency": 0.20,
        "verification": 0.15,
        "clarity": 0.10,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Green Claims Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict] = []
        self.requirements_db = EU_GREEN_CLAIMS_REQUIREMENTS
        self.greenwashing_patterns = GREENWASHING_PATTERNS
        self.valid_certifications = VALID_CERTIFICATIONS

        logger.info(f"GreenClaimsAgent initialized (version {self.VERSION})")

    def run(self, input_data: GreenClaimsInput) -> GreenClaimsOutput:
        """
        Execute the green claims verification.

        ZERO-HALLUCINATION verification:
        - All validation rules are explicit and deterministic
        - Scoring uses fixed weights and formulas
        - Pattern matching uses predefined patterns
        - No LLM in assessment path

        Args:
            input_data: Validated green claims input data

        Returns:
            Comprehensive verification result with scores and recommendations

        Raises:
            ValueError: If input validation fails
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []

        logger.info(
            f"Verifying green claim: type={input_data.claim_type}, "
            f"text='{input_data.claim_text[:50]}...'"
        )

        try:
            # Step 1: Validate claim against requirements
            claim_valid, validation_status, validation_messages = self._validate_claim(
                input_data
            )

            self._track_step("claim_validation", {
                "claim_type": input_data.claim_type.value,
                "claim_valid": claim_valid,
                "validation_status": validation_status.value,
                "message_count": len(validation_messages),
            })

            # Step 2: Calculate substantiation score
            score_breakdown = self._calculate_substantiation_score(input_data)

            self._track_step("substantiation_scoring", {
                "evidence_quality": score_breakdown.evidence_quality,
                "scope_accuracy": score_breakdown.scope_accuracy,
                "transparency": score_breakdown.transparency,
                "verification": score_breakdown.verification,
                "clarity": score_breakdown.clarity,
                "weighted_total": score_breakdown.weighted_total,
            })

            # Step 3: Detect greenwashing patterns
            red_flags, greenwashing_risk_score = self._detect_greenwashing(input_data)

            self._track_step("greenwashing_detection", {
                "red_flags_count": len(red_flags),
                "risk_score": greenwashing_risk_score,
                "detected_patterns": [rf.red_flag.value for rf in red_flags],
            })

            # Step 4: Generate compliance report
            compliance_report = self._generate_report(
                input_data,
                claim_valid,
                validation_messages,
                score_breakdown,
                red_flags,
            )

            self._track_step("report_generation", {
                "overall_status": compliance_report.overall_status,
                "risk_level": compliance_report.risk_level,
                "frameworks_assessed": len(compliance_report.framework_results),
            })

            # Step 5: Determine substantiation level
            substantiation_level = self._get_substantiation_level(
                score_breakdown.weighted_total
            )

            # Step 6: Determine overall compliance
            overall_compliance = (
                claim_valid
                and not red_flags
                and score_breakdown.weighted_total >= 60
            )

            # Step 7: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 8: Calculate processing time
            processing_time_ms = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000

            # Step 9: Create output
            output = GreenClaimsOutput(
                claim_text=input_data.claim_text,
                claim_type=input_data.claim_type.value,
                claim_valid=claim_valid,
                validation_status=validation_status.value,
                validation_messages=validation_messages,
                substantiation_score=round(score_breakdown.weighted_total, 2),
                substantiation_level=substantiation_level.value,
                score_breakdown=score_breakdown,
                greenwashing_detected=len(red_flags) > 0,
                red_flags=red_flags,
                greenwashing_risk_score=round(greenwashing_risk_score, 2),
                frameworks_assessed=[f.value for f in input_data.target_frameworks],
                overall_compliance=overall_compliance,
                compliance_report=compliance_report,
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time_ms, 2),
            )

            logger.info(
                f"Green claims verification complete: valid={claim_valid}, "
                f"score={score_breakdown.weighted_total:.1f}, "
                f"greenwashing_risk={greenwashing_risk_score:.1f} "
                f"(duration: {processing_time_ms:.2f}ms)"
            )

            return output

        except Exception as e:
            logger.error(f"Green claims verification failed: {str(e)}", exc_info=True)
            raise

    def _validate_claim(
        self,
        input_data: GreenClaimsInput,
    ) -> Tuple[bool, ValidationStatus, List[str]]:
        """
        Validate claim against EU Green Claims Directive requirements.

        ZERO-HALLUCINATION: All validation rules are explicit and deterministic.

        Args:
            input_data: The green claims input

        Returns:
            Tuple of (is_valid, status, messages)
        """
        messages: List[str] = []
        requirements = self.requirements_db.get(input_data.claim_type)

        if not requirements:
            messages.append(f"No requirements defined for claim type: {input_data.claim_type}")
            return False, ValidationStatus.INVALID, messages

        # Check 1: Required evidence
        evidence_types = [e.evidence_type.lower() for e in input_data.evidence_items]
        missing_evidence = []

        for required in requirements.required_evidence:
            if not any(required.lower() in et for et in evidence_types):
                missing_evidence.append(required)

        if missing_evidence:
            messages.append(
                f"Missing required evidence: {', '.join(missing_evidence)}"
            )

        # Check 2: Third-party verification requirement
        if requirements.requires_third_party_verification:
            has_third_party = any(e.is_third_party for e in input_data.evidence_items)
            if not has_third_party:
                messages.append(
                    "Third-party verification required but not provided"
                )

        # Check 3: Lifecycle coverage
        if input_data.claim_scope and requirements.minimum_lifecycle_coverage:
            provided_stages = set(input_data.claim_scope.lifecycle_stages)
            required_stages = set(requirements.minimum_lifecycle_coverage)
            missing_stages = required_stages - provided_stages

            if missing_stages:
                messages.append(
                    f"Missing lifecycle stages: {', '.join(missing_stages)}"
                )

        # Check 4: Offsetting disclosure for carbon claims
        if requirements.requires_offsetting_disclosure:
            has_offset_disclosure = any(
                "offset" in e.description.lower() or "offset" in e.evidence_type.lower()
                for e in input_data.evidence_items
            )
            if not has_offset_disclosure:
                messages.append(
                    "Offsetting disclosure required for this claim type"
                )

        # Check 5: Comparative claim requirements
        if input_data.comparative_claim and not input_data.comparison_baseline:
            messages.append(
                "Comparative claims must specify a baseline for comparison"
            )

        # Check 6: Specific claim type requirements
        for specific_req in requirements.specific_requirements:
            if "HIGH RISK" in specific_req:
                messages.append(f"WARNING: {specific_req}")

        # Determine validation status
        if not messages:
            return True, ValidationStatus.VALID, ["All requirements met"]

        if len(input_data.evidence_items) == 0:
            return False, ValidationStatus.INSUFFICIENT_EVIDENCE, messages

        critical_missing = (
            len(missing_evidence) > len(requirements.required_evidence) / 2
            or (requirements.requires_third_party_verification
                and not any(e.is_third_party for e in input_data.evidence_items))
        )

        if critical_missing:
            return False, ValidationStatus.INVALID, messages

        return False, ValidationStatus.REQUIRES_VERIFICATION, messages

    def _calculate_substantiation_score(
        self,
        input_data: GreenClaimsInput,
    ) -> SubstantiationScoreBreakdown:
        """
        Calculate 5-dimension substantiation score.

        ZERO-HALLUCINATION: Fixed weights and deterministic formulas.

        Weights:
        - Evidence Quality: 30%
        - Scope Accuracy: 25%
        - Transparency: 20%
        - Verification: 15%
        - Clarity: 10%

        Args:
            input_data: The green claims input

        Returns:
            Score breakdown with all dimensions
        """
        # Dimension 1: Evidence Quality (30%)
        evidence_quality = self._score_evidence_quality(input_data)

        # Dimension 2: Scope Accuracy (25%)
        scope_accuracy = self._score_scope_accuracy(input_data)

        # Dimension 3: Transparency (20%)
        transparency = self._score_transparency(input_data)

        # Dimension 4: Verification (15%)
        verification = self._score_verification(input_data)

        # Dimension 5: Clarity (10%)
        clarity = self._score_clarity(input_data)

        # ZERO-HALLUCINATION CALCULATION
        # Formula: weighted_total = sum(dimension_score * weight)
        weighted_total = (
            evidence_quality * self.SCORE_WEIGHTS["evidence_quality"]
            + scope_accuracy * self.SCORE_WEIGHTS["scope_accuracy"]
            + transparency * self.SCORE_WEIGHTS["transparency"]
            + verification * self.SCORE_WEIGHTS["verification"]
            + clarity * self.SCORE_WEIGHTS["clarity"]
        )

        return SubstantiationScoreBreakdown(
            evidence_quality=round(evidence_quality, 2),
            scope_accuracy=round(scope_accuracy, 2),
            transparency=round(transparency, 2),
            verification=round(verification, 2),
            clarity=round(clarity, 2),
            weighted_total=round(weighted_total, 2),
        )

    def _score_evidence_quality(self, input_data: GreenClaimsInput) -> float:
        """
        Score evidence quality dimension (0-100).

        Factors:
        - Number of evidence items
        - Evidence type diversity
        - Third-party evidence
        - Recency of evidence
        """
        score = 0.0
        evidence_items = input_data.evidence_items

        if not evidence_items:
            return 0.0

        # Factor 1: Number of evidence items (up to 25 points)
        evidence_count_score = min(len(evidence_items) * 5, 25)
        score += evidence_count_score

        # Factor 2: Evidence type diversity (up to 25 points)
        unique_types = len(set(e.evidence_type for e in evidence_items))
        diversity_score = min(unique_types * 6.25, 25)
        score += diversity_score

        # Factor 3: Third-party evidence (up to 30 points)
        third_party_count = sum(1 for e in evidence_items if e.is_third_party)
        third_party_score = min(third_party_count * 15, 30)
        score += third_party_score

        # Factor 4: Evidence recency and validity (up to 20 points)
        valid_evidence = sum(
            1 for e in evidence_items
            if e.source and e.description
        )
        validity_score = min(valid_evidence / max(len(evidence_items), 1) * 20, 20)
        score += validity_score

        return min(score, 100)

    def _score_scope_accuracy(self, input_data: GreenClaimsInput) -> float:
        """
        Score scope accuracy dimension (0-100).

        Factors:
        - Lifecycle stage coverage
        - Geographic scope definition
        - Time period specification
        - Exclusions disclosure
        """
        score = 0.0
        scope = input_data.claim_scope
        requirements = self.requirements_db.get(input_data.claim_type)

        if not scope:
            return 20.0  # Minimum score if no scope defined

        # Factor 1: Lifecycle coverage (up to 40 points)
        if scope.lifecycle_stages:
            if requirements and requirements.minimum_lifecycle_coverage:
                required_count = len(requirements.minimum_lifecycle_coverage)
                covered_count = len(
                    set(scope.lifecycle_stages) &
                    set(requirements.minimum_lifecycle_coverage)
                )
                lifecycle_score = (covered_count / required_count) * 40
            else:
                lifecycle_score = min(len(scope.lifecycle_stages) * 10, 40)
            score += lifecycle_score

        # Factor 2: Geographic scope (up to 20 points)
        if scope.geographic_scope:
            score += 20

        # Factor 3: Time period (up to 20 points)
        if scope.time_period:
            score += 20

        # Factor 4: Exclusions disclosure (up to 20 points)
        if scope.exclusions:
            score += 20
        elif scope.product_scope:
            score += 10

        return min(score, 100)

    def _score_transparency(self, input_data: GreenClaimsInput) -> float:
        """
        Score transparency dimension (0-100).

        Factors:
        - Evidence source disclosure
        - Verification URLs provided
        - Methodology disclosure
        - Limitations disclosure
        """
        score = 0.0
        evidence_items = input_data.evidence_items

        if not evidence_items:
            return 0.0

        # Factor 1: Source disclosure (up to 30 points)
        sources_provided = sum(1 for e in evidence_items if e.source)
        source_score = (sources_provided / len(evidence_items)) * 30
        score += source_score

        # Factor 2: Verification URLs (up to 30 points)
        urls_provided = sum(1 for e in evidence_items if e.verification_url)
        url_score = (urls_provided / len(evidence_items)) * 30
        score += url_score

        # Factor 3: Dates provided (up to 20 points)
        dates_provided = sum(1 for e in evidence_items if e.date_issued)
        date_score = (dates_provided / len(evidence_items)) * 20
        score += date_score

        # Factor 4: Scope and exclusions (up to 20 points)
        if input_data.claim_scope:
            if input_data.claim_scope.exclusions:
                score += 20
            elif input_data.claim_scope.product_scope:
                score += 10

        return min(score, 100)

    def _score_verification(self, input_data: GreenClaimsInput) -> float:
        """
        Score verification dimension (0-100).

        Factors:
        - Third-party verification presence
        - Recognized certification bodies
        - Certification validity
        """
        score = 0.0
        evidence_items = input_data.evidence_items

        if not evidence_items:
            return 0.0

        # Factor 1: Third-party verification (up to 50 points)
        third_party_items = [e for e in evidence_items if e.is_third_party]
        if third_party_items:
            third_party_score = min(len(third_party_items) * 25, 50)
            score += third_party_score

        # Factor 2: Recognized certifications (up to 30 points)
        recognized_certs = 0
        for evidence in evidence_items:
            evidence_text = (evidence.description + " " + evidence.evidence_type).lower()
            for cert_key in self.valid_certifications:
                if cert_key.replace("_", " ") in evidence_text:
                    recognized_certs += 1
                    break

        cert_score = min(recognized_certs * 15, 30)
        score += cert_score

        # Factor 3: Expiry information (up to 20 points)
        expiry_provided = sum(1 for e in evidence_items if e.expiry_date)
        if evidence_items:
            expiry_score = (expiry_provided / len(evidence_items)) * 20
            score += expiry_score

        return min(score, 100)

    def _score_clarity(self, input_data: GreenClaimsInput) -> float:
        """
        Score clarity dimension (0-100).

        Factors:
        - Claim specificity (not vague)
        - Quantifiable metrics
        - Clear scope statement
        """
        score = 50.0  # Base score

        claim_text = input_data.claim_text.lower()

        # Factor 1: Vague terms penalty (up to -30 points)
        vague_terms = ["green", "eco", "natural", "friendly", "sustainable"]
        vague_count = sum(1 for term in vague_terms if term in claim_text)
        vague_penalty = min(vague_count * 10, 30)
        score -= vague_penalty

        # Factor 2: Quantifiable metrics bonus (up to +25 points)
        has_percentage = bool(re.search(r'\d+\s*%', claim_text))
        has_numbers = bool(re.search(r'\d+', claim_text))
        if has_percentage:
            score += 25
        elif has_numbers:
            score += 15

        # Factor 3: Specific scope (up to +25 points)
        if input_data.claim_scope:
            if input_data.claim_scope.lifecycle_stages:
                score += 10
            if input_data.claim_scope.product_scope:
                score += 10
            if input_data.claim_scope.time_period:
                score += 5

        return max(min(score, 100), 0)

    def _detect_greenwashing(
        self,
        input_data: GreenClaimsInput,
    ) -> Tuple[List[GreenwashingDetection], float]:
        """
        Detect greenwashing red flag patterns.

        ZERO-HALLUCINATION: Pattern matching against predefined patterns.

        7 Red Flag Patterns:
        1. Vague Claims
        2. Hidden Tradeoffs
        3. False Labels
        4. Irrelevant Claims
        5. Lesser of Evils
        6. Fibbing
        7. False Certifications

        Args:
            input_data: The green claims input

        Returns:
            Tuple of (detected red flags, risk score)
        """
        red_flags: List[GreenwashingDetection] = []
        claim_text = input_data.claim_text.lower()

        # Pattern 1: Vague Claims
        vague_pattern = self.greenwashing_patterns[GreenwashingRedFlag.VAGUE_CLAIMS]
        vague_keywords_found = [
            kw for kw in vague_pattern["keywords"]
            if kw in claim_text
        ]
        if vague_keywords_found and len(input_data.evidence_items) < 2:
            red_flags.append(GreenwashingDetection(
                red_flag=GreenwashingRedFlag.VAGUE_CLAIMS,
                severity=vague_pattern["severity"],
                description=vague_pattern["description"],
                evidence=[f"Found vague terms: {', '.join(vague_keywords_found)}"],
                recommendation="Qualify vague terms with specific, measurable claims"
            ))

        # Pattern 2: Hidden Tradeoffs
        if input_data.claim_scope:
            if len(input_data.claim_scope.lifecycle_stages) == 1:
                red_flags.append(GreenwashingDetection(
                    red_flag=GreenwashingRedFlag.HIDDEN_TRADEOFFS,
                    severity="medium",
                    description="Claim focuses on single lifecycle stage",
                    evidence=[
                        f"Only covers: {input_data.claim_scope.lifecycle_stages}"
                    ],
                    recommendation="Expand scope to cover full lifecycle impacts"
                ))

        # Pattern 3: False Labels
        if input_data.evidence_items:
            for evidence in input_data.evidence_items:
                if (evidence.evidence_type.lower() == "certification"
                        and not evidence.is_third_party
                        and not evidence.verification_url):
                    red_flags.append(GreenwashingDetection(
                        red_flag=GreenwashingRedFlag.FALSE_LABELS,
                        severity="critical",
                        description="Self-declared certification without verification",
                        evidence=[f"Unverified: {evidence.description}"],
                        recommendation="Obtain third-party verification for certifications"
                    ))
                    break

        # Pattern 4: Irrelevant Claims
        irrelevant_terms = ["cfc-free", "cfc free", "lead-free", "lead free"]
        for term in irrelevant_terms:
            if term in claim_text:
                red_flags.append(GreenwashingDetection(
                    red_flag=GreenwashingRedFlag.IRRELEVANT_CLAIMS,
                    severity="medium",
                    description="Claim highlights legally required or banned attribute",
                    evidence=[f"Found irrelevant term: {term}"],
                    recommendation="Focus on environmental benefits beyond legal requirements"
                ))
                break

        # Pattern 5: Lesser of Evils
        harmful_categories = [
            "tobacco", "cigarette", "fossil fuel", "coal", "petroleum"
        ]
        product_category = (input_data.product_category or "").lower()
        for harmful in harmful_categories:
            if harmful in product_category or harmful in claim_text:
                red_flags.append(GreenwashingDetection(
                    red_flag=GreenwashingRedFlag.LESSER_OF_EVILS,
                    severity="high",
                    description="Environmental claim on inherently harmful product",
                    evidence=[f"Harmful category detected: {harmful}"],
                    recommendation="Consider if green claims are appropriate for this product"
                ))
                break

        # Pattern 6: Fibbing - Check for exaggerated claims
        exaggeration_terms = ["100%", "completely", "totally", "pure", "perfect"]
        for term in exaggeration_terms:
            if term in claim_text:
                has_supporting_evidence = any(
                    "100" in e.description or "full" in e.description.lower()
                    for e in input_data.evidence_items
                )
                if not has_supporting_evidence:
                    red_flags.append(GreenwashingDetection(
                        red_flag=GreenwashingRedFlag.FIBBING,
                        severity="critical",
                        description="Absolute claim without supporting evidence",
                        evidence=[f"Unsubstantiated absolute term: {term}"],
                        recommendation="Provide evidence for absolute claims or use qualified language"
                    ))
                    break

        # Pattern 7: False Certifications
        certification_keywords = [
            "certified", "approved", "endorsed", "verified", "accredited"
        ]
        has_cert_claim = any(kw in claim_text for kw in certification_keywords)
        if has_cert_claim:
            has_valid_cert = any(
                e.is_third_party and e.verification_url
                for e in input_data.evidence_items
            )
            if not has_valid_cert:
                red_flags.append(GreenwashingDetection(
                    red_flag=GreenwashingRedFlag.FALSE_CERTIFICATIONS,
                    severity="critical",
                    description="Certification claim without verifiable third-party evidence",
                    evidence=["Certification mentioned but not verified"],
                    recommendation="Provide verifiable third-party certification"
                ))

        # Calculate risk score
        # ZERO-HALLUCINATION CALCULATION
        severity_weights = {
            "critical": 30,
            "high": 20,
            "medium": 10,
            "low": 5,
        }

        risk_score = sum(
            severity_weights.get(rf.severity, 10)
            for rf in red_flags
        )
        risk_score = min(risk_score, 100)

        return red_flags, risk_score

    def _generate_report(
        self,
        input_data: GreenClaimsInput,
        claim_valid: bool,
        validation_messages: List[str],
        score_breakdown: SubstantiationScoreBreakdown,
        red_flags: List[GreenwashingDetection],
    ) -> ComplianceReport:
        """
        Generate comprehensive compliance report.

        Args:
            input_data: The green claims input
            claim_valid: Whether claim is valid
            validation_messages: Validation messages
            score_breakdown: Substantiation score breakdown
            red_flags: Detected greenwashing red flags

        Returns:
            Compliance report with recommendations
        """
        # Assess each framework
        framework_results: List[FrameworkComplianceResult] = []

        for framework in input_data.target_frameworks:
            result = self._assess_framework_compliance(
                framework,
                input_data,
                claim_valid,
                validation_messages,
                score_breakdown,
            )
            framework_results.append(result)

        # Determine overall compliance
        overall_compliant = all(fr.compliant for fr in framework_results)

        # Determine risk level
        if red_flags:
            critical_flags = sum(1 for rf in red_flags if rf.severity == "critical")
            if critical_flags > 0:
                risk_level = "critical"
            elif any(rf.severity == "high" for rf in red_flags):
                risk_level = "high"
            else:
                risk_level = "medium"
        elif not claim_valid:
            risk_level = "medium"
        elif score_breakdown.weighted_total < 60:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Generate summary
        if overall_compliant and not red_flags:
            summary = (
                f"Claim '{input_data.claim_type.value}' meets compliance requirements "
                f"with substantiation score of {score_breakdown.weighted_total:.1f}/100."
            )
        elif red_flags:
            summary = (
                f"ALERT: {len(red_flags)} greenwashing red flag(s) detected. "
                f"Risk level: {risk_level}. Immediate action required."
            )
        else:
            summary = (
                f"Claim requires attention. Substantiation score: "
                f"{score_breakdown.weighted_total:.1f}/100. "
                f"See recommendations below."
            )

        # Generate immediate actions
        immediate_actions: List[str] = []
        if red_flags:
            for rf in red_flags:
                if rf.severity in ["critical", "high"]:
                    immediate_actions.append(rf.recommendation)

        if not claim_valid:
            immediate_actions.extend([
                msg for msg in validation_messages
                if "missing" in msg.lower() or "required" in msg.lower()
            ])

        # Generate long-term recommendations
        long_term_recommendations: List[str] = []

        if score_breakdown.evidence_quality < 60:
            long_term_recommendations.append(
                "Strengthen evidence base with additional third-party verifications"
            )
        if score_breakdown.scope_accuracy < 60:
            long_term_recommendations.append(
                "Expand lifecycle coverage to include all relevant stages"
            )
        if score_breakdown.transparency < 60:
            long_term_recommendations.append(
                "Improve transparency by providing verification URLs and methodologies"
            )
        if score_breakdown.verification < 60:
            long_term_recommendations.append(
                "Obtain additional third-party certifications from recognized bodies"
            )
        if score_breakdown.clarity < 60:
            long_term_recommendations.append(
                "Make claims more specific with quantifiable metrics"
            )

        # Identify evidence gaps
        evidence_gaps: List[str] = []
        requirements = self.requirements_db.get(input_data.claim_type)

        if requirements:
            evidence_types = [e.evidence_type.lower() for e in input_data.evidence_items]
            for required in requirements.required_evidence:
                if not any(required.lower() in et for et in evidence_types):
                    evidence_gaps.append(required)

        # Determine overall status
        if overall_compliant and not red_flags and score_breakdown.weighted_total >= 80:
            overall_status = "COMPLIANT"
        elif risk_level == "critical":
            overall_status = "NON_COMPLIANT - CRITICAL ISSUES"
        elif risk_level == "high":
            overall_status = "NON_COMPLIANT - ACTION REQUIRED"
        elif claim_valid and score_breakdown.weighted_total >= 60:
            overall_status = "CONDITIONALLY COMPLIANT"
        else:
            overall_status = "REQUIRES REMEDIATION"

        return ComplianceReport(
            summary=summary,
            overall_status=overall_status,
            risk_level=risk_level,
            framework_results=framework_results,
            immediate_actions=immediate_actions,
            long_term_recommendations=long_term_recommendations,
            evidence_gaps=evidence_gaps,
        )

    def _assess_framework_compliance(
        self,
        framework: ComplianceFramework,
        input_data: GreenClaimsInput,
        claim_valid: bool,
        validation_messages: List[str],
        score_breakdown: SubstantiationScoreBreakdown,
    ) -> FrameworkComplianceResult:
        """
        Assess compliance against a specific framework.

        Args:
            framework: The compliance framework
            input_data: The green claims input
            claim_valid: Whether claim passed validation
            validation_messages: Validation messages
            score_breakdown: Substantiation score breakdown

        Returns:
            Framework-specific compliance result
        """
        requirements_met: List[str] = []
        requirements_failed: List[str] = []
        recommendations: List[str] = []

        if framework == ComplianceFramework.EU_GREEN_CLAIMS_DIRECTIVE:
            # EU Green Claims Directive requirements
            if claim_valid:
                requirements_met.append("Claim substantiation requirements")
            else:
                requirements_failed.append("Claim substantiation requirements")
                recommendations.append("Address validation issues in claim")

            if any(e.is_third_party for e in input_data.evidence_items):
                requirements_met.append("Independent verification")
            else:
                requirements_failed.append("Independent verification")
                recommendations.append("Obtain third-party verification")

            if input_data.claim_scope and input_data.claim_scope.lifecycle_stages:
                requirements_met.append("Lifecycle scope definition")
            else:
                requirements_failed.append("Lifecycle scope definition")
                recommendations.append("Define lifecycle coverage")

            if score_breakdown.transparency >= 60:
                requirements_met.append("Transparency and accessibility")
            else:
                requirements_failed.append("Transparency and accessibility")
                recommendations.append("Improve evidence accessibility")

        elif framework == ComplianceFramework.ISO_14021:
            # ISO 14021 - Self-declared environmental claims
            if not any(
                term in input_data.claim_text.lower()
                for term in ["eco", "green", "natural"]
            ):
                requirements_met.append("No vague/misleading terms")
            else:
                requirements_failed.append("Avoid vague terms")
                recommendations.append("Use specific, qualified language")

            if score_breakdown.evidence_quality >= 50:
                requirements_met.append("Verifiable claims")
            else:
                requirements_failed.append("Claims must be verifiable")
                recommendations.append("Provide verifiable evidence")

        elif framework == ComplianceFramework.FTC_GREEN_GUIDES:
            # FTC Green Guides requirements
            if input_data.claim_scope:
                requirements_met.append("Clear scope qualification")
            else:
                requirements_failed.append("Scope qualification")
                recommendations.append("Clearly qualify claim scope")

            if not input_data.comparative_claim or input_data.comparison_baseline:
                requirements_met.append("Comparative claims substantiated")
            else:
                requirements_failed.append("Comparative claim baseline")
                recommendations.append("Provide comparison baseline")

        else:
            # Generic assessment
            if claim_valid:
                requirements_met.append("Basic substantiation")
            else:
                requirements_failed.append("Basic substantiation")

        compliant = len(requirements_failed) == 0

        return FrameworkComplianceResult(
            framework=framework.value,
            compliant=compliant,
            requirements_met=requirements_met,
            requirements_failed=requirements_failed,
            recommendations=recommendations,
        )

    def _get_substantiation_level(self, score: float) -> SubstantiationLevel:
        """
        Get substantiation level based on score.

        ZERO-HALLUCINATION: Fixed thresholds.

        Args:
            score: Weighted total score

        Returns:
            Substantiation level
        """
        if score >= 80:
            return SubstantiationLevel.EXCELLENT
        elif score >= 60:
            return SubstantiationLevel.GOOD
        elif score >= 40:
            return SubstantiationLevel.MODERATE
        elif score >= 20:
            return SubstantiationLevel.WEAK
        else:
            return SubstantiationLevel.INSUFFICIENT

    def _track_step(self, step_type: str, data: Dict[str, Any]) -> None:
        """Track a processing step for provenance."""
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 hash of complete provenance chain.

        This hash enables:
        - Verification that assessment was deterministic
        - Audit trail for regulatory compliance
        - Reproducibility checking
        """
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def validate_claim(
        self,
        claim_text: str,
        claim_type: ClaimType,
        evidence_items: Optional[List[EvidenceItem]] = None,
    ) -> Tuple[bool, ValidationStatus, List[str]]:
        """
        Validate a claim against EU Green Claims Directive.

        Public API for claim validation without full processing.

        Args:
            claim_text: The claim text
            claim_type: Type of claim
            evidence_items: Optional evidence items

        Returns:
            Tuple of (is_valid, status, messages)
        """
        input_data = GreenClaimsInput(
            claim_text=claim_text,
            claim_type=claim_type,
            evidence_items=evidence_items or [],
        )
        return self._validate_claim(input_data)

    def calculate_substantiation_score(
        self,
        claim_text: str,
        claim_type: ClaimType,
        evidence_items: Optional[List[EvidenceItem]] = None,
        claim_scope: Optional[ClaimScope] = None,
    ) -> SubstantiationScoreBreakdown:
        """
        Calculate substantiation score for a claim.

        Public API for scoring without full processing.

        Args:
            claim_text: The claim text
            claim_type: Type of claim
            evidence_items: Optional evidence items
            claim_scope: Optional claim scope

        Returns:
            Score breakdown with all dimensions
        """
        input_data = GreenClaimsInput(
            claim_text=claim_text,
            claim_type=claim_type,
            evidence_items=evidence_items or [],
            claim_scope=claim_scope,
        )
        return self._calculate_substantiation_score(input_data)

    def detect_greenwashing(
        self,
        claim_text: str,
        claim_type: ClaimType,
        evidence_items: Optional[List[EvidenceItem]] = None,
        product_category: Optional[str] = None,
    ) -> Tuple[List[GreenwashingDetection], float]:
        """
        Detect greenwashing patterns in a claim.

        Public API for greenwashing detection without full processing.

        Args:
            claim_text: The claim text
            claim_type: Type of claim
            evidence_items: Optional evidence items
            product_category: Optional product category

        Returns:
            Tuple of (red flags, risk score)
        """
        input_data = GreenClaimsInput(
            claim_text=claim_text,
            claim_type=claim_type,
            evidence_items=evidence_items or [],
            product_category=product_category,
        )
        return self._detect_greenwashing(input_data)

    def generate_report(
        self,
        input_data: GreenClaimsInput,
    ) -> ComplianceReport:
        """
        Generate compliance report for a claim.

        Public API for report generation.

        Args:
            input_data: Complete green claims input

        Returns:
            Compliance report with recommendations
        """
        claim_valid, _, validation_messages = self._validate_claim(input_data)
        score_breakdown = self._calculate_substantiation_score(input_data)
        red_flags, _ = self._detect_greenwashing(input_data)

        return self._generate_report(
            input_data,
            claim_valid,
            validation_messages,
            score_breakdown,
            red_flags,
        )

    def get_claim_types(self) -> List[str]:
        """Get list of supported claim types."""
        return [ct.value for ct in ClaimType]

    def get_greenwashing_patterns(self) -> List[str]:
        """Get list of greenwashing red flag patterns."""
        return [gf.value for gf in GreenwashingRedFlag]

    def get_valid_certifications(self) -> List[Dict[str, Any]]:
        """Get list of recognized certifications."""
        return [
            {"id": k, **v}
            for k, v in self.valid_certifications.items()
        ]

    def get_claim_requirements(self, claim_type: ClaimType) -> Optional[Dict[str, Any]]:
        """Get requirements for a specific claim type."""
        requirements = self.requirements_db.get(claim_type)
        if requirements:
            return {
                "claim_type": requirements.claim_type.value,
                "required_evidence": requirements.required_evidence,
                "minimum_lifecycle_coverage": requirements.minimum_lifecycle_coverage,
                "requires_third_party_verification": requirements.requires_third_party_verification,
                "requires_offsetting_disclosure": requirements.requires_offsetting_disclosure,
                "specific_requirements": requirements.specific_requirements,
            }
        return None


# =============================================================================
# Pack Specification
# =============================================================================


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "regulatory/green_claims_v1",
    "name": "Green Claims Verification Agent",
    "version": "1.0.0",
    "summary": "Verify green claims against EU Green Claims Directive and detect greenwashing",
    "tags": [
        "green-claims",
        "greenwashing",
        "eu-directive",
        "environmental-marketing",
        "substantiation",
    ],
    "owners": ["regulatory-team"],
    "compute": {
        "entrypoint": "python://agents.gl_008_green_claims.agent:GreenClaimsAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "reg://eu/green-claims-directive/2023"},
        {"ref": "std://iso/14021/2016"},
        {"ref": "std://iso/14024/2018"},
    ],
    "provenance": {
        "directive_version": "2023/0085",
        "iso_version": "14021:2016",
        "enable_audit": True,
    },
}
