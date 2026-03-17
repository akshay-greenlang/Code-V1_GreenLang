# -*- coding: utf-8 -*-
"""
ClaimSubstantiationEngine - PACK-018 EU Green Claims Prep Engine 1
===================================================================

Assesses environmental claim substantiation per EU Green Claims Directive
Articles 3-4, evaluating scientific validity, data quality, scope
completeness, verification independence, and transparency.

The EU Green Claims Directive (Directive on Empowering Consumers for the
Green Transition, Proposal COM/2023/166) establishes requirements for
environmental claims made by traders to consumers.  Under Articles 3-4,
environmental claims must be substantiated by widely recognised scientific
evidence, be accurate, and take into account the full lifecycle of the
product or organisation.

Article 3 Requirements:
    - Para 1: Environmental claims shall be substantiated by widely
      recognised scientific evidence, using accurate information and
      taking into account relevant international standards.
    - Para 2: The substantiation shall demonstrate that the environmental
      impacts, aspects, or performance that are the subject of the claim
      are significant from a life cycle perspective.
    - Para 3: Where a claim relates to a product, all significant
      environmental aspects and impacts shall be identified.
    - Para 4: The substantiation shall include primary and/or robust
      secondary data for the product's lifecycle stages.

Article 4 Requirements:
    - Para 1: Environmental claims shall not be misleading.
    - Para 2: Claims using aggregated indicators of overall environmental
      impact shall be based on sufficient evidence.
    - Para 3: Claims shall not exaggerate the environmental benefit.
    - Para 4: Claims about future environmental performance shall include
      clear time-bound commitments and measurable targets.

Regulatory References:
    - EU Green Claims Directive Proposal COM/2023/166
    - EU Unfair Commercial Practices Directive 2005/29/EC (as amended)
    - ISO 14024 (Type I Environmental Labels)
    - ISO 14021 (Self-declared Environmental Claims)
    - PEF/OEF methodology (Commission Recommendation 2013/179/EU)

Zero-Hallucination:
    - All scoring uses deterministic weighted Decimal arithmetic
    - Compliance checks use rule-based threshold evaluation
    - Claim completeness uses set intersection and coverage ratios
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-018 EU Green Claims Prep
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClaimType(str, Enum):
    """Types of environmental claims subject to the Green Claims Directive.

    Per Article 2(1), an environmental claim is any message or
    representation that states or implies a product or trader has a
    positive or reduced impact on the environment.  These 16 categories
    cover the most common claim types encountered in the EU market.
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


class ClaimRiskLevel(str, Enum):
    """Risk level associated with an environmental claim.

    Claims with broader scope or more absolute language carry higher
    regulatory risk under the Green Claims Directive, as they are
    more likely to be considered misleading under Article 4.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SubstantiationLevel(str, Enum):
    """Substantiation quality level for an environmental claim.

    Determined by the overall weighted assessment score across all
    five substantiation dimensions (scientific validity, data quality,
    scope completeness, verification independence, transparency).
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"


class EvidenceType(str, Enum):
    """Types of evidence that can substantiate an environmental claim.

    Per Article 3(2), claims must be substantiated by widely recognised
    scientific evidence.  These categories represent the evidence types
    accepted under the Directive's substantiation framework.
    """
    CERTIFICATION = "certification"
    LCA_STUDY = "lca_study"
    TEST_REPORT = "test_report"
    AUDIT_REPORT = "audit_report"
    MEASUREMENT = "measurement"
    THIRD_PARTY_VERIFICATION = "third_party_verification"


class LifecycleStage(str, Enum):
    """Product lifecycle stages per PEF methodology.

    Per Article 3(1)(d), substantiation must consider the full lifecycle
    of the product, covering raw material extraction through end-of-life.
    """
    RAW_MATERIALS = "raw_materials"
    MANUFACTURING = "manufacturing"
    TRANSPORTATION = "transportation"
    DISTRIBUTION = "distribution"
    USE = "use"
    END_OF_LIFE = "end_of_life"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Substantiation dimension weights (must sum to 100).
# These weights reflect the relative importance of each dimension
# in assessing claim substantiation quality under the Directive.
DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    "scientific_validity": Decimal("30"),
    "data_quality": Decimal("25"),
    "scope_completeness": Decimal("20"),
    "verification_independence": Decimal("15"),
    "transparency": Decimal("10"),
}

# Substantiation level thresholds (overall score 0-100).
SUBSTANTIATION_THRESHOLDS: Dict[str, Decimal] = {
    "excellent": Decimal("85"),
    "good": Decimal("70"),
    "moderate": Decimal("50"),
    "weak": Decimal("30"),
    # Below 30 is "insufficient"
}

# Compliance threshold: claims scoring below this are non-compliant.
COMPLIANCE_THRESHOLD: Decimal = Decimal("50")

# Required evidence types and lifecycle stages per claim type.
# Each claim type maps to a dict with:
#   - required_evidence: list of EvidenceType values needed
#   - required_lifecycle_stages: list of LifecycleStage values needed
#   - risk_level: inherent risk level of the claim
CLAIM_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    ClaimType.CARBON_NEUTRAL.value: {
        "required_evidence": [
            EvidenceType.LCA_STUDY.value,
            EvidenceType.THIRD_PARTY_VERIFICATION.value,
            EvidenceType.CERTIFICATION.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.TRANSPORTATION.value,
            LifecycleStage.DISTRIBUTION.value,
            LifecycleStage.USE.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.CRITICAL.value,
    },
    ClaimType.CLIMATE_POSITIVE.value: {
        "required_evidence": [
            EvidenceType.LCA_STUDY.value,
            EvidenceType.THIRD_PARTY_VERIFICATION.value,
            EvidenceType.CERTIFICATION.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.TRANSPORTATION.value,
            LifecycleStage.DISTRIBUTION.value,
            LifecycleStage.USE.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.CRITICAL.value,
    },
    ClaimType.NET_ZERO.value: {
        "required_evidence": [
            EvidenceType.LCA_STUDY.value,
            EvidenceType.THIRD_PARTY_VERIFICATION.value,
            EvidenceType.AUDIT_REPORT.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.TRANSPORTATION.value,
            LifecycleStage.DISTRIBUTION.value,
            LifecycleStage.USE.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.CRITICAL.value,
    },
    ClaimType.CARBON_NEGATIVE.value: {
        "required_evidence": [
            EvidenceType.LCA_STUDY.value,
            EvidenceType.THIRD_PARTY_VERIFICATION.value,
            EvidenceType.CERTIFICATION.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.TRANSPORTATION.value,
            LifecycleStage.DISTRIBUTION.value,
            LifecycleStage.USE.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.CRITICAL.value,
    },
    ClaimType.ECO_FRIENDLY.value: {
        "required_evidence": [
            EvidenceType.LCA_STUDY.value,
            EvidenceType.TEST_REPORT.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.USE.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.HIGH.value,
    },
    ClaimType.SUSTAINABLE.value: {
        "required_evidence": [
            EvidenceType.LCA_STUDY.value,
            EvidenceType.AUDIT_REPORT.value,
            EvidenceType.THIRD_PARTY_VERIFICATION.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.TRANSPORTATION.value,
            LifecycleStage.USE.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.HIGH.value,
    },
    ClaimType.GREEN.value: {
        "required_evidence": [
            EvidenceType.LCA_STUDY.value,
            EvidenceType.TEST_REPORT.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.USE.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.HIGH.value,
    },
    ClaimType.RENEWABLE.value: {
        "required_evidence": [
            EvidenceType.CERTIFICATION.value,
            EvidenceType.TEST_REPORT.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
        ],
        "risk_level": ClaimRiskLevel.MEDIUM.value,
    },
    ClaimType.RECYCLABLE.value: {
        "required_evidence": [
            EvidenceType.TEST_REPORT.value,
            EvidenceType.CERTIFICATION.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.MEDIUM.value,
    },
    ClaimType.BIODEGRADABLE.value: {
        "required_evidence": [
            EvidenceType.TEST_REPORT.value,
            EvidenceType.CERTIFICATION.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.MEDIUM.value,
    },
    ClaimType.COMPOSTABLE.value: {
        "required_evidence": [
            EvidenceType.TEST_REPORT.value,
            EvidenceType.CERTIFICATION.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.MEDIUM.value,
    },
    ClaimType.PLASTIC_FREE.value: {
        "required_evidence": [
            EvidenceType.TEST_REPORT.value,
            EvidenceType.MEASUREMENT.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
        ],
        "risk_level": ClaimRiskLevel.LOW.value,
    },
    ClaimType.ZERO_WASTE.value: {
        "required_evidence": [
            EvidenceType.AUDIT_REPORT.value,
            EvidenceType.MEASUREMENT.value,
            EvidenceType.THIRD_PARTY_VERIFICATION.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.DISTRIBUTION.value,
            LifecycleStage.USE.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.HIGH.value,
    },
    ClaimType.LOW_CARBON.value: {
        "required_evidence": [
            EvidenceType.LCA_STUDY.value,
            EvidenceType.MEASUREMENT.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.TRANSPORTATION.value,
        ],
        "risk_level": ClaimRiskLevel.MEDIUM.value,
    },
    ClaimType.REDUCED_EMISSIONS.value: {
        "required_evidence": [
            EvidenceType.LCA_STUDY.value,
            EvidenceType.MEASUREMENT.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.TRANSPORTATION.value,
        ],
        "risk_level": ClaimRiskLevel.MEDIUM.value,
    },
    ClaimType.ENVIRONMENTALLY_FRIENDLY.value: {
        "required_evidence": [
            EvidenceType.LCA_STUDY.value,
            EvidenceType.TEST_REPORT.value,
            EvidenceType.THIRD_PARTY_VERIFICATION.value,
        ],
        "required_lifecycle_stages": [
            LifecycleStage.RAW_MATERIALS.value,
            LifecycleStage.MANUFACTURING.value,
            LifecycleStage.TRANSPORTATION.value,
            LifecycleStage.USE.value,
            LifecycleStage.END_OF_LIFE.value,
        ],
        "risk_level": ClaimRiskLevel.HIGH.value,
    },
}


# Human-readable descriptions for claim types.
CLAIM_TYPE_DESCRIPTIONS: Dict[str, str] = {
    ClaimType.CARBON_NEUTRAL.value: "Product or organisation has net zero carbon emissions through reduction and offsetting",
    ClaimType.CLIMATE_POSITIVE.value: "Product or organisation removes more CO2 than it emits",
    ClaimType.NET_ZERO.value: "Achieved science-based emission reductions with residual neutralisation",
    ClaimType.CARBON_NEGATIVE.value: "Removes more greenhouse gases from the atmosphere than emitted",
    ClaimType.ECO_FRIENDLY.value: "Product or process has reduced environmental impact",
    ClaimType.SUSTAINABLE.value: "Meets environmental, social, and economic sustainability criteria",
    ClaimType.GREEN.value: "General claim of environmental benefit or reduced harm",
    ClaimType.RENEWABLE.value: "Made from or powered by renewable resources",
    ClaimType.RECYCLABLE.value: "Product materials can be collected and reprocessed",
    ClaimType.BIODEGRADABLE.value: "Product decomposes naturally through biological processes",
    ClaimType.COMPOSTABLE.value: "Product breaks down in composting conditions per EN 13432 or similar",
    ClaimType.PLASTIC_FREE.value: "Product contains no plastic materials",
    ClaimType.ZERO_WASTE.value: "Operations or product lifecycle generates no waste to landfill",
    ClaimType.LOW_CARBON.value: "Product has lower carbon footprint compared to alternatives",
    ClaimType.REDUCED_EMISSIONS.value: "Product or process has quantifiably reduced emissions",
    ClaimType.ENVIRONMENTALLY_FRIENDLY.value: "General claim of environmental benefit across multiple dimensions",
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class EnvironmentalClaim(BaseModel):
    """An environmental claim subject to substantiation under the Directive.

    Represents a single claim made by a trader about a product or the
    organisation itself, including the scope and evidence references.
    """
    claim_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this claim",
    )
    claim_text: str = Field(
        ...,
        description="The verbatim text of the environmental claim",
        max_length=2000,
    )
    claim_type: ClaimType = Field(
        ...,
        description="Categorisation of the claim per Directive taxonomy",
    )
    product_or_org: str = Field(
        ...,
        description="Product name or organisation the claim applies to",
        max_length=500,
    )
    scope_description: str = Field(
        default="",
        description="Description of the claim scope and boundaries",
        max_length=2000,
    )
    lifecycle_stages_covered: List[str] = Field(
        default_factory=list,
        description="Lifecycle stages the claim covers (raw_materials, manufacturing, etc.)",
    )
    evidence_references: List[str] = Field(
        default_factory=list,
        description="List of evidence IDs supporting this claim",
    )

    @field_validator("claim_text")
    @classmethod
    def validate_claim_text_not_empty(cls, v: str) -> str:
        """Ensure claim text is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("Claim text must not be empty")
        return v


class ClaimEvidence(BaseModel):
    """A piece of evidence supporting an environmental claim.

    Per Article 3(2), evidence must be based on widely recognised
    scientific methods and independently verifiable data.
    """
    evidence_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this evidence",
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="Type of evidence (certification, LCA study, etc.)",
    )
    source: str = Field(
        ...,
        description="Source or issuing organisation of the evidence",
        max_length=500,
    )
    description: str = Field(
        default="",
        description="Description of what the evidence demonstrates",
        max_length=2000,
    )
    is_third_party: bool = Field(
        default=False,
        description="Whether the evidence was produced by an independent third party",
    )
    valid_from: Optional[str] = Field(
        default=None,
        description="Start date of evidence validity (ISO 8601 date string)",
        max_length=10,
    )
    valid_to: Optional[str] = Field(
        default=None,
        description="End date of evidence validity (ISO 8601 date string)",
        max_length=10,
    )
    accreditation_body: str = Field(
        default="",
        description="Accreditation body for the evidence issuer",
        max_length=500,
    )
    methodology_reference: str = Field(
        default="",
        description="Reference to the methodology or standard used",
        max_length=500,
    )
    data_quality_score: Decimal = Field(
        default=Decimal("0"),
        description="Data quality score (0-100 scale)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )


class SubstantiationAssessment(BaseModel):
    """Result of a substantiation assessment for an environmental claim.

    Contains the weighted scores across all five substantiation dimensions,
    overall compliance status, and recommendations for improvement.
    """
    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique assessment identifier",
    )
    claim_id: str = Field(
        ...,
        description="ID of the assessed claim",
    )
    overall_score: Decimal = Field(
        default=Decimal("0.00"),
        description="Overall substantiation score (0-100 scale)",
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    level: SubstantiationLevel = Field(
        default=SubstantiationLevel.INSUFFICIENT,
        description="Substantiation quality level",
    )
    dimension_scores: Dict[str, str] = Field(
        default_factory=dict,
        description="Scores per dimension (scientific_validity, data_quality, "
                    "scope_completeness, verification_independence, transparency)",
    )
    compliant: bool = Field(
        default=False,
        description="Whether the claim meets minimum compliance threshold",
    )
    risk_level: str = Field(
        default="",
        description="Inherent risk level of the claim type",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of identified issues with the substantiation",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving substantiation",
    )
    evidence_count: int = Field(
        default=0,
        description="Number of evidence items assessed",
    )
    lifecycle_coverage_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of required lifecycle stages covered",
    )
    evidence_type_coverage_pct: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of required evidence types provided",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Assessment timestamp (UTC)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the assessment result",
    )


class ClaimCompletenessResult(BaseModel):
    """Result of a completeness validation across multiple claims.

    Evaluates whether a set of claims covers the necessary evidence
    types, lifecycle stages, and substantiation depth.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    total_claims: int = Field(
        default=0,
        description="Total number of claims assessed",
    )
    compliant_claims: int = Field(
        default=0,
        description="Number of claims meeting compliance threshold",
    )
    non_compliant_claims: int = Field(
        default=0,
        description="Number of claims failing compliance threshold",
    )
    overall_compliance_rate: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of claims that are compliant (0-100)",
    )
    average_substantiation_score: Decimal = Field(
        default=Decimal("0.00"),
        description="Average substantiation score across all claims",
    )
    claims_by_risk_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of claims per risk level",
    )
    claims_by_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of claims per substantiation level",
    )
    high_priority_issues: List[str] = Field(
        default_factory=list,
        description="Issues from critical/high-risk claims",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Calculation timestamp (UTC)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the result",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ClaimSubstantiationEngine:
    """Substantiation assessment engine per EU Green Claims Directive Art. 3-4.

    Provides deterministic, zero-hallucination assessment of environmental
    claims against the substantiation requirements of the EU Green Claims
    Directive.  Evaluates claims across five dimensions:

    1. Scientific Validity (30%): Is the claim backed by scientific evidence?
    2. Data Quality (25%): Is the underlying data robust and primary?
    3. Scope Completeness (20%): Does the claim cover required lifecycle stages?
    4. Verification Independence (15%): Is evidence from independent parties?
    5. Transparency (10%): Is the claim transparent and non-misleading?

    All calculations use Decimal arithmetic with ROUND_HALF_UP rounding.
    Every result includes a SHA-256 provenance hash for audit trail.

    Usage::

        engine = ClaimSubstantiationEngine()
        claim = EnvironmentalClaim(
            claim_text="This product is carbon neutral",
            claim_type=ClaimType.CARBON_NEUTRAL,
            product_or_org="Widget X",
        )
        evidence = [
            ClaimEvidence(
                evidence_type=EvidenceType.LCA_STUDY,
                source="Accredited Lab",
                is_third_party=True,
            ),
        ]
        result = engine.assess_claim(claim, evidence)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise ClaimSubstantiationEngine."""
        logger.info(
            "ClaimSubstantiationEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Assess Claim                                                         #
    # ------------------------------------------------------------------ #

    def assess_claim(
        self,
        claim: EnvironmentalClaim,
        evidence_list: List[ClaimEvidence],
    ) -> Dict[str, Any]:
        """Assess an environmental claim against substantiation requirements.

        Performs a comprehensive assessment of the claim by evaluating
        all five substantiation dimensions and checking compliance
        against the claim-type-specific requirements.

        Args:
            claim: The environmental claim to assess.
            evidence_list: List of evidence items supporting the claim.

        Returns:
            Dict with keys: assessment (SubstantiationAssessment),
            provenance_hash (str).
        """
        t0 = time.perf_counter()

        # Calculate dimension scores
        scientific_score = self._score_scientific_validity(
            claim, evidence_list
        )
        data_quality_score = self._score_data_quality(evidence_list)
        scope_score = self._score_scope_completeness(claim)
        verification_score = self._score_verification_independence(
            evidence_list
        )
        transparency_score = self._score_transparency(claim, evidence_list)

        dimension_scores: Dict[str, Decimal] = {
            "scientific_validity": scientific_score,
            "data_quality": data_quality_score,
            "scope_completeness": scope_score,
            "verification_independence": verification_score,
            "transparency": transparency_score,
        }

        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(dimension_scores)

        # Determine substantiation level
        level = self._determine_level(overall_score)

        # Check compliance
        compliant = overall_score >= COMPLIANCE_THRESHOLD

        # Get risk level from claim requirements
        requirements = CLAIM_REQUIREMENTS.get(claim.claim_type.value, {})
        risk_level = requirements.get(
            "risk_level", ClaimRiskLevel.MEDIUM.value
        )

        # Identify issues and recommendations
        issues = self._identify_issues(
            claim, evidence_list, dimension_scores
        )
        recommendations = self._generate_recommendations(
            claim, evidence_list, dimension_scores, issues
        )

        # Calculate coverage metrics
        lifecycle_coverage = self._calculate_lifecycle_coverage(claim)
        evidence_type_coverage = self._calculate_evidence_type_coverage(
            claim, evidence_list
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        assessment = SubstantiationAssessment(
            claim_id=claim.claim_id,
            overall_score=_round_val(overall_score, 2),
            level=level,
            dimension_scores={
                k: str(_round_val(v, 2)) for k, v in dimension_scores.items()
            },
            compliant=compliant,
            risk_level=risk_level,
            issues=issues,
            recommendations=recommendations,
            evidence_count=len(evidence_list),
            lifecycle_coverage_pct=_round_val(lifecycle_coverage, 2),
            evidence_type_coverage_pct=_round_val(evidence_type_coverage, 2),
            processing_time_ms=elapsed_ms,
        )
        assessment.provenance_hash = _compute_hash(assessment)

        logger.info(
            "Assessed claim '%s' (type=%s): score=%s, level=%s, "
            "compliant=%s, risk=%s in %.3f ms",
            claim.claim_id,
            claim.claim_type.value,
            assessment.overall_score,
            level.value,
            compliant,
            risk_level,
            elapsed_ms,
        )

        return {
            "assessment": assessment,
            "provenance_hash": assessment.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Calculate Substantiation Score                                        #
    # ------------------------------------------------------------------ #

    def calculate_substantiation_score(
        self,
        claim: EnvironmentalClaim,
        evidence: List[ClaimEvidence],
    ) -> Dict[str, Any]:
        """Calculate the substantiation score for a claim.

        This is a focused calculation that returns only the scoring
        breakdown without the full compliance assessment.

        Args:
            claim: The environmental claim.
            evidence: List of evidence items.

        Returns:
            Dict with keys: overall_score (Decimal), dimension_scores (dict),
            level (str), provenance_hash (str).
        """
        t0 = time.perf_counter()

        dimension_scores: Dict[str, Decimal] = {
            "scientific_validity": self._score_scientific_validity(
                claim, evidence
            ),
            "data_quality": self._score_data_quality(evidence),
            "scope_completeness": self._score_scope_completeness(claim),
            "verification_independence": self._score_verification_independence(
                evidence
            ),
            "transparency": self._score_transparency(claim, evidence),
        }

        overall_score = self._calculate_weighted_score(dimension_scores)
        level = self._determine_level(overall_score)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = {
            "overall_score": str(_round_val(overall_score, 2)),
            "dimension_scores": {
                k: str(_round_val(v, 2)) for k, v in dimension_scores.items()
            },
            "level": level.value,
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Calculated substantiation score for claim '%s': %s (%s) in %.3f ms",
            claim.claim_id,
            result["overall_score"],
            level.value,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Check Compliance                                                      #
    # ------------------------------------------------------------------ #

    def check_compliance(
        self,
        claim: EnvironmentalClaim,
        evidence: List[ClaimEvidence],
    ) -> Dict[str, Any]:
        """Check whether a claim meets compliance requirements.

        Evaluates the claim against Article 3-4 requirements including
        evidence type coverage, lifecycle stage coverage, and minimum
        substantiation score.

        Args:
            claim: The environmental claim to check.
            evidence: List of evidence items.

        Returns:
            Dict with keys: compliant (bool), score (Decimal),
            issues (list), evidence_coverage (dict),
            lifecycle_coverage (dict), provenance_hash (str).
        """
        t0 = time.perf_counter()

        requirements = CLAIM_REQUIREMENTS.get(claim.claim_type.value, {})
        required_evidence_types = requirements.get("required_evidence", [])
        required_lifecycle = requirements.get("required_lifecycle_stages", [])
        risk_level = requirements.get(
            "risk_level", ClaimRiskLevel.MEDIUM.value
        )

        # Check evidence type coverage
        provided_evidence_types = {
            e.evidence_type.value for e in evidence
        }
        missing_evidence_types = [
            et for et in required_evidence_types
            if et not in provided_evidence_types
        ]
        evidence_coverage_pct = _safe_divide(
            _decimal(len(required_evidence_types) - len(missing_evidence_types)),
            _decimal(len(required_evidence_types)) if required_evidence_types else Decimal("1"),
        ) * Decimal("100")

        # Check lifecycle stage coverage
        provided_stages = set(claim.lifecycle_stages_covered)
        missing_stages = [
            ls for ls in required_lifecycle if ls not in provided_stages
        ]
        lifecycle_coverage_pct = _safe_divide(
            _decimal(len(required_lifecycle) - len(missing_stages)),
            _decimal(len(required_lifecycle)) if required_lifecycle else Decimal("1"),
        ) * Decimal("100")

        # Check third-party verification for critical/high risk
        has_third_party = any(e.is_third_party for e in evidence)

        # Check evidence validity dates
        expired_evidence = self._find_expired_evidence(evidence)

        # Build issues list
        issues: List[str] = []
        if missing_evidence_types:
            issues.append(
                f"Missing required evidence types: {', '.join(missing_evidence_types)}"
            )
        if missing_stages:
            issues.append(
                f"Missing required lifecycle stages: {', '.join(missing_stages)}"
            )
        if risk_level in (ClaimRiskLevel.CRITICAL.value, ClaimRiskLevel.HIGH.value):
            if not has_third_party:
                issues.append(
                    "High/critical risk claim requires third-party verification"
                )
        if expired_evidence:
            issues.append(
                f"{len(expired_evidence)} evidence item(s) have expired validity dates"
            )
        if not evidence:
            issues.append("No evidence provided for the claim")

        # Calculate overall score for compliance check
        dimension_scores: Dict[str, Decimal] = {
            "scientific_validity": self._score_scientific_validity(
                claim, evidence
            ),
            "data_quality": self._score_data_quality(evidence),
            "scope_completeness": self._score_scope_completeness(claim),
            "verification_independence": self._score_verification_independence(
                evidence
            ),
            "transparency": self._score_transparency(claim, evidence),
        }
        overall_score = self._calculate_weighted_score(dimension_scores)

        compliant = (
            overall_score >= COMPLIANCE_THRESHOLD
            and not missing_evidence_types
            and not missing_stages
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = {
            "compliant": compliant,
            "score": str(_round_val(overall_score, 2)),
            "risk_level": risk_level,
            "issues": issues,
            "evidence_coverage": {
                "required": required_evidence_types,
                "provided": sorted(provided_evidence_types),
                "missing": missing_evidence_types,
                "coverage_pct": str(_round_val(evidence_coverage_pct, 2)),
            },
            "lifecycle_coverage": {
                "required": required_lifecycle,
                "provided": sorted(provided_stages),
                "missing": missing_stages,
                "coverage_pct": str(_round_val(lifecycle_coverage_pct, 2)),
            },
            "has_third_party_verification": has_third_party,
            "expired_evidence_count": len(expired_evidence),
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Compliance check for claim '%s': compliant=%s, score=%s, "
            "%d issue(s) in %.3f ms",
            claim.claim_id,
            compliant,
            result["score"],
            len(issues),
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Validate Claim Completeness                                           #
    # ------------------------------------------------------------------ #

    def validate_claim_completeness(
        self,
        claims_list: List[Tuple[EnvironmentalClaim, List[ClaimEvidence]]],
    ) -> Dict[str, Any]:
        """Validate completeness across a portfolio of environmental claims.

        Assesses each claim and produces an aggregated completeness
        report with compliance rates, risk distributions, and
        high-priority issues.

        Args:
            claims_list: List of (claim, evidence_list) tuples.

        Returns:
            Dict with keys: result (ClaimCompletenessResult),
            claim_assessments (list), provenance_hash (str).
        """
        t0 = time.perf_counter()

        assessments: List[Dict[str, Any]] = []
        scores: List[Decimal] = []
        compliant_count = 0
        non_compliant_count = 0
        by_risk: Dict[str, int] = {}
        by_level: Dict[str, int] = {}
        high_priority_issues: List[str] = []

        for claim, evidence in claims_list:
            assessment_result = self.assess_claim(claim, evidence)
            assessment: SubstantiationAssessment = assessment_result["assessment"]
            assessments.append({
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type.value,
                "score": str(assessment.overall_score),
                "level": assessment.level.value,
                "compliant": assessment.compliant,
                "risk_level": assessment.risk_level,
                "issues_count": len(assessment.issues),
            })

            scores.append(assessment.overall_score)

            if assessment.compliant:
                compliant_count += 1
            else:
                non_compliant_count += 1

            # Risk level distribution
            rl = assessment.risk_level
            by_risk[rl] = by_risk.get(rl, 0) + 1

            # Substantiation level distribution
            lvl = assessment.level.value
            by_level[lvl] = by_level.get(lvl, 0) + 1

            # Collect high-priority issues from critical/high risk claims
            if rl in (ClaimRiskLevel.CRITICAL.value, ClaimRiskLevel.HIGH.value):
                for issue in assessment.issues:
                    high_priority_issues.append(
                        f"[{claim.claim_type.value}] {issue}"
                    )

        total = len(claims_list)
        avg_score = Decimal("0.00")
        compliance_rate = Decimal("0.00")
        if total > 0:
            avg_score = _round_val(
                sum(scores) / _decimal(total), 2
            )
            compliance_rate = _round_val(
                _decimal(compliant_count) / _decimal(total) * Decimal("100"), 2
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        completeness_result = ClaimCompletenessResult(
            total_claims=total,
            compliant_claims=compliant_count,
            non_compliant_claims=non_compliant_count,
            overall_compliance_rate=compliance_rate,
            average_substantiation_score=avg_score,
            claims_by_risk_level=by_risk,
            claims_by_level=by_level,
            high_priority_issues=high_priority_issues,
            processing_time_ms=elapsed_ms,
        )
        completeness_result.provenance_hash = _compute_hash(
            completeness_result
        )

        logger.info(
            "Validated %d claims: %d compliant, %d non-compliant, "
            "avg score=%s, compliance rate=%s%% in %.3f ms",
            total,
            compliant_count,
            non_compliant_count,
            avg_score,
            compliance_rate,
            elapsed_ms,
        )

        return {
            "result": completeness_result,
            "claim_assessments": assessments,
            "provenance_hash": completeness_result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Private Scoring Methods                                               #
    # ------------------------------------------------------------------ #

    def _score_scientific_validity(
        self,
        claim: EnvironmentalClaim,
        evidence: List[ClaimEvidence],
    ) -> Decimal:
        """Score the scientific validity of evidence for a claim (0-100).

        Evaluates presence of LCA studies, test reports, certifications,
        and third-party verified evidence.  Higher scores for claims
        backed by accredited, peer-reviewed scientific data.

        Args:
            claim: The environmental claim.
            evidence: List of evidence items.

        Returns:
            Scientific validity score (Decimal, 0-100).
        """
        if not evidence:
            return Decimal("0")

        score = Decimal("0")
        max_score = Decimal("100")

        # LCA study presence (up to 30 points)
        has_lca = any(
            e.evidence_type == EvidenceType.LCA_STUDY for e in evidence
        )
        if has_lca:
            score += Decimal("30")

        # Certification presence (up to 20 points)
        has_cert = any(
            e.evidence_type == EvidenceType.CERTIFICATION for e in evidence
        )
        if has_cert:
            score += Decimal("20")

        # Test report presence (up to 15 points)
        has_test = any(
            e.evidence_type == EvidenceType.TEST_REPORT for e in evidence
        )
        if has_test:
            score += Decimal("15")

        # Third-party verification (up to 20 points)
        third_party_count = sum(1 for e in evidence if e.is_third_party)
        if third_party_count > 0:
            tp_ratio = _safe_divide(
                _decimal(third_party_count), _decimal(len(evidence))
            )
            score += _round_val(tp_ratio * Decimal("20"), 2)

        # Methodology references present (up to 15 points)
        has_methodology = any(
            e.methodology_reference.strip() for e in evidence
        )
        if has_methodology:
            score += Decimal("15")

        return min(score, max_score)

    def _score_data_quality(
        self, evidence: List[ClaimEvidence]
    ) -> Decimal:
        """Score the data quality of evidence items (0-100).

        Evaluates evidence data quality scores, recency, and
        completeness of supporting documentation.

        Args:
            evidence: List of evidence items.

        Returns:
            Data quality score (Decimal, 0-100).
        """
        if not evidence:
            return Decimal("0")

        score = Decimal("0")
        max_score = Decimal("100")

        # Average data quality scores from evidence items (up to 40 points)
        quality_scores = [
            e.data_quality_score for e in evidence
            if e.data_quality_score > Decimal("0")
        ]
        if quality_scores:
            avg_quality = sum(quality_scores) / _decimal(len(quality_scores))
            score += _round_val(avg_quality * Decimal("0.4"), 2)

        # Evidence with descriptions (up to 20 points)
        described_count = sum(
            1 for e in evidence if e.description.strip()
        )
        desc_ratio = _safe_divide(
            _decimal(described_count), _decimal(len(evidence))
        )
        score += _round_val(desc_ratio * Decimal("20"), 2)

        # Evidence with valid dates (up to 20 points)
        dated_count = sum(
            1 for e in evidence if e.valid_from and e.valid_to
        )
        date_ratio = _safe_divide(
            _decimal(dated_count), _decimal(len(evidence))
        )
        score += _round_val(date_ratio * Decimal("20"), 2)

        # Multiple evidence types (up to 20 points)
        unique_types = len({e.evidence_type for e in evidence})
        type_diversity = min(
            _decimal(unique_types) / Decimal("3"), Decimal("1")
        )
        score += _round_val(type_diversity * Decimal("20"), 2)

        return min(score, max_score)

    def _score_scope_completeness(
        self, claim: EnvironmentalClaim
    ) -> Decimal:
        """Score the scope completeness of a claim (0-100).

        Evaluates whether the claim covers all required lifecycle
        stages for the given claim type.

        Args:
            claim: The environmental claim.

        Returns:
            Scope completeness score (Decimal, 0-100).
        """
        requirements = CLAIM_REQUIREMENTS.get(claim.claim_type.value, {})
        required_stages = requirements.get("required_lifecycle_stages", [])

        if not required_stages:
            # If no specific stages required, base score on coverage count
            if claim.lifecycle_stages_covered:
                return min(
                    _decimal(len(claim.lifecycle_stages_covered))
                    / Decimal("6") * Decimal("100"),
                    Decimal("100"),
                )
            return Decimal("0")

        provided = set(claim.lifecycle_stages_covered)
        required = set(required_stages)

        if not required:
            return Decimal("100")

        covered = required.intersection(provided)
        coverage_ratio = _safe_divide(
            _decimal(len(covered)), _decimal(len(required))
        )

        # Base score from coverage ratio (up to 80 points)
        score = _round_val(coverage_ratio * Decimal("80"), 2)

        # Bonus for covering additional stages beyond required (up to 20)
        extra_stages = provided - required
        if extra_stages:
            bonus = min(
                _decimal(len(extra_stages)) * Decimal("5"), Decimal("20")
            )
            score += bonus

        return min(score, Decimal("100"))

    def _score_verification_independence(
        self, evidence: List[ClaimEvidence]
    ) -> Decimal:
        """Score the independence of verification (0-100).

        Evaluates the proportion of third-party evidence and the
        presence of accredited verification bodies.

        Args:
            evidence: List of evidence items.

        Returns:
            Verification independence score (Decimal, 0-100).
        """
        if not evidence:
            return Decimal("0")

        score = Decimal("0")
        max_score = Decimal("100")

        # Third-party ratio (up to 50 points)
        third_party_count = sum(1 for e in evidence if e.is_third_party)
        tp_ratio = _safe_divide(
            _decimal(third_party_count), _decimal(len(evidence))
        )
        score += _round_val(tp_ratio * Decimal("50"), 2)

        # Has third-party verification evidence type (up to 20 points)
        has_tpv = any(
            e.evidence_type == EvidenceType.THIRD_PARTY_VERIFICATION
            for e in evidence
        )
        if has_tpv:
            score += Decimal("20")

        # Accreditation body present (up to 20 points)
        has_accreditation = any(
            e.accreditation_body.strip() for e in evidence
        )
        if has_accreditation:
            score += Decimal("20")

        # Audit report present (up to 10 points)
        has_audit = any(
            e.evidence_type == EvidenceType.AUDIT_REPORT for e in evidence
        )
        if has_audit:
            score += Decimal("10")

        return min(score, max_score)

    def _score_transparency(
        self,
        claim: EnvironmentalClaim,
        evidence: List[ClaimEvidence],
    ) -> Decimal:
        """Score the transparency of a claim (0-100).

        Evaluates whether the claim is clear, specific, and provides
        accessible substantiation per Article 4 requirements.

        Args:
            claim: The environmental claim.
            evidence: List of evidence items.

        Returns:
            Transparency score (Decimal, 0-100).
        """
        score = Decimal("0")
        max_score = Decimal("100")

        # Scope description provided (up to 25 points)
        if claim.scope_description.strip():
            score += Decimal("25")

        # Lifecycle stages explicitly stated (up to 25 points)
        if claim.lifecycle_stages_covered:
            coverage = min(
                _decimal(len(claim.lifecycle_stages_covered)) / Decimal("4"),
                Decimal("1"),
            )
            score += _round_val(coverage * Decimal("25"), 2)

        # Evidence references provided (up to 25 points)
        if claim.evidence_references:
            ref_score = min(
                _decimal(len(claim.evidence_references)) / Decimal("3"),
                Decimal("1"),
            )
            score += _round_val(ref_score * Decimal("25"), 2)

        # Evidence sources are specific and named (up to 25 points)
        if evidence:
            sourced_count = sum(
                1 for e in evidence if e.source.strip()
            )
            source_ratio = _safe_divide(
                _decimal(sourced_count), _decimal(len(evidence))
            )
            score += _round_val(source_ratio * Decimal("25"), 2)

        return min(score, max_score)

    # ------------------------------------------------------------------ #
    # Private Utility Methods                                               #
    # ------------------------------------------------------------------ #

    def _calculate_weighted_score(
        self, dimension_scores: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate the weighted overall score from dimension scores.

        Formula:
            overall = sum(dimension_score * weight / 100) for each dimension

        The weights are defined in DIMENSION_WEIGHTS and sum to 100.

        Args:
            dimension_scores: Dict mapping dimension name to score (0-100).

        Returns:
            Weighted overall score (Decimal, 0-100).
        """
        total = Decimal("0")
        for dimension, weight in DIMENSION_WEIGHTS.items():
            dim_score = dimension_scores.get(dimension, Decimal("0"))
            contribution = dim_score * weight / Decimal("100")
            total += contribution
        return _round_val(total, 2)

    def _determine_level(
        self, score: Decimal
    ) -> SubstantiationLevel:
        """Determine the substantiation level from a score.

        Args:
            score: Overall substantiation score (0-100).

        Returns:
            SubstantiationLevel enum value.
        """
        if score >= SUBSTANTIATION_THRESHOLDS["excellent"]:
            return SubstantiationLevel.EXCELLENT
        if score >= SUBSTANTIATION_THRESHOLDS["good"]:
            return SubstantiationLevel.GOOD
        if score >= SUBSTANTIATION_THRESHOLDS["moderate"]:
            return SubstantiationLevel.MODERATE
        if score >= SUBSTANTIATION_THRESHOLDS["weak"]:
            return SubstantiationLevel.WEAK
        return SubstantiationLevel.INSUFFICIENT

    def _identify_issues(
        self,
        claim: EnvironmentalClaim,
        evidence: List[ClaimEvidence],
        dimension_scores: Dict[str, Decimal],
    ) -> List[str]:
        """Identify substantiation issues for a claim.

        Args:
            claim: The environmental claim.
            evidence: List of evidence items.
            dimension_scores: Calculated dimension scores.

        Returns:
            List of issue descriptions.
        """
        issues: List[str] = []

        # No evidence at all
        if not evidence:
            issues.append(
                "No evidence provided to substantiate the claim"
            )
            return issues

        # Low scientific validity
        if dimension_scores.get(
            "scientific_validity", Decimal("0")
        ) < Decimal("40"):
            issues.append(
                "Scientific validity is weak - no LCA study or "
                "certified evidence found"
            )

        # Low data quality
        if dimension_scores.get(
            "data_quality", Decimal("0")
        ) < Decimal("40"):
            issues.append(
                "Data quality is insufficient - evidence lacks quality "
                "scores, descriptions, or validity dates"
            )

        # Incomplete scope
        if dimension_scores.get(
            "scope_completeness", Decimal("0")
        ) < Decimal("60"):
            issues.append(
                "Claim scope does not cover all required lifecycle stages "
                "for this claim type"
            )

        # Weak verification independence
        if dimension_scores.get(
            "verification_independence", Decimal("0")
        ) < Decimal("40"):
            issues.append(
                "Insufficient independent verification - most evidence "
                "is self-declared or unverified"
            )

        # Low transparency
        if dimension_scores.get(
            "transparency", Decimal("0")
        ) < Decimal("40"):
            issues.append(
                "Claim lacks transparency - missing scope description, "
                "lifecycle coverage, or evidence references"
            )

        # Check for expired evidence
        expired = self._find_expired_evidence(evidence)
        if expired:
            issues.append(
                f"{len(expired)} evidence item(s) have expired"
            )

        # Claim-type specific checks
        requirements = CLAIM_REQUIREMENTS.get(claim.claim_type.value, {})
        risk = requirements.get("risk_level", "")

        if risk == ClaimRiskLevel.CRITICAL.value:
            if not any(e.is_third_party for e in evidence):
                issues.append(
                    "Critical-risk claim requires at least one "
                    "third-party verified evidence item"
                )
            if not any(
                e.evidence_type == EvidenceType.LCA_STUDY for e in evidence
            ):
                issues.append(
                    "Critical-risk claim requires a lifecycle "
                    "assessment (LCA) study"
                )

        return issues

    def _generate_recommendations(
        self,
        claim: EnvironmentalClaim,
        evidence: List[ClaimEvidence],
        dimension_scores: Dict[str, Decimal],
        issues: List[str],
    ) -> List[str]:
        """Generate improvement recommendations for a claim.

        Args:
            claim: The environmental claim.
            evidence: List of evidence items.
            dimension_scores: Calculated dimension scores.
            issues: Previously identified issues.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if not evidence:
            recommendations.append(
                "Obtain substantiating evidence before making "
                "this environmental claim"
            )
            return recommendations

        # Recommendations based on dimension scores
        sci_score = dimension_scores.get(
            "scientific_validity", Decimal("0")
        )
        if sci_score < Decimal("70"):
            recommendations.append(
                "Commission a lifecycle assessment (LCA) study "
                "from an accredited laboratory to strengthen "
                "scientific validity"
            )

        dq_score = dimension_scores.get("data_quality", Decimal("0"))
        if dq_score < Decimal("70"):
            recommendations.append(
                "Improve data quality by adding detailed descriptions, "
                "validity dates, and quality scores to all evidence items"
            )

        scope_score = dimension_scores.get(
            "scope_completeness", Decimal("0")
        )
        if scope_score < Decimal("70"):
            requirements = CLAIM_REQUIREMENTS.get(
                claim.claim_type.value, {}
            )
            missing = set(
                requirements.get("required_lifecycle_stages", [])
            ) - set(claim.lifecycle_stages_covered)
            if missing:
                recommendations.append(
                    f"Extend claim scope to cover missing lifecycle stages: "
                    f"{', '.join(sorted(missing))}"
                )

        verif_score = dimension_scores.get(
            "verification_independence", Decimal("0")
        )
        if verif_score < Decimal("70"):
            recommendations.append(
                "Engage an independent, accredited third-party verifier "
                "to strengthen verification independence"
            )

        trans_score = dimension_scores.get(
            "transparency", Decimal("0")
        )
        if trans_score < Decimal("70"):
            recommendations.append(
                "Add a clear scope description and explicit lifecycle "
                "stage coverage to improve claim transparency"
            )

        return recommendations

    def _calculate_lifecycle_coverage(
        self, claim: EnvironmentalClaim
    ) -> Decimal:
        """Calculate lifecycle stage coverage percentage.

        Args:
            claim: The environmental claim.

        Returns:
            Coverage percentage (Decimal, 0-100).
        """
        requirements = CLAIM_REQUIREMENTS.get(claim.claim_type.value, {})
        required = requirements.get("required_lifecycle_stages", [])

        if not required:
            return Decimal("100") if claim.lifecycle_stages_covered else Decimal("0")

        provided = set(claim.lifecycle_stages_covered)
        covered = set(required).intersection(provided)

        return _safe_divide(
            _decimal(len(covered)) * Decimal("100"),
            _decimal(len(required)),
        )

    def _calculate_evidence_type_coverage(
        self,
        claim: EnvironmentalClaim,
        evidence: List[ClaimEvidence],
    ) -> Decimal:
        """Calculate evidence type coverage percentage.

        Args:
            claim: The environmental claim.
            evidence: List of evidence items.

        Returns:
            Coverage percentage (Decimal, 0-100).
        """
        requirements = CLAIM_REQUIREMENTS.get(claim.claim_type.value, {})
        required = requirements.get("required_evidence", [])

        if not required:
            return Decimal("100") if evidence else Decimal("0")

        provided = {e.evidence_type.value for e in evidence}
        covered = set(required).intersection(provided)

        return _safe_divide(
            _decimal(len(covered)) * Decimal("100"),
            _decimal(len(required)),
        )

    def _find_expired_evidence(
        self, evidence: List[ClaimEvidence]
    ) -> List[ClaimEvidence]:
        """Find evidence items with expired validity dates.

        Args:
            evidence: List of evidence items.

        Returns:
            List of expired evidence items.
        """
        today_str = date.today().isoformat()
        expired: List[ClaimEvidence] = []
        for e in evidence:
            if e.valid_to and e.valid_to < today_str:
                expired.append(e)
        return expired
