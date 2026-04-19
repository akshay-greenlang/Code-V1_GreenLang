# -*- coding: utf-8 -*-
"""
TraderObligationEngine - PACK-018 EU Green Claims Prep Engine 7
=================================================================

Tracks trader compliance with Articles 3 through 8 of the proposed
EU Green Claims Directive (COM/2023/166), managing the lifecycle of
environmental claims from initial drafting through substantiation,
verification, publication, and eventual expiry or withdrawal.

The Green Claims Directive imposes specific obligations on traders
who make environmental claims.  This engine assesses compliance
against each article's requirements and generates a comprehensive
compliance profile with remediation timelines.

Article 3 - Substantiation Requirements:
    - Environmental claims shall be substantiated by widely
      recognised scientific evidence and accurate information.
    - Substantiation shall demonstrate significance from a lifecycle
      perspective.
    - All significant environmental aspects and impacts shall be
      identified where the claim relates to a product.
    - Substantiation shall include primary and/or robust secondary
      data for lifecycle stages.

Article 4 - Communication Requirements:
    - Claims shall not be misleading.
    - Aggregated indicators shall be based on sufficient evidence.
    - Claims shall not exaggerate environmental benefits.
    - Future performance claims shall include time-bound
      commitments and measurable targets.

Article 5 - Comparative Claims:
    - Equivalent information and data for compared items.
    - Equivalent system boundaries and functional units.

Article 6 - Environmental Labelling:
    - Labels shall be based on recognised certification schemes
      or established by public authorities.

Article 7 - Verification:
    - Claims shall be verified by independent accredited verifiers
      before being communicated to consumers.

Article 8 - Record Keeping:
    - Traders shall maintain records of substantiation and
      verification for at least 5 years.

Regulatory References:
    - EU Green Claims Directive Proposal COM/2023/166
    - Empowering Consumers Directive 2024/825
    - Regulation (EC) No 765/2008 (Accreditation)

Zero-Hallucination:
    - Compliance checks use deterministic rule-based evaluation
    - Obligation scoring uses fixed-weight criteria checklists
    - Lifecycle tracking uses state machine transitions
    - SHA-256 provenance hash on every result
    - No LLM involvement in any assessment path

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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

__all__ = [
    "TraderObligationEngine",
    "ArticleReference",
    "ComplianceStatus",
    "ObligationItem",
    "ClaimLifecycleStage",
    "TraderObligationResult",
    "ClaimLifecycleRecord",
    "ARTICLE_3_CHECKLIST",
    "RECORD_RETENTION_YEARS",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class ArticleReference(str, Enum):
    """Green Claims Directive article references.

    Each article defines a specific set of obligations for traders
    making environmental claims.
    """
    ARTICLE_3 = "article_3"
    ARTICLE_4 = "article_4"
    ARTICLE_5 = "article_5"
    ARTICLE_6 = "article_6"
    ARTICLE_7 = "article_7"
    ARTICLE_8 = "article_8"

class ComplianceStatus(str, Enum):
    """Compliance status for an individual obligation.

    Reflects whether the trader has met the specific requirement
    of the Directive.
    """
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"

class ClaimLifecycleStage(str, Enum):
    """Lifecycle stages of an environmental claim.

    Tracks the progression of a claim from initial drafting through
    to publication, expiry, or withdrawal.
    """
    DRAFT = "draft"
    SUBSTANTIATED = "substantiated"
    SUBMITTED_FOR_VERIFICATION = "submitted_for_verification"
    VERIFIED = "verified"
    PUBLISHED = "published"
    EXPIRED = "expired"
    WITHDRAWN = "withdrawn"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Article 3 compliance checklist per COM/2023/166.
ARTICLE_3_CHECKLIST: List[str] = [
    "Claim substantiated by widely recognised scientific evidence",
    "Substantiation uses accurate and up-to-date information",
    "Substantiation takes into account relevant international standards",
    "Significance demonstrated from a lifecycle perspective",
    "All significant environmental aspects identified for product claims",
    "Primary and/or robust secondary data included for lifecycle stages",
    "Environmental impacts quantified with recognised methodology",
    "Substantiation considers the full value chain where relevant",
]

# Record retention period in years per Article 8.
RECORD_RETENTION_YEARS: int = 5

# Article-level weights for overall compliance score.
ARTICLE_WEIGHTS: Dict[str, Decimal] = {
    ArticleReference.ARTICLE_3.value: Decimal("30"),
    ArticleReference.ARTICLE_4.value: Decimal("20"),
    ArticleReference.ARTICLE_5.value: Decimal("10"),
    ArticleReference.ARTICLE_6.value: Decimal("10"),
    ArticleReference.ARTICLE_7.value: Decimal("20"),
    ArticleReference.ARTICLE_8.value: Decimal("10"),
}

# Valid lifecycle stage transitions.
VALID_TRANSITIONS: Dict[str, List[str]] = {
    ClaimLifecycleStage.DRAFT.value: [
        ClaimLifecycleStage.SUBSTANTIATED.value,
        ClaimLifecycleStage.WITHDRAWN.value,
    ],
    ClaimLifecycleStage.SUBSTANTIATED.value: [
        ClaimLifecycleStage.SUBMITTED_FOR_VERIFICATION.value,
        ClaimLifecycleStage.WITHDRAWN.value,
    ],
    ClaimLifecycleStage.SUBMITTED_FOR_VERIFICATION.value: [
        ClaimLifecycleStage.VERIFIED.value,
        ClaimLifecycleStage.SUBSTANTIATED.value,
        ClaimLifecycleStage.WITHDRAWN.value,
    ],
    ClaimLifecycleStage.VERIFIED.value: [
        ClaimLifecycleStage.PUBLISHED.value,
        ClaimLifecycleStage.WITHDRAWN.value,
    ],
    ClaimLifecycleStage.PUBLISHED.value: [
        ClaimLifecycleStage.EXPIRED.value,
        ClaimLifecycleStage.WITHDRAWN.value,
    ],
    ClaimLifecycleStage.EXPIRED.value: [],
    ClaimLifecycleStage.WITHDRAWN.value: [],
}

# Remediation priority by article (lower = higher priority).
REMEDIATION_PRIORITY: Dict[str, int] = {
    ArticleReference.ARTICLE_3.value: 1,
    ArticleReference.ARTICLE_7.value: 2,
    ArticleReference.ARTICLE_4.value: 3,
    ArticleReference.ARTICLE_6.value: 4,
    ArticleReference.ARTICLE_5.value: 5,
    ArticleReference.ARTICLE_8.value: 6,
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ObligationItem(BaseModel):
    """A single compliance obligation under the Directive.

    Represents one specific requirement from an article of the
    Green Claims Directive that the trader must fulfil.
    """
    obligation_id: str = Field(
        default_factory=_new_uuid,
        description="Unique obligation identifier",
    )
    article: ArticleReference = Field(
        ...,
        description="Directive article this obligation belongs to",
    )
    paragraph: str = Field(
        default="",
        description="Specific paragraph reference (e.g., '3(1)(a)')",
        max_length=50,
    )
    description: str = Field(
        ...,
        description="Description of the obligation requirement",
        max_length=1000,
    )
    status: ComplianceStatus = Field(
        default=ComplianceStatus.NON_COMPLIANT,
        description="Current compliance status",
    )
    evidence_ref: Optional[str] = Field(
        default=None,
        description="Reference to supporting evidence",
        max_length=500,
    )
    remediation_note: Optional[str] = Field(
        default=None,
        description="Remediation guidance if non-compliant",
        max_length=1000,
    )

    @field_validator("description")
    @classmethod
    def validate_description_not_empty(cls, v: str) -> str:
        """Ensure description is not empty."""
        if not v.strip():
            raise ValueError("Obligation description must not be empty")
        return v

class ClaimLifecycleRecord(BaseModel):
    """Lifecycle tracking record for an environmental claim.

    Maintains the current lifecycle stage and transition history
    for audit trail purposes per Article 8.
    """
    claim_id: str = Field(
        default_factory=_new_uuid,
        description="Unique claim identifier",
    )
    current_stage: ClaimLifecycleStage = Field(
        default=ClaimLifecycleStage.DRAFT,
        description="Current lifecycle stage",
    )
    transitions: List[Dict[str, str]] = Field(
        default_factory=list,
        description="History of stage transitions with timestamps",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="Record creation timestamp (UTC)",
    )
    last_updated: datetime = Field(
        default_factory=utcnow,
        description="Last update timestamp (UTC)",
    )

class TraderObligationResult(BaseModel):
    """Complete trader obligation compliance result.

    Contains the full compliance assessment across all articles
    with per-obligation details, overall score, and remediation
    timeline.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    overall_compliance: ComplianceStatus = Field(
        default=ComplianceStatus.NON_COMPLIANT,
        description="Overall compliance status",
    )
    overall_score: Decimal = Field(
        default=Decimal("0"),
        description="Overall compliance score (0-100)",
    )
    article_scores: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-article compliance scores",
    )
    article_statuses: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-article compliance statuses",
    )
    obligations: List[ObligationItem] = Field(
        default_factory=list,
        description="All assessed obligation items",
    )
    compliant_count: int = Field(
        default=0,
        description="Number of compliant obligations",
    )
    non_compliant_count: int = Field(
        default=0,
        description="Number of non-compliant obligations",
    )
    remediation_items: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Prioritised remediation action items",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Issues identified during assessment",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
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

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TraderObligationEngine:
    """Trader obligation compliance engine per Articles 3-8.

    Assesses trader compliance across all six core articles of the
    Green Claims Directive, tracking claim lifecycle transitions and
    generating prioritised remediation timelines.

    All scoring uses deterministic Decimal arithmetic.
    Every result includes a SHA-256 provenance hash for audit trail.

    Usage::

        engine = TraderObligationEngine()
        assessment_input = {
            "has_scientific_evidence": True,
            "has_lifecycle_assessment": True,
            "has_primary_data": True,
            "claims_not_misleading": True,
            "has_third_party_verification": True,
            "records_maintained": True,
            "record_retention_years": 5,
        }
        result = engine.calculate_overall_compliance(assessment_input)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise TraderObligationEngine."""
        self.engine_id: str = _new_uuid()
        self._lifecycle_records: Dict[str, ClaimLifecycleRecord] = {}
        logger.info(
            "TraderObligationEngine v%s initialised | engine_id=%s",
            self.engine_version,
            self.engine_id,
        )

    # ------------------------------------------------------------------ #
    # Assess Article 3                                                      #
    # ------------------------------------------------------------------ #

    def assess_article3(
        self,
        *,
        has_scientific_evidence: bool = False,
        has_lifecycle_assessment: bool = False,
        has_primary_data: bool = False,
        has_international_standards: bool = False,
        all_impacts_identified: bool = False,
        methodology_specified: bool = False,
        value_chain_considered: bool = False,
        data_up_to_date: bool = False,
    ) -> Dict[str, Any]:
        """Assess compliance with Article 3 (Substantiation).

        Evaluates each requirement of Article 3 and returns a
        detailed compliance assessment.

        Args:
            has_scientific_evidence: Backed by scientific evidence.
            has_lifecycle_assessment: Lifecycle perspective demonstrated.
            has_primary_data: Primary/robust secondary data included.
            has_international_standards: International standards used.
            all_impacts_identified: All significant impacts identified.
            methodology_specified: Recognised methodology used.
            value_chain_considered: Full value chain considered.
            data_up_to_date: Data is accurate and up-to-date.

        Returns:
            Dict with obligations, score, status, and provenance_hash.
        """
        t0 = time.perf_counter()
        logger.info("Assessing Article 3 compliance")

        checks = [
            (has_scientific_evidence, ARTICLE_3_CHECKLIST[0]),
            (data_up_to_date, ARTICLE_3_CHECKLIST[1]),
            (has_international_standards, ARTICLE_3_CHECKLIST[2]),
            (has_lifecycle_assessment, ARTICLE_3_CHECKLIST[3]),
            (all_impacts_identified, ARTICLE_3_CHECKLIST[4]),
            (has_primary_data, ARTICLE_3_CHECKLIST[5]),
            (methodology_specified, ARTICLE_3_CHECKLIST[6]),
            (value_chain_considered, ARTICLE_3_CHECKLIST[7]),
        ]

        obligations: List[ObligationItem] = []
        met_count = 0

        for met, description in checks:
            status = (
                ComplianceStatus.COMPLIANT if met
                else ComplianceStatus.NON_COMPLIANT
            )
            remediation = None if met else (
                f"Action required: {description}"
            )
            obligations.append(ObligationItem(
                article=ArticleReference.ARTICLE_3,
                description=description,
                status=status,
                remediation_note=remediation,
            ))
            if met:
                met_count += 1

        total = len(checks)
        score = _round_val(
            _safe_divide(
                _decimal(met_count) * Decimal("100"),
                _decimal(total),
            ), 2
        )

        article_status = self._determine_article_status(score)
        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = {
            "article": ArticleReference.ARTICLE_3.value,
            "obligations": [o.model_dump(mode="json") for o in obligations],
            "score": str(score),
            "status": article_status,
            "met_count": met_count,
            "total_count": total,
            "processing_time_ms": elapsed_ms,
            "engine_id": self.engine_id,
            "version": self.engine_version,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Article 3 assessed | score=%s status=%s %d/%d met",
            score, article_status, met_count, total,
        )
        return result

    # ------------------------------------------------------------------ #
    # Assess Article 4                                                      #
    # ------------------------------------------------------------------ #

    def assess_article4(
        self,
        *,
        claims_not_misleading: bool = False,
        aggregated_indicators_substantiated: bool = False,
        no_exaggeration: bool = False,
        future_claims_have_targets: bool = False,
    ) -> Dict[str, Any]:
        """Assess compliance with Article 4 (Communication).

        Args:
            claims_not_misleading: Claims are not misleading.
            aggregated_indicators_substantiated: Aggregated indicators
                are based on sufficient evidence.
            no_exaggeration: Claims do not exaggerate benefits.
            future_claims_have_targets: Future claims have time-bound
                commitments.

        Returns:
            Dict with obligations, score, status, and provenance_hash.
        """
        t0 = time.perf_counter()
        logger.info("Assessing Article 4 compliance")

        checks = [
            (claims_not_misleading,
             "Environmental claims shall not be misleading (Art. 4(1))"),
            (aggregated_indicators_substantiated,
             "Aggregated indicators based on sufficient evidence (Art. 4(2))"),
            (no_exaggeration,
             "Claims shall not exaggerate environmental benefit (Art. 4(3))"),
            (future_claims_have_targets,
             "Future claims include time-bound commitments and "
             "measurable targets (Art. 4(4))"),
        ]

        return self._assess_article_generic(
            ArticleReference.ARTICLE_4, checks, t0
        )

    # ------------------------------------------------------------------ #
    # Assess Article 5                                                      #
    # ------------------------------------------------------------------ #

    def assess_article5(
        self,
        *,
        equivalent_data: bool = False,
        equivalent_boundaries: bool = False,
        equivalent_functional_units: bool = False,
        methodology_consistent: bool = False,
    ) -> Dict[str, Any]:
        """Assess compliance with Article 5 (Comparative Claims).

        Args:
            equivalent_data: Equivalent information and data used.
            equivalent_boundaries: Equivalent system boundaries.
            equivalent_functional_units: Equivalent functional units.
            methodology_consistent: Consistent methodology applied.

        Returns:
            Dict with obligations, score, status, and provenance_hash.
        """
        t0 = time.perf_counter()
        logger.info("Assessing Article 5 compliance")

        checks = [
            (equivalent_data,
             "Equivalent information and data for compared items (Art. 5(1))"),
            (equivalent_boundaries,
             "Equivalent system boundaries for comparison (Art. 5(1))"),
            (equivalent_functional_units,
             "Equivalent functional units for comparison (Art. 5(1))"),
            (methodology_consistent,
             "Consistent methodology across compared items (Art. 5(1))"),
        ]

        return self._assess_article_generic(
            ArticleReference.ARTICLE_5, checks, t0
        )

    # ------------------------------------------------------------------ #
    # Assess Article 6                                                      #
    # ------------------------------------------------------------------ #

    def assess_article6(
        self,
        *,
        labels_from_recognized_schemes: bool = False,
        labels_third_party_verified: bool = False,
        no_self_created_labels: bool = False,
        labels_have_transparent_criteria: bool = False,
    ) -> Dict[str, Any]:
        """Assess compliance with Article 6 (Environmental Labelling).

        Args:
            labels_from_recognized_schemes: Labels from recognised
                certification schemes.
            labels_third_party_verified: Labels are third-party verified.
            no_self_created_labels: No self-created labels are used.
            labels_have_transparent_criteria: Label criteria are
                publicly transparent.

        Returns:
            Dict with obligations, score, status, and provenance_hash.
        """
        t0 = time.perf_counter()
        logger.info("Assessing Article 6 compliance")

        checks = [
            (labels_from_recognized_schemes,
             "Labels based on recognised certification schemes or "
             "established by public authorities (Art. 6)"),
            (labels_third_party_verified,
             "Labels verified by independent third party (Art. 6)"),
            (no_self_created_labels,
             "No self-created or proprietary sustainability labels "
             "are displayed (Art. 6)"),
            (labels_have_transparent_criteria,
             "Label criteria are publicly available and transparent "
             "(Art. 6)"),
        ]

        return self._assess_article_generic(
            ArticleReference.ARTICLE_6, checks, t0
        )

    # ------------------------------------------------------------------ #
    # Assess Article 7                                                      #
    # ------------------------------------------------------------------ #

    def assess_article7(
        self,
        *,
        has_independent_verifier: bool = False,
        verifier_accredited: bool = False,
        verification_before_publication: bool = False,
        certificate_of_conformity: bool = False,
    ) -> Dict[str, Any]:
        """Assess compliance with Article 7 (Verification).

        Args:
            has_independent_verifier: Independent verifier is engaged.
            verifier_accredited: Verifier is accredited under
                Regulation (EC) No 765/2008.
            verification_before_publication: Verification completed
                before claim communication.
            certificate_of_conformity: Certificate of conformity
                has been issued.

        Returns:
            Dict with obligations, score, status, and provenance_hash.
        """
        t0 = time.perf_counter()
        logger.info("Assessing Article 7 compliance")

        checks = [
            (has_independent_verifier,
             "Independent verifier engaged for claim verification "
             "(Art. 7(1))"),
            (verifier_accredited,
             "Verifier accredited under Regulation (EC) No 765/2008 "
             "(Art. 7(4))"),
            (verification_before_publication,
             "Verification completed before claim communication to "
             "consumers (Art. 7(1))"),
            (certificate_of_conformity,
             "Certificate of conformity issued confirming Directive "
             "compliance (Art. 7(3))"),
        ]

        return self._assess_article_generic(
            ArticleReference.ARTICLE_7, checks, t0
        )

    # ------------------------------------------------------------------ #
    # Assess Article 8                                                      #
    # ------------------------------------------------------------------ #

    def assess_article8(
        self,
        *,
        records_maintained: bool = False,
        record_retention_years: int = 0,
        substantiation_documented: bool = False,
        verification_records_kept: bool = False,
    ) -> Dict[str, Any]:
        """Assess compliance with Article 8 (Record Keeping).

        Args:
            records_maintained: Substantiation records are maintained.
            record_retention_years: Number of years records are kept.
            substantiation_documented: Substantiation is documented.
            verification_records_kept: Verification records are kept.

        Returns:
            Dict with obligations, score, status, and provenance_hash.
        """
        t0 = time.perf_counter()
        logger.info("Assessing Article 8 compliance")

        retention_met = record_retention_years >= RECORD_RETENTION_YEARS

        checks = [
            (records_maintained,
             "Substantiation and verification records are maintained "
             "(Art. 8)"),
            (retention_met,
             f"Records retained for at least {RECORD_RETENTION_YEARS} "
             f"years (Art. 8)"),
            (substantiation_documented,
             "Substantiation methodology and data are fully documented "
             "(Art. 8)"),
            (verification_records_kept,
             "Verification certificates and correspondence are kept "
             "(Art. 8)"),
        ]

        return self._assess_article_generic(
            ArticleReference.ARTICLE_8, checks, t0
        )

    # ------------------------------------------------------------------ #
    # Calculate Overall Compliance                                          #
    # ------------------------------------------------------------------ #

    def calculate_overall_compliance(
        self, assessment_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate overall compliance across all articles.

        Runs assessments for Articles 3-8 using the provided input
        and produces a weighted overall compliance score.

        Args:
            assessment_input: Dict with boolean flags for each
                compliance criterion across all articles.

        Returns:
            Dict with result (TraderObligationResult),
            provenance_hash (str).
        """
        t0 = time.perf_counter()
        logger.info("Calculating overall trader compliance")

        # Assess each article
        a3 = self.assess_article3(
            has_scientific_evidence=assessment_input.get(
                "has_scientific_evidence", False),
            has_lifecycle_assessment=assessment_input.get(
                "has_lifecycle_assessment", False),
            has_primary_data=assessment_input.get(
                "has_primary_data", False),
            has_international_standards=assessment_input.get(
                "has_international_standards", False),
            all_impacts_identified=assessment_input.get(
                "all_impacts_identified", False),
            methodology_specified=assessment_input.get(
                "methodology_specified", False),
            value_chain_considered=assessment_input.get(
                "value_chain_considered", False),
            data_up_to_date=assessment_input.get(
                "data_up_to_date", False),
        )
        a4 = self.assess_article4(
            claims_not_misleading=assessment_input.get(
                "claims_not_misleading", False),
            aggregated_indicators_substantiated=assessment_input.get(
                "aggregated_indicators_substantiated", False),
            no_exaggeration=assessment_input.get(
                "no_exaggeration", False),
            future_claims_have_targets=assessment_input.get(
                "future_claims_have_targets", False),
        )
        a5 = self.assess_article5(
            equivalent_data=assessment_input.get(
                "equivalent_data", False),
            equivalent_boundaries=assessment_input.get(
                "equivalent_boundaries", False),
            equivalent_functional_units=assessment_input.get(
                "equivalent_functional_units", False),
            methodology_consistent=assessment_input.get(
                "methodology_consistent", False),
        )
        a6 = self.assess_article6(
            labels_from_recognized_schemes=assessment_input.get(
                "labels_from_recognized_schemes", False),
            labels_third_party_verified=assessment_input.get(
                "labels_third_party_verified", False),
            no_self_created_labels=assessment_input.get(
                "no_self_created_labels", False),
            labels_have_transparent_criteria=assessment_input.get(
                "labels_have_transparent_criteria", False),
        )
        a7 = self.assess_article7(
            has_independent_verifier=assessment_input.get(
                "has_independent_verifier", False),
            verifier_accredited=assessment_input.get(
                "verifier_accredited", False),
            verification_before_publication=assessment_input.get(
                "verification_before_publication", False),
            certificate_of_conformity=assessment_input.get(
                "certificate_of_conformity", False),
        )
        a8 = self.assess_article8(
            records_maintained=assessment_input.get(
                "records_maintained", False),
            record_retention_years=assessment_input.get(
                "record_retention_years", 0),
            substantiation_documented=assessment_input.get(
                "substantiation_documented", False),
            verification_records_kept=assessment_input.get(
                "verification_records_kept", False),
        )

        article_results = {
            ArticleReference.ARTICLE_3.value: a3,
            ArticleReference.ARTICLE_4.value: a4,
            ArticleReference.ARTICLE_5.value: a5,
            ArticleReference.ARTICLE_6.value: a6,
            ArticleReference.ARTICLE_7.value: a7,
            ArticleReference.ARTICLE_8.value: a8,
        }

        # Calculate weighted overall score
        overall_score = Decimal("0")
        article_scores: Dict[str, str] = {}
        article_statuses: Dict[str, str] = {}
        all_obligations: List[ObligationItem] = []
        issues: List[str] = []
        remediation_items: List[Dict[str, str]] = []

        compliant_total = 0
        non_compliant_total = 0

        for art_key, art_result in article_results.items():
            art_score = _decimal(art_result["score"])
            weight = ARTICLE_WEIGHTS.get(art_key, Decimal("10"))
            overall_score += art_score * weight / Decimal("100")

            article_scores[art_key] = str(_round_val(art_score, 2))
            article_statuses[art_key] = art_result["status"]

            met = art_result.get("met_count", 0)
            total = art_result.get("total_count", 0)
            compliant_total += met
            non_compliant_total += (total - met)

            if art_result["status"] != ComplianceStatus.COMPLIANT.value:
                issues.append(
                    f"{art_key}: {art_result['status']} "
                    f"({met}/{total} obligations met)"
                )

            # Build remediation items from non-compliant obligations
            for obl in art_result.get("obligations", []):
                if obl.get("status") == ComplianceStatus.NON_COMPLIANT.value:
                    remediation_items.append({
                        "article": art_key,
                        "priority": str(REMEDIATION_PRIORITY.get(
                            art_key, 99)),
                        "description": obl.get("description", ""),
                        "remediation": obl.get(
                            "remediation_note",
                            f"Address: {obl.get('description', '')}",
                        ),
                    })

        overall_score = _round_val(overall_score, 2)
        overall_status = self._determine_article_status(overall_score)

        # Sort remediation items by priority
        remediation_items.sort(key=lambda x: int(x.get("priority", "99")))

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        obligation_result = TraderObligationResult(
            overall_compliance=ComplianceStatus(overall_status),
            overall_score=overall_score,
            article_scores=article_scores,
            article_statuses=article_statuses,
            obligations=all_obligations,
            compliant_count=compliant_total,
            non_compliant_count=non_compliant_total,
            remediation_items=remediation_items,
            issues=issues,
            processing_time_ms=elapsed_ms,
        )
        obligation_result.provenance_hash = _compute_hash(
            obligation_result
        )

        logger.info(
            "Overall compliance calculated | score=%s status=%s "
            "compliant=%d non_compliant=%d in %.3f ms",
            overall_score, overall_status,
            compliant_total, non_compliant_total, elapsed_ms,
        )

        return {
            "result": obligation_result,
            "article_results": article_results,
            "provenance_hash": obligation_result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Generate Remediation Timeline                                         #
    # ------------------------------------------------------------------ #

    def generate_remediation_timeline(
        self, assessment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a prioritised remediation timeline.

        Analyses the overall compliance result and produces a
        time-phased remediation plan with prioritised action items.

        Args:
            assessment_result: Output from calculate_overall_compliance.

        Returns:
            Dict with timeline phases, action items,
            and provenance_hash.
        """
        t0 = time.perf_counter()
        logger.info("Generating remediation timeline")

        obligation_result = assessment_result.get("result")
        if obligation_result is None:
            return {
                "phases": [],
                "total_items": 0,
                "provenance_hash": _compute_hash({"empty": True}),
            }

        remediation_items = (
            obligation_result.remediation_items
            if hasattr(obligation_result, "remediation_items")
            else []
        )

        # Phase 1: Critical (priority 1-2, immediate action)
        phase_1: List[Dict[str, str]] = [
            item for item in remediation_items
            if int(item.get("priority", "99")) <= 2
        ]

        # Phase 2: High (priority 3-4, within 30 days)
        phase_2: List[Dict[str, str]] = [
            item for item in remediation_items
            if 3 <= int(item.get("priority", "99")) <= 4
        ]

        # Phase 3: Standard (priority 5+, within 90 days)
        phase_3: List[Dict[str, str]] = [
            item for item in remediation_items
            if int(item.get("priority", "99")) >= 5
        ]

        phases = [
            {
                "phase": "Phase 1 - Immediate (0-14 days)",
                "priority": "Critical",
                "items": phase_1,
                "item_count": len(phase_1),
            },
            {
                "phase": "Phase 2 - Short-term (15-30 days)",
                "priority": "High",
                "items": phase_2,
                "item_count": len(phase_2),
            },
            {
                "phase": "Phase 3 - Medium-term (31-90 days)",
                "priority": "Standard",
                "items": phase_3,
                "item_count": len(phase_3),
            },
        ]

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = {
            "phases": phases,
            "total_items": len(remediation_items),
            "overall_score": str(obligation_result.overall_score),
            "overall_status": obligation_result.overall_compliance.value,
            "processing_time_ms": elapsed_ms,
            "engine_id": self.engine_id,
            "version": self.engine_version,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Remediation timeline generated | total=%d "
            "phase1=%d phase2=%d phase3=%d in %.3f ms",
            len(remediation_items),
            len(phase_1), len(phase_2), len(phase_3), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Track Claim Lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def track_claim_lifecycle(
        self,
        claim_id: str,
        target_stage: ClaimLifecycleStage,
    ) -> Dict[str, Any]:
        """Transition a claim to a new lifecycle stage.

        Validates the transition against the allowed state machine
        and records the transition with timestamp for audit trail.

        Args:
            claim_id: ID of the claim to transition.
            target_stage: Target lifecycle stage.

        Returns:
            Dict with claim_id, current_stage, transition_valid,
            transitions history, and provenance_hash.
        """
        t0 = time.perf_counter()
        logger.info(
            "Tracking claim lifecycle | claim_id=%s target=%s",
            claim_id, target_stage.value,
        )

        # Get or create lifecycle record
        if claim_id not in self._lifecycle_records:
            self._lifecycle_records[claim_id] = ClaimLifecycleRecord(
                claim_id=claim_id,
            )

        record = self._lifecycle_records[claim_id]
        current = record.current_stage.value
        target = target_stage.value

        # Validate transition
        allowed = VALID_TRANSITIONS.get(current, [])
        transition_valid = target in allowed

        if transition_valid:
            timestamp = utcnow()
            record.transitions.append({
                "from": current,
                "to": target,
                "timestamp": str(timestamp),
            })
            record.current_stage = target_stage
            record.last_updated = timestamp
            logger.info(
                "Claim lifecycle transitioned | claim_id=%s %s -> %s",
                claim_id, current, target,
            )
        else:
            logger.warning(
                "Invalid lifecycle transition | claim_id=%s "
                "%s -> %s (allowed: %s)",
                claim_id, current, target, allowed,
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = {
            "claim_id": claim_id,
            "previous_stage": current,
            "current_stage": record.current_stage.value,
            "target_stage": target,
            "transition_valid": transition_valid,
            "allowed_transitions": allowed,
            "transitions": record.transitions,
            "created_at": str(record.created_at),
            "last_updated": str(record.last_updated),
            "processing_time_ms": elapsed_ms,
            "engine_id": self.engine_id,
            "version": self.engine_version,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Private Methods                                                       #
    # ------------------------------------------------------------------ #

    def _assess_article_generic(
        self,
        article: ArticleReference,
        checks: List[tuple],
        t0: float,
    ) -> Dict[str, Any]:
        """Generic article assessment from a list of (bool, desc) checks.

        Args:
            article: Article reference enum.
            checks: List of (met: bool, description: str) tuples.
            t0: Start time from time.perf_counter().

        Returns:
            Dict with obligations, score, status, and provenance_hash.
        """
        obligations: List[Dict[str, Any]] = []
        met_count = 0

        for met, description in checks:
            status = (
                ComplianceStatus.COMPLIANT.value if met
                else ComplianceStatus.NON_COMPLIANT.value
            )
            remediation = None if met else f"Action required: {description}"
            obligations.append({
                "obligation_id": _new_uuid(),
                "article": article.value,
                "description": description,
                "status": status,
                "remediation_note": remediation,
            })
            if met:
                met_count += 1

        total = len(checks)
        score = _round_val(
            _safe_divide(
                _decimal(met_count) * Decimal("100"),
                _decimal(total),
            ), 2
        )

        article_status = self._determine_article_status(score)
        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = {
            "article": article.value,
            "obligations": obligations,
            "score": str(score),
            "status": article_status,
            "met_count": met_count,
            "total_count": total,
            "processing_time_ms": elapsed_ms,
            "engine_id": self.engine_id,
            "version": self.engine_version,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "%s assessed | score=%s status=%s %d/%d met",
            article.value, score, article_status, met_count, total,
        )
        return result

    def _determine_article_status(self, score: Decimal) -> str:
        """Determine compliance status from a score.

        Args:
            score: Compliance score (0-100).

        Returns:
            ComplianceStatus value string.
        """
        if score >= Decimal("100"):
            return ComplianceStatus.COMPLIANT.value
        if score >= Decimal("50"):
            return ComplianceStatus.PARTIALLY_COMPLIANT.value
        return ComplianceStatus.NON_COMPLIANT.value
