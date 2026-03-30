# -*- coding: utf-8 -*-
"""
LabelComplianceEngine - PACK-018 EU Green Claims Prep Engine 4
================================================================

Assesses environmental label and scheme compliance per EU Green Claims
Directive Articles 6-9, including label recognition, governance
evaluation, and claims validation.

The EU Green Claims Directive (Proposal COM/2023/166) establishes
requirements for environmental labelling schemes used to communicate
environmental performance to consumers.  Articles 6-9 set rules for
both existing and new labelling schemes.

Article 6 Requirements (Environmental Labelling Schemes):
    - Para 1: Environmental labelling schemes shall provide reliable,
      transparent, and comparable information about the environmental
      performance of products.
    - Para 2: Environmental labels shall be based on a certification
      scheme that includes objective and science-based requirements.
    - Para 3: The governance of the scheme shall ensure independent
      and impartial oversight.

Article 7 Requirements (Governance):
    - Para 1: Labelling schemes shall have a transparent governance
      structure with clear decision-making procedures.
    - Para 2: Stakeholder consultation shall be part of the scheme
      development and revision process.
    - Para 3: Schemes shall have a complaint and dispute resolution
      mechanism accessible to all parties.

Article 8 Requirements (Verification):
    - Para 1: Labels shall be subject to third-party verification
      by accredited conformity assessment bodies.
    - Para 2: Verification shall cover the full scope of the label's
      claimed environmental attributes.
    - Para 3: Regular monitoring and periodic review shall be in place.

Article 9 Requirements (New Schemes):
    - Para 1: New environmental labelling schemes shall be approved
      by Member States before being introduced.
    - Para 2: Approval requires demonstration of added value compared
      to existing EU or Member State schemes.

Regulatory References:
    - EU Green Claims Directive Proposal COM/2023/166, Articles 6-9
    - Regulation (EC) No 66/2010 (EU Ecolabel)
    - ISO 14024 (Type I Environmental Labelling)
    - ISO 14021 (Self-declared Environmental Claims - Type II)
    - ISO 14025 (Environmental Product Declarations - Type III)
    - Regulation (EC) No 765/2008 (Accreditation)

Zero-Hallucination:
    - Label recognition uses deterministic lookup against known list
    - Governance scoring uses weighted criteria with Decimal arithmetic
    - Compliance checks use rule-based threshold evaluation
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
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

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

class LabelType(str, Enum):
    """Classification of environmental label types per ISO 14020 series.

    Environmental labels are classified by the ISO 14020 family of
    standards into three types, plus additional scheme categories
    relevant to the Green Claims Directive.
    """
    TYPE_I_ECOLABEL = "type_i_ecolabel"
    TYPE_II_SELF_DECLARED = "type_ii_self_declared"
    TYPE_III_EPD = "type_iii_epd"
    PRIVATE_SCHEME = "private_scheme"
    PUBLIC_SCHEME = "public_scheme"
    COMPANY_OWN = "company_own"

class LabelGovernanceLevel(str, Enum):
    """Governance quality level for an environmental labelling scheme.

    Per Articles 6-7, labelling schemes must meet governance standards
    including transparency, independence, and accountability.
    """
    EXCELLENT = "excellent"
    ADEQUATE = "adequate"
    INSUFFICIENT = "insufficient"
    NON_COMPLIANT = "non_compliant"

class LabelComplianceStatus(str, Enum):
    """Overall compliance status for a label assessment.

    Indicates whether the label meets all requirements of Articles 6-9.
    """
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recognised environmental labels in the EU market.
# These are well-established, third-party verified labelling schemes
# with transparent governance structures.
RECOGNIZED_LABELS: Dict[str, Dict[str, Any]] = {
    "EU_ECOLABEL": {
        "full_name": "EU Ecolabel",
        "label_type": LabelType.TYPE_I_ECOLABEL.value,
        "governing_body": "European Commission",
        "iso_standard": "ISO 14024",
        "third_party_verified": True,
        "eu_official": True,
        "scope": "Multiple product categories",
        "regulation": "Regulation (EC) No 66/2010",
    },
    "BLUE_ANGEL": {
        "full_name": "Blue Angel (Blauer Engel)",
        "label_type": LabelType.TYPE_I_ECOLABEL.value,
        "governing_body": "German Federal Ministry for the Environment",
        "iso_standard": "ISO 14024",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Multiple product categories",
        "regulation": "German Blue Angel scheme",
    },
    "NORDIC_SWAN": {
        "full_name": "Nordic Swan Ecolabel",
        "label_type": LabelType.TYPE_I_ECOLABEL.value,
        "governing_body": "Nordic Council of Ministers",
        "iso_standard": "ISO 14024",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Multiple product categories (Nordic countries)",
        "regulation": "Nordic Swan regulations",
    },
    "CRADLE_TO_CRADLE": {
        "full_name": "Cradle to Cradle Certified",
        "label_type": LabelType.PRIVATE_SCHEME.value,
        "governing_body": "Cradle to Cradle Products Innovation Institute",
        "iso_standard": "Proprietary standard",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Products, materials, circular economy",
        "regulation": "C2C Certified Standard",
    },
    "FSC": {
        "full_name": "Forest Stewardship Council",
        "label_type": LabelType.PRIVATE_SCHEME.value,
        "governing_body": "Forest Stewardship Council International",
        "iso_standard": "ISO 14024 aligned",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Forest products, timber, paper",
        "regulation": "FSC Standards",
    },
    "PEFC": {
        "full_name": "Programme for the Endorsement of Forest Certification",
        "label_type": LabelType.PRIVATE_SCHEME.value,
        "governing_body": "PEFC International",
        "iso_standard": "ISO 14024 aligned",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Forest products, timber, paper",
        "regulation": "PEFC Standards",
    },
    "GOTS": {
        "full_name": "Global Organic Textile Standard",
        "label_type": LabelType.PRIVATE_SCHEME.value,
        "governing_body": "GOTS International Working Group",
        "iso_standard": "Proprietary standard",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Organic textiles",
        "regulation": "GOTS Standard",
    },
    "OEKO_TEX": {
        "full_name": "OEKO-TEX Standard 100",
        "label_type": LabelType.PRIVATE_SCHEME.value,
        "governing_body": "International OEKO-TEX Association",
        "iso_standard": "Proprietary standard",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Textiles, harmful substances testing",
        "regulation": "OEKO-TEX Standards",
    },
    "ENERGY_STAR": {
        "full_name": "ENERGY STAR",
        "label_type": LabelType.PUBLIC_SCHEME.value,
        "governing_body": "U.S. EPA (recognised in EU)",
        "iso_standard": "Product-specific protocols",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Energy-efficient electronics and appliances",
        "regulation": "EPA ENERGY STAR program",
    },
    "EPEAT": {
        "full_name": "Electronic Product Environmental Assessment Tool",
        "label_type": LabelType.PRIVATE_SCHEME.value,
        "governing_body": "Global Electronics Council",
        "iso_standard": "IEEE 1680 series",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Electronic products",
        "regulation": "IEEE 1680 Standards",
    },
    "TCO_CERTIFIED": {
        "full_name": "TCO Certified",
        "label_type": LabelType.PRIVATE_SCHEME.value,
        "governing_body": "TCO Development",
        "iso_standard": "Proprietary standard",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "IT products, displays",
        "regulation": "TCO Certified criteria",
    },
    "EU_ORGANIC": {
        "full_name": "EU Organic Label",
        "label_type": LabelType.PUBLIC_SCHEME.value,
        "governing_body": "European Commission",
        "iso_standard": "Not applicable (regulatory)",
        "third_party_verified": True,
        "eu_official": True,
        "scope": "Organic agricultural products and food",
        "regulation": "Regulation (EU) 2018/848",
    },
    "FAIRTRADE": {
        "full_name": "Fairtrade International",
        "label_type": LabelType.PRIVATE_SCHEME.value,
        "governing_body": "Fairtrade International",
        "iso_standard": "ISEAL Alliance member",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Fair trade agricultural products",
        "regulation": "Fairtrade Standards",
    },
    "RAINFOREST_ALLIANCE": {
        "full_name": "Rainforest Alliance Certified",
        "label_type": LabelType.PRIVATE_SCHEME.value,
        "governing_body": "Rainforest Alliance",
        "iso_standard": "ISEAL Alliance member",
        "third_party_verified": True,
        "eu_official": False,
        "scope": "Agriculture, forestry, tourism",
        "regulation": "Rainforest Alliance Sustainable Agriculture Standard",
    },
}

# Governance assessment criteria and their weights.
# These criteria are derived from Articles 6-7 requirements.
GOVERNANCE_CRITERIA: Dict[str, Dict[str, Any]] = {
    "scientific_basis": {
        "description": "Label criteria are based on scientific evidence and "
                       "recognised environmental assessment methodologies",
        "weight": Decimal("20"),
        "article_reference": "Article 6(2)",
    },
    "transparency": {
        "description": "Scheme governance, criteria, and decision-making "
                       "processes are publicly accessible and transparent",
        "weight": Decimal("20"),
        "article_reference": "Article 7(1)",
    },
    "third_party_verification": {
        "description": "Label criteria are verified by independent, accredited "
                       "third-party conformity assessment bodies",
        "weight": Decimal("20"),
        "article_reference": "Article 8(1)",
    },
    "complaint_mechanism": {
        "description": "Scheme has an accessible complaint and dispute "
                       "resolution mechanism for all parties",
        "weight": Decimal("15"),
        "article_reference": "Article 7(3)",
    },
    "periodic_review": {
        "description": "Label criteria and governance are subject to "
                       "periodic review and update based on new evidence",
        "weight": Decimal("15"),
        "article_reference": "Article 8(3)",
    },
    "accreditation": {
        "description": "Scheme owner or verifiers are accredited per "
                       "Regulation (EC) No 765/2008",
        "weight": Decimal("10"),
        "article_reference": "Article 10(4)",
    },
}

# Governance level thresholds (overall score 0-100).
GOVERNANCE_THRESHOLDS: Dict[str, Decimal] = {
    "excellent": Decimal("85"),
    "adequate": Decimal("60"),
    "insufficient": Decimal("35"),
    # Below 35 is "non_compliant"
}

# Compliance threshold for label assessment.
LABEL_COMPLIANCE_THRESHOLD: Decimal = Decimal("60")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class LabelData(BaseModel):
    """Input data for an environmental label assessment.

    Represents the information about an environmental labelling scheme
    that needs to be assessed for compliance with Articles 6-9.
    """
    label_id: str = Field(
        default_factory=_new_uuid,
        description="Unique label identifier",
    )
    label_name: str = Field(
        ...,
        description="Name of the environmental label or scheme",
        max_length=500,
    )
    label_type: LabelType = Field(
        ...,
        description="Type of environmental label (ISO classification)",
    )
    scheme_owner: str = Field(
        default="",
        description="Organisation owning or governing the scheme",
        max_length=500,
    )
    accredited: bool = Field(
        default=False,
        description="Whether the scheme/verifier is accredited per Reg 765/2008",
    )
    scientific_basis: bool = Field(
        default=False,
        description="Whether the label criteria are based on scientific evidence",
    )
    transparency: bool = Field(
        default=False,
        description="Whether governance and criteria are publicly transparent",
    )
    third_party_verification: bool = Field(
        default=False,
        description="Whether the label is verified by independent third parties",
    )
    complaint_mechanism: bool = Field(
        default=False,
        description="Whether there is a complaint/dispute resolution mechanism",
    )
    periodic_review: bool = Field(
        default=False,
        description="Whether criteria undergo periodic review and update",
    )
    scope_description: str = Field(
        default="",
        description="Description of the label's scope and product categories",
        max_length=2000,
    )
    issuing_body: str = Field(
        default="",
        description="Body that issues the label certifications",
        max_length=500,
    )
    iso_standard_alignment: str = Field(
        default="",
        description="ISO standard the label aligns with (e.g., ISO 14024)",
        max_length=200,
    )
    active_since: Optional[str] = Field(
        default=None,
        description="Date the scheme has been active (ISO 8601 date string)",
        max_length=10,
    )
    last_review_date: Optional[str] = Field(
        default=None,
        description="Date of the last periodic review (ISO 8601 date string)",
        max_length=10,
    )

    @field_validator("label_name")
    @classmethod
    def validate_label_name_not_empty(cls, v: str) -> str:
        """Ensure label name is not empty."""
        if not v.strip():
            raise ValueError("Label name must not be empty")
        return v

class LabelAssessment(BaseModel):
    """Result of an environmental label compliance assessment.

    Contains the governance score, compliance status, and identified
    issues for a single environmental labelling scheme.
    """
    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique assessment identifier",
    )
    label_id: str = Field(
        ...,
        description="ID of the assessed label",
    )
    label_name: str = Field(
        default="",
        description="Name of the assessed label",
    )
    label_type: str = Field(
        default="",
        description="Type of the assessed label",
    )
    scheme_owner: str = Field(
        default="",
        description="Scheme owner/governing body",
    )
    accredited: bool = Field(
        default=False,
        description="Whether the scheme is accredited",
    )
    governance_score: Decimal = Field(
        default=Decimal("0.00"),
        description="Overall governance score (0-100)",
    )
    governance_level: str = Field(
        default="",
        description="Governance quality level",
    )
    scientific_basis: bool = Field(
        default=False,
        description="Whether label has scientific basis",
    )
    transparency: bool = Field(
        default=False,
        description="Whether governance is transparent",
    )
    third_party_verification: bool = Field(
        default=False,
        description="Whether third-party verification is in place",
    )
    complaint_mechanism: bool = Field(
        default=False,
        description="Whether a complaint mechanism exists",
    )
    periodic_review: bool = Field(
        default=False,
        description="Whether periodic review is conducted",
    )
    compliant: bool = Field(
        default=False,
        description="Whether the label meets compliance threshold",
    )
    compliance_status: str = Field(
        default=LabelComplianceStatus.NON_COMPLIANT.value,
        description="Overall compliance status",
    )
    is_recognized: bool = Field(
        default=False,
        description="Whether the label is in the recognised labels list",
    )
    criteria_met: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of each governance criterion",
    )
    criteria_scores: Dict[str, str] = Field(
        default_factory=dict,
        description="Score per governance criterion",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Issues found in the assessment",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement",
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

class LabelRecognitionResult(BaseModel):
    """Result of checking whether a label is recognised.

    Provides details about the label's recognition status and
    associated metadata from the recognised labels registry.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    label_name: str = Field(
        default="",
        description="Label name that was checked",
    )
    is_recognized: bool = Field(
        default=False,
        description="Whether the label is in the recognised list",
    )
    matched_key: str = Field(
        default="",
        description="Key in RECOGNIZED_LABELS that matched",
    )
    label_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Details from the recognised labels registry",
    )
    similar_labels: List[str] = Field(
        default_factory=list,
        description="Similar recognised labels (for non-matches)",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Check timestamp (UTC)",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the result",
    )

class LabelPortfolioResult(BaseModel):
    """Result of validating a portfolio of environmental labels.

    Aggregated assessment across multiple labels used by an organisation
    or applied to a product.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    total_labels: int = Field(
        default=0,
        description="Total number of labels assessed",
    )
    compliant_labels: int = Field(
        default=0,
        description="Number of compliant labels",
    )
    non_compliant_labels: int = Field(
        default=0,
        description="Number of non-compliant labels",
    )
    compliance_rate: Decimal = Field(
        default=Decimal("0.00"),
        description="Percentage of compliant labels (0-100)",
    )
    average_governance_score: Decimal = Field(
        default=Decimal("0.00"),
        description="Average governance score across all labels",
    )
    recognized_count: int = Field(
        default=0,
        description="Number of recognised labels",
    )
    labels_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of labels per type",
    )
    labels_by_governance_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of labels per governance level",
    )
    high_priority_issues: List[str] = Field(
        default_factory=list,
        description="High-priority issues from non-compliant labels",
    )
    label_assessments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary of each individual label assessment",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Validation timestamp (UTC)",
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

class LabelComplianceEngine:
    """Environmental label compliance engine per Green Claims Directive Art. 6-9.

    Provides deterministic, zero-hallucination assessment of
    environmental labelling schemes:

    - Assess label governance against Article 6-7 criteria
    - Check label recognition against known EU-market labels
    - Evaluate governance quality (scientific basis, transparency, etc.)
    - Validate portfolios of labels for overall compliance

    All calculations use Decimal arithmetic with ROUND_HALF_UP rounding.
    Every result includes a SHA-256 provenance hash for audit trail.

    Usage::

        engine = LabelComplianceEngine()
        label = LabelData(
            label_name="EU Ecolabel",
            label_type=LabelType.TYPE_I_ECOLABEL,
            scheme_owner="European Commission",
            accredited=True,
            scientific_basis=True,
            transparency=True,
            third_party_verification=True,
            complaint_mechanism=True,
            periodic_review=True,
        )
        result = engine.assess_label(label)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise LabelComplianceEngine."""
        logger.info(
            "LabelComplianceEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Assess Label                                                          #
    # ------------------------------------------------------------------ #

    def assess_label(
        self,
        label_data: LabelData,
    ) -> Dict[str, Any]:
        """Assess an environmental label for compliance with Articles 6-9.

        Evaluates the label against governance criteria, checks
        recognition status, and determines overall compliance.

        Args:
            label_data: Label information to assess.

        Returns:
            Dict with keys: assessment (LabelAssessment),
            provenance_hash (str).
        """
        t0 = time.perf_counter()

        # Check recognition
        recognition = self._check_recognition_internal(
            label_data.label_name
        )

        # Calculate governance score
        governance_result = self._evaluate_governance_internal(label_data)
        governance_score = governance_result["score"]
        governance_level = governance_result["level"]
        criteria_met = governance_result["criteria_met"]
        criteria_scores = governance_result["criteria_scores"]

        # Determine compliance
        compliant = governance_score >= LABEL_COMPLIANCE_THRESHOLD
        compliance_status = self._determine_compliance_status(
            governance_score, label_data
        )

        # Identify issues
        issues = self._identify_label_issues(
            label_data, governance_score, governance_level, criteria_met
        )

        # Generate recommendations
        recommendations = self._generate_label_recommendations(
            label_data, governance_score, criteria_met, issues
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        assessment = LabelAssessment(
            label_id=label_data.label_id,
            label_name=label_data.label_name,
            label_type=label_data.label_type.value,
            scheme_owner=label_data.scheme_owner,
            accredited=label_data.accredited,
            governance_score=_round_val(governance_score, 2),
            governance_level=governance_level,
            scientific_basis=label_data.scientific_basis,
            transparency=label_data.transparency,
            third_party_verification=label_data.third_party_verification,
            complaint_mechanism=label_data.complaint_mechanism,
            periodic_review=label_data.periodic_review,
            compliant=compliant,
            compliance_status=compliance_status,
            is_recognized=recognition["is_recognized"],
            criteria_met=criteria_met,
            criteria_scores={
                k: str(_round_val(v, 2)) for k, v in criteria_scores.items()
            },
            issues=issues,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        assessment.provenance_hash = _compute_hash(assessment)

        logger.info(
            "Assessed label '%s' (type=%s): governance=%s (%s), "
            "compliant=%s, recognised=%s in %.3f ms",
            label_data.label_name,
            label_data.label_type.value,
            assessment.governance_score,
            governance_level,
            compliant,
            recognition["is_recognized"],
            elapsed_ms,
        )

        return {
            "assessment": assessment,
            "provenance_hash": assessment.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Check Label Recognition                                               #
    # ------------------------------------------------------------------ #

    def check_label_recognition(
        self,
        label_name: str,
    ) -> Dict[str, Any]:
        """Check whether a label is recognised in the EU market.

        Searches the RECOGNIZED_LABELS registry for an exact or
        partial match of the label name.

        Args:
            label_name: Name of the label to check.

        Returns:
            Dict with keys: result (LabelRecognitionResult),
            provenance_hash (str).
        """
        t0 = time.perf_counter()

        internal = self._check_recognition_internal(label_name)

        # Find similar labels if not recognised
        similar: List[str] = []
        if not internal["is_recognized"]:
            similar = self._find_similar_labels(label_name)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = LabelRecognitionResult(
            label_name=label_name,
            is_recognized=internal["is_recognized"],
            matched_key=internal.get("matched_key", ""),
            label_details=internal.get("details", {}),
            similar_labels=similar,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Checked recognition for '%s': recognised=%s, matched='%s' "
            "in %.3f ms",
            label_name,
            internal["is_recognized"],
            internal.get("matched_key", ""),
            elapsed_ms,
        )

        return {
            "result": result,
            "provenance_hash": result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Evaluate Governance                                                   #
    # ------------------------------------------------------------------ #

    def evaluate_governance(
        self,
        label_data: LabelData,
    ) -> Dict[str, Any]:
        """Evaluate the governance quality of a labelling scheme.

        Assesses the label against each governance criterion defined
        in GOVERNANCE_CRITERIA and produces a weighted overall score.

        Args:
            label_data: Label information to evaluate.

        Returns:
            Dict with keys: score (str), level (str), criteria_met (dict),
            criteria_scores (dict), provenance_hash (str).
        """
        t0 = time.perf_counter()

        internal = self._evaluate_governance_internal(label_data)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = {
            "label_name": label_data.label_name,
            "score": str(_round_val(internal["score"], 2)),
            "level": internal["level"],
            "criteria_met": internal["criteria_met"],
            "criteria_scores": {
                k: str(_round_val(v, 2))
                for k, v in internal["criteria_scores"].items()
            },
            "criteria_details": {
                name: {
                    "description": info["description"],
                    "weight": str(info["weight"]),
                    "article_reference": info["article_reference"],
                    "met": internal["criteria_met"].get(name, False),
                    "score": str(_round_val(
                        internal["criteria_scores"].get(name, Decimal("0")),
                        2,
                    )),
                }
                for name, info in GOVERNANCE_CRITERIA.items()
            },
            "processing_time_ms": elapsed_ms,
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Evaluated governance for '%s': score=%s, level=%s in %.3f ms",
            label_data.label_name,
            result["score"],
            internal["level"],
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Validate Label Claims                                                 #
    # ------------------------------------------------------------------ #

    def validate_label_claims(
        self,
        labels_list: List[LabelData],
    ) -> Dict[str, Any]:
        """Validate a portfolio of environmental labels.

        Assesses each label individually and produces an aggregated
        report with compliance rates, governance distributions, and
        high-priority issues.

        Args:
            labels_list: List of labels to validate.

        Returns:
            Dict with keys: result (LabelPortfolioResult),
            provenance_hash (str).
        """
        t0 = time.perf_counter()

        assessments_summary: List[Dict[str, Any]] = []
        scores: List[Decimal] = []
        compliant_count = 0
        non_compliant_count = 0
        recognized_count = 0
        by_type: Dict[str, int] = {}
        by_level: Dict[str, int] = {}
        high_priority_issues: List[str] = []

        for label in labels_list:
            assess_result = self.assess_label(label)
            assessment: LabelAssessment = assess_result["assessment"]

            assessments_summary.append({
                "label_id": label.label_id,
                "label_name": label.label_name,
                "label_type": label.label_type.value,
                "governance_score": str(assessment.governance_score),
                "governance_level": assessment.governance_level,
                "compliant": assessment.compliant,
                "is_recognized": assessment.is_recognized,
                "issues_count": len(assessment.issues),
            })

            scores.append(assessment.governance_score)

            if assessment.compliant:
                compliant_count += 1
            else:
                non_compliant_count += 1

            if assessment.is_recognized:
                recognized_count += 1

            # By type
            lt = label.label_type.value
            by_type[lt] = by_type.get(lt, 0) + 1

            # By governance level
            gl = assessment.governance_level
            by_level[gl] = by_level.get(gl, 0) + 1

            # Collect high-priority issues from non-compliant labels
            if not assessment.compliant:
                for issue in assessment.issues:
                    high_priority_issues.append(
                        f"[{label.label_name}] {issue}"
                    )

        total = len(labels_list)
        avg_score = Decimal("0.00")
        compliance_rate = Decimal("0.00")
        if total > 0:
            avg_score = _round_val(
                sum(scores) / _decimal(total), 2
            )
            compliance_rate = _round_val(
                _decimal(compliant_count) / _decimal(total) * Decimal("100"),
                2,
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        portfolio_result = LabelPortfolioResult(
            total_labels=total,
            compliant_labels=compliant_count,
            non_compliant_labels=non_compliant_count,
            compliance_rate=compliance_rate,
            average_governance_score=avg_score,
            recognized_count=recognized_count,
            labels_by_type=by_type,
            labels_by_governance_level=by_level,
            high_priority_issues=high_priority_issues,
            label_assessments=assessments_summary,
            processing_time_ms=elapsed_ms,
        )
        portfolio_result.provenance_hash = _compute_hash(portfolio_result)

        logger.info(
            "Validated %d labels: %d compliant, %d non-compliant, "
            "%d recognised, avg governance=%s, rate=%s%% in %.3f ms",
            total,
            compliant_count,
            non_compliant_count,
            recognized_count,
            avg_score,
            compliance_rate,
            elapsed_ms,
        )

        return {
            "result": portfolio_result,
            "provenance_hash": portfolio_result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # Private Methods                                                       #
    # ------------------------------------------------------------------ #

    def _check_recognition_internal(
        self, label_name: str
    ) -> Dict[str, Any]:
        """Internal recognition check against RECOGNIZED_LABELS.

        Args:
            label_name: Label name to check.

        Returns:
            Dict with is_recognized (bool), matched_key (str),
            details (dict).
        """
        name_lower = label_name.lower().strip()

        for key, info in RECOGNIZED_LABELS.items():
            # Check exact key match
            if key.lower() == name_lower:
                return {
                    "is_recognized": True,
                    "matched_key": key,
                    "details": info,
                }
            # Check full name match
            if info["full_name"].lower() == name_lower:
                return {
                    "is_recognized": True,
                    "matched_key": key,
                    "details": info,
                }
            # Check partial match (label name contained in full name
            # or key)
            key_normalized = key.lower().replace("_", " ")
            full_normalized = info["full_name"].lower()
            if (
                name_lower in key_normalized
                or name_lower in full_normalized
                or key_normalized in name_lower
                or full_normalized in name_lower
            ):
                return {
                    "is_recognized": True,
                    "matched_key": key,
                    "details": info,
                }

        return {
            "is_recognized": False,
            "matched_key": "",
            "details": {},
        }

    def _find_similar_labels(self, label_name: str) -> List[str]:
        """Find similar labels from the recognised list.

        Uses simple word overlap to suggest potentially matching
        labels when an exact match is not found.

        Args:
            label_name: Label name to find similarities for.

        Returns:
            List of similar recognised label names (up to 3).
        """
        name_words = set(label_name.lower().split())
        scored: List[tuple] = []

        for key, info in RECOGNIZED_LABELS.items():
            full_words = set(info["full_name"].lower().split())
            key_words = set(key.lower().replace("_", " ").split())
            all_words = full_words | key_words

            overlap = len(name_words & all_words)
            if overlap > 0:
                scored.append((info["full_name"], overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored[:3]]

    def _evaluate_governance_internal(
        self, label_data: LabelData
    ) -> Dict[str, Any]:
        """Internal governance evaluation logic.

        Args:
            label_data: Label information.

        Returns:
            Dict with score (Decimal), level (str), criteria_met (dict),
            criteria_scores (dict).
        """
        criteria_met: Dict[str, bool] = {
            "scientific_basis": label_data.scientific_basis,
            "transparency": label_data.transparency,
            "third_party_verification": label_data.third_party_verification,
            "complaint_mechanism": label_data.complaint_mechanism,
            "periodic_review": label_data.periodic_review,
            "accreditation": label_data.accredited,
        }

        criteria_scores: Dict[str, Decimal] = {}
        total_score = Decimal("0")

        for criterion, info in GOVERNANCE_CRITERIA.items():
            weight = info["weight"]
            met = criteria_met.get(criterion, False)

            if met:
                # Full score for met criteria
                score = weight
            else:
                score = Decimal("0")

            criteria_scores[criterion] = score
            total_score += score

        # Determine governance level
        if total_score >= GOVERNANCE_THRESHOLDS["excellent"]:
            level = LabelGovernanceLevel.EXCELLENT.value
        elif total_score >= GOVERNANCE_THRESHOLDS["adequate"]:
            level = LabelGovernanceLevel.ADEQUATE.value
        elif total_score >= GOVERNANCE_THRESHOLDS["insufficient"]:
            level = LabelGovernanceLevel.INSUFFICIENT.value
        else:
            level = LabelGovernanceLevel.NON_COMPLIANT.value

        return {
            "score": total_score,
            "level": level,
            "criteria_met": criteria_met,
            "criteria_scores": criteria_scores,
        }

    def _determine_compliance_status(
        self,
        governance_score: Decimal,
        label_data: LabelData,
    ) -> str:
        """Determine the overall compliance status of a label.

        Args:
            governance_score: Calculated governance score.
            label_data: Label information.

        Returns:
            Compliance status string.
        """
        if governance_score >= LABEL_COMPLIANCE_THRESHOLD:
            # Check if all critical criteria are met
            critical_criteria = [
                label_data.scientific_basis,
                label_data.third_party_verification,
            ]
            if all(critical_criteria):
                return LabelComplianceStatus.COMPLIANT.value
            return LabelComplianceStatus.PARTIALLY_COMPLIANT.value

        if governance_score >= Decimal("35"):
            return LabelComplianceStatus.PARTIALLY_COMPLIANT.value

        return LabelComplianceStatus.NON_COMPLIANT.value

    def _identify_label_issues(
        self,
        label_data: LabelData,
        governance_score: Decimal,
        governance_level: str,
        criteria_met: Dict[str, bool],
    ) -> List[str]:
        """Identify issues with a label's compliance.

        Args:
            label_data: Label information.
            governance_score: Calculated governance score.
            governance_level: Determined governance level.
            criteria_met: Status of each criterion.

        Returns:
            List of issue descriptions.
        """
        issues: List[str] = []

        if not criteria_met.get("scientific_basis", False):
            issues.append(
                "Label criteria lack a scientific basis per Article 6(2); "
                "criteria must be based on recognised scientific evidence"
            )

        if not criteria_met.get("transparency", False):
            issues.append(
                "Scheme governance is not sufficiently transparent per "
                "Article 7(1); decision-making processes must be public"
            )

        if not criteria_met.get("third_party_verification", False):
            issues.append(
                "Label lacks third-party verification per Article 8(1); "
                "must be verified by accredited conformity assessment bodies"
            )

        if not criteria_met.get("complaint_mechanism", False):
            issues.append(
                "No complaint or dispute resolution mechanism per "
                "Article 7(3); must be accessible to all parties"
            )

        if not criteria_met.get("periodic_review", False):
            issues.append(
                "Label criteria are not subject to periodic review per "
                "Article 8(3); criteria must be regularly updated"
            )

        if not criteria_met.get("accreditation", False):
            issues.append(
                "Scheme or verifiers not accredited per Regulation (EC) "
                "No 765/2008 as required by Article 10(4)"
            )

        # Type-specific issues
        if label_data.label_type == LabelType.COMPANY_OWN:
            issues.append(
                "Company-own labels face heightened scrutiny under "
                "Article 9; must demonstrate added value over existing "
                "EU schemes"
            )

        if label_data.label_type == LabelType.TYPE_II_SELF_DECLARED:
            if not criteria_met.get("third_party_verification", False):
                issues.append(
                    "Self-declared (Type II) claims without third-party "
                    "verification are at high risk of non-compliance"
                )

        return issues

    def _generate_label_recommendations(
        self,
        label_data: LabelData,
        governance_score: Decimal,
        criteria_met: Dict[str, bool],
        issues: List[str],
    ) -> List[str]:
        """Generate recommendations for improving label compliance.

        Args:
            label_data: Label information.
            governance_score: Calculated governance score.
            criteria_met: Status of each criterion.
            issues: Previously identified issues.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if not criteria_met.get("scientific_basis", False):
            recommendations.append(
                "Establish label criteria based on widely recognised "
                "scientific evidence and environmental assessment "
                "methodologies (e.g., PEF, ISO 14040)"
            )

        if not criteria_met.get("transparency", False):
            recommendations.append(
                "Publish governance structure, criteria development "
                "process, and decision-making procedures publicly"
            )

        if not criteria_met.get("third_party_verification", False):
            recommendations.append(
                "Engage accredited conformity assessment bodies (per "
                "Regulation (EC) No 765/2008) for independent "
                "verification of label criteria"
            )

        if not criteria_met.get("complaint_mechanism", False):
            recommendations.append(
                "Implement an accessible complaint and dispute "
                "resolution mechanism for label applicants, users, "
                "and the public"
            )

        if not criteria_met.get("periodic_review", False):
            recommendations.append(
                "Establish a periodic review cycle (recommended: every "
                "3-5 years) for updating label criteria based on new "
                "scientific evidence and stakeholder feedback"
            )

        if not criteria_met.get("accreditation", False):
            recommendations.append(
                "Obtain accreditation for verification bodies per "
                "Regulation (EC) No 765/2008, or transition to "
                "accredited verifiers"
            )

        if (
            label_data.label_type == LabelType.COMPANY_OWN
            and governance_score < Decimal("85")
        ):
            recommendations.append(
                "Consider transitioning from a company-own label to a "
                "recognised third-party scheme (e.g., EU Ecolabel) to "
                "strengthen credibility and compliance"
            )

        if not recommendations:
            recommendations.append(
                "Label meets all governance criteria; maintain current "
                "standards and continue periodic review cycle"
            )

        return recommendations
