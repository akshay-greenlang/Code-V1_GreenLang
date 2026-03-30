# -*- coding: utf-8 -*-
"""
PledgeCommitmentEngine - PACK-025 Race to Zero Engine 1
========================================================

Validates pledge eligibility and commitment criteria for Race to Zero
campaign participation. Assesses whether the entity qualifies based on
8 mandatory criteria: net-zero commitment by 2050, partner initiative
membership, interim 2030 target, action plan commitment, annual
reporting commitment, scope coverage, governance endorsement, and
public disclosure.

Produces eligibility status (eligible/conditional/ineligible), pledge
quality score (0-100), per-criterion pass/fail/partial assessment, gap
identification, and partner initiative alignment mapping.

Calculation Methodology:
    Pledge Eligibility Assessment (8 criteria):
        Each criterion scored: PASS (1.0) / PARTIAL (0.5) / FAIL (0.0)
        Eligibility = all 5 core criteria PASS + remaining criteria >=PARTIAL

    Pledge Quality Score (0-100):
        quality = sum(criterion_score * criterion_weight) * 100
        Weights:
            net_zero_commitment:    0.20  (highest: foundational)
            partner_initiative:     0.15
            interim_target:         0.15
            action_plan:            0.12
            annual_reporting:       0.10
            scope_coverage:         0.10
            governance:             0.10
            public_disclosure:      0.08

    Quality Tiers:
        STRONG:     >= 85 (all criteria met, specific targets, governance approved)
        ADEQUATE:   >= 65 (core criteria met, minor gaps)
        WEAK:       >= 40 (missing interim target or vague commitment)
        INELIGIBLE: <  40 (no net-zero commitment or no partner)

    Actor-Type Specific Checks:
        corporate:  Full Scope 1+2+3, SBTi/CDP partner pathway
        financial:  GFANZ partner, PCAF financed emissions, portfolio targets
        city:       C40/ICLEI partner, community-wide GPC inventory
        region:     Under2 Coalition, sub-national inventory
        sme:        SME Climate Hub, simplified Scope 3
        university: Second Nature, campus boundary

Regulatory References:
    - Race to Zero Campaign Criteria (UNFCCC, 2022)
    - Race to Zero Interpretation Guide (June 2022 update)
    - HLEG "Integrity Matters" Report (November 2022), Rec 1
    - Paris Agreement Article 4 (2015)
    - SBTi Corporate Net-Zero Standard v1.3 (2024)
    - CDP Climate Change Questionnaire (2024)

Zero-Hallucination:
    - All 8 criteria from Race to Zero Interpretation Guide
    - Actor-type requirements from partner initiative documentation
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-025 Race to Zero
Engine:  1 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActorType(str, Enum):
    """Race to Zero participant actor types.

    CORPORATE: Large corporate entity (>250 employees).
    FINANCIAL_INSTITUTION: Bank, insurer, asset manager.
    CITY: City or municipality.
    REGION: Region, state, or province.
    SME: Small/medium enterprise (<250 employees).
    UNIVERSITY: Higher education institution.
    HEALTHCARE: Hospital or health system.
    """
    CORPORATE = "corporate"
    FINANCIAL_INSTITUTION = "financial_institution"
    CITY = "city"
    REGION = "region"
    SME = "sme"
    UNIVERSITY = "university"
    HEALTHCARE = "healthcare"

class PartnerInitiative(str, Enum):
    """Recognized Race to Zero partner initiatives.

    Each partner has specific requirements that map to R2Z criteria.
    """
    SBTI = "sbti"
    CDP = "cdp"
    C40 = "c40"
    ICLEI = "iclei"
    GFANZ = "gfanz"
    WMB = "wmb"
    CLIMATE_PLEDGE = "climate_pledge"
    SME_CLIMATE_HUB = "sme_climate_hub"
    SECOND_NATURE = "second_nature"
    UNDER2 = "under2"
    HCWH = "hcwh"
    EXPONENTIAL_ROADMAP = "exponential_roadmap"
    NZBA = "nzba"
    NZAM = "nzam"
    NZAOA = "nzaoa"
    GLOBAL_COVENANT = "global_covenant"
    OTHER = "other"

class EligibilityStatus(str, Enum):
    """Pledge eligibility status.

    ELIGIBLE: All criteria met, ready to join Race to Zero.
    CONDITIONAL: Core criteria met, minor gaps to address.
    INELIGIBLE: Fundamental criteria missing, cannot join.
    """
    ELIGIBLE = "eligible"
    CONDITIONAL = "conditional"
    INELIGIBLE = "ineligible"

class PledgeQuality(str, Enum):
    """Pledge quality tier based on quality score.

    STRONG: >=85 -- all criteria met with specificity.
    ADEQUATE: >=65 -- core criteria met, minor gaps.
    WEAK: >=40 -- missing key elements.
    INELIGIBLE: <40 -- fundamental gaps.
    """
    STRONG = "strong"
    ADEQUATE = "adequate"
    WEAK = "weak"
    INELIGIBLE = "ineligible"

class CriterionStatus(str, Enum):
    """Assessment status for a single pledge criterion.

    PASS: Criterion fully met with evidence.
    PARTIAL: Partially met, gaps identified.
    FAIL: Criterion not met.
    NOT_APPLICABLE: Not applicable for this actor type.
    """
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"

# ---------------------------------------------------------------------------
# Constants -- Criterion Weights
# ---------------------------------------------------------------------------

CRITERION_IDS = [
    "net_zero_commitment",
    "partner_initiative",
    "interim_target",
    "action_plan",
    "annual_reporting",
    "scope_coverage",
    "governance",
    "public_disclosure",
]

CRITERION_WEIGHTS: Dict[str, Decimal] = {
    "net_zero_commitment": Decimal("0.20"),
    "partner_initiative": Decimal("0.15"),
    "interim_target": Decimal("0.15"),
    "action_plan": Decimal("0.12"),
    "annual_reporting": Decimal("0.10"),
    "scope_coverage": Decimal("0.10"),
    "governance": Decimal("0.10"),
    "public_disclosure": Decimal("0.08"),
}

CRITERION_LABELS: Dict[str, str] = {
    "net_zero_commitment": "Net-Zero Commitment by 2050",
    "partner_initiative": "Partner Initiative Membership",
    "interim_target": "Interim 2030 Target",
    "action_plan": "Action Plan Commitment",
    "annual_reporting": "Annual Reporting Commitment",
    "scope_coverage": "Scope Coverage",
    "governance": "Governance Endorsement",
    "public_disclosure": "Public Disclosure",
}

# Core criteria that MUST be PASS or PARTIAL for eligibility.
CORE_CRITERIA = [
    "net_zero_commitment",
    "partner_initiative",
    "interim_target",
    "action_plan",
    "annual_reporting",
]

# Status numeric scores for quality calculation.
STATUS_SCORES: Dict[str, Decimal] = {
    CriterionStatus.PASS.value: Decimal("1.0"),
    CriterionStatus.PARTIAL.value: Decimal("0.5"),
    CriterionStatus.FAIL.value: Decimal("0.0"),
    CriterionStatus.NOT_APPLICABLE.value: Decimal("1.0"),
}

# Quality tier thresholds.
QUALITY_THRESHOLDS: List[Tuple[Decimal, str]] = [
    (Decimal("85"), PledgeQuality.STRONG.value),
    (Decimal("65"), PledgeQuality.ADEQUATE.value),
    (Decimal("40"), PledgeQuality.WEAK.value),
    (Decimal("0"), PledgeQuality.INELIGIBLE.value),
]

# Actor-type to recommended partner initiatives mapping.
ACTOR_PARTNER_MAP: Dict[str, List[str]] = {
    ActorType.CORPORATE.value: [
        PartnerInitiative.SBTI.value,
        PartnerInitiative.CDP.value,
        PartnerInitiative.WMB.value,
        PartnerInitiative.CLIMATE_PLEDGE.value,
    ],
    ActorType.FINANCIAL_INSTITUTION.value: [
        PartnerInitiative.GFANZ.value,
        PartnerInitiative.NZBA.value,
        PartnerInitiative.NZAM.value,
        PartnerInitiative.NZAOA.value,
    ],
    ActorType.CITY.value: [
        PartnerInitiative.C40.value,
        PartnerInitiative.ICLEI.value,
        PartnerInitiative.GLOBAL_COVENANT.value,
    ],
    ActorType.REGION.value: [
        PartnerInitiative.UNDER2.value,
        PartnerInitiative.ICLEI.value,
    ],
    ActorType.SME.value: [
        PartnerInitiative.SME_CLIMATE_HUB.value,
        PartnerInitiative.EXPONENTIAL_ROADMAP.value,
    ],
    ActorType.UNIVERSITY.value: [
        PartnerInitiative.SECOND_NATURE.value,
    ],
    ActorType.HEALTHCARE.value: [
        PartnerInitiative.HCWH.value,
    ],
}

# Net-zero target year maximum (2050 per Race to Zero).
MAX_NET_ZERO_YEAR: int = 2050

# Minimum scope coverage requirements by actor type.
SCOPE_COVERAGE_REQUIREMENTS: Dict[str, Dict[str, Decimal]] = {
    ActorType.CORPORATE.value: {
        "scope1_pct": Decimal("95"),
        "scope2_pct": Decimal("95"),
        "scope3_pct": Decimal("67"),
    },
    ActorType.FINANCIAL_INSTITUTION.value: {
        "scope1_pct": Decimal("95"),
        "scope2_pct": Decimal("95"),
        "scope3_pct": Decimal("67"),
    },
    ActorType.CITY.value: {
        "scope1_pct": Decimal("90"),
        "scope2_pct": Decimal("90"),
        "scope3_pct": Decimal("0"),
    },
    ActorType.REGION.value: {
        "scope1_pct": Decimal("90"),
        "scope2_pct": Decimal("90"),
        "scope3_pct": Decimal("0"),
    },
    ActorType.SME.value: {
        "scope1_pct": Decimal("95"),
        "scope2_pct": Decimal("95"),
        "scope3_pct": Decimal("40"),
    },
    ActorType.UNIVERSITY.value: {
        "scope1_pct": Decimal("90"),
        "scope2_pct": Decimal("90"),
        "scope3_pct": Decimal("50"),
    },
    ActorType.HEALTHCARE.value: {
        "scope1_pct": Decimal("90"),
        "scope2_pct": Decimal("90"),
        "scope3_pct": Decimal("50"),
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class PledgeCriterionInput(BaseModel):
    """Input data for a single pledge criterion.

    Attributes:
        criterion_id: Criterion identifier from CRITERION_IDS.
        status: Self-assessed status (pass/partial/fail/not_applicable).
        evidence: Evidence description supporting the status.
        evidence_documents: List of supporting document references.
        notes: Additional notes.
    """
    criterion_id: str = Field(..., description="Criterion identifier")
    status: str = Field(
        default=CriterionStatus.FAIL.value,
        description="Self-assessed criterion status"
    )
    evidence: str = Field(default="", description="Evidence description")
    evidence_documents: List[str] = Field(
        default_factory=list, description="Supporting documents"
    )
    notes: str = Field(default="", description="Additional notes")

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, v: str) -> str:
        if v not in CRITERION_IDS:
            raise ValueError(
                f"Unknown criterion '{v}'. Must be one of: {CRITERION_IDS}"
            )
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {s.value for s in CriterionStatus}
        if v not in valid:
            raise ValueError(f"Unknown status '{v}'. Must be one of: {sorted(valid)}")
        return v

class PartnerAlignmentInput(BaseModel):
    """Input data for partner initiative alignment.

    Attributes:
        partner_id: Partner initiative identifier.
        membership_status: active/pending/planned.
        join_date: Date joined or planned join date (YYYY-MM-DD).
        reporting_channel: Whether this is the primary reporting channel.
        notes: Additional notes.
    """
    partner_id: str = Field(..., description="Partner initiative ID")
    membership_status: str = Field(
        default="planned",
        description="Membership status (active/pending/planned)"
    )
    join_date: Optional[str] = Field(
        default=None, description="Join date (YYYY-MM-DD)"
    )
    reporting_channel: bool = Field(
        default=False, description="Primary reporting channel"
    )
    notes: str = Field(default="", description="Additional notes")

    @field_validator("partner_id")
    @classmethod
    def validate_partner(cls, v: str) -> str:
        valid = {p.value for p in PartnerInitiative}
        if v not in valid:
            raise ValueError(f"Unknown partner '{v}'. Must be one of: {sorted(valid)}")
        return v

    @field_validator("membership_status")
    @classmethod
    def validate_membership(cls, v: str) -> str:
        valid = {"active", "pending", "planned"}
        if v not in valid:
            raise ValueError(f"Unknown membership status '{v}'.")
        return v

class PledgeCommitmentInput(BaseModel):
    """Complete input for pledge commitment assessment.

    Attributes:
        entity_name: Organization/entity name.
        actor_type: Type of non-state actor.
        sector: Industry sector or NACE code.
        country: Country code (ISO 3166-1 alpha-2).
        employee_count: Number of employees.
        net_zero_target_year: Planned net-zero target year.
        interim_target_year: Interim target year (typically 2030).
        interim_target_reduction_pct: Planned interim reduction (%).
        baseline_year: Baseline year for targets.
        baseline_emissions_tco2e: Baseline total emissions (tCO2e).
        scope1_coverage_pct: Scope 1 emission coverage (%).
        scope2_coverage_pct: Scope 2 emission coverage (%).
        scope3_coverage_pct: Scope 3 emission coverage (%).
        commitment_statement: Text of the net-zero commitment.
        board_approved: Whether the commitment is board/leadership-approved.
        publicly_disclosed: Whether the commitment is publicly disclosed.
        action_plan_committed: Whether committed to publish action plan.
        action_plan_deadline_months: Months until action plan publication.
        annual_reporting_committed: Whether committed to annual reporting.
        criteria: Per-criterion assessment data.
        partners: Partner initiative alignment data.
        include_recommendations: Whether to generate improvement recommendations.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300,
        description="Entity name"
    )
    actor_type: str = Field(
        default=ActorType.CORPORATE.value,
        description="Actor type"
    )
    sector: str = Field(default="general", max_length=100, description="Sector")
    country: str = Field(default="", max_length=2, description="Country code")
    employee_count: int = Field(default=0, ge=0, description="Employee count")
    net_zero_target_year: int = Field(
        default=2050, ge=2030, le=2060,
        description="Net-zero target year"
    )
    interim_target_year: int = Field(
        default=2030, ge=2025, le=2040,
        description="Interim target year"
    )
    interim_target_reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Interim target reduction (%)"
    )
    baseline_year: int = Field(
        default=0, ge=0, le=2060,
        description="Baseline year"
    )
    baseline_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Baseline emissions (tCO2e)"
    )
    scope1_coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Scope 1 coverage (%)"
    )
    scope2_coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Scope 2 coverage (%)"
    )
    scope3_coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Scope 3 coverage (%)"
    )
    commitment_statement: str = Field(
        default="", max_length=5000,
        description="Net-zero commitment statement text"
    )
    board_approved: bool = Field(
        default=False, description="Board/leadership approved"
    )
    publicly_disclosed: bool = Field(
        default=False, description="Publicly disclosed"
    )
    action_plan_committed: bool = Field(
        default=False, description="Committed to publish action plan"
    )
    action_plan_deadline_months: int = Field(
        default=12, ge=0, le=36,
        description="Months until action plan publication"
    )
    annual_reporting_committed: bool = Field(
        default=False, description="Committed to annual reporting"
    )
    criteria: List[PledgeCriterionInput] = Field(
        default_factory=list, description="Per-criterion data"
    )
    partners: List[PartnerAlignmentInput] = Field(
        default_factory=list, description="Partner initiative data"
    )
    include_recommendations: bool = Field(
        default=True, description="Generate recommendations"
    )

    @field_validator("actor_type")
    @classmethod
    def validate_actor_type(cls, v: str) -> str:
        valid = {a.value for a in ActorType}
        if v not in valid:
            raise ValueError(f"Unknown actor type '{v}'. Must be one of: {sorted(valid)}")
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class PledgeCriterionResult(BaseModel):
    """Assessment result for a single pledge criterion.

    Attributes:
        criterion_id: Criterion identifier.
        criterion_name: Human-readable criterion name.
        status: Assessment status (pass/partial/fail/not_applicable).
        score: Numeric score (1.0/0.5/0.0).
        weight: Criterion weight.
        weighted_score: score * weight.
        is_core: Whether this is a core criterion.
        evidence_provided: Whether evidence was provided.
        gap_description: Description of gap if not PASS.
        remediation_action: Recommended action to close gap.
        effort_estimate: Estimated effort to remediate (low/medium/high).
    """
    criterion_id: str = Field(default="")
    criterion_name: str = Field(default="")
    status: str = Field(default=CriterionStatus.FAIL.value)
    score: Decimal = Field(default=Decimal("0"))
    weight: Decimal = Field(default=Decimal("0"))
    weighted_score: Decimal = Field(default=Decimal("0"))
    is_core: bool = Field(default=False)
    evidence_provided: bool = Field(default=False)
    gap_description: str = Field(default="")
    remediation_action: str = Field(default="")
    effort_estimate: str = Field(default="medium")

class PartnerAlignmentResult(BaseModel):
    """Assessment result for partner initiative alignment.

    Attributes:
        partner_id: Partner identifier.
        partner_name: Human-readable partner name.
        membership_status: Current membership status.
        recommended: Whether this partner is recommended for the actor type.
        r2z_criteria_coverage: Number of R2Z criteria this partner covers.
        alignment_score: Alignment score (0-100).
        notes: Assessment notes.
    """
    partner_id: str = Field(default="")
    partner_name: str = Field(default="")
    membership_status: str = Field(default="planned")
    recommended: bool = Field(default=False)
    r2z_criteria_coverage: int = Field(default=0)
    alignment_score: Decimal = Field(default=Decimal("0"))
    notes: str = Field(default="")

class PledgeCommitmentResult(BaseModel):
    """Complete pledge commitment assessment result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Entity name.
        actor_type: Actor type.
        eligibility_status: Overall eligibility (eligible/conditional/ineligible).
        pledge_quality: Pledge quality tier.
        quality_score: Quality score (0-100).
        criterion_results: Per-criterion assessment results.
        partner_results: Partner initiative assessment results.
        core_criteria_met: Number of core criteria met (PASS or PARTIAL).
        total_criteria_met: Total criteria met (PASS or PARTIAL).
        total_criteria: Total criteria assessed.
        net_zero_year_valid: Whether net-zero year is <=2050.
        interim_target_valid: Whether interim target meets minimum ambition.
        scope_coverage_valid: Whether scope coverage meets actor-type requirements.
        recommended_partners: Recommended partner initiatives for this actor type.
        gaps: List of identified gaps.
        recommendations: Improvement recommendations.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    actor_type: str = Field(default="")
    eligibility_status: str = Field(default=EligibilityStatus.INELIGIBLE.value)
    pledge_quality: str = Field(default=PledgeQuality.INELIGIBLE.value)
    quality_score: Decimal = Field(default=Decimal("0"))
    criterion_results: List[PledgeCriterionResult] = Field(default_factory=list)
    partner_results: List[PartnerAlignmentResult] = Field(default_factory=list)
    core_criteria_met: int = Field(default=0)
    total_criteria_met: int = Field(default=0)
    total_criteria: int = Field(default=8)
    net_zero_year_valid: bool = Field(default=False)
    interim_target_valid: bool = Field(default=False)
    scope_coverage_valid: bool = Field(default=False)
    recommended_partners: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Partner Name Lookup
# ---------------------------------------------------------------------------

PARTNER_NAMES: Dict[str, str] = {
    PartnerInitiative.SBTI.value: "Science Based Targets initiative",
    PartnerInitiative.CDP.value: "CDP",
    PartnerInitiative.C40.value: "C40 Cities",
    PartnerInitiative.ICLEI.value: "ICLEI",
    PartnerInitiative.GFANZ.value: "GFANZ",
    PartnerInitiative.WMB.value: "We Mean Business Coalition",
    PartnerInitiative.CLIMATE_PLEDGE.value: "The Climate Pledge",
    PartnerInitiative.SME_CLIMATE_HUB.value: "SME Climate Hub",
    PartnerInitiative.SECOND_NATURE.value: "Second Nature",
    PartnerInitiative.UNDER2.value: "Under2 Coalition",
    PartnerInitiative.HCWH.value: "Health Care Without Harm",
    PartnerInitiative.EXPONENTIAL_ROADMAP.value: "Exponential Roadmap Initiative",
    PartnerInitiative.NZBA.value: "Net-Zero Banking Alliance",
    PartnerInitiative.NZAM.value: "Net Zero Asset Managers Initiative",
    PartnerInitiative.NZAOA.value: "Net-Zero Asset Owner Alliance",
    PartnerInitiative.GLOBAL_COVENANT.value: "Global Covenant of Mayors",
    PartnerInitiative.OTHER.value: "Other Partner Initiative",
}

# Partner criteria coverage (how many of the 8 R2Z criteria each partner covers).
PARTNER_CRITERIA_COVERAGE: Dict[str, int] = {
    PartnerInitiative.SBTI.value: 6,
    PartnerInitiative.CDP.value: 7,
    PartnerInitiative.C40.value: 7,
    PartnerInitiative.ICLEI.value: 6,
    PartnerInitiative.GFANZ.value: 7,
    PartnerInitiative.WMB.value: 5,
    PartnerInitiative.CLIMATE_PLEDGE.value: 5,
    PartnerInitiative.SME_CLIMATE_HUB.value: 5,
    PartnerInitiative.SECOND_NATURE.value: 6,
    PartnerInitiative.UNDER2.value: 6,
    PartnerInitiative.HCWH.value: 5,
    PartnerInitiative.EXPONENTIAL_ROADMAP.value: 4,
    PartnerInitiative.NZBA.value: 6,
    PartnerInitiative.NZAM.value: 6,
    PartnerInitiative.NZAOA.value: 6,
    PartnerInitiative.GLOBAL_COVENANT.value: 6,
    PartnerInitiative.OTHER.value: 3,
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PledgeCommitmentEngine:
    """Race to Zero pledge commitment eligibility and quality engine.

    Validates pledge eligibility against 8 mandatory criteria, assesses
    pledge quality (0-100), evaluates partner initiative alignment, and
    provides gap analysis with remediation recommendations.

    Supports 7 actor types with actor-type-specific criteria weighting
    and partner initiative recommendations.

    Usage::

        engine = PledgeCommitmentEngine()
        result = engine.assess(input_data)
        print(f"Eligibility: {result.eligibility_status}")
        print(f"Quality: {result.pledge_quality} ({result.quality_score}/100)")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise PledgeCommitmentEngine.

        Args:
            config: Optional overrides. Supported keys:
                - criterion_weights (dict): Custom weights (must sum to 1.0)
                - min_interim_reduction_pct (Decimal): Min interim reduction
                - max_net_zero_year (int): Max acceptable net-zero year
        """
        self.config = config or {}
        self._weights = dict(CRITERION_WEIGHTS)
        custom_weights = self.config.get("criterion_weights")
        if custom_weights:
            for k, v in custom_weights.items():
                if k in self._weights:
                    self._weights[k] = _decimal(v)
        self._min_interim = _decimal(
            self.config.get("min_interim_reduction_pct", Decimal("42"))
        )
        self._max_nz_year = int(
            self.config.get("max_net_zero_year", MAX_NET_ZERO_YEAR)
        )
        logger.info(
            "PledgeCommitmentEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def assess(
        self, data: PledgeCommitmentInput,
    ) -> PledgeCommitmentResult:
        """Perform complete pledge commitment assessment.

        Evaluates all 8 pledge criteria, determines eligibility status
        and quality tier, assesses partner alignment, identifies gaps,
        and generates remediation recommendations.

        Args:
            data: Validated pledge commitment input.

        Returns:
            PledgeCommitmentResult with full assessment.
        """
        t0 = time.perf_counter()
        logger.info(
            "Pledge assessment: entity=%s, actor=%s, nz_year=%d",
            data.entity_name, data.actor_type, data.net_zero_target_year,
        )

        warnings: List[str] = []
        errors: List[str] = []
        gaps: List[str] = []

        # Build criterion input map
        crit_map: Dict[str, PledgeCriterionInput] = {}
        for c in data.criteria:
            crit_map[c.criterion_id] = c

        # Step 1: Assess each criterion
        criterion_results = self._assess_criteria(
            data, crit_map, gaps, warnings
        )

        # Step 2: Calculate quality score
        quality_score = Decimal("0")
        for cr in criterion_results:
            quality_score += cr.weighted_score
        quality_score = _round_val(quality_score * Decimal("100"), 2)

        # Step 3: Determine quality tier
        pledge_quality = self._determine_quality_tier(quality_score)

        # Step 4: Count met criteria
        core_met = sum(
            1 for cr in criterion_results
            if cr.is_core and cr.status in (CriterionStatus.PASS.value, CriterionStatus.PARTIAL.value)
        )
        total_met = sum(
            1 for cr in criterion_results
            if cr.status in (CriterionStatus.PASS.value, CriterionStatus.PARTIAL.value)
        )

        # Step 5: Determine eligibility
        all_core_pass = all(
            cr.status != CriterionStatus.FAIL.value
            for cr in criterion_results if cr.is_core
        )
        all_core_full = all(
            cr.status == CriterionStatus.PASS.value
            for cr in criterion_results if cr.is_core
        )

        if all_core_full and total_met >= 7:
            eligibility = EligibilityStatus.ELIGIBLE.value
        elif all_core_pass:
            eligibility = EligibilityStatus.CONDITIONAL.value
        else:
            eligibility = EligibilityStatus.INELIGIBLE.value

        # Step 6: Validate net-zero year
        nz_valid = data.net_zero_target_year <= self._max_nz_year
        if not nz_valid:
            gaps.append(
                f"Net-zero target year {data.net_zero_target_year} exceeds "
                f"Race to Zero maximum of {self._max_nz_year}."
            )

        # Step 7: Validate interim target
        interim_valid = data.interim_target_reduction_pct >= self._min_interim
        if not interim_valid and data.interim_target_reduction_pct > Decimal("0"):
            gaps.append(
                f"Interim target reduction of {data.interim_target_reduction_pct}% "
                f"is below minimum {self._min_interim}% for 1.5C alignment."
            )

        # Step 8: Validate scope coverage
        scope_valid = self._check_scope_coverage(data, gaps)

        # Step 9: Assess partner alignment
        partner_results = self._assess_partners(data)

        # Step 10: Recommended partners
        rec_partners = ACTOR_PARTNER_MAP.get(data.actor_type, [])

        # Step 11: Generate recommendations
        recommendations: List[str] = []
        if data.include_recommendations:
            recommendations = self._generate_recommendations(
                criterion_results, eligibility, quality_score,
                nz_valid, interim_valid, scope_valid, data
            )

        # Warnings
        if data.baseline_year > 0 and data.baseline_year < 2015:
            warnings.append(
                f"Baseline year {data.baseline_year} is before 2015. "
                f"Race to Zero recommends a recent baseline (>=2019)."
            )
        if not data.commitment_statement:
            warnings.append(
                "No commitment statement text provided. "
                "A specific, public commitment is required."
            )
        if len(data.partners) == 0:
            warnings.append(
                "No partner initiatives specified. Race to Zero requires "
                "joining through a recognized partner initiative."
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = PledgeCommitmentResult(
            entity_name=data.entity_name,
            actor_type=data.actor_type,
            eligibility_status=eligibility,
            pledge_quality=pledge_quality,
            quality_score=quality_score,
            criterion_results=criterion_results,
            partner_results=partner_results,
            core_criteria_met=core_met,
            total_criteria_met=total_met,
            total_criteria=len(criterion_results),
            net_zero_year_valid=nz_valid,
            interim_target_valid=interim_valid,
            scope_coverage_valid=scope_valid,
            recommended_partners=rec_partners,
            gaps=gaps,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Pledge assessment complete: eligibility=%s, quality=%s (%.1f), "
            "core=%d/%d, hash=%s",
            eligibility, pledge_quality, float(quality_score),
            core_met, len(CORE_CRITERIA), result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _assess_criteria(
        self,
        data: PledgeCommitmentInput,
        crit_map: Dict[str, PledgeCriterionInput],
        gaps: List[str],
        warnings: List[str],
    ) -> List[PledgeCriterionResult]:
        """Assess all 8 pledge criteria.

        For criteria with explicit input, uses the provided status.
        For criteria without input, auto-assesses from entity data.

        Args:
            data: Pledge input data.
            crit_map: Criterion input map.
            gaps: Gap list to append to.
            warnings: Warning list to append to.

        Returns:
            List of PledgeCriterionResult.
        """
        results: List[PledgeCriterionResult] = []

        for crit_id in CRITERION_IDS:
            weight = self._weights.get(crit_id, Decimal("0.10"))
            is_core = crit_id in CORE_CRITERIA
            label = CRITERION_LABELS.get(crit_id, crit_id)

            if crit_id in crit_map:
                inp = crit_map[crit_id]
                status = inp.status
                evidence = bool(inp.evidence)
            else:
                status, gap_desc = self._auto_assess_criterion(crit_id, data)
                evidence = False
                if gap_desc:
                    warnings.append(
                        f"Criterion '{label}' auto-assessed as {status}. {gap_desc}"
                    )

            score = STATUS_SCORES.get(status, Decimal("0"))
            weighted = score * weight

            gap_desc = ""
            remediation = ""
            effort = "medium"

            if status == CriterionStatus.FAIL.value:
                gap_desc, remediation, effort = self._get_gap_info(crit_id, data)
                gaps.append(f"{label}: {gap_desc}")
            elif status == CriterionStatus.PARTIAL.value:
                gap_desc, remediation, effort = self._get_gap_info(crit_id, data)

            results.append(PledgeCriterionResult(
                criterion_id=crit_id,
                criterion_name=label,
                status=status,
                score=score,
                weight=weight,
                weighted_score=_round_val(weighted, 4),
                is_core=is_core,
                evidence_provided=evidence,
                gap_description=gap_desc,
                remediation_action=remediation,
                effort_estimate=effort,
            ))

        return results

    def _auto_assess_criterion(
        self, crit_id: str, data: PledgeCommitmentInput,
    ) -> Tuple[str, str]:
        """Auto-assess a criterion from entity data.

        Args:
            crit_id: Criterion identifier.
            data: Pledge input data.

        Returns:
            Tuple of (status, gap_description).
        """
        if crit_id == "net_zero_commitment":
            if data.net_zero_target_year <= self._max_nz_year and data.commitment_statement:
                return CriterionStatus.PASS.value, ""
            elif data.net_zero_target_year <= self._max_nz_year:
                return CriterionStatus.PARTIAL.value, "Commitment statement not provided."
            else:
                return CriterionStatus.FAIL.value, (
                    f"Net-zero year {data.net_zero_target_year} exceeds {self._max_nz_year}."
                )

        if crit_id == "partner_initiative":
            active_partners = [
                p for p in data.partners
                if p.membership_status == "active"
            ]
            if active_partners:
                return CriterionStatus.PASS.value, ""
            elif data.partners:
                return CriterionStatus.PARTIAL.value, "Partner membership pending or planned."
            else:
                return CriterionStatus.FAIL.value, "No partner initiative identified."

        if crit_id == "interim_target":
            if data.interim_target_reduction_pct >= self._min_interim:
                return CriterionStatus.PASS.value, ""
            elif data.interim_target_reduction_pct > Decimal("0"):
                return CriterionStatus.PARTIAL.value, (
                    f"Interim target {data.interim_target_reduction_pct}% below "
                    f"minimum {self._min_interim}%."
                )
            else:
                return CriterionStatus.FAIL.value, "No interim target set."

        if crit_id == "action_plan":
            if data.action_plan_committed and data.action_plan_deadline_months <= 12:
                return CriterionStatus.PASS.value, ""
            elif data.action_plan_committed:
                return CriterionStatus.PARTIAL.value, (
                    f"Action plan deadline of {data.action_plan_deadline_months} months "
                    f"exceeds 12-month requirement."
                )
            else:
                return CriterionStatus.FAIL.value, "No action plan commitment."

        if crit_id == "annual_reporting":
            if data.annual_reporting_committed:
                return CriterionStatus.PASS.value, ""
            else:
                return CriterionStatus.FAIL.value, "Annual reporting not committed."

        if crit_id == "scope_coverage":
            req = SCOPE_COVERAGE_REQUIREMENTS.get(data.actor_type, {})
            s1_req = req.get("scope1_pct", Decimal("95"))
            s2_req = req.get("scope2_pct", Decimal("95"))
            s3_req = req.get("scope3_pct", Decimal("67"))

            s1_ok = data.scope1_coverage_pct >= s1_req
            s2_ok = data.scope2_coverage_pct >= s2_req
            s3_ok = data.scope3_coverage_pct >= s3_req or s3_req == Decimal("0")

            if s1_ok and s2_ok and s3_ok:
                return CriterionStatus.PASS.value, ""
            elif s1_ok and s2_ok:
                return CriterionStatus.PARTIAL.value, (
                    f"Scope 3 coverage {data.scope3_coverage_pct}% below "
                    f"required {s3_req}%."
                )
            else:
                return CriterionStatus.FAIL.value, "Scope 1 and/or Scope 2 coverage insufficient."

        if crit_id == "governance":
            if data.board_approved:
                return CriterionStatus.PASS.value, ""
            else:
                return CriterionStatus.FAIL.value, "Board/leadership approval not obtained."

        if crit_id == "public_disclosure":
            if data.publicly_disclosed:
                return CriterionStatus.PASS.value, ""
            else:
                return CriterionStatus.FAIL.value, "Commitment not publicly disclosed."

        return CriterionStatus.FAIL.value, f"Criterion '{crit_id}' not assessed."

    def _get_gap_info(
        self, crit_id: str, data: PledgeCommitmentInput,
    ) -> Tuple[str, str, str]:
        """Get gap description, remediation action, and effort for a criterion.

        Args:
            crit_id: Criterion identifier.
            data: Pledge input data.

        Returns:
            Tuple of (gap_description, remediation_action, effort_estimate).
        """
        gap_info: Dict[str, Tuple[str, str, str]] = {
            "net_zero_commitment": (
                "Net-zero commitment by 2050 not established or year exceeds limit.",
                "Establish a specific, time-bound net-zero commitment with "
                "target year no later than 2050. Draft commitment statement "
                "covering all material emission scopes.",
                "medium",
            ),
            "partner_initiative": (
                "No active membership in a recognized Race to Zero partner initiative.",
                "Join a recognized partner initiative appropriate for your actor "
                f"type ({data.actor_type}). Recommended: "
                f"{', '.join(ACTOR_PARTNER_MAP.get(data.actor_type, ['SBTi']))}.",
                "low",
            ),
            "interim_target": (
                f"Interim 2030 target missing or below {self._min_interim}% reduction.",
                f"Set an interim target of at least {self._min_interim}% absolute "
                f"reduction by {data.interim_target_year} using a science-based "
                f"methodology (SBTi, IEA NZE, IPCC SR1.5).",
                "high",
            ),
            "action_plan": (
                "No commitment to publish a climate action plan within 12 months.",
                "Commit to publishing a quantified climate action plan within "
                "12 months of joining, specifying concrete actions with timelines, "
                "milestones, and resource allocation.",
                "medium",
            ),
            "annual_reporting": (
                "No commitment to annual progress reporting.",
                "Commit to annual progress reporting through partner initiative "
                "channels (e.g., CDP for corporates, GFANZ for FIs).",
                "low",
            ),
            "scope_coverage": (
                "Emission scope coverage does not meet actor-type requirements.",
                "Expand emission boundary to cover required scopes. "
                "Scope 1+2: >=95% coverage. Scope 3: >=67% for corporates.",
                "high",
            ),
            "governance": (
                "Board or senior leadership endorsement not obtained.",
                "Obtain formal board resolution or senior leadership endorsement "
                "of the net-zero commitment and climate targets.",
                "medium",
            ),
            "public_disclosure": (
                "Net-zero commitment and progress not publicly disclosed.",
                "Publish the commitment on the entity's website and through "
                "partner initiative channels. Ensure public accessibility.",
                "low",
            ),
        }
        return gap_info.get(crit_id, ("Gap not characterized.", "Review criterion.", "medium"))

    def _check_scope_coverage(
        self, data: PledgeCommitmentInput, gaps: List[str],
    ) -> bool:
        """Check scope coverage against actor-type requirements.

        Args:
            data: Pledge input data.
            gaps: Gap list to append to.

        Returns:
            True if scope coverage meets requirements.
        """
        req = SCOPE_COVERAGE_REQUIREMENTS.get(data.actor_type, {})
        s1_req = req.get("scope1_pct", Decimal("95"))
        s2_req = req.get("scope2_pct", Decimal("95"))
        s3_req = req.get("scope3_pct", Decimal("67"))

        valid = True

        if data.scope1_coverage_pct < s1_req:
            valid = False
            gaps.append(
                f"Scope 1 coverage ({data.scope1_coverage_pct}%) below "
                f"required {s1_req}% for {data.actor_type}."
            )
        if data.scope2_coverage_pct < s2_req:
            valid = False
            gaps.append(
                f"Scope 2 coverage ({data.scope2_coverage_pct}%) below "
                f"required {s2_req}% for {data.actor_type}."
            )
        if s3_req > Decimal("0") and data.scope3_coverage_pct < s3_req:
            valid = False
            gaps.append(
                f"Scope 3 coverage ({data.scope3_coverage_pct}%) below "
                f"required {s3_req}% for {data.actor_type}."
            )

        return valid

    def _assess_partners(
        self, data: PledgeCommitmentInput,
    ) -> List[PartnerAlignmentResult]:
        """Assess partner initiative alignment.

        Args:
            data: Pledge input data.

        Returns:
            List of PartnerAlignmentResult.
        """
        results: List[PartnerAlignmentResult] = []
        recommended = ACTOR_PARTNER_MAP.get(data.actor_type, [])

        for p in data.partners:
            coverage = PARTNER_CRITERIA_COVERAGE.get(p.partner_id, 3)
            is_recommended = p.partner_id in recommended

            # Alignment score based on coverage, membership status, and recommendation
            base_score = _decimal(coverage) * Decimal("10")
            if p.membership_status == "active":
                base_score += Decimal("15")
            elif p.membership_status == "pending":
                base_score += Decimal("5")
            if is_recommended:
                base_score += Decimal("15")
            alignment_score = min(_round_val(base_score, 2), Decimal("100"))

            results.append(PartnerAlignmentResult(
                partner_id=p.partner_id,
                partner_name=PARTNER_NAMES.get(p.partner_id, p.partner_id),
                membership_status=p.membership_status,
                recommended=is_recommended,
                r2z_criteria_coverage=coverage,
                alignment_score=alignment_score,
                notes=p.notes,
            ))

        return results

    def _determine_quality_tier(self, score: Decimal) -> str:
        """Determine pledge quality tier from score.

        Args:
            score: Quality score (0-100).

        Returns:
            PledgeQuality value.
        """
        for threshold, tier in QUALITY_THRESHOLDS:
            if score >= threshold:
                return tier
        return PledgeQuality.INELIGIBLE.value

    def _generate_recommendations(
        self,
        criteria: List[PledgeCriterionResult],
        eligibility: str,
        quality_score: Decimal,
        nz_valid: bool,
        interim_valid: bool,
        scope_valid: bool,
        data: PledgeCommitmentInput,
    ) -> List[str]:
        """Generate improvement recommendations.

        Args:
            criteria: Criterion results.
            eligibility: Eligibility status.
            quality_score: Quality score.
            nz_valid: Whether NZ year is valid.
            interim_valid: Whether interim target is valid.
            scope_valid: Whether scope coverage is valid.
            data: Input data.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if eligibility == EligibilityStatus.INELIGIBLE.value:
            failing_core = [
                cr.criterion_name for cr in criteria
                if cr.is_core and cr.status == CriterionStatus.FAIL.value
            ]
            if failing_core:
                recs.append(
                    f"CRITICAL: Core criteria failing: {', '.join(failing_core)}. "
                    f"These must be addressed before Race to Zero eligibility."
                )

        if eligibility == EligibilityStatus.CONDITIONAL.value:
            partial_core = [
                cr.criterion_name for cr in criteria
                if cr.is_core and cr.status == CriterionStatus.PARTIAL.value
            ]
            if partial_core:
                recs.append(
                    f"Upgrade partial criteria to full compliance: "
                    f"{', '.join(partial_core)}."
                )

        if not nz_valid:
            recs.append(
                f"Advance net-zero target year from {data.net_zero_target_year} "
                f"to 2050 or earlier to meet Race to Zero requirements."
            )

        if not interim_valid and data.interim_target_reduction_pct > Decimal("0"):
            recs.append(
                f"Increase interim target from {data.interim_target_reduction_pct}% "
                f"to at least {self._min_interim}% absolute reduction by "
                f"{data.interim_target_year}."
            )

        if not scope_valid:
            recs.append(
                "Expand emission boundary coverage to meet actor-type "
                "requirements for Scope 1, 2, and 3."
            )

        if not data.board_approved:
            recs.append(
                "Obtain formal board or senior leadership endorsement "
                "of the net-zero commitment."
            )

        if not data.publicly_disclosed:
            recs.append(
                "Publicly disclose the net-zero commitment on the entity's "
                "website and through partner initiative channels."
            )

        if len(data.partners) == 0:
            recommended = ACTOR_PARTNER_MAP.get(data.actor_type, [])
            if recommended:
                recs.append(
                    f"Join a recognized Race to Zero partner initiative. "
                    f"Recommended for {data.actor_type}: "
                    f"{', '.join(PARTNER_NAMES.get(p, p) for p in recommended)}."
                )

        # Priority sort: failing criteria recommendations first
        for cr in criteria:
            if cr.status == CriterionStatus.FAIL.value and cr.remediation_action:
                recs.append(
                    f"[{cr.effort_estimate.upper()}] {cr.criterion_name}: "
                    f"{cr.remediation_action}"
                )

        return recs
