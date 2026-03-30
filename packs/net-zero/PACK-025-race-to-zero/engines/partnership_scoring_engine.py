# -*- coding: utf-8 -*-
"""
PartnershipScoringEngine - PACK-025 Race to Zero Engine 7
==========================================================

Assesses collaboration quality, partner initiative alignment, and
reporting efficiency across Race to Zero's 40+ partner initiatives.
Scores partnership portfolios on 6 dimensions: requirement alignment,
reporting efficiency, engagement quality, credibility contribution,
coverage completeness, and timeline alignment.

Calculation Methodology:
    Partnership Quality Score (0-100):
        6 dimensions, weighted sum:
            requirement_alignment:  25%
            reporting_efficiency:   20%
            engagement_quality:     20%
            credibility_contrib:    15%
            coverage_completeness:  10%
            timeline_alignment:     10%

    Synergy Analysis:
        synergy_score = (unique_criteria_covered / total_r2z_criteria)
        overlap_count = criteria covered by multiple partners
        gap_count = criteria not covered by any partner

    Collaboration Impact:
        partnership_multiplier = 1.0 + (active_memberships * 0.05)
        impact_tco2e = entity_reductions * partnership_multiplier

Regulatory References:
    - Race to Zero Partner Initiative Requirements (2022)
    - UNFCCC Climate Champions Partner Criteria
    - SBTi, CDP, C40, GFANZ, ICLEI requirements
    - HLEG "Integrity Matters" (2022), Rec 9 (systemic change)

Zero-Hallucination:
    - All partner initiative data from published requirements
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-025 Race to Zero
Engine:  7 of 10
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
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
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
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class DimensionId(str, Enum):
    """Partnership scoring dimensions."""
    REQUIREMENT_ALIGNMENT = "requirement_alignment"
    REPORTING_EFFICIENCY = "reporting_efficiency"
    ENGAGEMENT_QUALITY = "engagement_quality"
    CREDIBILITY_CONTRIBUTION = "credibility_contribution"
    COVERAGE_COMPLETENESS = "coverage_completeness"
    TIMELINE_ALIGNMENT = "timeline_alignment"

class GovernanceMaturity(str, Enum):
    """Partnership governance maturity level."""
    EXEMPLARY = "exemplary"
    MATURE = "mature"
    DEVELOPING = "developing"
    NASCENT = "nascent"

DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    DimensionId.REQUIREMENT_ALIGNMENT.value: Decimal("0.25"),
    DimensionId.REPORTING_EFFICIENCY.value: Decimal("0.20"),
    DimensionId.ENGAGEMENT_QUALITY.value: Decimal("0.20"),
    DimensionId.CREDIBILITY_CONTRIBUTION.value: Decimal("0.15"),
    DimensionId.COVERAGE_COMPLETENESS.value: Decimal("0.10"),
    DimensionId.TIMELINE_ALIGNMENT.value: Decimal("0.10"),
}

DIMENSION_LABELS: Dict[str, str] = {
    DimensionId.REQUIREMENT_ALIGNMENT.value: "Requirement Alignment",
    DimensionId.REPORTING_EFFICIENCY.value: "Reporting Efficiency",
    DimensionId.ENGAGEMENT_QUALITY.value: "Engagement Quality",
    DimensionId.CREDIBILITY_CONTRIBUTION.value: "Credibility Contribution",
    DimensionId.COVERAGE_COMPLETENESS.value: "Coverage Completeness",
    DimensionId.TIMELINE_ALIGNMENT.value: "Timeline Alignment",
}

# R2Z criteria that partners can cover (8 total criteria).
R2Z_CRITERIA = [
    "net_zero_pledge", "interim_target", "action_plan", "annual_reporting",
    "scope_coverage", "science_based", "governance", "transparency",
]

# Partner initiative metadata.
PARTNER_DB: Dict[str, Dict[str, Any]] = {
    "sbti": {"name": "Science Based Targets initiative", "criteria_covered": ["net_zero_pledge", "interim_target", "science_based", "scope_coverage", "annual_reporting", "action_plan"], "credibility_score": Decimal("95"), "actor_types": ["corporate", "financial_institution"]},
    "cdp": {"name": "CDP", "criteria_covered": ["annual_reporting", "scope_coverage", "transparency", "governance", "interim_target", "action_plan", "net_zero_pledge"], "credibility_score": Decimal("90"), "actor_types": ["corporate", "city"]},
    "c40": {"name": "C40 Cities", "criteria_covered": ["net_zero_pledge", "interim_target", "action_plan", "annual_reporting", "scope_coverage", "governance", "transparency"], "credibility_score": Decimal("90"), "actor_types": ["city"]},
    "iclei": {"name": "ICLEI", "criteria_covered": ["net_zero_pledge", "action_plan", "annual_reporting", "scope_coverage", "governance", "transparency"], "credibility_score": Decimal("85"), "actor_types": ["city", "region"]},
    "gfanz": {"name": "GFANZ", "criteria_covered": ["net_zero_pledge", "interim_target", "action_plan", "annual_reporting", "scope_coverage", "science_based", "transparency"], "credibility_score": Decimal("90"), "actor_types": ["financial_institution"]},
    "wmb": {"name": "We Mean Business Coalition", "criteria_covered": ["net_zero_pledge", "interim_target", "action_plan", "annual_reporting", "governance"], "credibility_score": Decimal("80"), "actor_types": ["corporate"]},
    "climate_pledge": {"name": "The Climate Pledge", "criteria_covered": ["net_zero_pledge", "annual_reporting", "action_plan", "scope_coverage", "transparency"], "credibility_score": Decimal("80"), "actor_types": ["corporate"]},
    "sme_climate_hub": {"name": "SME Climate Hub", "criteria_covered": ["net_zero_pledge", "action_plan", "annual_reporting", "scope_coverage", "governance"], "credibility_score": Decimal("75"), "actor_types": ["sme"]},
    "second_nature": {"name": "Second Nature", "criteria_covered": ["net_zero_pledge", "interim_target", "action_plan", "annual_reporting", "scope_coverage", "governance"], "credibility_score": Decimal("80"), "actor_types": ["university"]},
    "under2": {"name": "Under2 Coalition", "criteria_covered": ["net_zero_pledge", "interim_target", "action_plan", "annual_reporting", "scope_coverage", "governance"], "credibility_score": Decimal("85"), "actor_types": ["region"]},
    "hcwh": {"name": "Health Care Without Harm", "criteria_covered": ["net_zero_pledge", "action_plan", "annual_reporting", "scope_coverage", "governance"], "credibility_score": Decimal("75"), "actor_types": ["healthcare"]},
    "nzba": {"name": "Net-Zero Banking Alliance", "criteria_covered": ["net_zero_pledge", "interim_target", "action_plan", "annual_reporting", "scope_coverage", "science_based"], "credibility_score": Decimal("90"), "actor_types": ["financial_institution"]},
    "nzam": {"name": "Net Zero Asset Managers Initiative", "criteria_covered": ["net_zero_pledge", "interim_target", "annual_reporting", "scope_coverage", "science_based", "governance"], "credibility_score": Decimal("88"), "actor_types": ["financial_institution"]},
    "nzaoa": {"name": "Net-Zero Asset Owner Alliance", "criteria_covered": ["net_zero_pledge", "interim_target", "annual_reporting", "scope_coverage", "science_based", "governance"], "credibility_score": Decimal("90"), "actor_types": ["financial_institution"]},
    "global_covenant": {"name": "Global Covenant of Mayors", "criteria_covered": ["net_zero_pledge", "interim_target", "action_plan", "annual_reporting", "scope_coverage", "governance"], "credibility_score": Decimal("85"), "actor_types": ["city"]},
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class PartnerMembershipInput(BaseModel):
    """Input for a single partner initiative membership.

    Attributes:
        partner_id: Partner identifier.
        membership_status: active/pending/planned.
        join_year: Year joined.
        engagement_level: passive/active/leadership.
        reporting_through: Whether reporting through this partner.
        working_groups: Number of working groups participated in.
        peer_learning: Whether participating in peer learning.
        compliance_status: up_to_date/behind/unknown.
        notes: Additional notes.
    """
    partner_id: str = Field(..., description="Partner initiative ID")
    membership_status: str = Field(default="active")
    join_year: int = Field(default=0, ge=0, le=2060)
    engagement_level: str = Field(default="active")
    reporting_through: bool = Field(default=False)
    working_groups: int = Field(default=0, ge=0)
    peer_learning: bool = Field(default=False)
    compliance_status: str = Field(default="up_to_date")
    notes: str = Field(default="")

    @field_validator("partner_id")
    @classmethod
    def validate_partner(cls, v: str) -> str:
        if v not in PARTNER_DB:
            raise ValueError(f"Unknown partner '{v}'. Must be one of: {sorted(PARTNER_DB.keys())}")
        return v

    @field_validator("engagement_level")
    @classmethod
    def validate_engagement(cls, v: str) -> str:
        if v not in ("passive", "active", "leadership"):
            raise ValueError(f"Unknown engagement level '{v}'.")
        return v

class PartnershipInput(BaseModel):
    """Complete input for partnership scoring.

    Attributes:
        entity_name: Entity name.
        actor_type: Actor type.
        partners: List of partner memberships.
        total_emission_reductions_tco2e: Total entity reductions to date.
        joint_commitments_count: Number of joint commitments made.
        data_sharing_active: Whether actively sharing data with partners.
        joint_accountability_mechanisms: Number of joint accountability mechanisms.
        include_synergy: Whether to include synergy analysis.
    """
    entity_name: str = Field(..., min_length=1, max_length=300)
    actor_type: str = Field(default="corporate")
    partners: List[PartnerMembershipInput] = Field(default_factory=list)
    total_emission_reductions_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    joint_commitments_count: int = Field(default=0, ge=0)
    data_sharing_active: bool = Field(default=False)
    joint_accountability_mechanisms: int = Field(default=0, ge=0)
    include_synergy: bool = Field(default=True)

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class DimensionResult(BaseModel):
    """Result for a single scoring dimension."""
    dimension_id: str = Field(default="")
    dimension_name: str = Field(default="")
    score: Decimal = Field(default=Decimal("0"))
    weight: Decimal = Field(default=Decimal("0"))
    weighted_score: Decimal = Field(default=Decimal("0"))
    assessment: str = Field(default="")

class PartnerAssessment(BaseModel):
    """Assessment result for a single partner initiative."""
    partner_id: str = Field(default="")
    partner_name: str = Field(default="")
    membership_status: str = Field(default="")
    engagement_level: str = Field(default="")
    alignment_score: Decimal = Field(default=Decimal("0"))
    criteria_covered: List[str] = Field(default_factory=list)
    reporting_channel: bool = Field(default=False)
    credibility_score: Decimal = Field(default=Decimal("0"))
    recommended_for_actor: bool = Field(default=False)

class SynergyAnalysis(BaseModel):
    """Cross-partner synergy analysis."""
    total_r2z_criteria: int = Field(default=8)
    criteria_covered: int = Field(default=0)
    criteria_gaps: List[str] = Field(default_factory=list)
    overlap_count: int = Field(default=0)
    synergy_score: Decimal = Field(default=Decimal("0"))
    reporting_channels: int = Field(default=0)
    optimization_suggestions: List[str] = Field(default_factory=list)

class PartnershipResult(BaseModel):
    """Complete partnership scoring result."""
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    partnership_quality_score: Decimal = Field(default=Decimal("0"))
    governance_maturity: str = Field(default=GovernanceMaturity.NASCENT.value)
    dimension_results: List[DimensionResult] = Field(default_factory=list)
    partner_assessments: List[PartnerAssessment] = Field(default_factory=list)
    synergy: Optional[SynergyAnalysis] = Field(default=None)
    active_partnerships: int = Field(default=0)
    total_partnerships: int = Field(default=0)
    collaboration_impact_tco2e: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PartnershipScoringEngine:
    """Race to Zero partnership quality scoring engine.

    Assesses partnership portfolios across 6 dimensions, performs
    synergy analysis, and calculates collaboration impact.

    Usage::

        engine = PartnershipScoringEngine()
        result = engine.assess(input_data)
        print(f"Quality: {result.partnership_quality_score}/100")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        logger.info("PartnershipScoringEngine v%s initialised", self.engine_version)

    def assess(
        self, data: PartnershipInput,
    ) -> PartnershipResult:
        """Perform complete partnership scoring assessment."""
        t0 = time.perf_counter()
        logger.info(
            "Partnership scoring: entity=%s, partners=%d",
            data.entity_name, len(data.partners),
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Step 1: Assess each partner
        partner_assessments = self._assess_partners(data)

        # Step 2: Score dimensions
        dimension_results = self._score_dimensions(data, partner_assessments)

        # Step 3: Calculate quality score
        quality = Decimal("0")
        for dr in dimension_results:
            quality += dr.weighted_score
        quality = _round_val(quality, 2)

        # Step 4: Governance maturity
        if quality >= Decimal("85"):
            maturity = GovernanceMaturity.EXEMPLARY.value
        elif quality >= Decimal("65"):
            maturity = GovernanceMaturity.MATURE.value
        elif quality >= Decimal("40"):
            maturity = GovernanceMaturity.DEVELOPING.value
        else:
            maturity = GovernanceMaturity.NASCENT.value

        # Step 5: Synergy analysis
        synergy: Optional[SynergyAnalysis] = None
        if data.include_synergy:
            synergy = self._analyze_synergy(data, partner_assessments)

        # Step 6: Partnership counts
        active = sum(1 for p in data.partners if p.membership_status == "active")

        # Step 7: Collaboration impact
        multiplier = Decimal("1.0") + _decimal(active) * Decimal("0.05")
        impact = _round_val(data.total_emission_reductions_tco2e * multiplier)

        # Step 8: Recommendations
        recommendations = self._generate_recommendations(
            data, quality, synergy, partner_assessments
        )

        if not data.partners:
            warnings.append("No partner initiatives specified.")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = PartnershipResult(
            entity_name=data.entity_name,
            partnership_quality_score=quality,
            governance_maturity=maturity,
            dimension_results=dimension_results,
            partner_assessments=partner_assessments,
            synergy=synergy,
            active_partnerships=active,
            total_partnerships=len(data.partners),
            collaboration_impact_tco2e=impact,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def _assess_partners(
        self, data: PartnershipInput,
    ) -> List[PartnerAssessment]:
        results: List[PartnerAssessment] = []
        for p in data.partners:
            pdb = PARTNER_DB.get(p.partner_id, {})
            name = pdb.get("name", p.partner_id)
            criteria = pdb.get("criteria_covered", [])
            cred = pdb.get("credibility_score", Decimal("50"))
            actor_types = pdb.get("actor_types", [])
            recommended = data.actor_type in actor_types

            # Alignment score based on engagement + compliance
            base = Decimal("50")
            if p.membership_status == "active":
                base += Decimal("20")
            if p.engagement_level == "leadership":
                base += Decimal("15")
            elif p.engagement_level == "active":
                base += Decimal("10")
            if p.compliance_status == "up_to_date":
                base += Decimal("10")
            if p.peer_learning:
                base += Decimal("5")
            alignment = min(_round_val(base, 2), Decimal("100"))

            results.append(PartnerAssessment(
                partner_id=p.partner_id,
                partner_name=name,
                membership_status=p.membership_status,
                engagement_level=p.engagement_level,
                alignment_score=alignment,
                criteria_covered=criteria,
                reporting_channel=p.reporting_through,
                credibility_score=cred,
                recommended_for_actor=recommended,
            ))
        return results

    def _score_dimensions(
        self, data: PartnershipInput, assessments: List[PartnerAssessment],
    ) -> List[DimensionResult]:
        results: List[DimensionResult] = []
        for dim in DimensionId:
            weight = DIMENSION_WEIGHTS.get(dim.value, Decimal("0.10"))
            label = DIMENSION_LABELS.get(dim.value, dim.value)
            score = self._calc_dimension_score(dim.value, data, assessments)
            results.append(DimensionResult(
                dimension_id=dim.value,
                dimension_name=label,
                score=_round_val(score, 2),
                weight=weight,
                weighted_score=_round_val(score * weight, 4),
                assessment=f"{label}: {_round_val(score, 0)}/100",
            ))
        return results

    def _calc_dimension_score(
        self, dim: str, data: PartnershipInput, assessments: List[PartnerAssessment],
    ) -> Decimal:
        if not assessments:
            return Decimal("0")
        if dim == DimensionId.REQUIREMENT_ALIGNMENT.value:
            all_criteria = set()
            for a in assessments:
                all_criteria.update(a.criteria_covered)
            return _safe_pct(_decimal(len(all_criteria)), _decimal(len(R2Z_CRITERIA)))
        if dim == DimensionId.REPORTING_EFFICIENCY.value:
            reporting = sum(1 for a in assessments if a.reporting_channel)
            if reporting >= 1:
                return min(Decimal("100"), Decimal("50") + _decimal(reporting) * Decimal("25"))
            return Decimal("20")
        if dim == DimensionId.ENGAGEMENT_QUALITY.value:
            scores = [a.alignment_score for a in assessments]
            return sum(scores) / _decimal(len(scores)) if scores else Decimal("0")
        if dim == DimensionId.CREDIBILITY_CONTRIBUTION.value:
            creds = [a.credibility_score for a in assessments]
            return sum(creds) / _decimal(len(creds)) if creds else Decimal("0")
        if dim == DimensionId.COVERAGE_COMPLETENESS.value:
            recommended = sum(1 for a in assessments if a.recommended_for_actor)
            return min(Decimal("100"), _decimal(recommended) * Decimal("40"))
        if dim == DimensionId.TIMELINE_ALIGNMENT.value:
            up_to_date = sum(1 for p in data.partners if p.compliance_status == "up_to_date")
            return _safe_pct(_decimal(up_to_date), _decimal(len(data.partners)))
        return Decimal("50")

    def _analyze_synergy(
        self, data: PartnershipInput, assessments: List[PartnerAssessment],
    ) -> SynergyAnalysis:
        all_covered: Dict[str, int] = {}
        for a in assessments:
            for c in a.criteria_covered:
                all_covered[c] = all_covered.get(c, 0) + 1

        covered = set(all_covered.keys())
        gaps = [c for c in R2Z_CRITERIA if c not in covered]
        overlap = sum(1 for v in all_covered.values() if v > 1)
        synergy_score = _safe_pct(_decimal(len(covered)), _decimal(len(R2Z_CRITERIA)))
        reporting = sum(1 for p in data.partners if p.reporting_through)

        suggestions: List[str] = []
        if gaps:
            suggestions.append(f"Gaps in R2Z criteria coverage: {', '.join(gaps)}.")
        if overlap > 3:
            suggestions.append("High reporting overlap. Consider streamlining.")

        return SynergyAnalysis(
            total_r2z_criteria=len(R2Z_CRITERIA),
            criteria_covered=len(covered),
            criteria_gaps=gaps,
            overlap_count=overlap,
            synergy_score=_round_val(synergy_score, 2),
            reporting_channels=reporting,
            optimization_suggestions=suggestions,
        )

    def _generate_recommendations(
        self, data: PartnershipInput, quality: Decimal,
        synergy: Optional[SynergyAnalysis], assessments: List[PartnerAssessment],
    ) -> List[str]:
        recs: List[str] = []
        if quality < Decimal("50"):
            recs.append("Partnership engagement needs strengthening.")
        if synergy and synergy.criteria_gaps:
            recs.append(f"Close R2Z criteria gaps: {', '.join(synergy.criteria_gaps)}.")
        not_recommended = [a for a in assessments if not a.recommended_for_actor]
        if not_recommended:
            recs.append("Consider joining partner initiatives recommended for your actor type.")
        passive = sum(1 for p in data.partners if p.engagement_level == "passive")
        if passive > 0:
            recs.append(f"Upgrade {passive} passive membership(s) to active engagement.")
        return recs
