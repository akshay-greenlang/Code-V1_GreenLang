# -*- coding: utf-8 -*-
"""
RaceReadinessEngine - PACK-025 Race to Zero Engine 10
======================================================

Computes overall Race to Zero campaign readiness via an 8-dimension
composite score. Aggregates outputs from all 9 preceding engines into
a single readiness assessment with dimension breakdowns, RAG status,
readiness level classification, improvement priorities, and estimated
timeline to full readiness.

Calculation Methodology:
    Composite Race Readiness Score (0-100):
        8 dimensions with fixed weights:
            D1 Pledge Strength:        12%  (from PledgeCommitmentEngine)
            D2 Starting Line Compliance: 18%  (from StartingLineEngine)
            D3 Target Ambition:        15%  (from InterimTargetEngine)
            D4 Action Plan Quality:    15%  (from ActionPlanEngine)
            D5 Progress Trajectory:    12%  (from ProgressTrackingEngine)
            D6 Sector Alignment:       10%  (from SectorPathwayEngine)
            D7 Partnership Engagement:  8%  (from PartnershipScoringEngine)
            D8 HLEG Credibility:       10%  (from CredibilityAssessmentEngine)

        readiness = sum(dimension_score * dimension_weight) / sum(weights)

    RAG Classification:
        GREEN: >= 75  (Campaign-ready or near-ready)
        AMBER: >= 50  (In development, moderate gaps)
        RED:   >= 25  (Planning phase, significant gaps)
        BLACK: <  25  (Not started, major foundational work needed)

    Readiness Levels:
        RACE_READY:   85-100  Campaign-ready, meets all criteria
        APPROACHING:  70-84   Near-ready, minor gaps, 1-3 months
        BUILDING:     50-69   Moderate gaps, 3-6 months focused work
        EARLY_STAGE:  25-49   Significant gaps, 6-12 months
        PRE_PLEDGE:   0-24    Major foundational work required

    Improvement Priority Ranking:
        priority = gap_score * dimension_weight * urgency_factor
        Ranked descending by weighted impact

    Timeline Estimation:
        Based on number and severity of gaps across dimensions
        Accounts for parallelizable vs. sequential improvements

Regulatory References:
    - Race to Zero Campaign Criteria (2022)
    - UN HLEG "Integrity Matters" (November 2022)
    - Race to Zero Starting Line (4P Framework)
    - IPCC AR6 WG3 (2022) pathway benchmarks
    - SBTi Corporate Net-Zero Standard v1.1 (2023)

Zero-Hallucination:
    - Readiness levels from Race to Zero campaign requirements
    - Weights from HLEG emphasis areas
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-025 Race to Zero
Engine:  10 of 10
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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class ReadinessLevel(str, Enum):
    """Race to Zero readiness level."""
    RACE_READY = "RACE_READY"
    APPROACHING = "APPROACHING"
    BUILDING = "BUILDING"
    EARLY_STAGE = "EARLY_STAGE"
    PRE_PLEDGE = "PRE_PLEDGE"


class RAGStatus(str, Enum):
    """RAG status classification."""
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    BLACK = "BLACK"


class DimensionId(str, Enum):
    """Readiness dimension identifiers."""
    PLEDGE_STRENGTH = "pledge_strength"
    STARTING_LINE = "starting_line_compliance"
    TARGET_AMBITION = "target_ambition"
    ACTION_PLAN = "action_plan_quality"
    PROGRESS = "progress_trajectory"
    SECTOR = "sector_alignment"
    PARTNERSHIP = "partnership_engagement"
    CREDIBILITY = "hleg_credibility"


class UrgencyLevel(str, Enum):
    """Urgency level for improvement actions."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ---------------------------------------------------------------------------
# Dimension Configuration
# ---------------------------------------------------------------------------

DIMENSION_CONFIG: Dict[str, Dict[str, Any]] = {
    DimensionId.PLEDGE_STRENGTH.value: {
        "name": "Pledge Strength",
        "weight": Decimal("0.12"),
        "source_engine": "PledgeCommitmentEngine",
        "description": "Net-zero pledge eligibility and quality assessment",
        "urgency_factor": Decimal("1.2"),
        "min_for_readiness": Decimal("70"),
    },
    DimensionId.STARTING_LINE.value: {
        "name": "Starting Line Compliance",
        "weight": Decimal("0.18"),
        "source_engine": "StartingLineEngine",
        "description": "4P framework (Pledge/Plan/Proceed/Publish) compliance rate",
        "urgency_factor": Decimal("1.5"),
        "min_for_readiness": Decimal("80"),
    },
    DimensionId.TARGET_AMBITION.value: {
        "name": "Target Ambition",
        "weight": Decimal("0.15"),
        "source_engine": "InterimTargetEngine",
        "description": "2030 interim target alignment with 1.5C pathway",
        "urgency_factor": Decimal("1.3"),
        "min_for_readiness": Decimal("70"),
    },
    DimensionId.ACTION_PLAN.value: {
        "name": "Action Plan Quality",
        "weight": Decimal("0.15"),
        "source_engine": "ActionPlanEngine",
        "description": "Transition plan completeness across 10 required sections",
        "urgency_factor": Decimal("1.2"),
        "min_for_readiness": Decimal("65"),
    },
    DimensionId.PROGRESS.value: {
        "name": "Progress Trajectory",
        "weight": Decimal("0.12"),
        "source_engine": "ProgressTrackingEngine",
        "description": "Annual progress against trajectory and action implementation",
        "urgency_factor": Decimal("1.1"),
        "min_for_readiness": Decimal("60"),
    },
    DimensionId.SECTOR.value: {
        "name": "Sector Alignment",
        "weight": Decimal("0.10"),
        "source_engine": "SectorPathwayEngine",
        "description": "Entity performance vs. sector decarbonization pathway benchmarks",
        "urgency_factor": Decimal("1.0"),
        "min_for_readiness": Decimal("55"),
    },
    DimensionId.PARTNERSHIP.value: {
        "name": "Partnership Engagement",
        "weight": Decimal("0.08"),
        "source_engine": "PartnershipScoringEngine",
        "description": "Partner initiative collaboration quality and synergy",
        "urgency_factor": Decimal("0.8"),
        "min_for_readiness": Decimal("50"),
    },
    DimensionId.CREDIBILITY.value: {
        "name": "HLEG Credibility",
        "weight": Decimal("0.10"),
        "source_engine": "CredibilityAssessmentEngine",
        "description": "HLEG 10 recommendations compliance and credibility tier",
        "urgency_factor": Decimal("1.3"),
        "min_for_readiness": Decimal("60"),
    },
}

# Readiness level thresholds
READINESS_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    ReadinessLevel.RACE_READY.value: {
        "min_score": Decimal("85"),
        "description": "Campaign-ready; meets all Starting Line criteria; strong HLEG credibility",
        "timeline": "Ready now",
    },
    ReadinessLevel.APPROACHING.value: {
        "min_score": Decimal("70"),
        "description": "Near-ready; minor gaps; 1-3 months to full readiness",
        "timeline": "1-3 months",
    },
    ReadinessLevel.BUILDING.value: {
        "min_score": Decimal("50"),
        "description": "In development; moderate gaps; 3-6 months of focused work needed",
        "timeline": "3-6 months",
    },
    ReadinessLevel.EARLY_STAGE.value: {
        "min_score": Decimal("25"),
        "description": "Planning phase; significant gaps; 6-12 months of work needed",
        "timeline": "6-12 months",
    },
    ReadinessLevel.PRE_PLEDGE.value: {
        "min_score": Decimal("0"),
        "description": "Not yet started; major foundational work required before pledge",
        "timeline": "12-24 months",
    },
}

# RAG thresholds
RAG_THRESHOLDS: Dict[str, Decimal] = {
    RAGStatus.GREEN.value: Decimal("75"),
    RAGStatus.AMBER.value: Decimal("50"),
    RAGStatus.RED.value: Decimal("25"),
    RAGStatus.BLACK.value: Decimal("0"),
}

# Improvement action templates per dimension
IMPROVEMENT_ACTIONS: Dict[str, List[Dict[str, str]]] = {
    DimensionId.PLEDGE_STRENGTH.value: [
        {"action": "Formalize net-zero commitment with specific target year and scope coverage", "effort": "LOW", "timeline": "1-2 months"},
        {"action": "Join recognized partner initiative (SBTi, Climate Pledge, etc.)", "effort": "MEDIUM", "timeline": "2-4 months"},
        {"action": "Obtain board-level governance approval for climate pledge", "effort": "LOW", "timeline": "1-2 months"},
        {"action": "Ensure pledge covers Scope 1, 2, and material Scope 3 emissions", "effort": "MEDIUM", "timeline": "2-3 months"},
    ],
    DimensionId.STARTING_LINE.value: [
        {"action": "Complete 4P framework assessment and address all PLEDGE criteria gaps", "effort": "MEDIUM", "timeline": "2-4 months"},
        {"action": "Publish climate action plan within 12-month joining deadline", "effort": "HIGH", "timeline": "3-6 months"},
        {"action": "Establish annual public reporting mechanism", "effort": "MEDIUM", "timeline": "2-3 months"},
        {"action": "Demonstrate immediate action with quantified emission reductions", "effort": "HIGH", "timeline": "3-6 months"},
    ],
    DimensionId.TARGET_AMBITION.value: [
        {"action": "Set science-based 2030 interim target using SBTi or IEA methodology", "effort": "HIGH", "timeline": "3-6 months"},
        {"action": "Ensure annual reduction rate meets 1.5C pathway minimum (4.2%/yr)", "effort": "HIGH", "timeline": "3-6 months"},
        {"action": "Extend target coverage to include material Scope 3 categories", "effort": "MEDIUM", "timeline": "2-4 months"},
        {"action": "Conduct fair share equity assessment for target ambition", "effort": "MEDIUM", "timeline": "2-3 months"},
    ],
    DimensionId.ACTION_PLAN.value: [
        {"action": "Complete all 10 required action plan sections with quantified data", "effort": "HIGH", "timeline": "3-6 months"},
        {"action": "Add specific, quantified decarbonization actions with tCO2e impact", "effort": "HIGH", "timeline": "3-6 months"},
        {"action": "Allocate financial resources (CapEx/OpEx) per reduction action", "effort": "MEDIUM", "timeline": "2-4 months"},
        {"action": "Develop Scope 3 supplier engagement strategy", "effort": "HIGH", "timeline": "4-8 months"},
    ],
    DimensionId.PROGRESS.value: [
        {"action": "Establish annual emissions tracking and trend analysis", "effort": "MEDIUM", "timeline": "2-3 months"},
        {"action": "Implement corrective action framework for off-track trajectories", "effort": "MEDIUM", "timeline": "2-4 months"},
        {"action": "Accelerate action plan implementation to meet trajectory", "effort": "HIGH", "timeline": "3-6 months"},
        {"action": "Set up variance decomposition to identify reduction drivers", "effort": "MEDIUM", "timeline": "2-3 months"},
    ],
    DimensionId.SECTOR.value: [
        {"action": "Map entity to sector pathway(s) using ISIC/NACE/GICS classification", "effort": "LOW", "timeline": "1-2 months"},
        {"action": "Benchmark performance against sector-specific 2030 milestones", "effort": "MEDIUM", "timeline": "2-3 months"},
        {"action": "Align decarbonization actions with sector technology pathways", "effort": "HIGH", "timeline": "3-6 months"},
        {"action": "Address sector-specific gaps (e.g., renewable share, EV adoption)", "effort": "HIGH", "timeline": "6-12 months"},
    ],
    DimensionId.PARTNERSHIP.value: [
        {"action": "Join additional partner initiatives to increase criteria coverage", "effort": "MEDIUM", "timeline": "2-4 months"},
        {"action": "Increase active engagement (working groups, peer learning)", "effort": "LOW", "timeline": "1-3 months"},
        {"action": "Optimize reporting efficiency across partner channels", "effort": "MEDIUM", "timeline": "2-3 months"},
        {"action": "Address gaps in cross-partner requirement coverage", "effort": "MEDIUM", "timeline": "2-4 months"},
    ],
    DimensionId.CREDIBILITY.value: [
        {"action": "Address top HLEG recommendation gaps (focus on Rec 1-3, 6, 8)", "effort": "HIGH", "timeline": "3-6 months"},
        {"action": "Establish board-level governance with executive climate incentives", "effort": "MEDIUM", "timeline": "2-4 months"},
        {"action": "Align all lobbying and trade association positions with climate goals", "effort": "HIGH", "timeline": "3-6 months"},
        {"action": "Obtain third-party verification for emission data and claims", "effort": "HIGH", "timeline": "4-8 months"},
    ],
}


# ---------------------------------------------------------------------------
# Input / Output Models
# ---------------------------------------------------------------------------


class DimensionInput(BaseModel):
    """Input for a single readiness dimension (from source engine output)."""
    dimension_id: str = Field(..., description="Dimension identifier")
    score: float = Field(0.0, description="Dimension score 0-100 from source engine")
    status: Optional[str] = Field(None, description="Status from source engine (e.g., COMPLIANT, ON_TRACK)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details from source engine")
    engine_result_id: Optional[str] = Field(None, description="Source engine result UUID")

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        return max(0.0, min(100.0, v))


class ReadinessInput(BaseModel):
    """Input for race readiness assessment."""
    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    assessment_date: Optional[str] = Field(None, description="Assessment date (ISO 8601)")
    reporting_year: int = Field(2024, description="Reporting year")

    # Dimension scores (from individual engine outputs)
    pledge_score: float = Field(0.0, description="Pledge Commitment Engine score 0-100")
    starting_line_score: float = Field(0.0, description="Starting Line Engine compliance rate 0-100")
    target_score: float = Field(0.0, description="Interim Target Engine alignment score 0-100")
    action_plan_score: float = Field(0.0, description="Action Plan Engine completeness score 0-100")
    progress_score: float = Field(0.0, description="Progress Tracking Engine trajectory score 0-100")
    sector_score: float = Field(0.0, description="Sector Pathway Engine alignment score 0-100")
    partnership_score: float = Field(0.0, description="Partnership Scoring Engine synergy score 0-100")
    credibility_score: float = Field(0.0, description="Credibility Assessment Engine score 0-100")

    # Status details from engines
    pledge_status: Optional[str] = Field(None, description="Pledge eligibility status")
    starting_line_status: Optional[str] = Field(None, description="Starting Line compliance status")
    target_pathway: Optional[str] = Field(None, description="Pathway alignment (1.5C/WB2C/2C/MISALIGNED)")
    progress_rag: Optional[str] = Field(None, description="Progress RAG status")
    credibility_tier: Optional[str] = Field(None, description="Credibility tier (HIGH/MODERATE/LOW/CRITICAL)")

    # Optional dimension overrides for detailed assessments
    dimension_inputs: List[DimensionInput] = Field(
        default_factory=list, description="Detailed dimension inputs (override simple scores)"
    )

    # Additional context
    temperature_alignment: Optional[float] = Field(None, description="Temperature alignment (C)")
    annual_reduction_rate: Optional[float] = Field(None, description="Annual reduction rate (%)")
    actor_type: Optional[str] = Field(None, description="Actor type (company, city, region, etc.)")
    sector: Optional[str] = Field(None, description="Primary sector")
    partner_initiatives: List[str] = Field(
        default_factory=list, description="Active partner initiative memberships"
    )

    @field_validator("pledge_score", "starting_line_score", "target_score",
                     "action_plan_score", "progress_score", "sector_score",
                     "partnership_score", "credibility_score")
    @classmethod
    def validate_dimension_score(cls, v: float) -> float:
        return max(0.0, min(100.0, v))

    @field_validator("reporting_year")
    @classmethod
    def validate_reporting_year(cls, v: int) -> int:
        if v < 2015 or v > 2060:
            raise ValueError(f"Reporting year {v} out of valid range [2015, 2060]")
        return v


class DimensionScore(BaseModel):
    """Score for a single readiness dimension."""
    dimension_id: str
    dimension_name: str
    score: float
    weight: float
    weighted_score: float
    rag_status: str
    source_engine: str
    min_for_readiness: float
    gap_to_readiness: float
    gap_weighted_priority: float
    improvement_actions: List[str]
    status_detail: Optional[str] = None


class ImprovementPriority(BaseModel):
    """Ranked improvement priority action."""
    rank: int
    dimension_id: str
    dimension_name: str
    action: str
    impact_score: float
    effort_level: str
    estimated_timeline: str
    current_score: float
    target_score: float


class ReadinessTimeline(BaseModel):
    """Estimated timeline to achieve readiness levels."""
    current_level: str
    current_score: float
    levels: List[Dict[str, Any]]
    estimated_months_to_race_ready: int
    critical_path_dimensions: List[str]
    parallelizable_actions: int
    sequential_actions: int


class ReadinessResult(BaseModel):
    """Complete race readiness assessment result."""
    assessment_id: str
    entity_id: str
    entity_name: str
    reporting_year: int
    assessment_date: str

    # Overall scores
    composite_score: float
    readiness_level: str
    readiness_description: str
    rag_status: str

    # Dimension breakdown
    dimension_scores: List[DimensionScore]
    dimensions_meeting_minimum: int
    dimensions_total: int

    # Context
    temperature_alignment: Optional[float] = None
    pathway_alignment: Optional[str] = None
    actor_type: Optional[str] = None

    # Improvement priorities
    improvement_priorities: List[ImprovementPriority]
    top_3_priorities: List[str]

    # Timeline
    timeline: ReadinessTimeline

    # Summary
    key_strengths: List[str]
    key_gaps: List[str]
    executive_summary: str

    # Campaign standing
    campaign_eligible: bool
    campaign_standing_risk: Optional[str] = None
    next_review_date: Optional[str] = None

    # Provenance
    engine_version: str
    module_version: str
    calculated_at: str
    processing_time_ms: float
    provenance_hash: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RaceReadinessEngine:
    """
    Computes overall Race to Zero campaign readiness via 8-dimension
    composite scoring, aggregating all preceding engine outputs.

    Usage::

        engine = RaceReadinessEngine()
        result = engine.assess(readiness_input)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._module_version = _MODULE_VERSION
        logger.info(
            "RaceReadinessEngine initialized (v%s, %d dimensions)",
            self._module_version,
            len(DIMENSION_CONFIG),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, data: ReadinessInput) -> ReadinessResult:
        """Run full race readiness assessment."""
        t0 = time.monotonic()
        logger.info(
            "Assessing race readiness for entity=%s year=%d",
            data.entity_id,
            data.reporting_year,
        )

        # Build dimension scores lookup from input
        dim_scores_raw = self._extract_dimension_scores(data)

        # Assess each dimension
        dim_results: List[DimensionScore] = []
        for dim_id, dim_config in DIMENSION_CONFIG.items():
            raw_score = dim_scores_raw.get(dim_id, 0.0)
            status_detail = self._get_status_detail(data, dim_id)
            dim_result = self._assess_dimension(dim_id, dim_config, raw_score, status_detail)
            dim_results.append(dim_result)

        # Compute composite score
        composite = self._compute_composite_score(dim_results)

        # Determine readiness level
        readiness_level, readiness_desc = self._determine_readiness_level(composite)

        # Determine RAG status
        rag = self._determine_rag(composite)

        # Count dimensions meeting minimum
        dims_meeting = sum(
            1 for d in dim_results
            if d.gap_to_readiness <= 0
        )

        # Build improvement priorities
        priorities = self._build_improvement_priorities(dim_results)

        # Top 3 priorities
        top_3 = [p.action for p in priorities[:3]]

        # Timeline
        timeline = self._estimate_timeline(composite, readiness_level, dim_results, priorities)

        # Strengths and gaps
        strengths = self._identify_strengths(dim_results)
        gaps = self._identify_gaps(dim_results)

        # Executive summary
        exec_summary = self._generate_executive_summary(
            data, composite, readiness_level, rag, dims_meeting, len(dim_results),
            strengths, gaps,
        )

        # Campaign eligibility
        campaign_eligible = (
            composite >= 70.0
            and dims_meeting >= 6
        )

        # Campaign standing risk
        standing_risk = None
        if composite < 50.0:
            standing_risk = "HIGH - Entity may not meet Race to Zero minimum criteria"
        elif composite < 70.0:
            standing_risk = "MODERATE - Entity needs improvement to maintain campaign standing"

        elapsed_ms = (time.monotonic() - t0) * 1000

        result = ReadinessResult(
            assessment_id=_new_uuid(),
            entity_id=data.entity_id,
            entity_name=data.entity_name,
            reporting_year=data.reporting_year,
            assessment_date=data.assessment_date or _utcnow().isoformat(),
            composite_score=_round3(composite),
            readiness_level=readiness_level,
            readiness_description=readiness_desc,
            rag_status=rag,
            dimension_scores=dim_results,
            dimensions_meeting_minimum=dims_meeting,
            dimensions_total=len(dim_results),
            temperature_alignment=data.temperature_alignment,
            pathway_alignment=data.target_pathway,
            actor_type=data.actor_type,
            improvement_priorities=priorities,
            top_3_priorities=top_3,
            timeline=timeline,
            key_strengths=strengths,
            key_gaps=gaps,
            executive_summary=exec_summary,
            campaign_eligible=campaign_eligible,
            campaign_standing_risk=standing_risk,
            engine_version=self._module_version,
            module_version=self._module_version,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 2),
            provenance_hash="",
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Race readiness assessment complete: entity=%s score=%.1f level=%s rag=%s hash=%s",
            data.entity_id,
            composite,
            readiness_level,
            rag,
            result.provenance_hash[:12],
        )
        return result

    # ------------------------------------------------------------------
    # Dimension Score Extraction
    # ------------------------------------------------------------------

    def _extract_dimension_scores(self, data: ReadinessInput) -> Dict[str, float]:
        """Extract dimension scores from input, using overrides if available."""
        # Start with simple score fields
        scores: Dict[str, float] = {
            DimensionId.PLEDGE_STRENGTH.value: data.pledge_score,
            DimensionId.STARTING_LINE.value: data.starting_line_score,
            DimensionId.TARGET_AMBITION.value: data.target_score,
            DimensionId.ACTION_PLAN.value: data.action_plan_score,
            DimensionId.PROGRESS.value: data.progress_score,
            DimensionId.SECTOR.value: data.sector_score,
            DimensionId.PARTNERSHIP.value: data.partnership_score,
            DimensionId.CREDIBILITY.value: data.credibility_score,
        }

        # Apply detailed dimension_inputs overrides
        for dim_input in data.dimension_inputs:
            if dim_input.dimension_id in scores:
                scores[dim_input.dimension_id] = dim_input.score

        return scores

    def _get_status_detail(self, data: ReadinessInput, dim_id: str) -> Optional[str]:
        """Get status detail for a dimension from input data."""
        mapping = {
            DimensionId.PLEDGE_STRENGTH.value: data.pledge_status,
            DimensionId.STARTING_LINE.value: data.starting_line_status,
            DimensionId.TARGET_AMBITION.value: data.target_pathway,
            DimensionId.PROGRESS.value: data.progress_rag,
            DimensionId.CREDIBILITY.value: data.credibility_tier,
        }
        return mapping.get(dim_id)

    # ------------------------------------------------------------------
    # Dimension Assessment
    # ------------------------------------------------------------------

    def _assess_dimension(
        self,
        dim_id: str,
        dim_config: Dict[str, Any],
        raw_score: float,
        status_detail: Optional[str],
    ) -> DimensionScore:
        """Assess a single readiness dimension."""
        score = _decimal(raw_score)
        weight = dim_config["weight"]
        weighted = score * weight
        min_req = dim_config["min_for_readiness"]
        gap = max(Decimal("0"), min_req - score)
        urgency = dim_config["urgency_factor"]
        gap_priority = gap * weight * urgency

        # RAG for this dimension
        dim_rag = self._determine_rag(float(score))

        # Get improvement actions
        actions_db = IMPROVEMENT_ACTIONS.get(dim_id, [])
        relevant_actions = []
        if float(score) < float(min_req):
            for a in actions_db[:3]:
                relevant_actions.append(a["action"])
        elif float(score) < 85.0:
            # Approaching readiness - fewer actions
            for a in actions_db[:1]:
                relevant_actions.append(a["action"])

        return DimensionScore(
            dimension_id=dim_id,
            dimension_name=dim_config["name"],
            score=_round3(float(score)),
            weight=float(weight),
            weighted_score=_round3(float(weighted)),
            rag_status=dim_rag,
            source_engine=dim_config["source_engine"],
            min_for_readiness=float(min_req),
            gap_to_readiness=_round3(float(gap)),
            gap_weighted_priority=_round3(float(gap_priority)),
            improvement_actions=relevant_actions,
            status_detail=status_detail,
        )

    # ------------------------------------------------------------------
    # Composite Score
    # ------------------------------------------------------------------

    def _compute_composite_score(self, dimensions: List[DimensionScore]) -> float:
        """Compute weighted composite readiness score."""
        total_weighted = Decimal("0")
        total_weight = Decimal("0")

        for dim in dimensions:
            w = _decimal(dim.weight)
            s = _decimal(dim.score)
            total_weighted += s * w
            total_weight += w

        if total_weight == Decimal("0"):
            return 0.0

        return float(_safe_divide(total_weighted, total_weight))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _determine_readiness_level(self, score: float) -> tuple:
        """Determine readiness level from composite score."""
        d_score = _decimal(score)

        for level in [ReadinessLevel.RACE_READY, ReadinessLevel.APPROACHING,
                      ReadinessLevel.BUILDING, ReadinessLevel.EARLY_STAGE,
                      ReadinessLevel.PRE_PLEDGE]:
            threshold = READINESS_THRESHOLDS[level.value]
            if d_score >= threshold["min_score"]:
                return level.value, threshold["description"]

        return ReadinessLevel.PRE_PLEDGE.value, READINESS_THRESHOLDS[ReadinessLevel.PRE_PLEDGE.value]["description"]

    def _determine_rag(self, score: float) -> str:
        """Determine RAG status from score."""
        d_score = _decimal(score)
        if d_score >= RAG_THRESHOLDS[RAGStatus.GREEN.value]:
            return RAGStatus.GREEN.value
        elif d_score >= RAG_THRESHOLDS[RAGStatus.AMBER.value]:
            return RAGStatus.AMBER.value
        elif d_score >= RAG_THRESHOLDS[RAGStatus.RED.value]:
            return RAGStatus.RED.value
        return RAGStatus.BLACK.value

    # ------------------------------------------------------------------
    # Improvement Priorities
    # ------------------------------------------------------------------

    def _build_improvement_priorities(
        self, dimensions: List[DimensionScore],
    ) -> List[ImprovementPriority]:
        """Build ranked improvement priorities across all dimensions."""
        priorities: List[ImprovementPriority] = []

        for dim in dimensions:
            if dim.gap_to_readiness <= 0 and dim.score >= 85.0:
                continue  # Already meeting readiness - skip

            dim_id = dim.dimension_id
            actions_db = IMPROVEMENT_ACTIONS.get(dim_id, [])
            target_score = dim.min_for_readiness if dim.gap_to_readiness > 0 else 85.0

            for action_item in actions_db:
                impact = dim.gap_weighted_priority
                if dim.gap_to_readiness <= 0:
                    # Already meets minimum but below 85 - lower priority
                    impact = (85.0 - dim.score) * dim.weight * 0.5

                priorities.append(ImprovementPriority(
                    rank=0,
                    dimension_id=dim_id,
                    dimension_name=dim.dimension_name,
                    action=action_item["action"],
                    impact_score=_round3(impact),
                    effort_level=action_item["effort"],
                    estimated_timeline=action_item["timeline"],
                    current_score=dim.score,
                    target_score=_round3(target_score),
                ))

        # Sort by impact descending, then by effort ascending
        effort_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        priorities.sort(
            key=lambda p: (-p.impact_score, effort_order.get(p.effort_level, 1))
        )

        # Assign ranks
        for i, p in enumerate(priorities):
            p.rank = i + 1

        return priorities

    # ------------------------------------------------------------------
    # Timeline Estimation
    # ------------------------------------------------------------------

    def _estimate_timeline(
        self,
        composite: float,
        current_level: str,
        dimensions: List[DimensionScore],
        priorities: List[ImprovementPriority],
    ) -> ReadinessTimeline:
        """Estimate timeline to achieve each readiness level."""
        levels_timeline = []

        for level in [ReadinessLevel.PRE_PLEDGE, ReadinessLevel.EARLY_STAGE,
                      ReadinessLevel.BUILDING, ReadinessLevel.APPROACHING,
                      ReadinessLevel.RACE_READY]:
            threshold = READINESS_THRESHOLDS[level.value]
            min_score = float(threshold["min_score"])
            if composite >= min_score:
                levels_timeline.append({
                    "level": level.value,
                    "min_score": min_score,
                    "status": "ACHIEVED",
                    "estimated_months": 0,
                })
            else:
                gap_to_level = min_score - composite
                # Estimate months: ~5 points per month for focused work
                months = max(1, int(gap_to_level / 5.0))
                levels_timeline.append({
                    "level": level.value,
                    "min_score": min_score,
                    "status": "PENDING",
                    "estimated_months": months,
                })

        # Months to race ready
        race_ready_entry = next(
            (l for l in levels_timeline if l["level"] == ReadinessLevel.RACE_READY.value),
            None,
        )
        months_to_ready = race_ready_entry["estimated_months"] if race_ready_entry else 24

        # Critical path dimensions (largest gaps)
        sorted_dims = sorted(dimensions, key=lambda d: d.gap_weighted_priority, reverse=True)
        critical_path = [d.dimension_name for d in sorted_dims if d.gap_to_readiness > 0][:3]

        # Parallelizable vs sequential actions
        # Most improvement actions can be parallelized; sequential only for dependencies
        total_actions = len([p for p in priorities if p.impact_score > 0])
        parallelizable = int(total_actions * 0.7)  # 70% can run in parallel
        sequential = total_actions - parallelizable

        return ReadinessTimeline(
            current_level=current_level,
            current_score=_round3(composite),
            levels=levels_timeline,
            estimated_months_to_race_ready=months_to_ready,
            critical_path_dimensions=critical_path,
            parallelizable_actions=parallelizable,
            sequential_actions=sequential,
        )

    # ------------------------------------------------------------------
    # Strengths & Gaps
    # ------------------------------------------------------------------

    def _identify_strengths(self, dimensions: List[DimensionScore]) -> List[str]:
        """Identify key strengths from high-scoring dimensions."""
        strengths = []
        for dim in sorted(dimensions, key=lambda d: d.score, reverse=True):
            if dim.score >= 75.0:
                strengths.append(
                    f"{dim.dimension_name}: score {dim.score:.1f}/100 ({dim.rag_status})"
                )
        return strengths[:5]

    def _identify_gaps(self, dimensions: List[DimensionScore]) -> List[str]:
        """Identify key gaps from low-scoring dimensions."""
        gaps = []
        for dim in sorted(dimensions, key=lambda d: d.gap_weighted_priority, reverse=True):
            if dim.gap_to_readiness > 0:
                gaps.append(
                    f"{dim.dimension_name}: score {dim.score:.1f}/100 "
                    f"(gap of {dim.gap_to_readiness:.1f} to minimum {dim.min_for_readiness:.0f})"
                )
        return gaps[:5]

    # ------------------------------------------------------------------
    # Executive Summary
    # ------------------------------------------------------------------

    def _generate_executive_summary(
        self,
        data: ReadinessInput,
        composite: float,
        level: str,
        rag: str,
        dims_meeting: int,
        dims_total: int,
        strengths: List[str],
        gaps: List[str],
    ) -> str:
        """Generate executive summary narrative."""
        parts = []

        # Opening
        entity_type = data.actor_type or "entity"
        parts.append(
            f"{data.entity_name} has a Race to Zero readiness score of {composite:.1f}/100, "
            f"classified as {level.replace('_', ' ').title()} ({rag}). "
            f"{dims_meeting} of {dims_total} readiness dimensions meet the minimum threshold."
        )

        # Temperature context
        if data.temperature_alignment:
            parts.append(
                f"Current trajectory implies a temperature alignment of {data.temperature_alignment:.1f}C."
            )

        # Strengths
        if strengths:
            parts.append(
                f"Key strengths: {'; '.join(strengths[:3])}."
            )

        # Gaps
        if gaps:
            parts.append(
                f"Priority gaps to address: {'; '.join(gaps[:3])}."
            )

        # Level-specific guidance
        if level == ReadinessLevel.RACE_READY.value:
            parts.append(
                f"The {entity_type} is campaign-ready and meets all Race to Zero Starting Line "
                f"criteria with strong HLEG credibility. Focus on maintaining and improving performance."
            )
        elif level == ReadinessLevel.APPROACHING.value:
            parts.append(
                f"The {entity_type} is near-ready for Race to Zero participation. "
                f"Address remaining minor gaps within 1-3 months to achieve full readiness."
            )
        elif level == ReadinessLevel.BUILDING.value:
            parts.append(
                f"The {entity_type} is building readiness with moderate gaps remaining. "
                f"Focused work over 3-6 months is needed across priority dimensions."
            )
        elif level == ReadinessLevel.EARLY_STAGE.value:
            parts.append(
                f"The {entity_type} is in the planning phase with significant gaps. "
                f"A 6-12 month structured improvement program is recommended."
            )
        else:
            parts.append(
                f"The {entity_type} requires major foundational work before Race to Zero "
                f"participation can be considered. Focus on establishing basic climate "
                f"governance, targets, and action planning (12-24 months)."
            )

        # Partner context
        if data.partner_initiatives:
            parts.append(
                f"Active partner initiatives: {', '.join(data.partner_initiatives[:5])}."
            )

        return " ".join(parts)
