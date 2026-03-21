# -*- coding: utf-8 -*-
"""
NetZeroScorecardEngine - PACK-021 Net Zero Starter Engine 7
==============================================================

Net-zero readiness and maturity assessment engine evaluating
organizations across eight dimensions of net-zero preparedness.

This engine assesses an organization's net-zero maturity using a
structured, evidence-based framework.  Each of the eight dimensions
is scored 0-100 and mapped to a five-level maturity model.  The
overall score is a weighted aggregate with configurable weights.

Assessment Dimensions:
    1. GHG Inventory Completeness - Scope 1+2+3 coverage, data quality
    2. Target Ambition - SBTi alignment, 1.5C pathway, coverage
    3. Reduction Progress - Actual vs. target, trend direction
    4. Decarbonization Actions - Actions identified, costed, scheduled
    5. Governance & Strategy - Board oversight, climate integration
    6. Financial Planning - Climate CapEx, carbon pricing
    7. Offset/Neutralization - Credit quality, CDR pipeline
    8. Reporting & Disclosure - CDP, TCFD, ESRS alignment

Maturity Levels (per ISO 14097 / TCFD aligned):
    Level 1 (0-20):  Awareness - Organization recognizes the need
    Level 2 (21-40): Foundation - Basic data collection started
    Level 3 (41-60): Developing - Targets set, initial actions
    Level 4 (61-80): Advanced - Systematic reduction programme
    Level 5 (81-100): Leading - On-track net-zero with best practices

Regulatory and Framework References:
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - TCFD Recommendations (2017, updated 2021)
    - CDP Climate Change Questionnaire (2024)
    - ESRS E1 Climate Change (CSRD)
    - ISO 14064:2018 Parts 1-3
    - ISO 14097:2021 - Framework for climate-related assessments
    - GHG Protocol Corporate Standard (2015)

Zero-Hallucination:
    - All dimension scores are deterministic weighted sums of indicators
    - Maturity level assignment uses fixed threshold lookup
    - Gap analysis uses arithmetic difference calculations
    - Recommendation priority uses deterministic ranking
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-021 Net Zero Starter
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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning default on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


def _clamp(value: Decimal, lo: Decimal = Decimal("0"), hi: Decimal = Decimal("100")) -> Decimal:
    """Clamp a Decimal value between lo and hi."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MaturityLevel(str, Enum):
    """Organizational maturity level for net-zero readiness.

    Based on a five-tier model aligned with ISO 14097 and TCFD
    implementation maturity frameworks.
    """
    AWARENESS = "awareness"
    FOUNDATION = "foundation"
    DEVELOPING = "developing"
    ADVANCED = "advanced"
    LEADING = "leading"


class ScorecardDimension(str, Enum):
    """Assessment dimension for the net-zero scorecard.

    Eight dimensions covering the full scope of net-zero readiness
    from data quality to governance and disclosure.
    """
    GHG_INVENTORY = "ghg_inventory"
    TARGET_AMBITION = "target_ambition"
    REDUCTION_PROGRESS = "reduction_progress"
    DECARBONIZATION_ACTIONS = "decarbonization_actions"
    GOVERNANCE_STRATEGY = "governance_strategy"
    FINANCIAL_PLANNING = "financial_planning"
    OFFSET_NEUTRALIZATION = "offset_neutralization"
    REPORTING_DISCLOSURE = "reporting_disclosure"


class RecommendationPriority(str, Enum):
    """Priority level for scorecard recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Reference Data Constants
# ---------------------------------------------------------------------------


# Maturity level thresholds and descriptions.
MATURITY_LEVELS: Dict[str, Dict[str, Any]] = {
    MaturityLevel.AWARENESS.value: {
        "min_score": Decimal("0"),
        "max_score": Decimal("20"),
        "label": "Level 1: Awareness",
        "description": (
            "Organization recognizes the need for net-zero transition. "
            "No formal GHG inventory or climate targets in place. "
            "Limited awareness of regulatory requirements."
        ),
        "typical_characteristics": [
            "No GHG inventory or limited Scope 1 only",
            "No formal climate targets",
            "Climate not discussed at board level",
            "No dedicated climate budget",
        ],
    },
    MaturityLevel.FOUNDATION.value: {
        "min_score": Decimal("21"),
        "max_score": Decimal("40"),
        "label": "Level 2: Foundation",
        "description": (
            "Basic data collection started. Scope 1 and 2 inventory "
            "in progress. Initial climate governance being established. "
            "Exploring target-setting frameworks."
        ),
        "typical_characteristics": [
            "Scope 1 and 2 inventory initiated",
            "Exploring SBTi commitment",
            "Climate mentioned in annual report",
            "Initial climate risk assessment",
        ],
    },
    MaturityLevel.DEVELOPING.value: {
        "min_score": Decimal("41"),
        "max_score": Decimal("60"),
        "label": "Level 3: Developing",
        "description": (
            "Targets set and initial reduction actions underway. "
            "Scope 1, 2, and partial Scope 3 inventory complete. "
            "Climate strategy being integrated into business planning."
        ),
        "typical_characteristics": [
            "SBTi targets committed or approved",
            "Scope 3 screening completed",
            "Transition plan drafted",
            "Climate governance formalized",
        ],
    },
    MaturityLevel.ADVANCED.value: {
        "min_score": Decimal("61"),
        "max_score": Decimal("80"),
        "label": "Level 4: Advanced",
        "description": (
            "Systematic reduction programme in place. Comprehensive "
            "Scope 3 inventory. On-track to meet near-term targets. "
            "Climate fully integrated into corporate strategy."
        ),
        "typical_characteristics": [
            "SBTi targets validated (near-term and long-term)",
            "Comprehensive Scope 3 measurement",
            "Transition plan with CapEx allocation",
            "CDP A-list or A- score",
        ],
    },
    MaturityLevel.LEADING.value: {
        "min_score": Decimal("81"),
        "max_score": Decimal("100"),
        "label": "Level 5: Leading",
        "description": (
            "On track for net-zero with best practices across all "
            "dimensions. Comprehensive reduction programme, CDR "
            "procurement, and sector-leading disclosure."
        ),
        "typical_characteristics": [
            "On-track or ahead of SBTi pathway",
            "CDR procurement pipeline established",
            "Science-based transition plan published",
            "Industry peer leader in disclosure and action",
        ],
    },
}


# Default dimension weights (must sum to 1.0).
DEFAULT_DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    ScorecardDimension.GHG_INVENTORY.value: Decimal("0.15"),
    ScorecardDimension.TARGET_AMBITION.value: Decimal("0.15"),
    ScorecardDimension.REDUCTION_PROGRESS.value: Decimal("0.20"),
    ScorecardDimension.DECARBONIZATION_ACTIONS.value: Decimal("0.15"),
    ScorecardDimension.GOVERNANCE_STRATEGY.value: Decimal("0.10"),
    ScorecardDimension.FINANCIAL_PLANNING.value: Decimal("0.10"),
    ScorecardDimension.OFFSET_NEUTRALIZATION.value: Decimal("0.05"),
    ScorecardDimension.REPORTING_DISCLOSURE.value: Decimal("0.10"),
}


# Indicator definitions per dimension with scoring guidance.
DIMENSION_INDICATORS: Dict[str, List[Dict[str, Any]]] = {
    ScorecardDimension.GHG_INVENTORY.value: [
        {
            "id": "inv_scope1",
            "name": "Scope 1 emissions measured",
            "max_points": Decimal("15"),
            "description": "Complete Scope 1 direct emissions inventory",
        },
        {
            "id": "inv_scope2",
            "name": "Scope 2 emissions measured (location + market)",
            "max_points": Decimal("15"),
            "description": "Both location-based and market-based Scope 2",
        },
        {
            "id": "inv_scope3_screen",
            "name": "Scope 3 screening completed",
            "max_points": Decimal("10"),
            "description": "All 15 categories screened for relevance",
        },
        {
            "id": "inv_scope3_measure",
            "name": "Material Scope 3 categories measured",
            "max_points": Decimal("20"),
            "description": "Top material categories quantified",
        },
        {
            "id": "inv_data_quality",
            "name": "Data quality score",
            "max_points": Decimal("15"),
            "description": "Proportion of primary vs. estimated data",
        },
        {
            "id": "inv_verification",
            "name": "Third-party verification",
            "max_points": Decimal("15"),
            "description": "Limited or reasonable assurance obtained",
        },
        {
            "id": "inv_base_year",
            "name": "Base year established with recalculation policy",
            "max_points": Decimal("10"),
            "description": "Documented base year and restatement criteria",
        },
    ],
    ScorecardDimension.TARGET_AMBITION.value: [
        {
            "id": "tgt_near_term",
            "name": "Near-term SBTi target (2030)",
            "max_points": Decimal("25"),
            "description": "SBTi-validated near-term target in place",
        },
        {
            "id": "tgt_long_term",
            "name": "Long-term SBTi target (2050)",
            "max_points": Decimal("20"),
            "description": "SBTi-validated long-term net-zero target",
        },
        {
            "id": "tgt_pathway",
            "name": "1.5C pathway alignment",
            "max_points": Decimal("20"),
            "description": "Targets aligned with 1.5C rather than 2C",
        },
        {
            "id": "tgt_scope_coverage",
            "name": "Target covers all material scopes",
            "max_points": Decimal("15"),
            "description": "Scope 1, 2, and 3 included in targets",
        },
        {
            "id": "tgt_interim",
            "name": "Interim milestones set",
            "max_points": Decimal("10"),
            "description": "5-year milestones between now and target year",
        },
        {
            "id": "tgt_validation",
            "name": "Third-party target validation",
            "max_points": Decimal("10"),
            "description": "Target validated by SBTi or equivalent body",
        },
    ],
    ScorecardDimension.REDUCTION_PROGRESS.value: [
        {
            "id": "red_actual_vs_target",
            "name": "Actual reduction vs. linear pathway",
            "max_points": Decimal("30"),
            "description": "On-track or ahead of linear trajectory",
        },
        {
            "id": "red_annual_rate",
            "name": "Annual reduction rate achieved",
            "max_points": Decimal("25"),
            "description": "Actual annual reduction rate vs. required",
        },
        {
            "id": "red_scope3_progress",
            "name": "Scope 3 reduction progress",
            "max_points": Decimal("20"),
            "description": "Measurable Scope 3 reductions achieved",
        },
        {
            "id": "red_trend",
            "name": "Emission trend direction",
            "max_points": Decimal("15"),
            "description": "3-year trend: declining, stable, or increasing",
        },
        {
            "id": "red_intensity",
            "name": "Intensity metric improvement",
            "max_points": Decimal("10"),
            "description": "Revenue or production intensity declining",
        },
    ],
    ScorecardDimension.DECARBONIZATION_ACTIONS.value: [
        {
            "id": "act_identified",
            "name": "Decarbonization levers identified",
            "max_points": Decimal("15"),
            "description": "Comprehensive list of reduction opportunities",
        },
        {
            "id": "act_costed",
            "name": "Actions costed with business cases",
            "max_points": Decimal("20"),
            "description": "Marginal abatement cost analysis completed",
        },
        {
            "id": "act_scheduled",
            "name": "Implementation timeline defined",
            "max_points": Decimal("20"),
            "description": "Actions scheduled with milestones",
        },
        {
            "id": "act_implemented",
            "name": "Actions in implementation",
            "max_points": Decimal("25"),
            "description": "Active projects delivering reductions",
        },
        {
            "id": "act_supplier_engagement",
            "name": "Supplier engagement programme",
            "max_points": Decimal("20"),
            "description": "Active supplier decarbonization programme",
        },
    ],
    ScorecardDimension.GOVERNANCE_STRATEGY.value: [
        {
            "id": "gov_board_oversight",
            "name": "Board-level climate oversight",
            "max_points": Decimal("25"),
            "description": "Board committee or member with climate mandate",
        },
        {
            "id": "gov_exec_kpi",
            "name": "Executive KPIs linked to climate",
            "max_points": Decimal("20"),
            "description": "Climate metrics in executive compensation",
        },
        {
            "id": "gov_transition_plan",
            "name": "Published transition plan",
            "max_points": Decimal("25"),
            "description": "Detailed, time-bound transition plan published",
        },
        {
            "id": "gov_risk_integration",
            "name": "Climate risk in enterprise risk management",
            "max_points": Decimal("15"),
            "description": "Climate risks integrated into ERM framework",
        },
        {
            "id": "gov_stakeholder",
            "name": "Stakeholder engagement on climate",
            "max_points": Decimal("15"),
            "description": "Active engagement with investors and stakeholders",
        },
    ],
    ScorecardDimension.FINANCIAL_PLANNING.value: [
        {
            "id": "fin_capex",
            "name": "Climate CapEx allocation",
            "max_points": Decimal("25"),
            "description": "Dedicated budget for decarbonization projects",
        },
        {
            "id": "fin_carbon_price",
            "name": "Internal carbon price set",
            "max_points": Decimal("20"),
            "description": "Internal carbon price applied to investment decisions",
        },
        {
            "id": "fin_scenario_analysis",
            "name": "Climate scenario analysis completed",
            "max_points": Decimal("20"),
            "description": "TCFD-aligned scenario analysis with financial impacts",
        },
        {
            "id": "fin_green_revenue",
            "name": "Green revenue tracking",
            "max_points": Decimal("15"),
            "description": "Revenue from climate solutions or aligned activities",
        },
        {
            "id": "fin_stranded_assets",
            "name": "Stranded asset risk assessed",
            "max_points": Decimal("20"),
            "description": "Fossil fuel or high-carbon asset impairment evaluated",
        },
    ],
    ScorecardDimension.OFFSET_NEUTRALIZATION.value: [
        {
            "id": "off_credit_quality",
            "name": "Carbon credit quality",
            "max_points": Decimal("25"),
            "description": "Average quality score of credit portfolio",
        },
        {
            "id": "off_cdr_pipeline",
            "name": "CDR procurement pipeline",
            "max_points": Decimal("25"),
            "description": "Advance purchase agreements for carbon removals",
        },
        {
            "id": "off_sbti_distinction",
            "name": "BVCM vs. neutralization distinction",
            "max_points": Decimal("20"),
            "description": "Clear classification of credit use per SBTi",
        },
        {
            "id": "off_oxford_shift",
            "name": "Oxford Principles portfolio shift",
            "max_points": Decimal("15"),
            "description": "Increasing removal share over time",
        },
        {
            "id": "off_registry",
            "name": "Credit retirement and registry tracking",
            "max_points": Decimal("15"),
            "description": "Credits registered and retired in public registries",
        },
    ],
    ScorecardDimension.REPORTING_DISCLOSURE.value: [
        {
            "id": "rep_cdp",
            "name": "CDP disclosure",
            "max_points": Decimal("20"),
            "description": "Annual CDP Climate Change response submitted",
        },
        {
            "id": "rep_tcfd",
            "name": "TCFD-aligned reporting",
            "max_points": Decimal("20"),
            "description": "Full TCFD disclosure across four pillars",
        },
        {
            "id": "rep_esrs",
            "name": "ESRS E1 compliance",
            "max_points": Decimal("20"),
            "description": "ESRS E1 Climate Change disclosure complete",
        },
        {
            "id": "rep_assurance",
            "name": "Third-party assurance on climate data",
            "max_points": Decimal("20"),
            "description": "Limited or reasonable assurance obtained",
        },
        {
            "id": "rep_digital",
            "name": "Digital/XBRL tagging readiness",
            "max_points": Decimal("10"),
            "description": "Climate data tagged for digital reporting",
        },
        {
            "id": "rep_transparency",
            "name": "Public climate commitments and tracking",
            "max_points": Decimal("10"),
            "description": "Public dashboard or annual progress report",
        },
    ],
}


# Benchmarking context: what "good" looks like at each level.
BENCHMARK_CONTEXT: Dict[str, Dict[str, str]] = {
    ScorecardDimension.GHG_INVENTORY.value: {
        "leading": "Full Scope 1+2+3 with primary data, third-party verified",
        "advanced": "All material scopes measured, limited assurance",
        "developing": "Scope 1+2 complete, Scope 3 screening done",
        "foundation": "Scope 1+2 in progress, no Scope 3",
        "awareness": "No formal inventory",
    },
    ScorecardDimension.TARGET_AMBITION.value: {
        "leading": "SBTi-validated near-term + long-term net-zero, 1.5C aligned",
        "advanced": "SBTi near-term validated, long-term committed",
        "developing": "SBTi commitment letter submitted",
        "foundation": "Internal targets set, not yet SBTi-aligned",
        "awareness": "No climate targets",
    },
    ScorecardDimension.REDUCTION_PROGRESS.value: {
        "leading": "Ahead of linear pathway, >5% annual reduction",
        "advanced": "On-track with 4%+ annual reduction",
        "developing": "Some reduction achieved, below required rate",
        "foundation": "Emissions stable or slight increase",
        "awareness": "No measurement of progress",
    },
    ScorecardDimension.DECARBONIZATION_ACTIONS.value: {
        "leading": "Full MACC implemented, supplier programme at scale",
        "advanced": "Key actions in implementation, supplier engagement active",
        "developing": "Actions identified and costed, early implementation",
        "foundation": "Initial assessment of reduction opportunities",
        "awareness": "No decarbonization actions identified",
    },
    ScorecardDimension.GOVERNANCE_STRATEGY.value: {
        "leading": "Board oversight, exec KPIs, published transition plan",
        "advanced": "Board committee, climate in strategy",
        "developing": "Governance formalizing, transition plan in draft",
        "foundation": "Initial climate governance steps",
        "awareness": "No formal climate governance",
    },
    ScorecardDimension.FINANCIAL_PLANNING.value: {
        "leading": "Dedicated CapEx, ICP applied, full scenario analysis",
        "advanced": "Climate CapEx allocated, ICP piloted",
        "developing": "Initial climate budget, scenario analysis started",
        "foundation": "Exploring financial implications",
        "awareness": "No climate financial planning",
    },
    ScorecardDimension.OFFSET_NEUTRALIZATION.value: {
        "leading": "CDR pipeline secured, high-quality removal portfolio",
        "advanced": "CDR procurement started, quality credits",
        "developing": "Offset strategy defined, quality assessment done",
        "foundation": "Exploring offset options",
        "awareness": "No offset or neutralization strategy",
    },
    ScorecardDimension.REPORTING_DISCLOSURE.value: {
        "leading": "CDP A-list, full TCFD+ESRS, assured, XBRL-ready",
        "advanced": "CDP A/A-, comprehensive TCFD",
        "developing": "CDP B+, partial TCFD alignment",
        "foundation": "Initial CDP response, basic reporting",
        "awareness": "No climate disclosure",
    },
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class DimensionInput(BaseModel):
    """Input scores for a single scorecard dimension.

    Provides indicator-level scores that are aggregated into
    a dimension score.
    """
    dimension: ScorecardDimension = Field(
        ..., description="Scorecard dimension"
    )
    indicator_scores: Dict[str, Decimal] = Field(
        ...,
        description="Indicator ID -> score (0 to max_points for that indicator)",
    )
    notes: str = Field(
        default="", description="Assessment notes", max_length=2000
    )


class ScorecardInput(BaseModel):
    """Input data for net-zero scorecard assessment.

    Contains dimension-level inputs and metadata for the
    assessment.
    """
    entity_name: str = Field(
        default="",
        description="Organization name",
        max_length=300,
    )
    sector: str = Field(
        default="",
        description="Sector classification",
        max_length=100,
    )
    assessment_year: int = Field(
        default=2026,
        description="Assessment year",
        ge=2020,
        le=2100,
    )
    dimensions: List[DimensionInput] = Field(
        ...,
        description="Dimension-level input scores",
    )
    custom_weights: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Custom dimension weights (must sum to 1.0)",
    )


class DimensionScore(BaseModel):
    """Score result for a single scorecard dimension."""
    dimension: ScorecardDimension = Field(
        ..., description="Dimension"
    )
    dimension_label: str = Field(
        default="", description="Human-readable label"
    )
    raw_score: Decimal = Field(
        default=Decimal("0"), description="Raw score (0-100)"
    )
    weighted_score: Decimal = Field(
        default=Decimal("0"), description="Weighted contribution to overall"
    )
    weight: Decimal = Field(
        default=Decimal("0"), description="Dimension weight"
    )
    maturity_level: MaturityLevel = Field(
        default=MaturityLevel.AWARENESS, description="Maturity level"
    )
    maturity_label: str = Field(
        default="", description="Maturity level label"
    )
    indicator_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Per-indicator results"
    )
    benchmark_context: str = Field(
        default="", description="What good looks like at this level"
    )
    gap_to_next_level: Decimal = Field(
        default=Decimal("0"), description="Points needed for next level"
    )
    notes: str = Field(
        default="", description="Assessment notes"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash"
    )


class ScorecardRecommendation(BaseModel):
    """A prioritized recommendation from the scorecard assessment."""
    recommendation_id: str = Field(
        default_factory=_new_uuid, description="Recommendation ID"
    )
    priority: RecommendationPriority = Field(
        ..., description="Priority level"
    )
    dimension: ScorecardDimension = Field(
        ..., description="Related dimension"
    )
    title: str = Field(
        ..., description="Short recommendation title", max_length=200
    )
    description: str = Field(
        default="", description="Detailed description", max_length=2000
    )
    impact_score_points: Decimal = Field(
        default=Decimal("0"),
        description="Estimated score improvement if implemented",
    )
    effort_level: str = Field(
        default="medium",
        description="Implementation effort (low/medium/high)",
    )
    timeframe: str = Field(
        default="",
        description="Expected timeframe (e.g., '3-6 months')",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash"
    )


class ScorecardResult(BaseModel):
    """Complete net-zero scorecard result.

    Contains dimension scores, overall score, maturity level,
    gap analysis, and prioritized recommendations.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Unique result ID"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    entity_name: str = Field(
        default="", description="Organization name"
    )
    sector: str = Field(
        default="", description="Sector"
    )
    assessment_year: int = Field(
        default=0, description="Assessment year"
    )
    dimension_scores: List[DimensionScore] = Field(
        default_factory=list, description="Per-dimension scores"
    )
    overall_score: Decimal = Field(
        default=Decimal("0"), description="Overall weighted score (0-100)"
    )
    maturity_level: MaturityLevel = Field(
        default=MaturityLevel.AWARENESS, description="Overall maturity"
    )
    maturity_label: str = Field(
        default="", description="Maturity level label"
    )
    maturity_description: str = Field(
        default="", description="Maturity level description"
    )
    strongest_dimensions: List[str] = Field(
        default_factory=list, description="Top scoring dimensions"
    )
    weakest_dimensions: List[str] = Field(
        default_factory=list, description="Lowest scoring dimensions"
    )
    recommendations: List[ScorecardRecommendation] = Field(
        default_factory=list, description="Prioritized recommendations"
    )
    action_priorities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Actions grouped by priority level",
    )
    gap_summary: Dict[str, str] = Field(
        default_factory=dict,
        description="Gap summary per dimension",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class NetZeroScorecardEngine:
    """Net-zero readiness and maturity assessment engine.

    Provides deterministic, zero-hallucination scorecard assessment:
    - Eight-dimension maturity scoring (0-100 each)
    - Overall weighted score with configurable weights
    - Five-level maturity model assignment
    - Gap analysis with points-to-next-level
    - Prioritized recommendations with impact estimates
    - Benchmarking context for each dimension and level

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Usage::

        engine = NetZeroScorecardEngine()
        result = engine.assess(ScorecardInput(
            entity_name="Acme Corp",
            sector="manufacturing",
            dimensions=[
                DimensionInput(
                    dimension=ScorecardDimension.GHG_INVENTORY,
                    indicator_scores={
                        "inv_scope1": Decimal("15"),
                        "inv_scope2": Decimal("15"),
                        "inv_scope3_screen": Decimal("10"),
                        ...
                    },
                ),
                ...
            ],
        ))
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialize NetZeroScorecardEngine."""
        logger.info(
            "NetZeroScorecardEngine v%s initialized", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Main Assessment                                                      #
    # ------------------------------------------------------------------ #

    def assess(self, input_data: ScorecardInput) -> ScorecardResult:
        """Run the complete net-zero scorecard assessment.

        Scores each dimension, computes the overall weighted score,
        assigns maturity levels, and generates recommendations.

        Args:
            input_data: Validated ScorecardInput.

        Returns:
            ScorecardResult with complete assessment.
        """
        t0 = time.perf_counter()

        logger.info(
            "Assessing net-zero scorecard for '%s' (sector=%s, year=%d)",
            input_data.entity_name, input_data.sector,
            input_data.assessment_year,
        )

        # Resolve weights
        weights = self._resolve_weights(input_data.custom_weights)

        # Step 1: Score each dimension
        dimension_scores = self._score_dimensions(
            input_data.dimensions, weights
        )

        # Step 2: Calculate overall score
        overall = self._calculate_overall_score(dimension_scores)

        # Step 3: Determine maturity level
        maturity = self._determine_maturity(overall)
        maturity_info = MATURITY_LEVELS[maturity.value]

        # Step 4: Identify strongest and weakest dimensions
        sorted_dims = sorted(
            dimension_scores, key=lambda d: d.raw_score, reverse=True
        )
        strongest = [d.dimension.value for d in sorted_dims[:2]]
        weakest = [d.dimension.value for d in sorted_dims[-2:]]

        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores)

        # Step 6: Build action priorities
        action_priorities = self._build_action_priorities(recommendations)

        # Step 7: Build gap summary
        gap_summary = self._build_gap_summary(dimension_scores)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ScorecardResult(
            entity_name=input_data.entity_name,
            sector=input_data.sector,
            assessment_year=input_data.assessment_year,
            dimension_scores=dimension_scores,
            overall_score=overall,
            maturity_level=maturity,
            maturity_label=maturity_info["label"],
            maturity_description=maturity_info["description"],
            strongest_dimensions=strongest,
            weakest_dimensions=weakest,
            recommendations=recommendations,
            action_priorities=action_priorities,
            gap_summary=gap_summary,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Scorecard complete: overall=%.1f, maturity=%s, "
            "%d recommendations in %.3f ms",
            float(overall), maturity.value,
            len(recommendations), elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------ #
    # Dimension Scoring                                                    #
    # ------------------------------------------------------------------ #

    def _score_dimensions(
        self,
        dimension_inputs: List[DimensionInput],
        weights: Dict[str, Decimal],
    ) -> List[DimensionScore]:
        """Score all provided dimensions.

        For each dimension, sums the indicator scores, normalizes
        to 0-100, and assigns a maturity level.

        Args:
            dimension_inputs: List of dimension inputs.
            weights: Dimension weight map.

        Returns:
            List of DimensionScore results.
        """
        results: List[DimensionScore] = []
        input_map = {d.dimension.value: d for d in dimension_inputs}

        for dim_key in ScorecardDimension:
            dim_input = input_map.get(dim_key.value)
            weight = weights.get(dim_key.value, Decimal("0"))

            if dim_input is not None:
                raw_score, indicator_results = self._score_single_dimension(
                    dim_key, dim_input.indicator_scores
                )
                notes = dim_input.notes
            else:
                raw_score = Decimal("0")
                indicator_results = {}
                notes = "Dimension not provided in input"

            weighted = _round_val(raw_score * weight, 2)
            maturity = self._determine_maturity(raw_score)
            maturity_info = MATURITY_LEVELS[maturity.value]

            # Gap to next level
            gap = self._calculate_gap_to_next(raw_score, maturity)

            # Benchmark context for current level
            bench = BENCHMARK_CONTEXT.get(dim_key.value, {})
            context = bench.get(maturity.value, "")

            # Human-readable label
            dim_labels = {
                ScorecardDimension.GHG_INVENTORY.value: "GHG Inventory Completeness",
                ScorecardDimension.TARGET_AMBITION.value: "Target Ambition",
                ScorecardDimension.REDUCTION_PROGRESS.value: "Reduction Progress",
                ScorecardDimension.DECARBONIZATION_ACTIONS.value: "Decarbonization Actions",
                ScorecardDimension.GOVERNANCE_STRATEGY.value: "Governance & Strategy",
                ScorecardDimension.FINANCIAL_PLANNING.value: "Financial Planning",
                ScorecardDimension.OFFSET_NEUTRALIZATION.value: "Offset/Neutralization",
                ScorecardDimension.REPORTING_DISCLOSURE.value: "Reporting & Disclosure",
            }

            ds = DimensionScore(
                dimension=dim_key,
                dimension_label=dim_labels.get(dim_key.value, dim_key.value),
                raw_score=_round_val(raw_score, 1),
                weighted_score=weighted,
                weight=weight,
                maturity_level=maturity,
                maturity_label=maturity_info["label"],
                indicator_results=indicator_results,
                benchmark_context=context,
                gap_to_next_level=gap,
                notes=notes,
            )
            ds.provenance_hash = _compute_hash(ds)
            results.append(ds)

        return results

    def _score_single_dimension(
        self,
        dimension: ScorecardDimension,
        indicator_scores: Dict[str, Decimal],
    ) -> tuple[Decimal, Dict[str, Dict[str, Any]]]:
        """Score a single dimension from indicator scores.

        Sums the indicator scores and normalizes to 0-100 based
        on the maximum possible points for that dimension.

        Args:
            dimension: The dimension being scored.
            indicator_scores: Indicator ID -> achieved score.

        Returns:
            Tuple of (normalized_score, indicator_results).
        """
        indicators = DIMENSION_INDICATORS.get(dimension.value, [])

        max_total = Decimal("0")
        achieved_total = Decimal("0")
        indicator_results: Dict[str, Dict[str, Any]] = {}

        for ind in indicators:
            ind_id = ind["id"]
            max_pts = ind["max_points"]
            max_total += max_pts

            achieved = indicator_scores.get(ind_id, Decimal("0"))
            # Clamp to 0..max_pts
            achieved = _clamp(achieved, Decimal("0"), max_pts)
            achieved_total += achieved

            indicator_results[ind_id] = {
                "name": ind["name"],
                "max_points": str(max_pts),
                "achieved_points": str(_round_val(achieved, 1)),
                "score_pct": str(_round_val(
                    _safe_divide(achieved, max_pts) * Decimal("100"), 1
                )),
                "description": ind["description"],
            }

        # Normalize to 0-100
        normalized = _safe_divide(achieved_total, max_total) * Decimal("100")
        normalized = _clamp(normalized)

        return normalized, indicator_results

    # ------------------------------------------------------------------ #
    # Overall Score                                                        #
    # ------------------------------------------------------------------ #

    def _calculate_overall_score(
        self, dimension_scores: List[DimensionScore]
    ) -> Decimal:
        """Calculate the weighted overall score.

        Args:
            dimension_scores: Per-dimension scores.

        Returns:
            Overall score (0-100).
        """
        total = sum(ds.weighted_score for ds in dimension_scores)
        return _round_val(_clamp(total), 1)

    # ------------------------------------------------------------------ #
    # Maturity Level                                                       #
    # ------------------------------------------------------------------ #

    def _determine_maturity(self, score: Decimal) -> MaturityLevel:
        """Determine maturity level from a score.

        Args:
            score: Score (0-100).

        Returns:
            MaturityLevel enum value.
        """
        if score >= Decimal("81"):
            return MaturityLevel.LEADING
        elif score >= Decimal("61"):
            return MaturityLevel.ADVANCED
        elif score >= Decimal("41"):
            return MaturityLevel.DEVELOPING
        elif score >= Decimal("21"):
            return MaturityLevel.FOUNDATION
        else:
            return MaturityLevel.AWARENESS

    def _calculate_gap_to_next(
        self, score: Decimal, current_level: MaturityLevel
    ) -> Decimal:
        """Calculate points needed to reach the next maturity level.

        Args:
            score: Current score.
            current_level: Current maturity level.

        Returns:
            Points needed for next level, or 0 if at Leading.
        """
        level_thresholds = {
            MaturityLevel.AWARENESS: Decimal("21"),
            MaturityLevel.FOUNDATION: Decimal("41"),
            MaturityLevel.DEVELOPING: Decimal("61"),
            MaturityLevel.ADVANCED: Decimal("81"),
            MaturityLevel.LEADING: Decimal("101"),  # Already at top
        }

        next_threshold = level_thresholds.get(current_level, Decimal("101"))
        gap = next_threshold - score

        if gap < Decimal("0"):
            return Decimal("0")
        if current_level == MaturityLevel.LEADING:
            return Decimal("0")
        return _round_val(gap, 1)

    # ------------------------------------------------------------------ #
    # Weights                                                              #
    # ------------------------------------------------------------------ #

    def _resolve_weights(
        self, custom_weights: Optional[Dict[str, Decimal]]
    ) -> Dict[str, Decimal]:
        """Resolve dimension weights, using defaults or custom.

        If custom weights are provided, validates they sum to 1.0
        (within tolerance).  Otherwise uses DEFAULT_DIMENSION_WEIGHTS.

        Args:
            custom_weights: Optional custom weight map.

        Returns:
            Resolved weight map.
        """
        if custom_weights is None:
            return dict(DEFAULT_DIMENSION_WEIGHTS)

        total = sum(custom_weights.values())
        tolerance = Decimal("0.01")

        if abs(total - Decimal("1")) > tolerance:
            logger.warning(
                "Custom weights sum to %s (expected 1.0), using defaults",
                total,
            )
            return dict(DEFAULT_DIMENSION_WEIGHTS)

        return custom_weights

    # ------------------------------------------------------------------ #
    # Recommendations                                                      #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self, dimension_scores: List[DimensionScore]
    ) -> List[ScorecardRecommendation]:
        """Generate prioritized recommendations based on dimension scores.

        For each dimension below the Leading level, generates one
        or more recommendations focused on the highest-impact
        improvements.

        Args:
            dimension_scores: Per-dimension scores.

        Returns:
            List of ScorecardRecommendation sorted by priority.
        """
        recs: List[ScorecardRecommendation] = []

        for ds in dimension_scores:
            if ds.maturity_level == MaturityLevel.LEADING:
                continue

            dim_recs = self._recommendations_for_dimension(ds)
            recs.extend(dim_recs)

        # Sort by priority (critical first), then by impact
        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3,
        }
        recs.sort(
            key=lambda r: (
                priority_order.get(r.priority, 99),
                -float(r.impact_score_points),
            )
        )

        return recs

    def _recommendations_for_dimension(
        self, ds: DimensionScore
    ) -> List[ScorecardRecommendation]:
        """Generate recommendations for a single dimension.

        Args:
            ds: DimensionScore for the dimension.

        Returns:
            List of recommendations.
        """
        recs: List[ScorecardRecommendation] = []
        dim = ds.dimension

        # Determine priority based on current level
        if ds.raw_score < Decimal("21"):
            priority = RecommendationPriority.CRITICAL
        elif ds.raw_score < Decimal("41"):
            priority = RecommendationPriority.HIGH
        elif ds.raw_score < Decimal("61"):
            priority = RecommendationPriority.MEDIUM
        else:
            priority = RecommendationPriority.LOW

        # Find lowest-scoring indicators
        lowest_indicators: List[tuple[str, Decimal]] = []
        for ind_id, ind_data in ds.indicator_results.items():
            score_pct = _decimal(ind_data["score_pct"])
            lowest_indicators.append((ind_id, score_pct))

        lowest_indicators.sort(key=lambda x: x[1])

        # Generate recommendations for bottom indicators
        for ind_id, score_pct in lowest_indicators[:3]:
            ind_data = ds.indicator_results.get(ind_id, {})
            ind_name = ind_data.get("name", ind_id)
            max_pts = _decimal(ind_data.get("max_points", "0"))
            achieved = _decimal(ind_data.get("achieved_points", "0"))
            potential_gain = max_pts - achieved

            # Estimate impact on overall score (rough)
            weight = ds.weight
            indicator_count = max(len(ds.indicator_results), 1)
            impact = _round_val(
                potential_gain / _decimal(indicator_count) * weight * Decimal("100") / Decimal("100"),
                1,
            )

            # Build recommendation text
            title, description, effort, timeframe = self._build_rec_text(
                dim, ind_id, ind_name, score_pct
            )

            rec = ScorecardRecommendation(
                priority=priority,
                dimension=dim,
                title=title,
                description=description,
                impact_score_points=impact,
                effort_level=effort,
                timeframe=timeframe,
            )
            rec.provenance_hash = _compute_hash(rec)
            recs.append(rec)

        return recs

    def _build_rec_text(
        self,
        dim: ScorecardDimension,
        ind_id: str,
        ind_name: str,
        score_pct: Decimal,
    ) -> tuple[str, str, str, str]:
        """Build recommendation title, description, effort, and timeframe.

        Uses deterministic mapping from indicator ID and score to
        recommendation text.

        Args:
            dim: Scorecard dimension.
            ind_id: Indicator ID.
            ind_name: Indicator name.
            score_pct: Current indicator score (%).

        Returns:
            Tuple of (title, description, effort, timeframe).
        """
        # Recommendation templates by dimension and indicator
        rec_map: Dict[str, Dict[str, tuple[str, str, str]]] = {
            ScorecardDimension.GHG_INVENTORY.value: {
                "inv_scope1": (
                    "Complete Scope 1 direct emissions inventory",
                    "medium", "3-6 months",
                ),
                "inv_scope2": (
                    "Implement dual reporting (location + market-based) for Scope 2",
                    "medium", "3-6 months",
                ),
                "inv_scope3_screen": (
                    "Conduct full Scope 3 category screening across all 15 categories",
                    "medium", "2-4 months",
                ),
                "inv_scope3_measure": (
                    "Quantify material Scope 3 categories with primary supplier data",
                    "high", "6-12 months",
                ),
                "inv_data_quality": (
                    "Improve data quality by replacing estimates with primary data",
                    "high", "6-12 months",
                ),
                "inv_verification": (
                    "Obtain third-party verification (limited assurance) for GHG inventory",
                    "medium", "3-6 months",
                ),
                "inv_base_year": (
                    "Establish formal base year with documented recalculation policy",
                    "low", "1-2 months",
                ),
            },
            ScorecardDimension.TARGET_AMBITION.value: {
                "tgt_near_term": (
                    "Set and submit a near-term SBTi target aligned with 1.5C",
                    "medium", "3-6 months",
                ),
                "tgt_long_term": (
                    "Commit to a long-term SBTi net-zero target",
                    "medium", "3-6 months",
                ),
                "tgt_pathway": (
                    "Upgrade target pathway from well-below 2C to 1.5C alignment",
                    "medium", "3-6 months",
                ),
                "tgt_scope_coverage": (
                    "Extend targets to cover all material Scope 3 categories",
                    "medium", "3-6 months",
                ),
                "tgt_interim": (
                    "Define 5-year interim milestones between now and target year",
                    "low", "1-2 months",
                ),
                "tgt_validation": (
                    "Submit targets for SBTi validation",
                    "medium", "6-12 months",
                ),
            },
            ScorecardDimension.REDUCTION_PROGRESS.value: {
                "red_actual_vs_target": (
                    "Accelerate reduction actions to close gap to linear pathway",
                    "high", "12-24 months",
                ),
                "red_annual_rate": (
                    "Increase annual reduction rate through priority decarbonization actions",
                    "high", "12-24 months",
                ),
                "red_scope3_progress": (
                    "Launch Scope 3 reduction programme targeting key categories",
                    "high", "12-24 months",
                ),
                "red_trend": (
                    "Reverse emission increase trend through operational efficiency",
                    "medium", "6-12 months",
                ),
                "red_intensity": (
                    "Track and reduce carbon intensity metrics alongside absolute targets",
                    "low", "3-6 months",
                ),
            },
            ScorecardDimension.DECARBONIZATION_ACTIONS.value: {
                "act_identified": (
                    "Map all decarbonization levers across value chain",
                    "medium", "3-6 months",
                ),
                "act_costed": (
                    "Develop marginal abatement cost curve (MACC) for all levers",
                    "medium", "3-6 months",
                ),
                "act_scheduled": (
                    "Create implementation roadmap with timelines and milestones",
                    "medium", "2-4 months",
                ),
                "act_implemented": (
                    "Accelerate implementation of priority decarbonization projects",
                    "high", "6-24 months",
                ),
                "act_supplier_engagement": (
                    "Launch supplier decarbonization engagement programme",
                    "high", "6-12 months",
                ),
            },
            ScorecardDimension.GOVERNANCE_STRATEGY.value: {
                "gov_board_oversight": (
                    "Establish board-level climate oversight committee",
                    "medium", "3-6 months",
                ),
                "gov_exec_kpi": (
                    "Link executive compensation to climate KPIs",
                    "medium", "6-12 months",
                ),
                "gov_transition_plan": (
                    "Develop and publish a credible net-zero transition plan",
                    "high", "6-12 months",
                ),
                "gov_risk_integration": (
                    "Integrate climate risk into enterprise risk management framework",
                    "medium", "3-6 months",
                ),
                "gov_stakeholder": (
                    "Enhance investor and stakeholder engagement on climate strategy",
                    "low", "3-6 months",
                ),
            },
            ScorecardDimension.FINANCIAL_PLANNING.value: {
                "fin_capex": (
                    "Allocate dedicated climate CapEx in annual budgeting",
                    "medium", "3-6 months",
                ),
                "fin_carbon_price": (
                    "Implement internal carbon pricing for investment decisions",
                    "medium", "3-6 months",
                ),
                "fin_scenario_analysis": (
                    "Conduct TCFD-aligned climate scenario analysis",
                    "high", "6-12 months",
                ),
                "fin_green_revenue": (
                    "Track and report green revenue streams",
                    "low", "3-6 months",
                ),
                "fin_stranded_assets": (
                    "Assess stranded asset risk for high-carbon investments",
                    "medium", "3-6 months",
                ),
            },
            ScorecardDimension.OFFSET_NEUTRALIZATION.value: {
                "off_credit_quality": (
                    "Improve carbon credit portfolio quality to 70+ average",
                    "medium", "3-6 months",
                ),
                "off_cdr_pipeline": (
                    "Establish CDR advance purchase agreements",
                    "high", "6-12 months",
                ),
                "off_sbti_distinction": (
                    "Classify credits as BVCM or neutralization per SBTi guidance",
                    "low", "1-2 months",
                ),
                "off_oxford_shift": (
                    "Shift portfolio towards removal credits per Oxford Principles",
                    "medium", "6-12 months",
                ),
                "off_registry": (
                    "Ensure all credits are retired in public registries",
                    "low", "1-3 months",
                ),
            },
            ScorecardDimension.REPORTING_DISCLOSURE.value: {
                "rep_cdp": (
                    "Submit annual CDP Climate Change response",
                    "medium", "3-6 months",
                ),
                "rep_tcfd": (
                    "Achieve full TCFD-aligned disclosure across four pillars",
                    "medium", "6-12 months",
                ),
                "rep_esrs": (
                    "Prepare ESRS E1 Climate Change disclosure for CSRD compliance",
                    "high", "6-12 months",
                ),
                "rep_assurance": (
                    "Obtain third-party assurance on climate disclosures",
                    "medium", "3-6 months",
                ),
                "rep_digital": (
                    "Prepare climate data for XBRL/digital tagging",
                    "medium", "3-6 months",
                ),
                "rep_transparency": (
                    "Publish public climate progress dashboard or report",
                    "low", "2-4 months",
                ),
            },
        }

        dim_recs = rec_map.get(dim.value, {})
        rec_data = dim_recs.get(ind_id)

        if rec_data:
            title = f"Improve {ind_name}"
            description = rec_data[0]
            effort = rec_data[1]
            timeframe = rec_data[2]
        else:
            title = f"Improve: {ind_name}"
            description = f"Address gaps in '{ind_name}' (current: {score_pct}%)."
            effort = "medium"
            timeframe = "3-6 months"

        return title, description, effort, timeframe

    # ------------------------------------------------------------------ #
    # Action Priorities                                                    #
    # ------------------------------------------------------------------ #

    def _build_action_priorities(
        self, recommendations: List[ScorecardRecommendation]
    ) -> Dict[str, List[str]]:
        """Group recommendation titles by priority level.

        Args:
            recommendations: List of recommendations.

        Returns:
            Dict mapping priority level to list of action titles.
        """
        priorities: Dict[str, List[str]] = {
            RecommendationPriority.CRITICAL.value: [],
            RecommendationPriority.HIGH.value: [],
            RecommendationPriority.MEDIUM.value: [],
            RecommendationPriority.LOW.value: [],
        }

        for rec in recommendations:
            priorities[rec.priority.value].append(rec.title)

        return priorities

    # ------------------------------------------------------------------ #
    # Gap Summary                                                          #
    # ------------------------------------------------------------------ #

    def _build_gap_summary(
        self, dimension_scores: List[DimensionScore]
    ) -> Dict[str, str]:
        """Build a gap summary for each dimension.

        Args:
            dimension_scores: Per-dimension scores.

        Returns:
            Dict mapping dimension to gap description.
        """
        summary: Dict[str, str] = {}

        for ds in dimension_scores:
            if ds.maturity_level == MaturityLevel.LEADING:
                summary[ds.dimension.value] = (
                    f"{ds.dimension_label}: Leading ({ds.raw_score}/100). "
                    f"Maintain current performance."
                )
            else:
                summary[ds.dimension.value] = (
                    f"{ds.dimension_label}: {ds.maturity_label} ({ds.raw_score}/100). "
                    f"Gap to next level: {ds.gap_to_next_level} points."
                )

        return summary

    # ------------------------------------------------------------------ #
    # Convenience Methods                                                  #
    # ------------------------------------------------------------------ #

    def get_maturity_info(self, level: MaturityLevel) -> Dict[str, Any]:
        """Get detailed information about a maturity level.

        Args:
            level: MaturityLevel to look up.

        Returns:
            Dict with level details.
        """
        info = MATURITY_LEVELS.get(level.value, {})
        return {
            "level": level.value,
            "label": info.get("label", ""),
            "description": info.get("description", ""),
            "typical_characteristics": info.get("typical_characteristics", []),
            "score_range": f"{info.get('min_score', 0)}-{info.get('max_score', 0)}",
        }

    def get_dimension_indicators(
        self, dimension: ScorecardDimension
    ) -> List[Dict[str, Any]]:
        """Get indicator definitions for a dimension.

        Args:
            dimension: Scorecard dimension.

        Returns:
            List of indicator definitions.
        """
        indicators = DIMENSION_INDICATORS.get(dimension.value, [])
        return [
            {
                "id": ind["id"],
                "name": ind["name"],
                "max_points": str(ind["max_points"]),
                "description": ind["description"],
            }
            for ind in indicators
        ]

    def get_benchmark_context(
        self, dimension: ScorecardDimension, level: MaturityLevel
    ) -> str:
        """Get benchmark context for a dimension at a maturity level.

        Args:
            dimension: Scorecard dimension.
            level: Maturity level.

        Returns:
            Description of what "good" looks like.
        """
        bench = BENCHMARK_CONTEXT.get(dimension.value, {})
        return bench.get(level.value, "No benchmark context available.")

    def get_all_dimensions(self) -> List[Dict[str, Any]]:
        """Get all dimension definitions with their weights and indicators.

        Returns:
            List of dimension definitions.
        """
        dims: List[Dict[str, Any]] = []
        for dim in ScorecardDimension:
            indicators = DIMENSION_INDICATORS.get(dim.value, [])
            max_points = sum(ind["max_points"] for ind in indicators)
            dims.append({
                "dimension": dim.value,
                "weight": str(DEFAULT_DIMENSION_WEIGHTS.get(dim.value, 0)),
                "indicator_count": len(indicators),
                "max_points": str(max_points),
                "indicators": [ind["id"] for ind in indicators],
            })
        return dims
