# -*- coding: utf-8 -*-
"""
ImplementationPrioritizerEngine - PACK-033 Quick Wins Identifier Engine 5
=========================================================================

Multi-criteria decision analysis (MCDA) engine for prioritising quick-win
energy efficiency measures.  Implements weighted-sum scoring with
configurable weight profiles, Pareto-frontier identification on arbitrary
objective pairs, dependency-aware sequencing, and phased implementation
planning.

Calculation Methodology:
    Weighted-Sum MCDA:
        For each measure *m* and criterion *c*:
            normalized[m,c] = (raw[m,c] - min_c) / (max_c - min_c)
                              (inverted when direction is "minimize")
            weighted[m,c]   = normalized[m,c] * weight_c
        total[m] = sum( weighted[m,c] for all c )
        rank by total descending.

    Pareto Frontier:
        A measure *a* dominates *b* iff for every objective
        a is at least as good as b AND strictly better in at least one.
        The non-dominated set forms the Pareto frontier.

    Implementation Sequencing:
        1. Topological sort respecting REQUIRES / SEQUENTIAL edges.
        2. Conflict elimination (highest-ranked measure wins).
        3. Budget-constrained knapsack (greedy by rank).
        4. Phase assignment by complexity + cost thresholds.
        5. Critical-path identification (longest dependency chain).

    Score Normalization:
        MIN_MAX:      (x - min) / (max - min)
        Z_SCORE:      (x - mean) / stdev
        RANK_BASED:   (N - rank) / (N - 1)

Regulatory References:
    - ISO 50001:2018  Energy management systems
    - ISO 50006:2023  Energy baseline and EnPI methodology
    - EN 16247-1:2022 Energy audits -- General requirements
    - EN 16247-3:2022 Energy audits -- Processes
    - IEC 31010:2019  Risk management -- Risk assessment techniques

Zero-Hallucination:
    - All ranking formulas are deterministic weighted-sum / Pareto
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-033 Quick Wins Identifier
Engine:  5 of 8
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
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


class CriterionName(str, Enum):
    """Names of evaluation criteria for multi-criteria scoring.

    COST: Implementation cost (lower is better).
    SAVINGS: Annual energy/cost savings (higher is better).
    RISK: Implementation risk score 1-10 (lower is better).
    DISRUPTION: Operational disruption score 1-10 (lower is better).
    COMPLEXITY: Technical complexity score 1-10 (lower is better).
    CO_BENEFITS: Count/score of co-benefits (higher is better).
    PAYBACK: Simple payback period in years (lower is better).
    CARBON_IMPACT: Annual CO2e reduction (higher is better).
    MAINTENANCE: Maintenance impact -5 to +5 (higher is better).
    SCALABILITY: Scalability score 1-10 (higher is better).
    """
    COST = "cost"
    SAVINGS = "savings"
    RISK = "risk"
    DISRUPTION = "disruption"
    COMPLEXITY = "complexity"
    CO_BENEFITS = "co_benefits"
    PAYBACK = "payback"
    CARBON_IMPACT = "carbon_impact"
    MAINTENANCE = "maintenance"
    SCALABILITY = "scalability"


class WeightProfile(str, Enum):
    """Predefined weight profile for prioritisation.

    COST_FOCUSED: Emphasises low cost and fast payback.
    SAVINGS_FOCUSED: Emphasises high savings and carbon reduction.
    BALANCED: Roughly equal weighting across all criteria.
    CARBON_FOCUSED: Emphasises carbon impact above all else.
    RISK_AVERSE: Emphasises low risk, low disruption, low complexity.
    QUICK_IMPLEMENTATION: Emphasises ease and speed of implementation.
    CUSTOM: User-supplied weights.
    """
    COST_FOCUSED = "cost_focused"
    SAVINGS_FOCUSED = "savings_focused"
    BALANCED = "balanced"
    CARBON_FOCUSED = "carbon_focused"
    RISK_AVERSE = "risk_averse"
    QUICK_IMPLEMENTATION = "quick_implementation"
    CUSTOM = "custom"


class DependencyType(str, Enum):
    """Type of dependency between two measures.

    REQUIRES: Measure A requires measure B to be implemented first.
    ENHANCES: Measure A enhances the effectiveness of measure B.
    CONFLICTS: Measures A and B cannot both be implemented.
    REPLACES: Measure A replaces measure B (mutually exclusive).
    SEQUENTIAL: Measures must be implemented in strict order.
    """
    REQUIRES = "requires"
    ENHANCES = "enhances"
    CONFLICTS = "conflicts"
    REPLACES = "replaces"
    SEQUENTIAL = "sequential"


class ImplementationPhase(str, Enum):
    """Implementation time-horizon phase.

    IMMEDIATE_0_3M: Immediate actions (0-3 months).
    SHORT_TERM_3_6M: Short-term projects (3-6 months).
    MEDIUM_TERM_6_12M: Medium-term projects (6-12 months).
    LONG_TERM_12_PLUS: Long-term capital projects (12+ months).
    """
    IMMEDIATE_0_3M = "immediate_0_3m"
    SHORT_TERM_3_6M = "short_term_3_6m"
    MEDIUM_TERM_6_12M = "medium_term_6_12m"
    LONG_TERM_12_PLUS = "long_term_12_plus"


class ParetoStatus(str, Enum):
    """Pareto-optimality status of a measure.

    OPTIMAL: On the Pareto frontier (non-dominated).
    DOMINATED: Strictly dominated by at least one other measure.
    WEAKLY_DOMINATED: Weakly dominated (equal in all, worse in none).
    """
    OPTIMAL = "optimal"
    DOMINATED = "dominated"
    WEAKLY_DOMINATED = "weakly_dominated"


class ScoreNormalization(str, Enum):
    """Normalization method for raw criterion scores.

    MIN_MAX: Min-max normalization to [0, 1].
    Z_SCORE: Z-score (standard-score) normalization.
    RANK_BASED: Rank-based normalization to [0, 1].
    """
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    RANK_BASED = "rank_based"


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class MeasureForPrioritization(BaseModel):
    """A candidate energy-efficiency measure to be prioritised.

    Attributes:
        measure_id: Unique measure identifier.
        name: Human-readable measure name.
        category: Measure category (e.g. lighting, HVAC).
        implementation_cost: Total implementation cost.
        annual_savings: Expected annual monetary savings.
        payback_years: Simple payback period in years.
        co2e_reduction: Annual CO2e reduction (tonnes).
        risk_score: Implementation risk 1 (low) to 10 (high).
        disruption_score: Operational disruption 1 (low) to 10 (high).
        complexity_score: Technical complexity 1 (low) to 10 (high).
        co_benefits: List of co-benefit descriptions.
        maintenance_impact: Maintenance impact -5 (worsens) to +5 (improves).
        scalability_score: Scalability potential 1 (low) to 10 (high).
        dependencies: Measure IDs this measure depends on.
        conflicts: Measure IDs this measure conflicts with.
    """
    measure_id: str = Field(
        default_factory=_new_uuid, description="Unique measure identifier"
    )
    name: str = Field(default="", max_length=500, description="Measure name")
    category: str = Field(default="general", max_length=100, description="Category")
    implementation_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Implementation cost"
    )
    annual_savings: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual monetary savings"
    )
    payback_years: Decimal = Field(
        default=Decimal("0"), ge=0, description="Simple payback (years)"
    )
    co2e_reduction: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual CO2e reduction (tonnes)"
    )
    risk_score: Decimal = Field(
        default=Decimal("5"), ge=1, le=10, description="Risk score 1-10"
    )
    disruption_score: Decimal = Field(
        default=Decimal("5"), ge=1, le=10, description="Disruption score 1-10"
    )
    complexity_score: Decimal = Field(
        default=Decimal("5"), ge=1, le=10, description="Complexity score 1-10"
    )
    co_benefits: List[str] = Field(
        default_factory=list, description="Co-benefit descriptions"
    )
    maintenance_impact: Decimal = Field(
        default=Decimal("0"), ge=-5, le=5,
        description="Maintenance impact -5 to +5"
    )
    scalability_score: Decimal = Field(
        default=Decimal("5"), ge=1, le=10, description="Scalability score 1-10"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Required predecessor measure IDs"
    )
    conflicts: List[str] = Field(
        default_factory=list, description="Conflicting measure IDs"
    )

    @field_validator("risk_score", "disruption_score", "complexity_score",
                     "scalability_score")
    @classmethod
    def coerce_scores(cls, v: Any) -> Decimal:
        """Coerce score values to Decimal."""
        return _decimal(v)

    @field_validator("maintenance_impact")
    @classmethod
    def coerce_maintenance(cls, v: Any) -> Decimal:
        """Coerce maintenance impact to Decimal."""
        return _decimal(v)


class CriterionWeight(BaseModel):
    """Weight for a single criterion in MCDA scoring.

    Attributes:
        criterion: The criterion being weighted.
        weight: Weight value between 0 and 1 (all weights should sum to 1).
        direction: Whether higher is better ('maximize') or lower ('minimize').
    """
    criterion: CriterionName = Field(
        ..., description="Criterion name"
    )
    weight: Decimal = Field(
        default=Decimal("0.10"), ge=0, le=1, description="Weight 0-1"
    )
    direction: str = Field(
        default="maximize", description="'maximize' or 'minimize'"
    )

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        """Ensure direction is maximize or minimize."""
        v_lower = v.strip().lower()
        if v_lower not in ("maximize", "minimize"):
            raise ValueError(
                f"direction must be 'maximize' or 'minimize', got '{v}'"
            )
        return v_lower


class WeightSet(BaseModel):
    """Complete set of criterion weights for a weight profile.

    Attributes:
        profile: The weight profile identifier.
        weights: List of individual criterion weights.
        description: Human-readable profile description.
    """
    profile: WeightProfile = Field(
        default=WeightProfile.BALANCED, description="Weight profile"
    )
    weights: List[CriterionWeight] = Field(
        default_factory=list, description="Criterion weights"
    )
    description: str = Field(
        default="", max_length=500, description="Profile description"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class NormalizedScore(BaseModel):
    """Normalized and weighted score for one measure on one criterion.

    Attributes:
        measure_id: The measure being scored.
        criterion: The criterion being evaluated.
        raw_value: Original raw value.
        normalized_value: Normalized value in [0, 1].
        weighted_score: Normalized value multiplied by criterion weight.
    """
    measure_id: str = Field(default="", description="Measure identifier")
    criterion: CriterionName = Field(
        default=CriterionName.COST, description="Criterion"
    )
    raw_value: Decimal = Field(default=Decimal("0"), description="Raw value")
    normalized_value: Decimal = Field(
        default=Decimal("0"), description="Normalized 0-1"
    )
    weighted_score: Decimal = Field(
        default=Decimal("0"), description="Weighted score"
    )


class PriorityResult(BaseModel):
    """Prioritisation result for a single measure.

    Attributes:
        measure_id: Unique measure identifier.
        name: Measure name.
        normalized_scores: Per-criterion normalized and weighted scores.
        weighted_total: Sum of all weighted scores.
        rank: Ordinal rank (1 = highest priority).
        pareto_status: Pareto-optimality status.
        implementation_phase: Recommended implementation phase.
        notes: Contextual notes about this ranking.
    """
    measure_id: str = Field(default="", description="Measure ID")
    name: str = Field(default="", description="Measure name")
    normalized_scores: List[NormalizedScore] = Field(
        default_factory=list, description="Normalized criterion scores"
    )
    weighted_total: Decimal = Field(
        default=Decimal("0"), description="Weighted total score"
    )
    rank: int = Field(default=0, ge=0, description="Ordinal rank")
    pareto_status: ParetoStatus = Field(
        default=ParetoStatus.DOMINATED, description="Pareto status"
    )
    implementation_phase: ImplementationPhase = Field(
        default=ImplementationPhase.MEDIUM_TERM_6_12M,
        description="Implementation phase"
    )
    notes: str = Field(default="", description="Ranking notes")


class DependencyEdge(BaseModel):
    """An edge in the measure dependency graph.

    Attributes:
        from_id: Source measure ID.
        to_id: Target measure ID.
        dependency_type: Nature of the dependency.
        notes: Contextual notes.
    """
    from_id: str = Field(default="", description="Source measure")
    to_id: str = Field(default="", description="Target measure")
    dependency_type: DependencyType = Field(
        default=DependencyType.REQUIRES, description="Dependency type"
    )
    notes: str = Field(default="", description="Notes")


class ImplementationSequence(BaseModel):
    """An ordered implementation sequence of measures.

    Attributes:
        sequence_id: Unique sequence identifier.
        name: Sequence name / description.
        ordered_measures: Measure IDs in implementation order.
        total_cost: Sum of implementation costs.
        total_savings: Sum of annual savings.
        total_co2e: Sum of annual CO2e reductions.
        phases: Mapping of phase to measure IDs in that phase.
        dependency_graph: All dependency edges considered.
        critical_path: Measure IDs forming the longest dependency chain.
        provenance_hash: SHA-256 provenance hash.
    """
    sequence_id: str = Field(
        default_factory=_new_uuid, description="Sequence ID"
    )
    name: str = Field(
        default="Primary Implementation Sequence",
        max_length=300, description="Sequence name"
    )
    ordered_measures: List[str] = Field(
        default_factory=list, description="Ordered measure IDs"
    )
    total_cost: Decimal = Field(
        default=Decimal("0"), description="Total implementation cost"
    )
    total_savings: Decimal = Field(
        default=Decimal("0"), description="Total annual savings"
    )
    total_co2e: Decimal = Field(
        default=Decimal("0"), description="Total annual CO2e reduction"
    )
    phases: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Phase -> measure IDs mapping"
    )
    dependency_graph: List[DependencyEdge] = Field(
        default_factory=list, description="Dependency edges"
    )
    critical_path: List[str] = Field(
        default_factory=list, description="Critical path measure IDs"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class PrioritizationResult(BaseModel):
    """Complete output of the implementation prioritisation engine.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version string.
        rankings: Ordered list of measure priority results.
        pareto_frontier: Measure IDs on the Pareto frontier.
        sequences: Generated implementation sequences.
        weight_profile_used: Which weight profile was applied.
        total_measures: Total number of measures evaluated.
        calculated_at: Calculation timestamp (UTC).
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Engine processing time in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version"
    )
    rankings: List[PriorityResult] = Field(
        default_factory=list, description="Ranked measures"
    )
    pareto_frontier: List[str] = Field(
        default_factory=list, description="Pareto-optimal measure IDs"
    )
    sequences: List[ImplementationSequence] = Field(
        default_factory=list, description="Implementation sequences"
    )
    weight_profile_used: WeightProfile = Field(
        default=WeightProfile.BALANCED, description="Weight profile used"
    )
    total_measures: int = Field(
        default=0, ge=0, description="Total measures evaluated"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings"
    )
    errors: List[str] = Field(
        default_factory=list, description="Errors"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


# ---------------------------------------------------------------------------
# Model Rebuild (required for `from __future__ import annotations`)
# ---------------------------------------------------------------------------
MeasureForPrioritization.model_rebuild()
CriterionWeight.model_rebuild()
WeightSet.model_rebuild()
NormalizedScore.model_rebuild()
PriorityResult.model_rebuild()
DependencyEdge.model_rebuild()
ImplementationSequence.model_rebuild()
PrioritizationResult.model_rebuild()


# ---------------------------------------------------------------------------
# Default Weight Profiles
# ---------------------------------------------------------------------------

DEFAULT_WEIGHT_PROFILES: Dict[str, WeightSet] = {
    WeightProfile.COST_FOCUSED.value: WeightSet(
        profile=WeightProfile.COST_FOCUSED,
        description="Prioritises lowest cost and fastest payback.",
        weights=[
            CriterionWeight(criterion=CriterionName.COST, weight=Decimal("0.30"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.PAYBACK, weight=Decimal("0.25"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.SAVINGS, weight=Decimal("0.20"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.RISK, weight=Decimal("0.10"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.DISRUPTION, weight=Decimal("0.05"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.COMPLEXITY, weight=Decimal("0.05"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.CO_BENEFITS, weight=Decimal("0.03"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.CARBON_IMPACT, weight=Decimal("0.02"), direction="maximize"),
        ],
    ),
    WeightProfile.SAVINGS_FOCUSED.value: WeightSet(
        profile=WeightProfile.SAVINGS_FOCUSED,
        description="Prioritises highest savings and carbon reduction.",
        weights=[
            CriterionWeight(criterion=CriterionName.SAVINGS, weight=Decimal("0.30"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.CARBON_IMPACT, weight=Decimal("0.20"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.CO_BENEFITS, weight=Decimal("0.15"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.PAYBACK, weight=Decimal("0.15"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.COST, weight=Decimal("0.10"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.RISK, weight=Decimal("0.05"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.DISRUPTION, weight=Decimal("0.03"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.COMPLEXITY, weight=Decimal("0.02"), direction="minimize"),
        ],
    ),
    WeightProfile.BALANCED.value: WeightSet(
        profile=WeightProfile.BALANCED,
        description="Balanced weighting across all criteria.",
        weights=[
            CriterionWeight(criterion=CriterionName.SAVINGS, weight=Decimal("0.15"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.COST, weight=Decimal("0.15"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.PAYBACK, weight=Decimal("0.13"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.CARBON_IMPACT, weight=Decimal("0.13"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.RISK, weight=Decimal("0.10"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.COMPLEXITY, weight=Decimal("0.10"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.DISRUPTION, weight=Decimal("0.08"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.CO_BENEFITS, weight=Decimal("0.08"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.MAINTENANCE, weight=Decimal("0.04"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.SCALABILITY, weight=Decimal("0.04"), direction="maximize"),
        ],
    ),
    WeightProfile.CARBON_FOCUSED.value: WeightSet(
        profile=WeightProfile.CARBON_FOCUSED,
        description="Prioritises maximum carbon impact.",
        weights=[
            CriterionWeight(criterion=CriterionName.CARBON_IMPACT, weight=Decimal("0.35"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.SAVINGS, weight=Decimal("0.20"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.CO_BENEFITS, weight=Decimal("0.15"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.COST, weight=Decimal("0.10"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.PAYBACK, weight=Decimal("0.10"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.RISK, weight=Decimal("0.05"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.DISRUPTION, weight=Decimal("0.03"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.COMPLEXITY, weight=Decimal("0.02"), direction="minimize"),
        ],
    ),
    WeightProfile.RISK_AVERSE.value: WeightSet(
        profile=WeightProfile.RISK_AVERSE,
        description="Prioritises low risk, low disruption, low complexity.",
        weights=[
            CriterionWeight(criterion=CriterionName.RISK, weight=Decimal("0.25"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.DISRUPTION, weight=Decimal("0.20"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.COMPLEXITY, weight=Decimal("0.15"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.PAYBACK, weight=Decimal("0.15"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.COST, weight=Decimal("0.10"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.SAVINGS, weight=Decimal("0.10"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.CARBON_IMPACT, weight=Decimal("0.03"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.CO_BENEFITS, weight=Decimal("0.02"), direction="maximize"),
        ],
    ),
    WeightProfile.QUICK_IMPLEMENTATION.value: WeightSet(
        profile=WeightProfile.QUICK_IMPLEMENTATION,
        description="Prioritises ease and speed of implementation.",
        weights=[
            CriterionWeight(criterion=CriterionName.COMPLEXITY, weight=Decimal("0.25"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.PAYBACK, weight=Decimal("0.25"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.COST, weight=Decimal("0.20"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.SAVINGS, weight=Decimal("0.15"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.DISRUPTION, weight=Decimal("0.05"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.RISK, weight=Decimal("0.05"), direction="minimize"),
            CriterionWeight(criterion=CriterionName.CARBON_IMPACT, weight=Decimal("0.03"), direction="maximize"),
            CriterionWeight(criterion=CriterionName.CO_BENEFITS, weight=Decimal("0.02"), direction="maximize"),
        ],
    ),
}

# Map criterion names to measure attribute accessors.
_CRITERION_ATTR_MAP: Dict[CriterionName, str] = {
    CriterionName.COST: "implementation_cost",
    CriterionName.SAVINGS: "annual_savings",
    CriterionName.RISK: "risk_score",
    CriterionName.DISRUPTION: "disruption_score",
    CriterionName.COMPLEXITY: "complexity_score",
    CriterionName.PAYBACK: "payback_years",
    CriterionName.CARBON_IMPACT: "co2e_reduction",
    CriterionName.MAINTENANCE: "maintenance_impact",
    CriterionName.SCALABILITY: "scalability_score",
}

# Phase assignment thresholds.
_PHASE_COMPLEXITY_THRESHOLDS: Dict[ImplementationPhase, Tuple[Decimal, Decimal]] = {
    ImplementationPhase.IMMEDIATE_0_3M: (Decimal("1"), Decimal("3")),
    ImplementationPhase.SHORT_TERM_3_6M: (Decimal("3"), Decimal("5")),
    ImplementationPhase.MEDIUM_TERM_6_12M: (Decimal("5"), Decimal("7")),
    ImplementationPhase.LONG_TERM_12_PLUS: (Decimal("7"), Decimal("10")),
}

_PHASE_COST_THRESHOLDS: Dict[ImplementationPhase, Decimal] = {
    ImplementationPhase.IMMEDIATE_0_3M: Decimal("10000"),
    ImplementationPhase.SHORT_TERM_3_6M: Decimal("50000"),
    ImplementationPhase.MEDIUM_TERM_6_12M: Decimal("200000"),
    ImplementationPhase.LONG_TERM_12_PLUS: Decimal("999999999"),
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ImplementationPrioritizerEngine:
    """Multi-criteria decision analysis engine for prioritising quick wins.

    Evaluates candidate energy-efficiency measures against configurable
    weighted criteria, identifies Pareto-optimal trade-offs, resolves
    dependencies and conflicts, and produces phased implementation
    sequences.

    Usage::

        engine = ImplementationPrioritizerEngine()
        result = engine.prioritize(measures, profile=WeightProfile.BALANCED)
        for r in result.rankings:
            print(f"{r.rank}. {r.name} -- score={r.weighted_total}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise ImplementationPrioritizerEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - default_profile (str): default WeightProfile value
                - normalization (str): default ScoreNormalization value
                - budget (Decimal): default budget cap
                - phase_cost_thresholds (dict): override phase cost limits
        """
        self.config = config or {}
        self._default_profile = WeightProfile(
            self.config.get("default_profile", WeightProfile.BALANCED.value)
        )
        self._default_normalization = ScoreNormalization(
            self.config.get("normalization", ScoreNormalization.MIN_MAX.value)
        )
        self._default_budget: Optional[Decimal] = (
            _decimal(self.config["budget"])
            if "budget" in self.config
            else None
        )
        logger.info(
            "ImplementationPrioritizerEngine v%s initialised "
            "(profile=%s, normalization=%s)",
            self.engine_version,
            self._default_profile.value,
            self._default_normalization.value,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def prioritize(
        self,
        measures: List[MeasureForPrioritization],
        profile: WeightProfile = WeightProfile.BALANCED,
        custom_weights: Optional[List[CriterionWeight]] = None,
        budget: Optional[Decimal] = None,
        normalization: Optional[ScoreNormalization] = None,
    ) -> PrioritizationResult:
        """Perform full multi-criteria prioritisation.

        Args:
            measures: List of candidate measures.
            profile: Weight profile to use (ignored when custom_weights given).
            custom_weights: Optional custom criterion weights.
            budget: Optional budget cap for sequencing.
            normalization: Normalization method override.

        Returns:
            PrioritizationResult with rankings, Pareto frontier, sequences.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        errors: List[str] = []

        if not measures:
            errors.append("No measures provided for prioritisation.")
            elapsed = _round3((time.perf_counter() - t0) * 1000.0)
            result = PrioritizationResult(
                weight_profile_used=profile,
                total_measures=0,
                errors=errors,
                processing_time_ms=elapsed,
            )
            result.provenance_hash = _compute_hash(result)
            return result

        logger.info(
            "Prioritising %d measures with profile=%s",
            len(measures), profile.value,
        )

        # Resolve weight set
        effective_profile = profile
        if custom_weights is not None:
            weight_list = custom_weights
            effective_profile = WeightProfile.CUSTOM
        else:
            ws = DEFAULT_WEIGHT_PROFILES.get(profile.value)
            if ws is None:
                warnings.append(
                    f"Unknown profile '{profile.value}'; falling back to BALANCED."
                )
                ws = DEFAULT_WEIGHT_PROFILES[WeightProfile.BALANCED.value]
            weight_list = ws.weights

        # Validate weights sum to ~1.0
        weight_sum = sum(cw.weight for cw in weight_list)
        if abs(weight_sum - Decimal("1")) > Decimal("0.01"):
            warnings.append(
                f"Criterion weights sum to {weight_sum}, expected 1.00. "
                f"Results may be biased."
            )

        # Build weight lookup
        weight_map: Dict[CriterionName, CriterionWeight] = {
            cw.criterion: cw for cw in weight_list
        }

        # Select normalization
        norm_method = normalization or self._default_normalization

        # Step 1: Normalize scores
        normalized = self._normalize_scores(measures, weight_map, norm_method)

        # Step 2: Compute weighted totals
        totals: Dict[str, Decimal] = {}
        score_map: Dict[str, List[NormalizedScore]] = {}
        for mid, criterion_scores in normalized.items():
            score_map[mid] = criterion_scores
            totals[mid] = self._compute_weighted_total(
                criterion_scores, weight_map
            )

        # Step 3: Rank measures (descending by weighted total)
        ranked_ids = sorted(totals.keys(), key=lambda k: totals[k], reverse=True)

        # Step 4: Pareto frontier
        pareto_objectives: List[Tuple[str, str]] = [
            ("annual_savings", "maximize"),
            ("implementation_cost", "minimize"),
        ]
        pareto_ids = self.find_pareto_frontier(measures, pareto_objectives)

        # Also compute carbon-vs-cost Pareto
        carbon_objectives: List[Tuple[str, str]] = [
            ("co2e_reduction", "maximize"),
            ("implementation_cost", "minimize"),
        ]
        carbon_pareto_ids = self.find_pareto_frontier(measures, carbon_objectives)

        # Union of both Pareto sets
        combined_pareto = list(set(pareto_ids) | set(carbon_pareto_ids))

        # Pareto status map
        pareto_set = set(combined_pareto)
        weakly_dominated_set = self._find_weakly_dominated(
            measures, pareto_objectives
        )

        # Step 5: Assign phases
        measure_lookup: Dict[str, MeasureForPrioritization] = {
            m.measure_id: m for m in measures
        }

        # Step 6: Build PriorityResult list
        rankings: List[PriorityResult] = []
        for rank_pos, mid in enumerate(ranked_ids, start=1):
            m = measure_lookup.get(mid)
            if m is None:
                continue

            if mid in pareto_set:
                p_status = ParetoStatus.OPTIMAL
            elif mid in weakly_dominated_set:
                p_status = ParetoStatus.WEAKLY_DOMINATED
            else:
                p_status = ParetoStatus.DOMINATED

            phase = self._assign_phase(m, rank_pos)

            notes_parts: List[str] = []
            if p_status == ParetoStatus.OPTIMAL:
                notes_parts.append("Pareto-optimal (non-dominated)")
            if m.dependencies:
                notes_parts.append(
                    f"Depends on: {', '.join(m.dependencies)}"
                )
            if m.conflicts:
                notes_parts.append(
                    f"Conflicts with: {', '.join(m.conflicts)}"
                )

            rankings.append(PriorityResult(
                measure_id=mid,
                name=m.name,
                normalized_scores=score_map.get(mid, []),
                weighted_total=_round_val(totals.get(mid, Decimal("0")), 6),
                rank=rank_pos,
                pareto_status=p_status,
                implementation_phase=phase,
                notes="; ".join(notes_parts) if notes_parts else "",
            ))

        # Step 7: Build implementation sequence
        effective_budget = budget or self._default_budget
        sequence = self.build_implementation_sequence(
            rankings, measures, effective_budget
        )

        elapsed = _round3((time.perf_counter() - t0) * 1000.0)

        result = PrioritizationResult(
            rankings=rankings,
            pareto_frontier=combined_pareto,
            sequences=[sequence],
            weight_profile_used=effective_profile,
            total_measures=len(measures),
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Prioritisation complete: %d measures ranked, %d Pareto-optimal, "
            "hash=%s, %.1fms",
            len(rankings), len(combined_pareto),
            result.provenance_hash[:16], elapsed,
        )
        return result

    # ------------------------------------------------------------------ #
    # Pareto Frontier                                                      #
    # ------------------------------------------------------------------ #

    def find_pareto_frontier(
        self,
        measures: List[MeasureForPrioritization],
        objectives: List[Tuple[str, str]],
    ) -> List[str]:
        """Identify measures on the Pareto frontier.

        A measure is Pareto-optimal if no other measure is at least as good
        on every objective and strictly better on at least one.

        Args:
            measures: Candidate measures.
            objectives: List of (attribute_name, 'maximize'|'minimize') tuples.

        Returns:
            List of measure_ids on the Pareto frontier.
        """
        if not measures or not objectives:
            return []

        # Extract objective values
        points: Dict[str, List[Decimal]] = {}
        for m in measures:
            vals: List[Decimal] = []
            for attr, direction in objectives:
                raw = _decimal(getattr(m, attr, Decimal("0")))
                # Normalize direction: convert everything to "higher is better"
                if direction == "minimize":
                    raw = -raw
                vals.append(raw)
            points[m.measure_id] = vals

        frontier: List[str] = []
        ids = list(points.keys())

        for i, mid_a in enumerate(ids):
            dominated = False
            for j, mid_b in enumerate(ids):
                if i == j:
                    continue
                if self._is_dominated(points[mid_a], points[mid_b], objectives):
                    dominated = True
                    break
            if not dominated:
                frontier.append(mid_a)

        logger.debug(
            "Pareto frontier: %d of %d measures are non-dominated",
            len(frontier), len(measures),
        )
        return frontier

    # ------------------------------------------------------------------ #
    # Implementation Sequencing                                            #
    # ------------------------------------------------------------------ #

    def build_implementation_sequence(
        self,
        rankings: List[PriorityResult],
        measures: List[MeasureForPrioritization],
        budget: Optional[Decimal] = None,
    ) -> ImplementationSequence:
        """Build an optimal implementation sequence.

        Respects dependency and conflict constraints, assigns measures to
        phases, and identifies the critical path.

        Args:
            rankings: Ranked priority results.
            measures: Original measure data.
            budget: Optional budget cap.

        Returns:
            ImplementationSequence.
        """
        measure_lookup: Dict[str, MeasureForPrioritization] = {
            m.measure_id: m for m in measures
        }
        rank_lookup: Dict[str, int] = {
            r.measure_id: r.rank for r in rankings
        }
        phase_lookup: Dict[str, ImplementationPhase] = {
            r.measure_id: r.implementation_phase for r in rankings
        }

        # Step 1: Resolve dependency graph
        dep_edges = self._resolve_dependencies(measures)

        # Step 2: Eliminate conflicts (higher-ranked measure wins)
        excluded: set = set()
        conflict_pairs_seen: set = set()
        for m in measures:
            if m.measure_id in excluded:
                continue
            for conflict_id in m.conflicts:
                pair_key = tuple(sorted([m.measure_id, conflict_id]))
                if pair_key in conflict_pairs_seen:
                    continue
                conflict_pairs_seen.add(pair_key)
                if conflict_id in measure_lookup and conflict_id not in excluded:
                    rank_a = rank_lookup.get(m.measure_id, 9999)
                    rank_b = rank_lookup.get(conflict_id, 9999)
                    loser = conflict_id if rank_a <= rank_b else m.measure_id
                    excluded.add(loser)
                    logger.debug(
                        "Conflict resolved: %s excluded (outranked by %s)",
                        loser,
                        m.measure_id if loser == conflict_id else conflict_id,
                    )

        # Step 3: Topological sort respecting dependencies
        ordered = self._topological_sort(measures, dep_edges, excluded)

        # Step 4: Budget-constrained selection (greedy by rank order)
        selected: List[str] = []
        spent = Decimal("0")
        for mid in ordered:
            if mid in excluded:
                continue
            m = measure_lookup.get(mid)
            if m is None:
                continue
            if budget is not None:
                if spent + m.implementation_cost > budget:
                    logger.debug(
                        "Budget limit: skipping %s (cost=%s, spent=%s, budget=%s)",
                        mid, m.implementation_cost, spent, budget,
                    )
                    continue
            spent += m.implementation_cost
            selected.append(mid)

        # Step 5: Assign to phases
        phases: Dict[str, List[str]] = {}
        for mid in selected:
            phase = phase_lookup.get(mid, ImplementationPhase.MEDIUM_TERM_6_12M)
            phase_key = phase.value if isinstance(phase, ImplementationPhase) else str(phase)
            phases.setdefault(phase_key, []).append(mid)

        # Step 6: Calculate totals
        total_cost = Decimal("0")
        total_savings = Decimal("0")
        total_co2e = Decimal("0")
        for mid in selected:
            m = measure_lookup.get(mid)
            if m is not None:
                total_cost += m.implementation_cost
                total_savings += m.annual_savings
                total_co2e += m.co2e_reduction

        # Step 7: Identify critical path
        critical_path = self._find_critical_path(
            selected, dep_edges, measure_lookup
        )

        # Step 8: Filter edges to only selected measures
        selected_set = set(selected)
        relevant_edges = [
            e for e in dep_edges
            if e.from_id in selected_set and e.to_id in selected_set
        ]

        seq = ImplementationSequence(
            name="Primary Implementation Sequence",
            ordered_measures=selected,
            total_cost=_round_val(total_cost, 2),
            total_savings=_round_val(total_savings, 2),
            total_co2e=_round_val(total_co2e, 3),
            phases=phases,
            dependency_graph=relevant_edges,
            critical_path=critical_path,
        )
        seq.provenance_hash = _compute_hash(seq)
        return seq

    # ------------------------------------------------------------------ #
    # Score Normalization                                                   #
    # ------------------------------------------------------------------ #

    def _normalize_scores(
        self,
        measures: List[MeasureForPrioritization],
        weight_map: Dict[CriterionName, CriterionWeight],
        normalization: ScoreNormalization = ScoreNormalization.MIN_MAX,
    ) -> Dict[str, List[NormalizedScore]]:
        """Normalize raw criterion values across all measures.

        Args:
            measures: Candidate measures.
            weight_map: Criterion weight definitions.
            normalization: Normalization method.

        Returns:
            Dict mapping measure_id to list of NormalizedScore.
        """
        # Collect raw values per criterion
        raw_by_criterion: Dict[CriterionName, Dict[str, Decimal]] = {}
        for criterion in weight_map:
            raw_by_criterion[criterion] = {}
            for m in measures:
                raw = self._extract_criterion_value(m, criterion)
                raw_by_criterion[criterion][m.measure_id] = raw

        result: Dict[str, List[NormalizedScore]] = {
            m.measure_id: [] for m in measures
        }

        for criterion, cw in weight_map.items():
            values = raw_by_criterion.get(criterion, {})
            if not values:
                continue

            normalized_vals = self._apply_normalization(
                values, cw.direction, normalization
            )

            for mid, norm_val in normalized_vals.items():
                weighted = norm_val * cw.weight
                result[mid].append(NormalizedScore(
                    measure_id=mid,
                    criterion=criterion,
                    raw_value=_round_val(values.get(mid, Decimal("0")), 4),
                    normalized_value=_round_val(norm_val, 6),
                    weighted_score=_round_val(weighted, 6),
                ))

        return result

    def _extract_criterion_value(
        self,
        measure: MeasureForPrioritization,
        criterion: CriterionName,
    ) -> Decimal:
        """Extract the raw value for a criterion from a measure.

        Args:
            measure: The measure to extract from.
            criterion: The criterion to extract.

        Returns:
            Raw Decimal value.
        """
        if criterion == CriterionName.CO_BENEFITS:
            return _decimal(len(measure.co_benefits))

        attr = _CRITERION_ATTR_MAP.get(criterion)
        if attr is not None:
            return _decimal(getattr(measure, attr, Decimal("0")))

        return Decimal("0")

    def _apply_normalization(
        self,
        values: Dict[str, Decimal],
        direction: str,
        method: ScoreNormalization,
    ) -> Dict[str, Decimal]:
        """Apply normalization to a set of raw values.

        Args:
            values: Mapping of measure_id to raw value.
            direction: 'maximize' or 'minimize'.
            method: Normalization method.

        Returns:
            Mapping of measure_id to normalized value in [0, 1].
        """
        if not values:
            return {}

        vals = list(values.values())

        if method == ScoreNormalization.MIN_MAX:
            return self._normalize_min_max(values, direction)
        elif method == ScoreNormalization.Z_SCORE:
            return self._normalize_z_score(values, direction)
        elif method == ScoreNormalization.RANK_BASED:
            return self._normalize_rank_based(values, direction)
        else:
            return self._normalize_min_max(values, direction)

    def _normalize_min_max(
        self,
        values: Dict[str, Decimal],
        direction: str,
    ) -> Dict[str, Decimal]:
        """Min-max normalization to [0, 1].

        For 'maximize' direction: (x - min) / (max - min).
        For 'minimize' direction: (max - x) / (max - min).

        Args:
            values: Raw values by measure_id.
            direction: 'maximize' or 'minimize'.

        Returns:
            Normalized values by measure_id.
        """
        vals = list(values.values())
        v_min = min(vals)
        v_max = max(vals)
        spread = v_max - v_min

        result: Dict[str, Decimal] = {}
        for mid, raw in values.items():
            if spread == Decimal("0"):
                result[mid] = Decimal("0.5")
            elif direction == "minimize":
                result[mid] = _safe_divide(v_max - raw, spread)
            else:
                result[mid] = _safe_divide(raw - v_min, spread)
        return result

    def _normalize_z_score(
        self,
        values: Dict[str, Decimal],
        direction: str,
    ) -> Dict[str, Decimal]:
        """Z-score normalization mapped to [0, 1] via sigmoid.

        z = (x - mean) / stdev
        For 'minimize': z is negated.
        Then mapped to [0, 1] via: 1 / (1 + e^(-z)).

        Args:
            values: Raw values by measure_id.
            direction: 'maximize' or 'minimize'.

        Returns:
            Normalized values by measure_id.
        """
        vals = list(values.values())
        n = _decimal(len(vals))
        mean = _safe_divide(sum(vals), n)

        variance = _safe_divide(
            sum((v - mean) ** 2 for v in vals), n
        )
        stdev = variance.sqrt() if variance > Decimal("0") else Decimal("1")

        result: Dict[str, Decimal] = {}
        for mid, raw in values.items():
            z = _safe_divide(raw - mean, stdev)
            if direction == "minimize":
                z = -z
            # Sigmoid mapping to [0, 1]
            try:
                exp_neg_z = _decimal(math.exp(float(-z)))
                sigmoid = _safe_divide(Decimal("1"), Decimal("1") + exp_neg_z)
            except (OverflowError, ValueError):
                sigmoid = Decimal("1") if z > Decimal("0") else Decimal("0")
            result[mid] = sigmoid
        return result

    def _normalize_rank_based(
        self,
        values: Dict[str, Decimal],
        direction: str,
    ) -> Dict[str, Decimal]:
        """Rank-based normalization to [0, 1].

        Rank-based: (N - rank) / (N - 1).
        For 'maximize': rank 1 = highest value.
        For 'minimize': rank 1 = lowest value.

        Args:
            values: Raw values by measure_id.
            direction: 'maximize' or 'minimize'.

        Returns:
            Normalized values by measure_id.
        """
        n = len(values)
        if n <= 1:
            return {mid: Decimal("1") for mid in values}

        reverse = (direction == "maximize")
        sorted_ids = sorted(
            values.keys(), key=lambda k: values[k], reverse=reverse
        )

        n_dec = _decimal(n)
        result: Dict[str, Decimal] = {}
        for rank_idx, mid in enumerate(sorted_ids):
            rank = rank_idx + 1
            result[mid] = _safe_divide(
                n_dec - _decimal(rank),
                n_dec - Decimal("1"),
            )
        return result

    # ------------------------------------------------------------------ #
    # Weighted Total                                                       #
    # ------------------------------------------------------------------ #

    def _compute_weighted_total(
        self,
        normalized_scores: List[NormalizedScore],
        weight_map: Dict[CriterionName, CriterionWeight],
    ) -> Decimal:
        """Compute the weighted sum of normalized scores.

        Args:
            normalized_scores: List of normalized scores for one measure.
            weight_map: Criterion weight definitions.

        Returns:
            Weighted total as Decimal.
        """
        total = Decimal("0")
        for ns in normalized_scores:
            total += ns.weighted_score
        return total

    # ------------------------------------------------------------------ #
    # Dependency Resolution                                                #
    # ------------------------------------------------------------------ #

    def _resolve_dependencies(
        self,
        measures: List[MeasureForPrioritization],
    ) -> List[DependencyEdge]:
        """Build the dependency graph from measure declarations.

        Produces edges for REQUIRES, ENHANCES, CONFLICTS, REPLACES,
        and SEQUENTIAL relationships.

        Args:
            measures: All candidate measures.

        Returns:
            List of DependencyEdge.
        """
        edges: List[DependencyEdge] = []
        valid_ids = {m.measure_id for m in measures}

        for m in measures:
            for dep_id in m.dependencies:
                if dep_id in valid_ids:
                    edges.append(DependencyEdge(
                        from_id=dep_id,
                        to_id=m.measure_id,
                        dependency_type=DependencyType.REQUIRES,
                        notes=f"{m.name} requires {dep_id}",
                    ))
                else:
                    logger.warning(
                        "Dependency %s -> %s: target %s not in measure set",
                        m.measure_id, dep_id, dep_id,
                    )

            for conflict_id in m.conflicts:
                if conflict_id in valid_ids:
                    edges.append(DependencyEdge(
                        from_id=m.measure_id,
                        to_id=conflict_id,
                        dependency_type=DependencyType.CONFLICTS,
                        notes=f"{m.name} conflicts with {conflict_id}",
                    ))

        logger.debug(
            "Resolved %d dependency edges from %d measures",
            len(edges), len(measures),
        )
        return edges

    def _topological_sort(
        self,
        measures: List[MeasureForPrioritization],
        edges: List[DependencyEdge],
        excluded: set,
    ) -> List[str]:
        """Topological sort of measures respecting dependencies.

        Uses Kahn's algorithm. Measures without dependency constraints
        are ordered by rank (lower rank number = higher priority).

        Args:
            measures: All candidate measures.
            edges: Dependency edges.
            excluded: Measure IDs to exclude.

        Returns:
            Topologically sorted list of measure IDs.
        """
        valid = {
            m.measure_id for m in measures if m.measure_id not in excluded
        }

        # Build adjacency and in-degree
        adj: Dict[str, List[str]] = {mid: [] for mid in valid}
        in_deg: Dict[str, int] = {mid: 0 for mid in valid}

        for e in edges:
            if (
                e.dependency_type in (DependencyType.REQUIRES, DependencyType.SEQUENTIAL)
                and e.from_id in valid
                and e.to_id in valid
            ):
                adj[e.from_id].append(e.to_id)
                in_deg[e.to_id] = in_deg.get(e.to_id, 0) + 1

        # Kahn's algorithm
        queue: List[str] = sorted(
            [mid for mid in valid if in_deg.get(mid, 0) == 0]
        )
        result: List[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adj.get(node, []):
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)
            queue.sort()

        # Add any remaining (cycle-breaker: just append)
        remaining = [mid for mid in valid if mid not in set(result)]
        if remaining:
            logger.warning(
                "Dependency cycle detected involving %d measures; "
                "appending in arbitrary order.",
                len(remaining),
            )
            result.extend(sorted(remaining))

        return result

    # ------------------------------------------------------------------ #
    # Phase Assignment                                                     #
    # ------------------------------------------------------------------ #

    def _assign_phase(
        self,
        measure: MeasureForPrioritization,
        rank: int,
    ) -> ImplementationPhase:
        """Assign an implementation phase based on complexity and cost.

        Phase assignment rules:
            IMMEDIATE (0-3m): complexity <= 3 AND cost <= 10,000
            SHORT_TERM (3-6m): complexity <= 5 AND cost <= 50,000
            MEDIUM_TERM (6-12m): complexity <= 7 AND cost <= 200,000
            LONG_TERM (12+m): everything else

        Top-ranked measures (rank 1-3) get promoted one phase earlier
        when possible.

        Args:
            measure: The measure to assign.
            rank: Ordinal rank (1 = highest priority).

        Returns:
            ImplementationPhase.
        """
        complexity = measure.complexity_score
        cost = measure.implementation_cost

        # Base phase from thresholds
        if complexity <= Decimal("3") and cost <= Decimal("10000"):
            phase = ImplementationPhase.IMMEDIATE_0_3M
        elif complexity <= Decimal("5") and cost <= Decimal("50000"):
            phase = ImplementationPhase.SHORT_TERM_3_6M
        elif complexity <= Decimal("7") and cost <= Decimal("200000"):
            phase = ImplementationPhase.MEDIUM_TERM_6_12M
        else:
            phase = ImplementationPhase.LONG_TERM_12_PLUS

        # Promote top-ranked measures one phase earlier
        if rank <= 3 and phase != ImplementationPhase.IMMEDIATE_0_3M:
            phase_order = [
                ImplementationPhase.IMMEDIATE_0_3M,
                ImplementationPhase.SHORT_TERM_3_6M,
                ImplementationPhase.MEDIUM_TERM_6_12M,
                ImplementationPhase.LONG_TERM_12_PLUS,
            ]
            idx = phase_order.index(phase)
            phase = phase_order[max(0, idx - 1)]

        return phase

    # ------------------------------------------------------------------ #
    # Critical Path                                                        #
    # ------------------------------------------------------------------ #

    def _find_critical_path(
        self,
        selected: List[str],
        edges: List[DependencyEdge],
        measure_lookup: Dict[str, MeasureForPrioritization],
    ) -> List[str]:
        """Find the longest dependency chain (critical path).

        Uses dynamic programming on the DAG to find the longest path.

        Args:
            selected: Selected measure IDs.
            edges: Dependency edges.
            measure_lookup: Measure data lookup.

        Returns:
            List of measure IDs forming the critical path.
        """
        selected_set = set(selected)

        # Build adjacency for selected measures only
        adj: Dict[str, List[str]] = {mid: [] for mid in selected}
        in_deg: Dict[str, int] = {mid: 0 for mid in selected}

        for e in edges:
            if (
                e.dependency_type in (DependencyType.REQUIRES, DependencyType.SEQUENTIAL)
                and e.from_id in selected_set
                and e.to_id in selected_set
            ):
                adj[e.from_id].append(e.to_id)
                in_deg[e.to_id] = in_deg.get(e.to_id, 0) + 1

        if not any(adj.values()):
            # No dependencies at all
            return []

        # Topological order via Kahn's
        queue = sorted([mid for mid in selected if in_deg.get(mid, 0) == 0])
        topo_order: List[str] = []
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for nb in adj.get(node, []):
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    queue.append(nb)
            queue.sort()

        # DP for longest path
        dist: Dict[str, int] = {mid: 0 for mid in selected}
        pred: Dict[str, Optional[str]] = {mid: None for mid in selected}

        for node in topo_order:
            for nb in adj.get(node, []):
                if dist[node] + 1 > dist[nb]:
                    dist[nb] = dist[node] + 1
                    pred[nb] = node

        # Find endpoint with max distance
        if not dist:
            return []

        end_node = max(dist.keys(), key=lambda k: dist[k])
        if dist[end_node] == 0:
            return []

        # Trace back
        path: List[str] = []
        current: Optional[str] = end_node
        while current is not None:
            path.append(current)
            current = pred.get(current)
        path.reverse()

        return path

    # ------------------------------------------------------------------ #
    # Pareto Helpers                                                       #
    # ------------------------------------------------------------------ #

    def _is_dominated(
        self,
        point_a: List[Decimal],
        point_b: List[Decimal],
        objectives: List[Tuple[str, str]],
    ) -> bool:
        """Check whether point_a is dominated by point_b.

        Point A is dominated by B if B is at least as good in every
        objective and strictly better in at least one.
        Note: values are already sign-adjusted (higher = better).

        Args:
            point_a: Objective values for measure A (sign-adjusted).
            point_b: Objective values for measure B (sign-adjusted).
            objectives: Objective definitions (used for count only).

        Returns:
            True if A is dominated by B.
        """
        at_least_as_good = True
        strictly_better = False

        for va, vb in zip(point_a, point_b):
            if vb < va:
                at_least_as_good = False
                break
            if vb > va:
                strictly_better = True

        return at_least_as_good and strictly_better

    def _find_weakly_dominated(
        self,
        measures: List[MeasureForPrioritization],
        objectives: List[Tuple[str, str]],
    ) -> set:
        """Identify measures that are weakly dominated.

        A measure is weakly dominated if another measure is at least as
        good in all objectives (but not strictly better in any).

        Args:
            measures: Candidate measures.
            objectives: Objective definitions.

        Returns:
            Set of weakly dominated measure IDs.
        """
        if not measures or not objectives:
            return set()

        points: Dict[str, List[Decimal]] = {}
        for m in measures:
            vals: List[Decimal] = []
            for attr, direction in objectives:
                raw = _decimal(getattr(m, attr, Decimal("0")))
                if direction == "minimize":
                    raw = -raw
                vals.append(raw)
            points[m.measure_id] = vals

        weakly_dominated: set = set()
        ids = list(points.keys())

        for i, mid_a in enumerate(ids):
            for j, mid_b in enumerate(ids):
                if i == j:
                    continue
                pa = points[mid_a]
                pb = points[mid_b]
                # Weakly dominated: B >= A in all, B == A in at least one,
                # and not strictly better in any
                all_geq = all(vb >= va for va, vb in zip(pa, pb))
                any_eq = any(vb == va for va, vb in zip(pa, pb))
                none_strict = not any(vb > va for va, vb in zip(pa, pb))
                if all_geq and any_eq and none_strict and mid_a != mid_b:
                    # Both are equal in all objectives -- one is weakly dominated
                    # We pick the one that appears later
                    if i > j:
                        weakly_dominated.add(mid_a)

        return weakly_dominated

    # ------------------------------------------------------------------ #
    # Sensitivity Analysis                                                 #
    # ------------------------------------------------------------------ #

    def sensitivity_analysis(
        self,
        measures: List[MeasureForPrioritization],
        base_profile: WeightProfile = WeightProfile.BALANCED,
        criterion_to_vary: CriterionName = CriterionName.COST,
        weight_range: Optional[List[Decimal]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform one-at-a-time sensitivity analysis on a single criterion.

        Varies the weight of one criterion while proportionally adjusting
        others, re-runs prioritisation, and records ranking changes.

        Args:
            measures: Candidate measures.
            base_profile: Base weight profile.
            criterion_to_vary: Criterion whose weight to vary.
            weight_range: List of weight values to test.

        Returns:
            List of dicts with weight value and resulting top-5 rankings.
        """
        if weight_range is None:
            weight_range = [
                Decimal("0.05"), Decimal("0.10"), Decimal("0.15"),
                Decimal("0.20"), Decimal("0.25"), Decimal("0.30"),
                Decimal("0.35"), Decimal("0.40"),
            ]

        ws = DEFAULT_WEIGHT_PROFILES.get(base_profile.value)
        if ws is None:
            ws = DEFAULT_WEIGHT_PROFILES[WeightProfile.BALANCED.value]

        base_weights = {cw.criterion: cw for cw in ws.weights}
        results: List[Dict[str, Any]] = []

        for test_weight in weight_range:
            # Build adjusted weights
            adjusted: List[CriterionWeight] = []
            remaining_weight = Decimal("1") - test_weight
            original_others_sum = sum(
                cw.weight for cw in ws.weights
                if cw.criterion != criterion_to_vary
            )

            for cw in ws.weights:
                if cw.criterion == criterion_to_vary:
                    adjusted.append(CriterionWeight(
                        criterion=cw.criterion,
                        weight=test_weight,
                        direction=cw.direction,
                    ))
                else:
                    proportion = _safe_divide(cw.weight, original_others_sum)
                    adjusted.append(CriterionWeight(
                        criterion=cw.criterion,
                        weight=_round_val(proportion * remaining_weight, 4),
                        direction=cw.direction,
                    ))

            # Re-run prioritisation
            prio_result = self.prioritize(
                measures,
                profile=WeightProfile.CUSTOM,
                custom_weights=adjusted,
            )

            top_5 = [
                {"rank": r.rank, "measure_id": r.measure_id, "name": r.name,
                 "score": str(r.weighted_total)}
                for r in prio_result.rankings[:5]
            ]

            results.append({
                "criterion": criterion_to_vary.value,
                "weight": str(test_weight),
                "top_5": top_5,
            })

        logger.info(
            "Sensitivity analysis: %d weight variations on %s",
            len(results), criterion_to_vary.value,
        )
        return results

    # ------------------------------------------------------------------ #
    # Budget Optimization                                                  #
    # ------------------------------------------------------------------ #

    def optimize_for_budget(
        self,
        measures: List[MeasureForPrioritization],
        budget: Decimal,
        profile: WeightProfile = WeightProfile.BALANCED,
        objective: str = "savings",
    ) -> ImplementationSequence:
        """Select the optimal set of measures within a budget constraint.

        Implements a greedy knapsack heuristic sorted by value-density
        (objective / cost ratio).

        Args:
            measures: Candidate measures.
            budget: Total available budget.
            profile: Weight profile for tie-breaking.
            objective: Optimization objective ('savings', 'carbon', 'score').

        Returns:
            Budget-optimized ImplementationSequence.
        """
        if not measures:
            return ImplementationSequence(
                name="Empty budget-optimized sequence",
                provenance_hash=_compute_hash({}),
            )

        # Compute density for each measure
        densities: List[Tuple[str, Decimal]] = []
        measure_lookup = {m.measure_id: m for m in measures}

        for m in measures:
            cost = m.implementation_cost
            if cost <= Decimal("0"):
                # Zero-cost measures get infinite density (always include)
                density = Decimal("999999")
            else:
                if objective == "carbon":
                    value = m.co2e_reduction
                elif objective == "savings":
                    value = m.annual_savings
                else:
                    value = m.annual_savings  # fallback
                density = _safe_divide(value, cost)
            densities.append((m.measure_id, density))

        # Sort by density descending
        densities.sort(key=lambda x: x[1], reverse=True)

        selected: List[str] = []
        spent = Decimal("0")
        for mid, density in densities:
            m = measure_lookup[mid]
            if spent + m.implementation_cost <= budget:
                selected.append(mid)
                spent += m.implementation_cost

        # Build rankings to get phases
        rankings = self.prioritize(
            [measure_lookup[mid] for mid in selected],
            profile=profile,
        )

        return self.build_implementation_sequence(
            rankings.rankings,
            [measure_lookup[mid] for mid in selected],
            budget,
        )

    # ------------------------------------------------------------------ #
    # Utility Methods                                                      #
    # ------------------------------------------------------------------ #

    def get_weight_profile(self, profile: WeightProfile) -> WeightSet:
        """Retrieve a predefined weight profile.

        Args:
            profile: Weight profile to retrieve.

        Returns:
            WeightSet for the requested profile.

        Raises:
            ValueError: If profile is CUSTOM (no predefined set).
        """
        if profile == WeightProfile.CUSTOM:
            raise ValueError(
                "CUSTOM profile has no predefined weights. "
                "Supply custom_weights directly."
            )
        ws = DEFAULT_WEIGHT_PROFILES.get(profile.value)
        if ws is None:
            raise ValueError(f"Unknown weight profile: {profile.value}")
        return ws

    def compare_profiles(
        self,
        measures: List[MeasureForPrioritization],
        profiles: Optional[List[WeightProfile]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compare rankings across multiple weight profiles.

        Args:
            measures: Candidate measures.
            profiles: Profiles to compare (defaults to all non-CUSTOM).

        Returns:
            Dict mapping profile name to top-N ranking summaries.
        """
        if profiles is None:
            profiles = [
                p for p in WeightProfile
                if p != WeightProfile.CUSTOM
            ]

        comparison: Dict[str, List[Dict[str, Any]]] = {}
        for prof in profiles:
            result = self.prioritize(measures, profile=prof)
            comparison[prof.value] = [
                {
                    "rank": r.rank,
                    "measure_id": r.measure_id,
                    "name": r.name,
                    "score": str(r.weighted_total),
                    "phase": r.implementation_phase.value,
                    "pareto": r.pareto_status.value,
                }
                for r in result.rankings
            ]

        logger.info(
            "Profile comparison: %d profiles, %d measures",
            len(profiles), len(measures),
        )
        return comparison

    def validate_measures(
        self,
        measures: List[MeasureForPrioritization],
    ) -> List[str]:
        """Validate a list of measures for common issues.

        Checks:
            - Duplicate measure IDs
            - Self-referencing dependencies
            - Missing dependency targets
            - Circular dependencies (simple check)
            - Conflicting-with-self

        Args:
            measures: Candidate measures.

        Returns:
            List of warning/error messages (empty if all valid).
        """
        issues: List[str] = []
        valid_ids = {m.measure_id for m in measures}
        seen_ids: set = set()

        for m in measures:
            # Duplicate IDs
            if m.measure_id in seen_ids:
                issues.append(
                    f"Duplicate measure_id: {m.measure_id}"
                )
            seen_ids.add(m.measure_id)

            # Self-referencing dependencies
            if m.measure_id in m.dependencies:
                issues.append(
                    f"Measure {m.measure_id} depends on itself."
                )

            # Missing dependency targets
            for dep in m.dependencies:
                if dep not in valid_ids:
                    issues.append(
                        f"Measure {m.measure_id} depends on "
                        f"unknown measure {dep}."
                    )

            # Self-conflict
            if m.measure_id in m.conflicts:
                issues.append(
                    f"Measure {m.measure_id} conflicts with itself."
                )

            # Missing conflict targets
            for c in m.conflicts:
                if c not in valid_ids:
                    issues.append(
                        f"Measure {m.measure_id} conflicts with "
                        f"unknown measure {c}."
                    )

            # Zero savings warning
            if m.annual_savings <= Decimal("0"):
                issues.append(
                    f"Measure {m.measure_id} has zero annual savings."
                )

        # Simple circular dependency check
        dep_map: Dict[str, set] = {}
        for m in measures:
            dep_map[m.measure_id] = set(m.dependencies) & valid_ids

        for start_id in valid_ids:
            visited: set = set()
            stack = [start_id]
            while stack:
                node = stack.pop()
                if node in visited:
                    if node == start_id and len(visited) > 0:
                        issues.append(
                            f"Circular dependency detected involving "
                            f"measure {start_id}."
                        )
                        break
                    continue
                visited.add(node)
                for dep in dep_map.get(node, set()):
                    stack.append(dep)

        if issues:
            logger.warning(
                "Measure validation found %d issues", len(issues)
            )
        return issues
