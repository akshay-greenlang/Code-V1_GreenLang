# -*- coding: utf-8 -*-
"""
ProgressTrackerEngine - PACK-029 Interim Targets Pack Engine 3
================================================================

Compares actual emissions against interim targets to assess progress
towards net-zero goals.  Provides annual and quarterly comparisons,
variance calculation (absolute tCO2e, percentage, and trajectory
deviation), red/amber/green performance scoring, milestone achievement
tracking, and progress rate analysis.

Calculation Methodology:
    Absolute Variance:
        variance_tco2e = actual_tco2e - target_tco2e
        positive = over target (bad), negative = under target (good)

    Percentage Variance:
        variance_pct = (actual - target) / target * 100

    Trajectory Deviation:
        deviation = actual_annual_rate - required_annual_rate
        positive = ahead of schedule, negative = behind

    Performance Scoring (RAG):
        GREEN:  variance_pct <= -5% (ahead of target)
        AMBER:  -5% < variance_pct <= +10% (close to target)
        RED:    variance_pct > +10% (behind target)

    Progress Rate:
        actual_reduction_rate = 1 - (current / baseline)^(1/years)
        required_reduction_rate = 1 - (target / baseline)^(1/remaining)
        gap = required - actual

    Milestone Achievement:
        achieved = actual_reduction_pct >= milestone_reduction_pct

Regulatory References:
    - SBTi Target Tracking Protocol v2.0
    - SBTi Corporate Net-Zero Standard v1.2 (2024) -- progress reporting
    - GHG Protocol Corporate Standard Ch. 5 -- tracking over time
    - CDP Climate Change C4.1 -- targets, C4.2 -- progress
    - CSRD ESRS E1-4 -- GHG reduction targets
    - TCFD Recommendations -- metrics & targets

Zero-Hallucination:
    - All variance calculations use deterministic Decimal arithmetic
    - RAG thresholds hard-coded per SBTi guidance
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-029 Interim Targets
Engine:  3 of 10
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

def _safe_divide(n: Decimal, d: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    if d == Decimal("0"):
        return default
    return n / d

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    q = "0." + "0" * places
    return value.quantize(Decimal(q), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RAGStatus(str, Enum):
    """Red/Amber/Green performance status."""
    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    NOT_ASSESSED = "not_assessed"

class ProgressDirection(str, Enum):
    """Progress direction relative to target."""
    AHEAD = "ahead"
    ON_TRACK = "on_track"
    SLIGHTLY_BEHIND = "slightly_behind"
    BEHIND = "behind"
    SIGNIFICANTLY_BEHIND = "significantly_behind"
    NOT_STARTED = "not_started"

class MilestoneStatus(str, Enum):
    """Achievement status for a milestone."""
    ACHIEVED = "achieved"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    MISSED = "missed"
    UPCOMING = "upcoming"
    NOT_DUE = "not_due"

class ScopeType(str, Enum):
    """GHG scope type."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"
    ALL_SCOPES = "all_scopes"

class DataQuality(str, Enum):
    """Data quality tier."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"

# ---------------------------------------------------------------------------
# Constants -- RAG Thresholds
# ---------------------------------------------------------------------------

# Performance scoring thresholds (variance_pct from target)
RAG_THRESHOLDS: Dict[str, Dict[str, Decimal]] = {
    "default": {
        "green_upper": Decimal("-5"),      # <= -5% variance (ahead)
        "amber_upper": Decimal("10"),      # <= +10% variance (close)
        # > +10% = red
    },
    "strict": {
        "green_upper": Decimal("-2"),
        "amber_upper": Decimal("5"),
    },
    "relaxed": {
        "green_upper": Decimal("-10"),
        "amber_upper": Decimal("20"),
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class ActualEmissionsPoint(BaseModel):
    """Actual emissions data for a period.

    Attributes:
        year: Calendar year.
        quarter: Quarter (1-4, or 0 for annual).
        emissions_tco2e: Actual emissions (tCO2e).
        scope: Which scope this covers.
        is_verified: Third-party verified.
        data_source: Source of the data.
    """
    year: int = Field(..., ge=2015, le=2050, description="Year")
    quarter: int = Field(default=0, ge=0, le=4, description="Quarter (0=annual)")
    emissions_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Actual emissions (tCO2e)"
    )
    scope: ScopeType = Field(default=ScopeType.ALL_SCOPES, description="Scope")
    is_verified: bool = Field(default=False, description="Verified")
    data_source: str = Field(default="", max_length=200, description="Data source")

class TargetPoint(BaseModel):
    """Target emissions for a period.

    Attributes:
        year: Calendar year.
        quarter: Quarter (1-4, or 0 for annual).
        target_tco2e: Target emissions (tCO2e).
        scope: Which scope.
        milestone_name: Name of milestone (e.g., "2030 Near-Term").
        is_sbti_target: Whether this is an SBTi-validated target.
    """
    year: int = Field(..., ge=2015, le=2070, description="Year")
    quarter: int = Field(default=0, ge=0, le=4, description="Quarter")
    target_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Target emissions (tCO2e)"
    )
    scope: ScopeType = Field(default=ScopeType.ALL_SCOPES, description="Scope")
    milestone_name: str = Field(default="", max_length=200, description="Milestone name")
    is_sbti_target: bool = Field(default=False, description="SBTi target")

class ProgressTrackerInput(BaseModel):
    """Input for progress tracking.

    Attributes:
        entity_name: Company name.
        entity_id: Entity identifier.
        baseline_year: Baseline year.
        baseline_tco2e: Baseline emissions.
        actual_emissions: Actual emissions data points.
        targets: Target data points.
        net_zero_year: Net-zero target year.
        rag_threshold_mode: RAG threshold mode (default/strict/relaxed).
        scope: Primary scope to track.
        reporting_year: Current reporting year.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    entity_id: str = Field(default="", max_length=100, description="Entity identifier")
    baseline_year: int = Field(..., ge=2015, le=2025, description="Baseline year")
    baseline_tco2e: Decimal = Field(
        ..., ge=Decimal("0"), description="Baseline emissions (tCO2e)"
    )
    actual_emissions: List[ActualEmissionsPoint] = Field(
        ..., min_length=1, description="Actual emissions"
    )
    targets: List[TargetPoint] = Field(
        ..., min_length=1, description="Target points"
    )
    net_zero_year: int = Field(default=2050, ge=2030, le=2070, description="Net-zero year")
    rag_threshold_mode: str = Field(
        default="default", description="RAG threshold mode"
    )
    scope: ScopeType = Field(default=ScopeType.ALL_SCOPES, description="Scope")
    reporting_year: int = Field(default=2024, ge=2020, le=2030, description="Reporting year")

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class VariancePoint(BaseModel):
    """Variance analysis for a single period.

    Attributes:
        year: Calendar year.
        quarter: Quarter (0=annual).
        scope: Scope.
        actual_tco2e: Actual emissions.
        target_tco2e: Target emissions.
        variance_tco2e: Absolute variance (actual - target).
        variance_pct: Percentage variance.
        reduction_from_baseline_actual_pct: Actual reduction from baseline.
        reduction_from_baseline_target_pct: Target reduction from baseline.
        rag_status: RAG status.
        progress_direction: Progress direction.
        trajectory_deviation_pct: Deviation from required trajectory.
    """
    year: int = Field(default=0)
    quarter: int = Field(default=0)
    scope: str = Field(default="")
    actual_tco2e: Decimal = Field(default=Decimal("0"))
    target_tco2e: Decimal = Field(default=Decimal("0"))
    variance_tco2e: Decimal = Field(default=Decimal("0"))
    variance_pct: Decimal = Field(default=Decimal("0"))
    reduction_from_baseline_actual_pct: Decimal = Field(default=Decimal("0"))
    reduction_from_baseline_target_pct: Decimal = Field(default=Decimal("0"))
    rag_status: str = Field(default=RAGStatus.NOT_ASSESSED.value)
    progress_direction: str = Field(default=ProgressDirection.NOT_STARTED.value)
    trajectory_deviation_pct: Decimal = Field(default=Decimal("0"))

class MilestoneAssessment(BaseModel):
    """Assessment of a specific milestone.

    Attributes:
        milestone_name: Milestone name.
        year: Milestone year.
        target_reduction_pct: Target reduction %.
        actual_reduction_pct: Actual reduction % (if data available).
        status: Achievement status.
        gap_pct: Gap to milestone (positive = behind).
        projected_achievement_year: When milestone would be achieved at current rate.
        notes: Assessment notes.
    """
    milestone_name: str = Field(default="")
    year: int = Field(default=0)
    target_reduction_pct: Decimal = Field(default=Decimal("0"))
    actual_reduction_pct: Decimal = Field(default=Decimal("0"))
    status: str = Field(default=MilestoneStatus.NOT_DUE.value)
    gap_pct: Decimal = Field(default=Decimal("0"))
    projected_achievement_year: int = Field(default=0)
    notes: List[str] = Field(default_factory=list)

class ProgressRateAnalysis(BaseModel):
    """Analysis of progress rate vs required rate.

    Attributes:
        actual_annual_rate_pct: Actual annual reduction rate.
        required_annual_rate_pct: Required annual rate to hit target.
        rate_gap_pct: Gap between required and actual.
        years_of_data: Number of years of actual data.
        trend_direction: Improving/stable/declining.
        years_to_catch_up: Years to return to target trajectory.
        projected_target_miss_tco2e: Projected miss at target year.
        projected_target_miss_pct: Projected miss as percentage.
    """
    actual_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    required_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    rate_gap_pct: Decimal = Field(default=Decimal("0"))
    years_of_data: int = Field(default=0)
    trend_direction: str = Field(default="stable")
    years_to_catch_up: int = Field(default=0)
    projected_target_miss_tco2e: Decimal = Field(default=Decimal("0"))
    projected_target_miss_pct: Decimal = Field(default=Decimal("0"))

class OverallAssessment(BaseModel):
    """Overall progress assessment.

    Attributes:
        rag_status: Overall RAG status.
        progress_direction: Overall direction.
        total_periods_assessed: Periods assessed.
        green_count: Number of green periods.
        amber_count: Number of amber periods.
        red_count: Number of red periods.
        best_year: Year with best performance.
        worst_year: Year with worst performance.
        consecutive_improvements: Consecutive years of improvement.
        on_track_for_2030: Whether on track for 2030 target.
        on_track_for_net_zero: Whether on track for net-zero.
    """
    rag_status: str = Field(default=RAGStatus.NOT_ASSESSED.value)
    progress_direction: str = Field(default=ProgressDirection.NOT_STARTED.value)
    total_periods_assessed: int = Field(default=0)
    green_count: int = Field(default=0)
    amber_count: int = Field(default=0)
    red_count: int = Field(default=0)
    best_year: int = Field(default=0)
    worst_year: int = Field(default=0)
    consecutive_improvements: int = Field(default=0)
    on_track_for_2030: bool = Field(default=False)
    on_track_for_net_zero: bool = Field(default=False)

class ProgressTrackerResult(BaseModel):
    """Complete progress tracking result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        entity_id: Entity identifier.
        baseline_year: Baseline year.
        baseline_tco2e: Baseline emissions.
        net_zero_year: Net-zero year.
        scope: Scope tracked.
        variance_points: Per-period variance analysis.
        milestone_assessments: Milestone achievement analysis.
        progress_rate: Progress rate analysis.
        overall_assessment: Overall assessment.
        data_quality: Data quality.
        recommendations: Recommendations.
        warnings: Warnings.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    entity_id: str = Field(default="")
    baseline_year: int = Field(default=0)
    baseline_tco2e: Decimal = Field(default=Decimal("0"))
    net_zero_year: int = Field(default=0)
    scope: str = Field(default="")
    variance_points: List[VariancePoint] = Field(default_factory=list)
    milestone_assessments: List[MilestoneAssessment] = Field(default_factory=list)
    progress_rate: Optional[ProgressRateAnalysis] = Field(default=None)
    overall_assessment: Optional[OverallAssessment] = Field(default=None)
    data_quality: str = Field(default=DataQuality.MEDIUM.value)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ProgressTrackerEngine:
    """Progress tracking engine for PACK-029.

    Compares actual vs target emissions with RAG scoring, milestone
    tracking, and progress rate analysis.

    Usage::

        engine = ProgressTrackerEngine()
        result = await engine.calculate(progress_input)
        print(f"Overall: {result.overall_assessment.rag_status}")
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def calculate(self, data: ProgressTrackerInput) -> ProgressTrackerResult:
        """Run complete progress tracking analysis.

        Args:
            data: Validated progress tracking input.

        Returns:
            ProgressTrackerResult with variance, milestones, and assessment.
        """
        t0 = time.perf_counter()
        logger.info(
            "Progress tracking: entity=%s, actuals=%d, targets=%d",
            data.entity_name, len(data.actual_emissions), len(data.targets),
        )

        # Get RAG thresholds
        thresholds = RAG_THRESHOLDS.get(
            data.rag_threshold_mode, RAG_THRESHOLDS["default"]
        )

        # Calculate variance points
        variance_points = self._calculate_variances(data, thresholds)

        # Assess milestones
        milestones = self._assess_milestones(data, variance_points)

        # Progress rate analysis
        progress_rate = self._analyze_progress_rate(data)

        # Overall assessment
        overall = self._build_overall_assessment(
            data, variance_points, milestones, progress_rate,
        )

        # Data quality
        dq = self._assess_data_quality(data)

        # Recommendations
        recs = self._generate_recommendations(data, overall, progress_rate)

        # Warnings
        warns = self._generate_warnings(data, variance_points, overall)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ProgressTrackerResult(
            entity_name=data.entity_name,
            entity_id=data.entity_id,
            baseline_year=data.baseline_year,
            baseline_tco2e=data.baseline_tco2e,
            net_zero_year=data.net_zero_year,
            scope=data.scope.value,
            variance_points=variance_points,
            milestone_assessments=milestones,
            progress_rate=progress_rate,
            overall_assessment=overall,
            data_quality=dq,
            recommendations=recs,
            warnings=warns,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Progress tracking complete: entity=%s, rag=%s, direction=%s",
            data.entity_name,
            overall.rag_status if overall else "n/a",
            overall.progress_direction if overall else "n/a",
        )
        return result

    async def calculate_batch(
        self, inputs: List[ProgressTrackerInput],
    ) -> List[ProgressTrackerResult]:
        """Track progress for multiple entities."""
        results: List[ProgressTrackerResult] = []
        for inp in inputs:
            try:
                results.append(await self.calculate(inp))
            except Exception as exc:
                logger.error("Batch error for %s: %s", inp.entity_name, exc)
                results.append(ProgressTrackerResult(
                    entity_name=inp.entity_name,
                    warnings=[f"Calculation error: {exc}"],
                ))
        return results

    # ------------------------------------------------------------------ #
    # Variance Calculation                                                 #
    # ------------------------------------------------------------------ #

    def _calculate_variances(
        self,
        data: ProgressTrackerInput,
        thresholds: Dict[str, Decimal],
    ) -> List[VariancePoint]:
        """Calculate variance for each actual-vs-target pair.

        Matches actual data points with corresponding targets by
        year and quarter, then computes variance metrics.
        """
        points: List[VariancePoint] = []

        # Build target lookup
        target_map: Dict[Tuple[int, int], TargetPoint] = {}
        for t in data.targets:
            target_map[(t.year, t.quarter)] = t

        for actual in data.actual_emissions:
            key = (actual.year, actual.quarter)
            target = target_map.get(key)

            if target is None:
                # Try annual match if quarterly data
                if actual.quarter > 0:
                    target = target_map.get((actual.year, 0))
                if target is None:
                    continue

            target_tco2e = target.target_tco2e
            if actual.quarter > 0 and target.quarter == 0:
                # Quarterly actual vs annual target: use quarterly portion
                target_tco2e = target_tco2e / Decimal("4")

            variance = actual.emissions_tco2e - target_tco2e
            variance_pct = _safe_pct(variance, target_tco2e) if target_tco2e > Decimal("0") else Decimal("0")

            actual_red = _safe_pct(
                data.baseline_tco2e - actual.emissions_tco2e,
                data.baseline_tco2e,
            )
            target_red = _safe_pct(
                data.baseline_tco2e - target_tco2e,
                data.baseline_tco2e,
            )

            rag = self._assess_rag(variance_pct, thresholds)
            direction = self._assess_direction(variance_pct)
            traj_dev = actual_red - target_red

            points.append(VariancePoint(
                year=actual.year,
                quarter=actual.quarter,
                scope=actual.scope.value,
                actual_tco2e=_round_val(actual.emissions_tco2e, 2),
                target_tco2e=_round_val(target_tco2e, 2),
                variance_tco2e=_round_val(variance, 2),
                variance_pct=_round_val(variance_pct, 2),
                reduction_from_baseline_actual_pct=_round_val(actual_red, 2),
                reduction_from_baseline_target_pct=_round_val(target_red, 2),
                rag_status=rag,
                progress_direction=direction,
                trajectory_deviation_pct=_round_val(traj_dev, 2),
            ))

        points.sort(key=lambda p: (p.year, p.quarter))
        return points

    def _assess_rag(
        self, variance_pct: Decimal, thresholds: Dict[str, Decimal],
    ) -> str:
        """Assess RAG status from variance percentage.

        GREEN: variance <= green_upper (ahead)
        AMBER: green_upper < variance <= amber_upper
        RED:   variance > amber_upper (behind)
        """
        if variance_pct <= thresholds["green_upper"]:
            return RAGStatus.GREEN.value
        elif variance_pct <= thresholds["amber_upper"]:
            return RAGStatus.AMBER.value
        return RAGStatus.RED.value

    def _assess_direction(self, variance_pct: Decimal) -> str:
        """Assess progress direction from variance percentage."""
        if variance_pct <= Decimal("-10"):
            return ProgressDirection.AHEAD.value
        elif variance_pct <= Decimal("0"):
            return ProgressDirection.ON_TRACK.value
        elif variance_pct <= Decimal("10"):
            return ProgressDirection.SLIGHTLY_BEHIND.value
        elif variance_pct <= Decimal("25"):
            return ProgressDirection.BEHIND.value
        return ProgressDirection.SIGNIFICANTLY_BEHIND.value

    # ------------------------------------------------------------------ #
    # Milestone Assessment                                                 #
    # ------------------------------------------------------------------ #

    def _assess_milestones(
        self,
        data: ProgressTrackerInput,
        variance_points: List[VariancePoint],
    ) -> List[MilestoneAssessment]:
        """Assess achievement of each milestone target."""
        assessments: List[MilestoneAssessment] = []

        # Build actual emissions by year
        actual_by_year: Dict[int, Decimal] = {}
        for ae in data.actual_emissions:
            if ae.quarter == 0:
                actual_by_year[ae.year] = ae.emissions_tco2e

        for target in data.targets:
            if target.quarter != 0:
                continue  # Only assess annual milestones

            target_red_pct = _safe_pct(
                data.baseline_tco2e - target.target_tco2e,
                data.baseline_tco2e,
            )

            actual_tco2e = actual_by_year.get(target.year)
            actual_red_pct = Decimal("0")
            status = MilestoneStatus.NOT_DUE.value
            gap = Decimal("0")
            projected_year = 0

            if actual_tco2e is not None:
                actual_red_pct = _safe_pct(
                    data.baseline_tco2e - actual_tco2e,
                    data.baseline_tco2e,
                )
                gap = target_red_pct - actual_red_pct

                if actual_red_pct >= target_red_pct:
                    status = MilestoneStatus.ACHIEVED.value
                elif gap <= Decimal("5"):
                    status = MilestoneStatus.ON_TRACK.value
                elif gap <= Decimal("15"):
                    status = MilestoneStatus.AT_RISK.value
                else:
                    status = MilestoneStatus.MISSED.value

                # Project when milestone would be achieved
                if actual_red_pct > Decimal("0") and gap > Decimal("0"):
                    years_elapsed = target.year - data.baseline_year
                    if years_elapsed > 0:
                        rate = float(actual_red_pct) / years_elapsed
                        if rate > 0:
                            projected_year = data.baseline_year + int(
                                float(target_red_pct) / rate
                            )
            elif target.year <= data.reporting_year:
                status = MilestoneStatus.MISSED.value
            elif target.year <= data.reporting_year + 2:
                status = MilestoneStatus.UPCOMING.value

            notes: List[str] = []
            if status == MilestoneStatus.ACHIEVED.value:
                notes.append(f"Target achieved: {_round_val(actual_red_pct, 1)}% "
                           f">= {_round_val(target_red_pct, 1)}% target.")
            elif status == MilestoneStatus.AT_RISK.value:
                notes.append(f"Gap of {_round_val(gap, 1)}% to target. "
                           f"Accelerated action needed.")
            elif status == MilestoneStatus.MISSED.value and actual_tco2e is not None:
                notes.append(f"Missed by {_round_val(gap, 1)}%. "
                           f"Corrective action required.")

            assessments.append(MilestoneAssessment(
                milestone_name=target.milestone_name or f"Target {target.year}",
                year=target.year,
                target_reduction_pct=_round_val(target_red_pct, 2),
                actual_reduction_pct=_round_val(actual_red_pct, 2),
                status=status,
                gap_pct=_round_val(gap, 2),
                projected_achievement_year=projected_year,
                notes=notes,
            ))

        return assessments

    # ------------------------------------------------------------------ #
    # Progress Rate Analysis                                               #
    # ------------------------------------------------------------------ #

    def _analyze_progress_rate(
        self, data: ProgressTrackerInput,
    ) -> ProgressRateAnalysis:
        """Analyze actual vs required reduction rate.

        Formulas:
            actual_rate = 1 - (latest / baseline)^(1 / years_elapsed)
            required_rate = 1 - (target / baseline)^(1 / years_remaining)
        """
        # Get annual actual emissions sorted
        annuals = sorted(
            [ae for ae in data.actual_emissions if ae.quarter == 0],
            key=lambda a: a.year,
        )

        if not annuals:
            return ProgressRateAnalysis()

        latest = annuals[-1]
        years_elapsed = latest.year - data.baseline_year

        # Actual annual rate
        actual_rate = Decimal("0")
        if years_elapsed > 0 and data.baseline_tco2e > Decimal("0"):
            ratio = float(_safe_divide(latest.emissions_tco2e, data.baseline_tco2e))
            if ratio > 0:
                try:
                    actual_rate = _decimal(
                        (1.0 - ratio ** (1.0 / years_elapsed)) * 100
                    )
                except (OverflowError, ValueError):
                    pass

        # Required rate from latest to net-zero
        years_remaining = data.net_zero_year - latest.year
        required_rate = Decimal("0")

        # Find the final target
        final_target = min(
            (t for t in data.targets if t.quarter == 0),
            key=lambda t: t.target_tco2e,
            default=None,
        )

        if final_target and years_remaining > 0 and latest.emissions_tco2e > Decimal("0"):
            ratio = float(_safe_divide(final_target.target_tco2e, latest.emissions_tco2e))
            if ratio > 0:
                try:
                    required_rate = _decimal(
                        (1.0 - ratio ** (1.0 / years_remaining)) * 100
                    )
                except (OverflowError, ValueError):
                    pass

        rate_gap = required_rate - actual_rate

        # Trend direction
        trend = "stable"
        if len(annuals) >= 3:
            recent_changes = []
            for i in range(1, len(annuals)):
                change = annuals[i].emissions_tco2e - annuals[i - 1].emissions_tco2e
                recent_changes.append(change)
            avg_change = sum(recent_changes) / _decimal(len(recent_changes))
            if avg_change < Decimal("-100"):
                trend = "improving"
            elif avg_change > Decimal("100"):
                trend = "declining"

        # Years to catch up
        catch_up = 0
        if rate_gap > Decimal("0") and actual_rate > Decimal("0"):
            catch_up = min(int(float(rate_gap / actual_rate) * years_remaining), 99)

        # Projected miss
        projected_miss = Decimal("0")
        projected_miss_pct = Decimal("0")
        if final_target and actual_rate > Decimal("0") and years_remaining > 0:
            try:
                projected_at_target = latest.emissions_tco2e * _decimal(
                    (1.0 - float(actual_rate) / 100.0) ** years_remaining
                )
                projected_miss = max(projected_at_target - final_target.target_tco2e, Decimal("0"))
                projected_miss_pct = _safe_pct(projected_miss, final_target.target_tco2e)
            except (OverflowError, ValueError):
                pass

        return ProgressRateAnalysis(
            actual_annual_rate_pct=_round_val(actual_rate, 3),
            required_annual_rate_pct=_round_val(required_rate, 3),
            rate_gap_pct=_round_val(rate_gap, 3),
            years_of_data=len(annuals),
            trend_direction=trend,
            years_to_catch_up=catch_up,
            projected_target_miss_tco2e=_round_val(projected_miss, 2),
            projected_target_miss_pct=_round_val(projected_miss_pct, 2),
        )

    # ------------------------------------------------------------------ #
    # Overall Assessment                                                   #
    # ------------------------------------------------------------------ #

    def _build_overall_assessment(
        self,
        data: ProgressTrackerInput,
        variances: List[VariancePoint],
        milestones: List[MilestoneAssessment],
        progress_rate: ProgressRateAnalysis,
    ) -> OverallAssessment:
        """Build overall progress assessment."""
        green = sum(1 for v in variances if v.rag_status == RAGStatus.GREEN.value)
        amber = sum(1 for v in variances if v.rag_status == RAGStatus.AMBER.value)
        red = sum(1 for v in variances if v.rag_status == RAGStatus.RED.value)
        total = len(variances)

        # Overall RAG
        if total == 0:
            overall_rag = RAGStatus.NOT_ASSESSED.value
        elif red > total * 0.3:
            overall_rag = RAGStatus.RED.value
        elif red + amber > total * 0.5:
            overall_rag = RAGStatus.AMBER.value
        else:
            overall_rag = RAGStatus.GREEN.value

        # Direction
        if total == 0:
            direction = ProgressDirection.NOT_STARTED.value
        elif progress_rate.rate_gap_pct <= Decimal("0"):
            direction = ProgressDirection.AHEAD.value
        elif progress_rate.rate_gap_pct <= Decimal("1"):
            direction = ProgressDirection.ON_TRACK.value
        elif progress_rate.rate_gap_pct <= Decimal("3"):
            direction = ProgressDirection.SLIGHTLY_BEHIND.value
        else:
            direction = ProgressDirection.BEHIND.value

        # Best/worst year
        best_year = 0
        worst_year = 0
        if variances:
            best = min(variances, key=lambda v: v.variance_pct)
            worst = max(variances, key=lambda v: v.variance_pct)
            best_year = best.year
            worst_year = worst.year

        # Consecutive improvements
        consec = 0
        sorted_vars = sorted(variances, key=lambda v: v.year)
        for i in range(len(sorted_vars) - 1, -1, -1):
            if sorted_vars[i].variance_pct <= Decimal("0"):
                consec += 1
            else:
                break

        # 2030 and net-zero track
        on_2030 = progress_rate.actual_annual_rate_pct >= Decimal("4.2")
        on_nz = progress_rate.rate_gap_pct <= Decimal("1")

        return OverallAssessment(
            rag_status=overall_rag,
            progress_direction=direction,
            total_periods_assessed=total,
            green_count=green,
            amber_count=amber,
            red_count=red,
            best_year=best_year,
            worst_year=worst_year,
            consecutive_improvements=consec,
            on_track_for_2030=on_2030,
            on_track_for_net_zero=on_nz,
        )

    # ------------------------------------------------------------------ #
    # Data Quality / Recommendations / Warnings                            #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(self, data: ProgressTrackerInput) -> str:
        score = 0
        annuals = [a for a in data.actual_emissions if a.quarter == 0]
        if len(annuals) >= 3:
            score += 3
        elif len(annuals) >= 1:
            score += 1
        verified = sum(1 for a in data.actual_emissions if a.is_verified)
        if verified >= len(data.actual_emissions) * 0.5:
            score += 2
        if len(data.targets) >= 3:
            score += 2
        if data.entity_id:
            score += 1
        if data.baseline_tco2e > Decimal("0"):
            score += 2

        if score >= 8:
            return DataQuality.HIGH.value
        elif score >= 5:
            return DataQuality.MEDIUM.value
        elif score >= 2:
            return DataQuality.LOW.value
        return DataQuality.ESTIMATED.value

    def _generate_recommendations(
        self, data: ProgressTrackerInput,
        overall: OverallAssessment, rate: ProgressRateAnalysis,
    ) -> List[str]:
        recs: List[str] = []
        if overall.rag_status == RAGStatus.RED.value:
            recs.append(
                "Overall performance is RED. Immediate corrective action "
                "required to return to target pathway."
            )
        if rate.rate_gap_pct > Decimal("2"):
            recs.append(
                f"Actual reduction rate ({rate.actual_annual_rate_pct}%/yr) "
                f"is {rate.rate_gap_pct}% below required rate "
                f"({rate.required_annual_rate_pct}%/yr). Accelerate reductions."
            )
        if rate.years_of_data < 3:
            recs.append(
                "Less than 3 years of actual data. Increase measurement "
                "frequency for more reliable trend analysis."
            )
        verified = sum(1 for a in data.actual_emissions if a.is_verified)
        if verified < len(data.actual_emissions):
            recs.append(
                f"Only {verified}/{len(data.actual_emissions)} data points are "
                f"third-party verified. Seek external assurance."
            )
        return recs

    def _generate_warnings(
        self, data: ProgressTrackerInput,
        variances: List[VariancePoint], overall: OverallAssessment,
    ) -> List[str]:
        warns: List[str] = []
        if data.baseline_tco2e <= Decimal("0"):
            warns.append("Baseline emissions are zero. Cannot assess progress.")
        red_pts = [v for v in variances if v.rag_status == RAGStatus.RED.value]
        if len(red_pts) >= 3:
            warns.append(
                f"{len(red_pts)} periods with RED status. "
                f"Systematic underperformance detected."
            )
        if overall.consecutive_improvements == 0 and len(variances) >= 2:
            warns.append(
                "No consecutive improvement periods detected. "
                "Review effectiveness of current reduction initiatives."
            )
        return warns

    def get_rag_thresholds(self) -> Dict[str, Dict[str, str]]:
        """Return available RAG threshold modes."""
        return {
            mode: {k: str(v) for k, v in thresholds.items()}
            for mode, thresholds in RAG_THRESHOLDS.items()
        }
