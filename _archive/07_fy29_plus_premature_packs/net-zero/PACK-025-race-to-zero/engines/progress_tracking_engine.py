# -*- coding: utf-8 -*-
"""
ProgressTrackingEngine - PACK-025 Race to Zero Engine 5
========================================================

Tracks annual progress against Race to Zero interim and long-term
targets. Calculates year-over-year emission changes, assesses
trajectory alignment with 2030 and 2050 targets, identifies
on-track/off-track status using RAG classification, generates
variance analysis, and triggers corrective action recommendations.

Calculation Methodology:
    Year-over-Year Change:
        yoy_change_pct = (current - prior) / prior * 100
        yoy_reduction = -yoy_change_pct  (positive = reduced)

    Cumulative Reduction:
        cumulative_pct = (baseline - current) / baseline * 100

    Trajectory Assessment:
        required_annual_rate = 1 - (target / baseline) ^ (1 / years_remaining)
        actual_annual_rate = 1 - (current / baseline) ^ (1 / years_elapsed)
        variance = actual_annual_rate - required_annual_rate

    Progress Status (RAG):
        ON_TRACK (GREEN):   cumulative ahead of linear trajectory
        CAUTION (AMBER):    within 10% of required trajectory
        OFF_TRACK (RED):    >10% behind required trajectory
        CRITICAL (BLACK):   no measurable progress or increasing

    Remaining Carbon Budget:
        required_total_reduction = baseline - target
        remaining_budget = target - current  (if current > target)
        years_remaining = target_year - current_year
        required_annual = remaining_budget / years_remaining

    Verification Status:
        VERIFIED:    Third-party verified emissions data
        SELF_REPORTED: Self-reported, not yet verified
        PENDING:     Verification in progress

Regulatory References:
    - Race to Zero Interpretation Guide (June 2022), SL-D1..SL-D5
    - HLEG "Integrity Matters" (November 2022), Rec 8
    - GHG Protocol Corporate Standard (2015)
    - ISAE 3410: Assurance on GHG statements
    - SBTi Progress Report Requirements

Zero-Hallucination:
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-025 Race to Zero
Engine:  5 of 10
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
# Enums
# ---------------------------------------------------------------------------

class ProgressStatus(str, Enum):
    """Progress tracking RAG status."""
    ON_TRACK = "on_track"
    CAUTION = "caution"
    OFF_TRACK = "off_track"
    CRITICAL = "critical"

class VerificationStatus(str, Enum):
    """Data verification status."""
    VERIFIED = "verified"
    SELF_REPORTED = "self_reported"
    PENDING = "pending"

class TrendDirection(str, Enum):
    """Emission trend direction."""
    DECREASING = "decreasing"
    STABLE = "stable"
    INCREASING = "increasing"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RAG thresholds (gap to trajectory as % of required reduction).
RAG_CAUTION_THRESHOLD: Decimal = Decimal("10")
RAG_OFF_TRACK_THRESHOLD: Decimal = Decimal("20")

# Variance thresholds.
VARIANCE_ON_TRACK: Decimal = Decimal("5")
VARIANCE_AT_RISK: Decimal = Decimal("15")

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class AnnualEmissionRecord(BaseModel):
    """Annual emission record for a single year.

    Attributes:
        year: Reporting year.
        scope1_tco2e: Scope 1 emissions.
        scope2_tco2e: Scope 2 emissions.
        scope3_tco2e: Scope 3 emissions.
        total_tco2e: Total emissions.
        verification_status: Data verification status.
        notes: Notes about this year's data.
    """
    year: int = Field(..., ge=2010, le=2060)
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    verification_status: str = Field(
        default=VerificationStatus.SELF_REPORTED.value
    )
    notes: str = Field(default="")

    @field_validator("verification_status")
    @classmethod
    def validate_verification(cls, v: str) -> str:
        valid = {s.value for s in VerificationStatus}
        if v not in valid:
            raise ValueError(f"Unknown verification status '{v}'.")
        return v

class ActionPlanStatusInput(BaseModel):
    """Status of action plan implementation.

    Attributes:
        total_actions: Total planned actions.
        actions_initiated: Actions initiated.
        actions_completed: Actions completed.
        actions_on_schedule: Actions on schedule.
        actions_delayed: Actions delayed.
        abatement_realized_tco2e: Realized abatement to date.
        abatement_planned_tco2e: Total planned abatement.
        budget_spent_usd: Budget spent to date.
        budget_total_usd: Total budget allocated.
    """
    total_actions: int = Field(default=0, ge=0)
    actions_initiated: int = Field(default=0, ge=0)
    actions_completed: int = Field(default=0, ge=0)
    actions_on_schedule: int = Field(default=0, ge=0)
    actions_delayed: int = Field(default=0, ge=0)
    abatement_realized_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    abatement_planned_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    budget_spent_usd: Decimal = Field(default=Decimal("0"), ge=0)
    budget_total_usd: Decimal = Field(default=Decimal("0"), ge=0)

class ProgressTrackingInput(BaseModel):
    """Complete input for progress tracking.

    Attributes:
        entity_name: Entity name.
        baseline_year: Baseline year.
        baseline_emissions_tco2e: Baseline emissions.
        interim_target_year: Interim target year (2030).
        interim_target_tco2e: Interim target emissions.
        net_zero_year: Net-zero target year (2050).
        current_year: Current reporting year.
        annual_records: Historical annual emission records.
        action_plan_status: Action plan implementation status.
        scope3_coverage_pct: Current Scope 3 coverage.
        include_projections: Whether to include forward projections.
        include_corrective_actions: Whether to generate corrective actions.
    """
    entity_name: str = Field(..., min_length=1, max_length=300)
    baseline_year: int = Field(..., ge=2010, le=2060)
    baseline_emissions_tco2e: Decimal = Field(..., ge=0)
    interim_target_year: int = Field(default=2030, ge=2025, le=2040)
    interim_target_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    net_zero_year: int = Field(default=2050, ge=2030, le=2060)
    current_year: int = Field(..., ge=2010, le=2060)
    annual_records: List[AnnualEmissionRecord] = Field(default_factory=list)
    action_plan_status: Optional[ActionPlanStatusInput] = Field(default=None)
    scope3_coverage_pct: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    include_projections: bool = Field(default=True)
    include_corrective_actions: bool = Field(default=True)

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class TrajectoryPoint(BaseModel):
    """A single point on the emission trajectory.

    Attributes:
        year: Year.
        actual_tco2e: Actual emissions (None if projected).
        target_tco2e: Linear target trajectory emissions.
        gap_tco2e: Gap (actual - target, positive = over).
        gap_pct: Gap as percentage of target.
        cumulative_reduction_pct: Cumulative reduction from baseline.
        status: RAG status for this point.
        is_projected: Whether this is a projection.
    """
    year: int = Field(default=0)
    actual_tco2e: Optional[Decimal] = Field(default=None)
    target_tco2e: Decimal = Field(default=Decimal("0"))
    gap_tco2e: Decimal = Field(default=Decimal("0"))
    gap_pct: Decimal = Field(default=Decimal("0"))
    cumulative_reduction_pct: Decimal = Field(default=Decimal("0"))
    status: str = Field(default=ProgressStatus.ON_TRACK.value)
    is_projected: bool = Field(default=False)

class VarianceAnalysis(BaseModel):
    """Variance decomposition analysis.

    Attributes:
        total_variance_tco2e: Total variance from target trajectory.
        total_variance_pct: Variance as percentage.
        scope1_variance_tco2e: Scope 1 contribution to variance.
        scope2_variance_tco2e: Scope 2 contribution to variance.
        scope3_variance_tco2e: Scope 3 contribution to variance.
        trend_direction: Overall emission trend.
        annual_rate_actual_pct: Actual annual reduction rate.
        annual_rate_required_pct: Required annual rate.
        rate_gap_pct: Gap between actual and required rate.
    """
    total_variance_tco2e: Decimal = Field(default=Decimal("0"))
    total_variance_pct: Decimal = Field(default=Decimal("0"))
    scope1_variance_tco2e: Decimal = Field(default=Decimal("0"))
    scope2_variance_tco2e: Decimal = Field(default=Decimal("0"))
    scope3_variance_tco2e: Decimal = Field(default=Decimal("0"))
    trend_direction: str = Field(default=TrendDirection.STABLE.value)
    annual_rate_actual_pct: Decimal = Field(default=Decimal("0"))
    annual_rate_required_pct: Decimal = Field(default=Decimal("0"))
    rate_gap_pct: Decimal = Field(default=Decimal("0"))

class CorrectiveAction(BaseModel):
    """Recommended corrective action.

    Attributes:
        action: Action description.
        impact_tco2e: Estimated additional abatement.
        priority: Priority (1=highest).
        timeline: Suggested implementation timeline.
        category: Action category.
    """
    action: str = Field(default="")
    impact_tco2e: Decimal = Field(default=Decimal("0"))
    priority: int = Field(default=3)
    timeline: str = Field(default="")
    category: str = Field(default="")

class ActionPlanProgress(BaseModel):
    """Summary of action plan implementation progress.

    Attributes:
        implementation_pct: % of actions initiated or completed.
        completion_pct: % of actions completed.
        on_schedule_pct: % of actions on schedule.
        abatement_realization_pct: % of planned abatement realized.
        budget_utilization_pct: % of budget spent.
    """
    implementation_pct: Decimal = Field(default=Decimal("0"))
    completion_pct: Decimal = Field(default=Decimal("0"))
    on_schedule_pct: Decimal = Field(default=Decimal("0"))
    abatement_realization_pct: Decimal = Field(default=Decimal("0"))
    budget_utilization_pct: Decimal = Field(default=Decimal("0"))

class ProgressTrackingResult(BaseModel):
    """Complete progress tracking result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Entity name.
        progress_status: Overall RAG status.
        current_year: Current reporting year.
        current_emissions_tco2e: Current year total emissions.
        baseline_emissions_tco2e: Baseline emissions.
        yoy_change_pct: Year-over-year change percentage.
        cumulative_reduction_pct: Cumulative reduction from baseline.
        trajectory_points: Full trajectory with actuals and projections.
        variance_analysis: Variance analysis.
        action_plan_progress: Action plan implementation summary.
        corrective_actions: Recommended corrective actions.
        years_to_interim: Years remaining to interim target.
        years_to_net_zero: Years remaining to net-zero target.
        interim_target_achievable: Whether interim target appears achievable.
        required_acceleration_pct: Additional annual reduction needed.
        data_years_available: Number of years with data.
        latest_verification: Latest verification status.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    progress_status: str = Field(default=ProgressStatus.CRITICAL.value)
    current_year: int = Field(default=0)
    current_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    baseline_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    yoy_change_pct: Decimal = Field(default=Decimal("0"))
    cumulative_reduction_pct: Decimal = Field(default=Decimal("0"))
    trajectory_points: List[TrajectoryPoint] = Field(default_factory=list)
    variance_analysis: Optional[VarianceAnalysis] = Field(default=None)
    action_plan_progress: Optional[ActionPlanProgress] = Field(default=None)
    corrective_actions: List[CorrectiveAction] = Field(default_factory=list)
    years_to_interim: int = Field(default=0)
    years_to_net_zero: int = Field(default=0)
    interim_target_achievable: bool = Field(default=False)
    required_acceleration_pct: Decimal = Field(default=Decimal("0"))
    data_years_available: int = Field(default=0)
    latest_verification: str = Field(default=VerificationStatus.SELF_REPORTED.value)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ProgressTrackingEngine:
    """Race to Zero annual progress tracking engine.

    Tracks progress against interim and long-term targets, produces
    RAG-classified trajectory analysis, variance decomposition, and
    corrective action recommendations.

    Usage::

        engine = ProgressTrackingEngine()
        result = engine.track(input_data)
        print(f"Status: {result.progress_status}")
        print(f"Reduction: {result.cumulative_reduction_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise ProgressTrackingEngine."""
        self.config = config or {}
        self._caution_threshold = _decimal(
            self.config.get("caution_threshold_pct", RAG_CAUTION_THRESHOLD)
        )
        self._off_track_threshold = _decimal(
            self.config.get("off_track_threshold_pct", RAG_OFF_TRACK_THRESHOLD)
        )
        logger.info("ProgressTrackingEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def track(
        self, data: ProgressTrackingInput,
    ) -> ProgressTrackingResult:
        """Perform complete progress tracking assessment.

        Args:
            data: Validated progress tracking input.

        Returns:
            ProgressTrackingResult.
        """
        t0 = time.perf_counter()
        logger.info(
            "Progress tracking: entity=%s, year=%d",
            data.entity_name, data.current_year,
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Sort records by year
        records = sorted(data.annual_records, key=lambda r: r.year)
        record_map = {r.year: r for r in records}

        # Current year data
        current_record = record_map.get(data.current_year)
        current_emissions = Decimal("0")
        if current_record:
            current_emissions = current_record.total_tco2e
        elif records:
            current_emissions = records[-1].total_tco2e
            warnings.append(
                f"No data for current year {data.current_year}. "
                f"Using latest available ({records[-1].year})."
            )

        # Step 1: Year-over-year change
        yoy_change = Decimal("0")
        if len(records) >= 2:
            prev = records[-2].total_tco2e
            curr = records[-1].total_tco2e
            yoy_change = _round_val(_safe_pct(curr - prev, prev), 2)

        # Step 2: Cumulative reduction
        cumulative = _round_val(
            _safe_pct(
                data.baseline_emissions_tco2e - current_emissions,
                data.baseline_emissions_tco2e,
            ), 2
        )

        # Step 3: Build trajectory
        trajectory = self._build_trajectory(
            data, records, current_emissions
        )

        # Step 4: Variance analysis
        variance = self._analyze_variance(
            data, records, current_emissions
        )

        # Step 5: Progress status
        status = self._determine_status(
            data, current_emissions, cumulative, variance
        )

        # Step 6: Timeline
        years_to_interim = max(0, data.interim_target_year - data.current_year)
        years_to_nz = max(0, data.net_zero_year - data.current_year)

        # Step 7: Achievability assessment
        achievable = False
        acceleration = Decimal("0")
        if years_to_interim > 0 and current_emissions > Decimal("0"):
            required_remaining = current_emissions - data.interim_target_tco2e
            if required_remaining > Decimal("0"):
                required_annual = _safe_divide(
                    required_remaining, _decimal(years_to_interim)
                )
                actual_rate = Decimal("0")
                if variance:
                    actual_rate = variance.annual_rate_actual_pct
                achievable = actual_rate > Decimal("0") and status != ProgressStatus.CRITICAL.value
                if variance and variance.annual_rate_required_pct > actual_rate:
                    acceleration = _round_val(
                        variance.annual_rate_required_pct - actual_rate, 2
                    )
            else:
                achievable = True

        # Step 8: Action plan progress
        ap_progress: Optional[ActionPlanProgress] = None
        if data.action_plan_status:
            ap_progress = self._assess_action_plan(data.action_plan_status)

        # Step 9: Corrective actions
        corrective: List[CorrectiveAction] = []
        if data.include_corrective_actions and status in (
            ProgressStatus.CAUTION.value,
            ProgressStatus.OFF_TRACK.value,
            ProgressStatus.CRITICAL.value,
        ):
            corrective = self._generate_corrective_actions(
                data, status, variance, acceleration
            )

        # Step 10: Latest verification
        latest_verif = VerificationStatus.SELF_REPORTED.value
        if records:
            latest_verif = records[-1].verification_status

        if latest_verif != VerificationStatus.VERIFIED.value:
            warnings.append(
                "Latest emissions data is not third-party verified. "
                "Race to Zero recommends annual verification."
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = ProgressTrackingResult(
            entity_name=data.entity_name,
            progress_status=status,
            current_year=data.current_year,
            current_emissions_tco2e=_round_val(current_emissions),
            baseline_emissions_tco2e=_round_val(data.baseline_emissions_tco2e),
            yoy_change_pct=yoy_change,
            cumulative_reduction_pct=cumulative,
            trajectory_points=trajectory,
            variance_analysis=variance,
            action_plan_progress=ap_progress,
            corrective_actions=corrective,
            years_to_interim=years_to_interim,
            years_to_net_zero=years_to_nz,
            interim_target_achievable=achievable,
            required_acceleration_pct=acceleration,
            data_years_available=len(records),
            latest_verification=latest_verif,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Progress tracking complete: status=%s, reduction=%.1f%%, "
            "yoy=%.1f%%, hash=%s",
            status, float(cumulative), float(yoy_change),
            result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _build_trajectory(
        self,
        data: ProgressTrackingInput,
        records: List[AnnualEmissionRecord],
        current: Decimal,
    ) -> List[TrajectoryPoint]:
        """Build emission trajectory with actual and projected points.

        Args:
            data: Input data.
            records: Sorted annual records.
            current: Current emissions.

        Returns:
            List of TrajectoryPoint.
        """
        points: List[TrajectoryPoint] = []
        record_map = {r.year: r for r in records}
        total_years = data.interim_target_year - data.baseline_year
        if total_years <= 0:
            return points

        annual_reduction = _safe_divide(
            data.baseline_emissions_tco2e - data.interim_target_tco2e,
            _decimal(total_years),
        )

        for year in range(data.baseline_year, data.interim_target_year + 1):
            elapsed = year - data.baseline_year
            target_emissions = data.baseline_emissions_tco2e - (annual_reduction * _decimal(elapsed))
            target_emissions = max(target_emissions, Decimal("0"))

            actual = None
            is_projected = True
            if year in record_map:
                actual = record_map[year].total_tco2e
                is_projected = False

            if actual is not None:
                gap = actual - target_emissions
                gap_pct = _safe_pct(gap, target_emissions)
                cum_reduction = _safe_pct(
                    data.baseline_emissions_tco2e - actual,
                    data.baseline_emissions_tco2e,
                )
            else:
                gap = Decimal("0")
                gap_pct = Decimal("0")
                cum_reduction = _safe_pct(
                    data.baseline_emissions_tco2e - target_emissions,
                    data.baseline_emissions_tco2e,
                )

            # RAG for this point
            if actual is not None:
                abs_gap_pct = abs(gap_pct)
                if gap <= Decimal("0"):
                    pt_status = ProgressStatus.ON_TRACK.value
                elif abs_gap_pct <= self._caution_threshold:
                    pt_status = ProgressStatus.CAUTION.value
                elif abs_gap_pct <= self._off_track_threshold:
                    pt_status = ProgressStatus.OFF_TRACK.value
                else:
                    pt_status = ProgressStatus.CRITICAL.value
            else:
                pt_status = ProgressStatus.ON_TRACK.value

            points.append(TrajectoryPoint(
                year=year,
                actual_tco2e=_round_val(actual) if actual is not None else None,
                target_tco2e=_round_val(target_emissions),
                gap_tco2e=_round_val(gap),
                gap_pct=_round_val(gap_pct, 2),
                cumulative_reduction_pct=_round_val(cum_reduction, 2),
                status=pt_status,
                is_projected=is_projected,
            ))

        return points

    def _analyze_variance(
        self,
        data: ProgressTrackingInput,
        records: List[AnnualEmissionRecord],
        current: Decimal,
    ) -> VarianceAnalysis:
        """Perform variance analysis.

        Args:
            data: Input data.
            records: Sorted annual records.
            current: Current emissions.

        Returns:
            VarianceAnalysis.
        """
        total_years = data.interim_target_year - data.baseline_year
        elapsed = max(1, data.current_year - data.baseline_year)
        years_remaining = max(1, data.interim_target_year - data.current_year)

        # Target trajectory for current year
        if total_years > 0:
            annual_reduction = _safe_divide(
                data.baseline_emissions_tco2e - data.interim_target_tco2e,
                _decimal(total_years),
            )
            expected = data.baseline_emissions_tco2e - annual_reduction * _decimal(elapsed)
        else:
            expected = data.interim_target_tco2e

        total_variance = current - expected
        variance_pct = _safe_pct(total_variance, expected)

        # Per-scope variance (if enough data)
        s1_var = Decimal("0")
        s2_var = Decimal("0")
        s3_var = Decimal("0")
        if records:
            latest = records[-1]
            baseline_rec = next((r for r in records if r.year == data.baseline_year), None)
            if baseline_rec:
                s1_var = latest.scope1_tco2e - baseline_rec.scope1_tco2e
                s2_var = latest.scope2_tco2e - baseline_rec.scope2_tco2e
                s3_var = latest.scope3_tco2e - baseline_rec.scope3_tco2e

        # Actual annual rate
        actual_rate = Decimal("0")
        if elapsed > 0 and data.baseline_emissions_tco2e > Decimal("0") and current > Decimal("0"):
            ratio = float(current / data.baseline_emissions_tco2e)
            if ratio > 0:
                actual_rate = _decimal((1.0 - math.pow(ratio, 1.0 / elapsed)) * 100.0)
                actual_rate = max(Decimal("0"), actual_rate)

        # Required rate from now
        required_rate = Decimal("0")
        if years_remaining > 0 and current > Decimal("0") and data.interim_target_tco2e < current:
            ratio = float(data.interim_target_tco2e / current)
            if ratio > 0:
                required_rate = _decimal((1.0 - math.pow(ratio, 1.0 / years_remaining)) * 100.0)
                required_rate = max(Decimal("0"), required_rate)

        # Trend direction
        if len(records) >= 2:
            recent_change = records[-1].total_tco2e - records[-2].total_tco2e
            if recent_change < -data.baseline_emissions_tco2e * Decimal("0.01"):
                trend = TrendDirection.DECREASING.value
            elif recent_change > data.baseline_emissions_tco2e * Decimal("0.01"):
                trend = TrendDirection.INCREASING.value
            else:
                trend = TrendDirection.STABLE.value
        else:
            trend = TrendDirection.STABLE.value

        return VarianceAnalysis(
            total_variance_tco2e=_round_val(total_variance),
            total_variance_pct=_round_val(variance_pct, 2),
            scope1_variance_tco2e=_round_val(s1_var),
            scope2_variance_tco2e=_round_val(s2_var),
            scope3_variance_tco2e=_round_val(s3_var),
            trend_direction=trend,
            annual_rate_actual_pct=_round_val(actual_rate, 2),
            annual_rate_required_pct=_round_val(required_rate, 2),
            rate_gap_pct=_round_val(max(Decimal("0"), required_rate - actual_rate), 2),
        )

    def _determine_status(
        self,
        data: ProgressTrackingInput,
        current: Decimal,
        cumulative: Decimal,
        variance: VarianceAnalysis,
    ) -> str:
        """Determine overall progress status (RAG).

        Args:
            data: Input data.
            current: Current emissions.
            cumulative: Cumulative reduction percentage.
            variance: Variance analysis.

        Returns:
            ProgressStatus value.
        """
        if current >= data.baseline_emissions_tco2e:
            return ProgressStatus.CRITICAL.value

        abs_variance = abs(variance.total_variance_pct)

        if variance.total_variance_tco2e <= Decimal("0"):
            return ProgressStatus.ON_TRACK.value
        elif abs_variance <= self._caution_threshold:
            return ProgressStatus.CAUTION.value
        elif abs_variance <= self._off_track_threshold:
            return ProgressStatus.OFF_TRACK.value
        else:
            return ProgressStatus.CRITICAL.value

    def _assess_action_plan(
        self, status: ActionPlanStatusInput,
    ) -> ActionPlanProgress:
        """Assess action plan implementation progress.

        Args:
            status: Action plan status input.

        Returns:
            ActionPlanProgress.
        """
        total = max(1, status.total_actions)
        impl = _safe_pct(
            _decimal(status.actions_initiated + status.actions_completed),
            _decimal(total),
        )
        comp = _safe_pct(_decimal(status.actions_completed), _decimal(total))
        sched = _safe_pct(_decimal(status.actions_on_schedule), _decimal(total))
        abatement = _safe_pct(
            status.abatement_realized_tco2e,
            status.abatement_planned_tco2e,
        )
        budget = _safe_pct(status.budget_spent_usd, status.budget_total_usd)

        return ActionPlanProgress(
            implementation_pct=_round_val(impl, 2),
            completion_pct=_round_val(comp, 2),
            on_schedule_pct=_round_val(sched, 2),
            abatement_realization_pct=_round_val(abatement, 2),
            budget_utilization_pct=_round_val(budget, 2),
        )

    def _generate_corrective_actions(
        self,
        data: ProgressTrackingInput,
        status: str,
        variance: Optional[VarianceAnalysis],
        acceleration: Decimal,
    ) -> List[CorrectiveAction]:
        """Generate corrective action recommendations.

        Args:
            data: Input data.
            status: Current progress status.
            variance: Variance analysis.
            acceleration: Additional annual reduction needed.

        Returns:
            List of CorrectiveAction.
        """
        actions: List[CorrectiveAction] = []

        if status == ProgressStatus.CRITICAL.value:
            actions.append(CorrectiveAction(
                action="Convene emergency climate review with senior leadership.",
                impact_tco2e=Decimal("0"),
                priority=1,
                timeline="Immediate (within 30 days)",
                category="governance",
            ))

        if variance and variance.scope1_variance_tco2e > Decimal("0"):
            actions.append(CorrectiveAction(
                action="Accelerate Scope 1 reduction actions: energy efficiency, "
                       "fuel switching, process optimization.",
                impact_tco2e=abs(variance.scope1_variance_tco2e),
                priority=2,
                timeline="Next 6 months",
                category="scope1",
            ))

        if variance and variance.scope2_variance_tco2e > Decimal("0"):
            actions.append(CorrectiveAction(
                action="Increase renewable energy procurement (PPAs, RECs) "
                       "to reduce Scope 2 emissions.",
                impact_tco2e=abs(variance.scope2_variance_tco2e),
                priority=2,
                timeline="Next 6 months",
                category="scope2",
            ))

        if variance and variance.scope3_variance_tco2e > Decimal("0"):
            actions.append(CorrectiveAction(
                action="Intensify supplier engagement and Scope 3 reduction "
                       "programs across material categories.",
                impact_tco2e=abs(variance.scope3_variance_tco2e),
                priority=3,
                timeline="Next 12 months",
                category="scope3",
            ))

        if acceleration > Decimal("0"):
            actions.append(CorrectiveAction(
                action=f"Accelerate annual reduction rate by an additional "
                       f"{acceleration}% per year to meet interim target.",
                impact_tco2e=Decimal("0"),
                priority=1,
                timeline="Ongoing",
                category="acceleration",
            ))

        return actions
