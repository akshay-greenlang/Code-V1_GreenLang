# -*- coding: utf-8 -*-
"""
Performance Analysis Workflow - EnPI Tracking
===================================

3-phase workflow for analysing energy performance indicators (EnPIs)
within PACK-034 ISO 50001 Energy Management System Pack.

Phases:
    1. EnPICalculation   -- Calculate all EnPIs for the analysis period
    2. CUSUMCheck        -- Update CUSUM monitors and check for shifts
    3. TrendReporting    -- Generate trend analysis and performance reports

The workflow follows GreenLang zero-hallucination principles: all EnPI
calculations use deterministic arithmetic on validated meter data,
CUSUM monitoring uses standard statistical formulas, and trend analysis
is based on simple linear regression. SHA-256 provenance hashes
guarantee auditability.

Schedule: monthly / quarterly
Estimated duration: 20 minutes

Author: GreenLang Team
Version: 34.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class AnalysisPhase(str, Enum):
    """Phases of the performance analysis workflow."""

    ENPI_CALCULATION = "enpi_calculation"
    CUSUM_CHECK = "cusum_check"
    TREND_REPORTING = "trend_reporting"


class PerformanceTrend(str, Enum):
    """Performance trend classification."""

    STRONG_IMPROVEMENT = "strong_improvement"
    MODERATE_IMPROVEMENT = "moderate_improvement"
    STABLE = "stable"
    MODERATE_DETERIORATION = "moderate_deterioration"
    STRONG_DETERIORATION = "strong_deterioration"


class CUSUMStatus(str, Enum):
    """CUSUM chart status indicators."""

    IN_CONTROL = "in_control"
    WARNING = "warning"
    OUT_OF_CONTROL = "out_of_control"
    IMPROVING = "improving"
    DETERIORATING = "deteriorating"


# =============================================================================
# PERFORMANCE THRESHOLDS (Zero-Hallucination)
# =============================================================================

# Trend classification thresholds (% change per period)
TREND_THRESHOLDS: Dict[str, float] = {
    "strong_improvement": -5.0,
    "moderate_improvement": -2.0,
    "stable_upper": 2.0,
    "moderate_deterioration": 5.0,
}

# CUSUM parameters
CUSUM_H: float = 5.0
CUSUM_K: float = 0.5

# Performance score weights
SCORE_WEIGHTS: Dict[str, float] = {
    "enpi_improvement": 0.40,
    "target_achievement": 0.30,
    "cusum_stability": 0.20,
    "data_quality": 0.10,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class EnPIResult(BaseModel):
    """Calculated EnPI result for a single indicator."""

    enpi_id: str = Field(default_factory=lambda: f"enpi-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="", description="EnPI name")
    category: str = Field(default="", description="Energy end-use category")
    current_value: Decimal = Field(default=Decimal("0"), description="Current period value")
    previous_value: Decimal = Field(default=Decimal("0"), description="Previous period value")
    baseline_value: Decimal = Field(default=Decimal("0"), description="Baseline value")
    target_value: Decimal = Field(default=Decimal("0"), description="Target value")
    change_vs_baseline_pct: Decimal = Field(default=Decimal("0"))
    change_vs_previous_pct: Decimal = Field(default=Decimal("0"))
    on_target: bool = Field(default=False, description="True if meeting or exceeding target")
    unit: str = Field(default="kWh/unit")
    trend: PerformanceTrend = Field(default=PerformanceTrend.STABLE)


class CUSUMCheckResult(BaseModel):
    """CUSUM check result for an EnPI."""

    enpi_id: str = Field(default="")
    enpi_name: str = Field(default="")
    cusum_plus: Decimal = Field(default=Decimal("0"))
    cusum_minus: Decimal = Field(default=Decimal("0"))
    status: CUSUMStatus = Field(default=CUSUMStatus.IN_CONTROL)
    shift_detected: bool = Field(default=False)
    shift_direction: str = Field(default="none", description="improving|deteriorating|none")
    cumulative_deviation: Decimal = Field(default=Decimal("0"))
    periods_since_last_reset: int = Field(default=0, ge=0)


class TrendAnalysis(BaseModel):
    """Trend analysis result for an EnPI."""

    enpi_id: str = Field(default="")
    enpi_name: str = Field(default="")
    trend_direction: PerformanceTrend = Field(default=PerformanceTrend.STABLE)
    slope_per_period: Decimal = Field(default=Decimal("0"), description="Linear trend slope")
    r_squared: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    projected_next_value: Decimal = Field(default=Decimal("0"))
    projected_year_end_value: Decimal = Field(default=Decimal("0"))
    recommendation: str = Field(default="")


class ImprovementSummary(BaseModel):
    """Summary of energy performance improvements."""

    total_enpis: int = Field(default=0, ge=0)
    improving_count: int = Field(default=0, ge=0)
    stable_count: int = Field(default=0, ge=0)
    deteriorating_count: int = Field(default=0, ge=0)
    on_target_count: int = Field(default=0, ge=0)
    total_savings_vs_baseline_kwh: Decimal = Field(default=Decimal("0"))
    total_savings_vs_baseline_pct: Decimal = Field(default=Decimal("0"))


class PerformanceAnalysisInput(BaseModel):
    """Input data model for PerformanceAnalysisWorkflow."""

    enms_id: str = Field(default="", description="EnMS program identifier")
    enpi_definitions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="EnPI definitions: [{name, category, unit, baseline_value, target_value}]",
    )
    measurement_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Current period measurements: [{enpi_name, value, period}]",
    )
    baseline_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Baseline data: [{enpi_name, baseline_value, target_value}]",
    )
    historical_values: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Historical EnPI values by name for trend analysis",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class PerformanceAnalysisResult(BaseModel):
    """Complete result from performance analysis workflow."""

    analysis_id: str = Field(..., description="Unique analysis execution ID")
    enms_id: str = Field(default="", description="EnMS program identifier")
    enpi_results: List[EnPIResult] = Field(default_factory=list)
    cusum_status: List[CUSUMCheckResult] = Field(default_factory=list)
    trend_analysis: List[TrendAnalysis] = Field(default_factory=list)
    improvement_summary: ImprovementSummary = Field(
        default_factory=ImprovementSummary,
    )
    performance_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Overall performance score 0-100",
    )
    phases_completed: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PerformanceAnalysisWorkflow:
    """
    3-phase performance analysis workflow for EnPI tracking.

    Calculates all EnPIs for the analysis period, runs CUSUM monitoring
    for performance shift detection, and generates trend analysis with
    projections and recommendations.

    Zero-hallucination: EnPI calculations use deterministic arithmetic,
    CUSUM uses standard statistical formulas, and trend analysis uses
    simple linear regression. No LLM calls in the numeric computation path.

    Attributes:
        analysis_id: Unique analysis execution identifier.
        _enpi_results: Calculated EnPI results.
        _cusum_results: CUSUM check results.
        _trend_analyses: Trend analysis results.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = PerformanceAnalysisWorkflow()
        >>> inp = PerformanceAnalysisInput(
        ...     enms_id="enms-001",
        ...     enpi_definitions=[{"name": "HVAC Intensity", "category": "hvac"}],
        ...     measurement_data=[{"enpi_name": "HVAC Intensity", "value": 45000}],
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.performance_score >= 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PerformanceAnalysisWorkflow."""
        self.analysis_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._enpi_results: List[EnPIResult] = []
        self._cusum_results: List[CUSUMCheckResult] = []
        self._trend_analyses: List[TrendAnalysis] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: PerformanceAnalysisInput) -> PerformanceAnalysisResult:
        """
        Execute the 3-phase performance analysis workflow.

        Args:
            input_data: Validated performance analysis input.

        Returns:
            PerformanceAnalysisResult with EnPIs, CUSUM, trends, and score.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting performance analysis workflow %s enms=%s enpis=%d",
            self.analysis_id, input_data.enms_id, len(input_data.enpi_definitions),
        )

        self._phase_results = []
        self._enpi_results = []
        self._cusum_results = []
        self._trend_analyses = []

        try:
            phase1 = self._phase_enpi_calculation(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_cusum_check(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_trend_reporting(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error(
                "Performance analysis workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        improvement_summary = self._build_improvement_summary()
        performance_score = self._calculate_performance_score()

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = PerformanceAnalysisResult(
            analysis_id=self.analysis_id,
            enms_id=input_data.enms_id,
            enpi_results=self._enpi_results,
            cusum_status=self._cusum_results,
            trend_analysis=self._trend_analyses,
            improvement_summary=improvement_summary,
            performance_score=performance_score,
            phases_completed=completed_phases,
            execution_time_ms=round(elapsed_ms, 2),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Performance analysis workflow %s completed in %.0fms "
            "enpis=%d score=%.1f improving=%d deteriorating=%d",
            self.analysis_id, elapsed_ms, len(self._enpi_results),
            float(performance_score), improvement_summary.improving_count,
            improvement_summary.deteriorating_count,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: EnPI Calculation
    # -------------------------------------------------------------------------

    def _phase_enpi_calculation(
        self, input_data: PerformanceAnalysisInput
    ) -> PhaseResult:
        """Calculate all EnPIs for the analysis period."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Build lookup maps
        baseline_map: Dict[str, Dict[str, Any]] = {}
        for bl in input_data.baseline_data:
            baseline_map[bl.get("enpi_name", "")] = bl

        measurement_map: Dict[str, float] = {}
        for md in input_data.measurement_data:
            measurement_map[md.get("enpi_name", "")] = float(md.get("value", 0))

        for enpi_def in input_data.enpi_definitions:
            name = enpi_def.get("name", "")
            category = enpi_def.get("category", "")
            unit = enpi_def.get("unit", "kWh/unit")

            # Current value from measurements
            current = measurement_map.get(name, 0.0)

            # Baseline and target values
            bl_data = baseline_map.get(name, {})
            baseline = float(bl_data.get("baseline_value", enpi_def.get("baseline_value", 0)))
            target = float(bl_data.get("target_value", enpi_def.get("target_value", 0)))

            # Previous period value from history
            history = input_data.historical_values.get(name, [])
            previous = history[-1] if history else baseline

            # Calculate changes
            change_vs_baseline = Decimal("0")
            if baseline > 0:
                change_vs_baseline = Decimal(str(round(
                    (current - baseline) / baseline * 100.0, 2
                )))

            change_vs_previous = Decimal("0")
            if previous > 0:
                change_vs_previous = Decimal(str(round(
                    (current - previous) / previous * 100.0, 2
                )))

            # On target check
            on_target = current <= target if target > 0 else False

            # Classify trend
            trend = self._classify_trend(float(change_vs_baseline))

            enpi_result = EnPIResult(
                name=name,
                category=category,
                current_value=Decimal(str(round(current, 2))),
                previous_value=Decimal(str(round(previous, 2))),
                baseline_value=Decimal(str(round(baseline, 2))),
                target_value=Decimal(str(round(target, 2))),
                change_vs_baseline_pct=change_vs_baseline,
                change_vs_previous_pct=change_vs_previous,
                on_target=on_target,
                unit=unit,
                trend=trend,
            )
            self._enpi_results.append(enpi_result)

        outputs["enpis_calculated"] = len(self._enpi_results)
        outputs["on_target"] = sum(1 for e in self._enpi_results if e.on_target)
        outputs["off_target"] = sum(1 for e in self._enpi_results if not e.on_target)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 EnPICalculation: %d EnPIs, %d on target",
            len(self._enpi_results), outputs["on_target"],
        )
        return PhaseResult(
            phase_name=AnalysisPhase.ENPI_CALCULATION.value, phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _classify_trend(self, change_pct: float) -> PerformanceTrend:
        """Classify performance trend based on % change."""
        if change_pct <= TREND_THRESHOLDS["strong_improvement"]:
            return PerformanceTrend.STRONG_IMPROVEMENT
        elif change_pct <= TREND_THRESHOLDS["moderate_improvement"]:
            return PerformanceTrend.MODERATE_IMPROVEMENT
        elif change_pct <= TREND_THRESHOLDS["stable_upper"]:
            return PerformanceTrend.STABLE
        elif change_pct <= TREND_THRESHOLDS["moderate_deterioration"]:
            return PerformanceTrend.MODERATE_DETERIORATION
        else:
            return PerformanceTrend.STRONG_DETERIORATION

    # -------------------------------------------------------------------------
    # Phase 2: CUSUM Check
    # -------------------------------------------------------------------------

    def _phase_cusum_check(
        self, input_data: PerformanceAnalysisInput
    ) -> PhaseResult:
        """Update CUSUM monitors and check for performance shifts."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for enpi in self._enpi_results:
            history = input_data.historical_values.get(enpi.name, [])
            baseline = float(enpi.baseline_value)

            if not history or baseline <= 0:
                self._cusum_results.append(CUSUMCheckResult(
                    enpi_id=enpi.enpi_id,
                    enpi_name=enpi.name,
                    status=CUSUMStatus.IN_CONTROL,
                    periods_since_last_reset=0,
                ))
                continue

            # Calculate standard deviation
            n = len(history)
            mean_h = sum(history) / n
            std_dev = math.sqrt(
                sum((x - mean_h) ** 2 for x in history) / max(n - 1, 1)
            ) if n > 1 else abs(mean_h * 0.1)

            if std_dev <= 0:
                std_dev = 1.0

            k = CUSUM_K * std_dev
            h = CUSUM_H * std_dev

            cusum_plus = 0.0
            cusum_minus = 0.0
            shift_detected = False

            all_values = history + [float(enpi.current_value)]
            for val in all_values:
                deviation = val - baseline
                cusum_plus = max(0.0, cusum_plus + deviation - k)
                cusum_minus = max(0.0, cusum_minus - deviation - k)
                if cusum_plus > h or cusum_minus > h:
                    shift_detected = True

            # Determine shift direction
            shift_direction = "none"
            if shift_detected:
                if float(enpi.current_value) < baseline:
                    shift_direction = "improving"
                else:
                    shift_direction = "deteriorating"

            # Determine status
            if shift_detected and shift_direction == "deteriorating":
                status = CUSUMStatus.OUT_OF_CONTROL
            elif shift_detected and shift_direction == "improving":
                status = CUSUMStatus.IMPROVING
            elif cusum_plus > h * 0.7 or cusum_minus > h * 0.7:
                status = CUSUMStatus.WARNING
            else:
                status = CUSUMStatus.IN_CONTROL

            cusum_result = CUSUMCheckResult(
                enpi_id=enpi.enpi_id,
                enpi_name=enpi.name,
                cusum_plus=Decimal(str(round(cusum_plus, 4))),
                cusum_minus=Decimal(str(round(cusum_minus, 4))),
                status=status,
                shift_detected=shift_detected,
                shift_direction=shift_direction,
                cumulative_deviation=Decimal(str(round(baseline - float(enpi.current_value), 2))),
                periods_since_last_reset=len(all_values),
            )
            self._cusum_results.append(cusum_result)

        outputs["cusum_analyses"] = len(self._cusum_results)
        outputs["in_control"] = sum(1 for c in self._cusum_results if c.status == CUSUMStatus.IN_CONTROL)
        outputs["shifts_detected"] = sum(1 for c in self._cusum_results if c.shift_detected)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 CUSUMCheck: %d analyses, %d in control, %d shifts",
            len(self._cusum_results), outputs["in_control"], outputs["shifts_detected"],
        )
        return PhaseResult(
            phase_name=AnalysisPhase.CUSUM_CHECK.value, phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Trend Reporting
    # -------------------------------------------------------------------------

    def _phase_trend_reporting(
        self, input_data: PerformanceAnalysisInput
    ) -> PhaseResult:
        """Generate trend analysis and performance reports."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for enpi in self._enpi_results:
            history = input_data.historical_values.get(enpi.name, [])

            if len(history) < 3:
                self._trend_analyses.append(TrendAnalysis(
                    enpi_id=enpi.enpi_id,
                    enpi_name=enpi.name,
                    trend_direction=enpi.trend,
                    recommendation="Insufficient data for trend analysis; continue monitoring",
                ))
                continue

            # Linear regression for trend
            all_values = history + [float(enpi.current_value)]
            n = len(all_values)
            x = list(range(n))
            y = all_values

            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi ** 2 for xi in x)

            mean_x = sum_x / n
            mean_y = sum_y / n

            denom = sum_x2 - (sum_x ** 2) / n
            if abs(denom) < 1e-10:
                slope = 0.0
                intercept = mean_y
            else:
                slope = (sum_xy - sum_x * sum_y / n) / denom
                intercept = mean_y - slope * mean_x

            # R-squared
            predicted = [intercept + slope * xi for xi in x]
            ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, predicted))
            ss_tot = sum((yi - mean_y) ** 2 for yi in y)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Projections
            projected_next = intercept + slope * n
            projected_year_end = intercept + slope * (n + 11)  # 12 periods ahead

            # Recommendation
            recommendation = self._generate_trend_recommendation(
                enpi.name, enpi.trend, slope, float(enpi.baseline_value),
            )

            trend_analysis = TrendAnalysis(
                enpi_id=enpi.enpi_id,
                enpi_name=enpi.name,
                trend_direction=enpi.trend,
                slope_per_period=Decimal(str(round(slope, 4))),
                r_squared=Decimal(str(round(max(0.0, r_squared), 4))),
                projected_next_value=Decimal(str(round(projected_next, 2))),
                projected_year_end_value=Decimal(str(round(projected_year_end, 2))),
                recommendation=recommendation,
            )
            self._trend_analyses.append(trend_analysis)

        outputs["trends_analysed"] = len(self._trend_analyses)
        outputs["trend_distribution"] = {
            t.value: sum(1 for ta in self._trend_analyses if ta.trend_direction == t)
            for t in PerformanceTrend
        }

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 TrendReporting: %d trends analysed",
            len(self._trend_analyses),
        )
        return PhaseResult(
            phase_name=AnalysisPhase.TREND_REPORTING.value, phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_trend_recommendation(
        self, name: str, trend: PerformanceTrend, slope: float, baseline: float
    ) -> str:
        """Generate recommendation based on trend analysis."""
        if trend in (PerformanceTrend.STRONG_IMPROVEMENT, PerformanceTrend.MODERATE_IMPROVEMENT):
            return (
                f"{name}: Positive trend. Continue current energy management practices. "
                f"Consider documenting successful strategies for replication."
            )
        elif trend == PerformanceTrend.STABLE:
            return (
                f"{name}: Stable performance. Review action plans for additional "
                f"improvement opportunities."
            )
        elif trend == PerformanceTrend.MODERATE_DETERIORATION:
            return (
                f"{name}: Moderate deterioration detected (slope={slope:.2f}/period). "
                f"Investigate root causes and review operating criteria."
            )
        else:
            return (
                f"{name}: Significant deterioration (slope={slope:.2f}/period). "
                f"Immediate investigation required. Review operational controls "
                f"and consider corrective action plan."
            )

    # -------------------------------------------------------------------------
    # Supporting Methods
    # -------------------------------------------------------------------------

    def _build_improvement_summary(self) -> ImprovementSummary:
        """Build summary of energy performance improvements."""
        improving = sum(
            1 for e in self._enpi_results
            if e.trend in (PerformanceTrend.STRONG_IMPROVEMENT, PerformanceTrend.MODERATE_IMPROVEMENT)
        )
        stable = sum(1 for e in self._enpi_results if e.trend == PerformanceTrend.STABLE)
        deteriorating = sum(
            1 for e in self._enpi_results
            if e.trend in (PerformanceTrend.MODERATE_DETERIORATION, PerformanceTrend.STRONG_DETERIORATION)
        )
        on_target = sum(1 for e in self._enpi_results if e.on_target)

        total_baseline = sum(float(e.baseline_value) for e in self._enpi_results)
        total_current = sum(float(e.current_value) for e in self._enpi_results)
        savings = total_baseline - total_current
        savings_pct = (savings / total_baseline * 100.0) if total_baseline > 0 else 0.0

        return ImprovementSummary(
            total_enpis=len(self._enpi_results),
            improving_count=improving,
            stable_count=stable,
            deteriorating_count=deteriorating,
            on_target_count=on_target,
            total_savings_vs_baseline_kwh=Decimal(str(round(savings, 2))),
            total_savings_vs_baseline_pct=Decimal(str(round(savings_pct, 2))),
        )

    def _calculate_performance_score(self) -> Decimal:
        """Calculate overall performance score (0-100)."""
        if not self._enpi_results:
            return Decimal("0")

        total = len(self._enpi_results)

        # EnPI improvement component
        improving = sum(
            1 for e in self._enpi_results
            if e.trend in (PerformanceTrend.STRONG_IMPROVEMENT, PerformanceTrend.MODERATE_IMPROVEMENT)
        )
        enpi_score = (improving / total) * 100.0

        # Target achievement component
        on_target = sum(1 for e in self._enpi_results if e.on_target)
        target_score = (on_target / total) * 100.0

        # CUSUM stability component
        in_control = sum(
            1 for c in self._cusum_results
            if c.status in (CUSUMStatus.IN_CONTROL, CUSUMStatus.IMPROVING)
        )
        cusum_total = max(len(self._cusum_results), 1)
        cusum_score = (in_control / cusum_total) * 100.0

        # Data quality component (assume good if we have results)
        dq_score = 80.0

        weighted = (
            SCORE_WEIGHTS["enpi_improvement"] * enpi_score
            + SCORE_WEIGHTS["target_achievement"] * target_score
            + SCORE_WEIGHTS["cusum_stability"] * cusum_score
            + SCORE_WEIGHTS["data_quality"] * dq_score
        )

        return Decimal(str(round(min(weighted, 100.0), 1)))

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: PerformanceAnalysisResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
