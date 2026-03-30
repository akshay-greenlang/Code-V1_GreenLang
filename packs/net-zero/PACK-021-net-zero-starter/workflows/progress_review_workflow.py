# -*- coding: utf-8 -*-
"""
Progress Review Workflow
============================

4-phase workflow for annual net-zero progress review within
PACK-021 Net-Zero Starter Pack.  The workflow ingests new emissions
data, calculates year-over-year and cumulative progress, performs a
gap analysis against the target pathway, and generates a progress
report with corrective action recommendations.

Phases:
    1. DataUpdate       -- Ingest and validate new year's emissions data
    2. ProgressCalc     -- Calculate YoY change, cumulative progress, intensity metrics
    3. GapAnalysis      -- Compare trajectory to target pathway, RAG status
    4. ReportGeneration -- Generate progress report with trend data

Zero-hallucination: all calculations use deterministic formulas.
SHA-256 provenance hashes for auditability.

Author: GreenLang Team
Version: 21.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "21.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

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

class RAGStatus(str, Enum):
    """Red-Amber-Green status indicator."""

    GREEN = "green"      # On-track or ahead
    AMBER = "amber"      # Slightly behind (within threshold)
    RED = "red"          # Significantly behind target

class TrendDirection(str, Enum):
    """Emission trend direction."""

    DECREASING = "decreasing"
    FLAT = "flat"
    INCREASING = "increasing"

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class AnnualEmissions(BaseModel):
    """Emissions data for a single reporting year."""

    year: int = Field(..., ge=2015, le=2060)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    revenue_usd: Optional[float] = Field(None, ge=0.0, description="Revenue for intensity calc")
    employee_count: Optional[int] = Field(None, ge=0)
    floor_area_sqm: Optional[float] = Field(None, ge=0.0)
    data_quality_score: float = Field(default=3.0, ge=1.0, le=5.0)

class TargetPathwayPoint(BaseModel):
    """Expected emissions at a given year on the target pathway."""

    year: int = Field(..., ge=2015, le=2060)
    target_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_from_base_pct: float = Field(default=0.0, ge=0.0, le=100.0)

class ProgressReviewConfig(BaseModel):
    """Configuration for the progress review workflow."""

    review_year: int = Field(..., ge=2020, le=2060, description="Year under review")
    base_year: int = Field(default=2024, ge=2015, le=2050)
    base_year_emissions: AnnualEmissions = Field(default_factory=lambda: AnnualEmissions(year=2024))
    historical_emissions: List[AnnualEmissions] = Field(
        default_factory=list, description="Prior year emissions"
    )
    current_year_emissions: AnnualEmissions = Field(
        default_factory=lambda: AnnualEmissions(year=2025),
        description="Emissions for review year",
    )
    target_pathway: List[TargetPathwayPoint] = Field(
        default_factory=list, description="Target trajectory points"
    )
    annual_reduction_rate_pct: float = Field(default=4.2, ge=0.0, le=20.0)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2040)
    near_term_target_reduction_pct: float = Field(default=42.0, ge=0.0, le=100.0)
    comparison_years: List[int] = Field(default_factory=list, description="Years to compare")
    include_projections: bool = Field(default=True)
    alert_threshold_pct: float = Field(default=5.0, ge=0.0, le=50.0, description="RAG amber threshold")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class YearOverYearChange(BaseModel):
    """Year-over-year change metrics."""

    from_year: int = Field(default=0)
    to_year: int = Field(default=0)
    absolute_change_tco2e: float = Field(default=0.0)
    percentage_change: float = Field(default=0.0)
    scope1_change_pct: float = Field(default=0.0)
    scope2_change_pct: float = Field(default=0.0)
    scope3_change_pct: float = Field(default=0.0)

class CumulativeProgress(BaseModel):
    """Cumulative progress against base year."""

    base_year: int = Field(default=2024)
    review_year: int = Field(default=2025)
    base_year_total_tco2e: float = Field(default=0.0, ge=0.0)
    current_total_tco2e: float = Field(default=0.0, ge=0.0)
    absolute_reduction_tco2e: float = Field(default=0.0)
    percentage_reduction: float = Field(default=0.0)
    years_elapsed: int = Field(default=0, ge=0)
    annualised_reduction_rate_pct: float = Field(default=0.0)

class IntensityMetric(BaseModel):
    """Emission intensity metric."""

    metric_name: str = Field(default="")
    unit: str = Field(default="")
    base_year_value: float = Field(default=0.0)
    current_value: float = Field(default=0.0)
    change_pct: float = Field(default=0.0)

class GapAnalysisResult(BaseModel):
    """Result of gap analysis against target pathway."""

    rag_status: RAGStatus = Field(default=RAGStatus.RED)
    target_tco2e_for_year: float = Field(default=0.0, ge=0.0)
    actual_tco2e: float = Field(default=0.0, ge=0.0)
    gap_tco2e: float = Field(default=0.0, description="Positive = behind target")
    gap_pct: float = Field(default=0.0)
    trend: TrendDirection = Field(default=TrendDirection.FLAT)
    projected_tco2e_at_target_year: float = Field(default=0.0, ge=0.0)
    on_track_for_near_term: bool = Field(default=False)
    additional_reduction_needed_tco2e: float = Field(default=0.0, ge=0.0)
    additional_annual_rate_needed_pct: float = Field(default=0.0, ge=0.0)

class TrendDataPoint(BaseModel):
    """Single point for trend chart data."""

    year: int = Field(default=0)
    actual_tco2e: Optional[float] = Field(None, ge=0.0)
    target_tco2e: Optional[float] = Field(None, ge=0.0)
    projected_tco2e: Optional[float] = Field(None, ge=0.0)

class CorrectiveAction(BaseModel):
    """Recommended corrective action."""

    priority: int = Field(default=0, ge=0)
    action: str = Field(default="")
    expected_impact_tco2e: float = Field(default=0.0, ge=0.0)
    timeframe: str = Field(default="")

class ProgressSummary(BaseModel):
    """Summary of progress for the review year."""

    review_year: int = Field(default=0)
    yoy_changes: List[YearOverYearChange] = Field(default_factory=list)
    cumulative_progress: CumulativeProgress = Field(default_factory=CumulativeProgress)
    intensity_metrics: List[IntensityMetric] = Field(default_factory=list)

class ProgressReviewResult(BaseModel):
    """Complete result from the progress review workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="progress_review")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    progress_summary: ProgressSummary = Field(default_factory=ProgressSummary)
    gap_analysis: GapAnalysisResult = Field(default_factory=GapAnalysisResult)
    trajectory_status: RAGStatus = Field(default=RAGStatus.RED)
    trend_data: List[TrendDataPoint] = Field(default_factory=list)
    recommendations: List[CorrectiveAction] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ProgressReviewWorkflow:
    """
    4-phase progress review workflow for net-zero tracking.

    Ingests new emissions data, calculates progress metrics, performs
    gap analysis against the target pathway, and generates a progress
    report with corrective action recommendations.

    Zero-hallucination: all calculations are deterministic.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = ProgressReviewWorkflow()
        >>> cfg = ProgressReviewConfig(review_year=2026, ...)
        >>> result = await wf.execute(cfg)
        >>> assert result.trajectory_status in [RAGStatus.GREEN, RAGStatus.AMBER]
    """

    def __init__(self) -> None:
        """Initialise ProgressReviewWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._progress: ProgressSummary = ProgressSummary()
        self._gap: GapAnalysisResult = GapAnalysisResult()
        self._trend_data: List[TrendDataPoint] = []
        self._recommendations: List[CorrectiveAction] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: ProgressReviewConfig) -> ProgressReviewResult:
        """
        Execute the 4-phase progress review workflow.

        Args:
            config: Review configuration with current and historical data,
                target pathway, and alerting thresholds.

        Returns:
            ProgressReviewResult with progress summary, gap analysis, and recommendations.
        """
        started_at = utcnow()
        self.logger.info(
            "Starting progress review workflow %s, review_year=%d",
            self.workflow_id, config.review_year,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_data_update(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"DataUpdate failed: {phase1.errors}")

            phase2 = await self._phase_progress_calc(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_gap_analysis(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_report_generation(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Progress review workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        result = ProgressReviewResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            progress_summary=self._progress,
            gap_analysis=self._gap,
            trajectory_status=self._gap.rag_status,
            trend_data=self._trend_data,
            recommendations=self._recommendations,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Progress review %s completed in %.2fs, RAG=%s",
            self.workflow_id, elapsed, self._gap.rag_status.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Update
    # -------------------------------------------------------------------------

    async def _phase_data_update(self, config: ProgressReviewConfig) -> PhaseResult:
        """Ingest and validate new year's emissions data."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        current = config.current_year_emissions

        # Validate year matches review_year
        if current.year != config.review_year:
            warnings.append(
                f"Current emissions year ({current.year}) differs from review year ({config.review_year})"
            )

        # Validate total consistency
        expected_total = current.scope1_tco2e + current.scope2_location_tco2e + current.scope3_tco2e
        if current.total_tco2e <= 0 and expected_total > 0:
            current.total_tco2e = expected_total
            warnings.append("total_tco2e was zero; auto-calculated from scope components")
        elif current.total_tco2e > 0 and abs(current.total_tco2e - expected_total) > 0.01 * expected_total:
            warnings.append(
                f"total_tco2e ({current.total_tco2e:.2f}) differs from sum of scopes ({expected_total:.2f})"
            )

        # Validate base year data
        base = config.base_year_emissions
        if base.total_tco2e <= 0:
            errors.append("Base year emissions total is zero; cannot calculate progress")

        # Validate data quality
        if current.data_quality_score > 3.5:
            warnings.append(f"Data quality score ({current.data_quality_score:.1f}) is below average")

        # Check structural consistency with previous year
        if config.historical_emissions:
            prev = config.historical_emissions[-1]
            if prev.scope3_tco2e > 0 and current.scope3_tco2e <= 0:
                warnings.append("Scope 3 data present in previous year but missing in current year")
            if prev.scope1_tco2e > 0 and current.scope1_tco2e <= 0:
                warnings.append("Scope 1 data present in previous year but missing in current year")

        outputs["review_year"] = config.review_year
        outputs["current_total_tco2e"] = current.total_tco2e
        outputs["base_year_total_tco2e"] = base.total_tco2e
        outputs["historical_years"] = len(config.historical_emissions)
        outputs["data_quality_score"] = current.data_quality_score

        elapsed = (utcnow() - started).total_seconds()
        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        return PhaseResult(
            phase_name="data_update",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Progress Calculation
    # -------------------------------------------------------------------------

    async def _phase_progress_calc(self, config: ProgressReviewConfig) -> PhaseResult:
        """Calculate year-over-year change, cumulative progress, intensity metrics."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        current = config.current_year_emissions
        base = config.base_year_emissions

        # Year-over-year changes
        yoy_changes: List[YearOverYearChange] = []

        # Compare to previous year
        all_years = sorted(config.historical_emissions + [current], key=lambda e: e.year)
        for i in range(1, len(all_years)):
            prev_yr = all_years[i - 1]
            curr_yr = all_years[i]
            yoy = self._calc_yoy(prev_yr, curr_yr)
            yoy_changes.append(yoy)

        # Cumulative progress vs base year
        years_elapsed = config.review_year - config.base_year
        base_total = base.total_tco2e or 1.0
        current_total = current.total_tco2e
        abs_reduction = base_total - current_total
        pct_reduction = (abs_reduction / base_total) * 100.0 if base_total > 0 else 0.0
        ann_rate = (pct_reduction / years_elapsed) if years_elapsed > 0 else 0.0

        cumulative = CumulativeProgress(
            base_year=config.base_year,
            review_year=config.review_year,
            base_year_total_tco2e=round(base_total, 4),
            current_total_tco2e=round(current_total, 4),
            absolute_reduction_tco2e=round(abs_reduction, 4),
            percentage_reduction=round(pct_reduction, 2),
            years_elapsed=years_elapsed,
            annualised_reduction_rate_pct=round(ann_rate, 2),
        )

        # Intensity metrics
        intensities: List[IntensityMetric] = []
        if current.revenue_usd and current.revenue_usd > 0:
            curr_intensity = current.total_tco2e / current.revenue_usd * 1_000_000
            base_intensity = (base.total_tco2e / base.revenue_usd * 1_000_000) if base.revenue_usd and base.revenue_usd > 0 else 0
            change = ((curr_intensity - base_intensity) / base_intensity * 100) if base_intensity > 0 else 0
            intensities.append(IntensityMetric(
                metric_name="Carbon Intensity (Revenue)",
                unit="tCO2e per $M revenue",
                base_year_value=round(base_intensity, 2),
                current_value=round(curr_intensity, 2),
                change_pct=round(change, 2),
            ))

        if current.employee_count and current.employee_count > 0:
            curr_per_emp = current.total_tco2e / current.employee_count
            base_per_emp = (base.total_tco2e / base.employee_count) if base.employee_count and base.employee_count > 0 else 0
            change = ((curr_per_emp - base_per_emp) / base_per_emp * 100) if base_per_emp > 0 else 0
            intensities.append(IntensityMetric(
                metric_name="Carbon Intensity (Employee)",
                unit="tCO2e per employee",
                base_year_value=round(base_per_emp, 2),
                current_value=round(curr_per_emp, 2),
                change_pct=round(change, 2),
            ))

        if current.floor_area_sqm and current.floor_area_sqm > 0:
            curr_per_sqm = current.total_tco2e / current.floor_area_sqm
            base_per_sqm = (base.total_tco2e / base.floor_area_sqm) if base.floor_area_sqm and base.floor_area_sqm > 0 else 0
            change = ((curr_per_sqm - base_per_sqm) / base_per_sqm * 100) if base_per_sqm > 0 else 0
            intensities.append(IntensityMetric(
                metric_name="Carbon Intensity (Floor Area)",
                unit="tCO2e per sqm",
                base_year_value=round(base_per_sqm, 4),
                current_value=round(curr_per_sqm, 4),
                change_pct=round(change, 2),
            ))

        self._progress = ProgressSummary(
            review_year=config.review_year,
            yoy_changes=yoy_changes,
            cumulative_progress=cumulative,
            intensity_metrics=intensities,
        )

        outputs["cumulative_reduction_pct"] = cumulative.percentage_reduction
        outputs["annualised_rate_pct"] = cumulative.annualised_reduction_rate_pct
        outputs["yoy_comparisons"] = len(yoy_changes)
        outputs["intensity_metrics"] = len(intensities)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Progress calc: cumulative %.1f%% reduction, annualised %.2f%%/yr",
            pct_reduction, ann_rate,
        )
        return PhaseResult(
            phase_name="progress_calc",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _calc_yoy(self, prev: AnnualEmissions, curr: AnnualEmissions) -> YearOverYearChange:
        """Calculate year-over-year change between two annual records."""
        prev_total = prev.total_tco2e or 1.0
        abs_change = curr.total_tco2e - prev.total_tco2e
        pct_change = (abs_change / prev_total) * 100.0 if prev_total > 0 else 0.0

        s1_change = ((curr.scope1_tco2e - prev.scope1_tco2e) / prev.scope1_tco2e * 100) if prev.scope1_tco2e > 0 else 0
        s2_change = ((curr.scope2_location_tco2e - prev.scope2_location_tco2e) / prev.scope2_location_tco2e * 100) if prev.scope2_location_tco2e > 0 else 0
        s3_change = ((curr.scope3_tco2e - prev.scope3_tco2e) / prev.scope3_tco2e * 100) if prev.scope3_tco2e > 0 else 0

        return YearOverYearChange(
            from_year=prev.year,
            to_year=curr.year,
            absolute_change_tco2e=round(abs_change, 4),
            percentage_change=round(pct_change, 2),
            scope1_change_pct=round(s1_change, 2),
            scope2_change_pct=round(s2_change, 2),
            scope3_change_pct=round(s3_change, 2),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Gap Analysis
    # -------------------------------------------------------------------------

    async def _phase_gap_analysis(self, config: ProgressReviewConfig) -> PhaseResult:
        """Compare actual trajectory to target pathway, assess RAG status."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        current = config.current_year_emissions
        base = config.base_year_emissions
        base_total = base.total_tco2e or 1.0

        # Determine target for review year
        target_for_year = self._get_target_for_year(config, config.review_year)

        # Calculate gap
        actual = current.total_tco2e
        gap = actual - target_for_year  # positive = behind target
        gap_pct = (gap / target_for_year * 100) if target_for_year > 0 else 0

        # Determine trend
        trend = self._determine_trend(config)

        # RAG status
        threshold = config.alert_threshold_pct
        if gap <= 0:
            rag = RAGStatus.GREEN
        elif gap_pct <= threshold:
            rag = RAGStatus.AMBER
        else:
            rag = RAGStatus.RED

        # Projection: extrapolate current rate to near-term target year
        cumulative = self._progress.cumulative_progress
        ann_rate = cumulative.annualised_reduction_rate_pct
        years_to_nt = max(config.near_term_target_year - config.review_year, 1)
        projected_additional_pct = ann_rate * years_to_nt
        projected_total_reduction = cumulative.percentage_reduction + projected_additional_pct
        projected_tco2e_at_nt = base_total * (1.0 - projected_total_reduction / 100.0)
        projected_tco2e_at_nt = max(projected_tco2e_at_nt, 0.0)

        # On track check
        nt_target = base_total * (1.0 - config.near_term_target_reduction_pct / 100.0)
        on_track = projected_tco2e_at_nt <= nt_target

        # Additional reduction needed
        additional_needed = max(actual - nt_target, 0.0)
        additional_ann_rate = (additional_needed / actual * 100 / years_to_nt) if actual > 0 and years_to_nt > 0 else 0

        self._gap = GapAnalysisResult(
            rag_status=rag,
            target_tco2e_for_year=round(target_for_year, 4),
            actual_tco2e=round(actual, 4),
            gap_tco2e=round(gap, 4),
            gap_pct=round(gap_pct, 2),
            trend=trend,
            projected_tco2e_at_target_year=round(projected_tco2e_at_nt, 4),
            on_track_for_near_term=on_track,
            additional_reduction_needed_tco2e=round(additional_needed, 4),
            additional_annual_rate_needed_pct=round(additional_ann_rate, 2),
        )

        # Build trend data
        self._trend_data = self._build_trend_data(config)

        outputs["rag_status"] = rag.value
        outputs["target_tco2e"] = target_for_year
        outputs["actual_tco2e"] = actual
        outputs["gap_tco2e"] = round(gap, 4)
        outputs["on_track"] = on_track
        outputs["trend"] = trend.value

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Gap analysis: RAG=%s, gap=%.1f tCO2e (%.1f%%), on_track=%s",
            rag.value, gap, gap_pct, on_track,
        )
        return PhaseResult(
            phase_name="gap_analysis",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _get_target_for_year(self, config: ProgressReviewConfig, year: int) -> float:
        """Get target emissions for a given year from pathway or linear interpolation."""
        # Check explicit pathway
        for pt in config.target_pathway:
            if pt.year == year:
                return pt.target_tco2e

        # Linear interpolation using annual reduction rate
        base_total = config.base_year_emissions.total_tco2e
        years = year - config.base_year
        reduction_pct = config.annual_reduction_rate_pct * years
        return max(base_total * (1.0 - reduction_pct / 100.0), 0.0)

    def _determine_trend(self, config: ProgressReviewConfig) -> TrendDirection:
        """Determine emissions trend from historical data."""
        all_data = sorted(
            config.historical_emissions + [config.current_year_emissions],
            key=lambda e: e.year,
        )
        if len(all_data) < 2:
            return TrendDirection.FLAT

        # Use last 3 years if available
        recent = all_data[-3:] if len(all_data) >= 3 else all_data
        totals = [e.total_tco2e for e in recent]

        if len(totals) < 2:
            return TrendDirection.FLAT

        # Simple linear trend
        changes = [totals[i] - totals[i - 1] for i in range(1, len(totals))]
        avg_change = sum(changes) / len(changes)
        threshold = totals[0] * 0.005  # 0.5% of first value

        if avg_change < -threshold:
            return TrendDirection.DECREASING
        elif avg_change > threshold:
            return TrendDirection.INCREASING
        else:
            return TrendDirection.FLAT

    def _build_trend_data(self, config: ProgressReviewConfig) -> List[TrendDataPoint]:
        """Build trend chart data including actuals, targets, and projections."""
        points: List[TrendDataPoint] = []

        # Historical actuals
        for em in sorted(config.historical_emissions, key=lambda e: e.year):
            target = self._get_target_for_year(config, em.year)
            points.append(TrendDataPoint(
                year=em.year,
                actual_tco2e=em.total_tco2e,
                target_tco2e=round(target, 4),
            ))

        # Current year
        current = config.current_year_emissions
        target_now = self._get_target_for_year(config, current.year)
        points.append(TrendDataPoint(
            year=current.year,
            actual_tco2e=current.total_tco2e,
            target_tco2e=round(target_now, 4),
        ))

        # Projections (if enabled)
        if config.include_projections:
            ann_rate = self._progress.cumulative_progress.annualised_reduction_rate_pct
            last_actual = current.total_tco2e
            for yr in range(current.year + 1, config.near_term_target_year + 1):
                projected = last_actual * (1.0 - ann_rate / 100.0)
                projected = max(projected, 0.0)
                target = self._get_target_for_year(config, yr)
                points.append(TrendDataPoint(
                    year=yr,
                    projected_tco2e=round(projected, 4),
                    target_tco2e=round(target, 4),
                ))
                last_actual = projected

        return points

    # -------------------------------------------------------------------------
    # Phase 4: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(self, config: ProgressReviewConfig) -> PhaseResult:
        """Generate progress report with recommendations and corrective actions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._recommendations = self._generate_corrective_actions(config)

        outputs["recommendation_count"] = len(self._recommendations)
        outputs["rag_status"] = self._gap.rag_status.value
        outputs["cumulative_reduction_pct"] = self._progress.cumulative_progress.percentage_reduction
        outputs["trend"] = self._gap.trend.value
        outputs["trend_data_points"] = len(self._trend_data)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Report generated: %d corrective actions", len(self._recommendations))
        return PhaseResult(
            phase_name="report_generation",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _generate_corrective_actions(self, config: ProgressReviewConfig) -> List[CorrectiveAction]:
        """Generate corrective action recommendations based on gap analysis."""
        actions: List[CorrectiveAction] = []
        gap = self._gap
        progress = self._progress.cumulative_progress

        if gap.rag_status == RAGStatus.GREEN:
            actions.append(CorrectiveAction(
                priority=1,
                action="Maintain current trajectory. Review reduction roadmap for additional opportunities.",
                expected_impact_tco2e=0.0,
                timeframe="Ongoing",
            ))
            return actions

        if gap.rag_status == RAGStatus.RED:
            actions.append(CorrectiveAction(
                priority=1,
                action=(
                    f"URGENT: Emissions are {gap.gap_tco2e:.0f} tCO2e above target. "
                    f"Accelerate reduction rate to {gap.additional_annual_rate_needed_pct:.1f}%/yr."
                ),
                expected_impact_tco2e=gap.gap_tco2e,
                timeframe="Immediate (next 12 months)",
            ))

        if gap.rag_status in {RAGStatus.RED, RAGStatus.AMBER}:
            # Scope-specific recommendations
            current = config.current_year_emissions
            base = config.base_year_emissions
            total = current.total_tco2e or 1.0

            if current.scope2_location_tco2e > 0.2 * total:
                actions.append(CorrectiveAction(
                    priority=2,
                    action="Scope 2 contributes >20% of total. Accelerate renewable energy procurement (PPAs, GoOs).",
                    expected_impact_tco2e=round(current.scope2_location_tco2e * 0.3, 2),
                    timeframe="Short-term (0-2 years)",
                ))

            if current.scope1_tco2e > 0.2 * total:
                actions.append(CorrectiveAction(
                    priority=3,
                    action="Scope 1 contributes >20% of total. Accelerate electrification and fuel switching.",
                    expected_impact_tco2e=round(current.scope1_tco2e * 0.2, 2),
                    timeframe="Medium-term (2-5 years)",
                ))

            if current.scope3_tco2e > 0.4 * total:
                actions.append(CorrectiveAction(
                    priority=4,
                    action="Scope 3 dominates emissions. Intensify supplier engagement and green procurement.",
                    expected_impact_tco2e=round(current.scope3_tco2e * 0.1, 2),
                    timeframe="Medium-term (2-5 years)",
                ))

        if gap.trend == TrendDirection.INCREASING:
            actions.append(CorrectiveAction(
                priority=1,
                action="Emissions trend is INCREASING. Conduct root cause analysis immediately.",
                expected_impact_tco2e=gap.gap_tco2e,
                timeframe="Immediate",
            ))

        if not gap.on_track_for_near_term:
            actions.append(CorrectiveAction(
                priority=2,
                action=(
                    f"Not on track for {config.near_term_target_year} near-term target. "
                    f"Additional {gap.additional_reduction_needed_tco2e:.0f} tCO2e reduction required."
                ),
                expected_impact_tco2e=gap.additional_reduction_needed_tco2e,
                timeframe=f"By {config.near_term_target_year}",
            ))

        # Data quality recommendation
        if config.current_year_emissions.data_quality_score > 3.0:
            actions.append(CorrectiveAction(
                priority=5,
                action="Improve data quality to reduce measurement uncertainty and strengthen reporting credibility.",
                expected_impact_tco2e=0.0,
                timeframe="Ongoing",
            ))

        actions.sort(key=lambda a: a.priority)
        return actions
