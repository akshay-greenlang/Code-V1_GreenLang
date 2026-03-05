"""
Progress Tracking Engine -- Annual Target Progress and Variance Analysis

Implements annual progress recording, variance analysis, RAG (Red/Amber/Green)
status assessment, cumulative reduction tracking, projected achievement
estimation, trend analysis, multi-year summaries, and MRV data integration
for SBTi target progress monitoring.

All numeric calculations are deterministic (zero-hallucination).

Reference:
    - SBTi Criteria and Recommendations v5.1 (2023), Criterion C12
    - SBTi Corporate Net-Zero Standard v1.2, Section 8 (Progress Reporting)
    - GHG Protocol Corporate Standard, Chapter 9 (Tracking Emissions Over Time)

Example:
    >>> from services.config import SBTiAppConfig
    >>> engine = ProgressTrackingEngine(SBTiAppConfig())
    >>> record = engine.record_progress("org-1", "tgt-1", 2023, 95000.0)
    >>> print(record.on_track)
    True
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    SBTiAppConfig,
)
from .models import (
    ProgressRecord,
    ProgressSummary,
    Target,
    EmissionsInventory,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class RAGStatus(BaseModel):
    """Red/Amber/Green status for target progress."""

    target_id: str = Field(...)
    year: int = Field(...)
    status: str = Field(default="amber", description="red, amber, green")
    variance_pct: float = Field(default=0.0)
    message: str = Field(default="")
    threshold_green_pct: float = Field(default=0.0)
    threshold_amber_pct: float = Field(default=5.0)
    assessed_at: datetime = Field(default_factory=_now)


class TrendAnalysis(BaseModel):
    """Trend analysis across multiple reporting years."""

    target_id: str = Field(...)
    years_analyzed: int = Field(default=0)
    trend_direction: str = Field(
        default="flat", description="improving, deteriorating, flat",
    )
    avg_annual_reduction_pct: float = Field(default=0.0)
    required_annual_reduction_pct: float = Field(default=0.0)
    acceleration_needed: bool = Field(default=False)
    projected_target_year_emissions: float = Field(default=0.0)
    projected_achievement_pct: float = Field(default=0.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    yearly_reductions: List[Dict[str, float]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ProgressDashboard(BaseModel):
    """Aggregated progress dashboard for an organization."""

    org_id: str = Field(...)
    total_targets: int = Field(default=0)
    on_track_count: int = Field(default=0)
    off_track_count: int = Field(default=0)
    no_data_count: int = Field(default=0)
    overall_reduction_pct: float = Field(default=0.0)
    latest_year: Optional[int] = Field(None)
    target_summaries: List[Dict[str, Any]] = Field(default_factory=list)
    alerts: List[Dict[str, str]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_now)


class AnnualComparison(BaseModel):
    """Year-over-year emissions comparison."""

    target_id: str = Field(...)
    year_1: int = Field(...)
    year_2: int = Field(...)
    emissions_year_1: float = Field(default=0.0, ge=0.0)
    emissions_year_2: float = Field(default=0.0, ge=0.0)
    absolute_change: float = Field(default=0.0)
    percentage_change: float = Field(default=0.0)
    direction: str = Field(
        default="flat", description="decrease, increase, flat",
    )


class ProgressReport(BaseModel):
    """Comprehensive progress report for a target."""

    target_id: str = Field(...)
    org_id: str = Field(...)
    base_year: int = Field(default=2020)
    target_year: int = Field(default=2030)
    base_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    records: List[Dict[str, Any]] = Field(default_factory=list)
    trend: Dict[str, Any] = Field(default_factory=dict)
    rag_status: str = Field(default="amber")
    cumulative_reduction_pct: float = Field(default=0.0)
    years_remaining: int = Field(default=0)
    required_annual_rate_remaining: float = Field(default=0.0)
    generated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# ProgressTrackingEngine
# ---------------------------------------------------------------------------

class ProgressTrackingEngine:
    """
    Annual progress tracking and variance analysis engine per SBTi C12.

    Records annual emissions data, calculates variance against the reduction
    pathway, assigns RAG status, performs trend analysis, projects target
    year achievement, and generates progress dashboards and reports.

    Attributes:
        config: Application configuration.
        _targets: In-memory target store keyed by target_id.
        _records: In-memory progress records keyed by target_id.
        _inventories: In-memory inventories keyed by org_id.

    Example:
        >>> engine = ProgressTrackingEngine(SBTiAppConfig())
        >>> record = engine.record_progress("org-1", "tgt-1", 2023, 95000.0)
    """

    # RAG thresholds: green = on/ahead of target, amber = within 5%,
    # red = more than 5% behind
    RAG_AMBER_THRESHOLD_PCT: float = 5.0
    RAG_RED_THRESHOLD_PCT: float = 10.0

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """Initialize the ProgressTrackingEngine."""
        self.config = config or SBTiAppConfig()
        self._targets: Dict[str, Dict[str, Any]] = {}
        self._records: Dict[str, List[ProgressRecord]] = {}
        self._inventories: Dict[str, EmissionsInventory] = {}
        logger.info("ProgressTrackingEngine initialized")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_target(self, target_data: Dict[str, Any]) -> None:
        """
        Register a target for progress tracking.

        Args:
            target_data: Dict with keys: id, org_id, base_year, target_year,
                        base_year_emissions, target_reduction_pct,
                        annual_reduction_rate.
        """
        target_id = target_data.get("id", _new_id())
        self._targets[target_id] = target_data
        logger.info("Registered target %s for progress tracking", target_id)

    def register_inventory(self, inventory: EmissionsInventory) -> None:
        """Register an emissions inventory for MRV integration."""
        self._inventories[inventory.org_id] = inventory

    # ------------------------------------------------------------------
    # Record Progress
    # ------------------------------------------------------------------

    def record_progress(
        self,
        org_id: str,
        target_id: str,
        year: int,
        actual_total_tco2e: float,
        actual_scope1: float = 0.0,
        actual_scope2: float = 0.0,
        actual_scope3: float = 0.0,
        actual_intensity: Optional[float] = None,
        data_quality: str = "estimated",
    ) -> ProgressRecord:
        """
        Record annual progress against a target.

        Calculates the pathway expected value for the given year, computes
        variance, and determines on-track status.

        Args:
            org_id: Organization identifier.
            target_id: Target identifier.
            year: Reporting year.
            actual_total_tco2e: Actual total emissions in tCO2e.
            actual_scope1: Scope 1 emissions (optional breakdown).
            actual_scope2: Scope 2 emissions (optional breakdown).
            actual_scope3: Scope 3 emissions (optional breakdown).
            actual_intensity: Actual intensity value (for SDA targets).
            data_quality: Data quality tier.

        Returns:
            ProgressRecord with variance and on-track assessment.
        """
        start = datetime.utcnow()

        target = self._targets.get(target_id, {})
        base_year = target.get("base_year", 2020)
        target_year = target.get("target_year", 2030)
        base_emissions = target.get("base_year_emissions", 0.0)
        reduction_pct = target.get("target_reduction_pct", 42.0)

        # Calculate expected emissions for this year
        expected = self._calculate_expected_emissions(
            base_emissions, base_year, target_year, reduction_pct, year,
        )

        # Calculate cumulative reduction from base year
        cumulative = 0.0
        if base_emissions > 0:
            cumulative = ((base_emissions - actual_total_tco2e) / base_emissions) * 100.0
            cumulative = max(cumulative, 0.0)

        record = ProgressRecord(
            tenant_id="default",
            org_id=org_id,
            target_id=target_id,
            year=year,
            actual_scope1_tco2e=Decimal(str(actual_scope1)),
            actual_scope2_tco2e=Decimal(str(actual_scope2)),
            actual_scope3_tco2e=Decimal(str(actual_scope3)),
            actual_total_tco2e=Decimal(str(actual_total_tco2e)),
            actual_intensity_value=(
                Decimal(str(actual_intensity)) if actual_intensity is not None else None
            ),
            pathway_expected_tco2e=Decimal(str(round(expected, 2))),
            cumulative_reduction_pct=Decimal(str(round(cumulative, 2))),
            data_quality=data_quality,
        )

        self._records.setdefault(target_id, []).append(record)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Progress recorded for target %s year %d: actual=%.0f expected=%.0f "
            "on_track=%s in %.1f ms",
            target_id, year, actual_total_tco2e, expected,
            record.on_track, elapsed_ms,
        )
        return record

    # ------------------------------------------------------------------
    # RAG Status
    # ------------------------------------------------------------------

    def assess_rag_status(self, target_id: str, year: int) -> RAGStatus:
        """
        Determine Red/Amber/Green status for a target in a given year.

        Green: at or ahead of pathway (variance <= 0).
        Amber: behind pathway by up to 5%.
        Red: behind pathway by more than 5%.

        Args:
            target_id: Target identifier.
            year: Reporting year.

        Returns:
            RAGStatus with color and explanation.
        """
        records = self._records.get(target_id, [])
        year_record = next((r for r in records if r.year == year), None)

        if year_record is None:
            return RAGStatus(
                target_id=target_id,
                year=year,
                status="amber",
                message="No progress data available for this year.",
            )

        variance_pct = float(year_record.variance_pct)

        if variance_pct <= 0:
            status = "green"
            message = f"On track: emissions are {abs(variance_pct):.1f}% below pathway."
        elif variance_pct <= self.RAG_AMBER_THRESHOLD_PCT:
            status = "amber"
            message = f"Slightly behind: emissions are {variance_pct:.1f}% above pathway."
        else:
            status = "red"
            message = f"Off track: emissions are {variance_pct:.1f}% above pathway."

        return RAGStatus(
            target_id=target_id,
            year=year,
            status=status,
            variance_pct=round(variance_pct, 2),
            message=message,
            threshold_green_pct=0.0,
            threshold_amber_pct=self.RAG_AMBER_THRESHOLD_PCT,
        )

    # ------------------------------------------------------------------
    # Trend Analysis
    # ------------------------------------------------------------------

    def analyze_trend(self, target_id: str) -> TrendAnalysis:
        """
        Perform trend analysis across all recorded years for a target.

        Calculates the average annual reduction rate achieved, determines
        the trend direction, projects target year emissions, and estimates
        projected achievement percentage.

        Args:
            target_id: Target identifier.

        Returns:
            TrendAnalysis with projections and confidence.
        """
        start = datetime.utcnow()
        records = sorted(
            self._records.get(target_id, []),
            key=lambda r: r.year,
        )
        target = self._targets.get(target_id, {})
        base_year = target.get("base_year", 2020)
        target_year = target.get("target_year", 2030)
        base_emissions = target.get("base_year_emissions", 0.0)
        reduction_pct = target.get("target_reduction_pct", 42.0)

        if len(records) < 2 or base_emissions <= 0:
            return TrendAnalysis(
                target_id=target_id,
                years_analyzed=len(records),
                required_annual_reduction_pct=(
                    reduction_pct / (target_year - base_year)
                    if target_year > base_year else 0.0
                ),
                confidence=0.3,
            )

        # Year-over-year reductions
        yearly_reductions: List[Dict[str, float]] = []
        for i in range(1, len(records)):
            prev = float(records[i - 1].actual_total_tco2e)
            curr = float(records[i].actual_total_tco2e)
            if prev > 0:
                pct = ((prev - curr) / prev) * 100.0
            else:
                pct = 0.0
            yearly_reductions.append({
                "from_year": records[i - 1].year,
                "to_year": records[i].year,
                "reduction_pct": round(pct, 2),
            })

        avg_reduction = (
            sum(r["reduction_pct"] for r in yearly_reductions) / len(yearly_reductions)
            if yearly_reductions else 0.0
        )

        required_rate = (
            reduction_pct / (target_year - base_year)
            if target_year > base_year else 0.0
        )

        # Trend direction
        if avg_reduction > required_rate:
            direction = "improving"
        elif avg_reduction > 0:
            direction = "improving"  # Positive but may need acceleration
        elif avg_reduction < -1.0:
            direction = "deteriorating"
        else:
            direction = "flat"

        acceleration_needed = avg_reduction < required_rate

        # Project target year emissions using current trend
        latest = float(records[-1].actual_total_tco2e)
        years_remaining = target_year - records[-1].year
        if avg_reduction > 0 and years_remaining > 0:
            projected = latest * ((1 - avg_reduction / 100.0) ** years_remaining)
        else:
            projected = latest

        target_emissions = base_emissions * (1 - reduction_pct / 100.0)
        achievement = 0.0
        if base_emissions > target_emissions:
            achieved_reduction = base_emissions - projected
            required_reduction = base_emissions - target_emissions
            achievement = (achieved_reduction / required_reduction) * 100.0
            achievement = min(max(achievement, 0.0), 200.0)

        # Confidence based on data points
        confidence = min(0.3 + len(records) * 0.1, 0.95)

        provenance = _sha256(
            f"trend:{target_id}:{avg_reduction}:{projected}"
        )

        result = TrendAnalysis(
            target_id=target_id,
            years_analyzed=len(records),
            trend_direction=direction,
            avg_annual_reduction_pct=round(avg_reduction, 2),
            required_annual_reduction_pct=round(required_rate, 2),
            acceleration_needed=acceleration_needed,
            projected_target_year_emissions=round(projected, 2),
            projected_achievement_pct=round(achievement, 2),
            confidence=round(confidence, 2),
            yearly_reductions=yearly_reductions,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Trend analysis for target %s: avg_rate=%.2f%%/yr, direction=%s, "
            "projected=%.0f in %.1f ms",
            target_id, avg_reduction, direction, projected, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Annual Comparison
    # ------------------------------------------------------------------

    def compare_years(
        self, target_id: str, year_1: int, year_2: int,
    ) -> AnnualComparison:
        """
        Compare emissions between two reporting years.

        Args:
            target_id: Target identifier.
            year_1: First year.
            year_2: Second year.

        Returns:
            AnnualComparison with change analysis.
        """
        records = self._records.get(target_id, [])
        rec_1 = next((r for r in records if r.year == year_1), None)
        rec_2 = next((r for r in records if r.year == year_2), None)

        e1 = float(rec_1.actual_total_tco2e) if rec_1 else 0.0
        e2 = float(rec_2.actual_total_tco2e) if rec_2 else 0.0

        absolute_change = e2 - e1
        pct_change = ((e2 - e1) / e1 * 100.0) if e1 > 0 else 0.0

        if pct_change < -0.5:
            direction = "decrease"
        elif pct_change > 0.5:
            direction = "increase"
        else:
            direction = "flat"

        return AnnualComparison(
            target_id=target_id,
            year_1=year_1,
            year_2=year_2,
            emissions_year_1=round(e1, 2),
            emissions_year_2=round(e2, 2),
            absolute_change=round(absolute_change, 2),
            percentage_change=round(pct_change, 2),
            direction=direction,
        )

    # ------------------------------------------------------------------
    # Progress Summary
    # ------------------------------------------------------------------

    def get_progress_summary(self, target_id: str) -> ProgressSummary:
        """
        Generate a comprehensive progress summary for a target.

        Args:
            target_id: Target identifier.

        Returns:
            ProgressSummary with cumulative reduction and on-track status.
        """
        target = self._targets.get(target_id, {})
        records = sorted(
            self._records.get(target_id, []),
            key=lambda r: r.year,
        )

        base_year = target.get("base_year", 2020)
        target_year = target.get("target_year", 2030)
        base_emissions = target.get("base_year_emissions", 0.0)
        reduction_pct = target.get("target_reduction_pct", 42.0)

        latest_year = records[-1].year if records else None
        latest_emissions = float(records[-1].actual_total_tco2e) if records else None

        # Cumulative reduction
        cumulative = 0.0
        if base_emissions > 0 and latest_emissions is not None:
            cumulative = ((base_emissions - latest_emissions) / base_emissions) * 100.0
            cumulative = max(cumulative, 0.0)

        # Current annual rate (based on latest data point)
        current_rate = 0.0
        if latest_year and latest_year > base_year and base_emissions > 0:
            years_elapsed = latest_year - base_year
            current_rate = cumulative / years_elapsed

        # Required annual rate for remaining years
        required_rate = 0.0
        if latest_year and target_year > latest_year and base_emissions > 0:
            remaining_reduction = reduction_pct - cumulative
            years_remaining = target_year - latest_year
            if years_remaining > 0:
                required_rate = remaining_reduction / years_remaining

        # Projected target year emissions
        projected = None
        if latest_emissions is not None and current_rate > 0:
            years_remaining = target_year - (latest_year or base_year)
            projected = latest_emissions * ((1 - current_rate / 100.0) ** years_remaining)

        # Achievement percentage
        projected_achievement = 0.0
        if projected is not None and base_emissions > 0:
            target_emissions = base_emissions * (1 - reduction_pct / 100.0)
            if base_emissions > target_emissions:
                achieved = base_emissions - projected
                required = base_emissions - target_emissions
                projected_achievement = (achieved / required) * 100.0

        on_track = cumulative >= (
            reduction_pct * (
                (latest_year - base_year) / (target_year - base_year)
                if target_year > base_year and latest_year else 0
            )
        ) if latest_year else False

        summary = ProgressSummary(
            target_id=target_id,
            base_year=base_year,
            target_year=target_year,
            base_emissions_tco2e=Decimal(str(round(base_emissions, 2))),
            latest_year=latest_year,
            latest_emissions_tco2e=(
                Decimal(str(round(latest_emissions, 2)))
                if latest_emissions is not None else None
            ),
            cumulative_reduction_pct=Decimal(str(round(cumulative, 2))),
            required_reduction_pct=Decimal(str(round(reduction_pct, 2))),
            current_annual_rate=Decimal(str(round(current_rate, 2))),
            required_annual_rate=Decimal(str(round(required_rate, 2))),
            projected_target_year_emissions=(
                Decimal(str(round(projected, 2))) if projected is not None else None
            ),
            projected_achievement_pct=Decimal(str(round(projected_achievement, 2))),
            on_track=on_track,
            years_tracked=len(records),
            records=records,
        )

        logger.info(
            "Progress summary for target %s: cumulative=%.1f%%, on_track=%s",
            target_id, cumulative, on_track,
        )
        return summary

    # ------------------------------------------------------------------
    # Progress Dashboard
    # ------------------------------------------------------------------

    def get_progress_dashboard(self, org_id: str) -> ProgressDashboard:
        """
        Generate an aggregated progress dashboard for an organization.

        Summarizes all targets with their progress status, on-track
        determination, and any alerts.

        Args:
            org_id: Organization identifier.

        Returns:
            ProgressDashboard with target summaries and alerts.
        """
        start = datetime.utcnow()

        org_targets = {
            tid: t for tid, t in self._targets.items()
            if t.get("org_id") == org_id
        }

        on_track = 0
        off_track = 0
        no_data = 0
        target_summaries: List[Dict[str, Any]] = []
        alerts: List[Dict[str, str]] = []
        latest_year: Optional[int] = None
        total_base = 0.0
        total_latest = 0.0

        for tid, target in org_targets.items():
            records = self._records.get(tid, [])
            if not records:
                no_data += 1
                target_summaries.append({
                    "target_id": tid,
                    "status": "no_data",
                    "message": "No progress data recorded.",
                })
                continue

            latest_record = max(records, key=lambda r: r.year)
            base_emissions = target.get("base_year_emissions", 0.0)

            if latest_record.on_track:
                on_track += 1
                status = "on_track"
            else:
                off_track += 1
                status = "off_track"
                alerts.append({
                    "severity": "high",
                    "message": (
                        f"Target {tid} is off track in {latest_record.year}. "
                        f"Variance: {float(latest_record.variance_pct):.1f}%."
                    ),
                })

            if latest_year is None or latest_record.year > latest_year:
                latest_year = latest_record.year

            total_base += base_emissions
            total_latest += float(latest_record.actual_total_tco2e)

            target_summaries.append({
                "target_id": tid,
                "status": status,
                "latest_year": latest_record.year,
                "latest_emissions": float(latest_record.actual_total_tco2e),
                "variance_pct": float(latest_record.variance_pct),
                "on_track": latest_record.on_track,
            })

        overall_reduction = 0.0
        if total_base > 0:
            overall_reduction = ((total_base - total_latest) / total_base) * 100.0

        dashboard = ProgressDashboard(
            org_id=org_id,
            total_targets=len(org_targets),
            on_track_count=on_track,
            off_track_count=off_track,
            no_data_count=no_data,
            overall_reduction_pct=round(overall_reduction, 2),
            latest_year=latest_year,
            target_summaries=target_summaries,
            alerts=alerts,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Progress dashboard for org %s: %d targets, %d on track, "
            "%d off track in %.1f ms",
            org_id, len(org_targets), on_track, off_track, elapsed_ms,
        )
        return dashboard

    # ------------------------------------------------------------------
    # Progress Report
    # ------------------------------------------------------------------

    def generate_progress_report(self, target_id: str) -> ProgressReport:
        """
        Generate a comprehensive progress report for a target.

        Combines progress records, trend analysis, RAG status, and
        remaining reduction requirements into a single report.

        Args:
            target_id: Target identifier.

        Returns:
            ProgressReport with all progress data.
        """
        start = datetime.utcnow()
        target = self._targets.get(target_id, {})
        records = sorted(
            self._records.get(target_id, []),
            key=lambda r: r.year,
        )

        org_id = target.get("org_id", "")
        base_year = target.get("base_year", 2020)
        target_year = target.get("target_year", 2030)
        base_emissions = target.get("base_year_emissions", 0.0)
        reduction_pct = target.get("target_reduction_pct", 42.0)

        # Trend analysis
        trend = self.analyze_trend(target_id)

        # Latest RAG
        rag = "amber"
        if records:
            rag_result = self.assess_rag_status(target_id, records[-1].year)
            rag = rag_result.status

        # Cumulative reduction
        cumulative = 0.0
        if records and base_emissions > 0:
            latest = float(records[-1].actual_total_tco2e)
            cumulative = ((base_emissions - latest) / base_emissions) * 100.0

        # Years remaining and required rate
        latest_year = records[-1].year if records else base_year
        years_remaining = max(target_year - latest_year, 0)
        remaining_reduction = reduction_pct - cumulative
        required_rate = (
            remaining_reduction / years_remaining
            if years_remaining > 0 else 0.0
        )

        provenance = _sha256(
            f"progress_report:{target_id}:{cumulative}:{rag}"
        )

        report = ProgressReport(
            target_id=target_id,
            org_id=org_id,
            base_year=base_year,
            target_year=target_year,
            base_emissions_tco2e=round(base_emissions, 2),
            records=[r.model_dump() for r in records],
            trend=trend.model_dump(),
            rag_status=rag,
            cumulative_reduction_pct=round(cumulative, 2),
            years_remaining=years_remaining,
            required_annual_rate_remaining=round(required_rate, 2),
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Progress report for target %s: RAG=%s, cumulative=%.1f%%, "
            "%d years remaining in %.1f ms",
            target_id, rag, cumulative, years_remaining, elapsed_ms,
        )
        return report

    # ------------------------------------------------------------------
    # Retrieve Records
    # ------------------------------------------------------------------

    def get_records(self, target_id: str) -> List[ProgressRecord]:
        """
        Get all progress records for a target, sorted by year.

        Args:
            target_id: Target identifier.

        Returns:
            List of ProgressRecord sorted by year ascending.
        """
        records = self._records.get(target_id, [])
        return sorted(records, key=lambda r: r.year)

    def get_latest_record(
        self, target_id: str,
    ) -> Optional[ProgressRecord]:
        """
        Get the most recent progress record for a target.

        Args:
            target_id: Target identifier.

        Returns:
            Latest ProgressRecord or None.
        """
        records = self._records.get(target_id, [])
        if not records:
            return None
        return max(records, key=lambda r: r.year)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_expected_emissions(
        self,
        base_emissions: float,
        base_year: int,
        target_year: int,
        reduction_pct: float,
        current_year: int,
    ) -> float:
        """
        Calculate expected emissions for a given year on a linear pathway.

        Uses linear interpolation between base year emissions and
        target year emissions to determine the expected value.

        Args:
            base_emissions: Base year emissions in tCO2e.
            base_year: Base year.
            target_year: Target year.
            reduction_pct: Total reduction percentage.
            current_year: Year to calculate expected emissions for.

        Returns:
            Expected emissions in tCO2e.
        """
        if base_emissions <= 0 or target_year <= base_year:
            return base_emissions

        target_emissions = base_emissions * (1 - reduction_pct / 100.0)
        total_years = target_year - base_year
        elapsed = min(max(current_year - base_year, 0), total_years)
        progress = elapsed / total_years

        return base_emissions + (target_emissions - base_emissions) * progress
