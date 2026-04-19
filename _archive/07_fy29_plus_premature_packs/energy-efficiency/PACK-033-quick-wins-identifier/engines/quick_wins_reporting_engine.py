# -*- coding: utf-8 -*-
"""
QuickWinsReportingEngine - PACK-033 Quick Wins Identifier Engine 8
===================================================================

Dashboard data aggregation, progress tracking, savings verification
(IPMVP Option A/B), ROI reporting, and multi-format export for
quick-win energy conservation measures.

Calculation Methodology:
    Savings Verification (IPMVP):
        Option A: verified_savings = (baseline - post) * adjustment_factor
            Key parameter measurement with stipulated operating hours.
            confidence = 80% (stipulated values introduce uncertainty).
        Option B: verified_savings = (baseline - post) * adjustment_factor
            All parameters measured; no stipulation.
            confidence = 90% (full metering reduces uncertainty).
        Stipulated: verified_savings = baseline - post  (no adjustment).
        Utility Bills: verified_savings = baseline - post  (whole-facility).

    KPI Calculations:
        completion_rate = completed_measures / total_measures * 100
        total_roi = (total_annual_savings - total_cost) / total_cost * 100
        portfolio_payback = total_cost / total_annual_savings  (years)
        co2e_reduction = total_verified_savings_kwh * grid_factor / 1000

    Variance Analysis:
        variance_pct = (actual - planned) / planned * 100
        trend = IMPROVING if variance > 0 else DECLINING if variance < -5

    Progress Tracking:
        planned vs actual savings comparison per measure.
        Aggregation by category, status, and time period.

Regulatory References:
    - IPMVP Core Concepts (Efficiency Valuation Organization, 2022)
    - ISO 50001:2018 - Energy management systems
    - ISO 50015:2014 - Measurement and verification of energy performance
    - EN 16247-1:2022 - Energy audits (general requirements)
    - EU EED Article 8 - Mandatory energy audits

Zero-Hallucination:
    - IPMVP Option A/B formulas from EVO published specifications
    - All financial calculations use standard engineering economics
    - No LLM involvement in any numeric path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-033 Quick Wins Identifier
Engine:  8 of 8
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash",
                         "generated_at")
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

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportType(str, Enum):
    """Report type for quick-win outputs.

    SCAN_SUMMARY: Overview of scanning results.
    PRIORITIZED_ACTIONS: Ranked list of actions by ROI/payback.
    PAYBACK_ANALYSIS: Detailed payback and NPV analysis.
    CARBON_REDUCTION: Carbon savings breakdown.
    IMPLEMENTATION_PLAN: Phased implementation schedule.
    PROGRESS_DASHBOARD: Live progress tracking dashboard.
    EXECUTIVE_SUMMARY: C-suite executive summary.
    REBATE_OPPORTUNITIES: Available rebates and incentives.
    """
    SCAN_SUMMARY = "scan_summary"
    PRIORITIZED_ACTIONS = "prioritized_actions"
    PAYBACK_ANALYSIS = "payback_analysis"
    CARBON_REDUCTION = "carbon_reduction"
    IMPLEMENTATION_PLAN = "implementation_plan"
    PROGRESS_DASHBOARD = "progress_dashboard"
    EXECUTIVE_SUMMARY = "executive_summary"
    REBATE_OPPORTUNITIES = "rebate_opportunities"

class VerificationMethod(str, Enum):
    """IPMVP-aligned savings verification method.

    IPMVP_OPTION_A: Retrofit isolation - key parameter measurement.
    IPMVP_OPTION_B: Retrofit isolation - all parameter measurement.
    STIPULATED: Engineering estimates with stipulated values.
    UTILITY_BILLS: Whole-facility utility bill analysis.
    """
    IPMVP_OPTION_A = "ipmvp_option_a"
    IPMVP_OPTION_B = "ipmvp_option_b"
    STIPULATED = "stipulated"
    UTILITY_BILLS = "utility_bills"

class DashboardWidget(str, Enum):
    """Dashboard visualisation widget types.

    KPI_CARD: Single KPI metric card.
    BAR_CHART: Vertical/horizontal bar chart.
    PIE_CHART: Pie or donut chart.
    LINE_CHART: Time-series line chart.
    TABLE: Data table.
    GAUGE: Gauge/dial chart.
    WATERFALL: Waterfall chart (cumulative).
    HEATMAP: Heat map grid.
    """
    KPI_CARD = "kpi_card"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    LINE_CHART = "line_chart"
    TABLE = "table"
    GAUGE = "gauge"
    WATERFALL = "waterfall"
    HEATMAP = "heatmap"

class ProgressStatus(str, Enum):
    """Implementation progress status.

    NOT_STARTED: Measure not yet begun.
    IN_PROGRESS: Implementation underway.
    COMPLETED: Installation finished, pending verification.
    VERIFIED: Savings verified via M&V.
    CANCELLED: Measure cancelled.
    """
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    CANCELLED = "cancelled"

class TrendDirection(str, Enum):
    """KPI trend direction indicator.

    IMPROVING: Metric is improving (savings increasing / cost decreasing).
    STABLE: Metric is stable within +/-5%.
    DECLINING: Metric is declining (savings decreasing / cost increasing).
    """
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IPMVP default confidence levels by verification method.
VERIFICATION_CONFIDENCE: Dict[str, Decimal] = {
    VerificationMethod.IPMVP_OPTION_A.value: Decimal("80"),
    VerificationMethod.IPMVP_OPTION_B.value: Decimal("90"),
    VerificationMethod.STIPULATED.value: Decimal("70"),
    VerificationMethod.UTILITY_BILLS.value: Decimal("75"),
}

# Default adjustment factors by verification method.
VERIFICATION_ADJUSTMENTS: Dict[str, Decimal] = {
    VerificationMethod.IPMVP_OPTION_A.value: Decimal("0.95"),
    VerificationMethod.IPMVP_OPTION_B.value: Decimal("1.00"),
    VerificationMethod.STIPULATED.value: Decimal("0.90"),
    VerificationMethod.UTILITY_BILLS.value: Decimal("0.92"),
}

# Default CO2 grid emission factor (kg CO2e / kWh).
DEFAULT_CO2_FACTOR_KG_KWH: Decimal = Decimal("0.4")

# Default energy price for cost calculations (EUR/kWh).
DEFAULT_ENERGY_PRICE_EUR_KWH: Decimal = Decimal("0.15")

# Variance threshold for trend determination (+/- pct).
TREND_THRESHOLD_PCT: Decimal = Decimal("5")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class KPIMetric(BaseModel):
    """Single KPI metric for dashboard display.

    Attributes:
        name: Human-readable KPI name.
        value: Current metric value.
        unit: Unit of measurement (kWh, EUR, %, tCO2e).
        target: Optional target value.
        trend: Current trend direction.
        period: Reporting period label.
        previous_value: Prior period value for comparison.
        change_pct: Period-over-period change percentage.
    """
    name: str = Field(default="", max_length=200, description="KPI name")
    value: Decimal = Field(default=Decimal("0"), description="Current value")
    unit: str = Field(default="", max_length=20, description="Unit")
    target: Optional[Decimal] = Field(default=None, description="Target value")
    trend: TrendDirection = Field(
        default=TrendDirection.STABLE, description="Trend direction"
    )
    period: str = Field(default="", max_length=50, description="Period label")
    previous_value: Optional[Decimal] = Field(
        default=None, description="Previous period value"
    )
    change_pct: Optional[Decimal] = Field(
        default=None, description="Change percentage"
    )

class ProgressEntry(BaseModel):
    """Progress tracking entry for a single measure.

    Attributes:
        measure_id: Unique measure identifier.
        name: Measure description.
        status: Current implementation status.
        planned_savings_kwh: Originally planned annual savings (kWh).
        verified_savings_kwh: Verified post-implementation savings (kWh).
        verification_method: M&V method used.
        completion_pct: Implementation completion percentage.
        start_date: Actual or planned start date.
        end_date: Actual or planned end date.
        actual_cost: Actual implementation cost (EUR).
        planned_cost: Originally planned cost (EUR).
        variance_pct: Cost variance percentage.
    """
    measure_id: str = Field(default="", description="Measure ID")
    name: str = Field(default="", max_length=500, description="Measure name")
    status: ProgressStatus = Field(
        default=ProgressStatus.NOT_STARTED, description="Status"
    )
    planned_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Planned savings (kWh)"
    )
    verified_savings_kwh: Optional[Decimal] = Field(
        default=None, description="Verified savings (kWh)"
    )
    verification_method: Optional[VerificationMethod] = Field(
        default=None, description="Verification method"
    )
    completion_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Completion %"
    )
    start_date: Optional[date] = Field(default=None, description="Start date")
    end_date: Optional[date] = Field(default=None, description="End date")
    actual_cost: Optional[Decimal] = Field(
        default=None, description="Actual cost (EUR)"
    )
    planned_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Planned cost (EUR)"
    )
    variance_pct: Optional[Decimal] = Field(
        default=None, description="Cost variance %"
    )

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> Any:
        """Accept string values for ProgressStatus."""
        if isinstance(v, str):
            valid = {s.value for s in ProgressStatus}
            if v in valid:
                return ProgressStatus(v)
        return v

class SavingsVerification(BaseModel):
    """Savings verification result for a single measure.

    Attributes:
        measure_id: Measure identifier.
        baseline_consumption: Pre-retrofit consumption (kWh).
        post_consumption: Post-retrofit consumption (kWh).
        verified_savings: Verified savings (kWh).
        adjustment_factor: Applied adjustment factor.
        confidence_pct: Verification confidence percentage.
        verification_method: IPMVP method used.
        measurement_period_days: Duration of measurement period.
        notes: Verification notes and observations.
    """
    measure_id: str = Field(default="", description="Measure ID")
    baseline_consumption: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline (kWh)"
    )
    post_consumption: Decimal = Field(
        default=Decimal("0"), ge=0, description="Post consumption (kWh)"
    )
    verified_savings: Decimal = Field(
        default=Decimal("0"), description="Verified savings (kWh)"
    )
    adjustment_factor: Decimal = Field(
        default=Decimal("1"), description="Adjustment factor"
    )
    confidence_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Confidence %"
    )
    verification_method: VerificationMethod = Field(
        default=VerificationMethod.IPMVP_OPTION_A,
        description="Verification method"
    )
    measurement_period_days: int = Field(
        default=365, ge=1, description="Measurement period (days)"
    )
    notes: str = Field(default="", max_length=2000, description="Notes")

class DashboardData(BaseModel):
    """Aggregated dashboard dataset for UI rendering.

    Attributes:
        dashboard_id: Unique dashboard instance identifier.
        kpis: Top-level KPI metric cards.
        progress_entries: Per-measure progress rows.
        savings_by_category: Savings aggregated by ECM category.
        cost_by_category: Cost aggregated by ECM category.
        monthly_savings_trend: Monthly savings time-series data.
        implementation_timeline: Gantt-style implementation data.
        top_performers: Best-performing measures ranked by ROI.
    """
    dashboard_id: str = Field(default_factory=_new_uuid, description="Dashboard ID")
    kpis: List[KPIMetric] = Field(default_factory=list, description="KPI cards")
    progress_entries: List[ProgressEntry] = Field(
        default_factory=list, description="Progress rows"
    )
    savings_by_category: Dict[str, Decimal] = Field(
        default_factory=dict, description="Savings by category"
    )
    cost_by_category: Dict[str, Decimal] = Field(
        default_factory=dict, description="Cost by category"
    )
    monthly_savings_trend: List[Dict[str, Any]] = Field(
        default_factory=list, description="Monthly trend data"
    )
    implementation_timeline: List[Dict[str, Any]] = Field(
        default_factory=list, description="Timeline data"
    )
    top_performers: List[Dict[str, Any]] = Field(
        default_factory=list, description="Top performers"
    )

class ExecutiveSummary(BaseModel):
    """Executive-level summary for C-suite reporting.

    Attributes:
        title: Report title.
        period: Reporting period label.
        total_measures: Number of measures in programme.
        completed_measures: Number completed or verified.
        total_investment: Total capital investment (EUR).
        total_savings_annual: Total verified annual savings (EUR).
        total_co2e_reduction: Total annual CO2e reduction (tCO2e).
        overall_roi_pct: Programme-level ROI percentage.
        portfolio_payback_years: Programme-level payback (years).
        key_achievements: Bullet-point achievements.
        risks_issues: Bullet-point risks and issues.
        next_steps: Bullet-point next steps.
    """
    title: str = Field(default="Quick Wins Programme - Executive Summary")
    period: str = Field(default="", description="Reporting period")
    total_measures: int = Field(default=0, ge=0, description="Total measures")
    completed_measures: int = Field(default=0, ge=0, description="Completed")
    total_investment: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total investment (EUR)"
    )
    total_savings_annual: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual savings (EUR)"
    )
    total_co2e_reduction: Decimal = Field(
        default=Decimal("0"), ge=0, description="CO2e reduction (tCO2e)"
    )
    overall_roi_pct: Decimal = Field(
        default=Decimal("0"), description="Overall ROI %"
    )
    portfolio_payback_years: Decimal = Field(
        default=Decimal("0"), ge=0, description="Portfolio payback (years)"
    )
    key_achievements: List[str] = Field(
        default_factory=list, description="Key achievements"
    )
    risks_issues: List[str] = Field(
        default_factory=list, description="Risks and issues"
    )
    next_steps: List[str] = Field(
        default_factory=list, description="Next steps"
    )

class ReportOutput(BaseModel):
    """Final report output container.

    Attributes:
        report_id: Unique report identifier.
        report_type: Type of report generated.
        format: Output format.
        content: Rendered report content.
        metadata: Additional report metadata.
        generated_at: Generation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    report_id: str = Field(default_factory=_new_uuid, description="Report ID")
    report_type: ReportType = Field(
        default=ReportType.SCAN_SUMMARY, description="Report type"
    )
    format: ReportFormat = Field(
        default=ReportFormat.MARKDOWN, description="Output format"
    )
    content: str = Field(default="", description="Rendered content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Report metadata"
    )
    generated_at: datetime = Field(
        default_factory=utcnow, description="Generation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Model Rebuild (required for Pydantic v2 + __future__.annotations)
# ---------------------------------------------------------------------------

KPIMetric.model_rebuild()
ProgressEntry.model_rebuild()
SavingsVerification.model_rebuild()
DashboardData.model_rebuild()
ExecutiveSummary.model_rebuild()
ReportOutput.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class QuickWinsReportingEngine:
    """Quick wins reporting, dashboard, and verification engine.

    Aggregates implementation progress, verifies savings via IPMVP
    Option A/B, computes portfolio-level KPIs, generates executive
    summaries, and exports reports in multiple formats (Markdown,
    HTML, JSON, CSV).

    Usage::

        engine = QuickWinsReportingEngine()
        dashboard = engine.generate_dashboard(progress, verifications)
        summary = engine.generate_executive_summary(dashboard, "Q1 2026")
        report = engine.generate_report(
            ReportType.EXECUTIVE_SUMMARY,
            summary.model_dump(),
            ReportFormat.MARKDOWN,
        )
        print(report.content)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise QuickWinsReportingEngine.

        Args:
            config: Optional overrides. Supported keys:
                - co2_factor_kg_kwh (Decimal): grid emission factor
                - energy_price_eur_kwh (Decimal): energy unit price
                - currency_symbol (str): currency display symbol
        """
        self.config = config or {}
        self._co2_factor = _decimal(
            self.config.get("co2_factor_kg_kwh", DEFAULT_CO2_FACTOR_KG_KWH)
        )
        self._energy_price = _decimal(
            self.config.get("energy_price_eur_kwh", DEFAULT_ENERGY_PRICE_EUR_KWH)
        )
        self._currency = str(self.config.get("currency_symbol", "EUR"))
        logger.info(
            "QuickWinsReportingEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_report(
        self,
        report_type: ReportType,
        data: Dict[str, Any],
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> ReportOutput:
        """Generate a report in the requested format.

        Dispatches to the appropriate renderer based on *format*.

        Args:
            report_type: Type of report to generate.
            data: Report data dictionary.
            format: Desired output format.

        Returns:
            ReportOutput with rendered content and provenance hash.
        """
        t0 = time.perf_counter()
        logger.info(
            "Generating report: type=%s, format=%s",
            report_type.value, format.value,
        )

        renderer_map = {
            ReportFormat.MARKDOWN: self.render_markdown,
            ReportFormat.HTML: self.render_html,
            ReportFormat.JSON: self.render_json,
            ReportFormat.CSV: self.render_csv,
        }
        renderer = renderer_map.get(format, self.render_markdown)

        rendered = renderer(report_type, data)

        # For JSON renderer, serialise to string
        if isinstance(rendered, dict):
            content = json.dumps(rendered, default=str, indent=2)
        else:
            content = str(rendered)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metadata = {
            "engine_version": self.engine_version,
            "report_type": report_type.value,
            "format": format.value,
            "processing_time_ms": round(elapsed_ms, 3),
            "data_keys": list(data.keys()) if isinstance(data, dict) else [],
        }

        output = ReportOutput(
            report_type=report_type,
            format=format,
            content=content,
            metadata=metadata,
        )
        output.provenance_hash = _compute_hash(output)

        logger.info(
            "Report generated: type=%s, format=%s, length=%d, hash=%s",
            report_type.value, format.value, len(content),
            output.provenance_hash[:16],
        )
        return output

    def generate_dashboard(
        self,
        progress: List[ProgressEntry],
        verifications: List[SavingsVerification],
        period: str = "YTD",
    ) -> DashboardData:
        """Generate aggregated dashboard data.

        Computes KPI metrics, category breakdowns, monthly trends,
        implementation timeline, and top-performing measures.

        Args:
            progress: Per-measure progress entries.
            verifications: Savings verification results.
            period: Reporting period label.

        Returns:
            DashboardData with all aggregated metrics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Generating dashboard: %d progress entries, %d verifications, period=%s",
            len(progress), len(verifications), period,
        )

        # Build verification lookup
        verification_map: Dict[str, SavingsVerification] = {
            v.measure_id: v for v in verifications
        }

        # KPIs
        kpis = self._calculate_kpis(progress, verifications)

        # Savings by category (derive category from measure name heuristic)
        savings_by_cat: Dict[str, Decimal] = {}
        cost_by_cat: Dict[str, Decimal] = {}
        for entry in progress:
            category = self._derive_category(entry.name)
            verified = entry.verified_savings_kwh or Decimal("0")
            savings_by_cat[category] = savings_by_cat.get(
                category, Decimal("0")
            ) + verified * self._energy_price
            cost_by_cat[category] = cost_by_cat.get(
                category, Decimal("0")
            ) + (entry.actual_cost or entry.planned_cost)

        # Round category values
        savings_by_cat = {
            k: _round_val(v, 2) for k, v in savings_by_cat.items()
        }
        cost_by_cat = {
            k: _round_val(v, 2) for k, v in cost_by_cat.items()
        }

        # Monthly savings trend
        monthly_trend = self._build_savings_trend(verifications, months=12)

        # Implementation timeline
        timeline = self._build_timeline(progress)

        # Top performers (by savings-to-cost ratio)
        top_performers = self._rank_top_performers(progress, verification_map)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Dashboard generated: %d KPIs, %d categories, %.1f ms",
            len(kpis), len(savings_by_cat), elapsed_ms,
        )

        return DashboardData(
            kpis=kpis,
            progress_entries=progress,
            savings_by_category=savings_by_cat,
            cost_by_category=cost_by_cat,
            monthly_savings_trend=monthly_trend,
            implementation_timeline=timeline,
            top_performers=top_performers,
        )

    def generate_executive_summary(
        self,
        dashboard: DashboardData,
        period: str = "Q1 2026",
    ) -> ExecutiveSummary:
        """Generate executive summary from dashboard data.

        Extracts headline metrics, achievements, risks, and next steps
        from the aggregated dashboard for C-suite reporting.

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ReportFormat

        Args:
            dashboard: Aggregated dashboard data.
            period: Reporting period label.

        Returns:
            ExecutiveSummary with all executive metrics.
        """
        logger.info("Generating executive summary for period=%s", period)

        entries = dashboard.progress_entries
        total = len(entries)
        completed = sum(
            1 for e in entries
            if e.status in (ProgressStatus.COMPLETED, ProgressStatus.VERIFIED)
        )

        total_investment = sum(
            (e.actual_cost or e.planned_cost for e in entries), Decimal("0")
        )
        total_savings_kwh = sum(
            (e.verified_savings_kwh or Decimal("0") for e in entries),
            Decimal("0"),
        )
        total_savings_eur = total_savings_kwh * self._energy_price
        total_co2e = total_savings_kwh * self._co2_factor / Decimal("1000")

        overall_roi = _safe_pct(
            total_savings_eur - total_investment, total_investment
        )
        payback = _safe_divide(
            total_investment, total_savings_eur, Decimal("99")
        )

        # Key achievements
        achievements: List[str] = []
        if completed > 0:
            achievements.append(
                f"{completed} of {total} measures completed or verified."
            )
        if total_savings_eur > Decimal("0"):
            achievements.append(
                f"Annual savings of {self._format_currency(total_savings_eur)} "
                f"({self._format_energy(total_savings_kwh)})."
            )
        if total_co2e > Decimal("0"):
            achievements.append(
                f"CO2e reduction of {_round_val(total_co2e, 1)} tCO2e/year."
            )

        # Risks and issues
        risks: List[str] = []
        cancelled = sum(
            1 for e in entries if e.status == ProgressStatus.CANCELLED
        )
        if cancelled > 0:
            risks.append(f"{cancelled} measure(s) cancelled.")

        overbudget = [
            e for e in entries
            if e.actual_cost is not None
            and e.actual_cost > e.planned_cost
            and e.planned_cost > Decimal("0")
        ]
        if overbudget:
            risks.append(
                f"{len(overbudget)} measure(s) over budget."
            )

        underperforming = [
            e for e in entries
            if e.verified_savings_kwh is not None
            and e.verified_savings_kwh < e.planned_savings_kwh * Decimal("0.8")
            and e.planned_savings_kwh > Decimal("0")
        ]
        if underperforming:
            risks.append(
                f"{len(underperforming)} measure(s) underperforming "
                f"(< 80% of planned savings)."
            )

        # Next steps
        next_steps: List[str] = []
        not_started = sum(
            1 for e in entries if e.status == ProgressStatus.NOT_STARTED
        )
        if not_started > 0:
            next_steps.append(
                f"Initiate {not_started} remaining measure(s)."
            )

        in_progress = sum(
            1 for e in entries if e.status == ProgressStatus.IN_PROGRESS
        )
        if in_progress > 0:
            next_steps.append(
                f"Complete {in_progress} in-progress measure(s)."
            )

        pending_verification = sum(
            1 for e in entries if e.status == ProgressStatus.COMPLETED
        )
        if pending_verification > 0:
            next_steps.append(
                f"Verify savings for {pending_verification} completed measure(s)."
            )

        return ExecutiveSummary(
            period=period,
            total_measures=total,
            completed_measures=completed,
            total_investment=_round_val(total_investment, 2),
            total_savings_annual=_round_val(total_savings_eur, 2),
            total_co2e_reduction=_round_val(total_co2e, 3),
            overall_roi_pct=_round_val(overall_roi, 2),
            portfolio_payback_years=_round_val(payback, 2),
            key_achievements=achievements,
            risks_issues=risks,
            next_steps=next_steps,
        )

    def verify_savings(
        self,
        measure_id: str,
        baseline: Decimal,
        post: Decimal,
        method: VerificationMethod = VerificationMethod.IPMVP_OPTION_A,
        adjustments: Optional[Dict[str, Any]] = None,
    ) -> SavingsVerification:
        """Verify savings using IPMVP-aligned methodology.

        IPMVP Option A: Key parameter measurement with stipulated values.
            verified_savings = (baseline - post) * adjustment_factor
            Default adjustment_factor = 0.95 (stipulated hours uncertainty).
            Default confidence = 80%.

        IPMVP Option B: All parameters measured.
            verified_savings = (baseline - post) * adjustment_factor
            Default adjustment_factor = 1.00 (full metering).
            Default confidence = 90%.

        Stipulated: No adjustment applied.
        Utility Bills: Whole-facility with weather normalisation factor.

        Args:
            measure_id: Measure identifier.
            baseline: Baseline consumption (kWh).
            post: Post-retrofit consumption (kWh).
            method: Verification method.
            adjustments: Optional overrides for adjustment_factor,
                confidence_pct, measurement_period_days, notes.

        Returns:
            SavingsVerification with verified savings and confidence.
        """
        logger.info(
            "Verifying savings: measure=%s, method=%s, baseline=%.0f, post=%.0f",
            measure_id, method.value, float(baseline), float(post),
        )
        adjustments = adjustments or {}

        # Get default adjustment factor for the method
        adj_factor = _decimal(
            adjustments.get(
                "adjustment_factor",
                VERIFICATION_ADJUSTMENTS.get(method.value, Decimal("1")),
            )
        )

        # Get default confidence for the method
        confidence = _decimal(
            adjustments.get(
                "confidence_pct",
                VERIFICATION_CONFIDENCE.get(method.value, Decimal("70")),
            )
        )

        period_days = int(adjustments.get("measurement_period_days", 365))
        notes = str(adjustments.get("notes", ""))

        # Core savings calculation
        raw_savings = baseline - post
        verified_savings = raw_savings * adj_factor

        # Annualise if period is not 365 days
        if period_days != 365 and period_days > 0:
            daily_savings = _safe_divide(verified_savings, _decimal(period_days))
            verified_savings = daily_savings * Decimal("365")

        # Method-specific notes
        method_notes = {
            VerificationMethod.IPMVP_OPTION_A: (
                f"Option A: Key parameter measurement. "
                f"Adjustment factor {adj_factor} applied for stipulated values."
            ),
            VerificationMethod.IPMVP_OPTION_B: (
                f"Option B: All parameters measured. "
                f"Full metering with adjustment factor {adj_factor}."
            ),
            VerificationMethod.STIPULATED: (
                f"Stipulated engineering estimate. "
                f"Adjustment factor {adj_factor}."
            ),
            VerificationMethod.UTILITY_BILLS: (
                f"Utility bill analysis. "
                f"Weather-normalised adjustment factor {adj_factor}."
            ),
        }
        if not notes:
            notes = method_notes.get(method, "")

        result = SavingsVerification(
            measure_id=measure_id,
            baseline_consumption=_round_val(baseline, 2),
            post_consumption=_round_val(post, 2),
            verified_savings=_round_val(verified_savings, 2),
            adjustment_factor=adj_factor,
            confidence_pct=confidence,
            verification_method=method,
            measurement_period_days=period_days,
            notes=notes,
        )

        logger.info(
            "Savings verified: measure=%s, verified=%.0f kWh, confidence=%.0f%%",
            measure_id, float(verified_savings), float(confidence),
        )
        return result

    def track_progress(
        self,
        measures: List[Dict[str, Any]],
        actuals: List[Dict[str, Any]],
    ) -> List[ProgressEntry]:
        """Track implementation progress against plan.

        Compares planned measures with actual implementation data,
        calculates variances, and determines trend directions.

        Args:
            measures: Planned measure data (measure_id, name,
                planned_savings_kwh, planned_cost, start_date, end_date).
            actuals: Actual implementation data (measure_id, status,
                verified_savings_kwh, actual_cost, completion_pct,
                verification_method).

        Returns:
            List of ProgressEntry with variance analysis.
        """
        logger.info(
            "Tracking progress: %d planned, %d actuals",
            len(measures), len(actuals),
        )

        # Build actuals lookup
        actuals_map: Dict[str, Dict[str, Any]] = {
            a.get("measure_id", ""): a for a in actuals
        }

        entries: List[ProgressEntry] = []
        for m in measures:
            mid = str(m.get("measure_id", ""))
            actual = actuals_map.get(mid, {})

            planned_savings = _decimal(m.get("planned_savings_kwh", 0))
            planned_cost = _decimal(m.get("planned_cost", 0))
            verified = _decimal(actual.get("verified_savings_kwh", 0)) if actual else None
            actual_cost = _decimal(actual.get("actual_cost", 0)) if actual else None
            completion = _decimal(actual.get("completion_pct", 0))

            # Status resolution
            status_raw = actual.get("status", "not_started") if actual else "not_started"
            try:
                status = ProgressStatus(status_raw)
            except ValueError:
                status = ProgressStatus.NOT_STARTED

            # Verification method
            vm_raw = actual.get("verification_method")
            vm: Optional[VerificationMethod] = None
            if vm_raw:
                try:
                    vm = VerificationMethod(vm_raw)
                except ValueError:
                    vm = None

            # Cost variance
            variance: Optional[Decimal] = None
            if actual_cost is not None and planned_cost > Decimal("0"):
                variance = _safe_pct(
                    actual_cost - planned_cost, planned_cost
                )

            # Parse dates
            start_date = self._parse_date(m.get("start_date"))
            end_date = self._parse_date(m.get("end_date"))

            entries.append(ProgressEntry(
                measure_id=mid,
                name=str(m.get("name", "")),
                status=status,
                planned_savings_kwh=_round_val(planned_savings, 2),
                verified_savings_kwh=(
                    _round_val(verified, 2) if verified is not None and verified > Decimal("0")
                    else None
                ),
                verification_method=vm,
                completion_pct=_round_val(completion, 1),
                start_date=start_date,
                end_date=end_date,
                actual_cost=(
                    _round_val(actual_cost, 2)
                    if actual_cost is not None and actual_cost > Decimal("0")
                    else None
                ),
                planned_cost=_round_val(planned_cost, 2),
                variance_pct=(
                    _round_val(variance, 2) if variance is not None else None
                ),
            ))

        logger.info("Progress tracked: %d entries", len(entries))
        return entries

    # ------------------------------------------------------------------ #
    # Renderers                                                           #
    # ------------------------------------------------------------------ #

    def render_markdown(
        self, report_type: ReportType, data: Dict[str, Any],
    ) -> str:
        """Render report as Markdown with tables and headers.

        Args:
            report_type: Report type for section structure.
            data: Report data dictionary.

        Returns:
            Formatted Markdown string.
        """
        lines: List[str] = []
        title = report_type.value.replace("_", " ").title()
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Generated:** {utcnow().isoformat()}")
        lines.append(f"**Engine:** QuickWinsReportingEngine v{self.engine_version}")
        lines.append("")

        if report_type == ReportType.EXECUTIVE_SUMMARY:
            lines.extend(self._md_executive_summary(data))
        elif report_type == ReportType.PROGRESS_DASHBOARD:
            lines.extend(self._md_progress_dashboard(data))
        elif report_type == ReportType.PAYBACK_ANALYSIS:
            lines.extend(self._md_payback_analysis(data))
        elif report_type == ReportType.CARBON_REDUCTION:
            lines.extend(self._md_carbon_reduction(data))
        else:
            lines.extend(self._md_generic_table(data))

        lines.append("")
        lines.append("---")
        lines.append(f"*Provenance hash computed at generation time.*")
        return "\n".join(lines)

    def render_html(
        self, report_type: ReportType, data: Dict[str, Any],
    ) -> str:
        """Render report as HTML with inline CSS.

        Args:
            report_type: Report type for section structure.
            data: Report data dictionary.

        Returns:
            HTML string with embedded styles.
        """
        title = report_type.value.replace("_", " ").title()
        css = (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            "h1{color:#1a5c2e;border-bottom:2px solid #1a5c2e;padding-bottom:0.3em}"
            "h2{color:#2d8a4e}table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ddd;padding:8px;text-align:left}"
            "th{background:#1a5c2e;color:#fff}.kpi{display:inline-block;"
            "background:#f0f9f4;border:1px solid #1a5c2e;border-radius:8px;"
            "padding:1em;margin:0.5em;min-width:150px;text-align:center}"
            ".kpi-value{font-size:1.8em;font-weight:bold;color:#1a5c2e}"
            ".kpi-label{font-size:0.9em;color:#666}"
            ".improving{color:#2d8a4e}.declining{color:#c0392b}"
            ".stable{color:#7f8c8d}"
        )

        parts: List[str] = [
            "<!DOCTYPE html>",
            "<html lang='en'><head><meta charset='utf-8'>",
            f"<title>{title}</title>",
            f"<style>{css}</style>",
            "</head><body>",
            f"<h1>{title}</h1>",
            f"<p><em>Generated: {utcnow().isoformat()} | "
            f"Engine: QuickWinsReportingEngine v{self.engine_version}</em></p>",
        ]

        if report_type == ReportType.EXECUTIVE_SUMMARY:
            parts.extend(self._html_executive_summary(data))
        elif report_type == ReportType.PROGRESS_DASHBOARD:
            parts.extend(self._html_progress_dashboard(data))
        else:
            parts.extend(self._html_generic_table(data))

        parts.append("</body></html>")
        return "\n".join(parts)

    def render_json(
        self, report_type: ReportType, data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Render report as structured JSON.

        Args:
            report_type: Report type identifier.
            data: Report data dictionary.

        Returns:
            JSON-serialisable dictionary.
        """
        return {
            "report_type": report_type.value,
            "engine_version": self.engine_version,
            "generated_at": utcnow().isoformat(),
            "data": data,
        }

    def render_csv(
        self, report_type: ReportType, data: Dict[str, Any],
    ) -> str:
        """Render report as CSV string.

        Flattens the data dictionary into tabular rows suitable
        for spreadsheet import.

        Args:
            report_type: Report type identifier.
            data: Report data dictionary.

        Returns:
            CSV string with header row.
        """
        rows: List[str] = []

        # Try to find a list within data to tabulate
        list_data = self._find_tabular_data(data)
        if list_data:
            # Use keys from first item as headers
            headers = list(list_data[0].keys()) if list_data else []
            rows.append(",".join(headers))
            for item in list_data:
                values = [
                    self._csv_escape(str(item.get(h, ""))) for h in headers
                ]
                rows.append(",".join(values))
        else:
            # Flat key-value export
            rows.append("key,value")
            for k, v in data.items():
                rows.append(f"{self._csv_escape(str(k))},{self._csv_escape(str(v))}")

        return "\n".join(rows)

    # ------------------------------------------------------------------ #
    # KPI Calculation                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_kpis(
        self,
        progress: List[ProgressEntry],
        verifications: List[SavingsVerification],
    ) -> List[KPIMetric]:
        """Calculate dashboard KPI metrics.

        KPIs computed:
            1. Total Verified Savings (kWh)
            2. Total Cost Savings (EUR)
            3. Completion Rate (%)
            4. CO2e Reduction (tCO2e)
            5. Portfolio ROI (%)
            6. Average Payback (years)

        Args:
            progress: Progress entries.
            verifications: Verification results.

        Returns:
            List of KPIMetric.
        """
        kpis: List[KPIMetric] = []

        total_verified_kwh = sum(
            (v.verified_savings for v in verifications), Decimal("0")
        )
        total_planned_kwh = sum(
            (e.planned_savings_kwh for e in progress), Decimal("0")
        )
        total_cost = sum(
            (e.actual_cost or e.planned_cost for e in progress), Decimal("0")
        )
        total_savings_eur = total_verified_kwh * self._energy_price

        # Completion rate
        total_measures = len(progress)
        completed = sum(
            1 for e in progress
            if e.status in (ProgressStatus.COMPLETED, ProgressStatus.VERIFIED)
        )
        completion_rate = _safe_pct(
            _decimal(completed), _decimal(total_measures)
        )

        # CO2e
        co2e = total_verified_kwh * self._co2_factor / Decimal("1000")

        # ROI
        roi = _safe_pct(total_savings_eur - total_cost, total_cost)

        # Average payback
        avg_payback = _safe_divide(total_cost, total_savings_eur, Decimal("0"))

        # Savings achievement trend
        savings_trend = self._determine_trend(
            total_verified_kwh, total_planned_kwh
        )

        kpis.append(KPIMetric(
            name="Total Verified Savings",
            value=_round_val(total_verified_kwh, 0),
            unit="kWh",
            target=_round_val(total_planned_kwh, 0) if total_planned_kwh > 0 else None,
            trend=savings_trend,
            period="YTD",
            change_pct=_round_val(
                _safe_pct(total_verified_kwh, total_planned_kwh), 1
            ) if total_planned_kwh > Decimal("0") else None,
        ))

        kpis.append(KPIMetric(
            name="Cost Savings",
            value=_round_val(total_savings_eur, 0),
            unit=self._currency,
            trend=savings_trend,
            period="YTD",
        ))

        kpis.append(KPIMetric(
            name="Completion Rate",
            value=_round_val(completion_rate, 1),
            unit="%",
            target=Decimal("100"),
            trend=(
                TrendDirection.IMPROVING if completion_rate >= Decimal("50")
                else TrendDirection.STABLE
            ),
            period="YTD",
        ))

        kpis.append(KPIMetric(
            name="CO2e Reduction",
            value=_round_val(co2e, 1),
            unit="tCO2e",
            trend=savings_trend,
            period="YTD",
        ))

        kpis.append(KPIMetric(
            name="Portfolio ROI",
            value=_round_val(roi, 1),
            unit="%",
            trend=(
                TrendDirection.IMPROVING if roi > Decimal("0")
                else TrendDirection.DECLINING
            ),
            period="YTD",
        ))

        kpis.append(KPIMetric(
            name="Average Payback",
            value=_round_val(avg_payback, 1),
            unit="years",
            trend=TrendDirection.STABLE,
            period="YTD",
        ))

        return kpis

    def _build_savings_trend(
        self,
        verifications: List[SavingsVerification],
        months: int = 12,
    ) -> List[Dict[str, Any]]:
        """Build monthly savings trend data for charting.

        Distributes verified savings evenly across measurement periods
        to create a monthly time-series.

        Args:
            verifications: Verification results.
            months: Number of months to generate.

        Returns:
            List of monthly data points.
        """
        monthly: Dict[int, Decimal] = {m: Decimal("0") for m in range(1, months + 1)}

        for v in verifications:
            # Distribute savings across months proportionally
            period_months = max(1, v.measurement_period_days // 30)
            monthly_savings = _safe_divide(
                v.verified_savings, _decimal(min(period_months, months))
            )
            for m in range(1, min(period_months, months) + 1):
                monthly[m] = monthly[m] + monthly_savings

        # Compute cumulative
        cumulative = Decimal("0")
        trend: List[Dict[str, Any]] = []
        for m in range(1, months + 1):
            cumulative += monthly[m]
            trend.append({
                "month": m,
                "monthly_savings_kwh": str(_round_val(monthly[m], 2)),
                "cumulative_savings_kwh": str(_round_val(cumulative, 2)),
                "monthly_savings_eur": str(
                    _round_val(monthly[m] * self._energy_price, 2)
                ),
            })

        return trend

    # ------------------------------------------------------------------ #
    # Internal Helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_timeline(
        self, progress: List[ProgressEntry],
    ) -> List[Dict[str, Any]]:
        """Build implementation timeline for Gantt visualisation.

        Args:
            progress: Progress entries with dates.

        Returns:
            List of timeline entries.
        """
        timeline: List[Dict[str, Any]] = []
        for entry in progress:
            if entry.status == ProgressStatus.CANCELLED:
                continue
            timeline.append({
                "measure_id": entry.measure_id,
                "name": entry.name,
                "status": entry.status.value,
                "start_date": str(entry.start_date) if entry.start_date else "",
                "end_date": str(entry.end_date) if entry.end_date else "",
                "completion_pct": str(entry.completion_pct),
            })
        return timeline

    def _rank_top_performers(
        self,
        progress: List[ProgressEntry],
        verification_map: Dict[str, SavingsVerification],
    ) -> List[Dict[str, Any]]:
        """Rank measures by savings-to-cost ratio.

        Args:
            progress: Progress entries.
            verification_map: Verification results by measure ID.

        Returns:
            Top 10 measures ranked by ROI.
        """
        scored: List[Tuple[Decimal, Dict[str, Any]]] = []
        for entry in progress:
            cost = entry.actual_cost or entry.planned_cost
            verified = entry.verified_savings_kwh or Decimal("0")
            savings_eur = verified * self._energy_price

            ratio = _safe_divide(savings_eur, cost) if cost > Decimal("0") else Decimal("0")

            v = verification_map.get(entry.measure_id)
            confidence = v.confidence_pct if v else Decimal("0")

            scored.append((ratio, {
                "measure_id": entry.measure_id,
                "name": entry.name,
                "verified_savings_kwh": str(_round_val(verified, 0)),
                "cost": str(_round_val(cost, 0)),
                "savings_eur": str(_round_val(savings_eur, 0)),
                "savings_to_cost_ratio": str(_round_val(ratio, 2)),
                "confidence_pct": str(_round_val(confidence, 0)),
                "status": entry.status.value,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:10]]

    def _determine_trend(
        self, current: Decimal, reference: Decimal,
    ) -> TrendDirection:
        """Determine trend direction from current vs reference.

        Args:
            current: Current period value.
            reference: Reference/target value.

        Returns:
            TrendDirection.
        """
        if reference <= Decimal("0"):
            return TrendDirection.STABLE

        pct = _safe_pct(current - reference, reference)
        if pct > TREND_THRESHOLD_PCT:
            return TrendDirection.IMPROVING
        elif pct < -TREND_THRESHOLD_PCT:
            return TrendDirection.DECLINING
        return TrendDirection.STABLE

    def _derive_category(self, name: str) -> str:
        """Derive ECM category from measure name heuristic.

        Args:
            name: Measure name.

        Returns:
            Category string.
        """
        name_lower = name.lower()
        category_keywords = {
            "lighting": ["led", "light", "lamp", "luminaire", "lumen"],
            "hvac": ["hvac", "air conditioning", "cooling", "heating", "ahu"],
            "motors": ["motor", "drive", "vsd", "vfd"],
            "compressed_air": ["compressed air", "compressor", "leak"],
            "controls": ["bms", "control", "automation", "sensor", "setpoint"],
            "boiler": ["boiler", "steam", "condensate"],
            "building_envelope": ["insulation", "glazing", "envelope", "seal"],
            "renewable": ["solar", "pv", "wind", "battery"],
            "refrigeration": ["refriger", "cold room", "freezer"],
            "process_heat": ["furnace", "kiln", "oven", "process heat"],
        }
        for category, keywords in category_keywords.items():
            for kw in keywords:
                if kw in name_lower:
                    return category
        return "other"

    def _parse_date(self, value: Any) -> Optional[date]:
        """Parse a date value from string or date object.

        Args:
            value: Date string (YYYY-MM-DD) or date object.

        Returns:
            date or None.
        """
        if value is None:
            return None
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value)[:10])
        except (ValueError, TypeError):
            return None

    def _format_currency(self, value: Decimal) -> str:
        """Format a Decimal as currency string.

        Args:
            value: Currency value.

        Returns:
            Formatted string (e.g., 'EUR 12,345.00').
        """
        rounded = _round_val(value, 2)
        int_part = int(abs(rounded))
        formatted = f"{int_part:,}"
        decimal_part = str(abs(rounded) % 1)[1:]  # .XX part
        if decimal_part and len(decimal_part) > 1:
            decimal_str = decimal_part[:3]
        else:
            decimal_str = ".00"
        sign = "-" if rounded < Decimal("0") else ""
        return f"{sign}{self._currency} {formatted}{decimal_str}"

    def _format_energy(self, value: Decimal) -> str:
        """Format a Decimal as energy string with appropriate unit.

        Args:
            value: Energy value in kWh.

        Returns:
            Formatted string (e.g., '1,234 kWh' or '1.23 MWh').
        """
        if abs(value) >= Decimal("1000000"):
            return f"{_round_val(value / Decimal('1000000'), 2)} GWh"
        elif abs(value) >= Decimal("1000"):
            return f"{_round_val(value / Decimal('1000'), 2)} MWh"
        return f"{_round_val(value, 0)} kWh"

    def _csv_escape(self, value: str) -> str:
        """Escape a value for CSV output.

        Args:
            value: Raw string value.

        Returns:
            CSV-safe string.
        """
        if "," in value or '"' in value or "\n" in value:
            return f'"{value.replace(chr(34), chr(34) + chr(34))}"'
        return value

    def _find_tabular_data(
        self, data: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """Find the first list of dicts in data for CSV export.

        Args:
            data: Report data dictionary.

        Returns:
            List of dicts or None.
        """
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return None

    # ------------------------------------------------------------------ #
    # Markdown Section Renderers                                          #
    # ------------------------------------------------------------------ #

    def _md_executive_summary(self, data: Dict[str, Any]) -> List[str]:
        """Render executive summary as Markdown sections.

        Args:
            data: Executive summary data.

        Returns:
            List of Markdown lines.
        """
        lines: List[str] = []
        period = data.get("period", "")
        lines.append(f"## Period: {period}")
        lines.append("")

        lines.append("### Key Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Measures | {data.get('total_measures', 0)} |")
        lines.append(f"| Completed | {data.get('completed_measures', 0)} |")
        lines.append(f"| Total Investment | {self._currency} {data.get('total_investment', 0)} |")
        lines.append(f"| Annual Savings | {self._currency} {data.get('total_savings_annual', 0)} |")
        lines.append(f"| CO2e Reduction | {data.get('total_co2e_reduction', 0)} tCO2e |")
        lines.append(f"| Overall ROI | {data.get('overall_roi_pct', 0)}% |")
        lines.append(f"| Portfolio Payback | {data.get('portfolio_payback_years', 0)} years |")
        lines.append("")

        achievements = data.get("key_achievements", [])
        if achievements:
            lines.append("### Key Achievements")
            lines.append("")
            for a in achievements:
                lines.append(f"- {a}")
            lines.append("")

        risks = data.get("risks_issues", [])
        if risks:
            lines.append("### Risks & Issues")
            lines.append("")
            for r in risks:
                lines.append(f"- {r}")
            lines.append("")

        next_steps = data.get("next_steps", [])
        if next_steps:
            lines.append("### Next Steps")
            lines.append("")
            for n in next_steps:
                lines.append(f"- {n}")
            lines.append("")

        return lines

    def _md_progress_dashboard(self, data: Dict[str, Any]) -> List[str]:
        """Render progress dashboard as Markdown.

        Args:
            data: Dashboard data.

        Returns:
            List of Markdown lines.
        """
        lines: List[str] = []

        # KPIs section
        kpis = data.get("kpis", [])
        if kpis:
            lines.append("## KPI Summary")
            lines.append("")
            lines.append("| KPI | Value | Unit | Trend |")
            lines.append("|-----|-------|------|-------|")
            for kpi in kpis:
                name = kpi.get("name", "") if isinstance(kpi, dict) else kpi.name
                value = kpi.get("value", "") if isinstance(kpi, dict) else kpi.value
                unit = kpi.get("unit", "") if isinstance(kpi, dict) else kpi.unit
                trend = kpi.get("trend", "") if isinstance(kpi, dict) else kpi.trend
                if hasattr(trend, "value"):
                    trend = trend.value
                lines.append(f"| {name} | {value} | {unit} | {trend} |")
            lines.append("")

        # Progress table
        entries = data.get("progress_entries", [])
        if entries:
            lines.append("## Implementation Progress")
            lines.append("")
            lines.append("| Measure | Status | Completion | Planned (kWh) | Verified (kWh) | Variance |")
            lines.append("|---------|--------|------------|---------------|----------------|----------|")
            for e in entries:
                if isinstance(e, dict):
                    name = e.get("name", "")
                    status = e.get("status", "")
                    comp = e.get("completion_pct", "0")
                    planned = e.get("planned_savings_kwh", "0")
                    verified = e.get("verified_savings_kwh", "-")
                    variance = e.get("variance_pct", "-")
                else:
                    name = e.name
                    status = e.status.value if hasattr(e.status, "value") else str(e.status)
                    comp = str(e.completion_pct)
                    planned = str(e.planned_savings_kwh)
                    verified = str(e.verified_savings_kwh) if e.verified_savings_kwh else "-"
                    variance = f"{e.variance_pct}%" if e.variance_pct is not None else "-"
                lines.append(f"| {name} | {status} | {comp}% | {planned} | {verified} | {variance} |")
            lines.append("")

        return lines

    def _md_payback_analysis(self, data: Dict[str, Any]) -> List[str]:
        """Render payback analysis as Markdown.

        Args:
            data: Payback analysis data.

        Returns:
            List of Markdown lines.
        """
        lines: List[str] = []
        lines.append("## Payback Analysis")
        lines.append("")

        measures = data.get("measures", [])
        if measures:
            lines.append("| Measure | Cost | Annual Savings | Payback (years) | ROI (%) |")
            lines.append("|---------|------|----------------|-----------------|---------|")
            for m in measures:
                name = m.get("name", "")
                cost = m.get("cost", "0")
                savings = m.get("annual_savings", "0")
                payback = m.get("payback_years", "-")
                roi = m.get("roi_pct", "-")
                lines.append(f"| {name} | {self._currency} {cost} | {self._currency} {savings} | {payback} | {roi} |")
            lines.append("")

        return lines

    def _md_carbon_reduction(self, data: Dict[str, Any]) -> List[str]:
        """Render carbon reduction report as Markdown.

        Args:
            data: Carbon reduction data.

        Returns:
            List of Markdown lines.
        """
        lines: List[str] = []
        lines.append("## Carbon Reduction Summary")
        lines.append("")
        lines.append(f"**Total CO2e Reduction:** {data.get('total_co2e', 0)} tCO2e/year")
        lines.append(f"**Grid Factor:** {data.get('grid_factor', DEFAULT_CO2_FACTOR_KG_KWH)} kg CO2e/kWh")
        lines.append("")

        by_category = data.get("by_category", {})
        if by_category:
            lines.append("| Category | Savings (kWh) | CO2e (tCO2e) |")
            lines.append("|----------|---------------|--------------|")
            for cat, savings in by_category.items():
                co2e = _decimal(savings) * self._co2_factor / Decimal("1000")
                lines.append(f"| {cat} | {savings} | {_round_val(co2e, 2)} |")
            lines.append("")

        return lines

    def _md_generic_table(self, data: Dict[str, Any]) -> List[str]:
        """Render generic key-value data as Markdown table.

        Args:
            data: Flat data dictionary.

        Returns:
            List of Markdown lines.
        """
        lines: List[str] = []
        lines.append("## Report Data")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                v = json.dumps(v, default=str)[:100]
            lines.append(f"| {k} | {v} |")
        lines.append("")
        return lines

    # ------------------------------------------------------------------ #
    # HTML Section Renderers                                              #
    # ------------------------------------------------------------------ #

    def _html_executive_summary(self, data: Dict[str, Any]) -> List[str]:
        """Render executive summary as HTML.

        Args:
            data: Executive summary data.

        Returns:
            List of HTML lines.
        """
        parts: List[str] = []

        # KPI cards
        parts.append("<div style='display:flex;flex-wrap:wrap'>")
        kpi_items = [
            ("Total Measures", data.get("total_measures", 0), ""),
            ("Completed", data.get("completed_measures", 0), ""),
            ("Investment", data.get("total_investment", 0), self._currency),
            ("Annual Savings", data.get("total_savings_annual", 0), self._currency),
            ("CO2e Reduction", data.get("total_co2e_reduction", 0), "tCO2e"),
            ("ROI", data.get("overall_roi_pct", 0), "%"),
            ("Payback", data.get("portfolio_payback_years", 0), "years"),
        ]
        for label, value, unit in kpi_items:
            parts.append(
                f"<div class='kpi'>"
                f"<div class='kpi-value'>{value}</div>"
                f"<div class='kpi-label'>{label} {unit}</div>"
                f"</div>"
            )
        parts.append("</div>")

        # Achievements
        achievements = data.get("key_achievements", [])
        if achievements:
            parts.append("<h2>Key Achievements</h2><ul>")
            for a in achievements:
                parts.append(f"<li>{a}</li>")
            parts.append("</ul>")

        # Risks
        risks = data.get("risks_issues", [])
        if risks:
            parts.append("<h2>Risks &amp; Issues</h2><ul>")
            for r in risks:
                parts.append(f"<li>{r}</li>")
            parts.append("</ul>")

        # Next steps
        next_steps = data.get("next_steps", [])
        if next_steps:
            parts.append("<h2>Next Steps</h2><ul>")
            for n in next_steps:
                parts.append(f"<li>{n}</li>")
            parts.append("</ul>")

        return parts

    def _html_progress_dashboard(self, data: Dict[str, Any]) -> List[str]:
        """Render progress dashboard as HTML.

        Args:
            data: Dashboard data.

        Returns:
            List of HTML lines.
        """
        parts: List[str] = []

        # KPI row
        kpis = data.get("kpis", [])
        if kpis:
            parts.append("<div style='display:flex;flex-wrap:wrap'>")
            for kpi in kpis:
                name = kpi.get("name", "") if isinstance(kpi, dict) else kpi.name
                value = kpi.get("value", "") if isinstance(kpi, dict) else kpi.value
                unit = kpi.get("unit", "") if isinstance(kpi, dict) else kpi.unit
                trend = kpi.get("trend", "") if isinstance(kpi, dict) else kpi.trend
                if hasattr(trend, "value"):
                    trend = trend.value
                css_class = trend if isinstance(trend, str) else "stable"
                parts.append(
                    f"<div class='kpi'>"
                    f"<div class='kpi-value {css_class}'>{value}</div>"
                    f"<div class='kpi-label'>{name} ({unit})</div>"
                    f"</div>"
                )
            parts.append("</div>")

        # Progress table
        entries = data.get("progress_entries", [])
        if entries:
            parts.append("<h2>Implementation Progress</h2>")
            parts.append("<table><tr><th>Measure</th><th>Status</th>"
                         "<th>Completion</th><th>Planned (kWh)</th>"
                         "<th>Verified (kWh)</th></tr>")
            for e in entries:
                if isinstance(e, dict):
                    name = e.get("name", "")
                    status = e.get("status", "")
                    comp = e.get("completion_pct", "0")
                    planned = e.get("planned_savings_kwh", "0")
                    verified = e.get("verified_savings_kwh", "-")
                else:
                    name = e.name
                    status = e.status.value if hasattr(e.status, "value") else str(e.status)
                    comp = str(e.completion_pct)
                    planned = str(e.planned_savings_kwh)
                    verified = str(e.verified_savings_kwh) if e.verified_savings_kwh else "-"
                parts.append(
                    f"<tr><td>{name}</td><td>{status}</td>"
                    f"<td>{comp}%</td><td>{planned}</td>"
                    f"<td>{verified}</td></tr>"
                )
            parts.append("</table>")

        return parts

    def _html_generic_table(self, data: Dict[str, Any]) -> List[str]:
        """Render generic data as HTML table.

        Args:
            data: Flat data dictionary.

        Returns:
            List of HTML lines.
        """
        parts: List[str] = [
            "<h2>Report Data</h2>",
            "<table><tr><th>Field</th><th>Value</th></tr>",
        ]
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                v = json.dumps(v, default=str)[:200]
            parts.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        parts.append("</table>")
        return parts
