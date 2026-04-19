# -*- coding: utf-8 -*-
"""
PeakReportingEngine - PACK-038 Peak Shaving Engine 10
=======================================================

Dashboard and report generation engine for peak shaving analysis.
Generates KPI dashboards, comprehensive reports, executive summaries,
verification reports, and multi-format exports covering load profiles,
peak events, demand charges, BESS dispatch, load shifting, CP status,
power factor, and financial performance.

Calculation Methodology:
    KPI Computations:
        peak_reduction_pct = (baseline_peak - managed_peak) / baseline_peak * 100
        demand_charge_savings = baseline_charge - actual_charge
        bess_utilisation_pct = actual_discharge_kwh / usable_capacity_kwh * 100
        roi_pct = net_annual_savings / investment_cost * 100
        load_factor = avg_demand / peak_demand * 100

    Trend Analysis:
        mom_change = (current_month - prior_month) / prior_month * 100
        yoy_change = (current_year - prior_year) / prior_year * 100
        trend_direction: IMPROVING | STABLE | DECLINING | VOLATILE

    Report Generation:
        Markdown template rendering with inline data tables
        HTML conversion with inline CSS for email-safe formatting
        CSV extraction for data tables
        JSON structured export for API consumers

    Verification Report:
        M&V per IPMVP Option A/B/C/D methodology references
        baseline_model_fit (R2, CVRMSE)
        adjusted_baseline vs actual comparison
        savings_uncertainty at 90% confidence

Regulatory References:
    - IPMVP (International Performance Measurement and Verification Protocol)
    - ASHRAE Guideline 14-2014 - Measurement of Energy Savings
    - ISO 50015:2014 - Measurement and Verification of Energy Performance
    - FERC Order 745 - Demand Response Compensation
    - NAESB WEQ Business Practice Standards

Zero-Hallucination:
    - KPI calculations use deterministic arithmetic on measured data
    - Trend analysis uses simple period-over-period comparison
    - Report templates are static Markdown with data interpolation
    - No LLM involvement in any calculation or formatting path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-038 Peak Shaving
Engine:  10 of 10
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
from greenlang.schemas.enums import ReportFormat

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

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DashboardPanel(str, Enum):
    """Dashboard panel type.

    LOAD_PROFILE:   Load profile and demand curve panel.
    PEAK_EVENTS:    Peak event detection and history panel.
    DEMAND_CHARGES: Demand charge analysis panel.
    BESS_DISPATCH:  BESS dispatch and SOC tracking panel.
    LOAD_SHIFTING:  Load shifting schedule and results panel.
    CP_STATUS:      Coincident peak status and prediction panel.
    POWER_FACTOR:   Power factor and reactive power panel.
    FINANCIAL:      Financial performance and ROI panel.
    """
    LOAD_PROFILE = "load_profile"
    PEAK_EVENTS = "peak_events"
    DEMAND_CHARGES = "demand_charges"
    BESS_DISPATCH = "bess_dispatch"
    LOAD_SHIFTING = "load_shifting"
    CP_STATUS = "cp_status"
    POWER_FACTOR = "power_factor"
    FINANCIAL = "financial"

class ReportType(str, Enum):
    """Report type.

    LOAD_ANALYSIS:   Comprehensive load profile analysis report.
    PEAK_ASSESSMENT: Peak demand assessment and reduction report.
    BESS_SIZING:     BESS sizing and dispatch optimisation report.
    LOAD_SHIFTING:   Load shifting analysis and recommendations.
    CP_MANAGEMENT:   Coincident peak management report.
    FINANCIAL:       Financial analysis and investment case report.
    VERIFICATION:    M&V savings verification report.
    """
    LOAD_ANALYSIS = "load_analysis"
    PEAK_ASSESSMENT = "peak_assessment"
    BESS_SIZING = "bess_sizing"
    LOAD_SHIFTING = "load_shifting"
    CP_MANAGEMENT = "cp_management"
    FINANCIAL = "financial"
    VERIFICATION = "verification"

class WidgetType(str, Enum):
    """Dashboard widget type.

    KPI:      Key performance indicator (single value + trend).
    CHART:    Chart placeholder (line, bar, area).
    TABLE:    Data table.
    TIMELINE: Event timeline.
    HEATMAP:  Demand heatmap (day-of-week x hour).
    """
    KPI = "kpi"
    CHART = "chart"
    TABLE = "table"
    TIMELINE = "timeline"
    HEATMAP = "heatmap"

class TrendDirection(str, Enum):
    """Metric trend direction.

    IMPROVING: Metric is improving (desirable direction).
    STABLE:    Metric is stable (within +/-2% band).
    DECLINING: Metric is declining (undesirable direction).
    VOLATILE:  Metric shows high variability.
    """
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# KPI threshold bands for RAG (Red-Amber-Green) classification.
KPI_THRESHOLDS: Dict[str, Dict[str, Decimal]] = {
    "peak_reduction_pct": {
        "green": Decimal("20"),
        "amber": Decimal("10"),
        "red": Decimal("0"),
    },
    "demand_charge_savings_pct": {
        "green": Decimal("15"),
        "amber": Decimal("5"),
        "red": Decimal("0"),
    },
    "bess_utilisation_pct": {
        "green": Decimal("70"),
        "amber": Decimal("40"),
        "red": Decimal("0"),
    },
    "load_factor_pct": {
        "green": Decimal("60"),
        "amber": Decimal("40"),
        "red": Decimal("0"),
    },
    "roi_pct": {
        "green": Decimal("15"),
        "amber": Decimal("5"),
        "red": Decimal("0"),
    },
}

# Trend stability band (%).
TREND_STABILITY_BAND: Decimal = Decimal("2.0")

# Trend volatility threshold (coefficient of variation %).
TREND_VOLATILITY_THRESHOLD: Decimal = Decimal("15.0")

# IPMVP acceptable CVRMSE thresholds.
IPMVP_CVRMSE_MONTHLY: Decimal = Decimal("15.0")
IPMVP_CVRMSE_HOURLY: Decimal = Decimal("25.0")

# Maximum panels per dashboard.
MAX_DASHBOARD_PANELS: int = 20

# ---------------------------------------------------------------------------
# Pydantic Models -- Input / Core
# ---------------------------------------------------------------------------

class KPIMetric(BaseModel):
    """Key Performance Indicator metric.

    Attributes:
        kpi_id: KPI identifier.
        name: KPI display name.
        value: Current KPI value.
        unit: Unit of measurement.
        target: Target value.
        prior_value: Prior period value.
        change_pct: Period-over-period change (%).
        trend: Trend direction.
        rag_status: RAG classification (green/amber/red).
        description: KPI description.
    """
    kpi_id: str = Field(
        default_factory=_new_uuid, description="KPI ID"
    )
    name: str = Field(
        default="", max_length=200, description="KPI name"
    )
    value: Decimal = Field(
        default=Decimal("0"), description="Current value"
    )
    unit: str = Field(
        default="", max_length=50, description="Unit"
    )
    target: Decimal = Field(
        default=Decimal("0"), description="Target value"
    )
    prior_value: Decimal = Field(
        default=Decimal("0"), description="Prior period value"
    )
    change_pct: Decimal = Field(
        default=Decimal("0"), description="Change (%)"
    )
    trend: TrendDirection = Field(
        default=TrendDirection.STABLE, description="Trend"
    )
    rag_status: str = Field(
        default="green", description="RAG status"
    )
    description: str = Field(
        default="", max_length=500, description="Description"
    )

class DashboardWidget(BaseModel):
    """Dashboard widget configuration and data.

    Attributes:
        widget_id: Widget identifier.
        widget_type: Widget type.
        panel: Dashboard panel this widget belongs to.
        title: Widget title.
        data: Widget data payload.
        position: Grid position (row, col).
        size: Widget size (rows, cols).
    """
    widget_id: str = Field(
        default_factory=_new_uuid, description="Widget ID"
    )
    widget_type: WidgetType = Field(
        default=WidgetType.KPI, description="Widget type"
    )
    panel: DashboardPanel = Field(
        default=DashboardPanel.LOAD_PROFILE, description="Panel"
    )
    title: str = Field(
        default="", max_length=200, description="Title"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Widget data"
    )
    position: Dict[str, int] = Field(
        default_factory=lambda: {"row": 0, "col": 0},
        description="Grid position"
    )
    size: Dict[str, int] = Field(
        default_factory=lambda: {"rows": 1, "cols": 1},
        description="Widget size"
    )

class ReportSection(BaseModel):
    """Report section with content.

    Attributes:
        section_id: Section identifier.
        title: Section title.
        content_markdown: Markdown content.
        data_tables: Embedded data tables.
        kpis: KPI metrics for this section.
        order: Display order.
    """
    section_id: str = Field(
        default_factory=_new_uuid, description="Section ID"
    )
    title: str = Field(
        default="", max_length=300, description="Section title"
    )
    content_markdown: str = Field(
        default="", description="Markdown content"
    )
    data_tables: List[Dict[str, Any]] = Field(
        default_factory=list, description="Data tables"
    )
    kpis: List[KPIMetric] = Field(
        default_factory=list, description="KPIs"
    )
    order: int = Field(
        default=0, ge=0, description="Display order"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class DashboardData(BaseModel):
    """Dashboard data payload.

    Attributes:
        dashboard_id: Dashboard identifier.
        title: Dashboard title.
        panels: Active panels.
        widgets: Dashboard widgets.
        kpis: Top-level KPIs.
        generated_at: Generation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    dashboard_id: str = Field(
        default_factory=_new_uuid, description="Dashboard ID"
    )
    title: str = Field(
        default="Peak Shaving Dashboard", max_length=300,
        description="Dashboard title"
    )
    panels: List[DashboardPanel] = Field(
        default_factory=list, description="Active panels"
    )
    widgets: List[DashboardWidget] = Field(
        default_factory=list, description="Widgets"
    )
    kpis: List[KPIMetric] = Field(
        default_factory=list, description="Top-level KPIs"
    )
    generated_at: datetime = Field(
        default_factory=utcnow, description="Generation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class ReportOutput(BaseModel):
    """Generated report output.

    Attributes:
        report_id: Report identifier.
        report_type: Report type.
        title: Report title.
        sections: Report sections.
        executive_summary: Executive summary text.
        export_format: Export format.
        content: Rendered content (markdown/html/json/csv).
        metadata: Report metadata.
        generated_at: Generation timestamp.
        processing_time_ms: Processing time (ms).
        provenance_hash: SHA-256 audit hash.
    """
    report_id: str = Field(
        default_factory=_new_uuid, description="Report ID"
    )
    report_type: ReportType = Field(
        default=ReportType.PEAK_ASSESSMENT, description="Report type"
    )
    title: str = Field(
        default="", max_length=500, description="Report title"
    )
    sections: List[ReportSection] = Field(
        default_factory=list, description="Report sections"
    )
    executive_summary: str = Field(
        default="", description="Executive summary"
    )
    export_format: ReportFormat = Field(
        default=ReportFormat.MARKDOWN, description="Export format"
    )
    content: str = Field(
        default="", description="Rendered content"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata"
    )
    generated_at: datetime = Field(
        default_factory=utcnow, description="Generation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PeakReportingEngine:
    """Dashboard and report generation engine for peak shaving analysis.

    Generates KPI dashboards, comprehensive reports, executive
    summaries, verification reports, and multi-format exports.

    Usage::

        engine = PeakReportingEngine()
        dashboard = engine.generate_dashboard(kpis, panels)
        report = engine.generate_report(report_type, data)
        summary = engine.generate_executive_summary(data)
        verification = engine.generate_verification_report(mv_data)
        exported = engine.export_report(report, format)

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise PeakReportingEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - facility_name (str): facility name for reports
                - currency (str): currency symbol
                - date_format (str): date format string
        """
        self.config = config or {}
        self._facility = self.config.get("facility_name", "Facility")
        self._currency = self.config.get("currency", "USD")
        self._date_fmt = self.config.get("date_format", "%Y-%m-%d")
        logger.info(
            "PeakReportingEngine v%s initialised (facility=%s, currency=%s)",
            self.engine_version, self._facility, self._currency,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_dashboard(
        self,
        kpi_data: Dict[str, Any],
        panels: Optional[List[DashboardPanel]] = None,
    ) -> DashboardData:
        """Generate a peak shaving KPI dashboard.

        Computes KPIs from raw data, assigns RAG status, determines
        trends, and assembles widgets into a dashboard layout.

        Args:
            kpi_data: Dictionary with raw KPI values.  Expected keys:
                - baseline_peak_kw, managed_peak_kw
                - baseline_charge_usd, actual_charge_usd
                - bess_discharge_kwh, bess_usable_kwh
                - avg_demand_kw, peak_demand_kw
                - annual_savings_usd, investment_cost_usd
                - prior_* variants for trend calculation
            panels: Panels to include (None = all).

        Returns:
            DashboardData with KPIs, widgets, and layout.
        """
        t0 = time.perf_counter()
        active_panels = panels or list(DashboardPanel)
        logger.info("Generating dashboard: %d panels", len(active_panels))

        # Compute KPIs
        kpis = self._compute_kpis(kpi_data)

        # Build widgets
        widgets = self._build_widgets(kpis, active_panels, kpi_data)

        dashboard = DashboardData(
            title=f"{self._facility} - Peak Shaving Dashboard",
            panels=active_panels,
            widgets=widgets,
            kpis=kpis,
        )
        dashboard.provenance_hash = _compute_hash(dashboard)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Dashboard generated: %d KPIs, %d widgets, %d panels, "
            "hash=%s (%.1f ms)",
            len(kpis), len(widgets), len(active_panels),
            dashboard.provenance_hash[:16], elapsed,
        )
        return dashboard

    def generate_report(
        self,
        report_type: ReportType,
        data: Dict[str, Any],
        export_format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> ReportOutput:
        """Generate a typed report from analysis data.

        Renders report sections with Markdown content, data tables,
        and KPIs.

        Args:
            report_type: Type of report to generate.
            data: Analysis data for the report.
            export_format: Output format.

        Returns:
            ReportOutput with rendered content.
        """
        t0 = time.perf_counter()
        logger.info(
            "Generating report: type=%s, format=%s",
            report_type.value, export_format.value,
        )

        title = self._get_report_title(report_type)
        sections = self._build_report_sections(report_type, data)

        # Render content
        content = self._render_content(title, sections, export_format)

        # Metadata
        metadata: Dict[str, Any] = {
            "facility": self._facility,
            "report_type": report_type.value,
            "export_format": export_format.value,
            "section_count": len(sections),
            "generated_by": f"PeakReportingEngine v{self.engine_version}",
            "pack": "PACK-038 Peak Shaving",
        }

        elapsed = (time.perf_counter() - t0) * 1000.0

        report = ReportOutput(
            report_type=report_type,
            title=title,
            sections=sections,
            export_format=export_format,
            content=content,
            metadata=metadata,
            processing_time_ms=round(elapsed, 2),
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Report generated: type=%s, %d sections, %d chars, "
            "hash=%s (%.1f ms)",
            report_type.value, len(sections), len(content),
            report.provenance_hash[:16], elapsed,
        )
        return report

    def generate_executive_summary(
        self,
        data: Dict[str, Any],
    ) -> ReportOutput:
        """Generate an executive summary for peak shaving results.

        Creates a concise summary with key findings, savings achieved,
        and recommendations.

        Args:
            data: Analysis data with results from all engines.

        Returns:
            ReportOutput with executive summary.
        """
        t0 = time.perf_counter()
        logger.info("Generating executive summary")

        # Extract key metrics
        peak_reduction = _decimal(data.get("peak_reduction_pct", 0))
        demand_savings = _decimal(data.get("demand_charge_savings_usd", 0))
        total_savings = _decimal(data.get("total_annual_savings_usd", 0))
        investment = _decimal(data.get("investment_cost_usd", 0))
        payback = _decimal(data.get("payback_years", 0))
        npv = _decimal(data.get("npv_usd", 0))
        irr = _decimal(data.get("irr_pct", 0))

        # Build summary
        summary_lines = [
            f"# {self._facility} - Peak Shaving Executive Summary",
            f"",
            f"**Report Date:** {utcnow().strftime(self._date_fmt)}",
            f"**Generated By:** PACK-038 Peak Shaving Pack v{self.engine_version}",
            f"",
            f"## Key Results",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Peak Demand Reduction | {_round_val(peak_reduction, 1)}% |",
            f"| Annual Demand Charge Savings | ${_round_val(demand_savings, 0):,} |",
            f"| Total Annual Savings | ${_round_val(total_savings, 0):,} |",
            f"| Investment Required | ${_round_val(investment, 0):,} |",
            f"| Simple Payback | {_round_val(payback, 1)} years |",
            f"| Net Present Value (NPV) | ${_round_val(npv, 0):,} |",
            f"| Internal Rate of Return (IRR) | {_round_val(irr, 1)}% |",
            f"",
        ]

        # Findings
        findings: List[str] = []
        if peak_reduction >= Decimal("20"):
            findings.append("Significant peak reduction achieved (>20%)")
        if demand_savings > Decimal("10000"):
            findings.append("Substantial demand charge savings identified")
        if payback < Decimal("5"):
            findings.append("Investment payback within 5 years")
        if npv > Decimal("0"):
            findings.append("Positive NPV indicates financially sound investment")

        if findings:
            summary_lines.append("## Key Findings")
            summary_lines.append("")
            for f in findings:
                summary_lines.append(f"- {f}")
            summary_lines.append("")

        # Recommendation
        if npv > Decimal("0") and payback < Decimal("5"):
            rec = "**Recommendation:** Proceed with investment. Strong financial returns expected."
        elif npv > Decimal("0"):
            rec = "**Recommendation:** Investment has positive NPV. Review timeline with stakeholders."
        else:
            rec = "**Recommendation:** Further analysis recommended before investment decision."

        summary_lines.append("## Recommendation")
        summary_lines.append("")
        summary_lines.append(rec)
        summary_lines.append("")

        summary_md = "\n".join(summary_lines)

        sections = [
            ReportSection(
                title="Executive Summary",
                content_markdown=summary_md,
                order=0,
            )
        ]

        elapsed = (time.perf_counter() - t0) * 1000.0

        report = ReportOutput(
            report_type=ReportType.PEAK_ASSESSMENT,
            title=f"{self._facility} - Executive Summary",
            sections=sections,
            executive_summary=summary_md,
            export_format=ReportFormat.MARKDOWN,
            content=summary_md,
            metadata={
                "facility": self._facility,
                "report_type": "executive_summary",
                "peak_reduction_pct": str(_round_val(peak_reduction, 1)),
                "total_savings_usd": str(_round_val(total_savings, 0)),
                "payback_years": str(_round_val(payback, 1)),
            },
            processing_time_ms=round(elapsed, 2),
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Executive summary: reduction=%.1f%%, savings=$%.0f, "
            "payback=%.1f yr, hash=%s (%.1f ms)",
            float(peak_reduction), float(total_savings), float(payback),
            report.provenance_hash[:16], elapsed,
        )
        return report

    def generate_verification_report(
        self,
        mv_data: Dict[str, Any],
    ) -> ReportOutput:
        """Generate M&V savings verification report.

        Creates a verification report per IPMVP methodology with
        baseline model fit metrics, savings calculations, and
        uncertainty analysis.

        Args:
            mv_data: M&V data including:
                - baseline_model_r2: Baseline model R-squared
                - baseline_cvrmse_pct: Baseline model CVRMSE (%)
                - baseline_demand_kw: Baseline period average demand
                - reporting_demand_kw: Reporting period average demand
                - adjusted_baseline_kw: Weather-adjusted baseline
                - ipmvp_option: IPMVP option (A, B, C, D)
                - savings_confidence_pct: Savings confidence level
                - measurement_period_months: M&V period (months)

        Returns:
            ReportOutput with verification report.
        """
        t0 = time.perf_counter()
        logger.info("Generating verification report")

        # Extract M&V data
        r2 = _decimal(mv_data.get("baseline_model_r2", 0))
        cvrmse = _decimal(mv_data.get("baseline_cvrmse_pct", 0))
        baseline_kw = _decimal(mv_data.get("baseline_demand_kw", 0))
        reporting_kw = _decimal(mv_data.get("reporting_demand_kw", 0))
        adjusted_kw = _decimal(mv_data.get("adjusted_baseline_kw", baseline_kw))
        ipmvp = mv_data.get("ipmvp_option", "B")
        confidence = _decimal(mv_data.get("savings_confidence_pct", 90))
        period = int(mv_data.get("measurement_period_months", 12))

        # Savings calculation
        savings_kw = adjusted_kw - reporting_kw
        savings_pct = _safe_pct(savings_kw, adjusted_kw)

        # Uncertainty
        # Simplified: uncertainty proportional to CVRMSE
        uncertainty_pct = cvrmse * Decimal("1.26")  # t-stat for 90% confidence
        savings_low = savings_kw * (Decimal("1") - uncertainty_pct / Decimal("100"))
        savings_high = savings_kw * (Decimal("1") + uncertainty_pct / Decimal("100"))

        # Model compliance
        if period >= 12:
            cvrmse_threshold = IPMVP_CVRMSE_MONTHLY
        else:
            cvrmse_threshold = IPMVP_CVRMSE_HOURLY

        model_compliant = cvrmse <= cvrmse_threshold and r2 >= Decimal("0.75")

        # Build report
        report_lines = [
            f"# {self._facility} - M&V Savings Verification Report",
            f"",
            f"**IPMVP Option:** {ipmvp}",
            f"**Measurement Period:** {period} months",
            f"**Report Date:** {utcnow().strftime(self._date_fmt)}",
            f"",
            f"## Baseline Model Quality",
            f"",
            f"| Metric | Value | Threshold | Status |",
            f"|--------|-------|-----------|--------|",
            f"| R-squared | {_round_val(r2, 4)} | >= 0.75 | {'PASS' if r2 >= Decimal('0.75') else 'FAIL'} |",
            f"| CV(RMSE) | {_round_val(cvrmse, 2)}% | <= {cvrmse_threshold}% | {'PASS' if cvrmse <= cvrmse_threshold else 'FAIL'} |",
            f"| Model Compliance | {'Compliant' if model_compliant else 'Non-Compliant'} | | |",
            f"",
            f"## Savings Determination",
            f"",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Adjusted Baseline Demand | {_round_val(adjusted_kw, 2)} kW |",
            f"| Reporting Period Demand | {_round_val(reporting_kw, 2)} kW |",
            f"| Demand Savings | {_round_val(savings_kw, 2)} kW |",
            f"| Savings Percentage | {_round_val(savings_pct, 1)}% |",
            f"",
            f"## Uncertainty Analysis ({_round_val(confidence, 0)}% Confidence)",
            f"",
            f"| Bound | Savings (kW) |",
            f"|-------|-------------|",
            f"| Lower ({_round_val(confidence, 0)}% CI) | {_round_val(savings_low, 2)} kW |",
            f"| Point Estimate | {_round_val(savings_kw, 2)} kW |",
            f"| Upper ({_round_val(confidence, 0)}% CI) | {_round_val(savings_high, 2)} kW |",
            f"| Uncertainty | +/-{_round_val(uncertainty_pct, 1)}% |",
            f"",
        ]

        content = "\n".join(report_lines)

        sections = [
            ReportSection(
                title="M&V Verification",
                content_markdown=content,
                order=0,
            )
        ]

        elapsed = (time.perf_counter() - t0) * 1000.0

        report = ReportOutput(
            report_type=ReportType.VERIFICATION,
            title=f"{self._facility} - M&V Verification Report",
            sections=sections,
            executive_summary=(
                f"Savings of {_round_val(savings_kw, 0)} kW ({_round_val(savings_pct, 1)}%) "
                f"verified using IPMVP Option {ipmvp}. "
                f"Model {'compliant' if model_compliant else 'non-compliant'} "
                f"with IPMVP standards."
            ),
            export_format=ReportFormat.MARKDOWN,
            content=content,
            metadata={
                "ipmvp_option": ipmvp,
                "model_r2": str(_round_val(r2, 4)),
                "cvrmse_pct": str(_round_val(cvrmse, 2)),
                "savings_kw": str(_round_val(savings_kw, 2)),
                "savings_pct": str(_round_val(savings_pct, 1)),
                "model_compliant": model_compliant,
            },
            processing_time_ms=round(elapsed, 2),
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Verification report: savings=%.0f kW (%.1f%%), "
            "R2=%.4f, CVRMSE=%.2f%%, compliant=%s, hash=%s (%.1f ms)",
            float(savings_kw), float(savings_pct),
            float(r2), float(cvrmse),
            model_compliant, report.provenance_hash[:16], elapsed,
        )
        return report

    def export_report(
        self,
        report: ReportOutput,
        export_format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> ReportOutput:
        """Export a report in the specified format.

        Converts Markdown content to the target format.

        Args:
            report: Report to export.
            export_format: Target format.

        Returns:
            ReportOutput with content in target format.
        """
        t0 = time.perf_counter()
        logger.info(
            "Exporting report: id=%s, format=%s",
            report.report_id, export_format.value,
        )

        if export_format == ReportFormat.MARKDOWN:
            content = report.content

        elif export_format == ReportFormat.HTML:
            content = self._markdown_to_html(report.content)

        elif export_format == ReportFormat.JSON:
            content = json.dumps(
                report.model_dump(mode="json"),
                indent=2, default=str,
            )

        elif export_format == ReportFormat.CSV:
            content = self._extract_csv_tables(report.sections)

        else:
            content = report.content

        elapsed = (time.perf_counter() - t0) * 1000.0

        exported = ReportOutput(
            report_id=report.report_id,
            report_type=report.report_type,
            title=report.title,
            sections=report.sections,
            executive_summary=report.executive_summary,
            export_format=export_format,
            content=content,
            metadata={**report.metadata, "exported_format": export_format.value},
            processing_time_ms=round(elapsed, 2),
        )
        exported.provenance_hash = _compute_hash(exported)

        logger.info(
            "Report exported: format=%s, %d chars, hash=%s (%.1f ms)",
            export_format.value, len(content),
            exported.provenance_hash[:16], elapsed,
        )
        return exported

    # ------------------------------------------------------------------ #
    # Internal: KPI Computation                                           #
    # ------------------------------------------------------------------ #

    def _compute_kpis(self, data: Dict[str, Any]) -> List[KPIMetric]:
        """Compute KPIs from raw data.

        Args:
            data: Raw KPI data.

        Returns:
            List of computed KPIMetric objects.
        """
        kpis: List[KPIMetric] = []

        # Peak Reduction %
        baseline_peak = _decimal(data.get("baseline_peak_kw", 0))
        managed_peak = _decimal(data.get("managed_peak_kw", 0))
        if baseline_peak > Decimal("0"):
            reduction = _safe_pct(baseline_peak - managed_peak, baseline_peak)
            prior_reduction = _decimal(data.get("prior_peak_reduction_pct", 0))
            kpis.append(self._make_kpi(
                "Peak Demand Reduction", reduction, "%",
                prior_reduction, "peak_reduction_pct", higher_is_better=True,
            ))

        # Demand Charge Savings
        baseline_charge = _decimal(data.get("baseline_charge_usd", 0))
        actual_charge = _decimal(data.get("actual_charge_usd", 0))
        savings = baseline_charge - actual_charge
        prior_savings = _decimal(data.get("prior_demand_savings_usd", 0))
        kpis.append(self._make_kpi(
            "Demand Charge Savings", savings, self._currency,
            prior_savings, "demand_charge_savings_pct", higher_is_better=True,
        ))

        # BESS Utilisation
        bess_discharge = _decimal(data.get("bess_discharge_kwh", 0))
        bess_usable = _decimal(data.get("bess_usable_kwh", 0))
        if bess_usable > Decimal("0"):
            util = _safe_pct(bess_discharge, bess_usable)
            prior_util = _decimal(data.get("prior_bess_utilisation_pct", 0))
            kpis.append(self._make_kpi(
                "BESS Utilisation", util, "%",
                prior_util, "bess_utilisation_pct", higher_is_better=True,
            ))

        # Load Factor
        avg_demand = _decimal(data.get("avg_demand_kw", 0))
        peak_demand = _decimal(data.get("peak_demand_kw", 0))
        if peak_demand > Decimal("0"):
            lf = _safe_pct(avg_demand, peak_demand)
            prior_lf = _decimal(data.get("prior_load_factor_pct", 0))
            kpis.append(self._make_kpi(
                "Load Factor", lf, "%",
                prior_lf, "load_factor_pct", higher_is_better=True,
            ))

        # ROI
        annual_savings = _decimal(data.get("annual_savings_usd", 0))
        investment = _decimal(data.get("investment_cost_usd", 0))
        if investment > Decimal("0"):
            roi = _safe_pct(annual_savings, investment)
            prior_roi = _decimal(data.get("prior_roi_pct", 0))
            kpis.append(self._make_kpi(
                "Return on Investment", roi, "%",
                prior_roi, "roi_pct", higher_is_better=True,
            ))

        return kpis

    def _make_kpi(
        self,
        name: str,
        value: Decimal,
        unit: str,
        prior_value: Decimal,
        threshold_key: str,
        higher_is_better: bool = True,
    ) -> KPIMetric:
        """Create a KPIMetric with trend and RAG.

        Args:
            name: KPI name.
            value: Current value.
            unit: Unit string.
            prior_value: Prior period value.
            threshold_key: Key into KPI_THRESHOLDS.
            higher_is_better: Whether higher values are better.

        Returns:
            Computed KPIMetric.
        """
        change = _safe_pct(value - prior_value, prior_value) if prior_value != Decimal("0") else Decimal("0")
        trend = self._compute_trend(change, higher_is_better)
        rag = self._compute_rag(value, threshold_key)

        return KPIMetric(
            name=name,
            value=_round_val(value, 2),
            unit=unit,
            target=Decimal("0"),
            prior_value=_round_val(prior_value, 2),
            change_pct=_round_val(change, 2),
            trend=trend,
            rag_status=rag,
            description=f"{name}: {_round_val(value, 1)}{unit}",
        )

    def _compute_trend(
        self,
        change_pct: Decimal,
        higher_is_better: bool,
    ) -> TrendDirection:
        """Compute trend direction from change percentage.

        Args:
            change_pct: Period-over-period change (%).
            higher_is_better: Whether positive change is improvement.

        Returns:
            TrendDirection.
        """
        if abs(change_pct) <= TREND_STABILITY_BAND:
            return TrendDirection.STABLE

        if higher_is_better:
            return TrendDirection.IMPROVING if change_pct > Decimal("0") else TrendDirection.DECLINING
        else:
            return TrendDirection.IMPROVING if change_pct < Decimal("0") else TrendDirection.DECLINING

    def _compute_rag(self, value: Decimal, threshold_key: str) -> str:
        """Compute RAG status from KPI value and thresholds.

        Args:
            value: KPI value.
            threshold_key: Key into KPI_THRESHOLDS.

        Returns:
            RAG status string.
        """
        thresholds = KPI_THRESHOLDS.get(threshold_key)
        if thresholds is None:
            return "green"

        if value >= thresholds["green"]:
            return "green"
        elif value >= thresholds["amber"]:
            return "amber"
        else:
            return "red"

    # ------------------------------------------------------------------ #
    # Internal: Widget Generation                                         #
    # ------------------------------------------------------------------ #

    def _build_widgets(
        self,
        kpis: List[KPIMetric],
        panels: List[DashboardPanel],
        data: Dict[str, Any],
    ) -> List[DashboardWidget]:
        """Build dashboard widgets from KPIs and panels.

        Args:
            kpis: Computed KPIs.
            panels: Active panels.
            data: Raw data for charts/tables.

        Returns:
            List of DashboardWidget objects.
        """
        widgets: List[DashboardWidget] = []
        row = 0

        # KPI widgets (first row)
        for col, kpi in enumerate(kpis[:4]):
            widgets.append(DashboardWidget(
                widget_type=WidgetType.KPI,
                panel=DashboardPanel.LOAD_PROFILE,
                title=kpi.name,
                data={
                    "value": str(kpi.value),
                    "unit": kpi.unit,
                    "change_pct": str(kpi.change_pct),
                    "trend": kpi.trend.value,
                    "rag": kpi.rag_status,
                },
                position={"row": row, "col": col},
                size={"rows": 1, "cols": 1},
            ))
        row += 1

        # Panel-specific widgets
        for panel in panels:
            if panel == DashboardPanel.LOAD_PROFILE:
                widgets.append(DashboardWidget(
                    widget_type=WidgetType.CHART,
                    panel=panel,
                    title="Load Profile (24-Hour)",
                    data={"chart_type": "area", "x_axis": "hour", "y_axis": "kW"},
                    position={"row": row, "col": 0},
                    size={"rows": 2, "cols": 2},
                ))
            elif panel == DashboardPanel.DEMAND_CHARGES:
                widgets.append(DashboardWidget(
                    widget_type=WidgetType.CHART,
                    panel=panel,
                    title="Monthly Demand Charges",
                    data={"chart_type": "bar", "x_axis": "month", "y_axis": "USD"},
                    position={"row": row, "col": 2},
                    size={"rows": 2, "cols": 2},
                ))
            elif panel == DashboardPanel.PEAK_EVENTS:
                widgets.append(DashboardWidget(
                    widget_type=WidgetType.TIMELINE,
                    panel=panel,
                    title="Peak Event History",
                    data={"events": data.get("peak_events", [])},
                    position={"row": row + 2, "col": 0},
                    size={"rows": 1, "cols": 4},
                ))
            elif panel == DashboardPanel.BESS_DISPATCH:
                widgets.append(DashboardWidget(
                    widget_type=WidgetType.CHART,
                    panel=panel,
                    title="BESS SOC and Dispatch",
                    data={"chart_type": "line", "x_axis": "time", "y_axis": "% / kW"},
                    position={"row": row + 3, "col": 0},
                    size={"rows": 2, "cols": 2},
                ))
            elif panel == DashboardPanel.FINANCIAL:
                widgets.append(DashboardWidget(
                    widget_type=WidgetType.TABLE,
                    panel=panel,
                    title="Financial Summary",
                    data={"columns": ["Metric", "Value"], "rows": []},
                    position={"row": row + 3, "col": 2},
                    size={"rows": 2, "cols": 2},
                ))

        return widgets

    # ------------------------------------------------------------------ #
    # Internal: Report Building                                           #
    # ------------------------------------------------------------------ #

    def _get_report_title(self, report_type: ReportType) -> str:
        """Get report title for type.

        Args:
            report_type: Report type.

        Returns:
            Report title string.
        """
        titles = {
            ReportType.LOAD_ANALYSIS: "Load Profile Analysis Report",
            ReportType.PEAK_ASSESSMENT: "Peak Demand Assessment Report",
            ReportType.BESS_SIZING: "BESS Sizing and Dispatch Report",
            ReportType.LOAD_SHIFTING: "Load Shifting Analysis Report",
            ReportType.CP_MANAGEMENT: "Coincident Peak Management Report",
            ReportType.FINANCIAL: "Financial Analysis Report",
            ReportType.VERIFICATION: "M&V Savings Verification Report",
        }
        return f"{self._facility} - {titles.get(report_type, 'Analysis Report')}"

    def _build_report_sections(
        self,
        report_type: ReportType,
        data: Dict[str, Any],
    ) -> List[ReportSection]:
        """Build report sections for the given type.

        Args:
            report_type: Report type.
            data: Analysis data.

        Returns:
            List of ReportSection objects.
        """
        sections: List[ReportSection] = []

        # Common header section
        sections.append(ReportSection(
            title="Introduction",
            content_markdown=(
                f"This report presents the {report_type.value.replace('_', ' ')} "
                f"analysis for {self._facility}.\n\n"
                f"**Analysis Date:** {utcnow().strftime(self._date_fmt)}\n\n"
                f"**Generated By:** PACK-038 Peak Shaving Pack v{self.engine_version}\n"
            ),
            order=0,
        ))

        # Type-specific sections
        if report_type == ReportType.PEAK_ASSESSMENT:
            sections.extend(self._peak_assessment_sections(data))
        elif report_type == ReportType.FINANCIAL:
            sections.extend(self._financial_sections(data))
        elif report_type == ReportType.BESS_SIZING:
            sections.extend(self._bess_sections(data))
        else:
            # Generic data section
            sections.append(ReportSection(
                title="Analysis Results",
                content_markdown=self._data_to_markdown_table(data),
                order=1,
            ))

        return sections

    def _peak_assessment_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Build peak assessment report sections.

        Args:
            data: Peak assessment data.

        Returns:
            List of sections.
        """
        peak_kw = _decimal(data.get("peak_demand_kw", 0))
        avg_kw = _decimal(data.get("avg_demand_kw", 0))
        lf = _safe_pct(avg_kw, peak_kw) if peak_kw > Decimal("0") else Decimal("0")

        return [
            ReportSection(
                title="Demand Profile Summary",
                content_markdown=(
                    f"| Metric | Value |\n"
                    f"|--------|-------|\n"
                    f"| Peak Demand | {_round_val(peak_kw, 0)} kW |\n"
                    f"| Average Demand | {_round_val(avg_kw, 0)} kW |\n"
                    f"| Load Factor | {_round_val(lf, 1)}% |\n"
                ),
                order=1,
            ),
            ReportSection(
                title="Peak Reduction Opportunities",
                content_markdown=(
                    "Based on load profile analysis, the following peak "
                    "reduction strategies are recommended:\n\n"
                    "1. **BESS Deployment** - Battery storage for peak clipping\n"
                    "2. **Load Shifting** - Schedule deferrable loads to off-peak\n"
                    "3. **Demand Limiting** - Automated demand control system\n"
                ),
                order=2,
            ),
        ]

    def _financial_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Build financial report sections.

        Args:
            data: Financial analysis data.

        Returns:
            List of sections.
        """
        npv = _decimal(data.get("npv_usd", 0))
        irr = _decimal(data.get("irr_pct", 0))
        payback = _decimal(data.get("payback_years", 0))

        return [
            ReportSection(
                title="Financial Summary",
                content_markdown=(
                    f"| Metric | Value |\n"
                    f"|--------|-------|\n"
                    f"| Net Present Value | ${_round_val(npv, 0):,} |\n"
                    f"| Internal Rate of Return | {_round_val(irr, 1)}% |\n"
                    f"| Simple Payback | {_round_val(payback, 1)} years |\n"
                ),
                order=1,
            ),
        ]

    def _bess_sections(self, data: Dict[str, Any]) -> List[ReportSection]:
        """Build BESS sizing report sections.

        Args:
            data: BESS analysis data.

        Returns:
            List of sections.
        """
        capacity_kw = _decimal(data.get("bess_capacity_kw", 0))
        capacity_kwh = _decimal(data.get("bess_capacity_kwh", 0))

        return [
            ReportSection(
                title="BESS Sizing Recommendation",
                content_markdown=(
                    f"| Parameter | Value |\n"
                    f"|-----------|-------|\n"
                    f"| Power Capacity | {_round_val(capacity_kw, 0)} kW |\n"
                    f"| Energy Capacity | {_round_val(capacity_kwh, 0)} kWh |\n"
                    f"| Duration | {_round_val(_safe_divide(capacity_kwh, capacity_kw), 1)} hours |\n"
                ),
                order=1,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Internal: Format Conversion                                         #
    # ------------------------------------------------------------------ #

    def _render_content(
        self,
        title: str,
        sections: List[ReportSection],
        export_format: ReportFormat,
    ) -> str:
        """Render report content in target format.

        Args:
            title: Report title.
            sections: Report sections.
            export_format: Target format.

        Returns:
            Rendered content string.
        """
        if export_format == ReportFormat.MARKDOWN:
            parts = [f"# {title}\n"]
            for section in sorted(sections, key=lambda s: s.order):
                if section.title:
                    parts.append(f"\n## {section.title}\n")
                parts.append(section.content_markdown)
            return "\n".join(parts)

        elif export_format == ReportFormat.HTML:
            md = self._render_content(title, sections, ReportFormat.MARKDOWN)
            return self._markdown_to_html(md)

        elif export_format == ReportFormat.JSON:
            return json.dumps({
                "title": title,
                "sections": [s.model_dump(mode="json") for s in sections],
            }, indent=2, default=str)

        elif export_format == ReportFormat.CSV:
            return self._extract_csv_tables(sections)

        return self._render_content(title, sections, ReportFormat.MARKDOWN)

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert Markdown to HTML with inline CSS.

        Simplified converter for headings, tables, lists, and bold.

        Args:
            markdown: Markdown content.

        Returns:
            HTML string with inline CSS.
        """
        lines = markdown.split("\n")
        html_parts = [
            "<html><head><style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "h1 { color: #2c3e50; } h2 { color: #34495e; }",
            ".kpi-green { color: #27ae60; } .kpi-amber { color: #f39c12; }",
            ".kpi-red { color: #e74c3c; }",
            "</style></head><body>",
        ]

        in_table = False
        for line in lines:
            stripped = line.strip()

            if stripped.startswith("# "):
                if in_table:
                    html_parts.append("</table>")
                    in_table = False
                html_parts.append(f"<h1>{stripped[2:]}</h1>")
            elif stripped.startswith("## "):
                if in_table:
                    html_parts.append("</table>")
                    in_table = False
                html_parts.append(f"<h2>{stripped[3:]}</h2>")
            elif stripped.startswith("| ") and "---" not in stripped:
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if not in_table:
                    html_parts.append("<table>")
                    in_table = True
                    tag = "th"
                else:
                    tag = "td"
                row_html = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
                html_parts.append(f"<tr>{row_html}</tr>")
            elif stripped.startswith("- "):
                if in_table:
                    html_parts.append("</table>")
                    in_table = False
                html_parts.append(f"<li>{stripped[2:]}</li>")
            elif stripped.startswith("**") and stripped.endswith("**"):
                html_parts.append(f"<p><strong>{stripped[2:-2]}</strong></p>")
            elif stripped:
                if in_table:
                    html_parts.append("</table>")
                    in_table = False
                # Replace inline bold
                text = stripped
                while "**" in text:
                    text = text.replace("**", "<strong>", 1)
                    text = text.replace("**", "</strong>", 1)
                html_parts.append(f"<p>{text}</p>")

        if in_table:
            html_parts.append("</table>")
        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    def _extract_csv_tables(self, sections: List[ReportSection]) -> str:
        """Extract CSV data from report sections.

        Parses Markdown tables and converts to CSV format.

        Args:
            sections: Report sections.

        Returns:
            CSV content string.
        """
        csv_parts: List[str] = []

        for section in sections:
            lines = section.content_markdown.split("\n")
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("| ") and "---" not in stripped:
                    cells = [c.strip() for c in stripped.split("|")[1:-1]]
                    csv_parts.append(",".join(f'"{c}"' for c in cells))

        return "\n".join(csv_parts) if csv_parts else "No tabular data"

    def _data_to_markdown_table(self, data: Dict[str, Any]) -> str:
        """Convert a flat dictionary to a Markdown table.

        Args:
            data: Key-value data.

        Returns:
            Markdown table string.
        """
        if not data:
            return "No data available."

        lines = ["| Key | Value |", "|-----|-------|"]
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                continue
            lines.append(f"| {key} | {value} |")

        return "\n".join(lines)
