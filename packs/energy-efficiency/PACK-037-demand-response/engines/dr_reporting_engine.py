# -*- coding: utf-8 -*-
"""
DRReportingEngine - PACK-037 Demand Response Engine 10
========================================================

Dashboard data aggregation, multi-format report generation, executive
summary production, and settlement package assembly for demand response
programmes.  Provides eight dashboard panels, seven report types, and
five export formats for comprehensive DR programme visibility.

Calculation Methodology:
    Dashboard KPIs:
        total_flexibility_kw  = sum(nominated_kw across programmes)
        total_events          = count(events in period)
        compliance_rate_pct   = compliant / total * 100
        total_revenue         = sum(event_revenue + capacity_revenue)
        total_avoided_co2e    = sum(event_avoided_co2e)
        avg_performance_ratio = mean(event performance ratios)
        net_income            = revenue - penalties - costs
        roi_pct               = net_income / investment * 100

    Executive Summary:
        Aggregates programme portfolio, seasonal performance,
        financial results, carbon impact, and strategic outlook
        into a C-suite-ready narrative with key metrics.

    Settlement Package:
        Compiles event-by-event performance data, baseline
        calculations, metering evidence, and compliance
        determination for programme settlement submissions.

Regulatory References:
    - FERC Order 745 - DR Compensation in Wholesale Markets
    - NAESB WEQ DR Business Practices (settlement data standards)
    - PJM Manual 18 - DR Measurement & Verification
    - ISO-NE FCM Performance Assessment Requirements
    - NYISO ICAP Reporting Requirements
    - EU EED Article 15 - DR Reporting Obligations
    - GHG Protocol - Corporate Standard (carbon reporting)

Zero-Hallucination:
    - All KPIs computed from deterministic formulas
    - Settlement data structured per NAESB/ISO standards
    - No LLM involvement in any numeric path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-037 Demand Response
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


class DashboardPanel(str, Enum):
    """Dashboard panel types for DR programme monitoring.

    FLEXIBILITY_PROFILE:  Load flexibility capacity and availability.
    PROGRAM_PORTFOLIO:    Programme enrollment and status overview.
    EVENT_HISTORY:        Historical DR event timeline and outcomes.
    REVENUE_TRACKER:      Revenue stream tracking and forecasting.
    CARBON_IMPACT:        Carbon impact metrics and trends.
    DER_PERFORMANCE:      DER asset performance monitoring.
    COMPLIANCE_MONITOR:   Compliance rate and risk monitoring.
    REVENUE_FORECAST:     Forward-looking revenue projections.
    """
    FLEXIBILITY_PROFILE = "flexibility_profile"
    PROGRAM_PORTFOLIO = "program_portfolio"
    EVENT_HISTORY = "event_history"
    REVENUE_TRACKER = "revenue_tracker"
    CARBON_IMPACT = "carbon_impact"
    DER_PERFORMANCE = "der_performance"
    COMPLIANCE_MONITOR = "compliance_monitor"
    REVENUE_FORECAST = "revenue_forecast"


class ReportType(str, Enum):
    """Report type for DR programme outputs.

    FLEXIBILITY_REPORT:   Load flexibility assessment report.
    PROGRAM_REPORT:       Programme enrollment and terms report.
    EVENT_REPORT:         Single event detailed report.
    REVENUE_REPORT:       Financial and revenue analysis report.
    CARBON_REPORT:        Carbon impact and avoided emissions report.
    EXECUTIVE_SUMMARY:    C-suite executive summary.
    SETTLEMENT_PACKAGE:   Programme settlement submission package.
    """
    FLEXIBILITY_REPORT = "flexibility_report"
    PROGRAM_REPORT = "program_report"
    EVENT_REPORT = "event_report"
    REVENUE_REPORT = "revenue_report"
    CARBON_REPORT = "carbon_report"
    EXECUTIVE_SUMMARY = "executive_summary"
    SETTLEMENT_PACKAGE = "settlement_package"


class ExportFormat(str, Enum):
    """Supported export formats for reports.

    MARKDOWN:  Markdown text with tables.
    HTML:      HTML with inline CSS.
    PDF:       PDF document (rendered from HTML).
    JSON:      Structured JSON object.
    CSV:       Comma-separated values (tabular data).
    """
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"


class WidgetType(str, Enum):
    """Dashboard widget types for data visualisation.

    KPI_CARD:   Single KPI metric card.
    BAR_CHART:  Vertical/horizontal bar chart.
    PIE_CHART:  Pie or donut chart.
    LINE_CHART: Time-series line chart.
    TABLE:      Data table.
    GAUGE:      Gauge/dial chart.
    TIMELINE:   Event timeline.
    HEATMAP:    Heat map grid.
    """
    KPI_CARD = "kpi_card"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    LINE_CHART = "line_chart"
    TABLE = "table"
    GAUGE = "gauge"
    TIMELINE = "timeline"
    HEATMAP = "heatmap"


class TrendDirection(str, Enum):
    """KPI trend direction indicator.

    IMPROVING:  Metric trending favourably.
    STABLE:     Metric within +/-5% of prior period.
    DECLINING:  Metric trending unfavourably.
    """
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Trend determination threshold.
TREND_THRESHOLD_PCT: Decimal = Decimal("5")


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class KPIMetric(BaseModel):
    """Single KPI metric for dashboard display.

    Attributes:
        name: Human-readable KPI name.
        value: Current metric value.
        unit: Unit of measurement.
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


class DashboardWidget(BaseModel):
    """Single dashboard widget definition.

    Attributes:
        widget_id: Widget identifier.
        widget_type: Widget visualisation type.
        title: Widget title.
        data: Widget data payload.
        panel: Dashboard panel this widget belongs to.
    """
    widget_id: str = Field(default_factory=_new_uuid, description="Widget ID")
    widget_type: WidgetType = Field(
        default=WidgetType.KPI_CARD, description="Widget type"
    )
    title: str = Field(default="", max_length=200, description="Widget title")
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Widget data"
    )
    panel: DashboardPanel = Field(
        default=DashboardPanel.FLEXIBILITY_PROFILE, description="Panel"
    )


class ProgrammeMetrics(BaseModel):
    """Metrics for a single DR programme.

    Attributes:
        programme_id: Programme identifier.
        programme_name: Programme name.
        nominated_kw: Nominated curtailment capacity (kW).
        events_count: Total events in period.
        compliant_events: Compliant event count.
        total_curtailment_kwh: Total curtailment (kWh).
        total_revenue: Total revenue (USD).
        total_penalties: Total penalties (USD).
        total_avoided_co2e_tonnes: Total avoided emissions (tCO2e).
        compliance_rate_pct: Compliance rate (%).
        avg_performance_ratio_pct: Average performance ratio (%).
    """
    programme_id: str = Field(default="", description="Programme ID")
    programme_name: str = Field(default="", max_length=500)
    nominated_kw: Decimal = Field(default=Decimal("0"), ge=0)
    events_count: int = Field(default=0, ge=0)
    compliant_events: int = Field(default=0, ge=0)
    total_curtailment_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    total_revenue: Decimal = Field(default=Decimal("0"), ge=0)
    total_penalties: Decimal = Field(default=Decimal("0"), ge=0)
    total_avoided_co2e_tonnes: Decimal = Field(default=Decimal("0"), ge=0)
    compliance_rate_pct: Decimal = Field(default=Decimal("0"))
    avg_performance_ratio_pct: Decimal = Field(default=Decimal("0"))


class EventSummary(BaseModel):
    """Summary data for a single DR event.

    Attributes:
        event_id: Event identifier.
        event_date: Event date/time.
        programme_name: Programme name.
        duration_hours: Duration (hours).
        baseline_kwh: Baseline load (kWh).
        actual_kwh: Actual load (kWh).
        curtailment_kwh: Curtailment achieved (kWh).
        performance_ratio_pct: Performance ratio (%).
        compliance_status: Compliance status.
        revenue: Revenue earned (USD).
        penalty: Penalty incurred (USD).
        avoided_co2e_tonnes: Avoided emissions (tCO2e).
    """
    event_id: str = Field(default="", description="Event ID")
    event_date: datetime = Field(default_factory=_utcnow)
    programme_name: str = Field(default="", max_length=500)
    duration_hours: Decimal = Field(default=Decimal("1"))
    baseline_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    actual_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    curtailment_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    performance_ratio_pct: Decimal = Field(default=Decimal("0"))
    compliance_status: str = Field(default="pending")
    revenue: Decimal = Field(default=Decimal("0"), ge=0)
    penalty: Decimal = Field(default=Decimal("0"), ge=0)
    avoided_co2e_tonnes: Decimal = Field(default=Decimal("0"), ge=0)


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class DashboardData(BaseModel):
    """Complete dashboard data payload.

    Attributes:
        dashboard_id: Dashboard instance identifier.
        reporting_period: Period label.
        panels: Dashboard panel data.
        kpis: Top-level KPI metrics.
        widgets: Dashboard widgets.
        generated_at: Generation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    dashboard_id: str = Field(default_factory=_new_uuid)
    reporting_period: str = Field(default="")
    panels: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    kpis: List[KPIMetric] = Field(default_factory=list)
    widgets: List[DashboardWidget] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class ReportOutput(BaseModel):
    """Generated report output.

    Attributes:
        report_id: Report identifier.
        report_type: Type of report.
        title: Report title.
        format: Export format.
        content: Report content (markdown/html/json string).
        metadata: Report metadata.
        generated_at: Generation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    report_id: str = Field(default_factory=_new_uuid)
    report_type: ReportType = Field(default=ReportType.EXECUTIVE_SUMMARY)
    title: str = Field(default="", max_length=500)
    format: ExportFormat = Field(default=ExportFormat.MARKDOWN)
    content: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class ExecutiveSummary(BaseModel):
    """Executive summary report output.

    Attributes:
        summary_id: Summary identifier.
        reporting_period: Period label.
        total_programmes: Number of active programmes.
        total_nominated_kw: Total nominated capacity (kW).
        total_events: Total DR events.
        overall_compliance_pct: Overall compliance rate (%).
        total_revenue: Total revenue (USD).
        total_penalties: Total penalties (USD).
        net_income: Net income (USD).
        total_avoided_co2e_tonnes: Total avoided emissions (tCO2e).
        roi_pct: Portfolio ROI (%).
        key_findings: Key findings list.
        recommendations: Strategic recommendations.
        programme_rankings: Programme rankings by performance.
        content_md: Markdown-formatted executive summary.
        generated_at: Generation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    summary_id: str = Field(default_factory=_new_uuid)
    reporting_period: str = Field(default="")
    total_programmes: int = Field(default=0, ge=0)
    total_nominated_kw: Decimal = Field(default=Decimal("0"))
    total_events: int = Field(default=0, ge=0)
    overall_compliance_pct: Decimal = Field(default=Decimal("0"))
    total_revenue: Decimal = Field(default=Decimal("0"))
    total_penalties: Decimal = Field(default=Decimal("0"))
    net_income: Decimal = Field(default=Decimal("0"))
    total_avoided_co2e_tonnes: Decimal = Field(default=Decimal("0"))
    roi_pct: Decimal = Field(default=Decimal("0"))
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    programme_rankings: List[Dict[str, Any]] = Field(default_factory=list)
    content_md: str = Field(default="")
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class SettlementPackage(BaseModel):
    """DR programme settlement submission package.

    Attributes:
        package_id: Package identifier.
        programme_id: Programme identifier.
        programme_name: Programme name.
        settlement_period: Settlement period label.
        events: Event-level settlement data.
        total_nominated_kwh: Total nominated energy (kWh).
        total_curtailment_kwh: Total verified curtailment (kWh).
        total_performance_ratio_pct: Overall performance ratio (%).
        total_revenue_claimed: Total revenue claimed (USD).
        total_penalties_assessed: Total penalties assessed (USD).
        net_settlement: Net settlement amount (USD).
        compliance_summary: Compliance summary statistics.
        verification_method: M&V method used.
        certifications: List of certification statements.
        generated_at: Generation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    package_id: str = Field(default_factory=_new_uuid)
    programme_id: str = Field(default="")
    programme_name: str = Field(default="", max_length=500)
    settlement_period: str = Field(default="")
    events: List[EventSummary] = Field(default_factory=list)
    total_nominated_kwh: Decimal = Field(default=Decimal("0"))
    total_curtailment_kwh: Decimal = Field(default=Decimal("0"))
    total_performance_ratio_pct: Decimal = Field(default=Decimal("0"))
    total_revenue_claimed: Decimal = Field(default=Decimal("0"))
    total_penalties_assessed: Decimal = Field(default=Decimal("0"))
    net_settlement: Decimal = Field(default=Decimal("0"))
    compliance_summary: Dict[str, int] = Field(default_factory=dict)
    verification_method: str = Field(default="interval_metering")
    certifications: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DRReportingEngine:
    """Reporting engine for demand response programmes.

    Aggregates data across DR programmes into dashboards, generates
    multi-format reports, produces executive summaries, and assembles
    settlement packages for programme submissions.

    Usage::

        engine = DRReportingEngine()
        dashboard = engine.generate_dashboard(programmes, events, "2026-Q2")
        report = engine.generate_report(
            programmes, events, ReportType.REVENUE_REPORT
        )
        summary = engine.generate_executive_summary(
            programmes, events, "2026-Q2", investment
        )
        settlement = engine.generate_settlement(
            programme, events, "2026-Jul"
        )
        exported = engine.export_report(report, ExportFormat.HTML)

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise DRReportingEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - company_name (str): company name for reports
                - currency_symbol (str): currency symbol (default "$")
                - default_format (str): default export format
        """
        self.config = config or {}
        self._company = self.config.get("company_name", "Organisation")
        self._currency = self.config.get("currency_symbol", "$")
        self._default_format = ExportFormat(
            self.config.get("default_format", ExportFormat.MARKDOWN.value)
        )
        logger.info(
            "DRReportingEngine v%s initialised (company=%s)",
            self.engine_version, self._company,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_dashboard(
        self,
        programmes: List[ProgrammeMetrics],
        events: List[EventSummary],
        reporting_period: str = "",
    ) -> DashboardData:
        """Generate dashboard data with all eight panels.

        Aggregates programme and event data into KPI cards, charts,
        tables, and timeline widgets across all dashboard panels.

        Args:
            programmes: List of programme metrics.
            events: List of event summaries.
            reporting_period: Period label.

        Returns:
            DashboardData with panels, KPIs, and widgets.
        """
        t0 = time.perf_counter()
        logger.info(
            "Generating dashboard: %d programmes, %d events, period=%s",
            len(programmes), len(events), reporting_period,
        )

        # Top-level KPIs
        kpis = self._compute_kpis(programmes, events)

        # Build panels
        panels: Dict[str, Dict[str, Any]] = {}
        widgets: List[DashboardWidget] = []

        # Panel 1: Flexibility Profile
        flex_panel, flex_widgets = self._build_flexibility_panel(programmes)
        panels[DashboardPanel.FLEXIBILITY_PROFILE.value] = flex_panel
        widgets.extend(flex_widgets)

        # Panel 2: Programme Portfolio
        prog_panel, prog_widgets = self._build_portfolio_panel(programmes)
        panels[DashboardPanel.PROGRAM_PORTFOLIO.value] = prog_panel
        widgets.extend(prog_widgets)

        # Panel 3: Event History
        event_panel, event_widgets = self._build_event_panel(events)
        panels[DashboardPanel.EVENT_HISTORY.value] = event_panel
        widgets.extend(event_widgets)

        # Panel 4: Revenue Tracker
        rev_panel, rev_widgets = self._build_revenue_panel(programmes, events)
        panels[DashboardPanel.REVENUE_TRACKER.value] = rev_panel
        widgets.extend(rev_widgets)

        # Panel 5: Carbon Impact
        carbon_panel, carbon_widgets = self._build_carbon_panel(events)
        panels[DashboardPanel.CARBON_IMPACT.value] = carbon_panel
        widgets.extend(carbon_widgets)

        # Panel 6: DER Performance
        der_panel, der_widgets = self._build_der_panel(programmes)
        panels[DashboardPanel.DER_PERFORMANCE.value] = der_panel
        widgets.extend(der_widgets)

        # Panel 7: Compliance Monitor
        comp_panel, comp_widgets = self._build_compliance_panel(programmes, events)
        panels[DashboardPanel.COMPLIANCE_MONITOR.value] = comp_panel
        widgets.extend(comp_widgets)

        # Panel 8: Revenue Forecast
        forecast_panel, forecast_widgets = self._build_forecast_panel(programmes)
        panels[DashboardPanel.REVENUE_FORECAST.value] = forecast_panel
        widgets.extend(forecast_widgets)

        dashboard = DashboardData(
            reporting_period=reporting_period,
            panels=panels,
            kpis=kpis,
            widgets=widgets,
        )
        dashboard.provenance_hash = _compute_hash(dashboard)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Dashboard generated: %d KPIs, %d widgets, %d panels, "
            "hash=%s (%.1f ms)",
            len(kpis), len(widgets), len(panels),
            dashboard.provenance_hash[:16], elapsed,
        )
        return dashboard

    def generate_report(
        self,
        programmes: List[ProgrammeMetrics],
        events: List[EventSummary],
        report_type: ReportType,
        fmt: Optional[ExportFormat] = None,
    ) -> ReportOutput:
        """Generate a report of the specified type.

        Produces structured content for the chosen report type in
        the requested export format.

        Args:
            programmes: Programme metrics.
            events: Event summaries.
            report_type: Type of report to generate.
            fmt: Export format (default: markdown).

        Returns:
            ReportOutput with formatted content.
        """
        t0 = time.perf_counter()
        output_format = fmt or self._default_format
        logger.info(
            "Generating report: type=%s, format=%s",
            report_type.value, output_format.value,
        )

        title = self._get_report_title(report_type)
        content = self._render_report_content(
            programmes, events, report_type, output_format,
        )

        report = ReportOutput(
            report_type=report_type,
            title=title,
            format=output_format,
            content=content,
            metadata={
                "programmes": len(programmes),
                "events": len(events),
                "engine_version": self.engine_version,
                "company": self._company,
            },
        )
        report.provenance_hash = _compute_hash(report)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Report generated: type=%s, format=%s, length=%d, "
            "hash=%s (%.1f ms)",
            report_type.value, output_format.value, len(content),
            report.provenance_hash[:16], elapsed,
        )
        return report

    def generate_executive_summary(
        self,
        programmes: List[ProgrammeMetrics],
        events: List[EventSummary],
        reporting_period: str = "",
        total_investment: Decimal = Decimal("0"),
    ) -> ExecutiveSummary:
        """Generate an executive summary for C-suite stakeholders.

        Aggregates performance, financial, and carbon metrics with
        key findings and strategic recommendations.

        Args:
            programmes: Programme metrics.
            events: Event summaries.
            reporting_period: Period label.
            total_investment: Total DR investment (USD).

        Returns:
            ExecutiveSummary with metrics and narrative.
        """
        t0 = time.perf_counter()
        logger.info(
            "Generating executive summary: %d programmes, %d events",
            len(programmes), len(events),
        )

        # Aggregate metrics
        total_kw = sum((p.nominated_kw for p in programmes), Decimal("0"))
        total_events = sum((p.events_count for p in programmes), 0)
        total_compliant = sum((p.compliant_events for p in programmes), 0)
        compliance_pct = _safe_pct(_decimal(total_compliant), _decimal(total_events))

        total_revenue = sum((p.total_revenue for p in programmes), Decimal("0"))
        total_penalties = sum((p.total_penalties for p in programmes), Decimal("0"))
        net_income = total_revenue - total_penalties
        roi_pct = _safe_pct(net_income, total_investment) if total_investment > Decimal("0") else Decimal("0")

        total_co2e = sum(
            (p.total_avoided_co2e_tonnes for p in programmes), Decimal("0")
        )

        # Key findings
        findings = self._generate_findings(
            programmes, total_events, compliance_pct,
            net_income, total_co2e,
        )

        # Recommendations
        recommendations = self._generate_exec_recommendations(
            programmes, compliance_pct, net_income, roi_pct,
        )

        # Programme rankings
        rankings = self._rank_programmes(programmes)

        # Markdown content
        content_md = self._render_executive_md(
            reporting_period, total_kw, total_events,
            compliance_pct, total_revenue, total_penalties,
            net_income, total_co2e, roi_pct,
            findings, recommendations, rankings,
        )

        summary = ExecutiveSummary(
            reporting_period=reporting_period,
            total_programmes=len(programmes),
            total_nominated_kw=_round_val(total_kw, 2),
            total_events=total_events,
            overall_compliance_pct=_round_val(compliance_pct, 2),
            total_revenue=_round_val(total_revenue, 2),
            total_penalties=_round_val(total_penalties, 2),
            net_income=_round_val(net_income, 2),
            total_avoided_co2e_tonnes=_round_val(total_co2e, 6),
            roi_pct=_round_val(roi_pct, 2),
            key_findings=findings,
            recommendations=recommendations,
            programme_rankings=rankings,
            content_md=content_md,
        )
        summary.provenance_hash = _compute_hash(summary)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Executive summary: %d programmes, compliance=%.1f%%, "
            "net=%.2f, CO2e=%.4f tCO2e, hash=%s (%.1f ms)",
            len(programmes), float(compliance_pct), float(net_income),
            float(total_co2e), summary.provenance_hash[:16], elapsed,
        )
        return summary

    def generate_settlement(
        self,
        programme: ProgrammeMetrics,
        events: List[EventSummary],
        settlement_period: str = "",
    ) -> SettlementPackage:
        """Generate a settlement package for programme submission.

        Compiles event-level data, compliance determinations, and
        financial claims into a structured settlement package.

        Args:
            programme: Programme metrics.
            events: Event summaries for this programme.
            settlement_period: Settlement period label.

        Returns:
            SettlementPackage ready for submission.
        """
        t0 = time.perf_counter()
        logger.info(
            "Generating settlement: programme=%s, %d events, period=%s",
            programme.programme_name, len(events), settlement_period,
        )

        # Totals
        total_nominated = sum(
            (e.baseline_kwh for e in events), Decimal("0")
        )
        total_curtailment = sum(
            (e.curtailment_kwh for e in events), Decimal("0")
        )
        total_revenue = sum((e.revenue for e in events), Decimal("0"))
        total_penalties = sum((e.penalty for e in events), Decimal("0"))
        net = total_revenue - total_penalties

        # Performance ratio
        perf_ratio = _safe_pct(total_curtailment, total_nominated)

        # Compliance summary
        status_counts: Dict[str, int] = {}
        for e in events:
            status = e.compliance_status
            status_counts[status] = status_counts.get(status, 0) + 1

        # Certifications
        certs = [
            f"Settlement period: {settlement_period}",
            f"Programme: {programme.programme_name} ({programme.programme_id})",
            f"Total events: {len(events)}",
            f"Verification method: interval_metering",
            f"Data integrity: SHA-256 provenance hashed",
            f"Generated: {_utcnow().isoformat()}",
            f"Engine version: {self.engine_version}",
        ]

        package = SettlementPackage(
            programme_id=programme.programme_id,
            programme_name=programme.programme_name,
            settlement_period=settlement_period,
            events=events,
            total_nominated_kwh=_round_val(total_nominated, 2),
            total_curtailment_kwh=_round_val(total_curtailment, 2),
            total_performance_ratio_pct=_round_val(perf_ratio, 2),
            total_revenue_claimed=_round_val(total_revenue, 2),
            total_penalties_assessed=_round_val(total_penalties, 2),
            net_settlement=_round_val(net, 2),
            compliance_summary=status_counts,
            verification_method="interval_metering",
            certifications=certs,
        )
        package.provenance_hash = _compute_hash(package)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Settlement package: programme=%s, events=%d, "
            "net=%.2f, ratio=%.1f%%, hash=%s (%.1f ms)",
            programme.programme_name, len(events), float(net),
            float(perf_ratio), package.provenance_hash[:16], elapsed,
        )
        return package

    def export_report(
        self,
        report: ReportOutput,
        fmt: ExportFormat,
    ) -> ReportOutput:
        """Export a report to a different format.

        Converts existing report content to the requested format.

        Args:
            report: Existing report output.
            fmt: Target export format.

        Returns:
            New ReportOutput in the requested format.
        """
        t0 = time.perf_counter()
        logger.info(
            "Exporting report: id=%s, from=%s, to=%s",
            report.report_id, report.format.value, fmt.value,
        )

        if fmt == ExportFormat.JSON:
            content = json.dumps(
                report.model_dump(mode="json"), indent=2, default=str,
            )
        elif fmt == ExportFormat.HTML:
            content = self._markdown_to_html(report.content)
        elif fmt == ExportFormat.CSV:
            content = self._extract_csv(report.content)
        elif fmt == ExportFormat.PDF:
            content = self._markdown_to_html(report.content)
        else:
            content = report.content

        exported = ReportOutput(
            report_id=report.report_id,
            report_type=report.report_type,
            title=report.title,
            format=fmt,
            content=content,
            metadata=report.metadata,
        )
        exported.provenance_hash = _compute_hash(exported)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Export complete: format=%s, length=%d, hash=%s (%.1f ms)",
            fmt.value, len(content), exported.provenance_hash[:16], elapsed,
        )
        return exported

    # ------------------------------------------------------------------ #
    # Internal: KPI Computation                                           #
    # ------------------------------------------------------------------ #

    def _compute_kpis(
        self,
        programmes: List[ProgrammeMetrics],
        events: List[EventSummary],
    ) -> List[KPIMetric]:
        """Compute top-level dashboard KPIs.

        Args:
            programmes: Programme metrics.
            events: Event summaries.

        Returns:
            List of KPIMetric objects.
        """
        total_kw = sum((p.nominated_kw for p in programmes), Decimal("0"))
        total_events = len(events)
        compliant = sum(
            1 for e in events if e.compliance_status == "compliant"
        )
        compliance_pct = _safe_pct(_decimal(compliant), _decimal(total_events))
        total_revenue = sum((e.revenue for e in events), Decimal("0"))
        total_penalties = sum((e.penalty for e in events), Decimal("0"))
        net_income = total_revenue - total_penalties
        total_co2e = sum((e.avoided_co2e_tonnes for e in events), Decimal("0"))
        total_curtailment = sum((e.curtailment_kwh for e in events), Decimal("0"))

        avg_ratio = Decimal("0")
        if events:
            avg_ratio = sum(
                (e.performance_ratio_pct for e in events), Decimal("0")
            ) / _decimal(total_events)

        return [
            KPIMetric(
                name="Total Flexibility", value=_round_val(total_kw, 0),
                unit="kW", trend=TrendDirection.STABLE,
            ),
            KPIMetric(
                name="Total Events", value=_decimal(total_events),
                unit="events", trend=TrendDirection.STABLE,
            ),
            KPIMetric(
                name="Compliance Rate", value=_round_val(compliance_pct, 1),
                unit="%", trend=TrendDirection.STABLE,
            ),
            KPIMetric(
                name="Total Revenue", value=_round_val(total_revenue, 2),
                unit="USD", trend=TrendDirection.STABLE,
            ),
            KPIMetric(
                name="Net Income", value=_round_val(net_income, 2),
                unit="USD", trend=TrendDirection.STABLE,
            ),
            KPIMetric(
                name="Avoided CO2e", value=_round_val(total_co2e, 4),
                unit="tCO2e", trend=TrendDirection.STABLE,
            ),
            KPIMetric(
                name="Total Curtailment", value=_round_val(total_curtailment, 0),
                unit="kWh", trend=TrendDirection.STABLE,
            ),
            KPIMetric(
                name="Avg Performance", value=_round_val(avg_ratio, 1),
                unit="%", trend=TrendDirection.STABLE,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Internal: Panel Builders                                            #
    # ------------------------------------------------------------------ #

    def _build_flexibility_panel(
        self,
        programmes: List[ProgrammeMetrics],
    ) -> Tuple[Dict[str, Any], List[DashboardWidget]]:
        """Build Flexibility Profile panel."""
        total_kw = sum((p.nominated_kw for p in programmes), Decimal("0"))
        by_programme = {
            p.programme_name: str(_round_val(p.nominated_kw, 2))
            for p in programmes
        }
        panel_data = {
            "total_nominated_kw": str(_round_val(total_kw, 2)),
            "by_programme": by_programme,
            "programme_count": len(programmes),
        }
        widgets = [
            DashboardWidget(
                widget_type=WidgetType.GAUGE, title="Total Flexibility (kW)",
                data={"value": str(_round_val(total_kw, 0))},
                panel=DashboardPanel.FLEXIBILITY_PROFILE,
            ),
            DashboardWidget(
                widget_type=WidgetType.BAR_CHART, title="Flexibility by Programme",
                data=by_programme,
                panel=DashboardPanel.FLEXIBILITY_PROFILE,
            ),
        ]
        return panel_data, widgets

    def _build_portfolio_panel(
        self,
        programmes: List[ProgrammeMetrics],
    ) -> Tuple[Dict[str, Any], List[DashboardWidget]]:
        """Build Programme Portfolio panel."""
        rows = []
        for p in programmes:
            rows.append({
                "programme": p.programme_name,
                "nominated_kw": str(_round_val(p.nominated_kw, 2)),
                "events": p.events_count,
                "compliance_pct": str(_round_val(p.compliance_rate_pct, 1)),
                "revenue": str(_round_val(p.total_revenue, 2)),
            })
        panel_data = {"programmes": rows, "count": len(programmes)}
        widgets = [
            DashboardWidget(
                widget_type=WidgetType.TABLE, title="Programme Portfolio",
                data={"rows": rows},
                panel=DashboardPanel.PROGRAM_PORTFOLIO,
            ),
        ]
        return panel_data, widgets

    def _build_event_panel(
        self,
        events: List[EventSummary],
    ) -> Tuple[Dict[str, Any], List[DashboardWidget]]:
        """Build Event History panel."""
        panel_data = {
            "total_events": len(events),
            "recent_events": [
                {
                    "id": e.event_id,
                    "date": e.event_date.isoformat() if e.event_date else "",
                    "programme": e.programme_name,
                    "curtailment_kwh": str(_round_val(e.curtailment_kwh, 2)),
                    "status": e.compliance_status,
                }
                for e in events[-10:]
            ],
        }
        widgets = [
            DashboardWidget(
                widget_type=WidgetType.TIMELINE, title="Event History",
                data=panel_data,
                panel=DashboardPanel.EVENT_HISTORY,
            ),
        ]
        return panel_data, widgets

    def _build_revenue_panel(
        self,
        programmes: List[ProgrammeMetrics],
        events: List[EventSummary],
    ) -> Tuple[Dict[str, Any], List[DashboardWidget]]:
        """Build Revenue Tracker panel."""
        total_rev = sum((p.total_revenue for p in programmes), Decimal("0"))
        total_pen = sum((p.total_penalties for p in programmes), Decimal("0"))
        net = total_rev - total_pen
        panel_data = {
            "total_revenue": str(_round_val(total_rev, 2)),
            "total_penalties": str(_round_val(total_pen, 2)),
            "net_income": str(_round_val(net, 2)),
        }
        widgets = [
            DashboardWidget(
                widget_type=WidgetType.KPI_CARD, title="Net Income",
                data={"value": str(_round_val(net, 2)), "unit": "USD"},
                panel=DashboardPanel.REVENUE_TRACKER,
            ),
            DashboardWidget(
                widget_type=WidgetType.BAR_CHART, title="Revenue vs Penalties",
                data={"revenue": str(_round_val(total_rev, 2)),
                      "penalties": str(_round_val(total_pen, 2))},
                panel=DashboardPanel.REVENUE_TRACKER,
            ),
        ]
        return panel_data, widgets

    def _build_carbon_panel(
        self,
        events: List[EventSummary],
    ) -> Tuple[Dict[str, Any], List[DashboardWidget]]:
        """Build Carbon Impact panel."""
        total_co2e = sum((e.avoided_co2e_tonnes for e in events), Decimal("0"))
        panel_data = {
            "total_avoided_co2e_tonnes": str(_round_val(total_co2e, 6)),
            "events_with_carbon": sum(
                1 for e in events if e.avoided_co2e_tonnes > Decimal("0")
            ),
        }
        widgets = [
            DashboardWidget(
                widget_type=WidgetType.KPI_CARD, title="Avoided CO2e",
                data={"value": str(_round_val(total_co2e, 4)), "unit": "tCO2e"},
                panel=DashboardPanel.CARBON_IMPACT,
            ),
        ]
        return panel_data, widgets

    def _build_der_panel(
        self,
        programmes: List[ProgrammeMetrics],
    ) -> Tuple[Dict[str, Any], List[DashboardWidget]]:
        """Build DER Performance panel."""
        panel_data = {
            "total_programmes": len(programmes),
            "avg_performance": str(_round_val(
                sum((p.avg_performance_ratio_pct for p in programmes), Decimal("0"))
                / _decimal(max(len(programmes), 1)), 1,
            )),
        }
        widgets = [
            DashboardWidget(
                widget_type=WidgetType.GAUGE, title="Avg DER Performance",
                data=panel_data,
                panel=DashboardPanel.DER_PERFORMANCE,
            ),
        ]
        return panel_data, widgets

    def _build_compliance_panel(
        self,
        programmes: List[ProgrammeMetrics],
        events: List[EventSummary],
    ) -> Tuple[Dict[str, Any], List[DashboardWidget]]:
        """Build Compliance Monitor panel."""
        total_events = sum((p.events_count for p in programmes), 0)
        total_compliant = sum((p.compliant_events for p in programmes), 0)
        compliance_pct = _safe_pct(_decimal(total_compliant), _decimal(total_events))
        panel_data = {
            "compliance_rate_pct": str(_round_val(compliance_pct, 1)),
            "total_events": total_events,
            "compliant_events": total_compliant,
        }
        widgets = [
            DashboardWidget(
                widget_type=WidgetType.GAUGE, title="Compliance Rate",
                data={"value": str(_round_val(compliance_pct, 1)), "unit": "%"},
                panel=DashboardPanel.COMPLIANCE_MONITOR,
            ),
        ]
        return panel_data, widgets

    def _build_forecast_panel(
        self,
        programmes: List[ProgrammeMetrics],
    ) -> Tuple[Dict[str, Any], List[DashboardWidget]]:
        """Build Revenue Forecast panel."""
        monthly_rev = sum(
            (p.total_revenue for p in programmes), Decimal("0")
        )
        annual_est = monthly_rev * Decimal("4")
        panel_data = {
            "current_revenue": str(_round_val(monthly_rev, 2)),
            "annual_estimate": str(_round_val(annual_est, 2)),
        }
        widgets = [
            DashboardWidget(
                widget_type=WidgetType.LINE_CHART, title="Revenue Forecast",
                data=panel_data,
                panel=DashboardPanel.REVENUE_FORECAST,
            ),
        ]
        return panel_data, widgets

    # ------------------------------------------------------------------ #
    # Internal: Report Rendering                                          #
    # ------------------------------------------------------------------ #

    def _get_report_title(self, report_type: ReportType) -> str:
        """Get human-readable report title.

        Args:
            report_type: Report type.

        Returns:
            Title string.
        """
        titles = {
            ReportType.FLEXIBILITY_REPORT: "Load Flexibility Assessment Report",
            ReportType.PROGRAM_REPORT: "DR Programme Enrollment Report",
            ReportType.EVENT_REPORT: "DR Event Detailed Report",
            ReportType.REVENUE_REPORT: "Revenue and Financial Analysis Report",
            ReportType.CARBON_REPORT: "Carbon Impact Assessment Report",
            ReportType.EXECUTIVE_SUMMARY: "Executive Summary",
            ReportType.SETTLEMENT_PACKAGE: "Settlement Package",
        }
        return titles.get(report_type, "Demand Response Report")

    def _render_report_content(
        self,
        programmes: List[ProgrammeMetrics],
        events: List[EventSummary],
        report_type: ReportType,
        fmt: ExportFormat,
    ) -> str:
        """Render report content in the specified format.

        Args:
            programmes: Programme metrics.
            events: Event summaries.
            report_type: Report type.
            fmt: Export format.

        Returns:
            Formatted content string.
        """
        # Build markdown first, then convert if needed
        md = self._render_markdown(programmes, events, report_type)

        if fmt == ExportFormat.MARKDOWN:
            return md
        if fmt == ExportFormat.HTML:
            return self._markdown_to_html(md)
        if fmt == ExportFormat.JSON:
            return json.dumps({
                "report_type": report_type.value,
                "programmes": [p.model_dump(mode="json") for p in programmes],
                "events_count": len(events),
                "content_md": md,
            }, indent=2, default=str)
        if fmt == ExportFormat.CSV:
            return self._extract_csv(md)
        return md

    def _render_markdown(
        self,
        programmes: List[ProgrammeMetrics],
        events: List[EventSummary],
        report_type: ReportType,
    ) -> str:
        """Render report content as Markdown.

        Args:
            programmes: Programme metrics.
            events: Event summaries.
            report_type: Report type.

        Returns:
            Markdown string.
        """
        title = self._get_report_title(report_type)
        lines: List[str] = [
            f"# {title}",
            f"",
            f"**Organisation:** {self._company}",
            f"**Generated:** {_utcnow().isoformat()}",
            f"**Engine:** DRReportingEngine v{self.engine_version}",
            f"",
        ]

        # Summary section
        total_kw = sum((p.nominated_kw for p in programmes), Decimal("0"))
        total_revenue = sum((p.total_revenue for p in programmes), Decimal("0"))
        total_penalties = sum((p.total_penalties for p in programmes), Decimal("0"))
        total_co2e = sum(
            (e.avoided_co2e_tonnes for e in events), Decimal("0")
        )

        lines.extend([
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Active Programmes | {len(programmes)} |",
            f"| Total Flexibility | {_round_val(total_kw, 0)} kW |",
            f"| Total Events | {len(events)} |",
            f"| Total Revenue | {self._currency}{_round_val(total_revenue, 2)} |",
            f"| Total Penalties | {self._currency}{_round_val(total_penalties, 2)} |",
            f"| Net Income | {self._currency}{_round_val(total_revenue - total_penalties, 2)} |",
            f"| Avoided CO2e | {_round_val(total_co2e, 4)} tCO2e |",
            "",
        ])

        # Programme table
        if programmes:
            lines.extend([
                "## Programme Performance",
                "",
                "| Programme | Nominated kW | Events | Compliance | Revenue |",
                "|-----------|-------------|--------|------------|---------|",
            ])
            for p in programmes:
                lines.append(
                    f"| {p.programme_name} | {_round_val(p.nominated_kw, 0)} | "
                    f"{p.events_count} | {_round_val(p.compliance_rate_pct, 1)}% | "
                    f"{self._currency}{_round_val(p.total_revenue, 2)} |"
                )
            lines.append("")

        # Provenance footer
        lines.extend([
            "---",
            f"*SHA-256 provenance hashing applied to all calculations.*",
            f"*Zero-hallucination: all values computed deterministically.*",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internal: Executive Summary                                         #
    # ------------------------------------------------------------------ #

    def _generate_findings(
        self,
        programmes: List[ProgrammeMetrics],
        total_events: int,
        compliance_pct: Decimal,
        net_income: Decimal,
        total_co2e: Decimal,
    ) -> List[str]:
        """Generate key findings for executive summary.

        Args:
            programmes: Programme metrics.
            total_events: Total events.
            compliance_pct: Overall compliance rate.
            net_income: Net income.
            total_co2e: Total avoided CO2e.

        Returns:
            List of finding strings.
        """
        findings: List[str] = []

        findings.append(
            f"Portfolio of {len(programmes)} DR programmes with "
            f"{total_events} events in the reporting period."
        )

        if compliance_pct >= Decimal("90"):
            findings.append(
                f"Overall compliance rate of {float(compliance_pct):.1f}% "
                f"exceeds programme requirements."
            )
        else:
            findings.append(
                f"Overall compliance rate of {float(compliance_pct):.1f}% "
                f"is below the 90% target threshold."
            )

        if net_income > Decimal("0"):
            findings.append(
                f"Net income of {self._currency}{float(net_income):,.2f} "
                f"demonstrates positive financial performance."
            )
        else:
            findings.append(
                f"Net loss of {self._currency}{float(abs(net_income)):,.2f} "
                f"indicates programme adjustments needed."
            )

        if total_co2e > Decimal("0"):
            findings.append(
                f"DR activities avoided {float(total_co2e):.4f} tCO2e, "
                f"contributing to decarbonization targets."
            )

        return findings

    def _generate_exec_recommendations(
        self,
        programmes: List[ProgrammeMetrics],
        compliance_pct: Decimal,
        net_income: Decimal,
        roi_pct: Decimal,
    ) -> List[str]:
        """Generate executive recommendations.

        Args:
            programmes: Programme metrics.
            compliance_pct: Overall compliance rate.
            net_income: Net income.
            roi_pct: Return on investment.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if compliance_pct < Decimal("85"):
            recs.append(
                "Invest in DER assets and automated controls to improve "
                "curtailment reliability and compliance rate."
            )

        if roi_pct < Decimal("10"):
            recs.append(
                "Review programme terms and negotiate improved capacity "
                "payment rates to improve return on investment."
            )

        low_performers = [
            p for p in programmes
            if p.compliance_rate_pct < Decimal("75")
        ]
        if low_performers:
            names = ", ".join(p.programme_name for p in low_performers[:3])
            recs.append(
                f"Address performance gaps in: {names}. Consider "
                f"reducing nominated capacity or adding DER resources."
            )

        high_performers = [
            p for p in programmes
            if p.compliance_rate_pct >= Decimal("95")
            and p.avg_performance_ratio_pct > Decimal("110")
        ]
        if high_performers:
            recs.append(
                "High-performing programmes show capacity for increased "
                "nomination. Evaluate capacity expansion opportunities."
            )

        if not recs:
            recs.append(
                "Programme portfolio is performing well. Continue "
                "monitoring and evaluate expansion opportunities."
            )

        return recs

    def _rank_programmes(
        self,
        programmes: List[ProgrammeMetrics],
    ) -> List[Dict[str, Any]]:
        """Rank programmes by net performance.

        Args:
            programmes: Programme metrics.

        Returns:
            Ranked list of programme summaries.
        """
        scored = []
        for p in programmes:
            net = p.total_revenue - p.total_penalties
            score = (
                p.compliance_rate_pct * Decimal("0.4")
                + p.avg_performance_ratio_pct * Decimal("0.3")
                + _safe_pct(net, p.total_revenue + Decimal("1")) * Decimal("0.3")
            )
            scored.append({
                "programme_name": p.programme_name,
                "compliance_pct": str(_round_val(p.compliance_rate_pct, 1)),
                "avg_performance_pct": str(_round_val(p.avg_performance_ratio_pct, 1)),
                "net_revenue": str(_round_val(net, 2)),
                "score": str(_round_val(score, 2)),
            })

        scored.sort(key=lambda x: _decimal(x["score"]), reverse=True)
        for i, s in enumerate(scored, 1):
            s["rank"] = i

        return scored

    def _render_executive_md(
        self,
        period: str,
        total_kw: Decimal,
        total_events: int,
        compliance_pct: Decimal,
        total_revenue: Decimal,
        total_penalties: Decimal,
        net_income: Decimal,
        total_co2e: Decimal,
        roi_pct: Decimal,
        findings: List[str],
        recommendations: List[str],
        rankings: List[Dict[str, Any]],
    ) -> str:
        """Render executive summary as Markdown.

        Args:
            Various aggregated metrics and analysis results.

        Returns:
            Markdown-formatted executive summary.
        """
        lines: List[str] = [
            "# Demand Response Executive Summary",
            "",
            f"**Organisation:** {self._company}",
            f"**Reporting Period:** {period}",
            f"**Generated:** {_utcnow().isoformat()}",
            "",
            "## Key Performance Indicators",
            "",
            "| KPI | Value |",
            "|-----|-------|",
            f"| Total Flexibility | {_round_val(total_kw, 0)} kW |",
            f"| Total Events | {total_events} |",
            f"| Compliance Rate | {_round_val(compliance_pct, 1)}% |",
            f"| Total Revenue | {self._currency}{_round_val(total_revenue, 2)} |",
            f"| Total Penalties | {self._currency}{_round_val(total_penalties, 2)} |",
            f"| Net Income | {self._currency}{_round_val(net_income, 2)} |",
            f"| Avoided CO2e | {_round_val(total_co2e, 4)} tCO2e |",
            f"| ROI | {_round_val(roi_pct, 1)}% |",
            "",
            "## Key Findings",
            "",
        ]
        for i, f in enumerate(findings, 1):
            lines.append(f"{i}. {f}")
        lines.append("")

        lines.append("## Recommendations")
        lines.append("")
        for i, r in enumerate(recommendations, 1):
            lines.append(f"{i}. {r}")
        lines.append("")

        if rankings:
            lines.extend([
                "## Programme Rankings",
                "",
                "| Rank | Programme | Compliance | Performance | Net Revenue | Score |",
                "|------|-----------|------------|-------------|-------------|-------|",
            ])
            for r in rankings:
                lines.append(
                    f"| {r.get('rank', '-')} | {r['programme_name']} | "
                    f"{r['compliance_pct']}% | {r['avg_performance_pct']}% | "
                    f"{self._currency}{r['net_revenue']} | {r['score']} |"
                )
            lines.append("")

        lines.extend([
            "---",
            "*Generated by DRReportingEngine. All calculations deterministic.*",
            "*SHA-256 provenance hashing applied.*",
        ])

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internal: Format Conversion                                         #
    # ------------------------------------------------------------------ #

    def _markdown_to_html(self, md: str) -> str:
        """Convert Markdown to basic HTML.

        Simple conversion for headings, tables, bold, and lists.

        Args:
            md: Markdown content.

        Returns:
            HTML string.
        """
        lines = md.split("\n")
        html_lines: List[str] = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:40px;line-height:1.6;}",
            "table{border-collapse:collapse;width:100%;margin:16px 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}",
            "th{background-color:#f4f4f4;font-weight:bold;}",
            "h1{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px;}",
            "h2{color:#34495e;margin-top:24px;}",
            "</style></head><body>",
        ]

        in_table = False
        in_list = False
        for line in lines:
            stripped = line.strip()

            if stripped.startswith("# "):
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                html_lines.append(f"<h1>{stripped[2:]}</h1>")
            elif stripped.startswith("## "):
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                html_lines.append(f"<h2>{stripped[3:]}</h2>")
            elif stripped.startswith("|") and "---" in stripped:
                continue
            elif stripped.startswith("|"):
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if not in_table:
                    html_lines.append("<table>")
                    html_lines.append(
                        "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"
                    )
                    in_table = True
                else:
                    html_lines.append(
                        "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                    )
            elif stripped.startswith("- ") or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] == "."):
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
                if not in_list:
                    html_lines.append("<ol>" if stripped[0].isdigit() else "<ul>")
                    in_list = True
                text = stripped.lstrip("0123456789.- ")
                html_lines.append(f"<li>{text}</li>")
            elif stripped == "":
                if in_list:
                    html_lines.append("</ol>")
                    in_list = False
                if in_table:
                    html_lines.append("</table>")
                    in_table = False
            elif stripped.startswith("**") and stripped.endswith("**"):
                html_lines.append(f"<p><strong>{stripped[2:-2]}</strong></p>")
            elif stripped.startswith("*") and stripped.endswith("*"):
                html_lines.append(f"<p><em>{stripped[1:-1]}</em></p>")
            elif stripped == "---":
                html_lines.append("<hr>")
            else:
                html_lines.append(f"<p>{stripped}</p>")

        if in_table:
            html_lines.append("</table>")
        if in_list:
            html_lines.append("</ol>")

        html_lines.append("</body></html>")
        return "\n".join(html_lines)

    def _extract_csv(self, md: str) -> str:
        """Extract tabular data from Markdown to CSV.

        Args:
            md: Markdown content with tables.

        Returns:
            CSV string.
        """
        lines = md.split("\n")
        csv_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and "---" not in stripped:
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                csv_lines.append(",".join(f'"{c}"' for c in cells))

        return "\n".join(csv_lines) if csv_lines else "No tabular data found"
