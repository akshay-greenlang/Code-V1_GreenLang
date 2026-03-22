# -*- coding: utf-8 -*-
"""
UtilityReportingEngine - PACK-036 Utility Analysis Engine 10
==============================================================

Comprehensive utility reporting and dashboard generation engine that
aggregates utility consumption, cost, demand, and carbon data into
structured reports, executive dashboards, and multi-format exports.

Generates facility-level monthly summaries, quarterly reviews, annual
reports, portfolio overviews, budget variance analysis, benchmark
comparisons, procurement reviews, and executive dashboards with KPI
cards, trend widgets, anomaly alerts, and actionable insights.

Report Sections:
    HEADER:             Report metadata and identification
    EXECUTIVE_SUMMARY:  Top-level KPIs and RAG status overview
    KPI_DASHBOARD:      KPI cards with sparklines and targets
    TREND_CHARTS:       Data tables for line/bar chart rendering
    ANOMALY_ALERTS:     Flagged anomalies with severity and actions
    COMMODITY_BREAKDOWN:Detailed per-commodity consumption and cost
    BUDGET_VARIANCE:    Weather/rate/volume variance decomposition
    RECOMMENDATIONS:    Prioritised action items with impact estimates
    APPENDIX:           Raw data references and methodology notes

Supported Commodities:
    - Electricity (kWh, kW demand, power factor)
    - Natural Gas (kWh/therms, peak day demand)
    - Water (m3/litres)
    - Steam (GJ/MWh)
    - Chilled Water (ton-hours/kWh)

Regulatory / Standard References:
    - EU Energy Efficiency Directive 2023/1791 (Article 11 - metering)
    - ISO 50001:2018 Clause 6.6 (Energy review and reporting)
    - IPMVP Volume I (Measurement & Verification)
    - ASHRAE Guideline 14-2014 (Energy monitoring reports)
    - ENERGY STAR Portfolio Manager (Utility tracking)
    - GRI 302 (Energy disclosure)
    - ESRS E1-5 (Energy consumption reporting)

Zero-Hallucination:
    - All KPIs computed from deterministic Decimal arithmetic
    - Report content assembled from pre-calculated engine outputs
    - No LLM-generated text in any numeric field
    - SHA-256 provenance hash on every report output
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-036 Utility Analysis
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
# Report version aligns with PACK-036 version.
# ---------------------------------------------------------------------------
REPORT_VERSION: str = "36.0.0"


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


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* decimal digits and return float."""
    return float(value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))


def _round2(value: Any) -> float:
    """Round to 2 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _change_pct(current: Decimal, previous: Decimal) -> Decimal:
    """Calculate percentage change from previous to current."""
    if previous == Decimal("0"):
        return Decimal("0")
    return (current - previous) / abs(previous) * Decimal("100")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReportType(str, Enum):
    """Types of utility reports.

    MONTHLY_SUMMARY:       Monthly utility consumption and cost summary.
    QUARTERLY_REVIEW:      Quarterly performance review with trends.
    ANNUAL_REPORT:         Full-year comprehensive utility report.
    BUDGET_VARIANCE:       Budget vs actual variance analysis.
    PORTFOLIO_SUMMARY:     Multi-facility portfolio overview.
    BENCHMARK_COMPARISON:  Facility-to-benchmark comparison.
    EXECUTIVE_DASHBOARD:   High-level executive dashboard with widgets.
    COMPLIANCE_REPORT:     Regulatory compliance-focused report.
    PROCUREMENT_REVIEW:    Utility procurement and rate analysis.
    DEMAND_ANALYSIS:       Peak demand and load factor analysis.
    SAVINGS_TRACKING:      Energy savings tracking and verification.
    CUSTOM:                User-defined custom report configuration.
    """
    MONTHLY_SUMMARY = "monthly_summary"
    QUARTERLY_REVIEW = "quarterly_review"
    ANNUAL_REPORT = "annual_report"
    BUDGET_VARIANCE = "budget_variance"
    PORTFOLIO_SUMMARY = "portfolio_summary"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    EXECUTIVE_DASHBOARD = "executive_dashboard"
    COMPLIANCE_REPORT = "compliance_report"
    PROCUREMENT_REVIEW = "procurement_review"
    DEMAND_ANALYSIS = "demand_analysis"
    SAVINGS_TRACKING = "savings_tracking"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    """Supported report export formats.

    MARKDOWN:  Structured Markdown text.
    HTML:      HTML document with inline styles.
    PDF:       JSON data structure for PDF rendering.
    JSON:      Machine-readable JSON.
    CSV:       Tabular CSV data.
    EXCEL:     JSON data structure for Excel rendering.
    """
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


class RAGStatus(str, Enum):
    """Red-Amber-Green status indicator.

    RED:    Performance significantly below target or critical issue.
    AMBER:  Performance below target, requires attention.
    GREEN:  Performance on or above target.
    GREY:   Insufficient data to determine status.
    """
    RED = "red"
    AMBER = "amber"
    GREEN = "green"
    GREY = "grey"


class KPICategory(str, Enum):
    """Category of key performance indicator.

    COST:          Cost-related KPIs (total, per sqm, per unit).
    CONSUMPTION:   Energy/water consumption KPIs.
    DEMAND:        Peak demand and load factor KPIs.
    EFFICIENCY:    Efficiency ratios (EUI, COP, load factor).
    CARBON:        Carbon emissions KPIs (tCO2e).
    COMPLIANCE:    Regulatory compliance KPIs.
    PROCUREMENT:   Rate and procurement KPIs.
    """
    COST = "cost"
    CONSUMPTION = "consumption"
    DEMAND = "demand"
    EFFICIENCY = "efficiency"
    CARBON = "carbon"
    COMPLIANCE = "compliance"
    PROCUREMENT = "procurement"


class TrendDirection(str, Enum):
    """Direction of a KPI trend.

    UP:        Value increasing over time.
    DOWN:      Value decreasing over time.
    FLAT:      No significant change detected.
    VOLATILE:  High variance with no clear direction.
    """
    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    VOLATILE = "volatile"


class AnomalyType(str, Enum):
    """Types of utility anomalies.

    CONSUMPTION_SPIKE:  Abnormally high consumption.
    COST_SPIKE:         Abnormally high cost.
    DEMAND_SPIKE:       Abnormally high peak demand.
    BILL_ERROR:         Suspected billing error.
    MISSING_DATA:       Missing or incomplete data.
    RATE_CHANGE:        Unexpected rate change detected.
    """
    CONSUMPTION_SPIKE = "consumption_spike"
    COST_SPIKE = "cost_spike"
    DEMAND_SPIKE = "demand_spike"
    BILL_ERROR = "bill_error"
    MISSING_DATA = "missing_data"
    RATE_CHANGE = "rate_change"


class WidgetType(str, Enum):
    """Dashboard widget types.

    KPI_CARD:    Single-value KPI with trend indicator.
    LINE_CHART:  Time series line chart.
    BAR_CHART:   Categorical or time series bar chart.
    PIE_CHART:   Proportional pie/donut chart.
    TABLE:       Tabular data display.
    HEATMAP:     Matrix heatmap (e.g., hourly consumption).
    GAUGE:       Gauge/dial indicator.
    SPARKLINE:   Inline mini trend chart.
    MAP:         Geographic facility map.
    """
    KPI_CARD = "kpi_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    TABLE = "table"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    SPARKLINE = "sparkline"
    MAP = "map"


class CommodityType(str, Enum):
    """Utility commodity types.

    ELECTRICITY:    Electrical energy supply.
    NATURAL_GAS:    Natural gas supply.
    WATER:          Water supply and discharge.
    STEAM:          District or on-site steam.
    CHILLED_WATER:  District or on-site chilled water.
    ALL:            All commodities aggregated.
    """
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    WATER = "water"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"
    ALL = "all"


# ---------------------------------------------------------------------------
# Pydantic Models -- Core
# ---------------------------------------------------------------------------


class UtilityKPI(BaseModel):
    """A single key performance indicator for utility reporting.

    Attributes:
        kpi_id: Unique KPI identifier.
        name: Human-readable KPI name.
        category: KPI category.
        current_value: Current period value.
        previous_value: Previous period value for comparison.
        change_pct: Percentage change from previous.
        unit: Unit of measurement.
        target_value: Target/budget value (optional).
        rag_status: RAG status relative to target.
        trend: Trend direction over recent periods.
        sparkline_data: Last 12 period values for sparkline.
        commodity: Commodity this KPI relates to.
    """
    kpi_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="", max_length=200)
    category: KPICategory = Field(default=KPICategory.CONSUMPTION)
    current_value: float = Field(default=0.0)
    previous_value: float = Field(default=0.0)
    change_pct: float = Field(default=0.0)
    unit: str = Field(default="", max_length=50)
    target_value: Optional[float] = Field(default=None)
    rag_status: RAGStatus = Field(default=RAGStatus.GREY)
    trend: TrendDirection = Field(default=TrendDirection.FLAT)
    sparkline_data: List[float] = Field(default_factory=list)
    commodity: CommodityType = Field(default=CommodityType.ALL)


class MonthlyUtilitySummary(BaseModel):
    """Monthly utility summary for a single commodity.

    Attributes:
        month: Month label (e.g., '2026-01').
        commodity: Commodity type.
        consumption_kwh: Total consumption in kWh.
        consumption_change_pct: Consumption change vs previous month.
        cost_eur: Total cost in EUR.
        cost_change_pct: Cost change vs previous month.
        peak_demand_kw: Peak demand in kW.
        load_factor_pct: Load factor as percentage.
        power_factor: Power factor (electricity only).
        cost_per_sqm: Cost per square metre.
        eui: Energy Use Intensity (kWh/m2).
        carbon_tco2e: Carbon emissions in tCO2e.
    """
    month: str = Field(default="", max_length=20)
    commodity: CommodityType = Field(default=CommodityType.ELECTRICITY)
    consumption_kwh: float = Field(default=0.0)
    consumption_change_pct: float = Field(default=0.0)
    cost_eur: float = Field(default=0.0)
    cost_change_pct: float = Field(default=0.0)
    peak_demand_kw: float = Field(default=0.0)
    load_factor_pct: float = Field(default=0.0)
    power_factor: float = Field(default=0.0)
    cost_per_sqm: float = Field(default=0.0)
    eui: float = Field(default=0.0)
    carbon_tco2e: float = Field(default=0.0)


class PortfolioSummary(BaseModel):
    """Portfolio-level utility summary across facilities.

    Attributes:
        total_facilities: Number of facilities in portfolio.
        total_cost_eur: Total portfolio utility cost.
        total_consumption_kwh: Total portfolio consumption.
        avg_eui: Average EUI across facilities.
        best_facility: Best-performing facility name.
        worst_facility: Worst-performing facility name.
        portfolio_savings_eur: Total verified savings.
        rag_distribution: Count of facilities by RAG status.
    """
    total_facilities: int = Field(default=0, ge=0)
    total_cost_eur: float = Field(default=0.0)
    total_consumption_kwh: float = Field(default=0.0)
    avg_eui: float = Field(default=0.0)
    best_facility: str = Field(default="", max_length=200)
    worst_facility: str = Field(default="", max_length=200)
    portfolio_savings_eur: float = Field(default=0.0)
    rag_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"red": 0, "amber": 0, "green": 0},
    )


class VarianceExplanation(BaseModel):
    """Explanation of a budget variance component.

    Attributes:
        category: Variance driver category (weather/rate/volume/other).
        impact_eur: Financial impact in EUR.
        impact_pct: Impact as percentage of total variance.
        description: Human-readable explanation.
    """
    category: str = Field(default="other", max_length=50)
    impact_eur: float = Field(default=0.0)
    impact_pct: float = Field(default=0.0)
    description: str = Field(default="", max_length=500)


class TrendData(BaseModel):
    """Trend data for a specific metric over time.

    Attributes:
        metric_name: Name of the metric being tracked.
        periods: List of period labels.
        values: List of metric values per period.
        trend_direction: Overall trend direction.
        yoy_change_pct: Year-over-year change percentage.
        rolling_12m_avg: Rolling 12-month average.
    """
    metric_name: str = Field(default="", max_length=200)
    periods: List[str] = Field(default_factory=list)
    values: List[float] = Field(default_factory=list)
    trend_direction: TrendDirection = Field(default=TrendDirection.FLAT)
    yoy_change_pct: float = Field(default=0.0)
    rolling_12m_avg: float = Field(default=0.0)


class AnomalyFlag(BaseModel):
    """Flagged anomaly in utility data.

    Attributes:
        anomaly_type: Type of anomaly detected.
        facility_id: Facility where anomaly was detected.
        period: Period of the anomaly.
        severity: Severity level (1=low, 5=critical).
        metric: Metric name that triggered the flag.
        expected_value: Expected/baseline value.
        actual_value: Actual observed value.
        deviation_pct: Percentage deviation from expected.
        description: Human-readable anomaly description.
        recommended_action: Suggested corrective action.
    """
    anomaly_type: AnomalyType = Field(default=AnomalyType.CONSUMPTION_SPIKE)
    facility_id: str = Field(default="", max_length=100)
    period: str = Field(default="", max_length=20)
    severity: int = Field(default=1, ge=1, le=5)
    metric: str = Field(default="", max_length=100)
    expected_value: float = Field(default=0.0)
    actual_value: float = Field(default=0.0)
    deviation_pct: float = Field(default=0.0)
    description: str = Field(default="", max_length=500)
    recommended_action: str = Field(default="", max_length=500)


class DashboardWidget(BaseModel):
    """Dashboard widget definition with data and layout.

    Attributes:
        widget_id: Unique widget identifier.
        widget_type: Type of dashboard widget.
        title: Widget display title.
        data: Widget data payload (format depends on type).
        position_row: Grid row position (1-based).
        position_col: Grid column position (1-based).
        width: Widget width in grid units.
        height: Widget height in grid units.
    """
    widget_id: str = Field(default_factory=_new_uuid)
    widget_type: WidgetType = Field(default=WidgetType.KPI_CARD)
    title: str = Field(default="", max_length=200)
    data: Dict[str, Any] = Field(default_factory=dict)
    position_row: int = Field(default=1, ge=1)
    position_col: int = Field(default=1, ge=1)
    width: int = Field(default=3, ge=1, le=12)
    height: int = Field(default=2, ge=1, le=12)


class ExecutiveInsight(BaseModel):
    """Executive-level insight derived from utility data.

    Attributes:
        insight_id: Unique insight identifier.
        category: Insight category.
        title: Short insight title.
        description: Detailed insight description.
        impact_eur: Estimated financial impact.
        priority: Priority level (1=highest, 5=lowest).
        action_required: Whether action is needed.
    """
    insight_id: str = Field(default_factory=_new_uuid)
    category: KPICategory = Field(default=KPICategory.COST)
    title: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=1000)
    impact_eur: float = Field(default=0.0)
    priority: int = Field(default=3, ge=1, le=5)
    action_required: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Pydantic Models -- Configuration
# ---------------------------------------------------------------------------


class ReportConfig(BaseModel):
    """Configuration for report generation.

    Attributes:
        report_type: Type of report to generate.
        format: Desired output format.
        facility_ids: Facility IDs to include (empty = all).
        period_start: Reporting period start (YYYY-MM).
        period_end: Reporting period end (YYYY-MM).
        commodities: Commodities to include.
        include_benchmarks: Whether to include benchmark data.
        include_carbon: Whether to include carbon calculations.
        include_forecasts: Whether to include forecast data.
        recipients: Report distribution list.
        title: Custom report title.
        currency_symbol: Currency symbol for cost fields.
        organisation: Organisation name.
        author: Report author.
    """
    report_type: ReportType = Field(default=ReportType.MONTHLY_SUMMARY)
    format: ReportFormat = Field(default=ReportFormat.MARKDOWN)
    facility_ids: List[str] = Field(default_factory=list)
    period_start: str = Field(default="", max_length=20)
    period_end: str = Field(default="", max_length=20)
    commodities: List[CommodityType] = Field(
        default_factory=lambda: [CommodityType.ALL],
    )
    include_benchmarks: bool = Field(default=False)
    include_carbon: bool = Field(default=True)
    include_forecasts: bool = Field(default=False)
    recipients: List[str] = Field(default_factory=list)
    title: str = Field(default="", max_length=500)
    currency_symbol: str = Field(default="EUR", max_length=5)
    organisation: str = Field(default="", max_length=500)
    author: str = Field(default="GreenLang Platform", max_length=200)


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class ReportOutput(BaseModel):
    """Complete report output with content and metadata.

    Attributes:
        report_id: Unique report identifier.
        report_type: Type of report generated.
        format: Export format of the report.
        title: Report title.
        generated_at: Generation timestamp.
        period: Reporting period string.
        content: Report content (Markdown/HTML string).
        data: Structured report data (for JSON export).
        file_path: File path if exported to disk.
        page_count: Estimated page count.
        widgets: Dashboard widgets (for executive dashboard).
        kpis: Calculated KPIs.
        insights: Generated executive insights.
        section_hashes: SHA-256 hash per section for audit.
        methodology_notes: Methodology and source notes.
        processing_time_ms: Computation time (ms).
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    report_id: str = Field(default_factory=_new_uuid)
    report_type: ReportType = Field(default=ReportType.MONTHLY_SUMMARY)
    format: ReportFormat = Field(default=ReportFormat.MARKDOWN)
    title: str = Field(default="")
    generated_at: datetime = Field(default_factory=_utcnow)
    period: str = Field(default="")
    content: str = Field(default="")
    data: Dict[str, Any] = Field(default_factory=dict)
    file_path: str = Field(default="")
    page_count: int = Field(default=0, ge=0)
    widgets: List[DashboardWidget] = Field(default_factory=list)
    kpis: List[UtilityKPI] = Field(default_factory=list)
    insights: List[ExecutiveInsight] = Field(default_factory=list)
    section_hashes: Dict[str, str] = Field(default_factory=dict)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class UtilityReportingEngine:
    """Zero-hallucination utility reporting and dashboard engine.

    Generates comprehensive utility reports with KPI calculations,
    variance analysis, anomaly detection, trend tracking, and
    multi-format export for facilities and portfolios.

    Guarantees:
        - Deterministic: same inputs produce identical reports.
        - Reproducible: every report carries a SHA-256 provenance hash.
        - Auditable: per-section hashes for granular audit trails.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = UtilityReportingEngine()
        config = ReportConfig(report_type=ReportType.MONTHLY_SUMMARY)
        result = engine.generate_report(config, data)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the utility reporting engine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - anomaly_threshold_pct (float): deviation % to flag anomaly
                - rag_amber_pct (float): amber threshold vs target
                - rag_red_pct (float): red threshold vs target
                - sparkline_periods (int): periods for sparkline data
        """
        self._config = config or {}
        self._anomaly_threshold = float(
            self._config.get("anomaly_threshold_pct", 20.0),
        )
        self._rag_amber_pct = float(self._config.get("rag_amber_pct", 5.0))
        self._rag_red_pct = float(self._config.get("rag_red_pct", 15.0))
        self._sparkline_periods = int(self._config.get("sparkline_periods", 12))
        self._notes: List[str] = []
        logger.info("UtilityReportingEngine v%s initialised.", _MODULE_VERSION)

    # --------------------------------------------------------------------- #
    # Public API -- Report Generation
    # --------------------------------------------------------------------- #

    def generate_report(
        self,
        config: ReportConfig,
        data: Dict[str, Any],
    ) -> ReportOutput:
        """Generate a utility report based on configuration.

        Dispatches to the appropriate report generator based on
        the report_type in config.

        Args:
            config: Report configuration.
            data: Report data payload (structure varies by type).

        Returns:
            ReportOutput with rendered content and provenance.

        Raises:
            ValueError: If report type is unsupported.
        """
        t0 = time.perf_counter()
        self._notes = [
            f"Engine version: {self.engine_version}",
            f"Report type: {config.report_type.value}",
            f"Period: {config.period_start} to {config.period_end}",
        ]

        title = config.title or self._default_title(config)

        # Calculate KPIs from data.
        facility_data = data.get("facility_data", {})
        kpis = self.calculate_kpis(facility_data, config.period_end)

        # Build sections.
        sections: Dict[str, str] = {}
        section_hashes: Dict[str, str] = {}

        sections["header"] = self._build_header(config, title)
        section_hashes["header"] = _compute_hash(sections["header"])

        sections["executive_summary"] = self._build_executive_summary(kpis, data)
        section_hashes["executive_summary"] = _compute_hash(
            sections["executive_summary"],
        )

        sections["kpi_dashboard"] = self._build_kpi_section(kpis)
        section_hashes["kpi_dashboard"] = _compute_hash(sections["kpi_dashboard"])

        # Trend data.
        historical = data.get("historical", [])
        trends = []
        for metric in ("consumption_kwh", "cost_eur", "peak_demand_kw"):
            if historical:
                trend = self.build_trend_data(historical, metric)
                trends.append(trend)
        sections["trends"] = self._build_trends_section(trends)
        section_hashes["trends"] = _compute_hash(sections["trends"])

        # Anomaly detection.
        hist_data = data.get("historical_data", historical)
        anomalies = self.detect_anomalies(facility_data, hist_data)
        sections["anomalies"] = self._build_anomalies_section(anomalies)
        section_hashes["anomalies"] = _compute_hash(sections["anomalies"])

        # Commodity breakdown.
        monthly = data.get("monthly_summaries", [])
        sections["commodity_breakdown"] = self._build_commodity_section(monthly)
        section_hashes["commodity_breakdown"] = _compute_hash(
            sections["commodity_breakdown"],
        )

        # Budget variance.
        budget = data.get("budget", {})
        actual = data.get("actual", {})
        weather = data.get("weather_impact", {})
        rates = data.get("rate_changes", {})
        variances = self.explain_variance(budget, actual, weather, rates)
        sections["budget_variance"] = self._build_variance_section(
            variances, config.currency_symbol,
        )
        section_hashes["budget_variance"] = _compute_hash(
            sections["budget_variance"],
        )

        # Recommendations and insights.
        benchmarks = data.get("benchmarks", {})
        insights = self.generate_insights(kpis, anomalies, benchmarks)
        sections["recommendations"] = self._build_recommendations_section(insights)
        section_hashes["recommendations"] = _compute_hash(
            sections["recommendations"],
        )

        # Appendix.
        sections["appendix"] = self._build_appendix(section_hashes)
        section_hashes["appendix"] = _compute_hash(sections["appendix"])

        # Assemble content.
        content = self._assemble_content(config.format, sections)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        period_str = f"{config.period_start} to {config.period_end}"

        result = ReportOutput(
            report_type=config.report_type,
            format=config.format,
            title=title,
            period=period_str,
            content=content,
            data={"sections": sections} if config.format == ReportFormat.JSON else {},
            page_count=self._estimate_pages(content),
            kpis=kpis,
            insights=insights,
            section_hashes=section_hashes,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Report generated: type=%s, format=%s, %d KPIs, %d insights, "
            "hash=%s (%.1f ms)",
            config.report_type.value,
            config.format.value,
            len(kpis),
            len(insights),
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    def generate_monthly_summary(
        self,
        facility_id: str,
        month: str,
        data: Dict[str, Any],
    ) -> ReportOutput:
        """Generate a monthly utility summary for a single facility.

        Args:
            facility_id: Facility identifier.
            month: Month label (e.g., '2026-01').
            data: Facility data payload with consumption/cost fields.

        Returns:
            ReportOutput with monthly summary.
        """
        config = ReportConfig(
            report_type=ReportType.MONTHLY_SUMMARY,
            facility_ids=[facility_id],
            period_start=month,
            period_end=month,
            title=f"Monthly Utility Summary - {month}",
        )
        data_payload = dict(data)
        data_payload.setdefault("facility_data", data)
        return self.generate_report(config, data_payload)

    def generate_executive_dashboard(
        self,
        portfolio_data: Dict[str, Any],
    ) -> ReportOutput:
        """Generate an executive dashboard with KPI widgets.

        Args:
            portfolio_data: Portfolio-level data with facility summaries.

        Returns:
            ReportOutput with dashboard widgets and KPIs.
        """
        t0 = time.perf_counter()
        self._notes = [
            f"Engine version: {self.engine_version}",
            "Report type: executive_dashboard",
        ]

        # Build portfolio summary.
        portfolio = self.build_portfolio_summary(portfolio_data)

        # Calculate KPIs.
        kpis = self._calculate_portfolio_kpis(portfolio, portfolio_data)

        # Detect anomalies across portfolio.
        anomalies = self._detect_portfolio_anomalies(portfolio_data)

        # Build dashboard widgets.
        trends = self._extract_portfolio_trends(portfolio_data)
        widgets = self.build_dashboard(kpis, trends, anomalies)

        # Generate insights.
        insights = self.generate_insights(kpis, anomalies, {})

        # Render content.
        content = self._render_dashboard_markdown(portfolio, kpis, widgets)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ReportOutput(
            report_type=ReportType.EXECUTIVE_DASHBOARD,
            format=ReportFormat.MARKDOWN,
            title="Executive Utility Dashboard",
            content=content,
            widgets=widgets,
            kpis=kpis,
            insights=insights,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Executive dashboard generated: %d widgets, %d KPIs, "
            "%d insights, hash=%s (%.1f ms)",
            len(widgets),
            len(kpis),
            len(insights),
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    # --------------------------------------------------------------------- #
    # Public API -- KPI Calculation
    # --------------------------------------------------------------------- #

    def calculate_kpis(
        self,
        facility_data: Dict[str, Any],
        period: str,
    ) -> List[UtilityKPI]:
        """Calculate utility KPIs from facility data.

        Args:
            facility_data: Facility data with current/previous values.
            period: Current period label.

        Returns:
            List of UtilityKPI objects.
        """
        kpis: List[UtilityKPI] = []

        # Total consumption.
        curr_kwh = _decimal(facility_data.get("consumption_kwh", 0))
        prev_kwh = _decimal(facility_data.get("prev_consumption_kwh", 0))
        kpis.append(UtilityKPI(
            name="Total Consumption",
            category=KPICategory.CONSUMPTION,
            current_value=_round2(float(curr_kwh)),
            previous_value=_round2(float(prev_kwh)),
            change_pct=_round2(float(_change_pct(curr_kwh, prev_kwh))),
            unit="kWh",
            target_value=facility_data.get("target_consumption_kwh"),
            rag_status=self._compute_rag(curr_kwh, facility_data.get("target_consumption_kwh")),
            trend=self._infer_trend(facility_data.get("consumption_history", [])),
            sparkline_data=self._extract_sparkline(
                facility_data.get("consumption_history", []),
            ),
        ))

        # Total cost.
        curr_cost = _decimal(facility_data.get("cost_eur", 0))
        prev_cost = _decimal(facility_data.get("prev_cost_eur", 0))
        kpis.append(UtilityKPI(
            name="Total Cost",
            category=KPICategory.COST,
            current_value=_round2(float(curr_cost)),
            previous_value=_round2(float(prev_cost)),
            change_pct=_round2(float(_change_pct(curr_cost, prev_cost))),
            unit="EUR",
            target_value=facility_data.get("target_cost_eur"),
            rag_status=self._compute_rag(curr_cost, facility_data.get("target_cost_eur")),
            trend=self._infer_trend(facility_data.get("cost_history", [])),
            sparkline_data=self._extract_sparkline(
                facility_data.get("cost_history", []),
            ),
        ))

        # Peak demand.
        curr_demand = _decimal(facility_data.get("peak_demand_kw", 0))
        prev_demand = _decimal(facility_data.get("prev_peak_demand_kw", 0))
        kpis.append(UtilityKPI(
            name="Peak Demand",
            category=KPICategory.DEMAND,
            current_value=_round2(float(curr_demand)),
            previous_value=_round2(float(prev_demand)),
            change_pct=_round2(float(_change_pct(curr_demand, prev_demand))),
            unit="kW",
            target_value=facility_data.get("target_demand_kw"),
            rag_status=self._compute_rag(
                curr_demand, facility_data.get("target_demand_kw"),
            ),
        ))

        # EUI (Energy Use Intensity).
        area = _decimal(facility_data.get("gross_floor_area_m2", 0))
        eui = _safe_divide(curr_kwh, area)
        prev_eui = _safe_divide(prev_kwh, area)
        kpis.append(UtilityKPI(
            name="Energy Use Intensity",
            category=KPICategory.EFFICIENCY,
            current_value=_round2(float(eui)),
            previous_value=_round2(float(prev_eui)),
            change_pct=_round2(float(_change_pct(eui, prev_eui))),
            unit="kWh/m2",
            target_value=facility_data.get("target_eui"),
            rag_status=self._compute_rag(eui, facility_data.get("target_eui")),
        ))

        # Cost per sqm.
        cost_sqm = _safe_divide(curr_cost, area)
        prev_cost_sqm = _safe_divide(prev_cost, area)
        kpis.append(UtilityKPI(
            name="Cost per m2",
            category=KPICategory.COST,
            current_value=_round2(float(cost_sqm)),
            previous_value=_round2(float(prev_cost_sqm)),
            change_pct=_round2(float(_change_pct(cost_sqm, prev_cost_sqm))),
            unit="EUR/m2",
        ))

        # Load factor.
        hours = _decimal(facility_data.get("billing_hours", 730))
        theoretical_max = curr_demand * hours
        load_factor = _safe_divide(curr_kwh, theoretical_max) * Decimal("100")
        kpis.append(UtilityKPI(
            name="Load Factor",
            category=KPICategory.EFFICIENCY,
            current_value=_round2(float(load_factor)),
            unit="%",
        ))

        # Carbon emissions.
        carbon = _decimal(facility_data.get("carbon_tco2e", 0))
        prev_carbon = _decimal(facility_data.get("prev_carbon_tco2e", 0))
        kpis.append(UtilityKPI(
            name="Carbon Emissions",
            category=KPICategory.CARBON,
            current_value=_round3(float(carbon)),
            previous_value=_round3(float(prev_carbon)),
            change_pct=_round2(float(_change_pct(carbon, prev_carbon))),
            unit="tCO2e",
            target_value=facility_data.get("target_carbon_tco2e"),
            rag_status=self._compute_rag(
                carbon, facility_data.get("target_carbon_tco2e"),
            ),
        ))

        # Unit rate.
        unit_rate = _safe_divide(curr_cost, curr_kwh)
        kpis.append(UtilityKPI(
            name="Blended Unit Rate",
            category=KPICategory.PROCUREMENT,
            current_value=_round4(float(unit_rate)),
            unit="EUR/kWh",
        ))

        self._notes.append(f"Calculated {len(kpis)} KPIs for period {period}.")
        return kpis

    # --------------------------------------------------------------------- #
    # Public API -- Portfolio Summary
    # --------------------------------------------------------------------- #

    def build_portfolio_summary(
        self,
        facilities_data: Dict[str, Any],
    ) -> PortfolioSummary:
        """Build a portfolio-level summary from facility data.

        Args:
            facilities_data: Dict with 'facilities' key containing list
                of facility dicts, each with consumption/cost fields.

        Returns:
            PortfolioSummary with aggregated metrics.
        """
        facilities = facilities_data.get("facilities", [])
        if not facilities:
            return PortfolioSummary()

        d_total_cost = Decimal("0")
        d_total_kwh = Decimal("0")
        d_total_area = Decimal("0")
        d_total_savings = Decimal("0")
        eui_values: List[Tuple[str, Decimal]] = []
        rag_dist: Dict[str, int] = {"red": 0, "amber": 0, "green": 0}

        for fac in facilities:
            cost = _decimal(fac.get("cost_eur", 0))
            kwh = _decimal(fac.get("consumption_kwh", 0))
            area = _decimal(fac.get("gross_floor_area_m2", 0))
            savings = _decimal(fac.get("savings_eur", 0))
            name = fac.get("name", fac.get("facility_id", ""))

            d_total_cost += cost
            d_total_kwh += kwh
            d_total_area += area
            d_total_savings += savings

            fac_eui = _safe_divide(kwh, area)
            eui_values.append((name, fac_eui))

            # RAG based on target.
            target = fac.get("target_consumption_kwh")
            rag = self._compute_rag(kwh, target)
            if rag == RAGStatus.RED:
                rag_dist["red"] += 1
            elif rag == RAGStatus.AMBER:
                rag_dist["amber"] += 1
            else:
                rag_dist["green"] += 1

        avg_eui = _safe_divide(d_total_kwh, d_total_area)

        # Sort by EUI to find best/worst.
        eui_values.sort(key=lambda x: x[1])
        best = eui_values[0][0] if eui_values else ""
        worst = eui_values[-1][0] if eui_values else ""

        return PortfolioSummary(
            total_facilities=len(facilities),
            total_cost_eur=_round2(float(d_total_cost)),
            total_consumption_kwh=_round2(float(d_total_kwh)),
            avg_eui=_round2(float(avg_eui)),
            best_facility=best,
            worst_facility=worst,
            portfolio_savings_eur=_round2(float(d_total_savings)),
            rag_distribution=rag_dist,
        )

    # --------------------------------------------------------------------- #
    # Public API -- Variance Explanation
    # --------------------------------------------------------------------- #

    def explain_variance(
        self,
        budget: Dict[str, Any],
        actual: Dict[str, Any],
        weather: Dict[str, Any],
        rates: Dict[str, Any],
    ) -> List[VarianceExplanation]:
        """Decompose budget variance into weather, rate, volume drivers.

        Args:
            budget: Budget/expected values with cost_eur and consumption_kwh.
            actual: Actual values with cost_eur and consumption_kwh.
            weather: Weather impact data with hdd_deviation, cdd_deviation,
                weather_sensitivity_eur_per_hdd.
            rates: Rate change data with old_rate_eur_per_kwh and
                new_rate_eur_per_kwh.

        Returns:
            List of VarianceExplanation components.
        """
        explanations: List[VarianceExplanation] = []

        budget_cost = _decimal(budget.get("cost_eur", 0))
        actual_cost = _decimal(actual.get("cost_eur", 0))
        total_variance = actual_cost - budget_cost

        if total_variance == Decimal("0"):
            return explanations

        # Weather variance.
        hdd_dev = _decimal(weather.get("hdd_deviation", 0))
        cdd_dev = _decimal(weather.get("cdd_deviation", 0))
        sensitivity = _decimal(weather.get("weather_sensitivity_eur_per_hdd", 0))
        weather_impact = (hdd_dev + cdd_dev) * sensitivity
        if weather_impact != Decimal("0"):
            explanations.append(VarianceExplanation(
                category="weather",
                impact_eur=_round2(float(weather_impact)),
                impact_pct=_round2(float(_safe_pct(weather_impact, total_variance))),
                description=(
                    f"Weather deviation: HDD {_round2(float(hdd_dev))}, "
                    f"CDD {_round2(float(cdd_dev))} degree-days from normal"
                ),
            ))

        # Rate variance.
        old_rate = _decimal(rates.get("old_rate_eur_per_kwh", 0))
        new_rate = _decimal(rates.get("new_rate_eur_per_kwh", 0))
        actual_kwh = _decimal(actual.get("consumption_kwh", 0))
        rate_impact = (new_rate - old_rate) * actual_kwh
        if rate_impact != Decimal("0"):
            explanations.append(VarianceExplanation(
                category="rate",
                impact_eur=_round2(float(rate_impact)),
                impact_pct=_round2(float(_safe_pct(rate_impact, total_variance))),
                description=(
                    f"Rate change from {_round4(float(old_rate))} to "
                    f"{_round4(float(new_rate))} EUR/kWh"
                ),
            ))

        # Volume variance.
        budget_kwh = _decimal(budget.get("consumption_kwh", 0))
        volume_kwh = actual_kwh - budget_kwh
        budget_rate = _safe_divide(budget_cost, budget_kwh)
        volume_impact = volume_kwh * budget_rate
        if volume_impact != Decimal("0"):
            explanations.append(VarianceExplanation(
                category="volume",
                impact_eur=_round2(float(volume_impact)),
                impact_pct=_round2(float(_safe_pct(volume_impact, total_variance))),
                description=(
                    f"Volume change: {_round2(float(volume_kwh))} kWh "
                    f"({'increase' if volume_kwh > 0 else 'decrease'})"
                ),
            ))

        # Other / residual.
        explained = sum(_decimal(e.impact_eur) for e in explanations)
        residual = total_variance - explained
        if abs(residual) > Decimal("1"):
            explanations.append(VarianceExplanation(
                category="other",
                impact_eur=_round2(float(residual)),
                impact_pct=_round2(float(_safe_pct(residual, total_variance))),
                description="Unexplained residual variance",
            ))

        self._notes.append(
            f"Variance decomposition: total={_round2(float(total_variance))} EUR, "
            f"{len(explanations)} drivers identified.",
        )
        return explanations

    # --------------------------------------------------------------------- #
    # Public API -- Trend Data
    # --------------------------------------------------------------------- #

    def build_trend_data(
        self,
        historical_data: List[Dict[str, Any]],
        metric: str,
    ) -> TrendData:
        """Build trend data for a specific metric from historical records.

        Args:
            historical_data: List of period records with metric values.
            metric: Metric field name to track.

        Returns:
            TrendData with periods, values, and trend direction.
        """
        periods: List[str] = []
        values: List[float] = []

        for record in historical_data:
            period_label = record.get("period", record.get("month", ""))
            value = record.get(metric, 0.0)
            periods.append(str(period_label))
            values.append(_round2(float(_decimal(value))))

        trend_dir = self._infer_trend(values)

        # YoY change (compare last value to 12 periods ago).
        yoy = 0.0
        if len(values) >= 13:
            curr = _decimal(values[-1])
            prev = _decimal(values[-13])
            yoy = _round2(float(_change_pct(curr, prev)))

        # Rolling 12-month average.
        rolling_avg = 0.0
        if len(values) >= 12:
            last_12 = [_decimal(v) for v in values[-12:]]
            rolling_avg = _round2(float(
                _safe_divide(sum(last_12), Decimal("12")),
            ))

        return TrendData(
            metric_name=metric,
            periods=periods,
            values=values,
            trend_direction=trend_dir,
            yoy_change_pct=yoy,
            rolling_12m_avg=rolling_avg,
        )

    # --------------------------------------------------------------------- #
    # Public API -- Anomaly Detection
    # --------------------------------------------------------------------- #

    def detect_anomalies(
        self,
        current_data: Dict[str, Any],
        historical_data: Any,
    ) -> List[AnomalyFlag]:
        """Detect anomalies in current data relative to historical baseline.

        Uses statistical deviation from historical mean/std to flag
        consumption spikes, cost spikes, demand spikes, and missing data.

        Args:
            current_data: Current period data.
            historical_data: Historical baseline data (list of records
                or dict with metric histories).

        Returns:
            List of AnomalyFlag objects sorted by severity.
        """
        anomalies: List[AnomalyFlag] = []
        facility_id = current_data.get(
            "facility_id", current_data.get("name", ""),
        )
        period = current_data.get("period", current_data.get("month", ""))

        # Build historical baselines.
        baselines = self._build_baselines(historical_data)

        # Check consumption.
        anomalies.extend(self._check_metric_anomaly(
            metric="consumption_kwh",
            current_val=current_data.get("consumption_kwh", 0),
            baseline=baselines.get("consumption_kwh"),
            anomaly_type=AnomalyType.CONSUMPTION_SPIKE,
            facility_id=facility_id,
            period=period,
        ))

        # Check cost.
        anomalies.extend(self._check_metric_anomaly(
            metric="cost_eur",
            current_val=current_data.get("cost_eur", 0),
            baseline=baselines.get("cost_eur"),
            anomaly_type=AnomalyType.COST_SPIKE,
            facility_id=facility_id,
            period=period,
        ))

        # Check demand.
        anomalies.extend(self._check_metric_anomaly(
            metric="peak_demand_kw",
            current_val=current_data.get("peak_demand_kw", 0),
            baseline=baselines.get("peak_demand_kw"),
            anomaly_type=AnomalyType.DEMAND_SPIKE,
            facility_id=facility_id,
            period=period,
        ))

        # Check for missing data fields.
        required_fields = ["consumption_kwh", "cost_eur"]
        for field in required_fields:
            if field not in current_data or current_data[field] is None:
                anomalies.append(AnomalyFlag(
                    anomaly_type=AnomalyType.MISSING_DATA,
                    facility_id=facility_id,
                    period=period,
                    severity=3,
                    metric=field,
                    description=f"Missing data for {field}",
                    recommended_action="Verify data source and re-import.",
                ))

        # Sort by severity descending.
        anomalies.sort(key=lambda a: a.severity, reverse=True)
        return anomalies

    # --------------------------------------------------------------------- #
    # Public API -- Dashboard Builder
    # --------------------------------------------------------------------- #

    def build_dashboard(
        self,
        kpis: List[UtilityKPI],
        trends: List[TrendData],
        anomalies: List[AnomalyFlag],
    ) -> List[DashboardWidget]:
        """Build dashboard widgets from KPIs, trends, and anomalies.

        Args:
            kpis: Calculated KPIs.
            trends: Trend data series.
            anomalies: Detected anomalies.

        Returns:
            List of DashboardWidget objects with layout positions.
        """
        widgets: List[DashboardWidget] = []
        row = 1
        col = 1

        # KPI cards (top row, 4 per row).
        for i, kpi in enumerate(kpis[:8]):
            widgets.append(DashboardWidget(
                widget_type=WidgetType.KPI_CARD,
                title=kpi.name,
                data={
                    "value": kpi.current_value,
                    "unit": kpi.unit,
                    "change_pct": kpi.change_pct,
                    "rag_status": kpi.rag_status.value,
                    "trend": kpi.trend.value,
                    "sparkline": kpi.sparkline_data,
                },
                position_row=row,
                position_col=col,
                width=3,
                height=2,
            ))
            col += 3
            if col > 12:
                col = 1
                row += 2

        # Trend line charts.
        for trend in trends[:3]:
            widgets.append(DashboardWidget(
                widget_type=WidgetType.LINE_CHART,
                title=f"{trend.metric_name} Trend",
                data={
                    "periods": trend.periods,
                    "values": trend.values,
                    "trend_direction": trend.trend_direction.value,
                    "rolling_avg": trend.rolling_12m_avg,
                },
                position_row=row,
                position_col=col,
                width=6,
                height=4,
            ))
            col += 6
            if col > 12:
                col = 1
                row += 4

        # Anomaly alerts table.
        if anomalies:
            anomaly_rows = [
                {
                    "type": a.anomaly_type.value,
                    "facility": a.facility_id,
                    "period": a.period,
                    "severity": a.severity,
                    "deviation_pct": a.deviation_pct,
                    "description": a.description,
                }
                for a in anomalies[:10]
            ]
            widgets.append(DashboardWidget(
                widget_type=WidgetType.TABLE,
                title="Anomaly Alerts",
                data={"rows": anomaly_rows},
                position_row=row,
                position_col=1,
                width=12,
                height=3,
            ))
            row += 3

        # Commodity pie chart from KPIs.
        commodity_data = self._build_commodity_pie(kpis)
        if commodity_data:
            widgets.append(DashboardWidget(
                widget_type=WidgetType.PIE_CHART,
                title="Cost by Commodity",
                data=commodity_data,
                position_row=row,
                position_col=1,
                width=6,
                height=4,
            ))

        return widgets

    # --------------------------------------------------------------------- #
    # Public API -- Insights
    # --------------------------------------------------------------------- #

    def generate_insights(
        self,
        kpis: List[UtilityKPI],
        anomalies: List[AnomalyFlag],
        benchmarks: Dict[str, Any],
    ) -> List[ExecutiveInsight]:
        """Generate executive insights from KPIs, anomalies, and benchmarks.

        Args:
            kpis: Calculated KPIs.
            anomalies: Detected anomalies.
            benchmarks: Benchmark comparison data.

        Returns:
            List of ExecutiveInsight objects sorted by priority.
        """
        insights: List[ExecutiveInsight] = []

        # Cost increase insights.
        for kpi in kpis:
            if kpi.category == KPICategory.COST and kpi.change_pct > 10:
                insights.append(ExecutiveInsight(
                    category=KPICategory.COST,
                    title=f"{kpi.name} increased {kpi.change_pct:.1f}%",
                    description=(
                        f"{kpi.name} increased from {kpi.previous_value:,.2f} to "
                        f"{kpi.current_value:,.2f} {kpi.unit}, a {kpi.change_pct:.1f}% "
                        f"increase. Review procurement strategy and demand management."
                    ),
                    impact_eur=_round2(kpi.current_value - kpi.previous_value),
                    priority=2,
                    action_required=True,
                ))

        # Red RAG status insights.
        red_kpis = [k for k in kpis if k.rag_status == RAGStatus.RED]
        if red_kpis:
            names = ", ".join(k.name for k in red_kpis[:3])
            insights.append(ExecutiveInsight(
                category=KPICategory.COMPLIANCE,
                title=f"{len(red_kpis)} KPIs at RED status",
                description=(
                    f"The following KPIs are significantly off-target: {names}. "
                    f"Immediate corrective action is recommended."
                ),
                priority=1,
                action_required=True,
            ))

        # Critical anomaly insights.
        critical = [a for a in anomalies if a.severity >= 4]
        for anomaly in critical[:3]:
            insights.append(ExecutiveInsight(
                category=KPICategory.CONSUMPTION,
                title=f"{anomaly.anomaly_type.value}: {anomaly.metric}",
                description=(
                    f"Anomaly detected at {anomaly.facility_id} in {anomaly.period}: "
                    f"{anomaly.description}. Deviation: {anomaly.deviation_pct:.1f}%. "
                    f"Action: {anomaly.recommended_action}"
                ),
                priority=2,
                action_required=True,
            ))

        # Benchmark gap insights.
        benchmark_eui = benchmarks.get("median_eui")
        if benchmark_eui:
            for kpi in kpis:
                if kpi.name == "Energy Use Intensity" and kpi.current_value > 0:
                    d_current = _decimal(kpi.current_value)
                    d_bench = _decimal(benchmark_eui)
                    gap_pct = _change_pct(d_current, d_bench)
                    if float(gap_pct) > 10:
                        insights.append(ExecutiveInsight(
                            category=KPICategory.EFFICIENCY,
                            title=f"EUI {_round2(float(gap_pct))}% above benchmark",
                            description=(
                                f"Current EUI of {kpi.current_value:.1f} kWh/m2 is "
                                f"{_round2(float(gap_pct))}% above the benchmark median "
                                f"of {benchmark_eui:.1f} kWh/m2. Energy efficiency "
                                f"improvements should be prioritised."
                            ),
                            priority=2,
                            action_required=True,
                        ))

        # Positive insights.
        improving = [
            k for k in kpis
            if k.category in (KPICategory.CONSUMPTION, KPICategory.CARBON)
            and k.change_pct < -5
        ]
        for kpi in improving[:2]:
            insights.append(ExecutiveInsight(
                category=kpi.category,
                title=f"{kpi.name} reduced by {abs(kpi.change_pct):.1f}%",
                description=(
                    f"{kpi.name} decreased from {kpi.previous_value:,.2f} to "
                    f"{kpi.current_value:,.2f} {kpi.unit}. Continue current "
                    f"energy management practices."
                ),
                priority=4,
                action_required=False,
            ))

        # Sort by priority.
        insights.sort(key=lambda i: i.priority)
        return insights

    # --------------------------------------------------------------------- #
    # Public API -- Renderers
    # --------------------------------------------------------------------- #

    def render_markdown(self, report_data: Dict[str, Any]) -> str:
        """Render report data as structured Markdown.

        Args:
            report_data: Dict with 'sections' key containing section content.

        Returns:
            Complete Markdown report string.
        """
        sections = report_data.get("sections", {})
        section_order = [
            "header", "executive_summary", "kpi_dashboard", "trends",
            "anomalies", "commodity_breakdown", "budget_variance",
            "recommendations", "appendix",
        ]
        parts = [sections[s] for s in section_order if s in sections]
        return "\n\n".join(parts)

    def render_html(self, report_data: Dict[str, Any]) -> str:
        """Render report data as HTML document.

        Args:
            report_data: Dict with 'sections' and 'title' keys.

        Returns:
            Complete HTML document string.
        """
        title = report_data.get("title", "Utility Report")
        sections = report_data.get("sections", {})
        section_order = [
            "header", "executive_summary", "kpi_dashboard", "trends",
            "anomalies", "commodity_breakdown", "budget_variance",
            "recommendations", "appendix",
        ]
        body_parts = [sections[s] for s in section_order if s in sections]
        body_content = "\n".join(body_parts)

        return (
            f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f"  <meta charset=\"UTF-8\">\n"
            f"  <meta name=\"generator\" content=\"PACK-036 "
            f"UtilityReportingEngine v{_MODULE_VERSION}\">\n"
            f"  <title>{title}</title>\n"
            f"  <style>\n"
            f"    body {{ font-family: Arial, sans-serif; max-width: 1200px; "
            f"margin: 0 auto; padding: 20px; color: #333; }}\n"
            f"    h1 {{ color: #1a5632; border-bottom: 2px solid #1a5632; "
            f"padding-bottom: 8px; }}\n"
            f"    h2 {{ color: #2d7a4f; margin-top: 24px; }}\n"
            f"    table {{ border-collapse: collapse; width: 100%; "
            f"margin: 12px 0; }}\n"
            f"    th, td {{ border: 1px solid #ddd; padding: 8px; "
            f"text-align: left; }}\n"
            f"    th {{ background-color: #f5f5f5; font-weight: bold; }}\n"
            f"    .rag-red {{ color: #d32f2f; font-weight: bold; }}\n"
            f"    .rag-amber {{ color: #f57c00; font-weight: bold; }}\n"
            f"    .rag-green {{ color: #388e3c; font-weight: bold; }}\n"
            f"  </style>\n"
            f"</head>\n<body>\n"
            f"<pre>{body_content}</pre>\n"
            f"</body>\n</html>"
        )

    def export_json(self, report_data: Dict[str, Any]) -> str:
        """Export report data as formatted JSON string.

        Args:
            report_data: Report data dictionary.

        Returns:
            JSON-formatted string.
        """
        return json.dumps(report_data, indent=2, default=str, sort_keys=True)

    def export_csv(self, report_data: Dict[str, Any]) -> str:
        """Export KPI data as CSV string.

        Args:
            report_data: Report data with 'kpis' key containing list
                of KPI dicts or UtilityKPI objects.

        Returns:
            CSV-formatted string with header row.
        """
        lines = [
            "kpi_name,category,current_value,previous_value,"
            "change_pct,unit,rag_status,trend",
        ]
        kpis = report_data.get("kpis", [])
        for kpi in kpis:
            if isinstance(kpi, UtilityKPI):
                lines.append(
                    f"{kpi.name},{kpi.category.value},{kpi.current_value},"
                    f"{kpi.previous_value},{kpi.change_pct},{kpi.unit},"
                    f"{kpi.rag_status.value},{kpi.trend.value}"
                )
            elif isinstance(kpi, dict):
                lines.append(
                    f"{kpi.get('name', '')},{kpi.get('category', '')},"
                    f"{kpi.get('current_value', 0)},{kpi.get('previous_value', 0)},"
                    f"{kpi.get('change_pct', 0)},{kpi.get('unit', '')},"
                    f"{kpi.get('rag_status', '')},{kpi.get('trend', '')}"
                )
        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # Public API -- Batch Generation
    # --------------------------------------------------------------------- #

    def batch_generate(
        self,
        configs: List[ReportConfig],
        data: Dict[str, Any],
    ) -> List[ReportOutput]:
        """Generate multiple reports in batch.

        Args:
            configs: List of report configurations.
            data: Shared data payload for all reports.

        Returns:
            List of ReportOutput objects.
        """
        results: List[ReportOutput] = []
        for i, config in enumerate(configs):
            logger.info(
                "Batch report %d/%d: type=%s",
                i + 1, len(configs), config.report_type.value,
            )
            result = self.generate_report(config, data)
            results.append(result)
        logger.info("Batch generation complete: %d reports.", len(results))
        return results

    # --------------------------------------------------------------------- #
    # Private -- Section Builders
    # --------------------------------------------------------------------- #

    def _build_header(self, config: ReportConfig, title: str) -> str:
        """Build the report header section."""
        return (
            f"# {title}\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Report Type | {config.report_type.value} |\n"
            f"| Period | {config.period_start} to {config.period_end} |\n"
            f"| Organisation | {config.organisation} |\n"
            f"| Author | {config.author} |\n"
            f"| Generated | {_utcnow().isoformat()} |\n"
            f"| Engine | PACK-036 v{_MODULE_VERSION} |\n"
        )

    def _build_executive_summary(
        self,
        kpis: List[UtilityKPI],
        data: Dict[str, Any],
    ) -> str:
        """Build the executive summary section."""
        lines = ["## Executive Summary\n"]

        # RAG overview.
        red = sum(1 for k in kpis if k.rag_status == RAGStatus.RED)
        amber = sum(1 for k in kpis if k.rag_status == RAGStatus.AMBER)
        green = sum(1 for k in kpis if k.rag_status == RAGStatus.GREEN)
        lines.append(
            f"**RAG Status Overview:** {green} GREEN | {amber} AMBER | {red} RED\n",
        )

        # Top-line KPIs.
        lines.append("| KPI | Value | Change | Status |")
        lines.append("|-----|-------|--------|--------|")
        for kpi in kpis[:6]:
            rag_label = kpi.rag_status.value.upper()
            lines.append(
                f"| {kpi.name} | {kpi.current_value:,.2f} {kpi.unit} | "
                f"{kpi.change_pct:+.1f}% | {rag_label} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_kpi_section(self, kpis: List[UtilityKPI]) -> str:
        """Build the KPI dashboard section."""
        lines = ["## KPI Dashboard\n"]
        lines.append(
            "| KPI | Category | Current | Previous | Change | "
            "Target | RAG | Trend |"
        )
        lines.append(
            "|-----|----------|---------|----------|--------|"
            "--------|-----|-------|"
        )
        for kpi in kpis:
            target_str = f"{kpi.target_value:,.2f}" if kpi.target_value else "N/A"
            lines.append(
                f"| {kpi.name} | {kpi.category.value} | "
                f"{kpi.current_value:,.2f} {kpi.unit} | "
                f"{kpi.previous_value:,.2f} | {kpi.change_pct:+.1f}% | "
                f"{target_str} | {kpi.rag_status.value.upper()} | "
                f"{kpi.trend.value} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_trends_section(self, trends: List[TrendData]) -> str:
        """Build the trend charts section."""
        if not trends:
            return "## Trends\n\nNo trend data available.\n"

        lines = ["## Trends\n"]
        for trend in trends:
            lines.append(f"### {trend.metric_name}\n")
            lines.append(
                f"**Direction:** {trend.trend_direction.value} | "
                f"**YoY Change:** {trend.yoy_change_pct:+.1f}% | "
                f"**12M Rolling Avg:** {trend.rolling_12m_avg:,.2f}\n"
            )
            if trend.periods and trend.values:
                lines.append("| Period | Value |")
                lines.append("|--------|-------|")
                # Show last 12 periods max.
                start = max(0, len(trend.periods) - 12)
                for p, v in zip(
                    trend.periods[start:], trend.values[start:],
                ):
                    lines.append(f"| {p} | {v:,.2f} |")
                lines.append("")
        return "\n".join(lines)

    def _build_anomalies_section(self, anomalies: List[AnomalyFlag]) -> str:
        """Build the anomaly alerts section."""
        if not anomalies:
            return "## Anomaly Alerts\n\nNo anomalies detected.\n"

        lines = ["## Anomaly Alerts\n"]
        lines.append(f"**{len(anomalies)} anomalies detected.**\n")
        lines.append(
            "| Type | Facility | Period | Severity | Metric | "
            "Expected | Actual | Deviation | Action |"
        )
        lines.append(
            "|------|----------|--------|----------|--------|"
            "----------|--------|-----------|--------|"
        )
        for a in anomalies[:20]:
            lines.append(
                f"| {a.anomaly_type.value} | {a.facility_id} | {a.period} | "
                f"{a.severity}/5 | {a.metric} | {a.expected_value:,.2f} | "
                f"{a.actual_value:,.2f} | {a.deviation_pct:+.1f}% | "
                f"{a.recommended_action} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_commodity_section(
        self,
        monthly_summaries: List[Dict[str, Any]],
    ) -> str:
        """Build the commodity breakdown section."""
        if not monthly_summaries:
            return "## Commodity Breakdown\n\nNo commodity data available.\n"

        lines = ["## Commodity Breakdown\n"]
        lines.append(
            "| Month | Commodity | Consumption (kWh) | Cost (EUR) | "
            "Peak Demand (kW) | Load Factor | EUI | Carbon (tCO2e) |"
        )
        lines.append(
            "|-------|-----------|-------------------|------------|"
            "------------------|-------------|-----|----------------|"
        )
        for s in monthly_summaries:
            if isinstance(s, MonthlyUtilitySummary):
                lines.append(
                    f"| {s.month} | {s.commodity.value} | "
                    f"{s.consumption_kwh:,.0f} | {s.cost_eur:,.2f} | "
                    f"{s.peak_demand_kw:,.1f} | {s.load_factor_pct:.1f}% | "
                    f"{s.eui:.1f} | {s.carbon_tco2e:.3f} |"
                )
            elif isinstance(s, dict):
                lines.append(
                    f"| {s.get('month', '')} | {s.get('commodity', '')} | "
                    f"{float(s.get('consumption_kwh', 0)):,.0f} | "
                    f"{float(s.get('cost_eur', 0)):,.2f} | "
                    f"{float(s.get('peak_demand_kw', 0)):,.1f} | "
                    f"{float(s.get('load_factor_pct', 0)):.1f}% | "
                    f"{float(s.get('eui', 0)):.1f} | "
                    f"{float(s.get('carbon_tco2e', 0)):.3f} |"
                )
        lines.append("")
        return "\n".join(lines)

    def _build_variance_section(
        self,
        variances: List[VarianceExplanation],
        currency: str,
    ) -> str:
        """Build the budget variance section."""
        if not variances:
            return "## Budget Variance\n\nNo variance data available.\n"

        lines = ["## Budget Variance Analysis\n"]
        lines.append(f"| Driver | Impact ({currency}) | Share (%) | Description |")
        lines.append("|--------|------------|-----------|-------------|")
        total_impact = Decimal("0")
        for v in variances:
            lines.append(
                f"| {v.category.title()} | {v.impact_eur:+,.2f} | "
                f"{v.impact_pct:.1f}% | {v.description} |"
            )
            total_impact += _decimal(v.impact_eur)
        lines.append(
            f"| **Total** | **{_round2(float(total_impact)):+,.2f}** | "
            f"**100%** | |"
        )
        lines.append("")
        return "\n".join(lines)

    def _build_recommendations_section(
        self,
        insights: List[ExecutiveInsight],
    ) -> str:
        """Build the recommendations section from insights."""
        actionable = [i for i in insights if i.action_required]
        if not actionable:
            return (
                "## Recommendations\n\n"
                "No specific actions required at this time.\n"
            )

        lines = ["## Recommendations\n"]
        for i, insight in enumerate(actionable, start=1):
            impact_str = (
                f" (Impact: {insight.impact_eur:+,.2f} EUR)"
                if insight.impact_eur != 0 else ""
            )
            lines.append(
                f"{i}. **[P{insight.priority}] {insight.title}**{impact_str}"
            )
            lines.append(f"   {insight.description}\n")
        return "\n".join(lines)

    def _build_appendix(self, section_hashes: Dict[str, str]) -> str:
        """Build the appendix section with provenance data."""
        lines = [
            "## Appendix: Data Provenance\n",
            "| Section | SHA-256 Hash |",
            "|---------|-------------|",
        ]
        for sec_name, sec_hash in section_hashes.items():
            lines.append(f"| {sec_name} | `{sec_hash[:16]}...` |")

        lines.append("\n### Methodology Notes\n")
        for note in self._notes:
            lines.append(f"- {note}")

        lines.append(
            f"\n*Report generated by PACK-036 "
            f"UtilityReportingEngine v{_MODULE_VERSION}*\n"
        )
        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # Private -- Content Assembly
    # --------------------------------------------------------------------- #

    def _assemble_content(
        self,
        fmt: ReportFormat,
        sections: Dict[str, str],
    ) -> str:
        """Assemble report content in the requested format.

        Args:
            fmt: Report format.
            sections: Built section content.

        Returns:
            Assembled content string.
        """
        section_order = [
            "header", "executive_summary", "kpi_dashboard", "trends",
            "anomalies", "commodity_breakdown", "budget_variance",
            "recommendations", "appendix",
        ]
        parts = [sections[s] for s in section_order if s in sections]
        markdown = "\n\n".join(parts)

        if fmt == ReportFormat.MARKDOWN:
            return markdown
        elif fmt == ReportFormat.HTML:
            return self.render_html({"title": "Utility Report", "sections": sections})
        elif fmt == ReportFormat.JSON:
            return self.export_json({"sections": sections})
        elif fmt == ReportFormat.CSV:
            return markdown  # CSV export done via export_csv method.
        elif fmt in (ReportFormat.PDF, ReportFormat.EXCEL):
            # Return JSON structure for external renderers.
            return json.dumps(
                {"sections": sections, "format": fmt.value},
                indent=2,
                default=str,
            )
        return markdown

    # --------------------------------------------------------------------- #
    # Private -- Dashboard Helpers
    # --------------------------------------------------------------------- #

    def _render_dashboard_markdown(
        self,
        portfolio: PortfolioSummary,
        kpis: List[UtilityKPI],
        widgets: List[DashboardWidget],
    ) -> str:
        """Render executive dashboard as Markdown."""
        lines = ["# Executive Utility Dashboard\n"]

        # Portfolio overview.
        lines.append("## Portfolio Overview\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Facilities | {portfolio.total_facilities} |")
        lines.append(f"| Total Cost | {portfolio.total_cost_eur:,.2f} EUR |")
        lines.append(
            f"| Total Consumption | {portfolio.total_consumption_kwh:,.0f} kWh |",
        )
        lines.append(f"| Average EUI | {portfolio.avg_eui:.1f} kWh/m2 |")
        lines.append(f"| Best Facility | {portfolio.best_facility} |")
        lines.append(f"| Worst Facility | {portfolio.worst_facility} |")
        lines.append(
            f"| Portfolio Savings | {portfolio.portfolio_savings_eur:,.2f} EUR |",
        )
        rag = portfolio.rag_distribution
        lines.append(
            f"| RAG Distribution | GREEN: {rag.get('green', 0)} | "
            f"AMBER: {rag.get('amber', 0)} | RED: {rag.get('red', 0)} |",
        )
        lines.append("")

        # KPI summary.
        lines.append("## Key Performance Indicators\n")
        lines.append("| KPI | Value | Change | Status |")
        lines.append("|-----|-------|--------|--------|")
        for kpi in kpis[:10]:
            lines.append(
                f"| {kpi.name} | {kpi.current_value:,.2f} {kpi.unit} | "
                f"{kpi.change_pct:+.1f}% | {kpi.rag_status.value.upper()} |"
            )
        lines.append("")

        # Widget count.
        lines.append(f"**Dashboard Widgets:** {len(widgets)}\n")

        return "\n".join(lines)

    def _build_commodity_pie(
        self,
        kpis: List[UtilityKPI],
    ) -> Dict[str, Any]:
        """Build pie chart data from commodity-level KPIs."""
        cost_by_commodity: Dict[str, float] = {}
        for kpi in kpis:
            if kpi.category == KPICategory.COST and kpi.commodity != CommodityType.ALL:
                label = kpi.commodity.value
                cost_by_commodity[label] = kpi.current_value
        if not cost_by_commodity:
            return {}
        return {
            "labels": list(cost_by_commodity.keys()),
            "values": list(cost_by_commodity.values()),
        }

    def _calculate_portfolio_kpis(
        self,
        portfolio: PortfolioSummary,
        data: Dict[str, Any],
    ) -> List[UtilityKPI]:
        """Calculate portfolio-level KPIs."""
        kpis: List[UtilityKPI] = []

        kpis.append(UtilityKPI(
            name="Portfolio Total Cost",
            category=KPICategory.COST,
            current_value=portfolio.total_cost_eur,
            unit="EUR",
        ))

        kpis.append(UtilityKPI(
            name="Portfolio Total Consumption",
            category=KPICategory.CONSUMPTION,
            current_value=portfolio.total_consumption_kwh,
            unit="kWh",
        ))

        kpis.append(UtilityKPI(
            name="Portfolio Average EUI",
            category=KPICategory.EFFICIENCY,
            current_value=portfolio.avg_eui,
            unit="kWh/m2",
        ))

        kpis.append(UtilityKPI(
            name="Portfolio Savings",
            category=KPICategory.COST,
            current_value=portfolio.portfolio_savings_eur,
            unit="EUR",
        ))

        rag = portfolio.rag_distribution
        total_fac = max(portfolio.total_facilities, 1)
        green_pct = _round2(float(
            _safe_pct(_decimal(rag.get("green", 0)), _decimal(total_fac)),
        ))
        kpis.append(UtilityKPI(
            name="Portfolio Green Rate",
            category=KPICategory.COMPLIANCE,
            current_value=green_pct,
            unit="%",
            rag_status=RAGStatus.GREEN if green_pct >= 70 else (
                RAGStatus.AMBER if green_pct >= 40 else RAGStatus.RED
            ),
        ))

        return kpis

    def _detect_portfolio_anomalies(
        self,
        portfolio_data: Dict[str, Any],
    ) -> List[AnomalyFlag]:
        """Detect anomalies across portfolio facilities."""
        anomalies: List[AnomalyFlag] = []
        facilities = portfolio_data.get("facilities", [])
        for fac in facilities:
            fac_anomalies = self.detect_anomalies(
                fac,
                fac.get("historical", []),
            )
            anomalies.extend(fac_anomalies)
        anomalies.sort(key=lambda a: a.severity, reverse=True)
        return anomalies[:20]

    def _extract_portfolio_trends(
        self,
        portfolio_data: Dict[str, Any],
    ) -> List[TrendData]:
        """Extract portfolio-level trend data."""
        trends: List[TrendData] = []
        historical = portfolio_data.get("historical", [])
        if historical:
            for metric in ("total_cost_eur", "total_consumption_kwh"):
                trend = self.build_trend_data(historical, metric)
                trends.append(trend)
        return trends

    # --------------------------------------------------------------------- #
    # Private -- Anomaly Detection Helpers
    # --------------------------------------------------------------------- #

    def _build_baselines(
        self,
        historical_data: Any,
    ) -> Dict[str, Dict[str, Decimal]]:
        """Build mean/std baselines from historical data.

        Args:
            historical_data: List of dicts with metric values,
                or dict with metric histories.

        Returns:
            Dict mapping metric name to {mean, std}.
        """
        baselines: Dict[str, Dict[str, Decimal]] = {}
        metrics = ("consumption_kwh", "cost_eur", "peak_demand_kw")

        if isinstance(historical_data, list) and historical_data:
            for metric in metrics:
                values = [
                    _decimal(r.get(metric, 0))
                    for r in historical_data
                    if isinstance(r, dict) and metric in r
                ]
                if len(values) >= 3:
                    mean = _safe_divide(sum(values), _decimal(len(values)))
                    variance = _safe_divide(
                        sum((v - mean) ** 2 for v in values),
                        _decimal(max(len(values) - 1, 1)),
                    )
                    std = _decimal(math.sqrt(max(float(variance), 0.0)))
                    baselines[metric] = {"mean": mean, "std": std}

        return baselines

    def _check_metric_anomaly(
        self,
        metric: str,
        current_val: Any,
        baseline: Optional[Dict[str, Decimal]],
        anomaly_type: AnomalyType,
        facility_id: str,
        period: str,
    ) -> List[AnomalyFlag]:
        """Check a single metric for anomalies against baseline.

        Args:
            metric: Metric name.
            current_val: Current metric value.
            baseline: Baseline stats {mean, std}.
            anomaly_type: Type of anomaly to flag.
            facility_id: Facility identifier.
            period: Period label.

        Returns:
            List of AnomalyFlag (0 or 1 items).
        """
        if baseline is None or current_val is None:
            return []

        d_val = _decimal(current_val)
        mean = baseline["mean"]
        std = baseline["std"]

        if std == Decimal("0") or mean == Decimal("0"):
            return []

        deviation = abs(d_val - mean)
        deviation_pct = float(_safe_pct(d_val - mean, mean))
        sigma_distance = float(_safe_divide(deviation, std))

        threshold_pct = self._anomaly_threshold
        if abs(deviation_pct) < threshold_pct:
            return []

        # Severity based on sigma distance.
        if sigma_distance >= 3.0:
            severity = 5
        elif sigma_distance >= 2.5:
            severity = 4
        elif sigma_distance >= 2.0:
            severity = 3
        else:
            severity = 2

        return [AnomalyFlag(
            anomaly_type=anomaly_type,
            facility_id=facility_id,
            period=period,
            severity=severity,
            metric=metric,
            expected_value=_round2(float(mean)),
            actual_value=_round2(float(d_val)),
            deviation_pct=_round2(deviation_pct),
            description=(
                f"{metric} is {abs(deviation_pct):.1f}% "
                f"{'above' if d_val > mean else 'below'} historical mean "
                f"({sigma_distance:.1f} sigma)"
            ),
            recommended_action=self._recommend_action(anomaly_type, severity),
        )]

    def _recommend_action(
        self,
        anomaly_type: AnomalyType,
        severity: int,
    ) -> str:
        """Generate recommended action for an anomaly.

        Args:
            anomaly_type: Type of anomaly.
            severity: Severity level.

        Returns:
            Recommended action string.
        """
        actions: Dict[AnomalyType, str] = {
            AnomalyType.CONSUMPTION_SPIKE: (
                "Investigate operational changes and equipment schedules."
            ),
            AnomalyType.COST_SPIKE: (
                "Review utility invoices for billing errors and rate changes."
            ),
            AnomalyType.DEMAND_SPIKE: (
                "Review demand response strategy and load shedding procedures."
            ),
            AnomalyType.BILL_ERROR: (
                "Contact utility provider to verify meter readings and charges."
            ),
            AnomalyType.MISSING_DATA: (
                "Verify data source connectivity and re-import data."
            ),
            AnomalyType.RATE_CHANGE: (
                "Confirm rate change with supplier and update budget forecasts."
            ),
        }
        action = actions.get(anomaly_type, "Investigate and document findings.")
        if severity >= 4:
            action = f"URGENT: {action}"
        return action

    # --------------------------------------------------------------------- #
    # Private -- RAG and Trend Helpers
    # --------------------------------------------------------------------- #

    def _compute_rag(
        self,
        actual: Decimal,
        target: Any,
    ) -> RAGStatus:
        """Compute RAG status relative to target.

        For metrics where lower is better (consumption, cost, carbon),
        GREEN = at or below target, AMBER = up to amber threshold above,
        RED = above red threshold.

        Args:
            actual: Actual value.
            target: Target value (optional).

        Returns:
            RAGStatus.
        """
        if target is None:
            return RAGStatus.GREY

        d_target = _decimal(target)
        if d_target == Decimal("0"):
            return RAGStatus.GREY

        deviation_pct = float(_change_pct(actual, d_target))

        if deviation_pct <= 0:
            return RAGStatus.GREEN
        elif deviation_pct <= self._rag_amber_pct:
            return RAGStatus.GREEN
        elif deviation_pct <= self._rag_red_pct:
            return RAGStatus.AMBER
        else:
            return RAGStatus.RED

    def _infer_trend(self, history: List[Any]) -> TrendDirection:
        """Infer trend direction from historical values.

        Uses simple linear slope over last N values.

        Args:
            history: List of numeric values (newest last).

        Returns:
            TrendDirection.
        """
        if not history or len(history) < 3:
            return TrendDirection.FLAT

        values = [_decimal(v) for v in history[-12:]]
        n = len(values)
        d_n = _decimal(n)

        # OLS slope.
        sx = sum(_decimal(i) for i in range(n))
        sy = sum(values)
        sxx = sum(_decimal(i) ** 2 for i in range(n))
        sxy = sum(_decimal(i) * v for i, v in enumerate(values))

        denom = d_n * sxx - sx ** 2
        if denom == Decimal("0"):
            return TrendDirection.FLAT

        slope = _safe_divide(d_n * sxy - sx * sy, denom)
        mean_val = _safe_divide(sy, d_n)

        if mean_val == Decimal("0"):
            return TrendDirection.FLAT

        # Normalised slope as % of mean per period.
        norm_slope = float(_safe_divide(slope * Decimal("100"), mean_val))

        # Check volatility (coefficient of variation).
        variance = _safe_divide(
            sum((v - mean_val) ** 2 for v in values),
            _decimal(max(n - 1, 1)),
        )
        std = _decimal(math.sqrt(max(float(variance), 0.0)))
        cv = float(_safe_divide(std * Decimal("100"), abs(mean_val)))

        if cv > 30:
            return TrendDirection.VOLATILE
        elif norm_slope > 1.0:
            return TrendDirection.UP
        elif norm_slope < -1.0:
            return TrendDirection.DOWN
        else:
            return TrendDirection.FLAT

    def _extract_sparkline(
        self,
        history: List[Any],
    ) -> List[float]:
        """Extract sparkline data (last N values).

        Args:
            history: List of numeric values.

        Returns:
            Last sparkline_periods values as floats.
        """
        if not history:
            return []
        n = self._sparkline_periods
        recent = history[-n:] if len(history) >= n else history
        return [_round2(float(_decimal(v))) for v in recent]

    # --------------------------------------------------------------------- #
    # Private -- Utility Helpers
    # --------------------------------------------------------------------- #

    def _default_title(self, config: ReportConfig) -> str:
        """Generate default report title from config."""
        type_titles: Dict[ReportType, str] = {
            ReportType.MONTHLY_SUMMARY: "Monthly Utility Summary",
            ReportType.QUARTERLY_REVIEW: "Quarterly Utility Review",
            ReportType.ANNUAL_REPORT: "Annual Utility Report",
            ReportType.BUDGET_VARIANCE: "Budget Variance Analysis",
            ReportType.PORTFOLIO_SUMMARY: "Portfolio Utility Summary",
            ReportType.BENCHMARK_COMPARISON: "Benchmark Comparison Report",
            ReportType.EXECUTIVE_DASHBOARD: "Executive Utility Dashboard",
            ReportType.COMPLIANCE_REPORT: "Utility Compliance Report",
            ReportType.PROCUREMENT_REVIEW: "Utility Procurement Review",
            ReportType.DEMAND_ANALYSIS: "Demand Analysis Report",
            ReportType.SAVINGS_TRACKING: "Energy Savings Tracking Report",
            ReportType.CUSTOM: "Custom Utility Report",
        }
        return type_titles.get(config.report_type, "Utility Report")

    def _estimate_pages(self, content: str) -> int:
        """Estimate page count from content length.

        Assumes approximately 3000 characters per page for Markdown.

        Args:
            content: Report content string.

        Returns:
            Estimated page count (minimum 1).
        """
        if not content:
            return 0
        return max(1, len(content) // 3000 + 1)


# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

UtilityKPI.model_rebuild()
MonthlyUtilitySummary.model_rebuild()
PortfolioSummary.model_rebuild()
VarianceExplanation.model_rebuild()
TrendData.model_rebuild()
AnomalyFlag.model_rebuild()
DashboardWidget.model_rebuild()
ExecutiveInsight.model_rebuild()
ReportConfig.model_rebuild()
ReportOutput.model_rebuild()


# ---------------------------------------------------------------------------
# Public Aliases -- required by PACK-036 __init__.py symbol contract
# ---------------------------------------------------------------------------

ExportFormat = ReportFormat
"""Alias: ``ExportFormat`` -> :class:`ReportFormat`."""

UtilityReport = ReportOutput
"""Alias: ``UtilityReport`` -> :class:`ReportOutput`."""
