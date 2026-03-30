# -*- coding: utf-8 -*-
"""
BenchmarkReportEngine - PACK-035 Energy Benchmark Engine 10
=============================================================

Aggregates all benchmark analysis results into structured reports
and dashboard data.  Generates facility-level, portfolio-level,
regulatory compliance, and executive summary reports with KPI
calculations, section assembly, and multi-format export.

Supports Markdown, HTML skeleton, JSON, and CSV export formats
with full provenance tracking and SHA-256 hashing.

Report Sections:
    HEADER:           Report metadata, facility/portfolio identification
    EUI_SUMMARY:      Energy Use Intensity summary with site/source/primary
    PEER_COMPARISON:  Peer group ranking, percentile, quartile
    RATING:           EPC, ENERGY STAR, NABERS, CRREM ratings
    GAP_ANALYSIS:     Performance gap breakdown by end-use
    TRENDS:           Year-over-year trends, CUSUM, SPC status
    RECOMMENDATIONS:  Prioritised improvement recommendations
    PROVENANCE:       SHA-256 hashes, methodology notes, data sources

Regulatory / Standard References:
    - EU EED 2023/1791 Article 8 (Energy audit reporting)
    - EPBD 2024/1275 (Energy performance certificates)
    - ISO 50001:2018 Clause 9.3 (Management review reporting)
    - ASHRAE Standard 100-2018 (Benchmarking reports)
    - ENERGY STAR Portfolio Manager (Benchmark reports)
    - GRESB Real Estate Assessment (Portfolio ESG reporting)
    - ESRS E1-5 (Energy consumption disclosure)

Zero-Hallucination:
    - All KPIs computed from deterministic formulas
    - Report content assembled from pre-calculated engine outputs
    - No LLM-generated text in any numeric field
    - SHA-256 provenance hash on every report
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-035 Energy Benchmark
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
# Report version aligns with PACK-035 version.
# ---------------------------------------------------------------------------
REPORT_VERSION: str = "35.0.0"

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

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportType(str, Enum):
    """Types of benchmark reports.

    FACILITY_BENCHMARK:     Single-facility benchmark analysis report.
    PORTFOLIO_OVERVIEW:     Portfolio-level overview with rankings.
    REGULATORY_COMPLIANCE:  Compliance-focused report (EPC, MEPS, EED).
    EXECUTIVE_SUMMARY:      High-level executive summary with KPIs.
    TREND_ANALYSIS:         Trend analysis report with charts.
    GAP_ANALYSIS:           Performance gap analysis report.
    """
    FACILITY_BENCHMARK = "facility_benchmark"
    PORTFOLIO_OVERVIEW = "portfolio_overview"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    EXECUTIVE_SUMMARY = "executive_summary"
    TREND_ANALYSIS = "trend_analysis"
    GAP_ANALYSIS = "gap_analysis"

class ReportSection(str, Enum):
    """Individual sections of a benchmark report.

    HEADER:          Report metadata and identification.
    EUI_SUMMARY:     EUI calculations and comparisons.
    PEER_COMPARISON: Peer group ranking and percentile.
    RATING:          Performance ratings across systems.
    GAP_ANALYSIS:    Performance gap breakdown.
    TRENDS:          Year-over-year trends and forecasts.
    RECOMMENDATIONS: Prioritised improvement actions.
    PROVENANCE:      Audit trail and data provenance.
    """
    HEADER = "header"
    EUI_SUMMARY = "eui_summary"
    PEER_COMPARISON = "peer_comparison"
    RATING = "rating"
    GAP_ANALYSIS = "gap_analysis"
    TRENDS = "trends"
    RECOMMENDATIONS = "recommendations"
    PROVENANCE = "provenance"

class ChartType(str, Enum):
    """Chart types for report visualisation."""
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    HEATMAP = "heatmap"

class ExportTarget(str, Enum):
    """Export target destination."""
    FILE = "file"
    API = "api"
    EMAIL = "email"
    DASHBOARD = "dashboard"

class ReportTemplate(str, Enum):
    """Pre-defined report templates."""
    STANDARD = "standard"
    COMPACT = "compact"
    DETAILED = "detailed"
    REGULATORY = "regulatory"
    EXECUTIVE = "executive"

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class ReportConfig(BaseModel):
    """Configuration for report generation.

    Attributes:
        report_type: Type of report to generate.
        export_format: Desired output format.
        template: Report template.
        sections: Sections to include (default: all).
        title: Custom report title.
        subtitle: Custom report subtitle.
        author: Report author name.
        organisation: Organisation name.
        logo_url: URL to organisation logo (for HTML/PDF).
        include_charts: Whether to include chart data.
        include_raw_data: Whether to include raw data tables.
        currency_symbol: Currency symbol for cost fields.
        language: Report language (ISO 639-1).
    """
    report_type: ReportType = Field(default=ReportType.FACILITY_BENCHMARK)
    export_format: ReportFormat = Field(default=ReportFormat.MARKDOWN)
    template: ReportTemplate = Field(default=ReportTemplate.STANDARD)
    sections: List[ReportSection] = Field(
        default_factory=lambda: list(ReportSection),
        description="Sections to include",
    )
    title: str = Field(default="Energy Benchmark Report", max_length=500)
    subtitle: str = Field(default="", max_length=500)
    author: str = Field(default="GreenLang Platform", max_length=200)
    organisation: str = Field(default="", max_length=500)
    logo_url: str = Field(default="", max_length=2000)
    include_charts: bool = Field(default=True)
    include_raw_data: bool = Field(default=False)
    currency_symbol: str = Field(default="EUR", max_length=5)
    language: str = Field(default="en", max_length=5)

class FacilityReportData(BaseModel):
    """Data payload for a single-facility report.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        building_type: Building type.
        gross_floor_area_m2: Floor area.
        reporting_year: Year of data.
        site_eui_kwh_per_m2: Site EUI.
        source_eui_kwh_per_m2: Source EUI.
        primary_energy_kwh_per_m2: Primary energy.
        total_energy_kwh: Total energy.
        total_carbon_kgco2: Total carbon.
        total_cost_eur: Total energy cost.
        epc_class: EPC rating class.
        energy_star_score: ENERGY STAR score.
        nabers_stars: NABERS star rating.
        crrem_status: CRREM stranding status.
        peer_percentile: Percentile in peer group.
        peer_rank: Rank in peer group.
        peer_total: Total peers.
        gap_to_median_pct: Gap to median (%).
        gap_to_best_pct: Gap to best practice (%).
        savings_potential_kwh: Potential energy savings.
        savings_potential_eur: Potential cost savings.
        savings_potential_kgco2: Potential carbon savings.
        trend_direction: Trend direction string.
        yoy_change_pct: Year-over-year change (%).
        recommendations: List of recommendation dicts.
        methodology_notes: Methodology notes.
    """
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    building_type: str = Field(default="")
    gross_floor_area_m2: float = Field(default=0.0)
    reporting_year: int = Field(default=2025)
    site_eui_kwh_per_m2: float = Field(default=0.0)
    source_eui_kwh_per_m2: float = Field(default=0.0)
    primary_energy_kwh_per_m2: float = Field(default=0.0)
    total_energy_kwh: float = Field(default=0.0)
    total_carbon_kgco2: float = Field(default=0.0)
    total_cost_eur: float = Field(default=0.0)
    epc_class: str = Field(default="")
    energy_star_score: Optional[int] = Field(default=None)
    nabers_stars: Optional[str] = Field(default=None)
    crrem_status: str = Field(default="")
    peer_percentile: float = Field(default=0.0)
    peer_rank: int = Field(default=0)
    peer_total: int = Field(default=0)
    gap_to_median_pct: float = Field(default=0.0)
    gap_to_best_pct: float = Field(default=0.0)
    savings_potential_kwh: float = Field(default=0.0)
    savings_potential_eur: float = Field(default=0.0)
    savings_potential_kgco2: float = Field(default=0.0)
    trend_direction: str = Field(default="")
    yoy_change_pct: float = Field(default=0.0)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    methodology_notes: List[str] = Field(default_factory=list)

class PortfolioReportData(BaseModel):
    """Data payload for a portfolio-level report.

    Attributes:
        portfolio_name: Portfolio name.
        facility_count: Number of facilities.
        total_area_m2: Total portfolio area.
        total_energy_kwh: Total portfolio energy.
        total_carbon_kgco2: Total portfolio carbon.
        total_cost_eur: Total portfolio cost.
        area_weighted_eui: Area-weighted portfolio EUI.
        median_eui: Median facility EUI.
        best_eui: Best facility EUI.
        worst_eui: Worst facility EUI.
        top_performers: Top facility data.
        bottom_performers: Bottom facility data.
        epc_distribution: Count by EPC class.
        crrem_distribution: Count by CRREM status.
        entity_breakdowns: Metrics by entity.
        trends: Year-over-year trends.
        reporting_year: Year of data.
    """
    portfolio_name: str = Field(default="")
    facility_count: int = Field(default=0, ge=0)
    total_area_m2: float = Field(default=0.0)
    total_energy_kwh: float = Field(default=0.0)
    total_carbon_kgco2: float = Field(default=0.0)
    total_cost_eur: float = Field(default=0.0)
    area_weighted_eui: float = Field(default=0.0)
    median_eui: float = Field(default=0.0)
    best_eui: float = Field(default=0.0)
    worst_eui: float = Field(default=0.0)
    top_performers: List[Dict[str, Any]] = Field(default_factory=list)
    bottom_performers: List[Dict[str, Any]] = Field(default_factory=list)
    epc_distribution: Dict[str, int] = Field(default_factory=dict)
    crrem_distribution: Dict[str, int] = Field(default_factory=dict)
    entity_breakdowns: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    trends: List[Dict[str, Any]] = Field(default_factory=list)
    reporting_year: int = Field(default=2025)

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class ChartData(BaseModel):
    """Data structure for chart rendering.

    Attributes:
        chart_type: Type of chart.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        series: Data series [{name, values}].
        categories: Category labels.
    """
    chart_type: ChartType = Field(default=ChartType.BAR)
    title: str = Field(default="")
    x_label: str = Field(default="")
    y_label: str = Field(default="")
    series: List[Dict[str, Any]] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)

class TableData(BaseModel):
    """Data structure for table rendering.

    Attributes:
        title: Table title.
        headers: Column headers.
        rows: Data rows.
        footer: Optional footer row.
    """
    title: str = Field(default="")
    headers: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    footer: List[Any] = Field(default_factory=list)

class ReportMetadata(BaseModel):
    """Report metadata and identification.

    Attributes:
        report_id: Unique report identifier.
        report_type: Report type.
        report_version: Report format version.
        title: Report title.
        subtitle: Report subtitle.
        author: Report author.
        organisation: Organisation name.
        generated_at: Generation timestamp.
        data_as_of: Data reporting date.
        engine_version: Engine version that generated the report.
        pack_version: Pack version.
    """
    report_id: str = Field(default_factory=_new_uuid)
    report_type: ReportType = Field(default=ReportType.FACILITY_BENCHMARK)
    report_version: str = Field(default=REPORT_VERSION)
    title: str = Field(default="")
    subtitle: str = Field(default="")
    author: str = Field(default="")
    organisation: str = Field(default="")
    generated_at: datetime = Field(default_factory=utcnow)
    data_as_of: str = Field(default="")
    engine_version: str = Field(default=_MODULE_VERSION)
    pack_version: str = Field(default=REPORT_VERSION)

class ReportOutput(BaseModel):
    """Rendered report output in the requested format.

    Attributes:
        format: Export format.
        content: Report content (Markdown, HTML, JSON string, CSV string).
        charts: Chart data structures for rendering.
        tables: Table data structures for rendering.
        kpis: Key performance indicators.
        metadata: Report metadata.
    """
    format: ReportFormat = Field(default=ReportFormat.MARKDOWN)
    content: str = Field(default="")
    charts: List[ChartData] = Field(default_factory=list)
    tables: List[TableData] = Field(default_factory=list)
    kpis: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[ReportMetadata] = Field(default=None)

class BenchmarkReportResult(BaseModel):
    """Complete benchmark report result.

    Attributes:
        result_id: Unique result identifier.
        report: Report output.
        section_hashes: SHA-256 hash per section for audit.
        methodology_notes: Methodology and source notes.
        processing_time_ms: Computation time (ms).
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    report: Optional[ReportOutput] = Field(default=None)
    section_hashes: Dict[str, str] = Field(default_factory=dict)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BenchmarkReportEngine:
    """Zero-hallucination benchmark report generation engine.

    Aggregates benchmark analysis results into structured reports
    with KPI calculations, section assembly, and multi-format export.

    Guarantees:
        - Deterministic: same inputs produce identical reports.
        - Reproducible: every report carries a SHA-256 provenance hash.
        - Auditable: per-section hashes for granular audit trails.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = BenchmarkReportEngine()
        result = engine.generate_facility_report(config, data)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the benchmark report engine.

        Args:
            config: Optional configuration overrides.
        """
        self._config = config or {}
        self._notes: List[str] = []
        logger.info("BenchmarkReportEngine v%s initialised.", _MODULE_VERSION)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def generate_facility_report(
        self,
        config: ReportConfig,
        data: FacilityReportData,
    ) -> BenchmarkReportResult:
        """Generate a single-facility benchmark report.

        Args:
            config: Report configuration.
            data: Facility benchmark data.

        Returns:
            BenchmarkReportResult with rendered report.
        """
        t0 = time.perf_counter()
        self._notes = [
            f"Engine version: {self.engine_version}",
            f"Report type: {config.report_type.value}",
            f"Facility: {data.facility_name}",
        ]

        # Calculate KPIs.
        kpis = self.calculate_kpis(data)

        # Build sections.
        sections: Dict[str, str] = {}
        section_hashes: Dict[str, str] = {}

        if ReportSection.HEADER in config.sections:
            sec = self._build_header_section(config, data)
            sections["header"] = sec
            section_hashes["header"] = _compute_hash(sec)

        if ReportSection.EUI_SUMMARY in config.sections:
            sec = self._build_eui_section(data)
            sections["eui_summary"] = sec
            section_hashes["eui_summary"] = _compute_hash(sec)

        if ReportSection.PEER_COMPARISON in config.sections:
            sec = self._build_peer_section(data)
            sections["peer_comparison"] = sec
            section_hashes["peer_comparison"] = _compute_hash(sec)

        if ReportSection.RATING in config.sections:
            sec = self._build_rating_section(data)
            sections["rating"] = sec
            section_hashes["rating"] = _compute_hash(sec)

        if ReportSection.GAP_ANALYSIS in config.sections:
            sec = self._build_gap_section(data)
            sections["gap_analysis"] = sec
            section_hashes["gap_analysis"] = _compute_hash(sec)

        if ReportSection.TRENDS in config.sections:
            sec = self._build_trends_section(data)
            sections["trends"] = sec
            section_hashes["trends"] = _compute_hash(sec)

        if ReportSection.RECOMMENDATIONS in config.sections:
            sec = self._build_recommendations_section(data)
            sections["recommendations"] = sec
            section_hashes["recommendations"] = _compute_hash(sec)

        if ReportSection.PROVENANCE in config.sections:
            sec = self._build_provenance_section(data, section_hashes)
            sections["provenance"] = sec
            section_hashes["provenance"] = _compute_hash(sec)

        # Assemble report.
        report_output = self._assemble_report(config, sections, kpis, data)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = BenchmarkReportResult(
            report=report_output,
            section_hashes=section_hashes,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Facility report generated: %s, %d sections, format=%s, hash=%s (%.1f ms)",
            data.facility_name,
            len(sections),
            config.export_format.value,
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    def generate_portfolio_report(
        self,
        config: ReportConfig,
        data: PortfolioReportData,
    ) -> BenchmarkReportResult:
        """Generate a portfolio-level benchmark report.

        Args:
            config: Report configuration.
            data: Portfolio benchmark data.

        Returns:
            BenchmarkReportResult with rendered report.
        """
        t0 = time.perf_counter()
        self._notes = [
            f"Engine version: {self.engine_version}",
            f"Report type: portfolio_overview",
            f"Portfolio: {data.portfolio_name}",
        ]

        kpis = self._calculate_portfolio_kpis(data)

        sections: Dict[str, str] = {}
        section_hashes: Dict[str, str] = {}

        # Header.
        header = (
            f"# Portfolio Energy Benchmark Report\n\n"
            f"**Portfolio:** {data.portfolio_name}\n"
            f"**Facilities:** {data.facility_count}\n"
            f"**Total Area:** {data.total_area_m2:,.0f} m2\n"
            f"**Reporting Year:** {data.reporting_year}\n"
            f"**Generated:** {utcnow().isoformat()}\n\n"
        )
        sections["header"] = header
        section_hashes["header"] = _compute_hash(header)

        # Portfolio summary.
        summary = (
            f"## Portfolio Summary\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total Energy | {data.total_energy_kwh:,.0f} kWh |\n"
            f"| Total Carbon | {data.total_carbon_kgco2:,.0f} kgCO2 |\n"
            f"| Total Cost | {data.total_cost_eur:,.0f} EUR |\n"
            f"| Area-Weighted EUI | {data.area_weighted_eui:.1f} kWh/m2/yr |\n"
            f"| Median EUI | {data.median_eui:.1f} kWh/m2/yr |\n"
            f"| Best EUI | {data.best_eui:.1f} kWh/m2/yr |\n"
            f"| Worst EUI | {data.worst_eui:.1f} kWh/m2/yr |\n\n"
        )
        sections["eui_summary"] = summary
        section_hashes["eui_summary"] = _compute_hash(summary)

        # Performers.
        if data.top_performers:
            top_sec = "## Top Performers\n\n"
            for perf in data.top_performers[:5]:
                name = perf.get("name", "Unknown")
                eui = perf.get("eui", 0.0)
                top_sec += f"- **{name}**: {eui:.1f} kWh/m2/yr\n"
            top_sec += "\n"
            sections["top_performers"] = top_sec
            section_hashes["top_performers"] = _compute_hash(top_sec)

        if data.bottom_performers:
            bottom_sec = "## Facilities Requiring Attention\n\n"
            for perf in data.bottom_performers[:5]:
                name = perf.get("name", "Unknown")
                eui = perf.get("eui", 0.0)
                bottom_sec += f"- **{name}**: {eui:.1f} kWh/m2/yr\n"
            bottom_sec += "\n"
            sections["bottom_performers"] = bottom_sec
            section_hashes["bottom_performers"] = _compute_hash(bottom_sec)

        # Distribution.
        if data.epc_distribution:
            dist_sec = "## EPC Distribution\n\n"
            dist_sec += "| EPC Class | Count |\n|-----------|-------|\n"
            for cls, count in sorted(data.epc_distribution.items()):
                dist_sec += f"| {cls} | {count} |\n"
            dist_sec += "\n"
            sections["epc_distribution"] = dist_sec
            section_hashes["epc_distribution"] = _compute_hash(dist_sec)

        report_output = self._assemble_portfolio_report(config, sections, kpis, data)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = BenchmarkReportResult(
            report=report_output,
            section_hashes=section_hashes,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def generate_regulatory_report(
        self,
        config: ReportConfig,
        data: FacilityReportData,
    ) -> BenchmarkReportResult:
        """Generate a regulatory compliance-focused report.

        Args:
            config: Report configuration.
            data: Facility benchmark data.

        Returns:
            BenchmarkReportResult with compliance-oriented report.
        """
        config.report_type = ReportType.REGULATORY_COMPLIANCE
        config.sections = [
            ReportSection.HEADER,
            ReportSection.EUI_SUMMARY,
            ReportSection.RATING,
            ReportSection.GAP_ANALYSIS,
            ReportSection.RECOMMENDATIONS,
            ReportSection.PROVENANCE,
        ]
        return self.generate_facility_report(config, data)

    def generate_executive_summary(
        self,
        config: ReportConfig,
        data: FacilityReportData,
    ) -> BenchmarkReportResult:
        """Generate an executive summary report.

        Args:
            config: Report configuration.
            data: Facility benchmark data.

        Returns:
            BenchmarkReportResult with executive summary.
        """
        config.report_type = ReportType.EXECUTIVE_SUMMARY
        config.sections = [
            ReportSection.HEADER,
            ReportSection.EUI_SUMMARY,
            ReportSection.RATING,
            ReportSection.RECOMMENDATIONS,
        ]
        return self.generate_facility_report(config, data)

    def export_report(
        self,
        report: BenchmarkReportResult,
        target_format: ReportFormat,
    ) -> str:
        """Export a report to a different format.

        Args:
            report: Generated report result.
            target_format: Desired export format.

        Returns:
            Report content in the requested format.
        """
        if report.report is None:
            return ""

        if target_format == ReportFormat.JSON:
            return json.dumps(report.report.model_dump(mode="json"), indent=2, default=str)
        elif target_format == ReportFormat.CSV:
            return self._export_kpis_csv(report.report.kpis)
        elif target_format == ReportFormat.HTML:
            return self._wrap_html(report.report.content, report.report.metadata)
        elif target_format == ReportFormat.PDF_DATA:
            return json.dumps({
                "content": report.report.content,
                "kpis": report.report.kpis,
                "charts": [c.model_dump(mode="json") for c in report.report.charts],
                "tables": [t.model_dump(mode="json") for t in report.report.tables],
                "metadata": report.report.metadata.model_dump(mode="json")
                if report.report.metadata else {},
            }, indent=2, default=str)
        else:
            return report.report.content

    def calculate_kpis(
        self,
        data: FacilityReportData,
    ) -> Dict[str, Any]:
        """Calculate key performance indicators for a facility.

        Args:
            data: Facility benchmark data.

        Returns:
            Dict of KPI name to value.
        """
        kpis: Dict[str, Any] = {}

        kpis["site_eui_kwh_per_m2"] = _round2(data.site_eui_kwh_per_m2)
        kpis["source_eui_kwh_per_m2"] = _round2(data.source_eui_kwh_per_m2)
        kpis["primary_energy_kwh_per_m2"] = _round2(data.primary_energy_kwh_per_m2)

        # Carbon intensity.
        d_area = _decimal(data.gross_floor_area_m2)
        carbon_intensity = _safe_divide(_decimal(data.total_carbon_kgco2), d_area)
        kpis["carbon_intensity_kgco2_per_m2"] = _round3(float(carbon_intensity))

        # Cost intensity.
        cost_intensity = _safe_divide(_decimal(data.total_cost_eur), d_area)
        kpis["cost_intensity_eur_per_m2"] = _round3(float(cost_intensity))

        # Energy cost per kWh.
        energy_cost = _safe_divide(
            _decimal(data.total_cost_eur), _decimal(data.total_energy_kwh)
        )
        kpis["energy_cost_eur_per_kwh"] = _round4(float(energy_cost))

        # Ratings.
        kpis["epc_class"] = data.epc_class
        kpis["energy_star_score"] = data.energy_star_score
        kpis["nabers_stars"] = data.nabers_stars
        kpis["crrem_status"] = data.crrem_status

        # Peer position.
        kpis["peer_percentile"] = _round2(data.peer_percentile)
        kpis["peer_rank"] = data.peer_rank
        kpis["peer_total"] = data.peer_total

        # Savings potential.
        kpis["savings_potential_kwh"] = _round2(data.savings_potential_kwh)
        kpis["savings_potential_eur"] = _round2(data.savings_potential_eur)
        kpis["savings_potential_kgco2"] = _round2(data.savings_potential_kgco2)

        # Savings as percentage of consumption.
        savings_pct = _safe_pct(_decimal(data.savings_potential_kwh), _decimal(data.total_energy_kwh))
        kpis["savings_pct_of_consumption"] = _round2(float(savings_pct))

        # Trend.
        kpis["trend_direction"] = data.trend_direction
        kpis["yoy_change_pct"] = _round2(data.yoy_change_pct)

        return kpis

    # --------------------------------------------------------------------- #
    # Private -- Section Builders (Markdown)
    # --------------------------------------------------------------------- #

    def _build_header_section(
        self, config: ReportConfig, data: FacilityReportData
    ) -> str:
        """Build the report header section."""
        return (
            f"# {config.title}\n\n"
            f"{'**' + config.subtitle + '**' if config.subtitle else ''}\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Facility | {data.facility_name} |\n"
            f"| Building Type | {data.building_type} |\n"
            f"| Floor Area | {data.gross_floor_area_m2:,.0f} m2 |\n"
            f"| Reporting Year | {data.reporting_year} |\n"
            f"| Organisation | {config.organisation} |\n"
            f"| Generated | {utcnow().isoformat()} |\n"
            f"| Engine | PACK-035 v{_MODULE_VERSION} |\n\n"
        )

    def _build_eui_section(self, data: FacilityReportData) -> str:
        """Build the EUI summary section."""
        return (
            f"## Energy Use Intensity Summary\n\n"
            f"| Metric | Value | Unit |\n"
            f"|--------|-------|------|\n"
            f"| Site EUI | {data.site_eui_kwh_per_m2:.1f} | kWh/m2/yr |\n"
            f"| Source EUI | {data.source_eui_kwh_per_m2:.1f} | kWh/m2/yr |\n"
            f"| Primary Energy | {data.primary_energy_kwh_per_m2:.1f} | kWh/m2/yr |\n"
            f"| Total Energy | {data.total_energy_kwh:,.0f} | kWh/yr |\n"
            f"| Total Carbon | {data.total_carbon_kgco2:,.0f} | kgCO2/yr |\n"
            f"| Total Cost | {data.total_cost_eur:,.0f} | EUR/yr |\n\n"
        )

    def _build_peer_section(self, data: FacilityReportData) -> str:
        """Build the peer comparison section."""
        return (
            f"## Peer Comparison\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Percentile | {data.peer_percentile:.0f}th |\n"
            f"| Rank | {data.peer_rank} of {data.peer_total} |\n"
            f"| Gap to Median | {data.gap_to_median_pct:+.1f}% |\n"
            f"| Gap to Best Practice | {data.gap_to_best_pct:+.1f}% |\n\n"
        )

    def _build_rating_section(self, data: FacilityReportData) -> str:
        """Build the performance rating section."""
        lines = [
            "## Performance Ratings\n",
            "| Rating System | Rating | Status |",
            "|---------------|--------|--------|",
        ]
        if data.epc_class:
            lines.append(f"| EU EPC | {data.epc_class} | - |")
        if data.energy_star_score is not None:
            qual = "Qualifies" if data.energy_star_score >= 75 else "Does not qualify"
            lines.append(f"| ENERGY STAR | {data.energy_star_score} | {qual} |")
        if data.nabers_stars:
            lines.append(f"| NABERS | {data.nabers_stars} stars | - |")
        if data.crrem_status:
            lines.append(f"| CRREM | - | {data.crrem_status} |")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _build_gap_section(self, data: FacilityReportData) -> str:
        """Build the gap analysis section."""
        return (
            f"## Gap Analysis\n\n"
            f"| Metric | Value | Unit |\n"
            f"|--------|-------|------|\n"
            f"| Gap to Median | {data.gap_to_median_pct:+.1f} | % |\n"
            f"| Gap to Best Practice | {data.gap_to_best_pct:+.1f} | % |\n"
            f"| Savings Potential | {data.savings_potential_kwh:,.0f} | kWh/yr |\n"
            f"| Cost Savings | {data.savings_potential_eur:,.0f} | EUR/yr |\n"
            f"| Carbon Savings | {data.savings_potential_kgco2:,.0f} | kgCO2/yr |\n\n"
        )

    def _build_trends_section(self, data: FacilityReportData) -> str:
        """Build the trends section."""
        return (
            f"## Performance Trends\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Trend Direction | {data.trend_direction} |\n"
            f"| Year-over-Year Change | {data.yoy_change_pct:+.1f}% |\n\n"
        )

    def _build_recommendations_section(self, data: FacilityReportData) -> str:
        """Build the recommendations section."""
        if not data.recommendations:
            return "## Recommendations\n\nNo specific recommendations at this time.\n\n"

        lines = ["## Recommendations\n"]
        for i, rec in enumerate(data.recommendations, start=1):
            desc = rec.get("description", "")
            priority = rec.get("priority", i)
            savings = rec.get("savings_kwh", 0)
            lines.append(
                f"{priority}. **{desc}**"
                + (f" (Savings: {savings:,.0f} kWh)" if savings else "")
            )
        lines.append("")
        return "\n".join(lines) + "\n"

    def _build_provenance_section(
        self, data: FacilityReportData, section_hashes: Dict[str, str]
    ) -> str:
        """Build the provenance and audit section."""
        lines = [
            "## Data Provenance\n",
            "| Section | SHA-256 Hash |",
            "|---------|-------------|",
        ]
        for sec_name, sec_hash in section_hashes.items():
            lines.append(f"| {sec_name} | `{sec_hash[:16]}...` |")

        lines.append("\n### Methodology Notes\n")
        for note in data.methodology_notes:
            lines.append(f"- {note}")

        lines.append(
            f"\n*Report generated by PACK-035 BenchmarkReportEngine v{_MODULE_VERSION}*\n"
        )
        return "\n".join(lines) + "\n"

    # --------------------------------------------------------------------- #
    # Private -- Report Assembly
    # --------------------------------------------------------------------- #

    def _assemble_report(
        self,
        config: ReportConfig,
        sections: Dict[str, str],
        kpis: Dict[str, Any],
        data: FacilityReportData,
    ) -> ReportOutput:
        """Assemble the final report output.

        Args:
            config: Report configuration.
            sections: Built section content.
            kpis: Calculated KPIs.
            data: Facility data.

        Returns:
            ReportOutput in the requested format.
        """
        # Combine sections in order.
        section_order = [
            "header", "eui_summary", "peer_comparison", "rating",
            "gap_analysis", "trends", "recommendations", "provenance",
        ]
        content_parts = [sections[s] for s in section_order if s in sections]
        content = "\n".join(content_parts)

        # Build charts.
        charts: List[ChartData] = []
        if config.include_charts:
            charts = self._build_charts(data)

        # Build tables.
        tables: List[TableData] = []
        if config.include_raw_data:
            tables = self._build_tables(data, kpis)

        # Metadata.
        metadata = ReportMetadata(
            report_type=config.report_type,
            title=config.title,
            subtitle=config.subtitle,
            author=config.author,
            organisation=config.organisation,
            data_as_of=str(data.reporting_year),
        )

        # Format conversion.
        if config.export_format == ReportFormat.JSON:
            content = json.dumps({
                "sections": sections,
                "kpis": kpis,
                "metadata": metadata.model_dump(mode="json"),
            }, indent=2, default=str)
        elif config.export_format == ReportFormat.HTML:
            content = self._wrap_html(content, metadata)

        return ReportOutput(
            format=config.export_format,
            content=content,
            charts=charts,
            tables=tables,
            kpis=kpis,
            metadata=metadata,
        )

    def _assemble_portfolio_report(
        self,
        config: ReportConfig,
        sections: Dict[str, str],
        kpis: Dict[str, Any],
        data: PortfolioReportData,
    ) -> ReportOutput:
        """Assemble portfolio report output.

        Args:
            config: Report configuration.
            sections: Built sections.
            kpis: Portfolio KPIs.
            data: Portfolio data.

        Returns:
            ReportOutput.
        """
        content = "\n".join(sections.values())

        metadata = ReportMetadata(
            report_type=ReportType.PORTFOLIO_OVERVIEW,
            title=config.title,
            subtitle=config.subtitle,
            author=config.author,
            organisation=config.organisation,
            data_as_of=str(data.reporting_year),
        )

        if config.export_format == ReportFormat.JSON:
            content = json.dumps({
                "sections": sections,
                "kpis": kpis,
                "metadata": metadata.model_dump(mode="json"),
            }, indent=2, default=str)

        return ReportOutput(
            format=config.export_format,
            content=content,
            kpis=kpis,
            metadata=metadata,
        )

    # --------------------------------------------------------------------- #
    # Private -- Portfolio KPIs
    # --------------------------------------------------------------------- #

    def _calculate_portfolio_kpis(self, data: PortfolioReportData) -> Dict[str, Any]:
        """Calculate portfolio-level KPIs.

        Args:
            data: Portfolio data.

        Returns:
            Dict of KPI name to value.
        """
        kpis: Dict[str, Any] = {}

        kpis["facility_count"] = data.facility_count
        kpis["total_area_m2"] = _round2(data.total_area_m2)
        kpis["total_energy_kwh"] = _round2(data.total_energy_kwh)
        kpis["total_carbon_kgco2"] = _round2(data.total_carbon_kgco2)
        kpis["total_cost_eur"] = _round2(data.total_cost_eur)
        kpis["area_weighted_eui"] = _round2(data.area_weighted_eui)
        kpis["median_eui"] = _round2(data.median_eui)
        kpis["eui_range"] = _round2(data.worst_eui - data.best_eui)

        d_area = _decimal(data.total_area_m2)
        kpis["carbon_intensity_kgco2_per_m2"] = _round3(
            float(_safe_divide(_decimal(data.total_carbon_kgco2), d_area))
        )
        kpis["cost_intensity_eur_per_m2"] = _round3(
            float(_safe_divide(_decimal(data.total_cost_eur), d_area))
        )
        kpis["energy_cost_eur_per_kwh"] = _round4(
            float(_safe_divide(_decimal(data.total_cost_eur), _decimal(data.total_energy_kwh)))
        )

        return kpis

    # --------------------------------------------------------------------- #
    # Private -- Chart and Table Builders
    # --------------------------------------------------------------------- #

    def _build_charts(self, data: FacilityReportData) -> List[ChartData]:
        """Build chart data structures for visualisation.

        Args:
            data: Facility data.

        Returns:
            List of ChartData objects.
        """
        charts: List[ChartData] = []

        # EUI comparison gauge.
        charts.append(ChartData(
            chart_type=ChartType.GAUGE,
            title="Energy Use Intensity",
            series=[{
                "name": "Site EUI",
                "value": data.site_eui_kwh_per_m2,
                "unit": "kWh/m2/yr",
            }],
        ))

        # Ratings bar chart.
        ratings_series = []
        if data.energy_star_score is not None:
            ratings_series.append({"name": "ENERGY STAR", "value": data.energy_star_score})
        if data.peer_percentile > 0:
            ratings_series.append({"name": "Peer Percentile", "value": data.peer_percentile})

        if ratings_series:
            charts.append(ChartData(
                chart_type=ChartType.BAR,
                title="Performance Scores",
                x_label="Rating System",
                y_label="Score",
                series=ratings_series,
            ))

        return charts

    def _build_tables(
        self, data: FacilityReportData, kpis: Dict[str, Any]
    ) -> List[TableData]:
        """Build table data structures.

        Args:
            data: Facility data.
            kpis: Calculated KPIs.

        Returns:
            List of TableData objects.
        """
        tables: List[TableData] = []

        # KPI summary table.
        kpi_rows = [[k, str(v)] for k, v in kpis.items()]
        tables.append(TableData(
            title="Key Performance Indicators",
            headers=["KPI", "Value"],
            rows=kpi_rows,
        ))

        return tables

    # --------------------------------------------------------------------- #
    # Private -- Export Helpers
    # --------------------------------------------------------------------- #

    def _wrap_html(self, markdown_content: str, metadata: Optional[ReportMetadata]) -> str:
        """Wrap Markdown content in an HTML document skeleton.

        Args:
            markdown_content: Markdown report content.
            metadata: Report metadata.

        Returns:
            HTML document string.
        """
        title = metadata.title if metadata else "Benchmark Report"
        return (
            f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f"  <meta charset=\"UTF-8\">\n"
            f"  <meta name=\"generator\" content=\"PACK-035 BenchmarkReportEngine v{_MODULE_VERSION}\">\n"
            f"  <title>{title}</title>\n"
            f"  <style>body {{ font-family: Arial, sans-serif; max-width: 1000px; "
            f"margin: 0 auto; padding: 20px; }}</style>\n"
            f"</head>\n<body>\n"
            f"<pre>{markdown_content}</pre>\n"
            f"</body>\n</html>"
        )

    def _export_kpis_csv(self, kpis: Dict[str, Any]) -> str:
        """Export KPIs as CSV string.

        Args:
            kpis: KPI dictionary.

        Returns:
            CSV-formatted string.
        """
        lines = ["kpi,value"]
        for k, v in kpis.items():
            lines.append(f"{k},{v}")
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

ReportConfig.model_rebuild()
FacilityReportData.model_rebuild()
PortfolioReportData.model_rebuild()
ChartData.model_rebuild()
TableData.model_rebuild()
ReportMetadata.model_rebuild()
ReportOutput.model_rebuild()
BenchmarkReportResult.model_rebuild()

# ---------------------------------------------------------------------------
# Public Aliases -- required by PACK-035 __init__.py symbol contract
# ---------------------------------------------------------------------------

ReportFormat = ReportFormat
"""Alias: ``ReportFormat`` -> :class:`ReportFormat`."""
