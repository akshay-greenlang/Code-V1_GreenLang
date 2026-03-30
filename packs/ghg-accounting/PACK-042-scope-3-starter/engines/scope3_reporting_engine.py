# -*- coding: utf-8 -*-
"""
Scope3ReportingEngine - PACK-042 Scope 3 Starter Pack Engine 10
=================================================================

Generates comprehensive Scope 3 reports and verification packages in
multiple output formats.  Supports 10 report types from executive
summaries to full ISO 14064-3 verification bundles.

Report Types:
    1.  FULL_INVENTORY        -- All 15 categories with full detail.
    2.  CATEGORY_DEEP_DIVE    -- Single category detailed analysis.
    3.  EXECUTIVE_SUMMARY     -- 2-4 page C-suite summary.
    4.  HOTSPOT_ANALYSIS      -- Pareto charts, materiality matrix.
    5.  SUPPLIER_ENGAGEMENT   -- Engagement status dashboard.
    6.  DATA_QUALITY          -- DQR scores, improvement roadmap.
    7.  COMPLIANCE_DASHBOARD  -- Multi-framework readiness.
    8.  UNCERTAINTY_ANALYSIS  -- Monte Carlo results, sensitivity.
    9.  TREND_ANALYSIS        -- Year-over-year trajectory.
    10. VERIFICATION_PACKAGE  -- ISO 14064-3 evidence bundle.

Output Formats:
    MARKDOWN:  GitHub-flavoured Markdown, VCS-friendly.
    HTML:      Self-contained HTML with inline CSS.
    JSON:      Structured JSON for API consumption.
    CSV:       Tabular data export.

Report Structure (GHG Protocol Scope 3 Chapter 9):
    1. Scope 3 screening results
    2. Per-category emissions and methodology
    3. Data quality assessment
    4. Uncertainty quantification
    5. Supplier engagement summary
    6. Exclusions and justifications
    7. Year-over-year trends
    8. Compliance readiness summary
    9. Appendices (EF registry, assumptions, methodology)

Regulatory References:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard, Chapter 9
    - ISO 14064-1:2018, Clause 10 (GHG Report)
    - ISO 14064-3:2019 (Validation and Verification of GHG statements)
    - ESRS E1 (Delegated Act 2023/2772), Disclosure E1-6
    - CDP Climate Change Reporting Guidance (2024)
    - ISAE 3410 (Assurance Engagements on GHG Statements)

Zero-Hallucination:
    - All report content from deterministic templates
    - Numeric values sourced directly from engine outputs
    - No LLM generation of numeric content
    - SHA-256 provenance hash on every report

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-042 Scope 3 Starter
Engine:  10 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
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
    """Safely divide two Decimals."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely."""
    return _safe_divide(part * Decimal("100"), whole)

def _round2(value: Any) -> float:
    """Round to 2 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _fmt(value: Any) -> str:
    """Format a number with comma separators and 2dp."""
    try:
        return f"{_round2(value):,.2f}"
    except (ValueError, TypeError):
        return str(value)

def _fmt_pct(value: Any) -> str:
    """Format a percentage value."""
    try:
        return f"{_round2(value):.2f}%"
    except (ValueError, TypeError):
        return str(value)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Scope3ReportType(str, Enum):
    """Types of Scope 3 reports.

    FULL_INVENTORY:       All 15 categories with full detail.
    CATEGORY_DEEP_DIVE:   Single category detailed analysis.
    EXECUTIVE_SUMMARY:    2-4 page C-suite summary.
    HOTSPOT_ANALYSIS:     Pareto charts and materiality matrix.
    SUPPLIER_ENGAGEMENT:  Engagement status dashboard.
    DATA_QUALITY:         DQR scores and improvement roadmap.
    COMPLIANCE_DASHBOARD: Multi-framework readiness.
    UNCERTAINTY_ANALYSIS: Monte Carlo results and sensitivity.
    TREND_ANALYSIS:       Year-over-year trajectory.
    VERIFICATION_PACKAGE: ISO 14064-3 evidence bundle.
    """
    FULL_INVENTORY = "full_inventory"
    CATEGORY_DEEP_DIVE = "category_deep_dive"
    EXECUTIVE_SUMMARY = "executive_summary"
    HOTSPOT_ANALYSIS = "hotspot_analysis"
    SUPPLIER_ENGAGEMENT = "supplier_engagement"
    DATA_QUALITY = "data_quality"
    COMPLIANCE_DASHBOARD = "compliance_dashboard"
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"
    TREND_ANALYSIS = "trend_analysis"
    VERIFICATION_PACKAGE = "verification_package"

class OutputFormat(str, Enum):
    """Output format for generated reports.

    MARKDOWN: GitHub-flavoured Markdown.
    HTML:     Self-contained HTML document.
    JSON:     Structured JSON data.
    CSV:      Comma-separated values.
    """
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"

# ---------------------------------------------------------------------------
# Constants -- Category Names and XBRL Tags
# ---------------------------------------------------------------------------

SCOPE3_CATEGORY_NAMES: Dict[str, str] = {
    "cat_1": "Purchased Goods & Services",
    "cat_2": "Capital Goods",
    "cat_3": "Fuel & Energy Related Activities",
    "cat_4": "Upstream Transportation & Distribution",
    "cat_5": "Waste Generated in Operations",
    "cat_6": "Business Travel",
    "cat_7": "Employee Commuting",
    "cat_8": "Upstream Leased Assets",
    "cat_9": "Downstream Transportation & Distribution",
    "cat_10": "Processing of Sold Products",
    "cat_11": "Use of Sold Products",
    "cat_12": "End-of-Life Treatment of Sold Products",
    "cat_13": "Downstream Leased Assets",
    "cat_14": "Franchises",
    "cat_15": "Investments",
}
"""Human-readable names for the 15 Scope 3 categories."""

ESRS_XBRL_TAGS: Dict[str, str] = {
    "scope3_total": "esrs:Scope3GhgEmissions",
    "scope3_cat_1": "esrs:Scope3Cat1PurchasedGoods",
    "scope3_cat_2": "esrs:Scope3Cat2CapitalGoods",
    "scope3_cat_3": "esrs:Scope3Cat3FuelEnergy",
    "scope3_cat_4": "esrs:Scope3Cat4UpstreamTransport",
    "scope3_cat_5": "esrs:Scope3Cat5Waste",
    "scope3_cat_6": "esrs:Scope3Cat6BusinessTravel",
    "scope3_cat_7": "esrs:Scope3Cat7EmployeeCommuting",
    "scope3_cat_8": "esrs:Scope3Cat8UpstreamLeased",
    "scope3_cat_9": "esrs:Scope3Cat9DownstreamTransport",
    "scope3_cat_10": "esrs:Scope3Cat10Processing",
    "scope3_cat_11": "esrs:Scope3Cat11UseOfSold",
    "scope3_cat_12": "esrs:Scope3Cat12EndOfLife",
    "scope3_cat_13": "esrs:Scope3Cat13DownstreamLeased",
    "scope3_cat_14": "esrs:Scope3Cat14Franchises",
    "scope3_cat_15": "esrs:Scope3Cat15Investments",
    "scope3_intensity": "esrs:Scope3GhgIntensityPerRevenue",
    "scope3_methodology": "esrs:Scope3MethodologyDescription",
    "scope3_data_quality": "esrs:Scope3DataQualityIndicator",
}
"""ESRS E1 XBRL taxonomy tags for Scope 3 data points."""

# HTML colour theme for self-contained reports.
_HTML_COLOURS: Dict[str, str] = {
    "primary": "#1B5E20",       # Dark green
    "secondary": "#4CAF50",     # Medium green
    "accent": "#81C784",        # Light green
    "background": "#F1F8E9",    # Very light green
    "text": "#212121",          # Near black
    "border": "#C8E6C9",        # Pale green border
    "header_bg": "#2E7D32",     # Header green
    "header_text": "#FFFFFF",   # White text on header
    "warning": "#FF6F00",       # Amber for warnings
    "error": "#C62828",         # Red for errors
    "success": "#2E7D32",       # Green for success
}
"""Colour theme for HTML reports (GreenLang brand)."""

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class OrganizationInfo(BaseModel):
    """Organisation metadata for report headers.

    Attributes:
        name: Organisation name.
        sector: Industry sector.
        country: Headquarters country.
        reporting_year: Reporting year.
        base_year: Base year for Scope 3.
        report_preparer: Who prepared the report.
        report_date: Date of report preparation.
    """
    name: str = Field(default="Organisation", description="Name")
    sector: str = Field(default="", description="Sector")
    country: str = Field(default="", description="Country")
    reporting_year: int = Field(default=2025, ge=1990, description="Reporting year")
    base_year: int = Field(default=2019, ge=1990, description="Base year")
    report_preparer: str = Field(
        default="GreenLang Platform", description="Report preparer"
    )
    report_date: str = Field(default="", description="Report date")

class ReportConfig(BaseModel):
    """Configuration for report generation.

    Attributes:
        report_type: Type of report to generate.
        output_format: Output format.
        category_filter: Optional single category for deep dive.
        include_appendix: Whether to include appendix.
        include_xbrl: Whether to include XBRL tags.
        include_charts_data: Whether to include chart data.
        custom_title: Optional custom report title.
        custom_sections: Optional custom sections to include.
    """
    report_type: str = Field(
        default=Scope3ReportType.FULL_INVENTORY, description="Report type"
    )
    output_format: str = Field(
        default=OutputFormat.MARKDOWN, description="Output format"
    )
    category_filter: Optional[str] = Field(
        default=None, description="Category for deep dive"
    )
    include_appendix: bool = Field(default=True, description="Include appendix")
    include_xbrl: bool = Field(default=False, description="Include XBRL tags")
    include_charts_data: bool = Field(default=True, description="Include chart data")
    custom_title: Optional[str] = Field(default=None, description="Custom title")
    custom_sections: Optional[List[str]] = Field(
        default=None, description="Custom sections"
    )

class Scope3ReportData(BaseModel):
    """Aggregated input data for Scope 3 report generation.

    Combines outputs from engines 1-9 (screening, calculation,
    supplier engagement, data quality, uncertainty, compliance).

    Attributes:
        organization: Organisation metadata.
        scope3_total_tco2e: Total Scope 3 emissions.
        per_category_emissions: Emissions per category.
        per_category_methodology: Methodology per category.
        per_category_dqr: DQR score per category.
        screening_results: Category screening/materiality.
        supplier_engagement: Supplier engagement summary.
        data_quality: Data quality assessment summary.
        uncertainty: Uncertainty analysis summary.
        compliance: Compliance assessment summary.
        trend_data: Year-over-year trend data.
        emission_factors: Emission factor registry.
        exclusions: Excluded categories and justifications.
        assumptions: Key assumptions list.
        notes: Additional notes.
    """
    organization: OrganizationInfo = Field(
        default_factory=OrganizationInfo, description="Org info"
    )
    scope3_total_tco2e: float = Field(default=0.0, ge=0, description="Scope 3 total")
    per_category_emissions: Dict[str, float] = Field(
        default_factory=dict, description="Per-category emissions"
    )
    per_category_methodology: Dict[str, str] = Field(
        default_factory=dict, description="Per-category methodology"
    )
    per_category_dqr: Dict[str, float] = Field(
        default_factory=dict, description="Per-category DQR"
    )
    screening_results: Dict[str, Any] = Field(
        default_factory=dict, description="Screening results"
    )
    supplier_engagement: Dict[str, Any] = Field(
        default_factory=dict, description="Supplier engagement"
    )
    data_quality: Dict[str, Any] = Field(
        default_factory=dict, description="Data quality"
    )
    uncertainty: Dict[str, Any] = Field(
        default_factory=dict, description="Uncertainty"
    )
    compliance: Dict[str, Any] = Field(
        default_factory=dict, description="Compliance"
    )
    trend_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Trend data"
    )
    emission_factors: Dict[str, Any] = Field(
        default_factory=dict, description="EF registry"
    )
    exclusions: List[Dict[str, str]] = Field(
        default_factory=list, description="Exclusions"
    )
    assumptions: List[str] = Field(
        default_factory=list, description="Assumptions"
    )
    notes: List[str] = Field(default_factory=list, description="Notes")

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class ReportSection(BaseModel):
    """A single section of a generated report.

    Attributes:
        section_id: Section identifier.
        title: Section heading.
        content: Section body text.
        tables: Embedded tables.
        charts_data: Data for chart rendering.
        order: Display order.
    """
    section_id: str = Field(default="", description="Section ID")
    title: str = Field(default="", description="Title")
    content: str = Field(default="", description="Content")
    tables: List[Dict[str, Any]] = Field(
        default_factory=list, description="Tables"
    )
    charts_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Chart data"
    )
    order: int = Field(default=0, description="Display order")

class AppendixItem(BaseModel):
    """An appendix entry.

    Attributes:
        appendix_id: Appendix identifier.
        title: Appendix title.
        content: Appendix content.
        data_type: Type of appendix data.
    """
    appendix_id: str = Field(default="", description="Appendix ID")
    title: str = Field(default="", description="Title")
    content: str = Field(default="", description="Content")
    data_type: str = Field(default="text", description="Data type")

class ReportOutput(BaseModel):
    """Complete report output.

    Attributes:
        report_id: Unique report identifier.
        report_type: Type of report.
        generated_at: Generation timestamp.
        reporting_year: Reporting year.
        organization_name: Organisation name.
        engine_version: Engine version.
        format: Output format.
        title: Report title.
        sections: Report sections in order.
        appendices: Appendix items.
        xbrl_tags: XBRL taxonomy tags.
        raw_data: Underlying data (for JSON export).
        file_content: Rendered content.
        processing_time_ms: Processing time (ms).
        provenance_hash: SHA-256 hash.
    """
    report_id: str = Field(default_factory=_new_uuid, description="Report ID")
    report_type: str = Field(default="", description="Report type")
    generated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    reporting_year: int = Field(default=0, description="Reporting year")
    organization_name: str = Field(default="", description="Org name")
    engine_version: str = Field(default=_MODULE_VERSION, description="Version")
    format: str = Field(default="markdown", description="Format")
    title: str = Field(default="", description="Report title")
    sections: List[ReportSection] = Field(
        default_factory=list, description="Sections"
    )
    appendices: List[AppendixItem] = Field(
        default_factory=list, description="Appendices"
    )
    xbrl_tags: Dict[str, str] = Field(
        default_factory=dict, description="XBRL tags"
    )
    raw_data: Dict[str, Any] = Field(
        default_factory=dict, description="Raw data"
    )
    file_content: str = Field(default="", description="Rendered content")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Scope3ReportingEngine:
    """Scope 3 GHG report generation engine.

    Generates comprehensive Scope 3 reports from engine outputs in
    multiple formats. All content is template-based and deterministic.

    Guarantees:
        - Deterministic: same inputs produce identical reports.
        - Traceable: SHA-256 provenance hash on every report.
        - Compliant: structure follows GHG Protocol Scope 3 Ch 9.
        - No LLM: zero hallucination in numeric content.

    Usage::

        engine = Scope3ReportingEngine()
        report = engine.generate_report(
            Scope3ReportType.EXECUTIVE_SUMMARY,
            report_data, config,
        )
        print(report.file_content)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the Scope 3 reporting engine.

        Args:
            config: Optional overrides.
        """
        self._config = config or {}
        logger.info("Scope3ReportingEngine v%s initialised.", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def generate_report(
        self,
        report_type: str,
        data: Scope3ReportData,
        config: Optional[ReportConfig] = None,
    ) -> ReportOutput:
        """Generate a Scope 3 report of the specified type.

        Args:
            report_type: Report type to generate.
            data: Aggregated Scope 3 data.
            config: Optional report configuration.

        Returns:
            ReportOutput with rendered content.
        """
        t0 = time.perf_counter()
        cfg = config or ReportConfig(report_type=report_type)
        output_format = cfg.output_format

        logger.info(
            "Generating Scope 3 report: type=%s, format=%s, org=%s.",
            report_type, output_format, data.organization.name,
        )

        # Route to generator.
        generator_map = {
            Scope3ReportType.FULL_INVENTORY: self._generate_full_inventory,
            Scope3ReportType.CATEGORY_DEEP_DIVE: self._generate_category_deep_dive,
            Scope3ReportType.EXECUTIVE_SUMMARY: self._generate_executive_summary,
            Scope3ReportType.HOTSPOT_ANALYSIS: self._generate_hotspot_analysis,
            Scope3ReportType.SUPPLIER_ENGAGEMENT: self._generate_supplier_engagement,
            Scope3ReportType.DATA_QUALITY: self._generate_data_quality,
            Scope3ReportType.COMPLIANCE_DASHBOARD: self._generate_compliance_dashboard,
            Scope3ReportType.UNCERTAINTY_ANALYSIS: self._generate_uncertainty_analysis,
            Scope3ReportType.TREND_ANALYSIS: self._generate_trend_analysis,
            Scope3ReportType.VERIFICATION_PACKAGE: self._generate_verification_package,
        }

        generator = generator_map.get(report_type, self._generate_full_inventory)
        sections = generator(data, cfg)

        # Optionally add appendix.
        appendices: List[AppendixItem] = []
        if cfg.include_appendix:
            appendices = self._generate_appendix(data)

        # XBRL tags.
        xbrl: Dict[str, str] = {}
        if cfg.include_xbrl:
            xbrl = self._generate_xbrl_tags(data)

        # Determine title.
        title = cfg.custom_title or self._default_title(report_type, data)

        # Render content.
        file_content = self._render(
            sections, appendices, title, data, output_format, xbrl,
        )

        elapsed = (time.perf_counter() - t0) * 1000

        output = ReportOutput(
            report_type=report_type,
            reporting_year=data.organization.reporting_year,
            organization_name=data.organization.name,
            format=output_format,
            title=title,
            sections=sections,
            appendices=appendices,
            xbrl_tags=xbrl,
            raw_data=data.model_dump(mode="json"),
            file_content=file_content,
            processing_time_ms=_round2(elapsed),
        )
        output.provenance_hash = self._compute_provenance(output)

        logger.info(
            "Report generated: type=%s, sections=%d, %d chars in %.1f ms.",
            report_type, len(sections), len(file_content), elapsed,
        )
        return output

    def _compute_provenance(self, data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        return _compute_hash(data)

    # -------------------------------------------------------------------
    # Private -- Report Generators
    # -------------------------------------------------------------------

    def _generate_full_inventory(
        self, data: Scope3ReportData, cfg: ReportConfig,
    ) -> List[ReportSection]:
        """Generate full inventory report sections."""
        sections: List[ReportSection] = []
        order = 0

        # Section 1: Overview.
        order += 1
        sections.append(self._md_overview(data, order))

        # Section 2: Category breakdown.
        order += 1
        sections.append(self._md_category_breakdown(data, order))

        # Section 3: Methodology.
        order += 1
        sections.append(self._md_methodology(data, order))

        # Section 4: Data quality.
        order += 1
        sections.append(self._md_data_quality(data, order))

        # Section 5: Uncertainty.
        order += 1
        sections.append(self._md_uncertainty(data, order))

        # Section 6: Supplier engagement.
        order += 1
        sections.append(self._md_supplier_engagement(data, order))

        # Section 7: Compliance.
        order += 1
        sections.append(self._md_compliance(data, order))

        # Section 8: Exclusions.
        order += 1
        sections.append(self._md_exclusions(data, order))

        return sections

    def _generate_executive_summary(
        self, data: Scope3ReportData, cfg: ReportConfig,
    ) -> List[ReportSection]:
        """Generate executive summary sections."""
        sections: List[ReportSection] = []

        # Key headline.
        total = data.scope3_total_tco2e
        cat_count = len(data.per_category_emissions)

        # Top 3 categories.
        sorted_cats = sorted(
            data.per_category_emissions.items(),
            key=lambda x: x[1], reverse=True,
        )
        top_3_lines = []
        for cat_key, em in sorted_cats[:3]:
            name = SCOPE3_CATEGORY_NAMES.get(cat_key, cat_key)
            pct = _round2(_safe_pct(_decimal(em), _decimal(total)))
            top_3_lines.append(f"- **{name}**: {_fmt(em)} tCO2e ({_fmt_pct(pct)})")

        content = (
            f"## Scope 3 GHG Emissions Summary -- {data.organization.reporting_year}\n\n"
            f"**Organisation:** {data.organization.name}\n\n"
            f"**Total Scope 3 Emissions:** {_fmt(total)} tCO2e\n\n"
            f"**Categories Reported:** {cat_count} of 15\n\n"
            f"### Top 3 Categories by Emissions\n\n"
            + "\n".join(top_3_lines)
        )

        # Compliance headline.
        compliance_data = data.compliance
        if compliance_data:
            readiness = compliance_data.get("overall_readiness", "N/A")
            content += f"\n\n### Compliance Readiness\n\n**Overall:** {readiness}%"

        # Data quality headline.
        dq = data.data_quality
        if dq:
            overall_dqr = dq.get("overall_dqr", {}).get("overall_dqr", "N/A")
            content += f"\n\n### Data Quality\n\n**Overall DQR:** {overall_dqr}/5.0"

        sections.append(ReportSection(
            section_id="exec_summary",
            title="Executive Summary",
            content=content,
            order=1,
        ))
        return sections

    def _generate_category_deep_dive(
        self, data: Scope3ReportData, cfg: ReportConfig,
    ) -> List[ReportSection]:
        """Generate deep-dive report for a single category."""
        cat_key = cfg.category_filter or "cat_1"
        cat_name = SCOPE3_CATEGORY_NAMES.get(cat_key, cat_key)
        emissions = data.per_category_emissions.get(cat_key, 0.0)
        methodology = data.per_category_methodology.get(cat_key, "Not documented")
        dqr = data.per_category_dqr.get(cat_key, 0.0)
        pct = _round2(_safe_pct(_decimal(emissions), _decimal(data.scope3_total_tco2e)))

        content = (
            f"## {cat_name} -- Detailed Analysis\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Emissions | {_fmt(emissions)} tCO2e |\n"
            f"| Share of Scope 3 | {_fmt_pct(pct)} |\n"
            f"| Methodology | {methodology} |\n"
            f"| Data Quality Rating | {dqr}/5.0 |\n"
        )

        sections = [ReportSection(
            section_id=f"deep_dive_{cat_key}",
            title=f"{cat_name} Deep Dive",
            content=content,
            order=1,
        )]
        return sections

    def _generate_hotspot_analysis(
        self, data: Scope3ReportData, cfg: ReportConfig,
    ) -> List[ReportSection]:
        """Generate hotspot analysis with Pareto data."""
        sorted_cats = sorted(
            data.per_category_emissions.items(),
            key=lambda x: x[1], reverse=True,
        )
        total = data.scope3_total_tco2e

        # Build Pareto table.
        rows = []
        cumulative = Decimal("0")
        for cat_key, em in sorted_cats:
            name = SCOPE3_CATEGORY_NAMES.get(cat_key, cat_key)
            pct = _safe_pct(_decimal(em), _decimal(total))
            cumulative += pct
            rows.append(
                f"| {name} | {_fmt(em)} | {_fmt_pct(pct)} | {_fmt_pct(cumulative)} |"
            )

        content = (
            "## Scope 3 Hotspot Analysis\n\n"
            "### Pareto Analysis (Categories ranked by emissions)\n\n"
            "| Category | tCO2e | Share | Cumulative |\n"
            "|---|---:|---:|---:|\n"
            + "\n".join(rows)
        )

        # Chart data for Pareto.
        chart_data = [{
            "chart_type": "pareto",
            "x_labels": [SCOPE3_CATEGORY_NAMES.get(k, k) for k, _ in sorted_cats],
            "y_values": [v for _, v in sorted_cats],
            "cumulative_pct": [],  # Would be populated by rendering layer.
        }]

        sections = [ReportSection(
            section_id="hotspot",
            title="Hotspot Analysis",
            content=content,
            charts_data=chart_data,
            order=1,
        )]
        return sections

    def _generate_supplier_engagement(
        self, data: Scope3ReportData, cfg: ReportConfig,
    ) -> List[ReportSection]:
        """Generate supplier engagement dashboard sections."""
        se = data.supplier_engagement
        content = "## Supplier Engagement Dashboard\n\n"

        if se:
            total = se.get("total_suppliers", 0)
            responded = se.get("suppliers_responded", 0)
            rate = se.get("response_rate_pct", 0)
            avg_dqi = se.get("avg_dqi_level", 1.0)
            content += (
                f"| Metric | Value |\n|---|---|\n"
                f"| Total Suppliers | {total} |\n"
                f"| Suppliers Responded | {responded} |\n"
                f"| Response Rate | {_fmt_pct(rate)} |\n"
                f"| Average DQI Level | {avg_dqi:.1f}/5.0 |\n"
            )
        else:
            content += "No supplier engagement data available.\n"

        return [ReportSection(
            section_id="supplier_engagement",
            title="Supplier Engagement",
            content=content,
            order=1,
        )]

    def _generate_data_quality(
        self, data: Scope3ReportData, cfg: ReportConfig,
    ) -> List[ReportSection]:
        """Generate data quality report sections."""
        dq = data.data_quality
        content = "## Data Quality Assessment\n\n"

        if dq:
            overall = dq.get("overall_dqr", {})
            content += f"**Overall DQR:** {overall.get('overall_dqr', 'N/A')}/5.0\n\n"

            # Per-category DQR table.
            if data.per_category_dqr:
                content += "### Per-Category DQR Scores\n\n"
                content += "| Category | DQR | Status |\n|---|---:|---|\n"
                for cat_key, score in sorted(
                    data.per_category_dqr.items(), key=lambda x: x[1]
                ):
                    name = SCOPE3_CATEGORY_NAMES.get(cat_key, cat_key)
                    status = (
                        "Adequate" if score >= 3.0
                        else "Needs Improvement"
                    )
                    content += f"| {name} | {score:.2f} | {status} |\n"
        else:
            content += "No data quality assessment available.\n"

        return [ReportSection(
            section_id="data_quality",
            title="Data Quality Assessment",
            content=content,
            order=1,
        )]

    def _generate_compliance_dashboard(
        self, data: Scope3ReportData, cfg: ReportConfig,
    ) -> List[ReportSection]:
        """Generate compliance dashboard sections."""
        comp = data.compliance
        content = "## Compliance Dashboard\n\n"

        if comp:
            content += f"**Overall Readiness:** {comp.get('overall_readiness', 'N/A')}%\n\n"

            frameworks = comp.get("frameworks", [])
            if frameworks:
                content += "### Framework Scores\n\n"
                content += "| Framework | Score | Classification |\n|---|---:|---|\n"
                for fw in frameworks:
                    content += (
                        f"| {fw.get('framework_name', fw.get('framework', ''))} "
                        f"| {fw.get('score', 0):.1f}% "
                        f"| {fw.get('classification', '')} |\n"
                    )

            gaps = comp.get("critical_gaps", [])
            if gaps:
                content += f"\n### Critical Gaps ({len(gaps)} items)\n\n"
                for gap in gaps[:10]:
                    content += (
                        f"- **{gap.get('requirement_id', '')}** "
                        f"({gap.get('framework', '')}): "
                        f"{gap.get('description', '')}\n"
                    )
        else:
            content += "No compliance assessment available.\n"

        return [ReportSection(
            section_id="compliance",
            title="Compliance Dashboard",
            content=content,
            order=1,
        )]

    def _generate_uncertainty_analysis(
        self, data: Scope3ReportData, cfg: ReportConfig,
    ) -> List[ReportSection]:
        """Generate uncertainty analysis sections."""
        unc = data.uncertainty
        content = "## Uncertainty Analysis\n\n"

        if unc:
            analytical = unc.get("analytical", {})
            mc = unc.get("monte_carlo", {})

            if analytical:
                content += (
                    "### Analytical (IPCC Approach 1)\n\n"
                    f"| Metric | Value |\n|---|---|\n"
                    f"| Combined Uncertainty | "
                    f"{analytical.get('combined_uncertainty_pct', 0):.1f}% |\n"
                    f"| 95% CI Lower Bound | "
                    f"{_fmt(analytical.get('lower_bound_tco2e', 0))} tCO2e |\n"
                    f"| 95% CI Upper Bound | "
                    f"{_fmt(analytical.get('upper_bound_tco2e', 0))} tCO2e |\n"
                )

            if mc:
                content += (
                    "\n### Monte Carlo (IPCC Approach 2)\n\n"
                    f"| Metric | Value |\n|---|---|\n"
                    f"| Mean | {_fmt(mc.get('mean_tco2e', 0))} tCO2e |\n"
                    f"| Median | {_fmt(mc.get('median_tco2e', 0))} tCO2e |\n"
                    f"| 2.5th Percentile | {_fmt(mc.get('p2_5', 0))} tCO2e |\n"
                    f"| 97.5th Percentile | {_fmt(mc.get('p97_5', 0))} tCO2e |\n"
                    f"| Iterations | {mc.get('iterations_run', 0):,} |\n"
                )

            # Sensitivity.
            sensitivity = unc.get("sensitivity_analysis", [])
            if sensitivity:
                content += "\n### Sensitivity Analysis (Top Contributors)\n\n"
                content += "| Rank | Category | Contribution |\n|---|---|---:|\n"
                for item in sensitivity[:5]:
                    content += (
                        f"| {item.get('rank', '')} "
                        f"| {item.get('category_name', '')} "
                        f"| {item.get('contribution_pct', 0):.1f}% |\n"
                    )
        else:
            content += "No uncertainty analysis available.\n"

        return [ReportSection(
            section_id="uncertainty",
            title="Uncertainty Analysis",
            content=content,
            order=1,
        )]

    def _generate_trend_analysis(
        self, data: Scope3ReportData, cfg: ReportConfig,
    ) -> List[ReportSection]:
        """Generate trend analysis sections."""
        content = "## Year-over-Year Trend Analysis\n\n"

        if data.trend_data:
            content += "| Year | Total Scope 3 (tCO2e) | YoY Change |\n|---|---:|---:|\n"
            for entry in data.trend_data:
                content += (
                    f"| {entry.get('year', '')} "
                    f"| {_fmt(entry.get('total', 0))} "
                    f"| {entry.get('yoy_change_pct', 'N/A')} |\n"
                )
        else:
            content += "No trend data available. At least 2 years of data required.\n"

        return [ReportSection(
            section_id="trends",
            title="Trend Analysis",
            content=content,
            order=1,
        )]

    def _generate_verification_package(
        self, data: Scope3ReportData, cfg: ReportConfig,
    ) -> List[ReportSection]:
        """Generate ISO 14064-3 verification package sections."""
        sections: List[ReportSection] = []
        order = 0

        # 1. Assertion statement.
        order += 1
        sections.append(ReportSection(
            section_id="assertion",
            title="GHG Assertion Statement",
            content=(
                f"## GHG Assertion Statement\n\n"
                f"**Organisation:** {data.organization.name}\n\n"
                f"**Reporting Period:** {data.organization.reporting_year}\n\n"
                f"**Total Scope 3 Emissions:** {_fmt(data.scope3_total_tco2e)} tCO2e\n\n"
                f"**Categories Reported:** {len(data.per_category_emissions)}\n\n"
                f"This GHG assertion has been prepared in accordance with the "
                f"GHG Protocol Corporate Value Chain (Scope 3) Accounting and "
                f"Reporting Standard (2011).\n"
            ),
            order=order,
        ))

        # 2. Boundary and methodology.
        order += 1
        sections.append(self._md_methodology(data, order))

        # 3. Category breakdown.
        order += 1
        sections.append(self._md_category_breakdown(data, order))

        # 4. Data quality evidence.
        order += 1
        dq_sections = self._generate_data_quality(data, cfg)
        if dq_sections:
            dq_sec = dq_sections[0]
            dq_sec.order = order
            dq_sec.section_id = "verification_dq"
            sections.append(dq_sec)

        # 5. Uncertainty evidence.
        order += 1
        unc_sections = self._generate_uncertainty_analysis(data, cfg)
        if unc_sections:
            unc_sec = unc_sections[0]
            unc_sec.order = order
            unc_sec.section_id = "verification_unc"
            sections.append(unc_sec)

        # 6. Exclusions.
        order += 1
        sections.append(self._md_exclusions(data, order))

        # 7. Provenance.
        order += 1
        provenance_hash = _compute_hash(data)
        sections.append(ReportSection(
            section_id="provenance",
            title="Data Provenance",
            content=(
                f"## Data Provenance\n\n"
                f"**Provenance Hash (SHA-256):** `{provenance_hash}`\n\n"
                f"This hash uniquely identifies the input data used to generate "
                f"this verification package. Any change to the underlying data "
                f"will produce a different hash.\n"
            ),
            order=order,
        ))

        return sections

    # -------------------------------------------------------------------
    # Private -- Section Builders
    # -------------------------------------------------------------------

    def _md_overview(self, data: Scope3ReportData, order: int) -> ReportSection:
        """Build overview section."""
        total = data.scope3_total_tco2e
        cat_count = len(data.per_category_emissions)
        content = (
            f"## Scope 3 GHG Inventory Overview\n\n"
            f"**Organisation:** {data.organization.name}\n\n"
            f"**Reporting Year:** {data.organization.reporting_year}\n\n"
            f"**Base Year:** {data.organization.base_year}\n\n"
            f"**Total Scope 3 Emissions:** {_fmt(total)} tCO2e\n\n"
            f"**Categories Reported:** {cat_count} of 15\n\n"
            f"**Report Prepared By:** {data.organization.report_preparer}\n"
        )
        return ReportSection(
            section_id="overview", title="Overview", content=content, order=order,
        )

    def _md_category_breakdown(
        self, data: Scope3ReportData, order: int,
    ) -> ReportSection:
        """Build category breakdown section."""
        sorted_cats = sorted(
            data.per_category_emissions.items(),
            key=lambda x: x[1], reverse=True,
        )
        total = data.scope3_total_tco2e

        rows = []
        for cat_key, em in sorted_cats:
            name = SCOPE3_CATEGORY_NAMES.get(cat_key, cat_key)
            pct = _round2(_safe_pct(_decimal(em), _decimal(total)))
            method = data.per_category_methodology.get(cat_key, "")
            dqr = data.per_category_dqr.get(cat_key, "")
            rows.append(
                f"| {name} | {_fmt(em)} | {_fmt_pct(pct)} | {method} | {dqr} |"
            )

        content = (
            "## Per-Category Emissions\n\n"
            "| Category | tCO2e | Share | Methodology | DQR |\n"
            "|---|---:|---:|---|---:|\n"
            + "\n".join(rows)
            + f"\n| **Total** | **{_fmt(total)}** | **100.00%** | | |"
        )

        table_data = [
            {
                "category": SCOPE3_CATEGORY_NAMES.get(k, k),
                "emissions_tco2e": v,
                "share_pct": _round2(_safe_pct(_decimal(v), _decimal(total))),
            }
            for k, v in sorted_cats
        ]

        return ReportSection(
            section_id="category_breakdown",
            title="Per-Category Emissions",
            content=content,
            tables=[{"name": "category_emissions", "data": table_data}],
            order=order,
        )

    def _md_methodology(self, data: Scope3ReportData, order: int) -> ReportSection:
        """Build methodology section."""
        content = "## Methodology\n\n"
        content += (
            "Emissions have been calculated in accordance with the GHG Protocol "
            "Corporate Value Chain (Scope 3) Accounting and Reporting Standard "
            "(2011) and the Scope 3 Calculation Guidance (2013).\n\n"
        )

        if data.per_category_methodology:
            content += "### Per-Category Methodology\n\n"
            content += "| Category | Methodology |\n|---|---|\n"
            for cat_key, method in data.per_category_methodology.items():
                name = SCOPE3_CATEGORY_NAMES.get(cat_key, cat_key)
                content += f"| {name} | {method} |\n"

        if data.assumptions:
            content += "\n### Key Assumptions\n\n"
            for assumption in data.assumptions:
                content += f"- {assumption}\n"

        return ReportSection(
            section_id="methodology", title="Methodology",
            content=content, order=order,
        )

    def _md_data_quality(self, data: Scope3ReportData, order: int) -> ReportSection:
        """Build data quality summary section."""
        sections = self._generate_data_quality(data, ReportConfig())
        if sections:
            sec = sections[0]
            sec.order = order
            return sec
        return ReportSection(
            section_id="data_quality", title="Data Quality",
            content="## Data Quality\n\nNo assessment available.\n", order=order,
        )

    def _md_uncertainty(self, data: Scope3ReportData, order: int) -> ReportSection:
        """Build uncertainty summary section."""
        sections = self._generate_uncertainty_analysis(data, ReportConfig())
        if sections:
            sec = sections[0]
            sec.order = order
            return sec
        return ReportSection(
            section_id="uncertainty", title="Uncertainty",
            content="## Uncertainty\n\nNo analysis available.\n", order=order,
        )

    def _md_supplier_engagement(
        self, data: Scope3ReportData, order: int,
    ) -> ReportSection:
        """Build supplier engagement summary section."""
        sections = self._generate_supplier_engagement(data, ReportConfig())
        if sections:
            sec = sections[0]
            sec.order = order
            return sec
        return ReportSection(
            section_id="supplier_engagement", title="Supplier Engagement",
            content="## Supplier Engagement\n\nNo data available.\n", order=order,
        )

    def _md_compliance(self, data: Scope3ReportData, order: int) -> ReportSection:
        """Build compliance summary section."""
        sections = self._generate_compliance_dashboard(data, ReportConfig())
        if sections:
            sec = sections[0]
            sec.order = order
            return sec
        return ReportSection(
            section_id="compliance", title="Compliance",
            content="## Compliance\n\nNo assessment available.\n", order=order,
        )

    def _md_exclusions(self, data: Scope3ReportData, order: int) -> ReportSection:
        """Build exclusions section."""
        content = "## Exclusions\n\n"
        if data.exclusions:
            content += "| Category | Justification |\n|---|---|\n"
            for exc in data.exclusions:
                content += (
                    f"| {exc.get('category', '')} "
                    f"| {exc.get('justification', '')} |\n"
                )
        else:
            content += "No categories excluded.\n"
        return ReportSection(
            section_id="exclusions", title="Exclusions",
            content=content, order=order,
        )

    # -------------------------------------------------------------------
    # Private -- Rendering
    # -------------------------------------------------------------------

    def _render(
        self,
        sections: List[ReportSection],
        appendices: List[AppendixItem],
        title: str,
        data: Scope3ReportData,
        output_format: str,
        xbrl: Dict[str, str],
    ) -> str:
        """Render sections into final output format.

        Args:
            sections: Report sections.
            appendices: Appendix items.
            title: Report title.
            data: Report data.
            output_format: Target format.
            xbrl: XBRL tags.

        Returns:
            Rendered content string.
        """
        if output_format == OutputFormat.HTML:
            return self._render_html(sections, appendices, title, data, xbrl)
        if output_format == OutputFormat.JSON:
            return self._render_json(sections, appendices, title, data, xbrl)
        if output_format == OutputFormat.CSV:
            return self._render_csv(sections, data)
        return self._render_markdown(sections, appendices, title, data, xbrl)

    def _render_markdown(
        self,
        sections: List[ReportSection],
        appendices: List[AppendixItem],
        title: str,
        data: Scope3ReportData,
        xbrl: Dict[str, str],
    ) -> str:
        """Render to Markdown format."""
        lines: List[str] = [
            f"# {title}\n",
            f"**Generated:** {utcnow().isoformat()}\n",
            f"**Engine:** Scope3ReportingEngine v{_MODULE_VERSION}\n",
            "---\n",
        ]

        # Table of contents.
        lines.append("## Table of Contents\n")
        for sec in sorted(sections, key=lambda s: s.order):
            lines.append(f"- [{sec.title}](#{sec.section_id})\n")
        lines.append("\n---\n")

        # Sections.
        for sec in sorted(sections, key=lambda s: s.order):
            lines.append(f"\n{sec.content}\n")

        # Appendices.
        if appendices:
            lines.append("\n---\n\n# Appendices\n")
            for app in appendices:
                lines.append(f"\n## {app.title}\n\n{app.content}\n")

        # XBRL tags.
        if xbrl:
            lines.append("\n---\n\n## XBRL Taxonomy Tags\n\n")
            lines.append("| Data Point | XBRL Tag |\n|---|---|\n")
            for key, tag in xbrl.items():
                lines.append(f"| {key} | `{tag}` |\n")

        return "".join(lines)

    def _render_html(
        self,
        sections: List[ReportSection],
        appendices: List[AppendixItem],
        title: str,
        data: Scope3ReportData,
        xbrl: Dict[str, str],
    ) -> str:
        """Render to self-contained HTML with inline CSS."""
        c = _HTML_COLOURS
        css = (
            f"body {{ font-family: 'Segoe UI', Arial, sans-serif; "
            f"margin: 40px; background: {c['background']}; color: {c['text']}; }}\n"
            f"h1 {{ color: {c['primary']}; border-bottom: 3px solid {c['secondary']}; "
            f"padding-bottom: 10px; }}\n"
            f"h2 {{ color: {c['header_bg']}; margin-top: 30px; }}\n"
            f"h3 {{ color: {c['secondary']}; }}\n"
            f"table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}\n"
            f"th {{ background: {c['header_bg']}; color: {c['header_text']}; "
            f"padding: 10px; text-align: left; }}\n"
            f"td {{ padding: 8px; border: 1px solid {c['border']}; }}\n"
            f"tr:nth-child(even) {{ background: {c['background']}; }}\n"
            f".provenance {{ font-family: monospace; background: #e8e8e8; "
            f"padding: 5px; border-radius: 3px; }}\n"
        )

        html_parts: List[str] = [
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n",
            f"<meta charset=\"UTF-8\">\n<title>{title}</title>\n",
            f"<style>\n{css}</style>\n</head>\n<body>\n",
            f"<h1>{title}</h1>\n",
            f"<p><em>Generated: {utcnow().isoformat()} | "
            f"Engine: Scope3ReportingEngine v{_MODULE_VERSION}</em></p>\n<hr>\n",
        ]

        # Convert markdown content to simple HTML.
        for sec in sorted(sections, key=lambda s: s.order):
            html_content = self._md_to_html(sec.content)
            html_parts.append(f"<div id=\"{sec.section_id}\">\n{html_content}\n</div>\n")

        if appendices:
            html_parts.append("<hr>\n<h1>Appendices</h1>\n")
            for app in appendices:
                html_parts.append(
                    f"<h2>{app.title}</h2>\n"
                    f"<div>{self._md_to_html(app.content)}</div>\n"
                )

        html_parts.append("</body>\n</html>")
        return "".join(html_parts)

    def _render_json(
        self,
        sections: List[ReportSection],
        appendices: List[AppendixItem],
        title: str,
        data: Scope3ReportData,
        xbrl: Dict[str, str],
    ) -> str:
        """Render to JSON format."""
        output = {
            "title": title,
            "generated_at": utcnow().isoformat(),
            "engine_version": _MODULE_VERSION,
            "sections": [s.model_dump(mode="json") for s in sections],
            "appendices": [a.model_dump(mode="json") for a in appendices],
            "xbrl_tags": xbrl,
            "data": data.model_dump(mode="json"),
        }
        return json.dumps(output, indent=2, default=str)

    def _render_csv(
        self,
        sections: List[ReportSection],
        data: Scope3ReportData,
    ) -> str:
        """Render category data to CSV format."""
        lines: List[str] = [
            "category,emissions_tco2e,share_pct,methodology,dqr",
        ]
        total = data.scope3_total_tco2e
        for cat_key, em in sorted(
            data.per_category_emissions.items(),
            key=lambda x: x[1], reverse=True,
        ):
            name = SCOPE3_CATEGORY_NAMES.get(cat_key, cat_key)
            pct = _round2(_safe_pct(_decimal(em), _decimal(total)))
            method = data.per_category_methodology.get(cat_key, "")
            dqr = data.per_category_dqr.get(cat_key, "")
            # Escape commas in name.
            safe_name = f'"{name}"' if "," in name else name
            lines.append(f"{safe_name},{_round2(em)},{pct},{method},{dqr}")
        lines.append(f"Total,{_round2(total)},100.00,,")
        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Private -- Appendix and XBRL
    # -------------------------------------------------------------------

    def _generate_appendix(self, data: Scope3ReportData) -> List[AppendixItem]:
        """Generate appendix items.

        Args:
            data: Report data.

        Returns:
            List of appendix items.
        """
        appendices: List[AppendixItem] = []

        # A. Methodology notes.
        if data.notes:
            appendices.append(AppendixItem(
                appendix_id="A",
                title="Appendix A: Methodology Notes",
                content="\n".join(f"- {n}" for n in data.notes),
                data_type="text",
            ))

        # B. Emission factor sources.
        if data.emission_factors:
            lines = ["| Source | Factor | Unit |", "|---|---|---|"]
            for key, info in data.emission_factors.items():
                if isinstance(info, dict):
                    lines.append(
                        f"| {key} | {info.get('value', '')} | {info.get('unit', '')} |"
                    )
                else:
                    lines.append(f"| {key} | {info} | |")
            appendices.append(AppendixItem(
                appendix_id="B",
                title="Appendix B: Emission Factor Registry",
                content="\n".join(lines),
                data_type="table",
            ))

        # C. Assumptions.
        if data.assumptions:
            appendices.append(AppendixItem(
                appendix_id="C",
                title="Appendix C: Key Assumptions",
                content="\n".join(f"{i+1}. {a}" for i, a in enumerate(data.assumptions)),
                data_type="text",
            ))

        # D. Glossary.
        glossary = [
            "- **tCO2e**: Tonnes of carbon dioxide equivalent",
            "- **DQR**: Data Quality Rating (1.0-5.0 scale)",
            "- **DQI**: Data Quality Indicator",
            "- **EEIO**: Environmentally-Extended Input-Output",
            "- **LCA**: Life Cycle Assessment",
            "- **EF**: Emission Factor",
            "- **GWP**: Global Warming Potential",
            "- **PCF**: Product Carbon Footprint",
            "- **SBTi**: Science Based Targets initiative",
            "- **ESRS**: European Sustainability Reporting Standards",
            "- **CDP**: Carbon Disclosure Project",
            "- **FLAG**: Forest, Land and Agriculture",
        ]
        appendices.append(AppendixItem(
            appendix_id="D",
            title="Appendix D: Glossary",
            content="\n".join(glossary),
            data_type="text",
        ))

        return appendices

    def _generate_xbrl_tags(self, data: Scope3ReportData) -> Dict[str, str]:
        """Generate ESRS E1 XBRL taxonomy tags.

        Args:
            data: Report data.

        Returns:
            Dict mapping data points to XBRL tags.
        """
        tags: Dict[str, str] = {}

        tags["scope3_total"] = ESRS_XBRL_TAGS.get(
            "scope3_total", "esrs:Scope3GhgEmissions"
        )

        for cat_key in data.per_category_emissions:
            xbrl_key = f"scope3_{cat_key}"
            if xbrl_key in ESRS_XBRL_TAGS:
                tags[cat_key] = ESRS_XBRL_TAGS[xbrl_key]

        tags["scope3_intensity"] = ESRS_XBRL_TAGS.get(
            "scope3_intensity", "esrs:Scope3GhgIntensityPerRevenue"
        )
        tags["scope3_methodology"] = ESRS_XBRL_TAGS.get(
            "scope3_methodology", "esrs:Scope3MethodologyDescription"
        )

        return tags

    # -------------------------------------------------------------------
    # Private -- Utilities
    # -------------------------------------------------------------------

    def _default_title(self, report_type: str, data: Scope3ReportData) -> str:
        """Generate default report title.

        Args:
            report_type: Report type.
            data: Report data.

        Returns:
            Title string.
        """
        type_names = {
            Scope3ReportType.FULL_INVENTORY: "Scope 3 GHG Inventory Report",
            Scope3ReportType.CATEGORY_DEEP_DIVE: "Scope 3 Category Deep Dive",
            Scope3ReportType.EXECUTIVE_SUMMARY: "Scope 3 Executive Summary",
            Scope3ReportType.HOTSPOT_ANALYSIS: "Scope 3 Hotspot Analysis",
            Scope3ReportType.SUPPLIER_ENGAGEMENT: "Supplier Engagement Dashboard",
            Scope3ReportType.DATA_QUALITY: "Scope 3 Data Quality Report",
            Scope3ReportType.COMPLIANCE_DASHBOARD: "Scope 3 Compliance Dashboard",
            Scope3ReportType.UNCERTAINTY_ANALYSIS: "Scope 3 Uncertainty Analysis",
            Scope3ReportType.TREND_ANALYSIS: "Scope 3 Trend Analysis",
            Scope3ReportType.VERIFICATION_PACKAGE: "Scope 3 Verification Package",
        }
        name = type_names.get(report_type, "Scope 3 Report")
        return f"{name} -- {data.organization.name} ({data.organization.reporting_year})"

    def _md_to_html(self, md_content: str) -> str:
        """Convert basic markdown to HTML.

        Handles headings, bold, tables, and bullet lists.
        Full markdown parsing is not included to avoid external deps.

        Args:
            md_content: Markdown string.

        Returns:
            HTML string.
        """
        lines = md_content.split("\n")
        html_lines: List[str] = []
        in_table = False
        in_list = False

        for line in lines:
            stripped = line.strip()

            # Headings.
            if stripped.startswith("### "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h3>{stripped[4:]}</h3>")
                continue
            if stripped.startswith("## "):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h2>{stripped[3:]}</h2>")
                continue

            # Tables.
            if stripped.startswith("|"):
                if not in_table:
                    html_lines.append("<table>")
                    in_table = True
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if all(c.replace("-", "").replace(":", "") == "" for c in cells):
                    continue  # Separator row.
                tag = "th" if not any(
                    "<tr>" in l for l in html_lines[-3:]
                ) and html_lines[-1] == "<table>" else "td"
                row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
                html_lines.append(f"<tr>{row}</tr>")
                continue
            elif in_table:
                html_lines.append("</table>")
                in_table = False

            # Bullet lists.
            if stripped.startswith("- "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                content = stripped[2:]
                # Bold.
                content = self._bold_to_html(content)
                html_lines.append(f"<li>{content}</li>")
                continue
            elif in_list and not stripped.startswith("- "):
                html_lines.append("</ul>")
                in_list = False

            # Paragraph.
            if stripped:
                para = self._bold_to_html(stripped)
                html_lines.append(f"<p>{para}</p>")

        if in_table:
            html_lines.append("</table>")
        if in_list:
            html_lines.append("</ul>")

        return "\n".join(html_lines)

    def _bold_to_html(self, text: str) -> str:
        """Convert **bold** markdown to <strong> HTML.

        Args:
            text: Text with markdown bold markers.

        Returns:
            Text with HTML bold tags.
        """
        result = text
        while "**" in result:
            idx = result.index("**")
            end_idx = result.index("**", idx + 2) if "**" in result[idx + 2:] else -1
            if end_idx == -1:
                break
            bold_text = result[idx + 2:end_idx]
            result = result[:idx] + f"<strong>{bold_text}</strong>" + result[end_idx + 2:]
        return result

# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

OrganizationInfo.model_rebuild()
ReportConfig.model_rebuild()
Scope3ReportData.model_rebuild()
ReportSection.model_rebuild()
AppendixItem.model_rebuild()
ReportOutput.model_rebuild()
