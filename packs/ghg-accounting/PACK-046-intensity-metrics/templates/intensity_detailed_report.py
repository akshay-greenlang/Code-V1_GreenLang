# -*- coding: utf-8 -*-
"""
IntensityDetailedReport - Comprehensive Intensity Report for PACK-046.

Generates a detailed intensity report covering methodology, scope
configuration, denominator details, intensity tables by scope and
denominator, multi-year time series, entity-level breakdowns, data
sources, and limitations.

Sections:
    1. Methodology
    2. Scope Configuration
    3. Denominator Details
    4. Intensity by Scope (table)
    5. Intensity by Denominator (table)
    6. Time Series (multi-year)
    7. Entity Breakdown (per business unit)
    8. Data Sources
    9. Limitations

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - PDF (via HTML rendering pipeline)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 46.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats."""
    MD = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"


class ScopeType(str, Enum):
    """GHG emission scope types."""
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"
    TOTAL = "total"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class DenominatorDetail(BaseModel):
    """Denominator definition and metadata."""
    denominator_id: str = Field(..., description="Unique denominator identifier")
    name: str = Field(..., description="Human-readable denominator name")
    unit: str = Field(..., description="Denominator unit (e.g., M USD, FTE)")
    value: float = Field(..., description="Denominator value for current period")
    source: str = Field("", description="Data source for denominator")
    data_quality: str = Field("medium", description="Data quality rating")
    notes: str = Field("", description="Additional notes")


class IntensityByScope(BaseModel):
    """Intensity result for a single scope."""
    scope: str = Field(..., description="Scope label (e.g., Scope 1)")
    emissions_tco2e: float = Field(..., description="Emissions in tCO2e")
    denominator_value: float = Field(..., description="Denominator value")
    denominator_unit: str = Field("", description="Denominator unit")
    intensity_value: float = Field(..., description="Calculated intensity")
    intensity_unit: str = Field("", description="Intensity unit string")


class IntensityByDenominator(BaseModel):
    """Intensity result for a single denominator across scopes."""
    denominator_name: str = Field(..., description="Denominator name")
    denominator_unit: str = Field("", description="Denominator unit")
    scope_1: Optional[float] = Field(None, description="Scope 1 intensity")
    scope_2_location: Optional[float] = Field(None, description="Scope 2 location intensity")
    scope_2_market: Optional[float] = Field(None, description="Scope 2 market intensity")
    scope_3: Optional[float] = Field(None, description="Scope 3 intensity")
    total_s1_s2: Optional[float] = Field(None, description="Scope 1+2 combined intensity")
    total_all: Optional[float] = Field(None, description="All scopes combined intensity")


class TimeSeriesRow(BaseModel):
    """Single year in multi-year time series."""
    year: int = Field(..., description="Reporting year")
    scope_1_intensity: Optional[float] = Field(None, description="Scope 1 intensity")
    scope_2_intensity: Optional[float] = Field(None, description="Scope 2 intensity")
    scope_3_intensity: Optional[float] = Field(None, description="Scope 3 intensity")
    total_intensity: Optional[float] = Field(None, description="Total intensity")
    denominator_value: Optional[float] = Field(None, description="Denominator value")
    denominator_unit: str = Field("", description="Denominator unit")


class EntityBreakdown(BaseModel):
    """Business unit / entity level breakdown."""
    entity_name: str = Field(..., description="Entity or business unit name")
    entity_type: str = Field("business_unit", description="Entity type")
    emissions_tco2e: float = Field(0.0, description="Entity emissions")
    denominator_value: float = Field(0.0, description="Entity denominator value")
    intensity_value: float = Field(0.0, description="Entity intensity")
    share_of_total_pct: float = Field(0.0, description="Share of total emissions %")


class DataSourceEntry(BaseModel):
    """Data source reference."""
    source_name: str = Field(..., description="Source system or document name")
    source_type: str = Field("", description="Source type (ERP, manual, API)")
    coverage: str = Field("", description="Coverage description")
    last_updated: str = Field("", description="Last update date")
    quality_score: Optional[float] = Field(None, description="Data quality score 0-100")


class DetailedReportInput(BaseModel):
    """Complete input model for IntensityDetailedReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date")
    methodology_description: str = Field("", description="Methodology narrative")
    calculation_approach: str = Field("", description="Calculation approach description")
    scope_configuration: Dict[str, Any] = Field(
        default_factory=dict, description="Scope configuration details"
    )
    denominator_details: List[DenominatorDetail] = Field(
        default_factory=list, description="Denominator definitions"
    )
    intensity_by_scope: List[IntensityByScope] = Field(
        default_factory=list, description="Intensity results by scope"
    )
    intensity_by_denominator: List[IntensityByDenominator] = Field(
        default_factory=list, description="Intensity results by denominator"
    )
    time_series: List[TimeSeriesRow] = Field(
        default_factory=list, description="Multi-year time series"
    )
    entity_breakdown: List[EntityBreakdown] = Field(
        default_factory=list, description="Entity-level breakdown"
    )
    data_sources: List[DataSourceEntry] = Field(
        default_factory=list, description="Data source references"
    )
    limitations: List[str] = Field(
        default_factory=list, description="Report limitations"
    )


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class IntensityDetailedReport:
    """
    Comprehensive intensity detailed report template.

    Renders a full technical report with methodology, scope configuration,
    denominator details, multiple intensity tables, multi-year trends,
    entity breakdowns, data sources, and limitations. All outputs include
    SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = IntensityDetailedReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IntensityDetailedReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render detailed report as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render detailed report as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render detailed report as JSON-serializable dict."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_methodology(data),
            self._md_scope_config(data),
            self._md_denominators(data),
            self._md_intensity_by_scope(data),
            self._md_intensity_by_denominator(data),
            self._md_time_series(data),
            self._md_entity_breakdown(data),
            self._md_data_sources(data),
            self._md_limitations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Intensity Metrics Detailed Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology section."""
        desc = self._get_val(data, "methodology_description", "")
        approach = self._get_val(data, "calculation_approach", "")
        if not desc and not approach:
            return "## 1. Methodology\n\nNo methodology description provided."
        lines = ["## 1. Methodology", ""]
        if desc:
            lines.append(desc)
            lines.append("")
        if approach:
            lines.append(f"**Calculation Approach:** {approach}")
        return "\n".join(lines)

    def _md_scope_config(self, data: Dict[str, Any]) -> str:
        """Render Markdown scope configuration."""
        config = data.get("scope_configuration", {})
        if not config:
            return "## 2. Scope Configuration\n\nDefault scope configuration applied."
        lines = ["## 2. Scope Configuration", ""]
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for key, val in config.items():
            display_key = key.replace("_", " ").title()
            lines.append(f"| {display_key} | {val} |")
        return "\n".join(lines)

    def _md_denominators(self, data: Dict[str, Any]) -> str:
        """Render Markdown denominator details."""
        denoms = data.get("denominator_details", [])
        if not denoms:
            return "## 3. Denominator Details\n\nNo denominator data available."
        lines = [
            "## 3. Denominator Details",
            "",
            "| Denominator | Unit | Value | Source | Quality |",
            "|------------|------|-------|--------|---------|",
        ]
        for d in denoms:
            name = d.get("name", "")
            unit = d.get("unit", "")
            value = d.get("value", 0)
            source = d.get("source", "-")
            quality = d.get("data_quality", "-")
            lines.append(f"| {name} | {unit} | {value:,.2f} | {source} | {quality} |")
        return "\n".join(lines)

    def _md_intensity_by_scope(self, data: Dict[str, Any]) -> str:
        """Render Markdown intensity by scope table."""
        rows = data.get("intensity_by_scope", [])
        if not rows:
            return "## 4. Intensity by Scope\n\nNo scope intensity data available."
        lines = [
            "## 4. Intensity by Scope",
            "",
            "| Scope | Emissions (tCO2e) | Denominator | Intensity | Unit |",
            "|-------|-------------------|-------------|-----------|------|",
        ]
        for r in rows:
            scope = r.get("scope", "")
            emissions = r.get("emissions_tco2e", 0)
            denom = r.get("denominator_value", 0)
            denom_unit = r.get("denominator_unit", "")
            intensity = r.get("intensity_value", 0)
            int_unit = r.get("intensity_unit", "")
            lines.append(
                f"| {scope} | {emissions:,.1f} | {denom:,.2f} {denom_unit} | "
                f"{intensity:,.4f} | {int_unit} |"
            )
        return "\n".join(lines)

    def _md_intensity_by_denominator(self, data: Dict[str, Any]) -> str:
        """Render Markdown intensity by denominator table."""
        rows = data.get("intensity_by_denominator", [])
        if not rows:
            return "## 5. Intensity by Denominator\n\nNo data available."
        lines = [
            "## 5. Intensity by Denominator",
            "",
            "| Denominator | Unit | Scope 1 | Scope 2 (Loc) | Scope 2 (Mkt) | Scope 3 | S1+S2 | Total |",
            "|------------|------|---------|---------------|---------------|---------|-------|-------|",
        ]
        for r in rows:
            name = r.get("denominator_name", "")
            unit = r.get("denominator_unit", "")
            s1 = r.get("scope_1")
            s2l = r.get("scope_2_location")
            s2m = r.get("scope_2_market")
            s3 = r.get("scope_3")
            s12 = r.get("total_s1_s2")
            total = r.get("total_all")
            fmt = lambda v: f"{v:,.4f}" if v is not None else "-"
            lines.append(
                f"| {name} | {unit} | {fmt(s1)} | {fmt(s2l)} | "
                f"{fmt(s2m)} | {fmt(s3)} | {fmt(s12)} | {fmt(total)} |"
            )
        return "\n".join(lines)

    def _md_time_series(self, data: Dict[str, Any]) -> str:
        """Render Markdown multi-year time series."""
        series = data.get("time_series", [])
        if not series:
            return "## 6. Time Series\n\nNo time series data available."
        lines = [
            "## 6. Time Series",
            "",
            "| Year | Scope 1 | Scope 2 | Scope 3 | Total | Denominator |",
            "|------|---------|---------|---------|-------|-------------|",
        ]
        fmt = lambda v: f"{v:,.4f}" if v is not None else "-"
        for r in series:
            year = r.get("year", "")
            s1 = r.get("scope_1_intensity")
            s2 = r.get("scope_2_intensity")
            s3 = r.get("scope_3_intensity")
            total = r.get("total_intensity")
            denom = r.get("denominator_value")
            denom_unit = r.get("denominator_unit", "")
            denom_str = f"{denom:,.2f} {denom_unit}" if denom is not None else "-"
            lines.append(
                f"| {year} | {fmt(s1)} | {fmt(s2)} | {fmt(s3)} | "
                f"{fmt(total)} | {denom_str} |"
            )
        return "\n".join(lines)

    def _md_entity_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Markdown entity-level breakdown."""
        entities = data.get("entity_breakdown", [])
        if not entities:
            return "## 7. Entity Breakdown\n\nNo entity breakdown data available."
        lines = [
            "## 7. Entity Breakdown",
            "",
            "| Entity | Emissions (tCO2e) | Denominator | Intensity | Share (%) |",
            "|--------|-------------------|-------------|-----------|-----------|",
        ]
        for e in entities:
            name = e.get("entity_name", "")
            emissions = e.get("emissions_tco2e", 0)
            denom = e.get("denominator_value", 0)
            intensity = e.get("intensity_value", 0)
            share = e.get("share_of_total_pct", 0)
            lines.append(
                f"| {name} | {emissions:,.1f} | {denom:,.2f} | "
                f"{intensity:,.4f} | {share:.1f}% |"
            )
        return "\n".join(lines)

    def _md_data_sources(self, data: Dict[str, Any]) -> str:
        """Render Markdown data sources."""
        sources = data.get("data_sources", [])
        if not sources:
            return "## 8. Data Sources\n\nNo data source information provided."
        lines = [
            "## 8. Data Sources",
            "",
            "| Source | Type | Coverage | Last Updated | Quality |",
            "|--------|------|----------|--------------|---------|",
        ]
        for s in sources:
            name = s.get("source_name", "")
            stype = s.get("source_type", "-")
            coverage = s.get("coverage", "-")
            updated = s.get("last_updated", "-")
            quality = s.get("quality_score")
            q_str = f"{quality:.0f}/100" if quality is not None else "-"
            lines.append(f"| {name} | {stype} | {coverage} | {updated} | {q_str} |")
        return "\n".join(lines)

    def _md_limitations(self, data: Dict[str, Any]) -> str:
        """Render Markdown limitations."""
        limitations = data.get("limitations", [])
        if not limitations:
            return ""
        lines = ["## 9. Limitations", ""]
        for i, lim in enumerate(limitations, 1):
            lines.append(f"{i}. {lim}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-046 Intensity Metrics v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_methodology(data),
            self._html_scope_config(data),
            self._html_denominators(data),
            self._html_intensity_by_scope(data),
            self._html_intensity_by_denominator(data),
            self._html_time_series(data),
            self._html_entity_breakdown(data),
            self._html_data_sources(data),
            self._html_limitations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Intensity Detailed Report - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".methodology-box{background:#f8f9fa;border-left:4px solid #2a9d8f;"
            "padding:1rem 1.5rem;margin:1rem 0;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            '<div class="section">\n'
            f"<h1>Intensity Metrics Detailed Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period}</p>\n"
            "<hr>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology section."""
        desc = self._get_val(data, "methodology_description", "")
        approach = self._get_val(data, "calculation_approach", "")
        if not desc and not approach:
            return ""
        content = ""
        if desc:
            content += f"<p>{desc}</p>\n"
        if approach:
            content += f"<p><strong>Calculation Approach:</strong> {approach}</p>\n"
        return (
            '<div class="section">\n<h2>1. Methodology</h2>\n'
            f'<div class="methodology-box">{content}</div>\n</div>'
        )

    def _html_scope_config(self, data: Dict[str, Any]) -> str:
        """Render HTML scope configuration table."""
        config = data.get("scope_configuration", {})
        if not config:
            return ""
        rows = ""
        for key, val in config.items():
            display_key = key.replace("_", " ").title()
            rows += f"<tr><td>{display_key}</td><td>{val}</td></tr>\n"
        return (
            '<div class="section">\n<h2>2. Scope Configuration</h2>\n'
            "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_denominators(self, data: Dict[str, Any]) -> str:
        """Render HTML denominator details table."""
        denoms = data.get("denominator_details", [])
        if not denoms:
            return ""
        rows = ""
        for d in denoms:
            name = d.get("name", "")
            unit = d.get("unit", "")
            value = d.get("value", 0)
            source = d.get("source", "-")
            quality = d.get("data_quality", "-")
            rows += (
                f"<tr><td>{name}</td><td>{unit}</td><td>{value:,.2f}</td>"
                f"<td>{source}</td><td>{quality}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Denominator Details</h2>\n'
            "<table><thead><tr><th>Denominator</th><th>Unit</th><th>Value</th>"
            "<th>Source</th><th>Quality</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_intensity_by_scope(self, data: Dict[str, Any]) -> str:
        """Render HTML intensity by scope table."""
        items = data.get("intensity_by_scope", [])
        if not items:
            return ""
        rows = ""
        for r in items:
            scope = r.get("scope", "")
            emissions = r.get("emissions_tco2e", 0)
            denom = r.get("denominator_value", 0)
            denom_unit = r.get("denominator_unit", "")
            intensity = r.get("intensity_value", 0)
            int_unit = r.get("intensity_unit", "")
            rows += (
                f"<tr><td>{scope}</td><td>{emissions:,.1f}</td>"
                f"<td>{denom:,.2f} {denom_unit}</td>"
                f"<td>{intensity:,.4f}</td><td>{int_unit}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Intensity by Scope</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Emissions (tCO2e)</th>"
            "<th>Denominator</th><th>Intensity</th><th>Unit</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_intensity_by_denominator(self, data: Dict[str, Any]) -> str:
        """Render HTML intensity by denominator table."""
        items = data.get("intensity_by_denominator", [])
        if not items:
            return ""
        rows = ""
        fmt = lambda v: f"{v:,.4f}" if v is not None else "-"
        for r in items:
            name = r.get("denominator_name", "")
            unit = r.get("denominator_unit", "")
            s1 = r.get("scope_1")
            s2l = r.get("scope_2_location")
            s2m = r.get("scope_2_market")
            s3 = r.get("scope_3")
            s12 = r.get("total_s1_s2")
            total = r.get("total_all")
            rows += (
                f"<tr><td>{name}</td><td>{unit}</td>"
                f"<td>{fmt(s1)}</td><td>{fmt(s2l)}</td><td>{fmt(s2m)}</td>"
                f"<td>{fmt(s3)}</td><td>{fmt(s12)}</td><td>{fmt(total)}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Intensity by Denominator</h2>\n'
            "<table><thead><tr><th>Denominator</th><th>Unit</th>"
            "<th>Scope 1</th><th>S2 (Loc)</th><th>S2 (Mkt)</th>"
            "<th>Scope 3</th><th>S1+S2</th><th>Total</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_time_series(self, data: Dict[str, Any]) -> str:
        """Render HTML multi-year time series table."""
        series = data.get("time_series", [])
        if not series:
            return ""
        rows = ""
        fmt = lambda v: f"{v:,.4f}" if v is not None else "-"
        for r in series:
            year = r.get("year", "")
            s1 = r.get("scope_1_intensity")
            s2 = r.get("scope_2_intensity")
            s3 = r.get("scope_3_intensity")
            total = r.get("total_intensity")
            rows += (
                f"<tr><td>{year}</td><td>{fmt(s1)}</td>"
                f"<td>{fmt(s2)}</td><td>{fmt(s3)}</td><td>{fmt(total)}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>6. Time Series</h2>\n'
            "<table><thead><tr><th>Year</th><th>Scope 1</th>"
            "<th>Scope 2</th><th>Scope 3</th><th>Total</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_entity_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML entity breakdown table."""
        entities = data.get("entity_breakdown", [])
        if not entities:
            return ""
        rows = ""
        for e in entities:
            name = e.get("entity_name", "")
            emissions = e.get("emissions_tco2e", 0)
            denom = e.get("denominator_value", 0)
            intensity = e.get("intensity_value", 0)
            share = e.get("share_of_total_pct", 0)
            rows += (
                f"<tr><td>{name}</td><td>{emissions:,.1f}</td>"
                f"<td>{denom:,.2f}</td><td>{intensity:,.4f}</td>"
                f"<td>{share:.1f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>7. Entity Breakdown</h2>\n'
            "<table><thead><tr><th>Entity</th><th>Emissions (tCO2e)</th>"
            "<th>Denominator</th><th>Intensity</th><th>Share</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_data_sources(self, data: Dict[str, Any]) -> str:
        """Render HTML data sources table."""
        sources = data.get("data_sources", [])
        if not sources:
            return ""
        rows = ""
        for s in sources:
            name = s.get("source_name", "")
            stype = s.get("source_type", "-")
            coverage = s.get("coverage", "-")
            updated = s.get("last_updated", "-")
            quality = s.get("quality_score")
            q_str = f"{quality:.0f}/100" if quality is not None else "-"
            rows += (
                f"<tr><td>{name}</td><td>{stype}</td><td>{coverage}</td>"
                f"<td>{updated}</td><td>{q_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>8. Data Sources</h2>\n'
            "<table><thead><tr><th>Source</th><th>Type</th>"
            "<th>Coverage</th><th>Last Updated</th><th>Quality</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_limitations(self, data: Dict[str, Any]) -> str:
        """Render HTML limitations section."""
        limitations = data.get("limitations", [])
        if not limitations:
            return ""
        items = "".join(f"<li>{lim}</li>\n" for lim in limitations)
        return (
            '<div class="section">\n<h2>9. Limitations</h2>\n'
            f"<ol>{items}</ol>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-046 Intensity Metrics v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render detailed report as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "intensity_detailed_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "methodology_description": self._get_val(data, "methodology_description", ""),
            "calculation_approach": self._get_val(data, "calculation_approach", ""),
            "scope_configuration": data.get("scope_configuration", {}),
            "denominator_details": data.get("denominator_details", []),
            "intensity_by_scope": data.get("intensity_by_scope", []),
            "intensity_by_denominator": data.get("intensity_by_denominator", []),
            "time_series": data.get("time_series", []),
            "entity_breakdown": data.get("entity_breakdown", []),
            "data_sources": data.get("data_sources", []),
            "limitations": data.get("limitations", []),
        }
