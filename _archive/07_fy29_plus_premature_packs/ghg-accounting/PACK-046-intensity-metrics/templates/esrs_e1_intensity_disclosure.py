# -*- coding: utf-8 -*-
"""
ESRSE1IntensityDisclosure - ESRS E1-6 Intensity Disclosure for PACK-046.

Generates an ESRS E1-6 compliant intensity disclosure report with GHG
intensity per net revenue (mandatory under ESRS E1), sector-specific
physical intensity metrics, methodology description, data quality
statement, and comparative prior year figures. Supports XBRL tagging.

ESRS Reference:
    - ESRS E1: Climate Change
    - DR E1-6: Gross Scopes 1, 2, 3 and Total GHG emissions -
      GHG intensity per net revenue
    - Paragraph 53-55: Mandatory intensity metrics

Sections:
    1. ESRS Reference
    2. Reporting Period
    3. Scope Coverage
    4. Revenue Intensity Table (mandatory)
    5. Physical Intensity Table (if applicable)
    6. Methodology Description
    7. Data Quality Statement
    8. Comparative Prior Year

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - PDF (via HTML rendering pipeline)
    - JSON (structured disclosure data)
    - XBRL (tagged disclosure elements)

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    XBRL = "xbrl"
    CSV = "csv"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class RevenueIntensityRow(BaseModel):
    """Revenue-based intensity metric row (mandatory under E1-6)."""
    scope_label: str = Field(..., description="Scope label (e.g., Scope 1, Scope 1+2)")
    gross_emissions_tco2e: float = Field(0.0, description="Gross emissions (tCO2e)")
    net_revenue_eur: float = Field(0.0, description="Net revenue (EUR millions)")
    intensity_tco2e_per_m_eur: float = Field(0.0, description="Intensity (tCO2e per M EUR)")
    prior_year_intensity: Optional[float] = Field(None, description="Prior year intensity")
    change_pct: Optional[float] = Field(None, description="Year-over-year change %")

class PhysicalIntensityRow(BaseModel):
    """Sector-specific physical intensity metric."""
    metric_name: str = Field(..., description="Physical intensity metric name")
    numerator_value: float = Field(0.0, description="Numerator value (e.g., tCO2e)")
    numerator_unit: str = Field("tCO2e", description="Numerator unit")
    denominator_value: float = Field(0.0, description="Denominator value")
    denominator_unit: str = Field("", description="Denominator unit")
    intensity_value: float = Field(0.0, description="Calculated intensity")
    intensity_unit: str = Field("", description="Intensity unit")
    sector_standard: str = Field("", description="Sector standard reference")
    prior_year_intensity: Optional[float] = Field(None, description="Prior year value")
    change_pct: Optional[float] = Field(None, description="YoY change %")

class ESRSDisclosureInput(BaseModel):
    """Complete input model for ESRS E1 Intensity Disclosure."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period_start: str = Field("", description="Period start (ISO date)")
    reporting_period_end: str = Field("", description="Period end (ISO date)")
    reporting_year: int = Field(0, description="Reporting year")
    prior_year: Optional[int] = Field(None, description="Prior comparative year")
    esrs_reference: str = Field(
        "ESRS E1-6 (DR E1-6, paragraphs 53-55)",
        description="ESRS disclosure reference",
    )
    scope_coverage_description: str = Field(
        "", description="Description of scopes included"
    )
    scope_1_included: bool = Field(True, description="Scope 1 included")
    scope_2_included: bool = Field(True, description="Scope 2 included")
    scope_3_included: bool = Field(False, description="Scope 3 included")
    scope_2_method: str = Field(
        "location-based", description="Scope 2 method (location-based / market-based)"
    )
    revenue_intensity_table: List[RevenueIntensityRow] = Field(
        default_factory=list, description="Revenue intensity rows"
    )
    physical_intensity_table: List[PhysicalIntensityRow] = Field(
        default_factory=list, description="Physical intensity rows (sector-specific)"
    )
    methodology_description: str = Field(
        "", description="Methodology description paragraph"
    )
    data_quality_statement: str = Field(
        "", description="Data quality statement"
    )
    consolidation_approach: str = Field(
        "operational control", description="Consolidation approach"
    )
    currency: str = Field("EUR", description="Revenue currency")

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ESRSE1IntensityDisclosure:
    """
    ESRS E1-6 intensity disclosure template.

    Renders ESRS-compliant intensity disclosures with mandatory
    GHG intensity per net revenue, optional sector-specific physical
    intensity metrics, methodology descriptions, data quality
    statements, and comparative prior year figures. Supports XBRL
    tagging for digital reporting.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = ESRSE1IntensityDisclosure()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> xbrl = template.render_xbrl(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESRSE1IntensityDisclosure."""
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
        """Render ESRS disclosure as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ESRS disclosure as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ESRS disclosure as JSON dict."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_xbrl(self, data: Dict[str, Any]) -> str:
        """
        Render ESRS disclosure as XBRL-tagged XML.

        Args:
            data: Disclosure data dict.

        Returns:
            XBRL XML string with ESRS taxonomy tags.
        """
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_xbrl(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_esrs_reference(data),
            self._md_scope_coverage(data),
            self._md_revenue_intensity(data),
            self._md_physical_intensity(data),
            self._md_methodology(data),
            self._md_data_quality(data),
            self._md_comparative(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# ESRS E1-6 Intensity Disclosure - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_esrs_reference(self, data: Dict[str, Any]) -> str:
        """Render Markdown ESRS reference."""
        ref = self._get_val(data, "esrs_reference", "ESRS E1-6")
        return (
            "## DR E1-6: GHG Intensity per Net Revenue\n\n"
            f"**Disclosure Requirement:** {ref}\n\n"
            "The undertaking shall disclose the GHG intensity per net revenue "
            "as required under ESRS E1, paragraphs 53-55."
        )

    def _md_scope_coverage(self, data: Dict[str, Any]) -> str:
        """Render Markdown scope coverage."""
        s1 = data.get("scope_1_included", True)
        s2 = data.get("scope_2_included", True)
        s3 = data.get("scope_3_included", False)
        method = self._get_val(data, "scope_2_method", "location-based")
        consolidation = self._get_val(data, "consolidation_approach", "operational control")
        desc = self._get_val(data, "scope_coverage_description", "")
        scopes = []
        if s1:
            scopes.append("Scope 1")
        if s2:
            scopes.append(f"Scope 2 ({method})")
        if s3:
            scopes.append("Scope 3")
        lines = [
            "### Scope Coverage",
            "",
            f"**Scopes Included:** {', '.join(scopes)}",
            f"**Consolidation Approach:** {consolidation.title()}",
        ]
        if desc:
            lines.append(f"\n{desc}")
        return "\n".join(lines)

    def _md_revenue_intensity(self, data: Dict[str, Any]) -> str:
        """Render Markdown revenue intensity table (mandatory)."""
        rows = data.get("revenue_intensity_table", [])
        currency = self._get_val(data, "currency", "EUR")
        if not rows:
            return (
                "### GHG Intensity per Net Revenue\n\n"
                "No revenue intensity data available."
            )
        lines = [
            "### GHG Intensity per Net Revenue",
            "",
            f"| Scope | Gross Emissions (tCO2e) | Net Revenue (M {currency}) | "
            f"Intensity (tCO2e/M {currency}) | Prior Year | Change |",
            "|-------|------------------------|---------------------------|"
            "-------------------------------|------------|--------|",
        ]
        for r in rows:
            scope = r.get("scope_label", "")
            emissions = r.get("gross_emissions_tco2e", 0)
            revenue = r.get("net_revenue_eur", 0)
            intensity = r.get("intensity_tco2e_per_m_eur", 0)
            prior = r.get("prior_year_intensity")
            change = r.get("change_pct")
            prior_str = f"{prior:,.2f}" if prior is not None else "-"
            change_str = f"{change:+.1f}%" if change is not None else "-"
            lines.append(
                f"| {scope} | {emissions:,.1f} | {revenue:,.1f} | "
                f"{intensity:,.2f} | {prior_str} | {change_str} |"
            )
        return "\n".join(lines)

    def _md_physical_intensity(self, data: Dict[str, Any]) -> str:
        """Render Markdown physical intensity table (sector-specific)."""
        rows = data.get("physical_intensity_table", [])
        if not rows:
            return ""
        lines = [
            "### Sector-Specific Physical Intensity Metrics",
            "",
            "| Metric | Intensity | Unit | Sector Standard | Prior Year | Change |",
            "|--------|-----------|------|-----------------|------------|--------|",
        ]
        for r in rows:
            name = r.get("metric_name", "")
            intensity = r.get("intensity_value", 0)
            unit = r.get("intensity_unit", "")
            standard = r.get("sector_standard", "-")
            prior = r.get("prior_year_intensity")
            change = r.get("change_pct")
            prior_str = f"{prior:,.4f}" if prior is not None else "-"
            change_str = f"{change:+.1f}%" if change is not None else "-"
            lines.append(
                f"| {name} | {intensity:,.4f} | {unit} | "
                f"{standard} | {prior_str} | {change_str} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology description."""
        desc = self._get_val(data, "methodology_description", "")
        if not desc:
            desc = (
                "GHG intensity is calculated by dividing gross GHG emissions "
                "(in tCO2e) by net revenue (in millions of EUR). The calculation "
                "follows the GHG Protocol Corporate Standard and ESRS E1-6 "
                "requirements."
            )
        return f"### Methodology\n\n{desc}"

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality statement."""
        stmt = self._get_val(data, "data_quality_statement", "")
        if not stmt:
            return ""
        return f"### Data Quality Statement\n\n{stmt}"

    def _md_comparative(self, data: Dict[str, Any]) -> str:
        """Render Markdown comparative prior year note."""
        prior_year = data.get("prior_year")
        if not prior_year:
            return ""
        return (
            f"### Comparative Information\n\n"
            f"Comparative figures for {prior_year} are presented alongside "
            f"the current reporting period in the tables above."
        )

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
            self._html_esrs_reference(data),
            self._html_scope_coverage(data),
            self._html_revenue_intensity(data),
            self._html_physical_intensity(data),
            self._html_methodology(data),
            self._html_data_quality(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>ESRS E1-6 Disclosure - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #003399;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#264653;margin-top:1.5rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#e8eef6;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".esrs-ref{background:#f0f4f8;border-left:4px solid #003399;"
            "padding:1rem 1.5rem;margin:1rem 0;font-size:0.9rem;}\n"
            ".change-positive{color:#e76f51;}\n"
            ".change-negative{color:#2a9d8f;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>ESRS E1-6: GHG Intensity Disclosure &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n<hr>\n</div>"
        )

    def _html_esrs_reference(self, data: Dict[str, Any]) -> str:
        """Render HTML ESRS reference box."""
        ref = self._get_val(data, "esrs_reference", "ESRS E1-6")
        return (
            '<div class="section">\n<h2>DR E1-6: GHG Intensity per Net Revenue</h2>\n'
            f'<div class="esrs-ref">\n'
            f"<p><strong>Disclosure Requirement:</strong> {ref}</p>\n"
            "<p>The undertaking shall disclose the GHG intensity per net revenue "
            "as required under ESRS E1, paragraphs 53-55.</p>\n"
            "</div>\n</div>"
        )

    def _html_scope_coverage(self, data: Dict[str, Any]) -> str:
        """Render HTML scope coverage."""
        s1 = data.get("scope_1_included", True)
        s2 = data.get("scope_2_included", True)
        s3 = data.get("scope_3_included", False)
        method = self._get_val(data, "scope_2_method", "location-based")
        scopes = []
        if s1:
            scopes.append("Scope 1")
        if s2:
            scopes.append(f"Scope 2 ({method})")
        if s3:
            scopes.append("Scope 3")
        return (
            '<div class="section">\n<h3>Scope Coverage</h3>\n'
            f"<p><strong>Scopes Included:</strong> {', '.join(scopes)}</p>\n</div>"
        )

    def _html_revenue_intensity(self, data: Dict[str, Any]) -> str:
        """Render HTML revenue intensity table."""
        rows_data = data.get("revenue_intensity_table", [])
        currency = self._get_val(data, "currency", "EUR")
        if not rows_data:
            return ""
        rows = ""
        for r in rows_data:
            scope = r.get("scope_label", "")
            emissions = r.get("gross_emissions_tco2e", 0)
            revenue = r.get("net_revenue_eur", 0)
            intensity = r.get("intensity_tco2e_per_m_eur", 0)
            prior = r.get("prior_year_intensity")
            change = r.get("change_pct")
            prior_str = f"{prior:,.2f}" if prior is not None else "-"
            if change is not None:
                change_css = "change-positive" if change > 0 else "change-negative"
                change_str = f'<span class="{change_css}">{change:+.1f}%</span>'
            else:
                change_str = "-"
            rows += (
                f"<tr><td>{scope}</td><td>{emissions:,.1f}</td>"
                f"<td>{revenue:,.1f}</td><td><strong>{intensity:,.2f}</strong></td>"
                f"<td>{prior_str}</td><td>{change_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h3>GHG Intensity per Net Revenue</h3>\n'
            "<table><thead><tr><th>Scope</th><th>Gross Emissions (tCO2e)</th>"
            f"<th>Net Revenue (M {currency})</th>"
            f"<th>Intensity (tCO2e/M {currency})</th>"
            "<th>Prior Year</th><th>Change</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_physical_intensity(self, data: Dict[str, Any]) -> str:
        """Render HTML physical intensity table."""
        rows_data = data.get("physical_intensity_table", [])
        if not rows_data:
            return ""
        rows = ""
        for r in rows_data:
            name = r.get("metric_name", "")
            intensity = r.get("intensity_value", 0)
            unit = r.get("intensity_unit", "")
            standard = r.get("sector_standard", "-")
            rows += (
                f"<tr><td>{name}</td><td>{intensity:,.4f}</td>"
                f"<td>{unit}</td><td>{standard}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h3>Physical Intensity Metrics</h3>\n'
            "<table><thead><tr><th>Metric</th><th>Intensity</th>"
            "<th>Unit</th><th>Standard</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology."""
        desc = self._get_val(data, "methodology_description", "")
        if not desc:
            return ""
        return (
            '<div class="section">\n<h3>Methodology</h3>\n'
            f"<p>{desc}</p>\n</div>"
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality statement."""
        stmt = self._get_val(data, "data_quality_statement", "")
        if not stmt:
            return ""
        return (
            '<div class="section">\n<h3>Data Quality Statement</h3>\n'
            f"<p>{stmt}</p>\n</div>"
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
    # XBRL RENDERING
    # ==================================================================

    def _render_xbrl(self, data: Dict[str, Any]) -> str:
        """Render ESRS disclosure as XBRL-tagged XML."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        period_start = self._get_val(data, "reporting_period_start", f"{year}-01-01")
        period_end = self._get_val(data, "reporting_period_end", f"{year}-12-31")
        provenance = self._compute_provenance(data)
        currency = self._get_val(data, "currency", "EUR")

        revenue_facts = ""
        for r in data.get("revenue_intensity_table", []):
            scope = r.get("scope_label", "").replace(" ", "").replace("+", "And")
            intensity = r.get("intensity_tco2e_per_m_eur", 0)
            emissions = r.get("gross_emissions_tco2e", 0)
            revenue_facts += (
                f'    <esrs:GHGIntensityPerNetRevenue_{scope} '
                f'contextRef="c_{year}" unitRef="tCO2ePerM{currency}" '
                f'decimals="2">{intensity}</esrs:GHGIntensityPerNetRevenue_{scope}>\n'
                f'    <esrs:GrossGHGEmissions_{scope} '
                f'contextRef="c_{year}" unitRef="tCO2e" '
                f'decimals="1">{emissions}</esrs:GrossGHGEmissions_{scope}>\n'
            )

        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"\n'
            '    xmlns:esrs="http://xbrl.efrag.org/taxonomy/esrs/2024"\n'
            '    xmlns:iso4217="http://www.xbrl.org/2003/iso4217"\n'
            '    xmlns:link="http://www.xbrl.org/2003/linkbase">\n\n'
            '  <!-- Context -->\n'
            f'  <xbrli:context id="c_{year}">\n'
            f'    <xbrli:entity>\n'
            f'      <xbrli:identifier scheme="http://greenlang.io">{company}</xbrli:identifier>\n'
            f'    </xbrli:entity>\n'
            f'    <xbrli:period>\n'
            f'      <xbrli:startDate>{period_start}</xbrli:startDate>\n'
            f'      <xbrli:endDate>{period_end}</xbrli:endDate>\n'
            f'    </xbrli:period>\n'
            f'  </xbrli:context>\n\n'
            '  <!-- Units -->\n'
            f'  <xbrli:unit id="tCO2ePerM{currency}">\n'
            f'    <xbrli:measure>esrs:tCO2ePerMillionRevenue</xbrli:measure>\n'
            f'  </xbrli:unit>\n'
            f'  <xbrli:unit id="tCO2e">\n'
            f'    <xbrli:measure>esrs:tCO2e</xbrli:measure>\n'
            f'  </xbrli:unit>\n\n'
            '  <!-- DR E1-6: GHG Intensity per Net Revenue -->\n'
            f'{revenue_facts}\n'
            f'  <!-- Provenance: {provenance} -->\n\n'
            '</xbrli:xbrl>\n'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ESRS disclosure as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "esrs_e1_intensity_disclosure",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "esrs_reference": self._get_val(data, "esrs_reference", "ESRS E1-6"),
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year", ""),
            "scope_coverage": {
                "scope_1": data.get("scope_1_included", True),
                "scope_2": data.get("scope_2_included", True),
                "scope_3": data.get("scope_3_included", False),
                "scope_2_method": self._get_val(data, "scope_2_method", "location-based"),
                "consolidation": self._get_val(data, "consolidation_approach", ""),
            },
            "revenue_intensity_table": data.get("revenue_intensity_table", []),
            "physical_intensity_table": data.get("physical_intensity_table", []),
            "methodology_description": self._get_val(data, "methodology_description", ""),
            "data_quality_statement": self._get_val(data, "data_quality_statement", ""),
        }
