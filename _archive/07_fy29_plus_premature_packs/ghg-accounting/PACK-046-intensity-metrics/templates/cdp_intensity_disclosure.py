# -*- coding: utf-8 -*-
"""
CDPIntensityDisclosure - CDP C6.10 Intensity Disclosure for PACK-046.

Generates a CDP-compliant intensity disclosure for question C6.10
covering Scope 1+2 intensity figures, metric denominators, percentage
change from previous year, direction of change, reason for change,
and sector module specific metrics.

CDP Reference:
    - C6.10: What are your gross global Scope 1 emissions in metric tons
      CO2e per unit currency total revenue and per full-time equivalent
      (FTE) employee?
    - Sector-specific C-XX intensity questions

Sections:
    1. C6.10 Data Table
    2. Sector-Specific Metrics
    3. Methodology Notes
    4. Change Explanation

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - PDF (via HTML rendering pipeline)
    - JSON (structured CDP response data)

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

class ChangeDirection(str, Enum):
    """CDP change direction options."""
    INCREASED = "increased"
    DECREASED = "decreased"
    NO_CHANGE = "no_change"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class CDPC610Row(BaseModel):
    """Single C6.10 intensity metric row."""
    intensity_figure: float = Field(0.0, description="Intensity figure (metric tons CO2e per unit)")
    metric_numerator: str = Field(
        "Gross global combined Scope 1 and 2 emissions, metric tons CO2e",
        description="CDP metric numerator text",
    )
    metric_denominator: str = Field(
        "", description="CDP metric denominator (e.g., unit total revenue)"
    )
    metric_denominator_unit: str = Field(
        "", description="Denominator unit currency or unit"
    )
    pct_change_from_previous_year: Optional[float] = Field(
        None, description="% change from previous year (direction-specific)"
    )
    direction_of_change: ChangeDirection = Field(
        ChangeDirection.NO_CHANGE, description="Direction of change"
    )
    reason_for_change: str = Field(
        "", description="Explanation for the change"
    )
    scope_coverage: str = Field(
        "Scope 1+2 (location-based)",
        description="Scope coverage for this metric",
    )

class CDPSectorMetric(BaseModel):
    """CDP sector-specific intensity metric."""
    sector_module: str = Field("", description="CDP sector module (e.g., C-EU, C-OG)")
    question_number: str = Field("", description="CDP question number")
    metric_name: str = Field(..., description="Metric name")
    intensity_value: float = Field(0.0, description="Intensity value")
    intensity_unit: str = Field("", description="Intensity unit")
    numerator_value: float = Field(0.0, description="Numerator value")
    numerator_unit: str = Field("", description="Numerator unit")
    denominator_value: float = Field(0.0, description="Denominator value")
    denominator_unit: str = Field("", description="Denominator unit")
    prior_year_value: Optional[float] = Field(None, description="Prior year intensity")
    change_pct: Optional[float] = Field(None, description="YoY change %")

class CDPDisclosureInput(BaseModel):
    """Complete input model for CDP Intensity Disclosure."""
    company_name: str = Field("Organization", description="Company name")
    reporting_year: int = Field(0, description="CDP reporting year")
    cdp_submission_year: Optional[int] = Field(None, description="CDP submission year")
    c6_10_data: List[CDPC610Row] = Field(
        default_factory=list, description="C6.10 response rows"
    )
    sector_metrics: List[CDPSectorMetric] = Field(
        default_factory=list, description="Sector-specific metrics"
    )
    methodology_notes: str = Field("", description="Methodology notes")
    change_explanation: str = Field(
        "", description="Overall explanation for changes"
    )
    verification_status: str = Field(
        "", description="Third-party verification status"
    )

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class CDPIntensityDisclosure:
    """
    CDP C6.10 intensity disclosure template.

    Renders CDP-compliant intensity disclosures with C6.10 data table,
    sector-specific module metrics, methodology notes, and change
    explanations. All outputs include SHA-256 provenance hashing.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = CDPIntensityDisclosure()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CDPIntensityDisclosure."""
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
        """Render CDP disclosure as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render CDP disclosure as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render CDP disclosure as JSON dict."""
        start = time.monotonic()
        self.generated_at = utcnow()
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
            self._md_c610_table(data),
            self._md_sector_metrics(data),
            self._md_methodology(data),
            self._md_change_explanation(data),
            self._md_verification(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# CDP C6.10 Intensity Disclosure - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_c610_table(self, data: Dict[str, Any]) -> str:
        """Render Markdown C6.10 data table."""
        rows = data.get("c6_10_data", [])
        if not rows:
            return "## C6.10 Intensity Metrics\n\nNo C6.10 data available."
        lines = [
            "## C6.10 Intensity Metrics",
            "",
        ]
        for i, r in enumerate(rows, 1):
            figure = r.get("intensity_figure", 0)
            numerator = r.get("metric_numerator", "")
            denominator = r.get("metric_denominator", "")
            denom_unit = r.get("metric_denominator_unit", "")
            pct_change = r.get("pct_change_from_previous_year")
            direction = ChangeDirection(r.get("direction_of_change", "no_change"))
            reason = r.get("reason_for_change", "")
            scope = r.get("scope_coverage", "")
            pct_str = f"{pct_change:.1f}" if pct_change is not None else "N/A"
            lines.append(f"### Metric {i}")
            lines.append("")
            lines.append("| Field | Value |")
            lines.append("|-------|-------|")
            lines.append(f"| **Intensity figure** | {figure:,.6f} |")
            lines.append(f"| Metric numerator | {numerator} |")
            lines.append(f"| Metric denominator | {denominator} ({denom_unit}) |")
            lines.append(f"| Scope | {scope} |")
            lines.append(f"| % change from previous year | {pct_str} |")
            lines.append(f"| Direction of change | {direction.value.replace('_', ' ').title()} |")
            lines.append(f"| Reason for change | {reason} |")
            lines.append("")
        return "\n".join(lines)

    def _md_sector_metrics(self, data: Dict[str, Any]) -> str:
        """Render Markdown sector-specific metrics."""
        metrics = data.get("sector_metrics", [])
        if not metrics:
            return ""
        lines = [
            "## Sector-Specific Intensity Metrics",
            "",
            "| Module | Question | Metric | Intensity | Unit | Prior Year | Change |",
            "|--------|----------|--------|-----------|------|------------|--------|",
        ]
        for m in metrics:
            module = m.get("sector_module", "")
            question = m.get("question_number", "")
            name = m.get("metric_name", "")
            intensity = m.get("intensity_value", 0)
            unit = m.get("intensity_unit", "")
            prior = m.get("prior_year_value")
            change = m.get("change_pct")
            prior_str = f"{prior:,.4f}" if prior is not None else "-"
            change_str = f"{change:+.1f}%" if change is not None else "-"
            lines.append(
                f"| {module} | {question} | {name} | {intensity:,.4f} | "
                f"{unit} | {prior_str} | {change_str} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology notes."""
        notes = self._get_val(data, "methodology_notes", "")
        if not notes:
            return ""
        return f"## Methodology Notes\n\n{notes}"

    def _md_change_explanation(self, data: Dict[str, Any]) -> str:
        """Render Markdown change explanation."""
        explanation = self._get_val(data, "change_explanation", "")
        if not explanation:
            return ""
        return f"## Change Explanation\n\n{explanation}"

    def _md_verification(self, data: Dict[str, Any]) -> str:
        """Render Markdown verification status."""
        status = self._get_val(data, "verification_status", "")
        if not status:
            return ""
        return f"## Verification Status\n\n{status}"

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
            self._html_c610_table(data),
            self._html_sector_metrics(data),
            self._html_methodology(data),
            self._html_change_explanation(data),
            self._html_verification(data),
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
            f"<title>CDP C6.10 Disclosure - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #006837;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#264653;margin-top:1.5rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#e8f5e9;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".cdp-card{background:#f0f9f4;border:1px solid #006837;border-radius:8px;"
            "padding:1rem 1.5rem;margin:1rem 0;}\n"
            ".intensity-value{font-size:1.5rem;font-weight:700;color:#006837;}\n"
            ".dir-increased{color:#e76f51;}\n"
            ".dir-decreased{color:#2a9d8f;}\n"
            ".dir-no-change{color:#888;}\n"
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
            f"<h1>CDP C6.10: Intensity Disclosure &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n<hr>\n</div>"
        )

    def _html_c610_table(self, data: Dict[str, Any]) -> str:
        """Render HTML C6.10 data as metric cards."""
        rows = data.get("c6_10_data", [])
        if not rows:
            return ""
        cards = ""
        for i, r in enumerate(rows, 1):
            figure = r.get("intensity_figure", 0)
            numerator = r.get("metric_numerator", "")
            denominator = r.get("metric_denominator", "")
            denom_unit = r.get("metric_denominator_unit", "")
            pct_change = r.get("pct_change_from_previous_year")
            direction = ChangeDirection(r.get("direction_of_change", "no_change"))
            reason = r.get("reason_for_change", "")
            scope = r.get("scope_coverage", "")
            dir_css = f"dir-{direction.value}"
            dir_label = direction.value.replace("_", " ").title()
            pct_str = f"{pct_change:.1f}%" if pct_change is not None else "N/A"
            cards += (
                f'<div class="cdp-card">\n'
                f"<h3>C6.10 Metric {i}</h3>\n"
                f'<div class="intensity-value">{figure:,.6f}</div>\n'
                f"<p><strong>Numerator:</strong> {numerator}</p>\n"
                f"<p><strong>Denominator:</strong> {denominator} ({denom_unit})</p>\n"
                f"<p><strong>Scope:</strong> {scope}</p>\n"
                f'<p><strong>Change:</strong> {pct_str} '
                f'<span class="{dir_css}">({dir_label})</span></p>\n'
                f"<p><strong>Reason:</strong> {reason}</p>\n"
                "</div>\n"
            )
        return f'<div class="section">\n<h2>C6.10 Intensity Metrics</h2>\n{cards}</div>'

    def _html_sector_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML sector-specific metrics table."""
        metrics = data.get("sector_metrics", [])
        if not metrics:
            return ""
        rows = ""
        for m in metrics:
            module = m.get("sector_module", "")
            question = m.get("question_number", "")
            name = m.get("metric_name", "")
            intensity = m.get("intensity_value", 0)
            unit = m.get("intensity_unit", "")
            rows += (
                f"<tr><td>{module}</td><td>{question}</td><td>{name}</td>"
                f"<td>{intensity:,.4f}</td><td>{unit}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>Sector-Specific Metrics</h2>\n'
            "<table><thead><tr><th>Module</th><th>Question</th>"
            "<th>Metric</th><th>Intensity</th><th>Unit</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology notes."""
        notes = self._get_val(data, "methodology_notes", "")
        if not notes:
            return ""
        return (
            '<div class="section">\n<h2>Methodology Notes</h2>\n'
            f"<p>{notes}</p>\n</div>"
        )

    def _html_change_explanation(self, data: Dict[str, Any]) -> str:
        """Render HTML change explanation."""
        explanation = self._get_val(data, "change_explanation", "")
        if not explanation:
            return ""
        return (
            '<div class="section">\n<h2>Change Explanation</h2>\n'
            f"<p>{explanation}</p>\n</div>"
        )

    def _html_verification(self, data: Dict[str, Any]) -> str:
        """Render HTML verification status."""
        status = self._get_val(data, "verification_status", "")
        if not status:
            return ""
        return (
            '<div class="section">\n<h2>Verification Status</h2>\n'
            f"<p>{status}</p>\n</div>"
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
        """Render CDP disclosure as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "cdp_intensity_disclosure",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year", ""),
            "cdp_submission_year": data.get("cdp_submission_year"),
            "c6_10_data": data.get("c6_10_data", []),
            "sector_metrics": data.get("sector_metrics", []),
            "methodology_notes": self._get_val(data, "methodology_notes", ""),
            "change_explanation": self._get_val(data, "change_explanation", ""),
            "verification_status": self._get_val(data, "verification_status", ""),
        }
