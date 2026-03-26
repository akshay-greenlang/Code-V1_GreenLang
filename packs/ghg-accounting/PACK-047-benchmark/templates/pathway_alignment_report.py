# -*- coding: utf-8 -*-
"""
PathwayAlignmentReport - Pathway Alignment Report for PACK-047.

Generates a pathway alignment report with multi-pathway graph data
(organisation trajectory vs 6+ reference pathways), gap-to-pathway
tables for each pathway and year, convergence year estimates, alignment
score summary, and methodology disclosure.

Regulatory References:
    - SBTi Corporate Net-Zero Standard v1.0: Pathway alignment
    - IEA Net Zero by 2050 Roadmap
    - IPCC AR6 1.5C pathways (SSP1-1.9, SSP1-2.6)
    - EU Climate Law (Regulation 2021/1119): -55% by 2030
    - Paris Agreement Article 2.1(a)
    - TPI (Transition Pathway Initiative) sector pathways

Sections:
    1. Alignment Score Summary
    2. Multi-Pathway Graph Data
    3. Gap-to-Pathway Table
    4. Convergence Year Estimates
    5. Methodology Disclosure
    6. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 47.0.0
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


class TrafficLight(str, Enum):
    """Traffic light status indicators."""
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class AlignmentStatus(str, Enum):
    """Pathway alignment status."""
    ALIGNED = "aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    NOT_ALIGNED = "not_aligned"
    INSUFFICIENT_DATA = "insufficient_data"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class PathwayPoint(BaseModel):
    """Single year-value point on a pathway."""
    year: int = Field(..., description="Year")
    value: float = Field(..., description="Emissions or intensity value")


class PathwayDefinition(BaseModel):
    """Reference pathway definition."""
    pathway_id: str = Field(..., description="Unique pathway identifier")
    pathway_name: str = Field(..., description="Pathway display name")
    source: str = Field("", description="Source organisation (e.g., IEA, SBTi)")
    scenario: str = Field("", description="Scenario name (e.g., NZE, 1.5C)")
    base_year: int = Field(2020, description="Base year of the pathway")
    target_year: int = Field(2050, description="Target year of the pathway")
    data_points: List[PathwayPoint] = Field(
        default_factory=list, description="Pathway trajectory data points"
    )
    description: str = Field("", description="Pathway description")


class OrgTrajectoryPoint(BaseModel):
    """Single year-value point for organisation trajectory."""
    year: int = Field(..., description="Year")
    value: float = Field(..., description="Actual or projected emissions")
    is_actual: bool = Field(True, description="True if actual data, False if projected")


class GapToPathwayRow(BaseModel):
    """Gap to a specific pathway in a specific year."""
    pathway_id: str = Field(..., description="Pathway identifier")
    pathway_name: str = Field("", description="Pathway display name")
    year: int = Field(..., description="Year")
    pathway_value: float = Field(0.0, description="Pathway value at this year")
    org_value: float = Field(0.0, description="Organisation value at this year")
    absolute_gap: float = Field(0.0, description="Absolute gap (org - pathway)")
    relative_gap_pct: float = Field(0.0, description="Relative gap percentage")


class ConvergenceEstimate(BaseModel):
    """Convergence year estimate for a pathway."""
    pathway_id: str = Field(..., description="Pathway identifier")
    pathway_name: str = Field("", description="Pathway display name")
    convergence_year: Optional[int] = Field(
        None, description="Estimated year of convergence (None if never)"
    )
    current_gap_pct: float = Field(0.0, description="Current gap to pathway (%)")
    converging: bool = Field(False, description="Whether gap is narrowing")
    methodology_note: str = Field("", description="Estimation methodology note")


class AlignmentScoreSummary(BaseModel):
    """Overall alignment score summary."""
    overall_score: float = Field(0.0, ge=0, le=100, description="Overall alignment (0-100)")
    primary_pathway: str = Field("", description="Primary reference pathway used")
    status: AlignmentStatus = Field(
        AlignmentStatus.NOT_ALIGNED, description="Alignment status"
    )
    traffic_light: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light")
    score_methodology: str = Field("", description="Scoring methodology description")


class MethodologyDisclosure(BaseModel):
    """Methodology disclosure for pathway alignment."""
    pathways_used: List[str] = Field(default_factory=list, description="Pathway names used")
    interpolation_method: str = Field(
        "linear", description="Interpolation method (linear, spline)"
    )
    base_year: int = Field(2020, description="Base year for alignment")
    metric_type: str = Field(
        "absolute_emissions", description="Metric type (absolute, intensity)"
    )
    sector_classification: str = Field("", description="Sector classification used")
    data_sources: List[str] = Field(default_factory=list, description="Data sources")
    limitations: List[str] = Field(default_factory=list, description="Methodology limitations")


class PathwayAlignmentInput(BaseModel):
    """Complete input model for PathwayAlignmentReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    alignment_summary: Optional[AlignmentScoreSummary] = Field(
        None, description="Overall alignment score summary"
    )
    pathways: List[PathwayDefinition] = Field(
        default_factory=list, description="Reference pathway definitions"
    )
    org_trajectory: List[OrgTrajectoryPoint] = Field(
        default_factory=list, description="Organisation trajectory data"
    )
    gap_to_pathway: List[GapToPathwayRow] = Field(
        default_factory=list, description="Gap-to-pathway table rows"
    )
    convergence_estimates: List[ConvergenceEstimate] = Field(
        default_factory=list, description="Convergence year estimates"
    )
    methodology: Optional[MethodologyDisclosure] = Field(
        None, description="Methodology disclosure"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tl_label(status: TrafficLight) -> str:
    """Return uppercase label for traffic light."""
    return status.value.upper()


def _alignment_label(status: AlignmentStatus) -> str:
    """Return human-readable alignment status."""
    mapping = {
        AlignmentStatus.ALIGNED: "Aligned",
        AlignmentStatus.PARTIALLY_ALIGNED: "Partially Aligned",
        AlignmentStatus.NOT_ALIGNED: "Not Aligned",
        AlignmentStatus.INSUFFICIENT_DATA: "Insufficient Data",
    }
    return mapping.get(status, "Unknown")


def _tl_color(status: TrafficLight) -> str:
    """Return hex colour for traffic light."""
    mapping = {
        TrafficLight.GREEN: "#2a9d8f",
        TrafficLight.AMBER: "#e9c46a",
        TrafficLight.RED: "#e76f51",
    }
    return mapping.get(status, "#e9c46a")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class PathwayAlignmentReport:
    """
    Pathway alignment report template for GHG emissions benchmarking.

    Renders multi-pathway alignment analysis with organisation trajectory
    vs 6+ reference pathways, gap-to-pathway tables, convergence year
    estimates, alignment score summary, and methodology disclosure. All
    outputs include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = PathwayAlignmentReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PathwayAlignmentReport."""
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

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Any:
        """Render in specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        raise ValueError(f"Unsupported format: {fmt}")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render pathway alignment as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render pathway alignment as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render pathway alignment as JSON dict."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def to_markdown(self, data: Dict[str, Any]) -> str:
        """Alias for render_markdown."""
        return self.render_markdown(data)

    def to_html(self, data: Dict[str, Any]) -> str:
        """Alias for render_html."""
        return self.render_html(data)

    def to_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for render_json."""
        return self.render_json(data)

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_alignment_summary(data),
            self._md_pathway_overview(data),
            self._md_gap_table(data),
            self._md_convergence(data),
            self._md_methodology(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Pathway Alignment Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_alignment_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown alignment score summary."""
        summary = data.get("alignment_summary")
        if not summary:
            return ""
        score = summary.get("overall_score", 0)
        pathway = summary.get("primary_pathway", "")
        status = AlignmentStatus(summary.get("status", "not_aligned"))
        tl = TrafficLight(summary.get("traffic_light", "amber"))
        methodology = summary.get("score_methodology", "")
        lines = [
            "## 1. Alignment Score Summary",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Overall Score | {score:.0f} / 100 |",
            f"| Primary Pathway | {pathway} |",
            f"| Status | {_alignment_label(status)} |",
            f"| Traffic Light | **{_tl_label(tl)}** |",
        ]
        if methodology:
            lines.append("")
            lines.append(f"**Methodology:** {methodology}")
        return "\n".join(lines)

    def _md_pathway_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown pathway overview table."""
        pathways = data.get("pathways", [])
        if not pathways:
            return ""
        lines = [
            "## 2. Reference Pathways",
            "",
            "| Pathway | Source | Scenario | Base Year | Target Year | Description |",
            "|---------|--------|----------|-----------|-------------|-------------|",
        ]
        for pw in pathways:
            lines.append(
                f"| {pw.get('pathway_name', '')} | {pw.get('source', '')} | "
                f"{pw.get('scenario', '')} | {pw.get('base_year', '')} | "
                f"{pw.get('target_year', '')} | {pw.get('description', '')} |"
            )
        return "\n".join(lines)

    def _md_gap_table(self, data: Dict[str, Any]) -> str:
        """Render Markdown gap-to-pathway table."""
        gaps = data.get("gap_to_pathway", [])
        if not gaps:
            return ""
        lines = [
            "## 3. Gap-to-Pathway Analysis",
            "",
            "| Pathway | Year | Pathway Value | Org Value | Absolute Gap | Relative Gap |",
            "|---------|------|---------------|-----------|--------------|--------------|",
        ]
        for g in gaps:
            abs_gap = g.get("absolute_gap", 0)
            rel_gap = g.get("relative_gap_pct", 0)
            lines.append(
                f"| {g.get('pathway_name', '')} | {g.get('year', '')} | "
                f"{g.get('pathway_value', 0):,.1f} | {g.get('org_value', 0):,.1f} | "
                f"{abs_gap:+,.1f} | {rel_gap:+.1f}% |"
            )
        return "\n".join(lines)

    def _md_convergence(self, data: Dict[str, Any]) -> str:
        """Render Markdown convergence year estimates."""
        estimates = data.get("convergence_estimates", [])
        if not estimates:
            return ""
        lines = [
            "## 4. Convergence Year Estimates",
            "",
            "| Pathway | Convergence Year | Current Gap | Converging | Note |",
            "|---------|-----------------|-------------|------------|------|",
        ]
        for c in estimates:
            conv_year = c.get("convergence_year")
            conv_str = str(conv_year) if conv_year else "Not converging"
            converging = "Yes" if c.get("converging", False) else "No"
            note = c.get("methodology_note", "")
            lines.append(
                f"| {c.get('pathway_name', '')} | {conv_str} | "
                f"{c.get('current_gap_pct', 0):+.1f}% | {converging} | {note} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology disclosure."""
        meth = data.get("methodology")
        if not meth:
            return ""
        lines = ["## 5. Methodology Disclosure", ""]
        interp = meth.get("interpolation_method", "linear")
        metric = meth.get("metric_type", "")
        base = meth.get("base_year", "")
        sector = meth.get("sector_classification", "")
        lines.append(f"**Interpolation Method:** {interp}")
        lines.append(f"**Metric Type:** {metric}")
        lines.append(f"**Base Year:** {base}")
        if sector:
            lines.append(f"**Sector Classification:** {sector}")
        lines.append("")
        pathways_used = meth.get("pathways_used", [])
        if pathways_used:
            lines.append("**Pathways Used:**")
            for p in pathways_used:
                lines.append(f"- {p}")
            lines.append("")
        sources = meth.get("data_sources", [])
        if sources:
            lines.append("**Data Sources:**")
            for s in sources:
                lines.append(f"- {s}")
            lines.append("")
        limitations = meth.get("limitations", [])
        if limitations:
            lines.append("**Limitations:**")
            for lim in limitations:
                lines.append(f"- {lim}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-047 GHG Benchmark v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_alignment_summary(data),
            self._html_pathway_overview(data),
            self._html_gap_table(data),
            self._html_convergence(data),
            self._html_methodology(data),
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
            f"<title>Pathway Alignment - {company}</title>\n"
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
            ".score-card{background:#f0f4f8;border-radius:8px;padding:1.5rem;"
            "text-align:center;margin:1rem 0;border-left:5px solid #2a9d8f;}\n"
            ".score-value{font-size:2rem;font-weight:700;color:#1b263b;}\n"
            ".score-label{font-size:0.9rem;color:#555;}\n"
            ".gap-positive{color:#e76f51;}\n"
            ".gap-negative{color:#2a9d8f;}\n"
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
            f"<h1>Pathway Alignment Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {_utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_alignment_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML alignment score summary."""
        summary = data.get("alignment_summary")
        if not summary:
            return ""
        score = summary.get("overall_score", 0)
        tl = TrafficLight(summary.get("traffic_light", "amber"))
        color = _tl_color(tl)
        status = AlignmentStatus(summary.get("status", "not_aligned"))
        pathway = summary.get("primary_pathway", "")
        return (
            '<div class="section">\n<h2>1. Alignment Score Summary</h2>\n'
            f'<div class="score-card" style="border-left-color:{color};">\n'
            f'<div class="score-value">{score:.0f} / 100</div>\n'
            f'<div class="score-label">{_alignment_label(status)} &mdash; {pathway}</div>\n'
            "</div>\n</div>"
        )

    def _html_pathway_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML pathway overview table."""
        pathways = data.get("pathways", [])
        if not pathways:
            return ""
        rows = ""
        for pw in pathways:
            rows += (
                f"<tr><td>{pw.get('pathway_name', '')}</td>"
                f"<td>{pw.get('source', '')}</td>"
                f"<td>{pw.get('scenario', '')}</td>"
                f"<td>{pw.get('base_year', '')}</td>"
                f"<td>{pw.get('target_year', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Reference Pathways</h2>\n'
            "<table><thead><tr><th>Pathway</th><th>Source</th><th>Scenario</th>"
            "<th>Base Year</th><th>Target Year</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_gap_table(self, data: Dict[str, Any]) -> str:
        """Render HTML gap-to-pathway table."""
        gaps = data.get("gap_to_pathway", [])
        if not gaps:
            return ""
        rows = ""
        for g in gaps:
            rel_gap = g.get("relative_gap_pct", 0)
            css = "gap-positive" if rel_gap > 0 else "gap-negative"
            rows += (
                f"<tr><td>{g.get('pathway_name', '')}</td>"
                f"<td>{g.get('year', '')}</td>"
                f"<td>{g.get('pathway_value', 0):,.1f}</td>"
                f"<td>{g.get('org_value', 0):,.1f}</td>"
                f"<td>{g.get('absolute_gap', 0):+,.1f}</td>"
                f'<td class="{css}">{rel_gap:+.1f}%</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>3. Gap-to-Pathway Analysis</h2>\n'
            "<table><thead><tr><th>Pathway</th><th>Year</th>"
            "<th>Pathway Value</th><th>Org Value</th>"
            "<th>Absolute Gap</th><th>Relative Gap</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_convergence(self, data: Dict[str, Any]) -> str:
        """Render HTML convergence estimates."""
        estimates = data.get("convergence_estimates", [])
        if not estimates:
            return ""
        rows = ""
        for c in estimates:
            conv_year = c.get("convergence_year")
            conv_str = str(conv_year) if conv_year else "Not converging"
            converging = "Yes" if c.get("converging", False) else "No"
            rows += (
                f"<tr><td>{c.get('pathway_name', '')}</td>"
                f"<td>{conv_str}</td>"
                f"<td>{c.get('current_gap_pct', 0):+.1f}%</td>"
                f"<td>{converging}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Convergence Year Estimates</h2>\n'
            "<table><thead><tr><th>Pathway</th><th>Convergence Year</th>"
            "<th>Current Gap</th><th>Converging</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology disclosure."""
        meth = data.get("methodology")
        if not meth:
            return ""
        interp = meth.get("interpolation_method", "linear")
        metric = meth.get("metric_type", "")
        content = (
            f"<p><strong>Interpolation:</strong> {interp} | "
            f"<strong>Metric:</strong> {metric}</p>\n"
        )
        limitations = meth.get("limitations", [])
        if limitations:
            content += "<p><strong>Limitations:</strong></p><ul>"
            for lim in limitations:
                content += f"<li>{lim}</li>"
            content += "</ul>\n"
        return (
            '<div class="section">\n<h2>5. Methodology Disclosure</h2>\n'
            f"{content}</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-047 GHG Benchmark v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render pathway alignment as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "pathway_alignment_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "alignment_summary": data.get("alignment_summary"),
            "pathways": data.get("pathways", []),
            "org_trajectory": data.get("org_trajectory", []),
            "gap_to_pathway": data.get("gap_to_pathway", []),
            "convergence_estimates": data.get("convergence_estimates", []),
            "methodology": data.get("methodology"),
            "chart_data": {
                "multi_pathway": self._build_multi_pathway_chart(data),
                "convergence_timeline": self._build_convergence_chart(data),
            },
        }

    def _build_multi_pathway_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build multi-pathway chart data."""
        result: Dict[str, Any] = {"series": []}
        # Organisation trajectory
        org_traj = data.get("org_trajectory", [])
        if org_traj:
            result["series"].append({
                "name": "Organisation",
                "type": "actual",
                "points": [
                    {"year": p.get("year", 0), "value": p.get("value", 0)}
                    for p in org_traj
                ],
            })
        # Pathways
        for pw in data.get("pathways", []):
            result["series"].append({
                "name": pw.get("pathway_name", ""),
                "type": "pathway",
                "source": pw.get("source", ""),
                "points": [
                    {"year": p.get("year", 0), "value": p.get("value", 0)}
                    for p in pw.get("data_points", [])
                ],
            })
        return result

    def _build_convergence_chart(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build convergence timeline chart data."""
        return [
            {
                "pathway": c.get("pathway_name", ""),
                "convergence_year": c.get("convergence_year"),
                "current_gap_pct": c.get("current_gap_pct", 0),
                "converging": c.get("converging", False),
            }
            for c in data.get("convergence_estimates", [])
        ]
