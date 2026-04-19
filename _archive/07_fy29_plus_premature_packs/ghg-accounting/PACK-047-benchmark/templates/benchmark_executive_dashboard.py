# -*- coding: utf-8 -*-
"""
BenchmarkExecutiveDashboard - Executive Dashboard for PACK-047.

Generates a concise executive dashboard for GHG emissions benchmarking
with key KPI cards (overall percentile rank, Implied Temperature Rise,
Compound Annual Reduction Rate, pathway alignment score, transition risk
score), traffic light indicators, sparkline data for 5-year trajectory,
peer group summary, top 3 strengths/improvement areas, and export in
multiple formats.

Regulatory References:
    - GHG Protocol Corporate Standard (Chapter 9: Setting a GHG Target)
    - TCFD Recommendations: Metrics and Targets
    - EU Benchmark Regulation (BMR) 2019/2089 - Climate Benchmarks
    - Paris Agreement Article 2.1(a) - 1.5C pathway alignment
    - SBTi Corporate Net-Zero Standard v1.0

Sections:
    1. Key KPI Cards (percentile, ITR, CARR, alignment, transition risk)
    2. Traffic Light Summary
    3. Sparkline Trajectory (5-year)
    4. Peer Group Summary
    5. Top 3 Strengths
    6. Top 3 Improvement Areas
    7. Provenance Footer

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

class TrafficLight(str, Enum):
    """Traffic light status indicators."""
    GREEN = "green"
    AMBER = "amber"
    RED = "red"

class ChangeDirection(str, Enum):
    """Direction of year-over-year change."""
    UP = "up"
    DOWN = "down"
    FLAT = "flat"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class SparklinePoint(BaseModel):
    """Single data point for sparkline visualisation."""
    year: int = Field(..., description="Reporting year")
    value: float = Field(..., description="Emissions or intensity value for the year")

class KPICard(BaseModel):
    """Single KPI card for the executive dashboard."""
    kpi_name: str = Field(..., description="KPI label (e.g., Overall Percentile Rank)")
    value: float = Field(..., description="KPI numeric value")
    unit: str = Field("", description="KPI unit (e.g., percentile, degC, %/yr)")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light status")
    description: str = Field("", description="Short description of the KPI")
    threshold_green: Optional[float] = Field(None, description="Green threshold")
    threshold_amber: Optional[float] = Field(None, description="Amber threshold")
    threshold_red: Optional[float] = Field(None, description="Red threshold")

class PercentileRankKPI(BaseModel):
    """Overall percentile rank KPI."""
    percentile: float = Field(50.0, ge=0, le=100, description="Percentile rank (0=worst, 100=best)")
    peer_group: str = Field("", description="Peer group used for ranking")
    rank_position: int = Field(0, ge=0, description="Absolute rank (1=best)")
    total_peers: int = Field(0, ge=0, description="Total peers in ranking")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light status")

class ITRKPI(BaseModel):
    """Implied Temperature Rise KPI."""
    itr_value: float = Field(2.0, description="ITR in degrees Celsius")
    target_pathway: str = Field("1.5C", description="Target pathway for comparison")
    methodology: str = Field("", description="ITR methodology (e.g., SBTi, MSCI)")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light status")

class CARRKPI(BaseModel):
    """Compound Annual Reduction Rate KPI."""
    carr_pct: float = Field(0.0, description="CARR as annual % reduction")
    required_carr_pct: float = Field(0.0, description="Required CARR for pathway alignment")
    period_start: int = Field(0, description="Start year of CARR calculation")
    period_end: int = Field(0, description="End year of CARR calculation")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light status")

class PathwayAlignmentKPI(BaseModel):
    """Pathway alignment score KPI."""
    alignment_score: float = Field(0.0, ge=0, le=100, description="Alignment score (0-100)")
    pathway_name: str = Field("", description="Reference pathway name")
    gap_pct: float = Field(0.0, description="Gap to aligned trajectory (%)")
    convergence_year: Optional[int] = Field(None, description="Estimated convergence year")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light status")

class TransitionRiskKPI(BaseModel):
    """Transition risk score KPI."""
    risk_score: float = Field(50.0, ge=0, le=100, description="Composite risk score (0-100)")
    risk_category: str = Field("Medium", description="Risk category label")
    carbon_price_exposure: Optional[float] = Field(None, description="Carbon price exposure (EUR)")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light status")

class PeerGroupSummary(BaseModel):
    """Summary of the peer group used."""
    peer_group_name: str = Field("", description="Peer group name")
    peer_count: int = Field(0, ge=0, description="Number of peers")
    sector: str = Field("", description="Industry sector")
    region: str = Field("", description="Geographic region")
    median_emissions: Optional[float] = Field(None, description="Median emissions (tCO2e)")
    best_in_class_emissions: Optional[float] = Field(None, description="Best-in-class emissions")
    data_year: Optional[int] = Field(None, description="Year of peer data")

class StrengthItem(BaseModel):
    """Single strength or improvement area."""
    rank: int = Field(1, ge=1, le=10, description="Rank (1=top)")
    area: str = Field(..., description="Area name")
    detail: str = Field("", description="Detail description")
    metric_value: Optional[float] = Field(None, description="Associated metric value")
    peer_comparison: str = Field("", description="Comparison to peers")

class DashboardInput(BaseModel):
    """Complete input model for BenchmarkExecutiveDashboard."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period (e.g., FY2025)")
    report_date: Optional[str] = Field(None, description="Report date (ISO format)")
    percentile_rank: Optional[PercentileRankKPI] = Field(
        None, description="Overall percentile rank KPI"
    )
    itr: Optional[ITRKPI] = Field(None, description="Implied Temperature Rise KPI")
    carr: Optional[CARRKPI] = Field(None, description="Compound Annual Reduction Rate KPI")
    pathway_alignment: Optional[PathwayAlignmentKPI] = Field(
        None, description="Pathway alignment score KPI"
    )
    transition_risk: Optional[TransitionRiskKPI] = Field(
        None, description="Transition risk score KPI"
    )
    sparkline_data: List[SparklinePoint] = Field(
        default_factory=list, description="5-year sparkline trajectory data"
    )
    peer_group_summary: Optional[PeerGroupSummary] = Field(
        None, description="Peer group summary"
    )
    top_strengths: List[StrengthItem] = Field(
        default_factory=list, description="Top 3 strengths"
    )
    top_improvements: List[StrengthItem] = Field(
        default_factory=list, description="Top 3 improvement areas"
    )
    additional_kpis: List[KPICard] = Field(
        default_factory=list, description="Additional KPI cards"
    )

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tl_label(status: TrafficLight) -> str:
    """Return uppercase label for traffic light."""
    return status.value.upper()

def _tl_css(status: TrafficLight) -> str:
    """Return CSS class for traffic light status."""
    mapping = {
        TrafficLight.GREEN: "tl-green",
        TrafficLight.AMBER: "tl-amber",
        TrafficLight.RED: "tl-red",
    }
    return mapping.get(status, "tl-amber")

def _tl_color(status: TrafficLight) -> str:
    """Return hex colour for traffic light status."""
    mapping = {
        TrafficLight.GREEN: "#2a9d8f",
        TrafficLight.AMBER: "#e9c46a",
        TrafficLight.RED: "#e76f51",
    }
    return mapping.get(status, "#e9c46a")

def _format_decimal(value: Optional[float], places: int = 2) -> str:
    """Format a float with specified decimal places, or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:,.{places}f}"

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class BenchmarkExecutiveDashboard:
    """
    Executive dashboard template for GHG emissions benchmarking.

    Renders a concise dashboard with five key KPI cards (overall percentile
    rank, ITR, CARR, pathway alignment score, transition risk score), traffic
    light indicators based on configurable thresholds, sparkline trajectory
    data, peer group summary, and top strengths/improvement areas. All outputs
    include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = BenchmarkExecutiveDashboard()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BenchmarkExecutiveDashboard."""
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
        """
        Render the dashboard in the specified format.

        Args:
            data: Dashboard data dict (or DashboardInput.dict()).
            fmt: Output format ('markdown', 'html', 'json').

        Returns:
            Rendered content.
        """
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        raise ValueError(f"Unsupported format: {fmt}")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render executive dashboard as Markdown.

        Args:
            data: Dashboard data dict.

        Returns:
            Markdown string with provenance hash.
        """
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render executive dashboard as HTML.

        Args:
            data: Dashboard data dict.

        Returns:
            Self-contained HTML string.
        """
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render executive dashboard as JSON-serializable dict.

        Args:
            data: Dashboard data dict.

        Returns:
            JSON-serializable dict with provenance hash.
        """
        start = time.monotonic()
        self.generated_at = utcnow()
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
            self._md_kpi_cards(data),
            self._md_sparkline(data),
            self._md_peer_summary(data),
            self._md_strengths(data),
            self._md_improvements(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# GHG Benchmark Executive Dashboard - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Render Markdown key KPI summary table."""
        lines = [
            "## Key Performance Indicators",
            "",
            "| KPI | Value | Unit | Status |",
            "|-----|-------|------|--------|",
        ]
        # Percentile rank
        pr = data.get("percentile_rank")
        if pr:
            status = TrafficLight(pr.get("status", "amber"))
            rank_info = ""
            if pr.get("rank_position") and pr.get("total_peers"):
                rank_info = f" (#{pr['rank_position']} of {pr['total_peers']})"
            lines.append(
                f"| Overall Percentile Rank | P{pr.get('percentile', 0):.0f}{rank_info} | "
                f"percentile | **{_tl_label(status)}** |"
            )
        # ITR
        itr = data.get("itr")
        if itr:
            status = TrafficLight(itr.get("status", "amber"))
            lines.append(
                f"| Implied Temperature Rise | {itr.get('itr_value', 0):.1f} | "
                f"degC | **{_tl_label(status)}** |"
            )
        # CARR
        carr = data.get("carr")
        if carr:
            status = TrafficLight(carr.get("status", "amber"))
            required = carr.get("required_carr_pct", 0)
            lines.append(
                f"| Compound Annual Reduction Rate | {carr.get('carr_pct', 0):+.1f} | "
                f"%/yr (req: {required:+.1f}) | **{_tl_label(status)}** |"
            )
        # Pathway alignment
        pa = data.get("pathway_alignment")
        if pa:
            status = TrafficLight(pa.get("status", "amber"))
            pathway = pa.get("pathway_name", "")
            lines.append(
                f"| Pathway Alignment Score ({pathway}) | {pa.get('alignment_score', 0):.0f} | "
                f"/ 100 | **{_tl_label(status)}** |"
            )
        # Transition risk
        tr = data.get("transition_risk")
        if tr:
            status = TrafficLight(tr.get("status", "amber"))
            cat = tr.get("risk_category", "")
            lines.append(
                f"| Transition Risk Score | {tr.get('risk_score', 0):.0f} ({cat}) | "
                f"/ 100 | **{_tl_label(status)}** |"
            )
        # Additional KPIs
        for kpi in data.get("additional_kpis", []):
            status = TrafficLight(kpi.get("status", "amber"))
            lines.append(
                f"| {kpi.get('kpi_name', '')} | {kpi.get('value', 0):,.2f} | "
                f"{kpi.get('unit', '')} | **{_tl_label(status)}** |"
            )
        return "\n".join(lines)

    def _md_sparkline(self, data: Dict[str, Any]) -> str:
        """Render Markdown 5-year trajectory table."""
        points = data.get("sparkline_data", [])
        if not points:
            return ""
        lines = [
            "## 5-Year Emissions Trajectory",
            "",
            "| Year | Value |",
            "|------|-------|",
        ]
        for p in points:
            lines.append(f"| {p.get('year', '')} | {p.get('value', 0):,.2f} |")
        return "\n".join(lines)

    def _md_peer_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown peer group summary."""
        pg = data.get("peer_group_summary")
        if not pg:
            return ""
        lines = [
            "## Peer Group Summary",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Peer Group | {pg.get('peer_group_name', '')} |",
            f"| Sector | {pg.get('sector', '')} |",
            f"| Region | {pg.get('region', '')} |",
            f"| Peer Count | {pg.get('peer_count', 0)} |",
        ]
        median = pg.get("median_emissions")
        if median is not None:
            lines.append(f"| Median Emissions | {median:,.1f} tCO2e |")
        best = pg.get("best_in_class_emissions")
        if best is not None:
            lines.append(f"| Best-in-Class | {best:,.1f} tCO2e |")
        data_year = pg.get("data_year")
        if data_year:
            lines.append(f"| Data Year | {data_year} |")
        return "\n".join(lines)

    def _md_strengths(self, data: Dict[str, Any]) -> str:
        """Render Markdown top 3 strengths."""
        items = data.get("top_strengths", [])
        if not items:
            return ""
        lines = ["## Top Strengths", ""]
        for item in items[:3]:
            rank = item.get("rank", 1)
            area = item.get("area", "")
            detail = item.get("detail", "")
            comparison = item.get("peer_comparison", "")
            line = f"**{rank}.** {area}"
            if detail:
                line += f" - {detail}"
            if comparison:
                line += f" *(Peers: {comparison})*"
            lines.append(line)
            lines.append("")
        return "\n".join(lines)

    def _md_improvements(self, data: Dict[str, Any]) -> str:
        """Render Markdown top 3 improvement areas."""
        items = data.get("top_improvements", [])
        if not items:
            return ""
        lines = ["## Top Improvement Areas", ""]
        for item in items[:3]:
            rank = item.get("rank", 1)
            area = item.get("area", "")
            detail = item.get("detail", "")
            comparison = item.get("peer_comparison", "")
            line = f"**{rank}.** {area}"
            if detail:
                line += f" - {detail}"
            if comparison:
                line += f" *(Peers: {comparison})*"
            lines.append(line)
            lines.append("")
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
            self._html_kpi_cards(data),
            self._html_sparkline(data),
            self._html_peer_summary(data),
            self._html_strengths(data),
            self._html_improvements(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>GHG Benchmark Dashboard - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".tl-green{color:#2a9d8f;font-weight:700;}\n"
            ".tl-amber{color:#e9c46a;font-weight:700;}\n"
            ".tl-red{color:#e76f51;font-weight:700;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".kpi-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:200px;"
            "border-left:4px solid #2a9d8f;vertical-align:top;}\n"
            ".kpi-value{font-size:1.6rem;font-weight:700;color:#1b263b;}\n"
            ".kpi-label{font-size:0.85rem;color:#555;margin-top:0.3rem;}\n"
            ".kpi-unit{font-size:0.75rem;color:#888;}\n"
            ".strength-item{background:#e8f5e9;border-left:3px solid #2a9d8f;"
            "padding:0.6rem 1rem;margin:0.5rem 0;border-radius:4px;}\n"
            ".improvement-item{background:#fff3e0;border-left:3px solid #e9c46a;"
            "padding:0.6rem 1rem;margin:0.5rem 0;border-radius:4px;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header section."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            '<div class="section">\n'
            f"<h1>GHG Benchmark Executive Dashboard &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Render HTML key KPI cards."""
        cards = ""
        # Percentile rank card
        pr = data.get("percentile_rank")
        if pr:
            status = TrafficLight(pr.get("status", "amber"))
            color = _tl_color(status)
            rank_info = ""
            if pr.get("rank_position") and pr.get("total_peers"):
                rank_info = f"#{pr['rank_position']} of {pr['total_peers']}"
            cards += (
                f'<div class="kpi-card" style="border-left-color:{color};">'
                f'<div class="kpi-value">P{pr.get("percentile", 0):.0f}</div>'
                f'<div class="kpi-label">Overall Percentile Rank</div>'
                f'<div class="kpi-unit">{rank_info}</div></div>\n'
            )
        # ITR card
        itr = data.get("itr")
        if itr:
            status = TrafficLight(itr.get("status", "amber"))
            color = _tl_color(status)
            cards += (
                f'<div class="kpi-card" style="border-left-color:{color};">'
                f'<div class="kpi-value">{itr.get("itr_value", 0):.1f}&deg;C</div>'
                f'<div class="kpi-label">Implied Temperature Rise</div>'
                f'<div class="kpi-unit">Target: {itr.get("target_pathway", "1.5C")}</div></div>\n'
            )
        # CARR card
        carr = data.get("carr")
        if carr:
            status = TrafficLight(carr.get("status", "amber"))
            color = _tl_color(status)
            cards += (
                f'<div class="kpi-card" style="border-left-color:{color};">'
                f'<div class="kpi-value">{carr.get("carr_pct", 0):+.1f}%/yr</div>'
                f'<div class="kpi-label">Annual Reduction Rate</div>'
                f'<div class="kpi-unit">Required: {carr.get("required_carr_pct", 0):+.1f}%/yr</div></div>\n'
            )
        # Pathway alignment card
        pa = data.get("pathway_alignment")
        if pa:
            status = TrafficLight(pa.get("status", "amber"))
            color = _tl_color(status)
            cards += (
                f'<div class="kpi-card" style="border-left-color:{color};">'
                f'<div class="kpi-value">{pa.get("alignment_score", 0):.0f}/100</div>'
                f'<div class="kpi-label">Pathway Alignment</div>'
                f'<div class="kpi-unit">{pa.get("pathway_name", "")}</div></div>\n'
            )
        # Transition risk card
        tr = data.get("transition_risk")
        if tr:
            status = TrafficLight(tr.get("status", "amber"))
            color = _tl_color(status)
            cards += (
                f'<div class="kpi-card" style="border-left-color:{color};">'
                f'<div class="kpi-value">{tr.get("risk_score", 0):.0f}</div>'
                f'<div class="kpi-label">Transition Risk ({tr.get("risk_category", "")})</div>'
                f'<div class="kpi-unit">Score / 100</div></div>\n'
            )
        if not cards:
            return ""
        return f'<div class="section">\n<h2>Key Performance Indicators</h2>\n<div>{cards}</div>\n</div>'

    def _html_sparkline(self, data: Dict[str, Any]) -> str:
        """Render HTML sparkline trajectory table."""
        points = data.get("sparkline_data", [])
        if not points:
            return ""
        rows = ""
        for p in points:
            rows += f"<tr><td>{p.get('year', '')}</td><td>{p.get('value', 0):,.2f}</td></tr>\n"
        return (
            '<div class="section">\n<h2>5-Year Emissions Trajectory</h2>\n'
            "<table><thead><tr><th>Year</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_peer_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML peer group summary."""
        pg = data.get("peer_group_summary")
        if not pg:
            return ""
        rows = (
            f"<tr><td>Peer Group</td><td>{pg.get('peer_group_name', '')}</td></tr>\n"
            f"<tr><td>Sector</td><td>{pg.get('sector', '')}</td></tr>\n"
            f"<tr><td>Region</td><td>{pg.get('region', '')}</td></tr>\n"
            f"<tr><td>Peer Count</td><td>{pg.get('peer_count', 0)}</td></tr>\n"
        )
        median = pg.get("median_emissions")
        if median is not None:
            rows += f"<tr><td>Median Emissions</td><td>{median:,.1f} tCO2e</td></tr>\n"
        best = pg.get("best_in_class_emissions")
        if best is not None:
            rows += f"<tr><td>Best-in-Class</td><td>{best:,.1f} tCO2e</td></tr>\n"
        return (
            '<div class="section">\n<h2>Peer Group Summary</h2>\n'
            "<table><thead><tr><th>Attribute</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_strengths(self, data: Dict[str, Any]) -> str:
        """Render HTML top strengths."""
        items = data.get("top_strengths", [])
        if not items:
            return ""
        content = ""
        for item in items[:3]:
            rank = item.get("rank", 1)
            area = item.get("area", "")
            detail = item.get("detail", "")
            comparison = item.get("peer_comparison", "")
            extra = ""
            if detail:
                extra += f" &mdash; {detail}"
            if comparison:
                extra += f" <em>(Peers: {comparison})</em>"
            content += (
                f'<div class="strength-item"><strong>{rank}.</strong> {area}{extra}</div>\n'
            )
        return f'<div class="section">\n<h2>Top Strengths</h2>\n{content}</div>'

    def _html_improvements(self, data: Dict[str, Any]) -> str:
        """Render HTML top improvement areas."""
        items = data.get("top_improvements", [])
        if not items:
            return ""
        content = ""
        for item in items[:3]:
            rank = item.get("rank", 1)
            area = item.get("area", "")
            detail = item.get("detail", "")
            comparison = item.get("peer_comparison", "")
            extra = ""
            if detail:
                extra += f" &mdash; {detail}"
            if comparison:
                extra += f" <em>(Peers: {comparison})</em>"
            content += (
                f'<div class="improvement-item"><strong>{rank}.</strong> {area}{extra}</div>\n'
            )
        return f'<div class="section">\n<h2>Top Improvement Areas</h2>\n{content}</div>'

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
        """Render dashboard as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "benchmark_executive_dashboard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "kpi_summary": {
                "percentile_rank": data.get("percentile_rank"),
                "itr": data.get("itr"),
                "carr": data.get("carr"),
                "pathway_alignment": data.get("pathway_alignment"),
                "transition_risk": data.get("transition_risk"),
                "additional_kpis": data.get("additional_kpis", []),
            },
            "sparkline_data": data.get("sparkline_data", []),
            "peer_group_summary": data.get("peer_group_summary"),
            "top_strengths": data.get("top_strengths", []),
            "top_improvements": data.get("top_improvements", []),
            "chart_data": {
                "sparkline": self._build_sparkline_chart(data),
                "kpi_gauge": self._build_gauge_data(data),
            },
        }

    def _build_sparkline_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build sparkline chart data."""
        points = data.get("sparkline_data", [])
        if not points:
            return {}
        return {
            "years": [p.get("year", 0) for p in points],
            "values": [p.get("value", 0) for p in points],
        }

    def _build_gauge_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build gauge chart data for KPIs."""
        gauges: List[Dict[str, Any]] = []
        pr = data.get("percentile_rank")
        if pr:
            gauges.append({
                "label": "Percentile Rank",
                "value": pr.get("percentile", 0),
                "min": 0,
                "max": 100,
                "status": pr.get("status", "amber"),
            })
        itr = data.get("itr")
        if itr:
            gauges.append({
                "label": "ITR",
                "value": itr.get("itr_value", 0),
                "min": 1.0,
                "max": 4.0,
                "status": itr.get("status", "amber"),
            })
        pa = data.get("pathway_alignment")
        if pa:
            gauges.append({
                "label": "Pathway Alignment",
                "value": pa.get("alignment_score", 0),
                "min": 0,
                "max": 100,
                "status": pa.get("status", "amber"),
            })
        tr = data.get("transition_risk")
        if tr:
            gauges.append({
                "label": "Transition Risk",
                "value": tr.get("risk_score", 0),
                "min": 0,
                "max": 100,
                "status": tr.get("status", "amber"),
            })
        return gauges
