# -*- coding: utf-8 -*-
"""
TrajectoryAnalysisReport - Trajectory Analysis Report for PACK-047.

Generates a trajectory analysis report with CARR ranking table (org vs
peers sorted by reduction rate), convergence trend charts (gap to median
over time), acceleration/deceleration indicators, structural break
annotations, and fan chart data (peer distribution envelope).

Regulatory References:
    - GHG Protocol Corporate Standard (Chapter 9: Setting a GHG Target)
    - SBTi Corporate Net-Zero Standard: Reduction trajectories
    - TCFD Metrics and Targets: Forward-looking trajectories
    - EU Benchmark Regulation 2019/2089: Climate benchmarks

Sections:
    1. CARR Ranking Table
    2. Convergence Trends (gap to median over time)
    3. Acceleration / Deceleration Indicators
    4. Structural Break Annotations
    5. Fan Chart Data (peer distribution envelope)
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

class MomentumIndicator(str, Enum):
    """Acceleration / deceleration indicator."""
    ACCELERATING = "accelerating"
    STEADY = "steady"
    DECELERATING = "decelerating"
    REVERSING = "reversing"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class CARRRankEntry(BaseModel):
    """Single entry in the CARR ranking table."""
    rank: int = Field(0, ge=0, description="Rank by CARR (1=fastest reducer)")
    entity_name: str = Field(..., description="Company / entity name")
    is_org: bool = Field(False, description="Whether this is the reporting org")
    carr_pct: float = Field(0.0, description="Compound annual reduction rate (%)")
    period_start: int = Field(0, description="Start year of CARR period")
    period_end: int = Field(0, description="End year of CARR period")
    base_emissions: Optional[float] = Field(None, description="Base year emissions")
    current_emissions: Optional[float] = Field(None, description="Current emissions")
    total_reduction_pct: Optional[float] = Field(None, description="Total reduction %")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light status")

class ConvergenceTrendPoint(BaseModel):
    """Gap to median at a point in time."""
    year: int = Field(..., description="Year")
    org_value: float = Field(0.0, description="Organisation value")
    median_value: float = Field(0.0, description="Peer median value")
    gap_absolute: float = Field(0.0, description="Absolute gap (org - median)")
    gap_pct: float = Field(0.0, description="Relative gap (%)")

class MomentumEntry(BaseModel):
    """Acceleration / deceleration entry for an entity."""
    entity_name: str = Field(..., description="Entity name")
    is_org: bool = Field(False, description="Whether this is the org")
    indicator: MomentumIndicator = Field(
        MomentumIndicator.STEADY, description="Momentum indicator"
    )
    recent_carr_pct: float = Field(0.0, description="Recent period CARR (%)")
    prior_carr_pct: float = Field(0.0, description="Prior period CARR (%)")
    change_in_carr: float = Field(0.0, description="Change in CARR (pp)")
    period_description: str = Field("", description="Period description")

class StructuralBreak(BaseModel):
    """Structural break annotation."""
    year: int = Field(..., description="Year of structural break")
    entity_name: str = Field("", description="Entity name (empty=all)")
    break_type: str = Field("", description="Type (e.g., M&A, methodology, restatement)")
    description: str = Field("", description="Description of the break")
    magnitude_pct: Optional[float] = Field(None, description="Magnitude as % change")

class FanChartBand(BaseModel):
    """Single year in the fan chart distribution envelope."""
    year: int = Field(..., description="Year")
    p10: float = Field(0.0, description="10th percentile value")
    p25: float = Field(0.0, description="25th percentile value")
    median: float = Field(0.0, description="Median value")
    p75: float = Field(0.0, description="75th percentile value")
    p90: float = Field(0.0, description="90th percentile value")
    org_value: Optional[float] = Field(None, description="Organisation value")

class TrajectoryAnalysisInput(BaseModel):
    """Complete input model for TrajectoryAnalysisReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    carr_ranking: List[CARRRankEntry] = Field(
        default_factory=list, description="CARR ranking table entries"
    )
    convergence_trends: List[ConvergenceTrendPoint] = Field(
        default_factory=list, description="Convergence trend data points"
    )
    momentum_indicators: List[MomentumEntry] = Field(
        default_factory=list, description="Momentum / acceleration entries"
    )
    structural_breaks: List[StructuralBreak] = Field(
        default_factory=list, description="Structural break annotations"
    )
    fan_chart_data: List[FanChartBand] = Field(
        default_factory=list, description="Fan chart distribution bands"
    )

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tl_label(status: TrafficLight) -> str:
    """Return uppercase label for traffic light."""
    return status.value.upper()

def _tl_color(status: TrafficLight) -> str:
    """Return hex colour for traffic light status."""
    mapping = {
        TrafficLight.GREEN: "#2a9d8f",
        TrafficLight.AMBER: "#e9c46a",
        TrafficLight.RED: "#e76f51",
    }
    return mapping.get(status, "#e9c46a")

def _momentum_label(ind: MomentumIndicator) -> str:
    """Return human-readable momentum label."""
    return ind.value.replace("_", " ").title()

def _momentum_arrow(ind: MomentumIndicator) -> str:
    """Return text arrow for momentum."""
    mapping = {
        MomentumIndicator.ACCELERATING: ">>",
        MomentumIndicator.STEADY: "->",
        MomentumIndicator.DECELERATING: ">.",
        MomentumIndicator.REVERSING: "<<",
    }
    return mapping.get(ind, "->")

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class TrajectoryAnalysisReport:
    """
    Trajectory analysis report template for GHG emissions benchmarking.

    Renders CARR ranking tables, convergence trends, momentum indicators,
    structural break annotations, and fan chart distribution envelopes.
    All outputs include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = TrajectoryAnalysisReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TrajectoryAnalysisReport."""
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
        """Render trajectory analysis as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render trajectory analysis as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render trajectory analysis as JSON dict."""
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
            self._md_carr_ranking(data),
            self._md_convergence(data),
            self._md_momentum(data),
            self._md_structural_breaks(data),
            self._md_fan_chart(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Trajectory Analysis Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_carr_ranking(self, data: Dict[str, Any]) -> str:
        """Render Markdown CARR ranking table."""
        entries = data.get("carr_ranking", [])
        if not entries:
            return "## 1. CARR Ranking\n\nNo CARR ranking data available."
        lines = [
            "## 1. Compound Annual Reduction Rate (CARR) Ranking",
            "",
            "| Rank | Entity | CARR (%/yr) | Period | Total Reduction | Status |",
            "|------|--------|-------------|--------|-----------------|--------|",
        ]
        for e in entries:
            rank = e.get("rank", 0)
            name = e.get("entity_name", "")
            is_org = e.get("is_org", False)
            marker = " **[ORG]**" if is_org else ""
            carr = e.get("carr_pct", 0)
            period_start = e.get("period_start", "")
            period_end = e.get("period_end", "")
            total = e.get("total_reduction_pct")
            total_str = f"{total:+.1f}%" if total is not None else "-"
            status = TrafficLight(e.get("status", "amber"))
            lines.append(
                f"| {rank} | {name}{marker} | {carr:+.2f} | "
                f"{period_start}-{period_end} | {total_str} | **{_tl_label(status)}** |"
            )
        return "\n".join(lines)

    def _md_convergence(self, data: Dict[str, Any]) -> str:
        """Render Markdown convergence trend table."""
        points = data.get("convergence_trends", [])
        if not points:
            return ""
        lines = [
            "## 2. Convergence Trends (Gap to Median)",
            "",
            "| Year | Org Value | Median | Absolute Gap | Relative Gap |",
            "|------|-----------|--------|--------------|--------------|",
        ]
        for p in points:
            lines.append(
                f"| {p.get('year', '')} | {p.get('org_value', 0):,.1f} | "
                f"{p.get('median_value', 0):,.1f} | {p.get('gap_absolute', 0):+,.1f} | "
                f"{p.get('gap_pct', 0):+.1f}% |"
            )
        return "\n".join(lines)

    def _md_momentum(self, data: Dict[str, Any]) -> str:
        """Render Markdown momentum indicators."""
        entries = data.get("momentum_indicators", [])
        if not entries:
            return ""
        lines = [
            "## 3. Acceleration / Deceleration Indicators",
            "",
            "| Entity | Indicator | Recent CARR | Prior CARR | Change (pp) | Period |",
            "|--------|-----------|-------------|------------|-------------|--------|",
        ]
        for e in entries:
            name = e.get("entity_name", "")
            is_org = e.get("is_org", False)
            marker = " **[ORG]**" if is_org else ""
            ind = MomentumIndicator(e.get("indicator", "steady"))
            recent = e.get("recent_carr_pct", 0)
            prior = e.get("prior_carr_pct", 0)
            change = e.get("change_in_carr", 0)
            period = e.get("period_description", "")
            lines.append(
                f"| {name}{marker} | {_momentum_arrow(ind)} {_momentum_label(ind)} | "
                f"{recent:+.2f}% | {prior:+.2f}% | {change:+.2f} | {period} |"
            )
        return "\n".join(lines)

    def _md_structural_breaks(self, data: Dict[str, Any]) -> str:
        """Render Markdown structural break annotations."""
        breaks = data.get("structural_breaks", [])
        if not breaks:
            return ""
        lines = [
            "## 4. Structural Break Annotations",
            "",
            "| Year | Entity | Type | Description | Magnitude |",
            "|------|--------|------|-------------|-----------|",
        ]
        for b in breaks:
            entity = b.get("entity_name", "All")
            magnitude = b.get("magnitude_pct")
            mag_str = f"{magnitude:+.1f}%" if magnitude is not None else "-"
            lines.append(
                f"| {b.get('year', '')} | {entity} | {b.get('break_type', '')} | "
                f"{b.get('description', '')} | {mag_str} |"
            )
        return "\n".join(lines)

    def _md_fan_chart(self, data: Dict[str, Any]) -> str:
        """Render Markdown fan chart distribution data."""
        bands = data.get("fan_chart_data", [])
        if not bands:
            return ""
        lines = [
            "## 5. Peer Distribution Envelope (Fan Chart)",
            "",
            "| Year | P10 | P25 | Median | P75 | P90 | Org Value |",
            "|------|-----|-----|--------|-----|-----|-----------|",
        ]
        for b in bands:
            org = b.get("org_value")
            org_str = f"{org:,.1f}" if org is not None else "-"
            lines.append(
                f"| {b.get('year', '')} | {b.get('p10', 0):,.1f} | "
                f"{b.get('p25', 0):,.1f} | {b.get('median', 0):,.1f} | "
                f"{b.get('p75', 0):,.1f} | {b.get('p90', 0):,.1f} | {org_str} |"
            )
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
            self._html_carr_ranking(data),
            self._html_convergence(data),
            self._html_momentum(data),
            self._html_structural_breaks(data),
            self._html_fan_chart(data),
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
            f"<title>Trajectory Analysis - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            "tr.org-row{background:#e8f5e9;font-weight:600;}\n"
            ".tl-green{color:#2a9d8f;font-weight:700;}\n"
            ".tl-amber{color:#e9c46a;font-weight:700;}\n"
            ".tl-red{color:#e76f51;font-weight:700;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".accelerating{color:#2a9d8f;font-weight:600;}\n"
            ".decelerating{color:#e9c46a;font-weight:600;}\n"
            ".reversing{color:#e76f51;font-weight:600;}\n"
            ".break-badge{display:inline-block;background:#fff3e0;border:1px solid #e9c46a;"
            "border-radius:4px;padding:0.1rem 0.4rem;font-size:0.75rem;margin:0.1rem;}\n"
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
            f"<h1>Trajectory Analysis Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_carr_ranking(self, data: Dict[str, Any]) -> str:
        """Render HTML CARR ranking table."""
        entries = data.get("carr_ranking", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            is_org = e.get("is_org", False)
            row_cls = ' class="org-row"' if is_org else ""
            status = TrafficLight(e.get("status", "amber"))
            css = f"tl-{status.value}"
            total = e.get("total_reduction_pct")
            total_str = f"{total:+.1f}%" if total is not None else "-"
            rows += (
                f"<tr{row_cls}><td>{e.get('rank', 0)}</td>"
                f"<td>{e.get('entity_name', '')}</td>"
                f"<td>{e.get('carr_pct', 0):+.2f}%</td>"
                f"<td>{e.get('period_start', '')}-{e.get('period_end', '')}</td>"
                f"<td>{total_str}</td>"
                f'<td class="{css}"><strong>{_tl_label(status)}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>1. CARR Ranking</h2>\n'
            "<table><thead><tr><th>Rank</th><th>Entity</th><th>CARR (%/yr)</th>"
            "<th>Period</th><th>Total Reduction</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_convergence(self, data: Dict[str, Any]) -> str:
        """Render HTML convergence trend table."""
        points = data.get("convergence_trends", [])
        if not points:
            return ""
        rows = ""
        for p in points:
            rows += (
                f"<tr><td>{p.get('year', '')}</td>"
                f"<td>{p.get('org_value', 0):,.1f}</td>"
                f"<td>{p.get('median_value', 0):,.1f}</td>"
                f"<td>{p.get('gap_absolute', 0):+,.1f}</td>"
                f"<td>{p.get('gap_pct', 0):+.1f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Convergence Trends</h2>\n'
            "<table><thead><tr><th>Year</th><th>Org</th><th>Median</th>"
            "<th>Abs Gap</th><th>Rel Gap</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_momentum(self, data: Dict[str, Any]) -> str:
        """Render HTML momentum indicators."""
        entries = data.get("momentum_indicators", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            is_org = e.get("is_org", False)
            row_cls = ' class="org-row"' if is_org else ""
            ind = MomentumIndicator(e.get("indicator", "steady"))
            ind_css = ind.value
            rows += (
                f"<tr{row_cls}><td>{e.get('entity_name', '')}</td>"
                f'<td class="{ind_css}">{_momentum_label(ind)}</td>'
                f"<td>{e.get('recent_carr_pct', 0):+.2f}%</td>"
                f"<td>{e.get('prior_carr_pct', 0):+.2f}%</td>"
                f"<td>{e.get('change_in_carr', 0):+.2f} pp</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Momentum Indicators</h2>\n'
            "<table><thead><tr><th>Entity</th><th>Indicator</th>"
            "<th>Recent CARR</th><th>Prior CARR</th><th>Change</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_structural_breaks(self, data: Dict[str, Any]) -> str:
        """Render HTML structural break annotations."""
        breaks = data.get("structural_breaks", [])
        if not breaks:
            return ""
        rows = ""
        for b in breaks:
            entity = b.get("entity_name", "All")
            magnitude = b.get("magnitude_pct")
            mag_str = f"{magnitude:+.1f}%" if magnitude is not None else "-"
            rows += (
                f"<tr><td>{b.get('year', '')}</td>"
                f"<td>{entity}</td>"
                f'<td><span class="break-badge">{b.get("break_type", "")}</span></td>'
                f"<td>{b.get('description', '')}</td>"
                f"<td>{mag_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Structural Breaks</h2>\n'
            "<table><thead><tr><th>Year</th><th>Entity</th><th>Type</th>"
            "<th>Description</th><th>Magnitude</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_fan_chart(self, data: Dict[str, Any]) -> str:
        """Render HTML fan chart distribution table."""
        bands = data.get("fan_chart_data", [])
        if not bands:
            return ""
        rows = ""
        for b in bands:
            org = b.get("org_value")
            org_str = f"{org:,.1f}" if org is not None else "-"
            rows += (
                f"<tr><td>{b.get('year', '')}</td>"
                f"<td>{b.get('p10', 0):,.1f}</td>"
                f"<td>{b.get('p25', 0):,.1f}</td>"
                f"<td><strong>{b.get('median', 0):,.1f}</strong></td>"
                f"<td>{b.get('p75', 0):,.1f}</td>"
                f"<td>{b.get('p90', 0):,.1f}</td>"
                f"<td>{org_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Peer Distribution Envelope</h2>\n'
            "<table><thead><tr><th>Year</th><th>P10</th><th>P25</th>"
            "<th>Median</th><th>P75</th><th>P90</th><th>Org</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
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
        """Render trajectory analysis as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "trajectory_analysis_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "carr_ranking": data.get("carr_ranking", []),
            "convergence_trends": data.get("convergence_trends", []),
            "momentum_indicators": data.get("momentum_indicators", []),
            "structural_breaks": data.get("structural_breaks", []),
            "fan_chart_data": data.get("fan_chart_data", []),
            "chart_data": {
                "carr_bar": self._build_carr_bar_chart(data),
                "convergence_line": self._build_convergence_chart(data),
                "fan_chart": self._build_fan_chart(data),
            },
        }

    def _build_carr_bar_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build CARR bar chart data."""
        entries = data.get("carr_ranking", [])
        if not entries:
            return {}
        return {
            "entities": [e.get("entity_name", "") for e in entries],
            "carr_values": [e.get("carr_pct", 0) for e in entries],
            "org_index": next(
                (i for i, e in enumerate(entries) if e.get("is_org")), None
            ),
        }

    def _build_convergence_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build convergence line chart data."""
        points = data.get("convergence_trends", [])
        if not points:
            return {}
        return {
            "years": [p.get("year", 0) for p in points],
            "org_values": [p.get("org_value", 0) for p in points],
            "median_values": [p.get("median_value", 0) for p in points],
            "gap_pct": [p.get("gap_pct", 0) for p in points],
        }

    def _build_fan_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build fan chart data."""
        bands = data.get("fan_chart_data", [])
        if not bands:
            return {}
        return {
            "years": [b.get("year", 0) for b in bands],
            "p10": [b.get("p10", 0) for b in bands],
            "p25": [b.get("p25", 0) for b in bands],
            "median": [b.get("median", 0) for b in bands],
            "p75": [b.get("p75", 0) for b in bands],
            "p90": [b.get("p90", 0) for b in bands],
            "org_values": [b.get("org_value") for b in bands],
        }
