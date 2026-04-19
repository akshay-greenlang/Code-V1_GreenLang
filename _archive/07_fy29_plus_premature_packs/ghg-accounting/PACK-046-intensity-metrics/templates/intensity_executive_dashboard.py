# -*- coding: utf-8 -*-
"""
IntensityExecutiveDashboard - Executive Dashboard for PACK-046.

Generates a concise executive dashboard covering top intensity metrics with
year-over-year change arrows, benchmark percentile positioning, target
progress (on-track / off-track), decomposition highlights, and prioritised
action items.

Sections:
    1. Key Intensity Metrics (top 5 with YoY change arrows)
    2. Benchmark Position (percentile rank summary)
    3. Target Progress (on-track / off-track with % achieved)
    4. Decomposition Highlights (activity / structure / intensity effects)
    5. Action Items (top 3 recommendations)

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - PDF (via HTML rendering pipeline)
    - JSON (structured with chart-ready sparkline data)

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
    value: float = Field(..., description="Intensity value for the year")

class IntensityMetricItem(BaseModel):
    """Single intensity metric for the dashboard."""
    metric_name: str = Field(..., description="Human-readable metric name")
    numerator_label: str = Field("tCO2e", description="Numerator unit label")
    denominator_label: str = Field("M revenue", description="Denominator unit label")
    current_value: float = Field(..., description="Current period intensity value")
    prior_value: Optional[float] = Field(None, description="Prior period intensity value")
    yoy_change_pct: Optional[float] = Field(None, description="YoY change percentage")
    direction: ChangeDirection = Field(ChangeDirection.FLAT, description="Change direction")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light status")
    sparkline: List[SparklinePoint] = Field(default_factory=list, description="Sparkline data")

    @validator("yoy_change_pct", always=True)
    def compute_yoy_if_missing(cls, v: Optional[float], values: Dict[str, Any]) -> Optional[float]:
        """Auto-compute YoY change if prior_value is available."""
        if v is not None:
            return v
        prior = values.get("prior_value")
        current = values.get("current_value")
        if prior and prior != 0 and current is not None:
            return round(((current - prior) / abs(prior)) * 100.0, 2)
        return None

class BenchmarkResult(BaseModel):
    """Benchmark position summary."""
    metric_name: str = Field(..., description="Metric being benchmarked")
    percentile_rank: float = Field(..., ge=0, le=100, description="Percentile rank (0-100)")
    peer_group: str = Field("", description="Peer group name")
    peer_average: Optional[float] = Field(None, description="Peer group average")
    best_in_class: Optional[float] = Field(None, description="Best-in-class value")
    org_value: Optional[float] = Field(None, description="Organisation value")

class TargetStatus(BaseModel):
    """Target progress status."""
    target_name: str = Field(..., description="Target name")
    target_year: int = Field(..., description="Target year")
    target_value: float = Field(..., description="Target intensity value")
    current_value: float = Field(..., description="Current intensity value")
    base_value: float = Field(..., description="Base year intensity value")
    pct_achieved: float = Field(0.0, description="Percentage of target achieved")
    on_track: bool = Field(True, description="Whether target is on track")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light status")

class DecompositionSummary(BaseModel):
    """Summary of LMDI decomposition effects."""
    period_start: int = Field(..., description="Start year")
    period_end: int = Field(..., description="End year")
    activity_effect_pct: float = Field(0.0, description="Activity effect as % change")
    structure_effect_pct: float = Field(0.0, description="Structure effect as % change")
    intensity_effect_pct: float = Field(0.0, description="Intensity effect as % change")
    total_change_pct: float = Field(0.0, description="Total change as % of base")
    key_driver: str = Field("", description="Primary driver of change")

class ActionItem(BaseModel):
    """Recommended action item."""
    priority: int = Field(1, ge=1, le=5, description="Priority (1=highest)")
    action: str = Field(..., description="Recommended action description")
    expected_impact: str = Field("", description="Expected impact description")
    owner: str = Field("", description="Responsible party")
    timeline: str = Field("", description="Expected timeline")

class DashboardInput(BaseModel):
    """Complete input model for IntensityExecutiveDashboard."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period (e.g. FY2025)")
    report_date: Optional[str] = Field(None, description="Report date (ISO format)")
    intensity_metrics: List[IntensityMetricItem] = Field(
        default_factory=list, description="Top intensity metrics"
    )
    benchmark_results: List[BenchmarkResult] = Field(
        default_factory=list, description="Benchmark position results"
    )
    target_status: List[TargetStatus] = Field(
        default_factory=list, description="Target progress status list"
    )
    decomposition_summary: Optional[DecompositionSummary] = Field(
        None, description="Decomposition highlights"
    )
    action_items: List[ActionItem] = Field(
        default_factory=list, description="Top action items"
    )

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _arrow(direction: ChangeDirection) -> str:
    """Return text arrow for change direction."""
    mapping = {
        ChangeDirection.UP: "^",
        ChangeDirection.DOWN: "v",
        ChangeDirection.FLAT: "-",
    }
    return mapping.get(direction, "-")

def _html_arrow(direction: ChangeDirection) -> str:
    """Return HTML arrow entity for change direction."""
    mapping = {
        ChangeDirection.UP: "&#9650;",
        ChangeDirection.DOWN: "&#9660;",
        ChangeDirection.FLAT: "&#9654;",
    }
    return mapping.get(direction, "&#9654;")

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

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class IntensityExecutiveDashboard:
    """
    Executive dashboard template for intensity metrics.

    Renders a concise dashboard overview of the organisation's intensity
    performance including top metrics with YoY trends, benchmark positioning,
    target progress, decomposition highlights, and prioritised action items.
    All outputs include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = IntensityExecutiveDashboard()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IntensityExecutiveDashboard."""
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
        """
        Render executive dashboard as Markdown.

        Args:
            data: Dashboard data dict (or DashboardInput.dict()).

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

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_key_metrics(data),
            self._md_benchmark(data),
            self._md_target_progress(data),
            self._md_decomposition(data),
            self._md_action_items(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Intensity Metrics Dashboard - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render Markdown key intensity metrics table."""
        metrics = data.get("intensity_metrics", [])
        if not metrics:
            return "## 1. Key Intensity Metrics\n\nNo intensity metrics available."
        lines = [
            "## 1. Key Intensity Metrics",
            "",
            "| Metric | Current | Prior | YoY Change | Trend | Status |",
            "|--------|---------|-------|------------|-------|--------|",
        ]
        for m in metrics[:5]:
            name = m.get("metric_name", "")
            current = m.get("current_value", 0)
            prior = m.get("prior_value")
            prior_str = f"{prior:,.2f}" if prior is not None else "N/A"
            yoy = m.get("yoy_change_pct")
            yoy_str = f"{yoy:+.1f}%" if yoy is not None else "N/A"
            direction = ChangeDirection(m.get("direction", "flat"))
            status = TrafficLight(m.get("status", "amber"))
            num_label = m.get("numerator_label", "tCO2e")
            den_label = m.get("denominator_label", "")
            unit_str = f"{num_label}/{den_label}" if den_label else num_label
            lines.append(
                f"| {name} ({unit_str}) | {current:,.2f} | {prior_str} | "
                f"{yoy_str} | {_arrow(direction)} | **{_tl_label(status)}** |"
            )
        return "\n".join(lines)

    def _md_benchmark(self, data: Dict[str, Any]) -> str:
        """Render Markdown benchmark position summary."""
        benchmarks = data.get("benchmark_results", [])
        if not benchmarks:
            return ""
        lines = [
            "## 2. Benchmark Position",
            "",
            "| Metric | Percentile | Peer Avg | Best-in-Class | Org Value | Gap to Best |",
            "|--------|-----------|----------|---------------|-----------|-------------|",
        ]
        for b in benchmarks:
            name = b.get("metric_name", "")
            pctile = b.get("percentile_rank", 0)
            peer_avg = b.get("peer_average")
            best = b.get("best_in_class")
            org = b.get("org_value")
            peer_str = f"{peer_avg:,.2f}" if peer_avg is not None else "N/A"
            best_str = f"{best:,.2f}" if best is not None else "N/A"
            org_str = f"{org:,.2f}" if org is not None else "N/A"
            gap = ""
            if org is not None and best is not None and best != 0:
                gap_val = ((org - best) / abs(best)) * 100
                gap = f"{gap_val:+.1f}%"
            lines.append(
                f"| {name} | P{pctile:.0f} | {peer_str} | "
                f"{best_str} | {org_str} | {gap} |"
            )
        return "\n".join(lines)

    def _md_target_progress(self, data: Dict[str, Any]) -> str:
        """Render Markdown target progress section."""
        targets = data.get("target_status", [])
        if not targets:
            return ""
        lines = [
            "## 3. Target Progress",
            "",
            "| Target | Target Value | Current | % Achieved | On Track | Status |",
            "|--------|-------------|---------|------------|----------|--------|",
        ]
        for t in targets:
            name = t.get("target_name", "")
            tgt_val = t.get("target_value", 0)
            current = t.get("current_value", 0)
            pct = t.get("pct_achieved", 0)
            on_track = "Yes" if t.get("on_track", False) else "No"
            status = TrafficLight(t.get("status", "amber"))
            lines.append(
                f"| {name} | {tgt_val:,.2f} | {current:,.2f} | "
                f"{pct:.1f}% | {on_track} | **{_tl_label(status)}** |"
            )
        return "\n".join(lines)

    def _md_decomposition(self, data: Dict[str, Any]) -> str:
        """Render Markdown decomposition highlights."""
        decomp = data.get("decomposition_summary")
        if not decomp:
            return ""
        period_start = decomp.get("period_start", "")
        period_end = decomp.get("period_end", "")
        activity = decomp.get("activity_effect_pct", 0)
        structure = decomp.get("structure_effect_pct", 0)
        intensity = decomp.get("intensity_effect_pct", 0)
        total = decomp.get("total_change_pct", 0)
        driver = decomp.get("key_driver", "")
        lines = [
            "## 4. Decomposition Highlights",
            "",
            f"**Period:** {period_start} to {period_end}",
            "",
            "| Effect | Contribution |",
            "|--------|-------------|",
            f"| Activity Effect | {activity:+.1f}% |",
            f"| Structure Effect | {structure:+.1f}% |",
            f"| Intensity Effect | {intensity:+.1f}% |",
            f"| **Total Change** | **{total:+.1f}%** |",
            "",
            f"**Key Driver:** {driver}",
        ]
        return "\n".join(lines)

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Render Markdown action items."""
        actions = data.get("action_items", [])
        if not actions:
            return ""
        lines = ["## 5. Action Items", ""]
        for a in actions[:3]:
            priority = a.get("priority", 1)
            action = a.get("action", "")
            impact = a.get("expected_impact", "")
            owner = a.get("owner", "")
            timeline = a.get("timeline", "")
            lines.append(f"**{priority}.** {action}")
            details = []
            if impact:
                details.append(f"Impact: {impact}")
            if owner:
                details.append(f"Owner: {owner}")
            if timeline:
                details.append(f"Timeline: {timeline}")
            if details:
                lines.append(f"   - {' | '.join(details)}")
            lines.append("")
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
            self._html_key_metrics(data),
            self._html_benchmark(data),
            self._html_target_progress(data),
            self._html_decomposition(data),
            self._html_action_items(data),
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
            f"<title>Intensity Dashboard - {company}</title>\n"
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
            ".metric-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:180px;"
            "border-left:4px solid #2a9d8f;}\n"
            ".metric-value{font-size:1.4rem;font-weight:700;color:#1b263b;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".metric-change{font-size:0.8rem;}\n"
            ".arrow-up{color:#e76f51;}\n"
            ".arrow-down{color:#2a9d8f;}\n"
            ".arrow-flat{color:#888;}\n"
            ".decomp-bar{display:inline-block;height:20px;border-radius:3px;margin:2px 0;}\n"
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
            f"<h1>Intensity Metrics Dashboard &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML key metrics as cards."""
        metrics = data.get("intensity_metrics", [])
        if not metrics:
            return ""
        cards = ""
        for m in metrics[:5]:
            name = m.get("metric_name", "")
            current = m.get("current_value", 0)
            yoy = m.get("yoy_change_pct")
            direction = ChangeDirection(m.get("direction", "flat"))
            status = TrafficLight(m.get("status", "amber"))
            color = _tl_color(status)
            arrow_cls = {
                ChangeDirection.UP: "arrow-up",
                ChangeDirection.DOWN: "arrow-down",
                ChangeDirection.FLAT: "arrow-flat",
            }.get(direction, "arrow-flat")
            yoy_str = f"{yoy:+.1f}%" if yoy is not None else ""
            arrow_html = _html_arrow(direction)
            cards += (
                f'<div class="metric-card" style="border-left-color:{color};">'
                f'<div class="metric-value">{current:,.2f}</div>'
                f'<div class="metric-label">{name}</div>'
                f'<div class="metric-change {arrow_cls}">'
                f'{arrow_html} {yoy_str}</div></div>\n'
            )
        return f'<div class="section">\n<h2>1. Key Intensity Metrics</h2>\n<div>{cards}</div>\n</div>'

    def _html_benchmark(self, data: Dict[str, Any]) -> str:
        """Render HTML benchmark comparison table."""
        benchmarks = data.get("benchmark_results", [])
        if not benchmarks:
            return ""
        rows = ""
        for b in benchmarks:
            name = b.get("metric_name", "")
            pctile = b.get("percentile_rank", 0)
            peer_avg = b.get("peer_average")
            best = b.get("best_in_class")
            org = b.get("org_value")
            peer_str = f"{peer_avg:,.2f}" if peer_avg is not None else "N/A"
            best_str = f"{best:,.2f}" if best is not None else "N/A"
            org_str = f"{org:,.2f}" if org is not None else "N/A"
            rows += (
                f"<tr><td>{name}</td><td>P{pctile:.0f}</td>"
                f"<td>{peer_str}</td><td>{best_str}</td><td>{org_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Benchmark Position</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Percentile</th>"
            "<th>Peer Avg</th><th>Best-in-Class</th>"
            "<th>Org Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_target_progress(self, data: Dict[str, Any]) -> str:
        """Render HTML target progress table."""
        targets = data.get("target_status", [])
        if not targets:
            return ""
        rows = ""
        for t in targets:
            name = t.get("target_name", "")
            current = t.get("current_value", 0)
            pct = t.get("pct_achieved", 0)
            on_track = t.get("on_track", False)
            status = TrafficLight(t.get("status", "amber"))
            css = _tl_css(status)
            label = "On Track" if on_track else "Off Track"
            rows += (
                f'<tr><td>{name}</td><td>{current:,.2f}</td><td>{pct:.1f}%</td>'
                f'<td class="{css}"><strong>{label}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>3. Target Progress</h2>\n'
            "<table><thead><tr><th>Target</th><th>Current</th>"
            "<th>% Achieved</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_decomposition(self, data: Dict[str, Any]) -> str:
        """Render HTML decomposition highlights."""
        decomp = data.get("decomposition_summary")
        if not decomp:
            return ""
        activity = decomp.get("activity_effect_pct", 0)
        structure = decomp.get("structure_effect_pct", 0)
        intensity = decomp.get("intensity_effect_pct", 0)
        total = decomp.get("total_change_pct", 0)
        driver = decomp.get("key_driver", "")
        rows = (
            f"<tr><td>Activity Effect</td><td>{activity:+.1f}%</td></tr>\n"
            f"<tr><td>Structure Effect</td><td>{structure:+.1f}%</td></tr>\n"
            f"<tr><td>Intensity Effect</td><td>{intensity:+.1f}%</td></tr>\n"
            f"<tr><td><strong>Total Change</strong></td>"
            f"<td><strong>{total:+.1f}%</strong></td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>4. Decomposition Highlights</h2>\n'
            "<table><thead><tr><th>Effect</th><th>Contribution</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n"
            f"<p><strong>Key Driver:</strong> {driver}</p>\n</div>"
        )

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Render HTML action items."""
        actions = data.get("action_items", [])
        if not actions:
            return ""
        items = ""
        for a in actions[:3]:
            priority = a.get("priority", 1)
            action = a.get("action", "")
            impact = a.get("expected_impact", "")
            owner = a.get("owner", "")
            detail = ""
            if impact:
                detail += f" &mdash; <em>{impact}</em>"
            if owner:
                detail += f" (Owner: {owner})"
            items += f"<li><strong>P{priority}:</strong> {action}{detail}</li>\n"
        return (
            '<div class="section">\n<h2>5. Action Items</h2>\n'
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
        """Render dashboard as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "intensity_executive_dashboard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "intensity_metrics": data.get("intensity_metrics", []),
            "benchmark_results": data.get("benchmark_results", []),
            "target_status": data.get("target_status", []),
            "decomposition_summary": data.get("decomposition_summary"),
            "action_items": data.get("action_items", []),
            "chart_data": {
                "sparklines": self._build_sparkline_data(data),
                "benchmark_radar": self._build_radar_data(data),
            },
        }

    def _build_sparkline_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build sparkline chart data from metrics."""
        result = []
        for m in data.get("intensity_metrics", []):
            sparkline = m.get("sparkline", [])
            if sparkline:
                result.append({
                    "metric_name": m.get("metric_name", ""),
                    "points": [
                        {"year": p.get("year", 0), "value": p.get("value", 0)}
                        for p in sparkline
                    ],
                })
        return result

    def _build_radar_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build radar/spider chart data from benchmark results."""
        benchmarks = data.get("benchmark_results", [])
        if not benchmarks:
            return {}
        return {
            "labels": [b.get("metric_name", "") for b in benchmarks],
            "org_values": [b.get("percentile_rank", 0) for b in benchmarks],
            "peer_avg_values": [50.0 for _ in benchmarks],
        }
