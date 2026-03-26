# -*- coding: utf-8 -*-
"""
TargetPathwayReport - SBTi Target Pathway Analysis for PACK-046.

Generates a target pathway report with SBTi methodology, base year
summary, pathway chart data (actual vs target line), annual progress
table, gap analysis, required reduction rate, trajectory projection,
and alignment assessment.

Sections:
    1. SBTi Methodology
    2. Base Year Summary
    3. Pathway Chart Data (actual vs target line)
    4. Annual Progress Table
    5. Gap Analysis
    6. Required Reduction Rate
    7. Trajectory Projection
    8. Alignment Assessment

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - PDF (via HTML rendering pipeline)
    - JSON (structured with line chart data)

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


class AlignmentStatus(str, Enum):
    """Target alignment classification."""
    ON_TRACK = "on_track"
    OFF_TRACK = "off_track"
    AHEAD = "ahead"
    NOT_STARTED = "not_started"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class BaseYearSummary(BaseModel):
    """Base year summary for target pathway."""
    base_year: int = Field(..., description="Base year")
    base_emissions_tco2e: float = Field(0.0, description="Base year emissions")
    base_intensity: float = Field(0.0, description="Base year intensity value")
    intensity_unit: str = Field("", description="Intensity unit")
    denominator_value: float = Field(0.0, description="Base year denominator value")
    denominator_unit: str = Field("", description="Denominator unit")


class PathwayPoint(BaseModel):
    """Single point on target or actual pathway."""
    year: int = Field(..., description="Year")
    target_intensity: Optional[float] = Field(None, description="Target pathway intensity")
    actual_intensity: Optional[float] = Field(None, description="Actual intensity")
    target_emissions: Optional[float] = Field(None, description="Target emissions")
    actual_emissions: Optional[float] = Field(None, description="Actual emissions")


class AnnualProgressRow(BaseModel):
    """Annual progress table row."""
    year: int = Field(..., description="Year")
    target_intensity: float = Field(0.0, description="Target intensity for year")
    actual_intensity: Optional[float] = Field(None, description="Actual intensity")
    variance: Optional[float] = Field(None, description="Actual - Target")
    variance_pct: Optional[float] = Field(None, description="Variance as % of target")
    cumulative_reduction_pct: Optional[float] = Field(None, description="Cumulative % reduction from base")
    on_track: Optional[bool] = Field(None, description="On track for this year")


class GapAnalysis(BaseModel):
    """Gap-to-target analysis."""
    current_year: int = Field(0, description="Current reporting year")
    current_intensity: float = Field(0.0, description="Current intensity")
    expected_intensity: float = Field(0.0, description="Expected target intensity this year")
    gap_absolute: float = Field(0.0, description="Absolute gap (current - expected)")
    gap_pct: float = Field(0.0, description="Gap as % of expected")
    remaining_reduction_needed_pct: float = Field(0.0, description="Remaining reduction needed to hit final target")
    years_remaining: int = Field(0, description="Years remaining to target year")


class TrajectoryProjection(BaseModel):
    """Forward trajectory projection."""
    projection_method: str = Field("linear", description="Projection method (linear/compound)")
    projected_target_year_intensity: float = Field(0.0, description="Projected intensity at target year")
    target_year: int = Field(0, description="Target year")
    target_intensity: float = Field(0.0, description="Target intensity at target year")
    projected_gap: float = Field(0.0, description="Projected gap at target year")
    probability_of_achievement: Optional[float] = Field(None, description="Estimated probability of meeting target")


class PathwayReportInput(BaseModel):
    """Complete input model for TargetPathwayReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    sbti_methodology: str = Field("", description="SBTi methodology description")
    target_name: str = Field("", description="Target name (e.g., Near-term SBTi)")
    target_type: str = Field("intensity", description="Target type (intensity / absolute)")
    scope_coverage: str = Field("", description="Scope coverage description")
    base_year_summary: Optional[BaseYearSummary] = Field(
        None, description="Base year summary data"
    )
    pathway_points: List[PathwayPoint] = Field(
        default_factory=list, description="Pathway chart data points"
    )
    annual_progress: List[AnnualProgressRow] = Field(
        default_factory=list, description="Annual progress table"
    )
    gap_analysis: Optional[GapAnalysis] = Field(
        None, description="Gap analysis results"
    )
    required_annual_reduction_pct: Optional[float] = Field(
        None, description="Required annual reduction rate (%)"
    )
    trajectory_projection: Optional[TrajectoryProjection] = Field(
        None, description="Trajectory projection results"
    )
    alignment_status: AlignmentStatus = Field(
        AlignmentStatus.NOT_STARTED, description="Overall alignment status"
    )
    alignment_narrative: str = Field("", description="Alignment assessment narrative")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _alignment_label(status: AlignmentStatus) -> str:
    """Return human-readable label for alignment status."""
    mapping = {
        AlignmentStatus.ON_TRACK: "ON TRACK",
        AlignmentStatus.OFF_TRACK: "OFF TRACK",
        AlignmentStatus.AHEAD: "AHEAD OF TARGET",
        AlignmentStatus.NOT_STARTED: "NOT STARTED",
    }
    return mapping.get(status, status.value.upper())


def _alignment_css(status: AlignmentStatus) -> str:
    """Return CSS class for alignment status."""
    mapping = {
        AlignmentStatus.ON_TRACK: "align-on-track",
        AlignmentStatus.OFF_TRACK: "align-off-track",
        AlignmentStatus.AHEAD: "align-ahead",
        AlignmentStatus.NOT_STARTED: "align-not-started",
    }
    return mapping.get(status, "align-not-started")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class TargetPathwayReport:
    """
    Target pathway report template.

    Renders SBTi target pathway analysis with actual vs target comparison,
    annual progress tracking, gap analysis, trajectory projections, and
    alignment assessment. All outputs include SHA-256 provenance hashing.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = TargetPathwayReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TargetPathwayReport."""
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
        """Render target pathway as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render target pathway as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render target pathway as JSON dict."""
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
            self._md_sbti_methodology(data),
            self._md_base_year(data),
            self._md_annual_progress(data),
            self._md_gap_analysis(data),
            self._md_reduction_rate(data),
            self._md_trajectory(data),
            self._md_alignment(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        target = self._get_val(data, "target_name", "")
        status = AlignmentStatus(self._get_val(data, "alignment_status", "not_started"))
        return (
            f"# Target Pathway Report - {company}\n\n"
            f"**Target:** {target} | "
            f"**Status:** {_alignment_label(status)} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_sbti_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown SBTi methodology."""
        method = self._get_val(data, "sbti_methodology", "")
        target_type = self._get_val(data, "target_type", "intensity")
        scope = self._get_val(data, "scope_coverage", "")
        lines = ["## 1. SBTi Methodology", ""]
        if method:
            lines.append(method)
            lines.append("")
        lines.append(f"**Target Type:** {target_type.title()}")
        if scope:
            lines.append(f"**Scope Coverage:** {scope}")
        return "\n".join(lines)

    def _md_base_year(self, data: Dict[str, Any]) -> str:
        """Render Markdown base year summary."""
        by = data.get("base_year_summary")
        if not by:
            return "## 2. Base Year Summary\n\nNo base year data available."
        lines = [
            "## 2. Base Year Summary",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Base Year | {by.get('base_year', '')} |",
            f"| Base Emissions | {by.get('base_emissions_tco2e', 0):,.1f} tCO2e |",
            f"| Base Intensity | {by.get('base_intensity', 0):,.4f} {by.get('intensity_unit', '')} |",
            f"| Denominator | {by.get('denominator_value', 0):,.2f} {by.get('denominator_unit', '')} |",
        ]
        return "\n".join(lines)

    def _md_annual_progress(self, data: Dict[str, Any]) -> str:
        """Render Markdown annual progress table."""
        progress = data.get("annual_progress", [])
        if not progress:
            return "## 3. Annual Progress\n\nNo progress data available."
        lines = [
            "## 3. Annual Progress",
            "",
            "| Year | Target | Actual | Variance | Var % | Cum. Reduction | On Track |",
            "|------|--------|--------|----------|-------|----------------|----------|",
        ]
        fmt = lambda v: f"{v:,.4f}" if v is not None else "-"
        for r in progress:
            year = r.get("year", "")
            target = r.get("target_intensity", 0)
            actual = r.get("actual_intensity")
            variance = r.get("variance")
            var_pct = r.get("variance_pct")
            cum_red = r.get("cumulative_reduction_pct")
            on_track = r.get("on_track")
            actual_str = fmt(actual)
            var_str = f"{variance:+,.4f}" if variance is not None else "-"
            var_pct_str = f"{var_pct:+.1f}%" if var_pct is not None else "-"
            cum_str = f"{cum_red:.1f}%" if cum_red is not None else "-"
            track_str = "Yes" if on_track else ("No" if on_track is not None else "-")
            lines.append(
                f"| {year} | {target:,.4f} | {actual_str} | "
                f"{var_str} | {var_pct_str} | {cum_str} | {track_str} |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown gap analysis."""
        gap = data.get("gap_analysis")
        if not gap:
            return ""
        lines = [
            "## 4. Gap Analysis",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Current Year | {gap.get('current_year', '')} |",
            f"| Current Intensity | {gap.get('current_intensity', 0):,.4f} |",
            f"| Expected (Target) | {gap.get('expected_intensity', 0):,.4f} |",
            f"| Gap (Absolute) | {gap.get('gap_absolute', 0):+,.4f} |",
            f"| Gap (%) | {gap.get('gap_pct', 0):+.1f}% |",
            f"| Remaining Reduction Needed | {gap.get('remaining_reduction_needed_pct', 0):.1f}% |",
            f"| Years Remaining | {gap.get('years_remaining', 0)} |",
        ]
        return "\n".join(lines)

    def _md_reduction_rate(self, data: Dict[str, Any]) -> str:
        """Render Markdown required reduction rate."""
        rate = data.get("required_annual_reduction_pct")
        if rate is None:
            return ""
        return (
            "## 5. Required Reduction Rate\n\n"
            f"To meet the target, a compound annual reduction rate of **{rate:.2f}%** "
            f"per year is required from the current reporting period."
        )

    def _md_trajectory(self, data: Dict[str, Any]) -> str:
        """Render Markdown trajectory projection."""
        proj = data.get("trajectory_projection")
        if not proj:
            return ""
        lines = [
            "## 6. Trajectory Projection",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Projection Method | {proj.get('projection_method', '')} |",
            f"| Target Year | {proj.get('target_year', '')} |",
            f"| Target Intensity | {proj.get('target_intensity', 0):,.4f} |",
            f"| Projected Intensity | {proj.get('projected_target_year_intensity', 0):,.4f} |",
            f"| Projected Gap | {proj.get('projected_gap', 0):+,.4f} |",
        ]
        prob = proj.get("probability_of_achievement")
        if prob is not None:
            lines.append(f"| Probability of Achievement | {prob:.0f}% |")
        return "\n".join(lines)

    def _md_alignment(self, data: Dict[str, Any]) -> str:
        """Render Markdown alignment assessment."""
        status = AlignmentStatus(self._get_val(data, "alignment_status", "not_started"))
        narrative = self._get_val(data, "alignment_narrative", "")
        lines = [
            "## 7. Alignment Assessment",
            "",
            f"**Overall Status:** {_alignment_label(status)}",
        ]
        if narrative:
            lines.append("")
            lines.append(narrative)
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
            self._html_sbti_methodology(data),
            self._html_base_year(data),
            self._html_annual_progress(data),
            self._html_gap_analysis(data),
            self._html_reduction_rate(data),
            self._html_trajectory(data),
            self._html_alignment(data),
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
            f"<title>Target Pathway Report - {company}</title>\n"
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
            ".align-on-track{color:#2a9d8f;font-weight:700;}\n"
            ".align-off-track{color:#e76f51;font-weight:700;}\n"
            ".align-ahead{color:#264653;font-weight:700;}\n"
            ".align-not-started{color:#888;font-weight:700;}\n"
            ".status-box{border-radius:8px;padding:1rem 1.5rem;margin:1rem 0;}\n"
            ".status-on-track{background:#e8f5e9;border:1px solid #2a9d8f;}\n"
            ".status-off-track{background:#fbe9e7;border:1px solid #e76f51;}\n"
            ".status-ahead{background:#e0f2f1;border:1px solid #264653;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        target = self._get_val(data, "target_name", "")
        status = AlignmentStatus(self._get_val(data, "alignment_status", "not_started"))
        css = _alignment_css(status)
        return (
            '<div class="section">\n'
            f"<h1>Target Pathway Report &mdash; {company}</h1>\n"
            f"<p><strong>Target:</strong> {target} | "
            f'<strong>Status:</strong> <span class="{css}">'
            f"{_alignment_label(status)}</span></p>\n<hr>\n</div>"
        )

    def _html_sbti_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML SBTi methodology."""
        method = self._get_val(data, "sbti_methodology", "")
        if not method:
            return ""
        return (
            '<div class="section">\n<h2>1. SBTi Methodology</h2>\n'
            f"<p>{method}</p>\n</div>"
        )

    def _html_base_year(self, data: Dict[str, Any]) -> str:
        """Render HTML base year summary."""
        by = data.get("base_year_summary")
        if not by:
            return ""
        rows = (
            f"<tr><td>Base Year</td><td>{by.get('base_year', '')}</td></tr>\n"
            f"<tr><td>Base Emissions</td><td>{by.get('base_emissions_tco2e', 0):,.1f} tCO2e</td></tr>\n"
            f"<tr><td>Base Intensity</td><td>{by.get('base_intensity', 0):,.4f} "
            f"{by.get('intensity_unit', '')}</td></tr>\n"
            f"<tr><td>Denominator</td><td>{by.get('denominator_value', 0):,.2f} "
            f"{by.get('denominator_unit', '')}</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>2. Base Year Summary</h2>\n'
            "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_annual_progress(self, data: Dict[str, Any]) -> str:
        """Render HTML annual progress table."""
        progress = data.get("annual_progress", [])
        if not progress:
            return ""
        rows = ""
        fmt = lambda v: f"{v:,.4f}" if v is not None else "-"
        for r in progress:
            year = r.get("year", "")
            target = r.get("target_intensity", 0)
            actual = r.get("actual_intensity")
            on_track = r.get("on_track")
            css = ""
            if on_track is True:
                css = ' class="align-on-track"'
            elif on_track is False:
                css = ' class="align-off-track"'
            track_str = "Yes" if on_track else ("No" if on_track is not None else "-")
            rows += (
                f"<tr><td>{year}</td><td>{target:,.4f}</td><td>{fmt(actual)}</td>"
                f"<td{css}><strong>{track_str}</strong></td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Annual Progress</h2>\n'
            "<table><thead><tr><th>Year</th><th>Target</th>"
            "<th>Actual</th><th>On Track</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis."""
        gap = data.get("gap_analysis")
        if not gap:
            return ""
        rows = (
            f"<tr><td>Current Intensity</td><td>{gap.get('current_intensity', 0):,.4f}</td></tr>\n"
            f"<tr><td>Expected (Target)</td><td>{gap.get('expected_intensity', 0):,.4f}</td></tr>\n"
            f"<tr><td>Gap (Absolute)</td><td>{gap.get('gap_absolute', 0):+,.4f}</td></tr>\n"
            f"<tr><td>Gap (%)</td><td>{gap.get('gap_pct', 0):+.1f}%</td></tr>\n"
            f"<tr><td>Years Remaining</td><td>{gap.get('years_remaining', 0)}</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>4. Gap Analysis</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_reduction_rate(self, data: Dict[str, Any]) -> str:
        """Render HTML required reduction rate."""
        rate = data.get("required_annual_reduction_pct")
        if rate is None:
            return ""
        return (
            '<div class="section">\n<h2>5. Required Reduction Rate</h2>\n'
            f"<p>Compound annual reduction rate required: <strong>{rate:.2f}%</strong> per year.</p>\n</div>"
        )

    def _html_trajectory(self, data: Dict[str, Any]) -> str:
        """Render HTML trajectory projection."""
        proj = data.get("trajectory_projection")
        if not proj:
            return ""
        rows = (
            f"<tr><td>Method</td><td>{proj.get('projection_method', '')}</td></tr>\n"
            f"<tr><td>Target Year</td><td>{proj.get('target_year', '')}</td></tr>\n"
            f"<tr><td>Target Intensity</td><td>{proj.get('target_intensity', 0):,.4f}</td></tr>\n"
            f"<tr><td>Projected Intensity</td><td>{proj.get('projected_target_year_intensity', 0):,.4f}</td></tr>\n"
            f"<tr><td>Projected Gap</td><td>{proj.get('projected_gap', 0):+,.4f}</td></tr>\n"
        )
        prob = proj.get("probability_of_achievement")
        if prob is not None:
            rows += f"<tr><td>Probability</td><td>{prob:.0f}%</td></tr>\n"
        return (
            '<div class="section">\n<h2>6. Trajectory Projection</h2>\n'
            "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_alignment(self, data: Dict[str, Any]) -> str:
        """Render HTML alignment assessment."""
        status = AlignmentStatus(self._get_val(data, "alignment_status", "not_started"))
        narrative = self._get_val(data, "alignment_narrative", "")
        css = _alignment_css(status)
        box_css = {
            AlignmentStatus.ON_TRACK: "status-on-track",
            AlignmentStatus.AHEAD: "status-ahead",
            AlignmentStatus.OFF_TRACK: "status-off-track",
        }.get(status, "")
        content = f'<p class="{css}"><strong>{_alignment_label(status)}</strong></p>\n'
        if narrative:
            content += f"<p>{narrative}</p>\n"
        return (
            '<div class="section">\n<h2>7. Alignment Assessment</h2>\n'
            f'<div class="status-box {box_css}">{content}</div>\n</div>'
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
        """Render target pathway as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "target_pathway_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "target_name": self._get_val(data, "target_name", ""),
            "target_type": self._get_val(data, "target_type", ""),
            "scope_coverage": self._get_val(data, "scope_coverage", ""),
            "alignment_status": self._get_val(data, "alignment_status", "not_started"),
            "base_year_summary": data.get("base_year_summary"),
            "annual_progress": data.get("annual_progress", []),
            "gap_analysis": data.get("gap_analysis"),
            "required_annual_reduction_pct": data.get("required_annual_reduction_pct"),
            "trajectory_projection": data.get("trajectory_projection"),
            "alignment_narrative": self._get_val(data, "alignment_narrative", ""),
            "chart_data": {
                "pathway_line": self._build_pathway_chart(data),
            },
        }

    def _build_pathway_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build line chart data for pathway vs actual trajectory."""
        points = data.get("pathway_points", [])
        if not points:
            return {}
        return {
            "years": [p.get("year", 0) for p in points],
            "target_intensity": [p.get("target_intensity") for p in points],
            "actual_intensity": [p.get("actual_intensity") for p in points],
            "target_emissions": [p.get("target_emissions") for p in points],
            "actual_emissions": [p.get("actual_emissions") for p in points],
        }
