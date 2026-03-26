# -*- coding: utf-8 -*-
"""
AssuranceReadinessDashboard - Readiness Dashboard for PACK-048.

Generates an assurance-readiness executive dashboard with an overall
readiness score (0-100) using traffic-light indicators, category-level
breakdowns across 8-10 assurance dimensions, sparkline trend data for
the last 3-5 assessment cycles, the top 5 remediation gaps, a
time-to-ready estimate, and standard-specific views (ISAE 3410,
ISO 14064-3, AA1000AS).

Regulatory References:
    - ISAE 3410: Assurance Engagements on GHG Statements
    - ISO 14064-3: Specification for validation/verification of GHG assertions
    - AA1000AS v3: AccountAbility Assurance Standard
    - CSRD / ESRS: EU Corporate Sustainability Reporting Directive
    - GHG Protocol Corporate Standard (Chapter 10: Verification)

Sections:
    1. Overall Readiness Score (traffic light)
    2. Category Breakdown (8-10 dimensions)
    3. Sparkline Trend (3-5 assessment cycles)
    4. Top 5 Remediation Gaps
    5. Time-to-Ready Estimate
    6. Standard-Specific View
    7. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 48.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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
    """Traffic-light status indicators."""
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class AssuranceStandard(str, Enum):
    """Supported assurance standards."""
    ISAE_3410 = "ISAE 3410"
    ISO_14064_3 = "ISO 14064-3"
    AA1000AS = "AA1000AS"


class GapSeverity(str, Enum):
    """Gap severity classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class CategoryScore(BaseModel):
    """Single category readiness score."""
    category_name: str = Field(..., description="Category name (e.g., Data Quality)")
    score: float = Field(0.0, ge=0, le=100, description="Score 0-100")
    max_score: float = Field(100.0, ge=0, description="Maximum possible score")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Traffic-light status")
    weight: float = Field(1.0, ge=0, description="Weight in overall score")
    findings_count: int = Field(0, ge=0, description="Number of open findings")
    description: str = Field("", description="Short category description")


class SparklinePoint(BaseModel):
    """Single data point in the readiness sparkline."""
    assessment_date: str = Field(..., description="Assessment date (ISO)")
    cycle_label: str = Field("", description="Cycle label (e.g., Q1 2026)")
    score: float = Field(0.0, ge=0, le=100, description="Readiness score")


class RemediationGap(BaseModel):
    """Single remediation gap item."""
    rank: int = Field(1, ge=1, description="Priority rank (1 = highest)")
    gap_title: str = Field(..., description="Gap title")
    category: str = Field("", description="Category this gap belongs to")
    severity: GapSeverity = Field(GapSeverity.MEDIUM, description="Severity")
    description: str = Field("", description="Detailed description")
    remediation_action: str = Field("", description="Recommended remediation action")
    estimated_effort_days: Optional[int] = Field(None, ge=0, description="Effort estimate (days)")
    owner: str = Field("", description="Responsible person / team")
    due_date: Optional[str] = Field(None, description="Target due date (ISO)")
    status: str = Field("open", description="Current status (open/in_progress/closed)")


class TimeToReadyEstimate(BaseModel):
    """Time-to-ready estimate."""
    estimated_weeks: float = Field(0.0, ge=0, description="Estimated weeks to readiness")
    confidence_level: str = Field("medium", description="Confidence (high/medium/low)")
    critical_path_items: List[str] = Field(
        default_factory=list, description="Critical path items"
    )
    assumptions: List[str] = Field(
        default_factory=list, description="Key assumptions"
    )
    target_date: Optional[str] = Field(None, description="Target ready date (ISO)")


class StandardSpecificView(BaseModel):
    """Standard-specific readiness view."""
    standard: AssuranceStandard = Field(
        AssuranceStandard.ISAE_3410, description="Assurance standard"
    )
    requirements_total: int = Field(0, ge=0, description="Total requirements")
    requirements_met: int = Field(0, ge=0, description="Requirements met")
    compliance_pct: float = Field(0.0, ge=0, le=100, description="Compliance %")
    key_gaps: List[str] = Field(
        default_factory=list, description="Key gaps for this standard"
    )
    notes: str = Field("", description="Additional notes")


class ReadinessDashboardInput(BaseModel):
    """Complete input model for AssuranceReadinessDashboard."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period (e.g., FY2025)")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    overall_score: float = Field(0.0, ge=0, le=100, description="Overall readiness score")
    overall_status: TrafficLight = Field(
        TrafficLight.AMBER, description="Overall traffic-light status"
    )
    category_scores: List[CategoryScore] = Field(
        default_factory=list, description="Category-level scores"
    )
    sparkline_data: List[SparklinePoint] = Field(
        default_factory=list, description="Sparkline trend data (3-5 cycles)"
    )
    top_gaps: List[RemediationGap] = Field(
        default_factory=list, description="Top 5 remediation gaps"
    )
    time_to_ready: Optional[TimeToReadyEstimate] = Field(
        None, description="Time-to-ready estimate"
    )
    standard_views: List[StandardSpecificView] = Field(
        default_factory=list, description="Standard-specific views"
    )
    assurance_level: str = Field("limited", description="Target assurance level")
    engagement_scope: str = Field("", description="Engagement scope description")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tl_label(status: TrafficLight) -> str:
    """Return uppercase label for traffic light."""
    return status.value.upper()


def _tl_color(status: TrafficLight) -> str:
    """Return hex colour for traffic-light status."""
    mapping = {
        TrafficLight.GREEN: "#2a9d8f",
        TrafficLight.AMBER: "#e9c46a",
        TrafficLight.RED: "#e76f51",
    }
    return mapping.get(status, "#e9c46a")


def _tl_css(status: TrafficLight) -> str:
    """Return CSS class for traffic-light status."""
    mapping = {
        TrafficLight.GREEN: "tl-green",
        TrafficLight.AMBER: "tl-amber",
        TrafficLight.RED: "tl-red",
    }
    return mapping.get(status, "tl-amber")


def _format_decimal(value: Optional[float], places: int = 1) -> str:
    """Format a float or return N/A."""
    if value is None:
        return "N/A"
    return f"{value:,.{places}f}"


def _severity_label(severity: str) -> str:
    """Return display label for severity."""
    return severity.upper()


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class AssuranceReadinessDashboard:
    """
    Assurance-readiness dashboard template for PACK-048.

    Renders a comprehensive readiness dashboard with overall score, category
    breakdowns, sparkline trends, top remediation gaps, time-to-ready
    estimate, and standard-specific views. All outputs include SHA-256
    provenance hashing for audit-trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = AssuranceReadinessDashboard()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AssuranceReadinessDashboard."""
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
            data: Dashboard data dict (or ReadinessDashboardInput.dict()).
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
        Render readiness dashboard as Markdown.

        Args:
            data: Dashboard data dict.

        Returns:
            Markdown string with provenance hash.
        """
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render readiness dashboard as HTML.

        Args:
            data: Dashboard data dict.

        Returns:
            Self-contained HTML string.
        """
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render readiness dashboard as JSON-serializable dict.

        Args:
            data: Dashboard data dict.

        Returns:
            JSON-serializable dict with provenance hash.
        """
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
            self._md_overall_score(data),
            self._md_category_breakdown(data),
            self._md_sparkline_trend(data),
            self._md_top_gaps(data),
            self._md_time_to_ready(data),
            self._md_standard_views(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        level = self._get_val(data, "assurance_level", "limited")
        return (
            f"# Assurance Readiness Dashboard - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Assurance Level:** {level.title()} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_overall_score(self, data: Dict[str, Any]) -> str:
        """Render Markdown overall readiness score."""
        score = data.get("overall_score", 0)
        status = TrafficLight(data.get("overall_status", "amber"))
        scope = data.get("engagement_scope", "")
        lines = [
            "## 1. Overall Readiness Score",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Readiness Score | **{score:.0f} / 100** |",
            f"| Status | **{_tl_label(status)}** |",
        ]
        if scope:
            lines.append(f"| Engagement Scope | {scope} |")
        return "\n".join(lines)

    def _md_category_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Markdown category breakdown table."""
        categories = data.get("category_scores", [])
        if not categories:
            return ""
        lines = [
            "## 2. Category Breakdown",
            "",
            "| Category | Score | Max | Status | Weight | Findings |",
            "|----------|-------|-----|--------|--------|----------|",
        ]
        for cat in categories:
            status = TrafficLight(cat.get("status", "amber"))
            lines.append(
                f"| {cat.get('category_name', '')} | "
                f"{cat.get('score', 0):.0f} | "
                f"{cat.get('max_score', 100):.0f} | "
                f"**{_tl_label(status)}** | "
                f"{cat.get('weight', 1.0):.1f} | "
                f"{cat.get('findings_count', 0)} |"
            )
        return "\n".join(lines)

    def _md_sparkline_trend(self, data: Dict[str, Any]) -> str:
        """Render Markdown sparkline trend table."""
        points = data.get("sparkline_data", [])
        if not points:
            return ""
        lines = [
            "## 3. Readiness Trend",
            "",
            "| Cycle | Date | Score |",
            "|-------|------|-------|",
        ]
        for p in points:
            lines.append(
                f"| {p.get('cycle_label', '')} | "
                f"{p.get('assessment_date', '')} | "
                f"{p.get('score', 0):.0f} |"
            )
        return "\n".join(lines)

    def _md_top_gaps(self, data: Dict[str, Any]) -> str:
        """Render Markdown top 5 remediation gaps."""
        gaps = data.get("top_gaps", [])
        if not gaps:
            return ""
        lines = [
            "## 4. Top Remediation Gaps",
            "",
            "| # | Gap | Category | Severity | Action | Effort (days) | Owner | Status |",
            "|---|-----|----------|----------|--------|---------------|-------|--------|",
        ]
        for g in gaps[:5]:
            effort = g.get("estimated_effort_days")
            effort_str = str(effort) if effort is not None else "-"
            lines.append(
                f"| {g.get('rank', '')} | "
                f"{g.get('gap_title', '')} | "
                f"{g.get('category', '')} | "
                f"**{_severity_label(g.get('severity', 'medium'))}** | "
                f"{g.get('remediation_action', '')} | "
                f"{effort_str} | "
                f"{g.get('owner', '')} | "
                f"{g.get('status', 'open')} |"
            )
        return "\n".join(lines)

    def _md_time_to_ready(self, data: Dict[str, Any]) -> str:
        """Render Markdown time-to-ready estimate."""
        ttr = data.get("time_to_ready")
        if not ttr:
            return ""
        lines = [
            "## 5. Time-to-Ready Estimate",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Estimated Weeks | {ttr.get('estimated_weeks', 0):.0f} |",
            f"| Confidence | {ttr.get('confidence_level', 'medium').title()} |",
        ]
        target = ttr.get("target_date")
        if target:
            lines.append(f"| Target Date | {target} |")
        critical = ttr.get("critical_path_items", [])
        if critical:
            lines.append("")
            lines.append("**Critical Path Items:**")
            for item in critical:
                lines.append(f"- {item}")
        assumptions = ttr.get("assumptions", [])
        if assumptions:
            lines.append("")
            lines.append("**Key Assumptions:**")
            for a in assumptions:
                lines.append(f"- {a}")
        return "\n".join(lines)

    def _md_standard_views(self, data: Dict[str, Any]) -> str:
        """Render Markdown standard-specific views."""
        views = data.get("standard_views", [])
        if not views:
            return ""
        lines = [
            "## 6. Standard-Specific Views",
            "",
            "| Standard | Requirements | Met | Compliance % | Key Gaps |",
            "|----------|-------------|-----|-------------|----------|",
        ]
        for v in views:
            gaps_str = "; ".join(v.get("key_gaps", [])[:3]) or "-"
            lines.append(
                f"| {v.get('standard', '')} | "
                f"{v.get('requirements_total', 0)} | "
                f"{v.get('requirements_met', 0)} | "
                f"{v.get('compliance_pct', 0):.1f}% | "
                f"{gaps_str} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-048 Assurance Prep v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overall_score(data),
            self._html_category_breakdown(data),
            self._html_sparkline_trend(data),
            self._html_top_gaps(data),
            self._html_time_to_ready(data),
            self._html_standard_views(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Assurance Readiness - {company}</title>\n"
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
            ".score-card{display:inline-block;background:#f0f4f8;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:200px;"
            "border-left:4px solid #2a9d8f;vertical-align:top;}\n"
            ".score-value{font-size:2rem;font-weight:700;color:#1b263b;}\n"
            ".score-label{font-size:0.85rem;color:#555;margin-top:0.3rem;}\n"
            ".gap-high{color:#e76f51;font-weight:700;}\n"
            ".gap-medium{color:#e9c46a;font-weight:700;}\n"
            ".gap-low{color:#2a9d8f;font-weight:700;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        level = self._get_val(data, "assurance_level", "limited")
        return (
            '<div class="section">\n'
            f"<h1>Assurance Readiness Dashboard &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Assurance Level:</strong> {level.title()} | "
            f"<strong>Report Date:</strong> {_utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overall_score(self, data: Dict[str, Any]) -> str:
        """Render HTML overall readiness score card."""
        score = data.get("overall_score", 0)
        status = TrafficLight(data.get("overall_status", "amber"))
        color = _tl_color(status)
        return (
            '<div class="section">\n<h2>1. Overall Readiness Score</h2>\n'
            f'<div class="score-card" style="border-left-color:{color};">'
            f'<div class="score-value">{score:.0f}</div>'
            f'<div class="score-label">out of 100 &mdash; '
            f'<span class="{_tl_css(status)}">{_tl_label(status)}</span>'
            f'</div></div>\n</div>'
        )

    def _html_category_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML category breakdown table."""
        categories = data.get("category_scores", [])
        if not categories:
            return ""
        rows = ""
        for cat in categories:
            status = TrafficLight(cat.get("status", "amber"))
            rows += (
                f"<tr><td>{cat.get('category_name', '')}</td>"
                f"<td>{cat.get('score', 0):.0f}</td>"
                f"<td>{cat.get('max_score', 100):.0f}</td>"
                f'<td class="{_tl_css(status)}">{_tl_label(status)}</td>'
                f"<td>{cat.get('weight', 1.0):.1f}</td>"
                f"<td>{cat.get('findings_count', 0)}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Category Breakdown</h2>\n'
            "<table><thead><tr><th>Category</th><th>Score</th><th>Max</th>"
            "<th>Status</th><th>Weight</th><th>Findings</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sparkline_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML sparkline trend table."""
        points = data.get("sparkline_data", [])
        if not points:
            return ""
        rows = ""
        for p in points:
            rows += (
                f"<tr><td>{p.get('cycle_label', '')}</td>"
                f"<td>{p.get('assessment_date', '')}</td>"
                f"<td>{p.get('score', 0):.0f}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Readiness Trend</h2>\n'
            "<table><thead><tr><th>Cycle</th><th>Date</th>"
            "<th>Score</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_top_gaps(self, data: Dict[str, Any]) -> str:
        """Render HTML top remediation gaps."""
        gaps = data.get("top_gaps", [])
        if not gaps:
            return ""
        rows = ""
        for g in gaps[:5]:
            severity = g.get("severity", "medium")
            css = f"gap-{severity}"
            effort = g.get("estimated_effort_days")
            effort_str = str(effort) if effort is not None else "-"
            rows += (
                f"<tr><td>{g.get('rank', '')}</td>"
                f"<td>{g.get('gap_title', '')}</td>"
                f"<td>{g.get('category', '')}</td>"
                f'<td class="{css}">{_severity_label(severity)}</td>'
                f"<td>{g.get('remediation_action', '')}</td>"
                f"<td>{effort_str}</td>"
                f"<td>{g.get('owner', '')}</td>"
                f"<td>{g.get('status', 'open')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Top Remediation Gaps</h2>\n'
            "<table><thead><tr><th>#</th><th>Gap</th><th>Category</th>"
            "<th>Severity</th><th>Action</th><th>Effort (days)</th>"
            "<th>Owner</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_time_to_ready(self, data: Dict[str, Any]) -> str:
        """Render HTML time-to-ready estimate."""
        ttr = data.get("time_to_ready")
        if not ttr:
            return ""
        target = ttr.get("target_date", "")
        target_row = f"<tr><td>Target Date</td><td>{target}</td></tr>\n" if target else ""
        rows = (
            f"<tr><td>Estimated Weeks</td><td>{ttr.get('estimated_weeks', 0):.0f}</td></tr>\n"
            f"<tr><td>Confidence</td><td>{ttr.get('confidence_level', 'medium').title()}</td></tr>\n"
            f"{target_row}"
        )
        critical = ttr.get("critical_path_items", [])
        critical_html = ""
        if critical:
            critical_html = "<p><strong>Critical Path:</strong></p><ul>"
            for item in critical:
                critical_html += f"<li>{item}</li>"
            critical_html += "</ul>\n"
        return (
            '<div class="section">\n<h2>5. Time-to-Ready Estimate</h2>\n'
            "<table><thead><tr><th>Attribute</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n{critical_html}</div>"
        )

    def _html_standard_views(self, data: Dict[str, Any]) -> str:
        """Render HTML standard-specific views."""
        views = data.get("standard_views", [])
        if not views:
            return ""
        rows = ""
        for v in views:
            gaps_str = "; ".join(v.get("key_gaps", [])[:3]) or "-"
            rows += (
                f"<tr><td>{v.get('standard', '')}</td>"
                f"<td>{v.get('requirements_total', 0)}</td>"
                f"<td>{v.get('requirements_met', 0)}</td>"
                f"<td>{v.get('compliance_pct', 0):.1f}%</td>"
                f"<td>{gaps_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>6. Standard-Specific Views</h2>\n'
            "<table><thead><tr><th>Standard</th><th>Requirements</th>"
            "<th>Met</th><th>Compliance %</th><th>Key Gaps</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-048 Assurance Prep v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render dashboard as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "assurance_readiness_dashboard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "assurance_level": self._get_val(data, "assurance_level", "limited"),
            "overall_score": data.get("overall_score", 0),
            "overall_status": data.get("overall_status", "amber"),
            "category_scores": data.get("category_scores", []),
            "sparkline_data": data.get("sparkline_data", []),
            "top_gaps": data.get("top_gaps", []),
            "time_to_ready": data.get("time_to_ready"),
            "standard_views": data.get("standard_views", []),
            "chart_data": {
                "readiness_gauge": self._build_gauge_chart(data),
                "category_bar": self._build_category_bar_chart(data),
                "sparkline": self._build_sparkline_chart(data),
                "gap_severity_pie": self._build_gap_severity_pie(data),
            },
        }

    def _build_gauge_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build gauge chart data for overall score."""
        return {
            "value": data.get("overall_score", 0),
            "min": 0,
            "max": 100,
            "status": data.get("overall_status", "amber"),
            "thresholds": {"green": 75, "amber": 50, "red": 0},
        }

    def _build_category_bar_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build category bar chart data."""
        categories = data.get("category_scores", [])
        if not categories:
            return {}
        return {
            "labels": [c.get("category_name", "") for c in categories],
            "scores": [c.get("score", 0) for c in categories],
            "max_scores": [c.get("max_score", 100) for c in categories],
            "statuses": [c.get("status", "amber") for c in categories],
        }

    def _build_sparkline_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build sparkline chart data."""
        points = data.get("sparkline_data", [])
        if not points:
            return {}
        return {
            "labels": [p.get("cycle_label", "") for p in points],
            "dates": [p.get("assessment_date", "") for p in points],
            "scores": [p.get("score", 0) for p in points],
        }

    def _build_gap_severity_pie(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build gap severity pie chart data."""
        gaps = data.get("top_gaps", [])
        if not gaps:
            return {}
        severity_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for g in gaps:
            sev = g.get("severity", "medium")
            if sev in severity_counts:
                severity_counts[sev] += 1
        return {
            "labels": list(severity_counts.keys()),
            "values": list(severity_counts.values()),
        }
