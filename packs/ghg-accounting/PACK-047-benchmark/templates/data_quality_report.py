# -*- coding: utf-8 -*-
"""
DataQualityReport - Data Quality Report for PACK-047.

Generates a data quality report for benchmark analysis with quality score
distribution (histogram), coverage analysis table (% of peers with data
per metric), source hierarchy breakdown, confidence interval summary,
quality improvement recommendations, and PCAF score distribution for
portfolio benchmarking.

Regulatory References:
    - GHG Protocol Corporate Standard (Chapter 7: Managing Inventory Quality)
    - PCAF Global GHG Accounting Standard: Data quality scoring
    - ESRS 1 Appendix B: Data quality requirements
    - TCFD: Data quality for climate metrics

Sections:
    1. Quality Score Distribution (histogram)
    2. Coverage Analysis Table
    3. Source Hierarchy Breakdown
    4. Confidence Interval Summary
    5. Quality Improvement Recommendations
    6. PCAF Score Distribution (portfolio)
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


class DataSource(str, Enum):
    """Data source hierarchy."""
    VERIFIED = "verified"
    REPORTED = "reported"
    ESTIMATED_PHYSICAL = "estimated_physical"
    ESTIMATED_ECONOMIC = "estimated_economic"
    PROXY = "proxy"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class QualityScoreBin(BaseModel):
    """Single bin in quality score histogram."""
    score_range: str = Field(..., description="Score range label (e.g., 0-10, 10-20)")
    count: int = Field(0, ge=0, description="Number of entities in bin")
    pct_of_total: float = Field(0.0, description="Percentage of total")
    org_in_bin: bool = Field(False, description="Whether org falls in this bin")


class CoverageMetric(BaseModel):
    """Coverage of a specific metric across peers."""
    metric_name: str = Field(..., description="Metric name")
    peers_with_data: int = Field(0, ge=0, description="Number of peers with data")
    total_peers: int = Field(0, ge=0, description="Total number of peers")
    coverage_pct: float = Field(0.0, ge=0, le=100, description="Coverage percentage")
    data_year: Optional[int] = Field(None, description="Year of data")
    quality_tier: str = Field("", description="Quality tier (high/medium/low)")
    status: TrafficLight = Field(TrafficLight.AMBER, description="Status")


class SourceBreakdown(BaseModel):
    """Source hierarchy breakdown entry."""
    source_type: DataSource = Field(
        DataSource.REPORTED, description="Data source type"
    )
    source_label: str = Field("", description="Human-readable label")
    entity_count: int = Field(0, ge=0, description="Number of entities using this source")
    pct_of_total: float = Field(0.0, description="Percentage of total")
    avg_quality_score: Optional[float] = Field(None, description="Average quality score")


class ConfidenceIntervalEntry(BaseModel):
    """Confidence interval for a metric."""
    metric_name: str = Field(..., description="Metric name")
    point_estimate: float = Field(0.0, description="Point estimate")
    ci_lower: float = Field(0.0, description="Lower confidence bound")
    ci_upper: float = Field(0.0, description="Upper confidence bound")
    ci_level_pct: float = Field(95.0, description="Confidence level (%)")
    half_width_pct: float = Field(0.0, description="Half-width as % of estimate")
    unit: str = Field("", description="Metric unit")


class QualityRecommendation(BaseModel):
    """Quality improvement recommendation."""
    priority: int = Field(1, ge=1, le=10, description="Priority (1=highest)")
    area: str = Field(..., description="Area for improvement")
    current_state: str = Field("", description="Current state description")
    recommended_action: str = Field("", description="Recommended action")
    expected_impact: str = Field("", description="Expected quality improvement")
    effort_level: str = Field("", description="Effort level (low/medium/high)")


class PCAFScoreEntry(BaseModel):
    """PCAF score distribution entry."""
    score: int = Field(1, ge=1, le=5, description="PCAF score (1-5)")
    description: str = Field("", description="Score description")
    portfolio_pct: float = Field(0.0, description="% of portfolio at this score")
    financed_emissions_pct: float = Field(0.0, description="% of financed emissions")
    entity_count: int = Field(0, ge=0, description="Number of entities")


class DataQualityInput(BaseModel):
    """Complete input model for DataQualityReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    overall_quality_score: Optional[float] = Field(
        None, ge=0, le=100, description="Overall quality score (0-100)"
    )
    overall_status: TrafficLight = Field(TrafficLight.AMBER, description="Overall status")
    quality_histogram: List[QualityScoreBin] = Field(
        default_factory=list, description="Quality score histogram"
    )
    coverage_analysis: List[CoverageMetric] = Field(
        default_factory=list, description="Coverage analysis per metric"
    )
    source_breakdown: List[SourceBreakdown] = Field(
        default_factory=list, description="Source hierarchy breakdown"
    )
    confidence_intervals: List[ConfidenceIntervalEntry] = Field(
        default_factory=list, description="Confidence intervals"
    )
    recommendations: List[QualityRecommendation] = Field(
        default_factory=list, description="Quality improvement recommendations"
    )
    pcaf_distribution: List[PCAFScoreEntry] = Field(
        default_factory=list, description="PCAF score distribution (portfolio)"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tl_label(status: TrafficLight) -> str:
    """Return uppercase label for traffic light."""
    return status.value.upper()


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

class DataQualityReport:
    """
    Data quality report template for GHG emissions benchmarking.

    Renders quality score distributions, coverage analysis, source
    hierarchy breakdowns, confidence intervals, improvement
    recommendations, and PCAF score distributions. All outputs include
    SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = DataQualityReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DataQualityReport."""
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
        """Render data quality report as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render data quality report as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render data quality report as JSON dict."""
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
            self._md_histogram(data),
            self._md_coverage(data),
            self._md_sources(data),
            self._md_confidence(data),
            self._md_recommendations(data),
            self._md_pcaf(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Data Quality Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_overall_score(self, data: Dict[str, Any]) -> str:
        """Render Markdown overall quality score."""
        score = data.get("overall_quality_score")
        if score is None:
            return ""
        status = TrafficLight(data.get("overall_status", "amber"))
        return (
            f"## Overall Quality Score: {score:.0f} / 100 - **{_tl_label(status)}**"
        )

    def _md_histogram(self, data: Dict[str, Any]) -> str:
        """Render Markdown quality score histogram."""
        bins = data.get("quality_histogram", [])
        if not bins:
            return ""
        lines = [
            "## 1. Quality Score Distribution",
            "",
            "| Score Range | Count | % of Total | Org |",
            "|-------------|-------|------------|-----|",
        ]
        for b in bins:
            org = "***" if b.get("org_in_bin", False) else ""
            lines.append(
                f"| {b.get('score_range', '')} | {b.get('count', 0)} | "
                f"{b.get('pct_of_total', 0):.1f}% | {org} |"
            )
        return "\n".join(lines)

    def _md_coverage(self, data: Dict[str, Any]) -> str:
        """Render Markdown coverage analysis table."""
        metrics = data.get("coverage_analysis", [])
        if not metrics:
            return ""
        lines = [
            "## 2. Coverage Analysis",
            "",
            "| Metric | Peers with Data | Total Peers | Coverage (%) | Tier | Status |",
            "|--------|----------------|-------------|--------------|------|--------|",
        ]
        for m in metrics:
            status = TrafficLight(m.get("status", "amber"))
            lines.append(
                f"| {m.get('metric_name', '')} | {m.get('peers_with_data', 0)} | "
                f"{m.get('total_peers', 0)} | {m.get('coverage_pct', 0):.1f}% | "
                f"{m.get('quality_tier', '')} | **{_tl_label(status)}** |"
            )
        return "\n".join(lines)

    def _md_sources(self, data: Dict[str, Any]) -> str:
        """Render Markdown source hierarchy breakdown."""
        sources = data.get("source_breakdown", [])
        if not sources:
            return ""
        lines = [
            "## 3. Source Hierarchy Breakdown",
            "",
            "| Source Type | Count | % of Total | Avg Quality |",
            "|------------|-------|------------|-------------|",
        ]
        for s in sources:
            label = s.get("source_label", "")
            if not label:
                src = DataSource(s.get("source_type", "reported"))
                label = src.value.replace("_", " ").title()
            avg_q = s.get("avg_quality_score")
            avg_str = f"{avg_q:.1f}" if avg_q is not None else "-"
            lines.append(
                f"| {label} | {s.get('entity_count', 0)} | "
                f"{s.get('pct_of_total', 0):.1f}% | {avg_str} |"
            )
        return "\n".join(lines)

    def _md_confidence(self, data: Dict[str, Any]) -> str:
        """Render Markdown confidence interval summary."""
        intervals = data.get("confidence_intervals", [])
        if not intervals:
            return ""
        lines = [
            "## 4. Confidence Interval Summary",
            "",
            "| Metric | Estimate | Lower | Upper | CI Level | Half-Width |",
            "|--------|----------|-------|-------|----------|------------|",
        ]
        for ci in intervals:
            lines.append(
                f"| {ci.get('metric_name', '')} | {ci.get('point_estimate', 0):,.2f} | "
                f"{ci.get('ci_lower', 0):,.2f} | {ci.get('ci_upper', 0):,.2f} | "
                f"{ci.get('ci_level_pct', 95):.0f}% | {ci.get('half_width_pct', 0):.1f}% |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render Markdown quality improvement recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        lines = ["## 5. Quality Improvement Recommendations", ""]
        for r in recs:
            priority = r.get("priority", 1)
            area = r.get("area", "")
            action = r.get("recommended_action", "")
            impact = r.get("expected_impact", "")
            effort = r.get("effort_level", "")
            lines.append(f"**{priority}. {area}**")
            if action:
                lines.append(f"   - Action: {action}")
            if impact:
                lines.append(f"   - Expected Impact: {impact}")
            if effort:
                lines.append(f"   - Effort: {effort}")
            lines.append("")
        return "\n".join(lines)

    def _md_pcaf(self, data: Dict[str, Any]) -> str:
        """Render Markdown PCAF score distribution."""
        entries = data.get("pcaf_distribution", [])
        if not entries:
            return ""
        lines = [
            "## 6. PCAF Score Distribution (Portfolio)",
            "",
            "| Score | Description | Portfolio (%) | Financed Emissions (%) | Entities |",
            "|-------|-------------|---------------|----------------------|----------|",
        ]
        for e in entries:
            lines.append(
                f"| {e.get('score', '')} | {e.get('description', '')} | "
                f"{e.get('portfolio_pct', 0):.1f}% | "
                f"{e.get('financed_emissions_pct', 0):.1f}% | "
                f"{e.get('entity_count', 0)} |"
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
            self._html_overall_score(data),
            self._html_histogram(data),
            self._html_coverage(data),
            self._html_sources(data),
            self._html_confidence(data),
            self._html_recommendations(data),
            self._html_pcaf(data),
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
            f"<title>Data Quality - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #457b9d;padding-bottom:0.5rem;}\n"
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
            ".score-card{background:#f0f4f8;border-radius:8px;padding:1.5rem;"
            "text-align:center;margin:1rem 0;border-left:5px solid #457b9d;}\n"
            ".score-value{font-size:2rem;font-weight:700;color:#1b263b;}\n"
            ".rec-item{background:#f0f9f4;border-left:3px solid #2a9d8f;"
            "padding:0.6rem 1rem;margin:0.5rem 0;border-radius:4px;}\n"
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
            f"<h1>Data Quality Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {_utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overall_score(self, data: Dict[str, Any]) -> str:
        """Render HTML overall quality score card."""
        score = data.get("overall_quality_score")
        if score is None:
            return ""
        status = TrafficLight(data.get("overall_status", "amber"))
        color = _tl_color(status)
        return (
            '<div class="section">\n'
            f'<div class="score-card" style="border-left-color:{color};">\n'
            f'<div class="score-value" style="color:{color};">{score:.0f} / 100</div>\n'
            f'<div>Overall Data Quality Score</div>\n</div>\n</div>'
        )

    def _html_histogram(self, data: Dict[str, Any]) -> str:
        """Render HTML quality score histogram."""
        bins = data.get("quality_histogram", [])
        if not bins:
            return ""
        rows = ""
        for b in bins:
            org = " (ORG)" if b.get("org_in_bin", False) else ""
            rows += (
                f"<tr><td>{b.get('score_range', '')}</td>"
                f"<td>{b.get('count', 0)}</td>"
                f"<td>{b.get('pct_of_total', 0):.1f}%{org}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>1. Quality Score Distribution</h2>\n'
            "<table><thead><tr><th>Score Range</th><th>Count</th>"
            "<th>% of Total</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_coverage(self, data: Dict[str, Any]) -> str:
        """Render HTML coverage analysis."""
        metrics = data.get("coverage_analysis", [])
        if not metrics:
            return ""
        rows = ""
        for m in metrics:
            status = TrafficLight(m.get("status", "amber"))
            css = f"tl-{status.value}"
            rows += (
                f"<tr><td>{m.get('metric_name', '')}</td>"
                f"<td>{m.get('peers_with_data', 0)}</td>"
                f"<td>{m.get('total_peers', 0)}</td>"
                f"<td>{m.get('coverage_pct', 0):.1f}%</td>"
                f"<td>{m.get('quality_tier', '')}</td>"
                f'<td class="{css}"><strong>{_tl_label(status)}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>2. Coverage Analysis</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Peers w/ Data</th>"
            "<th>Total</th><th>Coverage</th><th>Tier</th>"
            "<th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sources(self, data: Dict[str, Any]) -> str:
        """Render HTML source hierarchy breakdown."""
        sources = data.get("source_breakdown", [])
        if not sources:
            return ""
        rows = ""
        for s in sources:
            label = s.get("source_label", "")
            if not label:
                src = DataSource(s.get("source_type", "reported"))
                label = src.value.replace("_", " ").title()
            avg_q = s.get("avg_quality_score")
            avg_str = f"{avg_q:.1f}" if avg_q is not None else "-"
            rows += (
                f"<tr><td>{label}</td>"
                f"<td>{s.get('entity_count', 0)}</td>"
                f"<td>{s.get('pct_of_total', 0):.1f}%</td>"
                f"<td>{avg_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Source Hierarchy</h2>\n'
            "<table><thead><tr><th>Source</th><th>Count</th>"
            "<th>% of Total</th><th>Avg Quality</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_confidence(self, data: Dict[str, Any]) -> str:
        """Render HTML confidence intervals."""
        intervals = data.get("confidence_intervals", [])
        if not intervals:
            return ""
        rows = ""
        for ci in intervals:
            rows += (
                f"<tr><td>{ci.get('metric_name', '')}</td>"
                f"<td>{ci.get('point_estimate', 0):,.2f}</td>"
                f"<td>{ci.get('ci_lower', 0):,.2f}</td>"
                f"<td>{ci.get('ci_upper', 0):,.2f}</td>"
                f"<td>{ci.get('ci_level_pct', 95):.0f}%</td>"
                f"<td>{ci.get('half_width_pct', 0):.1f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Confidence Intervals</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Estimate</th>"
            "<th>Lower</th><th>Upper</th><th>CI Level</th>"
            "<th>Half-Width</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML quality improvement recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        content = ""
        for r in recs:
            priority = r.get("priority", 1)
            area = r.get("area", "")
            action = r.get("recommended_action", "")
            impact = r.get("expected_impact", "")
            effort = r.get("effort_level", "")
            details = ""
            if action:
                details += f"<br><em>Action:</em> {action}"
            if impact:
                details += f"<br><em>Impact:</em> {impact}"
            if effort:
                details += f"<br><em>Effort:</em> {effort}"
            content += (
                f'<div class="rec-item"><strong>{priority}. {area}</strong>'
                f"{details}</div>\n"
            )
        return (
            '<div class="section">\n<h2>5. Improvement Recommendations</h2>\n'
            f"{content}</div>"
        )

    def _html_pcaf(self, data: Dict[str, Any]) -> str:
        """Render HTML PCAF distribution."""
        entries = data.get("pcaf_distribution", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            rows += (
                f"<tr><td>{e.get('score', '')}</td>"
                f"<td>{e.get('description', '')}</td>"
                f"<td>{e.get('portfolio_pct', 0):.1f}%</td>"
                f"<td>{e.get('financed_emissions_pct', 0):.1f}%</td>"
                f"<td>{e.get('entity_count', 0)}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>6. PCAF Score Distribution</h2>\n'
            "<table><thead><tr><th>Score</th><th>Description</th>"
            "<th>Portfolio (%)</th><th>Financed Emissions (%)</th>"
            "<th>Entities</th></tr></thead>\n"
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
        """Render data quality as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "data_quality_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "overall_quality_score": data.get("overall_quality_score"),
            "overall_status": data.get("overall_status", "amber"),
            "quality_histogram": data.get("quality_histogram", []),
            "coverage_analysis": data.get("coverage_analysis", []),
            "source_breakdown": data.get("source_breakdown", []),
            "confidence_intervals": data.get("confidence_intervals", []),
            "recommendations": data.get("recommendations", []),
            "pcaf_distribution": data.get("pcaf_distribution", []),
            "chart_data": {
                "histogram": self._build_histogram_chart(data),
                "coverage_bar": self._build_coverage_chart(data),
                "source_pie": self._build_source_chart(data),
                "pcaf_bar": self._build_pcaf_chart(data),
            },
        }

    def _build_histogram_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build histogram chart data."""
        bins = data.get("quality_histogram", [])
        if not bins:
            return {}
        return {
            "labels": [b.get("score_range", "") for b in bins],
            "counts": [b.get("count", 0) for b in bins],
            "org_bin_index": next(
                (i for i, b in enumerate(bins) if b.get("org_in_bin")), None
            ),
        }

    def _build_coverage_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build coverage bar chart data."""
        metrics = data.get("coverage_analysis", [])
        if not metrics:
            return {}
        return {
            "labels": [m.get("metric_name", "") for m in metrics],
            "coverage_pct": [m.get("coverage_pct", 0) for m in metrics],
        }

    def _build_source_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build source pie chart data."""
        sources = data.get("source_breakdown", [])
        if not sources:
            return {}
        labels = []
        for s in sources:
            label = s.get("source_label", "")
            if not label:
                src = DataSource(s.get("source_type", "reported"))
                label = src.value.replace("_", " ").title()
            labels.append(label)
        return {
            "labels": labels,
            "values": [s.get("pct_of_total", 0) for s in sources],
        }

    def _build_pcaf_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build PCAF bar chart data."""
        entries = data.get("pcaf_distribution", [])
        if not entries:
            return {}
        return {
            "labels": [f"Score {e.get('score', '')}" for e in entries],
            "portfolio_pct": [e.get("portfolio_pct", 0) for e in entries],
            "emissions_pct": [e.get("financed_emissions_pct", 0) for e in entries],
        }
