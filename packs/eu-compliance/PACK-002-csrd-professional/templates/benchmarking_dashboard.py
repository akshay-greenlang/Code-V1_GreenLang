# -*- coding: utf-8 -*-
"""
PACK-002 Phase 3: Benchmarking Dashboard Template
===================================================

Peer comparison dashboard template showing metric-by-metric comparison,
quartile distribution, trend analysis, improvement priorities, and
sector best practices.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 2.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class Quartile(str, Enum):
    """Quartile classification."""
    Q1 = "Q1"  # Top 25%
    Q2 = "Q2"  # 25-50%
    Q3 = "Q3"  # 50-75%
    Q4 = "Q4"  # Bottom 25%


class EffortLevel(str, Enum):
    """Estimated effort level."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class PeerComparison(BaseModel):
    """Single metric peer comparison."""
    metric: str = Field(..., description="Metric name")
    company_value: float = Field(..., description="Company value")
    peer_median: float = Field(..., description="Peer group median")
    peer_p25: float = Field(..., description="Peer 25th percentile")
    peer_p75: float = Field(..., description="Peer 75th percentile")
    percentile_rank: float = Field(
        ..., ge=0.0, le=100.0, description="Company percentile rank"
    )
    quartile: Quartile = Field(..., description="Quartile classification")
    unit: str = Field("", description="Value unit")
    better_direction: Optional[str] = Field(
        None, description="Direction of improvement (higher/lower)"
    )


class TrendMetric(BaseModel):
    """Trend data for a metric over time."""
    metric: str = Field(..., description="Metric name")
    years: List[int] = Field(..., description="Years covered")
    company_values: List[float] = Field(..., description="Company values by year")
    peer_medians: List[float] = Field(..., description="Peer median values by year")
    unit: str = Field("", description="Value unit")


class ImprovementPriority(BaseModel):
    """Improvement priority for a metric."""
    metric: str = Field(..., description="Metric name")
    current_percentile: float = Field(
        ..., ge=0.0, le=100.0, description="Current percentile"
    )
    target_percentile: float = Field(
        ..., ge=0.0, le=100.0, description="Target percentile"
    )
    actions: List[str] = Field(
        default_factory=list, description="Improvement actions"
    )
    estimated_effort: EffortLevel = Field(
        EffortLevel.MEDIUM, description="Estimated effort"
    )
    expected_timeline: Optional[str] = Field(
        None, description="Expected timeline"
    )


class SectorLeader(BaseModel):
    """Sector leader for a specific metric."""
    metric: str = Field(..., description="Metric name")
    leader_name: Optional[str] = Field(None, description="Leader company name")
    leader_value: float = Field(..., description="Leader value")
    leader_quartile: Quartile = Field(Quartile.Q1, description="Leader quartile")
    unit: str = Field("", description="Value unit")


class BenchmarkingDashboardInput(BaseModel):
    """Complete input for the benchmarking dashboard."""
    organization_name: str = Field(..., description="Organization name")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    report_date: date = Field(
        default_factory=date.today, description="Report generation date"
    )
    sector: str = Field("", description="Industry sector")
    peer_count: Optional[int] = Field(None, ge=0, description="Number of peers")
    comparisons: List[PeerComparison] = Field(
        default_factory=list, description="Peer comparisons"
    )
    trend_data: List[TrendMetric] = Field(
        default_factory=list, description="Trend data"
    )
    improvement_priorities: List[ImprovementPriority] = Field(
        default_factory=list, description="Improvement priorities"
    )
    sector_leaders: List[SectorLeader] = Field(
        default_factory=list, description="Sector leader values"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_number(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    """Format numeric value."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K{suffix}"
    return f"{value:,.{decimals}f}{suffix}"


def _quartile_label(q: Quartile) -> str:
    """Human label for quartile."""
    labels = {
        Quartile.Q1: "Top 25%",
        Quartile.Q2: "Upper Mid",
        Quartile.Q3: "Lower Mid",
        Quartile.Q4: "Bottom 25%",
    }
    return labels.get(q, q.value)


def _effort_badge(effort: EffortLevel) -> str:
    """Badge for effort level."""
    return f"[{effort.value}]"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class BenchmarkingDashboardTemplate:
    """Generate peer comparison benchmarking dashboard.

    Sections:
        1. Summary Scorecard
        2. Metric-by-Metric Comparison
        3. Quartile Distribution
        4. Trend Analysis
        5. Improvement Priority Matrix
        6. Sector Best Practices

    Example:
        >>> template = BenchmarkingDashboardTemplate()
        >>> data = BenchmarkingDashboardInput(
        ...     organization_name="Acme", reporting_year=2025, sector="Manufacturing"
        ... )
        >>> md = template.render_markdown(data)
    """

    TEMPLATE_NAME = "benchmarking_dashboard"
    VERSION = "2.0.0"

    def __init__(self) -> None:
        """Initialize the benchmarking dashboard template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC RENDER METHODS
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: BenchmarkingDashboardInput) -> str:
        """Render as Markdown."""
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_summary_scorecard(data),
            self._md_metric_comparison(data),
            self._md_quartile_distribution(data),
            self._md_trend_analysis(data),
            self._md_improvement_priorities(data),
            self._md_sector_leaders(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: BenchmarkingDashboardInput) -> str:
        """Render as HTML document."""
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_summary_scorecard(data),
            self._html_metric_comparison(data),
            self._html_quartile_distribution(data),
            self._html_trend_analysis(data),
            self._html_improvement_priorities(data),
            self._html_sector_leaders(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.organization_name, data.reporting_year, body)

    def render_json(self, data: BenchmarkingDashboardInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict."""
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "organization_name": data.organization_name,
            "reporting_year": data.reporting_year,
            "sector": data.sector,
            "peer_count": data.peer_count,
            "comparisons": [c.model_dump(mode="json") for c in data.comparisons],
            "trend_data": [t.model_dump(mode="json") for t in data.trend_data],
            "improvement_priorities": [
                p.model_dump(mode="json") for p in data.improvement_priorities
            ],
            "sector_leaders": [
                s.model_dump(mode="json") for s in data.sector_leaders
            ],
        }

    def _compute_provenance(self, data: BenchmarkingDashboardInput) -> str:
        """SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: BenchmarkingDashboardInput) -> str:
        sector = data.sector or "All Sectors"
        peers = str(data.peer_count) if data.peer_count is not None else "N/A"
        return (
            f"# Peer Benchmarking Dashboard - {data.organization_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Sector:** {sector} | "
            f"**Peer Group Size:** {peers}\n\n---"
        )

    def _md_summary_scorecard(self, data: BenchmarkingDashboardInput) -> str:
        if not data.comparisons:
            return "## 1. Summary Scorecard\n\nNo comparison data available."
        avg_percentile = sum(c.percentile_rank for c in data.comparisons) / len(data.comparisons)
        q1_count = sum(1 for c in data.comparisons if c.quartile == Quartile.Q1)
        q4_count = sum(1 for c in data.comparisons if c.quartile == Quartile.Q4)
        total_metrics = len(data.comparisons)
        return (
            "## 1. Summary Scorecard\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Overall Percentile | {avg_percentile:.0f}th |\n"
            f"| Metrics Evaluated | {total_metrics} |\n"
            f"| Top Quartile (Q1) | {q1_count} |\n"
            f"| Bottom Quartile (Q4) | {q4_count} |"
        )

    def _md_metric_comparison(self, data: BenchmarkingDashboardInput) -> str:
        if not data.comparisons:
            return "## 2. Metric-by-Metric Comparison\n\nNo data available."
        lines = [
            "## 2. Metric-by-Metric Comparison",
            "",
            "| Metric | Company | Peer Median | P25 | P75 | Percentile | Quartile |",
            "|--------|---------|-------------|-----|-----|------------|----------|",
        ]
        for c in sorted(data.comparisons, key=lambda x: x.percentile_rank, reverse=True):
            unit = f" {c.unit}" if c.unit else ""
            lines.append(
                f"| {c.metric} | {_fmt_number(c.company_value, 1)}{unit} "
                f"| {_fmt_number(c.peer_median, 1)}{unit} "
                f"| {_fmt_number(c.peer_p25, 1)}{unit} "
                f"| {_fmt_number(c.peer_p75, 1)}{unit} "
                f"| {c.percentile_rank:.0f}th | {_quartile_label(c.quartile)} |"
            )
        return "\n".join(lines)

    def _md_quartile_distribution(self, data: BenchmarkingDashboardInput) -> str:
        if not data.comparisons:
            return "## 3. Quartile Distribution\n\nNo data available."
        q_counts = {q: 0 for q in Quartile}
        for c in data.comparisons:
            q_counts[c.quartile] += 1
        total = len(data.comparisons)
        lines = [
            "## 3. Quartile Distribution",
            "",
            "| Quartile | Label | Metrics | Share |",
            "|----------|-------|---------|-------|",
        ]
        for q in Quartile:
            count = q_counts[q]
            share = (count / total * 100) if total > 0 else 0
            lines.append(
                f"| {q.value} | {_quartile_label(q)} | {count} | {share:.0f}% |"
            )
        return "\n".join(lines)

    def _md_trend_analysis(self, data: BenchmarkingDashboardInput) -> str:
        if not data.trend_data:
            return "## 4. Trend Analysis\n\nNo trend data available."
        lines = ["## 4. Trend Analysis (Company vs Peer Median)", ""]
        for t in data.trend_data:
            unit = f" ({t.unit})" if t.unit else ""
            lines.extend([
                f"### {t.metric}{unit}",
                "",
                "| Year | Company | Peer Median |",
                "|------|---------|-------------|",
            ])
            for i, year in enumerate(t.years):
                cv = t.company_values[i] if i < len(t.company_values) else None
                pm = t.peer_medians[i] if i < len(t.peer_medians) else None
                lines.append(
                    f"| {year} | {_fmt_number(cv, 1)} | {_fmt_number(pm, 1)} |"
                )
            lines.append("")
        return "\n".join(lines)

    def _md_improvement_priorities(self, data: BenchmarkingDashboardInput) -> str:
        if not data.improvement_priorities:
            return "## 5. Improvement Priorities\n\nNo priorities identified."
        lines = ["## 5. Improvement Priority Matrix", ""]
        for i, p in enumerate(data.improvement_priorities, 1):
            timeline = p.expected_timeline or "TBD"
            lines.extend([
                f"### {i}. {p.metric}",
                f"- **Current Percentile:** {p.current_percentile:.0f}th",
                f"- **Target Percentile:** {p.target_percentile:.0f}th",
                f"- **Estimated Effort:** {_effort_badge(p.estimated_effort)}",
                f"- **Timeline:** {timeline}",
            ])
            if p.actions:
                lines.append("- **Actions:**")
                for action in p.actions:
                    lines.append(f"  - {action}")
            lines.append("")
        return "\n".join(lines)

    def _md_sector_leaders(self, data: BenchmarkingDashboardInput) -> str:
        if not data.sector_leaders:
            return "## 6. Sector Best Practices\n\nNo sector leader data."
        lines = [
            "## 6. Sector Best Practices",
            "",
            "| Metric | Leader Value | Quartile | Unit |",
            "|--------|-------------|----------|------|",
        ]
        for s in data.sector_leaders:
            unit = s.unit or "-"
            lines.append(
                f"| {s.metric} | {_fmt_number(s.leader_value, 1)} "
                f"| {_quartile_label(s.leader_quartile)} | {unit} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: BenchmarkingDashboardInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, org: str, year: int, body: str) -> str:
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Benchmarking Dashboard - {org} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "color:#1a1a2e;max-width:1200px;}\n"
            "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:0.5rem;}\n"
            "h2{color:#0f3460;border-bottom:1px solid #ddd;padding-bottom:0.3rem;margin-top:2rem;}\n"
            "h3{color:#533483;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left;}\n"
            "th{background:#f0f4f8;color:#16213e;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".q1{color:#1a7f37;font-weight:bold;}\n"
            ".q2{color:#2da44e;}\n"
            ".q3{color:#b08800;}\n"
            ".q4{color:#cf222e;font-weight:bold;}\n"
            ".metric-card{display:inline-block;text-align:center;padding:1rem 1.5rem;"
            "border:1px solid #ddd;border-radius:8px;margin:0.5rem;background:#f8f9fa;}\n"
            ".metric-value{font-size:1.5rem;font-weight:bold;color:#0f3460;}\n"
            ".metric-label{font-size:0.85rem;color:#666;}\n"
            ".priority-box{background:#f8f9fa;border-left:4px solid #0f3460;"
            "padding:1rem;margin:1rem 0;border-radius:0 6px 6px 0;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n</body>\n</html>"
        )

    def _html_header(self, data: BenchmarkingDashboardInput) -> str:
        sector = data.sector or "All Sectors"
        peers = str(data.peer_count) if data.peer_count is not None else "N/A"
        return (
            '<div class="section">\n'
            f"<h1>Peer Benchmarking Dashboard &mdash; {data.organization_name}</h1>\n"
            f"<p><strong>Year:</strong> {data.reporting_year} | "
            f"<strong>Sector:</strong> {sector} | "
            f"<strong>Peers:</strong> {peers}</p>\n"
            "<hr>\n</div>"
        )

    def _html_summary_scorecard(self, data: BenchmarkingDashboardInput) -> str:
        if not data.comparisons:
            return (
                '<div class="section"><h2>1. Summary</h2>'
                "<p>No data available.</p></div>"
            )
        avg_pct = sum(c.percentile_rank for c in data.comparisons) / len(data.comparisons)
        q1 = sum(1 for c in data.comparisons if c.quartile == Quartile.Q1)
        q4 = sum(1 for c in data.comparisons if c.quartile == Quartile.Q4)
        cards = [
            (f"{avg_pct:.0f}th", "Overall Percentile"),
            (str(len(data.comparisons)), "Metrics"),
            (str(q1), "Top Quartile"),
            (str(q4), "Bottom Quartile"),
        ]
        card_html = "\n".join(
            f'<div class="metric-card"><div class="metric-value">{v}</div>'
            f'<div class="metric-label">{l}</div></div>'
            for v, l in cards
        )
        return (
            '<div class="section">\n<h2>1. Summary Scorecard</h2>\n'
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_metric_comparison(self, data: BenchmarkingDashboardInput) -> str:
        if not data.comparisons:
            return (
                '<div class="section"><h2>2. Comparison</h2>'
                "<p>No data.</p></div>"
            )
        rows = []
        for c in sorted(data.comparisons, key=lambda x: x.percentile_rank, reverse=True):
            css = f"q{c.quartile.value[-1].lower()}" if c.quartile else ""
            unit = f" {c.unit}" if c.unit else ""
            rows.append(
                f"<tr><td>{c.metric}</td>"
                f"<td>{_fmt_number(c.company_value, 1)}{unit}</td>"
                f"<td>{_fmt_number(c.peer_median, 1)}{unit}</td>"
                f"<td>{_fmt_number(c.peer_p25, 1)}{unit}</td>"
                f"<td>{_fmt_number(c.peer_p75, 1)}{unit}</td>"
                f"<td>{c.percentile_rank:.0f}th</td>"
                f'<td class="{css}">{_quartile_label(c.quartile)}</td></tr>'
            )
        return (
            '<div class="section">\n<h2>2. Metric-by-Metric Comparison</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Company</th><th>Median</th>"
            "<th>P25</th><th>P75</th><th>Percentile</th>"
            f"<th>Quartile</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_quartile_distribution(self, data: BenchmarkingDashboardInput) -> str:
        if not data.comparisons:
            return ""
        q_counts = {q: 0 for q in Quartile}
        for c in data.comparisons:
            q_counts[c.quartile] += 1
        total = len(data.comparisons)
        rows = []
        for q in Quartile:
            count = q_counts[q]
            share = (count / total * 100) if total > 0 else 0
            css = f"q{q.value[-1].lower()}"
            rows.append(
                f'<tr><td class="{css}">{q.value}</td>'
                f"<td>{_quartile_label(q)}</td>"
                f"<td>{count}</td><td>{share:.0f}%</td></tr>"
            )
        return (
            '<div class="section">\n<h2>3. Quartile Distribution</h2>\n'
            "<table><thead><tr><th>Quartile</th><th>Label</th>"
            f"<th>Metrics</th><th>Share</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_trend_analysis(self, data: BenchmarkingDashboardInput) -> str:
        if not data.trend_data:
            return (
                '<div class="section"><h2>4. Trends</h2>'
                "<p>No trend data.</p></div>"
            )
        parts = ['<div class="section">\n<h2>4. Trend Analysis</h2>\n']
        for t in data.trend_data:
            unit = f" ({t.unit})" if t.unit else ""
            rows = []
            for i, year in enumerate(t.years):
                cv = t.company_values[i] if i < len(t.company_values) else None
                pm = t.peer_medians[i] if i < len(t.peer_medians) else None
                rows.append(
                    f"<tr><td>{year}</td><td>{_fmt_number(cv, 1)}</td>"
                    f"<td>{_fmt_number(pm, 1)}</td></tr>"
                )
            parts.append(
                f"<h3>{t.metric}{unit}</h3>\n"
                "<table><thead><tr><th>Year</th><th>Company</th>"
                f"<th>Peer Median</th></tr></thead>\n"
                f"<tbody>{''.join(rows)}</tbody></table>\n"
            )
        parts.append("</div>")
        return "".join(parts)

    def _html_improvement_priorities(self, data: BenchmarkingDashboardInput) -> str:
        if not data.improvement_priorities:
            return (
                '<div class="section"><h2>5. Priorities</h2>'
                "<p>No priorities identified.</p></div>"
            )
        parts = ['<div class="section">\n<h2>5. Improvement Priorities</h2>\n']
        for i, p in enumerate(data.improvement_priorities, 1):
            timeline = p.expected_timeline or "TBD"
            actions_html = ""
            if p.actions:
                items = "".join(f"<li>{a}</li>" for a in p.actions)
                actions_html = f"<ul>{items}</ul>\n"
            parts.append(
                f'<div class="priority-box">\n'
                f"<h3>{i}. {p.metric}</h3>\n"
                f"<p><strong>Current:</strong> {p.current_percentile:.0f}th &rarr; "
                f"<strong>Target:</strong> {p.target_percentile:.0f}th | "
                f"<strong>Effort:</strong> {p.estimated_effort.value} | "
                f"<strong>Timeline:</strong> {timeline}</p>\n"
                f"{actions_html}</div>\n"
            )
        parts.append("</div>")
        return "".join(parts)

    def _html_sector_leaders(self, data: BenchmarkingDashboardInput) -> str:
        if not data.sector_leaders:
            return (
                '<div class="section"><h2>6. Sector Leaders</h2>'
                "<p>No data.</p></div>"
            )
        rows = []
        for s in data.sector_leaders:
            unit = s.unit or "-"
            name = s.leader_name or "Anonymous"
            rows.append(
                f"<tr><td>{s.metric}</td><td>{name}</td>"
                f"<td>{_fmt_number(s.leader_value, 1)}</td>"
                f"<td>{_quartile_label(s.leader_quartile)}</td>"
                f"<td>{unit}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>6. Sector Best Practices</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Leader</th><th>Value</th>"
            f"<th>Quartile</th><th>Unit</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: BenchmarkingDashboardInput) -> str:
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Professional Pack v{self.VERSION} | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
