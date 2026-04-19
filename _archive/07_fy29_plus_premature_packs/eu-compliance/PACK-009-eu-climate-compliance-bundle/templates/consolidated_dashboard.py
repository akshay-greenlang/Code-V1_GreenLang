"""
ConsolidatedDashboardTemplate - Multi-regulation compliance overview dashboard.

This module implements the ConsolidatedDashboardTemplate for PACK-009
EU Climate Compliance Bundle. It renders a consolidated view across all
bundled regulations (CSRD, CBAM, EU Taxonomy, SFDR) with traffic-light
status indicators, per-regulation drill-down, trend sparklines, and
deadline tracking.

Example:
    >>> template = ConsolidatedDashboardTemplate()
    >>> data = DashboardData(
    ...     bundle_score=82.5,
    ...     per_regulation_metrics=[...],
    ...     trends=[...],
    ...     deadlines=[...],
    ... )
    >>> md = template.render(data, fmt="markdown")
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
#  Pydantic data models
# ---------------------------------------------------------------------------

class RegulationMetric(BaseModel):
    """Metrics for a single regulation within the bundle."""

    regulation: str = Field(..., description="Regulation code e.g. CSRD, CBAM")
    display_name: str = Field(..., description="Human-readable regulation name")
    compliance_pct: float = Field(..., ge=0.0, le=100.0, description="Compliance percentage")
    data_completeness_pct: float = Field(..., ge=0.0, le=100.0, description="Data completeness %")
    key_metrics: Dict[str, Any] = Field(default_factory=dict, description="Regulation-specific KPIs")
    deadline_status: str = Field("on_track", description="on_track | at_risk | overdue")
    gap_count: int = Field(0, ge=0, description="Number of open gaps")
    last_updated: str = Field("", description="ISO date of last data update")


class TrendPoint(BaseModel):
    """Single point in a trend sparkline."""

    period: str = Field(..., description="Period label e.g. 2025-Q1")
    value: float = Field(..., description="Metric value for the period")


class TrendSeries(BaseModel):
    """A named trend series for sparkline rendering."""

    regulation: str = Field(..., description="Regulation code")
    metric_name: str = Field("compliance_pct", description="Name of the tracked metric")
    points: List[TrendPoint] = Field(default_factory=list, description="Trend data points")


class DeadlineItem(BaseModel):
    """A regulatory deadline entry."""

    regulation: str = Field(..., description="Regulation code")
    description: str = Field(..., description="Deadline description")
    date: str = Field(..., description="ISO date string")
    days_remaining: int = Field(0, description="Days until deadline")
    status: str = Field("upcoming", description="upcoming | imminent | overdue | completed")


class DashboardConfig(BaseModel):
    """Configuration for the consolidated dashboard template."""

    title: str = Field(
        "EU Climate Compliance Bundle - Consolidated Dashboard",
        description="Report title",
    )
    show_trends: bool = Field(True, description="Whether to render trend sparklines")
    score_threshold_green: float = Field(80.0, description="Threshold for green status")
    score_threshold_amber: float = Field(50.0, description="Threshold for amber status")
    regulations_order: List[str] = Field(
        default_factory=lambda: ["CSRD", "CBAM", "EU_TAXONOMY", "SFDR"],
        description="Display order for regulations",
    )


class DashboardData(BaseModel):
    """Input data for the consolidated dashboard."""

    bundle_score: float = Field(..., ge=0.0, le=100.0, description="Overall bundle compliance score")
    per_regulation_metrics: List[RegulationMetric] = Field(
        default_factory=list, description="Per-regulation metric breakdowns"
    )
    trends: List[TrendSeries] = Field(default_factory=list, description="Trend sparkline data")
    deadlines: List[DeadlineItem] = Field(default_factory=list, description="Upcoming deadlines")
    reporting_period: str = Field("", description="Reporting period label e.g. FY2025")
    organization_name: str = Field("", description="Name of the reporting organization")
    generated_by: str = Field("PACK-009", description="Generator identifier")

    @field_validator("per_regulation_metrics")
    @classmethod
    def validate_metrics_non_empty(cls, v: List[RegulationMetric]) -> List[RegulationMetric]:
        """Ensure at least one regulation metric is provided."""
        if not v:
            raise ValueError("per_regulation_metrics must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
#  Template class
# ---------------------------------------------------------------------------

class ConsolidatedDashboardTemplate:
    """
    Multi-regulation consolidated compliance dashboard template.

    Generates formatted dashboards showing bundle-level compliance scores,
    per-regulation drill-downs with traffic-light status, trend sparklines,
    and deadline tracking across CSRD, CBAM, EU Taxonomy, and SFDR.

    Attributes:
        config: Dashboard configuration.
        generated_at: ISO timestamp of report generation.
    """

    TRAFFIC_LIGHT_COLORS = {
        "green": {"hex": "#2ecc71", "label": "GREEN", "symbol": "[G]"},
        "amber": {"hex": "#f39c12", "label": "AMBER", "symbol": "[A]"},
        "red": {"hex": "#e74c3c", "label": "RED", "symbol": "[R]"},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize ConsolidatedDashboardTemplate.

        Args:
            config: Optional configuration dictionary. Keys are passed to
                DashboardConfig for validation.
        """
        raw = config or {}
        self.config = DashboardConfig(**raw) if raw else DashboardConfig()
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def render(self, data: DashboardData, fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render the consolidated dashboard in the specified format.

        Args:
            data: Validated DashboardData input.
            fmt: Output format - 'markdown', 'html', or 'json'.

        Returns:
            Rendered output as string (markdown/html) or dict (json).

        Raises:
            ValueError: If fmt is not one of the supported formats.
        """
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'markdown', 'html', or 'json'.")

    def render_markdown(self, data: DashboardData) -> str:
        """Render as Markdown string."""
        sections: List[str] = [
            self._md_header(data),
            self._md_bundle_score(data),
            self._md_regulation_summary_table(data),
            self._md_regulation_drilldowns(data),
            self._md_trend_sparklines(data),
            self._md_deadlines(data),
            self._md_footer(),
        ]
        content = "\n\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance} -->"
        return content

    def render_html(self, data: DashboardData) -> str:
        """Render as self-contained HTML document."""
        sections: List[str] = [
            self._html_header(data),
            self._html_bundle_score(data),
            self._html_regulation_summary_table(data),
            self._html_regulation_drilldowns(data),
            self._html_trend_sparklines(data),
            self._html_deadlines(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(body)
        return self._wrap_html(body, provenance)

    def render_json(self, data: DashboardData) -> Dict[str, Any]:
        """Render as structured JSON-serializable dictionary."""
        report: Dict[str, Any] = {
            "report_type": "consolidated_dashboard",
            "template_version": "1.0",
            "generated_at": self.generated_at,
            "organization": data.organization_name,
            "reporting_period": data.reporting_period,
            "bundle_score": self._json_bundle_score(data),
            "regulation_summaries": self._json_regulation_summaries(data),
            "trends": self._json_trends(data),
            "deadlines": self._json_deadlines(data),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Traffic-light logic
    # ------------------------------------------------------------------ #

    def _get_traffic_light(self, pct: float) -> str:
        """Determine traffic-light color from a percentage value."""
        if pct >= self.config.score_threshold_green:
            return "green"
        elif pct >= self.config.score_threshold_amber:
            return "amber"
        return "red"

    def _traffic_symbol(self, pct: float) -> str:
        """Return text traffic-light symbol for markdown."""
        color = self._get_traffic_light(pct)
        return self.TRAFFIC_LIGHT_COLORS[color]["symbol"]

    def _traffic_hex(self, pct: float) -> str:
        """Return hex color for HTML rendering."""
        color = self._get_traffic_light(pct)
        return self.TRAFFIC_LIGHT_COLORS[color]["hex"]

    def _traffic_label(self, pct: float) -> str:
        """Return label string for the traffic-light color."""
        color = self._get_traffic_light(pct)
        return self.TRAFFIC_LIGHT_COLORS[color]["label"]

    # ------------------------------------------------------------------ #
    #  Markdown builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: DashboardData) -> str:
        """Build markdown header section."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        light = self._traffic_symbol(data.bundle_score)
        return (
            f"# {self.config.title}\n\n"
            f"**Organization:** {org}\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**Overall Status:** {light} {self._traffic_label(data.bundle_score)}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_bundle_score(self, data: DashboardData) -> str:
        """Build markdown bundle score gauge."""
        score = data.bundle_score
        filled = int(score / 5)
        empty = 20 - filled
        gauge = "[" + "#" * filled + "-" * empty + "]"
        return (
            "## Bundle Compliance Score\n\n"
            f"```\n{gauge} {score:.1f}/100\n```\n\n"
            f"**Status:** {self._traffic_label(score)}"
        )

    def _md_regulation_summary_table(self, data: DashboardData) -> str:
        """Build markdown summary table with traffic lights per regulation."""
        header = (
            "## Regulation Overview\n\n"
            "| Regulation | Status | Compliance | Data Completeness | Gaps | Deadline |\n"
            "|------------|--------|------------|-------------------|------|----------|\n"
        )
        rows: List[str] = []
        ordered = self._order_metrics(data.per_regulation_metrics)
        for m in ordered:
            light = self._traffic_symbol(m.compliance_pct)
            deadline_icon = self._deadline_status_icon(m.deadline_status)
            rows.append(
                f"| {m.display_name} | {light} | "
                f"{m.compliance_pct:.1f}% | "
                f"{m.data_completeness_pct:.1f}% | "
                f"{m.gap_count} | "
                f"{deadline_icon} {m.deadline_status.replace('_', ' ').title()} |"
            )
        return header + "\n".join(rows)

    def _md_regulation_drilldowns(self, data: DashboardData) -> str:
        """Build per-regulation drill-down sections."""
        parts: List[str] = ["## Per-Regulation Details"]
        ordered = self._order_metrics(data.per_regulation_metrics)
        for m in ordered:
            light = self._traffic_symbol(m.compliance_pct)
            section = (
                f"\n### {m.display_name} {light}\n\n"
                f"- **Compliance:** {m.compliance_pct:.1f}%\n"
                f"- **Data Completeness:** {m.data_completeness_pct:.1f}%\n"
                f"- **Open Gaps:** {m.gap_count}\n"
                f"- **Deadline Status:** {m.deadline_status.replace('_', ' ').title()}\n"
                f"- **Last Updated:** {m.last_updated or 'N/A'}\n"
            )
            if m.key_metrics:
                section += "\n**Key Metrics:**\n\n"
                section += "| Metric | Value |\n|--------|-------|\n"
                for k, v in m.key_metrics.items():
                    section += f"| {k.replace('_', ' ').title()} | {self._format_metric_value(v)} |\n"
            parts.append(section)
        return "\n".join(parts)

    def _md_trend_sparklines(self, data: DashboardData) -> str:
        """Build text-based sparkline trend section."""
        if not self.config.show_trends or not data.trends:
            return ""
        parts: List[str] = ["## Compliance Trends"]
        for series in data.trends:
            if not series.points:
                continue
            sparkline = self._text_sparkline(series.points)
            label = f"{series.regulation} - {series.metric_name.replace('_', ' ').title()}"
            first_val = series.points[0].value
            last_val = series.points[-1].value
            delta = last_val - first_val
            arrow = "^" if delta > 0 else "v" if delta < 0 else "="
            parts.append(
                f"\n**{label}:**\n"
                f"```\n{sparkline}\n```\n"
                f"Start: {first_val:.1f} -> End: {last_val:.1f} ({arrow} {abs(delta):.1f})"
            )
        return "\n".join(parts)

    def _md_deadlines(self, data: DashboardData) -> str:
        """Build markdown deadlines section."""
        if not data.deadlines:
            return "## Upcoming Deadlines\n\n*No deadlines configured.*"
        header = (
            "## Upcoming Deadlines\n\n"
            "| Date | Regulation | Description | Days Left | Status |\n"
            "|------|------------|-------------|-----------|--------|\n"
        )
        rows: List[str] = []
        sorted_dl = sorted(data.deadlines, key=lambda d: d.date)
        for dl in sorted_dl:
            icon = self._deadline_status_icon(dl.status)
            rows.append(
                f"| {dl.date[:10]} | {dl.regulation} | "
                f"{dl.description} | {dl.days_remaining} | "
                f"{icon} {dl.status.replace('_', ' ').title()} |"
            )
        return header + "\n".join(rows)

    def _md_footer(self) -> str:
        """Build markdown footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            "*Template: ConsolidatedDashboardTemplate v1.0 | PACK-009*"
        )

    # ------------------------------------------------------------------ #
    #  HTML builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: DashboardData) -> str:
        """Build HTML header banner."""
        color = self._traffic_hex(data.bundle_score)
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        return (
            '<div class="report-header">'
            f'<h1>{self.config.title}</h1>'
            '<div class="header-meta">'
            f'<div class="meta-item">Organization: {org}</div>'
            f'<div class="meta-item">Period: {period}</div>'
            f'<div class="meta-item" style="background:{color};color:#fff">'
            f'{self._traffic_label(data.bundle_score)}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div></div>'
        )

    def _html_bundle_score(self, data: DashboardData) -> str:
        """Build HTML bundle score gauge."""
        score = data.bundle_score
        color = self._traffic_hex(score)
        return (
            '<div class="section"><h2>Bundle Compliance Score</h2>'
            '<div class="score-gauge">'
            f'<div class="gauge-circle" style="border-color:{color}">'
            f'<span class="gauge-value">{score:.0f}</span>'
            '<span class="gauge-label">out of 100</span></div>'
            f'<div class="gauge-status" style="color:{color}">'
            f'{self._traffic_label(score)}</div>'
            '</div></div>'
        )

    def _html_regulation_summary_table(self, data: DashboardData) -> str:
        """Build HTML summary table."""
        rows = ""
        ordered = self._order_metrics(data.per_regulation_metrics)
        for m in ordered:
            color = self._traffic_hex(m.compliance_pct)
            rows += (
                f'<tr>'
                f'<td><strong>{m.display_name}</strong></td>'
                f'<td><span class="status-badge" style="background:{color}">'
                f'{self._traffic_label(m.compliance_pct)}</span></td>'
                f'<td class="num">{m.compliance_pct:.1f}%</td>'
                f'<td class="num">{m.data_completeness_pct:.1f}%</td>'
                f'<td class="num">{m.gap_count}</td>'
                f'<td>{m.deadline_status.replace("_", " ").title()}</td>'
                f'</tr>'
            )
        return (
            '<div class="section"><h2>Regulation Overview</h2>'
            '<table><thead><tr>'
            '<th>Regulation</th><th>Status</th><th>Compliance</th>'
            '<th>Data Completeness</th><th>Gaps</th><th>Deadline</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_regulation_drilldowns(self, data: DashboardData) -> str:
        """Build HTML per-regulation drill-down cards."""
        cards = ""
        ordered = self._order_metrics(data.per_regulation_metrics)
        for m in ordered:
            color = self._traffic_hex(m.compliance_pct)
            metrics_html = ""
            if m.key_metrics:
                metric_rows = ""
                for k, v in m.key_metrics.items():
                    metric_rows += (
                        f'<tr><td>{k.replace("_", " ").title()}</td>'
                        f'<td class="num">{self._format_metric_value(v)}</td></tr>'
                    )
                metrics_html = (
                    '<table class="kpi-table"><thead><tr>'
                    '<th>Metric</th><th>Value</th>'
                    f'</tr></thead><tbody>{metric_rows}</tbody></table>'
                )
            cards += (
                f'<div class="reg-card" style="border-top:4px solid {color}">'
                f'<h3>{m.display_name}</h3>'
                f'<div class="reg-stats">'
                f'<div class="stat"><span class="stat-val">{m.compliance_pct:.1f}%</span>'
                f'<span class="stat-lbl">Compliance</span></div>'
                f'<div class="stat"><span class="stat-val">{m.data_completeness_pct:.1f}%</span>'
                f'<span class="stat-lbl">Data Complete</span></div>'
                f'<div class="stat"><span class="stat-val">{m.gap_count}</span>'
                f'<span class="stat-lbl">Gaps</span></div>'
                f'</div>'
                f'{metrics_html}'
                f'<div class="reg-meta">Last updated: {m.last_updated or "N/A"}</div>'
                f'</div>'
            )
        return f'<div class="section"><h2>Per-Regulation Details</h2><div class="reg-grid">{cards}</div></div>'

    def _html_trend_sparklines(self, data: DashboardData) -> str:
        """Build HTML trend section with inline bar charts."""
        if not self.config.show_trends or not data.trends:
            return ""
        charts = ""
        for series in data.trends:
            if not series.points:
                continue
            max_val = max(p.value for p in series.points) or 1.0
            bars = ""
            for p in series.points:
                h = int((p.value / max_val) * 40) if max_val > 0 else 0
                bars += (
                    f'<div class="spark-bar" title="{p.period}: {p.value:.1f}"'
                    f' style="height:{h}px"></div>'
                )
            first_val = series.points[0].value
            last_val = series.points[-1].value
            delta = last_val - first_val
            delta_color = "#2ecc71" if delta >= 0 else "#e74c3c"
            label = f"{series.regulation} - {series.metric_name.replace('_', ' ').title()}"
            charts += (
                f'<div class="spark-chart">'
                f'<div class="spark-label">{label}</div>'
                f'<div class="spark-bars">{bars}</div>'
                f'<div class="spark-delta" style="color:{delta_color}">'
                f'{first_val:.1f} -> {last_val:.1f} '
                f'({"+" if delta >= 0 else ""}{delta:.1f})</div>'
                f'</div>'
            )
        return f'<div class="section"><h2>Compliance Trends</h2>{charts}</div>'

    def _html_deadlines(self, data: DashboardData) -> str:
        """Build HTML deadlines table."""
        if not data.deadlines:
            return (
                '<div class="section"><h2>Upcoming Deadlines</h2>'
                '<p class="note">No deadlines configured.</p></div>'
            )
        rows = ""
        sorted_dl = sorted(data.deadlines, key=lambda d: d.date)
        for dl in sorted_dl:
            if dl.status == "overdue":
                color = "#e74c3c"
            elif dl.status == "imminent":
                color = "#f39c12"
            else:
                color = "#2ecc71"
            rows += (
                f'<tr>'
                f'<td>{dl.date[:10]}</td>'
                f'<td>{dl.regulation}</td>'
                f'<td>{dl.description}</td>'
                f'<td class="num">{dl.days_remaining}</td>'
                f'<td><span class="status-badge" style="background:{color}">'
                f'{dl.status.replace("_", " ").title()}</span></td>'
                f'</tr>'
            )
        return (
            '<div class="section"><h2>Upcoming Deadlines</h2>'
            '<table><thead><tr>'
            '<th>Date</th><th>Regulation</th><th>Description</th>'
            '<th>Days Left</th><th>Status</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON builders
    # ------------------------------------------------------------------ #

    def _json_bundle_score(self, data: DashboardData) -> Dict[str, Any]:
        """Build JSON bundle score section."""
        return {
            "score": round(data.bundle_score, 1),
            "status": self._traffic_label(data.bundle_score),
            "threshold_green": self.config.score_threshold_green,
            "threshold_amber": self.config.score_threshold_amber,
        }

    def _json_regulation_summaries(self, data: DashboardData) -> List[Dict[str, Any]]:
        """Build JSON regulation summary list."""
        ordered = self._order_metrics(data.per_regulation_metrics)
        return [
            {
                "regulation": m.regulation,
                "display_name": m.display_name,
                "compliance_pct": round(m.compliance_pct, 1),
                "data_completeness_pct": round(m.data_completeness_pct, 1),
                "status": self._traffic_label(m.compliance_pct),
                "gap_count": m.gap_count,
                "deadline_status": m.deadline_status,
                "key_metrics": m.key_metrics,
                "last_updated": m.last_updated,
            }
            for m in ordered
        ]

    def _json_trends(self, data: DashboardData) -> List[Dict[str, Any]]:
        """Build JSON trends section."""
        results: List[Dict[str, Any]] = []
        for series in data.trends:
            points = [{"period": p.period, "value": round(p.value, 2)} for p in series.points]
            first_val = series.points[0].value if series.points else 0.0
            last_val = series.points[-1].value if series.points else 0.0
            results.append({
                "regulation": series.regulation,
                "metric_name": series.metric_name,
                "points": points,
                "start_value": round(first_val, 2),
                "end_value": round(last_val, 2),
                "delta": round(last_val - first_val, 2),
            })
        return results

    def _json_deadlines(self, data: DashboardData) -> List[Dict[str, Any]]:
        """Build JSON deadlines section."""
        sorted_dl = sorted(data.deadlines, key=lambda d: d.date)
        return [
            {
                "date": dl.date[:10],
                "regulation": dl.regulation,
                "description": dl.description,
                "days_remaining": dl.days_remaining,
                "status": dl.status,
            }
            for dl in sorted_dl
        ]

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _order_metrics(self, metrics: List[RegulationMetric]) -> List[RegulationMetric]:
        """Order metrics according to config regulations_order."""
        order_map = {code: i for i, code in enumerate(self.config.regulations_order)}
        return sorted(metrics, key=lambda m: order_map.get(m.regulation, 999))

    def _text_sparkline(self, points: List[TrendPoint]) -> str:
        """Render a text-based sparkline from trend points."""
        if not points:
            return ""
        values = [p.value for p in points]
        min_v = min(values)
        max_v = max(values)
        span = max_v - min_v if max_v != min_v else 1.0
        blocks = " _.-~*"
        chars: List[str] = []
        for v in values:
            idx = int(((v - min_v) / span) * (len(blocks) - 1))
            idx = max(0, min(idx, len(blocks) - 1))
            chars.append(blocks[idx])
        labels = [points[0].period, points[-1].period]
        line = "".join(chars)
        return f"{labels[0]} {line} {labels[1]}"

    def _deadline_status_icon(self, status: str) -> str:
        """Return a text icon for deadline status."""
        icons = {
            "on_track": "[OK]",
            "at_risk": "[!!]",
            "overdue": "[XX]",
            "upcoming": "[->]",
            "imminent": "[!!]",
            "completed": "[OK]",
        }
        return icons.get(status, "[??]")

    def _format_metric_value(self, value: Any) -> str:
        """Format a metric value for display."""
        if isinstance(value, float):
            return f"{value:,.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    def _generate_provenance_hash(self, content: str) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _wrap_html(self, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 12px 0;font-size:24px}"
            ".header-meta{display:flex;flex-wrap:wrap;gap:12px;font-size:14px}"
            ".meta-item{background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".score-gauge{text-align:center;margin:24px 0}"
            ".gauge-circle{display:inline-flex;flex-direction:column;align-items:center;"
            "justify-content:center;width:140px;height:140px;border-radius:50%;"
            "border:6px solid}"
            ".gauge-value{font-size:42px;font-weight:700;line-height:1}"
            ".gauge-label{font-size:12px;color:#7f8c8d}"
            ".gauge-status{font-size:16px;font-weight:600;margin-top:8px}"
            ".status-badge{display:inline-block;padding:2px 10px;border-radius:4px;"
            "color:#fff;font-size:11px;font-weight:bold}"
            ".reg-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));"
            "gap:16px}"
            ".reg-card{background:#f8f9fa;padding:16px;border-radius:8px}"
            ".reg-card h3{margin:0 0 12px 0;font-size:16px}"
            ".reg-stats{display:flex;gap:16px;margin-bottom:12px}"
            ".stat{text-align:center}"
            ".stat-val{display:block;font-size:20px;font-weight:700}"
            ".stat-lbl{display:block;font-size:11px;color:#7f8c8d}"
            ".kpi-table{margin-top:8px;font-size:13px}"
            ".reg-meta{font-size:12px;color:#95a5a6;margin-top:8px}"
            ".spark-chart{margin-bottom:16px;padding:12px;background:#f8f9fa;"
            "border-radius:8px}"
            ".spark-label{font-size:14px;font-weight:600;margin-bottom:8px}"
            ".spark-bars{display:flex;align-items:flex-end;gap:3px;height:44px}"
            ".spark-bar{width:12px;background:#1a5276;border-radius:2px 2px 0 0}"
            ".spark-delta{font-size:13px;margin-top:6px}"
            ".note{color:#7f8c8d;font-style:italic}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{self.config.title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: ConsolidatedDashboardTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
