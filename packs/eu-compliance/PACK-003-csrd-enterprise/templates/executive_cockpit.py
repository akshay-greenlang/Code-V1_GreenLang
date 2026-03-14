"""
ExecutiveCockpitTemplate - C-suite real-time dashboard for CSRD Enterprise Pack.

This module implements the executive cockpit template with key performance
indicators, risk exposure radar, compliance trajectory, peer benchmarking,
board action items, financial impact summary, sparklines, and quick links.

Example:
    >>> template = ExecutiveCockpitTemplate()
    >>> data = {"kpis": {...}, "risk_radar": [...], "trajectory": [...]}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ExecutiveCockpitTemplate:
    """
    Executive cockpit dashboard template for C-suite.

    Renders KPI cards with sparklines, risk exposure radar, compliance
    trajectory, peer benchmarking, board action items, financial impact,
    and quick links to detailed reports.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    RADAR_DIMENSIONS = [
        "Environmental", "Social", "Governance",
        "Climate", "Supply Chain", "Regulatory",
    ]

    KPI_DEFINITIONS = [
        {"key": "total_emissions", "label": "Total Emissions (tCO2e)", "format": "number"},
        {"key": "yoy_change", "label": "YoY Change", "format": "percentage"},
        {"key": "compliance_score", "label": "Compliance Score", "format": "number"},
        {"key": "sbti_progress", "label": "SBTi Progress", "format": "percentage"},
        {"key": "esg_rating_prediction", "label": "ESG Rating Prediction", "format": "text"},
        {"key": "financial_materiality", "label": "Financial Materiality", "format": "currency"},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ExecutiveCockpitTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render executive cockpit as Markdown.

        Args:
            data: Dashboard data with kpis, risk_radar, trajectory,
                  peer_benchmarking, board_actions, financial_impact,
                  sparklines, quick_links.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._render_md_header(data))
        sections.append(self._render_md_kpis(data))
        sections.append(self._render_md_risk_radar(data))
        sections.append(self._render_md_trajectory(data))
        sections.append(self._render_md_peer_benchmark(data))
        sections.append(self._render_md_board_actions(data))
        sections.append(self._render_md_financial_impact(data))
        sections.append(self._render_md_sparklines(data))
        sections.append(self._render_md_quick_links(data))
        sections.append(self._render_md_footer(data))

        content = "\n\n".join(sections)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- Provenance: {provenance} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive cockpit as self-contained HTML.

        Args:
            data: Dashboard data dict.

        Returns:
            Complete HTML string with inline styles.
        """
        self.generated_at = datetime.utcnow()
        css = self._build_css()
        body_parts: List[str] = []

        body_parts.append(self._render_html_header(data))
        body_parts.append(self._render_html_kpis(data))
        body_parts.append(self._render_html_risk_radar(data))
        body_parts.append(self._render_html_trajectory(data))
        body_parts.append(self._render_html_peer_benchmark(data))
        body_parts.append(self._render_html_board_actions(data))
        body_parts.append(self._render_html_financial_impact(data))
        body_parts.append(self._render_html_sparklines(data))
        body_parts.append(self._render_html_quick_links(data))
        body_parts.append(self._render_html_footer(data))

        body_html = "\n".join(body_parts)
        provenance = self._generate_provenance_hash(body_html)

        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>Executive Cockpit</title>\n<style>\n{css}\n</style>\n"
            "</head>\n<body>\n"
            f"<div class=\"cockpit-container\">\n{body_html}\n</div>\n"
            f"<!-- Provenance: {provenance} -->\n"
            "</body>\n</html>"
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render executive cockpit as structured JSON.

        Args:
            data: Dashboard data dict.

        Returns:
            Structured dict with all dashboard sections.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "executive_cockpit",
            "version": "1.0.0",
            "generated_at": self.generated_at.isoformat(),
            "kpis": data.get("kpis", {}),
            "risk_radar": data.get("risk_radar", []),
            "compliance_trajectory": data.get("trajectory", []),
            "peer_benchmarking": data.get("peer_benchmarking", {}),
            "board_actions": data.get("board_actions", {}),
            "financial_impact": data.get("financial_impact", {}),
            "sparklines": data.get("sparklines", {}),
            "quick_links": data.get("quick_links", []),
        }
        provenance = self._generate_provenance_hash(json.dumps(result, default=str))
        result["provenance_hash"] = provenance
        return result

    # ------------------------------------------------------------------
    # Markdown renderers
    # ------------------------------------------------------------------

    def _render_md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        title = data.get("title", "Executive Cockpit")
        ts = self._format_date(self.generated_at)
        return f"# {title}\n\n**Generated:** {ts}\n\n---"

    def _render_md_kpis(self, data: Dict[str, Any]) -> str:
        """Render key performance indicator cards."""
        kpis = data.get("kpis", {})
        if not kpis:
            return "## Key Performance Indicators\n\n_No KPI data available._"

        lines = [
            "## Key Performance Indicators",
            "",
            "| KPI | Value | Trend (90d) | Status |",
            "|-----|-------|------------|--------|",
        ]
        for kpi_def in self.KPI_DEFINITIONS:
            key = kpi_def["key"]
            label = kpi_def["label"]
            kpi_data = kpis.get(key, {})
            value = kpi_data.get("value", 0)
            trend = kpi_data.get("trend_90d", 0)
            status = kpi_data.get("status", "-")

            if kpi_def["format"] == "percentage":
                formatted = self._format_percentage(value)
            elif kpi_def["format"] == "currency":
                formatted = f"${self._format_number(value)}"
            elif kpi_def["format"] == "text":
                formatted = str(value)
            else:
                formatted = self._format_number(value)

            trend_str = f"{'+'if trend > 0 else ''}{self._format_percentage(trend)}"
            lines.append(f"| {label} | {formatted} | {trend_str} | {status} |")

        return "\n".join(lines)

    def _render_md_risk_radar(self, data: Dict[str, Any]) -> str:
        """Render risk exposure radar chart as a table."""
        radar: List[Dict[str, Any]] = data.get("risk_radar", [])
        if not radar:
            return "## Risk Exposure\n\n_No risk radar data available._"

        lines = [
            "## Risk Exposure Radar (E/S/G Dimensions)",
            "",
            "| Dimension | Score (0-100) | Level | Trend |",
            "|-----------|-------------|-------|-------|",
        ]
        for r in radar:
            dimension = r.get("dimension", "-")
            score = self._format_number(r.get("score", 0), 1)
            level = r.get("level", "-")
            trend = r.get("trend", "-")
            bar = self._text_bar(r.get("score", 0), 100.0)
            lines.append(f"| {dimension} | {score} {bar} | {level} | {trend} |")

        return "\n".join(lines)

    def _render_md_trajectory(self, data: Dict[str, Any]) -> str:
        """Render compliance trajectory."""
        trajectory: List[Dict[str, Any]] = data.get("trajectory", [])
        if not trajectory:
            return "## Compliance Trajectory\n\n_No trajectory data available._"

        lines = [
            "## Compliance Trajectory (Projected vs Target)",
            "",
            "| Period | Projected | Target | Gap | On Track |",
            "|--------|----------|--------|-----|----------|",
        ]
        for t in trajectory:
            period = t.get("period", "-")
            projected = self._format_number(t.get("projected", 0))
            target = self._format_number(t.get("target", 0))
            gap = self._format_number(t.get("gap", 0))
            on_track = "Yes" if t.get("on_track", False) else "No"
            lines.append(
                f"| {period} | {projected} | {target} | {gap} | {on_track} |"
            )

        return "\n".join(lines)

    def _render_md_peer_benchmark(self, data: Dict[str, Any]) -> str:
        """Render peer benchmarking comparison."""
        benchmark = data.get("peer_benchmarking", {})
        if not benchmark:
            return "## Peer Benchmarking\n\n_No benchmark data available._"

        company = benchmark.get("company_name", "Company")
        metrics: List[Dict[str, Any]] = benchmark.get("metrics", [])

        lines = [
            "## Peer Benchmarking",
            "",
            f"**Company:** {company}",
            "",
            "| Metric | Company | Sector Median | Percentile | Status |",
            "|--------|---------|-------------|-----------|--------|",
        ]
        for m in metrics:
            name = m.get("metric", "-")
            company_val = self._format_number(m.get("company_value", 0))
            median = self._format_number(m.get("sector_median", 0))
            percentile = m.get("percentile", 0)
            status = m.get("status", "-")
            lines.append(
                f"| {name} | {company_val} | {median} | P{percentile} | {status} |"
            )

        return "\n".join(lines)

    def _render_md_board_actions(self, data: Dict[str, Any]) -> str:
        """Render board action items."""
        actions = data.get("board_actions", {})
        if not actions:
            return "## Board Action Items\n\n_No action items._"

        lines = ["## Board Action Items", ""]

        approvals = actions.get("pending_approvals", [])
        if approvals:
            lines.append("### Pending Approvals")
            lines.append("")
            for a in approvals:
                lines.append(
                    f"- **{a.get('title', '-')}** - Due: {a.get('due_date', '-')} "
                    f"({a.get('priority', 'normal')})"
                )
            lines.append("")

        deadlines = actions.get("upcoming_deadlines", [])
        if deadlines:
            lines.append("### Upcoming Deadlines")
            lines.append("")
            for d in deadlines:
                lines.append(
                    f"- **{d.get('title', '-')}** - {d.get('deadline', '-')} "
                    f"({d.get('days_left', 0)} days)"
                )
            lines.append("")

        alerts = actions.get("critical_alerts", [])
        if alerts:
            lines.append("### Critical Alerts")
            lines.append("")
            for al in alerts:
                lines.append(
                    f"- **[{al.get('severity', 'HIGH').upper()}]** "
                    f"{al.get('message', '-')}"
                )

        return "\n".join(lines)

    def _render_md_financial_impact(self, data: Dict[str, Any]) -> str:
        """Render financial impact summary."""
        impact = data.get("financial_impact", {})
        if not impact:
            return "## Financial Impact\n\n_No financial data available._"

        lines = [
            "## Financial Impact Summary",
            "",
            "| Category | Value | Timeframe | Confidence |",
            "|----------|-------|-----------|-----------|",
            f"| Carbon Pricing Risk | ${self._format_number(impact.get('carbon_pricing_risk', 0))} | "
            f"{impact.get('carbon_pricing_timeframe', '-')} | "
            f"{self._format_percentage(impact.get('carbon_pricing_confidence', 0))} |",
            f"| Regulatory Exposure | ${self._format_number(impact.get('regulatory_exposure', 0))} | "
            f"{impact.get('regulatory_timeframe', '-')} | "
            f"{self._format_percentage(impact.get('regulatory_confidence', 0))} |",
            f"| Opportunity Value | ${self._format_number(impact.get('opportunity_value', 0))} | "
            f"{impact.get('opportunity_timeframe', '-')} | "
            f"{self._format_percentage(impact.get('opportunity_confidence', 0))} |",
        ]
        return "\n".join(lines)

    def _render_md_sparklines(self, data: Dict[str, Any]) -> str:
        """Render 90-day trend sparklines as text summaries."""
        sparklines = data.get("sparklines", {})
        if not sparklines:
            return "## 90-Day Trends\n\n_No sparkline data available._"

        lines = [
            "## 90-Day Trend Sparklines",
            "",
            "| KPI | 90-Day Min | 90-Day Max | Current | Trend Direction |",
            "|-----|-----------|-----------|---------|----------------|",
        ]
        for kpi_name, spark_data in sparklines.items():
            values = spark_data.get("values", [])
            if not values:
                continue
            min_val = min(values)
            max_val = max(values)
            current = values[-1] if values else 0
            direction = "Up" if len(values) > 1 and values[-1] > values[0] else "Down"
            lines.append(
                f"| {kpi_name} | {self._format_number(min_val)} | "
                f"{self._format_number(max_val)} | {self._format_number(current)} | "
                f"{direction} |"
            )

        return "\n".join(lines)

    def _render_md_quick_links(self, data: Dict[str, Any]) -> str:
        """Render quick links to detailed reports."""
        links: List[Dict[str, Any]] = data.get("quick_links", [])
        if not links:
            return "## Quick Links\n\n_No quick links configured._"

        lines = ["## Quick Links to Detailed Reports", ""]
        for link in links:
            title = link.get("title", "Report")
            url = link.get("url", "#")
            desc = link.get("description", "")
            lines.append(f"- [{title}]({url}){': ' + desc if desc else ''}")

        return "\n".join(lines)

    def _render_md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        ts = self._format_date(self.generated_at)
        return f"---\n_Executive Cockpit generated at {ts} | PACK-003 CSRD Enterprise_"

    # ------------------------------------------------------------------
    # HTML renderers
    # ------------------------------------------------------------------

    def _build_css(self) -> str:
        """Build inline CSS for executive cockpit."""
        return """
:root {
    --primary: #111827; --primary-accent: #1a56db; --success: #057a55;
    --warning: #e3a008; --danger: #e02424; --info: #1c64f2;
    --bg: #f9fafb; --card-bg: #fff; --text: #111827;
    --text-muted: #6b7280; --border: #e5e7eb;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --gold: #d4af37;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); }
.cockpit-container { max-width: 1400px; margin: 0 auto; padding: 24px; }
.cockpit-header { background: linear-gradient(135deg, #111827, #1e3a5f);
    color: #fff; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px;
    display: flex; justify-content: space-between; align-items: center; }
.cockpit-header h1 { font-size: 26px; }
.cockpit-header .subtitle { opacity: 0.75; font-size: 14px; }
.cockpit-header .live-badge { background: #10b981; color: #fff; padding: 4px 12px;
    border-radius: 20px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; }
.section { margin-bottom: 24px; background: var(--card-bg); border-radius: 10px;
    padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.section-title { font-size: 18px; font-weight: 600; color: var(--primary);
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid var(--primary-accent); }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; }
.kpi-card { background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; position: relative;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.kpi-card .kpi-label { font-size: 12px; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-card .kpi-value { font-size: 28px; font-weight: 700; margin: 4px 0;
    color: var(--primary); }
.kpi-card .kpi-trend { font-size: 12px; font-weight: 600; }
.kpi-card .kpi-trend.positive { color: var(--success); }
.kpi-card .kpi-trend.negative { color: var(--danger); }
.kpi-card .kpi-trend.neutral { color: var(--text-muted); }
.kpi-card .kpi-sparkline { margin-top: 8px; height: 24px; display: flex;
    align-items: flex-end; gap: 1px; }
.kpi-card .spark-bar { width: 3px; background: var(--primary-accent); border-radius: 1px;
    min-height: 2px; }
.kpi-card .kpi-status { position: absolute; top: 12px; right: 12px; width: 10px;
    height: 10px; border-radius: 50%; }
.kpi-card .kpi-status.good { background: var(--success); }
.kpi-card .kpi-status.warning { background: var(--warning); }
.kpi-card .kpi-status.critical { background: var(--danger); }
table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
th { background: #f1f5f9; color: var(--primary); padding: 10px 12px;
    text-align: left; font-size: 12px; font-weight: 600; }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
tr:hover { background: #f8fafc; }
.radar-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; }
.radar-item { border: 1px solid var(--border); border-radius: 8px; padding: 14px;
    text-align: center; }
.radar-item .radar-dimension { font-size: 12px; font-weight: 600; color: var(--primary);
    margin-bottom: 6px; }
.radar-item .radar-score { font-size: 28px; font-weight: 700; }
.radar-item .radar-level { font-size: 11px; margin-top: 4px; font-weight: 600; }
.radar-item .radar-bar { height: 6px; background: #e5e7eb; border-radius: 3px;
    margin-top: 6px; overflow: hidden; }
.radar-item .radar-fill { height: 100%; border-radius: 3px; }
.radar-fill.low { background: var(--success); }
.radar-fill.medium { background: var(--warning); }
.radar-fill.high { background: #f59e0b; }
.radar-fill.critical { background: var(--danger); }
.action-list { list-style: none; }
.action-item { display: flex; align-items: center; padding: 10px 0;
    border-bottom: 1px solid var(--border); }
.action-item:last-child { border-bottom: none; }
.action-icon { width: 32px; height: 32px; border-radius: 6px; display: flex;
    align-items: center; justify-content: center; font-size: 14px; font-weight: 700;
    margin-right: 12px; flex-shrink: 0; }
.action-icon.approval { background: #dbeafe; color: var(--primary-accent); }
.action-icon.deadline { background: #fef9c3; color: #92400e; }
.action-icon.alert { background: #fde8e8; color: var(--danger); }
.action-content { flex: 1; }
.action-title { font-size: 13px; font-weight: 500; }
.action-meta { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.action-priority { font-size: 10px; padding: 2px 8px; border-radius: 10px;
    font-weight: 700; text-transform: uppercase; }
.action-priority.high { background: #fde8e8; color: var(--danger); }
.action-priority.medium { background: #fef9c3; color: #92400e; }
.action-priority.normal { background: #d1fae5; color: var(--success); }
.impact-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.impact-card { border: 1px solid var(--border); border-radius: 10px; padding: 20px;
    text-align: center; }
.impact-card.risk { border-top: 3px solid var(--danger); }
.impact-card.exposure { border-top: 3px solid var(--warning); }
.impact-card.opportunity { border-top: 3px solid var(--success); }
.impact-card .impact-value { font-size: 24px; font-weight: 700; }
.impact-card .impact-label { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
.impact-card .impact-confidence { font-size: 11px; color: var(--text-muted); margin-top: 6px; }
.quick-links { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px; }
.quick-link { display: block; padding: 14px 16px; border: 1px solid var(--border);
    border-radius: 8px; text-decoration: none; color: var(--primary);
    font-weight: 500; font-size: 13px; transition: all 0.2s; }
.quick-link:hover { background: var(--primary-accent); color: #fff;
    border-color: var(--primary-accent); }
.quick-link .ql-desc { font-size: 11px; color: var(--text-muted); margin-top: 2px;
    font-weight: 400; }
.quick-link:hover .ql-desc { color: rgba(255,255,255,0.8); }
.benchmark-row .company-val { font-weight: 600; }
.benchmark-row .above-median { color: var(--success); }
.benchmark-row .below-median { color: var(--danger); }
.footer { text-align: center; color: var(--text-muted); font-size: 12px;
    padding: 16px 0; margin-top: 24px; border-top: 1px solid var(--border); }
"""

    def _render_html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        title = self._escape_html(data.get("title", "Executive Cockpit"))
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"cockpit-header\">\n"
            f"  <div>\n"
            f"    <h1>{title}</h1>\n"
            f"    <div class=\"subtitle\">Generated: {ts}</div>\n"
            f"  </div>\n"
            f"  <span class=\"live-badge\">Live Dashboard</span>\n"
            f"</div>"
        )

    def _render_html_kpis(self, data: Dict[str, Any]) -> str:
        """Render HTML KPI cards with sparklines."""
        kpis = data.get("kpis", {})
        sparklines = data.get("sparklines", {})
        if not kpis:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Key Performance Indicators</h2>\n"
                "  <p>No KPI data available.</p>\n</div>"
            )

        cards = ""
        for kpi_def in self.KPI_DEFINITIONS:
            key = kpi_def["key"]
            label = kpi_def["label"]
            kpi_data = kpis.get(key, {})
            value = kpi_data.get("value", 0)
            trend = kpi_data.get("trend_90d", 0)
            status = kpi_data.get("status", "good")

            if kpi_def["format"] == "percentage":
                formatted = self._format_percentage(value)
            elif kpi_def["format"] == "currency":
                formatted = f"${self._format_number(value)}"
            elif kpi_def["format"] == "text":
                formatted = str(value)
            else:
                formatted = self._format_number(value)

            trend_cls = (
                "positive" if trend < 0 and key == "total_emissions"
                else "positive" if trend > 0 and key != "total_emissions"
                else "negative" if trend != 0
                else "neutral"
            )
            trend_arrow = "v" if trend < 0 else ("^" if trend > 0 else "-")
            trend_str = f"{trend_arrow} {self._format_percentage(abs(trend))}"

            sparkline_html = self._build_sparkline_html(sparklines.get(key, {}))

            cards += (
                f"<div class=\"kpi-card\">\n"
                f"  <div class=\"kpi-status {status}\"></div>\n"
                f"  <div class=\"kpi-label\">{label}</div>\n"
                f"  <div class=\"kpi-value\">{formatted}</div>\n"
                f"  <div class=\"kpi-trend {trend_cls}\">{trend_str}</div>\n"
                f"  {sparkline_html}\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Key Performance Indicators</h2>\n"
            f"  <div class=\"kpi-grid\">{cards}</div>\n"
            "</div>"
        )

    def _render_html_risk_radar(self, data: Dict[str, Any]) -> str:
        """Render HTML risk exposure radar."""
        radar: List[Dict[str, Any]] = data.get("risk_radar", [])
        if not radar:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Risk Exposure</h2>\n"
                "  <p>No risk radar data available.</p>\n</div>"
            )

        items = ""
        for r in radar:
            score = r.get("score", 0)
            level = r.get("level", "low").lower()
            dimension = self._escape_html(r.get("dimension", "-"))
            trend = r.get("trend", "-")
            items += (
                f"<div class=\"radar-item\">\n"
                f"  <div class=\"radar-dimension\">{dimension}</div>\n"
                f"  <div class=\"radar-score\">{self._format_number(score, 0)}</div>\n"
                f"  <div class=\"radar-level\" style=\"color:"
                f"{'#e02424' if level == 'critical' else '#f59e0b' if level == 'high' else '#e3a008' if level == 'medium' else '#057a55'}\">"
                f"{level.upper()}</div>\n"
                f"  <div class=\"radar-bar\">"
                f"<div class=\"radar-fill {level}\" style=\"width:{score}%\"></div></div>\n"
                f"  <div style=\"font-size:11px;color:#6b7280;margin-top:4px\">"
                f"Trend: {trend}</div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Risk Exposure Radar</h2>\n"
            f"  <div class=\"radar-grid\">{items}</div>\n"
            "</div>"
        )

    def _render_html_trajectory(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance trajectory."""
        trajectory: List[Dict[str, Any]] = data.get("trajectory", [])
        if not trajectory:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Compliance Trajectory</h2>\n"
                "  <p>No trajectory data available.</p>\n</div>"
            )

        rows = ""
        for t in trajectory:
            on_track = t.get("on_track", False)
            track_cls = "above-median" if on_track else "below-median"
            rows += (
                f"<tr><td>{t.get('period', '-')}</td>"
                f"<td>{self._format_number(t.get('projected', 0))}</td>"
                f"<td>{self._format_number(t.get('target', 0))}</td>"
                f"<td class=\"{track_cls}\">{self._format_number(t.get('gap', 0))}</td>"
                f"<td class=\"{track_cls}\">"
                f"{'On Track' if on_track else 'Off Track'}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Compliance Trajectory</h2>\n"
            "  <table><thead><tr>"
            "<th>Period</th><th>Projected</th><th>Target</th>"
            "<th>Gap</th><th>Status</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_peer_benchmark(self, data: Dict[str, Any]) -> str:
        """Render HTML peer benchmarking."""
        benchmark = data.get("peer_benchmarking", {})
        if not benchmark:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Peer Benchmarking</h2>\n"
                "  <p>No benchmark data available.</p>\n</div>"
            )

        company = self._escape_html(benchmark.get("company_name", "Company"))
        metrics: List[Dict[str, Any]] = benchmark.get("metrics", [])

        rows = ""
        for m in metrics:
            company_val = m.get("company_value", 0)
            median = m.get("sector_median", 0)
            above = company_val >= median
            cls = "above-median" if above else "below-median"
            rows += (
                f"<tr class=\"benchmark-row\">"
                f"<td>{self._escape_html(m.get('metric', '-'))}</td>"
                f"<td class=\"company-val {cls}\">"
                f"{self._format_number(company_val)}</td>"
                f"<td>{self._format_number(median)}</td>"
                f"<td>P{m.get('percentile', 0)}</td>"
                f"<td>{m.get('status', '-')}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            f"  <h2 class=\"section-title\">Peer Benchmarking - {company}</h2>\n"
            "  <table><thead><tr>"
            "<th>Metric</th><th>Company</th><th>Sector Median</th>"
            "<th>Percentile</th><th>Status</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_board_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML board action items."""
        actions = data.get("board_actions", {})
        if not actions:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Board Action Items</h2>\n"
                "  <p>No action items.</p>\n</div>"
            )

        items_html = "<ul class=\"action-list\">\n"

        for a in actions.get("pending_approvals", []):
            priority = a.get("priority", "normal").lower()
            items_html += (
                f"<li class=\"action-item\">\n"
                f"  <div class=\"action-icon approval\">A</div>\n"
                f"  <div class=\"action-content\">\n"
                f"    <div class=\"action-title\">"
                f"{self._escape_html(a.get('title', '-'))}</div>\n"
                f"    <div class=\"action-meta\">Due: {a.get('due_date', '-')}</div>\n"
                f"  </div>\n"
                f"  <span class=\"action-priority {priority}\">{priority}</span>\n"
                f"</li>\n"
            )

        for d in actions.get("upcoming_deadlines", []):
            days = d.get("days_left", 0)
            urgency = "high" if days <= 7 else "medium" if days <= 30 else "normal"
            items_html += (
                f"<li class=\"action-item\">\n"
                f"  <div class=\"action-icon deadline\">D</div>\n"
                f"  <div class=\"action-content\">\n"
                f"    <div class=\"action-title\">"
                f"{self._escape_html(d.get('title', '-'))}</div>\n"
                f"    <div class=\"action-meta\">{d.get('deadline', '-')} "
                f"({days} days)</div>\n"
                f"  </div>\n"
                f"  <span class=\"action-priority {urgency}\">{urgency}</span>\n"
                f"</li>\n"
            )

        for al in actions.get("critical_alerts", []):
            items_html += (
                f"<li class=\"action-item\">\n"
                f"  <div class=\"action-icon alert\">!</div>\n"
                f"  <div class=\"action-content\">\n"
                f"    <div class=\"action-title\">"
                f"{self._escape_html(al.get('message', '-'))}</div>\n"
                f"    <div class=\"action-meta\">"
                f"{al.get('severity', 'HIGH').upper()}</div>\n"
                f"  </div>\n"
                f"</li>\n"
            )

        items_html += "</ul>\n"

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Board Action Items</h2>\n"
            f"  {items_html}\n"
            "</div>"
        )

    def _render_html_financial_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML financial impact summary."""
        impact = data.get("financial_impact", {})
        if not impact:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Financial Impact</h2>\n"
                "  <p>No financial data available.</p>\n</div>"
            )

        cards = [
            {
                "cls": "risk",
                "label": "Carbon Pricing Risk",
                "value": f"${self._format_number(impact.get('carbon_pricing_risk', 0))}",
                "timeframe": impact.get("carbon_pricing_timeframe", "-"),
                "confidence": impact.get("carbon_pricing_confidence", 0),
            },
            {
                "cls": "exposure",
                "label": "Regulatory Exposure",
                "value": f"${self._format_number(impact.get('regulatory_exposure', 0))}",
                "timeframe": impact.get("regulatory_timeframe", "-"),
                "confidence": impact.get("regulatory_confidence", 0),
            },
            {
                "cls": "opportunity",
                "label": "Opportunity Value",
                "value": f"${self._format_number(impact.get('opportunity_value', 0))}",
                "timeframe": impact.get("opportunity_timeframe", "-"),
                "confidence": impact.get("opportunity_confidence", 0),
            },
        ]

        cards_html = ""
        for c in cards:
            cards_html += (
                f"<div class=\"impact-card {c['cls']}\">\n"
                f"  <div class=\"impact-value\">{c['value']}</div>\n"
                f"  <div class=\"impact-label\">{c['label']}</div>\n"
                f"  <div class=\"impact-confidence\">{c['timeframe']} | "
                f"Confidence: {self._format_percentage(c['confidence'])}</div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Financial Impact Summary</h2>\n"
            f"  <div class=\"impact-grid\">{cards_html}</div>\n"
            "</div>"
        )

    def _render_html_sparklines(self, data: Dict[str, Any]) -> str:
        """Render HTML 90-day sparklines summary table."""
        sparklines = data.get("sparklines", {})
        if not sparklines:
            return ""

        rows = ""
        for kpi_name, spark_data in sparklines.items():
            values = spark_data.get("values", [])
            if not values:
                continue
            min_val = min(values)
            max_val = max(values)
            current = values[-1]
            direction = "Up" if len(values) > 1 and values[-1] > values[0] else "Down"
            sparkline = self._build_sparkline_html(spark_data)
            rows += (
                f"<tr><td>{self._escape_html(kpi_name)}</td>"
                f"<td>{self._format_number(min_val)}</td>"
                f"<td>{self._format_number(max_val)}</td>"
                f"<td><strong>{self._format_number(current)}</strong></td>"
                f"<td>{direction}</td><td>{sparkline}</td></tr>\n"
            )

        if not rows:
            return ""

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">90-Day Trend Sparklines</h2>\n"
            "  <table><thead><tr>"
            "<th>KPI</th><th>Min</th><th>Max</th><th>Current</th>"
            "<th>Direction</th><th>Sparkline</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_quick_links(self, data: Dict[str, Any]) -> str:
        """Render HTML quick links."""
        links: List[Dict[str, Any]] = data.get("quick_links", [])
        if not links:
            return ""

        link_cards = ""
        for link in links:
            title = self._escape_html(link.get("title", "Report"))
            url = link.get("url", "#")
            desc = self._escape_html(link.get("description", ""))
            desc_html = f"<div class=\"ql-desc\">{desc}</div>" if desc else ""
            link_cards += (
                f"<a class=\"quick-link\" href=\"{url}\">\n"
                f"  {title}\n"
                f"  {desc_html}\n"
                f"</a>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Quick Links</h2>\n"
            f"  <div class=\"quick-links\">{link_cards}</div>\n"
            "</div>"
        )

    def _render_html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"footer\">"
            f"Executive Cockpit generated at {ts} | PACK-003 CSRD Enterprise"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_sparkline_html(self, spark_data: Dict[str, Any]) -> str:
        """Build an inline sparkline from data values.

        Args:
            spark_data: Dict with 'values' list.

        Returns:
            HTML string with sparkline bars.
        """
        values = spark_data.get("values", [])
        if not values:
            return ""

        max_val = max(values) if values else 1
        bars = ""
        for v in values[-30:]:
            height = (v / max_val * 24) if max_val else 2
            bars += f"<div class=\"spark-bar\" style=\"height:{max(2, height):.0f}px\"></div>"

        return f"<div class=\"kpi-sparkline\">{bars}</div>"

    @staticmethod
    def _generate_provenance_hash(content: str) -> str:
        """Generate SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_number(value: Union[int, float], decimals: int = 2) -> str:
        """Format numeric value with thousands separator."""
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def _format_percentage(value: Union[int, float]) -> str:
        """Format value as percentage."""
        return f"{value:.1f}%"

    @staticmethod
    def _format_date(dt: Optional[datetime]) -> str:
        """Format datetime as string."""
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    @staticmethod
    def _text_bar(value: float, max_val: float = 100.0, width: int = 15) -> str:
        """Create a text-based bar for Markdown tables."""
        filled = int((value / max_val) * width) if max_val else 0
        return "|" + "=" * filled + " " * (width - filled) + "|"
