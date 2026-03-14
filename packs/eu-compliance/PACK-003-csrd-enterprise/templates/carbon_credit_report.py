"""
CarbonCreditReportTemplate - Carbon credit portfolio report for CSRD Enterprise Pack.

This module implements the carbon credit portfolio report template with
portfolio summary, vintage breakdown, credit quality distribution,
net-zero waterfall, retirement schedule, price trends, project map,
and SBTi compliance notes.

Example:
    >>> template = CarbonCreditReportTemplate()
    >>> data = {"portfolio": {...}, "vintage_breakdown": [...]}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CarbonCreditReportTemplate:
    """
    Carbon credit portfolio report template.

    Renders portfolio summaries, vintage breakdowns, quality distributions,
    net-zero waterfall charts, retirement schedules, price trends,
    project maps, and SBTi compliance notes.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    REGISTRIES = ["Verra (VCS)", "Gold Standard", "ACR", "CAR", "Plan Vivo", "Other"]

    CREDIT_TYPES = [
        "Forestry & Land Use", "Renewable Energy", "Energy Efficiency",
        "Methane Avoidance", "Direct Air Capture", "Blue Carbon", "Other",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CarbonCreditReportTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render carbon credit report as Markdown.

        Args:
            data: Report data with portfolio, vintage_breakdown,
                  quality_distribution, waterfall, retirement_schedule,
                  price_trends, project_map, sbti_note.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._render_md_header(data))
        sections.append(self._render_md_portfolio_summary(data))
        sections.append(self._render_md_vintage_breakdown(data))
        sections.append(self._render_md_quality_distribution(data))
        sections.append(self._render_md_waterfall(data))
        sections.append(self._render_md_retirement_schedule(data))
        sections.append(self._render_md_price_trends(data))
        sections.append(self._render_md_project_map(data))
        sections.append(self._render_md_sbti_note(data))
        sections.append(self._render_md_footer(data))

        content = "\n\n".join(sections)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- Provenance: {provenance} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render carbon credit report as self-contained HTML.

        Args:
            data: Report data dict.

        Returns:
            Complete HTML string with inline styles.
        """
        self.generated_at = datetime.utcnow()
        css = self._build_css()
        body_parts: List[str] = []

        body_parts.append(self._render_html_header(data))
        body_parts.append(self._render_html_portfolio_summary(data))
        body_parts.append(self._render_html_vintage_breakdown(data))
        body_parts.append(self._render_html_quality_distribution(data))
        body_parts.append(self._render_html_waterfall(data))
        body_parts.append(self._render_html_retirement_schedule(data))
        body_parts.append(self._render_html_price_trends(data))
        body_parts.append(self._render_html_project_map(data))
        body_parts.append(self._render_html_sbti_note(data))
        body_parts.append(self._render_html_footer(data))

        body_html = "\n".join(body_parts)
        provenance = self._generate_provenance_hash(body_html)

        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>Carbon Credit Portfolio</title>\n<style>\n{css}\n</style>\n"
            "</head>\n<body>\n"
            f"<div class=\"report-container\">\n{body_html}\n</div>\n"
            f"<!-- Provenance: {provenance} -->\n"
            "</body>\n</html>"
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render carbon credit report as structured JSON.

        Args:
            data: Report data dict.

        Returns:
            Structured dict with all report sections.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "carbon_credit_report",
            "version": "1.0.0",
            "generated_at": self.generated_at.isoformat(),
            "portfolio_summary": self._build_json_portfolio(data),
            "vintage_breakdown": self._build_json_vintage(data),
            "quality_distribution": self._build_json_quality(data),
            "net_zero_waterfall": self._build_json_waterfall(data),
            "retirement_schedule": self._build_json_retirement(data),
            "price_trends": self._build_json_price_trends(data),
            "project_map": self._build_json_project_map(data),
            "sbti_compliance_note": self._build_json_sbti(data),
        }
        provenance = self._generate_provenance_hash(json.dumps(result, default=str))
        result["provenance_hash"] = provenance
        return result

    # ------------------------------------------------------------------
    # Markdown renderers
    # ------------------------------------------------------------------

    def _render_md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        title = data.get("title", "Carbon Credit Portfolio Report")
        ts = self._format_date(self.generated_at)
        return f"# {title}\n\n**Generated:** {ts}\n\n---"

    def _render_md_portfolio_summary(self, data: Dict[str, Any]) -> str:
        """Render portfolio summary."""
        portfolio = data.get("portfolio", {})
        if not portfolio:
            return "## Portfolio Summary\n\n_No portfolio data available._"

        by_registry = portfolio.get("by_registry", {})
        by_type = portfolio.get("by_type", {})

        lines = [
            "## Portfolio Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Credits | {self._format_number(portfolio.get('total_credits', 0), 0)} |",
            f"| Total Value | ${self._format_number(portfolio.get('total_value_usd', 0))} |",
            f"| Avg Price per Credit | ${self._format_number(portfolio.get('avg_price', 0))} |",
            f"| Retired | {self._format_number(portfolio.get('retired_credits', 0), 0)} |",
            f"| Active | {self._format_number(portfolio.get('active_credits', 0), 0)} |",
            "",
            "### By Registry",
            "",
            "| Registry | Credits | % of Total |",
            "|----------|---------|-----------|",
        ]
        total_credits = portfolio.get("total_credits", 1)
        for reg, count in by_registry.items():
            pct = (count / total_credits * 100) if total_credits else 0
            lines.append(
                f"| {reg} | {self._format_number(count, 0)} | "
                f"{self._format_percentage(pct)} |"
            )

        lines.extend([
            "",
            "### By Credit Type",
            "",
            "| Type | Credits | % of Total |",
            "|------|---------|-----------|",
        ])
        for ctype, count in by_type.items():
            pct = (count / total_credits * 100) if total_credits else 0
            lines.append(
                f"| {ctype} | {self._format_number(count, 0)} | "
                f"{self._format_percentage(pct)} |"
            )

        return "\n".join(lines)

    def _render_md_vintage_breakdown(self, data: Dict[str, Any]) -> str:
        """Render vintage breakdown table."""
        vintages: List[Dict[str, Any]] = data.get("vintage_breakdown", [])
        if not vintages:
            return "## Vintage Breakdown\n\n_No vintage data available._"

        lines = [
            "## Vintage Breakdown",
            "",
            "| Vintage Year | Credits | Avg Price | Status |",
            "|-------------|---------|-----------|--------|",
        ]
        for v in vintages:
            year = v.get("year", "-")
            credits = self._format_number(v.get("credits", 0), 0)
            price = f"${self._format_number(v.get('avg_price', 0))}"
            status = v.get("status", "-")
            bar = self._text_bar(v.get("credits", 0), max(
                (vv.get("credits", 0) for vv in vintages), default=1
            ))
            lines.append(f"| {year} | {credits} {bar} | {price} | {status} |")

        return "\n".join(lines)

    def _render_md_quality_distribution(self, data: Dict[str, Any]) -> str:
        """Render credit quality distribution."""
        quality: List[Dict[str, Any]] = data.get("quality_distribution", [])
        if not quality:
            return "## Credit Quality\n\n_No quality data available._"

        lines = [
            "## Credit Quality Distribution (Additionality Scores)",
            "",
            "| Quality Tier | Credits | Additionality Score | % of Portfolio |",
            "|-------------|---------|-------------------|----------------|",
        ]
        for q in quality:
            tier = q.get("tier", "-")
            credits = self._format_number(q.get("credits", 0), 0)
            score = self._format_number(q.get("additionality_score", 0), 1)
            pct = self._format_percentage(q.get("pct_of_portfolio", 0))
            lines.append(f"| {tier} | {credits} | {score}/10 | {pct} |")

        return "\n".join(lines)

    def _render_md_waterfall(self, data: Dict[str, Any]) -> str:
        """Render net-zero accounting waterfall."""
        waterfall = data.get("waterfall", {})
        if not waterfall:
            return "## Net-Zero Waterfall\n\n_No waterfall data available._"

        lines = [
            "## Net-Zero Accounting Waterfall",
            "",
            "| Step | Value (tCO2e) | Running Total |",
            "|------|-------------|--------------|",
        ]
        steps: List[Dict[str, Any]] = waterfall.get("steps", [])
        for step in steps:
            label = step.get("label", "-")
            value = self._format_number(step.get("value", 0))
            running = self._format_number(step.get("running_total", 0))
            lines.append(f"| {label} | {value} | {running} |")

        return "\n".join(lines)

    def _render_md_retirement_schedule(self, data: Dict[str, Any]) -> str:
        """Render retirement schedule timeline."""
        schedule: List[Dict[str, Any]] = data.get("retirement_schedule", [])
        if not schedule:
            return "## Retirement Schedule\n\n_No retirement schedule available._"

        lines = [
            "## Retirement Schedule",
            "",
            "| Date | Credits | Registry | Project | Serial Numbers |",
            "|------|---------|----------|---------|---------------|",
        ]
        for r in schedule:
            date = r.get("retirement_date", "-")
            credits = self._format_number(r.get("credits", 0), 0)
            registry = r.get("registry", "-")
            project = r.get("project_name", "-")
            serials = r.get("serial_range", "-")
            lines.append(
                f"| {date} | {credits} | {registry} | {project} | {serials} |"
            )

        return "\n".join(lines)

    def _render_md_price_trends(self, data: Dict[str, Any]) -> str:
        """Render price trend table."""
        trends: List[Dict[str, Any]] = data.get("price_trends", [])
        if not trends:
            return "## Price Trends\n\n_No price data available._"

        lines = [
            "## Price Trends (Historical + Projected)",
            "",
            "| Period | Avg Price ($/tCO2e) | Volume Traded | Type |",
            "|--------|-------------------|--------------|----|",
        ]
        for t in trends:
            period = t.get("period", "-")
            price = f"${self._format_number(t.get('avg_price', 0))}"
            volume = self._format_number(t.get("volume_traded", 0), 0)
            trend_type = t.get("type", "historical")
            lines.append(f"| {period} | {price} | {volume} | {trend_type} |")

        return "\n".join(lines)

    def _render_md_project_map(self, data: Dict[str, Any]) -> str:
        """Render project map listing."""
        projects: List[Dict[str, Any]] = data.get("project_map", [])
        if not projects:
            return "## Project Locations\n\n_No project location data available._"

        lines = [
            "## Offset Project Locations",
            "",
            "| # | Project | Country | Type | Credits | Lat | Lon |",
            "|---|---------|---------|------|---------|-----|-----|",
        ]
        for idx, p in enumerate(projects, 1):
            name = p.get("name", "-")
            country = p.get("country", "-")
            ptype = p.get("type", "-")
            credits = self._format_number(p.get("credits", 0), 0)
            lat = self._format_number(p.get("latitude", 0), 4)
            lon = self._format_number(p.get("longitude", 0), 4)
            lines.append(
                f"| {idx} | {name} | {country} | {ptype} | {credits} | {lat} | {lon} |"
            )

        return "\n".join(lines)

    def _render_md_sbti_note(self, data: Dict[str, Any]) -> str:
        """Render SBTi compliance note."""
        sbti = data.get("sbti_note", {})
        note_text = sbti.get("note", (
            "In accordance with SBTi guidance, carbon offsets and credits "
            "supplement but do not replace direct emission reductions. "
            "Companies must prioritize absolute emission reductions in line "
            "with science-based targets before applying offsets toward "
            "net-zero claims."
        ))

        lines = [
            "## SBTi Compliance Note",
            "",
            f"> {note_text}",
            "",
            f"**Gross Emissions:** {self._format_number(sbti.get('gross_emissions', 0))} tCO2e",
            f"**Direct Reductions:** {self._format_number(sbti.get('direct_reductions', 0))} tCO2e",
            f"**Credits Applied:** {self._format_number(sbti.get('credits_applied', 0))} tCO2e",
            f"**Net Position:** {self._format_number(sbti.get('net_position', 0))} tCO2e",
        ]
        return "\n".join(lines)

    def _render_md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        ts = self._format_date(self.generated_at)
        return f"---\n_Carbon Credit Report generated at {ts} | PACK-003 CSRD Enterprise_"

    # ------------------------------------------------------------------
    # HTML renderers
    # ------------------------------------------------------------------

    def _build_css(self) -> str:
        """Build inline CSS for carbon credit report."""
        return """
:root {
    --primary: #1a56db; --primary-light: #e1effe; --success: #057a55;
    --warning: #e3a008; --danger: #e02424; --bg: #f9fafb;
    --card-bg: #fff; --text: #1f2937; --text-muted: #6b7280;
    --border: #e5e7eb;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --green-gradient: linear-gradient(135deg, #057a55, #10b981);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); }
.report-container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.report-header { background: var(--green-gradient); color: #fff;
    padding: 28px 32px; border-radius: 12px; margin-bottom: 24px; }
.report-header h1 { font-size: 26px; }
.report-header .subtitle { opacity: 0.85; margin-top: 4px; font-size: 14px; }
.section { margin-bottom: 24px; background: var(--card-bg); border-radius: 10px;
    padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.section-title { font-size: 18px; font-weight: 600; color: var(--primary);
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid var(--primary); }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 16px; }
.kpi-card { border: 1px solid var(--border); border-radius: 8px; padding: 16px;
    text-align: center; border-top: 3px solid var(--success); }
.kpi-card .kpi-value { font-size: 24px; font-weight: 700; color: var(--success); }
.kpi-card .kpi-label { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
th { background: var(--primary-light); color: var(--primary); padding: 10px 12px;
    text-align: left; font-size: 12px; font-weight: 600; }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
tr:hover { background: #f3f4f6; }
.vintage-bar { display: inline-block; height: 22px; background: var(--success);
    border-radius: 4px; vertical-align: middle; min-width: 4px;
    transition: width 0.3s; }
.quality-tier { display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; }
.quality-tier.high { background: #d1fae5; color: #057a55; }
.quality-tier.medium { background: #fef9c3; color: #92400e; }
.quality-tier.low { background: #fde8e8; color: #e02424; }
.waterfall { margin-bottom: 16px; }
.waterfall-step { display: flex; align-items: center; margin-bottom: 6px; }
.waterfall-label { width: 200px; font-size: 13px; font-weight: 500; }
.waterfall-bar-container { flex: 1; height: 28px; position: relative; }
.waterfall-bar { height: 100%; border-radius: 4px; position: absolute;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; color: #fff; }
.waterfall-bar.positive { background: #e02424; }
.waterfall-bar.negative { background: #057a55; }
.waterfall-bar.total { background: var(--primary); }
.waterfall-value { width: 120px; text-align: right; font-size: 13px; font-weight: 600;
    padding-left: 8px; }
.project-card { display: inline-block; width: 200px; border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; margin: 4px; vertical-align: top; }
.project-card .project-name { font-weight: 600; font-size: 13px; margin-bottom: 4px; }
.project-card .project-meta { font-size: 11px; color: var(--text-muted); }
.sbti-notice { background: #fffbeb; border: 1px solid #fcd34d; border-radius: 10px;
    padding: 20px; border-left: 4px solid #e3a008; }
.sbti-notice h3 { color: #92400e; margin-bottom: 8px; }
.sbti-notice .note-text { font-size: 14px; line-height: 1.7; margin-bottom: 12px; }
.sbti-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 10px; }
.sbti-metric { text-align: center; }
.sbti-metric .metric-val { font-size: 18px; font-weight: 700; }
.sbti-metric .metric-lbl { font-size: 11px; color: var(--text-muted); }
.timeline-item { display: flex; align-items: flex-start; margin-bottom: 12px;
    padding-bottom: 12px; border-bottom: 1px solid var(--border); }
.timeline-date { width: 100px; font-size: 12px; font-weight: 600; color: var(--primary);
    flex-shrink: 0; }
.timeline-content { flex: 1; }
.timeline-content .tl-title { font-weight: 500; font-size: 13px; }
.timeline-content .tl-meta { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.footer { text-align: center; color: var(--text-muted); font-size: 12px;
    padding: 16px 0; margin-top: 24px; border-top: 1px solid var(--border); }
"""

    def _render_html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        title = self._escape_html(data.get("title", "Carbon Credit Portfolio Report"))
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"report-header\">\n"
            f"  <h1>{title}</h1>\n"
            f"  <div class=\"subtitle\">Generated: {ts}</div>\n"
            f"</div>"
        )

    def _render_html_portfolio_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML portfolio summary with KPI cards and breakdown tables."""
        portfolio = data.get("portfolio", {})
        if not portfolio:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Portfolio Summary</h2>\n"
                "  <p>No portfolio data available.</p>\n</div>"
            )

        kpis = [
            ("Total Credits", self._format_number(portfolio.get("total_credits", 0), 0)),
            ("Total Value", f"${self._format_number(portfolio.get('total_value_usd', 0))}"),
            ("Avg Price", f"${self._format_number(portfolio.get('avg_price', 0))}"),
            ("Retired", self._format_number(portfolio.get("retired_credits", 0), 0)),
            ("Active", self._format_number(portfolio.get("active_credits", 0), 0)),
        ]
        kpi_html = ""
        for label, value in kpis:
            kpi_html += (
                f"<div class=\"kpi-card\">\n"
                f"  <div class=\"kpi-value\">{value}</div>\n"
                f"  <div class=\"kpi-label\">{label}</div>\n"
                f"</div>\n"
            )

        total_credits = portfolio.get("total_credits", 1)
        registry_rows = ""
        for reg, count in portfolio.get("by_registry", {}).items():
            pct = (count / total_credits * 100) if total_credits else 0
            registry_rows += (
                f"<tr><td>{self._escape_html(reg)}</td>"
                f"<td>{self._format_number(count, 0)}</td>"
                f"<td>{self._format_percentage(pct)}</td></tr>\n"
            )

        type_rows = ""
        for ctype, count in portfolio.get("by_type", {}).items():
            pct = (count / total_credits * 100) if total_credits else 0
            type_rows += (
                f"<tr><td>{self._escape_html(ctype)}</td>"
                f"<td>{self._format_number(count, 0)}</td>"
                f"<td>{self._format_percentage(pct)}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Portfolio Summary</h2>\n"
            f"  <div class=\"kpi-grid\">{kpi_html}</div>\n"
            "  <div style=\"display:grid;grid-template-columns:1fr 1fr;gap:16px\">\n"
            "    <div><h3 style=\"font-size:14px;margin-bottom:8px\">By Registry</h3>\n"
            "    <table><thead><tr><th>Registry</th><th>Credits</th><th>%</th></tr></thead>\n"
            f"    <tbody>{registry_rows}</tbody></table></div>\n"
            "    <div><h3 style=\"font-size:14px;margin-bottom:8px\">By Type</h3>\n"
            "    <table><thead><tr><th>Type</th><th>Credits</th><th>%</th></tr></thead>\n"
            f"    <tbody>{type_rows}</tbody></table></div>\n"
            "  </div>\n"
            "</div>"
        )

    def _render_html_vintage_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML vintage breakdown with bar indicators."""
        vintages: List[Dict[str, Any]] = data.get("vintage_breakdown", [])
        if not vintages:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Vintage Breakdown</h2>\n"
                "  <p>No vintage data available.</p>\n</div>"
            )

        max_credits = max((v.get("credits", 0) for v in vintages), default=1)
        rows = ""
        for v in vintages:
            year = v.get("year", "-")
            credits = v.get("credits", 0)
            price = v.get("avg_price", 0)
            status = v.get("status", "-")
            bar_width = (credits / max_credits * 200) if max_credits else 0
            rows += (
                f"<tr><td>{year}</td>"
                f"<td>{self._format_number(credits, 0)} "
                f"<span class=\"vintage-bar\" style=\"width:{bar_width:.0f}px\"></span></td>"
                f"<td>${self._format_number(price)}</td>"
                f"<td>{status}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Vintage Breakdown</h2>\n"
            "  <table><thead><tr><th>Vintage</th><th>Credits</th>"
            "<th>Avg Price</th><th>Status</th></tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_quality_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML credit quality distribution."""
        quality: List[Dict[str, Any]] = data.get("quality_distribution", [])
        if not quality:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Credit Quality</h2>\n"
                "  <p>No quality data available.</p>\n</div>"
            )

        rows = ""
        for q in quality:
            tier = q.get("tier", "-")
            tier_cls = self._quality_class(q.get("additionality_score", 0))
            rows += (
                f"<tr><td><span class=\"quality-tier {tier_cls}\">{tier}</span></td>"
                f"<td>{self._format_number(q.get('credits', 0), 0)}</td>"
                f"<td>{self._format_number(q.get('additionality_score', 0), 1)}/10</td>"
                f"<td>{self._format_percentage(q.get('pct_of_portfolio', 0))}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Credit Quality Distribution</h2>\n"
            "  <table><thead><tr><th>Tier</th><th>Credits</th>"
            "<th>Additionality</th><th>% Portfolio</th></tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_waterfall(self, data: Dict[str, Any]) -> str:
        """Render HTML net-zero waterfall chart."""
        waterfall = data.get("waterfall", {})
        if not waterfall:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Net-Zero Waterfall</h2>\n"
                "  <p>No waterfall data available.</p>\n</div>"
            )

        steps: List[Dict[str, Any]] = waterfall.get("steps", [])
        max_val = max(
            (abs(s.get("running_total", 0)) for s in steps), default=1
        )

        bars_html = ""
        for step in steps:
            label = self._escape_html(step.get("label", "-"))
            value = step.get("value", 0)
            running = step.get("running_total", 0)
            step_type = step.get("type", "adjustment")
            bar_width = (abs(running) / max_val * 80) if max_val else 0

            if step_type == "total":
                bar_cls = "total"
            elif value < 0:
                bar_cls = "negative"
            else:
                bar_cls = "positive"

            bars_html += (
                f"<div class=\"waterfall-step\">\n"
                f"  <div class=\"waterfall-label\">{label}</div>\n"
                f"  <div class=\"waterfall-bar-container\">"
                f"<div class=\"waterfall-bar {bar_cls}\" "
                f"style=\"width:{bar_width:.1f}%\">"
                f"{self._format_number(abs(value))}</div></div>\n"
                f"  <div class=\"waterfall-value\">"
                f"{self._format_number(running)} tCO2e</div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Net-Zero Accounting Waterfall</h2>\n"
            f"  <div class=\"waterfall\">{bars_html}</div>\n"
            "</div>"
        )

    def _render_html_retirement_schedule(self, data: Dict[str, Any]) -> str:
        """Render HTML retirement schedule timeline."""
        schedule: List[Dict[str, Any]] = data.get("retirement_schedule", [])
        if not schedule:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Retirement Schedule</h2>\n"
                "  <p>No retirement schedule available.</p>\n</div>"
            )

        items = ""
        for r in schedule:
            date = r.get("retirement_date", "-")
            credits = self._format_number(r.get("credits", 0), 0)
            registry = self._escape_html(r.get("registry", "-"))
            project = self._escape_html(r.get("project_name", "-"))
            serials = r.get("serial_range", "-")
            items += (
                f"<div class=\"timeline-item\">\n"
                f"  <div class=\"timeline-date\">{date}</div>\n"
                f"  <div class=\"timeline-content\">\n"
                f"    <div class=\"tl-title\">{project} - {credits} credits</div>\n"
                f"    <div class=\"tl-meta\">{registry} | Serials: {serials}</div>\n"
                f"  </div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Retirement Schedule</h2>\n"
            f"  {items}\n"
            "</div>"
        )

    def _render_html_price_trends(self, data: Dict[str, Any]) -> str:
        """Render HTML price trends table."""
        trends: List[Dict[str, Any]] = data.get("price_trends", [])
        if not trends:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Price Trends</h2>\n"
                "  <p>No price data available.</p>\n</div>"
            )

        rows = ""
        for t in trends:
            trend_type = t.get("type", "historical")
            style = "font-style:italic;color:#6b7280" if trend_type == "projected" else ""
            rows += (
                f"<tr style=\"{style}\"><td>{t.get('period', '-')}</td>"
                f"<td>${self._format_number(t.get('avg_price', 0))}</td>"
                f"<td>{self._format_number(t.get('volume_traded', 0), 0)}</td>"
                f"<td>{trend_type}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Price Trends</h2>\n"
            "  <table><thead><tr><th>Period</th><th>Avg Price</th>"
            "<th>Volume</th><th>Type</th></tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_project_map(self, data: Dict[str, Any]) -> str:
        """Render HTML project map listing."""
        projects: List[Dict[str, Any]] = data.get("project_map", [])
        if not projects:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Project Locations</h2>\n"
                "  <p>No project data available.</p>\n</div>"
            )

        cards = ""
        for p in projects:
            name = self._escape_html(p.get("name", "-"))
            country = self._escape_html(p.get("country", "-"))
            ptype = p.get("type", "-")
            credits = self._format_number(p.get("credits", 0), 0)
            lat = self._format_number(p.get("latitude", 0), 4)
            lon = self._format_number(p.get("longitude", 0), 4)
            cards += (
                f"<div class=\"project-card\">\n"
                f"  <div class=\"project-name\">{name}</div>\n"
                f"  <div class=\"project-meta\">{country} | {ptype}</div>\n"
                f"  <div class=\"project-meta\">{credits} credits</div>\n"
                f"  <div class=\"project-meta\">({lat}, {lon})</div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Offset Project Locations</h2>\n"
            f"  {cards}\n"
            "</div>"
        )

    def _render_html_sbti_note(self, data: Dict[str, Any]) -> str:
        """Render HTML SBTi compliance note."""
        sbti = data.get("sbti_note", {})
        note_text = sbti.get("note", (
            "In accordance with SBTi guidance, carbon offsets and credits "
            "supplement but do not replace direct emission reductions. "
            "Companies must prioritize absolute emission reductions in line "
            "with science-based targets before applying offsets toward "
            "net-zero claims."
        ))

        metrics = [
            ("Gross Emissions", f"{self._format_number(sbti.get('gross_emissions', 0))} tCO2e"),
            ("Direct Reductions", f"{self._format_number(sbti.get('direct_reductions', 0))} tCO2e"),
            ("Credits Applied", f"{self._format_number(sbti.get('credits_applied', 0))} tCO2e"),
            ("Net Position", f"{self._format_number(sbti.get('net_position', 0))} tCO2e"),
        ]
        metrics_html = ""
        for label, value in metrics:
            metrics_html += (
                f"<div class=\"sbti-metric\">\n"
                f"  <div class=\"metric-val\">{value}</div>\n"
                f"  <div class=\"metric-lbl\">{label}</div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <div class=\"sbti-notice\">\n"
            "    <h3>SBTi Compliance Note</h3>\n"
            f"    <div class=\"note-text\">{self._escape_html(note_text)}</div>\n"
            f"    <div class=\"sbti-metrics\">{metrics_html}</div>\n"
            "  </div>\n"
            "</div>"
        )

    def _render_html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"footer\">"
            f"Carbon Credit Report generated at {ts} | PACK-003 CSRD Enterprise"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _build_json_portfolio(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON portfolio section."""
        return data.get("portfolio", {})

    def _build_json_vintage(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON vintage breakdown."""
        return data.get("vintage_breakdown", [])

    def _build_json_quality(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON quality distribution."""
        return data.get("quality_distribution", [])

    def _build_json_waterfall(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON waterfall."""
        return data.get("waterfall", {})

    def _build_json_retirement(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON retirement schedule."""
        return data.get("retirement_schedule", [])

    def _build_json_price_trends(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON price trends."""
        return data.get("price_trends", [])

    def _build_json_project_map(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON project map."""
        return data.get("project_map", [])

    def _build_json_sbti(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON SBTi note."""
        return data.get("sbti_note", {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_provenance_hash(content: str) -> str:
        """Generate SHA-256 provenance hash.

        Args:
            content: Content to hash.

        Returns:
            Hexadecimal SHA-256 hash.
        """
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
    def _quality_class(score: float) -> str:
        """Return CSS class for quality tier based on additionality score.

        Args:
            score: Additionality score 0-10.

        Returns:
            CSS class name.
        """
        if score >= 7:
            return "high"
        elif score >= 4:
            return "medium"
        return "low"

    @staticmethod
    def _text_bar(value: float, max_val: float, width: int = 15) -> str:
        """Create a text-based bar for Markdown tables."""
        filled = int((value / max_val) * width) if max_val else 0
        return "|" + "=" * filled + " " * (width - filled) + "|"
