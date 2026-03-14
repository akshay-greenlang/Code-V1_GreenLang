"""
SupplyChainReportTemplate - Supply chain ESG scorecard for CSRD Enterprise Pack.

This module implements the supply chain report template rendering multi-tier
supply chain maps, supplier scorecards, risk distributions, sector benchmarks,
improvement plans, Scope 3 breakdowns, and ESG trend charts.

Example:
    >>> template = SupplyChainReportTemplate()
    >>> data = {"suppliers": [...], "risk_distribution": {...}}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SupplyChainReportTemplate:
    """
    Supply chain ESG scorecard template.

    Renders supplier risk maps, scorecards, risk distributions, sector
    benchmarks, improvement plans, Scope 3 breakdowns, and trend charts.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    RISK_TIERS = ["Critical", "High", "Medium", "Low"]

    RISK_COLORS: Dict[str, str] = {
        "Critical": "#e02424",
        "High": "#d97706",
        "Medium": "#e3a008",
        "Low": "#057a55",
    }

    SUPPLIER_TIERS = ["Tier 1", "Tier 2", "Tier 3", "Tier 4+"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SupplyChainReportTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render supply chain report as Markdown.

        Args:
            data: Report data with suppliers, risk_distribution, benchmarks,
                  improvement_plans, scope3_breakdown, trends, spotlight.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._render_md_header(data))
        sections.append(self._render_md_supply_chain_map(data))
        sections.append(self._render_md_supplier_scorecard(data))
        sections.append(self._render_md_risk_distribution(data))
        sections.append(self._render_md_sector_benchmark(data))
        sections.append(self._render_md_improvement_plans(data))
        sections.append(self._render_md_scope3_breakdown(data))
        sections.append(self._render_md_top_risk_spotlight(data))
        sections.append(self._render_md_trends(data))
        sections.append(self._render_md_footer(data))

        content = "\n\n".join(sections)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- Provenance: {provenance} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render supply chain report as self-contained HTML.

        Args:
            data: Report data dict.

        Returns:
            Complete HTML string with inline styles.
        """
        self.generated_at = datetime.utcnow()
        css = self._build_css()
        body_parts: List[str] = []

        body_parts.append(self._render_html_header(data))
        body_parts.append(self._render_html_supply_chain_map(data))
        body_parts.append(self._render_html_supplier_scorecard(data))
        body_parts.append(self._render_html_risk_distribution(data))
        body_parts.append(self._render_html_sector_benchmark(data))
        body_parts.append(self._render_html_improvement_plans(data))
        body_parts.append(self._render_html_scope3_breakdown(data))
        body_parts.append(self._render_html_top_risk_spotlight(data))
        body_parts.append(self._render_html_trends(data))
        body_parts.append(self._render_html_footer(data))

        body_html = "\n".join(body_parts)
        provenance = self._generate_provenance_hash(body_html)

        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>Supply Chain ESG Report</title>\n<style>\n{css}\n</style>\n"
            "</head>\n<body>\n"
            f"<div class=\"report-container\">\n{body_html}\n</div>\n"
            f"<!-- Provenance: {provenance} -->\n"
            "</body>\n</html>"
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render supply chain report as structured JSON.

        Args:
            data: Report data dict.

        Returns:
            Structured dict with all report sections.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "supply_chain_report",
            "version": "1.0.0",
            "generated_at": self.generated_at.isoformat(),
            "supply_chain_map": self._build_json_map(data),
            "supplier_scorecard": self._build_json_scorecard(data),
            "risk_distribution": self._build_json_risk_distribution(data),
            "sector_benchmarks": self._build_json_benchmarks(data),
            "improvement_plans": self._build_json_improvement_plans(data),
            "scope3_breakdown": self._build_json_scope3(data),
            "top_risk_spotlight": self._build_json_spotlight(data),
            "trends": self._build_json_trends(data),
        }
        provenance = self._generate_provenance_hash(json.dumps(result, default=str))
        result["provenance_hash"] = provenance
        return result

    # ------------------------------------------------------------------
    # Markdown renderers
    # ------------------------------------------------------------------

    def _render_md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        title = data.get("title", "Supply Chain ESG Scorecard")
        ts = self._format_date(self.generated_at)
        return f"# {title}\n\n**Generated:** {ts}\n\n---"

    def _render_md_supply_chain_map(self, data: Dict[str, Any]) -> str:
        """Render supply chain map as a tiered text representation."""
        chain_map: List[Dict[str, Any]] = data.get("supply_chain_map", [])
        if not chain_map:
            return "## Supply Chain Map\n\n_No supply chain map data available._"

        lines = ["## Supply Chain Map (Multi-Tier, Color-Coded by Risk)", ""]

        for tier_name in self.SUPPLIER_TIERS:
            tier_suppliers = [
                s for s in chain_map if s.get("tier", "") == tier_name
            ]
            if not tier_suppliers:
                continue
            lines.append(f"### {tier_name} ({len(tier_suppliers)} suppliers)")
            lines.append("")
            for s in tier_suppliers:
                name = s.get("name", "Unknown")
                risk = s.get("risk_tier", "Medium")
                country = s.get("country", "-")
                commodity = s.get("commodity", "-")
                lines.append(
                    f"- **{name}** [{risk}] - {country}, {commodity}"
                )
            lines.append("")

        return "\n".join(lines)

    def _render_md_supplier_scorecard(self, data: Dict[str, Any]) -> str:
        """Render supplier scorecard table."""
        suppliers: List[Dict[str, Any]] = data.get("suppliers", [])
        if not suppliers:
            return "## Supplier Scorecard\n\n_No supplier data available._"

        lines = [
            "## Supplier Scorecard",
            "",
            "| Supplier | Tier | E Score | S Score | G Score | Composite | Risk Tier |",
            "|----------|------|---------|---------|---------|-----------|-----------|",
        ]
        for s in suppliers:
            name = s.get("name", "-")
            tier = s.get("tier", "-")
            e_score = self._format_number(s.get("e_score", 0), 1)
            s_score = self._format_number(s.get("s_score", 0), 1)
            g_score = self._format_number(s.get("g_score", 0), 1)
            composite = self._format_number(s.get("composite_score", 0), 1)
            risk_tier = s.get("risk_tier", "-")
            lines.append(
                f"| {name} | {tier} | {e_score} | {s_score} | {g_score} | {composite} | {risk_tier} |"
            )

        return "\n".join(lines)

    def _render_md_risk_distribution(self, data: Dict[str, Any]) -> str:
        """Render risk distribution summary."""
        dist = data.get("risk_distribution", {})
        if not dist:
            return "## Risk Distribution\n\n_No risk distribution data available._"

        total = sum(dist.get(tier, 0) for tier in self.RISK_TIERS)
        lines = [
            "## Risk Distribution",
            "",
            "| Risk Tier | Count | Percentage |",
            "|-----------|-------|-----------|",
        ]
        for tier in self.RISK_TIERS:
            count = dist.get(tier, 0)
            pct = (count / total * 100) if total else 0
            bar = self._text_bar(pct, 100.0)
            lines.append(
                f"| {tier} | {count} | {self._format_percentage(pct)} {bar} |"
            )

        return "\n".join(lines)

    def _render_md_sector_benchmark(self, data: Dict[str, Any]) -> str:
        """Render sector benchmark comparison."""
        benchmarks: List[Dict[str, Any]] = data.get("sector_benchmarks", [])
        if not benchmarks:
            return "## Sector Benchmark\n\n_No benchmark data available._"

        lines = [
            "## Sector Benchmark Comparison",
            "",
            "| Supplier | Supplier Score | Sector Avg | Delta | Percentile |",
            "|----------|---------------|-----------|-------|-----------|",
        ]
        for b in benchmarks:
            name = b.get("supplier", "-")
            score = self._format_number(b.get("score", 0), 1)
            avg = self._format_number(b.get("sector_average", 0), 1)
            delta = self._format_number(b.get("delta", 0), 1)
            percentile = b.get("percentile", 0)
            lines.append(
                f"| {name} | {score} | {avg} | {delta} | P{percentile} |"
            )

        return "\n".join(lines)

    def _render_md_improvement_plans(self, data: Dict[str, Any]) -> str:
        """Render improvement plan tracker."""
        plans: List[Dict[str, Any]] = data.get("improvement_plans", [])
        if not plans:
            return "## Improvement Plans\n\n_No improvement plans tracked._"

        lines = [
            "## Improvement Plan Tracker",
            "",
            "| Action | Supplier | Deadline | Status | Responsible |",
            "|--------|----------|----------|--------|-------------|",
        ]
        for p in plans:
            action = p.get("action", "-")
            supplier = p.get("supplier", "-")
            deadline = p.get("deadline", "-")
            status = p.get("status", "-")
            responsible = p.get("responsible", "-")
            lines.append(
                f"| {action} | {supplier} | {deadline} | {status} | {responsible} |"
            )

        return "\n".join(lines)

    def _render_md_scope3_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Scope 3 upstream emission breakdown by tier."""
        breakdown: List[Dict[str, Any]] = data.get("scope3_breakdown", [])
        if not breakdown:
            return "## Scope 3 Upstream Breakdown\n\n_No Scope 3 data available._"

        lines = [
            "## Scope 3 Upstream Emission Breakdown by Supplier Tier",
            "",
            "| Tier | Emissions (tCO2e) | % of Total | Top Contributor |",
            "|------|------------------|-----------|-----------------|",
        ]
        for row in breakdown:
            tier = row.get("tier", "-")
            emissions = self._format_number(row.get("emissions_tco2e", 0))
            pct = self._format_percentage(row.get("pct_of_total", 0))
            top = row.get("top_contributor", "-")
            lines.append(f"| {tier} | {emissions} | {pct} | {top} |")

        return "\n".join(lines)

    def _render_md_top_risk_spotlight(self, data: Dict[str, Any]) -> str:
        """Render top risk suppliers spotlight."""
        spotlight: List[Dict[str, Any]] = data.get("top_risk_spotlight", [])
        if not spotlight:
            return "## Top Risk Suppliers\n\n_No high-risk suppliers identified._"

        lines = ["## Top Risk Suppliers Spotlight", ""]
        for idx, s in enumerate(spotlight, 1):
            name = s.get("name", "Unknown")
            risk = s.get("risk_tier", "-")
            reason = s.get("risk_reason", "-")
            action = s.get("recommended_action", "-")
            lines.extend([
                f"### {idx}. {name} ({risk} Risk)",
                f"- **Reason:** {reason}",
                f"- **Recommended Action:** {action}",
                "",
            ])

        return "\n".join(lines)

    def _render_md_trends(self, data: Dict[str, Any]) -> str:
        """Render ESG trend data table."""
        trends: List[Dict[str, Any]] = data.get("trends", [])
        if not trends:
            return "## ESG Trends\n\n_No trend data available._"

        lines = [
            "## ESG Score Trends Over Time",
            "",
            "| Period | Avg E Score | Avg S Score | Avg G Score | Avg Composite |",
            "|--------|-----------|-----------|-----------|--------------|",
        ]
        for t in trends:
            period = t.get("period", "-")
            e = self._format_number(t.get("avg_e", 0), 1)
            s = self._format_number(t.get("avg_s", 0), 1)
            g = self._format_number(t.get("avg_g", 0), 1)
            comp = self._format_number(t.get("avg_composite", 0), 1)
            lines.append(f"| {period} | {e} | {s} | {g} | {comp} |")

        return "\n".join(lines)

    def _render_md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        ts = self._format_date(self.generated_at)
        return f"---\n_Supply Chain Report generated at {ts} | PACK-003 CSRD Enterprise_"

    # ------------------------------------------------------------------
    # HTML renderers
    # ------------------------------------------------------------------

    def _build_css(self) -> str:
        """Build inline CSS for supply chain report."""
        return """
:root {
    --primary: #1a56db; --primary-light: #e1effe; --success: #057a55;
    --warning: #e3a008; --danger: #e02424; --info: #1c64f2;
    --bg: #f9fafb; --card-bg: #fff; --text: #1f2937;
    --text-muted: #6b7280; --border: #e5e7eb;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); }
.report-container { max-width: 1300px; margin: 0 auto; padding: 24px; }
.report-header { background: linear-gradient(135deg, #057a55, #1a56db);
    color: #fff; padding: 28px 32px; border-radius: 12px; margin-bottom: 24px; }
.report-header h1 { font-size: 26px; }
.report-header .subtitle { opacity: 0.85; margin-top: 4px; font-size: 14px; }
.section { margin-bottom: 24px; background: var(--card-bg); border-radius: 10px;
    padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.section-title { font-size: 18px; font-weight: 600; color: var(--primary);
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid var(--primary); }
table { width: 100%; border-collapse: collapse; margin-bottom: 12px; }
th { background: var(--primary-light); color: var(--primary); padding: 10px 12px;
    text-align: left; font-size: 12px; font-weight: 600; }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; }
tr:hover { background: #f3f4f6; }
.tier-section { margin-bottom: 16px; }
.tier-header { font-size: 15px; font-weight: 600; color: var(--primary);
    margin-bottom: 8px; }
.supplier-node { display: inline-block; padding: 6px 14px; border-radius: 6px;
    margin: 4px; font-size: 12px; font-weight: 500; border: 1px solid; }
.supplier-node.critical { background: #fde8e8; border-color: #e02424; color: #e02424; }
.supplier-node.high { background: #feecdc; border-color: #d97706; color: #d97706; }
.supplier-node.medium { background: #fef9c3; border-color: #92400e; color: #92400e; }
.supplier-node.low { background: #d1fae5; border-color: #057a55; color: #057a55; }
.risk-badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; }
.risk-badge.critical { background: #fde8e8; color: #e02424; }
.risk-badge.high { background: #feecdc; color: #d97706; }
.risk-badge.medium { background: #fef9c3; color: #92400e; }
.risk-badge.low { background: #d1fae5; color: #057a55; }
.pie-chart { display: flex; gap: 16px; align-items: center; margin-bottom: 16px; }
.pie-legend { list-style: none; }
.pie-legend li { display: flex; align-items: center; margin-bottom: 6px; font-size: 13px; }
.pie-legend .dot { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
.score-bar { display: inline-block; height: 14px; border-radius: 3px;
    vertical-align: middle; }
.score-bar.e { background: #10b981; }
.score-bar.s { background: #3b82f6; }
.score-bar.g { background: #8b5cf6; }
.spotlight-card { border: 1px solid var(--danger); border-radius: 10px;
    padding: 16px; margin-bottom: 12px; border-left: 4px solid var(--danger); }
.spotlight-card h3 { font-size: 15px; margin-bottom: 6px; }
.spotlight-card .reason { font-size: 13px; margin-bottom: 4px; }
.spotlight-card .action { font-size: 13px; color: var(--primary); font-weight: 500; }
.status-badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; }
.status-badge.completed { background: #d1fae5; color: #057a55; }
.status-badge.in_progress { background: #dbeafe; color: #1e40af; }
.status-badge.pending { background: #fef9c3; color: #92400e; }
.status-badge.overdue { background: #fde8e8; color: #e02424; }
.trend-table td.improving { color: var(--success); font-weight: 600; }
.trend-table td.declining { color: var(--danger); font-weight: 600; }
.footer { text-align: center; color: var(--text-muted); font-size: 12px;
    padding: 16px 0; margin-top: 24px; border-top: 1px solid var(--border); }
"""

    def _render_html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        title = self._escape_html(data.get("title", "Supply Chain ESG Scorecard"))
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"report-header\">\n"
            f"  <h1>{title}</h1>\n"
            f"  <div class=\"subtitle\">Generated: {ts}</div>\n"
            f"</div>"
        )

    def _render_html_supply_chain_map(self, data: Dict[str, Any]) -> str:
        """Render HTML supply chain map visualization."""
        chain_map: List[Dict[str, Any]] = data.get("supply_chain_map", [])
        if not chain_map:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Supply Chain Map</h2>\n"
                "  <p>No supply chain map data available.</p>\n</div>"
            )

        tiers_html = ""
        for tier_name in self.SUPPLIER_TIERS:
            tier_suppliers = [
                s for s in chain_map if s.get("tier", "") == tier_name
            ]
            if not tier_suppliers:
                continue
            nodes = ""
            for s in tier_suppliers:
                name = self._escape_html(s.get("name", "Unknown"))
                risk = s.get("risk_tier", "Medium").lower()
                country = self._escape_html(s.get("country", "-"))
                nodes += (
                    f"<span class=\"supplier-node {risk}\" "
                    f"title=\"{country}\">{name}</span>\n"
                )
            tiers_html += (
                f"<div class=\"tier-section\">\n"
                f"  <div class=\"tier-header\">{tier_name} "
                f"({len(tier_suppliers)} suppliers)</div>\n"
                f"  {nodes}\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Supply Chain Map</h2>\n"
            f"  {tiers_html}\n"
            "</div>"
        )

    def _render_html_supplier_scorecard(self, data: Dict[str, Any]) -> str:
        """Render HTML supplier scorecard table."""
        suppliers: List[Dict[str, Any]] = data.get("suppliers", [])
        if not suppliers:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Supplier Scorecard</h2>\n"
                "  <p>No supplier data available.</p>\n</div>"
            )

        rows = ""
        for s in suppliers:
            name = self._escape_html(s.get("name", "-"))
            tier = s.get("tier", "-")
            e = s.get("e_score", 0)
            s_val = s.get("s_score", 0)
            g = s.get("g_score", 0)
            composite = s.get("composite_score", 0)
            risk = s.get("risk_tier", "Medium").lower()

            rows += (
                f"<tr><td><strong>{name}</strong></td><td>{tier}</td>"
                f"<td>{self._format_number(e, 1)} "
                f"<span class=\"score-bar e\" style=\"width:{e}px\"></span></td>"
                f"<td>{self._format_number(s_val, 1)} "
                f"<span class=\"score-bar s\" style=\"width:{s_val}px\"></span></td>"
                f"<td>{self._format_number(g, 1)} "
                f"<span class=\"score-bar g\" style=\"width:{g}px\"></span></td>"
                f"<td><strong>{self._format_number(composite, 1)}</strong></td>"
                f"<td><span class=\"risk-badge {risk}\">{s.get('risk_tier', '-')}"
                f"</span></td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Supplier Scorecard</h2>\n"
            "  <table><thead><tr>"
            "<th>Supplier</th><th>Tier</th><th>E Score</th><th>S Score</th>"
            "<th>G Score</th><th>Composite</th><th>Risk</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_risk_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML risk distribution pie chart representation."""
        dist = data.get("risk_distribution", {})
        if not dist:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Risk Distribution</h2>\n"
                "  <p>No risk distribution data available.</p>\n</div>"
            )

        total = sum(dist.get(tier, 0) for tier in self.RISK_TIERS)
        legend_items = ""
        for tier in self.RISK_TIERS:
            count = dist.get(tier, 0)
            pct = (count / total * 100) if total else 0
            color = self.RISK_COLORS.get(tier, "#6b7280")
            legend_items += (
                f"<li><span class=\"dot\" style=\"background:{color}\"></span>"
                f"{tier}: {count} ({self._format_percentage(pct)})</li>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Risk Distribution</h2>\n"
            "  <div class=\"pie-chart\">\n"
            f"    <ul class=\"pie-legend\">{legend_items}</ul>\n"
            "  </div>\n"
            "</div>"
        )

    def _render_html_sector_benchmark(self, data: Dict[str, Any]) -> str:
        """Render HTML sector benchmark comparison."""
        benchmarks: List[Dict[str, Any]] = data.get("sector_benchmarks", [])
        if not benchmarks:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Sector Benchmark</h2>\n"
                "  <p>No benchmark data available.</p>\n</div>"
            )

        rows = ""
        for b in benchmarks:
            name = self._escape_html(b.get("supplier", "-"))
            score = b.get("score", 0)
            avg = b.get("sector_average", 0)
            delta = b.get("delta", 0)
            delta_cls = "improving" if delta > 0 else "declining"
            percentile = b.get("percentile", 0)
            rows += (
                f"<tr><td><strong>{name}</strong></td>"
                f"<td>{self._format_number(score, 1)}</td>"
                f"<td>{self._format_number(avg, 1)}</td>"
                f"<td class=\"{delta_cls}\">"
                f"{'+'if delta > 0 else ''}{self._format_number(delta, 1)}</td>"
                f"<td>P{percentile}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Sector Benchmark Comparison</h2>\n"
            "  <table><thead><tr>"
            "<th>Supplier</th><th>Score</th><th>Sector Avg</th>"
            "<th>Delta</th><th>Percentile</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_improvement_plans(self, data: Dict[str, Any]) -> str:
        """Render HTML improvement plan tracker."""
        plans: List[Dict[str, Any]] = data.get("improvement_plans", [])
        if not plans:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Improvement Plans</h2>\n"
                "  <p>No improvement plans tracked.</p>\n</div>"
            )

        rows = ""
        for p in plans:
            status = p.get("status", "pending").lower().replace(" ", "_")
            rows += (
                f"<tr><td>{self._escape_html(p.get('action', '-'))}</td>"
                f"<td>{self._escape_html(p.get('supplier', '-'))}</td>"
                f"<td>{p.get('deadline', '-')}</td>"
                f"<td><span class=\"status-badge {status}\">{p.get('status', '-')}"
                f"</span></td>"
                f"<td>{self._escape_html(p.get('responsible', '-'))}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Improvement Plan Tracker</h2>\n"
            "  <table><thead><tr>"
            "<th>Action</th><th>Supplier</th><th>Deadline</th>"
            "<th>Status</th><th>Responsible</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_scope3_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML Scope 3 breakdown by tier."""
        breakdown: List[Dict[str, Any]] = data.get("scope3_breakdown", [])
        if not breakdown:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Scope 3 Upstream Breakdown</h2>\n"
                "  <p>No Scope 3 data available.</p>\n</div>"
            )

        max_emissions = max(
            (r.get("emissions_tco2e", 0) for r in breakdown), default=1
        )
        rows = ""
        for row in breakdown:
            emissions = row.get("emissions_tco2e", 0)
            pct = row.get("pct_of_total", 0)
            bar_width = (emissions / max_emissions * 200) if max_emissions else 0
            rows += (
                f"<tr><td>{row.get('tier', '-')}</td>"
                f"<td>{self._format_number(emissions)}</td>"
                f"<td>{self._format_percentage(pct)} "
                f"<span class=\"score-bar e\" style=\"width:{bar_width:.0f}px\"></span></td>"
                f"<td>{self._escape_html(row.get('top_contributor', '-'))}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Scope 3 Upstream Emission Breakdown</h2>\n"
            "  <table><thead><tr>"
            "<th>Tier</th><th>Emissions (tCO2e)</th><th>% of Total</th>"
            "<th>Top Contributor</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_top_risk_spotlight(self, data: Dict[str, Any]) -> str:
        """Render HTML top risk suppliers spotlight."""
        spotlight: List[Dict[str, Any]] = data.get("top_risk_spotlight", [])
        if not spotlight:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Top Risk Suppliers</h2>\n"
                "  <p>No high-risk suppliers identified.</p>\n</div>"
            )

        cards = ""
        for s in spotlight:
            name = self._escape_html(s.get("name", "Unknown"))
            risk = s.get("risk_tier", "High")
            reason = self._escape_html(s.get("risk_reason", "-"))
            action = self._escape_html(s.get("recommended_action", "-"))
            cards += (
                f"<div class=\"spotlight-card\">\n"
                f"  <h3>{name} "
                f"<span class=\"risk-badge {risk.lower()}\">{risk}</span></h3>\n"
                f"  <div class=\"reason\">Reason: {reason}</div>\n"
                f"  <div class=\"action\">Action: {action}</div>\n"
                f"</div>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Top Risk Suppliers Spotlight</h2>\n"
            f"  {cards}\n"
            "</div>"
        )

    def _render_html_trends(self, data: Dict[str, Any]) -> str:
        """Render HTML ESG trend table."""
        trends: List[Dict[str, Any]] = data.get("trends", [])
        if not trends:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">ESG Trends</h2>\n"
                "  <p>No trend data available.</p>\n</div>"
            )

        rows = ""
        for t in trends:
            rows += (
                f"<tr><td>{t.get('period', '-')}</td>"
                f"<td>{self._format_number(t.get('avg_e', 0), 1)}</td>"
                f"<td>{self._format_number(t.get('avg_s', 0), 1)}</td>"
                f"<td>{self._format_number(t.get('avg_g', 0), 1)}</td>"
                f"<td><strong>{self._format_number(t.get('avg_composite', 0), 1)}"
                f"</strong></td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">ESG Score Trends Over Time</h2>\n"
            "  <table class=\"trend-table\"><thead><tr>"
            "<th>Period</th><th>Avg E</th><th>Avg S</th><th>Avg G</th>"
            "<th>Composite</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"footer\">"
            f"Supply Chain Report generated at {ts} | PACK-003 CSRD Enterprise"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _build_json_map(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON supply chain map."""
        return data.get("supply_chain_map", [])

    def _build_json_scorecard(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON supplier scorecard."""
        return data.get("suppliers", [])

    def _build_json_risk_distribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON risk distribution."""
        return data.get("risk_distribution", {})

    def _build_json_benchmarks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON sector benchmarks."""
        return data.get("sector_benchmarks", [])

    def _build_json_improvement_plans(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build JSON improvement plans."""
        return data.get("improvement_plans", [])

    def _build_json_scope3(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON Scope 3 breakdown."""
        return data.get("scope3_breakdown", [])

    def _build_json_spotlight(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON top risk spotlight."""
        return data.get("top_risk_spotlight", [])

    def _build_json_trends(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON trends."""
        return data.get("trends", [])

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
        """Format numeric value with thousands separator.

        Args:
            value: Numeric value.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def _format_percentage(value: Union[int, float]) -> str:
        """Format value as percentage.

        Args:
            value: Numeric value.

        Returns:
            Percentage string.
        """
        return f"{value:.1f}%"

    @staticmethod
    def _format_date(dt: Optional[datetime]) -> str:
        """Format datetime as string.

        Args:
            dt: Datetime object.

        Returns:
            Formatted date string.
        """
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Raw text.

        Returns:
            HTML-safe string.
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    @staticmethod
    def _text_bar(value: float, max_val: float = 100.0, width: int = 15) -> str:
        """Create a text-based bar for Markdown tables.

        Args:
            value: Current value.
            max_val: Maximum value for scaling.
            width: Bar width in characters.

        Returns:
            Text bar string.
        """
        filled = int((value / max_val) * width) if max_val else 0
        return "|" + "=" * filled + " " * (width - filled) + "|"
