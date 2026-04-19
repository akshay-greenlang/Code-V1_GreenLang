# -*- coding: utf-8 -*-
"""
ExecutiveSummaryReportTemplate - Executive summary report for PACK-035.

Generates concise executive-level summary reports for energy benchmarking
with headline KPIs, performance ratings, peer positioning, year-over-year
changes, top improvement opportunities, cost and carbon savings potential,
and recommended next steps.

Sections:
    1. Header
    2. Key Performance Indicators (3-5 headline numbers)
    3. Performance Rating
    4. Peer Position
    5. Year-over-Year Change
    6. Top 3 Improvement Opportunities
    7. Cost / Carbon Savings Potential
    8. Next Steps
    9. Provenance

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExecutiveSummaryReportTemplate:
    """
    Executive summary report template for energy benchmarking.

    Renders concise, board-level energy benchmark summaries with headline
    KPIs, performance ratings, peer positioning, improvement opportunities,
    and savings potential across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ExecutiveSummaryReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render executive summary as Markdown.

        Args:
            data: Executive summary data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_kpis(data),
            self._md_performance_rating(data),
            self._md_peer_position(data),
            self._md_yoy_change(data),
            self._md_top_opportunities(data),
            self._md_savings_potential(data),
            self._md_next_steps(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive summary as self-contained HTML.

        Args:
            data: Executive summary data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_kpis(data),
            self._html_performance_rating(data),
            self._html_peer_position(data),
            self._html_yoy_change(data),
            self._html_top_opportunities(data),
            self._html_savings_potential(data),
            self._html_next_steps(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Executive Summary - Energy Benchmark</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render executive summary as structured JSON.

        Args:
            data: Executive summary data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "executive_summary_report",
            "version": "35.0.0",
            "generated_at": self.generated_at.isoformat(),
            "kpis": data.get("kpis", {}),
            "performance_rating": data.get("performance_rating", {}),
            "peer_position": data.get("peer_position", {}),
            "yoy_change": data.get("yoy_change", {}),
            "top_opportunities": data.get("top_opportunities", [])[:3],
            "savings_potential": data.get("savings_potential", {}),
            "next_steps": data.get("next_steps", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Executive Summary: Energy Benchmark\n\n"
            f"**Facility:** {facility}  \n"
            f"**Prepared For:** {data.get('prepared_for', 'Executive Leadership')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-035 ExecutiveSummaryReportTemplate v35.0.0\n\n---"
        )

    def _md_kpis(self, data: Dict[str, Any]) -> str:
        """Render key performance indicators section."""
        k = data.get("kpis", {})
        return (
            "## 1. Key Performance Indicators\n\n"
            "| KPI | Value |\n|-----|-------|\n"
            f"| Site EUI | {self._fmt(k.get('site_eui', 0))} kWh/m2/yr |\n"
            f"| ENERGY STAR Score | {k.get('energy_star_score', '-')} / 100 |\n"
            f"| Annual Energy Cost | EUR {self._fmt(k.get('annual_cost_eur', 0))} |\n"
            f"| CO2 Emissions | {self._fmt(k.get('co2_tonnes', 0))} tonnes/yr |\n"
            f"| Cost per m2 | EUR {self._fmt(k.get('cost_per_m2', 0))} /m2/yr |"
        )

    def _md_performance_rating(self, data: Dict[str, Any]) -> str:
        """Render performance rating section."""
        pr = data.get("performance_rating", {})
        return (
            "## 2. Performance Rating\n\n"
            f"**Overall Rating:** **{pr.get('rating', '-')}**  \n"
            f"**Rating Basis:** {pr.get('basis', '-')}  \n"
            f"**Score:** {self._fmt(pr.get('score', 0), 0)} / {pr.get('max_score', 100)}  \n"
            f"**Interpretation:** {pr.get('interpretation', '-')}"
        )

    def _md_peer_position(self, data: Dict[str, Any]) -> str:
        """Render peer position section."""
        pp = data.get("peer_position", {})
        return (
            "## 3. Peer Position\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Percentile Rank | {self._fmt(pp.get('percentile', 0), 0)}th |\n"
            f"| Quartile | Q{pp.get('quartile', '-')} |\n"
            f"| Peer Group | {pp.get('peer_group', '-')} |\n"
            f"| Peer Count | {pp.get('peer_count', 0)} |\n"
            f"| Peer Mean EUI | {self._fmt(pp.get('peer_mean_eui', 0))} kWh/m2/yr |\n"
            f"| vs Peer Mean | {self._fmt(pp.get('vs_peer_mean_pct', 0))}% |"
        )

    def _md_yoy_change(self, data: Dict[str, Any]) -> str:
        """Render year-over-year change section."""
        yoy = data.get("yoy_change", {})
        return (
            "## 4. Year-over-Year Change\n\n"
            "| Metric | Prior Year | Current Year | Change | Change (%) |\n"
            "|--------|-----------|-------------|--------|------------|\n"
            f"| EUI (kWh/m2/yr) | {self._fmt(yoy.get('prior_eui', 0))} "
            f"| {self._fmt(yoy.get('current_eui', 0))} "
            f"| {self._fmt(yoy.get('eui_change', 0))} "
            f"| {self._fmt(yoy.get('eui_change_pct', 0))}% |\n"
            f"| Energy Cost (EUR) | {self._fmt(yoy.get('prior_cost', 0))} "
            f"| {self._fmt(yoy.get('current_cost', 0))} "
            f"| {self._fmt(yoy.get('cost_change', 0))} "
            f"| {self._fmt(yoy.get('cost_change_pct', 0))}% |\n"
            f"| CO2 (tonnes) | {self._fmt(yoy.get('prior_co2', 0))} "
            f"| {self._fmt(yoy.get('current_co2', 0))} "
            f"| {self._fmt(yoy.get('co2_change', 0))} "
            f"| {self._fmt(yoy.get('co2_change_pct', 0))}% |"
        )

    def _md_top_opportunities(self, data: Dict[str, Any]) -> str:
        """Render top 3 improvement opportunities section."""
        opps = data.get("top_opportunities", [])[:3]
        if not opps:
            return "## 5. Top Improvement Opportunities\n\n_No opportunities identified._"
        lines = [
            "## 5. Top 3 Improvement Opportunities\n",
            "| # | Opportunity | Savings (kWh/yr) | Cost Savings (EUR/yr) | Payback |",
            "|---|-----------|-----------------|---------------------|---------|",
        ]
        for i, o in enumerate(opps, 1):
            lines.append(
                f"| {i} | {o.get('name', '-')} "
                f"| {self._fmt(o.get('savings_kwh', 0), 0)} "
                f"| {self._fmt(o.get('cost_savings_eur', 0))} "
                f"| {o.get('payback', '-')} |"
            )
        return "\n".join(lines)

    def _md_savings_potential(self, data: Dict[str, Any]) -> str:
        """Render cost and carbon savings potential section."""
        sp = data.get("savings_potential", {})
        return (
            "## 6. Savings Potential\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Energy Savings | {self._fmt(sp.get('energy_savings_kwh', 0), 0)} kWh/yr |\n"
            f"| Energy Savings (%) | {self._fmt(sp.get('energy_savings_pct', 0))}% |\n"
            f"| Annual Cost Savings | EUR {self._fmt(sp.get('cost_savings_eur', 0))} |\n"
            f"| CO2 Reduction | {self._fmt(sp.get('co2_reduction_tonnes', 0))} tonnes/yr |\n"
            f"| Required Investment | EUR {self._fmt(sp.get('investment_eur', 0))} |\n"
            f"| Portfolio Payback | {self._fmt(sp.get('payback_years', 0), 1)} years |\n"
            f"| 5-Year NPV | EUR {self._fmt(sp.get('npv_5yr', 0))} |"
        )

    def _md_next_steps(self, data: Dict[str, Any]) -> str:
        """Render next steps section."""
        steps = data.get("next_steps", [])
        if not steps:
            steps = [
                {"step": "Review benchmark results with facilities team", "timeline": "Week 1"},
                {"step": "Prioritise improvement opportunities", "timeline": "Week 2"},
                {"step": "Develop implementation plan", "timeline": "Week 3-4"},
            ]
        lines = [
            "## 7. Next Steps\n",
            "| # | Action | Timeline |",
            "|---|--------|----------|",
        ]
        for i, s in enumerate(steps, 1):
            lines.append(
                f"| {i} | {s.get('step', '-')} | {s.get('timeline', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-035 Energy Benchmark Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Executive Summary: Energy Benchmark</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Prepared for: {data.get("prepared_for", "Executive Leadership")} | '
            f'Generated: {ts}</p>'
        )

    def _html_kpis(self, data: Dict[str, Any]) -> str:
        """Render HTML KPI cards."""
        k = data.get("kpis", {})
        return (
            '<h2>Key Performance Indicators</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Site EUI</span>'
            f'<span class="value">{self._fmt(k.get("site_eui", 0))}</span>'
            f'<span class="label">kWh/m2/yr</span></div>\n'
            f'  <div class="card"><span class="label">ENERGY STAR</span>'
            f'<span class="value">{k.get("energy_star_score", "-")}</span>'
            f'<span class="label">out of 100</span></div>\n'
            f'  <div class="card"><span class="label">Annual Cost</span>'
            f'<span class="value">EUR {self._fmt(k.get("annual_cost_eur", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">CO2</span>'
            f'<span class="value">{self._fmt(k.get("co2_tonnes", 0))} t</span></div>\n'
            f'  <div class="card"><span class="label">Cost/m2</span>'
            f'<span class="value">EUR {self._fmt(k.get("cost_per_m2", 0))}</span></div>\n'
            '</div>'
        )

    def _html_performance_rating(self, data: Dict[str, Any]) -> str:
        """Render HTML performance rating."""
        pr = data.get("performance_rating", {})
        score = pr.get("score", 0)
        max_score = pr.get("max_score", 100)
        pct = (score / max_score * 100) if max_score else 0
        color = "#198754" if pct >= 75 else ("#ffc107" if pct >= 50 else "#dc3545")
        return (
            '<h2>Performance Rating</h2>\n'
            f'<div class="rating-display">'
            f'<span class="rating-grade" style="color:{color};">'
            f'{pr.get("rating", "-")}</span>'
            f'<span class="rating-score">{self._fmt(score, 0)} / {max_score}</span></div>\n'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{pct}%;background:{color};"></div></div>\n'
            f'<p>{pr.get("interpretation", "")}</p>'
        )

    def _html_peer_position(self, data: Dict[str, Any]) -> str:
        """Render HTML peer position."""
        pp = data.get("peer_position", {})
        pct = pp.get("percentile", 50)
        return (
            '<h2>Peer Position</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Percentile</span>'
            f'<span class="value">{self._fmt(pct, 0)}th</span></div>\n'
            f'  <div class="card"><span class="label">Quartile</span>'
            f'<span class="value">Q{pp.get("quartile", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">vs Peer Mean</span>'
            f'<span class="value">{self._fmt(pp.get("vs_peer_mean_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Peer Group</span>'
            f'<span class="value">{pp.get("peer_group", "-")}</span></div>\n'
            '</div>'
        )

    def _html_yoy_change(self, data: Dict[str, Any]) -> str:
        """Render HTML year-over-year change."""
        yoy = data.get("yoy_change", {})
        eui_pct = yoy.get("eui_change_pct", 0)
        cls = "change-improve" if eui_pct < 0 else "change-worsen"
        return (
            '<h2>Year-over-Year Change</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card {cls}"><span class="label">EUI Change</span>'
            f'<span class="value">{self._fmt(eui_pct)}%</span></div>\n'
            f'  <div class="card"><span class="label">Cost Change</span>'
            f'<span class="value">{self._fmt(yoy.get("cost_change_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">CO2 Change</span>'
            f'<span class="value">{self._fmt(yoy.get("co2_change_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_top_opportunities(self, data: Dict[str, Any]) -> str:
        """Render HTML top improvement opportunities."""
        opps = data.get("top_opportunities", [])[:3]
        items = ""
        for i, o in enumerate(opps, 1):
            items += (
                f'<div class="opportunity">#{i} <strong>{o.get("name", "-")}</strong> | '
                f'Savings: {self._fmt(o.get("savings_kwh", 0), 0)} kWh/yr | '
                f'Cost: EUR {self._fmt(o.get("cost_savings_eur", 0))} | '
                f'Payback: {o.get("payback", "-")}</div>\n'
            )
        return f'<h2>Top 3 Improvement Opportunities</h2>\n{items}'

    def _html_savings_potential(self, data: Dict[str, Any]) -> str:
        """Render HTML savings potential."""
        sp = data.get("savings_potential", {})
        return (
            '<h2>Savings Potential</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card card-green"><span class="label">Energy Savings</span>'
            f'<span class="value">{self._fmt(sp.get("energy_savings_pct", 0))}%</span></div>\n'
            f'  <div class="card card-green"><span class="label">Cost Savings</span>'
            f'<span class="value">EUR {self._fmt(sp.get("cost_savings_eur", 0))}</span></div>\n'
            f'  <div class="card card-green"><span class="label">CO2 Reduction</span>'
            f'<span class="value">{self._fmt(sp.get("co2_reduction_tonnes", 0))} t</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(sp.get("payback_years", 0), 1)} yr</span></div>\n'
            '</div>'
        )

    def _html_next_steps(self, data: Dict[str, Any]) -> str:
        """Render HTML next steps."""
        steps = data.get("next_steps", [])
        items = "".join(
            f'<li><strong>{s.get("step", "-")}</strong> '
            f'({s.get("timeline", "-")})</li>\n'
            for s in steps
        )
        return f'<h2>Next Steps</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        kpis = data.get("kpis", {})
        pp = data.get("peer_position", {})
        opps = data.get("top_opportunities", [])[:3]
        return {
            "kpi_gauge": {
                "type": "gauge",
                "value": kpis.get("energy_star_score", 0),
                "max": 100,
            },
            "peer_percentile_bar": {
                "type": "bar",
                "labels": ["Your Facility", "Peer Mean"],
                "values": [kpis.get("site_eui", 0), pp.get("peer_mean_eui", 0)],
            },
            "opportunities_bar": {
                "type": "bar",
                "labels": [o.get("name", "") for o in opps],
                "values": [o.get("cost_savings_eur", 0) for o in opps],
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;"
            "text-align:center;min-width:150px;}"
            ".card-green{background:#d1e7dd;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".rating-display{text-align:center;margin:20px 0;}"
            ".rating-grade{font-size:3em;font-weight:700;}"
            ".rating-score{display:block;font-size:1.1em;color:#6c757d;}"
            ".progress-bar{height:24px;background:#e9ecef;border-radius:6px;margin:15px 0;overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:6px;}"
            ".change-improve{background:#d1e7dd;}"
            ".change-worsen{background:#f8d7da;}"
            ".opportunity{background:#f0f9ff;border-left:4px solid #0d6efd;"
            "padding:10px 15px;margin:8px 0;border-radius:0 4px 4px 0;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators.

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
