# -*- coding: utf-8 -*-
"""
PortfolioDashboardTemplate - Multi-facility portfolio dashboard for PACK-035.

Generates portfolio-level energy benchmark dashboards with KPI summaries,
facility rankings, EUI distribution analysis, quartile breakdowns,
best/worst performer identification, year-over-year improvement tracking,
regional breakdowns, and prioritised action items.

Sections:
    1. Portfolio Overview KPIs
    2. Facility Rankings Table
    3. EUI Distribution
    4. Quartile Analysis
    5. Best / Worst Performers
    6. Year-over-Year Improvement
    7. Regional Breakdown
    8. Action Items
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


class PortfolioDashboardTemplate:
    """
    Portfolio-level energy benchmark dashboard template.

    Renders multi-facility portfolio dashboards with rankings, EUI
    distribution analysis, year-over-year trends, and regional
    breakdowns across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PortfolioDashboardTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render portfolio dashboard as Markdown.

        Args:
            data: Portfolio benchmark data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview_kpis(data),
            self._md_facility_rankings(data),
            self._md_eui_distribution(data),
            self._md_quartile_analysis(data),
            self._md_best_worst(data),
            self._md_yoy_improvement(data),
            self._md_regional_breakdown(data),
            self._md_action_items(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render portfolio dashboard as self-contained HTML.

        Args:
            data: Portfolio benchmark data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview_kpis(data),
            self._html_facility_rankings(data),
            self._html_eui_distribution(data),
            self._html_quartile_analysis(data),
            self._html_best_worst(data),
            self._html_yoy_improvement(data),
            self._html_regional_breakdown(data),
            self._html_action_items(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Portfolio Energy Benchmark Dashboard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render portfolio dashboard as structured JSON.

        Args:
            data: Portfolio benchmark data from engine processing.

        Returns:
            Dict with structured dashboard sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "portfolio_dashboard",
            "version": "35.0.0",
            "generated_at": self.generated_at.isoformat(),
            "overview_kpis": data.get("overview_kpis", {}),
            "facility_rankings": data.get("facility_rankings", []),
            "eui_distribution": data.get("eui_distribution", []),
            "quartile_analysis": data.get("quartile_analysis", {}),
            "best_performers": data.get("best_performers", []),
            "worst_performers": data.get("worst_performers", []),
            "yoy_improvement": data.get("yoy_improvement", []),
            "regional_breakdown": data.get("regional_breakdown", []),
            "action_items": data.get("action_items", []),
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
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Portfolio Energy Benchmark Dashboard\n\n"
            f"**Portfolio:** {data.get('portfolio_name', 'Portfolio')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '-')}  \n"
            f"**Total Facilities:** {data.get('total_facilities', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-035 PortfolioDashboardTemplate v35.0.0\n\n---"
        )

    def _md_overview_kpis(self, data: Dict[str, Any]) -> str:
        """Render portfolio overview KPI section."""
        k = data.get("overview_kpis", {})
        return (
            "## 1. Portfolio Overview\n\n"
            "| KPI | Value |\n|-----|-------|\n"
            f"| Total Facilities | {k.get('total_facilities', 0)} |\n"
            f"| Total Floor Area | {self._fmt(k.get('total_floor_area_sqm', 0), 0)} m2 |\n"
            f"| Portfolio-Weighted EUI | {self._fmt(k.get('weighted_eui', 0))} kWh/m2/yr |\n"
            f"| Total Energy Consumption | {self._fmt(k.get('total_energy_mwh', 0))} MWh/yr |\n"
            f"| Total Energy Cost | EUR {self._fmt(k.get('total_cost_eur', 0))} /yr |\n"
            f"| Average EUI | {self._fmt(k.get('avg_eui', 0))} kWh/m2/yr |\n"
            f"| Median EUI | {self._fmt(k.get('median_eui', 0))} kWh/m2/yr |\n"
            f"| % Meeting Benchmark | {self._fmt(k.get('pct_meeting_benchmark', 0))}% |"
        )

    def _md_facility_rankings(self, data: Dict[str, Any]) -> str:
        """Render facility rankings table section."""
        rankings = data.get("facility_rankings", [])
        if not rankings:
            return "## 2. Facility Rankings\n\n_No ranking data available._"
        lines = [
            "## 2. Facility Rankings\n",
            "| Rank | Facility | Type | Area (m2) | EUI (kWh/m2) | Benchmark | Status |",
            "|------|---------|------|-----------|-------------|----------|--------|",
        ]
        for r in rankings:
            lines.append(
                f"| {r.get('rank', '-')} "
                f"| {r.get('name', '-')} "
                f"| {r.get('building_type', '-')} "
                f"| {self._fmt(r.get('area_sqm', 0), 0)} "
                f"| {self._fmt(r.get('eui', 0))} "
                f"| {self._fmt(r.get('benchmark_eui', 0))} "
                f"| {r.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_eui_distribution(self, data: Dict[str, Any]) -> str:
        """Render EUI distribution section."""
        dist = data.get("eui_distribution", [])
        if not dist:
            return "## 3. EUI Distribution\n\n_No distribution data available._"
        lines = [
            "## 3. EUI Distribution\n",
            "| EUI Range (kWh/m2/yr) | Facilities | Percentage |",
            "|----------------------|-----------|-----------|",
        ]
        for d in dist:
            lines.append(
                f"| {d.get('range_label', '-')} "
                f"| {d.get('count', 0)} "
                f"| {self._fmt(d.get('pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_quartile_analysis(self, data: Dict[str, Any]) -> str:
        """Render quartile analysis section."""
        qa = data.get("quartile_analysis", {})
        quartiles = qa.get("quartiles", [])
        lines = [
            "## 4. Quartile Analysis\n",
            "| Quartile | EUI Range | Count | Avg EUI | Avg Cost/m2 |",
            "|----------|----------|-------|---------|------------|",
        ]
        for q in quartiles:
            lines.append(
                f"| {q.get('label', '-')} "
                f"| {q.get('eui_range', '-')} "
                f"| {q.get('count', 0)} "
                f"| {self._fmt(q.get('avg_eui', 0))} "
                f"| EUR {self._fmt(q.get('avg_cost_per_m2', 0))} |"
            )
        return "\n".join(lines)

    def _md_best_worst(self, data: Dict[str, Any]) -> str:
        """Render best and worst performers section."""
        best = data.get("best_performers", [])[:5]
        worst = data.get("worst_performers", [])[:5]
        lines = ["## 5. Best & Worst Performers\n"]
        lines.append("### Top 5 Best Performers\n")
        lines.extend([
            "| Rank | Facility | EUI | vs Benchmark |",
            "|------|---------|-----|-------------|",
        ])
        for i, b in enumerate(best, 1):
            lines.append(
                f"| {i} | {b.get('name', '-')} "
                f"| {self._fmt(b.get('eui', 0))} "
                f"| {self._fmt(b.get('vs_benchmark_pct', 0))}% |"
            )
        lines.append("\n### Top 5 Worst Performers\n")
        lines.extend([
            "| Rank | Facility | EUI | vs Benchmark |",
            "|------|---------|-----|-------------|",
        ])
        for i, w in enumerate(worst, 1):
            lines.append(
                f"| {i} | {w.get('name', '-')} "
                f"| {self._fmt(w.get('eui', 0))} "
                f"| +{self._fmt(w.get('vs_benchmark_pct', 0))}% |"
            )
        return "\n".join(lines)

    def _md_yoy_improvement(self, data: Dict[str, Any]) -> str:
        """Render year-over-year improvement section."""
        yoy = data.get("yoy_improvement", [])
        if not yoy:
            return "## 6. Year-over-Year Improvement\n\n_No YoY data available._"
        lines = [
            "## 6. Year-over-Year Improvement\n",
            "| Year | Portfolio EUI | Change | Change (%) | Facilities Improved |",
            "|------|-------------|--------|-----------|-------------------|",
        ]
        for y in yoy:
            lines.append(
                f"| {y.get('year', '-')} "
                f"| {self._fmt(y.get('portfolio_eui', 0))} "
                f"| {self._fmt(y.get('change', 0))} "
                f"| {self._fmt(y.get('change_pct', 0))}% "
                f"| {y.get('improved_count', 0)} / {y.get('total_count', 0)} |"
            )
        return "\n".join(lines)

    def _md_regional_breakdown(self, data: Dict[str, Any]) -> str:
        """Render regional breakdown section."""
        regions = data.get("regional_breakdown", [])
        if not regions:
            return "## 7. Regional Breakdown\n\n_No regional data available._"
        lines = [
            "## 7. Regional Breakdown\n",
            "| Region | Facilities | Avg EUI | Weighted EUI | Total Cost |",
            "|--------|-----------|---------|-------------|------------|",
        ]
        for r in regions:
            lines.append(
                f"| {r.get('region', '-')} "
                f"| {r.get('count', 0)} "
                f"| {self._fmt(r.get('avg_eui', 0))} "
                f"| {self._fmt(r.get('weighted_eui', 0))} "
                f"| EUR {self._fmt(r.get('total_cost_eur', 0))} |"
            )
        return "\n".join(lines)

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Render action items section."""
        actions = data.get("action_items", [])
        if not actions:
            return "## 8. Action Items\n\n_No action items._"
        lines = [
            "## 8. Action Items\n",
            "| # | Action | Facility | Priority | Expected Impact |",
            "|---|--------|---------|----------|----------------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '-')} "
                f"| {a.get('facility', '-')} "
                f"| {a.get('priority', '-')} "
                f"| {a.get('expected_impact', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render dashboard footer."""
        return "---\n\n*Generated by GreenLang PACK-035 Energy Benchmark Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Portfolio Energy Benchmark Dashboard</h1>\n'
            f'<p class="subtitle">Portfolio: {data.get("portfolio_name", "Portfolio")} | '
            f'Facilities: {data.get("total_facilities", 0)} | '
            f'Generated: {ts}</p>'
        )

    def _html_overview_kpis(self, data: Dict[str, Any]) -> str:
        """Render HTML overview KPI cards."""
        k = data.get("overview_kpis", {})
        return (
            '<h2>Portfolio Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Facilities</span>'
            f'<span class="value">{k.get("total_facilities", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Weighted EUI</span>'
            f'<span class="value">{self._fmt(k.get("weighted_eui", 0))}</span>'
            f'<span class="label">kWh/m2/yr</span></div>\n'
            f'  <div class="card"><span class="label">Total Energy</span>'
            f'<span class="value">{self._fmt(k.get("total_energy_mwh", 0))} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Total Cost</span>'
            f'<span class="value">EUR {self._fmt(k.get("total_cost_eur", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Meeting Benchmark</span>'
            f'<span class="value">{self._fmt(k.get("pct_meeting_benchmark", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_facility_rankings(self, data: Dict[str, Any]) -> str:
        """Render HTML facility rankings table."""
        rankings = data.get("facility_rankings", [])
        rows = ""
        for r in rankings:
            status = r.get("status", "")
            cls = "status-pass" if status == "PASS" else ("status-fail" if status == "FAIL" else "")
            rows += (
                f'<tr><td>{r.get("rank", "-")}</td>'
                f'<td>{r.get("name", "-")}</td>'
                f'<td>{r.get("building_type", "-")}</td>'
                f'<td>{self._fmt(r.get("eui", 0))}</td>'
                f'<td class="{cls}">{status}</td></tr>\n'
            )
        return (
            '<h2>Facility Rankings</h2>\n'
            '<table>\n<tr><th>Rank</th><th>Facility</th><th>Type</th>'
            f'<th>EUI</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_eui_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML EUI distribution table."""
        dist = data.get("eui_distribution", [])
        rows = "".join(
            f'<tr><td>{d.get("range_label", "-")}</td>'
            f'<td>{d.get("count", 0)}</td>'
            f'<td>{self._fmt(d.get("pct", 0))}%</td></tr>\n'
            for d in dist
        )
        return (
            '<h2>EUI Distribution</h2>\n'
            '<table>\n<tr><th>EUI Range</th><th>Facilities</th>'
            f'<th>Percentage</th></tr>\n{rows}</table>'
        )

    def _html_quartile_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML quartile analysis."""
        quartiles = data.get("quartile_analysis", {}).get("quartiles", [])
        rows = "".join(
            f'<tr><td>{q.get("label", "-")}</td>'
            f'<td>{q.get("count", 0)}</td>'
            f'<td>{self._fmt(q.get("avg_eui", 0))}</td></tr>\n'
            for q in quartiles
        )
        return (
            '<h2>Quartile Analysis</h2>\n'
            '<table>\n<tr><th>Quartile</th><th>Count</th>'
            f'<th>Avg EUI</th></tr>\n{rows}</table>'
        )

    def _html_best_worst(self, data: Dict[str, Any]) -> str:
        """Render HTML best and worst performers."""
        best = data.get("best_performers", [])[:5]
        worst = data.get("worst_performers", [])[:5]
        best_items = "".join(
            f'<div class="performer best"><strong>{b.get("name", "-")}</strong> | '
            f'EUI: {self._fmt(b.get("eui", 0))} | '
            f'{self._fmt(b.get("vs_benchmark_pct", 0))}% below benchmark</div>\n'
            for b in best
        )
        worst_items = "".join(
            f'<div class="performer worst"><strong>{w.get("name", "-")}</strong> | '
            f'EUI: {self._fmt(w.get("eui", 0))} | '
            f'+{self._fmt(w.get("vs_benchmark_pct", 0))}% above benchmark</div>\n'
            for w in worst
        )
        return (
            '<h2>Best & Worst Performers</h2>\n'
            f'<h3>Top 5 Best</h3>\n{best_items}'
            f'<h3>Top 5 Worst</h3>\n{worst_items}'
        )

    def _html_yoy_improvement(self, data: Dict[str, Any]) -> str:
        """Render HTML year-over-year improvement."""
        yoy = data.get("yoy_improvement", [])
        rows = "".join(
            f'<tr><td>{y.get("year", "-")}</td>'
            f'<td>{self._fmt(y.get("portfolio_eui", 0))}</td>'
            f'<td>{self._fmt(y.get("change_pct", 0))}%</td>'
            f'<td>{y.get("improved_count", 0)} / {y.get("total_count", 0)}</td></tr>\n'
            for y in yoy
        )
        return (
            '<h2>Year-over-Year Improvement</h2>\n'
            '<table>\n<tr><th>Year</th><th>Portfolio EUI</th>'
            f'<th>Change</th><th>Improved</th></tr>\n{rows}</table>'
        )

    def _html_regional_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML regional breakdown."""
        regions = data.get("regional_breakdown", [])
        rows = "".join(
            f'<tr><td>{r.get("region", "-")}</td>'
            f'<td>{r.get("count", 0)}</td>'
            f'<td>{self._fmt(r.get("weighted_eui", 0))}</td>'
            f'<td>EUR {self._fmt(r.get("total_cost_eur", 0))}</td></tr>\n'
            for r in regions
        )
        return (
            '<h2>Regional Breakdown</h2>\n'
            '<table>\n<tr><th>Region</th><th>Facilities</th>'
            f'<th>Weighted EUI</th><th>Total Cost</th></tr>\n{rows}</table>'
        )

    def _html_action_items(self, data: Dict[str, Any]) -> str:
        """Render HTML action items."""
        actions = data.get("action_items", [])
        items = "".join(
            f'<li><strong>[{a.get("priority", "-")}]</strong> '
            f'{a.get("action", "-")} '
            f'({a.get("facility", "-")})</li>\n'
            for a in actions
        )
        return f'<h2>Action Items</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        rankings = data.get("facility_rankings", [])
        yoy = data.get("yoy_improvement", [])
        dist = data.get("eui_distribution", [])
        regions = data.get("regional_breakdown", [])
        return {
            "ranking_bar": {
                "type": "bar",
                "labels": [r.get("name", "") for r in rankings],
                "values": [r.get("eui", 0) for r in rankings],
            },
            "yoy_line": {
                "type": "line",
                "labels": [y.get("year", "") for y in yoy],
                "values": [y.get("portfolio_eui", 0) for y in yoy],
            },
            "distribution_histogram": {
                "type": "histogram",
                "labels": [d.get("range_label", "") for d in dist],
                "values": [d.get("count", 0) for d in dist],
            },
            "regional_pie": {
                "type": "pie",
                "labels": [r.get("region", "") for r in regions],
                "values": [r.get("count", 0) for r in regions],
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
            "h3{color:#0d6efd;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".status-pass{color:#198754;font-weight:700;}"
            ".status-fail{color:#dc3545;font-weight:700;}"
            ".performer{padding:8px 12px;margin:4px 0;border-radius:4px;}"
            ".best{background:#d1e7dd;border-left:4px solid #198754;}"
            ".worst{background:#f8d7da;border-left:4px solid #dc3545;}"
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
