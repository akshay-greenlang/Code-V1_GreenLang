# -*- coding: utf-8 -*-
"""
ExecutiveSummaryReportTemplate - C-suite DR program summary for PACK-037.

Generates 2-4 page executive summary reports for C-suite audiences covering
total DR revenue, carbon impact, program compliance scores, DER fleet
performance, strategic recommendations, and year-over-year comparisons.

Sections:
    1. Key Metrics (5-6 KPIs)
    2. Financial Performance
    3. Carbon Impact Summary
    4. Program Compliance Scorecard
    5. Strategic Recommendations
    6. Year-over-Year Comparison

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - FERC Order 2222 compliance
    - EU Electricity Directive 2019/944
    - CSRD / ESRS E1 reporting

Author: GreenLang Team
Version: 37.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class ExecutiveSummaryReportTemplate:
    """
    Executive summary report template for C-suite DR audiences.

    Renders concise, high-impact summaries with DR revenue, carbon impact,
    compliance scorecard, strategic recommendations, and year-over-year
    trends across markdown, HTML, and JSON formats.

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
            data: Executive summary engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_key_metrics(data),
            self._md_financial_performance(data),
            self._md_carbon_impact(data),
            self._md_compliance_scorecard(data),
            self._md_strategic_recommendations(data),
            self._md_yoy_comparison(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive summary as self-contained HTML.

        Args:
            data: Executive summary engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_key_metrics(data),
            self._html_financial_performance(data),
            self._html_carbon_impact(data),
            self._html_compliance_scorecard(data),
            self._html_strategic_recommendations(data),
            self._html_yoy_comparison(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>DR Executive Summary</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render executive summary as structured JSON.

        Args:
            data: Executive summary engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "executive_summary_report",
            "version": "37.0.0",
            "generated_at": self.generated_at.isoformat(),
            "key_metrics": self._json_key_metrics(data),
            "financial_performance": data.get("financial_performance", {}),
            "carbon_impact": data.get("carbon_impact", {}),
            "compliance_scorecard": data.get("compliance_scorecard", []),
            "strategic_recommendations": data.get("strategic_recommendations", []),
            "yoy_comparison": data.get("yoy_comparison", {}),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        org = data.get("organization_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Demand Response Executive Summary\n\n"
            f"**Organization:** {org}  \n"
            f"**Prepared For:** {data.get('prepared_for', 'Executive Leadership')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Programs Active:** {data.get('programs_active', 0)}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-037 ExecutiveSummaryReportTemplate v37.0.0\n\n---"
        )

    def _md_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render key metrics section with headline KPIs."""
        metrics = data.get("key_metrics", {})
        return (
            "## 1. Key Metrics\n\n"
            "| KPI | Value |\n|-----|-------|\n"
            f"| Total DR Revenue (Net) | {self._format_currency(metrics.get('total_net_revenue', 0))} |\n"
            f"| CO2e Avoided | {self._fmt(metrics.get('co2e_avoided_tonnes', 0))} tonnes |\n"
            f"| Events Participated | {metrics.get('events_participated', 0)} |\n"
            f"| Avg Performance Ratio | {self._fmt(metrics.get('avg_performance_ratio', 0), 2)} |\n"
            f"| Curtailable Capacity | {self._format_power(metrics.get('curtailable_capacity_kw', 0))} |\n"
            f"| Compliance Score | {self._fmt(metrics.get('compliance_score', 0), 1)}% |"
        )

    def _md_financial_performance(self, data: Dict[str, Any]) -> str:
        """Render financial performance section."""
        fin = data.get("financial_performance", {})
        return (
            "## 2. Financial Performance\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Gross Revenue | {self._format_currency(fin.get('gross_revenue', 0))} |\n"
            f"| Capacity Payments | {self._format_currency(fin.get('capacity_payments', 0))} |\n"
            f"| Energy Payments | {self._format_currency(fin.get('energy_payments', 0))} |\n"
            f"| Ancillary Payments | {self._format_currency(fin.get('ancillary_payments', 0))} |\n"
            f"| Performance Bonuses | {self._format_currency(fin.get('performance_bonuses', 0))} |\n"
            f"| Total Penalties | ({self._format_currency(fin.get('total_penalties', 0))}) |\n"
            f"| **Net Revenue** | **{self._format_currency(fin.get('net_revenue', 0))}** |\n"
            f"| Revenue per kW-yr | {self._format_currency(fin.get('revenue_per_kw_yr', 0))} |\n"
            f"| ROI on DR Investment | {self._fmt(fin.get('roi_pct', 0))}% |"
        )

    def _md_carbon_impact(self, data: Dict[str, Any]) -> str:
        """Render carbon impact summary section."""
        carbon = data.get("carbon_impact", {})
        return (
            "## 3. Carbon Impact\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| CO2e Avoided | {self._fmt(carbon.get('co2e_avoided_tonnes', 0))} tonnes |\n"
            f"| Energy Curtailed | {self._format_energy(carbon.get('energy_curtailed_mwh', 0))} |\n"
            f"| Avg Marginal EF | {self._fmt(carbon.get('avg_marginal_ef', 0), 4)} tCO2e/MWh |\n"
            f"| Carbon Value (shadow) | {self._format_currency(carbon.get('carbon_value', 0))} |\n"
            f"| SBTi Contribution | {self._fmt(carbon.get('sbti_contribution_pct', 0))}% |\n"
            f"| Equivalent Trees | {self._fmt(carbon.get('equivalent_trees', 0), 0)} |"
        )

    def _md_compliance_scorecard(self, data: Dict[str, Any]) -> str:
        """Render program compliance scorecard section."""
        programs = data.get("compliance_scorecard", [])
        if not programs:
            return "## 4. Program Compliance Scorecard\n\n_No compliance data available._"
        lines = [
            "## 4. Program Compliance Scorecard\n",
            "| Program | Events | Compliant | Score (%) | Status |",
            "|---------|-------:|----------:|----------:|--------|",
        ]
        for p in programs:
            lines.append(
                f"| {p.get('program_name', '-')} "
                f"| {p.get('total_events', 0)} "
                f"| {p.get('compliant_events', 0)} "
                f"| {self._fmt(p.get('score_pct', 0))}% "
                f"| {p.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_strategic_recommendations(self, data: Dict[str, Any]) -> str:
        """Render strategic recommendations section."""
        recs = data.get("strategic_recommendations", [])
        if not recs:
            recs = [
                {"recommendation": "Expand curtailable capacity by 20%", "priority": "High",
                 "impact": "Additional EUR 50K revenue"},
                {"recommendation": "Enroll in ancillary services program", "priority": "High",
                 "impact": "New revenue stream"},
                {"recommendation": "Deploy additional battery storage", "priority": "Medium",
                 "impact": "Improved performance ratio"},
            ]
        lines = [
            "## 5. Strategic Recommendations\n",
            "| # | Recommendation | Priority | Impact |",
            "|---|---------------|----------|--------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('recommendation', '-')} "
                f"| {r.get('priority', '-')} "
                f"| {r.get('impact', '-')} |"
            )
        return "\n".join(lines)

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render year-over-year comparison section."""
        yoy = data.get("yoy_comparison", {})
        metrics = yoy.get("metrics", [])
        if not metrics:
            return "## 6. Year-over-Year Comparison\n\n_No historical data available._"
        lines = [
            "## 6. Year-over-Year Comparison\n",
            f"**Current Period:** {yoy.get('current_period', '-')}  ",
            f"**Previous Period:** {yoy.get('previous_period', '-')}\n",
            "| Metric | Previous | Current | Change (%) |",
            "|--------|--------:|--------:|----------:|",
        ]
        for m in metrics:
            prev = m.get("previous", 0)
            curr = m.get("current", 0)
            change = ((curr - prev) / prev * 100) if prev != 0 else 0
            lines.append(
                f"| {m.get('metric', '-')} "
                f"| {self._fmt(prev)} "
                f"| {self._fmt(curr)} "
                f"| {self._fmt(change)}% |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-037 Demand Response Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        org = data.get("organization_name", "Organization")
        return (
            f'<h1>Demand Response Executive Summary</h1>\n'
            f'<p class="subtitle">Organization: {org} | '
            f'Period: {data.get("reporting_period", "-")} | '
            f'Prepared for: {data.get("prepared_for", "Executive Leadership")}</p>'
        )

    def _html_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML key metrics cards."""
        m = data.get("key_metrics", {})
        return (
            '<h2>Key Metrics</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Net Revenue</span>'
            f'<span class="value">{self._format_currency(m.get("total_net_revenue", 0))}</span></div>\n'
            f'  <div class="card card-green"><span class="label">CO2e Avoided</span>'
            f'<span class="value">{self._fmt(m.get("co2e_avoided_tonnes", 0))} t</span></div>\n'
            f'  <div class="card"><span class="label">Events</span>'
            f'<span class="value">{m.get("events_participated", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Performance</span>'
            f'<span class="value">{self._fmt(m.get("avg_performance_ratio", 0), 2)}</span></div>\n'
            f'  <div class="card"><span class="label">Capacity</span>'
            f'<span class="value">{self._fmt(m.get("curtailable_capacity_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Compliance</span>'
            f'<span class="value">{self._fmt(m.get("compliance_score", 0), 1)}%</span></div>\n'
            '</div>'
        )

    def _html_financial_performance(self, data: Dict[str, Any]) -> str:
        """Render HTML financial performance."""
        fin = data.get("financial_performance", {})
        return (
            '<h2>Financial Performance</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Gross Revenue</span>'
            f'<span class="value">{self._format_currency(fin.get("gross_revenue", 0))}</span></div>\n'
            f'  <div class="card card-red"><span class="label">Penalties</span>'
            f'<span class="value">({self._format_currency(fin.get("total_penalties", 0))})</span></div>\n'
            f'  <div class="card"><span class="label">Net Revenue</span>'
            f'<span class="value">{self._format_currency(fin.get("net_revenue", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">ROI</span>'
            f'<span class="value">{self._fmt(fin.get("roi_pct", 0))}%</span></div>\n'
            '</div>'
        )

    def _html_carbon_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML carbon impact."""
        carbon = data.get("carbon_impact", {})
        return (
            '<h2>Carbon Impact</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card card-green"><span class="label">CO2e Avoided</span>'
            f'<span class="value">{self._fmt(carbon.get("co2e_avoided_tonnes", 0))} t</span></div>\n'
            f'  <div class="card card-green"><span class="label">Energy Curtailed</span>'
            f'<span class="value">{self._format_energy(carbon.get("energy_curtailed_mwh", 0))}</span></div>\n'
            f'  <div class="card card-green"><span class="label">Carbon Value</span>'
            f'<span class="value">{self._format_currency(carbon.get("carbon_value", 0))}</span></div>\n'
            '</div>'
        )

    def _html_compliance_scorecard(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance scorecard table."""
        programs = data.get("compliance_scorecard", [])
        rows = ""
        for p in programs:
            score = p.get("score_pct", 0)
            cls = "status-pass" if score >= 90 else ("status-warn" if score >= 70 else "status-fail")
            rows += (
                f'<tr><td>{p.get("program_name", "-")}</td>'
                f'<td>{p.get("total_events", 0)}</td>'
                f'<td>{p.get("compliant_events", 0)}</td>'
                f'<td class="{cls}">{self._fmt(score)}%</td></tr>\n'
            )
        return (
            '<h2>Compliance Scorecard</h2>\n'
            '<table>\n<tr><th>Program</th><th>Events</th>'
            f'<th>Compliant</th><th>Score</th></tr>\n{rows}</table>'
        )

    def _html_strategic_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML strategic recommendations."""
        recs = data.get("strategic_recommendations", [])
        items = "".join(
            f'<li><strong>[{r.get("priority", "-")}]</strong> {r.get("recommendation", "-")} '
            f'- {r.get("impact", "-")}</li>\n'
            for r in recs
        )
        return f'<h2>Strategic Recommendations</h2>\n<ol>\n{items}</ol>'

    def _html_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML year-over-year comparison."""
        metrics = data.get("yoy_comparison", {}).get("metrics", [])
        rows = ""
        for m in metrics:
            prev = m.get("previous", 0)
            curr = m.get("current", 0)
            change = ((curr - prev) / prev * 100) if prev != 0 else 0
            cls = "variance-positive" if change >= 0 else "variance-negative"
            rows += (
                f'<tr><td>{m.get("metric", "-")}</td>'
                f'<td>{self._fmt(prev)}</td>'
                f'<td>{self._fmt(curr)}</td>'
                f'<td class="{cls}">{self._fmt(change)}%</td></tr>\n'
            )
        return (
            '<h2>Year-over-Year Comparison</h2>\n'
            '<table>\n<tr><th>Metric</th><th>Previous</th>'
            f'<th>Current</th><th>Change</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON key metrics."""
        m = data.get("key_metrics", {})
        return {
            "total_net_revenue": m.get("total_net_revenue", 0),
            "co2e_avoided_tonnes": m.get("co2e_avoided_tonnes", 0),
            "events_participated": m.get("events_participated", 0),
            "avg_performance_ratio": m.get("avg_performance_ratio", 0),
            "curtailable_capacity_kw": m.get("curtailable_capacity_kw", 0),
            "compliance_score": m.get("compliance_score", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        scorecard = data.get("compliance_scorecard", [])
        fin = data.get("financial_performance", {})
        yoy = data.get("yoy_comparison", {}).get("metrics", [])
        return {
            "compliance_bar": {
                "type": "bar",
                "labels": [p.get("program_name", "") for p in scorecard],
                "values": [p.get("score_pct", 0) for p in scorecard],
            },
            "revenue_waterfall": {
                "type": "waterfall",
                "items": [
                    {"label": "Capacity", "value": fin.get("capacity_payments", 0)},
                    {"label": "Energy", "value": fin.get("energy_payments", 0)},
                    {"label": "Ancillary", "value": fin.get("ancillary_payments", 0)},
                    {"label": "Bonuses", "value": fin.get("performance_bonuses", 0)},
                    {"label": "Penalties", "value": -fin.get("total_penalties", 0)},
                    {"label": "Net Revenue", "value": fin.get("net_revenue", 0)},
                ],
            },
            "yoy_bar": {
                "type": "grouped_bar",
                "labels": [m.get("metric", "") for m in yoy],
                "series": {
                    "previous": [m.get("previous", 0) for m in yoy],
                    "current": [m.get("current", 0) for m in yoy],
                },
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
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".card-green{background:#d1e7dd;}"
            ".card-red{background:#f8d7da;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".status-pass{color:#198754;font-weight:700;}"
            ".status-warn{color:#fd7e14;font-weight:600;}"
            ".status-fail{color:#dc3545;font-weight:700;}"
            ".variance-positive{color:#198754;font-weight:600;}"
            ".variance-negative{color:#dc3545;font-weight:600;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string.
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _format_power(self, val: Any) -> str:
        """Format a power value with units.

        Args:
            val: Power value in kW.

        Returns:
            Formatted power string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.1f} kW"
        return str(val)

    def _format_energy(self, val: Any) -> str:
        """Format an energy value with units.

        Args:
            val: Energy value in MWh.

        Returns:
            Formatted energy string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.2f} MWh"
        return str(val)

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

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
