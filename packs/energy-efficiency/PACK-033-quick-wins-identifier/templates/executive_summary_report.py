# -*- coding: utf-8 -*-
"""
ExecutiveSummaryReportTemplate - C-suite summary for PACK-033.

Generates executive-level summary reports for quick-win energy efficiency
programs, featuring key financial and environmental KPIs, strategic
recommendations, risk summaries, and top quick-win highlights suitable
for board-level presentation.

Sections:
    1. Key Metrics (4-5 KPIs)
    2. Financial Impact
    3. Environmental Impact
    4. Strategic Recommendations
    5. Risk Summary
    6. Quick Win Highlights (top 5)

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExecutiveSummaryReportTemplate:
    """
    Executive summary report template for C-suite audiences.

    Renders concise, high-impact summaries with key financial and
    environmental metrics, strategic recommendations, and top
    quick-win highlights across markdown, HTML, and JSON formats.

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
        """Render executive summary report as Markdown.

        Args:
            data: Executive summary engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_key_metrics(data),
            self._md_financial_impact(data),
            self._md_environmental_impact(data),
            self._md_strategic_recommendations(data),
            self._md_risk_summary(data),
            self._md_quick_win_highlights(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive summary report as self-contained HTML.

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
            self._html_financial_impact(data),
            self._html_environmental_impact(data),
            self._html_strategic_recommendations(data),
            self._html_risk_summary(data),
            self._html_quick_win_highlights(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Executive Summary Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render executive summary report as structured JSON.

        Args:
            data: Executive summary engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "executive_summary_report",
            "version": "33.0.0",
            "generated_at": self.generated_at.isoformat(),
            "key_metrics": self._json_key_metrics(data),
            "financial_impact": data.get("financial_impact", {}),
            "environmental_impact": data.get("environmental_impact", {}),
            "strategic_recommendations": data.get("strategic_recommendations", []),
            "risk_summary": data.get("risk_summary", []),
            "quick_win_highlights": data.get("quick_win_highlights", [])[:5],
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
        facility = data.get("facility_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Executive Summary: Quick Wins Energy Efficiency Program\n\n"
            f"**Organization:** {facility}  \n"
            f"**Prepared For:** {data.get('prepared_for', 'Executive Leadership')}  \n"
            f"**Date:** {data.get('report_date', '')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-033 ExecutiveSummaryReportTemplate v33.0.0\n\n---"
        )

    def _md_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render key metrics section with 4-5 headline KPIs."""
        metrics = data.get("key_metrics", {})
        return (
            "## 1. Key Metrics\n\n"
            "| KPI | Value |\n|-----|-------|\n"
            f"| Quick Wins Identified | {metrics.get('total_quick_wins', 0)} |\n"
            f"| Total Annual Savings | {self._format_currency(metrics.get('total_annual_savings', 0))} /yr |\n"
            f"| Total Investment Required | {self._format_currency(metrics.get('total_investment', 0))} |\n"
            f"| Average Payback Period | {self._fmt(metrics.get('avg_payback_months', 0), 1)} months |\n"
            f"| CO2e Reduction | {self._fmt(metrics.get('co2e_reduction_tonnes', 0))} tonnes/yr |"
        )

    def _md_financial_impact(self, data: Dict[str, Any]) -> str:
        """Render financial impact section."""
        fin = data.get("financial_impact", {})
        return (
            "## 2. Financial Impact\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Gross Annual Savings | {self._format_currency(fin.get('gross_annual_savings', 0))} |\n"
            f"| Implementation Cost | {self._format_currency(fin.get('implementation_cost', 0))} |\n"
            f"| Available Rebates | {self._format_currency(fin.get('available_rebates', 0))} |\n"
            f"| Net Investment | {self._format_currency(fin.get('net_investment', 0))} |\n"
            f"| 5-Year NPV | {self._format_currency(fin.get('npv_5yr', 0))} |\n"
            f"| Portfolio IRR | {self._fmt(fin.get('irr_pct', 0))}% |\n"
            f"| Benefit-Cost Ratio | {self._fmt(fin.get('benefit_cost_ratio', 0), 2)} |"
        )

    def _md_environmental_impact(self, data: Dict[str, Any]) -> str:
        """Render environmental impact section."""
        env = data.get("environmental_impact", {})
        return (
            "## 3. Environmental Impact\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Annual Energy Reduction | {self._format_energy(env.get('energy_reduction_mwh', 0))} |\n"
            f"| CO2e Reduction | {self._fmt(env.get('co2e_reduction_tonnes', 0))} tonnes/yr |\n"
            f"| Scope 1 Reduction | {self._fmt(env.get('scope1_reduction_pct', 0))}% |\n"
            f"| Scope 2 Reduction | {self._fmt(env.get('scope2_reduction_pct', 0))}% |\n"
            f"| Equivalent Trees Planted | {self._fmt(env.get('equivalent_trees', 0), 0)} |\n"
            f"| SBTi Contribution | {env.get('sbti_contribution', 'Not assessed')} |"
        )

    def _md_strategic_recommendations(self, data: Dict[str, Any]) -> str:
        """Render strategic recommendations section."""
        recs = data.get("strategic_recommendations", [])
        if not recs:
            recs = [
                {"recommendation": "Approve implementation budget", "priority": "High"},
                {"recommendation": "Assign dedicated implementation team", "priority": "High"},
                {"recommendation": "Engage utility for rebate programs", "priority": "Medium"},
            ]
        lines = [
            "## 4. Strategic Recommendations\n",
            "| # | Recommendation | Priority |",
            "|---|---------------|----------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('recommendation', '-')} "
                f"| {r.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_risk_summary(self, data: Dict[str, Any]) -> str:
        """Render risk summary section."""
        risks = data.get("risk_summary", [])
        if not risks:
            return "## 5. Risk Summary\n\n_No significant risks identified._"
        lines = [
            "## 5. Risk Summary\n",
            "| Risk | Severity | Likelihood | Mitigation |",
            "|------|----------|-----------|------------|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('risk', '-')} "
                f"| {r.get('severity', '-')} "
                f"| {r.get('likelihood', '-')} "
                f"| {r.get('mitigation', '-')} |"
            )
        return "\n".join(lines)

    def _md_quick_win_highlights(self, data: Dict[str, Any]) -> str:
        """Render top 5 quick win highlights section."""
        highlights = data.get("quick_win_highlights", [])[:5]
        if not highlights:
            return "## 6. Quick Win Highlights (Top 5)\n\n_No highlights available._"
        lines = [
            "## 6. Quick Win Highlights (Top 5)\n",
            "| # | Quick Win | Annual Savings | Payback | Impact |",
            "|---|-----------|---------------|---------|--------|",
        ]
        for i, h in enumerate(highlights, 1):
            lines.append(
                f"| {i} | {h.get('name', '-')} "
                f"| {self._format_currency(h.get('annual_savings', 0))} "
                f"| {self._fmt(h.get('payback_months', 0), 1)} mo "
                f"| {h.get('impact', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-033 Quick Wins Identifier Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Organization")
        return (
            f'<h1>Executive Summary: Quick Wins Energy Efficiency</h1>\n'
            f'<p class="subtitle">Organization: {facility} | '
            f'Prepared for: {data.get("prepared_for", "Executive Leadership")}</p>'
        )

    def _html_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML key metrics cards."""
        m = data.get("key_metrics", {})
        return (
            '<h2>Key Metrics</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Quick Wins</span>'
            f'<span class="value">{m.get("total_quick_wins", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Annual Savings</span>'
            f'<span class="value">{self._format_currency(m.get("total_annual_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Investment</span>'
            f'<span class="value">{self._format_currency(m.get("total_investment", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Payback</span>'
            f'<span class="value">{self._fmt(m.get("avg_payback_months", 0), 1)} mo</span></div>\n'
            f'  <div class="card"><span class="label">CO2e Reduction</span>'
            f'<span class="value">{self._fmt(m.get("co2e_reduction_tonnes", 0))} t/yr</span></div>\n'
            '</div>'
        )

    def _html_financial_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML financial impact."""
        fin = data.get("financial_impact", {})
        return (
            '<h2>Financial Impact</h2>\n'
            '<div class="impact-grid">\n'
            f'<div class="impact-item"><strong>5-Year NPV:</strong> '
            f'{self._format_currency(fin.get("npv_5yr", 0))}</div>\n'
            f'<div class="impact-item"><strong>IRR:</strong> '
            f'{self._fmt(fin.get("irr_pct", 0))}%</div>\n'
            f'<div class="impact-item"><strong>Net Investment:</strong> '
            f'{self._format_currency(fin.get("net_investment", 0))}</div>\n'
            f'<div class="impact-item"><strong>Benefit-Cost Ratio:</strong> '
            f'{self._fmt(fin.get("benefit_cost_ratio", 0), 2)}</div>\n'
            '</div>'
        )

    def _html_environmental_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML environmental impact."""
        env = data.get("environmental_impact", {})
        return (
            '<h2>Environmental Impact</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card card-green"><span class="label">Energy Saved</span>'
            f'<span class="value">{self._format_energy(env.get("energy_reduction_mwh", 0))}</span></div>\n'
            f'  <div class="card card-green"><span class="label">CO2e Avoided</span>'
            f'<span class="value">{self._fmt(env.get("co2e_reduction_tonnes", 0))} t/yr</span></div>\n'
            f'  <div class="card card-green"><span class="label">Trees Equivalent</span>'
            f'<span class="value">{self._fmt(env.get("equivalent_trees", 0), 0)}</span></div>\n'
            '</div>'
        )

    def _html_strategic_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML strategic recommendations."""
        recs = data.get("strategic_recommendations", [])
        items = "".join(
            f'<li><strong>[{r.get("priority", "-")}]</strong> {r.get("recommendation", "-")}</li>\n'
            for r in recs
        )
        return f'<h2>Strategic Recommendations</h2>\n<ol>\n{items}</ol>'

    def _html_risk_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML risk summary."""
        risks = data.get("risk_summary", [])
        rows = ""
        for r in risks:
            sev = r.get("severity", "Low").lower()
            cls = "risk-high" if sev == "high" else ("risk-medium" if sev == "medium" else "risk-low")
            rows += (
                f'<tr><td>{r.get("risk", "-")}</td>'
                f'<td class="{cls}">{r.get("severity", "-")}</td>'
                f'<td>{r.get("mitigation", "-")}</td></tr>\n'
            )
        return (
            '<h2>Risk Summary</h2>\n'
            '<table>\n<tr><th>Risk</th><th>Severity</th>'
            f'<th>Mitigation</th></tr>\n{rows}</table>'
        )

    def _html_quick_win_highlights(self, data: Dict[str, Any]) -> str:
        """Render HTML quick win highlights."""
        highlights = data.get("quick_win_highlights", [])[:5]
        items = ""
        for i, h in enumerate(highlights, 1):
            items += (
                f'<div class="highlight"><strong>#{i} {h.get("name", "-")}</strong> | '
                f'Savings: {self._format_currency(h.get("annual_savings", 0))}/yr | '
                f'Payback: {self._fmt(h.get("payback_months", 0), 1)} mo | '
                f'Impact: {h.get("impact", "-")}</div>\n'
            )
        return f'<h2>Top 5 Quick Wins</h2>\n{items}'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON key metrics."""
        m = data.get("key_metrics", {})
        return {
            "total_quick_wins": m.get("total_quick_wins", 0),
            "total_annual_savings": m.get("total_annual_savings", 0),
            "total_investment": m.get("total_investment", 0),
            "avg_payback_months": m.get("avg_payback_months", 0),
            "co2e_reduction_tonnes": m.get("co2e_reduction_tonnes", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        highlights = data.get("quick_win_highlights", [])[:5]
        fin = data.get("financial_impact", {})
        env = data.get("environmental_impact", {})
        return {
            "top5_savings_bar": {
                "type": "bar",
                "labels": [h.get("name", "") for h in highlights],
                "values": [h.get("annual_savings", 0) for h in highlights],
            },
            "financial_waterfall": {
                "type": "waterfall",
                "items": [
                    {"label": "Gross Savings", "value": fin.get("gross_annual_savings", 0)},
                    {"label": "Implementation", "value": -fin.get("implementation_cost", 0)},
                    {"label": "Rebates", "value": fin.get("available_rebates", 0)},
                    {"label": "Net Impact", "value": fin.get("net_investment", 0)},
                ],
            },
            "scope_reduction_pie": {
                "type": "pie",
                "labels": ["Scope 1", "Scope 2"],
                "values": [
                    env.get("scope1_reduction_pct", 0),
                    env.get("scope2_reduction_pct", 0),
                ],
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
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".impact-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:15px 0;}"
            ".impact-item{background:#f8f9fa;padding:12px;border-radius:6px;}"
            ".highlight{background:#f0f9ff;border-left:4px solid #0d6efd;padding:10px 15px;margin:8px 0;}"
            ".risk-high{color:#dc3545;font-weight:700;}"
            ".risk-medium{color:#fd7e14;font-weight:600;}"
            ".risk-low{color:#198754;}"
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
