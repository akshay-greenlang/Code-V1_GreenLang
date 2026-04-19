# -*- coding: utf-8 -*-
"""
PaybackAnalysisReportTemplate - Financial analysis of quick wins for PACK-033.

Generates detailed financial analysis reports for quick-win energy efficiency
measures, including NPV, IRR, payback period, ROI, cash flow projections,
sensitivity analysis, and investment recommendations.

Sections:
    1. Portfolio Summary
    2. Measure-by-Measure Analysis
    3. Cash Flow Analysis
    4. Sensitivity Analysis
    5. Investment Recommendations

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PaybackAnalysisReportTemplate:
    """
    Financial payback analysis report template.

    Renders financial analysis of quick-win measures with NPV, IRR,
    payback periods, cash flow projections, and sensitivity analysis
    across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PaybackAnalysisReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render payback analysis report as Markdown.

        Args:
            data: Financial analysis engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_portfolio_summary(data),
            self._md_measure_analysis(data),
            self._md_cash_flow(data),
            self._md_sensitivity(data),
            self._md_investment_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render payback analysis report as self-contained HTML.

        Args:
            data: Financial analysis engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_portfolio_summary(data),
            self._html_measure_analysis(data),
            self._html_cash_flow(data),
            self._html_sensitivity(data),
            self._html_investment_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Payback Analysis Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render payback analysis report as structured JSON.

        Args:
            data: Financial analysis engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "payback_analysis_report",
            "version": "33.0.0",
            "generated_at": self.generated_at.isoformat(),
            "portfolio_summary": self._json_portfolio_summary(data),
            "measures": data.get("measures", []),
            "cash_flow": data.get("cash_flow", {}),
            "sensitivity": data.get("sensitivity", {}),
            "investment_recommendations": data.get("investment_recommendations", []),
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
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Payback Analysis Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Analysis Date:** {data.get('analysis_date', '')}  \n"
            f"**Discount Rate:** {self._fmt(data.get('discount_rate_pct', 8))}%  \n"
            f"**Analysis Period:** {data.get('analysis_period_years', 10)} years  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-033 PaybackAnalysisReportTemplate v33.0.0\n\n---"
        )

    def _md_portfolio_summary(self, data: Dict[str, Any]) -> str:
        """Render portfolio summary section."""
        summary = data.get("portfolio_summary", {})
        return (
            "## 1. Portfolio Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Measures | {summary.get('total_measures', 0)} |\n"
            f"| Total Investment | {self._format_currency(summary.get('total_investment', 0))} |\n"
            f"| Total Annual Savings | {self._format_currency(summary.get('total_annual_savings', 0))} /yr |\n"
            f"| Portfolio NPV | {self._format_currency(summary.get('portfolio_npv', 0))} |\n"
            f"| Portfolio IRR | {self._fmt(summary.get('portfolio_irr_pct', 0))}% |\n"
            f"| Average Simple Payback | {self._fmt(summary.get('avg_simple_payback_months', 0), 1)} months |\n"
            f"| Portfolio ROI | {self._fmt(summary.get('portfolio_roi_pct', 0))}% |\n"
            f"| Benefit-Cost Ratio | {self._fmt(summary.get('benefit_cost_ratio', 0), 2)} |"
        )

    def _md_measure_analysis(self, data: Dict[str, Any]) -> str:
        """Render measure-by-measure analysis table."""
        measures = data.get("measures", [])
        if not measures:
            return "## 2. Measure-by-Measure Analysis\n\n_No measures to analyze._"
        lines = [
            "## 2. Measure-by-Measure Analysis\n",
            "| # | Measure | Investment | Annual Savings | NPV | IRR (%) | Payback (mo) | ROI (%) |",
            "|---|---------|-----------|---------------|-----|---------|-------------|---------|",
        ]
        for i, m in enumerate(measures, 1):
            lines.append(
                f"| {i} | {m.get('name', '-')} "
                f"| {self._format_currency(m.get('investment', 0))} "
                f"| {self._format_currency(m.get('annual_savings', 0))} "
                f"| {self._format_currency(m.get('npv', 0))} "
                f"| {self._fmt(m.get('irr_pct', 0))} "
                f"| {self._fmt(m.get('payback_months', 0), 1)} "
                f"| {self._fmt(m.get('roi_pct', 0))} |"
            )
        return "\n".join(lines)

    def _md_cash_flow(self, data: Dict[str, Any]) -> str:
        """Render cash flow analysis section."""
        cash_flow = data.get("cash_flow", {})
        annual = cash_flow.get("annual_projections", [])
        if not annual:
            return "## 3. Cash Flow Analysis\n\n_No cash flow data available._"
        lines = [
            "## 3. Cash Flow Analysis\n",
            f"**Initial Investment:** {self._format_currency(cash_flow.get('initial_investment', 0))}  ",
            f"**Cumulative NPV:** {self._format_currency(cash_flow.get('cumulative_npv', 0))}  ",
            f"**Breakeven Year:** {cash_flow.get('breakeven_year', '-')}\n",
            "| Year | Investment | Savings | Net Cash Flow | Cumulative |",
            "|------|-----------|---------|---------------|------------|",
        ]
        for yr in annual:
            lines.append(
                f"| {yr.get('year', '-')} "
                f"| {self._format_currency(yr.get('investment', 0))} "
                f"| {self._format_currency(yr.get('savings', 0))} "
                f"| {self._format_currency(yr.get('net_cash_flow', 0))} "
                f"| {self._format_currency(yr.get('cumulative', 0))} |"
            )
        return "\n".join(lines)

    def _md_sensitivity(self, data: Dict[str, Any]) -> str:
        """Render sensitivity analysis section."""
        sensitivity = data.get("sensitivity", {})
        scenarios = sensitivity.get("scenarios", [])
        if not scenarios:
            return "## 4. Sensitivity Analysis\n\n_No sensitivity data available._"
        lines = [
            "## 4. Sensitivity Analysis\n",
            f"**Base Case NPV:** {self._format_currency(sensitivity.get('base_npv', 0))}  ",
            f"**Variables Tested:** {', '.join(sensitivity.get('variables', []))}\n",
            "| Scenario | Variable | Change | NPV Impact | New Payback (mo) |",
            "|----------|----------|--------|-----------|-----------------|",
        ]
        for s in scenarios:
            lines.append(
                f"| {s.get('scenario', '-')} "
                f"| {s.get('variable', '-')} "
                f"| {s.get('change', '-')} "
                f"| {self._format_currency(s.get('npv_impact', 0))} "
                f"| {self._fmt(s.get('new_payback_months', 0), 1)} |"
            )
        return "\n".join(lines)

    def _md_investment_recommendations(self, data: Dict[str, Any]) -> str:
        """Render investment recommendations section."""
        recs = data.get("investment_recommendations", [])
        if not recs:
            recs = [
                {"action": "Proceed with all measures", "rationale": "Portfolio NPV positive"},
            ]
        lines = ["## 5. Investment Recommendations\n"]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"{i}. **{r.get('action', '-')}** - {r.get('rationale', '-')}"
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
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Payback Analysis Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Discount Rate: {self._fmt(data.get("discount_rate_pct", 8))}%</p>'
        )

    def _html_portfolio_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML portfolio summary cards."""
        s = data.get("portfolio_summary", {})
        return (
            '<h2>Portfolio Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Investment</span>'
            f'<span class="value">{self._format_currency(s.get("total_investment", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Portfolio NPV</span>'
            f'<span class="value">{self._format_currency(s.get("portfolio_npv", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Portfolio IRR</span>'
            f'<span class="value">{self._fmt(s.get("portfolio_irr_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Avg Payback</span>'
            f'<span class="value">{self._fmt(s.get("avg_simple_payback_months", 0), 1)} mo</span></div>\n'
            '</div>'
        )

    def _html_measure_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML measure-by-measure analysis table."""
        measures = data.get("measures", [])
        rows = ""
        for m in measures:
            rows += (
                f'<tr><td>{m.get("name", "-")}</td>'
                f'<td>{self._format_currency(m.get("investment", 0))}</td>'
                f'<td>{self._format_currency(m.get("annual_savings", 0))}</td>'
                f'<td>{self._format_currency(m.get("npv", 0))}</td>'
                f'<td>{self._fmt(m.get("irr_pct", 0))}%</td>'
                f'<td>{self._fmt(m.get("payback_months", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Measure-by-Measure Analysis</h2>\n'
            '<table>\n<tr><th>Measure</th><th>Investment</th><th>Savings/yr</th>'
            f'<th>NPV</th><th>IRR</th><th>Payback</th></tr>\n{rows}</table>'
        )

    def _html_cash_flow(self, data: Dict[str, Any]) -> str:
        """Render HTML cash flow section."""
        cash_flow = data.get("cash_flow", {})
        return (
            '<h2>Cash Flow Analysis</h2>\n'
            f'<p>Initial Investment: {self._format_currency(cash_flow.get("initial_investment", 0))} | '
            f'Breakeven: Year {cash_flow.get("breakeven_year", "-")} | '
            f'Cumulative NPV: {self._format_currency(cash_flow.get("cumulative_npv", 0))}</p>'
        )

    def _html_sensitivity(self, data: Dict[str, Any]) -> str:
        """Render HTML sensitivity analysis."""
        scenarios = data.get("sensitivity", {}).get("scenarios", [])
        rows = ""
        for s in scenarios:
            rows += (
                f'<tr><td>{s.get("scenario", "-")}</td>'
                f'<td>{s.get("variable", "-")}</td>'
                f'<td>{s.get("change", "-")}</td>'
                f'<td>{self._format_currency(s.get("npv_impact", 0))}</td></tr>\n'
            )
        return (
            '<h2>Sensitivity Analysis</h2>\n'
            '<table>\n<tr><th>Scenario</th><th>Variable</th><th>Change</th>'
            f'<th>NPV Impact</th></tr>\n{rows}</table>'
        )

    def _html_investment_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML investment recommendations."""
        recs = data.get("investment_recommendations", [])
        items = "".join(
            f'<li><strong>{r.get("action", "-")}</strong> - {r.get("rationale", "-")}</li>\n'
            for r in recs
        )
        return f'<h2>Investment Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_portfolio_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON portfolio summary."""
        s = data.get("portfolio_summary", {})
        return {
            "total_measures": s.get("total_measures", 0),
            "total_investment": s.get("total_investment", 0),
            "total_annual_savings": s.get("total_annual_savings", 0),
            "portfolio_npv": s.get("portfolio_npv", 0),
            "portfolio_irr_pct": s.get("portfolio_irr_pct", 0),
            "avg_simple_payback_months": s.get("avg_simple_payback_months", 0),
            "portfolio_roi_pct": s.get("portfolio_roi_pct", 0),
            "benefit_cost_ratio": s.get("benefit_cost_ratio", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        measures = data.get("measures", [])
        annual = data.get("cash_flow", {}).get("annual_projections", [])
        return {
            "npv_bar": {
                "type": "bar",
                "labels": [m.get("name", "") for m in measures],
                "values": [m.get("npv", 0) for m in measures],
            },
            "payback_bar": {
                "type": "horizontal_bar",
                "labels": [m.get("name", "") for m in measures],
                "values": [m.get("payback_months", 0) for m in measures],
            },
            "cumulative_cash_flow": {
                "type": "line",
                "labels": [str(yr.get("year", "")) for yr in annual],
                "series": {
                    "cumulative": [yr.get("cumulative", 0) for yr in annual],
                    "net_cash_flow": [yr.get("net_cash_flow", 0) for yr in annual],
                },
            },
            "irr_vs_payback": {
                "type": "scatter",
                "points": [
                    {
                        "x": m.get("payback_months", 0),
                        "y": m.get("irr_pct", 0),
                        "label": m.get("name", ""),
                    }
                    for m in measures
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
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
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
